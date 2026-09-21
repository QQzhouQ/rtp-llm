"""Device-side paged-state helpers shared by the GDN AscendC kernels.

Plan C (single-consumer path): eager decode and aclgraph capture run the same
fla_npu AscendC operators with device-tensor metadata.  These helpers replace
the former host-side metadata computation (``block_map.cpu().tolist()``) with
plain device ops so the whole decode step is capturable without any D2H
synchronisation:

* ``decode_state_indices`` gathers the read/write state pages from the block
  map — semantics identical to ``recurrent._resolve_state_pages``:
      read  page = block_map[b, max(len - 2, 0) // page]
      write page = block_map[b, (len - 1) // page]
  where ``len`` is ``sequence_lengths_plus_1``.
* ``seed_state_segment`` migrates the segment owned by ``seg_view`` (the ssm
  or conv slice of a page) from the read page to the write page when a block
  boundary is crossed.  The sibling segment of the destination page is left
  untouched — the other GDN operator owns it.

The pool views come from ``utils/typed_storage_view``: every page row packs
the ssm segment followed by the conv segment, so both segments are strided
views over one storage.

Perf constraints (measured on Ascend950PR / CANN 9.2.2, see the acl-graph
GDN performance notes):

* never ``index_select`` the 4D strided pool view directly — it degrades to
  element-wise copies (~3.8 ms); the 2D page-row view below keeps copies
  block-level (~0.04 ms);
* never assign into a mid-row slice (``row[:, off:off+len] = x``) — it lowers
  to aclnnInplaceCopy_Slice (element-wise again, ~5.9 ms); rebuild rows with
  ``torch.cat`` instead.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def decode_state_indices(
    block_map: torch.Tensor,
    sequence_lengths_plus_1: torch.Tensor,
    seq_size_per_block: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Gather (read_page, write_page) per sequence row from ``block_map``."""

    length = sequence_lengths_plus_1.reshape(-1).to(torch.int64)
    read_col = (length - 2).clamp_min(0) // seq_size_per_block
    write_col = (length - 1).clamp_min(0) // seq_size_per_block
    read_idx = block_map.gather(1, read_col.view(-1, 1)).squeeze(1)
    write_idx = block_map.gather(1, write_col.view(-1, 1)).squeeze(1)
    return read_idx, write_idx


def paged_row_view(seg_view: torch.Tensor) -> tuple[torch.Tensor, int, int]:
    """Derive a 2D page-row view ``[pages, page_stride]`` from a segment view.

    Returns ``(row_view, seg_offset, seg_len)``: page ``p``'s segment lives at
    ``row_view[p, seg_offset : seg_offset + seg_len]``.  Host-side view
    construction only — safe inside capture.
    """

    page_stride = seg_view.stride(0)
    off = seg_view.storage_offset()
    seg_offset = off % page_stride
    base = off - seg_offset
    seg_len = 1
    for s in seg_view.shape[1:]:
        seg_len *= s
    row_view = torch.empty(0, dtype=seg_view.dtype, device=seg_view.device)
    row_view.set_(seg_view.untyped_storage(), base,
                  (seg_view.shape[0], page_stride), (page_stride, 1))
    return row_view, seg_offset, seg_len


def seed_state_segment(
    seg_view: torch.Tensor,
    read_idx: torch.Tensor,
    write_idx: torch.Tensor,
) -> None:
    """Migrate ``seg_view``'s segment across a block boundary, in place.

    When ``read_idx != write_idx`` for a row the segment is copied from the
    read page into the write page (the AscendC operators read and update the
    state in place at the write page); equal indices keep the row unchanged.
    Only the segment columns of the destination row are rewritten — the
    sibling segment (ssm vs conv) is preserved.

    Perf notes (measured, Ascend950PR): the full-row gather/where/cat/scatter
    costs ~0.1 ms per GDN layer and is NOT a decode-step hotspot (profiling
    attributed the big Slice kernels to the FIA KV-cache write path instead).
    A narrower segment-column view variant was tried and is 40-60% SLOWER:
    F.embedding/index_copy_ on the non-contiguous column slice degrades to
    element-wise copies, so keep the contiguous full-row view.
    """

    row_view, seg_off, seg_len = paged_row_view(seg_view)
    r64 = read_idx.long()
    w64 = write_idx.long()
    row_w = F.embedding(w64, row_view)
    row_r = F.embedding(r64, row_view)
    same = (r64 == w64).view(-1, 1)
    seg = torch.where(same,
                      row_w[:, seg_off:seg_off + seg_len],
                      row_r[:, seg_off:seg_off + seg_len])
    row = torch.cat([row_w[:, :seg_off], seg, row_w[:, seg_off + seg_len:]], dim=1)
    row_view.index_copy_(0, w64, row)


__all__ = ["decode_state_indices", "paged_row_view", "seed_state_segment"]
