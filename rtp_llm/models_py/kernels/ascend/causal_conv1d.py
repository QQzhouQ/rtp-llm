"""Ascend causal convolution wrappers for Qwen3.5.

The Ascend implementation adapts RTP-LLM's paged convolution cache to the
layout expected by ``fla_npu``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Union

import torch

PAD_SLOT_ID = -1


@dataclass
class CausalConv1dMetadata:
    """Metadata compatible with the legacy Triton causal-convolution API.

    AscendC does not need the Triton launch metadata, so all three fields are
    empty for an NPU request.
    """

    batch_ptr: torch.Tensor
    token_chunk_offset_ptr: torch.Tensor
    total: int


def _activation_name(activation: Union[bool, str, None]) -> Union[str, None]:
    if activation is None or activation is False:
        return None
    if activation is True or activation in ("silu", "swish"):
        return "silu"
    raise ValueError("activation must be None, False, True, 'silu', or 'swish'")


def _load_npu_causal_conv1d():
    from fla_npu.ops.ascendc import npu_causal_conv1d

    return npu_causal_conv1d


def _load_npu_causal_conv1d_update():
    from fla_npu.ops.ascendc import npu_causal_conv1d_update

    return npu_causal_conv1d_update


def _causal_conv1d_update_device(
    x: torch.Tensor,
    conv_state: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
    activation: Union[bool, str, None],
    block_map: torch.Tensor,
    sequence_lengths: torch.Tensor,
    seq_size_per_block: int,
    original_dtype: torch.dtype,
    squeeze_token_axis: bool,
) -> torch.Tensor:
    """Single-token conv update with fully device-side metadata.

    Same AscendC operator family as the host-metadata path (single consumer):
    the write page per sequence is gathered from ``block_map`` on device and
    the cross-block conv-state seed uses block-level page-row copies, so the
    step is aclgraph-capturable.  ``conv_state`` is the raw paged pool view
    ``(pages, state_len, dim)``; the operator mutates it in place.
    """

    from rtp_llm.models_py.kernels.ascend.paged_state import (
        decode_state_indices,
        seed_state_segment,
    )

    npu_causal_conv1d_update = _load_npu_causal_conv1d_update()
    read_idx, write_idx = decode_state_indices(
        block_map, sequence_lengths, seq_size_per_block
    )
    seed_state_segment(conv_state, read_idx, write_idx)
    # the operator computes its result into the x argument; clone so the
    # caller's buffer (often a view of a larger transient) is not mutated
    out = npu_causal_conv1d_update(
        x=x.to(conv_state.dtype).clone(),
        conv_state=conv_state,
        weight=weight.t().contiguous(),
        bias=bias,
        activation=_activation_name(activation),
        conv_state_indices=write_idx.to(torch.int32),
        null_block_id=0,
    )
    out = out.to(original_dtype)
    # caller contract: (batch, dim) in, (batch, dim) out; otherwise
    # (batch, dim, 1)
    return out if squeeze_token_axis else out.unsqueeze(-1)


def _activation_mode(activation: Union[bool, str, None]) -> int:
    if activation is None or activation is False:
        return 0
    if activation is True or activation in ("silu", "swish"):
        return 1
    raise ValueError("activation must be None, False, True, 'silu', or 'swish'")


def _as_int_list(values) -> list[int]:
    if isinstance(values, (list, tuple)):
        return [int(value) for value in values]
    return [
        int(value) for value in values.detach().to(dtype=torch.int64).cpu().tolist()
    ]


def _as_int_rows(values) -> list[list[int]]:
    if isinstance(values, (list, tuple)):
        return [[int(value) for value in row] for row in values]
    return [
        [int(value) for value in row]
        for row in values.detach().to(dtype=torch.int64).cpu().tolist()
    ]


def _mapped_page(
    block_rows: list[list[int]],
    sequence_index: int,
    block_index: int,
    pad_slot_id: int,
) -> int:
    if block_index < 0 or sequence_index >= len(block_rows):
        return pad_slot_id
    row = block_rows[sequence_index]
    if block_index >= len(row):
        return pad_slot_id
    return int(row[block_index])


def prepare_causal_conv1d_metadata(
    query_start_loc: torch.Tensor,
    device: torch.device,
) -> CausalConv1dMetadata:
    """Return the no-op launch metadata expected by the shared model."""

    empty = torch.empty(0, dtype=torch.int32, device=device)
    return CausalConv1dMetadata(empty, empty, 0)


def _gather_prefill_states(
    x: torch.Tensor,
    conv_states: Optional[torch.Tensor],
    block_rows: Optional[list[list[int]]],
    prefix_values: list[int],
    seq_size_per_block: int,
    state_len: int,
    pad_slot_id: int,
) -> torch.Tensor:
    """Gather the state ending at each sequence prefix into NPU layout."""

    batch = len(prefix_values)
    dim = x.shape[0]
    initial_states = torch.zeros(
        (batch, state_len, dim), dtype=x.dtype, device=x.device
    )
    if conv_states is None or block_rows is None or state_len == 0:
        return initial_states

    for sequence_index, prefix_length in enumerate(prefix_values):
        if prefix_length <= 0:
            continue
        block_index = (prefix_length - 1) // seq_size_per_block
        page_index = _mapped_page(block_rows, sequence_index, block_index, pad_slot_id)
        if page_index == pad_slot_id:
            continue
        # RTP-LLM: (page, dim, state); AscendC: (page, state, dim).
        initial_states[sequence_index].copy_(
            conv_states[page_index, :, :state_len].transpose(0, 1)
        )
    return initial_states


def _history_ending_at(
    initial_state: torch.Tensor,
    sequence_x: torch.Tensor,
    end: int,
) -> torch.Tensor:
    """Return the fixed-width input history ending before ``end``."""

    state_len = initial_state.shape[0]
    if state_len == 0:
        return initial_state
    if end >= state_len:
        return sequence_x[end - state_len : end]
    return torch.cat((initial_state[end:], sequence_x[:end]), dim=0)


def _scatter_prefill_states(
    x: torch.Tensor,
    conv_states: Optional[torch.Tensor],
    block_rows: Optional[list[list[int]]],
    query_starts: list[int],
    prefix_values: list[int],
    seq_size_per_block: int,
    initial_states: torch.Tensor,
    state_len: int,
    pad_slot_id: int,
) -> None:
    """Write cache snapshots at every crossed block edge and sequence end."""

    if conv_states is None or block_rows is None or state_len == 0:
        return

    for sequence_index, prefix_length in enumerate(prefix_values):
        token_start = query_starts[sequence_index]
        token_end = query_starts[sequence_index + 1]
        sequence_x = x[:, token_start:token_end].transpose(0, 1)
        sequence_len = token_end - token_start

        for local_end in range(1, sequence_len + 1):
            absolute_end = prefix_length + local_end
            if absolute_end % seq_size_per_block != 0 and local_end != sequence_len:
                continue

            block_index = (absolute_end - 1) // seq_size_per_block
            page_index = _mapped_page(
                block_rows, sequence_index, block_index, pad_slot_id
            )
            if page_index == pad_slot_id:
                continue
            history = _history_ending_at(
                initial_states[sequence_index], sequence_x, local_end
            )
            conv_states[page_index, :, :state_len].copy_(history.transpose(0, 1))


def causal_conv1d_fn(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: Union[torch.Tensor, None],
    conv_states: Optional[torch.Tensor],
    query_start_loc: torch.Tensor,
    block_map: Optional[torch.Tensor],
    prefix_lengths: torch.Tensor,
    seq_size_per_block: int,
    activation: Optional[str] = "silu",
    pad_slot_id: int = PAD_SLOT_ID,
    metadata: Optional[CausalConv1dMetadata] = None,
    validate_data=False,
):
    """Run varlen causal convolution and update the paged cache in place."""

    if x.dim() != 2 or weight.dim() != 2:
        raise ValueError("NPU prefill expects x=(dim, tokens), weight=(dim, width)")
    if seq_size_per_block <= 0:
        raise ValueError("seq_size_per_block must be positive")

    original_dtype = x.dtype
    x_work = x.to(weight.dtype)
    dim, _ = x_work.shape
    weight_dim, width = weight.shape
    if dim != weight_dim:
        raise ValueError("x and weight feature dimensions must match")

    query_starts = _as_int_list(query_start_loc)
    prefix_values = _as_int_list(prefix_lengths)
    batch = len(query_starts) - 1
    if len(prefix_values) != batch:
        raise ValueError("prefix_lengths must contain one value per sequence")
    block_rows = _as_int_rows(block_map) if block_map is not None else None
    state_len = width - 1

    temporary_states = _gather_prefill_states(
        x_work,
        conv_states,
        block_rows,
        prefix_values,
        seq_size_per_block,
        state_len,
        pad_slot_id,
    )
    # The Ascend operator may update its state argument.  Cache scatter needs
    # the exact history that existed before this call.
    initial_states = temporary_states.clone()
    npu_causal_conv1d = _load_npu_causal_conv1d()
    output = npu_causal_conv1d(
        # x arrives as (dim, tokens); the operator reads a contiguous
        # (tokens, dim) buffer, so materialise the transpose instead of
        # handing it a strided view.
        x=x_work.transpose(0, 1).contiguous(),
        weight=weight.transpose(0, 1).contiguous(),
        bias=bias,
        conv_states=temporary_states,
        query_start_loc=query_starts,
        initial_state_mode=[int(prefix > 0) for prefix in prefix_values],
        activation_mode=_activation_mode(activation),
        pad_slot_id=pad_slot_id,
        run_mode=0,
        head_num=0,
    )

    _scatter_prefill_states(
        x_work,
        conv_states,
        block_rows,
        query_starts,
        prefix_values,
        seq_size_per_block,
        initial_states,
        state_len,
        pad_slot_id,
    )
    return output.transpose(0, 1).to(original_dtype)


def causal_conv1d_update(
    x: torch.Tensor,
    conv_state: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    activation: Union[bool, str, None] = None,
    cache_seqlens: Optional[torch.Tensor] = None,
    block_map: Optional[torch.Tensor] = None,
    seq_size_per_block: int = 1,
    sequence_lengths: Optional[torch.Tensor] = None,
    query_start_loc: Optional[torch.Tensor] = None,
    max_query_len: int = -1,
    pad_slot_id: int = PAD_SLOT_ID,
    metadata=None,
    validate_data=False,
):
    """Decode one or more tokens while preserving RTP-LLM's paged state.

    ``conv_state`` is the raw paged pool view ``(pages, state_len, dim)`` —
    the layout produced by ``typed_storage_view`` and consumed directly by the
    AscendC operator (note the different contract from ``causal_conv1d_fn``,
    which keeps the transposed ``(page, dim, state)`` view).

    Single-token decode runs the device-metadata path shared by eager and
    aclgraph capture; multi-token speculative decode keeps the host-metadata
    path (eager only — target-verify never runs inside the decode graph).
    """

    if seq_size_per_block <= 0:
        raise ValueError("seq_size_per_block must be positive")
    if block_map is None or sequence_lengths is None:
        raise ValueError("block_map and sequence_lengths are required on NPU")
    if cache_seqlens is not None or query_start_loc is not None:
        raise NotImplementedError(
            "Ascend paged decode does not support cache_seqlens or varlen "
            "query_start_loc"
        )

    original_dtype = x.dtype
    squeeze_token_axis = x.dim() == 2
    if squeeze_token_axis:
        x = x.unsqueeze(-1)
    if x.dim() != 3:
        raise ValueError("NPU decode expects x=(batch, dim, tokens)")

    batch, dim, token_count = x.shape
    if weight.dim() != 2 or weight.shape[0] != dim:
        raise ValueError("weight must have shape (dim, width)")

    # Standard decode (one token per sequence): device-metadata path.
    if (
        token_count == 1
        and block_map.ndim == 2
        and block_map.shape[0] == batch
    ):
        return _causal_conv1d_update_device(
            x[:, :, 0],
            conv_state,
            weight,
            bias,
            activation,
            block_map,
            sequence_lengths,
            int(seq_size_per_block),
            original_dtype,
            squeeze_token_axis,
        )

    x_work = x.to(conv_state.dtype)

    npu_causal_conv1d = _load_npu_causal_conv1d()
    npu_states = conv_state
    npu_weight = weight.transpose(0, 1).contiguous()
    output_tokens = []

    current_lengths = _as_int_list(sequence_lengths)
    if len(current_lengths) != batch:
        raise ValueError("sequence_lengths must contain one value per sequence")
    block_rows = _as_int_rows(block_map)

    read_pages = []
    write_block_starts = []
    for sequence_index, first_total_length in enumerate(current_lengths):
        if first_total_length <= 0:
            raise ValueError("decode sequence lengths must be positive")
        read_block = max(first_total_length - 2, 0) // seq_size_per_block
        read_page = _mapped_page(block_rows, sequence_index, read_block, pad_slot_id)
        read_pages.append(read_page)
        write_block_starts.append((first_total_length - 1) // seq_size_per_block)

    for token_index in range(token_count):
        cache_indices = []
        for sequence_index in range(batch):
            # Match the CUDA continuous-batching contract: every speculative
            # token is snapshotted into a consecutive block-map entry, even if
            # its logical position has not crossed a normal cache-block edge.
            target_block = write_block_starts[sequence_index] + token_index
            target_page = _mapped_page(
                block_rows, sequence_index, target_block, pad_slot_id
            )

            source_page = (
                read_pages[sequence_index]
                if token_index == 0
                else _mapped_page(
                    block_rows,
                    sequence_index,
                    target_block - 1,
                    pad_slot_id,
                )
            )
            if (
                source_page != pad_slot_id
                and target_page != pad_slot_id
                and source_page != target_page
            ):
                conv_state[target_page].copy_(conv_state[source_page])
            cache_indices.append(target_page)

        token_output = npu_causal_conv1d(
            x=x_work[:, :, token_index],
            weight=npu_weight,
            bias=bias,
            conv_states=npu_states,
            cache_indices=cache_indices,
            activation_mode=_activation_mode(activation),
            pad_slot_id=pad_slot_id,
            run_mode=1,
            head_num=0,
        )
        output_tokens.append(token_output)

    if output_tokens:
        output = torch.stack(output_tokens, dim=-1)
    else:
        output = torch.empty_like(x_work)
    if squeeze_token_axis:
        output = output.squeeze(-1)
    return output.to(original_dtype)
