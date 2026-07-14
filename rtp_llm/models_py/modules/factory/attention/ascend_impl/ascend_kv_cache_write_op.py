import torch
import torch_npu

from rtp_llm.ops.compute_ops import LayerKVCache


class AscendKVCacheWriteOp:
    """MHA KV Cache write using torch_npu.npu_scatter_pa_kv_cache.

    The combined KV buffer is [blocks, 2, seq, heads, dim] (BSND) after the
    C++ getLayerCache reshape.  kv_cache_base[:, 0/1] yields [blocks, seq,
    heads, dim] directly — no Python permute needed.  npu_scatter_pa_kv_cache
    supports non-contiguous inputs, so we scatter directly into the strided
    views without cloning.
    """

    def __init__(self, num_kv_heads, head_size, token_per_block):
        self.num_kv_heads = num_kv_heads
        self.head_size = head_size
        self.token_per_block = token_per_block
        self.params = None

    def set_params(self, params):
        self.params = params

    def forward(self, key, value, kv_cache):
        if kv_cache is None:
            return

        kv_base = kv_cache.kv_cache_base
        # Already BSND [blocks, seq, heads, dim] from C++ reshape — no permute
        k_view = kv_base[:, 0]
        v_view = kv_base[:, 1]

        slot_mapping = self.params.slot_mapping
        if slot_mapping.dtype not in (torch.int32, torch.int64):
            slot_mapping = slot_mapping.to(torch.int32)

        torch_npu.npu_scatter_pa_kv_cache(
            key, value, k_view, v_view, slot_mapping,
        )

    def _prepare_warmup_cache_indices(self, num_tokens, device):
        import torch
        batch_indices = torch.zeros(num_tokens, dtype=torch.int32, device=device)
        positions = torch.arange(num_tokens, dtype=torch.int32, device=device)
        max_num_pages = (num_tokens + self.token_per_block - 1) // self.token_per_block
        kv_page_indices = positions // self.token_per_block
        kv_page_indptr = torch.tensor([0, max_num_pages], dtype=torch.int32, device=device)
        last_page_len = num_tokens % self.token_per_block
        if last_page_len == 0:
            last_page_len = self.token_per_block
        kv_last_page_len = torch.tensor([last_page_len], dtype=torch.int32, device=device)
        return batch_indices, positions, kv_page_indices, kv_page_indptr, kv_last_page_len, max_num_pages
