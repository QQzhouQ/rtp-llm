import torch
import torch_npu

from rtp_llm.ops.compute_ops import LayerKVCache


class AscendKVCacheWriteOp:
    """MHA KV Cache write using flat indexing on the merged KV buffer.

    Avoids cloning the entire KV cache (which causes OOM on large caches).
    Uses kv_base.reshape(-1, nkv, dim) + flat index assignment instead of
    npu_scatter_pa_kv_cache (which requires contiguous inputs → clone).
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

        slot_mapping = self.params.slot_mapping
        if slot_mapping.dtype not in (torch.int32, torch.int64):
            slot_mapping = slot_mapping.to(torch.int32)

        slot_mapping_long = slot_mapping.long()
        block_ids = slot_mapping_long // self.token_per_block
        offsets = slot_mapping_long % self.token_per_block

        page = self.token_per_block
        kv_flat = kv_base.reshape(-1, self.num_kv_heads, self.head_size)
        flat_idx_k = block_ids * (2 * page) + offsets
        flat_idx_v = block_ids * (2 * page) + page + offsets
        kv_flat[flat_idx_k] = key
        kv_flat[flat_idx_v] = value
