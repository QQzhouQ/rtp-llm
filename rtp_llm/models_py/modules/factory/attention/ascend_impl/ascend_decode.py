import torch
import torch_npu

from rtp_llm.models_py.modules.factory.attention.ascend_impl.ascend_attn_params import (
    AscendAttnParams,
    compute_ascend_attn_params,
)
from rtp_llm.models_py.modules.factory.attention.ascend_impl.ascend_kv_cache_write_op import AscendKVCacheWriteOp
from rtp_llm.models_py.modules.factory.attention.ascend_impl.ascend_rope_emb import AscendRotaryEmbeddingOp
from rtp_llm.models_py.modules.factory.attention.fmha_impl_base import FMHAImplBase
from rtp_llm.models_py.modules.factory.attention import common


class AscendDecodeImpl(FMHAImplBase):
    """Ascend MHA Decode using torch_npu._npu_paged_attention.

    Composes RoPE -> KVCacheWrite -> write_cache_store -> paged_attention.
    """

    def __init__(self, attn_configs, attn_inputs, parallelism_config):
        self.need_rope_kv_cache = attn_configs.need_rope_kv_cache
        self.attn_configs = attn_configs
        self.attn_inputs = attn_inputs
        self.fmha_params = None

        self.fmha_impl = AscendDecodeAttnOp(attn_configs, attn_inputs)
        self.rope_impl = self._create_rope_impl(attn_configs)
        self.kv_cache_write_op = AscendKVCacheWriteOp(
            num_kv_heads=attn_configs.kv_head_num,
            head_size=attn_configs.size_per_head,
            token_per_block=attn_inputs.kv_cache.seq_size_per_block,
        )

        self.params = AscendAttnParams()
        if self.rope_impl is not None:
            self.rope_impl.set_params(self.params)
        self.kv_cache_write_op.set_params(self.params)

        self.fmha_impl.prepare(attn_inputs)
        self.write_cache_store_impl = common.create_write_cache_store_impl(attn_inputs)

    def _create_rope_impl(self, attn_configs):
        from rtp_llm.ops import RopeStyle
        if attn_configs.rope_config.style == RopeStyle.No:
            return None
        return AscendRotaryEmbeddingOp(attn_configs)

    def _split_qkv(self, qkv):
        qkv = qkv.reshape(qkv.shape[0], -1)
        num_heads = self.attn_configs.head_num
        num_kv_heads = self.attn_configs.kv_head_num
        head_dim = self.attn_configs.size_per_head
        q, k, v = torch.split(qkv, [
            head_dim * num_heads,
            head_dim * num_kv_heads,
            head_dim * num_kv_heads,
        ], dim=-1)
        query = q.reshape(q.shape[0], num_heads, head_dim)
        key = k.reshape(k.shape[0], num_kv_heads, head_dim)
        value = v.reshape(v.shape[0], num_kv_heads, head_dim)
        return query, key, value

    def _update_rope_kv_write_params(self, device):
        positions, slot_mapping = compute_ascend_attn_params(self.attn_inputs)
        self.params.positions_d = positions.to(device, non_blocking=True)
        self.params.slot_mapping = slot_mapping.to(device, non_blocking=True)

    def prepare(self, attn_inputs):
        self.fmha_impl.prepare(attn_inputs)
        self.attn_inputs = attn_inputs
        # TODO: Ascend Is not called outside, will be called in graph mode

    def forward(self, qkv, kv_cache, layer_idx=0):
        if self.need_rope_kv_cache:
            self._update_rope_kv_write_params(qkv.device)

            if self.rope_impl is not None:
                query, key, value = self.rope_impl.forward(qkv)
            else:
                query, key, value = self._split_qkv(qkv)

            self.kv_cache_write_op.forward(key, value, kv_cache)
            q = query
        else:
            q = qkv.chunk(3, dim=-1)[0]

        self.fmha_impl.context_lens = self.attn_inputs.sequence_lengths + 1

        common.apply_write_cache_store(
            self.write_cache_store_impl, self.attn_inputs, kv_cache
        )
        return self.fmha_impl.forward(q, kv_cache)

    @staticmethod
    def support(attn_configs, attn_inputs):
        return not attn_inputs.is_prefill and \
               not attn_configs.use_mla and \
               attn_inputs.kv_cache is not None and \
               attn_inputs.kv_cache.separate_kv_cache


class AscendDecodeAttnOp:
    """Encapsulate NPU decode attention using npu_fused_infer_attention_score.

    Uses FIA with TND layout + block_table (same as vllm-ascend default path)
    instead of _npu_paged_attention which has known aicore errors on certain
    block_table configurations.
    """

    _causal_mask = None

    @classmethod
    def _get_causal_mask(cls, device):
        if cls._causal_mask is None or cls._causal_mask.device.type != device.type:
            cls._causal_mask = torch.triu(
                torch.ones(2048, 2048, dtype=torch.int8), diagonal=1
            ).to(device)
        return cls._causal_mask

    def __init__(self, attn_configs, attn_inputs):
        self.num_heads = attn_configs.head_num
        self.num_kv_heads = attn_configs.kv_head_num
        self.head_dim = attn_configs.size_per_head
        self.scale = attn_configs.q_scaling * self.head_dim ** -0.5
        self.page_size = attn_inputs.kv_cache.seq_size_per_block if \
                         attn_inputs.kv_cache else 128
        self.block_table = None
        self.context_lens = None

    def set_params(self, params):
        self.params = params

    def prepare(self, attn_inputs):
        self.block_table = attn_inputs.kv_cache_kernel_block_id_host
        if self.block_table is not None:
            self.block_table = self.block_table.clamp(min=0)
            if self.block_table.ndim != 2:
                self.block_table = self.block_table.reshape(-1, self.block_table.shape[-1])
        self.context_lens = attn_inputs.prefix_lengths + attn_inputs.input_lengths

    def forward(self, q, kv_cache):
        k_cache = kv_cache.k_cache_base.permute(0, 2, 1, 3).contiguous()
        v_cache = kv_cache.v_cache_base.permute(0, 2, 1, 3).contiguous()
        block_table = self.block_table
        if block_table is not None and block_table.device.type != q.device.type:
            block_table = block_table.to(q.device)
        context_lens = self.context_lens
        if context_lens is not None and context_lens.device.type != q.device.type:
            context_lens = context_lens.to(q.device)
        batch_size = q.shape[0]
        actual_seq_q = torch.arange(
            1, batch_size + 1, dtype=torch.int32, device=q.device
        )
        actual_seq_kv = torch.cumsum(context_lens.to(torch.int32), dim=0)
        if actual_seq_kv.device.type != q.device.type:
            actual_seq_kv = actual_seq_kv.to(q.device)
        atten_mask = self._get_causal_mask(q.device)
        attn_output, _ = torch_npu.npu_fused_infer_attention_score(
            query=q, key=k_cache, value=v_cache,
            atten_mask=atten_mask,
            block_table=block_table,
            input_layout="TND",
            block_size=self.page_size,
            actual_seq_lengths=actual_seq_q,
            actual_seq_lengths_kv=actual_seq_kv,
            num_key_value_heads=self.num_kv_heads,
            num_heads=self.num_heads,
            scale=self.scale,
            sparse_mode=3,
        )
        return attn_output