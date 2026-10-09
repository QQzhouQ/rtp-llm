import torch
import torch_npu

from rtp_llm.models_py.modules.factory.attention.ascend_impl.ascend_attn_params import (
    AscendAttnParams,
    _blocks_per_phys_from_config,
    compute_ascend_attn_params,
)

from rtp_llm.models_py.modules.factory.attention.ascend_impl.ascend_kv_cache_write_op import AscendKVCacheWriteOp
from rtp_llm.models_py.modules.factory.attention.ascend_impl.ascend_rope_emb import AscendRotaryEmbeddingOp
from rtp_llm.models_py.modules.factory.attention.fmha_impl_base import FMHAImplBase
from rtp_llm.models_py.modules.factory.attention import common


class AscendPrefillImpl(FMHAImplBase):
    """Ascend MHA Prefill using npu_fused_infer_attention_score.

    Composes RoPE -> KVCacheWrite -> write_cache_store -> FMHA.
    """

    def __init__(self, attn_configs, attn_inputs, parallelism_config):
        self.need_rope_kv_cache = attn_configs.need_rope_kv_cache
        self.attn_configs = attn_configs
        self.attn_inputs = attn_inputs
        self.fmha_params = None

        self.fmha_impl = AscendPrefillAttnOp(attn_configs, attn_inputs)
        self.rope_impl = self._create_rope_impl(attn_configs)
        self.kv_cache_write_op = AscendKVCacheWriteOp(
            num_kv_heads=attn_configs.kv_head_num,
            head_size=attn_configs.size_per_head,
            token_per_block=attn_inputs.kv_cache.seq_size_per_block if attn_inputs.kv_cache else 128,
        )

        self.params = AscendAttnParams() # Only used by rope and KV cache write
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
        key = k.reshape(k.shape[0], num_kv_heads, head_dim).contiguous()
        value = v.reshape(v.shape[0], num_kv_heads, head_dim).contiguous()
        return query, key, value

    def _update_rope_kv_write_params(self, device, kv_cache, num_tokens: int, layer_idx: int = 0):
        # The kernel block granularity comes from the per-layer cache view; the
        # physical block size it maps onto drives slot_mapping and the writes.
        blocks_per_phys = _blocks_per_phys_from_config(self.attn_configs, self.attn_inputs)
        kernel_page = kv_cache.seq_size_per_block if kv_cache is not None else 0
        self.params.blocks_per_phys = blocks_per_phys
        positions, slot_mapping, mrope_positions = compute_ascend_attn_params(
            self.attn_inputs, layer_idx, kernel_page * blocks_per_phys
        )
        self.params.positions_d = positions.to(device, non_blocking=True)
        self.params.slot_mapping = slot_mapping.to(device, non_blocking=True)
        self.params.mrope_positions_d = (
            mrope_positions.to(device, non_blocking=True) if mrope_positions is not None else None
        )

    def _update_rope_kv_write_params_device(self, device, kernel_page, num_tokens: int):
        attn_inputs = self.attn_inputs
        if not kernel_page:
            kernel_page = (
                attn_inputs.kv_cache.seq_size_per_block
                if attn_inputs.kv_cache is not None
                else 128
            )
        # Verify feeds a uniform num_tokens_per_bs (= K+1) per stream and the
        # TND layout is stream-major, so token t belongs to stream t // apsk.
        seq_plus_1 = attn_inputs.sequence_lengths_plus_1_d  # [B] == prefix + 1
        num_streams = int(seq_plus_1.numel())
        tokens_per_stream = num_tokens // num_streams if num_streams > 0 else 0

        arange = torch.arange(tokens_per_stream, dtype=torch.int32, device=device)
        positions_d = (
            (seq_plus_1.to(torch.int32) - 1).unsqueeze(1) + arange.unsqueeze(0)
        ).reshape(-1)

        block_table = attn_inputs.kv_cache_kernel_block_id_device
        if (
            block_table is not None
            and block_table.numel() > 0
            and positions_d.numel() > 0
        ):
            if block_table.ndim != 2:
                block_table = block_table.reshape(-1, block_table.shape[-1])
            max_blocks = block_table.size(1)
            pos_long = positions_d.long()
            block_index = (pos_long // kernel_page).clamp(max=max_blocks - 1)
            block_offset = pos_long % kernel_page
            batch_ids = (
                torch.arange(num_streams, dtype=torch.long, device=device)
                .unsqueeze(1)
                .expand(num_streams, tokens_per_stream)
                .reshape(-1)
            )
            slot_block_numbers = block_table[batch_ids, block_index].long().clamp(min=0)
            slot_mapping = (slot_block_numbers * kernel_page + block_offset).to(torch.int64)
        else:
            slot_mapping = torch.empty(0, dtype=torch.int64, device=device)

        self.params.positions_d = positions_d
        self.params.slot_mapping = slot_mapping

    def prepare(self, attn_inputs):
        self.fmha_impl.prepare(attn_inputs)
        self.attn_inputs = attn_inputs

    def prepare_cuda_graph(self, attn_inputs, graph_bs=None):
        """Called by AscendGraphRunner::prepareInputs() before each replay."""
        self.attn_inputs = attn_inputs
        self.fmha_impl.prepare(attn_inputs)
        # Padded rows were zeroed by the C++ runner; the captured batch size
        # (graph_bs) is what the FIA graph task was built with, so the update
        # list must have exactly that many entries (pad rows use ctx=1).
        input_lengths = attn_inputs.input_lengths
        prefix_lengths = attn_inputs.prefix_lengths
        valid = int((input_lengths > 0).sum().item()) if input_lengths.numel() else 0
        valid = max(valid, 1)
        if graph_bs is None:
            graph_bs = valid
        kv_lens = (prefix_lengths[:valid] + input_lengths[:valid]).to(torch.int32).tolist()
        kv_lens = kv_lens[:graph_bs] + [1] * max(0, graph_bs - len(kv_lens))
        self.fmha_impl.update_graph_fia(kv_lens, graph_bs)

    def signal_graph_events(self, graph_bs):
        """Called by AscendGraphRunner before the post-capture sanity replay."""
        self.fmha_impl.signal_graph_events(graph_bs)

    def forward(self, qkv, kv_cache, layer_idx=0):
        if self.need_rope_kv_cache:
            self._update_rope_kv_write_params(qkv.device, kv_cache, qkv.shape[0], layer_idx)

            if self.rope_impl is not None:
                query, key, value = self.rope_impl.forward(qkv)
            else:
                query, key, value = self._split_qkv(qkv)

            self.kv_cache_write_op.forward(key, value, kv_cache)
            q = query
        else:
            q = qkv.chunk(3, dim=-1)[0]

        common.apply_write_cache_store(
            self.write_cache_store_impl, self.attn_inputs, kv_cache
        )
        return self.fmha_impl.forward(q, kv_cache)

    @staticmethod
    def support(attn_configs, attn_inputs):
        from rtp_llm.ops import RopeStyle
        # MRoPE is handled by AscendRotaryEmbeddingOp (npu_mrope fast path for
        # non-interleaved, pure-torch for interleaved), so both variants are
        # accepted here.
        return attn_inputs.is_prefill and \
               not attn_configs.use_mla and \
               torch.npu.is_available()


class AscendPrefillAttnOp:
    """Encapsulate NPU prefill attention op, reads cache only.

    Eager: FIA v2 with explicit actual_seq tensors.
    Graph (target verify): FIA v2 + graph_task_group/update (same vllm-ascend
    pattern as AscendDecodeAttnOp) — FIA needs the TND seq lengths on host at
    task-build time, so replay refreshes them through graph_task_update instead
    of re-capturing.
    """

    _causal_mask = None
    _shared_workspace = None
    _shared_update_stream = None

    @classmethod
    def _get_causal_mask(cls, device):
        if cls._causal_mask is None or cls._causal_mask.device.type != device.type:
            cls._causal_mask = torch.triu(
                torch.ones(2048, 2048, dtype=torch.int8), diagonal=1
            ).to(device)
        return cls._causal_mask

    def __init__(self, attn_configs, attn_inputs):
        self.attn_configs = attn_configs
        self.num_heads = attn_configs.head_num
        self.num_kv_heads = attn_configs.kv_head_num
        self.head_dim = attn_configs.size_per_head
        self.scale = attn_configs.q_scaling * (self.head_dim ** -0.5)
        self.page_size = attn_inputs.kv_cache.seq_size_per_block if \
                         attn_inputs.kv_cache else 128
        self.block_table = None
        self.actual_seq_q = None
        self.actual_seq_kv = None
        self.blocks_per_phys = _blocks_per_phys_from_config(attn_configs, attn_inputs)

        # FIA graph tasks grouped by captured stream count (mirrors decode).
        self._graph_handles = {}  # streams -> [task handles]
        self._graph_refs = {}     # streams -> [(q, k, v, table, mask, out, lse, page)]
        self._graph_events = {}   # streams -> [ExternalEvent]

    def set_params(self, params):
        self.params = params

    def prepare(self, attn_inputs):
        self.block_table = attn_inputs.kv_cache_kernel_block_id
        self.blocks_per_phys = _blocks_per_phys_from_config(self.attn_configs, attn_inputs)
        if getattr(attn_inputs, "is_cuda_graph", False):
            # Replay path: read the block table from the capture buffer (device)
            # so prepareInputs' in-place updates reach the captured FIA task.
            self.block_table = attn_inputs.kv_cache_kernel_block_id_device
            if self.block_table is not None and self.block_table.ndim != 2:
                self.block_table = self.block_table.reshape(-1, self.block_table.shape[-1])
            self.actual_seq_q = None
            self.actual_seq_kv = None
            return
        self.block_table = attn_inputs.kv_cache_kernel_block_id_host
        if self.block_table is not None:
            self.block_table = self.block_table.clamp(min=0)
            if self.block_table.ndim != 2:
                self.block_table = self.block_table.reshape(-1, self.block_table.shape[-1])

        seq_lens_q = attn_inputs.input_lengths
        seq_lens_kv = attn_inputs.prefix_lengths + attn_inputs.input_lengths
        self.actual_seq_q = torch.cumsum(seq_lens_q, dim=0)
        self.actual_seq_kv = seq_lens_kv

    def _split_kv(self, kv_cache):
        # Zero-copy K/V half views (base[:, 0/1]), matching the write op and
        # decode read addressing.  The old split_kv_kernel_blocks route
        # mis-paired K/V once the write op moved to half addressing.
        base = kv_cache.kv_cache_base
        k_cache = base[:, 0].reshape(base.shape[0], base.shape[2], -1)
        v_cache = base[:, 1].reshape(base.shape[0], base.shape[2], -1)
        return k_cache, v_cache, base.shape[2]

    def _tnd_seq_lens(self, num_tokens, num_streams):
        """TND cumulative query lengths for a uniform tokens-per-stream graph."""
        apsk = num_tokens // num_streams if num_streams > 0 else num_tokens
        return [apsk * (i + 1) for i in range(num_streams)]

    def forward(self, q, kv_cache):
        if (
            getattr(self.attn_inputs, "is_cuda_graph", False)
            and torch.npu.is_current_stream_capturing()
        ):
            return self._forward_fia_graph(q, kv_cache)

        k_cache, v_cache, page_size = self._split_kv(kv_cache)
        block_table = self.block_table
        if block_table is not None and block_table.device.type != q.device.type:
            block_table = block_table.to(q.device)
        if self.actual_seq_q is not None:
            actual_seq_q = self.actual_seq_q
            actual_seq_kv = self.actual_seq_kv
            if actual_seq_q.device.type != q.device.type:
                actual_seq_q = actual_seq_q.to(q.device)
            if actual_seq_kv is not None and actual_seq_kv.device.type != q.device.type:
                actual_seq_kv = actual_seq_kv.to(q.device)
        else:
            # Graph-mode prepare leaves these unset; only the non-capturing
            # warm-up lands here, so deriving them on host is fine.
            seq_lens_q = self.attn_inputs.input_lengths
            seq_lens_kv = self.attn_inputs.prefix_lengths + self.attn_inputs.input_lengths
            actual_seq_q = torch.cumsum(seq_lens_q, dim=0).to(q.device)
            actual_seq_kv = seq_lens_kv.to(q.device)
        atten_mask = self._get_causal_mask(q.device)
        attn_output, _ = torch_npu.npu_fused_infer_attention_score_v2(
            query=q, key=k_cache, value=v_cache,
            atten_mask=atten_mask,
            block_table=block_table,
            input_layout="TND",
            block_size=page_size,
            actual_seq_qlen=actual_seq_q,
            actual_seq_kvlen=actual_seq_kv,
            num_key_value_heads=self.num_kv_heads,
            num_query_heads=self.num_heads,
            softmax_scale=self.scale,
            sparse_mode=3,
        )
        return attn_output

    def _forward_fia_graph(self, q, kv_cache):
        k_cache, v_cache, page_size = self._split_kv(kv_cache)
        num_tokens = q.shape[0]
        num_streams = int(self.attn_inputs.input_lengths.numel())
        actual_seq_q = self._tnd_seq_lens(num_tokens, num_streams)
        # Real kv lengths are refreshed by update_graph_fia() at replay; the
        # capture-time placeholder only needs the right *shape* (stream count).
        actual_seq_kv = [page_size] * max(num_streams, 1)
        block_table = self.block_table
        if block_table is not None and block_table.device.type != q.device.type:
            block_table = block_table.to(q.device)
        atten_mask = self._get_causal_mask(q.device)
        out = torch.empty(
            num_tokens, self.num_heads, self.head_dim, dtype=q.dtype, device=q.device
        )
        lse = torch.empty(1, dtype=q.dtype, device=q.device)

        if AscendPrefillAttnOp._shared_workspace is None:
            AscendPrefillAttnOp._shared_workspace = (
                torch_npu._npu_fused_infer_attention_score_v2_get_max_workspace(
                    query=q, key=k_cache, value=v_cache,
                    atten_mask=atten_mask, block_table=block_table,
                    input_layout="TND", block_size=page_size,
                    actual_seq_qlen=actual_seq_q, actual_seq_kvlen=actual_seq_kv,
                    num_key_value_heads=self.num_kv_heads,
                    num_query_heads=self.num_heads,
                    softmax_scale=self.scale, sparse_mode=3,
                )
            )
            ws_bytes = (
                AscendPrefillAttnOp._shared_workspace.numel()
                * AscendPrefillAttnOp._shared_workspace.element_size()
            )
            AscendPrefillAttnOp._shared_workspace = torch.empty(
                ws_bytes // q.element_size(), dtype=q.dtype, device=q.device
            )

        stream = torch.npu.current_stream()
        event = torch.npu.ExternalEvent()
        event.wait(stream)
        event.reset(stream)
        torch.npu.graph_task_group_begin(stream)
        torch_npu.npu_fused_infer_attention_score_v2.out(
            query=q, key=k_cache, value=v_cache,
            atten_mask=atten_mask, block_table=block_table,
            input_layout="TND", block_size=page_size,
            actual_seq_qlen=actual_seq_q, actual_seq_kvlen=actual_seq_kv,
            num_key_value_heads=self.num_kv_heads,
            num_query_heads=self.num_heads,
            softmax_scale=self.scale, sparse_mode=3,
            workspace=AscendPrefillAttnOp._shared_workspace,
            out=[out, lse],
        )
        handle = torch.npu.graph_task_group_end(stream)

        self._graph_handles.setdefault(num_streams, []).append(handle)
        self._graph_refs.setdefault(num_streams, []).append(
            (q, k_cache, v_cache, block_table, atten_mask, out, lse, page_size)
        )
        self._graph_events.setdefault(num_streams, []).append(event)
        return out

    def signal_graph_events(self, num_streams):
        for ev in self._graph_events.get(num_streams, []):
            ev.record(torch.npu.current_stream())

    def update_graph_fia(self, ctx_list, num_streams):
        handles = self._graph_handles.get(num_streams)
        if not handles:
            return
        events = self._graph_events.get(num_streams, [])
        if AscendPrefillAttnOp._shared_update_stream is None:
            AscendPrefillAttnOp._shared_update_stream = torch.npu.Stream()
        us = AscendPrefillAttnOp._shared_update_stream
        refs = self._graph_refs.get(num_streams, [])
        with torch.npu.stream(us):
            for i, handle in enumerate(handles):
                q, k_cache, v_cache, block_table, atten_mask, out, lse, page_size = refs[i]
                if q is None or k_cache is None or out is None:
                    continue
                actual_seq_q = self._tnd_seq_lens(q.shape[0], num_streams)
                torch.npu.graph_task_update_begin(us, handle)
                torch_npu.npu_fused_infer_attention_score_v2.out(
                    query=q, key=k_cache, value=v_cache,
                    atten_mask=atten_mask, block_table=block_table,
                    input_layout="TND", block_size=page_size,
                    actual_seq_qlen=actual_seq_q, actual_seq_kvlen=ctx_list,
                    num_key_value_heads=self.num_kv_heads,
                    num_query_heads=self.num_heads,
                    softmax_scale=self.scale, sparse_mode=3,
                    workspace=AscendPrefillAttnOp._shared_workspace,
                    out=[out, lse],
                )
                torch.npu.graph_task_update_end(us)
                events[i].record(us)