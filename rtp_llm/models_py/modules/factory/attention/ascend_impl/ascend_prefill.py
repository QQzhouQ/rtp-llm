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
        # set by forward() each call: tokens per stream for the verify graph
        # (qkv rows / batch); host shape math only.
        self._verify_tokens_per_stream = 1

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

    def _update_rope_kv_write_params(self, device, kv_cache, layer_idx: int = 0):
        # The kernel block granularity comes from the per-layer cache view; the
        # physical block size it maps onto drives slot_mapping and the writes.
        blocks_per_phys = _blocks_per_phys_from_config(self.attn_configs, self.attn_inputs)
        kernel_page = kv_cache.seq_size_per_block if kv_cache is not None else 0
        self.params.blocks_per_phys = blocks_per_phys
        if getattr(self.attn_inputs, "is_cuda_graph", False):
            self._update_verify_params_device(kv_cache, kernel_page)
            return
        positions, slot_mapping = compute_ascend_attn_params(
            self.attn_inputs, layer_idx, kernel_page * blocks_per_phys
        )
        self.params.positions_d = positions.to(device, non_blocking=True)
        self.params.slot_mapping = slot_mapping.to(device, non_blocking=True)

    def _update_verify_params_device(self, kv_cache, kernel_page: int):
        """Device-side positions / slot_mapping / FIA actual_seq for aclgraph
        capture (target-verify, T tokens per stream, stream-major layout).

        Everything is computed with device ops from the capture buffers the
        C++ runner refreshes on every replay (``prefix_lengths_d`` /
        ``input_lengths_d`` / kernel block table), so the captured gather /
        cumsum chain recomputes automatically — no H2D from derived host
        tensors inside the graph.  The AttnOp's host-computed actual_seq is
        overwritten with the device tensors (its forward already moves
        device tensors through unchanged).
        """
        attn = self.attn_inputs
        prefix_d = attn.prefix_lengths_d
        input_d = attn.input_lengths_d
        block_table = attn.kv_cache_kernel_block_id_device
        if (prefix_d is None or input_d is None or prefix_d.numel() == 0
                or block_table is None or block_table.numel() == 0):
            return
        if not kernel_page:
            kernel_page = (
                attn.kv_cache.seq_size_per_block
                if getattr(attn, "kv_cache", None) is not None else 128
            )
        if block_table.ndim != 2:
            block_table = block_table.reshape(-1, block_table.shape[-1])

        # tokens per stream: set by forward() before this call (qkv rows /
        # batch — host shape math only, no D2H).
        tokens_per_stream = self._verify_tokens_per_stream
        batch = prefix_d.shape[0]

        ar = torch.arange(tokens_per_stream, device=prefix_d.device, dtype=prefix_d.dtype)
        positions = prefix_d.repeat_interleave(tokens_per_stream) + ar.repeat(batch)
        pos_long = positions.reshape(-1).long()
        max_blocks = block_table.shape[1]
        block_index = (pos_long // kernel_page).clamp(max=max_blocks - 1)
        block_offset = pos_long % kernel_page
        # per-row gather: stream-major [B*T] positions -> [B, T] column indices
        # against the [B, cols] block table
        slot_block_numbers = torch.gather(
            block_table, 1, block_index.view(batch, tokens_per_stream).to(block_table.dtype)
        ).reshape(-1).long().clamp(min=0)
        slot_mapping = (slot_block_numbers * kernel_page + block_offset).to(torch.int64)

        self.params.positions_d = positions
        self.params.slot_mapping = slot_mapping
        self.fmha_impl.actual_seq_q = torch.cumsum(input_d, dim=0).to(torch.int32)
        self.fmha_impl.actual_seq_kv = (prefix_d + input_d).to(torch.int32)
        self.fmha_impl.block_table = block_table

    def prepare(self, attn_inputs):
        self.fmha_impl.prepare(attn_inputs)
        self.attn_inputs = attn_inputs
        # TODO: Ascend Is not called outside, will be called in graph mode

    def prepare_cuda_graph(self, attn_inputs, graph_bs=None):
        """Called by AscendGraphRunner::prepareInputs() before each replay.

        Refreshes the captured FIA tasks' actual_seq_kvlen (= prefix + T per
        stream) via the AttnOp's graph-task update stream — the qlen side
        (cumsum of the fixed T tokens per stream) is static per bucket.
        The update must address the CAPTURED bucket's handles: the runner
        pads the request up to the bucket bs, so key by graph_bs (not the
        valid prefix rows) and pad kvlen with tps for the padding rows.
        """
        self.attn_inputs = attn_inputs
        self.fmha_impl.prepare(attn_inputs)
        AscendPrefillAttnOp.set_verify_tokens(self._verify_tokens_per_stream)
        tps = self._verify_tokens_per_stream
        prefix = attn_inputs.prefix_lengths
        if prefix is None or prefix.numel() == 0 or graph_bs is None:
            return
        bucket = int(graph_bs)
        valid = min(int(prefix.shape[0]), bucket)
        kvlen = (prefix[:valid].to(torch.int32) + tps).tolist()
        kvlen = kvlen[:bucket] + [tps] * max(0, bucket - len(kvlen))
        self.fmha_impl.update_graph_fia_verify(kvlen, bucket)

    def signal_graph_events(self, graph_bs):
        self.fmha_impl.signal_graph_events(graph_bs)

    def forward(self, qkv, kv_cache, layer_idx=0):
        # tokens per stream for the verify-graph device-metadata path (host
        # shape math only).  Falls back to 1 for plain prefill.
        prefix = getattr(self.attn_inputs, "prefix_lengths", None)
        if prefix is not None and prefix.numel() > 0:
            self._verify_tokens_per_stream = max(1, qkv.shape[0] // prefix.shape[0])
        else:
            self._verify_tokens_per_stream = 1
        AscendPrefillAttnOp.set_verify_tokens(self._verify_tokens_per_stream)

        if getattr(self.attn_inputs, "is_cuda_graph", False) and not self.need_rope_kv_cache:
            # FIA device metadata still required even without rope/KV-write
            kernel_page = kv_cache.seq_size_per_block if kv_cache is not None else 0
            self._update_verify_params_device(kv_cache, kernel_page)

        if self.need_rope_kv_cache:
            self._update_rope_kv_write_params(qkv.device, kv_cache, layer_idx)

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
        return attn_inputs.is_prefill and \
               not attn_configs.use_mla and \
               torch.npu.is_available()


class AscendPrefillAttnOp:
    """Encapsulate NPU prefill attention op, reads cache only."""

    _causal_mask = None

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
        AscendPrefillAttnOp._bind_class_config(self)

    def set_params(self, params):
        self.params = params

    def prepare(self, attn_inputs):
        self.block_table = attn_inputs.kv_cache_kernel_block_id_host
        self.blocks_per_phys = _blocks_per_phys_from_config(self.attn_configs, attn_inputs)
        if self.block_table is not None:
            self.block_table = self.block_table.clamp(min=0)
            if self.block_table.ndim != 2:
                self.block_table = self.block_table.reshape(-1, self.block_table.shape[-1])

        seq_lens_q = attn_inputs.input_lengths
        seq_lens_kv = attn_inputs.prefix_lengths + attn_inputs.input_lengths
        self.actual_seq_q = torch.cumsum(seq_lens_q, dim=0)
        self.actual_seq_kv = seq_lens_kv
    # ---- aclgraph capture (target-verify): official dispatch-mode machinery ----
    # Uses torch_npu's _GraphDispatchMode (the official wrapper behind
    # graph.update / auto_dispatch_capture, verified by sglang): capture-time
    # the mode intercepts the FIA v2 call, injects the ExternalEvent wait
    # point + graph task group and records the call; replay-time
    # update_capture_record re-launches each recorded task on the update
    # stream with the refreshed actual_seq_kvlen.  A hand-rolled variant of
    # this machinery produced wrong replays; the official one is bit-exact
    # (offline verified).  actual_seq_qlen (cumsum of the fixed T per stream)
    # is static per bucket and needs no update.
    _dispatch_modes = {}    # bs -> _GraphDispatchMode (records for that bucket)
    _tokens_per_stream = 3  # k + 1; refreshed from the impl before capture
    _num_kv_heads = None    # class-level copies for the classmethod update
    _num_heads = None
    _scale = None

    @classmethod
    def set_verify_tokens(cls, t: int):
        cls._tokens_per_stream = int(t)

    @classmethod
    def _bind_class_config(cls, instance):
        cls._num_kv_heads = instance.num_kv_heads
        cls._num_heads = instance.num_heads
        cls._scale = instance.scale

    @classmethod
    def signal_graph_events(cls, batch_size):
        mode = cls._dispatch_modes.get(batch_size)
        if mode is None:
            return
        stream = torch.npu.current_stream()
        for rec in mode.graph_dispatch_records:
            rec.event.record(stream)

    @classmethod
    def update_graph_fia_verify(cls, kvlen_list, batch_size):
        """Refresh each captured FIA task's actual_seq_kvlen via the official
        dispatch-mode record update (re-launch on the update stream inside
        graph_task_update_begin/end).  Called by the impl's
        prepare_cuda_graph before every replay."""
        mode = cls._dispatch_modes.get(batch_size)
        if mode is None or not mode.graph_dispatch_records:
            return
        mode.update_capture_record([{"actual_seq_kvlen": kvlen_list}])

    def _forward_fia_graph(self, q, k_cache, v_cache, block_table, actual_seq_q, actual_seq_kv, atten_mask, page_size, batch_size):
        from torch_npu.npu.graphs import _GraphDispatchMode
        tps = AscendPrefillAttnOp._tokens_per_stream
        actual_seq_q_list = [tps * (i + 1) for i in range(batch_size)]
        actual_seq_kv_list = [page_size] * batch_size  # placeholder; updated per replay
        mode = AscendPrefillAttnOp._dispatch_modes.setdefault(batch_size, _GraphDispatchMode())
        with mode:
            # plain .default call: the dispatch mode swaps in workspace/out
            # pre-allocation, records the task group and the ExternalEvent
            # wait point (official IFA v2 handler).  The handler returns
            # kwargs["out"] == [output, softmax_lse] — unwrap the attention
            # output for the caller.
            result = torch_npu.npu_fused_infer_attention_score_v2(
                query=q, key=k_cache, value=v_cache,
                atten_mask=atten_mask, block_table=block_table,
                input_layout="TND", block_size=page_size,
                actual_seq_qlen=actual_seq_q_list,
                actual_seq_kvlen=actual_seq_kv_list,
                num_key_value_heads=self.num_kv_heads,
                num_query_heads=self.num_heads,
                softmax_scale=self.scale, sparse_mode=3,
            )
            if isinstance(result, (list, tuple)):
                return result[0]
            return result


    def forward(self, q, kv_cache):
        # Zero-copy kernel-block K/V views in the 3-D form FIA requires —
        # mirroring AscendDecodeAttnOp._split_kv (the verified decode path).
        # The old split_kv_kernel_blocks route rebuilt a physical [K|V] view
        # over a kernel-interleaved layout: V got mis-mapped onto the K second
        # half, so prefill/verify read stale bytes as V (zeros on a fresh pool
        # -> silently degraded FIA layers; real K values after any request
        # crossed 512 tokens -> garbage V and cumulative corruption).
        base = kv_cache.kv_cache_base
        k_cache = base[:, 0].reshape(base.shape[0], base.shape[2], -1)
        v_cache = base[:, 1].reshape(base.shape[0], base.shape[2], -1)
        page_size = base.shape[2]
        block_table = self.block_table
        if block_table is not None and block_table.device.type != q.device.type:
            block_table = block_table.to(q.device)
        actual_seq_q = self.actual_seq_q
        if actual_seq_q is not None and actual_seq_q.device.type != q.device.type:
            actual_seq_q = actual_seq_q.to(q.device)
        actual_seq_kv = self.actual_seq_kv
        if actual_seq_kv is not None and actual_seq_kv.device.type != q.device.type:
            actual_seq_kv = actual_seq_kv.to(q.device)
        atten_mask = self._get_causal_mask(q.device)
        if torch.npu.is_current_stream_capturing():
            tps = AscendPrefillAttnOp._tokens_per_stream
            batch_size = max(1, q.shape[0] // tps)
            return self._forward_fia_graph(
                q, k_cache, v_cache, block_table,
                actual_seq_q, actual_seq_kv, atten_mask, page_size, batch_size)
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