"""Prefill FIA KV read/write pairing regression test (real NPU required).

Regression guard for the concurrent-prefill garbled-output bug: the KV
write op and the FIA reads must share one kernel-block addressing.  The
prefill read previously materialised the pre-unification physical layout,
so V of even and K of odd kernel blocks were pulled from the wrong half
of the physical block — zeros on fresh blocks, stale data on blocks
reused from the pool shared with the linear-attention group.

The test writes K/V through the real ``AscendKVCacheWriteOp`` into a pool
pre-filled with distinctive stale values, reads through the real
``AscendPrefillAttnOp`` for a merged multi-sequence batch (one sequence
crossing the kernel-block boundary), and compares against an fp32
reference computed from the canonical kernel-block views.

Skipped automatically when no NPU is present.
"""

import unittest
from types import SimpleNamespace

import torch

try:
    import torch_npu  # noqa: F401

    _HAS_NPU = torch.npu.is_available()
except Exception:
    _HAS_NPU = False


@unittest.skipUnless(_HAS_NPU, "requires Ascend NPU")
class PrefillKVPairingTest(unittest.TestCase):
    PHYS_BLOCKS = 64
    KERNEL_PAGE = 512
    PHYS_PAGE = 1024
    BPK = PHYS_PAGE // KERNEL_PAGE
    NUM_Q_HEADS = 8
    NUM_KV_HEADS = 4
    HEAD_DIM = 128
    DTYPE = torch.bfloat16

    def _build_pool(self, seed: int):
        num_kernel = self.PHYS_BLOCKS * self.BPK
        gen = torch.Generator().manual_seed(seed)
        # Distinctive stale content everywhere: simulates a reused block
        # (linear-attention state page or a previous longer sequence).
        stale = (
            torch.randn(
                num_kernel,
                2,
                self.KERNEL_PAGE,
                self.NUM_KV_HEADS,
                self.HEAD_DIM,
                generator=gen,
            )
            * 100.0
        ).to(self.DTYPE)
        return SimpleNamespace(
            kv_cache_base=stale.to("npu"),
            seq_size_per_block=self.KERNEL_PAGE,
        )

    def test_multi_sequence_prefill_matches_reference(self):
        from rtp_llm.models_py.modules.factory.attention.ascend_impl.ascend_attn_params import (
            compute_ascend_attn_params,
        )
        from rtp_llm.models_py.modules.factory.attention.ascend_impl.ascend_kv_cache_write_op import (
            AscendKVCacheWriteOp,
        )
        from rtp_llm.models_py.modules.factory.attention.ascend_impl.ascend_prefill import (
            AscendPrefillAttnOp,
        )

        seq_lens = [300, 700, 40]  # sequence 2 crosses the 512 kernel page
        prefix_lens = [0, 0, 0]
        batch = len(seq_lens)
        total_tokens = sum(seq_lens)
        scale = self.HEAD_DIM ** -0.5

        phys_table = torch.tensor([[11], [12], [14]], dtype=torch.int32)
        kernel_table = torch.tensor(
            [[22, 0], [24, 25], [28, 0]], dtype=torch.int32
        )
        kv_cache = self._build_pool(seed=0)
        num_kernel = kv_cache.kv_cache_base.shape[0]

        gen = torch.Generator().manual_seed(1)
        q = torch.randn(
            total_tokens, self.NUM_Q_HEADS, self.HEAD_DIM, generator=gen
        )
        k = torch.randn(
            total_tokens, self.NUM_KV_HEADS, self.HEAD_DIM, generator=gen
        )
        v = torch.randn(
            total_tokens, self.NUM_KV_HEADS, self.HEAD_DIM, generator=gen
        )

        # positions + slot_mapping exactly like AscendPrefillImpl does
        attn_inputs = SimpleNamespace(
            is_prefill=True,
            prefix_lengths=torch.tensor(prefix_lens, dtype=torch.int32),
            input_lengths=torch.tensor(seq_lens, dtype=torch.int32),
            sequence_lengths=torch.tensor([], dtype=torch.int32),
            kv_cache_block_id_host=phys_table,
            kv_cache_layer_to_group=None,
            kv_cache=None,
        )
        positions, slot_mapping = compute_ascend_attn_params(
            attn_inputs, layer_idx=0, phys_page_size=self.PHYS_PAGE
        )

        # the phys-flat slots must decompose onto the kernel table ids
        pos_l = positions.long()
        batch_ids = torch.repeat_interleave(
            torch.arange(batch), torch.tensor(seq_lens)
        )
        kb_expected = kernel_table[
            batch_ids, (pos_l // self.KERNEL_PAGE).clamp(max=kernel_table.shape[1] - 1)
        ]
        self.assertTrue(
            bool((slot_mapping // self.KERNEL_PAGE == kb_expected).all()),
            "phys-flat slots must map onto kernel block ids",
        )

        write_op = AscendKVCacheWriteOp(
            num_kv_heads=self.NUM_KV_HEADS,
            head_size=self.HEAD_DIM,
            token_per_block=self.KERNEL_PAGE,
        )
        write_op.set_params(
            SimpleNamespace(
                slot_mapping=slot_mapping.to("npu"), blocks_per_phys=self.BPK
            )
        )
        write_op.forward(k.to("npu").to(self.DTYPE), v.to("npu").to(self.DTYPE), kv_cache)

        attn_configs = SimpleNamespace(
            head_num=self.NUM_Q_HEADS,
            kv_head_num=self.NUM_KV_HEADS,
            size_per_head=self.HEAD_DIM,
            q_scaling=1.0,
            tokens_per_block=self.PHYS_PAGE,
            kernel_tokens_per_block=self.KERNEL_PAGE,
        )
        inputs_for_op = SimpleNamespace(
            kv_cache=SimpleNamespace(seq_size_per_block=self.KERNEL_PAGE),
            kv_cache_kernel_block_id_host=kernel_table,
            input_lengths=torch.tensor(seq_lens, dtype=torch.int32),
            prefix_lengths=torch.tensor(prefix_lens, dtype=torch.int32),
        )
        op = AscendPrefillAttnOp(attn_configs, inputs_for_op)
        op.prepare(inputs_for_op)
        out = op.forward(q.to("npu").to(self.DTYPE), kv_cache)
        torch.npu.synchronize()

        # fp32 reference from the canonical kernel-block views
        base = kv_cache.kv_cache_base
        k_cache = base[:, 0].reshape(num_kernel, self.KERNEL_PAGE, -1)
        v_cache = base[:, 1].reshape(num_kernel, self.KERNEL_PAGE, -1)
        slot_d = slot_mapping.to("npu")
        k_tok = (
            k_cache[slot_d // self.KERNEL_PAGE, slot_d % self.KERNEL_PAGE]
            .reshape(total_tokens, self.NUM_KV_HEADS, self.HEAD_DIM)
            .float()
        )
        v_tok = (
            v_cache[slot_d // self.KERNEL_PAGE, slot_d % self.KERNEL_PAGE]
            .reshape(total_tokens, self.NUM_KV_HEADS, self.HEAD_DIM)
            .float()
        )

        rep = self.NUM_Q_HEADS // self.NUM_KV_HEADS
        ref = torch.empty(
            total_tokens, self.NUM_Q_HEADS, self.HEAD_DIM, device="npu"
        )
        off = 0
        for length in seq_lens:
            qs = q[off : off + length].to("npu").float()
            ks = k_tok[off : off + length].repeat_interleave(rep, dim=1)
            vs = v_tok[off : off + length].repeat_interleave(rep, dim=1)
            scores = torch.einsum("lhd,mhd->hlm", qs, ks) * scale
            scores.masked_fill_(
                torch.triu(
                    torch.ones(length, length, dtype=torch.bool, device="npu"),
                    diagonal=1,
                ),
                float("-inf"),
            )
            probs = torch.softmax(scores, dim=-1)
            ref[off : off + length] = torch.einsum("hlm,mhd->lhd", probs, vs)
            off += length

        diff = (out.float() - ref).abs().max().item()
        self.assertLess(
            diff, 5e-2, f"prefill FIA read must match reference (max diff {diff})"
        )


if __name__ == "__main__":
    unittest.main()
