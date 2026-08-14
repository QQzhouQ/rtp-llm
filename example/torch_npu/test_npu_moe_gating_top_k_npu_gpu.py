# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Tianjin University, Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""
Test torch_npu.npu_moe_gating_top_k against the GPU topkGatingSoftmaxKernelLauncher
semantics (rtp-llm SelectTopkOp).

GPU (moe_routing_kernels.cu): per row, softmax over all experts (fp32) ->
sequential arg-max top-k (tie -> smallest index) -> if has_moe_norm
(RENORMALIZE) renormalize the k selected weights to sum to 1 (else keep raw
softmax probabilities). Outputs are written in place: expert_ids (int32/int64),
expert_scales (fp32).

NPU: torch_npu.npu_moe_gating_top_k(x, k, out_flag=True) with default params
(single group, softmax norm, renorm=0, no bias) covers both modes:
  - has_moe_norm=False: y = raw softmax top-k probs (GPU NONE);
  - has_moe_norm=True : y = y / y.sum(-1) post-hoc renormalization (GPU
    RENORMALIZE; the op's own renorm param only supports 0).
Returns (y, expert_idx, norm_out); norm_out (full softmax (M,E) fp32) is valid
with out_flag=True.

No GPU golden .pt: inputs are self-constructed (2D fp32 (token, expert), aligned
to the GPU interface); a fp64 CPU reference replicates the GPU semantics exactly.
Interface differences verified: GPU expert_ids int32/int64 vs NPU expert_idx
int32; GPU expert_scales always fp32 vs NPU y dtype follows input; GPU inplace
write vs NPU 3-tuple return.
"""

import hashlib
import os
import unittest

import torch
import torch_npu

torch.npu.set_device(int(os.environ.get("TEST_DEVICE_ID", 0)))


# ---------------------------------------------------------------------------
# CPU reference: exact replication of the GPU topkGatingSoftmax semantics.
# ---------------------------------------------------------------------------

def _gpu_topk_softmax_ref(x, k, has_moe_norm):
    """Replicate GPU SelectTopkOp.forward semantics (fp64 reference).

    Args:
        x: (M, E) CPU tensor, fp32/fp16/bf16.
        k: number of selected experts (0 < k <= E).
        has_moe_norm: renormalize the k selected weights to sum to 1.

    Returns:
        weights: (M, k) fp32 — softmax probabilities of the selected experts
                 (optionally renormalized).
        indices: (M, k) int64 — selected expert indices (tie -> smallest index).
    """
    M, E = x.shape
    soft = torch.softmax(x.double(), dim=-1)
    weights = torch.zeros(M, k, dtype=torch.float64)
    indices = torch.zeros(M, k, dtype=torch.long)
    work = soft.clone()
    for ki in range(k):
        # torch.max(dim=-1) returns the FIRST maximal index => smallest index on
        # ties, matching GPU cub::ArgMax (guarded by test_tie_breaking).
        maxv, argi = work.max(dim=-1)
        weights[:, ki] = maxv
        indices[:, ki] = argi
        work.scatter_(1, argi.unsqueeze(1), -float("inf"))
    if has_moe_norm:
        weights = weights / weights.sum(dim=-1, keepdim=True)
    return weights.float(), indices


# ---------------------------------------------------------------------------
# NPU op wrapper: npu_moe_gating_top_k (defaults) for both modes; RENORM via
# post-hoc y / y.sum(-1).
# ---------------------------------------------------------------------------

def _npu_moe_gating_topk(x_npu, k, has_moe_norm):
    """Use torch_npu.npu_moe_gating_top_k (default params: single group, softmax
    norm, renorm=0, no bias) for BOTH has_moe_norm modes.

    - has_moe_norm=False: y = raw softmax top-k probabilities (GPU NONE).
    - has_moe_norm=True : y = y / y.sum(-1) post-hoc renormalization, which is
      mathematically identical to GPU RENORMALIZE (softmax_i / sum(softmax_sel));
      the op's own ``renorm`` param only supports 0.

    Returns (y, expert_idx, norm_out); norm_out (full softmax, (M,E) fp32) is
    valid with out_flag=True.
    """
    y, ei, norm_out = torch_npu.npu_moe_gating_top_k(x_npu, k, out_flag=True)
    if has_moe_norm:
        y = y / y.sum(dim=-1, keepdim=True)
    return y, ei, norm_out


# ---------------------------------------------------------------------------
# Test cases (input shapes aligned to the GPU interface: (token, expert) 2D).
# Both power-of-2 expert counts (GPU fused topkGatingSoftmax path) and
# non-power-of-2 counts (GPU moeSoftmax+moeTopK default path) are covered.
# ---------------------------------------------------------------------------

_CASES = [
    ("decode",        (1, 8),   2,  "decode, pow2 experts, k=2"),
    ("batch_pow2",    (16, 64), 4,  "small batch, pow2 experts, k=4"),
    ("prefill_pow2",  (2047, 256), 8, "prefill, pow2 experts, k=8"),
    ("batch_nonpow2", (16, 10), 3,  "small batch, non-pow2 experts, k=3"),
    ("prefill_nonpow2", (2047, 100), 4, "prefill, non-pow2 experts, k=4"),
    ("prefill_E300",  (2047, 300), 8, "prefill, E>256 non-pow2, k=8"),
    ("topk1",         (16, 128), 1, "k=1"),
]


@unittest.skipIf(not torch.npu.is_available(), "NPU is not available")
class TestNpuMoeGatingTopKNpuVsGpu(unittest.TestCase):
    """Compare NPU moe top-k gating softmax against GPU kernel semantics."""

    rtol = 1e-2
    atol = 1e-2
    # NPU computes in the input precision; the reference is fp64 -> the only
    # error source is fp32 rounding of the op itself (~1e-7). Tolerance is
    # generous to absorb fp16 inputs too.

    def assertTensorClose(self, actual, expected, *, rtol=None, atol=None, msg_prefix=""):
        rtol = self.rtol if rtol is None else rtol
        atol = self.atol if atol is None else atol
        self.assertEqual(tuple(actual.shape), tuple(expected.shape), f"{msg_prefix} output shape mismatch")
        self.assertEqual(actual.dtype, expected.dtype, f"{msg_prefix} output dtype mismatch")
        actual_cpu = actual.detach().cpu().float()
        expected_cpu = expected.detach().cpu().float()
        self.assertTrue(
            torch.allclose(actual_cpu, expected_cpu, rtol=rtol, atol=atol),
            msg=f"{msg_prefix} max_abs_diff={(actual_cpu - expected_cpu).abs().max().item():.6f}",
        )

    def _run_case(self, tag, shape, k, has_moe_norm, x_dtype=torch.float32):
        M, E = shape
        # Stable per-case seed (hashlib, NOT builtin hash() which is salted per
        # process) so the constructed inputs are reproducible across runs.
        torch.manual_seed(int(hashlib.md5(tag.encode()).hexdigest()[:8], 16))
        x_cpu = torch.randn(shape, dtype=torch.float32)  # fp32 GPU-style logits
        if x_dtype != torch.float32:
            x_cpu = x_cpu.to(x_dtype)

        # CPU reference (GPU semantics)
        w_ref, idx_ref = _gpu_topk_softmax_ref(x_cpu.float(), k, has_moe_norm)
        sm_ref = torch.softmax(x_cpu.double(), dim=-1).float()  # full softmax

        # NPU op
        y, ei, third = _npu_moe_gating_topk(x_cpu.npu(), k, has_moe_norm)
        torch.npu.synchronize()

        # Interface checks
        self.assertEqual(tuple(y.shape), (M, k), f"[{tag}] y shape")
        self.assertEqual(y.dtype, x_cpu.dtype, f"[{tag}] y dtype follows input")
        self.assertEqual(ei.dtype, torch.int32, f"[{tag}] expert_idx is int32")
        self.assertEqual(tuple(ei.shape), (M, k), f"[{tag}] expert_idx shape")

        # Value checks (NPU y follows input dtype; convert the fp32 reference to
        # y's dtype so both sides are the same precision representation).
        self.assertTensorClose(y, w_ref.to(x_dtype), msg_prefix=f"[{tag}] weights")
        self.assertTrue(torch.equal(ei.cpu().long(), idx_ref), f"[{tag}] expert_idx mismatch")

        # Renorm: selected weights must sum to 1 (GPU RENORMALIZE / y/y.sum(-1)).
        if has_moe_norm:
            wsum = y.float().sum(dim=-1)
            self.assertTrue(
                torch.allclose(wsum, torch.ones_like(wsum), rtol=1e-3, atol=1e-3),
                f"[{tag}] renorm weights should sum to 1",
            )

        # Third return: npu_moe_gating_top_k's norm_out = full softmax (M,E)
        # fp32 (valid with out_flag=True), for both modes.
        self.assertEqual(tuple(third.shape), (M, E), f"[{tag}] norm_out shape (M,E)")
        self.assertEqual(third.dtype, torch.float32, f"[{tag}] norm_out dtype fp32")
        self.assertTrue(
            torch.allclose(third.cpu().float(), sm_ref, rtol=1e-5, atol=1e-5),
            msg=f"[{tag}] norm_out should match softmax(x)",
        )

    def test_none_mode(self):
        """has_moe_norm=False -> raw softmax top-k probs (GPU NONE)."""
        for tag, shape, k, desc in _CASES:
            with self.subTest(tag=tag, has_moe_norm=False):
                self._run_case(tag, shape, k, has_moe_norm=False)

    def test_renorm_mode(self):
        """has_moe_norm=True -> post-hoc y/y.sum(-1) (weights sum to 1)."""
        for tag, shape, k, desc in _CASES:
            with self.subTest(tag=tag, has_moe_norm=True):
                self._run_case(tag, shape, k, has_moe_norm=True)

    def test_fp16_input(self):
        """NPU y follows input dtype (fp16); GPU expert_scales is always fp32.
        Covers both NONE and RENORM modes with fp16 input."""
        with self.subTest(tag="batch_fp16_none", has_moe_norm=False):
            self._run_case("batch_fp16", (16, 64), 4, has_moe_norm=False, x_dtype=torch.float16)
        with self.subTest(tag="batch_fp16_renorm", has_moe_norm=True):
            self._run_case("batch_fp16r", (16, 64), 4, has_moe_norm=True, x_dtype=torch.float16)

    def test_tie_breaking(self):
        """All-equal logits -> uniform softmax; ties resolved to smallest index
        (matches GPU arg-max with strict '>' compare / cub ArgMax)."""
        x_cpu = torch.zeros((4, 8), dtype=torch.float32)
        for has_moe_norm in (False, True):
            with self.subTest(has_moe_norm=has_moe_norm):
                w_ref, idx_ref = _gpu_topk_softmax_ref(x_cpu, 2, has_moe_norm)
                y, ei, _ = _npu_moe_gating_topk(x_cpu.npu(), 2, has_moe_norm)
                torch.npu.synchronize()
                self.assertTensorClose(y, w_ref)
                self.assertTrue(torch.equal(ei.cpu().long(), idx_ref))
                self.assertTrue(torch.equal(idx_ref, torch.tensor([[0, 1]] * 4)))


if __name__ == "__main__":
    unittest.main()
