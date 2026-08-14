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
Test the MoE top-k gating softmax operator on NPU against the GPU kernel semantics.

GPU side (rtp-llm)
------------------
``SelectTopkOp.forward(router_logits_fp32, expert_ids, expert_scales)`` (C++,
SelectTopkOp.cc) launches ``tensorrt_llm::kernels::invokeSelectExpertsForTokens``
-> ``topkGatingSoftmaxKernelLauncher`` (moe_routing_kernels.cu). Semantics:
  1. softmax over all experts per row (fp32),
  2. sequential arg-max top-k (tie -> smallest expert index),
  3. if ``has_moe_norm`` (MOEExpertScaleNormalizationMode::RENORMALIZE):
     renormalize the k selected weights to sum to 1 (otherwise keep raw
     softmax probabilities).
It writes ``expert_ids`` (int32/int64) and ``expert_scales`` (fp32, shape
(token_num, top_k)) in place; the full softmax buffer and ``source_rows``
(= ``k_idx * num_rows + row``) are internal.

NPU side (torch_npu)
--------------------
  v1 ``torch_npu.npu_moe_gating_top_k_softmax(x, finished=None, k)``
     softmax -> topk (no renorm)     == GPU has_moe_norm=False
  v2 ``torch_npu.npu_moe_gating_top_k_softmax_v2(x, *, k=1, finished=None,
                                                 renorm=0, output_softmax=False)``
     renorm=0: softmax -> topk;  renorm=1: topk -> softmax
     renorm=1 == GPU RENORMALIZE (mathematically identical: softmax over all
     experts, then renormalize the selected k == softmax over the selected k).
The module-level call style is the officially documented one (also reachable
as ``torch.ops.npu.npu_moe_gating_top_k_softmax``; ``torch.npu.*`` does NOT
exist).
Returns ``(y, expert_idx, row_idx)``:
  - y           : top-k weights, dtype = x.dtype, shape (..., k)
  - expert_idx  : int32, shape (..., k)
  - row_idx     : v1 filled with ``k_idx*M + row`` (same convention as GPU
                  ``source_rows``); v2 returns an empty tensor (0,) in this build.

This test constructs its own input cases (no GPU golden .pt), aligns the input
shape to the GPU interface (2D fp32 contiguous ``(token_num, num_experts)``),
and checks the NPU op against a CPU reference that replicates the GPU kernel
semantics exactly.

Interface differences verified here:
  - GPU ``expert_ids`` may be int32 OR int64 (dispatch on dtype);
    NPU ``expert_idx`` is always int32.
  - GPU ``expert_scales`` is always fp32; NPU ``y`` follows the input dtype.
  - GPU ``source_rows`` is internal; NPU ``row_idx`` is returned (v1) / empty (v2).
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
# NPU op wrapper (v1 for has_moe_norm=False, v2 renorm=1 for has_moe_norm=True).
# ---------------------------------------------------------------------------

def _npu_topk_gating_softmax(x_npu, k, has_moe_norm):
    # Official / documented call style: torch_npu.npu_moe_gating_top_k_softmax
    # (also reachable as torch.ops.npu.npu_moe_gating_top_k_softmax).
    if has_moe_norm:
        y, ei, row_idx = torch_npu.npu_moe_gating_top_k_softmax_v2(
            x_npu, k=k, finished=None, renorm=1, output_softmax=False)
    else:
        y, ei, row_idx = torch_npu.npu_moe_gating_top_k_softmax(x_npu, None, k)
    return y, ei, row_idx


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
class TestTopkGatingSoftmaxNpuVsGpu(unittest.TestCase):
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

        # NPU op
        y, ei, row_idx = _npu_topk_gating_softmax(x_cpu.npu(), k, has_moe_norm)
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

        # Renorm: selected weights must sum to 1 (GPU RENORMALIZE / v2 renorm=1)
        if has_moe_norm:
            wsum = y.float().sum(dim=-1)
            self.assertTrue(
                torch.allclose(wsum, torch.ones_like(wsum), rtol=1e-3, atol=1e-3),
                f"[{tag}] renorm weights should sum to 1",
            )

        # row_idx conventions: v1 filled (k*M+row), v2 empty
        if has_moe_norm:
            self.assertEqual(tuple(row_idx.shape), (0,), f"[{tag}] v2 row_idx empty")
        else:
            self.assertEqual(tuple(row_idx.shape), (M, k), f"[{tag}] v1 row_idx shape")
            # GPU source_rows convention: source_rows[r, ki] = ki*M + r (k-major)
            expected_row_idx = (torch.arange(k, dtype=torch.long) * M).unsqueeze(0) + \
                torch.arange(M, dtype=torch.long).unsqueeze(1)
            self.assertTrue(
                torch.equal(row_idx.cpu(), expected_row_idx),
                f"[{tag}] v1 row_idx should be ki*M+r (GPU source_rows convention)",
            )

    def test_none_mode(self):
        """has_moe_norm=False -> NPU v1 (softmax then topk, no renorm)."""
        for tag, shape, k, desc in _CASES:
            with self.subTest(tag=tag, has_moe_norm=False):
                self._run_case(tag, shape, k, has_moe_norm=False)

    def test_renorm_mode(self):
        """has_moe_norm=True -> NPU v2 renorm=1 (weights sum to 1)."""
        for tag, shape, k, desc in _CASES:
            with self.subTest(tag=tag, has_moe_norm=True):
                self._run_case(tag, shape, k, has_moe_norm=True)

    def test_fp16_input(self):
        """NPU y follows input dtype (fp16); GPU expert_scales is always fp32.
        Covers both NONE (v1) and RENORM (v2 renorm=1) with fp16 input."""
        with self.subTest(tag="batch_fp16_none", has_moe_norm=False):
            self._run_case("batch_fp16", (16, 64), 4, has_moe_norm=False, x_dtype=torch.float16)
        with self.subTest(tag="batch_fp16_renorm", has_moe_norm=True):
            self._run_case("batch_fp16r", (16, 64), 4, has_moe_norm=True, x_dtype=torch.float16)

    def test_v2_renorm0_equals_v1(self):
        """v2 renorm=0 == v1 == GPU NONE (softmax->topk, no renorm)."""
        for tag, shape, k, desc in _CASES[:3]:
            with self.subTest(tag=tag):
                torch.manual_seed(int(hashlib.md5((tag + "_v2v1").encode()).hexdigest()[:8], 16))
                x_cpu = torch.randn(shape, dtype=torch.float32)
                y1, ei1, _ = torch_npu.npu_moe_gating_top_k_softmax(x_cpu.npu(), None, k)
                y2, ei2, _ = torch_npu.npu_moe_gating_top_k_softmax_v2(x_cpu.npu(), k=k, renorm=0)
                torch.npu.synchronize()
                self.assertTensorClose(y2, y1, msg_prefix=f"[{tag}] v2(renorm=0) vs v1")
                self.assertTrue(torch.equal(ei2.cpu(), ei1.cpu()), f"[{tag}] v2(renorm=0) vs v1 idx")

    def test_v2_output_softmax_structure(self):
        """v2's third return is overloaded: with renorm=0 + output_softmax=True it
        becomes the FULL softmax (M, E) (matches softmax(x)); otherwise it stays
        empty (0,). y / expert_idx are unchanged by the flag."""
        torch.manual_seed(0)
        x_cpu = torch.randn((16, 64), dtype=torch.float32)
        # renorm=0 + output_softmax=True -> 3rd = full softmax (M, E)
        ya, eia, sm = torch_npu.npu_moe_gating_top_k_softmax_v2(
            x_cpu.npu(), k=4, renorm=0, output_softmax=True)
        torch.npu.synchronize()
        self.assertEqual(tuple(sm.shape), (16, 64), "renorm=0 + output_softmax=True -> (M,E)")
        self.assertEqual(sm.dtype, torch.float32)
        sm_ref = torch.softmax(x_cpu.double(), dim=-1).float()
        self.assertTrue(
            torch.allclose(sm.cpu().float(), sm_ref, rtol=1e-5, atol=1e-5),
            msg="full softmax output should match softmax(x)",
        )
        # y / expert_idx unchanged vs output_softmax=False
        yb, eib, ria = torch_npu.npu_moe_gating_top_k_softmax_v2(
            x_cpu.npu(), k=4, renorm=0, output_softmax=False)
        torch.npu.synchronize()
        self.assertTensorClose(ya, yb, msg_prefix="output_softmax y unchanged")
        self.assertTrue(torch.equal(eia.cpu(), eib.cpu()), "output_softmax idx unchanged")
        self.assertEqual(tuple(ria.shape), (0,), "renorm=0 + output_softmax=False -> 3rd empty")
        # renorm=1 + output_softmax=True -> 3rd stays empty
        _, _, ri1 = torch_npu.npu_moe_gating_top_k_softmax_v2(
            x_cpu.npu(), k=4, renorm=1, output_softmax=True)
        torch.npu.synchronize()
        self.assertEqual(tuple(ri1.shape), (0,), "renorm=1 + output_softmax=True -> 3rd empty")

    def test_tie_breaking(self):
        """All-equal logits -> uniform softmax; ties resolved to smallest index
        (matches GPU arg-max with strict '>' compare / cub ArgMax)."""
        x_cpu = torch.zeros((4, 8), dtype=torch.float32)
        for has_moe_norm in (False, True):
            with self.subTest(has_moe_norm=has_moe_norm):
                w_ref, idx_ref = _gpu_topk_softmax_ref(x_cpu, 2, has_moe_norm)
                y, ei, _ = _npu_topk_gating_softmax(x_cpu.npu(), 2, has_moe_norm)
                torch.npu.synchronize()
                self.assertTensorClose(y, w_ref)
                self.assertTrue(torch.equal(ei.cpu().long(), idx_ref))
                self.assertTrue(torch.equal(idx_ref, torch.tensor([[0, 1]] * 4)))


if __name__ == "__main__":
    unittest.main()
