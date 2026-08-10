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
Test fused MoE (invoke_fused_moe_kernel) on NPU via Triton for Ascend, against
GPU-collected golden data.

- ``invoke_fused_moe_kernel`` / ``moe_align_block_size_torch`` are directly
  reused from rtp-llm (kernel works as-is on NPU).
- flash-linear-attention-npu (fla) currently ships no fused-MoE operator, so
  this test uses rtp-llm's native Triton kernel as the backend.
- The rtp-llm moe module is loaded via an importlib bootstrap (stub package
  hierarchy), bypassing ``rtp_llm.__init__`` heavy C++ dependencies.

GPU source: rtp-llm/rtp_llm/models_py/triton_kernels/moe/fused_moe_kernel.py

Golden layout (in each ``sample_moe/invoke_fused_moe_kernel/*.pt``)
------------------------------------------------------------------
  inputs:
    A                    : (M, K)         float16 — input tokens
    B                    : (E, N, K)      float16 — expert weights
    C                    : (M*top_k, N)   float16 — output buffer (inplace,
                                                      scattered via sorted_token_ids)
    topk_weights         : (M*top_k,)     float32 — flat routing weights
    topk_ids             : (M*top_k,)     int32   — flat expert ids
    sorted_token_ids     : (max_padded,)  int32   — aligned token permutation
    expert_ids           : (max_blocks,)  int32   — expert per M-block
    num_tokens_post_padded: (1,)          int32   — used row count
    mul_routed_weight    : bool
    top_k                : int
    config               : dict                    — kernel launch config
    compute_type         : str                    — 'fp16' / 'bf16'
  inplace_outputs:
    C                    : (M*top_k, N)   float16 — golden kernel result

Note: the six dumped cases cover both kernel stages. The ``topk1`` cases were
dumped from the routing-weight GEMM (``mul_routed_weight=True``, N=2048),
whereas the ``topk8`` cases were dumped from the input-projection GEMM
(``mul_routed_weight=False``, N=1024). The test honours whatever flag each
golden file carries.

Golden sanity: before running the NPU kernel, each case is cross-checked with a
CPU reference that mirrors the kernel's scatter semantics
(``flat = sorted_token_ids[i]; C[flat] += (topk_weights[flat] if
mul_routed_weight else 1.0) * A[flat // top_k] @
B[expert_ids[i // BLOCK_SIZE_M]].T``), guarding against a corrupted golden dump
being silently compared against.
"""

import importlib.util
import os
import sys
import types
import unittest

import torch

torch.npu.set_device(int(os.environ.get("TEST_DEVICE_ID", 0)))

# Workspace root: <workspace>/ contains rtp-llm/ and sample_moe/.
_WORKSPACE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_RTP_LLM_ROOT = os.path.join(_WORKSPACE_ROOT, "rtp-llm")
_SAMPLE_ROOT = os.path.join(_WORKSPACE_ROOT, "sample_moe", "invoke_fused_moe_kernel")


# ---------------------------------------------------------------------------
# Bootstrap: load the rtp-llm moe kernel module without triggering
# rtp_llm.__init__ (which requires libth_transformer_config.so). We create stub
# packages and load the specific .py file via importlib.
# ---------------------------------------------------------------------------

def _load_rtp_llm_moe_modules():
    moe_dir = os.path.join(_RTP_LLM_ROOT, "rtp_llm", "models_py", "triton_kernels", "moe")
    if not os.path.isdir(moe_dir):
        raise FileNotFoundError(f"rtp-llm moe dir not found: {moe_dir}")

    for pkg_path in [
        ("rtp_llm", os.path.join(_RTP_LLM_ROOT, "rtp_llm")),
        ("rtp_llm.models_py", os.path.join(_RTP_LLM_ROOT, "rtp_llm", "models_py")),
        ("rtp_llm.models_py.triton_kernels", os.path.join(_RTP_LLM_ROOT, "rtp_llm", "models_py", "triton_kernels")),
        ("rtp_llm.models_py.triton_kernels.moe", moe_dir),
    ]:
        name, path = pkg_path
        if name not in sys.modules:
            mod = types.ModuleType(name)
            mod.__path__ = [path]
            sys.modules[name] = mod

    spec = importlib.util.spec_from_file_location(
        "rtp_llm.models_py.triton_kernels.moe.fused_moe_kernel",
        os.path.join(moe_dir, "fused_moe_kernel.py"),
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


_moe_mod = _load_rtp_llm_moe_modules()
invoke_fused_moe_kernel = _moe_mod.invoke_fused_moe_kernel


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------

def _load_pt(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"golden file not found: {path}")
    return torch.load(path, map_location="cpu", weights_only=False)


def _compute_type(ct_str):
    import triton.language as tl
    return tl.float16 if ct_str == "fp16" else tl.bfloat16


def _cpu_reference(A, B, topk_weights, topk_ids, sorted_token_ids, expert_ids,
                   num_tokens_post_padded, block_size_m, top_k, mul_routed_weight):
    """CPU reference mirroring the kernel's scatter semantics.

    For each padded row i with a valid token, ``sorted_token_ids[i]`` is a flat
    position in ``[0, M*top_k)``; the kernel maps it back via ``flat // top_k``:
        flat   = sorted_token_ids[i]
        token  = flat // top_k
        expert = expert_ids[i // BLOCK_SIZE_M]
        C[flat] += A[token] @ B[expert].T
    If ``mul_routed_weight`` is set, ``topk_weights[flat]`` is applied first.
    """
    num_valid = topk_ids.numel()
    num_tokens = num_tokens_post_padded.item()
    N = B.shape[1]
    E = B.shape[0]

    A_f = A.double()
    B_f = B.double()
    w_f = topk_weights.double()
    out = torch.zeros(num_valid, N, dtype=torch.float64)

    pos = torch.arange(num_tokens)
    flat = sorted_token_ids[:num_tokens]
    valid_mask = flat < num_valid
    flat_v = flat[valid_mask].long()
    tok_v = flat_v // top_k
    exp_v = expert_ids[(pos[valid_mask] // block_size_m).long()]

    for e in range(E):
        idx = torch.nonzero(exp_v == e).flatten()
        if idx.numel() == 0:
            continue
        flat = flat_v[idx]
        o = A_f[tok_v[idx]] @ B_f[e].t()  # [R, N]
        if mul_routed_weight:
            o = o * w_f[flat][:, None]
        out[flat] += o

    return out.float().half()


@unittest.skipIf(not torch.npu.is_available(), "NPU is not available")
class TestFusedMoeGpuGolden(unittest.TestCase):
    """Compare rtp-llm invoke_fused_moe_kernel (Triton for Ascend) against GPU golden data."""

    rtol = 1e-2
    atol = 1e-2

    def assertTensorClose(self, actual, expected, *, rtol=None, atol=None):
        rtol = self.rtol if rtol is None else rtol
        atol = self.atol if atol is None else atol
        self.assertEqual(tuple(actual.shape), tuple(expected.shape), "output shape mismatch")
        actual_cpu = actual.detach().cpu().float()
        expected_cpu = expected.detach().cpu().float()
        self.assertTrue(
            torch.allclose(actual_cpu, expected_cpu, rtol=rtol, atol=atol),
            msg=f"max_abs_diff={(actual_cpu - expected_cpu).abs().max().item():.6f}",
        )

    def _run_case(self, filename):
        path = os.path.join(_SAMPLE_ROOT, filename)
        data = _load_pt(path)
        inputs = data["inputs"]
        outputs = data.get("inplace_outputs", {})

        A = inputs["A"]
        B = inputs["B"]
        C0 = inputs["C"]
        topk_weights = inputs["topk_weights"]
        topk_ids = inputs["topk_ids"]
        sorted_token_ids = inputs["sorted_token_ids"]
        expert_ids = inputs["expert_ids"]
        num_tokens_post_padded = inputs["num_tokens_post_padded"]
        mul_routed_weight = bool(inputs["mul_routed_weight"])
        top_k = int(inputs["top_k"])
        config = dict(inputs["config"])
        compute_type = _compute_type(inputs["compute_type"])

        C_expected = outputs["C"]  # golden kernel result (inplace)

        num_valid = topk_ids.numel()
        self.assertEqual(tuple(C_expected.shape), (num_valid, B.shape[1]))
        self.assertEqual(tuple(A.shape), (num_valid // top_k, B.shape[2]))

        # Golden self-consistency: a CPU reference mirroring the kernel's
        # scatter semantics must reproduce C, guarding against a corrupted
        # golden dump being silently compared against.
        ref = _cpu_reference(
            A, B, topk_weights, topk_ids, sorted_token_ids, expert_ids,
            num_tokens_post_padded, config["BLOCK_SIZE_M"], top_k, mul_routed_weight,
        )
        self.assertTrue(
            torch.allclose(ref.float(), C_expected.float(), rtol=1e-2, atol=1e-2),
            msg="CPU ref mismatch: "
            f"max_abs_diff={(ref.float() - C_expected.float()).abs().max().item():.6f}",
        )

        # Run the rtp-llm kernel on NPU with the golden routing tensors.
        C_actual = C0.clone().npu()
        invoke_fused_moe_kernel(
            A.npu(),
            B.npu(),
            C_actual,
            topk_weights.npu(),
            topk_ids.npu(),
            sorted_token_ids.npu(),
            expert_ids.npu(),
            num_tokens_post_padded.npu(),
            mul_routed_weight,
            top_k,
            config,
            compute_type,
        )
        torch.npu.synchronize()

        self.assertTensorClose(C_actual, C_expected)

    # ------------------------------------------------------------------
    # Case 1: M=8,   E=256, topk=1  — small decode batch
    # ------------------------------------------------------------------
    def test_M8_E256_topk1(self):
        self._run_case("M8_E256_topk1.pt")

    # ------------------------------------------------------------------
    # Case 2: M=128, E=256, topk=1
    # ------------------------------------------------------------------
    def test_M128_E256_topk1(self):
        self._run_case("M128_E256_topk1.pt")

    # ------------------------------------------------------------------
    # Case 3: M=16376, E=256, topk=1 — large prefill batch
    # ------------------------------------------------------------------
    def test_M16376_E256_topk1(self):
        self._run_case("M16376_E256_topk1.pt")

    # ------------------------------------------------------------------
    # Case 4: M=1, E=256, topk=8 — single token to 8 experts (decode)
    # ------------------------------------------------------------------
    def test_M1_E256_topk8(self):
        self._run_case("M1_E256_topk8.pt")

    # ------------------------------------------------------------------
    # Case 5: M=16, E=256, topk=8
    # ------------------------------------------------------------------
    def test_M16_E256_topk8(self):
        self._run_case("M16_E256_topk8.pt")

    # ------------------------------------------------------------------
    # Case 6: M=2047, E=256, topk=8 — large prefill
    # ------------------------------------------------------------------
    def test_M2047_E256_topk8(self):
        self._run_case("M2047_E256_topk8.pt")


if __name__ == "__main__":
    unittest.main()
