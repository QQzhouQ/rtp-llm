# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Tianjin University, Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not be in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""
Test recompute_wu_fwd NPU operator against GPU-collected golden data.

Loads .pt files dumped from GPU runs and compares NPU execution results
against the GPU reference output.
"""

import math
import os
import unittest

import torch

from fla_npu.ops import ascendc as ascendc_ops

torch.npu.set_device(int(os.environ.get("TEST_DEVICE_ID", 0)))

_DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "sample", "recompute_w_u_fwd")
_DATA_DIR = os.path.abspath(_DATA_DIR)


def _restore_strided_tensor(saved_data: torch.Tensor, meta: dict) -> torch.Tensor:
    if not meta or meta.get("contiguous", True):
        return saved_data
    dtype = getattr(torch, meta["dtype"].replace("torch.", ""))
    tensor = torch.empty_strided(meta["shape"], meta["stride"], dtype=dtype)
    tensor.copy_(saved_data)
    return tensor


def _prepare_chunk_indices(cu_seqlens, chunk_size):
    """Build flattened (seq_idx, chunk_idx) pairs for chunk_indices."""
    rows = []
    for seq_idx, (start, end) in enumerate(zip(cu_seqlens[:-1], cu_seqlens[1:])):
        for chunk_idx in range(math.ceil((end - start) / chunk_size)):
            rows.append((seq_idx, chunk_idx))
    return [v for row in rows for v in row]


def _load_gpu_case(filename: str) -> dict:
    """Load a .pt file containing GPU golden data.

    GPU layout (rtp-llm):
      k:        (B, T, Hk, Dk)  float16
      v:        (B, T, Hv, Dv)  float16
      beta:     (B, T, Hv)      float16
      A:        (B, T, Hv, chunk_size)  float16
      g_cumsum: (B, T, Hv)      float32
      cu_seqlens: (B+1,)        int32
      outputs[0] (w): (B, T, Hv, Dk)  float16
      outputs[1] (u): (B, T, Hv, Dv)  float16

    NPU layout:
      k:    (B, Hk, T, Dk)  float16
      v:    (B, Hv, T, Dv)  float16
      beta: (B, Hv, T)      float16
      A:    (B, Hv, T, chunk_size)  float16
      g:    (B, Hv, T)      float32
      w:    (B, Hv, T, Dk)  float16
      u:    (B, Hv, T, Dv)  float16
    """
    path = os.path.join(_DATA_DIR, filename)
    data = torch.load(path, map_location="cpu", weights_only=False)

    inputs = data["inputs"]
    meta = data.get("input_meta", {})

    # Restore strides (all contiguous in current dumps)
    k_gpu = _restore_strided_tensor(inputs["k"], meta.get("k", {}))
    v_gpu = _restore_strided_tensor(inputs["v"], meta.get("v", {}))
    beta_gpu = _restore_strided_tensor(inputs["beta"], meta.get("beta", {}))
    A_gpu = _restore_strided_tensor(inputs["A"], meta.get("A", {}))
    g_gpu = _restore_strided_tensor(inputs["g_cumsum"], meta.get("g_cumsum", {}))

    # GPU (B, T, H, D) → NPU (B, H, T, D) via transpose(1, 2)
    # ascend950: k/v/A = bfloat16, beta/g = float32 (op def variant 2/3)
    k_npu = k_gpu.transpose(1, 2).contiguous().to(torch.bfloat16)
    v_npu = v_gpu.transpose(1, 2).contiguous().to(torch.bfloat16)
    beta_npu = beta_gpu.transpose(1, 2).contiguous().to(torch.float32)
    A_npu = A_gpu.transpose(1, 2).contiguous().to(torch.bfloat16)
    g_npu = g_gpu.transpose(1, 2).contiguous()  # float32

    # cu_seqlens: int32 → int64 list
    cu_seqlens = [int(v) for v in inputs["cu_seqlens"].tolist()]

    # chunk_size: infer from A shape[-1]
    chunk_size = A_gpu.shape[-1]

    # chunk_indices
    chunk_indices = _prepare_chunk_indices(cu_seqlens, chunk_size)

    # GPU outputs: list of [w, u], each (B, T, Hv, D) → NPU (B, Hv, T, D)
    w_expected = data["outputs"][0].transpose(1, 2).contiguous()
    u_expected = data["outputs"][1].transpose(1, 2).contiguous()

    return {
        "k": k_npu,
        "v": v_npu,
        "beta": beta_npu,
        "A": A_npu,
        "g": g_npu,
        "cu_seqlens": cu_seqlens,
        "chunk_indices": chunk_indices,
        "chunk_size": chunk_size,
        "w_expected": w_expected,
        "u_expected": u_expected,
    }


@unittest.skipIf(not torch.npu.is_available(), "NPU is not available")
class TestRecomputeWuFwdGpuGolden(unittest.TestCase):
    """Compare NPU recompute_wu_fwd output against GPU golden data."""

    rtol = 5e-2
    atol = 1e-1

    def call_op(self, **kwargs):
        return ascendc_ops.npu_recompute_w_u_fwd(**kwargs)

    def assertTensorClose(self, actual: torch.Tensor, expected: torch.Tensor, *, rtol=None, atol=None):
        rtol = self.rtol if rtol is None else rtol
        atol = self.atol if atol is None else atol
        actual_cpu = actual.detach().cpu().float()
        expected_cpu = expected.detach().cpu().float()
        self.assertTrue(
            torch.allclose(actual_cpu, expected_cpu, rtol=rtol, atol=atol),
            msg=f"max_abs_diff={(actual_cpu - expected_cpu).abs().max().item():.6f}",
        )

    def _run_case(self, filename):
        case = _load_gpu_case(filename)

        w, u = self.call_op(
            k=case["k"].npu(),
            v=case["v"].npu(),
            beta=case["beta"].npu(),
            A=case["A"].npu(),
            chunk_size=case["chunk_size"],
            g=case["g"].npu(),
            cu_seqlens=case["cu_seqlens"],
            chunk_indices=case["chunk_indices"],
        )
        torch.npu.synchronize()

        self.assertTensorClose(w, case["w_expected"])
        self.assertTensorClose(u, case["u_expected"])

    def test_seq32(self):
        self._run_case("seq32.pt")

    def test_seq2047(self):
        self._run_case("seq2047.pt")


if __name__ == "__main__":
    unittest.main()
