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
Test chunk_scaled_dot_kkt NPU operator against GPU-collected golden data.

Loads .pt files dumped from GPU runs and compares NPU execution results
against the GPU reference output.
"""

import math
import os
import unittest

import torch

from fla_npu.ops import ascendc as ascendc_ops

torch.npu.set_device(int(os.environ.get("TEST_DEVICE_ID", 0)))

_DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "sample", "chunk_scaled_dot_kkt_fwd")
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
      beta:     (B, T, Hv)      float16
      g_cumsum: (B, T, Hv)      float32
      cu_seqlens: (B+1,)        int32
      output:   (B, T, Hk, chunk_size)  float32

    NPU layout:
      k:    (B, Hk, T, Dk)  float16/bfloat16
      g:    (B, Hv, T)      float32
      beta: (B, Hv, T)      float32
      cu_seqlens: int64 list
      chunk_indices: int64 list
      output: (B, Hk, T, chunk_size)  float32
    """
    path = os.path.join(_DATA_DIR, filename)
    data = torch.load(path, map_location="cpu", weights_only=False)

    inputs = data["inputs"]
    meta = data.get("input_meta", {})

    # Restore strides (all contiguous in current dumps)
    k_gpu = _restore_strided_tensor(inputs["k"], meta.get("k", {}))
    beta_gpu = _restore_strided_tensor(inputs["beta"], meta.get("beta", {}))
    g_gpu = _restore_strided_tensor(inputs["g_cumsum"], meta.get("g_cumsum", {}))

    # GPU (B, T, H, D) → NPU (B, H, T, D) via transpose(1, 2)
    k_npu = k_gpu.transpose(1, 2).contiguous()
    g_npu = g_gpu.transpose(1, 2).contiguous()
    # NPU requires beta as float32 (op def: DT_FLOAT)
    beta_npu = beta_gpu.transpose(1, 2).contiguous().to(torch.float32)

    # GQA: GPU g/beta have Hv heads, NPU uses Hk heads.
    # NPU head h_k uses g[h_k] and k[h_k]; GPU head h_v uses g[h_v] and k[h_v // ratio].
    # Subsample g/beta to match: g_npu[h_k] = g_gpu[h_k * ratio].
    Hk = k_npu.shape[1]
    Hv = g_npu.shape[1]
    ratio = Hv // Hk
    if ratio > 1:
        g_npu = g_npu[:, ::ratio].contiguous()
        beta_npu = beta_npu[:, ::ratio].contiguous()

    # cu_seqlens: int32 → int64 list
    cu_seqlens_gpu = inputs["cu_seqlens"]
    cu_seqlens = [int(v) for v in cu_seqlens_gpu.tolist()]

    # chunk_size: infer from output shape[-1]
    out_gpu = data["outputs"]
    chunk_size = out_gpu.shape[-1]

    # chunk_indices
    chunk_indices = _prepare_chunk_indices(cu_seqlens, chunk_size)

    # GPU output: (B, T, Hv, chunk_size) → NPU (B, Hk, T, chunk_size)
    # NPU outputs Hk heads. GPU head h_v=h_k*ratio matches NPU head h_k
    # (same k[h_k], same g[h_k*ratio]).
    Hk = k_npu.shape[1]
    out_expected_npu = out_gpu.transpose(1, 2).contiguous()[:, ::ratio][:, :Hk]

    return {
        "k": k_npu,
        "g": g_npu,
        "beta": beta_npu,
        "cu_seqlens": cu_seqlens,
        "chunk_indices": chunk_indices,
        "chunk_size": chunk_size,
        "out_expected": out_expected_npu,
    }


@unittest.skipIf(not torch.npu.is_available(), "NPU is not available")
class TestChunkScaledDotKktGpuGolden(unittest.TestCase):
    """Compare NPU chunk_scaled_dot_kkt output against GPU golden data."""

    rtol = 5e-2
    atol = 5e-2

    def call_op(self, **kwargs):
        return ascendc_ops.npu_chunk_scaled_dot_kkt(**kwargs)

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

        out = self.call_op(
            k=case["k"].npu(),
            g=case["g"].npu(),
            beta=case["beta"].npu(),
            cu_seqlens=case["cu_seqlens"],
            chunk_indices=case["chunk_indices"],
            chunk_size=case["chunk_size"],
        )
        torch.npu.synchronize()

        self.assertTensorClose(out, case["out_expected"])

    def test_seq32(self):
        self._run_case("seq32.pt")

    def test_seq2047(self):
        self._run_case("seq2047.pt")


if __name__ == "__main__":
    unittest.main()
