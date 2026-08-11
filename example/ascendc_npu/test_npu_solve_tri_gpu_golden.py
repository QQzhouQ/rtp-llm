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
Test solve_tri NPU operator against GPU-collected golden data.

Loads .pt files dumped from GPU runs and compares NPU execution results
against the GPU reference output.

GPU dump layout: A is (B, T, H, chunk_size) — time-first (BSND)
NPU layout: x is (B, T, H, chunk_size) — same BSND layout, no transpose needed.
NPU requires float16/bfloat16 input; GPU dump A is float32, output is float16.
"""

import math
import os
import unittest

import torch

from fla_npu.ops import ascendc as ascendc_ops

torch.npu.set_device(int(os.environ.get("TEST_DEVICE_ID", 0)))

_DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "sample", "solve_tril")
_DATA_DIR = os.path.abspath(_DATA_DIR)


def _restore_strided_tensor(saved_data: torch.Tensor, meta: dict) -> torch.Tensor:
    if not meta or meta.get("contiguous", True):
        return saved_data
    dtype = getattr(torch, meta["dtype"].replace("torch.", ""))
    tensor = torch.empty_strided(meta["shape"], meta["stride"], dtype=dtype)
    tensor.copy_(saved_data)
    return tensor


def _prepare_chunk_indices(cu_seqlens, chunk_size):
    rows = []
    for seq_idx, (start, end) in enumerate(zip(cu_seqlens[:-1], cu_seqlens[1:])):
        for chunk_idx in range(math.ceil((end - start) / chunk_size)):
            rows.append((seq_idx, chunk_idx))
    return [v for row in rows for v in row]


def _load_gpu_case(filename: str) -> dict:
    """Load a .pt file containing GPU golden data.

    GPU layout (rtp-llm):
      A: (B, T, H, chunk_size) float32 — time-first (BSND)
      cu_seqlens: (B+1,) int32
      output: (B, T, H, chunk_size) float16

    NPU layout:
      x: (B, T, H, chunk_size) bfloat16 — same BSND layout
      cu_seqlens: int64 list
      chunk_indices: int64 list
      output: (B, T, H, chunk_size) bfloat16
    """
    path = os.path.join(_DATA_DIR, filename)
    data = torch.load(path, map_location="cpu", weights_only=False)

    inputs = data["inputs"]
    meta = data.get("input_meta", {})

    A_gpu = _restore_strided_tensor(inputs["A"], meta.get("A", {}))
    cu_seqlens = [int(v) for v in inputs["cu_seqlens"].tolist()]

    # chunk_size from A shape[-1]
    chunk_size = A_gpu.shape[-1]

    # chunk_indices
    chunk_indices = _prepare_chunk_indices(cu_seqlens, chunk_size)

    # NPU requires float16/bfloat16; GPU dump A is float32, output_dtype is float16
    # Use float16 to match GPU output dtype (bfloat16 has precision issues on long sequences)
    A_npu = A_gpu.to(torch.float16)

    # GPU output
    out_expected = data["outputs"]

    return {
        "A": A_npu,
        "cu_seqlens": cu_seqlens,
        "chunk_indices": chunk_indices,
        "chunk_size": chunk_size,
        "out_expected": out_expected,
    }


@unittest.skipIf(not torch.npu.is_available(), "NPU is not available")
class TestSolveTriGpuGolden(unittest.TestCase):
    """Compare NPU solve_tri output against GPU golden data."""

    rtol = 5e-2
    atol = 5e-2

    def call_op(self, **kwargs):
        return ascendc_ops.npu_solve_tri(**kwargs)

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
            x=case["A"].npu(),
            cu_seqlens=case["cu_seqlens"],
            chunk_indices=case["chunk_indices"],
            layout="bsnd",
        )
        torch.npu.synchronize()

        self.assertTensorClose(out, case["out_expected"])

    def test_seq32(self):
        self._run_case("seq32.pt")

    def test_seq2047(self):
        self._run_case("seq2047.pt")


if __name__ == "__main__":
    unittest.main()
