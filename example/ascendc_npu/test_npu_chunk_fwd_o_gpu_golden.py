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
Test chunk_fwd_o NPU operator against GPU-collected golden data.

Loads .pt files dumped from GPU runs and compares NPU execution results
against the GPU reference output.
"""

import math
import os
import unittest

import torch

from fla_npu.ops import ascendc as ascendc_ops

torch.npu.set_device(int(os.environ.get("TEST_DEVICE_ID", 0)))

_DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "sample", "chunk_fwd_o")
_DATA_DIR = os.path.abspath(_DATA_DIR)


def _restore_strided_tensor(saved_data: torch.Tensor, meta: dict) -> torch.Tensor:
    if not meta or meta.get("contiguous", True):
        return saved_data
    dtype = getattr(torch, meta["dtype"].replace("torch.", ""))
    tensor = torch.empty_strided(meta["shape"], meta["stride"], dtype=dtype)
    tensor.copy_(saved_data)
    return tensor


def _prepare_chunk_indices(cu_seqlens, chunk_size):
    """Build flattened (seq_idx, chunk_idx) pairs for chunk_offsets."""
    rows = []
    for seq_idx, (start, end) in enumerate(zip(cu_seqlens[:-1], cu_seqlens[1:])):
        for chunk_idx in range(math.ceil((end - start) / chunk_size)):
            rows.append((seq_idx, chunk_idx))
    return [v for row in rows for v in row]


def _load_gpu_case(filename: str) -> dict:
    """Load a .pt file containing GPU golden data.

    GPU layout (rtp-llm):
      q: (B, T, NK, DK)  float16
      k: (B, T, NK, DK)  float16
      v: (B, T, NV, DV)  float16
      h: (B, num_chunks, NV, DK, DV)  float32
      g: (B, T, NV)      float32
      cu_seqlens: (B+1,)  int32

    NPU layout:
      q: (B, NK, T, DK)  bfloat16
      k: (B, NK, T, DK)  bfloat16
      v: (B, NV, T, DV)  bfloat16
      h: (B, NV, num_chunks, DK, DV)  bfloat16
      g: (B, NV, T)      float32
      cu_seqlens: int64 list
      chunk_indices: int64 list
    """
    path = os.path.join(_DATA_DIR, filename)
    data = torch.load(path, map_location="cpu", weights_only=False)

    inputs = data["inputs"]
    meta = data.get("input_meta", {})

    # Restore strides (all contiguous in current dumps, but keep for future)
    q_gpu = _restore_strided_tensor(inputs["q"], meta.get("q", {}))
    k_gpu = _restore_strided_tensor(inputs["k"], meta.get("k", {}))
    v_gpu = _restore_strided_tensor(inputs["v"], meta.get("v", {}))
    h_gpu = _restore_strided_tensor(inputs["h"], meta.get("h", {}))
    g_gpu = _restore_strided_tensor(inputs["g"], meta.get("g", {}))

    # GPU (B, T, H, D) → NPU (B, H, T, D) via transpose(1, 2)
    q_npu = q_gpu.transpose(1, 2).contiguous().to(torch.bfloat16)
    k_npu = k_gpu.transpose(1, 2).contiguous().to(torch.bfloat16)
    v_npu = v_gpu.transpose(1, 2).contiguous().to(torch.bfloat16)

    # GPU h: (B, num_chunks, NV, DK, DV) → NPU (B, NV, num_chunks, DK, DV)
    # Swap dim 1 (num_chunks) and dim 2 (NV)
    h_npu = h_gpu.transpose(1, 2).contiguous().to(torch.bfloat16)

    # GPU g: (B, T, NV) → NPU (B, NV, T) via transpose(1, 2)
    g_npu = g_gpu.transpose(1, 2).contiguous()

    # scale
    scale = inputs.get("scale")
    if scale is None:
        dk = q_npu.shape[-1]
        scale = float(dk ** -0.5)

    # cu_seqlens: int32 → int64 list
    cu_seqlens_gpu = inputs["cu_seqlens"]
    if cu_seqlens_gpu.dtype != torch.int64:
        cu_seqlens = [int(v) for v in cu_seqlens_gpu.tolist()]
    else:
        cu_seqlens = cu_seqlens_gpu.tolist()

    # chunk_size: not in dump, infer from h shape and cu_seqlens
    # num_chunks = h.shape[1] (GPU layout), seqlen = cu_seqlens[-1]
    num_chunks = h_gpu.shape[1]
    seqlen = cu_seqlens[-1]
    chunk_size = math.ceil(seqlen / num_chunks)
    # Round up to nearest power of 2 (NPU may require it for some kernels)
    # Actually chunk_size should match what was used; infer exactly
    chunk_size = max(1, seqlen // num_chunks)
    if seqlen % num_chunks != 0:
        chunk_size = math.ceil(seqlen / num_chunks)

    # chunk_indices: (seq_idx, chunk_idx) pairs, flattened
    chunk_indices = _prepare_chunk_indices(cu_seqlens, chunk_size)

    # GPU output: (B, T, NV, DV) → NPU (B, NV, T, DV) via transpose(1, 2)
    out_expected = data["outputs"]
    out_expected_npu = out_expected.transpose(1, 2).contiguous()

    return {
        "q": q_npu,
        "k": k_npu,
        "v": v_npu,
        "h": h_npu,
        "g": g_npu,
        "scale": scale,
        "cu_seqlens": cu_seqlens,
        "chunk_indices": chunk_indices,
        "chunk_size": chunk_size,
        "out_expected": out_expected_npu,
    }


@unittest.skipIf(not torch.npu.is_available(), "NPU is not available")
class TestChunkFwdOGpuGolden(unittest.TestCase):
    """Compare NPU chunk_fwd_o output against GPU golden data."""

    rtol = 5e-2
    atol = 5e-2

    def call_op(self, **kwargs):
        return ascendc_ops.npu_chunk_fwd_o(**kwargs)

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
            q=case["q"].npu(),
            k=case["k"].npu(),
            v=case["v"].npu(),
            h=case["h"].npu(),
            scale=case["scale"],
            g=case["g"].npu(),
            cu_seqlens=case["cu_seqlens"],
            chunk_indices=case["chunk_indices"],
            chunk_size=case["chunk_size"],
        )
        torch.npu.synchronize()

        self.assertTensorClose(out, case["out_expected"])

    def test_single_chunk_seq32(self):
        self._run_case("single_chunk_seq32_1chunks.pt")

    def test_multi_chunk_seq2047(self):
        self._run_case("multi_chunk_seq2047_32chunks.pt")


if __name__ == "__main__":
    unittest.main()
