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
Test chunk_local_cumsum NPU operator against GPU-collected golden data.

Loads .pt files dumped from GPU runs and compares NPU execution results
against the GPU reference output.  See
``docs/gpu_npu_comparison_guide.md`` for the methodology.

Layout note
-----------
The GPU dump stores ``g`` in ``(B, T, H)`` layout (``head_first=False``).
The NPU ``ChunkLocalCumsum`` tiling currently supports **only**
``head_first=True`` (``[B, H, T]``), so the test transposes ``g`` to
``(B, H, T)`` before calling the operator and transposes the NPU output
back to ``(B, T, H)`` for comparison against the GPU golden output.
"""

import math
import os
import unittest

import torch

from fla_npu.ops import ascendc as ascendc_ops

torch.npu.set_device(int(os.environ.get("TEST_DEVICE_ID", 0)))

# Path to the GPU-dumped golden data.  The data lives outside the repo
# tree under <workspace>/sample/chunk_local_cumsum; allow an env override
# for CI / container environments where the layout differs.
_DEFAULT_DATA_DIR = os.path.join(
    os.path.dirname(__file__), "..", "..", "..", "..", "sample", "chunk_local_cumsum"
)
_DATA_DIR = os.path.abspath(os.environ.get("CHUNK_LOCAL_CUMSUM_GOLDEN_DIR", _DEFAULT_DATA_DIR))


def _restore_strided_tensor(saved_data: torch.Tensor, meta: dict) -> torch.Tensor:
    """Restore a tensor's original (possibly non-contiguous) stride.

    ``.pt`` files store tensors in contiguous form, losing the original
    stride information.  The GPU dump also saves ``input_meta`` which
    records the original shape / stride / dtype / contiguous flag.

    If the original tensor was contiguous, return ``saved_data`` directly.
    Otherwise allocate a strided tensor and copy the data in.
    """
    if not meta or meta.get("contiguous", True):
        return saved_data
    dtype = getattr(torch, meta["dtype"].replace("torch.", ""))
    tensor = torch.empty_strided(tuple(meta["shape"]), tuple(meta["stride"]), dtype=dtype)
    tensor.copy_(saved_data)
    return tensor


def _resolve_scalar(value, default):
    """GPU dump stores ``reverse``/``scale`` as None when the default applies."""
    return default if value is None else value


def _next_power_of_two(value: int) -> int:
    value = max(value, 1)
    result = 1
    while result < value:
        result <<= 1
    return result


def _block_t(chunk_size: int) -> int:
    """Mirror the tiling-side BLOCK_T computation (1<<17)/chunk_size, next pow2."""
    return _next_power_of_two((1 << 17) // chunk_size)


def _prepare_chunk_indices(cu_seqlens, block_t: int):
    """Build the flattened chunk_indices_out list expected by the NPU op.

    Each (seq_idx, block_idx) pair is flattened to ``[seq_idx, block_idx, ...]``
    matching the reference implementation in the operator's local test.
    """
    rows = []
    for seq_idx, (start, end) in enumerate(zip(cu_seqlens[:-1], cu_seqlens[1:])):
        num_blocks = math.ceil((end - start) / block_t)
        for block_idx in range(num_blocks):
            rows.append((seq_idx, block_idx))
    return [value for row in rows for value in row]


def _load_gpu_case(filename: str) -> dict:
    """Load a .pt file containing GPU golden inputs/outputs.

    Restores the original (possibly non-contiguous) stride of ``g`` using
    the ``input_meta`` saved alongside the data.  The GPU layout is
    ``(B, T, H)`` (``head_first=False``); the NPU op requires
    ``head_first=True`` (``[B, H, T]``), so ``g`` is transposed before the
    call and the expected output is kept in GPU layout for a transposed
    comparison.
    """
    path = os.path.join(_DATA_DIR, filename)
    data = torch.load(path, map_location="cpu", weights_only=False)

    inputs = data["inputs"]
    meta = data.get("input_meta", {})

    g_saved = inputs["g"]
    g_meta = meta.get("g", {})
    g_gpu = _restore_strided_tensor(g_saved, g_meta) if g_meta else g_saved.contiguous()

    cu_seqlens = inputs["cu_seqlens"]
    # NPU wrapper accepts cu_seqlens as a Python list of int64; the GPU dump
    # stores int32, so convert explicitly.
    cu_seqlens_list = [int(v) for v in cu_seqlens.tolist()]

    chunk_size = int(inputs["chunk_size"])
    block_t = _block_t(chunk_size)
    chunk_indices_out = _prepare_chunk_indices(cu_seqlens_list, block_t)

    return {
        "g": g_gpu,
        "chunk_size": chunk_size,
        "reverse": _resolve_scalar(inputs.get("reverse"), False),
        "scale": float(_resolve_scalar(inputs.get("scale"), 1.0)),
        "cu_seqlens": cu_seqlens_list,
        "chunk_indices_out": chunk_indices_out,
        "y_expected": data["outputs"].contiguous(),
    }


@unittest.skipIf(not torch.npu.is_available(), "NPU is not available")
class TestChunkLocalCumsumGpuGolden(unittest.TestCase):
    """Compare NPU chunk_local_cumsum output against GPU golden data."""

    rtol = 1e-3
    atol = 2e-3

    def call_op(self, **kwargs):
        return ascendc_ops.npu_chunk_local_cumsum(**kwargs)

    def assertTensorClose(self, actual: torch.Tensor, expected: torch.Tensor, *, rtol=None, atol=None):
        rtol = self.rtol if rtol is None else rtol
        atol = self.atol if atol is None else atol
        actual_cpu = actual.detach().cpu().float()
        expected_cpu = expected.detach().cpu().float()
        self.assertTrue(
            torch.allclose(actual_cpu, expected_cpu, rtol=rtol, atol=atol),
            msg=f"max_abs_diff={(actual_cpu - expected_cpu).abs().max().item():.6f}",
        )

    def _run_case(self, case: dict, expected_shape) -> None:
        g_gpu = case["g"]                          # (B, T, H) — GPU layout
        self.assertEqual(tuple(g_gpu.shape), expected_shape, "g shape mismatch")

        # GPU (B, T, H) head_first=False  ->  NPU (B, H, T) head_first=True.
        # chunk_local_cumsum reduces along T; transposing keeps the T axis
        # values identical, only swapping the axis position.
        g_npu = g_gpu.transpose(1, 2).contiguous()  # (B, H, T)

        y_expected = case["y_expected"]            # (B, T, H)
        self.assertEqual(y_expected.dtype, torch.float32)

        y = self.call_op(
            g=g_npu.npu(),
            chunk_size=case["chunk_size"],
            cu_seqlens=case["cu_seqlens"],
            chunk_indices_out=case["chunk_indices_out"],
            reverse=case["reverse"],
            scale=case["scale"],
            head_first=True,
            output_dtype="float32",
        )

        # NPU output is (B, H, T); transpose back to GPU layout (B, T, H).
        self.assertEqual(y.dtype, torch.float32)
        y_gpu_layout = y.cpu().transpose(1, 2).contiguous()
        self.assertTensorClose(y_gpu_layout, y_expected)

    # ------------------------------------------------------------------
    # Case 1: seq2047
    #   Single sequence of 2047 tokens, H=32, chunk_size=64, fp32.
    #   GPU layout: g is (1, 2047, 32) = (B, T, H).
    #   cu_seqlens=[0, 2047], reverse=False, scale=1.0 (defaults).
    #   g is contiguous in the dump (stride=(65504, 32, 1)).
    # ------------------------------------------------------------------
    def test_seq2047(self):
        case = _load_gpu_case("seq2047.pt")
        self._run_case(case, (1, 2047, 32))

    # ------------------------------------------------------------------
    # Case 2: seq32
    #   Single sequence of 32 tokens (shorter than chunk_size=64), H=32, fp32.
    #   GPU layout: g is (1, 32, 32) = (B, T, H).
    #   cu_seqlens=[0, 32].  Exercises the partial-chunk (single chunk
    #   smaller than chunk_size) path.
    # ------------------------------------------------------------------
    def test_seq32(self):
        case = _load_gpu_case("seq32.pt")
        self._run_case(case, (1, 32, 32))


if __name__ == "__main__":
    unittest.main()
