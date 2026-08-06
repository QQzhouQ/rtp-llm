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
Test l2norm_fwd NPU operator against GPU-collected golden data.

Loads .pt files dumped from GPU runs and compares NPU execution results
against the GPU reference output.

Layout note
-----------
The GPU dump stores the input in ``(B, T, H, D)`` token-major layout:
  x: (B, T, H, D)  float16 — the tensor to be L2-normalized along the last dim

The kernel normalizes each row (flattened ``B*T*H`` rows of length ``D``) to
unit L2 norm:
  y = x / sqrt(sum(x * x, dim=-1) + eps)

The operator implementation lives in flash-linear-attention-npu as a Triton
kernel ``fla/ops/triton/triton_core/l2norm.py`` (``l2norm_fwd``), which returns
``(y, rstd)``. The golden dump stores only ``y``, so the test compares ``y``
against the golden output and also validates the unit-norm property.
"""

import os
import unittest

import torch

from fla.ops.triton.triton_core.l2norm import l2norm_fwd

# Path to the GPU-dumped golden data. The data lives outside the repo tree under
# <workspace>/sample/l2norm_fwd; allow an env override for CI / container
# environments where the layout differs.
_DEFAULT_DATA_DIR = os.path.join(
    os.path.dirname(__file__), "..", "..", "..", "sample", "l2norm_fwd"
)
_DATA_DIR = os.path.abspath(os.environ.get("L2NORM_FWD_GOLDEN_DIR", _DEFAULT_DATA_DIR))

torch.npu.set_device(int(os.environ.get("TEST_DEVICE_ID", 0)))


def _restore_strided_tensor(saved_data: torch.Tensor, meta: dict) -> torch.Tensor:
    """Restore a tensor's original (possibly non-contiguous) stride.

    ``.pt`` files store tensors in contiguous form, losing the original
    stride information. The GPU dump also saves ``input_meta`` which records
    the original shape / stride / dtype / contiguous flag.

    If the original tensor was contiguous, return ``saved_data`` directly.
    Otherwise allocate a strided tensor and copy the data in.
    """
    if not meta or meta.get("contiguous", True):
        return saved_data
    dtype = getattr(torch, meta["dtype"].replace("torch.", ""))
    tensor = torch.empty_strided(tuple(meta["shape"]), tuple(meta["stride"]), dtype=dtype)
    tensor.copy_(saved_data)
    return tensor


def _load_gpu_case(filename: str) -> dict:
    """Load a .pt file containing GPU golden inputs/outputs."""
    path = os.path.join(_DATA_DIR, filename)
    data = torch.load(path, map_location="cpu", weights_only=False)

    inputs = data["inputs"]
    meta = data.get("input_meta", {})

    x = _restore_strided_tensor(inputs["x"], meta.get("x", {}))

    # GPU output: y (same shape as x), float16.
    y_expected = data["outputs"]

    return {
        "x": x,
        "y_expected": y_expected,
    }


@unittest.skipIf(not torch.npu.is_available(), "NPU is not available")
class TestL2NormFwdGpuGolden(unittest.TestCase):
    """Compare flash-linear-attention-npu l2norm_fwd (Triton) output against GPU golden data."""

    rtol = 5e-2
    atol = 5e-2

    def call_op(self, x, eps=1e-6):
        return l2norm_fwd(x, eps=eps)

    def assertTensorClose(self, actual: torch.Tensor, expected: torch.Tensor, *, rtol=None, atol=None):
        rtol = self.rtol if rtol is None else rtol
        atol = self.atol if atol is None else atol
        self.assertEqual(tuple(actual.shape), tuple(expected.shape), "output shape mismatch")
        actual_cpu = actual.detach().cpu().float()
        expected_cpu = expected.detach().cpu().float()
        self.assertTrue(
            torch.allclose(actual_cpu, expected_cpu, rtol=rtol, atol=atol),
            msg=f"max_abs_diff={(actual_cpu - expected_cpu).abs().max().item():.6f}",
        )

    def _run_case(self, filename: str) -> None:
        case = _load_gpu_case(filename)

        # Sanity-check the loaded golden tensors before execution.
        self.assertEqual(tuple(case["y_expected"].shape), tuple(case["x"].shape))
        self.assertEqual(case["y_expected"].dtype, torch.float16)

        # The kernel normalizes the flattened (B*T*H, D) rows along the last dim.
        y, rstd = self.call_op(
            case["x"].npu(),
            eps=1e-6,
        )
        torch.npu.synchronize()

        # Compare the normalized output against the GPU golden output.
        self.assertEqual(tuple(y.shape), tuple(case["y_expected"].shape))
        self.assertTensorClose(y, case["y_expected"])

        # Validate the unit-L2-norm property of the normalized rows.
        y_flat = y.detach().cpu().float().reshape(-1, y.shape[-1])
        norms = y_flat.norm(dim=-1)
        self.assertTrue(
            torch.allclose(norms, torch.ones_like(norms), rtol=1e-2, atol=1e-2),
            msg=f"L2 norm not close to 1: min={norms.min().item():.6f} max={norms.max().item():.6f}",
        )

    # ------------------------------------------------------------------
    # Case 1: prefill_T256
    #   x: (1, 16, 16, 128) float16 — D=128 rows.
    # ------------------------------------------------------------------
    def test_prefill_T256(self):
        self._run_case("prefill_T256.pt")

    # ------------------------------------------------------------------
    # Case 2: prefill_T32752
    #   x: (1, 2047, 16, 128) float16 — larger token count.
    # ------------------------------------------------------------------
    def test_prefill_T32752(self):
        self._run_case("prefill_T32752.pt")


if __name__ == "__main__":
    unittest.main()
