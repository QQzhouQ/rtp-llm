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
Test fused_gdn_gating NPU operator against GPU-collected golden data.

Loads .pt files dumped from GPU runs and compares NPU execution results
against the GPU reference output.

Layout note
-----------
The GPU dump stores the gate inputs in token-major layout:
  A_log  : (H,)       float16 — per-head log-space gate coefficient
  a      : (S, H)     float16 — softplus input (non-contiguous in the dump,
                                stride (64, 1): each row lives in a wider
                                cache buffer and only the first H entries
                                are used)
  b      : (S, H)     float16 — sigmoid input (same stride semantics as a)
  dt_bias: (H,)       float16 — per-head bias added to a

The GPU kernel emits two outputs (with a leading singleton batch dim):
  g    : (1, S, H)    float32 — -exp(A_log) * softplus(a + dt_bias)
  beta : (1, S, H)    float16 — sigmoid(b)

The operator is the rtp-llm Triton kernel ``fused_gdn_gating`` in
``rtp_llm/models_py/triton_kernels/fla/gdn_gating.py``. ``rtp_llm``'s package
``__init__`` tries to load a compiled ``.so`` that is absent in a source tree,
so the module is loaded directly with ``importlib`` to keep the test standalone.
The kernel runs on NPU via triton-ascend and matches the vllm-ascend
``fused_gdn_gating_patch`` bit-for-bit (verified diff = 0).
"""

import importlib.util
import os
import unittest

import torch


def _load_rtp_fused_gdn_gating():
    """Load rtp-llm's ``fused_gdn_gating`` without importing the ``rtp_llm`` package.

    ``rtp_llm.__init__`` requires a compiled ``libth_transformer_config.so`` that
    is not present in a source checkout, so we load ``gdn_gating.py`` directly.
    """
    path = os.path.join(
        os.path.dirname(__file__),
        "..", "..", "..",
        "rtp-llm", "rtp_llm", "models_py", "triton_kernels", "fla", "gdn_gating.py",
    )
    spec = importlib.util.spec_from_file_location("_rtp_gdn_gating", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.fused_gdn_gating


_fused_gdn_gating = _load_rtp_fused_gdn_gating()

torch.npu.set_device(int(os.environ.get("TEST_DEVICE_ID", 0)))

# Path to the GPU-dumped golden data. The data lives outside the repo tree under
# <workspace>/sample/fused_gdn_gating; allow an env override for CI / container
# environments where the layout differs.
_DEFAULT_DATA_DIR = os.path.join(
    os.path.dirname(__file__), "..", "..", "..", "sample", "fused_gdn_gating"
)
_DATA_DIR = os.path.abspath(os.environ.get("FUSED_GDN_GATING_GOLDEN_DIR", _DEFAULT_DATA_DIR))


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
    """Load a .pt file containing GPU golden inputs/outputs.

    Restores the original (possibly non-contiguous) strides of ``a``/``b``
    from ``input_meta``: in the dump they are views with a row stride wider
    than H (e.g. (64, 1) for H=32).
    """
    path = os.path.join(_DATA_DIR, filename)
    data = torch.load(path, map_location="cpu", weights_only=False)

    inputs = data["inputs"]
    meta = data.get("input_meta", {})

    A_log = _restore_strided_tensor(inputs["A_log"], meta.get("A_log", {}))
    a = _restore_strided_tensor(inputs["a"], meta.get("a", {}))
    b = _restore_strided_tensor(inputs["b"], meta.get("b", {}))
    dt_bias = _restore_strided_tensor(inputs["dt_bias"], meta.get("dt_bias", {}))

    # GPU outputs: g (float32) then beta (float16), both (1, S, H).
    g_expected = data["outputs"][0]
    beta_expected = data["outputs"][1]

    return {
        "A_log": A_log,
        "a": a,
        "b": b,
        "dt_bias": dt_bias,
        "g_expected": g_expected,
        "beta_expected": beta_expected,
    }


@unittest.skipIf(not torch.npu.is_available(), "NPU is not available")
class TestFusedGdnGatingGpuGolden(unittest.TestCase):
    """Compare rtp-llm fused_gdn_gating (Triton) output against GPU golden data."""

    rtol = 5e-2
    atol = 5e-2

    def call_op(self, **kwargs):
        return _fused_gdn_gating(**kwargs)

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
        self.assertEqual(tuple(case["g_expected"].shape), (1,) + tuple(case["a"].shape))
        self.assertEqual(tuple(case["beta_expected"].shape), (1,) + tuple(case["b"].shape))
        self.assertEqual(case["g_expected"].dtype, torch.float32)
        self.assertEqual(case["beta_expected"].dtype, torch.float16)

        # The rtp-llm kernel supports non-contiguous a/b via ``stride_ab``, so the
        # restored views can be passed directly (no contiguous() needed).
        g, beta = self.call_op(
            A_log=case["A_log"].npu(),
            a=case["a"].npu(),
            b=case["b"].npu(),
            dt_bias=case["dt_bias"].npu(),
        )
        torch.npu.synchronize()

        # Compare NPU Triton results against the GPU golden output.
        self.assertEqual(tuple(g.shape), tuple(case["g_expected"].shape))
        self.assertEqual(tuple(beta.shape), tuple(case["beta_expected"].shape))
        self.assertTensorClose(g, case["g_expected"])
        self.assertTensorClose(beta, case["beta_expected"])

    # ------------------------------------------------------------------
    # Case 1: decode_seq1
    #   Single decode token (S=1), H=32. a/b are stride (64, 1) but dim0=1 so
    #   torch considers them contiguous.
    # ------------------------------------------------------------------
    def test_decode_seq1(self):
        self._run_case("decode_seq1.pt")

    # ------------------------------------------------------------------
    # Case 2: prefill_seq32
    #   32 prefill tokens, H=32. a/b are non-contiguous (stride (64, 1)).
    # ------------------------------------------------------------------
    def test_prefill_seq32(self):
        self._run_case("prefill_seq32.pt")

    # ------------------------------------------------------------------
    # Case 3: prefill_seq2047
    #   2047 prefill tokens, H=32. a/b are non-contiguous (stride (64, 1)).
    # ------------------------------------------------------------------
    def test_prefill_seq2047(self):
        self._run_case("prefill_seq2047.pt")


if __name__ == "__main__":
    unittest.main()
