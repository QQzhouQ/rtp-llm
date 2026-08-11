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
Test sigmoid_gate_scale_add_triton on NPU via Triton for Ascend, against
GPU-collected golden data.

- ``sigmoid_gate_scale_add_triton`` is directly reused from rtp-llm (kernel
  works as-is on NPU), loaded via an importlib bootstrap (stub package
  hierarchy) bypassing ``rtp_llm.__init__`` heavy C++ dependencies.

GPU source: rtp-llm/rtp_llm/models_py/triton_kernels/common/moe_gating.py

Semantics (in-place on *experts*):
    experts[t, :] = sigmoid(gate[t, 0]) * shared[t, :] + experts[t, :]

Golden layout (in each ``sample_moe/sigmoid_gate_scale_add_triton/*.pt``)
------------------------------------------------------------------------
  inputs:
    gate     : (T, 1)   float16 — scalar gate per token
    shared   : (T, H)   float16 — shared expert MLP output
    experts  : (T, H)   float16 — routed experts output (initial value)
  outputs   : (T, H)    float16 — golden kernel result (= final experts)
  inplace_outputs:
    experts  : (T, H)   float16 — same values as outputs (separate buffer)

Golden sanity: before running the NPU kernel, each case is cross-checked with a
CPU reference (``sigmoid(gate) * shared + experts`` in fp32), guarding against
a corrupted golden dump being silently compared against.
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
_SAMPLE_ROOT = os.path.join(_WORKSPACE_ROOT, "sample_moe", "sigmoid_gate_scale_add_triton")


# ---------------------------------------------------------------------------
# Bootstrap: load the rtp-llm moe_gating module without triggering
# rtp_llm.__init__ (which requires libth_transformer_config.so). We create stub
# packages and load the specific .py file via importlib.
# ---------------------------------------------------------------------------

def _load_rtp_llm_moe_gating_modules():
    common_dir = os.path.join(_RTP_LLM_ROOT, "rtp_llm", "models_py", "triton_kernels", "common")
    if not os.path.isdir(common_dir):
        raise FileNotFoundError(f"rtp-llm triton_kernels/common dir not found: {common_dir}")

    for pkg_path in [
        ("rtp_llm", os.path.join(_RTP_LLM_ROOT, "rtp_llm")),
        ("rtp_llm.models_py", os.path.join(_RTP_LLM_ROOT, "rtp_llm", "models_py")),
        ("rtp_llm.models_py.triton_kernels", os.path.join(_RTP_LLM_ROOT, "rtp_llm", "models_py", "triton_kernels")),
        ("rtp_llm.models_py.triton_kernels.common", common_dir),
    ]:
        name, path = pkg_path
        if name not in sys.modules:
            mod = types.ModuleType(name)
            mod.__path__ = [path]
            sys.modules[name] = mod

    spec = importlib.util.spec_from_file_location(
        "rtp_llm.models_py.triton_kernels.common.moe_gating",
        os.path.join(common_dir, "moe_gating.py"),
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


_moe_gating_mod = _load_rtp_llm_moe_gating_modules()
sigmoid_gate_scale_add_triton = _moe_gating_mod.sigmoid_gate_scale_add_triton


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------

def _load_pt(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"golden file not found: {path}")
    return torch.load(path, map_location="cpu", weights_only=False)


def _cpu_reference(gate, shared, experts):
    """Pure-PyTorch reference (fp32 arithmetic), mirroring the kernel."""
    result = torch.sigmoid(gate.float()) * shared.float() + experts.float()
    return result.to(shared.dtype)


@unittest.skipIf(not torch.npu.is_available(), "NPU is not available")
class TestSigmoidGateScaleAddGpuGolden(unittest.TestCase):
    """Compare rtp-llm sigmoid_gate_scale_add_triton (Triton for Ascend) against GPU golden data."""

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

        gate = inputs["gate"]      # [T, 1]
        shared = inputs["shared"]  # [T, H]
        experts = inputs["experts"]  # [T, H] (initial value)
        experts_expected = outputs["experts"]

        T, H = shared.shape
        self.assertEqual(tuple(gate.shape), (T, 1))
        self.assertEqual(tuple(experts.shape), (T, H))
        self.assertEqual(tuple(experts_expected.shape), (T, H))
        self.assertEqual(experts_expected.dtype, shared.dtype)

        # Golden self-consistency: a CPU reference must reproduce the golden
        # output, guarding against a corrupted golden dump being silently
        # compared against.
        ref = _cpu_reference(gate, shared, experts)
        self.assertTrue(
            torch.allclose(ref.float(), experts_expected.float(), rtol=1e-2, atol=1e-2),
            msg="CPU ref mismatch: "
            f"max_abs_diff={(ref.float() - experts_expected.float()).abs().max().item():.6f}",
        )

        # Run the rtp-llm kernel on NPU. The kernel modifies *experts*
        # in-place and returns the same object.
        gate_npu = gate.npu()
        shared_npu = shared.npu()
        experts_actual = experts.clone().npu()
        ret = sigmoid_gate_scale_add_triton(gate_npu, shared_npu, experts_actual)
        torch.npu.synchronize()

        self.assertTrue(ret is experts_actual, "kernel must modify experts in-place")
        self.assertTensorClose(experts_actual, experts_expected)

    # ------------------------------------------------------------------
    # Case 1: T=1,    H=2048 — single decode token
    # ------------------------------------------------------------------
    def test_T1_H2048(self):
        self._run_case("T1_H2048.pt")

    # ------------------------------------------------------------------
    # Case 2: T=16,   H=2048
    # ------------------------------------------------------------------
    def test_T16_H2048(self):
        self._run_case("T16_H2048.pt")

    # ------------------------------------------------------------------
    # Case 3: T=2047, H=2048 — large prefill batch
    # ------------------------------------------------------------------
    def test_T2047_H2048(self):
        self._run_case("T2047_H2048.pt")


if __name__ == "__main__":
    unittest.main()
