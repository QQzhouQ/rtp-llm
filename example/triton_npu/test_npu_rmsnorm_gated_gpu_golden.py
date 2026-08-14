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
Test RmsNormGated on NPU via Triton for Ascend, against GPU-collected golden data.

- The test exercises the actual model entry point ``RmsNormGated.forward(x, gate)``
  (the nn.Module wrapper in rtp-llm), which internally calls ``layer_norm_fwd``
  -> ``_layer_norm_fwd_1pass_kernel`` (rtp_llm/models_py/triton_kernels/common/layernorm_gated.py),
  loaded via an importlib bootstrap (stub package hierarchy) bypassing
  ``rtp_llm.__init__`` heavy C++ dependencies.
- The kernel runs as-is on NPU (heuristics + ``tl.sum`` + ``tl.sqrt`` +
  ``tl.sigmoid`` all compile under Triton for Ascend).

PORTING NOTE: rtp-llm's ``layer_norm_fwd`` wrapper hard-codes
``with torch.cuda.device(x.device.index):`` around the kernel launch. On an
NPU-only build this raises "PyTorch was compiled without CUDA support". This
test reuses the whole rtp-llm wrapper and simply swaps the device context
manager ``torch.cuda.device`` -> ``torch.npu.device`` (monkeypatch, restored
afterwards) — equivalent to editing that one line in rtp-llm source. The kernel
and all launch logic (BLOCK_N / num_warps / grid) are reused unchanged.

This follows the same "reuse what works, fix what doesn't" pattern as
``test_npu_block_ops_gpu_golden.py``: there the rtp-llm kernel needed a Triton
for Ascend rewrite (pointer reassignment -> ``tl.where``); here only the CUDA
device-context line needs a device-agnostic swap, since the kernel compiles
and runs on Triton for Ascend as-is.

Semantics (``is_rms_norm=True``, ``norm_before_gate=True``, ``activation='silu'``):
    rstd = 1 / sqrt(mean(x^2) + eps)
    y    = (x * rstd * weight) * silu(gate)

Golden layout (in each ``sample/RmsNormGated/*.pt``)
----------------------------------------------------
  inputs:
    x      : (M, N)   float16
    gate   : (M, N)   float16
  outputs : (M, N)    float16 — golden result
  model_state:
    weight     : (N,)      float16
    eps        : float     = 1e-6
    group_size : int       = 128
"""

import importlib.util
import os
import sys
import types
import unittest
from contextlib import contextmanager

import torch
import torch_npu

torch.npu.set_device(int(os.environ.get("TEST_DEVICE_ID", 0)))

# Workspace root: <workspace>/ contains rtp-llm/ and sample/.
_WORKSPACE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_RTP_LLM_ROOT = os.path.join(_WORKSPACE_ROOT, "rtp-llm")
_SAMPLE_ROOT = os.path.join(_WORKSPACE_ROOT, "sample", "RmsNormGated")


# ---------------------------------------------------------------------------
# Bootstrap: load the rtp-llm layernorm_gated module without triggering
# rtp_llm.__init__ (which requires libth_transformer_config.so).
# ---------------------------------------------------------------------------

def _load_rtp_llm_layernorm_gated():
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
        "rtp_llm.models_py.triton_kernels.common.layernorm_gated",
        os.path.join(common_dir, "layernorm_gated.py"),
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


_layernorm_gated_mod = _load_rtp_llm_layernorm_gated()
RmsNormGated = _layernorm_gated_mod.RmsNormGated


# ---------------------------------------------------------------------------
# NPU call: RmsNormGated.forward -> layer_norm_fwd -> with torch.cuda.device(...)
# rtp-llm hard-codes the CUDA device context manager; swap it for the NPU one
# (torch.npu.device) around the call, equivalent to editing that line in source.
# ---------------------------------------------------------------------------

@contextmanager
def _npu_device_ctx():
    """Temporarily use torch.npu.device as the device context manager."""
    orig = torch.cuda.device
    torch.cuda.device = torch.npu.device
    try:
        yield
    finally:
        torch.cuda.device = orig


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------

def _load_pt(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"golden file not found: {path}")
    return torch.load(path, map_location="cpu", weights_only=False)


def _cpu_reference(x, gate, weight, eps, group_size):
    """Pure-PyTorch reference (fp32), mirroring the kernel semantics."""
    M, N = x.shape
    ngroups = N // group_size
    x_f = x.float().view(M, ngroups, group_size)
    gate_f = gate.float().view(M, ngroups, group_size)
    w_f = weight.float().view(ngroups, group_size)
    rstd = 1.0 / torch.sqrt(x_f.pow(2).mean(dim=-1, keepdim=True) + eps)
    y = (x_f * rstd * w_f) * torch.nn.functional.silu(gate_f)
    return y.view(M, N).to(x.dtype)


@unittest.skipIf(not torch.npu.is_available(), "NPU is not available")
class TestRmsNormGatedGpuGolden(unittest.TestCase):
    """Compare the rtp-llm RmsNormGated kernel (Triton for Ascend) against GPU golden data."""

    rtol = 1e-2
    atol = 1e-2

    def assertTensorClose(self, actual, expected, *, rtol=None, atol=None):
        rtol = self.rtol if rtol is None else rtol
        atol = self.atol if atol is None else atol
        self.assertEqual(tuple(actual.shape), tuple(expected.shape), "output shape mismatch")
        self.assertEqual(actual.dtype, expected.dtype, "output dtype mismatch")
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
        model_state = data["model_state"]

        x = inputs["x"]
        gate = inputs["gate"]
        weight = model_state["weight"]
        eps = float(model_state["eps"])
        group_size = int(model_state["group_size"])
        out_expected = data["outputs"]

        M, N = x.shape
        self.assertEqual(tuple(gate.shape), (M, N))
        self.assertEqual(tuple(weight.shape), (N,))
        self.assertEqual(tuple(out_expected.shape), (M, N))

        # Golden self-consistency: a CPU reference must reproduce the golden
        # output, guarding against a corrupted golden dump.
        ref = _cpu_reference(x, gate, weight, eps, group_size)
        self.assertTrue(
            torch.allclose(ref.float(), out_expected.float(), rtol=1e-2, atol=1e-2),
            msg="CPU ref mismatch: "
            f"max_abs_diff={(ref.float() - out_expected.float()).abs().max().item():.6f}",
        )

        # Instantiate the actual rtp-llm RmsNormGated module and call
        # forward(x, gate) — the same entry point the model uses.
        module = RmsNormGated(weight.npu(), None, group_size, eps, activation="silu")
        with _npu_device_ctx():
            out = module(x.npu(), gate.npu())
        torch.npu.synchronize()

        self.assertTensorClose(out, out_expected)

    # ------------------------------------------------------------------
    # Case 1: M=32    — decode
    # ------------------------------------------------------------------
    def test_decode_seq1(self):
        self._run_case("decode_seq1.pt")

    # ------------------------------------------------------------------
    # Case 2: M=1024  — prefill seq 32
    # ------------------------------------------------------------------
    def test_prefill_seq32(self):
        self._run_case("prefill_seq32.pt")

    # ------------------------------------------------------------------
    # Case 3: M=65504 — prefill seq 2047
    # ------------------------------------------------------------------
    def test_prefill_seq2047(self):
        self._run_case("prefill_seq2047.pt")


if __name__ == "__main__":
    unittest.main()
