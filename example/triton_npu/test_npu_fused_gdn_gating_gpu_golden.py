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
Test fused_gdn_gating on NPU via Triton for Ascend, against GPU-collected golden data.

- fused_gdn_gating: directly reused from rtp-llm (kernel works as-is on NPU;
  it supports non-contiguous ``a``/``b`` via ``stride_ab``, so no contiguous()
  or kernel adaptation is needed).
- The rtp-llm fla module is loaded via an importlib bootstrap (stub package
  hierarchy), bypassing ``rtp_llm.__init__`` heavy C++ dependencies.

GPU source: rtp-llm/rtp_llm/models_py/triton_kernels/fla/gdn_gating.py

Golden layout (token-major)
---------------------------
  A_log  : (H,)       float16 — per-head log-space gate coefficient
  a      : (S, H)     float16 — softplus input (non-contiguous in the dump,
                                stride (64, 1): each row lives in a wider
                                cache buffer, only the first H entries used)
  b      : (S, H)     float16 — sigmoid input (same stride semantics as a)
  dt_bias: (H,)       float16 — per-head bias added to a

Two outputs (with a leading singleton batch dim):
  g    : (1, S, H)  float32 — -exp(A_log) * softplus(a + dt_bias)
  beta : (1, S, H)  float16 — sigmoid(b)
"""

import importlib.util
import os
import sys
import types
import unittest

import torch

torch.npu.set_device(int(os.environ.get("TEST_DEVICE_ID", 0)))

# Workspace root: <workspace>/ contains rtp-llm/ and sample/.
_WORKSPACE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_RTP_LLM_ROOT = os.path.join(_WORKSPACE_ROOT, "rtp-llm")
_SAMPLE_ROOT = os.path.join(_WORKSPACE_ROOT, "sample")


# ---------------------------------------------------------------------------
# Bootstrap: load rtp-llm fla modules without triggering rtp_llm.__init__
# (which requires libth_transformer_config.so). We create stub packages and
# load the specific .py files via importlib.
# ---------------------------------------------------------------------------

def _load_rtp_llm_fla_modules():
    fla_dir = os.path.join(_RTP_LLM_ROOT, "rtp_llm", "models_py", "triton_kernels", "fla")
    if not os.path.isdir(fla_dir):
        raise FileNotFoundError(f"rtp-llm fla dir not found: {fla_dir}")

    # Stub package hierarchy
    for pkg_path in [
        ("rtp_llm", os.path.join(_RTP_LLM_ROOT, "rtp_llm")),
        ("rtp_llm.models_py", os.path.join(_RTP_LLM_ROOT, "rtp_llm", "models_py")),
        ("rtp_llm.models_py.triton_kernels", os.path.join(_RTP_LLM_ROOT, "rtp_llm", "models_py", "triton_kernels")),
        ("rtp_llm.models_py.triton_kernels.fla", fla_dir),
    ]:
        name, path = pkg_path
        if name not in sys.modules:
            mod = types.ModuleType(name)
            mod.__path__ = [path]
            sys.modules[name] = mod

    def _load_module(name, filepath):
        spec = importlib.util.spec_from_file_location(name, filepath)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
        return mod

    gdn_mod = _load_module(
        "rtp_llm.models_py.triton_kernels.fla.gdn_gating",
        os.path.join(fla_dir, "gdn_gating.py"),
    )
    return gdn_mod


_gdn_mod = _load_rtp_llm_fla_modules()
fused_gdn_gating = _gdn_mod.fused_gdn_gating


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------

def _load_pt(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"golden file not found: {path}")
    return torch.load(path, map_location="cpu", weights_only=False)


def _restore_strided_tensor(saved_data, meta):
    """Restore a tensor's original (possibly non-contiguous) stride."""
    if not meta or meta.get("contiguous", True):
        return saved_data
    dtype = getattr(torch, meta["dtype"].replace("torch.", ""))
    tensor = torch.empty_strided(tuple(meta["shape"]), tuple(meta["stride"]), dtype=dtype)
    tensor.copy_(saved_data)
    return tensor


@unittest.skipIf(not torch.npu.is_available(), "NPU is not available")
class TestFusedGdnGatingGpuGolden(unittest.TestCase):
    """Compare Triton for Ascend fused_gdn_gating against GPU golden data."""

    rtol = 5e-2
    atol = 5e-2

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
        path = os.path.join(_SAMPLE_ROOT, "fused_gdn_gating", filename)
        data = _load_pt(path)
        inputs = data["inputs"]
        meta = data.get("input_meta", {})

        # Restore original (possibly non-contiguous) strides of a/b from the dump.
        A_log = _restore_strided_tensor(inputs["A_log"], meta.get("A_log", {}))
        a = _restore_strided_tensor(inputs["a"], meta.get("a", {}))
        b = _restore_strided_tensor(inputs["b"], meta.get("b", {}))
        dt_bias = _restore_strided_tensor(inputs["dt_bias"], meta.get("dt_bias", {}))

        # GPU outputs: g (float32) then beta (float16), both (1, S, H).
        g_expected = data["outputs"][0]
        beta_expected = data["outputs"][1]

        # Sanity-check the loaded golden tensors before execution.
        self.assertEqual(tuple(g_expected.shape), (1,) + tuple(a.shape))
        self.assertEqual(tuple(beta_expected.shape), (1,) + tuple(b.shape))
        self.assertEqual(g_expected.dtype, torch.float32)
        self.assertEqual(beta_expected.dtype, torch.float16)

        # Golden self-consistency: a CPU reference (default beta=1.0,
        # threshold=20.0) must reproduce g and beta, guarding against a
        # corrupted golden dump being silently compared against.
        beta, threshold = 1.0, 20.0
        x_ref = a.float() + dt_bias.float()
        softplus_x = torch.where(
            beta * x_ref <= threshold,
            (1 / beta) * torch.log1p(torch.exp(beta * x_ref)),
            x_ref,
        )
        g_ref = -A_log.float().exp() * softplus_x
        beta_ref = torch.sigmoid(b.float())
        self.assertTrue(
            torch.allclose(g_ref, g_expected.float(), rtol=1e-2, atol=1e-2),
            msg="CPU g_ref mismatch: "
            f"max_abs_diff={(g_ref - g_expected.float()).abs().max().item():.6f}",
        )
        self.assertTrue(
            torch.allclose(beta_ref.half(), beta_expected, rtol=1e-2, atol=1e-2),
            msg="CPU beta_ref mismatch: "
            f"max_abs_diff={(beta_ref.half().float() - beta_expected.float()).abs().max().item():.6f}",
        )

        # Non-contiguity guard: prefill a/b must remain non-contiguous with the
        # original cache-buffer stride after restore (decode S=1 is contiguous).
        if a.shape[0] > 1:
            self.assertFalse(a.is_contiguous(), "prefill a/b should be non-contiguous")
            self.assertFalse(b.is_contiguous(), "prefill a/b should be non-contiguous")
            self.assertEqual(tuple(a.stride()), (64, 1), "a stride should match GPU input_meta")
            self.assertEqual(tuple(b.stride()), (64, 1), "b stride should match GPU input_meta")

        # rtp-llm kernel supports non-contiguous a/b via stride_ab, so the
        # restored views can be passed directly (no contiguous() needed).
        g, beta = fused_gdn_gating(
            A_log=A_log.npu(),
            a=a.npu(),
            b=b.npu(),
            dt_bias=dt_bias.npu(),
        )
        torch.npu.synchronize()

        self.assertTensorClose(g, g_expected)
        self.assertTensorClose(beta, beta_expected)

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
