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
Test recurrent_gated_delta_rule NPU operator against GPU-collected golden data.

Loads .pt files dumped from GPU runs and compares NPU execution results
against the GPU reference output.
"""

import os
import unittest

import torch

from fla_npu.ops import ascendc as ascendc_ops

torch.npu.set_device(int(os.environ.get("TEST_DEVICE_ID", 0)))

# Path to GPU golden data
_DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "sample", "fused_recurrent_gated_delta_rule")
_DATA_DIR = os.path.abspath(_DATA_DIR)


def _restore_strided_tensor(saved_data: torch.Tensor, meta: dict) -> torch.Tensor:
    """Restore a tensor's original (possibly non-contiguous) stride."""
    if meta.get("contiguous", True):
        return saved_data
    dtype = getattr(torch, meta["dtype"].replace("torch.", ""))
    tensor = torch.empty_strided(meta["shape"], meta["stride"], dtype=dtype)
    tensor.copy_(saved_data)
    return tensor


def _load_gpu_case(filename: str) -> dict:
    """Load a .pt file containing GPU golden data for decode mode.

    GPU layout (rtp-llm):
      q:    (batch, seqlen, nk, dk)  float16
      k:    (batch, seqlen, nk, dk)  float16
      v:    (batch, seqlen, nv, dv)  float16
      beta: (batch, seqlen, nv)      float16
      g:    (batch, seqlen, nv)      float32
      initial_state: (num_pages, nv, dk, dv)  bfloat16  [non-contiguous]

    NPU layout (matching test_accuracy.py):
      query:  (t, nk, dk)  bfloat16
      key:    (t, nk, dk)  bfloat16
      value:  (t, nv, dv)  bfloat16
      beta:   (t, nv)      bfloat16
      g:      (t, nv)      float32
      state:  (num_states, nv, dv, dk)  bfloat16
    """
    path = os.path.join(_DATA_DIR, filename)
    data = torch.load(path, map_location="cpu", weights_only=False)

    inputs = data["inputs"]
    meta = data.get("input_meta", {})

    # Restore initial_state with original non-contiguous stride.
    state_saved = inputs["initial_state"]
    state_meta = meta.get("initial_state", {})
    if state_meta:
        state_restored = _restore_strided_tensor(state_saved, state_meta)
    else:
        state_restored = state_saved

    # GPU q/k/v are (batch, seqlen, n, d) float16 → NPU (t, n, d) bfloat16
    # GPU kernel applies L2 norm internally (use_qk_l2norm_in_kernel=True),
    # so the saved q/k are raw. Normalize here to match NPU kernel expectation.
    q_npu = inputs["q"].squeeze(0).squeeze(0).to(torch.float32)    # (16, 128)
    k_npu = inputs["k"].squeeze(0).squeeze(0).to(torch.float32)
    q_npu = torch.nn.functional.normalize(q_npu, p=2, dim=-1).to(torch.bfloat16)
    k_npu = torch.nn.functional.normalize(k_npu, p=2, dim=-1).to(torch.bfloat16)
    v_npu = inputs["v"].squeeze(0).squeeze(0).to(torch.bfloat16)    # (32, 128)
    beta_npu = inputs["beta"].squeeze(0).squeeze(0).to(torch.bfloat16)  # (32,)
    g_npu = inputs["g"].squeeze(0).squeeze(0)                          # (32,) float32

    # Ensure 3D/2D shape: (t, ...) where t = 1
    if q_npu.dim() == 2:
        q_npu = q_npu.unsqueeze(0)     # (1, 16, 128)
    if k_npu.dim() == 2:
        k_npu = k_npu.unsqueeze(0)
    if v_npu.dim() == 2:
        v_npu = v_npu.unsqueeze(0)     # (1, 32, 128)
    if beta_npu.dim() == 1:
        beta_npu = beta_npu.unsqueeze(0)  # (1, 32)
    if g_npu.dim() == 1:
        g_npu = g_npu.unsqueeze(0)        # (1, 32)

    # block_map maps batch → physical state page.
    block_map = inputs["block_map"]
    page_idx = int(block_map[0, 0].item())

    # actual_seq_lengths: [star_idx, batch0_len, ...]
    # star_idx=0 means new tokens start at index 0 (no prefix in q/k/v).
    actual_seq_lengths = torch.tensor([0, 1], dtype=torch.int32)

    # ssm_state_indices: maps each token to a state slot.
    # Use index 0 since we extract only the needed page.
    t = int(actual_seq_lengths.sum().item())
    ssm_state_indices = torch.tensor([0] * t, dtype=torch.int32)

    # scale: GPU may store None; default is dk ** -0.5
    dk = q_npu.shape[-1]
    scale = inputs.get("scale")
    if scale is None:
        scale = float(dk ** -0.5)

    # Extract the needed page and convert to bfloat16 for NPU.
    # GPU state layout is (nv, dk, dv); NPU expects (nv, dv, dk), so transpose.
    # The full paged state (293 pages) is ~461MB in float32, which can
    # cause device memory issues. We extract only the active page.
    # Non-contiguity is verified separately on the full restored tensor.
    state_page = state_restored[page_idx].transpose(-1, -2).contiguous().to(torch.bfloat16)  # (32, 128, 128)
    state_npu = state_page.unsqueeze(0)                       # (1, 32, 128, 128)

    # GPU outputs: list of [out, final_state]
    out_expected = data["outputs"][0]        # (1, 1, 32, 128) float16
    out_expected_npu = out_expected.squeeze(0).squeeze(0)  # (32, 128)
    if out_expected_npu.dim() == 2:
        out_expected_npu = out_expected_npu.unsqueeze(0)    # (1, 32, 128)

    # Expected final_state after in-place update
    state_expected = data["inplace_outputs"]["initial_state"]  # (293, 32, 128, 128) bfloat16

    return {
        "q": q_npu,
        "k": k_npu,
        "v": v_npu,
        "state": state_npu,                   # (1, 32, 128, 128) bfloat16
        "state_restored": state_restored,     # full paged, non-contiguous (for verification)
        "beta": beta_npu,
        "g": g_npu,
        "scale": scale,
        "actual_seq_lengths": actual_seq_lengths,
        "ssm_state_indices": ssm_state_indices,
        "page_idx": page_idx,
        "out_expected": out_expected_npu,
        "state_expected": state_expected,
    }


@unittest.skipIf(not torch.npu.is_available(), "NPU is not available")
class TestRecurrentGatedDeltaRuleGpuGolden(unittest.TestCase):
    """Compare NPU recurrent_gated_delta_rule output against GPU golden data."""

    rtol = 5e-2
    atol = 5e-2

    def call_op(self, q, k, v, state, **kwargs):
        return ascendc_ops.npu_recurrent_gated_delta_rule(q, k, v, state, **kwargs)

    def assertTensorClose(self, actual: torch.Tensor, expected: torch.Tensor, *, rtol=None, atol=None):
        rtol = self.rtol if rtol is None else rtol
        atol = self.atol if atol is None else atol
        actual_cpu = actual.detach().cpu().float()
        expected_cpu = expected.detach().cpu().float()
        self.assertTrue(
            torch.allclose(actual_cpu, expected_cpu, rtol=rtol, atol=atol),
            msg=f"max_abs_diff={(actual_cpu - expected_cpu).abs().max().item():.6f}",
        )

    # ------------------------------------------------------------------
    # Case 1: decode
    #   Single-sequence decode with paged initial_state.
    #   q/k/v: (1, 1, n, d) float16 → NPU (1, n, d) bfloat16
    #   initial_state: (293, 32, 128, 128) bfloat16, non-contiguous
    #   (stride=(1048576, 16384, 128, 1), channel-last on dim axis).
    #   Non-contiguity is verified on the full restored paged tensor.
    #   Only the active page is passed to NPU to avoid ~461MB device memory.
    # ------------------------------------------------------------------
    def test_decode(self):
        case = _load_gpu_case("decode.pt")

        # Verify state was restored as non-contiguous (matching GPU's paged layout)
        self.assertFalse(case["state_restored"].is_contiguous(),
                         "state should be non-contiguous after stride restoration")

        # Move inputs to NPU; state is mutated in-place so keep device reference
        state_device = case["state"].npu()

        # Call NPU operator (matching test_accuracy.py calling convention)
        out, _ = self.call_op(
            case["q"].npu(),
            case["k"].npu(),
            case["v"].npu(),
            state_device,
            beta=case["beta"].npu(),
            scale=case["scale"],
            actual_seq_lengths=case["actual_seq_lengths"].npu(),
            ssm_state_indices=case["ssm_state_indices"].npu(),
            g=case["g"].npu(),
        )

        self.assertTensorClose(out, case["out_expected"])

        # Compare the updated state page.
        # NPU state is (nv, dv, dk); GPU expected is (nv, dk, dv), so transpose back.
        page_idx = case["page_idx"]
        state_actual = state_device.cpu()[0].transpose(-1, -2)  # (nv, dk, dv)
        state_expected_page = case["state_expected"][page_idx].to(torch.float32)
        self.assertTensorClose(state_actual, state_expected_page)


if __name__ == "__main__":
    unittest.main()
