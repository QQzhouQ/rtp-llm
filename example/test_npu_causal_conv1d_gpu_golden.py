# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Tianjin University, Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""
Test causal_conv1d NPU operator against GPU-collected golden data.

Loads .pt files dumped from GPU runs and compares NPU execution results
against the GPU reference output.
"""

import os
import unittest

import torch

from fla_npu.ops import ascendc as ascendc_ops

torch.npu.set_device(int(os.environ.get("TEST_DEVICE_ID", 0)))

# Path to the GPU-dumped golden data
_DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "causal_conv1d_fn")
_DATA_DIR = os.path.abspath(_DATA_DIR)

# Path to GPU update/decode mode golden data
_DATA_DIR_UPDATE = os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "sample", "causal_conv1d_update")
_DATA_DIR_UPDATE = os.path.abspath(_DATA_DIR_UPDATE)


def _load_gpu_case(filename: str) -> dict:
    """Load a .pt file containing GPU golden inputs/outputs."""
    path = os.path.join(_DATA_DIR, filename)
    data = torch.load(path, map_location="cpu", weights_only=False)
    return {
        "x": data["inputs"]["x"].contiguous(),
        "weight": data["inputs"]["weight"].contiguous(),
        "bias": data["inputs"].get("bias"),
        "conv_states": data["inputs"].get("conv_states"),
        "query_start_loc": data["inputs"]["query_start_loc"],
        "y_expected": data["outputs"].contiguous(),
    }


def _gpu_x_to_npu_varlen(x_gpu: torch.Tensor) -> torch.Tensor:
    """Convert GPU x layout (D, S) to NPU varlen layout (cu_seqlen, dim)."""
    return x_gpu.contiguous().T.contiguous()


def _gpu_weight_to_npu(weight_gpu: torch.Tensor) -> torch.Tensor:
    """Convert GPU weight (D, W) to NPU format (W, D).

    NPU operator expects weight as (width, dim).
    """
    return weight_gpu.contiguous().T.contiguous()


def _gpu_out_to_npu(y_expected_gpu: torch.Tensor) -> torch.Tensor:
    """GPU output is (D, S), transpose to NPU varlen format (cu_seqlen, dim)."""
    return y_expected_gpu.T.contiguous()


def _make_zero_conv_states(batch: int, width: int, dim: int, dtype: torch.dtype, device: str = "cpu") -> torch.Tensor:
    """Create zero-initialized conv_states of shape (batch, width - 1, dim).

    NPU operator expects conv_states as (num_cache_lines, state_len, dim),
    where state_len >= width - 1.
    """
    return torch.zeros(batch, width - 1, dim, dtype=dtype, device=device)


def _load_gpu_update_case(filename: str) -> dict:
    """Load a .pt file containing GPU golden data for update/decode mode.

    The GPU data uses a nested ``inputs``/``outputs``/``inplace_outputs``
    structure with paged ``conv_state``. Returns a flat dict in the form
    consumed by the test methods.
    """
    path = os.path.join(_DATA_DIR_UPDATE, filename)
    data = torch.load(path, map_location="cpu", weights_only=False)

    inputs = data["inputs"]

    # x: GPU shape (1, dim, 1) → NPU decode expects (batch, dim)
    x_gpu = inputs["x"].contiguous()
    # weight: GPU shape (dim, width) → NPU expects (width, dim)
    weight_gpu = inputs["weight"].contiguous()
    # y: GPU shape (1, dim, 1) → NPU decode output is (1, dim)
    y_gpu = data["outputs"].contiguous()

    dim, width = weight_gpu.shape  # GPU: (D, W)

    # Extract the relevant cache line from GPU paged conv_state.
    # The GPU stores conv_state in paged layout (num_pages, dim, state_len).
    # ``block_map`` maps the sequence to a physical page.
    conv_state_page = inputs["conv_state"]
    block_map = inputs["block_map"]
    page_idx = int(block_map[0, 0].item())

    # Extracted page is (dim, state_len=3); transpose to NPU format
    # (num_cache_lines=1, state_len=3, dim).
    page_state = conv_state_page[page_idx].contiguous()           # (dim, state_len)
    conv_states_npu = page_state.T.contiguous().unsqueeze(0)      # (1, state_len, dim)

    # NPU conv state is mutated in-place; clone for expected comparison.
    conv_states_expected = data["inplace_outputs"]["conv_state"][page_idx].contiguous()
    conv_states_expected_npu = conv_states_expected.T.contiguous().unsqueeze(0)

    return {
        "x_npu": x_gpu.squeeze(-1).contiguous(),                 # (1, dim)
        "weight_npu": _gpu_weight_to_npu(weight_gpu),            # (width, dim)
        "bias": None,
        "conv_states_npu": conv_states_npu,                      # (1, state_len, dim)
        "conv_states_expected": conv_states_expected_npu,        # after in-place update
        "y_expected_npu": y_gpu.squeeze(-1).contiguous(),        # (1, dim)
        # activation is read from GPU metadata
        "activation_mode": 1 if inputs.get("activation") in ("silu", "swish") else 0,
    }


@unittest.skipIf(not torch.npu.is_available(), "NPU is not available")
class TestCausalConv1dGpuGolden(unittest.TestCase):
    """Compare NPU causal_conv1d output against GPU golden data."""

    rtol = 5e-2
    atol = 5e-2

    def call_op(self, **kwargs):
        return ascendc_ops.npu_causal_conv1d(**kwargs)

    def assertTensorClose(self, actual: torch.Tensor, expected: torch.Tensor, *, rtol=None, atol=None):
        rtol = self.rtol if rtol is None else rtol
        atol = self.atol if atol is None else atol
        actual_cpu = actual.detach().cpu().float()
        expected_cpu = expected.detach().cpu().float()
        self.assertTrue(
            torch.allclose(actual_cpu, expected_cpu, rtol=rtol, atol=atol),
            msg=(
                f"max_abs_diff="
                f"{(actual_cpu - expected_cpu).abs().max().item():.6f}"
            ),
        )

    # ------------------------------------------------------------------
    # Case 1: prefill_first_seq2047
    #   Single sequence of 2047 tokens, dim=8192, width=4, no bias, no initial states.
    #   GPU activation = SiLU (activation_mode=1).
    # ------------------------------------------------------------------
    def test_prefill_first_seq2047(self):
        case = _load_gpu_case("prefill_first_seq2047.pt")

        x_gpu = case["x"]           # (8192, 2047) float16
        weight_gpu = case["weight"] # (8192, 4)    float16
        qsl = case["query_start_loc"].to(torch.int64)  # [0, 2047]

        dim, width = weight_gpu.shape
        batch = int(qsl.shape[0] - 1)

        # Convert to NPU format
        x_npu = _gpu_x_to_npu_varlen(x_gpu)          # (S=2047, D=8192)
        weight_npu = _gpu_weight_to_npu(weight_gpu)   # (W=4, D=8192)
        conv_states_npu = _make_zero_conv_states(batch, width, dim, dtype=x_gpu.dtype)

        # Expected output from GPU
        y_expected_npu = _gpu_out_to_npu(case["y_expected"])  # (2047, 8192)

        # Move to NPU
        y = self.call_op(
            x=x_npu.npu(),
            weight=weight_npu.npu(),
            bias=None,
            conv_states=conv_states_npu.npu(),
            query_start_loc=qsl.tolist(),
            activation_mode=1,
            pad_slot_id=-1,
            run_mode=0,
            head_num=0,
        )

        self.assertTensorClose(y, y_expected_npu)

    # ------------------------------------------------------------------
    # Case 2: prefill_incr_seq32
    #   Single sequence of 32 tokens, dim=8192, width=4, has paged conv_states.
    #   Since prefix_lengths=0, initial states are effectively zero.
    # ------------------------------------------------------------------
    def test_prefill_incr_seq32(self):
        case = _load_gpu_case("prefill_incr_seq32.pt")

        x_gpu = case["x"]           # (8192, 32) float16
        weight_gpu = case["weight"] # (8192, 4)  float16
        qsl = case["query_start_loc"].to(torch.int64)  # [0, 32]

        dim, width = weight_gpu.shape
        batch = int(qsl.shape[0] - 1)

        # Convert to NPU format
        x_npu = _gpu_x_to_npu_varlen(x_gpu)          # (S=32, D=8192)
        weight_npu = _gpu_weight_to_npu(weight_gpu)   # (W=4, D=8192)
        conv_states_npu = _make_zero_conv_states(batch, width, dim, dtype=x_gpu.dtype)

        # Expected output from GPU
        y_expected_npu = _gpu_out_to_npu(case["y_expected"])  # (32, 8192)

        # Move to NPU
        y = self.call_op(
            x=x_npu.npu(),
            weight=weight_npu.npu(),
            bias=None,
            conv_states=conv_states_npu.npu(),
            query_start_loc=qsl.tolist(),
            activation_mode=1,
            pad_slot_id=-1,
            run_mode=0,
            head_num=0,
        )

        self.assertTensorClose(y, y_expected_npu)

    # ------------------------------------------------------------------
    # Case 3: decode_update
    #   Single-sequence decode (update) with paged conv_state.
    #   x: (1, 8192, 1), weight: (8192, 4), activation: SiLU, no bias.
    #   conv_state is paged (293, 8192, 3); the test extracts the
    #   relevant page via block_map.
    #   GPU golden captures both the output y and the in-place-updated
    #   conv_state for full validation.
    # ------------------------------------------------------------------
    def test_decode_update(self):
        case = _load_gpu_update_case("decode.pt")

        x_npu = case["x_npu"]                            # (1, 8192)
        weight_npu = case["weight_npu"]                  # (4, 8192)
        conv_states_npu = case["conv_states_npu"]        # (1, 3, 8192)
        y_expected_npu = case["y_expected_npu"]          # (1, 8192)
        conv_states_expected = case["conv_states_expected"]  # (1, 3, 8192)

        conv_states_device = conv_states_npu.npu()

        y = self.call_op(
            x=x_npu.npu(),
            weight=weight_npu.npu(),
            bias=None,
            conv_states=conv_states_device,
            cache_indices=[0],
            activation_mode=case["activation_mode"],
            pad_slot_id=-1,
            run_mode=1,
            head_num=0,
        )

        self.assertTensorClose(y, y_expected_npu)
        self.assertTensorClose(conv_states_device.cpu(), conv_states_expected)


if __name__ == "__main__":
    unittest.main()
