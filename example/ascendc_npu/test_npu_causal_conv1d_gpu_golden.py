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


def _restore_strided_tensor(saved_data: torch.Tensor, meta: dict) -> torch.Tensor:
    """Restore a tensor's original (possibly non-contiguous) stride.

    ``.pt`` files store tensors in contiguous form, losing the original
    stride information.  The GPU dump also saves ``input_meta`` which
    records the original shape / stride / dtype / contiguous flag.

    If the original tensor was contiguous, return ``saved_data`` directly.
    Otherwise allocate a strided tensor and copy the data in.
    """
    if meta.get("contiguous", True):
        return saved_data
    dtype = getattr(torch, meta["dtype"].replace("torch.", ""))
    tensor = torch.empty_strided(meta["shape"], meta["stride"], dtype=dtype)
    tensor.copy_(saved_data)
    return tensor


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


def _load_gpu_case(filename: str) -> dict:
    """Load a .pt file containing GPU golden inputs/outputs.

    Restores the original (non-contiguous) stride of x and conv_states
    using the ``input_meta`` saved alongside the data.
    """
    path = os.path.join(_DATA_DIR, filename)
    data = torch.load(path, map_location="cpu", weights_only=False)

    inputs = data["inputs"]
    meta = data.get("input_meta", {})

    # Restore original stride for x (GPU layout: (dim, seqlen), channel-last)
    x_saved = inputs["x"]
    x_meta = meta.get("x", {})
    x_gpu = _restore_strided_tensor(x_saved, x_meta) if x_meta else x_saved.contiguous()

    # Restore original stride for conv_states if present (GPU paged layout:
    # (num_pages, dim, state_len), channel-last with dim-axis stride=1)
    conv_states_saved = inputs.get("conv_states")
    cs_meta = meta.get("conv_states", {})
    if conv_states_saved is not None and cs_meta:
        conv_states = _restore_strided_tensor(conv_states_saved, cs_meta)
    else:
        conv_states = conv_states_saved

    return {
        "x": x_gpu,
        "weight": inputs["weight"].contiguous(),
        "bias": inputs.get("bias"),
        "conv_states": conv_states,
        "query_start_loc": inputs["query_start_loc"],
        "y_expected": data["outputs"].contiguous(),
    }


def _load_gpu_update_case(filename: str) -> dict:
    """Load a .pt file containing GPU golden data for update/decode mode.

    Restores the original (non-contiguous) stride of conv_state using the
    ``input_meta`` saved alongside the data.  The GPU data uses a nested
    ``inputs``/``outputs``/``inplace_outputs`` structure with paged
    ``conv_state``.
    """
    path = os.path.join(_DATA_DIR_UPDATE, filename)
    data = torch.load(path, map_location="cpu", weights_only=False)

    inputs = data["inputs"]
    meta = data.get("input_meta", {})

    x_gpu = inputs["x"].contiguous()
    weight_gpu = inputs["weight"].contiguous()
    y_gpu = data["outputs"].contiguous()

    dim, width = weight_gpu.shape  # GPU: (D, W)

    # Restore the full paged conv_state with original stride.
    # GPU layout: (num_pages, dim, state_len) stride=(1048576, 1, 8192)
    # — channel-last with dim-axis stride=1.
    conv_state_saved = inputs["conv_state"]
    cs_meta = meta.get("conv_state", {})
    if cs_meta:
        conv_state_restored = _restore_strided_tensor(conv_state_saved, cs_meta)
    else:
        conv_state_restored = conv_state_saved

    block_map = inputs["block_map"]
    page_idx = int(block_map[0, 0].item())

    # Transpose the full paged conv_state to NPU format
    # (num_pages, state_len, dim), preserving the non-contiguous stride.
    # GPU: (293, 8192, 3) stride=(1048576, 1, 8192)
    # After .transpose(1,2): (293, 3, 8192) stride=(1048576, 8192, 1)
    # — non-contiguous, dim axis (last) stride=1.
    conv_states_npu = conv_state_restored.transpose(1, 2)   # (num_pages, state_len, dim)

    # Expected conv_state after in-place update (GPU layout (dim, state_len)).
    # Extract the relevant page and transpose to NPU format for comparison.
    conv_states_expected = data["inplace_outputs"]["conv_state"][page_idx].contiguous()
    conv_states_expected_npu = conv_states_expected.T.contiguous().unsqueeze(0)

    return {
        "x_npu": x_gpu.squeeze(-1).contiguous(),                 # (1, dim)
        "weight_npu": _gpu_weight_to_npu(weight_gpu),            # (width, dim)
        "bias": None,
        "conv_states_npu": conv_states_npu,                      # (num_pages, state_len, dim) non-contiguous
        "page_idx": page_idx,                                    # cache index for this sequence
        "conv_states_expected": conv_states_expected_npu,        # (1, state_len, dim) after in-place update
        "y_expected_npu": y_gpu.squeeze(-1).contiguous(),        # (1, dim)
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
    #   x is non-contiguous on GPU (channel-last, stride=(1, 12288)).
    #   NPU AutoContiguous handles x; conv_states is zero-initialized.
    # ------------------------------------------------------------------
    def test_prefill_first_seq2047(self):
        case = _load_gpu_case("prefill_first_seq2047.pt")

        x_gpu = case["x"]           # (8192, 2047) — non-contiguous, stride=(1, 12288)
        weight_gpu = case["weight"] # (8192, 4)    float16
        qsl = case["query_start_loc"].to(torch.int64)  # [0, 2047]

        # Verify x is restored as non-contiguous (channel-last on GPU)
        self.assertFalse(x_gpu.is_contiguous(), "x should be non-contiguous after stride restoration")
        self.assertEqual(x_gpu.stride(), (1, 12288), "x stride should match GPU input_meta")

        dim, width = weight_gpu.shape
        batch = int(qsl.shape[0] - 1)

        # Convert to NPU format — .T preserves the channel-last stride
        x_npu = x_gpu.T              # (2047, 8192) stride=(12288, 1) — non-contiguous
        self.assertFalse(x_npu.is_contiguous(), "x_npu should be non-contiguous")
        weight_npu = _gpu_weight_to_npu(weight_gpu)
        conv_states_npu = _make_zero_conv_states(batch, width, dim, dtype=x_gpu.dtype)

        y_expected_npu = _gpu_out_to_npu(case["y_expected"])

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
    #   x is non-contiguous (stride=(1, 12288)); conv_states is non-contiguous
    #   (stride=(1048576, 1, 8192), channel-last on dim axis).
    # ------------------------------------------------------------------
    def test_prefill_incr_seq32(self):
        case = _load_gpu_case("prefill_incr_seq32.pt")

        x_gpu = case["x"]           # (8192, 32) — non-contiguous, stride=(1, 12288)
        weight_gpu = case["weight"]
        qsl = case["query_start_loc"].to(torch.int64)

        # Verify x is restored as non-contiguous (channel-last on GPU)
        self.assertFalse(x_gpu.is_contiguous(), "x should be non-contiguous after stride restoration")
        self.assertEqual(x_gpu.stride(), (1, 12288), "x stride should match GPU input_meta")

        dim, width = weight_gpu.shape
        batch = int(qsl.shape[0] - 1)

        x_npu = x_gpu.T              # (32, 8192) — non-contiguous
        self.assertFalse(x_npu.is_contiguous(), "x_npu should be non-contiguous")
        weight_npu = _gpu_weight_to_npu(weight_gpu)
        conv_states_npu = _make_zero_conv_states(batch, width, dim, dtype=x_gpu.dtype)

        y_expected_npu = _gpu_out_to_npu(case["y_expected"])

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
    #   conv_state is paged (293, 8192, 3) non-contiguous
    #   (stride=(1048576, 1, 8192), channel-last on dim axis).
    #   The extracted page is transposed to NPU format (1, 3, 8192)
    #   preserving the non-contiguous stride with dim-axis stride=1.
    # ------------------------------------------------------------------
    def test_decode_update(self):
        case = _load_gpu_update_case("decode.pt")

        x_npu = case["x_npu"]
        weight_npu = case["weight_npu"]
        conv_states_npu = case["conv_states_npu"]        # full paged, non-contiguous
        page_idx = case["page_idx"]
        y_expected_npu = case["y_expected_npu"]
        conv_states_expected = case["conv_states_expected"]

        # Verify conv_states is non-contiguous with dim-axis stride=1
        # (matching GPU's channel-last layout for conv_state)
        self.assertFalse(conv_states_npu.is_contiguous(),
                         "conv_states should be non-contiguous after stride restoration")
        self.assertEqual(conv_states_npu.stride(-1), 1,
                         "conv_states dim axis (last) must have stride=1 for NPU DataCopy")

        conv_states_device = conv_states_npu.npu()

        y = self.call_op(
            x=x_npu.npu(),
            weight=weight_npu.npu(),
            bias=None,
            conv_states=conv_states_device,
            cache_indices=[page_idx],
            activation_mode=case["activation_mode"],
            pad_slot_id=-1,
            run_mode=1,
            head_num=0,
        )

        self.assertTensorClose(y, y_expected_npu)
        # Extract the updated page from the full paged tensor for comparison
        conv_states_actual = conv_states_device.cpu()[page_idx].unsqueeze(0)
        self.assertTensorClose(conv_states_actual, conv_states_expected)


if __name__ == "__main__":
    unittest.main()
