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
Test block.py operators (load_initial_state_from_block_map, store_ssm_state_to_block_map)
on NPU via Triton for Ascend, against GPU-collected golden data.

- load_initial_state_from_block_map: directly reused from rtp-llm (kernel works as-is on NPU).
- store_ssm_state_to_block_map: rtp-llm kernel has a pointer reassignment pattern that
  Triton for Ascend does not support ("ptr type from different source not supported").
  The kernel is inlined with a tl.where fix; the Python wrapper is reused from rtp-llm
  via importlib (bypassing rtp_llm.__init__ heavy C++ dependencies).

GPU source: rtp-llm/rtp_llm/models_py/triton_kernels/fla/block.py
"""

import importlib.util
import os
import sys
import types
import unittest

import torch
import triton
import triton.language as tl

torch.npu.set_device(int(os.environ.get("TEST_DEVICE_ID", 0)))

_RTP_LLM_ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
_RTP_LLM_ROOT = os.path.abspath(_RTP_LLM_ROOT)

_SAMPLE_ROOT = os.path.join(os.path.dirname(__file__), "..", "..", "..", "sample")
_SAMPLE_ROOT = os.path.abspath(_SAMPLE_ROOT)


# ---------------------------------------------------------------------------
# Bootstrap: load rtp-llm fla modules without triggering rtp_llm.__init__
# (which requires libth_transformer_config.so). We create stub packages and
# load the specific .py files via importlib.
# ---------------------------------------------------------------------------

def _load_rtp_llm_fla_modules():
    fla_dir = os.path.join(_RTP_LLM_ROOT, "rtp_llm", "models_py", "triton_kernels", "fla")

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

    _load_module("rtp_llm.models_py.triton_kernels.fla.utils",
                 os.path.join(fla_dir, "utils.py"))
    _load_module("rtp_llm.models_py.triton_kernels.fla.index",
                 os.path.join(fla_dir, "index.py"))
    block_mod = _load_module("rtp_llm.models_py.triton_kernels.fla.block",
                             os.path.join(fla_dir, "block.py"))
    return block_mod


_block_mod = _load_rtp_llm_fla_modules()

# Directly reuse load_initial_state_from_block_map from rtp-llm
load_initial_state_from_block_map = _block_mod.load_initial_state_from_block_map

# store_ssm_state_to_block_map from rtp-llm cannot be directly reused because
# its kernel uses pointer reassignment across different base tensors.
# We reuse the Python wrapper logic but provide a fixed kernel below.
_prepare_chunk_indices = _block_mod.store_ssm_state_to_block_map  # not used directly


# ---------------------------------------------------------------------------
# Fixed store_ssm_state_to_block_map kernel for Triton for Ascend.
# Original GPU kernel reassigns source_ptr between final_states and h (different
# base tensors), which Triton for Ascend rejects. Fix: load from both sources,
# select via tl.where.
# ---------------------------------------------------------------------------

@triton.jit(do_not_specialize=["max_block_size"])
def _store_ssm_state_to_block_map_kernel_ascend(
    chunk_indices, h, final_states, prefix_lengths, cu_seqlens,
    block_map, ssm_states,
    max_block_size,
    HEAD_NUM: tl.constexpr, V: tl.constexpr, K: tl.constexpr,
    BLOCK_V: tl.constexpr, SEQ_SIZE_PER_BLOCK: tl.constexpr,
    CHUNK_SIZE: tl.constexpr, CONV_STRIDE_TOKEN: tl.constexpr,
):
    i_c, i_h, i_v = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    batch = tl.load(chunk_indices + i_c * 2).to(tl.int32)
    chunk = tl.load(chunk_indices + i_c * 2 + 1).to(tl.int32)
    SSM_PER_HEAD = K * V
    SSM_PER_BATCH = SSM_PER_HEAD * HEAD_NUM
    v_offset = i_v * BLOCK_V
    prefix = tl.load(prefix_lengths + batch)
    bos = tl.load(cu_seqlens + batch).to(tl.int32)
    eos = tl.load(cu_seqlens + batch + 1).to(tl.int32)
    input_len = eos - bos

    is_last_chunk = (chunk + 1) * CHUNK_SIZE >= input_len
    is_boundary = (chunk > 0) and ((chunk + 1) * CHUNK_SIZE % SEQ_SIZE_PER_BLOCK == 0)
    should_write = is_last_chunk or is_boundary

    if not should_write:
        return

    if is_last_chunk:
        dest_block_pos = (prefix + input_len - 1) // SEQ_SIZE_PER_BLOCK
    else:
        dest_block_pos = (prefix + chunk * CHUNK_SIZE + CHUNK_SIZE - 1) // SEQ_SIZE_PER_BLOCK

    block_idx = tl.load(block_map + batch * max_block_size + dest_block_pos).to(tl.int64)
    if block_idx <= 0:
        return

    dest_ptr = ssm_states + block_idx * CONV_STRIDE_TOKEN + i_h * SSM_PER_HEAD

    # Load from both sources, select via tl.where (Triton for Ascend cannot
    # reassign a pointer across different base tensors).
    p_final = tl.make_block_ptr(
        final_states + batch * SSM_PER_BATCH + i_h * SSM_PER_HEAD,
        (V, K), (K, 1), (v_offset, 0), (BLOCK_V, K), (1, 0),
    )
    p_h = tl.make_block_ptr(
        h + (i_c + 1) * SSM_PER_BATCH + i_h * SSM_PER_HEAD,
        (V, K), (K, 1), (v_offset, 0), (BLOCK_V, K), (1, 0),
    )

    b_final = tl.load(p_final, boundary_check=(0, 1))
    b_h = tl.load(p_h, boundary_check=(0, 1))
    b_src = tl.where(is_last_chunk, b_final, b_h)

    p_out = tl.make_block_ptr(
        dest_ptr, (V, K), (K, 1), (v_offset, 0), (BLOCK_V, K), (1, 0),
    )
    tl.store(p_out, b_src.to(ssm_states.dtype.element_ty), boundary_check=(0, 1))


def store_ssm_state_to_block_map(
    h, final_states, prefix_lengths, cu_seqlens, block_map, ssm_states,
    seq_size_per_block, chunk_size, block_v=64,
):
    """NPU-adapted version: uses fixed kernel with tl.where instead of pointer reassignment."""
    assert h.dtype == torch.float32 and final_states.dtype == torch.float32
    # Reuse prepare_chunk_indices from rtp-llm
    from rtp_llm.models_py.triton_kernels.fla.index import prepare_chunk_indices
    chunk_indices = prepare_chunk_indices(cu_seqlens, chunk_size)
    _, head_num, v, k = ssm_states.shape
    chunk_num = chunk_indices.shape[0]
    max_block_size = block_map.shape[1]
    grid = (chunk_num, head_num, triton.cdiv(v, block_v))
    token_stride_ssm_state = ssm_states.stride(0)
    _store_ssm_state_to_block_map_kernel_ascend[grid](
        chunk_indices, h, final_states, prefix_lengths, cu_seqlens,
        block_map, ssm_states,
        max_block_size,
        HEAD_NUM=head_num, V=v, K=k,
        BLOCK_V=block_v,
        SEQ_SIZE_PER_BLOCK=seq_size_per_block,
        CONV_STRIDE_TOKEN=token_stride_ssm_state,
        CHUNK_SIZE=chunk_size,
    )


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------

def _load_pt(path):
    return torch.load(path, map_location="cpu", weights_only=False)


@unittest.skipIf(not torch.npu.is_available(), "NPU is not available")
class TestBlockOpsGpuGolden(unittest.TestCase):
    """Compare Triton for Ascend block ops against GPU golden data."""

    rtol = 5e-2
    atol = 5e-2

    def assertTensorClose(self, actual, expected, *, rtol=None, atol=None):
        rtol = self.rtol if rtol is None else rtol
        atol = self.atol if atol is None else atol
        actual_cpu = actual.detach().cpu().float()
        expected_cpu = expected.detach().cpu().float()
        self.assertTrue(
            torch.allclose(actual_cpu, expected_cpu, rtol=rtol, atol=atol),
            msg=f"max_abs_diff={(actual_cpu - expected_cpu).abs().max().item():.6f}",
        )

    def test_load_initial_state_from_block_map(self):
        """load_initial_state_from_block_map: directly reused from rtp-llm."""
        path = os.path.join(_SAMPLE_ROOT, "load_initial_state_from_block_map", "prefill.pt")
        data = _load_pt(path)
        inputs = data["inputs"]

        prefix_lengths = inputs["prefix_lengths"].npu()
        block_map = inputs["block_map"].npu()
        conv_states = inputs["conv_states"].npu().contiguous()
        initial_states = inputs["initial_states"].npu().contiguous().clone()
        seq_size_per_block = inputs["seq_size_per_block"]

        load_initial_state_from_block_map(
            prefix_lengths, block_map, conv_states, initial_states, seq_size_per_block)
        torch.npu.synchronize()

        expected = data["inplace_outputs"]["initial_states"]
        self.assertTensorClose(initial_states, expected)

    def test_store_ssm_state_to_block_map(self):
        """store_ssm_state_to_block_map: kernel adapted for Triton for Ascend."""
        path = os.path.join(_SAMPLE_ROOT, "store_ssm_state_to_block_map", "prefill.pt")
        data = _load_pt(path)
        inputs = data["inputs"]

        h = inputs["h"].npu().contiguous()
        final_states = inputs["final_states"].npu().contiguous()
        prefix_lengths = inputs["prefix_lengths"].npu()
        cu_seqlens = inputs["cu_seqlens"].to(torch.int64).npu()
        block_map = inputs["block_map"].npu()
        ssm_states = inputs["ssm_states"].npu().contiguous().clone()
        seq_size_per_block = inputs["seq_size_per_block"]
        chunk_size = inputs["chunk_size"]

        store_ssm_state_to_block_map(
            h, final_states, prefix_lengths, cu_seqlens, block_map, ssm_states,
            seq_size_per_block, chunk_size)
        torch.npu.synchronize()

        expected = data["inplace_outputs"]["ssm_states"]
        self.assertTensorClose(ssm_states, expected)


if __name__ == "__main__":
    unittest.main()
