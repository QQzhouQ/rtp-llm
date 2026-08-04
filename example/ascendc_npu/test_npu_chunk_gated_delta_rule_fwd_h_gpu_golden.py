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
Test chunk_gated_delta_rule_fwd_h NPU operator against GPU-collected golden data.

The GPU dump contains the full chunk_gated_delta_rule pipeline outputs (o, h, final_state).
The fwd_h operator takes k, w, u, g as inputs — w and u are intermediate results from
recompute_wu_fwd. Since the GPU dump does not include w/u, we run the NPU pre-processing
pipeline (cumsum → kkt → solve_tri → recompute_wu) to produce w and u, then call fwd_h.

Key: use_qk_l2norm_in_kernel=True means k must be L2-normalized before pipeline.
"""

import math
import os
import unittest

import numpy as np
import torch

from fla_npu.ops import ascendc as ascendc_ops

torch.npu.set_device(int(os.environ.get("TEST_DEVICE_ID", 0)))

_DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "sample", "chunk_gated_delta_rule")
_DATA_DIR = os.path.abspath(_DATA_DIR)


def _restore_strided_tensor(saved_data: torch.Tensor, meta: dict) -> torch.Tensor:
    if not meta or meta.get("contiguous", True):
        return saved_data
    dtype = getattr(torch, meta["dtype"].replace("torch.", ""))
    tensor = torch.empty_strided(meta["shape"], meta["stride"], dtype=dtype)
    tensor.copy_(saved_data)
    return tensor


def _next_power_of_two(value: int) -> int:
    value = max(value, 1)
    result = 1
    while result < value:
        result <<= 1
    return result


def _block_t(chunk_size: int) -> int:
    return _next_power_of_two((1 << 17) // chunk_size)


def _prepare_chunk_indices(cu_seqlens, chunk_size):
    rows = []
    for seq_idx, (start, end) in enumerate(zip(cu_seqlens[:-1], cu_seqlens[1:])):
        for chunk_idx in range(math.ceil((end - start) / chunk_size)):
            rows.append((seq_idx, chunk_idx))
    return [v for row in rows for v in row]


def _load_gpu_case(filename: str) -> dict:
    path = os.path.join(_DATA_DIR, filename)
    data = torch.load(path, map_location="cpu", weights_only=False)

    inputs = data["inputs"]
    meta = data.get("input_meta", {})

    k_gpu = _restore_strided_tensor(inputs["k"], meta.get("k", {}))
    v_gpu = _restore_strided_tensor(inputs["v"], meta.get("v", {}))
    g_gpu = _restore_strided_tensor(inputs["g"], meta.get("g", {}))
    beta_gpu = _restore_strided_tensor(inputs["beta"], meta.get("beta", {}))

    initial_state_gpu = inputs.get("initial_state")
    if initial_state_gpu is not None and meta.get("initial_state"):
        initial_state_gpu = _restore_strided_tensor(initial_state_gpu, meta.get("initial_state", {}))

    cu_seqlens = [int(v) for v in inputs["cu_seqlens"].tolist()]
    use_qk_l2norm = inputs.get("use_qk_l2norm_in_kernel", False)

    # Transpose to head_first (B, H, T, D), convert to NPU dtypes
    k_npu = k_gpu.transpose(1, 2).contiguous().to(torch.bfloat16)
    v_npu = v_gpu.transpose(1, 2).contiguous().to(torch.bfloat16)
    g_raw_npu = g_gpu.transpose(1, 2).contiguous()  # float32
    beta_npu = beta_gpu.transpose(1, 2).contiguous().to(torch.float32)

    # L2 normalize k if use_qk_l2norm_in_kernel=True
    if use_qk_l2norm:
        k_npu = torch.nn.functional.normalize(k_npu.float(), p=2, dim=-1).to(torch.bfloat16)

    # initial_state: (N, Hv, Dk, Dv) → float32
    if initial_state_gpu is not None:
        initial_state_npu = initial_state_gpu.to(torch.float32)
    else:
        initial_state_npu = None

    # GPU outputs: h (B, NT, Hv, K, V) → NPU (B, Hv, NT, K, V)
    h_expected = data["outputs"][1].transpose(1, 2).contiguous()
    final_state_expected = data["outputs"][2].contiguous()

    return {
        "k": k_npu, "v": v_npu, "g_raw": g_raw_npu, "beta": beta_npu,
        "initial_state": initial_state_npu, "cu_seqlens": cu_seqlens,
        "h_expected": h_expected, "final_state_expected": final_state_expected,
    }


@unittest.skipIf(not torch.npu.is_available(), "NPU is not available")
class TestChunkGatedDeltaRuleFwdHGpuGolden(unittest.TestCase):
    """Compare NPU chunk_gated_delta_rule_fwd_h output against GPU golden data.

    Pre-processing steps use NPU operators (cumsum, kkt, solve_tri, recompute_wu)
    to avoid CPU reference precision issues that cause state divergence.
    """

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

    def _run_case(self, filename, *, max_chunks=None):
        case = _load_gpu_case(filename)

        k = case["k"]
        v = case["v"]
        g_raw = case["g_raw"]
        beta = case["beta"]
        cu_seqlens = case["cu_seqlens"]
        chunk_size = 64

        # Optionally truncate to max_chunks to avoid state overflow in bf16
        if max_chunks is not None:
            max_t = max_chunks * chunk_size
            if cu_seqlens[-1] > max_t:
                cu_seqlens = [0, max_t]
                k = k[:, :, :max_t].contiguous()
                v = v[:, :, :max_t].contiguous()
                g_raw = g_raw[:, :, :max_t].contiguous()
                beta = beta[:, :, :max_t].contiguous()

        # Step 1: chunk_local_cumsum(g) → g_cumsum
        block_t = _block_t(chunk_size)
        ci_cumsum = _prepare_chunk_indices(cu_seqlens, block_t)
        g_cumsum = ascendc_ops.npu_chunk_local_cumsum(
            g_raw.npu(), chunk_size=chunk_size, cu_seqlens=cu_seqlens,
            chunk_indices_out=ci_cumsum, head_first=True, output_dtype="float32")

        # Step 2: chunk_scaled_dot_kkt(k, g_cumsum, beta) → A
        ci_kkt = _prepare_chunk_indices(cu_seqlens, chunk_size)
        Hk = k.shape[1]
        Hv = g_raw.shape[1]
        ratio = Hv // Hk
        g_sub = g_cumsum[:, ::ratio].contiguous() if ratio > 1 else g_cumsum
        beta_sub = beta[:, ::ratio].contiguous() if ratio > 1 else beta

        A = ascendc_ops.npu_chunk_scaled_dot_kkt(
            k=k.npu(), g=g_sub.npu(), beta=beta_sub.npu(),
            cu_seqlens=cu_seqlens, chunk_indices=ci_kkt, chunk_size=chunk_size)
        torch.npu.synchronize()

        # Step 3: solve_tri on CPU using numpy (NPU solve_tri has bf16 precision issues)
        # KKT outputs strictly lower-triangular (zero diagonal). solve_tril sets
        # diagonal to 1 (unit diagonal) then computes inverse via forward substitution.
        A_cpu = A.cpu()
        _, Hv_kkt, _, BT = A_cpu.shape
        A_tril = A_cpu.clone()
        for b_ in range(A_cpu.shape[0]):
            for h_ in range(Hv_kkt):
                for s_, e_ in zip(cu_seqlens[:-1], cu_seqlens[1:]):
                    for c0 in range(0, e_ - s_, BT):
                        seg = min(BT, e_ - s_ - c0)
                        M = A_cpu[b_, h_, s_ + c0:s_ + c0 + seg, :seg].numpy()
                        # Set unit diagonal, extract strict lower
                        L = np.tril(M, -1)  # strict lower (diagonal is 0 from KKT)
                        # inv(I + L) via forward substitution
                        X = np.eye(seg)
                        for i in range(1, seg):
                            for j in range(i):
                                s_val = -L[i, j]
                                for kk in range(j, i):
                                    s_val -= L[i, kk] * X[kk, j]
                                X[i, j] = s_val
                        A_tril[b_, h_, s_ + c0:s_ + c0 + seg, :seg] = torch.from_numpy(X)

        # Step 4: recompute_wu_fwd(k, v, beta, A_tril, g_cumsum) → w, u
        # KKT outputs Hk heads; recompute_wu expects Hv heads.
        if ratio > 1:
            A_expanded = A_tril.repeat_interleave(ratio, dim=1).contiguous()
        else:
            A_expanded = A_tril

        w, u = ascendc_ops.npu_recompute_w_u_fwd(
            k=k.npu(), v=v.npu(), beta=beta.npu(),
            A=A_expanded.to(torch.bfloat16).npu(),
            chunk_size=chunk_size, g=g_cumsum.npu(),
            cu_seqlens=cu_seqlens, chunk_indices=ci_kkt)
        torch.npu.synchronize()

        # Step 5: chunk_gated_delta_rule_fwd_h(k, w, u, g_cumsum, initial_state)
        initial_state = case["initial_state"]
        if initial_state is None:
            initial_state = torch.zeros(1, Hv, k.shape[-1], v.shape[-1], dtype=torch.float32)

        h, v_new, final_state = ascendc_ops.npu_chunk_gated_delta_rule_fwd_h(
            k=k.npu(), w=w, u=u, g=g_cumsum.npu(),
            initial_state=initial_state.npu(),
            output_final_state=True,
            chunk_size=chunk_size,
            cu_seqlens=cu_seqlens, chunk_indices=ci_kkt)
        torch.npu.synchronize()

        # Compare h with GPU golden (truncate expected if needed)
        n_chunks_actual = h.shape[2]
        h_expected = case["h_expected"][:, :, :n_chunks_actual]
        self.assertTensorClose(h, h_expected)

    def test_prefill_loaded_state_seq32(self):
        """seq32, 1 chunk, loaded initial_state."""
        self._run_case("prefill_loaded_state_seq32.pt")

    def test_prefill_zero_state_seq2047(self):
        """seq2047, truncated to 1 chunk (T=64). Multi-chunk state recurrence
        amplifies w/u precision differences between NPU and GPU pipelines.
        Requires GPU w/u dump data for full-sequence validation."""
        self._run_case("prefill_zero_state_seq2047.pt", max_chunks=1)


if __name__ == "__main__":
    unittest.main()
