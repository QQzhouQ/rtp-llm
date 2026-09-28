#!/usr/bin/env python3
"""
Chain speculative sampling (MTP verify) tests.

Layer 1 (CPU, runs everywhere):
    reference_chain_sampling  — sequential transcription of the ROCm HIP
    kernel semantics (cpp/rocm/speculative_sampling/sampling.cu)
    vectorized_chain_sampling — python mirror of the ATen op sequence in
    CudaSampleOp.cc (USING_ASCEND branch) chainSpeculativeSampling.
    The two must agree exactly across an edge-case matrix + randomized fuzz.

Layer 2 (NPU, skipped without torch_npu / built module):
    calls the real execChainSpeculativeSampling via sampler_test_module and
    compares against the reference.

Usage:
  python rtp_llm/models_py/bindings/ascend/ops/tests/test_chain_speculative_sampling.py
"""

import os
import sys
import unittest

import torch

# ---------------------------------------------------------------------------
# Reference: sequential HIP kernel semantics.
#
# Per row b (k = propose_step, V = vocab):
#   accept draft token i while u * p < q (strict); first rejection breaks;
#   emitted = pos + 1;
#   if pos < k: resample position pos from relu(target[pos] - draft[pos])
#   via inverse CDF with u[b, min(pos+1, k)]; positions pos+1..k get -1.
# The all-accepted bonus slot [b, k] is left as -1 here (the HIP kernel
# leaves the zero-init value; the host overwrites the last emitted token
# with the target sampler's own token either way — dead value).
# ---------------------------------------------------------------------------


def reference_chain_sampling(draft_probs, draft_token_ids, uniform_samples, target_probs):
    batch, k, V = draft_probs.shape
    output = torch.full((batch, k + 1), -1, dtype=torch.int32)
    emitted = torch.zeros(batch, dtype=torch.int32)
    for b in range(batch):
        pos = k
        for i in range(k):
            draft_id = int(draft_token_ids[b, i])
            q = float(target_probs[b, i, draft_id])
            p = float(draft_probs[b, i, draft_id])
            u = float(uniform_samples[b, i])
            if u * p < q:
                output[b, i] = draft_id
            else:
                pos = i
                break
        emitted[b] += pos + 1
        if pos == k:
            continue
        resid = (target_probs[b, pos] - draft_probs[b, pos]).clamp_min(0)
        cdf = torch.cumsum(resid, dim=-1)
        u2 = float(uniform_samples[b, min(pos + 1, k)]) * float(cdf[-1])
        sampled = V - 1
        for j in range(V):
            if float(cdf[j]) > u2:
                sampled = j
                break
        output[b, pos] = sampled
    return output, emitted


# ---------------------------------------------------------------------------
# Vectorized mirror of the C++ (ATen) implementation. Keep the op sequence
# in sync with CudaSampleOp.cc chainSpeculativeSampling.
# ---------------------------------------------------------------------------


def vectorized_chain_sampling(draft_probs, draft_token_ids, uniform_samples, target_probs):
    batch, k, _ = draft_probs.shape
    device = draft_probs.device

    draft_ids_l = draft_token_ids.to(torch.int64)
    sel = draft_ids_l.unsqueeze(-1)  # [B, k, 1]
    q_sel = target_probs[:, :k].gather(-1, sel).squeeze(-1)  # [B, k]
    p_sel = draft_probs.gather(-1, sel).squeeze(-1)  # [B, k]

    accept = (uniform_samples[:, :k] * p_sel) < q_sel  # [B, k]
    accept_prefix = accept.to(torch.float32).cumprod(1).to(torch.bool)  # [B, k]
    pos = accept_prefix.sum(1)  # [B]

    neg_one = torch.full_like(draft_ids_l, -1)
    out_k = torch.where(accept_prefix, draft_ids_l, neg_one)  # [B, k]
    output = torch.full((batch, k + 1), -1, dtype=torch.int32, device=device)
    output[:, :k] = out_k

    rejected_rows = (pos < k).nonzero().squeeze(-1)  # [n]
    if rejected_rows.numel() > 0:
        pos_r = pos[rejected_rows]  # [n]
        t_row = target_probs[rejected_rows, pos_r]  # [n, V]
        d_row = draft_probs[rejected_rows, pos_r]  # [n, V]
        resid = (t_row - d_row).clamp_min(0)
        cdf = resid.cumsum(-1)  # [n, V]
        u_col = uniform_samples[rejected_rows, (pos_r + 1).clamp_max(k)]  # [n]
        u2 = u_col.unsqueeze(1) * cdf[:, -1:]  # [n, 1]
        sampled = (cdf <= u2).sum(-1).clamp_max(t_row.shape[1] - 1)  # [n]
        output[rejected_rows, pos_r] = sampled.to(torch.int32)

    emitted = (pos + 1).to(torch.int32)
    return output, emitted


def _make_case(draft_probs_list, draft_ids_list, uniforms_list, target_probs_list):
    """Lists of per-row [k, V] / [k] / [k+1] / [k+1, V] python data -> tensors."""
    draft_probs = torch.tensor(draft_probs_list, dtype=torch.float32)
    draft_ids = torch.tensor(draft_ids_list, dtype=torch.int32)
    uniforms = torch.tensor(uniforms_list, dtype=torch.float32)
    target_probs = torch.tensor(target_probs_list, dtype=torch.float32)
    return draft_probs, draft_ids, uniforms, target_probs


def _random_case(batch, k, V, seed):
    g = torch.Generator().manual_seed(seed)
    logits = torch.randn(batch, k, V, generator=g).abs()
    draft_probs = torch.softmax(logits, dim=-1)
    t_logits = torch.randn(batch, k + 1, V, generator=g).abs()
    target_probs = torch.softmax(t_logits, dim=-1)
    draft_ids = torch.randint(0, V, (batch, k), generator=g, dtype=torch.int32)
    uniforms = torch.rand(batch, k + 1, generator=g)
    return draft_probs, draft_ids, uniforms, target_probs


class TestChainSpeculativeSamplingCPU(unittest.TestCase):
    """Reference vs vectorized mirror (pure CPU, mirrors the C++ ATen code)."""

    def assert_case(self, draft_probs, draft_ids, uniforms, target_probs):
        ref_out, ref_emitted = reference_chain_sampling(draft_probs, draft_ids, uniforms, target_probs)
        vec_out, vec_emitted = vectorized_chain_sampling(draft_probs, draft_ids, uniforms, target_probs)
        torch.testing.assert_close(vec_out, ref_out, msg="output_token_ids mismatch")
        torch.testing.assert_close(vec_emitted, ref_emitted, msg="emitted mismatch")
        # chain invariant: 1 <= emitted <= k + 1
        k = draft_probs.shape[1]
        self.assertTrue(bool(((vec_emitted >= 1) & (vec_emitted <= k + 1)).all()))
        # emitted == 1 + number of non -1 accepted prefix slots (when pos < k,
        # the resampled slot is not a draft id; check via monotonic -1 tail)
        for b in range(vec_out.shape[0]):
            row = vec_out[b].tolist()
            e = int(vec_emitted[b])
            tail = row[e:]
            self.assertTrue(all(t == -1 for t in tail), f"row {b}: tail after emitted must be -1, got {row}")

    def test_all_accepted(self):
        # draft id 0 everywhere, target prob 1.0, draft prob 1.0, u tiny -> accept all
        V = 8
        dp, di, u, tp = _make_case(
            [[[1.0] * V]],
            [[0]],
            [[0.0, 0.5]],
            [[[1.0] * V, [1.0] * V]],
        )
        ref_out, ref_emitted = reference_chain_sampling(dp, di, u, tp)
        self.assertEqual(ref_emitted.tolist(), [2])
        self.assertEqual(ref_out[0].tolist(), [0, -1])
        self.assert_case(dp, di, u, tp)

    def test_first_rejected(self):
        # u * p == q exactly -> strict < rejects immediately (boundary)
        V = 6
        dp, di, u, tp = _make_case(
            [[[0.5, 0.5, 0.0, 0.0, 0.0, 0.0]]],
            [[1]],
            [[0.5, 0.25]],  # u * p = 0.25 == q(=0.25) -> reject at pos 0
            [[[0.25, 0.25, 0.1, 0.1, 0.15, 0.15], [0.0] * V]],
        )
        ref_out, ref_emitted = reference_chain_sampling(dp, di, u, tp)
        self.assertEqual(ref_emitted.tolist(), [1])
        # residual = relu(q - p) = [0, 0, .1, .1, .15, .15], sum=0.5
        # u2 = 0.25 * 0.5 = 0.125 -> first cdf > 0.125 is index 2 (cdf2=0.1<=0.125? no wait)
        # cdf = [0, 0, .1, .2, .35, .5]; first > 0.125 -> index 3
        self.assertEqual(ref_out[0, 0].item(), 3)
        self.assertEqual(ref_out[0, 1].item(), -1)
        self.assert_case(dp, di, u, tp)

    def test_middle_rejection(self):
        V = 4
        # row: k = 3; accept step0/1, reject step2
        dp, di, u, tp = _make_case(
            [
                [
                    [0.25, 0.25, 0.25, 0.25],
                    [0.25, 0.25, 0.25, 0.25],
                    [0.7, 0.1, 0.1, 0.1],
                ]
            ],
            [[0, 1, 2]],
            [[0.0, 0.0, 1.0, 0.9]],  # step2: u*p=0.7 >= q(=0.1 for id2) -> reject
            [
                [
                    [0.25, 0.25, 0.25, 0.25],  # step0: u*p=0 < q -> accept
                    [0.25, 0.25, 0.25, 0.25],  # step1: accept
                    [0.1, 0.2, 0.1, 0.6],  # step2 (reject pos): q[id2]=0.1
                    [0.0, 0.0, 0.0, 0.0],  # bonus slot unused
                ]
            ],
        )
        ref_out, ref_emitted = reference_chain_sampling(dp, di, u, tp)
        self.assertEqual(ref_emitted.tolist(), [3])  # pos=2 -> emitted=3
        self.assertEqual(ref_out[0, 0].item(), 0)
        self.assertEqual(ref_out[0, 1].item(), 1)
        # residual at pos2: relu([.1,.2,.1,.6] - [.7,.1,.1,.1]) = [0,.1,0,.5]
        # u2 = 0.9 * 0.6 = 0.54; cdf = [0,.1,.1,.6]; first > .54 -> 3
        self.assertEqual(ref_out[0, 2].item(), 3)
        self.assertEqual(ref_out[0, 3].item(), -1)
        self.assert_case(dp, di, u, tp)

    def test_k1_fast_path(self):
        V = 5
        for u0 in (0.0, 0.3, 0.99):
            dp, di, u, tp = _make_case(
                [[[0.2] * V]],
                [[3]],
                [[u0, 0.5]],
                [[[0.2] * V, [0.2] * V]],
            )
            self.assert_case(dp, di, u, tp)

    def test_residual_all_zero_clamps_to_last(self):
        # identical distributions + u == 1: strict < rejects, residual is
        # exactly 0 -> cdf 0, u2 = 0, (cdf <= u2) all True -> sampled = V-1
        # (matches the HIP kernel's sampled_id = d - 1 fallback)
        V = 6
        dp, di, u, tp = _make_case(
            [[[0.2, 0.2, 0.2, 0.2, 0.1, 0.1]]],
            [[0]],
            [[1.0, 0.5]],  # u*p = 0.2 == q -> strict < rejects
            [[[0.2, 0.2, 0.2, 0.2, 0.1, 0.1], [0.0] * V]],
        )
        ref_out, ref_emitted = reference_chain_sampling(dp, di, u, tp)
        self.assertEqual(ref_emitted.tolist(), [1])
        self.assertEqual(ref_out[0, 0].item(), V - 1)
        self.assert_case(dp, di, u, tp)

    def test_mixed_batch(self):
        V = 5
        k = 2
        # row0: accept all; row1: reject at 0; row2: reject at 1
        dp = torch.softmax(torch.tensor([[[3.0, 1.0, 1.0, 1.0, 1.0]] * k,
                                         [[1.0, 1.0, 1.0, 1.0, 1.0]] * k,
                                         [[1.0, 1.0, 1.0, 1.0, 1.0]] * k],
                                        dtype=torch.float32), dim=-1)
        di = torch.tensor([[0, 0], [4, 4], [2, 2]], dtype=torch.int32)
        u = torch.tensor([[0.0, 0.0, 0.0],
                          [1.0, 0.3, 0.7],
                          [0.0, 1.0, 0.2]], dtype=torch.float32)
        t_logits = torch.tensor([[[4.0, 1.0, 1.0, 1.0, 1.0]] * (k + 1),
                                 [[1.0, 1.0, 1.0, 1.0, 1.0]] * (k + 1),
                                 [[1.0, 1.0, 1.0, 1.0, 1.0]] * (k + 1)],
                                dtype=torch.float32)
        tp = torch.softmax(t_logits, dim=-1)
        self.assert_case(dp, di, u, tp)

    def test_fuzz_matrix(self):
        for seed in range(200):
            batch = 1 + (seed % 7)
            k = 1 + (seed % 6)  # up to 6 (rtp-llm proposes <= 7)
            V = 3 + (seed % 37)
            case = _random_case(batch, k, V, seed)
            with self.subTest(seed=seed, batch=batch, k=k, V=V):
                self.assert_case(*case)

    def test_fuzz_high_accept(self):
        # target == draft distribution with u drawn from [0, 0.999] — mostly accepted
        for seed in range(50):
            batch, k, V = 4, 3, 11
            g = torch.Generator().manual_seed(1000 + seed)
            t_logits = torch.randn(batch, k + 1, V, generator=g)
            tp = torch.softmax(t_logits, dim=-1)
            dp = tp[:, :k].clone()
            di = torch.multinomial(dp.reshape(-1, V), 1, generator=g).reshape(batch, k).to(torch.int32)
            u = torch.rand(batch, k + 1, generator=g) * 0.999
            with self.subTest(seed=seed):
                self.assert_case(dp, di, u, tp)


# ---------------------------------------------------------------------------
# NPU integration: real C++ implementation via sampler_test_module.so
# ---------------------------------------------------------------------------

_NPU_AVAILABLE = False
_sampler_module = None
try:
    import torch_npu  # noqa: F401

    _BAZEL_BIN = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "../../../../../../bazel-bin/rtp_llm/models_py/bindings/ascend/ops/tests",
    )
    if _BAZEL_BIN not in sys.path:
        sys.path.insert(0, _BAZEL_BIN)
    import sampler_test_module  # type: ignore[import-not-found]

    _sampler_module = sampler_test_module
    _NPU_AVAILABLE = torch.npu.is_available() if hasattr(torch, "npu") else False
except Exception:  # noqa: BLE001 — optional layer
    _NPU_AVAILABLE = False


@unittest.skipUnless(_NPU_AVAILABLE and _sampler_module is not None, "NPU / sampler_test_module unavailable")
class TestChainSpeculativeSamplingNPU(unittest.TestCase):
    NPU_DEVICE = torch.device("npu:0")

    def _to_npu(self, case):
        dp, di, u, tp = case
        return (
            dp.to(self.NPU_DEVICE),
            di.to(self.NPU_DEVICE),
            u.to(self.NPU_DEVICE),
            tp.to(self.NPU_DEVICE),
        )

    def test_edge_cases_on_npu(self):
        for name, case in [
            ("all_accepted", _make_case(
                [[[1.0] * 8]], [[0]], [[0.0, 0.5]],
                [[[1.0] * 8, [1.0] * 8]])),
            ("first_rejected", _make_case(
                [[[0.5, 0.5, 0.0, 0.0, 0.0, 0.0]]], [[1]], [[0.5, 0.25]],
                [[[0.25, 0.25, 0.1, 0.1, 0.15, 0.15], [0.0] * 6]])),
        ]:
            with self.subTest(case=name):
                ref_out, ref_emitted = reference_chain_sampling(*case)
                res = _sampler_module.chain_speculative_sampling(*self._to_npu(case))
                torch.testing.assert_close(res["output_token_ids"], ref_out)
                torch.testing.assert_close(res["output_emitted_token_num"], ref_emitted)

    def test_fuzz_on_npu(self):
        for seed in range(30):
            batch = 1 + (seed % 5)
            k = 1 + (seed % 5)
            V = 1519  # vocab-scale row length to exercise cumsum over V
            case = _random_case(batch, k, V, seed)
            ref_out, ref_emitted = reference_chain_sampling(*case)
            res = _sampler_module.chain_speculative_sampling(*self._to_npu(case))
            torch.testing.assert_close(
                res["output_token_ids"], ref_out, msg=f"seed={seed} output mismatch")
            torch.testing.assert_close(
                res["output_emitted_token_num"], ref_emitted, msg=f"seed={seed} emitted mismatch")


if __name__ == "__main__":
    unittest.main(verbosity=2)
