# -*- coding: utf-8 -*-
"""Parity tests for the pure-torch Ascend RoPE helper.

The
helper's `is_neox_style` flag names the *interleaved* (GPT-J adjacent-pair)
rotation when True and the NeoX half-split rotation when False -- inverted
versus RopeConfig.is_neox_style. These tests lock both pairings against
independent reference implementations, verify the flag actually changes the
result (guards against re-hardcoding), and pin the halves-[cos|sin] cache
contract. Pure torch math: runs on CPU, no torch_npu required.
"""

import unittest

import torch

from rtp_llm.models_py.modules.factory.attention.ascend_impl.ascend_rope import (
    apply_rope_pos_ids_nhd,
)


def build_halves_cache(max_pos: int, rope_dim: int, theta: float = 10000.0) -> torch.Tensor:
    """Halves [cos|sin] cache, mirroring get_rope_cache_once(interleave=False)."""
    inv_freq = 1.0 / (theta ** (torch.arange(0, rope_dim, 2).float() / rope_dim))
    t = torch.arange(max_pos).float()
    freqs = torch.outer(t, inv_freq)
    return torch.cat([freqs.cos(), freqs.sin()], dim=-1).contiguous()


def ref_rotate(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, interleaved: bool) -> torch.Tensor:
    """Reference rotation for one tensor: x [T, H, D], cos/sin [T, half]."""
    rope_dim = cos.shape[-1] * 2
    xr = x[..., :rope_dim]
    cos_e = cos.unsqueeze(-2)
    sin_e = sin.unsqueeze(-2)
    if interleaved:
        x2 = xr.reshape(*xr.shape[:-1], -1, 2)
        even, odd = x2[..., 0], x2[..., 1]
        rot = torch.stack(
            [even * cos_e - odd * sin_e, even * sin_e + odd * cos_e], dim=-1
        ).flatten(start_dim=-2)
    else:
        half = rope_dim // 2
        first, second = xr[..., :half], xr[..., half:]
        rot = torch.cat(
            [first * cos_e - second * sin_e, first * sin_e + second * cos_e], dim=-1
        )
    return torch.cat([rot, x[..., rope_dim:]], dim=-1)


class TestAscendRopeParity(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(114514)
        self.max_pos = 64
        self.tokens = 16
        self.q_heads = 4
        self.kv_heads = 2
        # head_dim > rope_dim: exercise the passthrough tail.
        self.rope_dim = 16
        self.head_dim = 24
        self.cache = build_halves_cache(self.max_pos, self.rope_dim)
        # Non-contiguous, non-zero position ids to exercise cache indexing.
        self.pos_ids = torch.tensor([3, 3, 17, 40, 41, 41, 5, 9, 63, 2, 2, 33, 8, 21, 50, 12],
                                    dtype=torch.int32)
        assert self.pos_ids.numel() == self.tokens

    def _make_qk(self):
        q = torch.randn(self.tokens, self.q_heads, self.head_dim)
        k = torch.randn(self.tokens, self.kv_heads, self.head_dim)
        return q, k

    def _cos_sin_at_pos(self):
        emb = self.cache[self.pos_ids.long()]
        half = self.rope_dim // 2
        return emb[..., :half], emb[..., half:]

    def test_neox_half_split_matches_reference(self):
        q, k = self._make_qk()
        q_ref, k_ref = q.clone(), k.clone()
        cos, sin = self._cos_sin_at_pos()
        apply_rope_pos_ids_nhd(q, k, self.cache, self.pos_ids, is_neox_style=False)
        self.assertTrue(torch.allclose(q, ref_rotate(q_ref, cos, sin, interleaved=False), atol=1e-5))
        self.assertTrue(torch.allclose(k, ref_rotate(k_ref, cos, sin, interleaved=False), atol=1e-5))

    def test_interleaved_adjacent_pairs_matches_reference(self):
        q, k = self._make_qk()
        q_ref, k_ref = q.clone(), k.clone()
        cos, sin = self._cos_sin_at_pos()
        apply_rope_pos_ids_nhd(q, k, self.cache, self.pos_ids, is_neox_style=True)
        self.assertTrue(torch.allclose(q, ref_rotate(q_ref, cos, sin, interleaved=True), atol=1e-5))
        self.assertTrue(torch.allclose(k, ref_rotate(k_ref, cos, sin, interleaved=True), atol=1e-5))

    def test_flag_changes_result_and_passthrough_preserved(self):
        # The flag must actually select the pairing (guards against
        # re-hardcoding one style), and dims beyond rope_dim must pass through
        # untouched in both styles.
        q0, _ = self._make_qk()
        q_neox, q_ij = q0.clone(), q0.clone()
        apply_rope_pos_ids_nhd(q_neox, q0.clone(), self.cache, self.pos_ids, is_neox_style=False)
        apply_rope_pos_ids_nhd(q_ij, q0.clone(), self.cache, self.pos_ids, is_neox_style=True)
        self.assertFalse(torch.allclose(q_neox[..., : self.rope_dim], q_ij[..., : self.rope_dim], atol=1e-4))
        self.assertTrue(torch.equal(q_neox[..., self.rope_dim:], q_ij[..., self.rope_dim:]))
        self.assertTrue(torch.allclose(q_neox[..., self.rope_dim:], q0[..., self.rope_dim:], atol=0.0))

    def test_independent_input_rotation_is_position_correct(self):
        # Rotation must depend only on each token's own position: identical
        # input rows under the same pos id rotate to identical outputs, and
        # the result matches the per-position reference.
        q2 = torch.randn(1, 1, self.head_dim).repeat(2, 1, 1)  # two identical rows
        pos2 = torch.tensor([7, 7], dtype=torch.int32)
        emb = self.cache[7]
        half = self.rope_dim // 2
        c = emb[:half].unsqueeze(0).expand(2, half)
        s = emb[half:].unsqueeze(0).expand(2, half)
        q2_ref = q2.clone()
        apply_rope_pos_ids_nhd(q2, q2.clone(), self.cache, pos2, is_neox_style=False)
        self.assertTrue(torch.allclose(q2, ref_rotate(q2_ref, c, s, interleaved=False), atol=1e-5))
        self.assertTrue(torch.allclose(q2[0], q2[1], atol=1e-6))


def _npu_available() -> bool:
    try:
        import torch_npu  # noqa: F401
        import torch

        return torch.npu.is_available()
    except Exception:
        return False


def ref_mrope_half(x: torch.Tensor, cache: torch.Tensor, pos: torch.Tensor, sections) -> torch.Tensor:
    """Reference MRoPE (HF Qwen2-VL / vLLM triton_mrope semantics): global NeoX
    half-split pairing (d, d + rope_dim/2); frequency f takes its position from
    the axis owning f's section (sections in dimension-pair units)."""
    rope_dim = cache.shape[-1]
    half = rope_dim // 2
    t_end = sections[0]
    h_end = t_end + sections[1]
    out = x.clone()
    for f in range(half):
        s = 0 if f < t_end else (1 if f < h_end else 2)
        idx = pos[:, s].long()
        cos = cache[idx, f].unsqueeze(1)
        sin = cache[idx, half + f].unsqueeze(1)
        u, v = x[..., f], x[..., f + half]
        out[..., f] = u * cos - v * sin
        out[..., f + half] = u * sin + v * cos
    return out


@unittest.skipIf(not _npu_available(), "npu_mrope parity requires an NPU device")
class TestAscendMRopeParity(unittest.TestCase):
    """Parity between torch_npu.npu_mrope and the pure-torch MRoPE reference.

    Contract locked on CANN 9.0 (see AscendRotaryEmbeddingOp): positions
    (3, T) int64, q/k 2-D, halves [cos|sin] cache, global half-split pairing,
    sections in dimension-pair units summing to rotary_dim/2.
    """

    def test_npu_mrope_matches_reference(self):
        import torch_npu

        torch.manual_seed(114514)
        dev = "npu:0"
        max_pos, tokens, q_heads, kv_heads, head_size = 128, 6, 4, 2, 128
        sections = [16, 24, 24]  # pair units; sum == head_size / 2

        inv = 1.0 / (10000.0 ** (torch.arange(0, head_size, 2).float() / head_size))
        freqs = torch.outer(torch.arange(max_pos).float(), inv)
        cache = torch.cat([freqs.cos(), freqs.sin()], -1)
        pos = torch.randint(0, max_pos, (tokens, 3), dtype=torch.int64)

        for dtype in (torch.float32, torch.float16):
            with self.subTest(dtype=dtype):
                cache_d = cache.to(dev, dtype)
                q = torch.randn(tokens, q_heads, head_size, device=dev).to(dtype)
                k = torch.randn(tokens, kv_heads, head_size, device=dev).to(dtype)
                q_out, k_out = torch_npu.npu_mrope(
                    pos.t().contiguous().to(dev),
                    q.reshape(tokens, -1),
                    k.reshape(tokens, -1),
                    cache_d,
                    head_size,
                    mrope_section=sections,
                    rotary_mode="half",
                    cache_mode="default",
                )
                # fp16 rounding at |x|~4 is ~2e-3, so the tolerance scales by dtype.
                atol = 1e-4 if dtype == torch.float32 else 1e-2
                self.assertTrue(
                    torch.allclose(q_out.reshape(q.shape).float().cpu(), ref_mrope_half(q.float().cpu(), cache, pos, sections), atol=atol)
                )
                self.assertTrue(
                    torch.allclose(k_out.reshape(k.shape).float().cpu(), ref_mrope_half(k.float().cpu(), cache, pos, sections), atol=atol)
                )


if __name__ == "__main__":
    unittest.main()
