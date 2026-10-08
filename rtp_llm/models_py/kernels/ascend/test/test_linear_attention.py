import ast
import importlib.util
import unittest
from pathlib import Path
from unittest import mock

MODULE_PATH = Path(__file__).resolve().parents[1] / "linear_attention.py"


class TestLinearAttentionWithoutRuntimeDependencies(unittest.TestCase):
    def test_module_is_valid_python(self):
        source = MODULE_PATH.read_text(encoding="utf-8")
        compile(source, str(MODULE_PATH), "exec")

    def test_accelerator_dependencies_are_lazy(self):
        tree = ast.parse(MODULE_PATH.read_text(encoding="utf-8"))
        eager_imports = [
            node for node in tree.body if isinstance(node, (ast.Import, ast.ImportFrom))
        ]
        names = []
        for node in eager_imports:
            if isinstance(node, ast.Import):
                names.extend(alias.name for alias in node.names)
            else:
                names.append(node.module or "")
        self.assertFalse(any("triton" in name or "fla_npu" in name for name in names))

    def test_head_first_uses_not_implemented_error(self):
        tree = ast.parse(MODULE_PATH.read_text(encoding="utf-8"))
        chunk_function = next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "chunk_gated_delta_rule"
        )
        raised_errors = [
            node.exc.func.id
            for node in ast.walk(chunk_function)
            if isinstance(node, ast.Raise)
            and isinstance(node.exc, ast.Call)
            and isinstance(node.exc.func, ast.Name)
        ]
        self.assertIn("NotImplementedError", raised_errors)
        self.assertNotIn("DeprecationWarning", raised_errors)


try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "PyTorch is not installed in this test environment")
class TestLinearAttentionTorch(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        spec = importlib.util.spec_from_file_location(
            "ascend_linear_attention", MODULE_PATH
        )
        cls.module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.module)

    def test_torch_l2norm_fallback_matches_rtp_formula(self):
        x = torch.randn(2, 3, 8, dtype=torch.float32)
        with mock.patch.object(self.module, "_get_npu_l2norm", return_value=None):
            actual = self.module.l2norm_fwd(x, eps=1e-6)
        expected = x * torch.rsqrt((x * x).sum(dim=-1, keepdim=True) + 1e-6)
        torch.testing.assert_close(actual, expected)

    def test_kkt_adapts_layout_and_expands_gqa_heads(self):
        class FakeOps:
            calls = None

            def __init__(self):
                self.calls = []

            def npu_chunk_scaled_dot_kkt(self, **kwargs):
                self.calls.append(kwargs)
                batch, heads, seqlen, _ = kwargs["k"].shape
                offset = kwargs["beta"][0, 0, 0]
                values = (torch.arange(heads, dtype=torch.float32) * 10 + offset).view(
                    1, heads, 1, 1
                )
                return values.expand(batch, heads, seqlen, kwargs["chunk_size"]).clone()

        fake = FakeOps()
        k = torch.randn(1, 4, 2, 8, dtype=torch.bfloat16)
        beta = torch.arange(16, dtype=torch.float32).reshape(1, 4, 4)
        g = beta + 100
        cu = torch.tensor([0, 4], dtype=torch.int32)
        with mock.patch.object(self.module, "_get_ascendc_ops", return_value=fake):
            out = self.module.chunk_scaled_dot_kkt_fwd(
                k, beta, g_cumsum=g, cu_seqlens=cu
            )

        self.assertEqual(len(fake.calls), 2)
        self.assertTrue(
            all(tuple(call["k"].shape) == (1, 2, 4, 8) for call in fake.calls)
        )
        self.assertTrue(
            all(tuple(call["beta"].shape) == (1, 2, 4) for call in fake.calls)
        )
        torch.testing.assert_close(fake.calls[0]["beta"], beta.transpose(1, 2)[:, 0::2])
        torch.testing.assert_close(fake.calls[1]["beta"], beta.transpose(1, 2)[:, 1::2])
        self.assertEqual(tuple(out.shape), (1, 4, 4, 64))
        self.assertFalse(torch.equal(out[:, :, 0], out[:, :, 1]))
        self.assertFalse(torch.equal(out[:, :, 2], out[:, :, 3]))

    def test_solve_tri_uses_fp16_npu_input(self):
        class FakeOps:
            dtype = None

            def npu_solve_tri(self, **kwargs):
                self.dtype = kwargs["x"].dtype
                return kwargs["x"]

        fake = FakeOps()
        A = torch.randn(1, 4, 2, 16, dtype=torch.float32)
        with mock.patch.object(self.module, "_get_ascendc_ops", return_value=fake):
            out = self.module.solve_tril(A, output_dtype=torch.float32)
        self.assertEqual(fake.dtype, torch.float16)
        self.assertEqual(out.dtype, torch.float32)

    def test_solve_tri_uses_tnd_layout_for_varlen(self):
        class FakeOps:
            def __init__(self):
                self.calls = []

            def npu_solve_tri(self, **kwargs):
                self.calls.append(kwargs)
                return kwargs["x"]

        fake = FakeOps()
        A = torch.randn(1, 9, 2, 16, dtype=torch.float32)
        cu = [0, 5, 9]  # sequence start 5 is not chunk-16 aligned
        with mock.patch.object(self.module, "_get_ascendc_ops", return_value=fake):
            out = self.module.solve_tril(A, cu_seqlens=cu, output_dtype=torch.float32)
        # varlen must go through the TND layout: BSND tiles chunks globally
        # and ignores the cu_seqlens starts
        self.assertEqual(len(fake.calls), 1)
        self.assertEqual(fake.calls[0]["layout"], "tnd")
        self.assertEqual(tuple(fake.calls[0]["x"].shape), (9, 2, 16))
        self.assertEqual(fake.calls[0]["cu_seqlens"], [0, 5, 9])
        self.assertEqual(
            fake.calls[0]["chunk_indices"], [0, 0, 1, 0]
        )
        self.assertEqual(tuple(out.shape), (1, 9, 2, 16))

    def test_solve_tri_without_cu_keeps_bsnd_layout(self):
        class FakeOps:
            def __init__(self):
                self.calls = []

            def npu_solve_tri(self, **kwargs):
                self.calls.append(kwargs)
                return kwargs["x"]

        fake = FakeOps()
        A = torch.randn(1, 32, 2, 16, dtype=torch.float32)
        with mock.patch.object(self.module, "_get_ascendc_ops", return_value=fake):
            self.module.solve_tril(A, output_dtype=torch.float32)
        self.assertEqual(len(fake.calls), 1)
        self.assertEqual(fake.calls[0]["layout"], "bsnd")
        self.assertIsNone(fake.calls[0]["cu_seqlens"])
        self.assertEqual(tuple(fake.calls[0]["x"].shape), (1, 32, 2, 16))

    def test_chunk_interface_rejects_head_first_layout(self):
        q = torch.zeros((1, 1, 1, 1), dtype=torch.float16)
        with self.assertRaisesRegex(NotImplementedError, "head_first=True"):
            self.module.chunk_gated_delta_rule(q, q, q, q, q, head_first=True)


try:
    import torch_npu  # noqa: F401

    _HAS_NPU = torch is not None and torch.npu.is_available()
except Exception:
    _HAS_NPU = False

try:
    from fla_npu.ops import ascendc as _ascendc  # noqa: F401

    _HAS_FLA_NPU = True
except Exception:
    _HAS_FLA_NPU = False


@unittest.skipUnless(
    torch is not None and _HAS_NPU and _HAS_FLA_NPU, "requires Ascend NPU + fla_npu"
)
class TestChunkVarlenMultiSequenceNPU(unittest.TestCase):
    """Regression guard for the merged-prefill garbled-output bug.

    The AscendC solve_tri tiles chunks at chunk_idx * chunk_size on the
    token axis, ignoring cu_seqlens starts, so a merged prefill whose
    later sequences start at chunk-unaligned positions reads across
    sequence boundaries and corrupts every non-first sequence.  The
    solve_tril wrapper must keep the pipeline bit-exact versus running
    each sequence through it alone.
    """

    NK, HV, DK, DV = 16, 32, 128, 128
    DIM = (2 * 16 + 32) * 128

    @classmethod
    def setUpClass(cls):
        spec = importlib.util.spec_from_file_location(
            "ascend_linear_attention", MODULE_PATH
        )
        cls.module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.module)

    def _cu_of(self, lens):
        return torch.tensor(
            [0] + list(torch.tensor(lens).cumsum(0)), dtype=torch.int32, device="npu"
        )

    def _run_chunk(self, xs, bs, as_, cu):
        from rtp_llm.models_py.kernels.ascend.common import fused_gdn_gating

        dev = "npu"
        NK, HV, DK, DV = self.NK, self.HV, self.DK, self.DV
        alog = torch.randn(HV, generator=torch.Generator().manual_seed(3))
        dt_bias = torch.randn(HV, generator=torch.Generator().manual_seed(4))
        g_, beta_ = fused_gdn_gating(
            alog.to(dev), as_.to(dev), bs.to(dev), dt_bias.to(dev)
        )
        q, k, v = torch.split(xs, [NK * DK, NK * DK, HV * DV], dim=-1)
        q = q.view(1, -1, NK, DK).to(dev).to(torch.bfloat16)
        k = k.view(1, -1, NK, DK).to(dev).to(torch.bfloat16)
        v = v.view(1, -1, HV, DV).to(dev).to(torch.bfloat16)
        o, _, fs = self.module.chunk_gated_delta_rule(
            q, k, v, g_, beta_, cu_seqlens=cu,
            output_final_state=True, use_qk_l2norm_in_kernel=True,
        )
        return o.squeeze(0), fs

    def test_multi_sequence_matches_per_sequence(self):
        NK, HV, DIM = self.NK, self.HV, self.DIM
        for seq_lens, seed in [([61, 58, 47], 7), ([300, 700, 40], 7), ([17, 9], 7)]:
            with self.subTest(seq_lens=seq_lens):
                gen = torch.Generator().manual_seed(seed)
                total = sum(seq_lens)
                x = torch.randn(total, DIM, generator=gen)
                b = torch.randn(total, HV, generator=gen)
                a = torch.randn(total, HV, generator=gen)

                o_multi, fs_multi = self._run_chunk(x, b, a, self._cu_of(seq_lens))
                o_singles, fs_singles = [], []
                off = 0
                for length in seq_lens:
                    o1, f1 = self._run_chunk(
                        x[off : off + length],
                        b[off : off + length],
                        a[off : off + length],
                        self._cu_of([length]),
                    )
                    o_singles.append(o1)
                    fs_singles.append(f1)
                    off += length

                ref = torch.cat(o_singles)
                out_diff = (o_multi.float() - ref.float()).abs().max().item()
                fs_diff = max(
                    (fs_multi[i].float() - fs_singles[i].float()).abs().max().item()
                    for i in range(len(seq_lens))
                )
                self.assertEqual(out_diff, 0.0, "output must be bit-exact")
                self.assertEqual(fs_diff, 0.0, "final states must be bit-exact")


if __name__ == "__main__":
    unittest.main()
