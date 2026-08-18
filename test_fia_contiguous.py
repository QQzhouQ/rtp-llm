"""FIA (npu_fused_infer_attention_score_v2) contiguous 拷贝隔离单测.

目的: 验证 decode 阶段的 aclnnContiguous_SliceAiCore_Slice 到底由谁引入:
  Case A: 仅 kv_base[:, 0/1].reshape(...)   (生产路径 FIA 前的 k/v 视图变换, 无 FIA)
  Case B: FIA + 完全连续的 k/v              (隔离 FIA 算子本身)
  Case C: 生产路径完整序列 reshape + FIA     (ascend_decode.py _forward_fia 同款)
  Case D: MatMul 对照 (非连续输入的 F.linear)

每个 case 独立 profile, 统计设备侧 kernel 直方图, 重点看 Contiguous 类 kernel.

运行: python test_fia_contiguous.py   (需在 NPU 环境)
"""

import collections
import glob
import json
import os
import shutil
import sys

import torch
import torch_npu
from torch_npu.profiler import profile, ProfilerActivity
import torch_npu.profiler

torch.manual_seed(0)
DEV = "npu:0"
DTYPE = torch.bfloat16

# Qwen3-0.6B decode 形状参数 (与 ascend_decode.py 生产路径一致)
NUM_HEADS, KV_HEADS, HEAD_DIM = 16, 8, 128
PAGE = 128            # kv_cache seq_size_per_block
BLOCKS = 16           # kv cache 总块数
BS = 1                # decode batch
CTX = 300             # 上下文长度 (跨 3 个 block)
ITERS = 20            # 每 case 采集迭代数
OUTROOT = "/tmp/opencode/fia_contig_test"

MASK = torch.triu(torch.ones(2048, 2048, dtype=torch.int8), diagonal=1).to(DEV)


def make_inputs():
    q = torch.randn(BS, NUM_HEADS, HEAD_DIM, dtype=DTYPE, device=DEV)
    # 生产布局: [blocks, 2, page, kv_heads, dim] (C++ getLayerCache reshape 后的 BSND)
    kv_base = torch.randn(BLOCKS, 2, PAGE, KV_HEADS, HEAD_DIM, dtype=DTYPE, device=DEV)
    block_table = torch.tensor([[0, 1, 2]], dtype=torch.int32, device=DEV)
    seq_q = torch.arange(1, BS + 1, dtype=torch.int32, device=DEV)
    seq_kv = torch.tensor([CTX], dtype=torch.int32, device=DEV)
    return q, kv_base, block_table, seq_q, seq_kv


def fia(q, k, v, block_table, seq_q, seq_kv):
    out, _ = torch_npu.npu_fused_infer_attention_score_v2(
        query=q, key=k, value=v,
        atten_mask=MASK, block_table=block_table,
        input_layout="TND", block_size=PAGE,
        actual_seq_qlen=seq_q, actual_seq_kvlen=seq_kv,
        num_key_value_heads=KV_HEADS, num_query_heads=NUM_HEADS,
        softmax_scale=HEAD_DIM ** -0.5, sparse_mode=3)
    return out


def case_a_reshape_only(q, kv_base, block_table, seq_q, seq_kv):
    """仅生产路径的 k/v 视图变换, 不调 FIA (与 _forward_fia 前两行相同)."""
    k = kv_base[:, 0].reshape(kv_base.shape[0], PAGE, -1)
    v = kv_base[:, 1].reshape(kv_base.shape[0], PAGE, -1)
    return k, v


def case_b_fia_contiguous_kv(q, kv_base, block_table, seq_q, seq_kv):
    """FIA + 连续 k/v (在 profile 之外预先构造, 隔离算子本身)."""
    k = kv_base[:, 0].reshape(kv_base.shape[0], PAGE, -1).contiguous()
    v = kv_base[:, 1].reshape(kv_base.shape[0], PAGE, -1).contiguous()
    return fia(q, k, v, block_table, seq_q, seq_kv)


def case_c_production(q, kv_base, block_table, seq_q, seq_kv):
    """生产路径: 视图变换 + FIA (ascend_decode.py _forward_fia 同款)."""
    k = kv_base[:, 0].reshape(kv_base.shape[0], PAGE, -1)
    v = kv_base[:, 1].reshape(kv_base.shape[0], PAGE, -1)
    return fia(q, k, v, block_table, seq_q, seq_kv)


_W = torch.randn(NUM_HEADS * HEAD_DIM, KV_HEADS * HEAD_DIM, dtype=DTYPE, device=DEV)  # [out=2048, in=1024]
_BIG = torch.randn(BLOCKS, NUM_HEADS * HEAD_DIM, dtype=DTYPE, device=DEV)  # [16, 2048]


def case_d_matmul_control(q, kv_base, block_table, seq_q, seq_kv):
    """MatMul 对照: 非连续输入的 F.linear (经典列切片, 保证是 view)."""
    x = _BIG[:, :KV_HEADS * HEAD_DIM]  # [16, 1024] 非连续 strided view
    return torch.nn.functional.linear(x, _W)


def run_case(name, fn, inputs):
    q, kv_base, block_table, seq_q, seq_kv = inputs
    # warmup (首次调用有编译/初始化噪音, 不计入)
    for _ in range(5):
        fn(q, kv_base, block_table, seq_q, seq_kv)
    torch.npu.synchronize()

    outdir = os.path.join(OUTROOT, name)
    if os.path.exists(outdir):
        shutil.rmtree(outdir)
    os.makedirs(outdir, exist_ok=True)

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.NPU],
        on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(outdir, worker_name=name),
    ):
        for _ in range(ITERS):
            fn(q, kv_base, block_table, seq_q, seq_kv)
        torch.npu.synchronize()

    traces = glob.glob(os.path.join(outdir, "*", "ASCEND_PROFILER_OUTPUT", "trace_view.json"))
    if not traces:
        print(f"[{name}] 未找到 trace!")
        return
    evs = json.load(open(traces[0]))
    evs = evs if isinstance(evs, list) else evs.get("traceEvents", [])

    # 设备侧 kernel: 排除 host 派发/元数据事件后的 X 事件
    def is_kernel(n):
        if any(s in n for s in ("Workspace", "Enqueue@", "Dequeue@", "AscendCL@")):
            return False
        if n.startswith(("aten::", "npu::", "empty_tensor", "Computing", "Free", "NOP", "PROFILING_")):
            return False
        return True

    kern = collections.Counter()
    for e in evs:
        if e.get("ph") != "X":
            continue
        n = e.get("name", "")
        if n and is_kernel(n):
            kern[n] += 1

    total_k = sum(kern.values())
    total_c = sum(c for n, c in kern.items() if "Contiguous" in n)
    print(f"\n=== Case {name}: {ITERS} 次迭代 ===")
    print(f"  设备 kernel 总数: {total_k:6d}  (平均 {total_k/ITERS:.1f}/iter)")
    print(f"  Contiguous 类 kernel: {total_c:6d}  (平均 {total_c/ITERS:.1f}/iter)")
    print("  kernel 直方图 (前 8):")
    for n, c in kern.most_common(8):
        mark = "  <-- Contiguous!" if "Contiguous" in n else ""
        print(f"    x{c:<5d} {n}{mark}")


def main():
    shutil.rmtree(OUTROOT, ignore_errors=True)
    inputs = make_inputs()
    # 正确性 sanity: 生产路径输出应有限
    out = case_c_production(*inputs)
    assert torch.isfinite(out.float()).all(), "FIA 输出异常"
    print("sanity: FIA 输出有限 ✅  out shape:", tuple(out.shape))

    run_case("A_reshape_only", case_a_reshape_only, inputs)
    run_case("B_fia_contig_kv", case_b_fia_contiguous_kv, inputs)
    run_case("C_production", case_c_production, inputs)
    run_case("D_matmul_control", case_d_matmul_control, inputs)

    print("\n=== 结论判读 ===")
    print("  A 有 Contiguous → 来自 reshape (视图变换), 与 FIA 无关")
    print("  B 有 Contiguous → FIA 算子本身引入")
    print("  C - (A+B) 增量  → 生产路径组合的额外拷贝")
    print("  D 有 Contiguous → MatMul 对非连续输入的已知行为 (对照)")


if __name__ == "__main__":
    main()
