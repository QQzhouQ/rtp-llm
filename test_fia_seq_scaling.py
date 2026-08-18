"""FIA Contiguous 拷贝占比 vs 序列长度 扫描实验.

验证: aclnnContiguous_SliceAiCore_Slice 的耗时占比是否随序列长度增加而快速变小.
原理: Contiguous 拷贝的是整个 k/v pool strided view (大小=pool 总块数, 与 ctx 无关),
      FIA attention 计算随 ctx 增长 => 占比应下降.

运行: python test_fia_seq_scaling.py   (需 NPU 环境)
"""

import collections
import glob
import json
import os
import shutil

import torch
import torch_npu
from torch_npu.profiler import profile, ProfilerActivity

torch.manual_seed(0)
DEV = "npu:0"
DTYPE = torch.bfloat16

NUM_HEADS, KV_HEADS, HEAD_DIM = 16, 8, 128
PAGE = 128
BLOCKS = 64                       # 固定 pool 大小 (64 块 = 16.7MB/k)
MAX_CTX = 8192
ITERS = 20
OUTROOT = "/tmp/opencode/fia_seq_scaling"

MASK_CACHE = {}


def get_mask(ctx):
    # 生产 _get_causal_mask 同款: 按 ctx 尺寸构造 causal mask (2048 封顶可放大)
    size = max(2048, ctx)
    if size not in MASK_CACHE:
        MASK_CACHE[size] = torch.triu(torch.ones(size, size, dtype=torch.int8), diagonal=1).to(DEV)
    return MASK_CACHE[size]


def run_ctx(ctx):
    q = torch.randn(1, NUM_HEADS, HEAD_DIM, dtype=DTYPE, device=DEV)
    kv_base = torch.randn(BLOCKS, 2, PAGE, KV_HEADS, HEAD_DIM, dtype=DTYPE, device=DEV)
    k = kv_base[:, 0].reshape(BLOCKS, PAGE, -1)   # 生产同款 strided view
    v = kv_base[:, 1].reshape(BLOCKS, PAGE, -1)
    nb = (ctx + PAGE - 1) // PAGE
    bt = torch.zeros(1, nb, dtype=torch.int32, device=DEV)
    bt[0, :] = torch.arange(nb, dtype=torch.int32)
    sq = torch.tensor([1], dtype=torch.int32, device=DEV)
    sk = torch.tensor([ctx], dtype=torch.int32, device=DEV)

    def fia():
        torch_npu.npu_fused_infer_attention_score_v2(
            query=q, key=k, value=v, atten_mask=get_mask(ctx), block_table=bt,
            input_layout="TND", block_size=PAGE,
            actual_seq_qlen=sq, actual_seq_kvlen=sk,
            num_key_value_heads=KV_HEADS, num_query_heads=NUM_HEADS,
            softmax_scale=HEAD_DIM ** -0.5, sparse_mode=3)

    for _ in range(5):
        fia()
    torch.npu.synchronize()

    outdir = os.path.join(OUTROOT, f"ctx{ctx}")
    shutil.rmtree(outdir, ignore_errors=True)
    os.makedirs(outdir, exist_ok=True)
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.NPU],
                 on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(outdir, worker_name=f"ctx{ctx}")):
        for _ in range(ITERS):
            fia()
        torch.npu.synchronize()

    tr = glob.glob(os.path.join(outdir, "*", "ASCEND_PROFILER_OUTPUT", "trace_view.json"))[0]
    evs = json.load(open(tr))
    evs = evs if isinstance(evs, list) else evs.get("traceEvents", [])
    dur = collections.defaultdict(float)
    for e in evs:
        if e.get("ph") != "X":
            continue
        n = e.get("name", "")
        d = e.get("dur")
        if not isinstance(d, (int, float)):
            continue
        if "Contiguous" in n:
            dur["contig"] += d
        elif n == "FusedInferAttentionScore":
            dur["fia"] += d
    return dur["contig"] / ITERS, dur["fia"] / ITERS


def main():
    shutil.rmtree(OUTROOT, ignore_errors=True)
    print(f"pool 固定: BLOCKS={BLOCKS} (K pool {BLOCKS*PAGE*KV_HEADS*HEAD_DIM*2/1e6:.1f} MB), FIA 每次拷贝整个 pool view\n")
    print(f"{'ctx':>6} | {'Contig(us)':>10} | {'FIA(us)':>9} | {'Contig/(C+F)':>12}")
    print("-" * 50)
    for ctx in [128, 256, 512, 1024, 2048, 4096, 8192]:
        c, f = run_ctx(ctx)
        ratio = c / (c + f) * 100 if (c + f) > 0 else 0
        print(f"{ctx:>6} | {c:>10.1f} | {f:>9.1f} | {ratio:>11.1f}%")


if __name__ == "__main__":
    main()
