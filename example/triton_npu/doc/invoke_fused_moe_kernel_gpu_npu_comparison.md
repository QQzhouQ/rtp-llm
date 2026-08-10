# invoke_fused_moe_kernel 算子 GPU-NPU 精度比对指南

本文档总结 fused MoE 算子 `invoke_fused_moe_kernel` NPU-vs-GPU 比对的实践经验，组织方式参考 `gpu_npu_comparison_guide.md` 的通用方法论。

## 整体流程

```
GPU 黄金数据 (.pt)  →  打印 key/shape 诊断  →  解析 inputs / inplace_outputs  →  CPU reference 验证语义  →  调用 rtp-llm invoke_fused_moe_kernel Triton kernel (NPU)  →  比对 scatter 后的 C
```

## 1. 理解 GPU 数据格式

GPU 端通过 `torch.save()` 导出的 `.pt` 文件位于 `<workspace>/sample_moe/invoke_fused_moe_kernel/`，共 6 个用例。

### 用例概览

| 文件 | M（token） | top_k | num_valid=M*top_k | N | K | `mul_routed_weight` | `BLOCK_SIZE_M` |
|------|-----------|-------|-------------------|------|------|---------------------|----------------|
| `M8_E256_topk1.pt` | 8 | 1 | 8 | 2048 | 512 | True | 16 |
| `M128_E256_topk1.pt` | 128 | 1 | 128 | 2048 | 512 | True | 16 |
| `M16376_E256_topk1.pt` | 16376 | 1 | 16376 | 2048 | 512 | True | 64 |
| `M1_E256_topk8.pt` | 1 | 8 | 8 | 1024 | 2048 | **False** | 16 |
| `M16_E256_topk8.pt` | 16 | 8 | 128 | 1024 | 2048 | **False** | 16 |
| `M2047_E256_topk8.pt` | 2047 | 8 | 16376 | 1024 | 2048 | **False** | 64 |

> **关键**：6 个用例按 `mul_routed_weight` 分为两组，分别对应 fused MoE 的两个 GEMM 阶段（**推断依据**：`mul_routed_weight` 标志 + N/K 取值，与 rtp-llm executor 语义一致；golden 的 `mode` 字段只记录文件名，未显式标注阶段）——
> - `topk1`（`mul_routed_weight=True`，N=2048）：带路由权重的**第二段 GEMM**（`A @ w2.T`，输出维度回到隐藏维 K=512）；
> - `topk8`（`mul_routed_weight=False`，N=1024）：**第一段 GEMM**（`A @ w1.T` 输入投影，N=2*inter，`inter` 为中间维度）。
>
> 测试必须尊重每个文件自带的 `mul_routed_weight`，不能假设所有用例语义一致。

### 典型数据内容（以 `M8_E256_topk1.pt` 为例）

| key | 类型 | 值 |
|-----|------|----|
| `op_name` | str | `"invoke_fused_moe_kernel"` |
| `mode` | str | `"M8_E256_topk1"` |
| `param_names` | list[12] | 记录 12 个调用参数名 |
| `inputs/A` | tensor | shape=(8, 512), dtype=float16, contiguous |
| `inputs/B` | tensor | shape=(256, 2048, 512), dtype=float16, contiguous |
| `inputs/C` | tensor | shape=(8, 2048), dtype=float16 — inplace 输出 buffer |
| `inputs/topk_weights` | tensor | shape=(8,), dtype=**float32** — 扁平路由权重 |
| `inputs/topk_ids` | tensor | shape=(8,), dtype=int32 — 扁平 expert id |
| `inputs/sorted_token_ids` | tensor | shape=(4104,), dtype=int32 — 对齐后的扁平 token 置换 |
| `inputs/expert_ids` | tensor | shape=(256,), dtype=int32 — 每个 M 块的 expert |
| `inputs/num_tokens_post_padded` | tensor | shape=(1,), dtype=int32, 值=[128] |
| `inputs/mul_routed_weight` | bool | True |
| `inputs/top_k` | int | 1 |
| `inputs/config` | dict | `{BLOCK_SIZE_M:16, BLOCK_SIZE_N:128, BLOCK_SIZE_K:64, GROUP_SIZE_M:1, num_warps:8, num_stages:4}` |
| `inputs/compute_type` | str | `"fp16"` |
| `inplace_outputs/C` | tensor | shape=(8, 2048), dtype=float16 — **golden 结果** |
| `input_meta` | dict | 各输入 shape/stride/dtype/contiguous（本算子全部连续） |
| `outputs` | dict | 空（本算子是 inplace，输出在 `inplace_outputs.C`） |

## 2. 理解 kernel 语义（CPU reference）

kernel 用 `sorted_token_ids` 做 scatter，理解它的语义是比对正确的关键：

```
对每个填充行 i（i < num_tokens_post_padded）：
    flat   = sorted_token_ids[i]          # 扁平序号，范围 [0, M*top_k)
    token  = flat // top_k                # 回映到 A 的行
    expert = expert_ids[i // BLOCK_SIZE_M] # 该行所属 M 块的 expert
    C[flat] += (topk_weights[flat] if mul_routed_weight else 1.0) * A[token] @ B[expert].T
```

三个容易踩的语义点：

1. **`sorted_token_ids` 存的是扁平序号，不是 token 行号**。值域是 `[0, M*top_k)`，kernel 通过 `offs_token // top_k` 回映到 A 的行、`topk_weights[offs_token]` 取权重、`C[offs_token]` 写回。
2. **`expert_ids` 按 M 块对齐**：长度 = `max_padded // BLOCK_SIZE_M`，由 `moe_align_block_size_torch(topk_ids, block_size=BLOCK_SIZE_M, E)` 生成，`block_size` 必须等于 kernel 配置的 `BLOCK_SIZE_M`。
3. **`topk_weights` 按扁平序号索引，且只有 `mul_routed_weight=True` 才乘**。

CPU reference（与 kernel 语义逐条对应，用 fp64 累加以消除参考自身的舍入干扰）：

```python
def _cpu_reference(A, B, topk_weights, topk_ids, sorted_token_ids, expert_ids,
                   num_tokens_post_padded, block_size_m, top_k, mul_routed_weight):
    num_valid = topk_ids.numel()
    num_tokens = num_tokens_post_padded.item()
    out = torch.zeros(num_valid, B.shape[1], dtype=torch.float64)

    A_f = A.double()
    B_f = B.double()
    w_f = topk_weights.double()

    pos = torch.arange(num_tokens)
    flat = sorted_token_ids[:num_tokens]
    valid_mask = flat < num_valid                       # 过滤 padding 哨兵
    flat_v = flat[valid_mask].long()
    tok_v = flat_v // top_k                             # flat -> A 行
    exp_v = expert_ids[(pos[valid_mask] // block_size_m).long()]

    for e in range(B.shape[0]):                         # 按 expert 分组批量 matmul
        idx = torch.nonzero(exp_v == e).flatten()
        if idx.numel() == 0:
            continue
        o = A_f[tok_v[idx]] @ B_f[e].t()
        if mul_routed_weight:
            o = o * w_f[flat_v[idx]][:, None]
        out[flat_v[idx]] += o
    return out.float().half()
```

## 3. 理解算子的输入格式（kernel 约束）

直接读 rtp-llm `fused_moe_kernel.py` 的签名与寻址逻辑：

| 参数 | 期望 | 说明 |
|------|------|------|
| `A` | `(M, K)` fp16 | 输入 token，按 `offs_token // top_k` 寻址 |
| `B` | `(E, N, K)` fp16 | expert 权重，按 `off_experts * stride_be` 寻址 |
| `C` | `(M*top_k, N)` fp16 | **inplace** 输出，按 `sorted_token_ids` scatter 写回 |
| `topk_weights` | `(M*top_k,)` fp32 | 扁平路由权重，仅 `MUL_ROUTED_WEIGHT=True` 时乘 |
| `topk_ids` | `(M*top_k,)` int32 | 仅用于取 `numel()`（num_valid_tokens） |
| `sorted_token_ids` | `(max_padded,)` int32 | 扁平 token 置换，`stride(0)==1`（kernel 有断言） |
| `expert_ids` | `(max_blocks,)` int32 | 每 M 块的 expert |
| `num_tokens_post_padded` | `(1,)` int32 | 实际使用行数，kernel 据此跳过 padding 块 |
| `mul_routed_weight` | bool | 是否乘路由权重 |
| `top_k` | int | 每个 token 的 expert 数（constexpr） |
| `config` | dict | `BLOCK_SIZE_M/N/K`、`GROUP_SIZE_M`、`num_warps`、`num_stages` |
| `compute_type` | `tl.float16`/`tl.bfloat16` | 累加器写回前转到的 dtype |

### 关键点：`moe_align_block_size_torch` 的 `block_size`

`expert_ids` 必须按 **`config["BLOCK_SIZE_M"]`** 对齐（rtp-llm executor 内也是传 `config["BLOCK_SIZE_M"]`）：

```python
sorted_token_ids, expert_ids, num_tokens_post_padded = moe_align_block_size_torch(
    topk_ids, block_m1, self.E)   # block_m1 = config1["BLOCK_SIZE_M"]
```

若 `block_size` 传错（如 64，而 kernel `BLOCK_SIZE_M=16`），`expert_ids` 长度（`max_padded // block_size`）会对不上 kernel 的 M 块数量（`cdiv(max_padded, BLOCK_SIZE_M)`），导致 kernel 越界读到垃圾 expert、输出错误（详见 5.1）。

## 4. 格式转换映射

GPU 与 NPU（rtp-llm Triton for Ascend）**布局完全一致，无需任何 transpose**。golden 已存好对齐后的路由张量，直接使用：

```python
import triton.language as tl
compute_type = tl.float16 if inputs["compute_type"] == "fp16" else tl.bfloat16

invoke_fused_moe_kernel(
    A.npu(), B.npu(), C_actual,            # C_actual = inputs["C"].clone().npu()
    topk_weights.npu(), topk_ids.npu(),
    sorted_token_ids.npu(), expert_ids.npu(), num_tokens_post_padded.npu(),
    bool(inputs["mul_routed_weight"]),
    int(inputs["top_k"]),
    dict(inputs["config"]),
    compute_type,
)
torch.npu.synchronize()
```

需要做的转换只有两类：**(1) 全部 `.npu()`**；**(2) `compute_type` 字符串 → `tl` dtype**。`config` 直接透传 golden 里的 dict。

## 5. 常见错误

### 5.1 `moe_align_block_size_torch` 的 `block_size` 与 `BLOCK_SIZE_M` 不一致

```python
# ✗ 错误：block_size=64，kernel 的 BLOCK_SIZE_M=16，expert_ids 长度对不上 M 块数量
sorted_token_ids, expert_ids, _ = moe_align_block_size_torch(topk_ids, 64, E)

# ✓ 正确：block_size = config["BLOCK_SIZE_M"]
sorted_token_ids, expert_ids, _ = moe_align_block_size_torch(topk_ids, config["BLOCK_SIZE_M"], E)
```

症状：第一个 expert 块的输出正常，第二个及以后的块输出错乱（越界读 `expert_ids` 得到垃圾 expert）。**最稳妥的做法是直接用 golden 里现成的路由张量，不自己对齐。**

### 5.2 CPU reference 漏掉 `flat // top_k`

`topk>1` 时 `sorted_token_ids` 是扁平序号，直接拿它索引 `A` 会越界：

```python
# ✗ 错误：topk=8 时 sorted_token_ids 最大到 M*topk-1，超出 A 的行数 (M)
o = A[flat] @ B[e].t()          # IndexError

# ✓ 正确：先 flat // top_k 回映到 A 的行
o = A[flat // top_k] @ B[e].t()
```

### 5.3 忽略 `mul_routed_weight` 一律乘权重

`topk8` 用例是 GEMM1（`mul_routed_weight=False`），CPU reference / 比对若一律乘 `topk_weights`，max_abs_diff 会到 ~4-6（权重量级），直接失败：

```python
# ✗ 错误：无条件乘权重
out[flat] += o * w[flat][:, None]

# ✓ 正确：按 flag 条件乘
if mul_routed_weight:
    o = o * w[flat][:, None]
out[flat] += o
```

### 5.4 直接 `import rtp_llm`

`rtp_llm.__init__` 会触发重 C++ 依赖（`libth_transformer_config.so`）。必须用 stub 包层次 + `importlib` 只加载 `fused_moe_kernel.py` 单文件（该文件仅依赖 `torch`/`triton`）。

### 5.5 把 `C` 的行号当成 token 号

`C[flat]` 的 `flat` 是扁平序号（含 topk 维），不是 token 行号。`topk=1` 时二者恰好相等，容易掩盖这个错误；`topk>1` 时必然出错。

## 6. 测试模板

完整测试文件位于 `/home/s60130915/work/test_npu_invoke_fused_moe_kernel_gpu_golden.py`。

```python
import importlib.util, os, sys, types, unittest
import torch

torch.npu.set_device(int(os.environ.get("TEST_DEVICE_ID", 0)))

def _load_rtp_llm_moe_modules():
    # stub 包层次 + importlib，绕过 rtp_llm.__init__ 的 C++ 依赖
    # （fused_moe_kernel.py 仅依赖 torch/triton，加载单文件即可）
    ...
invoke_fused_moe_kernel = _load_rtp_llm_moe_modules().invoke_fused_moe_kernel

class TestInvokeFusedMoeKernelGpuGolden(unittest.TestCase):
    rtol = 1e-2
    atol = 1e-2

    def _run_case(self, filename):
        # 解析 inputs / inplace_outputs.C
        # → CPU reference 自检 golden（防损坏）
        # → 用 golden 路由张量调 invoke_fused_moe_kernel
        # → torch.npu.synchronize() → 比对 C
        ...
```

六个用例覆盖 topk1（GEMM2）与 topk8（GEMM1）两个阶段、decode（M=1）与 prefill（M=16376）两种规模。

## 7. 调试技巧

1. **先打印所有 key/shape**：确认输出在 `inplace_outputs.C`（本算子是 inplace），`outputs` 为空
2. **打印 `sorted_token_ids` 的前几项**：值域是 `[0, M*top_k)`（扁平序号），`num_valid` 作 padding 哨兵
3. **打印 `mul_routed_weight`/`top_k`/`config`**：判断用例属于 GEMM1 还是 GEMM2，别猜
4. **先跑 CPU reference**：golden 自检不过就说明要么 dump 损坏、要么参考语义写错，别急着上 NPU
5. **读 kernel 源码**：确认 `offs_token // top_k`、`expert_ids[pid_m]`、`MUL_ROUTED_WEIGHT` 三处寻址逻辑
6. **小 shape 复现**：出问题时构造 `(M=4, E=2, N=4, K=8, topk=1)` 的合成 case 缩小范围

## 8. GPU 与 NPU 布局策略对比

| 维度 | GPU dump | NPU（rtp-llm Triton for Ascend） |
|------|---------|------------------|
| `A` / `B` 布局 | `(M,K)` / `(E,N,K)` fp16 连续 | 同，无需转换 |
| 路由张量 | golden 已存好对齐后的 `sorted_token_ids/expert_ids/num_tokens_post_padded` | 直接使用（避免自己对齐出错） |
| `topk_weights` | `(M*topk,)` fp32 | 同，kernel 内 fp32 累加后按 `compute_type` 写回 |
| 输出 `C` | `(M*topk, N)` fp16，**inplace** scatter | 同，inplace 写回 `inplace_outputs.C` |
| layout 转置 | 无 | **无需转置** |
| 权重乘入 | 由 `mul_routed_weight` 决定 | kernel 内 `MUL_ROUTED_WEIGHT` constexpr 控制 |
| 算子来源 | — | rtp-llm `fused_moe_kernel.py`（stub 包 + importlib 加载，源码零修改） |

## 9. 实测结果

在本环境（昇腾 NPU）下运行测试，输出与 GPU golden 的逐用例比对数据：

| 用例 | M | top_k | `mul_routed_weight` | `C` max_abs_diff | `C` mean_abs_diff |
|------|---|-------|---------------------|------------------|-------------------|
| `M8_E256_topk1` | 8 | 1 | True | 3.8e-6 | 8.0e-10 |
| `M128_E256_topk1` | 128 | 1 | True | 7.6e-6 | 8.5e-10 |
| `M16376_E256_topk1` | 16376 | 1 | True | 3.1e-5 | 8.4e-10 |
| `M1_E256_topk8` | 1 | 8 | False | 4.9e-4 | 6.3e-7 |
| `M16_E256_topk8` | 16 | 8 | False | 3.9e-3 | 5.6e-7 |
| `M2047_E256_topk8` | 2047 | 8 | False | 3.9e-3 | 5.5e-7 |

### 误差来源分析

kernel 内部用 **fp32 累加**（`accumulator = tl.zeros(..., dtype=tl.float32)`），仅在写回前 `.to(compute_type)` 转 fp16。误差来源主要有两类：

1. **fp16 输入量化**：`A`/`B` 本身是 fp16，相对误差约 $2^{-11}\approx4.9\times10^{-4}$（1 ULP）；
2. **`tl.dot` 归约顺序差异**：GPU 与 NPU 的 K 维累加顺序不同。由于中间用 fp32 累加、且最后只做一次 fp16 写回舍入，误差**不会**随 K 累积放大，整体相对误差应保持在几个 fp16 ULP（~$10^{-4}$）量级。

逐用例看与实测的对应关系：

- `topk1`（GEMM2）输出乘了路由权重（~0.1~0.3），数值整体偏小，`max_abs_diff` 仅 3.8e-6 ~ 3.1e-5；
- `topk8`（GEMM1）输出是未乘权重的原始 GEMM 结果，个别元素绝对值较大，`max_abs_diff` 达 3.9e-3，但相对误差仍在 ~$5\times10^{-4}$ 以内（几个 fp16 ULP）。

> **容差选择**：测试用 `rtol=atol=1e-2`，约为最坏相对误差的 20 倍，余量充足；同时比 AscendC 算子惯用的 `5e-2` 更严格——因为本算子与 GPU 是**同一份 Triton 源码**，不存在接口/实现差异，误差应逼近纯 fp16 舍入下界。若未来替换为 AscendC 或独立实现，需重新评估容差。

**结论**：`invoke_fused_moe_kernel` 是 fused MoE GEMM 算子（inplace scatter 写回），其 NPU 实现直接复用 rtp-llm 的 Triton kernel，源码零修改。比对的核心难点不在 layout 转换（GPU/NPU 布局一致、路由张量直接用 golden），而在三点——(1) 理解 `sorted_token_ids` 是扁平序号、须 `flat // top_k` 回映 A 行；(2) `expert_ids` 按 `BLOCK_SIZE_M` 对齐（自己对齐易越界）；(3) 尊重 `mul_routed_weight`（区分 GEMM1/GEMM2 两个阶段）。实测 6 个用例的相对误差均在几个 fp16 ULP 内，验证通过。
