# npu_moe_gating_top_k 算子 GPU-NPU 精度比对指南

本文档总结 MoE 路由 top-k 门控 softmax 算子 **NPU 端 `torch_npu.npu_moe_gating_top_k`**（GPU 端对应 rtp-llm `topkGatingSoftmaxKernelLauncher` / `SelectTopkOp`）的接口差异与精度比对经验。全部用例统一用 `npu_moe_gating_top_k`（默认参数）承载两种 `has_moe_norm` 模式。本算子**无 GPU golden .pt**，测试采用**自行构造输入**、以精确复刻 GPU kernel 语义的 CPU reference 作为基准。组织方式参考 `gpu_npu_comparison_guide.md` 的通用方法论。

## 整体流程

```
理解 GPU kernel 语义 (topkGatingSoftmax)  →  理解 NPU 接口 (npu_moe_gating_top_k 为主)  →  对齐输入 shape (M,E) fp32  →  构造用例  →  CPU reference 复刻 GPU 语义  →  调 npu_moe_gating_top_k（RENORM 加 y/sum(y) 后处理）→  比对 y / expert_idx / norm_out
```

## 1. GPU 侧实现

### 调用链

```
rtp-llm python: SelectTopk.forward(router_logits_fp32, topk_ids, topk_weights)
  └─ compute_ops.SelectTopkOp.forward(...)          # C++ pybind, SelectTopkOp.cc
       └─ invokeSelectExpertsForTokens<int32/int64> # tensorrt_llm kernels
            └─ topkGatingSoftmaxKernelLauncher      # moe_routing_kernels.cu
```

`SelectTopkOp` 由 `ModelConfig` 构造，成员 `expert_num_` / `moe_k_` / `has_moe_norm_`。

### kernel 语义（`topkGatingSoftmax`，精确复刻目标）

对每个 token 行（共 `num_rows`）：

1. **softmax over 所有 expert**（fp32）：先取行内 max 做指数稳定化，`softmax[i] = exp(x[i]-max)/sum_j exp(x[j]-max)`；
2. **顺序 arg-max top-k**（`k_idx = 0..k-1`）：每轮选当前剩余中的最大 softmax 值，写入 `output[k*row+k_idx]`、`indices[k*row+k_idx] = expert`，并把该 expert 排除（融合路径把其 softmax 概率置 `-10000`；默认路径跳过已选 expert，**机制不同、语义相同**）；
   - **tie 规则**：`if (val > max_val)` 严格大于 + `cub::ArgMax`（值相等取更小 key）→ **并列时取最小 expert 索引**；
3. **`has_moe_norm=True`（`MOEExpertScaleNormalizationMode::RENORMALIZE`）**：把选出的 k 个权重除以它们的和（`output[idx] *= 1/sum`），使其和为 1；`False`（`NONE`）则直接保留 softmax 概率。

> 路径分叉（**仅 GPU 端**）：E 为 2 的幂（1..256）走融合 `topkGatingSoftmax`，否则走 `moeSoftmax`+`moeTopK`；二者语义一致，两类 E 都要覆盖。NPU 端无此分叉：实测 E=255/256/257/300/512/1024/2048 精度一致（~1e-8）——`256` 只是 GPU 的编译期模板优化，非语义差异。

### `SelectTopkOp.forward` 接口

```cpp
void forward(torch::Tensor router_logits, torch::Tensor expert_ids, torch::Tensor expert_scales)
```
- `router_logits`：`(token_num, num_expert)` **float32、contiguous**（函数内 `.contiguous()`）；
- `expert_ids`：输出，`(token_num, top_k)`，**int32 或 int64**（按该 tensor 的 dtype 分派）；
- `expert_scales`：输出，`(token_num, top_k)`，**float32**（softmax 概率，可选 renorm）；
- 内部还有全量 `softmax_out (token_num, num_expert)` fp32 与 `source_rows (token_num, top_k)` int32（`source_rows[k*row+k_idx] = k_idx*num_rows + row`），**不返回 Python**。

## 2. NPU 侧接口（torch_npu）

| 接口 | 原型 | 语义 |
|------|------|------|
| `torch_npu.npu_moe_gating_top_k` | `(x, k, bias=None, k_group=1, group_count=1, group_select_mode=0, renorm=0, norm_type=0, out_flag=False, routed_scaling_factor=1.0, eps=1e-20) -> (y, expert_idx, norm_out)` | norm（softmax/sigmoid）+ 分组排序 topk；**默认参数退化为 softmax→topk 不归一**；`renorm` **仅支持 0**；`E≤2048` |
| （参考）`npu_moe_gating_top_k_softmax` / `_v2` | softmax-only 变体，本测试未使用 | v2 `renorm=1` 可表达 RENORMALIZE；本方案统一用 `npu_moe_gating_top_k` + 后处理 |

> **调用风格**：`torch_npu.npu_moe_gating_top_k` 等（模块级）是官方文档写法（`_op_plugin_docs.py` 示例），也可用 `torch.ops.npu.*`；**`torch.npu.*` 不存在**（会 `AttributeError`）。`npu_moe_gating_top_k` 除 `x`/`k` 外参数均 **keyword-only**。

- `x` 为 **2D**、fp16/bf16/fp32、ND、**E≤2048**；`k`：`1<=k<=E/group_count*k_group`；`bias` 可选加在 x 上；`group_count`/`k_group` 分组路由；`norm_type` 0=softmax / 1=sigmoid；`renorm` **仅支持 0**；`out_flag=True` 才输出 `norm_out`；`routed_scaling_factor`/`eps` 缩放。返回 `y`（权重，**dtype 与 x 相同**）、`expert_idx`（**int32**）、`norm_out`（`(M,E)` fp32）。

### 关键：`has_moe_norm` 的 NPU 映射

GPU `NONE` = softmax 后取 topk 不归一，等价于 `npu_moe_gating_top_k` 默认输出的 `y`（raw softmax topk 概率）。GPU `RENORMALIZE` = softmax 后取 topk 再归一化（kernel 里 `output *= 1/sum`），等价于对 `y` 做后处理 `y/y.sum(-1)`（数学上即"对选出的 k 个 softmax 值归一化"；`npu_moe_gating_top_k` 的 `renorm` 参数仅支持 0，故由测试侧完成）。

| GPU `has_moe_norm` | GPU 模式 | NPU 调用 |
|---|---|---|
| `False` | `NONE`（softmax→topk，权重不归一） | `torch_npu.npu_moe_gating_top_k(x, k, out_flag=True)`（默认参数，`y` 为 raw softmax 概率） |
| `True` | `RENORMALIZE`（topk 权重和=1） | `torch_npu.npu_moe_gating_top_k(x, k, out_flag=True)` + 后处理 `y = y / y.sum(-1)`（等价 GPU RENORMALIZE；该算子 `renorm` 参数仅支持 0） |

> **注意**：`norm_out` 仅在 `out_flag=True` 时有效（默认 `out_flag=False` 返回未初始化缓冲）。

## 3. 接口差异对照（GPU vs NPU）

| 维度 | GPU（`SelectTopkOp` / `topkGatingSoftmaxKernelLauncher`） | NPU（`npu_moe_gating_top_k`） |
|------|----------------------------------------------------------|------------------|
| 输入 dtype | float32 固定（内部 fp32 计算） | fp16/bf16/fp32（dtype 跟随输入） |
| 输入 shape | `(token_num, num_expert)` 2D、contiguous | 2D `(M,E)`、ND；测试对齐为 2D `(M, E)` |
| `topk`（k） | `moe_k_` 配置，无显式上限（kernel 模板限制） | `1<=k<=E/group_count*k_group` |
| 权重输出 dtype | `expert_scales` 恒 **float32** | `y` **dtype = x.dtype** |
| 索引输出 dtype | `expert_ids` **int32 或 int64**（分派） | `expert_idx` 恒 **int32** |
| 输出方式 | **inplace 写入** `expert_ids` / `expert_scales`，无返回值 | **返回** `(y, expert_idx, norm_out)` |
| renorm | `has_moe_norm` → `RENORMALIZE`（topk 权重和=1） | 算子 `renorm` 参数仅支持 0；测试侧对 `y` 做 `y/y.sum(-1)` 实现 RENORMALIZE（数学等价） |
| 全量 softmax | 内部 `softmax_out`，不返回 | `norm_out`（`out_flag=True`，`(M,E)` fp32，实测≈`softmax(x)` 差 ~1.5e-8） |
| `source_rows` | 内部，不返回（`k_idx*M+row`） | 无对应；第 3 个返回是 `norm_out`，非 row_idx |
| tie 规则 | 并列取**最小 expert 索引** | 一致（实测 tie diff=0） |

## 4. 构造用例（对齐 GPU 输入 shape）

GPU 输入是 `(token_num, num_expert)` fp32 连续，因此 NPU 测试统一用 2D `(M, E)` fp32。用例覆盖：

- **E 为 2 的幂**（GPU 走融合 `topkGatingSoftmax`）：`E=8/64/128/256`；
- **E 非 2 的幂**（GPU 走 `moeSoftmax+moeTopK` 默认路径）：`E=10/100/300`；
- **token 规模**：decode `M=1`、小 batch `M=16`、prefill `M=2047`；
- **k**：`1/2/3/4/8`；
- 每种 shape × `{has_moe_norm=False, True}` 两种模式；
- 附加：`fp16` 输入用例（验证 y dtype 跟随输入）、全零 logits tie 用例（验证最小索引）。

随机 logits 用基于 tag 的稳定 seed 生成（`hashlib.md5`，**非内置 `hash()`**——后者对字符串每进程随机加盐、跨进程不可复现），保证可复现。

## 5. CPU reference（精确复刻 GPU 语义）

```python
def _gpu_topk_softmax_ref(x, k, has_moe_norm):
    M, E = x.shape
    soft = torch.softmax(x.double(), dim=-1)     # fp64 参考
    weights = torch.zeros(M, k, dtype=torch.float64)
    indices = torch.zeros(M, k, dtype=torch.long)
    work = soft.clone()
    for ki in range(k):                          # 顺序 argmax，与 GPU 一致
        maxv, argi = work.max(dim=-1)            # torch first-max = tie 取最小索引
        weights[:, ki] = maxv
        indices[:, ki] = argi
        work.scatter_(1, argi.unsqueeze(1), -float("inf"))
    if has_moe_norm:
        weights = weights / weights.sum(dim=-1, keepdim=True)   # RENORMALIZE
    return weights.float(), indices
```

要点：
1. **顺序 argmax** 而非 `torch.topk`——`torch.topk` 的 tie 行为不保证最小索引，顺序 argmax + `torch.max(dim=-1)`（返回首个最大值 = 最小索引）与 GPU 的 `cub::ArgMax` 严格匹配；
2. **fp64 计算**消除参考自身舍入，作为高精度基准；
3. `has_moe_norm=True` 时按 GPU 语义"softmax 后 topk 再归一化"实现（与测试侧对 `y` 做 `y/y.sum(-1)` 后处理一致，数学上等价，数值差在 ~1e-7）。

## 6. 常见错误

### 6.1 用 `torch.topk` 当 GPU 参考
`torch.topk` 的 tie 处理不保证最小索引；必须用**顺序 argmax**（`torch.max(dim=-1)` 首现最大）复刻 GPU `cub::ArgMax`。

### 6.2 漏掉 RENORMALIZE 的后处理归一化
`npu_moe_gating_top_k` 默认输出 raw softmax 概率（不归一，wsum<1），对应 GPU `NONE`。`has_moe_norm=True` 时**必须对 `y` 做 `y/y.sum(-1)` 后处理**（权重和=1）；漏做则权重差一个归一化因子。该算子 `renorm` 参数仅支持 0，不能靠它开归一化。

### 6.3 忽略 dtype 差异
NPU `y` dtype **跟随输入**（fp16 输入 → fp16 输出），而 GPU `expert_scales` 恒 fp32。比对时把参考转到 `y.dtype` 再比数值；不要在断言里强制两边 dtype 相等。

### 6.4 拿第 3 个返回当 token 行号
`npu_moe_gating_top_k` 第 3 个返回是 **`norm_out`**（全量 softmax `(M,E)` fp32，**须 `out_flag=True`**），不是 token 行号，也不是 GPU 内部 `source_rows`。

### 6.5 传非 contiguous 或错误 dtype 输入
GPU 侧 `router_logits` 会 `.contiguous()`；NPU 要求 ND 格式、支持 fp16/bf16/fp32。构造用例统一用 fp32 contiguous（与 GPU 对齐）。

## 7. 测试模板

完整测试文件位于同目录 `test_npu_moe_gating_top_k_npu_gpu.py`。

```python
import os, unittest
import torch, torch_npu

torch.npu.set_device(int(os.environ.get("TEST_DEVICE_ID", 0)))

def _gpu_topk_softmax_ref(x, k, has_moe_norm):
    ...  # §5 的顺序 argmax 参考

def _npu_moe_gating_topk(x_npu, k, has_moe_norm):
    y, ei, third = torch_npu.npu_moe_gating_top_k(x_npu, k, out_flag=True)
    if has_moe_norm:
        y = y / y.sum(dim=-1, keepdim=True)   # post-hoc RENORMALIZE (== GPU)
    return y, ei, third

class TestNpuMoeGatingTopKNpuVsGpu(unittest.TestCase):
    rtol = atol = 1e-2
    def _run_case(self, tag, shape, k, has_moe_norm, x_dtype=torch.float32):
        x_cpu = torch.randn(shape, dtype=torch.float32)
        if x_dtype != torch.float32:
            x_cpu = x_cpu.to(x_dtype)
        w_ref, idx_ref = _gpu_topk_softmax_ref(x_cpu.float(), k, has_moe_norm)
        y, ei, third = _npu_moe_gating_topk(x_cpu.npu(), k, has_moe_norm)
        torch.npu.synchronize()
        # 接口断言: y.dtype==x.dtype, ei.dtype==int32, shape==(M,k)
        # 数值断言: y≈w_ref(y.dtype), ei==idx_ref, renorm 时 sum(y,-1)≈1
        # 3rd: npu_moe_gating_top_k 的 norm_out(M,E)≈softmax(x)（两种模式一致）
```

## 8. 调试技巧

1. **先跑小 shape**：`(M=4, E=8, k=2)` 快速验证语义，再上 prefill 规模
2. **区分两条 GPU 路径**：E 为 2 的幂（融合 kernel）与非 2 的幂（softmax+topk 两 kernel）都要覆盖
3. **验证 renorm 求和**：`has_moe_norm=True` 时打印 `y.sum(-1)`，应 ≈1（后处理 `y/y.sum(-1)` 生效）
4. **验证 tie**：全零 logits → softmax 均匀，选中索引应为 `[[0,1],...]`（最小索引）
5. **注意 keyword-only 参数**：`npu_moe_gating_top_k` 除 `x`/`k` 外参数都要用关键字传参，位置参数报错
6. **`norm_out` 别当行号用**：第 3 个返回是 `norm_out`（须 `out_flag=True`），见 §3/§6.4
7. **看 torch_npu 文档**：`_op_plugin_docs.py` 中 `npu_moe_gating_top_k` 的原型与约束

## 9. 实测结果

在本环境（昇腾 NPU）下运行测试（fp32 输入），NPU op 与 GPU 语义参考（fp64 参考）的比对：

| 模式 | 用例 | M | E | k | `y` max_abs_diff | `expert_idx` 一致 | 第 3 个返回 |
|------|------|---|---|---|------------------|-------------------|-----------|
| NONE（`npu_moe_gating_top_k`） | decode | 1 | 8 | 2 | 0.0 | ✓ | `norm_out`(M,E)≈softmax |
| NONE | batch_pow2 | 16 | 64 | 4 | 7.45e-9 | ✓ | `norm_out`(M,E)≈softmax |
| NONE | prefill_pow2 | 2047 | 256 | 8 | 1.49e-8 | ✓ | `norm_out`(M,E)≈softmax |
| NONE | batch_nonpow2 | 16 | 10 | 3 | 5.96e-8 | ✓ | `norm_out`(M,E)≈softmax |
| NONE | prefill_nonpow2 | 2047 | 100 | 4 | 2.98e-8 | ✓ | `norm_out`(M,E)≈softmax |
| NONE | prefill_E300 | 2047 | 300 | 8 | 1.49e-8 | ✓ | `norm_out`(M,E)≈softmax |
| NONE | topk1 | 16 | 128 | 1 | 7.45e-9 | ✓ | `norm_out`(M,E)≈softmax |
| RENORM（`npu_moe_gating_top_k`+`y/sum(y)`） | decode | 1 | 8 | 2 | 0.0 | ✓ | `norm_out`(M,E)≈softmax |
| RENORM | batch_pow2 | 16 | 64 | 4 | 5.96e-8 | ✓ | `norm_out`(M,E)≈softmax |
| RENORM | prefill_pow2 | 2047 | 256 | 8 | 5.96e-8 | ✓ | `norm_out`(M,E)≈softmax |
| RENORM | batch_nonpow2 | 16 | 10 | 3 | 5.96e-8 | ✓ | `norm_out`(M,E)≈softmax |
| RENORM | prefill_nonpow2 | 2047 | 100 | 4 | 5.96e-8 | ✓ | `norm_out`(M,E)≈softmax |
| RENORM | prefill_E300 | 2047 | 300 | 8 | 5.96e-8 | ✓ | `norm_out`(M,E)≈softmax |
| RENORM | topk1 | 16 | 128 | 1 | 0.0（k=1 归一化后恒为 1.0） | ✓ | `norm_out`(M,E)≈softmax |
| tie 用例 | — | 4 | 8 | 2 | 0.0（完全一致） | ✓ | — |

### 误差来源分析

本算子**无 K 维长归约**：每行只做一次 softmax（行内 max 稳定化 + 求和）+ 顺序 topk。误差来源仅两个 fp32 舍入环节：

1. **softmax 的 exp/sum**：NPU 与 GPU 在 fp32 下实现，`expf` 实现差异导致 ~1e-7 量级绝对误差；
2. **renorm 除法**（仅 RENORMALIZE）：`1/sum` 与逐项乘的舍入，~1e-7。

实测 `y` 最大绝对误差 ≤1.2e-7，远小于测试容差 `rtol=atol=1e-2`（约 5 个数量级余量）。tie 用例完全一致（diff=0）。`expert_idx` 全部用例逐元素相等。

> **容差选择**：因与 GPU 是"同一算法、独立实现"，误差来自 fp32 舍入而非语义差异，`1e-2` 余量充足；若用 fp16 输入，误差上限约 1e-3（fp16 softmax 舍入），`1e-2` 仍通过。

**结论**：`topkGatingSoftmaxKernelLauncher`（`SelectTopkOp`）的 NPU 等价实现是 torch_npu **`npu_moe_gating_top_k`**（默认参数）——`has_moe_norm=False` 直接用其 `y`（raw softmax topk 概率），`has_moe_norm=True` 对其 `y` 做 `y/y.sum(-1)` 后处理归一化（数学等价 GPU RENORMALIZE），**全部用例统一走该算子**。比对的核心难点不在数值（fp32 舍入下界 ~1e-7），而在**接口差异**：(1) RENORMALIZE 的表达方式（算子 `renorm` 仅支持 0 → 测试侧后处理 `y/sum(y)`）；(2) NPU `y` dtype 跟随输入、GPU `expert_scales` 恒 fp32；(3) NPU `expert_idx` 恒 int32、GPU 支持 int32/int64；(4) `npu_moe_gating_top_k` 第 3 个返回 `norm_out`（`out_flag=True`，`(M,E)` fp32）；(5) GPU 为 inplace 写、NPU 为返回三元组。测试用自行构造的用例（对齐 `(M,E)` fp32、覆盖 2 的幂与非 2 的幂 E、decode/prefill 规模、fp16 与 tie 边界）全部通过。
