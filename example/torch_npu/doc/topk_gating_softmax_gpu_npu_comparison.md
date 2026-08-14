# topkGatingSoftmaxKernelLauncher 算子 GPU-NPU 精度比对指南

本文档总结 MoE 路由 top-k 门控 softmax 算子（GPU 端 rtp-llm `topkGatingSoftmaxKernelLauncher` / `SelectTopkOp`）与 NPU 端（torch_npu `npu_moe_gating_top_k_softmax` / `_v2`）的接口差异与精度比对经验。本算子**无 GPU golden .pt**，测试采用**自行构造输入**、以精确复刻 GPU kernel 语义的 CPU reference 作为基准。组织方式参考 `gpu_npu_comparison_guide.md` 的通用方法论。

## 整体流程

```
理解 GPU kernel 语义 (topkGatingSoftmax)  →  理解 NPU 接口 (v1/v2)  →  对齐输入 shape (M,E) fp32  →  构造用例  →  CPU reference 复刻 GPU 语义  →  调 NPU op  →  比对 y / expert_idx / row_idx
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
2. **顺序 arg-max top-k**（`k_idx = 0..k-1`）：每轮选当前剩余中的最大 softmax 值，写入 `output[k*row+k_idx]`、`indices[k*row+k_idx] = expert`，并把该 expert 置为 `-10000` 排除；
   - **tie 规则**：`if (val > max_val)` 严格大于 + `cub::ArgMax`（值相等取更小 key）→ **并列时取最小 expert 索引**；
3. **`has_moe_norm=True`（`MOEExpertScaleNormalizationMode::RENORMALIZE`）**：把选出的 k 个权重除以它们的和（`output[idx] *= 1/sum`），使其和为 1；`False`（`NONE`）则直接保留 softmax 概率。

> 路径分叉（**仅 GPU 端**）：`num_experts` 为 2 的幂（1..256）走融合 `topkGatingSoftmax`；否则走 `moeSoftmax` + `moeTopK` 默认路径。二者语义一致，测试两类 E 都要覆盖。**NPU 端（v1/v2）没有这个 256 分叉**：官方文档无 E 相关阈值（仅 `0<k<=E`、`k<=1024`），且实测 E=255/256/257/300/512/1024/2048 精度一致（~1e-8）、v1≡v2、索引全对——`256` 只是 GPU 的编译期模板性能优化，不是语义差异。

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
| v1 `torch_npu.npu_moe_gating_top_k_softmax` | `(x, finished=None, k) -> (y, expert_idx, row_idx)` | **先 softmax 后 topk**（不归一） |
| v2 `torch_npu.npu_moe_gating_top_k_softmax_v2` | `(x, *, k=1, finished=None, renorm=0, output_softmax=False) -> (y, expert_idx, row_idx)` | `renorm=0` 先 softmax 后 topk；`renorm=1` 先 topk 后 softmax |

> **调用风格**：`torch_npu.npu_moe_gating_top_k_softmax`（模块级）是官方文档写法（`_op_plugin_docs.py` 示例），也可用 `torch.ops.npu.npu_moe_gating_top_k_softmax`；**`torch.npu.*` 不存在**（会 `AttributeError`）。v2 参数是 **keyword-only**。

- `x`：2D/3D，**float16 / bfloat16 / float32**，ND 格式；`k`：`0 < k <= 最后一维`，`k <= 1024`；
- `finished`：可选 bool 掩码（shape `x[:-1]`），`None` = 全部行参与；
- 返回：`y`（topk 权重，**dtype 与 x 相同**）、`expert_idx`（**int32**）、`row_idx`。

### 关键：`has_moe_norm` 的 NPU 映射

GPU `RENORMALIZE` = softmax 后取 topk 再归一化 ≡ 对选出的 k 个值做 softmax（数学等价：公共分母约掉）→ 正是 **v2 `renorm=1`**（先 topk 后 softmax）。而 GPU `NONE` = softmax 后取 topk 不归一 → **v1**（或 v2 `renorm=0`）。

| GPU `has_moe_norm` | GPU 模式 | NPU 调用 |
|---|---|---|
| `False` | `NONE`（softmax→topk，权重不归一） | `torch_npu.npu_moe_gating_top_k_softmax(x, None, k)` |
| `True` | `RENORMALIZE`（topk 权重和=1） | `torch_npu.npu_moe_gating_top_k_softmax_v2(x, k=k, renorm=1)` |

> **一致性**：v2 `renorm=0` 与 v1 输出一致（测试 `test_v2_renorm0_equals_v1` 断言）。v2 第三个返回是**重载**的：`renorm=0` + `output_softmax=True` 时输出全量 softmax `(M,E)`（测试 `test_v2_output_softmax_structure` 断言），`renorm=1` 时 `output_softmax` 无效果（3rd 仍空）。

## 3. 接口差异对照（GPU vs NPU）

| 维度 | GPU（`SelectTopkOp` / `topkGatingSoftmaxKernelLauncher`） | NPU（torch_npu） |
|------|----------------------------------------------------------|------------------|
| 输入 dtype | float32 固定（内部 fp32 计算） | fp16/bf16/fp32（dtype 跟随输入） |
| 输入 shape | `(token_num, num_expert)` 2D、contiguous | 2D/3D，ND；测试对齐为 2D `(M, E)` |
| `topk`（k） | `moe_k_` 配置，无显式上限（kernel 模板限制） | `0<k<=E`、`k<=1024` |
| 权重输出 dtype | `expert_scales` 恒 **float32** | `y` **dtype = x.dtype** |
| 索引输出 dtype | `expert_ids` **int32 或 int64**（分派） | `expert_idx` 恒 **int32** |
| 输出方式 | **inplace 写入** `expert_ids` / `expert_scales`，无返回值 | **返回** `(y, expert_idx, row_idx)` |
| renorm | `has_moe_norm` → `RENORMALIZE`（topk 权重和=1） | v2 `renorm=1`（先 topk 后 softmax） |
| 全量 softmax | 内部 `softmax_out`，不返回 | v2 第三个返回值**重载**：`renorm=0` + `output_softmax=True` 时为全量 softmax `(M,E)`（实测与 `softmax(x)` 差 ~6e-8）；其余组合恒为空 `(0,)` |
| `row_idx` / `source_rows` | `source_rows` 内部，不返回（`k_idx*M+row`） | v1 返回 `row_idx`（同 `k_idx*M+row` 约定）；**v2 当前构建下恒为空 `(0,)`** |
| finished 掩码 | 无（全部行 active） | `finished` 可选（`None`=全部） |
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
3. `has_moe_norm=True` 时按 GPU 语义"softmax 后 topk 再归一化"实现（与 v2 renorm=1 的"topk 后 softmax"数学等价，数值差在 ~1e-7）。

## 6. 常见错误

### 6.1 用 `torch.topk` 当 GPU 参考
`torch.topk` 的 tie 处理不保证最小索引；必须用**顺序 argmax**（`torch.max(dim=-1)` 首现最大）复刻 GPU `cub::ArgMax`。

### 6.2 用错 v1/v2 或 renorm 取值
`has_moe_norm=True` 必须用 **v2 `renorm=1`**（权重和=1）；v1 和 v2 `renorm=0` 都不归一。混淆后权重差一个归一化因子。

### 6.3 忽略 dtype 差异
NPU `y` dtype **跟随输入**（fp16 输入 → fp16 输出），而 GPU `expert_scales` 恒 fp32。比对时把参考转到 `y.dtype` 再比数值；不要在断言里强制两边 dtype 相等。

### 6.4 拿 `row_idx` 当 token 行号
NPU v1 `row_idx` 用的是 GPU `source_rows` 约定 **`k_idx*M + row`**（k 主序），**不是**简单的 token 行号。v2 的 `row_idx` 在该构建下**恒为空** `(0,)`，不要对其做 shape/值断言。**注意 v2 第三个返回是重载的**：`renorm=0` + `output_softmax=True` 时它变成全量 softmax `(M,E)`，别再当 row_idx 用。

### 6.5 传非 contiguous 或错误 dtype 输入
GPU 侧 `router_logits` 会 `.contiguous()`；NPU 要求 ND 格式、支持 fp16/bf16/fp32。构造用例统一用 fp32 contiguous（与 GPU 对齐）。

### 6.6 混淆 `finished` 参数
GPU `SelectTopkOp` 无 finished 概念（全部行 active）。NPU 传 `None` 即全部参与；不要误传行数/索引张量。注意：**本构建实测 `finished=False` 的行仍产出有效 topk 结果**（未观察到过滤效果），其确切语义与 GPU 无对应，测试不覆盖（避免臆断）。

## 7. 测试模板

完整测试文件位于同目录 `test_npu_topk_gating_softmax_npu_gpu.py`。

```python
import os, unittest
import torch, torch_npu

torch.npu.set_device(int(os.environ.get("TEST_DEVICE_ID", 0)))

def _gpu_topk_softmax_ref(x, k, has_moe_norm):
    ...  # §5 的顺序 argmax 参考

def _npu_topk_gating_softmax(x_npu, k, has_moe_norm):
    if has_moe_norm:
        y, ei, row_idx = torch_npu.npu_moe_gating_top_k_softmax_v2(
            x_npu, k=k, finished=None, renorm=1, output_softmax=False)
    else:
        y, ei, row_idx = torch_npu.npu_moe_gating_top_k_softmax(x_npu, None, k)
    return y, ei, row_idx

class TestTopkGatingSoftmaxNpuVsGpu(unittest.TestCase):
    rtol = atol = 1e-2
    def _run_case(self, tag, shape, k, has_moe_norm, x_dtype=torch.float32):
        x_cpu = torch.randn(shape, dtype=torch.float32)
        if x_dtype != torch.float32:
            x_cpu = x_cpu.to(x_dtype)
        w_ref, idx_ref = _gpu_topk_softmax_ref(x_cpu.float(), k, has_moe_norm)
        y, ei, row_idx = _npu_topk_gating_softmax(x_cpu.npu(), k, has_moe_norm)
        torch.npu.synchronize()
        # 接口断言: y.dtype==x.dtype, ei.dtype==int32, shape==(M,k)
        # 数值断言: y≈w_ref(y.dtype), ei==idx_ref, renorm 时 sum(y,-1)≈1
        # row_idx: v1 校验 == ki*M+r; v2 校验空 (0,)
```

## 8. 调试技巧

1. **先跑小 shape**：`(M=4, E=8, k=2)` 快速验证语义，再上 prefill 规模
2. **区分两条 GPU 路径**：E 为 2 的幂（融合 kernel）与非 2 的幂（softmax+topk 两 kernel）都要覆盖
3. **验证 renorm 求和**：`has_moe_norm=True` 时打印 `y.sum(-1)`，应 ≈1（`renorm=1`）
4. **验证 tie**：全零 logits → softmax 均匀，选中索引应为 `[[0,1],...]`（最小索引）
5. **注意 v2 是 keyword-only**：`k/finished/renorm/output_softmax` 必须用关键字传参，位置参数报错
6. **`row_idx` 别过度断言**：v1 是 `ki*M+r`（k 主序）、v2 为空，见 §3/§6.4
7. **看 torch_npu 文档**：`_op_plugin_docs.py` 中 `npu_moe_gating_top_k_softmax` / `_v2` 的原型与约束

## 9. 实测结果

在本环境（昇腾 NPU）下运行测试（fp32 输入），NPU op 与 GPU 语义参考（fp64 参考）的比对：

| 模式 | 用例 | M | E | k | `y` max_abs_diff | `expert_idx` 一致 | `row_idx` |
|------|------|---|---|---|------------------|-------------------|-----------|
| NONE (v1) | decode | 1 | 8 | 2 | 5.96e-8 | ✓ | `[0,1]` (ki*M+r) |
| NONE (v1) | batch_pow2 | 16 | 64 | 4 | 2.98e-8 | ✓ | `ki*M+r` |
| NONE (v1) | prefill_pow2 | 2047 | 256 | 8 | 2.98e-8 | ✓ | `ki*M+r` |
| NONE (v1) | batch_nonpow2 | 16 | 10 | 3 | 8.94e-8 | ✓ | `ki*M+r` |
| NONE (v1) | prefill_nonpow2 | 2047 | 100 | 4 | 5.96e-8 | ✓ | `ki*M+r` |
| NONE (v1) | prefill_E300 | 2047 | 300 | 8 | 1.49e-8 | ✓ | `ki*M+r` |
| NONE (v1) | topk1 | 16 | 128 | 1 | 7.45e-9 | ✓ | `ki*M+r` |
| RENORM (v2) | 上述各用例 | — | — | — | 2.98e-8 ~ 1.19e-7 | ✓ | 空 `(0,)` |
| RENORM (v2) | prefill_E300 | 2047 | 300 | 8 | 5.96e-8 | ✓ | 空 `(0,)` |
| RENORM (v2) | topk1 | 16 | 128 | 1 | 0.0（k=1 归一化后恒为 1.0） | ✓ | 空 `(0,)` |
| tie 用例 | 4 | 8 | 2 | 0.0（完全一致） | ✓ | — |

### 误差来源分析

本算子**无 K 维长归约**：每行只做一次 softmax（行内 max 稳定化 + 求和）+ 顺序 topk。误差来源仅两个 fp32 舍入环节：

1. **softmax 的 exp/sum**：NPU 与 GPU 在 fp32 下实现，`expf` 实现差异导致 ~1e-7 量级绝对误差；
2. **renorm 除法**（仅 RENORMALIZE）：`1/sum` 与逐项乘的舍入，~1e-7。

实测 `y` 最大绝对误差 ≤1.2e-7，远小于测试容差 `rtol=atol=1e-2`（约 5 个数量级余量）。tie 用例完全一致（diff=0）。`expert_idx` 全部用例逐元素相等。

> **容差选择**：因与 GPU 是"同一算法、独立实现"，误差来自 fp32 舍入而非语义差异，`1e-2` 余量充足；若用 fp16 输入，误差上限约 1e-3（fp16 softmax 舍入），`1e-2` 仍通过。

**结论**：`topkGatingSoftmaxKernelLauncher`（`SelectTopkOp`）的 NPU 等价实现是 torch_npu `npu_moe_gating_top_k_softmax`（`has_moe_norm=False`）与 `npu_moe_gating_top_k_softmax_v2(renorm=1)`（`has_moe_norm=True`），语义一致。比对的核心难点不在数值（fp32 舍入下界 ~1e-7），而在**接口差异**：(1) renorm 语义的 v1/v2 映射；(2) NPU `y` dtype 跟随输入、GPU `expert_scales` 恒 fp32；(3) NPU `expert_idx` 恒 int32、GPU 支持 int32/int64；(4) 第 3 个返回重载：v1 是 `row_idx`（`ki*M+r`，k 主序），v2 默认空、`renorm=0`+`output_softmax=True` 时为全量 softmax `(M,E)`；(5) GPU 为 inplace 写、NPU 为返回三元组。测试用自行构造的用例（对齐 `(M,E)` fp32、覆盖 2 的幂与非 2 的幂 E、decode/prefill 规模、fp16 与 tie 边界）全部通过。
