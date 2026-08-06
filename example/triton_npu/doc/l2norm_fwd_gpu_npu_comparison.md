# l2norm_fwd 算子 GPU-NPU 精度比对指南

本文档总结 `l2norm_fwd`（L2 归一化）算子 NPU-vs-GPU 比对的实践经验，供后续该算子维护和类似逐行规约类算子比对参考。通用方法论参见同目录下的 `gpu_npu_comparison_guide.md`。

> **重要**：该算子的 NPU 实现在 **flash-linear-attention-npu** 中（`fla/ops/triton/triton_core/l2norm.py` 的 `l2norm_fwd` Triton kernel）。

## 整体流程

```
GPU 黄金数据 (.pt)  →  恢复非连续 stride (x)  →  解析输入/输出  →  调用 fla l2norm_fwd Triton kernel  →  NPU 执行  →  与 golden 比对 + L2 范数≈1 性质校验
```

## 1. 理解 GPU 数据格式

### 数据来源

GPU 端通过 `torch.save()` 导出的 `.pt` 文件位于 `<workspace>/sample/l2norm_fwd/`，采用 `inputs`/`outputs`/`input_meta` 分层结构：

```python
data = torch.load(path, map_location="cpu", weights_only=False)
# 顶层 key: op_name, mode, param_names, inputs, outputs, input_meta, model_state
```

### 典型数据内容

| key | 类型 | 值 |
|-----|------|----|
| `op_name` | str | `"l2norm_fwd"` |
| `mode` | str | `"prefill_T256"` / `"prefill_T32752"` |
| `param_names` | list | `['x', 'eps', 'output_dtype']`（仅记录参数名，值不在 dump 中） |
| `inputs/x` | tensor | shape=(B, T, H, D), dtype=float16 |
| `outputs` | tensor | **单个 tensor** `y`，shape 与 `x` 相同，dtype=float16 |
| `inplace_outputs` | dict | 空（l2norm_fwd 非原地算子，输出为新 tensor） |
| `input_meta` | dict | 记录 x 的原始 shape/stride/dtype/contiguous |
| `model_state` | None | 无模型状态 |

> **与 `fused_gdn_gating` 的差异**：该算子 `outputs` 是 list（`[g, beta]`），而 `l2norm_fwd` 的 `outputs` 是**单个 tensor**。读取时直接 `data["outputs"]`，无需索引。

### 两个 sample 的具体 shape

| 文件 | x shape | B | T | H | D |
|------|---------|---|---|---|---|
| `prefill_T256.pt` | (1, 16, 16, 128) | 1 | 16 | 16 | 128 |
| `prefill_T32752.pt` | (1, 2047, 16, 128) | 1 | 2047 | 16 | 128 |

两个 sample 的 `x` 都是连续 tensor（`contiguous=True`），无需特殊的 stride 恢复逻辑；`_restore_strided_tensor` 在 `contiguous=True` 时直接返回原 tensor。

## 2. GPU kernel 的数学语义

GPU 端实现来自 `flash-linear-attention-npu/fla/ops/triton/triton_core/l2norm.py` 的 `l2norm_fwd`：

```python
# kernel 内部将 x 展平为 (T', D)，T' = B*T*H
x = x.view(-1, x.shape[-1])
b_x = tl.load(...).to(tl.float32)
b_rstd = 1 / tl.sqrt(tl.sum(b_x * b_x) + eps)
b_y = b_x * b_rstd
```

即对展平后的每一行（长度 `D`）做 L2 归一化：

$$y_i = \frac{x_i}{\sqrt{\sum_j x_j^2 + \text{eps}}}, \quad \text{rstd}_i = \frac{1}{\sqrt{\sum_j x_j^2 + \text{eps}}}$$

- 输出 `y` dtype 与输入一致（fp16）
- 输出 `rstd` 是 fp32，shape = (T',)，**golden 未保存**
- `eps` 默认 `1e-6`，加在根号内**防止除零**并保证数值稳定；零向量行的分母退化为 $\sqrt{\text{eps}}$（非零、有限），`y` 不会产生 NaN
- kernel 内部先 `.to(tl.float32)` 累加 `sum(x*x)` 再开方，**避免 fp16 下平方溢出/精度损失**（fp16 最大约 65504，$D=128$ 时 $x_j^2$ 求和易溢出）

### 与 D 相关的重要分支

```python
BD = min(65536 // x.element_size(), triton.next_power_of_2(D))
if D > BD:
    raise RuntimeError("This layer doesn't support feature dim >= 64KB.")
if D <= 512:
    # 使用 block_ptr 版 l2norm_fwd_kernel（T 维度分块）
else:
    # 使用 l2norm_fwd_kernel1（单行逐元素版）
```

本测试的两个 sample `D=128 <= 512`，走分块 kernel 路径。

## 3. 算子在 flash-linear-attention-npu 中的实现

Triton kernel 位于 `flash-linear-attention-npu/fla/ops/triton/triton_core/l2norm.py`：

- `l2norm_fwd(x, eps=1e-6, output_dtype=None)` → 返回 `(y, rstd)`
  - `output_dtype=None` 时 `y` 与 `x` 同 dtype（本测试场景即 fp16）；否则 `y` 按 `output_dtype` 分配
- `l2norm_bwd` 提供反向（`dy*rstd - sum(dy*y)*y*rstd`），本测试只验证前向
- kernel 内部 `x.view(-1, D)` 展平，`torch.view` 要求 `x` 在最后维连续，否则会触发隐式拷贝或报错；同时代码对输出 `y` 有 `assert y.stride(-1) == 1` 断言。**因此非连续输入应在调用前 `.contiguous()`**（当前 golden 的 `x` 连续，无需处理）

### 与 chunk_scaled_dot_kkt / solve_tril 的区别

| 维度 | chunk_scaled_dot_kkt | solve_tril | l2norm_fwd |
|------|---------------------|-----------|------------|
| NPU 算子来源 | fla_npu.ops.ascendc | fla_npu.ops.ascendc | **fla Triton kernel** |
| 输出结构 | 单个 tensor | 单个 tensor | **tuple `(y, rstd)`** |
| golden 是否含 rstd | — | — | **否**，只有 y |
| 是否需要 layout 转置 | (T,B,H,BT)→(B,Hk,T,BT) | 否 | **否**，逐行归一化与 layout 无关 |
| 输入 dtype 限制 | — | 仅 fp16/bf16 | 无限制 |

## 4. 测试实现

测试文件位于 `rtp-llm/example/ascendc_npu/test_npu_l2norm_fwd_gpu_golden.py`，核心结构：

```python
from fla.ops.triton.triton_core.l2norm import l2norm_fwd

@unittest.skipIf(not torch.npu.is_available(), "NPU is not available")
class TestL2NormFwdGpuGolden(unittest.TestCase):
    rtol = 5e-2
    atol = 5e-2

    def call_op(self, x, eps=1e-6):
        return l2norm_fwd(x, eps=eps)

    def _run_case(self, filename: str) -> None:
        case = _load_gpu_case(filename)

        # 校验 golden 合法性
        self.assertEqual(tuple(case["y_expected"].shape), tuple(case["x"].shape))
        self.assertEqual(case["y_expected"].dtype, torch.float16)

        # kernel 返回 (y, rstd)，golden 只有 y
        y, rstd = self.call_op(case["x"].npu(), eps=1e-6)
        torch.npu.synchronize()

        # 与 golden 比对
        self.assertTensorClose(y, case["y_expected"])

        # 额外性质校验：每行 L2 范数 ≈ 1
        y_flat = y.detach().cpu().float().reshape(-1, y.shape[-1])
        norms = y_flat.norm(dim=-1)
        self.assertTrue(torch.allclose(norms, torch.ones_like(norms), rtol=1e-2, atol=1e-2), ...)
```

要点说明：

- **加载**：`_load_gpu_case` 通过 `_restore_strided_tensor` 恢复 `x` 的原始 stride；因 `input_meta.x.contiguous=True`，直接返回保存的连续 tensor
- **参数**：测试硬编码 `eps=1e-6`（与 `l2norm_fwd` 默认值及 golden 的 `param_names` 中记录的 `eps` 一致），不传 `output_dtype`（保持 `y` 为 fp16）
- **校验**：先做 shape/dtype 合法性断言，再比对 `y` 与 golden，最后用 `norms = y.reshape(-1, D).norm(-1)` 校验逐行 L2 范数 ≈ 1

两个用例覆盖 `prefill_T256`（D=128，小 T）与 `prefill_T32752`（D=128，大 T）。

## 5. 常见错误

### 5.1 `call_op` 签名与位置参数不匹配

`_run_case` 用位置参数调用 `self.call_op(case["x"].npu(), eps=1e-6)`，若 `call_op` 写成 `**kwargs` 形式会报 `TypeError: takes 1 positional argument but 2 were given`：

```python
# ✗ 错误：**kwargs 无法接收位置参数 x
def call_op(self, **kwargs):
    return _l2norm_fwd(**kwargs)

# ✓ 正确：显式位置参数
def call_op(self, x, eps=1e-6):
    return _l2norm_fwd(x, eps=eps)
```

### 5.2 忘记 kernel 返回 tuple

`l2norm_fwd` 返回 `(y, rstd)`，不是单个 tensor：

```python
# ✗ 错误：当成单个输出
y = self.call_op(x)

# ✓ 正确：解包 tuple
y, rstd = self.call_op(x, eps=1e-6)
```


## 6. 调试技巧

1. **理解展平语义**：`l2norm_fwd` 内部 `x.view(-1, D)`，把 `(B, T, H, D)` 展平成 `(B*T*H, D)` 逐行归一化，输出 shape 与输入一致
2. **注意 D 分支**：`D <= 512` 走分块 kernel（`l2norm_fwd_kernel`），`D > 512` 走单行 kernel（`l2norm_fwd_kernel1`），两者语义一致
3. **额外校验范数**：golden 只有 y，可额外校验 `y.norm(dim=-1) ≈ 1` 作为自洽性检查（对零向量行应跳过或按 golden 语义处理）
4. **rstd 未保存**：golden 无 rstd，测试中 `rstd` 仅作解包使用，不参与比对

## 7. GPU 与 NPU 布局策略对比

| 维度 | GPU dump | NPU（fla Triton kernel） |
|------|---------|------------------|
| `x` 布局 | (B, T, H, D) fp16 连续 | 同 (B, T, H, D)，无需转置 |
| 展平语义 | — | `x.view(-1, D)`，逐行 (T'=B*T*H, D) 归一化 |
| 输出 `y` | (B, T, H, D) fp16 | 同 (B, T, H, D) fp16 |
| 输出 `rstd` | **未保存** | (B*T*H,) fp32，仅解包不比对 |
| layout 转置 | 无 | **无需转置** |
| 算子来源 | — | **fla `l2norm_fwd` Triton kernel**（非 fla_npu.ops.ascendc） |
| 非连续性 | x 连续 | 恢复 stride 后需保证最后维连续（`stride(-1)==1`） |

## 8. 实测结果

在本环境（昇腾 NPU，`fla` 已安装）下运行测试，逐行归一化输出与 GPU golden 的比对数据如下：

| 用例 | x shape | `max_abs_diff(y, y_gold)` | 输出 L2 范数 min / max | `rstd` dtype |
|------|---------|--------------------------|------------------------|--------------|
| `prefill_T256` | (1, 16, 16, 128) | **0.0** | 0.999775 / 1.000235 | float32 |
| `prefill_T32752` | (1, 2047, 16, 128) | **4.88e-4** | 0.999690 / 1.000284 | float32 |

说明：

- `prefill_T256` 逐元素完全一致（diff = 0）；`prefill_T32752` 最大绝对误差 4.88e-4 ≈ $2^{-11}$，是 fp16 存储 `y` 时与 golden 的**正常舍入差异**，远小于测试容差（`rtol=atol=5e-2`）
- 两个用例的逐行 L2 范数均落在 `[0.9997, 1.0003]` 内，验证归一化语义正确
- `rstd` 为 float32、shape=(B*T*H,)，范围 `[0.10, 3.48]`（与行向量的 $1/\lVert x\rVert$ 一致），golden 未保存故不参与比对

**结论**：`l2norm_fwd` 是逐行 L2 归一化算子，其 NPU 实现在 flash-linear-attention-npu 的 Triton kernel 中（`fla` 包已安装到环境）。比对的核心难点不在 layout 转换（GPU/NPU 布局一致、逐行归一化与 head-major/token-major 无关），而在两点——(1) 处理 `l2norm_fwd` 返回的 `(y, rstd)` tuple（golden 仅含 `y`）；(2) `call_op` 签名与位置参数匹配。实测 T256 最大误差 0.0、T32752 最大误差 4.88e-4（fp16 舍入量级），均在 fp16 容差内，验证通过。
