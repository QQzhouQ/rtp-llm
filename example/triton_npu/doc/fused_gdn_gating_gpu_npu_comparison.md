# fused_gdn_gating 算子 GPU-NPU 精度比对指南

本文档总结 `fused_gdn_gating`（Gated Delta Net 门控激活）算子 NPU-vs-GPU 比对的实践经验，组织方式参考同目录 `gpu_npu_comparison_guide.md` 的通用方法论。

## 整体流程

```
GPU 黄金数据 (.pt)  →  打印 key/shape 诊断  →  恢复非连续 stride (a/b)  →  CPU reference 验证语义  →  调用 rtp-llm fused_gdn_gating Triton kernel (NPU)  →  比对 g / beta
```

## 1. 理解 GPU 数据格式

GPU 端通过 `torch.save()` 导出的 `.pt` 文件位于 `<workspace>/sample/fused_gdn_gating/`，加载后先打印所有 key 和 shape：

```python
data = torch.load(path, map_location="cpu", weights_only=False)
for k, v in data.items():
    if torch.is_tensor(v):
        print(f"{k}: shape={list(v.shape)}, dtype={v.dtype}")
    elif isinstance(v, (list, tuple)):
        print(f"{k}: list[{len(v)}] -> {[(tuple(t.shape), str(t.dtype)) for t in v if torch.is_tensor(t)]}")
    else:
        print(f"{k}: {v}")
```

### 数据来源

| 文件 | S（token 数） | H（head 数） | `a`/`b` stride | `a`/`b` 连续性 |
|------|--------------|-------------|----------------|----------------|
| `decode_seq1.pt` | 1 | 32 | (64, 1) | 连续（dim0=1 时 stride 任意仍算连续） |
| `prefill_seq32.pt` | 32 | 32 | (64, 1) | **非连续**（行 stride 64 > H=32） |
| `prefill_seq2047.pt` | 2047 | 32 | (64, 1) | **非连续** |

### 典型数据内容

| key | 类型 | 值 |
|-----|------|----|
| `op_name` | str | `"fused_gdn_gating"` |
| `mode` | str | `"decode_seq1"` / `"prefill_seq32"` / `"prefill_seq2047"` |
| `param_names` | list | `['A_log', 'a', 'b', 'dt_bias', 'beta', 'threshold']`（仅记录参数名，值不在 dump 中；默认 `beta=1.0, threshold=20.0`） |
| `inputs/A_log` | tensor | shape=(H,), dtype=float16, contiguous |
| `inputs/a` | tensor | shape=(S, H), dtype=float16, **stride=(64, 1)** 非连续视图 |
| `inputs/b` | tensor | shape=(S, H), dtype=float16, **stride=(64, 1)** 与 `a` 相同 |
| `inputs/dt_bias` | tensor | shape=(H,), dtype=float16, contiguous |
| `outputs` | list（长度 2） | `[g, beta]` — 非 inplace 输出 |
| `outputs[0]` | tensor | shape=(1, S, H), dtype=float32 — 门控 `g` |
| `outputs[1]` | tensor | shape=(1, S, H), dtype=float16 — `beta` |
| `inplace_outputs` | dict | 空（非 inplace 算子，输出为新 tensor） |
| `input_meta` | dict | 记录 A_log/a/b/dt_bias 的原始 shape/stride/dtype/contiguous |
| `model_state` | None | 无模型状态 |

> **注意**：`outputs` 是长度为 2 的 list（`[g, beta]`），不是单个 tensor；`inplace_outputs` 为空（非 inplace 算子）。

### `input_meta` 与非连续 stride 恢复

`a`/`b` 在 GPU 上是 cache buffer 的**视图**：底层每行 64 个 float16，但算子只用前 `H=32` 个 head，行 stride 为 64。`torch.save()` 会以连续形式保存（prefill 时 stride 丢失），必须从 `input_meta` 恢复，否则无法验证 NPU 算子对非连续输入的处理能力。

## 2. 恢复非连续 stride

使用 `torch.empty_strided` 从 `input_meta` 恢复原始布局：

```python
def _restore_strided_tensor(saved_data: torch.Tensor, meta: dict) -> torch.Tensor:
    """从 input_meta 恢复张量的原始（可能非连续）stride。"""
    if not meta or meta.get("contiguous", True):
        return saved_data
    dtype = getattr(torch, meta["dtype"].replace("torch.", ""))
    tensor = torch.empty_strided(meta["shape"], meta["stride"], dtype=dtype)
    tensor.copy_(saved_data)
    return tensor
```

### 恢复后的典型 stride

| 参数 | GPU 原始 shape | GPU 原始 stride | 恢复后 contiguous |
|------|---------------|----------------|------------------|
| `A_log` / `dt_bias` | (H,) | (1,) | True |
| `a` (prefill) | (S, 32) | (64, 1) | **False** |
| `b` (prefill) | (S, 32) | (64, 1) | **False** |
| `a`/`b` (decode, S=1) | (1, 32) | (64, 1) | True（dim0=1，stride 任意仍算连续） |

## 3. 推断 GPU 的语义参数（CPU reference）

GPU dump 不直接包含 softplus 的 `beta`/`threshold` 值（`param_names` 中有名称但无值）。用 CPU reference 验证语义（默认 `beta=1.0, threshold=20.0`）：

```python
def softplus(x, beta=1.0, threshold=20.0):
    return torch.where(beta * x <= threshold,
                       (1 / beta) * torch.log1p(torch.exp(beta * x)), x)

g_ref = -A_log.float().exp() * softplus(a.float() + dt_bias)   # fp32
beta_ref = torch.sigmoid(b.float())                            # fp32

print("match g   :", torch.allclose(g_ref, g_golden.float(), rtol=1e-3, atol=1e-3))
print("match beta:", torch.allclose(beta_ref.half(), beta_golden, rtol=1e-3, atol=1e-3))
```

确认 GPU 语义后，再在 NPU 上验证算子。数学上：

$$g = -\exp(A\_log) \cdot \text{softplus}(a + dt\_bias), \qquad \beta = \text{sigmoid}(b)$$

其中 softplus 是带阈值分支的数值稳定形式：

$$\text{softplus}(x) = \begin{cases} \dfrac{1}{\beta}\log\bigl(1 + e^{\beta x}\bigr) & \beta x \le \text{threshold} \\[4pt] x & \text{否则} \end{cases}, \qquad \beta = 1.0,\ \text{threshold} = 20.0$$

阈值分支避免 `exp` 在大输入时溢出（$e^{\beta x}$ 数值爆炸），`x` 足够大时直接退化为线性。

## 4. 理解算子的输入格式

**不要猜测！** 直接读 rtp-llm kernel `fused_gdn_gating`（`rtp_llm/models_py/triton_kernels/fla/gdn_gating.py`）的 shape/stride 约束：

| 参数 | 期望 | 说明 |
|------|------|------|
| `A_log` | `(H,)` float16 | per-head 系数，访问时需 `head_off < NUM_HEADS` 掩码 |
| `a` | `(S, H)` float16 | 按 `stride_ab` 寻址，**行 stride 可为任意**（支持非连续） |
| `b` | `(S, H)` float16 | 与 `a` 同 stride |
| `dt_bias` | `(H,)` float16 | per-head bias |
| 输出 `g` | `(1, S, H)` float32 | 新 tensor，非 inplace |
| 输出 `beta` | `(1, S, H)` float16 | 新 tensor，dtype 跟随 `b` |

### 关键 kernel 断言

```python
stride_ab = a.stride(0)                        # batch 维 stride，kernel 据此寻址非连续输入
assert stride_ah == 1 and stride_bh == 1       # H 轴必须 stride=1
assert stride_ab == stride_bb
```

**非连续支持**：kernel 通过 `stride_ab` 显式支持非连续 `a`/`b`，因此**无需 `.contiguous()`**——非连续信息被 kernel 完整保留，这解释了 dump 中 `a`/`b` 的 stride 为 `(64, 1)`。

## 5. 格式转换映射

GPU 与 NPU（rtp-llm Triton for Ascend）布局一致，**无需 layout 转置**：

```python
# a/b 恢复非连续 stride 后直接传 NPU（kernel 用 stride_ab 寻址，无需 contiguous）
g, beta = fused_gdn_gating(
    A_log=A_log.npu(),
    a=a.npu(),            # 非连续视图
    b=b.npu(),            # 非连续视图
    dt_bias=dt_bias.npu(),
)
```

## 6. 常见错误

### 6.1 未恢复 `a`/`b` 非连续 stride

`torch.save()` 将非连续 `a`/`b` 以连续形式保存（stride 从 (64,1) 变 (32,1)）。不恢复 stride，就丢失 GPU 端真实内存布局，无法验证非连续寻址路径：

```python
# ✗ 错误：直接用保存的连续 tensor
a = inputs["a"]

# ✓ 正确：从 input_meta 恢复原始 stride
a = _restore_strided_tensor(inputs["a"], meta.get("a", {}))
```

### 6.2 把 `outputs` 当单个 tensor

`outputs` 是长度为 2 的 list，不是单个 tensor：

```python
# ✗ 错误：当单个 tensor 用
y_expected = data["outputs"]

# ✓ 正确：分别取 g 和 beta
g_expected = data["outputs"][0]      # (1,S,H) fp32
beta_expected = data["outputs"][1]   # (1,S,H) fp16
```

### 6.3 在 fp16 下计算 `a + dt_bias`

kernel 显式 `.to(tl.float32)` 后再相加。若在 fp16 下计算，长序列上可能溢出（A 变 `-inf`），导致 softplus 分支异常。这是 kernel 内部语义，参考实现应保持一致。

### 6.4 decode 与 prefill 连续性混淆

`decode_seq1` 的 `a` 因 dim0=1 被判为**连续**（stride 任意仍算连续），`prefill` 才是真非连续。不要假设所有 sample 一致：

```python
# 应打印确认，而不是猜测
print(a.stride(), a.is_contiguous())   # decode: (64,1) True; prefill: (64,1) False
```

### 6.5 缺少非连续性断言

恢复 stride 后应添加断言，避免 `input_meta` 缺失或数据被意外 `.contiguous()` 时静默通过：

```python
self.assertFalse(prefill_a.is_contiguous(), "prefill a/b should be non-contiguous")
self.assertEqual(prefill_a.stride(), (64, 1), "stride should match GPU input_meta")
```

## 7. 测试模板

```python
import importlib.util, os, sys, types, unittest
import torch

torch.npu.set_device(int(os.environ.get("TEST_DEVICE_ID", 0)))

def _load_rtp_llm_fla_modules():
    # stub 包层次 + importlib，绕过 rtp_llm.__init__ 的 C++ 依赖
    # （gdn_gating.py 无相对导入，加载该单文件即可）
    ...
fused_gdn_gating = _load_rtp_llm_fla_modules().fused_gdn_gating

class TestFusedGdnGatingGpuGolden(unittest.TestCase):
    rtol = 5e-2
    atol = 5e-2

    def assertTensorClose(self, actual, expected, *, rtol=None, atol=None):
        # 输出 max_abs_diff 便于定位
        ...

    def _run_case(self, filename):
        # 恢复非连续 stride → 调用 → 同步 → 比对 g/beta
        ...
```

三个用例覆盖 decode（S=1）与 prefill（S=32 / S=2047），其中 prefill 的 `a`/`b` 为非连续视图。

## 8. 调试技巧

1. **先打印所有 key/shape**：确认 `outputs` 是 list[2] 而非单 tensor，`input_meta` 存在
2. **先在 CPU 上验证语义**：用 softplus/sigmoid reference 复现 GPU 输出，确认 `beta`/`threshold` 默认值
3. **打印 `a`/`b` 的 stride 和 is_contiguous**：区分 decode（连续）与 prefill（非连续）
4. **读 kernel 源码**：确认 `stride_ab` 寻址逻辑和 fp32 计算，避免误传 `.contiguous()`
5. **加非连续性断言**：`assertFalse(prefill_a.is_contiguous())` + `assertEqual(stride, (64,1))`
6. **小 shape 复现**：若大数据出错，构造 (S=4, H=32) 的小 case 缩小范围

## 9. GPU 与 NPU 布局策略对比

| 维度 | GPU dump | NPU（rtp-llm Triton for Ascend） |
|------|---------|------------------|
| `A_log` / `dt_bias` | (H,) fp16 连续 | 同 (H,) fp16，无需转换 |
| `a` / `b` 布局 | (S, H) fp16，prefill 时 stride=(64,1) 非连续 | 同 (S, H)，恢复 stride 后直接传（kernel 用 `stride_ab` 寻址） |
| 输出 `g` | (1, S, H) fp32 | 同 (1, S, H) fp32 |
| 输出 `beta` | (1, S, H) fp16 | 同 (1, S, H) fp16 |
| layout 转置 | 无 | **无需转置** |
| 非连续性处理 | 非连续（cache 视图） | 保留非连续 stride，kernel 显式寻址，无需 `.contiguous()` |
| 算子来源 | — | rtp-llm `gdn_gating.py`（stub 包 + importlib 加载） |

## 10. 实测结果

在本环境（昇腾 NPU）下运行测试，输出与 GPU golden 的逐用例比对数据：

| 用例 | S | H | `g` max_abs_diff | `g` mean_abs_diff | `beta` max_abs_diff |
|------|---|---|------------------|-------------------|---------------------|
| `decode_seq1` | 1 | 32 | 1.14e-5 | 6.07e-7 | 0.0 |
| `prefill_seq32` | 32 | 32 | 1.91e-5 | 7.70e-7 | 0.0 |
| `prefill_seq2047` | 2047 | 32 | 1.91e-5 | 8.06e-7 | 2.44e-4 |

说明：

- `g`（fp32 输出）最大绝对误差 ~1.9e-5，为 fp32 计算下 GPU/NPU 的 `exp`/`log1p` 实现差异导致的正常舍入，远小于测试容差
- `beta`（fp16 输出）最大绝对误差 2.44e-4 ≈ $2^{-12}$，是 fp16 存储结果时的正常舍入差异；其中 `decode_seq1` 与 `prefill_seq32` 两个用例逐元素完全一致（diff = 0）
- 测试容差 `rtol=atol=5e-2`，三个用例的 `g` 与 `beta` 均通过

**结论**：`fused_gdn_gating` 是逐元素门控激活算子（非 inplace），其 NPU 实现直接复用 rtp-llm 的 Triton kernel。比对的核心难点不在 layout 转换（GPU/NPU 布局一致），而在三点——(1) 正确恢复 `a`/`b` 的非连续 stride；(2) 区分 `outputs` 为 list 结构；(3) 保持 kernel 的 fp32 计算语义。实测三个用例的 `g`（fp32）与 `beta`（fp16）误差均在 fp16/fp32 舍入量级内，验证通过。
