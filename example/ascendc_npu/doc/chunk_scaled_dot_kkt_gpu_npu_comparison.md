# chunk_scaled_dot_kkt 算子 GPU-NPU 精度比对指南

本文档总结 `chunk_scaled_dot_kkt` 算子 NPU-vs-GPU 比对的实践经验，供后续该算子维护和类似 KKT 类算子比对参考。通用方法论参见同目录下的 `gpu_npu_comparison_guide.md`。

## 整体流程

```
GPU 黄金数据 (.pt)  →  解析输入/输出/属性  →  Layout 转换 (B,T,H,D)→(B,H,T,D)  →  GQA head 子采样  →  beta dtype 转换 fp16→fp32  →  推断 chunk_size  →  计算 chunk_indices  →  调用 NPU 算子  →  GPU 输出 head 子采样  →  精度比对
```

## 1. 理解 GPU 数据格式

### 数据来源

GPU 端通过 `torch.save()` 导出的 `.pt` 文件位于 `<workspace>/sample/chunk_scaled_dot_kkt_fwd/`，采用 `inputs`/`outputs`/`input_meta` 分层结构：

```python
data = torch.load(path, map_location="cpu", weights_only=False)
# 顶层 key: op_name, mode, param_names, inputs, outputs, inplace_outputs, input_meta, model_state
```

### 典型数据内容

以 `seq32.pt` 为例：

| key | 类型 | 值 |
|-----|------|----|
| `inputs/k` | tensor | shape=(1, 32, 16, 128), dtype=float16, contiguous |
| `inputs/beta` | tensor | shape=(1, 32, 32), dtype=float16, contiguous |
| `inputs/g_cumsum` | tensor | shape=(1, 32, 32), dtype=float32, contiguous |
| `inputs/cu_seqlens` | tensor | shape=(2,), dtype=int32, 值=[0, 32] |
| `inputs/output_dtype` | dtype | torch.float32 |
| `outputs` | tensor | shape=(1, 32, 32, 64), dtype=float32 |
| `inplace_outputs` | dict | 空（chunk_scaled_dot_kkt 无原地更新） |
| `input_meta` | dict | 记录原始 shape/stride/dtype/contiguous |

另一个测试文件 `seq2047.pt`：
- seqlen=2047, chunk_size=64, 32 chunks
- k/beta/g_cumsum 的 dim 1 更大，其余格式相同

### GPU 布局说明

GPU dump 中所有 tensor 都是 `(B, T, H, ...)` 布局（`head_first=False`），T 在 dim 1：
- k: `(B, T, Hk, Dk)` — T 在 dim 1, Hk 在 dim 2
- beta: `(B, T, Hv)` — T 在 dim 1, Hv 在 dim 2
- g_cumsum: `(B, T, Hv)` — T 在 dim 1, Hv 在 dim 2
- output: `(B, T, Hv, chunk_size)` — T 在 dim 1, Hv 在 dim 2

### 与其他算子数据格式的差异

| 维度 | causal_conv1d | chunk_fwd_o | chunk_scaled_dot_kkt |
|------|--------------|-------------|---------------------|
| GPU 布局 | 多种 | `(B, T, H, D)` | `(B, T, H, D)` |
| 标量参数 | 嵌在 inputs | `chunk_size` 需推断 | `chunk_size` 需推断, `output_dtype` 在 inputs |
| 原地更新 | `conv_states` 被修改 | 无 | 无 |
| 非连续性 | 常非连续 | 全部 contiguous | 全部 contiguous |
| GQA | 无 | NV % NK == 0 | Hv % Hk == 0, **输出 head 维 = Hk** |
| beta dtype | N/A | bfloat16 | **float16 (GPU) → float32 (NPU)** |

## 2. 恢复非连续 stride

当前 GPU dump 中所有输入均为 contiguous，但保留 stride 恢复逻辑以兼容未来非连续 dump：

```python
k_gpu = _restore_strided_tensor(inputs["k"], meta.get("k", {}))
```

## 3. 推断 GPU 的语义参数

### 3.1 `chunk_size` 推断

GPU dump 不直接包含 `chunk_size`（`param_names` 中有但 `inputs` 字典中没有）。从 output shape 的最后一维推断：

```python
chunk_size = data["outputs"].shape[-1]  # 64
```

NPU 仅支持 `chunk_size ∈ {16, 32, 64, 128}`，dump 中的 64 在支持范围内。

### 3.2 `cu_seqlens` dtype 转换

GPU dump 中 `cu_seqlens` 是 int32，NPU wrapper 接受 Python list 并内部转为 int64：

```python
cu_seqlens = [int(v) for v in inputs["cu_seqlens"].tolist()]
```

## 4. 理解 NPU 算子的输入格式

### 关键 shape/dtype 约束

| 参数 | NPU 期望 | 说明 |
|------|---------|------|
| `k` | `(B, Hk, T, Dk)` float16/bfloat16 | 4D, head_first 布局 |
| `g` | `(B, Hv, T)` float32 | 3D, head_first 布局, chunk 内 cumulative gate |
| `beta` | `(B, Hv, T)` **float32** | 3D, head_first 布局, **GPU dump 为 float16 需转换** |
| `cu_seqlens` | int64 list | varlen 模式必填 |
| `chunk_indices` | int64 list | varlen 模式必填, 扁平化 (seq_idx, chunk_idx) 对 |
| `chunk_size` | int64 | 仅支持 16/32/64/128, 默认 64 |
| `out` | `(B, Hk, T, chunk_size)` float32 | **输出 head 维 = Hk**（非 Hv） |

### 关键校验逻辑（来自 `chunk_scaled_dot_kkt_tiling.cpp`）

```cpp
// k 必须是 4D: [B, Hk, T, K]
if (kShape.GetDimNum() != 4) return ge::GRAPH_FAILED;

// g 和 beta 必须是 3D: [B, Hv, T]
if (gShape.GetDimNum() != 3 || betaShape.GetDimNum() != 3) return ge::GRAPH_FAILED;

// g/beta 的 B 和 T 必须与 k 一致
if (gShape.GetDim(0) != b || gShape.GetDim(2) != t) return ge::GRAPH_FAILED;

// beta shape 必须精确匹配 [B, Hv, T]
if (!Shape3Equal(betaShape, b, hv, t)) return ge::GRAPH_FAILED;

// GQA: Hv % Hk == 0
if (hv % hk != 0) return ge::GRAPH_FAILED;

// k dtype: float16 或 bfloat16
// g/beta dtype: 必须 float32

// chunk_size 仅支持 16/32/64/128
if (!IsChunkSizeSupported(chunkSize)) return ge::GRAPH_FAILED;

// cu_seqlens 和 chunk_indices 必须同时存在或同时缺席
if (hasCuSeqlens != hasChunkIndices) return ge::GRAPH_FAILED;
```

## 5. 格式转换映射

### 5.1 Layout 转换

GPU dump 是 `(B, T, H, D)`，NPU 要求 `(B, H, T, D)`。通过 `transpose(1, 2)` 完成转换：

```python
k_npu = k_gpu.transpose(1, 2).contiguous()
g_npu = g_gpu.transpose(1, 2).contiguous()
beta_npu = beta_gpu.transpose(1, 2).contiguous().to(torch.float32)
```

### 5.2 GQA head 子采样（核心）

GPU dump 中 `g`/`beta` 有 `Hv` 个 head，`k` 有 `Hk` 个 head（`Hv = ratio × Hk`）。NPU 输出 `Hk` 个 head。

**GPU 和 NPU 的 head 映射差异**：

| 消费者 | head h 使用的 k | head h 使用的 g/beta | 输出 head 数 |
|--------|----------------|---------------------|-------------|
| GPU kernel | `k[h_v // ratio]` | `g[h_v]` | Hv |
| NPU kernel | `k[h_k]` | `g[h_k]` | Hk |

NPU head `h_k` 使用 `k[h_k]` 和 `g[h_k]`。GPU head `h_v = h_k × ratio` 也使用 `k[h_v // ratio] = k[h_k]` 和 `g[h_v] = g[h_k × ratio]`。

因此，**NPU head `h_k` 对应 GPU head `h_v = h_k × ratio`**。为了让 NPU 使用与 GPU 相同的 g/beta 值，需要对 g/beta 做子采样：

```python
Hk = k_npu.shape[1]  # 16
Hv = g_npu.shape[1]  # 32
ratio = Hv // Hk     # 2

# 子采样: g_npu[h_k] = g_gpu[h_k * ratio]
g_npu = g_npu[:, ::ratio].contiguous()       # (B, Hk, T)
beta_npu = beta_npu[:, ::ratio].contiguous()  # (B, Hk, T)
```

**不子采样会导致 head 0 匹配但 head 1+ 不匹配**（max_diff ≈ 0.34~0.99）。

### 5.3 beta dtype 转换

GPU dump 中 `beta` 是 float16，NPU 算子定义要求 `DT_FLOAT`：

```python
beta_npu = beta_gpu.transpose(1, 2).contiguous().to(torch.float32)
```

### 5.4 cu_seqlens dtype 转换

```python
cu_seqlens = [int(v) for v in inputs["cu_seqlens"].tolist()]
```

### 5.5 chunk_indices 计算

```python
def _prepare_chunk_indices(cu_seqlens, chunk_size):
    rows = []
    for seq_idx, (start, end) in enumerate(zip(cu_seqlens[:-1], cu_seqlens[1:])):
        for chunk_idx in range(math.ceil((end - start) / chunk_size)):
            rows.append((seq_idx, chunk_idx))
    return [v for row in rows for v in row]
```

### 5.6 输出 head 子采样与 Layout 转换

GPU 输出是 `(B, T, Hv, chunk_size)`，NPU 输出是 `(B, Hk, T, chunk_size)`。比较时需：
1. transpose(1, 2) 转为 head_first
2. 子采样取 `h_v = h_k × ratio` 对应的 head

```python
out_expected_npu = out_gpu.transpose(1, 2).contiguous()[:, ::ratio][:, :Hk]
```

### 验证转换正确性

测试通过两个 case 验证：
- `seq32.pt`: seqlen=32, chunk_size=64, 1 chunk, Hk=16, Hv=32 → PASS (max_diff ≈ 1e-6)
- `seq2047.pt`: seqlen=2047, chunk_size=64, 32 chunks, Hk=16, Hv=32 → PASS

## 6. 常见错误

### 6.1 输出 shape 不匹配（Hk vs Hv）

NPU 输出 `(B, Hk, T, chunk_size)`，GPU 输出 `(B, T, Hv, chunk_size)`。当 `Hv > Hk`（GQA）时，head 维数量不同导致 `allclose` 报错：

```
RuntimeError: The size of tensor a (16) must match the size of tensor b (32) at non-singleton dimension 1
```

**修复**：对 GPU 输出做 head 子采样 `[:, ::ratio][:, :Hk]` 后再比较。

### 6.2 GQA head 映射错误导致精度失败

不子采样 g/beta 直接传入 NPU，head 0 匹配但 head 1+ 不匹配（max_diff ≈ 0.34~0.99）：

```python
# ✗ 错误：直接使用全部 Hv 个 head 的 g/beta
g_npu = g_gpu.transpose(1, 2).contiguous()  # (B, Hv=32, T)

# ✓ 正确：子采样到 Hk 个 head
g_npu = g_gpu.transpose(1, 2).contiguous()[:, ::ratio]  # (B, Hk=16, T)
```

### 6.3 beta dtype 为 float16

NPU 算子定义要求 `beta` 为 `DT_FLOAT`。GPU dump 中 `beta` 是 float16，不转换会导致参数校验失败：

```python
# ✗ 错误：保持 float16
beta_npu = beta_gpu.transpose(1, 2).contiguous()

# ✓ 正确：转为 float32
beta_npu = beta_gpu.transpose(1, 2).contiguous().to(torch.float32)
```

### 6.4 chunk_size 不在支持列表

NPU 仅支持 `chunk_size ∈ {16, 32, 64, 128}`。若推断出的 chunk_size 不在此列表（如 48），tiling 会返回失败。

### 6.5 cu_seqlens/chunk_indices 未同时提供

NPU 要求 `cu_seqlens` 和 `chunk_indices` 必须同时存在或同时缺席。仅提供一个会导致 tiling 失败。

## 7. 测试模板

完整测试文件位于 `torch_custom/fla_npu/test/test_npu_chunk_scaled_dot_kkt_gpu_golden.py`，核心结构：

```python
import math
import os
import unittest
import torch
from fla_npu.ops import ascendc as ascendc_ops

torch.npu.set_device(int(os.environ.get("TEST_DEVICE_ID", 0)))
_DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__),
                                          "..", "..", "..", "..",
                                          "sample", "chunk_scaled_dot_kkt_fwd"))


def _prepare_chunk_indices(cu_seqlens, chunk_size):
    rows = []
    for seq_idx, (start, end) in enumerate(zip(cu_seqlens[:-1], cu_seqlens[1:])):
        for chunk_idx in range(math.ceil((end - start) / chunk_size)):
            rows.append((seq_idx, chunk_idx))
    return [v for row in rows for v in row]


def _load_gpu_case(filename):
    data = torch.load(os.path.join(_DATA_DIR, filename), map_location="cpu", weights_only=False)
    inputs = data["inputs"]

    k_gpu = inputs["k"]
    beta_gpu = inputs["beta"]
    g_gpu = inputs["g_cumsum"]

    # GPU (B,T,H,D) → NPU (B,H,T,D)
    k_npu = k_gpu.transpose(1, 2).contiguous()
    g_npu = g_gpu.transpose(1, 2).contiguous()
    beta_npu = beta_gpu.transpose(1, 2).contiguous().to(torch.float32)

    # GQA head 子采样
    Hk = k_npu.shape[1]
    Hv = g_npu.shape[1]
    ratio = Hv // Hk
    if ratio > 1:
        g_npu = g_npu[:, ::ratio].contiguous()
        beta_npu = beta_npu[:, ::ratio].contiguous()

    cu_seqlens = [int(v) for v in inputs["cu_seqlens"].tolist()]
    chunk_size = data["outputs"].shape[-1]
    chunk_indices = _prepare_chunk_indices(cu_seqlens, chunk_size)

    # GPU 输出 head 子采样 + layout 转换
    out_expected = data["outputs"].transpose(1, 2).contiguous()[:, ::ratio][:, :Hk]

    return {
        "k": k_npu, "g": g_npu, "beta": beta_npu,
        "cu_seqlens": cu_seqlens, "chunk_indices": chunk_indices,
        "chunk_size": chunk_size, "out_expected": out_expected,
    }


class TestChunkScaledDotKktGpuGolden(unittest.TestCase):
    rtol = 5e-2
    atol = 5e-2

    def call_op(self, **kwargs):
        return ascendc_ops.npu_chunk_scaled_dot_kkt(**kwargs)

    def _run_case(self, filename):
        case = _load_gpu_case(filename)
        out = self.call_op(
            k=case["k"].npu(), g=case["g"].npu(), beta=case["beta"].npu(),
            cu_seqlens=case["cu_seqlens"], chunk_indices=case["chunk_indices"],
            chunk_size=case["chunk_size"],
        )
        torch.npu.synchronize()
        self.assertTensorClose(out, case["out_expected"])

    def test_seq32(self):
        self._run_case("seq32.pt")

    def test_seq2047(self):
        self._run_case("seq2047.pt")
```

## 8. 调试技巧

1. **先检查 GPU dump 的 shape**：打印所有输入的 shape/dtype，确认 `Hk` 和 `Hv` 的关系
2. **head 0 总是匹配**：GQA 下 head 0 使用相同的 k[0] 和 g[0]，若 head 0 不匹配说明 layout 转换或 dtype 有问题
3. **逐 head 比较**：打印每个 head 的 max_diff，若 head 0 匹配但 head 1+ 不匹配，说明 GQA head 映射有问题
4. **读 tiling 代码**：`chunk_scaled_dot_kkt_tiling.cpp` 中 `IsChunkSizeSupported` 限制了 chunk_size 取值
5. **beta 必须是 float32**：op def 声明 `DT_FLOAT`，GPU dump 是 float16 需转换
6. **输出 head 维 = Hk**：NPU 输出 `(B, Hk, T, chunk_size)`，不是 `Hv`；需子采样 GPU 输出后比较
7. **chunk_indices 用 chunk_size**：直接按 chunk_size 分块，不需要 block_t
8. **g_cumsum 是累积值**：GPU dump 中的 `g_cumsum` 已是 chunk 内 cumulative gate，直接传给 NPU 的 `g` 参数

## 9. GPU 与 NPU 布局策略对比

| 维度 | GPU dump (rtp-llm) | NPU 算子 |
|------|---------------------|---------|
| `k` 布局 | `(B, T, Hk, Dk)` | `(B, Hk, T, Dk)` — 需 transpose(1,2) |
| `g` 布局 | `(B, T, Hv)` | `(B, Hv, T)` — 需 transpose(1,2) |
| `beta` 布局 | `(B, T, Hv)` | `(B, Hv, T)` — 需 transpose(1,2) |
| `beta` dtype | float16 | **float32** — 需 `.to(torch.float32)` |
| `g` dtype | float32 | float32（一致） |
| `k` dtype | float16 | float16/bfloat16（一致） |
| `cu_seqlens` dtype | int32 | int64 list |
| `chunk_size` | 需推断 | 仅支持 16/32/64/128 |
| `chunk_indices` | 需计算 | int64 list |
| GQA head 映射 | GPU head h_v 用 `k[h_v//ratio]` + `g[h_v]` | NPU head h_k 用 `k[h_k]` + `g[h_k]` |
| GQA head 子采样 | g/beta 有 Hv 个 head | 需子采样到 Hk 个 head: `[:, ::ratio]` |
| 输出 head 维 | Hv | **Hk** — 需子采样 GPU 输出 |
| 输出 `A` 布局 | `(B, T, Hv, chunk_size)` | `(B, Hk, T, chunk_size)` — 需 transpose + 子采样 |
| 原地更新 | 无 | 无 |
| 非连续性 | 全部 contiguous | `AutoContiguous()` 自动处理 |

**结论**：`chunk_scaled_dot_kkt` 的核心比对难点在于三个转换：(1) q/k/v/g/beta 的 layout 转换 `(B, T, H, D) → (B, H, T, D)`、(2) GQA head 子采样——NPU 输出 Hk 个 head 而非 Hv，需对 g/beta 和输出做 `[:, ::ratio]` 子采样使 head 对齐、(3) beta 的 dtype 从 float16 转为 float32。相比 `chunk_fwd_o`，该算子多了 GQA head 映射问题；相比 `recurrent_gated_delta_rule`，该算子无非连续 stride 问题、无 state 转置问题、无原地更新问题。
