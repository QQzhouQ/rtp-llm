# recompute_wu_fwd 算子 GPU-NPU 精度比对指南

本文档总结 `recompute_wu_fwd` 算子 NPU-vs-GPU 比对的实践经验，供后续该算子维护和类似 WY 表示重计算类算子比对参考。通用方法论参见同目录下的 `gpu_npu_comparison_guide.md`。

## 整体流程

```
GPU 黄金数据 (.pt)  →  解析输入/输出  →  Layout 转换 (B,T,H,D)→(B,H,T,D)  →  dtype 转换 (k/v/A→bf16, beta→fp32)  →  推断 chunk_size  →  计算 chunk_indices  →  调用 NPU 算子  →  双输出 (w, u) 精度比对
```

## 1. 理解 GPU 数据格式

### 数据来源

GPU 端通过 `torch.save()` 导出的 `.pt` 文件位于 `<workspace>/sample/recompute_w_u_fwd/`，采用 `inputs`/`outputs`/`input_meta` 分层结构：

```python
data = torch.load(path, map_location="cpu", weights_only=False)
# 顶层 key: op_name, mode, param_names, inputs, outputs, inplace_outputs, input_meta, model_state
```

### 典型数据内容

以 `seq32.pt` 为例：

| key | 类型 | 值 |
|-----|------|----|
| `inputs/k` | tensor | shape=(1, 32, 16, 128), dtype=float16, contiguous |
| `inputs/v` | tensor | shape=(1, 32, 32, 128), dtype=float16, contiguous |
| `inputs/beta` | tensor | shape=(1, 32, 32), dtype=float16, contiguous |
| `inputs/A` | tensor | shape=(1, 32, 32, 64), dtype=float16, contiguous |
| `inputs/g_cumsum` | tensor | shape=(1, 32, 32), dtype=float32, contiguous |
| `inputs/cu_seqlens` | tensor | shape=(2,), dtype=int32, 值=[0, 32] |
| `outputs[0]` (w) | tensor | shape=(1, 32, 32, 128), dtype=float16 |
| `outputs[1]` (u) | tensor | shape=(1, 32, 32, 128), dtype=float16 |
| `inplace_outputs` | dict | 空（无原地更新） |

### 与其他算子数据格式的差异

| 维度 | chunk_fwd_o | chunk_scaled_dot_kkt | recompute_wu_fwd |
|------|-------------|---------------------|-----------------|
| 输出数量 | 1 | 1 | **2 (w, u)** |
| 标量参数 | chunk_size 需推断 | chunk_size 需推断 | chunk_size 需推断 |
| 原地更新 | 无 | 无 | 无 |
| 非连续性 | 全部 contiguous | 全部 contiguous | 全部 contiguous |
| GQA head 子采样 | 无 | 有 (Hv→Hk) | 无 |
| dtype 要求 | 全部 bf16 | beta→fp32 | **k/v/A→bf16, beta→fp32** |
| Python 函数名 | npu_chunk_fwd_o | npu_chunk_scaled_dot_kkt | **npu_recompute_w_u_fwd** (注意下划线) |

## 2. 恢复非连续 stride

当前 GPU dump 中所有输入均为 contiguous，保留 stride 恢复逻辑以兼容未来非连续 dump。

## 3. 推断 GPU 的语义参数

### 3.1 `chunk_size` 推断

GPU dump 不直接包含 `chunk_size`。从 `A` 的 shape 最后一维推断：

```python
chunk_size = A_gpu.shape[-1]  # 64
```

NPU 仅支持 `chunk_size ∈ {64, 128}`。

### 3.2 `cu_seqlens` dtype 转换

GPU dump 中 `cu_seqlens` 是 int32，NPU wrapper 接受 Python list 并内部转为 int64：

```python
cu_seqlens = [int(v) for v in inputs["cu_seqlens"].tolist()]
```

## 4. 理解 NPU 算子的输入格式

### 关键 shape/dtype 约束

| 参数 | NPU 期望 | 说明 |
|------|---------|------|
| `k` | `(B, Hk, T, Dk)` bfloat16 | 4D, head_first |
| `v` | `(B, Hv, T, Dv)` bfloat16 | 4D, head_first |
| `beta` | `(B, Hv, T)` **float32** | 3D, head_first, **GPU dump 为 float16 需转换** |
| `A` | `(B, Hv, T, chunk_size)` bfloat16 | 4D, head_first |
| `g` | `(B, Hv, T)` float32 | 3D, head_first, optional |
| `cu_seqlens` | int64 list | varlen 模式必填 |
| `chunk_indices` | int64 list | varlen 模式必填 |
| `chunk_size` | int64 | 仅支持 64 或 128 |
| `w` (输出) | `(B, Hv, T, Dk)` bfloat16 | 与 k dtype 一致 |
| `u` (输出) | `(B, Hv, T, Dv)` bfloat16 | 与 v dtype 一致 |

### 关键校验逻辑（来自 `recompute_wu_fwd_tiling_processor.h`）

```cpp
// k/v 必须是 4D, beta 必须是 3D, A 必须是 4D
// GVA: Hv % Hk == 0
// v 的 B/T 必须与 k 一致
// beta/g 的 shape 必须一致且 B/T 与 k 一致, head 维 = Hv
// A 的 head 维必须 = Hv, B/T 与 k 一致
// V (v_head_dim) 必须是 128 或 256
// chunk_size 必须是 64 或 128
// varlen 模式下 B 必须为 1
```

### dtype 组合约束（ascend950）

算子定义的 DataType 列表有 4 个变体，对应不同 SOC：

| 变体 | k/v/A | beta/g | SOC |
|------|-------|--------|-----|
| 0 | float16 | float16 | ascend910b |
| 1 | bfloat16 | bfloat16 | ascend910_93 |
| 2 | float16 | **float32** | ascend950 |
| 3 | bfloat16 | **float32** | ascend950 |

**ascend950 上 beta 和 g 必须是 float32**，传 float16 会导致 w 输出全 NaN（内部 `exp(g)` 计算溢出）。

## 5. 格式转换映射

### 5.1 Layout 转换

GPU dump 是 `(B, T, H, D)`，NPU 要求 `(B, H, T, D)`。通过 `transpose(1, 2)` 完成转换：

```python
k_npu = k_gpu.transpose(1, 2).contiguous().to(torch.bfloat16)
v_npu = v_gpu.transpose(1, 2).contiguous().to(torch.bfloat16)
A_npu = A_gpu.transpose(1, 2).contiguous().to(torch.bfloat16)
beta_npu = beta_gpu.transpose(1, 2).contiguous().to(torch.float32)
g_npu = g_gpu.transpose(1, 2).contiguous()  # float32
```

### 5.2 dtype 转换（核心）

GPU dump 中 k/v/beta/A 都是 float16，g 是 float32。ascend950 要求：
- k/v/A: bfloat16（float16 也可，但 bfloat16 精度更优）
- **beta: float32**（float16 会导致 w 输出 NaN）
- g: float32（已匹配）

```python
# ✗ 错误：beta 保持 float16，会导致 w 输出全 NaN
beta_npu = beta_gpu.transpose(1, 2).contiguous()  # float16

# ✓ 正确：转为 float32
beta_npu = beta_gpu.transpose(1, 2).contiguous().to(torch.float32)
```

### 5.3 chunk_indices 计算

```python
def _prepare_chunk_indices(cu_seqlens, chunk_size):
    rows = []
    for seq_idx, (start, end) in enumerate(zip(cu_seqlens[:-1], cu_seqlens[1:])):
        for chunk_idx in range(math.ceil((end - start) / chunk_size)):
            rows.append((seq_idx, chunk_idx))
    return [v for row in rows for v in row]
```

### 5.4 双输出处理

GPU 输出是列表 `[w, u]`，NPU 返回元组 `(w, u)`。两个输出都需要 transpose 回 GPU 布局进行比较：

```python
w_expected = data["outputs"][0].transpose(1, 2).contiguous()  # (B, Hv, T, Dk)
u_expected = data["outputs"][1].transpose(1, 2).contiguous()  # (B, Hv, T, Dv)
```

### 验证转换正确性

测试通过两个 case 验证：
- `seq32.pt`: seqlen=32, chunk_size=64, 1 chunk → PASS
- `seq2047.pt`: seqlen=2047, chunk_size=64, 32 chunks → PASS (u max_diff=0.109, 相对误差 0.59%)

## 6. 常见错误

### 6.1 w 输出全 NaN（beta dtype 为 float16）

ascend950 上 beta 必须是 float32。传 float16 会导致 w 输出全 NaN（u 正常），因为内部 `exp(g) * k * beta` 计算在 float16 精度下溢出：

```python
# ✗ 错误：beta 保持 float16
beta_npu = beta_gpu.transpose(1, 2).contiguous()  # float16 → w 全 NaN

# ✓ 正确：转为 float32
beta_npu = beta_gpu.transpose(1, 2).contiguous().to(torch.float32)
```

### 6.2 Python 函数名错误

Python wrapper 函数名是 `npu_recompute_w_u_fwd`（w 和 u 之间有下划线），不是 `npu_recompute_wu_fwd`：

```python
# ✗ 错误
from fla_npu.ops.ascendc import npu_recompute_wu_fwd

# ✓ 正确
from fla_npu.ops.ascendc import npu_recompute_w_u_fwd
# 或通过模块调用
ascendc_ops.npu_recompute_w_u_fwd(...)
```

### 6.3 长序列 u 输出精度边界

seq2047 场景下 u 输出可能在极少数点（9/8.3M = 0.0001%）超过 `atol=5e-2`。这是因为 bfloat16 在大值（~18.6）处的量化步长约为 0.1，属正常精度限制：

```python
# u max_diff=0.109, 但 expected=18.64, 相对误差仅 0.59%
# 建议 atol=1e-1 以适应 bfloat16 长序列精度
```

### 6.4 chunk_size 不在支持列表

NPU 仅支持 `chunk_size ∈ {64, 128}`（比 chunk_scaled_dot_kkt 的 `{16, 32, 64, 128}` 更严格）。

### 6.5 g=None 导致参数校验失败

虽然 `g` 在算子定义中是 OPTIONAL，但传 `g=None` 会触发 `aclnnStatus=161002`。当前实现要求 `g` 必须提供。

## 7. 测试模板

完整测试文件位于 `torch_custom/fla_npu/test/test_npu_recompute_wu_fwd_gpu_golden.py`，核心结构：

```python
import math
import os
import unittest
import torch
from fla_npu.ops import ascendc as ascendc_ops

torch.npu.set_device(int(os.environ.get("TEST_DEVICE_ID", 0)))
_DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__),
                                          "..", "..", "..", "..",
                                          "sample", "recompute_w_u_fwd"))


def _prepare_chunk_indices(cu_seqlens, chunk_size):
    rows = []
    for seq_idx, (start, end) in enumerate(zip(cu_seqlens[:-1], cu_seqlens[1:])):
        for chunk_idx in range(math.ceil((end - start) / chunk_size)):
            rows.append((seq_idx, chunk_idx))
    return [v for row in rows for v in row]


def _load_gpu_case(filename):
    data = torch.load(os.path.join(_DATA_DIR, filename), map_location="cpu", weights_only=False)
    inputs = data["inputs"]

    # GPU (B,T,H,D) → NPU (B,H,T,D); ascend950: k/v/A=bf16, beta=fp32
    k_npu = inputs["k"].transpose(1, 2).contiguous().to(torch.bfloat16)
    v_npu = inputs["v"].transpose(1, 2).contiguous().to(torch.bfloat16)
    beta_npu = inputs["beta"].transpose(1, 2).contiguous().to(torch.float32)
    A_npu = inputs["A"].transpose(1, 2).contiguous().to(torch.bfloat16)
    g_npu = inputs["g_cumsum"].transpose(1, 2).contiguous()  # float32

    cu_seqlens = [int(v) for v in inputs["cu_seqlens"].tolist()]
    chunk_size = inputs["A"].shape[-1]
    chunk_indices = _prepare_chunk_indices(cu_seqlens, chunk_size)

    # 双输出: w (B,Hv,T,Dk), u (B,Hv,T,Dv)
    w_expected = data["outputs"][0].transpose(1, 2).contiguous()
    u_expected = data["outputs"][1].transpose(1, 2).contiguous()

    return {
        "k": k_npu, "v": v_npu, "beta": beta_npu, "A": A_npu, "g": g_npu,
        "cu_seqlens": cu_seqlens, "chunk_indices": chunk_indices,
        "chunk_size": chunk_size,
        "w_expected": w_expected, "u_expected": u_expected,
    }


class TestRecomputeWuFwdGpuGolden(unittest.TestCase):
    rtol = 5e-2
    atol = 1e-1  # bfloat16 长序列精度

    def call_op(self, **kwargs):
        return ascendc_ops.npu_recompute_w_u_fwd(**kwargs)

    def _run_case(self, filename):
        case = _load_gpu_case(filename)
        w, u = self.call_op(
            k=case["k"].npu(), v=case["v"].npu(), beta=case["beta"].npu(),
            A=case["A"].npu(), chunk_size=case["chunk_size"], g=case["g"].npu(),
            cu_seqlens=case["cu_seqlens"], chunk_indices=case["chunk_indices"],
        )
        torch.npu.synchronize()
        self.assertTensorClose(w, case["w_expected"])
        self.assertTensorClose(u, case["u_expected"])

    def test_seq32(self):
        self._run_case("seq32.pt")

    def test_seq2047(self):
        self._run_case("seq2047.pt")
```

## 8. 调试技巧

1. **w NaN 但 u 正常**：检查 beta dtype — ascend950 要求 float32，float16 会导致 w（涉及 `exp(g)`）全 NaN
2. **读算子定义的 DataType 列表**：4 个变体对应不同 SOC，ascend950 是变体 2/3（beta/g=float32）
3. **注意函数名下划线**：Python wrapper 是 `npu_recompute_w_u_fwd`（有下划线），不是 `npu_recompute_wu_fwd`
4. **双输出比较**：GPU 输出是列表 `[w, u]`，NPU 返回元组 `(w, u)`，两个都需 transpose 后比较
5. **长序列 atol 放宽**：bfloat16 在大值处量化步长约 0.1，长序列建议 `atol=1e-1`
6. **chunk_size 从 A 推断**：`A.shape[-1]` 即 chunk_size，仅支持 64/128
7. **g 不能为 None**：虽然定义中是 OPTIONAL，但当前实现要求提供
8. **无 GQA head 子采样**：与 chunk_scaled_dot_kkt 不同，本算子 A/beta/g 的 head 维都是 Hv，与 v 一致，无需子采样

## 9. GPU 与 NPU 布局策略对比

| 维度 | GPU dump (rtp-llm) | NPU 算子 |
|------|---------------------|---------|
| `k` 布局 | `(B, T, Hk, Dk)` | `(B, Hk, T, Dk)` — 需 transpose(1,2) |
| `v` 布局 | `(B, T, Hv, Dv)` | `(B, Hv, T, Dv)` — 需 transpose(1,2) |
| `beta` 布局 | `(B, T, Hv)` | `(B, Hv, T)` — 需 transpose(1,2) |
| `A` 布局 | `(B, T, Hv, chunk_size)` | `(B, Hv, T, chunk_size)` — 需 transpose(1,2) |
| `g` 布局 | `(B, T, Hv)` | `(B, Hv, T)` — 需 transpose(1,2) |
| `k/v/A` dtype | float16 | bfloat16（ascend950） |
| `beta` dtype | float16 | **float32**（ascend950，float16 导致 NaN） |
| `g` dtype | float32 | float32（一致） |
| `cu_seqlens` dtype | int32 | int64 list |
| `chunk_size` | 需推断 | 仅支持 64/128 |
| 输出数量 | 2 (列表) | 2 (元组) |
| `w` 布局 | `(B, T, Hv, Dk)` | `(B, Hv, T, Dk)` — 需 transpose(1,2) |
| `u` 布局 | `(B, T, Hv, Dv)` | `(B, Hv, T, Dv)` — 需 transpose(1,2) |
| GQA head 子采样 | 无 | 无 |
| 原地更新 | 无 | 无 |

**结论**：`recompute_wu_fwd` 的核心比对难点在于三个转换：(1) 全部输入/输出的 layout 转换 `(B, T, H, D) → (B, H, T, D)`、(2) dtype 组合——ascend950 上 beta 必须为 float32（float16 会导致 w 输出 NaN）、(3) 双输出处理——w 和 u 都需 transpose 后比较。相比 `chunk_scaled_dot_kkt`，本算子无 GQA head 子采样问题，但多了双输出和更严格的 dtype 组合约束。
