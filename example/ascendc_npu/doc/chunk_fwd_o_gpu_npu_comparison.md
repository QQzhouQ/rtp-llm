# chunk_fwd_o 算子 GPU-NPU 精度比对指南

本文档总结 `chunk_fwd_o` 算子 NPU-vs-GPU 比对的实践经验，供后续该算子维护和类似 chunk forward 类算子比对参考。通用方法论参见同目录下的 `gpu_npu_comparison_guide.md`。

## 整体流程

```
GPU 黄金数据 (.pt)  →  解析输入/输出/属性  →  Layout 转换 (B,T,H,D)→(B,H,T,D)  →  dtype 转换  →  推断 chunk_size  →  计算 chunk_indices  →  调用 NPU 算子  →  精度比对
```

## 1. 理解 GPU 数据格式

### 数据来源

GPU 端通过 `torch.save()` 导出的 `.pt` 文件位于 `<workspace>/sample/chunk_fwd_o/`，采用 `inputs`/`outputs`/`input_meta` 分层结构：

```python
data = torch.load(path, map_location="cpu", weights_only=False)
# 顶层 key: op_name, mode, param_names, inputs, outputs, inplace_outputs, input_meta, model_state
```

### 典型数据内容

以 `single_chunk_seq32_1chunks.pt` 为例：

| key | 类型 | 值 |
|-----|------|----|
| `inputs/q` | tensor | shape=(1, 32, 16, 128), dtype=float16, contiguous |
| `inputs/k` | tensor | shape=(1, 32, 16, 128), dtype=float16, contiguous |
| `inputs/v` | tensor | shape=(1, 32, 32, 128), dtype=float16, contiguous |
| `inputs/h` | tensor | shape=(1, 1, 32, 128, 128), dtype=float32, contiguous |
| `inputs/g` | tensor | shape=(1, 32, 32), dtype=float32, contiguous |
| `inputs/scale` | float | 0.0884 (= dk\*\*-0.5 = 128\*\*-0.5) |
| `inputs/cu_seqlens` | tensor | shape=(2,), dtype=int32, 值=[0, 32] |
| `outputs` | tensor | shape=(1, 32, 32, 128), dtype=float16 |
| `inplace_outputs` | dict | 空（chunk_fwd_o 无原地更新） |
| `input_meta` | dict | 记录原始 shape/stride/dtype/contiguous |

另一个测试文件 `multi_chunk_seq2047_32chunks.pt`：
- seqlen=2047, num_chunks=32, chunk_size=64
- q/k/v/h/g 的 dim 1 和 dim 2 更大，其余格式相同

### 与其他算子数据格式的差异

| 维度 | causal_conv1d | chunk_local_cumsum | recurrent_gated_delta_rule | chunk_fwd_o |
|------|--------------|-------------------|---------------------------|-------------|
| 标量参数存储 | 嵌在 `inputs` 中 | `reverse`/`scale` 为 None | `scale` 为 None | `scale` 在 inputs，`chunk_size` 需推断 |
| 原地更新 | `conv_states` 被修改 | 无 | `state` 被修改 | 无 |
| 非连续性 | `x`/`conv_state` 常非连续 | `g` 通常 contiguous | `initial_state` 非连续 | 全部 contiguous |
| 输出结构 | 单输出 tensor | 单输出 tensor | 列表 + `inplace_outputs` | 单输出 tensor |
| varlen 参数 | `query_start_loc` | `cu_seqlens` + `chunk_indices_out` | `actual_seq_lengths` + `ssm_state_indices` | `cu_seqlens` + `chunk_offsets` |

### `input_meta` 与 stride 恢复

当前 GPU dump 中所有 tensor 都是 contiguous，`input_meta` 记录的 `contiguous=True`。但仍保留 stride 恢复逻辑以兼容未来非连续 dump 数据：

```python
q_gpu = _restore_strided_tensor(inputs["q"], meta.get("q", {}))
# input_meta.q.contiguous=True 时直接返回 saved_data
```

## 2. 恢复非连续 stride

复用通用指南中的 `_restore_strided_tensor` 函数。当前 `chunk_fwd_o` 的 GPU dump 中所有输入均为 contiguous，但该函数保证未来出现非连续 dump 时测试无需修改。

## 3. 推断 GPU 的语义参数

### 3.1 `chunk_size` 推断

GPU dump 不直接包含 `chunk_size` 参数（`param_names` 列表中有但 `inputs` 字典中没有）。需从 `h` 的 shape 和 `cu_seqlens` 推断：

```python
num_chunks = h_gpu.shape[1]  # GPU h layout: (B, num_chunks, NV, DK, DV)
seqlen = cu_seqlens[-1]
chunk_size = math.ceil(seqlen / num_chunks)
```

以 `multi_chunk_seq2047_32chunks.pt` 为例：`seqlen=2047, num_chunks=32 → chunk_size=64`。

### 3.2 `scale` 默认值

GPU dump 中 `scale` 已提供（`0.0884 = 128**-0.5`），但若为 None 则使用默认值：

```python
scale = inputs.get("scale")
if scale is None:
    scale = float(dk ** -0.5)
```

### 3.3 `cu_seqlens` dtype 转换

GPU dump 中 `cu_seqlens` 是 int32，NPU wrapper 接受 Python list 并内部转为 int64：

```python
cu_seqlens = [int(v) for v in inputs["cu_seqlens"].tolist()]
```

## 4. 理解 NPU 算子的输入格式

### 关键 shape/dtype 约束

| 参数 | NPU 期望 | 说明 |
|------|---------|------|
| `q` | `(B, NK, T, DK)` bfloat16/float16 | 4D, head_first 布局 |
| `k` | `(B, NK, T, DK)` bfloat16/float16 | 必须与 q shape 完全一致 |
| `v` | `(B, NV, T, DV)` bfloat16/float16 | 4D, head_first 布局 |
| `h` | `(B, NV, num_chunks, DK, DV)` bfloat16/float16 | 5D, hidden state |
| `g` | `(B, NV, T)` float32 | 3D, head_first 布局 |
| `cu_seqlens` | int64 list | varlen 模式必填 |
| `chunk_indices` | int64 list | varlen 模式必填, 扁平化的 (seq_idx, chunk_idx) 对 |
| `scale` | float (double) | query 缩放因子 |
| `chunk_size` | int64 | 默认 64 |
| `out` | `(B, NV, T, DV)` | 同 v shape |

### 关键 C++ 校验逻辑（来自 `chunk_fwd_o_tiling_processor.h`）

```cpp
// q/k/v 必须是 4D
RequiredInputDimNumCheck(qShape, 4, "q");
RequiredInputDimNumCheck(kShape, 4, "k");
RequiredInputDimNumCheck(vShape, 4, "v");

// h 必须是 5D
RequiredInputDimNumCheck(hShape, 5, "h");

// g 必须是 3D
RequiredInputDimNumCheck(gShape, 3, "g");

// q 和 k 必须有完全相同的 shape
OP_CHECK_IF(qShape != kShape, ...);

// q/v 的 batch 和 seqlen 必须一致
OP_CHECK_IF(q.batch != v.batch || q.seqlen != v.seqlen, ...);

// g 必须匹配 v 的 batch/head_num/seqlen
OP_CHECK_IF(v.batch != g.batch || v.head_num != g.head_num || v.seqlen != g.seqlen, ...);

// h 必须匹配 v 的 batch/head_num, h.dk==q.dk, h.dv==v.dv
OP_CHECK_IF(h.batch != v.batch || h.head_num != v.head_num || h.dk != q.dk || h.dv != v.dv, ...);

// v 的 head_num 必须能被 q 的 head_num 整除 (GQA)
OP_CHECK_IF(v.head_num % q.head_num != 0, ...);

// v 的 head_dim 必须 <= 256
OP_CHECK_IF(v.head_dim > 256, ...);

// chunk_size 必须为正
OP_CHECK_IF(chunk_size <= 0, ...);

// chunk_offsets 的 dim 0 必须能被 2 整除 (每对是一个 (seq_idx, chunk_idx))
OP_CHECK_IF(chunk_offsets.dim0 % 2 != 0, ...);
```

### 与其他算子的约束差异

| 约束 | causal_conv1d | chunk_local_cumsum | chunk_fwd_o |
|------|--------------|-------------------|-------------|
| layout 限制 | 支持多种 layout | 仅 head_first=True | head_first (B,H,T,D) |
| dim 对齐 | `dim % 16 == 0` | 无对齐要求 | 无对齐要求 |
| chunk_size | 无 | 必须 2 的幂 | 必须 > 0 |
| varlen 必填参数 | `query_start_loc` | `cu_seqlens` + `chunk_indices_out` | `cu_seqlens` + `chunk_offsets` |
| 原地更新 | `conv_states` 被修改 | 无 | 无 |
| head_dim 限制 | 无 | 无 | DV <= 256 |
| GQA 支持 | 无 | 无 | NV % NK == 0 |

## 5. 格式转换映射

### 5.1 Layout 转换（核心）

GPU dump 是 `(B, T, H, D)`（`head_first=False`），NPU 要求 `(B, H, T, D)`（`head_first=True`）。通过 `transpose(1, 2)` 完成转换：

```python
# q/k: GPU (B, T, NK, DK) → NPU (B, NK, T, DK)
q_npu = q_gpu.transpose(1, 2).contiguous().to(torch.bfloat16)
k_npu = k_gpu.transpose(1, 2).contiguous().to(torch.bfloat16)

# v: GPU (B, T, NV, DV) → NPU (B, NV, T, DV)
v_npu = v_gpu.transpose(1, 2).contiguous().to(torch.bfloat16)

# g: GPU (B, T, NV) → NPU (B, NV, T)
g_npu = g_gpu.transpose(1, 2).contiguous()
```

所有输入在 `*_def.cpp` 中声明为 `AutoContiguous()`，转置后调用 `.contiguous()` 是安全的。

### 5.2 h 的 Layout 转换

GPU h 的布局是 `(B, num_chunks, NV, DK, DV)`，NPU 期望 `(B, NV, num_chunks, DK, DV)`。交换 dim 1 和 dim 2：

```python
# GPU: (B, num_chunks, NV, DK, DV) → NPU: (B, NV, num_chunks, DK, DV)
h_npu = h_gpu.transpose(1, 2).contiguous().to(torch.bfloat16)
```

### 5.3 dtype 转换

GPU dump 中 q/k/v 是 float16，h 是 float32。NPU 支持 bfloat16 和 float16：

```python
q_npu = q_gpu.transpose(1, 2).contiguous().to(torch.bfloat16)
h_npu = h_gpu.transpose(1, 2).contiguous().to(torch.bfloat16)
# g 保持 float32
```

### 5.4 cu_seqlens dtype 转换

GPU dump 中 `cu_seqlens` 是 int32，NPU wrapper 接受 Python list 并内部转为 int64：

```python
cu_seqlens = [int(v) for v in inputs["cu_seqlens"].tolist()]
```

### 5.5 chunk_indices 计算

varlen 模式下 NPU 要求提供 `chunk_indices`（即 `chunk_offsets`），格式为扁平化的 `(seq_idx, chunk_idx)` 对列表：

```python
def _prepare_chunk_indices(cu_seqlens, chunk_size):
    rows = []
    for seq_idx, (start, end) in enumerate(zip(cu_seqlens[:-1], cu_seqlens[1:])):
        for chunk_idx in range(math.ceil((end - start) / chunk_size)):
            rows.append((seq_idx, chunk_idx))
    return [v for row in rows for v in row]
```

以 `multi_chunk_seq2047_32chunks.pt` 为例：`cu_seqlens=[0, 2047]`, `chunk_size=64`
→ `chunk_indices = [0, 0, 0, 1, 0, 2, ..., 0, 31]`（32 对）。

### 5.6 输出 Layout 转换

GPU 输出是 `(B, T, NV, DV)`，NPU 输出是 `(B, NV, T, DV)`。比较时需将 NPU 输出转回 GPU 布局，或将 GPU 期望输出转为 NPU 布局：

```python
# GPU 期望输出转 NPU 布局
out_expected_npu = out_expected.transpose(1, 2).contiguous()
```

### 验证转换正确性

测试通过两个 case 验证：
- `single_chunk_seq32_1chunks.pt`: seqlen=32, chunk_size=32, 1 chunk → PASS
- `multi_chunk_seq2047_32chunks.pt`: seqlen=2047, chunk_size=64, 32 chunks → PASS

## 6. 常见错误

### 6.1 `aclnnStatus=161001`（参数校验失败）

通常是 shape/dtype 不匹配。逐项检查：

- [ ] **q/k/v 的 dim 顺序**：NPU 期望 `(B, H, T, D)` 而非 GPU dump 的 `(B, T, H, D)`
- [ ] **h 的 dim 顺序**：NPU 期望 `(B, NV, num_chunks, DK, DV)` 而非 GPU 的 `(B, num_chunks, NV, DK, DV)`
- [ ] **g 的 dim 顺序**：NPU 期望 `(B, NV, T)` 而非 GPU 的 `(B, T, NV)`
- [ ] **cu_seqlens dtype**：GPU dump 是 int32，需转为 int64 list
- [ ] **v head_dim > 256**：NPU 不支持
- [ ] **NV % NK != 0**：GQA ratio 必须整除
- [ ] **chunk_size <= 0**：必须为正

### 6.2 chunk_indices 计算错误导致 kernel 挂起

`chunk_indices` 必须使用 `chunk_size`（而非 `block_t`）计算 chunk 数量。使用错误的 block size 会导致 chunk 数量不匹配，kernel 挂起或结果错误：

```python
# ✗ 错误：使用 block_t（来自 chunk_local_cumsum 的公式）
block_t = next_power_of_two((1 << 17) // chunk_size)
chunk_indices = _prepare_chunk_indices(cu_seqlens, block_t)

# ✓ 正确：直接使用 chunk_size
chunk_indices = _prepare_chunk_indices(cu_seqlens, chunk_size)
```

### 6.3 chunk_size 推断错误

GPU dump 不直接包含 `chunk_size`。若推断错误（如使用 `seqlen // num_chunks` 而非 `ceil(seqlen / num_chunks)`），当 seqlen 不能被 num_chunks 整除时会导致错误：

```python
# ✗ 错误：整除（当 seqlen=2047, num_chunks=32 时得 63）
chunk_size = seqlen // num_chunks

# ✓ 正确：向上取整（得 64）
chunk_size = math.ceil(seqlen / num_chunks)
```

### 6.4 GPU .pt 文件路径找不到

测试默认从 `<workspace>/sample/chunk_fwd_o/` 加载数据。若路径不同，检查实际挂载位置：

```sh
find / -name "*.pt" -path "*/chunk_fwd_o/*" 2>/dev/null
```

### 6.5 scale 为 None 时未解析默认值

GPU dump 中 `scale` 已提供，但若为 None 表示使用默认值 `dk ** -0.5`：

```python
scale = inputs.get("scale")
if scale is None:
    scale = float(dk ** -0.5)
```

## 7. 测试模板

完整测试文件位于 `torch_custom/fla_npu/test/test_npu_chunk_fwd_o_gpu_golden.py`，核心结构：

```python
import math
import os
import unittest
import torch
from fla_npu.ops import ascendc as ascendc_ops

torch.npu.set_device(int(os.environ.get("TEST_DEVICE_ID", 0)))
_DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__),
                                          "..", "..", "..", "..",
                                          "sample", "chunk_fwd_o"))


def _restore_strided_tensor(saved_data, meta):
    if not meta or meta.get("contiguous", True):
        return saved_data
    dtype = getattr(torch, meta["dtype"].replace("torch.", ""))
    tensor = torch.empty_strided(meta["shape"], meta["stride"], dtype=dtype)
    tensor.copy_(saved_data)
    return tensor


def _prepare_chunk_indices(cu_seqlens, chunk_size):
    rows = []
    for seq_idx, (start, end) in enumerate(zip(cu_seqlens[:-1], cu_seqlens[1:])):
        for chunk_idx in range(math.ceil((end - start) / chunk_size)):
            rows.append((seq_idx, chunk_idx))
    return [v for row in rows for v in row]


def _load_gpu_case(filename):
    data = torch.load(os.path.join(_DATA_DIR, filename), map_location="cpu", weights_only=False)
    inputs = data["inputs"]
    meta = data.get("input_meta", {})

    # 恢复 stride（当前全部 contiguous）
    q_gpu = _restore_strided_tensor(inputs["q"], meta.get("q", {}))
    k_gpu = _restore_strided_tensor(inputs["k"], meta.get("k", {}))
    v_gpu = _restore_strided_tensor(inputs["v"], meta.get("v", {}))
    h_gpu = _restore_strided_tensor(inputs["h"], meta.get("h", {}))
    g_gpu = _restore_strided_tensor(inputs["g"], meta.get("g", {}))

    # GPU (B,T,H,D) → NPU (B,H,T,D) via transpose(1,2)
    q_npu = q_gpu.transpose(1, 2).contiguous().to(torch.bfloat16)
    k_npu = k_gpu.transpose(1, 2).contiguous().to(torch.bfloat16)
    v_npu = v_gpu.transpose(1, 2).contiguous().to(torch.bfloat16)
    h_npu = h_gpu.transpose(1, 2).contiguous().to(torch.bfloat16)
    g_npu = g_gpu.transpose(1, 2).contiguous()

    # cu_seqlens: int32 → int64 list
    cu_seqlens = [int(v) for v in inputs["cu_seqlens"].tolist()]

    # 推断 chunk_size
    num_chunks = h_gpu.shape[1]
    seqlen = cu_seqlens[-1]
    chunk_size = math.ceil(seqlen / num_chunks)

    # chunk_indices
    chunk_indices = _prepare_chunk_indices(cu_seqlens, chunk_size)

    # scale
    scale = inputs.get("scale")
    if scale is None:
        scale = float(q_npu.shape[-1] ** -0.5)

    # 输出转 NPU 布局
    out_expected = data["outputs"].transpose(1, 2).contiguous()

    return {
        "q": q_npu, "k": k_npu, "v": v_npu, "h": h_npu, "g": g_npu,
        "scale": scale, "cu_seqlens": cu_seqlens,
        "chunk_indices": chunk_indices, "chunk_size": chunk_size,
        "out_expected": out_expected,
    }


class TestChunkFwdOGpuGolden(unittest.TestCase):
    rtol = 5e-2
    atol = 5e-2

    def call_op(self, **kwargs):
        return ascendc_ops.npu_chunk_fwd_o(**kwargs)

    def assertTensorClose(self, actual, expected, *, rtol=None, atol=None):
        rtol = self.rtol if rtol is None else rtol
        atol = self.atol if atol is None else atol
        actual_cpu = actual.detach().cpu().float()
        expected_cpu = expected.detach().cpu().float()
        self.assertTrue(
            torch.allclose(actual_cpu, expected_cpu, rtol=rtol, atol=atol),
            msg=f"max_abs_diff={(actual_cpu - expected_cpu).abs().max().item():.6f}",
        )

    def _run_case(self, filename):
        case = _load_gpu_case(filename)
        out = self.call_op(
            q=case["q"].npu(), k=case["k"].npu(), v=case["v"].npu(),
            h=case["h"].npu(), scale=case["scale"], g=case["g"].npu(),
            cu_seqlens=case["cu_seqlens"],
            chunk_indices=case["chunk_indices"],
            chunk_size=case["chunk_size"],
        )
        torch.npu.synchronize()
        self.assertTensorClose(out, case["out_expected"])

    def test_single_chunk_seq32(self):
        self._run_case("single_chunk_seq32_1chunks.pt")

    def test_multi_chunk_seq2047(self):
        self._run_case("multi_chunk_seq2047_32chunks.pt")
```

## 8. 调试技巧

1. **先检查 GPU dump 的 shape**：打印所有输入的 shape/dtype/stride，确认 GPU 布局是 `(B, T, H, D)` 还是 `(B, H, T, D)`
2. **读 tiling processor 代码**：`chunk_fwd_o_tiling_processor.h` 中每个 `OP_CHECK_IF` 对应一条约束
3. **推断 chunk_size**：`chunk_size` 不在 dump 中，从 `h.shape[1]`（num_chunks）和 `cu_seqlens[-1]`（seqlen）推断
4. **chunk_indices 用 chunk_size 而非 block_t**：`chunk_fwd_o` 的 chunk_offsets 直接按 `chunk_size` 分块，不需要 `block_t`（那是 `chunk_local_cumsum` 的概念）
5. **所有输入都是 AutoContiguous**：转置后 `.contiguous()` 是安全的，NPU 框架会自动处理
6. **cu_seqlens 转为 int64 list**：GPU dump 是 int32 tensor，NPU wrapper 接受 Python list
7. **h 的 transpose(1,2)**：GPU h 是 `(B, num_chunks, NV, DK, DV)`，需要交换 dim 1/2 变为 `(B, NV, num_chunks, DK, DV)`
8. **输出也需要 transpose**：GPU 输出是 `(B, T, NV, DV)`，NPU 输出是 `(B, NV, T, DV)`

## 9. GPU 与 NPU 布局策略对比

| 维度 | GPU dump (rtp-llm) | NPU 算子 |
|------|---------------------|---------|
| `q/k` 布局 | `(B, T, NK, DK)` | `(B, NK, T, DK)` — 需 transpose(1,2) |
| `v` 布局 | `(B, T, NV, DV)` | `(B, NV, T, DV)` — 需 transpose(1,2) |
| `h` 布局 | `(B, num_chunks, NV, DK, DV)` | `(B, NV, num_chunks, DK, DV)` — 需 transpose(1,2) |
| `g` 布局 | `(B, T, NV)` | `(B, NV, T)` — 需 transpose(1,2) |
| `q/k/v` dtype | float16 | bfloat16（或 float16） |
| `h` dtype | float32 | bfloat16（或 float16） |
| `g` dtype | float32 | float32（一致） |
| `cu_seqlens` dtype | int32 | int64 list |
| `chunk_size` | 需推断 | int64，默认 64 |
| `chunk_indices` | 需计算 | int64 list，扁平化 (seq_idx, chunk_idx) 对 |
| 输出 `o` 布局 | `(B, T, NV, DV)` | `(B, NV, T, DV)` — 需 transpose(1,2) |
| 原地更新 | 无 | 无 |
| 非连续性 | 全部 contiguous | `AutoContiguous()` 自动处理 |

**结论**：`chunk_fwd_o` 的核心比对难点在于四个转换：(1) q/k/v/g 的 layout 转换 `(B, T, H, D) → (B, H, T, D)`、(2) h 的 layout 转换 `(B, num_chunks, NV, DK, DV) → (B, NV, num_chunks, DK, DV)`、(3) chunk_size 的推断（从 h shape 和 cu_seqlens）、(4) chunk_indices 的正确计算（使用 chunk_size 而非 block_t）。相比 `recurrent_gated_delta_rule`，该算子无非连续 stride 问题、无 state 转置问题、无原地更新问题，整体复杂度较低。
