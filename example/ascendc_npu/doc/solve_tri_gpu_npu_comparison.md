# solve_tri 算子 GPU-NPU 精度比对指南

本文档总结 `solve_tri` 算子 NPU-vs-GPU 比对的实践经验。通用方法论参见同目录下的 `gpu_npu_comparison_guide.md`。

## 整体流程

```
GPU 黄金数据 (.pt)  →  解析输入/输出  →  dtype 转换 float32→float16  →  cu_seqlens 转换  →  计算 chunk_indices  →  调用 NPU 算子  →  精度比对
```

## 1. 理解 GPU 数据格式

### 数据来源

GPU dump 位于 `<workspace>/sample/solve_tril/`。

### 典型数据内容

以 `seq32.pt` 为例：

| key | 类型 | 值 |
|-----|------|----|
| `inputs/A` | tensor | shape=(1, 32, 32, 64), dtype=float32, contiguous |
| `inputs/cu_seqlens` | tensor | shape=(2,), dtype=int32, 值=[0, 32] |
| `inputs/output_dtype` | dtype | torch.float16 |
| `outputs` | tensor | shape=(1, 32, 32, 64), dtype=float16 |

### 与其他算子的关键差异

| 维度 | chunk_fwd_o | chunk_scaled_dot_kkt | solve_tri |
|------|-------------|---------------------|-----------|
| GPU 布局 | `(B, T, H, D)` | `(B, T, H, D)` | `(B, T, H, D)` — **BSND，与 NPU 一致** |
| Layout 转换 | 需要 transpose(1,2) | 需要 transpose(1,2) | **不需要 transpose** |
| 输入 dtype | float16 | float16 | **float32**（GPU dump），NPU 要求 float16/bf16 |
| 输出 dtype | float16 | float32 | **float16**（由 output_dtype 决定） |
| 原地更新 | 无 | 无 | 无 |

**solve_tri 是最简单的算子**：GPU 和 NPU 的 layout 完全一致（BSND），无需 transpose；只需 dtype 转换和 cu_seqlens/chunk_indices 构造。

## 2. 格式转换

### 2.1 dtype 转换（核心）

GPU dump 中 A 是 float32，NPU 算子仅支持 float16/bfloat16。GPU dump 的 `output_dtype` 是 float16，应使用 float16 匹配：

```python
# ✓ 正确：使用 float16（匹配 GPU output_dtype）
A_npu = A_gpu.to(torch.float16)

# ✗ 错误：使用 bfloat16（长序列精度不足，max_diff=0.158 > atol=0.05）
A_npu = A_gpu.to(torch.bfloat16)
```

**bfloat16 精度问题**：seq2047（32 chunks）场景下，bfloat16 的 max_diff=0.158（相对误差 32x），而 float16 的 max_diff=0.025（在容差内）。这是因为 solve_tri 的前代法对中间值的精度敏感，bfloat16 的 7 位尾数不足以精确表示累积求和。

### 2.2 Layout — 无需转换

GPU dump 是 `(B, T, H, chunk_size)` BSND 布局，NPU `solve_tri` 的 `layout="bsnd"` 参数也期望相同布局。**无需 transpose**。

### 2.3 cu_seqlens dtype 转换

```python
cu_seqlens = [int(v) for v in inputs["cu_seqlens"].tolist()]
```

### 2.4 chunk_indices 计算

```python
chunk_indices = _prepare_chunk_indices(cu_seqlens, chunk_size)
```

### 2.5 chunk_size 推断

```python
chunk_size = A_gpu.shape[-1]  # 64
```

NPU 仅支持 chunk_size 为 64 或 128。

## 3. NPU 算子约束

| 参数 | NPU 期望 | 说明 |
|------|---------|------|
| `x` | `(B, T, H, chunk_size)` float16/bfloat16 | BSND 布局，**无需 transpose** |
| `cu_seqlens` | int64 list | varlen 模式必填 |
| `chunk_indices` | int64 list | varlen 模式必填 |
| `layout` | string | 默认 "bsnd"，支持 "bsnd"/"tnd" |
| `x_out` (输出) | `(B, T, H, chunk_size)` float16/bfloat16 | 同输入 dtype |

### 算子功能

计算 `(I + A)^{-1}`，其中 A 是严格下三角矩阵（对角线为 0）。

## 4. 常见错误

### 4.1 bfloat16 精度不足导致长序列失败

bfloat16 的 7 位尾数在 solve_tri 的前代法累积求和中精度不足。seq2047 场景下 max_diff=0.158（超出 atol=0.05），而 float16 的 max_diff=0.025（在容差内）。

```python
# ✗ 错误：bfloat16，seq2047 max_diff=0.158
A_npu = A_gpu.to(torch.bfloat16)

# ✓ 正确：float16，seq2047 max_diff=0.025
A_npu = A_gpu.to(torch.float16)
```

### 4.2 GPU dump A 是 float32

GPU dump 中 A 是 float32（KKT 的输出），但 NPU 算子仅支持 float16/bfloat16。不转换会导致 `aclnnStatus=161001`。

### 4.3 chunk_size 不在支持列表

NPU 仅支持 `chunk_size ∈ {64, 128}`。

## 5. 测试模板

完整测试文件位于 `torch_custom/fla_npu/test/test_npu_solve_tri_gpu_golden.py`。

```python
import math, os, unittest, torch
from fla_npu.ops import ascendc as ascendc_ops

torch.npu.set_device(int(os.environ.get("TEST_DEVICE_ID", 0)))
_DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__),
    "..", "..", "..", "..", "sample", "solve_tril"))

def _prepare_chunk_indices(cu_seqlens, chunk_size):
    rows = []
    for seq_idx, (start, end) in enumerate(zip(cu_seqlens[:-1], cu_seqlens[1:])):
        for chunk_idx in range(math.ceil((end - start) / chunk_size)):
            rows.append((seq_idx, chunk_idx))
    return [v for row in rows for v in row]

def _load_gpu_case(filename):
    data = torch.load(os.path.join(_DATA_DIR, filename), map_location="cpu", weights_only=False)
    inputs = data["inputs"]
    A_gpu = inputs["A"]
    cu_seqlens = [int(v) for v in inputs["cu_seqlens"].tolist()]
    chunk_size = A_gpu.shape[-1]
    chunk_indices = _prepare_chunk_indices(cu_seqlens, chunk_size)
    # float16 to match GPU output_dtype (bfloat16 has precision issues)
    A_npu = A_gpu.to(torch.float16)
    return {
        "A": A_npu, "cu_seqlens": cu_seqlens, "chunk_indices": chunk_indices,
        "chunk_size": chunk_size, "out_expected": data["outputs"],
    }

class TestSolveTriGpuGolden(unittest.TestCase):
    rtol = 5e-2
    atol = 5e-2

    def call_op(self, **kwargs):
        return ascendc_ops.npu_solve_tri(**kwargs)

    def _run_case(self, filename):
        case = _load_gpu_case(filename)
        out = self.call_op(
            x=case["A"].npu(),
            cu_seqlens=case["cu_seqlens"],
            chunk_indices=case["chunk_indices"],
            layout="bsnd")
        torch.npu.synchronize()
        self.assertTensorClose(out, case["out_expected"])

    def test_seq32(self):
        self._run_case("seq32.pt")

    def test_seq2047(self):
        self._run_case("seq2047.pt")
```

## 6. 调试技巧

1. **检查 output_dtype**：GPU dump 中的 `output_dtype` 决定应使用 float16 还是 bfloat16
2. **bfloat16 vs float16**：bfloat16 在前代法累积求和中精度不足，长序列建议用 float16
3. **无需 transpose**：solve_tri 使用 BSND 布局，GPU 和 NPU 一致
4. **chunk_size 从 A shape 推断**：`A.shape[-1]` 即 chunk_size
5. **A 是严格下三角**：对角线为 0，solve_tri 内部设为 1 后求逆

## 7. GPU 与 NPU 布局策略对比

| 维度 | GPU dump (rtp-llm) | NPU 算子 |
|------|---------------------|---------|
| `A/x` 布局 | `(B, T, H, chunk_size)` BSND | `(B, T, H, chunk_size)` BSND — **一致，无需 transpose** |
| `A` dtype | float32 | float16/bfloat16 — 需转换 |
| `output` dtype | float16（由 output_dtype 决定） | 同输入 dtype |
| `cu_seqlens` dtype | int32 | int64 list |
| `chunk_size` | 需推断 | 仅支持 64/128 |
| `chunk_indices` | 需计算 | int64 list |
| 原地更新 | 无 | 无 |

**结论**：`solve_tri` 是所有 GDN chunk 算子中最简单的——GPU 和 NPU 的 layout 完全一致（BSND），无需 transpose。唯一的转换是 dtype（float32 → float16）和 cu_seqlens/chunk_indices 的构造。核心注意点是使用 float16 而非 bfloat16，以避免长序列前代法的精度问题。
