# chunk_local_cumsum 算子 GPU-NPU 精度比对指南

本文档总结 `chunk_local_cumsum` 算子 NPU-vs-GPU 比对的实践经验，供后续该算子维护和类似 cumsum 类算子比对参考。通用方法论参见同目录下的 `gpu_npu_comparison_guide.md`。

## 整体流程

```
GPU 黄金数据 (.pt)  →  恢复非连续 stride  →  解析输入/输出/属性  →  Layout 转换 (B,T,H)→(B,H,T)  →  计算 chunk_indices_out  →  调用 NPU 算子  →  转置回 GPU 布局  →  精度比对
```

## 1. 理解 GPU 数据格式

### 数据来源

GPU 端通过 `torch.save()` 导出的 `.pt` 文件位于 `<workspace>/sample/chunk_local_cumsum/`，采用 `inputs`/`outputs`/`input_meta` 分层结构：

```python
data = torch.load(path, map_location="cpu", weights_only=False)
# 顶层 key: op_name, mode, param_names, inputs, outputs, input_meta, model_state
```

### 典型数据内容

以 `seq2047.pt` 为例：

| key | 类型 | 值 |
|-----|------|----|
| `op_name` | str | `"chunk_local_cumsum"` |
| `mode` | str | `"seq2047"` |
| `param_names` | list | `['g', 'chunk_size', 'reverse', 'scale', 'cu_seqlens', 'head_first', 'output_dtype', 'kwargs']` |
| `inputs/g` | tensor | shape=(1, 2047, 32), dtype=float32, stride=(65504, 32, 1), contiguous=True |
| `inputs/chunk_size` | int | 64 |
| `inputs/reverse` | None | None 表示使用默认值 False |
| `inputs/scale` | None | None 表示使用默认值 1.0 |
| `inputs/cu_seqlens` | tensor | shape=(2,), dtype=int32, 值=[0, 2047] |
| `outputs` | tensor | shape=(1, 2047, 32), dtype=float32 |
| `input_meta/g` | dict | 记录原始 shape/stride/dtype/contiguous |

### 与 causal_conv1d 数据格式的差异

| 维度 | causal_conv1d | chunk_local_cumsum |
|------|--------------|-------------------|
| 标量参数存储 | 嵌在 `inputs` 中 | `reverse`/`scale` 为 `None` 时表示使用默认值 |
| `cu_seqlens` dtype | int64 | **int32**（需转换） |
| 非连续性 | `x`/`conv_state` 常为非连续 | `g` 在 dump 中是 contiguous（`input_meta` 仍保留但不一定非连续） |
| 原地更新 | `conv_states` 会被原地修改 | 无原地更新，输出是独立 tensor |
| 复杂度 | 高（paged conv_state、weight 转置、activation 推断） | 低（单输入 `g`、无 weight、无 conv_state） |

### `input_meta` 与 stride 恢复

`chunk_local_cumsum` 的 GPU dump 中 `g` 的 `input_meta` 记录了 contiguous=True，因此 stride 恢复是直通的。但**仍应保留恢复逻辑**，以兼容未来非连续 dump 数据：

```python
g_gpu = _restore_strided_tensor(inputs["g"], meta.get("g", {}))
# input_meta.g.contiguous=True 时直接返回 saved_data
```

## 2. 恢复非连续 stride

复用通用指南中的 `_restore_strided_tensor` 函数：

```python
def _restore_strided_tensor(saved_data: torch.Tensor, meta: dict) -> torch.Tensor:
    if not meta or meta.get("contiguous", True):
        return saved_data
    dtype = getattr(torch, meta["dtype"].replace("torch.", ""))
    tensor = torch.empty_strided(tuple(meta["shape"]), tuple(meta["stride"]), dtype=dtype)
    tensor.copy_(saved_data)
    return tensor
```

当前 `chunk_local_cumsum` 的 GPU dump 中 `g` 均为 contiguous，但该函数保证未来出现非连续 dump 时测试无需修改。

## 3. 推断 GPU 的语义参数

GPU dump 中 `reverse` 和 `scale` 存为 `None`，表示使用算子默认值。需显式解析：

```python
def _resolve_scalar(value, default):
    """GPU dump 存储 reverse/scale 为 None 时表示使用默认值。"""
    return default if value is None else value

reverse = _resolve_scalar(inputs.get("reverse"), False)   # 默认 False
scale  = float(_resolve_scalar(inputs.get("scale"), 1.0)) # 默认 1.0
```

### 用 CPU reference 验证 cumsum 轴

GPU dump 的 `g` 是 `(1, T, 32)`，需要确认 cumsum 沿哪个轴进行。用 CPU 参考实现分别尝试 dim=1 和 dim=2：

```python
# 假设 head_first=False: (B, T, H)，cumsum 沿 dim=1 (T 轴)
def chunk_cumsum_dim1(t, chunk, csl):
    out = torch.empty_like(t)
    for s, e in zip(csl[:-1], csl[1:]):
        for c0 in range(0, e - s, chunk):
            seg = t[:, s+c0:min(s+c0+chunk, e), :]
            out[:, s+c0:min(s+c0+chunk, e), :] = torch.cumsum(seg, dim=1)
    return out

# 假设 head_first=True: (B, H, T)，cumsum 沿 dim=2 (T 轴)
def chunk_cumsum_dim2(t, chunk, csl):
    out = torch.empty_like(t)
    for s, e in zip(csl[:-1], csl[1:]):
        for c0 in range(0, e - s, chunk):
            seg = t[:, :, s+c0:min(s+c0+chunk, e)]
            out[:, :, s+c0:min(s+c0+chunk, e)] = torch.cumsum(seg, dim=2)
    return out

r1 = chunk_cumsum_dim1(g, cs, csl)  # head_first=False
r2 = chunk_cumsum_dim2(g, cs, csl)  # head_first=True
print("match dim=1 (head_first=False):", torch.allclose(r1, y_gpu, atol=1e-5))  # True
print("match dim=2 (head_first=True):",  torch.allclose(r2, y_gpu, atol=1e-5))  # False
```

**验证结果**：GPU dump 使用 `head_first=False`（`(B, T, H)` 布局），cumsum 沿 dim=1（T 轴）。

## 4. 理解 NPU 算子的输入格式

### 关键 shape/属性约束

| 参数 | NPU 期望 | 说明 |
|------|---------|------|
| `g` | `(B, H, T)` rank-3 | **必须 head_first=True**，tiling 显式拒绝 head_first=False |
| `cu_seqlens` | `int64` Python list | varlen 模式可选；GPU dump 为 int32 需转换 |
| `chunk_indices_out` | `int64` Python list | **varlen 模式必填**（cu_seqlens 提供时） |
| `out` | 同 `g` shape | 输出 dtype 由 `output_dtype` 属性决定 |
| `chunk_size` | int | **必须为 2 的幂** |
| `reverse` | bool | 默认 False |
| `scale` | float | 默认 1.0 |
| `head_first` | bool | **必须为 True**（当前实现限制） |
| `output_dtype` | str | `"float32"` / `"float16"` / `"bfloat16"` / `"same"` |

### 关键 C++ 校验逻辑（来自 `chunk_local_cumsum_tiling.cpp`）

```cpp
// g 必须是 rank-3
OP_CHECK_IF(gShape.GetDimNum() != 3, ..., "g must be rank 3 for [B, H, T]");

// head_first=false 被显式拒绝
OP_CHECK_IF(!*headFirstPtr, ...,
            "head_first=false is not supported; ChunkLocalCumsum currently supports "
            "only [B, H, T] layout.");

// chunk_size 必须是 2 的幂
OP_CHECK_IF(!IsPowerOfTwo(chunkSize), ..., "chunk_size must be a power of two");

// varlen 模式下 chunk_indices_out 必填
OP_CHECK_IF(context->GetOptionalInputDesc(CHUNK_INDICES_INDEX) == nullptr, ...,
            "chunk_indices_out is required when cu_seqlens is provided.");
OP_CHECK_IF(chunkIndicesShapePtr->GetStorageShape().GetShapeSize() == 0, ...,
            "chunk_indices_out is required when cu_seqlens is not empty.");

// varlen 模式下 B 必须为 1
OP_CHECK_IF(batch != 1, ..., "B must be 1 when cu_seqlens is provided");
```

### 与 causal_conv1d 的约束差异

| 约束 | causal_conv1d | chunk_local_cumsum |
|------|--------------|-------------------|
| layout 限制 | 支持多种 layout | **仅 head_first=True** |
| dim 对齐 | `dim % 16 == 0` | 无对齐要求 |
| chunk_size | 无幂次约束 | **必须为 2 的幂** |
| varlen 必填参数 | `query_start_loc` | `cu_seqlens` + `chunk_indices_out` |
| 原地更新 | `conv_states` 被修改 | 无 |

## 5. 格式转换映射

### Layout 转换（核心）

GPU dump 是 `(B, T, H)`（`head_first=False`），NPU 要求 `(B, H, T)`（`head_first=True`）。
`chunk_local_cumsum` 沿 T 轴做分段 cumsum，转置只交换轴位置，不改变 T 轴上的数值序列：

```python
# GPU: g 是 (B, T, H) head_first=False
# NPU: g 需要是 (B, H, T) head_first=True
g_npu = g_gpu.transpose(1, 2).contiguous()   # (B, H, T)

# NPU 输出是 (B, H, T)，转置回 GPU 布局 (B, T, H) 进行比对
y_gpu_layout = y_npu.cpu().transpose(1, 2).contiguous()
```

### cu_seqlens dtype 转换

GPU dump 中 `cu_seqlens` 是 `int32`，NPU wrapper 接受 Python list 并内部转为 `int64`：

```python
cu_seqlens_list = [int(v) for v in inputs["cu_seqlens"].tolist()]
```

### chunk_indices_out 计算

varlen 模式下 NPU 要求提供 `chunk_indices_out`，其逻辑需与 tiling 侧 `BLOCK_T` 计算一致：

```python
def _next_power_of_two(value: int) -> int:
    value = max(value, 1)
    result = 1
    while result < value:
        result <<= 1
    return result

def _block_t(chunk_size: int) -> int:
    """镜像 tiling 侧 BLOCK_T = next_pow2((1<<17)/chunk_size)。"""
    return _next_power_of_two((1 << 17) // chunk_size)

def _prepare_chunk_indices(cu_seqlens, block_t: int):
    """生成扁平化的 chunk_indices_out list。"""
    rows = []
    for seq_idx, (start, end) in enumerate(zip(cu_seqlens[:-1], cu_seqlens[1:])):
        num_blocks = math.ceil((end - start) / block_t)
        for block_idx in range(num_blocks):
            rows.append((seq_idx, block_idx))
    return [value for row in rows for value in row]
```

以 `chunk_size=64` 为例：`block_t = next_pow2(131072 // 64) = next_pow2(2048) = 2048`。
对于 `cu_seqlens=[0, 2047]`：`num_blocks = ceil(2047/2048) = 1`，`chunk_indices_out = [0, 0]`。

### 验证转换正确性

在 CPU 上用转置后的 `g` 跑参考实现，确认转置后仍能匹配 GPU 输出：

```python
# 转置到 (B, H, T) 后做 chunk-local cumsum 沿 dim=2
g_ht = g_gpu.transpose(1, 2).contiguous()   # (1, H, T)
out = torch.empty_like(g_ht)
for s, e in zip(csl[:-1], csl[1:]):
    for c0 in range(0, e - s, cs):
        seg = g_ht[:, :, s+c0:min(s+c0+cs, e)]
        out[:, :, s+c0:min(s+c0+cs, e)] = torch.cumsum(seg, dim=2)
# 转回 (B, T, H) 与 GPU 输出比对
out_t_th = out.transpose(1, 2).contiguous()
assert torch.allclose(out_t_th, y_gpu, atol=1e-5)
```

## 6. 常见错误

### 6.1 `aclnnStatus=161001`（参数校验失败）

与 causal_conv1d 的 `561002` 不同，`chunk_local_cumsum` 的参数校验失败码为 `161001`。常见原因：

- [ ] **head_first=False**：tiling 显式拒绝，必须传 `head_first=True` 并将 `g` 转置为 `(B, H, T)`
- [ ] **chunk_size 非 2 的幂**：如 `chunk_size=65` 会被拒绝
- [ ] **varlen 模式缺少 chunk_indices_out**：提供 `cu_seqlens` 时必须同时提供 `chunk_indices_out`
- [ ] **varlen 模式 B≠1**：提供 `cu_seqlens` 时 `g` 的 batch 维必须为 1
- [ ] **cu_seqlens 未转 int64**：GPU dump 是 int32，需 `[int(v) for v in ...]`
- [ ] **output_dtype 与 out tensor dtype 不匹配**：wrapper 根据 `output_dtype` 创建 out tensor，需保持一致

### 6.2 `aclnnStatus=507021` 或符号解析失败

```
AttributeError: Unable to resolve aclnn symbol aclnnChunkLocalCumsumGetWorkspaceSize.
```

说明安装的 `libcust_opapi.so` 中未编译进 `chunk_local_cumsum` 算子的 op_api 符号。检查方法：

```sh
SO=$(python -c "import fla_npu; print(fla_npu.load_ascendc_opapi_libraries()[0]._name)")
nm -D "$SO" | grep -i ChunkLocalCumsum
```

若返回为空，需用包含该算子的完整构建重新安装 wheel：

```sh
FLA_NPU_SOC=ascend950 python -m pip wheel --no-build-isolation --no-deps . -w dist
python -m pip install --force-reinstall --no-deps dist/flash_linear_attention_npu-*-950.*.whl
```

### 6.3 输出精度不达标

- GPU 与 CPU reference 之间的容许差约为 **5e-4**（float32，因 cumsum 累加顺序差异）
- NPU 与 GPU 之间的容许差建议用 `rtol=1e-3, atol=2e-3`（float32 场景）
- float16/bfloat16 场景需适当放宽至 `rtol=1e-2, atol=5e-2`

### 6.4 GPU .pt 文件路径找不到

测试默认从 `<workspace>/sample/chunk_local_cumsum/` 加载数据。若容器内挂载路径不同，设置环境变量：

```sh
export CHUNK_LOCAL_CUMSUM_GOLDEN_DIR=/path/to/gpu/dump/dir
```

### 6.5 reverse/scale 为 None 时未解析默认值

GPU dump 中 `reverse` 和 `scale` 存为 `None` 表示使用算子默认值。直接传 `None` 给 NPU wrapper 会触发类型错误，必须显式解析：

```python
# ✗ 错误：直接传 None
y = op(g, chunk_size, reverse=None, scale=None)

# ✓ 正确：解析为默认值
reverse = False if inputs.get("reverse") is None else inputs["reverse"]
scale  = 1.0 if inputs.get("scale") is None else float(inputs["scale"])
```

## 7. 测试模板

完整测试文件位于 `torch_custom/fla_npu/test/test_npu_chunk_local_cumsum_gpu_golden.py`，核心结构：

```python
import math
import os
import unittest
import torch
from fla_npu.ops import ascendc as ascendc_ops

torch.npu.set_device(int(os.environ.get("TEST_DEVICE_ID", 0)))
_DATA_DIR = os.environ.get("CHUNK_LOCAL_CUMSUM_GOLDEN_DIR",
                           os.path.abspath(os.path.join(os.path.dirname(__file__),
                                                         "..", "..", "..", "..",
                                                         "sample", "chunk_local_cumsum")))


def _restore_strided_tensor(saved_data, meta):
    if not meta or meta.get("contiguous", True):
        return saved_data
    dtype = getattr(torch, meta["dtype"].replace("torch.", ""))
    tensor = torch.empty_strided(tuple(meta["shape"]), tuple(meta["stride"]), dtype=dtype)
    tensor.copy_(saved_data)
    return tensor


def _block_t(chunk_size):
    return _next_power_of_two((1 << 17) // chunk_size)


def _prepare_chunk_indices(cu_seqlens, block_t):
    rows = []
    for seq_idx, (start, end) in enumerate(zip(cu_seqlens[:-1], cu_seqlens[1:])):
        for block_idx in range(math.ceil((end - start) / block_t)):
            rows.append((seq_idx, block_idx))
    return [v for row in rows for v in row]


def _load_gpu_case(filename):
    data = torch.load(os.path.join(_DATA_DIR, filename), map_location="cpu", weights_only=False)
    inputs = data["inputs"]
    meta = data.get("input_meta", {})
    g_gpu = _restore_strided_tensor(inputs["g"], meta.get("g", {}))
    cu_seqlens = [int(v) for v in inputs["cu_seqlens"].tolist()]
    chunk_size = int(inputs["chunk_size"])
    block_t = _block_t(chunk_size)
    return {
        "g": g_gpu,
        "chunk_size": chunk_size,
        "reverse": False if inputs.get("reverse") is None else inputs["reverse"],
        "scale": 1.0 if inputs.get("scale") is None else float(inputs["scale"]),
        "cu_seqlens": cu_seqlens,
        "chunk_indices_out": _prepare_chunk_indices(cu_seqlens, block_t),
        "y_expected": data["outputs"].contiguous(),
    }


class TestChunkLocalCumsumGpuGolden(unittest.TestCase):
    rtol = 1e-3
    atol = 2e-3

    def call_op(self, **kwargs):
        return ascendc_ops.npu_chunk_local_cumsum(**kwargs)

    def _run_case(self, case, expected_shape):
        g_gpu = case["g"]
        self.assertEqual(tuple(g_gpu.shape), expected_shape)

        # GPU (B, T, H) → NPU (B, H, T)
        g_npu = g_gpu.transpose(1, 2).contiguous()

        y = self.call_op(
            g=g_npu.npu(),
            chunk_size=case["chunk_size"],
            cu_seqlens=case["cu_seqlens"],
            chunk_indices_out=case["chunk_indices_out"],
            reverse=case["reverse"],
            scale=case["scale"],
            head_first=True,          # NPU 仅支持 head_first=True
            output_dtype="float32",
        )

        # NPU 输出 (B, H, T) → 转回 GPU 布局 (B, T, H)
        y_gpu_layout = y.cpu().transpose(1, 2).contiguous()
        self.assertTensorClose(y_gpu_layout, case["y_expected"])
```

## 8. 调试技巧

1. **先用 CPU 验证 cumsum 轴**：分别尝试 dim=1/dim=2，确认 GPU dump 的 `head_first` 语义
2. **读 tiling 代码而非猜测**：`chunk_local_cumsum_tiling.cpp` 中每个 `OP_CHECK_IF` 对应一条约束
3. **检查符号是否编译进 .so**：`nm -D libcust_opapi.so | grep ChunkLocalCumsum`，空则需重建 wheel
4. **验证 chunk_indices_out 计算**：打印 `block_t` 和 `chunk_indices_out`，确认与 tiling 侧 `BLOCK_T` 公式一致
5. **cu_seqlens dtype**：GPU dump 是 int32，NPU 需要 int64，用 `[int(v) for v in tensor.tolist()]` 转换
6. **reverse/scale 的 None 处理**：GPU dump 用 None 表示默认值，必须显式解析为 `False`/`1.0`
7. **转置后验证**：转置 `g` 后在 CPU 上跑参考实现，确认转置不改变 cumsum 结果
8. **输出转置回 GPU 布局**：NPU 输出是 `(B, H, T)`，必须转回 `(B, T, H)` 再与 GPU golden 比对

## 9. GPU 与 NPU 布局策略对比

| 维度 | GPU dump | NPU 算子 |
|------|---------|---------|
| `g` 布局 | `(B, T, H)` head_first=False | `(B, H, T)` head_first=True（强制） |
| cumsum 轴 | dim=1（T 轴） | dim=2（T 轴） |
| `cu_seqlens` dtype | int32 | int64（wrapper 内部转换） |
| `chunk_indices_out` | 不需要 | varlen 模式必填 |
| `chunk_size` 约束 | 无特殊约束 | 必须 2 的幂 |
| 输出布局 | `(B, T, H)` | `(B, H, T)`，需转置回 `(B, T, H)` 比对 |
| 非连续性 | `g` 在 dump 中 contiguous | `AutoContiguous()` 自动处理 |

**结论**：`chunk_local_cumsum` 的核心比对难点在于 GPU 与 NPU 的 layout 约定不同（`head_first` 语义相反），以及 varlen 模式下 `chunk_indices_out` 的计算。测试中通过 `transpose(1, 2)` 完成 layout 转换，并在 CPU 上验证转换不改变 cumsum 语义。
