# GPU-NPU 算子精度比对指南

本文档总结 causal_conv1d 算子 NPU-vs-GPU 比对的实践经验，供后续其他算子比对参考。

## 整体流程

```
GPU 黄金数据 (.pt)  →  恢复非连续 stride  →  解析输入/输出  →  格式转换  →  调用 NPU 算子  →  精度比对
```

## 1. 理解 GPU 数据格式

GPU 端通过 `torch.save()` 导出的 `.pt` 文件内容因来源项目而异，加载后先打印所有 key 和 shape：

```python
data = torch.load(path, map_location="cpu", weights_only=False)
for k, v in data.items():
    if torch.is_tensor(v):
        print(f"{k}: shape={list(v.shape)}, dtype={v.dtype}")
    else:
        print(f"{k}: {v}")
```

### 常见结构

| 来源 | 典型结构 |
|------|---------|
| 本项目 `gpu_dump_*.py` | 顶层 dict，`x`/`weight`/`y_expected` 等直接作为 key |
| 外部项目 (如 rtp_llm) | `data["inputs"]` / `data["outputs"]` 分层结构，含 `input_meta` |

### `input_meta` 与非连续 stride 恢复

GPU 算子（如 rtp-llm Triton kernel）使用 channel-last 布局，`x`、`convStates` 等张量在 GPU 上是**非连续**的。但 `torch.save()` 会将张量以连续形式保存，丢失原始 stride 信息。

GPU dump 脚本会同时保存 `input_meta` 字典，记录每个输入张量的原始 shape / stride / dtype / contiguous 标志：

```python
# input_meta 典型内容
{
    "x":           {"shape": (8192, 2047),    "stride": (1, 12288),       "dtype": "torch.float16", "contiguous": False},
    "conv_state":  {"shape": (293, 8192, 3),  "stride": (1048576, 1, 8192), "dtype": "torch.float16", "contiguous": False},
    "weight":      {"shape": (8192, 4),       "stride": (4, 1),           "dtype": "torch.float16", "contiguous": True},
}
```

**必须在测试中恢复原始非连续 stride**，否则无法验证 NPU 算子对非连续输入的处理能力。

## 2. 恢复非连续 stride

使用 `torch.empty_strided` 从 `input_meta` 恢复原始布局：

```python
def _restore_strided_tensor(saved_data: torch.Tensor, meta: dict) -> torch.Tensor:
    """从 input_meta 恢复张量的原始（可能非连续）stride。"""
    if meta.get("contiguous", True):
        return saved_data
    dtype = getattr(torch, meta["dtype"].replace("torch.", ""))
    tensor = torch.empty_strided(meta["shape"], meta["stride"], dtype=dtype)
    tensor.copy_(saved_data)
    return tensor
```

### 恢复后的典型 stride

| 参数 | GPU 原始 shape | GPU 原始 stride | 恢复后 contiguous |
|------|---------------|----------------|------------------|
| `x` (prefill) | `(dim, seqlen)` | `(1, 12288)` | False |
| `conv_state` (decode) | `(num_pages, dim, state_len)` | `(1048576, 1, 8192)` | False |
| `weight` | `(dim, width)` | `(4, 1)` | True |

## 3. 推断 GPU 的语义参数

GPU 数据可能不直接包含激活函数、pad_slot_id 等参数。用 **CPU reference** 验证：

```python
# 用 F.conv1d 做 CPU 参考，分别尝试 activation=None/SiLU
out_no_act = F.conv1d(x_ref, w_ref.unsqueeze(1), padding=W-1, groups=D)[..., :S]
out_silu   = F.silu(out_no_act.float()).half()

print("match no_act:", torch.allclose(gpu_out, out_no_act))
print("match silu:",   torch.allclose(gpu_out, out_silu))
```

## 4. 理解 NPU 算子的输入格式

**不要猜测！** 直接读 tiling validation 代码（`*_tiling_validation.h`）。

以 causal_conv1d 为例，关键 shape 约束：

| 参数 | NPU 期望 shape | 说明 |
|------|---------------|------|
| `x` (varlen) | `(cu_seqlen, dim)` | 2D，token 数 × 特征维 |
| `x` (batch) | `(batch, seqlen, dim)` | 3D |
| `weight` | `(width, dim)` | **W 是第一维，D 是第二维** |
| `conv_states` | `(num_cache_lines, state_len, dim)` | `state_len >= width - 1` |
| `bias` | `(dim,)` | 可选 |
| `query_start_loc` | `(batch+1,)` | int64，varlen 模式必需 |

### 关键 C++ 校验逻辑（来自 `causal_conv1d_tiling_validation.h`）

```cpp
// weight 必须是 (width, dim)
OP_CHECK_IF(wShape.GetDimNum() != 2, ..., "weight must be 2D: (width, dim)");
const int64_t width = wShape.GetDim(0);   // 第一维 = 卷积核宽度
const int64_t wDim  = wShape.GetDim(1);   // 第二维 = 特征维度，必须等于 x 的 dim

// conv_states 中 stateLen >= width - 1
OP_CHECK_IF(stateLen < (width - 1), ...);

// dim 必须对齐到 DIM_ALIGN_ELEMS (=16)
OP_CHECK_IF(dim % 16 != 0, ...);

// query_start_loc 数据类型必须是 int64
OP_CHECK_IF(qslDesc->GetDataType() != ge::DT_INT64, ...);
```

### NPU 对非连续性的处理策略

NPU 算子定义（`*_def.cpp`）中每个输入有不同的连续性策略：

| 参数 | 连续性策略 | 说明 |
|------|-----------|------|
| `x` | `AutoContiguous()` | 框架自动转为连续，非连续输入不影响结果 |
| `convStates` | `IgnoreContiguous()` | **保留原始 stride**，kernel 通过 `convStateStride0/1` 寻址 |
| `y` (输出) | `AutoContiguous()` | 框架自动转为连续 |

`convStates` 的 stride 通过 tiling 传递给 kernel（来自 `causal_conv1d_tiling_validation.h`）：

```cpp
auto inputStride = context->GetInputStride(CONV_STATES_INDEX);
if (inputStride != nullptr && inputStride->GetDimNum() == 3) {
    tiling.convStateStride0 = inputStride->GetStride(0);  // num_cache_lines 轴
    tiling.convStateStride1 = inputStride->GetStride(1);  // state_len 轴
} else {
    tiling.convStateStride0 = stateLen * sDim;             // 默认连续
    tiling.convStateStride1 = sDim;
}
```

kernel 中按 stride 寻址（`DataCopy` 复制 `baseDim` 个连续元素，要求 **dim 轴 stride=1**）：

```cpp
const int64_t stateOffset =
    cacheIdx * convStateStride0 + pos * convStateStride1 + channelStart;
DataCopy(ring[...], convStatesGm[stateOffset], baseDim);
```

## 5. 格式转换映射

GPU 和 NPU 的 layout 约定可能完全不同，必须显式转换。**保留非连续 stride** 而非强制 `.contiguous()`：

```python
# GPU: x 是 (D, S) channel-last stride=(1, 12288)，NPU varlen: x 是 (S, D)
# .T 保留 channel-last stride: (1, 12288) → (12288, 1)，dim 轴 stride=1
x_npu = x_gpu.T               # 非连续，NPU AutoContiguous 自动处理

# GPU: weight 是 (D, W)，NPU: weight 是 (W, D)
w_npu = w_gpu.contiguous().T.contiguous()  # weight 无需保留非连续

# GPU: output 是 (D, S)，NPU: output 是 (S, D)
y_expected_npu = y_expected_gpu.T.contiguous()
```

### convStates 格式转换（decode/update 模式）

GPU 的 paged conv_state 布局为 `(num_pages, dim, state_len)`，channel-last（dim 轴 stride=1）。NPU 需要 `(num_cache_lines, state_len, dim)`。

**关键**：必须传入完整的 paged conv_state（而非提取单个 page），否则转置后 stride 恰好变为连续，无法验证非连续寻址路径。

```python
# 恢复 GPU 原始非连续 stride
conv_state_restored = _restore_strided_tensor(inputs["conv_state"], meta["conv_state"])
# GPU: (293, 8192, 3) stride=(1048576, 1, 8192)

# 转置完整 paged tensor 为 NPU 格式，保留非连续 stride
conv_states_npu = conv_state_restored.transpose(1, 2)
# (293, 3, 8192) stride=(1048576, 8192, 1) — 非连续，dim 轴 stride=1 ✓

# 通过 cache_indices 指定使用哪个 page
cache_indices = [page_idx]  # 从 block_map 获取
```

**对比**：如果只提取单个 page 再转置，stride 会变为连续：

```python
# ✗ 错误：提取单个 page 后转置，stride 变连续，无法验证非连续寻址
page = conv_state_restored[page_idx]     # (8192, 3) stride=(1, 8192)
conv_states_npu = page.T.unsqueeze(0)    # (1, 3, 8192) stride=(8192, 1) — 连续！
```

### 验证转换正确性

在 CPU 上用参考实现跑一遍，确认转换后的格式能匹配 GPU 输出：

```python
# 用 NPU 格式的 x 和 weight 跑 CPU reference
out_cpu = ref_fn(x_npu, w_npu, activation="silu")
assert torch.allclose(out_cpu, y_expected_npu, rtol=1e-2, atol=1e-2)
```

## 6. 常见错误

### 6.1 `aclnnStatus=561002`（参数校验失败）

通常是 shape/dtype 不匹配。逐项检查：

- [ ] weight 的 dim 顺序：NPU 期望 `(W, D)` 而非 `(D, W)`
- [ ] dim 是否对齐到 16：`dim % 16 == 0`
- [ ] conv_states 的 state_len 是否 ≥ width-1
- [ ] query_start_loc 是否为 int64，`[0] == 0`，`[-1] == cu_seqlen`
- [ ] 所有 tensor dtype 一致（都是 float16 或都是 bf16）

### 6.2 输出精度不达标

- GPU 与 CPU reference 之间的容许差约为 **0.0156**（float16 单 ULP）
- NPU 与 GPU 之间的容许差建议用 `rtol=5e-2, atol=5e-2`
- 对于大批量/大 dim 场景可能需要更宽松的阈值

### 6.3 conv_states 为 None

NPU 算子中 `conv_states` 是 **REQUIRED** 输入。若 GPU 数据中为 None，需创建零填充：

```python
conv_states_npu = torch.zeros(batch, width - 1, dim, dtype=x.dtype)
```

### 6.4 conv_states 原地更新后比较 CPU 原始 tensor

NPU 算子会原地修改 `convStates`。如果直接传 `conv_states.cpu().npu()`，算子修改的是 NPU 上的副本，CPU 原始 tensor 不会更新。**必须保存 NPU tensor 引用**：

```python
# ✗ 错误：比较的是未更新的 CPU 原始 tensor
y = self.call_op(conv_states=conv_states_npu.npu(), ...)
self.assertTensorClose(conv_states_npu, conv_states_expected)

# ✓ 正确：保存 NPU 引用，算完后 .cpu() 同步
conv_states_device = conv_states_npu.npu()
y = self.call_op(conv_states=conv_states_device, ...)
self.assertTensorClose(conv_states_device.cpu(), conv_states_expected)
```

对于 paged conv_state（decode/update 模式），算子只更新 `cache_indices` 指定的 page，需从完整 paged tensor 中提取对应 page 再比较：

```python
conv_states_device = conv_states_npu.npu()  # 完整 paged tensor
y = self.call_op(conv_states=conv_states_device, cache_indices=[page_idx], ...)

# 提取被更新的 page 进行比较
conv_states_actual = conv_states_device.cpu()[page_idx].unsqueeze(0)
self.assertTensorClose(conv_states_actual, conv_states_expected)
```

### 6.5 未恢复非连续 stride

`torch.save()` 会将非连续张量以连续形式保存。如果不从 `input_meta` 恢复原始 stride，则无法验证 NPU 算子对非连续输入的处理能力。

```python
# ✗ 错误：直接使用保存的连续 tensor
x_npu = inputs["x"].contiguous().T.contiguous()

# ✓ 正确：从 input_meta 恢复原始非连续 stride
x_restored = _restore_strided_tensor(inputs["x"], meta["x"])
x_npu = x_restored.T   # 保留非连续 stride
```

### 6.6 提取单个 page 后转置导致 stride 变连续

对于 paged conv_state，如果先提取单个 page 再转置，stride 恰好变为连续，无法验证 NPU 的非连续 stride 寻址路径。

```python
# ✗ 错误：提取单 page 后转置，stride 变连续
page = conv_state_restored[page_idx]     # (8192, 3) stride=(1, 8192)
conv_states_npu = page.T.unsqueeze(0)    # (1, 3, 8192) stride=(8192, 1) — 连续！

# ✓ 正确：转置完整 paged tensor，保留非连续 stride
conv_states_npu = conv_state_restored.transpose(1, 2)  # (293, 3, 8192) stride=(1048576, 8192, 1)
# 通过 cache_indices=[page_idx] 指定使用哪个 page
```

### 6.7 测试中缺少非连续性断言

恢复 stride 后应添加断言验证非连续性确实被保留，避免 `input_meta` 缺失或数据被意外 `.contiguous()` 时静默通过：

```python
# 验证 x 非连续且 stride 匹配 input_meta
self.assertFalse(x_gpu.is_contiguous(), "x should be non-contiguous")
self.assertEqual(x_gpu.stride(), (1, 12288), "x stride should match GPU input_meta")

# 验证 conv_states 非连续且 dim 轴 stride=1
self.assertFalse(conv_states_npu.is_contiguous(), "conv_states should be non-contiguous")
self.assertEqual(conv_states_npu.stride(-1), 1, "dim axis must have stride=1")
```

## 7. 测试模板

```python
import os
import unittest
import torch
from fla_npu.ops import ascendc as ascendc_ops

torch.npu.set_device(int(os.environ.get("TEST_DEVICE_ID", 0)))
_DATA_DIR = "/path/to/gpu/dump/dir"


def _restore_strided_tensor(saved_data: torch.Tensor, meta: dict) -> torch.Tensor:
    """从 input_meta 恢复张量的原始（可能非连续）stride。"""
    if meta.get("contiguous", True):
        return saved_data
    dtype = getattr(torch, meta["dtype"].replace("torch.", ""))
    tensor = torch.empty_strided(meta["shape"], meta["stride"], dtype=dtype)
    tensor.copy_(saved_data)
    return tensor


def _load_gpu_case(filename: str) -> dict:
    data = torch.load(os.path.join(_DATA_DIR, filename), map_location="cpu", weights_only=False)
    inputs = data["inputs"]
    meta = data.get("input_meta", {})
    # 恢复非连续 stride
    x = _restore_strided_tensor(inputs["x"], meta.get("x", {})) if meta.get("x") else inputs["x"].contiguous()
    return {"x": x, "weight": inputs["weight"].contiguous(), "y_expected": data["outputs"].contiguous(), ...}


class TestOpGpuGolden(unittest.TestCase):
    rtol = 5e-2
    atol = 5e-2

    def call_op(self, **kwargs):
        return ascendc_ops.npu_<op_name>(**kwargs)

    def assertTensorClose(self, actual, expected, *, rtol=None, atol=None):
        rtol = self.rtol if rtol is None else rtol
        atol = self.atol if atol is None else atol
        self.assertTrue(
            torch.allclose(actual.cpu().float(), expected.cpu().float(), rtol=rtol, atol=atol),
            msg=f"max_abs_diff={(actual.cpu().float() - expected.cpu().float()).abs().max().item():.6f}",
        )

    def test_case_xxx(self):
        case = _load_gpu_case("xxx.pt")
        # 格式转换（保留非连续 stride）...

        # 验证非连续性
        self.assertFalse(x_gpu.is_contiguous(), "x should be non-contiguous")
        self.assertFalse(conv_states_npu.is_contiguous(), "conv_states should be non-contiguous")
        self.assertEqual(conv_states_npu.stride(-1), 1, "dim axis must have stride=1")

        # conv_states 原地更新：保存 NPU 引用
        conv_states_device = conv_states_npu.npu()
        y = self.call_op(x=x_npu.npu(), conv_states=conv_states_device, ...)
        self.assertTensorClose(y, y_expected_npu)
        # paged 模式：从完整 tensor 中提取被更新的 page
        conv_states_actual = conv_states_device.cpu()[page_idx].unsqueeze(0)
        self.assertTensorClose(conv_states_actual, conv_states_expected)
```

## 8. 调试技巧

1. **先在 CPU 上验证转换逻辑**：用 numpy/F.conv1d 跑 reference，确认 GPU 输出可复现
2. **逐参数排查**：先只传 x + weight（最小参数集），通过后再加 conv_states/bias
3. **用小 shape 复现**：若大数据出错，手工构造 (B=1, S=4, D=64) 的小 case 缩小范围
4. **读 tiling validation**：aclnnStatus 报错时，在 `*_tiling_validation.h` 中搜索对应的 `OP_CHECK_IF` / `OP_LOGE` 定位参数约束
5. **检查 input_meta**：打印 `data["input_meta"]` 确认 GPU 原始 stride，确保测试中正确恢复非连续性
6. **验证 stride 恢复**：恢复后打印 `tensor.stride()` 和 `tensor.is_contiguous()`，确认与 `input_meta` 一致
7. **convStates 原地更新**：算子修改 NPU 上的 tensor，必须保存引用并在算完后 `.cpu()` 同步，不能比较 CPU 原始 tensor
8. **paged conv_state 保持非连续**：转置完整 paged tensor 而非提取单 page，否则 stride 变连续无法验证非连续寻址（见 6.6 节）
9. **添加非连续性断言**：在测试中用 `assertFalse(tensor.is_contiguous())` 和 `assertEqual(tensor.stride(-1), 1)` 验证非连续性被保留

## 9. GPU 与 NPU 非连续性策略对比

| 维度 | GPU (rtp-llm Triton) | NPU (Ascend C++) |
|------|---------------------|------------------|
| `x` 处理方式 | stride 传递，kernel 按 stride 寻址 | `AutoContiguous()` 自动转连续 |
| `x` 非连续支持 | channel-last（dim 轴 stride=1） | 自动处理，无需关心 |
| `convStates` 处理方式 | stride 传递，kernel 按 stride 寻址 | `IgnoreContiguous()` + `convStateStride0/1` 寻址 |
| `convStates` 约束 | dim 轴 stride=1 | dim 轴 stride=1（`DataCopy` 要求连续） |
| 输出 `y` 布局 | `empty_like(x)` 继承非连续 stride | `AutoContiguous()` 自动连续 |
| kernel 寻址 | 手动指针运算 `ptr + idx * stride` | `DataCopy` + stride offset |

**结论**：GPU 和 NPU 都要求 `convStates` 的 dim 轴 stride=1，其余轴通过 stride 寻址。测试中应从 `input_meta` 恢复 GPU 原始非连续 stride，确保 NPU 算子的 stride 寻址路径被正确验证。
