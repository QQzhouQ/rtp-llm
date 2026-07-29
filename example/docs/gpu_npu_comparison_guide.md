# GPU-NPU 算子精度比对指南

本文档总结 causal_conv1d 算子 NPU-vs-GPU 比对的实践经验，供后续其他算子比对参考。

## 整体流程

```
GPU 黄金数据 (.pt)  →  解析输入/输出  →  格式转换  →  调用 NPU 算子  →  精度比对
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
| 外部项目 (如 rtp_llm) | `data["inputs"]` / `data["outputs"]` 分层结构，可能含 metadata |

## 2. 推断 GPU 的语义参数

GPU 数据可能不直接包含激活函数、pad_slot_id 等参数。用 **CPU reference** 验证：

```python
# 用 F.conv1d 做 CPU 参考，分别尝试 activation=None/SiLU
out_no_act = F.conv1d(x_ref, w_ref.unsqueeze(1), padding=W-1, groups=D)[..., :S]
out_silu   = F.silu(out_no_act.float()).half()

print("match no_act:", torch.allclose(gpu_out, out_no_act))
print("match silu:",   torch.allclose(gpu_out, out_silu))
```

## 3. 理解 NPU 算子的输入格式

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

## 4. 格式转换映射

GPU 和 NPU 的 layout 约定可能完全不同，必须显式转换：

```python
# GPU: x 是 (D, S)，NPU varlen: x 是 (S, D)
x_npu = x_gpu.contiguous().T.contiguous()

# GPU: weight 是 (D, W)，NPU: weight 是 (W, D)
w_npu = w_gpu.contiguous().T.contiguous()

# GPU: output 是 (D, S)，NPU: output 是 (S, D)
y_expected_npu = y_expected_gpu.T.contiguous()
```

### 验证转换正确性

在 CPU 上用参考实现跑一遍，确认转换后的格式能匹配 GPU 输出：

```python
# 用 NPU 格式的 x 和 weight 跑 CPU reference
out_cpu = ref_fn(x_npu, w_npu, activation="silu")
assert torch.allclose(out_cpu, y_expected_npu, rtol=1e-2, atol=1e-2)
```

## 5. 常见错误

### 5.1 `aclnnStatus=561002`（参数校验失败）

通常是 shape/dtype 不匹配。逐项检查：

- [ ] weight 的 dim 顺序：NPU 期望 `(W, D)` 而非 `(D, W)`
- [ ] dim 是否对齐到 16：`dim % 16 == 0`
- [ ] conv_states 的 state_len 是否 ≥ width-1
- [ ] query_start_loc 是否为 int64，`[0] == 0`，`[-1] == cu_seqlen`
- [ ] 所有 tensor dtype 一致（都是 float16 或都是 bf16）

### 5.2 输出精度不达标

- GPU 与 CPU reference 之间的容许差约为 **0.0156**（float16 单 ULP）
- NPU 与 GPU 之间的容许差建议用 `rtol=5e-2, atol=5e-2`
- 对于大批量/大 dim 场景可能需要更宽松的阈值

### 5.3 conv_states 为 None

NPU 算子中 `conv_states` 是 **REQUIRED** 输入。若 GPU 数据中为 None，需创建零填充：

```python
conv_states_npu = torch.zeros(batch, width - 1, dim, dtype=x.dtype)
```

## 6. 测试模板

```python
import os
import unittest
import torch
from fla_npu.ops import ascendc as ascendc_ops

torch.npu.set_device(int(os.environ.get("TEST_DEVICE_ID", 0)))
_DATA_DIR = "/path/to/gpu/dump/dir"

def _load_gpu_case(filename: str) -> dict:
    data = torch.load(os.path.join(_DATA_DIR, filename), map_location="cpu", weights_only=False)
    # 根据实际 GPU 数据结构解析
    return { ... }

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
        # 格式转换 ...
        y = self.call_op(x=x_npu.npu(), ...)
        self.assertTensorClose(y, y_expected_npu)
```

## 7. 调试技巧

1. **先在 CPU 上验证转换逻辑**：用 numpy/F.conv1d 跑 reference，确认 GPU 输出可复现
2. **逐参数排查**：先只传 x + weight（最小参数集），通过后再加 conv_states/bias
3. **用小 shape 复现**：若大数据出错，手工构造 (B=1, S=4, D=64) 的小 case 缩小范围
4. **读 tiling validation**：aclnnStatus 报错时，在 `*_tiling_validation.h` 中搜索对应的 `OP_CHECK_IF` / `OP_LOGE` 定位参数约束
