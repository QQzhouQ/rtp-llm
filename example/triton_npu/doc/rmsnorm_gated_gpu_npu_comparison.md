# rmsnorm_gated 算子 GPU-NPU 精度比对指南

本文档总结 Gated RMSNorm 算子 `RmsNormGated` NPU-vs-GPU 比对的实践经验，组织方式参考 `gpu_npu_comparison_guide.md` 的通用方法论。

## 整体流程

```
GPU 黄金数据 (.pt)  →  解析 inputs / model_state  →  CPU reference 验证语义  →  交换 torch.cuda.device→torch.npu.device 后调用 RmsNormGated.forward(x, gate) (NPU)  →  比对输出
```

## 1. 理解 GPU 数据格式

GPU 端通过 `torch.save()` 导出的 `.pt` 文件位于 `<workspace>/sample/RmsNormGated/`，共 3 个用例。

### 用例概览

| 文件 | M（行数） | N（隐藏维） | 场景 |
|------|-----------|-------------|------|
| `decode_seq1.pt` | 32 | 128 | decode |
| `prefill_seq32.pt` | 1024 | 128 | prefill seq=32 |
| `prefill_seq2047.pt` | 65504 | 128 | prefill seq=2047 |

### 典型数据内容（以 `prefill_seq32.pt` 为例）

| key | 类型 | 值 |
|-----|------|----|
| `op_name` | str | `"RmsNormGated"` |
| `mode` | str | `"prefill_seq32"` |
| `param_names` | list[2] | `['x', 'gate']`（仅前向参数） |
| `inputs/x` | tensor | shape=(1024, 128), dtype=float16, contiguous |
| `inputs/gate` | tensor | shape=(1024, 128), dtype=float16, contiguous |
| `outputs` | tensor | shape=(1024, 128), dtype=float16 — **golden 结果** |
| `model_state/weight` | tensor | shape=(128,), dtype=float16 — 模块权重（**不在 inputs 里**） |
| `model_state/eps` | float | `1e-6` |
| `model_state/group_size` | int | `128` |
| `inplace_outputs` | dict | 空（非 inplace，输出为新 tensor） |

> **关键**：`weight`/`eps`/`group_size` 是 `RmsNormGated` 模块的状态，**不在 `inputs` 中**，需要从 `model_state` 读取。`outputs` 是单 tensor（非 list）。

## 2. 理解 kernel 语义（CPU reference）

算子语义（`is_rms_norm=True`, `norm_before_gate=True`, `activation='silu'`）：

$$rstd = \frac{1}{\sqrt{\mathrm{mean}(x^2)+\epsilon}}, \qquad y = (x \cdot rstd \cdot weight) \cdot \mathrm{silu}(gate)$$

- **`norm_before_gate=True`**：门控 `silu(gate)` 在**归一化之后**乘（kernel 中 `y *= z*sigmoid(z)` 分支）；若为 False 则先乘到 `x` 上再归一化。
- kernel 内部全程 fp32（`x.to(tl.float32)`、`var=sum(x²)/N`、`rstd=1/sqrt(var+eps)`），最后 fp16 写回。

CPU reference（fp32 计算后转回）：

```python
def _cpu_reference(x, gate, weight, eps, group_size):
    M, N = x.shape
    ngroups = N // group_size
    x_f = x.float().view(M, ngroups, group_size)
    gate_f = gate.float().view(M, ngroups, group_size)
    w_f = weight.float().view(ngroups, group_size)
    rstd = 1.0 / torch.sqrt(x_f.pow(2).mean(dim=-1, keepdim=True) + eps)
    y = (x_f * rstd * w_f) * torch.nn.functional.silu(gate_f)
    return y.view(M, N).to(x.dtype)
```

> **返回结构**：`layer_norm_fwd` 实际返回 `(out, mean, rstd)` 三元组（RMS norm 下 `mean=None`），`RmsNormGated.forward` 只取 `[0]` 返回 `out`——所以比对对象是单个 `out` tensor，不要误以为返回 tuple。

## 3. 理解算子的输入格式（kernel 约束）

直接读 rtp-llm `layernorm_gated.py` 的 `_layer_norm_fwd_1pass_kernel`：

| 参数 | 期望 | 说明 |
|------|------|------|
| `x` | `(M, N)` fp16 | 输入，`stride(-1)==1` 断言，按 `stride_x_row` 跨行寻址 |
| `gate`（Z） | `(M, N)` fp16 | 门控分支，`stride(-1)==1` |
| `weight`（W） | `(N,)` fp16 | per-element 权重 |
| `bias`（B） | `(N,)` 可选 | 本算子无 bias（`HAS_BIAS=False`） |
| 输出 `y` | `(M, N)` fp16 | 新 tensor（非 inplace） |

### 关键点 1：`torch.cuda.device` 包装器（CUDA 硬编码）⚠️

`layer_norm_fwd` 在 kernel 启动外包了一层：

```python
with torch.cuda.device(x.device.index):      # ← CUDA 硬编码
    _layer_norm_fwd_1pass_kernel[grid](...)
```

**在 NPU 上直接调用会报错**（纯 NPU 构建的 torch 无 CUDA 支持）：

```
RuntimeError: PyTorch was compiled without CUDA support
```

**平移方案**：测试直接调用 `RmsNormGated.forward(x, gate)`（模型实际入口），并在调用期间把设备上下文 `torch.cuda.device` 交换为 `torch.npu.device`（monkeypatch、`finally` 恢复）——等价于改 rtp-llm 源码那一行。kernel 与 `layer_norm_fwd` 的启动逻辑（`BLOCK_N`/`num_warps`/`grid`）**全部零修改**，只换设备上下文。

### 关键点 2：`BLOCK_N` / `num_warps` / `grid`

```python
MAX_FUSED_SIZE = 65536 // x.element_size()     # fp16 → 32768
BLOCK_N  = min(MAX_FUSED_SIZE, triton.next_power_of_2(group_size))  # 128 → 128
num_warps = min(max(BLOCK_N // 256, 1), 8)      # 128 → 1
grid = (M, ngroups)                             # ngroups = N // group_size = 1
```

本 golden（N=group_size=128）→ `BLOCK_N=128`、`num_warps=1`、grid=`(M, 1)`。

## 4. 格式转换映射

GPU 与 NPU **布局一致、全部连续，无需 transpose**。只需 `.npu()` + 调用时交换设备上下文：

```python
@contextmanager
def _npu_device_ctx():
    """torch.cuda.device -> torch.npu.device（monkeypatch，调用后恢复）"""
    orig = torch.cuda.device
    torch.cuda.device = torch.npu.device
    try:
        yield
    finally:
        torch.cuda.device = orig

module = RmsNormGated(weight.npu(), None, group_size, eps, activation="silu")
with _npu_device_ctx():
    out = module(x.npu(), gate.npu())   # 直接调模型入口 forward(x, gate)
torch.npu.synchronize()                 # 等 kernel 完成后取回，避免异步竞态
```

需要做的：**(1) 全部 `.npu()`**；**(2) `weight/eps/group_size` 从 `model_state` 取**；**(3) 调用期间用 `_npu_device_ctx` 交换 `torch.cuda.device`→`torch.npu.device`**（唯一适配点，kernel 与启动逻辑零修改）。

> **注意**：`_npu_device_ctx` 是**全局** monkeypatch（替换的是 `torch.cuda.device` 属性）。仅适用于**串行**单测；若将来同进程内并行/多线程执行会互相干扰，需改为更细粒度的隔离方式。

## 5. 常见错误

### 5.1 不交换设备上下文直接调 `RmsNormGated.forward` 触发 CUDA 报错

```python
# ✗ 错误：未交换 torch.cuda.device，包装器内 with torch.cuda.device(...) 在 NPU 上抛
#     "PyTorch was compiled without CUDA support"
out = module(x.npu(), gate.npu())

# ✓ 正确：调用期间交换设备上下文（torch.cuda.device -> torch.npu.device）
with _npu_device_ctx():
    out = module(x.npu(), gate.npu())
```

### 5.2 从 `inputs` 找 weight

`weight/eps/group_size` 在 `model_state`，不在 `inputs`：

```python
# ✗ 错误：inputs 里没有 weight
w = inputs["weight"]

# ✓ 正确：从 model_state 读
w = data["model_state"]["weight"]
eps = float(data["model_state"]["eps"])
group_size = int(data["model_state"]["group_size"])
```

### 5.3 搞反 `norm_before_gate` 语义

`norm_before_gate=True` 时 gate 乘在**归一化之后**（`y *= silu(gate)`）；若按"先 gate 后 norm"写参考会错：

```python
# ✓ 正确：y = (x * rstd * w) * silu(gate)
y = (x * rstd * w) * silu(gate)
```

### 5.4 直接 `import rtp_llm`

`rtp_llm.__init__` 触发重 C++ 依赖。须用 stub 包层次 + `importlib` 只加载 `layernorm_gated.py`（仅依赖 torch/triton）。

### 5.5 在 fp16 下算 CPU reference

```python
# ✗ 错误：fp16 全程计算，与 kernel 的 fp32 中间精度不一致
rstd = 1 / torch.sqrt(x.pow(2).mean(-1, keepdim=True) + eps)

# ✓ 正确：fp32 计算后转回
rstd = 1 / torch.sqrt(x.float().pow(2).mean(-1, keepdim=True) + eps)
```

## 6. 测试模板

完整测试文件位于同目录 `test_npu_rmsnorm_gated_gpu_golden.py`。

```python
import importlib.util, os, sys, types, unittest
from contextlib import contextmanager
import torch, torch_npu

torch.npu.set_device(int(os.environ.get("TEST_DEVICE_ID", 0)))

def _load_rtp_llm_layernorm_gated():
    # stub 包层次 + importlib，加载 layernorm_gated.py
    ...
RmsNormGated = _load_rtp_llm_layernorm_gated().RmsNormGated

@contextmanager
def _npu_device_ctx():
    # torch.cuda.device -> torch.npu.device（调用后恢复）
    ...

class TestRmsNormGatedGpuGolden(unittest.TestCase):
    rtol = 1e-2
    atol = 1e-2

    def _run_case(self, filename):
        # 解析 inputs x/gate + model_state weight/eps/group_size
        # → CPU reference 自检 golden
        # → 实例化 RmsNormGated + with _npu_device_ctx(): out = module(x, gate)
        # → 比对 outputs
        ...
```

三个用例覆盖 decode 与 prefill 两种规模。

## 7. 调试技巧

1. **先打印 model_state**：`weight/eps/group_size` 在 `model_state`，确认已读出来
2. **先跑 CPU reference**：golden 自检不过就说明语义（尤其 `norm_before_gate`/gate 顺序）理解错，别急着上 NPU
3. **识别 CUDA 包装器报错**：`RuntimeError: PyTorch was compiled without CUDA support` = 撞上 `layer_norm_fwd` 的 `torch.cuda.device`，用 `_npu_device_ctx` 交换设备上下文
4. **读 kernel 源码**：确认 `IS_RMS_NORM`/`NORM_BEFORE_GATE`/`ACTIVATION` 三个 constexpr 的分支
5. **小 shape 复现**：出问题时构造 `(M=4, N=32, group_size=32)` 的合成 case
6. **注意 BLOCK_N/num_warps**：`group_size` 较小时 `BLOCK_N=next_pow2(group_size)`、`num_warps` 可能为 1，打印确认
7. **`torch.empty_like` 的 NPU 警告**：运行时 `out = torch.empty_like(x)` 会触发 torch_npu 的 `UserWarning`（"Cannot create tensor with internal format while allow_internal_format=False, tensor will be created with base format"）——这是**无害**的（Ascend 默认退回 base 内存格式），不要误判为错误。

## 8. GPU 与 NPU 布局策略对比

| 维度 | GPU dump | NPU（rtp-llm Triton for Ascend） |
|------|---------|------------------|
| `x` / `gate` / `weight` 布局 | `(M,N)`/`(M,N)`/`(N,)` fp16 连续 | 同，无需转换 |
| 中间精度 | fp32（sum/sqrt/silu） | fp32（同一 kernel） |
| 输出 | `outputs` 单 tensor（非 inplace） | 新 tensor `out` |
| 门控顺序 | `norm_before_gate=True`（gate 后乘） | 同 |
| 启动方式 | `layer_norm_fwd`（含 `torch.cuda.device` 包装） | **交换 `torch.cuda.device`→`torch.npu.device` 后调 `RmsNormGated.forward`**（kernel 零修改） |
| 算子来源 | — | rtp-llm `layernorm_gated.py`（stub 包 + importlib 加载，kernel 零修改） |

## 9. 实测结果

在本环境（昇腾 NPU）下运行测试，输出与 GPU golden 的逐用例比对数据：

| 用例 | M | N | max_abs_diff | mean_abs_diff |
|------|----|----|--------------|---------------|
| `decode_seq1` | 32 | 128 | 4.8e-7 | 1.2e-10 |
| `prefill_seq32` | 1024 | 128 | 3.8e-6 | 2.0e-10 |
| `prefill_seq2047` | 65504 | 128 | 1.2e-4 | 3.4e-10 |

### 误差来源分析

kernel 内 `sum(x²)/N` 与 `sqrt` 在 fp32 下做，误差来源只有两个 fp16 ULP 级环节：

1. **fp16 输入量化**：`x`/`gate`/`weight` 为 fp16，相对误差约 $2^{-11}\approx4.9\times10^{-4}$；
2. **写回舍入**：fp32 结果转回 fp16 时的 1 次 ULP 舍入。

中间归一化在 fp32 下完成、无 K 维长归约累积，NPU 与 GPU golden（同为 fp32 中间）的差异应逼近纯 fp16 舍入下界。实测三个用例最大误差 4.8e-7 ~ 1.2e-4，符合。

> **容差选择**：测试用 `rtol=atol=1e-2`，约为最坏误差的 80 倍，余量充足；因与 GPU 是同一份 Triton 源码（fp32 中间精度），误差应逼近纯 fp16 舍入下界。

**结论**：`RmsNormGated` 是 Gated RMSNorm 逐行算子（非 inplace），其 NPU 实现直接复用 rtp-llm 的 `_layer_norm_fwd_1pass_kernel`，kernel 零修改。比对的核心难点不在布局（GPU/NPU 一致、全连续）或数值（fp32 中间精度），而在**平移适配**：`layer_norm_fwd` 包装器硬编码 `with torch.cuda.device(...)`，在纯 NPU 构建上直接抛错，解法是把设备上下文交换为 `torch.npu.device`（monkeypatch）后直接调用 `RmsNormGated.forward(x, gate)`——kernel 与启动逻辑零修改。实测 3 个用例误差在纯 fp16 舍入量级内，验证通过。
