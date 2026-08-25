# sigmoid_gate_scale_add_triton 算子 GPU-NPU 精度比对指南

本文档总结 MoE shared-expert 门控融合算子 `sigmoid_gate_scale_add_triton` NPU-vs-GPU 比对的实践经验，组织方式参考 `gpu_npu_comparison_guide.md` 的通用方法论。

## 整体流程

```
GPU 黄金数据 (.pt)  →  打印 key/shape 诊断  →  解析 inputs / outputs  →  CPU reference 验证语义  →  调用 rtp-llm sigmoid_gate_scale_add_triton Triton kernel (NPU)  →  比对 inplace 修改后的 experts
```

## 1. 理解 GPU 数据格式

GPU 端通过 `torch.save()` 导出的 `.pt` 文件位于 `<workspace>/sample_moe/sigmoid_gate_scale_add_triton/`，共 3 个用例。

### 用例概览

| 文件 | T（token） | H（隐藏维） | 场景 |
|------|-----------|------------|------|
| `T1_H2048.pt` | 1 | 2048 | decode（单 token） |
| `T16_H2048.pt` | 16 | 2048 | 小 batch |
| `T2047_H2048.pt` | 2047 | 2048 | prefill（大 batch） |

### 典型数据内容（以 `T16_H2048.pt` 为例）

| key | 类型 | 值 |
|-----|------|----|
| `op_name` | str | `"sigmoid_gate_scale_add_triton"` |
| `mode` | str | `"T16_H2048"` |
| `param_names` | list[3] | 记录 3 个调用参数名（gate/shared/experts） |
| `inputs/gate` | tensor | shape=(T, 1), dtype=float16, contiguous |
| `inputs/shared` | tensor | shape=(T, H), dtype=float16, contiguous — shared expert MLP 输出 |
| `inputs/experts` | tensor | shape=(T, H), dtype=float16, contiguous — routed experts 输出（**初始值**） |
| `outputs` | tensor | shape=(T, H), dtype=float16 — **golden 结果**（= 最终 experts） |
| `inplace_outputs/experts` | tensor | shape=(T, H), dtype=float16 — 与 `outputs` 值相同、**不同对象** |
| `input_meta` | dict | gate/shared/experts 的 shape/stride/dtype/contiguous（全部连续） |
| `model_state` | None | 无模型状态 |

> **关键**：
> - `inputs/experts` 是 kernel 执行**前**的初始值，`outputs` 是执行**后**的结果；
> - `inplace_outputs/experts` 与 `outputs` 数值相同但是不同 buffer（GPU dump 时各存了一份），二者都可作为 golden 期望值；
> - 本算子**没有**独立的 `outputs` list——结果直接放在顶层 `outputs` 和 `inplace_outputs.experts`。

## 2. 理解 kernel 语义（CPU reference）

算子语义（**inplace**，在 `experts` 上原地计算）：

$$experts[t, :] = \text{sigmoid}(gate[t, 0]) \cdot shared[t, :] + experts[t, :]$$

kernel 内部计算精度：`gate` 先 `.to(tl.float32)`，`sigmoid` 在 fp32 下计算，乘加也在 fp32 下完成，最后 `.to(shared.dtype)` 写回 `experts`。

CPU reference（与 kernel 语义逐条对应，fp32 计算后转回 shared dtype）：

```python
def _cpu_reference(gate, shared, experts):
    result = torch.sigmoid(gate.float()) * shared.float() + experts.float()
    return result.to(shared.dtype)
```

### 与 rtp-llm 平台实现的差异

| 路径 | 实现 | 中间精度 |
|------|------|---------|
| CUDA（rtp-llm `cuda/moe_gating.py`） | 调 `sigmoid_gate_scale_add_triton`（本 kernel） | **fp32** |
| Ascend（rtp-llm `ascend/moe_gating.py`） | **PyTorch 兜底**：`experts.add_(torch.sigmoid(gate) * shared)` | **fp16** 全程 |
| 本测试（NPU） | 复用 CUDA 版 Triton kernel（经 Triton for Ascend 编译） | **fp32** |

> 即：rtp-llm 在 Ascend 平台默认并不走这个 Triton kernel，而是用纯 PyTorch 的 fp16 兜底。本测试的定位是**把 CUDA 用的 Triton kernel 搬到昇腾上运行**，验证其与 GPU golden（同样来自 CUDA Triton，fp32 中间精度）一致，而不是验证 rtp-llm Ascend 的 PyTorch 兜底路径。

## 3. 理解算子的输入格式（kernel 约束）

直接读 rtp-llm `moe_gating.py` 的签名与寻址逻辑：

| 参数 | 期望 | 说明 |
|------|------|------|
| `gate` | `(T, 1)` fp16/bf16/fp32 | 标量门控，按 `tid * stride_gate_t` 寻址 |
| `shared` | `(T, H)` | shared expert MLP 输出，按 `stride_shared_t` 寻址（支持非连续 token 维） |
| `experts` | `(T, H)` | routed experts 输出，**inplace 修改**，按 `stride_out_t` 寻址 |
| 输出 dtype | = `shared.dtype` | kernel 内 fp32 计算后转回 |

### 关键点：`BLOCK_H` 自动选择

kernel 不要求调用方指定分块，内部 `_select_block_h(T, H)` 自动选择：

```
目标总 program 数 ≈ 512（_MIN_TOTAL_PROGRAMS），上下界 128 ≤ BLOCK_H ≤ 4096
target_h_blocks = max(1, 512 // T)
BLOCK_H = clamp(next_pow2(max(1, H // target_h_blocks)), 128, 4096)
grid    = (T, ceil(H / BLOCK_H))
```

- 小 T（decode）用小块 BLOCK_H，把 H 维切多块保持 SM 占用；
- 大 T（prefill）自然用大块，减少 launch 开销。

以 H=2048 为例：`T=1` / `T=16` → `BLOCK_H=128`，grid 分别为 `(1,16)` / `(16,16)`；`T=2047` → `BLOCK_H=2048`，grid=`(2047,1)`。

## 4. 格式转换映射

GPU 与 NPU（rtp-llm Triton for Ascend）**布局完全一致，无需任何 transpose 或 dtype 转换**（golden 已是 fp16 连续）：

```python
gate_npu   = gate.npu()
shared_npu = shared.npu()
experts_actual = experts.clone().npu()   # 必须 clone：kernel 会 inplace 修改
ret = sigmoid_gate_scale_add_triton(gate_npu, shared_npu, experts_actual)
torch.npu.synchronize()
```

需要做的只有：**(1) 全部 `.npu()`**；**(2) `experts` 传入前 `clone()`**（因为 kernel 原地修改，避免污染 golden 输入缓冲）。调用后 `ret is experts_actual`，结果在 `experts_actual` 中。

## 5. 常见错误

### 5.1 忘记 `experts` 会被 inplace 修改

```python
# ✗ 错误：直接传 golden 的 experts，kernel 原地改写后无法再作基准
experts_actual = experts.npu()
sigmoid_gate_scale_add_triton(gate, shared, experts_actual)

# ✓ 正确：先 clone，再传入
experts_actual = experts.clone().npu()
sigmoid_gate_scale_add_triton(gate, shared, experts_actual)
```

### 5.2 混淆 `outputs` 与 `inplace_outputs.experts`

两者值相同但**不是同一对象**。比对用哪个都行，但不要假设 `data_ptr` 相同，也不要拿 `inputs/experts`（初始值）当期望。

### 5.3 直接 `import rtp_llm`

`rtp_llm.__init__` 会触发重 C++ 依赖（`libth_transformer_config.so`）。必须用 stub 包层次 + `importlib` 只加载 `moe_gating.py` 单文件（该文件仅依赖 `torch`/`triton`）。

### 5.4 在 fp16 下算 CPU reference

```python
# ✗ 错误：fp16 全程计算，与 kernel 的 fp32 中间精度不一致
ref = torch.sigmoid(gate) * shared + experts

# ✓ 正确：fp32 计算后转回
ref = (torch.sigmoid(gate.float()) * shared.float() + experts.float()).to(shared.dtype)
```

### 5.5 误以为 kernel 是 Ascend 平台的默认路径

rtp-llm 在 Ascend 用的是 `ascend/moe_gating.py` 的 PyTorch 兜底（fp16），Triton kernel 属 CUDA 路径。本测试是**主动把 Triton kernel 搬到 NPU** 验证其与 GPU golden 一致，不要与 Ascend 默认路径混淆。

## 6. 测试模板

完整测试文件位于 `/home/s60130915/work/test_npu_sigmoid_gate_scale_add_triton_gpu_golden.py`。

```python
import importlib.util, os, sys, types, unittest
import torch

torch.npu.set_device(int(os.environ.get("TEST_DEVICE_ID", 0)))

def _load_rtp_llm_moe_gating_modules():
    # stub 包层次 + importlib，绕过 rtp_llm.__init__ 的 C++ 依赖
    # （moe_gating.py 仅依赖 torch/triton，加载单文件即可）
    ...
sigmoid_gate_scale_add_triton = _load_rtp_llm_moe_gating_modules().sigmoid_gate_scale_add_triton

class TestSigmoidGateScaleAddGpuGolden(unittest.TestCase):
    rtol = 1e-2
    atol = 1e-2

    def _run_case(self, filename):
        # 解析 inputs / outputs → CPU reference 自检 golden（防损坏）
        # → gate/shared .npu() + experts clone().npu()
        # → 调 kernel → torch.npu.synchronize()
        # → 断言 ret is experts_actual（inplace）→ 比对 experts_actual
        ...
```

三个用例覆盖 decode（T=1）与 prefill（T=2047）两种规模。

## 7. 调试技巧

1. **先打印所有 key/shape**：确认结果在顶层 `outputs` 和 `inplace_outputs.experts`，`inputs/experts` 是初始值
2. **先跑 CPU reference**：golden 自检不过就说明要么 dump 损坏、要么语义理解错，别急着上 NPU
3. **验证 inplace**：`ret is experts_actual`，确认 kernel 原地修改并返回同一对象
4. **读 kernel 源码**：确认 fp32 中间精度与 `stride_*` 寻址（token 维按 stride，支持非连续）
5. **小 shape 复现**：出问题时构造 `(T=4, H=128)` 的合成 case 缩小范围
6. **注意 BLOCK_H 分支**：T 很小时走小 BLOCK_H 分块，T 很大时走单块，不同分支可用打印 `_select_block_h` 结果区分

## 8. GPU 与 NPU 布局策略对比

| 维度 | GPU dump | NPU（rtp-llm Triton for Ascend） |
|------|---------|------------------|
| `gate` / `shared` / `experts` 布局 | `(T,1)` / `(T,H)` / `(T,H)` fp16 连续 | 同，无需转换 |
| 中间精度 | fp32（CUDA Triton golden） | fp32（同一 kernel） |
| 输出 | `outputs` 与 `inplace_outputs.experts`（值同对象异） | inplace 写回 `experts_actual` |
| 原地更新 | 是 | 是（须 `clone()` 传入） |
| layout 转置 | 无 | **无需转置** |
| 算子来源 | — | rtp-llm `moe_gating.py`（stub 包 + importlib 加载，源码零修改） |

## 9. 实测结果

在本环境（昇腾 NPU）下运行测试，输出与 GPU golden 的逐用例比对数据：

| 用例 | T | H | `experts` max_abs_diff | `experts` mean_abs_diff |
|------|---|----|-----------------------|------------------------|
| `T1_H2048` | 1 | 2048 | 0.0 | 0.0 |
| `T16_H2048` | 16 | 2048 | 1.9e-6 | 6.9e-11 |
| `T2047_H2048` | 2047 | 2048 | 1.5e-5 | 1.8e-10 |

### 误差来源分析

kernel 计算极简（逐元素 sigmoid·乘加），无 K 维归约，误差来源只有两个 fp16 ULP 级环节：

1. **fp16 输入量化**：`gate`/`shared`/`experts` 为 fp16，相对误差约 $2^{-11}\approx4.9\times10^{-4}$；
2. **写回舍入**：fp32 结果转回 fp16 时的 1 次 ULP 舍入。

中间乘加全程 fp32，不引入累积误差，因此 NPU 与 GPU golden（同为 fp32 中间精度）的差异应逼近**纯 fp16 舍入下界**。实测 `T1` 逐元素完全一致（diff=0），`T16`/`T2047` 最大误差 ≤1.5e-5，处于 fp16 输入量化 + 写回舍入的正常量级（对典型 O(0.01) 量级的输出约 1~2 个 fp16 ULP），完全吻合。

> **容差选择**：测试用 `rtol=atol=1e-2`，约为最坏误差的 600+ 倍，余量充足；由于与 GPU 是同一份 Triton 源码（fp32 中间精度），比 rtp-llm 自测惯用的 `atol=5e-3` 更宽松也仍通过。

**结论**：`sigmoid_gate_scale_add_triton` 是逐元素融合门控算子（inplace 写回），其 NPU 实现直接复用 rtp-llm 的 Triton kernel，源码零修改，布局与语义和 GPU 完全一致。比对的核心难点不在 layout 转换（GPU/NPU 布局一致），而在三点——(1) 记得 `experts` 会被 inplace 修改、传入前须 `clone()`；(2) 区分 `inputs/experts`（初始值）与 `outputs`/`inplace_outputs.experts`（期望值）；(3) 保持 fp32 中间精度的 CPU reference，且理解该 kernel 是 CUDA 路径、并非 rtp-llm Ascend 平台默认实现。实测 3 个用例误差在纯 fp16 舍入量级内，验证通过。
