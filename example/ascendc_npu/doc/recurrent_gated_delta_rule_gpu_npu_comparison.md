# recurrent_gated_delta_rule 算子 GPU-NPU 精度比对指南

本文档总结 `recurrent_gated_delta_rule` 算子 NPU-vs-GPU 比对的实践经验，供后续该算子维护和类似 recurrent 类算子比对参考。通用方法论参见同目录下的 `gpu_npu_comparison_guide.md`。
- 该算子需要cann-9.1.0支持

## 整体流程

```
GPU 黄金数据 (.pt)  →  恢复非连续 stride  →  解析输入/输出/属性  →  q/k L2 归一化  →  state 转置 (dk,dv)→(dv,dk)  →  dtype 转换 float32→bfloat16  →  调用 NPU 算子  →  state 转置回 (dk,dv)  →  精度比对
```

## 1. 理解 GPU 数据格式

### 数据来源

GPU 端通过 `torch.save()` 导出的 `.pt` 文件位于 `<workspace>/sample/fused_recurrent_gated_delta_rule/`，采用 `inputs`/`outputs`/`input_meta`/`inplace_outputs` 分层结构：

```python
data = torch.load(path, map_location="cpu", weights_only=False)
# 顶层 key: inputs, outputs, input_meta, inplace_outputs
```

### 典型数据内容

以 `decode.pt` 为例：

| key | 类型 | 值 |
|-----|------|----|
| `inputs/q` | tensor | shape=(1, 1, 16, 128), dtype=float16 |
| `inputs/k` | tensor | shape=(1, 1, 16, 128), dtype=float16 |
| `inputs/v` | tensor | shape=(1, 1, 32, 128), dtype=float16 |
| `inputs/beta` | tensor | shape=(1, 1, 32), dtype=float16 |
| `inputs/g` | tensor | shape=(1, 1, 32), dtype=float32 |
| `inputs/initial_state` | tensor | shape=(293, 32, 128, 128), dtype=bfloat16, **非连续** |
| `inputs/scale` | None | None 表示使用默认值 `dk ** -0.5` |
| `inputs/use_qk_l2norm_in_kernel` | bool | True — GPU kernel 内部做 L2 归一化 |
| `inputs/block_map` | tensor | shape=(1, 1), dtype=int32, 值=[[2]] |
| `inputs/sequence_lengths` | tensor | shape=(1,), dtype=int32 |
| `outputs[0]` | tensor | shape=(1, 1, 32, 128), dtype=float16 — 注意力输出 |
| `outputs[1]` | tensor | shape=(293, 32, 128, 128), dtype=bfloat16 — final_state |
| `inplace_outputs/initial_state` | tensor | 同 outputs[1]，原地更新后的 state |
| `input_meta/initial_state` | dict | shape=(293,32,128,128), stride=(524288,16384,128,1), contiguous=False |


### `input_meta` 与 stride 恢复

GPU dump 中 `initial_state` 的 `input_meta` 记录了非连续 stride：

```python
# input_meta/initial_state 典型内容
{
    "shape": (293, 32, 128, 128),
    "stride": (524288, 16384, 128, 1),   # channel-last: (nv*dv*dk, dv*dk, dk, 1)
    "dtype": "torch.bfloat16",
    "contiguous": False
}
```

`torch.save()` 会将非连续张量以连续形式保存，必须从 `input_meta` 恢复原始 stride。

## 2. 恢复非连续 stride

复用通用指南中的 `_restore_strided_tensor` 函数：

```python
def _restore_strided_tensor(saved_data: torch.Tensor, meta: dict) -> torch.Tensor:
    if not meta or meta.get("contiguous", True):
        return saved_data
    dtype = getattr(torch, meta["dtype"].replace("torch.", ""))
    tensor = torch.empty_strided(meta["shape"], meta["stride"], dtype=dtype)
    tensor.copy_(saved_data)
    return tensor
```

恢复后的 `initial_state` 为非连续的 (293, 32, 128, 128) paged tensor，stride=(524288, 16384, 128, 1)。

## 3. 推断 GPU 的语义参数

### 3.1 `use_qk_l2norm_in_kernel` — q/k L2 归一化

GPU dump 中 `use_qk_l2norm_in_kernel=True`，表示 GPU kernel **内部**对 q/k 做 L2 归一化。因此保存的 q/k 是**原始未归一化**的值。NPU kernel 不做内部归一化，必须在输入前显式归一化：

```python
# 验证：检查 q/k 的 L2 范数
q = inputs["q"].squeeze(0).squeeze(0).to(torch.float32)  # (16, 128)
print(q.norm(p=2, dim=-1))  # 非 1 → 未归一化，需手动归一化

# 归一化后再转 bfloat16
q_npu = torch.nn.functional.normalize(q, p=2, dim=-1).to(torch.bfloat16)
```

**不归一化直接传 q/k 会导致 output max_diff ≈ 1.26**（远超容差）。

### 3.2 `scale` 为 None 时的默认值

GPU dump 中 `scale=None` 表示使用默认值 `dk ** -0.5`：

```python
dk = q_npu.shape[-1]
scale = inputs.get("scale")
if scale is None:
    scale = float(dk ** -0.5)
```

### 3.3 `block_map` 与 paged state

`block_map` 是 (1, 1) int32 tensor，值如 `[[2]]`，表示当前 batch 使用第 2 个 state page。测试中提取该 page 传给 NPU，避免传递完整的 293 pages（~461MB float32）：

```python
block_map = inputs["block_map"]
page_idx = int(block_map[0, 0].item())
```

## 4. 理解 NPU 算子的输入格式

### 关键 shape/dtype 约束

| 参数 | NPU 期望 | 说明 |
|------|---------|------|
| `query` | `(t, nk, dk)` bfloat16 | 必须是 bfloat16 |
| `key` | `(t, nk, dk)` bfloat16 | 必须是 bfloat16 |
| `value` | `(t, nv, dv)` bfloat16 | 必须是 bfloat16 |
| `beta` | `(t, nv)` bfloat16 | 必须是 bfloat16 |
| `state` | `(num_states, nv, dv, dk)` **bfloat16** | stateRef，**原地修改** |
| `actual_seq_lengths` | `(b,)` int32 | `[star_idx, batch0_len, ...]` |
| `ssm_state_indices` | `(t,)` int32 | 每个 token 到 state slot 的映射 |
| `g` | `(t, nv)` float32 | 衰减系数 α = exp(g) |
| `gk` | `(t, nv, dk)` float32 | 当前版本暂不支持，传 None |
| `num_accepted_tokens` | `(b,)` int32 | 可选，投机推理接受 token 数 |
| `scale` | float | query 缩放因子 |
| `out` | `(t, nv, dv)` bfloat16 | 注意力输出 |

### 关键 aclnn 接口签名（来自 `aclnn_recurrent_gated_delta_rule.h`）

```cpp
// 所有 tensor 的 dtype 约束：
// query/key/value/beta/state/out: bfloat16
// actualSeqLengths/ssmStateIndices/numAcceptedTokens: int32
// g/gk: float32
// scaleValue: float32

aclnnStatus aclnnRecurrentGatedDeltaRuleGetWorkspaceSize(
    const aclTensor *query, const aclTensor *key, const aclTensor *value,
    const aclTensor *beta, aclTensor *stateRef,
    const aclTensor *actualSeqLengths, const aclTensor *ssmStateIndices,
    const aclTensor *g, const aclTensor *gk,
    const aclTensor *numAcceptedTokens, float scaleValue,
    aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor);
```

### NPU 解耦路径 vs torch_npu 旧路径

| 维度 | `fla_npu.ops.ascendc`（解耦路径） | `torch_npu.npu_recurrent_gated_delta_rule`（旧路径） |
|------|----------------------------------|---------------------------------------------------|
| 调用方式 | ctypes 直调 aclnn | torch.ops.npu dispatcher |
| state 原地更新 | 修改传入的 state tensor | 修改传入的 state tensor |
| 返回值 | `(out, state)` 元组 | 单个 `out` tensor（state 通过原地修改获取） |
| star_idx 位置 state | **不修改** | **会修改**（已知行为差异） |
| 推荐使用 | 是 | 仅兼容性测试 |

**重要**：`torch_npu` 旧路径会修改 `actual_seq_lengths[0]`（star_idx）位置的 state，而 golden 参考实现不处理该位置。使用 `fla_npu.ops.ascendc` 解耦路径不存在此问题。

## 5. 格式转换映射

### 5.1 q/k L2 归一化（核心）

GPU kernel 内部做 L2 归一化（`use_qk_l2norm_in_kernel=True`），保存的 q/k 是原始值。NPU kernel 不做内部归一化，必须在外部归一化：

```python
# GPU: q 是原始未归一化的 (batch, seqlen, nk, dk) float16
# NPU: q 需要归一化后的 (t, nk, dk) bfloat16
q_raw = inputs["q"].squeeze(0).squeeze(0).to(torch.float32)
q_npu = torch.nn.functional.normalize(q_raw, p=2, dim=-1).to(torch.bfloat16)
```

### 5.2 state 布局转置（核心）

GPU state 布局是 `(nv, dk, dv)`，NPU 期望 `(nv, dv, dk)`。必须转置最后两个维度：

```python
# GPU: state_page 是 (nv, dk, dv)
# NPU: state 需要是 (nv, dv, dk)
state_page = state_restored[page_idx].transpose(-1, -2).contiguous().to(torch.bfloat16)
state_npu = state_page.unsqueeze(0)  # (1, nv, dv, dk)
```

NPU 输出的 state 也是 `(nv, dv, dk)`，比较时需转置回 `(nv, dk, dv)`：

```python
# NPU 输出 state 是 (nv, dv, dk)，转回 GPU 布局 (nv, dk, dv)
state_actual = state_device.cpu()[0].transpose(-1, -2)
```

### 5.3 dtype 转换

GPU dump 中 q/k/v/beta 是 float16，state 是 bfloat16。NPU 要求全部为 bfloat16：

```python
q_npu = ...to(torch.bfloat16)
k_npu = ...to(torch.bfloat16)
v_npu = inputs["v"].squeeze(0).squeeze(0).to(torch.bfloat16)
beta_npu = inputs["beta"].squeeze(0).squeeze(0).to(torch.bfloat16)
```

**state 必须是 bfloat16**，传 float32 会报 `aclnnStatus=161002`（参数错误）。

### 5.4 维度压缩

GPU dump 的 q/k/v/beta/g 是 4D/3D `(batch, seqlen, n, d)`，NPU 期望 3D/2D `(t, n, d)`：

```python
q_npu = inputs["q"].squeeze(0).squeeze(0)  # (batch=1, seqlen=1, nk, dk) → (nk, dk)
if q_npu.dim() == 2:
    q_npu = q_npu.unsqueeze(0)  # (1, nk, dk)
```

### 5.5 actual_seq_lengths 与 ssm_state_indices 构造

GPU dump 不直接提供这两个参数，需根据 `sequence_lengths` 和 `block_map` 构造：

```python
# actual_seq_lengths: [star_idx, batch0_len, ...]
# star_idx=0 表示新 token 从 index 0 开始（无前缀）
actual_seq_lengths = torch.tensor([0, 1], dtype=torch.int32)

# ssm_state_indices: 每个 token 映射到 state slot 0（只提取了单个 page）
t = int(actual_seq_lengths.sum().item())
ssm_state_indices = torch.tensor([0] * t, dtype=torch.int32)
```

### 验证转换正确性

在 CPU 上用 golden 参考实现验证转换后的输入能匹配 GPU 输出：

```python
# 用转置后的 state (nv, dv, dk) 跑 CPU golden
# golden 公式: S = S * alpha; delta = (v - S@k) * beta; S += outer(delta, k); o = S@q * scale
# 验证 output max_diff < 0.001, state max_diff < 0.04
```

关键验证结果（转置 vs 不转置）：

| 方案 | output max_diff | state max_diff |
|------|----------------|----------------|
| 不转置 state (nv, dv, dk) | 1.256836 | 9.896173 |
| **转置 state (nv, dk, dv)→(nv, dv, dk)** | **0.000309** | **0.031032** |

## 6. 常见错误

### 6.1 `aclnnStatus=161002`（参数校验失败）

```
AclNN_Parameter_Error(EZ1001): Tensor params.state not implemented for DT_FLOAT,
should be in dtype support list [DT_BFLOAT16,].
```

state 的 dtype 是 float32，但 NPU 算子仅支持 bfloat16。常见原因：

- [ ] **state dtype 为 float32**：必须 `.to(torch.bfloat16)`，NPU 不支持 float32 state
- [ ] **q/k/v/beta dtype 不一致**：NPU 要求全部 bfloat16
- [ ] **actual_seq_lengths/ssm_state_indices dtype 不是 int32**：必须为 int32

### 6.2 output max_diff ≈ 1.26（q/k 未归一化）

GPU dump 中 `use_qk_l2norm_in_kernel=True`，保存的 q/k 是原始值。如果不做 L2 归一化直接传给 NPU，output 会出现约 1.26 的最大误差。

```python
# ✗ 错误：直接使用 GPU dump 的 q/k
q_npu = inputs["q"].squeeze(0).squeeze(0).to(torch.bfloat16)

# ✓ 正确：先归一化再转 dtype
q_raw = inputs["q"].squeeze(0).squeeze(0).to(torch.float32)
q_npu = torch.nn.functional.normalize(q_raw, p=2, dim=-1).to(torch.bfloat16)
```

### 6.3 output/state max_diff ≈ 1.5~9.9（state 布局未转置）

GPU state 布局是 `(nv, dk, dv)`，NPU 期望 `(nv, dv, dk)`。不转置会导致矩阵乘法维度错配，产生极大误差。

```python
# ✗ 错误：直接使用 GPU 布局的 state
state_page = state_restored[page_idx].to(torch.bfloat16)  # (nv, dk, dv) — 错！

# ✓ 正确：转置最后两维
state_page = state_restored[page_idx].transpose(-1, -2).contiguous().to(torch.bfloat16)  # (nv, dv, dk)
```

比较时也需将 NPU 输出转回：

```python
# ✗ 错误：直接比较 NPU 输出的 state
state_actual = state_device.cpu()[0]  # (nv, dv, dk) — 与 GPU (nv, dk, dv) 不匹配

# ✓ 正确：转置回 GPU 布局
state_actual = state_device.cpu()[0].transpose(-1, -2)  # (nv, dk, dv)
```

### 6.4 torch_npu 旧路径修改 star_idx 位置的 state

使用 `torch_npu.npu_recurrent_gated_delta_rule`（torch_npu 内置实现）时，`actual_seq_lengths[0]`（star_idx）位置的 state 会被修改，而 golden 参考实现不处理该位置。这会导致 `final_state` 在 position 0 出现精度失败。

```
# torch_npu 旧路径:
state[0] 调用前=0.8828125 → 调用后=0.31640625 (被修改) → FAIL

# fla_npu.ops.ascendc 解耦路径:
state[0] 调用前=0.8828125 → 调用后=0.8828125 (不变) → PASS
```

**建议**：精度验证使用 `fla_npu.ops.ascendc` 解耦路径（`test_run_standalone.py`），而非 `torch_npu` 旧路径（`test_accuracy.py`）。

### 6.5 GPU .pt 文件路径找不到

测试默认从 `<workspace>/sample/fused_recurrent_gated_delta_rule/` 加载数据。若路径不同，检查实际挂载位置：

```sh
find / -name "decode.pt" -path "*/fused_recurrent_gated_delta_rule/*" 2>/dev/null
```

### 6.6 scale 为 None 时未解析默认值

GPU dump 中 `scale` 存为 `None` 表示使用默认值 `dk ** -0.5`。直接传 `None` 会触发类型错误：

```python
# ✗ 错误：直接传 None
scale = inputs.get("scale")  # None

# ✓ 正确：解析为默认值
dk = q_npu.shape[-1]
scale = inputs.get("scale")
if scale is None:
    scale = float(dk ** -0.5)
```

## 7. 测试模板

完整测试文件位于 `torch_custom/fla_npu/test/test_npu_recurrent_gated_delta_rule_gpu_golden.py`，核心结构：

```python
import os
import unittest
import torch
from fla_npu.ops import ascendc as ascendc_ops

torch.npu.set_device(int(os.environ.get("TEST_DEVICE_ID", 0)))
_DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__),
                                          "..", "..", "..", "..",
                                          "sample", "fused_recurrent_gated_delta_rule"))


def _restore_strided_tensor(saved_data, meta):
    if not meta or meta.get("contiguous", True):
        return saved_data
    dtype = getattr(torch, meta["dtype"].replace("torch.", ""))
    tensor = torch.empty_strided(meta["shape"], meta["stride"], dtype=dtype)
    tensor.copy_(saved_data)
    return tensor


def _load_gpu_case(filename):
    data = torch.load(os.path.join(_DATA_DIR, filename), map_location="cpu", weights_only=False)
    inputs = data["inputs"]
    meta = data.get("input_meta", {})

    # 恢复非连续 stride
    state_restored = _restore_strided_tensor(inputs["initial_state"], meta.get("initial_state", {}))

    # q/k L2 归一化（GPU kernel 内部做，NPU 需要外部做）
    q_npu = torch.nn.functional.normalize(
        inputs["q"].squeeze(0).squeeze(0).to(torch.float32), p=2, dim=-1
    ).to(torch.bfloat16)
    k_npu = torch.nn.functional.normalize(
        inputs["k"].squeeze(0).squeeze(0).to(torch.float32), p=2, dim=-1
    ).to(torch.bfloat16)
    v_npu = inputs["v"].squeeze(0).squeeze(0).to(torch.bfloat16)
    beta_npu = inputs["beta"].squeeze(0).squeeze(0).to(torch.bfloat16)
    g_npu = inputs["g"].squeeze(0).squeeze(0)  # float32

    # 确保 3D
    if q_npu.dim() == 2: q_npu = q_npu.unsqueeze(0)
    if k_npu.dim() == 2: k_npu = k_npu.unsqueeze(0)
    if v_npu.dim() == 2: v_npu = v_npu.unsqueeze(0)
    if beta_npu.dim() == 1: beta_npu = beta_npu.unsqueeze(0)
    if g_npu.dim() == 1: g_npu = g_npu.unsqueeze(0)

    # paged state: 提取活跃 page，转置 (dk,dv)→(dv,dk)，转 bfloat16
    page_idx = int(inputs["block_map"][0, 0].item())
    state_page = state_restored[page_idx].transpose(-1, -2).contiguous().to(torch.bfloat16)
    state_npu = state_page.unsqueeze(0)

    # scale 默认值
    dk = q_npu.shape[-1]
    scale = inputs.get("scale")
    if scale is None:
        scale = float(dk ** -0.5)

    # actual_seq_lengths / ssm_state_indices
    actual_seq_lengths = torch.tensor([0, 1], dtype=torch.int32)
    t = int(actual_seq_lengths.sum().item())
    ssm_state_indices = torch.tensor([0] * t, dtype=torch.int32)

    # GPU 期望输出
    out_expected = data["outputs"][0].squeeze(0).squeeze(0)
    if out_expected.dim() == 2:
        out_expected = out_expected.unsqueeze(0)

    return {
        "q": q_npu, "k": k_npu, "v": v_npu, "state": state_npu,
        "state_restored": state_restored, "beta": beta_npu, "g": g_npu,
        "scale": scale, "actual_seq_lengths": actual_seq_lengths,
        "ssm_state_indices": ssm_state_indices, "page_idx": page_idx,
        "out_expected": out_expected,
        "state_expected": data["inplace_outputs"]["initial_state"],
    }


class TestRecurrentGatedDeltaRuleGpuGolden(unittest.TestCase):
    rtol = 5e-2
    atol = 5e-2

    def call_op(self, q, k, v, state, **kwargs):
        return ascendc_ops.npu_recurrent_gated_delta_rule(q, k, v, state, **kwargs)

    def assertTensorClose(self, actual, expected, *, rtol=None, atol=None):
        rtol = self.rtol if rtol is None else rtol
        atol = self.atol if atol is None else atol
        actual_cpu = actual.detach().cpu().float()
        expected_cpu = expected.detach().cpu().float()
        self.assertTrue(
            torch.allclose(actual_cpu, expected_cpu, rtol=rtol, atol=atol),
            msg=f"max_abs_diff={(actual_cpu - expected_cpu).abs().max().item():.6f}",
        )

    def test_decode(self):
        case = _load_gpu_case("decode.pt")

        # 验证 state 恢复了非连续 stride
        self.assertFalse(case["state_restored"].is_contiguous())

        # state 原地更新：保存 NPU 引用
        state_device = case["state"].npu()

        out, _ = self.call_op(
            case["q"].npu(), case["k"].npu(), case["v"].npu(), state_device,
            beta=case["beta"].npu(), scale=case["scale"],
            actual_seq_lengths=case["actual_seq_lengths"].npu(),
            ssm_state_indices=case["ssm_state_indices"].npu(),
            g=case["g"].npu(),
        )

        self.assertTensorClose(out, case["out_expected"])

        # state 比较：NPU (nv,dv,dk) → 转回 GPU (nv,dk,dv)
        page_idx = case["page_idx"]
        state_actual = state_device.cpu()[0].transpose(-1, -2)
        state_expected_page = case["state_expected"][page_idx].to(torch.float32)
        self.assertTensorClose(state_actual, state_expected_page)
```

## 8. 调试技巧

1. **先用 CPU golden 验证 state 布局**：分别尝试 `(nv, dv, dk)` 和 `(nv, dk, dv)` 两种布局跑 golden，对比 GPU 期望输出，max_diff 最小的即为正确布局
2. **检查 q/k 是否已归一化**：打印 `q.norm(p=2, dim=-1)`，若非 1 则需要归一化
3. **读 aclnn 头文件**：`aclnn_recurrent_gated_delta_rule.h` 中注释明确标注了每个参数的 dtype 要求
4. **对比解耦路径与旧路径**：`fla_npu.ops.ascendc` 解耦路径不修改 star_idx 位置 state，`torch_npu` 旧路径会修改，用于定位行为差异
5. **验证 state 转置方向**：转置后用 CPU golden 跑一遍，确认 output max_diff < 0.001、state max_diff < 0.04
6. **paged state 内存优化**：完整 293 pages 在 float32 下约 461MB，提取单个 page 即可避免设备内存问题
7. **state 原地更新**：必须保存 NPU tensor 引用，算完后 `.cpu()` 同步，不能比较 CPU 原始 tensor
8. **输出结构差异**：GPU dump 的 `outputs` 是列表 `[out, final_state]`，`inplace_outputs` 是 dict，两者 `final_state` 内容一致

## 9. GPU 与 NPU 布局策略对比

| 维度 | GPU dump (rtp-llm) | NPU 算子 |
|------|---------------------|---------|
| `q/k` dtype | float16 | bfloat16 |
| `q/k` 归一化 | kernel 内部做 (`use_qk_l2norm_in_kernel=True`) | 外部预处理 |
| `v/beta` dtype | float16 | bfloat16 |
| `state` dtype | bfloat16 | bfloat16（**不支持 float32**） |
| `state` 布局 | `(nv, dk, dv)` | `(nv, dv, dk)` — 需转置 |
| `state` 非连续性 | paged, stride=(524288, 16384, 128, 1) | 提取单 page 后 contiguous |
| `g` dtype | float32 | float32（一致） |
| `scale` | None（默认 `dk**-0.5`） | float |
| `actual_seq_lengths` | 需构造 | int32 |
| `ssm_state_indices` | 需构造 | int32 |
| 输出结构 | `outputs[0]` + `inplace_outputs` | `(out, state)` 元组 |
| `star_idx` 位置 state | 不修改 | 解耦路径不修改；旧路径会修改 |
| 输出 `out` 布局 | `(1, 1, nv, dv)` float16 | `(t, nv, dv)` bfloat16 |

**结论**：`recurrent_gated_delta_rule` 的核心比对难点在于三个转换：(1) q/k 的 L2 归一化（GPU 内部做 vs NPU 外部做）、(2) state 的布局转置 `(dk, dv)→(dv, dk)`、(3) state 的 dtype 必须为 bfloat16。此外，`torch_npu` 旧路径对 star_idx 位置 state 的行为差异需注意，建议使用解耦路径进行精度验证。
