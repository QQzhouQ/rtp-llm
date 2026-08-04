# recurrent_gated_delta_rule 算子 GPU-NPU 精度比对指南

本文档总结 `recurrent_gated_delta_rule` 算子 NPU-vs-GPU 比对的实践经验，供后续该算子维护和类似 recurrent 类算子比对参考。通用方法论参见同目录下的 `gpu_npu_comparison_guide.md`。
- 该算子需要cann-9.1.0支持

## 整体流程

```
GPU 黄金数据 (.pt)  →  恢复非连续 stride  →  解析输入/输出/属性  →  q/k L2 归一化  →  构造非连续 paged state buffer (0维gap + dk内层stride=1)  →  dtype 转换 float32→bfloat16  →  调用 NPU 算子  →  从非连续 buffer 提取活跃 page + 转置回 (dk,dv)  →  精度比对
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
    "stride": (1048576, 16384, 128, 1),  # 0-dim stride=2*page_size, 有 2x gap
    "dtype": "torch.bfloat16",
    "contiguous": False
}
```

**0 维非连续性分析**：

```
shape = (293, 32, 128, 128)   # (num_pages, nv, dk, dv)
期望连续 stride[0] = 32*128*128 = 524288
实际 GPU stride[0] = 1048576  (= 2 × 524288)
```

stride[0] 是期望值的 **2 倍**，说明每个 page 之间有 524288 个元素的间隙（1MB，bfloat16）。这是 rtp-llm paged state 的典型布局——每个 page 分配双倍存储，stride[0] 的 gap 用于对齐或预留扩展空间。

`torch.save()` 会将非连续张量以连续形式保存（stride 变为 524288），必须从 `input_meta` 恢复原始 stride 才能还原 GPU 的真实非连续布局。

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

### 5.2 state 布局转置与非连续 0 维构造（核心）

GPU state 布局是 `(nv, dk, dv)`，NPU 期望 `(nv, dv, dk)`。必须转置最后两个维度。

同时，GPU 的 `initial_state` 在 0 维（num_pages 维）是非连续的（stride[0] = 2 × page_size）。NPU kernel 通过 `IgnoreContiguous()` + `stateStride0_` 支持非连续 0 维寻址，测试需要验证这条路径。

**关键约束**：kernel 要求 dim 3（dk，最内层）stride=1（`DataCopy` 按 `dk` 元素连续拷贝）。直接 `transpose(-1, -2)` 完整 paged tensor 会使 dk 的 stride 变为 128（非 1），违反约束。因此不能像 `causal_conv1d` 那样简单转置完整 paged tensor。

**解决方案**：分配一个 0 维非连续但内层连续的 NPU 布局 buffer，逐 page 转置拷贝：

```python
# GPU state: (num_pages, nv, dk, dv), stride=(1048576, 16384, 128, 1) — 0维非连续
# NPU state: (test_num_pages, nv, dv, dk), stride=(2*page_elems, dv*dk, dk, 1) — 0维非连续, dk内层连续

num_pages_gpu, nv, dk, dv = state_restored.shape
test_num_pages = 3                  # 小 buffer 节省内存
test_page_idx = 1                   # 活跃 page 放在 index 1
page_elems = nv * dv * dk           # 32 * 128 * 128 = 524288
npu_stride = (2 * page_elems, dv * dk, dk, 1)  # 0维有 2x gap，与 GPU 一致

state_npu = torch.empty_strided(
    (test_num_pages, nv, dv, dk), npu_stride, dtype=torch.bfloat16
)
state_npu.fill_(0)
# 逐 page 转置拷贝：(dk, dv) → (dv, dk)
state_npu[test_page_idx].copy_(
    state_restored[page_idx].transpose(-1, -2).to(torch.bfloat16)
)

# ssm_state_indices 指向 test_page_idx
ssm_state_indices = torch.tensor([test_page_idx] * t, dtype=torch.int32)
```

**为什么不能直接 transpose 完整 paged tensor**：

| 方案 | stride[2] (dv) | stride[3] (dk) | 问题 |
|------|----------------|----------------|------|
| GPU 原始 | 128 | 1 | dv 内层连续，dk 非 stride-1 |
| `transpose(-1,-2)` 完整 paged | 1 | 128 | dk 非 stride-1，kernel `DataCopy` 要求 dk stride=1 |
| **新方案（逐 page 拷贝）** | dk=128 | 1 | dk stride=1 ✓，0 维非连续 ✓ |

NPU 输出的 state 也是 `(nv, dv, dk)`，比较时需转置回 `(nv, dk, dv)`，并从非连续 buffer 中提取活跃 page：

```python
# NPU 输出 state 是 (nv, dv, dk)，转回 GPU 布局 (nv, dk, dv)
# 从非连续 buffer 中提取 test_page_idx 对应的 page
state_actual = state_device.cpu()[test_page_idx].transpose(-1, -2)  # (nv, dk, dv)
```

### 5.3 为什么 state 必须转置——存储视图与 kernel 实际访问的语义矛盾

**存储视图声明的布局**与 **kernel 实际访问的布局**在语义上相反，只在 `dk == dv`（方阵）时不可见。

#### 存储视图声明：`(nv, dv, dk)`，dk 连续

`get_ssm_state_tensor`（`typed_storage_view.py:111-131`）声明的 state 布局：

```python
size = (num_pages, local_num_v_heads, head_v_dim, head_k_dim)  # (N, nv, dv, dk)
stride_bytes = (block_size_bytes, dv*dk*item, dk*item, item)    # dk 是 stride=1（内层连续）
```

形式上与 NPU 期望的 `(nv, dv, dk)` 一致。但存储视图只是**声明**，实际数据语义取决于 kernel 如何写入和读取。

#### GPU Triton kernel 实际访问：`(dk, dv)`，dv 连续

`fused_recurrent_gated_delta_rule_fwd_kernel`（`fused_recurrent.py:33`）中，state tile 声明为 `[BK, BV] = [dk, dv]`，且指针运算硬编码了 dv 为内层连续维度：

```python
# fused_recurrent.py:115
b_h = tl.zeros([BK, BV], dtype=tl.float32)   # [dk, dv] — dk 是行，dv 是列

# fused_recurrent.py:124-128 — 加载 initial_state
p_h0 = p_h0 + i_hv * K * V + o_k[:, None] * V + o_v[None, :]
#                            ^^^^^^^^^^^^      ^^^^^^^^^^^
#                            dk 的 stride=V=dv   dv 的 stride=1（内层连续）
```

存储时（`fused_recurrent.py:163-167`）使用相同模式：

```python
p_ht = p_ht + i_hv * K * V + o_k[:, None] * V + o_v[None, :]
```

即 kernel 按 `(dk, dv)` 语义写入数据，**dv 是 stride=1 的内层连续维度**——与存储视图声明的 `(dv, dk)` dk 连续**相反**。

#### 方阵掩盖了语义矛盾

| 层 | 声明/假设 | 内层连续维度 | stride (dv=dk=128) |
|---|----------|------------|-------------------|
| 存储视图 `get_ssm_state_tensor` | `(nv, dv, dk)` | dk | `(..., 16384, 128, 1)` |
| Triton kernel 实际访问 | `(nv, dk, dv)` | dv | `(..., 16384, 128, 1)` |

当 `dk == dv == 128` 时，两种布局的 stride **完全相同**（都是 `(..., 128, 1)`），byte offset 一致，只是行/列的语义标签被调换。GPU 代码有显式假设：

```python
# qwen3_next.py:355
# asserr head_k_dim == head_v_dim
```

方阵下转置不可见，GPU 自洽运行。但 dump 出的数据**实际是数学 state 矩阵 S 的转置 S^T**。

#### NPU 比对时必须显式转置

| 消费者 | 读 dump 字节当 | 数学语义 | 是否正确 |
|--------|---------------|---------|---------|
| GPU kernel | `(dk, dv)` dv 连续 | 看到 S^T，kernel 数学适应了转置视角 | 正确 |
| NPU kernel（不转置） | `(dv, dk)` dk 连续 | 看到 S^T 却当成 S | **错误**，max_diff ≈ 1.5~9.9 |
| NPU kernel（转置后） | `(dv, dk)` dk 连续 | 看到 S，正确的数学 state 矩阵 | **正确** |

```python
# dump 数据实际是 S^T (dk, dv) — dv 连续
# transpose(-1, -2) 将 S^T 转为 S (dv, dk) — dk 连续
state_npu[test_page_idx].copy_(
    state_restored[page_idx].transpose(-1, -2).to(torch.bfloat16)
)
```

#### prefetch 路径的佐证

prefill 阶段的 `store_ssm_state_to_block_map_kernel`（`block.py:97-178`）和 `load_initial_state_from_block_map_kernel`（`block.py:8-63`）使用 `tl.make_block_ptr(..., (V, K), (K, 1), ...)`，即 `(dv, dk)` dk 连续——与存储视图声明一致，但与 decode kernel 的 `(dk, dv)` dv 连续相反。这说明 prefill 和 decode 之间也存在隐式转置，同样被 `dk == dv` 的方阵假设掩盖。

### 5.4 dtype 转换

GPU dump 中 q/k/v/beta 是 float16，state 是 bfloat16。NPU 要求全部为 bfloat16：

```python
q_npu = ...to(torch.bfloat16)
k_npu = ...to(torch.bfloat16)
v_npu = inputs["v"].squeeze(0).squeeze(0).to(torch.bfloat16)
beta_npu = inputs["beta"].squeeze(0).squeeze(0).to(torch.bfloat16)
```

**state 必须是 bfloat16**，传 float32 会报 `aclnnStatus=161002`（参数错误）。

### 5.5 维度压缩

GPU dump 的 q/k/v/beta/g 是 4D/3D `(batch, seqlen, n, d)`，NPU 期望 3D/2D `(t, n, d)`：

```python
q_npu = inputs["q"].squeeze(0).squeeze(0)  # (batch=1, seqlen=1, nk, dk) → (nk, dk)
if q_npu.dim() == 2:
    q_npu = q_npu.unsqueeze(0)  # (1, nk, dk)
```

### 5.6 actual_seq_lengths 与 ssm_state_indices 构造

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

> 根因分析详见 [5.3 节](#53-为什么-state-必须转置存储视图与-kernel-实际访问的语义矛盾)：GPU 存储视图声明 `(nv, dv, dk)` 与 NPU 一致，但 Triton kernel 实际按 `(nv, dk, dv)` 语义访问数据，方阵下 stride 碰巧一致掩盖了语义转置。

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

### 6.4 state 0 维非连续性未验证（提取单 page 后 .contiguous()）

GPU 的 `initial_state` 在 0 维（num_pages 维）是非连续的（stride[0] = 2 × page_size）。NPU kernel 通过 `stateStride0_` 支持非连续 0 维寻址（`IgnoreContiguous()`）。如果测试提取单个 page 后调用 `.contiguous()`，传入 NPU 的 state 是完全连续的，没有验证这条路径。

```python
# ✗ 错误：提取单 page 后 .contiguous()，stride 变连续，无法验证非连续寻址
state_page = state_restored[page_idx].transpose(-1, -2).contiguous().to(torch.bfloat16)
state_npu = state_page.unsqueeze(0)  # (1, nv, dv, dk) — 连续！

# ✓ 正确：分配非连续 paged buffer，0 维有 gap，活跃 page 放在中间
state_npu = torch.empty_strided(
    (3, nv, dv, dk), (2 * page_elems, dv * dk, dk, 1), dtype=torch.bfloat16
)
state_npu[1].copy_(state_restored[page_idx].transpose(-1, -2).to(torch.bfloat16))
ssm_state_indices = torch.tensor([1], dtype=torch.int32)  # 指向 page 1
```

**注意**：不能直接 `transpose(-1, -2)` 完整 paged tensor，因为这会使 dk（dim 3）的 stride 变为 128（非 1），而 kernel 的 `DataCopy` 要求 dk 内层 stride=1。必须分配 NPU 布局 buffer 后逐 page 转置拷贝。

测试中应添加非连续性断言：

```python
# 验证传入 NPU 的 state 确实是非连续的
self.assertFalse(state_npu.is_contiguous(), "NPU state should be non-contiguous (paged layout)")
self.assertNotEqual(state_npu.stride(0), state_npu.shape[1] * state_npu.stride(1),
                    "state stride[0] should have gap (non-contiguous 0-dim)")
```

### 6.5 torch_npu 旧路径修改 star_idx 位置的 state

使用 `torch_npu.npu_recurrent_gated_delta_rule`（torch_npu 内置实现）时，`actual_seq_lengths[0]`（star_idx）位置的 state 会被修改，而 golden 参考实现不处理该位置。这会导致 `final_state` 在 position 0 出现精度失败。

```
# torch_npu 旧路径:
state[0] 调用前=0.8828125 → 调用后=0.31640625 (被修改) → FAIL

# fla_npu.ops.ascendc 解耦路径:
state[0] 调用前=0.8828125 → 调用后=0.8828125 (不变) → PASS
```

**建议**：精度验证使用 `fla_npu.ops.ascendc` 解耦路径（`test_run_standalone.py`），而非 `torch_npu` 旧路径（`test_accuracy.py`）。

### 6.6 GPU .pt 文件路径找不到

测试默认从 `<workspace>/sample/fused_recurrent_gated_delta_rule/` 加载数据。若路径不同，检查实际挂载位置：

```sh
find / -name "decode.pt" -path "*/fused_recurrent_gated_delta_rule/*" 2>/dev/null
```

### 6.7 scale 为 None 时未解析默认值

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

    # 构造非连续 paged state buffer（验证 kernel 的 stateStride0_ 寻址路径）
    # GPU state: (num_pages, nv, dk, dv), 0维非连续 (stride[0]=2*page_size)
    # NPU state: (3, nv, dv, dk), 0维非连续, dk 内层 stride=1
    page_idx = int(inputs["block_map"][0, 0].item())
    num_pages_gpu, nv, dk, dv = state_restored.shape
    test_num_pages = 3
    test_page_idx = 1
    page_elems = nv * dv * dk
    npu_stride = (2 * page_elems, dv * dk, dk, 1)
    state_npu = torch.empty_strided(
        (test_num_pages, nv, dv, dk), npu_stride, dtype=torch.bfloat16
    )
    state_npu.fill_(0)
    state_npu[test_page_idx].copy_(
        state_restored[page_idx].transpose(-1, -2).to(torch.bfloat16)
    )

    # scale 默认值
    dk = q_npu.shape[-1]
    scale = inputs.get("scale")
    if scale is None:
        scale = float(dk ** -0.5)

    # actual_seq_lengths / ssm_state_indices（指向 test_page_idx）
    actual_seq_lengths = torch.tensor([0, 1], dtype=torch.int32)
    t = int(actual_seq_lengths.sum().item())
    ssm_state_indices = torch.tensor([test_page_idx] * t, dtype=torch.int32)

    # GPU 期望输出
    out_expected = data["outputs"][0].squeeze(0).squeeze(0)
    if out_expected.dim() == 2:
        out_expected = out_expected.unsqueeze(0)

    return {
        "q": q_npu, "k": k_npu, "v": v_npu, "state": state_npu,
        "state_restored": state_restored, "beta": beta_npu, "g": g_npu,
        "scale": scale, "actual_seq_lengths": actual_seq_lengths,
        "ssm_state_indices": ssm_state_indices, "page_idx": page_idx,
        "test_page_idx": test_page_idx,
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

        # 验证 GPU state 恢复了非连续 stride
        self.assertFalse(case["state_restored"].is_contiguous(),
                         "GPU state should be non-contiguous after stride restoration")

        # 验证传入 NPU 的 state 也是非连续的（0维有 gap）
        state_npu = case["state"]
        self.assertFalse(state_npu.is_contiguous(),
                         "NPU state should be non-contiguous (paged layout)")
        self.assertNotEqual(state_npu.stride(0), state_npu.shape[1] * state_npu.stride(1),
                            "state stride[0] should have gap (non-contiguous 0-dim)")

        # state 原地更新：保存 NPU 引用
        state_device = state_npu.npu()

        out, _ = self.call_op(
            case["q"].npu(), case["k"].npu(), case["v"].npu(), state_device,
            beta=case["beta"].npu(), scale=case["scale"],
            actual_seq_lengths=case["actual_seq_lengths"].npu(),
            ssm_state_indices=case["ssm_state_indices"].npu(),
            g=case["g"].npu(),
        )

        self.assertTensorClose(out, case["out_expected"])

        # state 比较：从非连续 buffer 提取活跃 page，转回 GPU 布局
        test_page_idx = case["test_page_idx"]
        page_idx = case["page_idx"]
        state_actual = state_device.cpu()[test_page_idx].transpose(-1, -2)  # (nv, dk, dv)
        state_expected_page = case["state_expected"][page_idx].to(torch.float32)
        self.assertTensorClose(state_actual, state_expected_page)
```

## 8. 调试技巧

1. **先用 CPU golden 验证 state 布局**：分别尝试 `(nv, dv, dk)` 和 `(nv, dk, dv)` 两种布局跑 golden，对比 GPU 期望输出，max_diff 最小的即为正确布局
2. **检查 q/k 是否已归一化**：打印 `q.norm(p=2, dim=-1)`，若非 1 则需要归一化
3. **读 aclnn 头文件**：`aclnn_recurrent_gated_delta_rule.h` 中注释明确标注了每个参数的 dtype 要求
4. **对比解耦路径与旧路径**：`fla_npu.ops.ascendc` 解耦路径不修改 star_idx 位置 state，`torch_npu` 旧路径会修改，用于定位行为差异
5. **验证 state 转置方向**：转置后用 CPU golden 跑一遍，确认 output max_diff < 0.001、state max_diff < 0.04
6. **paged state 内存优化**：完整 293 pages 在 float32 下约 461MB，使用 3-page 非连续 buffer 即可验证 0 维非连续寻址，同时避免设备内存问题
7. **state 原地更新**：必须保存 NPU tensor 引用，算完后 `.cpu()` 同步，不能比较 CPU 原始 tensor
8. **输出结构差异**：GPU dump 的 `outputs` 是列表 `[out, final_state]`，`inplace_outputs` 是 dict，两者 `final_state` 内容一致
9. **验证 state 0 维非连续性**：打印 `state_npu.stride()` 和 `state_npu.is_contiguous()`，确认 stride[0] 有 gap（= 2 × page_size），且 dim 3（dk）stride=1
10. **读 kernel 代码确认 stride 约束**：`recurrent_gated_delta_rule.h` 中 `DataCopyPad` 的 `DataCopyExtParams` row stride 为 0，意味着 dim 3（dk）必须 stride=1；`stateStride0_` 用于 0 维非连续寻址
11. **不能直接 transpose 完整 paged tensor**：transpose(-1,-2) 会使 dk 的 stride 变为 128（非 1），违反 kernel 约束；必须分配 NPU 布局 buffer 后逐 page 转置拷贝
12. **读 GPU Triton kernel 源码确认实际访问语义**：`fused_recurrent.py` 中 `b_h = tl.zeros([BK, BV])` 和 `p_h0 + o_k[:, None] * V + o_v[None, :]` 表明 kernel 按 `(dk, dv)` dv 连续访问，与存储视图声明的 `(dv, dk)` dk 连续相反——方阵下不可见但跨平台比对时必须转置

## 9. GPU 与 NPU 布局策略对比

| 维度 | GPU dump (rtp-llm) | NPU 算子 |
|------|---------------------|---------|
| `q/k` dtype | float16 | bfloat16 |
| `q/k` 归一化 | kernel 内部做 (`use_qk_l2norm_in_kernel=True`) | 外部预处理 |
| `v/beta` dtype | float16 | bfloat16 |
| `state` dtype | bfloat16 | bfloat16（**不支持 float32**） |
| `state` 布局 | 存储视图声明 `(nv, dv, dk)`，但 kernel 实际按 `(nv, dk, dv)` 访问（方阵下不可见） | `(nv, dv, dk)` — 需转置（详见 5.3 节） |
| `state` 0 维非连续性 | paged, stride=(1048576, 16384, 128, 1)，0维有 2x gap | `IgnoreContiguous()` + `stateStride0_` 寻址，测试用 3-page 非连续 buffer 验证 |
| `state` dim 3 约束 | kernel 访问时 dv stride=1；存储视图声明 dk stride=1（方阵下一致） | dk stride=1（kernel `DataCopy` 要求） |
| `g` dtype | float32 | float32（一致） |
| `scale` | None（默认 `dk**-0.5`） | float |
| `actual_seq_lengths` | 需构造 | int32 |
| `ssm_state_indices` | 需构造 | int32 |
| 输出结构 | `outputs[0]` + `inplace_outputs` | `(out, state)` 元组 |
| `star_idx` 位置 state | 不修改 | 解耦路径不修改；旧路径会修改 |
| 输出 `out` 布局 | `(1, 1, nv, dv)` float16 | `(t, nv, dv)` bfloat16 |

**结论**：`recurrent_gated_delta_rule` 的核心比对难点在于五个关键点：(1) q/k 的 L2 归一化（GPU 内部做 vs NPU 外部做）、(2) state 的布局转置 `(dk, dv)→(dv, dk)`——根因是 GPU 存储视图声明与 Triton kernel 实际访问语义相反，方阵下 stride 碰巧一致掩盖了转置（详见 5.3 节）、(3) state 的 dtype 必须为 bfloat16、(4) state 的 0 维非连续性验证（需构造非连续 paged buffer 而非提取单 page 后 `.contiguous()`）、(5) `torch_npu` 旧路径对 star_idx 位置 state 的行为差异，建议使用解耦路径进行精度验证。
