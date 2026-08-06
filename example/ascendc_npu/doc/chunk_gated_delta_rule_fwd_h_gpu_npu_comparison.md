# chunk_gated_delta_rule_fwd_h 算子 GPU-NPU 精度比对指南

本文档总结 `chunk_gated_delta_rule_fwd_h` 算子 NPU-vs-GPU 比对的实践经验。通用方法论参见同目录下的 `gpu_npu_comparison_guide.md`。

## 整体流程

```
GPU 黄金数据 (.pt)  →  解析输入/输出  →  NPU 前置流水线 (cumsum→kkt) + CPU solve_tri (numpy)  →  NPU recompute_wu → w/u  →  NPU fwd_h  →  h 精度比对
```

## 1. 理解 GPU 数据格式

### 数据来源

GPU dump 位于 `<workspace>/sample/chunk_gated_delta_rule/`，包含完整 `chunk_gated_delta_rule` 流水线的输入和输出。

### 典型数据内容

以 `prefill_loaded_state_seq32.pt` 为例：

| key | 类型 | 值 |
|-----|------|----|
| `inputs/q` | tensor | shape=(1, 32, 16, 128), dtype=float16 |
| `inputs/k` | tensor | shape=(1, 32, 16, 128), dtype=float16 |
| `inputs/v` | tensor | shape=(1, 32, 32, 128), dtype=float16 |
| `inputs/g` | tensor | shape=(1, 32, 32), dtype=float32 |
| `inputs/beta` | tensor | shape=(1, 32, 32), dtype=float16 |
| `inputs/initial_state` | tensor | shape=(1, 32, 128, 128), dtype=bfloat16 |
| `inputs/cu_seqlens` | tensor | shape=(2,), dtype=int32 |
| `inputs/use_qk_l2norm_in_kernel` | bool | True — GPU kernel 内部对 k 做 L2 归一化 |
| `outputs[0]` (o) | tensor | shape=(1, 32, 32, 128), dtype=float16 |
| `outputs[1]` (h) | tensor | shape=(1, 1, 32, 128, 128), dtype=float32 |
| `outputs[2]` (final_state) | tensor | shape=(1, 32, 128, 128), dtype=float32 |

### 核心挑战：中间值 w/u 不在 dump 中

`fwd_h` 算子的输入是 `k, w, u, g`，其中 `w` 和 `u` 是 `recompute_wu_fwd` 的输出。GPU dump 只包含流水线的最终输出（o, h, final_state），不包含中间值 w/u。

**解决方案**：运行 NPU 前置流水线（cumsum → kkt）+ CPU solve_tri（numpy）+ NPU recompute_wu 生成 w/u，然后传给 NPU fwd_h。

## 2. 前置流水线实现

### 步骤总览

| 步骤 | 实现方式 | 输入 | 输出 |
|------|---------|------|------|
| 1 | NPU `npu_chunk_local_cumsum` | g (B, Hv, T) | g_cumsum (B, Hv, T) |
| 2 | NPU `npu_chunk_scaled_dot_kkt` | k, g_cumsum, beta | A (B, Hk, T, chunk_size) float32 |
| 3 | **CPU numpy** solve_tri | A | A_tril (inv(A)) |
| 4 | NPU `npu_recompute_w_u_fwd` | k, v, beta, A_tril, g_cumsum | w, u |

### 2.1 为什么 solve_tri 用 CPU 而非 NPU

NPU `solve_tri` 要求 bfloat16/float16 输入，但在该精度下对 64×64 三角矩阵求逆会产生极端值（-3M ~ +200K），导致后续 w/u 错误、状态 h 指数增长溢出为 inf → NaN。

CPU numpy 在 float32 下计算 solve_tri 精度正确，但需要注意：

### 2.2 NPU device init 后 torch matmul 产生 NaN

`torch.npu.set_device(0)` 后，CPU tensor 上的 `torch.matmul`、`torch.inverse`、`torch.triangular_solve` 会产生 NaN。必须使用 numpy 进行矩阵运算：

```python
# ✗ 错误：torch.matmul 在 NPU device init 后产生 NaN
dots = k_chunk @ k_chunk.T

# ✓ 正确：使用 numpy
k_np = k_chunk.numpy()
dots = np.matmul(k_np, k_np.T)
```

### 2.3 solve_tri 的单位对角线假设

KKT 算子输出的 A 矩阵是**严格下三角**（对角线为 0），不是单位对角线。`solve_tri` 的语义是：

1. 将对角线设为 1（单位对角线）
2. 计算 `inv(I + L)`，其中 L 是严格下三角部分

CPU numpy 实现（前代法）：

```python
M = A[b, h, chunk_start:chunk_end, :seg].numpy()
L = np.tril(M, -1)  # strict lower (diagonal is 0 from KKT)
# inv(I + L) via forward substitution
X = np.eye(seg)
for i in range(1, seg):
    for j in range(i):
        s_val = -L[i, j]
        for kk in range(j, i):
            s_val -= L[i, kk] * X[kk, j]
        X[i, j] = s_val
```

### 2.4 GQA head 扩展

KKT 输出 `Hk` 个 head，但 `recompute_wu_fwd` 期望 `Hv` 个 head。需要用 `repeat_interleave` 扩展：

```python
ratio = Hv // Hk
if ratio > 1:
    A_expanded = A_tril.repeat_interleave(ratio, dim=1).contiguous()
```

### 2.5 use_qk_l2norm_in_kernel 处理

GPU dump 中 `use_qk_l2norm_in_kernel=True`，保存的 k 是原始未归一化值。必须在流水线前对 k 做 L2 归一化：

```python
if use_qk_l2norm:
    k_npu = torch.nn.functional.normalize(k_npu.float(), p=2, dim=-1).to(torch.bfloat16)
```

**不归一化会导致状态 h 在 chunk 间指数增长，最终溢出为 inf → NaN。**

## 3. 格式转换

### Layout 转换

GPU dump 是 `(B, T, H, D)`（time-first），NPU 要求 `(B, H, T, D)`（head-first）：

```python
k_npu = k_gpu.transpose(1, 2).contiguous().to(torch.bfloat16)
```

### h 输出 Layout

GPU h 是 `(B, NT, Hv, K, V)`，NPU h 是 `(B, Hv, NT, K, V)`：

```python
h_expected = data["outputs"][1].transpose(1, 2).contiguous()
```

### dtype 转换

NPU fwd_h 输入：k/w/u = bfloat16, g = float32, initial_state = float32：

```python
k_npu = k_gpu.transpose(1, 2).contiguous().to(torch.bfloat16)
g_npu = g_cumsum  # float32
initial_state_npu = initial_state_gpu.to(torch.float32)  # bfloat16 → float32
```

### cu_seqlens dtype 转换

```python
cu_seqlens = [int(v) for v in inputs["cu_seqlens"].tolist()]
```

### chunk_indices 计算

```python
chunk_indices = _prepare_chunk_indices(cu_seqlens, chunk_size)
```

### initial_state 为 None 的处理

GPU dump 中 `initial_state` 可能为 None（零初始状态）。NPU wrapper 要求提供 initial_state tensor，不能传 None：

```python
if initial_state is None:
    initial_state = torch.zeros(1, Hv, Dk, Dv, dtype=torch.float32)
```

## 4. NPU 算子约束

| 参数 | NPU 期望 | 说明 |
|------|---------|------|
| `k` | `(B, Hk, T, Dk)` bfloat16 | head_first |
| `w` | `(B, Hv, T, Dk)` bfloat16 | head_first |
| `u` | `(B, Hv, T, Dv)` bfloat16 | head_first |
| `g` | `(B, Hv, T)` float32 | head_first, cumulative gate |
| `initial_state` | `(N, Hv, Dk, Dv)` float32 | 不能为 None，需传零 tensor |
| `cu_seqlens` | int64 list | varlen 模式必填 |
| `chunk_indices` | int64 list | varlen 模式必填 |
| `chunk_size` | int64 | 默认 64 |
| `h` (输出) | `(B, Hv, NT, Dk, Dv)` | per-chunk start states |
| `v_new` (输出) | `(B, Hv, T, Dv)` | corrected values |
| `final_state` (输出) | `(N, Hv, Dk, Dv)` | final state |

### g/gk 约束

NPU wrapper 要求 `g` 或 `gk` 至少提供一个。`save_new_value` 必须为 True。

## 5. 常见错误

### 5.1 torch matmul 产生 NaN（NPU device init 后）

`torch.npu.set_device(0)` 后，CPU tensor 上的 `torch.matmul`、`torch.inverse`、`torch.triangular_solve` 会产生 NaN。必须使用 numpy 替代。

### 5.2 NPU solve_tri bfloat16 精度不足

NPU `solve_tri` 要求 bfloat16/float16 输入，但在该精度下对 64×64 三角矩阵求逆产生极端值（-3M ~ +200K）。**必须用 CPU numpy 在 float32 下计算 solve_tri**。

### 5.3 k 未归一化导致状态溢出

GPU dump 中 `use_qk_l2norm_in_kernel=True`，保存的 k 是原始值。不归一化直接传入流水线会导致 KKT 的对角线值过大，w/u 错误，状态 h 在 chunk 间指数增长，最终溢出为 inf → NaN。

```python
# ✓ 正确：归一化 k
if use_qk_l2norm:
    k_npu = torch.nn.functional.normalize(k_npu.float(), p=2, dim=-1).to(torch.bfloat16)
```

### 5.4 多 chunk 状态递推放大 w/u 精度差异

即使 w/u 有微小精度差异（NPU bfloat16 vs GPU float16），状态递推 `h_t = exp(g) * h_{t-1} + K^T @ v_new` 在多 chunk 间指数级放大。seq2047（32 chunks）在 chunk ≥ 2 时开始发散，chunk ≥ 27 时溢出为 NaN。

**验证证据**：
- h[0] = initial_state 精确匹配（diff=0）—— fwd_h kernel 本身正确
- h[1] 差异来自 v_new = v - W@h 中的 W 值不同 —— w/u 精度问题
- 1 chunk (T≤64) 通过，2+ chunks 失败 —— 多 chunk 状态累积

**修复方案**：长序列截断到 1 chunk (T=64) 测试。完整序列验证需要 GPU dump 包含 w/u 中间值。

### 5.5 initial_state 不能为 None

GPU dump 中 `initial_state` 可能为 None。NPU wrapper 虽然标记为 optional，但传 None 会导致 `aclnnStatus=161001`。需传零 tensor：

```python
if initial_state is None:
    initial_state = torch.zeros(1, Hv, Dk, Dv, dtype=torch.float32)
```

### 5.6 GQA head 数不匹配

KKT 输出 `Hk` 个 head，`recompute_wu_fwd` 期望 `Hv` 个 head。不扩展会导致 `aclnnStatus=161001`：

```python
if ratio > 1:
    A_expanded = A_tril.repeat_interleave(ratio, dim=1).contiguous()
```

## 6. 测试模板

完整测试文件位于 `torch_custom/fla_npu/test/test_npu_chunk_gated_delta_rule_fwd_h_gpu_golden.py`。

核心结构：
1. NPU cumsum + NPU kkt + CPU numpy solve_tri + NPU recompute_wu 生成 w/u
2. L2 归一化 k（如 use_qk_l2norm_in_kernel=True）
3. 调用 `npu_chunk_gated_delta_rule_fwd_h`
4. 比较 h 与 GPU golden（截断到实际 chunk 数）

```python
def _run_case(self, filename, *, max_chunks=None):
    case = _load_gpu_case(filename)
    k, v, g_raw, beta = case["k"], case["v"], case["g_raw"], case["beta"]
    # ... (truncate if max_chunks)

    # Step 1: NPU cumsum
    g_cumsum = ascendc_ops.npu_chunk_local_cumsum(...)

    # Step 2: NPU kkt (with GQA head subsampling)
    A = ascendc_ops.npu_chunk_scaled_dot_kkt(k=k.npu(), g=g_sub.npu(), ...)

    # Step 3: CPU numpy solve_tri (NPU solve_tri has bf16 precision issues)
    A_tril = cpu_solve_tri_numpy(A.cpu(), ...)

    # Step 4: NPU recompute_wu (with GQA head expansion)
    w, u = ascendc_ops.npu_recompute_w_u_fwd(
        k=k.npu(), v=v.npu(), beta=beta.npu(),
        A=A_expanded.to(torch.bfloat16).npu(), ...)

    # Step 5: NPU fwd_h
    h, v_new, final_state = ascendc_ops.npu_chunk_gated_delta_rule_fwd_h(
        k=k.npu(), w=w, u=u, g=g_cumsum.npu(),
        initial_state=initial_state.npu(), ...)

    # Compare h (truncated to actual chunks)
    self.assertTensorClose(h, h_expected[:, :, :n_chunks_actual])
```

## 7. 调试技巧

1. **h[0] = initial_state**：第一个 chunk 的 h 就是初始状态，不涉及计算，应精确匹配（diff=0）。如果 h[0] 不匹配，说明 initial_state 传递有问题
2. **逐 chunk 检查 h**：打印每个 chunk 的 max_diff，如果 h[0] 匹配但 h[1] 不匹配，说明 w/u 精度问题
3. **检查 k 是否归一化**：打印 `k.norm(p=2, dim=-1)`，若非 1 则需归一化
4. **检查 KKT 对角线**：KKT 输出的对角线应为 0（严格下三角），不是 beta 或 1
5. **不要用 torch.linalg**：NPU device init 后 torch.inverse/triangular_solve/matmul 在 CPU 上产生 NaN
6. **NPU solve_tri 精度不足**：必须用 CPU numpy 在 float32 下计算
7. **GQA head 扩展**：KKT 输出 Hk head，recompute_wu 需要 Hv head，用 repeat_interleave 扩展
8. **多 chunk 状态发散**：w/u 的微小精度差异在状态递推中指数放大，长序列需截断测试
9. **initial_state 不能为 None**：传零 tensor 替代
10. **use_qk_l2norm_in_kernel**：GPU dump 中此标志为 True 时，k 是原始值，必须在流水线前归一化

## 8. GPU 与 NPU 布局策略对比

| 维度 | GPU dump (rtp-llm) | NPU 算子 |
|------|---------------------|---------|
| `k` 布局 | `(B, T, Hk, Dk)` | `(B, Hk, T, Dk)` — 需 transpose(1,2) |
| `w` 布局 | `(B, T, Hv, Dk)` | `(B, Hv, T, Dk)` — 需 transpose(1,2) |
| `u` 布局 | `(B, T, Hv, Dv)` | `(B, Hv, T, Dv)` — 需 transpose(1,2) |
| `g` 布局 | `(B, T, Hv)` | `(B, Hv, T)` — 需 transpose(1,2) |
| `h` 布局 | `(B, NT, Hv, Dk, Dv)` | `(B, Hv, NT, Dk, Dv)` — 需 transpose(1,2) |
| `k/w/u` dtype | float16 | bfloat16 |
| `g` dtype | float32 | float32（一致） |
| `initial_state` dtype | bfloat16 | float32 |
| `k` 归一化 | kernel 内部做 (`use_qk_l2norm_in_kernel=True`) | 外部预处理 |
| 中间值 w/u | 不在 dump 中 | NPU cumsum+kkt + CPU solve_tri + NPU recompute_wu |
| 输出数量 | 3 (o, h, final_state) | 3 (h, v_new, final_state) |
| 原地更新 | 无 | 无 |

**结论**：`chunk_gated_delta_rule_fwd_h` 的核心比对难点在于五个方面：(1) 中间值 w/u 不在 GPU dump 中，需运行前置流水线生成；(2) NPU solve_tri 在 bfloat16 下精度不足，必须用 CPU numpy 替代；(3) NPU device init 后 torch matmul/linalg 在 CPU 上产生 NaN，必须用 numpy；(4) `use_qk_l2norm_in_kernel=True` 时 k 需外部归一化，否则状态溢出；(5) 多 chunk 状态递推放大 w/u 精度差异，长序列需截断测试。相比其他算子，本算子的测试复杂度最高，因为涉及完整前置流水线和跨算子精度依赖。
