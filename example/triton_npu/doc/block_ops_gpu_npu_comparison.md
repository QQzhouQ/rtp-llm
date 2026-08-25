# block ops 算子 GPU-NPU 精度比对指南

本文档总结 `load_initial_state_from_block_map` 和 `store_ssm_state_to_block_map` 两个算子的 NPU-vs-GPU 比对实践经验。

## 整体流程

```
GPU 黄金数据 (.pt)  →  克隆 inplace tensor  →  调用 Triton for Ascend kernel (NPU)  →  比对 inplace 输出
```

## 1. 理解 GPU 数据格式

### 数据来源

- `load_initial_state_from_block_map`: `<workspace>/sample/load_initial_state_from_block_map/prefill.pt`
- `store_ssm_state_to_block_map`: `<workspace>/sample/store_ssm_state_to_block_map/prefill.pt`

### load_initial_state_from_block_map 数据

| key | 类型 | 值 |
|-----|------|----|
| `inputs/prefix_lengths` | tensor | shape=(1,), dtype=int32 |
| `inputs/block_map` | tensor | shape=(1, 1), dtype=int32 |
| `inputs/conv_states` | tensor | shape=(293, 32, 128, 128), dtype=bfloat16, contiguous |
| `inputs/initial_states` | tensor | shape=(1, 32, 128, 128), dtype=bfloat16 |
| `inputs/seq_size_per_block` | int | 1024 |
| `inplace_outputs/initial_states` | tensor | shape=(1, 32, 128, 128), dtype=bfloat16 |
| `outputs` | None | 无非 inplace 输出 |

### store_ssm_state_to_block_map 数据

| key | 类型 | 值 |
|-----|------|----|
| `inputs/h` | tensor | shape=(1, 1, 32, 128, 128), dtype=float32 |
| `inputs/final_states` | tensor | shape=(1, 32, 128, 128), dtype=float32 |
| `inputs/prefix_lengths` | tensor | shape=(1,), dtype=int32 |
| `inputs/cu_seqlens` | tensor | shape=(2,), dtype=int32 |
| `inputs/block_map` | tensor | shape=(1, 1), dtype=int32 |
| `inputs/ssm_states` | tensor | shape=(293, 32, 128, 128), dtype=bfloat16, contiguous |
| `inputs/seq_size_per_block` | int | 1024 |
| `inputs/chunk_size` | int | 64 |
| `inplace_outputs/ssm_states` | tensor | shape=(293, 32, 128, 128), dtype=bfloat16 |
| `outputs` | None | 无非 inplace 输出 |

### 核心特点：inplace 操作

两个算子都是 **inplace 操作**——修改传入的 tensor（`initial_states` / `ssm_states`），不返回新 tensor。GPU dump 中 `outputs` 为 None，golden 数据在 `inplace_outputs` 中。

测试需 `.clone()` 输入 tensor 后运行 kernel，再与 `inplace_outputs` 比较。

## 2. 算子功能

### load_initial_state_from_block_map

从 paged `conv_states` 中根据 `block_map` 加载初始状态到 `initial_states`：

```
block_idx = block_map[batch][(prefix - 1) // seq_size_per_block]
initial_states[batch] = conv_states[block_idx]  (若 prefix > 0)
initial_states[batch] = 0                        (若 prefix == 0)
```

state 布局为 `(V, K)` 即 `(dv, dk)` — stride `(K, 1)`，dk 内层连续。

### store_ssm_state_to_block_map

将 `h`（per-chunk states）或 `final_states` 写回 paged `ssm_states`：

- **最后一个 chunk**：写 `final_states` → `ssm_states[block_map[batch][(prefix+input_len-1)//seq_size_per_block]]`
- **边界 chunk**（chunk > 0 且 (chunk+1)*chunk_size 是 seq_size_per_block 的倍数）：写 `h[chunk+1]` → 对应 block

## 3. 算子复用策略

### 3.1 直接复用 rtp-llm 的算子

**`load_initial_state_from_block_map`**：kernel 原样可复用，无需修改。通过 `importlib` 从 rtp-llm 加载，绕过 `rtp_llm.__init__` 的 C++ 依赖（`libth_transformer_config.so`）。

**`store_ssm_state_to_block_map` 的 Python wrapper**：`prepare_chunk_indices` 函数可复用。但 kernel 需要改写（见 3.2）。

### 3.2 不能直接复用的 kernel — store_ssm_state_to_block_map

GPU kernel 中 `source_ptr` 会被条件重赋值为 `final_states` 或 `h`（不同 base tensor）：

```python
# GPU 原始代码 — Triton for Ascend 不支持
source_ptr = final_states
if is_last_chunk:
    source_ptr = final_states + offset
elif is_boundary:
    source_ptr = h + offset  # 不同 base tensor
```

Triton for Ascend 报错 `Currently ptr type from different source not supported`。

**修复**：从两个 source 都加载数据，用 `tl.where` 选择：

```python
# 修复后 — 同时加载，条件选择
b_final = tl.load(p_final, boundary_check=(0, 1))
b_h = tl.load(p_h, boundary_check=(0, 1))
b_src = tl.where(is_last_chunk, b_final, b_h)
```

### 3.3 rtp-llm 依赖绕过

`rtp_llm/__init__.py` 会加载 `libth_transformer_config.so`（C++ 扩展），测试环境未编译。通过创建 stub 包 + `importlib` 加载特定 `.py` 文件绕过：

```python
# 创建 rtp_llm.models_py.triton_kernels.fla 的 stub 包
# 然后用 importlib 加载 utils.py → index.py → block.py
```

依赖链：`block.py → index.py(prepare_chunk_indices) → utils.py(tensor_cache)`。`tensor_cache` 是 LRU 缓存装饰器，不影响计算结果。

### 3.4 vllm-ascend 未提供等价算子

vllm-ascend 仓中不存在 `load_initial_state_from_block_map` 或 `store_ssm_state_to_block_map` 的实现。这两个算子是 rtp-llm 的 paged SSM state 管理专用，vllm-ascend 采用不同的 state 管理策略。

## 4. Triton for Ascend 适配

### 4.1 chunk_indices 设备

rtp-llm 的 `prepare_chunk_indices` 中 `torch.arange(n)` 默认创建 CPU tensor。需确保 `cu_seqlens` 在 NPU 上，使 `torch.arange` 继承 device：

```python
cu_seqlens = inputs["cu_seqlens"].to(torch.int64).npu()
```

### 4.2 contiguous 要求

所有输入 tensor 需 `.contiguous()` 后传入 Triton kernel。GPU dump 中 `conv_states`/`ssm_states` 已是 contiguous，`.contiguous()` 是防御性调用。

### 4.3 cu_seqlens dtype

GPU dump 中 `cu_seqlens` 是 int32，需转 int64：

```python
cu_seqlens = inputs["cu_seqlens"].to(torch.int64).npu()
```

## 5. 常见错误

### 5.1 inplace tensor 未 clone

两个算子都是 inplace 操作。不 clone 会导致原始输入被修改，无法重复测试：

```python
initial_states = inputs["initial_states"].npu().contiguous().clone()
```

### 5.2 指针条件重赋值（Triton for Ascend）

Triton for Ascend 不支持将指针变量重赋值为不同 base tensor。需用 `tl.where` 在数据层面选择。

### 5.3 rtp_llm.__init__ C++ 依赖

直接 `import rtp_llm` 会触发 `libth_transformer_config.so` 加载失败。需用 `importlib` + stub 包绕过。

### 5.4 cu_seqlens dtype 为 int32

GPU dump 中 `cu_seqlens` 是 int32，`prepare_chunk_indices` 需要 int64。

## 6. 测试模板

完整测试文件位于 `rtp-llm/example/triton_npu/test_npu_block_ops_gpu_golden.py`。

```python
# Bootstrap: load rtp-llm fla modules without triggering rtp_llm.__init__
_block_mod = _load_rtp_llm_fla_modules()

# Directly reuse load_initial_state_from_block_map
load_initial_state_from_block_map = _block_mod.load_initial_state_from_block_map

# store_ssm_state_to_block_map: use adapted kernel with tl.where fix
def store_ssm_state_to_block_map(h, final_states, ...):
    from rtp_llm.models_py.triton_kernels.fla.index import prepare_chunk_indices
    chunk_indices = prepare_chunk_indices(cu_seqlens, chunk_size)
    _store_ssm_state_to_block_map_kernel_ascend[grid](...)

class TestBlockOpsGpuGolden(unittest.TestCase):
    def test_load_initial_state_from_block_map(self):
        # Clone inplace tensor
        initial_states = inputs["initial_states"].npu().contiguous().clone()
        load_initial_state_from_block_map(...)  # directly reused from rtp-llm
        self.assertTensorClose(initial_states, data["inplace_outputs"]["initial_states"])

    def test_store_ssm_state_to_block_map(self):
        # Clone inplace tensor
        ssm_states = inputs["ssm_states"].npu().contiguous().clone()
        store_ssm_state_to_block_map(...)  # adapted kernel
        self.assertTensorClose(ssm_states, data["inplace_outputs"]["ssm_states"])
```

## 7. 调试技巧

1. **inplace 操作需 clone**：两个算子都修改传入 tensor，测试前必须 `.clone()`
2. **指针不能条件重赋值**：Triton for Ascend 不支持，用 `tl.where` 在数据层面选择
3. **rtp_llm C++ 依赖绕过**：用 `importlib` + stub 包加载特定 .py 文件
4. **state 布局 (V, K)**：stride `(K, 1)`，dk 内层连续
5. **h 和 final_states 必须 float32**：store_ssm_state_to_block_map 有 assert 检查
6. **cu_seqlens 转 int64**：GPU dump 是 int32
7. **conv_states/ssm_states 已 contiguous**：GPU dump 中 stride=(524288, 16384, 128, 1) 是标准连续布局

## 8. GPU 与 NPU 布局策略对比

| 维度 | GPU (rtp-llm) | NPU (Triton for Ascend) |
|------|---------------|------------------------|
| load_initial_state 实现 | rtp-llm Triton kernel | **直接复用 rtp-llm kernel**（原样） |
| store_ssm_state 实现 | rtp-llm Triton kernel | **kernel 改写**（指针→tl.where），wrapper 复用 |
| state 布局 | `(V, K)` stride `(K, 1)` | 一致，无需转换 |
| inplace 操作 | 是 | 是（需 clone 后运行） |
| 指针条件重赋值 | 支持 | **不支持**，需用 `tl.where` |
| prepare_chunk_indices | rtp-llm 内部 | **复用 rtp-llm**（通过 importlib） |
| cu_seqlens dtype | int32 | 需转 int64 |
| h/final_states dtype | float32 | 一致 |
| conv_states/ssm_states dtype | bfloat16 | 一致 |
| conv_states/ssm_states contiguous | 是 | 是（防御性 .contiguous()） |

**结论**：两个 block ops 是 paged SSM state 管理的辅助算子。`load_initial_state_from_block_map` 可直接复用 rtp-llm 的 kernel；`store_ssm_state_to_block_map` 需改写 kernel（指针条件重赋值→`tl.where`），但 Python wrapper 中的 `prepare_chunk_indices` 可复用。vllm-ascend 未提供等价算子。核心适配点：rtp_llm C++ 依赖绕过（importlib + stub 包）、指针重赋值改写、inplace 操作需 clone。
