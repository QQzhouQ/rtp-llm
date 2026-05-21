# rtp-llm Ascend NPU 适配修复汇总

## 概述

本文档汇总了为在华为 Ascend NPU 上运行 rtp-llm（Qwen3-0.6B 模型）所做的所有代码修复。

原始错误：`AttributeError: 'EmptyClass' object has no attribute 'init'`

## 修改文件清单

| # | 文件路径 | 修改类型 | 解决的问题 |
|---|---------|---------|-----------|
| 1 | `rtp_llm/ops/__init__.py` | Python | C++ 扩展模块加载失败，RtpLLMOp 降级为 EmptyClass |
| 2 | `rtp_llm/ops/compute_ops.py` | Python | stub .so 加载方式导致 double free |
| 3 | `rtp_llm/config/server_config_setup.py` | Python | Ascend 设备未自动设置 separate_kv_cache=True |
| 4 | `rtp_llm/cpp/cache/MemoryLayoutStrategy.cc` | C++ | NPU 张量不支持 reshape+select 操作；separate_kv_cache 模式下 stride 计算错误 |
| 5 | `rtp_llm/cpp/cache/MemoryLayoutConfig.h` | C++ | 缺少 k_pool_size_bytes 和 v_pool_size_bytes 字段 |
| 6 | `rtp_llm/cpp/cache/BlockPoolConfigHelper.h` | C++ | separate_kv_cache 模式下 kv_block_pool_size_bytes 计算错误 |
| 7 | `rtp_llm/cpp/cache/BlockPool.cc` | C++ | separate_kv_cache 模式下 K/V 缓冲区大小分配错误 |
| 8 | `start_npu_server.sh` | Shell | LD_LIBRARY_PATH 缺少 torch/torch_npu 库路径 |

## 详细修改说明

### 1. rtp_llm/ops/__init__.py

**问题**：`libth_transformer.so` 是用 CUDA 配置编译的，在 Ascend 环境（torch 2.9.0+cpu）中加载失败，导致 `RtpLLMOp` 被降级为 `EmptyClass`。

**修复**：
- 添加 `import torch_npu`（在 `import torch` 之后），确保 NPU 设备后端在加载 .so 之前初始化
- 在加载 `libth_transformer_config.so` 之前，预加载 `librtp_compute_ops_symbols.so`（提供缺失的 C++ 符号），使用 `RTLD_GLOBAL | RTLD_NOW | RTLD_NODELETE` 模式

### 2. rtp_llm/ops/compute_ops.py

**问题**：原始 `librtp_compute_ops_stub.so` 与 `librtp_compute_ops.so` 存在全局对象冲突，导致 double free。

**修复**：
- 将 stub .so 替换为轻量级的 `librtp_compute_ops_symbols.so`（仅提供缺失的 C++ 符号，不包含 pybind11 模块）
- 加载模式改为 `RTLD_GLOBAL | RTLD_NOW | RTLD_NODELETE`，防止 .so 被卸载导致 double free

### 3. rtp_llm/config/server_config_setup.py

**问题**：Ascend NPU 上 `separate_kv_cache` 默认为 `false`，但 NPU 的注意力算子（`torch_npu._npu_paged_attention`、`torch_npu.npu_fused_infer_attention_score`）要求 K/V cache 分离存储。未设置时导致 `unknown format type` 错误。

**修复**：
- 在 `setup_default_args()` 函数中，检测 `torch_npu` 和 `torch.npu.is_available()`，自动设置 `separate_kv_cache=True`

```python
try:
    import torch_npu
    if torch.npu.is_available() and not py_env_configs.kv_cache_config.separate_kv_cache:
        py_env_configs.kv_cache_config.separate_kv_cache = True
        logging.info("set separate_kv_cache=True by default on Ascend NPU")
except ImportError:
    pass
```

### 4. rtp_llm/cpp/cache/MemoryLayoutStrategy.cc

**问题1**：`separate_kv_cache` 模式下，`kv_block_stride_elems` 仍使用 K+V 合在一起的 stride，导致偏移量超出 K buffer 范围。

**修复1**：在 `processKVTensor()` 中，当 `separate_kv_cache_` 为 true 时，使用 `k_block_stride_bytes` 代替 `kv_block_stride_bytes`。

**问题2**：NPU 上的张量使用 ACL 内部格式，`torch::from_blob` + `reshape` + `select` 操作触发 `torch_npu` 的 `as_strided`，而 `torch_npu` 不支持对 NPU 格式张量执行此操作，导致 `unknown format type` 错误。

**修复2**：在 `separate_kv_cache` 模式下，使用 `torch::from_blob` 直接为每层创建独立的 2D 张量视图（`[block_num, stride]`），避免 `reshape` 和 `select` 操作。

同样修改了 `processScaleTensor()`，在 `separate_kv_cache` 模式下使用相同的直接创建方式。

### 5. rtp_llm/cpp/cache/MemoryLayoutConfig.h

**问题**：缺少 `k_pool_size_bytes` 和 `v_pool_size_bytes` 字段，无法在 `separate_kv_cache` 模式下分别记录 K 和 V 的池大小。

**修复**：添加两个字段：
```cpp
size_t k_pool_size_bytes = 0;
size_t v_pool_size_bytes = 0;
```

### 6. rtp_llm/cpp/cache/BlockPoolConfigHelper.h

**问题**：`kv_block_pool_size_bytes` 始终使用 `kv_block_stride_bytes`（K+V 合在一起的 stride）计算，在 `separate_kv_cache` 模式下，K buffer 只有 K 的数据，导致偏移量超出范围（`layout[0] kv tensor out of range`）。

**修复**：在 `separate_kv_cache` 模式下：
- `kv_block_pool_size_bytes` 改为使用 `k_block_stride_bytes`（仅 K 的 stride）
- 新增 `k_pool_size_bytes` 和 `v_pool_size_bytes` 的计算

```cpp
if (cache_config.separate_kv_cache) {
    cfg.k_pool_size_bytes =
        static_cast<size_t>(layer_num) * static_cast<size_t>(cfg.block_num) * cfg.k_block_stride_bytes;
    cfg.v_pool_size_bytes =
        static_cast<size_t>(layer_num) * static_cast<size_t>(cfg.block_num) * cfg.v_block_stride_bytes;
    cfg.kv_block_pool_size_bytes = cfg.k_pool_size_bytes;
}
```

### 7. rtp_llm/cpp/cache/BlockPool.cc

**问题**：`initializeCacheBuffer()` 中，`separate_kv_cache` 模式下简单地将 `total_size_bytes` 除以 2 分配给 K 和 V，但 `total_size_bytes` 包含了 K+V+scale 的大小，对半切分后 K buffer 不够用。

**修复**：使用 `k_pool_size_bytes + kv_scale_pool_size_bytes` 作为 K buffer 大小，`v_pool_size_bytes` 作为 V buffer 大小：

```cpp
if (separate_kv_cache_) {
    size_t k_buffer_bytes = 0;
    size_t v_buffer_bytes = 0;
    for (const auto& layout_cfg : config_.memory_layouts) {
        k_buffer_bytes += layout_cfg.k_pool_size_bytes + layout_cfg.kv_scale_pool_size_bytes;
        v_buffer_bytes += layout_cfg.v_pool_size_bytes;
    }
    if (k_buffer_bytes == 0) {
        k_buffer_bytes = config_.total_size_bytes / 2;
    }
    if (v_buffer_bytes == 0) {
        v_buffer_bytes = config_.total_size_bytes / 2;
    }
    k_cache_buffer_ = torch::empty({static_cast<int64_t>(k_buffer_bytes)}, options);
    v_cache_buffer_ = torch::empty({static_cast<int64_t>(v_buffer_bytes)}, options);
```

### 8. start_npu_server.sh

**问题**：`LD_LIBRARY_PATH` 缺少 `torch.libs`、`torch/lib`、`torch_npu/lib` 路径，导致动态链接器找不到 `libarm_compute-4dd6d3ec.so`、`libtorch_npu.so` 等依赖库。

**修复**：动态获取 torch 和 torch_npu 的库路径并添加到 `LD_LIBRARY_PATH`：

```bash
TORCH_LIBS="$(python -c 'import torch; import os; print(os.path.join(torch.__path__[0], "libs"))' 2>/dev/null)"
TORCH_LIB="$(python -c 'import torch; import os; print(os.path.join(torch.__path__[0], "lib"))' 2>/dev/null)"
TORCH_NPU_LIB="$(python -c 'import torch_npu; import os; print(os.path.join(torch_npu.__path__[0], "lib"))' 2>/dev/null)"
export LD_LIBRARY_PATH=${TORCH_LIBS}:${TORCH_LIB}:${TORCH_NPU_LIB}:/usr/local/Ascend/driver/lib64:/usr/local/Ascend/driver/lib64/driver:$LD_LIBRARY_PATH
```

## 额外需要的步骤（非代码修改）

1. **重新编译 C++ 扩展**：必须使用 `--config=ascend` 重新编译 `libth_transformer.so`、`librtp_compute_ops.so`、`libth_transformer_config.so`
2. **创建符号库**：编译 `librtp_compute_ops_symbols.so`，提供 `rtp_llm::sampleGreedy` 等缺失符号
3. **复制 Ascend Python 模块**：将 `models_py/modules/factory/` 和 `models_py/modules/base/` 下的 `ascend` 目录复制到 wheel 安装路径

## 已知遗留问题

推理请求时出现 `double free detected in tcache 2` 崩溃，原因是 `libth_transformer.so` 静态链接了 OpenSSL、cURL 等库，与 Python 进程中的同名库产生全局对象冲突。需要修改 Bazel BUILD 配置，将这些库改为动态链接。
