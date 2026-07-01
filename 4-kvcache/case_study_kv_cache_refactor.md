# 案例总结：KV Cache 适配与 Attention 接口重构

从「C++ 层 K/V 拆分」到「不拆分，靠 fused_infer_attention_score 吃非连续 KV」。

> 分支对照：`main`（重构前 / 最初适配方案）→ `feature/ascend-fia-discontinuous`（重构后 / 最终落地方案）
>
> 关联文件：`MemoryLayoutStrategy.cc`、`BlockPool.h`、`BufferTypes.h`、`CacheConfig.h`、`ascend_impl/ascend_prefill.py`、`ascend_impl/ascend_decode.py`

---

## 一、背景与问题来源

在把昇腾 NPU 接入 rtp-llm 框架时，kv_cache 的存储形态和 attention 算子的接口要求对不上：

1. **rtp-llm 原生形态**：KV 合并存储。`BlockPool` 申请一整块 buffer，`MemoryLayoutStrategy` 把它按层 narrow+view 成每层一个 `[block_num, kv_block_stride_elems]` 的 2D tensor（K 与 V 在同一块连续显存里拼接），再在 Python 侧经 `getLayerCache()` reshape 成 `[kernel_blocks, 2, kv_heads, kernel_seq, head_dim]`（dim=1 区分 K/V）。GPU（FlashInfer）和 AMD（AITer）都直接吃这个合并形态。
2. **昇腾 torch_npu 算子的接口要求**：`_npu_reshape_and_cache`、`_npu_paged_attention`、`npu_fused_infer_attention_score` 在参数表里把 `key_cache` 和 `value_cache` 当作**两个独立的 Tensor** 传入。

于是问题变成：**要不要在 C++ 层把合并的 KV buffer 物理拆成两块独立的 k_cache / v_cache？**

围绕这个问题，适配经历了两个阶段（重构前、重构后），下面分开讲。

---

## 二、重构前：C++ 层「K/V 拆分」方案

### 2.1 方案思路

最初（设计文档 v0.1～v0.3 的「目标」路径）倾向于「让框架层去贴合算子的接口要求」，即在 C++ 层做物理分离存储（`separate_kv_cache`）：

- `BlockPool` 不再申请一整块合并 buffer，而是申请 `k_cache_buffer_` 与 `v_cache_buffer_` 两块各占一半的独立 buffer；
- `MemoryLayoutStrategy` 维护两套 per-layer tensor（`layer_k_tensors_` / `layer_v_tensors_`），分别 narrow+view；
- `BufferTypes` / `CacheLayerLayout` 新增 `layers_to_k_buffer_ptrs`、`layers_to_v_buffer_ptrs` 字段；
- `CacheConfig` 新增 `separate_kv_cache`、`k_block_stride_bytes`、`v_block_stride_bytes` 等开关与步长；
- layout 从 GPU/AMD 用的 **HND** 切到 NPU 要求的 **NHD**。

到 Python ascend_impl 层，`k_cache_base` / `v_cache_base` 直接是两个独立 NHD tensor，喂给 torch_npu 算子时无需再 slice。

### 2.2 这个方案的问题

- **C++ 改动面太大**：`BlockPool`、`MemoryLayoutStrategy`、`BufferTypes`、`CacheConfig` 都要加 NPU 分支，等于给框架核心数据结构开第二条存储路径。
- **跟 GPU / AMD 路径分叉了**：原来三端一致的「合并 KV」逻辑被打破，NPU 变成需要单独维护的特例。框架后续对 cache 管理的任何改动（block pool、TP 切分、scale、MLA、hybrid attention）都得在 NPU 分支里再抄一遍。
- **跟随上游很累**：rtp-llm 主线一直在迭代 cache 管理，C++ 里的 NPU 特化分支会持续产生合并冲突。
- **拆分本身没带来收益**：费这么大劲只是为了满足「算子要两个独立 tensor」这个表象，性能和功能上都没捞到好处。

说白了，为了贴算子接口去动框架最核心的 cache 数据结构，不划算。

---

## 三、重构后：基于 `fused_infer_attention_score` 的「不拆分」方案

### 3.1 转折点：新算子能吃「非连续 KV cache」

当前分支引入了 `npu_fused_infer_attention_score`（FIA）。和早期只收「两块连续独立 tensor」的 paged attention 算子不一样，FIA 原生支持：

- **block_table 分页访问**：K/V 不用是一整块连续内存，靠 block_table 按页找就行；
- **非连续 view 的 key/value**：哪怕 Python 层传进来的 k_cache / v_cache 只是从合并 buffer 上 slice 出来的非连续 view，算子内部也能正确寻址。

「算子要两个独立且连续的物理 tensor」这个前提**不成立了**，C++ 层的物理拆分也就没有存在的必要。

### 3.2 落地方案

**C++ 层：零改动，完全复用 GPU/AMD 的合并存储路径。**

`MemoryLayoutStrategy.cc` 还是维护那块合并 buffer：

- 非 MLA：每层 view 成 `[block_num, kv_block_stride_elems]`，K/V 拼在同一段显存里；
- 只有**非对称 TP**（`createPartitionedBlockInfo`）场景下，才会用 `createPartitionedSubBlocks` 按指针偏移 `k_off` / `v_off` 取出 K/V 子块。这步是纯指针运算，不动顶层存储结构，也不是 NPU 专属逻辑。

`BlockPool`、`BufferTypes`、`CacheConfig` 都不用引入 `separate_kv_cache` 之类的 NPU 分支。

**Python ascend_impl 层：用零拷贝 view 取 K/V，再喂给 FIA。**

```python
# ascend_prefill.py:138-163  /  ascend_decode.py:145-174
def forward(self, q, kv_cache):
    k_cache = kv_cache.kv_cache_base[:, 0]   # dim=1 取 K，view 不拷贝
    v_cache = kv_cache.kv_cache_base[:, 1]   # dim=1 取 V，view 不拷贝
    attn_output, _ = torch_npu.npu_fused_infer_attention_score(
        query=q, key=k_cache, value=v_cache,
        block_table=block_table,
        input_layout="TND",
        block_size=self.page_size,
        actual_seq_lengths=actual_seq_q,
        actual_seq_lengths_kv=actual_seq_kv,
        num_key_value_heads=self.num_kv_heads,
        num_heads=self.num_heads,
        scale=self.scale,
        sparse_mode=3,
    )
    return attn_output
```

`kv_cache_base[:, 0]` / `[:, 1]` 只是 view（stride 不变、不拷显存），FIA 内部靠 block_table 分页寻址，能正确处理这种非连续输入。Prefill 和 Decode 都走 FIA，不再依赖有已知 aicore 问题的 `_npu_paged_attention`。

### 3.3 拿到了什么

- **C++ 层零改动**：`BlockPool` / `MemoryLayoutStrategy` / `BufferTypes` / `CacheConfig` 全部复用主线，NPU 适配不侵入框架核心。
- **跟 GPU / AMD 一致**：三端都是「合并 KV + 分页 attention」，NPU 不再是特例，维护时心智负担小很多。
- **适配面收敛**：NPU 适配基本集中在 `ascend_impl/` 下少数几个 Python 文件（`ascend_prefill.py`、`ascend_decode.py`、`ascend_kv_cache_write_op.py`、`ascend_rope*.py`），框架侧改动很少。
- **跟随上游轻松**：cache 管理后续演进，NPU 路径几乎零成本继承。
- **稳定性更好**：统一走经过验证的 FIA，躲开了旧 paged attention 算子在部分 block_table 配置下的 aicore 错误。

---

## 四、重构前后对比

| 维度 | 重构前（K/V 拆分） | 重构后（不拆分 + FIA） |
| --- | --- | --- |
| C++ 改动 | 大（BlockPool/MemoryLayout/BufferTypes/CacheConfig 加 NPU 分支） | 零（完全复用主线合并存储） |
| KV 物理布局 | 两块独立 buffer | 单块合并 buffer，view 取 K/V |
| layout | 需切到 NHD | 维持 HND（FIA 用 `input_layout="TND"` 适配） |
| 与 GPU/AMD 一致性 | 分叉，NPU 成特例 | 高度一致 |
| 适配面 | C++ + Python 双层侵入 | 收敛到少数 Python 文件 |
| 跟随上游成本 | 高（持续合并冲突） | 低 |
| 前置依赖 | 无（硬贴合算子接口） | 需要 FIA 支持非连续/分页 KV |
| Decode 算子 | `_npu_paged_attention`（有已知问题） | `npu_fused_infer_attention_score`（稳定） |

---

## 五、经验沉淀

1. **先摸算子能力，再决定动不动框架核心。** 早期方案栽在「让框架去贴算子接口」，于是重构了 cache 数据结构。等算子升级到能吃非连续、分页输入，框架层的拆分就成了纯负担。顺序应该是：算子能力评估 → 找最小侵入点 → 再决定要不要动 C++。反过来做容易白干。

2. **能用 view 解决的，别上物理拆分。** `kv_cache_base[:, 0]` 这种零拷贝 view 既满足算子「两个 tensor 入参」的接口契约，又不改底层存储，是侵入性最低的适配方式。前提是算子能容忍 view 带来的非连续 stride——这是 FIA 相比旧 paged attention 的进步所在。

3. **多端一致比单端最优更重要。** GPU、AMD、NPU 三端共享同一套「合并 KV + 分页 attention」语义，意味着任何一端对 cache 管理的改进都能自然惠及他端。一旦给某端开物理特化分支，等于把这一端从主线演进里隔离出去。NPU 适配能收敛到几个 Python 文件，靠的就是没破坏这个一致性。

4. **框架侧「不动」本身就是结果。** 这次重构值得记的不是写了多少新代码，而是省下了多少本该写的 C++ 代码——把适配压力从框架核心挪到了算子能力 + 薄薄一层 Python wrapper 上。

---

## 六、结论

这次重构做的事其实很简单：拿「算子支持非连续 KV」换「框架层零侵入」。`npu_fused_infer_attention_score` 能吃 block_table 分页、非连续输入，所以 rtp-llm 在 NPU 上不用像最初设计那样在 C++ 层物理拆分 k_cache / v_cache，而是跟 GPU / AMD 保持一致的合并存储，只在 Python 层做 view 适配。最后 NPU 对框架侧的修改被压到了极小范围，C++ 核心一行没动。
