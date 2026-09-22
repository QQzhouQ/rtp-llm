# Qwen3.5 Ascend MoE CANN 接入开发计划

## 1. 文档目的

本文档用于指导 AI 在 RTP-LLM 中完成 Qwen3.5 MoE 层的 Ascend 适配。它结合了：

- `rtp-llm-ascend-adaption-plan` 新增的《Qwen3.5 MoE 层整体迁移方案》；
- 新增的 MoE 算子说明和路由算子迁移建议；
- 当前 `qwen35-ascend-ops` 分支的真实代码结构；
- 当前 Ascend 依赖中固定的 `torch_npu==2.9.0.post3` 接口；
- 已完成的 Qwen3.5 线性注意力 Ascend 适配边界。

本文档是一份开发任务书，不代表相关能力已经实现或完成真机验收。

## 2. 结论与首期目标

新增指导的核心内容是：在已经接入 Qwen3.5 线性注意力算子的基础上，继续迁移
Qwen3.5 的 MoE 层。模型拓扑继续复用 `GenericMoeLayer`，仅替换 Ascend 下的
路由选择、token 排序、专家 GEMM、SwiGLU 和 token 合并实现。

首期交付范围限定为：

- 数据类型：BF16；
- 并行模式：单卡，以及 `ep_size=1、dp_size=1` 的纯 TP；
- 执行模式：eager；
- 路由：Qwen3.5 使用的 softmax top-k，并支持 `norm_topk_prob=True`；
- 专家计算：非量化 CANN MoE 流水；
- Shared Expert：保留现有 RTP-LLM 路径；
- 不支持的配置必须在初始化阶段明确报错，不能静默回退到 Triton 或 CPU。

首期不包含：

- Expert Parallelism（EP）通信；
- FP8、INT8、INT4 等量化 MoE；
- DeepSeek/Kimi 使用的 GroupTopK、correction bias；
- `fake_balance_expert`；
- ACLGraph、NPU Graph 和 MTP 的支持声明；
- MoE 性能调优和算子融合的扩展版本。

## 3. 当前代码现状与缺口

### 3.1 可以保留的框架

`rtp_llm/models_py/model_desc/generic_moe.py` 已经提供完整扩展点：

1. `LinearFactory` 计算 router logits；
2. `SelectTopk` 生成 `topk_weights/topk_ids`；
3. `FusedMoeFactory` 根据设备注册 Router 和 Executor；
4. `FusedMoe` 依次调用 `router.prepare`、`executor.execute`、
   `router.finalize`；
5. Shared Expert 及其 gate 在 MoE 输出后融合。

因此，原则上不修改 `GenericMoeLayer`，也不修改 Qwen3.5 模型拓扑文件
`model_desc/qwen3_next.py`。只有发现共享接口确实无法表达 CANN 契约时，才允许做
小范围、后端无关的接口扩展。

### 3.2 当前 Ascend 路由选择不是目标实现

`rtp_llm/models_py/modules/base/ascend/select_topk.py` 当前使用：

```python
torch.topk(router_logits_fp32)
torch.softmax(topk_logits)
```

它没有调用 CANN MoE gating 算子，并且只在“softmax 后选 top-k 并重新归一化”这一
特定语义下等价，不能完整表达 `has_moe_norm`。

目标实现应使用：

```python
torch_npu.npu_moe_gating_top_k(..., norm_type=0)
```

或在锁定 wheel 上验证等价的
`torch_npu.npu_moe_gating_top_k_softmax`。优先采用新增指导指定的
`npu_moe_gating_top_k`。

### 3.3 当前所谓 Ascend fallback 实际依赖 Triton

`impl/ascend/strategy/pytorch_fallback.py` 的注释称其为纯 PyTorch fallback，
但实际注册了：

- `BatchedDataRouter`；
- `BatchedTritonExperts`。

`BatchedTritonExperts` 会导入 Triton kernel，而 Ascend 构建明确排除了 Triton。
因此它不能作为可用 fallback。

本轮应实现真正的 CANN Strategy。对于 CANN Strategy 不支持的配置，首选行为是
明确报错。若确实需要保留 fallback，则必须先实现一个不含 Triton 的真实
PyTorch/torch_npu fallback，不能继续沿用当前名称和错误描述。

### 3.4 构建依赖需要补齐

当前 Ascend MoE 目录已有嵌套 `BUILD` 文件。Bazel 的 `glob` 不会跨越子包边界，
因此新增 Router、Executor、Strategy 和测试时，必须分别声明 `srcs`、`deps` 和
`data`，不能依赖父目录递归收集。

Ascend target 必须只在 `@//:using_ascend` 分支引入，且不得新增对
`//rtp_llm/models_py/triton_kernels` 的依赖。

## 4. 算子替换关系

| 阶段 | 当前 CUDA/Triton 语义 | Ascend 目标算子 | 首期状态 |
|---|---|---|---|
| Router Linear | `LinearFactory` | 保留现有实现 | 保留 |
| Softmax Top-K | `topkGatingSoftmaxKernelLauncher` | `npu_moe_gating_top_k` | 本轮开发 |
| Token 排序 | `moe_align_block_size_torch` 等 | `npu_moe_init_routing_v2` | 本轮开发 |
| GEMM1 | Triton fused/grouped GEMM | `npu_grouped_matmul` | 本轮开发 |
| SwiGLU | `silu_and_mul` | `npu_swiglu` | 本轮开发 |
| GEMM2 | Triton fused/grouped GEMM | `npu_grouped_matmul` | 本轮开发 |
| 权重乘与 Token 合并 | Triton epilogue/reduce | `npu_moe_token_unpermute` | 本轮开发 |
| TP 汇总 | `all_reduce(Group.TP)` | 保留现有实现 | 本轮接线 |
| EP 分发/合并 | 后端专用通信 | `npu_moe_distribute_dispatch_v2/combine_v2` | 后续阶段 |
| Shared Expert | `DenseMLP` + gate | 保留现有实现 | 保留 |
| GroupTopK | CUDA op | 暂无首期实现 | 明确拒绝 |
| Fake Balance | CUDA op | 暂无首期实现 | 明确拒绝 |

## 5. 必须修正的指导文档伪代码

新增指导给出了总体方向，但不能直接复制其中的伪代码。开发前必须处理以下差异。

### 5.1 CANN Python API 的真实返回值

以当前 Ascend requirements 固定的 torch_npu 版本为准，先在目标环境核对符号。
已核对的 op-plugin 接口为：

```text
npu_moe_gating_top_k(...) -> (Tensor, Tensor, Tensor)
npu_moe_init_routing_v2(...) -> (Tensor, Tensor, Tensor, Tensor)
npu_grouped_matmul(Tensor[] x, Tensor[] weight, ...) -> Tensor[]
npu_swiglu(Tensor, dim=-1) -> Tensor
npu_moe_token_unpermute(Tensor, Tensor, probs=None, ...) -> Tensor
```

因此：

- gating 不能只接两个返回值；
- init-routing 不能只接三个返回值；
- grouped-matmul 的 `x` 和 `weight` 是列表，返回值也是列表；
- `split_item=2/3` 时虽然只有一个输出，它仍位于返回列表中，需取 `[0]`。

目标机预检命令：

```bash
python - <<'PY'
import torch
import torch_npu

required = (
    "npu_moe_gating_top_k",
    "npu_moe_init_routing_v2",
    "npu_grouped_matmul",
    "npu_swiglu",
    "npu_moe_token_unpermute",
)
missing = [name for name in required if not hasattr(torch_npu, name)]
assert not missing, f"missing torch_npu operators: {missing}"
assert torch.npu.is_available(), "NPU is not available"
print(torch_npu.__version__)
PY
```

如果目标 wheel 的签名与上述契约不同，停止开发并记录版本、函数签名和报错，不得
通过捕获异常后静默降级来掩盖问题。

### 5.2 Qwen3.5 W1 的 gate/up 顺序不能只做 transpose

RTP-LLM 中 Qwen3.5 的内部权重布局是：

```text
W.moe_w1: [E, 2I, H]，通道顺序为 [up, gate]
W.moe_w2: [E, H, I]
```

依据：

- 普通 checkpoint 加载时先读取 `up_proj`，再读取 `gate_proj`；
- stacked checkpoint 的 `transpose_stack_moe_w1` 会把 `[gate, up]` 转成
  `[up, gate]`；
- 现有 Triton `silu_and_mul` 使用后半段作 gate，计算
  `silu(gate) * up`。

而 `torch_npu.npu_swiglu` 会把输入前半段作为 A，计算
`silu(A) * B`。所以若仅做：

```python
w1 = weights[W.moe_w1].transpose(1, 2)
```

最终会错误地计算 `silu(up) * gate`。

正确目标布局应为：

```text
[E, H, 2I]，输出通道顺序为 [gate, up]
```

功能验证阶段可用下列逻辑表达语义：

```python
up, gate = weights[W.moe_w1].chunk(2, dim=1)
w1_cann = torch.cat([gate, up], dim=1).transpose(1, 2).contiguous()
```

但生产实现不能在每次 forward 中重排权重，也不能长期无评估地保留一份完整重复
权重。优先顺序为：

1. 在 Ascend 权重加载阶段一次性生成 CANN 布局；
2. 若首期只能在 Executor 初始化时转换，必须测量峰值显存，并在模型启动完成后确认
   不再保留无用副本；
3. 不允许每 token 或每层 forward 做 `cat/transpose/contiguous`。

必须用非对称 gate/up 数据编写测试，避免随机数据或相同数据掩盖顺序错误。

### 5.3 Top-K 权重需要显式处理 renormalize

Qwen3.5 配置默认：

```python
config.has_moe_norm = config_json.get("norm_topk_prob", True)
```

`npu_moe_gating_top_k(norm_type=0)` 先对全部专家做 softmax，再选 top-k；当前接口的
`renorm` 参数只支持 `0`。当 `has_moe_norm=True` 时，还需要对选中的权重做：

```python
topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
```

注意数值稳定性和零分母保护。`has_moe_norm=False` 时不能执行这一步。输出必须复制到
调用方预分配的 `topk_weights`，expert id 必须复制到预分配的 `topk_ids`。

### 5.4 init-routing 与 unpermute 的索引方向必须成对验证

首期拟采用：

```text
quant_mode=-1
drop_pad_mode=0
expert_tokens_num_flag=True
expert_tokens_num_type=0
row_idx_type=1
```

其中 `expert_tokens_num_type=0` 使 `group_list` 表示各 expert token 数的累积和，匹配
`npu_grouped_matmul(group_list_type=0)`。

`row_idx_type=1` 产生 sorted-to-original 的 scatter 索引，预期可直接传给
`npu_moe_token_unpermute`。这项不能只通过 shape 判断，必须使用手工构造的 expert id
和可辨识 token 值做 golden 测试，确认最终 token 顺序和 top-k 权重都正确。

### 5.5 topk id 与 group_list dtype

- `npu_moe_gating_top_k` 输出 expert id 为 `int32`；
- `npu_moe_init_routing_v2` 要求 expert id 为 `int32`；
- `NpuFusedExpertsExecutor.topk_ids_dtype` 应返回 `torch.int32`；
- `npu_grouped_matmul` 的 Tensor `group_list` 使用 `int64`；
- 不允许通过 CPU `.tolist()` 构造生产路径 group list。

## 6. 目标架构与数据流

```text
GenericMoeLayer.forward
  ├─ gate(hidden_states) -> router_logits [M, E]
  ├─ Ascend SelectTopk
  │    └─ npu_moe_gating_top_k -> weights [M, K], ids [M, K]
  └─ FusedMoe
       ├─ NpuPureTpRouter.prepare
       │    └─ 原样传递 hidden_states、weights、ids
       ├─ NpuFusedExpertsExecutor.execute
       │    ├─ npu_moe_init_routing_v2
       │    ├─ npu_grouped_matmul (GEMM1)
       │    ├─ npu_swiglu
       │    ├─ npu_grouped_matmul (GEMM2)
       │    └─ npu_moe_token_unpermute
       └─ NpuPureTpRouter.finalize
            └─ tp_size > 1 时 all_reduce(Group.TP)
```

形状约定：

| 名称 | 形状 | dtype |
|---|---|---|
| `hidden_states` | `[M, H]` | BF16 |
| `router_logits_fp32` | `[M, E]` | FP32 |
| `topk_weights` | `[M, K]` | FP32 |
| `topk_ids` | `[M, K]` | INT32 |
| `sorted_x` | `[M*K, H]` | BF16 |
| `group_list` | `[E]` | INT64 |
| `gate_up` | `[M*K, 2I]` | BF16 |
| `activated` | `[M*K, I]` | BF16 |
| `expert_out` | `[M*K, H]` | BF16 |
| `output` | `[M, H]` | BF16 |

首期 grouped matmul 调用契约建议如下，最终以固定 wheel 的真机测试为准：

```python
gate_up = torch_npu.npu_grouped_matmul(
    [sorted_x],
    [w1_cann],
    group_list=group_list,
    split_item=3,
    group_type=0,
    group_list_type=0,
)[0]

activated = torch_npu.npu_swiglu(gate_up.contiguous(), dim=-1)

expert_out = torch_npu.npu_grouped_matmul(
    [activated],
    [w2_cann],
    group_list=group_list,
    split_item=3,
    group_type=0,
    group_list_type=0,
)[0]
```

这里采用“单 x、单 3D weight、单输出”的 GMM 模式。若固定 wheel 不接受该形式，
应先用最小复现确认其支持的输入组合，再调整为 expert weight 列表；不能凭猜测改变
`group_list` 语义。

## 7. 计划修改的文件

### 7.1 必改文件

1. `rtp_llm/models_py/modules/base/ascend/select_topk.py`
   - 接入 `npu_moe_gating_top_k`；
   - 正确接收三个返回值；
   - 支持 `has_moe_norm`；
   - 保持原有 in-place 输出接口。

2. `rtp_llm/models_py/modules/factory/fused_moe/impl/ascend/routers/npu_pure_tp_router.py`
   - 新增纯 TP Router；
   - `prepare` 不通信，只构造 `ExpertForwardPayload`；
   - `finalize` 在 `tp_size>1` 时执行 `all_reduce(Group.TP)`。

3. `rtp_llm/models_py/modules/factory/fused_moe/impl/ascend/executors/npu_fused_experts.py`
   - 新增 BF16 CANN Executor；
   - 实现 init-routing、两次 GMM、SwiGLU 和 unpermute；
   - 处理 W1 gate/up 顺序和权重布局；
   - 返回 `CombineForwardPayload`。

4. `rtp_llm/models_py/modules/factory/fused_moe/impl/ascend/strategy/cann.py`
   - 新增 `AscendCannStrategy`；
   - 仅匹配 BF16、非量化、纯 TP、eager 配置；
   - 使用 `NpuPureTpRouter` 和 `NpuFusedExpertsExecutor`。

5. `rtp_llm/models_py/modules/factory/fused_moe/impl/ascend/strategy/__init__.py`
   - 导出新 Strategy；
   - 清理不真实的 fallback 导出或将其改为明确的 unsupported 策略。

6. `rtp_llm/models_py/modules/factory/fused_moe/__init__.py`
   - 在 Ascend registry 中注册 `AscendCannStrategy`；
   - 不导入 CUDA/Triton Strategy；
   - 不为不支持配置静默选择 `BatchedTritonExperts`。

7. Ascend MoE 相关 `BUILD` 文件
   - 为 routers、executors、strategy 和 tests 建立明确 target；
   - 声明 `torch`、collective、fused_moe defs 等依赖；
   - 确保只由 `using_ascend` 分支引入；
   - 确保依赖图不含 RTP-LLM Triton kernel。

### 7.2 可能需要修改的文件

1. `rtp_llm/models/qwen3_next/qwen3_next_weight.py`
   - 仅当需要在加载阶段生成 Ascend CANN 权重布局时修改；
   - 必须保持 CUDA/ROCm 的 `[up, gate]` 语义不变；
   - 优先做后端专用 transform，避免共享逻辑产生分叉。

2. `rtp_llm/models_py/modules/factory/fused_moe/defs/type.py`
   - 当前已有 `RouterType.PURE_TP` 和 `ExecutorType.FUSED_MOE`，通常无需修改；
   - 只有优先级冲突被测试证明存在时，才新增 Ascend 专用类型。

3. `deps/requirements_ascend.txt`
   - 当前已包含目标 `torch_npu`，通常无需新增依赖；
   - 不得为了 MoE 引入 Triton。

### 7.3 原则上不修改的文件

- `rtp_llm/models_py/model_desc/generic_moe.py`；
- `rtp_llm/models_py/model_desc/qwen3_next.py`；
- CUDA、ROCm 的 Router、Executor、Strategy；
- 已完成的 `rtp_llm/models_py/kernels/ascend` 线性注意力实现。

## 8. 分阶段开发计划

### 阶段 0：环境、接口与基线确认

任务：

1. 记录代码分支、HEAD 和工作区状态；
2. 记录 CANN、PyTorch、torch_npu、SoC 型号；
3. 执行第 5.1 节的算子符号预检；
4. 在固定 wheel 上用最小输入确认五个算子的参数、返回数量、shape 和 dtype；
5. 保存一份 PyTorch 或 GPU reference，覆盖 gating 和完整 MoE 专家计算；
6. 确认 Qwen3.5 实际的 `E、K、H、I` 以及 TP 后本地权重 shape。

产出：

- 一份接口检查记录；
- 五个最小可运行 op probe；
- 不改生产代码的 reference 计算函数。

停止条件：

- 缺少任何必要 CANN 算子；
- pinned wheel 与文档签名不同且无法从官方说明确认；
- 目标 SoC 不支持 `npu_swiglu` 或目标 GMM 模式。

### 阶段 1：迁移 SelectTopk

任务：

1. 在 Ascend `SelectTopk` 中惰性或平台安全地获取 `torch_npu`；
2. 调用 `npu_moe_gating_top_k(router_logits_fp32, k, norm_type=0)`；
3. 正确处理三个返回值；
4. `has_moe_norm=True` 时对选中权重重新归一化；
5. 将结果复制到调用方的预分配 tensor；
6. 保持 `topk_ids=int32`、`topk_weights=float32`；
7. 对不支持的 GroupTopK 和 fake-balance 保留清晰异常。

验证：

- `has_moe_norm=True/False` 各一组；
- `M=1` 和 `M>1`；
- `K=1` 和 `K>1`；
- 包含极大/极小 logits 的数值稳定性；
- top-k 相同分数时只比较集合或按算子稳定排序契约比较；
- 对比 `softmax(logits).topk(K)` reference。

### 阶段 2：实现纯 TP Router

任务：

1. 新建 `NpuPureTpRouter`；
2. `prepare` 校验 `a1`、topk tensor shape/dtype 和非量化参数；
3. 直接传递 `a1/topk_ids/topk_weights`，不做 CPU 搬运；
4. `finalize` 直接取 Executor 输出；
5. 仅在 `tp_size>1` 时调用 `all_reduce(output, Group.TP)`；
6. 条件检查严格限制 `ep_size=1、dp_size=1`。

验证：

- 单卡不调用 all-reduce；
- TP2 调用一次正确 group 的 all-reduce；
- EP/DP 配置无法匹配该 Strategy；
- Router 不导入 Triton。

### 阶段 3：实现 CANN Executor

任务顺序：

1. `topk_ids_dtype` 返回 `torch.int32`；
2. 初始化时校验 BF16、非量化、权重 shape 和 activation；
3. 一次性准备符合 CANN 要求的 W1/W2；
4. 调用 `npu_moe_init_routing_v2` 并接收四个返回值；
5. 校验 `group_list` 是长度 E 的 INT64 cumsum；
6. 调用 GMM1，正确使用列表入参和列表返回值；
7. 调用 `npu_swiglu`，确保前半段是 gate、后半段是 up；
8. 调用 GMM2；
9. 调用 `npu_moe_token_unpermute`，融合 top-k 权重并恢复 `[M,H]`；
10. 返回 `CombineForwardPayload`。

必须覆盖的边界：

- 某个 expert 分到 0 个 token；
- 所有 token 分到同一 expert；
- `M=1` 的 decode；
- `M>1` 的 prefill；
- `K>1` 时同一 token 的多 expert 汇总；
- 非连续输入要么显式 `.contiguous()`，要么在入口明确拒绝；
- 输出 shape、dtype 与 `hidden_states` 完全一致。

显存要求：

- 不允许 forward 中重排完整权重；
- 记录 W1/W2 转换前后的 storage 和峰值显存；
- 若产生永久双份专家权重，在端到端验收前必须改为加载期布局转换或其他无重复方案。

### 阶段 4：注册 Strategy 与修复构建

任务：

1. 新建并导出 `AscendCannStrategy`；
2. `check_conditions` 至少检查：
   - BF16；
   - 无量化；
   - `ep_size=1`；
   - `dp_size=1`；
   - eager；
   - 非 fake-balance；
3. 在 Ascend registry 中优先注册；
4. 删除或修正错误的 Triton fallback；
5. 补全每个 Bazel 子包的 deps；
6. 用依赖检查确认 Ascend MoE target 到 Triton target 不存在路径；
7. 验证 CUDA/ROCm registry 和 imports 没有变化。

建议检查命令：

```bash
bazel build --config=ascend //rtp_llm/models_py:models
bazel cquery --config=ascend \
  'somepath(//rtp_llm/models_py:models, //rtp_llm/models_py/triton_kernels:triton_kernels)'
```

若整个 `models` 目标仍因仓库既有依赖包含 Triton，应区分“本次新增 Ascend MoE target
是否依赖 Triton”和“仓库历史全局依赖”两类问题，并给出准确说明，不能用文本扫描冒充
Bazel 依赖验证。

### 阶段 5：单算子与完整链路测试

先做 CPU/mock 静态测试，再做真实 NPU 测试。

#### 5.1 不依赖 NPU 的测试

- Strategy 选择和拒绝条件；
- Ascend 源码没有 RTP Triton import；
- 五个 CANN 算子的调用次数、参数名和返回值拆包；
- Top-K renormalize；
- W1 `[up,gate] -> [gate,up]` 的非对称测试；
- init-routing 索引方向；
- GMM 列表入参和 `[0]` 返回值；
- 单卡/TP 的 all-reduce 分支；
- unsupported 配置的异常信息。

mock 测试只能验证接线，不得宣称算子精度通过。

#### 5.2 NPU 单算子 golden

每项均与 PyTorch 或 GPU reference 比较：

1. `npu_moe_gating_top_k`；
2. `npu_moe_init_routing_v2` 的 sorted token、row index、group list；
3. GMM1；
4. `npu_swiglu`；
5. GMM2；
6. `npu_moe_token_unpermute`；
7. Router TP all-reduce。

测试数据必须包含零 token expert、重复 expert id、K>1、非对称 gate/up 和可辨识 token
编号。误差阈值由 BF16 golden 实测确定，不能先随意放宽到掩盖布局错误。

#### 5.3 完整 MoE 链路

至少覆盖：

| 场景 | 必测内容 |
|---|---|
| 单卡 decode | `M=1`，多轮 token，输出与 reference 对齐 |
| 单卡 prefill | 小 batch 和实际长序列 token 数 |
| Top-K | `K=1`、Qwen3.5 实际 K、renorm 开关 |
| Expert 分布 | 均匀、极端倾斜、零 token expert |
| Shared Expert | 有/无 shared gate，两种融合路径 |
| TP2 | 输出与单卡 reference 对齐，通信次数正确 |
| 模型 E2E | Qwen3.5 MoE checkpoint 的 prefill + decode |

E2E 需要同时检查：

- 首 token 和后续 token logits；
- 非 NaN/Inf；
- 输出 dtype/shape；
- 多层运行后误差是否累积；
- 峰值显存没有永久双份专家权重；
- 生产路径没有 `.cpu()`、`.numpy()`、`.tolist()` 等 host round-trip。

### 阶段 6：后续扩展，不与首期混做

首期通过后再分别立项：

1. EP Router：
   - `npu_moe_distribute_dispatch_v2`；
   - `npu_moe_distribute_combine_v2`；
   - 多机容错和通信域验证。
2. 量化 MoE：
   - 动态量化；
   - `npu_grouped_matmul_swiglu_quant`；
   - GMM2 量化路径；
   - 独立精度基线。
3. Graph：
   - 动态 token 数；
   - buffer 复用；
   - graph capture 兼容性。
4. 性能：
   - GMM tuning；
   - 权重预排布；
   - 减少中间 tensor 和同步；
   - 与 GPU/Triton 及 PyTorch fallback 对比。

## 9. 建议测试文件与 target

建议新增：

```text
rtp_llm/models_py/modules/base/ascend/test/select_topk_test.py
rtp_llm/models_py/modules/factory/fused_moe/impl/ascend/routers/test/npu_pure_tp_router_test.py
rtp_llm/models_py/modules/factory/fused_moe/impl/ascend/executors/test/npu_fused_experts_test.py
rtp_llm/models_py/modules/factory/fused_moe/impl/ascend/test/fused_moe_integration_test.py
```

每个 test 目录建立独立 `BUILD`。纯 mock 测试和 NPU 真机测试应分 target；NPU 测试可
标记 `manual`，但不能只放散落的 Python 文件而不进入任何测试清单。

建议最终验证命令：

```bash
python -m unittest discover \
  -s rtp_llm/models_py/modules/base/ascend/test \
  -p '*topk*test.py' -v

python -m unittest discover \
  -s rtp_llm/models_py/modules/factory/fused_moe/impl/ascend \
  -p '*test.py' -v

bazel build --config=ascend //rtp_llm/models_py:models
bazel test --config=ascend <新增的Ascend-MoE静态测试targets> --test_output=errors
bazel test --config=ascend <新增的Ascend-MoE真机测试targets> --test_output=errors
```

如有 CUDA 环境，再执行受影响的既有 MoE 测试，证明非 Ascend 路径没有回归。

## 10. 验收标准（Definition of Done）

只有同时满足以下条件，才可以说“Qwen3.5 Ascend MoE 首期接入完成”：

- [ ] Ascend SelectTopk 使用 CANN op，且 renormalize 语义正确；
- [ ] Ascend FusedMoe 使用新的 CANN Router/Executor；
- [ ] init-routing、GMM1、SwiGLU、GMM2、unpermute 全部在 NPU 上执行；
- [ ] W1 gate/up 顺序通过非对称 golden；
- [ ] 返回值数量、列表参数、dtype 和布局与 pinned wheel 一致；
- [ ] 单卡和纯 TP 路径通过；
- [ ] 不支持的 EP、量化、GroupTopK、fake-balance、graph 显式拒绝；
- [ ] Ascend MoE 新代码不导入 RTP Triton kernel；
- [ ] Bazel Ascend build 和新增测试 target 通过；
- [ ] NPU 单算子 golden 通过；
- [ ] Qwen3.5 MoE prefill/decode E2E 通过；
- [ ] Shared Expert 路径通过；
- [ ] 无 CPU round-trip；
- [ ] 无未处理的永久专家权重双份占用；
- [ ] CUDA/ROCm 行为未被修改或已通过回归验证；
- [ ] 文档记录实际 wheel、SoC、测试命令、精度阈值和结果。

只完成 mock 测试、只完成单算子测试，或模型能够启动但没有 E2E 精度对比，都不能
标记为完成。

## 11. 风险与处理原则

| 风险 | 后果 | 处理方式 |
|---|---|---|
| torch_npu 版本漂移 | 参数或返回值不匹配 | 以 requirements 固定 wheel 真机预检为准 |
| W1 半区顺序错误 | 数值看似正常但模型结果错误 | 非对称 gate/up golden，禁止只 transpose |
| row index 方向错误 | token 输出被错误还原 | 手工 token/id 测试验证 sorted-to-original |
| group_list 语义错误 | GMM expert 分组错位 | 验证 cumsum、最后一项和单调性 |
| fallback 仍引用 Triton | Ascend import/build 失败 | 删除静默 fallback，做 Bazel 依赖检查 |
| 永久双份权重 | 大模型加载 OOM | 优先加载期转换，验收峰值显存 |
| TP/EP 条件混淆 | 权重或输出重复/缺失 | 首期严格 `ep=1, dp=1`，EP 独立开发 |
| 只看单 op 精度 | 多层 E2E 误差未发现 | prefill/decode 完整模型 golden 是硬门槛 |

## 12. AI 开发约束

交给 AI 执行时，必须遵守：

1. 开始前运行 `git status --short --branch`，保留用户已有修改；
2. 先读本文档及第 13 节引用的新增指导，不得只看伪代码；
3. 按阶段 0 至阶段 5 顺序开发，一次只推进一个可验证阶段；
4. 每阶段结束报告修改文件、测试命令、真实结果和未验证项；
5. 不得把跳过的测试写成通过；
6. 不得在非 NPU 环境声称 NPU 精度或性能已验证；
7. 不得静默回退 Triton、CUDA 或 CPU；
8. 不得扩大到 EP、量化或 Graph，除非首期完成后另行授权；
9. 不得修改非 Ascend 后端行为；
10. 未经用户明确要求，不提交、不推送、不改远端分支。

可直接交给 AI 的任务描述：

```text
请严格按照 docs/references/qwen35_ascend_moe_development_plan.md，完成
Qwen3.5 Ascend MoE 首期 CANN 接入。

先执行阶段 0，核对当前工作树、pinned torch_npu 的五个算子接口和 Qwen3.5
实际权重布局。然后依次实现 Ascend SelectTopk、NpuPureTpRouter、
NpuFusedExpertsExecutor、AscendCannStrategy、Bazel targets 和测试。

重点检查：gating/init-routing 的真实返回数量、grouped-matmul 的列表接口、
W1 从 [up,gate] 到 [gate,up] 的顺序、top-k renormalize、row_idx_type 与
token_unpermute 的索引方向，以及专家权重不能永久双份占用。

范围仅限 BF16、单卡/纯 TP、eager、非量化。EP、GroupTopK、fake-balance、
量化和 Graph 必须明确拒绝，不能做 Triton/CPU fallback。

每完成一个阶段先运行对应测试并汇报，不要提交或推送。只有真实 NPU 单算子、
完整 MoE 链路和 Qwen3.5 prefill/decode E2E 全部完成后，才可标记任务完成。
```

## 13. 参考资料

- 新增迁移方案：
  `rtp-llm-ascend-adaption-plan/2-op/rtp-llm Qwen3.5 MoE 层整体迁移方案.md`
- 更新后的总指导：
  `rtp-llm-ascend-adaption-plan/2-op/rtp-llm Qwen3.5 算子迁移指导.md`
- MoE 调用关系：
  `rtp-llm-ascend-adaption-plan/2-op/MoE层算子调用关系与迁移方案.md`
- Top-K 迁移说明：
  `rtp-llm-ascend-adaption-plan/2-op/topkGatingSoftmaxKernelLauncher_算子解析与NPU迁移建议.md`
- 当前已完成的线性注意力接入说明：
  `docs/references/qwen35_ascend_ops.md`
- torch_npu 26.1 op-plugin 函数声明：
  <https://gitcode.com/Ascend/op-plugin/blob/26.1.0/op_plugin/config/op_plugin_functions.yaml>
- `npu_moe_init_routing_v2`：
  <https://gitcode.com/Ascend/op-plugin/blob/26.0.0/docs/zh/custom_APIs/torch_npu/torch_npu-npu_moe_init_routing_v2.md>
- `npu_grouped_matmul`：
  <https://gitcode.com/Ascend/op-plugin/blob/26.0.0/docs/zh/custom_APIs/torch_npu/torch_npu-npu_grouped_matmul.md>

开发时以本仓 `deps/requirements_ascend.txt` 固定的 wheel 行为为最终依据，线上最新
文档只能作为参考。
