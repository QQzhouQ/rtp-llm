# Qwen3.5 Ascend 算子接入

Qwen3.5 保留共享模型定义，并在 `model_desc/qwen3_next.py` 中选择后端实现：

- 非 Ascend 后端继续导入原有 Triton 实现；
- Ascend 导入 `rtp_llm.models_py.kernels.ascend`，该目录只包含 Ascend 实现，
  并仅在 `using_ascend` 构建分支中引入；
- chunk/recurrent/causal-conv 算子调用 `fla_npu.ops.ascendc`；
- GDN gating、gated RMSNorm、block-map 搬运使用 torch_npu 可执行的 PyTorch 算子；
- L2Norm 优先调用 FLA Triton-for-Ascend，未安装该可选模块时使用等价 PyTorch 实现。

Ascend 实现保持模型现有调用点的函数签名和返回布局，内部不再判断设备或回退到
CUDA/ROCm。Ascend recurrent decode 当前只支持模型生产路径使用的
`inplace_final_state=True`；其他模式会显式报错，不会静默产生形状不兼容的状态。

## 环境准备

`flash-linear-attention-npu` 的 wheel 与 SoC 绑定，不能作为通用 requirements
直接锁定。请先切换到已验 GPU golden 的 FLA-NPU commit，再在
`flash-linear-attention-npu` 源码目录中构建并安装（以下命令不是在 RTP-LLM
仓库根目录执行）：

```bash
git clone https://github.com/flashserve/flash-linear-attention-npu.git
cd flash-linear-attention-npu
# git checkout <已验的 commit 或 tag>
source /usr/local/Ascend/ascend-toolkit/set_env.sh
python -m pip install -r requirements.txt
python scripts/check_npu_env.py --build-only
FLA_NPU_SOC=ascend950 python -m pip wheel --no-build-isolation --no-deps . -w dist
python -m pip install --force-reinstall --no-deps dist/flash_linear_attention_npu-*.whl
```

`FLA_NPU_SOC` 按机器设置为 `ascend910b`、`ascend910_93` 或 `ascend950`。
安装后先检查生产路径所需符号：

```bash
python - <<'PY'
from fla_npu.ops import ascendc

required = (
    "npu_causal_conv1d",
    "npu_chunk_local_cumsum",
    "npu_chunk_scaled_dot_kkt",
    "npu_solve_tri",
    "npu_recompute_w_u_fwd",
    "npu_chunk_gated_delta_rule_fwd_h",
    "npu_chunk_fwd_o",
    "npu_recurrent_gated_delta_rule",
)
missing = [name for name in required if not hasattr(ascendc, name)]
assert not missing, f"missing FLA-NPU operators: {missing}"
PY
```

FLA-NPU 的 Python 入口仍在演进；若预检缺少 recurrent 或其他符号，应切换到
与本仓 golden 数据匹配的 commit/wheel，不能等到模型运行时再降级或绕过。

## 验证

不依赖设备的分发、布局和 cache 语义测试：

```bash
python -m unittest discover \
  -s rtp_llm/models_py/kernels/ascend/test \
  -p 'test_*.py' -v
```

Ascend 真机上，准备 GPU golden 数据后继续运行：

```bash
TEST_DEVICE_ID=0 python -m unittest discover \
  -s example/ascendc_npu -p 'test_*.py' -v
TEST_DEVICE_ID=0 python example/triton_npu/test_npu_l2norm_fwd_gpu_golden.py
TEST_DEVICE_ID=0 python example/triton_npu/test_npu_fused_gdn_gating_gpu_golden.py
```

正式验收还必须包含完整 Qwen3.5 Dense 链路：prefill seq32/seq2047、prefix
cache、跨 block、多 batch 单 token decode 和 target verify。单算子 golden 通过不能替代
长序列端到端精度验证。

当前 recurrent AscendC 算子仅支持 BF16 state，单序列一次最多 8 个 decode token；
NPU graph、context parallel 和 MoE grouped-GEMM 不在本阶段验收范围内。已验
wheel 后，还必须对 B>1、T=2..8 核对 `actual_seq_lengths`、
`num_accepted_tokens` 默认行为和每个 token 的 state page 快照；FLA-NPU 不同
commit 的 Python/ACLNN 契约仍在演进。
