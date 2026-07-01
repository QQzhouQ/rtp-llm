#!/bin/bash
# 用途：在 Bazel genrule 中调用 CMake + bisheng 编译 AscendC 自定义算子
# 输入：$1 = Bazel genrule 输出根目录（如 bazel-out/.../runfiles/）
#        $2 = 算子名称（分号分隔），如 "apply_top_k_top_p_custom" 或 "op_a;op_b;op_c"
#        $3 = SOC 版本（默认 ascend910b）
# 输出：$1/lib/libopapi.so、$1/lib/libop_host_aclnnExc.so、$1/opp/
set -e

OUT_DIR="$1"
OP_NAMES="${2:-ALL}"
SOC_VERSION="${3:-ascend910b}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 确保 OUT_DIR 是绝对路径（后续 cd ${BUILD_DIR} 会改变 CWD）
if [[ "${OUT_DIR}" != /* ]]; then
    OUT_DIR="$(pwd)/${OUT_DIR}"
fi

# ====================================================================
# 1. 确定 CANN 安装路径（与 vllm-ascend/csrc/build.sh 逻辑一致）
# ====================================================================
if [ -n "${ASCEND_HOME_PATH}" ]; then
    ASCEND_CANN_PACKAGE_PATH="${ASCEND_HOME_PATH}"
elif [ -n "${ASCEND_OPP_PATH}" ]; then
    ASCEND_CANN_PACKAGE_PATH="$(dirname "${ASCEND_OPP_PATH}")"
elif [ -d "/usr/local/Ascend/ascend-toolkit/latest" ]; then
    ASCEND_CANN_PACKAGE_PATH="/usr/local/Ascend/ascend-toolkit/latest"
elif [ -d "${HOME}/Ascend/latest" ]; then
    ASCEND_CANN_PACKAGE_PATH="${HOME}/Ascend/latest"
else
    echo "[AscendC] ERROR: Cannot find CANN toolkit." >&2
    echo "  Set ASCEND_HOME_PATH or ASCEND_OPP_PATH env variable." >&2
    exit 1
fi
echo "[AscendC] CANN path: ${ASCEND_CANN_PACKAGE_PATH}"

# 加载 CANN 编译环境（设置 bisheng PATH、CANN lib 路径等）
source "${ASCEND_CANN_PACKAGE_PATH}/bin/setenv.bash" || {
    echo "[AscendC] ERROR: Failed to source setenv.bash" >&2
    exit 1
}

# 确保 conda Python 在 PATH 最前面（CANN kernel 编译脚本依赖 numpy）
if [ -d "/root/miniconda3/envs/py310/bin" ]; then
    export PATH="/root/miniconda3/envs/py310/bin:${PATH}"
    HOST_PYTHON="/root/miniconda3/envs/py310/bin/python3"
else
    HOST_PYTHON="/usr/bin/python3.10"
fi

# ====================================================================
# 2. 验证 bisheng 编译器可用
# ====================================================================
BISHENG_PATH="$(which bisheng 2>/dev/null || true)"
if [ -z "${BISHENG_PATH}" ]; then
    echo "[AscendC] ERROR: bisheng compiler not found." >&2
    echo "  Please check CANN toolkit installation." >&2
    exit 1
fi
echo "[AscendC] bisheng: ${BISHENG_PATH}"

# ====================================================================
# 3. CMake 配置 + 编译
# ====================================================================
BUILD_DIR="$(mktemp -d)"
echo "[AscendC] Building in ${BUILD_DIR} for ${SOC_VERSION}"

cd "${BUILD_DIR}"

# 调用源项目自带的 build.sh（支持 -n 分号分隔多算子、-c 指定芯片型号）
# ⚠️ 注意：build.sh 内部硬编码了 BUILD_DIR=${SCRIPT_DIR}/build，不是这里的临时目录
bash "${SCRIPT_DIR}/build.sh" -n "${OP_NAMES}" -c "${SOC_VERSION}"

# build.sh 使用的实际构建目录
REAL_BUILD_DIR="${SCRIPT_DIR}/build"

# ====================================================================
# 4. 收集产物到 Bazel 输出目录
# ====================================================================
echo "[AscendC] Collecting artifacts to ${OUT_DIR}"
mkdir -p "${OUT_DIR}/lib"

# CPack 将产物安装到 _CPack_Packages 目录，然后打包成 .run
CPACK_DIR=$(find "${REAL_BUILD_DIR}/_CPack_Packages" -type d -name "CANN-custom_ops*" 2>/dev/null | head -1)
VENDOR_DIR="${CPACK_DIR}/packages/vendors/${VENDOR_NAME:-rtp-llm}"

# 4a. libcust_opapi.so — ACLNN 算子接口（dlopen 加载的核心库）
OPAPI_SRC=$(find "${VENDOR_DIR}" -name 'libcust_opapi.so' 2>/dev/null | head -1)
if [ -n "${OPAPI_SRC}" ]; then
    cp "${OPAPI_SRC}" "${OUT_DIR}/lib/"
    echo "[AscendC]   libcust_opapi.so ✓"
else
    echo "[AscendC] ERROR: libcust_opapi.so not found!" >&2
    exit 1
fi

# 4b. libcust_opsproto_rt2.0.so — 算子原型库
OPSPROTO_SRC=$(find "${VENDOR_DIR}" -name 'libcust_opsproto_rt2.0.so' 2>/dev/null | head -1)
if [ -n "${OPSPROTO_SRC}" ]; then
    cp "${OPSPROTO_SRC}" "${OUT_DIR}/lib/"
    echo "[AscendC]   libcust_opsproto_rt2.0.so ✓"
fi

# 4c. libcust_opmaster_rt2.0.so — Tiling 库
OPMASTER_SRC=$(find "${VENDOR_DIR}" -name 'libcust_opmaster_rt2.0.so' 2>/dev/null | head -1)
if [ -n "${OPMASTER_SRC}" ]; then
    cp "${OPMASTER_SRC}" "${OUT_DIR}/lib/"
    echo "[AscendC]   libcust_opmaster_rt2.0.so ✓"
fi

# 4d. opp/ 目录 — kernel 二进制 + 配置文件（CANN Runtime 通过 ASCEND_CUSTOM_OPP_PATH 加载）
# CANN 要求 ASCEND_CUSTOM_OPP_PATH 指向包含 op_impl/ 子目录的路径，
# 因此必须保留 op_impl/ 目录名而非仅复制其内容。
OP_IMPL_DIR=$(find "${VENDOR_DIR}" -type d -path "*/op_impl" 2>/dev/null | head -1)
if [ -n "${OP_IMPL_DIR}" ]; then
    mkdir -p "${OUT_DIR}/opp"
    cp -r "${OP_IMPL_DIR}" "${OUT_DIR}/opp/"
    echo "[AscendC]   opp/op_impl/ ✓"
else
    echo "[AscendC] WARNING: op_impl directory not found, kernel binaries may be missing" >&2
fi

rm -rf "${REAL_BUILD_DIR}" "${BUILD_DIR}"

echo "[AscendC] ========================================"
echo "[AscendC] Build complete. Artifacts:"
ls -lh "${OUT_DIR}/lib/" 2>/dev/null || echo "  (no .so files)"
if [ -d "${OUT_DIR}/opp/op_impl" ]; then
    echo "[AscendC] Kernel binaries:"
    find "${OUT_DIR}/opp/op_impl" -name '*.o' -exec ls -lh {} \; 2>/dev/null || true
fi
echo "[AscendC] ========================================"