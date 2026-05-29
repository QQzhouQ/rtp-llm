import os, sys, logging as _logging

_bazel_bin = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "..", "bazel-bin")
_th_so = os.path.join(_bazel_bin, "libth_transformer.so")
if os.path.exists(_th_so):
    try:
        from ctypes import CDLL; import os as _os
        CDLL(_th_so, mode=_os.RTLD_GLOBAL | _os.RTLD_NOW)
        _logging.info(f"pre-loaded {_th_so} with RTLD_GLOBAL")
    except BaseException as e:
        _logging.warning(f"Failed to pre-load libth_transformer.so: {e}")

import logging

from librtp_compute_ops import *
from librtp_compute_ops.rtp_llm_ops import *

# === Python-level stubs for Ascend build (registerExecCtxOps is CUDA/ROCm only) ===
# These functions are registered by registerExecCtxOps() in ComputeInit.cc,
# which is only compiled under #if USING_CUDA || USING_ROCM.
# On Ascend, we provide no-op Python stubs so imports don't fail.
import librtp_compute_ops as _lco
if not hasattr(_lco, 'get_device_id'):
    def get_device_id():
        import torch
        try:
            return torch.npu.current_device()
        except Exception:
            return 0

    def preprocess_gemm_weight_by_key(key, weight, use_arm_gemm_use_kai=False):
        return weight

    def preprocess_weight_scale(weight, scale):
        return weight

    _lco.get_device_id = get_device_id
    _lco.preprocess_gemm_weight_by_key = preprocess_gemm_weight_by_key
    _lco.preprocess_weight_scale = preprocess_weight_scale
    _logging.info("Injected Ascend stubs: get_device_id, preprocess_gemm_weight_by_key, preprocess_weight_scale")
else:
    _logging.info("registerExecCtxOps symbols already present, skipping stubs")
# === End Python-level stubs ===

from rtp_llm.models_py.utils.arch import is_cuda

if is_cuda():
    logging.info("Use rtp_kernel FusedRopeKVCacheOp on CUDA device.")

    from .fused_rope_kvcache_op import (
        FusedRopeKVCacheDecodeOp,
        FusedRopeKVCachePrefillOpQKVOut,
        FusedRopeKVCachePrefillOpQOut,
    )
else:
    logging.info(
        "Fallback to default implementation of FusedRopeKVCacheOp on non-CUDA device."
    )

    try:
        from librtp_compute_ops.rtp_llm_ops import (
            FusedRopeKVCacheDecodeOp,
            FusedRopeKVCachePrefillOpQKVOut,
            FusedRopeKVCachePrefillOpQOut,
        )

        logging.info("Loaded C++ FusedRopeKVCacheOp from librtp_compute_ops")
    except ImportError as e:
        logging.warning(f"Failed to load C++ FusedRopeKVCacheOp: {e}")
