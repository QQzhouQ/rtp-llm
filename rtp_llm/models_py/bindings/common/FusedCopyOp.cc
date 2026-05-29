#include <cstring>
#include "rtp_llm/models_py/bindings/core/ExecOps.h"
#include "rtp_llm/models_py/bindings/core/Types.h"
#include "rtp_llm/models_py/bindings/common/kernels/fuse_copy_kernel.h"

#if USING_CUDA
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#endif
#if USING_ROCM
#include <ATen/hip/HIPContext.h>
#include "rtp_llm/models_py/bindings/rocm/cuda_shims.h"
#include <hip/hip_runtime.h>
#endif
#if USING_ASCEND
#include <acl/acl.h>
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-function"
#include <torch_npu/csrc/core/npu/NPUStream.h>
#pragma GCC diagnostic pop
#include "rtp_llm/models_py/bindings/ascend/ascend_types_hdr.h"
#endif

namespace rtp_llm {

void fusedCopy(const FusedD2DCopyParams& params) {
#if USING_CUDA
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    invokeFusedCopy(params, stream);
#elif USING_ROCM
    hipStream_t stream = at::hip::getCurrentHIPStream();
    invokeFusedCopy(params, stream);
#elif USING_ASCEND
    aclrtStream stream = c10_npu::getCurrentNPUStream().stream();
    for (int i = 0; i < params.num_copies; ++i) {
        aclrtMemcpyKind kind = rtp_llm::ascend::getMemcpyKind(params.src[i], params.dst[i]);
        ASCEND_CHECK(aclrtMemcpyAsync(params.dst[i], params.size[i],
                                       const_cast<void*>(params.src[i]), params.size[i],
                                       kind, stream));
    }
#else
    for (int i = 0; i < params.num_copies; ++i) {
        memcpy(params.dst[i], params.src[i], params.size[i]);
    }
#endif
}

void fusedStridedCopy(const FusedStridedCopyParams& params) {
#if USING_CUDA
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    invokeFusedStridedCopy(params, stream);
#elif USING_ROCM
    hipStream_t stream = at::hip::getCurrentHIPStream();
    invokeFusedStridedCopy(params, stream);
#elif USING_ASCEND
    aclrtStream stream = c10_npu::getCurrentNPUStream().stream();
    for (int i = 0; i < params.num_copies; ++i) {
        aclrtMemcpyKind kind = rtp_llm::ascend::getMemcpyKind(params.src[i], params.dst[i]);
        const char* src_ptr = static_cast<const char*>(params.src[i]);
        char* dst_ptr = static_cast<char*>(params.dst[i]);
        for (size_t row = 0; row < params.num_rows[i]; ++row) {
            ASCEND_CHECK(aclrtMemcpyAsync(dst_ptr, params.row_bytes[i],
                                           const_cast<void*>(static_cast<const void*>(src_ptr)),
                                           params.row_bytes[i],
                                           kind, stream));
            src_ptr += params.src_row_stride[i];
            dst_ptr += params.dst_row_stride[i];
        }
    }
#else
    for (int i = 0; i < params.num_copies; ++i) {
        const char* src_ptr = static_cast<const char*>(params.src[i]);
        char* dst_ptr = static_cast<char*>(params.dst[i]);
        for (size_t row = 0; row < params.num_rows[i]; ++row) {
            memcpy(dst_ptr, src_ptr, params.row_bytes[i]);
            src_ptr += params.src_row_stride[i];
            dst_ptr += params.dst_row_stride[i];
        }
    }
#endif
}

}  // namespace rtp_llm
