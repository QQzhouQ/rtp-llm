#pragma once

#if USING_ASCEND
#include <acl/acl.h>

namespace rtp_llm {
namespace ascend {

using ascendStream_t = aclrtStream;
using ascendEvent_t  = aclrtEvent;

template<typename T>
void check(T result, const char* const file, int const line);

void syncAndCheckInDebug(const char* const file, int const line);

inline aclrtMemcpyKind getMemcpyKind(const void* src, const void* dst) {
    aclrtPtrAttributes src_attr{};
    aclrtPtrAttributes dst_attr{};
    aclrtPointerGetAttributes(src, &src_attr);
    aclrtPointerGetAttributes(dst, &dst_attr);
    bool src_is_device = (src_attr.location.type == ACL_MEM_LOCATION_TYPE_DEVICE);
    bool dst_is_device = (dst_attr.location.type == ACL_MEM_LOCATION_TYPE_DEVICE);
    if (src_is_device && dst_is_device) {
        return ACL_MEMCPY_DEVICE_TO_DEVICE;
    } else if (!src_is_device && dst_is_device) {
        return ACL_MEMCPY_HOST_TO_DEVICE;
    } else if (src_is_device && !dst_is_device) {
        return ACL_MEMCPY_DEVICE_TO_HOST;
    }
    return ACL_MEMCPY_HOST_TO_HOST;
}

}  // namespace ascend
}  // namespace rtp_llm

#define ASCEND_CHECK(val) rtp_llm::ascend::check((val), __FILE__, __LINE__)
#define ASCEND_CHECK_ERROR() rtp_llm::ascend::syncAndCheckInDebug(__FILE__, __LINE__)

#endif  // USING_ASCEND
