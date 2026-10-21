#include "rtp_llm/models_py/bindings/RegisterOps.h"

namespace rtp_llm {
void registerPyModuleOps(pybind11::module& m) {
    // Qwen3.5 / linear-attention models persist SSM & KV states through the
    // cache store after each prefill. On this lineage the write_cache_store
    // binding lives in the core exec-ops registration (the legacy
    // common/WriteCacheStoreOp.cc wrapper was removed upstream in
    // ed96f0485); nothing to register here for now.
}
}
