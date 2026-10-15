#include <mutex>

#include "dooms/dooms.h"

namespace dooms {

#ifdef DOOMS_WITH_LLAMA
Status register_llama_bridge();
#endif

Status register_builtin_models() {
    static std::once_flag once;
    static Status result;
    std::call_once(once, [] {
#ifdef DOOMS_WITH_LLAMA
        result = register_llama_bridge();
#endif
    });
    return result;
}

}  // namespace dooms
