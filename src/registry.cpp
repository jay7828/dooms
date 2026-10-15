#include "dooms/dooms.h"

namespace dooms {

Registry & Registry::get() {
    static Registry r;
    return r;
}

Status Registry::add(const std::string & id, ModelFactory factory) {
    if (id.empty()) return invalid_argument("model id must not be empty");
    if (!factory) return invalid_argument("model '" + id + "': factory must not be empty");
    std::lock_guard<std::mutex> lock(mutex_);
    if (!factories_.emplace(id, std::move(factory)).second) {
        return invalid_argument("model id '" + id + "' is already registered");
    }
    return Status();
}

Result<std::unique_ptr<Model>> Registry::create(const std::string & id) const {
    ModelFactory factory;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = factories_.find(id);
        if (it == factories_.end()) return not_found("unknown model '" + id + "'");
        factory = it->second;
    }
    std::unique_ptr<Model> m = factory();
    if (!m) return internal_error("factory for model '" + id + "' returned null");
    return m;
}

std::vector<std::string> Registry::ids() const {
    std::lock_guard<std::mutex> lock(mutex_);
    std::vector<std::string> out;
    for (const auto & kv : factories_) out.push_back(kv.first);  // std::map iterates in sorted order
    return out;
}

}  // namespace dooms
