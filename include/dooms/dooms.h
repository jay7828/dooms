#pragma once
// dooms.cpp public model API: a named, loadable thing that maps input tensors to
// output tensors. Today only the llama.cpp bridge implements it; IR-based models
// (vit) arrive in M4 and will replace the host-tensor interface here with
// one built on core/TensorView. No ggml or llama types appear in this header.

#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "core/status.h"

namespace dooms {

struct Tensor {
    std::vector<int64_t> shape;  // row-major, outermost first
    std::vector<float> data;     // host fp32
};

using TensorMap = std::map<std::string, Tensor>;

class Model {
public:
    virtual ~Model() = default;
    virtual std::string name() const = 0;
    // Load from `path`. Fails with NotFound (no such file), DataLoss (unreadable/corrupt) or Unsupported.
    virtual Status load(const std::string & path) = 0;
    virtual Status run(const TensorMap & inputs, TensorMap & outputs) = 0;
};

using ModelFactory = std::function<std::unique_ptr<Model>()>;

// Maps a model id to a factory. Thread-safe.
class Registry {
public:
    static Registry & get();
    // InvalidArgument if the id is empty or already registered; ids are never silently replaced.
    Status add(const std::string & id, ModelFactory factory);
    // NotFound if the id is not registered.
    Result<std::unique_ptr<Model>> create(const std::string & id) const;
    std::vector<std::string> ids() const;  // sorted

private:
    mutable std::mutex mutex_;
    std::map<std::string, ModelFactory> factories_;
};

// Registers every model compiled into this build. Idempotent.
Status register_builtin_models();

}  // namespace dooms
