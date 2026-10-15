// LLM path: delegate to llama.cpp for GGUF text/VLM backbones.
// Everything that touches llama.h lives in this file because llama.cpp's API changes
// often. This version only loads the model; token generation is M9 (docs/PLAN.md).

#include <fstream>
#include <mutex>

#include "core/log.h"
#include "dooms/dooms.h"
#include "llama.h"

namespace dooms {

namespace {

constexpr double kParamsPerBillion = 1e9;

class LlamaBridge : public Model {
public:
    ~LlamaBridge() override {
        if (model_) llama_model_free(model_);
    }

    std::string name() const override { return "llama"; }

    Status load(const std::string & path) override {
        if (model_) return invalid_argument("llama: a model is already loaded");
        if (!std::ifstream(path, std::ios::binary)) return not_found("llama: cannot open '" + path + "'");

        static std::once_flag backend_once;
        std::call_once(backend_once, [] { llama_backend_init(); });

        model_ = llama_model_load_from_file(path.c_str(), llama_model_default_params());
        if (!model_) return data_loss("llama: llama.cpp could not load '" + path + "' as a GGUF model");
        DOOMS_LOG_INFO("llama: loaded %s (%.2f B params)", path.c_str(),
                       static_cast<double>(llama_model_n_params(model_)) / kParamsPerBillion);
        return Status();
    }

    Status run(const TensorMap &, TensorMap &) override {
        return unsupported("llama: generation is not implemented yet (M9)");
    }

private:
    llama_model * model_ = nullptr;
};

}  // namespace

Status register_llama_bridge() {
    return Registry::get().add("llama", [] { return std::make_unique<LlamaBridge>(); });
}

}  // namespace dooms
