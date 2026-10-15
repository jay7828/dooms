#include "models/vlm/tokenizer.h"

#include <fstream>
#include <mutex>

#include "llama.h"

namespace dooms {

struct Tokenizer::Impl {
    llama_model * model = nullptr;
    const llama_vocab * vocab = nullptr;
    int32_t n_vocab = 0, pad = -1;
    ~Impl() { if (model) llama_model_free(model); }
};

Tokenizer::Tokenizer(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}
Tokenizer::Tokenizer(Tokenizer &&) noexcept = default;
Tokenizer & Tokenizer::operator=(Tokenizer &&) noexcept = default;
Tokenizer::~Tokenizer() = default;

Result<Tokenizer> Tokenizer::load(const std::string & path) {
    if (!std::ifstream(path).good()) return not_found("no such file: " + path);
    static std::once_flag backend_once;
    std::call_once(backend_once, [] { llama_backend_init(); });
    auto impl = std::make_unique<Impl>();
    llama_model_params mp = llama_model_default_params();
    mp.vocab_only = true;
    impl->model = llama_model_load_from_file(path.c_str(), mp);
    if (!impl->model) return data_loss("llama.cpp could not load the vocabulary '" + path + "'");
    impl->vocab = llama_model_get_vocab(impl->model);
    impl->n_vocab = llama_vocab_n_tokens(impl->vocab);
    impl->pad = llama_vocab_pad(impl->vocab);
    if (impl->pad < 0 || impl->pad >= impl->n_vocab) return data_loss("the vocabulary '" + path + "' has no padding token");
    return Tokenizer(std::move(impl));
}

int64_t Tokenizer::vocab_size() const { return impl_->n_vocab; }
int32_t Tokenizer::pad_id() const { return impl_->pad; }

Result<PaddedTokens> Tokenizer::encode_task(const std::string & text, int64_t max_length) const {
    if (text.empty()) return invalid_argument("the task text is empty");
    if (max_length < 1 || max_length > 4096) return invalid_argument("max_length must be in [1, 4096]");
    std::string t = text;
    if (t.back() != '\n') t += '\n';
    std::vector<llama_token> ids(t.size() + 8);
    int32_t n = llama_tokenize(impl_->vocab, t.data(), static_cast<int32_t>(t.size()), ids.data(), static_cast<int32_t>(ids.size()), /*add_special=*/false, /*parse_special=*/false);
    if (n < 0) {
        ids.resize(static_cast<size_t>(-n));
        n = llama_tokenize(impl_->vocab, t.data(), static_cast<int32_t>(t.size()), ids.data(), static_cast<int32_t>(ids.size()), false, false);
    }
    if (n < 0) return internal_error("llama_tokenize failed");
    PaddedTokens out;
    out.truncated = n > max_length;
    out.tokens.assign(static_cast<size_t>(max_length), impl_->pad);
    out.mask.assign(static_cast<size_t>(max_length), 0);
    for (int64_t i = 0; i < std::min<int64_t>(n, max_length); i++) {
        out.tokens[static_cast<size_t>(i)] = ids[static_cast<size_t>(i)];
        out.mask[static_cast<size_t>(i)] = 1;
    }
    return out;
}

}  // namespace dooms
