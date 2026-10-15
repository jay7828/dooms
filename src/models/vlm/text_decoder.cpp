#include "models/vlm/text_decoder.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <mutex>

#include "core/argmax.h"
#include "format/gguf.h"
#include <sys/sysctl.h>

#include "ggml-cpu.h"
#include "llama.h"

namespace dooms {

struct TextDecoder::Impl {
    llama_model * model = nullptr;
    llama_context * ctx = nullptr;
    const llama_vocab * vocab = nullptr;
    TextDecoderOptions options;
    int32_t pos = 0;
    int32_t n_embd = 0, n_vocab = 0;
    bool has_logits = false;
    std::unique_ptr<GgufFile> gguf;           // the model file again, mapped, for the token embedding table
    const GgufTensorInfo * token_embd = nullptr;
    int32_t logits_index = -1;  // row of the last position inside the last decoded chunk
    llama_sampler * sampler = nullptr;  // the current sampling session (a chain), or null
    bool greedy_only = false;           // the session is plain argmax: take the vectorized path instead of building llama.cpp's candidate array per token
    ggml_threadpool * threadpool = nullptr;  // persistent CPU worker threads; without one ggml starts and joins threads for every graph it computes (~3 ms per token)

    ~Impl() {
        if (sampler) llama_sampler_free(sampler);
        if (ctx) llama_free(ctx);
        if (threadpool) ggml_threadpool_free(threadpool);
        if (model) llama_model_free(model);
    }

    Status decode_chunk(llama_batch & batch, int32_t n) {
        batch.n_tokens = n;
        if (llama_decode(ctx, batch) != 0) return internal_error("llama_decode failed");
        logits_index = n - 1;
        has_logits = true;
        pos += n;
        return Status();
    }
};

TextDecoder::TextDecoder(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}
TextDecoder::TextDecoder(TextDecoder &&) noexcept = default;
TextDecoder & TextDecoder::operator=(TextDecoder &&) noexcept = default;
TextDecoder::~TextDecoder() = default;

Result<TextDecoder> TextDecoder::load(const std::string & path, const TextDecoderOptions & o) {
    if (o.n_ctx < 16 || o.n_batch < 1 || o.n_batch > o.n_ctx || o.n_gpu_layers < -1 || o.n_threads < 0) {
        return invalid_argument("n_ctx must be >= 16, n_batch in [1, n_ctx], n_gpu_layers >= -1, n_threads >= 0");
    }
    if (!std::ifstream(path, std::ios::binary)) return not_found("cannot open '" + path + "'");
    static std::once_flag backend_once;
    std::call_once(backend_once, [] {
        // llama.cpp and ggml log every tensor and kernel to stderr; a library should be quiet. Warnings and errors still show; DOOMS_LOG=info|debug restores the rest.
        static const char * level = std::getenv("DOOMS_LOG");
        static const bool verbose = level && (std::string(level) == "info" || std::string(level) == "debug");
        llama_log_set([](ggml_log_level lvl, const char * text, void *) {
            if (verbose || lvl == GGML_LOG_LEVEL_ERROR || lvl == GGML_LOG_LEVEL_WARN) std::fputs(text, stderr);
        }, nullptr);
        llama_backend_init();
    });

    auto impl = std::make_unique<Impl>();
    impl->options = o;
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = o.n_gpu_layers < 0 ? 999 : o.n_gpu_layers;
    impl->model = llama_model_load_from_file(path.c_str(), mp);
    if (!impl->model) return data_loss("llama.cpp could not load '" + path + "' as a GGUF model");
    impl->vocab = llama_model_get_vocab(impl->model);
    impl->n_embd = llama_model_n_embd(impl->model);
    impl->n_vocab = llama_vocab_n_tokens(impl->vocab);
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = static_cast<uint32_t>(o.n_ctx);
    cp.n_batch = static_cast<uint32_t>(o.n_batch);
    cp.n_ubatch = static_cast<uint32_t>(o.n_batch);
    cp.no_perf = true;
    if (o.n_threads > 0) { cp.n_threads = o.n_threads; cp.n_threads_batch = o.n_threads; }
    impl->ctx = llama_init_from_model(impl->model, cp);
    if (!impl->ctx) return internal_error("llama.cpp could not create a context for '" + path + "'");
    {
        // Default thread count: the performance cores (the efficiency cores only slow a synchronous pool down).
        int32_t n = o.n_threads;
        if (n <= 0) {
            int perf = 0;
            size_t len = sizeof perf;
            n = (sysctlbyname("hw.perflevel0.physicalcpu", &perf, &len, nullptr, 0) == 0 && perf > 0) ? perf : 4;
        }
        ggml_threadpool_params tpp = ggml_threadpool_params_default(n);
        impl->threadpool = ggml_threadpool_new(&tpp);
        if (impl->threadpool) {
            llama_set_n_threads(impl->ctx, n, n);
            llama_attach_threadpool(impl->ctx, impl->threadpool, impl->threadpool);
        }
    }
    // The embedding table is read straight from the file (a read-only mapping shared with llama.cpp's own through the page cache); absent or unreadable is not an error.
    auto file = GgufFile::open(path);
    if (file.is_ok()) {
        impl->gguf = std::make_unique<GgufFile>(std::move(file).value());
        auto t = impl->gguf->find_tensor("token_embd.weight");
        if (t.is_ok() && t.value()->shape.rank() == 2 && t.value()->shape.dim(0) == impl->n_vocab && t.value()->shape.dim(1) == impl->n_embd) impl->token_embd = t.value();
    }
    return TextDecoder(std::move(impl));
}

int64_t TextDecoder::embedding_dim() const { return impl_->n_embd; }
int64_t TextDecoder::vocab_size() const { return impl_->n_vocab; }
int32_t TextDecoder::position() const { return impl_->pos; }
int32_t TextDecoder::context_size() const { return impl_->options.n_ctx; }

Result<std::vector<int32_t>> TextDecoder::tokenize(const std::string & text, bool parse_special, bool add_bos) const {
    if (text.empty()) return invalid_argument("cannot tokenize an empty string");
    std::vector<llama_token> tokens(text.size() + 8);
    int32_t n = llama_tokenize(impl_->vocab, text.data(), static_cast<int32_t>(text.size()), tokens.data(), static_cast<int32_t>(tokens.size()), add_bos, parse_special);
    if (n < 0) {
        tokens.resize(static_cast<size_t>(-n));
        n = llama_tokenize(impl_->vocab, text.data(), static_cast<int32_t>(text.size()), tokens.data(), static_cast<int32_t>(tokens.size()), add_bos, parse_special);
    }
    if (n < 0) return internal_error("llama_tokenize failed");
    tokens.resize(static_cast<size_t>(n));
    return std::vector<int32_t>(tokens.begin(), tokens.end());
}

std::string TextDecoder::piece(int32_t token) const {
    if (token < 0 || token >= impl_->n_vocab) return std::string();
    char buf[256];
    int32_t n = llama_token_to_piece(impl_->vocab, token, buf, sizeof buf, 0, true);
    if (n < 0) {
        std::string big(static_cast<size_t>(-n), '\0');
        n = llama_token_to_piece(impl_->vocab, token, big.data(), static_cast<int32_t>(big.size()), 0, true);
        return n < 0 ? std::string() : big.substr(0, static_cast<size_t>(n));
    }
    return std::string(buf, static_cast<size_t>(n));
}

bool TextDecoder::is_end_of_generation(int32_t token) const { return token >= 0 && token < impl_->n_vocab && llama_vocab_is_eog(impl_->vocab, token); }

void TextDecoder::reset() {
    llama_memory_clear(llama_get_memory(impl_->ctx), true);
    impl_->pos = 0;
    impl_->has_logits = false;
}

Status TextDecoder::eval_tokens(const std::vector<int32_t> & tokens) {
    Impl & m = *impl_;
    if (tokens.empty()) return invalid_argument("no tokens to evaluate");
    for (int32_t t : tokens) if (t < 0 || t >= m.n_vocab) return invalid_argument("token " + std::to_string(t) + " is outside the vocabulary [0, " + std::to_string(m.n_vocab) + ")");
    if (static_cast<int64_t>(m.pos) + static_cast<int64_t>(tokens.size()) > m.options.n_ctx) return resource_exhausted("the context of " + std::to_string(m.options.n_ctx) + " positions would overflow");
    llama_batch batch = llama_batch_init(m.options.n_batch, 0, 1);
    Status st;
    for (size_t start = 0; start < tokens.size() && st.is_ok(); start += static_cast<size_t>(m.options.n_batch)) {
        const int32_t n = static_cast<int32_t>(std::min<size_t>(static_cast<size_t>(m.options.n_batch), tokens.size() - start));
        for (int32_t i = 0; i < n; i++) {
            batch.token[i] = tokens[start + static_cast<size_t>(i)];
            batch.pos[i] = m.pos + i;
            batch.n_seq_id[i] = 1;
            batch.seq_id[i][0] = 0;
            batch.logits[i] = (i == n - 1);
        }
        st = m.decode_chunk(batch, n);
    }
    llama_batch_free(batch);
    return st;
}

Status TextDecoder::eval_embeddings(const float * embeddings, int64_t n_positions) {
    Impl & m = *impl_;
    if (!embeddings || n_positions < 1) return invalid_argument("no embeddings to evaluate");
    const int64_t total = n_positions * m.n_embd;
    for (int64_t i = 0; i < total; i++) if (!std::isfinite(embeddings[i])) return invalid_argument("the embeddings contain NaN or Inf");
    if (static_cast<int64_t>(m.pos) + n_positions > m.options.n_ctx) return resource_exhausted("the context of " + std::to_string(m.options.n_ctx) + " positions would overflow");
    llama_batch batch = llama_batch_init(m.options.n_batch, m.n_embd, 1);
    Status st;
    for (int64_t start = 0; start < n_positions && st.is_ok(); start += m.options.n_batch) {
        const int32_t n = static_cast<int32_t>(std::min<int64_t>(m.options.n_batch, n_positions - start));
        std::copy(embeddings + start * m.n_embd, embeddings + (start + n) * m.n_embd, batch.embd);
        for (int32_t i = 0; i < n; i++) {
            batch.pos[i] = m.pos + i;
            batch.n_seq_id[i] = 1;
            batch.seq_id[i][0] = 0;
            batch.logits[i] = (i == n - 1);
        }
        st = m.decode_chunk(batch, n);
    }
    llama_batch_free(batch);
    return st;
}

Result<std::vector<float>> TextDecoder::embed_tokens(const std::vector<int32_t> & tokens) const {
    const Impl & m = *impl_;
    if (tokens.empty()) return invalid_argument("no tokens to embed");
    if (!m.token_embd || (m.token_embd->dtype != DType::F32 && m.token_embd->dtype != DType::F16 && m.token_embd->dtype != DType::BF16)) {
        return unsupported("the token embedding table is missing or not f32/f16/bf16");
    }
    const uint8_t * base = m.gguf->tensor_data(*m.token_embd);
    const size_t elem = m.token_embd->dtype == DType::F32 ? 4 : 2;
    std::vector<float> out(tokens.size() * static_cast<size_t>(m.n_embd));
    for (size_t t = 0; t < tokens.size(); t++) {
        if (tokens[t] < 0 || tokens[t] >= m.n_vocab) return invalid_argument("token " + std::to_string(tokens[t]) + " is outside the vocabulary [0, " + std::to_string(m.n_vocab) + ")");
        const uint8_t * row = base + static_cast<size_t>(tokens[t]) * static_cast<size_t>(m.n_embd) * elem;
        float * dst = out.data() + t * static_cast<size_t>(m.n_embd);
        for (int32_t i = 0; i < m.n_embd; i++) {
            if (m.token_embd->dtype == DType::F32) { std::memcpy(&dst[i], row + static_cast<size_t>(i) * 4, 4); }
            else if (m.token_embd->dtype == DType::F16) { __fp16 h; std::memcpy(&h, row + static_cast<size_t>(i) * 2, 2); dst[i] = static_cast<float>(h); }
            else { uint16_t b; std::memcpy(&b, row + static_cast<size_t>(i) * 2, 2); const uint32_t bits = static_cast<uint32_t>(b) << 16; std::memcpy(&dst[i], &bits, 4); }
        }
    }
    return out;
}

Result<std::vector<float>> TextDecoder::logits() const {
    const Impl & m = *impl_;
    if (!m.has_logits) return invalid_argument("nothing has been evaluated yet");
    const float * l = llama_get_logits_ith(m.ctx, m.logits_index);
    if (!l) return internal_error("llama.cpp returned no logits");
    return std::vector<float>(l, l + m.n_vocab);
}

Status TextDecoder::start_sampling(const SamplerOptions & o) {
    Impl & m = *impl_;
    if (!(o.temperature >= 0.f) || o.top_k < 0 || !(o.top_p > 0.f && o.top_p <= 1.f) || !(o.repeat_penalty > 0.f)) {
        return invalid_argument("sampler: temperature >= 0, top_k >= 0, top_p in (0, 1], repeat_penalty > 0 are required");
    }
    if (m.sampler) { llama_sampler_free(m.sampler); m.sampler = nullptr; }
    llama_sampler * chain = llama_sampler_chain_init(llama_sampler_chain_default_params());
    if (o.repeat_penalty != 1.f) llama_sampler_chain_add(chain, llama_sampler_init_penalties(m.n_vocab, 64, o.repeat_penalty, 0.f, 0.f));
    if (o.temperature == 0.f) {
        llama_sampler_chain_add(chain, llama_sampler_init_greedy());
    } else {
        if (o.top_k > 0) llama_sampler_chain_add(chain, llama_sampler_init_top_k(o.top_k));
        if (o.top_p < 1.f) llama_sampler_chain_add(chain, llama_sampler_init_top_p(o.top_p, 1));
        llama_sampler_chain_add(chain, llama_sampler_init_temp(o.temperature));
        llama_sampler_chain_add(chain, llama_sampler_init_dist(o.seed));
    }
    m.sampler = chain;
    m.greedy_only = o.temperature == 0.f && o.repeat_penalty == 1.f;
    return Status();
}

Result<int32_t> TextDecoder::sample_next() {
    Impl & m = *impl_;
    if (!m.has_logits) return invalid_argument("nothing has been evaluated yet");
    if (!m.sampler) return invalid_argument("start_sampling() was not called");
    if (m.greedy_only) return greedy_next();
    return static_cast<int32_t>(llama_sampler_sample(m.sampler, m.ctx, m.logits_index));
}

Result<std::string> TextDecoder::apply_chat_template(const std::vector<ChatMessage> & messages, bool add_generation_prompt) const {
    const Impl & m = *impl_;
    if (messages.empty()) return invalid_argument("no messages");
    std::vector<llama_chat_message> chat;
    chat.reserve(messages.size());
    size_t chars = 0;
    for (const ChatMessage & msg : messages) {
        if (msg.role.empty()) return invalid_argument("a message has an empty role");
        chat.push_back({msg.role.c_str(), msg.content.c_str()});
        chars += msg.role.size() + msg.content.size();
    }
    const char * tmpl = llama_model_chat_template(m.model, nullptr);
    if (!tmpl) return unsupported("the model file has no chat template");
    std::vector<char> buf(2 * chars + 256);
    int32_t n = llama_chat_apply_template(tmpl, chat.data(), chat.size(), add_generation_prompt, buf.data(), static_cast<int32_t>(buf.size()));
    if (n > static_cast<int32_t>(buf.size())) {
        buf.resize(static_cast<size_t>(n));
        n = llama_chat_apply_template(tmpl, chat.data(), chat.size(), add_generation_prompt, buf.data(), static_cast<int32_t>(buf.size()));
    }
    if (n < 0) return unsupported("llama.cpp does not support this model's chat template");
    return std::string(buf.data(), static_cast<size_t>(n));
}

Result<int32_t> TextDecoder::greedy_next() const {
    const Impl & m = *impl_;
    if (!m.has_logits) return invalid_argument("nothing has been evaluated yet");
    const float * l = llama_get_logits_ith(m.ctx, m.logits_index);
    if (!l) return internal_error("llama.cpp returned no logits");
    return static_cast<int32_t>(argmax_f32(l, static_cast<size_t>(m.n_vocab)));
}

}  // namespace dooms
