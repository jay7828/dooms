#include "pipeline/vlm_pipeline.h"

#include <chrono>
#include <cmath>
#include <future>

#include "format/gguf.h"
#include "pipeline/native_llama.h"

namespace dooms {

namespace {

using Clock = std::chrono::steady_clock;
double ms_since(Clock::time_point t) { return std::chrono::duration<double, std::milli>(Clock::now() - t).count(); }
constexpr int kMaxNewTokens = 4096;

template <class T>
Result<T> unwrap(Result<T> r, const char * what) {
    if (!r.is_ok()) return Status(r.status().code(), std::string(what) + ": " + r.status().message());
    return r;
}

}  // namespace

struct VlmPipeline::Impl {
    std::unique_ptr<CoreMlModel> vision;
    std::unique_ptr<TextDecoder> text;
    std::unique_ptr<MetalLlamaDecoder> native;  // set when options.text_engine == "metal"; `text` then only tokenizes
    int32_t queue_depth = 1;
    VlmImageOptions image_options;
    std::string prefix, infix, suffix;
    int64_t n_image = 0, embedding_dim = 0, image_token_id = 0;
    Shape input_shape, output_shape;
    std::vector<float> pixels, embeddings;
};

VlmPipeline::VlmPipeline(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}
VlmPipeline::VlmPipeline(VlmPipeline &&) noexcept = default;
VlmPipeline & VlmPipeline::operator=(VlmPipeline &&) noexcept = default;
VlmPipeline::~VlmPipeline() = default;

int64_t VlmPipeline::n_image_tokens() const { return impl_->n_image; }
const char * VlmPipeline::text_engine_used() const { return impl_->native ? "metal" : "llama.cpp"; }
int64_t VlmPipeline::embedding_dim() const { return impl_->embedding_dim; }
const VlmImageOptions & VlmPipeline::image_options() const { return impl_->image_options; }

Result<VlmPipeline> VlmPipeline::create(const VlmPaths & paths, const VlmOptions & options) {
    auto impl = std::make_unique<Impl>();
    auto cfg = GgufFile::open(paths.config);
    if (!cfg.is_ok()) return cfg.status();
    const GgufFile & c = cfg.value();
    auto arch = c.get_string("general.architecture");
    if (!arch.is_ok()) return arch.status();
    if (arch.value() != "vlm_config") return invalid_argument("'" + paths.config + "' is not a vlm-config file (architecture '" + arch.value() + "')");
    auto family = unwrap(c.get_string("vlm_config.family"), "vlm_config.family");
    if (!family.is_ok()) return family.status();
    if (family.value() != "idefics3") return unsupported("model family '" + family.value() + "' is not supported (only idefics3)");
    auto size = c.get_uint("vlm_config.image_size"), n_image = c.get_uint("vlm_config.n_image_tokens"), tok_id = c.get_uint("vlm_config.image_token_id"), dim = c.get_uint("vlm_config.embedding_dim");
    if (!size.is_ok()) return size.status();
    if (!n_image.is_ok()) return n_image.status();
    if (!tok_id.is_ok()) return tok_id.status();
    if (!dim.is_ok()) return dim.status();
    if (size.value() < 1 || size.value() > 8192 || n_image.value() < 1 || n_image.value() > 4096 || dim.value() < 1 || dim.value() > 65536) return data_loss("vlm-config has values outside their valid ranges");
    impl->image_options.image_size = static_cast<int64_t>(size.value());
    impl->n_image = static_cast<int64_t>(n_image.value());
    impl->embedding_dim = static_cast<int64_t>(dim.value());
    impl->image_token_id = static_cast<int64_t>(tok_id.value());
    for (int ch = 0; ch < 3; ch++) {
        auto mean = c.get_float("vlm_config.image_mean_" + std::to_string(ch)), sd = c.get_float("vlm_config.image_std_" + std::to_string(ch));
        if (!mean.is_ok()) return mean.status();
        if (!sd.is_ok()) return sd.status();
        impl->image_options.mean[ch] = mean.value();
        impl->image_options.std[ch] = sd.value();
    }
    for (const char * key : {"prefix", "infix", "suffix"}) {
        auto v = c.get_string(std::string("vlm_config.") + key);
        if (!v.is_ok()) return v.status();
        (std::string(key) == "prefix" ? impl->prefix : std::string(key) == "infix" ? impl->infix : impl->suffix) = v.value();
    }
    auto resample = c.get_string("vlm_config.resample");
    if (!resample.is_ok()) return resample.status();
    if (resample.value() != "lanczos") return unsupported("resampling '" + resample.value() + "' is not implemented (only lanczos)");

    auto units = parse_coreml_units(options.vision_units);
    if (!units.is_ok()) return units.status();
    auto vision = CoreMlModel::load(paths.vision, units.value());
    if (!vision.is_ok()) return vision.status();
    impl->vision = std::make_unique<CoreMlModel>(std::move(vision).value());
    auto in_shape = Shape::create({1, 3, impl->image_options.image_size, impl->image_options.image_size});
    auto out_shape = Shape::create({1, impl->n_image, impl->embedding_dim});
    if (!in_shape.is_ok() || !out_shape.is_ok()) return internal_error("bad vision shapes");
    impl->input_shape = in_shape.value();
    impl->output_shape = out_shape.value();
    const auto & vin = impl->vision->inputs();
    const auto & vout = impl->vision->outputs();
    if (vin.size() != 1 || vin[0].name != "pixel_values" || vin[0].shape != impl->input_shape) {
        return invalid_argument("the vision package must take one input 'pixel_values' of shape " + impl->input_shape.to_string());
    }
    if (vout.size() != 1 || vout[0].name != "image_embeddings" || vout[0].shape != impl->output_shape) {
        return invalid_argument("the vision package must produce one output 'image_embeddings' of shape " + impl->output_shape.to_string());
    }

    if (options.text_engine != "auto" && options.text_engine != "llama.cpp" && options.text_engine != "metal") {
        return invalid_argument("text_engine must be \"auto\", \"metal\" or \"llama.cpp\", got \"" + options.text_engine + "\"");
    }
    if (options.metal_queue_depth < 1) return invalid_argument("metal_queue_depth must be >= 1");
    TextDecoderOptions text_options = options.text;
    if (options.text_engine != "llama.cpp") {
        NativeLlamaOptions no;
        no.n_ctx = options.text.n_ctx;
        no.n_batch = options.text.n_batch;
        no.spin_wait = options.metal_spin_wait;
        auto native = load_native_llama(paths.text, no);
        // "auto" falls back to llama.cpp only for files the own decoder does not run; a missing or broken file is an error either way.
        if (!native.is_ok() && (options.text_engine == "metal" || native.status().code() != Code::Unsupported)) {
            return Status(native.status().code(), "text_engine \"" + options.text_engine + "\": " + native.status().message());
        }
        if (native.is_ok()) impl->native = std::move(native).value();
    }
    if (impl->native) {
        impl->queue_depth = options.metal_queue_depth;
        text_options.vocab_only = true;  // llama.cpp only tokenizes: no weights, no context, no worker threads competing with preprocessing
    }
    auto text = TextDecoder::load(paths.text, text_options);
    if (!text.is_ok()) return text.status();
    impl->text = std::make_unique<TextDecoder>(std::move(text).value());
    if (impl->text->embedding_dim() != impl->embedding_dim) {
        return invalid_argument("the language model has width " + std::to_string(impl->text->embedding_dim()) + " but the image embeddings have width " + std::to_string(impl->embedding_dim));
    }
    // The image token of the layout must be the vocabulary's: a text model of another family would read the embeddings at the wrong places.
    auto image_token = c.get_string("vlm_config.image_token");
    if (!image_token.is_ok()) return image_token.status();
    auto parsed = impl->text->tokenize(image_token.value(), true);
    if (!parsed.is_ok() || parsed.value().size() != 1 || parsed.value()[0] != impl->image_token_id) {
        return invalid_argument("the language model does not map '" + image_token.value() + "' to the single token " + std::to_string(impl->image_token_id) + " the config expects");
    }
    impl->pixels.resize(static_cast<size_t>(3 * impl->image_options.image_size * impl->image_options.image_size));
    impl->embeddings.resize(static_cast<size_t>(impl->n_image * impl->embedding_dim));
    return VlmPipeline(std::move(impl));
}

Status VlmPipeline::encode_image(const RgbImage & image, std::vector<float> * embeddings, VlmTimings * timings) {
    Impl & m = *impl_;
    if (!embeddings) return invalid_argument("null output");
    auto t = Clock::now();
    auto pixels = idefics3_pixel_values(image, m.image_options);
    if (!pixels.is_ok()) return pixels.status();
    m.pixels = std::move(pixels).value();
    const double pre_ms = ms_since(t);
    t = Clock::now();
    embeddings->assign(static_cast<size_t>(m.n_image * m.embedding_dim), 0.f);
    auto in = TensorView::create(m.pixels.data(), m.input_shape, DType::F32);
    auto out = TensorView::create(embeddings->data(), m.output_shape, DType::F32);
    if (!in.is_ok() || !out.is_ok()) return internal_error("could not bind the vision buffers");
    Status st = m.vision->predict({{"pixel_values", in.value()}}, {{"image_embeddings", out.value()}});
    if (!st.is_ok()) return st;
    if (timings) { timings->preprocess_ms = pre_ms; timings->vision_ms = ms_since(t); }
    return Status();
}

Result<VlmGeneration> VlmPipeline::generate(const RgbImage & image, const std::string & prompt, int max_new_tokens, const std::function<bool(const std::string &)> & on_text, VlmTimings * timings) {
    if (prompt.empty()) return invalid_argument("the prompt is empty");
    std::vector<float> emb;
    VlmTimings local;
    Status st = encode_image(image, &emb, &local);
    if (!st.is_ok()) return st;
    auto r = generate_from_embeddings(emb, prompt, max_new_tokens, on_text, &local);
    if (timings) *timings = local;
    return r;
}

namespace {

// The Metal decoder's path: the whole prompt as one embedding sequence, then pipelined greedy decoding (the GPU reads each step's input token itself).
Result<VlmGeneration> generate_native(TextDecoder & vocab, MetalLlamaDecoder & native, int64_t embedding_dim, int32_t queue_depth, const std::vector<float> & image, const std::vector<int32_t> & before, const std::vector<int32_t> & after,
                                      int max_new_tokens, const std::function<bool(const std::string &)> & on_text, VlmTimings * timings, VlmTimings tm,
                                      Clock::time_point t0) {
    auto head = vocab.embed_tokens(before);
    auto tail = vocab.embed_tokens(after);
    if (!head.is_ok()) return head.status();
    if (!tail.is_ok()) return tail.status();
    std::vector<float> all;
    all.reserve(head.value().size() + image.size() + tail.value().size());
    all.insert(all.end(), head.value().begin(), head.value().end());
    all.insert(all.end(), image.begin(), image.end());
    all.insert(all.end(), tail.value().begin(), tail.value().end());
    native.reset();
    Status st = native.eval_embeddings(all.data(), static_cast<int32_t>(all.size() / static_cast<size_t>(embedding_dim)));
    if (!st.is_ok()) return st;
    VlmGeneration gen;
    auto t_decode = Clock::now();
    st = native.generate(max_new_tokens, queue_depth, [&](int32_t token) {
        if (gen.tokens.empty()) {
            // The first token is ready once the prefill has run: the GPU works asynchronously, so this is where its time shows.
            tm.prefill_ms = tm.first_token_ms = ms_since(t0);
            t_decode = Clock::now();
        }
        gen.tokens.push_back(token);
        if (vocab.is_end_of_generation(token)) { gen.stopped = true; return false; }
        const std::string piece = vocab.piece(token);
        gen.text += piece;
        if (on_text && !on_text(piece)) { gen.cancelled = true; return false; }
        return true;
    });
    if (!st.is_ok()) return st;
    tm.generated_tokens = static_cast<int64_t>(gen.tokens.size());
    tm.decode_ms = ms_since(t_decode);
    if (timings) *timings = tm;
    return gen;
}

}  // namespace

Result<VlmGeneration> VlmPipeline::generate_from_embeddings(const std::vector<float> & embeddings, const std::string & prompt, int max_new_tokens,
                                                            const std::function<bool(const std::string &)> & on_text, VlmTimings * timings) {
    Impl & m = *impl_;
    if (prompt.empty()) return invalid_argument("the prompt is empty");
    if (max_new_tokens < 1 || max_new_tokens > kMaxNewTokens) return invalid_argument("max_new_tokens must be in [1, " + std::to_string(kMaxNewTokens) + "]");
    if (static_cast<int64_t>(embeddings.size()) != m.n_image * m.embedding_dim) return invalid_argument("expected " + std::to_string(m.n_image * m.embedding_dim) + " embedding values, got " + std::to_string(embeddings.size()));
    VlmTimings tm = timings ? *timings : VlmTimings();
    const auto t0 = Clock::now();

    auto before = m.text->tokenize(m.prefix, true);
    auto after = m.text->tokenize(m.infix + prompt + m.suffix, true);
    if (!before.is_ok()) return before.status();
    if (!after.is_ok()) return after.status();
    tm.prompt_tokens = static_cast<int64_t>(before.value().size()) + m.n_image + static_cast<int64_t>(after.value().size());
    if (tm.prompt_tokens + max_new_tokens > m.text->context_size()) {
        return resource_exhausted("the prompt (" + std::to_string(tm.prompt_tokens) + " tokens including the image) plus " + std::to_string(max_new_tokens) + " new tokens exceed the context of " + std::to_string(m.text->context_size()));
    }
    if (m.native) return generate_native(*m.text, *m.native, m.embedding_dim, m.queue_depth, embeddings, before.value(), after.value(), max_new_tokens, on_text, timings, tm, t0);
    m.text->reset();
    Status st;
    // One model call for the whole prompt when the token embedding table can be read (f32/f16/bf16 models): the prompt is then a single embedding sequence
    // [prefix tokens | image | rest]. Quantized embedding tables fall back to one call per segment (same result, more launch overhead).
    auto head = m.text->embed_tokens(before.value());
    auto tail = m.text->embed_tokens(after.value());
    if (head.is_ok() && tail.is_ok()) {
        std::vector<float> all;
        all.reserve(head.value().size() + embeddings.size() + tail.value().size());
        all.insert(all.end(), head.value().begin(), head.value().end());
        all.insert(all.end(), embeddings.begin(), embeddings.end());
        all.insert(all.end(), tail.value().begin(), tail.value().end());
        st = m.text->eval_embeddings(all.data(), static_cast<int64_t>(before.value().size()) + m.n_image + static_cast<int64_t>(after.value().size()));
    } else {
        st = m.text->eval_tokens(before.value());
        if (st.is_ok()) st = m.text->eval_embeddings(embeddings.data(), m.n_image);
        if (st.is_ok()) st = m.text->eval_tokens(after.value());
    }
    if (!st.is_ok()) return st;
    // The GPU runs asynchronously: reading the first logits is what waits for the prefill, so the prefill is timed up to and including the first sample.
    auto first = m.text->greedy_next();
    if (!first.is_ok()) return first.status();
    tm.prefill_ms = ms_since(t0);

    VlmGeneration gen;
    auto t_first = Clock::now();
    auto t_decode = t_first;
    for (int i = 0; i < max_new_tokens; i++) {
        auto next = i == 0 ? first : m.text->greedy_next();
        if (!next.is_ok()) return next.status();
        if (i == 0) { tm.first_token_ms = ms_since(t0); t_decode = Clock::now(); }
        gen.tokens.push_back(next.value());
        if (m.text->is_end_of_generation(next.value())) { gen.stopped = true; break; }
        const std::string piece = m.text->piece(next.value());
        gen.text += piece;
        if (on_text && !on_text(piece)) { gen.cancelled = true; break; }
        if (i + 1 < max_new_tokens) {
            st = m.text->eval_tokens({next.value()});
            if (!st.is_ok()) return st;
        }
    }
    tm.generated_tokens = static_cast<int64_t>(gen.tokens.size());
    tm.decode_ms = ms_since(t_decode);
    if (timings) *timings = tm;
    return gen;
}

Result<std::vector<VlmGeneration>> VlmPipeline::generate_many(const std::vector<VlmRequest> & requests, int max_new_tokens, double * wall_ms) {
    for (const VlmRequest & r : requests) {
        if (!r.image) return invalid_argument("a request has no image");
        if (r.prompt.empty()) return invalid_argument("a request has an empty prompt");
    }
    const auto t0 = Clock::now();
    std::vector<VlmGeneration> answers;
    // Vision (worker thread: Core ML only) and decoding (this thread: llama.cpp only) touch disjoint state, so they can run at the same time.
    auto encode = [this](const RgbImage * image) {
        std::vector<float> emb;
        Status st = encode_image(*image, &emb);
        return std::make_pair(std::move(st), std::move(emb));
    };
    std::future<std::pair<Status, std::vector<float>>> pending;
    if (!requests.empty()) pending = std::async(std::launch::async, encode, requests[0].image);
    for (size_t i = 0; i < requests.size(); i++) {
        auto encoded = pending.get();
        if (i + 1 < requests.size()) pending = std::async(std::launch::async, encode, requests[i + 1].image);
        Status err = encoded.first;
        if (err.is_ok()) {
            auto r = generate_from_embeddings(encoded.second, requests[i].prompt, max_new_tokens);
            if (r.is_ok()) { answers.push_back(std::move(r).value()); continue; }
            err = r.status();
        }
        if (pending.valid()) pending.wait();  // never leave the worker running against a destroyed pipeline
        return err;
    }
    if (wall_ms) *wall_ms = ms_since(t0);
    return answers;
}

}  // namespace dooms
