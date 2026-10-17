#include "pipeline/whisper_pipeline.h"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <map>

#include "backend/coreml/coreml_model.h"
#include "backend/metal/metal_whisper_decoder.h"
#include "format/gguf.h"
#include "pipeline/audio_io.h"
#include "pipeline/whisper_features.h"

namespace dooms {

namespace {

using Clock = std::chrono::steady_clock;
double ms_since(Clock::time_point t) { return std::chrono::duration<double, std::milli>(Clock::now() - t).count(); }

constexpr const char * kArch = "whisper";
constexpr const char * kEncoderInput = "input_features";
constexpr const char * kEncoderOutput = "encoder_out";

// GPT-2's byte <-> unicode table (the vocabulary's pieces spell bytes as these code points): the inverse maps a code point back to its byte.
std::map<uint32_t, uint8_t> byte_decoder() {
    std::vector<int> bs;
    for (int b = '!'; b <= '~'; b++) bs.push_back(b);
    for (int b = 0xA1; b <= 0xAC; b++) bs.push_back(b);
    for (int b = 0xAE; b <= 0xFF; b++) bs.push_back(b);
    std::vector<int> cs = bs;
    int n = 0;
    for (int b = 0; b < 256; b++) {
        if (std::find(bs.begin(), bs.end(), b) == bs.end()) { bs.push_back(b); cs.push_back(256 + n); n++; }
    }
    std::map<uint32_t, uint8_t> m;
    for (size_t i = 0; i < bs.size(); i++) m[static_cast<uint32_t>(cs[i])] = static_cast<uint8_t>(bs[i]);
    return m;
}

// UTF-8 -> code points (the pieces are valid UTF-8 written by the converter).
std::vector<uint32_t> code_points(const std::string & s) {
    std::vector<uint32_t> out;
    for (size_t i = 0; i < s.size();) {
        const unsigned char c = static_cast<unsigned char>(s[i]);
        const int len = c < 0x80 ? 1 : c < 0xE0 ? 2 : c < 0xF0 ? 3 : 4;
        uint32_t cp = len == 1 ? c : len == 2 ? (c & 0x1F) : len == 3 ? (c & 0x0F) : (c & 0x07);
        for (int k = 1; k < len && i + static_cast<size_t>(k) < s.size(); k++) cp = (cp << 6) | (static_cast<unsigned char>(s[i + static_cast<size_t>(k)]) & 0x3F);
        out.push_back(cp);
        i += static_cast<size_t>(len);
    }
    return out;
}

const GgufValue * kv(const GgufFile & f, const std::string & key) {
    auto it = f.metadata().find(key);
    return it == f.metadata().end() ? nullptr : &it->second;
}

// An integer array element, whichever integer type the writer chose (gguf-py writes Python ints as I32).
int64_t as_int(const GgufValue & v) {
    switch (v.type) {
        case GgufValueType::I8: case GgufValueType::I16: case GgufValueType::I32: case GgufValueType::I64: return v.i;
        default: return static_cast<int64_t>(v.u);
    }
}

}  // namespace

struct WhisperPipeline::Impl {
    WhisperOptions options;
    std::unique_ptr<CoreMlModel> encoder;
    std::unique_ptr<MetalWhisperDecoder> decoder;
    std::unique_ptr<WhisperFeatures> features;
    int32_t sample_rate = 0, chunk = 0, n_mels = 0, n_audio_ctx = 0, n_state = 0, n_text_ctx = 0, n_text_tokens = 0;
    int32_t sot = 0, eot = 0, notimestamps = 0, lang = 0, task = 0;
    std::vector<std::string> pieces;
    std::map<uint32_t, uint8_t> bytes;
    std::vector<float> mel, states;
};

WhisperPipeline::WhisperPipeline(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}
WhisperPipeline::WhisperPipeline(WhisperPipeline &&) noexcept = default;
WhisperPipeline & WhisperPipeline::operator=(WhisperPipeline &&) noexcept = default;
WhisperPipeline::~WhisperPipeline() = default;

int32_t WhisperPipeline::sample_rate() const { return impl_->sample_rate; }
int32_t WhisperPipeline::window_samples() const { return impl_->chunk; }
int32_t WhisperPipeline::n_mels() const { return impl_->n_mels; }
int32_t WhisperPipeline::n_frames() const { return impl_->features->n_frames(); }
MetalWhisperDecoder & WhisperPipeline::decoder() { return *impl_->decoder; }
bool WhisperPipeline::is_end_of_text(int32_t token) const { return token == impl_->eot; }

Result<WhisperPipeline> WhisperPipeline::create(const WhisperPaths & paths, const WhisperOptions & o) {
    if (o.max_tokens_per_window < 1 || o.queue_depth < 1) return invalid_argument("max_tokens_per_window and queue_depth must be >= 1");
    if (o.task != "transcribe" && o.task != "translate") return invalid_argument("task must be \"transcribe\" or \"translate\", got \"" + o.task + "\"");
    auto file = GgufFile::open(paths.decoder);
    if (!file.is_ok()) return file.status();
    const GgufFile & f = file.value();
    auto arch = f.get_string("general.architecture");
    if (!arch.is_ok() || arch.value() != kArch) return invalid_argument("'" + paths.decoder + "' is not a whisper decoder file (general.architecture)");
    auto u = [&](const std::string & key) -> Result<int32_t> {
        auto v = f.get_uint(std::string(kArch) + "." + key);
        if (!v.is_ok()) return invalid_argument("'" + paths.decoder + "' lacks " + kArch + "." + key);
        if (v.value() > static_cast<uint64_t>(INT32_MAX)) return data_loss(std::string(kArch) + "." + key + " is out of range");
        return static_cast<int32_t>(v.value());
    };
    auto impl = std::make_unique<Impl>();
    Impl & m = *impl;
    m.options = o;
    WhisperDecoderConfig dc;
    int32_t n_fft = 0, hop = 0;
    for (auto [key, dst] : std::initializer_list<std::pair<const char *, int32_t *>>{
             {"n_mels", &m.n_mels}, {"n_audio_ctx", &m.n_audio_ctx}, {"n_text_state", &m.n_state}, {"n_text_ctx", &m.n_text_ctx}, {"n_text_head", &dc.n_head},
             {"n_text_layer", &dc.n_layer}, {"n_text_ff", &dc.n_ff}, {"n_vocab", &dc.n_vocab}, {"sample_rate", &m.sample_rate}, {"n_fft", &n_fft},
             {"hop_length", &hop}, {"chunk_samples", &m.chunk}, {"token.sot", &m.sot}, {"token.eot", &m.eot}, {"token.notimestamps", &m.notimestamps},
             {"n_text_tokens", &m.n_text_tokens}}) {
        auto v = u(key);
        if (!v.is_ok()) return v.status();
        *dst = v.value();
    }
    auto task_id = u(std::string("token.") + o.task);
    if (!task_id.is_ok()) return task_id.status();
    m.task = task_id.value();
    auto eps = f.get_float(std::string(kArch) + ".layer_norm_eps");
    if (!eps.is_ok()) return invalid_argument("'" + paths.decoder + "' lacks whisper.layer_norm_eps");
    dc.n_state = m.n_state;
    dc.n_text_ctx = m.n_text_ctx;
    dc.n_audio_ctx = m.n_audio_ctx;
    dc.ln_eps = static_cast<float>(eps.value());
    dc.spin_wait = o.spin_wait;
    if (o.max_tokens_per_window > m.n_text_ctx / 2) return invalid_argument("max_tokens_per_window must be at most n_text_ctx / 2 = " + std::to_string(m.n_text_ctx / 2));

    // Language token.
    const GgufValue * codes = kv(f, "whisper.language.codes");
    const GgufValue * ids = kv(f, "whisper.language.ids");
    if (!codes || !ids || codes->type != GgufValueType::Array || ids->type != GgufValueType::Array || codes->items.size() != ids->items.size()) {
        return invalid_argument("'" + paths.decoder + "' has no consistent whisper.language.codes / ids");
    }
    m.lang = -1;
    for (size_t i = 0; i < codes->items.size(); i++) if (codes->items[i].s == o.language) m.lang = static_cast<int32_t>(as_int(ids->items[i]));
    if (m.lang < 0) return invalid_argument("the model does not know language \"" + o.language + "\"");

    // Vocabulary.
    const GgufValue * toks = kv(f, "whisper.tokens");
    if (!toks || toks->type != GgufValueType::Array || static_cast<int32_t>(toks->items.size()) != dc.n_vocab) {
        return invalid_argument("'" + paths.decoder + "' has no whisper.tokens of n_vocab entries");
    }
    for (const GgufValue & t : toks->items) m.pieces.push_back(t.s);
    m.bytes = byte_decoder();

    // Suppression lists.
    auto int_list = [&](const char * key) {
        std::vector<int32_t> out;
        if (const GgufValue * v = kv(f, key); v && v->type == GgufValueType::Array) {
            for (const GgufValue & e : v->items) out.push_back(static_cast<int32_t>(as_int(e)));
        }
        return out;
    };
    const std::vector<int32_t> suppress = int_list("whisper.suppress_tokens"), begin = int_list("whisper.begin_suppress_tokens");

    // Weights.
    auto ref = [&](const std::string & name, WeightRef * out) -> Status {
        auto t = f.find_tensor(name);
        if (!t.is_ok()) return invalid_argument("'" + paths.decoder + "' has no tensor " + name);
        const GgufTensorInfo & info = *t.value();
        out->data = f.tensor_data(info);
        out->dtype = info.dtype;
        if (info.shape.rank() == 1) { out->rows = 1; out->cols = info.shape.dim(0); }
        else if (info.shape.rank() == 2) { out->rows = info.shape.dim(0); out->cols = info.shape.dim(1); }
        else return invalid_argument("tensor " + name + " has rank " + std::to_string(info.shape.rank()));
        return Status();
    };
    WhisperDecoderWeights w;
    for (auto [name, dst] : std::initializer_list<std::pair<const char *, WeightRef *>>{
             {"token_embd.weight", &w.token_embd}, {"pos_embd.weight", &w.pos_embd}, {"ln.weight", &w.ln_w}, {"ln.bias", &w.ln_b}}) {
        if (Status st = ref(name, dst); !st.is_ok()) return st;
    }
    w.layers.resize(static_cast<size_t>(dc.n_layer));
    for (int32_t l = 0; l < dc.n_layer; l++) {
        WhisperLayerWeights & L = w.layers[static_cast<size_t>(l)];
        const std::string b = "blk." + std::to_string(l) + ".";
        for (auto [name, dst] : std::initializer_list<std::pair<const char *, WeightRef *>>{
                 {"attn_ln.weight", &L.attn_ln_w}, {"attn_ln.bias", &L.attn_ln_b}, {"attn_q.weight", &L.attn_q_w}, {"attn_q.bias", &L.attn_q_b}, {"attn_k.weight", &L.attn_k_w},
                 {"attn_v.weight", &L.attn_v_w}, {"attn_v.bias", &L.attn_v_b}, {"attn_out.weight", &L.attn_out_w}, {"attn_out.bias", &L.attn_out_b},
                 {"cross_ln.weight", &L.cross_ln_w}, {"cross_ln.bias", &L.cross_ln_b}, {"cross_q.weight", &L.cross_q_w}, {"cross_q.bias", &L.cross_q_b},
                 {"cross_k.weight", &L.cross_k_w}, {"cross_v.weight", &L.cross_v_w}, {"cross_v.bias", &L.cross_v_b}, {"cross_out.weight", &L.cross_out_w},
                 {"cross_out.bias", &L.cross_out_b}, {"mlp_ln.weight", &L.mlp_ln_w}, {"mlp_ln.bias", &L.mlp_ln_b}, {"fc1.weight", &L.fc1_w}, {"fc1.bias", &L.fc1_b},
                 {"fc2.weight", &L.fc2_w}, {"fc2.bias", &L.fc2_b}}) {
            if (Status st = ref(b + name, dst); !st.is_ok()) return st;
        }
    }
    auto dec = MetalWhisperDecoder::create(dc, w);
    if (!dec.is_ok()) return dec.status();
    m.decoder = std::move(dec).value();
    if (Status st = m.decoder->set_suppression(suppress, begin); !st.is_ok()) return st;

    // Features.
    WeightRef mel;
    if (Status st = ref("mel_filters", &mel); !st.is_ok()) return st;
    if (mel.dtype != DType::F32 || mel.rows != m.n_mels || mel.cols != n_fft / 2 + 1) return invalid_argument("mel_filters must be f32 [n_mels][n_fft/2 + 1]");
    WhisperFeatureConfig fc;
    fc.n_fft = n_fft;
    fc.hop = hop;
    fc.n_mels = m.n_mels;
    fc.chunk_samples = m.chunk;
    fc.mel_filters.assign(static_cast<const float *>(mel.data), static_cast<const float *>(mel.data) + mel.rows * mel.cols);
    auto feats = WhisperFeatures::create(fc);
    if (!feats.is_ok()) return feats.status();
    m.features = std::make_unique<WhisperFeatures>(std::move(feats).value());
    if (m.features->n_frames() != 2 * m.n_audio_ctx) return invalid_argument("the encoder takes " + std::to_string(2 * m.n_audio_ctx) + " frames, the features give " + std::to_string(m.features->n_frames()));

    // Encoder.
    auto units = parse_coreml_units(o.encoder_units);
    if (!units.is_ok()) return units.status();
    auto enc = CoreMlModel::load(paths.encoder, units.value());
    if (!enc.is_ok()) return enc.status();
    m.encoder = std::make_unique<CoreMlModel>(std::move(enc).value());
    auto in_shape = Shape::create({1, m.n_mels, m.features->n_frames()});
    auto out_shape = Shape::create({1, m.n_audio_ctx, m.n_state});
    if (!in_shape.is_ok() || !out_shape.is_ok()) return internal_error("bad encoder shapes");
    const auto & vin = m.encoder->inputs();
    const auto & vout = m.encoder->outputs();
    if (vin.size() != 1 || vin[0].name != kEncoderInput || vin[0].shape != in_shape.value()) {
        return invalid_argument(std::string("the encoder package must take one input '") + kEncoderInput + "' of shape " + in_shape.value().to_string());
    }
    if (vout.size() != 1 || vout[0].name != kEncoderOutput || vout[0].shape != out_shape.value()) {
        return invalid_argument(std::string("the encoder package must produce one output '") + kEncoderOutput + "' of shape " + out_shape.value().to_string());
    }
    m.mel.resize(static_cast<size_t>(m.n_mels) * static_cast<size_t>(m.features->n_frames()));
    m.states.resize(static_cast<size_t>(m.n_audio_ctx) * static_cast<size_t>(m.n_state));
    return WhisperPipeline(std::move(impl));
}

std::vector<int32_t> WhisperPipeline::prompt() const { return {impl_->sot, impl_->lang, impl_->task, impl_->notimestamps}; }

std::string WhisperPipeline::decode_text(const std::vector<int32_t> & tokens) const {
    const Impl & m = *impl_;
    std::string out;
    for (int32_t t : tokens) {
        if (t < 0 || t >= m.n_text_tokens || t >= static_cast<int32_t>(m.pieces.size())) continue;   // special tokens carry no text
        for (uint32_t cp : code_points(m.pieces[static_cast<size_t>(t)])) {
            auto it = m.bytes.find(cp);
            if (it != m.bytes.end()) out.push_back(static_cast<char>(it->second));
        }
    }
    return out;
}

Status WhisperPipeline::features(const float * samples, size_t n, std::vector<float> * mel) { return impl_->features->compute(samples, n, mel); }

Status WhisperPipeline::encode(const std::vector<float> & mel, std::vector<float> * states) {
    Impl & m = *impl_;
    if (!states) return invalid_argument("null output");
    if (mel.size() != m.mel.size()) return invalid_argument("expected " + std::to_string(m.mel.size()) + " feature values");
    states->assign(m.states.size(), 0.f);
    auto in_shape = Shape::create({1, m.n_mels, m.features->n_frames()});
    auto out_shape = Shape::create({1, m.n_audio_ctx, m.n_state});
    auto in = TensorView::create(const_cast<float *>(mel.data()), in_shape.value(), DType::F32);
    auto out = TensorView::create(states->data(), out_shape.value(), DType::F32);
    if (!in.is_ok() || !out.is_ok()) return internal_error("could not bind the encoder buffers");
    return m.encoder->predict({{kEncoderInput, in.value()}}, {{kEncoderOutput, out.value()}});
}

Result<WhisperResult> WhisperPipeline::transcribe(const std::vector<float> & samples, WhisperTimings * timings) {
    Impl & m = *impl_;
    if (samples.empty()) return invalid_argument("no samples");
    WhisperTimings tm;
    tm.audio_s = static_cast<double>(samples.size()) / m.sample_rate;
    const auto t_all = Clock::now();
    WhisperResult result;
    const std::vector<int32_t> pr = prompt();
    for (size_t start = 0; start < samples.size(); start += static_cast<size_t>(m.chunk)) {
        const size_t n = std::min<size_t>(static_cast<size_t>(m.chunk), samples.size() - start);
        auto t = Clock::now();
        if (Status st = m.features->compute(samples.data() + start, n, &m.mel); !st.is_ok()) return st;
        tm.features_ms += ms_since(t);
        t = Clock::now();
        if (Status st = encode(m.mel, &m.states); !st.is_ok()) return st;
        tm.encoder_ms += ms_since(t);
        t = Clock::now();
        if (Status st = m.decoder->set_audio(m.states.data()); !st.is_ok()) return st;
        for (size_t i = 0; i < pr.size(); i++) {
            if (Status st = m.decoder->eval_token(pr[i], i + 1 == pr.size()); !st.is_ok()) return st;
        }
        std::vector<int32_t> window;
        Status st = m.decoder->generate(m.options.max_tokens_per_window, m.options.queue_depth, [&](int32_t tok) {
            if (tok == m.eot) return false;
            window.push_back(tok);
            return true;
        });
        if (!st.is_ok()) return st;
        tm.decode_ms += ms_since(t);
        tm.tokens += static_cast<int64_t>(window.size());
        tm.windows++;
        result.tokens.insert(result.tokens.end(), window.begin(), window.end());
        result.text += decode_text(window);
    }
    tm.total_ms = ms_since(t_all);
    if (timings) *timings = tm;
    return result;
}

Result<WhisperResult> WhisperPipeline::transcribe_file(const std::string & path, WhisperTimings * timings) {
    auto audio = load_audio(path, impl_->sample_rate);
    if (!audio.is_ok()) return audio.status();
    return transcribe(audio.value(), timings);
}

}  // namespace dooms
