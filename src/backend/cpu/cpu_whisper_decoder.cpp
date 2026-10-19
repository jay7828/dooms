#include "backend/cpu/cpu_whisper_decoder.h"

#define ACCELERATE_NEW_LAPACK
#include <Accelerate/Accelerate.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <string>

namespace dooms {

namespace {

// f16 or f32 host tensor -> f32 copy.
std::vector<float> to_f32(const WeightRef & r) {
    const size_t n = static_cast<size_t>(r.rows * r.cols);
    std::vector<float> out(n);
    if (r.dtype == DType::F16) {
        const __fp16 * p = static_cast<const __fp16 *>(r.data);
        for (size_t i = 0; i < n; i++) out[i] = static_cast<float>(p[i]);
    } else {
        std::memcpy(out.data(), r.data, n * sizeof(float));
    }
    return out;
}

void layernorm(const float * x, const float * w, const float * b, int n, float eps, float * y) {
    double mean = 0;
    for (int i = 0; i < n; i++) mean += x[i];
    mean /= n;
    double var = 0;
    for (int i = 0; i < n; i++) { const double d = x[i] - mean; var += d * d; }
    const float inv = static_cast<float>(1.0 / std::sqrt(var / n + eps)), m = static_cast<float>(mean);
    for (int i = 0; i < n; i++) y[i] = (x[i] - m) * inv * w[i] + b[i];
}

// y = W x (+ bias) (+ y when accumulate); W row-major [rows][cols].
void gemv(const std::vector<float> & W, const float * x, int rows, int cols, const float * bias, float * y, bool accumulate) {
    cblas_sgemv(CblasRowMajor, CblasNoTrans, rows, cols, 1.f, W.data(), cols, x, 1, accumulate ? 1.f : 0.f, y, 1);
    if (bias) for (int i = 0; i < rows; i++) y[i] += bias[i];
}

struct Layer {
    std::vector<float> attn_ln_w, attn_ln_b, wq, bq, wk, wv, bv, wo, bo;
    std::vector<float> cross_ln_w, cross_ln_b, cq, cbq, ck, cv, cbv, co, cbo;
    std::vector<float> mlp_ln_w, mlp_ln_b, fc1, fc1_b, fc2, fc2_b;
};

}  // namespace

struct CpuWhisperDecoder::Impl {
    WhisperDecoderConfig cfg;
    int hd = 0;
    std::vector<float> token_embd, pos_embd, ln_w, ln_b;
    std::vector<Layer> layers;
    std::vector<std::vector<float>> kc, vc, ck, cv;   // per layer: self cache [n_text_ctx][d], cross keys/values [n_audio_ctx][d]
    std::vector<float> mask_always, mask_begin, logits_v, x, h, q, k, v, attn, scores, ff;
    int32_t pos = 0, pending = -1;
    bool has_audio = false, has_pending = false;

    // One position: logits of the token at `p` and its masked greedy successor.
    void step(int32_t token, int32_t p, bool first) {
        const int d = cfg.n_state, F = cfg.n_ff, V = cfg.n_vocab, A = cfg.n_audio_ctx, H = cfg.n_head;
        const float scale = 1.f / std::sqrt(static_cast<float>(hd));
        for (int i = 0; i < d; i++) x[i] = token_embd[static_cast<size_t>(token) * d + i] + pos_embd[static_cast<size_t>(p) * d + i];
        for (size_t l = 0; l < layers.size(); l++) {
            const Layer & L = layers[l];
            // self-attention: k, v of this position into the cache, then attention over positions 0..p
            layernorm(x.data(), L.attn_ln_w.data(), L.attn_ln_b.data(), d, cfg.ln_eps, h.data());
            gemv(L.wq, h.data(), d, d, L.bq.data(), q.data(), false);
            gemv(L.wk, h.data(), d, d, nullptr, kc[l].data() + static_cast<size_t>(p) * d, false);
            gemv(L.wv, h.data(), d, d, L.bv.data(), vc[l].data() + static_cast<size_t>(p) * d, false);
            attend(q.data(), kc[l].data(), vc[l].data(), p + 1, scale, H);
            gemv(L.wo, attn.data(), d, d, L.bo.data(), x.data(), true);
            // cross-attention over the window
            layernorm(x.data(), L.cross_ln_w.data(), L.cross_ln_b.data(), d, cfg.ln_eps, h.data());
            gemv(L.cq, h.data(), d, d, L.cbq.data(), q.data(), false);
            attend(q.data(), ck[l].data(), cv[l].data(), A, scale, H);
            gemv(L.co, attn.data(), d, d, L.cbo.data(), x.data(), true);
            // MLP
            layernorm(x.data(), L.mlp_ln_w.data(), L.mlp_ln_b.data(), d, cfg.ln_eps, h.data());
            gemv(L.fc1, h.data(), F, d, L.fc1_b.data(), ff.data(), false);
            for (int i = 0; i < F; i++) ff[i] = 0.5f * ff[i] * (1.f + std::erf(ff[i] * 0.70710678118654752f));
            gemv(L.fc2, ff.data(), d, F, L.fc2_b.data(), x.data(), true);
        }
        layernorm(x.data(), ln_w.data(), ln_b.data(), d, cfg.ln_eps, h.data());
        gemv(token_embd, h.data(), V, d, nullptr, logits_v.data(), false);
        const std::vector<float> & mask = first ? mask_begin : mask_always;
        int32_t best = 0;
        float bv = -std::numeric_limits<float>::infinity();
        for (int i = 0; i < V; i++) {
            const float s = logits_v[i] + mask[i];
            if (s > bv) { bv = s; best = i; }   // the lowest index among equal maxima, as the GPU argmax
        }
        pending = best;
    }

    // attn[head] = softmax(q_head . K_head^T * scale) V_head over n positions of the [n][d] caches.
    void attend(const float * qv, const float * K, const float * Vv, int n, float scale, int H) {
        const int d = cfg.n_state;
        for (int hh = 0; hh < H; hh++) {
            cblas_sgemv(CblasRowMajor, CblasNoTrans, n, hd, scale, K + hh * hd, d, qv + hh * hd, 1, 0.f, scores.data(), 1);
            float mx = -std::numeric_limits<float>::infinity();
            for (int j = 0; j < n; j++) mx = std::max(mx, scores[j]);
            double sum = 0;
            for (int j = 0; j < n; j++) { scores[j] = std::exp(scores[j] - mx); sum += scores[j]; }
            const float inv = static_cast<float>(1.0 / sum);
            cblas_sgemv(CblasRowMajor, CblasTrans, n, hd, inv, Vv + hh * hd, d, scores.data(), 1, 0.f, attn.data() + hh * hd, 1);
        }
    }
};

CpuWhisperDecoder::CpuWhisperDecoder(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}
CpuWhisperDecoder::~CpuWhisperDecoder() = default;
const WhisperDecoderConfig & CpuWhisperDecoder::config() const { return impl_->cfg; }
int32_t CpuWhisperDecoder::position() const { return impl_->pos; }

Result<std::unique_ptr<CpuWhisperDecoder>> CpuWhisperDecoder::create(const WhisperDecoderConfig & c, const WhisperDecoderWeights & w) {
    if (c.n_layer < 1 || c.n_state < 1 || c.n_head < 1 || c.n_ff < 1 || c.n_vocab < 1 || c.n_text_ctx < 1 || c.n_audio_ctx < 1 || !(c.ln_eps > 0.f)) {
        return invalid_argument("whisper decoder config: every size must be positive and ln_eps > 0");
    }
    if (c.n_state % c.n_head != 0) return invalid_argument("n_state is not a multiple of n_head");
    if (static_cast<int64_t>(w.layers.size()) != c.n_layer) return invalid_argument("expected " + std::to_string(c.n_layer) + " layers of weights");
    const int64_t d = c.n_state, F = c.n_ff, V = c.n_vocab;
    auto check = [](const WeightRef & r, int64_t rows, int64_t cols, const std::string & name) -> Status {
        if (!r.data) return invalid_argument("weight '" + name + "' is missing");
        if (r.rows != rows || r.cols != cols) return invalid_argument("weight '" + name + "' is " + std::to_string(r.rows) + "x" + std::to_string(r.cols) + ", expected " + std::to_string(rows) + "x" + std::to_string(cols));
        if (r.dtype != DType::F16 && r.dtype != DType::F32) return invalid_argument("weight '" + name + "' must be f16 or f32");
        return Status();
    };
    Status st = check(w.token_embd, V, d, "token_embd");
    if (st.is_ok()) st = check(w.pos_embd, c.n_text_ctx, d, "pos_embd");
    if (st.is_ok()) st = check(w.ln_w, 1, d, "ln.weight");
    if (st.is_ok()) st = check(w.ln_b, 1, d, "ln.bias");
    for (int32_t l = 0; l < c.n_layer && st.is_ok(); l++) {
        const WhisperLayerWeights & L = w.layers[static_cast<size_t>(l)];
        const std::string b = "blk." + std::to_string(l) + ".";
        for (auto [ref, name] : std::initializer_list<std::pair<const WeightRef *, const char *>>{
                 {&L.attn_ln_w, "attn_ln.weight"}, {&L.attn_ln_b, "attn_ln.bias"}, {&L.attn_q_b, "attn_q.bias"}, {&L.attn_v_b, "attn_v.bias"}, {&L.attn_out_b, "attn_out.bias"},
                 {&L.cross_ln_w, "cross_ln.weight"}, {&L.cross_ln_b, "cross_ln.bias"}, {&L.cross_q_b, "cross_q.bias"}, {&L.cross_v_b, "cross_v.bias"},
                 {&L.cross_out_b, "cross_out.bias"}, {&L.mlp_ln_w, "mlp_ln.weight"}, {&L.mlp_ln_b, "mlp_ln.bias"}, {&L.fc2_b, "fc2.bias"}}) {
            if (st.is_ok()) st = check(*ref, 1, d, b + name);
        }
        if (st.is_ok()) st = check(L.fc1_b, 1, F, b + "fc1.bias");
        for (auto [ref, name] : std::initializer_list<std::pair<const WeightRef *, const char *>>{
                 {&L.attn_q_w, "attn_q.weight"}, {&L.attn_k_w, "attn_k.weight"}, {&L.attn_v_w, "attn_v.weight"}, {&L.attn_out_w, "attn_out.weight"},
                 {&L.cross_q_w, "cross_q.weight"}, {&L.cross_k_w, "cross_k.weight"}, {&L.cross_v_w, "cross_v.weight"}, {&L.cross_out_w, "cross_out.weight"}}) {
            if (st.is_ok()) st = check(*ref, d, d, b + name);
        }
        if (st.is_ok()) st = check(L.fc1_w, F, d, b + "fc1.weight");
        if (st.is_ok()) st = check(L.fc2_w, d, F, b + "fc2.weight");
    }
    if (!st.is_ok()) return st;
    auto m = std::make_unique<Impl>();
    m->cfg = c;
    m->hd = static_cast<int>(d / c.n_head);
    m->token_embd = to_f32(w.token_embd);
    m->pos_embd = to_f32(w.pos_embd);
    m->ln_w = to_f32(w.ln_w);
    m->ln_b = to_f32(w.ln_b);
    for (const WhisperLayerWeights & L : w.layers) {
        Layer o;
        o.attn_ln_w = to_f32(L.attn_ln_w); o.attn_ln_b = to_f32(L.attn_ln_b); o.wq = to_f32(L.attn_q_w); o.bq = to_f32(L.attn_q_b); o.wk = to_f32(L.attn_k_w);
        o.wv = to_f32(L.attn_v_w); o.bv = to_f32(L.attn_v_b); o.wo = to_f32(L.attn_out_w); o.bo = to_f32(L.attn_out_b);
        o.cross_ln_w = to_f32(L.cross_ln_w); o.cross_ln_b = to_f32(L.cross_ln_b); o.cq = to_f32(L.cross_q_w); o.cbq = to_f32(L.cross_q_b); o.ck = to_f32(L.cross_k_w);
        o.cv = to_f32(L.cross_v_w); o.cbv = to_f32(L.cross_v_b); o.co = to_f32(L.cross_out_w); o.cbo = to_f32(L.cross_out_b);
        o.mlp_ln_w = to_f32(L.mlp_ln_w); o.mlp_ln_b = to_f32(L.mlp_ln_b); o.fc1 = to_f32(L.fc1_w); o.fc1_b = to_f32(L.fc1_b); o.fc2 = to_f32(L.fc2_w); o.fc2_b = to_f32(L.fc2_b);
        m->layers.push_back(std::move(o));
    }
    const size_t td = static_cast<size_t>(c.n_text_ctx) * d, ad = static_cast<size_t>(c.n_audio_ctx) * d;
    m->kc.assign(static_cast<size_t>(c.n_layer), std::vector<float>(td));
    m->vc.assign(static_cast<size_t>(c.n_layer), std::vector<float>(td));
    m->ck.assign(static_cast<size_t>(c.n_layer), std::vector<float>(ad));
    m->cv.assign(static_cast<size_t>(c.n_layer), std::vector<float>(ad));
    m->mask_always.assign(static_cast<size_t>(V), 0.f);
    m->mask_begin.assign(static_cast<size_t>(V), 0.f);
    m->logits_v.assign(static_cast<size_t>(V), 0.f);
    for (auto * v : {&m->x, &m->h, &m->q, &m->k, &m->v, &m->attn}) v->assign(static_cast<size_t>(d), 0.f);
    m->ff.assign(static_cast<size_t>(F), 0.f);
    m->scores.assign(static_cast<size_t>(std::max(c.n_text_ctx, c.n_audio_ctx)), 0.f);
    return std::unique_ptr<CpuWhisperDecoder>(new CpuWhisperDecoder(std::move(m)));
}

Status CpuWhisperDecoder::set_suppression(const std::vector<int32_t> & always, const std::vector<int32_t> & at_begin) {
    Impl & m = *impl_;
    for (int32_t t : always) if (t < 0 || t >= m.cfg.n_vocab) return invalid_argument("suppressed token " + std::to_string(t) + " is outside the vocabulary");
    for (int32_t t : at_begin) if (t < 0 || t >= m.cfg.n_vocab) return invalid_argument("begin-suppressed token " + std::to_string(t) + " is outside the vocabulary");
    std::fill(m.mask_always.begin(), m.mask_always.end(), 0.f);
    for (int32_t t : always) m.mask_always[static_cast<size_t>(t)] = -std::numeric_limits<float>::infinity();
    m.mask_begin = m.mask_always;
    for (int32_t t : at_begin) m.mask_begin[static_cast<size_t>(t)] = -std::numeric_limits<float>::infinity();
    return Status();
}

Status CpuWhisperDecoder::set_audio(const float * states) {
    Impl & m = *impl_;
    if (!states) return invalid_argument("no encoder states");
    const int A = m.cfg.n_audio_ctx, d = m.cfg.n_state;
    const size_t n = static_cast<size_t>(A) * d;
    for (size_t i = 0; i < n; i++) if (!std::isfinite(states[i])) return invalid_argument("the encoder states contain NaN or Inf");
    for (size_t l = 0; l < m.layers.size(); l++) {   // K = X Wk^T, V = X Wv^T + bv
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, A, d, d, 1.f, states, d, m.layers[l].ck.data(), d, 0.f, m.ck[l].data(), d);
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, A, d, d, 1.f, states, d, m.layers[l].cv.data(), d, 0.f, m.cv[l].data(), d);
        const float * bv = m.layers[l].cbv.data();
        for (int a = 0; a < A; a++) for (int i = 0; i < d; i++) m.cv[l][static_cast<size_t>(a) * d + i] += bv[i];
    }
    m.pos = 0;
    m.has_audio = true;
    m.has_pending = false;
    return Status();
}

Status CpuWhisperDecoder::eval_token(int32_t token, bool first_step) {
    Impl & m = *impl_;
    if (!m.has_audio) return invalid_argument("set_audio() has not been called");
    if (token < 0 || token >= m.cfg.n_vocab) return invalid_argument("token " + std::to_string(token) + " is outside the vocabulary");
    if (m.pos >= m.cfg.n_text_ctx) return resource_exhausted("the text context of " + std::to_string(m.cfg.n_text_ctx) + " positions is full");
    m.step(token, m.pos, first_step);
    m.pos++;
    m.has_pending = true;
    return Status();
}

Status CpuWhisperDecoder::eval_prompt(const std::vector<int32_t> & tokens) {
    Impl & m = *impl_;
    if (!m.has_audio) return invalid_argument("set_audio() has not been called");
    if (tokens.empty()) return invalid_argument("empty prompt");
    for (int32_t t : tokens) if (t < 0 || t >= m.cfg.n_vocab) return invalid_argument("token " + std::to_string(t) + " is outside the vocabulary");
    if (m.pos + static_cast<int64_t>(tokens.size()) > m.cfg.n_text_ctx) return resource_exhausted("the prompt does not fit in the text context");
    for (size_t i = 0; i < tokens.size(); i++) {
        if (Status st = eval_token(tokens[i], i + 1 == tokens.size()); !st.is_ok()) return st;
    }
    return Status();
}

Result<int32_t> CpuWhisperDecoder::pending_token() {
    if (!impl_->has_pending) return invalid_argument("nothing has been evaluated yet");
    return impl_->pending;
}

Result<std::vector<float>> CpuWhisperDecoder::logits() {
    if (!impl_->has_pending) return invalid_argument("nothing has been evaluated yet");
    return impl_->logits_v;
}

Status CpuWhisperDecoder::generate(int32_t max_tokens, int32_t depth, const std::function<bool(int32_t)> & on_token) {
    Impl & m = *impl_;
    if (!m.has_pending) return invalid_argument("nothing has been evaluated yet");
    if (max_tokens < 1 || depth < 1) return invalid_argument("max_tokens and depth must be >= 1");
    if (static_cast<int64_t>(m.pos) + max_tokens - 1 > m.cfg.n_text_ctx) return resource_exhausted("the text context of " + std::to_string(m.cfg.n_text_ctx) + " positions would overflow");
    for (int32_t produced = 0; produced < max_tokens; produced++) {
        const int32_t token = m.pending;
        if (!on_token(token) || produced + 1 == max_tokens) break;
        m.step(token, m.pos, false);
        m.pos++;
    }
    m.has_pending = false;
    return Status();
}

}  // namespace dooms
