// Tests the engine's own Metal Llama decoder (backend/metal/metal_llama_decoder.h, loader pipeline/native_llama.h).
// Always: a small random Llama (grouped-query attention, two layers) against an independent double-precision reference written here (same f16 weights, keys
// and values rounded to f16 like the decoder's cache): logits after a prompt, greedy tokens over a generation, chunked prefill, token vs embedding input,
// queue depth, stop semantics, position accounting and argument errors.
// Quantized matrices (q4_0, q5_0, q8_0, q4_K, q6_K and a Q4_K_M-like mix): random blocks dequantized by ggml itself into the same reference.
// Opt-in (DOOMS_VLM_DIR with text-f16.gguf, vlm-config.gguf, reference.gguf): SmolVLM's text model against llama.cpp (TextDecoder) and transformers.
// Opt-in (DOOMS_QWEN_F16 / DOOMS_QWEN_GGUF): Qwen2.5-0.5B against llama.cpp on the same file (f16, or a quantized one such as Q4_K_M).
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include "backend/metal/metal_llama_decoder.h"
#include "check.h"
#include "format/gguf.h"
#include "ggml.h"
#include "models/vlm/text_decoder.h"
#include "pipeline/native_llama.h"

using namespace dooms;

namespace {

float to_half_and_back(float v) { return static_cast<float>(static_cast<__fp16>(v)); }

struct TinyModel {
    LlamaDecoderConfig cfg;
    std::vector<__fp16> embd, output;
    std::vector<float> output_norm;
    struct Layer {
        std::vector<float> attn_norm, ffn_norm, qkv_bias;  // qkv_bias empty: none (llama); filled: q | k | v biases (qwen2)
        std::vector<__fp16> qkv, wo, gate, up, down;
    };
    std::vector<Layer> layers;
};

std::vector<__fp16> random_halves(std::mt19937 & rng, size_t n, float scale) {
    std::normal_distribution<float> d(0.f, scale);
    std::vector<__fp16> v(n);
    for (auto & x : v) x = static_cast<__fp16>(d(rng));
    return v;
}

std::vector<float> random_norm(std::mt19937 & rng, size_t n) {
    std::uniform_real_distribution<float> d(0.7f, 1.3f);
    std::vector<float> v(n);
    for (auto & x : v) x = d(rng);
    return v;
}

// qwen = true: qwen2-style block (q/k/v biases, split-half rotary pairs). n_ff 2208 also makes the single-token down projection read a long input row.
// head_dim 64 selects the flash-attention prefill kernel (head_dim 16 runs the per-query one).
TinyModel make_tiny(uint32_t seed, int32_t n_batch, int32_t n_ctx, bool qwen = false, int32_t head_dim = 16) {
    TinyModel m;
    LlamaDecoderConfig & c = m.cfg;
    c.n_layer = 2; c.n_embd = 64; c.n_head = 4; c.n_head_kv = 2; c.head_dim = head_dim; c.n_ff = 96; c.n_vocab = 50;
    c.rope_base = 10000.f; c.rms_eps = 1e-5f; c.n_ctx = n_ctx; c.n_batch = n_batch;
    c.rope_neox = qwen;
    if (qwen) c.n_ff = 2208;  // above kMaxStagedInput (the single-token down projection reads its input from device memory), a multiple of 32 (64x32 prefill)
    std::mt19937 rng(seed);
    const size_t E = 64, F = static_cast<size_t>(c.n_ff), V = 50, Q = static_cast<size_t>(4 * head_dim), qkv_rows = Q + 2 * static_cast<size_t>(2 * head_dim);
    m.embd = random_halves(rng, V * E, 1.f);
    m.output = random_halves(rng, V * E, 0.2f);
    m.output_norm = random_norm(rng, E);
    for (int l = 0; l < c.n_layer; l++) {
        TinyModel::Layer L;
        L.attn_norm = random_norm(rng, E);
        L.ffn_norm = random_norm(rng, E);
        L.qkv = random_halves(rng, qkv_rows * E, 0.15f);
        if (qwen) { std::normal_distribution<float> d(0.f, 0.3f); for (size_t i = 0; i < qkv_rows; i++) L.qkv_bias.push_back(d(rng)); }
        L.wo = random_halves(rng, E * Q, 0.15f);
        L.gate = random_halves(rng, F * E, 0.15f);
        L.up = random_halves(rng, F * E, 0.15f);
        L.down = random_halves(rng, E * F, 0.15f);
        m.layers.push_back(std::move(L));
    }
    return m;
}

LlamaDecoderWeights weights_of(const TinyModel & m) {
    const int64_t E = m.cfg.n_embd, F = m.cfg.n_ff, V = m.cfg.n_vocab, Q = m.cfg.n_head * m.cfg.head_dim, KV = m.cfg.n_head_kv * m.cfg.head_dim;
    LlamaDecoderWeights w;
    w.token_embd = {m.embd.data(), DType::F16, V, E};
    w.output = {m.output.data(), DType::F16, V, E};
    w.output_norm = {m.output_norm.data(), DType::F32, 1, E};
    for (const auto & L : m.layers) {
        LlamaLayerWeights lw;
        lw.attn_norm = {L.attn_norm.data(), DType::F32, 1, E};
        lw.ffn_norm = {L.ffn_norm.data(), DType::F32, 1, E};
        lw.attn_qkv = {L.qkv.data(), DType::F16, Q + 2 * KV, E};
        if (!L.qkv_bias.empty()) {  // given split, the way a qwen2 file stores them
            lw.attn_q_bias = {L.qkv_bias.data(), DType::F32, 1, Q};
            lw.attn_k_bias = {L.qkv_bias.data() + Q, DType::F32, 1, KV};
            lw.attn_v_bias = {L.qkv_bias.data() + Q + KV, DType::F32, 1, KV};
        }
        lw.attn_output = {L.wo.data(), DType::F16, E, Q};
        lw.ffn_gate = {L.gate.data(), DType::F16, F, E};
        lw.ffn_up = {L.up.data(), DType::F16, F, E};
        lw.ffn_down = {L.down.data(), DType::F16, E, F};
        w.layers.push_back(lw);
    }
    return w;
}

// Independent reference: plain loops in double, the decoder's math written out from the Llama definition.
class Reference {
public:
    explicit Reference(const TinyModel & m) : m_(m), kc_(m.layers.size()), vc_(m.layers.size()) {}

    std::vector<double> step(const std::vector<double> & input) {
        const auto & c = m_.cfg;
        const int E = c.n_embd, F = c.n_ff, hd = c.head_dim, nh = c.n_head, nkv = c.n_head_kv, Q = nh * hd, KV = nkv * hd;
        std::vector<double> x = input;
        const int pos = static_cast<int>(kc_[0].size()) / KV;
        for (size_t l = 0; l < m_.layers.size(); l++) {
            const auto & L = m_.layers[l];
            const std::vector<double> xn = rms(x, L.attn_norm);
            std::vector<double> qkv = matvec(L.qkv, xn, Q + 2 * KV, E);
            for (size_t i = 0; i < L.qkv_bias.size(); i++) qkv[i] += L.qkv_bias[i];
            for (int s = 0; s < nh + nkv; s++) rope(&qkv[static_cast<size_t>(s * hd)], pos);
            for (int i = 0; i < KV; i++) {
                kc_[l].push_back(to_half_and_back(static_cast<float>(qkv[static_cast<size_t>(Q + i)])));
                vc_[l].push_back(to_half_and_back(static_cast<float>(qkv[static_cast<size_t>(Q + KV + i)])));
            }
            std::vector<double> att(static_cast<size_t>(Q), 0.0);
            for (int h = 0; h < nh; h++) {
                const int kvh = h / (nh / nkv);
                std::vector<double> sc(static_cast<size_t>(pos + 1));
                double mx = -1e300;
                for (int j = 0; j <= pos; j++) {
                    double s = 0;
                    for (int d = 0; d < hd; d++) s += qkv[static_cast<size_t>(h * hd + d)] * kc_[l][static_cast<size_t>(j * KV + kvh * hd + d)];
                    sc[static_cast<size_t>(j)] = s / std::sqrt(static_cast<double>(hd));
                    mx = std::max(mx, sc[static_cast<size_t>(j)]);
                }
                double sum = 0;
                for (double & s : sc) { s = std::exp(s - mx); sum += s; }
                for (int d = 0; d < hd; d++) {
                    double a = 0;
                    for (int j = 0; j <= pos; j++) a += sc[static_cast<size_t>(j)] * vc_[l][static_cast<size_t>(j * KV + kvh * hd + d)];
                    att[static_cast<size_t>(h * hd + d)] = a / sum;
                }
            }
            const std::vector<double> o = matvec(L.wo, att, E, Q);
            for (int i = 0; i < E; i++) x[static_cast<size_t>(i)] += o[static_cast<size_t>(i)];
            const std::vector<double> fn = rms(x, L.ffn_norm);
            const std::vector<double> g = matvec(L.gate, fn, F, E), u = matvec(L.up, fn, F, E);
            std::vector<double> hid(static_cast<size_t>(F));
            for (int i = 0; i < F; i++) { const double a = g[static_cast<size_t>(i)]; hid[static_cast<size_t>(i)] = a / (1 + std::exp(-a)) * u[static_cast<size_t>(i)]; }
            const std::vector<double> dn = matvec(L.down, hid, E, F);
            for (int i = 0; i < E; i++) x[static_cast<size_t>(i)] += dn[static_cast<size_t>(i)];
        }
        return matvec(m_.output, rms(x, m_.output_norm), c.n_vocab, E);
    }

    std::vector<double> embedding(int32_t token) const {
        std::vector<double> e(static_cast<size_t>(m_.cfg.n_embd));
        for (int i = 0; i < m_.cfg.n_embd; i++) e[static_cast<size_t>(i)] = static_cast<double>(m_.embd[static_cast<size_t>(token * m_.cfg.n_embd + i)]);
        return e;
    }

private:
    std::vector<double> rms(const std::vector<double> & x, const std::vector<float> & g) const {
        double ss = 0;
        for (double v : x) ss += v * v;
        const double inv = 1.0 / std::sqrt(ss / static_cast<double>(x.size()) + m_.cfg.rms_eps);
        std::vector<double> y(x.size());
        for (size_t i = 0; i < x.size(); i++) y[i] = x[i] * inv * g[i];
        return y;
    }
    static std::vector<double> matvec(const std::vector<__fp16> & W, const std::vector<double> & x, int rows, int cols) {
        std::vector<double> y(static_cast<size_t>(rows), 0.0);
        for (int r = 0; r < rows; r++) for (int k = 0; k < cols; k++) y[static_cast<size_t>(r)] += static_cast<double>(W[static_cast<size_t>(r * cols + k)]) * x[static_cast<size_t>(k)];
        return y;
    }
    void rope(double * v, int pos) const {
        const int hd = m_.cfg.head_dim;
        for (int i = 0; i < hd / 2; i++) {
            const double th = pos * std::pow(static_cast<double>(m_.cfg.rope_base), -2.0 * i / hd);
            const int i0 = m_.cfg.rope_neox ? i : 2 * i, i1 = m_.cfg.rope_neox ? i + hd / 2 : 2 * i + 1;
            const double a = v[i0], b = v[i1];
            v[i0] = a * std::cos(th) - b * std::sin(th);
            v[i1] = a * std::sin(th) + b * std::cos(th);
        }
    }
    const TinyModel & m_;
    std::vector<std::vector<double>> kc_, vc_;
};

int32_t argmax(const std::vector<double> & v) { return static_cast<int32_t>(std::max_element(v.begin(), v.end()) - v.begin()); }

double rel_l2(const std::vector<float> & a, const std::vector<double> & b) {
    double d = 0, n = 0;
    for (size_t i = 0; i < b.size(); i++) { d += (a[i] - b[i]) * (a[i] - b[i]); n += b[i] * b[i]; }
    return std::sqrt(d / n);
}

const std::vector<int32_t> kPrompt = {3, 17, 42, 8, 8, 21, 0, 49, 12, 30, 5};

// ---- Quantized matrices ----
// Random ggml blocks with power-of-two scales, so every dequantized value is exact in f16 (asserted): the double reference then sees exactly the weights the
// GPU reads, and any difference is the kernels'. ggml's own to_float dequantizes (independent of the Metal code).
struct QuantMat {
    DType dt;
    std::vector<uint8_t> bytes;
    std::vector<__fp16> values;   // ggml's dequantization, row-major [rows][cols]
};

ggml_type ggml_of(DType t) {
    switch (t) {
        case DType::Q4_0: return GGML_TYPE_Q4_0;
        case DType::Q5_0: return GGML_TYPE_Q5_0;
        case DType::Q8_0: return GGML_TYPE_Q8_0;
        case DType::Q4_K: return GGML_TYPE_Q4_K;
        case DType::Q6_K: return GGML_TYPE_Q6_K;
        default: return GGML_TYPE_F16;
    }
}

void put_half(uint8_t * p, float v) { const __fp16 h = static_cast<__fp16>(v); std::memcpy(p, &h, 2); }

QuantMat random_quant(std::mt19937 & rng, DType dt, int64_t rows, int64_t cols, bool * exact) {
    QuantMat q{dt, {}, {}};
    const int64_t nb = rows * cols / dtype_block_elems(dt), bb = dtype_block_bytes(dt);
    q.bytes.resize(static_cast<size_t>(nb * bb));
    std::uniform_int_distribution<int> byte(0, 255), sc6(-63, 63);
    for (int64_t b = 0; b < nb; b++) {
        uint8_t * p = q.bytes.data() + b * bb;
        for (int64_t i = 0; i < bb; i++) p[i] = static_cast<uint8_t>(byte(rng));
        switch (dt) {   // the scales: powers of two sized for weights of std ~0.14 (block layouts: core/dtype.cpp)
            case DType::Q4_0: put_half(p, 1.f / 32); break;
            case DType::Q5_0: put_half(p, 1.f / 64); break;
            case DType::Q8_0: put_half(p, 1.f / 512); break;
            case DType::Q4_K: put_half(p, 1.f / 1024); put_half(p + 2, 1.f / 1024); break;   // 6-bit scales and mins stay random
            case DType::Q6_K:
                for (int i = 0; i < 16; i++) p[192 + i] = static_cast<uint8_t>(static_cast<int8_t>(sc6(rng)));   // |scale| <= 63: |scale * q| < 2048, exact in f16
                put_half(p + 208, 1.f / 4096);
                break;
            default: break;
        }
    }
    std::vector<float> f(static_cast<size_t>(rows * cols));
    ggml_get_type_traits(ggml_of(dt))->to_float(q.bytes.data(), f.data(), rows * cols);
    q.values.resize(f.size());
    for (size_t i = 0; i < f.size(); i++) {
        q.values[i] = static_cast<__fp16>(f[i]);
        if (static_cast<float>(q.values[i]) != f[i]) *exact = false;
    }
    return q;
}

struct QuantPlan { const char * name; DType embd, output, q, k, v, wo, gate, down0, down1; };

struct QuantTiny {
    TinyModel ref;   // the dequantized values (for Reference)
    std::vector<QuantMat> mats;
    LlamaDecoderWeights w;
};

// A qwen2-style tiny model with every matrix quantized as the plan says (widths multiples of 256 for the K types; n_ff above kMaxStagedInput).
QuantTiny make_quant(const QuantPlan & plan, uint32_t seed, bool * exact) {
    QuantTiny t;
    LlamaDecoderConfig & c = t.ref.cfg;
    c.n_layer = 2; c.n_embd = 256; c.n_head = 4; c.n_head_kv = 2; c.head_dim = 64; c.n_ff = 2304; c.n_vocab = 50;
    c.rope_base = 10000.f; c.rms_eps = 1e-5f; c.n_ctx = 128; c.n_batch = 64; c.rope_neox = true;
    std::mt19937 rng(seed);
    const int64_t E = 256, F = 2304, V = 50, Q = 256, KV = 128;
    t.mats.reserve(64);   // stable addresses for the WeightRefs
    auto mat = [&](DType dt, int64_t rows, int64_t cols) -> const QuantMat & { t.mats.push_back(random_quant(rng, dt, rows, cols, exact)); return t.mats.back(); };
    auto ref = [](const QuantMat & m, int64_t rows, int64_t cols) { return WeightRef{m.bytes.data(), m.dt, rows, cols}; };
    const QuantMat & embd = mat(plan.embd, V, E);
    const QuantMat & out = mat(plan.output, V, E);
    t.ref.embd = embd.values;
    t.ref.output = out.values;
    t.ref.output_norm = random_norm(rng, E);
    t.w.token_embd = ref(embd, V, E);
    t.w.output = ref(out, V, E);
    t.w.output_norm = {t.ref.output_norm.data(), DType::F32, 1, E};
    t.ref.layers.resize(2);
    for (int l = 0; l < 2; l++) {
        TinyModel::Layer & L = t.ref.layers[static_cast<size_t>(l)];
        L.attn_norm = random_norm(rng, E);
        L.ffn_norm = random_norm(rng, E);
        std::normal_distribution<float> d(0.f, 0.3f);
        for (int64_t i = 0; i < Q + 2 * KV; i++) L.qkv_bias.push_back(d(rng));
        const QuantMat & mq = mat(plan.q, Q, E), & mk = mat(plan.k, KV, E), & mv = mat(plan.v, KV, E);
        L.qkv = mq.values;
        L.qkv.insert(L.qkv.end(), mk.values.begin(), mk.values.end());
        L.qkv.insert(L.qkv.end(), mv.values.begin(), mv.values.end());
        const QuantMat & mo = mat(plan.wo, E, Q), & mg = mat(plan.gate, F, E), & mu = mat(plan.gate, F, E), & md = mat(l == 0 ? plan.down0 : plan.down1, E, F);
        L.wo = mo.values; L.gate = mg.values; L.up = mu.values; L.down = md.values;
        LlamaLayerWeights lw;
        lw.attn_norm = {L.attn_norm.data(), DType::F32, 1, E};
        lw.ffn_norm = {L.ffn_norm.data(), DType::F32, 1, E};
        lw.attn_q = ref(mq, Q, E); lw.attn_k = ref(mk, KV, E); lw.attn_v = ref(mv, KV, E);
        lw.attn_q_bias = {L.qkv_bias.data(), DType::F32, 1, Q};
        lw.attn_k_bias = {L.qkv_bias.data() + Q, DType::F32, 1, KV};
        lw.attn_v_bias = {L.qkv_bias.data() + Q + KV, DType::F32, 1, KV};
        lw.attn_output = ref(mo, E, Q); lw.ffn_gate = ref(mg, F, E); lw.ffn_up = ref(mu, F, E); lw.ffn_down = ref(md, E, F);
        t.w.layers.push_back(lw);
    }
    return t;
}

}  // namespace

TEST(prompt_logits_and_greedy_generation_match_the_reference) {
    const TinyModel m = make_tiny(1, 64, 128);
    auto dec = MetalLlamaDecoder::create(m.cfg, weights_of(m));
    EXPECT(dec.is_ok());
    if (!dec.is_ok()) { std::printf("        %s\n", dec.status().to_string().c_str()); return; }
    MetalLlamaDecoder & d = *dec.value();
    Reference ref(m);
    std::vector<double> want;
    for (int32_t t : kPrompt) want = ref.step(ref.embedding(t));
    // Prompt in one call: the multi-token path rounds the projections' operand tiles to half (as llama.cpp's Metal matmul does), so it sits ~1e-3 from the
    // double reference; one token per call runs the single-token kernels (f32 activations, f16 weights) and must be much closer.
    EXPECT(d.eval_tokens(kPrompt).is_ok());
    auto lg = d.logits();
    EXPECT(lg.is_ok());
    if (!lg.is_ok()) return;
    std::printf("        prompt logits relative L2 vs the reference: %.2e in one call\n", rel_l2(lg.value(), want));
    EXPECT(rel_l2(lg.value(), want) < 3e-3);
    EXPECT_EQ(d.pending_token().value(), argmax(want));
    d.reset();
    for (int32_t t : kPrompt) EXPECT(d.eval_tokens({t}).is_ok());
    lg = d.logits();
    EXPECT(lg.is_ok());
    if (!lg.is_ok()) return;
    std::printf("        prompt logits relative L2 vs the reference: %.2e one token per call\n", rel_l2(lg.value(), want));
    EXPECT(rel_l2(lg.value(), want) < 1e-4);
    EXPECT_EQ(d.pending_token().value(), argmax(want));
    EXPECT_EQ(d.position(), static_cast<int32_t>(kPrompt.size()));

    // Free-running greedy generation: the reference feeds back its own argmax.
    std::vector<int32_t> want_tokens;
    int32_t t = argmax(want);
    for (int i = 0; i < 24; i++) {
        want_tokens.push_back(t);
        t = argmax(ref.step(ref.embedding(t)));
    }
    std::vector<int32_t> got;
    EXPECT(d.generate(24, 3, [&](int32_t tok) { got.push_back(tok); return true; }).is_ok());
    EXPECT(got == want_tokens);
    EXPECT_EQ(d.position(), static_cast<int32_t>(kPrompt.size()) + 23);  // the last token was not fed back
}

// The qwen2 block: biases on q/k/v, split-half rotary pairs, and a feed-forward wide enough that the down projection's input row is long.
TEST(qwen_style_block_matches_the_reference) {
    const TinyModel m = make_tiny(4, 64, 128, true);
    auto dec = MetalLlamaDecoder::create(m.cfg, weights_of(m));
    EXPECT(dec.is_ok());
    if (!dec.is_ok()) { std::printf("        %s\n", dec.status().to_string().c_str()); return; }
    MetalLlamaDecoder & d = *dec.value();
    Reference ref(m);
    std::vector<double> want;
    for (int32_t t : kPrompt) want = ref.step(ref.embedding(t));
    EXPECT(d.eval_tokens(kPrompt).is_ok());
    std::printf("        qwen-style prompt logits relative L2: %.2e in one call\n", rel_l2(d.logits().value(), want));
    EXPECT(rel_l2(d.logits().value(), want) < 3e-3);
    d.reset();
    for (int32_t t : kPrompt) EXPECT(d.eval_tokens({t}).is_ok());
    std::printf("        qwen-style prompt logits relative L2: %.2e one token per call\n", rel_l2(d.logits().value(), want));
    EXPECT(rel_l2(d.logits().value(), want) < 1e-4);
    std::vector<int32_t> want_tokens;
    int32_t t = argmax(want);
    for (int i = 0; i < 16; i++) { want_tokens.push_back(t); t = argmax(ref.step(ref.embedding(t))); }
    std::vector<int32_t> got;
    EXPECT(d.generate(16, 2, [&](int32_t tok) { got.push_back(tok); return true; }).is_ok());
    EXPECT(got == want_tokens);
    // A bias for q but not k/v is malformed.
    LlamaDecoderWeights w = weights_of(m);
    w.layers[0].attn_k_bias = WeightRef();
    EXPECT(!MetalLlamaDecoder::create(m.cfg, w).is_ok());
}

// Every quantized type, alone and mixed like a Q4_K_M file (q/k in q5_0 and v in q8_0: separate q/k/v dispatches; the down projections in q4_K and q6_K):
// one token per call (the matvec kernels, f32 activations on exact weights) within 1e-4 of the double reference, the prompt in one call (dequantized to half,
// the prefill matmuls; both tile kernels) within 3e-3, and the greedy tokens of a generation equal.
TEST(quantized_matrices_match_the_reference) {
    const DType F16 = DType::F16, Q40 = DType::Q4_0, Q50 = DType::Q5_0, Q80 = DType::Q8_0, Q4K = DType::Q4_K, Q6K = DType::Q6_K;
    (void)F16;
    for (const QuantPlan & plan : {QuantPlan{"q4_0", Q40, Q40, Q40, Q40, Q40, Q40, Q40, Q40, Q40}, QuantPlan{"q5_0", Q50, Q50, Q50, Q50, Q50, Q50, Q50, Q50, Q50},
                                   QuantPlan{"q8_0", Q80, Q80, Q80, Q80, Q80, Q80, Q80, Q80, Q80}, QuantPlan{"q4_K", Q4K, Q4K, Q4K, Q4K, Q4K, Q4K, Q4K, Q4K, Q4K},
                                   QuantPlan{"q6_K", Q6K, Q6K, Q6K, Q6K, Q6K, Q6K, Q6K, Q6K, Q6K},
                                   QuantPlan{"Q4_K_M-like mix", Q80, Q6K, Q50, Q50, Q80, Q50, Q50, Q4K, Q6K}}) {
        bool exact = true;
        const QuantTiny qt = make_quant(plan, 11, &exact);
        EXPECT(exact);   // the reference's f16 weights are ggml's values exactly
        for (int tile : {1, 2}) {
            LlamaDecoderConfig cfg = qt.ref.cfg;
            cfg.prefill_tile = tile;
            auto dec = MetalLlamaDecoder::create(cfg, qt.w);
            EXPECT(dec.is_ok());
            if (!dec.is_ok()) { std::printf("        %s: %s\n", plan.name, dec.status().to_string().c_str()); break; }
            MetalLlamaDecoder & d = *dec.value();
            Reference ref(qt.ref);
            std::vector<double> want;
            for (int32_t t : kPrompt) want = ref.step(ref.embedding(t));
            EXPECT(d.eval_tokens(kPrompt).is_ok());
            const double batched = rel_l2(d.logits().value(), want);
            EXPECT(batched < 3e-3);
            EXPECT_EQ(d.pending_token().value(), argmax(want));
            d.reset();
            for (int32_t t : kPrompt) EXPECT(d.eval_tokens({t}).is_ok());
            const double single = rel_l2(d.logits().value(), want);
            EXPECT(single < 1e-4);
            std::vector<int32_t> want_tokens;
            int32_t t = argmax(want);
            for (int i = 0; i < 16; i++) { want_tokens.push_back(t); t = argmax(ref.step(ref.embedding(t))); }
            std::vector<int32_t> got;
            EXPECT(d.generate(16, 2, [&](int32_t tok) { got.push_back(tok); return true; }).is_ok());
            EXPECT(got == want_tokens);
            if (tile == 1 || batched >= 3e-3) {
                std::printf("        %-16s prompt logits relative L2: %.2e one token per call, %.2e in one call (tile %d); %zu greedy tokens %s\n", plan.name, single,
                            batched, tile, got.size(), got == want_tokens ? "equal" : "DIFFERENT");
            }
        }
    }
    // Longer prompts in one call: 120 tokens (the 64x32 kernels unpacking the quantized tiles) and 330 (from 320 tokens on, each matrix dequantized to half
    // once first), every format, against the reference; then a decode step.
    for (const QuantPlan & plan : {QuantPlan{"q4_0", Q40, Q40, Q40, Q40, Q40, Q40, Q40, Q40, Q40}, QuantPlan{"q5_0", Q50, Q50, Q50, Q50, Q50, Q50, Q50, Q50, Q50},
                                   QuantPlan{"q8_0", Q80, Q80, Q80, Q80, Q80, Q80, Q80, Q80, Q80}, QuantPlan{"q4_K", Q4K, Q4K, Q4K, Q4K, Q4K, Q4K, Q4K, Q4K, Q4K},
                                   QuantPlan{"q6_K", Q6K, Q6K, Q6K, Q6K, Q6K, Q6K, Q6K, Q6K, Q6K},
                                   QuantPlan{"Q4_K_M-like mix", Q80, Q6K, Q50, Q50, Q80, Q50, Q50, Q4K, Q6K}}) {
        bool exact_long = true;
        QuantTiny qt = make_quant(plan, 31, &exact_long);
        qt.ref.cfg.n_ctx = 400;
        qt.ref.cfg.n_batch = 400;
        auto dec = MetalLlamaDecoder::create(qt.ref.cfg, qt.w);
        EXPECT(dec.is_ok() && exact_long);
        if (!dec.is_ok()) continue;
        MetalLlamaDecoder & d = *dec.value();
        for (int len : {120, 330}) {
            std::vector<int32_t> prompt;
            for (int i = 0; i < len; i++) prompt.push_back((i * 13 + 5) % qt.ref.cfg.n_vocab);
            Reference ref(qt.ref);
            std::vector<double> want;
            for (int32_t t : prompt) want = ref.step(ref.embedding(t));
            d.reset();
            EXPECT(d.eval_tokens(prompt).is_ok());
            const double err = rel_l2(d.logits().value(), want);
            EXPECT(err < 3e-3);
            EXPECT_EQ(d.pending_token().value(), argmax(want));
            const int32_t next = argmax(want);
            want = ref.step(ref.embedding(next));
            EXPECT(d.eval_tokens({next}).is_ok());
            const double step_err = rel_l2(d.logits().value(), want);
            EXPECT(step_err < 3e-3);
            std::printf("        %-16s %d-token prompt in one call: logits relative L2 %.2e, next step %.2e\n", plan.name, len, err, step_err);
        }
    }
    // A row that is not a whole number of blocks, and gate/up of different types, are refused.
    bool exact = true;
    QuantTiny qt = make_quant(QuantPlan{"x", Q40, Q40, Q40, Q40, Q40, Q40, Q40, Q40, Q40}, 12, &exact);
    LlamaDecoderWeights w = qt.w;
    w.layers[1].ffn_up.dtype = Q80;
    EXPECT(MetalLlamaDecoder::create(qt.ref.cfg, w).status().code() == Code::Unsupported);
}

// Decode steps from 512 positions on attend through attention_split + attention_combine (the cache split into chunks over many threadgroups); both rotary
// layouts, grouped-query attention, against the reference: the logits of each step and the greedy tokens.
TEST(long_context_decode_matches_the_reference) {
    for (bool qwen : {false, true}) {
        const TinyModel m = make_tiny(qwen ? 21 : 22, 64, 600, qwen, 64);
        auto dec = MetalLlamaDecoder::create(m.cfg, weights_of(m));
        EXPECT(dec.is_ok());
        if (!dec.is_ok()) { std::printf("        %s\n", dec.status().to_string().c_str()); return; }
        MetalLlamaDecoder & d = *dec.value();
        Reference ref(m);
        std::vector<int32_t> prompt;
        for (int i = 0; i < 520; i++) prompt.push_back((i * 7 + 3) % m.cfg.n_vocab);
        std::vector<double> want;
        for (int32_t t : prompt) want = ref.step(ref.embedding(t));
        EXPECT(d.eval_tokens(prompt).is_ok());
        // One token per call from position 520 on: each step's logits against the reference (f32 activations: close), greedy tokens fed back.
        double worst = 0;
        int32_t t = argmax(want);
        bool same = true;
        for (int i = 0; i < 24; i++) {
            want = ref.step(ref.embedding(t));
            EXPECT(d.eval_tokens({t}).is_ok());
            worst = std::max(worst, rel_l2(d.logits().value(), want));
            same = same && d.pending_token().value() == argmax(want);
            t = argmax(want);
        }
        std::printf("        %s, positions 520..543: worst logits relative L2 %.2e, greedy tokens %s\n", qwen ? "qwen-style" : "llama", worst, same ? "equal" : "DIFFERENT");
        EXPECT(worst < 3e-3);   // the 520-token prompt went through the half-rounded prefill matmuls
        EXPECT(same);
    }
}

// Both prefill kernels (16x16 tiles, 64-row x 32-token tiles) on a 45-token prompt (not a multiple of either tile) against the reference, llama and qwen
// blocks; the tokens that follow must be the reference's.
TEST(both_prefill_kernels_match_the_reference) {
    std::vector<int32_t> prompt;
    for (int i = 0; i < 45; i++) prompt.push_back((i * 7 + 3) % 50);
    for (int variant = 0; variant < 4; variant++) {
        const bool qwen = variant % 2 == 1;
        const int32_t hd = variant < 2 ? 16 : 64;            // 64: flash-attention prefill
        TinyModel m = make_tiny(5, 64, 128, qwen, hd);
        Reference ref(m);
        std::vector<double> want;
        for (int32_t t : prompt) want = ref.step(ref.embedding(t));
        std::vector<int32_t> want_tokens;
        int32_t t = argmax(want);
        for (int i = 0; i < 8; i++) { want_tokens.push_back(t); t = argmax(ref.step(ref.embedding(t))); }
        for (int32_t tile : {1, 2, 3}) {
            // 3: the 64x32 kernel with the prompt fed in chunks of 20 (later chunks start mid-sequence)
            m.cfg.prefill_tile = tile == 3 ? 2 : tile;
            m.cfg.n_batch = tile == 3 ? 20 : 64;
            auto dec = MetalLlamaDecoder::create(m.cfg, weights_of(m));
            EXPECT(dec.is_ok());
            if (!dec.is_ok()) continue;
            EXPECT(dec.value()->eval_tokens(prompt).is_ok());
            const double rel = rel_l2(dec.value()->logits().value(), want);
            std::printf("        %s block, head_dim %d, prefill %s: logits relative L2 %.2e\n", qwen ? "qwen" : "llama", hd,
                        tile == 1 ? "16x16" : tile == 2 ? "64x32" : "64x32 in chunks of 20", rel);
            EXPECT(rel < 3e-3);
            std::vector<int32_t> got;
            EXPECT(dec.value()->generate(8, 1, [&](int32_t tok) { got.push_back(tok); return true; }).is_ok());
            EXPECT(got == want_tokens);
        }
    }
    LlamaDecoderConfig bad = make_tiny(5, 64, 128).cfg;
    bad.prefill_tile = 3;
    EXPECT(MetalLlamaDecoder::create(bad, weights_of(make_tiny(5, 64, 128))).status().code() == Code::InvalidArgument);
}

TEST(chunked_prefill_embeddings_and_queue_depth_give_the_same_tokens) {
    const TinyModel m = make_tiny(2, 64, 128);
    const TinyModel small_batch = make_tiny(2, 4, 128);  // same weights, prompt split into chunks of 4
    auto a = MetalLlamaDecoder::create(m.cfg, weights_of(m));
    auto b = MetalLlamaDecoder::create(small_batch.cfg, weights_of(small_batch));
    EXPECT(a.is_ok() && b.is_ok());
    if (!a.is_ok() || !b.is_ok()) return;
    auto run = [](MetalLlamaDecoder & d, int depth) {
        std::vector<int32_t> out;
        (void)d.generate(16, depth, [&](int32_t tok) { out.push_back(tok); return true; });
        return out;
    };
    EXPECT(a.value()->eval_tokens(kPrompt).is_ok());
    const std::vector<int32_t> base = run(*a.value(), 1);
    EXPECT(b.value()->eval_tokens(kPrompt).is_ok());
    EXPECT(run(*b.value(), 6) == base);
    // The same prompt as embeddings (the table rows as floats) instead of token ids.
    std::vector<float> emb;
    for (int32_t t : kPrompt) for (int i = 0; i < m.cfg.n_embd; i++) emb.push_back(static_cast<float>(m.embd[static_cast<size_t>(t * m.cfg.n_embd + i)]));
    a.value()->reset();
    EXPECT_EQ(a.value()->position(), 0);
    EXPECT(a.value()->eval_embeddings(emb.data(), static_cast<int32_t>(kPrompt.size())).is_ok());
    EXPECT(run(*a.value(), 2) == base);
    // Continuing after a generation: feeding the last delivered token gives what a longer generation would have produced next.
    a.value()->reset();
    EXPECT(a.value()->eval_tokens(kPrompt).is_ok());
    std::vector<int32_t> first;
    EXPECT(a.value()->generate(8, 4, [&](int32_t tok) { first.push_back(tok); return true; }).is_ok());
    EXPECT(a.value()->eval_tokens({first.back()}).is_ok());
    EXPECT_EQ(a.value()->pending_token().value(), base[8]);
}

TEST(stopping_early_and_errors) {
    const TinyModel m = make_tiny(3, 8, 32);
    auto dec = MetalLlamaDecoder::create(m.cfg, weights_of(m));
    EXPECT(dec.is_ok());
    if (!dec.is_ok()) return;
    MetalLlamaDecoder & d = *dec.value();
    EXPECT(!d.pending_token().is_ok());
    EXPECT(!d.logits().is_ok());
    EXPECT(!d.generate(4, 2, [](int32_t) { return true; }).is_ok());
    EXPECT(!d.eval_tokens({}).is_ok());
    EXPECT(!d.eval_tokens({-1}).is_ok());
    EXPECT(!d.eval_tokens({m.cfg.n_vocab}).is_ok());
    EXPECT(!d.eval_embeddings(nullptr, 1).is_ok());
    const float nan_row[64] = {std::nanf("")};
    EXPECT(!d.eval_embeddings(nan_row, 1).is_ok());
    EXPECT(d.eval_tokens(kPrompt).is_ok());
    EXPECT(!d.generate(0, 1, [](int32_t) { return true; }).is_ok());
    EXPECT(!d.generate(4, 0, [](int32_t) { return true; }).is_ok());
    EXPECT(d.generate(32, 1, [](int32_t) { return true; }).code() == Code::ResourceExhausted);  // 11 + 31 > 32
    int calls = 0;
    EXPECT(d.generate(10, 4, [&](int32_t) { return ++calls < 3; }).is_ok());
    EXPECT_EQ(calls, 3);
    EXPECT_EQ(d.position(), static_cast<int32_t>(kPrompt.size()) + 2);
    std::vector<int32_t> too_long(40, 1);
    d.reset();
    EXPECT(d.eval_tokens(too_long).code() == Code::ResourceExhausted);
    // Shape and dtype checks at creation.
    LlamaDecoderWeights w = weights_of(m);
    w.layers[1].ffn_up.rows = 95;
    EXPECT(MetalLlamaDecoder::create(m.cfg, w).status().code() == Code::InvalidArgument);
    w = weights_of(m);
    w.layers[0].ffn_down.dtype = DType::F32;
    EXPECT(MetalLlamaDecoder::create(m.cfg, w).status().code() == Code::Unsupported);
    w = weights_of(m);
    w.layers.pop_back();
    EXPECT(!MetalLlamaDecoder::create(m.cfg, w).is_ok());
    LlamaDecoderConfig bad = m.cfg;
    bad.n_head_kv = 3;
    EXPECT(!MetalLlamaDecoder::create(bad, weights_of(m)).is_ok());
    bad = m.cfg;
    bad.head_dim = 24;  // not a power of two: the attention kernel's thread layout needs one
    EXPECT(MetalLlamaDecoder::create(bad, weights_of(m)).status().code() == Code::Unsupported);
}

TEST(loader_refuses_what_it_cannot_run) {
    EXPECT(load_native_llama("/nonexistent/model.gguf").status().code() == Code::NotFound);
    NativeLlamaOptions o;
    o.n_ctx = 0;
    EXPECT(load_native_llama("/nonexistent/model.gguf", o).status().code() == Code::InvalidArgument);
    const char * dir = std::getenv("DOOMS_VLM_DIR");
    if (!dir) return;
    // A quantized file of the supported block types loads (the Q4_K_M one mixes q5_0, q8_0, q4_K and q6_K); other types stay Unsupported (the callers then
    // fall back to llama.cpp).
    const std::string q = std::string(dir) + "/text-Q4_K_M.gguf";
    if (FILE * f = std::fopen(q.c_str(), "rb")) {
        std::fclose(f);
        auto r = load_native_llama(q);
        EXPECT(r.is_ok() || r.status().code() == Code::Unsupported);
        if (!r.is_ok()) std::printf("        %s\n", r.status().to_string().c_str());
    }
}

// SmolVLM's text model: the Metal decoder against llama.cpp on the same prompt embeddings (prefix tokens, transformers' image embeddings, rest of the prompt),
// and teacher-forced against transformers' greedy tokens like the TextDecoder test.
TEST(smolvlm_text_model_matches_llama_cpp_and_transformers) {
    const char * env = std::getenv("DOOMS_VLM_DIR");
    if (!env) { std::printf("        (skipped: set DOOMS_VLM_DIR)\n"); return; }
    const std::string dir = env;
    auto native = load_native_llama(dir + "/text-f16.gguf");
    auto llama = TextDecoder::load(dir + "/text-f16.gguf");
    auto cfg = GgufFile::open(dir + "/vlm-config.gguf");
    auto ref = GgufFile::open(dir + "/reference.gguf");
    EXPECT(native.is_ok() && llama.is_ok() && cfg.is_ok() && ref.is_ok());
    if (!native.is_ok() || !llama.is_ok() || !cfg.is_ok() || !ref.is_ok()) { if (!native.is_ok()) std::printf("        %s\n", native.status().to_string().c_str()); return; }
    MetalLlamaDecoder & d = *native.value();
    TextDecoder & t = llama.value();
    const GgufFile & c = cfg.value();
    const std::string prefix = c.get_string("vlm_config.prefix").value(), infix = c.get_string("vlm_config.infix").value(), suffix = c.get_string("vlm_config.suffix").value();
    const int64_t n_image = static_cast<int64_t>(c.get_uint("vlm_config.n_image_tokens").value());
    auto tensor = [&](const std::string & name) {
        const GgufTensorInfo * info = ref.value().find_tensor(name).value();
        const float * p = reinterpret_cast<const float *>(ref.value().tensor_data(*info));
        return std::vector<float>(p, p + info->shape.numel());
    };
    for (int k = 0; k < 4; k++) {
        const std::string p = "case" + std::to_string(k) + ".";
        const std::vector<float> image = tensor(p + "image_embeddings"), want_logits = tensor(p + "logits");
        std::vector<int32_t> generated;
        for (float v : tensor(p + "generated")) generated.push_back(static_cast<int32_t>(v));
        EXPECT_EQ(static_cast<int64_t>(image.size()), n_image * t.embedding_dim());
        const std::string prompt = ref.value().get_string(p + "prompt").value();
        std::vector<float> all = t.embed_tokens(t.tokenize(prefix, true).value()).value();
        all.insert(all.end(), image.begin(), image.end());
        const std::vector<float> tail = t.embed_tokens(t.tokenize(infix + prompt + suffix, true).value()).value();
        all.insert(all.end(), tail.begin(), tail.end());
        const int32_t n = static_cast<int32_t>(all.size() / static_cast<size_t>(t.embedding_dim()));
        d.reset();
        t.reset();
        EXPECT(d.eval_embeddings(all.data(), n).is_ok());
        EXPECT(t.eval_embeddings(all.data(), n).is_ok());
        const std::vector<float> ours = d.logits().value(), theirs = t.logits().value();
        std::vector<double> theirs_d(theirs.begin(), theirs.end()), want_d(want_logits.begin(), want_logits.end());
        const double vs_llama = rel_l2(ours, theirs_d), vs_hf = rel_l2(ours, want_d);
        // Teacher-forced: transformers' token must be our argmax (within a hair) at every step.
        double worst_gap = 0;
        size_t free_agree = 0;
        bool agreeing = true;
        for (size_t i = 0; i < generated.size(); i++) {
            const std::vector<float> cur = d.logits().value();
            worst_gap = std::max(worst_gap, static_cast<double>(*std::max_element(cur.begin(), cur.end()) - cur[static_cast<size_t>(generated[i])]));
            if (agreeing && d.pending_token().value() == generated[i]) free_agree++; else agreeing = false;
            EXPECT(d.eval_tokens({generated[i]}).is_ok());
        }
        std::printf("        case %d: logits relative L2 vs llama.cpp %.2e, vs transformers %.2e; free-running greedy equal for %zu of %zu tokens, largest margin %.4f\n",
                    k, vs_llama, vs_hf, free_agree, generated.size(), worst_gap);
        EXPECT(vs_llama < 5e-3);
        EXPECT(vs_hf < 2e-2);
        EXPECT(worst_gap < 0.1);
        EXPECT(free_agree >= 3);
    }
}

// Qwen2.5-0.5B (f16 GGUF, DOOMS_QWEN_F16): logits against llama.cpp on a chat prompt, and llama.cpp's own greedy continuation teacher-forced: each of
// its tokens must be our argmax within a hair (f16 noise can swap near-ties in a free-running comparison).
TEST(qwen2_matches_llama_cpp) {
    const char * path = std::getenv("DOOMS_QWEN_GGUF") ? std::getenv("DOOMS_QWEN_GGUF") : std::getenv("DOOMS_QWEN_F16");
    if (!path) { std::printf("        (skipped: set DOOMS_QWEN_F16 or DOOMS_QWEN_GGUF)\n"); return; }
    NativeLlamaOptions no;
    no.n_ctx = 512;
    auto native = load_native_llama(path, no);
    TextDecoderOptions to;
    to.n_ctx = 512;
    auto llama = TextDecoder::load(path, to);
    EXPECT(native.is_ok() && llama.is_ok());
    if (!native.is_ok() || !llama.is_ok()) { if (!native.is_ok()) std::printf("        %s\n", native.status().to_string().c_str()); return; }
    MetalLlamaDecoder & d = *native.value();
    TextDecoder & t = llama.value();
    for (const char * question : {"Name three colours.", "Explain in one sentence why the sky is blue.", "Write a haiku about a robot arm."}) {
        const std::string prompt = t.apply_chat_template({{"user", question}}).value();
        const std::vector<int32_t> ids = t.tokenize(prompt, true).value();
        t.reset();
        d.reset();
        EXPECT(t.eval_tokens(ids).is_ok());
        EXPECT(d.eval_tokens(ids).is_ok());
        const std::vector<float> theirs = t.logits().value();
        const double vs_llama = rel_l2(d.logits().value(), std::vector<double>(theirs.begin(), theirs.end()));
        double worst_gap = 0;
        size_t agree = 0;
        bool agreeing = true;
        for (int i = 0; i < 48; i++) {
            const int32_t want = t.greedy_next().value();
            const std::vector<float> ours = d.logits().value();
            worst_gap = std::max(worst_gap, static_cast<double>(*std::max_element(ours.begin(), ours.end()) - ours[static_cast<size_t>(want)]));
            if (agreeing && d.pending_token().value() == want) agree++; else agreeing = false;
            if (t.is_end_of_generation(want)) break;
            EXPECT(t.eval_tokens({want}).is_ok());
            EXPECT(d.eval_tokens({want}).is_ok());
        }
        std::printf("        \"%s\": logits relative L2 vs llama.cpp %.2e, free-running greedy equal for %zu tokens, largest margin %.4f\n", question, vs_llama, agree, worst_gap);
        EXPECT(vs_llama < 1e-2);
        EXPECT(worst_gap < 0.15);
        EXPECT(agree >= 8);
    }
}

int main() { return dooms_test::run_all(); }
