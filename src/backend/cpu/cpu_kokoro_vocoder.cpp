#include "backend/cpu/cpu_kokoro_vocoder.h"

#define ACCELERATE_NEW_LAPACK
#include <Accelerate/Accelerate.h>

#include <dispatch/dispatch.h>

#include <algorithm>
#include <cmath>
#include <cstring>

namespace dooms {

namespace {

// Fixed by Kokoro's architecture (kokoro/istftnet.py, kokoro/modules.py); the same constants as the Metal vocoder.
constexpr float kBlockSlope = 0.2f, kGenSlope = 0.1f, kPostSlope = 0.01f, kNormEps = 1e-5f;
constexpr int kCurveKernel = 3, kCurveStride = 2, kCurvePad = 1;
constexpr int kDecodeBlocks = 4;
constexpr int kResDilations = 3;
constexpr int kDilation[kResDilations] = {1, 3, 5};
constexpr int kColumns = 2048;   // columns per im2col block
const float kInvSqrt2 = 1.f / std::sqrt(2.f);

enum Act : int { kNone = 0, kLeaky = 1, kSnake = 2 };
enum Flag : int { kResidual = 2, kAccumulate = 4 };

struct CConv {
    std::vector<std::vector<float>> w;   // [oc][ic * taps] per output phase
    std::vector<float> b;                // empty: no bias
    int oc = 0, ic = 0, taps = 0, stride = 1, pad = 0, dil = 1;
    bool transposed = false;
};
struct CAda { std::vector<float> w, b; int channels = 0, gb_off = 0; };
struct CResBlk { CAda n1, n2; CConv c1, c2, sc; bool up = false; std::vector<float> pool_w, pool_b; int cin = 0, cout = 0; };
struct CAdaRes1 { int ch = 0, k = 0; CAda a1[kResDilations], a2[kResDilations]; CConv c1[kResDilations], c2[kResDilations]; std::vector<float> alpha1[kResDilations], alpha2[kResDilations]; };

float act(float v, int a, float slope, float alpha) {
    if (a == kLeaky) return v >= 0.f ? v : v * slope;
    if (a == kSnake) { const float s = std::sin(alpha * v); return v + s * s / alpha; }
    return v;
}

}  // namespace

struct CpuKokoroVocoder::Impl {
    KokoroVocoderConfig cfg;
    int out_ch = 0, gen_up = 1, style_rows = 0;
    std::vector<float> style_w, style_b, gb;
    CConv f0_conv, n_conv, asr_res, conv_post;
    CResBlk encode;
    std::vector<CResBlk> decode;
    std::vector<CConv> noise_convs, ups;
    std::vector<CAdaRes1> noise_res, resblocks;
    std::vector<float> prep, col, tmp;   // the activated input of the next convolution; im2col block; matmul block

    // y[oc][t] over tout positions of a convolution of x[ic][tin] (nearest-upsampled by `up` first). Epilogue as the Metal kernel: + bias, + res, * post_scale,
    // accumulate into y.
    void conv(const CConv & c, const float * x, int tin, int up, float * y, int tout, const float * res, int flags, float post_scale, int out_shift = 0) {
        const int phases = c.transposed ? c.stride : 1, K = c.ic * c.taps, vlen = tin * up;
        for (int ph = 0; ph < phases; ph++) {
            int ncol, in_stride, dil, in_off, out_stride, out_off;
            if (c.transposed) { in_stride = 1; dil = -1; in_off = 0; out_stride = c.stride; out_off = ph - c.pad + out_shift; ncol = (tout - out_off + c.stride - 1) / c.stride; }
            else { in_stride = c.stride; dil = c.dil; in_off = -c.pad; out_stride = 1; out_off = out_shift; ncol = tout - out_shift; }
            for (int q0 = 0; q0 < ncol; q0 += kColumns) {
                const int B = std::min(kColumns, ncol - q0);
                const bool contiguous = in_stride == 1 && up == 1;   // a tap's columns are one slice of the input row
                float * colp = col.data();
                dispatch_apply(static_cast<size_t>(c.ic), dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), ^(size_t ic) {
                    const float * xr = x + ic * static_cast<size_t>(tin);
                    for (int j = 0; j < c.taps; j++) {
                        float * cr = colp + (ic * static_cast<size_t>(c.taps) + static_cast<size_t>(j)) * static_cast<size_t>(B);
                        if (contiguous) {
                            const int s0 = q0 + j * dil + in_off;   // source of column b: s0 + b
                            const int lo = std::max(0, -s0), hi = std::min(B, vlen - s0);
                            if (hi <= lo) { std::memset(cr, 0, static_cast<size_t>(B) * sizeof(float)); continue; }
                            if (lo > 0) std::memset(cr, 0, static_cast<size_t>(lo) * sizeof(float));
                            std::memcpy(cr + lo, xr + s0 + lo, static_cast<size_t>(hi - lo) * sizeof(float));
                            if (hi < B) std::memset(cr + hi, 0, static_cast<size_t>(B - hi) * sizeof(float));
                        } else {
                            for (int b = 0; b < B; b++) {
                                const int src = (q0 + b) * in_stride + j * dil + in_off;
                                cr[b] = (src >= 0 && src < vlen) ? xr[up == 1 ? src : src / up] : 0.f;
                            }
                        }
                    }
                });
                cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, c.oc, B, K, 1.f, c.w[static_cast<size_t>(ph)].data(), K, col.data(), B, 0.f, tmp.data(), B);
                const float * tp = tmp.data();
                dispatch_apply(static_cast<size_t>(c.oc), dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), ^(size_t oi) {
                    const int o = static_cast<int>(oi);
                    const float bias = c.b.empty() ? 0.f : c.b[static_cast<size_t>(o)];
                    for (int b = 0; b < B; b++) {
                        const int t = (q0 + b) * out_stride + out_off;
                        if (t < 0 || t >= tout) continue;
                        const size_t i = static_cast<size_t>(o) * tout + t;
                        float v = tp[static_cast<size_t>(o) * B + b] + bias;
                        if (flags & kResidual) v += res[i];
                        v *= post_scale;
                        y[i] = (flags & kAccumulate) ? y[i] + v : v;
                    }
                });
            }
        }
    }

    // prep = activate(x) (ch rows of len).
    void prep_act(const float * x, int ch, int len, int a, float slope) {
        float * pp = prep.data();
        dispatch_apply(static_cast<size_t>(ch), dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), ^(size_t c) {
            for (size_t i = c * static_cast<size_t>(len); i < (c + 1) * static_cast<size_t>(len); i++) pp[i] = act(x[i], a, slope, 1.f);
        });
    }
    // prep = activate(AdaIN(x)): instance-norm statistics (two passes), the style's gamma / beta, one affine map per channel, then the activation.
    void prep_adain(const CAda & a, const float * x, int len, int ac, float slope, const std::vector<float> * alpha) {
        float * pp = prep.data();
        const float * gbp = gb.data();
        dispatch_apply(static_cast<size_t>(a.channels), dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), ^(size_t ci) {
            const int c = static_cast<int>(ci);
            const float * xr = x + static_cast<size_t>(c) * len;
            double mean = 0;
            for (int t = 0; t < len; t++) mean += xr[t];
            mean /= len;
            double var = 0;
            for (int t = 0; t < len; t++) { const double d = xr[t] - mean; var += d * d; }
            const float rstd = static_cast<float>(1.0 / std::sqrt(var / len + kNormEps));
            const float g1 = 1.f + gbp[a.gb_off + c], beta = gbp[a.gb_off + a.channels + c];
            const float scale = g1 * a.w[static_cast<size_t>(c)] * rstd, shift = g1 * a.b[static_cast<size_t>(c)] - static_cast<float>(mean) * scale + beta;
            float * pr = pp + static_cast<size_t>(c) * len;
            if (ac == kSnake) {   // x + sin(a x)^2 / a with the sines of the whole row at once (vForce)
                const float al = (*alpha)[static_cast<size_t>(c)];
                std::vector<float> s(static_cast<size_t>(len));
                for (int t = 0; t < len; t++) { pr[t] = xr[t] * scale + shift; s[static_cast<size_t>(t)] = al * pr[t]; }
                const int n = len;
                vvsinf(s.data(), s.data(), &n);
                const float inv = 1.f / al;
                for (int t = 0; t < len; t++) pr[t] += s[static_cast<size_t>(t)] * s[static_cast<size_t>(t)] * inv;
            } else {
                for (int t = 0; t < len; t++) pr[t] = act(xr[t] * scale + shift, ac, slope, 1.f);
            }
        });
    }

    void res_blk(const CResBlk & b, const float * x, int T, float * y, std::vector<float> & h1, std::vector<float> & pool) {
        const int To = b.up ? 2 * T : T;
        prep_adain(b.n1, x, T, kLeaky, kBlockSlope, nullptr);
        if (b.up) {   // depthwise ConvTranspose1d (k 3, stride 2, padding 1, output padding 1) of the prepped input
            for (int c = 0; c < b.cin; c++) {
                for (int t = 0; t < To; t++) {
                    float v = b.pool_b[static_cast<size_t>(c)];
                    for (int k = 0; k < 3; k++) {
                        const int u = t + 1 - k;
                        if (u < 0 || (u & 1)) continue;
                        const int src = u >> 1;
                        if (src < T) v += b.pool_w[static_cast<size_t>(c) * 3 + k] * prep[static_cast<size_t>(c) * T + src];
                    }
                    pool[static_cast<size_t>(c) * To + t] = v;
                }
            }
            conv(b.c1, pool.data(), To, 1, h1.data(), To, nullptr, 0, 1.f);
        } else {
            conv(b.c1, prep.data(), T, 1, h1.data(), To, nullptr, 0, 1.f);
        }
        conv(b.sc, x, T, b.up ? 2 : 1, y, To, nullptr, 0, 1.f);
        prep_adain(b.n2, h1.data(), To, kLeaky, kBlockSlope, nullptr);
        conv(b.c2, prep.data(), To, 1, y, To, y, kResidual, kInvSqrt2);
    }

    void ada_res1(const CAdaRes1 & b, const float * x, int T, float * r, float * t, float * xs, float scale, bool accumulate) {
        const float * cur = x;
        for (int d = 0; d < kResDilations; d++) {
            prep_adain(b.a1[d], cur, T, kSnake, 0.f, &b.alpha1[d]);
            conv(b.c1[d], prep.data(), T, 1, t, T, nullptr, 0, 1.f);
            prep_adain(b.a2[d], t, T, kSnake, 0.f, &b.alpha2[d]);
            const bool last = d + 1 == kResDilations;
            float * out = (last && xs) ? xs : r;
            conv(b.c2[d], prep.data(), T, 1, out, T, cur, kResidual | ((last && xs && accumulate) ? kAccumulate : 0), (last && xs) ? scale : 1.f);
            cur = r;
        }
    }
};

CpuKokoroVocoder::CpuKokoroVocoder(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}
CpuKokoroVocoder::~CpuKokoroVocoder() = default;
int32_t CpuKokoroVocoder::har_length(int32_t frames) const { return frames * impl_->gen_up + 1; }

Result<std::unique_ptr<CpuKokoroVocoder>> CpuKokoroVocoder::create(const KokoroVocoderConfig & c, const KokoroTensors & t) {
    if (c.hidden < 1 || c.style_dim < 1 || c.n_fft < 2 || c.initial_channel < 2) return invalid_argument("kokoro vocoder config: sizes must be positive");
    if (c.upsample_rates.empty() || c.upsample_rates.size() != c.upsample_kernels.size() || c.resblock_kernels.empty()) {
        return invalid_argument("kokoro vocoder config: one kernel size per upsampling rate, at least one residual block kernel");
    }
    auto impl = std::make_unique<Impl>();
    Impl & m = *impl;
    m.cfg = c;
    m.out_ch = c.n_fft + 2;
    m.gen_up = 2;
    for (int32_t r : c.upsample_rates) m.gen_up *= r;
    Status st;
    auto find = [&](const std::string & name, const std::vector<int64_t> & shape) -> const KokoroTensor * {
        if (!st.is_ok()) return nullptr;
        auto it = t.find("dec." + name);
        if (it == t.end() || !it->second.data) { st = invalid_argument("kokoro vocoder: tensor 'dec." + name + "' is missing"); return nullptr; }
        if (it->second.shape != shape) { st = invalid_argument("kokoro vocoder: tensor 'dec." + name + "' has the wrong shape"); return nullptr; }
        if (it->second.dtype != DType::F16 && it->second.dtype != DType::F32) { st = invalid_argument("kokoro vocoder: tensor 'dec." + name + "' must be f16 or f32"); return nullptr; }
        return &it->second;
    };
    auto vals = [](const KokoroTensor & k) {
        size_t n = 1;
        for (int64_t s : k.shape) n *= static_cast<size_t>(s);
        std::vector<float> out(n);
        for (size_t i = 0; i < n; i++) out[i] = k.dtype == DType::F16 ? static_cast<float>(static_cast<const __fp16 *>(k.data)[i]) : static_cast<const float *>(k.data)[i];
        return out;
    };
    auto conv = [&](const std::string & name, int oc, int ic, int k, int stride, int pad, int dil, bool bias, CConv * out) {
        const KokoroTensor * w = find(name + ".weight", {oc, ic, k});
        if (!w) return;
        out->oc = oc; out->ic = ic; out->taps = k; out->stride = stride; out->pad = pad; out->dil = dil; out->transposed = false;
        out->w = {vals(*w)};   // [oc][ic][k] is already [oc][ic * taps + j]
        out->b.clear();
        if (bias) { const KokoroTensor * b = find(name + ".bias", {oc}); if (b) out->b = vals(*b); }
    };
    auto convt = [&](const std::string & name, int ic, int oc, int k, int stride, int pad, CConv * out) {
        if (k % stride != 0) { st = unsupported("kokoro vocoder: transposed convolution '" + name + "' has a kernel that is not a multiple of its stride"); return; }
        const KokoroTensor * w = find(name + ".weight", {ic, oc, k});
        const KokoroTensor * b = find(name + ".bias", {oc});
        if (!w || !b) return;
        const std::vector<float> W = vals(*w);
        out->oc = oc; out->ic = ic; out->taps = k / stride; out->stride = stride; out->pad = pad; out->dil = -1; out->transposed = true;
        out->w.assign(static_cast<size_t>(stride), std::vector<float>(static_cast<size_t>(oc) * ic * out->taps));
        for (int ph = 0; ph < stride; ph++)
            for (int o = 0; o < oc; o++) for (int i = 0; i < ic; i++) for (int j = 0; j < out->taps; j++)
                out->w[static_cast<size_t>(ph)][(static_cast<size_t>(o) * ic + i) * out->taps + j] = W[(static_cast<size_t>(i) * oc + o) * k + static_cast<size_t>(ph + j * stride)];
        out->b = vals(*b);
    };
    auto ada = [&](const std::string & name, int ch, CAda * out) {
        const KokoroTensor *nw = find(name + ".norm.weight", {ch}), *nb = find(name + ".norm.bias", {ch});
        const KokoroTensor *fw = find(name + ".fc.weight", {2 * ch, c.style_dim}), *fb = find(name + ".fc.bias", {2 * ch});
        if (!nw || !nb || !fw || !fb) return;
        out->channels = ch; out->w = vals(*nw); out->b = vals(*nb);
        out->gb_off = m.style_rows;
        m.style_rows += 2 * ch;
        const std::vector<float> W = vals(*fw), B = vals(*fb);
        m.style_w.insert(m.style_w.end(), W.begin(), W.end());
        m.style_b.insert(m.style_b.end(), B.begin(), B.end());
    };
    auto res_blk = [&](const std::string & name, int cin, int cout, bool up, CResBlk * b) {
        b->cin = cin; b->cout = cout; b->up = up;
        ada(name + ".norm1", cin, &b->n1);
        conv(name + ".conv1", cout, cin, 3, 1, 1, 1, true, &b->c1);
        ada(name + ".norm2", cout, &b->n2);
        conv(name + ".conv2", cout, cout, 3, 1, 1, 1, true, &b->c2);
        conv(name + ".conv1x1", cout, cin, 1, 1, 0, 1, false, &b->sc);
        if (up) {
            const KokoroTensor *pw = find(name + ".pool.weight", {cin, 1, 3}), *pb = find(name + ".pool.bias", {cin});
            if (pw && pb) { b->pool_w = vals(*pw); b->pool_b = vals(*pb); }
        }
    };
    auto ada_res1 = [&](const std::string & name, int ch, int k, CAdaRes1 * b) {
        b->ch = ch; b->k = k;
        for (int d = 0; d < kResDilations; d++) {
            const std::string i = std::to_string(d);
            const int dil = kDilation[d];
            conv(name + ".convs1." + i, ch, ch, k, 1, (k * dil - dil) / 2, dil, true, &b->c1[d]);
            conv(name + ".convs2." + i, ch, ch, k, 1, (k - 1) / 2, 1, true, &b->c2[d]);
            ada(name + ".adain1." + i, ch, &b->a1[d]);
            ada(name + ".adain2." + i, ch, &b->a2[d]);
            const KokoroTensor *a1 = find(name + ".alpha1." + i, {1, ch, 1}), *a2 = find(name + ".alpha2." + i, {1, ch, 1});
            if (a1 && a2) { b->alpha1[d] = vals(*a1); b->alpha2[d] = vals(*a2); }
        }
    };
    const int H = c.hidden, enc_out = 2 * H, asr_ch = 64, dec_in = enc_out + asr_ch + 2;
    conv("F0_conv", 1, 1, kCurveKernel, kCurveStride, kCurvePad, 1, true, &m.f0_conv);
    conv("N_conv", 1, 1, kCurveKernel, kCurveStride, kCurvePad, 1, true, &m.n_conv);
    conv("asr_res.0", asr_ch, H, 1, 1, 0, 1, true, &m.asr_res);
    res_blk("encode", H + 2, enc_out, false, &m.encode);
    m.decode.assign(kDecodeBlocks, CResBlk());
    for (int i = 0; i < kDecodeBlocks; i++) res_blk("decode." + std::to_string(i), dec_in, i + 1 == kDecodeBlocks ? c.initial_channel : enc_out, i + 1 == kDecodeBlocks, &m.decode[static_cast<size_t>(i)]);
    const size_t nup = c.upsample_rates.size(), nk = c.resblock_kernels.size();
    m.noise_convs.assign(nup, CConv()); m.ups.assign(nup, CConv()); m.noise_res.assign(nup, CAdaRes1()); m.resblocks.assign(nup * nk, CAdaRes1());
    for (size_t i = 0; i < nup && st.is_ok(); i++) {
        const int cin = c.initial_channel >> i, cout = c.initial_channel >> (i + 1), u = c.upsample_rates[i], k = c.upsample_kernels[i];
        convt("generator.ups." + std::to_string(i), cin, cout, k, u, (k - u) / 2, &m.ups[i]);
        if (i + 1 < nup) {
            int sf = 1;
            for (size_t j = i + 1; j < nup; j++) sf *= c.upsample_rates[j];
            conv("generator.noise_convs." + std::to_string(i), cout, m.out_ch, 2 * sf, sf, (sf + 1) / 2, 1, true, &m.noise_convs[i]);
            ada_res1("generator.noise_res." + std::to_string(i), cout, 7, &m.noise_res[i]);
        } else {
            conv("generator.noise_convs." + std::to_string(i), cout, m.out_ch, 1, 1, 0, 1, true, &m.noise_convs[i]);
            ada_res1("generator.noise_res." + std::to_string(i), cout, 11, &m.noise_res[i]);
        }
        for (size_t j = 0; j < nk; j++) ada_res1("generator.resblocks." + std::to_string(i * nk + j), cout, c.resblock_kernels[j], &m.resblocks[i * nk + j]);
    }
    conv("generator.conv_post", m.out_ch, c.initial_channel >> nup, 7, 1, 3, 1, true, &m.conv_post);
    if (!st.is_ok()) return st;
    m.gb.assign(static_cast<size_t>(m.style_rows), 0.f);
    return std::unique_ptr<CpuKokoroVocoder>(new CpuKokoroVocoder(std::move(impl)));
}

Status CpuKokoroVocoder::run(const float * asr, int32_t frames, const float * f0, const float * n, const float * style, const float * har, std::vector<float> * out) {
    Impl & m = *impl_;
    if (!asr || !f0 || !n || !style || !har || !out) return invalid_argument("kokoro vocoder: null input or output");
    if (frames < 1) return invalid_argument("kokoro vocoder: frames must be >= 1");
    const int F = frames, H = m.cfg.hidden, L2 = har_length(F);
    const size_t f = static_cast<size_t>(F);
    auto finite = [](const float * p, size_t n_) { for (size_t i = 0; i < n_; i++) if (!std::isfinite(p[i])) return false; return true; };
    if (!finite(asr, static_cast<size_t>(H) * f) || !finite(f0, 2 * f) || !finite(n, 2 * f) || !finite(style, static_cast<size_t>(m.cfg.style_dim)) ||
        !finite(har, static_cast<size_t>(m.out_ch) * static_cast<size_t>(L2))) {
        return invalid_argument("kokoro vocoder: the input contains NaN or Inf");
    }
    // buffers: the decoder at frame rate, the generator stages
    const int dec_x = m.encode.cout, asr_ch = m.asr_res.oc, dec_ch = m.decode.front().cin;
    size_t big = 1, len = 2 * f, max_k = 1;
    for (size_t i = 0; i < m.ups.size(); i++) {
        len = len * static_cast<size_t>(m.ups[i].stride) + (i + 1 == m.ups.size() ? 1 : 0);
        big = std::max(big, static_cast<size_t>(m.ups[i].oc) * len);
    }
    for (const CResBlk * b : {&m.encode}) max_k = std::max(max_k, static_cast<size_t>(b->cin) * 3);
    for (const CResBlk & b : m.decode) max_k = std::max(max_k, static_cast<size_t>(b.cin) * 3);
    for (const CAdaRes1 & b : m.resblocks) max_k = std::max(max_k, static_cast<size_t>(b.ch) * b.k);
    for (const CAdaRes1 & b : m.noise_res) max_k = std::max(max_k, static_cast<size_t>(b.ch) * b.k);
    for (const CConv & c : m.noise_convs) max_k = std::max(max_k, static_cast<size_t>(c.ic) * c.taps);
    for (const CConv & c : m.ups) max_k = std::max(max_k, static_cast<size_t>(c.ic) * c.taps);
    max_k = std::max(max_k, static_cast<size_t>(m.conv_post.ic) * m.conv_post.taps);
    size_t max_oc = static_cast<size_t>(std::max({dec_x, m.decode.back().cout, m.out_ch, m.ups.front().oc}));
    m.prep.assign(std::max({big, static_cast<size_t>(dec_ch) * 2 * f, static_cast<size_t>(m.out_ch) * L2}), 0.f);
    m.col.assign(max_k * kColumns, 0.f);
    m.tmp.assign(max_oc * kColumns, 0.f);
    std::vector<float> E(static_cast<size_t>(m.encode.cin) * f), D0(static_cast<size_t>(dec_ch) * f), D1(D0.size()), H1(static_cast<size_t>(dec_x) * f),
        POOL(static_cast<size_t>(dec_ch) * 2 * f), X0(static_cast<size_t>(m.decode.back().cout) * 2 * f), A(big), B(big), C(big), Dd(big), XS(big);
    std::memcpy(E.data(), asr, static_cast<size_t>(H) * f * 4);
    // style projections
    cblas_sgemv(CblasRowMajor, CblasNoTrans, m.style_rows, m.cfg.style_dim, 1.f, m.style_w.data(), m.cfg.style_dim, style, 1, 0.f, m.gb.data(), 1);
    for (int i = 0; i < m.style_rows; i++) m.gb[static_cast<size_t>(i)] += m.style_b[static_cast<size_t>(i)];
    // F0 / N into the encoder input and both decoder buffers; asr_res into both decoder buffers
    const float * curves[2] = {f0, n};
    for (int cv = 0; cv < 2; cv++) {
        const CConv & cc = cv == 0 ? m.f0_conv : m.n_conv;
        m.conv(cc, curves[cv], 2 * F, 1, E.data() + static_cast<size_t>(H + cv) * f, F, nullptr, 0, 1.f);
        for (std::vector<float> * d : {&D0, &D1}) m.conv(cc, curves[cv], 2 * F, 1, d->data() + static_cast<size_t>(dec_x + asr_ch + cv) * f, F, nullptr, 0, 1.f);
    }
    for (std::vector<float> * d : {&D0, &D1}) m.conv(m.asr_res, asr, F, 1, d->data() + static_cast<size_t>(dec_x) * f, F, nullptr, 0, 1.f);
    m.res_blk(m.encode, E.data(), F, D0.data(), H1, POOL);
    std::vector<float> * cur = &D0, * nxt = &D1;
    for (size_t i = 0; i < m.decode.size(); i++) {
        const bool last = i + 1 == m.decode.size();
        m.res_blk(m.decode[i], cur->data(), F, last ? X0.data() : nxt->data(), H1, POOL);
        std::swap(cur, nxt);
    }
    // generator
    const float * x = X0.data();
    int T = 2 * F;
    const size_t nup = m.ups.size(), nk = m.cfg.resblock_kernels.size();
    for (size_t i = 0; i < nup; i++) {
        const CConv & up = m.ups[i];
        const bool last = i + 1 == nup;
        const int To = T * up.stride + (last ? 1 : 0);
        m.conv(m.noise_convs[i], har, L2, 1, A.data(), To, nullptr, 0, 1.f);
        m.ada_res1(m.noise_res[i], A.data(), To, A.data(), Dd.data(), nullptr, 1.f, false);
        m.prep_act(x, up.ic, T, kLeaky, kGenSlope);
        m.conv(up, m.prep.data(), T, 1, B.data(), To, A.data(), kResidual, 1.f, last ? 1 : 0);
        if (last) for (int ch = 0; ch < up.oc; ch++) { const size_t r = static_cast<size_t>(ch) * To; B[r] = (B[r + 2] - A[r + 2]) + A[r]; }   // reflection pad
        for (size_t j = 0; j < nk; j++) m.ada_res1(m.resblocks[i * nk + j], B.data(), To, C.data(), Dd.data(), XS.data(), 1.f / static_cast<float>(nk), j > 0);
        x = XS.data();
        T = To;
    }
    m.prep_act(x, m.conv_post.ic, T, kLeaky, kPostSlope);
    out->assign(static_cast<size_t>(m.out_ch) * static_cast<size_t>(L2), 0.f);
    m.conv(m.conv_post, m.prep.data(), T, 1, out->data(), T, nullptr, 0, 1.f);
    return Status();
}

}  // namespace dooms
