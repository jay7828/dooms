#include "backend/metal/metal_kokoro_vocoder.h"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <utility>

#include "backend/metal/kokoro_kernels.h"
#include "core/dtype.h"

namespace dooms {

namespace {

// Must equal the kernel source's constants.
constexpr int kBM = 128, kBN = 64, kBK = 32;   // kBM: the taller tile (weights are padded to it); 64-row tiles run where oc <= 64
constexpr size_t kAlign = 256;
// Fixed by Kokoro's architecture (kokoro/istftnet.py, kokoro/modules.py), not by its config: the residual blocks' activations, F0/N downsampling, the
// decoder's block layout and the AdaIN epsilon (InstanceNorm1d's default).
constexpr float kBlockSlope = 0.2f, kGenSlope = 0.1f, kPostSlope = 0.01f, kNormEps = 1e-5f;
constexpr int kCurveKernel = 3, kCurveStride = 2, kCurvePad = 1;   // F0_conv / N_conv
constexpr int kDecodeBlocks = 4;                                   // decode.0 .. decode.3; the last one upsamples by 2
constexpr int kResDilations = 3;                                   // AdaINResBlock1: dilations (1, 3, 5) as written in the model code
constexpr int kDilation[kResDilations] = {1, 3, 5};
const float kInvSqrt2 = 1.f / std::sqrt(2.f);

enum Act : int { kNone = 0, kLeaky = 1, kSnake = 2 };   // activate() in the kernel source
enum Flag : int { kBias = 1, kResidual = 2, kAccumulate = 4 };

struct C1Params { int ic, oc, taps, tin, up, ncol, in_stride, dil, in_off, out_stride, out_off, tout, K, Kp; float post_scale; int flags; };
struct NormParams { int channels, len, gb_off; float eps; int act; float slope; };
struct ActParams { int channels, len, act; float slope; };
struct DwParams { int channels, tin, tout; };
static_assert(sizeof(C1Params) == 64 && sizeof(NormParams) == 24 && sizeof(ActParams) == 16 && sizeof(DwParams) == 12,
              "parameter structs must match the kernel source field for field");

std::string describe(NSError * e) { return e ? std::string([[e localizedDescription] UTF8String]) : std::string("unknown error"); }
size_t align_up(size_t v, size_t a) { return (v + a - 1) / a * a; }

}  // namespace

struct KConv {
    std::vector<size_t> w;     // packed half weights, one per output phase (1 for an ordinary convolution)
    size_t b = SIZE_MAX;       // f32 bias, SIZE_MAX: none
    int oc = 0, ic = 0, taps = 0, Kp = 0, stride = 1, pad = 0, dil = 1;
    bool transposed = false;
    bool tap_major = false;    // reduction index k = j * ic + ic_index (ic a multiple of the step depth): the kernel's fast path
};
struct KAda { size_t w = 0, b = 0; int channels = 0, gb_off = 0; };
struct KResBlk {      // AdainResBlk1d
    KAda n1, n2;
    KConv c1, c2, sc;
    bool up = false;
    size_t pool_w = 0, pool_b = 0;
    int cin = 0, cout = 0;
};
struct KAdaRes1 {     // AdaINResBlock1
    int ch = 0, k = 0;
    KAda a1[kResDilations], a2[kResDilations];
    KConv c1[kResDilations], c2[kResDilations];
    size_t alpha1[kResDilations] = {}, alpha2[kResDilations] = {};
};

struct MetalKokoroVocoder::Impl {
    KokoroVocoderConfig cfg;
    id<MTLDevice> device;
    id<MTLCommandQueue> queue;
    id<MTLComputePipelineState> p_conv, p_conv64, p_conv64_tap, p_prep, p_act, p_style, p_dw, p_reflect;
    id<MTLBuffer> weights, style_w, style_b;
    int style_rows = 0;
    KConv f0_conv, n_conv, asr_res, conv_post;
    KResBlk encode;
    std::vector<KResBlk> decode;
    std::vector<KConv> noise_convs, ups;
    std::vector<KAdaRes1> noise_res, resblocks;
    int out_ch = 0, gen_up = 1;
    // per-call buffers, grown as needed
    int cap_frames = 0;
    id<MTLBuffer> E, D0, D1, H1, POOL, X0, CURVES, HAR, A, B, C, D, XS, OUT, PREP, GB, STYLE;
    id<MTLCommandBuffer> cb_;
    id<MTLComputeCommandEncoder> enc_;
    bool profiling = false;
    bool tall_tiles = false;  // 128-row tiles for convolutions with more than 64 output channels: measured 7x slower (32 KB staging, 32 accumulators)
    std::vector<std::pair<std::string, double>> prof;

    void begin() {
        cb_ = profiling ? [queue commandBuffer] : [queue commandBufferWithUnretainedReferences];
        enc_ = [cb_ computeCommandEncoder];
    }
    id<MTLCommandBuffer> end() {
        [enc_ endEncoding];
        [cb_ commit];
        id<MTLCommandBuffer> cb = cb_;
        enc_ = nil;
        cb_ = nil;
        return cb;
    }
    void cut(const char * kind) {
        if (!profiling) return;
        id<MTLCommandBuffer> cb = end();
        [cb waitUntilCompleted];
        prof.emplace_back(kind, (cb.GPUEndTime - cb.GPUStartTime) * 1e3);
        begin();
    }

    // PREP = half(activate(x)) for ch rows of len (x_off bytes into xb).
    void prep_act(id<MTLBuffer> xb, size_t x_off, int ch, int len, int act, float slope) {
        ActParams p{ch, len, act, slope};
        [enc_ setComputePipelineState:p_act];
        [enc_ setBuffer:xb offset:x_off atIndex:0];
        [enc_ setBuffer:PREP offset:0 atIndex:1];
        [enc_ setBytes:&p length:sizeof p atIndex:2];
        [enc_ dispatchThreads:MTLSizeMake(static_cast<NSUInteger>(len), static_cast<NSUInteger>(ch), 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        cut("activation");
    }
    // PREP = half(activate(AdaIN(x))) (statistics of x [channels][len] and the style's gamma/beta).
    void prep_adain(const KAda & a, id<MTLBuffer> xb, int len, int act, float slope, size_t alpha) {
        NormParams p{a.channels, len, a.gb_off, kNormEps, act, slope};
        [enc_ setComputePipelineState:p_prep];
        [enc_ setBuffer:xb offset:0 atIndex:0];
        [enc_ setBuffer:weights offset:a.w atIndex:1];
        [enc_ setBuffer:weights offset:a.b atIndex:2];
        [enc_ setBuffer:GB offset:0 atIndex:3];
        [enc_ setBuffer:PREP offset:0 atIndex:4];
        [enc_ setBytes:&p length:sizeof p atIndex:5];
        [enc_ setBuffer:weights offset:alpha atIndex:6];
        [enc_ dispatchThreadgroups:MTLSizeMake(static_cast<NSUInteger>(a.channels), 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        cut("AdaIN + activation");
    }

    // y[oc][t] over tout positions (row stride tout) of a convolution of the half input hb[ic][tin] (row stride tin), upsampled by `up` first.
    void conv(const char * kind, const KConv & c, id<MTLBuffer> hb, int tin, int up, id<MTLBuffer> yb, size_t y_off, int tout,
              id<MTLBuffer> rb, size_t r_off, int flags, float post_scale, int out_shift = 0) {
        const int phases = c.transposed ? c.stride : 1;
        for (int ph = 0; ph < phases; ph++) {
            C1Params p{};
            p.ic = c.ic; p.oc = c.oc; p.taps = c.taps; p.tin = tin; p.up = up; p.K = c.ic * c.taps; p.Kp = c.Kp;
            p.post_scale = post_scale; p.flags = flags | (c.b != SIZE_MAX ? kBias : 0); p.tout = tout;
            if (c.transposed) {   // phase ph = (t + pad) mod stride: t = q * stride + ph - pad, input index q - j
                p.in_stride = 1; p.dil = -1; p.in_off = 0; p.out_stride = c.stride; p.out_off = ph - c.pad + out_shift;
                p.ncol = (tout - p.out_off + c.stride - 1) / c.stride;
            } else {
                p.in_stride = c.stride; p.dil = c.dil; p.in_off = -c.pad; p.out_stride = 1; p.out_off = out_shift; p.ncol = tout - out_shift;
            }
            const bool tall = c.oc > 64 && tall_tiles;
            [enc_ setComputePipelineState:tall ? p_conv : (c.tap_major ? p_conv64_tap : p_conv64)];
            [enc_ setBuffer:hb offset:0 atIndex:0];
            [enc_ setBuffer:weights offset:c.w[static_cast<size_t>(ph)] atIndex:1];
            [enc_ setBuffer:weights offset:(c.b != SIZE_MAX ? c.b : 0) atIndex:2];
            [enc_ setBuffer:(rb ? rb : yb) offset:(rb ? r_off : y_off) atIndex:3];
            [enc_ setBuffer:yb offset:y_off atIndex:4];
            [enc_ setBytes:&p length:sizeof p atIndex:5];
            const int tm = tall ? 128 : 64;
            [enc_ dispatchThreadgroups:MTLSizeMake((static_cast<NSUInteger>(std::max(p.ncol, 1)) + kBN - 1) / kBN, (c.oc + tm - 1) / tm, 1)
                 threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
            cut(kind);
        }
    }

    // AdainResBlk1d: x [cin][T] -> y [cout][T or 2T] (rows from 0 of yb, row stride = the output length).
    void res_blk(const KResBlk & b, id<MTLBuffer> xb, int T, id<MTLBuffer> yb) {
        const int To = b.up ? 2 * T : T;
        prep_adain(b.n1, xb, T, kLeaky, kBlockSlope, 0);
        if (b.up) {
            DwParams dp{b.cin, T, To};
            [enc_ setComputePipelineState:p_dw];
            [enc_ setBuffer:PREP offset:0 atIndex:0];
            [enc_ setBuffer:weights offset:b.pool_w atIndex:1];
            [enc_ setBuffer:weights offset:b.pool_b atIndex:2];
            [enc_ setBuffer:POOL offset:0 atIndex:4];
            [enc_ setBytes:&dp length:sizeof dp atIndex:5];
            [enc_ dispatchThreads:MTLSizeMake(static_cast<NSUInteger>(To), static_cast<NSUInteger>(b.cin), 1) threadsPerThreadgroup:MTLSizeMake(64, 4, 1)];
            cut("depthwise upsampling");
            conv("decoder conv", b.c1, POOL, To, 1, H1, 0, To, nil, 0, 0, 1.f);
        } else {
            conv("decoder conv", b.c1, PREP, T, 1, H1, 0, To, nil, 0, 0, 1.f);
        }
        prep_act(xb, 0, b.cin, T, kNone, 0.f);
        conv("decoder shortcut", b.sc, PREP, T, b.up ? 2 : 1, yb, 0, To, nil, 0, 0, 1.f);
        prep_adain(b.n2, H1, To, kLeaky, kBlockSlope, 0);
        conv("decoder conv", b.c2, PREP, To, 1, yb, 0, To, yb, 0, kResidual, kInvSqrt2);   // (residual + shortcut) / sqrt(2)
    }

    // AdaINResBlock1 on x [ch][T]: three (dilated conv, conv) pairs with residuals. The chain runs in `r` (r = x after the first pair); with `xs`, the last pair
    // writes (or, with accumulate, adds) scale * result there instead.
    void ada_res1(const char * kind, const KAdaRes1 & b, id<MTLBuffer> xb, int T, id<MTLBuffer> r, id<MTLBuffer> t, id<MTLBuffer> xs, float scale, bool accumulate) {
        id<MTLBuffer> cur = xb;
        for (int d = 0; d < kResDilations; d++) {
            prep_adain(b.a1[d], cur, T, kSnake, 0.f, b.alpha1[d]);
            conv(kind, b.c1[d], PREP, T, 1, t, 0, T, nil, 0, 0, 1.f);
            prep_adain(b.a2[d], t, T, kSnake, 0.f, b.alpha2[d]);
            const bool last = d + 1 == kResDilations;
            id<MTLBuffer> out = (last && xs) ? xs : r;
            conv(kind, b.c2[d], PREP, T, 1, out, 0, T, cur, 0, kResidual | ((last && xs && accumulate) ? kAccumulate : 0), (last && xs) ? scale : 1.f);
            cur = r;
        }
    }

    Status ensure(int F) {
        if (F <= cap_frames) return Status();
        const size_t f = static_cast<size_t>(F);
        const size_t L2 = f * static_cast<size_t>(gen_up) + 1;   // generator output length
        size_t big = 1, len = 2 * f;                               // the largest generator stage (channels x length)
        for (size_t i = 0; i < ups.size(); i++) {
            len = len * static_cast<size_t>(ups[i].stride) + (i + 1 == ups.size() ? 1 : 0);
            big = std::max(big, static_cast<size_t>(ups[i].oc) * len);
        }
        auto buf = [&](size_t floats, id<MTLBuffer> __strong & out) -> Status {
            out = [device newBufferWithLength:std::max<size_t>(floats, 4) * sizeof(float) options:MTLResourceStorageModePrivate];
            if (!out) return resource_exhausted("could not allocate " + std::to_string(floats * 4) + " bytes for the vocoder");
            return Status();
        };
        const size_t dec_ch = static_cast<size_t>(decode.front().cin);
        const size_t prep = std::max({big, dec_ch * 2 * f, static_cast<size_t>(out_ch) * L2});   // the largest convolution input (half)
        if (Status st = buf((prep + 1) / 2, PREP); !st.is_ok()) return st;
        for (auto & [floats, slot] : std::initializer_list<std::pair<size_t, id<MTLBuffer> __strong *>>{
                 {dec_ch * f, &D0}, {dec_ch * f, &D1}, {static_cast<size_t>(encode.cout) * f, &H1}, {(dec_ch * 2 * f + 1) / 2, &POOL},
                 {static_cast<size_t>(decode.back().cout) * 2 * f, &X0}, {big, &A}, {big, &B}, {big, &C}, {big, &D}, {big, &XS}}) {
            if (Status st = buf(floats, *slot); !st.is_ok()) return st;
        }
        // host-written inputs need shared storage
        CURVES = [device newBufferWithLength:4 * f * sizeof(float) options:MTLResourceStorageModeShared];
        HAR = [device newBufferWithLength:static_cast<size_t>(out_ch) * L2 * sizeof(float) options:MTLResourceStorageModeShared];
        OUT = [device newBufferWithLength:static_cast<size_t>(out_ch) * L2 * sizeof(float) options:MTLResourceStorageModeShared];
        id<MTLBuffer> e = [device newBufferWithLength:static_cast<size_t>(encode.cin) * f * sizeof(float) options:MTLResourceStorageModeShared];
        if (!CURVES || !HAR || !OUT || !e) return resource_exhausted("could not allocate the vocoder's input buffers");
        E = e;
        cap_frames = F;
        return Status();
    }
};

MetalKokoroVocoder::MetalKokoroVocoder(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}
MetalKokoroVocoder::~MetalKokoroVocoder() = default;
void MetalKokoroVocoder::set_profiling(bool on) { impl_->profiling = on; }
const std::vector<std::pair<std::string, double>> & MetalKokoroVocoder::profile() const { return impl_->prof; }
int32_t MetalKokoroVocoder::har_length(int32_t frames) const { return frames * impl_->gen_up + 1; }

Result<std::unique_ptr<MetalKokoroVocoder>> MetalKokoroVocoder::create(const KokoroVocoderConfig & c, const KokoroTensors & t) {
    if (c.hidden < 1 || c.style_dim < 1 || c.n_fft < 2 || c.initial_channel < 2) return invalid_argument("kokoro vocoder config: sizes must be positive");
    if (c.upsample_rates.empty() || c.upsample_rates.size() != c.upsample_kernels.size() || c.resblock_kernels.empty()) {
        return invalid_argument("kokoro vocoder config: one kernel size per upsampling rate, at least one residual block kernel");
    }
    auto impl = std::make_unique<Impl>();
    Impl & m = *impl;
    m.cfg = c;
    m.out_ch = c.n_fft + 2;
    m.gen_up = 2;   // the decoder's last block doubles the frame rate
    for (int32_t r : c.upsample_rates) m.gen_up *= r;

    // ---------------------------------------------------------------- tensor lookup and packing
    auto find = [&](const std::string & name, const std::vector<int64_t> & shape, const KokoroTensor ** out) -> Status {
        auto it = t.find("dec." + name);
        if (it == t.end() || !it->second.data) return invalid_argument("kokoro vocoder: tensor 'dec." + name + "' is missing");
        if (it->second.shape != shape) {
            std::string got, want;
            for (int64_t s : it->second.shape) got += std::to_string(s) + " ";
            for (int64_t s : shape) want += std::to_string(s) + " ";
            return invalid_argument("kokoro vocoder: tensor 'dec." + name + "' has shape [ " + got + "], expected [ " + want + "]");
        }
        if (it->second.dtype != DType::F16 && it->second.dtype != DType::F32) return invalid_argument("kokoro vocoder: tensor 'dec." + name + "' must be f16 or f32");
        *out = &it->second;
        return Status();
    };
    auto value = [](const KokoroTensor & k, size_t i) -> float {
        return k.dtype == DType::F16 ? static_cast<float>(static_cast<const __fp16 *>(k.data)[i]) : static_cast<const float *>(k.data)[i];
    };
    // Two passes over the layer list: the first sizes the weight buffer, the second fills it (same code, `base` null in the first).
    std::vector<float> style_w, style_b;
    const bool tap_major_ok = c.tap_major;
    int gb_count = 0;   // rows of the style projections so far (both passes)
    size_t total = 0;
    uint8_t * base = nullptr;
    auto reserve = [&](size_t bytes) { const size_t off = total; total = align_up(total + bytes, kAlign); return off; };
    auto put_f32 = [&](const KokoroTensor & k, size_t n) {
        const size_t off = reserve(n * 4);
        if (base) for (size_t i = 0; i < n; i++) reinterpret_cast<float *>(base + off)[i] = value(k, i);
        return off;
    };
    Status st;
    auto conv = [&](const std::string & name, int oc, int ic, int k, int stride, int pad, int dil, bool has_bias, KConv * out) {
        if (!st.is_ok()) return;
        const KokoroTensor * w = nullptr;
        if (Status s = find(name + ".weight", {oc, ic, k}, &w); !s.is_ok()) { st = s; return; }
        out->oc = oc; out->ic = ic; out->taps = k; out->stride = stride; out->pad = pad; out->dil = dil; out->transposed = false;
        out->tap_major = tap_major_ok && ic % kBK == 0;
        out->Kp = static_cast<int>(align_up(static_cast<size_t>(ic) * k, kBK));
        const size_t rows = align_up(static_cast<size_t>(oc), kBM);
        const size_t off = reserve(rows * static_cast<size_t>(out->Kp) * 2);
        if (base) {
            __fp16 * dst = reinterpret_cast<__fp16 *>(base + off);
            std::memset(dst, 0, rows * static_cast<size_t>(out->Kp) * 2);
            for (int o = 0; o < oc; o++) for (int i = 0; i < ic; i++) for (int j = 0; j < k; j++) {
                const size_t col = out->tap_major ? static_cast<size_t>(j) * ic + i : static_cast<size_t>(i) * k + j;
                dst[static_cast<size_t>(o) * out->Kp + col] = static_cast<__fp16>(value(*w, (static_cast<size_t>(o) * ic + i) * k + j));
            }
        }
        out->w = {off};
        out->b = SIZE_MAX;
        if (has_bias) {
            const KokoroTensor * b = nullptr;
            if (Status s = find(name + ".bias", {oc}, &b); !s.is_ok()) { st = s; return; }
            out->b = put_f32(*b, static_cast<size_t>(oc));
        }
    };
    // ConvTranspose1d weight [ic][oc][k] -> one packed matrix per output phase ph: A_ph[o][i * taps + j] = W[i][o][ph + j * stride].
    auto convt = [&](const std::string & name, int ic, int oc, int k, int stride, int pad, KConv * out) {
        if (!st.is_ok()) return;
        if (k % stride != 0) { st = unsupported("kokoro vocoder: transposed convolution '" + name + "' has a kernel that is not a multiple of its stride"); return; }
        const KokoroTensor * w = nullptr;
        if (Status s = find(name + ".weight", {ic, oc, k}, &w); !s.is_ok()) { st = s; return; }
        out->oc = oc; out->ic = ic; out->taps = k / stride; out->stride = stride; out->pad = pad; out->dil = -1; out->transposed = true;
        out->tap_major = tap_major_ok && ic % kBK == 0;
        out->Kp = static_cast<int>(align_up(static_cast<size_t>(ic) * out->taps, kBK));
        const size_t rows = align_up(static_cast<size_t>(oc), kBM);
        out->w.clear();
        for (int ph = 0; ph < stride; ph++) {
            const size_t off = reserve(rows * static_cast<size_t>(out->Kp) * 2);
            if (base) {
                __fp16 * dst = reinterpret_cast<__fp16 *>(base + off);
                std::memset(dst, 0, rows * static_cast<size_t>(out->Kp) * 2);
                for (int o = 0; o < oc; o++) for (int i = 0; i < ic; i++) for (int j = 0; j < out->taps; j++) {
                    const size_t col = out->tap_major ? static_cast<size_t>(j) * ic + i : static_cast<size_t>(i) * out->taps + j;
                    dst[static_cast<size_t>(o) * out->Kp + col] =
                        static_cast<__fp16>(value(*w, (static_cast<size_t>(i) * oc + o) * k + static_cast<size_t>(ph + j * stride)));
                }
            }
            out->w.push_back(off);
        }
        const KokoroTensor * b = nullptr;
        if (Status s = find(name + ".bias", {oc}, &b); !s.is_ok()) { st = s; return; }
        out->b = put_f32(*b, static_cast<size_t>(oc));
    };
    auto ada = [&](const std::string & name, int ch, KAda * out) {
        if (!st.is_ok()) return;
        const KokoroTensor *nw = nullptr, *nb = nullptr, *fw = nullptr, *fb = nullptr;
        for (auto [n, shape, slot] : std::initializer_list<std::tuple<const char *, std::vector<int64_t>, const KokoroTensor **>>{
                 {".norm.weight", {ch}, &nw}, {".norm.bias", {ch}, &nb}, {".fc.weight", {2 * ch, c.style_dim}, &fw}, {".fc.bias", {2 * ch}, &fb}}) {
            if (Status s = find(name + n, shape, slot); !s.is_ok()) { st = s; return; }
        }
        out->channels = ch;
        out->w = put_f32(*nw, static_cast<size_t>(ch));
        out->b = put_f32(*nb, static_cast<size_t>(ch));
        out->gb_off = gb_count;
        gb_count += 2 * ch;
        if (!base) {   // the style projections are collected once (first pass)
            for (int64_t i = 0; i < 2 * ch * c.style_dim; i++) style_w.push_back(value(*fw, static_cast<size_t>(i)));
            for (int64_t i = 0; i < 2 * ch; i++) style_b.push_back(value(*fb, static_cast<size_t>(i)));
        }
    };
    auto res_blk = [&](const std::string & name, int cin, int cout, bool up, KResBlk * b) {
        b->cin = cin; b->cout = cout; b->up = up;
        ada(name + ".norm1", cin, &b->n1);
        conv(name + ".conv1", cout, cin, 3, 1, 1, 1, true, &b->c1);
        ada(name + ".norm2", cout, &b->n2);
        conv(name + ".conv2", cout, cout, 3, 1, 1, 1, true, &b->c2);
        conv(name + ".conv1x1", cout, cin, 1, 1, 0, 1, false, &b->sc);
        if (up && st.is_ok()) {
            const KokoroTensor *pw = nullptr, *pb = nullptr;
            if (Status s = find(name + ".pool.weight", {cin, 1, 3}, &pw); !s.is_ok()) { st = s; return; }
            if (Status s = find(name + ".pool.bias", {cin}, &pb); !s.is_ok()) { st = s; return; }
            b->pool_w = put_f32(*pw, static_cast<size_t>(cin) * 3);
            b->pool_b = put_f32(*pb, static_cast<size_t>(cin));
        }
    };
    auto ada_res1 = [&](const std::string & name, int ch, int k, KAdaRes1 * b) {
        b->ch = ch; b->k = k;
        for (int d = 0; d < kResDilations && st.is_ok(); d++) {
            const std::string i = std::to_string(d);
            const int dil = kDilation[d];
            conv(name + ".convs1." + i, ch, ch, k, 1, (k * dil - dil) / 2, dil, true, &b->c1[d]);
            conv(name + ".convs2." + i, ch, ch, k, 1, (k - 1) / 2, 1, true, &b->c2[d]);
            ada(name + ".adain1." + i, ch, &b->a1[d]);
            ada(name + ".adain2." + i, ch, &b->a2[d]);
            const KokoroTensor *a1 = nullptr, *a2 = nullptr;
            if (st.is_ok()) st = find(name + ".alpha1." + i, {1, ch, 1}, &a1);
            if (st.is_ok()) st = find(name + ".alpha2." + i, {1, ch, 1}, &a2);
            if (st.is_ok()) { b->alpha1[d] = put_f32(*a1, static_cast<size_t>(ch)); b->alpha2[d] = put_f32(*a2, static_cast<size_t>(ch)); }
        }
    };
    auto layers = [&]() {
        st = Status();
        gb_count = 0;
        const int H = c.hidden, enc_out = 2 * H;                     // the decoder's width: 1024
        const int asr_ch = 64;                                         // asr_res output (fixed in kokoro's Decoder)
        const int dec_in = enc_out + asr_ch + 2;                      // 1090 = x + asr_res + F0 + N
        conv("F0_conv", 1, 1, kCurveKernel, kCurveStride, kCurvePad, 1, true, &m.f0_conv);
        conv("N_conv", 1, 1, kCurveKernel, kCurveStride, kCurvePad, 1, true, &m.n_conv);
        conv("asr_res.0", asr_ch, H, 1, 1, 0, 1, true, &m.asr_res);
        res_blk("encode", H + 2, enc_out, false, &m.encode);
        m.decode.assign(kDecodeBlocks, KResBlk());
        for (int i = 0; i < kDecodeBlocks; i++) {
            const bool last = i + 1 == kDecodeBlocks;
            res_blk("decode." + std::to_string(i), dec_in, last ? c.initial_channel : enc_out, last, &m.decode[static_cast<size_t>(i)]);
        }
        const size_t nup = c.upsample_rates.size();
        m.noise_convs.assign(nup, KConv());
        m.ups.assign(nup, KConv());
        m.noise_res.assign(nup, KAdaRes1());
        m.resblocks.assign(nup * c.resblock_kernels.size(), KAdaRes1());
        for (size_t i = 0; i < nup && st.is_ok(); i++) {
            const int cin = c.initial_channel >> i, cout = c.initial_channel >> (i + 1);
            const int u = c.upsample_rates[i], k = c.upsample_kernels[i];
            convt("generator.ups." + std::to_string(i), cin, cout, k, u, (k - u) / 2, &m.ups[i]);
            if (i + 1 < nup) {   // noise branch at this stage's rate: stride = the remaining upsampling (kokoro's Generator)
                int sf = 1;
                for (size_t j = i + 1; j < nup; j++) sf *= c.upsample_rates[j];
                conv("generator.noise_convs." + std::to_string(i), cout, m.out_ch, 2 * sf, sf, (sf + 1) / 2, 1, true, &m.noise_convs[i]);
                ada_res1("generator.noise_res." + std::to_string(i), cout, 7, &m.noise_res[i]);
            } else {
                conv("generator.noise_convs." + std::to_string(i), cout, m.out_ch, 1, 1, 0, 1, true, &m.noise_convs[i]);
                ada_res1("generator.noise_res." + std::to_string(i), cout, 11, &m.noise_res[i]);
            }
            for (size_t j = 0; j < c.resblock_kernels.size(); j++) {
                ada_res1("generator.resblocks." + std::to_string(i * c.resblock_kernels.size() + j), cout, c.resblock_kernels[j], &m.resblocks[i * c.resblock_kernels.size() + j]);
            }
        }
        conv("generator.conv_post", m.out_ch, c.initial_channel >> nup, 7, 1, 3, 1, true, &m.conv_post);
        return st;
    };
    if (Status s = layers(); !s.is_ok()) return s;

    @autoreleasepool {
        m.device = MTLCreateSystemDefaultDevice();
        if (!m.device) return unsupported("no Metal device");
        m.queue = [m.device newCommandQueue];
        m.weights = [m.device newBufferWithLength:std::max<size_t>(total, 16) options:MTLResourceStorageModeShared];
        if (!m.queue || !m.weights) return resource_exhausted("could not allocate " + std::to_string(total) + " bytes for the vocoder's weights");
        base = static_cast<uint8_t *>([m.weights contents]);
        total = 0;
        const std::vector<float> sw = style_w, sb = style_b;
        if (Status s = layers(); !s.is_ok()) return s;   // second pass: fill
        m.style_rows = static_cast<int>(sb.size());
        m.style_w = [m.device newBufferWithBytes:sw.data() length:sw.size() * 4 options:MTLResourceStorageModeShared];
        m.style_b = [m.device newBufferWithBytes:sb.data() length:sb.size() * 4 options:MTLResourceStorageModeShared];
        m.GB = [m.device newBufferWithLength:sb.size() * 4 options:MTLResourceStorageModePrivate];
        m.STYLE = [m.device newBufferWithLength:static_cast<size_t>(c.style_dim) * 4 options:MTLResourceStorageModeShared];
        if (!m.style_w || !m.style_b || !m.GB || !m.STYLE) return resource_exhausted("could not allocate the vocoder's style buffers");
        NSError * error = nil;
        id<MTLLibrary> lib = [m.device newLibraryWithSource:[NSString stringWithUTF8String:kokoro_kernel_source()] options:[MTLCompileOptions new] error:&error];
        if (!lib) return internal_error("could not compile the vocoder kernels: " + describe(error));
        for (auto & [name, slot] : std::initializer_list<std::pair<const char *, id<MTLComputePipelineState> __strong *>>{
                 {"conv1d_128", &m.p_conv}, {"conv1d_64", &m.p_conv64}, {"conv1d_64_tap", &m.p_conv64_tap}, {"adain_prep", &m.p_prep}, {"act_prep", &m.p_act}, {"style_proj", &m.p_style}, {"dwconvt2", &m.p_dw},
                 {"reflect_first", &m.p_reflect}}) {
            id<MTLFunction> fn = [lib newFunctionWithName:[NSString stringWithUTF8String:name]];
            NSError * e = nil;
            *slot = fn ? [m.device newComputePipelineStateWithFunction:fn error:&e] : nil;
            if (!*slot) return internal_error(std::string("could not build kernel '") + name + "': " + describe(e));
        }
        if (m.p_conv.maxTotalThreadsPerThreadgroup < 128 || m.p_prep.maxTotalThreadsPerThreadgroup < 256) return unsupported("the vocoder kernels cannot run their threadgroup sizes");
    }
    return std::unique_ptr<MetalKokoroVocoder>(new MetalKokoroVocoder(std::move(impl)));
}

Status MetalKokoroVocoder::run(const float * asr, int32_t frames, const float * f0, const float * n, const float * style, const float * har, std::vector<float> * out) {
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
    if (Status st = m.ensure(F); !st.is_ok()) return st;
    // inputs: asr into rows 0..H-1 of E (rows H, H + 1 get F0 / N after their convolutions), the curves, the harmonic spectrum, the style
    std::memcpy([m.E contents], asr, static_cast<size_t>(H) * f * 4);
    float * curves = static_cast<float *>([m.CURVES contents]);
    std::memcpy(curves, f0, 2 * f * 4);
    std::memcpy(curves + 2 * f, n, 2 * f * 4);
    std::memcpy([m.HAR contents], har, static_cast<size_t>(m.out_ch) * static_cast<size_t>(L2) * 4);
    std::memcpy([m.STYLE contents], style, static_cast<size_t>(m.cfg.style_dim) * 4);
    m.prof.clear();
    const size_t row = f * 4;
    @autoreleasepool {
        m.begin();
        const int sp[2] = {m.style_rows, m.cfg.style_dim};
        [m.enc_ setComputePipelineState:m.p_style];
        [m.enc_ setBuffer:m.style_w offset:0 atIndex:0];
        [m.enc_ setBuffer:m.style_b offset:0 atIndex:1];
        [m.enc_ setBuffer:m.STYLE offset:0 atIndex:2];
        [m.enc_ setBuffer:m.GB offset:0 atIndex:3];
        [m.enc_ setBytes:sp length:sizeof sp atIndex:4];
        [m.enc_ dispatchThreadgroups:MTLSizeMake(static_cast<NSUInteger>(m.style_rows), 1, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
        m.cut("style projections");
        // F0 / N at frame rate into the encoder input and both decoder buffers; asr_res into both decoder buffers
        const int dec_x = m.encode.cout, asr_ch = m.asr_res.oc;
        for (int curve = 0; curve < 2; curve++) {   // F0, then N
            m.prep_act(m.CURVES, static_cast<size_t>(curve) * 2 * f * 4, 1, 2 * F, kNone, 0.f);
            for (auto [buf, r0] : std::initializer_list<std::pair<id<MTLBuffer>, int>>{{m.E, H}, {m.D0, dec_x + asr_ch}, {m.D1, dec_x + asr_ch}}) {
                m.conv("F0/N", curve == 0 ? m.f0_conv : m.n_conv, m.PREP, 2 * F, 1, buf, static_cast<size_t>(r0 + curve) * row, F, nil, 0, 0, 1.f);
            }
        }
        m.prep_act(m.E, 0, H, F, kNone, 0.f);
        for (id<MTLBuffer> buf : {m.D0, m.D1}) m.conv("asr_res", m.asr_res, m.PREP, F, 1, buf, static_cast<size_t>(dec_x) * row, F, nil, 0, 0, 1.f);
        m.res_blk(m.encode, m.E, F, m.D0);
        id<MTLBuffer> cur = m.D0, nxt = m.D1;
        for (size_t i = 0; i < m.decode.size(); i++) {
            const bool last = i + 1 == m.decode.size();
            m.res_blk(m.decode[i], cur, F, last ? m.X0 : nxt);
            std::swap(cur, nxt);
        }
        // generator
        id<MTLBuffer> x = m.X0;
        int T = 2 * F;
        const size_t nup = m.ups.size(), nk = m.cfg.resblock_kernels.size();
        for (size_t i = 0; i < nup; i++) {
            const KConv & up = m.ups[i];
            const bool last = i + 1 == nup;
            const int To = T * up.stride + (last ? 1 : 0);   // + the reflection pad's column
            id<MTLBuffer> src = m.A, g = m.B, r = m.C, t = m.D, xs = m.XS;
            // the noise branch at this rate (its own residual block, in place)
            m.prep_act(m.HAR, 0, m.out_ch, L2, kNone, 0.f);
            m.conv("noise conv", m.noise_convs[i], m.PREP, L2, 1, src, 0, To, nil, 0, 0, 1.f);
            m.ada_res1("noise resblock", m.noise_res[i], src, To, src, t, nil, 1.f, false);
            // upsample (leaky first) and add the noise branch; the last stage is shifted by the reflection pad's column
            m.prep_act(x, 0, up.ic, T, kLeaky, kGenSlope);
            m.conv("upsample", up, m.PREP, T, 1, g, 0, To, src, 0, kResidual, 1.f, last ? 1 : 0);
            if (last) {
                const int rp[2] = {up.oc, To};
                [m.enc_ setComputePipelineState:m.p_reflect];
                [m.enc_ setBuffer:g offset:0 atIndex:0];
                [m.enc_ setBuffer:src offset:0 atIndex:1];
                [m.enc_ setBytes:rp length:sizeof rp atIndex:2];
                [m.enc_ dispatchThreads:MTLSizeMake(static_cast<NSUInteger>(up.oc), 1, 1) threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
                m.cut("reflection pad");
            }
            // the average of the residual blocks
            for (size_t j = 0; j < nk; j++) m.ada_res1("resblock", m.resblocks[i * nk + j], g, To, r, t, xs, 1.f / static_cast<float>(nk), j > 0);
            x = xs;   // the next stage reads it only in its upsampling, before its residual blocks overwrite XS
            T = To;
        }
        m.prep_act(x, 0, m.conv_post.ic, T, kLeaky, kPostSlope);
        m.conv("conv_post", m.conv_post, m.PREP, T, 1, m.OUT, 0, T, nil, 0, 0, 1.f);
        id<MTLCommandBuffer> cb = m.end();
        if (m.cfg.spin_wait && !m.profiling) {
            while (cb.status != MTLCommandBufferStatusCompleted && cb.status != MTLCommandBufferStatusError) __builtin_arm_yield();
        } else {
            [cb waitUntilCompleted];
        }
        if (cb.status == MTLCommandBufferStatusError) return internal_error("a Metal command buffer failed: " + describe(cb.error));
    }
    const float * o = static_cast<const float *>([m.OUT contents]);
    out->assign(o, o + static_cast<size_t>(m.out_ch) * static_cast<size_t>(L2));
    return Status();
}

}  // namespace dooms
