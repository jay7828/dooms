#include "pipeline/kokoro_pipeline.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <map>
#include <utility>

#include "backend/coreml/coreml_model.h"
#include "backend/kokoro_vocoder.h"
#include "core/parallel.h"
#include "core/tensor.h"
#include "format/gguf.h"
#include "pipeline/kokoro_vocoder_loader.h"

namespace dooms {

namespace {

using Clock = std::chrono::steady_clock;
double ms_since(Clock::time_point t) { return std::chrono::duration<double, std::milli>(Clock::now() - t).count(); }

constexpr double kPi = 3.14159265358979323846;

// torch F.interpolate(mode='linear', align_corners=False) with a given scale factor along the time axis of x [n][cols]: source (i + 0.5) * scale - 0.5 in float32,
// clamped at 0 (as torch's CPU kernel).
void interp_linear(const float * x, int64_t n, int cols, int64_t out_len, float scale, float * y) {
    for (int64_t i = 0; i < out_len; i++) {
        float src = (static_cast<float>(i) + 0.5f) * scale - 0.5f;
        if (src < 0.f) src = 0.f;
        int64_t i0 = static_cast<int64_t>(src);
        if (i0 > n - 1) i0 = n - 1;
        const int64_t i1 = std::min(i0 + 1, n - 1);
        const float lam = src - static_cast<float>(i0);
        for (int h = 0; h < cols; h++) y[i * cols + h] = (1.f - lam) * x[i0 * cols + h] + lam * x[i1 * cols + h];
    }
}

double hann(int i, int n) { return 0.5 - 0.5 * std::cos(2.0 * kPi * i / n); }   // periodic

// UTF-8 -> code points (invalid bytes become U+FFFD, which no vocabulary holds).
std::vector<std::string> code_points(const std::string & s) {
    std::vector<std::string> out;
    for (size_t i = 0; i < s.size();) {
        const unsigned char c = static_cast<unsigned char>(s[i]);
        size_t n = c < 0x80 ? 1 : (c >> 5) == 6 ? 2 : (c >> 4) == 14 ? 3 : (c >> 3) == 30 ? 4 : 0;
        if (n == 0 || i + n > s.size()) { out.emplace_back("\xEF\xBF\xBD"); i++; continue; }
        out.push_back(s.substr(i, n));
        i += n;
    }
    return out;
}

struct Xoshiro {   // xoshiro256** (Blackman & Vigna), seeded with splitmix64
    uint64_t s[4];
    explicit Xoshiro(uint64_t seed) {
        for (uint64_t & v : s) { seed += 0x9e3779b97f4a7c15ULL; uint64_t z = seed; z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL; z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL; v = z ^ (z >> 31); }
    }
    static uint64_t rotl(uint64_t x, int k) { return (x << k) | (x >> (64 - k)); }
    uint64_t next() {
        const uint64_t r = rotl(s[1] * 5, 7) * 9, t = s[1] << 17;
        s[2] ^= s[0]; s[3] ^= s[1]; s[1] ^= s[2]; s[0] ^= s[3]; s[2] ^= t; s[3] = rotl(s[3], 45);
        return r;
    }
    double uniform() { return static_cast<double>(next() >> 11) * 0x1.0p-53; }   // [0, 1)
};

}  // namespace

void kokoro_align(const std::vector<float> & durations, const std::vector<float> & d, int32_t d_width, const std::vector<float> & t_en, int32_t t_width, float speed,
                  std::vector<int32_t> * frames, std::vector<float> * en, std::vector<float> * asr) {
    const size_t T = durations.size();
    frames->assign(T, 1);
    int64_t F = 0;
    for (size_t i = 0; i < T; i++) {
        const float v = std::nearbyint(durations[i] / speed);   // round half to even, as torch.round (the default rounding mode)
        (*frames)[i] = std::max(1, static_cast<int32_t>(v));
        F += (*frames)[i];
    }
    en->assign(static_cast<size_t>(d_width) * static_cast<size_t>(F), 0.f);
    asr->assign(static_cast<size_t>(t_width) * static_cast<size_t>(F), 0.f);
    int64_t f = 0;
    for (size_t i = 0; i < T; i++) {
        for (int32_t r = 0; r < (*frames)[i]; r++, f++) {
            for (int32_t c = 0; c < d_width; c++) (*en)[static_cast<size_t>(c) * F + f] = d[i * static_cast<size_t>(d_width) + c];
            for (int32_t c = 0; c < t_width; c++) (*asr)[static_cast<size_t>(c) * F + f] = t_en[static_cast<size_t>(c) * T + i];
        }
    }
}

void kokoro_sine_source(const std::vector<float> & f0, const KokoroNoise & nz, const KokoroSourceConfig & c, const float * w, float b, std::vector<float> * out) {
    const int64_t up = c.upsample, H = c.harmonics, L = static_cast<int64_t>(f0.size()) * up, D = static_cast<int64_t>(f0.size());
    std::vector<float> rad(static_cast<size_t>(L * H));
    // The per-sample loops run on several threads: each sample's values depend on its own inputs only, so the results equal one thread's.
    parallel_for(L, H * 8, [&](int64_t t0, int64_t t1) {
        for (int64_t t = t0; t < t1; t++) {
            const float f = f0[static_cast<size_t>(t / up)];   // nearest upsampling
            for (int64_t h = 0; h < H; h++) {
                const float v = f * static_cast<float>(h + 1) / static_cast<float>(c.sample_rate);
                float r = std::fmod(v, 1.f);
                if (r != 0.f && r < 0.f) r += 1.f;   // numpy / torch: the remainder takes the divisor's sign
                rad[static_cast<size_t>(t * H + h)] = r;
            }
        }
    });
    for (int64_t h = 1; h < H; h++) rad[static_cast<size_t>(h)] += nz.rand_ini[static_cast<size_t>(h)];   // the fundamental starts at phase 0
    std::vector<float> down(static_cast<size_t>(D * H)), phase(static_cast<size_t>(D * H)), up_phase(static_cast<size_t>(L * H));
    interp_linear(rad.data(), L, static_cast<int>(H), D, static_cast<float>(up), down.data());
    for (int64_t h = 0; h < H; h++) {   // cumsum accumulated in double (torch's CPU kernel), then * 2 * pi in float32
        double acc = 0;
        for (int64_t i = 0; i < D; i++) {
            acc += down[static_cast<size_t>(i * H + h)];
            phase[static_cast<size_t>(i * H + h)] = static_cast<float>(acc) * 2.f * static_cast<float>(kPi) * static_cast<float>(up);
        }
    }
    interp_linear(phase.data(), D, static_cast<int>(H), L, 1.f / static_cast<float>(up), up_phase.data());
    out->assign(static_cast<size_t>(L), 0.f);
    const float unvoiced_amp = c.sine_amp / 3.f;
    parallel_for(L, H * 16, [&](int64_t t0, int64_t t1) {
        for (int64_t t = t0; t < t1; t++) {
            const float uv = f0[static_cast<size_t>(t / up)] > c.voiced_threshold ? 1.f : 0.f;
            const float amp = uv * c.noise_std + (1.f - uv) * unvoiced_amp;
            float acc = 0.f;
            for (int64_t h = 0; h < H; h++) {
                const size_t i = static_cast<size_t>(t * H + h);
                const float s = std::sin(up_phase[i]) * c.sine_amp * uv + amp * nz.noise[i];
                acc += s * w[h];
            }
            (*out)[static_cast<size_t>(t)] = std::tanh(acc + b);
        }
    });
}

void kokoro_stft(const std::vector<float> & x, int32_t n_fft, int32_t hop, std::vector<float> * out, int32_t * frames) {
    const int pad = n_fft / 2, bins = n_fft / 2 + 1;
    const int64_t n = static_cast<int64_t>(x.size()), len = n + 2 * pad;
    auto at = [&](int64_t i) {   // reflect padding (without repeating the edge)
        int64_t j = i - pad;
        if (j < 0) j = -j;
        if (j >= n) j = 2 * (n - 1) - j;
        return static_cast<double>(x[static_cast<size_t>(j)]);
    };
    const int32_t fr = static_cast<int32_t>(1 + (len - n_fft) / hop);
    *frames = fr;
    out->assign(static_cast<size_t>(2 * bins) * static_cast<size_t>(fr), 0.f);
    std::vector<double> win(static_cast<size_t>(n_fft)), cs(static_cast<size_t>(bins * n_fft)), sn(static_cast<size_t>(bins * n_fft));
    for (int i = 0; i < n_fft; i++) win[static_cast<size_t>(i)] = hann(i, n_fft);
    for (int k = 0; k < bins; k++) for (int i = 0; i < n_fft; i++) {
        const double a = 2.0 * kPi * k * i / n_fft;
        cs[static_cast<size_t>(k * n_fft + i)] = std::cos(a);
        sn[static_cast<size_t>(k * n_fft + i)] = -std::sin(a);
    }
    parallel_for(fr, static_cast<int64_t>(bins) * n_fft * 4, [&](int64_t f0, int64_t f1) {   // frames are independent: the same values as one thread
    std::vector<double> seg(static_cast<size_t>(n_fft));
    for (int32_t f = static_cast<int32_t>(f0); f < static_cast<int32_t>(f1); f++) {
        for (int i = 0; i < n_fft; i++) seg[static_cast<size_t>(i)] = at(static_cast<int64_t>(f) * hop + i) * win[static_cast<size_t>(i)];
        for (int k = 0; k < bins; k++) {
            double re = 0, im = 0;
            for (int i = 0; i < n_fft; i++) { re += seg[static_cast<size_t>(i)] * cs[static_cast<size_t>(k * n_fft + i)]; im += seg[static_cast<size_t>(i)] * sn[static_cast<size_t>(k * n_fft + i)]; }
            // a real FFT's DC and Nyquist bins have an imaginary part of exactly +0 (torch / pocketfft): the phase of a negative one is then +pi. A summed
            // -0 or 1e-16 residue would give -pi, and the vocoder's noise branch reads these phases.
            if (k == 0 || 2 * k == n_fft) im = 0.0;
            (*out)[static_cast<size_t>(k) * fr + f] = static_cast<float>(std::sqrt(re * re + im * im));
            (*out)[static_cast<size_t>(bins + k) * fr + f] = static_cast<float>(std::atan2(im, re));
        }
    }
    });
}

void kokoro_istft(const std::vector<float> & x, int32_t frames, int32_t n_fft, int32_t hop, std::vector<float> * out) {
    const int bins = n_fft / 2 + 1, pad = n_fft / 2;
    const int64_t total = n_fft + static_cast<int64_t>(hop) * (frames - 1), length = static_cast<int64_t>(hop) * (frames - 1);
    std::vector<double> acc(static_cast<size_t>(total), 0.0), env(static_cast<size_t>(total), 0.0), win(static_cast<size_t>(n_fft));
    for (int i = 0; i < n_fft; i++) win[static_cast<size_t>(i)] = hann(i, n_fft);
    std::vector<double> cs(static_cast<size_t>(bins * n_fft)), sn(static_cast<size_t>(bins * n_fft));
    for (int k = 0; k < bins; k++) for (int i = 0; i < n_fft; i++) {
        const double a = 2.0 * kPi * k * i / n_fft;
        cs[static_cast<size_t>(k * n_fft + i)] = std::cos(a);
        sn[static_cast<size_t>(k * n_fft + i)] = std::sin(a);
    }
    // Each frame's inverse DFT on several threads into its own row, then the overlap-add in frame order (the same sums in the same order as one thread).
    std::vector<double> rows(static_cast<size_t>(frames) * static_cast<size_t>(n_fft));
    parallel_for(frames, static_cast<int64_t>(bins) * n_fft * 4, [&](int64_t f0, int64_t f1) {
        std::vector<double> re(static_cast<size_t>(bins)), im(static_cast<size_t>(bins));
        for (int64_t f = f0; f < f1; f++) {
            for (int k = 0; k < bins; k++) {
                const double mag = std::exp(static_cast<double>(x[static_cast<size_t>(k) * frames + f]));
                const double ph = std::sin(static_cast<double>(x[static_cast<size_t>(bins + k) * frames + f]));
                re[static_cast<size_t>(k)] = mag * std::cos(ph);
                im[static_cast<size_t>(k)] = mag * std::sin(ph);
            }
            for (int i = 0; i < n_fft; i++) {   // irfft: the imaginary parts of DC and Nyquist are ignored (numpy, torch)
                double v = re[0] + re[static_cast<size_t>(bins - 1)] * ((i & 1) ? -1.0 : 1.0);
                for (int k = 1; k < bins - 1; k++) v += 2.0 * (re[static_cast<size_t>(k)] * cs[static_cast<size_t>(k * n_fft + i)] - im[static_cast<size_t>(k)] * sn[static_cast<size_t>(k * n_fft + i)]);
                rows[static_cast<size_t>(f) * static_cast<size_t>(n_fft) + static_cast<size_t>(i)] = v;
            }
        }
    });
    for (int32_t f = 0; f < frames; f++) {
        for (int i = 0; i < n_fft; i++) {
            const size_t o = static_cast<size_t>(static_cast<int64_t>(f) * hop + i);
            acc[o] += rows[static_cast<size_t>(f) * static_cast<size_t>(n_fft) + static_cast<size_t>(i)] / n_fft * win[static_cast<size_t>(i)];
            env[o] += win[static_cast<size_t>(i)] * win[static_cast<size_t>(i)];
        }
    }
    out->assign(static_cast<size_t>(length), 0.f);
    for (int64_t i = 0; i < length; i++) (*out)[static_cast<size_t>(i)] = static_cast<float>(acc[static_cast<size_t>(i + pad)] / env[static_cast<size_t>(i + pad)]);
}

KokoroNoise kokoro_draw_noise(int64_t samples, int32_t harmonics, uint64_t seed) {
    Xoshiro rng(seed);
    KokoroNoise n;
    n.rand_ini.resize(static_cast<size_t>(harmonics));
    for (float & v : n.rand_ini) v = static_cast<float>(rng.uniform());
    n.noise.resize(static_cast<size_t>(samples * harmonics));
    // Box-Muller, two normals per pair of uniforms: the uniforms drawn in sequence (the generator's order), the transforms on several threads.
    const int64_t pairs = static_cast<int64_t>((n.noise.size() + 1) / 2);
    std::vector<double> u(static_cast<size_t>(2 * pairs));
    for (int64_t p = 0; p < pairs; p++) { u[static_cast<size_t>(2 * p)] = 1.0 - rng.uniform(); u[static_cast<size_t>(2 * p + 1)] = rng.uniform(); }
    parallel_for(pairs, 64, [&](int64_t p0, int64_t p1) {
        for (int64_t p = p0; p < p1; p++) {
            const size_t i = static_cast<size_t>(2 * p);
            const double r = std::sqrt(-2.0 * std::log(u[i])), a = 2.0 * kPi * u[i + 1];
            n.noise[i] = static_cast<float>(r * std::cos(a));
            if (i + 1 < n.noise.size()) n.noise[i + 1] = static_cast<float>(r * std::sin(a));
        }
    });
    return n;
}

struct KokoroPipeline::Impl {
    std::unique_ptr<CoreMlModel> prosody, f0n;
    std::unique_ptr<KokoroVocoder> vocoder;
    std::map<std::string, int32_t> vocab;
    std::map<std::string, std::vector<float>> voices;   // [rows][2 * style_dim]
    KokoroSourceConfig src;
    std::vector<float> lin_w;
    float lin_b = 0;
    int32_t style_dim = 0, hidden = 0, max_tokens = 0, voice_rows = 0;
};

KokoroPipeline::KokoroPipeline(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}
KokoroPipeline::KokoroPipeline(KokoroPipeline &&) noexcept = default;
KokoroPipeline & KokoroPipeline::operator=(KokoroPipeline &&) noexcept = default;
KokoroPipeline::~KokoroPipeline() = default;

int32_t KokoroPipeline::sample_rate() const { return impl_->src.sample_rate; }
const char * KokoroPipeline::device() const { return impl_->vocoder->device(); }
int32_t KokoroPipeline::max_phonemes() const { return std::min(impl_->voice_rows, impl_->max_tokens - 2); }
std::vector<std::string> KokoroPipeline::voices() const {
    std::vector<std::string> out;
    for (const auto & [name, rows] : impl_->voices) out.push_back(name);
    return out;
}

Result<KokoroPipeline> KokoroPipeline::create(const KokoroPaths & paths, const KokoroOptions & o) {
    auto units = parse_coreml_units(o.units);
    if (!units.is_ok()) return units.status();
    auto file = GgufFile::open(paths.model);
    if (!file.is_ok()) return file.status();
    const GgufFile & f = file.value();
    auto impl = std::make_unique<Impl>();
    Impl & m = *impl;
    auto u = [&](const char * key, int32_t * dst) -> Status {
        auto v = f.get_uint(key);
        if (!v.is_ok()) return invalid_argument("'" + paths.model + "': " + v.status().to_string());
        *dst = static_cast<int32_t>(v.value());
        return Status();
    };
    auto fl = [&](const char * key, float * dst) -> Status {
        auto v = f.get_float(key);
        if (!v.is_ok()) return invalid_argument("'" + paths.model + "': " + v.status().to_string());
        *dst = static_cast<float>(v.value());
        return Status();
    };
    for (Status st : {u("kokoro.sample_rate", &m.src.sample_rate), u("kokoro.n_fft", &m.src.n_fft), u("kokoro.hop", &m.src.hop), u("kokoro.upsample", &m.src.upsample),
                      u("kokoro.harmonics", &m.src.harmonics), u("kokoro.style_dim", &m.style_dim), u("kokoro.hidden_dim", &m.hidden), u("kokoro.max_tokens", &m.max_tokens),
                      fl("kokoro.sine_amp", &m.src.sine_amp), fl("kokoro.noise_std", &m.src.noise_std), fl("kokoro.voiced_threshold", &m.src.voiced_threshold)}) {
        if (!st.is_ok()) return st;
    }
    auto kv = [&](const char * key) -> const GgufValue * {
        auto it = f.metadata().find(key);
        return it == f.metadata().end() ? nullptr : &it->second;
    };
    const GgufValue * syms = kv("kokoro.vocab.symbols");
    const GgufValue * ids = kv("kokoro.vocab.ids");
    const GgufValue * vnames = kv("kokoro.voices");
    if (!syms || !ids || !vnames || syms->items.size() != ids->items.size()) return invalid_argument("'" + paths.model + "' has no vocabulary or voice list");
    for (size_t i = 0; i < syms->items.size(); i++) {
        const GgufValue & v = ids->items[i];
        m.vocab[syms->items[i].s] = static_cast<int32_t>(v.type == GgufValueType::I32 || v.type == GgufValueType::I64 ? v.i : static_cast<int64_t>(v.u));
    }
    for (const GgufValue & name : vnames->items) {
        auto t = f.find_tensor("voice." + name.s);
        if (!t.is_ok()) return invalid_argument("'" + paths.model + "' lists voice '" + name.s + "' without its tensor");
        const GgufTensorInfo & info = *t.value();
        if (info.dtype != DType::F32 || info.shape.rank() != 2 || info.shape.dim(1) != 2 * m.style_dim) return invalid_argument("voice '" + name.s + "' has the wrong shape or type");
        const float * p = reinterpret_cast<const float *>(f.tensor_data(info));
        m.voices[name.s].assign(p, p + info.shape.numel());
        m.voice_rows = m.voice_rows == 0 ? static_cast<int32_t>(info.shape.dim(0)) : std::min(m.voice_rows, static_cast<int32_t>(info.shape.dim(0)));
    }
    auto lw = f.find_tensor("source.linear.weight"), lb = f.find_tensor("source.linear.bias");
    if (!lw.is_ok() || !lb.is_ok() || lw.value()->shape.numel() != m.src.harmonics) return invalid_argument("'" + paths.model + "' has no source merge layer");
    const float * w = reinterpret_cast<const float *>(f.tensor_data(*lw.value()));
    m.lin_w.assign(w, w + m.src.harmonics);
    m.lin_b = reinterpret_cast<const float *>(f.tensor_data(*lb.value()))[0];
    auto voc = load_kokoro_vocoder(f, o.spin_wait, o.device);
    if (!voc.is_ok()) return voc.status();
    m.vocoder = std::move(voc).value();
    auto p = CoreMlModel::load(paths.prosody, units.value());
    if (!p.is_ok()) return p.status();
    m.prosody = std::make_unique<CoreMlModel>(std::move(p).value());
    auto fn = CoreMlModel::load(paths.f0n, units.value());
    if (!fn.is_ok()) return fn.status();
    m.f0n = std::make_unique<CoreMlModel>(std::move(fn).value());
    return KokoroPipeline(std::move(impl));
}

Result<std::vector<int32_t>> KokoroPipeline::ids(const std::string & phonemes, int32_t * n_phonemes) const {
    const std::vector<std::string> cps = code_points(phonemes);
    std::vector<int32_t> out{0};
    for (const std::string & c : cps) {
        auto it = impl_->vocab.find(c);
        if (it != impl_->vocab.end()) out.push_back(it->second);
    }
    out.push_back(0);
    if (out.size() == 2) return invalid_argument("no phoneme the model knows");
    if (static_cast<int32_t>(cps.size()) > max_phonemes()) return invalid_argument(std::to_string(cps.size()) + " phonemes: at most " + std::to_string(max_phonemes()) + " per call");
    if (n_phonemes) *n_phonemes = static_cast<int32_t>(cps.size());
    return out;
}

Result<std::vector<float>> KokoroPipeline::style(const std::string & voice, int32_t n_phonemes) const {
    auto it = impl_->voices.find(voice);
    if (it == impl_->voices.end()) return invalid_argument("unknown voice '" + voice + "'");
    if (n_phonemes < 1 || n_phonemes > impl_->voice_rows) return invalid_argument("phoneme count outside the voice's table");
    const size_t w = 2 * static_cast<size_t>(impl_->style_dim);
    return std::vector<float>(it->second.begin() + static_cast<std::ptrdiff_t>((static_cast<size_t>(n_phonemes) - 1) * w),
                              it->second.begin() + static_cast<std::ptrdiff_t>(static_cast<size_t>(n_phonemes) * w));
}

Result<std::vector<float>> KokoroPipeline::synthesize(const std::string & phonemes, const std::string & voice, float speed, uint64_t seed, KokoroTimings * timings) {
    int32_t n = 0;
    auto i = ids(phonemes, &n);
    if (!i.is_ok()) return i.status();
    auto s = style(voice, n);
    if (!s.is_ok()) return s.status();
    return synthesize_ids(i.value(), s.value(), speed, nullptr, seed, timings);
}

Result<std::vector<float>> KokoroPipeline::synthesize_ids(const std::vector<int32_t> & ids, const std::vector<float> & ref_s, float speed, const KokoroNoise * noise,
                                                          uint64_t seed, KokoroTimings * timings, std::vector<int32_t> * frames_out) {
    Impl & m = *impl_;
    if (!(speed > 0.f) || !std::isfinite(speed)) return invalid_argument("speed must be > 0");
    if (ids.size() < 3 || static_cast<int32_t>(ids.size()) > m.max_tokens) return invalid_argument("expected 3.." + std::to_string(m.max_tokens) + " token ids (0, ids..., 0)");
    if (static_cast<int32_t>(ref_s.size()) != 2 * m.style_dim) return invalid_argument("the style vector must have " + std::to_string(2 * m.style_dim) + " values");
    KokoroTimings tm;
    const auto t_all = Clock::now();
    const int64_t T = static_cast<int64_t>(ids.size());
    const int32_t dw = m.hidden + m.style_dim;
    std::vector<float> idf(ids.begin(), ids.end()), s_pred(ref_s.begin() + m.style_dim, ref_s.end());
    std::vector<float> dur(static_cast<size_t>(T)), d(static_cast<size_t>(T * dw)), t_en(static_cast<size_t>(m.hidden * T));
    auto t = Clock::now();
    {
        auto in_ids = TensorView::create(idf.data(), Shape::create({1, T}).value(), DType::F32);
        auto in_s = TensorView::create(s_pred.data(), Shape::create({1, m.style_dim}).value(), DType::F32);
        auto o_dur = TensorView::create(dur.data(), Shape::create({1, T}).value(), DType::F32);
        auto o_d = TensorView::create(d.data(), Shape::create({1, T, dw}).value(), DType::F32);
        auto o_t = TensorView::create(t_en.data(), Shape::create({1, m.hidden, T}).value(), DType::F32);
        if (Status st = m.prosody->predict({{"input_ids", in_ids.value()}, {"style", in_s.value()}}, {{"durations", o_dur.value()}, {"d", o_d.value()}, {"t_en", o_t.value()}}); !st.is_ok()) return st;
    }
    tm.prosody_ms = ms_since(t);
    std::vector<int32_t> frames;
    std::vector<float> en, asr;
    kokoro_align(dur, d, dw, t_en, m.hidden, speed, &frames, &en, &asr);
    int64_t F = 0;
    for (int32_t v : frames) F += v;
    t = Clock::now();
    std::vector<float> f0(static_cast<size_t>(2 * F)), nn(static_cast<size_t>(2 * F));
    {
        auto in_en = TensorView::create(en.data(), Shape::create({1, dw, F}).value(), DType::F32);
        auto in_s = TensorView::create(s_pred.data(), Shape::create({1, m.style_dim}).value(), DType::F32);
        auto o_f0 = TensorView::create(f0.data(), Shape::create({1, 2 * F}).value(), DType::F32);
        auto o_n = TensorView::create(nn.data(), Shape::create({1, 2 * F}).value(), DType::F32);
        if (Status st = m.f0n->predict({{"en", in_en.value()}, {"style", in_s.value()}}, {{"f0", o_f0.value()}, {"n", o_n.value()}}); !st.is_ok()) return st;
    }
    tm.f0n_ms = ms_since(t);
    t = Clock::now();
    const int64_t L = 2 * F * m.src.upsample;
    KokoroNoise drawn;
    if (!noise) drawn = kokoro_draw_noise(L, m.src.harmonics, seed);
    const KokoroNoise & nz = noise ? *noise : drawn;
    if (static_cast<int64_t>(nz.rand_ini.size()) != m.src.harmonics || static_cast<int64_t>(nz.noise.size()) != L * m.src.harmonics) {
        return invalid_argument("noise must have " + std::to_string(m.src.harmonics) + " initial phases and " + std::to_string(L) + " x " + std::to_string(m.src.harmonics) + " values");
    }
    std::vector<float> source, har;
    kokoro_sine_source(f0, nz, m.src, m.lin_w.data(), m.lin_b, &source);
    int32_t hf = 0;
    kokoro_stft(source, m.src.n_fft, m.src.hop, &har, &hf);
    tm.source_ms = ms_since(t);
    if (hf != m.vocoder->har_length(static_cast<int32_t>(F))) return internal_error("the harmonic spectrum has " + std::to_string(hf) + " frames, the vocoder expects " + std::to_string(m.vocoder->har_length(static_cast<int32_t>(F))));
    t = Clock::now();
    std::vector<float> x;
    if (Status st = m.vocoder->run(asr.data(), static_cast<int32_t>(F), f0.data(), nn.data(), ref_s.data(), har.data(), &x); !st.is_ok()) return st;
    tm.vocoder_ms = ms_since(t);
    t = Clock::now();
    std::vector<float> audio;
    kokoro_istft(x, hf, m.src.n_fft, m.src.hop, &audio);
    tm.istft_ms = ms_since(t);
    tm.total_ms = ms_since(t_all);
    tm.tokens = T;
    tm.frames = F;
    tm.audio_s = static_cast<double>(audio.size()) / m.src.sample_rate;
    if (timings) *timings = tm;
    if (frames_out) *frames_out = frames;
    return audio;
}

}  // namespace dooms
