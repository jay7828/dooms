// Kokoro text to speech against kokoro 0.9.4 / PyTorch fp32 (tools/parity/kokoro_reference.py; DOOMS_KOKORO_DIR with kokoro.gguf and kokoro-reference.gguf):
// each stage fed the reference's own inputs. Without the directory only the argument checks run.
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "backend/cpu/cpu_kokoro_vocoder.h"
#include "backend/metal/metal_kokoro_vocoder.h"
#include "check.h"
#include "format/gguf.h"
#include "pipeline/kokoro_pipeline.h"
#include "pipeline/kokoro_vocoder_loader.h"

using namespace dooms;

namespace {

std::string dir() {
    const char * d = std::getenv("DOOMS_KOKORO_DIR");
    return d ? d : "";
}

std::vector<float> tensor(const GgufFile & f, const std::string & name) {
    auto info = f.find_tensor(name);
    if (!info.is_ok()) { std::fprintf(stderr, "  missing %s\n", name.c_str()); dooms_test::current_failures()++; return {}; }
    const float * p = reinterpret_cast<const float *>(f.tensor_data(*info.value()));
    return std::vector<float>(p, p + info.value()->shape.numel());
}

double rel_l2(const std::vector<float> & a, const std::vector<float> & b) {
    double d = 0, n = 0;
    for (size_t i = 0; i < b.size() && i < a.size(); i++) { d += (double(a[i]) - b[i]) * (double(a[i]) - b[i]); n += double(b[i]) * b[i]; }
    return std::sqrt(d / n);
}

// Mean absolute difference of log10 power spectrograms (1024-point Hann frames, hop 256): what a listener hears, insensitive to the phase drift the sine
// source turns tiny F0 differences into. The reference model itself differs from itself by ~0.12 between two random seeds.
double log_spec_diff(const std::vector<float> & a, const std::vector<float> & b) {
    const int n = 1024, hop = 256, bins = n / 2 + 1;
    const size_t len = std::min(a.size(), b.size());
    if (len < static_cast<size_t>(n)) return 1e9;
    const size_t frames = 1 + (len - n) / hop;
    std::vector<double> win(n), cs(static_cast<size_t>(bins) * n), sn(static_cast<size_t>(bins) * n);
    for (int i = 0; i < n; i++) win[static_cast<size_t>(i)] = 0.5 - 0.5 * std::cos(2 * M_PI * i / (n - 1));
    for (int k = 0; k < bins; k++) for (int i = 0; i < n; i++) { cs[static_cast<size_t>(k) * n + i] = std::cos(2 * M_PI * k * i / n); sn[static_cast<size_t>(k) * n + i] = std::sin(2 * M_PI * k * i / n); }
    double total = 0;
    for (size_t f = 0; f < frames; f++) {
        for (int k = 0; k < bins; k++) {
            double ar = 0, ai = 0, br = 0, bi = 0;
            for (int i = 0; i < n; i++) {
                const size_t t = f * hop + static_cast<size_t>(i);
                const double w = win[static_cast<size_t>(i)], c = cs[static_cast<size_t>(k) * n + i], s_ = sn[static_cast<size_t>(k) * n + i];
                ar += a[t] * w * c; ai += a[t] * w * s_; br += b[t] * w * c; bi += b[t] * w * s_;
            }
            total += std::fabs(std::log10(ar * ar + ai * ai + 1e-10) - std::log10(br * br + bi * bi + 1e-10));
        }
    }
    return total / static_cast<double>(frames * bins);
}

}  // namespace

// Durations round half to even (torch.round), at least one frame per token, and the expansion repeats each token's columns.
TEST(alignment_rounds_like_torch) {
    std::vector<int32_t> frames;
    std::vector<float> en, asr;
    const std::vector<float> dur = {2.5f, 3.5f, 0.2f, 1.49f}, d = {1, 2, 3, 4}, t = {10, 20, 30, 40};   // d [T][1], t_en [1][T]
    kokoro_align(dur, d, 1, t, 1, 1.f, &frames, &en, &asr);
    EXPECT(frames == std::vector<int32_t>({2, 4, 1, 1}));
    EXPECT(en == std::vector<float>({1, 1, 2, 2, 2, 2, 3, 4}));
    EXPECT(asr == std::vector<float>({10, 10, 20, 20, 20, 20, 30, 40}));
    kokoro_align({5.f}, {1}, 1, {1}, 1, 2.f, &frames, &en, &asr);   // speed 2: 2.5 -> 2
    EXPECT(frames == std::vector<int32_t>({2}));
}

TEST(vocoder_arguments_are_validated) {
    KokoroVocoderConfig c;
    EXPECT(!MetalKokoroVocoder::create(c, {}).is_ok());
    c.hidden = 512; c.style_dim = 128; c.n_fft = 20; c.initial_channel = 512; c.upsample_rates = {10, 6}; c.upsample_kernels = {20, 12}; c.resblock_kernels = {3, 7, 11};
    auto v = MetalKokoroVocoder::create(c, {});
    EXPECT(!v.is_ok() && v.status().code() == Code::InvalidArgument);   // no tensors
    auto cv = CpuKokoroVocoder::create(c, {});
    EXPECT(!cv.is_ok() && cv.status().code() == Code::InvalidArgument);
}

TEST(vocoder_matches_pytorch) {
    const std::string d = dir();
    if (d.empty()) { std::printf("        (skipped: set DOOMS_KOKORO_DIR)\n"); return; }
    auto model = GgufFile::open(d + "/kokoro.gguf");
    auto ref = GgufFile::open(d + "/kokoro-reference.gguf");
    EXPECT(model.is_ok() && ref.is_ok());
    if (!model.is_ok() || !ref.is_ok()) return;
    auto voc = load_kokoro_vocoder(model.value());
    EXPECT(voc.is_ok());
    if (!voc.is_ok()) { std::printf("        %s\n", voc.status().to_string().c_str()); return; }
    const GgufFile & r = ref.value();
    const int64_t n_cases = static_cast<int64_t>(r.get_uint("kokoro_reference.n_cases").value());
    for (int64_t k = 0; k < n_cases; k++) {
        const std::string c = "case" + std::to_string(k) + ".";
        const std::vector<float> asr = tensor(r, c + "asr"), f0 = tensor(r, c + "f0"), n = tensor(r, c + "n"), ref_s = tensor(r, c + "ref_s"), har = tensor(r, c + "har"),
                                 want = tensor(r, c + "x");
        const int32_t frames = static_cast<int32_t>(f0.size() / 2);
        EXPECT(static_cast<int64_t>(har.size()) == 22LL * voc.value()->har_length(frames));
        std::vector<float> got;
        Status st = voc.value()->run(asr.data(), frames, f0.data(), n.data(), ref_s.data(), har.data(), &got);
        EXPECT(st.is_ok());
        if (!st.is_ok()) { std::printf("        %s\n", st.to_string().c_str()); continue; }
        const double err = rel_l2(got, want);
        // column 0 alone (the reflection pad's column, written by its own kernel; 1 of ~30000 columns, invisible in the whole tensor's error)
        const size_t H = got.size() / 22;
        std::vector<float> c0, w0;
        for (size_t ch = 0; ch < 22; ch++) { c0.push_back(got[ch * H]); w0.push_back(want[ch * H]); }
        const double err0 = rel_l2(c0, w0);
        std::printf("        case %lld: %d frames, vocoder output rel %.2e (column 0: %.1e)\n", static_cast<long long>(k), frames, err, err0);
        EXPECT(err < 2e-2);
        EXPECT(err0 < 2e-2);
    }
}

// The host stages against their reference values, then the whole path (Core ML prosody and F0/N, host, Metal vocoder) with the reference's own random draws.
TEST(pipeline_matches_pytorch) {
    const std::string d = dir();
    if (d.empty()) { std::printf("        (skipped: set DOOMS_KOKORO_DIR)\n"); return; }
    auto ref = GgufFile::open(d + "/kokoro-reference.gguf");
    auto pipe = KokoroPipeline::create({d + "/prosody-fp32.mlpackage", d + "/f0n-fp32.mlpackage", d + "/kokoro.gguf"});
    EXPECT(ref.is_ok() && pipe.is_ok());
    if (!ref.is_ok() || !pipe.is_ok()) { if (!pipe.is_ok()) std::printf("        %s\n", pipe.status().to_string().c_str()); return; }
    const GgufFile & r = ref.value();
    KokoroPipeline & p = pipe.value();
    auto model = GgufFile::open(d + "/kokoro.gguf");
    const std::vector<float> lin_w = tensor(model.value(), "source.linear.weight"), lin_b = tensor(model.value(), "source.linear.bias");
    KokoroSourceConfig sc;
    const int64_t n_cases = static_cast<int64_t>(r.get_uint("kokoro_reference.n_cases").value());
    for (int64_t k = 0; k < n_cases; k++) {
        const std::string c = "case" + std::to_string(k) + ".";
        // ids and style row from the phonemes
        int32_t n_ph = 0;
        auto ids = p.ids(r.get_string(c + "phonemes").value(), &n_ph);
        EXPECT(ids.is_ok());
        if (!ids.is_ok()) continue;
        std::vector<int32_t> want_ids;
        {
            auto it = r.metadata().find(c + "ids");
            for (const GgufValue & v : it->second.items) want_ids.push_back(static_cast<int32_t>(v.type == GgufValueType::I32 || v.type == GgufValueType::I64 ? v.i : static_cast<int64_t>(v.u)));
        }
        EXPECT(ids.value() == want_ids);
        auto style = p.style(r.get_string(c + "voice").value(), n_ph);
        const std::vector<float> ref_s = tensor(r, c + "ref_s");
        EXPECT(style.is_ok() && style.value() == ref_s);
        // align
        std::vector<int32_t> frames;
        std::vector<float> en, asr;
        const std::vector<float> dur = tensor(r, c + "durations"), dd = tensor(r, c + "d"), t_en = tensor(r, c + "t_en");
        kokoro_align(dur, dd, 640, t_en, 512, 1.f, &frames, &en, &asr);
        const std::vector<float> want_frames = tensor(r, c + "frames");
        bool frames_equal = frames.size() == want_frames.size();
        for (size_t i = 0; frames_equal && i < frames.size(); i++) frames_equal = frames[i] == static_cast<int32_t>(want_frames[i]);
        EXPECT(frames_equal);
        EXPECT(en == tensor(r, c + "en") && asr == tensor(r, c + "asr"));
        // sine source with the recorded draws, STFT of the reference source, iSTFT of the reference vocoder output
        KokoroNoise nz{tensor(r, c + "rand_ini"), tensor(r, c + "noise")};
        std::vector<float> source;
        kokoro_sine_source(tensor(r, c + "f0"), nz, sc, lin_w.data(), lin_b[0], &source);
        const double src_err = rel_l2(source, tensor(r, c + "source"));
        std::vector<float> har;
        int32_t hf = 0;
        kokoro_stft(tensor(r, c + "source"), sc.n_fft, sc.hop, &har, &hf);
        const std::vector<float> want_har = tensor(r, c + "har");
        const size_t bins = static_cast<size_t>(sc.n_fft / 2 + 1);
        const std::vector<float> mag(har.begin(), har.begin() + static_cast<std::ptrdiff_t>(bins * hf)), want_mag(want_har.begin(), want_har.begin() + static_cast<std::ptrdiff_t>(bins * hf));
        const double mag_err = rel_l2(mag, want_mag);
        std::vector<float> audio_from_x;
        kokoro_istft(tensor(r, c + "x"), hf, sc.n_fft, sc.hop, &audio_from_x);
        const std::vector<float> want_audio = tensor(r, c + "audio");
        const double istft_err = rel_l2(audio_from_x, want_audio);
        // the host + vocoder chain from the reference's own F0 / N / asr (no Core ML): isolates the packages' contribution
        {
            auto voc = load_kokoro_vocoder(model.value());
            std::vector<float> src2, har2, x2, a2;
            int32_t hf2 = 0;
            const std::vector<float> f0r = tensor(r, c + "f0"), nr = tensor(r, c + "n"), asrr = tensor(r, c + "asr");
            kokoro_sine_source(f0r, nz, sc, lin_w.data(), lin_b[0], &src2);
            kokoro_stft(src2, sc.n_fft, sc.hop, &har2, &hf2);
            EXPECT(voc.value()->run(asrr.data(), static_cast<int32_t>(f0r.size() / 2), f0r.data(), nr.data(), ref_s.data(), har2.data(), &x2).is_ok());
            kokoro_istft(x2, hf2, sc.n_fft, sc.hop, &a2);
            const double chain_err = rel_l2(a2, want_audio), chain_spec = log_spec_diff(a2, want_audio);
            std::printf("        case %lld: chain from the reference F0/N/asr: waveform rel %.2e, log-spectrogram %.3f; har rel %.1e\n", static_cast<long long>(k),
                        chain_err, chain_spec, rel_l2(har2, want_har));
            EXPECT(chain_err < 1e-2);
            EXPECT(chain_spec < 0.01);
        }
        // the whole path
        KokoroTimings tm;
        std::vector<int32_t> got_frames;
        auto audio = p.synthesize_ids(ids.value(), ref_s, 1.f, &nz, 0, &tm, &got_frames);
        EXPECT(audio.is_ok());
        const double err = audio.is_ok() ? rel_l2(audio.value(), want_audio) : 1.0;
        const double spec = audio.is_ok() ? log_spec_diff(audio.value(), want_audio) : 1e9;
        std::printf("        case %lld: frames %s, sine source rel %.1e, STFT magnitude rel %.1e, iSTFT rel %.1e, whole path: waveform rel %.2e, log-spectrogram %.3f "
                    "(%.0f ms for %.2f s: prosody %.1f, f0n %.1f, source %.1f, vocoder %.1f, istft %.1f)\n",
                    static_cast<long long>(k), frames_equal ? "equal" : "DIFFERENT", src_err, mag_err, istft_err, err, spec, tm.total_ms, tm.audio_s, tm.prosody_ms,
                    tm.f0n_ms, tm.source_ms, tm.vocoder_ms, tm.istft_ms);
        EXPECT(src_err < 1e-2);
        EXPECT(mag_err < 1e-5);
        EXPECT(istft_err < 1e-5);
        EXPECT(got_frames == frames);
        EXPECT(audio.is_ok() && audio.value().size() == want_audio.size());
        // the waveform follows the sine source's phase, which integrates F0 over every sample (the packages' F0 is within ~1e-6, yet drifts the phase of the
        // 9th harmonic by ~0.07 rad): KModel differs from itself by ~0.12 between two seeds, the numpy split path with torch's own networks by 0.8e-2 .. 2.3e-2
        EXPECT(err < 3e-2);
        EXPECT(spec < 0.03);   // what is heard (log-power spectrogram): KModel vs another seed of itself 0.12, the numpy split path 0.016
    }
}

TEST(seeded_synthesis_is_reproducible) {
    const std::string d = dir();
    if (d.empty()) { std::printf("        (skipped: set DOOMS_KOKORO_DIR)\n"); return; }
    auto pipe = KokoroPipeline::create({d + "/prosody-fp32.mlpackage", d + "/f0n-fp32.mlpackage", d + "/kokoro.gguf"});
    EXPECT(pipe.is_ok());
    if (!pipe.is_ok()) return;
    KokoroPipeline & p = pipe.value();
    const std::string ph = "həlˈO wˈɜɹld.";   // "həlˈO wˈɜɹld."
    auto a = p.synthesize(ph, "af_heart", 1.f, 42), b = p.synthesize(ph, "af_heart", 1.f, 42), c = p.synthesize(ph, "af_heart", 1.f, 43);
    EXPECT(a.is_ok() && b.is_ok() && c.is_ok());
    if (!a.is_ok() || !b.is_ok() || !c.is_ok()) return;
    EXPECT(a.value() == b.value());
    EXPECT(a.value() != c.value() && a.value().size() == c.value().size());
    auto slow = p.synthesize(ph, "af_heart", 0.5f, 42);
    EXPECT(slow.is_ok() && slow.value().size() > a.value().size() * 3 / 2);
    EXPECT(p.synthesize(ph, "nobody", 1.f, 0).status().code() == Code::InvalidArgument);
    EXPECT(p.synthesize("", "af_heart", 1.f, 0).status().code() == Code::InvalidArgument);
    EXPECT(p.synthesize("###", "af_heart", 1.f, 0).status().code() == Code::InvalidArgument);   // no known phoneme
    EXPECT(p.synthesize(ph, "af_heart", 0.f, 0).status().code() == Code::InvalidArgument);
    EXPECT(p.synthesize(std::string(static_cast<size_t>(p.max_phonemes()) + 1, 'a'), "af_heart", 1.f, 0).status().code() == Code::InvalidArgument);
    EXPECT(p.sample_rate() == 24000 && !p.voices().empty());
}

int main() { return dooms_test::run_all(); }
