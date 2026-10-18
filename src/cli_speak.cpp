#include "cli_speak.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "pipeline/kokoro_pipeline.h"

namespace dooms {

namespace {

constexpr int kExitOk = 0, kExitFailed = 1, kExitUsage = 2;
constexpr char kUsage[] =
    "usage: dooms-cli speak --model DIR --phonemes TEXT --out FILE.wav [--voice NAME] [--speed S] [--seed N] [--units U] [--stats]\n"
    "  --model DIR      prosody-fp32.mlpackage, f0n-fp32.mlpackage and kokoro.gguf from tools/convert/convert_kokoro.py\n"
    "  --phonemes TEXT  misaki's phonemes (the Python binding converts English text: dooms.TTS(...).speak(text))\n"
    "  --voice NAME     default af_heart (--voice list prints them)     --speed S  default 1     --seed N  the sine source's noise (default 0)\n"
    "  --units U        the Core ML packages: cpu (default), cpu_gpu, all     --stats  print the timings to stderr\n";

// 16-bit PCM mono WAV.
bool write_wav(const std::string & path, const std::vector<float> & a, int32_t rate) {
    FILE * f = std::fopen(path.c_str(), "wb");
    if (!f) return false;
    auto u32 = [&](uint32_t v) { const unsigned char b[4] = {static_cast<unsigned char>(v), static_cast<unsigned char>(v >> 8), static_cast<unsigned char>(v >> 16), static_cast<unsigned char>(v >> 24)}; std::fwrite(b, 1, 4, f); };
    auto u16 = [&](uint16_t v) { const unsigned char b[2] = {static_cast<unsigned char>(v), static_cast<unsigned char>(v >> 8)}; std::fwrite(b, 1, 2, f); };
    const uint32_t data = static_cast<uint32_t>(a.size() * 2);
    std::fwrite("RIFF", 1, 4, f); u32(36 + data); std::fwrite("WAVEfmt ", 1, 8, f); u32(16); u16(1); u16(1); u32(static_cast<uint32_t>(rate));
    u32(static_cast<uint32_t>(rate) * 2); u16(2); u16(16); std::fwrite("data", 1, 4, f); u32(data);
    for (float v : a) {
        const float c = v > 1.f ? 1.f : v < -1.f ? -1.f : v;
        u16(static_cast<uint16_t>(static_cast<int16_t>(std::lround(c * 32767.f))));
    }
    return std::fclose(f) == 0;
}

}  // namespace

int run_speak_command(int argc, char ** argv) {
    std::string model, phonemes, out, voice = "af_heart";
    KokoroOptions o;
    float speed = 1.f;
    unsigned long long seed = 0;
    bool stats = false;
    for (int i = 1; i < argc; i++) {
        const std::string flag = argv[i];
        if (flag == "--stats") { stats = true; continue; }
        if (i + 1 >= argc) { std::fprintf(stderr, "error: missing value for %s\n%s", flag.c_str(), kUsage); return kExitUsage; }
        const char * v = argv[++i];
        char * end = nullptr;
        if (flag == "--model") model = v;
        else if (flag == "--phonemes") phonemes = v;
        else if (flag == "--out") out = v;
        else if (flag == "--voice") voice = v;
        else if (flag == "--units") o.units = v;
        else if (flag == "--speed") { speed = std::strtof(v, &end); if (end == v || *end || !(speed > 0.f)) { std::fprintf(stderr, "error: --speed must be > 0\n"); return kExitUsage; } }
        else if (flag == "--seed") { seed = std::strtoull(v, &end, 10); if (end == v || *end) { std::fprintf(stderr, "error: --seed must be a number\n"); return kExitUsage; } }
        else { std::fprintf(stderr, "error: unknown option '%s'\n%s", flag.c_str(), kUsage); return kExitUsage; }
    }
    if (model.empty() || ((phonemes.empty() || out.empty()) && voice != "list")) { std::fprintf(stderr, "error: --model, --phonemes and --out are required\n%s", kUsage); return kExitUsage; }
    auto tts = KokoroPipeline::create({model + "/prosody-fp32.mlpackage", model + "/f0n-fp32.mlpackage", model + "/kokoro.gguf"}, o);
    if (!tts.is_ok()) { std::fprintf(stderr, "error: %s\n", tts.status().to_string().c_str()); return kExitFailed; }
    if (voice == "list") { for (const std::string & v : tts.value().voices()) std::printf("%s\n", v.c_str()); return kExitOk; }
    KokoroTimings tm;
    auto audio = tts.value().synthesize(phonemes, voice, speed, seed, &tm);
    if (!audio.is_ok()) { std::fprintf(stderr, "error: %s\n", audio.status().to_string().c_str()); return kExitFailed; }
    if (!write_wav(out, audio.value(), tts.value().sample_rate())) { std::fprintf(stderr, "error: cannot write %s\n", out.c_str()); return kExitFailed; }
    if (stats) {
        std::fprintf(stderr, "%.2f s of audio in %.0f ms (RTF %.4f): prosody %.1f, F0/N %.1f, source %.1f, vocoder %.1f, iSTFT %.1f ms; %lld tokens, %lld frames\n", tm.audio_s,
                     tm.total_ms, tm.rtf(), tm.prosody_ms, tm.f0n_ms, tm.source_ms, tm.vocoder_ms, tm.istft_ms, static_cast<long long>(tm.tokens), static_cast<long long>(tm.frames));
    }
    return kExitOk;
}

}  // namespace dooms
