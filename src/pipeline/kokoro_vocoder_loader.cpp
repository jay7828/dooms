#include "pipeline/kokoro_vocoder_loader.h"

#include "backend/cpu/cpu_kokoro_vocoder.h"
#include "backend/metal/metal_kokoro_vocoder.h"
#include "pipeline/device.h"

#include <string>
#include <vector>

namespace dooms {

namespace {

Result<std::vector<int32_t>> ints(const GgufFile & f, const std::string & key) {
    auto it = f.metadata().find(key);
    if (it == f.metadata().end() || it->second.type != GgufValueType::Array) return invalid_argument("kokoro.gguf has no integer array '" + key + "'");
    std::vector<int32_t> out;
    for (const GgufValue & v : it->second.items) {
        const bool is_signed = v.type == GgufValueType::I8 || v.type == GgufValueType::I16 || v.type == GgufValueType::I32 || v.type == GgufValueType::I64;
        out.push_back(static_cast<int32_t>(is_signed ? v.i : static_cast<int64_t>(v.u)));
    }
    return out;
}

}  // namespace

Result<std::unique_ptr<KokoroVocoder>> load_kokoro_vocoder(const GgufFile & f, bool spin_wait, const std::string & device) {
    KokoroVocoderConfig c;
    for (auto [key, dst] : std::initializer_list<std::pair<const char *, int32_t *>>{
             {"kokoro.hidden_dim", &c.hidden}, {"kokoro.style_dim", &c.style_dim}, {"kokoro.n_fft", &c.n_fft}, {"kokoro.upsample_initial_channel", &c.initial_channel}}) {
        auto v = f.get_uint(key);
        if (!v.is_ok()) return invalid_argument(std::string("kokoro.gguf: ") + v.status().to_string());
        *dst = static_cast<int32_t>(v.value());
    }
    auto rates = ints(f, "kokoro.upsample_rates"), kernels = ints(f, "kokoro.upsample_kernel_sizes"), res = ints(f, "kokoro.resblock_kernel_sizes");
    if (!rates.is_ok()) return rates.status();
    if (!kernels.is_ok()) return kernels.status();
    if (!res.is_ok()) return res.status();
    c.upsample_rates = rates.value();
    c.upsample_kernels = kernels.value();
    c.resblock_kernels = res.value();
    c.spin_wait = spin_wait;
    KokoroTensors t;
    for (const GgufTensorInfo & info : f.tensors()) {
        if (info.name.rfind("dec.", 0) != 0) continue;
        KokoroTensor k;
        k.data = f.tensor_data(info);
        k.dtype = info.dtype;
        for (int i = 0; i < info.shape.rank(); i++) k.shape.push_back(info.shape.dim(i));
        t.emplace(info.name, std::move(k));
    }
    auto choice = resolve_device(device);
    if (!choice.is_ok()) return choice.status();
    if (choice.value().try_gpu) {
        auto v = MetalKokoroVocoder::create(c, t);
        if (v.is_ok()) return std::unique_ptr<KokoroVocoder>(std::move(v).value());
        if (!choice.value().allow_fallback) return v.status();
    }
    auto v = CpuKokoroVocoder::create(c, t);
    if (!v.is_ok()) return v.status();
    return std::unique_ptr<KokoroVocoder>(std::move(v).value());
}

}  // namespace dooms
