#pragma once
// Byte-level GGUF builder for tests: produces valid files, and lets tests override any field to
// produce specific corruptions. Independent of the code under test (src/format).
#include <cstdint>
#include <cstring>
#include <optional>
#include <string>
#include <vector>

#include "format/gguf.h"

namespace dooms_test_gguf {

using dooms::GgufValueType;
using Bytes = std::vector<uint8_t>;

// ggml_type ids used in files (see ggml.h).
constexpr int32_t kTypeF32 = 0, kTypeF16 = 1, kTypeQ4_0 = 2, kTypeI32 = 26, kTypeBF16 = 30;
constexpr uint32_t kDefaultAlign = 32;

template <class T>
void put(Bytes & b, T v) {
    const auto * p = reinterpret_cast<const uint8_t *>(&v);
    b.insert(b.end(), p, p + sizeof(T));
}
void put_str(Bytes & b, const std::string & s) {
    put<uint64_t>(b, s.size());
    b.insert(b.end(), s.begin(), s.end());
}
uint64_t pad_to(uint64_t n, uint64_t a) { return (n + a - 1) / a * a; }

Bytes kv_head(const std::string & key, GgufValueType t) {
    Bytes b;
    put_str(b, key);
    put<int32_t>(b, static_cast<int32_t>(t));
    return b;
}
template <class T>
Bytes kv(const std::string & key, GgufValueType t, T v) {
    Bytes b = kv_head(key, t);
    put<T>(b, v);
    return b;
}
Bytes kv_string(const std::string & key, const std::string & v) {
    Bytes b = kv_head(key, GgufValueType::String);
    put_str(b, v);
    return b;
}

template <class T>
Bytes raw_bytes(const std::vector<T> & v) {
    Bytes b(v.size() * sizeof(T));
    if (!b.empty()) std::memcpy(b.data(), v.data(), b.size());
    return b;
}

struct TensorSpec {
    std::string name;
    std::vector<int64_t> file_dims;  // fastest-varying first, as stored in the file
    int32_t type = kTypeF32;
    Bytes data;
    std::optional<uint64_t> offset;  // override the computed data offset
};

struct Builder {
    uint32_t version = 3;
    std::optional<uint32_t> alignment;  // written as general.alignment when set
    std::vector<Bytes> kvs;
    std::vector<TensorSpec> tensors;
    std::optional<uint64_t> n_tensors_override, n_kv_override;
    mutable uint64_t info_end = 0;      // absolute end of the tensor infos (before padding), set by build()
    mutable uint64_t data_start = 0;    // absolute start of the data section, set by build()
    mutable uint64_t data_end = 0;      // absolute end of the last tensor's bytes, set by build()

    Bytes build() const {
        const uint64_t align = alignment.value_or(kDefaultAlign);
        Bytes out = {'G', 'G', 'U', 'F'};
        put<uint32_t>(out, version);
        put<uint64_t>(out, n_tensors_override.value_or(tensors.size()));
        put<uint64_t>(out, n_kv_override.value_or(kvs.size() + (alignment ? 1 : 0)));
        for (const Bytes & k : kvs) out.insert(out.end(), k.begin(), k.end());
        if (alignment) {
            Bytes k = kv<uint32_t>("general.alignment", GgufValueType::U32, *alignment);
            out.insert(out.end(), k.begin(), k.end());
        }
        uint64_t running = 0;
        for (const TensorSpec & t : tensors) {
            put_str(out, t.name);
            put<uint32_t>(out, static_cast<uint32_t>(t.file_dims.size()));
            for (int64_t d : t.file_dims) put<int64_t>(out, d);
            put<int32_t>(out, t.type);
            put<uint64_t>(out, t.offset.value_or(running));
            running += pad_to(t.data.size(), align);
        }
        info_end = out.size();
        if (!tensors.empty()) out.resize(pad_to(out.size(), align), 0);
        data_start = out.size();
        data_end = out.size();
        for (const TensorSpec & t : tensors) {
            const uint64_t start = out.size();
            out.insert(out.end(), t.data.begin(), t.data.end());
            data_end = start + t.data.size();
            out.resize(start + pad_to(t.data.size(), align), 0);
        }
        return out;
    }
};

}  // namespace dooms_test_gguf
