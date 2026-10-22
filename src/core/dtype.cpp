#include "core/dtype.h"

#include <string>

namespace dooms {

namespace {

struct DTypeInfo {
    DType type;
    const char * name;
    size_t size;      // bytes per block
    bool is_float;
    int64_t block;    // elements per block
};

// Block layouts (ggml-common.h): q4_0 {f16 d; u8 qs[16]}, q5_0 {f16 d; u8 qh[4]; u8 qs[16]}, q8_0 {f16 d; i8 qs[32]},
// q4_K {f16 d, dmin; u8 scales[12]; u8 qs[128]}, q6_K {u8 ql[128]; u8 qh[64]; i8 scales[16]; f16 d}.
constexpr DTypeInfo kDTypes[] = {
    {DType::F32,  "f32",  4, true, 1},
    {DType::F16,  "f16",  2, true, 1},
    {DType::BF16, "bf16", 2, true, 1},
    {DType::I32,  "i32",  4, false, 1},
    {DType::Q4_0, "q4_0", 18, true, 32},
    {DType::Q5_0, "q5_0", 22, true, 32},
    {DType::Q8_0, "q8_0", 34, true, 32},
    {DType::Q4_K, "q4_k", 144, true, 256},
    {DType::Q6_K, "q6_k", 210, true, 256},
};

const DTypeInfo & info(DType t) {
    for (const DTypeInfo & i : kDTypes) {
        if (i.type == t) return i;
    }
    DOOMS_CHECK(false, "unknown DType value");
    return kDTypes[0];  // unreachable
}

}  // namespace

size_t dtype_size(DType t) {
    DOOMS_CHECK(info(t).block == 1, "dtype_size of a quantized block type");
    return info(t).size;
}
bool dtype_is_quantized(DType t) { return info(t).block > 1; }
int64_t dtype_block_elems(DType t) { return info(t).block; }
int64_t dtype_block_bytes(DType t) { return static_cast<int64_t>(info(t).size); }
Result<uint64_t> dtype_bytes(DType t, int64_t n) {
    const DTypeInfo & i = info(t);
    if (n < 0 || n % i.block != 0) return invalid_argument(std::to_string(n) + " elements are not a whole number of " + i.name + " blocks of " + std::to_string(i.block));
    return static_cast<uint64_t>(n / i.block) * i.size;
}
const char * dtype_name(DType t) { return info(t).name; }
bool dtype_is_float(DType t) { return info(t).is_float; }

Result<DType> dtype_from_name(std::string_view name) {
    for (const DTypeInfo & i : kDTypes) {
        if (name == i.name) return i.type;
    }
    return invalid_argument("unknown dtype name '" + std::string(name) + "'");
}

}  // namespace dooms
