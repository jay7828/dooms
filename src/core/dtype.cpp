#include "core/dtype.h"

#include <string>

namespace dooms {

namespace {

struct DTypeInfo {
    DType type;
    const char * name;
    size_t size;
    bool is_float;
};

constexpr DTypeInfo kDTypes[] = {
    {DType::F32,  "f32",  4, true},
    {DType::F16,  "f16",  2, true},
    {DType::BF16, "bf16", 2, true},
    {DType::I32,  "i32",  4, false},
};

const DTypeInfo & info(DType t) {
    for (const DTypeInfo & i : kDTypes) {
        if (i.type == t) return i;
    }
    DOOMS_CHECK(false, "unknown DType value");
    return kDTypes[0];  // unreachable
}

}  // namespace

size_t dtype_size(DType t) { return info(t).size; }
const char * dtype_name(DType t) { return info(t).name; }
bool dtype_is_float(DType t) { return info(t).is_float; }

Result<DType> dtype_from_name(std::string_view name) {
    for (const DTypeInfo & i : kDTypes) {
        if (name == i.name) return i.type;
    }
    return invalid_argument("unknown dtype name '" + std::string(name) + "'");
}

}  // namespace dooms
