#include "backend/coreml/coreml_model.h"

#import <CoreML/CoreML.h>
#import <Foundation/Foundation.h>

#include <array>
#include <cstdint>
#include <cstring>
#include <map>

namespace dooms {

namespace {

std::string to_std(NSString * s) { return s ? std::string([s UTF8String]) : std::string(); }

std::string describe(NSError * e) { return e ? to_std([e localizedDescription]) : std::string("unknown error"); }

bool ends_with(const std::string & s, const std::string & suffix) {
    return s.size() >= suffix.size() && s.compare(s.size() - suffix.size(), suffix.size(), suffix) == 0;
}

MLComputeUnits to_ml(CoreMlUnits u) {
    switch (u) {
        case CoreMlUnits::All: return MLComputeUnitsAll;
        case CoreMlUnits::CpuAndGpu: return MLComputeUnitsCPUAndGPU;
        case CoreMlUnits::CpuAndNeuralEngine: return MLComputeUnitsCPUAndNeuralEngine;
        case CoreMlUnits::CpuOnly: return MLComputeUnitsCPUOnly;
    }
    return MLComputeUnitsAll;
}

// Element types a model may declare for its tensors; everything is exchanged with the caller as float32.
enum class Elem { F32, F16 };

struct Port {
    CoreMlTensorInfo info;
    Elem elem = Elem::F32;
};

Status read_port(MLFeatureDescription * d, Port * out) {
    const std::string name = to_std(d.name);
    if (d.type != MLFeatureTypeMultiArray) return invalid_argument("'" + name + "' is not a multi-array tensor");
    MLMultiArrayConstraint * c = d.multiArrayConstraint;
    if (c.shape.count == 0) {  // not declared: the shape follows the input (enumerated input shapes); every call is checked against the returned array
        out->info = {name, Shape::create({1}).value(), {}, true, {}};
        out->info.allowed.push_back(out->info.shape);
        switch (c.dataType) {
            case MLMultiArrayDataTypeFloat32: out->elem = Elem::F32; break;
            case MLMultiArrayDataTypeFloat16: out->elem = Elem::F16; break;
            default: return unsupported("'" + name + "' has an element type other than float32/float16");
        }
        return Status();
    }
    if (c.shape.count > static_cast<NSUInteger>(kMaxDims)) return invalid_argument("'" + name + "' has rank " + std::to_string(c.shape.count) + " (supported: 1.." + std::to_string(kMaxDims) + ")");
    int64_t dims[kMaxDims];
    for (NSUInteger i = 0; i < c.shape.count; i++) dims[i] = [c.shape[i] longLongValue];
    auto shape = Shape::create(dims, static_cast<int>(c.shape.count));
    if (!shape.is_ok()) return invalid_argument("'" + name + "': " + shape.status().message() + " (a flexible shape is not supported)");
    switch (c.dataType) {
        case MLMultiArrayDataTypeFloat32: out->elem = Elem::F32; break;
        case MLMultiArrayDataTypeFloat16: out->elem = Elem::F16; break;  // converted from / to the caller's float32
        default: return unsupported("'" + name + "' has an element type other than float32/float16");
    }
    out->info = {name, shape.value(), {shape.value()}, false, {}};
    if (c.shapeConstraint.type == MLMultiArrayShapeConstraintTypeRange) {
        NSArray<NSValue *> * ranges = c.shapeConstraint.sizeRangeForDimension;
        if (ranges.count != c.shape.count) return invalid_argument("'" + name + "' has a range constraint of a different rank");
        for (NSValue * v : ranges) {
            const NSRange r = v.rangeValue;
            const int64_t lo = static_cast<int64_t>(r.location);
            const int64_t hi = r.length >= static_cast<NSUInteger>(INT64_MAX) - r.location ? INT64_MAX : static_cast<int64_t>(r.location + r.length);
            out->info.range.emplace_back(lo, hi);
        }
    }
    if (c.shapeConstraint.type == MLMultiArrayShapeConstraintTypeEnumerated) {
        out->info.allowed.clear();
        for (NSArray<NSNumber *> * dims_ns in c.shapeConstraint.enumeratedShapes) {
            if (dims_ns.count != c.shape.count) return invalid_argument("'" + name + "' has enumerated shapes of a different rank");
            int64_t d[kMaxDims];
            for (NSUInteger i = 0; i < dims_ns.count; i++) d[i] = [dims_ns[i] longLongValue];
            auto sh = Shape::create(d, static_cast<int>(dims_ns.count));
            if (!sh.is_ok()) return invalid_argument("'" + name + "': " + sh.status().message());
            out->info.allowed.push_back(sh.value());
        }
        if (out->info.allowed.empty()) out->info.allowed.push_back(shape.value());
    }
    return Status();
}

}  // namespace

bool CoreMlTensorInfo::accepts(const Shape & s) const {
    if (s == shape) return true;
    for (const Shape & a : allowed) if (s == a) return true;
    if (range.empty() || s.rank() != static_cast<int>(range.size())) return false;
    for (int i = 0; i < s.rank(); i++) if (s.dim(i) < range[static_cast<size_t>(i)].first || s.dim(i) > range[static_cast<size_t>(i)].second) return false;
    return true;
}

Result<CoreMlUnits> parse_coreml_units(const std::string & name) {
    if (name == "all") return CoreMlUnits::All;
    if (name == "cpu_gpu") return CoreMlUnits::CpuAndGpu;
    if (name == "cpu_ne") return CoreMlUnits::CpuAndNeuralEngine;
    if (name == "cpu") return CoreMlUnits::CpuOnly;
    return invalid_argument("unknown Core ML compute units '" + name + "' (expected all, cpu_gpu, cpu_ne or cpu)");
}

struct CoreMlModel::Impl {
    MLModel * model = nil;
    std::vector<Port> inputs, outputs;
    std::vector<CoreMlTensorInfo> input_infos, output_infos;
};

CoreMlModel::CoreMlModel(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}
CoreMlModel::CoreMlModel(CoreMlModel &&) noexcept = default;
CoreMlModel & CoreMlModel::operator=(CoreMlModel &&) noexcept = default;
CoreMlModel::~CoreMlModel() = default;

const std::vector<CoreMlTensorInfo> & CoreMlModel::inputs() const { return impl_->input_infos; }
const std::vector<CoreMlTensorInfo> & CoreMlModel::outputs() const { return impl_->output_infos; }

Result<CoreMlModel> CoreMlModel::load(const std::string & path, CoreMlUnits units) {
    @autoreleasepool {
        NSFileManager * fm = [NSFileManager defaultManager];
        NSString * ns_path = [NSString stringWithUTF8String:path.c_str()];
        if (!ns_path || ![fm fileExistsAtPath:ns_path]) return not_found("Core ML model '" + path + "' does not exist");

        NSURL * compiled = nil;
        NSError * error = nil;
        if (ends_with(path, ".mlmodelc")) {
            compiled = [NSURL fileURLWithPath:ns_path isDirectory:YES];
        } else if (ends_with(path, ".mlpackage")) {
            // Compile once and keep the result next to the package; recompile when the package is newer.
            NSString * cache = [ns_path stringByAppendingString:@".mlmodelc"];
            NSDate * pkg_date = [[fm attributesOfItemAtPath:ns_path error:nil] fileModificationDate];
            NSDate * cache_date = [fm fileExistsAtPath:cache] ? [[fm attributesOfItemAtPath:cache error:nil] fileModificationDate] : nil;
            if (!cache_date || (pkg_date && [cache_date compare:pkg_date] == NSOrderedAscending)) {
                NSURL * temp = [MLModel compileModelAtURL:[NSURL fileURLWithPath:ns_path isDirectory:YES] error:&error];
                if (!temp) return internal_error("Core ML could not compile '" + path + "': " + describe(error));
                [fm removeItemAtPath:cache error:nil];
                if (![fm moveItemAtURL:temp toURL:[NSURL fileURLWithPath:cache isDirectory:YES] error:&error]) {
                    return internal_error("cannot store the compiled model at '" + to_std(cache) + "': " + describe(error));
                }
            }
            compiled = [NSURL fileURLWithPath:cache isDirectory:YES];
        } else {
            return invalid_argument("'" + path + "' is neither an .mlpackage nor an .mlmodelc");
        }

        MLModelConfiguration * cfg = [[MLModelConfiguration alloc] init];
        cfg.computeUnits = to_ml(units);
        MLModel * model = [MLModel modelWithContentsOfURL:compiled configuration:cfg error:&error];
        if (!model) return internal_error("Core ML could not load '" + path + "': " + describe(error));

        auto impl = std::make_unique<Impl>();
        impl->model = model;
        // Sorted by name so the order is deterministic.
        for (NSString * key in [[model.modelDescription.inputDescriptionsByName allKeys] sortedArrayUsingSelector:@selector(compare:)]) {
            Port p;
            Status st = read_port(model.modelDescription.inputDescriptionsByName[key], &p);
            if (!st.is_ok()) return Status(st.code(), "Core ML model '" + path + "': " + st.message());
            impl->input_infos.push_back(p.info);
            impl->inputs.push_back(p);
        }
        for (NSString * key in [[model.modelDescription.outputDescriptionsByName allKeys] sortedArrayUsingSelector:@selector(compare:)]) {
            Port p;
            Status st = read_port(model.modelDescription.outputDescriptionsByName[key], &p);
            if (!st.is_ok()) return Status(st.code(), "Core ML model '" + path + "': " + st.message());
            impl->output_infos.push_back(p.info);
            impl->outputs.push_back(p);
        }
        if (impl->inputs.empty() || impl->outputs.empty()) return invalid_argument("Core ML model '" + path + "' has no inputs or no outputs");
        return CoreMlModel(std::move(impl));
    }
}

Status CoreMlModel::predict(const std::vector<NamedTensor> & inputs, const std::vector<NamedTensor> & outputs) {
    @autoreleasepool {
        Impl & m = *impl_;
        bool flexible_input = false;
        for (const Port & p : m.inputs) flexible_input = flexible_input || p.info.flexible();
        auto check = [flexible_input](const std::vector<Port> & ports, const std::vector<NamedTensor> & given, const char * what, bool is_input) -> Status {
            if (given.size() != ports.size()) return invalid_argument(std::string("expected ") + std::to_string(ports.size()) + " " + what + " tensors, got " + std::to_string(given.size()));
            std::map<std::string, int> seen;
            for (const NamedTensor & t : given) {
                if (++seen[t.name] > 1) return invalid_argument(std::string(what) + " '" + t.name + "' is bound twice");
                const Port * port = nullptr;
                for (const Port & p : ports) if (p.info.name == t.name) port = &p;
                if (!port) return invalid_argument(std::string("the model has no ") + what + " '" + t.name + "'");
                if (t.tensor.dtype() != DType::F32 && t.tensor.dtype() != DType::F16) return invalid_argument(std::string(what) + " '" + t.name + "' must be float32 or float16");
                // a float16 binding hands over the model's own element type without a conversion: only for ports the model declares as float16
                if (t.tensor.dtype() == DType::F16 && port->elem != Elem::F16) return invalid_argument(std::string(what) + " '" + t.name + "' is bound as float16 but the model declares it as float32");
                bool shape_ok = port->info.accepts(t.tensor.shape());
                // outputs of a model with enumerated input shapes take shapes that depend on the input; Core ML's returned shape is checked below
                if (!is_input && (flexible_input || port->info.shape_unknown)) shape_ok = port->info.shape_unknown || t.tensor.shape().rank() == port->info.shape.rank();
                if (!shape_ok) {
                    return invalid_argument(std::string(what) + " '" + t.name + "' has shape " + t.tensor.shape().to_string() + ", the model declares " + port->info.shape.to_string());
                }
                if (!t.tensor.is_contiguous()) return invalid_argument(std::string(what) + " '" + t.name + "' must be contiguous");
            }
            return Status();
        };
        DOOMS_RETURN_IF_ERROR(check(m.inputs, inputs, "input", true));
        DOOMS_RETURN_IF_ERROR(check(m.outputs, outputs, "output", false));

        NSMutableDictionary<NSString *, MLFeatureValue *> * feed = [NSMutableDictionary dictionary];
        std::vector<std::vector<__fp16>> half_inputs;  // converted copies, alive until the prediction returns
        half_inputs.reserve(inputs.size());
        for (const NamedTensor & t : inputs) {
            Elem elem = Elem::F32;
            for (const Port & p : m.inputs) if (p.info.name == t.name) elem = p.elem;
            NSMutableArray<NSNumber *> * shape = [NSMutableArray array];
            NSMutableArray<NSNumber *> * strides = [NSMutableArray array];
            const int64_t elem_bytes = t.tensor.dtype() == DType::F16 ? 2 : 4;
            for (int i = 0; i < t.tensor.shape().rank(); i++) {
                [shape addObject:@(t.tensor.shape().dim(i))];
                [strides addObject:@(t.tensor.stride_bytes(i) / elem_bytes)];
            }
            NSError * error = nil;
            void * data = t.tensor.data();
            MLMultiArrayDataType type = t.tensor.dtype() == DType::F16 ? MLMultiArrayDataTypeFloat16 : MLMultiArrayDataTypeFloat32;
            if (elem == Elem::F16 && t.tensor.dtype() == DType::F32) {
                const float * src = static_cast<const float *>(t.tensor.data());
                half_inputs.emplace_back(static_cast<size_t>(t.tensor.shape().numel()));
                for (size_t i = 0; i < half_inputs.back().size(); i++) half_inputs.back()[i] = static_cast<__fp16>(src[i]);
                data = half_inputs.back().data();
                type = MLMultiArrayDataTypeFloat16;
                for (int i = 0; i < t.tensor.shape().rank(); i++) strides[static_cast<NSUInteger>(i)] = @(t.tensor.stride_bytes(i) / static_cast<int64_t>(sizeof(float)));  // same element strides
            }
            MLMultiArray * arr = [[MLMultiArray alloc] initWithDataPointer:data shape:shape dataType:type strides:strides deallocator:nil error:&error];
            if (!arr) return internal_error("cannot wrap input '" + t.name + "': " + describe(error));
            feed[[NSString stringWithUTF8String:t.name.c_str()]] = [MLFeatureValue featureValueWithMultiArray:arr];
        }
        NSError * error = nil;
        MLDictionaryFeatureProvider * provider = [[MLDictionaryFeatureProvider alloc] initWithDictionary:feed error:&error];
        if (!provider) return internal_error("cannot build the input features: " + describe(error));
        id<MLFeatureProvider> result = [m.model predictionFromFeatures:provider error:&error];
        if (!result) return internal_error("Core ML prediction failed: " + describe(error));

        for (const NamedTensor & t : outputs) {
            MLFeatureValue * v = [result featureValueForName:[NSString stringWithUTF8String:t.name.c_str()]];
            MLMultiArray * arr = v.multiArrayValue;
            if (!arr) return internal_error("Core ML returned no tensor for output '" + t.name + "'");
            const int rank = t.tensor.shape().rank();
            if (static_cast<int>(arr.shape.count) != rank) return internal_error("output '" + t.name + "' came back with rank " + std::to_string(arr.shape.count));
            for (int i = 0; i < rank; i++) {
                if ([arr.shape[static_cast<NSUInteger>(i)] longLongValue] != t.tensor.shape().dim(i)) return internal_error("output '" + t.name + "' came back with a different shape");
            }
            // Core ML may pad rows (strides larger than the dense layout): read element by element through the strides.
            std::array<int64_t, kMaxDims> stride{};
            for (int i = 0; i < rank; i++) stride[static_cast<size_t>(i)] = [arr.strides[static_cast<NSUInteger>(i)] longLongValue];
            const bool want_half = t.tensor.dtype() == DType::F16;
            float * dst = static_cast<float *>(t.tensor.data());
            __fp16 * dst_half = static_cast<__fp16 *>(t.tensor.data());
            const int64_t n = t.tensor.shape().numel();
            const bool half = arr.dataType == MLMultiArrayDataTypeFloat16;
            if (!half && arr.dataType != MLMultiArrayDataTypeFloat32) return unsupported("output '" + t.name + "' came back in an unsupported element type");
            if (want_half && !half) return internal_error("output '" + t.name + "' came back as float32 but was bound as float16");
            std::array<int64_t, kMaxDims> dims{};
            for (int i = 0; i < rank; i++) dims[static_cast<size_t>(i)] = t.tensor.shape().dim(i);
            [arr getBytesWithHandler:^(const void * bytes, NSInteger size) {
                (void) size;
                // Row-wise copy: the innermost dimension is walked with its own stride in a tight loop (vectorizable, a plain memcpy when dense float32);
                // the outer dimensions use the strides Core ML reports, which may include row padding.
                const int64_t inner = dims[static_cast<size_t>(rank - 1)], inner_stride = stride[static_cast<size_t>(rank - 1)];
                const int64_t rows = n / inner;
                std::array<int64_t, kMaxDims> idx{};
                for (int64_t row = 0; row < rows; row++) {
                    int64_t off = 0;
                    for (int i = 0; i < rank - 1; i++) off += idx[static_cast<size_t>(i)] * stride[static_cast<size_t>(i)];
                    float * out_row = dst + row * inner;
                    if (want_half) {
                        const __fp16 * src_row = static_cast<const __fp16 *>(bytes) + off;
                        if (inner_stride == 1) std::memcpy(dst_half + row * inner, src_row, static_cast<size_t>(inner) * sizeof(__fp16));
                        else for (int64_t k = 0; k < inner; k++) dst_half[row * inner + k] = src_row[k * inner_stride];
                    } else if (half) {
                        const __fp16 * src_row = static_cast<const __fp16 *>(bytes) + off;
                        for (int64_t k = 0; k < inner; k++) out_row[k] = static_cast<float>(src_row[k * inner_stride]);
                    } else {
                        const float * src_row = static_cast<const float *>(bytes) + off;
                        if (inner_stride == 1) std::memcpy(out_row, src_row, static_cast<size_t>(inner) * sizeof(float));
                        else for (int64_t k = 0; k < inner; k++) out_row[k] = src_row[k * inner_stride];
                    }
                    for (int i = rank - 2; i >= 0; i--) {
                        if (++idx[static_cast<size_t>(i)] < dims[static_cast<size_t>(i)]) break;
                        idx[static_cast<size_t>(i)] = 0;
                    }
                }
            }];
        }
        return Status();
    }
}

}  // namespace dooms
