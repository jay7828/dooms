#pragma once
// Runs a Core ML model (an .mlpackage or compiled .mlmodelc) on the Apple GPU, CPU or Neural Engine. This is not an IR backend: it
// executes a model that was converted ahead of time (tools/convert), because the Neural Engine is only reachable
// through Core ML. Tensors are host float32, contiguous; the model's own description (names, shapes, element types) is checked against
// every call, so a mismatched file fails with a message instead of producing garbage.
// Apple platforms only (ADR 0005).

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "backend/backend.h"
#include "core/shape.h"
#include "core/status.h"

namespace dooms {

enum class CoreMlUnits {
    All,                 // let Core ML choose: GPU, CPU and the Neural Engine
    CpuAndGpu,
    CpuAndNeuralEngine,
    CpuOnly,
};

// "all", "cpu_gpu", "cpu_ne", "cpu"; InvalidArgument otherwise.
Result<CoreMlUnits> parse_coreml_units(const std::string & name);

struct CoreMlTensorInfo {
    std::string name;
    Shape shape;                  // the declared (default) shape
    std::vector<Shape> allowed;   // an input declared with enumerated shapes lists them all (including `shape`); otherwise just `shape`
    bool flexible() const { return allowed.size() > 1 || !range.empty(); }
    bool accepts(const Shape & s) const;              // `shape`, one of `allowed`, or within `range`
    bool shape_unknown = false;   // an output whose shape Core ML does not declare (models with enumerated input shapes); `shape` is then a placeholder
    std::vector<std::pair<int64_t, int64_t>> range;   // an input declared with a range constraint: [min, max] per dimension (any shape within); empty otherwise
};

class CoreMlModel {
public:
    // `path` is an .mlpackage (compiled once and cached next to it as <path>.mlmodelc, rebuilt when the package is newer) or an .mlmodelc.
    //   NotFound         no such path
    //   InvalidArgument  not a Core ML model, or inputs/outputs the runner does not support (non-float tensors, rank > 4)
    //   Internal         Core ML refused to compile or load the model
    static Result<CoreMlModel> load(const std::string & path, CoreMlUnits units);

    CoreMlModel(CoreMlModel &&) noexcept;
    CoreMlModel & operator=(CoreMlModel &&) noexcept;
    ~CoreMlModel();

    const std::vector<CoreMlTensorInfo> & inputs() const;
    const std::vector<CoreMlTensorInfo> & outputs() const;

    // inputs/outputs must cover the model's inputs/outputs exactly once, as contiguous float32 host tensors. Shapes must be the declared ones; an input
    // with enumerated shapes may take any of them (with a range constraint, any shape within it), and then the outputs take the shapes the model produces for it (checked against what Core ML returns).
    // Not thread-safe. InvalidArgument for a binding that does not match; Internal if Core ML fails.
    Status predict(const std::vector<NamedTensor> & inputs, const std::vector<NamedTensor> & outputs);

private:
    struct Impl;
    explicit CoreMlModel(std::unique_ptr<Impl> impl);
    std::unique_ptr<Impl> impl_;
};

}  // namespace dooms
