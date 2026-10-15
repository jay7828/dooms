#include <set>

#include "backend/backend.h"

namespace dooms {

namespace {

Status check_one(const Graph & graph, const NamedTensor & b, ValueId id, const char * what) {
    const TypeInfo & want = graph.values()[id].type;
    if (b.tensor.dtype() != want.dtype || b.tensor.shape() != want.shape) {
        return invalid_argument(std::string(what) + " '" + b.name + "' is " + dtype_name(b.tensor.dtype()) +
                                b.tensor.shape().to_string() + " but the graph expects " +
                                dtype_name(want.dtype) + want.shape.to_string());
    }
    if (!b.tensor.is_contiguous()) {
        return invalid_argument(std::string(what) + " '" + b.name + "' must be contiguous");
    }
    return Status();
}

}  // namespace

Status check_input_bindings(const Graph & graph, const std::vector<NamedTensor> & inputs, std::vector<ValueId> * input_ids) {
    input_ids->clear();
    std::set<std::string> seen;
    for (const NamedTensor & b : inputs) {
        if (!seen.insert(b.name).second) return invalid_argument("input '" + b.name + "' is bound twice");
        auto id = graph.find_named(b.name);
        if (!id.is_ok() || graph.values()[id.value()].kind != ValueKind::Input) {
            return not_found("graph has no input named '" + b.name + "'");
        }
        DOOMS_RETURN_IF_ERROR(check_one(graph, b, id.value(), "input"));
        input_ids->push_back(id.value());
    }
    size_t n_graph_inputs = 0;
    for (const Value & v : graph.values()) {
        if (v.kind == ValueKind::Input) {
            n_graph_inputs++;
            if (!seen.count(v.name)) return invalid_argument("input '" + v.name + "' was not provided");
        }
    }
    if (n_graph_inputs != inputs.size()) return internal_error("input count mismatch after validation");
    return Status();
}

Status check_bindings(const Graph & graph, const std::vector<NamedTensor> & inputs,
                      const std::vector<NamedTensor> & outputs, std::vector<ValueId> * input_ids,
                      std::vector<ValueId> * output_ids) {
    output_ids->clear();
    DOOMS_RETURN_IF_ERROR(check_input_bindings(graph, inputs, input_ids));
    std::set<std::string> seen;
    for (const NamedTensor & b : outputs) {
        if (!seen.insert(b.name).second) return invalid_argument("output '" + b.name + "' is bound twice");
        ValueId id = kInvalidValue;
        for (const GraphOutput & o : graph.outputs()) {
            if (o.name == b.name) id = o.value;
        }
        if (id == kInvalidValue) return not_found("graph has no output named '" + b.name + "'");
        DOOMS_RETURN_IF_ERROR(check_one(graph, b, id, "output"));
        output_ids->push_back(id);
    }
    for (const GraphOutput & o : graph.outputs()) {
        if (!seen.count(o.name)) return invalid_argument("output '" + o.name + "' has no destination buffer");
    }
    return Status();
}

Result<ValueId> check_weight(const Graph & graph, std::string_view name, const TensorView & data) {
    auto id = graph.find_named(name);
    if (!id.is_ok() || graph.values()[id.value()].kind != ValueKind::Weight) {
        return not_found("graph has no weight named '" + std::string(name) + "'");
    }
    const TypeInfo & want = graph.values()[id.value()].type;
    if (data.dtype() != want.dtype || data.shape() != want.shape) {
        return invalid_argument("weight '" + std::string(name) + "' is " + dtype_name(data.dtype()) +
                                data.shape().to_string() + " but the graph expects " + dtype_name(want.dtype) +
                                want.shape.to_string());
    }
    if (!data.is_contiguous()) return invalid_argument("weight '" + std::string(name) + "' must be contiguous");
    return id.value();
}

}  // namespace dooms
