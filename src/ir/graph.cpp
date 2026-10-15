#include "ir/graph.h"

#include <set>
#include <sstream>

namespace dooms {

namespace {

std::string node_label(size_t index, const Node & n) {
    return "node " + std::to_string(index) + " (" + op_name(n.op) + ")";
}

std::string type_str(const TypeInfo & t) { return std::string(dtype_name(t.dtype)) + t.shape.to_string(); }

}  // namespace

Result<Graph> Graph::from_parts(std::vector<Value> values, std::vector<Node> nodes,
                                std::vector<GraphOutput> outputs) {
    Graph g;
    g.values_ = std::move(values);
    g.nodes_ = std::move(nodes);
    g.outputs_ = std::move(outputs);
    Status st = g.validate();
    if (!st.is_ok()) return st;
    return g;
}

Result<ValueId> Graph::add_named(ValueKind kind, std::string name, const Shape & shape, DType dtype) {
    if (name.empty()) return invalid_argument("input/weight name must not be empty");
    if (find_named(name).is_ok()) return invalid_argument("duplicate input/weight name '" + name + "'");
    Value v;
    v.kind = kind;
    v.type = TypeInfo{shape, dtype};
    v.name = std::move(name);
    values_.push_back(std::move(v));
    return static_cast<ValueId>(values_.size() - 1);
}

Result<ValueId> Graph::add_input(std::string name, const Shape & shape, DType dtype) {
    return add_named(ValueKind::Input, std::move(name), shape, dtype);
}

Result<ValueId> Graph::add_weight(std::string name, const Shape & shape, DType dtype) {
    return add_named(ValueKind::Weight, std::move(name), shape, dtype);
}

Result<ValueId> Graph::add_node(OpKind op, std::vector<ValueId> inputs, Attrs attrs) {
    std::vector<TypeInfo> types;
    types.reserve(inputs.size());
    for (ValueId id : inputs) {
        if (!has_value(id)) {
            return invalid_argument(std::string(op_name(op)) + ": input value id " + std::to_string(id) + " does not exist");
        }
        types.push_back(values_[id].type);
    }
    auto out = infer_output(op, types, attrs);
    if (!out.is_ok()) return out.status();

    Value v;
    v.kind = ValueKind::Intermediate;
    v.type = out.value();
    values_.push_back(std::move(v));
    const ValueId out_id = static_cast<ValueId>(values_.size() - 1);

    Node n;
    n.op = op;
    n.inputs = std::move(inputs);
    n.output = out_id;
    n.attrs = std::move(attrs);
    nodes_.push_back(std::move(n));
    return out_id;
}

Result<ValueId> Graph::add_add(ValueId a, ValueId b) { return add_node(OpKind::Add, {a, b}); }
Result<ValueId> Graph::add_mul(ValueId a, ValueId b) { return add_node(OpKind::Mul, {a, b}); }

Result<ValueId> Graph::add_matmul(ValueId a, ValueId b, bool transpose_b) {
    Attrs at;
    at.transpose_b = transpose_b;
    return add_node(OpKind::MatMul, {a, b}, std::move(at));
}

Result<ValueId> Graph::add_gelu(ValueId x) { return add_node(OpKind::Gelu, {x}); }

Result<ValueId> Graph::add_softmax(ValueId x, int64_t axis) {
    Attrs at;
    at.axis = axis;
    return add_node(OpKind::Softmax, {x}, std::move(at));
}

Result<ValueId> Graph::add_layer_norm(ValueId x, ValueId gamma, ValueId beta, float eps) {
    Attrs at;
    at.eps = eps;
    return add_node(OpKind::LayerNorm, {x, gamma, beta}, std::move(at));
}

Result<ValueId> Graph::add_reshape(ValueId x, std::vector<int64_t> dims) {
    Attrs at;
    at.ints = std::move(dims);
    return add_node(OpKind::Reshape, {x}, std::move(at));
}

Result<ValueId> Graph::add_permute(ValueId x, std::vector<int64_t> perm) {
    Attrs at;
    at.ints = std::move(perm);
    return add_node(OpKind::Permute, {x}, std::move(at));
}

Result<ValueId> Graph::add_scale(ValueId x, float factor) {
    Attrs at;
    at.scalar = factor;
    return add_node(OpKind::Scale, {x}, std::move(at));
}

Result<ValueId> Graph::add_self_attention(ValueId packed_qkv, const SelfAttentionParams & params) {
    return add_node(OpKind::SelfAttention, {packed_qkv}, self_attention_attrs(params));
}

Result<ValueId> Graph::add_relu(ValueId x) { return add_node(OpKind::Relu, {x}); }
Result<ValueId> Graph::add_silu(ValueId x) { return add_node(OpKind::Silu, {x}); }
Result<ValueId> Graph::add_max_pool2d(ValueId x, const MaxPool2dParams & params) { return add_node(OpKind::MaxPool2d, {x}, max_pool2d_attrs(params)); }
Result<ValueId> Graph::add_upsample_nearest2d(ValueId x, const UpsampleParams & params) { return add_node(OpKind::UpsampleNearest2d, {x}, upsample_nearest2d_attrs(params)); }

Result<ValueId> Graph::add_conv_transpose2d(ValueId x, ValueId weight, const ConvTranspose2dParams & params) {
    return add_node(OpKind::ConvTranspose2d, {x, weight}, conv_transpose2d_attrs(params));
}

Result<ValueId> Graph::add_slice(ValueId x, int64_t axis, int64_t start, int64_t end) {
    Attrs at;
    at.axis = axis;
    at.ints = {start, end};
    return add_node(OpKind::Slice, {x}, std::move(at));
}

Result<ValueId> Graph::add_concat(std::vector<ValueId> inputs, int64_t axis) {
    Attrs at;
    at.axis = axis;
    return add_node(OpKind::Concat, std::move(inputs), std::move(at));
}

Result<ValueId> Graph::add_conv2d(ValueId x, ValueId weight, const Conv2dParams & params) {
    return add_node(OpKind::Conv2d, {x, weight}, conv2d_attrs(params));
}

Status Graph::mark_output(std::string name, ValueId value) {
    if (name.empty()) return invalid_argument("output name must not be empty");
    if (!has_value(value)) return invalid_argument("output '" + name + "': value id " + std::to_string(value) + " does not exist");
    for (const GraphOutput & o : outputs_) {
        if (o.name == name) return invalid_argument("duplicate output name '" + name + "'");
    }
    outputs_.push_back({std::move(name), value});
    return Status();
}

Result<ValueId> Graph::find_named(std::string_view name) const {
    for (size_t i = 0; i < values_.size(); i++) {
        if (values_[i].kind != ValueKind::Intermediate && values_[i].name == name) return static_cast<ValueId>(i);
    }
    return not_found("no input or weight named '" + std::string(name) + "'");
}

// Checks: names of inputs/weights are non-empty and unique; intermediates are unnamed;
// every node has the right arity, references only earlier existing values, writes an
// Intermediate that no other node writes, and its stored output type equals what
// infer_output computes; every Intermediate has a producer; outputs exist, are named, unique.
Status Graph::validate() const {
    std::set<std::string> names;
    for (size_t i = 0; i < values_.size(); i++) {
        const Value & v = values_[i];
        const std::string label = "value " + std::to_string(i);
        if (v.kind == ValueKind::Intermediate) {
            if (!v.name.empty()) return invalid_argument(label + ": intermediate values must be unnamed");
        } else {
            if (v.name.empty()) return invalid_argument(label + ": input/weight must be named");
            if (!names.insert(v.name).second) return invalid_argument(label + ": duplicate name '" + v.name + "'");
        }
    }

    std::vector<bool> produced(values_.size(), false);
    for (size_t i = 0; i < nodes_.size(); i++) {
        const Node & n = nodes_[i];
        const std::string label = node_label(i, n);
        if (!has_value(n.output)) return invalid_argument(label + ": output id " + std::to_string(n.output) + " does not exist");
        if (values_[n.output].kind != ValueKind::Intermediate) return invalid_argument(label + ": output must be an intermediate value");
        if (produced[n.output]) return invalid_argument(label + ": value " + std::to_string(n.output) + " already produced by another node");
        produced[n.output] = true;

        std::vector<TypeInfo> types;
        for (ValueId in : n.inputs) {
            if (!has_value(in)) return invalid_argument(label + ": input id " + std::to_string(in) + " does not exist");
            if (in >= n.output) return invalid_argument(label + ": input value " + std::to_string(in) + " is not defined before its use");
            types.push_back(values_[in].type);
        }
        auto expect = infer_output(n.op, types, n.attrs);
        if (!expect.is_ok()) return Status(expect.status().code(), label + ": " + expect.status().message());
        if (expect.value() != values_[n.output].type) {
            return invalid_argument(label + ": stored type " + type_str(values_[n.output].type) +
                                    " but inference gives " + type_str(expect.value()));
        }
    }
    for (size_t i = 0; i < values_.size(); i++) {
        if (values_[i].kind == ValueKind::Intermediate && !produced[i]) {
            return invalid_argument("value " + std::to_string(i) + ": intermediate has no producing node");
        }
    }

    if (outputs_.empty()) return invalid_argument("graph has no outputs");
    std::set<std::string> out_names;
    for (const GraphOutput & o : outputs_) {
        if (o.name.empty()) return invalid_argument("output name must not be empty");
        if (!has_value(o.value)) return invalid_argument("output '" + o.name + "' refers to missing value " + std::to_string(o.value));
        if (!out_names.insert(o.name).second) return invalid_argument("duplicate output name '" + o.name + "'");
    }
    return Status();
}

std::string Graph::dump() const {
    std::ostringstream os;
    std::vector<int> producer(values_.size(), -1);
    for (size_t i = 0; i < nodes_.size(); i++) producer[nodes_[i].output] = static_cast<int>(i);
    for (size_t i = 0; i < values_.size(); i++) {
        const Value & v = values_[i];
        os << "%" << i << " = ";
        if (v.kind == ValueKind::Intermediate) {
            const Node & n = nodes_[producer[i]];
            os << op_name(n.op) << "(";
            for (size_t k = 0; k < n.inputs.size(); k++) os << (k ? ", " : "") << "%" << n.inputs[k];
            os << ")";
            if (n.attrs.transpose_b) os << " transpose_b";
            if (n.op == OpKind::Softmax || n.op == OpKind::Slice || n.op == OpKind::Concat) os << " axis=" << n.attrs.axis;
            if (n.op == OpKind::LayerNorm) os << " eps=" << n.attrs.eps;
            if (n.op == OpKind::Scale) os << " scalar=" << n.attrs.scalar;
            if (n.op == OpKind::SelfAttention) os << " scale=" << n.attrs.scalar;
            if (!n.attrs.ints.empty()) {
                os << " [";
                for (size_t k = 0; k < n.attrs.ints.size(); k++) os << (k ? ", " : "") << n.attrs.ints[k];
                os << "]";
            }
        } else {
            os << (v.kind == ValueKind::Input ? "input" : "weight") << " \"" << v.name << "\"";
        }
        os << " : " << type_str(v.type) << "\n";
    }
    for (const GraphOutput & o : outputs_) os << "output \"" << o.name << "\" = %" << o.value << "\n";
    return os.str();
}

}  // namespace dooms
