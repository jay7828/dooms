#pragma once

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

#include "core/status.h"
#include "ir/op.h"
#include "ir/shape_infer.h"

namespace dooms {

using ValueId = int32_t;
constexpr ValueId kInvalidValue = -1;

enum class ValueKind {
    Input,         // runtime-provided tensor, named
    Weight,        // constant parameter loaded from the model file, named
    Intermediate,  // produced by exactly one node
};

struct Value {
    ValueKind kind = ValueKind::Intermediate;
    TypeInfo type;
    std::string name;  // non-empty for Input and Weight, empty for Intermediate
};

struct Node {
    OpKind op = OpKind::Add;
    std::vector<ValueId> inputs;
    ValueId output = kInvalidValue;
    Attrs attrs;
};

struct GraphOutput {
    std::string name;
    ValueId value = kInvalidValue;
};

// Static-shape SSA graph. Values are numbered in creation order and a node's
// inputs always have smaller ids than its output, so the node list is already
// topologically sorted and cycles cannot exist.
//
// Graphs are built through the add_* methods (each type-checks via
// infer_output) or reconstructed by from_parts() (used by passes and loaders).
// Either way, a Graph handed out by this API satisfies validate().
class Graph {
public:
    Graph() = default;

    // Build a graph from raw parts; fails if the result is not valid.
    static Result<Graph> from_parts(std::vector<Value> values, std::vector<Node> nodes,
                                    std::vector<GraphOutput> outputs);

    Result<ValueId> add_input(std::string name, const Shape & shape, DType dtype);
    Result<ValueId> add_weight(std::string name, const Shape & shape, DType dtype);

    // Generic node builder; the typed helpers below call it.
    Result<ValueId> add_node(OpKind op, std::vector<ValueId> inputs, Attrs attrs = Attrs());

    Result<ValueId> add_add(ValueId a, ValueId b);
    Result<ValueId> add_mul(ValueId a, ValueId b);
    Result<ValueId> add_matmul(ValueId a, ValueId b, bool transpose_b = false);
    Result<ValueId> add_gelu(ValueId x);
    Result<ValueId> add_softmax(ValueId x, int64_t axis);
    Result<ValueId> add_layer_norm(ValueId x, ValueId gamma, ValueId beta, float eps);
    Result<ValueId> add_reshape(ValueId x, std::vector<int64_t> dims);
    Result<ValueId> add_permute(ValueId x, std::vector<int64_t> perm);
    Result<ValueId> add_scale(ValueId x, float factor);
    Result<ValueId> add_relu(ValueId x);
    Result<ValueId> add_silu(ValueId x);
    Result<ValueId> add_max_pool2d(ValueId x, const MaxPool2dParams & params);
    Result<ValueId> add_upsample_nearest2d(ValueId x, const UpsampleParams & params);
    Result<ValueId> add_conv_transpose2d(ValueId x, ValueId weight, const ConvTranspose2dParams & params);
    Result<ValueId> add_self_attention(ValueId packed_qkv, const SelfAttentionParams & params);
    Result<ValueId> add_slice(ValueId x, int64_t axis, int64_t start, int64_t end);
    Result<ValueId> add_concat(std::vector<ValueId> inputs, int64_t axis);
    Result<ValueId> add_conv2d(ValueId x, ValueId weight, const Conv2dParams & params);

    Status mark_output(std::string name, ValueId value);

    const std::vector<Value> & values() const { return values_; }
    const std::vector<Node> & nodes() const { return nodes_; }
    const std::vector<GraphOutput> & outputs() const { return outputs_; }
    Result<ValueId> find_named(std::string_view name) const;  // inputs and weights

    // Full structural and type check; see graph.cpp for the list.
    Status validate() const;
    // Human-readable listing, stable across runs (used in tests and debugging).
    std::string dump() const;

private:
    Result<ValueId> add_named(ValueKind kind, std::string name, const Shape & shape, DType dtype);
    bool has_value(ValueId id) const { return id >= 0 && static_cast<size_t>(id) < values_.size(); }

    std::vector<Value> values_;
    std::vector<Node> nodes_;
    std::vector<GraphOutput> outputs_;
};

}  // namespace dooms
