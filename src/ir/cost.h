#pragma once
// Analytical cost of an IR node, used to turn measured times into achieved FLOP/s and bytes/s and to
// reason about what limits an operation (compute vs memory bandwidth).

#include "ir/graph.h"

namespace dooms {

struct NodeCost {
    double flops = 0;          // multiply-add counted as 2; only matmul and conv2d count, the rest is memory-bound
    double bytes_read = 0;     // each input element read once (optimistic lower bound on traffic)
    double bytes_written = 0;  // each output element written once
    double bytes() const { return bytes_read + bytes_written; }
};

// `node` must belong to `graph`. Reshape is a view and costs no traffic; permute, slice and concat
// are counted as one read and one write of their output (the lowering materializes them).
NodeCost node_cost(const Graph & graph, const Node & node);

}  // namespace dooms
