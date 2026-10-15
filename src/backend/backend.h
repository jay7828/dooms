#pragma once
// Backend interface. A backend compiles an IR Graph into an Executable that
// owns device memory and runs the graph. This header has no dependency on any
// concrete runtime (ggml today, native Metal later).

#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include "core/status.h"
#include "core/tensor.h"
#include "ir/graph.h"

namespace dooms {

// A host tensor bound to a graph input or output by name.
struct NamedTensor {
    std::string name;
    TensorView tensor;
};

// Time attributed to one IR node by Executable::profile().
struct NodeProfile {
    int node_index = 0;       // index into graph().nodes()
    OpKind op = OpKind::Add;
    double ms = 0;            // median over iterations of the summed time of the node's backend operations
    int backend_ops = 0;      // how many backend operations the node was lowered to (views included)
};

struct ProfileReport {
    std::vector<NodeProfile> nodes;   // one per IR node, in graph order
    double whole_graph_ms = 0;        // median wall time of the normal, unobserved run
    // Observing a node forces the backend to synchronize after it, which costs time even for a no-op.
    // This is the median observed time of a backend operation that does no work (a view such as a
    // reshape): a floor for the per-operation overhead included in every NodeProfile::ms.
    double observation_overhead_ms = 0;
    int iterations = 0;
};

// Not thread-safe: one Executable must not be used from several threads at once.
class Executable {
public:
    virtual ~Executable() = default;

    // Provide the value of a Weight. Type must match the graph; data must be contiguous.
    // Every weight must be set before run().
    virtual Status set_weight(std::string_view name, const TensorView & data) = 0;

    // Copy `inputs` in, compute, copy results into `outputs`. Both lists must cover
    // the graph's inputs / outputs exactly once, with matching shape and dtype and
    // contiguous memory. No allocation of device memory happens here.
    virtual Status run(const std::vector<NamedTensor> & inputs, const std::vector<NamedTensor> & outputs) = 0;

    // Runs the graph `iterations` times with per-node observation and reports where the time goes.
    // The node times are serialized (the backend synchronizes after each operation), so their sum is
    // usually larger than whole_graph_ms; use them for ranking and for achieved-throughput estimates,
    // not as an additive decomposition of the real run. Inputs are validated as in run(); no outputs are
    // produced (use run() for results). All weights must be set.
    virtual Result<ProfileReport> profile(const std::vector<NamedTensor> & inputs, int iterations) {
        (void) inputs; (void) iterations;
        return unsupported("this backend does not implement profiling");
    }

    virtual const Graph & graph() const = 0;
};

class Backend {
public:
    virtual ~Backend() = default;
    virtual const char * name() const = 0;
    // Why this backend cannot run the graph (Unsupported), or Ok.
    virtual Status supports(const Graph & graph) const = 0;
    virtual Result<std::unique_ptr<Executable>> compile(const Graph & graph) = 0;
};

// Validates run() bindings against a graph (shared by all backends). On success,
// input_ids[i] / output_ids[i] hold the graph value each binding refers to.
Status check_bindings(const Graph & graph, const std::vector<NamedTensor> & inputs,
                      const std::vector<NamedTensor> & outputs, std::vector<ValueId> * input_ids,
                      std::vector<ValueId> * output_ids);

// The input half of check_bindings(), for operations that need no output buffers (profile()).
Status check_input_bindings(const Graph & graph, const std::vector<NamedTensor> & inputs, std::vector<ValueId> * input_ids);

// Validates a weight tensor against the graph; returns the weight's value id.
Result<ValueId> check_weight(const Graph & graph, std::string_view name, const TensorView & data);

}  // namespace dooms
