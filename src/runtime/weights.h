#pragma once
// Binds the weights of a compiled graph to tensors in a GGUF file.

#include <cstdint>
#include <string>
#include <vector>

#include "backend/backend.h"
#include "core/status.h"
#include "format/gguf.h"

namespace dooms {

struct WeightLoadReport {
    size_t loaded = 0;                              // graph weights set
    uint64_t bytes = 0;                             // total bytes copied to the executable
    std::vector<std::string> unused_file_tensors;   // in the file but not used by the graph, in file order
};

// Sets every Weight of `exe`'s graph from the tensor of the same name in `file`.
//   NotFound         a graph weight has no tensor in the file
//   InvalidArgument  shape or dtype of a file tensor differs from the graph's (no implicit
//                    conversion: a dtype change is an explicit decision, see docs/PLAN.md M6)
// On error the executable may be partially loaded and must not be run. Tensors that the graph does
// not use are not an error; they are listed in the report so the caller can decide.
Result<WeightLoadReport> load_weights(const GgufFile & file, Executable & exe);

// Same, with the weights spread over several files: each graph weight is taken from the first file that has a tensor of that name.
// `unused_file_tensors` then lists the tensors of every file that no weight used (file order, then tensor order).
//   InvalidArgument  a null file pointer or an empty list
//   NotFound         a graph weight is in none of the files
Result<WeightLoadReport> load_weights(const std::vector<const GgufFile *> & files, Executable & exe);

}  // namespace dooms
