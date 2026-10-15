#include "runtime/weights.h"

#include <set>

namespace dooms {

Result<WeightLoadReport> load_weights(const std::vector<const GgufFile *> & files, Executable & exe) {
    if (files.empty()) return invalid_argument("no weight files given");
    for (const GgufFile * f : files) if (!f) return invalid_argument("null weight file");
    WeightLoadReport report;
    std::set<std::pair<size_t, std::string>> used;  // (file index, tensor name)
    for (const Value & v : exe.graph().values()) {
        if (v.kind != ValueKind::Weight) continue;
        size_t found = files.size();
        Status last = not_found("no files");
        Result<TensorView> view = not_found("graph weight '" + v.name + "' is not in any file");
        for (size_t i = 0; i < files.size(); i++) {
            auto candidate = files[i]->tensor_view(v.name);
            if (candidate.is_ok()) { view = candidate; found = i; break; }
            last = candidate.status();
        }
        if (found == files.size()) {
            return not_found("graph weight '" + v.name + "' is not in " + (files.size() == 1 ? "the file" : "any of the " + std::to_string(files.size()) + " files") + " (" + last.message() + ")");
        }
        DOOMS_RETURN_IF_ERROR(exe.set_weight(v.name, view.value()));
        used.insert({found, v.name});
        report.loaded++;
        report.bytes += static_cast<uint64_t>(view.value().span_bytes());
    }
    for (size_t i = 0; i < files.size(); i++) {
        for (const GgufTensorInfo & t : files[i]->tensors()) {
            if (!used.count({i, t.name})) report.unused_file_tensors.push_back(t.name);
        }
    }
    return report;
}

Result<WeightLoadReport> load_weights(const GgufFile & file, Executable & exe) {
    return load_weights(std::vector<const GgufFile *>{&file}, exe);
}

}  // namespace dooms
