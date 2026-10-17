// dooms-cli: thin client of the model registry.
//   dooms-cli list                      list models compiled into this build
//   dooms-cli load <model> <path>       load a model file and report success
//   dooms-cli transcribe --model DIR --audio F       speech to text (Whisper)
// Exit codes: 0 success, 1 operation failed, 2 usage error.
#include <cstdio>
#include <cstring>

#include "cli_transcribe.h"
#include "core/log.h"
#ifdef DOOMS_WITH_VLM
#include "cli_llm.h"
#include "cli_vlm.h"
#include "cli_vla.h"
#endif
#include "dooms/dooms.h"

namespace {

constexpr int kExitOk = 0, kExitFailed = 1, kExitUsage = 2;

int usage() {
    std::fprintf(stderr, "usage: dooms-cli list\n       dooms-cli load <model> <path>\n"
                 "       dooms-cli llm --model F.gguf --prompt TEXT [options]\n       dooms-cli vlm --image F --prompt TEXT --models DIR [options]\n       dooms-cli vla --models DIR --prompt TEXT --camera F [--camera F ...] --state v1,v2,... [options]\n"
                 "       dooms-cli transcribe --model DIR --audio FILE [options]\n");
    return kExitUsage;
}

}  // namespace

int main(int argc, char ** argv) {
    dooms::Status env = dooms::init_log_from_env();
    if (!env.is_ok()) std::fprintf(stderr, "warning: %s\n", env.to_string().c_str());

    dooms::Status reg = dooms::register_builtin_models();
    if (!reg.is_ok()) {
        std::fprintf(stderr, "error: %s\n", reg.to_string().c_str());
        return kExitFailed;
    }
    if (argc < 2) return usage();

    if (!std::strcmp(argv[1], "list")) {
        if (argc != 2) return usage();
        std::printf("models:\n");
        for (const std::string & id : dooms::Registry::get().ids()) std::printf("  %s\n", id.c_str());
        return kExitOk;
    }

    if (!std::strcmp(argv[1], "load")) {
        if (argc != 4) return usage();
        auto model = dooms::Registry::get().create(argv[2]);
        if (!model.is_ok()) {
            std::fprintf(stderr, "error: %s\n", model.status().to_string().c_str());
            return kExitFailed;
        }
        dooms::Status st = model.value()->load(argv[3]);
        if (!st.is_ok()) {
            std::fprintf(stderr, "error: %s\n", st.to_string().c_str());
            return kExitFailed;
        }
        std::printf("loaded %s from %s\n", argv[2], argv[3]);
        return kExitOk;
    }


    if (!std::strcmp(argv[1], "transcribe")) return dooms::run_transcribe_command(argc - 1, argv + 1);

#ifdef DOOMS_WITH_VLM
    if (!std::strcmp(argv[1], "llm")) return dooms::run_llm_command(argc - 1, argv + 1);
    if (!std::strcmp(argv[1], "vlm")) return dooms::run_vlm_command(argc - 1, argv + 1);
    if (!std::strcmp(argv[1], "vla")) return dooms::run_vla_command(argc - 1, argv + 1);
#endif

    return usage();
}
