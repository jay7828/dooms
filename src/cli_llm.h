#pragma once
// `dooms-cli llm`: text prompt (or chat) -> answer, from any GGUF language model. Returns the process exit code: 0 ok, 1 failed, 2 usage error.
namespace dooms {
int run_llm_command(int argc, char ** argv);  // argv[0] is the sub-command name "llm"
}
