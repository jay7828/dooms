#pragma once
// `dooms-cli vlm`: image + question -> answer (SmolVLM). Returns the process exit code: 0 ok, 1 failed, 2 usage error.
namespace dooms {
int run_vlm_command(int argc, char ** argv);  // argv[0] is the sub-command name "vlm"
}
