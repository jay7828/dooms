#pragma once
// `dooms-cli vla`: camera photos + task text + robot state -> a chunk of actions (SmolVLA). Returns the process exit code: 0 ok, 1 failed, 2 usage error.
namespace dooms {
int run_vla_command(int argc, char ** argv);  // argv[0] is the sub-command name "vla"
}
