#include "pipeline/device.h"

#include <cstdlib>

namespace dooms {

Result<DeviceChoice> resolve_device(const std::string & requested) {
    std::string d = requested.empty() ? "auto" : requested;
    if (d == "auto") {
        if (const char * env = std::getenv("DOOMS_DEVICE"); env && *env) {
            d = env;
            if (d != "auto" && d != "gpu" && d != "cpu") return invalid_argument("DOOMS_DEVICE must be auto, gpu or cpu, got \"" + d + "\"");
        }
    }
    if (d == "auto") return DeviceChoice{true, true};
    if (d == "gpu") return DeviceChoice{true, false};
    if (d == "cpu") return DeviceChoice{false, false};
    return invalid_argument("device must be \"auto\", \"gpu\" or \"cpu\", got \"" + d + "\"");
}

}  // namespace dooms
