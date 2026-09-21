#pragma once
// DeviceSpec from an AISimulate systems/<system>.yaml: flops, memory[0] and kernel_family only.
#include <map>
#include <string>

#include "config/config.h"

namespace dlsim {

std::string kernel_family_for_system(const std::string& system);
DeviceSpec device_spec_from_system_yaml(const std::string& yaml_path, const std::string& system);
// Every <name>.yaml under systems_dir that has a gpu section, keyed by system name.
std::map<std::string, DeviceSpec> load_system_specs(const std::string& systems_dir);

}  // namespace dlsim
