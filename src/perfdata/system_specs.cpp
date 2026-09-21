#include "perfdata/system_specs.h"

#include <yaml-cpp/yaml.h>

#include <filesystem>

namespace dlsim {

std::string kernel_family_for_system(const std::string& s) {
  const auto starts = [&](const char* p) { return s.rfind(p, 0) == 0; };
  if (starts("a100") || starts("a30")) return "sm80";
  if (starts("h100") || starts("h200")) return "sm90";
  if (starts("b200") || starts("b300") || starts("gb200") || starts("gb300")) return "sm100";
  if (starts("rtx_pro_6000")) return "sm120";
  if (starts("l40s") || starts("l4")) return "sm89";
  if (starts("b60")) return "xe";
  return "";
}

DeviceSpec device_spec_from_system_yaml(const std::string& path, const std::string& system) {
  YAML::Node n = YAML::LoadFile(path);
  YAML::Node gpu = n["gpu"];
  if (!gpu) throw std::runtime_error(path + ": no gpu section");
  DeviceSpec d;
  d.name = system;
  d.perf_tables = system;
  d.kernel_family = kernel_family_for_system(system);
  const auto take = [&](const char* yaml_key, const char* flops_key) {
    if (gpu[yaml_key]) d.flops[flops_key] = gpu[yaml_key].as<double>();
  };
  take("bfloat16_tc_flops", "bf16");
  take("fp8_tc_flops", "fp8");
  take("fp4_tc_flops", "fp4");
  take("int8_tc_flops", "int8");
  take("fp32_flops", "fp32");
  take("float32_tc_flops", "fp32");
  MemoryTier m;
  m.name = "hbm";
  m.capacity_bytes = gpu["mem_capacity"] ? gpu["mem_capacity"].as<double>() : 0;
  m.bandwidth_Bps = gpu["mem_bw"] ? gpu["mem_bw"].as<double>() : 0;
  d.memory.push_back(m);
  return d;
}

std::map<std::string, DeviceSpec> load_system_specs(const std::string& systems_dir) {
  std::map<std::string, DeviceSpec> out;
  for (const auto& f : std::filesystem::directory_iterator(systems_dir)) {
    if (!f.is_regular_file() || f.path().extension() != ".yaml") continue;
    YAML::Node n = YAML::LoadFile(f.path());
    if (!n.IsMap() || !n["gpu"]) continue;
    const std::string name = f.path().stem();
    out[name] = device_spec_from_system_yaml(f.path(), name);
  }
  return out;
}

}  // namespace dlsim
