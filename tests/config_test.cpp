#include <cassert>
#include <cstdio>
#include <fstream>
#include <stdexcept>

#include "config/config.h"
#include "fabric/fabric.h"

using namespace dlsim;

int main() {
  DeviceSpec d = load_device("configs/device/gb300.yaml");
  assert(d.name == "gb300");
  assert(d.flops.at("fp4") == 15000e12);
  assert(d.memory.size() == 2 && d.memory[0].name == "hbm" && d.memory[1].name == "host");

  FabricSpec f = load_fabric("configs/fabric/nvl72.yaml");
  assert(f.scaleup_domain && *f.scaleup_domain == 72);
  assert(f.scaleout.present && f.scaleout.bandwidth_Bps && *f.scaleout.bandwidth_Bps == 100e9);
  assert(unlimited_resources(f).size() == 1);  // remote absent

  StackSpec s = load_stack("configs/stack/dynamo-trt.yaml");
  assert(s.mtp_nextn == 3 && s.mtp_accept_dist.size() == 4);
  assert(!s.mix_prefill_decode && s.kv_offload);

  {
    std::ofstream bad("/tmp/dlsim_bad_stack.yaml");
    bad << "framework: trtllm\nno_such_switch: true\n";
  }
  bool threw = false;
  try { load_stack("/tmp/dlsim_bad_stack.yaml"); } catch (const std::runtime_error&) { threw = true; }
  assert(threw);

  {
    std::ofstream bad("/tmp/dlsim_bad_map.yaml");
    bad << "disaggregated: false\ndecode: {device: gb300, tp: 8, attention_dp: 1, moe_tp: 1, moe_ep: 4}\n";
  }
  threw = false;
  try { load_mapping("/tmp/dlsim_bad_map.yaml"); } catch (const std::runtime_error&) { threw = true; }
  assert(threw);

  std::puts("config_test ok");
  return 0;
}
