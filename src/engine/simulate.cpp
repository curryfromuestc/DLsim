#include "engine/engine.h"

namespace dlsim {

SimResult simulate(const trace::TraceSet& traces, const DeviceSet& devices, const FabricSpec& fabric,
                   const MappingSpec& mapping, const StackSpec& stack, const RunSpec& run,
                   const OpLatencySource& ops, const std::string& model) {
  return simulate_with(traces, devices, fabric, mapping, stack, run, [&](const PoolSpec& p) {
    return make_step_latency(model, devices.at(p.device), p, stack, fabric, ops);
  });
}

}  // namespace dlsim
