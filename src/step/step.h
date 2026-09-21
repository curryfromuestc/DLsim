#pragma once
// StepLatency: one forward iteration of one worker. Design: step-latency.md.
#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "config/config.h"
#include "core/types.h"
#include "perfdata/op_latency.h"

namespace dlsim {

enum class Phase : uint8_t { Prefill, Decode };

struct StepRequest {
  int64_t new_tokens = 0;     // prefill: tokens computed this iteration; decode: 1 (draft handled by nextn)
  int64_t past_kv = 0;        // cached context before this iteration
  bool prefill = true;
};

struct StepInput {
  // One entry per attention DP rank of the worker; each rank batches its own requests.
  std::vector<std::vector<StepRequest>> ranks;
  int nextn = 0;              // MTP draft length applied to decode requests
};

struct StepResult {
  double total_ms = 0;
  double compute_ms = 0, membw_ms = 0, comm_ms = 0, fixed_ms = 0;  // resource decomposition
  double lo_ms = 0, hi_ms = 0;
  Source source = Source::Measured;   // weakest source among the operators
  std::string note;
  std::map<std::string, double> per_op;   // operator name -> ms charged in this step (before overlap)
};

struct MemoryPlan {
  double weight_bytes_per_gpu = 0;
  double kv_bytes_per_token = 0;      // not split by TP for DeepSeek compressed KV
  int64_t kv_capacity_tokens(const MemoryTier& tier, const DeviceSpec& d) const;
};

class StepLatency {
 public:
  virtual ~StepLatency() = default;
  virtual StepResult step(const StepInput& in) const = 0;
  virtual const MemoryPlan& memory() const = 0;
};

// Model is selected by name ("deepseek-v4-pro"); model descriptions live in src/model.
std::unique_ptr<StepLatency> make_step_latency(const std::string& model, const DeviceSpec& dev, const PoolSpec& pool,
                                               const StackSpec& stack, const FabricSpec& fabric,
                                               const OpLatencySource& ops);

}  // namespace dlsim
