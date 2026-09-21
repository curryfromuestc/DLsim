#pragma once
// State engine: virtual clock replay of AgentX. Design: state-engine.md.
#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "config/config.h"
#include "perfdata/op_latency.h"
#include "step/step.h"
#include "trace/trace.h"

namespace dlsim {

struct RequestRecord {
  int32_t trace = -1, request = -1, lane = -1;
  int32_t play = -1;                   // session-tree instance; lane's initial play is 0
  int32_t worker = -1;                 // decode worker (or aggregated worker)
  int32_t prefill_worker = -1;
  bool warmup = false;                 // before t*: builds cache state, excluded from metrics
  double arrive_s = 0, admit_s = 0, first_token_s = 0, end_s = 0;
  std::vector<float> itl_ms;           // inter-token latencies after the first token
  int64_t isl = 0, osl = 0;
  int64_t computed_prefill_tokens = 0; // after prefix hits
  std::map<std::string, int64_t> tier_hits;  // tier name -> tokens served from that tier
  double delivery_ms = 0;              // P/D KV delivery
  double ttft_ms() const { return (first_token_s - arrive_s) * 1e3; }
  double e2el_ms() const { return (end_s - arrive_s) * 1e3; }
};

struct SimResult {
  std::vector<RequestRecord> records;
  double profiled_s = 0;               // measured window length
  std::map<std::string, double> time_share;   // compute, membw, capacity, scaleup, scaleout
  std::map<std::string, int64_t> tier_hits;   // totals
  double bulk_overlap_fraction = 0;
  std::vector<std::string> unlimited;         // fabric resources not limiting this run
  int64_t kv_pool_tokens = 0;
  std::map<std::string, double> extra;        // free-form diagnostics
  std::string unsupported_note;               // first operator note of a pass whose latency source was Unsupported
};

struct DeviceSet {
  std::map<std::string, DeviceSpec> devices;  // by name
  const DeviceSpec& at(const std::string& n) const;
};

SimResult simulate(const trace::TraceSet& traces, const DeviceSet& devices, const FabricSpec& fabric,
                   const MappingSpec& mapping, const StackSpec& stack, const RunSpec& run,
                   const OpLatencySource& ops, const std::string& model = "deepseek-v4-pro");

using StepFactory = std::function<std::unique_ptr<StepLatency>(const PoolSpec&)>;
SimResult simulate_with(const trace::TraceSet& traces, const DeviceSet& devices, const FabricSpec& fabric,
                        const MappingSpec& mapping, const StackSpec& stack, const RunSpec& run,
                        const StepFactory& step_factory);

// Per-token costs of a model for the closed-form screening; a zero entry drops that term.
struct ScreenModel {
  double flops_per_token = 0;
  double kv_bytes_per_token = 0;
  double weight_bytes_per_gpu = 0;
  double comm_bytes_per_token = 0;
};

// First-level closed-form screening: sessions a device group can hold (architecture.md "两级精度").
double screen_sessions(const DeviceSpec& d, const FabricSpec& f, const PoolSpec& pool, const StackSpec& stack,
                       const trace::Stats& workload, const std::string& model = "deepseek-v4-pro");
double screen_sessions(const DeviceSpec& d, const FabricSpec& f, const PoolSpec& pool, const StackSpec& stack,
                       const trace::Stats& workload, const ScreenModel& model, double request_interval_s);

}  // namespace dlsim
