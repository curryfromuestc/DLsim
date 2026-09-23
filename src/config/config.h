#pragma once
// The four configuration objects (design: architecture.md "四类配置对象").
// Every field that can be omitted in YAML keeps a documented default here.
#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <vector>

namespace dlsim {

struct MemoryTier {
  std::string name;              // "hbm", "lpddr", "host", "remote"
  double capacity_bytes = 0;
  double bandwidth_Bps = 0;      // 0 = unlimited (only meaningful for the device tier)
  double usable_fraction = 0.9;  // share of capacity available to weights + KV
};

struct DeviceSpec {
  std::string name;                       // "gb300"
  std::map<std::string, double> flops;    // dtype -> FLOP/s, keys: fp32 bf16 fp8 fp4 int8
  std::vector<MemoryTier> memory;         // [0] is the device main memory; later entries are lower tiers
  double power_w = 0;
  double price_usd = 0;
  // Operator latency data: name of the measured table set (e.g. "gb300") or empty for a hypothetical device.
  std::string perf_tables;
  std::string framework;                  // default table selection, e.g. "trtllm"
  std::string version;                    // e.g. "1.3.0rc23"
  std::map<std::string, std::string> versions;   // table version per framework, e.g. {sglang: 0.5.14, vllm: 0.25.0}
  std::string kernel_family;              // e.g. "sm100"; decomposition transfers only within a family
  double activation_reserve_bytes = 0;    // stack reserve on the device tier
  double comm_reserve_bytes = 0;
};

struct Link {
  bool present = false;
  std::optional<double> alpha_s;          // per-message latency; nullopt = zero
  std::optional<double> bandwidth_Bps;    // per-device unidirectional; nullopt = unlimited
};

struct FabricSpec {
  std::optional<int> scaleup_domain;      // S; nullopt = all devices share one domain
  Link scaleup, scaleout, host, remote;
  double scaleout_oversub = 1.0;          // r
  double bulk_share = 0.0;                // share of link bandwidth taken by bulk transfers during steps
  double tail_latency_mult = 1.0;
  bool alltoall_serial_latency = false;   // false: one per-message latency per all-to-all (switched fabric)
  std::optional<int> devices_per_node;    // node granularity of measured comm tables
  std::optional<int> devices_per_rack;
};

struct PoolSpec {
  std::string device;                     // DeviceSpec::name
  int workers = 1;
  int tp = 1, ep = 1, attention_dp = 1, pp = 1, cp = 1;
  int moe_tp = 1, moe_ep = 1;
  int gpus_per_worker() const { return tp * attention_dp * cp * pp; }
  int gpus() const { return workers * gpus_per_worker(); }
};

struct MappingSpec {
  bool disaggregated = false;
  PoolSpec prefill;                       // used only when disaggregated
  PoolSpec decode;                        // aggregated deployments use this pool for both phases
  std::string routing = "round_robin";    // or "kv_aware"
  double w_prefill = 1, w_decode = 1, w_active = 1;
};

struct StackSpec {
  std::string framework;                  // "trtllm" "sglang" "vllm"
  int mtp_nextn = 0;                      // draft length; 0 = no MTP
  std::vector<double> mtp_accept_dist;    // P(accepted = k), k = 0..nextn; required when mtp_nextn > 0
  bool chunked_prefill = true;
  int64_t chunk_tokens = 16384;
  bool mix_prefill_decode = true;         // prefill and decode may share an iteration
  int prefill_interval = 1;               // decode iterations between prefill turns; 1 = every iteration
  int64_t max_num_batched_tokens = 16384;
  int max_num_seqs = 256;
  bool prefix_cache = true;
  std::string prefix_policy = "per_conversation";
  bool kv_offload = false;                // lower memory tiers are used for KV
  std::string kv_delivery = "full";       // P/D: "full" or "missing"
  bool graph_replay = true;
  double t_step_fixed_ms = 0;
  int prefill_max_seqs = 0;                // disaggregated prefill pool caps; 0 = same as max_num_seqs / max_num_batched_tokens
  int64_t prefill_max_tokens = 0;
  double request_overhead_ms = 0;         // per-request pipeline latency before prefill admission (router, disagg hand-off); calibrated
  bool overlap_comm = false;
  bool overlap_bulk = false;
  std::string eviction = "lru";
  double residency_limit_s = 0;           // 0 = none
  std::string moe_distribution = "power_law_1.01";
  bool wide_ep = false;
  std::string preemption = "recompute";
};

struct RunSpec {
  int concurrency = 1;
  double duration_s = 1800;
  double idle_shift_s = 10;               // --system-idle-gap-cap-seconds
  double t_star_min = 0.25, t_star_max = 0.75;   // --trajectory-start-{min,max}-ratio
  int warmup_requests_per_lane = 10;      // --warmup-requests-per-lane: turns replayed without delays after the primers
  double trace_idle_gap_cap_s = 300;      // --trace-idle-gap-cap-seconds; 0 = none
  uint64_t seed = 42;              // harness --random-seed; drives the per-(trace, lane) t* draw
  std::string trace_path;
  std::string cache_dir;
};

// YAML loaders. Unknown keys and unsupported switches raise std::runtime_error (design: scope.md).
DeviceSpec load_device(const std::string& path);
FabricSpec load_fabric(const std::string& path);
MappingSpec load_mapping(const std::string& path);
StackSpec load_stack(const std::string& path);
RunSpec load_run(const std::string& path);

}  // namespace dlsim
