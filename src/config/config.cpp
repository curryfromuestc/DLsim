#include "config/config.h"

#include <yaml-cpp/yaml.h>

#include <set>
#include <stdexcept>

namespace dlsim {
namespace {

void check_keys(const YAML::Node& n, const std::set<std::string>& allowed, const std::string& where) {
  if (!n.IsMap()) throw std::runtime_error(where + ": expected a mapping");
  for (const auto& kv : n) {
    const auto k = kv.first.as<std::string>();
    if (!allowed.count(k)) throw std::runtime_error(where + ": unsupported key '" + k + "'");
  }
}

template <class T>
T get(const YAML::Node& n, const char* key, T def) {
  return n[key] ? n[key].as<T>() : def;
}

std::optional<double> opt_double(const YAML::Node& n, const char* key) {
  if (!n[key] || n[key].IsNull()) return std::nullopt;
  return n[key].as<double>();
}

Link parse_link(const YAML::Node& n, const std::string& where) {
  Link l;
  if (!n || n.IsNull()) return l;
  check_keys(n, {"alpha_s", "bandwidth_Bps"}, where);
  l.present = true;
  l.alpha_s = opt_double(n, "alpha_s");
  l.bandwidth_Bps = opt_double(n, "bandwidth_Bps");
  return l;
}

PoolSpec parse_pool(const YAML::Node& n, const std::string& where) {
  PoolSpec p;
  if (!n) return p;
  check_keys(n, {"device", "workers", "tp", "ep", "attention_dp", "pp", "cp", "moe_tp", "moe_ep"}, where);
  p.device = get<std::string>(n, "device", "");
  p.workers = get(n, "workers", 1);
  p.tp = get(n, "tp", 1);
  p.ep = get(n, "ep", 1);
  p.attention_dp = get(n, "attention_dp", 1);
  p.pp = get(n, "pp", 1);
  p.cp = get(n, "cp", 1);
  p.moe_tp = get(n, "moe_tp", 1);
  p.moe_ep = get(n, "moe_ep", p.ep);
  if (p.tp * p.attention_dp * p.cp != p.moe_tp * p.moe_ep)
    throw std::runtime_error(where + ": tp*attention_dp*cp must equal moe_tp*moe_ep");
  return p;
}

}  // namespace

DeviceSpec load_device(const std::string& path) {
  YAML::Node n = YAML::LoadFile(path);
  check_keys(n, {"name", "flops", "memory", "power_w", "price_usd", "perf_tables", "framework", "version", "versions",
                 "kernel_family", "activation_reserve_bytes", "comm_reserve_bytes"}, path);
  DeviceSpec d;
  d.name = n["name"].as<std::string>();
  for (const auto& kv : n["flops"]) d.flops[kv.first.as<std::string>()] = kv.second.as<double>();
  for (const auto& t : n["memory"]) {
    check_keys(t, {"name", "capacity_bytes", "bandwidth_Bps", "usable_fraction"}, path + ": memory");
    MemoryTier m;
    m.name = t["name"].as<std::string>();
    m.capacity_bytes = t["capacity_bytes"].as<double>();
    m.bandwidth_Bps = get(t, "bandwidth_Bps", 0.0);
    m.usable_fraction = get(t, "usable_fraction", 0.9);
    d.memory.push_back(m);
  }
  if (d.memory.empty()) throw std::runtime_error(path + ": memory must list at least the device tier");
  d.power_w = get(n, "power_w", 0.0);
  d.price_usd = get(n, "price_usd", 0.0);
  d.perf_tables = get<std::string>(n, "perf_tables", "");
  d.framework = get<std::string>(n, "framework", "");
  d.version = get<std::string>(n, "version", "");
  if (n["versions"])
    for (const auto& kv : n["versions"]) d.versions[kv.first.as<std::string>()] = kv.second.as<std::string>();
  if (!d.framework.empty() && !d.version.empty()) d.versions.emplace(d.framework, d.version);
  d.kernel_family = get<std::string>(n, "kernel_family", "");
  d.activation_reserve_bytes = get(n, "activation_reserve_bytes", 0.0);
  d.comm_reserve_bytes = get(n, "comm_reserve_bytes", 0.0);
  return d;
}

FabricSpec load_fabric(const std::string& path) {
  YAML::Node n = YAML::LoadFile(path);
  check_keys(n, {"scaleup_domain", "scaleup", "scaleout", "host", "remote", "scaleout_oversub", "bulk_share",
                 "tail_latency_mult", "alltoall_serial_latency", "devices_per_node", "devices_per_rack"}, path);
  FabricSpec f;
  if (n["scaleup_domain"] && !n["scaleup_domain"].IsNull()) f.scaleup_domain = n["scaleup_domain"].as<int>();
  f.scaleup = parse_link(n["scaleup"], path + ": scaleup");
  f.scaleout = parse_link(n["scaleout"], path + ": scaleout");
  f.host = parse_link(n["host"], path + ": host");
  f.remote = parse_link(n["remote"], path + ": remote");
  f.scaleout_oversub = get(n, "scaleout_oversub", 1.0);
  f.bulk_share = get(n, "bulk_share", 0.0);
  f.tail_latency_mult = get(n, "tail_latency_mult", 1.0);
  f.alltoall_serial_latency = get(n, "alltoall_serial_latency", false);
  if (n["devices_per_node"]) f.devices_per_node = n["devices_per_node"].as<int>();
  if (n["devices_per_rack"]) f.devices_per_rack = n["devices_per_rack"].as<int>();
  return f;
}

MappingSpec load_mapping(const std::string& path) {
  YAML::Node n = YAML::LoadFile(path);
  check_keys(n, {"disaggregated", "prefill", "decode", "routing", "w_prefill", "w_decode", "w_active"}, path);
  MappingSpec m;
  m.disaggregated = get(n, "disaggregated", false);
  m.decode = parse_pool(n["decode"], path + ": decode");
  if (m.disaggregated) {
    if (!n["prefill"]) throw std::runtime_error(path + ": disaggregated mapping needs a prefill pool");
    m.prefill = parse_pool(n["prefill"], path + ": prefill");
  } else if (n["prefill"]) {
    throw std::runtime_error(path + ": prefill pool given but disaggregated is false");
  }
  m.routing = get<std::string>(n, "routing", "round_robin");
  if (m.routing != "round_robin" && m.routing != "kv_aware")
    throw std::runtime_error(path + ": unsupported routing '" + m.routing + "'");
  m.w_prefill = get(n, "w_prefill", 1.0);
  m.w_decode = get(n, "w_decode", 1.0);
  m.w_active = get(n, "w_active", 1.0);
  return m;
}

StackSpec load_stack(const std::string& path) {
  YAML::Node n = YAML::LoadFile(path);
  check_keys(n, {"framework", "mtp_nextn", "mtp_accept_dist", "chunked_prefill", "chunk_tokens", "mix_prefill_decode",
                 "max_num_batched_tokens", "max_num_seqs", "prefix_cache", "prefix_policy", "kv_offload",
                 "kv_delivery", "graph_replay", "t_step_fixed_ms", "request_overhead_ms", "prefill_max_seqs", "prefill_max_tokens", "overlap_comm", "overlap_bulk", "eviction",
                 "residency_limit_s", "moe_distribution", "wide_ep", "preemption", "prefill_interval"}, path);
  StackSpec s;
  s.framework = get<std::string>(n, "framework", "");
  s.mtp_nextn = get(n, "mtp_nextn", 0);
  if (n["mtp_accept_dist"]) s.mtp_accept_dist = n["mtp_accept_dist"].as<std::vector<double>>();
  if (s.mtp_nextn > 0 && (int)s.mtp_accept_dist.size() != s.mtp_nextn + 1)
    throw std::runtime_error(path + ": mtp_accept_dist needs mtp_nextn + 1 entries");
  s.chunked_prefill = get(n, "chunked_prefill", true);
  s.chunk_tokens = get<int64_t>(n, "chunk_tokens", 16384);
  s.mix_prefill_decode = get(n, "mix_prefill_decode", true);
  s.prefill_interval = get(n, "prefill_interval", 1);
  s.max_num_batched_tokens = get<int64_t>(n, "max_num_batched_tokens", 16384);
  s.max_num_seqs = get(n, "max_num_seqs", 256);
  s.prefix_cache = get(n, "prefix_cache", true);
  s.prefix_policy = get<std::string>(n, "prefix_policy", "per_conversation");
  s.kv_offload = get(n, "kv_offload", false);
  s.kv_delivery = get<std::string>(n, "kv_delivery", "full");
  if (s.kv_delivery != "full" && s.kv_delivery != "missing")
    throw std::runtime_error(path + ": unsupported kv_delivery '" + s.kv_delivery + "'");
  s.graph_replay = get(n, "graph_replay", true);
  s.t_step_fixed_ms = get(n, "t_step_fixed_ms", 0.0);
  s.request_overhead_ms = get(n, "request_overhead_ms", 0.0);
  s.prefill_max_seqs = get<int>(n, "prefill_max_seqs", 0);
  s.prefill_max_tokens = get<int64_t>(n, "prefill_max_tokens", 0);
  s.overlap_comm = get(n, "overlap_comm", false);
  s.overlap_bulk = get(n, "overlap_bulk", false);
  s.eviction = get<std::string>(n, "eviction", "lru");
  if (s.eviction != "lru") throw std::runtime_error(path + ": unsupported eviction '" + s.eviction + "'");
  s.residency_limit_s = get(n, "residency_limit_s", 0.0);
  s.moe_distribution = get<std::string>(n, "moe_distribution", "power_law_1.01");
  s.wide_ep = get(n, "wide_ep", false);
  s.preemption = get<std::string>(n, "preemption", "recompute");
  if (s.preemption != "recompute") throw std::runtime_error(path + ": unsupported preemption '" + s.preemption + "'");
  return s;
}

RunSpec load_run(const std::string& path) {
  YAML::Node n = YAML::LoadFile(path);
  check_keys(n, {"concurrency", "duration_s", "idle_shift_s", "t_star_min", "t_star_max", "warmup_requests_per_lane",
                 "trace_idle_gap_cap_s", "seed", "trace_path", "cache_dir"}, path);
  RunSpec r;
  r.concurrency = get(n, "concurrency", 1);
  r.duration_s = get(n, "duration_s", 1800.0);
  if (r.duration_s < 900) throw std::runtime_error(path + ": duration_s below the 900 s minimum");
  r.idle_shift_s = get(n, "idle_shift_s", 10.0);
  r.t_star_min = get(n, "t_star_min", 0.25);
  r.t_star_max = get(n, "t_star_max", 0.75);
  r.warmup_requests_per_lane = get(n, "warmup_requests_per_lane", 10);
  r.trace_idle_gap_cap_s = get(n, "trace_idle_gap_cap_s", 300.0);
  r.seed = get<uint64_t>(n, "seed", 42);
  r.trace_path = get<std::string>(n, "trace_path", "");
  r.cache_dir = get<std::string>(n, "cache_dir", "");
  return r;
}

}  // namespace dlsim
