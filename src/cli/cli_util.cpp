#include "cli/cli_util.h"

#include <simdjson.h>
#include <yaml-cpp/yaml.h>

#include <filesystem>
#include <fstream>
#include <stdexcept>

namespace dlsim::cli {

Args parse_args(int argc, char** argv) {
  Args a;
  if (argc >= 2) a.command = argv[1];
  for (int i = 2; i < argc; ++i) {
    std::string s = argv[i];
    if (s.rfind("--", 0) == 0) {
      std::string key = s.substr(2);
      auto eq = key.find('=');
      if (eq != std::string::npos) {
        a.options[key.substr(0, eq)] = key.substr(eq + 1);
      } else if (i + 1 < argc && std::string(argv[i + 1]).rfind("--", 0) != 0) {
        a.options[key] = argv[++i];
      } else {
        a.options[key] = "1";
      }
    } else {
      a.positional.push_back(s);
    }
  }
  return a;
}

void apply_stack_override(StackSpec& s, const std::string& key, const std::string& value) {
  auto b = [&] { return value == "true" || value == "1"; };
  if (key == "mtp_nextn") {
    // Keep the accepted-length mean of the current distribution when the draft length changes.
    double mean = 0;
    for (size_t k = 0; k < s.mtp_accept_dist.size(); ++k) mean += double(k) * s.mtp_accept_dist[k];
    s.mtp_nextn = std::stoi(value);
    if (s.mtp_nextn > 0) apply_stack_override(s, "mtp_accept_mean", std::to_string(std::min(mean, double(s.mtp_nextn))));
    else s.mtp_accept_dist.clear();
  }
  else if (key == "t_step_fixed_ms") s.t_step_fixed_ms = std::stod(value);
  else if (key == "request_overhead_ms") s.request_overhead_ms = std::stod(value);
  else if (key == "prefill_max_seqs") s.prefill_max_seqs = std::stoi(value);
  else if (key == "prefill_max_tokens") s.prefill_max_tokens = std::stoll(value);
  else if (key == "chunk_tokens") s.chunk_tokens = std::stoll(value);
  else if (key == "max_num_batched_tokens") s.max_num_batched_tokens = std::stoll(value);
  else if (key == "max_num_seqs") s.max_num_seqs = std::stoi(value);
  else if (key == "kv_offload") s.kv_offload = b();
  else if (key == "kv_delivery") s.kv_delivery = value;
  else if (key == "mix_prefill_decode") s.mix_prefill_decode = b();
  else if (key == "overlap_comm") s.overlap_comm = b();
  else if (key == "overlap_bulk") s.overlap_bulk = b();
  else if (key == "residency_limit_s") s.residency_limit_s = std::stod(value);
  else if (key == "prefix_cache") s.prefix_cache = b();
  else if (key == "wide_ep") s.wide_ep = b();
  else if (key == "mtp_accept_mean") {
    // Truncated geometric acceptance distribution with the requested mean accepted length.
    const int n = s.mtp_nextn;
    const double target = std::stod(value);
    if (n <= 0) return;
    double lo = 1e-6, hi = 1e6;
    std::vector<double> d(n + 1);
    for (int it = 0; it < 200; ++it) {
      double r = std::sqrt(lo * hi), z = 0, mean = 0;
      for (int k = 0; k <= n; ++k) { d[k] = std::pow(r, k); z += d[k]; }
      for (int k = 0; k <= n; ++k) { d[k] /= z; mean += k * d[k]; }
      if (mean < target) lo = r; else hi = r;
    }
    s.mtp_accept_dist = d;
  } else {
    throw std::runtime_error("unknown stack override '" + key + "'");
  }
}

static PoolSpec pool_from(const YAML::Node& n) {
  PoolSpec p;
  p.device = n["device"].as<std::string>();
  p.workers = n["workers"].as<int>();
  p.tp = n["tp"].as<int>();
  p.attention_dp = n["attention_dp"].as<int>();
  p.moe_tp = n["moe_tp"].as<int>();
  p.moe_ep = n["moe_ep"].as<int>();
  p.ep = p.moe_ep;
  return p;
}

BenchPoint load_point(const std::string& path, const std::map<std::string, std::string>& overrides) {
  YAML::Node n = YAML::LoadFile(path);
  BenchPoint p;
  p.path = path;
  p.id = n["id"].as<int>();
  p.hardware = n["hardware"].as<std::string>();
  p.framework = n["framework"].as<std::string>();
  p.precision = n["precision"].as<std::string>();
  p.spec_method = n["spec_method"].as<std::string>();
  p.offload_mode = n["offload_mode"].IsNull() ? "none" : n["offload_mode"].as<std::string>();
  p.is_multinode = n["is_multinode"].as<bool>();
  p.num_prefill_gpu = n["num_prefill_gpu"].IsNull() ? 0 : n["num_prefill_gpu"].as<int>();
  p.num_decode_gpu = n["num_decode_gpu"].as<int>();
  DeviceSpec d = load_device(n["device"].as<std::string>());
  p.devices.devices[d.name] = d;
  p.fabric = load_fabric(n["fabric"].as<std::string>());
  if (n["stack"].IsNull()) throw std::runtime_error(path + ": no stack config for framework " + p.framework);
  p.stack = load_stack(n["stack"].as<std::string>());
  if (n["stack_overrides"])   // per-point engine settings read from the InferenceX recipe (tools/reference/gen_points.py)
    for (const auto& kv : n["stack_overrides"]) apply_stack_override(p.stack, kv.first.as<std::string>(), kv.second.as<std::string>());
  p.mapping.disaggregated = n["mapping"]["disaggregated"].as<bool>();
  p.mapping.routing = n["mapping"]["routing"] ? n["mapping"]["routing"].as<std::string>() : "round_robin";
  if (p.mapping.routing != "round_robin" && p.mapping.routing != "kv_aware") throw std::runtime_error(path + ": unsupported routing");
  if (n["mapping"]["w_prefill"]) p.mapping.w_prefill = n["mapping"]["w_prefill"].as<double>();
  if (n["mapping"]["w_decode"]) p.mapping.w_decode = n["mapping"]["w_decode"].as<double>();
  if (n["mapping"]["w_active"]) p.mapping.w_active = n["mapping"]["w_active"].as<double>();
  p.mapping.decode = pool_from(n["mapping"]["decode"]);
  if (p.mapping.disaggregated) p.mapping.prefill = pool_from(n["mapping"]["prefill"]);
  p.run.concurrency = n["run"]["concurrency"].as<int>();
  p.run.duration_s = n["run"]["duration_s"].as<double>();
  p.run.seed = n["run"]["seed"].as<uint64_t>();
  if (p.spec_method == "none") p.stack.mtp_nextn = 0;
  if (p.offload_mode == "none" || p.offload_mode == "off") p.stack.kv_offload = false;
  for (const auto& [k, v] : overrides) apply_stack_override(p.stack, k, v);
  if (p.stack.mtp_nextn > 0 && (int)p.stack.mtp_accept_dist.size() != p.stack.mtp_nextn + 1)
    throw std::runtime_error(path + ": mtp_accept_dist size does not match mtp_nextn");
  return p;
}

std::map<std::string, DeviceSpec> load_device_specs(const std::string& dir) {
  std::map<std::string, DeviceSpec> out;
  for (const auto& e : std::filesystem::directory_iterator(dir)) {
    if (e.path().extension() != ".yaml") continue;
    DeviceSpec d = load_device(e.path().string());
    out[d.name] = std::move(d);
  }
  return out;
}

RowMeta row_meta(const BenchPoint& p) {
  RowMeta m;
  m.hardware = p.hardware;
  m.framework = p.framework;
  m.precision = p.precision;
  m.spec_method = p.spec_method;
  m.offload_mode = p.offload_mode;
  m.disagg = p.mapping.disaggregated;
  m.is_multinode = p.is_multinode;
  return m;
}

std::map<int, SnapshotRow> load_snapshot(const std::string& benchmarks_json, const std::string& derived_json) {
  std::map<int, SnapshotRow> out;
  simdjson::dom::parser parser;
  simdjson::dom::element doc = parser.load(benchmarks_json);
  for (simdjson::dom::element r : doc.get_array()) {
    std::string_view bt = r["benchmark_type"].get_string().value();
    if (bt != "agentic_traces") continue;
    int id = std::stoi(std::string(r["id"].get_string().value()));
    SnapshotRow row;
    row.hardware = std::string(r["hardware"].get_string().value());
    row.framework = std::string(r["framework"].get_string().value());
    for (auto [k, v] : r["metrics"].get_object()) {
      if (v.is_double()) row.metrics[std::string(k)] = v.get_double().value();
      else if (v.is_int64()) row.metrics[std::string(k)] = double(v.get_int64().value());
    }
    out[id] = row;
  }
  if (!derived_json.empty()) {
    simdjson::dom::parser p2;
    simdjson::dom::element d = p2.load(derived_json);
    for (auto [k, v] : d.get_object()) {
      int id = std::stoi(std::string(k));
      auto it = out.find(id);
      if (it == out.end()) continue;
      for (auto [mk, mv] : v.get_object()) {
        if (mv.is_double()) it->second.metrics[std::string(mk)] = mv.get_double().value();
      }
    }
  }
  return out;
}

std::string json_escape(const std::string& s) {
  std::string o;
  for (char c : s) {
    if (c == '"' || c == '\\') { o += '\\'; o += c; }
    else if (c == '\n') o += "\\n";
    else o += c;
  }
  return o;
}

void write_file(const std::string& path, const std::string& content) {
  std::ofstream f(path);
  if (!f) throw std::runtime_error("cannot write " + path);
  f << content;
}

}  // namespace dlsim::cli
