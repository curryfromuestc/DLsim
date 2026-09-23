#include "metrics/metrics.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <string>

namespace dlsim {

double percentile(std::vector<double> v, double q) {
  if (v.empty()) return 0;
  std::sort(v.begin(), v.end());
  const size_t n = v.size();
  size_t k = size_t(std::ceil(q / 100.0 * double(n)));
  k = std::max<size_t>(1, std::min(k, n));
  return v[k - 1];
}

namespace {

double mean(const std::vector<double>& v) {
  if (v.empty()) return 0;
  double s = 0;
  for (double x : v) s += x;
  return s / double(v.size());
}

double stddev(const std::vector<double>& v) {
  if (v.empty()) return 0;
  const double m = mean(v);
  double s = 0;
  for (double x : v) s += (x - m) * (x - m);
  return std::sqrt(s / double(v.size()));
}

void stats(Metrics& m, const std::string& name, const std::vector<double>& v) {
  m.values["mean_" + name] = mean(v);
  m.values["median_" + name] = percentile(v, 50);
  m.values["p75_" + name] = percentile(v, 75);
  m.values["p90_" + name] = percentile(v, 90);
  m.values["p95_" + name] = percentile(v, 95);
  m.values["p99_" + name] = percentile(v, 99);
  m.values["std_" + name] = stddev(v);
}

bool completed(const RequestRecord& r) { return !r.warmup && !r.failed && (r.end_s > 0 || r.first_token_s > 0); }

std::string num(double x) {
  char buf[64];
  if (std::isfinite(x) && x == std::floor(x) && std::fabs(x) < 1e15) std::snprintf(buf, sizeof buf, "%.1f", x);
  else std::snprintf(buf, sizeof buf, "%.17g", x);
  return buf;
}

std::string str(const std::string& s) {
  std::string o = "\"";
  for (char c : s) {
    if (c == '"' || c == '\\') o += '\\';
    o += c;
  }
  return o + "\"";
}

}  // namespace

Metrics compute_metrics(const SimResult& r, const MappingSpec& m) {
  Metrics mt;
  std::vector<double> ttft, tpot, e2el, ratio, inv, isl, osl;
  int64_t in = 0, out = 0, n = 0, hit = 0, computed = 0;
  for (const RequestRecord& rec : r.records) {
    if (!completed(rec)) continue;
    n++;
    in += rec.isl;
    out += rec.osl;
    computed += rec.computed_prefill_tokens;
    for (const auto& th : rec.tier_hits) hit += th.second;
    const double t = rec.ttft_ms() / 1e3, e = rec.e2el_ms() / 1e3;
    ttft.push_back(t);
    e2el.push_back(e);
    isl.push_back(double(rec.isl));
    osl.push_back(double(rec.osl));
    if (rec.osl > 1) {
      // aiperf inter_token_latency: per-request (latency - ttft) / (osl - 1); percentiles run over requests.
      const double v = (e - t) / double(rec.osl - 1);
      tpot.push_back(v);
      inv.push_back(v > 0 ? 1.0 / v : 0.0);
    }
    if (rec.osl > 0) ratio.push_back(e / double(rec.osl));
  }
  const double gpus = double(m.disaggregated ? m.prefill.gpus() + m.decode.gpus() : m.decode.gpus());
  const double secs = r.profiled_s > 0 ? r.profiled_s : 1.0;
  auto& v = mt.values;
  v["tput_per_gpu"] = double(in + out) / secs / gpus;
  v["input_tput_per_gpu"] = double(in) / secs / gpus;
  v["output_tput_per_gpu"] = double(out) / secs / gpus;
  v["input_tput_tps"] = double(in) / secs;
  v["output_tput_tps"] = double(out) / secs;
  v["total_tput_tps"] = double(in + out) / secs;
  stats(mt, "ttft", ttft);
  stats(mt, "tpot", tpot);
  stats(mt, "e2el", e2el);
  stats(mt, "itl", tpot);
  stats(mt, "full_response_itl", tpot);
  for (const char* s : {"mean", "median", "p75", "p90", "p95", "p99"}) {
    const double x = v[std::string(s) + "_itl"];
    v[std::string(s) + "_intvty"] = x > 0 ? 1.0 / x : 0.0;
    v[std::string(s) + "_full_response_intvty"] = x > 0 ? 1.0 / x : 0.0;
  }
  v["std_intvty"] = stddev(inv);
  v["std_full_response_intvty"] = stddev(inv);
  const double r90 = percentile(ratio, 90), r75 = percentile(ratio, 75);
  v["p90_e2e_norm_intvty"] = r90 > 0 ? 1.0 / r90 : 0.0;
  v["p75_e2e_norm_intvty"] = r75 > 0 ? 1.0 / r75 : 0.0;
  stats(mt, "input_tokens", isl);
  stats(mt, "output_tokens_actual", osl);
  stats(mt, "output_tokens_expected", osl);
  v["total_requests_completed"] = double(n);
  v["total_prompt_tokens"] = double(in);
  v["total_generation_tokens"] = double(out);
  v["duration_seconds"] = r.profiled_s;
  v["kv_cache_pool_tokens"] = double(r.kv_pool_tokens);
  v["cache_hit_rate"] = in > 0 ? double(hit) / double(in) : 0.0;          // prefix tokens served from any tier
  v["computed_prefill_tokens"] = double(computed);
  return mt;
}

std::string to_inferencex_row_json(const SimResult& r, const Metrics& mt, const RowMeta& meta, const MappingSpec& m,
                                   int concurrency) {
  const PoolSpec& pf = m.disaggregated ? m.prefill : m.decode;
  const PoolSpec& dc = m.decode;
  std::string o = "{";
  o += "\"hardware\":" + str(meta.hardware) + ",\"framework\":" + str(meta.framework) + ",\"model\":" + str(meta.model);
  o += ",\"precision\":" + str(meta.precision) + ",\"spec_method\":" + str(meta.spec_method);
  o += std::string(",\"disagg\":") + (meta.disagg ? "true" : "false");
  o += std::string(",\"is_multinode\":") + (meta.is_multinode ? "true" : "false");
  o += ",\"prefill_tp\":" + std::to_string(pf.tp) + ",\"prefill_ep\":" + std::to_string(pf.moe_ep);
  o += std::string(",\"prefill_dp_attention\":") + (pf.attention_dp > 1 ? "true" : "false");
  o += ",\"prefill_num_workers\":" + std::to_string(pf.workers);
  o += ",\"decode_tp\":" + std::to_string(dc.tp) + ",\"decode_ep\":" + std::to_string(dc.moe_ep);
  o += std::string(",\"decode_dp_attention\":") + (dc.attention_dp > 1 ? "true" : "false");
  o += ",\"decode_num_workers\":" + std::to_string(dc.workers);
  o += ",\"num_prefill_gpu\":" + std::to_string(m.disaggregated ? pf.gpus() : 0);
  o += ",\"num_decode_gpu\":" + std::to_string(dc.gpus());
  o += ",\"benchmark_type\":\"agentic_traces\",\"offload_mode\":" + str(meta.offload_mode);
  o += ",\"isl\":null,\"osl\":null,\"conc\":" + std::to_string(concurrency);
  o += ",\"metrics\":{";
  bool first = true;
  for (const auto& kv : mt.values) {
    o += (first ? "" : ",") + str(kv.first) + ":" + num(kv.second);
    first = false;
  }
  o += ",\"decode_pp\":" + std::to_string(dc.pp) + ",\"prefill_pp\":" + std::to_string(pf.pp);
  o += ",\"offload_mode\":" + str(meta.offload_mode) + "}";
  o += ",\"dlsim\":{\"tier_hits\":{";
  first = true;
  for (const auto& kv : r.tier_hits) {
    o += (first ? "" : ",") + str(kv.first) + ":" + std::to_string(kv.second);
    first = false;
  }
  o += "},\"time_share\":{";
  first = true;
  for (const auto& kv : r.time_share) {
    o += (first ? "" : ",") + str(kv.first) + ":" + num(kv.second);
    first = false;
  }
  o += "},\"bulk_overlap_fraction\":" + num(r.bulk_overlap_fraction) + ",\"unlimited\":[";
  first = true;
  for (const auto& s : r.unlimited) {
    o += (first ? "" : ",") + str(s);
    first = false;
  }
  o += "],\"kv_pool_tokens\":" + std::to_string(r.kv_pool_tokens) + ",\"extra\":{";
  first = true;
  for (const auto& kv : r.extra) {
    o += (first ? "" : ",") + str(kv.first) + ":" + num(kv.second);
    first = false;
  }
  o += "}";
  if (!r.unsupported_note.empty()) o += ",\"unsupported_note\":" + str(r.unsupported_note);
  o += "}}";
  return o;
}

}  // namespace dlsim
