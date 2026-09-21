#include <simdjson.h>

#include <cassert>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include "metrics/metrics.h"

using namespace dlsim;

namespace {

bool near(double a, double b, double tol = 1e-9) { return std::fabs(a - b) <= tol; }

RequestRecord rec(double first_s, double end_s, int64_t isl, int64_t osl, std::vector<float> itl, bool warmup = false) {
  RequestRecord r;
  r.arrive_s = 0;
  r.admit_s = 0;
  r.first_token_s = first_s;
  r.end_s = end_s;
  r.isl = isl;
  r.osl = osl;
  r.itl_ms = std::move(itl);
  r.warmup = warmup;
  return r;
}

void test_manual() {
  assert(near(percentile({}, 90), 0));
  assert(near(percentile({5, 1, 3}, 50), 3));
  assert(near(percentile({5, 1, 3}, 90), 5));
  assert(near(percentile({5, 1, 3}, 0), 1));
  assert(near(percentile({1, 2, 3, 4, 5, 6, 7, 8, 9, 10}, 90), 9));
  SimResult s;
  s.profiled_s = 10;
  s.kv_pool_tokens = 12345;
  s.records.push_back(rec(1.0, 3.0, 1000, 5, {500, 500, 500, 500}));
  s.records.push_back(rec(0.5, 1.4, 2000, 10, std::vector<float>(9, 100.0f)));
  s.records.push_back(rec(0.1, 0.2, 9999, 3, {50, 50}, true));
  s.records.push_back(rec(2.0, 2.0, 500, 1, {}));
  s.time_share = {{"compute", 0.4}, {"membw", 0.3}, {"capacity", 0.1}, {"scaleup", 0.05}, {"scaleout", 0.0}};
  s.tier_hits["hbm"] = 777;
  s.unlimited = {"scaleout:absent"};
  MappingSpec m;
  m.decode.device = "dev";
  m.decode.workers = 2;
  m.decode.tp = 4;
  m.decode.moe_ep = 4;
  Metrics mt = compute_metrics(s, m);
  const auto& v = mt.values;
  assert(near(v.at("tput_per_gpu"), 3516.0 / 10 / 8));
  assert(near(v.at("input_tput_per_gpu"), 3500.0 / 10 / 8));
  assert(near(v.at("output_tput_per_gpu"), 16.0 / 10 / 8));
  assert(near(v.at("total_tput_tps"), 351.6));
  assert(near(v.at("mean_ttft"), 3.5 / 3));
  assert(near(v.at("median_ttft"), 1.0));
  assert(near(v.at("p75_ttft"), 2.0));
  assert(near(v.at("p90_ttft"), 2.0));
  assert(near(v.at("p99_ttft"), 2.0));
  assert(near(v.at("mean_tpot"), 0.3));
  assert(near(v.at("median_tpot"), 0.1));
  assert(near(v.at("p90_tpot"), 0.5));
  assert(near(v.at("mean_itl"), 0.3));  // per-request average ITL, as aiperf defines it
  assert(near(v.at("median_itl"), 0.1));
  assert(near(v.at("p90_itl"), 0.5));
  assert(near(v.at("p90_intvty"), 2.0));
  assert(near(v.at("median_intvty"), 10.0));
  assert(near(v.at("mean_e2el"), 6.4 / 3));
  assert(near(v.at("p90_e2el"), 3.0));
  assert(near(v.at("p90_e2e_norm_intvty"), 0.5));
  assert(near(v.at("p75_e2e_norm_intvty"), 0.5));
  assert(near(v.at("total_requests_completed"), 3));
  assert(near(v.at("total_prompt_tokens"), 3500));
  assert(near(v.at("total_generation_tokens"), 16));
  assert(near(v.at("kv_cache_pool_tokens"), 12345));
  assert(near(v.at("median_input_tokens"), 1000));
  assert(near(v.at("duration_seconds"), 10));
  RowMeta meta;
  meta.hardware = "gb300";
  meta.framework = "dlsim";
  const std::string j = to_inferencex_row_json(s, mt, meta, m, 4);
  assert(j.find("\"hardware\":\"gb300\"") != std::string::npos);
  assert(j.find("\"conc\":4") != std::string::npos);
  assert(j.find("\"num_decode_gpu\":8") != std::string::npos);
  assert(j.find("\"num_prefill_gpu\":0") != std::string::npos);
  assert(j.find("\"decode_ep\":4") != std::string::npos);
  assert(j.find("\"benchmark_type\":\"agentic_traces\"") != std::string::npos);
  assert(j.find("\"tput_per_gpu\":43.95") != std::string::npos);
  assert(j.find("\"p90_intvty\":2.0") != std::string::npos);
  assert(j.find("\"tier_hits\":{\"hbm\":777}") != std::string::npos);
  assert(j.find("\"unlimited\":[\"scaleout:absent\"]") != std::string::npos);
  assert(j.find("\"capacity\":0.1") != std::string::npos);
  simdjson::dom::parser parser;
  simdjson::dom::element doc = parser.parse(simdjson::padded_string(j));
  assert(double(doc["metrics"]["p90_e2e_norm_intvty"]) == 0.5);
  assert(int64_t(doc["dlsim"]["kv_pool_tokens"]) == 12345);
  std::printf("manual metrics ok\n%s\n", j.c_str());
}

void test_timeline() {
  const char* path = "traces/inferencex/request-timeline_440971_2026-09-20.json";
  simdjson::dom::parser parser;
  simdjson::dom::element doc;
  auto err = parser.load(path).get(doc);
  if (err) {
    std::printf("SKIP timeline check: cannot load %s (%s)\n", path, simdjson::error_message(err));
    return;
  }
  SimResult s;
  s.profiled_s = double(doc["durationS"]);
  int64_t warm = 0;
  for (simdjson::dom::element r : doc["requests"]) {
    std::string_view phase = r["phase"];
    if (phase != "profiling") { warm++; continue; }
    if (bool(r["cancelled"])) continue;
    const double ttft = double(r["ttftMs"]), tpot = double(r["tpotMs"]);
    const int64_t osl = int64_t(r["osl"]), isl = int64_t(r["isl"]);
    const double e2e = ttft + tpot * double(osl - 1);
    s.records.push_back(rec(ttft / 1e3, e2e / 1e3, isl, osl, std::vector<float>(size_t(std::max<int64_t>(0, osl - 1)), float(tpot))));
  }
  MappingSpec m;
  m.disaggregated = true;
  m.prefill.device = m.decode.device = "gb300";
  m.prefill.tp = 4;
  m.prefill.moe_ep = 4;
  m.decode.tp = 8;
  m.decode.moe_ep = 8;
  m.decode.workers = 4;
  Metrics mt = compute_metrics(s, m);
  const auto& v = mt.values;
  const double pub_p90 = 50.61349942743663, pub_p75 = 81.20213196300286;
  const double row_median_ttft = 0.756, row_p90_intvty = 171.52658662092625, row_p90_tpot = 0.00581;
  std::printf("timeline 440971: %zu profiling requests (%lld warmup rows)\n", s.records.size(), (long long)warm);
  std::printf("  p90_e2e_norm_intvty %.4f (published %.4f, diff %.2f%%)\n", v.at("p90_e2e_norm_intvty"), pub_p90,
              (v.at("p90_e2e_norm_intvty") / pub_p90 - 1) * 100);
  std::printf("  p75_e2e_norm_intvty %.4f (published %.4f, diff %.2f%%)\n", v.at("p75_e2e_norm_intvty"), pub_p75,
              (v.at("p75_e2e_norm_intvty") / pub_p75 - 1) * 100);
  std::printf("  median_ttft %.6f s (row %.3f)  p90_tpot %.5f s (row %.5f)  p90_intvty from per-request tpot %.3f (row %.3f)\n",
              v.at("median_ttft"), row_median_ttft, v.at("p90_tpot"), row_p90_tpot, 1.0 / v.at("p90_tpot"), row_p90_intvty);
  assert(near(v.at("p90_intvty"), 1.0 / v.at("p90_tpot")));
  assert(std::fabs(v.at("p90_intvty") / row_p90_intvty - 1) < 0.01);
  assert(std::fabs(v.at("p90_e2e_norm_intvty") / pub_p90 - 1) < 0.03);
  assert(std::fabs(v.at("p75_e2e_norm_intvty") / pub_p75 - 1) < 0.01);
  assert(std::fabs(v.at("median_ttft") - row_median_ttft) < 1e-3);
  assert(std::fabs(v.at("p90_tpot") / row_p90_tpot - 1) < 0.01);
}

}  // namespace

int main() {
  test_manual();
  test_timeline();
  std::puts("metrics_test ok");
  return 0;
}
