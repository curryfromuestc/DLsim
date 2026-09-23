#include <algorithm>
#include <chrono>
#include <deque>
#include <future>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <sstream>
#include <stdexcept>

#include <yaml-cpp/yaml.h>

#include "cli/cli_util.h"
#include "engine/engine.h"
#include "fabric/calibrate.h"
#include "fabric/collectivex.h"
#include "metrics/metrics.h"
#include "perfdata/measured.h"
#include "solver/solver.h"
#include "trace/trace.h"
#include "trace/trace_dump.h"

namespace fs = std::filesystem;
using namespace dlsim;
using namespace dlsim::cli;

namespace {

std::map<std::string, std::string> overrides_of(const Args& a) {
  std::map<std::string, std::string> o;
  std::string s = a.get("stack-override");
  std::stringstream ss(s);
  std::string item;
  while (std::getline(ss, item, ',')) {
    auto eq = item.find('=');
    if (eq == std::string::npos) continue;
    o[item.substr(0, eq)] = item.substr(eq + 1);
  }
  return o;
}

trace::TraceSet load_traces(const Args& a) {
  std::string tr = a.get("trace", "traces/agentx/cc-traces-weka-062126/traces.jsonl");
  std::string cache = a.get("cache", "traces/cache");
  return trace::load(tr, cache);
}

struct SimOut {
  SimResult result;
  Metrics metrics;
  double wall_s = 0;
};

SimOut run_point(const BenchPoint& p, const trace::TraceSet& traces, const OpLatencySource& ops) {
  auto t0 = std::chrono::steady_clock::now();
  SimOut o;
  o.result = simulate(traces, p.devices, p.fabric, p.mapping, p.stack, p.run, ops);
  o.metrics = compute_metrics(o.result, p.mapping);
  o.wall_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  return o;
}

double rel_err(double pred, double meas) { return meas != 0 ? (pred - meas) / meas : NAN; }
double median(std::vector<double> v) {
  v.erase(std::remove_if(v.begin(), v.end(), [](double x) { return std::isnan(x); }), v.end());
  if (v.empty()) return NAN;
  std::sort(v.begin(), v.end());
  return v.size() % 2 ? v[v.size() / 2] : 0.5 * (v[v.size() / 2 - 1] + v[v.size() / 2]);
}

const std::vector<std::string> kCompareMetrics = {"tput_per_gpu", "p90_e2e_norm_intvty", "p90_intvty", "median_ttft"};

std::vector<BenchPoint> select_points(const Args& a) {
  std::vector<BenchPoint> pts;
  auto ov = overrides_of(a);
  std::string dir = a.get("points-dir", "configs/points");
  std::string hw = a.get("hardware"), fw = a.get("framework");
  std::set<int> ids;
  {
    std::stringstream ss(a.get("ids"));
    std::string t;
    while (std::getline(ss, t, ',')) if (!t.empty()) ids.insert(std::stoi(t));
  }
  std::vector<std::string> files;
  for (auto& e : fs::directory_iterator(dir)) if (e.path().extension() == ".yaml") files.push_back(e.path().string());
  std::sort(files.begin(), files.end());
  for (const auto& f : files) {
    BenchPoint p = load_point(f, ov);
    if (!hw.empty() && p.hardware != hw) continue;
    if (!fw.empty() && p.framework != fw) continue;
    if (!ids.empty() && !ids.count(p.id)) continue;
    pts.push_back(std::move(p));
  }
  return pts;
}

std::string metrics_json(const Metrics& m) {
  std::string s = "{";
  bool first = true;
  for (const auto& [k, v] : m.values) {
    if (!first) s += ",";
    first = false;
    s += "\"" + k + "\":" + (std::isfinite(v) ? std::to_string(v) : "null");
  }
  return s + "}";
}

int cmd_trace_stats(const Args& a) {
  auto ts = load_traces(a);
  trace::Stats st = trace::compute_stats(ts);
  std::printf("{\"traces\":%lld,\"main_turns\":%lld,\"subagent_groups\":%lld,\"subagent_inner_requests\":%lld,"
              "\"total_model_requests\":%lld,\"total_input_tokens\":%lld,\"total_output_tokens\":%lld,"
              "\"ideal_prefix_hit_rate\":%.6f}\n",
              (long long)st.traces, (long long)st.main_turns, (long long)st.subagent_groups,
              (long long)st.subagent_inner_requests, (long long)st.total_requests, (long long)st.total_input_tokens,
              (long long)st.total_output_tokens, st.ideal_prefix_hit_rate);
  return 0;
}

int cmd_trace_graph(const Args& a) {
  auto ts = load_traces(a);
  size_t i = std::stoul(a.get("index", "0"));
  if (i >= ts.traces.size()) throw std::runtime_error("index out of range");
  std::printf("%s\n", trace::dump_graph_json(ts.traces[i]).c_str());
  return 0;
}

// Prints every operator query and its answer; used by `step --verbose`.
struct LoggingSource : OpLatencySource {
  const OpLatencySource& inner;
  explicit LoggingSource(const OpLatencySource& i) : inner(i) {}
  Latency query(const std::string& device, const OpQuery& q) const override {
    Latency l = inner.query(device, q);
    std::string keys;
    for (const auto& [k, v] : q.num) keys += " " + k + "=" + std::to_string(v);
    for (const auto& [k, v] : q.cat) keys += " " + k + "=" + v;
    std::fprintf(stderr, "%s%s -> %.5f ms [%s] %s\n", q.op.c_str(), keys.c_str(), l.ms, to_string(l.source), l.note.c_str());
    return l;
  }
};

std::map<std::string, std::string> kv_list(const std::string& s) {
  std::map<std::string, std::string> o;
  std::stringstream ss(s);
  std::string item;
  while (std::getline(ss, item, ',')) {
    auto eq = item.find('=');
    if (eq != std::string::npos) o[item.substr(0, eq)] = item.substr(eq + 1);
  }
  return o;
}

int cmd_op(const Args& a) {
  MeasuredTables ops(a.get("perfdata", "perfdata"), load_device_specs(a.get("devices-dir", "configs/device")));
  OpQuery q;
  q.op = a.get("op");
  for (const auto& [k, v] : kv_list(a.get("num"))) q.num[k] = std::stod(v);
  q.cat = kv_list(a.get("cat"));
  const Latency l = ops.query(a.get("device"), q);
  std::printf("%.6f ms [%.6f, %.6f] source %s\nnote %s\n", l.ms, l.lo_ms, l.hi_ms, to_string(l.source), l.note.c_str());
  return 0;
}

int cmd_step(const Args& a) {
  BenchPoint p = load_point(a.get("point"), overrides_of(a));
  MeasuredTables ops(a.get("perfdata", "perfdata"), load_device_specs(a.get("devices-dir", "configs/device")));
  const bool prefill_pool = a.has("prefill");
  const PoolSpec& pool = prefill_pool && p.mapping.disaggregated ? p.mapping.prefill : p.mapping.decode;
  LoggingSource logged(ops);
  auto step = make_step_latency("deepseek-v4-pro", p.devices.at(pool.device), pool, p.stack, p.fabric,
                                a.has("verbose") ? static_cast<const OpLatencySource&>(logged) : ops);
  StepInput in;
  in.nextn = p.stack.mtp_nextn;
  in.ranks.resize(pool.attention_dp);
  const int64_t past = std::stoll(a.get("past", "0"));
  if (prefill_pool) {
    in.ranks[0].push_back(StepRequest{std::stoll(a.get("prefill")), past, true});
  } else {
    const int n = std::stoi(a.get("decode", "1"));
    for (int i = 0; i < n; ++i) in.ranks[i % pool.attention_dp].push_back(StepRequest{1, past, false});
  }
  const StepResult r = step->step(in);
  std::printf("total_ms %.4f compute_ms %.4f membw_ms %.4f comm_ms %.4f fixed_ms %.4f lo_ms %.4f hi_ms %.4f source %s\n",
              r.total_ms, r.compute_ms, r.membw_ms, r.comm_ms, r.fixed_ms, r.lo_ms, r.hi_ms, to_string(r.source));
  std::printf("note %s\n", r.note.c_str());
  for (const auto& [k, v] : r.per_op) std::printf("  %-32s %10.4f ms\n", k.c_str(), v);
  const MemoryPlan& mp = step->memory();
  std::printf("weight_bytes_per_gpu %.4g kv_bytes_per_token %.1f\n", mp.weight_bytes_per_gpu, mp.kv_bytes_per_token);
  return 0;
}

int cmd_sim(const Args& a) {
  BenchPoint p = load_point(a.get("point"), overrides_of(a));
  auto ts = load_traces(a);
  MeasuredTables ops(a.get("perfdata", "perfdata"), load_device_specs(a.get("devices-dir", "configs/device")));
  SimOut o = run_point(p, ts, ops);
  std::string row = to_inferencex_row_json(o.result, o.metrics, row_meta(p), p.mapping, p.run.concurrency);
  if (a.has("out")) write_file(a.get("out"), row + "\n");
  if (a.has("records")) {   // per-request records, one line each
    std::string csv = "trace,request,agent,play,lane,prefill_worker,worker,warmup,isl,osl,hit_tokens,computed_prefill_tokens,arrive_s,admit_s,first_token_s,end_s,delivery_ms\n";
    for (const RequestRecord& r : o.result.records) {
      int64_t hits = 0;
      for (const auto& th : r.tier_hits) hits += th.second;
      const int agent = r.trace >= 0 && r.request >= 0 ? ts.traces[r.trace].requests[r.request].agent : -1;
      char buf[512];
      std::snprintf(buf, sizeof buf, "%d,%d,%d,%d,%d,%d,%d,%d,%lld,%lld,%lld,%lld,%.3f,%.3f,%.3f,%.3f,%.3f\n", r.trace, r.request, agent, r.play,
                    r.lane, r.prefill_worker, r.worker, int(r.warmup), (long long)r.isl, (long long)r.osl, (long long)hits,
                    (long long)r.computed_prefill_tokens, r.arrive_s, r.admit_s, r.first_token_s, r.end_s, r.delivery_ms);
      csv += buf;
    }
    write_file(a.get("records"), csv);
  }
  std::printf("%s\n", row.c_str());
  std::fprintf(stderr, "wall %.2f s, requests %zu, unsupported passes %.0f of %.0f, oversized requests %.0f\n", o.wall_s,
               o.result.records.size(), o.result.extra.count("unsupported_passes") ? o.result.extra.at("unsupported_passes") : 0.0,
               o.result.extra.count("passes") ? o.result.extra.at("passes") : 0.0,
               o.result.extra.count("oversized_requests") ? o.result.extra.at("oversized_requests") : 0.0);
  if (!o.result.unsupported_note.empty()) std::fprintf(stderr, "first unsupported: %s\n", o.result.unsupported_note.c_str());
  return 0;
}

struct PointEval {
  BenchPoint p;
  Metrics m;
  double wall_s = 0;
  double unsupported_passes = 0;
  double oversized = 0;   // requests that could never fit a rank pool (capacity shortfall)
  std::map<std::string, double> err;   // relative errors per compared metric
};

int g_jobs = 32;   // parallel point evaluations (--jobs)

std::vector<PointEval> evaluate_points(const std::vector<BenchPoint>& pts, const trace::TraceSet& ts, const OpLatencySource& ops,
                                       const std::map<int, SnapshotRow>& snap, bool verbose) {
  std::vector<PointEval> out;
  std::deque<std::future<PointEval>> pending;
  auto eval_one = [&](const BenchPoint& p) {
    PointEval e;
    e.p = p;
    SimOut o = run_point(p, ts, ops);
    e.m = o.metrics;
    e.wall_s = o.wall_s;
    e.unsupported_passes = o.result.extra.count("unsupported_passes") ? o.result.extra.at("unsupported_passes") : 0;
    e.oversized = o.result.extra.count("oversized_requests") ? o.result.extra.at("oversized_requests") : 0;
    auto it = snap.find(p.id);
    if (it != snap.end()) {
      for (const auto& k : kCompareMetrics) {
        auto pm = e.m.values.find(k);
        auto mm = it->second.metrics.find(k);
        if (pm != e.m.values.end() && mm != it->second.metrics.end()) e.err[k] = rel_err(pm->second, mm->second);
      }
    }
    return e;
  };
  auto drain = [&]() {
    PointEval e = pending.front().get();
    pending.pop_front();
    if (verbose) {
      auto it = snap.find(e.p.id);
      std::printf("%d %s %s conc=%d wall=%.1fs unsupported=%.0f oversized=%.0f", e.p.id, e.p.hardware.c_str(), e.p.framework.c_str(),
                  e.p.run.concurrency, e.wall_s, e.unsupported_passes, e.oversized);
      for (const auto& k : kCompareMetrics) {
        double pv = e.m.values.count(k) ? e.m.values.at(k) : NAN;
        double mv = it != snap.end() && it->second.metrics.count(k) ? it->second.metrics.at(k) : NAN;
        std::printf(" | %s pred=%.3g meas=%.3g err=%+.0f%%", k.c_str(), pv, mv, 100 * rel_err(pv, mv));
      }
      std::printf("\n");
      std::fflush(stdout);
    }
    out.push_back(std::move(e));
  };
  for (const auto& p : pts) {
    if (int(pending.size()) >= std::max(1, g_jobs)) drain();
    pending.push_back(std::async(std::launch::async, eval_one, std::cref(p)));
  }
  while (!pending.empty()) drain();
  return out;
}

std::string report_json(const std::vector<PointEval>& evals) {
  std::string s = "{\"points\":[";
  for (size_t i = 0; i < evals.size(); ++i) {
    const auto& e = evals[i];
    if (i) s += ",";
    s += "{\"id\":" + std::to_string(e.p.id) + ",\"hardware\":\"" + e.p.hardware + "\",\"framework\":\"" + e.p.framework +
         "\",\"conc\":" + std::to_string(e.p.run.concurrency) + ",\"wall_s\":" + std::to_string(e.wall_s) + ",\"oversized_requests\":" + std::to_string(e.oversized) +
         ",\"metrics\":" + metrics_json(e.m) + ",\"rel_err\":{";
    bool first = true;
    for (const auto& [k, v] : e.err) {
      if (!first) s += ",";
      first = false;
      s += "\"" + k + "\":" + (std::isfinite(v) ? std::to_string(v) : "null");
    }
    s += "}}";
  }
  s += "],\"median_abs_rel_err\":{";
  bool first = true;
  for (const auto& k : kCompareMetrics) {
    std::vector<double> v;
    for (const auto& e : evals) if (e.err.count(k)) v.push_back(std::fabs(e.err.at(k)));
    if (!first) s += ",";
    first = false;
    double m = median(v);
    s += "\"" + k + "\":" + (std::isfinite(m) ? std::to_string(m) : "null");
  }
  return s + "}}";
}

// Cross-system ratio errors on matched points: same framework, spec, disagg, offload, topology and concurrency.
std::string ratio_report(const std::vector<PointEval>& evals, const std::map<int, SnapshotRow>& snap) {
  auto key = [](const BenchPoint& p) {
    std::ostringstream k;
    k << p.framework << '|' << p.spec_method << '|' << p.mapping.disaggregated << '|' << p.offload_mode << '|'
      << p.mapping.prefill.workers << 'x' << p.mapping.prefill.tp << '.' << p.mapping.prefill.attention_dp << '.'
      << p.mapping.prefill.moe_ep << '|' << p.mapping.decode.workers << 'x' << p.mapping.decode.tp << '.'
      << p.mapping.decode.attention_dp << '.' << p.mapping.decode.moe_ep << '|' << p.run.concurrency;
    return k.str();
  };
  std::map<std::string, std::vector<const PointEval*>> groups;
  for (const auto& e : evals) groups[key(e.p)].push_back(&e);
  std::string s = "{\"pairs\":[";
  std::vector<double> errs;
  bool first = true;
  for (const auto& [k, v] : groups) {
    for (size_t i = 0; i < v.size(); ++i)
      for (size_t j = i + 1; j < v.size(); ++j) {
        if (v[i]->p.hardware == v[j]->p.hardware) continue;
        for (const auto& mk : kCompareMetrics) {
          auto a = v[i]->m.values.find(mk), b = v[j]->m.values.find(mk);
          auto sa = snap.find(v[i]->p.id), sb = snap.find(v[j]->p.id);
          if (a == v[i]->m.values.end() || b == v[j]->m.values.end() || sa == snap.end() || sb == snap.end()) continue;
          if (!sa->second.metrics.count(mk) || !sb->second.metrics.count(mk)) continue;
          double pred = a->second / b->second, meas = sa->second.metrics.at(mk) / sb->second.metrics.at(mk);
          double err = pred / meas - 1;
          errs.push_back(std::fabs(err));
          if (!first) s += ",";
          first = false;
          s += "{\"key\":\"" + json_escape(k) + "\",\"metric\":\"" + mk + "\",\"a\":" + std::to_string(v[i]->p.id) +
               ",\"b\":" + std::to_string(v[j]->p.id) + ",\"pred_ratio\":" + std::to_string(pred) +
               ",\"meas_ratio\":" + std::to_string(meas) + ",\"ratio_err\":" + std::to_string(err) + "}";
        }
      }
  }
  double m = median(errs);
  return s + "],\"median_abs_ratio_err\":" + (std::isfinite(m) ? std::to_string(m) : "null") + "}";
}

int cmd_validate_points(const Args& a) {
  auto pts = select_points(a);
  if (pts.empty()) throw std::runtime_error("no points selected");
  auto snap = load_snapshot(a.get("snapshot", "traces/inferencex/benchmarks_DeepSeek-V4-Pro_2026-09-20.json"),
                            a.get("derived", "traces/inferencex/derived-agentic-metrics_2026-09-20.json"));
  auto ts = load_traces(a);
  MeasuredTables ops(a.get("perfdata", "perfdata"), load_device_specs(a.get("devices-dir", "configs/device")));
  auto evals = evaluate_points(pts, ts, ops, snap, true);
  std::string rep = report_json(evals);
  std::string ratios = ratio_report(evals, snap);
  std::string out = "{\"report\":" + rep + ",\"ratios\":" + ratios + "}\n";
  if (a.has("out")) write_file(a.get("out"), out);
  std::printf("%s", out.c_str());
  return 0;
}

int cmd_validate_fabric(const Args& a) {
  const std::string cx = a.get("collectivex", "traces/inferencex/collectivex_2026-09-20");
  const std::string systems = a.get("systems", "third_party/aisimulate/python/aisimulate/src/aisimulate_core/systems");
  const auto rows = load_collectivex_dir(cx);
  std::map<std::string, FabricSpec> so;
  for (const auto& [sku, yaml] : std::map<std::string, std::string>{{"b200-nscale", "b200_sxm"}, {"b300", "b300_sxm"}, {"h100-dgxc", "h100_sxm"},
                                                                     {"h200-dgxc", "h200_sxm"}, {"gb200", "gb200"}, {"gb300", "gb300"}})
    so[sku] = scaleout_from_aisimulate(systems + "/" + yaml + ".yaml");
  const std::string report = predict_ep16_report(rows, so);
  if (a.has("out")) write_file(a.get("out"), report + "\n");
  else std::printf("%s\n", report.c_str());
  std::fprintf(stderr, "collectivex rows %zu\n", rows.size());
  return 0;
}

int cmd_calibrate(const Args& a) {
  auto pts = select_points(a);
  if (pts.empty()) throw std::runtime_error("no points selected");
  auto snap = load_snapshot(a.get("snapshot", "traces/inferencex/benchmarks_DeepSeek-V4-Pro_2026-09-20.json"),
                            a.get("derived", "traces/inferencex/derived-agentic-metrics_2026-09-20.json"));
  auto ts = load_traces(a);
  MeasuredTables ops(a.get("perfdata", "perfdata"), load_device_specs(a.get("devices-dir", "configs/device")));
  std::vector<double> fixed, accept, overhead;
  {
    std::stringstream f(a.get("fixed-ms", "0,0.5,1,2,4,8")), g(a.get("accept-mean", "0.5,1,1.5,2,2.5")),
        h(a.get("overhead-ms", "0"));
    std::string t;
    while (std::getline(f, t, ',')) fixed.push_back(std::stod(t));
    while (std::getline(g, t, ',')) accept.push_back(std::stod(t));
    while (std::getline(h, t, ',')) overhead.push_back(std::stod(t));
  }
  std::string best;
  double best_obj = INFINITY;
  std::string grid = "[";
  bool first = true;
  for (double fm : fixed)
    for (double am : accept)
    for (double om : overhead) {
      std::vector<BenchPoint> variant = pts;
      for (auto& p : variant) {
        apply_stack_override(p.stack, "t_step_fixed_ms", std::to_string(fm));
        apply_stack_override(p.stack, "request_overhead_ms", std::to_string(om));
        if (p.stack.mtp_nextn > 0) apply_stack_override(p.stack, "mtp_accept_mean", std::to_string(am));
      }
      auto evals = evaluate_points(variant, ts, ops, snap, false);
      // Objective: mean |log(1 + err)| over the four calibration metrics and the selected points.
      const char* keys[] = {"tput_per_gpu", "p90_e2e_norm_intvty", "median_ttft", "p90_intvty"};
      double sum = 0;
      int cnt = 0;
      std::string errs = "[";
      for (size_t i = 0; i < evals.size(); ++i) {
        const auto& e = evals[i];
        errs += std::string(i ? "," : "") + "{\"id\":" + std::to_string(variant[i].id);
        for (const char* k : keys) {
          if (!e.err.count(k) || !std::isfinite(e.err.at(k))) continue;
          sum += std::fabs(std::log1p(e.err.at(k)));
          cnt++;
          errs += std::string(",\"") + k + "\":" + std::to_string(e.err.at(k));
        }
        errs += "}";
      }
      errs += "]";
      double obj = cnt ? sum / cnt : INFINITY;
      std::printf("t_step_fixed_ms=%.2f accept_mean=%.2f request_overhead_ms=%.1f objective=%.4f errors=%s\n", fm, am, om, obj,
                  errs.c_str());
      std::fflush(stdout);
      if (!first) grid += ",";
      first = false;
      grid += "{\"t_step_fixed_ms\":" + std::to_string(fm) + ",\"accept_mean\":" + std::to_string(am) +
              ",\"request_overhead_ms\":" + std::to_string(om) + ",\"objective\":" + std::to_string(obj) + ",\"errors\":" + errs + "}";
      if (obj < best_obj) {
        best_obj = obj;
        best = "{\"t_step_fixed_ms\":" + std::to_string(fm) + ",\"mtp_accept_mean\":" + std::to_string(am) +
               ",\"request_overhead_ms\":" + std::to_string(om) + ",\"objective\":" + std::to_string(obj) + "}";
      }
    }
  std::string out = "{\"best\":" + best + ",\"grid\":" + grid + "]}\n";
  if (a.has("out")) write_file(a.get("out"), out);
  std::printf("%s", out.c_str());
  return 0;
}

// Design-variable paths for inverse solving.
void set_path(BenchPoint& p, const std::string& name, double v) {
  DeviceSpec& d = p.devices.devices.begin()->second;
  auto starts = [&](const char* s) { return name.rfind(s, 0) == 0; };
  if (starts("device.flops.")) d.flops[name.substr(13)] = v;
  else if (name == "device.memory.0.bandwidth_Bps") d.memory[0].bandwidth_Bps = v;
  else if (name == "device.memory.0.capacity_bytes") d.memory[0].capacity_bytes = v;
  else if (name == "device.memory.1.bandwidth_Bps") d.memory.at(1).bandwidth_Bps = v;
  else if (name == "device.memory.1.capacity_bytes") d.memory.at(1).capacity_bytes = v;
  else if (name == "device.price_usd") d.price_usd = v;
  else if (name == "device.power_w") d.power_w = v;
  else if (name == "fabric.scaleup_domain") p.fabric.scaleup_domain = int(v);
  else if (name == "fabric.scaleup.bandwidth_Bps") p.fabric.scaleup.bandwidth_Bps = v;
  else if (name == "fabric.scaleup.alpha_s") p.fabric.scaleup.alpha_s = v;
  else if (name == "fabric.scaleout.bandwidth_Bps") p.fabric.scaleout.bandwidth_Bps = v;
  else if (name == "fabric.scaleout.alpha_s") p.fabric.scaleout.alpha_s = v;
  else if (name == "fabric.scaleout_oversub") p.fabric.scaleout_oversub = v;
  else if (name == "fabric.host.bandwidth_Bps") p.fabric.host.bandwidth_Bps = v;
  else if (name == "mapping.decode.workers") p.mapping.decode.workers = int(v);
  else if (name == "mapping.prefill.workers") p.mapping.prefill.workers = int(v);
  else if (name == "run.concurrency") p.run.concurrency = int(v);
  else if (name == "stack.t_step_fixed_ms") p.stack.t_step_fixed_ms = v;
  else if (name == "stack.request_overhead_ms") p.stack.request_overhead_ms = v;
  else if (name == "stack.mtp_accept_mean") apply_stack_override(p.stack, "mtp_accept_mean", std::to_string(v));
  else if (name == "stack.residency_limit_s") p.stack.residency_limit_s = v;
  else throw std::runtime_error("unsupported design variable '" + name + "'");
}

struct CostModel {
  double depreciation_s = 5 * 365.25 * 86400, electricity_usd_per_kwh = 0.1, pue = 1.3, fabric_usd_per_gpu_s = 0;
};

BenchPoint apply_point(const BenchPoint& base, const BenchPoint& mods, const std::map<std::string, double>& x) {
  (void)mods;
  BenchPoint p = base;
  for (const auto& [k, v] : x) if (k != "seed") set_path(p, k, v);
  return p;
}

int cmd_solve(const Args& a) {
  YAML::Node q = YAML::LoadFile(a.get("query"));
  BenchPoint base = load_point(q["point"].as<std::string>(), overrides_of(a));
  auto ts = load_traces(a);
  MeasuredTables ops(a.get("perfdata", "perfdata"), load_device_specs(a.get("devices-dir", "configs/device")));
  Query query;
  std::string kind = q["kind"].as<std::string>("feasible");
  query.kind = kind == "optimum" ? QueryKind::Optimum : kind == "pareto" ? QueryKind::Pareto : QueryKind::FeasibleSet;
  for (const auto& v : q["variables"]) {
    Variable var;
    var.name = v["name"].as<std::string>();
    var.lo = v["lo"].as<double>(0);
    var.hi = v["hi"].as<double>(0);
    var.log_scale = v["log_scale"].as<bool>(true);
    if (v["choices"]) var.choices = v["choices"].as<std::vector<double>>();
    query.variables.push_back(var);
  }
  for (const auto& t : q["targets"]) query.targets.push_back({t["metric"].as<std::string>(), t["min"].as<double>()});
  for (const auto& o : q["objectives"]) query.objectives.push_back({o["metric"].as<std::string>(), o["minimize"].as<bool>(true)});
  for (const auto& c : q["constraints"]) {
    LinearConstraint lc;
    lc.name = c["name"].as<std::string>("");
    lc.bound = c["bound"].as<double>();
    for (const auto& w : c["weights"]) lc.weights[w.first.as<std::string>()] = w.second.as<double>();
    query.constraints.push_back(lc);
  }
  query.grid_per_axis = q["grid_per_axis"].as<int>(6);
  query.sensitivity_step = q["sensitivity_step"].as<double>(0.05);
  query.seeds = q["seeds"].as<int>(1);
  CostModel cost;
  if (q["cost"]) {
    cost.depreciation_s = q["cost"]["depreciation_s"].as<double>(cost.depreciation_s);
    cost.electricity_usd_per_kwh = q["cost"]["electricity_usd_per_kwh"].as<double>(cost.electricity_usd_per_kwh);
    cost.pue = q["cost"]["pue"].as<double>(cost.pue);
    cost.fabric_usd_per_gpu_s = q["cost"]["fabric_usd_per_gpu_s"].as<double>(cost.fabric_usd_per_gpu_s);
  }
  trace::Stats st = trace::compute_stats(ts);
  Forward forward = [&](const std::map<std::string, double>& x) {
    BenchPoint p = apply_point(base, base, x);
    if (x.count("seed")) p.run.seed = uint64_t(x.at("seed"));
    SimOut o = run_point(p, ts, ops);
    std::map<std::string, double> y = o.metrics.values;
    const DeviceSpec& d = p.devices.devices.begin()->second;
    int gpus = p.mapping.decode.gpus() + (p.mapping.disaggregated ? p.mapping.prefill.gpus() : 0);
    double tput = y.count("tput_per_gpu") ? y["tput_per_gpu"] * gpus : 0;
    double usd_per_s = gpus * d.price_usd / cost.depreciation_s + gpus * d.power_w / 1000.0 * cost.electricity_usd_per_kwh / 3600.0 * cost.pue +
                       gpus * cost.fabric_usd_per_gpu_s;
    y["gpus"] = gpus;
    y["cost_per_mtoken"] = tput > 0 ? usd_per_s / tput * 1e6 : INFINITY;
    return y;
  };
  Forward screen = [&](const std::map<std::string, double>& x) {
    BenchPoint p = apply_point(base, base, x);
    const DeviceSpec& d = p.devices.devices.begin()->second;
    const PoolSpec& pool = p.mapping.decode;
    std::map<std::string, double> y;
    y["sessions_capacity"] = screen_sessions(d, p.fabric, pool, p.stack, st) * pool.workers;
    return y;
  };
  Result r = solve(query, forward, q["screen"].as<bool>(true) ? screen : nullptr);
  std::string out = "{\"evaluated\":" + std::to_string(r.evaluated.size()) + ",\"solutions\":[";
  for (size_t i = 0; i < r.solutions.size(); ++i) {
    if (i) out += ",";
    out += "{\"x\":{";
    bool f = true;
    for (const auto& [k, v] : r.solutions[i].x) { if (!f) out += ","; f = false; out += "\"" + k + "\":" + std::to_string(v); }
    out += "},\"y\":{";
    f = true;
    for (const auto& [k, v] : r.solutions[i].y) { if (!f) out += ","; f = false; out += "\"" + k + "\":" + std::to_string(v); }
    out += "}";
    if (i < r.sensitivities.size()) {
      out += ",\"unidentifiable\":[";
      for (size_t j = 0; j < r.sensitivities[i].unidentifiable.size(); ++j) {
        if (j) out += ",";
        out += "\"" + r.sensitivities[i].unidentifiable[j] + "\"";
      }
      out += "],\"condition_number\":" + std::to_string(r.sensitivities[i].condition_number);
    }
    out += "}";
  }
  out += "],\"thresholds\":{";
  bool f = true;
  for (const auto& [k, v] : r.thresholds) { if (!f) out += ","; f = false; out += "\"" + k + "\":" + std::to_string(v); }
  out += "},\"monotone\":" + std::string(r.monotone ? "true" : "false") + ",\"active_constraints\":[";
  for (size_t i = 0; i < r.active_constraints.size(); ++i) out += (i ? ",\"" : "\"") + r.active_constraints[i] + "\"";
  out += "],\"inactive_constraints\":[";
  for (size_t i = 0; i < r.inactive_constraints.size(); ++i) out += (i ? ",\"" : "\"") + r.inactive_constraints[i] + "\"";
  out += "],\"points\":[";
  for (size_t i = 0; i < r.evaluated.size(); ++i) {
    if (i) out += ",";
    out += "{\"x\":{";
    bool g = true;
    for (const auto& [k, v] : r.evaluated[i].x) { if (!g) out += ","; g = false; out += "\"" + k + "\":" + std::to_string(v); }
    out += "},\"y\":{";
    g = true;
    for (const auto& [k, v] : r.evaluated[i].y) { if (!g) out += ","; g = false; out += "\"" + k + "\":" + (std::isfinite(v) ? std::to_string(v) : "null"); }
    out += "},\"feasible\":" + std::string(r.evaluated[i].feasible ? "true" : "false") + "}";
  }
  out += "]}\n";
  if (a.has("out")) write_file(a.get("out"), out);
  std::printf("%s", out.c_str());
  return 0;
}

int cmd_screen(const Args& a) {
  BenchPoint p = load_point(a.get("point"), overrides_of(a));
  auto ts = load_traces(a);
  trace::Stats st = trace::compute_stats(ts);
  const DeviceSpec& d = p.devices.devices.begin()->second;
  double n = screen_sessions(d, p.fabric, p.mapping.decode, p.stack, st);
  std::printf("{\"sessions_per_worker\":%.2f,\"sessions_total\":%.2f}\n", n, n * p.mapping.decode.workers);
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  Args a = parse_args(argc, argv);
  try {
    if (a.command == "trace-stats") return cmd_trace_stats(a);
    if (a.command == "trace-graph") return cmd_trace_graph(a);
    if (a.command == "sim") return cmd_sim(a);
    if (a.command == "validate-points") return cmd_validate_points(a);
    if (a.has("jobs")) g_jobs = std::stoi(a.get("jobs"));
    if (a.command == "calibrate") return cmd_calibrate(a);
    if (a.command == "validate-fabric") return cmd_validate_fabric(a);
    if (a.command == "step") return cmd_step(a);
    if (a.command == "op") return cmd_op(a);
    if (a.command == "solve") return cmd_solve(a);
    if (a.command == "screen") return cmd_screen(a);
    std::fprintf(stderr,
                 "usage: dlsim <trace-stats|trace-graph|op|step|sim|validate-points|validate-fabric|calibrate|solve|screen> [--key value ...]\n"
                 "  common: --trace <jsonl> --cache <dir> --perfdata <dir> --stack-override k=v,k=v\n"
                 "  sim: --point <yaml> [--out row.json]\n"
                 "  validate-points: [--points-dir d] [--hardware h] [--framework f] [--ids a,b] [--snapshot j] [--derived j] [--out r.json]\n"
                 "  calibrate: same selection, [--fixed-ms 0,1,2] [--accept-mean 1,2] [--overhead-ms 0,500] [--out r.json]\n"
                 "  solve: --query <yaml> [--out r.json]\n");
    return 2;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "dlsim: %s\n", e.what());
    return 1;
  }
}
