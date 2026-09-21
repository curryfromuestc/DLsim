#include <cassert>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <string>
#include <vector>

#include "perfdata/decomposition.h"
#include "perfdata/measured.h"
#include "perfdata/roofline.h"
#include "perfdata/system_specs.h"

using namespace dlsim;

namespace {

Column num(const std::string& name, std::vector<double> v) {
  Column c;
  c.name = name;
  c.kind = Column::Kind::Int;
  c.num = std::move(v);
  return c;
}
Column flt(const std::string& name, std::vector<double> v) {
  Column c = num(name, std::move(v));
  c.kind = Column::Kind::Float;
  return c;
}
Column str(const std::string& name, const std::vector<std::string>& v) {
  Column c;
  c.name = name;
  c.kind = Column::Kind::Str;
  for (const auto& s : v) c.idx.push_back(c.intern(s));
  return c;
}
Table make(std::vector<Column> cols) {
  Table t;
  t.rows = cols[0].is_str() ? cols[0].idx.size() : cols[0].num.size();
  t.cols = std::move(cols);
  return t;
}
std::vector<uint32_t> all_rows(const Table& t) {
  std::vector<uint32_t> r(t.rows);
  for (uint32_t i = 0; i < t.rows; ++i) r[i] = i;
  return r;
}
bool near(double a, double b, double tol = 1e-9) { return std::abs(a - b) <= tol * std::max(1.0, std::abs(b)); }

void test_gemm() {
  std::vector<double> m, n, k, lat;
  std::vector<std::string> dt;
  for (auto [nn, kk] : std::vector<std::pair<double, double>>{{1024, 1024}, {2048, 1024}, {4096, 4096}})
    for (double mm : {1, 2, 4, 8, 16}) {
      m.push_back(mm);
      n.push_back(nn);
      k.push_back(kk);
      lat.push_back(0.01 + mm * nn * kk * 1e-9);
      dt.push_back("bfloat16");
    }
  Table t = make({str("gemm_dtype", dt), num("m", m), num("n", n), num("k", k), flt("latency", lat)});
  DeviceSpec spec;
  spec.flops["bf16"] = 1e15;
  spec.memory.push_back({"hbm", 1e11, 1e12, 0.9});
  OpQuery q;
  q.op = "gemm";
  q.cat["gemm_dtype"] = "bfloat16";
  q.num = {{"m", 4}, {"n", 1024}, {"k", 1024}};
  Latency l = query_rows(t, all_rows(t), q, &spec);
  assert(l.source == Source::Measured && near(l.ms, 0.01 + 4 * 1024 * 1024 * 1e-9));
  q.num["m"] = 3;
  l = query_rows(t, all_rows(t), q, &spec);
  const double lo = 0.01 + 2 * 1024 * 1024 * 1e-9, hi = 0.01 + 4 * 1024 * 1024 * 1e-9;
  assert(l.source == Source::Interpolated && l.ms > lo && l.ms < hi && near(l.ms, (lo + hi) / 2));
  q.num["n"] = 1536;
  l = query_rows(t, all_rows(t), q, &spec);
  assert(l.source == Source::Interpolated && l.note.find("util transfer") != std::string::npos && l.ms > 0);
  q.num["n"] = 1024;
  q.num["m"] = 32;   // above the measured m range: linear in m from the last two points (synthetic curve is exactly linear)
  l = query_rows(t, all_rows(t), q, &spec);
  assert(l.source == Source::Extrapolated && near(l.ms, 0.01 + 32.0 * 1024 * 1024 * 1e-9));
  q.num["m"] = 0.5;
  l = query_rows(t, all_rows(t), q, &spec);
  assert(l.source == Source::Unsupported);
  q.num["m"] = 4;
  q.cat.erase("gemm_dtype");
  l = query_rows(t, all_rows(t), q, &spec);
  assert(l.source == Source::Measured);
}

void test_grid_and_extrapolation() {
  std::vector<double> b, step, lat;
  std::vector<std::string> model;
  for (double bb : {1, 2})
    for (double s : {1024, 2048, 4096}) {
      b.push_back(bb);
      step.push_back(s);
      lat.push_back(0.05 * bb + 1e-5 * bb * s);
      model.push_back("dsv4");
    }
  Table t = make({str("model", model), num("batch_size", b), num("isl", std::vector<double>(6, 1)), num("step", step),
                  num("compress_ratio", std::vector<double>(6, 4)), flt("latency", lat)});
  OpQuery q;
  q.op = "dsv4_csa_generation_module";
  q.num = {{"batch_size", 1}, {"isl", 1}, {"step", 2048}};
  Latency l = query_rows(t, all_rows(t), q, nullptr);
  assert(l.source == Source::Measured && near(l.ms, 0.05 + 1e-5 * 2048));
  q.num["step"] = 3072;
  l = query_rows(t, all_rows(t), q, nullptr);
  assert(l.source == Source::Interpolated && l.ms > 0.05 + 1e-5 * 2048 && l.ms < 0.05 + 1e-5 * 4096);
  q.num["batch_size"] = 1.5;
  q.num["step"] = 2048;
  l = query_rows(t, all_rows(t), q, nullptr);
  assert(l.source == Source::Interpolated && near(l.ms, 0.5 * (0.05 + 1e-5 * 2048) + 0.5 * (0.1 + 2e-5 * 2048)));
  q.num["batch_size"] = 1;
  q.num["step"] = 8192;
  l = query_rows(t, all_rows(t), q, nullptr);
  assert(l.source == Source::Extrapolated && l.note.find("query/edge=2") != std::string::npos);
  assert(near(l.ms, 0.05 + 1e-5 * 8192));
  q.num["step"] = 512;
  l = query_rows(t, all_rows(t), q, nullptr);
  assert(l.source == Source::Unsupported);

  q.op = "dsv4_csa_context_module";
  q.num = {{"batch_size", 1}, {"isl", 1}, {"step", 8192}};
  l = query_rows(t, all_rows(t), q, nullptr);
  const double t_e = 0.05 + 1e-5 * 4096, t_p = 0.05 + 1e-5 * 2048, alpha = (t_e - t_p) / 2048;
  assert(l.source == Source::Extrapolated && near(l.ms, t_e + alpha * (8192 - 4096)) && l.note.find("query/edge=") != std::string::npos);

  q.op = "dsv4_hca_context_module";
  l = query_rows(t, all_rows(t), q, nullptr);
  assert(l.source == Source::Extrapolated && near(l.ms, t_e + alpha * (8192 - 4096)) && l.note.find("query/edge=") != std::string::npos);

  q.op = "context_attention";
  l = query_rows(t, all_rows(t), q, nullptr);
  assert(l.source == Source::Unsupported);

  // Ragged grid: the batch 2 line stops at 2048 while batch 1 reaches 4096. Above its own edge the batch 2 line takes the
  // batch 1 line's shape scaled at the shared step: t(2, 2048) * t(1, 4096) / t(1, 2048).
  {
    std::vector<double> b2, s2, l2;
    std::vector<std::string> m2;
    for (double s : {1024, 2048, 4096}) { b2.push_back(1); s2.push_back(s); l2.push_back(0.05 + 1e-5 * s); m2.push_back("dsv4"); }
    for (double s : {1024, 2048}) { b2.push_back(2); s2.push_back(s); l2.push_back(0.1 + 4e-5 * s); m2.push_back("dsv4"); }
    Table tr = make({str("model", m2), num("batch_size", b2), num("isl", std::vector<double>(5, 1)), num("step", s2),
                     num("compress_ratio", std::vector<double>(5, 4)), flt("latency", l2)});
    OpQuery qr;
    qr.op = "dsv4_csa_generation_module";
    qr.num = {{"batch_size", 2}, {"isl", 1}, {"step", 4096}};
    Latency lr = query_rows(tr, all_rows(tr), qr, nullptr);
    const double want = (0.1 + 4e-5 * 2048) * (0.05 + 1e-5 * 4096) / (0.05 + 1e-5 * 2048);
    assert(lr.source == Source::Extrapolated && near(lr.ms, want) && lr.note.find("shape from the batch_size=1 line") != std::string::npos);
  }
}

void test_decomposition() {
  const std::vector<std::string> dev = {"A", "B", "C", "D"};
  const std::vector<double> C = {1e15, 2e15, 4e15, 1e15}, Bw = {1e12, 4e12, 2e12, 8e12};
  const double a = 2e12, b = 4e9, c = 0.3;
  std::vector<double> t;
  for (size_t i = 0; i < 4; ++i) t.push_back(a / C[i] + b / Bw[i] + c);
  DecompositionFit f = fit_decomposition(dev, C, Bw, t);
  assert(f.rank == DecompositionFit::Rank::Full);
  assert(near(f.a, a, 1e-6) && near(f.b, b, 1e-6) && near(f.c, c, 1e-6));
  assert(f.residual_rel < 1e-9 && f.loo_rel < 1e-6);
  Latency l = decomposed_latency(f, 2e15, 4e12, C, Bw, t, Latency::unsupported(""));
  assert(l.source == Source::Decomposed && near(l.ms, a / 2e15 + b / 4e12 + c, 1e-6) && l.lo_ms <= l.ms && l.ms <= l.hi_ms);

  const std::vector<double> Bw_same = {8e12, 8e12, 8e12, 8e12};
  std::vector<double> t2;
  for (size_t i = 0; i < 4; ++i) t2.push_back(a / C[i] + b / Bw_same[i] + c);
  f = fit_decomposition(dev, C, Bw_same, t2);
  assert(f.rank == DecompositionFit::Rank::BwDeficient);
  l = decomposed_latency(f, 2e15, 1.79e12, C, Bw_same, t2, Latency::unsupported(""));
  assert(l.source == Source::Bound && l.lo_ms <= l.hi_ms && l.lo_ms > 0);
  const double truth = a / 2e15 + b / 1.79e12 + c;
  assert(l.lo_ms <= truth && truth <= l.hi_ms);

  f = fit_decomposition({"A", "B"}, {C[0], C[1]}, {Bw[0], Bw[1]}, {t[0], t[1]});
  assert(f.rank == DecompositionFit::Rank::BwDeficient);
  l = decomposed_latency(f, 2e15, 4e12, {C[0], C[1]}, {Bw[0], Bw[1]}, {t[0], t[1]}, Latency::unsupported(""));
  assert(l.source == Source::Bound && l.lo_ms <= l.hi_ms);
  f = fit_decomposition({"A"}, {C[0]}, {Bw[0]}, {t[0]});
  assert(f.rank == DecompositionFit::Rank::Underdetermined);
  l = decomposed_latency(f, 2e15, 4e12, {C[0]}, {Bw[0]}, {t[0]}, Latency::unsupported(""));
  assert(l.source == Source::Bound && l.lo_ms <= l.hi_ms);
}

void test_roofline() {
  DeviceSpec spec;
  spec.name = "x";
  spec.flops["bf16"] = 1e15;
  spec.memory.push_back({"hbm", 1e11, 1e12, 0.9});
  RooflineSource rf({{"x", spec}});
  OpQuery q;
  q.op = "gemm";
  q.cat["gemm_dtype"] = "bfloat16";
  q.num = {{"m", 1}, {"n", 1024}, {"k", 1024}};
  Latency l = rf.query("x", q);
  assert(l.source == Source::Roofline && near(l.ms, std::max(2.0 * 1024 * 1024 / 1e15, 2.0 * (1024 + 1024 + 1024 * 1024) / 1e12) * 1000));
  q.op = "generation_attention";
  q.cat = {{"attn_dtype", "bfloat16"}, {"kv_cache_dtype", "fp8"}};
  q.num = {{"batch_size", 2}, {"num_heads", 8}, {"num_key_value_heads", 2}, {"head_dim", 128}, {"isl", 1}, {"step", 1023}};
  l = rf.query("x", q);
  assert(l.source == Source::Roofline && near(l.ms, std::max(2.0 * 2 * 8 * 128 * 2 * 1024 / 1e15, 2.0 * 2 * 2 * 1024 * 128 / 1e12) * 1000));
  assert(rf.query("x", OpQuery{"mhc_module", {}, {}}).source == Source::Unsupported);
}

void test_measured_tables() {
  const std::string dir = std::string(DLSIM_SOURCE_DIR) + "/perfdata";
  const std::string systems = std::string(DLSIM_SOURCE_DIR) + "/third_party/aisimulate/python/aisimulate/src/aisimulate_core/systems";
  if (!std::filesystem::exists(dir + "/manifest.json")) {
    std::puts("perfdata/manifest.json missing: skipping the imported-table checks");
    return;
  }
  MeasuredTables tables(dir, load_system_specs(systems));
  auto t = tables.table("gb300", "trtllm", "1.3.0rc23", "gemm");
  assert(t && t->rows > 0);
  OpQuery q;
  q.op = "gemm";
  q.cat = {{"framework", "trtllm"}, {"version", "1.3.0rc23"}, {"gemm_dtype", "bfloat16"}};
  q.num = {{"m", t->find("m")->num[0]}, {"n", t->find("n")->num[0]}, {"k", t->find("k")->num[0]}};
  const Column* dt = t->find("gemm_dtype");
  q.cat["gemm_dtype"] = dt->str(0);
  Latency l = tables.query("gb300", q);
  assert(l.source == Source::Measured && near(l.ms, t->find("latency")->num[0]));
  q.cat.erase("version");
  l = tables.query("gb300", q);
  assert(l.source == Source::Measured && l.note.find("default: latest") != std::string::npos);
  assert(tables.resolve_version("gb300", "trtllm", "next") == "1.3.0rc23");
}

}  // namespace

int main() {
  test_gemm();
  test_grid_and_extrapolation();
  test_decomposition();
  test_roofline();
  test_measured_tables();
  std::puts("perfdata_test ok");
  return 0;
}
