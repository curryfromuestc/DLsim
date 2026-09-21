#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <set>
#include <string>
#include <vector>

#include "engine/engine.h"

using namespace dlsim;

namespace {

struct FakeStep : StepLatency {
  double per_prefill_token_ms, per_decode_step_ms, fixed_ms;
  MemoryPlan plan;
  mutable int seen_nextn = -1;   // draft length the engine handed to the step model
  FakeStep(double a, double b, double f, double kv_bytes) : per_prefill_token_ms(a), per_decode_step_ms(b), fixed_ms(f) {
    plan.kv_bytes_per_token = kv_bytes;
  }
  StepResult step(const StepInput& in) const override {
    StepResult r;
    seen_nextn = in.nextn;
    for (const auto& rank : in.ranks) {
      if (rank.empty()) continue;
      double compute = 0, membw = 0;
      for (const auto& q : rank) {
        if (q.prefill) compute += double(q.new_tokens) * per_prefill_token_ms;
        else membw = per_decode_step_ms;
      }
      const double t = compute + membw + fixed_ms;
      if (t > r.total_ms) {
        r.total_ms = t;
        r.compute_ms = compute;
        r.membw_ms = membw;
        r.fixed_ms = fixed_ms;
      }
    }
    r.lo_ms = r.hi_ms = r.total_ms;
    return r;
  }
  const MemoryPlan& memory() const override { return plan; }
};

uint64_t mix(uint64_t a, uint64_t b) {
  uint64_t x = a ^ (b + 0x9e3779b97f4a7c15ULL + (a << 6) + (a >> 2));
  x ^= x >> 33;
  x *= 0xff51afd7ed558ccdULL;
  x ^= x >> 33;
  return x;
}

// Chained prefix hashes: block i depends on block i-1, so equal id implies equal prefix.
std::vector<uint64_t> chain(const std::vector<uint64_t>& prefix, uint64_t salt, size_t extra) {
  std::vector<uint64_t> c = prefix;
  uint64_t h = c.empty() ? 0x1234567ULL : c.back();
  for (size_t i = 0; i < extra; ++i) {
    h = mix(h, salt * 1000003ULL + i + 1);
    c.push_back(h);
  }
  return c;
}

struct TraceBuilder {
  trace::Trace t;
  TraceBuilder(const std::string& id) {
    t.id = id;
    t.models.push_back("m");
    trace::Agent main;
    t.agents.push_back(main);
  }
  int agent(int parent, bool background = false) {
    trace::Agent a;
    a.parent = parent;
    a.kind = trace::AgentKind::Subagent;
    a.background = background;
    t.agents.push_back(a);
    return int(t.agents.size()) - 1;
  }
  int req(int agent, double t_s, double api_s, const std::vector<uint64_t>& blocks, int64_t out) {
    trace::Request r;
    r.agent = agent;
    r.t_s = t_s;
    r.api_time_s = api_s;
    r.hash_begin = uint32_t(t.hash_ids.size());
    r.hash_count = uint32_t(blocks.size());
    r.in_tokens = int64_t(blocks.size()) * t.block_size;
    r.out_tokens = out;
    t.hash_ids.insert(t.hash_ids.end(), blocks.begin(), blocks.end());
    t.requests.push_back(r);
    return int(t.requests.size()) - 1;
  }
  void edge(int from, int to, trace::EdgeKind k, double delay) { t.edges.push_back({from, to, k, delay}); }
};

// workload-and-metrics.md: one seen set per trace, global time order, consecutive hits from the start.
double theoretical_hit_rate(const trace::Trace& t) {
  std::vector<int> order(t.requests.size());
  for (size_t i = 0; i < order.size(); ++i) order[i] = int(i);
  std::stable_sort(order.begin(), order.end(), [&](int a, int b) { return t.requests[a].t_s < t.requests[b].t_s; });
  std::set<uint64_t> seen;
  int64_t hit = 0, total = 0;
  for (int i : order) {
    const auto& r = t.requests[i];
    total += r.hash_count;
    bool run = true;
    for (uint32_t k = 0; k < r.hash_count; ++k) {
      const uint64_t id = t.hash_ids[r.hash_begin + k];
      if (run && seen.count(id)) hit++;
      else run = false;
      seen.insert(id);
    }
  }
  return total ? double(hit) / double(total) : 0.0;
}

DeviceSet devices(double kv_tokens_capacity, double kv_bytes) {
  DeviceSpec d;
  d.name = "dev";
  d.flops["fp4"] = 1e15;
  MemoryTier hbm;
  hbm.name = "hbm";
  hbm.capacity_bytes = kv_tokens_capacity * kv_bytes;
  hbm.usable_fraction = 1.0;
  hbm.bandwidth_Bps = 8e12;
  d.memory.push_back(hbm);
  DeviceSet ds;
  ds.devices[d.name] = d;
  return ds;
}

StackSpec stack_default() {
  StackSpec s;
  s.chunk_tokens = 4096;
  s.max_num_batched_tokens = 8192;
  s.max_num_seqs = 64;
  return s;
}

MappingSpec agg(int workers, int dp = 1) {
  MappingSpec m;
  m.decode.device = "dev";
  m.decode.workers = workers;
  m.decode.attention_dp = dp;
  m.decode.moe_ep = dp;
  return m;
}

RunSpec run_spec(int conc, double dur, uint64_t seed) {
  RunSpec r;
  r.concurrency = conc;
  r.duration_s = dur;
  r.seed = seed;
  r.idle_shift_s = 10;
  // The mechanics tests predate the InferenceX replay protocol defaults: full t* range, no follow-on warmup, no tree cap.
  r.t_star_min = 0;
  r.t_star_max = 1;
  r.warmup_requests_per_lane = 0;
  r.trace_idle_gap_cap_s = 0;
  return r;
}

std::string dump(const SimResult& r) {
  std::string s;
  char buf[256];
  for (const auto& x : r.records) {
    std::snprintf(buf, sizeof buf, "%d %d %d %d %d %d %d %a %a %a %a %lld %lld %lld %a|", x.trace, x.request, x.lane, x.play,
                  x.worker, x.prefill_worker, int(x.warmup), x.arrive_s, x.admit_s, x.first_token_s, x.end_s,
                  (long long)x.isl, (long long)x.osl, (long long)x.computed_prefill_tokens, x.delivery_ms);
    s += buf;
    for (float f : x.itl_ms) {
      std::snprintf(buf, sizeof buf, "%a,", double(f));
      s += buf;
    }
    for (const auto& kv : x.tier_hits) s += kv.first + "=" + std::to_string(kv.second) + ";";
    s += "\n";
  }
  for (const auto& kv : r.extra) {
    std::snprintf(buf, sizeof buf, "%s=%a\n", kv.first.c_str(), kv.second);
    s += buf;
  }
  return s;
}

// Sequential single-agent session: growing context with a compaction seam and a fork.
trace::Trace sequential_trace(const std::string& id, int turns, uint64_t salt) {
  TraceBuilder b(id);
  std::vector<uint64_t> ctx = chain({}, salt, 40);
  double t = 0;
  int prev = -1;
  for (int k = 0; k < turns; ++k) {
    int64_t out = 30 + (k % 5) * 10;
    if (k == turns / 2) ctx = chain(std::vector<uint64_t>(ctx.begin(), ctx.begin() + 20), salt + 7, 15);
    if (k == turns / 3) {
      std::vector<uint64_t> fork = chain(std::vector<uint64_t>(ctx.begin(), ctx.begin() + 30), salt + 99, 12);
      int r = b.req(0, t, 0.5, fork, 20);
      if (prev >= 0) b.edge(prev, r, trace::EdgeKind::Sequential, 0.2);
      prev = r;
      t += 1.0;
    }
    int r = b.req(0, t, 0.5, ctx, out);
    if (prev >= 0) b.edge(prev, r, trace::EdgeKind::Sequential, 0.2);
    prev = r;
    ctx = chain(ctx, salt, 6 + k % 3);
    t += 1.0;
  }
  return b.t;
}

void test_sequential_cache_and_ttft() {
  trace::TraceSet ts;
  ts.traces.push_back(sequential_trace("seq", 12, 11));
  const double a = 0.01, dstep = 2.0, fixed = 0.5;
  auto factory = [&](const PoolSpec&) { return std::make_unique<FakeStep>(a, dstep, fixed, 1.0); };
  StackSpec st = stack_default();
  st.chunk_tokens = 1000;
  st.max_num_batched_tokens = 1000;
  SimResult r = simulate_with(ts, devices(1e9, 1.0), FabricSpec{}, agg(1), st, run_spec(1, 400, 3), factory);
  int64_t hit = 0, total = 0;
  int n1 = 0;
  for (const auto& x : r.records) {
    assert(x.end_s >= x.first_token_s && x.first_token_s >= x.arrive_s);
    assert(int64_t(x.itl_ms.size()) == std::max<int64_t>(0, x.osl - 1));
    if (x.play != 1) continue;
    n1++;
    total += x.isl;
    hit += x.isl - x.computed_prefill_tokens;
    const double chunks = std::ceil(double(x.computed_prefill_tokens) / double(st.chunk_tokens));
    const double expect = chunks * fixed + double(x.computed_prefill_tokens) * a;
    assert(std::fabs(x.ttft_ms() - expect) < 1e-6);
    assert(std::fabs(x.admit_s - x.arrive_s) < 1e-12);
    const double e2e_expect = expect + double(x.osl - 1) * (dstep + fixed);
    assert(std::fabs(x.e2el_ms() - e2e_expect) < 1e-6);
  }
  assert(n1 == int(ts.traces[0].requests.size()));
  const double theory = theoretical_hit_rate(ts.traces[0]);
  const double got = double(hit) / double(total);
  std::printf("cache hit rate: engine %.6f theory %.6f\n", got, theory);
  assert(std::fabs(got - theory) < 1e-12);
  assert(r.time_share.at("capacity") == 0);
  // request_overhead_ms delays admission by a constant; TTFT and e2e shift by exactly that amount.
  st.request_overhead_ms = 250;
  SimResult ro = simulate_with(ts, devices(1e9, 1.0), FabricSpec{}, agg(1), st, run_spec(1, 400, 3), factory);
  int shifted = 0;
  for (const auto& x : ro.records) {
    if (x.play != 1) continue;
    assert(std::fabs(x.admit_s - x.arrive_s - 0.25) < 1e-9);
    shifted++;
  }
  assert(shifted == n1);
}

void test_determinism() {
  trace::TraceSet ts;
  ts.traces.push_back(sequential_trace("a", 8, 1));
  ts.traces.push_back(sequential_trace("b", 10, 2));
  ts.traces.push_back(sequential_trace("c", 6, 3));
  const FakeStep* last = nullptr;
  auto factory = [&](const PoolSpec&) {
    auto f = std::make_unique<FakeStep>(0.02, 3.0, 0.3, 1.0);
    last = f.get();
    return f;
  };
  StackSpec st = stack_default();
  st.mtp_nextn = 2;
  st.mtp_accept_dist = {0.3, 0.3, 0.4};
  MappingSpec m = agg(2, 2);
  m.routing = "kv_aware";
  SimResult r1 = simulate_with(ts, devices(5e5, 1.0), FabricSpec{}, m, st, run_spec(5, 60, 42), factory);
  assert(last && last->seen_nextn == 2);
  SimResult r2 = simulate_with(ts, devices(5e5, 1.0), FabricSpec{}, m, st, run_spec(5, 60, 42), factory);
  assert(dump(r1) == dump(r2));
  assert(r1.extra.at("warmup_requests") > 0);
  int completed = 0;
  for (const auto& x : r1.records) {
    assert(x.end_s > 0);
    assert(int64_t(x.itl_ms.size()) == std::max<int64_t>(0, x.osl - 1));
    completed++;
  }
  std::printf("determinism: %d records, passes %.0f rank_passes %.0f\n", completed, r1.extra.at("passes"),
              r1.extra.at("rank_passes"));
  for (const auto& x : r1.records) {
    for (const auto& y : r1.records)
      if (x.lane == y.lane && x.play == y.play) assert(x.worker == y.worker);
  }
}

void test_join_background_and_recycle() {
  TraceBuilder b("jb");
  std::vector<uint64_t> ctx = chain({}, 5, 10);
  int t1 = b.req(0, 0, 0, ctx, 5);
  int sa = b.agent(0), sb = b.agent(0, true);
  int ra = b.req(sa, 0, 0, chain(ctx, 21, 4), 5);
  int rb = b.req(sb, 0, 0, chain(ctx, 22, 4), 5000);
  int t2 = b.req(0, 0, 0, chain(ctx, 5, 8), 5);
  const double d_spawn_a = 0.5, d_spawn_b = 0.7, d_join = 2.0, d_seq = 0.1;
  b.edge(t1, ra, trace::EdgeKind::Spawn, d_spawn_a);
  b.edge(t1, rb, trace::EdgeKind::Background, d_spawn_b);
  b.edge(ra, t2, trace::EdgeKind::Join, d_join);
  b.edge(t1, t2, trace::EdgeKind::Sequential, d_seq);
  trace::TraceSet ts;
  ts.traces.push_back(b.t);
  auto factory = [&](const PoolSpec&) { return std::make_unique<FakeStep>(0.01, 1.0, 0.1, 1.0); };
  SimResult r = simulate_with(ts, devices(1e8, 1.0), FabricSpec{}, agg(1), stack_default(), run_spec(1, 30, 1), factory);
  const RequestRecord* rec[4] = {nullptr, nullptr, nullptr, nullptr};
  const RequestRecord* next_play = nullptr;
  for (const auto& x : r.records) {
    if (x.play == 0) rec[x.request] = &x;
    else if (x.play == 1 && x.request == t1) next_play = &x;
  }
  for (int i = 0; i < 4; ++i) assert(rec[i] && !rec[i]->warmup);
  assert(std::fabs(rec[ra]->arrive_s - (rec[t1]->end_s + d_spawn_a)) < 1e-9);
  assert(std::fabs(rec[rb]->arrive_s - (rec[t1]->end_s + d_spawn_b)) < 1e-9);
  const double expect_t2 = std::max(rec[t1]->end_s + d_seq, rec[ra]->end_s + d_join);
  assert(std::fabs(rec[t2]->arrive_s - expect_t2) < 1e-9);
  assert(rec[rb]->end_s > rec[t2]->end_s);
  assert(next_play && next_play->arrive_s >= rec[rb]->end_s);
  std::printf("join/background: t2 at %.4f (join gate %.4f), background ends %.4f, recycle at %.4f\n", rec[t2]->arrive_s,
              rec[ra]->end_s + d_join, rec[rb]->end_s, next_play->arrive_s);
}

void test_idle_shift() {
  TraceBuilder b("idle");
  std::vector<uint64_t> ctx = chain({}, 9, 10);
  int r1 = b.req(0, 0, 0, ctx, 5);
  int r2 = b.req(0, 0, 0, chain(ctx, 9, 3), 5);
  b.edge(r1, r2, trace::EdgeKind::Sequential, 1000.0);
  trace::TraceSet ts;
  ts.traces.push_back(b.t);
  auto factory = [&](const PoolSpec&) { return std::make_unique<FakeStep>(0.01, 1.0, 0.1, 1.0); };
  RunSpec rs = run_spec(1, 50, 1);
  SimResult r = simulate_with(ts, devices(1e8, 1.0), FabricSpec{}, agg(1), stack_default(), rs, factory);
  const RequestRecord *a = nullptr, *c = nullptr;
  for (const auto& x : r.records) {
    if (x.play == 0 && x.request == r1) a = &x;
    if (x.play == 0 && x.request == r2) c = &x;
  }
  assert(a && c);
  assert(std::fabs(c->arrive_s - (a->end_s + rs.idle_shift_s)) < 1e-9);
  assert(r.extra.at("idle_shifts") >= 1);
  // Per-tree idle cap: with two lanes the system is never globally idle, so only the tree cap can bound the 1000 s gap.
  ts.traces.push_back(b.t);
  RunSpec two = run_spec(2, 50, 1);
  two.trace_idle_gap_cap_s = 7;
  two.idle_shift_s = 1e9;
  SimResult rr = simulate_with(ts, devices(1e8, 1.0), FabricSpec{}, agg(1), stack_default(), two, factory);
  a = c = nullptr;
  for (const auto& x : rr.records) {
    if (x.lane == 0 && x.play == 0 && x.request == r1) a = &x;
    if (x.lane == 0 && x.play == 0 && x.request == r2) c = &x;
  }
  assert(a && c);
  assert(std::fabs(c->arrive_s - (a->end_s + two.trace_idle_gap_cap_s)) < 1e-9);
  assert(rr.extra.at("tree_idle_jumps") >= 1);
}

void test_capacity_and_preemption() {
  trace::TraceSet ts;
  {
    TraceBuilder b("cap");
    b.req(0, 0, 0, chain({}, 31, 16), 500);
    ts.traces.push_back(b.t);
  }
  {
    TraceBuilder b("cap2");
    b.req(0, 0, 0, chain({}, 32, 16), 500);
    ts.traces.push_back(b.t);
  }
  auto factory = [&](const PoolSpec&) { return std::make_unique<FakeStep>(0.01, 1.0, 0.1, 1.0); };
  SimResult r = simulate_with(ts, devices(2200, 1.0), FabricSpec{}, agg(1), stack_default(), run_spec(2, 20, 7), factory);
  int done = 0;
  for (const auto& x : r.records) {
    if (x.play == 0 && x.end_s > 0) done++;
  }
  assert(done >= 2);
  std::printf("capacity: preemptions %.0f capacity share %.4f\n", r.extra.at("preemptions"), r.time_share.at("capacity"));
  assert(r.extra.at("preemptions") >= 1);
  assert(r.time_share.at("capacity") > 0);
  for (const auto& x : r.records) if (x.play == 0) assert(x.computed_prefill_tokens >= x.isl);
}

void test_pd_delivery() {
  trace::TraceSet ts;
  ts.traces.push_back(sequential_trace("pd", 6, 77));
  const double kv_bytes = 1000.0, alpha = 0.001, L = 1e9;
  auto factory = [&](const PoolSpec&) { return std::make_unique<FakeStep>(0.01, 1.0, 0.1, kv_bytes); };
  MappingSpec m;
  m.disaggregated = true;
  m.prefill.device = "dev";
  m.decode.device = "dev";
  FabricSpec f;
  f.scaleup.present = true;
  f.scaleup.alpha_s = alpha;
  f.scaleup.bandwidth_Bps = L;
  StackSpec st = stack_default();
  SimResult r = simulate_with(ts, devices(1e7, kv_bytes), f, m, st, run_spec(1, 100, 5), factory);
  int n1 = 0;
  for (const auto& x : r.records) {
    assert(x.end_s > 0);
    if (x.play != 1) continue;
    n1++;
    const double expect = alpha * 1e3 + double(x.isl) * kv_bytes / L * 1e3;
    const double chunks = std::ceil(double(x.computed_prefill_tokens) / double(st.chunk_tokens));
    const double prefill_ms = chunks * 0.1 + double(x.computed_prefill_tokens) * 0.01;
    if (x.osl > 1) {
      assert(std::fabs(x.delivery_ms - expect) < 1e-6);
      assert(x.prefill_worker == 0 && x.worker == 1);
      // The first token is stamped at decode admission: after the prefill and the KV delivery.
      assert(x.ttft_ms() >= prefill_ms + x.delivery_ms - 1e-6);
    } else {
      assert(std::fabs(x.ttft_ms() - prefill_ms) < 1e-6);
    }
  }
  assert(n1 == int(ts.traces[0].requests.size()));
  assert(r.extra.at("deliveries") > 0);
  assert(r.bulk_overlap_fraction >= 0);
  std::printf("P/D: deliveries %.0f bulk overlap %.4f\n", r.extra.at("deliveries"), r.bulk_overlap_fraction);
}

// Disaggregated decode admission under a per-rank cap of one: the continuing sequence keeps its slot every pass
// (ITL stays one step) and the newcomer waits at the decode worker, so its wait lands in TTFT.
void test_pd_decode_cap() {
  trace::TraceSet ts;
  ts.traces.push_back(sequential_trace("cap", 4, 91));
  auto factory = [&](const PoolSpec&) { return std::make_unique<FakeStep>(0.01, 1.0, 0.1, 1000.0); };
  MappingSpec m;
  m.disaggregated = true;
  m.prefill.device = "dev";
  m.decode.device = "dev";
  FabricSpec f;
  f.scaleup.present = true;
  f.scaleup.alpha_s = 0.001;
  f.scaleup.bandwidth_Bps = 1e9;
  StackSpec st = stack_default();
  st.max_num_seqs = 1;
  st.mtp_nextn = 0;
  st.overlap_bulk = true;   // keep KV deliveries from blocking the decode worker; the check is the admission order
  SimResult r = simulate_with(ts, devices(1e7, 1000.0), f, m, st, run_spec(2, 100, 5), factory);
  int n = 0;
  bool waited = false;
  for (const auto& x : r.records) {
    if (x.osl <= 1 || x.end_s <= 0) continue;
    n++;
    const double itl = (x.end_s - x.first_token_s) * 1e3 / double(x.osl - 1);
    assert(std::fabs(itl - 1.1) < 1e-6);
    if (x.ttft_ms() > 5.0) waited = true;
  }
  assert(n > 0 && waited);
  std::printf("P/D decode cap: %d decoded requests, newcomer waited\n", n);
}

// Two unrelated contexts visited alternately: the idle one is evicted to host and reloaded on its next turn.
trace::Trace alternating_trace(const std::string& id, int turns) {
  TraceBuilder b(id);
  std::vector<uint64_t> a = chain({}, 501, 50), c = chain({}, 502, 50);
  int prev = -1;
  double t = 0;
  for (int k = 0; k < turns; ++k) {
    std::vector<uint64_t>& ctx = (k % 2 == 0) ? a : c;
    int r = b.req(0, t, 0.5, ctx, 30);
    if (prev >= 0) b.edge(prev, r, trace::EdgeKind::Sequential, 0.2);
    prev = r;
    ctx = chain(ctx, 500 + k % 2, 2);
    t += 1.0;
  }
  return b.t;
}

void test_offload_and_residency() {
  trace::TraceSet ts;
  ts.traces.push_back(alternating_trace("off", 12));
  auto factory = [&](const PoolSpec&) { return std::make_unique<FakeStep>(0.01, 1.0, 0.1, 1.0); };
  DeviceSet ds = devices(5000, 1.0);
  MemoryTier host;
  host.name = "host";
  host.capacity_bytes = 1e9;
  host.usable_fraction = 1.0;
  ds.devices["dev"].memory.push_back(host);
  FabricSpec f;
  f.host.present = true;
  f.host.alpha_s = 0.0001;
  f.host.bandwidth_Bps = 1e8;
  StackSpec st = stack_default();
  st.kv_offload = true;
  SimResult r = simulate_with(ts, ds, f, agg(1), st, run_spec(1, 200, 9), factory);
  for (const auto& x : r.records) assert(x.end_s > 0);
  std::printf("offload: tier loads %.0f writebacks %.0f hits hbm %lld host %lld\n", r.extra.at("tier_loads"),
              r.extra.at("writebacks"), (long long)r.tier_hits["hbm"], (long long)r.tier_hits["host"]);
  assert(r.extra.at("writebacks") > 0);
  assert(r.tier_hits["host"] > 0);
  st.residency_limit_s = 0.05;
  st.kv_offload = false;
  ts.traces[0] = sequential_trace("res", 10, 3);
  SimResult r2 = simulate_with(ts, devices(1e8, 1.0), FabricSpec{}, agg(1), st, run_spec(1, 100, 9), factory);
  for (const auto& x : r2.records) {
    if (x.play == 1) assert(x.computed_prefill_tokens == x.isl);
  }
}

// A one-block context whose node was offloaded, then revisited with an identical prefix: the full hit backs off one
// block to the parent in HBM while the deepest node is still on host. Before the fix that scheduled a zero-length
// load and re-admitted the request at the same instant forever (the livelock guard aborts the process).
trace::Trace full_hit_trace(const std::string& id, int turns) {
  TraceBuilder b(id);
  std::vector<uint64_t> big = chain({}, 601, 10), one = chain({}, 602, 1);
  int prev = -1;
  double t = 0;
  for (int k = 0; k < turns; ++k) {
    int r = b.req(0, t, 0.5, (k % 2 == 0) ? big : one, 2);
    if (prev >= 0) b.edge(prev, r, trace::EdgeKind::Sequential, 0.2);
    prev = r;
    t += 1.0;
  }
  return b.t;
}

void test_full_hit_on_offloaded_leaf() {
  trace::TraceSet ts;
  ts.traces.push_back(full_hit_trace("fullhit", 6));
  auto factory = [&](const PoolSpec&) { return std::make_unique<FakeStep>(0.01, 1.0, 0.1, 1.0); };
  DeviceSet ds = devices(11 * 64, 1.0);   // the ten-block context plus one block: each visit evicts the other one
  MemoryTier host;
  host.name = "host";
  host.capacity_bytes = 1e9;
  host.usable_fraction = 1.0;
  ds.devices["dev"].memory.push_back(host);
  FabricSpec f;
  f.host.present = true;
  f.host.alpha_s = 0.0001;
  f.host.bandwidth_Bps = 1e8;
  StackSpec st = stack_default();
  st.kv_offload = true;
  SimResult r = simulate_with(ts, ds, f, agg(1), st, run_spec(1, 20, 9), factory);
  int done = 0;
  for (const auto& x : r.records) if (x.end_s > 0) done++;
  std::printf("full hit: %d done, loads %.0f writebacks %.0f\n", done, r.extra.at("tier_loads"), r.extra.at("writebacks"));
  assert(done >= 6);
  assert(r.extra.at("writebacks") > 0);
}

void test_scale() {
  trace::TraceSet ts;
  for (int i = 0; i < 16; ++i) {
    TraceBuilder b("big" + std::to_string(i));
    std::vector<uint64_t> ctx = chain({}, 1000 + i, 400);
    int prev = -1;
    double t = 0;
    for (int k = 0; k < 60; ++k) {
      int r = b.req(0, t, 2.0, ctx, 300 + (k * 37) % 900);
      if (prev >= 0) b.edge(prev, r, trace::EdgeKind::Sequential, 4.0 + (k % 7) * 3.0);
      prev = r;
      ctx = chain(ctx, 1000 + i, 30 + k % 40);
      t += 8.0;
    }
    ts.traces.push_back(b.t);
  }
  auto factory = [&](const PoolSpec&) { return std::make_unique<FakeStep>(0.002, 25.0, 0.5, 1.0); };
  MappingSpec m = agg(10, 2);
  m.routing = "kv_aware";
  StackSpec st = stack_default();
  st.chunk_tokens = 16384;
  st.max_num_batched_tokens = 16384;
  st.max_num_seqs = 256;
  auto t0 = std::chrono::steady_clock::now();
  SimResult r = simulate_with(ts, devices(3e6, 1.0), FabricSpec{}, m, st, run_spec(1152, 1800, 2026), factory);
  const double wall = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  std::printf("scale: %.2f s wall, records %zu passes %.0f rank_passes %.0f preemptions %.0f capacity share %.3f\n", wall,
              r.records.size(), r.extra.at("passes"), r.extra.at("rank_passes"), r.extra.at("preemptions"),
              r.time_share.at("capacity"));
  assert(r.extra.at("passes") > 1e4);
  assert(wall < 60);
}

void test_screen() {
  trace::Stats s;
  s.traces = 1;
  s.total_requests = 100;
  s.total_input_tokens = 100 * 200000;
  s.total_output_tokens = 100 * 600;
  s.ideal_prefix_hit_rate = 0.98;
  DeviceSet ds = devices(1e7, 5620);
  PoolSpec p;
  p.device = "dev";
  p.workers = 2;
  p.tp = 4;
  p.moe_ep = 4;
  const double n = screen_sessions(ds.at("dev"), FabricSpec{}, p, stack_default(), s);
  const double expect = 1e7 * 5620 * 8 / (5620.0 * 200000.0);
  std::printf("screen: %.3f sessions (capacity term %.3f)\n", n, expect);
  assert(std::fabs(n - expect) / expect < 1e-9);
}

}  // namespace

int main() {
  test_sequential_cache_and_ttft();
  test_determinism();
  test_join_background_and_recycle();
  test_idle_shift();
  test_capacity_and_preemption();
  test_pd_delivery();
  test_pd_decode_cap();
  test_offload_and_residency();
  test_full_hit_on_offloaded_leaf();
  test_screen();
  test_scale();
  std::puts("engine_test ok");
  return 0;
}
