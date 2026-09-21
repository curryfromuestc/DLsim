#include "engine/engine.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <deque>
#include <limits>
#include <queue>
#include <random>
#include <set>
#include <stdexcept>

#include "engine/harness_rng.h"
#include "fabric/fabric.h"

namespace dlsim {

const DeviceSpec& DeviceSet::at(const std::string& n) const {
  auto it = devices.find(n);
  if (it == devices.end()) throw std::runtime_error("unknown device '" + n + "'");
  return it->second;
}

namespace {

enum class RState : uint8_t { Pending, Queued, Loading, Prefill, Delivering, DecodeQueued, Decode, Done };
enum EvType : uint8_t { EvTransfer = 0, EvPass = 1, EvReady = 2, EvEnd = 3 };
enum XferKind : int32_t { XferLoad = 0, XferDeliver = 1, XferWriteback = 2 };

struct Event {
  double t = 0;
  uint8_t type = 0;
  uint64_t seq = 0;
  int32_t a = -1, b = -1;
};

struct EventLater {
  bool operator()(const Event& x, const Event& y) const {
    if (x.t != y.t) return x.t > y.t;
    if (x.type != y.type) return x.type > y.type;
    return x.seq > y.seq;
  }
};

struct Timer {
  double stored = 0;
  uint64_t seq = 0;
  int32_t req = -1;
  uint32_t ver = 0;   // stale when it no longer matches Req::timer_ver (per-tree idle cap re-schedules)
};

struct TimerLater {
  bool operator()(const Timer& x, const Timer& y) const {
    if (x.stored != y.stored) return x.stored > y.stored;
    return x.seq > y.seq;
  }
};

struct Graph {
  std::vector<int32_t> off, to;
  std::vector<double> delay;
  double t_first = 0, t_last = 0;  // request start-time bounds of the whole tree (harness snapshot window)
};

struct Node {
  int32_t play = -1, unit = -1;
  int32_t parent = -1, first_child = -1, next_sibling = -1, prev_sibling = -1;
  uint32_t hash_begin = 0, nblocks = 0;
  int64_t tail = 0;
  int32_t tier = 0, pins = 0;
  bool in_lru = false;
  double last_access = 0;
};

struct Req {
  int32_t play = -1, tr = -1, lane = -1;
  RState st = RState::Pending;
  bool warmup = false, skipped = false, loaded = false;
  bool primer = false;        // warmup primer: dispatched at t = 0 without dependencies
  bool pending_timer = false;
  uint32_t timer_ver = 0;
  int32_t preds_left = 0;
  double ready = 0, arrive = 0, due = 0;
  int32_t unit = -1, dec_unit = -1;
  int32_t hit_node = -1, dec_hit_node = -1, own_node = -1;
  uint32_t hit_blocks = 0, dec_hit_blocks = 0;
  int64_t isl = 0, osl = 0, hit_tokens = 0, dec_hit_tokens = 0, delivered = 0;
  int64_t prefill_left = 0, past = 0, generated = 0, alloc = 0, dec_alloc = 0;
  double last_tok = 0, xfer_req = 0;
  double due_base = 0;        // clock when the timer was armed (warmup phase): the hand-off keeps due - due_base
  int32_t rec = -1;
};

struct Play {
  int32_t trace = -1, lane = -1, index = -1;
  int32_t first_req = 0, nreq = 0, remaining = 0;
  int32_t in_flight = 0;      // requests between arrival and completion (the harness's per-tree in_flight)
  int block_size = 64;
  double t_star = 0;
  int32_t sticky_worker[2] = {-1, -1};   // router-history worker per pool for pools without block reuse
  std::vector<std::pair<int32_t, int32_t>> roots;
};

struct BatchEntry {
  int32_t req = -1;
  int64_t tokens = 0;
  bool prefill = false;
};

struct Unit {
  int32_t worker = -1, pool = -1;
  bool caches = false;
  std::vector<int64_t> cap, used;
  std::vector<std::set<std::pair<double, int32_t>>> lru;
  std::vector<double> tier_free_at;
  std::vector<int32_t> running;
  std::deque<int32_t> waiting, dec_waiting;
  int32_t loading = 0;
  int active() const { return int(running.size() + waiting.size() + dec_waiting.size()) + loading; }
};

struct Worker {
  int32_t pool = -1, id = -1, domain = 0;
  std::vector<int32_t> units;
  bool busy = false, cap_blocked = false;
  int32_t trees = 0;   // live trees pinned here by the router history (pools without block reuse)
  double blocked_until = 0, pass_start = 0, xfer_free_at = 0;
  StepInput in;
  std::vector<std::vector<BatchEntry>> batch;
  double compute_ms = 0, membw_ms = 0, comm_ms = 0, cap_ms = 0, busy_ms = 0;
  int64_t passes = 0, rank_passes = 0;
};

struct Pool {
  const PoolSpec* spec = nullptr;
  const DeviceSpec* dev = nullptr;
  std::unique_ptr<StepLatency> step;
  std::vector<int32_t> workers;
  int32_t rr = 0;
  bool cross_domain = false, caches = false;
  int ntiers = 1;
  std::vector<int64_t> tier_cap;
  std::vector<std::string> tier_link;
  double kv_bytes = 0;
};

struct Hit {
  int32_t node = -1;
  uint32_t blocks = 0, off = 0;
  bool lower = false;
};

class Engine {
 public:
  Engine(const trace::TraceSet& ts, const DeviceSet& devices, const FabricSpec& fabric, const MappingSpec& mapping,
         const StackSpec& stack, const RunSpec& run, const StepFactory& factory)
      : ts_(ts), fabric_(fabric), mapping_(mapping), stack_(stack), run_(run), rng_(run.seed) {
    if (ts.traces.empty()) throw std::runtime_error("simulate: no traces");
    if (run.concurrency < 1) throw std::runtime_error("simulate: concurrency must be >= 1");
    graphs_.reserve(ts.traces.size());
    for (const auto& t : ts.traces) graphs_.push_back(build_graph(t));
    int32_t dev_base = 0;
    if (mapping.disaggregated) {
      add_pool(mapping.prefill, devices, factory, dev_base, stack.prefix_cache);
      add_pool(mapping.decode, devices, factory, dev_base, stack.prefix_cache && stack.kv_delivery == "missing");
      if (pools_[0].kv_bytes != pools_[1].kv_bytes)
        throw std::runtime_error("P/D pools have different KV bytes per token; format conversion is unsupported");
    } else {
      add_pool(mapping.decode, devices, factory, dev_base, stack.prefix_cache);
    }
    if (stack.mtp_nextn > 0 && int(stack.mtp_accept_dist.size()) != stack.mtp_nextn + 1)
      throw std::runtime_error("mtp_accept_dist needs mtp_nextn + 1 entries");
  }

  SimResult run() {
    lane_plays_.assign(run_.concurrency, 0);
    for (int lane = 0; lane < run_.concurrency; ++lane) start_play(lane, true);
    for (const Worker& w : workers_) try_start_pass(w.id);
    while (!heap_.empty() || !warm_timers_.empty()) {
      if (!warm_timers_.empty() && (heap_.empty() || warm_timers_.top().stored <= heap_.top().t)) {
        Timer tm = warm_timers_.top();
        warm_timers_.pop();
        if (tm.ver != reqs_[tm.req].timer_ver) continue;
        reqs_[tm.req].pending_timer = false;
        advance(tm.stored);
        on_ready(tm.req);
        continue;
      }
      Event e = heap_.top();
      heap_.pop();
      advance(e.t);
      handle(e);
    }
    t0_ = now_;
    profiling_ = true;
    // Harness hand-off: every live stream carries its full next-turn delay into profiling (waiting for the global
    // warmup barrier does not consume it), the phase-wide earliest fires at profiling-time 0, and idle trees are capped.
    std::vector<int32_t> pending;
    double earliest = INFINITY;
    for (int32_t i = 0; i < int32_t(reqs_.size()); ++i)
      if (reqs_[i].pending_timer && !reqs_[i].warmup) {
        pending.push_back(i);
        earliest = std::min(earliest, reqs_[i].due - reqs_[i].due_base);
      }
    for (int32_t i : pending) push_timer(t0_ + (reqs_[i].due - reqs_[i].due_base) - earliest, i);
    for (int32_t p = 0; p < int32_t(plays_.size()); ++p) if (plays_[p].in_flight == 0) cap_tree_idle(p);
    push_event(t0_ + run_.duration_s, EvEnd, -1, -1);
    double stall_t = now_;
    uint64_t stall_n = 0;
    while (true) {
      bool have_main = !heap_.empty(), have_timer = !ended_ && !timers_.empty();
      if (!have_main && !have_timer) break;
      // Livelock guard: the clock must advance within a bounded number of events (same-time events are finite).
      if (now_ == stall_t) {
        if (++stall_n > 20000000) {
          std::fprintf(stderr, "livelock at t=%.6f: heap=%zu timers=%zu busy=%d active=%d bulk=%d top_type=%d\n", now_, heap_.size(),
                       timers_.size(), busy_workers_, active_reqs_, bulk_inflight_, heap_.empty() ? -1 : int(heap_.top().type));
          std::abort();
        }
      } else {
        stall_t = now_;
        stall_n = 0;
      }
      bool take_timer = false;
      if (have_timer) {
        const Timer& tm = timers_.top();
        double tt = tm.stored - timer_offset_;
        if (!have_main) take_timer = true;
        else {
          const Event& e = heap_.top();
          take_timer = tt < e.t || (tt == e.t && (EvReady < e.type || (EvReady == e.type && tm.seq < e.seq)));
        }
      }
      if (take_timer) {
        Timer tm = timers_.top();
        timers_.pop();
        if (tm.ver != reqs_[tm.req].timer_ver) continue;   // re-scheduled by the per-tree idle cap
        reqs_[tm.req].pending_timer = false;
        advance(tm.stored - timer_offset_);
        on_ready(tm.req);
      } else {
        Event e = heap_.top();
        heap_.pop();
        advance(e.t);
        handle(e);
      }
      if (ended_ && heap_.empty()) break;
      if (!ended_ && busy_workers_ == 0 && bulk_inflight_ == 0 && active_reqs_ == 0 && !timers_.empty()) {
        double next = timers_.top().stored - timer_offset_;
        if (next - now_ > run_.idle_shift_s) {
          double d = next - now_ - run_.idle_shift_s;
          timer_offset_ += d;
          idle_shifts_++;
          idle_skipped_s_ += d;
        }
      }
    }
    return finalize();
  }

 private:
  const trace::TraceSet& ts_;
  const FabricSpec& fabric_;
  const MappingSpec& mapping_;
  const StackSpec& stack_;
  const RunSpec& run_;
  std::mt19937_64 rng_;
  std::vector<Graph> graphs_;
  std::vector<Pool> pools_;
  std::vector<Worker> workers_;
  std::vector<Unit> units_;
  std::vector<Node> nodes_;
  std::vector<int32_t> free_nodes_;
  std::vector<Req> reqs_;
  std::vector<Play> plays_;
  std::vector<RequestRecord> records_;
  std::vector<int32_t> lane_plays_;
  size_t draw_ = 0;
  std::priority_queue<Event, std::vector<Event>, EventLater> heap_;
  std::priority_queue<Timer, std::vector<Timer>, TimerLater> timers_, warm_timers_;
  int64_t tree_idle_jumps_ = 0;
  double tree_idle_skipped_s_ = 0;
  double timer_offset_ = 0;
  uint64_t seq_ = 0;
  double now_ = 0, t0_ = 0, overlap_s_ = 0, idle_skipped_s_ = 0;
  bool profiling_ = false, ended_ = false;
  int32_t busy_workers_ = 0, bulk_inflight_ = 0, active_reqs_ = 0;
  int64_t unsupported_passes_ = 0;
  std::string unsupported_note_;
  std::set<std::string> unsupported_notes_;
  int64_t idle_shifts_ = 0, preemptions_ = 0, warmup_reqs_ = 0, loads_ = 0, deliveries_ = 0, writebacks_ = 0;

  double uni() { return double(rng_() >> 11) * 0x1.0p-53; }

  int sample_accept() {
    const auto& d = stack_.mtp_accept_dist;
    double u = uni(), acc = 0;
    for (size_t k = 0; k < d.size(); ++k) {
      acc += d[k];
      if (u < acc) return int(k);
    }
    return int(d.size()) - 1;
  }

  static Graph build_graph(const trace::Trace& t) {
    Graph g;
    const int32_t n = int32_t(t.requests.size());
    if (n == 0) throw std::runtime_error("trace '" + t.id + "' has no requests");
    std::vector<int32_t> cnt(n + 1, 0);
    for (const auto& e : t.edges) {
      if (e.from < 0 || e.from >= n || e.to < 0 || e.to >= n) throw std::runtime_error("trace edge out of range");
      cnt[e.from + 1]++;
    }
    g.off.assign(n + 1, 0);
    for (int32_t i = 0; i < n; ++i) g.off[i + 1] = g.off[i] + cnt[i + 1];
    g.to.resize(t.edges.size());
    g.delay.resize(t.edges.size());
    std::vector<int32_t> fill(g.off.begin(), g.off.end() - 1);
    for (const auto& e : t.edges) {
      int32_t k = fill[e.from]++;
      g.to[k] = e.to;
      g.delay[k] = std::max(0.0, e.delay_s);
    }
    bool first = true;
    for (const auto& r : t.requests) {
      g.t_first = first ? r.t_s : std::min(g.t_first, r.t_s);
      g.t_last = first ? r.t_s : std::max(g.t_last, r.t_s);
      first = false;
    }
    return g;
  }

  void add_pool(const PoolSpec& spec, const DeviceSet& devices, const StepFactory& factory, int32_t& dev_base,
                bool caches) {
    Pool p;
    p.spec = &spec;
    p.dev = &devices.at(spec.device);
    p.step = factory(spec);
    if (!p.step) throw std::runtime_error("step factory returned null");
    const MemoryPlan& plan = p.step->memory();
    p.kv_bytes = plan.kv_bytes_per_token;
    p.caches = caches;
    p.ntiers = stack_.kv_offload ? int(p.dev->memory.size()) : 1;
    for (int i = 0; i < p.ntiers; ++i) {
      const MemoryTier& tier = p.dev->memory[i];
      p.tier_cap.push_back(plan.kv_capacity_tokens(tier, *p.dev));
      p.tier_link.push_back(tier.name == "host" || tier.name == "remote" ? tier.name : std::string());
    }
    if (p.tier_cap[0] <= 0) throw std::runtime_error("device tier of '" + spec.device + "' holds no KV tokens");
    const int group = spec.gpus_per_worker();
    p.cross_domain = fabric_.scaleup_domain && group > *fabric_.scaleup_domain;
    if (p.cross_domain && !fabric_.scaleout.present)
      throw std::runtime_error("worker spans scale-up domains but no scale-out layer is configured");
    const int32_t pool_id = int32_t(pools_.size());
    for (int w = 0; w < spec.workers; ++w) {
      Worker wk;
      wk.pool = pool_id;
      wk.id = int32_t(workers_.size());
      wk.domain = fabric_.scaleup_domain ? (dev_base + w * group) / *fabric_.scaleup_domain : 0;
      wk.in.ranks.resize(spec.attention_dp);
      wk.in.nextn = stack_.mtp_nextn;   // the step model prices 1 + nextn tokens per decode sequence
      wk.batch.resize(spec.attention_dp);
      for (int r = 0; r < spec.attention_dp; ++r) {
        Unit u;
        u.worker = wk.id;
        u.pool = pool_id;
        u.caches = caches;
        u.cap = p.tier_cap;
        u.used.assign(p.ntiers, 0);
        u.lru.resize(p.ntiers);
        u.tier_free_at.assign(p.ntiers, 0.0);
        wk.units.push_back(int32_t(units_.size()));
        units_.push_back(std::move(u));
      }
      p.workers.push_back(wk.id);
      workers_.push_back(std::move(wk));
    }
    dev_base += spec.gpus();
    pools_.push_back(std::move(p));
  }

  void push_event(double t, uint8_t type, int32_t a, int32_t b) { heap_.push(Event{t, type, seq_++, a, b}); }
  void push_timer(double t, int32_t req) {
    Req& r = reqs_[req];
    r.due = t;
    r.due_base = now_;
    r.pending_timer = true;
    r.timer_ver++;
    if (r.warmup) warm_timers_.push(Timer{t, seq_++, req, r.timer_ver});
    else timers_.push(Timer{t + timer_offset_, seq_++, req, r.timer_ver});
  }

  // Per-tree idle cap (--trace-idle-gap-cap-seconds): once a tree has nothing in flight, its earliest pending timer is
  // advanced to at most cap seconds from now and the tree's other pending timers shift by the same amount.
  void cap_tree_idle(int32_t play) {
    if (run_.trace_idle_gap_cap_s <= 0 || !profiling_) return;
    const Play& p = plays_[play];
    double earliest = INFINITY;
    for (int32_t i = 0; i < p.nreq; ++i) {
      const Req& r = reqs_[p.first_req + i];
      if (r.pending_timer && !r.warmup) earliest = std::min(earliest, r.due);
    }
    if (!std::isfinite(earliest)) return;  // nothing pending: the tree ends or waits on in-flight successors
    const double d = earliest - now_ - run_.trace_idle_gap_cap_s;
    if (!(d > 0)) return;
    for (int32_t i = 0; i < p.nreq; ++i) {
      Req& r = reqs_[p.first_req + i];
      if (r.pending_timer && !r.warmup) push_timer(r.due - d, p.first_req + i);
    }
    tree_idle_jumps_++;
    tree_idle_skipped_s_ += d;
  }

  void advance(double t) {
    if (t < now_) t = now_;
    if (profiling_ && busy_workers_ > 0 && bulk_inflight_ > 0) overlap_s_ += t - now_;
    now_ = t;
  }

  const trace::Trace& trace_of(const Play& p) const { return ts_.traces[p.trace]; }
  const trace::Request& treq(const Req& r) const { return trace_of(plays_[r.play]).requests[r.tr]; }
  const uint64_t* hashes(const Req& r) const { return trace_of(plays_[r.play]).hash_ids.data() + treq(r).hash_begin; }
  const uint64_t* node_hashes(const Node& n) const { return trace_of(plays_[n.play]).hash_ids.data() + n.hash_begin; }
  int64_t node_tokens(const Node& n) const { return int64_t(n.nblocks) * plays_[n.play].block_size + n.tail; }
  const Pool& pool_of_unit(int32_t u) const { return pools_[units_[u].pool]; }
  const std::string& tier_name(int32_t u, int tier) const { return pool_of_unit(u).dev->memory[tier].name; }

  // ---- lanes and plays ----

  void start_play(int lane, bool initial) {
    Play p;
    p.trace = int32_t(draw_++ % ts_.traces.size());  // harness sequential sampler: dataset order, wrapping
    p.lane = lane;
    p.index = lane_plays_[lane]++;
    const trace::Trace& t = trace_of(p);
    const Graph& g = graphs_[p.trace];
    p.block_size = t.block_size;
    p.first_req = int32_t(reqs_.size());
    p.nreq = int32_t(t.requests.size());
    const int32_t play_id = int32_t(plays_.size());
    if (initial) {
      // Harness snapshot instant: uniform in [start + min_ratio * span, start + max_ratio * span] over the
      // request start times of the whole tree, drawn from the per-(trace, lane) seeded generator.
      const double span = g.t_last - g.t_first;
      const double lo = g.t_first + run_.t_star_min * span, hi = std::max(lo, g.t_first + run_.t_star_max * span);
      p.t_star = hi == lo ? lo : lo + harness_uniform01(run_.seed, t.id, lane) * (hi - lo);
    }
    std::vector<uint8_t> cls(p.nreq, 0);  // 0 profiling, 1 warmup primer, 2 skipped, 3 warmup follow-on (no delays)
    if (initial) {
      std::vector<int32_t> last_before(t.agents.size(), -1);
      std::vector<uint8_t> has_after(t.agents.size(), 0);
      for (int32_t i = 0; i < p.nreq; ++i) {
        const auto& r = t.requests[i];
        if (r.t_s < p.t_star) last_before[r.agent] = i;
        else has_after[r.agent] = 1;
      }
      for (int32_t i = 0; i < p.nreq; ++i) {
        const auto& r = t.requests[i];
        if (r.t_s >= p.t_star) continue;
        cls[i] = (has_after[r.agent] && last_before[r.agent] == i) ? 1 : 2;
      }
      // Cache-pressure warmup: the next warmup_requests_per_lane requests after t* replay in dependency order without
      // think-time delays and are excluded from the metrics; profiling resumes from the hand-off state.
      int budget = run_.warmup_requests_per_lane;
      std::vector<int32_t> order(p.nreq);
      for (int32_t i = 0; i < p.nreq; ++i) order[i] = i;
      std::stable_sort(order.begin(), order.end(), [&](int32_t a, int32_t b) { return t.requests[a].t_s < t.requests[b].t_s; });
      for (int32_t i : order) {
        if (budget <= 0) break;
        if (cls[i] != 0) continue;
        cls[i] = 3;
        budget--;
      }
    }
    for (int32_t i = 0; i < p.nreq; ++i) {
      Req q;
      q.play = play_id;
      q.tr = i;
      q.lane = lane;
      q.warmup = cls[i] == 1 || cls[i] == 3;
      q.primer = cls[i] == 1;
      q.skipped = cls[i] == 2;
      q.isl = t.requests[i].in_tokens;
      q.osl = t.requests[i].out_tokens;
      reqs_.push_back(q);
    }
    int32_t remaining = 0;
    for (int32_t i = 0; i < p.nreq; ++i) if (!reqs_[p.first_req + i].skipped) remaining++;
    for (int32_t i = 0; i < p.nreq; ++i) {
      const Req& from = reqs_[p.first_req + i];
      for (int32_t k = g.off[i]; k < g.off[i + 1]; ++k) {
        Req& to = reqs_[p.first_req + g.to[k]];
        if (to.skipped || to.primer) continue;
        if (from.skipped) {
          if (!to.warmup) to.ready = std::max(to.ready, now_ + std::max(0.0, t.requests[g.to[k]].t_s - p.t_star));
        } else {
          to.preds_left++;
        }
      }
    }
    p.remaining = remaining;
    plays_.push_back(std::move(p));
    const Play& pp = plays_.back();
    for (int32_t i = 0; i < pp.nreq; ++i) {
      Req& q = reqs_[pp.first_req + i];
      const int32_t id = pp.first_req + i;
      if (q.skipped) continue;
      if (q.primer) {
        warmup_reqs_++;
        on_ready(id, false);
        continue;
      }
      if (q.preds_left == 0) {
        double at = q.warmup ? now_
                    : initial ? now_ + std::max(0.0, t.requests[i].t_s - pp.t_star)
                              : now_ + std::max(0.0, t.requests[i].t_s - g.t_first);
        if (q.warmup) warmup_reqs_++;
        push_timer(std::max(at, q.ready), id);
      }
    }
    if (pp.remaining == 0 && !ended_) start_play(lane, false);
  }

  void on_ready(int32_t id, bool kick = true) {
    Req& r = reqs_[id];
    r.st = RState::Queued;
    r.arrive = now_;
    plays_[r.play].in_flight++;
    RequestRecord rec;
    const Play& p = plays_[r.play];
    rec.trace = p.trace;
    rec.request = r.tr;
    rec.lane = r.lane;
    rec.play = p.index;
    rec.warmup = r.warmup;
    rec.arrive_s = now_;
    rec.isl = r.isl;
    rec.osl = r.osl;
    rec.itl_ms.reserve(size_t(std::max<int64_t>(0, r.osl - 1)));
    r.rec = int32_t(records_.size());
    records_.push_back(std::move(rec));
    r.prefill_left = r.isl;
    active_reqs_++;
    if (stack_.request_overhead_ms > 0) {  // pipeline latency before the request reaches a prefill queue
      push_event(now_ + stack_.request_overhead_ms / 1e3, EvReady, id, 0);
      return;
    }
    enqueue(id, kick);
  }

  void enqueue(int32_t id, bool kick) {
    Req& r = reqs_[id];
    const int32_t w = route(0, id);
    const int32_t u = pick_unit(w, id);
    r.unit = u;
    records_[r.rec].prefill_worker = w;
    if (!mapping_.disaggregated) records_[r.rec].worker = w;
    units_[u].waiting.push_back(id);
    if (kick) try_start_pass(w);
  }

  int32_t route(int32_t pool_id, int32_t id) {
    Pool& p = pools_[pool_id];
    if (mapping_.routing != "kv_aware" || p.workers.size() == 1) {
      int32_t w = p.workers[size_t(p.rr) % p.workers.size()];
      p.rr = int32_t((size_t(p.rr) + 1) % p.workers.size());
      return w;
    }
    const Req& r = reqs_[id];
    const trace::Request& tr = treq(r);
    if (!p.caches) {
      // No engine-side prefix index in this pool (block reuse off), but the Dynamo KV router predicts locality from its
      // own routing history and session affinity: a tree stays on the worker its earlier requests went to; a new tree
      // goes to the least loaded worker.
      int32_t& sticky = plays_[r.play].sticky_worker[pool_id];
      if (sticky >= 0) return sticky;
      int32_t best_w = -1;
      for (size_t k = 0; k < p.workers.size(); ++k) {   // fewest live trees, ties round-robin
        const int32_t w = p.workers[(size_t(p.rr) + k) % p.workers.size()];
        if (best_w < 0 || workers_[w].trees < workers_[best_w].trees) best_w = w;
      }
      p.rr = int32_t((size_t(p.rr) + 1) % p.workers.size());
      workers_[best_w].trees++;
      sticky = best_w;
      return best_w;
    }
    double best = 0;
    int32_t best_w = -1;
    for (int32_t w : p.workers) {
      const Worker& wk = workers_[w];
      uint32_t reusable = 0;
      int decode_load = 0, active = 0;
      for (int32_t u : wk.units) {
        const Unit& un = units_[u];
        if (un.caches) reusable = std::max(reusable, lookup(u, r.play, tr.hash_begin, tr.hash_count).blocks);
        active += un.active();
        for (int32_t x : un.running) if (reqs_[x].st == RState::Decode) decode_load++;
      }
      double cost = mapping_.w_prefill * double(std::max<int64_t>(0, int64_t(tr.hash_count) - reusable)) +
                    mapping_.w_decode * decode_load + mapping_.w_active * active;
      if (best_w < 0 || cost < best) {
        best = cost;
        best_w = w;
      }
    }
    return best_w;
  }

  // Attention-DP rank inside a worker: with the KV-aware router the rank holding the longest reusable prefix, ties
  // (and the round-robin router) go to the least loaded rank.
  int32_t pick_unit(int32_t w, int32_t id) {
    const Worker& wk = workers_[w];
    int32_t best = wk.units[0];
    uint32_t best_reuse = 0;
    const bool kv_aware = mapping_.routing == "kv_aware" && wk.units.size() > 1;
    if (kv_aware) {
      const Req& r = reqs_[id];
      const trace::Request& tr = treq(r);
      if (units_[best].caches) best_reuse = lookup(best, r.play, tr.hash_begin, tr.hash_count).blocks;
      for (int32_t u : wk.units) {
        const uint32_t reuse = units_[u].caches ? lookup(u, r.play, tr.hash_begin, tr.hash_count).blocks : 0;
        if (reuse > best_reuse || (reuse == best_reuse && units_[u].active() < units_[best].active())) {
          best = u;
          best_reuse = reuse;
        }
      }
      return best;
    }
    for (int32_t u : wk.units) if (units_[u].active() < units_[best].active()) best = u;
    return best;
  }

  // ---- KV radix tree ----

  int32_t new_node() {
    if (!free_nodes_.empty()) {
      int32_t id = free_nodes_.back();
      free_nodes_.pop_back();
      nodes_[id] = Node{};
      return id;
    }
    nodes_.push_back(Node{});
    return int32_t(nodes_.size() - 1);
  }

  int32_t root_of(int32_t play, int32_t unit, bool create) {
    Play& p = plays_[play];
    for (const auto& pr : p.roots) if (pr.first == unit) return pr.second;
    if (!create) return -1;
    int32_t id = new_node();
    Node& n = nodes_[id];
    n.play = play;
    n.unit = unit;
    n.pins = 1;
    n.last_access = now_;
    p.roots.emplace_back(unit, id);
    return id;
  }

  void lru_erase(Node& n, int32_t id) {
    if (!n.in_lru) return;
    units_[n.unit].lru[n.tier].erase({n.last_access, id});
    n.in_lru = false;
  }

  bool evictable(const Node& n) const {
    if (n.pins > 0 || n.parent == -1) return false;
    for (int32_t c = n.first_child; c != -1; c = nodes_[c].next_sibling) if (nodes_[c].tier <= n.tier) return false;
    return true;
  }

  void lru_update(int32_t id) {
    Node& n = nodes_[id];
    const bool ev = evictable(n);
    if (n.in_lru && !ev) lru_erase(n, id);
    else if (!n.in_lru && ev) {
      units_[n.unit].lru[n.tier].insert({n.last_access, id});
      n.in_lru = true;
    }
  }

  void touch(int32_t id) {
    for (int32_t n = id; n != -1; n = nodes_[n].parent) {
      Node& x = nodes_[n];
      if (x.last_access >= now_) break;
      bool was = x.in_lru;
      if (was) lru_erase(x, n);
      x.last_access = now_;
      if (was) {
        units_[x.unit].lru[x.tier].insert({now_, n});
        x.in_lru = true;
      }
    }
  }

  void link_child(int32_t parent, int32_t child) {
    Node& p = nodes_[parent];
    Node& c = nodes_[child];
    c.parent = parent;
    c.prev_sibling = -1;
    c.next_sibling = p.first_child;
    if (p.first_child != -1) nodes_[p.first_child].prev_sibling = child;
    p.first_child = child;
    lru_update(parent);
  }

  void unlink(int32_t id) {
    Node& n = nodes_[id];
    if (n.prev_sibling != -1) nodes_[n.prev_sibling].next_sibling = n.next_sibling;
    else if (n.parent != -1) nodes_[n.parent].first_child = n.next_sibling;
    if (n.next_sibling != -1) nodes_[n.next_sibling].prev_sibling = n.prev_sibling;
    n.prev_sibling = n.next_sibling = -1;
  }

  bool subtree_pinned(int32_t id) const {
    if (nodes_[id].pins > 0) return true;
    for (int32_t c = nodes_[id].first_child; c != -1; c = nodes_[c].next_sibling) if (subtree_pinned(c)) return true;
    return false;
  }

  void free_subtree(int32_t id) {
    Node& n = nodes_[id];
    for (int32_t c = n.first_child; c != -1;) {
      int32_t nx = nodes_[c].next_sibling;
      free_subtree(c);
      c = nx;
    }
    lru_erase(n, id);
    units_[n.unit].used[n.tier] -= node_tokens(n);
    free_nodes_.push_back(id);
  }

  void remove_subtree(int32_t id) {
    int32_t parent = nodes_[id].parent;
    unlink(id);
    free_subtree(id);
    if (parent != -1) lru_update(parent);
  }

  // Removes a leaf whose tokens were already returned through Req::alloc accounting.
  void drop_node_unaccounted(int32_t id) {
    Node& n = nodes_[id];
    int32_t parent = n.parent;
    lru_erase(n, id);
    unlink(id);
    free_nodes_.push_back(id);
    if (parent != -1) lru_update(parent);
  }

  int32_t split(int32_t id, uint32_t off) {
    int32_t up_id = new_node();
    Node& up = nodes_[up_id];
    Node& lo = nodes_[id];
    up.play = lo.play;
    up.unit = lo.unit;
    up.hash_begin = lo.hash_begin;
    up.nblocks = off;
    up.tier = lo.tier;
    up.last_access = lo.last_access;
    up.parent = lo.parent;
    up.prev_sibling = lo.prev_sibling;
    up.next_sibling = lo.next_sibling;
    if (lo.prev_sibling != -1) nodes_[lo.prev_sibling].next_sibling = up_id;
    else if (lo.parent != -1) nodes_[lo.parent].first_child = up_id;
    if (lo.next_sibling != -1) nodes_[lo.next_sibling].prev_sibling = up_id;
    lo.hash_begin += off;
    lo.nblocks -= off;
    lo.parent = up_id;
    lo.prev_sibling = lo.next_sibling = -1;
    up.first_child = id;
    return up_id;
  }

  Hit lookup(int32_t unit, int32_t play, uint32_t hb, uint32_t cnt) {
    Hit h;
    int32_t node = root_of(play, unit, false);
    if (node == -1) return h;
    const uint64_t* H = trace_of(plays_[play]).hash_ids.data() + hb;
    uint32_t pos = 0;
    while (pos < cnt) {
      int32_t found = -1;
      for (int32_t c = nodes_[node].first_child; c != -1;) {
        int32_t nx = nodes_[c].next_sibling;
        const Node& cn = nodes_[c];
        if (cn.nblocks > 0) {
          if (stack_.residency_limit_s > 0 && now_ - cn.last_access > stack_.residency_limit_s && !subtree_pinned(c)) {
            remove_subtree(c);
            c = nx;
            continue;
          }
          if (node_hashes(cn)[0] == H[pos]) {
            found = c;
            break;
          }
        }
        c = nx;
      }
      if (found == -1) break;
      const Node& fn = nodes_[found];
      const uint64_t* C = node_hashes(fn);
      uint32_t lo = 1, hi = std::min(fn.nblocks, cnt - pos);
      while (lo < hi) {
        uint32_t mid = (lo + hi + 1) / 2;
        if (C[mid - 1] == H[pos + mid - 1]) lo = mid;
        else hi = mid - 1;
      }
      pos += lo;
      node = found;
      h.off = lo;
      if (lo < fn.nblocks) break;
    }
    h.node = node;
    h.blocks = pos;
    if (nodes_[node].parent == -1) h.off = 0;
    for (int32_t n = node; n != -1; n = nodes_[n].parent) if (nodes_[n].tier > 0) h.lower = true;
    touch(node);
    return h;
  }

  // Splits at a partial match and pins the fully matched node; returns it (root when nothing matched).
  int32_t pin_hit(int32_t unit, int32_t play, Hit& h) {
    if (h.node == -1) h.node = root_of(play, unit, true);
    Node& n = nodes_[h.node];
    if (n.parent != -1 && h.off < n.nblocks) {
      h.node = split(h.node, h.off);
      h.off = nodes_[h.node].nblocks;
    }
    nodes_[h.node].pins++;
    lru_update(h.node);
    return h.node;
  }

  void unpin(int32_t id) {
    if (id == -1) return;
    nodes_[id].pins--;
    lru_update(id);
  }

  int32_t insert_child(int32_t parent, uint32_t hb, uint32_t nblocks) {
    if (nodes_[parent].tail > 0) {
      int32_t t = new_node();
      const Node& p = nodes_[parent];
      Node& tn = nodes_[t];
      tn.play = p.play;
      tn.unit = p.unit;
      tn.tail = p.tail;
      tn.tier = p.tier;
      tn.last_access = p.last_access;
      nodes_[parent].tail = 0;
      link_child(parent, t);
      lru_update(t);
    }
    int32_t id = new_node();
    Node& n = nodes_[id];
    n.play = nodes_[parent].play;
    n.unit = nodes_[parent].unit;
    n.hash_begin = hb;
    n.nblocks = nblocks;
    n.tier = 0;
    n.pins = 1;
    n.last_access = now_;
    link_child(parent, id);
    return id;
  }

  double tier_move_ms(int32_t unit, int tier, double bytes) const {
    const Pool& p = pool_of_unit(unit);
    const std::string& link = p.tier_link[tier];
    if (!link.empty()) {
      const Link& l = link == "host" ? fabric_.host : fabric_.remote;
      return l.present ? bulk_ms(link, bytes) : 0.0;
    }
    const double bw = p.dev->memory[tier].bandwidth_Bps;
    return bw > 0 ? bytes / bw * 1e3 : 0.0;
  }

  double bulk_ms(const std::string& link, double bytes) const {
    const double share = fabric_.bulk_share > 0 ? fabric_.bulk_share : 1.0;
    return bulk_transfer_ms(fabric_, link, bytes / share);
  }

  double queue_transfer(double& free_at, double ms, int32_t worker) {
    double start = std::max(now_, free_at);
    free_at = start + ms / 1e3;
    bulk_inflight_++;
    if (!stack_.overlap_bulk) workers_[worker].blocked_until = std::max(workers_[worker].blocked_until, free_at);
    return free_at;
  }

  bool ensure_free(int32_t unit, int tier, int64_t need) {
    Unit& u = units_[unit];
    while (u.used[tier] + need > u.cap[tier]) {
      auto& set = u.lru[tier];
      if (set.empty()) return false;
      int32_t victim = set.begin()->second;
      evict(victim);
    }
    return true;
  }

  void evict(int32_t id) {
    Node& n = nodes_[id];
    Unit& u = units_[n.unit];
    const Pool& p = pools_[u.pool];
    const int tier = n.tier;
    const int64_t tok = node_tokens(n);
    if (tier + 1 < p.ntiers && ensure_free(n.unit, tier + 1, tok)) {
      lru_erase(n, id);
      u.used[tier] -= tok;
      u.used[tier + 1] += tok;
      nodes_[id].tier = tier + 1;
      lru_update(id);
      lru_update(nodes_[id].parent);
      double ms = tier_move_ms(n.unit, tier + 1, double(tok) * p.kv_bytes);
      double end = queue_transfer(u.tier_free_at[tier + 1], ms, u.worker);
      push_event(end, EvTransfer, -1, XferWriteback);
      writebacks_++;
      return;
    }
    remove_subtree(id);
  }

  // ---- admission and passes ----

  enum class Admit { Ok, Blocked, Loading };

  Admit admit_prefill(int32_t unit, int32_t id, int64_t first_chunk) {
    Req& r = reqs_[id];
    Unit& u = units_[unit];
    const trace::Request& tr = treq(r);
    Hit h;
    if (u.caches) h = lookup(unit, r.play, tr.hash_begin, tr.hash_count);
    if (h.blocks > 0 && h.blocks == tr.hash_count) {
      h.blocks--;
      if (h.off > 0) h.off--;
      if (h.off == 0 && nodes_[h.node].parent != -1) {
        h.node = nodes_[h.node].parent;
        h.off = nodes_[h.node].nblocks;
        // The block left behind is recomputed, so a lower-tier node there must not schedule a load: with nothing to
        // move the load would complete at once and re-admission would return here forever.
        h.lower = false;
        for (int32_t n = h.node; n != -1; n = nodes_[n].parent) if (nodes_[n].tier > 0) h.lower = true;
      }
    }
    const int64_t hit_tokens = int64_t(h.blocks) * plays_[r.play].block_size;
    if (h.lower) {
      unpin(r.hit_node);
      r.hit_node = h.node;
      nodes_[h.node].pins++;
      lru_update(h.node);
      int64_t lower_tok = 0;
      for (int32_t n = h.node; n != -1; n = nodes_[n].parent) if (nodes_[n].tier > 0) lower_tok += node_tokens(nodes_[n]);
      if (!ensure_free(unit, 0, lower_tok)) return Admit::Blocked;
      double end = now_;
      RequestRecord& rec = records_[r.rec];
      std::vector<int64_t> per_tier(u.cap.size(), 0);
      for (int32_t n = h.node; n != -1; n = nodes_[n].parent) per_tier[nodes_[n].tier] += node_tokens(nodes_[n]);
      for (size_t t = 0; t < per_tier.size(); ++t) if (per_tier[t] > 0) rec.tier_hits[tier_name(unit, int(t))] += per_tier[t];
      for (int32_t n = h.node; n != -1; n = nodes_[n].parent) {
        Node& x = nodes_[n];
        if (x.tier == 0) continue;
        const int64_t tok = node_tokens(x);
        lru_erase(x, n);
        u.used[x.tier] -= tok;
        u.used[0] += tok;
        x.tier = 0;
      }
      for (int32_t n = h.node; n != -1; n = nodes_[n].parent) lru_update(n);
      for (size_t t = 1; t < per_tier.size(); ++t) {
        if (per_tier[t] == 0) continue;
        double ms = tier_move_ms(unit, int(t), double(per_tier[t]) * pool_of_unit(unit).kv_bytes);
        end = std::max(end, queue_transfer(u.tier_free_at[t], ms, u.worker));
      }
      r.loaded = true;
      r.st = RState::Loading;
      u.loading++;
      loads_++;
      push_event(end, EvTransfer, id, XferLoad);
      return Admit::Loading;
    }
    unpin(r.hit_node);
    r.hit_node = u.caches ? pin_hit(unit, r.play, h) : -1;
    int64_t chunk = std::min(first_chunk, r.isl - hit_tokens + r.generated);
    if (!ensure_free(unit, 0, chunk)) return Admit::Blocked;
    r.hit_blocks = h.blocks;
    r.hit_tokens = hit_tokens;
    r.past = hit_tokens;
    r.prefill_left = r.isl - hit_tokens + (r.generated > 0 ? r.generated : 0);
    RequestRecord& rec = records_[r.rec];
    if (!r.loaded && hit_tokens > 0) rec.tier_hits[tier_name(unit, 0)] += hit_tokens;
    rec.computed_prefill_tokens += r.prefill_left;
    if (rec.admit_s == 0 && r.generated == 0) rec.admit_s = now_;
    r.loaded = false;
    r.st = RState::Prefill;
    return Admit::Ok;
  }

  int64_t alloc(int32_t unit, int32_t id, int64_t tok) {
    units_[unit].used[0] += tok;
    reqs_[id].alloc += tok;
    return tok;
  }

  void preempt(int32_t unit, int32_t id) {
    Req& r = reqs_[id];
    Unit& u = units_[unit];
    int64_t kept = 0;
    if (r.own_node != -1) {
      Node& n = nodes_[r.own_node];
      if (n.pins > 1 || n.first_child != -1) {
        kept = node_tokens(n);
        n.pins--;
        lru_update(r.own_node);
      } else {
        drop_node_unaccounted(r.own_node);
      }
      r.own_node = -1;
    }
    u.used[0] -= r.alloc - kept;
    r.alloc = 0;
    unpin(r.hit_node);
    r.hit_node = -1;
    r.st = RState::Queued;
    r.prefill_left = r.isl + r.generated;
    r.past = 0;
    u.waiting.push_front(id);
    preemptions_++;
  }

  void try_start_pass(int32_t w) {
    Worker& wk = workers_[w];
    if (wk.busy || now_ < wk.blocked_until) return;
    if (!form_batch(w)) return;
    const Pool& p = pools_[wk.pool];
    StepResult res = p.step->step(wk.in);
    if (res.source == Source::Unsupported) {
      unsupported_passes_++;
      if (unsupported_notes_.size() < 5 && unsupported_notes_.insert(res.note).second)
        unsupported_note_ += (unsupported_note_.empty() ? "" : " | ") + res.note;
    }
    double total = res.total_ms;
    if (fabric_.bulk_share > 0 && fabric_.bulk_share < 1 && bulk_inflight_ > 0 && !stack_.overlap_comm)
      total += res.comm_ms * (1.0 / (1.0 - fabric_.bulk_share) - 1.0);
    if (profiling_ && now_ < t0_ + run_.duration_s) {
      wk.compute_ms += res.compute_ms;
      wk.membw_ms += res.membw_ms;
      wk.comm_ms += res.comm_ms;
      wk.busy_ms += total;
      if (wk.cap_blocked) wk.cap_ms += total;
    }
    // DLSIM_PASS_DEBUG=<pool>: every 40th pass of that pool prints its batch and the request states (diagnostics).
    static const char* dbg = std::getenv("DLSIM_PASS_DEBUG");
    if (dbg && wk.pool == std::atoi(dbg) && wk.passes % 40 == 0) {
      std::string sizes;
      for (const auto& b : wk.batch) { int64_t tk = 0; for (const auto& e : b) tk += e.tokens; sizes += std::to_string(b.size()) + "/" + std::to_string(tk) + ","; }
      int st[8] = {0};
      for (const Req& r : reqs_) if (r.st != RState::Done && r.st != RState::Pending) st[int(r.st)]++;
      int dq = 0, pw = 0;
      for (const Unit& u : units_) { dq += int(u.dec_waiting.size()); pw += int(u.waiting.size()); }
      std::fprintf(stderr, "pass pool=%d t=%.3f total_ms=%.3f compute=%.3f membw=%.3f comm=%.3f queued=%d loading=%d prefill=%d delivering=%d decqueued=%d decode=%d dec_waiting=%d pf_waiting=%d sizes=%s\n",
                   wk.pool, now_, total, res.compute_ms, res.membw_ms, res.comm_ms, st[1], st[2], st[3], st[4], st[5], st[6], dq, pw, sizes.c_str());
    }
    wk.busy = true;
    wk.pass_start = now_;
    wk.passes++;
    for (const auto& b : wk.batch) if (!b.empty()) wk.rank_passes++;
    busy_workers_++;
    push_event(now_ + total / 1e3, EvPass, w, -1);
  }

  bool form_batch(int32_t w) {
    Worker& wk = workers_[w];
    const int nextn = stack_.mtp_nextn;
    const int64_t chunk_cap = stack_.chunked_prefill ? stack_.chunk_tokens : std::numeric_limits<int64_t>::max();
    // Pool caps: the disaggregated prefill pool has its own recipe limits; decode and aggregated pools use the stack values.
    const bool pf_pool = mapping_.disaggregated && wk.pool == 0;
    const int seq_cap = pf_pool && stack_.prefill_max_seqs > 0 ? stack_.prefill_max_seqs : stack_.max_num_seqs;
    const int64_t tok_cap = pf_pool && stack_.prefill_max_tokens > 0 ? stack_.prefill_max_tokens : stack_.max_num_batched_tokens;
    bool any = false;
    wk.cap_blocked = false;
    for (size_t ri = 0; ri < wk.units.size(); ++ri) {
      const int32_t uid = wk.units[ri];
      Unit& u = units_[uid];
      auto& list = wk.in.ranks[ri];
      auto& batch = wk.batch[ri];
      list.clear();
      batch.clear();
      int64_t budget = tok_cap;
      int seqs = 0;
      auto add_prefill_chunk = [&](int32_t id) {
        Req& r = reqs_[id];
        int64_t chunk = std::min({r.prefill_left, chunk_cap, budget});
        if (!stack_.chunked_prefill && r.prefill_left > budget) chunk = list.empty() ? r.prefill_left : 0;
        if (chunk <= 0) return false;
        if (!ensure_free(uid, 0, chunk)) {
          wk.cap_blocked = true;
          return false;
        }
        alloc(uid, id, chunk);
        list.push_back(StepRequest{chunk, r.past, true});
        batch.push_back(BatchEntry{id, chunk, true});
        budget -= chunk;
        seqs++;
        return true;
      };
      auto add_decodes = [&]() {
        const int64_t need = 1 + nextn;
        for (size_t i = 0; i < u.running.size(); ++i) {
          int32_t id = u.running[i];
          Req& r = reqs_[id];
          if (r.st != RState::Decode) continue;
          if (budget < need || seqs >= seq_cap) break;
          if (mapping_.disaggregated) {
            list.push_back(StepRequest{1, r.past, false});
            batch.push_back(BatchEntry{id, need, false});
            budget -= need;
            seqs++;
            continue;
          }
          bool ok = ensure_free(uid, 0, need);
          while (!ok) {
            int32_t victim = -1;
            for (size_t j = u.running.size(); j-- > i + 1;) {
              if (reqs_[u.running[j]].st == RState::Decode) { victim = int32_t(j); break; }
            }
            if (victim == -1) break;
            preempt(uid, u.running[victim]);
            u.running.erase(u.running.begin() + victim);
            ok = ensure_free(uid, 0, need);
          }
          if (!ok) {
            wk.cap_blocked = true;
            continue;
          }
          alloc(uid, id, need);
          list.push_back(StepRequest{1, r.past, false});
          batch.push_back(BatchEntry{id, need, false});
          budget -= need;
          seqs++;
        }
        // Disaggregated decode admission fills the cap left after the continuing sequences (admitting first starved them).
        while (!u.dec_waiting.empty() && budget >= 1 + nextn && seqs < seq_cap) {
          int32_t id = u.dec_waiting.front();
          Req& r = reqs_[id];
          const int64_t need = r.delivered + std::max<int64_t>(0, r.osl - 1);
          if (!ensure_free(uid, 0, need)) {
            wk.cap_blocked = true;
            break;
          }
          u.dec_waiting.pop_front();
          u.used[0] += need;
          r.dec_alloc = need;
          r.st = RState::Decode;
          r.past = r.isl;
          if (records_[r.rec].first_token_s <= 0) {
            records_[r.rec].first_token_s = now_;
            r.last_tok = now_;
          }
          u.running.push_back(id);
          list.push_back(StepRequest{1, r.past, false});
          batch.push_back(BatchEntry{id, 1 + nextn, false});
          budget -= 1 + nextn;
          seqs++;
        }
      };
      auto add_prefills = [&]() {
        for (int32_t id : u.running) {
          if (reqs_[id].st != RState::Prefill) continue;
          if (budget <= 0 || seqs >= seq_cap) break;
          add_prefill_chunk(id);
        }
        while (!u.waiting.empty() && budget > 0 && seqs < seq_cap) {
          int32_t id = u.waiting.front();
          int64_t first_chunk = std::min(chunk_cap, budget);
          if (!stack_.chunked_prefill) first_chunk = std::numeric_limits<int64_t>::max();
          Admit a = admit_prefill(uid, id, first_chunk);
          if (a == Admit::Blocked) {
            wk.cap_blocked = true;
            break;
          }
          u.waiting.pop_front();
          if (a == Admit::Loading) continue;
          u.running.push_back(id);
          if (!add_prefill_chunk(id)) break;
        }
      };
      if (stack_.mix_prefill_decode) {
        add_decodes();
        add_prefills();
      } else {
        add_prefills();
        bool has_prefill = false;
        for (const auto& b : batch) if (b.prefill) has_prefill = true;
        if (!has_prefill) add_decodes();
      }
      if (!list.empty()) any = true;
    }
    return any;
  }

  void commit_pass(int32_t w) {
    Worker& wk = workers_[w];
    wk.busy = false;
    busy_workers_--;
    const bool pd = mapping_.disaggregated;
    const bool prefill_pool = pd && wk.pool == 0;
    for (size_t ri = 0; ri < wk.units.size(); ++ri) {
      Unit& u = units_[wk.units[ri]];
      for (const BatchEntry& b : wk.batch[ri]) {
        Req& r = reqs_[b.req];
        if (b.prefill) {
          r.prefill_left -= b.tokens;
          r.past += b.tokens;
          if (r.prefill_left > 0) continue;
          RequestRecord& rec = records_[r.rec];
          if (r.generated == 0) {
            // Disaggregated: the decode worker streams the response, so the client sees the first token at decode
            // admission (Dynamo runs the remote prefill first, then routes to a decode worker); a request that
            // completes in prefill is stamped here.
            if (!prefill_pool || r.osl <= 1) rec.first_token_s = now_;
            r.generated = 1;
            r.last_tok = now_;
          }
          if (u.caches) {
            const trace::Request& tr = treq(r);
            r.own_node = insert_child(r.hit_node, tr.hash_begin + r.hit_blocks, tr.hash_count - r.hit_blocks);
          }
          if (r.generated >= r.osl) {
            finish(b.req, wk.units[ri]);
          } else if (prefill_pool) {
            start_delivery(b.req);
          } else {
            r.st = RState::Decode;
          }
        } else {
          int accepted = stack_.mtp_nextn > 0 ? sample_accept() : 0;
          int64_t produced = std::min<int64_t>(1 + accepted, r.osl - r.generated);
          if (!pd) {
            int64_t unused = b.tokens - produced;
            u.used[0] -= unused;
            r.alloc -= unused;
          }
          RequestRecord& rec = records_[r.rec];
          const float itl = float((now_ - r.last_tok) * 1e3 / double(produced));
          for (int64_t k = 0; k < produced; ++k) rec.itl_ms.push_back(itl);
          r.last_tok = now_;
          r.generated += produced;
          r.past += produced;
          if (r.generated >= r.osl) finish(b.req, wk.units[ri]);
        }
      }
      size_t keep = 0;
      for (size_t i = 0; i < u.running.size(); ++i) {
        RState s = reqs_[u.running[i]].st;
        if (s == RState::Prefill || s == RState::Decode) u.running[keep++] = u.running[i];
      }
      u.running.resize(keep);
    }
  }

  void start_delivery(int32_t id) {
    Req& r = reqs_[id];
    const trace::Request& tr = treq(r);
    Pool& dp = pools_[1];
    const int32_t dw = route(1, id);
    const int32_t du = pick_unit(dw, id);
    r.dec_unit = du;
    records_[r.rec].worker = dw;
    Unit& u = units_[du];
    if (u.caches) {
      Hit h = lookup(du, r.play, tr.hash_begin, tr.hash_count);
      r.dec_hit_node = pin_hit(du, r.play, h);
      r.dec_hit_blocks = h.blocks;
      r.dec_hit_tokens = int64_t(h.blocks) * plays_[r.play].block_size;
      if (r.dec_hit_tokens > 0) records_[r.rec].tier_hits["decode:" + tier_name(du, 0)] += r.dec_hit_tokens;
    }
    r.delivered = stack_.kv_delivery == "full" ? r.isl : r.isl - r.dec_hit_tokens;
    const Worker& pw = workers_[units_[r.unit].worker];
    const Worker& dwk = workers_[dw];
    const bool same = pw.domain == dwk.domain;
    const Link& link = same ? fabric_.scaleup : fabric_.scaleout;
    if (!same && !fabric_.scaleout.present) throw std::runtime_error("P/D delivery crosses domains but no scale-out layer is configured");
    double ms = link.present ? bulk_ms(same ? "scaleup" : "scaleout", double(r.delivered) * dp.kv_bytes) : 0.0;
    double end = queue_transfer(workers_[dw].xfer_free_at, ms, dw);
    r.xfer_req = now_;
    r.st = RState::Delivering;
    deliveries_++;
    push_event(end, EvTransfer, id, XferDeliver);
  }

  void finish(int32_t id, int32_t unit) {
    Req& r = reqs_[id];
    Unit& u = units_[unit];
    RequestRecord& rec = records_[r.rec];
    rec.end_s = now_;
    r.st = RState::Done;
    active_reqs_--;
    if (mapping_.disaggregated && wk_pool(unit) == 1) {
      if (u.caches) {
        const trace::Request& tr = treq(r);
        int32_t n = insert_child(r.dec_hit_node, tr.hash_begin + r.dec_hit_blocks, tr.hash_count - r.dec_hit_blocks);
        nodes_[n].tail = r.generated - 1;
        nodes_[n].pins = 0;
        lru_update(n);
        unpin(r.dec_hit_node);
      } else {
        u.used[0] -= r.dec_alloc;
      }
      r.dec_hit_node = -1;
    } else if (mapping_.disaggregated) {
      release_prefill_side(id);
    } else {
      if (u.caches) {
        nodes_[r.own_node].tail = r.generated - 1;
        unpin(r.own_node);
        unpin(r.hit_node);
      } else {
        u.used[0] -= r.alloc;
      }
      r.own_node = r.hit_node = -1;
    }
    Play& p = plays_[r.play];
    p.remaining--;
    p.in_flight--;
    const Graph& g = graphs_[p.trace];
    for (int32_t k = g.off[r.tr]; k < g.off[r.tr + 1]; ++k) {
      Req& to = reqs_[p.first_req + g.to[k]];
      if (to.skipped || to.primer) continue;
      to.ready = std::max(to.ready, to.warmup ? now_ : now_ + g.delay[k]);
      if (--to.preds_left == 0 && !ended_) {
        if (to.warmup) warmup_reqs_++;
        push_timer(to.ready, p.first_req + g.to[k]);
      }
    }
    if (p.in_flight == 0) cap_tree_idle(r.play);
    if (p.remaining == 0) {
      for (int32_t w : p.sticky_worker) if (w >= 0) workers_[w].trees--;
      if (!ended_) start_play(p.lane, false);
    }
  }

  int32_t wk_pool(int32_t unit) const { return units_[unit].pool; }

  void release_prefill_side(int32_t id) {
    Req& r = reqs_[id];
    Unit& u = units_[r.unit];
    if (u.caches) {
      unpin(r.own_node);
      unpin(r.hit_node);
    } else {
      u.used[0] -= r.alloc;
    }
    r.own_node = r.hit_node = -1;
  }

  void handle(const Event& e) {
    switch (e.type) {
      case EvPass: {
        commit_pass(e.a);
        try_start_pass(e.a);
        break;
      }
      case EvTransfer: {
        bulk_inflight_--;
        if (e.b == XferWriteback) {
          for (auto& wk : workers_) if (wk.blocked_until <= now_) try_start_pass(wk.id);
          break;
        }
        Req& r = reqs_[e.a];
        if (e.b == XferLoad) {
          Unit& u = units_[r.unit];
          u.loading--;
          r.st = RState::Queued;
          u.waiting.push_front(e.a);
          try_start_pass(u.worker);
        } else {
          records_[r.rec].delivery_ms = (now_ - r.xfer_req) * 1e3;
          release_prefill_side(e.a);
          Unit& du = units_[r.dec_unit];
          r.st = RState::DecodeQueued;
          du.dec_waiting.push_back(e.a);
          try_start_pass(units_[r.unit].worker);
          try_start_pass(du.worker);
        }
        break;
      }
      case EvReady: {
        enqueue(e.a, true);
        break;
      }
      case EvEnd: {
        ended_ = true;
        while (!timers_.empty()) timers_.pop();
        break;
      }
      default: break;
    }
  }

  SimResult finalize() {
    SimResult res;
    res.records = std::move(records_);
    res.profiled_s = run_.duration_s;
    const double worker_s = run_.duration_s * double(workers_.size());
    double compute = 0, membw = 0, cap = 0, up = 0, out = 0, busy = 0;
    int64_t passes = 0, rank_passes = 0;
    for (const Worker& w : workers_) {
      compute += w.compute_ms;
      membw += w.membw_ms;
      cap += w.cap_ms;
      busy += w.busy_ms;
      (pools_[w.pool].cross_domain ? out : up) += w.comm_ms;
      passes += w.passes;
      rank_passes += w.rank_passes;
    }
    res.time_share["compute"] = compute / 1e3 / worker_s;
    res.time_share["membw"] = membw / 1e3 / worker_s;
    res.time_share["capacity"] = cap / 1e3 / worker_s;
    res.time_share["scaleup"] = up / 1e3 / worker_s;
    res.time_share["scaleout"] = out / 1e3 / worker_s;
    int64_t completed = 0, warm = 0;
    for (const RequestRecord& r : res.records) {
      if (r.warmup) { warm++; continue; }
      if (r.end_s > 0) completed++;
      for (const auto& kv : r.tier_hits) res.tier_hits[kv.first] += kv.second;
    }
    res.bulk_overlap_fraction = overlap_s_ / run_.duration_s;
    res.unlimited = unlimited_resources(fabric_);
    for (const Unit& u : units_) res.kv_pool_tokens += u.cap[0];
    res.extra["unsupported_passes"] = double(unsupported_passes_);
    if (unsupported_passes_ > 0) res.unsupported_note = unsupported_note_;
    res.extra["passes"] = double(passes);
    res.extra["rank_passes"] = double(rank_passes);
    res.extra["busy_share"] = busy / 1e3 / worker_s;
    res.extra["requests"] = double(res.records.size());
    res.extra["completed"] = double(completed);
    res.extra["warmup_requests"] = double(warm);
    res.extra["plays"] = double(plays_.size());
    res.extra["boundary_s"] = t0_;
    res.extra["end_s"] = now_;
    res.extra["idle_shifts"] = double(idle_shifts_);
    res.extra["idle_skipped_s"] = idle_skipped_s_;
    res.extra["tree_idle_jumps"] = double(tree_idle_jumps_);
    res.extra["tree_idle_skipped_s"] = tree_idle_skipped_s_;
    res.extra["preemptions"] = double(preemptions_);
    res.extra["tier_loads"] = double(loads_);
    res.extra["writebacks"] = double(writebacks_);
    res.extra["deliveries"] = double(deliveries_);
    res.extra["nodes"] = double(nodes_.size());
    return res;
  }
};

}  // namespace

SimResult simulate_with(const trace::TraceSet& traces, const DeviceSet& devices, const FabricSpec& fabric,
                        const MappingSpec& mapping, const StackSpec& stack, const RunSpec& run,
                        const StepFactory& step_factory) {
  Engine e(traces, devices, fabric, mapping, stack, run, step_factory);
  return e.run();
}

double screen_sessions(const DeviceSpec& d, const FabricSpec& f, const PoolSpec& pool, const StackSpec& stack,
                       const trace::Stats& workload, const ScreenModel& model, double request_interval_s) {
  if (workload.total_requests <= 0) throw std::runtime_error("screen_sessions: workload has no requests");
  if (request_interval_s <= 0) throw std::runtime_error("screen_sessions: request interval must be positive");
  const double in_per_req = double(workload.total_input_tokens) / double(workload.total_requests);
  const double out_per_req = double(workload.total_output_tokens) / double(workload.total_requests);
  const double computed = stack.prefix_cache ? in_per_req * (1.0 - workload.ideal_prefix_hit_rate) : in_per_req;
  const double tok_rate = (computed + out_per_req) / request_interval_s;
  const double dec_rate = out_per_req / request_interval_s;
  const double context = in_per_req;
  const double eta = 1.0;
  const double gpus = double(pool.gpus());
  double n = std::numeric_limits<double>::infinity();
  double peak_flops = 0;
  for (const auto& kv : d.flops) peak_flops = std::max(peak_flops, kv.second);
  if (model.flops_per_token > 0 && peak_flops > 0) n = std::min(n, peak_flops * gpus * eta / (model.flops_per_token * tok_rate));
  if (model.kv_bytes_per_token > 0) {
    const double bw = d.memory[0].bandwidth_Bps;
    const double b = dec_rate * model.kv_bytes_per_token * context;
    if (bw > 0 && b > 0) n = std::min(n, bw * gpus / b);
    double m_total = 0;
    const int ntiers = stack.kv_offload ? int(d.memory.size()) : 1;
    for (int i = 0; i < ntiers; ++i) m_total += d.memory[i].capacity_bytes * d.memory[i].usable_fraction;
    m_total = (m_total - model.weight_bytes_per_gpu - d.activation_reserve_bytes - d.comm_reserve_bytes) * gpus;
    n = std::min(n, std::max(0.0, m_total) / (model.kv_bytes_per_token * context));
  }
  if (model.comm_bytes_per_token > 0 && f.scaleup.present && f.scaleup.bandwidth_Bps) {
    const double l = model.comm_bytes_per_token * tok_rate;
    n = std::min(n, *f.scaleup.bandwidth_Bps * gpus / l);
  }
  return n;
}

double screen_sessions(const DeviceSpec& d, const FabricSpec& f, const PoolSpec& pool, const StackSpec& stack,
                       const trace::Stats& workload, const std::string& model) {
  if (model != "deepseek-v4-pro") throw std::runtime_error("screen_sessions: no screening constants for model '" + model + "'");
  ScreenModel m;
  m.kv_bytes_per_token = 5620;
  return screen_sessions(d, f, pool, stack, workload, m, 4.9);
}

}  // namespace dlsim
