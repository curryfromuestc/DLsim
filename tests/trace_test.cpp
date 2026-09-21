#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <string>

#include "trace/trace.h"
#include "trace/trace_dump.h"

using namespace dlsim::trace;
namespace fs = std::filesystem;

namespace {

const std::string kFixtures = std::string(DLSIM_SOURCE_DIR) + "/third_party/agentx-harness/tests/fixtures/";

bool near(double a, double b) { return std::fabs(a - b) < 1e-9; }

const Agent& agent_by_label(const Trace& t, const std::string& label) {
  for (const Agent& a : t.agents)
    if (a.label == label) return a;
  assert(false && "agent label not found");
  return t.agents[0];
}

int agent_index(const Trace& t, const std::string& label) { return static_cast<int>(&agent_by_label(t, label) - t.agents.data()); }

std::vector<int> requests_of(const Trace& t, int agent) {
  std::vector<int> out;
  for (size_t i = 0; i < t.requests.size(); ++i)
    if (t.requests[i].agent == agent) out.push_back(static_cast<int>(i));
  return out;
}

const Edge* find_edge(const Trace& t, int from, int to, EdgeKind kind) {
  for (const Edge& e : t.edges)
    if (e.from == from && e.to == to && e.kind == kind) return &e;
  return nullptr;
}

Trace load_one(const std::string& file) {
  TraceSet ts = load_weka(kFixtures + file);
  assert(ts.traces.size() == 1);
  return ts.traces[0];
}

void test_simple() {
  Trace t = load_one("weka_traces/simple.json");
  assert(t.id == "trace_simple");
  assert(t.block_size == 64);
  assert(t.agents.size() == 1 && t.agents[0].kind == AgentKind::Main && t.agents[0].parent == -1);
  assert(t.requests.size() == 2);
  assert(t.requests[0].hash_count == 3 && t.requests[1].hash_count == 4);
  assert(t.requests[0].in_tokens == 200 && t.requests[1].out_tokens == 40);
  assert(t.hash_ids.size() == 7 && t.hash_ids[3] == 1 && t.hash_ids[6] == 4);
  assert(t.models[t.requests[0].model] == "claude-opus-4-5-20251101");
  assert(t.edges.size() == 1 && t.edges[0].kind == EdgeKind::Sequential && near(t.edges[0].delay_s, 4.0));
  assert(near(ideal_prefix_hit_rate(t), 3.0 / 7.0));
}

void test_one_subagent() {
  Trace t = load_one("weka_traces/one_subagent.json");
  assert(t.agents.size() == 2);
  const Agent& sa = agent_by_label(t, "sa:agent_001");
  assert(sa.kind == AgentKind::Subagent && sa.parent == 0 && !sa.background && sa.chain_index == 0);
  std::vector<int> main = requests_of(t, 0), child = requests_of(t, 1);
  assert(main.size() == 2 && child.size() == 1);
  assert(near(t.requests[child[0]].t_s, 2.0));
  assert(t.requests[child[0]].outer_idx == 1 && t.requests[main[1]].outer_idx == 2);
  assert(t.models[t.requests[child[0]].model] == "claude-haiku-4-5-20251001");
  assert(t.edges.size() == 3);
  const Edge* seq = find_edge(t, main[0], main[1], EdgeKind::Sequential);
  const Edge* spawn = find_edge(t, main[0], child[0], EdgeKind::Spawn);
  const Edge* join = find_edge(t, child[0], main[1], EdgeKind::Join);
  assert(seq && near(seq->delay_s, 5.0));
  assert(spawn && near(spawn->delay_s, 1.0));
  assert(join && near(join->delay_s, 3.5));
}

void test_terminal_subagent() {
  Trace t = load_one("weka_traces/terminal_subagent.json");
  assert(t.agents.size() == 2);
  const Agent& sa = agent_by_label(t, "sa:agent_term");
  assert(sa.background && sa.kind == AgentKind::Subagent);
  std::vector<int> main = requests_of(t, 0), child = requests_of(t, 1);
  assert(main.size() == 1 && child.size() == 1);
  assert(near(t.requests[child[0]].t_s, 1.0));
  assert(t.edges.size() == 1);
  const Edge* bg = find_edge(t, main[0], child[0], EdgeKind::Background);
  assert(bg && near(bg->delay_s, 1.0));
}

void test_fanout() {
  Trace t = load_one("weka_traces_fanout/fanout.json");
  assert(t.agents.size() == 3);
  int fa = agent_index(t, "fa:000"), aux = agent_index(t, "aux:001");
  assert(t.agents[fa].kind == AgentKind::FlatIndependent && t.agents[fa].parent == 0 && !t.agents[fa].background);
  assert(t.agents[aux].kind == AgentKind::FlatAux && !t.agents[aux].background);
  std::vector<int> main = requests_of(t, 0), w0 = requests_of(t, fa), w1 = requests_of(t, aux);
  assert(main.size() == 3 && w0.size() == 2 && w1.size() == 1);
  assert(near(t.requests[main[1]].t_s, 9.0) && near(t.requests[main[2]].t_s, 12.0));
  assert(near(t.requests[w0[1]].t_s, 8.5) && near(t.requests[w1[0]].t_s, 2.5));
  assert(t.edges.size() == 7);
  assert(near(find_edge(t, main[0], main[1], EdgeKind::Sequential)->delay_s, 8.0));
  assert(near(find_edge(t, main[1], main[2], EdgeKind::Sequential)->delay_s, 2.0));
  assert(near(find_edge(t, w0[0], w0[1], EdgeKind::Sequential)->delay_s, 0.5));
  assert(near(find_edge(t, main[0], w0[0], EdgeKind::Spawn)->delay_s, 1.0));
  assert(near(find_edge(t, main[0], w1[0], EdgeKind::Spawn)->delay_s, 1.5));
  assert(near(find_edge(t, w1[0], main[1], EdgeKind::Join)->delay_s, 2.5));
  assert(near(find_edge(t, w0[1], main[2], EdgeKind::Join)->delay_s, 2.5));
  assert(near(ideal_prefix_hit_rate(t), 16.0 / 27.0));
}

void test_overlap_groups() {
  Trace t = load_one("weka_traces_overlap_groups/abc_join_d.json");
  assert(t.agents.size() == 3);
  int a0 = agent_index(t, "aux:000"), a1 = agent_index(t, "aux:001");
  std::vector<int> main = requests_of(t, 0), c0 = requests_of(t, a0), c1 = requests_of(t, a1);
  assert(main.size() == 2 && c0.size() == 1 && c1.size() == 1);
  assert(near(find_edge(t, main[0], main[1], EdgeKind::Sequential)->delay_s, 1.0));
  assert(near(find_edge(t, main[0], c0[0], EdgeKind::Spawn)->delay_s, 0.0));
  assert(near(find_edge(t, c0[0], main[1], EdgeKind::Join)->delay_s, 0.9));
  assert(near(find_edge(t, c1[0], main[1], EdgeKind::Join)->delay_s, 0.8));
}

void test_async_subagent_with_seams() {
  Trace t = load_one("weka_traces/async_subagent_with_parallel_inner.json");
  assert(t.agents.size() == 3);
  int sa = agent_index(t, "sa:codex_subagent_001"), fork = agent_index(t, "sa:codex_subagent_001:fa:000");
  assert(t.agents[sa].kind == AgentKind::Subagent && t.agents[sa].chain_index == 0);
  assert(t.agents[fork].kind == AgentKind::FlatIndependent && t.agents[fork].parent == sa && t.agents[fork].chain_index == 1);
  assert(!t.agents[sa].background && !t.agents[fork].background);
  std::vector<int> main = requests_of(t, 0), c0 = requests_of(t, sa), c1 = requests_of(t, fork);
  assert(main.size() == 7 && c0.size() == 1 && c1.size() == 1);
  assert(find_edge(t, main[3], c0[0], EdgeKind::Spawn) && find_edge(t, main[3], c1[0], EdgeKind::Spawn));
  assert(find_edge(t, c0[0], main[6], EdgeKind::Join) && find_edge(t, c1[0], main[6], EdgeKind::Join));
  assert(t.edges.size() == 6 + 4);
}

bool same(const TraceSet& a, const TraceSet& b) {
  if (a.source_path != b.source_path || a.source_size != b.source_size || a.traces.size() != b.traces.size()) return false;
  for (size_t i = 0; i < a.traces.size(); ++i) {
    const Trace &x = a.traces[i], &y = b.traces[i];
    if (x.id != y.id || x.block_size != y.block_size || x.models != y.models || x.hash_ids != y.hash_ids) return false;
    if (x.agents.size() != y.agents.size() || x.requests.size() != y.requests.size() || x.edges.size() != y.edges.size()) return false;
    for (size_t j = 0; j < x.agents.size(); ++j) {
      const Agent &p = x.agents[j], &q = y.agents[j];
      if (p.parent != q.parent || p.kind != q.kind || p.label != q.label || p.background != q.background || p.chain_index != q.chain_index) return false;
    }
    for (size_t j = 0; j < x.requests.size(); ++j) {
      const Request &p = x.requests[j], &q = y.requests[j];
      if (p.agent != q.agent || p.t_s != q.t_s || p.api_time_s != q.api_time_s || p.in_tokens != q.in_tokens || p.out_tokens != q.out_tokens ||
          p.hash_begin != q.hash_begin || p.hash_count != q.hash_count || p.model != q.model || p.streaming != q.streaming || p.outer_idx != q.outer_idx)
        return false;
    }
    for (size_t j = 0; j < x.edges.size(); ++j) {
      const Edge &p = x.edges[j], &q = y.edges[j];
      if (p.from != q.from || p.to != q.to || p.kind != q.kind || p.delay_s != q.delay_s) return false;
    }
    if (dump_graph_json(x) != dump_graph_json(y)) return false;
  }
  return true;
}

void test_cache_roundtrip() {
  fs::path dir = fs::temp_directory_path() / "dlsim_trace_test";
  fs::remove_all(dir);
  for (const char* f : {"weka_traces/one_subagent.json", "weka_traces_fanout/fanout.json", "weka_traces/async_subagent_with_parallel_inner.json"}) {
    TraceSet ts = load_weka(kFixtures + f);
    fs::create_directories(dir);
    save_cache(ts, (dir / "x.cache").string());
    auto back = load_cache((dir / "x.cache").string());
    assert(back && same(ts, *back));
    TraceSet through = load(kFixtures + f, dir.string());
    assert(same(ts, through));
    assert(fs::exists(dir / (fs::path(f).filename().string() + ".dlsimcache")));
    TraceSet again = load(kFixtures + f, dir.string());
    assert(same(ts, again));
  }
  assert(!load_cache((dir / "missing.cache").string()));
  fs::remove_all(dir);
}

void test_stats_on_fixtures() {
  TraceSet ts = load_weka(kFixtures + "weka_traces/async_subagent_with_parallel_inner.json");
  Stats s = compute_stats(ts);
  assert(s.traces == 1 && s.main_turns == 7 && s.subagent_groups == 1 && s.subagent_inner_requests == 2 && s.total_requests == 9);
  assert(s.total_input_tokens == 31107 + 31738 + 32271 + 32885 + 32992 + 37911 + 79425 + 149826 + 147046);
  assert(s.total_output_tokens == 618 + 441 + 526 + 94 + 73 + 277 + 3250 + 8654 + 9549);
  assert(near(s.ideal_prefix_hit_rate, ideal_prefix_hit_rate(ts.traces[0])));
}

int run_dump(const std::string& path, size_t limit) {
  TraceSet ts = load_weka(path);
  for (size_t i = 0; i < ts.traces.size() && i < limit; ++i) std::cout << dump_graph_json(ts.traces[i]) << "\n";
  return 0;
}

int run_stats(const std::string& path, const std::string& cache_dir) {
  using clock = std::chrono::steady_clock;
  auto t0 = clock::now();
  TraceSet ts = cache_dir.empty() ? load_weka(path) : load(path, cache_dir);
  auto t1 = clock::now();
  Stats s = compute_stats(ts);
  auto t2 = clock::now();
  std::printf("load_s %.3f stats_s %.3f\n", std::chrono::duration<double>(t1 - t0).count(), std::chrono::duration<double>(t2 - t1).count());
  std::printf("traces %lld\nmain_turns %lld\nsubagent_groups %lld\nsubagent_inner_requests %lld\ntotal_model_requests %lld\n"
              "total_input_tokens %lld\ntotal_output_tokens %lld\nideal_prefix_hit_rate %.6f\n",
              (long long)s.traces, (long long)s.main_turns, (long long)s.subagent_groups, (long long)s.subagent_inner_requests,
              (long long)s.total_requests, (long long)s.total_input_tokens, (long long)s.total_output_tokens, s.ideal_prefix_hit_rate);
  size_t agents = 0, edges = 0;
  int64_t kinds[5] = {0, 0, 0, 0, 0}, background = 0;
  for (const Trace& t : ts.traces) {
    agents += t.agents.size();
    edges += t.edges.size();
    for (const Agent& a : t.agents) {
      ++kinds[static_cast<int>(a.kind)];
      background += a.background;
    }
  }
  std::printf("agents %zu edges %zu main %lld subagent %lld flat_independent %lld flat_worker_group %lld flat_aux %lld background %lld\n", agents, edges,
              (long long)kinds[0], (long long)kinds[1], (long long)kinds[2], (long long)kinds[3], (long long)kinds[4], (long long)background);
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc >= 3 && std::strcmp(argv[1], "dump") == 0) return run_dump(argv[2], argc >= 4 ? std::stoul(argv[3]) : SIZE_MAX);
  if (argc >= 3 && std::strcmp(argv[1], "stats") == 0) return run_stats(argv[2], argc >= 4 ? argv[3] : "");
  test_simple();
  test_one_subagent();
  test_terminal_subagent();
  test_fanout();
  test_overlap_groups();
  test_async_subagent_with_seams();
  test_cache_roundtrip();
  test_stats_on_fixtures();
  std::puts("trace_test ok");
  return 0;
}
