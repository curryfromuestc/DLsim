#pragma once
// AgentX Weka trace: loading, binary cache, session dependency graph.
// Design: workload-and-metrics.md (format, replay semantics), state-engine.md (dependency graph).
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace dlsim::trace {

enum class AgentKind : uint8_t { Main, Subagent, FlatIndependent, FlatWorkerGroup, FlatAux };

struct Agent {
  int32_t parent = -1;        // index into Trace::agents; -1 for the main agent
  AgentKind kind = AgentKind::Main;
  std::string label;          // subagent_type or classification label
  bool background = false;    // subagent without a later parent turn: parent does not wait
  int32_t chain_index = 0;    // 0 = main chain of its entry; >0 = spawned chain detected inside a subagent (parent is that subagent)
};

struct Request {
  int32_t agent = 0;          // index into Trace::agents
  double t_s = 0;             // recorded start, session-relative seconds
  double api_time_s = 0;      // recorded service time
  int64_t in_tokens = 0;      // hash_count * block_size
  int64_t out_tokens = 0;
  uint32_t hash_begin = 0;    // slice into Trace::hash_ids
  uint32_t hash_count = 0;
  uint16_t model = 0;         // index into Trace::models
  bool streaming = false;
  int32_t outer_idx = -1;     // index of the top-level entry in the session file; the marker index for subagent inner requests
};

enum class EdgeKind : uint8_t { Sequential, Spawn, Join, Background };

struct Edge {
  int32_t from = -1;
  int32_t to = -1;
  EdgeKind kind = EdgeKind::Sequential;
  double delay_s = 0;         // recorded end-of-from to start-of-to gap
};

struct Trace {
  std::string id;
  int block_size = 64;
  std::vector<std::string> models;
  std::vector<Agent> agents;
  std::vector<Request> requests;   // in file order within each agent; global order by t_s available via edges
  std::vector<Edge> edges;
  std::vector<uint64_t> hash_ids;  // block ids; equal id at equal position means equal prefix
};

struct TraceSet {
  std::string source_path;
  uint64_t source_size = 0;   // byte size of source_path when parsed; cache invalidation key
  std::vector<Trace> traces;
};

struct Stats {
  int64_t traces = 0, main_turns = 0, subagent_groups = 0, subagent_inner_requests = 0;
  int64_t total_requests = 0, total_input_tokens = 0, total_output_tokens = 0;
  double ideal_prefix_hit_rate = 0;  // global time order, one seen set per trace file
};

TraceSet load_weka(const std::string& jsonl_path);
void save_cache(const TraceSet& ts, const std::string& cache_path);
std::optional<TraceSet> load_cache(const std::string& cache_path);
// Cache-through: parse once, then read the binary cache.
TraceSet load(const std::string& jsonl_path, const std::string& cache_dir);

Stats compute_stats(const TraceSet& ts);
double ideal_prefix_hit_rate(const Trace& t);

}  // namespace dlsim::trace
