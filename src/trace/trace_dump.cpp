#include "trace/trace_dump.h"

#include <algorithm>
#include <cstdio>
#include <tuple>
#include <vector>

namespace dlsim::trace {
namespace {

std::string quote(const std::string& s) {
  std::string out = "\"";
  for (char c : s) {
    if (c == '"' || c == '\\') out += '\\';
    out += c;
  }
  return out + "\"";
}

std::string num(double v) {
  char buf[64];
  std::snprintf(buf, sizeof(buf), "%.6f", v);
  return buf;
}

const char* edge_kind(EdgeKind k) {
  switch (k) {
    case EdgeKind::Sequential: return "seq";
    case EdgeKind::Spawn: return "spawn";
    case EdgeKind::Join: return "join";
    case EdgeKind::Background: return "background";
  }
  return "";
}

}  // namespace

std::string dump_graph_json(const Trace& t) {
  std::vector<std::string> sid(t.agents.size());
  for (size_t i = 0; i < t.agents.size(); ++i) sid[i] = t.agents[i].label.empty() ? t.id : t.id + "::" + t.agents[i].label;

  std::vector<int32_t> k_of(t.requests.size());
  std::vector<std::vector<uint32_t>> reqs_of(t.agents.size());
  for (uint32_t i = 0; i < t.requests.size(); ++i) {
    k_of[i] = static_cast<int32_t>(reqs_of[t.requests[i].agent].size());
    reqs_of[t.requests[i].agent].push_back(i);
  }

  std::vector<size_t> order(t.agents.size());
  for (size_t i = 0; i < order.size(); ++i) order[i] = i;
  std::sort(order.begin(), order.end(), [&](size_t a, size_t b) { return sid[a] < sid[b]; });

  std::string out = "{\"id\":" + quote(t.id) + ",\"agents\":[";
  bool first_agent = true;
  for (size_t a : order) {
    if (!first_agent) out += ",";
    first_agent = false;
    out += "{\"sid\":" + quote(sid[a]) + ",\"background\":" + (t.agents[a].background ? "true" : "false") + ",\"requests\":[";
    bool first_req = true;
    for (uint32_t i : reqs_of[a]) {
      const Request& r = t.requests[i];
      if (!first_req) out += ",";
      first_req = false;
      out += "[" + num(r.t_s) + "," + std::to_string(r.in_tokens) + "," + std::to_string(r.out_tokens) + "," +
             std::to_string(r.hash_count) + "," + quote(t.models[r.model]) + "]";
    }
    out += "]}";
  }

  using Row = std::tuple<std::string, int32_t, std::string, int32_t, std::string, double>;
  std::vector<Row> rows;
  for (const Edge& e : t.edges)
    rows.emplace_back(sid[t.requests[e.from].agent], k_of[e.from], sid[t.requests[e.to].agent], k_of[e.to], edge_kind(e.kind), e.delay_s);
  std::sort(rows.begin(), rows.end());
  out += "],\"edges\":[";
  bool first_edge = true;
  for (const Row& r : rows) {
    if (!first_edge) out += ",";
    first_edge = false;
    out += "[" + quote(std::get<0>(r)) + "," + std::to_string(std::get<1>(r)) + "," + quote(std::get<2>(r)) + "," +
           std::to_string(std::get<3>(r)) + "," + quote(std::get<4>(r)) + "," + num(std::get<5>(r)) + "]";
  }
  return out + "]}";
}

}  // namespace dlsim::trace
