#include "trace/trace.h"

#include <simdjson.h>

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <stdexcept>
#include <unordered_set>

#include "trace/weka_raw.h"

namespace dlsim::trace {
namespace {

using namespace simdjson;
using detail::RawRequest;
using detail::RawSubagent;
using detail::RawTrace;

uint16_t model_index(RawTrace& raw, std::string_view name) {
  for (size_t i = 0; i < raw.models.size(); ++i)
    if (raw.models[i] == name) return static_cast<uint16_t>(i);
  raw.models.emplace_back(name);
  return static_cast<uint16_t>(raw.models.size() - 1);
}

struct Entry {
  std::string type;
  RawRequest req;
  RawSubagent sa;
};

void parse_request_fields(ondemand::object obj, RawTrace& raw, Entry& e, bool allow_subagent) {
  for (auto field : obj) {
    std::string_view key = field.unescaped_key();
    ondemand::value val = field.value();
    if (key == "type") {
      e.type = std::string(std::string_view(val.get_string()));
    } else if (key == "t") {
      double t = val.get_double();
      e.req.t = t;
      e.sa.t = t;
    } else if (key == "model") {
      e.req.model = model_index(raw, std::string_view(val.get_string()));
    } else if (key == "in") {
      e.req.in_tokens = val.get_int64();
    } else if (key == "out") {
      e.req.out_tokens = val.get_int64();
    } else if (key == "hash_ids") {
      for (auto h : val.get_array()) e.req.hash_ids.push_back(static_cast<uint64_t>(int64_t(h.get_int64())));
    } else if (key == "api_time") {
      if (!bool(val.is_null())) e.req.api_time = val.get_double();
    } else if (key == "agent_id") {
      e.sa.agent_id = std::string(std::string_view(val.get_string()));
    } else if (key == "subagent_type") {
      e.sa.subagent_type = std::string(std::string_view(val.get_string()));
    } else if (key == "duration_ms") {
      if (!bool(val.is_null())) e.sa.duration_ms = val.get_double();
    } else if (key == "requests" && allow_subagent) {
      for (auto inner : val.get_array()) {
        Entry ie;
        parse_request_fields(inner.get_object(), raw, ie, false);
        e.sa.requests.push_back(std::move(ie.req));
      }
    }
  }
}

RawTrace parse_document(ondemand::document_reference doc) {
  RawTrace raw;
  for (auto field : doc.get_object()) {
    std::string_view key = field.unescaped_key();
    ondemand::value val = field.value();
    if (key == "id") {
      raw.id = std::string(std::string_view(val.get_string()));
    } else if (key == "models") {
      for (auto m : val.get_array()) model_index(raw, std::string_view(m.get_string()));
    } else if (key == "block_size") {
      raw.block_size = static_cast<int>(int64_t(val.get_int64()));
    } else if (key == "requests") {
      int32_t outer = 0;
      for (auto item : val.get_array()) {
        Entry e;
        parse_request_fields(item.get_object(), raw, e, true);
        if (e.type == "subagent") {
          e.sa.outer_idx = outer;
          raw.subagents.push_back(std::move(e.sa));
        } else {
          e.req.outer_idx = outer;
          e.req.streaming = e.type == "s";
          raw.normals.push_back(std::move(e.req));
        }
        ++outer;
      }
    }
  }
  return raw;
}

constexpr char kMagic[8] = {'D', 'L', 'S', 'I', 'M', 'T', 'R', '1'};

struct Writer {
  std::FILE* f;
  void raw(const void* p, size_t n) {
    if (std::fwrite(p, 1, n, f) != n) throw std::runtime_error("cache write failed");
  }
  template <class T>
  void pod(const T& v) { raw(&v, sizeof(T)); }
  void str(const std::string& s) {
    pod(static_cast<uint32_t>(s.size()));
    raw(s.data(), s.size());
  }
  template <class T>
  void vec(const std::vector<T>& v) {
    pod(static_cast<uint64_t>(v.size()));
    raw(v.data(), v.size() * sizeof(T));
  }
};

struct Reader {
  std::FILE* f;
  bool ok = true;
  void raw(void* p, size_t n) {
    if (ok && std::fread(p, 1, n, f) != n) ok = false;
  }
  template <class T>
  T pod() {
    T v{};
    raw(&v, sizeof(T));
    return v;
  }
  std::string str() {
    std::string s(pod<uint32_t>(), '\0');
    raw(s.data(), s.size());
    return s;
  }
  template <class T>
  std::vector<T> vec() {
    std::vector<T> v(pod<uint64_t>());
    raw(v.data(), v.size() * sizeof(T));
    return v;
  }
};

struct HitCount {
  int64_t hits = 0, total = 0;
};

HitCount count_prefix_hits(const Trace& t) {
  struct Key {
    double t;
    int32_t outer, chain, k;
    uint32_t req;
  };
  std::vector<Key> keys;
  keys.reserve(t.requests.size());
  int32_t prev_agent = -1, k = 0;
  for (uint32_t i = 0; i < t.requests.size(); ++i) {
    const Request& r = t.requests[i];
    k = r.agent == prev_agent ? k + 1 : 0;
    prev_agent = r.agent;
    keys.push_back({r.t_s, r.outer_idx, t.agents[r.agent].chain_index, k, i});
  }
  std::sort(keys.begin(), keys.end(), [](const Key& a, const Key& b) {
    if (a.t != b.t) return a.t < b.t;
    if (a.outer != b.outer) return a.outer < b.outer;
    if (a.chain != b.chain) return a.chain < b.chain;
    return a.k < b.k;
  });
  std::unordered_set<uint64_t> seen;
  HitCount c;
  for (const Key& key : keys) {
    const Request& r = t.requests[key.req];
    const uint64_t* h = t.hash_ids.data() + r.hash_begin;
    uint32_t hits = 0;
    while (hits < r.hash_count && seen.count(h[hits])) ++hits;
    c.hits += hits;
    c.total += r.hash_count;
    seen.insert(h, h + r.hash_count);
  }
  return c;
}

}  // namespace

TraceSet load_weka(const std::string& jsonl_path) {
  padded_string json = padded_string::load(jsonl_path);
  TraceSet ts;
  ts.source_path = jsonl_path;
  ts.source_size = json.size();
  ondemand::parser parser;
  ondemand::document_stream stream = parser.iterate_many(json, std::max<size_t>(json.size(), 1 << 20));
  for (auto doc : stream) ts.traces.push_back(detail::build_trace(parse_document(doc.value())));
  return ts;
}

void save_cache(const TraceSet& ts, const std::string& cache_path) {
  std::FILE* f = std::fopen(cache_path.c_str(), "wb");
  if (!f) throw std::runtime_error("cannot write " + cache_path);
  Writer w{f};
  w.raw(kMagic, sizeof(kMagic));
  w.str(ts.source_path);
  w.pod(ts.source_size);
  w.pod(static_cast<uint32_t>(ts.traces.size()));
  for (const Trace& t : ts.traces) {
    w.str(t.id);
    w.pod(static_cast<int32_t>(t.block_size));
    w.pod(static_cast<uint32_t>(t.models.size()));
    for (const std::string& m : t.models) w.str(m);
    w.pod(static_cast<uint32_t>(t.agents.size()));
    for (const Agent& a : t.agents) {
      w.pod(a.parent);
      w.pod(a.kind);
      w.pod(a.background);
      w.pod(a.chain_index);
      w.str(a.label);
    }
    w.vec(t.requests);
    w.vec(t.edges);
    w.vec(t.hash_ids);
  }
  std::fclose(f);
}

std::optional<TraceSet> load_cache(const std::string& cache_path) {
  std::FILE* f = std::fopen(cache_path.c_str(), "rb");
  if (!f) return std::nullopt;
  Reader r{f};
  char magic[8];
  r.raw(magic, sizeof(magic));
  if (!r.ok || std::memcmp(magic, kMagic, sizeof(magic)) != 0) {
    std::fclose(f);
    return std::nullopt;
  }
  TraceSet ts;
  ts.source_path = r.str();
  ts.source_size = r.pod<uint64_t>();
  ts.traces.resize(r.pod<uint32_t>());
  for (Trace& t : ts.traces) {
    if (!r.ok) break;
    t.id = r.str();
    t.block_size = r.pod<int32_t>();
    t.models.resize(r.pod<uint32_t>());
    for (std::string& m : t.models) m = r.str();
    t.agents.resize(r.pod<uint32_t>());
    for (Agent& a : t.agents) {
      a.parent = r.pod<int32_t>();
      a.kind = r.pod<AgentKind>();
      a.background = r.pod<bool>();
      a.chain_index = r.pod<int32_t>();
      a.label = r.str();
    }
    t.requests = r.vec<Request>();
    t.edges = r.vec<Edge>();
    t.hash_ids = r.vec<uint64_t>();
  }
  std::fclose(f);
  if (!r.ok) return std::nullopt;
  return ts;
}

TraceSet load(const std::string& jsonl_path, const std::string& cache_dir) {
  namespace fs = std::filesystem;
  fs::path cache_path = fs::path(cache_dir) / (fs::path(jsonl_path).filename().string() + ".dlsimcache");
  uint64_t size = fs::file_size(jsonl_path);
  if (auto cached = load_cache(cache_path.string()); cached && cached->source_path == jsonl_path && cached->source_size == size)
    return std::move(*cached);
  TraceSet ts = load_weka(jsonl_path);
  fs::create_directories(cache_dir);
  save_cache(ts, cache_path.string());
  return ts;
}

Stats compute_stats(const TraceSet& ts) {
  Stats s;
  HitCount hc;
  for (const Trace& t : ts.traces) {
    ++s.traces;
    for (const Agent& a : t.agents)
      if (a.kind == AgentKind::Subagent) ++s.subagent_groups;
    for (const Request& r : t.requests) {
      const Agent& a = t.agents[r.agent];
      bool nested = a.kind == AgentKind::Subagent || (a.parent >= 0 && t.agents[a.parent].kind == AgentKind::Subagent);
      ++s.total_requests;
      s.total_input_tokens += r.in_tokens;
      s.total_output_tokens += r.out_tokens;
      if (nested)
        ++s.subagent_inner_requests;
      else
        ++s.main_turns;
    }
    HitCount c = count_prefix_hits(t);
    hc.hits += c.hits;
    hc.total += c.total;
  }
  s.ideal_prefix_hit_rate = hc.total ? static_cast<double>(hc.hits) / static_cast<double>(hc.total) : 0.0;
  return s;
}

double ideal_prefix_hit_rate(const Trace& t) {
  HitCount c = count_prefix_hits(t);
  return c.total ? static_cast<double>(c.hits) / static_cast<double>(c.total) : 0.0;
}

}  // namespace dlsim::trace
