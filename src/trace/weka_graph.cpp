#include <algorithm>
#include <cmath>
#include <cstdio>
#include <functional>
#include <map>
#include <queue>
#include <unordered_map>
#include <unordered_set>
#include <utility>

#include "trace/weka_raw.h"

namespace dlsim::trace::detail {
namespace {

constexpr double kEps = 1e-6;
constexpr double kSeamMaxGapSeconds = 3600.0;
constexpr double kSeamMinOverlapRatio = 0.5;
constexpr int kAuxMaxRequests = 1;
constexpr double kAuxIslRatio = 0.10;
constexpr int64_t kAuxIslFloor = 16384;
constexpr int64_t kAuxReductionOslMax = 4000;
constexpr double kAuxReductionRatio = 20.0;
constexpr int kWorkerGroupMin = 3;
constexpr int64_t kTitleGenMaxOutputTokens = 64;

struct Item {
  int32_t oi;
  const RawRequest* req;
};

double req_end(const RawRequest& r) {
  double d = std::isfinite(r.api_time) ? r.api_time : 0.0;
  return r.t + std::max(d, 0.0);
}

double end_to_start_delay(const RawRequest& prev, const RawRequest& cur) {
  double api = std::isfinite(prev.api_time) ? prev.api_time : 0.0;
  return std::max(0.0, (cur.t - prev.t) - api);
}

size_t lcp(const std::vector<uint64_t>& a, const std::vector<uint64_t>& b) {
  size_t n = std::min(a.size(), b.size()), i = 0;
  while (i < n && a[i] == b[i]) ++i;
  return i;
}

bool by_time_then_oi(const Item& a, const Item& b) {
  if (a.req->t != b.req->t) return a.req->t < b.req->t;
  return a.oi < b.oi;
}

struct Fork {
  int32_t parent_chain = -1;
  int32_t fork_outer = -1;
  int64_t depth = 0;
  double fork_time = 0;
};

struct Chain {
  std::vector<Item> requests;
  bool has_fork = false;
  Fork fork;
  int32_t spliced_into = -1;
  int32_t tail_outer = -1;
  const std::vector<uint64_t>* tail_hash = nullptr;
  double tail_end = 0;
  uint16_t tail_model = 0;
  size_t tail_len() const { return tail_hash ? tail_hash->size() : 0; }
};

struct Detection {
  std::vector<Chain> chains;
  int32_t main_index = 0;
  std::vector<int32_t> workers;
};

int32_t last_hash_outer(const Chain& c) {
  for (auto it = c.requests.rbegin(); it != c.requests.rend(); ++it)
    if (!it->req->hash_ids.empty()) return it->oi;
  return -1;
}

int32_t find_extension_target(const std::vector<Chain>& chains, const std::vector<uint64_t>& h, double t, uint16_t model) {
  int32_t best = -1;
  int64_t best_len = -1;
  for (size_t idx = 0; idx < chains.size(); ++idx) {
    const Chain& c = chains[idx];
    int64_t tl = static_cast<int64_t>(c.tail_len());
    if (tl == 0 || tl > static_cast<int64_t>(h.size()) || tl <= best_len) continue;
    if (c.tail_model != model) continue;
    if (c.tail_end > t + kEps) continue;
    if ((*c.tail_hash)[tl - 1] != h[tl - 1]) continue;
    if (std::equal(c.tail_hash->begin(), c.tail_hash->end(), h.begin())) {
      best = static_cast<int32_t>(idx);
      best_len = tl;
    }
  }
  return best;
}

std::pair<int32_t, int64_t> max_lcp_chain(const std::vector<Chain>& chains, const std::vector<uint64_t>& h) {
  int32_t best_idx = -1;
  std::pair<int64_t, int64_t> best_key{0, 0};
  for (size_t idx = 0; idx < chains.size(); ++idx) {
    const Chain& c = chains[idx];
    if (c.tail_len() == 0) continue;
    int64_t d = static_cast<int64_t>(lcp(*c.tail_hash, h));
    if (d == 0) continue;
    std::pair<int64_t, int64_t> key{d, static_cast<int64_t>(c.tail_len())};
    if (key > best_key) {
      best_idx = static_cast<int32_t>(idx);
      best_key = key;
    }
  }
  return {best_idx, best_key.first};
}

struct Phase1 {
  std::vector<Chain> chains;
  std::unordered_map<int32_t, int32_t> chain_of_request;
  std::map<int32_t, std::vector<int32_t>> forks_by_tail;
  std::unordered_map<int32_t, const RawRequest*> req_by_outer;

  void append(int32_t ci, Item item) {
    Chain& c = chains[ci];
    c.requests.push_back(item);
    chain_of_request[item.oi] = ci;
    if (!item.req->hash_ids.empty()) {
      c.tail_outer = item.oi;
      c.tail_hash = &item.req->hash_ids;
      c.tail_end = req_end(*item.req);
      c.tail_model = item.req->model;
    }
  }

  void classify(Item item) {
    req_by_outer[item.oi] = item.req;
    const std::vector<uint64_t>& h = item.req->hash_ids;
    if (h.empty()) {
      if (chains.empty()) chains.emplace_back();
      chains[0].requests.push_back(item);
      chain_of_request[item.oi] = 0;
      return;
    }
    if (chains.empty()) {
      chains.emplace_back();
      append(0, item);
      return;
    }
    int32_t target = find_extension_target(chains, h, item.req->t, item.req->model);
    if (target >= 0) {
      append(target, item);
      return;
    }
    auto [parent, depth] = max_lcp_chain(chains, h);
    if (parent < 0 && std::all_of(chains.begin(), chains.end(), [](const Chain& c) { return c.tail_len() == 0; })) {
      append(0, item);
      return;
    }
    Chain nc;
    nc.has_fork = true;
    nc.fork = {parent, parent >= 0 ? chains[parent].tail_outer : -1, depth, item.req->t};
    int32_t new_idx = static_cast<int32_t>(chains.size());
    chains.push_back(std::move(nc));
    append(new_idx, item);
    if (chains[new_idx].fork.fork_outer >= 0 && depth > 0) forks_by_tail[chains[new_idx].fork.fork_outer].push_back(new_idx);
  }
};

int32_t elect_continuation(const std::vector<Chain>& chains, const std::vector<int32_t>& registered, const RawRequest& tail_req) {
  double t_end = req_end(tail_req);
  double tail_blocks = static_cast<double>(tail_req.hash_ids.size());
  int32_t best = -1;
  for (int32_t ci : registered) {
    const Chain& c = chains[ci];
    if (!c.has_fork || c.fork.depth <= 0) continue;
    const RawRequest& first = *c.requests[0].req;
    if (!(t_end <= first.t + kEps)) continue;
    if (first.model != tail_req.model) continue;
    if (tail_blocks > 0) {
      double gap = first.t - t_end;
      double overlap = static_cast<double>(c.fork.depth) / tail_blocks;
      if (gap > kSeamMaxGapSeconds && overlap < kSeamMinOverlapRatio) continue;
    }
    if (best < 0) {
      best = ci;
      continue;
    }
    const Chain& b = chains[best];
    if (c.fork.depth != b.fork.depth) {
      if (c.fork.depth > b.fork.depth) best = ci;
    } else if (c.fork.fork_time != b.fork.fork_time) {
      if (c.fork.fork_time < b.fork.fork_time) best = ci;
    } else if (ci < best) {
      best = ci;
    }
  }
  return best;
}

bool rekey_leftover_forks(Phase1& st, const std::vector<int32_t>& registered, int32_t elected, int32_t owner, int32_t new_tail_outer) {
  const std::vector<uint64_t>& new_tail_hash = st.req_by_outer[new_tail_outer]->hash_ids;
  bool rekeyed = false;
  for (int32_t ci : registered) {
    Chain& c = st.chains[ci];
    if (ci == elected || c.spliced_into >= 0 || !c.has_fork) continue;
    int64_t d = static_cast<int64_t>(lcp(new_tail_hash, c.requests[0].req->hash_ids));
    if (d <= 0) continue;
    c.fork.fork_outer = new_tail_outer;
    c.fork.depth = d;
    c.fork.parent_chain = owner;
    st.forks_by_tail[new_tail_outer].push_back(ci);
    rekeyed = true;
  }
  return rekeyed;
}

void resolve_seams(Phase1& st) {
  std::unordered_map<int32_t, int32_t> alias;
  auto resolve = [&](int32_t i) {
    while (alias.count(i)) i = alias[i];
    return i;
  };
  std::priority_queue<int32_t, std::vector<int32_t>, std::greater<int32_t>> keys;
  for (const auto& kv : st.forks_by_tail) keys.push(kv.first);
  std::unordered_set<int32_t> processed;
  while (!keys.empty()) {
    int32_t fo = keys.top();
    keys.pop();
    if (processed.count(fo)) continue;
    processed.insert(fo);
    int32_t owner = resolve(st.chain_of_request[fo]);
    if (last_hash_outer(st.chains[owner]) != fo) continue;
    std::vector<int32_t> registered;
    for (int32_t ci : st.forks_by_tail[fo])
      if (st.chains[ci].spliced_into < 0) registered.push_back(ci);
    int32_t elected = elect_continuation(st.chains, registered, *st.req_by_outer[fo]);
    if (elected < 0) continue;
    std::vector<Item> moved = st.chains[elected].requests;
    st.chains[owner].requests.insert(st.chains[owner].requests.end(), moved.begin(), moved.end());
    for (const Item& it : moved) st.chain_of_request[it.oi] = owner;
    st.chains[elected].spliced_into = owner;
    alias[elected] = owner;
    int32_t new_tail = last_hash_outer(st.chains[owner]);
    if (new_tail < 0) continue;
    if (rekey_leftover_forks(st, registered, elected, owner, new_tail)) {
      processed.erase(new_tail);
      keys.push(new_tail);
    }
  }
}

Detection detect_agent_chains(std::vector<Item> items) {
  Detection out;
  if (items.empty()) return out;
  std::sort(items.begin(), items.end(), by_time_then_oi);
  Phase1 st;
  for (const Item& it : items) st.classify(it);
  resolve_seams(st);
  std::unordered_map<int32_t, int32_t> alias;
  for (size_t i = 0; i < st.chains.size(); ++i)
    if (st.chains[i].spliced_into >= 0) alias[static_cast<int32_t>(i)] = st.chains[i].spliced_into;
  auto resolve = [&](int32_t i) {
    while (alias.count(i)) i = alias[i];
    return i;
  };
  for (Chain& c : st.chains)
    if (c.spliced_into < 0 && c.has_fork && c.fork.parent_chain >= 0) c.fork.parent_chain = resolve(c.fork.parent_chain);
  out.main_index = resolve(st.chain_of_request[items[0].oi]);
  for (size_t i = 0; i < st.chains.size(); ++i)
    if (st.chains[i].spliced_into < 0 && static_cast<int32_t>(i) != out.main_index) out.workers.push_back(static_cast<int32_t>(i));
  std::sort(out.workers.begin(), out.workers.end(), [&](int32_t a, int32_t b) {
    return by_time_then_oi(st.chains[a].requests[0], st.chains[b].requests[0]);
  });
  out.chains = std::move(st.chains);
  return out;
}

struct Cand {
  double t0, t1;
  int32_t first_oi, ci;
};

std::vector<std::vector<Cand>> overlap_components(std::vector<Cand> cands) {
  std::vector<std::vector<Cand>> comps;
  if (cands.empty()) return comps;
  std::vector<size_t> order(cands.size());
  for (size_t i = 0; i < order.size(); ++i) order[i] = i;
  std::sort(order.begin(), order.end(), [&](size_t a, size_t b) {
    if (cands[a].t0 != cands[b].t0) return cands[a].t0 < cands[b].t0;
    return cands[a].first_oi < cands[b].first_oi;
  });
  std::vector<size_t> parent(cands.size());
  for (size_t i = 0; i < parent.size(); ++i) parent[i] = i;
  auto find = [&](size_t x) {
    while (parent[x] != x) {
      parent[x] = parent[parent[x]];
      x = parent[x];
    }
    return x;
  };
  using Active = std::pair<double, size_t>;
  std::priority_queue<Active, std::vector<Active>, std::greater<Active>> active;
  for (size_t i : order) {
    while (!active.empty() && active.top().first <= cands[i].t0) active.pop();
    if (!active.empty()) parent[find(i)] = find(active.top().second);
    active.push({cands[i].t1, i});
  }
  std::map<size_t, size_t> comp_of_root;
  for (size_t i = 0; i < cands.size(); ++i) {
    size_t r = find(i);
    auto it = comp_of_root.find(r);
    if (it == comp_of_root.end()) {
      it = comp_of_root.emplace(r, comps.size()).first;
      comps.emplace_back();
    }
    comps[it->second].push_back(cands[i]);
  }
  return comps;
}

std::unordered_map<int32_t, std::pair<int, int>> worker_group_assignment(const Detection& det) {
  std::map<std::pair<int32_t, int32_t>, std::vector<Cand>> buckets;
  for (int32_t ci : det.workers) {
    const Chain& c = det.chains[ci];
    if (!c.has_fork || c.fork.depth <= 0 || c.fork.parent_chain < 0 || c.fork.fork_outer < 0 || c.requests.empty()) continue;
    double t1 = -INFINITY;
    for (const Item& it : c.requests) t1 = std::max(t1, req_end(*it.req));
    buckets[{c.fork.parent_chain, c.fork.fork_outer}].push_back({c.requests[0].req->t, t1, c.requests[0].oi, ci});
  }
  std::vector<std::vector<Cand>> groups;
  for (auto& kv : buckets)
    for (auto& comp : overlap_components(kv.second))
      if (static_cast<int>(comp.size()) >= kWorkerGroupMin) groups.push_back(std::move(comp));
  auto key_of = [](const Cand& c) { return std::make_pair(c.t0, c.first_oi); };
  auto min_key = [&](const std::vector<Cand>& comp) {
    auto k = key_of(comp[0]);
    for (const Cand& c : comp) k = std::min(k, key_of(c));
    return k;
  };
  std::sort(groups.begin(), groups.end(), [&](const auto& a, const auto& b) { return min_key(a) < min_key(b); });
  std::unordered_map<int32_t, std::pair<int, int>> out;
  for (size_t g = 0; g < groups.size(); ++g) {
    std::sort(groups[g].begin(), groups[g].end(), [&](const Cand& a, const Cand& b) { return key_of(a) < key_of(b); });
    for (size_t m = 0; m < groups[g].size(); ++m) out[groups[g][m].ci] = {static_cast<int>(g), static_cast<int>(m)};
  }
  return out;
}

bool is_aux_chain(const std::vector<Item>& reqs, int64_t main_peak_isl, bool has_main_model, uint16_t main_model) {
  if (reqs.empty() || static_cast<int>(reqs.size()) > kAuxMaxRequests) return false;
  const RawRequest& first = *reqs[0].req;
  if (has_main_model && first.model != main_model) return true;
  int64_t threshold = std::max(kAuxIslFloor, static_cast<int64_t>(kAuxIslRatio * static_cast<double>(main_peak_isl)));
  return first.in_tokens < threshold;
}

bool is_reduction_chain(const std::vector<Item>& reqs) {
  if (kAuxReductionOslMax <= 0 || reqs.size() != 1) return false;
  const RawRequest& first = *reqs[0].req;
  int64_t osl = first.out_tokens;
  if (osl <= 0 || osl >= kAuxReductionOslMax) return false;
  if (first.in_tokens < kAuxIslFloor) return false;
  return static_cast<double>(first.in_tokens) > kAuxReductionRatio * static_cast<double>(osl);
}

std::pair<std::vector<Item>, std::vector<Item>> split_off_preamble(const std::vector<Item>& items) {
  if (items.size() < 2) return {{}, items};
  std::vector<Item> ordered = items;
  std::sort(ordered.begin(), ordered.end(), by_time_then_oi);
  const Item& lead = ordered[0];
  if (lead.req->hash_ids.empty()) return {{}, items};
  std::vector<Item> rest(ordered.begin() + 1, ordered.end());
  for (const Item& other : rest)
    if (!other.req->hash_ids.empty() && lcp(lead.req->hash_ids, other.req->hash_ids) > 0) return {{}, items};
  if (lead.req->out_tokens > kTitleGenMaxOutputTokens) {
    std::unordered_set<uint64_t> other_blocks;
    for (const Item& other : rest) other_blocks.insert(other.req->hash_ids.begin(), other.req->hash_ids.end());
    for (uint64_t h : lead.req->hash_ids)
      if (other_blocks.count(h)) return {{}, items};
  }
  std::sort(rest.begin(), rest.end(), [](const Item& a, const Item& b) { return a.oi < b.oi; });
  return {{lead}, rest};
}

struct ChainOut {
  AgentKind kind;
  std::string suffix;
  std::vector<Item> requests;
};

std::string worker_suffix(int n, bool aux, const std::pair<int, int>* wg) {
  char buf[32];
  if (aux)
    std::snprintf(buf, sizeof(buf), "aux:%03d", n);
  else if (wg)
    std::snprintf(buf, sizeof(buf), "wg:%03d_%03d", wg->first, wg->second);
  else
    std::snprintf(buf, sizeof(buf), "fa:%03d", n);
  return buf;
}

ChainOut classify_worker(const Detection& det, int n, int32_t ci, const std::unordered_map<int32_t, std::pair<int, int>>& wg_coords,
                         int64_t main_peak_isl, bool has_main_model, uint16_t main_model) {
  const std::vector<Item>& reqs = det.chains[ci].requests;
  bool aux = is_aux_chain(reqs, main_peak_isl, has_main_model, main_model);
  bool reduction = !aux && is_reduction_chain(reqs);
  const std::pair<int, int>* wg = nullptr;
  if (!aux && !reduction) {
    auto it = wg_coords.find(ci);
    if (it != wg_coords.end()) wg = &it->second;
  }
  AgentKind kind = (aux || reduction) ? AgentKind::FlatAux : wg ? AgentKind::FlatWorkerGroup : AgentKind::FlatIndependent;
  return {kind, worker_suffix(n, aux || reduction, wg), reqs};
}

std::vector<ChainOut> expand_subagent(const RawSubagent& sa, std::vector<RawRequest>& normalized) {
  normalized = sa.requests;
  for (RawRequest& r : normalized)
    if (r.t + kEps < sa.t) r.t = sa.t + r.t;
  std::vector<Item> items;
  for (size_t i = 0; i < normalized.size(); ++i) items.push_back({static_cast<int32_t>(i), &normalized[i]});
  std::vector<ChainOut> out;
  if (items.empty()) {
    out.push_back({AgentKind::Subagent, "sa:" + sa.agent_id, {}});
    return out;
  }
  auto [preamble, detect_inner] = split_off_preamble(items);
  Detection det = detect_agent_chains(detect_inner);
  const std::vector<Item>& detected_main = det.chains[det.main_index].requests;
  std::vector<Item> main_requests = detected_main;
  if (!preamble.empty()) {
    main_requests.insert(main_requests.end(), preamble.begin(), preamble.end());
    std::sort(main_requests.begin(), main_requests.end(), by_time_then_oi);
  }
  auto wg_coords = worker_group_assignment(det);
  int64_t main_peak_isl = 0;
  for (const Item& it : detected_main) main_peak_isl = std::max(main_peak_isl, it.req->in_tokens);
  bool has_main_model = !detected_main.empty();
  uint16_t main_model = has_main_model ? detected_main[0].req->model : 0;
  out.push_back({AgentKind::Subagent, "sa:" + sa.agent_id, main_requests});
  for (size_t n = 0; n < det.workers.size(); ++n) {
    ChainOut co = classify_worker(det, static_cast<int>(n), det.workers[n], wg_coords, main_peak_isl, has_main_model, main_model);
    co.suffix = "sa:" + sa.agent_id + ":" + co.suffix;
    out.push_back(std::move(co));
  }
  return out;
}

double sa_end_seconds(const RawSubagent& sa, const std::vector<RawRequest>& normalized) {
  if (std::isfinite(sa.duration_ms)) {
    double d = sa.duration_ms / 1000.0;
    if (!std::isfinite(d) || d < 0.0) d = 0.0;
    return sa.t + d;
  }
  if (!normalized.empty()) {
    double e = -INFINITY;
    for (const RawRequest& r : normalized) e = std::max(e, req_end(r));
    return e;
  }
  return sa.t;
}

struct Builder {
  Trace& t;
  std::vector<Item> main;

  int32_t add_request(int32_t agent, const RawRequest& r) {
    Request q;
    q.agent = agent;
    q.t_s = r.t;
    q.api_time_s = std::isfinite(r.api_time) ? r.api_time : 0.0;
    q.in_tokens = r.in_tokens;
    q.out_tokens = r.out_tokens;
    q.hash_begin = static_cast<uint32_t>(t.hash_ids.size());
    q.hash_count = static_cast<uint32_t>(r.hash_ids.size());
    q.model = r.model;
    q.streaming = r.streaming;
    q.outer_idx = r.outer_idx;
    t.hash_ids.insert(t.hash_ids.end(), r.hash_ids.begin(), r.hash_ids.end());
    t.requests.push_back(q);
    return static_cast<int32_t>(t.requests.size() - 1);
  }

  void edge(int32_t from, int32_t to, EdgeKind kind) {
    t.edges.push_back({from, to, kind, end_to_start_delay(*req_ptr[from], *req_ptr[to])});
  }

  std::vector<const RawRequest*> req_ptr;

  std::vector<int32_t> add_chain(int32_t agent, const std::vector<Item>& items, int32_t marker_outer) {
    std::vector<int32_t> ids;
    for (const Item& it : items) {
      int32_t id = add_request(agent, *it.req);
      if (marker_outer >= 0) t.requests[id].outer_idx = marker_outer;
      req_ptr.push_back(it.req);
      ids.push_back(id);
    }
    for (size_t i = 1; i < ids.size(); ++i) edge(ids[i - 1], ids[i], EdgeKind::Sequential);
    return ids;
  }

  int32_t preceding_pos(int32_t before_outer, bool default_zero) const {
    int32_t best = default_zero ? 0 : -1;
    for (size_t pos = 0; pos < main.size(); ++pos)
      if (main[pos].oi < before_outer) best = std::max(best, static_cast<int32_t>(pos));
    return best;
  }

  int32_t join_pos(int32_t after_outer, double end) const {
    std::vector<size_t> order(main.size());
    for (size_t i = 0; i < order.size(); ++i) order[i] = i;
    std::sort(order.begin(), order.end(), [&](size_t a, size_t b) { return main[a].oi < main[b].oi; });
    for (size_t pos : order) {
      if (main[pos].oi <= after_outer) continue;
      if (main[pos].req->t + kEps >= end) return static_cast<int32_t>(pos);
    }
    return -1;
  }

  void attach(const std::vector<int32_t>& chain_ids, const std::vector<int32_t>& main_ids, int32_t preceding, int32_t join) {
    if (chain_ids.empty()) return;
    if (join < 0) {
      edge(main_ids[preceding], chain_ids.front(), EdgeKind::Background);
    } else {
      edge(main_ids[preceding], chain_ids.front(), EdgeKind::Spawn);
      edge(chain_ids.back(), main_ids[join], EdgeKind::Join);
    }
  }
};

}  // namespace

Trace build_trace(RawTrace&& raw) {
  Trace t;
  t.id = raw.id;
  t.block_size = raw.block_size;
  t.models = raw.models;

  std::vector<Item> normals;
  for (const RawRequest& r : raw.normals) normals.push_back({r.outer_idx, &r});

  std::vector<ChainOut> flat;
  std::vector<Item> main_items = normals;
  if (normals.size() > 1) {
    auto [preamble, detect_normals] = split_off_preamble(normals);
    Detection det = detect_agent_chains(detect_normals);
    if (!det.workers.empty()) {
      const Chain& main_chain = det.chains[det.main_index];
      int64_t main_peak_isl = 0;
      for (const Item& it : main_chain.requests) main_peak_isl = std::max(main_peak_isl, it.req->in_tokens);
      bool has_main_model = !main_chain.requests.empty();
      uint16_t main_model = has_main_model ? main_chain.requests[0].req->model : 0;
      auto wg_coords = worker_group_assignment(det);
      for (size_t n = 0; n < det.workers.size(); ++n)
        flat.push_back(classify_worker(det, static_cast<int>(n), det.workers[n], wg_coords, main_peak_isl, has_main_model, main_model));
      main_items = main_chain.requests;
      if (!preamble.empty()) {
        main_items.insert(main_items.end(), preamble.begin(), preamble.end());
        std::sort(main_items.begin(), main_items.end(), by_time_then_oi);
      }
    }
  }

  Builder b{t, main_items};
  t.agents.push_back({-1, AgentKind::Main, "", false, 0});
  std::vector<int32_t> main_ids = b.add_chain(0, main_items, -1);

  for (const RawSubagent& sa : raw.subagents) {
    int32_t preceding = b.preceding_pos(sa.outer_idx, false);
    if (preceding < 0) continue;
    std::vector<RawRequest> normalized;
    std::vector<ChainOut> chains = expand_subagent(sa, normalized);
    int32_t join = b.join_pos(sa.outer_idx, sa_end_seconds(sa, normalized));
    int32_t sa_agent = static_cast<int32_t>(t.agents.size());
    for (size_t ci = 0; ci < chains.size(); ++ci) {
      int32_t agent = static_cast<int32_t>(t.agents.size());
      t.agents.push_back({ci == 0 ? 0 : sa_agent, chains[ci].kind, chains[ci].suffix, join < 0, static_cast<int32_t>(ci)});
      std::vector<int32_t> ids = b.add_chain(agent, chains[ci].requests, sa.outer_idx);
      b.attach(ids, main_ids, preceding, join);
    }
  }

  for (const ChainOut& fc : flat) {
    int32_t first_outer = fc.requests[0].oi;
    int32_t preceding = b.preceding_pos(first_outer, true);
    double end = -INFINITY;
    for (const Item& it : fc.requests) end = std::max(end, req_end(*it.req));
    int32_t join = b.join_pos(first_outer, end);
    int32_t agent = static_cast<int32_t>(t.agents.size());
    t.agents.push_back({0, fc.kind, fc.suffix, join < 0, 0});
    std::vector<int32_t> ids = b.add_chain(agent, fc.requests, -1);
    b.attach(ids, main_ids, preceding, join);
  }
  return t;
}

}  // namespace dlsim::trace::detail
