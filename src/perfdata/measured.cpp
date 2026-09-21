#include "perfdata/measured.h"

#include <simdjson.h>

#include <algorithm>
#include <cmath>
#include <set>
#include <sstream>
#include <unordered_set>
#include <variant>

#include "perfdata/dtype.h"
#include "perfdata/version.h"

namespace dlsim {

bool is_provenance_column(const std::string& name) { return name == "framework" || name == "version" || name == "device"; }
bool is_key_column(const Column& c) { return c.kind != Column::Kind::Float && !is_provenance_column(c.name); }

namespace {

std::string fmt(double v) {
  std::ostringstream o;
  o.precision(6);
  o << v;
  return o.str();
}

std::string join_values(const std::set<std::string>& s) {
  std::string o;
  for (const auto& v : s) o += (o.empty() ? "" : ", ") + v;
  return o;
}

struct Axis {
  const Column* col;
  double q;
  bool sqrt_space;
};

struct Node {
  std::map<double, Node> ch;
  double val = std::nan("");
};

struct OutOfRange {
  size_t axis;
  bool above;
};
struct Miss {};
using Res = std::variant<double, OutOfRange, Miss>;

double to_space(bool s, double v) { return s ? std::sqrt(v) : v; }
double from_space(bool s, double v) { return s ? v * v : v; }

// Step-axis extrapolation state for the attention module tables (design: operator-latency.md 实现一).
struct Extrap {
  bool enabled = false;   // the query's table kind has a sequence-axis rule
  std::string line_axis;  // key whose values are grid lines with their own step range (batch_size on the generation grids)
  double isl = 0;         // context tables: the note reports query/edge in isl + step
  bool used = false;
  std::string note;
};

Res resolve(const Node& n, std::vector<Axis>& axes, size_t d, Extrap* ex);

// From the node below a line key, follow the exact query keys of the intermediate axes down to the node whose children
// are the step values. Null when an intermediate key is absent. step_depth receives the step axis index.
const Node* walk_to_step(const Node& n, const std::vector<Axis>& axes, size_t d, size_t& step_depth) {
  const Node* cur = &n;
  for (; d < axes.size(); ++d) {
    if (axes[d].col->name == "step") {
      step_depth = d;
      return cur;
    }
    auto it = cur->ch.find(axes[d].q);
    if (it == cur->ch.end()) return nullptr;
    cur = &it->second;
  }
  return nullptr;
}

// Sequence-axis extrapolation across grid lines. The line of key `key` (batch, or isl for the context tables) ends at s_e
// below the query step; the sibling line with the longest step coverage supplies the shape beyond s_e and is scaled to
// this line at s_e: value = t(key, s_e) * t(ref, c) / t(ref, s_e). The sparse-attention generation grids are measured to
// much shorter steps at large batch, and the slope of a line's last two points before its truncation does not continue.
Res ratio_extrapolate(const Node& n, double key, std::vector<Axis>& axes, size_t d, size_t sd, double s_e, Extrap* ex) {
  const double c = axes[sd].q;
  const Node* ref = nullptr;
  double ref_key = 0, ref_edge = s_e;
  for (const auto& [k, ch] : n.ch) {
    if (k == key) continue;
    size_t sd2 = 0;
    const Node* sn = walk_to_step(ch, axes, d + 1, sd2);
    if (!sn || sn->ch.empty()) continue;
    const double e = sn->ch.rbegin()->first;
    if (e > ref_edge || (e == ref_edge && ref && std::fabs(k - key) < std::fabs(ref_key - key))) {
      ref = &ch;
      ref_key = k;
      ref_edge = e;
    }
  }
  if (!ref) return Miss{};
  axes[sd].q = s_e;
  const Res t_e = resolve(n.ch.at(key), axes, d + 1, ex), r_e = resolve(*ref, axes, d + 1, ex);
  axes[sd].q = c;
  const Res r_c = resolve(*ref, axes, d + 1, ex);
  if (!std::holds_alternative<double>(t_e) || !std::holds_alternative<double>(r_e) || !std::holds_alternative<double>(r_c))
    return Miss{};
  if (std::get<double>(r_e) <= 0) return Miss{};
  const double ratio = std::get<double>(r_c) / std::get<double>(r_e);
  ex->used = true;
  ex->note = "extrapolated along step: query/edge=" + fmt((ex->isl + c) / (ex->isl + s_e)) + " (" + fmt(ex->isl + c) + " vs " +
             fmt(ex->isl + s_e) + " tokens of isl+step), edge_latency=" + fmt(std::get<double>(t_e)) + ", shape from the " +
             axes[d].col->name + "=" + fmt(ref_key) + " line (measured to step " + fmt(ref_edge) + "), ratio " + fmt(ratio);
  return std::get<double>(t_e) * ratio;
}

Res resolve(const Node& n, std::vector<Axis>& axes, size_t d, Extrap* ex) {
  if (d == axes.size()) return std::isnan(n.val) ? Res(Miss{}) : Res(n.val);
  const double c = axes[d].q;
  if (auto it = n.ch.find(c); it != n.ch.end()) {
    if (ex && ex->enabled && axes[d].col->name == ex->line_axis) {
      size_t sd = 0;
      const Node* sn = walk_to_step(it->second, axes, d + 1, sd);
      if (sn && !sn->ch.empty() && axes[sd].q > sn->ch.rbegin()->first) {
        const Res rr = ratio_extrapolate(n, it->first, axes, d, sd, sn->ch.rbegin()->first, ex);
        if (std::holds_alternative<double>(rr)) return rr;
      }
    }
    return resolve(it->second, axes, d + 1, ex);
  }
  if (n.ch.empty()) return Miss{};
  if (c < n.ch.begin()->first) return OutOfRange{d, false};
  if (c > n.ch.rbegin()->first) {
    const std::string& name = axes[d].col->name;
    if (ex && (name == "num_tokens" || name == "m" || name == "message_size") && n.ch.size() >= 2) {
      // Token / message axes: above the grid the operator is throughput-bound, latency linear in the axis.
      auto b = n.ch.rbegin(), a = std::next(b);
      const Res rb = resolve(b->second, axes, d + 1, ex), ra = resolve(a->second, axes, d + 1, ex);
      if (!std::holds_alternative<double>(rb)) return rb;
      if (!std::holds_alternative<double>(ra)) return ra;
      const double slope = std::max(0.0, (std::get<double>(rb) - std::get<double>(ra)) / (b->first - a->first));
      if (!ex->used) {
        ex->used = true;
        ex->note = "extrapolated along " + name + " beyond " + fmt(b->first) + " (linear, slope " + fmt(slope) + " ms per unit from " +
                   fmt(a->first) + ".." + fmt(b->first) + ")";
      }
      return std::get<double>(rb) + slope * (c - b->first);
    }
    if (!ex || !ex->enabled || name != "step") return OutOfRange{d, true};
    // Edge slope from this node's own step list: the largest measured step, and the next resolvable one at or below
    // half of it (the grids fill the last few thousand tokens before 65,536 densely; a slope across that gap is noise).
    std::vector<std::pair<double, double>> edges;
    for (auto it = n.ch.rbegin(); it != n.ch.rend() && edges.size() < 2; ++it) {
      if (edges.size() == 1 && it->first > 0.5 * edges[0].first) continue;
      const Res e = resolve(it->second, axes, d + 1, ex);
      if (std::holds_alternative<double>(e)) edges.push_back({it->first, std::get<double>(e)});
    }
    if (edges.size() < 2) return OutOfRange{d, true};
    const auto [s_e, t_e] = edges[0];
    const auto [s_p, t_p] = edges[1];
    const double alpha = std::max(0.0, (t_e - t_p) / (s_e - s_p));
    const double value = t_e + alpha * (c - s_e);
    if (!ex->used) {
      ex->used = true;
      ex->note = "extrapolated along step: query/edge=" + fmt((ex->isl + c) / (ex->isl + s_e)) + " (" + fmt(ex->isl + c) + " vs " +
                 fmt(ex->isl + s_e) + " tokens of isl+step), edge_latency=" + fmt(t_e) + ", slope " + fmt(alpha) +
                 " ms/token from steps " + fmt(s_p) + ".." + fmt(s_e) + ", constant part " + fmt(t_e - alpha * (ex->isl + s_e)) + " ms";
    }
    return value;
  }
  const auto hi = n.ch.lower_bound(c);
  const auto lo = std::prev(hi);
  const Res rl = resolve(lo->second, axes, d + 1, ex);
  const Res rh = resolve(hi->second, axes, d + 1, ex);
  if (!std::holds_alternative<double>(rl)) return rl;
  if (!std::holds_alternative<double>(rh)) return rh;
  const double w = (c - lo->first) / (hi->first - lo->first);
  const bool s = axes[d].sqrt_space;
  const double a = to_space(s, std::get<double>(rl)), b = to_space(s, std::get<double>(rh));
  return from_space(s, a + (b - a) * w);
}

Node build_trie(const std::vector<uint32_t>& rows, const std::vector<Axis>& axes, const Column& lat) {
  Node root;
  for (uint32_t r : rows) {
    Node* n = &root;
    for (const Axis& a : axes) n = &n->ch[a.col->num[r]];
    if (std::isnan(n->val)) n->val = lat.num[r];
  }
  return root;
}

// Nesting depth of a numeric axis: structural keys outermost, the blended sequence / token / batch axes innermost.
// Two nesting orders. In range, step outermost: the trtllm / vllm attention grids run along step + isl = 65,536, so at a
// fixed step the isl neighbours are dense and interpolation over them is accurate (leave-one-out 1.7% on the csa context
// table versus 5.5% the other way round). Beyond the range, batch and isl outermost: each (batch, isl) grid line then
// has its own step range and the sequence-axis extrapolation runs on that line's own edge points.
int axis_depth(const std::string& name, bool line_order) {
  if (line_order) {
    if (name == "batch_size") return 1;
    if (name == "isl" || name == "seq_len") return 2;
    if (name == "step") return 3;
  } else {
    if (name == "step") return 1;
    if (name == "isl" || name == "seq_len") return 2;
    if (name == "batch_size") return 4;
  }
  if (name == "num_tokens" || name == "message_size" || name == "m") return 3;
  return 0;
}

enum class AttnKind { None, Generation, CsaContext, HcaContext, OtherContext };

AttnKind attention_kind(const std::string& op) {
  const bool attn = op.find("attention") != std::string::npos || op.find("dsa") != std::string::npos ||
                    op.find("mla") != std::string::npos || op.find("csa") != std::string::npos ||
                    op.find("hca") != std::string::npos || op.find("context") != std::string::npos ||
                    op.find("generation") != std::string::npos;
  if (!attn) return AttnKind::None;
  if (op.find("generation") != std::string::npos) return AttnKind::Generation;
  if (op.find("csa_context") != std::string::npos) return AttnKind::CsaContext;
  if (op.find("hca_context") != std::string::npos) return AttnKind::HcaContext;
  return AttnKind::OtherContext;
}

double gemm_sol_ms(const DeviceSpec& spec, const std::string& dtype, double m, double n, double k) {
  const std::string key = flops_key_for_dtype(dtype);
  const auto it = spec.flops.find(key);
  if (it == spec.flops.end() || spec.memory.empty() || spec.memory[0].bandwidth_Bps <= 0) return std::nan("");
  const double math = 2 * m * n * k / it->second * 1000;
  const double mem = weight_bytes_for_dtype(dtype) * (m * n + m * k + n * k) / spec.memory[0].bandwidth_Bps * 1000;
  return std::max(math, mem);
}

Latency query_gemm(const Table& t, const std::vector<uint32_t>& rows, const OpQuery& q, const DeviceSpec* spec,
                   const std::string& dtype, const std::string& prefix) {
  const Column *cm = t.find("m"), *cn = t.find("n"), *ck = t.find("k"), *lat = t.find("latency");
  const double m = q.num.at("m"), n = q.num.at("n"), k = q.num.at("k");
  using Curve = std::vector<std::pair<double, double>>;
  std::map<std::pair<double, double>, Curve> sites;
  for (uint32_t r : rows) sites[{cn->num[r], ck->num[r]}].push_back({cm->num[r], lat->num[r]});
  for (auto& [_, c] : sites) std::sort(c.begin(), c.end());
  bool extrapolated = false;
  // Above the measured m range the GEMM is compute-bound and its latency grows linearly in m: extend the curve with the
  // slope of its last two points. Below the range there is no rule.
  const auto eval = [&](const Curve& c, double x) {
    const auto hi = std::lower_bound(c.begin(), c.end(), std::make_pair(x, -1.0));
    if (hi != c.end() && hi->first == x) return hi->second;
    if (hi == c.begin()) return std::nan("");
    if (hi == c.end()) {
      if (c.size() < 2) return std::nan("");
      const auto& a = c[c.size() - 2];
      const auto& b = c.back();
      extrapolated = true;
      return b.second + std::max(0.0, (b.second - a.second) / (b.first - a.first)) * (x - b.first);
    }
    const auto lo = std::prev(hi);
    const double w = (x - lo->first) / (hi->first - lo->first);
    return lo->second + (hi->second - lo->second) * w;
  };
  if (auto it = sites.find({n, k}); it != sites.end()) {
    const Curve& c = it->second;
    if (m < c.front().first)
      return Latency::unsupported(prefix + "m=" + fmt(m) + " below measured m range [" + fmt(c.front().first) + ", " +
                                  fmt(c.back().first) + "] of site (n=" + fmt(n) + ", k=" + fmt(k) + ")");
    const double v = eval(c, m);
    return Latency::exact(v, extrapolated ? Source::Extrapolated : Source::Interpolated,
                          prefix + (extrapolated ? "gemm m-curve linear extrapolation beyond m=" + fmt(c.back().first) : "gemm m-curve interpolation") +
                              " at site (n=" + fmt(n) + ", k=" + fmt(k) + ")");
  }
  if (!spec) return Latency::unsupported(prefix + "unknown gemm site (n=" + fmt(n) + ", k=" + fmt(k) + ") and no device spec for SOL transfer");
  const double sol_q = gemm_sol_ms(*spec, dtype, m, n, k);
  if (!(sol_q > 0)) return Latency::unsupported(prefix + "no flops entry for dtype " + dtype + " in device spec");
  struct Cand {
    double d;
    const std::pair<double, double>* site;
    const Curve* curve;
  };
  std::vector<Cand> cands;
  for (const auto& [s, c] : sites) {
    if (m < c.front().first) continue;
    const double dn = std::log2(s.first) - std::log2(n), dk = std::log2(s.second) - std::log2(k);
    const double d = std::sqrt(dn * dn + dk * dk);
    if (d <= 2.0) cands.push_back({d, &s, &c});
  }
  if (cands.empty()) return Latency::unsupported(prefix + "no measured (n, k) site within 2 octaves covering m=" + fmt(m));
  std::sort(cands.begin(), cands.end(), [](const Cand& a, const Cand& b) { return a.d < b.d; });
  if (cands.size() > 4) cands.resize(4);
  double wsum = 0, usum = 0;
  for (const Cand& c : cands) {
    const double lat_i = eval(*c.curve, m), sol_i = gemm_sol_ms(*spec, dtype, m, c.site->first, c.site->second);
    if (!(lat_i > 0) || !(sol_i > 0)) continue;
    const double w = 1.0 / (c.d * c.d + 1e-12);
    wsum += w;
    usum += w * sol_i / lat_i;
  }
  if (wsum <= 0) return Latency::unsupported(prefix + "no usable neighbour site");
  return Latency::exact(sol_q / (usum / wsum), extrapolated ? Source::Extrapolated : Source::Interpolated,
                        prefix + "gemm util transfer from " + std::to_string(cands.size()) + " sites" +
                            (extrapolated ? " (m beyond a site's measured range, linear)" : "") + ", nearest (n=" +
                            fmt(cands[0].site->first) + ", k=" + fmt(cands[0].site->second) + ") log2 distance " + fmt(cands[0].d));
}

}  // namespace

Latency query_rows(const Table& t, const std::vector<uint32_t>& rows_in, const OpQuery& q, const DeviceSpec* spec,
                   const std::string& prefix) {
  const Column* lat = t.find("latency");
  if (!lat || lat->kind != Column::Kind::Float) return Latency::unsupported(prefix + "table has no latency column");
  std::vector<uint32_t> rows = rows_in;
  std::map<std::string, std::string> cat;
  for (const auto& [k, v] : q.cat) {
    if (is_provenance_column(k)) continue;
    const Column* c = t.find(k);
    if (!c || !c->is_str()) return Latency::unsupported(prefix + "unknown categorical key " + k);
    cat[k] = v;
  }
  {
    std::vector<uint32_t> kept;
    std::vector<std::pair<const Column*, int64_t>> conds;
    for (const auto& [k, v] : cat) {
      if (k == "kernel_source" && v == "*") continue;
      const Column* c = t.find(k);
      const auto it = std::find(c->dict.begin(), c->dict.end(), v);
      conds.push_back({c, it == c->dict.end() ? -1 : it - c->dict.begin()});
    }
    for (uint32_t r : rows) {
      bool ok = true;
      for (const auto& [c, i] : conds)
        if (static_cast<int64_t>(c->idx[r]) != i) { ok = false; break; }
      if (ok) kept.push_back(r);
    }
    rows.swap(kept);
  }
  if (rows.empty()) {
    std::string f;
    for (const auto& [k, v] : cat) f += (f.empty() ? "" : ", ") + k + "=" + v;
    return Latency::unsupported(prefix + "no rows match {" + f + "}");
  }
  // kernel_source lanes are the framework's size-dependent dispatch (e.g. mhc pre: fma / splitk / dg_nosplit over
  // disjoint token ranges). Without an explicit lane, query each lane and keep the fastest supported one; if no lane
  // brackets the point on its own, interpolate over the union of lanes ("*").
  if (const Column* ks = t.find("kernel_source"); ks && ks->is_str() && !cat.count("kernel_source")) {
    std::set<uint32_t> lanes;
    for (uint32_t r : rows) lanes.insert(ks->idx[r]);
    if (lanes.size() > 1) {
      Latency best = Latency::unsupported("");
      for (uint32_t lane : lanes) {
        OpQuery lq = q;
        lq.cat["kernel_source"] = ks->dict[lane];
        Latency l = query_rows(t, rows, lq, spec, prefix);
        if (l.source != Source::Unsupported && (best.source == Source::Unsupported || l.ms < best.ms)) best = l;
      }
      if (best.source != Source::Unsupported) {
        best.note += " [fastest of " + std::to_string(lanes.size()) + " kernel_source lanes]";
        return best;
      }
      OpQuery uq = q;
      uq.cat["kernel_source"] = "*";
      Latency l = query_rows(t, rows, uq, spec, prefix);
      if (l.source != Source::Unsupported) l.note += " [union of " + std::to_string(lanes.size()) + " kernel_source lanes]";
      return l;
    }
  }
  std::string dtype;
  for (const Column& c : t.cols) {
    if (!c.is_str() || is_provenance_column(c.name)) continue;
    if (c.name == "kernel_source" && cat.count(c.name) && cat.at(c.name) == "*") continue;
    if (!cat.count(c.name)) {
      std::set<uint32_t> seen;
      for (uint32_t r : rows) seen.insert(c.idx[r]);
      if (seen.size() > 1) {
        std::set<std::string> vals;
        for (uint32_t i : seen) vals.insert(c.dict[i]);
        return Latency::unsupported(prefix + "categorical key " + c.name + " not given and not unique: " + join_values(vals));
      }
      cat[c.name] = c.dict[*seen.begin()];
    }
  }
  for (const char* k : kDtypeKeys)
    if (auto it = cat.find(k); it != cat.end()) { dtype = it->second; break; }
  for (const auto& [k, _] : q.num) {
    const Column* c = t.find(k);
    if (!c || c->is_str() || c->kind == Column::Kind::Float) return Latency::unsupported(prefix + "unknown numeric key " + k);
  }
  std::vector<Axis> axes;
  const bool context = q.op.find("context") != std::string::npos;
  for (const Column& c : t.cols) {
    if (!is_key_column(c) || c.is_str()) continue;
    if (auto it = q.num.find(c.name); it != q.num.end()) {
      axes.push_back({&c, it->second, context && c.name == "isl"});
      continue;
    }
    std::set<double> seen;
    for (uint32_t r : rows) seen.insert(c.num[r]);
    if (seen.size() > 1) {
      std::set<std::string> vals;
      for (double v : seen) vals.insert(fmt(v));
      return Latency::unsupported(prefix + "numeric key " + c.name + " not given and not unique: " + join_values(vals));
    }
  }
  for (uint32_t r : rows) {
    bool hit = true;
    for (const Axis& a : axes)
      if (a.col->num[r] != a.q) { hit = false; break; }
    if (hit) return Latency::exact(lat->num[r], Source::Measured, prefix + "exact");
  }
  if (q.op == "gemm" && t.find("m") && t.find("n") && t.find("k") && q.num.count("m") && q.num.count("n") && q.num.count("k"))
    return query_gemm(t, rows, q, spec, dtype, prefix);

  const AttnKind kind = attention_kind(q.op);
  const bool seq_rule = kind == AttnKind::Generation || kind == AttnKind::CsaContext || kind == AttnKind::HcaContext;
  Extrap ex;
  // Cross-line rule for the generation grids only: on the context grids the own-line slope is the better predictor
  // (leave-one-out csa context 0.3% median with it, 1.3% with the cross-line rule).
  ex.line_axis = kind == AttnKind::Generation ? "batch_size" : "";
  if (kind != AttnKind::Generation)
    if (auto it = q.num.find("isl"); it != q.num.end()) ex.isl = it->second;
  Res r = Miss{};
  for (bool line_order : {false, true}) {   // first pure interpolation, then extrapolation on the grid lines
    std::stable_sort(axes.begin(), axes.end(), [&](const Axis& x, const Axis& y) {
      return axis_depth(x.col->name, line_order) < axis_depth(y.col->name, line_order);
    });
    Node root = build_trie(rows, axes, *lat);
    ex.used = false;
    ex.enabled = line_order && seq_rule;
    r = resolve(root, axes, 0, line_order ? &ex : nullptr);
    if (std::holds_alternative<double>(r)) break;
  }
  ex.enabled = seq_rule;
  if (std::holds_alternative<double>(r)) {
    if (!ex.used) return Latency::exact(std::get<double>(r), Source::Interpolated, prefix + "grid interpolation");
    const char* structure = kind == AttnKind::Generation ? " (kv_len-linear part + constant part)"
                            : kind == AttnKind::HcaContext ? " (kv_len/128-linear part + constant part)"
                            : kind == AttnKind::CsaContext ? " (indexer linear in isl+step + constant top-k part)" : "";
    return Latency::exact(std::get<double>(r), Source::Extrapolated, prefix + ex.note + structure);
  }
  if (std::holds_alternative<Miss>(r)) return Latency::unsupported(prefix + "no complete bracket in the measured grid");
  const OutOfRange oor = std::get<OutOfRange>(r);
  const std::string axis = axes[oor.axis].col->name;
  const double qv = axes[oor.axis].q;
  if (axis == "step" && oor.above) {
    if (kind == AttnKind::OtherContext)
      return Latency::unsupported(prefix + "step=" + fmt(qv) + " above the measured range; no sequence-axis extrapolation rule for " + q.op);
    if (ex.enabled) return Latency::unsupported(prefix + "sequence-axis extrapolation needs two resolvable edge points below step=" + fmt(qv));
  }
  return Latency::unsupported(prefix + axis + "=" + fmt(qv) + " " + (oor.above ? "above" : "below") + " the measured range" +
                              (kind != AttnKind::None && axis != "step" ? "; extrapolation covers the step axis only" : ""));
}

MeasuredTables::MeasuredTables(std::string perfdata_dir, std::map<std::string, DeviceSpec> specs)
    : dir_(std::move(perfdata_dir)), specs_(std::move(specs)) {
  simdjson::dom::parser parser;
  simdjson::dom::element doc = parser.load(dir_ + "/manifest.json");
  commit_ = std::string(std::string_view(doc["source_commit"]));
  for (simdjson::dom::element f : doc["files"]) {
    ImportedFile e;
    e.path = std::string(std::string_view(f["path"]));
    e.source = std::string(std::string_view(f["source"]));
    e.rows = uint64_t(f["rows"]);
    e.system = std::string(std::string_view(f["system"]));
    e.family = std::string(std::string_view(f["family"]));
    e.framework = std::string(std::string_view(f["framework"]));
    e.version = std::string(std::string_view(f["version"]));
    e.table = std::string(std::string_view(f["table"]));
    files_.push_back(e);
  }
  for (simdjson::dom::element s : doc["version_slots"]) {
    VersionSlots v{std::string(std::string_view(s["current"])), std::string(std::string_view(s["previous"])),
                   std::string(std::string_view(s["next"]))};
    slots_[{std::string(std::string_view(s["system"])), std::string(std::string_view(s["framework"]))}] = v;
  }
  for (simdjson::dom::element a : doc["aliases"]) {
    TableAlias t;
    t.system = std::string(std::string_view(a["system"]));
    t.framework = std::string(std::string_view(a["framework"]));
    t.version = std::string(std::string_view(a["version"]));
    t.table = std::string(std::string_view(a["table"]));
    for (simdjson::dom::element s : a["sources"]) {
      TableSource src;
      src.path = std::string(std::string_view(s["path"]));
      src.channel = std::string(std::string_view(s["channel"]));
      for (simdjson::dom::element k : s["kernel_sources"]) src.kernel_sources.push_back(std::string(std::string_view(k)));
      t.sources.push_back(src);
    }
    alias_index_[{t.system, t.framework, t.version, t.table}] = aliases_.size();
    aliases_.push_back(std::move(t));
  }
}

std::vector<std::string> MeasuredTables::systems() const {
  std::set<std::string> s;
  for (const auto& a : aliases_) s.insert(a.system);
  return {s.begin(), s.end()};
}

std::vector<std::string> MeasuredTables::versions(const std::string& system, const std::string& framework, const std::string& table) const {
  std::vector<std::string> v;
  for (const auto& a : aliases_)
    if (a.system == system && a.framework == framework && a.table == table) v.push_back(a.version);
  std::sort(v.begin(), v.end(), [](const std::string& a, const std::string& b) { return compare_versions(a, b) < 0; });
  return v;
}

std::string MeasuredTables::latest_version(const std::string& system, const std::string& framework, const std::string& table) const {
  const auto v = versions(system, framework, table);
  return v.empty() ? "" : v.back();
}

std::string MeasuredTables::resolve_version(const std::string& system, const std::string& framework, const std::string& version) const {
  if (version != "current" && version != "previous" && version != "next") return version;
  const auto it = slots_.find({system, framework});
  if (it == slots_.end()) return "";
  return version == "current" ? it->second.current : version == "previous" ? it->second.previous : it->second.next;
}

bool MeasuredTables::has(const std::string& system, const std::string& framework, const std::string& version, const std::string& table) const {
  return alias_index_.count({system, framework, version, table}) > 0;
}

namespace {

void append_row(Table& dst, const Table& src, uint32_t row) {
  for (const Column& sc : src.cols) {
    if (dst.find(sc.name)) continue;
    Column c;
    c.name = sc.name;
    c.kind = sc.kind;
    if (c.is_str()) {
      c.dict.push_back("");
      c.idx.assign(dst.rows, 0);
    } else {
      c.num.assign(dst.rows, std::nan(""));
    }
    dst.cols.push_back(std::move(c));
  }
  for (Column& dc : dst.cols) {
    const Column* sc = src.find(dc.name);
    if (dc.is_str()) dc.idx.push_back(sc && sc->is_str() ? dc.intern(sc->str(row)) : dc.intern(""));
    else dc.num.push_back(sc && !sc->is_str() ? sc->num[row] : std::nan(""));
  }
  ++dst.rows;
}

}  // namespace

std::shared_ptr<const Table> MeasuredTables::table(const std::string& system, const std::string& framework, const std::string& version,
                                                   const std::string& table) const {
  const auto it = alias_index_.find({system, framework, version, table});
  if (it == alias_index_.end()) return nullptr;
  std::lock_guard<std::mutex> lock(mu_);
  if (auto c = cache_.find(it->second); c != cache_.end()) return c->second;
  const TableAlias& alias = aliases_[it->second];
  auto merged = std::make_shared<Table>();
  std::unordered_set<std::string> keys;
  for (const TableSource& src : alias.sources) {
    const Table t = read_binary_table(dir_ + "/" + src.path);
    const Column* ks = t.find("kernel_source");
    std::set<uint32_t> allowed;
    if (!src.kernel_sources.empty()) {
      if (!ks) continue;
      for (uint32_t i = 0; i < ks->dict.size(); ++i)
        if (std::find(src.kernel_sources.begin(), src.kernel_sources.end(), ks->dict[i]) != src.kernel_sources.end()) allowed.insert(i);
    }
    for (uint32_t r = 0; r < t.rows; ++r) {
      if (!src.kernel_sources.empty() && !allowed.count(ks->idx[r])) continue;
      std::string key;
      for (const Column& c : t.cols) {
        if (!is_key_column(c)) continue;
        key += c.name;
        key += '=';
        key += c.is_str() ? c.str(r) : fmt(c.num[r]);
        key += '\x1f';
      }
      if (!keys.insert(key).second) continue;
      append_row(*merged, t, r);
    }
  }
  cache_[it->second] = merged;
  return merged;
}

Latency MeasuredTables::query(const std::string& device, const OpQuery& q) const {
  std::string key = device + "|" + q.op;
  for (const auto& [k, v] : q.num) key += "|" + k + "=" + fmt(v);
  for (const auto& [k, v] : q.cat) key += "|" + k + "=" + v;
  {
    std::lock_guard<std::mutex> lock(mu_);
    if (auto it = memo_.find(key); it != memo_.end()) return it->second;
  }
  Latency l = query_uncached(device, q);
  std::lock_guard<std::mutex> lock(mu_);
  memo_.emplace(std::move(key), l);
  return l;
}

Latency MeasuredTables::query_uncached(const std::string& device, const OpQuery& q) const {
  std::string framework, version;
  bool default_version = false;
  if (auto it = q.cat.find("framework"); it != q.cat.end()) framework = it->second;
  if (auto it = q.cat.find("version"); it != q.cat.end()) version = it->second;
  const auto spec_it = specs_.find(device);
  const DeviceSpec* spec = spec_it == specs_.end() ? nullptr : &spec_it->second;
  if (framework.empty() && spec) framework = spec->framework;
  if (framework.empty()) return Latency::unsupported("table=" + q.op + " system=" + device + ": framework not given");
  if (version.empty() && spec)
    if (auto it = spec->versions.find(framework); it != spec->versions.end()) version = it->second;
  if (version.empty()) {
    version = latest_version(spec && !spec->perf_tables.empty() ? spec->perf_tables : device, framework, q.op);
    default_version = true;
  }
  version = resolve_version(spec && !spec->perf_tables.empty() ? spec->perf_tables : device, framework, version);
  const std::string prefix = "table=" + q.op + " system=" + (spec && !spec->perf_tables.empty() ? spec->perf_tables : device) +
                             " framework=" + framework + " version=" + version +
                             (default_version ? " (default: latest)" : "") + ": ";
  const std::string system = spec && !spec->perf_tables.empty() ? spec->perf_tables : device;
  auto t = table(system, framework, version, q.op);
  if (!t) return Latency::unsupported(prefix + "no such table");
  std::vector<uint32_t> rows(t->rows);
  for (uint32_t r = 0; r < t->rows; ++r) rows[r] = r;
  return query_rows(*t, rows, q, spec, prefix);
}

}  // namespace dlsim
