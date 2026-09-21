#include "perfdata/reports.h"

#include <algorithm>
#include <cmath>
#include <random>
#include <set>
#include <sstream>

#include "perfdata/decomposition.h"
#include "perfdata/dtype.h"

#include <simdjson.h>

namespace dlsim {
namespace {

std::string js(const std::string& s) {
  std::string o = "\"";
  for (char c : s) {
    if (c == '"') o += "\\\"";
    else if (c == '\\') o += "\\\\";
    else if (c == '\n') o += "\\n";
    else o += c;
  }
  return o + "\"";
}

std::string jn(double v) {
  if (std::isnan(v) || std::isinf(v)) return "null";
  std::ostringstream o;
  o.precision(6);
  o << v;
  return o.str();
}

struct Stats {
  std::vector<double> abs, signed_;
  void add(double pred, double meas) {
    const double e = (pred - meas) / meas;
    abs.push_back(std::abs(e));
    signed_.push_back(e);
  }
  static double pct(std::vector<double> v, double p) {
    if (v.empty()) return std::nan("");
    std::sort(v.begin(), v.end());
    return v[static_cast<size_t>(p * (v.size() - 1))];
  }
  std::string json() const {
    std::ostringstream o;
    o << "{\"n\": " << abs.size() << ", \"median\": " << jn(pct(abs, 0.5)) << ", \"p90\": " << jn(pct(abs, 0.9))
      << ", \"signed_median\": " << jn(pct(signed_, 0.5)) << "}";
    return o.str();
  }
};

std::vector<const Column*> cat_columns(const Table& t) {
  std::vector<const Column*> v;
  for (const Column& c : t.cols)
    if (c.is_str() && is_key_column(c)) v.push_back(&c);
  return v;
}

std::vector<const Column*> num_columns(const Table& t) {
  std::vector<const Column*> v;
  for (const Column& c : t.cols)
    if (!c.is_str() && is_key_column(c)) v.push_back(&c);
  return v;
}

// Rows grouped by the full categorical key; the group label is "k=v k=v".
std::map<std::string, std::vector<uint32_t>> slices(const Table& t) {
  std::map<std::string, std::vector<uint32_t>> out;
  const auto cats = cat_columns(t);
  for (uint32_t r = 0; r < t.rows; ++r) {
    std::string key;
    for (const Column* c : cats) key += (key.empty() ? "" : " ") + c->name + "=" + c->str(r);
    out[key].push_back(r);
  }
  return out;
}

OpQuery row_query(const Table& t, uint32_t r, const std::string& op) {
  OpQuery q;
  q.op = op;
  for (const Column* c : cat_columns(t)) q.cat[c->name] = c->str(r);
  for (const Column* c : num_columns(t)) q.num[c->name] = c->num[r];
  return q;
}

std::vector<uint32_t> sample(const std::vector<uint32_t>& rows, size_t n, uint64_t seed) {
  std::vector<uint32_t> v = rows;
  std::mt19937_64 rng(seed);
  std::shuffle(v.begin(), v.end(), rng);
  if (v.size() > n) v.resize(n);
  std::sort(v.begin(), v.end());
  return v;
}

std::string slice_json(const std::string& label, const Stats& s, const std::map<std::string, int>& other) {
  std::ostringstream o;
  o << "{\"slice\": " << js(label) << ", \"interpolated\": " << s.json() << ", \"other\": {";
  bool first = true;
  for (const auto& [k, v] : other) {
    o << (first ? "" : ", ") << js(k) << ": " << v;
    first = false;
  }
  o << "}}";
  return o.str();
}

double query_lat(const Table& t, uint32_t r) { return t.find("latency")->num[r]; }

}  // namespace

std::string import_row_counts(const MeasuredTables& tables) {
  std::ostringstream o;
  uint64_t bin_total = 0, manifest_total = 0, source_total = 0;
  std::vector<std::string> mismatches;
  o << "{\"source_commit\": " << js(tables.source_commit()) << ", \"files\": " << tables.files().size() << ", \"mismatches\": [";
  for (const ImportedFile& f : tables.files()) {
    const uint64_t bin = binary_table_rows(tables.dir() + "/" + f.path);
    bin_total += bin;
    manifest_total += f.rows;
    if (bin != f.rows) mismatches.push_back(f.path);
  }
  {
    simdjson::dom::parser parser;
    simdjson::dom::element doc = parser.load(tables.dir() + "/manifest.json");
    for (simdjson::dom::element f : doc["files"]) {
      const uint64_t src = uint64_t(f["source_rows"]);
      source_total += src;
      if (src != uint64_t(f["rows"])) mismatches.push_back(std::string(std::string_view(f["path"])) + " (parquet metadata)");
    }
  }
  for (size_t i = 0; i < mismatches.size(); ++i) o << (i ? ", " : "") << js(mismatches[i]);
  o << "], \"bin_rows\": " << bin_total << ", \"manifest_rows\": " << manifest_total << ", \"parquet_metadata_rows\": " << source_total
    << ", \"consistent\": " << (mismatches.empty() && bin_total == source_total ? "true" : "false") << "}";
  return o.str();
}

std::string interp_leave_one_out(const MeasuredTables& tables, const std::string& system, const std::string& framework,
                                 const std::string& version, const std::string& table, int folds, uint64_t seed) {
  auto t = tables.table(system, framework, version, table);
  if (!t) return "{\"error\": \"no such table\"}";
  const DeviceSpec* spec = nullptr;
  if (auto it = tables.specs().find(system); it != tables.specs().end()) spec = &it->second;
  Stats all, all_site;
  std::map<std::string, int> other_all;
  std::ostringstream slices_json;
  bool first = true;
  const Column *cn = t->find("n"), *ck = t->find("k");
  const bool gemm = table == "gemm" && cn && ck;
  for (const auto& [label, rows] : slices(*t)) {
    Stats s, site;
    std::map<std::string, int> other;
    for (uint32_t r : sample(rows, folds, seed)) {
      const OpQuery q = row_query(*t, r, table);
      std::vector<uint32_t> rest;
      rest.reserve(rows.size());
      for (uint32_t x : rows)
        if (x != r) rest.push_back(x);
      const Latency l = query_rows(*t, rest, q, spec);
      if (l.source == Source::Interpolated) {
        s.add(l.ms, query_lat(*t, r));
        all.add(l.ms, query_lat(*t, r));
      } else {
        ++other[to_string(l.source)];
        ++other_all[to_string(l.source)];
      }
      if (gemm) {
        std::vector<uint32_t> no_site;
        for (uint32_t x : rows)
          if (cn->num[x] != cn->num[r] || ck->num[x] != ck->num[r]) no_site.push_back(x);
        const Latency ls = query_rows(*t, no_site, q, spec);
        if (ls.source == Source::Interpolated) {
          site.add(ls.ms, query_lat(*t, r));
          all_site.add(ls.ms, query_lat(*t, r));
        } else {
          ++other["site_holdout_" + std::string(to_string(ls.source))];
          ++other_all["site_holdout_" + std::string(to_string(ls.source))];
        }
      }
    }
    slices_json << (first ? "" : ", ") << slice_json(label, s, other);
    if (gemm) {
      std::string sj = slices_json.str();
      sj.pop_back();
      slices_json.str("");
      slices_json << sj << ", \"site_holdout\": " << site.json() << "}";
    }
    first = false;
  }
  std::ostringstream o;
  o << "{\"system\": " << js(system) << ", \"framework\": " << js(framework) << ", \"version\": " << js(version) << ", \"table\": "
    << js(table) << ", \"rows\": " << t->rows << ", \"folds_per_slice\": " << folds << ", \"point_holdout\": " << all.json();
  if (gemm) o << ", \"site_holdout\": " << all_site.json();
  o << ", \"other\": {";
  bool f2 = true;
  for (const auto& [k, v] : other_all) {
    o << (f2 ? "" : ", ") << js(k) << ": " << v;
    f2 = false;
  }
  o << "}, \"slices\": [" << slices_json.str() << "]}";
  return o.str();
}

std::string extrapolation_check_sglang(const MeasuredTables& tables, const std::string& system, bool ragged, int samples, uint64_t seed) {
  const char* names[] = {"dsv4_csa_context_module", "dsv4_hca_context_module", "dsv4_csa_generation_module", "dsv4_hca_generation_module"};
  const double limit = 65536;
  std::ostringstream o;
  // ragged: generation lines are cut at batch-dependent steps in the shape of the trtllm and vllm grids (whose lines end
  // at 65,536 / 32,768 / 16,384 / 6,144 as batch grows), so the cross-line rule is what gets tested, at query/edge ratios
  // up to the 1,048,576 rows: batch <= 2 keeps everything, batch 4 to 65,536, batch 8 to 16,384, batch >= 16 to 6,144.
  o << "{\"system\": " << js(system) << ", \"framework\": \"sglang\", \"version\": \"0.5.14\", \"truncate_at\": " << limit
    << ", \"mode\": " << js(ragged ? "ragged" : "uniform") << ", \"tables\": [";
  bool first_t = true;
  for (const char* name : names) {
    auto t = tables.table(system, "sglang", "0.5.14", name);
    if (!t) continue;
    const bool context = std::string(name).find("context") != std::string::npos;
    const Column *ci = t->find("isl"), *cs = t->find("step"), *cb = t->find("batch_size");
    std::vector<uint32_t> kept, held;
    for (uint32_t r = 0; r < t->rows; ++r) {
      const double b = cb ? cb->num[r] : 1;
      const double line_limit = !(ragged && !context) ? limit : b <= 2 ? 1e12 : b <= 4 ? limit : b <= 8 ? 16384 : 6144;
      ((context ? ci->num[r] : 0) + cs->num[r] <= line_limit ? kept : held).push_back(r);
    }
    Stats all, proportional;
    std::map<std::string, Stats> by_ratio, by_rule;
    std::map<std::string, int> other;
    std::map<std::string, std::vector<uint32_t>> kept_by_slice;
    const auto cats = cat_columns(*t);
    const auto slice_key = [&](uint32_t r) {
      std::string key;
      for (const Column* c : cats) key += c->str(r) + "\x1f";
      return key;
    };
    for (uint32_t r : kept) kept_by_slice[slice_key(r)].push_back(r);
    for (uint32_t r : sample(held, samples, seed)) {
      const OpQuery q = row_query(*t, r, name);
      const Latency l = query_rows(*t, kept_by_slice[slice_key(r)], q, nullptr);
      if (l.source != Source::Extrapolated) {
        ++other[to_string(l.source)];
        continue;
      }
      const double meas = query_lat(*t, r);
      all.add(l.ms, meas);
      const auto pos = l.note.find("query/edge=");
      const double ratio = pos == std::string::npos ? 0 : std::atof(l.note.c_str() + pos + 11);
      const auto epos = l.note.find("edge_latency=");
      if (epos != std::string::npos) proportional.add(std::atof(l.note.c_str() + epos + 13) * ratio, meas);
      const char* bucket = ratio <= 2 ? "(1,2]" : ratio <= 4 ? "(2,4]" : ratio <= 8 ? "(4,8]" : ratio <= 16 ? "(8,16]" : ">16";
      by_ratio[bucket].add(l.ms, meas);
      by_rule[l.note.find("shape from") != std::string::npos ? "cross_line" : "own_line"].add(l.ms, meas);
    }
    o << (first_t ? "" : ", ") << "{\"table\": " << js(name) << ", \"rows_kept\": " << kept.size() << ", \"rows_beyond\": " << held.size()
      << ", \"sampled\": " << std::min<size_t>(held.size(), samples) << ", \"error\": " << all.json()
      << ", \"error_if_proportional\": " << proportional.json() << ", \"by_query_edge_ratio\": {";
    bool f = true;
    for (const auto& [k, s] : by_ratio) {
      o << (f ? "" : ", ") << js(k) << ": " << s.json();
      f = false;
    }
    o << "}, \"by_rule\": {";
    f = true;
    for (const auto& [k, s] : by_rule) {
      o << (f ? "" : ", ") << js(k) << ": " << s.json();
      f = false;
    }
    o << "}, \"other\": {";
    f = true;
    for (const auto& [k, v] : other) {
      o << (f ? "" : ", ") << js(k) << ": " << v;
      f = false;
    }
    o << "}}";
    first_t = false;
  }
  o << "]}";
  return o.str();
}

std::string layer1_leave_one_device_out(const MeasuredTables& tables, const std::map<std::string, DeviceSpec>& devices,
                                        const std::string& framework, const std::string& version, int samples, uint64_t seed) {
  const char* ops[] = {"gemm", "moe", "dsv4_csa_context_module", "dsv4_csa_generation_module", "dsv4_hca_context_module",
                       "dsv4_hca_generation_module"};
  struct Group {
    Stats point;
    int n = 0, covered = 0, with_interval = 0;
    std::map<std::string, int> sources;
  };
  std::map<std::string, Group> groups;
  std::map<std::string, int> ranks;  // "op|dtype|device|rank" -> count
  std::map<std::string, double> conds;
  std::ostringstream unsupported;
  int unsupported_n = 0;
  for (const char* op : ops) {
    std::vector<std::string> names;
    for (const auto& [name, _] : devices)
      if (tables.has(name, framework, version, op)) names.push_back(name);
    if (names.empty()) continue;
    DecomposedSource src(tables, names, devices);
    for (const std::string& d : names) {
      auto t = tables.table(d, framework, version, op);
      std::vector<std::string> others;
      for (const std::string& o : names)
        if (o != d) others.push_back(o);
      for (const auto& [label, rows] : slices(*t)) {
        for (uint32_t r : sample(rows, samples, seed)) {
          OpQuery q = row_query(*t, r, op);
          q.cat["framework"] = framework;
          q.cat["version"] = version;
          std::string dtype;
          for (const char* k : kDtypeKeys)
            if (auto it = q.cat.find(k); it != q.cat.end()) { dtype = it->second; break; }
          const double meas = query_lat(*t, r);
          std::string bucket;
          if (std::string(op) == "gemm" || std::string(op) == "moe") {
            const double x = q.num.count("m") ? q.num["m"] : q.num["num_tokens"];
            bucket = x <= 64 ? "tokens<=64" : x <= 512 ? "tokens<=512" : "tokens>512";
          } else {
            const double x = q.num["isl"] + q.num["step"];
            bucket = x <= 4096 ? "ctx<=4096" : x <= 65536 ? "ctx<=65536" : "ctx>65536";
          }
          const Latency l = src.query_with_references(d, q, others);
          Group& g = groups[std::string(op) + "|" + dtype + "|" + bucket];
          ++g.n;
          ++g.sources[to_string(l.source)];
          if (l.has_point()) g.point.add(l.ms, meas);
          if (l.source != Source::Unsupported) {
            ++g.with_interval;
            if (l.lo_ms <= meas && meas <= l.hi_ms) ++g.covered;
            const auto pos = l.note.find("rank=");
            const auto cpos = l.note.find("cond=");
            if (pos != std::string::npos) {
              const std::string rank = l.note.substr(pos + 5, l.note.find(' ', pos) - pos - 5);
              const std::string key = std::string(op) + "|" + dtype + "|" + d + "|" + rank;
              ++ranks[key];
              if (cpos != std::string::npos) conds[key] = std::max(conds[key], std::atof(l.note.c_str() + cpos + 5));
            }
          } else if (unsupported_n++ < 20) {
            unsupported << (unsupported_n > 1 ? ", " : "") << js(d + " " + op + ": " + l.note);
          }
        }
      }
    }
  }
  std::ostringstream o;
  o << "{\"framework\": " << js(framework) << ", \"version\": " << js(version) << ", \"samples_per_slice\": " << samples
    << ", \"condition_limit\": " << kConditionLimit << ", \"groups\": [";
  bool first = true;
  for (const auto& [key, g] : groups) {
    const auto p1 = key.find('|'), p2 = key.rfind('|');
    o << (first ? "" : ", ") << "{\"op\": " << js(key.substr(0, p1)) << ", \"dtype\": " << js(key.substr(p1 + 1, p2 - p1 - 1))
      << ", \"shape\": " << js(key.substr(p2 + 1)) << ", \"n\": " << g.n << ", \"point_error\": " << g.point.json()
      << ", \"interval_coverage\": " << jn(g.with_interval ? double(g.covered) / g.with_interval : std::nan(""))
      << ", \"with_interval\": " << g.with_interval << ", \"sources\": {";
    bool f = true;
    for (const auto& [k, v] : g.sources) {
      o << (f ? "" : ", ") << js(k) << ": " << v;
      f = false;
    }
    o << "}}";
    first = false;
  }
  o << "], \"rank_by_op_dtype_device\": [";
  first = true;
  for (const auto& [key, n] : ranks) {
    std::vector<std::string> parts;
    std::stringstream ss(key);
    std::string part;
    while (std::getline(ss, part, '|')) parts.push_back(part);
    o << (first ? "" : ", ") << "{\"op\": " << js(parts[0]) << ", \"dtype\": " << js(parts[1]) << ", \"held_out\": " << js(parts[2])
      << ", \"rank\": " << js(parts[3]) << ", \"n\": " << n << ", \"max_cond\": " << jn(conds[key]) << "}";
    first = false;
  }
  o << "], \"unsupported_examples\": [" << unsupported.str() << "], \"unsupported_total\": " << unsupported_n << "}";
  return o.str();
}

}  // namespace dlsim
