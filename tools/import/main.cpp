#include <arrow/api.h>
#include <arrow/io/api.h>
#include <parquet/arrow/reader.h>
#include <yaml-cpp/yaml.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "perfdata/binary_table.h"
#include "perfdata/version.h"

namespace fs = std::filesystem;
using dlsim::Column;
using dlsim::Table;

namespace {

struct Imported {
  std::string system, family, framework, version, table;
  std::string source;
  std::string out;
  uint64_t rows = 0;
  int64_t source_rows = 0;
};

struct ReuseEntry {
  std::string table, from_version;
};

struct ManifestGroup {
  std::string table, kernel_source, tier;
  std::vector<std::string> frameworks;
};

std::string table_name(const std::string& stem) {
  const std::string suffix = "_perf";
  if (stem.size() > suffix.size() && stem.compare(stem.size() - suffix.size(), suffix.size(), suffix) == 0)
    return stem.substr(0, stem.size() - suffix.size());
  return stem;
}

Table convert(const std::shared_ptr<arrow::Table>& at) {
  Table t;
  t.rows = at->num_rows();
  for (int ci = 0; ci < at->num_columns(); ++ci) {
    const auto& field = at->schema()->field(ci);
    const auto chunks = at->column(ci);
    Column c;
    c.name = field->name();
    switch (field->type()->id()) {
      case arrow::Type::INT64:
      case arrow::Type::INT32:
      case arrow::Type::UINT64:
      case arrow::Type::UINT32:
        c.kind = Column::Kind::Int;
        break;
      case arrow::Type::DOUBLE:
      case arrow::Type::FLOAT:
        c.kind = Column::Kind::Float;
        break;
      case arrow::Type::BOOL:
        c.kind = Column::Kind::Bool;
        break;
      case arrow::Type::STRING:
      case arrow::Type::LARGE_STRING:
        c.kind = Column::Kind::Str;
        break;
      default:
        throw std::runtime_error("unsupported column type " + field->type()->ToString() + " in " + c.name);
    }
    if (c.is_str()) {
      c.idx.reserve(t.rows);
      std::map<std::string, uint32_t> seen;
      for (const auto& ch : chunks->chunks()) {
        for (int64_t r = 0; r < ch->length(); ++r) {
          std::string s;
          if (ch->IsNull(r)) s = "";
          else if (ch->type_id() == arrow::Type::STRING) s = std::static_pointer_cast<arrow::StringArray>(ch)->GetString(r);
          else s = std::static_pointer_cast<arrow::LargeStringArray>(ch)->GetString(r);
          auto it = seen.find(s);
          if (it == seen.end()) {
            it = seen.emplace(s, static_cast<uint32_t>(c.dict.size())).first;
            c.dict.push_back(s);
          }
          c.idx.push_back(it->second);
        }
      }
    } else {
      c.num.reserve(t.rows);
      for (const auto& ch : chunks->chunks()) {
        for (int64_t r = 0; r < ch->length(); ++r) {
          double v = std::nan("");
          if (!ch->IsNull(r)) {
            switch (ch->type_id()) {
              case arrow::Type::INT64: v = std::static_pointer_cast<arrow::Int64Array>(ch)->Value(r); break;
              case arrow::Type::INT32: v = std::static_pointer_cast<arrow::Int32Array>(ch)->Value(r); break;
              case arrow::Type::UINT64: v = std::static_pointer_cast<arrow::UInt64Array>(ch)->Value(r); break;
              case arrow::Type::UINT32: v = std::static_pointer_cast<arrow::UInt32Array>(ch)->Value(r); break;
              case arrow::Type::DOUBLE: v = std::static_pointer_cast<arrow::DoubleArray>(ch)->Value(r); break;
              case arrow::Type::FLOAT: v = std::static_pointer_cast<arrow::FloatArray>(ch)->Value(r); break;
              case arrow::Type::BOOL: v = std::static_pointer_cast<arrow::BooleanArray>(ch)->Value(r) ? 1.0 : 0.0; break;
              default: break;
            }
          }
          c.num.push_back(v);
        }
      }
    }
    t.cols.push_back(std::move(c));
  }
  return t;
}

Table read_parquet(const std::string& path, int64_t& source_rows) {
  auto infile = arrow::io::ReadableFile::Open(path).ValueOrDie();
  auto opened = parquet::arrow::OpenFile(infile, arrow::default_memory_pool());
  if (!opened.ok()) throw std::runtime_error("parquet open failed: " + path + ": " + opened.status().ToString());
  source_rows = (*opened)->parquet_reader()->metadata()->num_rows();
  std::shared_ptr<arrow::Table> at;
  const auto st = (*opened)->ReadTable(&at);
  if (!st.ok()) throw std::runtime_error("parquet read failed: " + path + ": " + st.ToString());
  return convert(at);
}

std::string json_str(const std::string& s) {
  std::string o = "\"";
  for (char c : s) {
    switch (c) {
      case '"': o += "\\\""; break;
      case '\\': o += "\\\\"; break;
      case '\n': o += "\\n"; break;
      default: o += c;
    }
  }
  return o + "\"";
}

std::string run(const std::string& cmd) {
  std::string out;
  FILE* p = popen(cmd.c_str(), "r");
  if (!p) return out;
  char buf[256];
  while (fgets(buf, sizeof buf, p)) out += buf;
  pclose(p);
  while (!out.empty() && (out.back() == '\n' || out.back() == ' ')) out.pop_back();
  return out;
}

std::string now_iso() {
  const auto t = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
  char buf[32];
  std::strftime(buf, sizeof buf, "%Y-%m-%dT%H:%M:%SZ", std::gmtime(&t));
  return buf;
}

struct Alias {
  std::string system, framework, version, table;
  struct Src {
    std::string path, channel;
    std::vector<std::string> kernel_sources;
  };
  std::vector<Src> sources;
};

}  // namespace

int main(int argc, char** argv) {
  if (argc != 3) {
    std::cerr << "usage: dlsim-import <aisimulate_systems_dir> <out_dir>\n";
    return 2;
  }
  const fs::path systems_dir = argv[1];
  const fs::path data_dir = systems_dir / "data";
  const fs::path out_dir = argv[2];
  if (!fs::is_directory(data_dir)) {
    std::cerr << "not a directory: " << data_dir << "\n";
    return 1;
  }

  std::vector<Imported> files;
  // (system, framework) -> versions declared by any family dir
  std::map<std::pair<std::string, std::string>, std::set<std::string>> versions;
  // (system, framework, version, table) -> imported index
  std::map<std::tuple<std::string, std::string, std::string, std::string>, size_t> primary;
  // (system, table) -> family
  std::map<std::pair<std::string, std::string>, std::string> family_of;
  // (system, framework, version) -> reuse entries
  std::map<std::tuple<std::string, std::string, std::string>, std::vector<ReuseEntry>> reuse;

  std::vector<fs::path> parquet_files;
  for (const auto& f : fs::recursive_directory_iterator(data_dir)) {
    if (!f.is_regular_file()) continue;
    const fs::path rel = fs::relative(f.path(), data_dir);
    std::vector<std::string> parts;
    for (const auto& p : rel) parts.push_back(p.string());
    if (f.path().filename() == "reuse.yaml" && parts.size() == 5) {
      YAML::Node n = YAML::LoadFile(f.path());
      for (const auto& e : n["reuse"])
        reuse[{parts[0], parts[2], parts[3]}].push_back({table_name(e["table"].as<std::string>()), e["from_version"].as<std::string>()});
      continue;
    }
    if (f.path().extension() == ".parquet" && (parts.size() == 4 || parts.size() == 5)) parquet_files.push_back(f.path());
  }
  std::sort(parquet_files.begin(), parquet_files.end());
  for (const fs::path& p : parquet_files) {
    const fs::path rel = fs::relative(p, data_dir);
    std::vector<std::string> parts;
    for (const auto& q : rel) parts.push_back(q.string());
    Imported im;
    im.system = parts[0];
    im.family = parts.size() == 5 ? parts[1] : "";
    im.framework = parts[parts.size() - 3];
    im.version = parts[parts.size() - 2];
    im.table = table_name(p.stem());
    im.source = fs::relative(p, systems_dir).string();
    im.out = (rel.parent_path() / (im.table + ".bin")).string();
    versions[{im.system, im.framework}].insert(im.version);
    fs::create_directories(out_dir / rel.parent_path());
    Table t = read_parquet(p, im.source_rows);
    im.rows = t.rows;
    // moe_a2a stores latency in microseconds (AISimulate divides it by 1000 on read); every other table is in ms.
    if (im.table == "moe_a2a")
      for (Column& c : t.cols)
        if (c.name == "latency")
          for (double& v : c.num) v /= 1000.0;
    dlsim::write_binary_table((out_dir / im.out).string(), t);
    std::cerr << im.out << " " << im.rows << "\n";
    primary[{im.system, im.framework, im.version, im.table}] = files.size();
    family_of[{im.system, im.table}] = im.family;
    files.push_back(im);
  }
  for (const auto& [key, _] : reuse) versions[{std::get<0>(key), std::get<1>(key)}].insert(std::get<2>(key));

  std::vector<ManifestGroup> groups;
  {
    YAML::Node n = YAML::LoadFile((systems_dir / "perf_data_reuse_manifest.yaml").string());
    for (const auto& g : n["groups"]) {
      ManifestGroup mg;
      mg.table = table_name(fs::path(g["op_file"].as<std::string>()).stem());
      mg.kernel_source = g["kernel_source"].as<std::string>();
      mg.tier = g["tier"].as<std::string>();
      for (const auto& f : g["frameworks"]) mg.frameworks.push_back(f.as<std::string>());
      groups.push_back(mg);
    }
  }

  std::map<std::pair<std::string, std::string>, std::pair<std::string, std::string>> slots;  // (system, fw) -> current, previous
  {
    YAML::Node n = YAML::LoadFile((systems_dir / "query_versions.yaml").string());
    auto read = [](const YAML::Node& m, std::pair<std::string, std::string>& out) {
      out.first = m["current"] && !m["current"].IsNull() ? m["current"].as<std::string>() : "";
      out.second = m["previous"] && !m["previous"].IsNull() ? m["previous"].as<std::string>() : "";
    };
    std::map<std::string, std::pair<std::string, std::string>> defaults;
    for (const auto& kv : n["defaults"]) read(kv.second, defaults[kv.first.as<std::string>()]);
    for (const auto& [key, _] : versions) {
      auto it = defaults.find(key.second);
      if (it != defaults.end()) slots[key] = it->second;
    }
    for (const auto& ov : n["overrides"])
      for (const auto& kv : ov.second) read(kv.second, slots[{ov.first.as<std::string>(), kv.first.as<std::string>()}]);
  }

  const auto sorted_versions = [](const std::set<std::string>& s, bool newest_first) {
    std::vector<std::string> v(s.begin(), s.end());
    std::sort(v.begin(), v.end(), [&](const std::string& a, const std::string& b) {
      const int c = dlsim::compare_versions(a, b);
      return newest_first ? c > 0 : c < 0;
    });
    return v;
  };
  const std::set<std::string> versioned_comm = {"sglang", "trtllm", "vllm"};

  std::vector<Alias> aliases;
  for (const auto& [key, vers] : versions) {
    const auto& [system, framework] = key;
    std::set<std::string> tables;
    for (const auto& [k, _] : family_of)
      if (k.first == system) tables.insert(k.second);
    for (const std::string& version : vers) {
      for (const std::string& table : tables) {
        const std::string family = family_of[{system, table}];
        const bool agnostic = table == "nccl" || table == "oneccl";
        const bool comm = family == "comm";
        Alias a{system, framework, version, table, {}};
        auto add = [&](const std::string& fw, const std::string& v, const char* channel, std::vector<std::string> ks) {
          auto it = primary.find({system, fw, v, table});
          if (it == primary.end()) return false;
          a.sources.push_back({files[it->second].out, channel, std::move(ks)});
          return true;
        };
        add(framework, version, "primary", {});
        if (agnostic || (comm && !versioned_comm.count(framework))) {
          if (!a.sources.empty()) aliases.push_back(a);
          continue;
        }
        std::set<std::string> declared;
        if (!comm) {
          auto it = reuse.find({system, framework, version});
          if (it != reuse.end())
            for (const auto& e : it->second)
              if (e.table == table && !declared.count(e.from_version) && add(framework, e.from_version, "declared_reuse", {}))
                declared.insert(e.from_version);
        }
        for (const std::string& v : sorted_versions(vers, true)) {
          if (v == version || declared.count(v) || !dlsim::parse_pep440(v).ok || dlsim::compare_versions(v, version) >= 0) continue;
          add(framework, v, "fallback", {});
        }
        if (!comm) {
          std::map<std::string, std::set<std::string>> ks_filter;
          for (const auto& g : groups) {
            if (g.table != table || (g.tier != "shared" && g.tier != "shared_fallback")) continue;
            if (std::find(g.frameworks.begin(), g.frameworks.end(), framework) == g.frameworks.end()) continue;
            for (const auto& fw : g.frameworks)
              if (fw != framework) ks_filter[fw].insert(g.kernel_source);
          }
          for (const auto& [fw, ks] : ks_filter) {
            auto vit = versions.find({system, fw});
            if (vit == versions.end()) continue;
            for (const std::string& v : sorted_versions(vit->second, true))
              add(fw, v, "cross_backend", std::vector<std::string>(ks.begin(), ks.end()));
          }
        }
        if (!a.sources.empty()) aliases.push_back(a);
      }
    }
  }

  const std::string commit = run("git -C " + (systems_dir / "../../../../..").lexically_normal().string() + " rev-parse HEAD 2>/dev/null");
  std::ofstream m(out_dir / "manifest.json");
  m << "{\n";
  m << "  \"source_commit\": " << json_str(commit) << ",\n";
  m << "  \"systems_dir\": " << json_str(fs::absolute(systems_dir).lexically_normal().string()) << ",\n";
  m << "  \"imported_at\": " << json_str(now_iso()) << ",\n";
  m << "  \"files\": [\n";
  for (size_t i = 0; i < files.size(); ++i) {
    const auto& f = files[i];
    m << "    {\"path\": " << json_str(f.out) << ", \"source\": " << json_str(f.source) << ", \"rows\": " << f.rows << ", \"source_rows\": " << f.source_rows
      << ", \"system\": " << json_str(f.system) << ", \"family\": " << json_str(f.family) << ", \"framework\": "
      << json_str(f.framework) << ", \"version\": " << json_str(f.version) << ", \"table\": " << json_str(f.table) << "}"
      << (i + 1 < files.size() ? ",\n" : "\n");
  }
  m << "  ],\n  \"version_slots\": [\n";
  {
    size_t i = 0;
    for (const auto& [key, cp] : slots) {
      std::string next;
      auto vit = versions.find(key);
      if (vit != versions.end() && !cp.first.empty())
        for (const std::string& v : sorted_versions(vit->second, true))
          if (dlsim::parse_pep440(v).ok && dlsim::compare_versions(v, cp.first) > 0) {
            next = v;
            break;
          }
      m << "    {\"system\": " << json_str(key.first) << ", \"framework\": " << json_str(key.second) << ", \"current\": "
        << json_str(cp.first) << ", \"previous\": " << json_str(cp.second) << ", \"next\": " << json_str(next) << "}"
        << (++i < slots.size() ? ",\n" : "\n");
    }
  }
  m << "  ],\n  \"aliases\": [\n";
  for (size_t i = 0; i < aliases.size(); ++i) {
    const auto& a = aliases[i];
    m << "    {\"system\": " << json_str(a.system) << ", \"framework\": " << json_str(a.framework) << ", \"version\": "
      << json_str(a.version) << ", \"table\": " << json_str(a.table) << ", \"sources\": [";
    for (size_t j = 0; j < a.sources.size(); ++j) {
      const auto& s = a.sources[j];
      m << (j ? ", " : "") << "{\"path\": " << json_str(s.path) << ", \"channel\": " << json_str(s.channel) << ", \"kernel_sources\": [";
      for (size_t k = 0; k < s.kernel_sources.size(); ++k) m << (k ? ", " : "") << json_str(s.kernel_sources[k]);
      m << "]}";
    }
    m << "]}" << (i + 1 < aliases.size() ? ",\n" : "\n");
  }
  m << "  ]\n}\n";
  uint64_t total = 0;
  for (const auto& f : files) total += f.rows;
  std::cout << "files " << files.size() << " rows " << total << " aliases " << aliases.size() << " commit " << commit << "\n";
  return 0;
}
