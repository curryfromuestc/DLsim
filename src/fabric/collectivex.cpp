#include "fabric/collectivex.h"

#include <simdjson.h>

#include <algorithm>
#include <filesystem>
#include <stdexcept>

namespace dlsim {
namespace {

std::string text(simdjson::dom::element e) { return std::string{std::string_view(e)}; }
std::string text_or_empty(simdjson::dom::element e) { return e.is_null() ? std::string() : text(e); }

}  // namespace

std::vector<CollectiveXRow> load_collectivex_run(const std::string& path) {
  simdjson::dom::parser parser;
  simdjson::dom::element doc = parser.load(path);
  std::vector<CollectiveXRow> rows;
  const std::string run_id = text(doc["run"]["run_id"]);
  simdjson::dom::element series;
  if (doc["series"].get(series) != simdjson::SUCCESS || series.is_null()) return rows;
  for (simdjson::dom::element s : simdjson::dom::array(series)) {
    CollectiveXRow base;
    base.run_id = run_id;
    base.series_id = text(s["series_id"]);
    base.phase = text(s["phase"]);
    base.mode = text(s["mode"]);
    base.precision = text(s["precision"]);
    base.backend = text(s["backend"]);
    simdjson::dom::element sys = s["system"];
    base.sku = text(sys["sku"]);
    base.ep = int(int64_t(sys["ep_size"]));
    base.nodes = int(int64_t(sys["nodes"]));
    base.gpus_per_node = int(int64_t(sys["gpus_per_node"]));
    base.scale_up_domain = int(int64_t(sys["scale_up_domain"]));
    base.scale_up_transport = text_or_empty(sys["scale_up_transport"]);
    base.scale_out_transport = text_or_empty(sys["scale_out_transport"]);
    for (simdjson::dom::element p : simdjson::dom::array(s["points"])) {
      CollectiveXRow r = base;
      r.tokens_per_rank = int(int64_t(p["tokens_per_rank"]));
      simdjson::dom::element comp = p["components"];
      simdjson::dom::element d = comp["dispatch"], c = comp["combine"];
      r.dispatch_payload_bytes = double(d["payload_bytes"]);
      r.combine_payload_bytes = double(c["payload_bytes"]);
      r.dispatch_p50_us = double(d["latency_us"]["p50"]);
      r.combine_p50_us = double(c["latency_us"]["p50"]);
      rows.push_back(std::move(r));
    }
  }
  return rows;
}

std::vector<CollectiveXRow> load_collectivex_dir(const std::string& dir) {
  std::vector<std::string> files;
  for (const auto& e : std::filesystem::directory_iterator(dir)) {
    const std::string name = e.path().filename().string();
    if (name.rfind("run_", 0) == 0 && e.path().extension() == ".json") files.push_back(e.path().string());
  }
  std::sort(files.begin(), files.end());
  std::vector<CollectiveXRow> rows;
  for (const auto& f : files) {
    auto r = load_collectivex_run(f);
    rows.insert(rows.end(), r.begin(), r.end());
  }
  return rows;
}

}  // namespace dlsim
