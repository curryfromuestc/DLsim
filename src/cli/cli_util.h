#pragma once
#include <map>
#include <string>
#include <vector>

#include "config/config.h"
#include "engine/engine.h"
#include "metrics/metrics.h"

namespace dlsim::cli {

struct Args {
  std::string command;
  std::vector<std::string> positional;
  std::map<std::string, std::string> options;   // --key value ; --flag -> "1"
  std::string get(const std::string& k, const std::string& def = "") const {
    auto it = options.find(k);
    return it == options.end() ? def : it->second;
  }
  bool has(const std::string& k) const { return options.count(k) > 0; }
};
Args parse_args(int argc, char** argv);

// A benchmark point: configs/points/*.yaml plus the four configs it references.
struct BenchPoint {
  int id = 0;
  std::string hardware, framework, precision, spec_method, offload_mode, path;
  bool is_multinode = false;
  int num_prefill_gpu = 0, num_decode_gpu = 0;
  DeviceSet devices;
  FabricSpec fabric;
  MappingSpec mapping;
  StackSpec stack;
  RunSpec run;
};
BenchPoint load_point(const std::string& path, const std::map<std::string, std::string>& stack_overrides);
void apply_stack_override(StackSpec& s, const std::string& key, const std::string& value);
RowMeta row_meta(const BenchPoint& p);

// Every configs/device/*.yaml keyed by DeviceSpec::name (for GEMM site transfer in MeasuredTables).
std::map<std::string, DeviceSpec> load_device_specs(const std::string& dir);

// Snapshot rows keyed by id: metric name -> value (only the fields needed for comparison).
struct SnapshotRow { std::map<std::string, double> metrics; std::string hardware, framework; };
std::map<int, SnapshotRow> load_snapshot(const std::string& benchmarks_json, const std::string& derived_json);

std::string json_escape(const std::string& s);
void write_file(const std::string& path, const std::string& content);

}  // namespace dlsim::cli
