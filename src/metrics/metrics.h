#pragma once
// InferenceX row output and the two interactivity metrics. Design: workload-and-metrics.md.
#include <map>
#include <string>
#include <vector>

#include "config/config.h"
#include "engine/engine.h"

namespace dlsim {

struct RowMeta {
  std::string hardware, framework, model = "DeepSeek-V4-Pro", precision = "fp4", spec_method = "none";
  std::string offload_mode = "none";
  bool disagg = false, is_multinode = false;
};

struct Metrics {
  std::map<std::string, double> values;    // tput_per_gpu, p90_intvty, p90_e2e_norm_intvty, median_ttft, ...
};

Metrics compute_metrics(const SimResult& r, const MappingSpec& m);
double percentile(std::vector<double> v, double q);    // q in [0,100], nearest-rank on sorted copy

// One InferenceX benchmarks row plus DLsim extra fields, as JSON text.
std::string to_inferencex_row_json(const SimResult& r, const Metrics& mt, const RowMeta& meta, const MappingSpec& m,
                                   int concurrency);

}  // namespace dlsim
