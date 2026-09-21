#pragma once
// CollectiveX run reader (InferenceX EP dispatch/combine measurements). Design: data-sources.md "CollectiveX".
#include <string>
#include <vector>

namespace dlsim {

struct CollectiveXRow {
  std::string run_id, series_id;
  std::string sku, backend, mode, phase, precision;
  int ep = 0, nodes = 0, gpus_per_node = 0, scale_up_domain = 0;
  std::string scale_up_transport, scale_out_transport;
  int tokens_per_rank = 0;
  double dispatch_payload_bytes = 0, combine_payload_bytes = 0;   // summed over all ranks
  double dispatch_p50_us = 0, combine_p50_us = 0;
  bool single_domain() const { return scale_up_domain >= ep; }
  int per_domain() const { return single_domain() ? ep : gpus_per_node; }
};

std::vector<CollectiveXRow> load_collectivex_run(const std::string& path);
std::vector<CollectiveXRow> load_collectivex_dir(const std::string& dir);   // every run_*.json in dir

}  // namespace dlsim
