#pragma once
// Link calibration against CollectiveX and the EP8 -> EP16 domain-overflow check. Design: validation.md step 8.
#include <map>
#include <string>
#include <vector>

#include "config/config.h"
#include "fabric/collectivex.h"
#include "fabric/fabric.h"

namespace dlsim {

struct LinkPoint {
  double bytes = 0;
  double latency_s = 0;
};

// Least squares of latency = alpha + bytes / L with alpha >= 0 and 1/L >= 0 (1/L == 0 -> bandwidth_Bps == 0).
LinkParams fit_link(const std::vector<LinkPoint>& points);

enum class EpComponent { Dispatch, Combine };

// Inverts the single-domain all-to-all closed form (alpha_up, or (p-1) alpha_up when alltoall_serial_latency,
// plus (p-1)/p n / L_up with n = payload / p) on EP8 rows and stores alpha_up, L_up into f.scaleup.
FabricSpec calibrate_scaleup(FabricSpec f, const std::vector<CollectiveXRow>& ep8_rows, EpComponent c);

// Scale-out link taken from an AISimulate system yaml: node.inter_node_bw, node.p2p_latency.
FabricSpec scaleout_from_aisimulate(const std::string& yaml_path);

// Per (sku, backend, mode, phase, precision): calibrate scaleup on EP8 rows, predict EP16 dispatch and combine
// through collective_cost, and report predicted / measured ratios. JSON text.
std::string predict_ep16_report(const std::vector<CollectiveXRow>& rows, const std::map<std::string, FabricSpec>& scaleout_by_sku);

}  // namespace dlsim
