#pragma once
// Interconnect cost model. Design: fabric-and-heterogeneity.md.
#include <string>

#include "config/config.h"
#include "core/types.h"

namespace dlsim {

enum class Collective { AllReduce, AllGather, ReduceScatter, AllToAll, P2P };

// Placement of one parallel group: size devices, per_domain of them share a scale-up domain.
struct Group {
  int size = 1;
  int per_domain = 1;          // g; domains k = ceil(size / per_domain)
  bool spans_domains() const { return per_domain < size; }
};

// Effective per-link parameters after calibration against measured tables (nullopt = take FabricSpec).
struct LinkParams {
  double alpha_s = 0;
  double bandwidth_Bps = 0;    // 0 = unlimited
};

struct CommCost {
  double ms = 0;
  double in_domain_bytes = 0, cross_domain_bytes = 0;
  double in_domain_msgs = 0, cross_domain_msgs = 0;
  bool cross_domain = false;
  bool unlimited = false;      // no link term applied (fabric fields omitted)
};

LinkParams scaleup_params(const FabricSpec& f);
LinkParams scaleout_params(const FabricSpec& f);   // bandwidth already divided by oversubscription
LinkParams host_params(const FabricSpec& f);
LinkParams remote_params(const FabricSpec& f);

// Closed-form collective cost. bytes_per_device is the message size n of the design tables.
CommCost collective_cost(const FabricSpec& f, Collective c, Group g, double bytes_per_device, int rounds = 1);

// Bulk transfer over one link kind ("scaleup" "scaleout" "host" "remote"): alpha + bytes / L.
double bulk_transfer_ms(const FabricSpec& f, const std::string& link, double bytes);

// Which fabric resources are unlimited in this configuration (for the run output).
std::vector<std::string> unlimited_resources(const FabricSpec& f);

}  // namespace dlsim
