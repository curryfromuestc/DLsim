#undef NDEBUG
#include <cassert>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <map>
#include <string>

#include "fabric/calibrate.h"
#include "fabric/collectivex.h"
#include "fabric/fabric.h"

using namespace dlsim;

static bool near(double a, double b, double rel = 1e-9) { return std::fabs(a - b) <= rel * std::max(1.0, std::fabs(b)); }

static FabricSpec two_tier() {
  FabricSpec f;
  f.scaleup.present = true;
  f.scaleup.alpha_s = 2e-6;
  f.scaleup.bandwidth_Bps = 400e9;
  f.scaleout.present = true;
  f.scaleout.alpha_s = 10e-6;
  f.scaleout.bandwidth_Bps = 100e9;
  f.scaleout_oversub = 2.0;
  return f;
}

int main() {
  const double n = 16e6;
  {
    FabricSpec f = two_tier();
    const CommCost a = collective_cost(f, Collective::AllToAll, Group{8, 8}, n);
    const CommCost b = collective_cost(f, Collective::AllToAll, Group{8, 72}, n);
    assert(!a.cross_domain && !b.cross_domain);
    assert(near(a.ms, b.ms));
    assert(near(a.ms, (1 * 2e-6 + 7.0 / 8 * n / 400e9) * 1e3));
    FabricSpec serial = two_tier();
    serial.alltoall_serial_latency = true;
    assert(near(collective_cost(serial, Collective::AllToAll, Group{8, 8}, n).ms, (7 * 2e-6 + 7.0 / 8 * n / 400e9) * 1e3));
    const CommCost ar = collective_cost(f, Collective::AllReduce, Group{8, 8}, n);
    assert(near(ar.ms, (14 * 2e-6 + 2 * 7.0 / 8 * n / 400e9) * 1e3));
  }
  {
    FabricSpec f;
    f.scaleup.present = true;
    f.scaleout.present = true;
    for (Collective c : {Collective::AllReduce, Collective::AllGather, Collective::ReduceScatter, Collective::AllToAll, Collective::P2P}) {
      assert(collective_cost(f, c, Group{16, 8}, n).ms == 0.0);
      assert(collective_cost(f, c, Group{16, 16}, n).ms == 0.0);
    }
    FabricSpec absent;
    const CommCost c = collective_cost(absent, Collective::AllToAll, Group{8, 8}, n);
    assert(c.ms == 0.0 && c.unlimited);
  }
  {
    FabricSpec f = two_tier();
    const CommCost c = collective_cost(f, Collective::AllToAll, Group{16, 8}, n);
    assert(c.cross_domain);
    assert(near(c.cross_domain_bytes / n, 8.0 / 16));
    assert(near(c.in_domain_bytes / n, 7.0 / 16));
    assert(near(c.cross_domain_msgs, 8) && near(c.in_domain_msgs, 7));
    const double expect = 10e-6 + std::max(7.0 / 16 * n / 400e9, 8.0 / 16 * n / 50e9);
    assert(near(c.ms, expect * 1e3));
    FabricSpec serial = two_tier();
    serial.alltoall_serial_latency = true;
    const double expect_serial = 8 * 10e-6 + 7 * 2e-6 + std::max(7.0 / 16 * n / 400e9, 8.0 / 16 * n / 50e9);
    assert(near(collective_cost(serial, Collective::AllToAll, Group{16, 8}, n).ms, expect_serial * 1e3));
    const CommCost k4 = collective_cost(f, Collective::AllToAll, Group{16, 4}, n);
    assert(near(k4.cross_domain_bytes / n, 12.0 / 16) && near(k4.cross_domain_msgs, 12));
  }
  {
    FabricSpec f = two_tier();
    const CommCost c = collective_cost(f, Collective::AllReduce, Group{16, 8}, n);
    const double rs = 7 * 2e-6 + 7.0 / 8 * n / 400e9;
    const double ar = 2 * 10e-6 + 2 * 1.0 / 2 * (n / 8) / 50e9;
    const double ag = rs;
    assert(near(c.ms, (rs + ar + ag) * 1e3));
    assert(near(c.in_domain_msgs, 14) && near(c.cross_domain_msgs, 2));
  }
  {
    std::vector<LinkPoint> pts;
    for (double b : {1e6, 2e6, 4e6, 8e6, 16e6}) pts.push_back({b, 5e-6 + b / 4e11});
    const LinkParams lp = fit_link(pts);
    assert(near(lp.alpha_s, 5e-6, 1e-6) && near(lp.bandwidth_Bps, 4e11, 1e-6));
    std::vector<LinkPoint> flat = {{1e6, 3e-6}, {2e6, 2e-6}, {4e6, 1e-6}};
    const LinkParams fl = fit_link(flat);
    assert(fl.bandwidth_Bps == 0.0 && near(fl.alpha_s, 2e-6));
    std::vector<CollectiveXRow> ep8;
    for (double b : {1e6, 2e6, 4e6, 8e6}) {
      CollectiveXRow r;
      r.ep = 8; r.scale_up_domain = 8; r.gpus_per_node = 8;
      r.dispatch_payload_bytes = b * 8;
      r.dispatch_p50_us = (3e-6 + 7.0 / 8 * b / 2e11) * 1e6;
      ep8.push_back(r);
    }
    const FabricSpec cal = calibrate_scaleup(FabricSpec{}, ep8, EpComponent::Dispatch);
    const LinkParams up = scaleup_params(cal);
    assert(near(up.alpha_s, 3e-6, 1e-6) && near(up.bandwidth_Bps, 2e11, 1e-6));
  }
  const std::string traces = std::string(DLSIM_SOURCE_DIR) + "/traces/inferencex/collectivex_2026-09-20";
  const std::string systems = std::string(DLSIM_SOURCE_DIR) + "/third_party/aisimulate/python/aisimulate/src/aisimulate_core/systems/";
  {
    const auto rows = load_collectivex_run(traces + "/run_33477867072.json");
    assert(rows.size() == 162);
    assert(rows.front().sku == "b200-nscale" && rows.front().ep == 8 && rows.front().tokens_per_rank == 1);
    assert(rows.front().dispatch_payload_bytes == 688128 && rows.front().scale_out_transport.empty());
    int ep16 = 0;
    for (const auto& r : rows) ep16 += r.ep == 16;
    assert(ep16 == 54);
  }
  {
    const auto rows = load_collectivex_dir(traces);
    assert(rows.size() == 1321);
    std::map<std::string, FabricSpec> so;
    for (const auto& [sku, yaml] : std::map<std::string, std::string>{{"b200-nscale", "b200_sxm"}, {"b300", "b300_sxm"}, {"h100-dgxc", "h100_sxm"},
                                                                       {"h200-dgxc", "h200_sxm"}, {"gb200", "gb200"}, {"gb300", "gb300"}})
      so[sku] = scaleout_from_aisimulate(systems + yaml + ".yaml");
    assert(near(*so["b200-nscale"].scaleout.bandwidth_Bps, 50e9) && near(*so["b200-nscale"].scaleout.alpha_s, 10e-6));
    const std::string report = predict_ep16_report(rows, so);
    assert(report.find("\"single_domain_match\":false") == std::string::npos);
    assert(report.find("\"single_domain_equivalence\":true") != std::string::npos);
    std::ofstream(std::string(DLSIM_SOURCE_DIR) + "/build/collectivex_ep16_report.json") << report;
    std::printf("report bytes %zu\n", report.size());
  }
  std::puts("fabric_test ok");
  return 0;
}
