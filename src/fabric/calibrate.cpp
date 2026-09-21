#include "fabric/calibrate.h"

#include <yaml-cpp/yaml.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <tuple>

namespace dlsim {
namespace {

double payload(const CollectiveXRow& r, EpComponent c) {
  return c == EpComponent::Dispatch ? r.dispatch_payload_bytes : r.combine_payload_bytes;
}
double p50_us(const CollectiveXRow& r, EpComponent c) {
  return c == EpComponent::Dispatch ? r.dispatch_p50_us : r.combine_p50_us;
}
const char* name(EpComponent c) { return c == EpComponent::Dispatch ? "dispatch" : "combine"; }

double quantile(std::vector<double> v, double q) {
  if (v.empty()) return NAN;
  std::sort(v.begin(), v.end());
  const double pos = q * (v.size() - 1);
  const size_t i = size_t(pos);
  if (i + 1 >= v.size()) return v.back();
  return v[i] + (pos - i) * (v[i + 1] - v[i]);
}

std::string num(double v) {
  char buf[32];
  if (std::isnan(v)) return "null";
  std::snprintf(buf, sizeof buf, "%.6g", v);
  return buf;
}
std::string str(const std::string& s) { return "\"" + s + "\""; }

using GroupKey = std::tuple<std::string, std::string, std::string, std::string, std::string>;
GroupKey key_of(const CollectiveXRow& r) { return {r.sku, r.backend, r.mode, r.phase, r.precision}; }

}  // namespace

LinkParams fit_link(const std::vector<LinkPoint>& pts) {
  if (pts.empty()) throw std::runtime_error("fit_link: no points");
  const double n = double(pts.size());
  double sx = 0, sy = 0, sxx = 0, sxy = 0;
  for (const auto& p : pts) { sx += p.bytes; sy += p.latency_s; sxx += p.bytes * p.bytes; sxy += p.bytes * p.latency_s; }
  const double det = n * sxx - sx * sx;
  double a = sy / n, b = 0;
  if (det > 0) {
    b = (n * sxy - sx * sy) / det;
    a = (sy - b * sx) / n;
  }
  if (b < 0) { b = 0; a = sy / n; }
  if (a < 0) { a = 0; b = sxx > 0 ? sxy / sxx : 0; }
  LinkParams lp;
  lp.alpha_s = a;
  lp.bandwidth_Bps = b > 0 ? 1.0 / b : 0.0;
  return lp;
}

FabricSpec calibrate_scaleup(FabricSpec f, const std::vector<CollectiveXRow>& rows, EpComponent c) {
  std::vector<LinkPoint> pts;
  for (const auto& r : rows) {
    if (!r.single_domain() || r.ep < 2) continue;
    const double p = r.ep;
    if (f.alltoall_serial_latency) pts.push_back({payload(r, c) / (p * p), p50_us(r, c) * 1e-6 / (p - 1)});
    else pts.push_back({payload(r, c) / p * (p - 1) / p, p50_us(r, c) * 1e-6});
  }
  const LinkParams lp = fit_link(pts);
  f.scaleup.present = true;
  f.scaleup.alpha_s = lp.alpha_s;
  if (lp.bandwidth_Bps > 0) f.scaleup.bandwidth_Bps = lp.bandwidth_Bps;
  else f.scaleup.bandwidth_Bps.reset();
  return f;
}

FabricSpec scaleout_from_aisimulate(const std::string& path) {
  YAML::Node node = YAML::LoadFile(path)["node"];
  FabricSpec f;
  f.scaleout.present = true;
  f.scaleout.bandwidth_Bps = node["inter_node_bw"].as<double>();
  f.scaleout.alpha_s = node["p2p_latency"].as<double>();
  f.devices_per_node = node["num_gpus_per_node"].as<int>();
  return f;
}

std::string predict_ep16_report(const std::vector<CollectiveXRow>& rows, const std::map<std::string, FabricSpec>& scaleout_by_sku) {
  std::map<GroupKey, std::vector<const CollectiveXRow*>> groups;
  for (const auto& r : rows) groups[key_of(r)].push_back(&r);
  std::string out = "{\"groups\":[";
  bool first_group = true;
  for (const auto& [key, members] : groups) {
    std::vector<CollectiveXRow> ep8, ep16;
    for (const auto* r : members) {
      if (r->ep == 8) ep8.push_back(*r);
      else if (r->ep == 16) ep16.push_back(*r);
    }
    if (ep8.empty()) continue;
    const auto& [sku, backend, mode, phase, precision] = key;
    const auto so = scaleout_by_sku.find(sku);
    FabricSpec base = so == scaleout_by_sku.end() ? FabricSpec{} : so->second;
    if (!first_group) out += ",";
    first_group = false;
    out += "{\"sku\":" + str(sku) + ",\"backend\":" + str(backend) + ",\"mode\":" + str(mode) + ",\"phase\":" + str(phase) +
           ",\"precision\":" + str(precision) + ",\"ep8_points\":" + num(double(ep8.size())) + ",\"ep16_points\":" + num(double(ep16.size()));
    if (!ep16.empty()) {
      out += ",\"ep16_topology\":" + str(ep16.front().single_domain() ? "single-domain" : "cross-domain") +
             ",\"ep16_nodes\":" + num(ep16.front().nodes) + ",\"ep16_gpus_per_node\":" + num(ep16.front().gpus_per_node) +
             ",\"scaleout_configured\":" + (base.scaleout.present ? "true" : "false");
    }
    out += ",\"components\":{";
    for (EpComponent c : {EpComponent::Dispatch, EpComponent::Combine}) {
      const FabricSpec f = calibrate_scaleup(base, ep8, c);
      const LinkParams up = scaleup_params(f);
      if (c == EpComponent::Combine) out += ",";
      out += str(name(c)) + ":{\"alpha_up_s\":" + num(up.alpha_s) + ",\"L_up_Bps\":" + num(up.bandwidth_Bps);
      std::vector<double> ratios;
      std::string points = "[";
      bool first_pt = true;
      bool single_ok = true;
      for (const auto& r : ep16) {
        const double n = payload(r, c) / r.ep;
        const Group g{r.ep, r.per_domain()};
        if (g.spans_domains() && !f.scaleout.present) continue;
        const CommCost cost = collective_cost(f, Collective::AllToAll, g, n);
        const double pred_us = cost.ms * 1e3;
        const double meas_us = p50_us(r, c);
        const double ratio = pred_us / meas_us;
        ratios.push_back(ratio);
        if (!first_pt) points += ",";
        first_pt = false;
        points += "{\"run_id\":" + str(r.run_id) + ",\"tokens_per_rank\":" + num(r.tokens_per_rank) +
                  ",\"bytes_per_device\":" + num(n) + ",\"measured_us\":" + num(meas_us) + ",\"predicted_us\":" + num(pred_us) +
                  ",\"ratio\":" + num(ratio) + ",\"cross_domain_bytes\":" + num(cost.cross_domain_bytes) +
                  ",\"in_domain_bytes\":" + num(cost.in_domain_bytes);
        if (r.single_domain()) {
          const double p = r.ep;
          const double msgs = f.alltoall_serial_latency ? p - 1 : 1.0;
          const double single_us = (msgs * up.alpha_s + (up.bandwidth_Bps > 0 ? (p - 1) / p * n / up.bandwidth_Bps : 0.0)) * 1e6;
          const bool match = std::fabs(single_us - pred_us) <= 1e-9 * std::max(1.0, std::fabs(single_us));
          single_ok = single_ok && match;
          points += ",\"single_domain_us\":" + num(single_us) + ",\"single_domain_match\":" + (match ? "true" : "false");
        }
        points += "}";
      }
      points += "]";
      out += ",\"ratio\":{\"n\":" + num(double(ratios.size())) + ",\"median\":" + num(quantile(ratios, 0.5)) +
             ",\"p10\":" + num(quantile(ratios, 0.1)) + ",\"p90\":" + num(quantile(ratios, 0.9)) + "}";
      if (!ep16.empty() && ep16.front().single_domain()) out += std::string(",\"single_domain_equivalence\":") + (single_ok ? "true" : "false");
      out += ",\"points\":" + points + "}";
    }
    out += "}}";
  }
  out += "]}";
  return out;
}

}  // namespace dlsim
