#include "fabric/fabric.h"

#include <cmath>
#include <stdexcept>

namespace dlsim {
namespace {

LinkParams params_of(const Link& l, double oversub) {
  LinkParams p;
  p.alpha_s = l.alpha_s.value_or(0.0);
  p.bandwidth_Bps = l.bandwidth_Bps ? *l.bandwidth_Bps / oversub : 0.0;
  return p;
}

double bytes_ms(double bytes, const LinkParams& p) { return p.bandwidth_Bps > 0 ? bytes / p.bandwidth_Bps * 1e3 : 0.0; }

// Closed-form terms for one collective inside a single domain of size p.
struct Terms { double byte_factor; double msgs; };

Terms terms(Collective c, int p) {
  const double pm1 = p - 1;
  switch (c) {
    case Collective::AllReduce: return {2.0 * pm1 / p, 2.0 * pm1};
    case Collective::AllGather:
    case Collective::ReduceScatter: return {pm1 / p, pm1};
    case Collective::AllToAll: return {pm1 / p, pm1};
    case Collective::P2P: return {1.0, 1.0};
  }
  return {0, 0};
}

}  // namespace

LinkParams scaleup_params(const FabricSpec& f) { return params_of(f.scaleup, 1.0); }
LinkParams scaleout_params(const FabricSpec& f) { return params_of(f.scaleout, f.scaleout_oversub); }
LinkParams host_params(const FabricSpec& f) { return params_of(f.host, 1.0); }
LinkParams remote_params(const FabricSpec& f) { return params_of(f.remote, 1.0); }

CommCost collective_cost(const FabricSpec& f, Collective c, Group g, double bytes_per_device, int rounds) {
  if (g.size <= 1) return {};
  CommCost r;
  const LinkParams up = scaleup_params(f);
  const LinkParams out = scaleout_params(f);
  r.unlimited = !f.scaleup.present && !f.scaleout.present;
  const int p = g.size;
  const bool constant_a2a = c == Collective::AllToAll && !f.alltoall_serial_latency;
  if (!g.spans_domains()) {
    const Terms t = terms(c, p);
    r.in_domain_bytes = t.byte_factor * bytes_per_device * rounds;
    r.in_domain_msgs = t.msgs * rounds;
    const double latency_ms = (constant_a2a ? rounds : r.in_domain_msgs) * up.alpha_s * 1e3;
    r.ms = (latency_ms + bytes_ms(r.in_domain_bytes, up)) * f.tail_latency_mult;
    return r;
  }
  if (!f.scaleout.present) throw std::runtime_error("parallel group spans scale-up domains but no scale-out layer is configured");
  r.cross_domain = true;
  const int gsz = g.per_domain;
  const int k = (p + gsz - 1) / gsz;
  if (c == Collective::AllToAll || c == Collective::P2P) {
    const double cross_frac = c == Collective::P2P ? 1.0 : double(p - gsz) / p;
    const double in_frac = c == Collective::P2P ? 0.0 : double(gsz - 1) / p;
    const double cross_dests = c == Collective::P2P ? 1.0 : double(p - gsz);
    const double in_dests = c == Collective::P2P ? 0.0 : double(gsz - 1);
    r.cross_domain_bytes = cross_frac * bytes_per_device * rounds;
    r.in_domain_bytes = in_frac * bytes_per_device * rounds;
    r.cross_domain_msgs = cross_dests * rounds;
    r.in_domain_msgs = in_dests * rounds;
    const double latency_ms = constant_a2a ? rounds * std::max(out.alpha_s, up.alpha_s) * 1e3
                                           : r.cross_domain_msgs * out.alpha_s * 1e3 + r.in_domain_msgs * up.alpha_s * 1e3;
    r.ms = (latency_ms + std::max(bytes_ms(r.in_domain_bytes, up), bytes_ms(r.cross_domain_bytes, out))) *
           f.tail_latency_mult;
    return r;
  }
  // Hierarchical: in-domain reduce-scatter, cross-domain all-reduce over k domain leaders, in-domain all-gather.
  const Terms rs = terms(Collective::ReduceScatter, gsz);
  const Terms ar = terms(Collective::AllReduce, k);
  const double shard = bytes_per_device / gsz;
  double ms = 0;
  if (c == Collective::AllReduce || c == Collective::ReduceScatter) {
    r.in_domain_bytes += rs.byte_factor * bytes_per_device * rounds;
    r.in_domain_msgs += rs.msgs * rounds;
    ms += rs.msgs * rounds * up.alpha_s * 1e3 + bytes_ms(rs.byte_factor * bytes_per_device * rounds, up);
  }
  r.cross_domain_bytes = ar.byte_factor * shard * rounds * (c == Collective::AllReduce ? 1.0 : 0.5);
  r.cross_domain_msgs = ar.msgs * rounds;
  ms += r.cross_domain_msgs * out.alpha_s * 1e3 + bytes_ms(r.cross_domain_bytes, out);
  if (c == Collective::AllReduce || c == Collective::AllGather) {
    r.in_domain_bytes += rs.byte_factor * bytes_per_device * rounds;
    r.in_domain_msgs += rs.msgs * rounds;
    ms += rs.msgs * rounds * up.alpha_s * 1e3 + bytes_ms(rs.byte_factor * bytes_per_device * rounds, up);
  }
  r.ms = ms * f.tail_latency_mult;
  return r;
}

double bulk_transfer_ms(const FabricSpec& f, const std::string& link, double bytes) {
  const Link* l = nullptr;
  double oversub = 1.0;
  if (link == "scaleup") l = &f.scaleup;
  else if (link == "scaleout") { l = &f.scaleout; oversub = f.scaleout_oversub; }
  else if (link == "host") l = &f.host;
  else if (link == "remote") l = &f.remote;
  else throw std::runtime_error("unknown link '" + link + "'");
  if (!l->present) throw std::runtime_error("bulk transfer over absent link '" + link + "'");
  const LinkParams p = params_of(*l, oversub);
  return p.alpha_s * 1e3 + bytes_ms(bytes, p);
}

std::vector<std::string> unlimited_resources(const FabricSpec& f) {
  std::vector<std::string> r;
  auto check = [&](const char* name, const Link& l) {
    if (!l.present) { r.push_back(std::string(name) + ":absent"); return; }
    if (!l.bandwidth_Bps) r.push_back(std::string(name) + ":bandwidth");
    if (!l.alpha_s) r.push_back(std::string(name) + ":alpha");
  };
  check("scaleup", f.scaleup);
  check("scaleout", f.scaleout);
  check("host", f.host);
  check("remote", f.remote);
  return r;
}

}  // namespace dlsim
