#include "perfdata/roofline.h"

#include <algorithm>
#include <cmath>
#include <sstream>

#include "perfdata/dtype.h"

namespace dlsim {
namespace {

struct Ctx {
  const OpQuery& q;
  std::string missing;
  double num(const char* k, double def = std::nan("")) {
    const auto it = q.num.find(k);
    if (it != q.num.end()) return it->second;
    if (std::isnan(def)) missing += std::string(missing.empty() ? "" : ", ") + k;
    return def;
  }
  std::string cat(const char* k, const std::string& def = {}) {
    const auto it = q.cat.find(k);
    if (it != q.cat.end()) return it->second;
    if (def.empty()) missing += std::string(missing.empty() ? "" : ", ") + k;
    return def;
  }
};

}  // namespace

Latency RooflineSource::query(const std::string& device, const OpQuery& q) const {
  const auto sit = specs_.find(device);
  if (sit == specs_.end()) return Latency::unsupported("roofline: no device spec for " + device);
  const DeviceSpec& d = sit->second;
  if (d.memory.empty() || d.memory[0].bandwidth_Bps <= 0) return Latency::unsupported("roofline: device " + device + " has no memory bandwidth");
  const double bw = d.memory[0].bandwidth_Bps;
  Ctx c{q, {}};
  double flop = 0, bytes = 0;
  std::string dtype;
  const std::string& op = q.op;
  if (op == "gemm") {
    const double m = c.num("m"), n = c.num("n"), k = c.num("k");
    dtype = c.cat("gemm_dtype");
    flop = 2 * m * n * k;
    bytes = weight_bytes_for_dtype(dtype) * (m * n + m * k + n * k);
  } else if (op == "moe") {
    const double T = c.num("num_tokens"), K = c.num("topk"), h = c.num("hidden_size"), inter = c.num("inter_size"),
                 E = c.num("num_experts"), ep = c.num("moe_ep_size", 1), tp = c.num("moe_tp_size", 1);
    dtype = c.cat("moe_dtype");
    flop = T * K * h * inter * 3 * 2 / (ep * tp);
    const double hit = std::min(E / ep, T * K);
    bytes = hit * 3 * h * inter / tp * weight_bytes_for_dtype(dtype) + T * h * 2 * 2;
  } else if (op.find("attention") != std::string::npos && op.find("context") != std::string::npos) {
    const double b = c.num("batch_size"), s = c.num("isl"), p = c.num("step", 0), n = c.num("num_heads"), h = c.num("head_dim");
    double nkv = c.num("num_key_value_heads", -1);
    if (nkv < 0) nkv = c.num("num_kv_heads", n);
    if (nkv == 0) nkv = n;
    dtype = c.cat("attn_dtype", "bfloat16");
    const double kvb = weight_bytes_for_dtype(c.cat("kv_cache_dtype", "bfloat16"));
    const double S = s + p;
    flop = 2 * b * (S * S - p * p) * n * h;
    bytes = b * s * n * h * 2 * 2 + b * S * 2 * nkv * h * kvb + b * s * 2 * nkv * h * kvb;
  } else if (op.find("attention") != std::string::npos && op.find("generation") != std::string::npos) {
    const double b = c.num("batch_size"), n = c.num("num_heads"), h = c.num("head_dim"), kv = c.num("isl", 1) + c.num("step");
    double nkv = c.num("num_key_value_heads", -1);
    if (nkv < 0) nkv = c.num("num_kv_heads", n);
    if (nkv == 0) nkv = n;
    dtype = c.cat("attn_dtype", "bfloat16");
    const double kvb = weight_bytes_for_dtype(c.cat("kv_cache_dtype", "bfloat16"));
    flop = 2 * b * n * h * 2 * kv;
    bytes = b * 2 * nkv * kv * h * kvb;
  } else {
    return Latency::unsupported("roofline: no closed form for " + op);
  }
  if (!c.missing.empty()) return Latency::unsupported("roofline " + op + ": missing keys " + c.missing);
  const std::string key = flops_key_for_dtype(dtype);
  const auto fit = d.flops.find(key);
  if (fit == d.flops.end() || fit->second <= 0) return Latency::unsupported("roofline " + op + ": device " + device + " has no flops for dtype " + dtype);
  const double t_math = flop / fit->second * 1000, t_mem = bytes / bw * 1000;
  std::ostringstream note;
  note << "roofline " << op << " on " << device << ": flop=" << flop << " bytes=" << bytes << " t_math=" << t_math << " t_mem=" << t_mem;
  return Latency::exact(std::max(t_math, t_mem), Source::Roofline, note.str());
}

}  // namespace dlsim
