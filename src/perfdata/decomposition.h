#pragma once
// Cross-device scaling decomposition t_d = a / C_d + b / Bw_d + c (design: operator-latency.md 实现二).
#include <map>
#include <string>
#include <vector>

#include "config/config.h"
#include "perfdata/measured.h"
#include "perfdata/roofline.h"

namespace dlsim {

constexpr double kConditionLimit = 30.0;

struct DecompositionFit {
  enum class Rank { Full, BwDeficient, CDeficient, Underdetermined };
  Rank rank = Rank::Underdetermined;
  double a = 0, b = 0, c = 0;   // Full
  double k = 0;                 // BwDeficient: t = a / C + k;  CDeficient: t = b / Bw + k
  double cond = 0;              // condition number of the column-scaled full design matrix
  double residual_rel = 0;      // max |fit - t| / t over the references
  double loo_rel = 0;           // max leave-one-reference-out relative error; 0 when fewer than 4 references
  std::vector<std::string> devices;
};

const char* to_string(DecompositionFit::Rank r);

// Non-negative least squares by active-set enumeration over the three columns.
DecompositionFit fit_decomposition(const std::vector<std::string>& devices, const std::vector<double>& C,
                                   const std::vector<double>& Bw, const std::vector<double>& t);

// Latency of a target device from a fit; reference C / Bw / t are needed for the interval rules.
Latency decomposed_latency(const DecompositionFit& fit, double C_t, double Bw_t, const std::vector<double>& C,
                           const std::vector<double>& Bw, const std::vector<double>& t, const Latency& roofline_lo);

class DecomposedSource : public OpLatencySource {
 public:
  DecomposedSource(const MeasuredTables& tables, std::vector<std::string> reference_devices, std::map<std::string, DeviceSpec> specs);
  Latency query(const std::string& device, const OpQuery& q) const override;
  Latency query_with_references(const std::string& device, const OpQuery& q, const std::vector<std::string>& refs) const;

  struct HoldOut {
    std::string device;
    Latency measured;
    Latency predicted;
  };
  // For each reference device: its own measured value and the prediction from the other references.
  std::vector<HoldOut> leave_one_device_out(const OpQuery& q) const;
  const std::vector<std::string>& references() const { return refs_; }

 private:
  const MeasuredTables& tables_;
  std::vector<std::string> refs_;
  std::map<std::string, DeviceSpec> specs_;
  RooflineSource roofline_;
};

}  // namespace dlsim
