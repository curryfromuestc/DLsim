#pragma once
// Closed-form roofline (design: operator-latency.md 实现三): gemm, moe, dense context / generation attention.
#include <map>
#include <string>

#include "config/config.h"
#include "perfdata/op_latency.h"

namespace dlsim {

class RooflineSource : public OpLatencySource {
 public:
  explicit RooflineSource(std::map<std::string, DeviceSpec> specs) : specs_(std::move(specs)) {}
  Latency query(const std::string& device, const OpQuery& q) const override;

 private:
  std::map<std::string, DeviceSpec> specs_;
};

}  // namespace dlsim
