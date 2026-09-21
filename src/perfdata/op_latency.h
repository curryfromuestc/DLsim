#pragma once
// OpLatencySource: device + operator class + dtype + shape -> latency with provenance.
// Design: operator-latency.md.
#include <map>
#include <memory>
#include <string>

#include "core/types.h"

namespace dlsim {

struct OpQuery {
  std::string op;                            // table name: gemm, moe, dsv4_csa_context_module, dsv4_csa_generation_module,
                                             // dsv4_hca_context_module, dsv4_hca_generation_module, mhc_module, nccl,
                                             // custom_allreduce, moe_a2a, quantize, embedding ...
  std::map<std::string, double> num;         // numeric keys: m n k batch_size isl step num_heads tp_size compress_ratio
                                             // num_tokens hidden_size inter_size topk num_experts moe_tp_size moe_ep_size
                                             // message_size num_gpus ...
  std::map<std::string, std::string> cat;    // categorical keys: gemm_dtype moe_dtype kv_cache_dtype gemm_type distribution
                                             // op_name comm_backend phase framework version kernel_source ...
};

class OpLatencySource {
 public:
  virtual ~OpLatencySource() = default;
  virtual Latency query(const std::string& device, const OpQuery& q) const = 0;
};

}  // namespace dlsim
