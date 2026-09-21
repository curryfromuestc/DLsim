#pragma once
#include <string>

namespace dlsim {

// Table dtype label -> DeviceSpec::flops key; empty when the label has no tensor-core mapping.
inline std::string flops_key_for_dtype(const std::string& d) {
  if (d == "bfloat16" || d == "float16" || d == "int8_wo" || d == "int4_wo" || d.rfind("w4a16", 0) == 0) return "bf16";
  if (d == "fp8" || d == "fp8_block" || d == "fp8_e4m3" || d == "fp8_static" || d == "sq" || d.rfind("w4a8", 0) == 0) return "fp8";
  if (d == "nvfp4" || d == "fp4" || d == "w4a4") return "fp4";
  if (d == "int8") return "int8";
  return "";
}

// Bytes per weight element as used by the SOL memory term.
inline double weight_bytes_for_dtype(const std::string& d) {
  if (d == "nvfp4" || d == "fp4" || d == "w4a4" || d.rfind("w4a16", 0) == 0) return 0.5625;
  if (d == "int4_wo" || d.rfind("w4a8", 0) == 0) return 0.5;
  if (d == "fp8" || d == "fp8_block" || d == "fp8_e4m3" || d == "fp8_static" || d == "sq" || d == "int8_wo" || d == "int8") return 1;
  return 2;
}

// The categorical key that names an operator's dtype, in lookup order.
inline const char* const kDtypeKeys[] = {"gemm_dtype", "moe_dtype", "gemm_type", "attn_dtype", "mla_dtype", "bmm_dtype",
                                          "quant_dtype", "allreduce_dtype", "nccl_dtype", "comm_dtype"};

}  // namespace dlsim
