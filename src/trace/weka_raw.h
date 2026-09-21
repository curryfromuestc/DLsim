#pragma once
#include <cmath>
#include <cstdint>
#include <string>
#include <vector>

#include "trace/trace.h"

namespace dlsim::trace::detail {

struct RawRequest {
  int32_t outer_idx = -1;
  double t = 0;
  double api_time = NAN;
  int64_t in_tokens = 0;
  int64_t out_tokens = 0;
  std::vector<uint64_t> hash_ids;
  uint16_t model = 0;
  bool streaming = false;
};

struct RawSubagent {
  int32_t outer_idx = -1;
  double t = 0;
  double duration_ms = NAN;
  std::string agent_id;
  std::string subagent_type;
  std::vector<RawRequest> requests;
};

struct RawTrace {
  std::string id;
  int block_size = 64;
  std::vector<std::string> models;
  std::vector<RawRequest> normals;
  std::vector<RawSubagent> subagents;
};

Trace build_trace(RawTrace&& raw);

}  // namespace dlsim::trace::detail
