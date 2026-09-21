#pragma once
#include <string>

#include "trace/trace.h"

namespace dlsim::trace {

// Canonical JSON of one trace's session graph for comparison with the agentx-harness loader:
// agents sorted by harness session id, each with its request sequence [t, in, out, hash_count, model],
// and edges [from_sid, from_k, to_sid, to_k, kind, delay_s] sorted lexicographically.
std::string dump_graph_json(const Trace& t);

}  // namespace dlsim::trace
