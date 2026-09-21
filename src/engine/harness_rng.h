#pragma once

#include <cstdint>
#include <string>

namespace dlsim {

// Reproduces the AgentX harness draw for a lane's snapshot instant:
//   seed = int.from_bytes(sha256(f"{base_seed}:{trace_id}:{lane}")[:8], "big")
//   np.random.default_rng(seed).uniform(0, 1)
// (numpy SeedSequence -> PCG64 XSL-RR, first double). t* = lo + u * (hi - lo).
double harness_uniform01(uint64_t base_seed, const std::string& trace_id, int lane);

}  // namespace dlsim
