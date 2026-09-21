#include <cassert>
#include <cmath>
#include <cstdio>

#include "engine/harness_rng.h"

// Reference values from: hashlib.sha256(f"{seed}:{id}:{lane}").digest()[:8] big-endian,
// np.random.default_rng(seed).uniform(0.0, 1.0)   (numpy 1.24.3)
int main() {
  struct Case {
    uint64_t seed;
    const char* id;
    int lane;
    double u;
  };
  const Case cases[] = {
      {42, "002001296e8a8c38ad9d7cc436d691afc602", 0, 0.564252023124218},
      {42, "0196085d85d2075a50b74cd8795ffbdcea9a", 3, 0.629704416692281},
      {42, "ffc9ee68609fce322f066be4f7daf3c62eff", 392, 0.8078173042357791},
      {0, "res", 0, 0.759309461837102},
      {7, "x", 1, 0.7413778696082045},
  };
  for (const Case& c : cases) {
    const double u = dlsim::harness_uniform01(c.seed, c.id, c.lane);
    if (u != c.u) {
      std::printf("mismatch %llu:%s:%d got %.17g want %.17g\n", (unsigned long long)c.seed, c.id, c.lane, u, c.u);
      return 1;
    }
  }
  std::puts("harness_rng_test ok");
  return 0;
}
