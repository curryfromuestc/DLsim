#include "engine/harness_rng.h"

#include <openssl/evp.h>

#include <vector>

namespace dlsim {
namespace {

constexpr uint32_t kInitA = 0x43b0d7e5, kMultA = 0x931e8875, kInitB = 0x8b51f9dd, kMultB = 0x58f38ded;
constexpr uint32_t kMixL = 0xca01f9dd, kMixR = 0x4973f715;

uint32_t hashmix(uint32_t v, uint32_t& hc, uint32_t mult) {
  v ^= hc;
  hc *= mult;
  v *= hc;
  v ^= v >> 16;
  return v;
}

uint32_t mix(uint32_t x, uint32_t y) {
  uint32_t r = kMixL * x - kMixR * y;
  r ^= r >> 16;
  return r;
}

// numpy.random.SeedSequence(entropy).generate_state(4, np.uint64), pool size 4.
void seed_sequence_state(uint64_t entropy, uint64_t out[4]) {
  std::vector<uint32_t> ent;
  if (entropy == 0) ent.push_back(0);
  for (uint64_t n = entropy; n > 0; n >>= 32) ent.push_back(uint32_t(n & 0xffffffffu));
  uint32_t hc = kInitA, pool[4];
  for (int i = 0; i < 4; ++i) pool[i] = hashmix(i < int(ent.size()) ? ent[i] : 0u, hc, kMultA);
  for (int s = 0; s < 4; ++s)
    for (int d = 0; d < 4; ++d)
      if (s != d) pool[d] = mix(pool[d], hashmix(pool[s], hc, kMultA));
  for (size_t s = 4; s < ent.size(); ++s)
    for (int d = 0; d < 4; ++d) pool[d] = mix(pool[d], hashmix(ent[s], hc, kMultA));
  hc = kInitB;
  uint32_t st[8];
  for (int i = 0; i < 8; ++i) st[i] = hashmix(pool[i % 4], hc, kMultB);
  for (int k = 0; k < 4; ++k) out[k] = uint64_t(st[2 * k]) | (uint64_t(st[2 * k + 1]) << 32);
}

using u128 = unsigned __int128;
constexpr u128 kPcgMult = (u128(2549297995355413924ULL) << 64) | 4865540595714422341ULL;

uint64_t rotr64(uint64_t v, unsigned r) { return (v >> r) | (v << ((-r) & 63u)); }

}  // namespace

double harness_uniform01(uint64_t base_seed, const std::string& trace_id, int lane) {
  const std::string s = std::to_string(base_seed) + ":" + trace_id + ":" + std::to_string(lane);
  unsigned char md[EVP_MAX_MD_SIZE];
  unsigned int len = 0;
  EVP_Digest(s.data(), s.size(), md, &len, EVP_sha256(), nullptr);
  uint64_t seed = 0;
  for (int i = 0; i < 8; ++i) seed = (seed << 8) | md[i];
  uint64_t w[4];
  seed_sequence_state(seed, w);
  const u128 init_state = (u128(w[0]) << 64) | w[1], init_seq = (u128(w[2]) << 64) | w[3];
  u128 state = 0;
  const u128 inc = (init_seq << 1) | 1;
  state = state * kPcgMult + inc;
  state += init_state;
  state = state * kPcgMult + inc;
  state = state * kPcgMult + inc;  // first output steps before extracting
  const uint64_t hi = uint64_t(state >> 64), lo = uint64_t(state);
  const uint64_t x = rotr64(hi ^ lo, unsigned(hi >> 58));
  return double(x >> 11) * 0x1.0p-53;
}

}  // namespace dlsim
