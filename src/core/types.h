#pragma once
#include <cstdint>
#include <string>

namespace dlsim {

// Provenance of a latency value, ordered from strongest to weakest evidence.
enum class Source : uint8_t {
  Measured,      // exact hit in a measured table
  Interpolated,  // inside the measured range
  Extrapolated,  // outside the measured range along a known cost axis
  Decomposed,    // cross-device scaling decomposition
  Roofline,      // closed form
  Bound,         // only lo_ms / hi_ms are meaningful
  Unsupported    // no basis for a number
};

const char* to_string(Source s);
inline Source weaker(Source a, Source b) { return a > b ? a : b; }

struct Latency {
  double ms = 0;               // point estimate; meaningless for Bound / Unsupported
  double lo_ms = 0;            // interval; equals ms when the source is exact
  double hi_ms = 0;
  Source source = Source::Unsupported;
  std::string note;            // table, framework, version, extrapolation ratio, borrowed terms

  bool has_point() const { return source < Source::Bound; }
  static Latency exact(double v, Source s, std::string n = {}) { return {v, v, v, s, std::move(n)}; }
  static Latency unsupported(std::string n) { return {0, 0, 0, Source::Unsupported, std::move(n)}; }
};

// Sum of two latencies with interval and provenance propagation.
Latency operator+(const Latency& a, const Latency& b);
Latency operator*(const Latency& a, double k);

}  // namespace dlsim
