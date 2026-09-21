#include "core/types.h"

namespace dlsim {

const char* to_string(Source s) {
  switch (s) {
    case Source::Measured: return "measured";
    case Source::Interpolated: return "interpolated";
    case Source::Extrapolated: return "extrapolated";
    case Source::Decomposed: return "decomposed";
    case Source::Roofline: return "roofline";
    case Source::Bound: return "bound";
    case Source::Unsupported: return "unsupported";
  }
  return "?";
}

Latency operator+(const Latency& a, const Latency& b) {
  Latency r;
  r.ms = a.ms + b.ms;
  r.lo_ms = a.lo_ms + b.lo_ms;
  r.hi_ms = a.hi_ms + b.hi_ms;
  r.source = weaker(a.source, b.source);
  if (r.source == Source::Unsupported) r.note = a.source == Source::Unsupported ? a.note : b.note;
  return r;
}

Latency operator*(const Latency& a, double k) {
  Latency r = a;
  r.ms *= k;
  r.lo_ms *= k;
  r.hi_ms *= k;
  return r;
}

}  // namespace dlsim
