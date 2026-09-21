#include "perfdata/decomposition.h"

#include <Eigen/Dense>

#include <algorithm>
#include <cmath>
#include <limits>
#include <sstream>

#include "perfdata/dtype.h"

namespace dlsim {

const char* to_string(DecompositionFit::Rank r) {
  switch (r) {
    case DecompositionFit::Rank::Full: return "full";
    case DecompositionFit::Rank::BwDeficient: return "bw_deficient";
    case DecompositionFit::Rank::CDeficient: return "c_deficient";
    case DecompositionFit::Rank::Underdetermined: return "underdetermined";
  }
  return "?";
}

namespace {

double condition_number(const Eigen::MatrixXd& A) {
  if (A.rows() < A.cols()) return std::numeric_limits<double>::infinity();
  Eigen::JacobiSVD<Eigen::MatrixXd> svd(A);
  const auto& s = svd.singularValues();
  const double mn = s(s.size() - 1);
  return mn > 0 ? s(0) / mn : std::numeric_limits<double>::infinity();
}

// min ||A x - t||, x >= 0, by enumerating the active set.
Eigen::VectorXd nnls(const Eigen::MatrixXd& A, const Eigen::VectorXd& t) {
  const int p = A.cols();
  Eigen::VectorXd best = Eigen::VectorXd::Zero(p);
  double best_res = t.norm();
  for (int mask = 1; mask < (1 << p); ++mask) {
    std::vector<int> cols;
    for (int j = 0; j < p; ++j)
      if (mask & (1 << j)) cols.push_back(j);
    if (A.rows() < static_cast<long>(cols.size())) continue;
    Eigen::MatrixXd S(A.rows(), cols.size());
    for (size_t j = 0; j < cols.size(); ++j) S.col(j) = A.col(cols[j]);
    Eigen::VectorXd y = S.colPivHouseholderQr().solve(t);
    if ((y.array() < 0).any()) continue;
    const double res = (S * y - t).norm();
    if (res < best_res - 1e-12 * std::max(1.0, t.norm())) {
      best_res = res;
      best.setZero();
      for (size_t j = 0; j < cols.size(); ++j) best(cols[j]) = y(j);
    }
  }
  return best;
}

DecompositionFit fit_impl(const std::vector<std::string>& devices, const std::vector<double>& C, const std::vector<double>& Bw,
                          const std::vector<double>& t, bool with_loo) {
  DecompositionFit f;
  f.devices = devices;
  const int n = t.size();
  Eigen::MatrixXd A(n, 3);
  Eigen::VectorXd y(n);
  for (int i = 0; i < n; ++i) {
    A(i, 0) = 1.0 / C[i];
    A(i, 1) = 1.0 / Bw[i];
    A(i, 2) = 1.0;
    y(i) = t[i];
  }
  Eigen::Vector3d scale;
  for (int j = 0; j < 3; ++j) scale(j) = A.col(j).cwiseAbs().maxCoeff();
  Eigen::MatrixXd As = A;
  for (int j = 0; j < 3; ++j) As.col(j) /= scale(j);
  f.cond = condition_number(As);
  const double cond_c = condition_number(As(Eigen::all, std::vector<int>{0, 2}));
  const double cond_b = condition_number(As(Eigen::all, std::vector<int>{1, 2}));
  std::vector<int> cols;
  if (f.cond <= kConditionLimit) {
    f.rank = DecompositionFit::Rank::Full;
    cols = {0, 1, 2};
  } else if (cond_c <= kConditionLimit) {
    f.rank = DecompositionFit::Rank::BwDeficient;
    cols = {0, 2};
  } else if (cond_b <= kConditionLimit) {
    f.rank = DecompositionFit::Rank::CDeficient;
    cols = {1, 2};
  } else {
    f.rank = DecompositionFit::Rank::Underdetermined;
    return f;
  }
  Eigen::MatrixXd Ar = As(Eigen::all, cols);
  Eigen::VectorXd x = nnls(Ar, y);
  Eigen::VectorXd full = Eigen::VectorXd::Zero(3);
  for (size_t j = 0; j < cols.size(); ++j) full(cols[j]) = x(j) / scale(cols[j]);
  f.a = full(0);
  f.b = full(1);
  f.c = full(2);
  if (f.rank != DecompositionFit::Rank::Full) f.k = full(2);
  Eigen::VectorXd pred = A * full;
  for (int i = 0; i < n; ++i) f.residual_rel = std::max(f.residual_rel, std::abs(pred(i) - t[i]) / t[i]);
  if (with_loo && f.rank == DecompositionFit::Rank::Full && n >= 4) {
    for (int i = 0; i < n; ++i) {
      std::vector<std::string> dv;
      std::vector<double> c2, b2, t2;
      for (int j = 0; j < n; ++j) {
        if (j == i) continue;
        dv.push_back(devices[j]);
        c2.push_back(C[j]);
        b2.push_back(Bw[j]);
        t2.push_back(t[j]);
      }
      const DecompositionFit g = fit_impl(dv, c2, b2, t2, false);
      if (g.rank != DecompositionFit::Rank::Full) continue;
      const double p = g.a / C[i] + g.b / Bw[i] + g.c;
      f.loo_rel = std::max(f.loo_rel, std::abs(p - t[i]) / t[i]);
    }
  }
  return f;
}

std::string fmt(double v) {
  std::ostringstream o;
  o.precision(4);
  o << v;
  return o.str();
}

std::string join(const std::vector<std::string>& v) {
  std::string o;
  for (const auto& s : v) o += (o.empty() ? "" : ",") + s;
  return o;
}

}  // namespace

DecompositionFit fit_decomposition(const std::vector<std::string>& devices, const std::vector<double>& C,
                                   const std::vector<double>& Bw, const std::vector<double>& t) {
  return fit_impl(devices, C, Bw, t, true);
}

Latency decomposed_latency(const DecompositionFit& f, double C_t, double Bw_t, const std::vector<double>& C,
                           const std::vector<double>& Bw, const std::vector<double>& t, const Latency& roofline_lo) {
  Latency r;
  std::ostringstream note;
  note << "references=" << join(f.devices) << " rank=" << to_string(f.rank) << " cond=" << fmt(f.cond);
  const double mean = [](const std::vector<double>& v) {
    double s = 0;
    for (double x : v) s += x;
    return s / v.size();
  }(f.rank == DecompositionFit::Rank::BwDeficient ? Bw : C);
  switch (f.rank) {
    case DecompositionFit::Rank::Full:
      r.ms = f.a / C_t + f.b / Bw_t + f.c;
      r.lo_ms = r.ms * std::max(0.0, 1 - f.loo_rel);
      r.hi_ms = r.ms * (1 + f.loo_rel);
      r.source = Source::Decomposed;
      note << " a=" << fmt(f.a) << " b=" << fmt(f.b) << " c=" << fmt(f.c) << " residual=" << fmt(f.residual_rel) << " loo=" << fmt(f.loo_rel);
      break;
    case DecompositionFit::Rank::BwDeficient: {
      const double lo = f.a / C_t + f.k, hi = f.a / C_t + f.k * mean / Bw_t;
      r.lo_ms = std::min(lo, hi);
      r.hi_ms = std::max(lo, hi);
      r.ms = (r.lo_ms + r.hi_ms) / 2;
      r.source = Source::Bound;
      note << " interval rule: t=a/C+k with a=" << fmt(f.a) << " k=" << fmt(f.k) << " Bw_ref=" << fmt(mean) << " residual=" << fmt(f.residual_rel);
      break;
    }
    case DecompositionFit::Rank::CDeficient: {
      const double lo = f.b / Bw_t + f.k, hi = f.b / Bw_t + f.k * mean / C_t;
      r.lo_ms = std::min(lo, hi);
      r.hi_ms = std::max(lo, hi);
      r.ms = (r.lo_ms + r.hi_ms) / 2;
      r.source = Source::Bound;
      note << " interval rule: t=b/Bw+k with b=" << fmt(f.b) << " k=" << fmt(f.k) << " C_ref=" << fmt(mean) << " residual=" << fmt(f.residual_rel);
      break;
    }
    case DecompositionFit::Rank::Underdetermined: {
      double hi = 0, lo = std::numeric_limits<double>::infinity();
      for (size_t i = 0; i < t.size(); ++i) {
        hi = std::max(hi, t[i] * std::max(C[i] / C_t, Bw[i] / Bw_t));
        lo = std::min(lo, t[i] * std::min(C[i] / C_t, Bw[i] / Bw_t));
      }
      if (roofline_lo.has_point()) lo = roofline_lo.ms;
      r.lo_ms = std::min(lo, hi);
      r.hi_ms = hi;
      r.ms = (r.lo_ms + r.hi_ms) / 2;
      r.source = Source::Bound;
      note << " bound: hi=max reference scaled by the least favourable resource, lo=" << (roofline_lo.has_point() ? "roofline" : "min reference scaled by the most favourable resource");
      break;
    }
  }
  r.note = note.str();
  return r;
}

DecomposedSource::DecomposedSource(const MeasuredTables& tables, std::vector<std::string> reference_devices,
                                   std::map<std::string, DeviceSpec> specs)
    : tables_(tables), refs_(std::move(reference_devices)), specs_(std::move(specs)), roofline_(specs_) {}

Latency DecomposedSource::query(const std::string& device, const OpQuery& q) const { return query_with_references(device, q, refs_); }

Latency DecomposedSource::query_with_references(const std::string& device, const OpQuery& q, const std::vector<std::string>& refs) const {
  const auto sit = specs_.find(device);
  if (sit == specs_.end()) return Latency::unsupported("decomposition: no device spec for " + device);
  const DeviceSpec& target = sit->second;
  if (target.memory.empty() || target.memory[0].bandwidth_Bps <= 0) return Latency::unsupported("decomposition: " + device + " has no memory bandwidth");
  std::string dtype;
  for (const char* k : kDtypeKeys)
    if (auto it = q.cat.find(k); it != q.cat.end()) { dtype = it->second; break; }
  if (dtype.empty()) return Latency::unsupported("decomposition: query needs an explicit dtype key");
  const std::string key = flops_key_for_dtype(dtype);
  const auto fit_t = target.flops.find(key);
  if (key.empty() || fit_t == target.flops.end()) return Latency::unsupported("decomposition: " + device + " has no flops for dtype " + dtype);
  std::string framework, version;
  if (auto it = q.cat.find("framework"); it != q.cat.end()) framework = it->second;
  else framework = target.framework;
  if (auto it = q.cat.find("version"); it != q.cat.end()) version = it->second;
  else version = target.version;
  if (framework.empty() || version.empty()) return Latency::unsupported("decomposition: framework and version must be explicit");
  OpQuery rq = q;
  rq.cat["framework"] = framework;
  rq.cat["version"] = version;
  std::vector<std::string> used, excluded;
  std::vector<double> C, Bw, t;
  Source weakest = Source::Measured;
  for (const std::string& ref : refs) {
    if (ref == device) continue;
    const auto rit = specs_.find(ref);
    if (rit == specs_.end()) { excluded.push_back(ref + "(no spec)"); continue; }
    const DeviceSpec& rs = rit->second;
    if (rs.kernel_family != target.kernel_family) { excluded.push_back(ref + "(kernel family " + rs.kernel_family + ")"); continue; }
    const auto fc = rs.flops.find(key);
    if (fc == rs.flops.end() || rs.memory.empty()) { excluded.push_back(ref + "(no " + key + " flops)"); continue; }
    if (!tables_.has(ref, framework, version, q.op)) { excluded.push_back(ref + "(no table)"); continue; }
    const Latency l = tables_.query(ref, rq);
    if (!l.has_point()) { excluded.push_back(ref + "(" + l.note + ")"); continue; }
    used.push_back(ref);
    C.push_back(fc->second);
    Bw.push_back(rs.memory[0].bandwidth_Bps);
    t.push_back(l.ms);
    weakest = weaker(weakest, l.source);
  }
  std::string ex;
  for (const auto& e : excluded) ex += (ex.empty() ? "" : "; ") + e;
  if (used.empty()) return Latency::unsupported("decomposition: no usable reference for " + device + " (" + q.op + " " + framework + " " + version + ") excluded: " + ex);
  const DecompositionFit f = fit_decomposition(used, C, Bw, t);
  Latency r = decomposed_latency(f, fit_t->second, target.memory[0].bandwidth_Bps, C, Bw, t, roofline_.query(device, q));
  r.note = "decomposition for " + device + " (" + q.op + " " + framework + " " + version + " " + dtype + "): " + r.note +
           " reference_source=" + to_string(weakest) + (ex.empty() ? "" : " excluded: " + ex);
  return r;
}

std::vector<DecomposedSource::HoldOut> DecomposedSource::leave_one_device_out(const OpQuery& q) const {
  std::vector<HoldOut> out;
  for (const std::string& d : refs_) {
    HoldOut h;
    h.device = d;
    OpQuery mq = q;
    if (const auto sit = specs_.find(d); sit != specs_.end()) {
      if (!mq.cat.count("framework") && !sit->second.framework.empty()) mq.cat["framework"] = sit->second.framework;
      if (!mq.cat.count("version") && !sit->second.version.empty()) mq.cat["version"] = sit->second.version;
    }
    h.measured = tables_.query(d, mq);
    std::vector<std::string> others;
    for (const std::string& o : refs_)
      if (o != d) others.push_back(o);
    h.predicted = query_with_references(d, q, others);
    out.push_back(std::move(h));
  }
  return out;
}

}  // namespace dlsim
