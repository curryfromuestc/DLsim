#include "solver/solver.h"

#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <limits>
#include <set>
#include <stdexcept>

namespace dlsim {
namespace {

bool discrete(const Variable& v) { return !v.choices.empty(); }
double coord(const Variable& v, double x) { return v.log_scale ? std::log(x) : x; }
double value(const Variable& v, double t) { return v.log_scale ? std::exp(t) : t; }

double grid_value(const Variable& v, int i, int n) {
  if (discrete(v)) return v.choices[i];
  if (n == 1) return v.lo;
  double a = coord(v, v.lo), b = coord(v, v.hi);
  return value(v, a + (b - a) * i / (n - 1));
}

double weighted_sum(const LinearConstraint& c, const Point& x) {
  double s = 0;
  for (const auto& [name, w] : c.weights) s += w * x.at(name);
  return s;
}

struct Search {
  const Query& q;
  const Forward& full;
  Result r;
  std::map<Point, int> cache;
  std::vector<std::string> metrics;
  std::vector<int> shape;
  std::map<std::vector<int>, int> node;

  Search(const Query& query, const Forward& f, bool dense) : q(query), full(f) {
    for (const Target& t : q.targets) add_metric(t.metric);
    for (const Objective& o : q.objectives) add_metric(o.metric);
    for (const Variable& v : q.variables)
      shape.push_back(discrete(v) ? (int)v.choices.size() : (dense ? 4 : 1) * q.grid_per_axis);
  }

  void add_metric(const std::string& m) {
    if (std::find(metrics.begin(), metrics.end(), m) == metrics.end()) metrics.push_back(m);
  }

  bool within_constraints(const Point& x) const {
    for (const LinearConstraint& c : q.constraints)
      if (weighted_sum(c, x) > c.bound) return false;
    return true;
  }

  bool meets_targets(const Point& y) const {
    for (const Target& t : q.targets)
      if (y.at(t.metric) < t.min_value) return false;
    return true;
  }

  int run(const Point& x) {
    auto it = cache.find(x);
    if (it != cache.end()) return it->second;
    Point y;
    if (q.seeds > 1) {
      for (int s = 0; s < q.seeds; ++s) {
        Point xs = x;
        xs["seed"] = s;
        for (const auto& [k, v] : full(xs)) y[k] += v / q.seeds;
      }
    } else {
      y = full(x);
    }
    bool feasible = meets_targets(y) && within_constraints(x);
    r.evaluated.push_back({x, y, feasible});
    int idx = (int)r.evaluated.size() - 1;
    cache[x] = idx;
    return idx;
  }

  int probe(const Point& x) { return within_constraints(x) ? run(x) : -1; }
  bool feasible(int idx) const { return idx >= 0 && r.evaluated[idx].feasible; }

  Point point(const std::vector<int>& idx) const {
    Point x;
    for (size_t a = 0; a < shape.size(); ++a) x[q.variables[a].name] = grid_value(q.variables[a], idx[a], shape[a]);
    return x;
  }

  bool next(std::vector<int>& idx) const {
    for (size_t a = 0; a < shape.size(); ++a) {
      if (++idx[a] < shape[a]) return true;
      idx[a] = 0;
    }
    return false;
  }

  void evaluate_grid(const Forward& screen) {
    std::set<std::vector<int>> candidates;
    std::vector<int> idx(shape.size(), 0);
    do {
      Point x = point(idx);
      if (!within_constraints(x)) {
        node[idx] = -1;
        continue;
      }
      if (!screen) {
        node[idx] = run(x);
        continue;
      }
      if (!meets_targets(screen(x))) continue;
      candidates.insert(idx);
      for (size_t a = 0; a < shape.size(); ++a)
        for (int d : {-1, 1}) {
          std::vector<int> nb = idx;
          nb[a] += d;
          if (nb[a] >= 0 && nb[a] < shape[a]) candidates.insert(nb);
        }
    } while (next(idx));
    for (const auto& c : candidates)
      if (!node.count(c)) node[c] = run(point(c));
  }

  template <class F>
  void adjacent_pairs(size_t a, F fn) {
    std::vector<int> idx(shape.size(), 0);
    do {
      if (idx[a] != 0) continue;
      for (int i = 0; i + 1 < shape[a]; ++i) {
        std::vector<int> lo = idx, hi = idx;
        lo[a] = i;
        hi[a] = i + 1;
        auto il = node.find(lo), ih = node.find(hi);
        if (il != node.end() && ih != node.end()) fn(il->second, ih->second, i);
      }
    } while (next(idx));
  }

  bool monotone_axis(size_t a) {
    bool mono = true;
    for (const Target& t : q.targets) {
      int sign = 0;
      adjacent_pairs(a, [&](int lo, int hi, int) {
        if (lo < 0 || hi < 0) return;
        double d = r.evaluated[hi].y.at(t.metric) - r.evaluated[lo].y.at(t.metric);
        int s = (d > 0) - (d < 0);
        if (s == 0) return;
        if (sign == 0) sign = s;
        else if (sign != s) mono = false;
      });
    }
    return mono;
  }

  int bisect(int f, const Variable& v, double b, double tol) {
    double a = coord(v, r.evaluated[f].x.at(v.name));
    while (std::abs(a - b) > tol) {
      double m = (a + b) / 2;
      Point x = r.evaluated[f].x;
      x[v.name] = value(v, m);
      int i = probe(x);
      if (feasible(i)) {
        a = m;
        f = i;
      } else {
        b = m;
      }
    }
    return f;
  }

  void sweep_axes() {
    std::set<int> sol;
    for (size_t a = 0; a < shape.size(); ++a) {
      const Variable& v = q.variables[a];
      if (discrete(v)) continue;
      bool mono = monotone_axis(a);
      if (!mono) r.monotone = false;
      double tol = 1e-4 * std::abs(coord(v, v.hi) - coord(v, v.lo));
      adjacent_pairs(a, [&](int lo, int hi, int i) {
        bool fl = feasible(lo), fh = feasible(hi);
        if (fl == fh) return;
        int f = fl ? lo : hi;
        double b = coord(v, grid_value(v, fl ? i + 1 : i, shape[a]));
        sol.insert(mono ? bisect(f, v, b, tol) : f);
      });
    }
    for (int i : sol) r.solutions.push_back(r.evaluated[i]);
  }
};

}  // namespace

Result solve(const Query& q, const Forward& full, const Forward& screen) {
  if (q.kind == QueryKind::Optimum && q.objectives.size() != 1)
    throw std::invalid_argument("Optimum query requires exactly one objective");
  if (q.kind == QueryKind::Pareto && q.objectives.size() < 2)
    throw std::invalid_argument("Pareto query requires two or more objectives");

  Search s(q, full, screen != nullptr);
  s.evaluate_grid(screen);
  s.sweep_axes();
  Result& r = s.r;

  int continuous = 0;
  std::string single;
  for (const Variable& v : q.variables)
    if (!discrete(v)) {
      ++continuous;
      single = v.name;
    }
  if (continuous == 1 && !r.solutions.empty()) {
    double best = std::numeric_limits<double>::infinity();
    for (const Evaluation& e : r.solutions) best = std::min(best, e.x.at(single));
    r.thresholds[single] = best;
  }

  if (q.kind == QueryKind::Optimum) {
    const Objective& o = q.objectives[0];
    int best = -1;
    for (size_t i = 0; i < r.evaluated.size(); ++i) {
      if (!r.evaluated[i].feasible) continue;
      double v = r.evaluated[i].y.at(o.metric);
      if (best < 0 || (o.minimize ? v < r.evaluated[best].y.at(o.metric) : v > r.evaluated[best].y.at(o.metric)))
        best = (int)i;
    }
    r.solutions.clear();
    if (best >= 0) r.solutions.push_back(r.evaluated[best]);
  } else if (q.kind == QueryKind::Pareto) {
    auto key = [&](const Evaluation& e) {
      std::vector<double> k;
      for (const Objective& o : q.objectives) k.push_back(o.minimize ? e.y.at(o.metric) : -e.y.at(o.metric));
      return k;
    };
    auto dominates = [](const std::vector<double>& a, const std::vector<double>& b) {
      bool strict = false;
      for (size_t i = 0; i < a.size(); ++i) {
        if (a[i] > b[i]) return false;
        if (a[i] < b[i]) strict = true;
      }
      return strict;
    };
    std::vector<std::pair<int, std::vector<double>>> pool;
    for (size_t i = 0; i < r.evaluated.size(); ++i)
      if (r.evaluated[i].feasible) pool.emplace_back((int)i, key(r.evaluated[i]));
    r.solutions.clear();
    for (const auto& [i, ki] : pool) {
      bool dominated = false;
      for (const auto& [j, kj] : pool)
        if (j != i && dominates(kj, ki)) {
          dominated = true;
          break;
        }
      if (!dominated) r.solutions.push_back(r.evaluated[i]);
    }
  }

  for (const LinearConstraint& c : q.constraints) {
    bool active = std::any_of(r.solutions.begin(), r.solutions.end(), [&](const Evaluation& e) {
      return std::abs(c.bound - weighted_sum(c, e.x)) <= 0.01 * std::abs(c.bound);
    });
    (active ? r.active_constraints : r.inactive_constraints).push_back(c.name);
  }

  Forward recorded = [&](const Point& x) { return r.evaluated[s.run(x)].y; };
  for (const Evaluation& sol : r.solutions)
    r.sensitivities.push_back(sensitivity(sol.x, q.variables, s.metrics, recorded, q.sensitivity_step));
  return std::move(r);
}

Sensitivity sensitivity(const Point& x, const std::vector<Variable>& vars, const std::vector<std::string>& metrics,
                        const Forward& f, double rel_step) {
  Sensitivity s;
  s.metrics = metrics;
  for (const Variable& v : vars)
    if (!discrete(v)) s.variables.push_back(v.name);
  size_t m = metrics.size(), n = s.variables.size();
  s.J.assign(m, std::vector<double>(n, 0));
  double denom = std::log(1 + rel_step) - std::log(1 - rel_step);
  for (size_t j = 0; j < n; ++j) {
    Point up = x, down = x;
    up[s.variables[j]] *= 1 + rel_step;
    down[s.variables[j]] *= 1 - rel_step;
    Point yu = f(up), yd = f(down);
    for (size_t i = 0; i < m; ++i) s.J[i][j] = (std::log(yu.at(metrics[i])) - std::log(yd.at(metrics[i]))) / denom;
  }
  if (m == 0 || n == 0) return s;

  Eigen::MatrixXd J(m, n);
  for (size_t i = 0; i < m; ++i)
    for (size_t j = 0; j < n; ++j) J(i, j) = s.J[i][j];
  for (size_t j = 0; j < n; ++j) s.column_norms.push_back(J.col(j).norm());

  Eigen::JacobiSVD<Eigen::MatrixXd> svd(J, Eigen::ComputeFullV);
  const Eigen::VectorXd& sv = svd.singularValues();
  s.singular_values.assign(sv.data(), sv.data() + sv.size());
  double smax = sv(0), smin = sv(sv.size() - 1);
  s.condition_number = smin > 0 ? smax / smin : std::numeric_limits<double>::infinity();

  std::vector<bool> bad(n, false);
  double nmax = *std::max_element(s.column_norms.begin(), s.column_norms.end());
  for (size_t j = 0; j < n; ++j)
    if (s.column_norms[j] <= 1e-3 * nmax) bad[j] = true;
  const Eigen::MatrixXd& V = svd.matrixV();
  for (size_t k = 0; k < n; ++k) {
    double sigma = (int)k < sv.size() ? sv(k) : 0;
    if (sigma >= 1e-3 * smax) continue;
    double wmax = V.col(k).cwiseAbs().maxCoeff();
    for (size_t j = 0; j < n; ++j)
      if (std::abs(V(j, k)) >= 0.1 * wmax) bad[j] = true;
  }
  for (size_t j = 0; j < n; ++j)
    if (bad[j]) s.unidentifiable.push_back(s.variables[j]);
  return s;
}

}  // namespace dlsim
