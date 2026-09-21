#include "solver/solver.h"

#include <cassert>
#include <cmath>
#include <cstdio>
#include <stdexcept>

using namespace dlsim;

double get(const Point& x, const char* k, double d) {
  auto it = x.find(k);
  return it == x.end() ? d : it->second;
}

Point forward(const Point& x) {
  double C = get(x, "C", 100), B = get(x, "B", 50), L = get(x, "L_out", 10);
  return {{"y1", std::pow(C, 0.6) * std::pow(B, 0.4) / (1 + L * 0)}, {"y2", C + 2 * B}};
}

Variable var(const char* name, double lo, double hi) { return {name, lo, hi, true, {}}; }

bool near(double a, double b, double tol) { return std::abs(a - b) <= tol; }

void test_recovery() {
  Point x0 = {{"C", 100}, {"B", 50}, {"L_out", 10}};
  Point y0 = forward(x0);
  Query q;
  q.variables = {var("B", 1, 1000)};
  q.targets = {{"y1", y0.at("y1")}};
  int calls = 0;
  Forward fixed = [&](const Point& x) {
    ++calls;
    Point z = x0;
    z["B"] = x.at("B");
    return forward(z);
  };
  Result r = solve(q, fixed);
  double Bs = r.thresholds.at("B");
  double grid_ratio = std::pow(1000.0, 1.0 / 7);
  std::printf("recovery: y0=%.6f B*=%.6f rel_err=%.2e grid_ratio=%.3f full_calls=%d evaluated=%zu solutions=%zu\n",
              y0.at("y1"), Bs, Bs / 50 - 1, grid_ratio, calls, r.evaluated.size(), r.solutions.size());
  assert(r.monotone);
  assert(r.solutions.size() == 1);
  assert(Bs >= 50 * (1 - 1e-9) && Bs <= 50 * 1.01);
  assert(Bs / 50 < grid_ratio);
  assert(r.solutions[0].feasible && near(r.solutions[0].x.at("B"), Bs, 1e-12));
  assert(r.sensitivities.size() == 1 && near(r.sensitivities[0].J[0][0], 0.4, 1e-9));
  assert((int)r.evaluated.size() == calls);
  assert(r.active_constraints.empty() && r.inactive_constraints.empty());
}

void test_seeds() {
  Query q;
  q.variables = {var("B", 1, 1000)};
  q.targets = {{"y1", 1e9}};
  q.seeds = 3;
  bool saw_seed = false;
  Forward noisy = [&](const Point& x) {
    saw_seed = x.count("seed") > 0;
    Point y = forward(x);
    y["y1"] += 0.01 * (x.at("seed") - 1);
    return y;
  };
  Result r = solve(q, noisy);
  Point clean = forward(r.evaluated[0].x);
  std::printf("seeds: saw_seed=%d y1_avg=%.12f y1_clean=%.12f diff=%.2e\n", saw_seed, r.evaluated[0].y.at("y1"),
              clean.at("y1"), r.evaluated[0].y.at("y1") - clean.at("y1"));
  assert(saw_seed);
  assert(near(r.evaluated[0].y.at("y1"), clean.at("y1"), 1e-9));
  assert(r.evaluated[0].x.count("seed") == 0);
  assert(r.solutions.empty() && r.thresholds.empty());
}

void test_unidentifiable() {
  Point x0 = {{"C", 100}, {"B", 50}, {"L_out", 10}};
  Sensitivity s = sensitivity(x0, {var("C", 1, 1000), var("B", 1, 1000), var("L_out", 1, 100)}, {"y1", "y2"},
                              forward, 0.05);
  std::printf("unidentifiable: J=[[%.4f %.4f %.4f][%.4f %.4f %.4f]] norms=[%.4f %.4f %.2e] sv=[%.4f %.4f] cond=%.3f flagged=",
              s.J[0][0], s.J[0][1], s.J[0][2], s.J[1][0], s.J[1][1], s.J[1][2], s.column_norms[0], s.column_norms[1],
              s.column_norms[2], s.singular_values[0], s.singular_values[1], s.condition_number);
  for (const auto& n : s.unidentifiable) std::printf("%s ", n.c_str());
  std::printf("\n");
  assert(near(s.J[0][0], 0.6, 1e-9) && near(s.J[0][1], 0.4, 1e-9) && near(s.J[0][2], 0, 1e-12));
  assert(near(s.J[1][0], 0.5, 1e-2) && near(s.J[1][1], 0.5, 1e-2) && near(s.J[1][2], 0, 1e-12));
  assert(s.column_norms[2] == 0);
  assert(s.singular_values.size() == 2 && std::isfinite(s.condition_number));
  assert(s.unidentifiable.size() == 1 && s.unidentifiable[0] == "L_out");
}

void test_insensitive_threshold() {
  Query q;
  q.variables = {var("L_out", 1, 100)};
  q.targets = {{"y1", 10}};
  Result r = solve(q, forward);
  int feasible = 0;
  for (const Evaluation& e : r.evaluated) feasible += e.feasible;
  std::printf("insensitive: evaluated=%zu feasible=%d solutions=%zu thresholds=%zu\n", r.evaluated.size(), feasible,
              r.solutions.size(), r.thresholds.size());
  assert(r.evaluated.size() == 8 && feasible == 8);
  assert(r.solutions.empty() && r.thresholds.empty() && r.monotone);
}

void test_collinear() {
  Forward f = [](const Point& x) {
    double C = x.at("C"), B = x.at("B"), D = x.at("D");
    return Point{{"y1", C * D}, {"y2", B * std::pow(C, 0.5) * std::pow(D, 0.5001)}, {"y3", B * B}};
  };
  Point x = {{"C", 3}, {"B", 7}, {"D", 5}};
  Sensitivity s = sensitivity(x, {var("C", 1, 10), var("B", 1, 10), var("D", 1, 10)}, {"y1", "y2", "y3"}, f, 0.05);
  std::printf("collinear: norms=[%.4f %.4f %.4f] sv=[%.4f %.4f %.3e] cond=%.3e flagged=", s.column_norms[0],
              s.column_norms[1], s.column_norms[2], s.singular_values[0], s.singular_values[1], s.singular_values[2],
              s.condition_number);
  for (const auto& n : s.unidentifiable) std::printf("%s ", n.c_str());
  std::printf("\n");
  assert(s.singular_values.size() == 3);
  assert(s.singular_values[2] < 1e-3 * s.singular_values[0]);
  assert(s.condition_number > 1e3);
  assert(s.unidentifiable.size() == 2 && s.unidentifiable[0] == "C" && s.unidentifiable[1] == "D");
}

void test_pareto() {
  Query q;
  q.kind = QueryKind::Pareto;
  q.variables = {var("C", 1, 100), var("B", 1, 100)};
  q.objectives = {{"y1", false}, {"y2", true}};
  Result r = solve(q, forward);
  auto dominates = [](const Evaluation& a, const Evaluation& b) {
    bool ge = a.y.at("y1") >= b.y.at("y1") && a.y.at("y2") <= b.y.at("y2");
    bool strict = a.y.at("y1") > b.y.at("y1") || a.y.at("y2") < b.y.at("y2");
    return ge && strict;
  };
  size_t grid_points = 64;
  int dominated_by_front = 0;
  for (size_t i = 0; i < grid_points; ++i) {
    const Evaluation& e = r.evaluated[i];
    bool on_front = false, dom = false;
    for (const Evaluation& s : r.solutions) {
      if (s.x == e.x) on_front = true;
      if (dominates(s, e)) dom = true;
    }
    assert(on_front || dom);
    dominated_by_front += dom;
  }
  for (const Evaluation& a : r.solutions)
    for (const Evaluation& b : r.solutions) assert(!dominates(a, b));
  bool has_min_cost = false, has_max_y1 = false;
  for (const Evaluation& s : r.solutions) {
    has_min_cost |= near(s.x.at("C"), 1, 1e-12) && near(s.x.at("B"), 1, 1e-12);
    has_max_y1 |= near(s.x.at("C"), 100, 1e-9) && near(s.x.at("B"), 100, 1e-9);
  }
  std::printf("pareto: front=%zu dominated=%d evaluated=%zu sensitivities=%zu unidentifiable_in_first=%zu\n",
              r.solutions.size(), dominated_by_front, r.evaluated.size(), r.sensitivities.size(),
              r.sensitivities[0].unidentifiable.size());
  assert(r.solutions.size() >= 2 && has_min_cost && has_max_y1);
  assert(r.sensitivities.size() == r.solutions.size());
  assert(r.monotone);
}

void test_constraints() {
  Query q;
  q.variables = {var("C", 1, 100), var("B", 1, 100)};
  q.targets = {{"y1", 10}};
  q.constraints = {{{{"C", 1}, {"B", 2}}, 60, "budget"}, {{{"C", 1}, {"B", 1}}, 1000, "slack"}};
  Result r = solve(q, forward);
  double tightest = 0;
  int violating = 0;
  for (const Evaluation& e : r.evaluated) {
    double v = e.x.at("C") + 2 * e.x.at("B");
    if (v > 60) {
      ++violating;
      assert(!e.feasible);
    }
  }
  for (const Evaluation& s : r.solutions) {
    assert(s.feasible && s.y.at("y1") >= 10);
    double v = s.x.at("C") + 2 * s.x.at("B");
    assert(v <= 60);
    tightest = std::max(tightest, v);
  }
  std::printf("constraints: solutions=%zu evaluated=%zu violating_probes=%d tightest_budget=%.4f active=[",
              r.solutions.size(), r.evaluated.size(), violating, tightest);
  for (const auto& n : r.active_constraints) std::printf("%s ", n.c_str());
  std::printf("] inactive=[");
  for (const auto& n : r.inactive_constraints) std::printf("%s ", n.c_str());
  std::printf("]\n");
  assert(tightest >= 60 * 0.99);
  assert(r.active_constraints.size() == 1 && r.active_constraints[0] == "budget");
  assert(r.inactive_constraints.size() == 1 && r.inactive_constraints[0] == "slack");
  assert(r.monotone);
}

void test_optimum() {
  Query q;
  q.kind = QueryKind::Optimum;
  q.variables = {var("C", 1, 100), var("B", 1, 100)};
  q.targets = {{"y1", 10}};
  q.objectives = {{"y2", true}};
  Result r = solve(q, forward);
  const Evaluation& o = r.solutions.at(0);
  double continuous_optimum = 10 / std::pow(3, 0.6) * 5;
  std::printf("optimum: C=%.4f B=%.4f y1=%.4f y2=%.4f continuous_optimum=%.4f unidentifiable=%zu\n", o.x.at("C"),
              o.x.at("B"), o.y.at("y1"), o.y.at("y2"), continuous_optimum, r.sensitivities[0].unidentifiable.size());
  assert(r.solutions.size() == 1 && o.feasible);
  assert(o.y.at("y1") >= 10);
  assert(o.y.at("y2") >= continuous_optimum && o.y.at("y2") < 27);
  for (const Evaluation& e : r.evaluated)
    if (e.feasible) assert(e.y.at("y2") >= o.y.at("y2"));
  assert(r.sensitivities.size() == 1 && r.sensitivities[0].unidentifiable.empty());
}

void test_rejected_queries() {
  Query q;
  q.variables = {var("B", 1, 1000)};
  q.kind = QueryKind::Optimum;
  bool thrown = false;
  try {
    solve(q, forward);
  } catch (const std::invalid_argument&) {
    thrown = true;
  }
  assert(thrown);
  q.kind = QueryKind::Pareto;
  q.objectives = {{"y2", true}};
  thrown = false;
  try {
    solve(q, forward);
  } catch (const std::invalid_argument&) {
    thrown = true;
  }
  assert(thrown);
  std::printf("rejected: Optimum without objective and Pareto with one objective both throw\n");
}

void test_screen() {
  Query q;
  q.variables = {var("B", 1, 1000)};
  Point y0 = forward({{"C", 100}, {"B", 500}});
  q.targets = {{"y1", y0.at("y1")}};
  int full_calls = 0, screen_calls = 0;
  Forward full = [&](const Point& x) {
    ++full_calls;
    return forward(x);
  };
  Forward screen = [&](const Point& x) {
    ++screen_calls;
    Point y = forward(x);
    y["y1"] *= 1.1;
    return y;
  };
  Result plain = solve(q, full);
  int plain_calls = full_calls;
  full_calls = 0;
  Result screened = solve(q, full, screen);
  std::printf("screen: full_calls plain=%d screened=%d screen_calls=%d B* plain=%.4f screened=%.4f\n", plain_calls,
              full_calls, screen_calls, plain.thresholds.at("B"), screened.thresholds.at("B"));
  assert(full_calls < plain_calls);
  assert(screen_calls == 32);
  assert(near(plain.thresholds.at("B"), 500, 500 * 1e-2) && near(screened.thresholds.at("B"), 500, 500 * 1e-2));
  assert(screened.thresholds.at("B") >= 500 * (1 - 1e-9));
  assert(screened.solutions.size() == 1);
}

int main() {
  test_recovery();
  test_seeds();
  test_unidentifiable();
  test_insensitive_threshold();
  test_collinear();
  test_pareto();
  test_constraints();
  test_optimum();
  test_rejected_queries();
  test_screen();
  std::printf("solver_test: all assertions passed\n");
  return 0;
}
