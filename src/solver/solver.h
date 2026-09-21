#pragma once
// Inverse solving and sweeps. Design: inverse-solving.md.
#include <functional>
#include <map>
#include <string>
#include <vector>

namespace dlsim {

using Point = std::map<std::string, double>;              // design variable name -> value
using Forward = std::function<Point(const Point&)>;      // x -> metrics (y)

struct Variable {
  std::string name;
  double lo = 0, hi = 0;
  bool log_scale = true;
  std::vector<double> choices;   // non-empty = discrete
};

struct Target { std::string metric; double min_value; };                    // y >= min_value
struct Objective { std::string metric; bool minimize = true; };
struct LinearConstraint { std::map<std::string, double> weights; double bound; std::string name; };  // sum w x <= bound

enum class QueryKind { FeasibleSet, Optimum, Pareto };

struct Query {
  QueryKind kind = QueryKind::FeasibleSet;
  std::vector<Variable> variables;
  std::vector<Target> targets;
  std::vector<Objective> objectives;       // Optimum: exactly one; Pareto: two or more
  std::vector<LinearConstraint> constraints;
  int grid_per_axis = 8;
  double sensitivity_step = 0.05;          // relative step for the log-log Jacobian
  int seeds = 1;
};

struct Evaluation { Point x; Point y; bool feasible = false; };

struct Sensitivity {
  std::vector<std::string> metrics, variables;
  std::vector<std::vector<double>> J;      // d log y_i / d log x_j
  std::vector<double> column_norms;
  std::vector<double> singular_values;
  double condition_number = 0;
  std::vector<std::string> unidentifiable; // variables with near-zero column norm or in near-null combinations
};

struct Result {
  std::vector<Evaluation> evaluated;       // every forward run, for traceability
  std::vector<Evaluation> solutions;       // feasible set boundary / optimum / Pareto front
  std::map<std::string, double> thresholds;   // conditional thresholds for single-variable queries
  std::vector<Sensitivity> sensitivities;  // one per solution
  std::vector<std::string> active_constraints, inactive_constraints;
  bool monotone = true;                    // monotonicity check on the bisection axes
};

Result solve(const Query& q, const Forward& full, const Forward& screen = nullptr);
Sensitivity sensitivity(const Point& x, const std::vector<Variable>& vars, const std::vector<std::string>& metrics,
                        const Forward& f, double rel_step);

}  // namespace dlsim
