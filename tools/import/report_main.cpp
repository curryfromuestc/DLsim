#include <cstdlib>
#include <iostream>
#include <string>

#include "perfdata/measured.h"
#include "perfdata/reports.h"
#include "perfdata/system_specs.h"

int main(int argc, char** argv) {
  if (argc < 4) {
    std::cerr << "usage: dlsim-perfdata-report <perfdata_dir> <systems_dir> rows\n"
                 "       dlsim-perfdata-report <perfdata_dir> <systems_dir> loo <system> <framework> <version> <table> [folds]\n"
                 "       dlsim-perfdata-report <perfdata_dir> <systems_dir> extrap <system> [samples] [ragged]\n"
                 "       dlsim-perfdata-report <perfdata_dir> <systems_dir> layer1 <framework> <version> [samples]\n";
    return 2;
  }
  const auto specs = dlsim::load_system_specs(argv[2]);
  const dlsim::MeasuredTables tables(argv[1], specs);
  const std::string cmd = argv[3];
  if (cmd == "rows") {
    std::cout << dlsim::import_row_counts(tables) << "\n";
  } else if (cmd == "loo" && argc >= 8) {
    std::cout << dlsim::interp_leave_one_out(tables, argv[4], argv[5], argv[6], argv[7], argc > 8 ? std::atoi(argv[8]) : 300) << "\n";
  } else if (cmd == "extrap" && argc >= 5) {
    std::cout << dlsim::extrapolation_check_sglang(tables, argv[4], argc > 6 && std::string(argv[6]) == "ragged", argc > 5 ? std::atoi(argv[5]) : 1500) << "\n";
  } else if (cmd == "layer1" && argc >= 6) {
    std::cout << dlsim::layer1_leave_one_device_out(tables, specs, argv[4], argv[5], argc > 6 ? std::atoi(argv[6]) : 150) << "\n";
  } else {
    std::cerr << "unknown report\n";
    return 2;
  }
  return 0;
}
