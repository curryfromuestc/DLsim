#pragma once
// Validation reports as JSON strings (design: validation.md 第一层, 步骤 2 与 3).
#include <cstdint>
#include <map>
#include <string>

#include "config/config.h"
#include "perfdata/measured.h"

namespace dlsim {

// Imported binary row counts vs the manifest's converted and parquet-metadata row counts.
std::string import_row_counts(const MeasuredTables& tables);

// Hold out one measured row (and, for gemm, one whole (n, k) site) per fold; relative error distribution.
std::string interp_leave_one_out(const MeasuredTables& tables, const std::string& system, const std::string& framework,
                                 const std::string& version, const std::string& table, int folds_per_slice = 300,
                                 uint64_t seed = 1);

// Truncate SGLang 0.5.14 dsv4 csa / hca context and generation tables at 65,536 and predict the rows beyond.
std::string extrapolation_check_sglang(const MeasuredTables& tables, const std::string& system, bool ragged = false, int samples_per_table = 1500,
                                       uint64_t seed = 1);

// Layer-1 leave-one-device-out for gemm, moe and the dsv4 modules on the given framework / version.
std::string layer1_leave_one_device_out(const MeasuredTables& tables, const std::map<std::string, DeviceSpec>& devices,
                                        const std::string& framework, const std::string& version, int samples_per_slice = 150,
                                        uint64_t seed = 1);

}  // namespace dlsim
