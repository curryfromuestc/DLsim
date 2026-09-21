#pragma once
// MeasuredTables: OpLatencySource over the imported operator tables (design: operator-latency.md 实现一).
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <tuple>
#include <vector>

#include "config/config.h"
#include "perfdata/binary_table.h"
#include "perfdata/op_latency.h"

namespace dlsim {

struct TableSource {
  std::string path;
  std::string channel;                       // primary, declared_reuse, fallback, cross_backend
  std::vector<std::string> kernel_sources;   // row filter for cross_backend sources
};

struct TableAlias {
  std::string system, framework, version, table;
  std::vector<TableSource> sources;
};

struct ImportedFile {
  std::string path, source;
  uint64_t rows = 0;
  std::string system, family, framework, version, table;
};

struct VersionSlots {
  std::string current, previous, next;
};

bool is_provenance_column(const std::string& name);  // framework, version, device
bool is_key_column(const Column& c);                  // Int/Bool/Str columns that are not provenance

// Resolve a query against the given rows of one table: exact hit, interpolation, or the attention
// sequence-axis extrapolation. `spec` is needed only for GEMM site transfer (SOL ratio).
Latency query_rows(const Table& t, const std::vector<uint32_t>& rows, const OpQuery& q, const DeviceSpec* spec,
                   const std::string& note_prefix = {});

class MeasuredTables : public OpLatencySource {
 public:
  explicit MeasuredTables(std::string perfdata_dir, std::map<std::string, DeviceSpec> specs = {});
  Latency query(const std::string& device, const OpQuery& q) const override;   // memoized per (device, query)
  Latency query_uncached(const std::string& device, const OpQuery& q) const;

  // Merged table for (system, framework, version, table) following the manifest aliases; nullptr when unknown.
  std::shared_ptr<const Table> table(const std::string& system, const std::string& framework, const std::string& version,
                                     const std::string& table) const;
  std::vector<std::string> versions(const std::string& system, const std::string& framework, const std::string& table) const;
  std::string latest_version(const std::string& system, const std::string& framework, const std::string& table) const;
  // Maps the slot names current / previous / next to a version; other strings pass through.
  std::string resolve_version(const std::string& system, const std::string& framework, const std::string& version) const;
  bool has(const std::string& system, const std::string& framework, const std::string& version, const std::string& table) const;
  std::vector<std::string> systems() const;

  const std::string& dir() const { return dir_; }
  const std::string& source_commit() const { return commit_; }
  const std::vector<ImportedFile>& files() const { return files_; }
  const std::vector<TableAlias>& aliases() const { return aliases_; }
  const std::map<std::string, DeviceSpec>& specs() const { return specs_; }

 private:
  std::string dir_;
  std::map<std::string, DeviceSpec> specs_;
  std::string commit_;
  std::vector<ImportedFile> files_;
  std::vector<TableAlias> aliases_;
  std::map<std::tuple<std::string, std::string, std::string, std::string>, size_t> alias_index_;
  std::map<std::pair<std::string, std::string>, VersionSlots> slots_;
  mutable std::mutex mu_;
  mutable std::map<size_t, std::shared_ptr<const Table>> cache_;
  mutable std::map<std::string, Latency> memo_;
};

}  // namespace dlsim
