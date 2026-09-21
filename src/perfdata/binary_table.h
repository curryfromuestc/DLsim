#pragma once
#include <cstdint>
#include <string>
#include <vector>

namespace dlsim {

struct Column {
  enum class Kind : uint8_t { Int = 0, Float = 1, Str = 2, Bool = 3 };
  std::string name;
  Kind kind = Kind::Float;
  std::vector<double> num;
  std::vector<uint32_t> idx;
  std::vector<std::string> dict;

  bool is_str() const { return kind == Kind::Str; }
  const std::string& str(size_t row) const { return dict[idx[row]]; }
  uint32_t intern(const std::string& s);
};

struct Table {
  std::vector<Column> cols;
  uint64_t rows = 0;
  const Column* find(const std::string& name) const;
  Column* find(const std::string& name);
};

void write_binary_table(const std::string& path, const Table& t);
Table read_binary_table(const std::string& path);
uint64_t binary_table_rows(const std::string& path);

}  // namespace dlsim
