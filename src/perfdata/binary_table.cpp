#include "perfdata/binary_table.h"

#include <cstring>
#include <fstream>
#include <stdexcept>

namespace dlsim {
namespace {

constexpr char kMagic[8] = {'D', 'L', 'S', 'I', 'M', 'T', 'B', '1'};

template <class T>
void put(std::ostream& o, const T& v) {
  o.write(reinterpret_cast<const char*>(&v), sizeof v);
}
template <class T>
T get(std::istream& i) {
  T v;
  i.read(reinterpret_cast<char*>(&v), sizeof v);
  if (!i) throw std::runtime_error("binary table: truncated");
  return v;
}
void put_str(std::ostream& o, const std::string& s) {
  put<uint32_t>(o, s.size());
  o.write(s.data(), s.size());
}
std::string get_str(std::istream& i) {
  std::string s(get<uint32_t>(i), '\0');
  i.read(s.data(), s.size());
  if (!i) throw std::runtime_error("binary table: truncated");
  return s;
}

}  // namespace

uint32_t Column::intern(const std::string& s) {
  for (uint32_t i = 0; i < dict.size(); ++i)
    if (dict[i] == s) return i;
  dict.push_back(s);
  return dict.size() - 1;
}

const Column* Table::find(const std::string& name) const {
  for (const auto& c : cols)
    if (c.name == name) return &c;
  return nullptr;
}
Column* Table::find(const std::string& name) {
  for (auto& c : cols)
    if (c.name == name) return &c;
  return nullptr;
}

void write_binary_table(const std::string& path, const Table& t) {
  std::ofstream o(path, std::ios::binary);
  if (!o) throw std::runtime_error("cannot write " + path);
  o.write(kMagic, 8);
  put<uint32_t>(o, t.cols.size());
  put<uint64_t>(o, t.rows);
  for (const auto& c : t.cols) {
    put<uint8_t>(o, static_cast<uint8_t>(c.kind));
    put_str(o, c.name);
    if (c.is_str()) {
      put<uint32_t>(o, c.dict.size());
      for (const auto& s : c.dict) put_str(o, s);
      o.write(reinterpret_cast<const char*>(c.idx.data()), c.idx.size() * sizeof(uint32_t));
    } else {
      o.write(reinterpret_cast<const char*>(c.num.data()), c.num.size() * sizeof(double));
    }
  }
}

Table read_binary_table(const std::string& path) {
  std::ifstream i(path, std::ios::binary);
  if (!i) throw std::runtime_error("cannot read " + path);
  char magic[8];
  i.read(magic, 8);
  if (!i || std::memcmp(magic, kMagic, 8) != 0) throw std::runtime_error("bad magic in " + path);
  Table t;
  const auto ncols = get<uint32_t>(i);
  t.rows = get<uint64_t>(i);
  t.cols.resize(ncols);
  for (auto& c : t.cols) {
    c.kind = static_cast<Column::Kind>(get<uint8_t>(i));
    c.name = get_str(i);
    if (c.is_str()) {
      c.dict.resize(get<uint32_t>(i));
      for (auto& s : c.dict) s = get_str(i);
      c.idx.resize(t.rows);
      i.read(reinterpret_cast<char*>(c.idx.data()), t.rows * sizeof(uint32_t));
    } else {
      c.num.resize(t.rows);
      i.read(reinterpret_cast<char*>(c.num.data()), t.rows * sizeof(double));
    }
    if (!i) throw std::runtime_error("binary table: truncated " + path);
  }
  return t;
}

uint64_t binary_table_rows(const std::string& path) {
  std::ifstream i(path, std::ios::binary);
  if (!i) throw std::runtime_error("cannot read " + path);
  char magic[8];
  i.read(magic, 8);
  if (!i || std::memcmp(magic, kMagic, 8) != 0) throw std::runtime_error("bad magic in " + path);
  get<uint32_t>(i);
  return get<uint64_t>(i);
}

}  // namespace dlsim
