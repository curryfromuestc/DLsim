#pragma once
#include <cctype>
#include <string>
#include <vector>

namespace dlsim {

struct Pep440 {
  std::vector<long> release;
  int phase = 0;  // dev -2, a/b/rc -1, final 0, post 1
  long number = 0;
  bool ok = false;
};

inline Pep440 parse_pep440(const std::string& s) {
  Pep440 v;
  size_t i = 0;
  while (i < s.size()) {
    if (!std::isdigit(static_cast<unsigned char>(s[i]))) break;
    long n = 0;
    while (i < s.size() && std::isdigit(static_cast<unsigned char>(s[i]))) n = n * 10 + (s[i++] - '0');
    v.release.push_back(n);
    if (i < s.size() && s[i] == '.' && i + 1 < s.size() && std::isdigit(static_cast<unsigned char>(s[i + 1]))) ++i;
  }
  if (v.release.empty()) return v;
  while (i < s.size() && (s[i] == '.' || s[i] == '-' || s[i] == '_')) ++i;
  std::string tag;
  while (i < s.size() && std::isalpha(static_cast<unsigned char>(s[i]))) tag += static_cast<char>(std::tolower(s[i++]));
  while (i < s.size() && std::isdigit(static_cast<unsigned char>(s[i]))) v.number = v.number * 10 + (s[i++] - '0');
  if (tag.empty()) v.phase = 0;
  else if (tag == "dev") v.phase = -2;
  else if (tag == "post") v.phase = 1;
  else if (tag == "rc" || tag == "a" || tag == "b" || tag == "c") v.phase = -1;
  else return v;
  v.ok = i == s.size();
  return v;
}

// Parseable versions order by PEP 440; unparseable ones sort below and by string.
inline int compare_versions(const std::string& a, const std::string& b) {
  const Pep440 pa = parse_pep440(a), pb = parse_pep440(b);
  if (pa.ok != pb.ok) return pa.ok ? 1 : -1;
  if (!pa.ok) return a < b ? -1 : a > b ? 1 : 0;
  const size_t n = std::max(pa.release.size(), pb.release.size());
  for (size_t i = 0; i < n; ++i) {
    const long x = i < pa.release.size() ? pa.release[i] : 0;
    const long y = i < pb.release.size() ? pb.release[i] : 0;
    if (x != y) return x < y ? -1 : 1;
  }
  if (pa.phase != pb.phase) return pa.phase < pb.phase ? -1 : 1;
  if (pa.number != pb.number) return pa.number < pb.number ? -1 : 1;
  return 0;
}

}  // namespace dlsim
