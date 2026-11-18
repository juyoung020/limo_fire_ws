// 실물 측정 → 시뮬 파라미터 후보. ROS 의존 없음. CSV 줄은 쉼표로 가르고 첫 줄은 머리글(무시).
//   echo : angle_deg,true_m,range_m[,material]   range 가 비었거나 inf·nan·0 이면 메아리 없음
//   slip : true_m,odom_m                          줄자로 잰 실제 이동량 대 /odom 이동량
//   drift: t_s,yaw_deg                            정지(또는 일정 주행) 때 IMU yaw 기록
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <limits>
#include <map>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

namespace real_calib {

inline std::vector<std::string> split(const std::string& s) {
  std::vector<std::string> out;
  std::string cur;
  for (char c : s) {
    if (c == ',') { out.push_back(cur); cur.clear(); }
    else if (c != '\r' && c != '\n') cur += c;
  }
  out.push_back(cur);
  return out;
}

inline std::optional<double> num(const std::string& s) {
  if (s.empty()) return std::nullopt;
  char* end = nullptr;
  const double v = std::strtod(s.c_str(), &end);
  if (end == s.c_str() || *end != '\0') return std::nullopt;
  return v;
}

struct Stat {
  size_t n = 0;
  double mean = 0, sd = 0;
};
inline Stat stat(const std::vector<double>& v) {
  Stat s;
  s.n = v.size();
  if (v.empty()) return s;
  for (double x : v) s.mean += x;
  s.mean /= static_cast<double>(v.size());
  if (v.size() > 1) {
    double q = 0;
    for (double x : v) q += (x - s.mean) * (x - s.mean);
    s.sd = std::sqrt(q / static_cast<double>(v.size() - 1));
  }
  return s;
}

// ---- echo ----
struct EchoRow {
  double angle, true_m;
  std::optional<double> range;   // 없으면 메아리 없음
};

inline std::vector<EchoRow> read_echo(std::istream& in, size_t* skipped = nullptr) {
  std::vector<EchoRow> rows;
  std::string line;
  bool first = true;
  size_t bad = 0;
  while (std::getline(in, line)) {
    if (first) { first = false; if (!num(split(line)[0])) continue; }   // 머리글
    if (line.empty() || line == "\r") continue;
    const auto f = split(line);
    const auto a = f.size() > 0 ? num(f[0]) : std::nullopt, t = f.size() > 1 ? num(f[1]) : std::nullopt;
    if (!a || !t || f.size() < 3) { ++bad; continue; }
    EchoRow r{*a, *t, std::nullopt};
    const auto rg = num(f[2]);
    if (rg && std::isfinite(*rg) && *rg > 0.0) r.range = *rg;
    rows.push_back(r);
  }
  if (skipped) *skipped = bad;
  return rows;
}

struct AngleBin {
  double angle;
  size_t n = 0, echoes = 0;
  double rate() const { return n ? static_cast<double>(echoes) / static_cast<double>(n) : 0.0; }
  double bias = 0, sd = 0, sd_rel = 0;   // 메아리 온 것만: 잰 − 참, 표준편차(m), 참거리에 대한 비율
};

inline std::vector<AngleBin> by_angle(const std::vector<EchoRow>& rows) {
  std::map<long, std::vector<const EchoRow*>> g;
  for (const auto& r : rows) g[std::lround(r.angle * 10.0)].push_back(&r);   // 0.1° 로 묶음
  std::vector<AngleBin> out;
  for (const auto& [k, v] : g) {
    AngleBin b;
    b.angle = static_cast<double>(k) / 10.0;
    b.n = v.size();
    std::vector<double> err, rel;
    for (const auto* r : v)
      if (r->range) { ++b.echoes; err.push_back(*r->range - r->true_m); rel.push_back((*r->range - r->true_m) / r->true_m); }
    const Stat e = stat(err);
    b.bias = e.mean;
    b.sd = e.sd;
    b.sd_rel = stat(rel).sd;
    out.push_back(b);
  }
  return out;
}

// 메아리 비율이 문턱 이상으로 이어지는 마지막 각도(hard 문턱 후보)와 처음 ≤ zero_rate 가 되는 각도(soft 끝 후보). 없으면 NaN
struct EchoShape {
  double full_deg = std::numeric_limits<double>::quiet_NaN();   // 이 각도까지는 늘 메아리 (비율 ≥ full_rate)
  double none_deg = std::numeric_limits<double>::quiet_NaN();   // 이 각도부터는 거의 없음 (비율 ≤ zero_rate)
};
inline EchoShape echo_shape(const std::vector<AngleBin>& bins, double full_rate = 0.95, double zero_rate = 0.05) {
  EchoShape s;
  // 각도 순으로 본다. full: 처음부터 이어서 full_rate 이상인 마지막 각도. none: full 뒤에 처음 zero_rate 이하
  size_t i = 0;
  while (i < bins.size() && bins[i].rate() >= full_rate) { s.full_deg = bins[i].angle; ++i; }
  for (; i < bins.size(); ++i)
    if (bins[i].rate() <= zero_rate) { s.none_deg = bins[i].angle; break; }
  return s;
}

// 노이즈: 작은 각도(정면)에서 참거리마다 잰 − 참 의 평균(편향)·표준편차
struct DistBin {
  double true_m;
  size_t n = 0;
  double bias = 0, sd = 0;
};
inline std::vector<DistBin> by_distance(const std::vector<EchoRow>& rows, double max_angle) {
  std::map<long, std::vector<double>> g;
  for (const auto& r : rows)
    if (std::fabs(r.angle) <= max_angle && r.range) g[std::lround(r.true_m * 100.0)].push_back(*r.range - r.true_m);
  std::vector<DistBin> out;
  for (const auto& [k, v] : g) {
    const Stat s = stat(v);
    out.push_back({static_cast<double>(k) / 100.0, s.n, s.mean, s.sd});
  }
  return out;
}

// ---- slip ----
struct Slip {
  Stat ratio;   // 참 / odom
};
inline std::optional<Slip> read_slip(std::istream& in, size_t* used = nullptr) {
  std::vector<double> r;
  std::string line;
  while (std::getline(in, line)) {
    const auto f = split(line);
    if (f.size() < 2) continue;
    const auto t = num(f[0]), o = num(f[1]);
    if (!t || !o || *o == 0.0) continue;   // 머리글·빈 줄·0 나누기
    r.push_back(*t / *o);
  }
  if (used) *used = r.size();
  if (r.empty()) return std::nullopt;
  return Slip{stat(r)};
}

// ---- drift ----
struct Drift {
  double deg_per_min = 0, resid_sd_deg = 0;
  size_t n = 0;
};
inline std::optional<Drift> read_drift(std::istream& in) {
  std::vector<double> t, y;
  std::string line;
  while (std::getline(in, line)) {
    const auto f = split(line);
    if (f.size() < 2) continue;
    const auto a = num(f[0]), b = num(f[1]);
    if (!a || !b) continue;
    double v = *b;
    if (!y.empty()) {   // 360° 넘김 펴기
      while (v - y.back() > 180.0) v -= 360.0;
      while (v - y.back() < -180.0) v += 360.0;
    }
    t.push_back(*a);
    y.push_back(v);
  }
  if (t.size() < 3) return std::nullopt;
  const double n = static_cast<double>(t.size());
  double mt = 0, my = 0;
  for (size_t i = 0; i < t.size(); ++i) { mt += t[i]; my += y[i]; }
  mt /= n;
  my /= n;
  double sxy = 0, sxx = 0;
  for (size_t i = 0; i < t.size(); ++i) { sxy += (t[i] - mt) * (y[i] - my); sxx += (t[i] - mt) * (t[i] - mt); }
  if (sxx <= 0) return std::nullopt;
  const double slope = sxy / sxx;   // deg / s
  double q = 0;
  for (size_t i = 0; i < t.size(); ++i) { const double e = y[i] - (my + slope * (t[i] - mt)); q += e * e; }
  return Drift{slope * 60.0, std::sqrt(q / (n - 1.0)), t.size()};
}

}  // namespace real_calib
