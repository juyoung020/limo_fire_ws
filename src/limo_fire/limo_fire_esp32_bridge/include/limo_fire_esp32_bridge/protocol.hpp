// ESP32 시리얼 줄 해석 — 펌웨어 firmware/limo_fire_esp32/include/protocol.h 와 같은 형식 (README "시리얼 형식").
//   $<종류>,<필드>,...*<XOR 체크섬 16 진 2 자리>   — 체크섬 = '$' 와 '*' 사이 글자의 XOR (NMEA 0183 꼴). 줄 끝 \r\n 은 떼고 넘긴다
// ROS 의존 없음(테스트용).
#pragma once

#include <cmath>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace limo_fire_esp32_bridge {

struct Line {
  std::string type;                 // "US", "HB", …
  std::vector<std::string> fields;  // 종류 뒤 필드
};

inline int hex_digit(char c) {
  if (c >= '0' && c <= '9') return c - '0';
  if (c >= 'A' && c <= 'F') return c - 'A' + 10;
  if (c >= 'a' && c <= 'f') return c - 'a' + 10;
  return -1;
}

// 형식·체크섬이 맞으면 Line, 아니면 없음 (깨진 줄은 버린다 — 시리얼 잡음·연결 직후 반쪽 줄)
inline std::optional<Line> parse_line(std::string s) {
  while (!s.empty() && (s.back() == '\r' || s.back() == '\n')) s.pop_back();
  const auto dollar = s.find('$');
  if (dollar == std::string::npos) return std::nullopt;
  s = s.substr(dollar);
  const auto star = s.rfind('*');
  if (star == std::string::npos || star + 3 != s.size() || star < 2) return std::nullopt;
  const int h = hex_digit(s[star + 1]), l = hex_digit(s[star + 2]);
  if (h < 0 || l < 0) return std::nullopt;
  uint8_t c = 0;
  for (size_t i = 1; i < star; ++i) c ^= static_cast<uint8_t>(s[i]);
  if (c != static_cast<uint8_t>(h * 16 + l)) return std::nullopt;
  Line out;
  std::string body = s.substr(1, star - 1), cur;
  bool first = true;
  for (char ch : body + ",") {
    if (ch != ',') { cur += ch; continue; }
    if (first) out.type = cur; else out.fields.push_back(cur);
    first = false;
    cur.clear();
  }
  if (out.type.empty()) return std::nullopt;
  return out;
}

// 초음파 줄: $US,<번호>,<mm>,<ms>. mm 0 = 메아리 없음
struct Sonar {
  int index;
  uint32_t mm;
  uint32_t ms;
};

inline std::optional<Sonar> as_sonar(const Line& l) {
  if (l.type != "US" || l.fields.size() != 3) return std::nullopt;
  try {
    size_t p = 0;
    Sonar s{};
    s.index = std::stoi(l.fields[0], &p);
    if (p != l.fields[0].size() || s.index < 0) return std::nullopt;
    s.mm = static_cast<uint32_t>(std::stoul(l.fields[1], &p));
    if (p != l.fields[1].size()) return std::nullopt;
    s.ms = static_cast<uint32_t>(std::stoul(l.fields[2], &p));
    if (p != l.fields[2].size()) return std::nullopt;
    return s;
  } catch (...) {
    return std::nullopt;
  }
}

// 열화상 줄: $TH,<ms>,<ta>,<t1>…<t768> — 모두 0.1 °C 정수(ta = 모듈 온도), 점은 32×24 행 먼저
constexpr int kThermalCols = 32, kThermalRows = 24, kThermalPoints = kThermalCols * kThermalRows;
struct Thermal {
  uint32_t ms;
  double ambient_c;
  std::vector<float> c;   // °C, 768 개
};

inline std::optional<int> parse_int(const std::string& f) {
  try {
    size_t p = 0;
    const int v = std::stoi(f, &p);
    if (p != f.size()) return std::nullopt;
    return v;
  } catch (...) {
    return std::nullopt;
  }
}

inline std::optional<Thermal> as_thermal(const Line& l) {
  if (l.type != "TH" || l.fields.size() != static_cast<size_t>(2 + kThermalPoints)) return std::nullopt;
  const auto ms = parse_int(l.fields[0]), ta = parse_int(l.fields[1]);
  if (!ms || *ms < 0 || !ta) return std::nullopt;
  Thermal t;
  t.ms = static_cast<uint32_t>(*ms);
  t.ambient_c = *ta * 0.1;
  t.c.reserve(kThermalPoints);
  for (int i = 0; i < kThermalPoints; ++i) {
    const auto v = parse_int(l.fields[2 + i]);
    if (!v) return std::nullopt;
    t.c.push_back(static_cast<float>(*v * 0.1));
  }
  return t;
}

// 레이더 줄: $RD,<ms>,x1,y1,v1,x2,y2,v2,x3,y3,v3 — x 오른쪽 + · y 앞 (mm), v cm/s(LD2450 원본 부호), 표적 없으면 0,0,0
constexpr int kRadarTargets = 3;
struct RadarTarget {
  double x_mm, y_mm, v_cms;
};
struct Radar {
  uint32_t ms;
  std::vector<RadarTarget> targets;   // 있는 표적만(없는 칸 0,0 은 뺀다)
};

inline std::optional<Radar> as_radar(const Line& l) {
  if (l.type != "RD" || l.fields.size() != static_cast<size_t>(1 + 3 * kRadarTargets)) return std::nullopt;
  const auto ms = parse_int(l.fields[0]);
  if (!ms || *ms < 0) return std::nullopt;
  Radar r;
  r.ms = static_cast<uint32_t>(*ms);
  for (int t = 0; t < kRadarTargets; ++t) {
    const auto x = parse_int(l.fields[1 + 3 * t]), y = parse_int(l.fields[2 + 3 * t]), v = parse_int(l.fields[3 + 3 * t]);
    if (!x || !y || !v) return std::nullopt;
    if (*x == 0 && *y == 0) continue;
    r.targets.push_back({static_cast<double>(*x), static_cast<double>(*y), static_cast<double>(*v)});
  }
  return r;
}

// 발사대 응답 줄: $LA,<몸통>  — 펌웨어 include/launcher.h 머리 주석과 같다
//   boot · on · off · cocked · fired · home · lever,<각> · state,<각>,<on|off>,<busy 0|1> · blocked,<까닭> · error,<까닭>
// 각은 레버 서보 명령 각(도, 0~180). 서보가 실제로 간 각을 잰 값이 아니다.
struct LauncherReply {
  enum Kind { kBoot, kOn, kOff, kCocked, kFired, kHome, kLever, kState, kBlocked, kError };
  explicit LauncherReply(Kind k = kBoot) : kind(k) {}
  Kind kind;
  double lever_deg = 0.0;
  bool on = false, busy = false;
  std::string detail;   // blocked·error 까닭 (off · busy · bad · attach)
};

inline std::optional<double> parse_double(const std::string& f) {
  try {
    size_t p = 0;
    const double v = std::stod(f, &p);
    if (p != f.size() || !std::isfinite(v)) return std::nullopt;
    return v;
  } catch (...) {
    return std::nullopt;
  }
}

inline std::optional<LauncherReply> as_launcher(const Line& l) {
  if (l.type != "LA" || l.fields.empty()) return std::nullopt;
  const auto& f = l.fields;
  const std::string& k = f[0];
  LauncherReply r{LauncherReply::kBoot};
  if (f.size() == 1) {
    if (k == "boot") r.kind = LauncherReply::kBoot;
    else if (k == "on") r.kind = LauncherReply::kOn;
    else if (k == "off") r.kind = LauncherReply::kOff;
    else if (k == "cocked") r.kind = LauncherReply::kCocked;
    else if (k == "fired") r.kind = LauncherReply::kFired;
    else if (k == "home") r.kind = LauncherReply::kHome;
    else return std::nullopt;
    return r;
  }
  if ((k == "blocked" || k == "error") && f.size() == 2 && !f[1].empty()) {
    r.kind = k == "blocked" ? LauncherReply::kBlocked : LauncherReply::kError;
    r.detail = f[1];
    return r;
  }
  if (k == "lever" && f.size() == 2) {
    const auto d = parse_double(f[1]);
    if (!d || *d < 0.0 || *d > 180.0) return std::nullopt;
    r.kind = LauncherReply::kLever;
    r.lever_deg = *d;
    return r;
  }
  if (k == "state" && f.size() == 4) {
    const auto d = parse_double(f[1]);
    if (!d || *d < 0.0 || *d > 180.0 || (f[2] != "on" && f[2] != "off") || (f[3] != "0" && f[3] != "1")) return std::nullopt;
    r.kind = LauncherReply::kState;
    r.lever_deg = *d;
    r.on = f[2] == "on";
    r.busy = f[3] == "1";
    return r;
  }
  return std::nullopt;
}

}  // namespace limo_fire_esp32_bridge
