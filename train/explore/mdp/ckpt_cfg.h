// 체크포인트 설정 꼬리 (2026-11-16) — 형식 3 체크포인트(ppo.cu save/load) 끝에 붙는 "KEY=VALUE" 줄들. 어떤 열쇠를 적는지는 sim/include/low_ckpt_cfg.h.
// 꼴: 형식 3 내용 · "KEY=VALUE\n" 줄들 · [줄 바이트 수 u32] · [매직 'XCFG' u32]. 형식 3 을 읽는 쪽(PPO::load · network/policy_cpu.h · ppo/tools/ckpt_*)은
// 고정 꼬리(반복·학습률·호출 수·씨앗)까지만 읽으므로 이 꼬리를 무시한다 — 옛 바이너리도 새 체크포인트를 읽음. 꼬리 없는 옛 체크포인트는 그대로 환경변수로.
// apply_ckpt_cfg: 꼬리의 SIM_* 를 환경변수로 넣는다 (load_env_cfg 앞에). 이미 준 환경변수가 다른 값이면 멈춘다 (명시한 것이 체크포인트와 어긋남).
// 체크포인트를 바꾸는 도구(ppo/tools/ckpt_add_*)는 꼬리를 읽어 바뀐 열쇠만 고쳐 다시 붙인다 (꼬리 없는 입력은 꼬리 없이).
#pragma once
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace ckcfg {

constexpr uint32_t kMagic = 0x47464358;   // "XCFG"
using Kv = std::vector<std::pair<std::string, std::string>>;

// 꼬리 바이트 (PPO::save 의 extra)
inline std::string trailer(const Kv& kv) {
  std::string t;
  for (const auto& [k, v] : kv) t += k + "=" + v + "\n";
  const uint32_t n = (uint32_t)t.size(), mg = kMagic;
  t.append(reinterpret_cast<const char*>(&n), 4);
  t.append(reinterpret_cast<const char*>(&mg), 4);
  return t;
}

// 꼬리 읽기 (없으면 빈 것). *trailer_bytes: 꼬리 전체 바이트 수 (없으면 0 — 앞 내용만 견줄 때)
inline Kv read(const std::string& path, size_t* trailer_bytes = nullptr) {
  if (trailer_bytes) *trailer_bytes = 0;
  std::ifstream f(path, std::ios::binary | std::ios::ate);
  if (!f) throw std::runtime_error("체크포인트 없음: " + path);
  const std::streamoff size = f.tellg();
  if (size < 8) return {};
  uint32_t tail[2];
  f.seekg(size - 8);
  f.read(reinterpret_cast<char*>(tail), 8);
  if (!f || tail[1] != kMagic || (std::streamoff)tail[0] > size - 8) return {};
  std::string t(tail[0], '\0');
  f.seekg(size - 8 - (std::streamoff)tail[0]);
  f.read(&t[0], tail[0]);
  if (!f) return {};
  Kv kv;
  size_t p = 0;
  while (p < t.size()) {
    size_t e = t.find('\n', p);
    if (e == std::string::npos) e = t.size();
    const std::string line = t.substr(p, e - p);
    const size_t q = line.find('=');
    if (q != std::string::npos) kv.push_back({line.substr(0, q), line.substr(q + 1)});
    p = e + 1;
  }
  if (trailer_bytes) *trailer_bytes = (size_t)tail[0] + 8;
  return kv;
}

inline std::string get(const Kv& kv, const std::string& k, const std::string& d = "") {
  for (const auto& [a, b] : kv)
    if (a == k) return b;
  return d;
}

// 체크포인트 꼬리의 SIM_* 를 환경변수로 (load_env_cfg 앞에). 이미 준 환경변수가 다른 값이면 멈춤 (같은 값 = env_cfg.cpp 가 읽는 float 로 같음).
// 반환: 꼬리 (없으면 빈 것 — 옛 체크포인트, 환경변수 그대로). log: 무엇을 했는지 한 줄 (도구가 찍음)
inline Kv apply(const std::string& path, std::string* log = nullptr) {
  const Kv kv = read(path);
  if (kv.empty()) {
    if (log) *log = "체크포인트에 관측 설정 꼬리 없음 (옛 체크포인트) — 환경변수 SIM_LOW_* 그대로";
    return kv;
  }
  std::string set, same;
  for (const auto& [k, v] : kv) {
    if (k.rfind("SIM_", 0) != 0) continue;
    const char* e = std::getenv(k.c_str());
    if (e && e[0]) {
      if ((float)std::atof(e) != (float)std::atof(v.c_str()))
        throw std::runtime_error("환경변수 " + k + "=" + e + " 가 체크포인트(" + path + ")의 학습 설정 " + k + "=" + v + " 와 다름 — 환경변수를 지우면 체크포인트 값을 씀");
      same += " " + k;
      continue;
    }
    setenv(k.c_str(), v.c_str(), 1);
    set += " " + k + "=" + v;
  }
  if (log) *log = "체크포인트 관측 설정을 씀:" + (set.empty() ? std::string(" (모두 환경변수와 같음)") : set) + (same.empty() ? "" : " · 환경변수와 같음:" + same);
  return kv;
}


inline void set(Kv& kv, const std::string& k, const std::string& v) {
  for (auto& [a, b] : kv)
    if (a == k) { b = v; return; }
  kv.push_back({k, v});
}

}  // namespace ckcfg
