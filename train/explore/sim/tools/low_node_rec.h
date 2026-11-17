// 실물 합친 탐사 노드 녹화 꼴 (limo_fire_rl low_rl record:= · 시뮬 room_run --rec 가 쓰고, low_explore_replay · node_stats 가 읽는다):
// RobotEnv 가 받은 입력을 부른 차례대로 + 그때 낸 명령. 노드 차례를 그대로 다시 돌려 위층 결정·명령이 같은지 본다.
//
// 판 5 (2026-11-17 —): 판 4 와 같고 EnvHostCfg 에 EnvCfg::face_tol·face_w_min(LowCmd::face 로 돌아 서기) 칸. 판 4 이하는 그 칸에 0 (그때는 face 를 안 씀).
// 판 4 (2026-11-17): 판 3 과 같고 EnvHostCfg 에 EnvCfg::cam_lw·cam_lh(깊이 층 전용 격자) 칸, 사건 'L' (층 전용 깊이 바이트 cam_lw × cam_lh — 'D' 바로 뒤, 격자가 있을 때만).
//   판 3 · 2 · 1 은 그 칸에 0 (= 층은 관측 깊이 그대로, 그때 동작)을 끼워 읽는다.
// 판 3 (2026-11-17): 판 2 와 같고 EnvHostCfg 에 EnvCfg::cam_every(깊이 층 걸음마다 갱신) 칸이 더해짐. 판 2 녹화는 그 칸에 0 (= 그때 동작)을 끼워 읽는다.
// 판 2 (2026-11-17): 머리 = 매직 "LOWN" · 판 번호 2 · 구조체 크기(EnvHostCfg · Proprio · Window) · EnvHostCfg 바이트. 그 뒤 사건(한 바이트 종류 + 몸):
//   'R' 리셋 (x0 y0 th0 · Proprio · fov · Window)   'S' 걸음 (Proprio · 빔 수 · 거리 · 세기 수 · 세기 · angle_min angle_increment time_increment x y th w n_rev scan_time)
//   'D' 깊이 바이트 (DEPTH_SIZE)   'A' 낸 명령 v w · 그때 아래층 목적지 (세계 x y, 없으면 NaN — 판단 모델 위층처럼 다시 못 내는 목표를 남김)
//   — 읽는 쪽은 여기서 set_slot · low_prepare · low_act 를 부르고 견줌
//   EnvHostCfg 를 바이트 그대로 담으므로 그 구조체를 바꾸면 판 번호를 올리고, 옛 판을 새 구조체로 옮기는 길을 Reader 에 더한다.
// 판 1 (— 2026-11-17, 판 번호 칸 없음 — 머리 둘째 칸이 EnvHostCfg 크기): 그만둔 아래층 정책 관측 'O' · 정책 목표 'G' 사건과 'A' 의 정책 출력 3 칸이 더 있다.
//   Reader 가 읽으며 건너뛰고, EnvHostCfg 는 그때 판(e5a7c74 의 1852 바이트 꼴)에서 지운 칸 자리를 빼 옮긴다 (아래 kV1Gone).
//   그보다 옛 판 1 (깊이 늦음 칸 · 달리며 돌 때 ω 몫 칸이 없던 것)도 그 칸에 0 을 넣어 읽는다. 판 1 을 그때 코드로 읽으려면 커밋 e5a7c74.
#pragma once
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "limo_fire_rl/robot_env.h"

namespace lowrec {
constexpr int32_t kMagic = 0x4E574F4C;   // "LOWN"
constexpr int32_t kVersion = 5;
// 판 3 · 4 에서 더한 칸 (지금 구조체 안 자리, 바이트) — 옛 판은 이 자리에 0 을 끼움
constexpr size_t kV3Off = offsetof(sim::EnvHostCfg, cfg) + offsetof(sim::EnvCfg, cam_every), kV3Len = sizeof(int32_t);
constexpr size_t kV4Off = offsetof(sim::EnvHostCfg, cfg) + offsetof(sim::EnvCfg, cam_lw), kV4Len = 2 * sizeof(int32_t);
constexpr size_t kV5Off = offsetof(sim::EnvHostCfg, cfg) + offsetof(sim::EnvCfg, face_tol), kV5Len = 2 * sizeof(float);
constexpr size_t kV4Size = sizeof(sim::EnvHostCfg) - kV5Len;
constexpr size_t kV3Size = kV4Size - kV4Len;
constexpr size_t kV2Size = kV3Size - kV3Len;
using Window = limo_fire_rl::RobotEnv::Window;
// 'A' 사건의 아래층 목적지: 0 없음 · 1 세계 좌표 (x, y) · 2 판 1 녹화 — 지도 자세 기준 (x = 거리 m, y = 방향 rad, 그 걸음 지도 자세에서 셈)
struct Goal { int kind = 0; float x = 0.f, y = 0.f; };

class Writer {
 public:
  Writer(const std::string& path, const sim::EnvHostCfg& hc) : f_(std::fopen(path.c_str(), "wb")) {
    if (!f_) throw std::runtime_error("녹화 파일을 못 엶: " + path);
    const int32_t hd[5] = {kMagic, kVersion, (int32_t)sizeof(sim::EnvHostCfg), (int32_t)sizeof(limo_fire_rl::Proprio), (int32_t)sizeof(Window)};
    w(hd, sizeof(hd));
    w(&hc, sizeof(hc));
  }
  ~Writer() { if (f_) std::fclose(f_); }
  void reset(float x0, float y0, float th0, const limo_fire_rl::Proprio& p, float fov, const Window& win) {
    tag('R');
    const float a[3] = {x0, y0, th0};
    w(a, sizeof(a));
    w(&p, sizeof(p));
    w(&fov, 4);
    w(&win, sizeof(win));
  }
  void act(float v, float wv, float gx, float gy) {
    tag('A');
    const float a[4] = {v, wv, gx, gy};
    w(a, sizeof(a));
  }
  void step(const limo_fire_rl::Proprio& p, const limo_fire_rl::Scan& s) {
    tag('S');
    w(&p, sizeof(p));
    vec(s.ranges);
    vec(s.intensities);
    const float f[9] = {s.angle_min, s.angle_increment, s.time_increment, s.x, s.y, s.th, s.w, s.n_rev, s.scan_time};
    w(f, sizeof(f));
  }
  void depth(const uint8_t* d) { tag('D'); w(d, (size_t)mdp::obs::DEPTH_SIZE); }
  void layer(const uint8_t* d, size_t n) { tag('L'); w(d, n); }   // 깊이 층 전용 격자 (cam_lw × cam_lh, 'D' 바로 뒤)

 private:
  FILE* f_;
  void w(const void* p, size_t n) {
    if (n && std::fwrite(p, 1, n, f_) != n) throw std::runtime_error("녹화 쓰기 실패");
  }
  void tag(char c) { w(&c, 1); }
  void vec(const std::vector<float>& v) {
    const int32_t n = (int32_t)v.size();
    w(&n, 4);
    w(v.data(), 4 * v.size());
  }
};

// 판 1 의 EnvHostCfg (e5a7c74 꼴, 1852 바이트) 에서 지운 칸 [시작, 끝) 바이트 — 그만둔 아래층 RL 칸(gone_rl_*) · 커리큘럼 · 그래프 모드 ·
// 단독 과제의 목표 수 · 시간 · 사람 방위 칸. 나머지 칸은 차례·크기 그대로라 이 자리만 빼면 지금 구조체가 된다
constexpr int kV1Size = 1852;
constexpr std::pair<int, int> kV1Gone[] = {{796, 920}, {984, 1020}, {1708, 1780}, {1788, 1792}, {1796, 1800}, {1808, 1812}, {1816, 1824}, {1832, 1836}};
constexpr int v1_kept() {
  int n = kV1Size;
  for (const auto& g : kV1Gone) n -= g.second - g.first;
  return n;
}
// 지금 구조체 안 자리 → 판 1 꼴(1852 바이트) 안 자리
constexpr size_t v1_off(size_t cur) {
  size_t o = cur;
  for (const auto& g : kV1Gone)
    if (o >= (size_t)g.first) o += (size_t)(g.second - g.first);
  return o;
}
static_assert(v1_kept() == (int)kV2Size,
              "EnvHostCfg 가 바뀜 — 녹화 판 번호(kVersion)를 올리고 판 1 · 판 2 를 새 구조체로 옮기는 길을 Reader 에 더할 것");

class Reader {
 public:
  explicit Reader(const std::string& path) : f_(std::fopen(path.c_str(), "rb")) {
    if (!f_) throw std::runtime_error("녹화 파일을 못 엶: " + path);
    int32_t hd[2];
    r(hd, sizeof(hd));
    if (hd[0] != kMagic) throw std::runtime_error("녹화 머리가 아님");
    if (hd[1] >= 2 && hd[1] <= kVersion) {
      int32_t sz[3];
      r(sz, sizeof(sz));
      const size_t want = hd[1] == kVersion ? sizeof(sim::EnvHostCfg) : hd[1] == 4 ? kV4Size : hd[1] == 3 ? kV3Size : kV2Size;
      if (sz[0] != (int32_t)want || sz[1] != (int32_t)sizeof(limo_fire_rl::Proprio) || sz[2] != (int32_t)sizeof(Window))
        throw std::runtime_error("구조체 크기가 녹화와 다름 (소스 판이 다름?)");
      version = hd[1];
      std::vector<uint8_t> raw(want);
      r(raw.data(), raw.size());
      from_old(raw.data(), version);
      return;
    }
    if (hd[1] < 100) throw std::runtime_error("모르는 녹화 판 " + std::to_string(hd[1]));
    read_v1(hd[1]);
  }
  ~Reader() { if (f_) std::fclose(f_); }
  sim::EnvHostCfg hc{};
  int version = kVersion;
  bool truncated = false;   // next() 가 건너뛰는 사건(판 1 'O' · 'G')에서 녹화가 끝남
  // 다음 사건 종류 'R' · 'S' · 'D' · 'L'(판 4) · 'A' (끝이면 0). 판 1 의 'O' · 'G' 는 건너뜀
  char next() {
    for (;;) {
      char c;
      if (std::fread(&c, 1, 1, f_) != 1) return 0;
      try {
        if (version == 1 && c == 'O') { read_v1_obs(); continue; }
        if (version == 1 && c == 'G') { skip(4); continue; }
      } catch (const std::runtime_error&) {   // 끝 사건이 잘림 (노드를 끌 때)
        truncated = true;
        return 0;
      }
      return c;
    }
  }
  void reset(float* xyz, limo_fire_rl::Proprio* p, float* fov, Window* win) {
    r(xyz, 12);
    r(p, sizeof(*p));
    r(fov, 4);
    r(win, sizeof(*win));
  }
  void act(float* v, float* wv, Goal* g) {
    if (version == 1) {
      skip(4 * 3);   // 그만둔 정책 출력 (v, ω, 선언 자리)
      r(v, 4);
      r(wv, 4);
      *g = goal1_;
      return;
    }
    float a[4];
    r(a, sizeof(a));
    *v = a[0];
    *wv = a[1];
    *g = a[2] == a[2] ? Goal{1, a[2], a[3]} : Goal{};
  }
  void step(limo_fire_rl::Proprio* p, limo_fire_rl::Scan* s) {
    r(p, sizeof(*p));
    vec(s->ranges);
    vec(s->intensities);
    float f[9];
    r(f, sizeof(f));
    s->angle_min = f[0];
    s->angle_increment = f[1];
    s->time_increment = f[2];
    s->x = f[3];
    s->y = f[4];
    s->th = f[5];
    s->w = f[6];
    s->n_rev = f[7];
    s->scan_time = f[8];
  }
  void depth(uint8_t* d) { r(d, (size_t)mdp::obs::DEPTH_SIZE); }
  size_t layer_n() const { return hc.cfg.cam_lw > 0 ? (size_t)sim::cam_npix(hc.cfg) : 0; }
  void layer(uint8_t* d) { r(d, layer_n()); }   // 'L' 사건 몸 (판 4)

 private:
  FILE* f_;
  int ni1_ = 0, n11_ = 0;   // 판 1: 'O' 사건의 그림 바이트 · 1D 칸 수
  Goal goal1_;              // 판 1: 마지막 'O' 의 목적지 (1D 칸 24 있음 · 11 거리 / 15 m · 12 sin · 13 cos — 그때 아래층 관측 꼴)
  void read_v1_obs() {
    skip((size_t)ni1_);
    std::vector<float> o((size_t)n11_);
    r(o.data(), 4 * o.size());
    goal1_ = n11_ > 24 && o[24] > 0.5f ? Goal{2, o[11] * 15.f, std::atan2(o[12], o[13])} : Goal{};
  }
  // 판 1 머리 (둘째 칸부터): EnvHostCfg 크기 · Proprio · Window 크기 · 그림 · 1D, 그 뒤 EnvHostCfg 바이트 (그때 크기)
  void read_v1(int32_t cfg_size) {
    version = 1;
    int32_t hd[4];
    r(hd, sizeof(hd));
    if (hd[0] != (int32_t)sizeof(limo_fire_rl::Proprio) || hd[1] != (int32_t)sizeof(Window)) throw std::runtime_error("구조체 크기가 녹화와 다름 (소스 판이 다름?)");
    ni1_ = hd[2];
    n11_ = hd[3];
    // 더 옛 판 1: 칸이 더해진 차례대로 (1852 바이트 꼴 안 자리, 바이트). 크기가 "1852 − 뒤에서 j 개 합" 이면 그 j 개 자리에 0 을 끼움 (0 = 그 기능 끔 = 그때 동작)
    //   1) 2a17a5a 깊이 영상 늦음 범위 DepthModel::lat_lo·lat_hi (8) — 1116 실물 녹화들   2) 구동 달리며 돌 때 ω 몫 DriveModel::n_tg·tg_v·tg_g
    const std::pair<size_t, size_t> ins[2] = {{v1_off(offsetof(sim::EnvHostCfg, cfg) + offsetof(sim::EnvCfg, depth) + offsetof(sim::DepthModel, lat_lo)), 8},
                                              {v1_off(offsetof(sim::EnvHostCfg, cfg) + offsetof(sim::EnvCfg, drive) + offsetof(sim::DriveModel, n_tg)),
                                               sizeof(int32_t) + 2 * sizeof(float) * sim::kTurnGainPts}};
    int miss = -1;
    for (int j = 0, sum = 0; j <= 2; ++j) {
      if (cfg_size + sum == kV1Size) { miss = j; break; }
      if (j < 2) sum += (int)ins[1 - j].second;
    }
    if (miss < 0) throw std::runtime_error("녹화 판 1 의 EnvHostCfg 크기를 모름 (" + std::to_string(cfg_size) + ")");
    std::vector<uint8_t> raw((size_t)cfg_size), full(kV1Size, 0);
    r(raw.data(), raw.size());
    {   // 빠진 칸(뒤쪽 miss 개)을 0 으로 끼워 1852 바이트 꼴로
      std::vector<std::pair<size_t, size_t>> gaps(ins + 2 - miss, ins + 2);
      std::sort(gaps.begin(), gaps.end());
      size_t src = 0, at = 0;
      for (const auto& g : gaps) {
        std::memcpy(full.data() + at, raw.data() + src, g.first - at);
        src += g.first - at;
        at = g.first + g.second;
      }
      std::memcpy(full.data() + at, raw.data() + src, kV1Size - at);
    }
    std::vector<uint8_t> v2(kV2Size);   // 지운 칸 자리를 빼고 차례대로 → 판 2 꼴 → 판 3 칸 끼움
    uint8_t* dst = v2.data();
    size_t at = 0, out = 0;
    for (const auto& g : kV1Gone) {
      std::memcpy(dst + out, full.data() + at, (size_t)g.first - at);
      out += (size_t)g.first - at;
      at = (size_t)g.second;
    }
    std::memcpy(dst + out, full.data() + at, kV1Size - at);
    from_old(v2.data(), 2);
  }
  // 판 ver 꼴 EnvHostCfg 바이트 → 지금 구조체 (그 뒤 판에서 더한 칸 자리에 0)
  void from_old(const uint8_t* src, int ver) {
    std::vector<std::pair<size_t, size_t>> gaps;
    if (ver < 3) gaps.emplace_back(kV3Off, kV3Len);
    if (ver < 4) gaps.emplace_back(kV4Off, kV4Len);
    if (ver < 5) gaps.emplace_back(kV5Off, kV5Len);
    std::sort(gaps.begin(), gaps.end());
    uint8_t* d = reinterpret_cast<uint8_t*>(&hc);
    size_t at = 0, from = 0;
    for (const auto& g : gaps) {
      std::memcpy(d + at, src + from, g.first - at);
      from += g.first - at;
      std::memset(d + g.first, 0, g.second);
      at = g.first + g.second;
    }
    std::memcpy(d + at, src + from, sizeof(hc) - at);
  }
  void skip(size_t n) {
    if (n && std::fseek(f_, (long)n, SEEK_CUR) != 0) throw std::runtime_error("녹화가 짧음");
  }
  void r(void* p, size_t n) {
    if (n && std::fread(p, 1, n, f_) != n) throw std::runtime_error("녹화가 짧음");
  }
  void vec(std::vector<float>& v) {
    int32_t n;
    r(&n, 4);
    if (n < 0 || n > (1 << 20)) throw std::runtime_error("녹화 빔 수");
    v.resize((size_t)n);
    r(v.data(), 4 * (size_t)n);
  }
};
}  // namespace lowrec
