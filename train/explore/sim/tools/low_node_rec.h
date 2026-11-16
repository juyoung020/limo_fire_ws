// 실물 아래층 관측 녹화 꼴 (low_node_check --dump 가 쓰고 low_node_replay 가 읽는다): RobotEnv 아래층 모드가 받은 입력을 부른 차례대로 + 기대 관측·명령.
// 머리: 매직 · 구조체 크기(EnvHostCfg · Proprio · Window — 같은 소스 판인지) · 그림 바이트 · 1D 칸 · EnvHostCfg 바이트. 그 뒤 사건(한 바이트 종류 + 몸):
//   'R' 리셋 (x0 y0 th0 · Proprio · fov · Window)  'G' 목표 칸 (int32)  'A' 정책 출력 a[ACT] + 기대 명령 v w
//   'S' 걸음 (Proprio · 빔 수 · 거리 · 세기 수 · 세기 · angle_min angle_increment time_increment x y th w n_rev scan_time)
//   'D' 깊이 바이트 (DEPTH_SIZE)   'O' 기대 아래층 관측 (그림 · 1D) — 읽는 쪽은 여기서 low_obs 를 부르고 견줌
#pragma once
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

#include "limo_fire_rl/robot_env.h"

namespace lowrec {
constexpr int32_t kMagic = 0x4E574F4C;   // "LOWN"
using Window = limo_fire_rl::RobotEnv::Window;

class Writer {
 public:
  Writer(const std::string& path, const sim::EnvHostCfg& hc, int ni, int n1) : f_(std::fopen(path.c_str(), "wb")), ni_(ni), n1_(n1) {
    if (!f_) throw std::runtime_error("녹화 파일을 못 엶: " + path);
    const int32_t hd[6] = {kMagic, (int32_t)sizeof(sim::EnvHostCfg), (int32_t)sizeof(limo_fire_rl::Proprio), (int32_t)sizeof(Window), ni, n1};
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
  void goal(int32_t cell) { tag('G'); w(&cell, 4); }
  void act(const float* a, float v, float wv) {
    tag('A');
    w(a, 4 * mdp::low::ACT);
    w(&v, 4);
    w(&wv, 4);
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
  void obs(const uint8_t* img, const float* o1) {
    tag('O');
    w(img, (size_t)ni_);
    w(o1, 4 * (size_t)n1_);
  }

 private:
  FILE* f_;
  int ni_, n1_;
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

class Reader {
 public:
  explicit Reader(const std::string& path) : f_(std::fopen(path.c_str(), "rb")) {
    if (!f_) throw std::runtime_error("녹화 파일을 못 엶: " + path);
    int32_t hd[6];
    r(hd, sizeof(hd));
    if (hd[0] != kMagic) throw std::runtime_error("녹화 머리가 아님");
    // 옛 판도 받는다: EnvHostCfg 에 칸이 더해진 차례대로 (새 구조체 안 자리, 바이트). 녹화 크기가 "지금 − 뒤에서 j 개 합" 이면 그 j 개 자리에 0 을 끼움 (0 = 그 기능 끔 = 그때 동작)
    //   1) 2a17a5a 깊이 영상 늦음 범위 DepthModel::lat_lo·lat_hi (8) — 1116 실물 녹화들   2) 구동 달리며 돌 때 ω 몫 DriveModel::n_tg·tg_v·tg_g
    const size_t ins_off[2] = {offsetof(sim::EnvHostCfg, cfg) + offsetof(sim::EnvCfg, depth) + offsetof(sim::DepthModel, lat_lo),
                               offsetof(sim::EnvHostCfg, cfg) + offsetof(sim::EnvCfg, drive) + offsetof(sim::DriveModel, n_tg)};
    const size_t ins_len[2] = {8, sizeof(int32_t) + 2 * sizeof(float) * sim::kTurnGainPts};
    int miss = -1;   // 빠진 뒤쪽 칸 수
    for (int j = 0, sum = 0; j <= 2; ++j) {
      if (hd[1] + sum == (int32_t)sizeof(sim::EnvHostCfg)) { miss = j; break; }
      if (j < 2) sum += (int)ins_len[1 - j];
    }
    if (miss < 0 || hd[2] != (int32_t)sizeof(limo_fire_rl::Proprio) || hd[3] != (int32_t)sizeof(Window))
      throw std::runtime_error("구조체 크기가 녹화와 다름 (소스 판이 다름?)");
    ni = hd[4];
    n1 = hd[5];
    if (miss > 0) {
      std::vector<uint8_t> raw((size_t)hd[1]);
      r(raw.data(), raw.size());
      // 새 구조체를 앞에서부터 채우며, 빠진 칸(뒤쪽 miss 개 — 자리 순서로)이 오면 0
      std::vector<std::pair<size_t, size_t>> gaps;
      for (int j = 2 - miss; j < 2; ++j) gaps.emplace_back(ins_off[j], ins_len[j]);
      std::sort(gaps.begin(), gaps.end());
      uint8_t* dst = reinterpret_cast<uint8_t*>(&hc);
      size_t src = 0, at = 0;
      for (const auto& g : gaps) {
        std::memcpy(dst + at, raw.data() + src, g.first - at);
        src += g.first - at;
        std::memset(dst + g.first, 0, g.second);
        at = g.first + g.second;
      }
      std::memcpy(dst + at, raw.data() + src, sizeof(hc) - at);
      old_layout = true;
    } else {
      r(&hc, sizeof(hc));
    }
  }
  ~Reader() { if (f_) std::fclose(f_); }
  sim::EnvHostCfg hc{};
  bool old_layout = false;   // 깊이 영상 늦음 칸이 없던 옛 판 녹화 (위)
  int ni = 0, n1 = 0;
  // 다음 사건 종류 (끝이면 0)
  char next() {
    char c;
    return std::fread(&c, 1, 1, f_) == 1 ? c : 0;
  }
  void reset(float* xyz, limo_fire_rl::Proprio* p, float* fov, Window* win) {
    r(xyz, 12);
    r(p, sizeof(*p));
    r(fov, 4);
    r(win, sizeof(*win));
  }
  int32_t goal() { int32_t c; r(&c, 4); return c; }
  void act(float* a, float* v, float* wv) {
    r(a, 4 * mdp::low::ACT);
    r(v, 4);
    r(wv, 4);
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
  void obs(uint8_t* img, float* o1) {
    r(img, (size_t)ni);
    r(o1, 4 * (size_t)n1);
  }

 private:
  FILE* f_;
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
