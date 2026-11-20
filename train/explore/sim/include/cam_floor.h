// 깊이 카메라 숙임·높이를 바닥 면으로 재기 (호스트 전용). 실물 노드(explorer — 설정과 어긋나면 알림)와 녹화 도구(low_explore_replay --floor-fit)가 같은 함수.
// 까닭 (2026-11-20 실물, limo_upper_keep/reports/real_spin.md): 카메라 마운트는 손으로 고정이라 숙임이 바뀐다(1104 5.2° · 1116 5.0~6.3° · 1120 7.2°).
//   설정(env.yaml body.camera_pitch)보다 더 숙이면 먼 바닥이 높게 계산돼 깊이 층 띠 아래 끝(min_obstacle_height 0.03 m)을 넘고, 로봇 둘레 0.9 m 밖 바닥이
//   장애물 고리로 찍혀 갈 수 있는 칸이 로봇 곁에 갇힘 → 지도 닫힘·후보 0 → 제자리 회전 · '대충 다 앎' 끝.
// 방법: 관측 깊이(64 × 40 바이트)를 지금 설정으로 base_link 에 놓아 높이 ±8 cm · 센서 수평 2 m 안 점을 바닥 후보로 모으고,
//   바닥 높이 0 = H − z (sin p + cos p (v − cy) / fy) 를 (H, p) 최소제곱 (p 를 거친 격자 → 고운 격자, p 마다 H 는 닫힌 꼴), 잔차 1.5 cm 밖 점을 버리고 두 번 더.
//   옆으로 기욺(roll)은 안 봄 — 마운트가 앞뒤로만 숙음. 바닥이 아닌 낮은 물체가 많으면 잔차가 커짐 (rms 로 봄).
#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

#include "env.h"

namespace sim {

class FloorFit {
 public:
  struct Res {
    bool ok = false;
    double pitch = 0, h = 0, rms = 0;   // rad · m · m
    long n = 0, n_all = 0;              // 맞춤에 쓴 점 · 모은 점
  };
  explicit FloorFit(size_t cap = 200000) : cap_(cap) {}
  // 관측 깊이 한 장 (mdp::obs::DEPTH_SIZE 바이트) 의 바닥 후보 점을 모음. 가득 차면 가장 오래된 것부터 덮음
  void add(const EnvCfg& c, const uint8_t* dep) {
    const DepthSpec& D = c.depth.spec;
    const float sv = (float)D.h / (float)mdp::obs::DEPTH_H;
    for (int i = 0; i < mdp::obs::DEPTH_SIZE; ++i) {
      float bx, by, bz;
      if (!cam_point_base(c, dep[i], i, &bx, &by, &bz) || std::fabs(bz) > 0.08f || std::hypot(bx - c.cam_x, by - c.cam_y) > 2.0f) continue;
      const float v = ((float)(i / mdp::obs::DEPTH_W) + 0.5f) * sv - 0.5f;
      const float z = D.min_z + ((float)(dep[i] - 1) + 0.5f) * ((D.obs_max_z - D.min_z) / 254.f);
      if (pts_.size() < cap_) pts_.push_back({v, z});
      else pts_[head_++ % cap_] = {v, z};
    }
  }
  size_t size() const { return pts_.size(); }
  void clear() { pts_.clear(); head_ = 0; }
  Res fit(const EnvCfg& c) const {
    const DepthSpec& D = c.depth.spec;
    Res r;
    r.n_all = (long)pts_.size();
    std::vector<char> use(pts_.size(), 1);
    const auto hgt = [&](size_t j, double sp, double cp) { return pts_[j].z * (sp + cp * (pts_[j].v - D.cy) / D.fy); };   // = H (바닥이면)
    const auto at = [&](double p, double* H, double* var, long* m) {
      const double sp = std::sin(p), cp = std::cos(p);
      double s1 = 0, s2 = 0;
      long k = 0;
      for (size_t j = 0; j < pts_.size(); ++j) {
        if (!use[j]) continue;
        const double h = hgt(j, sp, cp);
        s1 += h; s2 += h * h; ++k;
      }
      *m = k;
      if (k == 0) return false;
      *H = s1 / k;
      *var = std::max(0.0, s2 / k - (*H) * (*H));
      return true;
    };
    const auto search = [&](double lo, double hi, double step, double* P, double* H, double* var, long* m) {
      double best = 1e300;
      for (double p = lo; p <= hi + 1e-12; p += step) {
        double h, v;
        long k;
        if (at(p, &h, &v, &k) && v < best) { best = v; *P = p; *H = h; *var = v; *m = k; }
      }
      return best < 1e300;
    };
    double P = 0, H = 0, var = 0;
    long m = 0;
    for (int it = 0; it < 3; ++it) {
      if (it > 0) {
        const double sp = std::sin(P), cp = std::cos(P);
        for (size_t j = 0; j < pts_.size(); ++j) use[j] = std::fabs(H - hgt(j, sp, cp)) < 0.015;
      }
      double P0 = 0;
      if (!search(-0.10, 0.40, 0.005, &P0, &H, &var, &m)) return r;   // 숙임 −5.7° ~ 23° 거친 격자
      if (!search(P0 - 0.005, P0 + 0.005, 0.0005, &P, &H, &var, &m)) return r;
      if (m < kMinPts) return r;
    }
    r.ok = true;
    r.pitch = P;
    r.h = H;
    r.rms = std::sqrt(var);
    r.n = m;
    return r;
  }
  static constexpr long kMinPts = 500;   // 이보다 적으면 맞춤 안 함 (바닥이 거의 안 보임 — 반들한 바닥은 1 m 밖이 구멍)

 private:
  struct Pt { float v, z; };   // 센서 픽셀 행 · 광축 깊이 (m)
  std::vector<Pt> pts_;
  size_t cap_;
  size_t head_ = 0;
};

// 설정 숙임 (rad)
inline double cfg_pitch(const EnvCfg& c) { return std::atan2((double)c.cam_sp, (double)c.cam_cp); }

}  // namespace sim
