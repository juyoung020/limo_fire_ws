// 카메라 키프레임 기억 (판단 모델 위층의 후보별 지난 장면 — scripts/low_decider.h, 실물 노드 upper:=decider decider_cam).
// 걸음 every 마다 줄인 컬러(RGB)·깊이(mm)와 그때 지도 자세를 고리(cap 장)에 담고, 후보 자리(바닥 점)를 가장 잘 본 장면을 기하로 찾는다:
//   시야 안 (가장자리 margin px 안쪽) · 저장 깊이로 가려지지 않음 (그 픽셀 깊이 0 = 모름, 또는 ≥ 거리 − occl_m) · 점수 = 거리 + age_w × 지난 초 (가깝고 최근일수록 좋음).
// 카메라 자세: 몸통(지도 x, y, th) + 달린 자리(mx, my 몸통 틀, 바닥 위 높이 h) + 아래로 숙인 각 pitch. 광학 틀 x 오른 · y 아래 · z 앞.
// 선행: ReMEmbR (arXiv 2409.13682) 의 "시각 · 자리와 함께 담는 기억"을 의미 임베딩 대신 기하 검색으로 (질문이 "그 자리가 어떻게 보였나"라 정확하고 쌈).
#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <deque>
#include <vector>

namespace camm {

struct Intr { float fx = 0, fy = 0, cx = 0, cy = 0; int w = 0, h = 0; };   // 담은(줄인) 그림 기준
struct Mount { float mx = 0.15f, my = 0.f, h = 0.19f, pitch = 0.115f; };   // 몸통 틀 m · 바닥 위 m · 아래로 rad (실물 tf base_link→camera_link, 1116 녹화)
struct Frame {
  int step = 0;
  float x = 0, y = 0, th = 0;          // 지도 자세
  std::vector<uint8_t> rgb;            // [h][w][3]
  std::vector<uint16_t> depth;         // [dh][dw] mm (0 = 모름), 컬러와 같은 시야
  int dw = 0, dh = 0;
};
struct Hit { int idx = -1; float u = 0, v = 0, range = 0, age_s = 0; };

class Memory {
 public:
  int every = 10;        // 걸음 (1 s)
  size_t cap = 120;      // 장 (2 분)
  float occl_m = 0.3f, age_w = 0.02f;
  int margin = 4;
  Intr in;
  Mount mt;
  bool due(int step) const { return step % every == 0 && (f_.empty() || f_.back().step != step); }
  void push(Frame&& fr) { f_.push_back(std::move(fr)); if (f_.size() > cap) f_.pop_front(); }
  void clear() { f_.clear(); }
  const std::deque<Frame>& frames() const { return f_; }
  // 바닥 점 (px, py) 를 프레임 k 에 투영: 보이면 true (u, v 담은 그림 픽셀, range = 광학 z)
  bool project(const Frame& fr, float px, float py, float* u, float* v, float* zc) const {
    const float c = std::cos(fr.th), s = std::sin(fr.th);
    const float camx = fr.x + c * mt.mx - s * mt.my, camy = fr.y + s * mt.mx + c * mt.my;
    const float dx = px - camx, dy = py - camy, dz = -mt.h;
    const float bx = c * dx + s * dy, by = -s * dx + c * dy, bz = dz;   // 몸통 틀
    const float cp = std::cos(mt.pitch), sp = std::sin(mt.pitch);
    const float z = bx * cp - bz * sp, xo = -by, yo = -(bx * sp + bz * cp);
    if (z < 0.1f) return false;
    *u = in.fx * xo / z + in.cx;
    *v = in.fy * yo / z + in.cy;
    *zc = z;
    return *u >= margin && *v >= margin && *u < in.w - margin && *v < in.h - margin;
  }
  // 그 자리를 가장 잘 본 프레임 (가깝고 최근 · 안 가려짐)
  bool find(float px, float py, int now_step, double dt, Hit* out) const {
    float best = 1e30f;
    bool ok = false;
    for (int k = 0; k < (int)f_.size(); ++k) {
      const Frame& fr = f_[(size_t)k];
      float u, v, zc;
      if (!project(fr, px, py, &u, &v, &zc)) continue;
      if (!fr.depth.empty() && fr.dw > 0) {
        const int du = std::min(fr.dw - 1, std::max(0, (int)(u * fr.dw / in.w))), dv = std::min(fr.dh - 1, std::max(0, (int)(v * fr.dh / in.h)));
        const uint16_t d = fr.depth[(size_t)dv * fr.dw + du];
        if (d != 0 && d * 1e-3f < zc - occl_m) continue;   // 앞에 막힌 것이 있음
      }
      const float age = (float)((now_step - fr.step) * dt);
      const float sc = zc + age_w * age;
      if (sc < best) { best = sc; ok = true; *out = Hit{k, u, v, zc, age}; }
    }
    return ok;
  }

 private:
  std::deque<Frame> f_;
};

}  // namespace camm
