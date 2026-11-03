// 움직이는 사람의 광선 기하: 세로 원기둥(바닥 z = 0 ~ 높이 h). scene.h raycast 가 정적 장면 BVH 와 따로 이 원기둥들을 본다
// (SceneView::cyl — 환경마다 그 판 사람들, env.h with_people). 호스트·장치 공용, detmath 규칙(곱·합 따로)이라 CPU == GPU 비트 같음.
// 사람 동작·충돌·놓기는 people.h · people_sim.h.
#pragma once
#include <cstdint>

#include "detmath.h"

namespace sim {

constexpr uint16_t kObjPerson = 0xfffe;   // raycast hit_obj: 사람 원기둥을 맞음 (장면 물체 번호와 안 겹침)

struct Cyl {
  float x, y, r, h;   // 가운데 (세계 m), 반지름, 높이
};

// 광선 o + t·d 가 원기둥 옆면에 처음 들어가는 t (d 단위). 못 맞으면 3e38.
// 옆면만 본다: 센서(라이다 0.11 m · 깊이 0.21 m)는 사람 키(1.6 m~)보다 낮아 윗면으로 들어가는 광선이 없다. 원점이 원기둥 안이면 안 봄
DEV float cyl_hit(const Cyl& c, float ox, float oy, float oz, float dx, float dy, float dz) {
  const float px = ox - c.x, py = oy - c.y;
  const float a = dx * dx + dy * dy;
  if (a < 1e-12f) return 3e38f;
  const float b = px * dx + py * dy;
  const float cc = (px * px + py * py) - c.r * c.r;
  if (cc <= 0.f || b >= 0.f) return 3e38f;   // 안에 있음 · 멀어지는 쪽
  const float disc = b * b - a * cc;
  if (disc < 0.f) return 3e38f;
  const float t = (-b - sqrtf(disc)) / a;
  const float z = oz + t * dz;
  if (!(t > 1e-6f) || z < 0.f || z > c.h) return 3e38f;
  return t;
}

// 맞은 점 (hx, hy) 에서 입사 cos (1 = 정면) — 법선은 가운데에서 바깥 (수평)
DEV float cyl_cos(const Cyl& c, float hx, float hy, float dx, float dy, float dz) {
  const float nx = hx - c.x, ny = hy - c.y;
  const float nn = nx * nx + ny * ny, dd = (dx * dx + dy * dy) + dz * dz;
  const float nd = nx * dx + ny * dy;
  return dm::absf(nd) / sqrtf(nn * dd);
}

}  // namespace sim
