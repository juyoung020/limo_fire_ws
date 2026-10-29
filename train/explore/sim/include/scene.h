// 장면: 삼각형 + BVH, 광선 추적. CPU 참조판·GPU 커널·ROS 검증 노드가 이 헤더를 같이 쓴다(호스트·장치 공용, ROS 없음).
// 비트 동일: detmath.h 규칙 그대로 — 곱·합을 따로(fma 안 씀, -ffp-contract=off / --fmad=false), 나눗셈·제곱근은 IEEE 정확.
// 같은 장면·같은 광선이면 CPU·sm_87(젯슨)·sm_120(데스크톱)이 같은 거리를 낸다 (tools/sim_verify).
#pragma once
#include <cstdint>

#include "detmath.h"

namespace sim {

struct V3 {
  float x, y, z;
};
DEV V3 v3(float x, float y, float z) { return V3{x, y, z}; }
DEV V3 add(V3 a, V3 b) { return V3{a.x + b.x, a.y + b.y, a.z + b.z}; }
DEV V3 sub(V3 a, V3 b) { return V3{a.x - b.x, a.y - b.y, a.z - b.z}; }
DEV V3 scale(V3 a, float s) { return V3{a.x * s, a.y * s, a.z * s}; }
DEV float dot(V3 a, V3 b) {
  const float xx = a.x * b.x, yy = a.y * b.y, zz = a.z * b.z;
  return (xx + yy) + zz;
}
DEV V3 cross(V3 a, V3 b) {
  const float x1 = a.y * b.z, x2 = a.z * b.y, y1 = a.z * b.x, y2 = a.x * b.z, z1 = a.x * b.y, z2 = a.y * b.x;
  return V3{x1 - x2, y1 - y2, z1 - z2};
}

// 3×3 회전(행 우선) + 위치: 센서·물체 자세. world = R · local + p
struct Pose {
  float r[9];
  V3 p;
};
DEV V3 rotate(const Pose& T, V3 v) {
  const float x0 = T.r[0] * v.x, x1 = T.r[1] * v.y, x2 = T.r[2] * v.z;
  const float y0 = T.r[3] * v.x, y1 = T.r[4] * v.y, y2 = T.r[5] * v.z;
  const float z0 = T.r[6] * v.x, z1 = T.r[7] * v.y, z2 = T.r[8] * v.z;
  return V3{(x0 + x1) + x2, (y0 + y1) + y2, (z0 + z1) + z2};
}

struct Tri {
  V3 a, b, c;
};

// BVH 마디: count > 0 이면 잎(삼각형 [first, first + count)), 아니면 안쪽(왼쪽 = first, 오른쪽 = first + 1)
struct Node {
  float lo[3], hi[3];
  int32_t first;
  int32_t count;
};

// 장치·호스트 공용 보기
struct SceneView {
  const Node* nodes;
  const Tri* tris;
  const uint16_t* obj;   // 삼각형마다 물체 번호 (Scene::obj_names)
};

constexpr float kInf = 3.0e38f;
constexpr int kStack = 64;   // 쌓기 깊이 한도. 장면을 만들 때 BVH 깊이 ≤ kMaxDepth 를 보장한다
constexpr int kMaxDepth = 48;

// Möller–Trumbore. 앞·뒷면 모두 맞음. 못 맞으면 kInf
DEV float ray_tri(V3 o, V3 d, const Tri& t) {
  const V3 e1 = sub(t.b, t.a), e2 = sub(t.c, t.a);
  const V3 p = cross(d, e2);
  const float det = dot(e1, p);
  if (dm::absf(det) < 1e-12f) return kInf;
  const float inv = 1.0f / det;
  const V3 s = sub(o, t.a);
  const float u = dot(s, p) * inv;
  if (u < 0.f || u > 1.f) return kInf;
  const V3 q = cross(s, e1);
  const float v = dot(d, q) * inv;
  if (v < 0.f || u + v > 1.f) return kInf;
  const float tt = dot(e2, q) * inv;
  return tt > 1e-6f ? tt : kInf;
}

// 상자와의 [들어감, 나감]. inv = 1/d (0 방향은 ±큰 값)
DEV bool ray_box(V3 o, V3 inv, const Node& n, float tmax, float* tin) {
  float t0 = 0.f, t1 = tmax;
  const float oo[3] = {o.x, o.y, o.z}, ii[3] = {inv.x, inv.y, inv.z};
  for (int k = 0; k < 3; ++k) {
    float a = (n.lo[k] - oo[k]) * ii[k];
    float b = (n.hi[k] - oo[k]) * ii[k];
    if (a > b) { const float c = a; a = b; b = c; }
    t0 = a > t0 ? a : t0;
    t1 = b < t1 ? b : t1;
    if (t0 > t1) return false;
  }
  *tin = t0;
  return true;
}

DEV float safe_inv(float x) { return dm::absf(x) < 1e-20f ? (x < 0.f ? -1e30f : 1e30f) : 1.0f / x; }

// 가장 가까운 맞음까지 거리(d 단위). 없으면 kInf. hit_obj 에 물체 번호(못 맞으면 0xffff)
DEV float raycast(const SceneView& s, V3 o, V3 d, float tmax, uint16_t* hit_obj) {
  const V3 inv{safe_inv(d.x), safe_inv(d.y), safe_inv(d.z)};
  float best = tmax;
  int best_tri = -1;
  int stack[kStack];
  int sp = 0;
  stack[sp++] = 0;
  while (sp > 0) {
    const Node& n = s.nodes[stack[--sp]];
    float tin;
    if (!ray_box(o, inv, n, best, &tin)) continue;
    if (n.count > 0) {
      for (int i = n.first; i < n.first + n.count; ++i) {
        const float t = ray_tri(o, d, s.tris[i]);
        if (t < best) { best = t; best_tri = i; }
      }
    } else {
      // 가까운 쪽을 먼저 보려고 먼 쪽을 먼저 쌓는다
      float ta, tb;
      const bool ha = ray_box(o, inv, s.nodes[n.first], best, &ta);
      const bool hb = ray_box(o, inv, s.nodes[n.first + 1], best, &tb);
      if (ha && hb) {
        if (ta <= tb) { stack[sp++] = n.first + 1; stack[sp++] = n.first; }
        else { stack[sp++] = n.first; stack[sp++] = n.first + 1; }
      } else if (ha) {
        stack[sp++] = n.first;
      } else if (hb) {
        stack[sp++] = n.first + 1;
      }
    }
  }
  if (hit_obj) *hit_obj = best_tri >= 0 ? s.obj[best_tri] : (uint16_t)0xffff;
  return best_tri >= 0 ? best : kInf;
}

}  // namespace sim
