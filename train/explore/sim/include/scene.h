// 장면: 삼각형 + BVH, 광선 추적. CPU 참조판·GPU 커널·ROS 검증 노드가 이 헤더를 같이 쓴다(호스트·장치 공용, ROS 없음).
// 비트 동일: detmath.h 규칙 그대로 — 곱·합을 따로(fma 안 씀, -ffp-contract=off / --fmad=false), 나눗셈·제곱근은 IEEE 정확.
// 같은 장면·같은 광선이면 CPU·sm_87(젯슨)·sm_120(데스크톱)이 같은 거리를 낸다 (tools/sim_verify).
#pragma once
#include <cstdint>

#include "detmath.h"
#include "people_geom.h"

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
  const float* obj_size; // 물체마다 바닥 투영 크기(가로·세로 중 긴 쪽, m) — 큰 구조물(벽)인지 가린다
  const Cyl* cyl;        // 움직이는 사람 원기둥 (people_geom.h, 없으면 n_cyl 0) — 맞으면 hit_obj kObjPerson, hit_tri −2 − 번호
  int32_t n_cyl;
};

constexpr float kInf = 3.0e38f;
constexpr int kStack = 64;   // 쌓기 깊이 한도. 장면을 만들 때 BVH 깊이 ≤ kMaxDepth 를 보장한다
constexpr int kMaxDepth = 48;

// Möller–Trumbore. 앞·뒷면 모두 맞음. 못 맞으면 kInf.
// 곱·합을 _rn 내장 함수로 적는다(장치): 이 함수는 OptiX 교차 프로그램(별도 컴파일러)에서도 돌아 fma 로 합쳐지면 안 된다 — 값은 --fmad=false 와 같다
namespace rt {
#ifdef __CUDA_ARCH__
DEV float m(float a, float b) { return __fmul_rn(a, b); }
DEV float a(float x, float y) { return __fadd_rn(x, y); }
DEV float s(float x, float y) { return __fsub_rn(x, y); }
DEV float inv(float x) { return __fdiv_rn(1.0f, x); }
#else
DEV float m(float a, float b) { return a * b; }
DEV float a(float x, float y) { return x + y; }
DEV float s(float x, float y) { return x - y; }
DEV float inv(float x) { return 1.0f / x; }
#endif
DEV V3 sub(V3 p, V3 q) { return V3{s(p.x, q.x), s(p.y, q.y), s(p.z, q.z)}; }
DEV float dot(V3 p, V3 q) { return a(a(m(p.x, q.x), m(p.y, q.y)), m(p.z, q.z)); }
DEV V3 cross(V3 p, V3 q) { return V3{s(m(p.y, q.z), m(p.z, q.y)), s(m(p.z, q.x), m(p.x, q.z)), s(m(p.x, q.y), m(p.y, q.x))}; }
}  // namespace rt
DEV float ray_tri(V3 o, V3 d, const Tri& t) {
  const V3 e1 = rt::sub(t.b, t.a), e2 = rt::sub(t.c, t.a);
  const V3 p = rt::cross(d, e2);
  const float det = rt::dot(e1, p);
  if (dm::absf(det) < 1e-12f) return kInf;
  const float inv = rt::inv(det);
  const V3 s = rt::sub(o, t.a);
  const float u = rt::m(rt::dot(s, p), inv);
  if (u < 0.f || u > 1.f) return kInf;
  const V3 q = rt::cross(s, e1);
  const float v = rt::m(rt::dot(d, q), inv);
  if (v < 0.f || rt::a(u, v) > 1.f) return kInf;
  const float tt = rt::m(rt::dot(e2, q), inv);
  return tt > 1e-6f ? tt : kInf;
}

// 상자와의 [들어감, 나감]. inv = 1/d (0 방향은 ±큰 값).
// 축마다 들어감·나감을 다 구한 뒤 한 번 비교한다: 들어감은 늘기만·나감은 줄기만 하므로 축마다 일찍 끝내던 판과 참·거짓·tin 이 같다
// (tin 의 ±0 부호만 다를 수 있는데 tin 은 크기 비교에만 쓴다). tmax 가 클수록 참이 되기 쉽다(단조) — raycast 가 이걸 쓴다
DEV bool ray_box(V3 o, V3 inv, const Node& n, float tmax, float* tin) {
  const float a0 = (n.lo[0] - o.x) * inv.x, b0 = (n.hi[0] - o.x) * inv.x;
  const float a1 = (n.lo[1] - o.y) * inv.y, b1 = (n.hi[1] - o.y) * inv.y;
  const float a2 = (n.lo[2] - o.z) * inv.z, b2 = (n.hi[2] - o.z) * inv.z;
  const float t0 = dm::maxf(dm::maxf(dm::maxf(0.f, dm::minf(a0, b0)), dm::minf(a1, b1)), dm::minf(a2, b2));
  const float t1 = dm::minf(dm::minf(dm::minf(tmax, dm::maxf(a0, b0)), dm::maxf(a1, b1)), dm::maxf(a2, b2));
  *tin = t0;
  return !(t0 > t1);
}

DEV float safe_inv(float x) { return dm::absf(x) < 1e-20f ? (x < 0.f ? -1e30f : 1e30f) : 1.0f / x; }

// BVH 마디 읽기 (GPU 는 읽기 전용 캐시로 16 바이트씩)
DEV Node load_node(const Node* nodes, int i) {
#ifdef __CUDA_ARCH__
  const float4* p = reinterpret_cast<const float4*>(nodes + i);
  const float4 x = __ldg(p), y = __ldg(p + 1);
  Node n;
  n.lo[0] = x.x; n.lo[1] = x.y; n.lo[2] = x.z; n.hi[0] = x.w;
  n.hi[1] = y.x; n.hi[2] = y.y; n.first = __float_as_int(y.z); n.count = __float_as_int(y.w);
  return n;
#else
  return nodes[i];
#endif
}
DEV Tri load_tri(const Tri* tris, int i) {
#ifdef __CUDA_ARCH__
  const float* p = reinterpret_cast<const float*>(tris + i);
  return Tri{V3{__ldg(p), __ldg(p + 1), __ldg(p + 2)}, V3{__ldg(p + 3), __ldg(p + 4), __ldg(p + 5)}, V3{__ldg(p + 6), __ldg(p + 7), __ldg(p + 8)}};
#else
  return tris[i];
#endif
}

// 광선 하나의 순회 상태. raycast 는 trav_init → trav_step 을 끝날 때까지 → trav_result 다. GPU 의 광선 묶음 커널(env_kernel.cu)은
// 같은 걸음을 스레드마다 하나씩 돌리고, 끝난 스레드가 새 광선을 받아 워프를 채운다 — 광선마다 걸음 순서는 같아 결과도 같다.
// 순회 순서·자르기는 처음 판과 같다(같은 순서로 같은 삼각형을 먼저 만나 같은 결과). 꺼낸 마디를 다시 재는 시험은 쌓은 뒤 best 가 줄었을 때만 —
// best 가 그대로면 쌓을 때와 같은 입력이라 같은 답(참)이다
struct RayTrav {
  V3 o, d, inv;
  float best;
  int best_tri;
  int sp, leaf_i, leaf_end;
  bool has_cur;   // 바로 앞 걸음이 쌓았다가 곧 꺼낼 가까운 아이를 레지스터에 둠 (쌓고 꺼내기·다시 읽기를 건너뜀 — 그사이 best 가 그대로라 재시험도 없음)
  Node cur;
};
// 쌓기 (레지스터에 둘 낱값과 따로 — 마디 번호, 쌓을 때의 best: 뿌리는 늘 다시 잼)
struct TravStack {
  int node[kStack];
  float pushed[kStack];
};

DEV void trav_init(RayTrav& r, TravStack& k_, const SceneView& s, V3 o, V3 d, float tmax) {
  r.o = o;
  r.d = d;
  r.inv = V3{safe_inv(d.x), safe_inv(d.y), safe_inv(d.z)};
  r.best = tmax;
  r.best_tri = -1;
  for (int k = 0; k < s.n_cyl; ++k) {   // 사람 먼저 (BVH 를 더 짧게 자름)
    const float t = cyl_hit(s.cyl[k], o.x, o.y, o.z, d.x, d.y, d.z);
    if (t < r.best) { r.best = t; r.best_tri = -2 - k; }
  }
  k_.node[0] = 0;
  k_.pushed[0] = -1.f;
  r.sp = 1;
  r.leaf_i = r.leaf_end = 0;
  r.has_cur = false;
}

// 상자 자르기 거리: best 보다 조금 멀리까지 본다(상대 1e−5) — 반올림으로 best 와 같은 t 의 삼각형(동점)이 든 상자를 놓치지 않게.
// 그래야 답이 순회 순서·상자 반올림과 무관한 (t, 번호) 최소가 되어 RT 코어 경로(키운 상자)와 같다
DEV float cull_t(float best) { return best + best * 1e-5f; }

// 한 걸음 = 잎의 삼각형 하나(trav_tri, 잎 안이면) 또는 마디 하나 꺼내기(trav_node, 쌓기가 비면 끝 → false)
DEV bool trav_in_leaf(const RayTrav& r) { return r.leaf_i < r.leaf_end; }
DEV void trav_tri(RayTrav& r, const SceneView& s) {
  const float t = ray_tri(r.o, r.d, load_tri(s.tris, r.leaf_i));
  // (t, 삼각형 번호) 사전순 최소 — 순회 순서와 무관한 동점 규칙(RT 코어 경로와 같은 답). 사람 원기둥보다는 엄격히 가까워야
  if (t < r.best || (t == r.best && r.best_tri >= 0 && r.leaf_i < r.best_tri)) { r.best = t; r.best_tri = r.leaf_i; }
  ++r.leaf_i;
}
DEV bool trav_node(RayTrav& r, TravStack& k_, const SceneView& s) {
  Node n;
  if (r.has_cur) {
    n = r.cur;
    r.has_cur = false;
  } else {
    if (r.sp == 0) return false;
    --r.sp;
    n = load_node(s.nodes, k_.node[r.sp]);
    float tin;
    if (k_.pushed[r.sp] != r.best && !ray_box(r.o, r.inv, n, cull_t(r.best), &tin)) return true;
  }
  if (n.count > 0) {
    r.leaf_i = n.first;
    r.leaf_end = n.first + n.count;
    return true;
  }
  // 가까운 쪽을 먼저 보려고 먼 쪽을 먼저 쌓는다
  // (가까운 쪽은 쌓았다 바로 꺼내는 것과 같게 레지스터 cur 로)
  float ta, tb;
  const Node na = load_node(s.nodes, n.first), nb = load_node(s.nodes, n.first + 1);
  const bool ha = ray_box(r.o, r.inv, na, cull_t(r.best), &ta);
  const bool hb = ray_box(r.o, r.inv, nb, cull_t(r.best), &tb);
  if (ha && hb) {
    const bool a_near = ta <= tb;
    k_.node[r.sp] = a_near ? n.first + 1 : n.first;
    k_.pushed[r.sp++] = r.best;
    r.cur = a_near ? na : nb;
    r.has_cur = true;
  } else if (ha || hb) {
    r.cur = ha ? na : nb;
    r.has_cur = true;
  }
  return true;
}
DEV bool trav_step(RayTrav& r, TravStack& k_, const SceneView& s) {
  if (trav_in_leaf(r)) {
    trav_tri(r, s);
    return true;
  }
  return trav_node(r, k_, s);
}

// 끝난 순회의 답: 거리(못 맞으면 kInf), 삼각형 번호(사람이면 −2 − 번호, 못 맞으면 −1)
DEV float trav_result(const RayTrav& r, int* hit_tri) {
  *hit_tri = r.best_tri;
  return r.best_tri != -1 ? r.best : kInf;
}
DEV uint16_t hit_obj_of(const SceneView& s, int tri) { return tri >= 0 ? s.obj[tri] : (tri < -1 ? kObjPerson : (uint16_t)0xffff); }

// 가장 가까운 맞음까지 거리(d 단위). 없으면 kInf. hit_obj 에 물체 번호(사람이면 kObjPerson, 못 맞으면 0xffff), hit_tri 에 삼각형 번호(사람이면 −2 − 사람 번호, 못 맞으면 −1)
DEV float raycast(const SceneView& s, V3 o, V3 d, float tmax, uint16_t* hit_obj, int* hit_tri = nullptr) {
  RayTrav r;
  TravStack k_;
  trav_init(r, k_, s, o, d, tmax);
  while (trav_step(r, k_, s)) {}
  int tri;
  const float t = trav_result(r, &tri);
  if (hit_obj) *hit_obj = hit_obj_of(s, tri);
  if (hit_tri) *hit_tri = tri;
  return t;
}

}  // namespace sim
