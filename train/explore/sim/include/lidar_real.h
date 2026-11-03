// 실물 라이다 스캔: YDLIDAR T-mini Plus + ydlidar_ros2_driver 가 내는 /scan 을 흉내 낸다. 값과 근거는 config/sensor_model.yaml,
// analysis/real_limo.md. 같은 장면·ScanIn·LidarNoise 면 CPU·GPU 비트 동일 (detmath 규칙, 난수는 장·빔 번호로 정해지는 splitmix).
//
//   빔 격자 (드라이버 ydlidar_ros2_driver_node.cpp:203-209): 한 바퀴 점 수 N 이 장마다 바뀐다(실측 391~421).
//     inc = 2π/N, 빔 수 n = (int)(fov/inc + 1), 점은 index = ceil((각도 − min)/inc) 칸에 들어간다
//     → 빔 j 의 실제 각도 = −fov/2 + j·inc − phase·inc. phase 는 장마다 0.5 ± phase_jitter/2
//     (실물 정지: 모서리 빔도 값이 거의 안 흔들림 → 점 각도가 장마다 거의 같다). N 은 실측처럼 정수(분위 표의 값)
//   시각: 빔 j 는 스캔 시작 + j·(scan_time/N). 로봇이 그동안 움직인다(v, w).
//   거리: 모서리(빔 폭 안 광선 5 개 중 이웃 셋이 직선에서 벗어남 > edge_gap — 비스듬한 평면은 모서리가 아님)면
//         가까운 면에 걸친 몫 f 로 p_edge_drop·4f(1 − f) 만큼 빠지고, 남으면 p_mix 로 두 거리 사이 값(혼합 점)
//         (실물 두 장소 모서리 빔: 63 % 빠짐, 남는 빔 중 24 % 가 3 cm 넘게 흔들림).
//         재질(판 world_seed · 물체마다 정해져 장이 바뀌어도 같음)이 빠짐과 잡음을 같이 정한다 (실물 정지 615 장):
//           안 보임(빔 7 % 가 유효율 < 5 %) · 약한 반사(가끔 빠지고 σ 가 두 배: 유효율 50~95 % 빔 σ 9.8 mm vs 늘 보이는 빔 4.7 mm) · 보통.
//           바닥 투영이 structure_size 보다 큰 물체(벽 같은 구조물)는 늘 보통 — 실측 평평한 벽은 거의 안 빠졌다.
//           σ 는 물체마다 배율 exp(sigma_spread·N(0,1)) (실측 표면 빔 σ 중앙 4.7 mm, p90 12 mm — 같은 면의 빔들이 함께 큼).
//         비스듬함: 평면 빔 평균 빠짐이 입사각과 함께 는다(두 장소: 0~30° 0.4~2 %, 30~60° 7 %, 60~90° 11~19 %). 그런데 빔마다는
//           거의 늘 보이거나 늘 빠진다 → 표면 조각(patch_size 칸)마다 반사 순위 u 를 정하고 u < p_graze·(1 − cos) 이면 빠짐,
//           그 경계 ± graze_flicker 안이면 장마다 반반(깜빡임).
//         점: 모든 표면(큰 벽 포함)에 빔 크기 칸(spot_size), f_spot 은 늘 안 보이고 f_spot_mid 는 가끔 빠진다 — 실물은 정지·주행 모두
//           이웃 둘은 보이는데 혼자 0 인 빔이 장마다 7~8 % (정지에선 같은 빔이 늘 0), 정지 빔별 유효율 50~99 % 인 빔이 23 %.
//         빠짐 = 1 − (1 − 무작위)(1 − 점)(1 − 재질)(1 − 비스듬함). 잡음 σ = (sigma0 + sigma1·r)·배율 가우스, quantum(1 mm) 반올림, [min, max) 밖은 0.
#pragma once
#include <cstdint>

#include "sensors.h"

namespace sim {

constexpr int kMaxBeams = 272;        // N ≤ 445 까지
constexpr float kTwoPi = 6.28318531f;

struct LidarNoise {
  float fov, min_range, max_range;
  float scan_time;                     // 한 바퀴 (s)
  float phase_jitter;                  // 장마다 점 각도 흔들림 (빔 간격 단위)
  float n_rev_q[17];                   // 한 바퀴 점 수 분위 0, 1/16, …, 1
  float sigma0, sigma1, quantum;
  float p_random;
  float f_dark, p_dark;                // 안 보이는 재질 물체 비율·그때 빠짐 확률
  float f_mid, mid_lo, mid_hi, mid_sigma;   // 약한 반사 재질 물체 비율·빠짐 확률 범위·σ 배율
  float sigma_spread;                  // 물체마다 σ 배율 exp(spread·N(0,1))
  float structure_size;                // 이보다 큰 물체(m)는 재질을 뽑지 않고 보통
  float patch_size, p_graze, graze_flicker;   // 비스듬함 빠짐: 조각 칸(m), 기울기, 깜빡임 폭
  float spot_size, f_spot, f_spot_mid;   // 점: 모든 표면에 spot_size 칸, f_spot 은 늘 빠짐, f_spot_mid 는 가끔(빠짐 확률 mid_lo~mid_hi)
  float footprint, edge_gap, p_edge_drop, p_mix;
  float int_r[5], int_c;               // 세기 평균 = 거리 표(kIntensityR 마디 사이 선형) + c·(입사 cos − 0.8) (lidar_intensity)
  float int_obj, int_sigma, int_drop;  // 물체마다 세기 차 σ, 장마다 흔들림 σ, 빠진 빔이 내는 세기
};

struct ScanIn {
  Pose pose;            // 스캔 시작(빔 0) 때 라이다 자세 (세계)
  float v, w;           // 로봇 속도 (m/s, rad/s, 세계 z 축 회전)
  float n_rev;          // 이번 장 한 바퀴 점 수 (sample_n_rev)
  float phase;          // [0, 1) — phase_of 로
  uint64_t seed;        // 이번 장 난수
  uint64_t world_seed;  // 판 난수 (표면 성질)
  int32_t noise;        // LidarNoise 배열 번호 (환경마다 무작위화)
  int32_t pad;
};

DEV int scan_beams(const LidarNoise& N, float n_rev) {
  const float inc = kTwoPi / n_rev;
  const int n = (int)(N.fov / inc + 1.f);
  return n < kMaxBeams ? n : kMaxBeams;
}

// u ∈ [0, 1) → 실측 분위 표의 값 하나 (양 끝 0·1 분위는 드문 값이라 뺀다)
DEV float sample_n_rev(const LidarNoise& N, float u) {
  int i = 1 + (int)(u * 15.f);
  if (i > 15) i = 15;
  return N.n_rev_q[i];
}

DEV float phase_of(const LidarNoise& N, float u) { return 0.5f + N.phase_jitter * (u - 0.5f); }

DEV float gauss(uint64_t& s) {
  const float u1 = 1.f - dm::rand01(s);   // (0, 1]
  const float u2 = dm::rand01(s);
  float sn, cs;
  dm::sincosf_d(kTwoPi * u2, &sn, &cs);
  return sqrtf(-2.f * dm::lnf_d(u1)) * cs;
}

struct Material {
  float p_drop, sigma_scale;
};

// 물체 재질 (판·물체로 정해짐)
DEV Material material_of(const LidarNoise& N, uint64_t world_seed, uint16_t obj, float size) {
  uint64_t h = world_seed ^ ((uint64_t)(obj + 1) * 0xD6E8FEB86659FD93ull);
  const float u = dm::rand01(h), v = dm::rand01(h);
  const float g = gauss(h);
  float ln_s = N.sigma_spread * g;
  Material m{0.f, 1.f};
  if (size >= N.structure_size) {
    // 벽 같은 큰 구조물: 보통 재질 (σ 배율만)
  } else if (u < N.f_dark) {
    m.p_drop = N.p_dark;
  } else if (u < N.f_dark + N.f_mid) {
    m.p_drop = N.mid_lo + (N.mid_hi - N.mid_lo) * v;
    ln_s = ln_s + dm::lnf_d(N.mid_sigma);
  }
  // exp(ln_s) = 2^(ln_s / ln 2): 결정적 exp 가 없어 정수·소수 나눠 소수는 급수
  const float y = ln_s * 1.44269504f;
  const int k = (int)floorf(y);
  const float f = (y - (float)k) * 0.693147181f;   // [0, ln 2)
  float e = 1.f, term = 1.f;
  for (int i = 1; i < 10; ++i) { term = term * f / (float)i; e = e + term; }
  m.sigma_scale = ldexpf(e, k);
  return m;
}

// 표면 조각의 순위 [0, 1) (판·물체·칸·용도(salt)로 정해짐 — 장이 바뀌어도 같은 자리는 같다)
DEV float patch_rank(float size, uint64_t salt, uint64_t world_seed, uint16_t obj, V3 p) {
  const float inv = 1.f / size;
  const int cx = dm::floor_div(p.x, size, inv), cy = dm::floor_div(p.y, size, inv), cz = dm::floor_div(p.z, size, inv);
  uint64_t h = (world_seed ^ salt) ^ ((uint64_t)(obj + 1) * 0x94D049BB133111EBull) ^ ((uint64_t)(uint32_t)cx * 0x9E3779B97F4A7C15ull) ^
               ((uint64_t)(uint32_t)cy * 0xC2B2AE3D27D4EB4Full) ^ ((uint64_t)(uint32_t)cz * 0x165667B19E3779F9ull);
  return dm::rand01(h);
}

constexpr uint64_t kSaltGraze = 0x0, kSaltSpot = 0x5bd1e995abcdef12ull;

// 스캔 중 시각 t 의 빔 방향(세계): 시작 자세의 센서 방향을 세계 z 축으로 w·t 만큼 돌린다
DEV V3 beam_dir(const ScanIn& in, float a, float sw, float cw) {
  float sa, ca;
  dm::sincosf_d(a, &sa, &ca);
  const V3 d = rotate(in.pose, V3{ca, sa, 0.f});
  const float x0 = cw * d.x, x1 = sw * d.y, y0 = sw * d.x, y1 = cw * d.y;
  return V3{x0 - x1, y0 + y1, d.z};
}

// 빔 하나의 기하 (오차 전): 맞은 거리·물체·입사 cos·모서리 걸침. 라이다(lidar_beam_real)와 실물 비교 분석이 같이 쓴다
struct BeamGeom {
  bool hit;
  float r;            // 가운데 광선 거리
  uint16_t obj;
  float cosi;         // 입사 cos (1 = 정면)
  bool edge;
  float f, lo, hi;    // 모서리면 가까운 면에 걸친 몫, 빔 폭 안 가장 가까운·먼 거리
  V3 o, d;            // 광선 원점·방향 (스캔 중 움직임 포함)
  float a, sw, cw;    // 가운데 광선 각(센서 기준)·스캔 중 돈 각의 sin·cos (beam_edges 가 옆 광선에 씀)
};

// 가운데 광선만 (hit·r·obj·cosi·o·d). 모서리 판정(edge·f·lo·hi)은 beam_edges. GPU 는 광선을 묶어 쏘려고 단계로 부른다:
// beam_center_ray(광선) → 거리·삼각형 → beam_center_hit
DEV void beam_center_ray(const LidarNoise& N, const ScanIn& in, int j, BeamGeom& g) {
  g = BeamGeom{};
  const float inc = kTwoPi / in.n_rev;
  const float t = (float)j * (N.scan_time / in.n_rev);
  dm::sincosf_d(in.w * t, &g.sw, &g.cw);
  g.o = add(in.pose.p, scale(rotate(in.pose, V3{1.f, 0.f, 0.f}), in.v * t));
  g.a = -0.5f * N.fov + (float)j * inc - in.phase * inc;
  g.d = beam_dir(in, g.a, g.sw, g.cw);
}
DEV void beam_center_hit(const SceneView& s, BeamGeom& g, float r, int tri) {
  g.r = r;
  g.obj = hit_obj_of(s, tri);
  g.hit = tri != -1;   // −2 − k = 사람 k (scene.h raycast)
  if (!g.hit) return;
  if (tri < -1) { const V3 hp = add(g.o, scale(g.d, g.r)); g.cosi = cyl_cos(s.cyl[-2 - tri], hp.x, hp.y, g.d.x, g.d.y, g.d.z); return; }   // 사람
  const Tri& T = s.tris[tri];
  const V3 nrm = cross(sub(T.b, T.a), sub(T.c, T.a));
  g.cosi = dm::absf(dot(nrm, g.d)) / sqrtf(dot(nrm, nrm));
}
DEV BeamGeom beam_center(const SceneView& s, const LidarNoise& N, const ScanIn& in, int j) {
  BeamGeom g;
  beam_center_ray(N, in, j, g);
  int tri;
  const float r = raycast(s, g.o, g.d, N.max_range, nullptr, &tri);
  beam_center_hit(s, g, r, tri);
  return g;
}

// 모서리: 빔 폭 안 광선 5 개(양끝·사분·가운데). 이웃 셋이 직선에서 벗어나면 모서리 (비스듬한 평면은 직선에 가까움). 가운데가 맞은 빔만.
// 옆 광선 k = 0..3 은 각 a − h, a − h/2, a + h/2, a + h (h = footprint/2) — beam_side_dir, 거리 rk[0, 1, 3, 4] → beam_edges_from
DEV V3 beam_side_dir(const LidarNoise& N, const ScanIn& in, const BeamGeom& g, int k) {
  const float h = 0.5f * N.footprint;
  const float a = k == 0 ? g.a - h : (k == 1 ? g.a - 0.5f * h : (k == 2 ? g.a + 0.5f * h : g.a + h));
  return beam_dir(in, a, g.sw, g.cw);
}
DEV void beam_edges_from(const LidarNoise& N, BeamGeom& g, const float side[4]) {
  const float rk[5] = {side[0], side[1], g.r, side[2], side[3]};
  bool all_hit = true;
  g.lo = rk[0];
  g.hi = rk[0];
  for (int k = 0; k < 5; ++k) {
    all_hit = all_hit && rk[k] < kInf;
    g.lo = dm::minf(g.lo, rk[k]);
    g.hi = dm::maxf(g.hi, rk[k]);
  }
  for (int k = 1; k < 4 && all_hit; ++k) g.edge = g.edge || dm::absf((rk[k - 1] + rk[k + 1]) - 2.f * rk[k]) > N.edge_gap;
  if (g.edge) {
    const float mid = 0.5f * (g.lo + g.hi);
    int near = 0;
    for (int k = 0; k < 5; ++k) near += rk[k] < mid;
    g.f = (float)near * 0.2f;
  }
}
DEV void beam_edges(const SceneView& s, const LidarNoise& N, const ScanIn& in, BeamGeom& g) {
  float side[4];
  for (int k = 0; k < 4; ++k) side[k] = raycast(s, g.o, beam_side_dir(N, in, g, k), N.max_range, nullptr);
  beam_edges_from(N, g, side);
}

DEV BeamGeom beam_geom(const SceneView& s, const LidarNoise& N, const ScanIn& in, int j) {
  BeamGeom g = beam_center(s, N, in, j);
  if (g.hit) beam_edges(s, N, in, g);
  return g;
}

// 모서리에 걸친 몫 f 로 빠질 확률: 반반일수록 잘 빠진다
DEV float edge_drop_prob(const LidarNoise& N, const BeamGeom& g) { return g.edge ? N.p_edge_drop * (4.f * g.f * (1.f - g.f)) : 0.f; }

// 판 재질을 평균한 기대 빠짐 확률 — 실물 녹화와 조건별로 비교·맞출 때 (replay_check)
DEV float expected_drop(const LidarNoise& N, const BeamGeom& g, float obj_size) {
  if (!g.hit) return 1.f;
  const float pe = edge_drop_prob(N, g);
  const float pm = obj_size >= N.structure_size ? 0.f : N.f_dark * N.p_dark + N.f_mid * 0.5f * (N.mid_lo + N.mid_hi);
  const float pg = dm::minf(1.f, dm::maxf(0.f, N.p_graze * (1.f - g.cosi)));
  const float ps = N.f_spot + N.f_spot_mid * 0.5f * (N.mid_lo + N.mid_hi);
  return pe + (1.f - pe) * (1.f - (1.f - N.p_random) * (1.f - ps) * (1.f - pm) * (1.f - pg));
}

// ROS 순서 j 번째 빔 (m, 0 = 없음). j ≥ scan_beams 면 0. ray 가 있으면 이 빔의 참 광선(원점·방향, 스캔 중 움직임 포함)을 넣는다 —
// 끝점 = o + d · 돌려준 거리 (학습 환경의 벽 덮음, env.h scan_beam)
struct BeamRay {
  V3 o, d;
  uint16_t obj;   // 가운데 광선이 맞은 물체 (사람이면 kObjPerson — 덮음에 안 셈)
  float cosi;     // 가운데 광선 입사 cos (세기 lidar_intensity 용)
};
// 빔 하나의 난수 (장 seed · 빔 번호로 정해짐)
struct BeamDraw {
  float u_edge, u_mix, u_at, u_drop, g;
};
DEV BeamDraw beam_draw(const ScanIn& in, int j) {
  uint64_t rs = in.seed ^ ((uint64_t)(j + 1) * 0x9E3779B97F4A7C15ull);
  BeamDraw w;
  w.u_edge = dm::rand01(rs);
  w.u_mix = dm::rand01(rs);
  w.u_at = dm::rand01(rs);
  w.u_drop = dm::rand01(rs);
  w.g = gauss(rs);
  return w;
}

// 가운데 광선 뒤: 0 빠짐(거리 0), 1 옆 광선 없이 끝냄, 2 옆 광선이 있어야 함. mat 에 재질.
// 빠짐: 재질 · 비스듬함(가운데 광선이 맞은 표면 조각) · 무작위. 옆 광선 4 개는 모서리 여부가 답을 바꿀 수 있을 때만 —
// 빠짐 문턱(걸친 몫 f = 0, 0.2, …, 1 중 가장 큰 값) 이상이고 섞임도 안 뽑혔으면 모서리든 아니든 r 이 같다
// (재질·점 빠짐과 모서리 빠짐은 모두 0 을 내므로 어느 쪽을 먼저 보든 같다)
DEV int beam_mid(const SceneView& s, const LidarNoise& N, const ScanIn& in, const BeamGeom& geo, const BeamDraw& w, Material* mat) {
  if (!geo.hit) return 0;
  *mat = material_of(N, in.world_seed, geo.obj, geo.obj == kObjPerson ? kInf : s.obj_size[geo.obj]);
  const float thr = N.p_graze * (1.f - geo.cosi);
  const V3 hitp = add(geo.o, scale(geo.d, geo.r));
  const float u = patch_rank(N.patch_size, kSaltGraze, in.world_seed, geo.obj, hitp);
  const float p_graze = u < thr - N.graze_flicker ? 1.f : (u < thr + N.graze_flicker ? 0.5f : 0.f);
  const float us = patch_rank(N.spot_size, kSaltSpot, in.world_seed, geo.obj, hitp);
  if (us < N.f_spot) return 0;   // 안 보이는 점
  // 가끔 빠지는 점: 그 점의 빠짐 확률은 순위로 정해져 늘 같다
  const float p_spot = us < N.f_spot + N.f_spot_mid ? N.mid_lo + (N.mid_hi - N.mid_lo) * ((us - N.f_spot) / N.f_spot_mid) : 0.f;
  const float keep = (1.f - N.p_random) * (1.f - p_spot) * (1.f - mat->p_drop) * (1.f - p_graze);
  if (w.u_drop >= keep) return 0;
  float pe_max = 0.f;
  for (int k = 0; k <= 5; ++k) {
    const float f = (float)k * 0.2f;
    pe_max = dm::maxf(pe_max, N.p_edge_drop * (4.f * f * (1.f - f)));
  }
  return (w.u_edge < pe_max || w.u_mix < N.p_mix) ? 2 : 1;
}

// 마무리 (beam_mid 가 1 이면 edges = false, 2 면 beam_edges 뒤 true): 모서리 빠짐·섞임 → 잡음·양자화·범위
DEV float beam_finish(const LidarNoise& N, const BeamGeom& geo, const BeamDraw& w, const Material& mat, bool edges) {
  float r = geo.r;
  if (edges && geo.edge) {
    // 반반일수록 잘 빠지고, 남으면 두 거리 사이 값이 섞여 나온다
    if (w.u_edge < edge_drop_prob(N, geo)) return 0.f;
    if (w.u_mix < N.p_mix) r = geo.lo + w.u_at * (geo.hi - geo.lo);
  }
  r = r + (N.sigma0 + N.sigma1 * r) * mat.sigma_scale * w.g;
  if (!(r >= N.min_range && r < N.max_range)) return 0.f;
  return (float)(int)(r / N.quantum + 0.5f) * N.quantum;
}

DEV void beam_ray_out(const BeamGeom& geo, BeamRay* ray) {
  if (ray) { ray->o = geo.o; ray->d = geo.d; ray->obj = geo.obj; ray->cosi = geo.cosi; }
}

DEV float lidar_beam_real(const SceneView& s, const LidarNoise& N, const ScanIn& in, int j, BeamRay* ray = nullptr) {
  const int n = scan_beams(N, in.n_rev);
  if (j >= n) return 0.f;
  const BeamDraw w = beam_draw(in, j);
  BeamGeom geo = beam_center(s, N, in, j);
  if (!geo.hit) return 0.f;
  beam_ray_out(geo, ray);
  Material mat;
  const int st = beam_mid(s, N, in, geo, w, &mat);
  if (st == 0) return 0.f;
  if (st == 2) beam_edges(s, N, in, geo);
  return beam_finish(N, geo, w, mat, st == 2);
}

}  // namespace sim

namespace sim {

// 빔 세기 (실물 /scan intensities 흉내, sensor_model.yaml lidar.intensity — tools/intensity_check 로 맞춤).
//   맞은 빔: 거리 표(마디 kIntensityR, 사이 선형, 끝 밖은 끝값) + c·(입사 cos − 0.8) + 물체마다 차(판·물체로 정해짐, σ int_obj)
//   + 장마다 흔들림(σ int_sigma), 정수로 반올림, 0 아래는 0. 실측 세기는 거리에 단조가 아님(0.75 m 가 가장 밝음) — 표로 둔다.
//   빠진 빔(거리 0): 드라이버가 세기는 그대로 내므로 실측 평균 int_drop.
//   (재질 모델의 안 보임·약한 반사와 세기의 관계는 아직 못 잼 — 시연장 재질판 녹화 R3 로)
constexpr float kIntensityR[5] = {0.25f, 0.75f, 1.5f, 3.f, 6.f};
DEV float intensity_mean(const LidarNoise& N, float r, float cosi) {
  const float kIntensityR[5] = {0.25f, 0.75f, 1.5f, 3.f, 6.f};   // 위 상수와 같은 값 (장치 코드는 바깥 constexpr 배열을 못 읽음)
  float m = N.int_r[0];
  if (r >= kIntensityR[4]) m = N.int_r[4];
  else
    for (int i = 0; i < 4; ++i)
      if (r >= kIntensityR[i] && r < kIntensityR[i + 1]) m = N.int_r[i] + (N.int_r[i + 1] - N.int_r[i]) * ((r - kIntensityR[i]) / (kIntensityR[i + 1] - kIntensityR[i]));
  return m + N.int_c * (cosi - 0.8f);
}
DEV float lidar_intensity(const LidarNoise& N, uint64_t world_seed, uint16_t obj, float r, float cosi, bool dropped, uint64_t seed) {
  if (dropped) return N.int_drop;
  uint64_t h = world_seed ^ ((uint64_t)(obj + 1) * 0xBF58476D1CE4E5B9ull) ^ 0x1D872B41ull;
  const float off = N.int_obj * gauss(h);
  const float x = (intensity_mean(N, r, cosi) + off) + N.int_sigma * gauss(seed);
  return x <= 0.f ? 0.f : (float)(int)(x + 0.5f);
}

}  // namespace sim
