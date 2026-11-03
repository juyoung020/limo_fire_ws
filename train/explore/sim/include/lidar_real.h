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
  const int cx = (int)floorf(p.x / size), cy = (int)floorf(p.y / size), cz = (int)floorf(p.z / size);
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
};

DEV BeamGeom beam_geom(const SceneView& s, const LidarNoise& N, const ScanIn& in, int j) {
  BeamGeom g{};
  const float inc = kTwoPi / in.n_rev;
  const float t = (float)j * (N.scan_time / in.n_rev);
  float sw, cw;
  dm::sincosf_d(in.w * t, &sw, &cw);
  g.o = add(in.pose.p, scale(rotate(in.pose, V3{1.f, 0.f, 0.f}), in.v * t));
  const float a = -0.5f * N.fov + (float)j * inc - in.phase * inc;
  g.d = beam_dir(in, a, sw, cw);
  int tri;
  g.r = raycast(s, g.o, g.d, N.max_range, &g.obj, &tri);
  g.hit = tri >= 0;
  if (!g.hit) return g;
  // 모서리: 빔 폭 안 광선 5 개(양끝·사분·가운데). 이웃 셋이 직선에서 벗어나면 모서리 (비스듬한 평면은 직선에 가까움)
  const float h = 0.5f * N.footprint;
  float rk[5];
  rk[2] = g.r;
  rk[0] = raycast(s, g.o, beam_dir(in, a - h, sw, cw), N.max_range, nullptr);
  rk[1] = raycast(s, g.o, beam_dir(in, a - 0.5f * h, sw, cw), N.max_range, nullptr);
  rk[3] = raycast(s, g.o, beam_dir(in, a + 0.5f * h, sw, cw), N.max_range, nullptr);
  rk[4] = raycast(s, g.o, beam_dir(in, a + h, sw, cw), N.max_range, nullptr);
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
  const Tri& T = s.tris[tri];
  const V3 nrm = cross(sub(T.b, T.a), sub(T.c, T.a));
  g.cosi = dm::absf(dot(nrm, g.d)) / sqrtf(dot(nrm, nrm));
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
};
DEV float lidar_beam_real(const SceneView& s, const LidarNoise& N, const ScanIn& in, int j, BeamRay* ray = nullptr) {
  const int n = scan_beams(N, in.n_rev);
  if (j >= n) return 0.f;
  uint64_t rs = in.seed ^ ((uint64_t)(j + 1) * 0x9E3779B97F4A7C15ull);
  const float u_edge = dm::rand01(rs), u_mix = dm::rand01(rs), u_at = dm::rand01(rs), u_drop = dm::rand01(rs), g = gauss(rs);
  const BeamGeom geo = beam_geom(s, N, in, j);
  if (!geo.hit) return 0.f;
  if (ray) { ray->o = geo.o; ray->d = geo.d; }
  float r = geo.r;
  if (geo.edge) {
    // 반반일수록 잘 빠지고, 남으면 두 거리 사이 값이 섞여 나온다
    if (u_edge < edge_drop_prob(N, geo)) return 0.f;
    if (u_mix < N.p_mix) r = geo.lo + u_at * (geo.hi - geo.lo);
  }
  // 빠짐: 재질 · 비스듬함(가운데 광선이 맞은 표면 조각) · 무작위
  const Material mat = material_of(N, in.world_seed, geo.obj, s.obj_size[geo.obj]);
  const float thr = N.p_graze * (1.f - geo.cosi);
  const V3 hitp = add(geo.o, scale(geo.d, geo.r));
  const float u = patch_rank(N.patch_size, kSaltGraze, in.world_seed, geo.obj, hitp);
  const float p_graze = u < thr - N.graze_flicker ? 1.f : (u < thr + N.graze_flicker ? 0.5f : 0.f);
  const float us = patch_rank(N.spot_size, kSaltSpot, in.world_seed, geo.obj, hitp);
  if (us < N.f_spot) return 0.f;   // 안 보이는 점
  // 가끔 빠지는 점: 그 점의 빠짐 확률은 순위로 정해져 늘 같다
  const float p_spot = us < N.f_spot + N.f_spot_mid ? N.mid_lo + (N.mid_hi - N.mid_lo) * ((us - N.f_spot) / N.f_spot_mid) : 0.f;
  const float keep = (1.f - N.p_random) * (1.f - p_spot) * (1.f - mat.p_drop) * (1.f - p_graze);
  if (u_drop >= keep) return 0.f;
  // 잡음·양자화·범위
  r = r + (N.sigma0 + N.sigma1 * r) * mat.sigma_scale * g;
  if (!(r >= N.min_range && r < N.max_range)) return 0.f;
  return (float)(int)(r / N.quantum + 0.5f) * N.quantum;
}

}  // namespace sim
