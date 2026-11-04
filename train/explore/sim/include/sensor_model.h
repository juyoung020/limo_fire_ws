// 센서 모델 파일(config/sensor_model.yaml) 읽기(호스트)와 판마다 무작위화(호스트·장치 공용 — 환경 커널이 리셋 때 부른다)
#pragma once
#include <cstdint>
#include <string>

#include "imu.h"
#include "lidar_real.h"

namespace sim {

struct Range2 {
  float lo = 1.f, hi = 1.f;
};

struct LidarModel {
  LidarNoise nominal{};
  float latency = 0.f;
  Range2 r_sigma, r_dropout, r_edge_drop;
};

LidarModel load_lidar_model(const std::string& yaml_path);

// 깊이 카메라 (sensor_model.yaml depth, 실물 녹화로 맞춤 — tools/depth_check): 핀홀 내부값, 유효 영역(가장자리 늘 0 인 띠),
// 시간 σ = k·z² × 가까운 거리 배율(near_x, near_z0 아래 전부 · near_z1 까지 1 로 줄어듦),
// 구멍: 바닥(수평 면) 입사 cos 구간 표 — 바닥마다 다름(반들한 실험실이 사무실보다 많음)이라 [lo, hi] 사이에서 판마다 한 값(EnvState::depth_floor),
// 모서리(옆 픽셀 깊이가 끊김) 거리 구간 표, 그 밖 면 한 값
constexpr int kDepthFloorBins = 7, kDepthEdgeBins = 6;
struct DepthModel {
  DepthSpec spec{};
  float k = 0.f;        // σ(z) = k·z² (m) × near(z)
  float near_x = 1.f, near_z0 = 0.f, near_z1 = 0.f;   // 가까운 거리 σ 배율: z ≤ near_z0 에서 near_x, near_z1 에서 1 (선형)
  float quantum = 0.001f;
  float floor_cos[kDepthFloorBins + 1];   // 바닥 입사 cos 구간 경계
  float floor_hole[kDepthFloorBins];      // 바닥 구멍 [lo] — 판마다 lo + u·(hi − lo)
  float floor_hole_hi[kDepthFloorBins];
  float edge_z[kDepthEdgeBins + 1];       // 모서리 거리 구간 경계 (m)
  float edge_hole[kDepthEdgeBins];
  float other_hole = 0.f;
  float edge_px = 3.f;                    // 모서리 판정: 좌우 이만큼 옆 픽셀 광선
};
DepthModel load_depth_model(const std::string& yaml_path);

// IMU (sensor_model.yaml imu, imu.h). dt: 물리 걸음 — AR 계수를 정한다
ImuModel load_imu_model(const std::string& yaml_path, float dt);

// 깊이 한 픽셀 (u, v) — 카메라 자세 cam(광축 = 센서 x), seed 로 정해지는 구멍·잡음. 구멍·범위 밖·유효 영역 밖 0, 아니면 z (m, quantum 반올림)
DEV int depth_bin(const float* edges, int n, float x) {
  for (int i = 0; i < n; ++i) if (x >= edges[i] && x < edges[i + 1]) return i;
  return -1;
}
// 깊이 한 픽셀을 단계로 (GPU 는 단계 사이에 광선을 묶어 쏜다 — env_kernel.cu, CPU 는 depth_pixel_real 이 차례로):
//   depth_valid → 가운데 광선(depth_dir, tmax = depth_tmax) → depth_hole_decided(u_hole, 구멍 확률) → 정해지지 않았으면 옆 광선 둘 → depth_edge → depth_value
DEV bool depth_valid(const DepthSpec& D, float u, float v) {
  return !(v < (float)D.valid_row0 || v >= (float)D.valid_row1 || u < (float)D.valid_col0 || u >= (float)D.valid_col1);
}
DEV V3 depth_dir(const DepthSpec& D, const Pose& cam, float u, float v) { return rotate(cam, depth_ray(D, u, v)); }
DEV float depth_tmax(const DepthSpec& D) { return D.max_z + 1.f; }

// 구멍 확률: 모서리면 거리 구간 표(p_edge), 아니면 면 종류(사람·바닥 입사 cos·그 밖, p_flat). d = 가운데 광선 방향, tri = 맞은 삼각형
// floor_u: 이 판 바닥 구멍 표 위치 [0, 1] (EnvState::depth_floor)
DEV void depth_hole_probs(const SceneView& s, const DepthModel& M, V3 d, float z, int tri, float floor_u, float* p_edge, float* p_flat) {
  {
    const int b = depth_bin(M.edge_z, kDepthEdgeBins, z);
    *p_edge = b >= 0 ? M.edge_hole[b] : 0.f;
  }
  if (tri < -1) {
    *p_flat = M.other_hole;   // 사람: 바닥이 아닌 면 구멍 비율 (옷 재질은 아직 못 잼 📏)
  } else {
    const Tri& T = s.tris[tri];
    const V3 nrm = cross(sub(T.b, T.a), sub(T.c, T.a));
    const float nn = sqrtf(dot(nrm, nrm));
    if (dm::absf(nrm.z) > 0.9f * nn) {   // 바닥 (수평 면)
      const float c = dm::absf(dot(nrm, d)) / (nn * sqrtf(dot(d, d)));
      const int b = depth_bin(M.floor_cos, kDepthFloorBins, c);
      *p_flat = b >= 0 ? M.floor_hole[b] + floor_u * (M.floor_hole_hi[b] - M.floor_hole[b]) : 0.f;
    } else {
      *p_flat = M.other_hole;
    }
  }
}

// 모서리인지가 구멍 여부를 바꾸나: 1 구멍, 0 아님, −1 옆 광선이 있어야 앎 (u_hole 이 두 확률 사이)
DEV int depth_hole_decided(float u_hole, float p_edge, float p_flat) {
  if (u_hole < p_edge && u_hole < p_flat) return 1;
  if (!(u_hole < p_edge) && !(u_hole < p_flat)) return 0;
  return -1;
}

// 옆 광선 하나(좌우 edge_px 옆)의 깊이 zs 가 끊김인가 (실측 정의: 둘레 p90 − p10 > 0.05·z + 0.02). 모서리 = 왼쪽 또는 오른쪽
DEV bool depth_edge(float z, float zs) { return !(zs < kInf) || dm::absf(zs - z) > 0.05f * z + 0.02f; }

// 시간 σ(z) (m)
DEV float depth_sigma(const DepthModel& M, float z) {
  const float t = M.near_z1 > M.near_z0 ? (z - M.near_z0) / (M.near_z1 - M.near_z0) : 1.f;
  const float g = t <= 0.f ? M.near_x : (t >= 1.f ? 1.f : M.near_x + t * (1.f - M.near_x));
  return ((M.k * z) * z) * g;
}

// 구멍이 아닌 픽셀 값: 시간 잡음·범위·양자화. seed 는 u_hole 을 뽑은 뒤의 상태
DEV float depth_value(const DepthModel& M, float z, uint64_t& seed) {
  const DepthSpec& D = M.spec;
  const float n = gauss(seed);
  const float zz = z + depth_sigma(M, z) * n;
  if (!(zz >= D.min_z && zz <= D.max_z)) return 0.f;
  return (float)(int)(zz / M.quantum + 0.5f) * M.quantum;
}

DEV float depth_pixel_real(const SceneView& s, const DepthModel& M, const Pose& cam, float u, float v, uint64_t seed, float floor_u) {
  const DepthSpec& D = M.spec;
  if (!depth_valid(D, u, v)) return 0.f;
  const V3 d = depth_dir(D, cam, u, v);
  uint16_t obj;
  int tri;
  const float z = raycast(s, cam.p, d, depth_tmax(D), &obj, &tri);
  if (tri == -1) return 0.f;   // 못 맞음 (tri < −1 = 움직이는 사람 원기둥, scene.h raycast)
  const float u_hole = dm::rand01(seed);
  float p_edge, p_flat;
  depth_hole_probs(s, M, d, z, tri, floor_u, &p_edge, &p_flat);
  // 옆 광선은 모서리인지가 구멍 여부를 바꿀 때만 쏜다 — u_hole 이 두 확률 모두보다 작거나 모두 이상이면 답이 같다(값은 늘 같음)
  int hole = depth_hole_decided(u_hole, p_edge, p_flat);
  if (hole < 0) {
    bool edge = depth_edge(z, raycast(s, cam.p, depth_dir(D, cam, u - M.edge_px, v), depth_tmax(D), nullptr));
    if (!edge) edge = depth_edge(z, raycast(s, cam.p, depth_dir(D, cam, u + M.edge_px, v), depth_tmax(D), nullptr));
    hole = u_hole < (edge ? p_edge : p_flat) ? 1 : 0;
  }
  if (hole) return 0.f;
  return depth_value(M, z, seed);
}

DEV float clamp01(float x) { return x < 0.f ? 0.f : (x > 1.f ? 1.f : x); }

// 실측 값에 판마다 배율을 곱한다 (확률은 [0, 1] 로 자름)
DEV LidarNoise randomize(const LidarModel& m, uint64_t& rng) {
  LidarNoise N = m.nominal;
  const float ks = dm::rand_range(rng, m.r_sigma.lo, m.r_sigma.hi);
  const float kd = dm::rand_range(rng, m.r_dropout.lo, m.r_dropout.hi);
  const float ke = dm::rand_range(rng, m.r_edge_drop.lo, m.r_edge_drop.hi);
  N.sigma0 = N.sigma0 * ks;
  N.sigma1 = N.sigma1 * ks;
  N.p_random = clamp01(N.p_random * kd);
  N.f_dark = clamp01(N.f_dark * kd);
  N.f_mid = clamp01(N.f_mid * kd);
  if (N.f_dark + N.f_mid > 1.f) N.f_mid = 1.f - N.f_dark;
  N.p_edge_drop = clamp01(N.p_edge_drop * ke);
  N.p_graze = clamp01(N.p_graze * kd);
  N.f_spot = clamp01(N.f_spot * kd);
  N.f_spot_mid = clamp01(N.f_spot_mid * kd);
  return N;
}

}  // namespace sim
