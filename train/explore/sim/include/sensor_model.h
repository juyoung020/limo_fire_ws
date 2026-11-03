// 센서 모델 파일(config/sensor_model.yaml) 읽기(호스트)와 판마다 무작위화(호스트·장치 공용 — 환경 커널이 리셋 때 부른다)
#pragma once
#include <cstdint>
#include <string>

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

// 깊이 카메라 (sensor_model.yaml depth): 핀홀 내부값 + 오차 σ = k·z² (스테레오, 자료 1 m 에서 6 mm)
struct DepthModel {
  DepthSpec spec{};
  float k = 0.f;        // σ(z) = k·z² (m)
  float quantum = 0.001f;
};
DepthModel load_depth_model(const std::string& yaml_path);

// 깊이 한 픽셀 (오차 넣음). 광선 방향 d(광축 성분 1)·원점 o, seed 로 정해지는 잡음. 범위 밖 0, 아니면 z (m, quantum 반올림)
DEV float depth_sample(const SceneView& s, const DepthModel& M, V3 o, V3 d, uint64_t seed) {
  const float z = raycast(s, o, d, M.spec.max_z + 1.f, nullptr);
  if (!(z < kInf)) return 0.f;
  const float n = gauss(seed);
  const float zz = (z + (M.k * z) * z * n);
  if (!(zz >= M.spec.min_z && zz <= M.spec.max_z)) return 0.f;
  return (float)(int)(zz / M.quantum + 0.5f) * M.quantum;
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
