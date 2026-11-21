// 시뮬 초음파를 실물 ESP32 다리(limo_fire_esp32_bridge)처럼 바꾸는 계산 (sonar_sim_node 가 쓴다, ROS 없이 시험).
// Webots DistanceSensor "sonar" 가 이미 흉내 내는 것은 여기서 다시 하지 않는다 (Webots 참조 문서 distancesensor.md):
//   원뿔 안 가장 가까운 광선 · 입사각 22.5° 넘는 광선은 안 돌아옴 · lookupTable 셋째 열 = 값에 비례한 정규 잡음.
// 여기서 더하는 것 = 실물에만 있는 것:
//   메아리 없음 표기(실물 펌웨어 0 → 다리 +inf) · 무작위 메아리 놓침 · 센서 고장(늘 메아리 없음) · 치우침·잡음 ·
//   측정 → 발행 지연(다리가 받은 시각으로 찍음) · 시각 앞섬(11-20 실물: 메아리 시각이 EKF TF 보다 앞서 연기 모드가 전부 버림).
#pragma once

#include <cmath>
#include <limits>
#include <random>

namespace sonar_sim
{

struct Effects
{
  double no_echo_ratio = 0.99;   // 값 ≥ 이 비율 × 메시지 max_range 면 메아리 없음 (Webots 는 4.0 m 를 냄)
  double min_range = 0.02;       // 실물 다리 메시지 값 (esp32_bridge.yaml, HC-SR04 데이터시트 2 cm)
  double max_range = 4.0;        //   〃 400 cm
  double drop_prob = 0.0;        // 메아리가 있어도 놓칠 확률 (0~1)
  double bias_m = 0.0;           // 거리 치우침 (m)
  double noise_sigma_m = 0.0;    // 더하는 정규 잡음 σ (m) — Webots 비례 잡음 위에
  bool dead = false;             // 센서 고장: 늘 메아리 없음 (11-20 실물 오른쪽 초음파가 늘 0)
};

// 시뮬 값 하나 → 실물 다리가 낼 값. 메아리 없음은 +inf (sensor_msgs/Range 규약, 다리와 같음)
template<class Rng>
double apply(const Effects & e, double sim_range, double sim_max_range, Rng & rng)
{
  constexpr double kInf = std::numeric_limits<double>::infinity();
  if (e.dead || !std::isfinite(sim_range)) {return kInf;}
  if (sim_range >= e.no_echo_ratio * sim_max_range) {return kInf;}
  if (e.drop_prob > 0.0 && std::uniform_real_distribution<double>(0.0, 1.0)(rng) < e.drop_prob) {return kInf;}
  double r = sim_range + e.bias_m;
  if (e.noise_sigma_m > 0.0) {r += std::normal_distribution<double>(0.0, e.noise_sigma_m)(rng);}
  // 펌웨어: 범위 밖이면 0 (= 메아리 없음)
  if (r < e.min_range || r > e.max_range) {return kInf;}
  return r;
}

}  // namespace sonar_sim
