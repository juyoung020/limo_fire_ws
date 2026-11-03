// 지도 자세 오차 (Cartographer map → odom 보정이 남기는 몫) — 학습 환경의 지도 자세 = 참값 + 이 오차. 호스트·장치 공용, CPU == GPU 비트 같음.
// 값과 근거: config/env.yaml pose_error, analysis/real_limo.md 지도 속도 절, 맞춤·확인 도구 tools/mapvel_check.
//
// 실물: 움직이면 map → odom 이 물리 걸음(50 Hz)의 63~93 % 에서 바뀌고(스캔 매칭 보정), 세워 두면 4~30 %. 지도 자세를 0.1 s 로 미분한
//   속도(관측 "지도 속도")가 바퀴 v·자이로 ω 와 어긋난다 (잔차 σ: 정지 0.01 m/s · 0.07 rad/s, 주행 0.1~0.16 m/s · 0.6~1 rad/s),
//   그 어긋남은 0.1 s 뒤에도 일부 남는다(자기상관 0.4~0.5 — 바퀴 헛돎·EKF 늦음처럼 한동안 이어지는 몫).
// 모델 (로봇 기준 앞·옆·yaw 마다):
//   보정: 물리 걸음마다 확률 p(움직이면 p_move, 아니면 p_rest)로 오차에 잡음 (1 − drift_f)·σ² 분산을 더한다
//   이어지는 몫: 표류 속도 d 가 상관 시간 tau_d 인 AR(1), 정상 분산 drift_f·σ²·p_move / (kPoseErrWinSteps · dt²) — 오차에 d·dt 를 더한다
//   σ: 정지 s0, 움직이면 σ² = q1 + qv·|v| + qw·|ω|. 오차는 걸음마다 decay = exp(−dt/tau) 로 0 쪽으로 (tau 잠정 — 지도에 다시 붙음)
// 움직임 판정은 IMU 와 같다 (imu.h imu_moving).
#pragma once
#include <cmath>
#include <cstdint>

#include "imu.h"

namespace sim {

constexpr int kPoseErrWinSteps = 5;   // 맞춘 창: 0.1 s = 정책 한 걸음 (물리 걸음 5 개)

struct PoseErrModel {
  float p_move, p_rest;        // 물리 걸음마다 보정이 일어날 확률
  float s0[3];                 // 정지 보정 잡음 σ (앞 m, 옆 m, yaw rad)
  float q1[3], qv[3], qw[3];   // 움직일 때 보정 잡음 분산 = q1 + qv·|v| + qw·|ω|
  float drift_f[3];            // 그중 이어지는 표류 몫 (0~1)
  float tau_d;                 // 표류 속도 상관 시간 (s)
  float tau;                   // 오차가 0 으로 돌아가는 시간 (s)
  // 호스트가 pose_err_set_dt 로 정함 (장치에 결정적 exp 가 없어서)
  float dt, decay, phi_d, jump_k[3], drift_k[3];
};

inline void pose_err_set_dt(PoseErrModel& m, float dt) {
  m.dt = dt;
  m.decay = m.tau > 0.f ? (float)std::exp(-(double)dt / (double)m.tau) : 0.f;
  m.phi_d = m.tau_d > 0.f ? (float)std::exp(-(double)dt / (double)m.tau_d) : 0.f;
  for (int k = 0; k < 3; ++k) {
    m.jump_k[k] = (float)std::sqrt(1.0 - m.drift_f[k]);
    m.drift_k[k] = (float)std::sqrt((double)m.drift_f[k] * m.p_move / (kPoseErrWinSteps * (double)dt * dt) * (1.0 - (double)m.phi_d * m.phi_d));
  }
}

// 한 걸음: 오차 e (ex, ey 세계, eth), 표류 속도 d (앞, 옆, yaw — 로봇 기준). th = 참값 방향, v·w = 몸통 속도. 난수는 늘 같은 개수
DEV void pose_err_step(const PoseErrModel& m, float* e, float* d, float th, float v, float w, uint64_t& rng) {
  const bool mv = imu_moving(v, w);
  const float u = dm::rand01(rng);
  const bool jump = u < (mv ? m.p_move : m.p_rest);
  float n[3];
  for (int k = 0; k < 3; ++k) {
    const float s = mv ? sqrtf((m.q1[k] + m.qv[k] * dm::absf(v)) + m.qw[k] * dm::absf(w)) : m.s0[k];
    const float g1 = imu_gauss(rng), g2 = imu_gauss(rng);
    n[k] = jump ? (mv ? m.jump_k[k] : 1.f) * s * g1 : 0.f;
    d[k] = mv ? m.phi_d * d[k] + m.drift_k[k] * s * g2 : m.phi_d * d[k];   // 멈추면 표류도 잦아듦
    n[k] = n[k] + d[k] * m.dt;
  }
  float sn, cs;
  dm::sincosf_d(th, &sn, &cs);
  e[0] = e[0] * m.decay + (cs * n[0] - sn * n[1]);
  e[1] = e[1] * m.decay + (sn * n[0] + cs * n[1]);
  e[2] = e[2] * m.decay + n[2];
}

}  // namespace sim
