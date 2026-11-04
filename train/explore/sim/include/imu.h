// IMU 측정 모델 (섀시 IMU, /imu 100 Hz — 학습 환경은 물리 걸음 50 Hz 마다). 호스트·장치 공용, CPU == GPU 비트 같음(detmath).
// 값과 근거: config/sensor_model.yaml imu, analysis/real_limo.md IMU 절, 맞춤·확인 도구 tools/imu_check.
//
// 2D 시뮬이라 몸통 기울기·가속은 구동 모델이 낸 몸통 속도(v, ω)와 그 변화(a)에서 만들고, 나머지는 실측 잡음·진동 모델로 채운다.
//   roll  = roll0  + b_roll · v·ω          (원심 가속에 몸통·자세 추정이 기움. roll0·pitch0 = 판마다 바닥 기울기 N(0, tilt_sigma))
//   pitch = pitch0 + b_pitch · a
//   ax = k_ax · (a − vy·ω)   − g · pitch   (소각: 중력이 기울기만큼 앞뒤로 섞임, 실측 정지 평균이 이 식과 1 mrad 안)
//   ay = k_ay · (v·ω + a_y) + g · roll
//   vy·a_y 는 메카넘 옆 이동(몸 옆 속도·옆 가속) — 실물 옆 이동 녹화가 정상 응답이 아니라(바퀴 확인 필요) 배율은 앞뒤·원심 값을 그대로 씀 (잠정 📏)
//   az = g                                  (실측 정지 크기 10.03 m/s² — 장치 배율 몫 포함)
//   gx, gy = 판마다 바이어스
// 각 통로에 잡음 n = σ(v, ω) · u. 정지면 σ = s0, 움직이면 분산 σ² = q1 + qv·|v| + qw·|ω| (주행 진동은 바퀴가 돌면 바로 생기고
// 속도와 함께 큼; 분산이 속도에 선형이면 실측 잔차 제곱의 최소제곱이 치우침 없다. 움직임 = |v| > kImuMoveV 또는 |ω| > kImuMoveW).
// u 는 상관 시간 tau 의 단위 AR(1) (실측 잔차의 0.1 s 자기상관으로 정함 — 정책은 0.1 s 마다 한 번 본다).
// 정책은 0.1 s 마다 한 번 본다).
// REP-103: x 앞 · y 왼쪽 · z 위, roll 은 x 축 · pitch 는 y 축 회전. 실물 드라이버가 내는 쿼터니언도 이 규약대로다(real_limo.md).
#pragma once
#include <cmath>
#include <cstdint>

#include "detmath.h"

namespace sim {

constexpr int kImuCh = 7;   // ax, ay, az, gx, gy, roll, pitch (ImuState::out 순서)

struct ImuChannel {
  float s0;           // 정지 σ
  float q1, qv, qw;   // 움직일 때 분산 σ² = q1 + qv·|v| + qw·|ω|
  float tau;          // AR(1) 상관 시간 (s), 0 이면 흰 잡음
  float phi;          // 걸음당 AR 계수 exp(−dt/tau) — 호스트가 imu_set_dt 로 정함 (장치에 결정적 exp 가 없어서)
};

struct ImuModel {
  float g;                    // 정지 가속도 크기 (m/s²)
  float k_ax, k_ay;           // 몸통 가속 a · 원심 v·ω → 가속도 배율
  float b_pitch, b_roll;      // a → pitch (rad per m/s²), v·ω → roll (rad per m/s²)
  float tilt_sigma;           // 판마다 바닥 기울기 roll0·pitch0 σ (rad)
  float gyro_bias;            // 판마다 자이로 x·y 바이어스 ± (rad/s)
  ImuChannel ch[kImuCh];
};

struct ImuState {
  float u[kImuCh];            // 단위 AR(1) 잡음 상태
  float roll0, pitch0, bgx, bgy;
  float out[kImuCh];          // 이번 걸음 측정 (ax, ay, az, gx, gy, roll, pitch)
};

// 결정적 몫 (잡음 빼고) — 확인 도구가 실측 잔차를 낼 때도 같은 식을 쓴다. vy·ay: 메카넘 옆 속도·옆 가속 (차동이면 0)
DEV void imu_mean(const ImuModel& m, const ImuState& s, float v, float w, float a, float* mean, float vy = 0.f, float ay = 0.f) {
  const float c = v * w;
  const float roll = s.roll0 + m.b_roll * c, pitch = s.pitch0 + m.b_pitch * a;
  mean[0] = m.k_ax * (a - vy * w) - m.g * pitch;
  mean[1] = m.k_ay * (c + ay) + m.g * roll;
  mean[2] = m.g;
  mean[3] = s.bgx;
  mean[4] = s.bgy;
  mean[5] = roll;
  mean[6] = pitch;
}

constexpr float kImuMoveV = 0.002f, kImuMoveW = 0.002f;   // 실물 바퀴 오도메트리 단위(1 mm/s · 0.001 rad/s) 위
DEV bool imu_moving(float v, float w) { return dm::absf(v) > kImuMoveV || dm::absf(w) > kImuMoveW; }
DEV float imu_sigma(const ImuChannel& c, float v, float w) {
  return imu_moving(v, w) ? sqrtf((c.q1 + c.qv * dm::absf(v)) + c.qw * dm::absf(w)) : c.s0;
}

// 표준 정규 (Box-Muller, lidar_real.h gauss 와 같은 식 — 이 헤더만으로도 쓰게 따로 둠)
DEV float imu_gauss(uint64_t& r) {
  const float u1 = 1.f - dm::rand01(r), u2 = dm::rand01(r);
  float sn, cs;
  dm::sincosf_d(6.28318531f * u2, &sn, &cs);
  return sqrtf(-2.f * dm::lnf_d(u1)) * cs;
}

// 호스트: 걸음 dt 로 AR 계수를 정한다 (설정 읽은 뒤 한 번)
inline void imu_set_dt(ImuModel& m, float dt) {
  for (int k = 0; k < kImuCh; ++k) m.ch[k].phi = m.ch[k].tau > 0.f ? (float)std::exp(-(double)dt / (double)m.ch[k].tau) : 0.f;
}

// 판 시작: 바닥 기울기·바이어스를 뽑고 잡음 상태를 정상 분포에서 시작
DEV void imu_reset(const ImuModel& m, ImuState& s, uint64_t& rng) {
  s.roll0 = m.tilt_sigma * imu_gauss(rng);
  s.pitch0 = m.tilt_sigma * imu_gauss(rng);
  s.bgx = dm::rand_range(rng, -m.gyro_bias, m.gyro_bias);
  s.bgy = dm::rand_range(rng, -m.gyro_bias, m.gyro_bias);
  for (int k = 0; k < kImuCh; ++k) { s.u[k] = imu_gauss(rng); s.out[k] = 0.f; }
}

// 한 걸음 (imu_set_dt 의 dt): 몸통 속도 v, ω 와 몸통 직진 가속 a (+ 옆 속도 vy·옆 가속 ay) 에서 측정을 out 에.
// 진동 σ 는 바퀴가 도는 빠르기로: 옆 이동도 바퀴를 같은 만큼 돌려 |v| + |vy|
DEV void imu_step(const ImuModel& m, ImuState& s, float v, float w, float a, uint64_t& rng, float vy = 0.f, float ay = 0.f) {
  float mean[kImuCh];
  imu_mean(m, s, v, w, a, mean, vy, ay);
  const float vs = dm::absf(v) + dm::absf(vy);
  for (int k = 0; k < kImuCh; ++k) {
    const ImuChannel& c = m.ch[k];
    const float n = imu_gauss(rng);
    if (c.phi > 0.f) {
      s.u[k] = c.phi * s.u[k] + sqrtf(1.f - c.phi * c.phi) * n;
    } else {
      s.u[k] = n;
    }
    s.out[k] = mean[k] + imu_sigma(c, vs, w) * s.u[k];
  }
}

}  // namespace sim
