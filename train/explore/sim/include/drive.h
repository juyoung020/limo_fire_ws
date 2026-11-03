// 바닥판 구동 모델 (실물 LIMO 녹화로 맞춤, tools/motion_check · config/env.yaml drive). 호스트·장치 공용.
//   명령(v, ω) → 지연(delay 걸음) → 바퀴 속도 = (1 − slow) · 빠른 갈래 + slow · 느린 갈래, 각 갈래는 모터 가속 한도 안에서
//   1 차로 따라감(시간상수 tau · tau_slow) — 실물은 명령을 받으면 0.04 s 에 목표의 ~87 % 까지 튀고 나머지를 0.2 s 에 걸쳐 채운다
//   → 몸통 속도는 바퀴 속도 / slip 를 1 차(Floor.tau)로, 바닥 마찰이 허락하는 가속(Floor.trac)까지만 따라감
// 세게 가속하면 바퀴가 몸통보다 앞서 헛돌고(바퀴 ω / 자이로 > 1), 일정해지면 따라잡는다 (사무실 바닥 실측: 명령으로
// 일정하게 돌면 0.99, 가속 중 1.08, 리모컨으로 세게 돌면 1.1~1.5). 바닥판 오도메트리(/odom)는 바퀴 속도를 낸다.
// 리모컨 녹화를 맞출 때는 스틱 값을 목표 속도로 바꾸는 표(rc_*)도 쓴다 — 정책은 /cmd_vel 로 바로 목표 속도를 준다.
#pragma once
#include "detmath.h"

namespace sim {

constexpr int kDelayMax = 16;   // 지연 최대 (dt 걸음)
constexpr int kRcPoints = 6;    // 스틱 |값| 0, 0.2, …, 1.0

struct DriveModel {
  int delay_v, delay_w;        // dt 걸음
  float tau_v, tau_w;          // 빠른 갈래 1 차 시간상수 (s), 0 이면 가속 한도만
  float slow_v, slow_w;        // 느린 갈래 몫 (0 이면 한 갈래)
  float tau_slow_v, tau_slow_w;   // 느린 갈래 시간상수 (s)
  float acc_v, acc_w;          // 가속 한도 (m/s², rad/s²)
  float v_max, w_max;          // 바퀴 최고 속도
  float rc_v[kRcPoints], rc_w[kRcPoints];   // 리모컨 스틱 |값| → 목표 속도 (부호는 스틱을 따름)
};

// 바닥 재질 (config/env.yaml floors): 일정할 때 바퀴 / 몸통 비, 몸통이 낼 수 있는 가속 (마찰)
struct Floor {
  float slip_v, slip_w;
  float trac_v, trac_w;        // m/s², rad/s²
  float tau_v, tau_w;          // s
};

struct DriveState {
  float buf_v[kDelayMax], buf_w[kDelayMax];
  int head;
  float wv, ww;                // 바퀴 속도 (오도메트리가 내는 값)
  float fv, fw, sv, sw;        // 빠른·느린 갈래
  float bv, bw;                // 몸통 속도
  // 옆 방향 (메카넘 vy) — 맞춘 녹화가 없어 앞뒤(v) 와 같은 응답으로 잠정 📏 (vy 명령 계단 녹화로 맞춤, mdp/README "행동")
  float buf_vy[kDelayMax];
  float wvy, fvy, svy, bvy;
};

DEV void drive_reset(DriveState& s) {
  for (int i = 0; i < kDelayMax; ++i) { s.buf_v[i] = 0.f; s.buf_w[i] = 0.f; s.buf_vy[i] = 0.f; }
  s.head = 0;
  s.wv = s.ww = s.bv = s.bw = s.fv = s.fw = s.sv = s.sw = 0.f;
  s.wvy = s.fvy = s.svy = s.bvy = 0.f;
}

// 리모컨 스틱 → 목표 속도 (표 사이 선형)
DEV float rc_target(const float* table, float stick) {
  const float a = dm::minf(1.f, dm::absf(stick)) * (float)(kRcPoints - 1);
  int i = (int)a;
  if (i >= kRcPoints - 1) i = kRcPoints - 2;
  const float f = a - (float)i;
  const float v = table[i] + f * (table[i + 1] - table[i]);
  return stick < 0.f ? -v : v;
}

DEV float follow(float cur, float target, float tau, float acc, float dt) {
  float d = tau > 0.f ? (target - cur) * (dt / (tau + dt)) : (target - cur);
  const float lim = acc * dt;
  d = d > lim ? lim : (d < -lim ? -lim : d);
  return cur + d;
}

// 한 걸음: 명령을 넣고 바퀴 속도를 갱신, 몸통은 바닥 마찰 한도 안에서 바퀴를 따라감
// 옆 방향 한 걸음 (drive_step 보다 먼저 부른다 — 같은 head 칸을 씀). 앞뒤 v 와 같은 지연·시간상수·가속 한도·바닥 비 (잠정 📏)
DEV float drive_lat(const DriveModel& m, DriveState& s, float cmd_vy, float dt, const Floor& f) {
  s.buf_vy[s.head] = dm::maxf(-m.v_max, dm::minf(m.v_max, cmd_vy));
  const int kv = (s.head - m.delay_v + kDelayMax) % kDelayMax;
  s.fvy = follow(s.fvy, s.buf_vy[kv], m.tau_v, m.acc_v, dt);
  s.svy = follow(s.svy, s.buf_vy[kv], m.tau_slow_v, m.acc_v, dt);
  s.wvy = (1.f - m.slow_v) * s.fvy + m.slow_v * s.svy;
  s.bvy = follow(s.bvy, s.wvy / f.slip_v, f.tau_v, f.trac_v, dt);
  return s.bvy;
}

DEV void drive_step(const DriveModel& m, DriveState& s, float cmd_v, float cmd_w, float dt, const Floor& f, float* body_v,
                    float* body_w) {
  s.buf_v[s.head] = dm::maxf(-m.v_max, dm::minf(m.v_max, cmd_v));
  s.buf_w[s.head] = dm::maxf(-m.w_max, dm::minf(m.w_max, cmd_w));
  const int kv = (s.head - m.delay_v + kDelayMax) % kDelayMax, kw = (s.head - m.delay_w + kDelayMax) % kDelayMax;
  s.head = (s.head + 1) % kDelayMax;
  s.fv = follow(s.fv, s.buf_v[kv], m.tau_v, m.acc_v, dt);
  s.fw = follow(s.fw, s.buf_w[kw], m.tau_w, m.acc_w, dt);
  s.sv = follow(s.sv, s.buf_v[kv], m.tau_slow_v, m.acc_v, dt);
  s.sw = follow(s.sw, s.buf_w[kw], m.tau_slow_w, m.acc_w, dt);
  s.wv = (1.f - m.slow_v) * s.fv + m.slow_v * s.sv;
  s.ww = (1.f - m.slow_w) * s.fw + m.slow_w * s.sw;
  s.bv = follow(s.bv, s.wv / f.slip_v, f.tau_v, f.trac_v, dt);
  s.bw = follow(s.bw, s.ww / f.slip_w, f.tau_w, f.trac_w, dt);
  *body_v = s.bv;
  *body_w = s.bw;
}

}  // namespace sim
