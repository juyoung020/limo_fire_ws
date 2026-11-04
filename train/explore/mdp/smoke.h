// 탐사 RL 연기 모드 — 연기 속에서 정책이 무엇을 보고 어떻게 움직이나. 시뮬(sim/include/env.h)·실물 노드(limo_fire_rl)가 같은 규칙을 쓴다.
// 호스트·장치 공용(detmath.h DEV). 연기 표시 e.smoke 가 0 이면 아무 일도 안 한다 → 지금 학습과 비트 같음.
//
// 연기 속에서 쓰는 센서 (지금 로봇에 달린 것만 — docs/hardware.md. 레이더·초음파·열화상은 아직 안 달아 다루지 않는다)
//
//   센서                     연기 속            정책 입력                                    여기서
//   ---------------------- ------------------ -------------------------------------------- ------------------------
//   바퀴 오도메트리            빛을 안 써 영향 없음   WHEEL_V · WHEELS4 · SLIP · EKF               그대로
//   IMU                       빛을 안 써 영향 없음   GYRO_Z · ACCEL · IMU_TILT · SLIP · EKF        그대로
//   2D 라이다                 연기 입자에 반사      LIDAR 4 장 · INTENSITY                       0 으로 가림
//   깊이 카메라 (Dabai DC1)    근적외선 패턴이 가림   깊이 1 × 40 × 64                              0 으로 가림
//   컬러 카메라 → YOLO        가림                지도 물체 채널 (지금도 📏 0)                  0 그대로
//   라이다 SLAM 자세 (Cartographer)  라이다를 씀   POSE · HOME · 지도 그리는 자리                 바퀴 + IMU 추측 항법으로 바꿈
//   라이다 안전층              라이다를 씀          (행동을 막음)                                 끔 → 대신 속도 한도
//
//   근거: Starr & Lattimer, "Evaluation of Navigation Sensors in Fire Smoke Environments", Fire Technology 50, 2014,
//   doi:10.1007/s10694-013-0356-3 — 항법 센서 11 종을 연기 속에서 비교: 레이더·적외선(열) 카메라가 가장 좋았고, LIDAR 는 시정 4 m 아래에서
//   거리 오차, 깊이 카메라(Kinect — Dabai DC1 처럼 근적외선 패턴을 쏘는 방식)·가시광 카메라도 시험 대상. 연기 농도별로 라이다를 어디까지
//   믿을지 잴 센서가 지금 없어 부분 사용 없이 통째로 뺀다.
//
// 지도: 연기 전에 쌓은 지도는 센서가 아니라 기억이라 남긴다. 연기 속에서는 더 쌓지 않고(라이다를 안 씀) 그대로 둔 채, 추측 항법 자세에
//   맞춰 가까운 지도·먼 지도를 그린다. 지도에서 셈하는 값(안 넓이·프론티어·진행)도 이 고정된 지도로 셈한다. "지나간 곳" 은 자세로 그리므로 계속 쌓는다.
//
// 연기 표시: 정책 1D 숫자 칸 SMOKE (obs_layout.h, 예전 예약 칸 — 입력 크기 그대로라 체크포인트에서 이어 학습).
//
// 아직 정하지 않은 것 (이 헤더 밖):
//   연기 속 과제·보상 — 라이다를 안 쓰면 벽 덮음(보상의 주 항)이 늘지 않는다. 시뮬은 연기 동안 덮음·진행 없음 끝을 멈추기만 한다(env.h).
//   언제 연기가 끼나 — 시뮬에서 켜는 쪽은 Env::set_smoke(환경마다). 실물은 연기 모드 신호(서비스·토픽)를 받는 노드가 켠다.
#pragma once
#include "detmath.h"
#include "obs_layout.h"

namespace mdp::smoke {

// ---- 속도 한도 (라이다 안전층 대신) — 잠정 ----
// 앞을 못 보므로 부딪혀도 멈출 수 있는 속도로. 안전층 제동 거리 실측(mdp/README "정할 것" 2: 1 m/s 에서 0.326 m)을 속도 제곱으로 줄이면
// 0.3 m/s 에서 약 3 cm. 연기 속 Nav2 한도로 잡았던 0.13 m/s 보다 빠르지만 정책이 스스로 줄일 수 있다. 실물 충돌 시험 뒤 정한다 📏
constexpr float kVFwd = 0.3f;    // m/s
constexpr float kVBack = 0.15f;  // m/s, 뒤는 원래도 라이다가 못 봐 앞의 절반 (obs_layout.h V_BACK 0.3 / V_FWD 1.0 보다 덜 줄임)
constexpr float kVy = 0.3f;      // m/s
constexpr float kW = 1.0f;       // rad/s, Nav2 회전 한도와 같게

// ---- 추측 항법 자세 오차 — 잠정 (바퀴 오도메트리 주행 오차는 아직 안 쟀다: docs/hardware.md "확인할 것") ----
// 연기 동안 지도 자세 = 들어갈 때 지도 자세 ∘ EKF(바퀴 + IMU) 이동. 오차는 이동마다 쌓인다(분산 ∝ 이동량, 오도메트리 오차의 흔한 꼴):
//   방향 오차 eth += kYawPerSqrtRad·√|dθ|·g₀ + kYawPerSqrtM·√ds·g₁
//   위치 오차 = 이번 이동을 지도 자세 방향 오차만큼 돌린 몫 + kXyPerSqrtM·√ds·g₂ (가로·세로 따로)
//   IMU 자이로 바이어스 실측 0.00005 rad/s 라 방향은 회전량 몫이 주로 남는다.
constexpr float kYawPerSqrtRad = 0.01f;   // rad / √rad
constexpr float kYawPerSqrtM = 0.005f;    // rad / √m (메카넘 바퀴 미끄러짐)
constexpr float kXyPerSqrtM = 0.02f;      // m / √m

// 행동: 연기 속이면 물리 단위 명령을 한도로 자른다 (안전층은 끈다 — env.h apply_action, 실물 노드도 같게)
DEV void cap_command(int smoke, float* v, float* vy, float* w) {
  if (!smoke) return;
  *v = dm::maxf(-kVBack, dm::minf(kVFwd, *v));
  *vy = dm::maxf(-kVy, dm::minf(kVy, *vy));
  *w = dm::maxf(-kW, dm::minf(kW, *w));
}
DEV bool lidar_safety(int smoke) { return !smoke; }

// 물리 한 걸음의 참 이동 (세계 dx, dy, dθ) → 지도 자세 오차 (ex, ey, eth) 갱신. g0~g3 = 표준 정규 난수 (부르는 쪽이 뽑음 — 결정적)
DEV void drift_step(float* ex, float* ey, float* eth, float dx, float dy, float dth, float err_scale, float g0, float g1,
                    float g2, float g3) {
  const float ds = sqrtf(dx * dx + dy * dy);
  const float sds = sqrtf(ds), sdt = sqrtf(dm::absf(dth));
  // 지도 자세의 이동 = 참 이동을 지금 방향 오차만큼 돌린 것 → 오차 증가 = R(eth)·d − d
  float s, c;
  dm::sincosf_d(*eth, &s, &c);
  const float rx = (c * dx - s * dy) - dx, ry = (s * dx + c * dy) - dy;
  *ex = *ex + rx + (kXyPerSqrtM * err_scale) * sds * g2;
  *ey = *ey + ry + (kXyPerSqrtM * err_scale) * sds * g3;
  *eth = *eth + (kYawPerSqrtRad * err_scale) * sdt * g0 + (kYawPerSqrtM * err_scale) * sds * g1;
}

// ---- 관측 가리기 ----
// 정책 1D (POLICY_1D): 라이다 4 장·세기를 0 으로, SMOKE 칸에 표시. 숫자 칸은 obs_finish 가 이미 추측 항법 자세로 셈한 값 그대로.
// 실물 노드도 같은 함수로 가린다(관측을 다 만든 뒤 마지막에).
DEV void mask_policy_1d(int smoke, float* O) {
  using namespace mdp::obs;
  if (!smoke) return;
  for (int i = 0; i < LIDAR + INTENSITY; ++i) O[i] = 0.f;
  O[LIDAR + INTENSITY + SMOKE] = 1.f;
}
// 깊이 픽셀 하나: 연기 속이면 0 (env.h 깊이 규칙에서 0 = 구멍 — 정책은 SMOKE 칸으로 "전부 구멍" 이 연기인지 안다)
DEV uint8_t depth_px(int smoke, uint8_t v) { return smoke ? (uint8_t)0 : v; }
// 지도 쌓기: 연기 속이면 라이다로 지도·덮음을 갱신하지 않는다
DEV bool map_update(int smoke) { return !smoke; }

}  // namespace mdp::smoke
