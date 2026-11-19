// 아래층 지역 계획기 설정 (include/local_planner.h · env.yaml action.path.local_planner). EnvCfg::lp 에 담김 — 호스트·장치 공용 평범한 구조체.
// on = 0 이면 지역 계획기를 안 씀 (옛 RPP 추종기 그대로, 비트 같음).
#pragma once
#include <cstdint>

namespace sim {

constexpr int kLpV = 8;     // 목표 속도 마디 상한
constexpr int kLpK1 = 16;   // 앞 곡률 마디 상한
constexpr int kLpK2 = 8;    // 뒤 곡률 마디 상한
constexpr int kLpN = 40;    // 지평 걸음 상한

struct LpCfg {
  int32_t on;            // 1 = path_step_a 가 RPP 추종기 대신 지역 계획기 (도착·목표 없음·경로 없음은 추종기 그대로)
  int32_t n_steps;       // 지평 걸음 수 (행동 주기 0.1 s 단위)
  int32_t k_switch;      // 앞 곡률 → 뒤 곡률 바꾸는 걸음
  int32_t block_steps;   // 가장 좋은 궤적도 이 걸음 안에 부딪히면 막힘 (기존 충돌 검사 막힘 가지 — 물러나기·복구)
  int32_t rot_on;        // 1 = 제자리 회전 궤적 둘도 묶음에 넣음
  int32_t bumper_scaled; // 1 = 앞 범퍼 정지 문턱(safety_dist, 1 m/s 정지 거리 기준)을 지금 명령 v / path_v 배로
  int32_t mon_real_w;    // 1 = collision_monitor 가 굴리는 ω 를 실제 ω (명령 × turn_gain(|v|)) 로
  int32_t n_v, n_k1, n_k2;
  float v[kLpV], k1[kLpK1], k2[kLpK2];   // 목표 속도 (m/s) · 곡률 (1/m)
  int32_t ring;          // 1 = 직전 라이다 장들(0.1~0.4 s 전)도 거리 장에 (뒤 사각)
  int32_t ppl;           // 1 = 움직이는 점 등속 예측 (지도가 빈 칸으로 아는 곳에 맞은 지금 장 점 ↔ 가장 오래된 링 장 점 최근접 짝 → 속도)
  float ppl_r, ppl_hard_t, ppl_vmin, ppl_vmax, w_ppl;   // 예측 점까지 가운데 틈 하한 (m) · 그 안이면 충돌로 보는 시간 (s, 넘으면 비용) · 움직임 문턱·상한 (m/s) · 비용 무게
  float ring_range;      // 그 장들에서 이 거리(센서에서 m) 안 점만
  float slow_dist, slow_min;   // 막힘 곁 감속: 가운데 틈 < slow_dist 면 목표 속도 ≤ max(slow_min, path_v · 틈 / slow_dist). slow_dist 0 = 끔
  float lag_mix;         // 몸통 속도 = lag_mix · 새 명령 + (1 − lag_mix) · 지난 몸통 (한 걸음 늦게 적용 — 섀시 지연 60–80 ms + 1 차)
  float center_min;      // 가운데 점에서 막힘까지 하한 (m) — 설 수 있는 칸 안에 머묾
  float margin;          // 단단한 충돌: 원 둘 발자국 반지름 + margin
  float r_soft;          // 틈 비용 거리 (m, 단단한 반지름 위로 더)
  float fwd_off;         // 거리 장 창 가운데를 로봇 앞으로 미는 거리 (m)
  float w_prog, w_v, w_dev, dev_free, w_head, w_clr, w_hys, w_rot, w_flip;
};

}  // namespace sim
