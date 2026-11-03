// 탐사 RL 커리큘럼 + 판 끝 코드 (값은 코드 안, mdp/README "커리큘럼"·"판 끝"). 호스트·장치 공용.
//
// 판 끝 (done 코드, 러너는 terminal() 로 "종료(뒤 가치 0)" 와 "끊김(가치로 메움)" 을 가른다)
//   0 진행 · 1 선언 성공(선언 때 벽 덮음 ≥ 95 %) · 2 벽 충돌 · 3 끊김(안전 시간) · 4 선언 실패(기준 미달) · 5 사람 충돌
//   1·2·4·5 = 종료, 3 = 끊김 (rsl_rl time_outs, IsaacLab is_finite_horizon=False 와 같은 쓰임).
//   한 걸음에 충돌과 선언이 같이 오면 충돌.
//
// 단계 (환경마다 하나, legged_gym terrain curriculum 꼴: legged_gym/envs/base/legged_robot.py _update_terrain_curriculum —
//   환경마다 지형 단계를 두고 판 끝에 잘했으면 올리고 못했으면 내림. 여기선 "잘함" = 최근 kWindow 판 성공률)
//   단계 s 의 맵 = 갈 수 있는 바닥 넓이가 kStages[s].max_area 이하이고 kStages[s−1].max_area 보다 큰 맵 (없으면 아래 단계 맵으로 내려감).
//   오차 배율 = 지도 자세 오차(OU σ)·자이로 잡음에 곱함. 사람 수 = 판마다 0 ~ max_people 명(people.h k_max, 커리큘럼이 켜져 있으면
//   Env::set_people_max 대신 이 값).
//   오르내림: 판 끝마다 결과를 환경의 최근 kWindow 판 기록에 넣고, kWindow 판이 차면 성공률 > kUp 이면 올림, < kDown 이면 내림, 그리고 기록을 비움.
//   잠정 값: kUp 0.7 · kDown 0.3 · kWindow 8 (8 판이면 0.7 초과 = 6 판 이상 성공, 0.3 미만 = 2 판 이하).
//   넓이 경계 30·80·150 m² = 학습 맵 묶음(S2, scenes/train/list.tsv grade 열)의 등급 경계와 같게 (등급별 143·143·142·92 맵).
//   목록 파일에 등급 열이 있으면 그 값을 단계로 쓴다(env_host.cpp load_scene_list).
#pragma once
#include <cstdint>

#include "detmath.h"

namespace mdp::cur {

enum Done : int32_t { kRunning = 0, kDeclareOk = 1, kCollision = 2, kTruncated = 3, kDeclareShort = 4, kCollisionPerson = 5, kNumDone = 6 };
DEV bool terminal(int32_t d) { return d != kRunning && d != kTruncated; }   // 종료 (뒤 가치 0)

struct Stage {
  float max_area;    // 이 단계 맵의 갈 수 있는 바닥 넓이 상한 (m²)
  float err_scale;   // 자세 오차·자이로 잡음 배율
  int max_people;    // 판마다 사람 0 ~ 이 수 (sim people.h)
};
constexpr int kNumStages = 4;
constexpr Stage kStages[kNumStages] = {
    {30.f, 0.5f, 0},     // 0: 작은 방 (방 1~2 개), 오차 절반, 사람 없음
    {80.f, 0.75f, 1},    // 1: 작은 집, 사람 0~1
    {150.f, 1.0f, 2},    // 2: AWS 작은 집·서점 크기(148·145 m²), 실측 오차, 사람 0~2
    {1e9f, 1.25f, 3},    // 3: 큰 집 (25.6 m 지도 한도까지), 오차 넉넉히, 사람 0~3
};
// 장치 코드용 (constexpr 배열은 장치에서 못 읽음 — 같은 값)
DEV float err_scale_of(int s) {
  return s <= 0 ? kStages[0].err_scale : (s == 1 ? kStages[1].err_scale : (s == 2 ? kStages[2].err_scale : kStages[3].err_scale));
}
DEV int people_of(int s) {
  return s <= 0 ? kStages[0].max_people : (s == 1 ? kStages[1].max_people : (s == 2 ? kStages[2].max_people : kStages[3].max_people));
}
constexpr int kWindow = 8;
constexpr float kUp = 0.7f, kDown = 0.3f;

// 넓이 → 맵 단계 (호스트)
inline int stage_of_area(float area_m2) {
  for (int s = 0; s < kNumStages; ++s)
    if (area_m2 <= kStages[s].max_area) return s;
  return kNumStages - 1;
}

// 판 끝: 기록에 넣고 단계를 고침. hist = 최근 결과 비트(1 = 성공), n = 기록 수. max_stage = 맵이 있는 가장 높은 단계
DEV void on_episode_end(int32_t* stage, uint32_t* hist, int32_t* n, bool success, int max_stage) {
  *hist = (*hist << 1) | (success ? 1u : 0u);
  *n += 1;
  if (*n < kWindow) return;
  int k = 0;
  for (int i = 0; i < kWindow; ++i) k += (int)((*hist >> i) & 1u);
  const float rate = (float)k / (float)kWindow;
  if (rate > kUp && *stage < max_stage) *stage += 1;
  else if (rate < kDown && *stage > 0) *stage -= 1;
  *hist = 0u;
  *n = 0;
}

}  // namespace mdp::cur
