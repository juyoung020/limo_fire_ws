// 탐사 RL 커리큘럼 + 판 끝 코드 (값은 코드 안, mdp/README "커리큘럼"·"판 끝"). 호스트·장치 공용.
//
// 판 끝 (done 코드, 러너는 terminal() 로 "종료(뒤 가치 0)" 와 "끊김(가치로 메움)" 을 가른다)
//   0 진행 · 1 선언 성공(선언 때 벽 덮음 ≥ 95 %) · 2 벽 충돌 · 3 끊김(안전 시간) · 4 선언 실패(기준 미달) · 5 사람 충돌
//   · 6 진행 없음(지도에서 새 칸을 mdp::cur::kIdleTime 동안 못 봄 — 종료, rewards.h '진행 없음')
//   1·2·4·5·6 = 종료, 3 = 끊김 (rsl_rl time_outs, IsaacLab is_finite_horizon=False 와 같은 쓰임).
//   한 걸음에 충돌과 선언이 같이 오면 충돌.
//
// 단계 (환경마다 하나, legged_gym terrain curriculum 꼴: legged_gym/envs/base/legged_robot.py _update_terrain_curriculum —
//   환경마다 지형 단계를 두고 판 끝에 잘했으면 올리고 못했으면 내림. 여기선 "잘함" = 최근 kWindow 판의 끝 벽 덮음 평균 — 진행량)
//   단계 s 의 맵 = 갈 수 있는 바닥 넓이가 kStages[s].max_area 이하이고 kStages[s−1].max_area 보다 큰 맵 (없으면 아래 단계 맵으로 내려감).
//   오차 배율 = 지도 자세 오차(OU σ)·자이로 잡음에 곱함. 사람 수 = 판마다 0 ~ max_people 명(people.h k_max, 커리큘럼이 켜져 있으면
//   Env::set_people_max 대신 이 값).
//   오르내림(진행량, legged_gym/envs/base/legged_robot.py:421-440 _update_terrain_curriculum 꼴 — 거기선 판마다 걸어간 거리가 지형 길이의
//   절반을 넘으면 올리고, 요구 거리의 절반 아래면 내림): 판 끝마다 끝 벽 덮음을 환경의 최근 kWindow 판 기록에 더하고, kWindow 판이 차면
//   덮음 평균 ≥ kProgUp 이면 올림, < kProgDown(= kProgUp 의 절반) 이면 내림, 그리고 기록을 비움. 한 판 운으로 오르지 않게 창으로 본다.
//   값(2026-11-04 비교, ppo/README "커리큘럼 비교"): kWindow 8 · kProgUp 0.8 · kProgDown 0.4. 옛 규칙(최근 8 판 성공 6 이상 올림)은 학습 성공률
//   0.4 근처에서 거의 못 올라 단계 0 에 갇혔고(시험 맵 성공 0), 판마다 바로 올리는 진행량 규칙은 씨앗에 따라 판을 끝내지 않는 쪽으로 무너졌다.
//   넓이 경계 30·80·150 m² = 학습 맵 묶음(S2, scenes/train/list.tsv grade 열)의 등급 경계와 같게 (등급별 143·143·142·92 맵).
//   목록 파일에 등급 열이 있으면 그 값을 단계로 쓴다(env_host.cpp load_scene_list).
#pragma once
#include <cstdint>

#include "detmath.h"

namespace mdp::cur {

enum Done : int32_t { kRunning = 0, kDeclareOk = 1, kCollision = 2, kTruncated = 3, kDeclareShort = 4, kCollisionPerson = 5, kNoProgress = 6,
                      kNumDone = 7 };
DEV bool terminal(int32_t d) { return d != kRunning && d != kTruncated; }   // 종료 (뒤 가치 0)

// ============================================================================================================
// 판 끝 조건 (사람이 바꾸는 값 — 보상 점수는 rewards.h)
// ============================================================================================================
constexpr float kSafeTime = 1200.f;   // 안전 시간 (s): 이만큼 지나면 끊김(3, 가치로 메움). 맵과 무관. frontier_nav2 가장 긴 판 1040 s 보다 넉넉히
constexpr float kIdleTime = 60.f;     // 진행 없음 (s): 새 벽 면 칸을 이만큼 못 보면 종료(6) + rewards.h kIdlePenalty
constexpr float kSuccessCov = 0.95f;  // 성공 기준: 끝 선언 때 벽 덮음 ≥ 이것이면 선언 성공(1), 아래면 선언 실패(4) (mdp/README)
//   진행 없음 끝을 둔 까닭: 끊김(안전 시간)은 time-out 이라 V 로 메워, 버티는 판의 값은 걸음당 시간 감점 할인 합
//   (작은 맵 ≈ 0.0004/(1−γ) ≈ 0.14)뿐 — 충돌(−10)·미달 선언(−3)보다 훨씬 싸서 '판을 안 끝내는' 붕괴가 남(cur-fix (ii) 씨앗 2, B·C 실험 씨앗 1 모두).
//   출처: Go-Explore (Ecoffet et al., Nature 2021, arXiv 2004.12919 Methods) "we terminate the episode early … if no new cells are
//   discovered for 1,000 steps (while exploring)".
//   kIdleTime 근거: (ii-a) 씨앗 3 반복 300 체크포인트, 학습 맵 15,318 판(학습식 표집)에서 판 최대 '새 벽 없는 간격' ≥ 60 s 는
//   성공 0.4 %·미달 선언 0.5 %·충돌 0.3 %, 시험 맵에서 1,200 s 끊긴 판은 28/28 이 걸림 (diag 측정).

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
constexpr float kProgUp = 0.8f, kProgDown = 0.4f;   // 창 평균 끝 덮음 문턱 (위 머리말)

// 넓이 → 맵 단계 (호스트)
inline int stage_of_area(float area_m2) {
  for (int s = 0; s < kNumStages; ++s)
    if (area_m2 <= kStages[s].max_area) return s;
  return kNumStages - 1;
}

// 판 끝: 끝 덮음 cov 를 기록에 더하고(hist = 덮음×1000 합, n = 기록 수) kWindow 판이 차면 단계를 고침. max_stage = 맵이 있는 가장 높은 단계
DEV void on_episode_end(int32_t* stage, uint32_t* hist, int32_t* n, float cov, int max_stage) {
  *hist += (uint32_t)(cov * 1000.f + 0.5f);
  *n += 1;
  if (*n < kWindow) return;
  const float avg = (float)*hist / (1000.f * (float)kWindow);
  if (avg >= kProgUp && *stage < max_stage) *stage += 1;
  else if (avg < kProgDown && *stage > 0) *stage -= 1;
  *hist = 0u;
  *n = 0;
}

}  // namespace mdp::cur
