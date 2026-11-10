// 탐사 RL 커리큘럼 + 판 끝 코드 (값은 코드 안, mdp/README "커리큘럼"·"판 끝"). 호스트·장치 공용.
//
// 판 끝 (done 코드, 러너는 terminal() 로 "종료(뒤 가치 0)" 와 "끊김(가치로 메움)" 을 가른다)
//   0 진행 · 1 선언 성공(선언 때 벽 덮음 ≥ 95 %) · 2 벽 충돌 · 3 끊김(안전 시간) · 4 선언 실패(기준 미달) · 5 사람 충돌
//   · 6 진행 없음(지도에서 판 처음 막힘이 된 칸이 한 행동에 kIdleOccMin 개 이상인 걸음이 kIdleTime 동안 없음 — 종료, rewards.h '진행 없음')
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
constexpr float kIdleTime = 60.f;     // 진행 없음 (s): 지도 진척(아래 kIdleOccMin)이 이만큼 없으면 종료(6) + rewards.h kIdlePenalty
constexpr int kIdleOccMin = 10;       // 진척 = 한 행동(0.1 s) 동안 판 처음 막힘(p > 0.65, frontier.h kOccL)이 된 지도 칸 ≥ 이것 (env.h map_progress)
constexpr float kSuccessCov = 0.95f;  // 성공 기준: 끝 선언 때 벽 덮음 ≥ 이것이면 선언 성공(1), 아래면 선언 실패(4) (mdp/README)
//   진행 없음 끝을 둔 까닭: 끊김(안전 시간)은 time-out 이라 V 로 메워, 버티는 판의 값은 걸음당 시간 감점 할인 합
//   (작은 맵 ≈ 0.0004/(1−γ) ≈ 0.14)뿐 — 충돌(−10)·미달 선언(−3)보다 훨씬 싸서 '판을 안 끝내는' 붕괴가 남(cur-fix (ii) 씨앗 2, B·C 실험 씨앗 1 모두).
//   출처: Go-Explore (Ecoffet et al., Nature 2021, arXiv 2004.12919 Methods) "we terminate the episode early … if no new cells are
//   discovered for 1,000 steps (while exploring)".
//   kIdleTime 근거: (ii-a) 씨앗 3 반복 300 체크포인트, 학습 맵 15,318 판(학습식 표집)에서 판 최대 '새 벽 없는 간격' ≥ 60 s 는
//   성공 0.4 %·미달 선언 0.5 %·충돌 0.3 %, 시험 맵에서 1,200 s 끊긴 판은 28/28 이 걸림 (diag 측정 — 그때 시계는 정답 벽 면 기준).
//   진척 시계를 정답 벽 면(since_wall)에서 지도(since_occ)로 바꾼 까닭 (2026-11-05): 정답 벽은 실물에 없어 정책이 볼 수 없었고(관측 시계는
//   지도 새 칸 기준이라 거의 늘 0 — 판 끝이 예고 없이 옴), 이제 판 끝 시계 == 관측 PROGRESS+1 시계라 실물에서도 같은 값을 센다.
//   since_wall 은 가치 전용 참값(obs_layout.h PRIV_IDLE)으로만 남긴다.
//   kIdleOccMin 근거 (idle_agree 일회성 측정: 학습 맵 목록, 256 환경 × 6000 걸음, 덩어리 따라가기 ½·무작위 ¼·거의 제자리 ¼, 정답 벽 시계로
//   진행 없음 끝난 604 판 · 다른 끝 6,310 판): 문턱 K 에서 지도 시계도 60 s 넘은 비율 / 다른 끝 판에서 넘은 판 —
//   K=1 1.3 % / 0 (멈춰도 지도 자세 오차 흐름·빔 잡음으로 막힘 칸이 드문드문 계속 생겨 시계가 안 감), K=5 77.8 % / 2, K=10 94.7 % / 8,
//   K=20 99.0 % / 16. 10 = 제자리를 거의 다 잡으면서 오발동이 적은 쪽. 지도 시계는 벽 시계보다 중앙 67 s 먼저 넘는다(벽 시계는 벽 칸 1 개에도
//   되돌아감) — 즉 같은 행동에서 판이 더 일찍 끝난다.

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

// 오르내림 규칙 (EnvCfg::cur_rule — env.yaml curriculum.rule)
enum CurRule : int32_t { kCurCoverage = 0, kCurSuccess = 1 };
constexpr int kCurWindowMax = 32;   // 성공 규칙 창 상한 (창 = cur_hist 비트)

// 판 끝 (덮음 규칙, 옛 규칙 그대로): 끝 덮음 cov 를 기록에 더하고(hist = 덮음×1000 합, n = 기록 수) window 판이 차면 단계를 고침. max_stage = 맵이 있는 가장 높은 단계
// top_random (기본 켬 — EnvCfg::cur_top_random, SIM_CUR_TOPRAND=0 이면 끔): 맨 위 단계에서 또 올림 조건을 넘으면 단계를 rnd % (max_stage + 1) 로 —
//   legged_gym/envs/base/legged_robot.py _update_terrain_curriculum "Robots that solve the last level are sent to a random one"
//   (torch.where(levels >= max_level, randint_like(levels, max_level), clip(levels, 0))). 근거: explore_v7 은 반복 ~5000 부터 단계 평균 2.9+ 로
//   거의 큰 맵(단계 3, 92 개)만 보고, 결정적 평가에서 단계 3 학습 맵 성공은 0.50 → 0.54 → 0.56(반복 3900 → 7000 → 14000)로 오르는데
//   시험 맵과 같은 크기의 단계 2 학습 맵은 0.37 → 0.37 → 0.34, 충돌 0.34 → 0.41 → 0.52 로 나빠졌다(맵 8 개 × 32 판, ppo/README "explore_v8 A/B").
// window·up·down 기본 = kWindow·kProgUp·kProgDown (env.yaml curriculum.coverage 가 같은 값이면 옛 빌드와 비트 같음)
DEV void on_episode_end(int32_t* stage, uint32_t* hist, int32_t* n, float cov, int max_stage, bool top_random = false, uint32_t rnd = 0u,
                        int window = kWindow, float up = kProgUp, float down = kProgDown) {
  *hist += (uint32_t)(cov * 1000.f + 0.5f);
  *n += 1;
  if (*n < window) return;
  const float avg = (float)*hist / (1000.f * (float)window);
  if (avg >= up && *stage < max_stage) *stage += 1;
  else if (avg >= up && top_random) *stage = (int32_t)(rnd % (uint32_t)(max_stage + 1));
  else if (avg < down && *stage > 0) *stage -= 1;
  *hist = 0u;
  *n = 0;
}

// 판 끝 (성공 규칙, 2026-11-09 — mdp/README "커리큘럼"): 최근 window 판(미끄러지는 창)의 성공(판 끝 코드 kDeclareOk = 지도 닫힘 성공) 비율.
//   hist = 최근 판 성공 비트(비트 0 이 가장 최근, window ≤ kCurWindowMax), n = 창에 든 판 수(≤ window).
//   창이 차면 판마다: 비율 ≥ up → 올림(맨 위면 top_random 규칙), 비율 < down → 내림, 단계가 바뀌면 창을 비움(새 단계 판만 셈). 그 사이면 그대로(창은 계속 미끄러짐).
//   근거: 덮음 규칙은 충돌로 끝난 판도 덮음만 넘으면 올렸다(성공 0.72 정체 때 단계 0~3 성공 0.94·0.79·0.73·0.66). legged_gym 은 판마다 결정하므로
//   창이 찬 뒤엔 판마다 보는 미끄러지는 창. 값과 출처는 env.yaml curriculum.success.
DEV void on_episode_end_success(int32_t* stage, uint32_t* hist, int32_t* n, bool success, int max_stage, bool top_random, uint32_t rnd,
                                int window, float up, float down) {
  const uint32_t mask = window >= 32 ? 0xffffffffu : ((1u << window) - 1u);
  *hist = ((*hist << 1) | (success ? 1u : 0u)) & mask;
  if (*n < window) *n += 1;
  if (*n < window) return;
  int k = 0;
  for (uint32_t h = *hist; h; h &= h - 1u) ++k;   // 성공 수 (비트 수)
  const int32_t s0 = *stage;
  // 문턱 비교는 정수로: k ≥ up·window ⇔ k·1000 ≥ round(up·1000)·window (부동소수 경계 흔들림 없이)
  const int up_m = (int)(up * 1000.f + 0.5f), down_m = (int)(down * 1000.f + 0.5f);
  if (k * 1000 >= up_m * window && *stage < max_stage) *stage += 1;
  else if (k * 1000 >= up_m * window && top_random) *stage = (int32_t)(rnd % (uint32_t)(max_stage + 1));
  else if (k * 1000 < down_m * window && *stage > 0) *stage -= 1;
  if (*stage != s0) { *hist = 0u; *n = 0; }
}

}  // namespace mdp::cur
