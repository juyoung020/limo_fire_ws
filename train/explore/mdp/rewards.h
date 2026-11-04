// 탐사 RL 보상. 호스트·장치 공용(sim/include/detmath.h DEV). 시뮬 한 걸음 끝(sim/include/env.h finish_step)에서 부른다.
// 항마다 따로 [N][kTerms] 로 내보내 로그에 남기고, 합이 보상.
//
// 사람이 바꾸는 곳은 바로 아래 두 묶음뿐이다:
//   1. 보상 값 — 항마다 주는 점수.
//   2. 항 모양 — 식의 모양을 정하는 값(기준 속도·거리·곡선 폭). 점수가 아니다.
// 판을 언제 끝내나(안전 시간·진행 없음·성공 덮음)는 보상이 아니라 끝 조건이라 curriculums.h "판 끝 조건" 에 있다.
// 값마다의 근거·보상 해킹 부등식은 이 파일 맨 아래. 바꾸면 mdp/tools/reward_probe 로 부등식을 다시 확인한다.
#pragma once
#include "curriculums.h"
#include "detmath.h"

namespace mdp::rew {

// ============================================================================================================
// 1. 보상 값 (점수) — 잠정, 학습 곡선을 보고 바꾸면 맨 아래 근거에 적는다
// ============================================================================================================
//                                          항 이름(로그)        언제·무엇에 곱하나
constexpr float kCov = 10.f;              // coverage          벽 덮음 비율 1.0 당 (이번 걸음에 새로 본 벽 몫)
constexpr float kTime = 0.5f;             // time              기준선 속도로 다 덮을 시간(T_ref) 동안 합 (초당 −kTime/T_ref)
constexpr float kCollWall = -10.f;        // collision_wall    벽·물체에 부딪혀 판 끝
constexpr float kCollPerson = -15.f;      // collision_person  사람과 부딪혀 판 끝 (벽보다 크게)
constexpr float kPersonNear = -0.5f;      // person_near       사람 근처(kNearDist 안)에서 움직인 거리 m 당
constexpr float kDeclareSuccess = 5.f;    // declare           끝 선언 때 덮음 ≥ 성공 기준(curriculums.h kSuccessCov)
constexpr float kDeclareFail = -3.f;      // declare           끝 선언 때 덮음이 0 에 가까울 때 (기준 근처로 갈수록 kDeclareNear 로)
constexpr float kSlip = -0.05f;           // slip              바퀴가 몸통보다 더 돈 각도 rad 당
constexpr float kIdlePenalty = -3.f;      // no_progress       새 벽을 kIdleTime 동안 못 봐 판 끝

// ============================================================================================================
// 2. 항 모양 (점수 아님)
// ============================================================================================================
constexpr float kRefCellsPerSec = 10.f;   // time: 기준선이 벽 칸을 덮는 속도 (칸/s) → T_ref = 벽 칸 수 / 이 값
constexpr float kTRefMin = cur::kSafeTime / 10.f;   // time: T_ref 하한 (s) — 작은 맵에서 시간 감점이 너무 세지 않게
constexpr float kNearDist = 0.5f;         // person_near: 몸통~사람 간격 이 안이면 감점 (m)
constexpr float kDeclareNear = 0.f;       // declare: 덮음이 성공 기준 바로 아래일 때 다가가는 값
constexpr float kDeclareWidth = 0.02f;    // declare: 성공 기준 아래로 이 폭(덮음 비율) 안에서 가파르게 kDeclareNear 로 오름

// ============================================================================================================
// 아래는 식 — 값을 바꿀 때는 건드릴 필요 없음
// ============================================================================================================

// ---- 항 번호 (reward_terms [N][kTerms]) ----
enum Term : int { kTermCov = 0, kTermTime, kTermCollWall, kTermCollPerson, kTermPersonNear, kTermDeclare, kTermSlip, kTermIdle, kTerms };
inline const char* term_name(int k) {
  static const char* n[kTerms] = {"coverage", "time", "collision_wall", "collision_person", "person_near", "declare", "slip", "no_progress"};
  return n[k];
}

// 맵 하나의 T_ref (호스트, 맵 만들 때 한 번)
inline float t_ref(int n_wall) {
  const float t = (float)n_wall / kRefCellsPerSec;
  return t < kTRefMin ? kTRefMin : t;
}

// 선언 보상 (덮음 c)
DEV float declare_reward(float c) {
  if (c >= cur::kSuccessCov) return kDeclareSuccess;
  const float x = (c - cur::kSuccessCov) / kDeclareWidth;   // ≤ 0
  // exp(x) = 2^(x·log2 e): detmath 에 exp 가 없어 정수 거듭제곱 + 다항식 없이 반복 제곱으로 (결정적, x ≥ −50 만 의미)
  float e;
  if (x < -40.f) e = 0.f;
  else {
    // exp(x) = (1 + x/1024)^1024 (상대 오차 < 0.5 % — 모양만 쓰는 값)
    float b = 1.f + x / 1024.f;
    if (b < 0.f) b = 0.f;
    for (int i = 0; i < 10; ++i) b = b * b;
    e = b;
  }
  return kDeclareFail + (kDeclareNear - kDeclareFail) * e;
}

// 한 걸음 항 입력
struct StepIn {
  float cov_gain;       // 이번 걸음 벽 덮음 증가 (첫 걸음은 0)
  float dt;             // 걸음 길이 (s)
  float t_ref;          // 이 맵 T_ref
  int collided_wall, collided_person;
  float near_person;    // 사람 근접 몫: 몸통~사람 간격 < kNearDist 이면 이번 걸음 이동 거리 (m) — 속도 × 시간
  int declared;         // 이번 걸음 선언 (충돌이면 선언 안 셈)
  float cov;            // 선언 때 덮음
  float slip;           // 헛돈 각도 합 (rad)
  int idle;             // 이번 걸음 진행 없음으로 끝 (충돌·선언이 아닐 때만)
};

// 항마다 값 → out[kTerms], 합을 돌려줌 (더하는 순서 고정 — 결정적)
DEV float terms(const StepIn& in, float* out) {
  out[kTermCov] = kCov * in.cov_gain;
  out[kTermTime] = -kTime * in.dt / in.t_ref;
  out[kTermCollWall] = in.collided_wall ? kCollWall : 0.f;
  out[kTermCollPerson] = in.collided_person ? kCollPerson : 0.f;
  out[kTermPersonNear] = kPersonNear * in.near_person;
  out[kTermDeclare] = in.declared ? declare_reward(in.cov) : 0.f;
  out[kTermSlip] = kSlip * in.slip;
  out[kTermIdle] = in.idle ? kIdlePenalty : 0.f;
  float r = 0.f;
  for (int k = 0; k < kTerms; ++k) r += out[k];
  return r;
}

}  // namespace mdp::rew

// ============================================================================================================
// 근거 (2026-11-04, 잠정 — 학습 곡선을 보고 바꾸면 여기 적는다)
// ============================================================================================================
//   크기 틀: 기준선 속도로 다 덮고 선언하면 판 합 ≈ +10(덮음) + 5(성공) − 0.5(시간) ≈ +14.5.
//   기준선 덮는 속도 kRefCellsPerSec = 10 칸/s:
//     runs/frontier_fast·frontier_nav2 의 episodes.jsonl(2026-11-04, 장면 small-house·bookstore 각 8 판)에서 판마다
//     n_wall × coverage / t 를 잼. 성공 판만: frontier_fast 13.7·11.4(small-house), 18.3·17.2·14.0·13.7·12.2·10.2(bookstore) → 중앙 13.7,
//     frontier_nav2 6.9·6.5·5.4(small-house), 5.5·3.5·3.0(bookstore) → 중앙 5.4. 두 기준선 사이 기하 평균 ≈ 8.6 → 10 으로 둠
//     (정책이 nav2 판보다 빨라야 "넘어섬" 이라 nav2 보다 빠른 쪽 기준).
//   T_ref = n_wall / kRefCellsPerSec (small-house 259 s, bookstore 321 s). 초당 감점 = kTime / T_ref → 기준선 속도면 시간 합 −kTime.
//   T_ref 는 kTRefMin = 안전 시간 / 10 아래로 안 내림 — 안전 시간까지 버틴 시간 감점이 kTime × 10 = 5 를 넘지 않게(아래 부등식).
//   보상 해킹 막기 (mdp/README): 충돌 < 가장 나쁜 선언 − 안전 시간 × 초당 감점
//     = kDeclareFail − kTime · kSafeTime / kTRefMin = −3 − 0.5 · 10 = −8  →  kCollWall = −10 (여유 2).
//     일부러 일찍 박아 시간 감점을 아끼는 것 < 끝까지 버티다 기준 미달로 선언하는 것.
//   0 걸음 선언: kDeclareFail ≈ −3 (덮음 0 에서) → 음수. 일찍 선언 = 남은 덮음 보상을 버림.
//   제자리: 출발 자리 첫 장(첫 행동)의 덮음은 보상하지 않는다(env.h finish_step: 첫 걸음 덮음 증가 0) — 안 움직여도 받는 몫이라
//     서 있기만 하면 시간 감점만 쌓이다가 진행 없음(curriculums.h kIdleTime)으로 끝나 kIdlePenalty.
//   선언 보상: 덮음 c ≥ kSuccessCov → +kDeclareSuccess. 아래면 kDeclareFail + (kDeclareNear − kDeclareFail)·exp((c − 0.95)/kDeclareWidth)
//     → c = 0.94 에서 −1.8, 0.90 에서 −2.75, 0 에서 −3. 95 % 근처에서 가파르게 오르고 문턱에서 +5 로 뛴다.
//   헛돎 −0.05 / rad: 구동 실측에 맞춘 보조 항(env.yaml 에 있던 값 그대로 옮김).
//   사람 충돌 −15(> 벽 −10 — 같은 부등식 −8 도 만족), 사람 근접: 몸통~사람 간격 0.5 m 안에서 움직인 거리 m 당 −0.5 (1 m/s 로 1 s 지나면 −0.5).
//     잠정 — 학습에서 사람을 너무 피해 탐사가 느려지면 줄인다.
//   진행 없음 감점 kIdlePenalty −3: 버티다 끝(−3) > 박기(−10), 버티다 끝 < 기준선 따라 선언 — reward_probe 가 검사.
//     (진행 없음 끝을 둔 까닭과 kIdleTime 근거는 curriculums.h "판 끝 조건".)
