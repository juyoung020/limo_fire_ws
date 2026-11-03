// 탐사 RL 보상 — 항 식 + 가중치 (값은 코드 안, mdp/README "보상 항"). 호스트·장치 공용(sim/include/detmath.h DEV).
// 시뮬 한 걸음 끝(sim/include/env.h finish_step)에서 부른다. 항마다 따로 [N][kTerms] 로 내보내 로그에 남기고, 합이 보상.
//
// 가중치 근거 (2026-11-04, 잠정 — 학습 곡선을 보고 바꾸면 여기 적는다)
//   크기 틀: 기준선 속도로 다 덮고 선언하면 판 합 ≈ +10(덮음) + 5(성공) − 0.5(시간) ≈ +14.5.
//   기준선 덮는 속도 kRefCellsPerSec = 10 칸/s:
//     runs/frontier_fast·frontier_nav2 의 episodes.jsonl(2026-11-04, 장면 small-house·bookstore 각 8 판)에서 판마다
//     n_wall × coverage / t 를 잼. 성공 판만: frontier_fast 13.7·11.4(small-house), 18.3·17.2·14.0·13.7·12.2·10.2(bookstore) → 중앙 13.7,
//     frontier_nav2 6.9·6.5·5.4(small-house), 5.5·3.5·3.0(bookstore) → 중앙 5.4. 두 기준선 사이 기하 평균 ≈ 8.6 → 10 으로 둠
//     (정책이 nav2 판보다 빨라야 "넘어섬" 이라 nav2 보다 빠른 쪽 기준).
//   T_ref = n_wall / kRefCellsPerSec (small-house 259 s, bookstore 321 s). 초당 감점 = kTime / T_ref → 기준선 속도면 시간 합 −kTime.
//   T_ref 는 kTRefMin = kSafeTime / 10 아래로 안 내림 — 안전 시간까지 버틴 시간 감점이 kTime × 10 = 5 를 넘지 않게(아래 부등식).
//   보상 해킹 막기 (mdp/README): 충돌 < 가장 나쁜 선언 − 안전 시간 × 초당 감점
//     = kDeclareFail − kTime · kSafeTime / kTRefMin = −3 − 0.5 · 10 = −8  →  kCollWall = −10 (여유 2).
//     일부러 일찍 박아 시간 감점을 아끼는 것 < 끝까지 버티다 기준 미달로 선언하는 것.
//   0 걸음 선언: kDeclareFail ≈ −3 (덮음 0 에서) → 음수. 일찍 선언 = 남은 덮음 보상을 버림.
//   제자리: 출발 자리 첫 장(첫 행동)의 덮음은 보상하지 않는다(env.h finish_step: 첫 걸음 덮음 증가 0) — 안 움직여도 받는 몫이라
//     서 있기만 하면 시간 감점만 쌓여 음수(안전 시간까지 −kTime·kSafeTime/T_ref).
//   선언 보상: 덮음 c ≥ kSuccessCov → +kDeclareSuccess. 아래면 kDeclareFail + (kDeclareNear − kDeclareFail)·exp((c − 0.95)/0.02)
//     → c = 0.94 에서 −1.8, 0.90 에서 −2.75, 0 에서 −3. 95 % 근처에서 가파르게 오르고 문턱에서 +5 로 뛴다.
//   헛돎 −0.05 / rad: 구동 실측에 맞춘 보조 항(env.yaml 에 있던 값 그대로 옮김).
//   사람 충돌 −15(> 벽 −10 — 같은 부등식 −8 도 만족), 사람 근접: 몸통~사람 간격 0.5 m 안에서 움직인 거리 m 당 −0.5 (1 m/s 로 1 s 지나면 −0.5).
//     잠정 — 학습에서 사람을 너무 피해 탐사가 느려지면 줄인다.
#pragma once
#include "detmath.h"

namespace mdp::rew {

// ---- 항 번호 (reward_terms [N][kTerms]) ----
enum Term : int { kTermCov = 0, kTermTime, kTermCollWall, kTermCollPerson, kTermPersonNear, kTermDeclare, kTermSlip, kTerms };
inline const char* term_name(int k) {
  static const char* n[kTerms] = {"coverage", "time", "collision_wall", "collision_person", "person_near", "declare", "slip"};
  return n[k];
}

// ---- 가중치 (위 근거) ----
constexpr float kCov = 10.f;              // 벽 덮음 비율 1.0 당
constexpr float kTime = 0.5f;             // 기준선 속도로 덮을 시간(T_ref) 동안 합
constexpr float kRefCellsPerSec = 10.f;   // 기준선 덮는 속도 (칸/s)
constexpr float kSafeTime = 1200.f;       // 안전 시간 (s) = 끊김. 맵과 무관. frontier_nav2 가장 긴 판 1040 s 보다 넉넉히
constexpr float kTRefMin = kSafeTime / 10.f;
constexpr float kCollWall = -10.f;
constexpr float kCollPerson = -15.f;      // 사람 충돌 (벽보다 크게)
constexpr float kPersonNear = -0.5f;      // 몸통~사람 간격 kNearDist 안에서 움직인 거리 m 당 (= 속도 비례)
constexpr float kNearDist = 0.5f;
constexpr float kSuccessCov = 0.95f;      // 성공 = 선언 때 벽 덮음 ≥ 이것 (mdp/README)
constexpr float kDeclareSuccess = 5.f;
constexpr float kDeclareFail = -3.f;
constexpr float kDeclareNear = 0.f;
constexpr float kDeclareWidth = 0.02f;
constexpr float kSlip = -0.05f;           // 헛돈 각도 rad 당

// 맵 하나의 T_ref (호스트, 맵 만들 때 한 번)
inline float t_ref(int n_wall) {
  const float t = (float)n_wall / kRefCellsPerSec;
  return t < kTRefMin ? kTRefMin : t;
}

// 선언 보상 (덮음 c)
DEV float declare_reward(float c) {
  if (c >= kSuccessCov) return kDeclareSuccess;
  const float x = (c - kSuccessCov) / kDeclareWidth;   // ≤ 0
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
  float r = 0.f;
  for (int k = 0; k < kTerms; ++k) r += out[k];
  return r;
}

}  // namespace mdp::rew
