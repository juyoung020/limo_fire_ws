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
constexpr float kCov = 20.f;              // coverage          벽 덮음 비율 1.0 당 (이번 걸음에 새로 본 벽 몫) — 10 → 20 (2026-11-05, 아래 근거)
constexpr float kOccCell = 20.f / 1600.f;   // coverage (로봇 지도, 2026-11-08) 판 처음 막힘이 된 지도 칸 하나에 — 한 행동에 kIdleOccMin(10) 개 이상 생긴
                                            // 걸음의 칸만 셈(진행 없음 시계와 같은 셈, env.h finish_step — 멈춰 있어도 잡음으로 드문드문 생기는 칸은 0).
                                            // 1600 = 시험 맵 크기(145~150 m²)를 끝까지 탐사했을 때 그렇게 센 칸 수 중앙(explore_v9 정책 64 판씩: bookstore 1507,
                                            // small-house 1712) → 그런 맵 다 훑으면 ≈ +20 (옛 정답 덮음 보상과 같은 크기). 정답 없이 셀 수 있게 맵마다 나누지 않고
                                            // 칸마다 고정 — 큰 맵일수록 합이 큼 (학습 맵 24 개 중앙: 단계 0 0.2~0.6 k, 1 0.5~1.4 k, 2 1.0~2.3 k, 3 1.5~4.2 k)
constexpr float kTime = 0.2f;             // time              기준선 속도로 다 덮을 시간(T_ref) 동안 합 (초당 −kTime/T_ref)
constexpr float kCollWall = -6.f;         // collision_wall    벽·물체에 부딪혀 판 끝
constexpr float kCollPerson = -9.f;       // collision_person  사람과 부딪혀 판 끝 (벽보다 크게)
constexpr float kPersonNear = -0.5f;      // person_near       사람 근처(kNearDist 안)에서 움직인 거리 m 당
constexpr float kDeclareSuccess = 10.f;   // declare           끝 선언 때 덮음 ≥ 성공 기준(curriculums.h kSuccessCov)
constexpr float kDeclareFail = -3.f;      // declare           끝 선언 때 덮음이 0 에 가까울 때 (기준 근처로 갈수록 kDeclareNear 로)
constexpr float kSlip = -0.05f;           // slip              바퀴가 몸통보다 더 돈 각도 rad 당
constexpr float kTurn = -0.001f;          // turn              몸통이 돈 각도 rad 당 (|ω|·걸음 길이) — 작게: 사각(라이다 220°)을 메우는 회전은 남기고 늘 최대로 도는 것만 줄이게
constexpr float kIdlePenalty = -4.f;      // no_progress       지도 진척(새 막힘 칸 ≥ kIdleOccMin / 행동)이 kIdleTime 동안 없어 판 끝

// ============================================================================================================
// 2. 항 모양 (점수 아님)
// ============================================================================================================
constexpr float kRefCellsPerSec = 10.f;   // time: 기준선이 벽 칸을 덮는 속도 (칸/s) → T_ref = 벽 칸 수 / 이 값
constexpr float kTRefMin = cur::kSafeTime / 10.f;   // time: T_ref 하한 (s) — 작은 맵에서 시간 감점이 너무 세지 않게
constexpr float kNearDist = 0.5f;         // person_near: 몸통~사람 간격 이 안이면 감점 (m)
constexpr float kDeclareNear = 0.f;       // declare: 덮음이 성공 기준 바로 아래일 때 다가가는 값
constexpr float kDeclareRampFrom = 0.5f;  // declare: 덮음 이것부터 성공 기준까지 kDeclareFail → kDeclareNear 로 곧게 오름 (11-05, 전엔 기준 아래 0.02 폭 지수)

// ============================================================================================================
// 아래는 식 — 값을 바꿀 때는 건드릴 필요 없음
// ============================================================================================================

// ---- 항 번호 (reward_terms [N][kTerms]) ----
enum Term : int { kTermCov = 0, kTermTime, kTermCollWall, kTermCollPerson, kTermPersonNear, kTermDeclare, kTermSlip, kTermIdle, kTermTurn, kTerms };
inline const char* term_name(int k) {
  static const char* n[kTerms] = {"coverage", "time", "collision_wall", "collision_person", "person_near", "declare", "slip", "no_progress", "turn"};
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
  if (c <= kDeclareRampFrom) return kDeclareFail;
  // kDeclareRampFrom → 성공 기준 사이 곧은 오름: 더 덮고 선언할수록 덜 나쁘게
  return kDeclareFail + (kDeclareNear - kDeclareFail) * (c - kDeclareRampFrom) / (cur::kSuccessCov - kDeclareRampFrom);
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
  int closed;           // 이번 걸음 지도 닫힘으로 성공 → 성공 보너스 kDeclareSuccess (declare 항 자리)
  int new_occ;          // 이번 걸음 로봇 지도에서 판 처음 막힘이 된 칸 수 (첫 걸음 0)
  int reward_occ;       // 1: 덮음 항 = kOccCell × new_occ (로봇 지도), 0: kCov × cov_gain (정답 벽 — 보관)
  float turn;           // 이번 걸음 몸통이 돈 각도 (rad, ≥ 0)
  int hier;             // 1: 계층형 행동(sim kActPath) — 헛돎·회전 항 0 (아래 근거 "계층형", 상위 정책이 못 고르는 하위 걸음 항)
};

// 항마다 값 → out[kTerms], 합을 돌려줌 (더하는 순서 고정 — 결정적)
DEV float terms(const StepIn& in, float* out) {
  out[kTermCov] = in.reward_occ ? kOccCell * (float)in.new_occ : kCov * in.cov_gain;
  out[kTermTime] = -kTime * in.dt / in.t_ref;
  out[kTermCollWall] = in.collided_wall ? kCollWall : 0.f;
  out[kTermCollPerson] = in.collided_person ? kCollPerson : 0.f;
  out[kTermPersonNear] = kPersonNear * in.near_person;
  out[kTermDeclare] = in.closed ? kDeclareSuccess : (in.declared ? declare_reward(in.cov) : 0.f);
  out[kTermSlip] = in.hier ? 0.f : kSlip * in.slip;
  out[kTermIdle] = in.idle ? kIdlePenalty : 0.f;
  out[kTermTurn] = in.hier ? 0.f : kTurn * in.turn;
  float r = 0.f;
  for (int k = 0; k < kTerms; ++k) r += out[k];
  return r;
}

}  // namespace mdp::rew

// ============================================================================================================
// 근거 (2026-11-04, 잠정 — 학습 곡선을 보고 바꾸면 여기 적는다)
// ============================================================================================================
//   크기 틀: 기준선 속도로 다 덮고 선언하면 판 합 ≈ +20(덮음) + 10(성공) − 0.2(시간) ≈ +29.8.
//   kCov 10 → 20 (2026-11-05, 잠정): explore_v4(kCov 10) 3000 반복 뒤 판의 34 % 가 진행 없음 끝(평균 510 s, 덮음 0.53, 판 합 −1.2) —
//     새 벽 몫이 작아 헤매도 손해가 거의 없었다. 그래서 같은 날 함께: 덮음 ×2(10 → 20), 성공 ×2(5 → 10, 덮음과 비율 유지),
//     방침(사용자): 먼저 벽을 빨리 훑는 법을 배우게 감점은 작게 — 덮음·성공이 보상 대부분, 감점은 최고 보상을 노리며 나중에 피하게.
//     충돌 감점이 크면 조심만 배우고 탐사를 못 배운다(사용자 경험). 그래서 충돌 −10 → −6, 사람 −15 → −9, 시간 0.5 → 0.2,
//     진행 없음 −3 → −4(기준 미달 선언 −3 보다는 나쁘게: 헤매느니 선언). 선언 실패 −3 은 그대로.
//   기준선 덮는 속도 kRefCellsPerSec = 10 칸/s:
//     runs/frontier_fast·frontier_nav2 의 episodes.jsonl(2026-11-04, 장면 small-house·bookstore 각 8 판)에서 판마다
//     n_wall × coverage / t 를 잼. 성공 판만: frontier_fast 13.7·11.4(small-house), 18.3·17.2·14.0·13.7·12.2·10.2(bookstore) → 중앙 13.7,
//     frontier_nav2 6.9·6.5·5.4(small-house), 5.5·3.5·3.0(bookstore) → 중앙 5.4. 두 기준선 사이 기하 평균 ≈ 8.6 → 10 으로 둠
//     (정책이 nav2 판보다 빨라야 "넘어섬" 이라 nav2 보다 빠른 쪽 기준).
//   T_ref = n_wall / kRefCellsPerSec (small-house 259 s, bookstore 321 s). 초당 감점 = kTime / T_ref → 기준선 속도면 시간 합 −kTime.
//   T_ref 는 kTRefMin = 안전 시간 / 10 아래로 안 내림 — 안전 시간까지 버틴 시간 감점이 kTime × 10 = 5 를 넘지 않게(아래 부등식).
//   보상 해킹 막기 (mdp/README): 충돌 < 가장 나쁜 선언 − 안전 시간 × 초당 감점
//     = kDeclareFail − kTime · kSafeTime / kTRefMin = −3 − 0.2 · 10 = −5  →  kCollWall = −6 (여유 1). (11-04: −3 − 0.5·10 = −8 → −10)
//     일부러 일찍 박아 시간 감점을 아끼는 것 < 끝까지 버티다 기준 미달로 선언하는 것.
//   0 걸음 선언: kDeclareFail ≈ −3 (덮음 0 에서) → 음수. 일찍 선언 = 남은 덮음 보상을 버림.
//   제자리: 출발 자리 첫 장(첫 행동)의 덮음은 보상하지 않는다(env.h finish_step: 첫 걸음 덮음 증가 0) — 안 움직여도 받는 몫이라
//     서 있기만 하면 시간 감점만 쌓이다가 진행 없음(curriculums.h kIdleTime — 지도에서 새 막힘 칸이 한 행동에 kIdleOccMin 개 이상 생긴
//     마지막 걸음 뒤 60 s. 멈춰 있으면 자세 오차·빔 잡음으로 생기는 막힘 칸은 드문드문이라 이 문턱을 못 넘음)으로 끝나 kIdlePenalty.
//   선언 보상: 덮음 c ≥ kSuccessCov → +kDeclareSuccess. 아래면 c ≤ 0.5 에서 kDeclareFail, 0.5 ~ 0.95 사이 곧게 kDeclareNear(0) 로.
//     2026-11-05: 전엔 kDeclareFail + (kDeclareNear − kDeclareFail)·exp((c − 0.95)/0.02) 였다 — 0.5 에서 선언하나 0.93 에서 하나 −3 근처로 같아
//     explore_v4 는 덮음 0.77 에서 선언(판의 20 %)하고 더 채울 까닭을 못 배웠다.
//     → 지금 c = 0.94 에서 −0.07, 0.77 에서 −1.2, 0.5 아래 −3, 문턱에서 +10 으로 뛴다.
//   헛돎 −0.05 / rad: 구동 실측에 맞춘 보조 항(env.yaml 에 있던 값 그대로 옮김).
//   사람 충돌 −9(> 벽 −6 — 같은 부등식 −5 도 만족), 사람 근접: 몸통~사람 간격 0.5 m 안에서 움직인 거리 m 당 −0.5 (1 m/s 로 1 s 지나면 −0.5).
//     잠정 — 학습에서 사람을 너무 피해 탐사가 느려지면 줄인다.
//   회전 감점 kTurn −0.001 / rad (2026-11-05, 잠정): explore_v4 정책은 늘 한쪽으로 거의 최대 회전(직전 ω 정규 평균 +0.74, 1 m 당 2.7 rad).
//     v4 성공 판(68 s, |ω| 2.8)이면 −0.19, 헤맨 판(510 s)이면 −1.4 — 덮음(20)에 비해 작게. 부등식에 안 넣는다: 제자리 회전은
//     한 바퀴 뒤엔 새 막힘 칸이 문턱(kIdleOccMin)을 못 넘어 진행 없음(60 s)으로 끝나므로 최대 3.14 · 60 · 0.001 ≈ −0.19 (첫 바퀴 몫 조금 더).
//   계층형(--hier, 2026-11-09): 상위 정책은 1 s 마다 후보만 고르고 (v, ω) 는 추종기(hier_path.h)가 낸다 — 헛돎(가속·바닥)·회전(제자리 회전·RPP 회전)은
//     후보 선택으로 거의 못 바꾸는 하위 걸음 몫이라 정책 기울기엔 잡음만 더한다 → 0 (StepIn::hier). 사람 근접(person_near)은 남긴다: 어느 후보(어느 길)를
//     고르냐가 사람 곁을 지나는 거리를 바꾸고, 사람 충돌(판 끝 −9)보다 먼저 오는 촘촘한 신호다. 덮음·시간·충돌·진행 없음·닫힘 성공은 그대로(판 끝 정의도 그대로).
//     ARiADNE(Cao et al., ICRA 2023, arXiv 2301.11575 III-A)의 결정당 보상 a·|새로 본 프론티어| − b·이동 거리 + 끝 보너스 20 과 맞대면
//     덮음(새 막힘 칸) ↔ 새로 본 프론티어, 시간(1 m/s 근처 일정 속도라 이동 거리에 비례) ↔ 거리, 닫힘 성공 ↔ 끝 보너스로 이미 같은 꼴 — 그래서 항을 더 바꾸지 않는다.
//   진행 없음 감점 kIdlePenalty −4: 버티다 끝(−4) > 박기(−6), 버티다 끝 < 기준선 따라 선언 — reward_probe 가 검사.
//     (진행 없음 끝을 둔 까닭과 kIdleTime 근거는 curriculums.h "판 끝 조건".)
