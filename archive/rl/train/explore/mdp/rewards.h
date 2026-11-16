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
#include "obs_layout.h"   // mdp::act::V_FWD — approach 자름 폭

namespace mdp::rew {

// ============================================================================================================
// 1. 보상 값 (점수) — 잠정, 학습 곡선을 보고 바꾸면 맨 아래 근거에 적는다
// ============================================================================================================
//                                          항 이름(로그)        언제·무엇에 곱하나
constexpr float kOccCell = 20.f / 1600.f;   // coverage (로봇 지도, 2026-11-08) 판 처음 막힘이 된 지도 칸 하나에 — 한 행동에 kIdleOccMin(10) 개 이상 생긴
                                            // 걸음의 칸만 셈(진행 없음 시계와 같은 셈, env.h finish_step — 멈춰 있어도 잡음으로 드문드문 생기는 칸은 0).
                                            // 1600 = 시험 맵 크기(145~150 m²)를 끝까지 탐사했을 때 그렇게 센 칸 수 중앙(explore_v9 정책 64 판씩: bookstore 1507,
                                            // small-house 1712) → 그런 맵 다 훑으면 ≈ +20 (옛 정답 덮음 보상과 같은 크기). 정답 없이 셀 수 있게 맵마다 나누지 않고
                                            // 칸마다 고정 — 큰 맵일수록 합이 큼 (학습 맵 24 개 중앙: 단계 0 0.2~0.6 k, 1 0.5~1.4 k, 2 1.0~2.3 k, 3 1.5~4.2 k)
constexpr float kTime = 0.2f;             // time              기준 시간 kTRef 동안 합 (초당 −kTime/kTRef)
constexpr float kCollWall = -6.f;         // collision_wall    닿음(센서 — sim contact.h)으로 판 끝
constexpr float kCollPerson = -9.f;       // collision_person  닿음 때 라이다 추적 사람이 몸통 곁 (벽보다 크게 — 추적은 아래층 RL 만, 없으면 벽으로 셈)
constexpr float kPersonNear = -0.5f;      // person_near       라이다 추적 사람 근처(kNearDist 안)에서 움직인 거리 m 당 (바퀴 오도메트리)
constexpr float kDeclareSuccess = 10.f;   // declare           지도 닫힘 성공 (항 이름은 옛 끝 선언 자리)
constexpr float kFast = 10.f;             // declare           성공 때 더: kFast × max(0, 1 − 걸린 시간 / kTRef) — 빨리 끝낼수록 큼 (2026-11-10, 잠정, 아래 근거 "빠르기")
constexpr float kSlip = -0.05f;           // slip              바퀴 오도메트리 ω 와 자이로가 어긋난 각도 rad 당
constexpr float kTurn = -0.001f;          // turn              자이로로 잰 돈 각도 rad 당 (|ω|·걸음 길이) — 작게: 사각(라이다 220°)을 메우는 회전은 남기고 늘 최대로 도는 것만 줄이게
constexpr float kIdlePenalty = -4.f;      // no_progress       지도 진척(새 막힘 칸 ≥ kIdleOccMin / 행동)이 kIdleTime 동안 없어 판 끝
// 아래 넷은 속도 행동(직접 제어) 모드에만 — 계층형(StepIn::hier)에서는 늘 0. 2026-11-11, 모두 잠정 (맨 아래 근거 "직접 제어 촘촘한 항")
constexpr float kApproach = 0.1f;         // approach          가장 가까운 프론티어 덩어리까지 BFS 길 거리가 줄어든 m 당 (늘면 −, 걸음마다 ±kApproachClip 걸음 폭에서 자름)
constexpr float kFrontCells = 1.f / 4000.f;  // front_cells    번진 곳 안 프론티어 칸 수가 한 칸 줄 때마다 (늘면 −): −kFrontCells × (지금 − 직전)
constexpr float kProx = -0.5f;            // proximity         센서 간격(sim contact.h s_clear) < kProxDist 에서 움직인 거리 m 당
constexpr float kSmooth = 1.5e-4f;        // smooth            −kSmooth × 이번·직전 행동 (v, vy, ω 정규 단위) 차이 제곱 합

// ============================================================================================================
// 2. 항 모양 (점수 아님)
// ============================================================================================================
constexpr float kTRef = 290.f;            // time·빠르기의 기준 시간 (s) — 잠정. 옛 맵별 T_ref(정답 벽 칸 ÷ 10 칸/s: small-house 259 s, bookstore 321 s)의 가운데.
                                          // 맵마다 두려면 정답 벽 칸 수가 들어서 2026-11-14 에 상수로 (실물이 못 재는 값)
constexpr float kNearDist = 0.5f;         // person_near: 몸통~추적 사람 간격 이 안이면 감점 (m)
constexpr float kApproachClip = 1.5f;     // approach: 한 걸음 거리 변화를 ±(앞 최고 속도 act::V_FWD × 걸음 길이) × 이 값에서 자름 (0.1 s 걸음이면 ±0.15 m) — 잠정
constexpr float kProxDist = 0.2f;         // proximity: 몸통~센서 간격 이 안이면 감점 (m) — 잠정

// ============================================================================================================
// 아래는 식 — 값을 바꿀 때는 건드릴 필요 없음
// ============================================================================================================

// ---- 항 번호 (reward_terms [N][kTerms]) ----
// 새 항은 끝에 붙인다: approach·front_cells 는 걸음 끝이 아니라 그 뒤 관측 때(프론티어 요약이 나온 뒤) 채워 보상에 더하므로(env.h shape_obs)
// 맨 끝 둘이어야 보상 = 항을 번호 순으로 더한 값 그대로다
enum Term : int {
  kTermCov = 0, kTermTime, kTermCollWall, kTermCollPerson, kTermPersonNear, kTermDeclare, kTermSlip, kTermIdle, kTermTurn,
  kTermProx, kTermSmooth, kTermApproach, kTermFrontCells, kTerms
};
inline const char* term_name(int k) {
  static const char* n[kTerms] = {"coverage", "time", "collision_wall", "collision_person", "person_near", "declare", "slip", "no_progress", "turn",
                                  "proximity", "smooth", "approach", "front_cells"};
  return n[k];
}

// 성공 빠르기 보너스 (성공 판에만): kTRef 안에 끝내면 0 ~ kFast, 넘으면 0
DEV float fast_bonus(float t) {
  const float f = 1.f - t / kTRef;
  return f > 0.f ? kFast * f : 0.f;
}

// 한 걸음 항 입력
struct StepIn {
  float dt;             // 걸음 길이 (s)
  int collided_wall, collided_person;   // 닿음 (센서) — 사람 곁이면 사람
  float near_person;    // 사람 근접 몫: 몸통~추적 사람 간격 < kNearDist 이면 이번 걸음 이동 거리 (m, 바퀴 오도메트리)
  float slip;           // 바퀴 오도메트리 ω − 자이로 어긋난 각도 합 (rad)
  int idle;             // 이번 걸음 진행 없음으로 끝 (닿음·성공이 아닐 때만)
  int closed;           // 이번 걸음 지도 닫힘으로 성공 → 성공 보너스 kDeclareSuccess (declare 항 자리)
  int new_occ;          // 이번 걸음 로봇 지도에서 판 처음 막힘이 된 칸 수 (첫 걸음 0)
  float turn;           // 이번 걸음 돈 각도 (rad, ≥ 0 — 자이로)
  float t;              // 판 시작부터 이번 걸음 끝까지 시간 (s) — 성공 빠르기 보너스 kFast
  int hier;             // 1: 계층형 행동(sim kActPath) — 헛돎·회전 항 0 (아래 근거 "계층형", 상위 정책이 못 고르는 하위 걸음 항)
  float move;           // 이번 걸음 이동 거리 (m) — 바퀴 오도메트리 속도 × 시간 (near_person 과 같은 셈). proximity
  float clearance;      // 몸통 센서 간격 (m, sim contact.h s_clear — 라이다 점·로봇 지도 막힘·깊이 층). 모르면 kProxDist 이상을 넣어 0. proximity
  float act_d2;         // 이번·직전 행동 (v, vy, ω 정규 단위, 자른 값) 차이 제곱 합 (판 첫 걸음 0). smooth
};

// 벽 근접 몫: 간격 0 에서 1, kProxDist 에서 0 (곧게)
DEV float prox_ramp(float clearance) {
  const float f = (kProxDist - clearance) / kProxDist;
  return f > 0.f ? f : 0.f;
}

// approach 항 (관측 때, env.h shape_obs): d = 로봇 칸 → 가장 가까운(BFS) 프론티어 덩어리까지 길 거리 (칸, −1 = 덩어리 없음 — 관측 FRONTIER 첫 칸과 같은 값).
// 직전·지금 둘 중 하나라도 덩어리가 없으면 0. 거리 변화는 ±kApproachClip 걸음 폭에서 자름 (덩어리가 풀려 다음 덩어리로 건너뛸 때의 큰 값을 막음)
DEV float approach_reward(int d_prev, int d_now, float res, float dt) {
  if (d_prev < 0 || d_now < 0) return 0.f;
  const float c = (act::V_FWD * dt) * kApproachClip;
  float dd = (float)(d_prev - d_now) * res;
  dd = dd > c ? c : (dd < -c ? -c : dd);
  return kApproach * dd;
}

// front_cells 항 (관측 때): 번진 곳 안 프론티어 칸 수 (관측 FRONT_CELLS 의 정규화 전 값) 가 줄면 +
DEV float front_cells_reward(int f_prev, int f_now) { return -kFrontCells * (float)(f_now - f_prev); }

// 항마다 값 → out[kTerms], 합을 돌려줌 (더하는 순서 고정 — 결정적)
DEV float terms(const StepIn& in, float* out) {
  out[kTermCov] = kOccCell * (float)in.new_occ;
  out[kTermTime] = -kTime * in.dt / kTRef;
  out[kTermCollWall] = in.collided_wall ? kCollWall : 0.f;
  out[kTermCollPerson] = in.collided_person ? kCollPerson : 0.f;
  out[kTermPersonNear] = kPersonNear * in.near_person;
  out[kTermDeclare] = in.closed ? kDeclareSuccess + fast_bonus(in.t) : 0.f;
  out[kTermSlip] = in.hier ? 0.f : kSlip * in.slip;
  out[kTermIdle] = in.idle ? kIdlePenalty : 0.f;
  out[kTermTurn] = in.hier ? 0.f : kTurn * in.turn;
  out[kTermProx] = in.hier ? 0.f : kProx * prox_ramp(in.clearance) * in.move;
  out[kTermSmooth] = in.hier ? 0.f : -kSmooth * in.act_d2;
  out[kTermApproach] = 0.f;     // 관측 때 채움 (env.h shape_obs — 계층형이면 0 그대로)
  out[kTermFrontCells] = 0.f;
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
//   빠르기 kFast 10 × max(0, 1 − t / T_ref) (2026-11-10, 잠정): explore_h3 반복 50~300 동안 시험 맵 결정적 평가의 성공까지 시간이 그대로
//     (bookstore 177~189 s, small-house 135~146 s). 그 판 시간 감점 합은 −0.2 × t / T_ref ≈ −0.11 로 덮음(+20)·성공(+10)에 묻혔다.
//     시간 감점(kTime)을 키우면 위 충돌 부등식(kCollWall < kDeclareFail − kTime · 10)이 깨져 "일찍 박아 감점 아끼기" 가 이득이 되므로,
//     대신 성공 판에만 빠르기 몫을 더한다 — 성공 끝 보상을 경로 효율(SPL)로 키우는 꼴(Wijmans et al. DD-PPO, arXiv 1911.00357;
//     곱셈형 길이 보정 arXiv 2603.10535: 덧셈 길이 감점은 과제 성능을 깎으며 악용될 수 있다). 실패·충돌 판엔 0 이라 위 부등식은 그대로.
//     크기: bookstore(T_ref 321 s) 180 s 성공 → +4.4, 120 s → +6.3 (10 s 당 +0.31, 할인 전). 10 = kDeclareSuccess 와 같은 크기로 둠.
//     닫힘 성공(로봇 지도로만 판정)에도 준다 — 가짜 닫힘(정답 덮음 < 0.95)이 늘면 평가의 참 성공을 보고 정답 덮음 조건을 붙인다.
//   직접 제어 촘촘한 항 (2026-11-11, 넷 모두 잠정): 속도 행동 모드의 explore_v10 이 반복 8 만에서 성공 ~0.5 로 멈췄다 — 덮음 보상은 새 벽을 볼 때만
//     나와 "어디로 가야 하나" 의 걸음마다 신호가 없다. 그래서 속도 행동 모드에만 넷을 더한다(계층형 StepIn::hier 은 넷 다 0 — 추종기가 모는 걸음 몫이라).
//     측정은 explore_v10 정책(반복 80235, 결정적)을 시험 맵 둘에서 64 판씩 돌린 판 합 평균 (bookstore / small-house):
//       덮음 +21.3 / +22.2, 선언(성공 + 빠르기) +8.2 / +5.5, approach +3.4 / +2.9, proximity −1.4 / −0.3, smooth −0.55 / −0.39, front_cells −0.02 / −0.02
//       (smooth·front_cells 는 아래에서 정한 값으로 다시 잰 것. 넷의 절댓값 합 ≈ 5.4 / 3.6 < 덮음 + 성공 ≈ 30.)
//   approach +0.1 / m: 가장 가까운 프론티어 덩어리(관측 FRONTIER 칸과 같은 값 — frontier.h reach_summary 의 1 번째 덩어리)까지 BFS 길 거리 d 가
//     줄어든 만큼. 퍼텐셜 Φ = −kApproach·d 의 차이 꼴(Ng, Harada, Russell, "Policy invariance under reward transformations", ICML 1999)이지만
//     엄밀한 퍼텐셜 기반은 아니다: (1) 할인 γ 를 안 곱한 Φ' − Φ 이고, (2) 한 걸음 변화를 ±kApproachClip × 걸음 폭(0.15 m)에서 자르며,
//     (3) 덩어리가 없는 걸음·판 첫 걸음·판 끝 걸음은 0 이다. 자르는 까닭: 덩어리를 풀면(다 봄) 가장 가까운 덩어리가 다음 것으로 바뀌어 d 가
//     몇 m 뛴다 — 그대로 두면 프론티어를 푼 걸음이 큰 감점을 받는다. 자른 탓에 "다가감(+) → 풀림(−0.015 에서 잘림)" 이 남아 판 합이 양수가 된다
//     (측정 +3 안팎 = 다가간 길 30 m 몫) — 정책 불변은 보장하지 않고, 앞뒤 왕복·제자리 회전으로는 못 번다(reward_probe osc·spin 이 검사).
//   front_cells −(1/4000) × (프론티어 칸 수 F 의 증가): Jonnarth, Zhao, Felsberg, "Learning Coverage Paths in Unknown Environments with Deep
//     Reinforcement Learning"(ICML 2024, arXiv 2306.16978) 식 (4) 의 증분 총변동 보상 R = −λ · (V(C_t) − V(C_{t−1})) / (2 · v_max · Δt)
//     (V = 덮은 지도의 총변동 식 (2), 탐사 과제 λ = 0.2, 면적 보상 λ = 1 보다 작게)에 맞댄 것 — 같은 식은 아니다: 우리 F 는 덮은 지도의
//     총변동이 아니라 "갈 수 있는 칸 중 모름과 맞닿은 칸 수"(관측 FRONT_CELLS 와 같은 값)이고, 걸음 폭(2 · v_max · Δt)으로 나누지 않고 판 전체
//     크기로 맞췄다. 크기: v10 측정에서 판 최대 F 평균 3718 / 3849 (중앙 3661 / 3859, 최대 5931 / 5063), 판 처음 F = 0 → 1/4000 이면 F 가 꼭대기까지
//     늘 때 합 −1, 다 줄 때 +1. 판 합은 망원 합이라 −kFrontCells × (끝 F): 닫히면 ≈ 0, 프론티어를 남기고 끝나면 음수(측정 끝 F 88 / 73 → −0.02).
//   proximity −0.5 / m: 몸통~막힌 칸 간격이 kProxDist 0.2 m 안일 때 움직인 거리에 (person_near 와 같은 꼴). 간격은 라이다가 아니라 정답 격자의
//     간격 지도(MapDev::clear, 사람 충돌 판정에 이미 쓰던 것)를 몸통 점 13 개(충돌 판정 body_hits 와 같은 점)에서 읽은 최소 − 반 칸 —
//     라이다(±110°)가 못 보는 뒤·옆도 잡힌다. 사람은 안 넣는다(person_near 몫). 간격 지도가 없는 곳(실물 노드)은 0.
//   smooth −1.5e-4 × ‖a_t − a_{t−1}‖²: a = 정책이 낸 (v, vy, ω) 를 [−1, 1] 로 자른 값(관측 LAST_ACT 와 같은 값, 바퀴 한도·안전층이 줄이기 전), 판 첫 걸음 0.
//     크기: v10 의 판 합 Σ‖Δa‖² 측정 3698 / 2625 → 판 합 ≈ −0.5 가 되게.
//   때: approach·front_cells 는 행동 뒤 관측의 프론티어 요약이 있어야 해서 걸음 끝이 아니라 관측 때 더한다(env.h shape_obs). 판을 끝낸 걸음은 0.
//   충돌 부등식: 판을 끝내는 걸음엔 approach·front_cells 가 0 이고 proximity·smooth ≤ 0 이라 박는 걸음이 더 받는 것은 없다. 박는 대신 선언하는 쪽이
//     더 잃을 수 있는 것은 smooth 한 걸음 최대 kSmooth × 3 × 2² = 0.0018 뿐 → kCollWall −6 < kDeclareFail − kTime · 10 − 0.0018 = −5.0018 그대로.
//     박으러 가는 길에 버는 approach 는 박지 않고 가도 똑같이 번다(프론티어로 가는 길이면). 벽으로 다가가는 길은 proximity 감점.
//   정답 입력 뺌 (2026-11-14, 사용자 결정 — "실물이 못 재는 것은 판단·학습 신호를 못 만든다"). 위 근거 가운데 바뀐 것:
//     덮음은 로봇 지도 칸만(정답 벽 덮음 kCov 없앰), 성공은 지도 닫힘만(끝 선언 + 정답 덮음 규칙·선언 보상 kDeclareFail·Near·RampFrom 없앰),
//     시간·빠르기의 기준 T_ref(정답 벽 칸 수) → 상수 kTRef 290 s, 충돌 → 센서 닿음(sim contact.h — 라이다·깊이 층·명령 대비 안 움직임),
//     사람(충돌·근접) → 아래층 RL 의 라이다 덩어리 추적, 근접 간격 → 센서 간격(라이다 점·로봇 지도 막힘·깊이 층, 지도 자세),
//     이동 거리 → 바퀴 오도메트리, 회전 → 자이로, 헛돎 → |바퀴 오도메트리 ω − 자이로|. 정답 값은 기록(EpInfo·통계)에만 남는다.
//     충돌 부등식 (선언 실패가 없어져 다시 셈): 박기 < 진행 없음 끝 − 안전 시간까지의 시간 감점 = kIdlePenalty − kTime · kSafeTime / kTRef
//     = −4 − 0.2 · 1200 / 290 = −4.83 → kCollWall −6 (여유 1.2). 일부러 박아 시간 감점을 아끼는 것 < 끝까지 버티다 진행 없음으로 끝나는 것.
