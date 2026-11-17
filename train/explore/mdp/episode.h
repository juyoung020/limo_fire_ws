// 탐사 판 끝 코드와 끝 조건 (값은 코드 안). 호스트·장치 공용. 시뮬 한 걸음 끝(sim/include/env.h finish_step)이 판정한다.
// 이름공간 mdp::cur 는 옛 이름(커리큘럼 시절)을 그대로 둔 것 — 판 끝 코드를 쓰는 곳(도구·실물 노드)을 바꾸지 않으려고.
//
// 판 끝 (done 코드)
//   0 진행 · 1 성공(지도 닫힘 — 닫힘 성공 규칙) · 2 닿음(센서 — sim contact.h) · 3 끊김(안전 시간)
//   · 4 옛 이른 선언(판단 모델 끝 선언 — 2026-11-18 지움, 지금은 안 나옴) · 5 옛 사람 닿음 코드(지금은 안 나옴)
//   · 6 진행 없음(지도에서 판 처음 막힘이 된 칸이 한 행동에 kIdleOccMin 개 이상인 걸음이 kIdleTime 동안 없음)
//   terminal() = 끊김이 아닌 끝 (옛 학습기가 가치를 메울지 가르던 것 — 도구 몇이 아직 씀)
// (보상·커리큘럼은 그만둔 강화학습 것이라 2026-11-17 에 archive/rl/train/explore/mdp 로 옮김)
#pragma once
#include <cstdint>

#include "detmath.h"

namespace mdp::cur {

enum Done : int32_t { kRunning = 0, kDeclareOk = 1, kCollision = 2, kTruncated = 3, kDeclareShort = 4, kCollisionPerson = 5, kNoProgress = 6,
                      kNumDone = 7 };
DEV bool terminal(int32_t d) { return d != kRunning && d != kTruncated; }

constexpr float kSafeTime = 1200.f;   // 안전 시간 (s): 이만큼 지나면 끊김(3). 맵과 무관. frontier_nav2 가장 긴 판 1040 s 보다 넉넉히
constexpr float kIdleTime = 60.f;     // 진행 없음 (s): 지도 진척(아래 kIdleOccMin)이 이만큼 없으면 끝(6)
constexpr int kIdleOccMin = 10;       // 진척 = 한 행동(0.1 s) 동안 판 처음 막힘(p > 0.65, frontier.h kOccL)이 된 지도 칸 ≥ 이것 (env.h map_progress)
// 진행 없음 판 끝(6)은 위 막힘 진척 또는 새로 안 칸(모름 → 앎, 빈 칸 포함)이 한 행동에 kIdleNewMin 개 이상이면 시계를 되돌린다 (env.h since_prog, 2026-11-17).
// 까닭: 큰 집(HSSD 480~600 m²)에서 먼 프론티어로 가며 빈 바닥만 새로 보는 동안 벽(막힘)이 60 s 넘게 안 늘어 멀쩡히 탐사 중인 판이 끝났다(hssd_105515175 씨앗 5 —
// 끝나기 전 60 s 에 42 m 를 감). kIdleNewMin 근거: bookstore 씨앗 6 한 판(1,349 걸음) 새로 안 칸 수 — 거의 서 있는 걸음(|v| < 0.05) 중앙 1, 달리는 걸음 중앙 13 (잠정).
// 닫힘 성공(close_idle_s)은 막힘 시계(since_occ) 그대로.
// 확인 (low_explore_check --upper cand, HSSD 학습 큰 집 5 곳 × 씨앗 4·5, c5d64c7 대비): 끝 닫힘 성공 0 → 3 / 10 · 진행 없음 7 → 4 · 닿음 3 → 3.
// 대가: 닫힘을 막는 안 풀리는 프론티어가 남은 판은 더 오래 돎 (hssd_105515175 씨앗 5: 굵은 덮음 0.98~1.0 인 채 228 → 409 s) — 닫힘 규칙 쪽 몫
constexpr int kIdleNewMin = 10;
constexpr float kSuccessCov = 0.95f;  // 기록 전용: 정답 벽 덮음이 이 값에 처음 닿은 시각 (EpInfo::t95 — 판 끝에는 안 씀)
//   진행 없음 끝을 둔 까닭: 지도가 안 늘어나는데 버티는 판을 안전 시간까지 끌지 않으려고.
//   출처: Go-Explore (Ecoffet et al., Nature 2021, arXiv 2004.12919 Methods) "we terminate the episode early … if no new cells are
//   discovered for 1,000 steps (while exploring)".
//   kIdleOccMin 근거 (idle_agree 일회성 측정: 학습 맵 목록, 256 환경 × 6000 걸음, 덩어리 따라가기 ½·무작위 ¼·거의 제자리 ¼, 정답 벽 시계로
//   진행 없음 끝난 604 판 · 다른 끝 6,310 판): 문턱 K 에서 지도 시계도 60 s 넘은 비율 / 다른 끝 판에서 넘은 판 —
//   K=1 1.3 % / 0 (멈춰도 지도 자세 오차 흐름·빔 잡음으로 막힘 칸이 드문드문 계속 생겨 시계가 안 감), K=5 77.8 % / 2, K=10 94.7 % / 8,
//   K=20 99.0 % / 16. 10 = 제자리를 거의 다 잡으면서 오발동이 적은 쪽.
//   진척 시계는 로봇 지도(since_occ) 기준 — 실물에서도 같은 값을 센다 (정답 벽 시계는 2026-11-05 에 뺌).

}  // namespace mdp::cur
