// 아래층(경로 추종) 계측·목표 넣기·합친 탐사의 기록 꼴 (env.h 맨 위에서 include). 걸음마다 부르는 함수는 low_step.h (env.h 가운데 — EnvState·EnvBatch·hier_path.h 뒤),
// 아래층 걸음 길(0.1 s 걸음)은 low_rl.h.
//
// 모두 스위치 — 끄면(EnvBatch::low·lowcmd·lowrl = nullptr) 아래 함수는 첫 줄에서 돌아가고 환경 상태는 한 비트도 안 바뀐다.
//  (1) 시간 손실 분해 (EnvCfg::low_stat): 속도 행동 걸음(0.1 s)마다 추종기가 "왜 그 속도였나" 를 하나로 분류해 판 누적(LowEp)에 더한다.
//      환경 상태(EnvState)는 읽기만 한다 → 켜도 판 자체는 끈 것과 비트 같다. 판 첫 걸음에 0 으로 — 판이 끝나도 다음 판 첫 걸음 전까지 남아
//      호스트가 done 을 본 뒤 읽는다 (Env::get_low).
//  (2) 목표 넣기 (EnvCfg::low_task — 합친 탐사 low_explore 와 같이만): 호스트(위층)가 환경마다 LowCmd 를 넣는다 (Env::set_low_cmd).
//      follow = 0 이면 후보 번호 행동(계층형 그대로), follow = 1 이면 행동 칸이 목표 칸 target 하나.
//  (3) 합친 탐사 (EnvCfg::low_explore — 도구 scripts/combine_run.cpp · 실물 노드 limo_fire_explore explorer): 위층이 고른 칸을 아래층 규칙 추종기가 몰아 탐사를 끝까지 한다.
//      판 끝은 탐사 규칙(닫힘 성공·진행 없음·닿음·안전 시간)만. 통과·경로 끝이 목표 밖은 LowEp 에 적기만(reached·off·hold),
//      정체는 막힘 신호(LowRl::stall)만 세우고 기준점을 새로 잡는다 (low_step.h low_finish).
//  (옛 단독 과제 — 목표 수·시간·정체로 판을 끝내던 아래층 RL 학습 판 — 은 2026-11-17 에 걷어냄, archive/rl)
#pragma once

namespace sim {

// 걸음 분류 (배타 — 걸음마다 정확히 하나). 차례는 기록 칸 번호라 바꾸지 말 것 (도구가 이름표를 같은 차례로 가짐)
enum : int32_t {
  kLowFull = 0,     // 제 속도: 추종기 목표·내보낸 명령 모두 ≥ 0.9 × path_v
  kLowAccel,        // 추종기 목표는 ≥ 0.9 × path_v 인데 velocity_smoother 가속 한도로 명령이 아직 못 미침 (직전 감속 원인별로 acc_* 에 따로)
  kLowCurv,         // RPP 곡률 감속이 묶음 (regulated_linear_scaling_min_radius)
  kLowCost,         // RPP 장애물 근접 감속이 묶음 (cost_scaling_dist·gain)
  kLowApproach,     // RPP 경로 끝 감속이 묶음 (approach_velocity_scaling_dist)
  kLowMon,          // collision_monitor 가 명령을 줄임 (k < 1) — 가는 걸음만 (회전 걸음은 회전 분류 그대로)
  kLowRotArrive,    // 경로 끝 도착 뒤 제자리 회전 (다음 결정을 기다림)
  kLowArriveStop,   // 도착했는데 회전도 막혀 섬
  kLowRotSwitch,    // 목적지가 바뀐 뒤 첫 전진 전의 방향 맞추기 회전
  kLowRotPlan,      // 같은 목적지로 경로를 다시 짠 뒤 첫 전진 전의 방향 맞추기 회전
  kLowRotMid,       // 그 밖(가던 중) 방향 맞추기 회전
  kLowRotNoGoal,    // 목적지 없음(후보 없음·포기) 제자리 회전
  kLowNoPath,       // 목적지는 있는데 경로를 못 짜 섬
  kLowImm,          // RPP 충돌 검사에 걸려 멈춤·물러남
  kLowRec,          // 복구 물러나기 (BT BackUp)
  kLowRotBlocked,   // 제자리 회전이 뒤 장애물로 막힘
  kLowSafety,       // 규칙 추종기의 앞 정지(EnvCfg::safety)·sweep_guard 가 명령을 0 으로·줄임 (정책 걸음엔 없음 — 2026-11-14 부터 정책 명령은 안 고침)
  kLowKinds
};
// 추종기 가지 (path_follow → PathWhy::kind)
enum : int32_t { kWhyNoGoal = 0, kWhyNoPath, kWhyArrive, kWhyArriveStop, kWhyRot, kWhyGo };
// 판 끝 까닭 (LowEp::end). 충돌 둘은 계측만 켜도 적는다
enum : int32_t { kLowEndNone = 0, kLowEndGoals, kLowEndTime, kLowEndStall, kLowEndWall, kLowEndPerson };   // 1·2·5 는 옛 단독 과제·사람 닿음 코드 (안 나옴 — 번호는 그대로)

constexpr int kLowAng = 6;   // 방향 변화 구간: 0–30 · 30–60 · 60–90 · 90–135 · 135–180° · 미정(새 경로가 생기기 전에 또 바뀜)
constexpr int kLowDst = 3;   // 목적지가 옮겨간 거리 구간: < 0.5 · 0.5–2 · ≥ 2 m

// 추종기가 이번 걸음에 낸 값 (path_follow 가 채움 — 쓰는 쪽이 없으면 안 채움)
struct PathWhy {
  int32_t kind;      // kWhy*
  int32_t bind;      // kWhyGo 일 때 목표 속도를 묶은 것: kLowFull(없음) · kLowCurv · kLowCost · kLowApproach
  float vc;          // 추종기 목표 속도
  float err;         // 당근 방향 − 지금 방향 (rad, kWhyRot·kWhyGo)
  float d_goal;      // 경로 끝까지 직선 거리 (m, kWhyRot·kWhyGo·도착)
};

// 호스트(위층) → 환경. 결정마다 호스트가 씀
struct LowCmd {
  int32_t target;    // 목표 칸 (cy · gw + cx, −1 없음) — follow 1 일 때만 봄
  int32_t follow;    // 0 보통 계층형 (target 은 안 봄 — 후보 번호 행동) · 1 목표 칸 target 을 규칙 추종기가 따라감
  // 아래 셋은 follow 1 일 때만 봄 — 기본값이면 예전과 같음 (사람 따라가기 · 물체 보고 서기, scripts/obj_follow.h)
  float vmax = 0.f;      // 추종기 속도 한도 (m/s, 0 = path_v 그대로)
  int32_t face = -1;     // 목표가 없거나 닿았을 때 이 칸 쪽으로 제자리에서 돌아 섬 (−1 = 예전처럼 rot_w 로 돎)
  float w_idle = 0.f;    // 목표 없을 때(face −1) 제자리 회전 ω (rad/s, 부호 = 방향, 0 = rot_w) — 놓친 사람을 천천히 둘러 찾기
};

// 판 누적 (환경마다 하나)
struct LowEp {
  // 분해: 분류마다 걸음 수 · 참 자세 이동 거리(걸음 처음·끝 사이 직선, m) · 명령 거리(|명령 v| × 걸음 시간, m)
  int32_t n[kLowKinds];
  float dist[kLowKinds];
  float cmd[kLowKinds];
  int32_t acc_n[kLowKinds + 1];    // kLowAccel 을 직전 감속 원인(마지막 Full·Accel 아닌 분류)별로. 끝 칸 = 판(과제) 시작
  float acc_dist[kLowKinds + 1];
  int32_t steps;                   // 분류한 걸음 수 (= Σ n — 자기 일관성: 판 걸음 수와 같아야)
  float dist_all;                  // 이동 거리 합 (= Σ dist)
  // 목적지 바뀜 (goal_cell 이 칸 → 다른 칸): [도착했었나 0·1][옮겨간 거리 구간][방향 변화 구간]
  int32_t sw[2][kLowDst][kLowAng];
  int32_t ab_n[kLowAng];           // 도착 전에 바뀐 목적지로 가던 구간의 걸음 수 (방향 변화 구간별) — 버린 구간
  float ab_dist[kLowAng];          // 그 이동 거리 (m)
  float left_sum;                  // 도착 전에 바뀔 때 옛 경로 끝까지 남았던 직선 거리의 합 (m)
  int32_t n_goal;                  // 목적지를 받은 수 (없음 → 칸 포함)
  int32_t end;                     // kLowEnd*
  // 목표 따라가기 (follow ≠ 0)
  int32_t following;               // 이 판이 목표 따라가기 구간에 들어감 (들어갈 때 위 누적을 0 으로)
  int32_t f_start;                 // 그때 걸음 번호
  int32_t reached;                 // 통과 반경에 든 목표 수
  int32_t reach_step;              // 마지막으로 든 걸음 번호 (걸음 끝 기준)
  int32_t off;                     // 경로 끝에 섰는데 목표가 통과 반경 밖 (계획이 목표 칸에 못 섬 — 도달 불가로 치고 새 목표)
  int32_t hold;                    // 이 결정이 끝난 걸음 (결정 안 걸음 수, 0 = 진행 중) — 그 뒤 걸음은 쉼 (다음 결정 앞 path_recovery 가 지움)
  int32_t d_step;                  // 이 결정 첫 걸음의 걸음 번호
  // 진행 중 (걸음 사이에 들고 가는 값)
  int32_t goal_prev, plans_prev;
  int32_t ctx;                     // 0 · 1 목적지 바뀐 뒤 첫 전진 전 · 2 다시 계획 뒤 첫 전진 전
  int32_t cause;                   // 마지막 Full·Accel 아닌 분류 (없으면 kLowKinds)
  int32_t arrived;                 // 지금 목적지의 경로 끝에 도착했었나
  int32_t pend, pend_arr, pend_d, pend_n;   // 방향 변화가 아직 안 정해진 바뀜 하나
  float pend_dist;
  int32_t seg_n;                   // 지금 목적지로 간 걸음 수·거리
  float seg_dist;
  float left;                      // 지난 걸음 경로 끝까지 거리
  int32_t kind;                    // 이번 걸음 분류 (low_finish 가 거리를 더할 칸)
  float x0, y0;                    // 이번 걸음 처음 참 자세
  int32_t a_step;                  // 정체 닻: 이 자리에서 low_idle_d 를 벗어나면 닻을 옮김
  float ax, ay;
};

// ---------- 아래층 걸음 길 (EnvCfg::low_rl — 걸음 함수는 low_rl.h) ----------
// 목표 넣기(low_task) 위에 얹는 스위치. 켜면 env.step 한 번 = 걸음 하나(0.1 s — 묶음 없음). 아래층은 규칙 추종기만.
// 끄면(EnvBatch::lowrl = nullptr) 관련 함수는 첫 줄에서 돌아가고 환경은 한 비트도 안 바뀐다.

// 환경마다 걸음 사이에 들고 가는 값
struct LowRl {
  int32_t p_step;      // 마지막으로 계획 단계를 돈 걸음 번호 (low_period 걸음마다 · 목표가 바뀌면 바로)
  int32_t tgt_seen, fol_seen;   // 그때 본 LowCmd
  // 위층에 알리는 막힘 신호: 이번 결정 구간(계획 단계 ~ 다음 계획 단계)에 정체 판정(low_idle_steps 동안 low_idle_d 를 못 벗어남 —
  // LowEp::end == kLowEndStall)이 났나 · 그때 가던 목적지 칸(−1 없음). 정체가 판을 끝내도 남는다 — 다음 계획 단계 뒤 첫 걸음이 지움.
  // 위층이 그 후보를 빼고 다시 고르는 데 쓴다 (호스트: Env::get_low_rl, 장치: EnvBatch::lowrl)
  int32_t stall;
  int32_t stall_tgt;
};


}  // namespace sim
