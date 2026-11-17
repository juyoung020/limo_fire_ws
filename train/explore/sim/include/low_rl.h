// 아래층 걸음 길(EnvCfg::low_rl — env.step 한 번 = 0.1 s 걸음 하나)의 걸음 사이 값·준비 단계 차례. env.h 가 namespace sim 안에서 포함한다 (low_step.h 뒤).
// 아래층은 규칙 추종기(A* 경로 + 추종 — hier_path.h)만: 위층이 LowCmd(목표 칸·follow)를 넣고, 여기서는 계획 차례와 막힘 신호만 다룬다.
// (이름의 "rl" 은 그만둔 아래층 강화학습 때 붙은 것 — 정책·보상·관측은 2026-11-16·17 에 archive/rl 로 옮기고 여기서 걷어냄)
//
// 한 걸음의 차례 — GPU(env_kernel.cu)·CPU 참조판(env_ref.cpp)·실물 RobotEnv 가 같은 함수를 같은 차례로 부른다:
//   [행동]  lowrl_begin      막힘 신호 지움(새 결정 구간)
//   [물리·라이다·지도 — env.h]
//   [걸음 끝] lowrl_finish   막힘 신호 올림 (정체 — 위층이 그 칸을 피함) · 판이 끝났으면 걸음 사이 값 비움
//   [준비]  lowrl_due 일 때 깊이 카메라 층 · 경로 계획 (hier_path.h — low_period 걸음마다 · 목표가 바뀌면 바로) → lowrl_planned
// 준비 단계는 호스트가 LowCmd 를 넣은 뒤 Env::low_prepare (실물 RobotEnv::low_prepare) 로 부른다 (위층이 목적지를 넣는 자리).
#pragma once

DEV int lowrl_period(const EnvCfg& c) { return c.low_period > 0 ? c.low_period : c.decide_steps; }

// 행동 앞머리 (env.h action_begin 몫)
DEV void lowrl_begin(const EnvBatch& b, EnvState& e, int env) {
  e.people.hit_person = 0;
  LowRl& R = b.lowrl[env];
  if (R.p_step == e.step) {   // 계획 단계 뒤 첫 걸음 = 새 결정 구간: 막힘 신호를 지움 (low_task.h LowRl::stall)
    R.stall = 0;
    R.stall_tgt = -1;
  }
}
// 판 시작 값 (막힘 신호 stall·stall_tgt 는 그대로 — 판을 끝낸 정체를 위층이 읽게)
DEV void lowrl_clear(LowRl& R) {
  R.p_step = 0;
  R.tgt_seen = -2;
  R.fol_seen = -1;
}

// 이번 준비 단계에 계획(과 깊이 카메라 층 갱신)을 도나 — 블록의 모든 스레드가 첫 동기화 앞에 읽는다 (고치는 것은 그 뒤 스레드 0 의 lowrl_planned)
DEV bool lowrl_due(const EnvBatch& b, const EnvState& e, int env) {
  if (!b.lowrl) return true;
  const LowRl& R = b.lowrl[env];
  return R.tgt_seen == -2 || e.step - R.p_step >= lowrl_period(b.cfg) || b.lowcmd[env].target != R.tgt_seen || b.lowcmd[env].follow != R.fol_seen;
}
// 깊이 카메라 층을 이번 걸음에 갱신하나: cam_every 면 걸음마다, 아니면 계획 차례만 (lowrl_due)
DEV bool lowrl_cam_due(const EnvBatch& b, const EnvState& e, int env) { return (b.lowrl && b.cfg.cam_every) || lowrl_due(b, e, env); }
DEV void lowrl_planned(const EnvBatch& b, const EnvState& e, int env) {
  LowRl& R = b.lowrl[env];
  R.p_step = e.step;
  R.tgt_seen = b.lowcmd[env].target;
  R.fol_seen = b.lowcmd[env].follow;
}

// ---------- 걸음 끝 (finish_step 맨 끝, 스레드 0): 막힘 신호 ----------
DEV void lowrl_finish(const EnvBatch& b, EnvState& e, int env) {
  if (!b.lowrl) return;
  LowRl& R = b.lowrl[env];
  if ((e.done || low_xenv(b, env)) && b.low[env].end == kLowEndStall) {   // 위층에 알리는 막힘 신호 (합친 탐사는 판을 안 끝내고 신호만)
    R.stall = 1;
    R.stall_tgt = b.lowcmd[env].follow != 0 ? b.lowcmd[env].target : e.goal_cell;
  }
  if (e.done) lowrl_clear(R);
}
