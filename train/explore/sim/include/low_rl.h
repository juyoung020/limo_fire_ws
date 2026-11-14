// 아래층 RL 의 걸음 함수 (기록 꼴·관측 배치는 low_task.h "아래층 RL", 설명은 mdp/README "아래층 RL"). env.h 가 namespace sim 안에서 포함한다 (low_step.h 뒤).
//
// 한 걸음(0.1 s)의 차례 — GPU(env_kernel.cu)·CPU 참조판(env_ref.cpp)이 같은 함수를 같은 차례로 부른다:
//   [행동]  lowrl_act        정책 환경(LowCmd::follow == 2)이면 행동 줄 (v, ω) → 바퀴 한도 → 명령 (고치는 층 없음) (추종기·복구 없음)
//   [물리·라이다·지도 — 기존 그대로]
//   [걸음 끝] lowrl_finish   보상 7 항(아래 상수)·누적. 판 끝 규칙은 단독 과제(low_step.h low_finish) 그대로
//   [판이 끝났으면 리셋·새 판 관측 — 기존 그대로]
//   [준비]  lowrl_goal_*     (low_auto) 새 목표 뽑기 — 1: 아는 칸에서 무작위(단독 과제) · 2: 위층 0 번 후보 규칙(탐사 판 학습, lowrl_goal_explore)
//           깊이 카메라 층 · 경로 계획 (hier_path.h — lowrl_due 일 때만: low_period 걸음마다 · 목표가 바뀌면 바로)
//           lowrl_pack       정책 관측 (low_img · low_1d) — 방금 짠 경로 기준
// 준비 단계는 low_auto 면 env.step·reset 이 끝에서 부르고, 아니면 호스트가 LowCmd 를 넣은 뒤 Env::low_prepare 로 부른다 (위층이 1 Hz 로 목적지를 넣는 자리).
#pragma once

// ---------- 보상 (모두 잠정 — 짧은 학습으로 오르는 것만 확인한 값. 근거·출처는 mdp/README "아래층 RL") ----------
// 꼴은 Long 외(arXiv 1709.10082, ICRA 2018)의 "목표 쪽 진행 + 도달 + 충돌 + 회전 벌점" 에 근접·후진 항을 더한 것. 진행은 직선 거리 대신
// 계획 경로 위 남은 길이의 차 (퍼텐셜 꼴 — Ng 외 1999: 정책 최적성을 안 바꾸는 꼴).
constexpr float kLrProgressK = 1.0f;     // 진행: × (남은 경로 길이 줄어든 m). 제 속도 1 m/s 면 걸음마다 +0.1
constexpr float kLrProgressClip = 0.2f;  // 한 걸음 진행 자름 (m) — 최고 속도 걸음 0.1 m 의 2 배 (경로로 돌아오는 걸음까지)
constexpr float kLrTimeK = -0.01f;       // 시간: 걸음마다 (제 속도 진행의 1/10 — 서 있으면 손해, 가면 이득)
constexpr float kLrReachK = 5.0f;        // 통과 반경에 듦 (목표 하나 ≈ 제 속도 5 m 진행 몫)
constexpr float kLrCollisionK = -10.0f;  // 충돌 (판 끝). 통과의 2 배 — 빠르지만 박는 정책이 남지 않게
constexpr float kLrProxDist = 0.2f;      // 근접: 몸통~막힌 칸·사람 간격이 이보다 가까우면
constexpr float kLrProxK = -0.05f;       //   × (1 − 간격 / kLrProxDist) 걸음마다 (닿기 직전 = 진행의 절반)
constexpr float kLrSmoothK = -0.01f;     // 매끄러움: × ‖이번 − 직전 행동‖ (정규 단위 v, ω)
constexpr float kLrBackK = -0.02f;       // 후진: × 후진 명령 크기 (정규 단위 0 ~ 1). 라이다가 뒤 140° 를 못 봄 — 막지는 않고 값만
constexpr float kLrTtcK = -0.1f;         // 충돌 시간(2026-11-14 low_s4): × (1 − TTC / kLrTtcT) 걸음마다, TTC = 진행 방향 센서 간격 / 다가가는 속도 (contact.h s_dir·s_close
constexpr float kLrTtcT = 0.6f;          //   — 바퀴 오도메트리·자이로). 1 m/s 면 0.6 m 앞부터 켜짐. 크기는 제 속도 진행(+0.1/걸음)과 같게 — 둘 다 잠정
// ---------- 벽 충돌 대책 (2026-11-13, 모두 잠정 — 재어 고른 값. 진단·견줌 표는 mdp/README "아래층 RL — 벽 충돌") ----------
// EnvCfg 기본값 (env_cfg.cpp 가 환경변수 SIM_LOW_NOBACK · SIM_LR_MPROX_K · _D 로 덮음)
constexpr int kLrNoBack = 1;             // 1: 정책 후진 명령을 0 으로 — v ∈ [0, 1.0] m/s (2026-11-13 기본. 뒤 140° 는 라이다 밖, Long 외·DRL-VO 도 v ≥ 0 — mdp/README "아래층 RL — 탐사 판 학습"). 0 = 옛 [−0.3, 1.0]
constexpr float kLrMproxK = -0.3f;       // 움직임 근접: × 몸통 가장 빠른 점 속도 (|v| + 꼭짓점 반경 × |ω|, m/s) × (1 − 간격 / kLrMproxD) 걸음마다.
constexpr float kLrMproxD = 0.2f;        //   간격이 이보다 가까울 때 (= kLrProxDist). 벽에 붙어 최대 속도로 돌면(0.61 m/s) 걸음마다 −0.18 = 제 속도 진행의 2 배 가까이
// 사람 충돌·속도 대책 (2026-11-13, mdp/README "아래층 RL — 사람 충돌·속도"). 실험 동안 0(끔) — 환경변수 SIM_LR_PPL_K 등으로 켠다
constexpr float kLrPplK = 0.0f;          // 사람 마주침: × (1 − 가장 가까워질 때까지 시간 / kLrPplT) 걸음마다 (음수로 켬). 아래 lowrl_ppl_risk
constexpr float kLrPplT = 2.0f;          //   내다보는 시간 (s) — 잠정. 진단: 닿기 2 s 전 간격 중앙 1.4 ~ 2.6 m, 그때 이미 라이다에 보임
constexpr float kLrPplM = 0.15f;         //   그때 몸통(반 길이 원) ~ 사람 간격이 이보다 좁으면 (m) — 잠정
constexpr float kLrPersonCollisionK = kLrCollisionK;   // 사람 충돌 (판 끝). 벽과 같게 시작
constexpr float kLrTurnK = -0.04f;       // 회전: × max(0, |정책 ω| − 바퀴 한도가 v 를 깎기 시작하는 ω) (rad/s) 걸음마다 (0 이면 끔). 꼴은 Long et al.(arXiv 1709.10082) 식 7
constexpr float kLrTrkWGate = 1.5f;      // 사람 추적: 장 사이 각속도 차가 이보다 크면 그 장을 덜 믿음 (rad/s) — 잠정
constexpr int kLrClus = 24;              //   라이다 한 장에서 보는 덩어리 수 상한
constexpr int kLrMovers = 8;             //   움직이는 덩어리(사람) 수 상한 = kMaxPeople
constexpr float kLrClusJump = 0.15f;     //   이웃 칸 거리 차가 이보다 크면 다른 덩어리 (m)
constexpr float kLrClusW0 = 0.10f, kLrClusW1 = 0.65f;   //   덩어리 폭 (m): 사람 원기둥 지름 0.4 ~ 0.6 이 보이는 현 길이
constexpr float kLrClusRange = 4.0f;     //   이 거리 안 덩어리만 (m)
constexpr float kLrMovVNorm = 1.6f;      // 관측 "움직이는 것" 속도 정규화 (m/s) — 추적이 사람으로 치는 속도 상한
constexpr float kLrClusSag = 0.15f;      //   칼만 추적: 둥근 덩어리의 활꼴 높이 / 현 길이 하한 — 잠정 (원기둥 ±50° 만 보여도 0.23)
constexpr float kLrClusRound0 = 0.25f;   //   칼만 추적: 둥근 덩어리의 현 길이 하한 (m) — 사람 원기둥 지름 0.4 ~ 0.6 — 잠정
constexpr float kLrMoveV0 = 0.25f;       //   이 속도 넘게 두 구간(0.2 s 씩) 모두 같은 쪽으로 움직여야 사람으로 본다 (m/s)
constexpr float kLrMoveV1 = 1.6f;        //   이보다 빠르면 사람이 아님 (사람 최고 1.3 m/s + 여유) — 가려졌다 드러나는 가장자리가 이렇게 보임
constexpr float kLrTrkGate = 0.30f;      //   잇기 문 (m): 사람 최고 1.3 m/s × 0.1 s + 덩어리 자리 떨림 — 잠정
constexpr float kLrTrkSz = 0.05f;        //   덩어리 자리 잡음 σ (m): 각도 칸 0.887° 가 3 m 에서 4.6 cm — 잠정
constexpr float kLrTrkQa = 1.5f;         //   가속 잡음 σ (m/s²) — 잠정
constexpr float kLrTrkV0 = 1.0f;         //   새 덩어리의 속도 σ (m/s) — 사람 속도 0 ~ 1.3 m/s
constexpr int kLrTrkMiss = 6;            //   못 이어도 이 장 수까지 같은 속도로 들고 간다 — 잠정
constexpr int kLrTrkConf = 3;            //   속도가 kLrMoveV0 ~ kLrMoveV1 인 장이 이만큼 이어져야 걷는 사람 — 잠정
constexpr float kLrTrkDisp = 0.15f;      //   그동안 이만큼 옮겨 가야 걷는 사람 (m) — 잠정
// ---------- 관측 ----------
constexpr float kLrLook0 = 1.0f, kLrLook1 = 2.5f;   // 부목표: 경로 위 가장 가까운 점에서 이만큼 앞 (m). 잠정 — 가까운 지도 창 반 변 3.2 m 안
constexpr float kLrNearCell = 0.10f;      // 가까운 지도 픽셀 (m) = mdp::obsgen::kNearCell (env_ref.cpp 가 같은지 확인)
constexpr float kLrSubNorm = 3.0f;       // 부목표 거리 정규화 (m)
constexpr float kLrGoalNorm = 15.0f;     // 목적지 거리·남은 경로 정규화 (m, 자름 1) — 과제 경로 길이 상한
constexpr float kLrXReachLen = 1.0f;     // 탐사 판 학습: 통과 보상은 처음 계획 길이가 이보다 긴 목표에 한 번만 (m) — 잠정. 통과 반경(0.5 m) 안에 생긴 후보에 서서 걸음마다 받는 것을 막음
// ---------- 자동 목표 (low_auto) ----------
constexpr int kLrProbes = 256;           // 한 번에 찔러 보는 칸 수 (아는 칸 상자 안 무작위)
// 직선 거리 등급 (m) — 앞 등급에 설 수 있는 칸이 하나라도 있으면 그 등급에서 뽑는다. 기준선 과제는 경로 3 ~ 15 m (scripts/low_run.cpp),
// 판 처음(첫 스캔만 안 때)·작은 맵은 그런 칸이 없어 아래 등급으로 내려간다
constexpr float kLrGoalD0 = 3.0f, kLrGoalD1 = 1.5f, kLrGoalD2 = 0.75f, kLrGoalDMax = 15.0f;

DEV bool lowrl_policy(const EnvBatch& b, int env) { return b.lowrl && b.lowcmd[env].follow == 2; }
DEV int lowrl_act_w(const EnvBatch& b) { return b.lowrl && b.cfg.low_act_w > 0 ? b.cfg.low_act_w : mdp::act::DIM; }
DEV int lowrl_period(const EnvCfg& c) { return c.low_period > 0 ? c.low_period : c.decide_steps; }

// 행동 앞머리 (env.h action_begin 몫 — 선언은 EnvCfg::finish_declare 를 켰을 때 호스트가 넣은 LowCmd{kLowFinish, 0} 만, 기본은 없음)
DEV void lowrl_begin(const EnvBatch& b, EnvState& e, int env) {
  e.people.hit_person = 0;
  e.declared = b.cfg.finish_declare && b.lowcmd[env].follow == 0 && b.lowcmd[env].target == kLowFinish ? 1 : 0;
  LowRl& R = b.lowrl[env];
  if (R.p_step == e.step) {   // 계획 단계 뒤 첫 걸음 = 새 결정 구간: 막힘 신호를 지움 (low_task.h LowRl::stall)
    R.stall = 0;
    R.stall_tgt = -1;
  }
}
// 판 시작 값 (난수는 그대로)
DEV void lowrl_clear(LowRl& R) {
  R.p_step = 0;
  R.tgt_seen = -2;
  R.fol_seen = -1;
  R.need_goal = 1;
  R.rem_ok = 0;
  R.rem_prev = 0.f;
  R.pol = 0;
  R.g_step = 0;
  R.g_len = -1.f;
  R.tk_head = -1;
  R.tk_n = 0;
  for (int i = 0; i < kLrTrk; ++i) {
    R.tk_hit[i] = R.tk_miss[i] = R.tk_mov[i] = 0;
    R.tk_x[i] = R.tk_y[i] = R.tk_vx[i] = R.tk_vy[i] = R.tk_pp[i] = R.tk_pv[i] = R.tk_vv[i] = R.tk_r[i] = R.tk_x0[i] = R.tk_y0[i] = 0.f;
  }
  R.x_redo = 1;
  R.x_last = 0;
  R.x_sig = 0;
  R.x_paid = 0;
  R.x_n = 0;
  for (int i = 0; i < kLowXBl; ++i) { R.x_cell[i] = -1; R.x_until[i] = 0; }
     // (막힘 신호 stall·stall_tgt 는 그대로 — 판을 끝낸 정체를 위층이 읽게)
}

// 이번 준비 단계에 계획(과 깊이 카메라 층 갱신)을 도나 — 블록의 모든 스레드가 첫 동기화 앞에 읽는다 (고치는 것은 그 뒤 스레드 0 의 lowrl_planned)
DEV bool lowrl_due(const EnvBatch& b, const EnvState& e, int env) {
  if (!b.lowrl) return true;
  const LowRl& R = b.lowrl[env];
  return R.tgt_seen == -2 || e.step - R.p_step >= lowrl_period(b.cfg) || b.lowcmd[env].target != R.tgt_seen || b.lowcmd[env].follow != R.fol_seen;
}
DEV void lowrl_planned(const EnvBatch& b, const EnvState& e, int env) {
  LowRl& R = b.lowrl[env];
  R.p_step = e.step;
  R.tgt_seen = b.lowcmd[env].target;
  R.fol_seen = b.lowcmd[env].follow;
}

// ---------- 자동 목표 ----------
// j 번째 탐침: 아는 칸 상자 안 무작위 칸이 로봇 지도에서 설 수 있고(계획 반경) 깊이 카메라 층에 안 막혔으면 열쇠 (등급 · 난수 · 칸), 아니면 ~0.
// 가장 작은 열쇠가 목표 — 차례와 무관 (GPU 는 스레드마다 하나씩 atomicMin)
DEV uint64_t lowrl_goal_probe(const EnvBatch& b, const EnvState& e, int env, int j, mdp::Robot rb) {
  const EnvCfg& c = b.cfg;
  const MapDev& m = b.maps[e.map];
  if (e.kb_x1 < 0) return ~0ull;
  uint64_t s = b.lowrl[env].rng + (uint64_t)(j + 1) * 0x9E3779B97F4A7C15ull;
  const uint64_t h = dm::splitmix64(s);
  const int w = e.kb_x1 - e.kb_x0 + 1, hh = e.kb_y1 - e.kb_y0 + 1;
  const int x = e.kb_x0 + (int)((h & 0xffffffull) % (uint64_t)w), y = e.kb_y0 + (int)(((h >> 24) & 0xfffull) * (uint64_t)hh >> 12);
  if (x < 0 || y < 0 || x >= m.gw || y >= m.gh) return ~0ull;
  const int16_t* L = b.lmap + (size_t)env * c.size * c.size;
  if (!mdp::standable(mdp::MapView{L, c.size, m.gw, m.gh}, x, y, rb)) return ~0ull;
  if (cam_lethal(b, env, x, y)) return ~0ull;
  const float dx = m.ox + ((float)x + 0.5f) * c.res - (e.x + e.ex), dy = m.oy + ((float)y + 0.5f) * c.res - (e.y + e.ey);
  const float d2 = dx * dx + dy * dy;
  if (d2 >= kLrGoalDMax * kLrGoalDMax || d2 < kLrGoalD2 * kLrGoalD2) return ~0ull;
  const uint64_t grade = d2 >= kLrGoalD0 * kLrGoalD0 ? 0 : (d2 >= kLrGoalD1 * kLrGoalD1 ? 1 : 2);
  return (grade << 60) | (((h >> 36) & 0xfffffffull) << 32) | (uint64_t)(uint32_t)(y * m.gw + x);
}
// 스레드 0: 가장 작은 열쇠 → LowCmd. 못 찾았으면 목표 없음(−1)으로 두고 다음 걸음에 다른 난수로 또
DEV void lowrl_goal_set(const EnvBatch& b, const EnvState& e, int env, uint64_t key) {
  LowRl& R = b.lowrl[env];
  uint64_t s = R.rng;
  R.rng = dm::splitmix64(s);
  LowCmd& cmd = b.lowcmd[env];
  cmd.follow = 2;
  cmd.target = key == ~0ull ? -1 : (int32_t)(uint32_t)(key & 0xffffffffull);
  R.need_goal = cmd.target < 0 ? 1 : 0;
  R.tgt_seen = -2;   // 같은 칸이 또 뽑혀도 새로 계획
  R.g_step = e.step;
  R.g_len = -1.f;
  R.rem_ok = 0;
}

// ---------- 탐사 판 학습의 목표 (EnvCfg::low_auto == 2, 스레드 0 — 준비 단계 맨 앞, 탐사 판 환경만: low_step.h low_xenv) ----------
// 위층 0 번 후보 규칙을 환경 안에서 돌린다. scripts/combine_run.cpp 의 호스트 규칙(평가)과 같은 차례·같은 식 — combine_run --auto 가 두 길의 결과가 같은지 잰다:
//  (걸음 뒤) 막힘 신호가 새로 서면 그 칸을 피하는 칸에 넣고 다시 고름 · 정책 환경이 통과 / 경로 끝이 목표 밖 / 경로 없음이면 (뒤 둘은 그 칸을 피하고) 다시 고름
//  (결정: 걸음 번호가 decide_steps 의 배수이거나 다시 고름) 피하는 칸이 아닌 첫 후보 칸. 그 덩어리가 지금 목표 칸을 품으면(cand_cur) 목표 칸을 그대로.
//  후보가 없으면 LowCmd{−1, 0} — 규칙 추종기의 "목표 없음"(제자리 회전)이 몬다 (경로 계획 단계가 후보 번호 −1 로 — env_kernel.cu env_plan_kernel · env_ref.cpp)
// 근거 (mdp/README "아래층 RL — 탐사 판 학습"): 평가 때 받는 목표(프론티어 후보 칸 · 직선 평균 2.5 m · 1.7 s 마다 바뀜 · 벽 가까이)를 학습 때 그대로 받게 한다
DEV bool lowx_avoided(const EnvCfg& c, const MapDev& m, const LowRl& R, int cell, int step) {
  if (cell < 0) return false;
  const float q = (c.rec_bl_r > 0.f ? c.rec_bl_r : 0.5f) / c.res;
  for (int k = 0; k < kLowXBl; ++k) {
    if (R.x_cell[k] < 0 || step >= R.x_until[k]) continue;
    const int dx = cell % m.gw - R.x_cell[k] % m.gw, dy = cell / m.gw - R.x_cell[k] / m.gw;
    if ((float)(dx * dx + dy * dy) <= q * q) return true;
  }
  return false;
}
DEV void lowx_avoid_add(const EnvCfg& c, LowRl& R, int cell, int step) {
  if (cell < 0) return;
  const int bl = c.rec_bl_steps > 0 ? c.rec_bl_steps : (int)(10.f / ((float)c.substeps * c.dt) + 0.5f);
  R.x_cell[R.x_n % kLowXBl] = cell;
  R.x_until[R.x_n % kLowXBl] = step + bl;
  R.x_n += 1;
}
DEV void lowrl_goal_explore(const EnvBatch& b, const EnvState& e, int env) {
  const EnvCfg& c = b.cfg;
  const MapDev& m = b.maps[e.map];
  LowRl& R = b.lowrl[env];
  LowCmd& cmd = b.lowcmd[env];
  const LowEp& L = b.low[env];
  if (e.step == 0) {   // 새 판 (걸음 사이 값은 lowrl_clear 가 지움)
    cmd.target = -1;
    cmd.follow = 0;
    R.x_redo = 1;
    R.x_sig = R.stall;   // 지난 판에서 남은 신호는 새 가장자리로 안 봄
  } else {
    if (R.stall && !R.x_sig) {
      lowx_avoid_add(c, R, R.stall_tgt, e.step);
      R.x_redo = 1;
    }
    R.x_sig = R.stall;
    if (cmd.follow == 2) {
      const bool reached = L.reach_step == e.step && L.reached > 0;
      const bool off = !reached && L.hold != 0;
      const bool nopath = !reached && !off && e.step > R.x_last && (e.goal_cell != cmd.target || e.p_n <= 0);
      if (reached || off || nopath) {
        if (!reached) lowx_avoid_add(c, R, cmd.target, e.step);
        R.x_redo = 1;
      }
    }
  }
  if (!(e.step % c.decide_steps == 0 || R.x_redo)) return;
  R.x_redo = 0;
  R.x_last = e.step;
  const int32_t* C = b.cand + (size_t)env * mdp::hier::K;
  int k = -1;
  for (int j = 0; j < mdp::hier::K && k < 0; ++j)
    if (C[j] >= 0 && !lowx_avoided(c, m, R, C[j], e.step)) k = j;
  int tgt = k >= 0 ? C[k] : -1;
  if (k >= 0 && ((b.cand_cur[env] >> k) & 1) && cmd.follow == 2 && cmd.target >= 0 && cmd.target == e.goal_cell && e.p_n > 0 && !lowx_avoided(c, m, R, cmd.target, e.step))
    tgt = cmd.target;
  if (tgt >= 0) {
    if (tgt != cmd.target || cmd.follow != 2) {   // 새 목표 (누적용: 처음 계획 길이·걸린 걸음을 다시 — lowrl_pack 이 채움)
      R.g_step = e.step;
      R.g_len = -1.f;
      R.x_paid = 0;
    }
    cmd.target = tgt;
    cmd.follow = 2;
  } else {
    cmd.target = -1;
    cmd.follow = 0;
  }
}

// ---------- 경로 위 자리 ----------
struct LowTrack {
  int32_t has_path, has_goal;
  float rem;               // 남은 경로 길이 (m): 로봇 ~ 경로 위 가장 가까운 점 + 그 점부터 경로 끝까지
  float sx[2], sy[2];      // 부목표 (세계 좌표, 지도 자세 기준)
  float gx, gy;            // 목적지 칸 가운데
};
// 가장 가까운 점은 *p_i (지금 구간)부터 앞으로만 찾고 *p_i 를 거기로 옮긴다 (hier_path.h path_follow 와 같은 규칙)
DEV void lowrl_track(const EnvBatch& b, const EnvState& e, int env, int* p_i, LowTrack& t) {
  const EnvCfg& c = b.cfg;
  const MapDev& m = b.maps[e.map];
  const float px = e.x + e.ex, py = e.y + e.ey;
  const int tgt = b.lowcmd[env].target;
  t.has_goal = b.lowcmd[env].follow != 0 && tgt >= 0 ? 1 : 0;
  t.gx = t.has_goal ? m.ox + ((float)(tgt % m.gw) + 0.5f) * c.res : px;
  t.gy = t.has_goal ? m.oy + ((float)(tgt / m.gw) + 0.5f) * c.res : py;
  t.has_path = t.has_goal && e.goal_cell == tgt && e.p_n > 0 ? 1 : 0;
  t.rem = 0.f;
  t.sx[0] = t.sx[1] = t.gx;
  t.sy[0] = t.sy[1] = t.gy;
  if (!t.has_path) return;
  const auto wx = [&](int i) { return m.ox + ((float)e.wp[2 * i] + 0.5f) * c.res; };
  const auto wy = [&](int i) { return m.oy + ((float)e.wp[2 * i + 1] + 0.5f) * c.res; };
  int bi = *p_i < 0 ? 0 : *p_i;
  if (bi > e.p_n - 1) bi = e.p_n - 1;
  float qx = wx(bi), qy = wy(bi);
  if (e.p_n >= 2) {
    float bd = 3e38f;
    for (int i = bi; i + 1 < e.p_n; ++i) {
      const float ax = wx(i), ay = wy(i), sx = wx(i + 1) - ax, sy = wy(i + 1) - ay;
      const float l2 = sx * sx + sy * sy;
      float u = l2 > 0.f ? ((px - ax) * sx + (py - ay) * sy) / l2 : 0.f;
      u = u < 0.f ? 0.f : (u > 1.f ? 1.f : u);
      const float x = ax + u * sx, y = ay + u * sy;
      const float d2 = (x - px) * (x - px) + (y - py) * (y - py);
      if (d2 < bd) { bd = d2; bi = i; qx = x; qy = y; }
    }
  } else {
    bi = 0;
  }
  *p_i = bi;
  // 경로 따라 걷기: 부목표 둘 · 남은 길이
  float along = 0.f, ux = qx, uy = qy;
  int got = 0;
  const float look[2] = {kLrLook0, kLrLook1};
  for (int i = bi; i + 1 < e.p_n; ++i) {
    const float ex2 = wx(i + 1), ey2 = wy(i + 1);
    const float d = sqrtf((ex2 - ux) * (ex2 - ux) + (ey2 - uy) * (ey2 - uy));
    while (got < 2 && d > 0.f && along + d >= look[got]) {
      const float r = (look[got] - along) / d;
      t.sx[got] = ux + (ex2 - ux) * r;
      t.sy[got] = uy + (ey2 - uy) * r;
      ++got;
    }
    along += d;
    ux = ex2;
    uy = ey2;
  }
  for (; got < 2; ++got) { t.sx[got] = ux; t.sy[got] = uy; }   // 경로가 짧으면 경로 끝
  t.rem = sqrtf((qx - px) * (qx - px) + (qy - py) * (qy - py)) + along;
}

// ---------- 사람 추적 (정책 관측 "가장 가까운 움직이는 것" — EnvCfg::lo_mov, 사람 닿음 구분 contact.h): 라이다 덩어리 ----------
// 움직이는 것 하나 (지도 좌표 자리·속도·반지름)
struct LrMover { float x, y, vx, vy, r; };

// wk = 그 장을 쏠 때 로봇 각속도. 관측 칸은 ROS 가 적는 각도(펴기 전)라 빔 시각(스캔 시작 + (각도 + fov/2)/2π × scan_time)만큼 돈 몫을 여기서 편다
// (env.h 의 빔 펴기와 같은 식 — 안 펴면 ω 가 바뀔 때 선 물체가 움직여 보인다: ω 0.3 rad/s 차이 · 2 m 거리에서 겉보기 0.2 m/s)
// sag > 0 (칼만 추적): 둥근 덩어리만 — 양 끝을 이은 현에서 센서 쪽으로 가장 튀어나온 점까지 거리(활꼴 높이)가 현 길이 × sag 넘는 것.
// 사람 원기둥(반지름 0.2 ~ 0.3 m)은 0.23 ~ 0.5, 납작한 면(선반 옆·벽 조각 — 로봇이 가면 보이는 끝이 미끄러져 움직여 보인다)은 0 근처
DEV int lowrl_clusters(const EnvBatch& b, const EnvState& e, int env, int slot, float wk, float* cx, float* cy, float* cr, float sag = 0.f, float w0 = 0.f) {
  const EnvCfg& c = b.cfg;
  const float* R = b.lidar_ring + ((size_t)env * kLidarRing + slot) * mdp::obs::LIDAR_BINS;
  const float px = e.lid_pose[3 * slot], py = e.lid_pose[3 * slot + 1];
  float s, co;
  dm::sincosf_d(e.lid_pose[3 * slot + 2], &s, &co);
  const float tk = e.noise.scan_time / kTwoPi;
  int n = 0, k0 = -1, kp = -1;   // kp = 덩어리의 마지막 유효 칸
  bool left_ok = false;
  float rmin = 0.f;
  for (int k = 0; k <= mdp::obs::LIDAR_BINS; ++k) {
    const bool valid = k < mdp::obs::LIDAR_BINS && R[k] < c.map_max_range;
    const bool cont = valid && k0 >= 0 && dm::absf(R[k] - R[kp]) < kLrClusJump;
    // 칼만 추적(sag > 0): 한 칸 빈 곳(빠진 빔)은 건너 잇는다 — 안 그러면 사람 원기둥이 조각나 둥근 모양을 못 잰다
    if (sag > 0.f && !valid && k0 >= 0 && kp == k - 1 && k + 1 < mdp::obs::LIDAR_BINS && R[k + 1] < c.map_max_range &&
        dm::absf(R[k + 1] - R[kp]) < kLrClusJump)
      continue;
    if (k0 >= 0 && !cont) {   // 덩어리 [k0, kp] 닫음
      const int k1 = kp;
      const bool right_ok = k < mdp::obs::LIDAR_BINS && (!valid || R[k] > R[k1] + kLrClusJump);   // 시야 끝에 걸린 덩어리는 버림
      if (left_ok && right_ok && k1 - k0 >= 2 && rmin < kLrClusRange && n < kLrClus) {
        const float g0 = mdp::obs::kLidarMin + ((float)k0 + 0.5f) * mdp::obs::kLidarBin, g1 = mdp::obs::kLidarMin + ((float)k1 + 0.5f) * mdp::obs::kLidarBin;
        const float a0 = g0 + wk * tk * (g0 + 0.5f * e.noise.fov), a1 = g1 + wk * tk * (g1 + 0.5f * e.noise.fov);
        float s0, c0, s1, c1;
        dm::sincosf_d(a0, &s0, &c0);
        dm::sincosf_d(a1, &s1, &c1);
        const float wx = R[k1] * c1 - R[k0] * c0, wy = R[k1] * s1 - R[k0] * s0;
        const float chord = sqrtf(wx * wx + wy * wy);
        bool round = true;
        if (sag > 0.f) {
          float top = 0.f;
          const float x0 = R[k0] * c0, y0 = R[k0] * s0, sg = (wx * y0 - wy * x0) >= 0.f ? -1.f : 1.f;   // 현에 수직인 두 쪽 가운데 센서(원점) 쪽이 +
          for (int j = k0 + 1; j < k1; ++j) {
            if (!(R[j] < c.map_max_range)) continue;
            const float gj = mdp::obs::kLidarMin + ((float)j + 0.5f) * mdp::obs::kLidarBin, aj = gj + wk * tk * (gj + 0.5f * e.noise.fov);
            float sj, cj;
            dm::sincosf_d(aj, &sj, &cj);
            const float dxj = R[j] * cj - x0, dyj = R[j] * sj - y0;
            top = dm::maxf(top, sg * (wx * dyj - wy * dxj));
          }
          round = chord >= w0 && top >= sag * chord * chord;   // top 은 (높이 × 현 길이)
        }
        if (round && chord >= kLrClusW0 && chord <= kLrClusW1) {
          const float rr = dm::clampf(0.5f * chord, 0.10f, 0.30f), d = rmin + rr;
          float sm, cm;
          dm::sincosf_d(0.5f * (a0 + a1), &sm, &cm);
          const float lx = c.mount_x + d * cm, ly = c.mount_y + d * sm;
          cx[n] = (px + co * lx) - s * ly;
          cy[n] = (py + s * lx) + co * ly;
          cr[n] = rr;
          ++n;
        }
      }
      k0 = -1;
    }
    if (valid && k0 < 0) {
      k0 = k;
      left_ok = k > 0 && (!(R[k - 1] < c.map_max_range) || R[k - 1] > R[k] + kLrClusJump);
      rmin = R[k];
      kp = k;
    } else if (valid) {
      rmin = dm::minf(rmin, R[k]);
      kp = k;
    }
  }
  return n;
}

// ---------- 사람 추적: 덩어리를 장마다 이어 가는 등속 칼만 필터 (lowrl_act 가 lo_mov 일 때 걸음마다) ----------
// (옛 세 장 잇기 lowrl_movers — 지운 사람 비켜 주기가 쓰던 것 — 는 걸음마다 세 장을 새로 이어 한 장이라도 덩어리가 빠지면(가림·벽에 붙음·폭 벗어남) 그 걸음은 사람이 없다 — 닿기 전 2 s 의
// 보인 걸음 가운데 24 ~ 29 % 만 잡음(low_p0, mdp/README "사람 추적"). 여기서는 덩어리마다 자리·속도를 들고 가며(LowRl::tk_*) 새 장의
// 덩어리와 가까운 것끼리 잇고, 못 이은 장은 같은 속도로 옮겨 둔다. 꼴은 leg_tracker(Leigh et al., ICRA 2015 "Person Tracking and Following
// with 2D Laser Scanners"): 덩어리 검출 → 등속 칼만 필터 → 가장 가까운 것끼리 잇기(전역 최근접). 두 축이 같은 잡음이라 공분산은 축 하나 몫만 든다.
// 걷는 사람 = 속도가 kLrMoveV0 ~ kLrMoveV1 인 장이 tk_conf 번 이어지고 그동안 tk_disp 넘게 옮겨 간 덩어리 (선 물체의 자리 떨림을 거름).
DEV void lowrl_trk_step(const EnvBatch& b, const EnvState& e, int env) {
  const EnvCfg& c = b.cfg;
  LowRl& R = b.lowrl[env];
  const int sA = e.lid_head;
  if (sA == R.tk_head) return;   // 새 장 없음
  const float step = (float)c.substeps * c.dt;
  int adv = 1;
  if (R.tk_head < 0) {
    for (int i = 0; i < kLrTrk; ++i) R.tk_hit[i] = 0;
  } else {
    adv = (sA - R.tk_head + kLidarRing) % kLidarRing;
  }
  R.tk_head = sA;
  const float dt = step * (float)adv, q = c.tk_qa * c.tk_qa;
  const int sP = (sA + kLidarRing - 1) % kLidarRing, sQ = (sA + kLidarRing - 2) % kLidarRing;
  const float wA = dm::wrap_pi(e.lid_pose[3 * sA + 2] - e.lid_pose[3 * sP + 2]) / step;
  const float wB = dm::wrap_pi(e.lid_pose[3 * sP + 2] - e.lid_pose[3 * sQ + 2]) / step;
  float rz = c.tk_sz * c.tk_sz;
  if (dm::absf(wA - wB) > kLrTrkWGate) rz = rz * 9.f;   // 각속도가 크게 바뀌는 중: 편 뒤에도 남는 오차가 큼 — 그 장은 덜 믿는다 (σ 3 배)
  // 예측 (등속, 가속 잡음 σ = tk_qa)
  for (int i = 0; i < kLrTrk; ++i) {
    if (R.tk_hit[i] <= 0) continue;
    R.tk_x[i] = R.tk_x[i] + R.tk_vx[i] * dt;
    R.tk_y[i] = R.tk_y[i] + R.tk_vy[i] * dt;
    R.tk_pp[i] = R.tk_pp[i] + dt * (2.f * R.tk_pv[i] + dt * R.tk_vv[i]) + 0.25f * q * dt * dt * dt * dt;
    R.tk_pv[i] = R.tk_pv[i] + dt * R.tk_vv[i] + 0.5f * q * dt * dt * dt;
    R.tk_vv[i] = R.tk_vv[i] + q * dt * dt;
  }
  float cx[kLrClus], cy[kLrClus], cr[kLrClus];
  const int nc = lowrl_clusters(b, e, env, sA, wA, cx, cy, cr, c.tk_sag, c.tk_w0);
  int used[kLrClus], done[kLrTrk];
  for (int j = 0; j < kLrClus; ++j) used[j] = 0;
  for (int i = 0; i < kLrTrk; ++i) done[i] = 0;
  // 잇기: 남은 (덩어리, 새 덩어리) 짝 가운데 가장 가까운 것부터 (문 tk_gate 안)
  for (;;) {
    float best = c.tk_gate * c.tk_gate;
    int bi = -1, bj = -1;
    for (int i = 0; i < kLrTrk; ++i) {
      if (R.tk_hit[i] <= 0 || done[i]) continue;
      for (int j = 0; j < nc; ++j) {
        if (used[j]) continue;
        const float dx = cx[j] - R.tk_x[i], dy = cy[j] - R.tk_y[i], dd = dx * dx + dy * dy;
        if (dd < best) { best = dd; bi = i; bj = j; }
      }
    }
    if (bi < 0) break;
    done[bi] = 1;
    used[bj] = 1;
    const float s = R.tk_pp[bi] + rz, kp = R.tk_pp[bi] / s, kv = R.tk_pv[bi] / s;
    const float ix = cx[bj] - R.tk_x[bi], iy = cy[bj] - R.tk_y[bi];
    R.tk_x[bi] = R.tk_x[bi] + kp * ix;
    R.tk_y[bi] = R.tk_y[bi] + kp * iy;
    R.tk_vx[bi] = R.tk_vx[bi] + kv * ix;
    R.tk_vy[bi] = R.tk_vy[bi] + kv * iy;
    R.tk_vv[bi] = R.tk_vv[bi] - kv * R.tk_pv[bi];
    R.tk_pv[bi] = (1.f - kp) * R.tk_pv[bi];
    R.tk_pp[bi] = (1.f - kp) * R.tk_pp[bi];
    R.tk_hit[bi] += 1;
    R.tk_miss[bi] = 0;
    R.tk_r[bi] = dm::maxf(R.tk_r[bi], cr[bj]);
    const float sp = R.tk_vx[bi] * R.tk_vx[bi] + R.tk_vy[bi] * R.tk_vy[bi];
    if (R.tk_hit[bi] >= 3 && sp >= kLrMoveV0 * kLrMoveV0 && sp <= kLrMoveV1 * kLrMoveV1) {
      R.tk_mov[bi] += 1;
    } else {   // 안 움직임: 옮겨 간 거리를 여기서부터 다시 잰다
      R.tk_mov[bi] = 0;
      R.tk_x0[bi] = R.tk_x[bi];
      R.tk_y0[bi] = R.tk_y[bi];
    }
  }
  int n = 0;
  for (int i = 0; i < kLrTrk; ++i) {
    if (R.tk_hit[i] <= 0) continue;
    if (!done[i]) {
      R.tk_miss[i] += 1;
      if (R.tk_miss[i] > c.tk_miss || (R.tk_hit[i] < 3 && R.tk_miss[i] > 1)) { R.tk_hit[i] = 0; continue; }
    }
    ++n;
  }
  // 못 이은 새 덩어리 = 새로 들고 갈 것 (빈 칸이 있을 때)
  int slot = 0;
  for (int j = 0; j < nc; ++j) {
    if (used[j]) continue;
    while (slot < kLrTrk && R.tk_hit[slot] > 0) ++slot;
    if (slot >= kLrTrk) break;
    R.tk_hit[slot] = 1;
    R.tk_miss[slot] = 0;
    R.tk_mov[slot] = 0;
    R.tk_x[slot] = R.tk_x0[slot] = cx[j];
    R.tk_y[slot] = R.tk_y0[slot] = cy[j];
    R.tk_vx[slot] = R.tk_vy[slot] = 0.f;
    R.tk_pp[slot] = c.tk_sz * c.tk_sz;
    R.tk_pv[slot] = 0.f;
    R.tk_vv[slot] = kLrTrkV0 * kLrTrkV0;
    R.tk_r[slot] = cr[j];
    ++n;
  }
  R.tk_n = n;
}
DEV int lowrl_trk_movers(const EnvBatch& b, int env, LrMover* M) {
  const EnvCfg& c = b.cfg;
  const LowRl& R = b.lowrl[env];
  int n = 0;
  for (int i = 0; i < kLrTrk && n < kLrMovers; ++i) {
    if (R.tk_hit[i] <= 0 || R.tk_mov[i] < c.tk_conf) continue;
    const float dx = R.tk_x[i] - R.tk_x0[i], dy = R.tk_y[i] - R.tk_y0[i];
    if (dx * dx + dy * dy < c.tk_disp * c.tk_disp) continue;
    LrMover& m = M[n++];
    m.x = R.tk_x[i];
    m.y = R.tk_y[i];
    m.vx = R.tk_vx[i];
    m.vy = R.tk_vy[i];
    m.r = R.tk_r[i];
  }
  return n;
}

// ---------- 행동 (걸음 첫머리, 스레드 0) ----------
// 정책 환경이면 명령을 정하고 true (호출한 쪽은 추종기를 건너뜀). a: 그 환경의 행동 줄 — a[0] = v, a[1] = ω (정규 단위 [−1, 1], 밖은 자름)
DEV bool lowrl_act(const EnvBatch& b, EnvState& e, int env, const float* a) {
  if (!b.lowrl) return false;
  LowRl& R = b.lowrl[env];
  R.pol = 0;
  if (b.lowcmd[env].follow != 2) return false;
  const EnvCfg& c = b.cfg;
  const float n0 = c.lg_noback ? dm::maxf(0.f, clip1(a[0])) : clip1(a[0]), n2 = clip1(a[1]);   // 후진 없음: 음수 v 행동은 0 (직전 행동 관측·매끄러움·후진 항도 자른 값으로 — 후진 항은 저절로 0)
  float v = n0 >= 0.f ? n0 * c.act_v_fwd : n0 * c.act_v_back, w = n2 * c.act_w_max;
  const float wheel = dm::absf(v) + c.wheel_k * dm::absf(w);   // 가장 빠른 바퀴가 한도를 넘으면 같은 비로 (속도 행동과 같은 규칙 — 물리 한도)
  if (wheel > c.wheel_max) {
    const float k = c.wheel_max / wheel;
    v = v * k;
    w = w * k;
  }
  // 정책 명령은 그대로 나간다 — 명령을 고치는 층(앞 틈 정지·몸통 지킴이·사람 비켜 주기)은 2026-11-14 에 없앰 (사용자 결정: 순수 RL)
  if (c.lo_mov) lowrl_trk_step(b, e, env);   // 사람 추적 (인지 — 관측 "가장 가까운 움직이는 것"·사람 닿음 구분)
  e.cmd_v = v;
  e.cmd_vy = 0.f;
  e.cmd_w = w;
  e.last_action = e.p_slot;
  e.vs = v;   // velocity_smoother 상태도 같이 (규칙 추종으로 넘어갈 때 이어지게)
  e.ws = w;
  e.p_rec = 0;
  PathWhy why;
  why.kind = kWhyGo;
  why.bind = kLowFull;
  why.vc = v;
  why.err = 0.f;
  why.d_goal = 0.f;
  low_begin(b, e, env, why, 0, false, false, v);
  for (int i = mdp::act::DIM * 4 - 1; i >= mdp::act::DIM; --i) e.last_act[i] = e.last_act[i - mdp::act::DIM];
  e.last_act[0] = n0;   // 정책이 낸 값 (바퀴 한도 앞) — 매끄러움 항·관측이 같은 값
  e.last_act[1] = 0.f;
  e.last_act[2] = n2;
  e.last_act[3] = 0.f;
  R.pol = 1;
  return true;
}

// ---------- 사람 마주침 (보상 전용, 라이다 덩어리 칼만 추적 — 센서만. 2026-11-14 전엔 정답 사람 자리·속도) ----------
// 로봇과 추적한 사람이 지금 속도 그대로 가면 lr_ppl_t 초 안에 가장 가까워지는 때 t* 와 그때 간격을 본다. 간격 < lr_ppl_m 인 사람 가운데 (1 − t* / lr_ppl_t) 의 최댓값(0 ~ 1).
// 로봇 속도 = 바퀴 오도메트리 v 를 지도 자세 방향으로. 사람 자리·속도 = 추적(lowrl_trk_movers, 지도 좌표). 추적이 꺼져 있으면 0.
// 근거: 사람 충돌 진단에서 닿은 사람은 모두 로봇을 무시하고 걷는 사람이고, 2 s 전부터 라이다에 보였는데 정책이 길을 안 비킴(mdp/README).
// 닿기 전부터 값을 주는 꼴은 DRL-VO(arXiv 2301.06512)의 속도 장애물(VO) 방향 보상·NavRL(arXiv 2409.15634)의 동적 안전 보상과 같은 자리
DEV float lowrl_ppl_risk(const EnvBatch& b, const EnvState& e, int env) {
  const EnvCfg& c = b.cfg;
  if (!c.lo_mov) return 0.f;   // 추적은 lo_mov 일 때만 돈다
  LrMover M[kLrMovers + 1];
  const int n = lowrl_trk_movers(b, env, M);
  float s, co;
  dm::sincosf_d(e.th + e.eth, &s, &co);
  const float rvx = e.drive.wv * co, rvy = e.drive.wv * s, rx = e.x + e.ex, ry = e.y + e.ey;
  float worst = 0.f;
  for (int i = 0; i < n; ++i) {
    const float px = M[i].x - rx, py = M[i].y - ry;
    const float ux = M[i].vx - rvx, uy = M[i].vy - rvy, u2 = ux * ux + uy * uy;
    float t = 0.f;
    if (u2 > 1e-6f) t = dm::clampf(-(px * ux + py * uy) / u2, 0.f, c.lr_ppl_t);
    const float qx = px + ux * t, qy = py + uy * t;
    const float gap = sqrtf(qx * qx + qy * qy) - M[i].r - c.half_len;
    if (gap < c.lr_ppl_m) worst = dm::maxf(worst, 1.f - t / c.lr_ppl_t);
  }
  return worst;
}

// ---------- 걸음 끝 (finish_step 맨 끝, 스레드 0): 보상·누적 ----------
DEV void lowrl_finish(const EnvBatch& b, EnvState& e, int env) {
  if (!b.lowrl) return;
  LowRl& R = b.lowrl[env];
  const bool xm = b.cfg.low_auto == 2 && low_xenv(b, env);   // 탐사 판 학습의 탐사 판 환경
  if (R.pol) {
    const EnvCfg& c = b.cfg;
    const LowEp& L = b.low[env];
    LowRlStat& S = b.lowstat[env];
    LowTrack t;
    int pi = e.p_i;
    lowrl_track(b, e, env, &pi, t);
    e.p_i = pi;
    const bool reached = L.reach_step == e.step && L.reached > 0;
    // 탐사 판 학습: 통과는 목표마다 처음 한 번만 센다 (탐사 판은 통과 반경 안에 있는 걸음마다 표시가 서고, 위층 규칙이 같은 칸을 다시 줄 수 있음)
    const bool reach_new = reached && !(xm && R.x_paid);
    if (xm && reached) R.x_paid = 1;
    const bool off = !reached && L.hold != 0;                                   // 경로 끝에 왔는데 통과 반경 밖
    const bool nopath = !reached && !off && t.has_goal && !t.has_path;          // 경로를 못 짬
    float r[kLrTerms];
    for (int k = 0; k < kLrTerms; ++k) r[k] = 0.f;
    if (R.rem_ok && t.has_path) r[kLrProgress] = kLrProgressK * dm::clampf(R.rem_prev - t.rem, -kLrProgressClip, kLrProgressClip);
    r[kLrTime] = kLrTimeK;
    if (reach_new && (!xm || R.g_len >= kLrXReachLen)) r[kLrReach] = kLrReachK;   // 탐사 판 학습: 처음 계획 길이 kLrXReachLen 넘는 목표만 (위 상수)
    if (e.contact) r[kLrCollision] = (e.s_kind & kConPerson) ? c.lr_pcol_k : kLrCollisionK;   // 닿음 (센서 — contact.h)
    {   // 근접 (2026-11-14 진행 방향): 몸통이 다가가는 쪽의 센서 간격 (contact.h s_dir — 이번 장 라이다 점 · 로봇 지도 막힘 · 깊이 층 가운데
        // 몸통 속도로 다가가는 점만. 옛 전방위 s_clear 는 서 있거나 비켜 가는 벽에도 벌을 줌). 크기·띠는 그대로
      const float d = e.s_dir;
      if (d < kLrProxDist) r[kLrProximity] = kLrProxK * (1.f - dm::maxf(d, 0.f) / kLrProxDist);
      if (c.lr_mprox_k != 0.f && d < c.lr_mprox_d) {   // 움직임 근접: 몸통 가장 빠른 점 속도 = |v| + 꼭짓점 반경 × |ω|
        const float rc = sqrtf(c.half_len * c.half_len + c.half_wid * c.half_wid);
        r[kLrProximity] += c.lr_mprox_k * (dm::absf(e.drive.wv) + rc * dm::absf(e.gyro)) * (1.f - dm::maxf(d, 0.f) / c.lr_mprox_d);   // 바퀴 오도메트리 v · 자이로
      }
      if (c.lr_ppl_k != 0.f) r[kLrProximity] += c.lr_ppl_k * lowrl_ppl_risk(b, e, env);
    }
    if (c.lr_ttc_k != 0.f && e.s_close > 1e-3f && e.s_dir < kSenseFar) {   // 충돌 시간 항
      const float ttc = dm::maxf(e.s_dir, 0.f) / e.s_close;
      if (ttc < kLrTtcT) r[kLrTtc] = c.lr_ttc_k * (1.f - ttc / kLrTtcT);
    }
    if (e.step > 1) {
      const float d0 = e.last_act[0] - e.last_act[mdp::act::DIM], d2 = e.last_act[2] - e.last_act[mdp::act::DIM + 2];
      r[kLrSmooth] = kLrSmoothK * sqrtf(d0 * d0 + d2 * d2);
    }
    if (c.lr_w_k != 0.f) r[kLrSmooth] += c.lr_w_k * dm::maxf(0.f, dm::absf(e.last_act[2]) * c.act_w_max - c.lr_w0);
    if (e.last_act[0] < 0.f) r[kLrBack] = kLrBackK * -e.last_act[0];
    float sum = 0.f;
    for (int k = 0; k < kLrTerms; ++k) {
      sum += r[k];
      S.term[k] += r[k];
    }
    e.reward = sum;
    S.steps += 1;
    S.back_steps += e.last_act[0] < 0.f ? 1 : 0;
    {
      const float dx = e.x - L.x0, dy = e.y - L.y0;
      S.dist += sqrtf(dx * dx + dy * dy);
    }
    if (reach_new) {
      S.reached += 1;
      if (R.g_len >= 0.f) {
        S.reach_len += R.g_len;
        S.reach_steps += e.step - R.g_step;
      }
    }
    if (off || nopath) S.giveup += 1;
    if (reached || off || nopath) R.need_goal = 1;
    if (e.done) {
      S.episodes += 1;
      S.gt_wall += (e.gt_touch & 1) != 0;   // 판 동안 정답 충돌 (기록 전용)
      S.gt_person += (e.gt_touch & 2) != 0;
      if (e.done == mdp::cur::kCollision || e.done == mdp::cur::kCollisionPerson) {   // 닿음 (센서)으로 끝
        if (e.done == mdp::cur::kCollisionPerson) S.hit_person += 1;
        else S.hit_wall += 1;
      } else if (xm) {   // 탐사 판: 닫힘 성공·진행 없음은 아래 x_* 에
      } else if (L.end == kLowEndStall) {
        S.stall += 1;
      } else if (L.end != kLowEndGoals) {
        S.timeout += 1;
      }
    }
  } else if (b.cfg.low_auto == 2) {   // 탐사 판 학습: 규칙 추종기가 몬 걸음(후보 없음)은 보상 0 — 정책 행동이 안 쓰인 걸음
    e.reward = 0.f;
    b.lowstat[env].x_rule_steps += 1;
  }
  if (xm && e.done) {
    LowRlStat& S = b.lowstat[env];
    S.x_eps += 1;
    if (e.done == mdp::cur::kDeclareOk) { S.x_ok += 1; S.x_ok_steps += e.step; }
    else if (e.done == mdp::cur::kCollision) S.x_wall += 1;
    else if (e.done == mdp::cur::kCollisionPerson) S.x_person += 1;
    else if (e.done == mdp::cur::kNoProgress) S.x_idle += 1;
  }
  if ((e.done || low_xenv(b, env)) && b.low[env].end == kLowEndStall) {   // 위층에 알리는 막힘 신호 (정책·규칙 추종 모두 — 합친 탐사는 판을 안 끝내고 신호만)
    R.stall = 1;
    R.stall_tgt = b.lowcmd[env].follow != 0 ? b.lowcmd[env].target : e.goal_cell;
  }
  if (e.done) lowrl_clear(R);
}

// ---------- 정책 관측 (준비 단계 끝 — 판이 끝났으면 리셋·새 판 관측 뒤) ----------
// 그림 한 바이트 (차례 무관 — GPU 는 블록 스레드가 나눠 채움)
DEV uint8_t lowrl_img(const EnvBatch& b, const EnvState& e, int env, int i) {
  const EnvCfg& c = b.cfg;
  const uint8_t* I = b.obs_img + (size_t)env * mdp::obs::IMG_SIZE;
  if (i < mdp::obs::NEAR_SIZE) {
    const int hw = mdp::obs::NEAR_H * mdp::obs::NEAR_W;
    if (i / hw != kLowCamCh) return I[i];
    if (!b.cvox) return 0;
    // 깊이 카메라 장애물 층 (hier_path.h — 계획기가 막힘으로 보는 층): 픽셀(10 cm) 안 2 × 2 점(지도 칸 5 cm) 중 하나라도 막힘이면 255 (프론티어 채널과 같은 꼴)
    const MapDev& m = b.maps[e.map];
    const int p = i % hw, row = p / mdp::obs::NEAR_W, col = p % mdp::obs::NEAR_W;
    const float px = e.x + e.ex, py = e.y + e.ey;
    float s, co;
    dm::sincosf_d(e.th + e.eth, &s, &co);
    const float fwd = ((float)(mdp::obs::NEAR_H / 2 - row) - 0.5f) * kLrNearCell, left = ((float)(mdp::obs::NEAR_W / 2 - col) - 0.5f) * kLrNearCell;
    for (int q = 0; q < 4; ++q) {
      const float f2 = fwd + ((q & 1) ? 0.25f : -0.25f) * kLrNearCell, l2 = left + ((q & 2) ? 0.25f : -0.25f) * kLrNearCell;
      const float wx = (px + co * f2) - s * l2, wy = (py + s * f2) + co * l2;
      int cx, cy;
      if (cell_of(c, m, wx, wy, &cx, &cy) >= 0 && cam_lethal(b, env, cx, cy)) return 255;
    }
    return 0;
  }
  if (i < kLowDepthOff) {
    // 라이다: 픽셀 (행, 열) 의 Z 차례 번호 z = 4 · 각도 칸 + 프레임 — 2 × 2 합성곱 4 층이 그대로 "프레임 4 장 → 이웃 4 칸 → 16 칸 → 64 칸" 1D CNN 이 된다
    const int p = i - kLowLidarOff, row = p / kLowLidarHW, col = p % kLowLidarHW;
    int z = 0;
    for (int k = 0; k < 5; ++k) z |= (((col >> k) & 1) << (2 * k)) | (((row >> k) & 1) << (2 * k + 1));
    const int bin = z >> 2, f = z & 3;
    if (bin >= mdp::obs::LIDAR_BINS) return 0;
    const float r = dm::clampf(b.obs_1d[(size_t)env * b.o1d + f * mdp::obs::LIDAR_BINS + bin], 0.f, 1.f);
    return (uint8_t)(255.f * sqrtf(r) + 0.5f);   // 제곱근: 가까운 거리에 눈금을 더 (8 m 를 255 칸 — 1 m 가 90, 0.2 m 가 40)
  }
  return I[mdp::obs::NEAR_SIZE + mdp::obs::FAR_SIZE + (i - kLowDepthOff)];   // 깊이 (위층 관측 그대로)
}
// 숫자 관측 + 걸음 사이 값 (스레드 0). 계획 뒤에 부른다 — 남은 경로 길이를 방금 짠 경로 기준으로 적어 둠 (다음 걸음 진행 항의 기준)
DEV void lowrl_pack(const EnvBatch& b, EnvState& e, int env) {
  LowRl& R = b.lowrl[env];
  LowTrack t;
  int pi = e.p_i;
  lowrl_track(b, e, env, &pi, t);
  if (b.lowcmd[env].follow == 2) e.p_i = pi;   // 규칙 추종 환경의 p_i 는 추종기 것
  R.rem_prev = t.rem;
  R.rem_ok = t.has_path;
  if (t.has_path && R.g_len < 0.f) {   // 이 목표의 첫 경로
    R.g_len = t.rem;
    if (b.lowcmd[env].follow == 2) b.lowstat[env].goals += 1;
  }
  float* O = b.low_1d + (size_t)env * kLow1d;
  const float* V = b.obs_1d + (size_t)env * b.o1d + mdp::obs::LIDAR + mdp::obs::INTENSITY;
  for (int k = 0; k < kLow1d; ++k) O[k] = 0.f;
  O[kLoWheelV] = V[mdp::obs::WHEEL_V];
  O[kLoGyro] = V[mdp::obs::GYRO_Z];
  O[kLoSlip] = V[mdp::obs::SLIP];
  O[kLoEkfV] = V[mdp::obs::EKF];
  O[kLoEkfW] = V[mdp::obs::EKF + 1];
  const float px = e.x + e.ex, py = e.y + e.ey, pth = e.th + e.eth;
  const auto rel = [&](float x, float y, float norm, float* o) {   // (거리 / norm 자름 1, sin, cos) — 지도 자세 기준 방향
    const float dx = x - px, dy = y - py, d = sqrtf(dx * dx + dy * dy);
    float s = 0.f, co = 1.f;
    if (d > 1e-6f) dm::sincosf_d(dm::wrap_pi(dm::atan2f_d(dy, dx) - pth), &s, &co);
    o[0] = dm::minf(1.f, d / norm);
    o[1] = s;
    o[2] = co;
  };
  if (t.has_goal) {
    rel(t.sx[0], t.sy[0], kLrSubNorm, O + kLoSub);
    rel(t.sx[1], t.sy[1], kLrSubNorm, O + kLoSub + 3);
    rel(t.gx, t.gy, kLrGoalNorm, O + kLoGoal);
  }
  for (int k = 0; k < 4; ++k) {
    O[kLoAct + 2 * k] = V[mdp::obs::LAST_ACT + mdp::act::DIM * k];
    O[kLoAct + 2 * k + 1] = V[mdp::obs::LAST_ACT + mdp::act::DIM * k + 2];
  }
  O[kLoHasPath] = (float)t.has_path;
  O[kLoRem] = dm::minf(1.f, t.rem / kLrGoalNorm);
  O[kLoHasGoal] = (float)t.has_goal;
  if (b.cfg.lo_mov) {   // 가장 가까운 "움직이는 것" (lowrl_trk — 직전 걸음 lowrl_act 가 이어 간 추적. 없으면 0)
    LrMover M[kLrMovers];
    const int nm = lowrl_trk_movers(b, env, M);
    int bi = -1;
    float bd = 3e38f;
    for (int i = 0; i < nm; ++i) {
      const float dx = M[i].x - px, dy = M[i].y - py, d2 = dx * dx + dy * dy;
      if (d2 < bd) { bd = d2; bi = i; }
    }
    if (bi >= 0) {
      float s, co;
      dm::sincosf_d(pth, &s, &co);
      O[kLoMov] = 1.f;
      rel(M[bi].x, M[bi].y, kLrClusRange, O + kLoMov + 1);
      O[kLoMov + 4] = dm::clampf((co * M[bi].vx + s * M[bi].vy) / kLrMovVNorm, -1.f, 1.f);   // 로봇 앞쪽 성분
      O[kLoMov + 5] = dm::clampf((co * M[bi].vy - s * M[bi].vx) / kLrMovVNorm, -1.f, 1.f);   // 로봇 왼쪽 성분
    }
  }
}
