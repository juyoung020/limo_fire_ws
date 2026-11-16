// 아래층 걸음 길(EnvCfg::low_rl — env.step 한 번 = 0.1 s 걸음 하나)의 걸음 사이 값·준비 단계·아래층 관측. env.h 가 namespace sim 안에서 포함한다 (low_step.h 뒤).
// 아래층은 규칙 추종기(A* 경로 + 추종 — hier_path.h)만: 위층이 LowCmd(목표 칸·follow)를 넣고, 여기서는 계획 차례·막힘 신호·아래층 관측만 다룬다.
// (이름의 "rl" 은 그만둔 아래층 강화학습 때 붙은 것 — 정책·보상·학습 버퍼는 2026-11-16 에 archive/rl 로 옮기고 여기서 걷어냄)
//
// 한 걸음의 차례 — GPU(env_kernel.cu)·CPU 참조판(env_ref.cpp)·실물 RobotEnv 가 같은 함수를 같은 차례로 부른다:
//   [행동]  lowrl_begin      막힘 신호 지움(새 결정 구간) · 끝 선언(finish_declare)
//   [물리·라이다·지도 — env.h]
//   [걸음 끝] lowrl_finish   막힘 신호 올림 (정체 — 위층이 그 칸을 피함) · 판이 끝났으면 걸음 사이 값 비움
//   [준비]  lowrl_due 일 때 깊이 카메라 층 · 경로 계획 (hier_path.h — low_period 걸음마다 · 목표가 바뀌면 바로) → lowrl_planned
//           lowrl_pack · lowrl_img  아래층 관측 (그림 · 1D — 실물 노드 녹화 꼴(low_node_rec.h)에 남고 sim == 실물 비교에 씀)
// 준비 단계는 호스트가 LowCmd 를 넣은 뒤 Env::low_prepare 로 부른다 (위층이 목적지를 넣는 자리).
#pragma once

// ---------- 관측 ----------
constexpr float kLrLook0 = 1.0f, kLrLook1 = 2.5f;   // 부목표: 경로 위 가장 가까운 점에서 이만큼 앞 (m). 잠정 — 가까운 지도 창 반 변 3.2 m 안
constexpr float kLrNearCell = 0.10f;      // 가까운 지도 픽셀 (m) = mdp::obsgen::kNearCell (env_ref.cpp 가 같은지 확인)
constexpr float kLrSubNorm = 3.0f;       // 부목표 거리 정규화 (m)
constexpr float kLrGoalNorm = 15.0f;     // 목적지 거리·남은 경로 정규화 (m, 자름 1) — 과제 경로 길이 상한

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
DEV void lowrl_planned(const EnvBatch& b, const EnvState& e, int env) {
  LowRl& R = b.lowrl[env];
  R.p_step = e.step;
  R.tgt_seen = b.lowcmd[env].target;
  R.fol_seen = b.lowcmd[env].follow;
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
DEV int low_img_n(const EnvCfg&) { return kLowImg; }   // 환경 하나 그림 바이트 수
DEV int low_1d_n(const EnvCfg&) { return kLow1d; }     // 환경 하나 1D 칸 수

// 숫자 관측 (스레드 0). 계획 뒤에 부른다 — 부목표·남은 경로 길이를 방금 짠 경로 기준으로
DEV void lowrl_pack(const EnvBatch& b, const EnvState& e, int env) {
  LowTrack t;
  int pi = e.p_i;   // 추종기의 p_i 는 그대로 (가장 가까운 점 찾기는 사본으로)
  lowrl_track(b, e, env, &pi, t);
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
}
