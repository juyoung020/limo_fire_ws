// 아래층 RL 의 걸음 함수 (기록 꼴·관측 배치는 low_task.h "아래층 RL", 설명은 mdp/README "아래층 RL"). env.h 가 namespace sim 안에서 포함한다 (low_step.h 뒤).
//
// 한 걸음(0.1 s)의 차례 — GPU(env_kernel.cu)·CPU 참조판(env_ref.cpp)이 같은 함수를 같은 차례로 부른다:
//   [행동]  lowrl_act        정책 환경(LowCmd::follow == 2)이면 행동 줄 (v, ω) → 바퀴 한도 → 안전층 → 명령 (추종기·복구 없음)
//   [물리·라이다·지도 — 기존 그대로]
//   [걸음 끝] lowrl_finish   보상 8 항(아래 상수)·누적. 판 끝 규칙은 단독 과제(low_step.h low_finish) 그대로
//   [판이 끝났으면 리셋·새 판 관측 — 기존 그대로]
//   [준비]  lowrl_goal_*     (low_auto) 새 목표 뽑기
//           깊이 카메라 층 · 경로 계획 (hier_path.h — lowrl_due 일 때만: low_period 걸음마다 · 목표가 바뀌면 바로)
//           lowrl_pack       정책 관측 (low_img · low_1d) — 방금 짠 경로 기준
// 준비 단계는 low_auto 면 env.step·reset 이 끝에서 부르고, 아니면 호스트가 LowCmd 를 넣은 뒤 Env::low_prepare 로 부른다 (위층이 1 Hz 로 목적지를 넣는 자리).
#pragma once

// ---------- 보상 (모두 잠정 — 짧은 학습으로 오르는 것만 확인한 값. 근거·출처는 mdp/README "아래층 RL") ----------
// 꼴은 Long 외(arXiv 1709.10082, ICRA 2018)의 "목표 쪽 진행 + 도달 + 충돌 + 회전 벌점" 에 근접·후진·안전층 항을 더한 것. 진행은 직선 거리 대신
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
constexpr float kLrSafetyK = -0.05f;     // 안전층(앞 틈·몸통 쓸림)이 명령을 고친 걸음
// ---------- 관측 ----------
constexpr float kLrLook0 = 1.0f, kLrLook1 = 2.5f;   // 부목표: 경로 위 가장 가까운 점에서 이만큼 앞 (m). 잠정 — 가까운 지도 창 반 변 3.2 m 안
constexpr float kLrNearCell = 0.10f;      // 가까운 지도 픽셀 (m) = mdp::obsgen::kNearCell (env_ref.cpp 가 같은지 확인)
constexpr float kLrSubNorm = 3.0f;       // 부목표 거리 정규화 (m)
constexpr float kLrGoalNorm = 15.0f;     // 목적지 거리·남은 경로 정규화 (m, 자름 1) — 과제 경로 길이 상한
// ---------- 자동 목표 (low_auto) ----------
constexpr int kLrProbes = 256;           // 한 번에 찔러 보는 칸 수 (아는 칸 상자 안 무작위)
// 직선 거리 등급 (m) — 앞 등급에 설 수 있는 칸이 하나라도 있으면 그 등급에서 뽑는다. 기준선 과제는 경로 3 ~ 15 m (scripts/low_run.cpp),
// 판 처음(첫 스캔만 안 때)·작은 맵은 그런 칸이 없어 아래 등급으로 내려간다
constexpr float kLrGoalD0 = 3.0f, kLrGoalD1 = 1.5f, kLrGoalD2 = 0.75f, kLrGoalDMax = 15.0f;

DEV bool lowrl_policy(const EnvBatch& b, int env) { return b.lowrl && b.lowcmd[env].follow == 2; }
DEV int lowrl_act_w(const EnvBatch& b) { return b.lowrl && b.cfg.low_act_w > 0 ? b.cfg.low_act_w : mdp::act::DIM; }
DEV int lowrl_period(const EnvCfg& c) { return c.low_period > 0 ? c.low_period : c.decide_steps; }

// 행동 앞머리 (env.h action_begin 몫 — 선언 없음)
DEV void lowrl_begin(const EnvBatch& b, EnvState& e, int env) {
  e.people.hit_person = 0;
  e.declared = 0;
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
  R.safe = 0;
  R.g_step = 0;
  R.g_len = -1.f;
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

// ---------- 행동 (걸음 첫머리, 스레드 0) ----------
// 정책 환경이면 명령을 정하고 true (호출한 쪽은 추종기를 건너뜀). a: 그 환경의 행동 줄 — a[0] = v, a[1] = ω (정규 단위 [−1, 1], 밖은 자름)
DEV bool lowrl_act(const EnvBatch& b, EnvState& e, int env, const float* a) {
  if (!b.lowrl) return false;
  LowRl& R = b.lowrl[env];
  R.pol = 0;
  R.safe = 0;
  if (b.lowcmd[env].follow != 2) return false;
  const EnvCfg& c = b.cfg;
  const float n0 = clip1(a[0]), n2 = clip1(a[1]);
  float v = n0 >= 0.f ? n0 * c.act_v_fwd : n0 * c.act_v_back, w = n2 * c.act_w_max;
  const float wheel = dm::absf(v) + c.wheel_k * dm::absf(w);   // 가장 빠른 바퀴가 한도를 넘으면 같은 비로 (속도 행동과 같은 규칙)
  if (wheel > c.wheel_max) {
    const float k = c.wheel_max / wheel;
    v = v * k;
    w = w * k;
  }
  bool safe = false;
  if (c.safety && v > 0.f && front_gap(b, e, env) < c.safety_dist) { v = 0.f; safe = true; }
  if (c.sweep_guard) {
    const float k2 = sweep_scale(b, e, env, v, 0.f, w);
    v *= k2;
    w *= k2;
    if (k2 < 1.f) safe = true;
  }
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
  low_begin(b, e, env, why, 0, false, safe, v);
  for (int i = mdp::act::DIM * 4 - 1; i >= mdp::act::DIM; --i) e.last_act[i] = e.last_act[i - mdp::act::DIM];
  e.last_act[0] = n0;   // 정책이 낸 값 (안전층 앞) — 매끄러움 항·관측이 같은 값
  e.last_act[1] = 0.f;
  e.last_act[2] = n2;
  e.last_act[3] = 0.f;
  R.pol = 1;
  R.safe = safe ? 1 : 0;
  return true;
}

// ---------- 걸음 끝 (finish_step 맨 끝, 스레드 0): 보상·누적 ----------
DEV void lowrl_finish(const EnvBatch& b, EnvState& e, int env) {
  if (!b.lowrl) return;
  LowRl& R = b.lowrl[env];
  if (R.pol) {
    const EnvCfg& c = b.cfg;
    const MapDev& m = b.maps[e.map];
    const LowEp& L = b.low[env];
    LowRlStat& S = b.lowstat[env];
    LowTrack t;
    int pi = e.p_i;
    lowrl_track(b, e, env, &pi, t);
    e.p_i = pi;
    const bool reached = L.reach_step == e.step && L.reached > 0;
    const bool off = !reached && L.hold != 0;                                   // 경로 끝에 왔는데 통과 반경 밖
    const bool nopath = !reached && !off && t.has_goal && !t.has_path;          // 경로를 못 짬
    float r[kLrTerms];
    for (int k = 0; k < kLrTerms; ++k) r[k] = 0.f;
    if (R.rem_ok && t.has_path) r[kLrProgress] = kLrProgressK * dm::clampf(R.rem_prev - t.rem, -kLrProgressClip, kLrProgressClip);
    r[kLrTime] = kLrTimeK;
    if (reached) r[kLrReach] = kLrReachK;
    if (e.collided) r[kLrCollision] = kLrCollisionK;
    {   // 근접: 충돌 판정과 같은 정답 격자(collision_band 에 걸린 칸 — 라이다 평면 밖 장애물도) · 사람
      float d = e.people.near_d;
      if (m.clear) d = dm::minf(d, body_clearance(c, m, e.x, e.y, e.th));
      if (d < kLrProxDist) r[kLrProximity] = kLrProxK * (1.f - dm::maxf(d, 0.f) / kLrProxDist);
    }
    if (e.step > 1) {
      const float d0 = e.last_act[0] - e.last_act[mdp::act::DIM], d2 = e.last_act[2] - e.last_act[mdp::act::DIM + 2];
      r[kLrSmooth] = kLrSmoothK * sqrtf(d0 * d0 + d2 * d2);
    }
    if (e.last_act[0] < 0.f) r[kLrBack] = kLrBackK * -e.last_act[0];
    if (R.safe) r[kLrSafety] = kLrSafetyK;
    float sum = 0.f;
    for (int k = 0; k < kLrTerms; ++k) {
      sum += r[k];
      S.term[k] += r[k];
    }
    e.reward = sum;
    S.steps += 1;
    S.safe_steps += R.safe;
    S.back_steps += e.last_act[0] < 0.f ? 1 : 0;
    {
      const float dx = e.x - L.x0, dy = e.y - L.y0;
      S.dist += sqrtf(dx * dx + dy * dy);
    }
    if (reached) {
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
      if (e.collided) {
        if (e.people.hit_person) S.hit_person += 1;
        else S.hit_wall += 1;
      } else if (L.end == kLowEndStall) {
        S.stall += 1;
      } else if (L.end != kLowEndGoals) {
        S.timeout += 1;
      }
    }
  }
  if (e.done && b.low[env].end == kLowEndStall) {   // 위층에 알리는 막힘 신호 (정책·규칙 추종 모두)
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
}
