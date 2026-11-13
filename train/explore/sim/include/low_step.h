// 아래층 계측·단독 과제의 걸음 함수 (기록 꼴·설명은 low_task.h). env.h 가운데서 include (hier_path.h·graph_env.h 뒤). 모두 스레드 0. env.h 가 namespace sim 안에서 포함한다.
#pragma once

// 이 환경이 합친 탐사 판인가 (EnvCfg::low_explore). 탐사 판 학습(low_auto 2)은 환경의 low_solo % 를 단독 과제로 돌린다 — 환경 번호 해시로 가름
// (맵 배정이 환경 번호 차례라 번호 그대로 나누면 맵과 얽힘). 그 밖의 모드는 설정값 그대로
DEV bool low_solo_env(const EnvCfg& c, int env) { return c.low_auto == 2 && (int)((((uint32_t)env * 2654435761u) >> 8) % 100u) < c.low_solo; }
DEV bool low_xenv(const EnvBatch& b, int env) { return b.cfg.low_explore != 0 && !low_solo_env(b.cfg, env); }

DEV int low_ang_bin(float err) {
  const float d = dm::absf(err) * 57.29578f;
  return d < 30.f ? 0 : (d < 60.f ? 1 : (d < 90.f ? 2 : (d < 135.f ? 3 : 4)));
}
// 방향 변화가 정해짐 → 바뀜 하나를 표에
DEV void low_resolve(LowEp& L, int abin) {
  L.sw[L.pend_arr][L.pend_d][abin] += 1;
  if (!L.pend_arr) {
    L.ab_n[abin] += L.pend_n;
    L.ab_dist[abin] += L.pend_dist;
  }
  L.pend = 0;
}

// 걸음 처음 (path_step_c 끝 — 명령이 정해진 뒤, 물리 앞). why: 추종기 가지, bkind: path_step_b 가지 (0 없음 · 1 복구 물러나기 · 2 충돌 검사 멈춤 ·
// 3 제자리 회전 막힘), mon: collision_monitor 가 줄임, safe: 안전층·sweep_guard 가 줄임, v: 내보낸 명령
DEV void low_begin(const EnvBatch& b, const EnvState& e, int env, const PathWhy& why, int bkind, bool mon, bool safe, float v) {
  if (!b.low) return;
  const EnvCfg& c = b.cfg;
  const MapDev& m = b.maps[e.map];
  LowEp& L = b.low[env];
  const bool fol = b.lowcmd && b.lowcmd[env].follow != 0;
  if (e.step == 0 || (fol && !L.following)) {   // 판 첫 걸음 · 단독 과제 구간 첫 걸음: 누적을 0 으로
    LowEp z = {};
    L = z;
    L.goal_prev = -1;
    L.plans_prev = e.p_plans;
    L.cause = kLowKinds;
    L.following = fol ? 1 : 0;
    L.f_start = e.step;
    L.a_step = e.step;
    L.ax = e.x;
    L.ay = e.y;
    L.d_step = e.step;   // 0 으로 만드는 걸음은 늘 결정 첫 걸음
  }
  // 목적지 바뀜 · 다시 계획
  if (e.goal_cell != L.goal_prev) {
    if (e.goal_cell >= 0) L.n_goal += 1;
    if (e.goal_cell >= 0 && L.goal_prev >= 0) {
      if (L.pend) low_resolve(L, kLowAng - 1);   // 앞 바뀜의 방향 변화를 재기 전에 또 바뀜
      const int dx = e.goal_cell % m.gw - L.goal_prev % m.gw, dy = e.goal_cell / m.gw - L.goal_prev / m.gw;
      const float d = sqrtf((float)(dx * dx + dy * dy)) * c.res;
      L.pend = 1;
      L.pend_arr = L.arrived;
      L.pend_d = d < 0.5f ? 0 : (d < 2.f ? 1 : 2);
      L.pend_n = L.seg_n;
      L.pend_dist = L.seg_dist;
      if (!L.arrived) L.left_sum += L.left;
    }
    L.ctx = 1;
    L.arrived = 0;
    L.seg_n = 0;
    L.seg_dist = 0.f;
    L.left = 0.f;
    L.goal_prev = e.goal_cell;
    L.plans_prev = e.p_plans;
    if (!low_xenv(b, env)) {   // 정체 닻: 새 목적지부터 다시 (합친 탐사는 목적지가 1 s 마다 바뀌어도 닻을 그대로 — 자리로만 잼)
      L.a_step = e.step;
      L.ax = e.x;
      L.ay = e.y;
    }
  } else if (e.p_plans != L.plans_prev) {
    if (L.ctx == 0) L.ctx = 2;
    L.plans_prev = e.p_plans;
  }
  if (L.pend && (why.kind == kWhyRot || why.kind == kWhyGo)) low_resolve(L, low_ang_bin(why.err));
  // 분류
  int kind;
  if (bkind == 1) kind = kLowRec;
  else if (bkind == 2) kind = kLowImm;
  else if (bkind == 3) kind = kLowRotBlocked;
  else if (why.kind == kWhyNoGoal) kind = kLowRotNoGoal;
  else if (why.kind == kWhyNoPath) kind = kLowNoPath;
  else if (why.kind == kWhyArrive) kind = kLowRotArrive;
  else if (why.kind == kWhyArriveStop) kind = kLowArriveStop;
  else if (why.kind == kWhyRot) kind = L.ctx == 1 ? kLowRotSwitch : (L.ctx == 2 ? kLowRotPlan : kLowRotMid);
  else if (safe) kind = kLowSafety;
  else if (mon) kind = kLowMon;
  else if (why.vc >= 0.9f * c.path_v) kind = v >= 0.9f * c.path_v ? kLowFull : kLowAccel;
  else kind = why.bind == kLowFull ? kLowCurv : why.bind;   // 목표가 0.9 아래면 묶은 항이 있다 (없을 수 없지만 칸은 곡률로)
  if (why.kind == kWhyGo) L.ctx = 0;
  if (why.kind == kWhyArrive || why.kind == kWhyArriveStop) { L.arrived = 1; L.left = 0.f; }
  if (why.kind == kWhyRot || why.kind == kWhyGo) L.left = why.d_goal;
  if (kind == kLowAccel) L.acc_n[L.cause] += 1;
  else if (kind != kLowFull) L.cause = kind;
  L.kind = kind;
  L.n[kind] += 1;
  L.cmd[kind] += dm::absf(v) * ((float)c.substeps * c.dt);
  L.steps += 1;
  L.seg_n += 1;
  L.x0 = e.x;
  L.y0 = e.y;
}

// 걸음 끝 (finish_step 의 e.step += 1 뒤, 판 끝 판정 앞). 단독 과제 판을 끝내야 하면 true (충돌은 기존 규칙이 끝냄)
DEV bool low_finish(const EnvBatch& b, const EnvState& e, int env) {
  if (!b.low) return false;
  const EnvCfg& c = b.cfg;
  const MapDev& m = b.maps[e.map];
  LowEp& L = b.low[env];
  {
    const float dx = e.x - L.x0, dy = e.y - L.y0, d = sqrtf(dx * dx + dy * dy);
    L.dist[L.kind] += d;
    if (L.kind == kLowAccel) L.acc_dist[L.cause] += d;
    L.dist_all += d;
    L.seg_dist += d;
  }
  if (e.collided) {
    L.end = e.people.hit_person ? kLowEndPerson : kLowEndWall;
    if (e.people.hit_person) {   // 가장 깊이 겹친 사람의 방위 (로봇 기준) — 앞 ±low_front 안인가
      float s, co;
      dm::sincosf_d(e.th, &s, &co);
      float g = 1e9f, bear = 0.f;
      for (int i = 0; i < e.people.n; ++i) {
        const float gi = body_cyl_gap(c, e.people.cyl[i], e.x, e.y, s, co);
        if (gi < g) {
          g = gi;
          bear = dm::wrap_pi(dm::atan2f_d(e.people.cyl[i].y - e.y, e.people.cyl[i].x - e.x) - e.th);
        }
      }
      L.hit_dir = dm::absf(bear) <= c.low_front ? 1 : 2;
    }
    return false;
  }
  if (low_xenv(b, env)) {   // 합친 탐사 (low_task.h): 판은 탐사 규칙이 끝낸다. 통과·경로 끝은 표시만(위층이 바로 다시 고름), 정체는 신호만 올리고 닻을 새로
    const int tgt = b.lowcmd && b.lowcmd[env].follow != 0 ? b.lowcmd[env].target : -1;
    const float px = e.x + e.ex, py = e.y + e.ey;
    if (tgt >= 0 && e.goal_cell == tgt) {
      const float tx = m.ox + ((float)(tgt % m.gw) + 0.5f) * c.res - px, ty = m.oy + ((float)(tgt / m.gw) + 0.5f) * c.res - py;
      if (tx * tx + ty * ty <= c.low_pass_r * c.low_pass_r) {
        L.reached += 1;
        L.reach_step = e.step;
        L.hold = e.step - L.d_step;
      } else if (e.p_n > 0 && !e.p_trunc) {
        const float ex = m.ox + ((float)e.wp[2 * (e.p_n - 1)] + 0.5f) * c.res - px, ey = m.oy + ((float)e.wp[2 * (e.p_n - 1) + 1] + 0.5f) * c.res - py;
        if (ex * ex + ey * ey <= c.goal_tol * c.goal_tol) {
          L.off += 1;
          L.hold = e.step - L.d_step;
        }
      }
    }
    L.end = 0;
    const float ax = e.x - L.ax, ay = e.y - L.ay;
    if (ax * ax + ay * ay >= c.low_idle_d * c.low_idle_d || e.step - L.a_step >= c.low_idle_steps) {
      if (ax * ax + ay * ay < c.low_idle_d * c.low_idle_d) L.end = kLowEndStall;   // 이 걸음에 정체 (lowrl_finish 가 신호로 옮김)
      L.a_step = e.step;
      L.ax = e.x;
      L.ay = e.y;
    }
    return false;
  }
  if (!L.following || !b.lowcmd) return false;
  const int tgt = b.lowcmd[env].target;
  const float px = e.x + e.ex, py = e.y + e.ey;   // 지도 자세 (로봇이 아는 자기 자리)
  if (tgt >= 0 && e.goal_cell == tgt) {
    const float tx = m.ox + ((float)(tgt % m.gw) + 0.5f) * c.res - px, ty = m.oy + ((float)(tgt / m.gw) + 0.5f) * c.res - py;
    if (tx * tx + ty * ty <= c.low_pass_r * c.low_pass_r) {
      L.reached += 1;
      L.reach_step = e.step;
      L.hold = e.step - L.d_step;
      if (L.reached >= c.low_goals) { L.end = kLowEndGoals; return true; }
    } else if (e.p_n > 0 && !e.p_trunc) {
      const float ex = m.ox + ((float)e.wp[2 * (e.p_n - 1)] + 0.5f) * c.res - px, ey = m.oy + ((float)e.wp[2 * (e.p_n - 1) + 1] + 0.5f) * c.res - py;
      if (ex * ex + ey * ey <= c.goal_tol * c.goal_tol) {
        L.off += 1;
        L.hold = e.step - L.d_step;
      }
    }
  }
  if (!L.hold) {
    const float ax = e.x - L.ax, ay = e.y - L.ay;
    if (ax * ax + ay * ay >= c.low_idle_d * c.low_idle_d) {
      L.a_step = e.step;
      L.ax = e.x;
      L.ay = e.y;
    } else if (e.step - L.a_step >= c.low_idle_steps) {
      L.end = kLowEndStall;
      return true;
    }
  }
  if (e.step - L.f_start >= c.low_time_steps) { L.end = kLowEndTime; return true; }
  return false;
}

