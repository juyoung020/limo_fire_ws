// 움직이는 사람 동작 (규칙은 people.h 위 설명). env.h 가 EnvState·MapDev 정의 뒤에 넣는다 — GPU 커널·CPU 참조판이 같은 함수를 스레드 0 에서 부른다.
#pragma once
// namespace sim 안에서 넣는다 (env.h)

// 이 판 사람을 광선에 넣은 장면 보기 (라이다·깊이)
DEV SceneView with_people(const SceneView& s, const People& P) {
  SceneView v = s;
  v.cyl = P.cyl;
  v.n_cyl = P.n;
  return v;
}

// 원기둥 (x, y, r) 이 설 수 있나: 맵 안 · 갈 수 있는 바닥 칸 · 가장 가까운 막힌 칸까지(칸 가운데끼리) > r + 칸 대각.
// 칸 안 어느 점이든 칸 가운데에서 반 대각 안이고 막힌 칸도 그 가운데에서 반 대각 안이므로, 이걸 넘으면 원기둥이 막힌 칸에 안 닿는다
DEV bool person_free(const EnvCfg& c, const MapDev& m, float x, float y, float r) {
  const int cx = dm::floor_div(x - m.ox, c.res, c.inv_res), cy = dm::floor_div(y - m.oy, c.res, c.inv_res);
  if (cx < 0 || cy < 0 || cx >= m.gw || cy >= m.gh) return false;
  const int k = cy * m.gw + cx;
  if (!(m.grid[k] & kCellFloor)) return false;
  return (float)m.clear[k] * 0.01f > r + 1.41421356f * c.res;
}

// 사람 i 가 (x, y) 로 가면 다른 사람과 겹치면서 더 가까워지나
DEV bool person_crowded(const People& P, int i, float x, float y) {
  const Cyl& a = P.cyl[i];
  for (int j = 0; j < P.n; ++j) {
    if (j == i) continue;
    const Cyl& o = P.cyl[j];
    const float dx = x - o.x, dy = y - o.y, ox = a.x - o.x, oy = a.y - o.y, rr = a.r + o.r;
    const float d2 = dx * dx + dy * dy;
    if (d2 < rr * rr && d2 < ox * ox + oy * oy) return true;
  }
  return false;
}

// 새 목표점: 시작 칸 후보(갈 수 있는 바닥, 넉넉한 칸) 중 이 사람이 설 수 있는 곳. 4 번 뽑아 안 되면 그대로
DEV void person_pick_goal(const EnvCfg& c, const MapDev& m, People& P, int i) {
  for (int t = 0; t < 4; ++t) {
    const int k = m.spawn[dm::splitmix64(P.rng) % (uint64_t)m.n_spawn];
    const float x = m.ox + ((float)(k % m.gw) + 0.5f) * c.res, y = m.oy + ((float)(k / m.gw) + 0.5f) * c.res;
    if (person_free(c, m, x, y, P.cyl[i].r)) { P.gx[i] = x; P.gy[i] = y; return; }
  }
}

// 다가옴·가로지름 목표 (people.h 머리말): 로봇 참 자세 (x, y, th) · 몸통 속도 v. 목표 자리에 못 서면(벽) 로봇 쪽 점 그대로 (막히면 people_step 이 새 목표)
constexpr int kPersonRetarget = 50;   // 다가옴·가로지름 목표를 다시 잡는 물리 걸음 (50 × 0.02 s = 1 s)
DEV void person_robot_goal(const EnvCfg& c, const MapDev& m, People& P, int i, float x, float y, float th, float v) {
  if (P.mode[i] & kPersonApproach) { P.gx[i] = x; P.gy[i] = y; return; }
  float s, co;
  dm::sincosf_d(th, &s, &co);
  const float ahead = dm::clampf(dm::absf(v) * 2.f, 1.f, 3.f);
  const float px = x + co * ahead, py = y + s * ahead;   // 로봇 앞 예측 길의 점
  const float dx = px - P.cyl[i].x, dy = py - P.cyl[i].y, d = sqrtf(dx * dx + dy * dy);
  float gx = px, gy = py;
  if (d > 1e-3f) { gx = px + dx / d; gy = py + dy / d; }   // 사람에서 그 점을 지나 1 m 더 — 로봇 앞을 가로지름
  if (!person_free(c, m, gx, gy, P.cyl[i].r)) { gx = px; gy = py; }
  P.gx[i] = gx;
  P.gy[i] = gy;
}

// 로봇 몸통 사각형(가운데 (x, y), 방향 th, half_len × half_wid) ~ 원기둥 간격 (겹치면 음수)
DEV float body_cyl_gap(const EnvCfg& c, const Cyl& C, float x, float y, float s, float co) {
  const float dx = C.x - x, dy = C.y - y;
  const float u = dm::absf(co * dx + s * dy), v = dm::absf(co * dy - s * dx);
  const float qx = dm::maxf(u - c.half_len, 0.f), qy = dm::maxf(v - c.half_wid, 0.f);
  if (qx == 0.f && qy == 0.f) return -(C.r + dm::minf(c.half_len - u, c.half_wid - v));   // 가운데가 몸통 안
  return sqrtf(qx * qx + qy * qy) - C.r;
}

// 가장 가까운 사람까지 몸통 간격 (없으면 kNoPerson)
DEV float people_gap(const EnvCfg& c, const People& P, float x, float y, float th) {
  if (P.n == 0) return kNoPerson;
  float s, co;
  dm::sincosf_d(th, &s, &co);
  float g = kNoPerson;
  for (int i = 0; i < P.n; ++i) g = dm::minf(g, body_cyl_gap(c, P.cyl[i], x, y, s, co));
  return g;
}

// 판 시작 (reset_state 끝, 로봇 자세·world_seed 를 정한 뒤). hit_person·k_max 는 안 건드림
DEV void people_reset(const EnvBatch& b, EnvState& e) {
  const EnvCfg& c = b.cfg;
  const PeopleCfg& pc = c.people;
  const MapDev& m = b.maps[e.map];
  People& P = e.people;
  P.rng = e.world_seed ^ 0x5851F42D4C957F2Dull;
  P.n = 0;
  P.near_d = kNoPerson;
  const int k = P.k_max < kMaxPeople ? P.k_max : kMaxPeople;
  if (k <= 0) return;
  const int want = (int)(dm::splitmix64(P.rng) % (uint64_t)(k + 1));
  const float keep2 = pc.spawn_keep * pc.spawn_keep;
  for (int i = 0; i < want; ++i) {
    const float r = dm::rand_range(P.rng, pc.r_lo, pc.r_hi), h = dm::rand_range(P.rng, pc.h_lo, pc.h_hi);
    bool ok = false;
    float x = 0.f, y = 0.f;
    for (int t = 0; t < 32 && !ok; ++t) {
      const int cell = m.spawn[dm::splitmix64(P.rng) % (uint64_t)m.n_spawn];
      x = m.ox + ((float)(cell % m.gw) + 0.5f) * c.res;
      y = m.oy + ((float)(cell / m.gw) + 0.5f) * c.res;
      const float dx = x - e.x, dy = y - e.y;
      if (dx * dx + dy * dy < keep2 || !person_free(c, m, x, y, r)) continue;
      ok = true;
      for (int j = 0; j < P.n && ok; ++j) {
        const float ex = x - P.cyl[j].x, ey = y - P.cyl[j].y, rr = (r + P.cyl[j].r) + 0.1f;
        ok = ex * ex + ey * ey >= rr * rr;
      }
    }
    if (!ok) break;   // 자리가 안 나면 그만큼 적게
    P.cyl[i] = Cyl{x, y, r, h};
    const float u_walk = dm::rand01(P.rng), u_ign = dm::rand01(P.rng);
    P.mode[i] = (u_walk >= pc.p_stand ? kPersonWalk : 0) | (u_ign < pc.p_ignore ? kPersonIgnore : 0);
    if (pc.p_approach > 0.f || pc.p_cross > 0.f) {   // 둘 다 0 이면 난수를 안 뽑아 옛 판과 비트 같음
      const float u = dm::rand01(P.rng);
      if (P.mode[i] & kPersonWalk) P.mode[i] |= u < pc.p_approach ? kPersonApproach : (u < pc.p_approach + pc.p_cross ? kPersonCross : 0);
    }
    P.speed[i] = dm::rand_range(P.rng, pc.v_lo, pc.v_hi);
    P.wait[i] = 0.f;
    P.gx[i] = x;
    P.gy[i] = y;
    P.n = i + 1;
    person_pick_goal(c, m, P, i);
    if (P.mode[i] & (kPersonApproach | kPersonCross)) person_robot_goal(c, m, P, i, e.x, e.y, e.th, e.v);
  }
  P.near_d = people_gap(c, P, e.x, e.y, e.th);
}

// 물리 한 걸음 (physics_step 처음, 로봇보다 먼저). 사람 번호 차례로 — 앞 사람이 옮긴 자리를 뒷사람이 본다
DEV void people_step(const EnvBatch& b, EnvState& e) {
  const EnvCfg& c = b.cfg;
  const PeopleCfg& pc = c.people;
  const MapDev& m = b.maps[e.map];
  People& P = e.people;
  const bool retarget = e.tick % kPersonRetarget == 0;
  for (int i = 0; i < P.n; ++i) {
    if (!(P.mode[i] & kPersonWalk)) continue;
    if (retarget && (P.mode[i] & (kPersonApproach | kPersonCross))) {   // 1 s 마다 로봇 쪽 목표 (쉬던 중이면 쉼을 끝냄)
      person_robot_goal(c, m, P, i, e.x, e.y, e.th, e.v);
      P.wait[i] = 0.f;
    }
    Cyl& C = P.cyl[i];
    float sx, sy;   // 이번 걸음 이동
    bool to_goal = false;
    const float rx = C.x - e.x, ry = C.y - e.y, d2r = rx * rx + ry * ry;
    if (!(P.mode[i] & kPersonIgnore) && d2r < pc.yield_dist * pc.yield_dist) {
      // 비켜섬: 로봇에서 멀어지는 쪽으로
      const float d = sqrtf(d2r);
      if (!(d > 1e-4f)) continue;
      const float k = pc.yield_speed * c.dt / d;
      sx = rx * k;
      sy = ry * k;
    } else if (P.wait[i] > 0.f) {
      P.wait[i] = P.wait[i] - c.dt;
      continue;
    } else {
      const float gx = P.gx[i] - C.x, gy = P.gy[i] - C.y;
      const float dg = sqrtf(gx * gx + gy * gy);
      if (dg < 0.05f) {   // 닿음: 잠깐 서고 새 목표
        P.wait[i] = pc.pause * dm::rand01(P.rng);
        person_pick_goal(c, m, P, i);
        continue;
      }
      const float k = dm::minf(P.speed[i] * c.dt, dg) / dg;
      sx = gx * k;
      sy = gy * k;
      to_goal = true;
    }
    const float nx = C.x + sx, ny = C.y + sy;
    if (person_free(c, m, nx, ny, C.r) && !person_crowded(P, i, nx, ny)) {
      C.x = nx;
      C.y = ny;
    } else if (to_goal) {   // 막힘(벽·사람): 잠깐 서고 새 목표
      P.wait[i] = 0.5f * pc.pause * dm::rand01(P.rng);
      person_pick_goal(c, m, P, i);
    }
  }
}

