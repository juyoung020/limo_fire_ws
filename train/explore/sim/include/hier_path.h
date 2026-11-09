// 계층형 행동 모드 (EnvCfg::action_mode == kActPath, mdp/README "계층형"): 정책이 1 s 마다 프론티어 후보(mdp::CandSummary) 하나를 고르면
// 로봇 자기 지도(lmap)로 경로를 짜고(결정 때만) 0.1 s 마다 추종기가 (v, ω) 를 낸다. 정답 격자(MapDev::grid·blocked·body_hits)는 쓰지 않는다 —
// 쓰는 것은 로봇 지도 칸 종류(mdp::cell_class)·지도 자세(참값 + 오차)·직전 라이다 장(안전층)뿐이라 실물에서도 같은 값을 셀 수 있다
// (sim/tools/env_verify --hier 가 정답 격자를 바꿔도 경로·명령이 같은지 확인).
//
// 경로 계획 = 설 수 있는 칸(로봇 반경 0.2 m 팽창, mdp::standable — 관측 프론티어 요약과 같은 칸) 위 무게 있는 최단 경로(칸 4 이웃, 들어가는 칸의 무게).
//   무게는 Nav2 Smac 2D 의 칸 비용(nav2_smac_planner node_2d.cpp getTraversalCost: 1 + cost_travel_multiplier · cost / 252)에
//   팽창 비용(nav2_costmap_2d inflation_layer.hpp computeCost: 252 · exp(−k (d − r_inscribed)), r_inscribed 0.2 · k 3 · 팽창 반경 0.45 m,
//   cost_travel_multiplier 2 — nav2_params.yaml)을 넣은 값을 막힘까지 거리 d 세 구간 정수로 근사:
//     d ≤ 0.3 m → 3 (Nav2 2.48~3.0) · 0.3 < d ≤ 0.45 m → 2 (1.94~2.48) · d > 0.45 m → 1 (팽창 밖, 비용 0)
//   0.3 m 경계는 잠정(구간을 나눈 자리). 벽에 붙은 최단 경로 대신 가운데로 가게 해 몸통 쓸림을 줄인다(기준선 A* 와 같은 뜻).
//   GPU 는 칸 비트 판 위 층 BFS(무게 ≤ 3 → 다음 층 버킷 4 개, Dial 알고리즘)로, CPU 참조판은 칸마다 거리 배열로 — 거리 장이 같으므로
//   뒤로 짚기(이웃 차례 mdp::nb4 에서 처음으로 거리 = 내 거리 − 내 무게 인 칸)도 같다. GPU 는 거리를 mod 7 로만 남긴다
//   (이웃 거리는 내 거리 ± 3 안이라 mod 7 이면 앞 칸이 하나로 정해짐).
//   웨이포인트: 칸 경로를 앞에서부터 직선(4 이웃 선, 지나는 칸이 모두 설 수 있고 무게가 건너뛴 경로 칸의 최대 무게 이하)으로 이을 수 있는 데까지
//   늘려 꺾이는 칸만 남김(길이 상한 kSegMax 칸), 최대 mdp::hier::kPathMax 개 — 넘으면 앞 부분만(p_trunc, 결정마다 다시 짬).
//   같은 후보를 다시 고르면(그 덩어리가 지금 목표 칸을 품음 — CandTop::cur) 남은 경로가 아직 설 수 있는 칸 위면 그대로 둔다.
//   약한 막힘 (2026-11-08 사용자 결정): 지도 막힘(p > 0.65, 4 번 맞아야 됨)이 되기 전의 맞음 > 지나감 칸(로그 오즈 > 0)도 로봇에서 장애물 층
//   obstacle_max_range(3 m) 안이면 막힘으로 보고 먼저 짠다(엄격) — Nav2 global costmap 의 obstacle_layer 가 스캔 끝점을 그 거리 안에서 바로 LETHAL 로
//   두고 팽창하는 것과 같은 뜻(기준선 lethal_src·추종기 충돌 검사 path_lethal 과 같은 규칙). 목표가 그 팽창 안이거나 엄격으로 못 닿으면 지도 막힘만으로(느슨)
//   다시 짠다 — Nav2 는 거기서 계획 실패지만, 후보는 지도 막힘 기준(닫힘 정의)으로 고른 칸이라 갈 길을 남긴다. 느슨 경로는 충돌 검사가 약한 막힘을 본다.
//   측정 (규칙 정책 slot0, 시험 맵 64 판씩): 계획의 45~63 % 가 느슨으로 내려가고(목표 프론티어 칸이 막 보인 벽 옆), 충돌은 서점 0.156 → 0.188 ·
//   작은 집 0.141 → 0.156 으로 줄지 않음(64 판 표준오차 ≈ 0.05). 약한 막힘을 지나기는 허용하고 무게만 주는 판도 같은 충돌(0.188 · 0.156)
//
// 추종기 = Nav2 Regulated Pure Pursuit 식 (baselines/frontier frontier_explorer.cpp follow 와 같은 식·같은 nav2_params 값):
//   경로 위 가장 가까운 점에서 lookahead 만큼 앞 점(당근) → 방향 오차 > rotate_to_heading_min_angle 이면 제자리 회전, 아니면
//   곡률 감속(regulated_linear_scaling_min_radius)·장애물 근접 감속(use_cost_regulated_linear_velocity_scaling — 로봇 칸에서 막힘까지 거리,
//   Nav2 기본 cost_scaling_dist 0.6 · gain 1.0, 팽창 반경 안에서만)·다가가기 감속 → velocity_smoother 가속 한도(행동 주기 0.1 s) → 안전층.
//   직진 속도는 기준선 fast 판과 같은 1 m/s (env.yaml action.path). 후보가 없으면 제자리 회전(rotate_to_heading 속도)으로 지도를 넓힌다.
#pragma once
// namespace sim 안에서 넣는다 (env.h — people_sim.h 와 같게)

constexpr int kPathPlanes = 11;   // GPU 계획 비트 판 수 (EnvBatch::pplanes): 무게 경계 2 · 버킷 4 · 거리 mod 7 3 비트 · 엄격 막힘 · 엄격 설 수 있음
enum : int { kPpC3 = 0, kPpC45 = 1, kPpB = 2, kPpD = 6, kPpLT = 9, kPpSST = 10 };
constexpr int kSegMax = 40;      // 웨이포인트 사이 경로 칸 수 상한 (2 m) — 잠정

DEV int path_weight(bool st, bool c_mid, bool c_out) { return !st ? 0 : (c_out ? 1 : (c_mid ? 2 : 3)); }
// 무게 경계 반경 (설 수 있는 칸 반경 mdp::obsgen::kRobotRadius 0.2 m 바깥 두 구간) → 로봇 반경 꼴
DEV mdp::Robot path_rb_mid(const EnvCfg& c) { return mdp::robot_cells(c.path_r_mid, c.res); }
DEV mdp::Robot path_rb_out(const EnvCfg& c) { return mdp::robot_cells(c.path_r_out, c.res); }

// (x0, y0) → (x1, y1) 4 이웃 선(한 걸음에 x 나 y 하나만)의 모든 칸이 0 < 무게 ≤ worst 인가. W(x, y) → 무게 (0 = 못 섬)
template <class W>
DEV bool path_los(int x0, int y0, int x1, int y1, int worst, const W& wfn) {
  // 격자 걷기(칸 가운데 선이 다음에 x 경계를 먼저 넘나 y 경계를 먼저 넘나 — 정수 비교, 같으면 y 먼저)
  const int nx = x1 > x0 ? x1 - x0 : x0 - x1, ny = y1 > y0 ? y1 - y0 : y0 - y1;
  const int sx = x0 < x1 ? 1 : -1, sy = y0 < y1 ? 1 : -1;
  int x = x0, y = y0;
  {
    const int w = wfn(x, y);
    if (w <= 0 || w > worst) return false;
  }
  for (int ix = 0, iy = 0; ix < nx || iy < ny;) {
    if ((1 + 2 * ix) * ny < (1 + 2 * iy) * nx) { x += sx; ++ix; }
    else { y += sy; ++iy; }
    const int w = wfn(x, y);
    if (w <= 0 || w > worst) return false;
  }
  return true;
}

// 칸 경로 cells[0..n) (y · gw + x, 시작 → 목표) → 웨이포인트 wp[2i] = x, wp[2i+1] = y (시작 칸 포함, 목표 칸으로 끝). 반환 = 웨이포인트 수.
// *trunc = 1 이면 max_wp 를 넘어 앞 부분만 담음(마지막 점 ≠ 목표)
template <class W>
DEV int path_simplify(const int32_t* cells, int n, int gw, const W& wfn, int16_t* wp, int max_wp, int* trunc) {
  *trunc = 0;
  if (n <= 0) return 0;
  int nw = 0;
  const auto put = [&](int cell) {
    wp[2 * nw] = (int16_t)(cell % gw);
    wp[2 * nw + 1] = (int16_t)(cell / gw);
    ++nw;
  };
  put(cells[0]);
  int a = 0, worst = wfn(cells[0] % gw, cells[0] / gw);
  int i = 1;
  while (i < n) {
    const int wi = wfn(cells[i] % gw, cells[i] / gw);
    const int w2 = wi > worst ? wi : worst;
    if (i - a <= kSegMax && path_los(cells[a] % gw, cells[a] / gw, cells[i] % gw, cells[i] / gw, w2, wfn)) {
      worst = w2;
      ++i;
      continue;
    }
    // i − 1 까지가 이어짐 (i − 1 == a 면 이웃 칸 하나도 안 이어지는 경우 — 경로 칸끼리는 4 이웃이라 늘 이어지므로 안 옴)
    if (nw >= max_wp - 1) { *trunc = 1; put(cells[i - 1]); return nw; }
    put(cells[i - 1]);
    a = i - 1;
    const int wa = wfn(cells[a] % gw, cells[a] / gw);
    worst = wa;
  }
  put(cells[n - 1]);
  return nw;
}

// 남은 경로(p_i 구간부터)가 지금 지도에서 모두 설 수 있는 칸 위인가. S(x, y) → 설 수 있음
template <class S>
DEV bool path_clear(const EnvState& e, const S& stand) {
  for (int i = e.p_i < 0 ? 0 : e.p_i; i + 1 < e.p_n; ++i) {
    const int x0 = e.wp[2 * i], y0 = e.wp[2 * i + 1], x1 = e.wp[2 * i + 2], y1 = e.wp[2 * i + 3];
    if (!path_los(x0, y0, x1, y1, 1, [&](int x, int y) { return stand(x, y) ? 1 : 0; })) return false;
  }
  if (e.p_n == 1 && !stand(e.wp[0], e.wp[1])) return false;
  return true;
}

DEV void path_set_none(EnvState& e) {
  e.p_n = 0;
  e.p_i = 0;
  e.p_trunc = 0;
}

// 지도 자세 칸에서 막힘(mdp::kOcc)까지 칸 가운데끼리 거리 (m), r 안에 없으면 r (기준선 obstacle_dist 와 같은 뜻 — 막힘 씨앗은 지도 막힘만)
DEV float path_obstacle_dist(const EnvBatch& b, const EnvState& e, int env, float r) {
  const EnvCfg& c = b.cfg;
  const MapDev& m = b.maps[e.map];
  const int16_t* L = b.lmap + (size_t)env * c.size * c.size;
  int cx, cy;
  cell_of(c, m, e.x + e.ex, e.y + e.ey, &cx, &cy);
  const int R = (int)ceilf(r / c.res);
  int best = 1 << 30;
  for (int dy = -R; dy <= R; ++dy)
    for (int dx = -R; dx <= R; ++dx) {
      const int u = cx + dx, v = cy + dy;
      if (u < 0 || v < 0 || u >= m.gw || v >= m.gh) continue;
      if (mdp::cell_class(L[v * c.size + u]) != mdp::kOcc) continue;
      const int d2 = dx * dx + dy * dy;
      best = d2 < best ? d2 : best;
    }
  if (best == (1 << 30)) return r;
  const float d = sqrtf((float)best) * c.res;
  return d < r ? d : r;
}

// RPP 충돌 검사 (regulated_pure_pursuit_controller.cpp isCollisionImminent, 기준선 frontier_explorer.cpp collision_imminent 와 같은 식):
// 막힘 씨앗 칸 = 지도 막힘(kOcc) 또는 로봇에서 obstacle_max_range 안의 맞음 > 지나감 칸(로그 오즈 > 0 — Nav2 장애물 층이 스캔 끝점을 바로
// 막힘으로 두는 몫, 기준선 lethal_src 와 같음). 발자국 = nav2 local_costmap footprint + footprint_padding 둘레 점(반 칸 간격)
// unk = true: 모름 칸도 막힘으로 (물러나기 — 라이다가 뒤 140° 를 못 봐 뒤 벽이 지도에 없을 수 있어서)
DEV bool path_lethal(const EnvBatch& b, const EnvState& e, int env, int x, int y, int rcx, int rcy, bool unk = false) {
  (void)e;
  const EnvCfg& c = b.cfg;
  const int16_t l = b.lmap[(size_t)env * c.size * c.size + (size_t)y * c.size + x];
  if (mdp::cell_class(l) == mdp::kOcc || (unk && l == mdp::kUnknownL)) return true;
  if (l == mdp::kUnknownL || l <= 0) return false;
  const float dx = (float)(x - rcx) * c.res, dy = (float)(y - rcy) * c.res;
  return dx * dx + dy * dy <= c.path_obs_range * c.path_obs_range;
}
DEV bool path_fp_hits(const EnvBatch& b, const EnvState& e, int env, float x, float y, float th, int rcx, int rcy, bool unk = false) {
  const EnvCfg& c = b.cfg;
  const MapDev& m = b.maps[e.map];
  float s, co;
  dm::sincosf_d(th, &s, &co);
  const float hl = c.half_len + c.path_fp_pad, hw = c.half_wid + c.path_fp_pad;
  const float cx[4] = {hl, -hl, -hl, hl}, cy[4] = {hw, hw, -hw, -hw};
  for (int k = 0; k < 4; ++k) {
    const float ax = cx[k], ay = cy[k], bx = cx[(k + 1) & 3], by = cy[(k + 1) & 3];
    const float len = sqrtf((bx - ax) * (bx - ax) + (by - ay) * (by - ay));
    int n = (int)ceilf(len / (0.5f * c.res));
    n = n < 1 ? 1 : n;
    for (int q = 0; q <= n; ++q) {
      const float u = ax + (bx - ax) * (float)q / (float)n, v = ay + (by - ay) * (float)q / (float)n;
      int qx, qy;
      if (cell_of(c, m, (x + co * u) - s * v, (y + s * u) + co * v, &qx, &qy) < 0) return true;
      if (path_lethal(b, e, env, qx, qy, rcx, rcy, unk)) return true;
    }
  }
  return false;
}
// check_now = false: 지금 자세는 안 봄 (물러나기 — 이미 발자국이 막힘에 닿아 있어도 빠져나오게)
DEV bool path_imminent(const EnvBatch& b, const EnvState& e, int env, float v, float w, float carrot_dist, bool check_now = true, bool unk = false) {
  const EnvCfg& c = b.cfg;
  const MapDev& m = b.maps[e.map];
  const float px = e.x + e.ex, py = e.y + e.ey, pth = e.th + e.eth;
  int rcx, rcy;
  cell_of(c, m, px, py, &rcx, &rcy);
  if (check_now && path_fp_hits(b, e, env, px, py, pth, rcx, rcy)) return true;
  float dt;
  if (dm::absf(v) < 0.01f && dm::absf(w) > 0.01f) {
    const float rmax = sqrtf(c.half_len * c.half_len + c.half_wid * c.half_wid);
    dt = 2.f * sinf((c.res * 0.5f) / rmax) / dm::absf(w);
  } else if (dm::absf(v) >= 0.01f) {
    dt = c.res / dm::absf(v);
  } else {
    return false;
  }
  float x = px, y = py, th = pth;
  for (int i = 1; (float)i * dt < c.path_coll_time; ++i) {
    float s, co;
    dm::sincosf_d(th, &s, &co);
    x += dt * v * co;
    y += dt * v * s;
    th += dt * w;
    if (sqrtf((x - px) * (x - px) + (y - py) * (y - py)) > carrot_dist) break;
    if (path_fp_hits(b, e, env, x, y, th, rcx, rcy, unk)) return true;
  }
  return false;
}

// 추종기 한 번 (행동 주기 0.1 s, 스레드 하나): 웨이포인트 따라 (vc, ωc) — velocity_smoother 앞 명령. p_i(지금 구간)를 앞으로만 옮긴다
DEV void path_follow(const EnvBatch& b, EnvState& e, int env, float* vc_o, float* wc_o, float* carrot_o = nullptr) {
  const EnvCfg& c = b.cfg;
  const MapDev& m = b.maps[e.map];
  *vc_o = 0.f;
  *wc_o = 0.f;
  if (carrot_o) *carrot_o = 1e9f;   // 제자리 회전이면 당근 거리 무한 (기준선과 같음)
  if (e.p_n <= 0) {   // 목표 없음: 후보가 없을 때(판 시작·모두 닫힘)는 제자리 회전, 목표는 있는데 경로를 못 짰으면 섬
    if (e.goal_cell < 0) *wc_o = c.rot_w;
    return;
  }
  const float px = e.x + e.ex, py = e.y + e.ey, pth = e.th + e.eth;
  const auto wx = [&](int i) { return m.ox + ((float)e.wp[2 * i] + 0.5f) * c.res; };
  const auto wy = [&](int i) { return m.oy + ((float)e.wp[2 * i + 1] + 0.5f) * c.res; };
  // 가장 가까운 점 (지금 구간부터 앞으로만)
  int bi = e.p_i < 0 ? 0 : e.p_i;
  float qx = wx(bi), qy = wy(bi);
  if (e.p_n >= 2) {
    float bd = 3e38f;
    for (int i = bi; i + 1 < e.p_n; ++i) {
      const float ax = wx(i), ay = wy(i), sx = wx(i + 1) - ax, sy = wy(i + 1) - ay;
      const float l2 = sx * sx + sy * sy;
      float t = l2 > 0.f ? ((px - ax) * sx + (py - ay) * sy) / l2 : 0.f;
      t = t < 0.f ? 0.f : (t > 1.f ? 1.f : t);
      const float x = ax + t * sx, y = ay + t * sy;
      const float d2 = (x - px) * (x - px) + (y - py) * (y - py);
      if (d2 < bd) { bd = d2; bi = i; qx = x; qy = y; }
    }
  } else {
    bi = 0;
  }
  e.p_i = bi;
  // 당근: 가장 가까운 점에서 경로 따라 lookahead
  float rem = c.lookahead, cx = qx, cy = qy;
  {
    int i = bi;
    float ux = qx, uy = qy;
    bool done = false;
    while (i + 1 < e.p_n) {
      const float ex2 = wx(i + 1), ey2 = wy(i + 1);
      const float d = sqrtf((ex2 - ux) * (ex2 - ux) + (ey2 - uy) * (ey2 - uy));
      if (d >= rem && d > 0.f) {
        cx = ux + (ex2 - ux) * (rem / d);
        cy = uy + (ey2 - uy) * (rem / d);
        done = true;
        break;
      }
      rem -= d;
      ux = ex2;
      uy = ey2;
      ++i;
    }
    if (!done) { cx = wx(e.p_n - 1); cy = wy(e.p_n - 1); }
  }
  const float gx = wx(e.p_n - 1), gy = wy(e.p_n - 1);
  const float d_goal = sqrtf((gx - px) * (gx - px) + (gy - py) * (gy - py));
  const float d_car = sqrtf((cx - px) * (cx - px) + (cy - py) * (cy - py));
  if (carrot_o) *carrot_o = d_car;
  // 도착: 다음 결정까지 제자리 회전 — 라이다가 뒤 140° 를 못 봐서(sensor_model.yaml fov 220°) 목표 덩어리가 로봇 뒤·발밑에 남는 일이 많다
  // (2026-11-08 규칙 정책 시험: 서 있기만 하면 0 번 후보가 거리 0 인 채로 안 풀려 128 판 모두 진행 없음 실패)
  if (d_goal <= c.goal_tol) {   // (모름 칸까지 막힘으로 보면 판 시작의 뒤쪽 모름 때문에 영영 안 돌아 128 판 모두 정체 — 지도 막힘만 봄)
    if (!path_imminent(b, e, env, 0.f, c.rot_w, 1e9f)) *wc_o = c.rot_w;
    return;
  }
  const float err = dm::wrap_pi(dm::atan2f_d(cy - py, cx - px) - pth);
  if (dm::absf(err) > c.rot_min_angle) {
    *wc_o = err > 0.f ? c.rot_w : -c.rot_w;
    return;
  }
  float vc = c.path_v;
  const float ld = dm::maxf(c.min_lookahead, dm::minf(c.lookahead, d_car));
  float se, ce;
  dm::sincosf_d(err, &se, &ce);
  const float curv = 2.f * se / ld;
  float v_curv = vc, v_cost = vc;
  {
    const float ac = dm::absf(curv);
    const float radius = ac > 1e-6f ? 1.f / ac : 1e6f;
    if (radius < c.reg_min_radius) v_curv = vc * radius / c.reg_min_radius;
  }
  if (c.path_cost_reg) {
    const float r_eff = dm::minf(c.path_cs_dist, c.path_r_out);
    const float d = path_obstacle_dist(b, e, env, r_eff);
    if (d < r_eff) v_cost = vc * c.path_cs_gain * d / c.path_cs_dist;
  }
  vc = dm::maxf(dm::minf(v_curv, v_cost), c.reg_min_speed);
  if (d_goal < c.approach_dist) vc = dm::minf(vc, dm::maxf(c.path_v * d_goal / c.approach_dist, c.approach_min_v));
  float wc = vc * curv;
  wc = dm::maxf(-c.w_max, dm::minf(c.w_max, wc));
  *vc_o = vc;
  *wc_o = wc;
}


// 물러나도 되나: 지금 방향 그대로 뒤로 d·k/4 (k = 1…4) 자세의 발자국이 막힘·모름 칸에 안 닿음 (기준선 recover 의 BackUp 앞 검사 꼴 — 뒤는 라이다가 못 봐서 모름도 막힘)
DEV bool path_backup_ok(const EnvBatch& b, const EnvState& e, int env, float d) {
  const EnvCfg& c = b.cfg;
  const MapDev& m = b.maps[e.map];
  const float px = e.x + e.ex, py = e.y + e.ey, pth = e.th + e.eth;
  int rcx, rcy;
  cell_of(c, m, px, py, &rcx, &rcy);
  float s, co;
  dm::sincosf_d(pth, &s, &co);
  for (int k = 1; k <= 4; ++k) {
    const float t = d * (float)k * 0.25f;
    if (path_fp_hits(b, e, env, px - co * t, py - s * t, pth, rcx, rcy, true)) return false;
  }
  return true;
}
