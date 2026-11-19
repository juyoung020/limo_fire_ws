// 아래층 지역 계획기 (local planner — EnvCfg::lp, env.yaml action.path.local_planner, 설계·근거 reports/local_planner_design.md).
// A* 전역 경로(웨이포인트 — 결정·재계획 때 hier_path.h 가 짬)는 길잡이로 두고, 걸음(0.1 s)마다 궤적 묶음을 섀시 실측 모형으로 굴려 비용이 가장 낮은 것의
// 첫 명령을 낸다 — Nav2 DWB 처럼 결정적 묶음에서 최솟값을 고르고(같으면 번호 작은 것 — CPU·GPU·실물 노드 비트 같음), 비용 항은 Nav2 MPPI critics 꼴.
// 길에 막힘이 보이면 서지 않고 다른 궤적으로 비켜 간다. 가장 좋은 궤적도 block_steps 안에 부딪히면(진짜 끼임) 그 걸음을 충돌 검사 막힘으로 넘긴다
// (env.h path_step_b 의 기존 가지 — 뒤가 비면 물러나기, 막힘 판정 창 · 복구).
//
// 입력은 실물에서도 같은 값만: 로봇 지도(막힘 + 3 m 안 약한 막힘 — path_lethal), 깊이 카메라 층, 지금 라이다 장 끝점, 깊이 열 점(cpts), 지도 자세,
// 지난 명령(cmd_v·cmd_w)·velocity_smoother 상태(vs·ws), 웨이포인트. 정답 격자·참값 속도는 안 씀.
//
// 거리 장: 로봇 앞 fwd_off 로 민 kLpW × kLpW 칸 창, 막힘 칸 0 → 3-4 chamfer 두 번 훑기 (칸 / 3 단위, uint8 — 85 칸에서 포화).
// 발자국: 몸통 사각형을 덮는 원 둘 (가운데 ±half_len/2, 반지름 √((half_len/2)² + half_wid²)) — 단단한 충돌 = 장 거리 < 반지름 + margin.
// 섀시 모형 굴리기 (걸음 = 행동 주기): 명령은 velocity_smoother 와 같은 식(env.h path_step_b — 가속 a_max · alpha_max), ω 명령 = vκ / turn_gain(v)
// (w_max 자름 — hier_path.h path_follow 의 돌며 가기와 같은 식), 몸통 = lag_mix · 이번 명령 + (1 − lag_mix) · 지난 명령 (섀시 지연 60–80 ms),
// 몸통 ω = turn_gain(|몸통 v|) × 명령 ω (drive.h drive_step 과 같은 몫).
// 맞추기 장면에서 더한 것 (reports/local_planner_design.md 사다리): 가운데 틈 하한(center_clear — 위층 후보·닫힘의 설 수 있는 칸 밖으로 안 나감,
// 벽에 붙으면 후보 찾기가 막혀 '닫힘' 으로 일찍 끝나던 것), 직전 라이다 장 기억(뒤 사각), 막힘 곁 감속(slow_dist — RPP cost 감속 꼴),
// 걸음마다 경로 옆 거리 비용, 도는 쪽 뒤집기 비용(좁은 곳에서 좌우로 떨며 서 있던 것), 움직이는 점 등속 예측(사람 — predict_moving).
// 이 헤더는 namespace sim 안에서 넣는다 (env.h — hier_path.h 뒤).
#pragma once

constexpr int kLpW = 64;    // 거리 장 창 (칸)

struct LpWin {
  int gx0, gy0;   // 창 왼쪽 아래 칸 (지도 칸 번호)
};

// 거리 장 (D[kLpW · kLpW], 칸 / 3 단위)
DEV void lp_field(const EnvBatch& b, const EnvState& e, int env, const LpWin& w, uint8_t* D) {
  const EnvCfg& c = b.cfg;
  const MapDev& m = b.maps[e.map];
  const float px = e.x + e.ex, py = e.y + e.ey;
  int rcx, rcy;
  cell_of(c, m, px, py, &rcx, &rcy);
  for (int j = 0; j < kLpW; ++j)
    for (int i = 0; i < kLpW; ++i) {
      const int gx = w.gx0 + i, gy = w.gy0 + j;
      const bool out = gx < 0 || gy < 0 || gx >= m.gw || gy >= m.gh;
      D[j * kLpW + i] = (out || path_lethal(b, e, env, gx, gy, rcx, rcy)) ? 0 : 255;
    }
  const auto mark = [&](float x, float y) {
    const int i = dm::floor_div(x - m.ox, c.res, c.inv_res) - w.gx0, j = dm::floor_div(y - m.oy, c.res, c.inv_res) - w.gy0;
    if (i >= 0 && j >= 0 && i < kLpW && j < kLpW) D[j * kLpW + i] = 0;
  };
  {   // 지금 라이다 장 끝점 (사람·지도에 아직 없는 것)
    float s, co;
    dm::sincosf_d(e.th + e.eth, &s, &co);
    for (int k = 0; k < mdp::obs::LIDAR_BINS; ++k) {
      float x, y, r;
      if (!scan_point(b, e, env, k, &x, &y, &r)) continue;
      mark(px + (co * x - s * y), py + (s * x + co * y));
    }
  }
  if (c.lp.ring) {   // 직전 장들 (링의 지난 kLidarRing − 1 장, 그때 지도 자세 lid_pose) — 돌며 뒤 140° 사각으로 들어간 점 (hier_path.h path_rotate_count 와 같은 기억)
    for (int f = 1; f < kLidarRing; ++f) {
      const int slot = (e.lid_head - f + kLidarRing) % kLidarRing;
      const float* R = b.lidar_ring + ((size_t)env * kLidarRing + slot) * mdp::obs::LIDAR_BINS;
      const float fx = e.lid_pose[3 * slot], fy = e.lid_pose[3 * slot + 1];
      float fs, fc;
      dm::sincosf_d(e.lid_pose[3 * slot + 2], &fs, &fc);
      for (int k = 0; k < mdp::obs::LIDAR_BINS; ++k) {
        const float r = R[k];
        if (!(r < c.map_max_range) || r > c.lp.ring_range) continue;
        const float an = mdp::obs::kLidarMin + ((float)k + 0.5f) * mdp::obs::kLidarBin;
        float s, co;
        dm::sincosf_d(an, &s, &co);
        const float bx = c.mount_x + r * co, by = c.mount_y + r * s;
        mark(fx + (fc * bx - fs * by), fy + (fs * bx + fc * by));
      }
    }
  }
  if (b.cpts) {   // 깊이 열 점 (collision_monitor 출처와 같은 점, 지도 좌표)
    const float* Q = b.cpts + (size_t)env * mdp::obs::DEPTH_W * 2;
    for (int i = 0; i < mdp::obs::DEPTH_W && Q[2 * i] != kCamNone; ++i) mark(Q[2 * i], Q[2 * i + 1]);
  }
  const auto relax = [&](int idx, int nb, int add) {
    const int v = (int)D[nb] + add;
    if (v < (int)D[idx]) D[idx] = (uint8_t)v;
  };
  for (int j = 0; j < kLpW; ++j)
    for (int i = 0; i < kLpW; ++i) {
      const int idx = j * kLpW + i;
      if (D[idx] == 0) continue;
      if (i > 0) relax(idx, idx - 1, 3);
      if (j > 0) {
        relax(idx, idx - kLpW, 3);
        if (i > 0) relax(idx, idx - kLpW - 1, 4);
        if (i + 1 < kLpW) relax(idx, idx - kLpW + 1, 4);
      }
    }
  for (int j = kLpW - 1; j >= 0; --j)
    for (int i = kLpW - 1; i >= 0; --i) {
      const int idx = j * kLpW + i;
      if (D[idx] == 0) continue;
      if (i + 1 < kLpW) relax(idx, idx + 1, 3);
      if (j + 1 < kLpW) {
        relax(idx, idx + kLpW, 3);
        if (i + 1 < kLpW) relax(idx, idx + kLpW + 1, 4);
        if (i > 0) relax(idx, idx + kLpW - 1, 4);
      }
    }
}
// 점 (지도 좌표)의 막힘까지 거리 (m). 창 밖이면 *out = true
DEV float lp_dist(const EnvCfg& c, const MapDev& m, const LpWin& w, const uint8_t* D, float x, float y, bool* out) {
  const int i = dm::floor_div(x - m.ox, c.res, c.inv_res) - w.gx0, j = dm::floor_div(y - m.oy, c.res, c.inv_res) - w.gy0;
  if (i < 0 || j < 0 || i >= kLpW || j >= kLpW) { *out = true; return 1e9f; }
  return (float)D[j * kLpW + i] * (c.res * (1.f / 3.f));
}

// 웨이포인트 꺾은선 (지도 좌표)과 누적 호 길이
struct LpPath {
  float x[mdp::hier::kPathMax], y[mdp::hier::kPathMax], cum[mdp::hier::kPathMax];
  int n, i0;
  float total;
};
// 투영: 점 (px, py) 에서 가장 가까운 경로 점 — 구간 i0 부터 호 길이 s_lim 까지만 (U 자 경로의 먼 쪽으로 뛰지 않게).
// *s 호 길이 (경로 끝을 넘으면 끝까지 남은 거리를 뺌 — 목표), *lat 옆 거리, *tan 그 구간 방향, *seg 구간 번호
DEV void lp_proj(const LpPath& P, float px, float py, float s_lim, float* s, float* lat, float* tan, int* seg) {
  if (P.n < 2) {
    const float dx = P.x[0] - px, dy = P.y[0] - py;
    *lat = sqrtf(dx * dx + dy * dy);
    *s = -*lat;
    *tan = dm::atan2f_d(dy, dx);
    if (seg) *seg = 0;
    return;
  }
  float bd = 3e38f, bs = 0.f, bt = 0.f;
  int bi = P.i0;
  for (int i = P.i0; i + 1 < P.n && P.cum[i] <= s_lim; ++i) {
    const float ax = P.x[i], ay = P.y[i], sx = P.x[i + 1] - ax, sy = P.y[i + 1] - ay;
    const float l2 = sx * sx + sy * sy;
    float t = l2 > 0.f ? ((px - ax) * sx + (py - ay) * sy) / l2 : 0.f;
    t = t < 0.f ? 0.f : (t > 1.f ? 1.f : t);
    const float qx = ax + t * sx, qy = ay + t * sy;
    const float d2 = (qx - px) * (qx - px) + (qy - py) * (qy - py);
    if (d2 < bd) { bd = d2; bi = i; bs = P.cum[i] + t * (P.cum[i + 1] - P.cum[i]); bt = dm::atan2f_d(sy, sx); }
  }
  *lat = sqrtf(bd);
  *s = bs >= P.total ? P.total - *lat : bs;
  *tan = bt;
  if (seg) *seg = bi;
}

// 지역 계획기 한 번: 쓰면 true (vc · wc = velocity_smoother 앞 목표 명령, *block = 진짜 끼임). 도착(goal_tol 안)·목표 없음·경로 없음이면 false —
// 아무 상태도 안 바꾸고 부르는 쪽(env.h path_step_a)이 RPP 추종기를 그대로 부른다
DEV bool lp_follow(const EnvBatch& b, EnvState& e, int env, float* vc_o, float* wc_o, int* block, PathWhy* why) {
  const EnvCfg& c = b.cfg;
  const LpCfg& L = c.lp;
  const MapDev& m = b.maps[e.map];
  if (e.p_n <= 0) return false;
  const float px = e.x + e.ex, py = e.y + e.ey, pth = e.th + e.eth;
  LpPath P;
  P.n = e.p_n;
  P.cum[0] = 0.f;
  for (int i = 0; i < P.n; ++i) {
    P.x[i] = m.ox + ((float)e.wp[2 * i] + 0.5f) * c.res;
    P.y[i] = m.oy + ((float)e.wp[2 * i + 1] + 0.5f) * c.res;
    if (i > 0) {
      const float dx = P.x[i] - P.x[i - 1], dy = P.y[i] - P.y[i - 1];
      P.cum[i] = P.cum[i - 1] + sqrtf(dx * dx + dy * dy);
    }
  }
  P.total = P.cum[P.n - 1];
  {
    const float gx = P.x[P.n - 1] - px, gy = P.y[P.n - 1] - py;
    if (sqrtf(gx * gx + gy * gy) <= c.goal_tol) return false;   // 도착: 추종기의 도착 회전
  }
  const float pv = path_vmax(b, env);
  const float dt = (float)c.substeps * c.dt;
  const int N = L.n_steps;
  // 지금 자세 투영 (구간은 앞으로만 — path_follow 의 p_i 와 같은 뜻)
  P.i0 = e.p_i < 0 ? 0 : (e.p_i > P.n - 2 ? (P.n >= 2 ? P.n - 2 : 0) : e.p_i);
  float s0, lat0, tan0;
  int seg0;
  lp_proj(P, px, py, 3e38f, &s0, &lat0, &tan0, &seg0);
  e.p_i = seg0;
  P.i0 = seg0;
  const float s_lim = s0 + pv * (float)N * dt + 1.f;
  // 거리 장
  LpWin w;
  {
    float s, co;
    dm::sincosf_d(pth, &s, &co);
    w.gx0 = dm::floor_div(px + L.fwd_off * co - m.ox, c.res, c.inv_res) - kLpW / 2;
    w.gy0 = dm::floor_div(py + L.fwd_off * s - m.oy, c.res, c.inv_res) - kLpW / 2;
  }
  uint8_t D[kLpW * kLpW];
  lp_field(b, e, env, w, D);
  const float a = 0.5f * c.half_len;
  const float r0 = sqrtf(a * a + c.half_wid * c.half_wid);
  float r_h = r0 + L.margin;
  float r_c = L.center_min;   // 가운데 점 틈 하한 — 위층 후보·계획이 쓰는 "설 수 있는 칸"(반경 0.2 m) 밖으로 안 나가게 (벽에 붙으면 후보 찾기·회전이 막힘)
  {   // 지금 자세가 이미 단단한 반지름 안이면 (붙어 있음) 지금보다 더 가까워지는 것만 충돌로 — 빠져나오게
    float s, co;
    dm::sincosf_d(pth, &s, &co);
    bool o1 = false, o2 = false;
    const float d0 = dm::minf(lp_dist(c, m, w, D, px + a * co, py + a * s, &o1), lp_dist(c, m, w, D, px - a * co, py - a * s, &o2));
    if (d0 < r_h) r_h = dm::maxf(0.f, d0 - 0.005f);
    const float dc = lp_dist(c, m, w, D, px, py, &o1);
    if (dc < r_c) r_c = dm::maxf(0.f, dc - 0.005f);
  }
  // 움직이는 점 (사람): 지금 장에서 로봇 지도가 막힘으로 굳히지 않은 칸(사람이 지나는 바닥 — 한 장 맞아 로그 오즈가 0 을 조금 넘어도)에 맞은 점 ↔
  // 링의 가장 오래된 장(0.4 s 전)의 같은 꼴 점, 최근접 짝으로 속도. 벽 점과 짝 짓지 않게 옛 점도 같은 거름
  constexpr int kDyn = 64;
  float dqx[kDyn], dqy[kDyn], dvx[kDyn], dvy[kDyn];
  int n_dyn = 0;
  if (L.ppl) {
    const int f = kLidarRing - 1;
    const int so = (e.lid_head - f + kLidarRing) % kLidarRing;
    const float Dt = (float)f * dt;
    const auto dyn_ok = [&](float wx, float wy) {
      int gx, gy;
      if (cell_of(c, m, wx, wy, &gx, &gy) < 0) return false;
      const int16_t l = b.lmap[(size_t)env * c.size * c.size + (size_t)gy * c.size + gx];
      return mdp::cell_class(l) != mdp::kOcc && !cam_lethal(b, env, gx, gy);
    };
    float s, co;
    dm::sincosf_d(pth, &s, &co);
    const float* Ro = b.lidar_ring + ((size_t)env * kLidarRing + so) * mdp::obs::LIDAR_BINS;
    const float ofx = e.lid_pose[3 * so], ofy = e.lid_pose[3 * so + 1];
    float ofs, ofc;
    dm::sincosf_d(e.lid_pose[3 * so + 2], &ofs, &ofc);
    for (int k = 0; k < mdp::obs::LIDAR_BINS && n_dyn < kDyn; k += 2) {   // 칸 하나 걸러 (사람 하나에 점 여럿)
      float bx, by, r;
      if (!scan_point(b, e, env, k, &bx, &by, &r) || r > 3.f) continue;
      const float wx = px + (co * bx - s * by), wy = py + (s * bx + co * by);
      if (!dyn_ok(wx, wy)) continue;
      float bd = 0.36f, qx = wx, qy = wy;   // 짝 반경 0.6 m (0.4 s · 1.5 m/s)
      for (int j = 0; j < mdp::obs::LIDAR_BINS; ++j) {
        const float ro = Ro[j];
        if (!(ro < c.map_max_range) || ro > 3.5f) continue;
        const float an = mdp::obs::kLidarMin + ((float)j + 0.5f) * mdp::obs::kLidarBin;
        float sa, ca;
        dm::sincosf_d(an, &sa, &ca);
        const float ox = c.mount_x + ro * ca, oy = c.mount_y + ro * sa;
        const float ux = ofx + (ofc * ox - ofs * oy), uy = ofy + (ofs * ox + ofc * oy);
        const float d2 = (ux - wx) * (ux - wx) + (uy - wy) * (uy - wy);
        if (d2 < bd && dyn_ok(ux, uy)) { bd = d2; qx = ux; qy = uy; }
      }
      float vx = (wx - qx) / Dt, vy = (wy - qy) / Dt;
      const float sp = sqrtf(vx * vx + vy * vy);
      if (sp < L.ppl_vmin) continue;
      if (sp > L.ppl_vmax) { vx *= L.ppl_vmax / sp; vy *= L.ppl_vmax / sp; }
      dqx[n_dyn] = wx; dqy[n_dyn] = wy; dvx[n_dyn] = vx; dvy[n_dyn] = vy;
      ++n_dyn;
    }
  }
  const int k_hard = (int)(L.ppl_hard_t / dt + 0.5f);
  const float sv = c.a_max * dt, sw = c.alpha_max * dt;
  const int n_lat = L.n_v * L.n_k1 * L.n_k2;
  const int n_all = n_lat + (L.rot_on ? 2 : 0);
  // 제자리 회전 둘: 추종기와 같은 RPP 회전 검사(패딩 발자국을 돌려 지도·깊이 층 막힘 — hier_path.h path_imminent)에 걸리면 뺌 (원 둘 근사는 벽에 붙어 돌 때 모서리를 놓침)
  bool rot_bad[2] = {false, false};
  if (L.rot_on) {
    rot_bad[0] = path_imminent(b, e, env, 0.f, c.rot_w, 1e9f);
    rot_bad[1] = path_imminent(b, e, env, 0.f, -c.rot_w, 1e9f);
  }
  float best = 3e38f, b_vc = 0.f, b_wc = 0.f, b_k1 = 0.f, b_clr = 0.f, b_vt = 0.f;
  int b_idx = -1, b_coll = -1;
  for (int idx = 0; idx < n_all; ++idx) {
    const bool rot = idx >= n_lat;
    float vt = 0.f, k1 = 0.f, k2 = 0.f, wr = 0.f;
    if (rot) {
      if (rot_bad[idx - n_lat]) continue;
      wr = idx == n_lat ? c.rot_w : -c.rot_w;
    } else {
      const int iv = idx / (L.n_k1 * L.n_k2), r = idx % (L.n_k1 * L.n_k2);
      vt = dm::minf(L.v[iv], pv);
      k1 = L.k1[r / L.n_k2];
      k2 = L.k2[r % L.n_k2];
    }
    float x = px, y = py, th = pth;
    float vs = e.vs, ws = e.ws, vp = e.cmd_v, wp = e.cmd_w;
    float vsum = 0.f, clr = 0.f, devs = 0.f, pcost = 0.f, f_vc = 0.f, f_wc = 0.f;
    int coll = -1, steps = 0;
    for (int k = 0; k < N; ++k) {
      // 목표 명령 → velocity_smoother (path_step_b 와 같은 식)
      float vtk, wtk;
      if (rot) {
        vtk = 0.f;
        wtk = wr;
      } else {
        vtk = vt;
        if (L.slow_dist > 0.f) {   // 막힘 곁 감속 (RPP use_cost_regulated_linear_velocity_scaling 꼴 — 센서가 못 본 낮은·높은 가구 곁을 천천히): 가운데 틈 / slow_dist 배
          bool o = false;
          const float dcen = lp_dist(c, m, w, D, x, y, &o);
          if (dcen < L.slow_dist) vtk = dm::minf(vtk, dm::maxf(L.slow_min, pv * dcen / L.slow_dist));
        }
        const float kap = k < L.k_switch ? k1 : k2;
        const float vn = vs + dm::maxf(-sv, dm::minf(sv, dm::maxf(-c.path_v, dm::minf(c.path_v, vtk)) - vs));
        wtk = (c.drive.n_tg > 0) ? vn * kap / dm::maxf(turn_gain(c.drive, vn), 0.05f) : vn * kap;
        wtk = dm::maxf(-c.w_max, dm::minf(c.w_max, wtk));
      }
      if (k == 0) { f_vc = vtk; f_wc = wtk; }
      const float vt2 = dm::maxf(-c.path_v, dm::minf(c.path_v, vtk));
      const float dv = vt2 - vs, dw = wtk - ws;
      vs = vs + (dv > sv ? sv : (dv < -sv ? -sv : dv));
      ws = ws + (dw > sw ? sw : (dw < -sw ? -sw : dw));
      // 몸통 (한 걸음 늦게 섞임) · 달리며 ω 몫
      const float vb = L.lag_mix * vs + (1.f - L.lag_mix) * vp;
      const float wb = turn_gain(c.drive, vb) * (L.lag_mix * ws + (1.f - L.lag_mix) * wp);
      vp = vs;
      wp = ws;
      float s, co;
      dm::sincosf_d(th + 0.5f * wb * dt, &s, &co);
      const float nx = x + vb * co * dt, ny = y + vb * s * dt, nth = th + wb * dt;
      dm::sincosf_d(nth, &s, &co);
      bool out = false;
      const float d = dm::minf(lp_dist(c, m, w, D, nx + a * co, ny + a * s, &out), lp_dist(c, m, w, D, nx - a * co, ny - a * s, &out));
      if (out) break;   // 창 밖: 그 앞까지로 셈
      if (d < r_h || lp_dist(c, m, w, D, nx, ny, &out) < r_c) { coll = k; break; }
      if (n_dyn > 0) {   // 움직이는 점 예측 자리 (걸음 k 끝 시각)
        const float tk = (float)(k + 1) * dt;
        float dmin2 = 1e9f;
        for (int q = 0; q < n_dyn; ++q) {
          const float ex = dqx[q] + dvx[q] * tk - nx, ey = dqy[q] + dvy[q] * tk - ny;
          dmin2 = dm::minf(dmin2, ex * ex + ey * ey);
        }
        if (dmin2 < L.ppl_r * L.ppl_r) {
          if (k < k_hard) { coll = k; break; }
          const float g = 1.f - sqrtf(dmin2) / L.ppl_r;
          pcost += g * g;
        }
      }
      x = nx;
      y = ny;
      th = nth;
      vsum += vb;
      const float q = dm::maxf(0.f, r_h + L.r_soft - d) / L.r_soft;
      clr += q * q;
      {   // 걸음마다 경로 옆 거리 (경로를 크게 벗어나 도는 호보다 제자리 회전 뒤 경로 위로 — 경로가 아는 막힘에서 가장 넓은 길)
        float sk, lk, tk;
        lp_proj(P, x, y, s_lim, &sk, &lk, &tk, nullptr);
        const float dk = dm::maxf(0.f, lk - L.dev_free);
        devs += dk * dk;
      }
      ++steps;
    }
    float cost = coll >= 0 ? 1000.f * (float)(N - coll) : 0.f;
    float se, late, tane;
    lp_proj(P, x, y, s_lim, &se, &late, &tane, nullptr);
    cost -= L.w_prog * (se - s0) / ((float)N * dt);
    cost -= L.w_v * vsum / (float)N;
    const float dev = dm::maxf(0.f, late - L.dev_free);
    cost += L.w_dev * (dev * dev + devs / (float)N);
    if (f_wc * e.ws < 0.f && dm::absf(e.ws) > 0.1f) cost += L.w_flip;   // 도는 쪽 뒤집기 (제자리에서 좌우로 떨기 막기)
    {
      float s, co;
      dm::sincosf_d(dm::wrap_pi(th - tane), &s, &co);
      cost += L.w_head * (1.f - co);
    }
    cost += L.w_clr * clr / (float)N + L.w_ppl * pcost / (float)N;
    if (rot) cost += L.w_rot;
    else if (e.lp_n > 0) cost += L.w_hys * dm::absf(k1 - e.lp_k1);
    if (cost < best) {
      best = cost;
      b_idx = idx;
      b_vc = f_vc;
      b_wc = f_wc;
      b_k1 = rot ? 0.f : k1;
      b_coll = coll;
      b_clr = steps > 0 ? clr / (float)steps : 0.f;
      b_vt = vt;
    }
  }
  const bool rot = b_idx >= n_lat;
  *block = (b_coll >= 0 && b_coll < L.block_steps) ? 1 : 0;
  *vc_o = *block ? 0.f : b_vc;
  *wc_o = *block ? 0.f : b_wc;
  e.lp_k1 = b_k1;
  e.lp_n = 1;
  if (why) {
    const float gx = P.x[P.n - 1] - px, gy = P.y[P.n - 1] - py;
    why->kind = rot ? kWhyRot : kWhyGo;
    why->bind = b_vt >= 0.9f * pv ? kLowFull : (b_clr > 0.f ? kLowCost : kLowCurv);
    why->vc = *vc_o;
    why->err = dm::wrap_pi(tan0 - pth);
    why->d_goal = sqrtf(gx * gx + gy * gy);
  }
  return true;
}
