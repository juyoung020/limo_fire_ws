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
//
// 지금 스캔 (2026-11-09 사용자 결정 — 사람·움직이는 장애물): 정답 사람 자리는 안 쓴다. 쓰는 것은 지금 라이다 장(lidar_ring 의 lid_head 칸, 248 각도 칸마다
// 가장 가까운 거리 — 관측·안전층 front_gap 과 같은 값, 실물 /scan 으로 같은 값을 셀 수 있음)뿐. 근거 측정: explore_h1 780 반복 충돌 0.23 중 사람 0.105,
// 단계 0~3 충돌 0.02 · 0.14 · 0.21 · 0.28 (사람 수 0 · 1 · 2 · 3 과 같이 오름). 지도는 사람이 빈 칸으로 여러 번 본 바닥에 서면 한 장의 맞음으로는
// 로그 오즈가 0 을 못 넘어(빈 칸 갱신이 쌓인 만큼) 약한 막힘에도 안 잡힌다.
//   (a) 계획: 스캔 끝점(센서에서 obstacle_max_range 3 m 안 — nav2_params global obstacle_layer)을 엄격·느슨 둘 다 막힘으로 더한다
//       (기본 all = 모든 점. map_free = 지도가 빈 칸으로 아는 칸에 맞은 점만 — 비교는 path_scan_cell).
//       Nav2 ObstacleLayer marking 과 같은 뜻 (nav2_costmap_2d obstacle_layer.cpp updateBounds: 범위 안 점 칸 → LETHAL_OBSTACLE, 기본 obstacle_max_range 2.5).
//       지우기(clearing, raytrace)는 따로 안 한다 — 다음 결정에 지금 장으로 다시 짜므로 지난 장 점이 남지 않음(층이 결정마다 새로 만들어지는 꼴).
//       팽창은 지도 막힘과 같다 (설 수 있음 = 로봇 반경 0.2 m 밖, 무게 경계 0.3 · 0.45 m). 각도는 칸 가운데 (칸 폭 0.89° → 3 m 에서 4.7 cm < 지도 칸 5 cm).
//   (b) 추종기: Nav2 collision_monitor 의 approach (FootprintApproach) — velocity_smoother 뒤, 안전층 앞. 지금 명령 (v, ω) 로 발자국(RPP 충돌 검사와 같은
//       footprint + padding)을 simulation_time_step 마다 time_before_collision 까지 굴려 발자국 안 스캔 점이 max_points 넘는 첫 시각 t 면 (v, ω) × t / T
//       (nav2_collision_monitor polygon.cpp getCollisionTime · collision_monitor_node.cpp processApproach, 값은 humble collision_monitor_params.yaml).
//       Nav2 와 다른 한 곳: 지금 발자국 안에 이미 든 점은 첫 걸음에 더 깊이 들 때만 센다(그 수 > max_points 면 0 = 멈춤, Nav2 의 0 초 충돌) —
//       Nav2 는 안에 든 점이 있으면 무조건 멈추는데, 그러면 곁에 붙은 사람·벽에서 돌거나 물러나 빠져나오지도 못하고 서 있다가 걸어오는 사람과
//       부딪힌다(시뮬은 사람이 다가와 겹쳐도 충돌). 처음 판(안에 든 점을 아예 안 셈)은 0.05 배까지 늦춘 뒤 패딩 안으로 들면 다시 제 속도로 밀고 들어갔다.
//   (c) 제자리 회전: 보이는 앞 220° 안 점은 (b) 가 발자국을 ω 로 돌려 보고 줄인다. 라이다가 못 보는 뒤 140° 는 지도 약한 막힘으로 RPP 충돌 검사
//       (path_imminent — 제자리 회전 세 곳 모두 거침)가 막고, 지도에 안 남는 사람(빈 칸으로 여러 번 본 바닥)은 직전 4 장(0.4 s)을 그때 자세와 함께
//       기억해 회전 원 안 뒤쪽 점이 있으면 그 걸음은 돌지 않는다(path_rotate_blocked).
//   측정(mdp/README "계층형" — 규칙 정책 256 판씩): 켜면 성공·정체는 뚜렷이 좋아지나 사람 충돌은 표준오차 안. 사람 충돌의 대부분은 로봇을 무시하고 걷는
//   사람이 서 있거나 제자리에서 도는 로봇에 걸어 들어온 것 — 지금 장(정지 장면 가정)만으로는 못 막고 움직임 추정·비켜서기가 필요하다. (c) 는 거의 안 걸림.
#pragma once
// namespace sim 안에서 넣는다 (env.h — people_sim.h 와 같게)

constexpr int kPathPlanes = 13;   // GPU 계획 비트 판 수 (EnvBatch::pplanes): 무게 경계 2 · 버킷 4 · 거리 mod 7 3 비트 · 엄격 막힘 · 엄격 설 수 있음 · 느슨 막힘(지도 ∪ 스캔) · 느슨 설 수 있음
enum : int { kPpC3 = 0, kPpC45 = 1, kPpB = 2, kPpD = 6, kPpLT = 9, kPpSST = 10, kPpLO = 11, kPpSLO = 12 };

// 지금 라이다 장 각도 칸 k 의 점 (base_link 기준 m). 맞은 점이 없으면 false. *r = 센서에서 거리
DEV bool scan_point(const EnvBatch& b, const EnvState& e, int env, int k, float* px, float* py, float* r_o) {
  const EnvCfg& c = b.cfg;
  const float r = b.lidar_ring[((size_t)env * kLidarRing + e.lid_head) * mdp::obs::LIDAR_BINS + k];
  if (!(r < c.map_max_range)) return false;
  const float a = mdp::obs::kLidarMin + ((float)k + 0.5f) * mdp::obs::kLidarBin;
  float s, co;
  dm::sincosf_d(a, &s, &co);
  *px = c.mount_x + r * co;
  *py = c.mount_y + r * s;
  *r_o = r;
  return true;
}
// (a) 계획에 넣을 스캔 끝점 칸: 센서에서 obstacle_max_range(path_obs_range) 안, 지도 자세로 칸. 맵 밖이면 false.
// path_scan 2 (all, 기본 — Nav2 그대로): 모든 점. 1 (map_free): 지도는 그 칸을 빈 칸으로 아는데(로그 오즈 ≤ 0) 지금 스캔이 맞은 점 = 지도와 어긋나는 점만
// (움직이는 물체) — 막 보인 벽 옆 목표를 느슨으로도 못 짜 서 버리는 판을 줄이려던 판. 2026-11-09 규칙 정책 256 판씩 비교에서 all 이 정답 덮음 ≥ 0.8 인
// 성공 0.32·0.48 vs map_free 0.27·0.45 (서점·작은 집, 표준오차 ≈ 0.03 — 차이 없다고 봄), 대신 성공까지 153·149 s vs 134·145 s → 출처 있는 Nav2 쪽을 기본으로
DEV bool path_scan_cell(const EnvBatch& b, const EnvState& e, int env, int k, int* cx, int* cy) {
  const EnvCfg& c = b.cfg;
  float x, y, r;
  if (!scan_point(b, e, env, k, &x, &y, &r) || r > c.path_obs_range) return false;
  const float pth = e.th + e.eth;
  float s, co;
  dm::sincosf_d(pth, &s, &co);
  const int ci = cell_of(c, b.maps[e.map], (e.x + e.ex) + (co * x - s * y), (e.y + e.ey) + (s * x + co * y), cx, cy);
  if (ci < 0) return false;
  if (c.path_scan == 2) return true;   // 모든 점
  const int16_t l = b.lmap[(size_t)env * c.size * c.size + ci];
  return l != mdp::kUnknownL && l <= 0;
}

// (c) 제자리 회전 검사: 직전 장들(링의 지난 kLidarRing − 1 장, 0.1~0.4 s 전 — 장마다 그때 지도 자세 lid_pose)의 점을 지금 로봇 기준으로 옮겨,
// 지금 라이다가 못 보는 곳(라이다에서 ±110° 밖 — 뒤 140°)에 있고 몸통이 돌며 쓰는 원(발자국 꼭짓점 반경) 안·지금 발자국 밖인 점이 max_points 넘으면 true.
// 보이는 곳은 지금 장이 더 새 값이라 (b) 가 맡는다. 지도로는 안 잡히는 경우(사람이 빈 칸으로 여러 번 본 바닥에 섬 — 위 머리말)를 위한 짧은 기억
DEV bool path_rotate_blocked(const EnvBatch& b, const EnvState& e, int env) {
  const EnvCfg& c = b.cfg;
  const float hl = c.half_len + c.path_fp_pad, hw = c.half_wid + c.path_fp_pad;
  const float rs2 = hl * hl + hw * hw;
  const float fov = -mdp::obs::kLidarMin;   // 110°
  const float px = e.x + e.ex, py = e.y + e.ey, pth = e.th + e.eth;
  float ps, pc;
  dm::sincosf_d(pth, &ps, &pc);
  int cnt = 0;
  for (int f = 1; f < kLidarRing; ++f) {
    const int slot = (e.lid_head - f + kLidarRing) % kLidarRing;
    const float* R = b.lidar_ring + ((size_t)env * kLidarRing + slot) * mdp::obs::LIDAR_BINS;
    const float fx = e.lid_pose[3 * slot], fy = e.lid_pose[3 * slot + 1];
    float fs, fc;
    dm::sincosf_d(e.lid_pose[3 * slot + 2], &fs, &fc);
    for (int k = 0; k < mdp::obs::LIDAR_BINS; ++k) {
      const float r = R[k];
      if (!(r < c.map_max_range) || r > 1.f) continue;   // 1 m 밖 점은 몸통 원(≈ 0.27 m)에 못 듦 — 라이다는 몸통 앞 0.1 m
      const float a = mdp::obs::kLidarMin + ((float)k + 0.5f) * mdp::obs::kLidarBin;
      float s, co;
      dm::sincosf_d(a, &s, &co);
      const float bx = c.mount_x + r * co, by = c.mount_y + r * s;            // 그 장 base_link
      const float wx = fx + (fc * bx - fs * by), wy = fy + (fs * bx + fc * by);   // 지도
      const float dx = wx - px, dy = wy - py;
      const float x = pc * dx + ps * dy, y = pc * dy - ps * dx;               // 지금 base_link
      if (x * x + y * y > rs2) continue;
      if (dm::absf(x) <= hl && dm::absf(y) <= hw) continue;
      if (dm::absf(dm::atan2f_d(y - c.mount_y, x - c.mount_x)) <= fov) continue;   // 지금 보이는 곳
      ++cnt;
    }
  }
  return cnt > c.mon_max_points;
}

// (b) collision_monitor approach: 명령 (v, ω) 에 곱할 배율 [0, 1] (1 = 걸림 없음)
DEV float path_monitor_scale(const EnvBatch& b, const EnvState& e, int env, float v, float w) {
  const EnvCfg& c = b.cfg;
  if (!c.mon_on || (v == 0.f && w == 0.f)) return 1.f;
  const int n = (int)(c.mon_t / c.mon_dt + 1e-3f);   // 굴릴 걸음 수 (≤ 64, env_host 가 확인)
  const float hl = c.half_len + c.path_fp_pad, hw = c.half_wid + c.path_fp_pad;
  // 굴린 자세 (지금 base_link 기준, Euler — nav2 projectState 와 같은 차례: 방향을 먼저 돌리고 그 방향으로 나아감)
  float qx[64], qy[64], qc[64], qs[64];
  int cnt[64];
  {
    float x = 0.f, y = 0.f, th = 0.f;
    for (int j = 0; j < n; ++j) {
      th += w * c.mon_dt;
      float s, co;
      dm::sincosf_d(th, &s, &co);
      x += v * co * c.mon_dt;
      y += v * s * c.mon_dt;
      qx[j] = x;
      qy[j] = y;
      qc[j] = co;
      qs[j] = s;
      cnt[j] = 0;
    }
  }
  const float reach = dm::absf(v) * c.mon_t + sqrtf(hl * hl + hw * hw);   // 이보다 먼 점은 어느 자세의 발자국에도 안 듦
  int cnt0 = 0;   // 지금 발자국 안이고 첫 걸음에 더 깊이 드는 점 (위 머리말 "Nav2 와 다른 한 곳")
  for (int k = 0; k < mdp::obs::LIDAR_BINS; ++k) {
    float px, py, r;
    if (!scan_point(b, e, env, k, &px, &py, &r)) continue;
    if (px * px + py * py > reach * reach) continue;
    if (dm::absf(px) <= hl && dm::absf(py) <= hw) {   // 지금 발자국 안: 발자국 가장자리까지 깊이가 첫 걸음에 늘면 지금 걸림, 아니면(빠져나감) 안 셈
      const float dx = px - qx[0], dy = py - qy[0];
      const float u = qc[0] * dx + qs[0] * dy, t = qc[0] * dy - qs[0] * dx;
      const float d0 = dm::minf(hl - dm::absf(px), hw - dm::absf(py)), d1 = dm::minf(hl - dm::absf(u), hw - dm::absf(t));
      cnt0 += d1 > d0;
      continue;
    }
    for (int j = 0; j < n; ++j) {
      const float dx = px - qx[j], dy = py - qy[j];
      const float u = qc[j] * dx + qs[j] * dy, t = qc[j] * dy - qs[j] * dx;
      if (dm::absf(u) <= hl && dm::absf(t) <= hw) cnt[j] += 1;
    }
  }
  if (cnt0 > c.mon_max_points) return 0.f;   // getCollisionTime 이 0 을 내는 경우 (processApproach → 멈춤)
  for (int j = 0; j < n; ++j)
    if (cnt[j] > c.mon_max_points) return (float)(j + 1) * c.mon_dt / c.mon_t;   // 첫 걸린 시각 / T (processApproach change_ratio)
  return 1.f;
}
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
