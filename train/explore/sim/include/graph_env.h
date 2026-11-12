// 그래프 모드 (mdp/graph.h) 의 시뮬 쪽 풀 — env.h 가 namespace sim 안에서 포함한다 (EnvBatch·EnvState·cell_of 뒤).
// GPU 커널(src/env_kernel.cu env_graph_kernel)·CPU 참조·실물 노드(include/obs_ref.h graph_ref)가 같이 쓴다. 지도와 지도 자세만 본다.

// 이 판의 그래프 인자: 노드 격자 위상 = 판 시작 칸 (로봇이 노드 위에서 시작)
DEV mdp::graphgen::Params graph_params(const EnvCfg& c, const MapDev& m, const EnvState& e) {
  mdp::graphgen::Params p;
  p.step = c.graph_step;
  int sx, sy;
  cell_of(c, m, e.x0, e.y0, &sx, &sy);
  p.ox = ((sx % p.step) + p.step) % p.step;
  p.oy = ((sy % p.step) + p.step) % p.step;
  p.gw = m.gw;
  p.gh = m.gh;
  p.util_r2 = c.graph_util_r * c.graph_util_r;
  p.min_util = c.graph_min_util;
  p.fvox = c.graph_fvox;
  p.util_norm = c.graph_util_norm;
  p.keep_vis = c.graph_view;
  p.cover = c.graph_cover;
  p.bx0 = e.kb_x0; p.by0 = e.kb_y0; p.bx1 = e.kb_x1; p.by1 = e.kb_y1;
  return p;
}

// 전역 숫자 [GLOB] — 논문에 없는 우리 추가 (신경망은 기본으로 읽지 않는다 — network/README graph_glob 0). 지도·지도 자세에서 나오는 값만:
//   0 지난 시간 / 안전 시간 · 1 진행 없음 시계 / kIdleTime (자름 1) · 2 아는 넓이 / 지도 넓이 · 3 지도 닫힘 0/1
//   4·5 로봇 위치 − 현재 노드 위치 (지도 좌표 x·y, / 노드 간격) · 6·7 지도 자세 방향 sin·cos · 8 ~ 15 = 0 (비워 둠)
DEV void graph_glob(const EnvBatch& b, const EnvState& e, int env, const mdp::graphgen::Params& p, int cur) {
  const EnvCfg& c = b.cfg;
  const MapDev& m = b.maps[e.map];
  float* G = b.gglob + (size_t)env * mdp::graph::GLOB;
  for (int i = 0; i < mdp::graph::GLOB; ++i) G[i] = 0.f;
  const float dt_act = (float)c.substeps * c.dt;
  const float node_m = (float)p.step * c.res;
  G[0] = (float)e.step * dt_act / mdp::cur::kSafeTime;
  G[1] = dm::minf(1.f, (float)e.since_occ * dt_act / mdp::cur::kIdleTime);
  G[2] = (float)e.known_free / (float)(c.size * c.size);
  G[3] = e.map_closed ? 1.f : 0.f;
  const float nx = m.ox + ((float)mdp::graphgen::node_x(p, cur) + 0.5f) * c.res, ny = m.oy + ((float)mdp::graphgen::node_y(p, cur) + 0.5f) * c.res;
  G[4] = ((e.x + e.ex) - nx) / node_m;
  G[5] = ((e.y + e.ey) - ny) / node_m;
  float s, co;
  dm::sincosf_d(e.th + e.eth, &s, &co);
  G[6] = s;
  G[7] = co;
}

// ---------- 결정 하나 = 이웃 노드 하나로 이동 (2026-11-11, 전작 worker.py 의 "한 걸음 = 고른 이웃 노드로 감") ----------
// 묶음(결정)의 걸음 수가 환경마다 다르다: 고른 노드에 닿거나 실패·시간 초과가 날 때까지 걷고, 끝난 환경은 묶음 끝까지 쉰다.
// 스레드 0 · 묶음 첫 걸음의 행동 적용 앞 (경로 계획 뒤 — e.goal_cell 이 고른 칸)
DEV void gdec_begin(const EnvBatch& b, const EnvState& e, int env) {
  if (!b.gdec) return;
  const EnvCfg& c = b.cfg;
  const MapDev& m = b.maps[e.map];
  GraphDec& d = b.gdec[env];
  const mdp::graphgen::Params p = graph_params(c, m, e);
  d.steps = 0;
  d.reason = kGdRun;
  d.target = e.goal_cell >= 0 ? mdp::graphgen::node_at(p, e.goal_cell % m.gw, e.goal_cell / m.gw) : -1;
  if (e.step == 0) d.n_dec = 0;
  d.n_util = -1;
  d.n_gone = 0;
  d.closed = 0;
  d.term = 0;
  d.dist = 0.f;
  d.px = e.x;
  d.py = e.y;
  d.r_dist = d.r_front = d.r_done = 0.f;
}
// 이 걸음(sub)을 쉬나 — 앞 걸음에서 결정이 끝남 (끝난 그 걸음은 끝까지 돈다: steps = 끝난 걸음 + 1)
DEV bool gdec_idle(const EnvBatch& b, int env, int sub) {
  if (b.lowcmd && b.low[env].hold > 0 && b.low[env].hold <= sub) return true;   // 아래층 단독 과제: 목표 통과 반경에 들어 결정이 앞 걸음에서 끝남 (low_task.h)
  return b.gdec && sub > 0 && b.gdec[env].reason != kGdRun && b.gdec[env].steps <= sub;
}
// 스레드 0 · 걸음 끝(finish_step 뒤, 판 끝 리셋 앞). pass(x, y) = 그래프가 지나는 칸 (env_graph_kernel·graph_ref 와 같은 함수)
template <class P>
DEV void gdec_step(const EnvBatch& b, const EnvState& e, int env, int sub, P pass) {
  if (!b.gdec || sub < 0) return;
  const EnvCfg& c = b.cfg;
  const MapDev& m = b.maps[e.map];
  GraphDec& d = b.gdec[env];
  const float mx = e.x - d.px, my = e.y - d.py;
  d.dist += sqrtf(mx * mx + my * my);
  d.px = e.x;
  d.py = e.y;
  d.steps = sub + 1;
  if (e.done) { d.reason = kGdEpisode; return; }
  const bool fixed = c.graph_fixed != 0;   // 고정 주기(1 Hz 재계획): 닿아도 주기 끝까지 가고 주기 끝에서만 사유를 정함
  if (fixed && sub + 1 < c.decide_steps) return;
  const float px = e.x + e.ex, py = e.y + e.ey, tol2 = c.graph_arrive * c.graph_arrive;
  if (d.target >= 0) {
    const mdp::graphgen::Params p = graph_params(c, m, e);
    const float nx = m.ox + ((float)mdp::graphgen::node_x(p, d.target) + 0.5f) * c.res - px, ny = m.oy + ((float)mdp::graphgen::node_y(p, d.target) + 0.5f) * c.res - py;
    bool at_end = false;   // 경로 끝에 섬 (추종기 도착 — hier_path.h path_follow 의 d_goal ≤ goal_tol. 잘린 경로의 끝도 — 다음 결정이 다시 짬)
    if (e.p_n > 0) {
      const float ex = m.ox + ((float)e.wp[2 * (e.p_n - 1)] + 0.5f) * c.res - px, ey = m.oy + ((float)e.wp[2 * (e.p_n - 1) + 1] + 0.5f) * c.res - py;
      at_end = ex * ex + ey * ey <= c.goal_tol * c.goal_tol;
    }
    if (nx * nx + ny * ny <= tol2 || at_end) {
      int rx, ry, forced;
      cell_of(c, m, px, py, &rx, &ry);
      const int cur = mdp::graphgen::find_cur(p, rx, ry, [&](int i) { return mdp::graphgen::node_raw(p, i, pass); }, pass, &forced);
      if (cur == d.target && !forced) { d.reason = kGdArrive; return; }
      if (at_end) { d.reason = kGdOff; return; }
    }
  }
  if (c.rec_on && e.p_blk_now * 2 > c.decide_steps) { d.reason = kGdBlocked; return; }
  if (sub + 1 >= c.decide_steps && (d.target < 0 || e.p_n == 0)) { d.reason = kGdNoPath; return; }
  if (fixed || sub + 1 >= c.graph_max_steps) d.reason = kGdTimeout;   // 고정 주기에선 "아직 가는 중" 도 이 사유
}

DEV void episode_end(const EnvBatch& b, EnvState& e, int env, float cov, float dt_act);   // env.h (판 끝 기록·커리큘럼)

// 스레드 0 · 묶음 끝 (걸음들 뒤, 끝난 판 리셋 앞). n_util: 결정 끝 관측의 유틸리티 노드 수 (판이 걸음에서 끝났으면 그래프를 안 만들고 −1),
// n_gone: 사라진 프론티어 복셀 수. 전작 방식(graph_term)이면 판 끝(유틸리티 합 0 → 성공 — worker.py:56-58)과 보상(env.py calculate_reward)을 여기서.
// 판을 끝내면 true (호출자가 b.done 을 씀 — 리셋은 묶음 끝 커널·CPU env_step_cpu)
DEV bool graph_finish(const EnvBatch& b, EnvState& e, int env, int n_util, int n_gone) {
  const EnvCfg& c = b.cfg;
  GraphDec& d = b.gdec[env];
  d.n_dec += 1;
  d.n_util = n_util;
  d.n_gone = n_gone;
  d.closed = e.map_closed;
  d.term = e.done ? 2 : 0;
  if (!c.graph_term) return false;
  bool end = false;
  // 남은 프론티어 맡기기(graph_cover, 우리 수정)를 켜면 "유틸리티 합 0" 은 지도가 닫혔을 때만 판을 끝낸다 — 그래프가 잠깐 없을 때(현재 노드가 외톨이 · 닿는 칸 요약이 시작 칸을 못 잡음)
  // 유틸리티 노드가 0 이 되어 판이 일찍 끝나는 것을 막음 (전작엔 없는 경우: 격자 세계라 로봇이 늘 노드 위)
  if (!e.done && n_util == 0 && (!c.graph_cover || e.map_closed)) {
    d.term = 1;
    e.done = mdp::cur::kDeclareOk;
    end = true;
  } else if (!e.done && c.graph_max_dec > 0 && d.n_dec >= c.graph_max_dec) {
    d.term = 3;
    e.done = mdp::cur::kTruncated;
    end = true;
  }
  if (end) episode_end(b, e, env, e.coverage, (float)c.substeps * c.dt);
  // 보상: −이동 거리/사거리 + 사라진 프론티어/정규화 + 끝 보너스. 판이 걸음에서 끝났으면(충돌·끊김) 거리 항만 — 전작엔 충돌이 없어 벌점도 없다
  d.r_dist = -d.dist * c.graph_rew_dist;
  d.r_front = (float)n_gone / (float)c.graph_util_norm;
  d.r_done = d.term == 1 ? c.graph_rew_done : 0.f;
  b.racc[env] = (d.r_dist + d.r_front) + d.r_done;
  return end;
}
