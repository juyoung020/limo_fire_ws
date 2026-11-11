// 그래프 모드 (mdp/graph.h) 의 시뮬 쪽 풀 — env.h 가 namespace sim 안에서 포함한다 (EnvBatch·EnvState·cell_of 뒤).
// GPU 커널(src/env_kernel.cu env_graph_kernel)·CPU 참조·실물 노드(include/obs_ref.h graph_ref)가 같이 쓴다. 지도와 지도 자세만 본다.

// 이 판의 그래프 인자: 노드 격자 위상 = 판 시작 칸 (로봇이 노드 위에서 시작)
DEV mdp::graphgen::Params graph_params(const EnvCfg& c, const MapDev& m, const EnvState& e) {
  mdp::graphgen::Params p;
  p.step = (int)(mdp::graph::NODE_RES * c.inv_res + 0.5f);
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
  p.bx0 = e.kb_x0; p.by0 = e.kb_y0; p.bx1 = e.kb_x1; p.by1 = e.kb_y1;
  return p;
}

// 전역 숫자 [GLOB] — 논문에 없는 우리 추가 (최소한, 모두 0 이어도 정책 구조는 돈다). 지도·지도 자세에서 나오는 값만:
//   0 지난 시간 / 안전 시간 · 1 진행 없음 시계 / kIdleTime (자름 1) · 2 아는 넓이 / 지도 넓이 · 3 지도 닫힘 0/1
//   4·5 로봇 위치 − 현재 노드 위치 (지도 좌표 x·y, / NODE_RES) · 6·7 지도 자세 방향 sin·cos · 8 ~ 15 = 0 (비워 둠)
DEV void graph_glob(const EnvBatch& b, const EnvState& e, int env, const mdp::graphgen::Params& p, int cur) {
  const EnvCfg& c = b.cfg;
  const MapDev& m = b.maps[e.map];
  float* G = b.gglob + (size_t)env * mdp::graph::GLOB;
  for (int i = 0; i < mdp::graph::GLOB; ++i) G[i] = 0.f;
  const float dt_act = (float)c.substeps * c.dt;
  G[0] = (float)e.step * dt_act / mdp::cur::kSafeTime;
  G[1] = dm::minf(1.f, (float)e.since_occ * dt_act / mdp::cur::kIdleTime);
  G[2] = (float)e.known_free / (float)(c.size * c.size);
  G[3] = e.map_closed ? 1.f : 0.f;
  const float nx = m.ox + ((float)mdp::graphgen::node_x(p, cur) + 0.5f) * c.res, ny = m.oy + ((float)mdp::graphgen::node_y(p, cur) + 0.5f) * c.res;
  G[4] = ((e.x + e.ex) - nx) / mdp::graph::NODE_RES;
  G[5] = ((e.y + e.ey) - ny) / mdp::graph::NODE_RES;
  float s, co;
  dm::sincosf_d(e.th + e.eth, &s, &co);
  G[6] = s;
  G[7] = co;
}
