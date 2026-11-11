// 관측 만들기 CPU 참조판 (호스트 전용, CUDA 없음): 프론티어 요약·계층형 후보 → 관측(write_obs), 깊이 카메라 장애물 층 갱신(camlayer_ref).
// 시뮬 CPU 참조판(src/env_ref.cpp)과 실물 노드(src/limo_fire/limo_fire_rl)가 이 함수들을 그대로 같이 부른다 — 실물 관측이 학습 관측과 갈라지지 않게
// 한 벌만 둔다. GPU 판(env_kernel.cu)은 같은 단계 함수로 같은 값을 낸다(tools/env_verify).
#pragma once
#include <cstdint>
#include <vector>

#include "env.h"

namespace sim {

// 도달 가능한 프론티어 요약 (mdp/frontier.h reach_summary = 기준선 frontier_search 그대로). GPU 는 env_kernel.cu reach_block — 같은 값.
// 계층형이면 같은 찾기에서 후보(mdp::cand_pick)·주변 프론티어 칸 수(mdp::cand_util)도 → 관측 후보 칸 (GPU env_reach_kernel 과 같은 값)
inline void reach_obs(const EnvBatch& b, const EnvState& e, int env, Shared& sh) {
  const EnvCfg& c = b.cfg;
  const MapDev& m = b.maps[e.map];
  mdp::reach_clear(&sh.reach);
  mdp::CandSummary cs;
  mdp::cand_clear(&cs);
  if (e.kb_x1 >= 0) {   // 아는 칸이 있을 때
    const int16_t* L = b.lmap + (size_t)env * c.size * c.size;
    const mdp::MapView mv{L, c.size, m.gw, m.gh};
    int n_tot = 0;   // 지도 전체 프론티어 칸 (모두 건드린 범위 안)
    for (int y = e.kb_y0; y <= e.kb_y1; ++y)
      for (int x = e.kb_x0; x <= e.kb_x1; ++x) n_tot += mdp::frontier_cell(mv, x, y) ? 1 : 0;
    thread_local std::vector<uint8_t> flag;
    thread_local std::vector<int32_t> qa, qb, dist, label;
    thread_local std::vector<mdp::Frontier> out;
    const size_t n = (size_t)m.gw * m.gh;
    if (flag.size() < n) { flag.resize(n); qa.resize(n); qb.resize(n); dist.resize(n); label.resize(n); }
    if (out.size() < (size_t)mdp::kReachCap + 1) out.resize(mdp::kReachCap + 1);
    int rx, ry;
    cell_of(c, m, e.x + e.ex, e.y + e.ey, &rx, &ry);
    mdp::SearchBuf sb{flag.data(), qa.data(), qb.data(), dist.data(), b.cand ? label.data() : nullptr};
    {   // 닫힘 규칙 (GPU reach_block 과 같은 값): 지나간 칸 씨앗·본 칸·덩어리 문턱·자격 반경
      const int ws = fmap_row(c), nf = c.size * ws;
      sb.vis = b.trail + (size_t)env * 2 * nf;
      sb.look = sb.vis + nf;
      sb.tws = ws;
      sb.close_min = c.close_min;
      sb.rbc = mdp::robot_cells(c.close_radius, c.res);
      sb.cam = b.cvox ? b.cvox + (size_t)env * c.size * c.size : nullptr;   // 깊이 카메라 층 (계층형 cam_on)
    }
    // reach_summary 와 같음: 문턱 1 로 모든 덩어리를 out 에, 문턱은 고를 때
    const mdp::SearchResult r = mdp::frontier_search(mv, rx, ry, mdp::obsgen::reach_robot(c), 1, sb, out.data(), (int)out.size());
    const int n_out = r.n_found < (int)out.size() ? r.n_found : (int)out.size();
    mdp::reach_pick(r, out.data(), n_out, mdp::obsgen::reach_min_cells(c), n_tot, &sh.reach);
    if (b.gfront) {   // 그래프 모드: 닫힘을 막는 칸 비트 판 (GPU reach_block 과 같은 값 — 넘침·시작 칸 없음이면 비움)
      const int gws = fmap_row(c);
      uint32_t* GF = b.gfront + (size_t)env * c.size * gws;
      for (int y = e.kb_y0; y <= e.kb_y1; ++y)
        for (int wx = e.kb_x0 >> 5; wx <= (e.kb_x1 >> 5); ++wx) GF[y * gws + wx] = 0u;
      if (n_tot <= mdp::kReachCap && r.start_free)
        for (int i = 0; i < r.n_reach; ++i) {
          const int idx = qa[i], x = idx % m.gw, y = idx / m.gw;
          if (!(flag[idx] & 2) || !mdp::frontier_cell(mv, x, y) || out[label[idx]].n < c.close_min) continue;
          if (!mdp::looked_cell(mv, sb.look, sb.vis, sb.tws, x, y)) GF[y * gws + (x >> 5)] |= 1u << (x & 31);
        }
    }
    if (b.cand) {
      thread_local std::vector<uint64_t> key2;
      if (key2.size() < out.size()) key2.resize(out.size());
      mdp::cand_keys(mv, mdp::obsgen::reach_robot(c), r, sb, out.data(), n_out, key2.data());
      mdp::cand_pick(out.data(), key2.data(), n_out, n_tot, e.goal_cell, flag.data(), label.data(), &cs);
      const int R = mdp::cand_util_radius(c.res);
      for (int k = 0; k < cs.n; ++k) cs.c[k].util = mdp::cand_util(mv, cs.c[k].cell % m.gw, cs.c[k].cell / m.gw, R);
    }
  }
  if (b.cand) mdp::obsgen::obs_cand(b, e, env, cs);
}

// depth = false: 깊이 칸을 안 그림 (계층형 묶음 가운데 걸음 — GPU 도 묶음 끝에만 그림, 깊이는 상태만의 함수라 끝 값이 같다)
inline void write_obs(const EnvBatch& b, EnvState& e, int env, Shared& sh, const mdp::obsgen::FarDirty& fd, bool depth = true) {
  trail_mark(b, e, env, 0, 1);   // 닫힘 규칙의 지나간 길 (GPU env_reach_kernel 처음과 같은 때)
  reach_obs(b, e, env, sh);
  e.map_closed = sh.reach.closed;   // 다음 걸음 finish_step 성공 판정 (GPU env_reach_kernel 과 같음)
  for (int tid = 0; tid < kThreads; ++tid) mdp::obsgen::obs_cells(b, e, env, sh, tid, fd);
  if (depth)
    for (int i = 0; i < mdp::obs::DEPTH_SIZE; ++i) mdp::obsgen::obs_depth(b, e, env, i);
  mdp::obsgen::obs_finish(b, e, env, sh);
}

// 깊이 카메라 장애물 층 갱신 (GPU env_camlayer_kernel 과 같은 차례: 모든 픽셀 지우기 → 모든 픽셀 표시 → 몸통 밑 지우기 — 끄기·켜기라 픽셀 차례와 무관)
inline void camlayer_ref(const EnvBatch& b, const EnvState& e, int env) {
  if (!b.cvox) return;
  uint8_t* V = b.cvox + (size_t)env * b.cfg.size * b.cfg.size;
  for (int i = 0; i < mdp::obs::DEPTH_SIZE; ++i) cam_pixel(b, e, env, i, 0, [&](int ci, uint8_t bit) { V[ci] &= (uint8_t)~bit; });
  for (int i = 0; i < mdp::obs::DEPTH_SIZE; ++i) cam_pixel(b, e, env, i, 1, [&](int ci, uint8_t bit) { V[ci] |= bit; });
  for (int col = 0; col < mdp::obs::DEPTH_W; ++col) cam_column(b, e, env, col);
  cam_compact(b, env);
  const int nfp = cam_fp_cells(b.cfg);
  for (int j = 0; j < nfp; ++j) {
    const int ci = cam_fp_cell(b, e, j);
    if (ci >= 0) V[ci] = 0;
  }
}

// 그래프 모드 관측 (mdp/graph.h graph_build) — graph_on 일 때만. 지도(lmap)와 지도 자세만 쓴다: 시뮬 CPU 참조와 실물 노드가 같은 함수.
// 부르는 때 = 결정 관측(묶음 끝·판 시작) 뒤. 판 첫 관측(e.step == 0)이 방문 표시를 비운다.
// finish = true: 묶음 끝의 안 끝난 환경 — 결정 결과·전작 방식 판 끝·보상 (graph_env.h graph_finish), 판을 끝냈으면 true (GPU env_graph_kernel gate 1 과 같음)
inline bool graph_ref(const EnvBatch& b, EnvState& e, int env, bool finish = false) {
  if (!b.gfeat) return false;
  const EnvCfg& c = b.cfg;
  const MapDev& m = b.maps[e.map];
  const mdp::MapView mv{b.lmap + (size_t)env * c.size * c.size, c.size, m.gw, m.gh};
  const mdp::Robot rb = mdp::obsgen::reach_robot(c);
  const mdp::graphgen::Params p = graph_params(c, m, e);
  const auto in_map = [&](int x, int y) { return x >= 0 && y >= 0 && x < m.gw && y < m.gh; };
  const auto free = [&](int x, int y) { return in_map(x, y) && mdp::cls_at(mv, x, y) == mdp::kFree; };
  const auto pass = [&](int x, int y) { return c.graph_stand ? (in_map(x, y) && mdp::standable(mv, x, y, rb)) : free(x, y); };
  const int tws = fmap_row(c);
  const uint32_t* gf = b.gfront + (size_t)env * c.size * tws;   // 닫힘을 막는 칸 (reach_obs 가 이번 관측에 씀)
  const auto front = [&](int x, int y) {
    if (!c.graph_open) return mdp::frontier_cell(mv, x, y);
    return x >= e.kb_x0 && x <= e.kb_x1 && y >= e.kb_y0 && y <= e.kb_y1 && mdp::plane_bit(gf, tws, x, y);
  };
  uint32_t* vis = b.gvis + (size_t)env * (mdp::graph::N / 32);
  if (e.step == 0)
    for (int i = 0; i < mdp::graph::N / 32; ++i) vis[i] = 0u;
  int rx, ry;
  cell_of(c, m, e.x + e.ex, e.y + e.ey, &rx, &ry);
  int n_gone = 0, n_util = 0;
  const int cur = mdp::graphgen::graph_build(p, rx, ry, pass, free, front, vis, b.gfeat + (size_t)env * mdp::graph::N * mdp::graph::FEAT,
                                             b.gedge + (size_t)env * mdp::graph::N, b.gcand + (size_t)env * mdp::graph::NB, nullptr,
                                             b.gprev + (size_t)env * mdp::graphgen::kVoxMax, e.step == 0, &n_gone, &n_util);
  b.gcur[env] = cur;
  graph_glob(b, e, env, p, cur);
  if (!finish) return false;
  const bool end = graph_finish(b, e, env, n_util, n_gone);
  if (end) b.done[env] = e.done;
  return end;
}

// 걸음 끝의 결정 끝 판정 (graph_env.h gdec_step) — 지나는 칸은 graph_ref 와 같은 함수
inline void gdec_step_ref(const EnvBatch& b, const EnvState& e, int env, int sub) {
  if (!b.gdec) return;
  const EnvCfg& c = b.cfg;
  const MapDev& m = b.maps[e.map];
  const mdp::MapView mv{b.lmap + (size_t)env * c.size * c.size, c.size, m.gw, m.gh};
  const mdp::Robot rb = mdp::obsgen::reach_robot(c);
  const auto in_map = [&](int x, int y) { return x >= 0 && y >= 0 && x < m.gw && y < m.gh; };
  gdec_step(b, e, env, sub, [&](int x, int y) {
    return in_map(x, y) && (c.graph_stand ? mdp::standable(mv, x, y, rb) : mdp::cls_at(mv, x, y) == mdp::kFree);
  });
}

}  // namespace sim
