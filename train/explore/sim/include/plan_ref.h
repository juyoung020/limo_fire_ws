// 계층형·아래층 경로 계획의 CPU 판 (sim/src/env_ref.cpp 참조판 · 실물 노드 limo_fire_rl RobotEnv 가 같은 함수 — 2026-11-16 env_ref.cpp 에서 옮김, 식 그대로).
// GPU env_plan_kernel 과 같은 규칙·같은 경로. obs_ref.h 뒤에 포함 (hier_path.h 함수들을 씀)
#pragma once
#include <algorithm>
#include <vector>

#include "obs_ref.h"

namespace sim {

// 계층형 결정 (GPU env_plan_kernel 과 같은 규칙·같은 경로): 엄격(지도 막힘 ∪ 약한 막힘) → 못 닿으면 느슨(지도 막힘만).
// 거리 배열 위 Dial 최단 경로(무게 hier_path.h) → 거리 = 내 거리 − 내 무게 인 첫 nb4 이웃으로 뒤로 짚기
inline void plan_ref(const EnvBatch& b, EnvState& e, int env, int k) {
  const EnvCfg& c = b.cfg;
  const MapDev& m = b.maps[e.map];
  const int16_t* L = b.lmap + (size_t)env * c.size * c.size;
  const mdp::MapView mv{L, c.size, m.gw, m.gh};
  const mdp::Robot rb = path_rb_plan(c, mdp::obsgen::reach_robot(c)), rm = path_rb_mid(c), ro = path_rb_out(c);   // rb: 계획 반경 (path_consistent 면 충돌 검사 발자국 반경, hier_path.h)
  int force = 0;   // 복구 (hier_path.h path_recovery): 다시 계획 · 포기한 목표면 다른 후보(없으면 −1)
  k = path_recovery(b, e, env, k, &force);
  e.p_slot = k;
  const int tgt = k >= 0 ? act_cells(b, env)[k] : -1;
  if (tgt < 0) {
    e.goal_cell = -1;
    path_set_none(e);
    return;
  }
  if (e.kb_x1 < 0) {
    e.goal_cell = tgt;
    path_set_none(e);
    return;
  }
  const size_t n = (size_t)m.gw * m.gh;
  thread_local std::vector<uint8_t> lt, flag;
  thread_local std::vector<int32_t> d2l, d2o, dist, tent, qa;
  if (lt.size() < n) { lt.resize(n); flag.resize(n); d2l.resize(n); d2o.resize(n); dist.resize(n); tent.resize(n); qa.resize(n); }
  int rcx, rcy;
  cell_of(c, m, e.x + e.ex, e.y + e.ey, &rcx, &rcy);
  const int x0 = e.kb_x0, x1 = e.kb_x1, y0 = e.kb_y0, y1 = e.kb_y1, R = ro.r;
  const float rr = c.path_obs_range * c.path_obs_range;
  // 막힘 칸: 지도 막힘(occ)·엄격 막힘(lt = occ ∪ 로그 오즈 > 0 이고 로봇 칸에서 obstacle_max_range 안 — path_lethal 과 같은 규칙)
  const int ex0 = std::max(0, x0 - R), ex1 = std::min(m.gw - 1, x1 + R), ey0 = std::max(0, y0 - R), ey1 = std::min(m.gh - 1, y1 + R);
  for (int y = ey0; y <= ey1; ++y)
    for (int x = ex0; x <= ex1; ++x) {
      const int16_t l = L[(size_t)y * c.size + x];
      uint8_t v = mdp::cell_class(l) == mdp::kOcc ? 3 : 0;   // bit0 occ · bit1 lt · bit2 지금 스캔 (아래) — 느슨 막힘 = bit0 · bit2
      if (!v && l != mdp::kUnknownL && l > 0) {
        const float dx = (float)(x - rcx) * c.res, dy = (float)(y - rcy) * c.res;
        if (dx * dx + dy * dy <= rr) v = 2;
      }
      lt[(size_t)y * m.gw + x] = v;
    }
  // 지금 스캔 끝점 (hier_path.h "지금 스캔" (a)): bit2 — 엄격·느슨 둘 다 막힘 (GPU LT·LO 판)
  if (c.path_scan)
    for (int k = 0; k < mdp::obs::LIDAR_BINS; ++k) {
      int x, y;
      if (!path_scan_cell(b, e, env, k, &x, &y)) continue;
      if (x < ex0 || x > ex1 || y < ey0 || y > ey1) continue;
      lt[(size_t)y * m.gw + x] |= 4;
    }
  // 깊이 카메라 장애물 층 (hier_path.h "깊이 카메라 장애물 층"): 지금 스캔과 같은 몫 — 엄격·느슨 둘 다 막힘 (GPU LT·LO 판)
  if (b.cvox)
    for (int y = ey0; y <= ey1; ++y)
      for (int x = ex0; x <= ex1; ++x)
        if (cam_lethal(b, env, x, y)) lt[(size_t)y * m.gw + x] |= 4;
  // 막힘까지 가장 가까운 거리² (칸, 반경 ro 안만) — 범위 안 칸마다 occ·lt 따로. mdp::standable 과 같은 원판 규칙
  for (int y = y0; y <= y1; ++y)
    for (int x = x0; x <= x1; ++x) d2l[(size_t)y * m.gw + x] = d2o[(size_t)y * m.gw + x] = 1 << 30;
  for (int y = ey0; y <= ey1; ++y)
    for (int x = ex0; x <= ex1; ++x) {
      const uint8_t v = lt[(size_t)y * m.gw + x];
      if (!v) continue;
      for (int dy = -R; dy <= R; ++dy)
        for (int dx = -R; dx <= R; ++dx) {
          const int u = x + dx, w = y + dy;
          if (u < x0 || w < y0 || u > x1 || w > y1) continue;
          const int d2 = dx * dx + dy * dy;
          int32_t& ql = d2l[(size_t)w * m.gw + u];
          ql = std::min(ql, d2);
          if (v & 5) {
            int32_t& qo = d2o[(size_t)w * m.gw + u];
            qo = std::min(qo, d2);
          }
        }
    }
  const auto inbox = [&](int x, int y) { return x >= x0 && y >= y0 && x <= x1 && y <= y1; };
  const auto known = [&](int x, int y) { return mdp::cls_at(mv, x, y) != mdp::kUnk; };
  // mode 1 엄격 · 2 느슨: 설 수 있음, 무게 (GPU: SST·ST 판과 stand_word(LT·OCC) 무게 경계)
  const mdp::Robot rb3 = mdp::obsgen::reach_robot(c);
  const auto pass = [&](int mode, int x, int y) -> bool {
    if (!inbox(x, y) || !known(x, y)) return false;
    const size_t i = (size_t)y * m.gw + x;
    // mode 3: 느슨 막힘 + 관측 BFS 반경 (계획 반경으로 못 짤 때 — hier_path.h "계획·충돌 검사 일치" (e))
    return mode == 1 ? (!lt[i] && d2l[i] > rb.r2) : (!(lt[i] & 5) && d2o[i] > (mode == 3 ? rb3.r2 : rb.r2));
  };
  const auto weight = [&](int mode, int x, int y) -> int {
    if (!pass(mode, x, y)) return 0;
    const size_t i = (size_t)y * m.gw + x;
    const int32_t d = mode == 1 ? d2l[i] : d2o[i];
    return path_weight(true, d > rm.r2, d > ro.r2);
  };
  // 같은 후보 · 경로 그대로?
  const int pm = e.p_mode == 1 ? 1 : 2;
  if (!force && act_same(b, e, env, k) && e.goal_cell >= 0 && e.p_n > 0 && !e.p_trunc && path_clear(e, [&](int x, int y) { return pass(pm, x, y); }))
    return;
  e.goal_cell = tgt;
  path_set_none(e);
  e.p_plans += 1;
  // 한 번 계획 (mode): 시작 = 로봇 칸에서 FIFO 4 이웃 BFS 로 처음 만나는 설 수 있는 칸 (frontier_search nearestCell 과 같은 차례), Dial
  int start = -1, goal = tgt;
  // 목표 칸들 (hier_path.h "목표 허용"): 가까운 목표 = 목표가 설 수 있으면 목표 하나, 아니면 목표에서 path_goal_tol(Smac tolerance) 안 설 수 있는 칸 —
  // 처음 닿는 층에서 멈춤. 못 닿고 BFS 가 끝나면 목표에서 path_goal_far 안 닿은 칸 중 목표에 가장 가까운 칸 (Smac 이 tolerance 안 가장 가까운 노드를 내는 꼴)
  const mdp::Robot rt = path_rb_tol(c), rf = path_rb_far(c);
  const int gx = tgt % m.gw, gy = tgt / m.gw;
  const auto in_goal = [&](int mode, int x, int y) -> bool {
    if (x == gx && y == gy) return pass(mode, x, y);
    if (pass(mode, gx, gy) || rt.r2 <= 0) return false;
    return (x - gx) * (x - gx) + (y - gy) * (y - gy) <= rt.r2 && pass(mode, x, y);
  };
  const auto any_goal = [&](int mode) -> bool {
    for (int dy = -rf.r; dy <= rf.r; ++dy)
      for (int dx = -rf.r; dx <= rf.r; ++dx)
        if (dx * dx + dy * dy <= rf.r2 && pass(mode, gx + dx, gy + dy)) return true;
    return pass(mode, gx, gy);
  };
  // BFS 가 끝난 뒤: 먼 허용 안 닿은 칸 중 (거리², 칸) 최소 (GPU 와 같은 규칙)
  const auto far_goal = [&](int mode) -> bool {
    int64_t best = -1;
    for (int dy = -rf.r; dy <= rf.r; ++dy)
      for (int dx = -rf.r; dx <= rf.r; ++dx) {
        const int x = gx + dx, y = gy + dy;
        if (dx * dx + dy * dy > rf.r2 || !pass(mode, x, y) || dist[(size_t)y * m.gw + x] < 0) continue;
        const int64_t key = ((int64_t)(dx * dx + dy * dy) << 20) | (int64_t)(y * m.gw + x);
        if (best < 0 || key < best) { best = key; goal = y * m.gw + x; }
      }
    return best >= 0;
  };
  const auto run = [&](int mode) -> bool {
    start = -1;
    if (rcx < 0 || rcy < 0 || rcx >= m.gw || rcy >= m.gh) return false;
    std::fill(flag.begin(), flag.begin() + n, 0);
    int head = 0, tail = 0;
    qa[tail++] = rcy * m.gw + rcx;
    flag[rcy * m.gw + rcx] = 1;
    while (head < tail) {
      const int idx = qa[head++], x = idx % m.gw, y = idx / m.gw;
      if (pass(mode, x, y)) { start = idx; break; }
      for (int q = 0; q < 4; ++q) {
        int dx, dy;
        mdp::nb4(q, &dx, &dy);
        const int u = x + dx, v = y + dy;
        if (u < 0 || v < 0 || u >= m.gw || v >= m.gh) continue;
        const int j = v * m.gw + u;
        if (flag[j]) continue;
        flag[j] = 1;
        qa[tail++] = j;
      }
    }
    if (start < 0) return false;
    for (int y = y0; y <= y1; ++y)
      for (int x = x0; x <= x1; ++x) { dist[(size_t)y * m.gw + x] = -1; tent[(size_t)y * m.gw + x] = 1 << 30; }
    thread_local std::vector<int32_t> bk[4], cur;   // 층 통 (용량은 걸음 사이에 남김 — 할당 없음)
    for (auto& q : bk) q.clear();
    bk[0].push_back(start);
    tent[start] = 0;
    for (int t = 0;; ++t) {
      if (bk[0].empty() && bk[1].empty() && bk[2].empty() && bk[3].empty()) return far_goal(mode);
      cur.clear();
      cur.swap(bk[t & 3]);
      int64_t best = -1;   // 층 t 에서 정해진 목표 칸 중 (목표까지 거리², 칸) 가장 작은 것 — GPU 와 같은 규칙
      for (int v : cur) {
        if (dist[v] >= 0 || tent[v] != t) continue;
        dist[v] = t;
        if (v == tgt) { best = 0; goal = tgt; }
        else if (best != 0 && in_goal(mode, v % m.gw, v / m.gw)) {
          const int ddx = v % m.gw - gx, ddy = v / m.gw - gy;
          const int64_t key = ((int64_t)(ddx * ddx + ddy * ddy) << 20) | v;
          if (best < 0 || key < best) { best = key; goal = v; }
        }
        const int x = v % m.gw, y = v / m.gw;
        for (int q = 0; q < 4; ++q) {
          int dx, dy;
          mdp::nb4(q, &dx, &dy);
          const int w = weight(mode, x + dx, y + dy);
          if (w <= 0) continue;
          const int u = (y + dy) * m.gw + x + dx;
          if (dist[u] >= 0 || t + w >= tent[u]) continue;
          tent[u] = t + w;
          bk[(t + w) & 3].push_back(u);
        }
      }
      if (best >= 0) return true;
    }
  };
  int mode = 0;
  if (any_goal(1) && run(1)) mode = 1;
  else if ((c.path_fallback & 1) && run(2)) mode = 2;
  else if ((c.path_fallback & 2) && c.path_consistent && rb3.r2 != rb.r2 && run(3)) mode = 3;
  if (!mode) return;   // 설 수 있는 칸이 없거나 못 닿음: 목표만 두고 섬
  const auto wfn = [&](int x, int y) { return weight(mode, x, y); };
  thread_local std::vector<int32_t> list;
  list.clear();
  int v = goal;
  list.push_back(v);
  while (v != start) {
    const int x = v % m.gw, y = v / m.gw, want = dist[v] - wfn(x, y);
    int u = -1;
    for (int q = 0; q < 4 && u < 0; ++q) {
      int dx, dy;
      mdp::nb4(q, &dx, &dy);
      if (wfn(x + dx, y + dy) > 0 && dist[(y + dy) * m.gw + x + dx] == want) u = (y + dy) * m.gw + x + dx;
    }
    if (u < 0 || (int)list.size() >= mdp::kReachCap) return;
    v = u;
    list.push_back(v);
  }
  std::reverse(list.begin(), list.end());
  int trunc = 0;
  e.p_n = path_simplify(list.data(), (int)list.size(), m.gw, wfn, e.wp, mdp::hier::kPathMax, &trunc);
  e.p_i = 0;
  e.p_trunc = trunc;
  e.p_mode = mode;
  if (mode >= 2) e.p_fb += 1;
}

}  // namespace sim
