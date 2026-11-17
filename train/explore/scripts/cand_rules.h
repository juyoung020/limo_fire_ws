// 정답 지도를 다 아는 순회 (combine_run --gt-bound — 시뮬 전용 진단, 정답 격자를 쓴다). 측정 전용, 호스트(CPU)에서만 돈다.
//   build_gt : 정답 격자에서 설 수 있는 칸 · 닫히려면 알아야 하는 칸 · 타일 닻과 닻마다 거리 표를 맵마다 한 번 만든다
//   gt_tour  : 그 닻으로 탐욕 덮기 + 최근접 이웃·2-opt 순회 — "지도를 미리 다 알면 얼마나 짧게 도나" 의 참고값 (아래)
// 후보 고르기 규칙(cand_gain · cand_tsp · cand_gt …)은 지웠다 — frontier_run --modes cand_* 는 2026-11-17 archive/rl, combine_run --upper cand_tsp 는 2026-11-18.
#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

#include "env_api.h"
#include "frontier.h"
#include "observations.h"

namespace cr {

struct Opt {
  float range_m = 0;     // 순회 시점의 사거리 (m). 0 = 지도 사거리(env.yaml map.max_range)
  float tile_m = 1.5f;   // 타일 변 (m, --gt-tile) — **잠정** (닻 수 ≈ 바닥 넓이 / 타일², 거리 표 메모리로 정함)
  int tile_min = 4;      // 타일의 남은 need 칸이 이 수 이상이면 남은 일 (닫힘 덩어리 문턱 close_min 과 같은 4 칸) — **잠정**
};
// 정답 격자에서 한 번 만드는 표 (맵마다)
struct GtMap {
  int gw = 0, gh = 0;
  std::vector<uint8_t> stand;    // GT 로 닫힘 반경(close_radius)만큼 떨어져 설 수 있고 시작 칸들과 이어진 칸
  std::vector<uint8_t> need;     // 닫히려면 알아야 하는 칸 = stand 칸과 그 4 이웃 (mdp/expert.h "핵심 목표" 와 같은 뜻)
  std::vector<int32_t> anchor;   // 타일 닻 칸 (타일 안 stand 칸 중 타일 가운데에 가장 가까운 칸)
  std::vector<int16_t> own;      // 칸 → 가장 가까운 닻 번호 (닻에서 동시에 번진 BFS, 막힘 무시 — 칸을 타일에 나누는 용도만), 없으면 −1
  std::vector<uint16_t> field;   // [닻][칸] 닻에서 stand 칸 4 이웃 BFS 거리 (칸), 못 닿음 = 0xffff
  int nA() const { return (int)anchor.size(); }
  uint16_t d(int a, int cell) const { return field[(size_t)a * gw * gh + cell]; }
};

inline GtMap build_gt(const sim::World& W, const sim::EnvCfg& c, const Opt& o) {
  GtMap G;
  G.gw = W.gw; G.gh = W.gh;
  const int n = W.gw * W.gh;
  const mdp::Robot rc = mdp::robot_cells(c.close_radius, c.res);
  std::vector<uint8_t> ok(n, 0);
  for (int y = 0; y < W.gh; ++y)
    for (int x = 0; x < W.gw; ++x) {
      if (!(W.grid[y * W.gw + x] & sim::kCellFloor)) continue;
      bool hit = false;
      for (int dy = -rc.r; dy <= rc.r && !hit; ++dy)
        for (int dx = -rc.r; dx <= rc.r && !hit; ++dx) {
          if (dx * dx + dy * dy > rc.r2) continue;
          const int u = x + dx, v = y + dy;
          hit = u >= 0 && v >= 0 && u < W.gw && v < W.gh && (W.grid[v * W.gw + u] & sim::kCellBlocked);
        }
      ok[y * W.gw + x] = !hit;
    }
  const int X4[4] = {-1, 1, 0, 0}, Y4[4] = {0, 0, -1, 1};
  G.stand.assign(n, 0);
  std::vector<int32_t> q;
  for (int32_t s : W.spawn)
    if (s >= 0 && s < n && ok[s] && !G.stand[s]) { G.stand[s] = 1; q.push_back(s); }
  for (size_t h = 0; h < q.size(); ++h) {
    const int x = q[h] % W.gw, y = q[h] / W.gw;
    for (int k = 0; k < 4; ++k) {
      const int u = x + X4[k], v = y + Y4[k];
      if (u < 0 || v < 0 || u >= W.gw || v >= W.gh || !ok[v * W.gw + u] || G.stand[v * W.gw + u]) continue;
      G.stand[v * W.gw + u] = 1;
      q.push_back(v * W.gw + u);
    }
  }
  G.need.assign(n, 0);
  for (int i = 0; i < n; ++i) {
    if (!G.stand[i]) continue;
    G.need[i] = 1;
    const int x = i % W.gw, y = i / W.gw;
    for (int k = 0; k < 4; ++k) {
      const int u = x + X4[k], v = y + Y4[k];
      if (u >= 0 && v >= 0 && u < W.gw && v < W.gh) G.need[v * W.gw + u] = 1;
    }
  }
  const int T = std::max(1, (int)std::lround(o.tile_m / c.res));
  for (int ty = 0; ty * T < W.gh; ++ty)
    for (int tx = 0; tx * T < W.gw; ++tx) {
      int best = -1, bd = 1 << 30;
      for (int y = ty * T; y < std::min(W.gh, ty * T + T); ++y)
        for (int x = tx * T; x < std::min(W.gw, tx * T + T); ++x) {
          if (!G.stand[y * W.gw + x]) continue;
          const int dx = 2 * x + 1 - (2 * tx + 1) * T, dy = 2 * y + 1 - (2 * ty + 1) * T, d2 = dx * dx + dy * dy;
          if (d2 < bd) { bd = d2; best = y * W.gw + x; }
        }
      if (best >= 0) G.anchor.push_back(best);
    }
  const int A = G.nA();
  G.own.assign(n, -1);
  q.clear();
  for (int a = 0; a < A; ++a) { G.own[G.anchor[a]] = (int16_t)a; q.push_back(G.anchor[a]); }
  for (size_t h = 0; h < q.size(); ++h) {
    const int x = q[h] % W.gw, y = q[h] / W.gw;
    for (int k = 0; k < 4; ++k) {
      const int u = x + X4[k], v = y + Y4[k];
      if (u < 0 || v < 0 || u >= W.gw || v >= W.gh || G.own[v * W.gw + u] >= 0) continue;
      G.own[v * W.gw + u] = G.own[q[h]];
      q.push_back(v * W.gw + u);
    }
  }
  G.field.assign((size_t)A * n, 0xffff);
  for (int a = 0; a < A; ++a) {
    uint16_t* F = &G.field[(size_t)a * n];
    q.clear();
    q.push_back(G.anchor[a]);
    F[G.anchor[a]] = 0;
    for (size_t h = 0; h < q.size(); ++h) {
      const int x = q[h] % W.gw, y = q[h] / W.gw;
      for (int k = 0; k < 4; ++k) {
        const int u = x + X4[k], v = y + Y4[k];
        if (u < 0 || v < 0 || u >= W.gw || v >= W.gh) continue;
        const int j = v * W.gw + u;
        if (!G.stand[j] || F[j] != 0xffff) continue;
        F[j] = (uint16_t)std::min(0xfffe, (int)F[q[h]] + 1);
        q.push_back(j);
      }
    }
  }
  return G;
}

// 정답 격자를 다 아는 순회 (combine_run --gt-bound 가 맵마다 한 번 찍음 — "지도를 미리 다 알면 얼마나 짧게 도나" 의 참고값. 온라인 정책이 닿을 수 없는 값이고 엄밀한 최솟값도 아님:
// 실제로 갈 수 있는 길 하나의 길이라 참 최솟값 ≤ 이 값).
//   시점 = 타일 닻. 닻마다 GT 에서 360° · 사거리 R 칸 안에 가림 없이 보이는 need 칸을 세고(맞은 벽 칸까지), 새로 보이는 칸이 가장 많은 닻을 차례로 고름(탐욕 집합 덮기),
//   남은 이득 < tile_min 이면 끝(시작 칸과 이어진 구역 안에서만). 시작 칸마다 고른 닻을 모두 도는 열린 길(최근접 이웃 + 2-opt, GT 최단거리)의 길이를 평균. 220° 시야는 제자리 회전(거리 0)으로 채운다고 봄.
struct GtTour { int n_view = 0; double cover = 0, len_m = 0, turn_rad = 0; };   // turn_rad: 다리 사이 직선 방향이 바뀐 각의 합 (첫 다리 앞 머리 방향은 셈 안 함)
inline GtTour gt_tour(const sim::World& W, const sim::EnvCfg& c, const GtMap& G, const Opt& o) {
  GtTour T;
  const int n = W.gw * W.gh, A = G.nA(), R = (int)((o.range_m > 0 ? o.range_m : c.map_max_range) * c.inv_res), nr = std::max(8, (int)std::ceil(2 * M_PI * R));
  std::vector<std::vector<int32_t>> see(A);
  std::vector<int32_t> stamp(n, -1);
  for (int a = 0; a < A; ++a) {
    const int cx = G.anchor[a] % W.gw, cy = G.anchor[a] / W.gw;
    for (int k = 0; k < nr; ++k) {
      const double ang = 2 * M_PI * (k + 0.5) / nr, ca = std::cos(ang), sa = std::sin(ang);
      for (int t = 0; t <= R; ++t) {
        const int x = cx + (int)std::lround(t * ca), y = cy + (int)std::lround(t * sa);
        if (x < 0 || y < 0 || x >= W.gw || y >= W.gh) break;
        const int idx = y * W.gw + x;
        if (G.need[idx] && stamp[idx] != a) { stamp[idx] = a; see[a].push_back(idx); }
        if (W.grid[idx] & sim::kCellBlocked) break;
      }
    }
  }
  // 시작 칸이 속한 구역(설 수 있는 칸으로 이어진 곳)마다 따로: 그 구역 닻만 시점으로, 그 구역 stand 칸과 4 이웃만 덮을 것으로
  std::vector<std::vector<int>> views(A);   // [구역 대표 닻] 고른 시점
  std::vector<double> covers(A, -1.0);
  const auto solve = [&](int rep) {
    const uint16_t* F = &G.field[(size_t)rep * n];
    std::vector<uint8_t> todo(n, 0);
    long need = 0, got = 0;
    for (int i = 0; i < n; ++i) {
      if (F[i] == 0xffff) continue;
      const int x = i % W.gw, y = i / W.gw, X4[5] = {0, -1, 1, 0, 0}, Y4[5] = {0, 0, 0, -1, 1};
      for (int k = 0; k < 5; ++k) {
        const int u = x + X4[k], v = y + Y4[k];
        if (u >= 0 && v >= 0 && u < W.gw && v < W.gh && !todo[v * W.gw + u]) { todo[v * W.gw + u] = 1; ++need; }
      }
    }
    for (;;) {
      int ba = -1;
      long bg = 0;
      for (int a = 0; a < A; ++a) {
        if (F[G.anchor[a]] == 0xffff) continue;
        long g = 0;
        for (int32_t idx : see[a]) g += todo[idx];
        if (g > bg) { bg = g; ba = a; }
      }
      if (ba < 0 || bg < o.tile_min) break;
      for (int32_t idx : see[ba]) todo[idx] = 0;
      got += bg;
      views[rep].push_back(ba);
    }
    covers[rep] = need > 0 ? (double)got / need : 0;
  };
  const size_t step = std::max<size_t>(1, W.spawn.size() / 64);   // 시작 칸은 많아야 64 개쯤만
  int ns = 0;
  for (size_t si = 0; si < W.spawn.size(); si += step) {
    const int sc = W.spawn[si];
    if (sc < 0 || sc >= n || !G.stand[sc]) continue;
    int rep = -1;
    for (int a = 0; a < A && rep < 0; ++a) if (G.d(a, sc) != 0xffff) rep = a;
    if (rep < 0) continue;
    if (covers[rep] < 0) solve(rep);
    const std::vector<int>& view = views[rep];
    const int m = (int)view.size();
    const auto dd = [&](int i, int j) { return (double)G.d(view[i], G.anchor[view[j]]); };
    const auto d0 = [&](int i) { return (double)G.d(view[i], sc); };
    std::vector<int> tour;
    std::vector<uint8_t> used(m, 0);
    for (int t = 0; t < m; ++t) {   // 최근접 이웃
      int bj = -1;
      double bd = 1e18;
      for (int j = 0; j < m; ++j) {
        if (used[j]) continue;
        const double x = tour.empty() ? d0(j) : dd(tour.back(), j);
        if (x < bd) { bd = x; bj = j; }
      }
      used[bj] = 1;
      tour.push_back(bj);
    }
    const auto dprev = [&](int t) { return t == 0 ? d0(tour[0]) : dd(tour[t - 1], tour[t]); };
    for (int pass = 0; pass < 50; ++pass) {   // 2-opt (시작 고정, 열린 길) — 더 안 줄 때까지
      bool ch = false;
      for (int a = 0; a < m - 1; ++a)
        for (int b = a + 1; b < m; ++b) {
          const double old = dprev(a) + (b + 1 < m ? dd(tour[b], tour[b + 1]) : 0.0);
          const double nw = (a == 0 ? d0(tour[b]) : dd(tour[a - 1], tour[b])) + (b + 1 < m ? dd(tour[a], tour[b + 1]) : 0.0);
          if (nw + 1e-9 < old) { std::reverse(tour.begin() + a, tour.begin() + b + 1); ch = true; }
        }
      if (!ch) break;
    }
    double len = 0, turn = 0, prev = 0;
    int px = sc % W.gw, py = sc / W.gw;
    for (int t = 0; t < m; ++t) {
      len += dprev(t);
      const int a = G.anchor[view[tour[t]]], ax = a % W.gw, ay = a / W.gw;
      const double h = std::atan2((double)(ay - py), (double)(ax - px));
      if (t > 0) turn += std::fabs(std::remainder(h - prev, 2 * M_PI));
      prev = h; px = ax; py = ay;
    }
    T.len_m += len * c.res;
    T.turn_rad += turn;
    T.cover += covers[rep];
    T.n_view += m;
    ++ns;
  }
  if (ns > 0) { T.len_m /= ns; T.cover /= ns; T.turn_rad /= ns; T.n_view = (int)std::lround((double)T.n_view / ns); }
  return T;
}

}  // namespace cr
