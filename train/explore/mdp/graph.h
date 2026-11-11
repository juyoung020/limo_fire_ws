// 그래프 모드 관측 (mdp/README.md "그래프 모드", 크기·칸 뜻은 obs_layout.h mdp::graph) — HEADER(Cao et al., arXiv 2510.15679) 식
// "노드 그래프 + 이웃 노드 선택". 정의는 논문과 전작 공개 코드(refs/code/large-scale-DRL-exploration node_manager.py·env.py·parameter.py)를
// 따르고, 다르게 한 곳은 README "논문과 다른 곳" 에 적는다.
//
// 이 파일은 지도(칸 종류)와 지도 자세만 쓰는 순수 함수다 — 시뮬 GT(정답 격자·참 자세)를 안 본다. 칸 접근은 호출자가 주는 함수로:
//   pass(x, y)  : 간선·노드가 지나도 되는 칸 (설정 graph_stand 1 = 로봇이 설 수 있는 칸 mdp::standable · 0 = 전작처럼 빈 칸)
//   free(x, y)  : 빈 칸 (앎 · 막힘 아님) — 유틸리티 시선
//   front(x, y) : 유틸리티가 세는 프론티어 칸 (설정 graph_open 1 = 닫힘을 막는 칸 EnvBatch::gfront · 0 = 모든 프론티어 칸 mdp::frontier_cell)
// 맵 밖 칸은 호출자 함수가 false 를 준다. GPU 커널(sim/src/env_kernel.cu env_graph_kernel)과 CPU 참조·실물 노드(아래 graph_build,
// sim/include/obs_ref.h graph_ref)가 같은 작은 함수를 불러 비트까지 같다 (env_verify --graph).
//
// 차례 (graph_build):
//   1 노드 격자: 지도 칸 격자에 고정, 간격 NODE_RES(칸 step), 위상 = 판 시작 칸 (로봇이 노드 위에서 시작 — 전작과 같음). 날 유효 = 맵 안 · pass
//   2 현재 노드 cur: 로봇 칸 둘레 5×5 노드 중 날 유효이고 로봇 칸에서 pass 직선이 닿는 것의 (거리², 번호) 최소 → 없으면 직선 없이 → 그래도 없으면
//     가장 가까운 격자점을 억지로 유효 (간선 없음 → 제자리 회전). cur 는 방문 표시(vis)에 더함
//   3 간선: 5×5 이웃과 pass 직선 (사이 칸 전부, 번호 작은 노드에서 큰 노드로 걸어 대칭)
//   4 cur 에서 간선으로 최단 거리 (간선 길이 = 유클리드 × 1000 정수). 안 이어진 노드는 버림 (전작: 로봇과 이어진 빈 칸만 노드)
//   5 프론티어 복셀: fvox 칸 네모마다 프론티어 칸 하나 (네모 최소 모서리에 가장 가까운 칸 — 전작 get_frontier_in_map 의 줄이기)
//   6 유틸리티: 노드에서 util_r 칸 안(거리 <)이고 free 직선이 닿는 복셀 수, min_util 이하면 0 (전작 MIN_UTILITY), 방문한 노드는 0
//   7 지역 길잡이: cur → 유틸리티 노드마다 최단 경로 위 노드 (cur 자신은 뺌 — 전작 경로 path[1:]), 앞 노드 = 거리가 맞는 가장 작은 스텐실 번호
//   8 특징 uint8 [N][FEAT], 간선 비트 [N], cur 의 이웃 칸 [NB] (행동 목표)
#pragma once
#include <cstdint>

#include "frontier.h"
#include "obs_layout.h"

namespace mdp::graphgen {

constexpr int kInf = 1 << 29;
constexpr int kVoxMax = 64 * 64;   // 프론티어 복셀 수 상한 (지도 512 칸 / fvox 8 칸)

struct Params {
  int step;        // 노드 간격 (칸)
  int ox, oy;      // 노드 (행 0, 열 0) 의 칸
  int gw, gh;      // 맵 격자
  int util_r2;     // 유틸리티 범위² (칸²)
  int min_util;    // 이 수 이하 유틸리티는 0
  int fvox;        // 프론티어 복셀 변 (칸)
  int util_norm;   // 유틸리티 정규화 분모
  int bx0, by0, bx1, by1;   // 지도에서 건드린 칸 범위 (프론티어는 이 안에만), bx1 < 0 이면 없음
};

DEV void stencil(int k, int* dy, int* dx) {
  const int kk = k < 12 ? k : k + 1;
  *dy = kk / 5 - 2;
  *dx = kk % 5 - 2;
}
// 간선 길이 × 1000 (유클리드, 노드 간격 단위)
DEV int edge_len(int k) {
  int dy, dx;
  stencil(k, &dy, &dx);
  const int d2 = dy * dy + dx * dx;
  return d2 == 1 ? 1000 : d2 == 2 ? 1414 : d2 == 4 ? 2000 : d2 == 5 ? 2236 : 2828;
}
DEV int node_x(const Params& p, int i) { return p.ox + (i % mdp::graph::GW) * p.step; }
DEV int node_y(const Params& p, int i) { return p.oy + (i / mdp::graph::GW) * p.step; }
DEV bool node_in_map(const Params& p, int i) { return node_x(p, i) < p.gw && node_y(p, i) < p.gh; }
// 스텐실 k 번 이웃 번호 (격자 밖 −1)
DEV int node_nb(int i, int k) {
  int dy, dx;
  stencil(k, &dy, &dx);
  const int r = i / mdp::graph::GW + dy, c = i % mdp::graph::GW + dx;
  return (r < 0 || c < 0 || r >= mdp::graph::GH || c >= mdp::graph::GW) ? -1 : r * mdp::graph::GW + c;
}
DEV int rdiv(int a, int n) { return a >= 0 ? (2 * a + n) / (2 * n) : -((-2 * a + n) / (2 * n)); }
// (x0, y0) → (x1, y1) 직선의 사이 칸(양 끝 뺌)이 모두 ok 인가
template <class F>
DEV bool line_ok(int x0, int y0, int x1, int y1, F ok) {
  const int dx = x1 - x0, dy = y1 - y0;
  const int ax = dx < 0 ? -dx : dx, ay = dy < 0 ? -dy : dy, n = ax > ay ? ax : ay;
  for (int i = 1; i < n; ++i)
    if (!ok(x0 + rdiv(i * dx, n), y0 + rdiv(i * dy, n))) return false;
  return true;
}
// 날 유효
template <class F>
DEV bool node_raw(const Params& p, int i, F pass) { return node_in_map(p, i) && pass(node_x(p, i), node_y(p, i)); }
// i < j 인 두 노드 사이 직선 (둘 다 유효일 때 부른다)
template <class F>
DEV bool edge_ok(const Params& p, int i, int j, F pass) { return line_ok(node_x(p, i), node_y(p, i), node_x(p, j), node_y(p, j), pass); }

// 현재 노드. valid(i) = 날 유효. *forced = 1 이면 억지로 유효로 만든 노드
template <class V, class F>
DEV int find_cur(const Params& p, int rx, int ry, V valid, F pass, int* forced) {
  const int cmax = (p.gw - 1 - p.ox) / p.step < mdp::graph::GW - 1 ? (p.gw - 1 - p.ox) / p.step : mdp::graph::GW - 1;
  const int rmax = (p.gh - 1 - p.oy) / p.step < mdp::graph::GH - 1 ? (p.gh - 1 - p.oy) / p.step : mdp::graph::GH - 1;
  int c0 = rdiv(rx - p.ox, p.step), r0 = rdiv(ry - p.oy, p.step);
  c0 = c0 < 0 ? 0 : c0 > cmax ? cmax : c0;
  r0 = r0 < 0 ? 0 : r0 > rmax ? rmax : r0;
  *forced = 0;
  for (int pass_n = 0; pass_n < 2; ++pass_n) {
    long long best = -1;
    for (int dr = -2; dr <= 2; ++dr)
      for (int dc = -2; dc <= 2; ++dc) {
        const int r = r0 + dr, c = c0 + dc;
        if (r < 0 || c < 0 || r > rmax || c > cmax) continue;
        const int i = r * mdp::graph::GW + c;
        if (!valid(i)) continue;
        const int x = node_x(p, i), y = node_y(p, i);
        if (pass_n == 0 && !line_ok(rx, ry, x, y, pass)) continue;
        const long long key = ((long long)((x - rx) * (x - rx) + (y - ry) * (y - ry)) << 10) | i;
        if (best < 0 || key < best) best = key;
      }
    if (best >= 0) return (int)(best & 1023);
  }
  *forced = 1;
  return r0 * mdp::graph::GW + c0;
}

// 프론티어 복셀 v (복셀 격자 vw = ⌈gw / fvox⌉ 열) 의 대표 칸 (y · gw + x), 없으면 −1
template <class F>
DEV int vox_rep(const Params& p, int v, F front) {
  const int vw = (p.gw + p.fvox - 1) / p.fvox;
  const int x0 = (v % vw) * p.fvox, y0 = (v / vw) * p.fvox;
  if (p.bx1 < 0 || y0 > p.by1 || y0 + p.fvox <= p.by0 || x0 > p.bx1 || x0 + p.fvox <= p.bx0) return -1;
  int best = kInf, cell = -1;
  for (int dy = 0; dy < p.fvox; ++dy)
    for (int dx = 0; dx < p.fvox; ++dx) {
      const int x = x0 + dx, y = y0 + dy;
      if (x >= p.gw || y >= p.gh || !front(x, y)) continue;
      const int key = dx * dx + dy * dy;   // 같으면 먼저 본 칸 (dy 작은 쪽)
      if (key < best) { best = key; cell = y * p.gw + x; }
    }
  return cell;
}
DEV int vox_count(const Params& p) { return ((p.gw + p.fvox - 1) / p.fvox) * ((p.gh + p.fvox - 1) / p.fvox); }
// 노드 i 가 프론티어 대표 칸을 보나 (범위 안 · free 직선)
template <class F>
DEV bool sees(const Params& p, int i, int cell, F free) {
  const int x = node_x(p, i), y = node_y(p, i), fx = cell % p.gw, fy = cell / p.gw;
  const int dx = fx - x, dy = fy - y;
  if (dx * dx + dy * dy >= p.util_r2) return false;
  return line_ok(x, y, fx, fy, free);
}
// 최단 거리 한 번 풀기: 노드 i 의 새 거리 (이웃 거리 + 간선 길이의 최솟값과 지금 값 중 작은 것)
DEV int relax(const int* dist, uint32_t edge, int i) {
  int best = dist[i];
  for (int k = 0; k < mdp::graph::NB; ++k) {
    if (!((edge >> k) & 1u)) continue;
    const int d = dist[node_nb(i, k)] + edge_len(k);
    best = d < best ? d : best;
  }
  return best;
}
// 최단 경로의 앞 노드 (거리가 맞는 가장 작은 스텐실 번호), cur 이거나 없으면 −1
DEV int pred_of(const int* dist, uint32_t edge, int i) {
  for (int k = 0; k < mdp::graph::NB; ++k) {
    if (!((edge >> k) & 1u)) continue;
    const int j = node_nb(i, k);
    if (dist[j] + edge_len(k) == dist[i]) return j;
  }
  return -1;
}
DEV uint8_t util_u8(const Params& p, int util) {
  if (util <= 0) return 0;
  const int v = util * 255 / p.util_norm;
  return (uint8_t)(v < 1 ? 1 : v > 255 ? 255 : v);
}
// 노드 특징 한 줄. 무효 노드는 모두 0
DEV void write_feat(uint8_t* F, int i, int cur, bool valid, int util, bool guide, const Params& p) {
  using namespace mdp::graph;
  for (int k = 0; k < FEAT; ++k) F[k] = 0;
  if (!valid) return;
  F[F_VALID] = 255;
  F[F_DX] = (uint8_t)(128 + 4 * (i % GW - cur % GW));   // 현재 노드 기준 지도 좌표 (돌리지 않음 — 전작), ±25.6 m → 0 ~ 255
  F[F_DY] = (uint8_t)(128 + 4 * (i / GW - cur / GW));
  F[F_UTIL] = util_u8(p, util);
  F[F_LGUIDE] = guide ? 255 : 0;
}

// CPU 참조·실물 노드: 한 환경 그래프 전부. vis: 방문 노드 비트 [N / 32] (호출자가 판 시작 때 비움, 여기서 cur 를 더함).
// feat [N · FEAT], edge [N], cand [NB] (cur 의 스텐실 k 번 이웃 칸 y · gw + x, 간선 없으면 −1), util [N] (날 수 — 도구·시험용, nullptr 가능)
template <class P, class Fr, class Fn>
inline int graph_build(const Params& p, int rx, int ry, P pass, Fr free, Fn front, uint32_t* vis, uint8_t* feat, uint32_t* edge, int32_t* cand,
                       int* util_out = nullptr) {
  using namespace mdp::graph;
  static thread_local int dist[N], util[N], reps[kVoxMax];
  static thread_local uint8_t valid[N], guide[N];
  for (int i = 0; i < N; ++i) {
    valid[i] = node_raw(p, i, pass);
    edge[i] = 0u;
    util[i] = 0;
    guide[i] = 0;
    dist[i] = kInf;
  }
  int forced = 0;
  const int cur = find_cur(p, rx, ry, [&](int i) { return valid[i] != 0; }, pass, &forced);
  valid[cur] = 1;
  vis[cur >> 5] |= 1u << (cur & 31);
  for (int i = 0; i < N; ++i) {
    if (!valid[i] || (forced && i == cur)) continue;
    for (int k = NB / 2; k < NB; ++k) {
      const int j = node_nb(i, k);
      if (j < 0 || !valid[j] || (forced && j == cur) || !edge_ok(p, i, j, pass)) continue;
      edge[i] |= 1u << k;
      edge[j] |= 1u << (NB - 1 - k);
    }
  }
  dist[cur] = 0;
  for (bool chg = true; chg;) {
    chg = false;
    for (int i = 0; i < N; ++i) {
      if (!valid[i]) continue;
      const int d = relax(dist, edge[i], i);
      if (d < dist[i]) { dist[i] = d; chg = true; }
    }
  }
  for (int i = 0; i < N; ++i)
    if (valid[i] && dist[i] >= kInf) { valid[i] = 0; edge[i] = 0u; }
  int nf = 0;
  const int nv = vox_count(p);
  for (int v = 0; v < nv; ++v) {
    const int c = vox_rep(p, v, front);
    if (c >= 0) reps[nf++] = c;
  }
  for (int i = 0; i < N; ++i) {
    if (!valid[i] || ((vis[i >> 5] >> (i & 31)) & 1u)) continue;
    int n = 0;
    for (int f = 0; f < nf; ++f) n += sees(p, i, reps[f], free) ? 1 : 0;
    util[i] = n > p.min_util ? n : 0;
  }
  for (int i = 0; i < N; ++i) {
    if (util[i] <= 0) continue;
    for (int v = i; v >= 0 && v != cur && !guide[v]; v = pred_of(dist, edge[v], v)) guide[v] = 1;
  }
  for (int i = 0; i < N; ++i) write_feat(feat + (size_t)i * FEAT, i, cur, valid[i] != 0, util[i], guide[i] != 0, p);
  for (int k = 0; k < NB; ++k) {
    const int j = ((edge[cur] >> k) & 1u) ? node_nb(cur, k) : -1;
    cand[k] = j >= 0 ? node_y(p, j) * p.gw + node_x(p, j) : -1;
  }
  if (util_out)
    for (int i = 0; i < N; ++i) util_out[i] = util[i];
  return cur;
}

}  // namespace mdp::graphgen
