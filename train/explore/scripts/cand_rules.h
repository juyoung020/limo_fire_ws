// 계층형 후보 고르기 규칙 정책 (frontier_run --modes cand_* — mdp/README "후보 고르기 규칙 견줌"). 측정 전용, 호스트(CPU)에서만 돈다.
// 시뮬이 준 후보 칸(Env::get_cand, 정책이 보는 것과 같은 K 개) 가운데 하나를 고른다 — 후보 밖은 못 고른다(정책과 같은 행동 공간).
//   cand_gain   : 정보량 / 거리 탐욕. 정보량 = 후보 칸에 닿는 방향(BFS 경로 끝 방향)으로 섰을 때 라이다 시야각(220°)·사거리 안에서 지도 막힘에 안 가리고 보이는
//                 모름 칸 수 (로봇 지도만 씀). 틀: Bircher 2016 NBVP 의 "보이는 모름 복셀 수", González-Baños 2002 의 정보량 ÷ 거리 꼴
//   cand_gainf  : 같은데 정보량 = 보이는 "닫힘을 막는 프론티어 칸" 수 (ARiADNE·HEADER 의 노드 유틸리티 = 범위 안에서 가림 없이 보이는 프론티어 수)
//   cand_tsp    : 후보들을 모두 도는 가장 짧은 차례(로봇 지도 BFS 거리, 정확한 동적 계획법 — 후보 ≤ 8)의 첫 후보 (FUEL 의 프론티어 순회 TSP 와 같은 틀, GT 안 씀)
//   cand_gtgain : cand_gain 의 정보량을 정답 격자(GT)로 — GT 막힘에 안 가리고 보이는 칸 중 "닫히려면 알아야 하는 칸"(GtMap::need)이고 아직 모름인 칸 수
//   cand_gt     : GT 순회. 남은 일 = need 칸 중 모름인 칸 + 지금 닫힘을 막는 프론티어 칸, 타일(GtMap) 닻으로 묶음. 후보마다
//                 비용 = 로봇→후보 거리 + (후보에서 GT 로 실제 보이는 일을 뺀 뒤) 남은 타일 닻을 모두 도는 순회 길이(GT 최단거리, 최근접 이웃 + 2-opt) → 가장 싼 후보.
//                 "남은 구역 순회의 첫 목적지" 를 후보 안에서 고르는 것 — 후보 고르기의 상한 **추정**(엄밀한 상한 아님: 순회는 근사고, 닻에 가면 다 보인다는 모델은 거칠다)
//   cand_stat   : 늘 0 번 (slot0 과 같은 판) — 후보 통계만 냄
// 후보 통계 (모든 cand_* 모드, Stat): 유효 후보 수 · 후보에 못 든 덩어리 수(닫힘을 막는 덩어리 − 후보 수) · 더 가까운 후보의 kNearM 안에 있는 후보 수(한곳에 몰림) ·
//   후보에 못 든 덩어리 가운데 가장 큰 것이 후보 가운데 가장 큰 것보다 큰 결정 비율(먼 큰 덩어리가 빠짐)
// 공통: --rule-stick x — 지금 목표 후보가 있으면, 가장 좋은 후보가 그보다 (1 + x) 배 넘게 좋을 때만 바꿈 (왔다 갔다 막기. 0 = 늘 가장 좋은 것)
#pragma once
#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <thread>
#include <vector>

#include "env_api.h"
#include "frontier.h"
#include "observations.h"

namespace cr {

enum Mode { kGain, kGainF, kTsp, kGtGain, kGt, kStat };
inline int mode_of(const std::string& m) {
  return m == "cand_gain" ? kGain : m == "cand_gainf" ? kGainF : m == "cand_tsp" ? kTsp : m == "cand_gtgain" ? kGtGain : m == "cand_gt" ? kGt : m == "cand_stat" ? kStat : -1;
}

struct Opt {
  float range_m = 0;     // 정보량 사거리 (m). 0 = 지도 사거리(env.yaml map.max_range)
  float fov_deg = 220;   // 정보량 시야각 — 라이다 시야각(frontier_run kLidarFov 와 같은 값)
  float stick = 0;       // 위 --rule-stick
  float turn_m = 0;      // --rule-turn: 첫 걸음 방향이 지금 머리 방향과 1 rad 어긋날 때마다 거리에 더하는 값 (m). FUEL 의 비용 max(경로/v, Δyaw/ω) 에서 도는 값을 거리로 바꿔 더한 꼴. 0 = 끔
  float tile_m = 1.5f;   // cand_gt 타일 변 (m) — **잠정** (닻 수 ≈ 바닥 넓이 / 타일², 거리 표 메모리로 정함)
  int tile_min = 4;      // cand_gt: 타일의 남은 need 칸이 이 수 이상이면 남은 일 (닫힘 덩어리 문턱 close_min 과 같은 4 칸) — **잠정**
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

// 정답 격자를 다 아는 순회 (cand_gt 를 돌릴 때 맵마다 한 번 찍음 — "지도를 미리 다 알면 얼마나 짧게 도나" 의 참고값. 온라인 정책이 닿을 수 없는 값이고 엄밀한 최솟값도 아님:
// 실제로 갈 수 있는 길 하나의 길이라 참 최솟값 ≤ 이 값).
//   시점 = 타일 닻. 닻마다 GT 에서 360° · 사거리 R 칸 안에 가림 없이 보이는 need 칸을 세고(맞은 벽 칸까지), 새로 보이는 칸이 가장 많은 닻을 차례로 고름(탐욕 집합 덮기),
//   남은 이득 < tile_min 이면 끝(시작 칸과 이어진 구역 안에서만). 시작 칸마다 고른 닻을 모두 도는 열린 길(최근접 이웃 + 2-opt, GT 최단거리)의 길이를 평균. 220° 시야는 제자리 회전(거리 0)으로 채운다고 봄.
struct GtTour { int n_view = 0; double cover = 0, len_m = 0; };
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
    double len = 0;
    for (int t = 0; t < m; ++t) len += dprev(t);
    T.len_m += len * c.res;
    T.cover += covers[rep];
    T.n_view += m;
    ++ns;
  }
  if (ns > 0) { T.len_m /= ns; T.cover /= ns; T.n_view = (int)std::lround((double)T.n_view / ns); }
  return T;
}

constexpr float kNearM = 1.0f;   // 통계: "한곳에 몰림" 을 재는 거리 (m) — 재는 자일 뿐 규칙엔 안 씀
struct Stat {
  long dec = 0, valid = 0, left = 0, crowd = 0, big_left = 0, full = 0, nonzero = 0, sw = 0;   // sw: 지금 목표가 있는데 다른 후보를 고른 결정
  void add(const Stat& o) { dec += o.dec; valid += o.valid; left += o.left; crowd += o.crowd; big_left += o.big_left; full += o.full; nonzero += o.nonzero; sw += o.sw; }
};

// 스레드마다 버퍼
struct Work {
  std::vector<uint8_t> flag;
  std::vector<int32_t> qa, qb, dist, label, d2, stamp, cnt, seenc;
  std::vector<mdp::Frontier> out;
  std::vector<int> pend, tour;
  std::vector<uint64_t> key2;
  Stat st;
  int stamp_id = 0;
  void fit(int n, int A) {
    if ((int)flag.size() < n) { flag.resize(n); qa.resize(n); qb.resize(n); dist.resize(n); label.resize(n); d2.resize(n); stamp.assign(n, 0); }
    if (out.size() < (size_t)mdp::kReachCap + 1) { out.resize(mdp::kReachCap + 1); key2.resize(mdp::kReachCap + 1); }
    if ((int)cnt.size() < A) { cnt.resize(A); seenc.resize(A); }
  }
};

// 판 하나의 결정에 필요한 것
struct In {
  const sim::EnvCfg* c;
  const sim::World* W;
  const GtMap* G;            // GT 규칙만
  const int16_t* L;          // 로봇 지도 (줄 = c->size)
  const uint32_t* trail;     // 지나간 칸·본 칸 판
  const uint8_t* cam;        // 깊이 카메라 층 (없으면 nullptr)
  const sim::EnvState* s;
  const int32_t* cand;       // 후보 칸 K 개
};

// 고른 후보 번호 (통계는 w.st 에 더함)
inline int choose(int mode, const In& in, const Opt& o, Work& w) {
  using mdp::hier::K;
  const sim::EnvCfg& c = *in.c;
  const sim::World& W = *in.W;
  const int n = W.gw * W.gh, A = in.G ? in.G->nA() : 0;
  w.fit(n, A);
  int v[K], nv = 0;
  for (int k = 0; k < K; ++k) if (in.cand[k] >= 0) v[nv++] = k;
  w.st.dec += 1;
  w.st.valid += nv;
  if (nv == 0) return 0;
  const mdp::MapView mv{in.L, c.size, W.gw, W.gh};
  const sim::EnvState& s = *in.s;
  const int rx = (int)std::floor((s.x + s.ex - W.ox) * c.inv_res), ry = (int)std::floor((s.y + s.ey - W.oy) * c.inv_res);
  mdp::SearchBuf sb{w.flag.data(), w.qa.data(), w.qb.data(), w.dist.data(), w.label.data()};
  const int ws = (c.size + 31) / 32, nf = c.size * ws;
  sb.vis = in.trail;
  sb.look = in.trail + nf;
  sb.tws = ws;
  sb.close_min = c.close_min;
  sb.rbc = mdp::robot_cells(c.close_radius, c.res);
  sb.cam = in.cam;
  const mdp::SearchResult r = mdp::frontier_search(mv, rx, ry, mdp::obsgen::reach_robot(c), 1, sb, w.out.data(), (int)w.out.size());
  const int n_out = std::min(r.n_found, (int)w.out.size());
  const auto blocking = [&](int idx) {   // 닫힘을 막는 프론티어 칸
    if (!(w.flag[idx] & 2)) return false;
    const int l = w.label[idx];
    if (l < 0 || l >= n_out || w.out[l].n < c.close_min) return false;
    return !mdp::looked_cell(mv, sb.look, sb.vis, ws, idx % W.gw, idx / W.gw);
  };
  {   // 후보 통계
    mdp::cand_keys(mv, mdp::obsgen::reach_robot(c), r, sb, w.out.data(), n_out, w.key2.data());
    int elig = 0, big_in = 0, big_out = 0;
    for (int i = 0; i < n_out; ++i) {
      if (w.key2[i] == ~0ull) continue;
      ++elig;
      bool in_c = false;
      for (int k = 0; k < nv; ++k) in_c = in_c || w.label[in.cand[v[k]]] == i;
      (in_c ? big_in : big_out) = std::max(in_c ? big_in : big_out, w.out[i].n);
    }
    w.st.left += std::max(0, elig - nv);
    w.st.full += elig > nv;
    w.st.big_left += elig > nv && big_out > big_in;
    const float lim = kNearM * c.inv_res;
    for (int a = 1; a < nv; ++a) {
      bool near = false;
      for (int b = 0; b < a && !near; ++b) {
        const int dx = in.cand[v[a]] % W.gw - in.cand[v[b]] % W.gw, dy = in.cand[v[a]] / W.gw - in.cand[v[b]] / W.gw;
        near = dx * dx + dy * dy <= lim * lim;
      }
      w.st.crowd += near;
    }
  }
  if (nv == 1 || mode == kStat) return v[0];
  int cur = -1;
  if (s.goal_cell >= 0 && s.goal_cell < n && (w.flag[s.goal_cell] & 2))
    for (int i = 0; i < nv; ++i)
      if ((w.flag[in.cand[v[i]]] & 2) && w.label[in.cand[v[i]]] == w.label[s.goal_cell]) cur = i;
  if (cur < 0 && c.cand_v2 && s.goal_cell >= 0 && s.goal_cell < n) {   // 후보 v2: 목표 덩어리가 다른 후보에 합쳐졌을 수 있음 → 합치는 반경 안 가장 가까운 후보
    long bd = (long)mdp::cand2_merge_cells(c.res) * mdp::cand2_merge_cells(c.res) + 1;
    for (int i = 0; i < nv; ++i) {
      const long dx = in.cand[v[i]] % W.gw - s.goal_cell % W.gw, dy = in.cand[v[i]] / W.gw - s.goal_cell / W.gw;
      if (dx * dx + dy * dy < bd) { bd = dx * dx + dy * dy; cur = i; }
    }
  }
  double dk[K];
  for (int i = 0; i < nv; ++i) {
    const int cell = in.cand[v[i]];
    dk[i] = (w.flag[cell] & 1) && w.dist[cell] >= 0 ? (double)w.dist[cell] : 1e6;
  }
  // 닿는 방향: 후보 칸에서 BFS 거리가 줄어드는 쪽으로 back 칸 거슬러 간 칸 → 후보 칸
  const auto heading = [&](int cell) {
    int at = cell;
    for (int b = 0; b < 10; ++b) {
      const int x = at % W.gw, y = at / W.gw;
      int nxt = -1;
      for (int k = 0; k < 4; ++k) {
        int dx, dy;
        mdp::nb4(k, &dx, &dy);
        const int u = x + dx, vv = y + dy;
        if (u < 0 || vv < 0 || u >= W.gw || vv >= W.gh) continue;
        const int j = vv * W.gw + u;
        if ((w.flag[j] & 1) && w.dist[j] >= 0 && w.dist[j] == w.dist[at] - 1) { nxt = j; break; }
      }
      if (nxt < 0) break;
      at = nxt;
    }
    if (at == cell) return std::atan2((double)(cell / W.gw - ry), (double)(cell % W.gw - rx));
    return std::atan2((double)(cell / W.gw - at / W.gw), (double)(cell % W.gw - at % W.gw));
  };
  if (o.turn_m > 0)   // 첫 걸음 방향: 후보에서 로봇까지 거슬러 가다 로봇에서 10 칸 떨어진 칸
    for (int i = 0; i < nv; ++i) {
      if (dk[i] >= 1e6) continue;
      int at = in.cand[v[i]];
      while (w.dist[at] > 10) {
        const int x = at % W.gw, y = at / W.gw;
        int nxt = -1;
        for (int k = 0; k < 4 && nxt < 0; ++k) {
          int dx, dy;
          mdp::nb4(k, &dx, &dy);
          const int u = x + dx, vv = y + dy;
          if (u < 0 || vv < 0 || u >= W.gw || vv >= W.gh) continue;
          const int j = vv * W.gw + u;
          if ((w.flag[j] & 1) && w.dist[j] >= 0 && w.dist[j] == w.dist[at] - 1) nxt = j;
        }
        if (nxt < 0) break;
        at = nxt;
      }
      const double d = std::atan2((double)(at / W.gw - ry), (double)(at % W.gw - rx)) - (double)(s.th + s.eth);
      dk[i] += o.turn_m * c.inv_res * std::fabs(std::atan2(std::sin(d), std::cos(d)));
    }
  const float range_m = o.range_m > 0 ? o.range_m : c.map_max_range;
  const int Rr = (int)(range_m * c.inv_res);
  const double fov = o.fov_deg * M_PI / 180.0;
  // 광선 쏘기: 후보 칸 가운데에서 시야 부채꼴로, 칸마다 한 번 visit(칸) — false 를 돌려주면 그 광선 끝
  const auto cast = [&](int cell, double th, const auto& visit) {
    const int cx = cell % W.gw, cy = cell / W.gw;
    const int nr = std::max(8, (int)std::ceil(fov * Rr));
    for (int a = 0; a < nr; ++a) {
      const double ang = th - 0.5 * fov + fov * (a + 0.5) / nr, ca = std::cos(ang), sa = std::sin(ang);
      for (int t = 1; t <= Rr; ++t) {
        const int x = cx + (int)std::lround(t * ca), y = cy + (int)std::lround(t * sa);
        if (x < 0 || y < 0 || x >= W.gw || y >= W.gh) break;
        if (!visit(y * W.gw + x)) break;
      }
    }
  };
  const auto fresh = [&](int idx) {
    if (w.stamp[idx] == w.stamp_id) return false;
    w.stamp[idx] = w.stamp_id;
    return true;
  };
  double score[K];   // 클수록 좋음
  if (mode == kGain || mode == kGainF || mode == kGtGain) {
    for (int i = 0; i < nv; ++i) {
      const int cell = in.cand[v[i]];
      ++w.stamp_id;
      long g = 0;
      const double th = heading(cell);
      if (mode == kGain)
        cast(cell, th, [&](int idx) {
          const uint8_t k = mdp::cls_at(mv, idx % W.gw, idx / W.gw);
          if (k == mdp::kOcc) return false;
          if (k == mdp::kUnk && fresh(idx)) ++g;
          return true;
        });
      else if (mode == kGainF)
        cast(cell, th, [&](int idx) {
          const uint8_t k = mdp::cls_at(mv, idx % W.gw, idx / W.gw);
          if (k != mdp::kFree) return false;   // 아는 빈 칸으로만 보임
          if (blocking(idx) && fresh(idx)) ++g;
          return true;
        });
      else
        cast(cell, th, [&](int idx) {
          if (W.grid[idx] & sim::kCellBlocked) {   // 맞은 벽 칸까지는 알게 됨
            if (in.G->need[idx] && mdp::cls_at(mv, idx % W.gw, idx / W.gw) == mdp::kUnk && fresh(idx)) ++g;
            return false;
          }
          if (in.G->need[idx] && mdp::cls_at(mv, idx % W.gw, idx / W.gw) == mdp::kUnk && fresh(idx)) ++g;
          return true;
        });
      score[i] = (double)g / std::max(1.0, dk[i]);
    }
    // 정보량이 모두 0 이면 가까운 순 (후보는 가까운 순이라 0 번)
    bool any = false;
    for (int i = 0; i < nv; ++i) any = any || score[i] > 0;
    if (!any) for (int i = 0; i < nv; ++i) score[i] = 1.0 / std::max(1.0, dk[i]);
  } else if (mode == kTsp) {
    double D[K][K];
    for (int i = 0; i < nv; ++i) {   // 후보 i 에서 번진 칸 BFS
      int head = 0, tail = 0;
      for (int q = 0; q < r.n_reach; ++q) w.d2[w.qa[q]] = -1;
      // qa 는 frontier_search 가 채운 번진 칸 목록 — 여기서 qb 를 큐로 씀
      w.qb[tail++] = in.cand[v[i]];
      w.d2[in.cand[v[i]]] = 0;
      while (head < tail) {
        const int idx = w.qb[head++], x = idx % W.gw, y = idx / W.gw;
        for (int k = 0; k < 4; ++k) {
          int dx, dy;
          mdp::nb4(k, &dx, &dy);
          const int u = x + dx, vv = y + dy;
          if (u < 0 || vv < 0 || u >= W.gw || vv >= W.gh) continue;
          const int j = vv * W.gw + u;
          if (!(w.flag[j] & 1) || w.d2[j] >= 0) continue;
          w.d2[j] = w.d2[idx] + 1;
          w.qb[tail++] = j;
        }
      }
      for (int j = 0; j < nv; ++j) D[i][j] = w.d2[in.cand[v[j]]] >= 0 ? (double)w.d2[in.cand[v[j]]] : 1e6;
    }
    // 정확한 DP: best[mask][last] = 로봇에서 mask 를 돌아 last 에서 끝나는 가장 짧은 길, first[mask][last] = 그 길의 첫 후보
    const int M = 1 << nv;
    std::vector<double> best((size_t)M * nv, 1e18);
    std::vector<int8_t> first((size_t)M * nv, -1);
    for (int i = 0; i < nv; ++i) { best[(size_t)(1 << i) * nv + i] = dk[i]; first[(size_t)(1 << i) * nv + i] = (int8_t)i; }
    for (int m = 1; m < M; ++m)
      for (int l = 0; l < nv; ++l) {
        const double b = best[(size_t)m * nv + l];
        if (!(m >> l & 1) || b >= 1e17) continue;
        for (int j = 0; j < nv; ++j) {
          if (m >> j & 1) continue;
          const size_t t = (size_t)(m | 1 << j) * nv + j;
          if (b + D[l][j] < best[t]) { best[t] = b + D[l][j]; first[t] = first[(size_t)m * nv + l]; }
        }
      }
    // 후보 i 로 시작하는 가장 짧은 순회 길이 → 점수 = −길이
    for (int i = 0; i < nv; ++i) score[i] = -1e18;
    for (int l = 0; l < nv; ++l) {
      const size_t t = (size_t)(M - 1) * nv + l;
      if (first[t] >= 0 && -best[t] > score[first[t]]) score[first[t]] = -best[t];
    }
  } else {   // kGt
    const GtMap& G = *in.G;
    for (int a = 0; a < A; ++a) w.cnt[a] = 0;
    std::vector<uint8_t>& fb = w.flag;   // bit3: 이 결정에서 "남은 일" 칸
    const int big = o.tile_min;          // 막는 프론티어 칸 하나 = 타일을 남은 일로 만듦
    for (int i = 0; i < n; ++i) {
      if (G.own[i] < 0) continue;
      const bool unk = G.need[i] && in.L[(i / W.gw) * c.size + i % W.gw] == mdp::kUnknownL;
      if (unk) w.cnt[G.own[i]] += 1;
      if (unk) fb[i] |= 8;
    }
    for (int q = 0; q < r.n_reach; ++q) {
      const int idx = w.qa[q];
      if (G.own[idx] >= 0 && blocking(idx)) { w.cnt[G.own[idx]] += big; fb[idx] |= 16; }
    }
    // GT 격자에서 칸을 stand 칸으로 (지도 잡음으로 후보 칸이 GT stand 가 아닐 때 둘레 가장 가까운 stand 칸)
    const auto snap = [&](int cell) {
      if (G.stand[cell]) return cell;
      const int x = cell % W.gw, y = cell / W.gw;
      int best = -1, bd = 1 << 30;
      for (int dy = -8; dy <= 8; ++dy)
        for (int dx = -8; dx <= 8; ++dx) {
          const int u = x + dx, vv = y + dy;
          if (u < 0 || vv < 0 || u >= W.gw || vv >= W.gh || !G.stand[vv * W.gw + u]) continue;
          if (dx * dx + dy * dy < bd) { bd = dx * dx + dy * dy; best = vv * W.gw + u; }
        }
      return best;
    };
    for (int i = 0; i < nv; ++i) {
      const int cell = in.cand[v[i]], sc = snap(cell);
      // 이 후보에 서면 GT 로 실제 보이는 남은 일을 타일에서 뺌
      for (int a = 0; a < A; ++a) w.seenc[a] = 0;
      ++w.stamp_id;
      cast(cell, heading(cell), [&](int idx) {
        const bool blk = W.grid[idx] & sim::kCellBlocked;
        if (G.own[idx] >= 0 && fresh(idx)) w.seenc[G.own[idx]] += ((fb[idx] & 8) ? 1 : 0) + ((fb[idx] & 16) ? big : 0);
        return !blk;
      });
      w.pend.clear();
      for (int a = 0; a < A; ++a)
        if (w.cnt[a] - w.seenc[a] >= o.tile_min) w.pend.push_back(a);
      double len = 0;
      const int m = (int)w.pend.size();
      if (m > 0 && sc >= 0) {
        // 최근접 이웃
        w.tour.assign(m, 0);
        std::vector<uint8_t> used(m, 0);
        const auto d0 = [&](int a) { const uint16_t x = G.d(a, sc); return x == 0xffff ? 1e5 : (double)x; };
        const auto dd = [&](int a, int b) { const uint16_t x = G.d(a, G.anchor[b]); return x == 0xffff ? 1e5 : (double)x; };
        int last = -1;
        for (int t = 0; t < m; ++t) {
          int bj = -1;
          double bdv = 1e18;
          for (int j = 0; j < m; ++j) {
            if (used[j]) continue;
            const double x = last < 0 ? d0(w.pend[j]) : dd(w.pend[last], w.pend[j]);
            if (x < bdv) { bdv = x; bj = j; }
          }
          used[bj] = 1;
          w.tour[t] = w.pend[bj];
          last = bj;
        }
        // 2-opt (시작 고정, 열린 길)
        const auto dprev = [&](int t) { return t == 0 ? d0(w.tour[0]) : dd(w.tour[t - 1], w.tour[t]); };
        for (int pass = 0; pass < 4; ++pass) {
          bool ch = false;
          for (int a = 0; a < m - 1; ++a)
            for (int b = a + 1; b < m; ++b) {
              // 구간 [a, b] 뒤집기: (a−1 → a) + (b → b+1) 대신 (a−1 → b) + (a → b+1)
              const double old = dprev(a) + (b + 1 < m ? dd(w.tour[b], w.tour[b + 1]) : 0.0);
              const double nw = (a == 0 ? d0(w.tour[b]) : dd(w.tour[a - 1], w.tour[b])) + (b + 1 < m ? dd(w.tour[a], w.tour[b + 1]) : 0.0);
              if (nw + 1e-9 < old) { std::reverse(w.tour.begin() + a, w.tour.begin() + b + 1); ch = true; }
            }
          if (!ch) break;
        }
        for (int t = 0; t < m; ++t) len += dprev(t);
      }
      score[i] = -(dk[i] + len);
    }
    for (int i = 0; i < n; ++i) fb[i] &= 7;
  }
  int bi = 0;
  for (int i = 1; i < nv; ++i)
    if (score[i] > score[bi]) bi = i;
  if (cur >= 0 && cur != bi && o.stick > 0) {
    // 비율 규칙: 점수가 양수면(정보량 ÷ 거리) best > (1 + x) · cur, 음수면(길이) |best| < |cur| / (1 + x)
    const bool better = score[bi] > 0 ? score[bi] > (1.0 + o.stick) * score[cur] : -score[bi] * (1.0 + o.stick) < -score[cur];
    if (!better) bi = cur;
  }
  w.st.nonzero += v[bi] != 0;
  w.st.sw += cur >= 0 && bi != cur;
  return v[bi];
}

// 환경 N 개를 스레드로 나눠 고름. act[DIM · i] 에 후보 번호를 적음 (끝난 판은 건너뜀). 통계는 *stat 에 더함
inline void choose_all(int mode, const sim::EnvCfg& c, const sim::World& W, const GtMap* G, const Opt& o, int N, const std::vector<int16_t>& lmap,
                       const std::vector<uint32_t>& trail, const std::vector<uint8_t>& cam, const std::vector<sim::EnvState>& st,
                       const std::vector<int32_t>& cand, const std::vector<int>& fin, float* act, int act_dim, Stat* stat) {
  const int nt = std::max(1u, std::min(12u, std::thread::hardware_concurrency()));
  std::atomic<int> next{0};
  static std::vector<Work> works;
  if ((int)works.size() < nt) works.resize(nt);
  const size_t ws = (size_t)(c.size + 31) / 32, per_trail = 2 * c.size * ws, per_map = (size_t)c.size * c.size;
  std::vector<std::thread> th;
  for (int t = 0; t < nt; ++t)
    th.emplace_back([&, t]() {
      for (int i; (i = next.fetch_add(1)) < N;) {
        if (fin[i]) continue;
        In in{&c, &W, G, &lmap[i * per_map], &trail[i * per_trail], cam.empty() ? nullptr : &cam[i * per_map], &st[i], &cand[(size_t)i * mdp::hier::K]};
        act[(size_t)act_dim * i] = (float)choose(mode, in, o, works[t]);
      }
    });
  for (auto& x : th) x.join();
  for (int t = 0; t < nt; ++t) { stat->add(works[t].st); works[t].st = Stat(); }
}

}  // namespace cr
