// 지도 닫힘·프론티어 — 호스트·장치 공용(sim/include/detmath.h 의 결정적 규칙: 정수만 쓰고, 순서가 정해진 순차 BFS 이거나
// 순서와 무관한 정수 합치기(합·최솟값)). 같은 지도면 CPU 와 GPU 가 비트까지 같은 결과를 낸다(baselines/frontier/tools/frontier_check).
//
// 정의 (mdp/README "덮음·닫힘·끝", 2026-11-04 정함). 높이는 라이다 평면(0.111 m) 2D 지도 그대로 — 깊이 카메라는 안 씀.
//   칸 셋: 모름 = 한 번도 안 건드린 칸(빔이 지나가지도 맞지도 않음, sim::kUnknown) · 막힘 = p > 0.65(같은 칸 약 4 번 맞음) ·
//          갈 수 있음 = 그 밖에 건드린 칸. 빈 칸 확정 문턱(p < 0.196, 지나감 36 번)은 쓰지 않는다 — 그 문턱이 95 % 뒤 긴 꼬리의 원인이었다.
//   로봇 중심이 설 수 있는 칸 = 갈 수 있음이고 반지름(로봇 반경) 안에 막힘이 없는 칸 (막힘을 로봇 반경만큼 팽창).
//   프론티어 칸 = 갈 수 있음 칸 중 모름에 4 이웃으로 닿은 것.
//   닫힘 = 로봇 칸(에서 가장 가까운 설 수 있는 칸)부터 설 수 있는 칸을 4 이웃 flood fill 해서 프론티어 칸에 하나도 안 닿음.
//
// 쓰는 곳
//   - 프론티어 탐사 기준선(baselines/frontier): frontier_search — 찾기·끝(닫힘) 조건
//   - RL 관측 "도달 가능한 프론티어 요약"(mdp/README "관측"): reach_summary = frontier_search(+ BFS 거리) 결과에서 덩어리 고르기.
//     GPU 는 같은 값을 블록 병렬로(sim/src/env_kernel.cu reach_block), 프론티어 칸 비트 지도(fmap)는 frontier_cell 로 고친다
//
// 지도: 시뮬 지도(sim/include/env.h EnvBatch::lmap) 그대로 — 로그 오즈 × 1000(int16), 아직 모름 = −32768(sim::kUnknown).
//
// 출처(덩어리 묶기·크기 문턱·비용의 틀): m-explore(hrnr/m-explore) 커밋 712bdd41027b645c9c876a4c0071478f090825ea
//   explore/src/frontier_search.cpp:27-95   searchFrom — 로봇 칸에서 가장 가까운 빈 칸부터 4 이웃 BFS, 프론티어 칸을 만나면 덩어리를 만듦,
//                                           크기 × 해상도 ≥ min_frontier_size 만 남김(78)
//   explore/src/frontier_search.cpp:97-170  buildNewFrontier — 8 이웃 BFS 로 프론티어 칸을 묶고 크기·중심·로봇에서 가장 가까운 칸(middle)을 셈
//   explore/include/explore/costmap_tools.h:91-  nearestCell — 4 이웃 BFS 로 가장 가까운 (설 수 있는) 칸
// 원본과 다른 점(의도한 것)
//   - 프론티어 칸: 원본은 "빈 칸에 닿은 모르는 칸", 여기선 위 정의대로 "모름에 닿은 갈 수 있는 칸".
//   - 번지기: 원본은 빈 칸 전체로 번지고, 여기선 로봇이 설 수 있는 칸(막힘 팽창 밖)으로만 — 로봇이 못 들어가는 틈 너머는 닫힘으로 본다.
//     덩어리도 번진 칸 안의 프론티어 칸끼리만 묶는다(목표가 갈 수 있는 곳에 남게).
//   - 원본은 첫 칸(initial)을 크기에는 넣고(105) 중심 합·최소 거리에는 안 넣는다(145-158) → 중심이 세계 원점 쪽으로 치우친다. 여기선 첫 칸도 넣는다.
#pragma once
#include <cstdint>

#include "detmath.h"
#include "obs_layout.h"

namespace mdp {

// ---------------------------------------------------------------------------------------------------------------
// 칸 종류: 갈 수 있음 · 막힘 · 모름
//
// 막힘 문턱은 Cartographer 가 지도를 ROS 지도 파일로 쓸 때의 확률 문턱과 같게 한다:
//   cartographer_ros(커밋 c138034db0c47fe0ea5a2abe516acae02190dbf5) cartographer_ros/cartographer_ros/ros_map.cc:44-45 "occupied_thresh: 0.65"
// 로그 오즈 × 1000 정수로: 1000·ln(0.65/0.35) = 619.04 → p > 0.65 ⇔ l ≥ 620. 시뮬 갱신값(env.yaml map: hit 0.55 → +201)으로 맞음 4 번.
// 모름 = 한 번도 관측 안 된 칸(sim::kUnknown) — Cartographer /map 의 −1 과 같다(msg_conversion.cc:407-410, observed == 0 → −1).
constexpr int16_t kUnknownL = -32768;   // sim::kUnknown 과 같아야 함 (frontier_check 가 확인)
constexpr int16_t kOccL = 620;          // l ≥ 이것 → 막힘
constexpr double kOccP = 0.65;          // 위 정수의 근거(확인용)

enum : uint8_t { kFree = 0, kOcc = 254, kUnk = 255 };   // kFree = 갈 수 있음. explore_lite costmap 값과 같은 순서

DEV uint8_t cell_class(int16_t l) {
  if (l == kUnknownL) return kUnk;
  if (l >= kOccL) return kOcc;
  return kFree;
}

// 지도 보기: 칸 (x, y) = L[y · stride + x], 0 ≤ x < w, 0 ≤ y < h (시뮬은 stride = cfg.size, w·h = 맵 격자 gw·gh)
struct MapView {
  const int16_t* L;
  int stride, w, h;
};
DEV uint8_t cls_at(const MapView& m, int x, int y) { return cell_class(m.L[y * m.stride + x]); }

// 이웃 차례: 원본 costmap_tools.h nhood4(왼·오른·아래·위)·nhood8(nhood4 뒤에 왼아래·왼위·오른아래·오른위) 과 같다
DEV void nb4(int k, int* dx, int* dy) {
  const int X[4] = {-1, 1, 0, 0}, Y[4] = {0, 0, -1, 1};
  *dx = X[k]; *dy = Y[k];
}
DEV void nb8(int k, int* dx, int* dy) {
  const int X[8] = {-1, 1, 0, 0, -1, -1, 1, 1}, Y[8] = {0, 0, -1, 1, -1, 1, -1, 1};
  *dx = X[k]; *dy = Y[k];
}

// 프론티어 칸: 갈 수 있는 칸이고 4 이웃에 모름이 있음 (맵 밖 이웃은 안 셈)
DEV bool frontier_cell(const MapView& m, int x, int y) {
  if (cls_at(m, x, y) != kFree) return false;
  for (int k = 0; k < 4; ++k) {
    int dx, dy;
    nb4(k, &dx, &dy);
    const int u = x + dx, v = y + dy;
    if (u >= 0 && v >= 0 && u < m.w && v < m.h && cls_at(m, u, v) == kUnk) return true;
  }
  return false;
}

// 로봇 반경(칸): 칸 가운데끼리 거리² ≤ r2 인 칸에 막힘이 있으면 못 섬. r = ceil(√r2)
struct Robot {
  int32_t r, r2;
};
// res(m) 칸 지도에서 반경 radius(m) → Robot (호스트·장치 같은 float 셈)
DEV Robot robot_cells(float radius, float res) {
  const float q = radius / res;
  Robot o;
  o.r2 = (int32_t)(q * q);
  o.r = 0;
  while ((o.r + 1) * (o.r + 1) <= o.r2) ++o.r;
  return o;
}

// 로봇 중심이 설 수 있는 칸: 갈 수 있음 + 반경 안에 막힘 없음 (맵 밖은 안 봄)
DEV bool standable(const MapView& m, int x, int y, Robot rb) {
  if (cls_at(m, x, y) != kFree) return false;
  for (int dy = -rb.r; dy <= rb.r; ++dy)
    for (int dx = -rb.r; dx <= rb.r; ++dx) {
      if (dx * dx + dy * dy > rb.r2) continue;
      const int u = x + dx, v = y + dy;
      if (u >= 0 && v >= 0 && u < m.w && v < m.h && cls_at(m, u, v) == kOcc) return false;
    }
  return true;
}

// ---------------------------------------------------------------------------------------------------------------
// 닫힘 + 프론티어 덩어리: 로봇 칸 (rx, ry) 에서 설 수 있는 칸으로 이어진 프론티어 덩어리들.
// 덩어리 하나 — 모든 값이 칸 단위 정수(결정적). 세계 좌표는 쓰는 쪽에서: 중심 = 원점 + (sx / n + ½) · res
struct Frontier {
  int32_t n;            // 칸 수
  int64_t sx, sy;       // 칸 x·y 합 (중심 = 합 / n)
  int32_t min_d2;       // 로봇 칸에서 가장 가까운 칸까지 거리² (칸²)
  int32_t middle;       // 그 칸 번호 (y · w + x)
  int32_t initial;      // 처음 찾은 칸 번호
  int32_t near_dist;    // SearchBuf::dist 가 있을 때: 덩어리 칸 중 시작 칸에서 설 수 있는 칸 4 이웃 BFS 거리(칸)가 가장 짧은 값 (없으면 −1)
  int32_t near_cell;    //   그 칸 번호 (같은 거리면 번호가 작은 칸 — 순서와 무관)
};

// 버퍼(쓰는 쪽이 w·h 칸씩): flag(u8), qa·qb(int32). 장치에서는 환경마다 따로.
struct SearchBuf {
  uint8_t* flag;        // bit0 번진 칸(설 수 있음), bit1 덩어리로 묶임, bit2 nearestCell 지나감
  int32_t* qa;          // nearestCell·번지기 큐
  int32_t* qb;          // 덩어리 BFS 큐
  int32_t* dist = nullptr;   // 있으면 번진 칸마다 시작 칸에서 4 이웃 BFS 거리(칸)를 적고 덩어리 near_dist·near_cell 을 셈 (번지기는 FIFO 라 거리 = 부모 + 1)
  int32_t* label = nullptr;  // 있으면 덩어리로 묶인 칸(flag bit1)마다 그 덩어리 번호(찾은 차례, 문턱과 무관 — 계층형 후보의 "지금 목표" 판정)
};

struct SearchResult {
  int32_t n_found;      // min_cells 이상 덩어리 수 (max_out 보다 많으면 앞의 max_out 개만 out 에)
  int32_t n_small;      // min_cells 미만이라 버린 덩어리 수
  int32_t n_cells;      // 번진 칸 안 프론티어 칸 수 (덩어리 크기와 무관하게 모두)
  int32_t n_reach;      // 번진(설 수 있는) 칸 수
  int32_t start;        // 번지기 시작 칸 (가장 가까운 설 수 있는 칸, 없으면 로봇 칸)
  int32_t start_free;   // 1 = 설 수 있는 칸에서 시작함 (0 이면 아직 설 수 있는 칸이 없음 — 시작 직후)
};

// 닫힘: 설 수 있는 칸에서 번졌고 프론티어 칸에 하나도 안 닿음
DEV bool closed(const SearchResult& r) { return r.start_free != 0 && r.n_cells == 0; }

// ---------------------------------------------------------------------------------------------------------------
// 굵은 격자 닫힘 (판 끝 성공 판정, 2026-11-08 — mdp/README "판 끝"): 지도를 f×f 칸으로 줄여(막힘 하나라도 있으면 막힘, 모두 모름이면 모름,
// 그 밖은 갈 수 있음) 그 격자에서 로봇 반경 radius 로 엄격 닫힘(closed). 버퍼는 쓰는 쪽이 굵은 칸 수만큼(L int16, flag u8, qa·qb int32).
struct CoarseBuf {
  int16_t* L;
  uint8_t* flag;
  int32_t* qa;
  int32_t* qb;
};
DEV int coarse_dim(int n, int f) { return (n + f - 1) / f; }
namespace detail {
// buildNewFrontier (원본 97-170): 8 이웃으로, 번진 칸 안의 프론티어 칸을 묶음. 첫 칸도 합·거리에 넣는다(위 "원본과 다른 점")
DEV Frontier build(const MapView& m, int init, int rx, int ry, SearchBuf b, int lab) {
  Frontier f;
  f.n = 0; f.sx = 0; f.sy = 0; f.min_d2 = 0x7fffffff; f.middle = init; f.initial = init; f.near_dist = -1; f.near_cell = -1;
  int head = 0, tail = 0;
  b.qb[tail++] = init;
  while (head < tail) {
    const int idx = b.qb[head++];
    const int x = idx % m.w, y = idx / m.w;
    f.n += 1; f.sx += x; f.sy += y;
    if (b.label) b.label[idx] = lab;
    const int dx0 = x - rx, dy0 = y - ry, d2 = dx0 * dx0 + dy0 * dy0;
    if (d2 < f.min_d2) { f.min_d2 = d2; f.middle = idx; }
    if (b.dist) {
      const int bd = b.dist[idx];
      if (f.near_dist < 0 || bd < f.near_dist || (bd == f.near_dist && idx < f.near_cell)) { f.near_dist = bd; f.near_cell = idx; }
    }
    for (int k = 0; k < 8; ++k) {
      int dx, dy;
      nb8(k, &dx, &dy);
      const int u = x + dx, v = y + dy;
      if (u < 0 || v < 0 || u >= m.w || v >= m.h) continue;
      const int j = v * m.w + u;
      if ((b.flag[j] & 2) || !(b.flag[j] & 1) || !frontier_cell(m, u, v)) continue;
      b.flag[j] |= 2;
      b.qb[tail++] = j;
    }
  }
  return f;
}
}  // namespace detail

// min_cells: 원본 min_frontier_size / 해상도 (원본 78 줄: size · res ≥ min_frontier_size). rb: 로봇 반경 (robot_cells)
DEV SearchResult frontier_search(const MapView& m, int rx, int ry, Robot rb, int min_cells, SearchBuf b, Frontier* out, int max_out) {
  SearchResult r;
  r.n_found = 0; r.n_small = 0; r.n_cells = 0; r.n_reach = 0; r.start = -1; r.start_free = 0;
  const int n = m.w * m.h;
  if (rx < 0 || ry < 0 || rx >= m.w || ry >= m.h) return r;   // 원본 33-36: 로봇이 지도 밖
  for (int i = 0; i < n; ++i) b.flag[i] = 0;
  const int pos = ry * m.w + rx;
  // nearestCell(설 수 있는 칸): 로봇이 벽 팽창 안에 있을 때(벽에 붙음) 가장 가까운 설 수 있는 칸에서 시작
  {
    int head = 0, tail = 0;
    b.qa[tail++] = pos;
    b.flag[pos] |= 4;
    r.start = pos;
    while (head < tail) {
      const int idx = b.qa[head++];
      const int x = idx % m.w, y = idx / m.w;
      if (standable(m, x, y, rb)) { r.start = idx; r.start_free = 1; break; }
      for (int k = 0; k < 4; ++k) {
        int dx, dy;
        nb4(k, &dx, &dy);
        const int u = x + dx, v = y + dy;
        if (u < 0 || v < 0 || u >= m.w || v >= m.h) continue;
        const int j = v * m.w + u;
        if (b.flag[j] & 4) continue;
        b.flag[j] |= 4;
        b.qa[tail++] = j;
      }
    }
  }
  if (!r.start_free) return r;
  // 번지기: 설 수 있는 칸, 4 이웃 (닫힘 판정과 같은 flood fill)
  int head = 0, tail = 0;
  b.qa[tail++] = r.start;
  b.flag[r.start] |= 1;
  if (b.dist) b.dist[r.start] = 0;
  while (head < tail) {
    const int idx = b.qa[head++];
    const int x = idx % m.w, y = idx / m.w;
    for (int k = 0; k < 4; ++k) {
      int dx, dy;
      nb4(k, &dx, &dy);
      const int u = x + dx, v = y + dy;
      if (u < 0 || v < 0 || u >= m.w || v >= m.h) continue;
      const int j = v * m.w + u;
      if ((b.flag[j] & 1) || !standable(m, u, v, rb)) continue;
      b.flag[j] |= 1;
      if (b.dist) b.dist[j] = b.dist[idx] + 1;
      b.qa[tail++] = j;
    }
  }
  r.n_reach = tail;
  // 덩어리: 번진 차례대로 훑어 아직 안 묶인 프론티어 칸에서 묶기 시작
  for (int i = 0; i < tail; ++i) {
    const int idx = b.qa[i];
    const int x = idx % m.w, y = idx / m.w;
    if (!frontier_cell(m, x, y)) continue;
    r.n_cells += 1;
    if (b.flag[idx] & 2) continue;
    b.flag[idx] |= 2;
    const Frontier f = detail::build(m, idx, rx, ry, b, r.n_found + r.n_small);
    if (f.n >= min_cells) {
      if (r.n_found < max_out) out[r.n_found] = f;
      r.n_found += 1;
    } else {
      r.n_small += 1;
    }
  }
  return r;
}

// ---------------------------------------------------------------------------------------------------------------
// RL 관측 "도달 가능한 프론티어 요약" (mdp/README "관측", obs_layout.h FRONTIER·FRONTIER2·FRONTIER3·FRONT_N·CLOSED·FRONT_CELLS).
// 기준선(baselines/frontier)과 같은 frontier_search 결과(설 수 있는 칸으로 번진 곳 안 프론티어 덩어리, 크기 문턱, 닫힘)에서
// 문턱을 넘는 덩어리를 시작 칸에서의 BFS 거리(설 수 있는 칸 4 이웃, 칸) 짧은 순으로 kReachTop 개 고른다.
// 순서 열쇠 = (near_dist << 32) | near_cell — 덩어리마다 다르고(칸은 한 덩어리에만) 순서와 무관해 GPU 병렬판과 같다.
// GPU 는 같은 값을 블록 병렬(비트 BFS + 합집합 찾기)로 낸다 — sim/src/env_kernel.cu reach_block, 비트 비교는 sim/tools/env_verify.
constexpr int kReachTop = 3;
constexpr int kReachCap = 8192;   // GPU 덩어리 묶기 목록 칸 수 (지도 전체 프론티어 칸). 넘으면 덩어리 요약을 비움(overflow = 1) — CPU 도 같은 규칙
struct ReachTop {
  int32_t n;      // 덩어리 칸 수 (0 = 없음)
  int32_t dist;   // 시작 칸 → 덩어리 가장 가까운 칸 BFS 거리 (칸)
  int32_t cell;   // 그 칸 번호 (y · w + x)
};
struct ReachSummary {
  int32_t start_free;   // frontier_search 와 같음
  int32_t closed;       // closed(r)
  int32_t n_found;      // 문턱 넘는 덩어리 수 (overflow 면 0)
  int32_t n_cells;      // 번진 곳 안 프론티어 칸 수
  int32_t overflow;     // 지도 전체 프론티어 칸 > kReachCap
  ReachTop top[kReachTop];
};
DEV void reach_clear(ReachSummary* s) {
  s->start_free = s->closed = s->n_found = s->n_cells = s->overflow = 0;
  for (int k = 0; k < kReachTop; ++k) { s->top[k].n = 0; s->top[k].dist = -1; s->top[k].cell = -1; }
}
// 위 열쇠로 kReachTop 개 (오름차순) 를 고름. n_tot = 지도 전체 프론티어 칸 수 (overflow 판정)
DEV void reach_pick(const SearchResult& r, const Frontier* out, int n_out, int min_cells, int n_tot, ReachSummary* s) {
  reach_clear(s);
  s->start_free = r.start_free;
  s->closed = closed(r) ? 1 : 0;
  s->n_cells = r.n_cells;
  if (n_tot > kReachCap) { s->overflow = 1; return; }
  uint64_t last = 0;
  bool have_last = false;
  for (int k = 0; k < kReachTop; ++k) {
    uint64_t best = ~0ull;
    int bi = -1;
    for (int i = 0; i < n_out; ++i) {
      if (out[i].n < min_cells) continue;
      const uint64_t key = ((uint64_t)(uint32_t)out[i].near_dist << 32) | (uint32_t)out[i].near_cell;
      if (have_last && key <= last) continue;
      if (key < best) { best = key; bi = i; }
    }
    if (bi < 0) break;
    s->top[k].n = out[bi].n;
    s->top[k].dist = out[bi].near_dist;
    s->top[k].cell = out[bi].near_cell;
    last = best;
    have_last = true;
  }
  for (int i = 0; i < n_out; ++i) s->n_found += out[i].n >= min_cells ? 1 : 0;
}
// CPU 참조: frontier_search(dist 켬) 그대로 → reach_pick. b.dist 필요, out 은 max_out ≥ 덩어리 수 (w·h / 1 이면 늘 충분)
inline void reach_summary(const MapView& m, int rx, int ry, Robot rb, int min_cells, SearchBuf b, Frontier* out, int max_out, int n_tot,
                          ReachSummary* s) {
  const SearchResult r = frontier_search(m, rx, ry, rb, 1, b, out, max_out);   // 문턱 1: 모든 덩어리를 out 에 (문턱은 reach_pick 에서)
  reach_pick(r, out, r.n_found < max_out ? r.n_found : max_out, min_cells, n_tot, s);
}

// ---------------------------------------------------------------------------------------------------------------
// 계층형 후보 (mdp/README "계층형", obs_layout.h mdp::hier): 번진 곳 안 덩어리를 크기와 무관하게(hier::kCandMinCells) 위 reach_pick 과 같은
// 열쇠 (BFS 거리 << 32 | 칸) 짧은 순으로 hier::K 개. 넘침 규칙: K 개보다 많으면 가까운 K 개만(먼 덩어리는 로봇이 다가가면 들어옴),
// 지도 전체 프론티어 칸 > kReachCap 이면 비움(reach_pick 과 같은 규칙). GPU 는 reach_block 이 같은 값을 블록 병렬로 (env_verify 가 비트 비교).
struct CandTop {
  int32_t n;      // 덩어리 칸 수 (0 = 빈 칸)
  int32_t dist;   // BFS 거리 (칸)
  int32_t cell;   // 덩어리에서 BFS 거리가 가장 짧은 칸 (y · w + x) = 경로 목표
  int32_t cur;    // 1: 지금 목표 칸(goal)이 이 덩어리 안
  int32_t util;   // 주변 프론티어 칸 수 (cand_util)
};
struct CandSummary {
  int32_t n;      // 채운 후보 수
  CandTop c[hier::K];
};
DEV void cand_clear(CandSummary* s) {
  s->n = 0;
  for (int k = 0; k < hier::K; ++k) { s->c[k].n = 0; s->c[k].dist = -1; s->c[k].cell = -1; s->c[k].cur = 0; s->c[k].util = 0; }
}
// CPU 참조: frontier_search(문턱 1, dist·label 켬) 결과에서. key2[i] = 덩어리 i 의 칸 중 BFS 거리 ≥ hier::kCandMinDist 인 칸의 (거리 << 32 | 칸) 최솟값
// (cand_keys, 없으면 ~0 — 후보 아님). goal = 지금 목표 칸 (없으면 −1), flag·label = 그 찾기의 버퍼
DEV void cand_pick(const Frontier* out, const uint64_t* key2, int n_out, int n_tot, int goal, const uint8_t* flag, const int32_t* label, CandSummary* s) {
  cand_clear(s);
  if (n_tot > kReachCap) return;
  const int glab = (goal >= 0 && (flag[goal] & 2)) ? label[goal] : -1;
  uint64_t last = 0;
  for (int k = 0; k < hier::K; ++k) {
    uint64_t best = ~0ull;
    int bi = -1;
    for (int i = 0; i < n_out; ++i) {
      if (out[i].n < hier::kCandMinCells || key2[i] == ~0ull) continue;
      const uint64_t key = key2[i];
      if (k > 0 && key <= last) continue;
      if (key < best) { best = key; bi = i; }
    }
    if (bi < 0) break;
    s->c[k].n = out[bi].n;
    s->c[k].dist = (int32_t)(best >> 32);
    s->c[k].cell = (int32_t)(best & 0xffffffffull);
    s->c[k].cur = bi == glab ? 1 : 0;
    s->n = k + 1;
    last = best;
  }
}
// 덩어리마다 후보 열쇠 (위 cand_pick 의 key2): 번진 칸(qa[0 … r.n_reach)) 중 덩어리로 묶인 프론티어 칸이고 거리 ≥ kCandMinDist 인 칸
DEV void cand_keys(const SearchResult& r, const SearchBuf& b, int n_out, uint64_t* key2) {
  for (int i = 0; i < n_out; ++i) key2[i] = ~0ull;
  for (int q = 0; q < r.n_reach; ++q) {
    const int idx = b.qa[q];
    if (!(b.flag[idx] & 2) || b.dist[idx] < hier::kCandMinDist) continue;
    const int l = b.label[idx];
    if (l < 0 || l >= n_out) continue;
    const uint64_t key = ((uint64_t)(uint32_t)b.dist[idx] << 32) | (uint32_t)idx;
    if (key < key2[l]) key2[l] = key;
  }
}
// 주변 프론티어 칸 수: 칸 (cx, cy) 둘레 ±R 칸 네모(맵 안) 안의 frontier_cell 수. GPU 는 프론티어 칸 비트 지도(fmap)를 세어 같은 값
DEV int cand_util_radius(float res) { return (int)(hier::kUtilRadius / res + 0.5f); }
DEV int cand_util(const MapView& m, int cx, int cy, int R) {
  int n = 0;
  for (int y = cy - R; y <= cy + R; ++y) {
    if (y < 0 || y >= m.h) continue;
    for (int x = cx - R; x <= cx + R; ++x)
      if (x >= 0 && x < m.w && frontier_cell(m, x, y)) ++n;
  }
  return n;
}


DEV bool coarse_closed(const MapView& m, int rx, int ry, float res, int f, float radius, CoarseBuf cb) {
  const int cw = coarse_dim(m.w, f), ch = coarse_dim(m.h, f);
  for (int cy = 0; cy < ch; ++cy)
    for (int cx = 0; cx < cw; ++cx) {
      bool occ = false, unk = true;
      for (int yy = cy * f; yy < cy * f + f && yy < m.h; ++yy)
        for (int xx = cx * f; xx < cx * f + f && xx < m.w; ++xx) {
          const uint8_t k = cls_at(m, xx, yy);
          occ = occ || k == kOcc;
          unk = unk && k == kUnk;
        }
      cb.L[cy * cw + cx] = occ ? (int16_t)1000 : (unk ? kUnknownL : (int16_t)-1000);
    }
  const MapView cv{cb.L, cw, cw, ch};
  const SearchResult r = frontier_search(cv, rx < 0 ? -1 : rx / f, ry < 0 ? -1 : ry / f, robot_cells(radius, res * (float)f), 1 << 30,
                                         SearchBuf{cb.flag, cb.qa, cb.qb}, nullptr, 0);
  return closed(r);
}

}  // namespace mdp
