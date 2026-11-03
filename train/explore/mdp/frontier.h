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
//   - RL 관측 "프론티어 요약"(mdp/README: 가장 가까운 프론티어 거리·방향, 프론티어 칸 수): frontier_cell + summary_key — 칸마다 따로라
//     GPU 에서 칸 하나 = 스레드 하나로 돌리고 atomicAdd·atomicMin 으로 합친다
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
// res(m) 칸 지도에서 반경 radius(m) → Robot (호스트에서 한 번)
inline Robot robot_cells(float radius, float res) {
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
// RL 관측 프론티어 요약: 프론티어 칸 수 + 기준 칸(로봇)에서 가장 가까운 프론티어 칸.
// 칸마다 summary_key 를 만들어 가장 작은 것을 고르면(atomicMin) 거리² 가 작고, 같으면 칸 번호가 작은 칸 — 순서와 무관해 결정적.
// 이 요약은 도달 가능 여부를 안 본다(지도 전체 프론티어 칸). 닫힘·기준선은 frontier_search(설 수 있는 칸으로 이어진 것만).
constexpr uint64_t kNoFrontier = ~0ull;
DEV uint64_t summary_key(const MapView& m, int x, int y, int rx, int ry) {
  if (!frontier_cell(m, x, y)) return kNoFrontier;
  const int dx = x - rx, dy = y - ry;
  const uint32_t d2 = (uint32_t)(dx * dx + dy * dy);
  return ((uint64_t)d2 << 32) | (uint32_t)(y * m.w + x);
}
struct FrontierSummary {
  int32_t n_cells;      // 프론티어 칸 수
  uint64_t nearest;     // summary_key 최솟값 (kNoFrontier = 없음). 칸 = (nearest & 0xffffffff) → (% w, / w), 거리² = nearest >> 32 (칸²)
};

// ---------------------------------------------------------------------------------------------------------------
// 닫힘 + 프론티어 덩어리: 로봇 칸 (rx, ry) 에서 설 수 있는 칸으로 이어진 프론티어 덩어리들.
// 덩어리 하나 — 모든 값이 칸 단위 정수(결정적). 세계 좌표는 쓰는 쪽에서: 중심 = 원점 + (sx / n + ½) · res
struct Frontier {
  int32_t n;            // 칸 수
  int64_t sx, sy;       // 칸 x·y 합 (중심 = 합 / n)
  int32_t min_d2;       // 로봇 칸에서 가장 가까운 칸까지 거리² (칸²)
  int32_t middle;       // 그 칸 번호 (y · w + x)
  int32_t initial;      // 처음 찾은 칸 번호
};

// 버퍼(쓰는 쪽이 w·h 칸씩): flag(u8), qa·qb(int32). 장치에서는 환경마다 따로.
struct SearchBuf {
  uint8_t* flag;        // bit0 번진 칸(설 수 있음), bit1 덩어리로 묶임, bit2 nearestCell 지나감
  int32_t* qa;          // nearestCell·번지기 큐
  int32_t* qb;          // 덩어리 BFS 큐
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

namespace detail {
// buildNewFrontier (원본 97-170): 8 이웃으로, 번진 칸 안의 프론티어 칸을 묶음. 첫 칸도 합·거리에 넣는다(위 "원본과 다른 점")
DEV Frontier build(const MapView& m, int init, int rx, int ry, SearchBuf b) {
  Frontier f;
  f.n = 0; f.sx = 0; f.sy = 0; f.min_d2 = 0x7fffffff; f.middle = init; f.initial = init;
  int head = 0, tail = 0;
  b.qb[tail++] = init;
  while (head < tail) {
    const int idx = b.qb[head++];
    const int x = idx % m.w, y = idx / m.w;
    f.n += 1; f.sx += x; f.sy += y;
    const int dx0 = x - rx, dy0 = y - ry, d2 = dx0 * dx0 + dy0 * dy0;
    if (d2 < f.min_d2) { f.min_d2 = d2; f.middle = idx; }
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
    const Frontier f = detail::build(m, idx, rx, ry, b);
    if (f.n >= min_cells) {
      if (r.n_found < max_out) out[r.n_found] = f;
      r.n_found += 1;
    } else {
      r.n_small += 1;
    }
  }
  return r;
}

}  // namespace mdp
