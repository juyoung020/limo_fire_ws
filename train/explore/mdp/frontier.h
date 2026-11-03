// 프론티어(아는 빈 칸과 모르는 칸의 경계) 찾기 — 호스트·장치 공용(sim/include/detmath.h 의 결정적 규칙: 정수만 쓰고, 순서가 정해진
// 순차 BFS 이거나 순서와 무관한 정수 합치기(합·최솟값)). 같은 지도면 CPU 와 GPU 가 비트까지 같은 결과를 낸다(baselines/frontier/tools/frontier_check).
//
// 쓰는 곳
//   - 프론티어 탐사 기준선(baselines/frontier): frontier_search — explore_lite 의 찾기를 그대로(아래 출처)
//   - RL 관측 "프론티어 요약"(mdp/README: 가장 가까운 프론티어 거리·방향, 프론티어 칸 수): frontier_cell + summary_key — 칸마다 따로라
//     GPU 에서 칸 하나 = 스레드 하나로 돌리고 atomicAdd·atomicMin 으로 합친다
//
// 지도: 시뮬 지도(sim/include/env.h EnvBatch::lmap) 그대로 — 로그 오즈 × 1000(int16), 아직 모름 = −32768(sim::kUnknown).
//
// 출처: m-explore(hrnr/m-explore) 커밋 712bdd41027b645c9c876a4c0071478f090825ea
//   explore/src/frontier_search.cpp:27-95   searchFrom — 로봇 칸에서 가장 가까운 빈 칸부터 4 이웃 BFS, 값이 같거나 낮은 칸으로만 퍼짐(70),
//                                           그 밖의 이웃이 새 프론티어 칸이면 덩어리를 만듦(75-81), 크기 × 해상도 ≥ min_frontier_size 만 남김(78)
//   explore/src/frontier_search.cpp:97-170  buildNewFrontier — 8 이웃 BFS 로 프론티어 칸을 묶고 크기·중심·로봇에서 가장 가까운 칸(middle)을 셈
//   explore/src/frontier_search.cpp:172-189 isNewFrontierCell — 모르는 칸이고, 아직 표시 안 됐고, 4 이웃에 빈 칸이 있음
//   explore/include/explore/costmap_tools.h:91-  nearestCell — 4 이웃 BFS 로 가장 가까운 빈 칸
// 원본과 다른 점(의도한 것)
//   - 원본은 첫 칸(initial)을 크기에는 넣고(105) 중심 합·최소 거리에는 안 넣는다(145-158) → 중심이 세계 원점 쪽으로 치우친다. 여기선 첫 칸도 넣는다.
//   - 원본 costmap 값(FREE 0 < … < LETHAL 254 < NO_INFORMATION 255)은 칸 종류 셋으로: kFree 0 · kOcc 254 · kUnk 255 (같은 순서라 70 줄의 "≤" 가 그대로 뜻이 맞음).
#pragma once
#include <cstdint>

#include "detmath.h"

namespace mdp {

// ---------------------------------------------------------------------------------------------------------------
// 칸 종류: 아는 빈 칸 · 막힌 칸 · 모름
//
// 문턱은 Cartographer 가 지도를 ROS 지도 파일로 쓸 때의 확률 문턱과 같게 한다:
//   cartographer_ros(커밋 c138034db0c47fe0ea5a2abe516acae02190dbf5) cartographer_ros/cartographer_ros/ros_map.cc:44-45
//     "occupied_thresh: 0.65 / free_thresh: 0.196" ("Magic constants taken directly from ros map_saver code")
//   map_server 는 p > occupied_thresh 를 막힘, p < free_thresh 를 빈 칸, 그 사이를 모름(-1)으로 읽는다 — 즉 "저장한 지도에서 빈 칸·벽으로 보이는 칸".
//   한 번도 관측 안 된 칸(sim::kUnknown)은 Cartographer /map 의 -1 과 같다(msg_conversion.cc:407-410, observed == 0 → -1).
// 로그 오즈 × 1000 정수로: 1000·ln(0.65/0.35) = 619.04 → p > 0.65 ⇔ l ≥ 620,  1000·ln(0.196/0.804) = −1411.54 → p < 0.196 ⇔ l ≤ −1412.
// 시뮬 갱신값(env.yaml map: hit 0.55 → +201, miss 0.49 → −40)으로는 막힘 = 맞음 4 번, 빈 칸 = 지나감 36 번(한 장에 칸마다 한 번).
// 빈 칸 쪽이 엄격하다(멀리서 한두 번 스친 칸은 아직 "모름") — 이 값이 바뀌면 frontier_check 가 상수와 식을 다시 맞춰 본다.
constexpr int16_t kUnknownL = -32768;   // sim::kUnknown 과 같아야 함 (frontier_check 가 확인)
constexpr int16_t kOccL = 620;          // l ≥ 이것 → 막힘
constexpr int16_t kFreeL = -1412;       // l ≤ 이것 → 아는 빈 칸
constexpr double kOccP = 0.65, kFreeP = 0.196;   // 위 정수의 근거(확인용)

enum : uint8_t { kFree = 0, kOcc = 254, kUnk = 255 };   // explore_lite costmap 값과 같은 순서

DEV uint8_t cell_class(int16_t l) {
  if (l == kUnknownL) return kUnk;
  if (l >= kOccL) return kOcc;
  if (l <= kFreeL) return kFree;
  return kUnk;   // 관측했지만 문턱 사이 → 저장한 지도에서 모름
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

// 프론티어 칸: 모르는 칸이고 4 이웃에 아는 빈 칸이 있음 (isNewFrontierCell 의 "아직 표시 안 됨" 빼고 — 칸마다 따로 판정)
DEV bool frontier_cell(const MapView& m, int x, int y) {
  if (cls_at(m, x, y) != kUnk) return false;
  for (int k = 0; k < 4; ++k) {
    int dx, dy;
    nb4(k, &dx, &dy);
    const int u = x + dx, v = y + dy;
    if (u >= 0 && v >= 0 && u < m.w && v < m.h && cls_at(m, u, v) == kFree) return true;
  }
  return false;
}

// ---------------------------------------------------------------------------------------------------------------
// RL 관측 프론티어 요약: 프론티어 칸 수 + 기준 칸(로봇)에서 가장 가까운 프론티어 칸.
// 칸마다 summary_key 를 만들어 가장 작은 것을 고르면(atomicMin) 거리² 가 작고, 같으면 칸 번호가 작은 칸 — 순서와 무관해 결정적.
// 이 요약은 도달 가능 여부를 안 본다(지도 전체 프론티어 칸). 기준선의 frontier_search 는 빈 칸으로 이어진 것만 본다.
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
// explore_lite searchFrom: 로봇 칸 (rx, ry) 에서 빈 칸으로 이어진 프론티어 덩어리들.
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
  uint8_t* flag;        // bit0 BFS 지나감, bit1 프론티어로 표시, bit2 nearestCell 지나감
  int32_t* qa;          // nearestCell·본 BFS 큐
  int32_t* qb;          // 덩어리 BFS 큐
};

struct SearchResult {
  int32_t n_found;      // min_cells 이상 덩어리 수 (max_out 보다 많으면 앞의 max_out 개만 out 에)
  int32_t n_small;      // min_cells 미만이라 버린 덩어리 수
  int32_t start;        // BFS 시작 칸 (가장 가까운 빈 칸, 없으면 로봇 칸)
  int32_t start_free;   // 1 = 빈 칸에서 시작함 (0 이면 지도에 아는 빈 칸이 아직 없음 — 원본은 경고만 하고 그 칸에서 시작)
};

namespace detail {
// buildNewFrontier (원본 97-170): 8 이웃으로 프론티어 칸을 묶음. 첫 칸도 합·거리에 넣는다(위 "원본과 다른 점")
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
      if ((b.flag[j] & 2) || !frontier_cell(m, u, v)) continue;
      b.flag[j] |= 2;
      b.qb[tail++] = j;
    }
  }
  return f;
}
}  // namespace detail

// min_cells: 원본 min_frontier_size / 해상도 (원본 78 줄: size · res ≥ min_frontier_size)
DEV SearchResult frontier_search(const MapView& m, int rx, int ry, int min_cells, SearchBuf b, Frontier* out, int max_out) {
  SearchResult r;
  r.n_found = 0; r.n_small = 0; r.start = -1; r.start_free = 0;
  const int n = m.w * m.h;
  if (rx < 0 || ry < 0 || rx >= m.w || ry >= m.h) return r;   // 원본 33-36: 로봇이 지도 밖
  for (int i = 0; i < n; ++i) b.flag[i] = 0;
  const int pos = ry * m.w + rx;
  // nearestCell(FREE)
  {
    int head = 0, tail = 0;
    b.qa[tail++] = pos;
    b.flag[pos] |= 4;
    r.start = pos;
    while (head < tail) {
      const int idx = b.qa[head++];
      const int x = idx % m.w, y = idx / m.w;
      if (cls_at(m, x, y) == kFree) { r.start = idx; r.start_free = 1; break; }
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
  // 본 BFS (원본 62-84)
  int head = 0, tail = 0;
  b.qa[tail++] = r.start;
  b.flag[r.start] |= 1;
  while (head < tail) {
    const int idx = b.qa[head++];
    const int x = idx % m.w, y = idx / m.w;
    const uint8_t c = cls_at(m, x, y);
    for (int k = 0; k < 4; ++k) {
      int dx, dy;
      nb4(k, &dx, &dy);
      const int u = x + dx, v = y + dy;
      if (u < 0 || v < 0 || u >= m.w || v >= m.h) continue;
      const int j = v * m.w + u;
      if (cls_at(m, u, v) <= c && !(b.flag[j] & 1)) {
        b.flag[j] |= 1;
        b.qa[tail++] = j;
      } else if (!(b.flag[j] & 2) && frontier_cell(m, u, v)) {
        b.flag[j] |= 2;
        const Frontier f = detail::build(m, j, rx, ry, b);
        if (f.n >= min_cells) {
          if (r.n_found < max_out) out[r.n_found] = f;
          r.n_found += 1;
        } else {
          r.n_small += 1;
        }
      }
    }
  }
  return r;
}

}  // namespace mdp
