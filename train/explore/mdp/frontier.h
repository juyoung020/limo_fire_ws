// 지도 닫힘·프론티어 — 호스트·장치 공용(sim/include/detmath.h 의 결정적 규칙: 정수만 쓰고, 순서가 정해진 순차 BFS 이거나
// 순서와 무관한 정수 합치기(합·최솟값)). 같은 지도면 CPU 와 GPU 가 비트까지 같은 결과를 낸다(baselines/frontier/tools/frontier_check).
//
// 정의 (mdp/README "덮음·닫힘·끝", 2026-11-04 정함). 높이는 라이다 평면(0.111 m) 2D 지도 그대로 — 깊이 카메라는 안 씀.
//   칸 셋: 모름 = 한 번도 안 건드린 칸(빔이 지나가지도 맞지도 않음, sim::kUnknown) · 막힘 = p > 0.65(같은 칸 약 4 번 맞음) ·
//          갈 수 있음 = 그 밖에 건드린 칸. 빈 칸 확정 문턱(p < 0.196, 지나감 36 번)은 쓰지 않는다 — 그 문턱이 95 % 뒤 긴 꼬리의 원인이었다.
//   로봇 중심이 설 수 있는 칸 = 갈 수 있음이고 반지름(로봇 반경) 안에 막힘이 없는 칸 (막힘을 로봇 반경만큼 팽창).
//   프론티어 칸 = 갈 수 있음 칸 중 모름에 4 이웃으로 닿은 것.
//   닫힘 = 로봇 칸(에서 가장 가까운 설 수 있는 칸)부터 설 수 있는 칸을 4 이웃 flood fill 해서 프론티어 칸에 하나도 안 닿음 (기준선·옛 정의).
//   sim 성공 판정의 닫힘 규칙 (2026-11-10, mdp/README "닫힘" — SearchBuf::vis 를 주면 켬):
//     (1) 번지기 = 계획기가 설 수 있는 칸(반경 rbc + 깊이 카메라 층 막힘도 팽창) — 로봇 시작 칸에서, 그리고 판 동안 지나간 칸(씨앗)에서.
//         막힘 아닌 지나간 칸은 설 수 없어도 지나감(몸통이 실제로 지나간 칸) — 문틀·가구 다리 사이에 선 로봇의 작은 주머니에 갇히지 않게.
//     (2) close_min 칸 미만 덩어리는 안 막음 (explore_lite min_frontier_size 와 같은 꼴의 크기 문턱, 값은 측정으로 4 칸).
//     (3) 본 칸: 프론티어 칸 옆 모름 칸이 모두 라이다 시야 부채꼴 안·센서에서 close_look 안에 든 적이 있거나(가까이서 봐도 모름 = 가려짐·빔 빠짐)
//         몸통이 지나간 칸이면 안 막음.
//     닫힘 = 시작 칸이 있고 막는 칸(n_block)이 0. 계층형 후보도 같은 칸만(cand_keys).
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

// 칸 비트 판 (줄마다 낱말 ws 개, 칸 x 는 낱말 x / 32 의 비트 x % 32 — sim fmap 과 같은 꼴)
DEV bool plane_bit(const uint32_t* P, int ws, int x, int y) { return (P[y * ws + (x >> 5)] >> (x & 31)) & 1u; }
// 칸 바이트 판 (sim cam 층 꼴: 칸마다 한 바이트, 줄 stride) 이 (x, y) 둘레 반경 rb 원판 안에 0 아닌 칸이 있나 (맵 밖은 안 봄)
DEV bool byte_near(const uint8_t* B, int stride, const MapView& m, int x, int y, Robot rb) {
  for (int dy = -rb.r; dy <= rb.r; ++dy)
    for (int dx = -rb.r; dx <= rb.r; ++dx) {
      if (dx * dx + dy * dy > rb.r2) continue;
      const int u = x + dx, v = y + dy;
      if (u >= 0 && v >= 0 && u < m.w && v < m.h && B[v * stride + u]) return true;
    }
  return false;
}
// 본 프론티어 칸 (SearchBuf::look): 4 이웃 모름 칸(맵 안)이 모두 look 비트 — 가까이서 라이다 시야에 들었는데도 모름으로 남은 칸 —
// 이거나 vis 비트(몸통 가운데가 지나간 칸: 라이다가 제 몸 밑은 못 보지만 몸통이 그 자리에 있었으니 빈 곳 — Nav2 footprint_clearing 과 같은 뜻)
DEV bool looked_cell(const MapView& m, const uint32_t* look, const uint32_t* vis, int ws, int x, int y) {
  for (int k = 0; k < 4; ++k) {
    int dx, dy;
    nb4(k, &dx, &dy);
    const int u = x + dx, v = y + dy;
    if (u < 0 || v < 0 || u >= m.w || v >= m.h || cls_at(m, u, v) != kUnk) continue;
    if (!plane_bit(look, ws, u, v) && !(vis && plane_bit(vis, ws, u, v))) return false;
  }
  return true;
}
// ---------------------------------------------------------------------------------------------------------------
// 닫힘 + 프론티어 덩어리: 로봇 칸 (rx, ry) 에서 설 수 있는 칸으로 이어진 프론티어 덩어리들.
// 덩어리 하나 — 모든 값이 칸 단위 정수(결정적). 세계 좌표는 쓰는 쪽에서: 중심 = 원점 + (sx / n + ½) · res
struct Frontier {
  int32_t n;            // 칸 수
  int32_t n_open;       // 그중 닫힘을 막을 수 있는 칸: 안 본(looked_cell 아님) 칸 — 본 칸 판이 없으면 n
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
  // 닫힘 규칙 (2026-11-10, mdp/README "닫힘"). 기본값이면 옛 정의(로봇 칸에서만 번짐, 번진 프론티어 칸 하나라도 있으면 안 닫힘)
  const uint32_t* vis = nullptr;    // 닫힘 규칙을 켬 + 지나간 칸 비트 판. 켜면 번지기 칸 = 계획기가 설 수 있는 칸(plan_ok: 반경 rbc, 지도 막힘·cam 둘 다 팽창)
                                    // — 로봇 시작 칸(nearestCell, 반경 rb)에서 이 칸으로 번진 곳이 BFS 거리·후보 쪽(qa[0 … n_reach0)), 그 위에
                                    // 지나간 칸 씨앗 + 막힘 아닌 지나간 칸(몸통이 실제로 지나감)으로 더 번진 곳(dist −1)까지가 닫힘 쪽
  const uint32_t* look = nullptr;   // 본 칸 비트 판 (sim EnvBatch::trail 판 1 — 라이다 시야 부채꼴 안·센서에서 close_look 안에 들었던 칸).
                                    // 프론티어 칸 옆 모름 칸(4 이웃)이 모두 본 칸이면(가까이서 시야에 들고도 모름 = 가려져 못 봄) 닫힘을 안 막음
  int tws = 0;                      // vis·look 판 한 줄 낱말 수
  int close_min = 1;                // 이 칸 수 미만 덩어리는 닫힘을 안 막음
  Robot rbc{-1, -1};                // vis 가 있을 때 번지기의 몸통 반경 = 계획기 반경 (r < 0 이면 rb)
  const uint8_t* cam = nullptr;     // vis 가 있을 때 계획기가 더 막는 칸 (sim 깊이 카메라 층 — 칸마다 한 바이트, 줄 = MapView::stride, 0 아님 = 막힘)
};

struct SearchResult {
  int32_t n_found;      // min_cells 이상 덩어리 수 (max_out 보다 많으면 앞의 max_out 개만 out 에)
  int32_t n_small;      // min_cells 미만이라 버린 덩어리 수
  int32_t n_cells;      // 번진 칸 안 프론티어 칸 수 (덩어리 크기와 무관하게 모두)
  int32_t n_reach;      // 번진(설 수 있는) 칸 수
  int32_t start;        // 번지기 시작 칸 (가장 가까운 설 수 있는 칸, 없으면 로봇 칸)
  int32_t start_free;   // 1 = 설 수 있는 칸에서 시작함 (0 이면 아직 설 수 있는 칸이 없음 — 시작 직후)
  int32_t n_reach0;     // 그중 로봇 시작 칸에서 번진 칸 수 (qa 앞부분 — BFS 거리가 있는 칸). 지나간 칸 씨앗이 없으면 n_reach
  int32_t n_block;      // 닫힘을 막는 칸 수 = close_min 이상 덩어리의 n_open 합
};

// 닫힘: 설 수 있는 칸에서 번졌고 프론티어 칸에 하나도 안 닿음
DEV bool closed(const SearchResult& r) { return r.start_free != 0 && r.n_block == 0; }

namespace detail {
// buildNewFrontier (원본 97-170): 8 이웃으로, 번진 칸 안의 프론티어 칸을 묶음. 첫 칸도 합·거리에 넣는다(위 "원본과 다른 점")
DEV Frontier build(const MapView& m, int init, int rx, int ry, Robot rb, SearchBuf b, int lab) {
  Frontier f;
  f.n = 0; f.n_open = 0; f.sx = 0; f.sy = 0; f.min_d2 = 0x7fffffff; f.middle = init; f.initial = init; f.near_dist = -1; f.near_cell = -1;
  int head = 0, tail = 0;
  b.qb[tail++] = init;
  while (head < tail) {
    const int idx = b.qb[head++];
    const int x = idx % m.w, y = idx / m.w;
    f.n += 1; f.sx += x; f.sy += y;
    if (!(b.look && looked_cell(m, b.look, b.vis, b.tws, x, y))) f.n_open += 1;
    if (b.label) b.label[idx] = lab;
    const int dx0 = x - rx, dy0 = y - ry, d2 = dx0 * dx0 + dy0 * dy0;
    if (d2 < f.min_d2) { f.min_d2 = d2; f.middle = idx; }
    if (b.dist) {
      const int bd = b.dist[idx];
      if (bd >= 0 && (f.near_dist < 0 || bd < f.near_dist || (bd == f.near_dist && idx < f.near_cell))) { f.near_dist = bd; f.near_cell = idx; }
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
  r.n_found = 0; r.n_small = 0; r.n_cells = 0; r.n_reach = 0; r.start = -1; r.start_free = 0; r.n_reach0 = 0; r.n_block = 0;
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
  // 번지기: 설 수 있는 칸, 4 이웃 (닫힘 판정과 같은 flood fill). 닫힘 규칙(b.vis)이면 계획기가 설 수 있는 칸 (plan_ok)
  const Robot rc = b.rbc.r < 0 ? rb : b.rbc;
  const auto plan_ok = [&](int x, int y) {
    if (!b.vis) return standable(m, x, y, rb);
    return standable(m, x, y, rc) && !(b.cam && byte_near(b.cam, m.stride, m, x, y, rc));
  };
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
      if ((b.flag[j] & 1) || !plan_ok(u, v)) continue;
      b.flag[j] |= 1;
      if (b.dist) b.dist[j] = b.dist[idx] + 1;
      b.qa[tail++] = j;
    }
  }
  r.n_reach0 = tail;
  // 지나간 칸 (닫힘 규칙): 판 동안 몸통 가운데가 지나간 칸에서도 번지고, 막힘이 아닌 지나간 칸은 설 수 없어도 지나간다 — 로봇이 문틀·가구 다리 사이처럼
  // 좁은 곳에 서 있거나 지나와서(지도 막힘 팽창으로 설 수 있는 칸이 끊김) 그 둘레 작은 주머니만 번지는 일이 없게. 결과는 집합이라 차례와 무관
  // (로봇 시작 칸 쪽 설 수 있는 칸 성분 위에 다시 번짐 + 씨앗). 거리는 위 로봇 시작 칸 BFS 만 (여기서 더한 칸은 −1)
  if (b.vis) {
    const auto pass = [&](int x, int y) {
      return plan_ok(x, y) || (plane_bit(b.vis, b.tws, x, y) && cls_at(m, x, y) != kOcc && !(b.cam && b.cam[y * m.stride + x]));
    };
    for (int i = 0; i < n; ++i) {
      const int x = i % m.w, y = i / m.w;
      if ((b.flag[i] & 1) || !plane_bit(b.vis, b.tws, x, y) || !pass(x, y)) continue;
      b.flag[i] |= 1;
      if (b.dist) b.dist[i] = -1;
      b.qa[tail++] = i;
    }
    head = 0;
    while (head < tail) {
      const int idx = b.qa[head++];
      const int x = idx % m.w, y = idx / m.w;
      for (int k = 0; k < 4; ++k) {
        int dx, dy;
        nb4(k, &dx, &dy);
        const int u = x + dx, v = y + dy;
        if (u < 0 || v < 0 || u >= m.w || v >= m.h) continue;
        const int j = v * m.w + u;
        if ((b.flag[j] & 1) || !pass(u, v)) continue;
        b.flag[j] |= 1;
        if (b.dist) b.dist[j] = -1;
        b.qa[tail++] = j;
      }
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
    const Frontier f = detail::build(m, idx, rx, ry, rb, b, r.n_found + r.n_small);
    if (f.n >= b.close_min) r.n_block += f.n_open;
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
  s->closed = closed(r) && n_tot <= kReachCap ? 1 : 0;   // 넘침이면 덩어리를 안 셈 → 안 닫힘으로 (GPU reach_block 과 같은 규칙)
  s->n_cells = r.n_cells;
  if (n_tot > kReachCap) { s->overflow = 1; return; }
  uint64_t last = 0;
  bool have_last = false;
  for (int k = 0; k < kReachTop; ++k) {
    uint64_t best = ~0ull;
    int bi = -1;
    for (int i = 0; i < n_out; ++i) {
      if (out[i].n < min_cells || out[i].near_dist < 0) continue;   // 지나간 칸 씨앗에서만 번진 덩어리는 로봇에서 BFS 로 안 닿음
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
// 덩어리마다 후보 열쇠 (위 cand_pick 의 key2): 로봇 시작 칸에서 번진 칸(qa[0 … r.n_reach0)) 중 덩어리로 묶인 프론티어 칸이고 거리 ≥ kCandMinDist 이며
// 닫힘을 막을 수 있는 칸(덩어리 칸 수 ≥ close_min, 자격 있음, 안 봄 — 닫힘 규칙과 같은 칸)
DEV void cand_keys(const MapView& m, Robot rb, const SearchResult& r, const SearchBuf& b, const Frontier* out, int n_out, uint64_t* key2) {
  for (int i = 0; i < n_out; ++i) key2[i] = ~0ull;
  for (int q = 0; q < r.n_reach0; ++q) {
    const int idx = b.qa[q];
    if (!(b.flag[idx] & 2) || b.dist[idx] < hier::kCandMinDist) continue;
    const int l = b.label[idx];
    if (l < 0 || l >= n_out || out[l].n < b.close_min) continue;
    const int x = idx % m.w, y = idx / m.w;
    if (b.look && looked_cell(m, b.look, b.vis, b.tws, x, y)) continue;
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

// ---- 후보 v2 (hier::kCand2…, EnvCfg::cand_v2). 기존 규칙의 세 약점을 고침 — (a) 빔 틈 조각이 저마다 후보가 돼 한곳에 몰림 (b) 가까운 순으로만 잘라 먼 큰 덩어리가 빠짐
// (c) "가면 얼마나 새로 보이나" 가 없음. 열쇠·자격(cand_keys)은 그대로이고 고르는 법만 다르다. 정수·순서 무관(CPU 순차 = GPU 블록 병렬):
//   자리 k = 0 … K − 1: 남은 덩어리 중 차례 값(cand2_order)이 가장 작은 것 — 앞 K − kCand2Far 자리는 열쇠(가까운 순), 뒤 kCand2Far 자리는 칸 수 큰 순(같으면 칸 번호).
//   고른 덩어리와 가까운(cand2_near) 남은 덩어리는 모두 그 후보에 합침: 칸 수를 더하고 목록에서 뺌 (지금 목표 덩어리가 합쳐지면 cur = 1).
// 0 번 자리는 기존과 같은 칸(가장 가까운 덩어리의 가장 가까운 칸). 다만 "지금 목표" 표시가 합쳐진 덩어리까지 넓어져 경로를 이어 쓰는 때(hier_path.h act_same)가 달라지므로
// 늘 0 번을 고르는 규칙(slot0)의 판 결과는 비트까지 같지는 않다(시험 맵 256 판 거리 차 1 % 안 — mdp/README "후보 v2").
DEV int cand2_merge_cells(float res) { return (int)(hier::kCand2Merge / res + 0.5f); }
DEV uint64_t cand2_order(uint64_t key, int n, bool far) { return far ? (((uint64_t)(0xffffffffu - (uint32_t)n) << 32) | (key & 0xffffffffull)) : key; }
DEV bool cand2_near(uint64_t a, uint64_t b, int w, int Rm) {
  const int ca = (int)(a & 0xffffffffull), cb = (int)(b & 0xffffffffull);
  const int dx = ca % w - cb % w, dy = ca / w - cb / w, dd = (int)(a >> 32) - (int)(b >> 32);
  return dx * dx + dy * dy <= Rm * Rm && (dd < 0 ? -dd : dd) <= 2 * Rm;
}
// CPU 참조 (GPU: reach_block 의 v2 고르기). key2 는 cand_keys 가 낸 것 — 여기서 지워 가며 씀
DEV void cand_pick2(const Frontier* out, uint64_t* key2, int n_out, int n_tot, int goal, const uint8_t* flag, const int32_t* label, int w, int Rm, CandSummary* s) {
  cand_clear(s);
  if (n_tot > kReachCap) return;
  const int glab = (goal >= 0 && (flag[goal] & 2)) ? label[goal] : -1;
  for (int k = 0; k < hier::K; ++k) {
    const bool far = k >= hier::K - hier::kCand2Far;
    uint64_t best = ~0ull;
    int bi = -1;
    for (int i = 0; i < n_out; ++i) {
      if (key2[i] == ~0ull || out[i].n < hier::kCandMinCells) continue;
      const uint64_t o = cand2_order(key2[i], out[i].n, far);
      if (o < best) { best = o; bi = i; }
    }
    if (bi < 0) break;
    const uint64_t key = key2[bi];
    int n = 0, cur = 0;
    for (int i = 0; i < n_out; ++i) {
      if (key2[i] == ~0ull || out[i].n < hier::kCandMinCells || !cand2_near(key2[i], key, w, Rm)) continue;   // 고른 덩어리 자신도 여기서
      n += out[i].n;
      cur |= i == glab ? 1 : 0;
      key2[i] = ~0ull;
    }
    s->c[k].n = n;
    s->c[k].dist = (int32_t)(key >> 32);
    s->c[k].cell = (int32_t)(key & 0xffffffffull);
    s->c[k].cur = cur;
    s->n = k + 1;
  }
}
// 예상 관측량 (후보 v2 의 CandTop::util): 칸 (cx, cy) 에서 광선 a 를 한 칸씩 R 칸까지 — 막힘 칸에서 끝, 모름 칸마다 그 거리 t 를 더함(호 길이 ∝ t 라 합이 넓이에 비례).
// 광선 kGainRays 개의 합 / cand_gain_norm = 시선이 닿는 모름 넓이 / 반지름 R 원 넓이. 선행연구와의 차이: ARiADNE·HEADER 는 시선이 닿는 프론티어 칸 수, FUEL 은 시점에서 보이는
// 프론티어 칸 수, DSVP 는 시야 안 모름 복셀 수 — 여기는 DSVP 쪽(모름 넓이)이고 시야는 220° 가 아니라 360° (닿을 때 머리 방향을 후보 칸만으론 모름 — 규칙 정책 쪽에서 220° 로 잼)
DEV int cand_gain_cells(float max_range, float res) { return (int)(hier::kGainRange * max_range / res); }
DEV int cand_gain_norm(int R) { return hier::kGainRays * R * (R + 1) / 2; }
DEV int cand_gain_ray(const MapView& m, int cx, int cy, int a, int R) {
  // 광선 방향 cos(2π(a + 0.5) / 64) × 16384 — 정수 표 (CPU·GPU 가 같은 값을 내게). sin 은 16 칸 앞 (hier::kGainRays = 64 에 맞춘 표)
  const int16_t T[64] = {16364,16207,15893,15426,14811,14053,13160,12140,11003,9760,8423,7005,5520,3981,2404,804,-804,-2404,-3981,-5520,-7005,-8423,-9760,-11003,-12140,-13160,-14053,-14811,-15426,-15893,-16207,-16364,-16364,-16207,-15893,-15426,-14811,-14053,-13160,-12140,-11003,-9760,-8423,-7005,-5520,-3981,-2404,-804,804,2404,3981,5520,7005,8423,9760,11003,12140,13160,14053,14811,15426,15893,16207,16364};
  static_assert(hier::kGainRays == 64, "광선 표는 64 방향");
  const int C = T[a], S = T[(a + 48) & 63];
  const auto rnd = [](int v) { v += 8192; return v >= 0 ? v >> 14 : -((-v + 16383) >> 14); };   // 내림 나눗셈 (음수 옮김은 구현마다 다를 수 있어 따로)
  int g = 0;
  for (int t = 1; t <= R; ++t) {
    const int x = cx + rnd(t * C), y = cy + rnd(t * S);
    if (x < 0 || y < 0 || x >= m.w || y >= m.h) break;
    const uint8_t cl = cls_at(m, x, y);
    if (cl == kOcc) break;
    if (cl == kUnk) g += t;
  }
  return g;
}

}  // namespace mdp
