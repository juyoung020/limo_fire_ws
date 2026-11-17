// 프론티어 찾기(frontier.h frontier_search)의 칸 판정을 걸음 사이에 기억 (호스트 전용 — 실물 노드 RobotEnv 가 걸음마다 부르는 관측의 찾기).
// 번지기 차례 · BFS 거리 · 덩어리 라벨은 로봇 시작 칸이 걸음마다 바뀌어 그대로 둘 수 없으므로 번지기는 걸음마다 다시 한다. 대신 비싼 칸 판정을 기억한다:
//   plan_ok  (닫힘 규칙 번지기의 몸통 원판 — 칸이 빈 칸이고 반경 rc 안에 지도 막힘 · 깊이 층 막힘이 없음)  → SearchBuf::pc (0 모름 · 1 못 섬 · 2 섬)
//   frontier_cell (빈 칸이고 4 이웃에 모름)                                                                  → SearchBuf::fc (늘 맞는 값)
//   지도 전체 프론티어 칸 수 (obs_ref.h reach_obs 의 n_tot)                                                   → n_front
// 걸음마다 update: 지난번 본 칸 종류(빈 칸 · 막힘 · 모름)와 깊이 층 막힘을 지금 지도와 견줘 바뀐 칸만 찾고, 그 둘레(plan_ok 는 반경 rc 네모, frontier_cell 은
// 그 칸과 4 이웃)만 무효로 하거나 다시 셈. 칸 종류는 아는 칸 상자(지금 것 ∪ 지난 것) 안만 견준다 — 밖은 모두 모름(지도에 쓴 칸은 상자 안, sim::known_box).
// 지도 크기 · 반경 · 깊이 층 유무가 바뀌면 전부 다시 (폴백). check_every > 0 이면 그 횟수마다 기억 전체를 처음부터 다시 센 값과 견주고, 다르면 n_bad 를 세고 전부 다시.
// 값은 frontier_search 가 기억 없이 셀 때와 같다 (같은 함수 · 같은 입력 — 입력이 바뀐 칸은 무효).
#pragma once
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <vector>

#include "frontier.h"

namespace mdp {

struct FrontierCache {
  int w = 0, h = 0, stride = 0;
  Robot rc{-1, -1};
  bool cam_on = false;
  std::vector<uint8_t> cls;     // 지난번 칸 종류 (cell_class 값)
  std::vector<uint8_t> camv;    // 지난번 깊이 층 바이트 그대로 (줄 = stride — 줄 통째 견줌)
  std::vector<uint8_t> pc, fc;  // SearchBuf::pc · fc
  long n_front = 0;             // fc 가 1 인 칸 수
  int bx0 = 0, by0 = 0, bx1 = -1, by1 = -1;   // 지난번 아는 칸 상자
  int check_every = 0;          // > 0: 이 횟수마다 처음부터 다시 센 값과 견줌
  long calls = 0, n_full = 0, n_check = 0, n_bad = 0, n_changed = 0;
  std::vector<int32_t> chg;     // 이번에 바뀐 칸 (일 버퍼)
};

namespace detail {
inline bool fc_at(const MapView& m, int x, int y) { return frontier_cell(m, x, y); }
inline bool pc_at(const MapView& m, const uint8_t* cam, Robot rc, int x, int y) {
  return standable(m, x, y, rc) && !(cam && byte_near(cam, m.stride, m, x, y, rc));
}
inline void cache_full(FrontierCache& C, const MapView& m, const uint8_t* cam, Robot rc, int kx0, int ky0, int kx1, int ky1) {
  const int n = m.w * m.h;
  C.w = m.w; C.h = m.h; C.stride = m.stride; C.rc = rc; C.cam_on = cam != nullptr;
  C.cls.assign(n, kUnk);
  C.camv.assign((size_t)m.stride * m.h, 0);
  C.pc.assign(n, 0);
  C.fc.assign(n, 0);
  C.n_front = 0;
  for (int y = 0; y < m.h; ++y)
    for (int x = 0; x < m.w; ++x) {
      C.cls[y * m.w + x] = cls_at(m, x, y);
      if (cam) C.camv[(size_t)y * m.stride + x] = cam[(size_t)y * m.stride + x];
    }
  for (int y = 0; y < m.h; ++y)
    for (int x = 0; x < m.w; ++x)
      if (C.cls[y * m.w + x] == kFree && fc_at(m, x, y)) { C.fc[y * m.w + x] = 1; ++C.n_front; }
  C.bx0 = kx0; C.by0 = ky0; C.bx1 = kx1; C.by1 = ky1;
  ++C.n_full;
}
// 처음부터 다시 센 값과 견줌 (pc 는 이미 센 칸만)
inline bool cache_ok(const FrontierCache& C, const MapView& m, const uint8_t* cam) {
  long nf = 0;
  for (int y = 0; y < m.h; ++y)
    for (int x = 0; x < m.w; ++x) {
      const int i = y * m.w + x;
      const bool f = fc_at(m, x, y);
      nf += f;
      if ((C.fc[i] != 0) != f) return false;
      if (C.pc[i] && (C.pc[i] == 2) != pc_at(m, cam, C.rc, x, y)) return false;
    }
  return nf == C.n_front;
}
}  // namespace detail

// 지금 지도(m) · 깊이 층(cam, 없으면 nullptr) · 반경(rc)에 맞춰 기억을 고침. (kx0, ky0)–(kx1, ky1) = 지금 아는 칸 상자 (kx1 < 0 = 없음)
inline void cache_update(FrontierCache& C, const MapView& m, const uint8_t* cam, Robot rc, int kx0, int ky0, int kx1, int ky1) {
  ++C.calls;
  if (C.w != m.w || C.h != m.h || C.stride != m.stride || C.rc.r != rc.r || C.rc.r2 != rc.r2 || C.cam_on != (cam != nullptr)) {
    detail::cache_full(C, m, cam, rc, kx0, ky0, kx1, ky1);
    return;
  }
  C.chg.clear();
  // 칸 종류: 지난 상자 ∪ 지금 상자
  int x0 = C.bx1 >= 0 ? C.bx0 : kx0, y0 = C.bx1 >= 0 ? C.by0 : ky0, x1 = C.bx1 >= 0 ? C.bx1 : kx1, y1 = C.bx1 >= 0 ? C.by1 : ky1;
  if (kx1 >= 0) { x0 = std::min(x0, kx0); y0 = std::min(y0, ky0); x1 = std::max(x1, kx1); y1 = std::max(y1, ky1); }
  if (x1 >= 0) {
    x0 = std::max(0, x0); y0 = std::max(0, y0); x1 = std::min(m.w - 1, x1); y1 = std::min(m.h - 1, y1);
    for (int y = y0; y <= y1; ++y)
      for (int x = x0; x <= x1; ++x) {
        const int i = y * m.w + x;
        const uint8_t k = cls_at(m, x, y);
        if (k != C.cls[i]) { C.cls[i] = k; C.chg.push_back(i); }
      }
  }
  const size_t n_cls = C.chg.size();
  // 깊이 층: 줄을 통째 견줘 다른 줄만 칸으로 (바이트가 바뀐 칸 — 막힘 여부가 그대로여도 무효로 할 뿐이라 값은 같음)
  if (cam)
    for (int y = 0; y < m.h; ++y) {
      const uint8_t* a = cam + (size_t)y * m.stride;
      uint8_t* s = C.camv.data() + (size_t)y * m.stride;
      if (std::memcmp(a, s, (size_t)m.w) == 0) continue;
      for (int x = 0; x < m.w; ++x)
        if (a[x] != s[x]) { s[x] = a[x]; C.chg.push_back(-1 - (y * m.w + x)); }   // 음수 = 깊이 층만 바뀜
    }
  C.n_changed += (long)C.chg.size();
  // 무효 · 다시 셈
  const int R = rc.r;
  for (size_t q = 0; q < C.chg.size(); ++q) {
    const int i = C.chg[q] >= 0 ? C.chg[q] : -1 - C.chg[q];
    const int x = i % m.w, y = i / m.w;
    for (int v = std::max(0, y - R); v <= std::min(m.h - 1, y + R); ++v)
      std::memset(&C.pc[v * m.w + std::max(0, x - R)], 0, (size_t)(std::min(m.w - 1, x + R) - std::max(0, x - R) + 1));
    if (q < n_cls) {   // 칸 종류가 바뀜 → 그 칸과 4 이웃의 frontier_cell
      for (int k = -1; k < 4; ++k) {
        int dx = 0, dy = 0;
        if (k >= 0) nb4(k, &dx, &dy);
        const int u = x + dx, w2 = y + dy;
        if (u < 0 || w2 < 0 || u >= m.w || w2 >= m.h) continue;
        const int j = w2 * m.w + u;
        const uint8_t f = detail::fc_at(m, u, w2) ? 1 : 0;
        if (f != C.fc[j]) { C.n_front += (long)f - (long)C.fc[j]; C.fc[j] = f; }
      }
    }
  }
  C.bx0 = kx0; C.by0 = ky0; C.bx1 = kx1; C.by1 = ky1;   // 다음번 견줄 상자 (없으면 kx1 < 0)
  if (C.check_every > 0 && C.calls % C.check_every == 0) {
    ++C.n_check;
    if (!detail::cache_ok(C, m, cam)) {
      ++C.n_bad;
      detail::cache_full(C, m, cam, rc, kx0, ky0, kx1, ky1);
    }
  }
}

}  // namespace mdp
