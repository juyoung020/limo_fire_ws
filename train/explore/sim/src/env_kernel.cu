// 환경 한 걸음 GPU: 단계 순서·값은 src/env_ref.cpp(CPU 참조판)와 같다 (비트 동일 — tools/env_verify). 일을 성격대로 커널로 나눈다:
//   1. 물리 (환경 하나 = 스레드 하나): 행동 → 물리 걸음을 다음 라이다 장까지 → 장 준비 (scan_begin)
//   2. 빔 (빔 하나 = 스레드 하나): 재고 지도 표시·각도 칸 (scan_beam — 장치 메모리 원자 연산, 정수 min/or 라 순서 무관)
//   3. 지도 갱신 (환경 하나 = 블록 하나): scan_apply + 정수 합 → 1 로 돌아가 다음 장 (목표점 행동은 한 행동에 장 여럿)
//   4. 마무리 (환경 하나 = 블록 하나): 남은 물리 → 보상·끝 → 라이다 링 → 끝난 판 리셋 → 관측 (깊이 빼고)
//   5. 깊이 (픽셀 하나 = 스레드 하나)
// 2·5 의 광선은 RT 코어(OptiX — src/optix_rays.cu 의 raygen 이 광선을 만들고 결과 뒤 단계까지)나 소프트웨어 순회(아래 광선 묶음 커널)로
// 커널 사이 상태는 b.st(EnvState)와 b.scratch(Shared, 환경마다)에 둔다.
#include "env_api.h"
#include "env_rays.h"
#ifdef SIM_OPTIX
#include "optix_rt.h"
#endif

namespace sim {
namespace {

constexpr int kBeamThreads = 64;
constexpr int kPhysThreads = 64;
constexpr int kDepthThreads = 128;

__device__ void reset_env(const EnvBatch& b, EnvState& e, int env, int tid) {
  reset_map(b, env, tid);
  if (tid == 0) reset_state(b, e);
  __syncthreads();
}

// 먼 지도: 다시 그릴 칸을 모아(공유 메모리 목록) 칸마다 8 스레드가 점 줄 하나씩 — 최댓값·OR 을 8 스레드끼리 합친다.
// 칸마다 스레드 하나가 64 점을 다 돌면 다시 그릴 칸이 흩어진 워프가 오래 놀아서 (CPU 참조판은 obs_cells 안에서 칸마다 차례로, 같은 함수)
__device__ void obs_far_block(const EnvBatch& b, const EnvState& e, int env, int tid, const mdp::obsgen::FarDirty& fd, bool statics) {
  using namespace mdp::obs;
  constexpr int kRowThreads = mdp::obsgen::kFarSub;   // 칸 하나 = 8 스레드
  constexpr int kPerRound = kThreads / kRowThreads;
  static_assert(32 % kRowThreads == 0 && kThreads % 32 == 0, "점 줄 스레드 묶음이 워프 안에");
  __shared__ int list[FAR_H * FAR_W];
  __shared__ int n_list;
  const EnvCfg& c = b.cfg;
  const MapDev& m = b.maps[e.map];
  const int16_t* L = b.lmap + (size_t)env * c.size * c.size;
  const uint32_t* M = b.marks + (size_t)env * (c.size * c.size / 4);
  uint8_t* F = b.obs_img + (size_t)env * IMG_SIZE + NEAR_SIZE;
  mdp::obsgen::FarFrame ff;
  mdp::obsgen::far_frame(e, e.x + e.ex, e.y + e.ey, &ff);
  if (tid == 0) n_list = 0;
  __syncthreads();
  for (int i = tid; i < FAR_H * FAR_W; i += kThreads)
    if (mdp::obsgen::far_cell_head(e, fd, ff, F, i, statics)) list[atomicAdd(&n_list, 1)] = i;
  __syncthreads();
  const int n = n_list, kr = tid % kRowThreads;
  for (int base = 0; base < n; base += kPerRound) {   // 모든 스레드가 같은 횟수 (셔플이 워프 전체)
    const int t = base + tid / kRowThreads;
    int lmax = kUnknown, vis = 0, fr = 0;
    if (t < n) mdp::obsgen::far_cell_row(c, m, L, M, b.fmap + (size_t)env * c.size * fmap_row(c), e, ff, list[t], kr, &lmax, &vis, &fr);
    for (int o = 1; o < kRowThreads; o <<= 1) {
      const int l2 = __shfl_xor_sync(0xffffffffu, lmax, o);
      lmax = l2 > lmax ? l2 : lmax;
      vis |= __shfl_xor_sync(0xffffffffu, vis, o);
      fr |= __shfl_xor_sync(0xffffffffu, fr, o);
    }
    if (t < n && kr == 0) mdp::obsgen::far_cell_put(c, F, list[t], lmax, vis, fr);
  }
}

// ---------- 도달 가능한 프론티어 요약 (mdp/frontier.h reach_summary 와 같은 값, 블록 병렬 — 아래 reach_block·reach_fix) ----------
constexpr int kRT = kThreads;     // env_reach_kernel 블록 스레드 (환경 하나)
constexpr int kAccShared = 1792; // 번진 곳·설 수 있는 칸·BFS 후보 비트를 공유 메모리에 둘 범위 낱말 수 상한 (7 KB 씩 둘)
__device__ int uf_find(int* par, int v) {   // par[x] ≤ x 가 늘 성립 (작은 쪽으로만 잇는다) — 중간 칸을 할아버지로 당김
  volatile int* P = par;
  int cur = P[v];
  if (cur != v) {
    int prev = v, next;
    while (cur > (next = P[cur])) {
      P[prev] = next;
      prev = cur;
      cur = next;
    }
  }
  return cur;
}
__device__ void uf_union(int* par, int a, int b) {
  while (true) {
    a = uf_find(par, a);
    b = uf_find(par, b);
    if (a == b) return;
    if (a > b) { const int t = a; a = b; b = t; }
    if (atomicCAS(&par[b], b, a) == b) return;
  }
}

// 설 수 있는 칸 낱말 (y, wx): 앎 & ~막힘 & ~(막힘을 로봇 반경 원판으로 팽창) — 줄 dy 마다 막힘 낱말을 ±hw(dy) 칸 밀어 OR (mdp::standable 과 같은 원판)
__device__ uint32_t stand_word(const uint32_t* OCC, const uint32_t* KN, int ws, int gh, int y, int wx, mdp::Robot rb) {
  uint32_t dil = 0u;
  for (int dy = -rb.r; dy <= rb.r; ++dy) {
    const int yy = y + dy;
    if (yy < 0 || yy >= gh || dy * dy > rb.r2) continue;
    int hw = 0;
    while ((hw + 1) * (hw + 1) + dy * dy <= rb.r2) ++hw;
    const uint32_t* row = OCC + yy * ws;
    const uint32_t a = row[wx], l = wx > 0 ? row[wx - 1] : 0u, r = wx + 1 < ws ? row[wx + 1] : 0u;
    uint32_t d = a;
    for (int h = 1; h <= hw; ++h) d |= (a << h) | (l >> (32 - h)) | (a >> h) | (r << (32 - h));
    dil |= d;
  }
  const int wi = y * ws + wx;
  return KN[wi] & ~OCC[wi] & ~dil;
}

// 지도 갱신 뒤 (모든 스레드): 칸 종류가 바뀐 칸(fchg 낱말 4 칸, 넘치면 장 범위 전체)의 막힘·앎 비트를 고치고,
// 그 칸에서 로봇 반경 안 칸의 설 수 있음 비트를 다시 — 다른 칸은 막힘·앎이 반경 안에서 그대로라 같은 값 (리셋이 0 으로 비움 = 모두 모름)
__device__ void reach_fix(const EnvBatch& b, const EnvState& e, int env, const Shared& sh, int tid, int n_chg) {
  __shared__ int s_nd;
  const EnvCfg& c = b.cfg;
  const MapDev& m = b.maps[e.map];
  const int16_t* L = b.lmap + (size_t)env * c.size * c.size;
  const int ws = fmap_row(c), nf = c.size * ws;
  uint32_t* RB = b.rbits + (size_t)env * kReachBits * nf;
  uint32_t* OCC = RB + kRbOcc * nf;
  uint32_t* KN = RB + kRbKnown * nf;
  uint32_t* QF = RB + kRbQueued * nf;   // 다시 셀 낱말 표시 (reach_block BFS 와 같은 칸 — 둘 다 쓰고 0 으로 되돌림)
  uint32_t* ST = RB + kRbStand * nf;
  uint32_t* STC = RB + kRbStandC * nf;
  const mdp::Robot rbc = mdp::robot_cells(c.close_radius, c.res);   // 닫힘·후보 번지기 반경 (= 계층형 계획기 반경)
  int32_t* QC = b.rlist + (size_t)env * mdp::kReachCap * kReachLists + 6 * mdp::kReachCap;
  const mdp::Robot rb = mdp::obsgen::reach_robot(c);
  const auto fix = [&](int x, int y) {
    if (x < 0 || y < 0 || x >= m.gw || y >= m.gh) return;
    const uint8_t k = mdp::cell_class(L[y * c.size + x]);
    const uint32_t bit = 1u << (x & 31);
    uint32_t* o = OCC + y * ws + (x >> 5);
    uint32_t* n = KN + y * ws + (x >> 5);
    if (k == mdp::kOcc) { if (!(*o & bit)) atomicOr(o, bit); } else if (*o & bit) atomicAnd(o, ~bit);
    if (k != mdp::kUnk) { if (!(*n & bit)) atomicOr(n, bit); } else if (*n & bit) atomicAnd(n, ~bit);
  };
  const bool listed = n_chg <= kClassLog;
  const int32_t* list = b.fchg + (size_t)env * kClassLog;
  if (listed) {
    for (int k = tid; k < 4 * n_chg; k += kThreads) {
      const int ci = list[k / 4];
      fix(ci % c.size + (k & 3), ci / c.size);
    }
  } else if (sh.bx1 >= 0) {
    const int w = sh.bx1 - sh.bx0 + 1, h = sh.by1 - sh.by0 + 1;
    for (int k = tid; k < w * h; k += kThreads) fix(sh.bx0 + k % w, sh.by0 + k / w);
  }
  const int RR = rb.r > rbc.r ? rb.r : rbc.r;   // 다시 셀 범위 = 두 반경 중 큰 것
  if (tid == 0) s_nd = 0;
  __syncthreads();   // 막힘·앎 다 고침
  if (listed) {
    // 바뀐 낱말(4 칸 x0..x0+3, 줄 y)에서 반경 r 안: 줄 y−r..y+r, 낱말 (x0−r)/32..(x0+3+r)/32 — 표시해 한 번씩만
    const int span = 2 * RR + 1;
    for (int k = tid; k < n_chg * span * 2; k += kThreads) {
      const int ci = list[k / (span * 2)], q = k % (span * 2);
      const int x0 = ci % c.size, y = ci / c.size + q / 2 - RR;
      const int xa = max(0, x0 - RR), xb = min(m.gw - 1, x0 + 3 + RR);
      const int wx = (q & 1) ? (xb >> 5) : (xa >> 5);
      if (y < 0 || y >= m.gh || ((q & 1) && (xb >> 5) == (xa >> 5))) continue;
      const int wi = y * ws + wx;
      if (QF[wi] == 0u && atomicExch(&QF[wi], 1u) == 0u) QC[atomicAdd(&s_nd, 1)] = wi;
    }
    __syncthreads();
    const int nd = s_nd;
    for (int i = tid; i < nd; i += kThreads) {
      const int wi = QC[i];
      ST[wi] = stand_word(OCC, KN, ws, m.gh, wi / ws, wi % ws, rb);
      STC[wi] = stand_word(OCC, KN, ws, m.gh, wi / ws, wi % ws, rbc);
      QF[wi] = 0u;
    }
  } else if (sh.bx1 >= 0) {   // 넘침: 장 범위 ± r 전체
    const int y0 = max(0, sh.by0 - RR), y1 = min(m.gh - 1, sh.by1 + RR);
    const int w0 = max(0, sh.bx0 - RR) >> 5, w1 = min(m.gw - 1, sh.bx1 + RR) >> 5, nw = w1 - w0 + 1;
    for (int k = tid; k < nw * (y1 - y0 + 1); k += kThreads) {
      const int y = y0 + k / nw, wx = w0 + k % nw;
      ST[y * ws + wx] = stand_word(OCC, KN, ws, m.gh, y, wx, rb);
      STC[y * ws + wx] = stand_word(OCC, KN, ws, m.gh, y, wx, rbc);
    }
  }
}

// 시작 칸 (스레드 하나) = frontier_search nearestCell 과 같은 칸: 로봇 칸 (rx, ry) 이 설 수 있으면 그 칸, 아니면 FIFO 차례 4 이웃 BFS 로
// 처음 만나는 설 수 있는 칸. 층 d 차례 cur → 층 d+1 차례 nxt: 칸마다 nb4 차례로 이웃 중 맨해튼 d+1 이고 아직 nxt 에 없는 칸을 붙인다
// (FIFO 와 같은 차례). 물결 목록 QA·QB 를 두 층 버퍼로. 범위(건드린 칸 범위 낱말) 밖은 설 수 없음. 설 수 있는 칸이 하나는 있어야 부른다
// (없으면 맵 전체를 돈다 — 부르는 쪽이 확인). reach_block 과 계층형 계획(env_plan_kernel)이 같이 쓴다
__device__ int stand_start(const MapDev& m, int rx, int ry, const uint32_t* ST, int ws, int by0, int by1, int bw0, int bw1, int32_t* QA,
                           int32_t* QB) {
  const auto st_bit = [&](int x, int y) -> bool {
    const int wx = x >> 5;
    if (y < by0 || y > by1 || wx < bw0 || wx > bw1) return false;
    return (ST[y * ws + wx] >> (x & 31)) & 1u;
  };
  if (st_bit(rx, ry)) return ry * m.gw + rx;
  int* cur = QA;
  int* nxt = QB;
  int nc = 1, d = 0;
  cur[0] = ry * m.gw + rx;
  while (nc > 0) {
    for (int i = 0; i < nc; ++i) {
      const int x = cur[i] % m.gw, y = cur[i] / m.gw;
      if (st_bit(x, y)) return cur[i];
    }
    int nn = 0;
    for (int i = 0; i < nc; ++i) {
      const int x = cur[i] % m.gw, y = cur[i] / m.gw;
      for (int k = 0; k < 4; ++k) {
        int dx, dy;
        mdp::nb4(k, &dx, &dy);
        const int u = x + dx, v = y + dy;
        if (u < 0 || v < 0 || u >= m.gw || v >= m.gh) continue;
        if (abs(u - rx) + abs(v - ry) != d + 1) continue;
        const int j = v * m.gw + u;
        bool seen = false;
        for (int q = nn - 1; q >= 0 && !seen; --q) seen = nxt[q] == j;
        if (!seen) nxt[nn++] = j;
      }
    }
    int* t = cur; cur = nxt; nxt = t;
    nc = nn;
    ++d;
  }
  return -1;
}

// 한 줄 낱말 안 이어 채우기: 씨앗 t(⊆ m)가 든 m 의 1 줄기를 위쪽 비트(칸 x 큰 쪽)로 끝까지 — 덧셈 자리올림이 1 줄기를 타고 오른다
__device__ __forceinline__ uint32_t run_fill_up(uint32_t t, uint32_t m) {
  return (((uint32_t)((unsigned long long)m + t) ^ m) | t) & m;
}

// env_reach_kernel (모든 스레드, 판이 끝났으면 리셋 뒤 지도): RS 를 채운다 — 끝에 __syncthreads.
// 값은 mdp::reach_summary(기준선 frontier_search)와 같고, 순서와 무관한 정수 연산으로만 낸다:
//   1. 프론티어 칸 목록 번호 = 건드린 범위 낱말을 줄 차례로 훑은 차례(낱말마다 앞합). 설 수 있는 칸 비트는 지도 갱신 때 reach_fix 가 고쳐 둠
//   2. 시작 칸 = frontier_search nearestCell (FIFO 차례 그대로, 스레드 0)
//   3. 번진 곳 R = 시작 칸의 4 이웃 연결 성분: 줄마다 낱말 덧셈 자리올림으로 가로 1 줄기를 한 번에, 낱말 열마다 세로로 위→아래·아래→위 쓸기를
//      바뀜이 없을 때까지 번갈아 (고정점 = 연결 성분, 걸음 차례와 무관). 층 BFS 와 달리 반복 수 = 길의 꺾임 수
//   4. 덩어리 = R 안 프론티어 칸의 8 이웃 연결 성분 (합집합 찾기 — 늘 작은 번호 쪽으로 atomicCAS, 뿌리 = 가장 작은 목록 번호, ECL-CC 꼴),
//      뿌리마다 칸 수, 문턱 넘는 덩어리 수
//   5. 거리 = 층 BFS(4 이웃, 설 수 있는 칸): 물결 낱말 목록만 돌고(넓이가 아니라 물결 길이에 비례), 문턱 넘는 덩어리를 min(3, 덩어리 수) 개
//      처음 맞힌 층을 마치면 멈춤 — 덩어리 열쇠 (거리 << 32 | 칸) 최솟값은 처음 맞힌 층에서 정해지고 안 맞힌 덩어리는 거리가 더 멀다
// 계층형(b.cand ≠ nullptr)이면 같은 BFS 에서 후보(mdp::cand_pick 과 같은 값 — 크기 문턱 hier::kCandMinCells, 열쇠 짧은 순 hier::K 개)와
// 지금 목표 칸(e.goal_cell)을 품은 덩어리를 CS 에 더 낸다. BFS 는 두 목록이 다 찰 때까지 — 기본 요약(가까운 3 개)은 그대로 (더 먼 층을 더 돌 뿐)
__device__ void reach_block(const EnvBatch& b, const EnvState& e, int env, mdp::ReachSummary& RS, mdp::CandSummary& CS, int tid) {
  constexpr int kHitCap = 64;
  __shared__ int s_tot, s_any, s_start, s_cells, s_found, s_hits, s_nhit, s_cntA[2], s_cntC[2];
  __shared__ int s_found_all, s_hits_all, s_groot, s_best_r, s_block;   // s_block: 닫힘을 막는 칸 수 (mdp::SearchResult::n_block)   // 계층형 후보: 문턱 kCandMinCells 덩어리 수·맞힌 수, 지금 목표 덩어리 뿌리
  __shared__ uint32_t s_a[kAccShared];   // 범위가 작으면: 채우기 동안 설 수 있는 칸, BFS 동안 후보 비트
  __shared__ uint32_t s_r[kAccShared];   // 범위가 작으면: 번진 곳 (RE 대신)
  __shared__ int s_scan[kRT];
  __shared__ int s_hit[kHitCap], s_hit2[kHitCap], s_nhit2;
  __shared__ unsigned long long s_best, s_key2;   // s_key2: 후보 v2 가 고른 덩어리의 열쇠
  __shared__ int s_best_n;
  const EnvCfg& c = b.cfg;
  const MapDev& m = b.maps[e.map];
  const bool hier = b.cand != nullptr;
  if (tid == 0) {
    mdp::reach_clear(&RS);
    if (hier) mdp::cand_clear(&CS);
  }
  if (e.kb_x1 < 0) { __syncthreads(); return; }   // 아는 칸 없음 (블록 전체 같은 값)
  const int ws = fmap_row(c), nf = c.size * ws;
  uint32_t* RB = b.rbits + (size_t)env * kReachBits * nf;
  const uint32_t* ST = RB + kRbStand * nf;
  const uint32_t* STC = RB + kRbStandC * nf;
  uint32_t* CAMW = RB + kRbCam * nf;    // 깊이 카메라 층 막힘 비트 (이번 관측, 범위 ± 반경)
  uint32_t* PLW = RB + kRbPlan * nf;    // 계획기가 설 수 있음 = STC & ~(CAMW 팽창) (이번 관측, 범위 안)
  const uint32_t* OCCB = RB + kRbOcc * nf;
  const uint32_t* VIS = b.trail + (size_t)env * 2 * nf;   // 닫힘 규칙의 지나간 칸 · 본 칸 (trail_mark)
  const uint32_t* LOOK = VIS + nf;
  const uint32_t* KNW = RB + kRbKnown * nf;
  // 본 프론티어 칸 (mdp::looked_cell 과 같은 값): 4 이웃 중 맵 안 모름 칸(앎 비트 0)이 모두 LOOK 또는 VIS(몸통이 지나간 칸)
  const auto looked = [&](int x, int y) -> bool {
    for (int k = 0; k < 4; ++k) {
      int dx, dy;
      mdp::nb4(k, &dx, &dy);
      const int u = x + dx, v = y + dy;
      if (u < 0 || v < 0 || u >= m.gw || v >= m.gh) continue;
      const int wi = v * ws + (u >> 5);
      const uint32_t bit = 1u << (u & 31);
      if (!(KNW[wi] & bit) && !((LOOK[wi] | VIS[wi]) & bit)) return false;
    }
    return true;
  };
  uint32_t* RE = RB + kRbReach * nf;
  uint32_t* ACC = RB + kRbAcc * nf;     // BFS 다음 층 후보 비트 — 쓰고 나면 0 으로 되돌려 늘 0 (리셋이 비움)
  const uint32_t* F = b.fmap + (size_t)env * nf;
  int32_t* PRE = b.rpre + (size_t)env * nf;
  int32_t* LST = b.rlist + (size_t)env * mdp::kReachCap * kReachLists;
  int32_t* PAR = LST;                                   // 합집합 찾기 부모 (목록 번호)
  int32_t* CNT = LST + mdp::kReachCap;                  // 뿌리마다 덩어리 칸 수
  int32_t* QA = LST + 2 * mdp::kReachCap;               // BFS 물결 낱말 (번호, 비트) 두 벌 + 후보 낱말 — 각 kReachCap(≥ 범위 낱말 수)
  int32_t* QB = LST + 4 * mdp::kReachCap;
  int32_t* QC = LST + 6 * mdp::kReachCap;
  unsigned long long* KEY = b.rkey + (size_t)env * mdp::kReachCap;
  unsigned long long* KEY2 = b.rkey2 ? b.rkey2 + (size_t)env * mdp::kReachCap : nullptr;   // 계층형 후보 열쇠
  const int by0 = e.kb_y0, by1 = e.kb_y1, bw0 = e.kb_x0 >> 5, bw1 = e.kb_x1 >> 5, nbw = bw1 - bw0 + 1, nbox = nbw * (by1 - by0 + 1);
  const int min_cells = mdp::obsgen::reach_min_cells(c);
  // 범위 낱말이 kAccShared 이하면 번진 곳·설 수 있는 칸·BFS 후보 비트를 공유 메모리에 (범위 안 번호 = (y − by0) · nbw + (wx − bw0)) — 같은 값
  const bool small = nbox <= kAccShared;
  const auto bidx = [&](int wi) { return (wi / ws - by0) * nbw + (wi % ws - bw0); };
  const auto REp = [&](int wi) -> uint32_t* { return small ? &s_r[bidx(wi)] : &RE[wi]; };
  // 0. 계획기가 설 수 있는 칸 (mdp::frontier_search plan_ok): 닫힘 반경 rbc 로 지도 막힘 팽창(STC, reach_fix) & 깊이 카메라 층 막힘을 같은 반경으로 팽창한 곳 밖
  const mdp::Robot rbc = mdp::robot_cells(c.close_radius, c.res);
  const bool cam = b.cvox != nullptr;
  if (cam) {   // 층 비트: 범위 ± rbc 줄 · ± 1 낱말 (팽창이 읽는 곳 — 맵 안만)
    const int y0 = max(0, by0 - rbc.r), y1 = min(m.gh - 1, by1 + rbc.r), w0 = max(0, bw0 - 1), w1 = min(ws - 1, bw1 + 1), nw = w1 - w0 + 1;
    const uint32_t* V = reinterpret_cast<const uint32_t*>(b.cvox + (size_t)env * c.size * c.size);
    for (int k = tid; k < nw * (y1 - y0 + 1); k += kRT) {
      const int y = y0 + k / nw, wx = w0 + k % nw;
      uint32_t bits = 0u;
      for (int q = 0; q < 8; ++q) {
        const int x = wx * 32 + q * 4;
        if (x >= c.size) break;
        const uint32_t v = V[((size_t)y * c.size + x) >> 2];
        for (int j = 0; j < 4; ++j) bits |= ((v >> (8 * j)) & 0xffu) && x + j < m.gw ? 1u << (q * 4 + j) : 0u;
      }
      CAMW[y * ws + wx] = bits;
    }
    __syncthreads();
  }
  for (int k = tid; k < nbox; k += kRT) {
    const int y = by0 + k / nbw, wx = bw0 + k % nbw, wi = y * ws + wx;
    uint32_t dil = 0u;
    if (cam)
      for (int dy = -rbc.r; dy <= rbc.r; ++dy) {
        const int yy = y + dy;
        if (yy < 0 || yy >= m.gh || dy * dy > rbc.r2) continue;
        int hw = 0;
        while ((hw + 1) * (hw + 1) + dy * dy <= rbc.r2) ++hw;
        const int i = yy * ws + wx;
        const uint32_t a = CAMW[i], l = wx > 0 ? CAMW[i - 1] : 0u, r = wx + 1 < ws ? CAMW[i + 1] : 0u;
        uint32_t d = a;
        for (int h = 1; h <= hw; ++h) d |= (a << h) | (l >> (32 - h)) | (a >> h) | (r << (32 - h));
        dil |= d;
      }
    PLW[wi] = STC[wi] & ~dil;
  }
  __syncthreads();
  // 번지기 칸 = 계획기가 설 수 있는 칸 ∪ 막힘(지도·카메라 층) 아닌 지나간 칸 ∪ 시작 칸 (mdp::frontier_search 의 pass — 닫힘 규칙).
  // 범위가 작으면 공유 메모리 사본 s_a (시작 칸 비트는 시작 칸을 찾은 뒤 더함)
  int swi = -1;
  uint32_t sbit = 0u;
  const auto CAMv = [&](int wi) -> uint32_t { return cam ? CAMW[wi] : 0u; };
  const auto STv = [&](int wi) -> uint32_t {
    return small ? s_a[bidx(wi)] : (PLW[wi] | (VIS[wi] & ~OCCB[wi] & ~CAMv(wi)) | (wi == swi ? sbit : 0u));
  };
  // 1. 번진 곳 비우기 (설 수 있는 칸 비트는 지도 갱신 때 reach_fix 가 바뀐 곳만 고쳐 둠)
  int any = 0;
  for (int k = tid; k < nbox; k += kRT) {
    const int wi = (by0 + k / nbw) * ws + bw0 + k % nbw;
    const uint32_t st = ST[wi];
    if (small) { s_r[k] = 0u; s_a[k] = PLW[wi] | (VIS[wi] & ~OCCB[wi] & ~CAMv(wi)); } else RE[wi] = 0u;
    any |= st != 0u;
  }
  // 프론티어 칸 앞합 (스레드마다 이어진 낱말 덩이)
  const int chunk = (nbox + kRT - 1) / kRT, k0 = tid * chunk, k1 = min(nbox, k0 + chunk);
  int part = 0;
  for (int k = k0; k < k1; ++k) part += __popc(F[(by0 + k / nbw) * ws + bw0 + k % nbw]);
  s_scan[tid] = part;
  any = __syncthreads_or(any);
  if (tid == 0) {
    int acc = 0;
    for (int i = 0; i < kRT; ++i) { const int v = s_scan[i]; s_scan[i] = acc; acc += v; }
    s_tot = acc;
    s_any = any;
    s_start = -1;
  }
  __syncthreads();
  {
    int acc = s_scan[tid];
    for (int k = k0; k < k1; ++k) {
      PRE[k] = acc;
      acc += __popc(F[(by0 + k / nbw) * ws + bw0 + k % nbw]);
    }
  }
  const int n_tot = s_tot;
  const bool over = n_tot > mdp::kReachCap;
  int rx, ry;
  cell_of(c, m, e.x + e.ex, e.y + e.ey, &rx, &ry);
  const bool in_map = rx >= 0 && ry >= 0 && rx < m.gw && ry < m.gh;
  const auto list_pos = [&](int x, int y) -> int {
    const int wx = x >> 5, bit = x & 31;
    return PRE[(y - by0) * nbw + (wx - bw0)] + __popc(F[y * ws + wx] & ((1u << bit) - 1u));
  };
  // 2. 시작 칸 (스레드 0, 물결 목록 QA·QB 를 두 층 버퍼로)
  __syncthreads();
  if (tid == 0 && in_map && s_any) s_start = stand_start(m, rx, ry, ST, ws, by0, by1, bw0, bw1, QA, QB);
  for (int i = tid; i < (over ? 0 : n_tot); i += kRT) {
    PAR[i] = i;
    CNT[i] = 0;
    KEY[i] = ~0ull;
    if (KEY2) KEY2[i] = ~0ull;
  }
  __syncthreads();
  const int start = s_start;
  const int sx = start % m.gw, sy = start / m.gw;
  if (start >= 0) { swi = sy * ws + (sx >> 5); sbit = 1u << (sx & 31); }
  if (tid == 0) {
    s_cells = 0;
    s_found = 0;
    s_hits = 0;
    s_nhit = 0;
    s_found_all = 0;
    s_nhit2 = 0;
    s_hits_all = 0;
    s_groot = -1;
    s_block = 0;
    if (start >= 0) {
      *REp(sy * ws + (sx >> 5)) = 1u << (sx & 31);
      if (small) s_a[bidx(sy * ws + (sx >> 5))] |= 1u << (sx & 31);   // 시작 칸은 번지기 칸 (계획 반경으로 설 수 없어도)
    }
  }
  __syncthreads();
  // 지나간 칸 씨앗 (mdp::frontier_search b.vis 와 같음): 시작 칸이 있을 때 지나간 칸 ∩ 설 수 있는 칸을 번진 곳에 더함 — 고정점은 씨앗들의 연결 성분 합
  if (start >= 0) {
    for (int k = tid; k < nbox; k += kRT) {
      const int wi = (by0 + k / nbw) * ws + bw0 + k % nbw;
      const uint32_t sd = VIS[wi] & STv(wi);
      if (sd) *REp(wi) |= sd;
    }
    __syncthreads();
  }
  // 3. 번진 곳 = 시작 칸의 4 이웃 연결 성분 (가로 줄 채우기 ↔ 세로 쓸기, 바뀜이 없을 때까지)
  if (start >= 0) {
    for (;;) {
      int chg = 0;   // 세로 쓸기가 바꾼 칸이 있나
      for (int y = by0 + tid; y <= by1; y += kRT) {   // 줄마다: 오른쪽으로 자리올림을 넘기며, 다음 왼쪽으로 (비트 뒤집어 같은 셈)
        uint32_t carry = 0u;
        for (int wx = bw0; wx <= bw1; ++wx) {
          uint32_t* rp = REp(y * ws + wx);
          const uint32_t mk = STv(y * ws + wx), old = *rp, t = (old | carry) & mk;
          const uint32_t f = t ? run_fill_up(t, mk) : 0u;
          carry = f >> 31;
          if (f != old) *rp = f;
        }
        carry = 0u;
        for (int wx = bw1; wx >= bw0; --wx) {
          uint32_t* rp = REp(y * ws + wx);
          const uint32_t mk = STv(y * ws + wx), old = *rp, t = (old | (carry << 31)) & mk;
          const uint32_t f = t ? __brev(run_fill_up(__brev(t), __brev(mk))) : 0u;
          carry = f & 1u;
          if (f != old) *rp = f;
        }
      }
      __syncthreads();
      {   // 낱말 열마다 (워프 하나가 한 열, 레인 = 32 줄 묶음): 세로 x_y = R_y | (S_y & x_{y−1}) 를 (g, p) 결합 워프 앞합으로 — 위→아래, 아래→위
        const int lane = tid & 31;
        for (int wx = bw0 + (tid >> 5); wx <= bw1; wx += kRT / 32) {
          for (int dir = 0; dir < 2; ++dir) {
            uint32_t carry = 0u;
            for (int base = 0; base <= by1 - by0; base += 32) {
              const int yo = base + lane, y = dir == 0 ? by0 + yo : by1 - yo;
              const bool ok = yo <= by1 - by0;
              const int wi = y * ws + wx;
              const uint32_t old = ok ? *REp(wi) : 0u, p0 = ok ? STv(wi) : 0u;
              // 묶음(32 줄 × 낱말) 안에서 세로(레인 차례 앞합 ↑·↓)와 가로(낱말 안 채우기)를 안 바뀔 때까지 — 묶음 안 꺾임을 이번에 다 풀어
              // 바깥 반복 수를 줄임. 들어오는 칸은 위 묶음에서 넘어온 carry 뿐이고, 더하는 칸은 모두 설 수 있는 칸으로 이어진 칸이라 고정점은 같다
              uint32_t g = lane == 0 ? (old | (carry & p0)) : old;
              for (;;) {
                uint32_t gu = g, pu = p0;
                for (int o = 1; o < 32; o <<= 1) {   // 레인 오름 차례
                  const uint32_t gp = __shfl_up_sync(0xffffffffu, gu, o), pp = __shfl_up_sync(0xffffffffu, pu, o);
                  if (lane >= o) { gu |= pu & gp; pu &= pp; }
                }
                uint32_t gd = gu, pd = p0;
                for (int o = 1; o < 32; o <<= 1) {   // 레인 내림 차례
                  const uint32_t gp = __shfl_down_sync(0xffffffffu, gd, o), pp = __shfl_down_sync(0xffffffffu, pd, o);
                  if (lane + o < 32) { gd |= pd & gp; pd &= pp; }
                }
                const uint32_t h = gd ? (run_fill_up(gd, p0) | __brev(run_fill_up(__brev(gd), __brev(p0)))) : 0u;
                const bool same = __all_sync(0xffffffffu, h == g);
                g = h;
                if (same) break;
              }
              if (ok && g != old) { *REp(wi) = g; chg = 1; }
              carry = __shfl_sync(0xffffffffu, g, 31);
            }
          }
        }
      }
      if (!__syncthreads_or(chg)) break;   // 세로가 안 바꿨으면 끝: 가로 채우기 뒤라 줄마다 가로로 닫혀 있고 세로도 닫힘 = 고정점
    }
  }
  // 4. 덩어리 = 번진 프론티어 칸(F & RE)의 8 이웃 연결 성분. 칸마다가 아니라 낱말 안 가로 줄기(이어진 1 비트)마다:
  //    (가) 줄기 칸의 부모 = 줄기 첫 칸 (목록 번호가 이어짐 — 작은 번호 쪽, par[x] ≤ x 그대로)
  //    (나) 줄기 끝이 낱말 끝(비트 31)이고 다음 낱말 비트 0 도 번진 프론티어면 잇기, 아래 줄 [첫 칸 − 1, 끝 칸 + 1] 안 줄기마다 첫 칸과 잇기
  //         (위 줄과의 연결은 위 줄 줄기가 맡음 — 8 이웃을 한 번씩)
  //    (다) 뿌리마다 칸 수 += 줄기 길이 — 칸 수가 문턱을 막 넘은 줄기에서 덩어리 수 하나
  //    칸 하나씩 잇던 것과 같은 성분·같은 뿌리(성분의 가장 작은 목록 번호)·같은 칸 수
  const auto for_runs = [&](uint32_t fr, auto&& f) {   // 낱말 안 줄기 (첫 비트, 길이)
    while (fr) {
      const int lo = __ffs(fr) - 1;
      const uint32_t t = fr >> lo;
      const int len = ~t ? __ffs(~t) - 1 : 32 - lo;
      f(lo, len);
      fr &= len == 32 ? 0u : ~(((1u << len) - 1u) << lo);
    }
  };
  int cells = 0;
  for (int k = tid; k < nbox; k += kRT) {
    const int y = by0 + k / nbw, wx = bw0 + k % nbw, wi = y * ws + wx;
    const uint32_t fr = F[wi] & *REp(wi);
    cells += __popc(fr);
    if (over || !fr) continue;
    for_runs(fr, [&](int lo, int len) {
      const int rep = list_pos(wx * 32 + lo, y);
      for (int q = 1; q < len; ++q) PAR[rep + q] = rep;
    });
  }
  atomicAdd(&s_cells, cells);
  __syncthreads();
  if (!over)
    for (int k = tid; k < nbox; k += kRT) {
      const int y = by0 + k / nbw, wx = bw0 + k % nbw, wi = y * ws + wx;
      const uint32_t fr = F[wi] & *REp(wi);
      if (!fr) continue;
      for_runs(fr, [&](int lo, int len) {
        const int a = wx * 32 + lo, bx = a + len - 1, rep = list_pos(a, y);
        if (lo + len == 32 && wx < bw1 && ((F[wi + 1] & *REp(wi + 1)) & 1u)) uf_union(PAR, rep, list_pos(bx + 1, y));
        if (y + 1 > by1) return;
        const int xl = max(a - 1, 0), xh = min(bx + 1, m.gw - 1);
        uint32_t carry = 0u;
        for (int ww = max(xl >> 5, bw0); ww <= min(xh >> 5, bw1); ++ww) {
          const int b0 = max(xl - ww * 32, 0), b1 = min(xh - ww * 32, 31);
          const uint32_t rng = (b1 == 31 ? ~0u : ((1u << (b1 + 1)) - 1u)) & ~((1u << b0) - 1u);
          const int wj = (y + 1) * ws + ww;
          const uint32_t mm = F[wj] & *REp(wj) & rng;
          for (uint32_t st = mm & ~((mm << 1) | carry); st; st &= st - 1u) uf_union(PAR, rep, list_pos(ww * 32 + __ffs(st) - 1, y + 1));
          carry = mm >> 31;
        }
      });
    }
  __syncthreads();
  // 뿌리마다 칸 수 (줄기 길이씩), 문턱을 막 넘은 순간 덩어리 수 하나
  int found = 0, found_all = 0;
  constexpr int kc = mdp::hier::kCandMinCells;
  if (!over)
    for (int k = tid; k < nbox; k += kRT) {
      const int y = by0 + k / nbw, wx = bw0 + k % nbw, wi = y * ws + wx;
      const uint32_t fr = F[wi] & *REp(wi);
      if (!fr) continue;
      for_runs(fr, [&](int lo, int len) {
        const int r = uf_find(PAR, list_pos(wx * 32 + lo, y));
        const int old = atomicAdd(&CNT[r], len);
        found += (old < min_cells && old + len >= min_cells) ? 1 : 0;
        found_all += (old < kc && old + len >= kc) ? 1 : 0;
      });
    }
  atomicAdd(&s_found, found);
  if (hier) atomicAdd(&s_found_all, found_all);
  __syncthreads();
  // 닫힘을 막는 칸 (mdp::frontier_search n_block 과 같은 값): 칸 수 ≥ close_min 덩어리의 칸 중 안 본(looked 아님) 칸
  {
    int blk = 0;
    uint32_t* GF = b.gfront ? b.gfront + (size_t)env * nf : nullptr;   // 그래프 모드: 막는 칸 비트 판 (범위 낱말마다 한 스레드가 통째로 씀)
    const bool live = !over && start >= 0;
    if (live || GF)
      for (int k = tid; k < nbox; k += kRT) {
        const int y = by0 + k / nbw, wx = bw0 + k % nbw, wi = y * ws + wx;
        const uint32_t fr = live ? (F[wi] & *REp(wi)) : 0u;
        uint32_t gw = 0u;
        if (fr)
          for_runs(fr, [&](int lo, int len) {
            if (CNT[uf_find(PAR, list_pos(wx * 32 + lo, y))] < c.close_min) return;
            const uint32_t run = (len == 32 ? ~0u : ((1u << len) - 1u)) << lo;
            for (uint32_t q = run; q; q &= q - 1u) {
              const int x = wx * 32 + __ffs(q) - 1;
              if (!looked(x, y)) { ++blk; gw |= 1u << (x & 31); }
            }
          });
        if (GF) GF[wi] = gw;
      }
    if (blk) atomicAdd(&s_block, blk);
  }
  const int need = over ? 0 : min(mdp::kReachTop, s_found);
  // 후보 v2 는 모든 덩어리의 열쇠가 있어야 함(합치기·큰 순) → 일찍 멈추지 않고 끝까지 번짐
  const bool v2 = hier && c.cand_v2 != 0;
  const int need_all = (over || !hier) ? 0 : (v2 ? (s_found_all > 0 ? 0x7fffffff : 0) : min(mdp::hier::K, s_found_all));
  // 계층형: 지금 목표 칸이 번진 프론티어 칸이면 그 덩어리 뿌리 (CPU 참조: frontier_search 의 flag bit1 · label)
  if (hier && tid == 0 && !over && e.goal_cell >= 0) {
    const int gx = e.goal_cell % m.gw, gy = e.goal_cell / m.gw, gwx = gx >> 5;
    if (gy >= by0 && gy <= by1 && gwx >= bw0 && gwx <= bw1) {
      const int wi = gy * ws + gwx;
      if (((F[wi] & *REp(wi)) >> (gx & 31)) & 1u) s_groot = uf_find(PAR, list_pos(gx, gy));
    }
  }
  // 맞힌 칸 (거리 dd): 문턱 넘는 덩어리면 열쇠를 줄이고, 처음 맞힌 덩어리면 셈
  // 계층형 후보는 따로 열쇠(KEY2)·목록: 크기 문턱 kc, 거리 hier::kCandMinDist 칸 이상인 칸만 (mdp::cand_pick 과 같은 규칙)
  const auto hit = [&](int x, int y, int dd) {
    const int r = uf_find(PAR, list_pos(x, y));
    const bool big = CNT[r] >= min_cells;
    // 후보 = 닫힘을 막을 수 있는 칸만 (mdp::cand_keys 와 같은 규칙): 덩어리 ≥ close_min, 안 봄
    const bool cand = hier && CNT[r] >= kc && dd >= mdp::hier::kCandMinDist && CNT[r] >= c.close_min && !looked(x, y);
    if (!big && !cand) return;
    const unsigned long long key = ((unsigned long long)(uint32_t)dd << 32) | (uint32_t)(y * m.gw + x);
    if (big && atomicMin(&KEY[r], key) == ~0ull) {
      atomicAdd(&s_hits, 1);
      const int q = atomicAdd(&s_nhit, 1);
      if (q < kHitCap) s_hit[q] = r;
    }
    if (cand && atomicMin(&KEY2[r], key) == ~0ull) {
      atomicAdd(&s_hits_all, 1);
      const int q = atomicAdd(&s_nhit2, 1);
      if (q < kHitCap) s_hit2[q] = r;
    }
  };
  // 5. 거리: 물결 낱말 목록 층 BFS (같은 층 안 차례는 결과에 안 남음 — 집합·최솟값만). 아직 안 간 칸 = RE 에 남은 칸
  //    (BFS 가 닿는 곳 = 번진 곳이라 RE 에서 지워 가며 씀 — RE 는 위에서 다 썼고 다음 관측이 다시 셈).
  //    후보 비트는 범위 낱말이 kAccShared 이하면 공유 메모리(범위 안 번호), 넘으면 장치 메모리 ACC — 같은 값. 층마다 __syncthreads 둘
  //    (셈 칸을 층 홀짝으로 두 벌 — 다음 층 칸은 지난 층이 다 읽은 뒤 스레드 0 이 비움)
  if (need > 0 || need_all > 0) {
    const auto acc_ref = [&](int wi) -> uint32_t* { return small ? &s_a[bidx(wi)] : &ACC[wi]; };
    if (small) {   // 채우기에 쓴 설 수 있는 칸 사본을 비워 후보 비트로
      for (int k = tid; k < nbox; k += kRT) s_a[k] = 0u;
      __syncthreads();
    }
    if (tid == 0) {
      const int wi = sy * ws + (sx >> 5);
      const uint32_t bit = 1u << (sx & 31);
      *REp(wi) &= ~bit;
      QA[0] = wi;
      QA[mdp::kReachCap] = (int32_t)bit;
      s_cntA[0] = 1;
      s_cntC[0] = 0;
      if (F[wi] & bit) hit(sx, sy, 0);
    }
    __syncthreads();
    int32_t* cur = QA;
    int32_t* nxt = QB;
    for (int d = 0;; ++d) {
      const int na = s_cntA[d & 1];
      if (na == 0 || (s_hits >= need && s_hits_all >= need_all)) break;   // 블록 전체 같은 값 (지난 층 끝 __syncthreads 뒤)
      if (tid == 0) { s_cntA[(d + 1) & 1] = 0; s_cntC[(d + 1) & 1] = 0; }
      // 밀기: 물결 낱말의 4 이웃 넓힘을 후보 비트에 OR — 이번 층에 처음 비트를 받은 낱말(전 값 0)만 후보 목록에
      int* nc_ = &s_cntC[d & 1];
      const auto push = [&](int wi, uint32_t g) {
        if (g && atomicOr(acc_ref(wi), g) == 0u) QC[atomicAdd(nc_, 1)] = wi;
      };
      for (int i = tid; i < na; i += kRT) {
        const int wi = cur[i], y = wi / ws, wx = wi % ws;
        const uint32_t a = (uint32_t)cur[mdp::kReachCap + i];
        push(wi, (a << 1) | (a >> 1));
        if (wx > bw0) push(wi - 1, a << 31);
        if (wx < bw1) push(wi + 1, a >> 31);
        if (y > by0) push(wi - ws, a);
        if (y < by1) push(wi + ws, a);
      }
      __syncthreads();
      const int nc = *nc_;
      for (int i = tid; i < nc; i += kRT) {
        const int wi = QC[i];
        uint32_t* ap = acc_ref(wi);
        uint32_t* rp = REp(wi);
        const uint32_t n = *ap & *rp & PLW[wi];   // 거리는 계획기가 설 수 있는 칸으로만 (지나간 칸으로만 이어진 칸은 거리 없음 — 후보 아님)
        *ap = 0u;
        if (!n) continue;
        *rp &= ~n;
        const int q = atomicAdd(&s_cntA[(d + 1) & 1], 1);
        nxt[q] = wi;
        nxt[mdp::kReachCap + q] = (int32_t)n;
        const int y = wi / ws, wx = wi % ws;
        for (uint32_t f = n & F[wi]; f; f &= f - 1u) hit(wx * 32 + __ffs(f) - 1, y, d + 1);
      }
      int32_t* t = cur; cur = nxt; nxt = t;
      __syncthreads();
    }
  }
  if (tid == 0) {
    mdp::ReachSummary& R = RS;
    R.start_free = start >= 0 ? 1 : 0;
    R.n_cells = s_cells;
    R.closed = (start >= 0 && !over && s_block == 0) ? 1 : 0;   // mdp::reach_pick 과 같음 (넘침이면 안 닫힘)
    R.overflow = over ? 1 : 0;
    R.n_found = over ? 0 : s_found;
    if (hier) CS.n = 0;
  }
  // 맞힌 덩어리에서 열쇠 작은 kcount 개 (칸 수 ≥ minc) → put(차례, 뿌리, 열쇠, 칸 수). 맞힌 목록이 넘쳤으면(드묾) 뿌리 전체를 kcount 번 훑음
  // (맞힌 뿌리만 열쇠가 ~0 아님). 안 맞힌 덩어리는 거리가 더 멂. put 은 스레드 0 만 부른다
  const auto pick = [&](const unsigned long long* KEY, const int* s_hit, int s_nhit, int kcount, int minc, auto&& put) {
    if (s_nhit <= kHitCap) {
      if (tid == 0) {
        unsigned long long last = 0ull;
        for (int t = 0; t < kcount; ++t) {
          unsigned long long best = ~0ull;
          int br = -1;
          for (int q = 0; q < s_nhit; ++q) {
            if (CNT[s_hit[q]] < minc) continue;
            const unsigned long long k = KEY[s_hit[q]];
            if ((t == 0 || k > last) && k < best) { best = k; br = s_hit[q]; }
          }
          if (br < 0) break;
          put(t, br, best, CNT[br]);
          last = best;
        }
      }
      return;
    }
    unsigned long long last = 0ull;
    for (int t = 0; t < kcount; ++t) {
      if (tid == 0) { s_best = ~0ull; s_best_n = 0; s_best_r = -1; }
      __syncthreads();
      unsigned long long mine = ~0ull;
      for (int i = tid; i < n_tot; i += kRT)
        if (KEY[i] != ~0ull && CNT[i] >= minc && (t == 0 || KEY[i] > last) && KEY[i] < mine) mine = KEY[i];
      if (mine != ~0ull) atomicMin(&s_best, mine);
      __syncthreads();
      const unsigned long long best = s_best;
      if (best == ~0ull) break;   // 블록 전체 같은 값
      for (int i = tid; i < n_tot; i += kRT)
        if (KEY[i] == best) { s_best_n = CNT[i]; s_best_r = i; }
      __syncthreads();
      if (tid == 0) put(t, s_best_r, best, s_best_n);
      last = best;
    }
  };
  pick(KEY, s_hit, s_nhit, mdp::kReachTop, min_cells, [&](int t, int, unsigned long long key, int n) {
    RS.top[t].n = n;
    RS.top[t].dist = (int32_t)(key >> 32);
    RS.top[t].cell = (int32_t)(key & 0xffffffffull);
  });
  if (v2) {   // mdp::cand_pick2 와 같은 값: 자리마다 뿌리 전체를 블록 병렬로 세 번 훑음 (차례 값 최소 → 그 뿌리 → 가까운 뿌리 합치고 지움)
    const int Rm = mdp::cand2_merge_cells(c.res);
    for (int t = 0; t < mdp::hier::K; ++t) {
      const bool far = t >= mdp::hier::K - mdp::hier::kCand2Far;
      __syncthreads();
      if (tid == 0) { s_best = ~0ull; s_best_n = 0; s_best_r = 0; s_key2 = ~0ull; }
      __syncthreads();
      unsigned long long mine = ~0ull;
      for (int i = tid; i < n_tot; i += kRT)
        if (KEY2[i] != ~0ull) {
          const unsigned long long o = mdp::cand2_order(KEY2[i], CNT[i], far);
          if (o < mine) mine = o;
        }
      if (mine != ~0ull) atomicMin(&s_best, mine);
      __syncthreads();
      const unsigned long long best = s_best;
      if (best == ~0ull) break;   // 블록 전체 같은 값
      for (int i = tid; i < n_tot; i += kRT)
        if (KEY2[i] != ~0ull && mdp::cand2_order(KEY2[i], CNT[i], far) == best) s_key2 = KEY2[i];   // 차례 값에 칸 번호가 들어 있어 하나뿐
      __syncthreads();
      const unsigned long long key = s_key2;
      int add = 0, cur = 0;
      for (int i = tid; i < n_tot; i += kRT) {
        if (KEY2[i] == ~0ull || !mdp::cand2_near(KEY2[i], key, m.gw, Rm)) continue;
        add += CNT[i];
        cur |= i == s_groot ? 1 : 0;
        KEY2[i] = ~0ull;
      }
      if (add) atomicAdd(&s_best_n, add);
      if (cur) atomicOr(&s_best_r, 1);
      __syncthreads();
      if (tid == 0) {
        CS.c[t].n = s_best_n;
        CS.c[t].dist = (int32_t)(key >> 32);
        CS.c[t].cell = (int32_t)(key & 0xffffffffull);
        CS.c[t].cur = s_best_r;
        CS.c[t].util = 0;
        CS.n = t + 1;
      }
    }
  } else if (hier) {
    __syncthreads();   // 위 훑기의 공유 칸(s_best …)을 다시 쓰기 전
    pick(KEY2, s_hit2, s_nhit2, mdp::hier::K, kc, [&](int t, int r, unsigned long long key, int n) {
      CS.c[t].n = n;
      CS.c[t].dist = (int32_t)(key >> 32);
      CS.c[t].cell = (int32_t)(key & 0xffffffffull);
      CS.c[t].cur = r == s_groot ? 1 : 0;
      CS.c[t].util = 0;
      CS.n = t + 1;
    });
  }
  __syncthreads();
}

// 관측 마무리 뒤 (env_finish_kernel·env_reset_kernel 다음): 환경마다 블록 하나가 요약을 내고 그 칸만 관측에 (observations.h obs_reach).
// 관측 커널과 따로 둔 까닭: 이 단계의 레지스터·공유 메모리가 관측 커널에 붙으면 그 커널 전체의 동시 블록 수가 줄어서
// gate (계층형 묶음): 0 모든 환경 · 1 판이 끝난(쉬는) 환경은 건너뜀 (묶음 가운데 걸음) · 2 끝난 환경만 (묶음 끝 리셋 뒤)
__global__ void __launch_bounds__(kRT, 6) env_reach_kernel(EnvBatch b, int gate, int sub) {
  __shared__ mdp::ReachSummary rs;
  __shared__ mdp::CandSummary cs;
  __shared__ int s_util[mdp::hier::K];
  const int env = blockIdx.x, tid = threadIdx.x;
  if (gate == 1 && b.done[env] != 0) return;   // 블록 전체 같은 값
  if (gate == 1 && gdec_idle(b, env, sub)) return;   // 그래프 모드: 결정이 앞 걸음에서 끝남 (끝난 걸음의 관측까지는 돈다)
  if (gate == 2 && b.done[env] == 0) return;
  const EnvState& e = b.st[env];
  trail_mark(b, e, env, tid, kRT);   // 닫힘 규칙의 지나간 길에 지금 자세 (CPU write_obs 와 같은 때)
  // 그래프 모드(전작 방식 판 끝) 결정 가운데 걸음: 요약을 읽는 곳이 없다 — 닫힘 값은 finish_step 이 안 보고(gterm), 보상(shape_obs)은 계층형에서 0,
  // 후보·막는 칸 판(gfront)·닫힘 값은 결정 끝 걸음의 것만 쓰인다. 덩어리 칸 관측은 지연 없는 환경이면 마지막 걸음 값이 그대로 덮어쓴다(O = P = 지금).
  // 지연 환경(e.delay > 0)은 O 가 한 걸음 전 값이라 걸음마다 내야 한다. → 지연 없는 환경은 결정이 끝난 걸음에만 요약 (값은 그대로)
  if (gate == 1 && b.gdec && b.cfg.graph_on && b.cfg.graph_term && b.gdec[env].reason == kGdRun && e.delay <= 0) return;   // 블록 전체 같은 값
  __syncthreads();
  reach_block(b, e, env, rs, cs, tid);
  if (tid == 0) {
    mdp::obsgen::obs_reach(b, e, env, rs);
    b.st[env].map_closed = rs.closed;   // 다음 걸음 finish_step 성공 판정 (CPU env_ref.cpp write_obs 와 같음)
    shape_obs(b, b.st[env], env, rs);     // approach·front_cells 보상 (env.h — CPU env_ref.cpp 도 write_obs 바로 뒤)
  }
  if (!b.cand) return;
  // 계층형 후보: 주변 프론티어 칸 수 = 프론티어 칸 비트 지도(fmap)의 ±R 네모 안 비트 수 (mdp::cand_util 과 같은 값) — (후보, 줄)마다 스레드
  const EnvCfg& c = b.cfg;
  const MapDev& m = b.maps[e.map];
  const int ws = fmap_row(c), R = mdp::cand_util_radius(c.res), rows = 2 * R + 1;
  const uint32_t* FM = b.fmap + (size_t)env * c.size * ws;
  if (tid < mdp::hier::K) s_util[tid] = 0;
  __syncthreads();
  if (c.cand_v2) {   // 후보 v2: 예상 관측량 (mdp::cand_gain_ray 의 합) — (후보, 광선)마다 스레드
    const mdp::MapView mv{b.lmap + (size_t)env * c.size * c.size, c.size, m.gw, m.gh};
    const int Rg = mdp::cand_gain_cells(c.map_max_range, c.res);
    for (int it = tid; it < mdp::hier::K * mdp::hier::kGainRays; it += kRT) {
      const int k = it / mdp::hier::kGainRays;
      if (cs.c[k].n <= 0) continue;
      const int g = mdp::cand_gain_ray(mv, cs.c[k].cell % m.gw, cs.c[k].cell / m.gw, it % mdp::hier::kGainRays, Rg);
      if (g) atomicAdd(&s_util[k], g);
    }
  } else for (int it = tid; it < mdp::hier::K * rows; it += kRT) {
    const int k = it / rows;
    if (cs.c[k].n <= 0) continue;
    const int cx = cs.c[k].cell % m.gw, y = cs.c[k].cell / m.gw + it % rows - R;
    if (y < 0 || y >= m.gh) continue;
    const int x0 = max(0, cx - R), x1 = min(m.gw - 1, cx + R);
    int n = 0;
    for (int w = x0 >> 5; w <= (x1 >> 5); ++w) {
      const int b0 = max(x0 - w * 32, 0), b1 = min(x1 - w * 32, 31);
      const uint32_t mk = (b1 == 31 ? ~0u : ((1u << (b1 + 1)) - 1u)) & ~((1u << b0) - 1u);
      n += __popc(FM[y * ws + w] & mk);
    }
    atomicAdd(&s_util[k], n);
  }
  __syncthreads();
  if (tid == 0) {
    for (int k = 0; k < mdp::hier::K; ++k) cs.c[k].util = s_util[k];
    mdp::obsgen::obs_cand(b, e, env, cs);
  }
}

// 관측: 칸마다(모든 스레드) → 합치기·숫자(스레드 0). statics: 늘 0 인 물체 채널(📏)도 씀 — 판이 시작할 때만 (observations.h obs_cells)
__device__ void write_obs(const EnvBatch& b, EnvState& e, int env, Shared& sh, int tid, const mdp::obsgen::FarDirty& fd, bool statics) {
  mdp::obsgen::obs_cells(b, e, env, sh, tid, fd, false, statics);
  obs_far_block(b, e, env, tid, fd, statics);
  __syncthreads();
  if (tid == 0) mdp::obsgen::obs_finish(b, e, env, sh, false);   // 프론티어 덩어리 칸은 env_reach_kernel
  __syncthreads();
}

// 물리 걸음을 k_next 부터 다음 라이다 장(그 장 준비까지) 또는 행동 끝까지. 장이 있으면 pending = 1
__device__ void physics_until_scan(const EnvBatch& b, EnvState& e, Shared& sh) {
  const EnvCfg& c = b.cfg;
  sh.pending = 0;
  int k = sh.k_next;
  while (k < c.substeps) {
    if (!e.collided) physics_step(b, e);
    e.tick += 1;
    const bool scan = (sh.tick0 + k + 1) % c.lidar_every == 0;
    ++k;
    if (scan) {
      scan_begin(b, e, sh);
      sh.pending = 1;
      break;
    }
  }
  sh.k_next = k;
}

// 걸음 첫머리: 이번 걸음의 장·지도 갱신 집계를 비움 (행동을 정한 뒤)
__device__ void step_begin(Shared& sh, const EnvState& e) {
  sh.new_wall = sh.new_floor = sh.d_known = sh.n_new = sh.n_occ = 0;
  mdp::obsgen::far_dirty_clear(sh);
  sh.scanned = 0;
  sh.k_next = 0;
  sh.tick0 = e.tick;
}

__global__ void __launch_bounds__(kPhysThreads) env_phys_kernel(EnvBatch b, const float* actions, int first, int sub) {
  const int env = blockIdx.x * kPhysThreads + threadIdx.x;
  if (env >= b.n_env) return;
  if (sub > 0 && b.done[env] != 0) return;   // 계층형 묶음: 판이 이 묶음 앞 걸음에서 끝남 — 묶음 끝까지 쉼
  if (gdec_idle(b, env, sub)) return;          // 그래프 모드: 결정이 앞 걸음에서 끝남
  EnvState& e = b.st[env];
  Shared& sh = b.scratch[env];
  if (first) {
    e.collided = 0;
    if (sub == 0) gdec_begin(b, e, env);
    apply_action(b, e, env, actions + mdp::act::DIM * env);
    step_begin(sh, e);
  }
  physics_until_scan(b, e, sh);
}

// 계층형 행동 (걸음 첫머리 — 다른 행동 방식에서 env_phys_kernel 의 first 몫): 블록 = 환경 하나, 스레드 kActT.
// 예전에는 환경마다 스레드 하나가 apply_action 을 통째로 돌아, 그 안의 긴 직렬 루프(막힘까지 거리 · RPP 충돌 검사의 굴린 발자국 · 제자리 회전 검사 ·
// collision_monitor 의 점 × 굴린 자세)가 걸음마다 커널 시간을 정했다 (커널 시간 = 가장 오래 걸린 스레드). 여기서는 그 루프들만 블록 스레드가 나눠 돌고
// (hier_path.h 의 t0·stride — 개수는 정수 합, 거리²는 정수 최솟값, 걸림은 OR 이라 나눠도 답이 같다), 차례가 있는 나머지(path_step_a·b·c)는 스레드 0 이
// CPU 참조판과 같은 함수로 돈다 → 값은 비트까지 그대로
constexpr int kActT = 128;
__global__ void __launch_bounds__(kActT) env_act_kernel(EnvBatch b, const float* actions, int sub) {
  const int env = blockIdx.x, tid = threadIdx.x;
  if (sub > 0 && b.done[env] != 0) return;   // 계층형 묶음: 판이 이 묶음 앞 걸음에서 끝남 — 묶음 끝까지 쉼
  if (gdec_idle(b, env, sub)) return;          // 그래프 모드: 결정이 앞 걸음에서 끝남
  __shared__ PathPre pre;
  __shared__ PathStep ps;
  __shared__ MonPoses mp;
  __shared__ int cnt[65];   // collision_monitor: 굴린 자세마다 점 수 [64] · 지금 발자국 안 점 수
  __shared__ int imm;
  const EnvCfg& c = b.cfg;
  EnvState& e = b.st[env];
  __syncthreads();   // 위 판정(블록 전체 같은 값)을 모두 읽은 뒤에 스레드 0 이 상태를 고친다
  if (tid == 0) {
    e.collided = 0;
    if (sub == 0) gdec_begin(b, e, env);
    const float* a = actions + (size_t)lowrl_act_w(b) * env;
    if (b.lowrl) lowrl_begin(b, e, env);
    else action_begin(c, e, a);
    if (lowrl_act(b, e, env, a)) step_begin(b.scratch[env], e);   // 아래층 RL: 정책이 모는 환경 — 추종기 없이 명령 (low_rl.h)
    pre.od_best = 1 << 30;
    pre.rot_cnt = 0;
    pre.now_hit = 0;
    pre.arr_hit = 0;
    imm = 0;
  }
  if (lowrl_policy(b, env)) return;   // 블록 전체 같은 값 (LowCmd 는 이 커널이 안 고침)
  if (tid < 65) cnt[tid] = 0;
  __syncthreads();
  {   // 걸음 앞 상태만으로 정해지는 값 (쓰일지는 추종기가 정한다 — 안 쓰이면 버림)
    const int od = c.path_cost_reg ? path_obstacle_d2(b, e, env, path_cs_r(c), tid, kActT) : (1 << 30);
    const int rc = c.mon_on ? path_rotate_count(b, e, env, tid, kActT) : 0;
    const bool nh = path_now_hits(b, e, env, tid, kActT);
    if (od < (1 << 30)) atomicMin(&pre.od_best, od);
    if (rc) atomicAdd(&pre.rot_cnt, rc);
    if (nh) atomicOr(&pre.now_hit, 1);
  }
  __syncthreads();
  if (path_imminent_now(c, pre.now_hit != 0)) {
    if (tid == 0) pre.arr_hit = 1;
  } else if (path_imminent_roll(b, e, env, 0.f, c.rot_w, 1e9f, pre.now_hit != 0, false, tid, kActT)) {
    atomicOr(&pre.arr_hit, 1);
  }
  __syncthreads();
  if (tid == 0) path_step_a(b, e, env, &pre, ps);
  __syncthreads();
  if (ps.want_imm) {   // RPP 충돌 검사 (블록 전체 같은 값)
    if (path_imminent_now(c, pre.now_hit != 0)) {
      if (tid == 0) imm = 1;
    } else if (path_imminent_roll(b, e, env, ps.vc, ps.wc, ps.cd, pre.now_hit != 0, false, tid, kActT)) {
      atomicOr(&imm, 1);
    }
  }
  __syncthreads();
  if (tid == 0) {
    path_step_b(b, e, env, imm != 0, &pre, ps);
    path_monitor_poses(b, env, ps.v, ps.w, mp);
  }
  __syncthreads();
  if (mp.n > 0) {
    int lc[64], l0 = 0;
    for (int j = 0; j < mp.n; ++j) lc[j] = 0;
    path_monitor_count(b, e, env, mp, tid, kActT, lc, &l0);
    if (l0) atomicAdd(&cnt[64], l0);
    for (int j = 0; j < mp.n; ++j)
      if (lc[j]) atomicAdd(&cnt[j], lc[j]);
  }
  __syncthreads();
  if (tid == 0) {
    path_step_c(b, e, env, path_monitor_result(c, mp, cnt, cnt[64]), ps);
    step_begin(b.scratch[env], e);
  }
}

constexpr int kSums = 5;
__device__ void block_sum(int v[kSums], int tid) {
  __shared__ int acc[kSums];
  if (tid < kSums) acc[tid] = 0;
  __syncthreads();
  for (int q = 0; q < kSums; ++q) {
    const int s = __reduce_add_sync(0xffffffffu, v[q]);
    if ((tid & 31) == 0) atomicAdd(&acc[q], s);
  }
  __syncthreads();
  for (int q = 0; q < kSums; ++q) v[q] = acc[q];
}

// 프론티어 칸 지도 고치기 (지도 갱신 뒤): 칸 종류가 바뀐 칸이 있는 낱말(4 칸)과 그 4 이웃(위·아래 줄 4 칸, 왼·오른 끝 칸)의 비트를 frontier_cell 로 다시 —
// 다른 칸은 칸 종류도 이웃도 그대로라 비트가 같다.
// 목록이 넘치면 이번 장 범위 ±1 전체를 다시. 같은 칸을 여러 스레드가 고쳐도 같은 지도에서 같은 값
__device__ void frontier_fix(const EnvBatch& b, const EnvState& e, int env, const Shared& sh, int tid, int n_chg) {
  const EnvCfg& c = b.cfg;
  const MapDev& m = b.maps[e.map];
  const mdp::MapView mv{b.lmap + (size_t)env * c.size * c.size, c.size, m.gw, m.gh};
  const int ws = fmap_row(c);
  uint32_t* F = b.fmap + (size_t)env * c.size * ws;
  const auto fix = [&](int x, int y) {
    if (x < 0 || y < 0 || x >= m.gw || y >= m.gh) return;
    const uint32_t bit = 1u << (x & 31);
    uint32_t* w = F + y * ws + (x >> 5);
    if (mdp::frontier_cell(mv, x, y)) {
      if (!(*w & bit)) atomicOr(w, bit);
    } else if (*w & bit) {
      atomicAnd(w, ~bit);
    }
  };
  if (n_chg <= kClassLog) {
    const int32_t* list = b.fchg + (size_t)env * kClassLog;
    for (int k = tid; k < 14 * n_chg; k += kThreads) {   // 낱말마다 14 칸: 줄 y−1·y·y+1 의 4 칸 + (x0 − 1, y) + (x0 + 4, y)
      const int ci = list[k / 14], x0 = ci % c.size, y = ci / c.size, r = k % 14;
      if (r < 12) fix(x0 + (r & 3), y + r / 4 - 1);
      else fix(r == 12 ? x0 - 1 : x0 + 4, y);
    }
  } else if (sh.bx1 >= 0) {
    const int x0 = sh.bx0 - 1, y0 = sh.by0 - 1, w = sh.bx1 - sh.bx0 + 3, h = sh.by1 - sh.by0 + 3;
    for (int k = tid; k < w * h; k += kThreads) fix(x0 + k % w, y0 + k / w);
  }
}

__global__ void __launch_bounds__(kThreads, 6) env_apply_kernel(EnvBatch b) {   // 6 블록: reach_fix 가 레지스터를 늘려 동시 블록이 줄지 않게
  const int env = blockIdx.x, tid = threadIdx.x;
  Shared& sh = b.scratch[env];
  if (!sh.pending) return;   // 블록 전체가 같은 값
  EnvState& e = b.st[env];
  __shared__ int n_chg;
  if (tid == 0) n_chg = 0;
  __syncthreads();
  const ClassLog log{b.fchg + (size_t)env * kClassLog, &n_chg};
  const ApplyCount n = scan_apply(b, e, env, sh, tid, kThreads, b.scratch[env].far_mask, &log);   // 로그 오즈가 바뀐 칸 → 먼 지도 칸 표시
  int v[kSums] = {n.wall, n.floor, n.known, n.fresh, n.occ};
  block_sum(v, tid);   // 끝에 __syncthreads — 지도 갱신이 다 보임
  frontier_fix(b, e, env, sh, tid, n_chg);
  reach_fix(b, e, env, sh, tid, n_chg);
  if (tid == 0) {
    sh.new_wall += v[0];
    sh.new_floor += v[1];
    sh.d_known += v[2];
    sh.n_new += v[3];
    sh.n_occ += v[4];
    known_box(b, e, sh);
    sh.pending = 0;
  }
}

// 상태(EnvState)를 블록 스레드 모두가 낱말(4 바이트)씩 옮긴다 — 스레드 하나가 구조체 전체를 옮기는 동안 블록이 기다리지 않게
__device__ void copy_state(EnvState& dst, const EnvState& src, int tid) {
  static_assert(sizeof(EnvState) % 4 == 0, "EnvState 낱말 복사");
  const uint32_t* s = (const uint32_t*)&src;
  uint32_t* d = (uint32_t*)&dst;
  for (int i = tid; i < (int)(sizeof(EnvState) / 4); i += kThreads) d[i] = s[i];
}

// 그래프 모드 결정 끝 판정 (graph_env.h gdec_step) — 지나는 칸은 env_graph_kernel 과 같은 비트 판
__device__ void gdec_step_dev(const EnvBatch& b, const EnvState& e, int env, int sub) {
  if (!b.gdec) return;
  const EnvCfg& c = b.cfg;
  const MapDev& m = b.maps[e.map];
  const int ws = fmap_row(c), nfw = c.size * ws;
  const uint32_t* RB = b.rbits + (size_t)env * kReachBits * nfw;
  const uint32_t* OCC = RB + kRbOcc * nfw;
  const uint32_t* KN = RB + kRbKnown * nfw;
  const uint32_t* ST = RB + kRbStand * nfw;
  const auto bit = [&](const uint32_t* P, int x, int y) -> uint32_t { return (P[y * ws + (x >> 5)] >> (x & 31)) & 1u; };
  gdec_step(b, e, env, sub, [&](int x, int y) {
    if (x < 0 || y < 0 || x >= m.gw || y >= m.gh) return false;
    return c.graph_stand ? bit(ST, x, y) != 0u : (bit(KN, x, y) && !bit(OCC, x, y));
  });
}

// sub < 0: 기본 (행동 하나 = 걸음 하나). sub ≥ 0: 계층형 묶음의 sub 번째 걸음 — 보상·항을 묶음 합에 더하고, 판이 끝나면 리셋·관측을 묶음 끝
// (env_bundle_end_kernel)으로 미룬다. 앞 걸음에서 끝난 환경은 쉼
__global__ void __launch_bounds__(kThreads) env_finish_kernel(EnvBatch b, int sub) {
  const int env = blockIdx.x, tid = threadIdx.x;
  if (sub > 0 && b.done[env] != 0) return;   // 블록 전체 같은 값 (b.done 은 아래 첫 __syncthreads 뒤에만 씀)
  if (gdec_idle(b, env, sub)) return;          // 그래프 모드: 결정이 앞 걸음에서 끝남 (b.gdec 은 스레드 0 이 아래 __syncthreads 뒤에만 씀)
  __shared__ EnvState e;
  __shared__ Shared sh;   // 관측용 (프론티어 합)
  Shared& g = b.scratch[env];
  copy_state(e, b.st[env], tid);
  __syncthreads();
  if (tid == 0) {
    // 마지막 장 뒤 남은 물리 걸음 (장 수만큼 1~3 을 돌렸으므로 여기선 장이 없다)
    while (g.k_next < b.cfg.substeps) {
      if (!e.collided) physics_step(b, e);
      e.tick += 1;
      g.k_next += 1;
    }
    map_progress(e, g.n_occ);   // 진행 없음 시계 — finish_step 판 끝이 씀
    finish_step(b, e, env, g.new_wall, g.new_floor, g.d_known, g.n_new);
    gdec_step_dev(b, e, env, sub);   // 그래프 모드: 결정 끝 판정 (CPU step_env 와 같은 자리)
    mdp::obsgen::far_dirty_robot(b, e, g);   // 지나간 곳 표시도 먼 지도 다시 그릴 곳에
    b.reward[env] = e.reward;
    b.done[env] = e.done;
    if (g.scanned) ring_advance(e);
    if (sub >= 0) {   // 계층형: 결정 동안 합 (걸음 차례대로 더함 — CPU 참조판과 같은 차례)
      const float* T = b.reward_terms + (size_t)env * mdp::rew::kTerms;
      float* A = b.tacc + (size_t)env * mdp::rew::kTerms;
      if (sub == 0) {
        b.racc[env] = e.reward;
        for (int k = 0; k < mdp::rew::kTerms; ++k) A[k] = T[k];
      } else {
        b.racc[env] += e.reward;
        for (int k = 0; k < mdp::rew::kTerms; ++k) A[k] += T[k];
      }
    }
  }
  __syncthreads();
  if (g.scanned) push_bins(b, e, env, g, tid);
  __syncthreads();
  const bool was_done = e.done != 0;
  __syncthreads();
  if (was_done && sub >= 0) {   // 계층형: 리셋·관측은 묶음 끝에
    copy_state(b.st[env], e, tid);
    if (tid == 0) g.cam = mdp::obsgen::depth_camera(b.cfg, e);
    return;
  }
  if (was_done) reset_env(b, e, env, tid);
  __shared__ mdp::obsgen::FarDirty fd;
  if (tid == 0) {
    fd = mdp::obsgen::far_dirty_take(b, e, g, was_done);
  }
  write_obs(b, e, env, sh, tid, fd, was_done);   // 끝에 __syncthreads
  copy_state(b.st[env], e, tid);
  if (tid == 0) g.cam = mdp::obsgen::depth_camera(b.cfg, e);
}

__global__ void env_reset_kernel(EnvBatch b) {
  const int env = blockIdx.x, tid = threadIdx.x;
  __shared__ EnvState e;
  __shared__ Shared sh;
  copy_state(e, b.st[env], tid);
  __syncthreads();
  reset_env(b, e, env, tid);
  const mdp::obsgen::FarDirty all{1, 0.f, 0.f, 0.f, 0.f};
  write_obs(b, e, env, sh, tid, all, true);
  if (tid == 0) {
    b.scratch[env].far_ok = 0;   // 다음 걸음은 다시 전부 (판 기록은 그때 남김)
    b.scratch[env].cam = mdp::obsgen::depth_camera(b.cfg, e);
  }
  copy_state(b.st[env], e, tid);
  if (tid == 0) {
    b.reward[env] = 0.f;
    b.done[env] = 0;
    for (int k = 0; k < mdp::rew::kTerms; ++k) b.reward_terms[(size_t)env * mdp::rew::kTerms + k] = 0.f;
  }
}

// ---------- 계층형 (kActPath, include/hier_path.h): 결정 때 경로 계획 · 묶음 끝 ----------
// 고른 후보 번호 (행동 a[0], 정수 값) = hier_path.h slot_of
// 환경 하나 = 블록 하나, 묶음 첫 걸음 앞. 후보 칸(관측 때 b.cand) → 목표. 같은 후보(그 덩어리가 지금 목표 칸을 품음)이고 남은 경로가 아직
// 지난번 계획과 같은 칸 기준(엄격·느슨)으로 설 수 있는 칸 위면 그대로, 아니면 무게 있는 최단 경로(hier_path.h 머리말 "약한 막힘"):
// 먼저 엄격(막힘 = 지도 막힘 ∪ 장애물 층 약한 막힘)으로, 못 닿으면 느슨(지도 막힘만)으로. 비트 판 층 BFS(Dial, 버킷 4 개) → 거리 mod 7 로 뒤로 짚기 → 웨이포인트.
// CPU 참조판 env_ref.cpp plan_cpu 와 같은 경로 (env_verify --hier)
__global__ void __launch_bounds__(kRT) env_plan_kernel(EnvBatch b, const float* actions) {
  const int env = blockIdx.x, tid = threadIdx.x;
  EnvState& e = b.st[env];
  const EnvCfg& c = b.cfg;
  const MapDev& m = b.maps[e.map];
  const int ws = fmap_row(c), nf = c.size * ws;
  const uint32_t* RB = b.rbits + (size_t)env * kReachBits * nf;
  const uint32_t* OCC = RB + kRbOcc * nf;
  const uint32_t* KN = RB + kRbKnown * nf;
  const uint32_t* ST = RB + kRbStand * nf;
  const int16_t* L = b.lmap + (size_t)env * c.size * c.size;
  __shared__ int s_plan, s_target, s_start, s_tgt_ok, s_goal, s_force;
  __shared__ unsigned long long s_key;
  const int by0 = e.kb_y0, by1 = e.kb_y1, bw0 = e.kb_x0 >> 5, bw1 = e.kb_x1 >> 5;
  const auto in_box = [&](int x, int y) { return x >= 0 && y >= 0 && x < m.gw && y < m.gh && y >= by0 && y <= by1 && (x >> 5) >= bw0 && (x >> 5) <= bw1; };
  const auto bit_of = [&](const uint32_t* P, int x, int y) -> uint32_t { return in_box(x, y) ? (P[y * ws + (x >> 5)] >> (x & 31)) & 1u : 0u; };
  uint32_t* P = b.pplanes + (size_t)env * kPathPlanes * nf;
  uint32_t* C3 = P + kPpC3 * nf;
  uint32_t* C45 = P + kPpC45 * nf;
  uint32_t* D0 = P + kPpD * nf;
  uint32_t* D1 = D0 + nf;
  uint32_t* D2 = D1 + nf;
  uint32_t* LT = P + kPpLT * nf;
  uint32_t* SST = P + kPpSST * nf;
  uint32_t* LO = P + kPpLO * nf;     // 느슨 막힘 = 지도 막힘 ∪ 지금 스캔 (path_scan 일 때만 씀)
  uint32_t* SLO = P + kPpSLO * nf;   // 그 설 수 있음
  const bool scan = c.path_scan != 0 || b.cvox != nullptr || path_rb_plan(c, mdp::obsgen::reach_robot(c)).r2 != mdp::obsgen::reach_robot(c).r2;   // 느슨 판 LO·SLO 를 따로 만드나 (지금 스캔 · 깊이 카메라 층 · 계획 반경)
  const auto Bp = [&](int t) { return P + (size_t)(kPpB + (t & 3)) * nf; };
  const bool due = lowrl_due(b, e, env);   // 아래층 RL: 계획 차례인 환경만 (끄면 늘 true). 고치는 것은 아래 동기화 뒤 스레드 0
  if (tid == 0 && due) {
    int force = 0;   // 복구 (hier_path.h path_recovery): 다시 계획 · 포기한 목표면 다른 후보(없으면 −1)
    const int k_in = lowrl_policy(b, env) ? 0 : (actions && c.low_auto != 2 ? slot_of(b, actions[(size_t)lowrl_act_w(b) * env]) : -1);   // 정책 환경: 행동 줄은 속도 — 목표는 LowCmd 하나
    const int k = path_recovery(b, e, env, k_in, &force);
    e.p_slot = k;
    const int tgt = k >= 0 ? act_cells(b, env)[k] : -1;
    s_force = force;
    s_target = tgt;
    if (tgt < 0) {
      e.goal_cell = -1;
      path_set_none(e);
    }
    s_plan = tgt >= 0 && e.kb_x1 >= 0 ? 1 : 0;
    s_start = -1;
  }
  __syncthreads();
  if (!due) return;   // 블록 전체 같은 값
  if (tid == 0 && b.lowrl) lowrl_planned(b, e, env);
  if (!s_plan) {   // 빈 후보: 목표 없음. 아는 칸이 없으면 목표만 (경로 없이 섬)
    if (tid == 0 && s_target >= 0) {
      e.goal_cell = s_target;
      path_set_none(e);
    }
    return;   // 블록 전체 같은 값
  }
  const int nbw = bw1 - bw0 + 1, nbox = nbw * (by1 - by0 + 1);
  int rcx, rcy;
  cell_of(c, m, e.x + e.ex, e.y + e.ey, &rcx, &rcy);
  const mdp::Robot rb = path_rb_plan(c, mdp::obsgen::reach_robot(c)), rm = path_rb_mid(c), ro = path_rb_out(c);   // rb: 계획 반경 (path_consistent 면 충돌 검사 발자국 반경, hier_path.h)
  // 엄격 막힘 판 LT = 지도 막힘 ∪ 약한 막힘(로그 오즈 > 0, 로봇 칸에서 obstacle_max_range 안 — path_lethal 과 같은 규칙), 엄격 설 수 있음 SST
  // LT 는 범위 둘레(줄 ± 팽창 반경, 낱말 ± 1)까지 — stand_word 가 거기를 읽는다 (범위 밖은 모름이라 지도 막힘과 같지만 판에는 옛 값이 남아 있음)
  const float rr = c.path_obs_range * c.path_obs_range;
  const int ly0 = max(0, by0 - ro.r), ly1 = min(m.gh - 1, by1 + ro.r), lw0 = max(0, bw0 - 1), lw1 = min(ws - 1, bw1 + 1), lnw = lw1 - lw0 + 1;
  for (int k = tid; k < lnw * (ly1 - ly0 + 1); k += kRT) {
    const int y = ly0 + k / lnw, wx = lw0 + k % lnw, wi = y * ws + wx;
    uint32_t w = OCC[wi];
    for (int q = 0; q < 32; ++q) {
      const int x = wx * 32 + q;
      if (x >= m.gw) break;
      const int16_t l = L[y * c.size + x];
      if (l == mdp::kUnknownL || l <= 0) continue;
      const float dx = (float)(x - rcx) * c.res, dy = (float)(y - rcy) * c.res;
      if (dx * dx + dy * dy <= rr) w |= 1u << q;
    }
    LT[wi] = w;
    if (scan) LO[wi] = OCC[wi];
  }
  __syncthreads();
  // 지금 스캔 끝점 (hier_path.h "지금 스캔" (a)) → 엄격·느슨 막힘 둘 다. 범위(LT 를 채운 줄·낱말) 밖 점은 둘레 계산에 안 닿으므로 버림 (CPU 와 같은 결과)
  if (b.cvox) {   // 깊이 카메라 장애물 층 (hier_path.h): 지금 스캔과 같은 몫 — 엄격·느슨 둘 다 막힘. 범위는 LT 를 채운 줄·낱말 (CPU ex0..ex1 보다 넓어도 둘레 계산 결과 같음)
    const uint8_t* V = b.cvox + (size_t)env * c.size * c.size;
    for (int k = tid; k < lnw * (ly1 - ly0 + 1); k += kRT) {
      const int y = ly0 + k / lnw, wx = lw0 + k % lnw, wi = y * ws + wx;
      uint32_t w = 0u;
      for (int q = 0; q < 32; ++q) {
        const int x = wx * 32 + q;
        if (x >= m.gw) break;
        if (V[y * c.size + x]) w |= 1u << q;
      }
      LT[wi] |= w;
      LO[wi] |= w;
    }
    __syncthreads();
  }
  if (c.path_scan) {
    for (int k = tid; k < mdp::obs::LIDAR_BINS; k += kRT) {
      int x, y;
      if (!path_scan_cell(b, e, env, k, &x, &y)) continue;
      if (y < ly0 || y > ly1 || (x >> 5) < lw0 || (x >> 5) > lw1) continue;
      atomicOr(&LT[y * ws + (x >> 5)], 1u << (x & 31));
      atomicOr(&LO[y * ws + (x >> 5)], 1u << (x & 31));
    }
    __syncthreads();
  }
  for (int k = tid; k < nbox; k += kRT) {
    const int y = by0 + k / nbw, wx = bw0 + k % nbw, wi = y * ws + wx;
    SST[wi] = stand_word(LT, KN, ws, m.gh, y, wx, rb);
    if (scan) SLO[wi] = stand_word(LO, KN, ws, m.gh, y, wx, rb);
  }
  __syncthreads();
  const uint32_t* LPASS = scan ? SLO : ST;    // 느슨 계획의 설 수 있음 · 무게 경계 막힘
  const uint32_t* LOBS = scan ? LO : OCC;
  // 같은 후보 · 경로 그대로? (지난 계획이 엄격이면 엄격 칸, 느슨이면 느슨 칸으로 남은 경로 확인)
  if (tid == 0) {
    const int k = e.p_slot;
    const bool same = !s_force && act_same(b, e, env, k) && e.goal_cell >= 0 && e.p_n > 0 && !e.p_trunc &&
                      path_clear(e, [&](int x, int y) { return bit_of(e.p_mode == 1 ? SST : LPASS, x, y) != 0u; });
    if (same) s_plan = 0;
    else {
      e.goal_cell = s_target;
      path_set_none(e);
      e.p_plans += 1;
    }
    // 엄격으로 짤 목표 칸이 있나: 목표가 설 수 있거나, 목표에서 path_goal_tol 안 설 수 있는 칸 (hier_path.h "계획·충돌 검사 일치", CPU any_goal)
    const int gx = s_target % m.gw, gy = s_target / m.gw;
    const mdp::Robot rt = path_rb_far(c);
    int ok = bit_of(SST, gx, gy) != 0u ? 1 : 0;
    for (int dy = -rt.r; dy <= rt.r && !ok && rt.r2 > 0; ++dy)
      for (int dx = -rt.r; dx <= rt.r && !ok; ++dx)
        ok = dx * dx + dy * dy <= rt.r2 && bit_of(SST, gx + dx, gy + dy) != 0u;
    s_tgt_ok = ok;
  }
  __syncthreads();
  if (!s_plan) return;
  const int target = s_target, tx = target % m.gw, ty = target / m.gw, twi = ty * ws + (tx >> 5);
  const mdp::Robot rt = path_rb_tol(c), rf = path_rb_far(c);
  const int tol_w = 2 * rt.r + 1, n_reg = rt.r2 > 0 ? tol_w * tol_w : 0;
  const int far_w = 2 * rf.r + 1, n_far = rf.r2 > 0 ? far_w * far_w : 0;
  const uint32_t tbit = 1u << (tx & 31);
  int32_t* LST = b.rlist + (size_t)env * mdp::kReachCap * kReachLists;
  // 한 번 계획: PASS = 설 수 있는 칸, OBS = 무게 경계를 셀 막힘 판. 닿으면 true (D0..D2 에 거리 mod 7, C3·C45 에 무게 경계)
  const auto run = [&](const uint32_t* PASS, const uint32_t* OBS) -> bool {
    // 목표가 PASS 면 목표 하나(옛 길), 아니면 목표 둘레 path_goal_tol 안 칸 중 처음 닿는 층의 (거리², 칸) 최소 — CPU plan_cpu 와 같은 규칙
    const bool exact = bit_of(PASS, tx, ty) != 0u || n_reg == 0;
    int any = 0;
    for (int k = tid; k < nbox; k += kRT) {
      const int y = by0 + k / nbw, wx = bw0 + k % nbw, wi = y * ws + wx;
      C3[wi] = stand_word(OBS, KN, ws, m.gh, y, wx, rm);
      C45[wi] = stand_word(OBS, KN, ws, m.gh, y, wx, ro);
      for (int q = 0; q < 4; ++q) Bp(q)[wi] = 0u;
      D0[wi] = D1[wi] = D2[wi] = 0u;
      any |= PASS[wi] != 0u;
    }
    any = __syncthreads_or(any);
    if (tid == 0) {
      s_start = -1;
      if (any && rcx >= 0 && rcy >= 0 && rcx < m.gw && rcy < m.gh)
        s_start = stand_start(m, rcx, rcy, PASS, ws, by0, by1, bw0, bw1, LST + 2 * mdp::kReachCap, LST + 4 * mdp::kReachCap);
      if (s_start >= 0) Bp(0)[(s_start / m.gw) * ws + ((s_start % m.gw) >> 5)] |= 1u << ((s_start % m.gw) & 31);
    }
    __syncthreads();
    if (s_start < 0) return false;
    // 층 t: 거리 t 로 정해지는 칸 F = 버킷[t] & ~정해진 칸 → 거리 mod 7 + 1 을 3 비트 판에, 이웃(설 수 있고 안 정해진 칸)을 무게만큼 뒤 버킷에
    const int t_max = 3 * nbox * 32 + 8;   // 못 닿을 때의 안전 상한
    for (int t = 0; t < t_max; ++t) {
      uint32_t* Bt = Bp(t);
      uint32_t* Bold = Bp(t + 3);   // = 지난 층 F (버킷 t − 1) — 이제 비움 (이번 층 밀기가 무게 3 칸을 여기 쓴다)
      const int code = t % 7 + 1;
      int a = 0, pend = 0, h = 0;
      for (int k = tid; k < nbox; k += kRT) {
        const int wi = (by0 + k / nbw) * ws + bw0 + k % nbw;
        Bold[wi] = 0u;
        const uint32_t f = Bt[wi] & ~(D0[wi] | D1[wi] | D2[wi]);
        Bt[wi] = f;
        if (f) {
          if (code & 1) D0[wi] |= f;
          if (code & 2) D1[wi] |= f;
          if (code & 4) D2[wi] |= f;
          a = 1;
          if (wi == twi && (f & tbit)) h = 1;
        }
        pend |= (Bp(t + 1)[wi] | Bp(t + 2)[wi]) != 0u;
      }
      a = __syncthreads_or(a);
      pend = __syncthreads_or(pend);
      h = __syncthreads_or(h);
      if (h) {
        if (tid == 0) s_goal = target;
        __syncthreads();
        return true;
      }
      if (!exact && a) {
        if (tid == 0) s_key = ~0ull;
        __syncthreads();
        for (int q = tid; q < n_reg; q += kRT) {
          const int dx = q % tol_w - rt.r, dy = q / tol_w - rt.r, x = tx + dx, y = ty + dy;
          if (dx * dx + dy * dy > rt.r2 || !bit_of(Bt, x, y)) continue;
          atomicMin(&s_key, ((unsigned long long)(dx * dx + dy * dy) << 20) | (unsigned long long)(y * m.gw + x));
        }
        __syncthreads();
        if (s_key != ~0ull) {
          const int g = (int)(s_key & 0xfffffull);
          __syncthreads();
          if (tid == 0) s_goal = g;
          __syncthreads();
          return true;
        }
      }
      if (!a && !pend) {   // BFS 끝: 먼 허용 안 정해진 칸 중 (거리², 칸) 최소 (CPU far_goal)
        if (n_far == 0) return false;
        if (tid == 0) s_key = ~0ull;
        __syncthreads();
        for (int q = tid; q < n_far; q += kRT) {
          const int dx = q % far_w - rf.r, dy = q / far_w - rf.r, x = tx + dx, y = ty + dy;
          if (dx * dx + dy * dy > rf.r2 || !bit_of(PASS, x, y) || !(bit_of(D0, x, y) | bit_of(D1, x, y) | bit_of(D2, x, y))) continue;
          atomicMin(&s_key, ((unsigned long long)(dx * dx + dy * dy) << 20) | (unsigned long long)(y * m.gw + x));
        }
        __syncthreads();
        const unsigned long long key = s_key;
        __syncthreads();
        if (key == ~0ull) return false;
        if (tid == 0) s_goal = (int)(key & 0xfffffull);
        __syncthreads();
        return true;
      }
      if (!a) continue;
      for (int k = tid; k < nbox; k += kRT) {
        const int y = by0 + k / nbw, wx = bw0 + k % nbw, wi = y * ws + wx;
        const uint32_t f = Bt[wi], fl = wx > bw0 ? Bt[wi - 1] : 0u, fr = wx < bw1 ? Bt[wi + 1] : 0u;
        const uint32_t fu = y > by0 ? Bt[wi - ws] : 0u, fd = y < by1 ? Bt[wi + ws] : 0u;
        const uint32_t n = ((f << 1) | (f >> 1) | (fl >> 31) | (fr << 31) | fu | fd) & PASS[wi] & ~(D0[wi] | D1[wi] | D2[wi]);
        if (!n) continue;
        const uint32_t c45 = C45[wi], c3 = C3[wi];
        Bp(t + 1)[wi] |= n & c45;
        Bp(t + 2)[wi] |= n & c3 & ~c45;
        Bp(t + 3)[wi] |= n & ~c3;
      }
      __syncthreads();
    }
    return false;
  };
  int mode = 0;
  if (s_tgt_ok && run(SST, LT)) mode = 1;           // 엄격 (블록 전체 같은 값)
  else if (run(LPASS, LOBS)) mode = 2;              // 느슨: 목표가 약한 막힘 둘레 안이거나 엄격으로 못 닿음 (지금 스캔은 느슨에서도 막힘)
  else if (c.path_consistent && rb.r2 != mdp::obsgen::reach_robot(c).r2) {   // 계획 반경으로 못 짬: 느슨 막힘 + 관측 BFS 반경 (hier_path.h (e)) — SLO 판을 고쳐 씀
    const mdp::Robot r3 = mdp::obsgen::reach_robot(c);
    for (int k = tid; k < nbox; k += kRT) {
      const int y = by0 + k / nbw, wx = bw0 + k % nbw;
      SLO[y * ws + wx] = stand_word(LO, KN, ws, m.gh, y, wx, r3);
    }
    __syncthreads();
    if (run(SLO, LO)) mode = 3;
  }
  if (tid != 0 || !mode) return;
  const uint32_t* PASS = mode == 1 ? SST : LPASS;
  // 뒤로 짚기 (스레드 0): 이웃 차례 nb4 에서 처음으로 거리 ≡ 내 거리 − 내 무게 (mod 7) 인 칸
  const int start = s_start;
  const auto dcode = [&](int x, int y) { return (int)(bit_of(D0, x, y) | (bit_of(D1, x, y) << 1) | (bit_of(D2, x, y) << 2)); };
  const auto wfn = [&](int x, int y) { return path_weight(bit_of(PASS, x, y) != 0u, bit_of(C3, x, y) != 0u, bit_of(C45, x, y) != 0u); };
  int32_t* list = LST + 6 * mdp::kReachCap;
  int n = 0, v = s_goal;
  list[n++] = v;
  while (v != start) {
    const int x = v % m.gw, y = v / m.gw;
    const int pc = ((dcode(x, y) - 1 - wfn(x, y)) % 7 + 7) % 7 + 1;
    int u = -1;
    for (int k = 0; k < 4 && u < 0; ++k) {
      int dx, dy;
      mdp::nb4(k, &dx, &dy);
      if (dcode(x + dx, y + dy) == pc) u = (y + dy) * m.gw + x + dx;
    }
    if (u < 0 || n >= mdp::kReachCap) return;   // 오지 않음 (거리 장이 맞으면 앞 칸이 늘 있음) — 경로 없이 섬
    v = u;
    list[n++] = v;
  }
  for (int i = 0; i < n / 2; ++i) { const int t = list[i]; list[i] = list[n - 1 - i]; list[n - 1 - i] = t; }
  int trunc = 0;
  e.p_n = path_simplify(list, n, m.gw, wfn, e.wp, mdp::hier::kPathMax, &trunc);
  e.p_i = 0;
  e.p_trunc = trunc;
  e.p_mode = mode;
  if (mode >= 2) e.p_fb += 1;
}

// 깊이 카메라 장애물 층 갱신 (환경 하나 = 블록 하나, 결정 앞 — hier_path.h "깊이 카메라 장애물 층"). CPU 참조판 env_ref.cpp hier_camlayer_cpu 와 같은 결과:
// 모든 픽셀 지우기(원자 AND) → 모든 픽셀 표시(원자 OR) → 몸통 밑 지우기 — 끄기·켜기라 스레드 차례와 무관
__global__ void __launch_bounds__(kRT) env_camlayer_kernel(EnvBatch b) {
  const int env = blockIdx.x, tid = threadIdx.x;
  const EnvState& e = b.st[env];
  if (!lowrl_due(b, e, env)) return;   // 아래층 RL: 계획 차례인 환경만 (끄면 늘 true)
  uint32_t* V = reinterpret_cast<uint32_t*>(b.cvox + (size_t)env * b.cfg.size * b.cfg.size);
  for (int i = tid; i < mdp::obs::DEPTH_SIZE; i += kRT)
    cam_pixel(b, e, env, i, 0, [&](int ci, uint8_t bit) { atomicAnd(&V[ci >> 2], ~((uint32_t)bit << (8 * (ci & 3)))); });
  __syncthreads();
  for (int i = tid; i < mdp::obs::DEPTH_SIZE; i += kRT)
    cam_pixel(b, e, env, i, 1, [&](int ci, uint8_t bit) { atomicOr(&V[ci >> 2], (uint32_t)bit << (8 * (ci & 3))); });
  for (int col = tid; col < mdp::obs::DEPTH_W; col += kRT) cam_column(b, e, env, col);
  __syncthreads();
  if (tid == 0) cam_compact(b, env);
  const int nfp = cam_fp_cells(b.cfg);
  for (int j = tid; j < nfp; j += kRT) {
    const int ci = cam_fp_cell(b, e, j);
    if (ci >= 0) atomicAnd(&V[ci >> 2], ~(0xffu << (8 * (ci & 3))));
  }
}

// 묶음 끝 (환경 하나 = 블록 하나): 보상·항 = 묶음 합, 판이 끝난 환경은 리셋 + 새 판 관측 (끝난 걸음 env_finish_kernel 이 미룬 것 — 그 커널의 was_done 길과 같은 차례)
__global__ void __launch_bounds__(kThreads) env_bundle_end_kernel(EnvBatch b) {
  const int env = blockIdx.x, tid = threadIdx.x;
  if (tid == 0) b.reward[env] = b.racc[env];
  if (tid < mdp::rew::kTerms) b.reward_terms[(size_t)env * mdp::rew::kTerms + tid] = b.tacc[(size_t)env * mdp::rew::kTerms + tid];
  if (b.done[env] == 0) return;   // 블록 전체 같은 값
  __shared__ EnvState e;
  __shared__ Shared sh;
  Shared& g = b.scratch[env];
  copy_state(e, b.st[env], tid);
  __syncthreads();
  reset_env(b, e, env, tid);
  __shared__ mdp::obsgen::FarDirty fd;
  if (tid == 0) fd = mdp::obsgen::far_dirty_take(b, e, g, true);
  write_obs(b, e, env, sh, tid, fd, true);
  copy_state(b.st[env], e, tid);
  if (tid == 0) g.cam = mdp::obsgen::depth_camera(b.cfg, e);
}

// ---------- 아래층 RL (include/low_rl.h — CPU 참조판 env_ref.cpp low_prepare_env 와 같은 차례·같은 함수) ----------
// 자동 목표: 환경 하나 = 블록 하나, 스레드마다 탐침 하나 → 가장 작은 열쇠
__global__ void __launch_bounds__(kLrProbes) env_low_goal_kernel(EnvBatch b) {
  const int env = blockIdx.x, tid = threadIdx.x;
  if (b.cfg.low_auto == 2 && low_xenv(b, env)) {   // 탐사 판 학습: 위층 0 번 후보 규칙 (스레드 하나)
    if (tid == 0) lowrl_goal_explore(b, b.st[env], env);
    return;
  }
  if (!b.lowrl[env].need_goal) return;   // 블록 전체 같은 값 (아래 둘째 동기화 뒤 스레드 0 만 고침)
  __shared__ unsigned long long key;
  if (tid == 0) key = ~0ull;
  __syncthreads();
  const EnvState& e = b.st[env];
  const unsigned long long k = lowrl_goal_probe(b, e, env, tid, path_rb_plan(b.cfg, mdp::obsgen::reach_robot(b.cfg)));
  if (k != ~0ull) atomicMin(&key, k);
  __syncthreads();
  if (tid == 0) lowrl_goal_set(b, e, env, key);
}
// 정책 관측: 숫자·걸음 사이 값은 스레드 0, 그림은 스레드가 나눠
constexpr int kLowPackT = 256;
__global__ void __launch_bounds__(kLowPackT) env_low_pack_kernel(EnvBatch b) {
  const int env = blockIdx.x, tid = threadIdx.x;
  EnvState& e = b.st[env];
  if (tid == 0) lowrl_pack(b, e, env);   // p_i 만 고침 — 그림은 자세·지도만 읽음
  uint8_t* O = b.low_img + (size_t)env * kLowImg;
  for (int i = tid; i < kLowImg; i += kLowPackT) O[i] = lowrl_img(b, e, env, i);
}
__global__ void env_low_clear_kernel(EnvBatch b) {
  const int env = blockIdx.x * blockDim.x + threadIdx.x;
  if (env < b.n_env) lowrl_clear(b.lowrl[env]);
}

// ---------- 소프트웨어 광선 묶음: 스레드마다 광선 하나씩 순회(trav_step)하고, 끝난 스레드는 워프 단위로 다음 광선을 받는다 ----------
// 광선마다 걸음 순서는 raycast 와 같아 결과가 같고, 워프 안 광선 길이 차이로 노는 스레드가 없어진다.
// G: make(b, job, sv, o, d, tmax) → 쏠 광선이면 true (아니면 none(b, job)), store(b, job, t, tri)
// ---------- 그래프 모드 관측 (mdp/graph.h — CPU 참조 include/obs_ref.h graph_ref 와 비트 같음, env_verify --graph) ----------
// 환경 하나 = 블록 하나, 결정 관측(묶음 끝·판 시작) 뒤. 지도 비트 판(막힘·앎·설 수 있음)·프론티어 칸 비트(fmap)·지도 자세만 읽는다.
// 차례는 graph.h 머리말 1 ~ 8. 합치는 곳은 모두 차례 무관(비트 OR · 수 더하기 · 최단 거리 고정점)이라 스레드 차례와 무관하게 같은 값.
// gate: 0 모든 환경 (판 시작·다시 만들기) · 1 묶음 끝, 리셋 앞: 안 끝난 환경은 다음 결정 관측 + 결정 결과·전작 방식 판 끝·보상 (graph_finish),
//   걸음에서 끝난 환경은 결과만 · 2 묶음 끝 리셋 뒤: 끝난 환경의 새 판 관측만
__global__ void __launch_bounds__(kRT) env_graph_kernel(EnvBatch b, int gate) {
  using namespace mdp::graph;
  namespace gg = mdp::graphgen;
  __shared__ uint32_t s_valid[N / 32], s_guide[N / 32];
  __shared__ int s_dist[N], s_util[N];
  __shared__ short s_list[N];
  __shared__ int s_cur, s_forced, s_chg, s_nf, s_nn, s_gone, s_nu;
  const int env = blockIdx.x, tid = threadIdx.x;
  if (gate == 1 && b.done[env] != 0) {   // 블록 전체 같은 값 (b.done 은 맨 끝 스레드 0 만 씀)
    if (tid == 0) graph_finish(b, b.st[env], env, -1, 0);
    return;
  }
  if (gate == 2 && b.done[env] == 0) return;
  const EnvState& e = b.st[env];
  const EnvCfg& c = b.cfg;
  const MapDev& m = b.maps[e.map];
  const int ws = fmap_row(c), nfw = c.size * ws;
  const uint32_t* RB = b.rbits + (size_t)env * kReachBits * nfw;
  const uint32_t* OCC = RB + kRbOcc * nfw;
  const uint32_t* KN = RB + kRbKnown * nfw;
  const uint32_t* ST = RB + kRbStand * nfw;
  const uint32_t* FM = b.fmap + (size_t)env * nfw;
  const gg::Params p = graph_params(c, m, e);
  const auto in_map = [&](int x, int y) { return x >= 0 && y >= 0 && x < m.gw && y < m.gh; };
  const auto bit = [&](const uint32_t* P, int x, int y) -> uint32_t { return (P[y * ws + (x >> 5)] >> (x & 31)) & 1u; };
  const auto free = [&](int x, int y) { return in_map(x, y) && bit(KN, x, y) && !bit(OCC, x, y); };
  const auto pass = [&](int x, int y) { return c.graph_stand ? (in_map(x, y) && bit(ST, x, y) != 0u) : free(x, y); };
  const uint32_t* GF = b.gfront + (size_t)env * nfw;
  const auto front = [&](int x, int y) -> bool {   // 유틸리티가 세는 프론티어 칸: graph_open 이면 닫힘을 막는 칸(gfront, 건드린 범위 안만 이번 관측 값), 아니면 모든 프론티어 칸
    if (!c.graph_open) return bit(FM, x, y) != 0u;
    return x >= e.kb_x0 && x <= e.kb_x1 && y >= e.kb_y0 && y <= e.kb_y1 && bit(GF, x, y) != 0u;
  };
  uint32_t* E = b.gedge + (size_t)env * N;
  uint32_t* VIS = b.gvis + (size_t)env * (N / 32);
  int32_t* REP = b.gwork + (size_t)env * gg::kVoxMax;
  int32_t* PREV = b.gprev + (size_t)env * gg::kVoxMax;   // 지난 관측의 복셀 대표 칸 (사라진 프론티어 세기 — graph.h graph_build prev)
  const bool first = e.step == 0;
  if (tid < N / 32) {
    s_valid[tid] = 0u;
    s_guide[tid] = 0u;
    if (e.step == 0) VIS[tid] = 0u;
  }
  for (int i = tid; i < N; i += kRT) {
    E[i] = 0u;
    s_dist[i] = gg::kInf;
    s_util[i] = 0;
  }
  if (tid == 0) s_nf = s_nn = s_gone = s_nu = 0;
  __syncthreads();
  for (int i = tid; i < N; i += kRT)
    if (gg::node_raw(p, i, pass)) atomicOr(&s_valid[i >> 5], 1u << (i & 31));
  __syncthreads();
  const auto valid = [&](int i) { return ((s_valid[i >> 5] >> (i & 31)) & 1u) != 0u; };
  if (tid == 0) {
    int rx, ry, forced;
    cell_of(c, m, e.x + e.ex, e.y + e.ey, &rx, &ry);
    const int cur = gg::find_cur(p, rx, ry, valid, pass, &forced);
    s_cur = cur;
    s_forced = forced;
    s_valid[cur >> 5] |= 1u << (cur & 31);
    VIS[cur >> 5] |= 1u << (cur & 31);
    s_dist[cur] = 0;
    b.gcur[env] = cur;
  }
  __syncthreads();
  const int cur = s_cur, forced = s_forced;
  for (int i = tid; i < N; i += kRT) {   // 간선: 번호 큰 쪽 이웃(스텐실 뒤 절반)만 보고 양쪽에 적음
    if (!valid(i) || (forced && i == cur)) continue;
    for (int k = NB / 2; k < NB; ++k) {
      const int j = gg::node_nb(i, k);
      if (j < 0 || !valid(j) || (forced && j == cur) || !gg::edge_ok(p, i, j, pass)) continue;
      atomicOr(&E[i], 1u << k);
      atomicOr(&E[j], 1u << (NB - 1 - k));
    }
  }
  __syncthreads();
  for (;;) {   // cur 에서 최단 거리: 바뀜이 없을 때까지 풀기 (거리는 줄기만 하고 고정점은 하나)
    if (tid == 0) s_chg = 0;
    __syncthreads();
    for (int i = tid; i < N; i += kRT) {
      if (!valid(i)) continue;
      const int d = gg::relax(s_dist, E[i], i);
      if (d < s_dist[i]) {
        s_dist[i] = d;
        s_chg = 1;
      }
    }
    __syncthreads();
    const int ch = s_chg;
    __syncthreads();
    if (!ch) break;
  }
  for (int i = tid; i < N; i += kRT)   // cur 와 안 이어진 노드는 버림
    if (valid(i) && s_dist[i] >= gg::kInf) {
      atomicAnd(&s_valid[i >> 5], ~(1u << (i & 31)));
      E[i] = 0u;
    }
  __syncthreads();
  const int nv = gg::vox_count(p);
  for (int v = tid; v < nv; v += kRT) {   // 프론티어 복셀 대표 칸 목록 (차례는 무관 — 세기만 한다)
    const int cell = gg::vox_rep(p, v, front);
    if (cell >= 0) REP[atomicAdd(&s_nf, 1)] = cell;
    if (!first && PREV[v] >= 0 && PREV[v] != cell) atomicAdd(&s_gone, 1);
    PREV[v] = cell;
  }
  for (int rep = 0;; ++rep) {   // keep_vis 2 면 유틸리티 노드가 없을 때 방문 표시를 지우고 한 번 더 (graph.h graph_build 와 같은 차례)
    for (int i = tid; i < N; i += kRT)
      if (valid(i) && (p.keep_vis == 1 ? i != cur : !((VIS[i >> 5] >> (i & 31)) & 1u))) s_list[atomicAdd(&s_nn, 1)] = (short)i;
    __syncthreads();
    const int nn = s_nn;
    for (int q = tid; q < nn * s_nf; q += kRT) {   // (노드, 복셀) 쌍마다
      const int i = s_list[q / s_nf];
      if (gg::sees(p, i, REP[q % s_nf], free)) atomicAdd(&s_util[i], 1);
    }
    __syncthreads();
    for (int i = tid; i < N; i += kRT) {
      if (s_util[i] <= p.min_util) s_util[i] = 0;
      else atomicAdd(&s_nu, 1);
    }
    __syncthreads();
    if (p.keep_vis != 2 || s_nu > 0 || s_nf == 0 || rep == 1) break;   // 블록 전체 같은 값
    if (tid < N / 32) VIS[tid] = tid == (cur >> 5) ? 1u << (cur & 31) : 0u;
    if (tid == 0) s_nn = 0;   // s_util 은 모두 0 (유틸리티 노드가 없었다)
    __syncthreads();
  }
  if (p.cover && s_nu == 0 && s_nf > 0) {   // 남은 프론티어 맡기기 (graph.h graph_build 6b — 우리 수정). 블록 전체 같은 값
    const int nf = s_nf;
    for (int f = tid; f < nf; f += kRT) {
      const int j = gg::cover_pick(p, cur, REP[f], valid, [&](int i) { return ((VIS[i >> 5] >> (i & 31)) & 1u) != 0u; }, free);
      if (j >= 0) atomicAdd(&s_util[j], 1);
    }
    __syncthreads();
    for (int i = tid; i < N; i += kRT)
      if (s_util[i] > 0) atomicAdd(&s_nu, 1);
    __syncthreads();
  }
  for (int i = tid; i < N; i += kRT) {   // 지역 길잡이: 유틸리티 노드에서 앞 노드를 따라 cur 까지 (이미 표시된 노드에서 멈춤 — 그 뒤는 표시한 쪽이 잇는다)
    if (s_util[i] <= 0) continue;
    for (int v = i; v >= 0 && v != cur && !((s_guide[v >> 5] >> (v & 31)) & 1u); v = gg::pred_of(s_dist, E[v], v)) atomicOr(&s_guide[v >> 5], 1u << (v & 31));
  }
  __syncthreads();
  for (int i = tid; i < N; i += kRT)
    gg::write_feat(b.gfeat + ((size_t)env * N + i) * FEAT, i, cur, valid(i), s_util[i], ((s_guide[i >> 5] >> (i & 31)) & 1u) != 0u, p);
  if (tid < NB) {
    const int j = ((E[cur] >> tid) & 1u) ? gg::node_nb(cur, tid) : -1;
    b.gcand[(size_t)env * NB + tid] = j >= 0 ? gg::node_y(p, j) * p.gw + gg::node_x(p, j) : -1;
  }
  if (tid == 0) {
    graph_glob(b, e, env, p, cur);
    if (gate == 1 && graph_finish(b, b.st[env], env, s_nu, s_gone)) b.done[env] = b.st[env].done;   // s_nu·s_gone 은 위 __syncthreads 앞에서 다 셈
  }
}

// 결정이 아직 안 끝난 환경 수 → counters[kCntGraphRun] (env_step_gpu 가 걸음 루프를 일찍 끝내는 데만 씀 — 결과는 안 바뀜)
__global__ void env_graph_active_kernel(EnvBatch b, int* out) {
  const int env = blockIdx.x * blockDim.x + threadIdx.x;
  if (env < b.n_env && b.done[env] == 0 && b.gdec[env].reason == kGdRun) atomicAdd(out, 1);
}

constexpr int kRayThreads = 128;
constexpr int kRayChunk = 64, kRayRefill = 16, kRaySteps = 4;   // 워프가 한 번에 받는 광선 수, 이만큼 놀면 채움
enum : int { kCntDepNext = 0, kCntDepSide = 1, kCntDepSideNext = 2, kCntBeamNext = 3, kCntBeamSide = 4, kCntBeamSideNext = 5, kCntGraphRun = 6, kCounters = 16 };

template <class G>
__global__ void __launch_bounds__(kRayThreads) ray_kernel(EnvBatch b, G g, int n_static, const int* n_dev, int* next) {
  const int n = n_dev ? *n_dev : n_static;
  const unsigned lane = threadIdx.x & 31u;
  RayTrav r;
  TravStack k;
  SceneView sv;
  int job = 0;
  bool active = false;
  int pool = 0, pool_end = 0;   // 워프가 받아 둔 광선 번호 [pool, pool_end) — 워프 안 같은 값
  bool exhausted = false;
  while (true) {
    const unsigned need = __ballot_sync(0xffffffffu, !active);
    const int n_need = __popc(need);
    if (n_need >= kRayRefill || n_need == 32) {   // 노는 스레드가 어느 만큼 모이면 한꺼번에 채운다
      if (pool == pool_end && !exhausted) {
        int base = 0;
        if (lane == 0) base = atomicAdd(next, kRayChunk);
        base = __shfl_sync(0xffffffffu, base, 0);
        if (base >= n) exhausted = true;
        else { pool = base; pool_end = min(base + kRayChunk, n); }
      }
      const int avail = pool_end - pool;
      if (!active) {
        const int rank = __popc(need & ((1u << lane) - 1u));
        if (rank < avail) {
          job = pool + rank;
          V3 o, d;
          float tmax;
          if (g.make(b, job, sv, o, d, tmax)) {
            trav_init(r, k, sv, o, d, tmax);
            active = true;
          } else {
            g.none(b, job);
          }
        }
      }
      pool += min(avail, n_need);
    }
    if (!__any_sync(0xffffffffu, active)) {
      if (exhausted && pool == pool_end) break;
      continue;
    }
    for (int s = 0; s < kRaySteps && active; ++s)   // 받기 검사 사이 걸음 몇 개
      if (!trav_step(r, k, sv)) {
        int tri;
        const float t = trav_result(r, &tri);
        g.store(b, job, t, tri);
        active = false;
      }
  }
}

template <class G>
void launch_rays(const EnvBatch& b, G g, int n_static, const int* n_dev, int counter, cudaStream_t st) {
  static int blocks = 0;
  if (!blocks) {
    int dev, sms, per;
    cudaGetDevice(&dev);
    cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per, ray_kernel<G>, kRayThreads, 0);
    blocks = sms * (per > 0 ? per : 1);
  }
  cudaMemsetAsync(b.counters + counter, 0, sizeof(int), st);
  ray_kernel<G><<<blocks, kRayThreads, 0, st>>>(b, g, n_static, n_dev, b.counters + counter);
}

// 깊이 가운데 광선: job = env · DEPTH_SIZE + 픽셀
struct DepthCenter {
  __device__ bool make(const EnvBatch& b, int job, SceneView& sv, V3& o, V3& d, float& tmax) const {
    const int env = job / mdp::obs::DEPTH_SIZE, i = job % mdp::obs::DEPTH_SIZE;
    const EnvState& e = b.st[env];
    const EnvCfg& c = b.cfg;
    float u, v;
    mdp::obsgen::depth_uv(c.depth, i, &u, &v);
    if (!depth_valid(c.depth.spec, u, v)) return false;
    const Pose cam = mdp::obsgen::depth_camera(c, e);
    sv = with_people(b.maps[e.map].scene, e.people);
    o = cam.p;
    d = depth_dir(c.depth.spec, cam, u, v);
    tmax = depth_tmax(c.depth.spec);
    return true;
  }
  __device__ void none(const EnvBatch& b, int job) const { b.dep_tri[job] = -1; }
  __device__ void store(const EnvBatch& b, int job, float t, int tri) const { b.dep_z[job] = t; b.dep_tri[job] = tri; }
};

// 깊이 옆 광선: job = 큐 칸 → 픽셀 · 2 + (0 왼쪽 u − edge_px, 1 오른쪽 u + edge_px)
struct DepthSide {
  __device__ bool make(const EnvBatch& b, int job, SceneView& sv, V3& o, V3& d, float& tmax) const {
    const int q = b.dep_jobs[job], px = q >> 1;
    const int env = px / mdp::obs::DEPTH_SIZE, i = px % mdp::obs::DEPTH_SIZE;
    const EnvState& e = b.st[env];
    const EnvCfg& c = b.cfg;
    float u, v;
    mdp::obsgen::depth_uv(c.depth, i, &u, &v);
    const Pose cam = mdp::obsgen::depth_camera(c, e);
    sv = with_people(b.maps[e.map].scene, e.people);
    o = cam.p;
    d = depth_dir(c.depth.spec, cam, (q & 1) ? u + c.depth.edge_px : u - c.depth.edge_px, v);
    tmax = depth_tmax(c.depth.spec);
    return true;
  }
  __device__ void none(const EnvBatch&, int) const {}
  __device__ void store(const EnvBatch& b, int job, float t, int) const { b.dep_side[b.dep_jobs[job]] = t; }
};

// 라이다 빔 가운데 광선: job = env · kMaxBeams + j (이번 장이 있는 환경의 j < n_beams 만)
struct BeamCenter {
  __device__ bool make(const EnvBatch& b, int job, SceneView& sv, V3& o, V3& d, float& tmax) const {
    const int env = job / kMaxBeams, j = job % kMaxBeams;
    if (!beam_live(b, env, j)) return false;
    const Shared& sh = b.scratch[env];
    const EnvState& e = b.st[env];
    BeamGeom g;
    beam_center_ray(e.noise, sh.scan, j, g);
    sv = with_people(b.maps[e.map].scene, e.people);
    o = g.o;
    d = g.d;
    tmax = e.noise.max_range;
    return true;
  }
  __device__ void none(const EnvBatch&, int) const {}
  __device__ void store(const EnvBatch& b, int job, float t, int tri) const { b.beam_t[job] = t; b.beam_tri[job] = tri; }
};

// 라이다 옆 광선: job = 큐 칸 → 빔 · 4 + k
struct BeamSide {
  __device__ bool make(const EnvBatch& b, int job, SceneView& sv, V3& o, V3& d, float& tmax) const {
    const int q = b.beam_jobs[job], bj = q >> 2;
    const int env = bj / kMaxBeams, j = bj % kMaxBeams;
    const Shared& sh = b.scratch[env];
    const EnvState& e = b.st[env];
    BeamGeom g;
    beam_center_ray(e.noise, sh.scan, j, g);
    sv = with_people(b.maps[e.map].scene, e.people);
    o = g.o;
    d = beam_side_dir(e.noise, sh.scan, g, q & 3);
    tmax = e.noise.max_range;
    return true;
  }
  __device__ void none(const EnvBatch&, int) const {}
  __device__ void store(const EnvBatch& b, int job, float t, int) const { b.beam_side[b.beam_jobs[job]] = t; }
};

// 빔마다 가운데 광선 뒤: 끝나면 지도 표시, 옆 광선이 있어야 하면 큐에 — PER = 4 (소프트웨어: 광선마다 칸 하나, 빔 · 4 + k) · 1 (RT 코어: 빔)
template <int PER>
__global__ void __launch_bounds__(kBeamThreads) env_beam_kernel(EnvBatch b) {
  const int env = blockIdx.y, j = blockIdx.x * kBeamThreads + threadIdx.x;
  if (!beam_live(b, env, j)) return;
  const int job = env * kMaxBeams + j;
  if (beam_center_done(b, env, j, b.beam_t[job], b.beam_tri[job])) return;
  const int q = atomicAdd(b.counters + kCntBeamSide, PER);
  if (PER == 1) b.beam_jobs[q] = job;
  else
    for (int k = 0; k < 4; ++k) b.beam_jobs[q + k] = 4 * job + k;
}

// 옆 광선 큐의 빔마다(PER 칸 묶음) 그 빔을 마무리 — 큐에 없는 빔은 첫 패스에서 끝났다
template <int PER>
__global__ void __launch_bounds__(kDepthThreads) env_beam_side_kernel(EnvBatch b) {
  const int g = blockIdx.x * kDepthThreads + threadIdx.x;
  if (PER * g >= b.counters[kCntBeamSide]) return;
  const int job = PER == 1 ? b.beam_jobs[g] : b.beam_jobs[4 * g] >> 2;
  beam_side_done(b, job / kMaxBeams, job % kMaxBeams, b.beam_t[job], b.beam_tri[job], b.beam_side + 4 * job);
}

// 픽셀마다 가운데 광선 뒤: 값이 정해지면 쓰고, 아니면 큐에 — PER = 2 (소프트웨어: 광선마다 칸 하나, 픽셀 · 2 + 왼0/오른1) · 1 (RT 코어: 픽셀)
template <int PER>
__global__ void __launch_bounds__(kDepthThreads) env_depth_pixel_kernel(EnvBatch b) {
  const int px = blockIdx.x * kDepthThreads + threadIdx.x;
  if (px >= b.n_env * mdp::obs::DEPTH_SIZE) return;
  const int tri = b.dep_tri[px];
  if (depth_px_center(b, px, b.dep_z[px], tri, tri == -1 ? V3{} : depth_px_dir(b, px))) return;
  const int q = atomicAdd(b.counters + kCntDepSide, PER);
  if (PER == 1) {
    b.dep_jobs[q] = px;
    return;
  }
  b.dep_jobs[q] = 2 * px;
  b.dep_jobs[q + 1] = 2 * px + 1;
}

// 옆 광선 큐의 픽셀마다(PER 칸 묶음) 그 픽셀을 마무리 — 큐에 없는 픽셀은 첫 패스에서 이미 썼다
template <int PER>
__global__ void __launch_bounds__(kDepthThreads) env_depth_side_kernel(EnvBatch b) {
  const int pair = blockIdx.x * kDepthThreads + threadIdx.x;
  if (PER * pair >= b.counters[kCntDepSide]) return;
  const int px = PER == 1 ? b.dep_jobs[pair] : b.dep_jobs[2 * pair] >> 1;
  depth_px_side(b, px, b.dep_z[px], b.dep_tri[px], depth_px_dir(b, px), b.dep_side + 2 * px);
}

#ifdef SIM_OPTIX
// RT 코어 (src/optix_rays.cu): raygen 이 광선을 만들어 쏜다 (깊이 옆 광선은 마무리까지). 1·3 은 픽셀·빔 전부, 2·4 는 큐 길이(장치, counter 칸)만큼
void rt_launch(const EnvBatch& b, RtProg prog, int n, int counter, cudaStream_t st) {
  RtParams p{};
  p.prog = prog;
  p.batch = b.dev_self;
  p.fast = b.rt_fast;
  p.n = n;
  if (prog == kRtDepthSide || prog == kRtBeamSide) p.n_dev = b.counters + counter;
  ((OptixRt*)b.rt)->launch(p, st);
}
#endif

void beams_gpu(const EnvBatch& b, cudaStream_t st) {
  const int n = b.n_env * kMaxBeams;
#ifdef SIM_OPTIX
  const dim3 grid((kMaxBeams + kBeamThreads - 1) / kBeamThreads, b.n_env);
  if (b.rt) {   // raygen 은 광선만 (결과 뒤는 아래 커널)
    rt_launch(b, kRtBeam, n, 0, st);
    cudaMemsetAsync(b.counters + kCntBeamSide, 0, sizeof(int), st);
    env_beam_kernel<1><<<grid, kBeamThreads, 0, st>>>(b);
    rt_launch(b, kRtBeamSide, n, kCntBeamSide, st);
    env_beam_side_kernel<1><<<(n + kDepthThreads - 1) / kDepthThreads, kDepthThreads, 0, st>>>(b);
    return;
  }
#endif
  launch_rays(b, BeamCenter{}, n, nullptr, kCntBeamNext, st);
  cudaMemsetAsync(b.counters + kCntBeamSide, 0, sizeof(int), st);
  env_beam_kernel<4><<<dim3((kMaxBeams + kBeamThreads - 1) / kBeamThreads, b.n_env), kBeamThreads, 0, st>>>(b);
  launch_rays(b, BeamSide{}, 0, b.counters + kCntBeamSide, kCntBeamSideNext, st);
  env_beam_side_kernel<4><<<(n + kDepthThreads - 1) / kDepthThreads, kDepthThreads, 0, st>>>(b);   // 옆 광선을 쏜 빔만 마무리
}

void depth_gpu(const EnvBatch& b, cudaStream_t st) {
  const int n_px = b.n_env * mdp::obs::DEPTH_SIZE;
  const int px_blocks = (n_px + kDepthThreads - 1) / kDepthThreads;
#ifdef SIM_OPTIX
  if (b.rt) {
    rt_launch(b, kRtDepth, n_px, 0, st);
    cudaMemsetAsync(b.counters + kCntDepSide, 0, sizeof(int), st);
    env_depth_pixel_kernel<1><<<px_blocks, kDepthThreads, 0, st>>>(b);
    rt_launch(b, kRtDepthSide, n_px, kCntDepSide, st);
    env_depth_side_kernel<1><<<px_blocks, kDepthThreads, 0, st>>>(b);
    return;
  }
#endif
  launch_rays(b, DepthCenter{}, n_px, nullptr, kCntDepNext, st);
  cudaMemsetAsync(b.counters + kCntDepSide, 0, sizeof(int), st);
  env_depth_pixel_kernel<2><<<px_blocks, kDepthThreads, 0, st>>>(b);
  launch_rays(b, DepthSide{}, 0, b.counters + kCntDepSide, kCntDepSideNext, st);
  env_depth_side_kernel<2><<<px_blocks, kDepthThreads, 0, st>>>(b);   // 옆 광선을 쏜 픽셀만 마무리
}

}  // namespace

// 요약 커널을 깊이 커널과 겹쳐 돌림: 둘은 서로 다른 메모리만 쓴다(요약 = obs_1d 의 덩어리 칸·vec_prev 그 칸, 깊이 = obs_img 깊이 칸).
// 옆 스트림에서 돌리고 st 가 끝을 기다린다 — 걸음 결과·차례는 그대로
namespace {
struct ReachStream {
  cudaStream_t s = nullptr;
  cudaEvent_t go = nullptr, done = nullptr;
};
ReachStream& reach_stream() {
  static ReachStream r;
  if (!r.s) {
    cudaStreamCreateWithFlags(&r.s, cudaStreamNonBlocking);
    cudaEventCreateWithFlags(&r.go, cudaEventDisableTiming);
    cudaEventCreateWithFlags(&r.done, cudaEventDisableTiming);
  }
  return r;
}
void reach_and_depth(const EnvBatch& b, cudaStream_t st, int gate = 0) {
  ReachStream& r = reach_stream();
  cudaEventRecord(r.go, st);
  cudaStreamWaitEvent(r.s, r.go, 0);
  env_reach_kernel<<<b.n_env, kRT, 0, r.s>>>(b, gate, -1);
  cudaEventRecord(r.done, r.s);
  depth_gpu(b, st);
  cudaStreamWaitEvent(st, r.done, 0);
}
}  // namespace

namespace {
// 행동 하나 (속도 행동 주기). sub < 0: 기본 · sub ≥ 0: 계층형 묶음의 sub 번째 걸음 (깊이는 묶음 끝에만 — 정책이 묶음 끝 관측만 보고 깊이는 상태만의 함수)
void step_once(const EnvBatch& b, const float* actions, cudaStream_t st, int sub) {
  const EnvCfg& c = b.cfg;
  // 한 행동의 라이다 장 수 ≤ ⌈substeps / lidar_every⌉ — 장마다 물리·빔·지도 갱신 한 바퀴
  const int n_scan = (c.substeps + c.lidar_every - 1) / c.lidar_every;
  const int phys_blocks = (b.n_env + kPhysThreads - 1) / kPhysThreads;
  const bool staged = c.action_mode == kActPath;   // 계층형: 행동(걸음 첫머리)은 env_act_kernel 이 — 무거운 검사를 블록 스레드로 나눠 본다
  for (int s = 0; s < n_scan; ++s) {
    if (s == 0 && staged) env_act_kernel<<<b.n_env, kActT, 0, st>>>(b, actions, sub);
    env_phys_kernel<<<phys_blocks, kPhysThreads, 0, st>>>(b, actions, s == 0 && !staged, sub);
    beams_gpu(b, st);
    env_apply_kernel<<<b.n_env, kThreads, 0, st>>>(b);
  }
  env_finish_kernel<<<b.n_env, kThreads, 0, st>>>(b, sub);
  if (sub < 0) reach_and_depth(b, st);
  else env_reach_kernel<<<b.n_env, kRT, 0, st>>>(b, 1, sub);   // 덩어리 칸 관측의 지연 차례(vec_prev)를 걸음마다 맞춤 — 끝난 환경은 묶음 끝에
}
}  // namespace

// 아래층 RL 준비 단계 (low_rl.h 머리말): [새 목표] → 깊이 카메라 층·경로 계획(차례인 환경만) → 정책 관측. actions: 규칙 추종 환경(follow 0)의 후보 번호 (없으면 nullptr)
void env_low_prepare_gpu(const EnvBatch& b, const float* actions, cudaStream_t st) {
  if (!b.lowrl) return;
  if (b.cfg.low_auto) env_low_goal_kernel<<<b.n_env, kLrProbes, 0, st>>>(b);
  if (b.cvox) env_camlayer_kernel<<<b.n_env, kRT, 0, st>>>(b);
  env_plan_kernel<<<b.n_env, kRT, 0, st>>>(b, actions);
  env_low_pack_kernel<<<b.n_env, kLowPackT, 0, st>>>(b);
}

void env_step_gpu(const EnvBatch& b, const float* actions, cudaStream_t st) {
  if (b.cfg.action_mode != kActPath) {
    step_once(b, actions, st, -1);
    return;
  }
  if (b.lowrl) {   // 아래층 RL: 한 번 = 걸음 하나 (묶음 길이 1 — 깊이도 걸음마다). 계획은 준비 단계가
    step_once(b, actions, st, 0);
    env_bundle_end_kernel<<<b.n_env, kThreads, 0, st>>>(b);
    reach_and_depth(b, st, 2);
    if (b.cfg.low_auto) env_low_prepare_gpu(b, actions, st);
    return;
  }
  // 계층형: 결정 하나 = 경로 계획 → 속도 행동 decide_steps 걸음 → 묶음 끝 (보상 합·끝난 판 리셋·관측)
  if (b.cvox) env_camlayer_kernel<<<b.n_env, kRT, 0, st>>>(b);   // 깊이 카메라 장애물 층 (cam_on) — 계획 앞
  env_plan_kernel<<<b.n_env, kRT, 0, st>>>(b, actions);
  if (b.gdec) {
    // 그래프 모드: 결정 하나 = 노드 하나 이동. 환경마다 닿거나 실패·시간 초과까지 걷고 끝난 환경은 쉰다 (graph_env.h gdec_step).
    // 결정 주기(decide_steps) 뒤부터 5 걸음마다 남은 환경 수를 읽어 모두 끝났으면 그만 돈다 (쉬는 걸음은 아무것도 안 바꿔 결과는 같음)
    const EnvCfg& c = b.cfg;
    for (int s = 0; s < c.graph_max_steps; ++s) {
      step_once(b, actions, st, s);
      if (s + 1 < c.graph_max_steps && (s + 1) % 5 == 0) {
        int n_run = 0;
        cudaMemsetAsync(b.counters + kCntGraphRun, 0, sizeof(int), st);
        env_graph_active_kernel<<<(b.n_env + 255) / 256, 256, 0, st>>>(b, b.counters + kCntGraphRun);
        cudaMemcpyAsync(&n_run, b.counters + kCntGraphRun, sizeof(int), cudaMemcpyDeviceToHost, st);
        cudaStreamSynchronize(st);
        if (n_run == 0) break;
      }
    }
    env_graph_kernel<<<b.n_env, kRT, 0, st>>>(b, 1);   // 다음 결정 관측 + 결정 결과·전작 방식 판 끝·보상 (리셋 앞)
    env_bundle_end_kernel<<<b.n_env, kThreads, 0, st>>>(b);
    reach_and_depth(b, st, 2);
    env_graph_kernel<<<b.n_env, kRT, 0, st>>>(b, 2);   // 끝난 환경의 새 판 관측
    return;
  }
  for (int s = 0; s < b.cfg.decide_steps; ++s) step_once(b, actions, st, s);
  env_bundle_end_kernel<<<b.n_env, kThreads, 0, st>>>(b);
  reach_and_depth(b, st, 2);
}

void env_graph_gpu(const EnvBatch& b, cudaStream_t st) {
  if (b.gfeat) env_graph_kernel<<<b.n_env, kRT, 0, st>>>(b, 0);
}

void env_reset_gpu(const EnvBatch& b, cudaStream_t st) {
  env_reset_kernel<<<b.n_env, kThreads, 0, st>>>(b);
  reach_and_depth(b, st);
  if (b.gfeat) env_graph_kernel<<<b.n_env, kRT, 0, st>>>(b, 0);
  if (b.lowrl) {
    env_low_clear_kernel<<<(b.n_env + 255) / 256, 256, 0, st>>>(b);
    if (b.cfg.low_auto) env_low_prepare_gpu(b, nullptr, st);
  }
}

}  // namespace sim
