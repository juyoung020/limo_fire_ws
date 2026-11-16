// 위층 고르기 "fuel" (combine_run --upper fuel) — 프론티어 덩어리 전체를 시간 비용으로 도는 순회의 첫 목적지. 호스트(CPU)에서만, 로봇이 아는 것만 씀:
// 로봇 지도(점유 격자) · 지도 자세(참값 + 추정 오차) · 지나간 칸/본 칸 판 · 깊이 카메라 층. 정답 격자는 안 씀.
//
// 틀은 FUEL (Zhou et al., "FUEL: Fast UAV Exploration using Incremental Frontier Structure and Hierarchical Planning", RA-L 2021, arXiv 2010.11561):
//   식 1  t_lb(x_a, x_b) = max( 길이(P(p_a, p_b)) / v_max , min(|ξ_a − ξ_b|, 2π − |ξ_a − ξ_b|) / ξ̇_max )        — 두 시점 사이 시간 하한 (자리·머리 방향)
//   식 3  M(0, k) = t_lb(x_0, x_k) + w_c · c_c(x_k),  식 4  c_c = acos( (p_k − p_0) · v_0 / (|p_k − p_0| |v_0|) )  — 지금 속도와 어긋남 벌점 (w_c = 1.5, VI-A)
//   식 5  M(k, 0) = 0 — 닫힌 순회 문제(ATSP)를 열린 순회로. 논문은 LKH 로 풂(VI-A); 여기는 덩어리 ≤ kExact 면 정확한 동적 계획법, 넘으면 최근접 이웃 + 2-opt · Or-opt
//   IV-C  덩어리마다 가운데 둘레 원통 좌표로 고르게 뽑은 시점 가운데 덮음이 큰 것 · IV-B 큰 덩어리는 주성분(PCA) 첫 축으로 둘로 나눔(되풀이)
// 다른 점: 길이는 로봇 지도 설 수 있는 칸 위 8 이웃 모따기 거리(5·7 가중, 모서리로 안 빠짐 — A* 대신 시작점마다 한 번 번짐), v_max·ξ̇_max 대신 실측 v_eff·ω_eff
//   (combine_run --leg-log 로 잰 구간 시간에 맞춤 — scripts/README "fuel"), 시점 점수 = 예상 새로 보이는 모름 넓이(frontier.h cand_gain_ray) ÷ 로봇에서 시간 (FUEL 은 덮음만).
//
// 부품마다 스위치 (Opt — 모두 끄면 "후보 K 개를 BFS 칸 거리로 도는 순회" = cand_tsp 와 같은 비용 꼴, 단 지금 목표 이어 가기는 이 파일 규칙):
//   time  : 비용 = 식 1 시간 (끄면 BFS 칸 거리 — cand_tsp 와 같음)
//   global: 순회 대상 = 지도 전체 프론티어 덩어리(합침·나눔 뒤 최대 max_nodes) (끄면 시뮬 후보 K 개)
//   vp    : 덩어리마다 시점을 둘레 설 수 있는 칸에서 고름 (끄면 덩어리의 가장 가까운 칸 = 후보 칸)
//   hyst  : 지금 목표 덩어리로 시작하는 순회가 가장 좋은 순회보다 (1 + hyst) 배 안이면 그대로 (0 = 늘 가장 좋은 것)
//   room  : 방 마무리 (TARE-lite) — 로봇 지도의 갈 수 있는 칸을 좁은 곳(벽까지 door_m 안 칸)에서 끊어 방으로 나누고, 로봇 방에 덩어리가 남았는데
//           방 밖으로 가는 구간과 방 밖에서 방으로 되돌아오는 구간에 room_pen 초를 더함 → 순회가 로봇 방 덩어리를 먼저 끝냄 (TARE 의 "지역 먼저, 전역은 나중" 꼴)
#pragma once
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <functional>
#include <vector>

#include "env.h"
#include "world.h"   // CUDA 없이 (실물 노드도 이 파일을 씀)
#include "frontier.h"
#include "observations.h"

namespace fu {

struct Opt {
  bool time = false, global = false, vp = false, room = false;
  float hyst = 0.f;        // 지금 목표 붙들기 여유 (비) — 잠정 시작값 0.2
  float v_eff = 0.70f;     // m/s — 실효 직진 속도 (--leg-log 맞춤으로 바꿈)
  float w_eff = 1.0f;      // rad/s — 실효 회전 속도 (--leg-log 맞춤으로 바꿈)
  float t0 = 1.0f;         // s — 시점 점수 분모에 더하는 고정 시간 (가까운 시점의 나눗셈 폭주 막기, 맞춤의 절편)
  float turn_w = 0.f;      // 로봇에서 첫 구간: 첫 걸음 방향과 지금 머리 방향 차 / ω 를 이 배로 더함 (차동 구동은 먼저 돌아야 함) — 0 = 끔
  float wc = 0.f;          // FUEL 식 3 의 w_c (논문 1.5) — 0 = 끔
  int max_nodes = 20;      // global: 순회에 넣는 덩어리 상한 (가까운 순 2/3 + 큰 순 1/3)
  float merge_m = 1.0f;    // global: 가장 가까운 칸끼리 이 거리 안이고 길 거리 차가 2 배 안이면 합침 (후보 v2 kCand2Merge 와 같은 값)
  float split_m = 1.5f;    // global: 첫 주성분 표준편차가 이 값(m)을 넘으면 둘로 (잠정)
  float room_pen = 10.f;   // room: 로봇 방을 덩어리 남기고 떠나는 구간 벌점 (s, 잠정)
  float door_m = 0.45f;    // room: 벽까지 거리 이 값 미만인 좁은 칸에서 방을 끊음 (잠정, 문 폭 0.8~1.1 m 의 반)
  // 시점 자격 (FUEL frontier_finder — 저자 저장소 HKUST-Aerial-Robotics/FUEL exploration_manager/launch/algorithm.xml 값):
  //   vp_min  : 로봇에서 이 거리(m) 안 시점은 안 씀 (FUEL frontier/min_candidate_dist 0.75 — getTopViewpointsInfo 의 거리 문턱). 덩어리 가장 가까운 칸도 예외 없음.
  //             통과 반경(low_pass_r) 안 목표는 내자마자 "통과" 라 결정만 되풀이된다 — 로봇 몸 둘레(라이다 뒤 140° 사각지대) 프론티어는 제자리 회전이 봄
  //   vp_clear: 시점 둘레 이 거리(m) 안에 모름 칸이 없어야 함 (FUEL frontier/min_candidate_clearance 0.21 — isNearUnknown). 막힘 쪽 간격은 pass (설 수 있음 + 깊이 층)
  //   둘 중 하나라도 > 0 이면: 시점 없는 덩어리는 순회에서 뺌 (FUEL 의 dormant — 시점이 없는 프론티어는 쉬게 둠). 간격(vp_clear) 맞는 시점이 없으면 거리 문턱만으로 한 번 더 찾고,
  //   그래도 없으면 뺌. 모든 덩어리가 빠지면 목표 없음(kNone — 부르는 쪽은 slot0 로 안 떨어지고 제자리 회전: 덩어리는 있는데 다 로봇 곁이면 돌아서 봄).
  //   다른 못 냄(시작 칸 없음 · 덩어리 없음)은 전처럼 −1 (slot0 차례) — kNone 으로 하면 시험 장면 성공이 0.71 → 0.39 (2026-11-16, 진척 없음 끝이 늚)
  float vp_min = 0.f, vp_clear = 0.f;
  bool first = false;      // 순회 대신 덩어리 차례(후보 K 개면 시뮬 후보 차례 = 가장 가까운 순)의 첫 시점 — slot0 를 시점으로 (--fuel-first). 지금 목표가 첫 덩어리면 그대로 (slot0 stick 과 같은 뜻)
};
// slot0 를 시점으로: 시뮬 후보 차례의 첫 덩어리 (순회 · 시간 비용 · 붙들기 · 방 없이), 시점 자격은 base 그대로 — 실물 노드 upper:=slot0 (slot_vp) ·
// combine_run --upper fuel --fuel-first 1 --fuel-global 0 --fuel-time 0 --fuel-vp 1 과 같은 값
inline Opt slot_view(Opt b) { b.first = true; b.global = false; b.time = false; b.vp = true; b.hyst = 0.f; b.room = false; return b; }
constexpr int kNone = -2;   // decide: 프론티어는 있지만 시점 자격이 되는 덩어리가 없음 → 목표 없음 (slot0 로 안 떨어짐)

// 환경마다 기억 — 지금 목표
struct Mem {
  int vp = -1;    // 보낸 목표 칸
  int anc = -1;   // 그 덩어리의 가장 가까운 프론티어 칸 (다음 결정에서 같은 덩어리를 찾는 열쇠)
};

struct Stat {
  long dec = 0, nodes = 0, kept = 0, sw = 0, fallback = 0, exact = 0;
  long slot_n = 0, slot_in = 0, slot_vp = 0;   // 후보 칸으로 나타낼 수 있나: 시뮬 후보가 있는 결정 수 · 고른 덩어리가 후보 K 개 중 하나의 덩어리 · 그 후보 칸에서 0.5 m 안이 목표 칸
  double slot_d = 0;                           // slot_in 결정에서 목표 칸과 그 후보 칸 사이 거리 합 (m)
  // 후보 모음 셈 (전역일 때): 원래 덩어리 · 합침·나눔 뒤 · 피하는 칸 뺀 뒤 수의 합, 상한(max_nodes)에 걸린 결정 · 빠진 덩어리 수 · 그 BFS 길(m)·칸 수(m) 합·최댓값,
  //   cand_cap 에 걸린 결정 · 빠진 후보 수 · 그 길(m)·틈(m) 합·최댓값 · 빠진 것 가운데 가장 앞(gpt 차례)
  long g_raw = 0, g_mid = 0, g_dec = 0, cap_dec = 0, cap_drop = 0, cc_dec = 0, cc_drop = 0;
  double cap_d = 0, cap_dmax = 0, cap_s = 0, cap_smax = 0, cc_d = 0, cc_dmax = 0, cc_s = 0, cc_smax = 0;
  long g_rawmax = 0, g_midmax = 0;
  double ms = 0, ms_max = 0;
  void add(const Stat& o) {
    g_raw += o.g_raw; g_mid += o.g_mid; g_dec += o.g_dec; cap_dec += o.cap_dec; cap_drop += o.cap_drop; cc_dec += o.cc_dec; cc_drop += o.cc_drop;
    cap_d += o.cap_d; cap_s += o.cap_s; cc_d += o.cc_d; cc_s += o.cc_s;
    cap_dmax = std::max(cap_dmax, o.cap_dmax); cap_smax = std::max(cap_smax, o.cap_smax); cc_dmax = std::max(cc_dmax, o.cc_dmax); cc_smax = std::max(cc_smax, o.cc_smax);
    g_rawmax = std::max(g_rawmax, o.g_rawmax); g_midmax = std::max(g_midmax, o.g_midmax); slot_n += o.slot_n; slot_in += o.slot_in; slot_vp += o.slot_vp; slot_d += o.slot_d; dec += o.dec; nodes += o.nodes; kept += o.kept; sw += o.sw; fallback += o.fallback; exact += o.exact; ms += o.ms; ms_max = std::max(ms_max, o.ms_max); }
};

struct Work {
  std::vector<uint8_t> flag;
  std::vector<int32_t> qa, qb, dist, label, cd, cd2, room;
  std::vector<int32_t> bk[8];
  std::vector<uint8_t> narrow;
  std::vector<mdp::Frontier> out;
  std::vector<uint64_t> key2;
  Stat st;
  void fit(int n) {
    if ((int)flag.size() < n) { flag.resize(n); qa.resize(n); qb.resize(n); dist.resize(n); label.resize(n); cd.assign(n, -1); cd2.assign(n, -1); room.assign(n, -1); narrow.assign(n, 0); }
    if (out.size() < (size_t)mdp::kReachCap + 1) { out.resize(mdp::kReachCap + 1); key2.resize(mdp::kReachCap + 1); }
  }
};

// 후보 하나와 그 특징 (decide 의 In::cands 를 주면 채움 — 판단 모델·규칙 견줌용, 로봇 지도·지도 자세·로봇이 추적한 사람만 씀)
struct CandOut {
  int vp = -1, anc = -1;       // 목표 칸(덩어리마다 고른 시점) · 덩어리의 가장 가까운 프론티어 칸
  float len_m = 0;             // 로봇 → 시점 길 (8 이웃 모따기, m)
  float turn_rad = 0;          // 첫 걸음 방향과 지금 머리 방향 차 (rad)
  float t_s = 0;               // 가는 시간 = len / v_eff + turn / ω_eff (s, 실측 맞춤 값)
  float gain = 0;              // 시점에서 예상 새로 보이는 모름 넓이 / 원 넓이 (frontier.h cand_gain_ray, 360°)
  float gpt = 0;               // gain / (t_s + t0)
  float open_m = 0;            // 덩어리(열린 틈) 길이 = 프론티어 칸 수 × 칸 (m)
  float depth_m = 0;           // 틈 너머 트임: 시점에서 볼 방향 ±60° 광선 9 개가 막힘에 닿기까지 평균 길이 (모름은 지나감, 8 m 상한) — 벽장 ≈ 짧음, 방·복도 ≈ 김
  int room_left = 0;           // 같은 방(좁은 곳에서 나눈 로봇 지도)의 다른 후보 덩어리 수
  bool same_room = false;      // 로봇과 같은 방
  float opp_s = 0;             // 떠나는 값: 이 후보로 시작하는 가장 좋은 순회 − 가장 좋은 순회 (s)
  float clear_m = 0;           // 길 위 가장 좁은 틈: 길 칸에서 가장 가까운 막힘 칸까지 거리의 최솟값 (0.6 m 상한)
  float clutter = 0;           // 길 둘레 0.5 m 안 막힘 칸 비율 평균 (가구 밀도)
  int people = 0;              // 길에서 1 m 안의 추적된 사람 수 (In::people — 로봇 라이다 추적)
  float bearing = 0;           // 시점 방향, 로봇 머리 기준 (rad, 왼쪽 +)
  bool cur = false, fuel = false;   // 지금 목표 덩어리 · fuel 순회가 고른 것
};

// 결정 하나의 속 (재기만 — 결정에 영향 없음): 다시 돌리기 도구(sim/tools/low_explore_replay)가 결정마다 왜 그 목표·왜 목표 없음인지 적는 데
struct Diag {
  int why = 0;          // 0 목표 냄 · 1 설 수 있는 시작 칸 없음 · 2 번진 곳에 프론티어 덩어리 없음 · 3 덩어리는 있으나 자격(거리·본 칸) 칸 없음 · 4 모두 피하는 칸 · 5 시점 자격 되는 덩어리 없음 (kNone)
  int n_clu = 0;        // 번진 곳(로봇 시작 칸에서) 프론티어 덩어리 수 (크기 무관)
  int n_node = 0;       // 합침·나눔·피하기 뒤 순회 노드 수
  int cur = -1, pick = -1, best_first = -1;   // 지금 목표 노드 · 고른 노드 · 가장 좋은 순회의 첫 노드
  double best = 0, cur_cost = 0;              // 가장 좋은 순회 값 · 지금 목표로 시작하는 순회 값 (cur ≥ 0 일 때)
  std::vector<int> vp, anc, ncell;            // 노드마다 시점 칸 · 가장 가까운 칸 · 칸 수
  std::vector<double> t0;                     // 노드마다 로봇 → 시점 비용
  std::vector<int> order;                     // 가장 좋은 순회 차례 (노드 번호 0 …)
};

struct In {
  const sim::EnvCfg* c;
  const sim::World* W;
  const int16_t* L;          // 로봇 지도 (줄 = c->size)
  const uint32_t* trail;     // 지나간 칸·본 칸 판
  const uint8_t* cam;        // 깊이 카메라 층 (없으면 nullptr)
  const sim::EnvState* s;
  const int32_t* cand;       // 시뮬 후보 칸 K 개 (global 끔일 때 순회 대상)
  std::function<bool(int)> avoided;   // 피하는 칸 둘레인가 (combine_run Avoid)
  std::vector<CandOut>* cands = nullptr;      // 주면 후보 특징을 채움 (gpt 큰 순, cand_cap 개 — 지금 목표·fuel 고른 것은 늘 넣음)
  int cand_cap = 16;
  int cand_big = 0;            // 상한 안에서 이 수만큼은 틈(open_m) 큰 순으로 채움 — 멀리 있는 큰 미탐사 구역이 gpt 차례로 밀려 빠지지 않게 (잠정 4)
  const std::vector<float>* people = nullptr;   // 추적된 사람 자리 (x, y 세계 m 쌍)
  Diag* diag = nullptr;                          // 주면 결정 속을 채움 (재기만)
};

namespace detail {
inline double wrap(double a) { return std::fabs(std::atan2(std::sin(a), std::cos(a))); }

// 8 이웃 모따기 거리 (직선 5 ·대각 7, 칸 단위 × 5) — 갈 수 있는 칸 = pass[idx]. 모서리로 안 빠짐(대각은 두 직선 이웃도 갈 수 있어야).
// 양의 정수 가중 다익스트라를 통 큐(Dial)로 — 통 8 개 고리(가장 큰 가중 7 < 8), 통마다 배열(같은 칸이 낡은 항목으로 여러 번 들 수 있음).
// 결과 d[idx] (못 닿음 −1). 지울 칸 목록 = cells (번진 칸 전체)
template <class Pass>
void chamfer(int src, int gw, int gh, const Pass& pass, const std::vector<int32_t>& cells, int ncells, std::vector<int32_t>& d, std::vector<int32_t> (&bk)[8]) {
  for (int q = 0; q < ncells; ++q) d[cells[q]] = -1;
  if (src < 0) return;
  for (auto& v : bk) v.clear();
  d[src] = 0;
  bk[0].push_back(src);
  long pending = 1;
  for (int cur = 0; pending > 0; ++cur) {
    std::vector<int32_t>& B = bk[cur & 7];
    for (size_t h = 0; h < B.size(); ++h) {   // 처리 중 같은 통에 안 들어옴 (가중 ≥ 5)
      const int idx = B[h];
      --pending;
      if (d[idx] != cur) continue;   // 낡은 항목
      const int x = idx % gw, y = idx / gw;
      for (int dy = -1; dy <= 1; ++dy)
        for (int dx = -1; dx <= 1; ++dx) {
          if (!dx && !dy) continue;
          const int u = x + dx, v = y + dy;
          if (u < 0 || v < 0 || u >= gw || v >= gh) continue;
          const int j = v * gw + u;
          if (!pass(j)) continue;
          if (dx && dy && (!pass(y * gw + u) || !pass(v * gw + x))) continue;
          const int nd = cur + (dx && dy ? 7 : 5);
          if (d[j] >= 0 && d[j] <= nd) continue;
          d[j] = nd;
          bk[nd & 7].push_back(j);
          ++pending;
        }
    }
    B.clear();
  }
}

// 열린 순회 (식 5: 돌아오는 값 0). C[i][j] = 노드 i → j (0 = 로봇). 노드 1 … m. first ≥ 1 이면 첫 노드를 고정. 차례(노드 번호)와 값을 냄
inline double tour(const std::vector<double>& C, int m, int first, std::vector<int>& order, bool* exact) {
  const int S = m + 1;
  order.clear();
  if (m == 0) return 0;
  constexpr int kExact = 12;
  if (m <= kExact) {
    *exact = true;
    const int M = 1 << m;
    std::vector<double> best((size_t)M * m, 1e18);
    std::vector<int8_t> prv((size_t)M * m, -1);
    for (int i = 0; i < m; ++i)
      if (first < 1 || first == i + 1) best[(size_t)(1 << i) * m + i] = C[0 * S + i + 1];
    for (int s = 1; s < M; ++s)
      for (int l = 0; l < m; ++l) {
        const double b = best[(size_t)s * m + l];
        if (!(s >> l & 1) || b >= 1e17) continue;
        for (int j = 0; j < m; ++j) {
          if (s >> j & 1) continue;
          const size_t t = (size_t)(s | 1 << j) * m + j;
          const double nb = b + C[(l + 1) * S + j + 1];
          if (nb < best[t]) { best[t] = nb; prv[t] = (int8_t)l; }
        }
      }
    int bl = -1;
    double bv = 1e18;
    for (int l = 0; l < m; ++l) if (best[(size_t)(M - 1) * m + l] < bv) { bv = best[(size_t)(M - 1) * m + l]; bl = l; }
    if (bl < 0) return 1e18;
    int s = M - 1, l = bl;
    while (l >= 0) { order.push_back(l + 1); const int p = prv[(size_t)s * m + l]; s &= ~(1 << l); l = p; }
    std::reverse(order.begin(), order.end());
    return bv;
  }
  *exact = false;
  // 최근접 이웃 (첫 노드 고정이면 그것부터)
  std::vector<uint8_t> used(S, 0);
  int at = 0;
  if (first >= 1) { order.push_back(first); used[first] = 1; at = first; }
  while ((int)order.size() < m) {
    int bj = -1;
    double bd = 1e18;
    for (int j = 1; j < S; ++j) if (!used[j] && C[at * S + j] < bd) { bd = C[at * S + j]; bj = j; }
    used[bj] = 1;
    order.push_back(bj);
    at = bj;
  }
  const auto cost = [&](const std::vector<int>& o) {
    double v = 0;
    int p = 0;
    for (int x : o) { v += C[p * S + x]; p = x; }
    return v;
  };
  double cv = cost(order);
  const int lo = first >= 1 ? 1 : 0;   // 고정된 첫 자리는 안 움직임
  std::vector<int> t;
  for (int pass = 0; pass < 20; ++pass) {
    bool ch = false;
    for (int a = lo; a < m - 1; ++a)   // 2-opt: 구간 뒤집기 (비대칭이라 값을 다시 셈)
      for (int b = a + 1; b < m; ++b) {
        t = order;
        std::reverse(t.begin() + a, t.begin() + b + 1);
        const double v = cost(t);
        if (v + 1e-9 < cv) { order.swap(t); cv = v; ch = true; }
      }
    for (int len = 1; len <= 3; ++len)   // Or-opt: 길이 1~3 구간 옮기기
      for (int a = lo; a + len <= m; ++a)
        for (int p = lo; p <= m - len; ++p) {
          if (p == a) continue;
          t = order;
          std::vector<int> seg(t.begin() + a, t.begin() + a + len);
          t.erase(t.begin() + a, t.begin() + a + len);
          t.insert(t.begin() + p, seg.begin(), seg.end());
          const double v = cost(t);
          if (v + 1e-9 < cv) { order.swap(t); cv = v; ch = true; }
        }
    if (!ch) break;
  }
  return cv;
}
}  // namespace detail

// 노드 (순회의 한 자리 = 프론티어 덩어리 하나)
struct Node {
  std::vector<int> cells;   // 자격 있는 프론티어 칸
  int anc = -1;             // 로봇에서 BFS 거리가 가장 짧은 칸
  int anc_d = 0;
  double cx = 0, cy = 0;    // 칸 가운데 (칸 단위)
  bool dead = false;        // 자격 되는 시점 없음 (vp_min · vp_clear 켬일 때 — FUEL dormant)
  int vp = -1;              // 시점 칸
  double yaw = 0;           // 시점에서 볼 방향
};

// 결정 하나: 보낼 목표 칸 (없으면 −1 → 부르는 쪽이 slot0 로 · kNone → 목표 없음)
inline int decide(const In& in, const Opt& o, Mem& mem, Work& w) {
  const auto t_start = std::chrono::steady_clock::now();
  const sim::EnvCfg& c = *in.c;
  const sim::World& W = *in.W;
  const int gw = W.gw, gh = W.gh, n = gw * gh;
  w.fit(n);
  Stat& st = w.st;
  st.dec += 1;
  const mdp::MapView mv{in.L, c.size, gw, gh};
  const sim::EnvState& s = *in.s;
  const double rxf = (s.x + s.ex - W.ox) * c.inv_res, ryf = (s.y + s.ey - W.oy) * c.inv_res, th = s.th + s.eth;
  const int rx = (int)std::floor(rxf), ry = (int)std::floor(ryf);
  mdp::SearchBuf sb{w.flag.data(), w.qa.data(), w.qb.data(), w.dist.data(), w.label.data()};
  const int ws = (c.size + 31) / 32, nf = c.size * ws;
  sb.vis = in.trail;
  sb.look = in.trail + nf;
  sb.tws = ws;
  sb.close_min = c.close_min;
  sb.rbc = mdp::robot_cells(c.close_radius, c.res);
  sb.cam = in.cam;
  const mdp::Robot rb = mdp::obsgen::reach_robot(c);
  const mdp::SearchResult r = mdp::frontier_search(mv, rx, ry, rb, 1, sb, w.out.data(), (int)w.out.size());
  const int n_out = std::min(r.n_found, (int)w.out.size());
  const auto done = [&](int v) {
    const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_start).count();
    st.ms += ms;
    st.ms_max = std::max(st.ms_max, ms);
    return v;
  };
  Diag* dg = in.diag;
  if (dg) *dg = Diag();
  const bool strict = o.vp_min > 0.f || o.vp_clear > 0.f;   // 시점 자격 켬 (아래 "시점")
  if (!r.start_free) { if (dg) dg->why = 1; mem = Mem(); st.fallback += 1; return done(-1); }
  const int nr0 = r.n_reach0;   // 로봇 시작 칸에서 번진 칸 = 갈 수 있는 칸 (qa[0 … nr0))
  const auto pass = [&](int idx) { return (w.flag[idx] & 1) && w.dist[idx] >= 0; };
  const auto eligible = [&](int idx) {   // cand_keys 와 같은 자격
    if (!(w.flag[idx] & 2) || w.dist[idx] < mdp::hier::kCandMinDist) return false;
    const int l = w.label[idx];
    if (l < 0 || l >= n_out || w.out[l].n < c.close_min || w.out[l].n < mdp::hier::kCandMinCells) return false;
    return !mdp::looked_cell(mv, sb.look, sb.vis, ws, idx % gw, idx / gw);
  };
  // ---- 노드 만들기
  std::vector<Node> nodes;
  int n_raw = 0;   // 자격 칸이 있는 덩어리 수 (전역) · 쓸 수 있는 후보 수 (후보 K 개)
  if (o.global) {
    std::vector<std::vector<int>> by(n_out);
    for (int q = 0; q < nr0; ++q) { const int idx = w.qa[q]; if (eligible(idx)) by[w.label[idx]].push_back(idx); }
    for (int l = 0; l < n_out; ++l) if (!by[l].empty()) { Node nd; nd.cells.swap(by[l]); nodes.push_back(std::move(nd)); }
    n_raw = (int)nodes.size();
    st.g_raw += (long)nodes.size(); st.g_rawmax = std::max(st.g_rawmax, (long)nodes.size()); st.g_dec += 1;
  } else {
    // 시뮬 후보 K 개: 후보 칸의 덩어리(라벨) 칸들. 후보 v2 의 합친 덩어리는 칸만 첫 덩어리 것 (가운데·시점 뽑기에만 씀)
    for (int k = 0; k < mdp::hier::K; ++k) {
      const int cell = in.cand[k];
      if (cell < 0 || cell >= n || in.avoided(cell) || !pass(cell)) continue;
      Node nd;
      if ((w.flag[cell] & 2) && w.label[cell] >= 0) {
        const int l = w.label[cell];
        for (int q = 0; q < nr0; ++q) { const int idx = w.qa[q]; if ((w.flag[idx] & 2) && w.label[idx] == l && eligible(idx)) nd.cells.push_back(idx); }
      }
      if (nd.cells.empty()) nd.cells.push_back(cell);
      nd.anc = cell;   // 후보 칸 그대로 (cand_tsp 와 같은 목표)
      nodes.push_back(std::move(nd));
      n_raw += 1;
    }
  }
  const auto fill = [&](Node& nd, bool keep_anc) {
    double sx = 0, sy = 0;
    int bd = 1 << 30, ba = -1;
    for (int idx : nd.cells) {
      sx += idx % gw + 0.5; sy += idx / gw + 0.5;
      const int d = w.dist[idx] >= 0 ? w.dist[idx] : (1 << 29);
      if (d < bd || (d == bd && idx < ba)) { bd = d; ba = idx; }
    }
    nd.cx = sx / nd.cells.size(); nd.cy = sy / nd.cells.size();
    if (!keep_anc) nd.anc = ba;
    nd.anc_d = w.dist[nd.anc] >= 0 ? w.dist[nd.anc] : bd;
  };
  for (Node& nd : nodes) fill(nd, !o.global ? true : false);
  if (o.global && !nodes.empty()) {
    // 합침: 가장 가까운 칸끼리 merge_m 안 · BFS 거리 차 2 배 안 (후보 v2 cand2_near 와 같은 꼴) — 합집합 찾기
    const int Rm = (int)(o.merge_m * c.inv_res + 0.5f);
    std::vector<int> par(nodes.size());
    for (size_t i = 0; i < par.size(); ++i) par[i] = (int)i;
    const std::function<int(int)> root = [&](int x) { return par[x] == x ? x : par[x] = root(par[x]); };
    for (size_t i = 0; i < nodes.size(); ++i)
      for (size_t j = i + 1; j < nodes.size(); ++j) {
        const int dx = nodes[i].anc % gw - nodes[j].anc % gw, dy = nodes[i].anc / gw - nodes[j].anc / gw, dd = std::abs(nodes[i].anc_d - nodes[j].anc_d);
        if (dx * dx + dy * dy <= Rm * Rm && dd <= 2 * Rm) par[root((int)i)] = root((int)j);
      }
    std::vector<Node> mg;
    std::vector<int> at(nodes.size(), -1);
    for (size_t i = 0; i < nodes.size(); ++i) {
      const int rt = root((int)i);
      if (at[rt] < 0) { at[rt] = (int)mg.size(); mg.emplace_back(); }
      Node& t = mg[at[rt]];
      t.cells.insert(t.cells.end(), nodes[i].cells.begin(), nodes[i].cells.end());
    }
    // 나눔: 첫 주성분 표준편차 > split_m 이면 가운데에서 그 축으로 둘로 (되풀이)
    const double lim = (double)o.split_m * c.inv_res;
    std::vector<Node> fin;
    while (!mg.empty()) {
      Node nd = std::move(mg.back());
      mg.pop_back();
      double mx = 0, my = 0;
      for (int idx : nd.cells) { mx += idx % gw; my += idx / gw; }
      mx /= nd.cells.size(); my /= nd.cells.size();
      double sxx = 0, syy = 0, sxy = 0;
      for (int idx : nd.cells) { const double dx = idx % gw - mx, dy = idx / gw - my; sxx += dx * dx; syy += dy * dy; sxy += dx * dy; }
      sxx /= nd.cells.size(); syy /= nd.cells.size(); sxy /= nd.cells.size();
      const double tr = sxx + syy, det = sxx * syy - sxy * sxy, l1 = 0.5 * tr + std::sqrt(std::max(0.0, 0.25 * tr * tr - det));
      if (nd.cells.size() >= 4 && l1 > lim * lim) {
        double ex = sxy, ey = l1 - sxx;   // 첫 고유벡터
        if (std::fabs(ex) + std::fabs(ey) < 1e-9) { ex = sxx >= syy ? 1 : 0; ey = sxx >= syy ? 0 : 1; }
        Node a, b;
        for (int idx : nd.cells) ((idx % gw - mx) * ex + (idx / gw - my) * ey < 0 ? a : b).cells.push_back(idx);
        if (!a.cells.empty() && !b.cells.empty()) { mg.push_back(std::move(a)); mg.push_back(std::move(b)); continue; }
      }
      fin.push_back(std::move(nd));
    }
    nodes.swap(fin);
    for (Node& nd : nodes) fill(nd, false);
    // 피하는 칸 빼기
    nodes.erase(std::remove_if(nodes.begin(), nodes.end(), [&](const Node& nd) { return in.avoided(nd.anc); }), nodes.end());
    // 차례를 정함 (결정적): 가장 가까운 칸 (거리, 칸 번호)
    std::sort(nodes.begin(), nodes.end(), [](const Node& a, const Node& b) { return a.anc_d != b.anc_d ? a.anc_d < b.anc_d : a.anc < b.anc; });
    st.g_mid += (long)nodes.size(); st.g_midmax = std::max(st.g_midmax, (long)nodes.size());
    if ((int)nodes.size() > o.max_nodes) {   // 상한: 가까운 순 2/3 + 나머지에서 큰 순
      const int near = std::max(1, o.max_nodes * 2 / 3);
      std::stable_sort(nodes.begin() + near, nodes.end(), [](const Node& a, const Node& b) { return a.cells.size() > b.cells.size(); });
      if (mem.anc >= 0) {   // 지금 목표 덩어리는 상한 밖이어도 남김 (붙들기가 뜻을 잃지 않게)
        const int Rm = (int)(o.merge_m * c.inv_res + 0.5f), ax = mem.anc % gw, ay = mem.anc / gw;
        for (size_t k = o.max_nodes; k < nodes.size(); ++k) {
          bool hit = false;
          for (int idx : nodes[k].cells) { const int dx = idx % gw - ax, dy = idx / gw - ay; if (dx * dx + dy * dy <= Rm * Rm) { hit = true; break; } }
          if (hit) { std::swap(nodes[o.max_nodes - 1], nodes[k]); break; }
        }
      }
      st.cap_dec += 1;
      for (size_t k = o.max_nodes; k < nodes.size(); ++k) {   // 빠지는 덩어리 (셈만)
        const double dm = nodes[k].anc_d * (double)c.res, sm = nodes[k].cells.size() * (double)c.res;
        st.cap_drop += 1; st.cap_d += dm; st.cap_s += sm; st.cap_dmax = std::max(st.cap_dmax, dm); st.cap_smax = std::max(st.cap_smax, sm);
      }
      nodes.resize(o.max_nodes);
    }
  }
  int m = (int)nodes.size();
  st.nodes += m;
  if (dg) {
    std::vector<uint8_t> seen(n_out, 0);
    for (int q = 0; q < nr0; ++q) { const int idx = w.qa[q]; if ((w.flag[idx] & 2) && w.label[idx] >= 0 && w.label[idx] < n_out && !seen[w.label[idx]]) { seen[w.label[idx]] = 1; dg->n_clu += 1; } }
    dg->n_node = m;
  }
  if (m == 0) {
    if (dg) dg->why = dg->n_clu == 0 ? 2 : (o.global && n_raw == 0 ? 3 : 4);
    mem = Mem(); st.fallback += 1; return done(-1);
  }
  // ---- 로봇에서 거리 (time 이면 모따기 거리, 아니면 BFS 칸 거리)
  std::vector<int32_t> cells0(w.qa.begin(), w.qa.begin() + nr0);
  if (o.time || o.vp || in.cands) detail::chamfer(r.start, gw, gh, pass, cells0, nr0, w.cd, w.bk);
  const double res = c.res;
  const auto len0 = [&](int idx) -> double {   // 로봇 → 칸 길이 (m), 못 닿음 1e6
    if (o.time || o.vp) return w.cd[idx] >= 0 ? w.cd[idx] * 0.2 * res : 1e6;
    return pass(idx) ? w.dist[idx] * res : 1e6;
  };
  // 첫 걸음 방향: 로봇 쪽으로 모따기 거리를 거슬러 로봇에서 0.5 m 떨어진 칸
  const auto first_dir = [&](int idx) {
    int at = idx;
    const int stop = (int)(0.5 / res) * 5;
    for (int g = 0; g < 4 * n && w.cd[at] > stop; ++g) {
      const int x = at % gw, y = at / gw;
      int bj = -1, bv = w.cd[at];
      for (int dy = -1; dy <= 1; ++dy)
        for (int dx = -1; dx <= 1; ++dx) {
          const int u = x + dx, v = y + dy;
          if ((!dx && !dy) || u < 0 || v < 0 || u >= gw || v >= gh) continue;
          const int j = v * gw + u;
          if (pass(j) && w.cd[j] >= 0 && w.cd[j] < bv) { bv = w.cd[j]; bj = j; }
        }
      if (bj < 0) break;
      at = bj;
    }
    return std::atan2(at / gw + 0.5 - ryf, at % gw + 0.5 - rxf);
  };
  // 볼 방향: 덩어리 가운데 둘레 1 m 안 모름 칸의 가운데 쪽 (없으면 덩어리 가운데 쪽)
  const int R1 = (int)(1.0 / res);
  const auto look_yaw = [&](const Node& nd, int vp) {
    double ux = 0, uy = 0;
    long k = 0;
    const int cx = (int)nd.cx, cy = (int)nd.cy;
    for (int y = std::max(0, cy - R1); y <= std::min(gh - 1, cy + R1); y += 2)
      for (int x = std::max(0, cx - R1); x <= std::min(gw - 1, cx + R1); x += 2)
        if (mdp::cls_at(mv, x, y) == mdp::kUnk) { ux += x + 0.5; uy += y + 0.5; ++k; }
    const double tx = k ? ux / k : nd.cx, ty = k ? uy / k : nd.cy;
    const double dx = tx - (vp % gw + 0.5), dy = ty - (vp / gw + 0.5);
    return std::fabs(dx) + std::fabs(dy) < 1e-6 ? th : std::atan2(dy, dx);
  };
  const auto t_robot = [&](int vp, double yaw) {   // 로봇 → 시점 시간 (time 이면 식 1 + 첫 회전·식 3, 아니면 길이 m)
    const double L = len0(vp);
    if (!o.time) return L;
    double t = std::max(L / o.v_eff, detail::wrap(yaw - th) / o.w_eff);
    if (o.turn_w > 0) t += o.turn_w * detail::wrap(first_dir(vp) - th) / o.w_eff;
    if (o.wc > 0 && std::fabs(s.v) > 0.1f) {   // 식 4: 지금 속도 방향 (오도메트리 앞 속도 부호로 머리 방향 또는 반대)
      const double vx = std::cos(th) * (s.v > 0 ? 1 : -1), vy = std::sin(th) * (s.v > 0 ? 1 : -1);
      const double dx = vp % gw + 0.5 - rxf, dy = vp / gw + 0.5 - ryf, dl = std::hypot(dx, dy);
      if (dl > 1e-6) t += o.wc * std::acos(std::max(-1.0, std::min(1.0, (dx * vx + dy * vy) / dl)));
    }
    return t;
  };
  // ---- 시점
  const int Rg = mdp::cand_gain_cells(c.map_max_range, c.res);
  const auto gain = [&](int vp) { long g = 0; for (int a = 0; a < mdp::hier::kGainRays; ++a) g += mdp::cand_gain_ray(mv, vp % gw, vp / gw, a, Rg); return (double)g / mdp::cand_gain_norm(Rg); };
  const auto los = [&](int a, int b) {   // 막힘 칸에 안 가리고 보임 (끝 칸 빼고)
    const int x0 = a % gw, y0 = a / gw, x1 = b % gw, y1 = b / gw, N = std::max(std::abs(x1 - x0), std::abs(y1 - y0));
    for (int t = 1; t < N; ++t) {
      const int x = x0 + (int)std::lround((double)(x1 - x0) * t / N), y = y0 + (int)std::lround((double)(y1 - y0) * t / N);
      if (mdp::cls_at(mv, x, y) == mdp::kOcc) return false;
    }
    return true;
  };
  const int Rc = (int)std::ceil(o.vp_clear * c.inv_res - 1e-4f);
  const auto near_unknown = [&](int cell) {   // FUEL isNearUnknown: 둘레 vp_clear 안 (원) 에 모름 칸
    const int x = cell % gw, y = cell / gw;
    for (int dy = -Rc; dy <= Rc; ++dy)
      for (int dx = -Rc; dx <= Rc; ++dx) {
        if ((dx * dx + dy * dy) * res * res > (double)o.vp_clear * o.vp_clear) continue;
        const int u = x + dx, v = y + dy;
        if (u >= 0 && v >= 0 && u < gw && v < gh && mdp::cls_at(mv, u, v) == mdp::kUnk) return true;
      }
    return false;
  };
  for (Node& nd : nodes) {
    nd.vp = nd.anc;
    if (o.vp) {
      double bs = -1;
      int bv = -1;
      bool clear_on = o.vp_clear > 0.f;
      const auto test = [&](int cell) {
        if (cell < 0 || cell >= n || !pass(cell) || in.avoided(cell) || w.dist[cell] < mdp::hier::kCandMinDist) return;
        const double dr = std::hypot(cell % gw + 0.5 - rxf, cell / gw + 0.5 - ryf) * res;
        if (strict) {
          if (dr < std::max((double)o.vp_min, (double)c.low_pass_r)) return;   // 통과 반경 안은 늘 뺌 (가자마자 통과)
          if (clear_on && near_unknown(cell)) return;
        } else if (cell != nd.anc && dr <= c.low_pass_r) {
          return;   // 이미 통과 반경 안 = 가도 그 자리 (옛 규칙: 가장 가까운 칸은 예외)
        }
        if (cell != nd.anc && !los(cell, nd.anc)) return;
        const double g = gain(cell);
        if (g <= 0) return;
        const double t = (o.time ? t_robot(cell, look_yaw(nd, cell)) : len0(cell) / o.v_eff) + o.t0;
        if (g / t > bs) { bs = g / t; bv = cell; }
      };
      const auto sample = [&]() {
        test(nd.anc);
        static const double kR[3] = {0.6, 1.2, 1.8};
        for (int ri = 0; ri < 3; ++ri)
          for (int a = 0; a < 12; ++a) {
            const double rr = kR[ri], ang = 2 * M_PI * (a + 0.5 * ri) / 12;
            const int x = (int)std::floor(nd.cx + rr / res * std::cos(ang)), y = (int)std::floor(nd.cy + rr / res * std::sin(ang));
            if (x >= 0 && y >= 0 && x < gw && y < gh) test(y * gw + x);
          }
      };
      sample();
      if (bv < 0 && clear_on) { clear_on = false; sample(); }   // 간격 맞는 시점이 없으면 거리 문턱만으로
      if (bv >= 0) nd.vp = bv;
      else if (strict) nd.dead = true;
    } else if (strict && std::hypot(nd.anc % gw + 0.5 - rxf, nd.anc / gw + 0.5 - ryf) * res < std::max((double)o.vp_min, (double)c.low_pass_r)) {
      nd.dead = true;
    }
    nd.yaw = look_yaw(nd, nd.vp);
  }
  // ---- 지금 목표 덩어리 (기억한 칸에서 merge_m 안 노드)
  int cur = -1;
  if (mem.anc >= 0 && mem.vp >= 0 && pass(mem.vp) && !in.avoided(mem.vp)) {
    const int Rm = (int)(o.merge_m * c.inv_res + 0.5f);
    long bd = (long)Rm * Rm + 1;
    const int ax = mem.anc % gw, ay = mem.anc / gw;
    for (int k = 0; k < m; ++k)
      for (int idx : nodes[k].cells) {
        const long dx = idx % gw - ax, dy = idx / gw - ay;
        if (dx * dx + dy * dy < bd) { bd = dx * dx + dy * dy; cur = k; }
      }
    if (cur >= 0) { nodes[cur].vp = mem.vp; nodes[cur].yaw = look_yaw(nodes[cur], mem.vp); nodes[cur].dead = false; }   // 같은 덩어리면 같은 목표 칸 (아래층 경로를 이어 감)
  }
  if (strict) {   // 시점 없는 덩어리 빼기 (지금 목표 덩어리는 그 목표 칸 그대로 남김)
    std::vector<Node> live;
    int cur2 = -1;
    for (int k = 0; k < m; ++k) {
      if (nodes[k].dead) continue;
      if (k == cur) cur2 = (int)live.size();
      live.push_back(std::move(nodes[k]));
    }
    nodes.swap(live);
    cur = cur2;
    m = (int)nodes.size();
    if (dg) dg->n_node = m;
    if (m == 0) { if (dg) dg->why = 5; mem = Mem(); st.fallback += 1; return done(kNone); }
  }
  // ---- 방 마무리 (room): 좁은 칸(벽까지 거리 < door_m)을 빼고 갈 수 있는 칸을 4 이웃으로 나눔 → 로봇 방
  std::vector<uint8_t> inroom(m, 0);
  std::vector<int> rid(m, -1);
  int rroom = -1;
  if (o.room || in.cands) {
    const int Rd = (int)std::ceil(o.door_m * c.inv_res);
    const mdp::Robot rd = mdp::robot_cells(o.door_m, c.res);
    (void)Rd;
    for (int q = 0; q < nr0; ++q) { const int idx = w.qa[q]; w.narrow[idx] = mdp::standable(mv, idx % gw, idx / gw, rd) ? 0 : 1; w.room[idx] = -1; }
    int nroom = 0;
    for (int q = 0; q < nr0; ++q) {
      const int s0 = w.qa[q];
      if (w.narrow[s0] || w.room[s0] >= 0) continue;
      int h = 0, t = 0;
      w.qb[t++] = s0;
      w.room[s0] = nroom;
      while (h < t) {
        const int idx = w.qb[h++], x = idx % gw, y = idx / gw;
        for (int k = 0; k < 4; ++k) {
          int dx, dy;
          mdp::nb4(k, &dx, &dy);
          const int u = x + dx, v = y + dy;
          if (u < 0 || v < 0 || u >= gw || v >= gh) continue;
          const int j = v * gw + u;
          if (!pass(j) || w.narrow[j] || w.room[j] >= 0) continue;
          w.room[j] = nroom;
          w.qb[t++] = j;
        }
      }
      ++nroom;
    }
    // 로봇 방: 로봇 시작 칸에서 가장 가까운(BFS 차례) 방 칸
    for (int q = 0; q < nr0 && rroom < 0; ++q) if (w.room[w.qa[q]] >= 0) rroom = w.room[w.qa[q]];
    // 좁은 칸의 방 = 가장 가까운 방 칸 (4 이웃 번짐) — 문틀 위 시점·프론티어도 방에 붙임
    const auto room_of = [&](int idx) {
      if (w.room[idx] >= 0) return w.room[idx];
      const int x = idx % gw, y = idx / gw;
      for (int rr = 1; rr <= Rd + 2; ++rr)
        for (int dy = -rr; dy <= rr; ++dy)
          for (int dx = -rr; dx <= rr; ++dx) {
            const int u = x + dx, v = y + dy;
            if (u < 0 || v < 0 || u >= gw || v >= gh) continue;
            if (pass(v * gw + u) && w.room[v * gw + u] >= 0) return w.room[v * gw + u];
          }
      return -1;
    };
    for (int k = 0; k < m; ++k) { rid[k] = room_of(nodes[k].vp); inroom[k] = o.room && rroom >= 0 && rid[k] == rroom; }
  }
  // ---- 비용 행렬 (0 = 로봇, 1 … m = 노드). 식 5: 노드 → 로봇 0  (first 면 순회를 안 짬 — 행렬도 로봇 줄만)
  const int S = m + 1;
  std::vector<double> C((size_t)S * S, 0.0);
  const double pen = o.time ? o.room_pen : o.room_pen * o.v_eff;   // 시간 끔이면 비용이 길이(m) — 같은 뜻의 길이로
  bool any_in = false;
  for (int k = 0; k < m; ++k) any_in = any_in || inroom[k];
  for (int k = 0; k < m; ++k) C[k + 1] = t_robot(nodes[k].vp, nodes[k].yaw) + (any_in && !inroom[k] ? pen : 0.0);
  if (m > 1 && !o.first) {
    for (int a = 0; a < m; ++a) {
      if (o.time || o.vp) detail::chamfer(nodes[a].vp, gw, gh, pass, cells0, nr0, w.cd2, w.bk);
      else {   // BFS 칸 거리 (cand_tsp 와 같음)
        for (int q = 0; q < nr0; ++q) w.cd2[w.qa[q]] = -1;
        int h = 0, t = 0;
        w.qb[t++] = nodes[a].vp;
        w.cd2[nodes[a].vp] = 0;
        while (h < t) {
          const int idx = w.qb[h++], x = idx % gw, y = idx / gw;
          for (int k = 0; k < 4; ++k) {
            int dx, dy;
            mdp::nb4(k, &dx, &dy);
            const int u = x + dx, v = y + dy;
            if (u < 0 || v < 0 || u >= gw || v >= gh) continue;
            const int j = v * gw + u;
            if (!pass(j) || w.cd2[j] >= 0) continue;
            w.cd2[j] = w.cd2[idx] + 1;
            w.qb[t++] = j;
          }
        }
      }
      for (int b = 0; b < m; ++b) {
        if (a == b) continue;
        const int d = w.cd2[nodes[b].vp];
        double L = d >= 0 ? ((o.time || o.vp) ? d * 0.2 * res : d * res) : 1e6;
        double t = o.time ? std::max(L / o.v_eff, detail::wrap(nodes[a].yaw - nodes[b].yaw) / o.w_eff) : L;
        C[(size_t)(a + 1) * S + b + 1] = t + (any_in && ((inroom[a] && !inroom[b]) || (!inroom[a] && inroom[b])) ? pen : 0.0);
      }
    }
  }
  std::vector<int> order;
  bool exact = false;
  double best = 0;
  if (o.first) order.push_back(1);   // 덩어리 차례의 첫 것 (지금 목표가 첫 것이면 위에서 목표 칸 그대로)
  else best = detail::tour(C, m, -1, order, &exact);
  st.exact += exact;
  int pick = order.empty() ? 0 : order[0] - 1;
  if (cur >= 0 && pick != cur && !o.first) {
    std::vector<int> o2;
    const double cc = o.hyst > 0 ? detail::tour(C, m, cur + 1, o2, &exact) : 1e18;
    if (cc <= (1.0 + o.hyst) * best) pick = cur;
  }
  if (dg) {
    dg->cur = cur; dg->best_first = order.empty() ? -1 : order[0] - 1; dg->best = best; dg->pick = pick;
    if (cur >= 0 && !o.first) { std::vector<int> o4; bool ex4 = false; dg->cur_cost = m > 1 ? detail::tour(C, m, cur + 1, o4, &ex4) : best; }
    for (int k = 0; k < m; ++k) { dg->vp.push_back(nodes[k].vp); dg->anc.push_back(nodes[k].anc); dg->ncell.push_back((int)nodes[k].cells.size()); dg->t0.push_back(C[k + 1]); }
    for (int x : order) dg->order.push_back(x - 1);
  }
  if (cur >= 0 && pick == cur) st.kept += 1;
  if (cur >= 0 && pick != cur) st.sw += 1;
  {   // 후보 칸으로 나타낼 수 있나 (재기만): 고른 덩어리 칸의 라벨을 가진 후보, 없으면 덩어리 칸에서 merge_m 안 후보
    int ns = 0, hit = -1;
    double bd = 1e18;
    const int Rm = (int)(o.merge_m * c.inv_res + 0.5f);
    for (int k = 0; k < mdp::hier::K; ++k) {
      const int cell = in.cand[k];
      if (cell < 0 || cell >= n) continue;
      ++ns;
      for (int idx : nodes[pick].cells) {
        const bool same = (w.flag[cell] & 2) && (w.flag[idx] & 2) && w.label[cell] == w.label[idx];
        const int dx = idx % gw - cell % gw, dy = idx / gw - cell / gw;
        if (same || dx * dx + dy * dy <= Rm * Rm) {
          const double d = std::hypot(nodes[pick].vp % gw - cell % gw, nodes[pick].vp / gw - cell / gw) * res;
          if (d < bd) { bd = d; hit = k; }
          break;
        }
      }
    }
    if (ns > 0) { st.slot_n += 1; if (hit >= 0) { st.slot_in += 1; st.slot_d += bd; st.slot_vp += bd <= 0.5; } }
  }
  if (in.cands && !o.first) {   // ---- 후보 특징 (판단 모델·규칙 견줌용)
    std::vector<CandOut> all(m);
    const int R6 = (int)std::lround(0.6 / res), R5 = (int)std::lround(0.5 / res);
    for (int k = 0; k < m; ++k) {
      CandOut& q = all[k];
      const Node& nd = nodes[k];
      q.vp = nd.vp; q.anc = nd.anc; q.cur = k == cur; q.fuel = k == pick;
      q.len_m = w.cd[nd.vp] >= 0 ? (float)(w.cd[nd.vp] * 0.2 * res) : 1e6f;
      q.turn_rad = (float)detail::wrap(first_dir(nd.vp) - th);
      q.t_s = q.len_m / o.v_eff + q.turn_rad / o.w_eff;
      q.gain = (float)gain(nd.vp);
      q.gpt = q.gain / (q.t_s + o.t0);
      q.open_m = (float)(nd.cells.size() * res);
      q.bearing = (float)std::atan2(std::sin(std::atan2(nd.vp / gw + 0.5 - ryf, nd.vp % gw + 0.5 - rxf) - th), std::cos(std::atan2(nd.vp / gw + 0.5 - ryf, nd.vp % gw + 0.5 - rxf) - th));
      double dsum = 0;
      for (int a = -4; a <= 4; ++a) {   // 틈 너머 트임
        const double ang = nd.yaw + a * (M_PI / 12), ca = std::cos(ang), sa = std::sin(ang);
        int t = 1;
        const int T = (int)(8.0 / res);
        for (; t <= T; ++t) {
          const int x = nd.vp % gw + (int)std::lround(t * ca), y = nd.vp / gw + (int)std::lround(t * sa);
          if (x < 0 || y < 0 || x >= gw || y >= gh || mdp::cls_at(mv, x, y) == mdp::kOcc) break;
        }
        dsum += (t - 1) * res;
      }
      q.depth_m = (float)(dsum / 9);
      q.same_room = rroom >= 0 && rid[k] == rroom;
      for (int j = 0; j < m; ++j) q.room_left += j != k && rid[j] >= 0 && rid[j] == rid[k];
      // 길: 시점에서 로봇 쪽으로 모따기 거리를 거슬러 감 (2 칸마다 표본)
      double cmin = 0.6, csum = 0;
      int ns = 0, np = 0;
      std::vector<uint8_t> near_p(in.people ? in.people->size() / 2 : 0, 0);
      int at = nd.vp;
      for (int g = 0; g < 4 * n && at >= 0 && w.cd[at] > 0; ++g) {
        if (g % 2 == 0) {
          const int x = at % gw, y = at / gw;
          int occ = 0, tot = 0;
          for (int dy = -R6; dy <= R6; ++dy)
            for (int dx = -R6; dx <= R6; ++dx) {
              const int u = x + dx, v = y + dy;
              if (u < 0 || v < 0 || u >= gw || v >= gh) continue;
              const bool oc = mdp::cls_at(mv, u, v) == mdp::kOcc;
              if (oc) cmin = std::min(cmin, std::hypot(dx, dy) * res);
              if (std::abs(dx) <= R5 && std::abs(dy) <= R5) { occ += oc; ++tot; }
            }
          csum += tot ? (double)occ / tot : 0;
          ++ns;
          for (size_t pi = 0; pi < near_p.size(); ++pi) {
            const double px = ((*in.people)[2 * pi] - W.ox) * c.inv_res, py = ((*in.people)[2 * pi + 1] - W.oy) * c.inv_res;
            if (std::hypot(px - (x + 0.5), py - (y + 0.5)) * res <= 1.0) near_p[pi] = 1;
          }
        }
        const int x = at % gw, y = at / gw;
        int bj = -1, bv = w.cd[at];
        for (int dy = -1; dy <= 1; ++dy)
          for (int dx = -1; dx <= 1; ++dx) {
            const int u = x + dx, v = y + dy;
            if ((!dx && !dy) || u < 0 || v < 0 || u >= gw || v >= gh) continue;
            const int j = v * gw + u;
            if (pass(j) && w.cd[j] >= 0 && w.cd[j] < bv) { bv = w.cd[j]; bj = j; }
          }
        at = bj;
      }
      for (uint8_t b : near_p) np += b;
      q.clear_m = (float)cmin;
      q.clutter = ns ? (float)(csum / ns) : 0.f;
      q.people = np;
    }
    std::vector<int> ord(m);
    for (int k = 0; k < m; ++k) ord[k] = k;
    std::stable_sort(ord.begin(), ord.end(), [&](int a, int b) { return all[a].gpt > all[b].gpt; });
    std::vector<int> keep;
    {   // gpt 큰 순 (cap − big) 개 + 남은 것에서 틈 큰 순 big 개 + 지금 목표 · fuel 고른 것
      const int big = std::max(0, std::min(in.cand_big, in.cand_cap - 1)), nsmall = in.cand_cap - big;
      std::vector<uint8_t> in_(m, 0);
      for (int k : ord) if ((int)keep.size() < nsmall) { keep.push_back(k); in_[k] = 1; }
      std::vector<int> rest;
      for (int k : ord) if (!in_[k]) rest.push_back(k);
      std::stable_sort(rest.begin(), rest.end(), [&](int a, int b) { return all[a].open_m > all[b].open_m; });
      for (size_t q = 0; q < rest.size() && (int)q < big; ++q) { keep.push_back(rest[q]); in_[rest[q]] = 1; }
      for (int k : ord) if (!in_[k] && (all[k].cur || all[k].fuel)) { keep.push_back(k); in_[k] = 1; }
      std::stable_sort(keep.begin(), keep.end(), [&](int a, int b) { return all[a].gpt > all[b].gpt; });   // 글자 차례는 gpt 큰 순 그대로
    }
    if (m > (int)keep.size()) {   // cand_cap 에 걸림 (셈만)
      st.cc_dec += 1;
      std::vector<uint8_t> kept(m, 0);
      for (int k : keep) kept[k] = 1;
      for (int k = 0; k < m; ++k) if (!kept[k]) {
        st.cc_drop += 1; st.cc_d += all[k].len_m; st.cc_s += all[k].open_m;
        st.cc_dmax = std::max(st.cc_dmax, (double)all[k].len_m); st.cc_smax = std::max(st.cc_smax, (double)all[k].open_m);
      }
    }
    in.cands->clear();
    for (int k : keep) {   // 떠나는 값: 이 후보로 시작하는 순회 − 가장 좋은 순회
      std::vector<int> o3;
      bool ex = false;
      const double ck = m > 1 ? detail::tour(C, m, k + 1, o3, &ex) : best;
      all[k].opp_s = (float)std::max(0.0, ck - best);
      in.cands->push_back(all[k]);
    }
  }
  mem.vp = nodes[pick].vp;
  mem.anc = nodes[pick].anc;
  return done(nodes[pick].vp);
}

}  // namespace fu
