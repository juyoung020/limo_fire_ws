// 위층 목표 후보 (판단 모델 · slot0 가 고르는 ≤ 8 곳) — 로봇 지도에서 호스트(CPU) 한 벌로 만든다. 시뮬(combine_run · low_explore_check)과 실물 노드가 같은 함수.
// 환경 안 hier 후보(mdp/frontier.h cand_pick — 프론티어 칸 자체, 가까운 순)는 그대로 두고, 위층은 이것을 쓴다 (아래층 규칙 추종기에는 목표 칸을 그대로 넘김).
//
// 후보 = 덩어리마다 고른 "서는 자리"(시점). 근거:
//   FUEL (Zhou 외, RA-L 2021, arXiv 2010.11561 · 저자 설정 algorithm.xml): 덩어리 둘레 고리에서 시점을 뽑고, 모름에서 min_candidate_clearance 0.21 m 안 · 로봇에서
//     min_candidate_dist 0.75 m 안 시점은 버림, 시점 없는 덩어리는 쉼(dormant).
//   TARE (Cao 외, RSS 2021): 시점 보상 = 아직 안 덮인 곳 덮음, 하나씩 고르며 이미 고른 시점과 겹치는 몫을 빼고 다시 셈(덮음은 submodular) → 방향이 몰리지 않음.
//   FALCON (Zhang 외, T-RO 2024, arXiv 2407.00577): 바뀐 상자 안만 다시 · 간격(clearance)으로 좁은 곳을 끊어 구역을 나눔.
//   TASG-Explore (arXiv 2609.08512): 이전 결정의 덩어리를 다음 결정의 열쇠로 → 같은 덩어리가 같은 번호.
// 차례 (한 결정):
//   1. 프론티어 찾기 = fuel_upper.h 와 같은 frontier_search (닫힘 규칙 버퍼 · 몸통 반경 · 깊이 층) — 갈 수 있는 칸 · BFS 거리 · 덩어리 라벨
//   2. 거리 지도 (번진 칸 둘레 상자만, 3·4 모따기 두 번 훑기): 막힘(지도 막힘 + 깊이 층)까지 · 모름까지 — 시점 간격·길 위 좁은 틈·방 나누기를 칸마다 O(1)
//   3. 자격 프론티어 칸 → 덩어리 → 합침·나눔 (fuel_upper.h detail::merge_split, 같은 값)
//   4. 로봇에서 모따기 거리 한 번 (fuel_upper.h detail::chamfer) — 가는 시간 = 길 / v_eff + 첫 회전 / ω_eff
//   5. 덩어리마다 시점: 가장 가까운 칸 + 가운데 둘레 0.6 · 1.2 · 1.8 m 고리 36 칸 중, 갈 수 있음 · 피하는 칸 아님 · 로봇에서 vp_min 밖 · 막힘에서 몸통 + margin 밖 ·
//      모름에서 unk_clear 밖 · 덩어리 칸이 막힘에 안 가리고 보임 → 점수 = 예상 새 넓이 / (가는 시간 + t0). 간격 맞는 것이 없으면 모름 간격 없이 한 번 더, 그래도 없으면 쉼
//   6. 안정 번호: 지난 결정의 후보(쉬지 않은 모든 덩어리)와 대표 칸 match_m 안으로 짝 → 같은 번호. 지난 시점이 아직 자격 있고 점수가 새 최선의 keep 배 이상이면 같은 칸
//   7. ≤ K 개 고르기: 남은 덩어리마다 "아직 안 덮인" 모름 넓이 / 시간 이 가장 큰 것을 차례로 (고른 것의 광선이 지난 모름 칸은 덮였다고 표시) — 0 번 = slot0
// 정수 · 순서가 정해진 차례만 쓰므로 같은 지도면 같은 결과 (시뮬 == 실물 — low_explore_check). 실수 셈도 같은 차례라 비트 같음.
#pragma once
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <functional>
#include <vector>

#include "fuel_upper.h"

namespace cand {

struct Opt {
  int k = mdp::hier::K;       // 후보 상한
  float vp_min = 0.75f;       // m — 로봇에서 이 거리 안 시점은 안 씀 (FUEL min_candidate_dist)
  float unk_clear = 0.0f;     // m — 시점 둘레 모름 간격 (FUEL min_candidate_clearance 0.21 — 3D 비행체 값. 2D 에서는 프론티어 가에 서야 가장 많이 봐 정답 견줌이 0.77 → 0.30 으로 떨어져 끔)
  float margin = 0.25f;       // m — 시점의 막힘 간격 = 몸통 반경(close_radius) + 이것. 라이다 면(0.11 m) 아래 가구 다리·받침·낮은 물체는 로봇 지도에 없어
                              //   막힘 곁 시점은 실제로 못 서는 일이 많음 — 정답 격자로 갈 수 있는 후보 0.72~0.79 → 0.88~0.91 (0.05 → 0.25, 시험·실물 방 장면)
  float margin_min = 0.05f;   // m — margin 으로 시점이 없는 덩어리는 이 간격으로 한 번 더 (좁은 문·복도)
  float merge_m = 1.0f;       // m — 덩어리 합침 (fuel_upper.h Opt::merge_m 과 같은 값)
  float split_m = 1.5f;       // m — 덩어리 나눔 (fuel_upper.h Opt::split_m)
  float gain_m = 4.0f;        // m — 예상 새 넓이 광선 길이 (잠정 — 실물 방 7 × 8 m, 라이다 8 m)
  int rays = 32;              // 광선 수 (64 의 약수)
  float v_eff = 0.981f, w_eff = 3.232f, t0 = 0.10f;   // 가는 시간 (fuel 과 같은 맞춤 값)
  float match_m = 1.0f;       // m — 지난 후보와 같은 덩어리로 보는 거리 (대표 칸 사이)
  float keep = 0.8f;          // 지난 시점 점수가 새 최선의 이 배 이상이면 그 칸 그대로
  float door_m = 0.45f;       // m — 막힘까지 이 거리 미만 칸에서 방을 끊음 (fuel_upper.h Opt::door_m)
  float unk_depth_m = 0.0f;   // m — 광선마다 처음 만난 모름에서 이 길이까지만 셈 (0 = 끝까지). 1 m 로 줄이면 정답 견줌이 나빠짐 (0.75 → 0.30, bookstore 씨앗 4)
  float path_unk_m = 0.0f;    // m — 길(시점 → 로봇)이 시점에서 path_skip_m 넘게 떨어진 곳에서는 모름에서 이만큼 떨어져야 (빔 줄 사이 가는 빈 칸 길 · 벽 틈 너머로 가는 길 막기, 0 = 끔)
  float path_skip_m = 0.6f;   // m — 시점 둘레 이 길이는 검사 안 함 (프론티어 곁이라 늘 모름 가까움)
  float coarse_m = 0.5f;      // m — 예상 넓이를 이 크기 굵은 칸 수로: 아는 칸 몫이 known_frac 미만인 굵은 칸이 그 자리에서 보면 넘어가는 수 (사용자 목표: 공간을 "대충" 빨리 앎 —
                              //   이미 대충 아는 굵은 칸 안 작은 모름 구멍은 값 0 → 그런 덩어리는 쉼). 0 = 칸 단위 모름 넓이 (옛 방식)
  float known_frac = 0.5f;    // 굵은 칸을 "대충 앎" 으로 보는 아는 칸 몫 (잠정)
  float sliver_m = 0.0f;      // m — 예상 넓이는 아는 칸에서 이 거리 이상인 모름만 (빔 사이 가는 모름 줄 빼기, 0 = 끔 — 0.1 m 는 정답 견줌에 차이 없음)
};

// 후보 하나 (판단 모델에 주는 값 — 로봇이 아는 것만)
struct Cand {
  int id = -1;            // 안정 번호 (판 안에서 같은 덩어리면 같은 번호)
  int age = 0;            // 이 번호가 이어진 결정 수 (0 = 이번에 새로)
  int vp = -1, anc = -1;  // 목표 칸(시점) · 덩어리 대표 칸 (로봇에서 BFS 가장 가까운 프론티어 칸)
  float len_m = 0;        // 로봇 → 시점 길 (모따기, m)
  float turn_rad = 0;     // 첫 걸음 방향과 머리 방향 차
  float t_s = 0;          // 가는 시간 = len / v_eff + turn / ω_eff
  float bearing = 0;      // 시점 방향 (머리 기준, 왼쪽 +)
  float gain_m2 = 0;      // 예상 새로 보이는 모름 넓이 (m², 광선 gain_m 안)
  float marg_m2 = 0;      // 그중 앞 후보들이 안 덮은 몫 (고른 차례의 값)
  float score = 0;        // marg / (t + t0) — 고른 차례
  float open_m = 0;       // 덩어리 폭 = 프론티어 칸 수 × 칸 (m)
  float clear_m = 0;      // 시점에서 막힘까지 (m)
  float path_clear_m = 0; // 길 위 가장 좁은 곳 (막힘까지, m)
  int same_room = 0;      // 로봇과 같은 방 (좁은 곳에서 끊은 갈 수 있는 칸)
  int room_left = 0;      // 같은 방의 다른 후보 수
  int cur = 0;            // 지금 목표 칸이 이 덩어리 · 시점
  int kept = 0;           // 지난 결정의 시점 칸을 그대로 씀
  // 기억 (Mem 이 결정 사이에 들고 감 — 부르는 쪽이 note_* 로 알림)
  int chosen = 0;         // 이 번호를 목표로 보낸 수
  int aborts = 0;         // 이 번호 목표가 실패로 끝난 수
  int last_fail = 0;      // 마지막 실패 까닭 (Why: 0 없음 · 1 경로 없음 · 2 막힘·정체 · 3 닿음 · 4 시간 넘김 · 5 경로 끝 밖 · 6 닿았는데 새로 본 것 없음)
  float since_fail_s = -1;   // 마지막 실패 뒤 시간 (s, 없으면 −1) — 이 번호 또는 시점 1 m 안 실패 자리
  int fail_near = 0;      // 시점에서 fail_m 안에 남은 실패 자리 수 (번호가 바뀌어도 자리로 이어 받음)
  int in_avoid = 0;       // 지금 피하는 칸 둘레 (Ctl 피하기 반경) 안
  int visited = 0;        // 로봇이 이 시점 visit_m 안에 들어간 적 있음 (판 안 자취)
  int recent_fail = 0;    // 최근 실패: 이 번호 또는 시점 fail_m 안 실패 자리가 avoid_s 안에 있었음 — 목록 맨 뒤로 (글자는 남김), 규칙(대신 고르기)은 안 고름
};
// 목표 끝 까닭 (Cand::last_fail · Mem::note_end)
enum Why : int { kNone = 0, kNoPath = 1, kStall = 2, kContact = 3, kTimeout = 4, kOffPath = 5, kNoGain = 6, kReached = 7, kSwitched = 8 };

struct Out {
  std::vector<Cand> c;    // 고른 차례 (0 = 가장 좋은 것)
  std::vector<std::vector<int32_t>> cells;   // c 와 같은 차례 — 그 덩어리의 프론티어 칸 (그림용)
  std::vector<int32_t> dormant;              // 쉬는 덩어리의 대표 칸 (그림용)
  int start_free = 0;     // 설 수 있는 시작 칸
  int n_clu = 0;          // 자격 프론티어 덩어리 (합침·나눔 뒤)
  int n_dormant = 0;      // 시점이 없어 쉬는 덩어리
  int n_near = 0;         // 그중 로봇 곁(대표 칸이 vp_min 안) — 로봇 몸·뒤 사각지대 몫
  int n_pocket = 0;       // 그중 구멍: 덩어리 둘레 굵은 칸이 모두 이미 대충 앎 (coarse_m) — 값 0 으로 쉼
  int rotate = 0;         // 후보 0 개인데 쉬는 덩어리가 모두 로봇 곁 → 제자리 회전으로 봄
  double ms = 0;          // 이 결정에 걸린 시간 (ms)
  double ms_part[6] = {0, 0, 0, 0, 0, 0};   // 찾기 · 거리 지도·방 · 덩어리 · 로봇 거리 · 시점 · 고르기
};

// 결정 사이 기억 (판마다 새로 — 판단 모델 그림·글에도 그대로 씀)
struct Mem {
  struct Prev { int id, anc, vp, age; float score; };
  std::vector<Prev> prev;       // 지난 결정의 쉬지 않은 모든 덩어리
  int next_id = 0;
  struct Hist { int chosen = 0, aborts = 0, last_fail = 0, fail_step = -1; };
  std::vector<Hist> hist;       // 번호마다
  struct Fail { float x, y; int why, step; };   // 실패 자리 (세계 m) — 번호가 바뀌어도 자리로 이어 받음
  std::vector<Fail> fails;
  struct Goal { int id; float x, y; int start, end, why; };   // 목표 기록 (보낸 차례, end −1 = 진행 중)
  std::vector<Goal> goals;
  struct Pose { float x, y, th; int step; };    // 로봇 자취 (pose_every 걸음마다, 최근 trail_n 개)
  std::vector<Pose> trail;
  std::vector<std::pair<float, float>> visits;  // 지나간 자리 (세계 m, visit_m 격자로 줄인 것)
  int pose_every = 10, trail_n = 60, goal_n = 20;
  long n_end[9] = {0, 0, 0, 0, 0, 0, 0, 0, 0};   // 목표 끝 까닭별 수 (Why — 판 동안, 재기용)
  float visit_m = 0.5f;
  Hist& h(int id) { if ((int)hist.size() <= id) hist.resize(id + 1); return hist[id]; }
  // 걸음마다 (지도 자세)
  void note_pose(int step, float x, float y, float th) {
    if (step % pose_every == 0) { trail.push_back(Pose{x, y, th, step}); if ((int)trail.size() > trail_n) trail.erase(trail.begin()); }
    for (const auto& v : visits) if (std::hypot(v.first - x, v.second - y) < 0.5f * visit_m) return;
    visits.emplace_back(x, y);
  }
  // 목표로 보냄 (같은 번호를 이어 보내면 새 기록 안 함)
  void note_target(int id, int step, float x, float y) {
    if (id < 0) return;
    if (!goals.empty() && goals.back().id == id && goals.back().end < 0) return;
    if (!goals.empty() && goals.back().end < 0) note_end(goals.back().id, step, kSwitched);
    h(id).chosen += 1;
    goals.push_back(Goal{id, x, y, step, -1, kNone});
    if ((int)goals.size() > goal_n) goals.erase(goals.begin());
  }
  // 번호 없는 목표(다른 위층이 보낸 것)의 실패 자리
  void note_fail_at(float x, float y, int why, int step) { fails.push_back(Fail{x, y, why, step}); }
  // 목표 끝 (why: Why). 실패면 그 자리를 기억
  void note_end(int id, int step, int why) {
    if (!goals.empty() && goals.back().id == id && goals.back().end < 0) { goals.back().end = step; goals.back().why = why; }
    if (why >= 0 && why < 9) n_end[why] += 1;
    if (id < 0 || why == kReached || why == kSwitched) return;
    Hist& q = h(id);
    q.aborts += 1; q.last_fail = why; q.fail_step = step;
    for (const Goal& g : goals) if (g.id == id) { fails.push_back(Fail{g.x, g.y, why, step}); break; }
  }
};

struct Work {
  fu::Work fw;                       // frontier_search · 모따기 버퍼 (fuel 과 같은 것)
  std::vector<int32_t> dto, dtu, dtk;   // 막힘·모름·아는 칸까지 3·4 모따기 거리 (칸 × 3)
  std::vector<int32_t> room;         // 방 번호
  std::vector<uint32_t> stamp;       // 덮음 표시 (번호가 cov_id 이면 덮임) · 한 후보 안 겹침 막기
  uint32_t cov_id = 1, seen_id = 1;
  std::vector<uint32_t> seen;
  std::vector<int32_t> bknown, btotal;   // 굵은 칸마다 아는 칸 · 맵 안 칸 수
  std::vector<double> bnew, bcov;        // 굵은 칸마다 이 시점이 새로 볼 칸 (어림) · 고른 후보들이 덮은 칸 (어림)
  std::vector<int32_t> btouch;
  void fit(int n) {
    fw.fit(n);
    if ((int)dto.size() < n) { dto.assign(n, 0); dtu.assign(n, 0); dtk.assign(n, 0); room.assign(n, -1); stamp.assign(n, 0); seen.assign(n, 0); }
  }
};

struct In {
  const sim::EnvCfg* c;
  const sim::World* W;
  const int16_t* L;          // 로봇 지도
  const uint32_t* trail;     // 지나간 칸·본 칸 판
  const uint8_t* cam;        // 깊이 카메라 층 (없으면 nullptr)
  const sim::EnvState* s;
  std::function<bool(int)> avoided;   // 피하는 칸 둘레 (low_explore_ctl.h Ctl::avoided)
  int cur_target = -1;                // 지금 목표 칸 (없으면 −1)
  float fail_m = 1.0f;                // m — 실패 자리 이어 받기 반경
  float avoid_s = 10.f;               // s — 실패 뒤 이 시간 동안 recent_fail (Ctl 피하기 시간과 같게 — 부르는 쪽이 줌)
};

namespace detail {
// 3·4 모따기 거리 (두 번 훑기) — 상자 [x0, x1] × [y0, y1] 안, 씨앗(seed(x, y) 참)에서. 상자 밖은 씨앗 아님으로 봄. 결과 D[y · gw + x] (칸 × 3)
template <class Seed>
inline void chamfer34(int gw, int x0, int y0, int x1, int y1, const Seed& seed, std::vector<int32_t>& D) {
  const int32_t INF = 1 << 28;
  for (int y = y0; y <= y1; ++y)
    for (int x = x0; x <= x1; ++x) D[y * gw + x] = seed(x, y) ? 0 : INF;
  for (int y = y0; y <= y1; ++y)
    for (int x = x0; x <= x1; ++x) {
      int32_t& d = D[y * gw + x];
      if (x > x0) d = std::min(d, D[y * gw + x - 1] + 3);
      if (y > y0) {
        d = std::min(d, D[(y - 1) * gw + x] + 3);
        if (x > x0) d = std::min(d, D[(y - 1) * gw + x - 1] + 4);
        if (x < x1) d = std::min(d, D[(y - 1) * gw + x + 1] + 4);
      }
    }
  for (int y = y1; y >= y0; --y)
    for (int x = x1; x >= x0; --x) {
      int32_t& d = D[y * gw + x];
      if (x < x1) d = std::min(d, D[y * gw + x + 1] + 3);
      if (y < y1) {
        d = std::min(d, D[(y + 1) * gw + x] + 3);
        if (x < x1) d = std::min(d, D[(y + 1) * gw + x + 1] + 4);
        if (x > x0) d = std::min(d, D[(y + 1) * gw + x - 1] + 4);
      }
    }
}
}  // namespace detail

inline Out build(const In& in, const Opt& o, Mem& mem, Work& w) {
  const auto t_start = std::chrono::steady_clock::now();
  Out out;
  const sim::EnvCfg& c = *in.c;
  const sim::World& W = *in.W;
  const int gw = W.gw, gh = W.gh, n = gw * gh;
  w.fit(n);
  const mdp::MapView mv{in.L, c.size, gw, gh};
  const sim::EnvState& s = *in.s;
  const double res = c.res;
  const double rxf = (s.x + s.ex - W.ox) * c.inv_res, ryf = (s.y + s.ey - W.oy) * c.inv_res, th = s.th + s.eth;
  const int rx = (int)std::floor(rxf), ry = (int)std::floor(ryf);
  // ---- 1. 프론티어 찾기 (fuel_upper.h decide 와 같은 버퍼·규칙)
  fu::Work& fw = w.fw;
  mdp::SearchBuf sb{fw.flag.data(), fw.qa.data(), fw.qb.data(), fw.dist.data(), fw.label.data()};
  const int ws = (c.size + 31) / 32, nf = c.size * ws;
  sb.vis = in.trail;
  sb.look = in.trail + nf;
  sb.tws = ws;
  sb.close_min = c.close_min;
  sb.rbc = mdp::robot_cells(c.close_radius, c.res);
  sb.cam = in.cam;
  const mdp::SearchResult r = mdp::frontier_search(mv, rx, ry, mdp::obsgen::reach_robot(c), 1, sb, fw.out.data(), (int)fw.out.size());
  const int n_out = std::min(r.n_found, (int)fw.out.size());
  auto t_mark = t_start;
  const auto lap = [&](int k) { const auto now = std::chrono::steady_clock::now(); out.ms_part[k] += std::chrono::duration<double, std::milli>(now - t_mark).count(); t_mark = now; };
  lap(0);
  const auto done = [&]() {
    out.ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_start).count();
    return out;
  };
  out.start_free = r.start_free;
  if (!r.start_free) { mem.prev.clear(); return done(); }
  const int nr0 = r.n_reach0;
  const auto pass = [&](int idx) { return (fw.flag[idx] & 1) && fw.dist[idx] >= 0; };
  // ---- 2. 거리 지도 (번진 칸 상자 + 광선 길이)
  int bx0 = gw, by0 = gh, bx1 = -1, by1 = -1;
  for (int q = 0; q < nr0; ++q) {
    const int idx = fw.qa[q], x = idx % gw, y = idx / gw;
    bx0 = std::min(bx0, x); bx1 = std::max(bx1, x); by0 = std::min(by0, y); by1 = std::max(by1, y);
  }
  const int Rg = std::max(1, (int)std::lround(o.gain_m * c.inv_res));
  bx0 = std::max(0, bx0 - Rg); by0 = std::max(0, by0 - Rg); bx1 = std::min(gw - 1, bx1 + Rg); by1 = std::min(gh - 1, by1 + Rg);
  detail::chamfer34(gw, bx0, by0, bx1, by1, [&](int x, int y) { return mdp::cls_at(mv, x, y) == mdp::kOcc || (in.cam && in.cam[y * c.size + x]); }, w.dto);
  if (o.unk_clear > 0.f || o.path_unk_m > 0.f)   // 모름까지 거리 — 쓰는 잣대가 켜졌을 때만 (기본 끔, 값은 같음)
    detail::chamfer34(gw, bx0, by0, bx1, by1, [&](int x, int y) { return mdp::cls_at(mv, x, y) == mdp::kUnk; }, w.dtu);
  if (o.sliver_m > 0.f)                           // 아는 칸까지 거리 — 가는 모름 줄 빼기를 켰을 때만
    detail::chamfer34(gw, bx0, by0, bx1, by1, [&](int x, int y) { return mdp::cls_at(mv, x, y) != mdp::kUnk; }, w.dtk);
  const int32_t sliver3 = (int32_t)std::lround(o.sliver_m * c.inv_res * 3.0);
  const auto dto_m = [&](int idx) { return w.dto[idx] * res / 3.0; };
  const auto dtu_m = [&](int idx) { return w.dtu[idx] * res / 3.0; };
  // 방: 막힘에서 door_m 이상인 갈 수 있는 칸을 4 이웃으로 (좁은 칸은 가장 가까운 방 — 아래 room_of)
  const int32_t door3 = (int32_t)std::lround(o.door_m * c.inv_res * 3.0);
  for (int q = 0; q < nr0; ++q) w.room[fw.qa[q]] = -1;
  int nroom = 0;
  for (int q = 0; q < nr0; ++q) {
    const int s0 = fw.qa[q];
    if (w.dto[s0] < door3 || w.room[s0] >= 0) continue;
    int h = 0, t = 0;
    fw.qb[t++] = s0;
    w.room[s0] = nroom;
    while (h < t) {
      const int idx = fw.qb[h++], x = idx % gw, y = idx / gw;
      for (int k = 0; k < 4; ++k) {
        int dx, dy;
        mdp::nb4(k, &dx, &dy);
        const int u = x + dx, v = y + dy;
        if (u < 0 || v < 0 || u >= gw || v >= gh) continue;
        const int j = v * gw + u;
        if (!pass(j) || w.dto[j] < door3 || w.room[j] >= 0) continue;
        w.room[j] = nroom;
        fw.qb[t++] = j;
      }
    }
    ++nroom;
  }
  const auto room_of = [&](int idx) {   // 좁은 칸: 둘레(차례 고정) 가장 가까운 방 칸
    if (w.room[idx] >= 0) return w.room[idx];
    const int x = idx % gw, y = idx / gw, Rd = (int)std::ceil(o.door_m * c.inv_res) + 2;
    for (int rr = 1; rr <= Rd; ++rr)
      for (int dy = -rr; dy <= rr; ++dy)
        for (int dx = -rr; dx <= rr; ++dx) {
          const int u = x + dx, v = y + dy;
          if (u < 0 || v < 0 || u >= gw || v >= gh) continue;
          if (pass(v * gw + u) && w.room[v * gw + u] >= 0) return w.room[v * gw + u];
        }
    return -1;
  };
  int rroom = -1;
  lap(1);
  for (int q = 0; q < nr0 && rroom < 0; ++q) if (w.room[fw.qa[q]] >= 0) rroom = w.room[fw.qa[q]];
  // ---- 3. 덩어리 (fuel 전역 노드와 같은 자격 · 합침 · 나눔)
  const auto eligible = [&](int idx) {
    if (!(fw.flag[idx] & 2) || fw.dist[idx] < mdp::hier::kCandMinDist) return false;
    const int l = fw.label[idx];
    if (l < 0 || l >= n_out || fw.out[l].n < c.close_min || fw.out[l].n < mdp::hier::kCandMinCells) return false;
    return !mdp::looked_cell(mv, sb.look, sb.vis, ws, idx % gw, idx / gw);
  };
  std::vector<fu::Node> nodes;
  {
    std::vector<std::vector<int>> by(n_out);
    for (int q = 0; q < nr0; ++q) { const int idx = fw.qa[q]; if (eligible(idx)) by[fw.label[idx]].push_back(idx); }
    for (int l = 0; l < n_out; ++l) if (!by[l].empty()) { fu::Node nd; nd.cells.swap(by[l]); nodes.push_back(std::move(nd)); }
  }
  const auto fill = [&](fu::Node& nd) {
    double sx = 0, sy = 0;
    int bd = 1 << 30, ba = -1;
    for (int idx : nd.cells) {
      sx += idx % gw + 0.5; sy += idx / gw + 0.5;
      const int d = fw.dist[idx] >= 0 ? fw.dist[idx] : (1 << 29);
      if (d < bd || (d == bd && idx < ba)) { bd = d; ba = idx; }
    }
    nd.cx = sx / nd.cells.size(); nd.cy = sy / nd.cells.size();
    nd.anc = ba;
    nd.anc_d = bd;
  };
  for (fu::Node& nd : nodes) fill(nd);
  fu::detail::merge_split(nodes, gw, (int)(o.merge_m * c.inv_res + 0.5f), (double)o.split_m * c.inv_res);
  for (fu::Node& nd : nodes) fill(nd);
  std::sort(nodes.begin(), nodes.end(), [](const fu::Node& a, const fu::Node& b) { return a.anc_d != b.anc_d ? a.anc_d < b.anc_d : a.anc < b.anc; });
  out.n_clu = (int)nodes.size();
  lap(2);
  // ---- 4. 로봇에서 모따기 거리
  std::vector<int32_t> cells0(fw.qa.begin(), fw.qa.begin() + nr0);
  fu::detail::chamfer(r.start, gw, gh, pass, cells0, nr0, fw.cd, fw.bk);
  lap(3);
  const auto len_m = [&](int idx) { return fw.cd[idx] >= 0 ? fw.cd[idx] * 0.2 * res : 1e6; };
  const auto descend = [&](int idx, const std::function<bool(int)>& visit) {   // 시점에서 로봇 쪽으로 모따기 거리를 거슬러 감 (visit 가 거짓이면 멈춤)
    int at = idx;
    for (int g = 0; g < 4 * n && at >= 0 && fw.cd[at] > 0; ++g) {
      if (!visit(at)) return;
      const int x = at % gw, y = at / gw;
      int bj = -1, bv = fw.cd[at];
      for (int dy = -1; dy <= 1; ++dy)
        for (int dx = -1; dx <= 1; ++dx) {
          const int u = x + dx, v = y + dy;
          if ((!dx && !dy) || u < 0 || v < 0 || u >= gw || v >= gh) continue;
          const int j = v * gw + u;
          if (pass(j) && fw.cd[j] >= 0 && fw.cd[j] < bv) { bv = fw.cd[j]; bj = j; }
        }
      at = bj;
    }
  };
  const auto first_dir = [&](int idx) {   // 로봇에서 0.5 m 떨어진 길 칸 방향
    const int stop = (int)(0.5 / res) * 5;
    int at = idx;
    descend(idx, [&](int a) { at = a; return fw.cd[a] > stop; });
    return std::atan2(at / gw + 0.5 - ryf, at % gw + 0.5 - rxf);
  };
  const auto t_of = [&](int idx) { return len_m(idx) / o.v_eff + fu::detail::wrap(first_dir(idx) - th) / o.w_eff; };
  // ---- 굵은 칸 (coarse_m) — 상자 안 굵은 칸마다 아는 칸 수
  const int B = o.coarse_m > 0.f ? std::max(1, (int)std::lround(o.coarse_m * c.inv_res)) : 0;
  const int bw = B ? (gw + B - 1) / B : 0, bh = B ? (gh + B - 1) / B : 0;
  (void)bh;
  if (B) {
    if ((int)w.bknown.size() < bw * bh) { w.bknown.assign(bw * bh, 0); w.btotal.assign(bw * bh, 0); w.bnew.assign(bw * bh, 0.0); w.bcov.assign(bw * bh, 0.0); }
    const int qx0 = bx0 / B, qy0 = by0 / B, qx1 = bx1 / B, qy1 = by1 / B;
    for (int qy = qy0; qy <= qy1; ++qy)
      for (int qx = qx0; qx <= qx1; ++qx) {
        int kn = 0, tot = 0;
        for (int y = qy * B; y < std::min(gh, qy * B + B); ++y)
          for (int x = qx * B; x < std::min(gw, qx * B + B); ++x) { ++tot; kn += mdp::cls_at(mv, x, y) != mdp::kUnk; }
        w.bknown[qy * bw + qx] = kn; w.btotal[qy * bw + qx] = tot; w.bnew[qy * bw + qx] = 0.0; w.bcov[qy * bw + qx] = 0.0;
      }
  }
  // ---- 5. 시점
  const int step = std::max(1, mdp::hier::kGainRays / std::max(1, o.rays));
  const int Ud = o.unk_depth_m > 0.f ? std::max(1, (int)std::lround(o.unk_depth_m * c.inv_res)) : 1 << 20;
  const double ray_k = (double)step * 2.0 * M_PI / mdp::hier::kGainRays;   // 광선 무게 t → 칸 수 (부채꼴 t · dθ)
  const auto gain = [&](int vp, bool marginal) {   // 광선이 닿는 모름 넓이 (칸 × t 무게 — cand_gain_ray 와 같은 무게). marginal: 덮인 칸 빼고, 같은 후보 안 겹침 한 번만
    long g = 0;
    ++w.seen_id;
    if (B) {   // 굵은 칸: 대충 앎으로 넘어가는 굵은 칸 수 (marginal: 앞 후보들이 덮은 몫을 아는 칸에 더함)
      w.btouch.clear();
      for (int a = 0; a < mdp::hier::kGainRays; a += step)
        mdp::gain_ray_walk(mv, vp % gw, vp / gw, a, Rg, [&](int x, int y, int t, uint8_t cl) {
          if (cl != mdp::kUnk) return;
          const int j = y * gw + x;
          if (w.seen[j] == w.seen_id || (marginal && w.stamp[j] == w.cov_id)) return;
          w.seen[j] = w.seen_id;
          const int b = (y / B) * bw + x / B;
          if (w.bnew[b] == 0.0) w.btouch.push_back(b);
          w.bnew[b] += t * ray_k;
        });
      for (int b : w.btouch) {
        const double base = w.bknown[b] + (marginal ? w.bcov[b] : 0.0), need = o.known_frac * w.btotal[b];
        if (base < need && base + w.bnew[b] >= need) ++g;
        w.bnew[b] = 0.0;
      }
      return g;
    }
    for (int a = 0; a < mdp::hier::kGainRays; a += step) {
      int t_u = -1;   // 이 광선에서 처음 만난 모름의 t
      mdp::gain_ray_walk(mv, vp % gw, vp / gw, a, Rg, [&](int x, int y, int t, uint8_t cl) {
        if (cl != mdp::kUnk) return;
        if (t_u < 0) t_u = t;
        if (t - t_u >= Ud) return;
        const int j = y * gw + x;
        if (sliver3 > 0 && w.dtk[j] < sliver3) return;   // 빔 사이 가는 모름 줄
        if (marginal && (w.stamp[j] == w.cov_id || w.seen[j] == w.seen_id)) return;
        w.seen[j] = w.seen_id;
        g += t;
      });
    }
    return g;
  };
  const double area_k = B ? (double)B * B * res * res : (double)step * 2.0 * M_PI / mdp::hier::kGainRays * res * res;   // 이득 → m² (굵은 칸 넓이 · 부채꼴 t · dθ · res²)
  const auto los = [&](int a, int b) {
    const int x0 = a % gw, y0 = a / gw, x1 = b % gw, y1 = b / gw, N = std::max(std::abs(x1 - x0), std::abs(y1 - y0));
    for (int t = 1; t < N; ++t) {
      const int x = x0 + (int)std::lround((double)(x1 - x0) * t / N), y = y0 + (int)std::lround((double)(y1 - y0) * t / N);
      if (mdp::cls_at(mv, x, y) == mdp::kOcc) return false;
    }
    return true;
  };
  const double dmin = std::max((double)o.vp_min, (double)c.low_pass_r);
  const auto vp_ok = [&](int cell, int anc, bool unk, double clear_need) {
    if (cell < 0 || cell >= n || !pass(cell) || in.avoided(cell) || fw.dist[cell] < mdp::hier::kCandMinDist) return false;
    if (std::hypot(cell % gw + 0.5 - rxf, cell / gw + 0.5 - ryf) * res < dmin) return false;
    if (dto_m(cell) < clear_need) return false;
    if (unk && dtu_m(cell) < o.unk_clear) return false;
    if (cell != anc && !los(cell, anc)) return false;
    if (o.path_unk_m > 0.f) {   // 길 위 모름 간격 (시점 곁 path_skip_m 은 뺌)
      const int skip = (int)std::lround(o.path_skip_m * c.inv_res * 5.0);   // 모따기 거리 단위 (칸 × 5)
      const int32_t need3 = (int32_t)std::lround(o.path_unk_m * c.inv_res * 3.0);
      const int c0 = fw.cd[cell];
      bool ok = true;
      descend(cell, [&](int a) { if (c0 - fw.cd[a] > skip && w.dtu[a] < need3) { ok = false; return false; } return true; });
      if (!ok) return false;
    }
    return true;
  };
  struct Live { int node, vp, id, age, kept; long g; double t; };
  std::vector<Live> live;
  std::vector<uint8_t> used_prev(mem.prev.size(), 0);
  const int Rm = (int)std::lround(o.match_m * c.inv_res);
  for (int k = 0; k < (int)nodes.size(); ++k) {
    const fu::Node& nd = nodes[k];
    int bv = -1;
    long bg = 0;
    double bs = -1, bt = 0;
    bool pocket = false;
    if (B) {   // 구멍: 덩어리 칸과 그 4 이웃이 든 굵은 칸이 모두 이미 대충 앎 → 값 0 (대충 아는 곳 안의 작은 모름은 탐사를 붙들지 않음)
      pocket = true;
      for (size_t q = 0; q < nd.cells.size() && pocket; ++q) {
        const int x0 = nd.cells[q] % gw, y0 = nd.cells[q] / gw;
        for (int k = -1; k < 4 && pocket; ++k) {
          int dx = 0, dy = 0;
          if (k >= 0) mdp::nb4(k, &dx, &dy);
          const int u = x0 + dx, v = y0 + dy;
          if (u < 0 || v < 0 || u >= gw || v >= gh) continue;
          const int b = (v / B) * bw + u / B;
          if (w.bknown[b] < o.known_frac * w.btotal[b]) pocket = false;
        }
      }
    }
    for (int pass_i = 0; pass_i < 3 && bv < 0 && !pocket; ++pass_i) {   // 0: 간격 margin + 모름 간격 · 1: margin · 2: margin_min
      if (pass_i == 1 && o.unk_clear <= 0.f) continue;   // 모름 간격을 안 쓰면 0 과 같음
      const bool unk = pass_i == 0;
      const double clear_need = c.close_radius + (pass_i < 2 ? o.margin : o.margin_min);
      const auto test = [&](int cell) {
        if (!vp_ok(cell, nd.anc, unk, clear_need)) return;
        const long g = gain(cell, false);
        if (g <= 0) return;
        const double t = t_of(cell), sc = g / (t + o.t0);
        if (sc > bs) { bs = sc; bv = cell; bg = g; bt = t; }
      };
      test(nd.anc);
      static const double kR[3] = {0.6, 1.2, 1.8};
      for (int ri = 0; ri < 3; ++ri)
        for (int a = 0; a < 12; ++a) {
          const double rr = kR[ri], ang = 2 * M_PI * (a + 0.5 * ri) / 12;
          const int x = (int)std::floor(nd.cx + rr / res * std::cos(ang)), y = (int)std::floor(nd.cy + rr / res * std::sin(ang));
          if (x >= 0 && y >= 0 && x < gw && y < gh) test(y * gw + x);
        }
    }
    // 지난 후보와 짝 (대표 칸이 match_m 안 — 가장 가까운 것, 쓰지 않은 것만)
    int pi = -1;
    long pd = (long)Rm * Rm + 1;
    for (int q = 0; q < (int)mem.prev.size(); ++q) {
      if (used_prev[q]) continue;
      const int ax = mem.prev[q].anc % gw, ay = mem.prev[q].anc / gw;
      for (int idx : nd.cells) {
        const long dx = idx % gw - ax, dy = idx / gw - ay;
        if (dx * dx + dy * dy < pd) { pd = dx * dx + dy * dy; pi = q; }
      }
    }
    if (bv < 0) {   // 쉼 (짝이 있어도 — 그 덩어리는 지금 설 자리가 없음)
      out.n_dormant += 1;
      out.dormant.push_back(nd.anc);
      out.n_pocket += pocket;
      if (std::hypot(nd.anc % gw + 0.5 - rxf, nd.anc / gw + 0.5 - ryf) * res < dmin) out.n_near += 1;
      continue;
    }
    Live L{k, bv, -1, 0, 0, bg, bt};
    if (pi >= 0) {
      used_prev[pi] = 1;
      const Mem::Prev& p = mem.prev[pi];
      L.id = p.id;
      L.age = p.age + 1;
      if (p.vp != bv && vp_ok(p.vp, nd.anc, false, c.close_radius + o.margin_min)) {   // 지난 칸이 아직 자격 있고 점수가 keep 배 이상이면 그대로
        const long g = gain(p.vp, false);
        const double t = t_of(p.vp);
        if (g > 0 && g / (t + o.t0) >= o.keep * bs) { L.vp = p.vp; L.g = g; L.t = t; L.kept = 1; }
      } else if (p.vp == bv) {
        L.kept = 1;
      }
    } else {
      L.id = mem.next_id++;
    }
    live.push_back(L);
  }
  lap(4);
  if (live.empty() && out.n_dormant > 0 && out.n_near == out.n_dormant) out.rotate = 1;
  // ---- 7. 고르기 (덮음 겹침 빼며 차례로)
  ++w.cov_id;
  std::vector<uint8_t> taken(live.size(), 0);
  std::vector<int> order;
  for (int it = 0; it < o.k && (int)order.size() < (int)live.size(); ++it) {
    int bi = -1;
    double bs = -1;
    long bm = 0;
    for (int i = 0; i < (int)live.size(); ++i) {
      if (taken[i]) continue;
      const long m = gain(live[i].vp, true);
      const double sc = m / (live[i].t + o.t0);
      if (sc > bs || (sc == bs && live[i].id < live[bi].id)) { bs = sc; bi = i; bm = m; }
    }
    if (bi < 0 || (bm <= 0 && !order.empty())) break;   // 남은 것이 모두 앞 후보에 덮임
    taken[bi] = 1;
    order.push_back(bi);
    // 덮음 표시
    ++w.seen_id;
    const int vp = live[bi].vp;
    for (int a = 0; a < mdp::hier::kGainRays; a += step)
      { int t_u = -1;
        mdp::gain_ray_walk(mv, vp % gw, vp / gw, a, Rg, [&](int x, int y, int t, uint8_t cl) {
          if (cl != mdp::kUnk) return;
          if (t_u < 0) t_u = t;
          const int j = y * gw + x;
          if (B) {   // 굵은 칸: 덮은 몫을 그 굵은 칸에 더함 (칸마다 한 번)
            if (w.stamp[j] == w.cov_id) return;
            w.stamp[j] = w.cov_id;
            w.bcov[(y / B) * bw + x / B] += t * ray_k;
            return;
          }
          if (t - t_u < Ud && (sliver3 <= 0 || w.dtk[j] >= sliver3)) w.stamp[j] = w.cov_id;
        }); }
    Cand cd;
    const Live& L = live[bi];
    const fu::Node& nd = nodes[L.node];
    cd.id = L.id; cd.age = L.age; cd.vp = L.vp; cd.anc = nd.anc; cd.kept = L.kept;
    cd.len_m = (float)len_m(L.vp);
    cd.turn_rad = (float)fu::detail::wrap(first_dir(L.vp) - th);
    cd.t_s = (float)L.t;
    const double b = std::atan2(L.vp / gw + 0.5 - ryf, L.vp % gw + 0.5 - rxf) - th;
    cd.bearing = (float)std::atan2(std::sin(b), std::cos(b));
    cd.gain_m2 = (float)(L.g * area_k);
    cd.marg_m2 = (float)(bm * area_k);
    cd.score = (float)bs;
    cd.open_m = (float)(nd.cells.size() * res);
    cd.clear_m = (float)dto_m(L.vp);
    double pc = cd.clear_m;
    descend(L.vp, [&](int a) { pc = std::min(pc, dto_m(a)); return true; });
    cd.path_clear_m = (float)pc;
    const int rm = room_of(L.vp);
    cd.same_room = rroom >= 0 && rm == rroom;
    cd.cur = in.cur_target >= 0 && (in.cur_target == L.vp || std::find(nd.cells.begin(), nd.cells.end(), in.cur_target) != nd.cells.end());
    {   // 기억
      const double wx = W.ox + (L.vp % gw + 0.5) * res, wy = W.oy + (L.vp / gw + 0.5) * res, dt = (double)c.substeps * c.dt;
      int fs = -1, fw_ = 0;
      if (L.id < (int)mem.hist.size()) { const Mem::Hist& q = mem.hist[L.id]; cd.chosen = q.chosen; cd.aborts = q.aborts; cd.last_fail = q.last_fail; fs = q.fail_step; fw_ = q.last_fail; }
      for (const Mem::Fail& f : mem.fails)
        if (std::hypot(f.x - wx, f.y - wy) < in.fail_m) {
          cd.fail_near += 1;
          if (f.step > fs) { fs = f.step; fw_ = f.why; }
        }
      if (fs >= 0) { cd.since_fail_s = (float)((s.step - fs) * dt); cd.last_fail = fw_; }
      cd.recent_fail = fs >= 0 && cd.since_fail_s < in.avoid_s;
      cd.in_avoid = in.avoided(L.vp) || in.avoided(nd.anc) ? 1 : 0;   // 시점은 피하는 칸을 안 고르므로 덩어리 대표 칸까지
      for (const auto& v : mem.visits) if (std::hypot(v.first - wx, v.second - wy) < mem.visit_m) { cd.visited = 1; break; }
    }
    out.c.push_back(cd);
    out.cells.push_back(nd.cells);
  }
  for (Cand& a : out.c) {   // 같은 방의 다른 후보 수
    const int ra = room_of(a.vp);
    for (const Cand& b : out.c) a.room_left += &a != &b && ra >= 0 && room_of(b.vp) == ra;
  }
  {   // 최근 실패한 후보는 목록 맨 뒤로 (차례는 유지 — 판단 모델은 실패 값과 함께 보지만 0 번·규칙으로는 안 감)
    std::vector<size_t> ord;
    for (size_t i = 0; i < out.c.size(); ++i) if (!out.c[i].recent_fail) ord.push_back(i);
    for (size_t i = 0; i < out.c.size(); ++i) if (out.c[i].recent_fail) ord.push_back(i);
    std::vector<Cand> c2;
    std::vector<std::vector<int32_t>> cl2;
    for (size_t i : ord) { c2.push_back(out.c[i]); cl2.push_back(std::move(out.cells[i])); }
    out.c.swap(c2);
    out.cells.swap(cl2);
  }
  lap(5);
  // 기억: 쉬지 않은 모든 덩어리 (고르지 않은 것도 — 번호가 상한 밖에서도 이어지게)
  mem.prev.clear();
  for (const Live& L : live) mem.prev.push_back(Mem::Prev{L.id, nodes[L.node].anc, L.vp, L.age, (float)(L.g / (L.t + o.t0))});
  return done();
}

}  // namespace cand
