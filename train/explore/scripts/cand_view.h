// 위층 목표 후보 (판단 모델 · slot0 가 고르는 ≤ 8 곳) — 로봇 지도에서 호스트(CPU) 한 벌로 만든다. 시뮬(combine_run · low_explore_check)과 실물 노드가 같은 함수.
// 환경 안 hier 후보(mdp/frontier.h cand_pick — 프론티어 칸 자체, 가까운 순)는 그대로 두고, 위층은 이것을 쓴다 (아래층 규칙 추종기에는 목표 칸을 그대로 넘김).
//
// 후보 = 덩어리마다 고른 "서는 자리"(시점). 근거:
//   FUEL (Zhou 외, RA-L 2021, arXiv 2010.11561 · 저자 설정 algorithm.xml): 덩어리 둘레 고리에서 시점을 뽑고, 로봇에서 min_candidate_dist 0.75 m 안 시점은 버림,
//     시점 없는 덩어리는 쉼(dormant).
//   TARE (Cao 외, RSS 2021): 시점 보상 = 아직 안 덮인 곳 덮음, 하나씩 고르며 이미 고른 시점과 겹치는 몫을 빼고 다시 셈(덮음은 submodular) → 방향이 몰리지 않음.
//   FALCON (Zhang 외, T-RO 2024, arXiv 2407.00577): 바뀐 상자 안만 다시 · 간격(clearance)으로 좁은 곳을 끊어 구역을 나눔.
//   TASG-Explore (arXiv 2609.08512): 이전 결정의 덩어리를 다음 결정의 열쇠로 → 같은 덩어리가 같은 번호.
// 한 결정 = build() 의 일곱 단계 (README "후보 뽑기" 에 단계별 그림):
//   1. 프론티어 칸     frontier_search (fuel_upper.h 와 같은 버퍼·닫힘 규칙) — 갈 수 있는 칸 · BFS 거리 · 프론티어 라벨. 번진 칸 둘레 상자 안만 거리 지도·방
//   2. 덩어리          자격 프론티어 칸 → 라벨별 → 합침·나눔 (fuel_upper.h detail::merge_split)
//   3. 시점 뽑기       덩어리마다 가장 가까운 칸 + 가운데 둘레 고리 3 × 12 칸
//   4. 거르기          갈 수 있음 · 피하는 칸 아님 · 로봇에서 vp_min 밖 · 막힘에서 몸통 + margin 밖 · 덩어리 칸이 보임. 하나도 없으면 margin_min 으로 한 번 더, 그래도 없으면 쉼
//   5. 점수            예상 새 넓이(굵은 칸 "대충 앎" 으로 넘어가는 수) / (가는 시간 + t0) — 덩어리마다 가장 큰 시점 하나
//   6. 번호 잇기       지난 결정의 덩어리와 대표 칸 match_m 안 → 같은 번호 (지난 시점이 아직 좋으면 그 칸 그대로). 7 의 같은 점수 가르기에 번호를 쓰므로 고르기보다 먼저
//   7. 고르기 · 기억   ≤ K 개를 덮음 겹침을 빼며 차례로 (0 번 = slot0) → 판단 모델 값 · 실패 기억 · 다음 결정용 기억
// 정수 · 순서가 정해진 차례만 쓰므로 같은 지도면 같은 결과 (시뮬 == 실물 — low_explore_check). 실수 셈도 같은 차례라 비트 같음.
#pragma once
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <functional>
#include <vector>

#include "fuel_upper.h"
#include "obs_ref.h"   // sim::ReachKeep

namespace cand {

// 상수 (단위 · 출처). "잠정" = 재지 않고 정한 값, "잰 값" = 시험·실물 장면에서 비교해 고른 값
struct Opt {
  int k = mdp::hier::K;       // 후보 상한 (판단 모델 슬롯 수)
  // 4. 거르기
  float vp_min = 0.75f;       // m — 로봇에서 이 거리 안 시점은 안 씀 (FUEL min_candidate_dist — 몸 · 뒤 사각지대)
  float margin = 0.25f;       // m — 시점의 막힘 간격 = 몸통 반경(close_radius) + 이것. 라이다 면(0.11 m) 아래 가구 다리·받침·낮은 물체는 로봇 지도에 없어
                              //   막힘 곁 시점은 실제로 못 서는 일이 많음 — 정답 격자로 갈 수 있는 후보 0.72~0.79 → 0.88~0.91 (잰 값: 0.05 → 0.25, 시험·실물 방 장면)
  float margin_min = 0.05f;   // m — margin 으로 시점이 없는 덩어리는 이 간격으로 한 번 더 (좁은 문·복도, 잠정)
  int give_up = 2;            // 고른 시점에서 fail_m 안 실패 자리(경로 없음·막힘·정체·경로 끝 밖)가 이만큼이면 그 덩어리를 버림(쉼 — '대충 다 앎' 판정에선 구멍과 같이 봄).
                              //   0 = 끔. 깊이 층이 촘촘하면(layer_grid) 끝 무렵 남은 시점이 층 가구에 막혀 실패 → 10 s 피함 → 다시 고름을 되풀이해 진행 없음으로 끝남
                              //   (sim/analysis/real_limo.md "층 전용 격자"). FUEL 처럼 못 가는 프론티어를 포기. 2 = 잰 값 아님 — 같은 자리 두 번 실패
  // 2. 덩어리
  float merge_m = 1.0f;       // m — 덩어리 합침 (fuel_upper.h Opt::merge_m 과 같은 값)
  float split_m = 1.5f;       // m — 덩어리 나눔 (fuel_upper.h Opt::split_m)
  // 3. 시점 뽑기
  double ring_m[3] = {0.6, 1.2, 1.8};   // m — 덩어리 가운데 둘레 고리 반지름 (FUEL 은 1.5~2.5 m 고리 · 3D 비행체 — 2D 실내 방 크기로 줄인 잠정값)
  int ring_n = 12;            // 고리마다 칸 수 (30°, 고리마다 반 칸씩 엇갈림 — 잠정)
  // 5. 점수
  float gain_m = 4.0f;        // m — 예상 새 넓이 광선 길이 (잠정 — 실물 방 7 × 8 m, 라이다 8 m)
  int rays = 32;              // 광선 수 (64 의 약수)
  float coarse_m = 0.5f;      // m — 예상 넓이를 이 크기 굵은 칸 수로: 아는 칸 몫이 known_frac 미만인 굵은 칸이 그 자리에서 보면 넘어가는 수 (사용자 목표: 공간을 "대충" 빨리 앎 —
                              //   이미 대충 아는 굵은 칸 안 작은 모름 구멍은 값 0 → 그런 덩어리는 쉼). 0 = 칸 단위 모름 넓이 (옛 방식 — 견줌용으로 남김)
  float known_frac = 0.5f;    // 굵은 칸을 "대충 앎" 으로 보는 아는 칸 몫 (잠정)
  float v_eff = 0.981f, w_eff = 3.232f, t0 = 0.10f;   // m/s · rad/s · s — 가는 시간 = 길 / v_eff + 첫 회전 / w_eff (fuel 과 같은 잰 값 — 실물 주행 맞춤)
  double dir_m = 0.5;         // m — 첫 회전은 로봇에서 이 거리 길 칸 방향으로 잼 (잠정)
  // 6. 번호 잇기
  float match_m = 1.0f;       // m — 지난 후보와 같은 덩어리로 보는 거리 (대표 칸 사이, 잠정)
  float keep = 0.8f;          // 지난 시점 점수가 새 최선의 이 배 이상이면 그 칸 그대로 (잠정)
  // 7. 기억 (방 수)
  float door_m = 0.45f;       // m — 막힘까지 이 거리 미만 칸에서 방을 끊음 (fuel_upper.h Opt::door_m)
  // 뺀 잣대 (재 보고 나빠서 끔 — README "후보 뽑기"): 시점 둘레 모름 간격 0.21 m (FUEL), 광선 모름 깊이 1 m, 길 위 모름 간격, 빔 사이 가는 모름 줄 빼기
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
  int n_gaveup = 0;       // 그중 버림: 시점 곁 실패가 Opt::give_up 번 이상 (구멍과 따로 셈)
  int rotate = 0;         // 후보 0 개인데 쉬는 덩어리가 모두 로봇 곁 → 제자리 회전으로 봄
  int reused = 0;         // 프론티어 찾기를 같은 걸음 관측 것으로 (In::reach)
  // 이 결정의 갈 수 있는 칸 (프론티어 찾기 BFS — 계획 몸통 반경·깊이 층 그대로): 칸 i 는 (reach_flag[i] & 1) && reach_dist[i] >= 0.
  // 다음 build 까지만 유효 (Ctl 버퍼 또는 RobotEnv::reach_keep). 물체 후보 설 자리가 같은 판정을 씀 (low_decider object_spots)
  const uint8_t* reach_flag = nullptr;
  const int32_t* reach_dist = nullptr;
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
  std::vector<int32_t> dto;          // 막힘까지 3·4 모따기 거리 (칸 × 3)
  std::vector<int32_t> room;         // 방 번호
  std::vector<uint32_t> stamp;       // 덮음 표시 (번호가 cov_id 이면 덮임) · 한 후보 안 겹침 막기
  uint32_t cov_id = 1, seen_id = 1;
  std::vector<uint32_t> seen;
  std::vector<int32_t> bknown, btotal;   // 굵은 칸마다 아는 칸 · 맵 안 칸 수
  std::vector<double> bnew, bcov;        // 굵은 칸마다 이 시점이 새로 볼 칸 (어림) · 고른 후보들이 덮은 칸 (어림)
  std::vector<int32_t> btouch;
  std::vector<int32_t> lab_n, lab_at;    // 덩어리 묶기 (라벨별 칸 수 · 노드 번호)
  void fit(int n) {
    fw.fit(n);
    if ((int)dto.size() < n) { dto.assign(n, 0); room.assign(n, -1); stamp.assign(n, 0); seen.assign(n, 0); }
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
  const sim::ReachKeep* reach = nullptr;   // 같은 걸음 관측의 프론티어 찾기 (RobotEnv::reach_keep — 입력이 같을 때만 씀, 없으면 직접 찾음)
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

// 한 결정의 공용 값 · 단계 함수 (build 가 차례로 부름). 버퍼는 Work, 기억은 Mem, 결과는 Out
struct Run {
  const In& in;
  const Opt& o;
  Mem& mem;
  Work& w;
  Out& out;
  const sim::EnvCfg& c;
  const sim::World& W;
  const sim::EnvState& s;
  const int gw, gh, n;
  const mdp::MapView mv;
  const double res, rxf, ryf, th;   // 로봇 칸 자리 (실수) · 머리 방향
  fu::Work& fw;
  mdp::SearchBuf sb;
  // 프론티어 찾기 결과 (fw 의 버퍼 또는 같은 걸음 관측이 남긴 것 — In::reach): 번진 칸 표시 · 번진 차례 · BFS 거리 · 덩어리 라벨 · 덩어리
  const uint8_t* FL;
  const std::vector<int32_t>* QA;
  const int32_t *DI, *LB;
  const mdp::Frontier* OUT;
  int reused = 0;
  const int ws;                      // 자취 판 한 줄 낱말 수
  const int Rg;                      // 광선 길이 (칸)
  const int B, bw, bh;               // 굵은 칸 크기 (칸, 0 = 끔) · 굵은 격자 크기
  const int step;                    // 광선 건너뛰기 (64 / rays)
  const double ray_k, area_k, dmin;  // 광선 무게 → 칸 수 · 이득 → m² · 로봇 곁 반경 (m)
  int nr0 = 0, start = -1;           // 번진 칸 수 · 시작 칸
  int bx0 = 0, by0 = 0, bx1 = -1, by1 = -1;   // 번진 칸 둘레 상자 (+ 광선 길이)
  int rroom = -1;                    // 로봇 방
  std::chrono::steady_clock::time_point t_start, t_mark;

  Run(const In& in_, const Opt& o_, Mem& m_, Work& w_, Out& out_)
      : in(in_), o(o_), mem(m_), w(w_), out(out_), c(*in_.c), W(*in_.W), s(*in_.s), gw(W.gw), gh(W.gh), n(gw * gh), mv{in_.L, c.size, gw, gh}, res(c.res),
        rxf((s.x + s.ex - W.ox) * c.inv_res), ryf((s.y + s.ey - W.oy) * c.inv_res), th(s.th + s.eth), fw(w_.fw),
        sb{fw.flag.data(), fw.qa.data(), fw.qb.data(), fw.dist.data(), fw.label.data()}, FL(fw.flag.data()), QA(&fw.qa), DI(fw.dist.data()), LB(fw.label.data()), OUT(fw.out.data()), ws((c.size + 31) / 32),
        Rg(std::max(1, (int)std::lround(o_.gain_m * c.inv_res))),
        B(o_.coarse_m > 0.f ? std::max(1, (int)std::lround(o_.coarse_m * c.inv_res)) : 0), bw(B ? (gw + B - 1) / B : 0), bh(B ? (gh + B - 1) / B : 0),
        step(std::max(1, mdp::hier::kGainRays / std::max(1, o_.rays))), ray_k((double)step * 2.0 * M_PI / mdp::hier::kGainRays),
        area_k(B ? (double)B * B * res * res : (double)step * 2.0 * M_PI / mdp::hier::kGainRays * res * res),
        dmin(std::max((double)o_.vp_min, (double)c.low_pass_r)), t_start(std::chrono::steady_clock::now()), t_mark(t_start) {}

  void lap(int k) { const auto now = std::chrono::steady_clock::now(); out.ms_part[k] += std::chrono::duration<double, std::milli>(now - t_mark).count(); t_mark = now; }
  void done() { out.ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_start).count(); }

  // ---- 바탕 (여러 단계가 같이 씀)
  bool pass(int idx) const { return (FL[idx] & 1) && DI[idx] >= 0; }   // 로봇에서 BFS 로 번진 갈 수 있는 칸
  double dto_m(int idx) const { return w.dto[idx] * res / 3.0; }                 // 막힘까지 (m)
  double len_m(int idx) const { return fw.cd[idx] >= 0 ? fw.cd[idx] * 0.2 * res : 1e6; }   // 로봇에서 길 (모따기, m)
  // 시점에서 로봇 쪽으로 모따기 거리를 거슬러 감 (visit 가 거짓이면 멈춤)
  template <class Visit>
  void descend(int idx, const Visit& visit) const {
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
  }
  // 로봇에서 dir_m 떨어진 길 칸 방향 (첫 회전)
  double first_dir(int idx) const {
    const int stop = (int)(o.dir_m / res) * 5;
    int at = idx;
    descend(idx, [&](int a) { at = a; return fw.cd[a] > stop; });
    return std::atan2(at / gw + 0.5 - ryf, at % gw + 0.5 - rxf);
  }
  double t_of(int idx) const { return len_m(idx) / o.v_eff + fu::detail::wrap(first_dir(idx) - th) / o.w_eff; }   // 가는 시간 (s)
  // 두 칸 사이 막힘 없음 (곧은 줄)
  bool los(int a, int b) const {
    const int x0 = a % gw, y0 = a / gw, x1 = b % gw, y1 = b / gw, N = std::max(std::abs(x1 - x0), std::abs(y1 - y0));
    for (int t = 1; t < N; ++t) {
      const int x = x0 + (int)std::lround((double)(x1 - x0) * t / N), y = y0 + (int)std::lround((double)(y1 - y0) * t / N);
      if (mdp::cls_at(mv, x, y) == mdp::kOcc) return false;
    }
    return true;
  }
  // 좁은 칸의 방: 둘레(차례 고정) 가장 가까운 방 칸
  int room_of(int idx) const {
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
  }

  // ---- 1. 프론티어 칸 — 로봇에서 BFS 로 갈 수 있는 칸을 번지며 프론티어 칸에 라벨. 거짓 = 로봇이 설 칸이 없음
  bool frontier() {
    const int nf = c.size * ws;
    sb.vis = in.trail;
    sb.look = in.trail + nf;
    sb.tws = ws;
    sb.close_min = c.close_min;
    sb.rbc = mdp::robot_cells(c.close_radius, c.res);
    sb.cam = in.cam;
    sb.pre = fw.pre.data();
    const int rx = (int)std::floor(rxf), ry = (int)std::floor(ryf);
    // 같은 걸음 관측이 같은 입력(지도 · 지나간 칸 판 · 깊이 층 · 시작 칸 · 걸음, 같은 설정 c 의 몸통 반경·닫힘 문턱)으로 이미 찾았으면 그 결과
    // (obs_ref.h reach_obs · RobotEnv::reach_keep — 실물 노드에서 결정마다 한 번 덜 찾음, 결과는 같은 함수 · 같은 입력이라 같음)
    const sim::ReachKeep* k = in.reach;
    mdp::SearchResult r;
    if (k && k->valid && k->step == s.step && k->L == in.L && k->trail == in.trail && k->cam == in.cam && k->rx == rx && k->ry == ry) {
      r = k->r;
      FL = k->flag.data(); QA = &k->qa; DI = k->dist.data(); LB = k->label.data(); OUT = k->out.data();
      n_out = std::min(r.n_found, (int)k->out.size());
      reused = 1;
    } else {
      r = mdp::frontier_search(mv, rx, ry, mdp::obsgen::reach_robot(c), 1, sb, fw.out.data(), (int)fw.out.size());
      n_out = std::min(r.n_found, (int)fw.out.size());
    }
    out.start_free = r.start_free;
    out.reused = reused;
    out.reach_flag = FL;
    out.reach_dist = DI;
    nr0 = r.n_reach0;
    start = r.start;
    return r.start_free;
  }
  int n_out = 0;   // 프론티어 라벨 수

  // ---- 1'. 거리 지도 · 방 — 번진 칸 둘레 상자 안만 (FALCON: 바뀐 곳만). 4 거르기의 간격 · 7 의 같은 방 수에 씀
  void dist_maps() {
    bx0 = gw; by0 = gh; bx1 = -1; by1 = -1;
    for (int q = 0; q < nr0; ++q) {
      const int idx = (*QA)[q], x = idx % gw, y = idx / gw;
      bx0 = std::min(bx0, x); bx1 = std::max(bx1, x); by0 = std::min(by0, y); by1 = std::max(by1, y);
    }
    bx0 = std::max(0, bx0 - Rg); by0 = std::max(0, by0 - Rg); bx1 = std::min(gw - 1, bx1 + Rg); by1 = std::min(gh - 1, by1 + Rg);
    // 막힘(지도 막힘 + 깊이 층)까지 3·4 모따기 거리
    chamfer34(gw, bx0, by0, bx1, by1, [&](int x, int y) { return mdp::cls_at(mv, x, y) == mdp::kOcc || (in.cam && in.cam[y * c.size + x]); }, w.dto);
    // 방: 막힘에서 door_m 이상인 갈 수 있는 칸을 4 이웃으로 (좁은 칸은 room_of)
    const int32_t door3 = (int32_t)std::lround(o.door_m * c.inv_res * 3.0);
    for (int q = 0; q < nr0; ++q) w.room[(*QA)[q]] = -1;
    int nroom = 0;
    for (int q = 0; q < nr0; ++q) {
      const int s0 = (*QA)[q];
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
    rroom = -1;
    for (int q = 0; q < nr0 && rroom < 0; ++q) if (w.room[(*QA)[q]] >= 0) rroom = w.room[(*QA)[q]];
  }

  // ---- 2. 덩어리 — 자격 프론티어 칸(BFS 거리 · 덩어리 크기 · 아직 안 본 칸)을 라벨별로 묶고 합침·나눔 (fuel 전역 노드와 같은 규칙). 로봇에서 가까운 차례
  std::vector<fu::Node> clusters() const {
    const auto eligible = [&](int idx) {
      if (!(FL[idx] & 2) || DI[idx] < mdp::hier::kCandMinDist) return false;
      const int l = LB[idx];
      if (l < 0 || l >= n_out || OUT[l].n < c.close_min || OUT[l].n < mdp::hier::kCandMinCells) return false;
      return !mdp::looked_cell(mv, sb.look, sb.vis, ws, idx % gw, idx / gw);
    };
    std::vector<fu::Node> nodes;
    {   // 라벨별로 (번진 차례 그대로) — 두 번 훑어 세고 나눔
      std::vector<int32_t>& cnt = w.lab_n;
      cnt.assign(n_out, 0);
      int ne = 0;
      for (int q = 0; q < nr0; ++q) { const int idx = (*QA)[q]; if (eligible(idx)) { fw.qb[ne++] = idx; cnt[LB[idx]] += 1; } }
      std::vector<int32_t>& at = w.lab_at;
      at.assign(n_out, -1);
      for (int l = 0; l < n_out; ++l) if (cnt[l] > 0) { at[l] = (int)nodes.size(); nodes.emplace_back(); nodes.back().cells.reserve(cnt[l]); }
      for (int e = 0; e < ne; ++e) nodes[at[LB[fw.qb[e]]]].cells.push_back(fw.qb[e]);
    }
    const auto fill = [&](fu::Node& nd) {   // 가운데 · 대표 칸 (BFS 가장 가까운 칸)
      double sx = 0, sy = 0;
      int bd = 1 << 30, ba = -1;
      for (int idx : nd.cells) {
        sx += idx % gw + 0.5; sy += idx / gw + 0.5;
        const int d = DI[idx] >= 0 ? DI[idx] : (1 << 29);
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
    return nodes;
  }

  // ---- 5 바탕: 로봇에서 모따기 길 (가는 시간) · 굵은 칸마다 아는 칸 수 (예상 넓이)
  void robot_dist() {
    fu::detail::chamfer(start, gw, gh, [&](int idx) { return pass(idx); }, *QA, nr0, fw.cd, fw.bk);   // 번진 칸 = qa[0 … nr0) (chamfer 는 읽기만)
  }
  void coarse_stats() {
    if (!B) return;
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

  // ---- 5. 예상 새 넓이 — vp 에서 광선(gain_m)이 닿는 모름. 굵은 칸이면 "대충 앎" 으로 넘어가는 굵은 칸 수, 아니면 모름 칸 × t 무게 (cand_gain_ray 와 같은 무게).
  //   marginal: 앞서 고른 후보들이 덮은 몫을 빼고 (TARE), 같은 후보 안 겹침은 한 번만
  long gain(int vp, bool marginal) {
    long g = 0;
    ++w.seen_id;
    if (B) {
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
    for (int a = 0; a < mdp::hier::kGainRays; a += step)
      mdp::gain_ray_walk(mv, vp % gw, vp / gw, a, Rg, [&](int x, int y, int t, uint8_t cl) {
        if (cl != mdp::kUnk) return;
        const int j = y * gw + x;
        if (marginal && (w.stamp[j] == w.cov_id || w.seen[j] == w.seen_id)) return;
        w.seen[j] = w.seen_id;
        g += t;
      });
    return g;
  }
  // 고른 후보의 광선이 지난 모름 칸을 덮었다고 표시 (다음 marginal 이 뺌)
  void mark_covered(int vp) {
    for (int a = 0; a < mdp::hier::kGainRays; a += step)
      mdp::gain_ray_walk(mv, vp % gw, vp / gw, a, Rg, [&](int x, int y, int t, uint8_t cl) {
        if (cl != mdp::kUnk) return;
        const int j = y * gw + x;
        if (B) {   // 굵은 칸: 덮은 몫을 그 굵은 칸에 더함 (칸마다 한 번)
          if (w.stamp[j] == w.cov_id) return;
          w.stamp[j] = w.cov_id;
          w.bcov[(y / B) * bw + x / B] += t * ray_k;
          return;
        }
        w.stamp[j] = w.cov_id;
      });
  }

  // ---- 4. 거르기 — 갈 수 있음(BFS 번진 칸) · 피하는 칸 아님 · 로봇 곁(몸 · 뒤 사각) 아님 · 막힘에서 clear_need 밖 · 덩어리 대표 칸이 보임
  bool vp_ok(int cell, int anc, double clear_need) const {
    if (cell < 0 || cell >= n || !pass(cell) || in.avoided(cell) || DI[cell] < mdp::hier::kCandMinDist) return false;
    if (std::hypot(cell % gw + 0.5 - rxf, cell / gw + 0.5 - ryf) * res < dmin) return false;
    if (dto_m(cell) < clear_need) return false;
    if (cell != anc && !los(cell, anc)) return false;
    return true;
  }

  // 덩어리 하나의 살아 있는 후보 (6 · 7 단계가 씀)
  struct Live { int node, vp, id, age, kept; long g; double t; };

  // 구멍: 덩어리 칸과 그 4 이웃이 든 굵은 칸이 모두 이미 대충 앎 → 값 0 (대충 아는 곳 안의 작은 모름은 탐사를 붙들지 않음)
  bool pocket(const fu::Node& nd) const {
    if (!B) return false;
    for (size_t q = 0; q < nd.cells.size(); ++q) {
      const int x0 = nd.cells[q] % gw, y0 = nd.cells[q] / gw;
      for (int k = -1; k < 4; ++k) {
        int dx = 0, dy = 0;
        if (k >= 0) mdp::nb4(k, &dx, &dy);
        const int u = x0 + dx, v = y0 + dy;
        if (u < 0 || v < 0 || u >= gw || v >= gh) continue;
        const int b = (v / B) * bw + u / B;
        if (w.bknown[b] < o.known_frac * w.btotal[b]) return false;
      }
    }
    return true;
  }
  // ---- 3 → 4 → 5. 덩어리마다 시점 뽑기 · 거르기 · 점수 → 가장 좋은 시점 (L.vp · g · t). 거짓 = 쉼 (구멍이거나 설 자리 없음)
  bool best_view(int k, const fu::Node& nd, Live& L) {
    L = Live{k, -1, -1, 0, 0, 0, 0};
    double bs = -1;
    for (int tier = 0; tier < 2 && L.vp < 0; ++tier) {   // 간격 margin → 없으면 margin_min
      const double clear_need = c.close_radius + (tier == 0 ? o.margin : o.margin_min);
      const auto test = [&](int cell) {
        if (!vp_ok(cell, nd.anc, clear_need)) return;   // 4
        const long g = gain(cell, false);               // 5
        if (g <= 0) return;
        const double t = t_of(cell), sc = g / (t + o.t0);
        if (sc > bs) { bs = sc; L.vp = cell; L.g = g; L.t = t; }
      };
      test(nd.anc);   // 3: 대표 칸 + 고리
      for (int ri = 0; ri < 3; ++ri)
        for (int a = 0; a < o.ring_n; ++a) {
          const double rr = o.ring_m[ri], ang = 2 * M_PI * (a + 0.5 * ri) / o.ring_n;
          const int x = (int)std::floor(nd.cx + rr / res * std::cos(ang)), y = (int)std::floor(nd.cy + rr / res * std::sin(ang));
          if (x >= 0 && y >= 0 && x < gw && y < gh) test(y * gw + x);
        }
    }
    return L.vp >= 0;
  }
  // 버림: 시점 L.vp 에서 fail_m 안 실패 자리가 give_up 개 이상 (describe 의 fail_near 와 같은 셈)
  bool gave_up(int vp) const {
    if (o.give_up <= 0) return false;
    const double wx = W.ox + (vp % gw + 0.5) * res, wy = W.oy + (vp / gw + 0.5) * res;
    int k = 0;
    for (const Mem::Fail& f : mem.fails)
      if (std::hypot(f.x - wx, f.y - wy) < in.fail_m && ++k >= o.give_up) return true;
    return false;
  }
  // 쉬는 덩어리 (그림 · 회전 판정용 수)
  void rest(const fu::Node& nd, bool is_pocket) {
    out.n_dormant += 1;
    out.dormant.push_back(nd.anc);
    out.n_pocket += is_pocket;
    if (std::hypot(nd.anc % gw + 0.5 - rxf, nd.anc / gw + 0.5 - ryf) * res < dmin) out.n_near += 1;
  }

  // ---- 6. 번호 잇기 — 지난 결정의 덩어리 중 대표 칸이 match_m 안인 가장 가까운 것 (쓰지 않은 것만) → 같은 번호.
  //   지난 시점이 아직 자격 있고 점수가 새 최선의 keep 배 이상이면 그 칸 그대로 (목표가 덜 흔들림)
  void match(const fu::Node& nd, Live& L, std::vector<uint8_t>& used_prev) {
    const int Rm = (int)std::lround(o.match_m * c.inv_res);
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
    if (pi < 0) { L.id = mem.next_id++; return; }
    used_prev[pi] = 1;
    const Mem::Prev& p = mem.prev[pi];
    L.id = p.id;
    L.age = p.age + 1;
    if (p.vp == L.vp) { L.kept = 1; return; }
    if (!vp_ok(p.vp, nd.anc, c.close_radius + o.margin_min)) return;
    const double bs = L.g / (L.t + o.t0);
    const long g = gain(p.vp, false);
    const double t = t_of(p.vp);
    if (g > 0 && g / (t + o.t0) >= o.keep * bs) { L.vp = p.vp; L.g = g; L.t = t; L.kept = 1; }
  }

  // ---- 7a. 고르기 — 남은 것 중 "아직 안 덮인" 넓이 / 시간이 가장 큰 것을 차례로 (같으면 작은 번호). 고른 것의 광선이 지난 모름은 덮였다고 표시 (TARE)
  struct Pick { int i; long m; double sc; };
  std::vector<Pick> select(const std::vector<Live>& live) {
    ++w.cov_id;
    std::vector<uint8_t> taken(live.size(), 0);
    std::vector<Pick> order;
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
      order.push_back(Pick{bi, bm, bs});
      mark_covered(live[bi].vp);
    }
    return order;
  }

  // ---- 7b. 판단 모델 값 · 기억 (번호마다 보낸 수 · 실패 수, 시점 fail_m 안 실패 자리, 지나간 자리)
  Cand describe(const Live& L, const fu::Node& nd, const Pick& p) const {
    Cand cd;
    cd.id = L.id; cd.age = L.age; cd.vp = L.vp; cd.anc = nd.anc; cd.kept = L.kept;
    cd.len_m = (float)len_m(L.vp);
    cd.turn_rad = (float)fu::detail::wrap(first_dir(L.vp) - th);
    cd.t_s = (float)L.t;
    const double b = std::atan2(L.vp / gw + 0.5 - ryf, L.vp % gw + 0.5 - rxf) - th;
    cd.bearing = (float)std::atan2(std::sin(b), std::cos(b));
    cd.gain_m2 = (float)(L.g * area_k);
    cd.marg_m2 = (float)(p.m * area_k);
    cd.score = (float)p.sc;
    cd.open_m = (float)(nd.cells.size() * res);
    cd.clear_m = (float)dto_m(L.vp);
    double pc = cd.clear_m;
    descend(L.vp, [&](int a) { pc = std::min(pc, dto_m(a)); return true; });
    cd.path_clear_m = (float)pc;
    const int rm = room_of(L.vp);
    cd.same_room = rroom >= 0 && rm == rroom;
    cd.cur = in.cur_target >= 0 && (in.cur_target == L.vp || std::find(nd.cells.begin(), nd.cells.end(), in.cur_target) != nd.cells.end());
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
    return cd;
  }
  // 같은 방의 다른 후보 수 · 최근 실패한 후보는 목록 맨 뒤로 (차례는 유지 — 판단 모델은 실패 값과 함께 보지만 0 번·규칙으로는 안 감)
  void finish_list() {
    for (Cand& a : out.c) {
      const int ra = room_of(a.vp);
      for (const Cand& b : out.c) a.room_left += &a != &b && ra >= 0 && room_of(b.vp) == ra;
    }
    if (std::none_of(out.c.begin(), out.c.end(), [](const Cand& a) { return a.recent_fail != 0; })) return;   // 대개 — 차례 그대로
    std::vector<size_t> ord;
    for (size_t i = 0; i < out.c.size(); ++i) if (!out.c[i].recent_fail) ord.push_back(i);
    for (size_t i = 0; i < out.c.size(); ++i) if (out.c[i].recent_fail) ord.push_back(i);
    std::vector<Cand> c2;
    std::vector<std::vector<int32_t>> cl2;
    for (size_t i : ord) { c2.push_back(out.c[i]); cl2.push_back(std::move(out.cells[i])); }
    out.c.swap(c2);
    out.cells.swap(cl2);
  }
};
}  // namespace detail

// 한 결정: 로봇 지도 → ≤ K 후보 (위 일곱 단계). 같은 지도 · 같은 기억이면 같은 결과
inline Out build(const In& in, const Opt& o, Mem& mem, Work& w) {
  Out out;
  w.fit(in.W->gw * in.W->gh);
  detail::Run R(in, o, mem, w, out);
  // 1. 프론티어 칸
  if (!R.frontier()) { R.lap(0); mem.prev.clear(); R.done(); return out; }
  R.lap(0);
  R.dist_maps();
  R.lap(1);
  // 2. 덩어리
  std::vector<fu::Node> nodes = R.clusters();
  out.n_clu = (int)nodes.size();
  R.lap(2);
  R.robot_dist();
  R.lap(3);
  R.coarse_stats();
  // 3 · 4 · 5. 덩어리마다 시점 뽑기 → 거르기 → 점수 (없으면 쉼)   6. 번호 잇기
  std::vector<detail::Run::Live> live;
  std::vector<uint8_t> used_prev(mem.prev.size(), 0);
  for (int k = 0; k < (int)nodes.size(); ++k) {
    detail::Run::Live L;
    const bool is_pocket = R.pocket(nodes[k]);
    if (is_pocket || !R.best_view(k, nodes[k], L)) { R.rest(nodes[k], is_pocket); continue; }
    if (R.gave_up(L.vp)) { R.rest(nodes[k], false); out.n_gaveup += 1; continue; }   // 같은 자리 실패 give_up 번 — 버림
    R.match(nodes[k], L, used_prev);
    live.push_back(L);
  }
  R.lap(4);
  if (live.empty() && out.n_dormant > 0 && out.n_near == out.n_dormant) out.rotate = 1;   // 쉬는 덩어리가 모두 로봇 곁 → 제자리 회전
  // 7. 고르기 · 기억
  for (const detail::Run::Pick& p : R.select(live)) {
    const detail::Run::Live& L = live[p.i];
    out.c.push_back(R.describe(L, nodes[L.node], p));
    out.cells.push_back(std::move(nodes[L.node].cells));   // 이 뒤로 nodes 칸은 안 씀 (기억은 anc 만)
  }
  R.finish_list();
  R.lap(5);
  mem.prev.clear();   // 다음 결정의 6 번호 잇기용: 쉬지 않은 모든 덩어리 (고르지 않은 것도 — 번호가 상한 밖에서도 이어지게)
  for (const detail::Run::Live& L : live) mem.prev.push_back(Mem::Prev{L.id, nodes[L.node].anc, L.vp, L.age, (float)(L.g / (L.t + o.t0))});
  R.done();
  return out;
}

}  // namespace cand
