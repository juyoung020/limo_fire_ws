// 프론티어 탐사 기준선(호스트): explore_lite 의 프론티어 고르기 + Nav2 식 경로(팽창 + Smac 2D 비용의 A*) + RPP 추종 → 속도 행동 (v, ω).
// 시뮬은 env_api.h 로만 쓴다: 정책처럼 지도(get_map)와 지도 자세(참값 + 오차)만 본다. 참값 자세·정답 맵은 안 본다.
//
// 한 걸음(0.1 s, 시뮬 속도 행동 한 번)마다 Explorer::step:
//   1. 시작: 아는 빈 칸이 생길 때까지 제자리 회전 (frontier.yaml start)
//   2. explore_lite makePlan (planner_frequency 마다, 목표 도착·실패 때는 바로): mdp::frontier_search → 비용 정렬 → 블랙리스트 아닌 첫 덩어리의 중심.
//      진척 없음 progress_timeout → 블랙리스트. 남은 것이 없으면 끝 선언
//   3. Nav2: 목표가 바뀌거나 replan_period 마다, 또는 앞 경로가 막히면 A* 다시. 실패 = ABORTED → 블랙리스트 + 바로 다시 고름
//   4. RPP(시뮬 goal 모드 추종기 physics_step 과 같은 식) → velocity_smoother 가속 한도 → (v, ω)
#pragma once
#include <cstdint>
#include <string>
#include <vector>

#include "frontier.h"

namespace fb {

struct Params {
  // Nav2 RPP · velocity_smoother · goal checker (sim::EnvCfg 와 같은 값, nav2_params.yaml)
  float v_des, v_max, w_max, a_max, alpha_max;
  float lookahead, min_lookahead, rot_w, rot_min_angle, approach_dist, approach_min_v, goal_tol;
  // Nav2 global costmap 팽창 · Smac 2D (nav2_params.yaml)
  float robot_radius, inflation_radius, cost_scaling, planner_tolerance, travel_mult;
  bool allow_unknown;
  float obstacle_range;   // global_costmap obstacle_layer scan.obstacle_max_range
  // RPP 충돌 검사·Nav2 BT 복구
  float fp_half_len, fp_half_wid;   // local_costmap footprint + footprint_padding (반 길이·반 폭)
  float collision_time;             // FollowPath max_allowed_time_to_collision_up_to_carrot
  int bt_retries;                   // frontier.yaml nav2 (기본 BT 값)
  float spin_dist, wait_duration, backup_dist, backup_speed, behavior_time_allowance;
  float replan_period, inscribed_escape_cost;
  // explore_lite (frontier.yaml)
  float planner_period, progress_timeout, potential_scale, gain_scale, min_frontier_size;
  int blacklist_tol;
  float wait_timeout;
  float t_safe;
  // 지도·제어
  float res, ctrl_dt;
  std::string mode;
};

// frontier.yaml + env.yaml(→ nav2_params.yaml). mode: frontier.yaml modes 의 이름(nav2 | fast)
Params load_params(const std::string& frontier_yaml, const std::string& env_yaml, const std::string& mode);

enum class Phase { Wait, Go, Done };

struct Action {
  float v = 0, w = 0;
  bool declare = false;       // 이번 걸음에 끝 선언
};

struct Stats {
  int n_plans = 0;            // makePlan 수
  int n_goals = 0;            // 새 목표 보낸 수
  int n_reached = 0;          // 목표 도착
  int n_aborted = 0;          // 경로 못 짬 → 블랙리스트
  int n_timeout = 0;          // 진척 없음 → 블랙리스트
  int n_astar = 0;            // A* 부른 수
  int n_blocked_replan = 0;   // 앞 경로가 막혀 다시 짬
  int n_frontiers = 0;        // 마지막 찾기의 덩어리 수(크기 문턱 넘은 것)
  int n_frontier_small = 0;   // 마지막 찾기에서 작아서 버린 수
  int n_collision_stops = 0;  // RPP 충돌 검사로 멈춤
  int n_recoveries = 0;       // Nav2 복구 동작 수
};

class Explorer {
 public:
  // 맵 격자 gw × gh, 원점 (ox, oy). 지도 L 의 한 줄 길이는 stride (시뮬 cfg.size)
  Explorer(const Params& p, int gw, int gh, int stride, float ox, float oy);

  // L: 이 환경의 지도(stride² 칸). (px, py, pth): 지도 자세. t: 판 시작부터 시각(s)
  Action step(const int16_t* L, float px, float py, float pth, double t);

  Phase phase() const { return phase_; }
  const std::string& end_reason() const { return reason_; }
  const Stats& stats() const { return st_; }
  bool has_goal() const { return has_goal_; }
  float goal_x() const { return gx_; }
  float goal_y() const { return gy_; }
  const std::vector<float>& path() const { return path_; }   // x0 y0 x1 y1 …
  size_t blacklist_size() const { return blacklist_.size(); }

 private:
  Params p_;
  int w_, h_, stride_;
  float ox_, oy_;
  Phase phase_ = Phase::Wait;
  std::string reason_;
  Stats st_;
  double t_wait0_ = -1;
  // explore_lite
  double next_plan_t_ = -1;
  bool has_goal_ = false, plan_now_ = false;
  float gx_ = 0, gy_ = 0;
  float prev_gx_ = 1e30f, prev_gy_ = 1e30f, prev_dist_ = 0;
  double last_progress_ = 0;
  std::vector<std::pair<float, float>> blacklist_;
  // Nav2
  std::vector<float> path_;
  std::vector<uint8_t> path_c_;   // 경로 점마다 짤 때 비용
  size_t path_i_ = 0;
  double next_replan_t_ = -1;
  float vs_ = 0, ws_ = 0;
  // 버퍼
  std::vector<uint8_t> flag_, cost_;
  std::vector<int32_t> qa_, qb_;
  std::vector<mdp::Frontier> fr_;
  struct Off { int dx, dy; uint8_t cost; };
  std::vector<Off> kernel_;
  int inscribed_cells_ = 0;
  std::vector<float> g_;
  std::vector<int32_t> parent_;
  std::vector<uint8_t> closed_;

  mdp::MapView view(const int16_t* L) const { return mdp::MapView{L, stride_, w_, h_}; }
  // 팽창의 씨앗 칸: 정적 층(지도 막힘 kOcc) + 장애물 층 흉내 — Nav2 global costmap 은 static_layer(/map) 위에 obstacle_layer(/scan 끝점을
  // obstacle_max_range 안에서 바로 LETHAL)를 얹는다. 시뮬 API 는 스캔을 안 내주므로, 지도에서 맞음이 지나감보다 많은 칸(l > 0, p > 0.5)을
  // 로봇에서 obstacle_range 안이면 끝점으로 본다. 4 번 맞아야 kOcc 가 되는 확률 지도보다 빨리 벽을 피한다
  bool lethal_src(const mdp::MapView& m, int x, int y) const {
    const int16_t l = m.L[y * m.stride + x];
    if (mdp::cell_class(l) == mdp::kOcc) return true;
    if (l == mdp::kUnknownL || l <= 0) return false;
    const float dx = (float)(x - rcx_) * p_.res, dy = (float)(y - rcy_) * p_.res;
    return dx * dx + dy * dy <= p_.obstacle_range * p_.obstacle_range;
  }
  int rcx_ = 0, rcy_ = 0;
  int cell(float x, float y, int* cx, int* cy) const;
  bool on_blacklist(float x, float y) const;
  // explore_lite makePlan. false = 끝 (reason_ 에 까닭)
  bool make_plan(const int16_t* L, float px, float py, double t);
  void build_costmap(const int16_t* L);
  bool astar(float px, float py, float gx, float gy);
  bool path_blocked(const int16_t* L) const;
  void follow(float px, float py, float pth, float* vc, float* wc, float* carrot_dist);
  bool footprint_hits(const int16_t* L, float x, float y, float th) const;
  bool collision_imminent(const int16_t* L, float px, float py, float pth, float v, float w, float carrot_dist) const;
  Action recover(const int16_t* L, float px, float py, float pth, double t);
  Action declare_end();
  // Nav2 BT 복구 상태
  enum class Rec { None, Clear, Spin, Wait, Backup };
  Rec rec_ = Rec::None;
  int fails_ = 0, rr_ = 0;
  double rec_end_t_ = 0;
  float rec_done_ = 0, rec_last_th_ = 0, rec_x0_ = 0, rec_y0_ = 0;
  Action smooth(float vc, float wc);
};

}  // namespace fb
