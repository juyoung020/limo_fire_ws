// 프론티어 탐사 노드 (limo_fire_rl/README "프론티어 탐사"): Cartographer /map 에서 프론티어를 찾아 Nav2 NavigateToPose 목표로 보낸다.
// 알고리즘은 새로 쓰지 않고 탐사 기준선과 같은 코드를 쓴다:
//   - 찾기·닫힘: train/explore/mdp/frontier.h frontier_search · closed (학습 시뮬·기준선·RL 노드 공용)
//   - 고르기: train/explore/baselines/frontier/include/explore_lite.h rank_frontiers · on_blacklist (기준선 Explorer::make_plan 과 같은 함수)
//   - 값: explore_lite 값은 기준선 설정(baselines/frontier/config/frontier.yaml), 로봇 반경은 Nav2 nav2_params.yaml global_costmap robot_radius
//   기준선의 경로 계획·추종(A*·RPP 흉내)은 여기서 Nav2 가 맡는다.
// 켜기·끄기: ~/enable (std_srvs/SetBool). 끄면 보낸 Nav2 목표·제자리 회전을 취소해 멈추고, 켜면 블랙리스트를 비우고 지금 지도에서 다시 고른다.
// 상태: ~/state (idle · running · done, transient local), ~/done (Bool), ~/status (사람이 읽는 한 줄), ~/frontiers (MarkerArray)
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <ament_index_cpp/get_package_share_directory.hpp>
#include <nav2_msgs/action/navigate_to_pose.hpp>
#include <nav2_msgs/action/spin.hpp>
#include <nav_msgs/msg/occupancy_grid.hpp>
#include <rclcpp/rclcpp.hpp>
#include <rclcpp_action/rclcpp_action.hpp>
#include <std_msgs/msg/bool.hpp>
#include <std_msgs/msg/string.hpp>
#include <std_srvs/srv/set_bool.hpp>
#include <tf2/LinearMath/Matrix3x3.h>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>
#include <visualization_msgs/msg/marker_array.hpp>
#include <yaml-cpp/yaml.h>

#include "explore_lite.h"
#include "frontier.h"

namespace limo_fire_rl {

using NavTo = nav2_msgs::action::NavigateToPose;
using Spin = nav2_msgs::action::Spin;

namespace {
YAML::Node need(const YAML::Node& n, const char* k, const std::string& where) {
  const YAML::Node v = n[k];
  if (!v) throw std::runtime_error(where + ": " + k + " 없음");
  return v;
}
}  // namespace

class ExploreFrontierNode : public rclcpp::Node {
 public:
  ExploreFrontierNode() : Node("explore_frontier") {
    const std::string fy = declare_parameter<std::string>("frontier_yaml", LIMO_FIRE_FRONTIER_YAML);
    std::string nav = declare_parameter<std::string>("nav2_params", "");
    if (nav.empty()) nav = ament_index_cpp::get_package_share_directory("limo_fire_navigation") + "/config/nav2_params.yaml";
    enabled_ = declare_parameter<bool>("enabled", false);
    map_topic_ = declare_parameter<std::string>("map_topic", "/map");
    map_frame_ = declare_parameter<std::string>("map_frame", "map");
    base_frame_ = declare_parameter<std::string>("base_frame", "base_link");
    occupied_thresh_ = declare_parameter<int>("occupied_thresh", 65);
    spin_angle_ = declare_parameter<double>("spin_angle", 1.57);
    map_timeout_ = declare_parameter<double>("map_timeout", 5.0);

    // explore_lite 값: 기준선과 같은 파일 (같은 값을 두 곳에 두지 않음)
    const YAML::Node f = YAML::LoadFile(fy);
    const YAML::Node el = need(f, "explore_lite", fy);
    planner_period_ = 1.0 / el["planner_frequency"].as<double>();
    progress_timeout_ = el["progress_timeout"].as<double>();
    potential_scale_ = el["potential_scale"].as<float>();
    gain_scale_ = el["gain_scale"].as<float>();
    min_frontier_size_ = el["min_frontier_size"].as<float>();
    blacklist_tol_ = el["blacklist_tolerance"].as<int>();
    wait_timeout_ = need(f, "start", fy)["wait_timeout"].as<double>();
    // 로봇 반경: Nav2 global costmap 과 같은 값 (설 수 있는 칸 = 막힘에서 이만큼 떨어진 칸)
    const YAML::Node np = YAML::LoadFile(nav);
    robot_radius_ = need(np["global_costmap"]["global_costmap"]["ros__parameters"], "robot_radius", nav).as<float>();
    RCLCPP_INFO(get_logger(), "explore_lite 값 %s (주기 %.1f s, 진척 %.0f s, potential %.1f, gain %.1f, 최소 %.2f m) · 로봇 반경 %.2f m (%s)", fy.c_str(),
                planner_period_, progress_timeout_, potential_scale_, gain_scale_, min_frontier_size_, robot_radius_, nav.c_str());

    tf_buf_ = std::make_shared<tf2_ros::Buffer>(get_clock());
    tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buf_);
    // Cartographer 는 /map 을 transient local 로 낸다. volatile 로 받으면 둘 다(map_server·Cartographer) 맞는다
    sub_map_ = create_subscription<nav_msgs::msg::OccupancyGrid>(map_topic_, rclcpp::QoS(1).reliable(),
                                                                 [this](nav_msgs::msg::OccupancyGrid::ConstSharedPtr m) { map_ = m; map_t_ = now(); });
    const auto latched = rclcpp::QoS(1).transient_local();
    pub_state_ = create_publisher<std_msgs::msg::String>("~/state", latched);
    pub_done_ = create_publisher<std_msgs::msg::Bool>("~/done", latched);
    pub_status_ = create_publisher<std_msgs::msg::String>("~/status", 10);
    pub_markers_ = create_publisher<visualization_msgs::msg::MarkerArray>("~/frontiers", 10);
    nav_ = rclcpp_action::create_client<NavTo>(this, "navigate_to_pose");
    spin_ = rclcpp_action::create_client<Spin>(this, "spin");
    srv_enable_ = create_service<std_srvs::srv::SetBool>(
        "~/enable", [this](const std::shared_ptr<std_srvs::srv::SetBool::Request> q, std::shared_ptr<std_srvs::srv::SetBool::Response> r) {
          if (q->data) start();
          else stop("끔 (~/enable false)", "idle");
          r->success = true;
          r->message = state_;
        });
    timer_ = create_wall_timer(std::chrono::milliseconds(200), [this] { tick(); });
    set_state(enabled_ ? "running" : "idle");
    if (enabled_) start();
  }

 private:
  // ---- 켜기·끄기
  void start() {
    blacklist_.clear();
    prev_gx_ = prev_gy_ = 1e30f;
    prev_dist_ = 0;
    t_on_ = now();
    last_progress_ = t_on_;
    t_wait0_ = rclcpp::Time(0, 0, RCL_ROS_TIME);
    started_ = false;
    plan_now_ = true;
    next_plan_ = t_on_;
    n_goals_ = n_reached_ = n_aborted_ = 0;
    enabled_ = true;
    set_state("running");
    publish_done(false);
    status("켬 — 프론티어 탐사 시작");
  }

  void stop(const std::string& why, const std::string& state) {
    enabled_ = false;
    cancel_goal();
    if (spinning_ && spin_->action_server_is_ready()) {
      if (spin_handle_) spin_->async_cancel_goal(spin_handle_);
    }
    spinning_ = false;
    spin_handle_.reset();
    set_state(state);
    status(why);
  }

  void finish(const std::string& reason) {
    const double el = (now() - t_on_).seconds();
    char b[256];
    std::snprintf(b, sizeof(b), "탐사 끝 (%s) — %.1f s, 목표 %d · 도착 %d · 포기 %d · 블랙리스트 %zu, 지도 %.1f m² 앎", reason.c_str(), el, n_goals_, n_reached_,
                  n_aborted_, blacklist_.size(), known_m2_);
    stop(b, "done");
    publish_done(true);
  }

  // ---- 주기 (0.2 s): explore_lite 타이머(planner_frequency) + 도착·실패 때 바로
  void tick() {
    if (!enabled_) return;
    const rclcpp::Time t = now();
    if (!plan_now_ && t < next_plan_) return;
    plan_now_ = false;
    next_plan_ = t + rclcpp::Duration::from_seconds(planner_period_);
    if (!map_ || (t - map_t_).seconds() > map_timeout_) {
      status(map_ ? "지도가 오래됨 — Cartographer 가 /map 을 내나?" : "지도 기다림 (" + map_topic_ + ")");
      return;
    }
    double px, py;
    if (!robot_pose(&px, &py)) return;
    make_plan(px, py, t);
  }

  bool robot_pose(double* x, double* y) {
    try {
      const auto T = tf_buf_->lookupTransform(map_frame_, base_frame_, tf2::TimePointZero);
      *x = T.transform.translation.x;
      *y = T.transform.translation.y;
      return true;
    } catch (const tf2::TransformException& e) {
      status(std::string("TF 없음 ") + map_frame_ + " → " + base_frame_ + ": " + e.what());
      return false;
    }
  }

  // /map (0~100 · −1) → frontier.h 칸 값: −1 = 모름, occupied_thresh 초과 = 막힘(p > 0.65, Cartographer ros_map.cc:44-45 와 같은 문턱), 나머지 = 갈 수 있음
  void load_map() {
    const auto& m = *map_;
    w_ = (int)m.info.width;
    h_ = (int)m.info.height;
    res_ = m.info.resolution;
    ox_ = (float)m.info.origin.position.x;
    oy_ = (float)m.info.origin.position.y;
    const size_t n = (size_t)w_ * h_;
    L_.resize(n);
    size_t known = 0;
    for (size_t i = 0; i < n; ++i) {
      const int8_t v = m.data[i];
      L_[i] = v < 0 ? mdp::kUnknownL : (v > occupied_thresh_ ? mdp::kOccL : (int16_t)0);
      known += v >= 0;
    }
    known_m2_ = (double)known * res_ * res_;
    if (flag_.size() < n) {
      flag_.resize(n);
      qa_.resize(n);
      qb_.resize(n);
    }
    fr_.resize(4096);
  }

  // explore_lite Explore::makePlan (explore.cpp:179-245) — 기준선 Explorer::make_plan 과 같은 차례. 목표 보내기만 Nav2 액션
  void make_plan(double px, double py, const rclcpp::Time& t) {
    load_map();
    const int rx = (int)std::floor((px - ox_) / res_), ry = (int)std::floor((py - oy_) / res_);
    const mdp::MapView view{L_.data(), w_, w_, h_};
    const mdp::Robot rb = mdp::robot_cells(robot_radius_, res_);
    const int min_cells = (int)std::ceil(min_frontier_size_ / res_ - 1e-4f);
    const mdp::SearchBuf sb{flag_.data(), qa_.data(), qb_.data()};
    for (int guard = 0; guard < 1000; ++guard) {
      mdp::SearchResult r = mdp::frontier_search(view, rx, ry, rb, min_cells, sb, fr_.data(), (int)fr_.size());
      // 시작: 설 수 있는 칸에서 번져 프론티어 칸이 보일 때까지 제자리 회전 (기준선 Phase::Wait, frontier.yaml start.wait_timeout)
      if (!started_) {
        if (!(r.start_free && r.n_cells > 0)) {
          if (t_wait0_.nanoseconds() == 0) t_wait0_ = t;
          if ((t - t_wait0_).seconds() > wait_timeout_) return finish("no_frontier_at_start");
          status("시작 — 설 수 있는 칸에서 프론티어가 보일 때까지 제자리 회전");
          send_spin();
          return;
        }
        started_ = true;
      }
      if (!r.start_free) return finish("no_standable_cell");
      if (mdp::closed(r)) return finish("closed");
      if (r.n_found == 0 && r.n_small > 0) r = mdp::frontier_search(view, rx, ry, rb, 1, sb, fr_.data(), (int)fr_.size());
      const int n = std::min<int>(r.n_found, (int)fr_.size());
      if (n == 0) return finish("closed");
      const std::vector<fb::Ranked> cs = fb::rank_frontiers(fr_.data(), n, res_, ox_, oy_, potential_scale_, gain_scale_);
      const float tol = (float)blacklist_tol_ * res_;
      auto it = std::find_if(cs.begin(), cs.end(), [&](const fb::Ranked& c) { return !fb::on_blacklist(blacklist_, c.cx, c.cy, tol); });
      publish_markers(cs, it == cs.end() ? -1 : (int)(it - cs.begin()), tol);
      if (it == cs.end()) return finish("all_blacklisted");
      const bool same = std::hypot(it->cx - prev_gx_, it->cy - prev_gy_) < 0.01f;   // explore.cpp:42-49, 213
      prev_gx_ = it->cx;
      prev_gy_ = it->cy;
      if (!same || prev_dist_ > it->dist) {
        last_progress_ = t;
        prev_dist_ = it->dist;
      }
      if ((t - last_progress_).seconds() > progress_timeout_) {   // explore.cpp:221-226
        blacklist_.push_back({it->cx, it->cy});
        RCLCPP_WARN(get_logger(), "(%.2f, %.2f) %.0f s 동안 진척 없음 — 블랙리스트", it->cx, it->cy, progress_timeout_);
        continue;
      }
      char b[200];
      std::snprintf(b, sizeof(b), "탐사 중 — 프론티어 %d (작은 것 %d), 목표 (%.2f, %.2f) 거리 %.2f m, 블랙리스트 %zu, 지도 %.1f m²", n, r.n_small, it->cx, it->cy,
                    it->dist, blacklist_.size(), known_m2_);
      status(b);
      if (same && goal_active_) return;   // explore.cpp:229-231 같은 목표면 다시 안 보냄
      send_goal(it->cx, it->cy, px, py);
      return;
    }
  }

  // ---- Nav2
  void send_goal(float gx, float gy, double rx, double ry) {
    if (!nav_->action_server_is_ready()) {
      status("Nav2 navigate_to_pose 서버 없음 — navigation.launch.xml localization:=false 를 켰나?");
      return;
    }
    cancel_spin();
    NavTo::Goal g;
    g.pose.header.frame_id = map_frame_;
    g.pose.header.stamp = now();
    g.pose.pose.position.x = gx;
    g.pose.pose.position.y = gy;
    tf2::Quaternion q;
    q.setRPY(0, 0, std::atan2(gy - ry, gx - rx));   // 도착 방향 = 가는 방향 (explore_lite 는 방향 없음 — 도착해 제자리 회전을 줄이려고)
    g.pose.pose.orientation.x = q.x();
    g.pose.pose.orientation.y = q.y();
    g.pose.pose.orientation.z = q.z();
    g.pose.pose.orientation.w = q.w();
    auto opt = rclcpp_action::Client<NavTo>::SendGoalOptions();
    const uint64_t id = ++goal_seq_;
    opt.goal_response_callback = [this, id](const rclcpp_action::ClientGoalHandle<NavTo>::SharedPtr& h) {
      if (!h) {
        RCLCPP_WARN(get_logger(), "Nav2 가 목표를 거절");
        if (id == goal_seq_) goal_active_ = false;
        return;
      }
      // 응답 전에 꺼졌거나 다른 목표로 바뀌었으면 바로 취소
      if (id != goal_seq_ || !enabled_) {
        nav_->async_cancel_goal(h);
        return;
      }
      goal_handle_ = h;
    };
    opt.result_callback = [this, id, gx, gy](const rclcpp_action::ClientGoalHandle<NavTo>::WrappedResult& r) {
      if (id != goal_seq_) return;
      goal_active_ = false;
      goal_handle_.reset();
      if (!enabled_) return;
      if (r.code == rclcpp_action::ResultCode::SUCCEEDED) {
        ++n_reached_;
        RCLCPP_INFO(get_logger(), "목표 (%.2f, %.2f) 도착", gx, gy);
      } else if (r.code == rclcpp_action::ResultCode::ABORTED) {   // explore.cpp:264-281 reachedGoal: 실패면 블랙리스트
        ++n_aborted_;
        blacklist_.push_back({gx, gy});
        RCLCPP_WARN(get_logger(), "Nav2 가 목표 (%.2f, %.2f) 포기 — 블랙리스트", gx, gy);
      } else {
        return;
      }
      plan_now_ = true;   // 도착·실패 때 바로 다시 고름
    };
    if (goal_handle_) nav_->async_cancel_goal(goal_handle_);
    goal_handle_.reset();
    goal_active_ = true;
    ++n_goals_;
    nav_->async_send_goal(g, opt);
    RCLCPP_INFO(get_logger(), "새 목표 (%.2f, %.2f)", gx, gy);
  }

  void cancel_goal() {
    ++goal_seq_;   // 응답이 아직 안 온 목표는 goal_response_callback 에서 취소
    if (goal_handle_) nav_->async_cancel_goal(goal_handle_);
    goal_handle_.reset();
    goal_active_ = false;
  }

  void send_spin() {
    if (spinning_) return;
    cancel_goal();
    if (!spin_->action_server_is_ready()) {
      status("시작 — Nav2 spin 서버 없음, 제자리에서 기다림");
      return;
    }
    Spin::Goal g;
    g.target_yaw = (float)spin_angle_;
    g.time_allowance.sec = 10;
    auto opt = rclcpp_action::Client<Spin>::SendGoalOptions();
    opt.goal_response_callback = [this](const rclcpp_action::ClientGoalHandle<Spin>::SharedPtr& h) {
      if (h && !enabled_) spin_->async_cancel_goal(h);
      else spin_handle_ = h;
    };
    opt.result_callback = [this](const rclcpp_action::ClientGoalHandle<Spin>::WrappedResult&) {
      spinning_ = false;
      spin_handle_.reset();
      if (enabled_) plan_now_ = true;
    };
    spinning_ = true;
    spin_->async_send_goal(g, opt);
  }

  void cancel_spin() {
    if (spinning_ && spin_handle_) spin_->async_cancel_goal(spin_handle_);
    spinning_ = false;
    spin_handle_.reset();
  }

  // ---- 알림
  void set_state(const std::string& s) {
    state_ = s;
    std_msgs::msg::String m;
    m.data = s;
    pub_state_->publish(m);
  }
  void publish_done(bool d) {
    std_msgs::msg::Bool m;
    m.data = d;
    pub_done_->publish(m);
  }
  void status(const std::string& s) {
    if (s != last_status_) RCLCPP_INFO(get_logger(), "%s", s.c_str());
    last_status_ = s;
    std_msgs::msg::String m;
    m.data = s;
    pub_status_->publish(m);
  }
  // 프론티어 덩어리 중심(파랑 = 후보, 회색 = 블랙리스트, 빨강 = 목표)
  void publish_markers(const std::vector<fb::Ranked>& cs, int chosen, float tol) {
    visualization_msgs::msg::MarkerArray a;
    visualization_msgs::msg::Marker clr;
    clr.action = visualization_msgs::msg::Marker::DELETEALL;
    a.markers.push_back(clr);
    int id = 0;
    for (size_t k = 0; k < cs.size(); ++k) {
      visualization_msgs::msg::Marker mk;
      mk.header.frame_id = map_frame_;
      mk.header.stamp = now();
      mk.ns = "frontier";
      mk.id = id++;
      mk.type = visualization_msgs::msg::Marker::SPHERE;
      mk.pose.position.x = cs[k].cx;
      mk.pose.position.y = cs[k].cy;
      mk.pose.position.z = 0.1;
      mk.pose.orientation.w = 1;
      const bool goal = (int)k == chosen, bl = fb::on_blacklist(blacklist_, cs[k].cx, cs[k].cy, tol);
      mk.scale.x = mk.scale.y = mk.scale.z = goal ? 0.25 : 0.15;
      mk.color.a = 0.9f;
      mk.color.r = goal ? 1.f : (bl ? 0.5f : 0.1f);
      mk.color.g = goal ? 0.1f : (bl ? 0.5f : 0.4f);
      mk.color.b = goal ? 0.1f : (bl ? 0.5f : 1.f);
      a.markers.push_back(mk);
    }
    pub_markers_->publish(a);
  }

  // 값
  double planner_period_ = 3, progress_timeout_ = 30, wait_timeout_ = 30, spin_angle_ = 1.57, map_timeout_ = 5;
  float potential_scale_ = 3, gain_scale_ = 1, min_frontier_size_ = 0.75f, robot_radius_ = 0.2f;
  int blacklist_tol_ = 5, occupied_thresh_ = 65;
  std::string map_topic_, map_frame_, base_frame_;
  // 상태
  bool enabled_ = false, plan_now_ = false, goal_active_ = false, spinning_ = false, started_ = false;
  std::string state_, last_status_;
  rclcpp::Time t_on_{0, 0, RCL_ROS_TIME}, last_progress_{0, 0, RCL_ROS_TIME}, next_plan_{0, 0, RCL_ROS_TIME}, t_wait0_{0, 0, RCL_ROS_TIME},
      map_t_{0, 0, RCL_ROS_TIME};
  float prev_gx_ = 1e30f, prev_gy_ = 1e30f, prev_dist_ = 0;
  std::vector<std::pair<float, float>> blacklist_;
  int n_goals_ = 0, n_reached_ = 0, n_aborted_ = 0;
  uint64_t goal_seq_ = 0;
  // 지도
  nav_msgs::msg::OccupancyGrid::ConstSharedPtr map_;
  std::vector<int16_t> L_;
  int w_ = 0, h_ = 0;
  float res_ = 0.05f, ox_ = 0, oy_ = 0;
  double known_m2_ = 0;
  std::vector<uint8_t> flag_;
  std::vector<int32_t> qa_, qb_;
  std::vector<mdp::Frontier> fr_;
  // ROS
  std::shared_ptr<tf2_ros::Buffer> tf_buf_;
  std::shared_ptr<tf2_ros::TransformListener> tf_listener_;
  rclcpp::Subscription<nav_msgs::msg::OccupancyGrid>::SharedPtr sub_map_;
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr pub_state_, pub_status_;
  rclcpp::Publisher<std_msgs::msg::Bool>::SharedPtr pub_done_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr pub_markers_;
  rclcpp::Service<std_srvs::srv::SetBool>::SharedPtr srv_enable_;
  rclcpp_action::Client<NavTo>::SharedPtr nav_;
  rclcpp_action::Client<Spin>::SharedPtr spin_;
  rclcpp_action::ClientGoalHandle<NavTo>::SharedPtr goal_handle_;
  rclcpp_action::ClientGoalHandle<Spin>::SharedPtr spin_handle_;
  rclcpp::TimerBase::SharedPtr timer_;
};

}  // namespace limo_fire_rl

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<limo_fire_rl::ExploreFrontierNode>());
  rclcpp::shutdown();
  return 0;
}
