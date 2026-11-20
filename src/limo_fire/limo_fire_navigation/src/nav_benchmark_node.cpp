// 주행 벤치마크 (시뮬 전용) — 월드 데이터 파일의 순찰점을 Nav2 로 차례로 가며, Webots 정답 자세와 비교해 표를 쓴다.
// 연기 모드 1 주차 데모 계획(docs/smoke_mode_plan.md) 1·3 단계: (a) 라이다 켬 · (b) 바퀴 + IMU 만(smoke:=true) · (c) + 초음파.
//
//   ros2 run limo_fire_navigation nav_benchmark --ros-args -p use_sim_time:=true
//     -p world_file:=<sim/webots/worlds/<월드>.yaml> -p label:=wheel_imu -p smoke:=true -p report_dir:=<docs/benchmarks>
//
// 정답: /sim/ground_truth (limo_fire_sim GroundTruthPose, Webots 세계 좌표 — 월드에서 LIMO supervisor TRUE 일 때만 나옴).
//   지도 좌표 = 세계 좌표를 robot_start 로 옮기고 돌린 것 (시뮬 지도 원점 = robot_start, sim/webots/README.md "월드 데이터 파일").
// 추정: TF map → base_link (Nav2 가 쓰는 자세 그대로).
// 잰 것: 목표마다 결과(성공·중단·시간 초과)·걸린 시간(시뮬)·도착 오차 |정답 − 목표|·추정 오차 |추정 − 정답|·정답으로 잰 이동 거리.
// 장애물 회피 시험(월드 데이터 파일에 contact_boxes · movers 가 있으면): 목표마다 몸통 사각형과 형상·걷는 사람 사이 가장 좁은 간격, 닿음 횟수,
//   갇힘(목표가 살아 있는 동안 stuck_s 넘게 stuck_m 안에 머문 가장 긴 시간) — docs/simulation.md "장애물 회피 시험".
// 정답은 평가에만 쓴다 — 주행·위치 추정에 들어가지 않는다.
#include <yaml-cpp/yaml.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <ctime>
#include <limits>
#include <filesystem>
#include <fstream>
#include <memory>
#include <mutex>
#include <optional>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include "geometry_msgs/msg/pose_stamped.hpp"
#include "nav2_msgs/action/navigate_to_pose.hpp"
#include "rclcpp/rclcpp.hpp"
#include "rclcpp_action/rclcpp_action.hpp"
#include "std_srvs/srv/set_bool.hpp"
#include "tf2/utils.h"
#include "tf2_geometry_msgs/tf2_geometry_msgs.hpp"
#include "tf2_ros/buffer.h"
#include "tf2_ros/transform_listener.h"

namespace {

using Nav = nav2_msgs::action::NavigateToPose;

struct Pose2 {
  double x = 0, y = 0, yaw = 0;
};

double wrap(double a) { return std::atan2(std::sin(a), std::cos(a)); }

struct Row {
  int i;
  double gx, gy;
  std::string result;
  double t, tx, ty, ex, ey, arrive, est_err, dist, dist_total, clear;
  double static_gap = NAN, mover_gap = NAN, stuck_max = 0.0;   // 형상 · 걷는 사람과 가장 좁은 간격, 가장 오래 갇힌 시간 (s)
  int contacts = 0, mover_contacts = 0;                        // 간격이 collide_gap 아래로 들어간 횟수
  std::string static_near;                                     // 가장 가까웠던 형상 이름
};

struct Box {   // 세계 좌표 축 정렬 상자 (가운데 · 가로 · 세로)
  double x, y, sx, sy;
  std::string name;
};

struct Mover {   // Webots pedestrian 컨트롤러: 시뮬 시각 t 에 고리 길 위 t × speed 자리
  std::vector<std::pair<double, double>> path;
  double speed = 0, radius = 0;
  std::string name;
  std::pair<double, double> at(double t) const {
    std::vector<double> cum{0.0};
    for (size_t i = 0; i < path.size(); ++i) {
      const auto& a = path[i];
      const auto& b = path[(i + 1) % path.size()];
      cum.push_back(cum.back() + std::hypot(b.first - a.first, b.second - a.second));
    }
    if (cum.back() <= 0) return path.front();
    const double d = std::fmod(t * speed, cum.back());
    size_t i = 0;
    while (i + 1 < cum.size() - 1 && cum[i + 1] <= d) ++i;
    const auto& a = path[i];
    const auto& b = path[(i + 1) % path.size()];
    const double r = (d - cum[i]) / std::max(1e-9, cum[i + 1] - cum[i]);
    return {a.first + r * (b.first - a.first), a.second + r * (b.second - a.second)};
  }
};

class Bench : public rclcpp::Node {
 public:
  Bench() : Node("nav_benchmark") {
    world_file_ = declare_parameter<std::string>("world_file", "");
    label_ = declare_parameter<std::string>("label", "lidar");
    smoke_ = declare_parameter<bool>("smoke", false);
    goal_timeout_ = declare_parameter<double>("goal_timeout", 180.0);
    report_dir_ = declare_parameter<std::string>("report_dir", ".");
    // 실물(real:=true): 정답 자세가 없다. 사람이 줄자로 잰 로봇 자리(지도 좌표)를 truth_topic 으로 알려 준다 — 시작 자리 한 번, 목표마다 멈춘 뒤 한 번.
    //   ros2 topic pub --once /real/measured_pose geometry_msgs/msg/PoseStamped "{header: {frame_id: map}, pose: {position: {x: 1.2, y: 0.8}}}"
    // 월드 데이터 파일은 순찰점을 지도 좌표 그대로 쓰도록 robot_start 를 0 0 0 으로 두거나 아예 뺀다.
    real_ = declare_parameter<bool>("real", false);
    real_wait_ = declare_parameter<double>("real_wait", 600.0);   // 목표 뒤 잰 값을 기다리는 시간 (s, 시계 시간)
    // 실물 안전(CLAUDE.md "실물 로봇"): 한 번에 움직이는 거리 한도. 이보다 먼 목표가 있으면 시작 전에 알리고 끝낸다. 0 = 안 막음(사람이 곁에서 보고 정한 뒤에만)
    max_leg_ = declare_parameter<double>("max_leg", 0.6);
    truth_topic_ = declare_parameter<std::string>("truth_topic", real_ ? "/real/measured_pose" : "/sim/ground_truth");
    max_goals_ = declare_parameter<int>("max_goals", 0);
    smoke_loc_ = declare_parameter<std::string>("smoke_localization", "hold");   // smoke:=true 일 때 연기 속 위치 (hold · wall)
    // 지도에 없는 장애물(4 단계): Webots 세계 좌표 [x, y, 가로, 세로] (m, 축 정렬). 비면 안 잰다.
    // 몸통 사각형(footprint, 자세 방향)과 상자 사이 가장 좁은 간격을 재고, collide_gap 보다 작으면 부딪힘으로 센다.
    // (2026-11-06 판은 몸통 가운데 거리 0.12 m 로 셌다 — 앞 범퍼가 가운데에서 0.16 m 라 닿아도 안 닿았다고 나왔다)
    obstacle_ = declare_parameter<std::vector<double>>("obstacle_box", std::vector<double>{});
    collide_clear_ = declare_parameter<double>("collide_gap", 0.01);
    foot_l_ = declare_parameter<double>("footprint_length", 0.32);   // nav2_params.yaml local_costmap footprint 와 같게
    foot_w_ = declare_parameter<double>("footprint_width", 0.22);
    if (!obstacle_.empty() && obstacle_.size() != 4) throw std::invalid_argument("nav_benchmark: obstacle_box 는 [x, y, 가로, 세로]");
    stuck_s_ = declare_parameter<double>("stuck_s", 10.0);   // 이보다 오래 stuck_m 안에 머물면 갇힘
    stuck_m_ = declare_parameter<double>("stuck_m", 0.15);
    if (world_file_.empty()) throw std::invalid_argument("nav_benchmark: world_file 파라미터가 비었음");

    tf_buffer_ = std::make_unique<tf2_ros::Buffer>(get_clock());
    tf_listener_ = std::make_unique<tf2_ros::TransformListener>(*tf_buffer_);
    // 주행 액션: Nav2 (navigate_to_pose) 또는 우리 탐사 스택 서버 (/explore/navigate_to_pose — limo_fire_explore README "주행 목표 서버")
    nav_ = rclcpp_action::create_client<Nav>(this, declare_parameter<std::string>("action_name", "navigate_to_pose"));
    smoke_cli_ = create_client<std_srvs::srv::SetBool>("smoke_mode/set");
    truth_sub_ = create_subscription<geometry_msgs::msg::PoseStamped>(
        truth_topic_, 10, [this](geometry_msgs::msg::PoseStamped::ConstSharedPtr m) {
          std::lock_guard<std::mutex> l(mu_);
          if (!loaded_) return;   // 월드(출발 자리)를 읽기 전에는 지도 좌표로 못 바꿈
          Pose2 w{m->pose.position.x, m->pose.position.y, tf2::getYaw(m->pose.orientation)};
          const Pose2 p = to_map(w);
          ++truth_seq_;
          if (truth_) dist_ += std::hypot(p.x - truth_->x, p.y - truth_->y);
          truth_ = p;
          if (obstacle_.size() == 4) clear_ = std::min(clear_, footprint_gap(w, Box{obstacle_[0], obstacle_[1], obstacle_[2], obstacle_[3], ""}));   // 세계 좌표에서
          track(w, now().seconds());   // 시뮬 시각 (use_sim_time) — 걷는 사람 자리도 이 시각으로
        });
  }

  // 결과 코드: 0 = 끝까지 돎, 그 밖 = 준비 실패
  int run() {
    load_world();
    RCLCPP_INFO(get_logger(), "월드 %s: 출발 (%.2f, %.2f, %.2f rad), 순찰점 %zu 개, 조건 '%s'%s", world_.c_str(), start_.x,
                start_.y, start_.yaw, goals_.size(), label_.c_str(), smoke_ ? " (연기 모드)" : "");
    if (!wait_ready()) return 1;
    if (real_ && max_leg_ > 0.0 && !legs_ok()) return 4;
    if (smoke_ && (!set_localization() || !set_smoke(true))) return 2;
    double total0 = traveled();
    for (size_t i = 0; i < goals_.size() && rclcpp::ok(); ++i) {
      if (max_goals_ > 0 && (int)i >= max_goals_) break;
      rows_.push_back(go((int)i + 1, goals_[i], total0));
      const Row& r = rows_.back();
      RCLCPP_INFO(get_logger(), "목표 %d (%.2f, %.2f): %s, %.1f s, 도착 오차 %.2f m, 추정 오차 %.2f m, 누적 이동 %.1f m", r.i,
                  r.gx, r.gy, r.result.c_str(), r.t, r.arrive, r.est_err, r.dist_total);
    }
    write_report();
    return 0;
  }

 private:
  void load_world() {
    const YAML::Node y = YAML::LoadFile(world_file_);
    world_ = std::filesystem::path(world_file_).stem().string();
    const auto s = y["robot_start"];
    if (!s && !real_) throw std::runtime_error(world_file_ + ": robot_start 없음");
    if (s) start_ = {s["x"].as<double>(), s["y"].as<double>(), s["yaw"].as<double>()};
    for (const auto& p : y["patrol"]) goals_.push_back(to_map({p[0].as<double>(), p[1].as<double>(), 0.0}));
    if (goals_.empty()) throw std::runtime_error(world_file_ + ": patrol 이 비었음");
    if (const auto cb = y["contact_boxes"]) {
      const auto nm = y["contact_names"];
      for (size_t i = 0; i < cb.size(); ++i)
        boxes_.push_back({cb[i][0].as<double>(), cb[i][1].as<double>(), cb[i][2].as<double>(), cb[i][3].as<double>(),
                          nm && i < nm.size() ? nm[i].as<std::string>() : "box" + std::to_string(i)});
    }
    for (const auto& g : y["corner_goals"]) corner_.push_back(g.as<int>());
    for (const auto& m : y["movers"]) {
      Mover mv;
      mv.name = m["name"].as<std::string>("mover");
      mv.speed = m["speed"].as<double>();
      mv.radius = m["radius"].as<double>(0.25);
      for (const auto& p : m["path"]) mv.path.emplace_back(p[0].as<double>(), p[1].as<double>());
      if (mv.path.size() < 2) throw std::runtime_error(world_file_ + ": movers 길은 점 2 개 이상");
      movers_.push_back(mv);
    }
    std::lock_guard<std::mutex> l(mu_);
    loaded_ = true;
  }

  // 몸통 사각형(가운데 w, 방향 w.yaw)과 축 정렬 상자 사이 가장 좁은 간격 (겹치면 0). 두 볼록 사각형: 겹침이 아니면 꼭짓점-변 거리의 최솟값
  double footprint_gap(const Pose2& w, const Box& b) const {
    const double c = std::cos(w.yaw), s = std::sin(w.yaw), hl = 0.5 * foot_l_, hw = 0.5 * foot_w_;
    const double bx0 = b.x - 0.5 * b.sx, bx1 = b.x + 0.5 * b.sx;
    const double by0 = b.y - 0.5 * b.sy, by1 = b.y + 0.5 * b.sy;
    double fx[4], fy[4];
    const double lx[4] = {hl, hl, -hl, -hl}, ly[4] = {hw, -hw, -hw, hw};
    for (int i = 0; i < 4; ++i) {
      fx[i] = w.x + c * lx[i] - s * ly[i];
      fy[i] = w.y + s * lx[i] + c * ly[i];
    }
    const double bxs[4] = {bx0, bx1, bx1, bx0}, bys[4] = {by0, by0, by1, by1};
    // 상자 꼭짓점이 몸통 안이면 겹침
    for (int k = 0; k < 4; ++k) {
      const double dx = bxs[k] - w.x, dy = bys[k] - w.y;
      if (std::fabs(c * dx + s * dy) <= hl && std::fabs(-s * dx + c * dy) <= hw) return 0.0;
    }
    auto seg = [](double px, double py, double ax, double ay, double bx, double by) {
      const double vx = bx - ax, vy = by - ay, l2 = vx * vx + vy * vy;
      const double t = l2 > 0 ? std::clamp(((px - ax) * vx + (py - ay) * vy) / l2, 0.0, 1.0) : 0.0;
      return std::hypot(px - (ax + t * vx), py - (ay + t * vy));
    };
    double g = std::numeric_limits<double>::infinity();
    for (int i = 0; i < 4; ++i) {
      if (fx[i] >= bx0 && fx[i] <= bx1 && fy[i] >= by0 && fy[i] <= by1) return 0.0;   // 몸통 꼭짓점이 상자 안
      const double dx = std::max({bx0 - fx[i], 0.0, fx[i] - bx1}), dy = std::max({by0 - fy[i], 0.0, fy[i] - by1});
      g = std::min(g, std::hypot(dx, dy));                                            // 몸통 꼭짓점 → 상자
      const int j = (i + 1) % 4;
      for (int k = 0; k < 4; ++k) g = std::min(g, seg(bxs[k], bys[k], fx[i], fy[i], fx[j], fy[j]));   // 상자 꼭짓점 → 몸통 변
    }
    return g;
  }

  // 몸통 사각형과 원(가운데 (cx, cy), 반지름 r) 사이 간격 (겹치면 0)
  double footprint_circle_gap(const Pose2& w, double cx, double cy, double r) const {
    const double c = std::cos(w.yaw), s = std::sin(w.yaw), dx = cx - w.x, dy = cy - w.y;
    const double lx = c * dx + s * dy, ly = -s * dx + c * dy;   // 몸통 틀
    const double ex = std::max(std::fabs(lx) - 0.5 * foot_l_, 0.0), ey = std::max(std::fabs(ly) - 0.5 * foot_w_, 0.0);
    return std::max(0.0, std::hypot(ex, ey) - r);
  }

  // 정답 자세 한 장마다 (mu_ 잡은 채): 목표가 살아 있을 때만 간격 · 닿음 · 갇힘
  void track(const Pose2& w, double t) {
    if (!active_) return;
    trace_.push_back({t, w.x, w.y, w.yaw, (double)rows_.size() + 1});   // 자취 (세계 좌표) — <표>_trace.csv
    for (const auto& b : boxes_) {
      const double g = footprint_gap(w, b);
      if (!(g >= cur_.static_gap)) { cur_.static_gap = g; cur_.static_near = b.name; }
    }
    if (!boxes_.empty()) {
      double g = INFINITY;
      for (const auto& b : boxes_) g = std::min(g, footprint_gap(w, b));
      const bool in = g < collide_clear_;
      if (in && !in_contact_) ++cur_.contacts;
      in_contact_ = in;
    }
    if (!movers_.empty()) {
      double g = INFINITY;
      for (const auto& mv : movers_) {
        const auto p = mv.at(t);
        g = std::min(g, footprint_circle_gap(w, p.first, p.second, mv.radius));
      }
      if (!(g >= cur_.mover_gap)) cur_.mover_gap = g;
      const bool in = g < collide_clear_;
      if (in && !in_mover_) ++cur_.mover_contacts;
      in_mover_ = in;
    }
    if (!anchor_ || std::hypot(w.x - anchor_->x, w.y - anchor_->y) > stuck_m_) {
      anchor_ = w;
      anchor_t_ = t;
    }
    cur_.stuck_max = std::max(cur_.stuck_max, t - anchor_t_);
  }

  // 세계 → 지도: 지도 원점 = robot_start
  Pose2 to_map(const Pose2& w) const {
    if (real_) return w;   // 실물: 잰 값도 순찰점도 이미 지도 좌표
    const double dx = w.x - start_.x, dy = w.y - start_.y, c = std::cos(-start_.yaw), s = std::sin(-start_.yaw);
    return {c * dx - s * dy, s * dx + c * dy, wrap(w.yaw - start_.yaw)};
  }

  std::optional<Pose2> truth() {
    std::lock_guard<std::mutex> l(mu_);
    return truth_;
  }
  double traveled() {
    std::lock_guard<std::mutex> l(mu_);
    return dist_;
  }
  std::optional<Pose2> estimate() {
    try {
      const auto t = tf_buffer_->lookupTransform("map", "base_link", tf2::TimePointZero);
      return Pose2{t.transform.translation.x, t.transform.translation.y, tf2::getYaw(t.transform.rotation)};
    } catch (const tf2::TransformException&) {
      return std::nullopt;
    }
  }

  // 준비 확인: Nav2 액션 서버 · 정답 자세 · map → base_link. 하나라도 없으면 까닭을 알리고 끝낸다
  bool wait_ready() {
    const auto until = std::chrono::steady_clock::now() + std::chrono::seconds(real_ ? static_cast<int>(real_wait_) : 120);   // 실물은 시작 자리를 재는 시간까지
    bool nav = false;
    while (std::chrono::steady_clock::now() < until && rclcpp::ok()) {
      nav = nav || nav_->wait_for_action_server(std::chrono::seconds(1));
      if (nav && truth() && estimate()) {
        if (smoke_ && !smoke_cli_->wait_for_service(std::chrono::seconds(5))) {
          RCLCPP_ERROR(get_logger(), "smoke_mode/set 서비스가 없음 (navigation.launch.xml 이 smoke_mode 노드를 띄워야 함)");
          return false;
        }
        return true;
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(500));
    }
    RCLCPP_ERROR(get_logger(), "준비 안 됨 (120 s): Nav2 %s · 정답 %s(%s — 시뮬은 월드에서 LIMO supervisor TRUE, 실물은 real:=true 로 잰 시작 자리를 알려 줘야 함) · map → base_link %s",
                 nav ? "됨" : "없음", truth() ? "됨" : "없음", truth_topic_.c_str(), estimate() ? "됨" : "없음");
    return false;
  }

  // 연기 모드 노드의 smoke_localization 을 바꾼다 (연기 모드가 꺼져 있을 때만 받아 준다)
  bool set_localization() {
    auto cli = std::make_shared<rclcpp::AsyncParametersClient>(this, "smoke_mode");
    if (!cli->wait_for_service(std::chrono::seconds(5))) {
      RCLCPP_ERROR(get_logger(), "smoke_mode 파라미터 서비스가 없음");
      return false;
    }
    auto f = cli->set_parameters({rclcpp::Parameter("smoke_localization", smoke_loc_)});
    if (f.wait_for(std::chrono::seconds(10)) != std::future_status::ready || !f.get().at(0).successful) {
      RCLCPP_ERROR(get_logger(), "smoke_localization=%s 설정 실패", smoke_loc_.c_str());
      return false;
    }
    RCLCPP_INFO(get_logger(), "연기 속 위치: %s", smoke_loc_.c_str());
    return true;
  }

  bool set_smoke(bool on) {
    auto req = std::make_shared<std_srvs::srv::SetBool::Request>();
    req->data = on;
    auto f = smoke_cli_->async_send_request(req);
    if (f.wait_for(std::chrono::seconds(10)) != std::future_status::ready) {
      RCLCPP_ERROR(get_logger(), "smoke_mode/set 응답 없음");
      return false;
    }
    const auto res = f.get();
    if (!res->success) RCLCPP_ERROR(get_logger(), "연기 모드 못 켬: %s", res->message.c_str());
    else RCLCPP_INFO(get_logger(), "연기 모드: %s", res->message.c_str());
    return res->success;
  }

  Row go(int i, const Pose2& g, double total0) {
    Row r{};
    r.i = i;
    r.gx = g.x;
    r.gy = g.y;
    const auto here = truth();
    Nav::Goal goal;
    goal.pose.header.frame_id = "map";
    goal.pose.header.stamp = now();
    goal.pose.pose.position.x = g.x;
    goal.pose.pose.position.y = g.y;
    const double yaw = here ? std::atan2(g.y - here->y, g.x - here->x) : 0.0;   // 오는 방향을 보게 (도착 뒤 제자리 회전을 줄임)
    goal.pose.pose.orientation.z = std::sin(yaw / 2);
    goal.pose.pose.orientation.w = std::cos(yaw / 2);
    const double d0 = traveled();
    {
      std::lock_guard<std::mutex> l(mu_);
      clear_ = std::numeric_limits<double>::infinity();
      cur_ = Row{};
      in_contact_ = in_mover_ = false;
      anchor_.reset();
      active_ = true;
    }
    const rclcpp::Time t0 = now();
    auto gh_f = nav_->async_send_goal(goal);
    if (gh_f.wait_for(std::chrono::seconds(10)) != std::future_status::ready || !gh_f.get()) {
      r.result = "rejected";
    } else {
      auto gh = gh_f.get();
      auto res_f = nav_->async_get_result(gh);
      // 시간 초과는 시뮬 시간으로 (Webots 가 실시간보다 느릴 수 있음)
      while (rclcpp::ok() && res_f.wait_for(std::chrono::milliseconds(200)) != std::future_status::ready) {
        if ((now() - t0).seconds() > goal_timeout_) break;
      }
      if (res_f.wait_for(std::chrono::seconds(0)) == std::future_status::ready) {
        switch (res_f.get().code) {
          case rclcpp_action::ResultCode::SUCCEEDED: r.result = "succeeded"; break;
          case rclcpp_action::ResultCode::ABORTED: r.result = "aborted"; break;
          case rclcpp_action::ResultCode::CANCELED: r.result = "canceled"; break;
          default: r.result = "unknown";
        }
      } else {
        r.result = "timeout";
        auto c = nav_->async_cancel_goal(gh);
        c.wait_for(std::chrono::seconds(5));
      }
    }
    r.t = (now() - t0).seconds();
    {
      std::lock_guard<std::mutex> l(mu_);
      active_ = false;
      r.static_gap = cur_.static_gap;
      r.static_near = cur_.static_near;
      r.mover_gap = cur_.mover_gap;
      r.contacts = cur_.contacts;
      r.mover_contacts = cur_.mover_contacts;
      r.stuck_max = cur_.stuck_max;
    }
    wait_measure(i, g);
    const auto tr = truth();
    const auto es = estimate();
    r.tx = tr ? tr->x : NAN;
    r.ty = tr ? tr->y : NAN;
    r.ex = es ? es->x : NAN;
    r.ey = es ? es->y : NAN;
    r.arrive = std::hypot(r.tx - g.x, r.ty - g.y);
    r.est_err = std::hypot(r.ex - r.tx, r.ey - r.ty);
    r.dist = traveled() - d0;
    r.dist_total = traveled() - total0;
    {
      std::lock_guard<std::mutex> l(mu_);
      r.clear = clear_;
    }
    return r;
  }

  // 실물: 시작 자리 → 목표 1 → 목표 2 … 직선 거리가 max_leg 를 넘는 구간이 있으면 막는다 (Nav2 경로는 더 길 수 있으니 직선은 하한)
  bool legs_ok() {
    auto p = truth();
    bool ok = true;
    Pose2 prev = p ? *p : Pose2{};
    for (size_t i = 0; i < goals_.size() && (max_goals_ <= 0 || (int)i < max_goals_); ++i) {
      const double d = std::hypot(goals_[i].x - prev.x, goals_[i].y - prev.y);
      if (d > max_leg_) {
        RCLCPP_ERROR(get_logger(), "목표 %zu 까지 직선 %.2f m > max_leg %.2f m — 실물은 한 번에 움직이는 거리를 제한(주변 비움·사람 곁에 확인 뒤 -p max_leg:=<m> 로 올리거나 0 으로)", i + 1, d, max_leg_);
        ok = false;
      }
      prev = goals_[i];
    }
    return ok;
  }

  // 실물: 목표 결과가 나오면 로봇 자리를 줄자로 재서 알려 줄 때까지 기다린다 (추정 오차·도착 오차의 정답). 시뮬이면 바로 지나감
  void wait_measure(int i, const Pose2& g) {
    if (!real_) return;
    uint64_t seq0;
    {
      std::lock_guard<std::mutex> l(mu_);
      seq0 = truth_seq_;
    }
    RCLCPP_WARN(get_logger(), "목표 %d (%.2f, %.2f) 끝 — 로봇 자리를 재서 %s 로 알려 주세요 (%.0f s 안)", i, g.x, g.y, truth_topic_.c_str(), real_wait_);
    const auto until = std::chrono::steady_clock::now() + std::chrono::duration<double>(real_wait_);
    while (rclcpp::ok() && std::chrono::steady_clock::now() < until) {
      {
        std::lock_guard<std::mutex> l(mu_);
        if (truth_seq_ > seq0) return;
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(200));
    }
    RCLCPP_ERROR(get_logger(), "목표 %d: 잰 값이 안 왔음 — 이전 값을 그대로 씀 (오차가 틀림)", i);
  }

  void write_report() {
    std::filesystem::create_directories(report_dir_);
    const std::time_t now_c = std::time(nullptr);
    char ts[32];
    std::strftime(ts, sizeof ts, "%Y%m%d_%H%M", std::localtime(&now_c));
    const std::string base = report_dir_ + "/" + ts + "_nav_" + label_ + "_" + world_;
    int ok = 0;
    double sum_arr = 0, max_est = 0;
    for (const auto& r : rows_) {
      ok += r.result == "succeeded";
      if (r.result == "succeeded") sum_arr += r.arrive;
      if (std::isfinite(r.est_err)) max_est = std::max(max_est, r.est_err);
    }
    {
      std::ofstream tr(base + "_trace.csv");
      tr << "t_s,world_x,world_y,world_yaw,goal\n";
      for (const auto& q : trace_) tr << q[0] << ',' << q[1] << ',' << q[2] << ',' << q[3] << ',' << (int)q[4] << '\n';
    }
    std::ofstream csv(base + ".csv");
    csv << "goal,goal_x,goal_y,result,time_s,truth_x,truth_y,est_x,est_y,arrive_err_m,est_err_m,dist_m,dist_total_m,obstacle_clear_m,"
           "static_gap_m,static_near,contacts,mover_gap_m,mover_contacts,stuck_max_s\n";
    for (const auto& r : rows_)
      csv << r.i << ',' << r.gx << ',' << r.gy << ',' << r.result << ',' << r.t << ',' << r.tx << ',' << r.ty << ',' << r.ex
          << ',' << r.ey << ',' << r.arrive << ',' << r.est_err << ',' << r.dist << ',' << r.dist_total << ',' << r.clear << ','
          << r.static_gap << ',' << r.static_near << ',' << r.contacts << ',' << r.mover_gap << ',' << r.mover_contacts << ',' << r.stuck_max << '\n';
    std::ofstream md(base + ".md");
    md << "# 주행 벤치마크 — " << world_ << " · " << label_
       << (smoke_ ? " (연기 모드: 라이다·카메라 끊음, 위치 " + smoke_loc_ + ")" : std::string()) << "\n\n";
    md << "- 날짜: " << ts << ", 도구: `limo_fire_navigation nav_benchmark`, 정답: " << (real_ ? std::string("실물 — 사람이 줄자로 잰 자리 (") : std::string("Webots supervisor (")) << truth_topic_ << ")\n";
    md << "- 목표: 월드 데이터 파일 순찰점 " << rows_.size() << " 개 (지도 좌표), 시간 한도 " << goal_timeout_ << " s (시뮬)\n";
    md << "- 도착 오차 = |정답 − 목표|, 추정 오차 = |Nav2 자세(map → base_link) − 정답|, 이동 = 정답으로 잰 거리\n\n";
    md << "| 목표 | 지도 (x, y) | 결과 | 시간 s | 도착 오차 m | 추정 오차 m | 이동 m | 누적 이동 m |\n|---|---|---|---|---|---|---|---|\n";
    char line[256];
    for (const auto& r : rows_) {
      std::snprintf(line, sizeof line, "| %d | (%.2f, %.2f) | %s | %.1f | %.2f | %.2f | %.1f | %.1f |\n", r.i, r.gx, r.gy,
                    r.result.c_str(), r.t, r.arrive, r.est_err, r.dist, r.dist_total);
      md << line;
    }
    std::snprintf(line, sizeof line, "\n**요약**: 성공 %d / %zu, 성공한 목표의 평균 도착 오차 %.2f m, 가장 큰 추정 오차 %.2f m\n", ok,
                  rows_.size(), ok ? sum_arr / ok : NAN, max_est);
    md << line;
    if (!boxes_.empty() || !movers_.empty()) {
      md << "\n| 목표 | 형상 간격 m (가장 가까운 것) | 닿음 | 사람 간격 m | 사람 닿음 | 가장 오래 갇힘 s |\n|---|---|---|---|---|---|\n";
      int nc = 0, nm = 0, ns = 0, nsc = 0;
      for (const auto& r : rows_) {
        std::snprintf(line, sizeof line, "| %d | %.3f (%s) | %d | %.3f | %d | %.1f |\n", r.i, r.static_gap, r.static_near.c_str(), r.contacts, r.mover_gap,
                      r.mover_contacts, r.stuck_max);
        md << line;
        nc += r.contacts;
        nm += r.mover_contacts;
        ns += r.stuck_max > stuck_s_;
        if (std::find(corner_.begin(), corner_.end(), r.i) != corner_.end()) nsc += r.stuck_max > stuck_s_;
      }
      std::snprintf(line, sizeof line, "\n**회피**: 형상 닿음 %d 번 · 사람 닿음 %d 번 · %.0f s 넘게 갇힌 목표 %d 개 (닿음 = 간격 %.2f m 미만, 갇힘 = %.2f m 안)\n", nc, nm,
                    stuck_s_, ns, collide_clear_, stuck_m_);
      md << line;
      RCLCPP_INFO(get_logger(), "%s", line);
      if (!corner_.empty()) {
        std::snprintf(line, sizeof line, "**구석**: 구석·막다른 곳 목표 %zu 개 중 %.0f s 넘게 갇힘 %d 개\n", corner_.size(), stuck_s_, nsc);
        md << line;
        RCLCPP_INFO(get_logger(), "%s", line);
      }
    }
    if (obstacle_.size() == 4) {
      double mc = std::numeric_limits<double>::infinity();
      for (const auto& r : rows_) mc = std::min(mc, r.clear);
      char ob[400];
      std::snprintf(ob, sizeof ob,
                    "\n**지도에 없는 장애물**: 상자 세계 (%.2f, %.2f) 크기 %.2f × %.2f m — 몸통 사각형(%.2f × %.2f m)과 상자 사이 가장 좁은 간격 %.3f m → %s "
                    "(%.2f m 미만이면 부딪힘)\n",
                    obstacle_[0], obstacle_[1], obstacle_[2], obstacle_[3], foot_l_, foot_w_, mc, mc < collide_clear_ ? "**부딪힘**" : "안 부딪힘",
                    collide_clear_);
      md << ob;
      RCLCPP_INFO(get_logger(), "%s", ob);
    }
    RCLCPP_INFO(get_logger(), "표: %s.md (%s.csv)%s", base.c_str(), base.c_str(), line);
  }

  std::string world_file_, label_, report_dir_, truth_topic_, world_, smoke_loc_;
  bool smoke_, real_ = false;
  double real_wait_ = 600.0, max_leg_ = 0.6;
  uint64_t truth_seq_ = 0;
  double goal_timeout_;
  int max_goals_;
  Pose2 start_;
  std::vector<Pose2> goals_;
  std::vector<Row> rows_;
  std::mutex mu_;
  std::optional<Pose2> truth_;
  std::vector<double> obstacle_;
  double collide_clear_ = 0.01, foot_l_ = 0.32, foot_w_ = 0.22, clear_ = std::numeric_limits<double>::infinity();
  bool loaded_ = false;
  double dist_ = 0.0;
  std::vector<Box> boxes_;
  std::vector<Mover> movers_;
  std::vector<int> corner_;   // 구석·막다른 곳 목표 번호 (월드 데이터 파일 corner_goals)
  double stuck_s_ = 10.0, stuck_m_ = 0.15, anchor_t_ = 0.0;
  bool active_ = false, in_contact_ = false, in_mover_ = false;
  std::optional<Pose2> anchor_;
  Row cur_{};
  std::vector<std::array<double, 5>> trace_;
  std::unique_ptr<tf2_ros::Buffer> tf_buffer_;
  std::unique_ptr<tf2_ros::TransformListener> tf_listener_;
  rclcpp_action::Client<Nav>::SharedPtr nav_;
  rclcpp::Client<std_srvs::srv::SetBool>::SharedPtr smoke_cli_;
  rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr truth_sub_;
};

}  // namespace

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  auto node = std::make_shared<Bench>();
  rclcpp::executors::MultiThreadedExecutor ex;
  ex.add_node(node);
  std::thread spin([&ex] { ex.spin(); });
  int code = 0;
  try {
    code = node->run();
  } catch (const std::exception& e) {
    RCLCPP_ERROR(node->get_logger(), "%s", e.what());
    code = 3;
  }
  ex.cancel();
  spin.join();
  rclcpp::shutdown();
  return code;
}
