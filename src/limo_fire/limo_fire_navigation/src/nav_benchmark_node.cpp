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
// 정답은 평가에만 쓴다 — 주행·위치 추정에 들어가지 않는다.
#include <yaml-cpp/yaml.h>

#include <chrono>
#include <cmath>
#include <ctime>
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
  double t, tx, ty, ex, ey, arrive, est_err, dist, dist_total;
};

class Bench : public rclcpp::Node {
 public:
  Bench() : Node("nav_benchmark") {
    world_file_ = declare_parameter<std::string>("world_file", "");
    label_ = declare_parameter<std::string>("label", "lidar");
    smoke_ = declare_parameter<bool>("smoke", false);
    goal_timeout_ = declare_parameter<double>("goal_timeout", 180.0);
    report_dir_ = declare_parameter<std::string>("report_dir", ".");
    truth_topic_ = declare_parameter<std::string>("truth_topic", "/sim/ground_truth");
    max_goals_ = declare_parameter<int>("max_goals", 0);
    if (world_file_.empty()) throw std::invalid_argument("nav_benchmark: world_file 파라미터가 비었음");

    tf_buffer_ = std::make_unique<tf2_ros::Buffer>(get_clock());
    tf_listener_ = std::make_unique<tf2_ros::TransformListener>(*tf_buffer_);
    nav_ = rclcpp_action::create_client<Nav>(this, "navigate_to_pose");
    smoke_cli_ = create_client<std_srvs::srv::SetBool>("smoke_mode/set");
    truth_sub_ = create_subscription<geometry_msgs::msg::PoseStamped>(
        truth_topic_, 10, [this](geometry_msgs::msg::PoseStamped::ConstSharedPtr m) {
          std::lock_guard<std::mutex> l(mu_);
          if (!loaded_) return;   // 월드(출발 자리)를 읽기 전에는 지도 좌표로 못 바꿈
          Pose2 w{m->pose.position.x, m->pose.position.y, tf2::getYaw(m->pose.orientation)};
          const Pose2 p = to_map(w);
          if (truth_) dist_ += std::hypot(p.x - truth_->x, p.y - truth_->y);
          truth_ = p;
        });
  }

  // 결과 코드: 0 = 끝까지 돎, 그 밖 = 준비 실패
  int run() {
    load_world();
    RCLCPP_INFO(get_logger(), "월드 %s: 출발 (%.2f, %.2f, %.2f rad), 순찰점 %zu 개, 조건 '%s'%s", world_.c_str(), start_.x,
                start_.y, start_.yaw, goals_.size(), label_.c_str(), smoke_ ? " (연기 모드)" : "");
    if (!wait_ready()) return 1;
    if (smoke_ && !set_smoke(true)) return 2;
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
    if (!s) throw std::runtime_error(world_file_ + ": robot_start 없음");
    start_ = {s["x"].as<double>(), s["y"].as<double>(), s["yaw"].as<double>()};
    for (const auto& p : y["patrol"]) goals_.push_back(to_map({p[0].as<double>(), p[1].as<double>(), 0.0}));
    if (goals_.empty()) throw std::runtime_error(world_file_ + ": patrol 이 비었음");
    std::lock_guard<std::mutex> l(mu_);
    loaded_ = true;
  }

  // 세계 → 지도: 지도 원점 = robot_start
  Pose2 to_map(const Pose2& w) const {
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
    const auto until = std::chrono::steady_clock::now() + std::chrono::seconds(120);
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
    RCLCPP_ERROR(get_logger(), "준비 안 됨 (120 s): Nav2 %s · 정답 %s(%s — 월드에서 LIMO supervisor TRUE?) · map → base_link %s",
                 nav ? "됨" : "없음", truth() ? "됨" : "없음", truth_topic_.c_str(), estimate() ? "됨" : "없음");
    return false;
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
    return r;
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
    std::ofstream csv(base + ".csv");
    csv << "goal,goal_x,goal_y,result,time_s,truth_x,truth_y,est_x,est_y,arrive_err_m,est_err_m,dist_m,dist_total_m\n";
    for (const auto& r : rows_)
      csv << r.i << ',' << r.gx << ',' << r.gy << ',' << r.result << ',' << r.t << ',' << r.tx << ',' << r.ty << ',' << r.ex
          << ',' << r.ey << ',' << r.arrive << ',' << r.est_err << ',' << r.dist << ',' << r.dist_total << '\n';
    std::ofstream md(base + ".md");
    md << "# 주행 벤치마크 — " << world_ << " · " << label_ << (smoke_ ? " (연기 모드: 라이다·카메라 끊음)" : "") << "\n\n";
    md << "- 날짜: " << ts << ", 도구: `limo_fire_navigation nav_benchmark`, 정답: Webots supervisor (" << truth_topic_ << ")\n";
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
    RCLCPP_INFO(get_logger(), "표: %s.md (%s.csv)%s", base.c_str(), base.c_str(), line);
  }

  std::string world_file_, label_, report_dir_, truth_topic_, world_;
  bool smoke_;
  double goal_timeout_;
  int max_goals_;
  Pose2 start_;
  std::vector<Pose2> goals_;
  std::vector<Row> rows_;
  std::mutex mu_;
  std::optional<Pose2> truth_;
  bool loaded_ = false;
  double dist_ = 0.0;
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
