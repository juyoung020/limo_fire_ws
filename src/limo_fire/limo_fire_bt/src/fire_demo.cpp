// Deterministic control demo: injected metric observations, actual Nav2 +
// Webots motion. This is NOT a YOLO accuracy benchmark. PASS requires action
// success and measured motion.
#include "limo_fire_bt/safety.hpp"
#include <fstream>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <iomanip>
#include <lifecycle_msgs/srv/get_state.hpp>
#include <limo_fire_msgs/msg/danger_zones.hpp>
#include <limo_fire_msgs/msg/fire_observation.hpp>
#include <nav2_msgs/action/navigate_to_pose.hpp>
#include <nav2_msgs/srv/manage_lifecycle_nodes.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <rclcpp/rclcpp.hpp>
#include <rclcpp_action/rclcpp_action.hpp>
#include <set>
#include <std_msgs/msg/string.hpp>
class Demo : public rclcpp::Node {
  using A = nav2_msgs::action::NavigateToPose;
  geometry_msgs::msg::PoseStamped::ConstSharedPtr truth_;
  rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr truth_sub_;
  double truth_speed_ = 0, freshness_ = 0, min_height_ = 0, max_height_ = 0,
         verified_stop_speed_ = 0;
  rclcpp_action::Client<A>::SharedPtr nav_;
  rclcpp::Publisher<limo_fire_msgs::msg::FireObservation>::SharedPtr fire_;
  rclcpp::Subscription<std_msgs::msg::String>::SharedPtr state_;
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_;
  rclcpp::Subscription<limo_fire_msgs::msg::DangerZones>::SharedPtr zones_;
  rclcpp::TimerBase::SharedPtr timer_;
  rclcpp::Client<lifecycle_msgs::srv::GetState>::SharedPtr readiness_;
  bool readiness_pending_ = false, navigation_active_ = false;
  bool expect_stop_ = false, mission_failed_ = false, safe_stop_ = false;
  double safe_since_ = 0, safe_hold_ = 0;
  std::string safe_reason_;
  using Shutdown = nav2_msgs::srv::ManageLifecycleNodes;
  std::vector<rclcpp::Client<Shutdown>::SharedPtr> shutdown_clients_;
  size_t shutdown_phase_ = 0;
  bool shutdown_pending_ = false;
  std::chrono::steady_clock::time_point shutdown_started_;
  double shutdown_timeout_ = 0;
  std::set<std::string> states_;
  std::vector<limo_fire_bt::Circle> hazards_;
  double start_ = 0, last_x_ = 0, last_y_ = 0, distance_ = 0, small_at_ = 0,
         stop_latency_ = -1, replan_at_ = 0, replan_latency_ = -1,
         path_length_ = -1;
  double odom_at_ = 0, speed_ = 0, resumed_x_ = 0;
  bool sent_ = false, small_ = false, large_ = false, resumed_ = false,
       have_pose_ = false, finished_ = false, invaded_ = false;
  std::vector<double> goal_, small_pos_, large_pos_;
  double timeout_, trigger_small_, trigger_large_, goal_tolerance_,
      movement_min_;
  std::string map_, report_;
  void finish(bool ok, const std::string &reason) {
    if (finished_)
      return;
    finished_ = true;
    std::ofstream out(report_);
    out << "result,mission_success,stop_latency_s,replan_latency_s,path_length_"
           "m,danger_intrusion,resume_success,distance_m,final_x,final_y,final_"
           "error_m,reason\n"
        << (ok ? "PASS" : "FAIL") << "," << ok << "," << stop_latency_ << ","
        << replan_latency_ << "," << path_length_ << "," << invaded_ << ","
        << resumed_ << "," << distance_ << "," << last_x_ << "," << last_y_
        << "," << std::hypot(last_x_ - goal_[0], last_y_ - goal_[1]) << ","
        << std::quoted(reason) << "\n";
    if (!out)
      RCLCPP_ERROR(get_logger(), "cannot write report %s", report_.c_str());
    RCLCPP_INFO(get_logger(), "%s actual simulation demo: %s",
                ok ? "PASS" : "FAIL", reason.c_str());
    result = ok ? 0 : 1;
    shutdown_started_ = std::chrono::steady_clock::now();
  }
  void shutdown_tick() {
    if (shutdown_phase_ >= shutdown_clients_.size()) {
      RCLCPP_INFO(get_logger(), "Nav2 lifecycle shutdown responses verified");
      done = true;
      return;
    }
    if (std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                      shutdown_started_)
            .count() > shutdown_timeout_) {
      RCLCPP_ERROR(get_logger(), "Nav2 lifecycle shutdown timeout");
      result = 1;
      done = true;
      return;
    }
    auto c = shutdown_clients_[shutdown_phase_];
    if (shutdown_pending_ || !c->service_is_ready())
      return;
    shutdown_pending_ = true;
    auto request = std::make_shared<Shutdown::Request>();
    request->command = Shutdown::Request::SHUTDOWN;
    c->async_send_request(
        request, [this](rclcpp::Client<Shutdown>::SharedFuture f) {
          if (!f.get()->success) {
            RCLCPP_ERROR(get_logger(), "Nav2 lifecycle shutdown rejected");
            result = 1;
          }
          ++shutdown_phase_;
          shutdown_pending_ = false;
        });
  }
  void inject(const std::string &id, const std::vector<double> &p,
              double width) {
    limo_fire_msgs::msg::FireObservation f;
    f.header.frame_id = map_;
    f.header.stamp = now();
    f.id = id;
    f.position.x = p[0];
    f.position.y = p[1];
    f.position_valid = true;
    f.size_valid = true;
    f.width_m = width;
    f.confidence = 1;
    fire_->publish(f);
  }
  void tick() {
    if (finished_) {
      shutdown_tick();
      return;
    }
    if (!navigation_active_ && !readiness_pending_ &&
        readiness_->service_is_ready()) {
      readiness_pending_ = true;
      readiness_->async_send_request(
          std::make_shared<lifecycle_msgs::srv::GetState::Request>(),
          [this](
              rclcpp::Client<lifecycle_msgs::srv::GetState>::SharedFuture f) {
            readiness_pending_ = false;
            navigation_active_ = f.get()->current_state.id == 3;
          });
    }
    if (finished_ || now().nanoseconds() == 0)
      return;
    if (start_ == 0)
      start_ = now().seconds();
    if (now().seconds() - start_ > timeout_) {
      finish(false, "demo timeout");
      return;
    }
    if (!truth_ ||
        (now() - rclcpp::Time(truth_->header.stamp)).seconds() > freshness_) {
      if (sent_)
        finish(false, "independent ground truth unavailable");
      return;
    }
    const auto &p = truth_->pose.position;
    if (truth_->header.frame_id != map_ || !std::isfinite(p.x + p.y + p.z) ||
        p.z < min_height_ || p.z > max_height_) {
      finish(false, "robot ground-truth height or coordinates invalid");
      return;
    }
    const double x = p.x, y = p.y;
    if (expect_stop_ && safe_stop_ && mission_failed_ &&
        truth_speed_ <= verified_stop_speed_ &&
        now().seconds() - safe_since_ >= safe_hold_) {
      RCLCPP_INFO(get_logger(), "PASS negative safety check: Action failure + "
                                "SAFE_STOP + independent physical stop");
      finish(false, "EXPECTED_SAFE_STOP_VERIFIED: " + safe_reason_);
      return;
    }
    if (!limo_fire_bt::segment_safe(have_pose_ ? last_x_ : x,
                                    have_pose_ ? last_y_ : y, x, y, hazards_)) {
      // A new zone can contain an already stopped robot. Negative tests permit
      // remaining stationary there, never movement or a successful mission.
      if (!expect_stop_ || truth_speed_ > verified_stop_speed_) {
        invaded_ = true;
        finish(false, "ground-truth danger zone intrusion");
        return;
      }
    }
    if (have_pose_)
      distance_ += std::hypot(x - last_x_, y - last_y_);
    last_x_ = x;
    last_y_ = y;
    have_pose_ = true;
    if (!sent_ && navigation_active_ && now().seconds() - odom_at_ < 1.0 &&
        odom_at_ > 0 && nav_->action_server_is_ready()) {
      sent_ = true;
      A::Goal g;
      g.pose.header.frame_id = map_;
      g.pose.header.stamp = now();
      g.pose.pose.position.x = goal_[0];
      g.pose.pose.position.y = goal_[1];
      g.pose.pose.orientation.w = 1;
      rclcpp_action::Client<A>::SendGoalOptions opts;
      opts.goal_response_callback = [this](auto h) {
        if (!h)
          finish(false, "mission rejected");
      };
      opts.result_callback = [this](const auto &r) {
        if (expect_stop_) {
          mission_failed_ = r.code == rclcpp_action::ResultCode::ABORTED ||
                            r.code == rclcpp_action::ResultCode::CANCELED;
          if (!mission_failed_)
            finish(false, "unexpected successful mission in negative test");
          return;
        }
        bool phases =
            states_.count("STOPPING") && states_.count("EXTINGUISHING") &&
            states_.count("VERIFY_RESULT") &&
            states_.count("RESUME_NAVIGATION") &&
            states_.count("UPDATE_DANGER_ZONE") &&
            states_.count("REPLANNING") && states_.count("FOLLOW_NEW_PATH");
        bool arrived = std::hypot(last_x_ - goal_[0], last_y_ - goal_[1]) <
                       goal_tolerance_;
        RCLCPP_INFO(get_logger(),
                    "final independent checks: action=%d phases=%d arrived=%d "
                    "x=%.3f y=%.3f error_m=%.3f",
                    int(r.code), int(phases), int(arrived), last_x_, last_y_,
                    std::hypot(last_x_ - goal_[0], last_y_ - goal_[1]));
        bool ok = r.code == rclcpp_action::ResultCode::SUCCEEDED && phases &&
                  small_ && large_ && resumed_ && arrived && !invaded_ &&
                  distance_ > movement_min_ && stop_latency_ >= 0 &&
                  replan_latency_ >= 0 && path_length_ > 0;
        finish(ok, ok ? "all stages and final pose verified"
                      : "action, phase, motion or final pose check failed");
      };
      nav_->async_send_goal(g, opts);
    }
    if (sent_ && !small_ && x >= trigger_small_ && speed_ > 0.02) {
      small_ = true;
      small_at_ = now().seconds();
      inject("demo_candle", small_pos_, 0.056);
    }
    if (resumed_ && !large_ && x >= trigger_large_ && x - resumed_x_ > 0.2 &&
        speed_ > 0.02) {
      large_ = true;
      inject("demo_floor", large_pos_, 0.7);
    }
    if (small_ && stop_latency_ < 0 && speed_ < 0.01 &&
        now().seconds() - odom_at_ < 1 && states_.count("STOPPING"))
      stop_latency_ = now().seconds() - small_at_;
  }

public:
  bool done = false;
  int result = 1;
  Demo() : Node("fire_demo") {
    shutdown_timeout_ = declare_parameter<double>("shutdown_timeout_s");
    shutdown_clients_.push_back(create_client<Shutdown>(
        declare_parameter<std::string>("navigation_shutdown_service")));
    shutdown_clients_.push_back(create_client<Shutdown>(
        declare_parameter<std::string>("localization_shutdown_service")));
    readiness_ = create_client<lifecycle_msgs::srv::GetState>(
        declare_parameter<std::string>("navigation_ready_service"));
    if (!get_parameter("use_sim_time").as_bool())
      throw std::runtime_error("simulation only");
    map_ = declare_parameter<std::string>("map_frame");
    report_ = declare_parameter<std::string>("report");
    goal_ = declare_parameter<std::vector<double>>("goal");
    small_pos_ = declare_parameter<std::vector<double>>("small_position");
    large_pos_ = declare_parameter<std::vector<double>>("large_position");
    timeout_ = declare_parameter<double>("timeout_s");
    trigger_small_ = declare_parameter<double>("small_trigger_x");
    trigger_large_ = declare_parameter<double>("large_trigger_x");
    goal_tolerance_ = declare_parameter<double>("goal_tolerance_m");
    movement_min_ = declare_parameter<double>("min_movement_m");
    freshness_ = declare_parameter<double>("freshness_s");
    min_height_ = declare_parameter<double>("min_base_height_m");
    max_height_ = declare_parameter<double>("max_base_height_m");
    verified_stop_speed_ = declare_parameter<double>("verified_stop_speed_mps");
    expect_stop_ = declare_parameter<bool>("expect_safe_stop");
    safe_hold_ = declare_parameter<double>("negative_stop_hold_s");
    truth_sub_ = create_subscription<geometry_msgs::msg::PoseStamped>(
        declare_parameter<std::string>("ground_truth_topic"), 10,
        [this](geometry_msgs::msg::PoseStamped::ConstSharedPtr m) {
          if (truth_) {
            const auto dt = (rclcpp::Time(m->header.stamp) -
                             rclcpp::Time(truth_->header.stamp))
                                .seconds();
            if (dt > 0)
              truth_speed_ =
                  std::hypot(m->pose.position.x - truth_->pose.position.x,
                             m->pose.position.y - truth_->pose.position.y) /
                  dt;
          }
          truth_ = m;
        });
    if (goal_.size() != 2 || small_pos_.size() != 2 || large_pos_.size() != 2)
      throw std::runtime_error("expected xy coordinates");
    nav_ = rclcpp_action::create_client<A>(
        this, declare_parameter<std::string>("mission_action"));
    fire_ = create_publisher<limo_fire_msgs::msg::FireObservation>(
        declare_parameter<std::string>("fire_topic"), 10);
    state_ = create_subscription<std_msgs::msg::String>(
        declare_parameter<std::string>("state_topic"), 20,
        [this](std_msgs::msg::String::ConstSharedPtr m) {
          auto s = m->data.substr(0, m->data.find('|'));
          states_.insert(s);
          if (s == "EXTINGUISHING" &&
              (!truth_ || truth_speed_ > verified_stop_speed_)) {
            finish(false, "extinguishing requested before physical stop");
            return;
          }
          if (s == "SAFE_STOP") {
            if (expect_stop_) {
              safe_stop_ = true;
              safe_since_ = now().seconds();
              safe_reason_ = m->data;
            } else
              finish(false, m->data);
            return;
          }
          if (s == "RESUME_NAVIGATION") {
            resumed_ = true;
            resumed_x_ = last_x_;
          }
          if (s == "REPLANNING")
            replan_at_ = now().seconds();
          if (s == "FOLLOW_NEW_PATH") {
            replan_latency_ = now().seconds() - replan_at_;
            auto p = m->data.find("path_length_m=");
            if (p != std::string::npos)
              path_length_ = std::stod(m->data.substr(p + 14));
          }
        });
    odom_ = create_subscription<nav_msgs::msg::Odometry>(
        declare_parameter<std::string>("odom_topic"), rclcpp::SensorDataQoS(),
        [this](nav_msgs::msg::Odometry::ConstSharedPtr m) {
          speed_ =
              std::hypot(m->twist.twist.linear.x, m->twist.twist.linear.y) +
              std::abs(m->twist.twist.angular.z);
          odom_at_ = rclcpp::Time(m->header.stamp).seconds();
        });
    zones_ = create_subscription<limo_fire_msgs::msg::DangerZones>(
        declare_parameter<std::string>("zones_topic"),
        rclcpp::QoS(1).transient_local(),
        [this](limo_fire_msgs::msg::DangerZones::ConstSharedPtr m) {
          hazards_.clear();
          for (size_t i = 0; i < m->centers.size() && i < m->radii.size(); ++i)
            hazards_.push_back({m->centers[i].x, m->centers[i].y, m->radii[i]});
        });
    timer_ = create_wall_timer(
        std::chrono::duration<double>(declare_parameter<double>("tick_s")),
        [this] { tick(); });
  }
};
int main(int argc, char **argv) {
  rclcpp::init(argc, argv);
  auto d = std::make_shared<Demo>();
  while (rclcpp::ok() && !d->done)
    rclcpp::spin_some(d);
  int result = d->result;
  rclcpp::shutdown();
  return result;
}
