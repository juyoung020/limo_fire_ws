#include "limo_fire_bt/safety.hpp"
#include <deque>
#include <geometry_msgs/msg/twist.hpp>
#include <limo_fire_msgs/action/extinguish.hpp>
#include <limo_fire_msgs/msg/danger_zones.hpp>
#include <limo_fire_msgs/msg/fire_observation.hpp>
#include <map>
#include <nav2_msgs/action/compute_path_to_pose.hpp>
#include <nav2_msgs/action/navigate_to_pose.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <rclcpp/rclcpp.hpp>
#include <rclcpp_action/rclcpp_action.hpp>
#include <sensor_msgs/msg/laser_scan.hpp>
#include <std_msgs/msg/string.hpp>
#include <std_msgs/msg/u_int64.hpp>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>
using namespace std::chrono_literals;
namespace limo_fire_bt {
class FireManager : public rclcpp::Node {
  using N = nav2_msgs::action::NavigateToPose;
  using P = nav2_msgs::action::ComputePathToPose;
  using E = limo_fire_msgs::action::Extinguish;
  using F = limo_fire_msgs::msg::FireObservation;
  using H = rclcpp_action::ServerGoalHandle<N>;
  using NH = rclcpp_action::ClientGoalHandle<N>;
  using EH = rclcpp_action::ClientGoalHandle<E>;
  rclcpp_action::Server<N>::SharedPtr server_;
  rclcpp_action::Client<N>::SharedPtr nav_;
  rclcpp_action::Client<P>::SharedPtr planner_;
  rclcpp_action::Client<E>::SharedPtr extinguisher_;
  std::shared_ptr<H> mission_;
  NH::SharedPtr nav_goal_;
  EH::SharedPtr ext_goal_;
  tf2_ros::Buffer tf_;
  tf2_ros::TransformListener listener_;
  rclcpp::Publisher<geometry_msgs::msg::Twist>::SharedPtr velocity_;
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr status_;
  rclcpp::Publisher<limo_fire_msgs::msg::DangerZones>::SharedPtr zone_pub_;
  std::vector<rclcpp::SubscriptionBase::SharedPtr> subs_;
  rclcpp::TimerBase::SharedPtr timer_;
  std::deque<F> queue_;
  std::map<std::string, Size> seen_;
  std::map<std::string, F> last_seen_;
  F fire_;
  std::vector<Circle> circles_;
  nav_msgs::msg::Path last_path_;
  limo_fire_msgs::msg::DangerZones zones_;
  std::string state_ = "NAVIGATING", reason_, map_frame_, base_frame_;
  rclcpp::Time entered_{0, 0, RCL_ROS_TIME}, odom_at_{0, 0, RCL_ROS_TIME},
      scan_at_{0, 0, RCL_ROS_TIME}, goal_sent_at_{0, 0, RCL_ROS_TIME},
      still_since_{0, 0, RCL_ROS_TIME}, command_at_{0, 0, RCL_ROS_TIME};
  geometry_msgs::msg::Twist command_;
  bool stopped_ = false, nav_pending_ = false, cancel_sent_ = false,
       cancel_ack_ = false, nav_terminal_ = true;
  bool plan_pending_ = false, ext_pending_ = false, ext_ok_ = false,
       ext_done_ = false, external_cancel_ = false;
  uint64_t epoch_ = 0, global_ack_ = 0, local_ack_ = 0;
  double small_, large_, radius_, clearance_, timeout_, fresh_, dwell_, linear_,
      angular_, stand_min_, stand_max_, obstacle_, min_score_,
      zone_update_distance_, clock_skew_;
  double nearest_ = 0;
  template <class T, class Fn>
  void sub(const std::string &param, Fn fn, rclcpp::QoS qos = rclcpp::QoS(10)) {
    subs_.push_back(
        create_subscription<T>(declare_parameter<std::string>(param), qos,
                               [fn](typename T::ConstSharedPtr m) { fn(m); }));
  }
  void state(const std::string &s, const std::string &why = "") {
    state_ = s;
    reason_ = why;
    entered_ = now();
    std_msgs::msg::String m;
    m.data = s + "|" + why;
    status_->publish(m);
    RCLCPP_INFO(get_logger(), "%s", m.data.c_str());
  }
  bool moving() const {
    return state_ == "NAVIGATING" || state_ == "FOLLOW_NEW_PATH";
  }
  bool fresh(rclcpp::Time t) {
    auto dt = (now() - t).seconds();
    return t.nanoseconds() > 0 && dt >= -clock_skew_ && dt < fresh_;
  }
  void fail(const std::string &why) {
    state("SAFE_STOP", why);
    command_ = geometry_msgs::msg::Twist();
    if (nav_goal_)
      nav_->async_cancel_goal(nav_goal_);
    if (ext_goal_)
      extinguisher_->async_cancel_goal(ext_goal_);
    if (mission_) {
      auto r = std::make_shared<N::Result>();
      if (mission_->is_canceling())
        mission_->canceled(r);
      else
        mission_->abort(r);
      mission_.reset();
    }
    // Keep SAFE_STOP latched. Restart after diagnosing; no automatic motion on
    // failure.
  }
  bool path_safe(const nav_msgs::msg::Path &path) {
    if (path.header.frame_id != map_frame_ || path.poses.empty())
      return false;
    for (size_t i = 0; i < path.poses.size(); ++i) {
      auto a = path.poses[i].pose.position,
           b = path.poses[i ? i - 1 : 0].pose.position;
      if (!segment_safe(a.x, a.y, b.x, b.y, circles_))
        return false;
    }
    return true;
  }
  void dispatch() {
    if (!mission_ || !nav_->action_server_is_ready()) {
      fail("Nav2 action unavailable");
      return;
    }
    const auto token = ++epoch_;
    nav_pending_ = true;
    nav_terminal_ = false;
    cancel_sent_ = false;
    cancel_ack_ = false;
    goal_sent_at_ = now();
    command_ = geometry_msgs::msg::Twist();
    command_at_ = rclcpp::Time(0, 0, RCL_ROS_TIME);
    auto goal = *mission_->get_goal();
    goal.behavior_tree.clear(); // use repository default replanning BT
    rclcpp_action::Client<N>::SendGoalOptions opts;
    opts.goal_response_callback = [this, token](NH::SharedPtr h) {
      if (token != epoch_ || !mission_ || state_ == "SAFE_STOP") {
        if (h)
          nav_->async_cancel_goal(h);
        return;
      }
      nav_pending_ = false;
      nav_goal_ = h;
      if (!h) {
        nav_terminal_ = true;
        fail("Nav2 goal rejected");
      }
    };
    opts.feedback_callback = [this, token](auto, auto f) {
      if (token == epoch_ && mission_)
        mission_->publish_feedback(std::make_shared<N::Feedback>(*f));
    };
    opts.result_callback = [this, token](const NH::WrappedResult &r) {
      if (token != epoch_)
        return;
      nav_terminal_ = true;
      nav_goal_.reset();
      if (!moving())
        return; // STOPPING requires cancellation acknowledgement AND terminal
                // result
      if (r.code == rclcpp_action::ResultCode::SUCCEEDED) {
        if (mission_) {
          mission_->succeed(std::make_shared<N::Result>(*r.result));
          mission_.reset();
        }
        state("NAVIGATING", "goal reached");
      } else
        fail("Nav2 action failed");
    };
    nav_->async_send_goal(goal, opts);
  }
  void stop_for_fire() {
    state("FIRE_DETECTED", fire_.id);
    state("CLASSIFY_FIRE");
    const auto size = classify(fire_.width_m, fire_.size_valid, small_, large_);
    state(size == Size::Small   ? "SMALL_FIRE"
          : size == Size::Large ? "LARGE_FIRE"
                                : "UNKNOWN");
    state("STOPPING");
    cancel_ack_ = nav_terminal_ && !nav_pending_;
    cancel_sent_ = false;
  }
  void avoid() {
    if (!fire_.position_valid) {
      fail("fire position unavailable");
      return;
    }
    circles_.push_back(
        {fire_.position.x, fire_.position.y, radius_ + clearance_});
    state("UPDATE_DANGER_ZONE",
          path_safe(last_path_)
              ? "previous path clear"
              : "previous path intersects hazard or is unavailable");
    zones_.header.frame_id = map_frame_;
    zones_.header.stamp = now();
    ++zones_.revision;
    zones_.centers.push_back(fire_.position);
    zones_.radii.push_back(radius_ + clearance_);
    zone_pub_->publish(zones_);
  }
  void plan() {
    if (!planner_->action_server_is_ready()) {
      fail("planner unavailable");
      return;
    }
    state("REPLANNING");
    plan_pending_ = true;
    const auto token = epoch_;
    P::Goal goal;
    goal.goal = mission_->get_goal()->pose;
    goal.planner_id = "GridBased";
    goal.use_start = false;
    rclcpp_action::Client<P>::SendGoalOptions opts;
    opts.goal_response_callback = [this, token](auto h) {
      if (token == epoch_ && state_ == "REPLANNING" && !h)
        fail("planner rejected");
    };
    opts.result_callback = [this, token](const auto &r) {
      if (token != epoch_ || state_ != "REPLANNING")
        return;
      plan_pending_ = false;
      if (r.code != rclcpp_action::ResultCode::SUCCEEDED ||
          !path_safe(r.result->path)) {
        fail("no safe path");
        return;
      }
      double length = 0;
      for (size_t i = 1; i < r.result->path.poses.size(); ++i) {
        auto a = r.result->path.poses[i - 1].pose.position,
             b = r.result->path.poses[i].pose.position;
        length += std::hypot(a.x - b.x, a.y - b.y);
      }
      state("FOLLOW_NEW_PATH", "path_length_m=" + std::to_string(length));
      dispatch();
    };
    planner_->async_send_goal(goal, opts);
  }
  void extinguish() {
    if (!fresh(rclcpp::Time(fire_.header.stamp))) {
      state("AVOID_OR_STOP", "fire estimate expired before extinguishing");
      avoid();
      return;
    }
    try {
      const auto t =
          tf_.lookupTransform(map_frame_, base_frame_, tf2::TimePointZero);
      const auto d = std::hypot(t.transform.translation.x - fire_.position.x,
                                t.transform.translation.y - fire_.position.y);
      if (!fire_.position_valid || d < stand_min_ || d > stand_max_ ||
          !fresh(scan_at_) || nearest_ < obstacle_) {
        state("AVOID_OR_STOP",
              "unsafe extinguishing: distance_m=" + std::to_string(d) +
                  ", nearest_obstacle_m=" + std::to_string(nearest_) +
                  ", scan_fresh=" + std::to_string(fresh(scan_at_)));
        avoid();
        return;
      }
    } catch (const tf2::TransformException &) {
      fail("standoff TF unavailable");
      return;
    }
    if (!extinguisher_->action_server_is_ready()) {
      state("AVOID_OR_STOP", "extinguisher unavailable");
      avoid();
      return;
    }
    state("EXTINGUISHING");
    ext_done_ = false;
    ext_ok_ = false;
    ext_pending_ = true;
    const auto token = epoch_;
    E::Goal g;
    g.fire_id = fire_.id;
    g.target.header = fire_.header;
    g.target.point = fire_.position;
    rclcpp_action::Client<E>::SendGoalOptions opts;
    opts.goal_response_callback = [this, token](EH::SharedPtr h) {
      if (token != epoch_ || state_ != "EXTINGUISHING") {
        if (h)
          extinguisher_->async_cancel_goal(h);
        return;
      }
      ext_pending_ = false;
      ext_goal_ = h;
      if (!h) {
        ext_done_ = true;
        ext_ok_ = false;
      }
    };
    opts.result_callback = [this, token](const auto &r) {
      if (token != epoch_ || (state_ != "EXTINGUISHING" && !external_cancel_))
        return;
      ext_done_ = true;
      ext_ok_ =
          r.code == rclcpp_action::ResultCode::SUCCEEDED && r.result->success;
      ext_goal_.reset();
    };
    extinguisher_->async_send_goal(g, opts);
  }
  void tick() {
    const auto zero = geometry_msgs::msg::Twist();
    velocity_->publish(moving() && mission_ && nav_goal_ && !nav_pending_ &&
                               queue_.empty() && fresh(command_at_) &&
                               fresh(odom_at_)
                           ? command_
                           : zero);
    if (state_ == "SAFE_STOP")
      return;
    if (nav_pending_ && (now() - goal_sent_at_).seconds() > timeout_) {
      fail("Nav2 goal acknowledgement timeout");
      return;
    }
    if (moving() && mission_ && !circles_.empty()) {
      try {
        auto t =
            tf_.lookupTransform(map_frame_, base_frame_, tf2::TimePointZero);
        if (!segment_safe(t.transform.translation.x, t.transform.translation.y,
                          t.transform.translation.x, t.transform.translation.y,
                          circles_)) {
          fail("robot entered danger zone");
          return;
        }
      } catch (const tf2::TransformException &) {
        fail("danger monitoring TF unavailable");
        return;
      }
    }
    if (external_cancel_) {
      if (nav_goal_ && !cancel_sent_) {
        cancel_sent_ = true;
        nav_->async_cancel_goal(nav_goal_);
      }
      if (ext_goal_)
        extinguisher_->async_cancel_goal(ext_goal_);
      if (nav_terminal_ && !nav_pending_ && !ext_goal_ && !ext_pending_ &&
          stopped_ && fresh(odom_at_)) {
        if (mission_)
          mission_->canceled(std::make_shared<N::Result>());
        mission_.reset();
        external_cancel_ = false;
        state("NAVIGATING", "mission canceled");
      } else if ((now() - entered_).seconds() > timeout_)
        fail("mission cancellation timeout");
      return;
    }
    if (moving() && mission_ && !queue_.empty()) {
      fire_ = queue_.front();
      queue_.pop_front();
      stop_for_fire();
    }
    if (!moving() && (now() - entered_).seconds() > timeout_) {
      if (state_ == "EXTINGUISHING") {
        if (ext_goal_)
          extinguisher_->async_cancel_goal(ext_goal_);
        state("AVOID_OR_STOP", "extinguishing timeout");
        avoid();
        fail("extinguishing timeout; motion inhibited until device shutdown is "
             "confirmed");
      } else
        fail(state_ + " timeout");
      return;
    }
    if (state_ == "STOPPING") {
      if (nav_goal_ && !cancel_sent_) {
        cancel_sent_ = true;
        nav_->async_cancel_goal(nav_goal_, [this](auto r) {
          cancel_ack_ = !r->goals_canceling.empty();
        });
      }
      if (nav_terminal_ && !nav_pending_ && !nav_goal_)
        cancel_ack_ = true;
      if (cancel_ack_ && nav_terminal_ && !nav_pending_ && stopped_ &&
          fresh(odom_at_) && (now() - still_since_).seconds() >= dwell_) {
        auto size = classify(fire_.width_m, fire_.size_valid, small_, large_);
        if (size == Size::Small)
          extinguish();
        else
          avoid();
      }
    } else if (state_ == "EXTINGUISHING" && ext_done_) {
      state("VERIFY_RESULT", ext_ok_ ? "simulator confirmed extinguished"
                                     : "extinguishing failed");
      if (ext_ok_) {
        state("RESUME_NAVIGATION");
        state("NAVIGATING", "resumed after extinguishing");
        dispatch();
      } else {
        state("AVOID_OR_STOP", "extinguishing failed");
        avoid();
      }
    } else if (state_ == "UPDATE_DANGER_ZONE" &&
               global_ack_ >= zones_.revision && local_ack_ >= zones_.revision)
      plan();
  }

public:
  FireManager() : Node("fire_manager"), tf_(get_clock()), listener_(tf_) {
    if (!get_parameter("use_sim_time").as_bool())
      throw std::runtime_error("fire_manager is simulation-only");
    small_ = declare_parameter<double>("small_width_m");
    large_ = declare_parameter<double>("large_width_m");
    radius_ = declare_parameter<double>("danger_radius_m");
    clearance_ = declare_parameter<double>("robot_clearance_m");
    timeout_ = declare_parameter<double>("phase_timeout_s");
    fresh_ = declare_parameter<double>("freshness_s");
    dwell_ = declare_parameter<double>("stop_dwell_s");
    linear_ = declare_parameter<double>("stop_linear_mps");
    angular_ = declare_parameter<double>("stop_angular_rps");
    stand_min_ = declare_parameter<double>("standoff_min_m");
    stand_max_ = declare_parameter<double>("standoff_max_m");
    obstacle_ = declare_parameter<double>("obstacle_clearance_m");
    min_score_ = declare_parameter<double>("min_confidence");
    zone_update_distance_ = declare_parameter<double>("zone_update_distance_m");
    clock_skew_ = declare_parameter<double>("clock_skew_tolerance_s");
    map_frame_ = declare_parameter<std::string>("map_frame");
    base_frame_ = declare_parameter<std::string>("base_frame");
    classify(.1, true, small_, large_);
    if (radius_ <= 0 || clearance_ < 0 || timeout_ <= 0 || fresh_ <= 0 ||
        dwell_ <= 0 || stand_max_ <= stand_min_)
      throw std::runtime_error("invalid safety parameters");
    status_ = create_publisher<std_msgs::msg::String>(
        declare_parameter<std::string>("state_topic"),
        rclcpp::QoS(20).transient_local());
    velocity_ = create_publisher<geometry_msgs::msg::Twist>(
        declare_parameter<std::string>("output_velocity_topic"), 10);
    zone_pub_ = create_publisher<limo_fire_msgs::msg::DangerZones>(
        declare_parameter<std::string>("zones_topic"),
        rclcpp::QoS(1).transient_local());
    nav_ = rclcpp_action::create_client<N>(
        this, declare_parameter<std::string>("backend_action"));
    planner_ = rclcpp_action::create_client<P>(
        this, declare_parameter<std::string>("planner_action"));
    extinguisher_ = rclcpp_action::create_client<E>(
        this, declare_parameter<std::string>("extinguish_action"));
    sub<geometry_msgs::msg::Twist>("input_velocity_topic", [this](auto m) {
      command_ = *m;
      command_at_ = now();
    });
    sub<nav_msgs::msg::Path>("planned_path_topic", [this](auto m) {
      last_path_ = *m;
      if (moving() && mission_ && !circles_.empty() && !path_safe(*m))
        fail("Nav2 published unsafe path");
    });
    sub<nav_msgs::msg::Odometry>(
        "odom_topic",
        [this](auto m) {
          odom_at_ = rclcpp::Time(m->header.stamp);
          const auto &v = m->twist.twist;
          bool s = std::hypot(v.linear.x, v.linear.y) <= linear_ &&
                   std::abs(v.angular.z) <= angular_;
          if (s && !stopped_)
            still_since_ = now();
          stopped_ = s;
        },
        rclcpp::SensorDataQoS());
    sub<sensor_msgs::msg::LaserScan>(
        "scan_topic",
        [this](auto m) {
          scan_at_ = rclcpp::Time(m->header.stamp);
          nearest_ = 0;
          bool valid = false;
          double d = std::numeric_limits<double>::infinity();
          for (auto r : m->ranges) {
            if (std::isfinite(r) && r >= m->range_min && r <= m->range_max) {
              valid = true;
              d = std::min(d, double(r));
            }
          }
          if (valid)
            nearest_ = d;
        },
        rclcpp::SensorDataQoS());
    sub<std_msgs::msg::UInt64>(
        "global_ack_topic", [this](auto m) { global_ack_ = m->data; },
        rclcpp::QoS(1).transient_local());
    sub<std_msgs::msg::UInt64>(
        "local_ack_topic", [this](auto m) { local_ack_ = m->data; },
        rclcpp::QoS(1).transient_local());
    sub<F>("fire_topic", [this](auto m) {
      if (m->id.empty() || m->header.frame_id != map_frame_ ||
          !fresh(rclcpp::Time(m->header.stamp)) ||
          !std::isfinite(m->confidence) || m->confidence < min_score_ ||
          m->confidence > 1) {
        if (mission_)
          fail("invalid fire observation");
        return;
      }
      if (m->position_valid && !std::isfinite(m->position.x + m->position.y)) {
        if (mission_)
          fail("invalid fire coordinates");
        return;
      }
      // A camera may detect fire before the depth sensor can localize it.
      // Keep following the Nav2 goal until object_mapper publishes a real pose;
      // an unlocalized hint is not safe input for danger zones or extinguishing.
      if (m->id == "unlocalized_fire" && !m->position_valid) {
        RCLCPP_WARN_THROTTLE(
            get_logger(), *get_clock(), 5000,
            "Fire detected without a usable depth pose; waiting for localization");
        return;
      }
      auto size = classify(m->width_m, m->size_valid, small_, large_);
      auto it = seen_.find(m->id);
      if (state_ == "STOPPING" && m->id == fire_.id)
        fire_ = *m;
      bool moved = it != seen_.end() && m->position_valid &&
                   (!last_seen_[m->id].position_valid ||
                    std::hypot(m->position.x - last_seen_[m->id].position.x,
                               m->position.y - last_seen_[m->id].position.y) >
                        zone_update_distance_);
      if (it != seen_.end() &&
          !(it->second == Size::Small && size != Size::Small) && !moved)
        return;
      seen_[m->id] = size;
      last_seen_[m->id] = *m;
      if (!moving() && m->id == fire_.id && state_ != "SAFE_STOP" &&
          size == Size::Large) {
        fire_ = *m;
        if (state_ == "EXTINGUISHING") {
          avoid();
          fail("fire escalated during extinguishing; device cancellation "
               "requested");
          return;
        }
        if (state_ == "STOPPING")
          return;
      }
      if (queue_.size() >= 100) {
        fail("fire queue overflow");
        return;
      }
      queue_.push_back(*m);
    });
    server_ = rclcpp_action::create_server<N>(
        this, declare_parameter<std::string>("mission_action"),
        [this](auto, auto g) {
          return !mission_ && state_ != "SAFE_STOP" &&
                         g->pose.header.frame_id == map_frame_ &&
                         std::isfinite(g->pose.pose.position.x +
                                       g->pose.pose.position.y)
                     ? rclcpp_action::GoalResponse::ACCEPT_AND_EXECUTE
                     : rclcpp_action::GoalResponse::REJECT;
        },
        [this](auto) {
          external_cancel_ = true;
          state("STOPPING", "client canceled mission");
          return rclcpp_action::CancelResponse::ACCEPT;
        },
        [this](auto h) {
          mission_ = h;
          dispatch();
        });
    timer_ = create_wall_timer(
        std::chrono::duration<double>(declare_parameter<double>("tick_s")),
        [this] { tick(); });
    state("NAVIGATING", "ready");
  }
};
} // namespace limo_fire_bt
#ifndef LIMO_FIRE_MANAGER_NO_MAIN
int main(int argc, char **argv) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<limo_fire_bt::FireManager>());
  rclcpp::shutdown();
}
#endif
