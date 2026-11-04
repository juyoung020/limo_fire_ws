#define LIMO_FIRE_MANAGER_NO_MAIN
#include "../src/fire_manager.cpp"
#include <iostream>
#include <rosgraph_msgs/msg/clock.hpp>
#include <tf2_ros/static_transform_broadcaster.h>
#include <thread>
class Fixture : public rclcpp::Node {
  using N = nav2_msgs::action::NavigateToPose;
  using P = nav2_msgs::action::ComputePathToPose;
  using E = limo_fire_msgs::action::Extinguish;
  using NH = rclcpp_action::ServerGoalHandle<N>;
  rclcpp_action::Server<N>::SharedPtr nav_;
  rclcpp_action::Server<P>::SharedPtr planner_;
  rclcpp_action::Server<E>::SharedPtr ext_;
  rclcpp_action::Client<N>::SharedPtr client_;
  std::shared_ptr<NH> active_;
  std::shared_ptr<rclcpp_action::ServerGoalHandle<E>> ext_active_;
  rclcpp::Publisher<rosgraph_msgs::msg::Clock>::SharedPtr clock_;
  rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr odom_;
  rclcpp::Publisher<sensor_msgs::msg::LaserScan>::SharedPtr scan_;
  rclcpp::Publisher<limo_fire_msgs::msg::FireObservation>::SharedPtr fire_;
  rclcpp::Publisher<std_msgs::msg::UInt64>::SharedPtr ga_, la_;
  rclcpp::Subscription<limo_fire_msgs::msg::DangerZones>::SharedPtr zones_;
  rclcpp::Subscription<std_msgs::msg::String>::SharedPtr state_;
  std::unique_ptr<tf2_ros::StaticTransformBroadcaster> tf_;
  std::string scenario_;
  bool sent_ = false, injected_ = false, resumed_ = false, safe_ = false,
       reached_ = false, stopping_ = false;
  std::chrono::steady_clock::time_point start_ =
      std::chrono::steady_clock::now();
  double fire_time_ = 0, active_time_ = 0;
  int nav_count_ = 0, ext_count_ = 0, plans_ = 0;

public:
  void cleanup() {
    if (ext_active_ && ext_active_->is_active()) {
      auto r = std::make_shared<E::Result>();
      if (ext_active_->is_canceling())
        ext_active_->canceled(r);
      else
        ext_active_->abort(r);
    }
    ext_active_.reset();
    if (active_ && active_->is_active())
      active_->abort(std::make_shared<N::Result>());
    active_.reset();
  }
  bool done = false, passed = false;
  Fixture() : Node("fixture") {
    scenario_ = declare_parameter<std::string>("scenario");
    clock_ = create_publisher<rosgraph_msgs::msg::Clock>("/clock", 10);
    odom_ = create_publisher<nav_msgs::msg::Odometry>("/odom", 10);
    scan_ = create_publisher<sensor_msgs::msg::LaserScan>("/scan", 10);
    fire_ = create_publisher<limo_fire_msgs::msg::FireObservation>(
        "/fire/observations", 10);
    ga_ = create_publisher<std_msgs::msg::UInt64>(
        "/fire/global_applied", rclcpp::QoS(1).transient_local());
    la_ = create_publisher<std_msgs::msg::UInt64>(
        "/fire/local_applied", rclcpp::QoS(1).transient_local());
    zones_ = create_subscription<limo_fire_msgs::msg::DangerZones>(
        "/fire/danger_zones", rclcpp::QoS(1).transient_local(),
        [this](limo_fire_msgs::msg::DangerZones::ConstSharedPtr m) {
          std_msgs::msg::UInt64 a;
          a.data = m->revision;
          ga_->publish(a);
          la_->publish(a);
        });
    state_ = create_subscription<std_msgs::msg::String>(
        "/fire/state", 20, [this](std_msgs::msg::String::ConstSharedPtr m) {
          stopping_ = m->data.find("STOPPING|") == 0;
          if (m->data.find("SAFE_STOP|") == 0)
            safe_ = true;
          if (m->data.find("RESUME_NAVIGATION|") == 0)
            resumed_ = true;
        });
    tf_ = std::make_unique<tf2_ros::StaticTransformBroadcaster>(this);
    geometry_msgs::msg::TransformStamped t;
    t.header.frame_id = "map";
    t.child_frame_id = "base_link";
    t.transform.rotation.w = 1;
    tf_->sendTransform(t);
    nav_ = rclcpp_action::create_server<N>(
        this, "/fire/nav2_navigate_to_pose",
        [](auto, auto) {
          return rclcpp_action::GoalResponse::ACCEPT_AND_EXECUTE;
        },
        [](auto) { return rclcpp_action::CancelResponse::ACCEPT; },
        [this](auto h) {
          active_ = h;
          active_time_ = now().seconds();
          ++nav_count_;
        });
    planner_ = rclcpp_action::create_server<P>(
        this, "/compute_path_to_pose",
        [](auto, auto) {
          return rclcpp_action::GoalResponse::ACCEPT_AND_EXECUTE;
        },
        [](auto) { return rclcpp_action::CancelResponse::ACCEPT; },
        [this](auto h) {
          ++plans_;
          auto r = std::make_shared<P::Result>();
          if (scenario_ == "no_path") {
            h->abort(r);
            return;
          }
          r->path.header.frame_id = "map";
          for (auto xy : std::vector<std::pair<double, double>>{
                   {0, 0}, {0, 2}, {4, 2}, {6, 0}}) {
            geometry_msgs::msg::PoseStamped p;
            p.header = r->path.header;
            p.pose.position.x = xy.first;
            p.pose.position.y = xy.second;
            p.pose.orientation.w = 1;
            r->path.poses.push_back(p);
          }
          h->succeed(r);
        });
    ext_ = rclcpp_action::create_server<E>(
        this, "/fire/mock_extinguish",
        [](auto, auto) {
          return rclcpp_action::GoalResponse::ACCEPT_AND_EXECUTE;
        },
        [](auto) { return rclcpp_action::CancelResponse::ACCEPT; },
        [this](auto h) {
          ++ext_count_;
          if (active_ || now().seconds() - fire_time_ < .5) {
            done = true;
            passed = false;
            return;
          }
          if (scenario_ == "ext_timeout") {
            ext_active_ = h;
            return;
          }
          auto r = std::make_shared<E::Result>();
          r->success = scenario_ != "ext_failure";
          if (r->success)
            h->succeed(r);
          else
            h->abort(r);
        });
    client_ = rclcpp_action::create_client<N>(this, "/navigate_to_pose");
  }
  void step() {
    const double elapsed =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - start_)
            .count();
    rosgraph_msgs::msg::Clock c;
    c.clock = rclcpp::Time(int64_t((10 + elapsed * 3) * 1e9));
    clock_->publish(c);
    nav_msgs::msg::Odometry o;
    o.header.stamp = c.clock;
    o.header.frame_id = "odom";
    if (injected_ && now().seconds() - fire_time_ < .3)
      o.twist.twist.linear.x = .1;
    odom_->publish(o);
    sensor_msgs::msg::LaserScan s;
    s.header.stamp = c.clock;
    s.range_min = .1;
    s.range_max = 5;
    s.ranges.assign(36, 2.0);
    scan_->publish(s);
    if (!sent_ && elapsed > .5 && client_->action_server_is_ready()) {
      sent_ = true;
      N::Goal g;
      g.pose.header.frame_id = "map";
      g.pose.pose.position.x = 6;
      g.pose.pose.orientation.w = 1;
      rclcpp_action::Client<N>::SendGoalOptions opts;
      opts.result_callback = [this](const auto &r) {
        reached_ = r.code == rclcpp_action::ResultCode::SUCCEEDED;
      };
      client_->async_send_goal(g, opts);
    }
    if (active_ && active_->is_canceling()) {
      active_->canceled(std::make_shared<N::Result>());
      active_.reset();
    }
    if (active_ && scenario_ == "nav_failure") {
      active_->abort(std::make_shared<N::Result>());
      active_.reset();
      injected_ = true;
    }
    if (active_ && !injected_ && now().seconds() - active_time_ > .2) {
      injected_ = true;
      fire_time_ = now().seconds();
      publish_fire();
    }
    // A continuing camera stream keeps estimates fresh while cancellation
    // completes.
    if ((scenario_ == "duplicate" || stopping_) && injected_ && !reached_)
      publish_fire();
    if (active_ && nav_count_ >= 2 && now().seconds() - active_time_ > 1) {
      active_->succeed(std::make_shared<N::Result>());
      active_.reset();
    }
    const bool expect_success =
        scenario_ == "small_success" || scenario_ == "duplicate" ||
        scenario_ == "large_success" || scenario_ == "ext_failure";
    if ((expect_success && reached_) || (!expect_success && safe_)) {
      done = true;
      passed = expect_success
                   ? nav_count_ == 2 &&
                         (scenario_ == "large_success"
                              ? ext_count_ == 0 && plans_ == 1
                              : ext_count_ == 1 &&
                                    (scenario_ == "ext_failure" ? plans_ == 1
                                                                : resumed_))
                   : true;
      if (scenario_ == "no_path")
        passed = passed && plans_ == 1 && ext_count_ == 0;
      if (scenario_ == "ext_failure" || scenario_ == "ext_timeout")
        passed = passed && ext_count_ == 1;
    }
    if (elapsed > 12) {
      done = true;
      passed = false;
    }
  }
  void publish_fire() {
    limo_fire_msgs::msg::FireObservation f;
    f.header.frame_id = "map";
    f.header.stamp = now();
    f.id = "test_fire";
    f.position_valid = true;
    f.size_valid = true;
    f.confidence = .9;
    bool large = scenario_ == "large_success" || scenario_ == "no_path";
    f.position.x = large ? 3 : 1.5;
    f.width_m = large ? .7 : .1;
    if (scenario_ == "bad_input")
      f.confidence = std::numeric_limits<double>::quiet_NaN();
    if (scenario_ == "depth_missing") {
      f.position_valid = false;
      f.size_valid = false;
    }
    fire_->publish(f);
  }
};
int main(int argc, char **argv) {
  rclcpp::init(argc, argv);
  auto manager = std::make_shared<limo_fire_bt::FireManager>();
  auto fixture = std::make_shared<Fixture>();
  rclcpp::executors::SingleThreadedExecutor ex;
  ex.add_node(manager);
  ex.add_node(fixture);
  while (rclcpp::ok() && !fixture->done) {
    fixture->step();
    ex.spin_some();
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
  bool pass = fixture->passed;
  fixture->cleanup();
  std::cout << (pass ? "PASS" : "FAIL")
            << " ROS action protocol fixture (not Webots)\n";
  rclcpp::shutdown();
  return pass ? 0 : 1;
}
