// approach_aim·launcher_terminal 실행 파일을 띄워 가짜 로봇(이 시험 안: Nav2 액션 서버·TF·costmap·오도메트리·표적·LIMO 상태)과 맞물려 본다.
// 마지막 시험은 펌웨어 원본 호스트 시뮬(/tmp/launcher_host_sim, firmware/limo_fire_esp32/test/run.sh)과 실제 esp32_bridge 까지 이어
// 터미널 a → 조준 → 다리 → 펌웨어 레버 PWM 을 확인한다(예전 Python FirmwareHostSimAutoFireRosTest 와 같은 흐름). 없으면 건너뜀.
// 실물 로봇·서보·공은 없다. 차체는 Nav2 목표 자리로 순간 이동한다.
#include <fcntl.h>
#include <gtest/gtest.h>
#include <signal.h>
#include <sys/wait.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <functional>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include "ament_index_cpp/get_package_prefix.hpp"
#include "geometry_msgs/msg/transform_stamped.hpp"
#include "geometry_msgs/msg/twist.hpp"
#include "limo_fire_msgs/action/launch.hpp"
#include "limo_fire_msgs/msg/fire_observation.hpp"
#include "limo_msgs/msg/limo_status.hpp"
#include "nav2_msgs/action/compute_path_to_pose.hpp"
#include "nav2_msgs/action/navigate_to_pose.hpp"
#include "nav_msgs/msg/occupancy_grid.hpp"
#include "nav_msgs/msg/odometry.hpp"
#include "rclcpp/rclcpp.hpp"
#include "rclcpp_action/rclcpp_action.hpp"
#include "std_msgs/msg/string.hpp"
#include "std_srvs/srv/trigger.hpp"
#include "tf2_ros/static_transform_broadcaster.h"
#include "tf2_ros/transform_broadcaster.h"

using namespace std::chrono_literals;
using Launch = limo_fire_msgs::action::Launch;
using Nav = nav2_msgs::action::NavigateToPose;
using Plan = nav2_msgs::action::ComputePathToPose;

namespace {

pid_t spawn(const std::vector<std::string>& argv, const std::string& out, int stdin_fd = -1) {
  const pid_t pid = fork();
  if (pid == 0) {
    setpgid(0, 0);
    if (stdin_fd >= 0) dup2(stdin_fd, 0);
    const int fd = ::open(out.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if (fd >= 0) {
      dup2(fd, 1);
      dup2(fd, 2);
    }
    std::vector<char*> a;
    for (const auto& s : argv) a.push_back(const_cast<char*>(s.c_str()));
    a.push_back(nullptr);
    execv(a[0], a.data());
    _exit(127);
  }
  return pid;
}
bool wait_exit(pid_t pid, double seconds) {
  const auto end = std::chrono::steady_clock::now() + std::chrono::duration<double>(seconds);
  while (std::chrono::steady_clock::now() < end) {
    if (waitpid(pid, nullptr, WNOHANG) == pid) return true;
    std::this_thread::sleep_for(50ms);
  }
  return false;
}
void stop(pid_t& pid) {
  if (pid <= 0) return;
  for (int sig : {SIGINT, SIGTERM, SIGKILL}) {
    kill(-pid, sig);
    if (wait_exit(pid, 4.0)) break;
  }
  pid = -1;
}
std::string slurp(const std::string& path) {
  std::ifstream f(path);
  std::stringstream s;
  s << f.rdbuf();
  return s.str();
}

// ---- 가짜 로봇
class FakeRobot : public rclcpp::Node {
 public:
  bool delay_navigation = false;
  double nav_yaw_bias_deg = 0.0;
  bool fake_launcher = false;
  std::atomic<int> nav_goals{0}, nav_cancels{0}, launch_goals{0}, nonzero_yaw{0};

  FakeRobot() : Node("fake_approach_aim_robot") {
    static_tf_ = std::make_shared<tf2_ros::StaticTransformBroadcaster>(*this);
    geometry_msgs::msg::TransformStamped m;
    m.header.stamp = now();
    m.header.frame_id = "base_link";
    m.child_frame_id = "launcher_pivot";
    m.transform.rotation.w = 1.0;
    static_tf_->sendTransform(m);
    tf_ = std::make_shared<tf2_ros::TransformBroadcaster>(*this);
    const auto latched = rclcpp::QoS(1).transient_local().reliable();
    grid_pub_ = create_publisher<nav_msgs::msg::OccupancyGrid>("/global_costmap/costmap", latched);
    fire_pub_ = create_publisher<limo_fire_msgs::msg::FireObservation>("/fire/observations", 10);
    odom_pub_ = create_publisher<nav_msgs::msg::Odometry>("/odometry/filtered", 10);
    base_pub_ = create_publisher<limo_msgs::msg::LimoStatus>("/limo_status", 10);
    yaw_sub_ = create_subscription<geometry_msgs::msg::Twist>("/cmd_vel", 10, [this](geometry_msgs::msg::Twist::SharedPtr t) {
      std::lock_guard<std::mutex> l(mu_);
      rate_ = t->angular.z;
      if (t->angular.z != 0.0) ++nonzero_yaw;
    });
    group_ = create_callback_group(rclcpp::CallbackGroupType::Reentrant);
    planner_ = rclcpp_action::create_server<Plan>(
        this, "/compute_path_to_pose", [](auto, auto) { return rclcpp_action::GoalResponse::ACCEPT_AND_EXECUTE; },
        [](auto) { return rclcpp_action::CancelResponse::ACCEPT; },
        [this](std::shared_ptr<rclcpp_action::ServerGoalHandle<Plan>> gh) {
          auto res = std::make_shared<Plan::Result>();
          res->path.header.frame_id = "map";
          res->path.header.stamp = now();
          res->path.poses.push_back(gh->get_goal()->goal);
          gh->succeed(res);
        },
        rcl_action_server_get_default_options(), group_);
    nav_ = rclcpp_action::create_server<Nav>(
        this, "/navigate_to_pose",
        [this](auto, auto) {
          ++nav_goals;
          return rclcpp_action::GoalResponse::ACCEPT_AND_EXECUTE;
        },
        [this](auto) {
          ++nav_cancels;
          return rclcpp_action::CancelResponse::ACCEPT;
        },
        [this](std::shared_ptr<rclcpp_action::ServerGoalHandle<Nav>> gh) { std::thread([this, gh] { navigate(gh); }).detach(); },
        rcl_action_server_get_default_options(), group_);
    sensor_timer_ = create_wall_timer(50ms, [this] { publish_sensors(); }, group_);
    grid_timer_ = create_wall_timer(200ms, [this] { publish_grid(); }, group_);
  }

  // approach_aim 이 auto_fire 일 때 쓰는 가짜 ESP32 다리 (발사 횟수만 셈)
  void start_fake_launcher() {
    fake_launcher = true;
    state_pub_ = create_publisher<std_msgs::msg::String>("/launcher/state", rclcpp::QoS(1).transient_local().reliable());
    std_msgs::msg::String s;
    s.data = "ready";
    state_pub_->publish(s);
    launch_srv_ = rclcpp_action::create_server<Launch>(
        this, "/launcher/launch",
        [this](auto, auto) {
          ++launch_goals;
          return rclcpp_action::GoalResponse::ACCEPT_AND_EXECUTE;
        },
        [](auto) { return rclcpp_action::CancelResponse::ACCEPT; },
        [](std::shared_ptr<rclcpp_action::ServerGoalHandle<Launch>> gh) {
          std::thread([gh] {
            std::this_thread::sleep_for(300ms);
            auto r = std::make_shared<Launch::Result>();
            r->success = true;
            r->message = "가짜 다리: 장전+발사 끝";
            gh->succeed(r);
          }).detach();
        },
        rcl_action_server_get_default_options(), group_);
  }

 private:
  void navigate(std::shared_ptr<rclcpp_action::ServerGoalHandle<Nav>> gh) {
    if (delay_navigation) {
      const auto end = std::chrono::steady_clock::now() + 5s;
      while (rclcpp::ok() && std::chrono::steady_clock::now() < end) {
        if (gh->is_canceling()) {
          gh->canceled(std::make_shared<Nav::Result>());
          return;
        }
        std::this_thread::sleep_for(20ms);
      }
      gh->abort(std::make_shared<Nav::Result>());
      return;
    }
    const auto& p = gh->get_goal()->pose.pose;
    double yaw = std::atan2(2 * p.orientation.w * p.orientation.z, 1 - 2 * p.orientation.z * p.orientation.z);
    {
      std::lock_guard<std::mutex> l(mu_);
      if (++navigated_ == 1) yaw += nav_yaw_bias_deg * M_PI / 180.0;
      x_ = p.position.x;
      y_ = p.position.y;
      yaw_ = yaw;
    }
    gh->succeed(std::make_shared<Nav::Result>());
  }
  void publish_grid() {
    nav_msgs::msg::OccupancyGrid g;
    g.header.frame_id = "map";
    g.header.stamp = now();
    g.info.resolution = 0.05;
    g.info.width = g.info.height = 200;
    g.info.origin.position.x = g.info.origin.position.y = -5.0;
    g.info.origin.orientation.w = 1.0;
    g.data.assign(40000, 0);
    grid_pub_->publish(g);
  }
  void publish_sensors() {
    const auto t = now();
    double x, y, yaw, rate;
    {
      std::lock_guard<std::mutex> l(mu_);
      const auto m = std::chrono::steady_clock::now();
      const double dt = std::min(0.1, std::chrono::duration<double>(m - last_).count());
      last_ = m;
      yaw_ += rate_ * dt;
      x = x_, y = y_, yaw = yaw_, rate = rate_;
    }
    geometry_msgs::msg::TransformStamped tr;
    tr.header.frame_id = "map";
    tr.header.stamp = t;
    tr.child_frame_id = "base_link";
    tr.transform.translation.x = x;
    tr.transform.translation.y = y;
    tr.transform.rotation.z = std::sin(yaw / 2);
    tr.transform.rotation.w = std::cos(yaw / 2);
    tf_->sendTransform(tr);
    limo_fire_msgs::msg::FireObservation f;
    f.header.frame_id = "map";
    f.header.stamp = t;
    f.id = "fixture_fire";
    f.position.x = 2.0;
    f.position.z = 0.4;
    f.position_valid = true;
    f.confidence = 0.9;
    fire_pub_->publish(f);
    nav_msgs::msg::Odometry o;
    o.header.frame_id = "odom";
    o.header.stamp = t;
    o.child_frame_id = "base_link";
    o.twist.twist.angular.z = rate;
    odom_pub_->publish(o);
    limo_msgs::msg::LimoStatus b;
    b.header.stamp = t;
    b.motion_mode = 0;
    base_pub_->publish(b);
  }

  std::shared_ptr<tf2_ros::StaticTransformBroadcaster> static_tf_;
  std::shared_ptr<tf2_ros::TransformBroadcaster> tf_;
  rclcpp::Publisher<nav_msgs::msg::OccupancyGrid>::SharedPtr grid_pub_;
  rclcpp::Publisher<limo_fire_msgs::msg::FireObservation>::SharedPtr fire_pub_;
  rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr odom_pub_;
  rclcpp::Publisher<limo_msgs::msg::LimoStatus>::SharedPtr base_pub_;
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr state_pub_;
  rclcpp::Subscription<geometry_msgs::msg::Twist>::SharedPtr yaw_sub_;
  rclcpp::CallbackGroup::SharedPtr group_;
  rclcpp_action::Server<Plan>::SharedPtr planner_;
  rclcpp_action::Server<Nav>::SharedPtr nav_;
  rclcpp_action::Server<Launch>::SharedPtr launch_srv_;
  rclcpp::TimerBase::SharedPtr sensor_timer_, grid_timer_;
  std::mutex mu_;
  double x_ = 0, y_ = 0, yaw_ = 0, rate_ = 0;
  int navigated_ = 0;
  std::chrono::steady_clock::time_point last_ = std::chrono::steady_clock::now();
};

class AimRos : public ::testing::Test {
 protected:
  // 시험마다 바꾸는 것
  std::vector<std::string> aim_params;
  bool fake_launcher = false, delay_navigation = false;
  double nav_yaw_bias_deg = 0.0;

  void SetUp() override {
    char tmpl[] = "/tmp/approach_aim_test_XXXXXX";
    dir_ = mkdtemp(tmpl);
    setenv("ROS_DOMAIN_ID", std::to_string(10 + getpid() % 40).c_str(), 1);
  }
  void start_fixture() {
    rclcpp::init(0, nullptr);
    robot_ = std::make_shared<FakeRobot>();
    robot_->delay_navigation = delay_navigation;
    robot_->nav_yaw_bias_deg = nav_yaw_bias_deg;
    if (fake_launcher) robot_->start_fake_launcher();
    probe_ = std::make_shared<rclcpp::Node>("aim_probe");
    state_sub_ = probe_->create_subscription<std_msgs::msg::String>(
        "/launcher/approach_aim/state", rclcpp::QoS(10).transient_local().reliable(), [this](std_msgs::msg::String::SharedPtr m) {
          std::lock_guard<std::mutex> l(mu_);
          states_.push_back(m->data);
        });
    start_ = probe_->create_client<std_srvs::srv::Trigger>("/launcher/approach_aim/start");
    cancel_ = probe_->create_client<std_srvs::srv::Trigger>("/launcher/approach_aim/cancel");
    exec_ = std::make_unique<rclcpp::executors::MultiThreadedExecutor>(rclcpp::ExecutorOptions(), 4);
    exec_->add_node(robot_);
    exec_->add_node(probe_);
    spin_ = std::thread([this] { exec_->spin(); });
    std::vector<std::string> argv = {APPROACH_AIM_EXE, "--ros-args", "--params-file", CONFIG_YAML, "-r", "__ns:=/launcher",
                                     "-r", "__node:=approach_aim", "-p", "mount_calibrated:=true", "-p", "stop_dwell_s:=0.1",
                                     "-p", "plan_timeout_s:=2.0", "-p", "navigation_timeout_s:=6.0", "-p",
                                     "reacquire_timeout_s:=3.0"};
    for (const auto& p : aim_params) argv.insert(argv.end(), {"-p", p});
    aim_pid_ = spawn(argv, dir_ + "/aim.log");
    ASSERT_TRUE(start_->wait_for_service(10s)) << slurp(dir_ + "/aim.log");
    ASSERT_TRUE(wait_state("idle", 5s));
    std::this_thread::sleep_for(1500ms);   // 조준 노드가 표적·costmap·TF·Nav2 서버를 받을 시간
  }
  void TearDown() override {
    stop(term_pid_);
    stop(aim_pid_);
    stop(bridge_pid_);
    stop(sim_pid_);
    if (exec_) exec_->cancel();
    if (spin_.joinable()) spin_.join();
    exec_.reset();
    start_.reset();
    cancel_.reset();
    state_sub_.reset();
    probe_.reset();
    robot_.reset();
    if (rclcpp::ok()) rclcpp::shutdown();
    if (!dir_.empty() && !HasFailure()) std::filesystem::remove_all(dir_);   // 실패하면 로그를 남긴다
  }

  std::vector<std::string> states() {
    std::lock_guard<std::mutex> l(mu_);
    return states_;
  }
  std::string state() {
    std::lock_guard<std::mutex> l(mu_);
    return states_.empty() ? "" : states_.back();
  }
  bool wait_state(const std::string& s, std::chrono::milliseconds limit) {
    const auto end = std::chrono::steady_clock::now() + limit;
    while (std::chrono::steady_clock::now() < end) {
      if (state() == s) return true;
      std::this_thread::sleep_for(20ms);
    }
    return false;
  }
  std_srvs::srv::Trigger::Response call(rclcpp::Client<std_srvs::srv::Trigger>::SharedPtr c) {
    auto f = c->async_send_request(std::make_shared<std_srvs::srv::Trigger::Request>());
    if (f.wait_for(3s) != std::future_status::ready) return std_srvs::srv::Trigger::Response();
    return *f.get();
  }
  bool seen(const std::string& s) {
    const auto v = states();
    return std::find(v.begin(), v.end(), s) != v.end();
  }
  // 터미널에 글자 줄을 넣고 끝날 때까지
  std::string run_terminal(const std::string& input, double timeout_s) {
    int fds[2];
    if (pipe(fds) != 0) return "";
    term_pid_ = spawn({LAUNCHER_TERMINAL_EXE, "--no-search", "--log", dir_ + "/terminal.jsonl"}, dir_ + "/terminal.log", fds[0]);
    ::close(fds[0]);
    const ssize_t w = ::write(fds[1], input.data(), input.size());
    (void)w;
    ::close(fds[1]);
    EXPECT_TRUE(wait_exit(term_pid_, timeout_s)) << "터미널이 끝나지 않음";
    term_pid_ = -1;
    return slurp(dir_ + "/terminal.log");
  }

  std::string dir_;
  pid_t aim_pid_ = -1, term_pid_ = -1, bridge_pid_ = -1, sim_pid_ = -1;
  std::shared_ptr<FakeRobot> robot_;
  rclcpp::Node::SharedPtr probe_;
  rclcpp::Subscription<std_msgs::msg::String>::SharedPtr state_sub_;
  rclcpp::Client<std_srvs::srv::Trigger>::SharedPtr start_, cancel_;
  std::unique_ptr<rclcpp::executors::MultiThreadedExecutor> exec_;
  std::thread spin_;
  std::mutex mu_;
  std::vector<std::string> states_;
};

}  // namespace

TEST_F(AimRos, PreviewHasNoActuatorCommands) {
  aim_params = {"motion_enabled:=false"};
  start_fixture();
  const auto r = call(start_);
  ASSERT_TRUE(r.success) << r.message;
  EXPECT_TRUE(wait_state("preview_ready", 5s)) << state() << "\n" << slurp(dir_ + "/aim.log");
  EXPECT_EQ(robot_->nav_goals.load(), 0);
  EXPECT_EQ(robot_->nonzero_yaw.load(), 0);
}

TEST_F(AimRos, Nav2ApproachAndAlignEndsAimedWithoutFiring) {
  aim_params = {"motion_enabled:=true"};
  start_fixture();
  const auto r = call(start_);
  ASSERT_TRUE(r.success) << r.message;
  EXPECT_TRUE(wait_state("aim_ready", 15s)) << state() << "\n" << slurp(dir_ + "/aim.log");
  EXPECT_EQ(robot_->nav_goals.load(), 2);   // 접근 + 좌우 정렬
  EXPECT_TRUE(seen("approaching"));
  EXPECT_TRUE(seen("aligning"));
}

TEST_F(AimRos, YawRateRotatesToTarget) {
  aim_params = {"motion_enabled:=true", "yaw_alignment_mode:=yaw_rate", "yaw_rate_drive_mode_verified:=true"};
  nav_yaw_bias_deg = 12.0;   // 접근 뒤 12° 틀어져 섬
  start_fixture();
  const auto r = call(start_);
  ASSERT_TRUE(r.success) << r.message;
  EXPECT_TRUE(wait_state("aim_ready", 20s)) << state() << "\n" << slurp(dir_ + "/aim.log");
  EXPECT_TRUE(seen("yaw_aligning"));
  EXPECT_GT(robot_->nonzero_yaw.load(), 0);
  EXPECT_EQ(robot_->nav_goals.load(), 1);
}

TEST_F(AimRos, UnverifiedYawDriveIsRefused) {
  aim_params = {"motion_enabled:=true", "yaw_alignment_mode:=yaw_rate"};
  start_fixture();
  const auto r = call(start_);
  EXPECT_FALSE(r.success);
  EXPECT_EQ(robot_->nav_goals.load(), 0);
}

TEST_F(AimRos, CancelDuringNavigationStopsAndReports) {
  aim_params = {"motion_enabled:=true"};
  delay_navigation = true;
  start_fixture();
  ASSERT_TRUE(call(start_).success);
  ASSERT_TRUE(wait_state("approaching", 5s)) << state();
  std::this_thread::sleep_for(300ms);
  EXPECT_TRUE(call(cancel_).success);
  EXPECT_TRUE(wait_state("cancelled", 8s)) << state() << "\n" << slurp(dir_ + "/aim.log");
  EXPECT_GE(robot_->nav_cancels.load(), 1);
}

TEST_F(AimRos, TerminalStartsExactlyOneShotUntilReloadConfirmed) {
  aim_params = {"motion_enabled:=true", "auto_fire_enabled:=true"};
  fake_launcher = true;
  start_fixture();
  // a → 임무 한 번, 미리 친 두 번째 a 는 거절, r 뒤 a 는 한 번 더 허용
  const auto out = run_terminal("a\na\nr\na\nq\n", 90.0);
  EXPECT_NE(out.find("shot_done"), std::string::npos) << out << "\n" << slurp(dir_ + "/aim.log");
  EXPECT_NE(out.find("r을 입력하세요"), std::string::npos) << out;
  EXPECT_EQ(robot_->launch_goals.load(), 2) << out;
  EXPECT_NE(slurp(dir_ + "/terminal.jsonl").find("reload_confirmed"), std::string::npos);
}

TEST_F(AimRos, TerminalRefusesWhenAutoFireDisabled) {
  aim_params = {"motion_enabled:=true"};
  fake_launcher = true;
  start_fixture();
  const auto out = run_terminal("a\nq\n", 30.0);
  EXPECT_NE(out.find("자동 발사가 꺼져"), std::string::npos) << out;
  EXPECT_EQ(robot_->launch_goals.load(), 0);
}

// 터미널 a → 조준 → 실제 esp32_bridge → 펌웨어 원본(가상 시리얼). 펌웨어 레버 PWM 이 장전·발사 자리로 갔는지
TEST_F(AimRos, TerminalDrivesRealBridgeAndFirmwareOnce) {
  const char* env = std::getenv("LAUNCHER_HOST_SIM");
  const std::string sim = env ? env : "/tmp/launcher_host_sim";
  if (access(sim.c_str(), X_OK) != 0) GTEST_SKIP() << sim << " 없음 — firmware/limo_fire_esp32/test/run.sh";
  std::string bridge;
  try {
    bridge = ament_index_cpp::get_package_prefix("limo_fire_esp32_bridge") + "/lib/limo_fire_esp32_bridge/esp32_bridge";
  } catch (...) {
  }
  if (bridge.empty() || access(bridge.c_str(), X_OK) != 0) GTEST_SKIP() << "esp32_bridge 실행 파일 없음";
  sim_pid_ = spawn({sim, dir_ + "/pty"}, dir_ + "/pwm.log");
  std::string port;
  for (int i = 0; i < 100 && port.empty(); ++i) {
    std::this_thread::sleep_for(50ms);
    std::ifstream f(dir_ + "/pty");
    std::getline(f, port);
  }
  ASSERT_FALSE(port.empty());
  bridge_pid_ = spawn({bridge, "--ros-args", "-p", "port:=" + port, "-p", "launcher.firing_enabled:=true"}, dir_ + "/bridge.log");
  aim_params = {"motion_enabled:=true", "auto_fire_enabled:=true"};
  start_fixture();
  // 다리가 펌웨어를 알아볼 때까지(/launcher/state off)
  auto node = std::make_shared<rclcpp::Node>("launcher_state_probe");
  std::atomic<bool> off{false};
  auto sub = node->create_subscription<std_msgs::msg::String>("/launcher/state", rclcpp::QoS(1).transient_local().reliable(),
                                                             [&](std_msgs::msg::String::SharedPtr m) { off = m->data == "off"; });
  for (int i = 0; i < 200 && !off; ++i) {
    rclcpp::spin_some(node);
    std::this_thread::sleep_for(50ms);
  }
  ASSERT_TRUE(off.load()) << slurp(dir_ + "/bridge.log");
  std::this_thread::sleep_for(500ms);   // 조준 노드도 같은 상태를 받을 시간

  const auto out = run_terminal("a\na\nq\n", 90.0);
  EXPECT_NE(out.find("shot_done"), std::string::npos) << out << "\n" << slurp(dir_ + "/aim.log") << slurp(dir_ + "/bridge.log");
  EXPECT_NE(out.find("r을 입력하세요"), std::string::npos) << out;
  std::istringstream pwm(slurp(dir_ + "/pwm.log"));
  std::string line;
  std::vector<int> lever;
  bool tilt_attached = false;
  while (std::getline(pwm, line)) {
    std::istringstream s(line);
    std::string t, kind, pin, val;
    s >> t >> kind >> pin >> val;
    if (kind == "pwm" && pin == "4") lever.push_back(std::atoi(val.c_str()));
    if (kind == "attach" && pin == "5") tilt_attached = true;
  }
  ASSERT_FALSE(lever.empty());
  EXPECT_LT(*std::min_element(lever.begin(), lever.end()), 900);    // 장전 30° → 833 µs
  EXPECT_GT(*std::max_element(lever.begin(), lever.end()), 1850);   // 발사 125° → 1889 µs
  EXPECT_FALSE(tilt_attached);                                      // 각도 서보는 없음(15° 고정)
}
