// 다리 실행 파일(esp32_bridge) ↔ 펌웨어 원본(firmware/limo_fire_esp32 src/main.cpp 를 PC 에서 가상 시리얼에 돌린 launcher_host_sim).
// 실물 보드·서보 없이 발사대 액션·서비스·상태 토픽과 센서 줄이 한 포트로 함께 도는지, 펌웨어 실제 타이밍으로 본다.
// 호스트 시뮬은 firmware/limo_fire_esp32/test/run.sh 가 /tmp/launcher_host_sim 으로 빌드한다(없으면 건너뜀, LAUNCHER_HOST_SIM 로 바꿀 수 있음).
// 서보 PWM 명령은 시뮬 표준 출력(pwm.log)에 남아 레버가 장전·발사 자리로 갔는지 본다(펄스 폭 = 명령, 실제 각은 모름).
#include <gtest/gtest.h>
#include <signal.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include "diagnostic_msgs/msg/diagnostic_array.hpp"
#include "limo_fire_msgs/action/launch.hpp"
#include "rclcpp/rclcpp.hpp"
#include "rclcpp_action/rclcpp_action.hpp"
#include "sensor_msgs/msg/joint_state.hpp"
#include "sensor_msgs/msg/range.hpp"
#include "std_msgs/msg/string.hpp"
#include "std_srvs/srv/set_bool.hpp"

using namespace std::chrono_literals;
using Launch = limo_fire_msgs::action::Launch;

namespace {
pid_t spawn(const std::vector<std::string>& argv, const std::string& out_path) {
  const pid_t pid = fork();
  if (pid == 0) {
    setpgid(0, 0);
    if (!out_path.empty() && freopen(out_path.c_str(), "w", stdout) == nullptr) _exit(127);
    if (!out_path.empty()) dup2(fileno(stdout), fileno(stderr));
    std::vector<char*> a;
    for (const auto& s : argv) a.push_back(const_cast<char*>(s.c_str()));
    a.push_back(nullptr);
    execv(a[0], a.data());
    _exit(127);
  }
  return pid;
}
void stop(pid_t pid) {
  if (pid <= 0) return;
  for (int sig : {SIGINT, SIGTERM, SIGKILL}) {
    kill(-pid, sig);
    for (int i = 0; i < 60; ++i) {
      if (waitpid(pid, nullptr, WNOHANG) == pid) return;
      std::this_thread::sleep_for(50ms);
    }
  }
}

class BridgeHostSim : public ::testing::Test {
 protected:
  void SetUp() override {
    const char* env = std::getenv("LAUNCHER_HOST_SIM");
    sim_ = env ? env : "/tmp/launcher_host_sim";
    if (access(sim_.c_str(), X_OK) != 0) GTEST_SKIP() << sim_ << " 없음 — firmware/limo_fire_esp32/test/run.sh 로 빌드";
    char tmpl[] = "/tmp/bridge_host_sim_XXXXXX";
    dir_ = mkdtemp(tmpl);
    const std::string pty_file = dir_ + "/pty";
    pwm_log_ = dir_ + "/pwm.log";
    sim_pid_ = spawn({sim_, pty_file}, pwm_log_);
    std::string port;
    for (int i = 0; i < 100 && port.empty(); ++i) {
      std::this_thread::sleep_for(50ms);
      std::ifstream f(pty_file);
      std::getline(f, port);
    }
    ASSERT_FALSE(port.empty()) << "호스트 시뮬이 pty 를 알리지 않음";
    // 다른 시험과 섞이지 않게 도메인을 따로 (자식 다리도 물려받음)
    setenv("ROS_DOMAIN_ID", std::to_string(60 + getpid() % 40).c_str(), 1);
    bridge_pid_ = spawn({ESP32_BRIDGE_EXE, "--ros-args", "-p", "port:=" + port, "-p", "launcher.firing_enabled:=true", "-p",
                         "launcher.query_period_s:=0.5"},
                        dir_ + "/bridge.log");
    rclcpp::init(0, nullptr);
    node_ = std::make_shared<rclcpp::Node>("bridge_host_sim_test");
    state_sub_ = node_->create_subscription<std_msgs::msg::String>(
        "/launcher/state", rclcpp::QoS(1).transient_local().reliable(), [this](std_msgs::msg::String::SharedPtr m) {
          std::lock_guard<std::mutex> l(mu_);
          states_.push_back(m->data);
        });
    range_sub_ = node_->create_subscription<sensor_msgs::msg::Range>(
        "/ultrasonic/front_left", rclcpp::SensorDataQoS(), [this](sensor_msgs::msg::Range::SharedPtr) { ++ranges_; });
    joint_sub_ = node_->create_subscription<sensor_msgs::msg::JointState>(
        "/launcher/joint_states", 10, [this](sensor_msgs::msg::JointState::SharedPtr m) {
          if (m->name.size() == 1 && m->name[0] == "launcher_lever_joint") ++joints_;
        });
    diag_sub_ = node_->create_subscription<diagnostic_msgs::msg::DiagnosticArray>(
        "/diagnostics", 10, [this](diagnostic_msgs::msg::DiagnosticArray::SharedPtr m) {
          for (const auto& s : m->status)
            if (s.name.find("launcher") != std::string::npos) ++diag_launcher_;
        });
    action_ = rclcpp_action::create_client<Launch>(node_, "/launcher/launch");
    enable_ = node_->create_client<std_srvs::srv::SetBool>("/launcher/enable");
    exec_ = std::make_unique<rclcpp::executors::SingleThreadedExecutor>();
    exec_->add_node(node_);
    spin_ = std::thread([this] { exec_->spin(); });
  }
  void TearDown() override {
    if (exec_) exec_->cancel();
    if (spin_.joinable()) spin_.join();
    exec_.reset();
    action_.reset();
    enable_.reset();
    state_sub_.reset();
    range_sub_.reset();
    joint_sub_.reset();
    diag_sub_.reset();
    node_.reset();
    if (rclcpp::ok()) rclcpp::shutdown();
    stop(bridge_pid_);
    stop(sim_pid_);
    if (!dir_.empty() && !HasFailure()) std::filesystem::remove_all(dir_);   // 실패하면 로그를 남긴다
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

  struct Outcome {
    rclcpp_action::ResultCode code = rclcpp_action::ResultCode::UNKNOWN;
    bool success = false;
    std::string message;
    double seconds = 0;
  };
  Outcome run(uint8_t mode, std::chrono::milliseconds cancel_after = 0ms) {
    Outcome o;
    Launch::Goal g;
    g.mode = mode;
    const auto t0 = std::chrono::steady_clock::now();
    auto gh_f = action_->async_send_goal(g);
    if (gh_f.wait_for(5s) != std::future_status::ready) return o;
    auto gh = gh_f.get();
    if (!gh) return o;
    if (cancel_after > 0ms) {
      std::this_thread::sleep_for(cancel_after);
      action_->async_cancel_goal(gh);
    }
    auto res_f = action_->async_get_result(gh);
    if (res_f.wait_for(15s) != std::future_status::ready) return o;
    const auto res = res_f.get();
    o.code = res.code;
    o.success = res.result->success;
    o.message = res.result->message;
    o.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    return o;
  }
  bool set_enabled(bool on, std::string* msg = nullptr) {
    auto req = std::make_shared<std_srvs::srv::SetBool::Request>();
    req->data = on;
    auto f = enable_->async_send_request(req);
    if (f.wait_for(5s) != std::future_status::ready) return false;
    const auto r = f.get();
    if (msg) *msg = r->message;
    return r->success;
  }
  std::vector<int> lever_pwm_us() {
    std::ifstream f(pwm_log_);
    std::string line;
    std::vector<int> us;
    while (std::getline(f, line)) {
      std::istringstream s(line);
      std::string t, kind, pin, val;
      s >> t >> kind >> pin >> val;
      if (kind == "pwm" && pin == "4") us.push_back(std::atoi(val.c_str()));
    }
    return us;
  }
  std::string pwm_log() {
    std::ifstream f(pwm_log_);
    std::stringstream ss;
    ss << f.rdbuf();
    return ss.str();
  }

  std::string sim_, dir_, pwm_log_;
  pid_t sim_pid_ = -1, bridge_pid_ = -1;
  rclcpp::Node::SharedPtr node_;
  rclcpp::Subscription<std_msgs::msg::String>::SharedPtr state_sub_;
  rclcpp::Subscription<sensor_msgs::msg::Range>::SharedPtr range_sub_;
  rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr joint_sub_;
  rclcpp::Subscription<diagnostic_msgs::msg::DiagnosticArray>::SharedPtr diag_sub_;
  rclcpp_action::Client<Launch>::SharedPtr action_;
  rclcpp::Client<std_srvs::srv::SetBool>::SharedPtr enable_;
  std::unique_ptr<rclcpp::executors::SingleThreadedExecutor> exec_;
  std::thread spin_;
  std::mutex mu_;
  std::vector<std::string> states_;
  std::atomic<int> ranges_{0}, joints_{0}, diag_launcher_{0};
};
}  // namespace

TEST_F(BridgeHostSim, ShotCockFireRefusalsAndCancelAgainstRealFirmware) {
  ASSERT_TRUE(wait_state("off", 15s)) << "다리가 펌웨어를 알아보지 못함 (마지막 상태 '" << state() << "')";
  ASSERT_TRUE(action_->wait_for_action_server(5s));
  ASSERT_TRUE(enable_->wait_for_service(5s));

  // 1) 레버가 꺼진 채 장전+발사 → 다리가 e 를 먼저 보내고 cocked·fired 를 받음 (펌웨어 시간 ≈ 2.5 s)
  auto o = run(Launch::Goal::SHOT);
  EXPECT_EQ(o.code, rclcpp_action::ResultCode::SUCCEEDED) << o.message;
  EXPECT_TRUE(o.success);
  EXPECT_GT(o.seconds, 2.0);
  EXPECT_LT(o.seconds, 6.0);
  EXPECT_TRUE(wait_state("ready", 2s)) << state();

  // 2) 장전 안 됐는데 발사 → 거절(abort, 까닭 담김)
  o = run(Launch::Goal::FIRE);
  EXPECT_EQ(o.code, rclcpp_action::ResultCode::ABORTED);
  EXPECT_NE(o.message.find("거절"), std::string::npos) << o.message;

  // 3) 장전 → cocked, 다시 장전·장전+발사 거절, 발사 → ready
  o = run(Launch::Goal::COCK);
  EXPECT_EQ(o.code, rclcpp_action::ResultCode::SUCCEEDED) << o.message;
  EXPECT_TRUE(wait_state("cocked", 2s)) << state();
  EXPECT_EQ(run(Launch::Goal::COCK).code, rclcpp_action::ResultCode::ABORTED);
  EXPECT_EQ(run(Launch::Goal::SHOT).code, rclcpp_action::ResultCode::ABORTED);
  o = run(Launch::Goal::FIRE);
  EXPECT_EQ(o.code, rclcpp_action::ResultCode::SUCCEEDED) << o.message;
  EXPECT_LT(o.seconds, 2.0);
  EXPECT_TRUE(wait_state("ready", 2s)) << state();

  // 4) 끄기 → off
  std::string msg;
  EXPECT_TRUE(set_enabled(false, &msg)) << msg;
  EXPECT_TRUE(wait_state("off", 2s)) << state();

  // 5) 장전+발사 도중 취소 → x, CANCELED, 고장(장전 여부 모름) → 끄기로 풀림
  o = run(Launch::Goal::SHOT, 600ms);
  EXPECT_EQ(o.code, rclcpp_action::ResultCode::CANCELED) << o.message;
  EXPECT_LT(o.seconds, 2.0);
  EXPECT_TRUE(wait_state("fault", 2s)) << state();
  EXPECT_EQ(run(Launch::Goal::SHOT).code, rclcpp_action::ResultCode::ABORTED);   // 고장이면 거절
  EXPECT_TRUE(set_enabled(false, &msg)) << msg;
  EXPECT_TRUE(wait_state("off", 2s)) << state();

  // 같은 포트에서 센서 줄도 계속 들어왔고, 레버 관절·진단도 나왔다
  std::this_thread::sleep_for(1500ms);
  EXPECT_GT(ranges_.load(), 10);
  EXPECT_GT(joints_.load(), 0);
  EXPECT_GT(diag_launcher_.load(), 0);

  // 펌웨어가 레버 서보(GPIO 4)만 켜고, 장전(30° → 833 µs)·발사(125° → 1889 µs) 자리로 명령했다. 각도 서보(GPIO 5)는 켜지 않음
  const auto us = lever_pwm_us();
  ASSERT_FALSE(us.empty());
  EXPECT_LT(*std::min_element(us.begin(), us.end()), 900);
  EXPECT_GT(*std::max_element(us.begin(), us.end()), 1850);
  const auto log = pwm_log();
  EXPECT_NE(log.find(" attach 4 "), std::string::npos);
  EXPECT_EQ(log.find(" attach 5 "), std::string::npos);
  EXPECT_NE(log.find(" detach 4"), std::string::npos);   // 끄기·취소 → x → PWM 끔
}
