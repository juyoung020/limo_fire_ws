// ESP32 ↔ ROS 다리 — 펌웨어 firmware/limo_fire_esp32 하나(센서 + 발사대 레버)의 USB 시리얼을 이 노드 하나만 연다.
//   센서: 초음파 HC-SR04 × 2 → sensor_msgs/Range, 열화상 → sensor_msgs/Image(32FC1 °C) + sensor_msgs/Temperature(최고 온도),
//         레이더 → geometry_msgs/PoseArray + 속도 Float32MultiArray. 토픽·프레임은 시뮬(Webots LimoFire)과 같다.
//   발사대: 액션 /launcher/launch (limo_fire_msgs/action/Launch: 장전·발사·장전+발사, 취소 = x), 서비스 /launcher/enable (std_srvs/SetBool),
//         상태 /launcher/state(짧은 영문, TRANSIENT_LOCAL) · /launcher/status(한글 한 줄, TRANSIENT_LOCAL), 레버 /launcher/joint_states.
//   건강: /diagnostics (diagnostic_updater) — 포트·살아 있음·줄 종류별 빈도·깨진 줄·펌웨어 재시작·발사대 고장.
// 시각(header.stamp)은 다리가 줄을 받은 시각이다(ESP32 ms 시계는 호스트와 달라 쓰지 않는다 — USB 지연 수 ms).
// 포트가 없거나 끊기면 알리고 1 초마다 다시 연다. 줄 형식은 protocol.hpp, 발사대 규칙은 launcher_logic.hpp.
#include <fcntl.h>
#include <termios.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstring>
#include <limits>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "diagnostic_updater/diagnostic_updater.hpp"
#include "geometry_msgs/msg/pose_array.hpp"
#include "limo_fire_esp32_bridge/launcher_logic.hpp"
#include "limo_fire_esp32_bridge/protocol.hpp"
#include "limo_fire_msgs/action/launch.hpp"
#include "rclcpp/rclcpp.hpp"
#include "rclcpp_action/rclcpp_action.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "sensor_msgs/msg/joint_state.hpp"
#include "sensor_msgs/msg/range.hpp"
#include "sensor_msgs/msg/temperature.hpp"
#include "std_msgs/msg/float32_multi_array.hpp"
#include "std_msgs/msg/string.hpp"
#include "std_srvs/srv/set_bool.hpp"

namespace limo_fire_esp32_bridge {

using Launch = limo_fire_msgs::action::Launch;
using GoalHandle = rclcpp_action::ServerGoalHandle<Launch>;
using SetBool = std_srvs::srv::SetBool;

class Esp32Bridge : public rclcpp::Node {
 public:
  Esp32Bridge() : Node("esp32_bridge"), diag_(this) {
    port_ = declare_parameter<std::string>("port", "/dev/esp32");
    baud_ = declare_parameter<int>("baud", 921600);
    hb_timeout_ = declare_parameter<double>("heartbeat_timeout", 3.0);
    names_ = declare_parameter<std::vector<std::string>>("ultrasonic.names", {"front_left", "front_right"});
    fov_ = declare_parameter<double>("ultrasonic.field_of_view", 0.2618);
    min_ = declare_parameter<double>("ultrasonic.min_range", 0.02);
    max_ = declare_parameter<double>("ultrasonic.max_range", 4.0);
    thermal_frame_ = declare_parameter<std::string>("thermal.frame_id", "thermal_link");
    radar_frame_ = declare_parameter<std::string>("radar.frame_id", "radar_link");
    launcher_on_ = declare_parameter<bool>("launcher.enabled", true);
    LauncherConfig lc;
    lc.firing_enabled = declare_parameter<bool>("launcher.firing_enabled", false);
    lc.ack_timeout_s = declare_parameter<double>("launcher.ack_timeout_s", 3.0);
    lc.action_timeout_s = declare_parameter<double>("launcher.action_timeout_s", 8.0);
    lc.query_period_s = declare_parameter<double>("launcher.query_period_s", 1.0);
    lever_joint_ = declare_parameter<std::string>("launcher.lever_joint", "launcher_lever_joint");
    if (names_.empty() || fov_ <= 0.0 || min_ < 0.0 || max_ <= min_ || hb_timeout_ <= 0.0)
      throw std::invalid_argument("esp32_bridge: ultrasonic.names·field_of_view·min/max_range·heartbeat_timeout 확인");
    if (lc.ack_timeout_s <= 0.0 || lc.action_timeout_s <= 0.0 || lc.query_period_s <= 0.0 || lever_joint_.empty())
      throw std::invalid_argument("esp32_bridge: launcher.ack_timeout_s·action_timeout_s·query_period_s·lever_joint 확인");
    if (speed_of(baud_) == B0) throw std::invalid_argument("esp32_bridge: 지원하지 않는 baud " + std::to_string(baud_));

    for (const auto& n : names_) {
      pubs_.push_back(create_publisher<sensor_msgs::msg::Range>("/ultrasonic/" + n, rclcpp::SensorDataQoS()));
      frames_.push_back("ultrasonic_" + n + "_link");
    }
    thermal_pub_ = create_publisher<sensor_msgs::msg::Image>("/thermal/image", rclcpp::SensorDataQoS());
    thermal_max_pub_ = create_publisher<sensor_msgs::msg::Temperature>("/thermal/max", rclcpp::SensorDataQoS());
    radar_pub_ = create_publisher<geometry_msgs::msg::PoseArray>("/radar/targets", rclcpp::SensorDataQoS());
    radar_vel_pub_ = create_publisher<std_msgs::msg::Float32MultiArray>("/radar/velocities", rclcpp::SensorDataQoS());

    LauncherLogic::Callbacks cb;
    cb.send = [this](const std::string& l) { return write_line(l); };
    cb.enable_done = [this](bool ok, const std::string& m) { finish_enable(ok, m); };
    cb.feedback = [this](const std::string& p) { goal_feedback(p); };
    cb.goal_done = [this](GoalEnd e, const std::string& m) { goal_done(e, m); };
    cb.changed = [this] { publish_state(); };
    cb.lever = [this](double deg) { publish_lever(deg); };
    logic_ = std::make_unique<LauncherLogic>(lc, std::move(cb));
    if (launcher_on_) {
      const auto latched = rclcpp::QoS(10).transient_local().reliable();   // 늦게 붙은 쪽은 마지막 것이 지금 상태
      state_pub_ = create_publisher<std_msgs::msg::String>("/launcher/state", latched);
      status_pub_ = create_publisher<std_msgs::msg::String>("/launcher/status", latched);
      joint_pub_ = create_publisher<sensor_msgs::msg::JointState>("/launcher/joint_states", 10);
      enable_srv_ = create_service<SetBool>(
          "/launcher/enable",
          [this](std::shared_ptr<rmw_request_id_t> h, std::shared_ptr<SetBool::Request> req) { on_enable(h, req); });
      action_ = rclcpp_action::create_server<Launch>(
          this, "/launcher/launch",
          [](const rclcpp_action::GoalUUID&, std::shared_ptr<const Launch::Goal> g) {
            // 모드 틀림만 여기서 거절. 상태 때문에 못 하는 것은 받은 뒤 까닭을 담아 abort(클라이언트가 까닭을 볼 수 있게)
            const bool ok = g->mode == Launch::Goal::COCK || g->mode == Launch::Goal::FIRE || g->mode == Launch::Goal::SHOT;
            return ok ? rclcpp_action::GoalResponse::ACCEPT_AND_EXECUTE : rclcpp_action::GoalResponse::REJECT;
          },
          [this](std::shared_ptr<GoalHandle> gh) {
            if (gh != goal_) return rclcpp_action::CancelResponse::REJECT;
            logic_->cancel_goal(mono());
            return rclcpp_action::CancelResponse::ACCEPT;
          },
          [this](std::shared_ptr<GoalHandle> gh) { on_goal(gh); });
      publish_state();
    }

    diag_.setHardwareID("esp32 " + port_);
    diag_.add("link", this, &Esp32Bridge::diag_link);
    if (launcher_on_) diag_.add("launcher", this, &Esp32Bridge::diag_launcher);

    last_hb_ = now();
    last_rate_ = std::chrono::steady_clock::now();
    timer_ = create_wall_timer(std::chrono::milliseconds(5), [this] { poll(); });
    RCLCPP_INFO(get_logger(), "ESP32 %s @ %d, 초음파 %zu 개 · 열화상 · 레이더%s", port_.c_str(), baud_, names_.size(),
                launcher_on_ ? (lc.firing_enabled ? " · 발사대(켜기·발사 허용)" : " · 발사대(firing_enabled=false: 상태만)") : "");
  }
  ~Esp32Bridge() override {
    if (fd_ >= 0 && launcher_on_ && logic_->lever_on()) write_line("x");   // 끌 때 레버 PWM 끔
    close_port();
  }

 private:
  static double mono() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

  static speed_t speed_of(int b) {
    switch (b) {
      case 9600: return B9600;
      case 57600: return B57600;
      case 115200: return B115200;
      case 230400: return B230400;
      case 460800: return B460800;
      case 921600: return B921600;
      default: return B0;
    }
  }

  // ---------------- 시리얼
  bool open_port() {
    fd_ = ::open(port_.c_str(), O_RDWR | O_NOCTTY | O_NONBLOCK);   // 읽기·쓰기 — 발사대 명령도 같은 포트로
    if (fd_ < 0) return false;
    termios t{};
    if (tcgetattr(fd_, &t) != 0) { close_port(); return false; }
    cfmakeraw(&t);
    cfsetispeed(&t, speed_of(baud_));
    cfsetospeed(&t, speed_of(baud_));
    t.c_cflag |= CLOCAL | CREAD;
    if (tcsetattr(fd_, TCSANOW, &t) != 0) { close_port(); return false; }
    tcflush(fd_, TCIOFLUSH);
    buf_.clear();
    RCLCPP_INFO(get_logger(), "ESP32 포트 열림: %s (칩이 다시 켜질 수 있음 — DTR/RTS 자동 리셋)", port_.c_str());
    if (launcher_on_) logic_->link_up(mono());
    return true;
  }
  void close_port() {
    if (fd_ >= 0) ::close(fd_);
    fd_ = -1;
  }
  void drop_port(const char* why) {
    RCLCPP_ERROR(get_logger(), "ESP32 포트 끊김 (%s) — 다시 엶", why);
    close_port();
    if (launcher_on_) logic_->link_down(mono());
  }
  bool write_line(const std::string& l) {
    if (fd_ < 0) return false;
    const std::string out = l + "\n";
    const ssize_t w = ::write(fd_, out.data(), out.size());
    if (w == static_cast<ssize_t>(out.size())) {
      RCLCPP_DEBUG(get_logger(), "→ ESP32 %s", l.c_str());
      return true;
    }
    RCLCPP_ERROR(get_logger(), "ESP32 에 '%s' 쓰기 실패 (%s)", l.c_str(), w < 0 ? std::strerror(errno) : "일부만");
    return false;
  }

  void poll() {
    if (fd_ < 0) {
      const auto t = now();
      if ((t - last_try_).seconds() < 1.0) return;
      last_try_ = t;
      if (!open_port()) {
        RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 10000, "ESP32 포트 %s 를 못 엶: %s (1 초마다 다시)", port_.c_str(),
                             std::strerror(errno));
        return;
      }
    }
    char tmp[4096];
    for (;;) {
      const ssize_t n = ::read(fd_, tmp, sizeof tmp);
      if (n > 0) { buf_.append(tmp, n); continue; }
      if (n == 0 || (errno != EAGAIN && errno != EWOULDBLOCK)) drop_port(n == 0 ? "EOF" : std::strerror(errno));
      break;
    }
    size_t nl;
    while ((nl = buf_.find('\n')) != std::string::npos) {
      handle(buf_.substr(0, nl));
      buf_.erase(0, nl + 1);
    }
    if (buf_.size() > 16384) { ++bad_; buf_.clear(); }   // 줄 끝이 안 오는 쓰레기 (열화상 한 줄 ≈ 3 KB 의 5 배)
    if (fd_ >= 0 && (now() - last_hb_).seconds() > hb_timeout_)
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 5000, "ESP32 살아 있음 줄이 %.0f s 넘게 없음", hb_timeout_);
    if (launcher_on_) logic_->tick(mono());
  }

  void handle(const std::string& raw) {
    const auto l = parse_line(raw);
    if (!l) { ++bad_; RCLCPP_DEBUG(get_logger(), "깨진 줄 버림 (%zu 번째)", bad_); return; }
    ++counts_[l->type];
    if (l->type == "HB") { last_hb_ = now(); return; }
    if (l->type == "TH") { handle_thermal(*l); return; }
    if (l->type == "RD") { handle_radar(*l); return; }
    if (l->type == "LA") {
      const auto r = as_launcher(*l);
      if (!r) { ++bad_; RCLCPP_WARN(get_logger(), "발사대 줄 형식 틀림: %s", raw.c_str()); return; }
      if (launcher_on_) logic_->reply(*r, mono());
      return;
    }
    const auto s = as_sonar(*l);
    if (!s) return;
    if (s->index >= static_cast<int>(pubs_.size())) {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 5000, "초음파 번호 %d 가 설정(%zu 개) 밖", s->index, pubs_.size());
      return;
    }
    sensor_msgs::msg::Range m;
    m.header.stamp = now();
    m.header.frame_id = frames_[s->index];
    m.radiation_type = sensor_msgs::msg::Range::ULTRASOUND;
    m.field_of_view = static_cast<float>(fov_);
    m.min_range = static_cast<float>(min_);
    m.max_range = static_cast<float>(max_);
    m.range = s->mm == 0 ? std::numeric_limits<float>::infinity() : static_cast<float>(s->mm * 1e-3);   // 메아리 없음 = +inf (Range 규약)
    pubs_[s->index]->publish(m);
  }

  void handle_thermal(const Line& l) {
    const auto t = as_thermal(l);
    if (!t) { ++bad_; RCLCPP_DEBUG(get_logger(), "열화상 줄 형식 틀림"); return; }
    const auto stamp = now();
    sensor_msgs::msg::Image m;
    m.header.stamp = stamp;
    m.header.frame_id = thermal_frame_;
    m.height = kThermalRows;
    m.width = kThermalCols;
    m.encoding = "32FC1";   // 값 = °C
    m.is_bigendian = 0;
    m.step = kThermalCols * sizeof(float);
    m.data.resize(kThermalPoints * sizeof(float));
    std::memcpy(m.data.data(), t->c.data(), m.data.size());
    thermal_pub_->publish(m);
    sensor_msgs::msg::Temperature mx;   // 그 프레임 화소 최고 온도. variance 0 = 모름 (Temperature.msg)
    mx.header.stamp = stamp;
    mx.header.frame_id = thermal_frame_;
    mx.temperature = *std::max_element(t->c.begin(), t->c.end());
    mx.variance = 0.0;
    thermal_max_pub_->publish(mx);
  }

  // LD2450 (x 오른쪽 + · y 앞, mm) → REP-103 (x 앞 · y 왼쪽, m): x_ros = y, y_ros = −x. 속도는 LD2450 원본 부호 그대로 m/s
  void handle_radar(const Line& l) {
    const auto r = as_radar(l);
    if (!r) { ++bad_; RCLCPP_DEBUG(get_logger(), "레이더 줄 형식 틀림"); return; }
    const auto stamp = now();
    geometry_msgs::msg::PoseArray pa;
    pa.header.stamp = stamp;
    pa.header.frame_id = radar_frame_;
    std_msgs::msg::Float32MultiArray v;
    for (const auto& t : r->targets) {
      geometry_msgs::msg::Pose p;
      p.position.x = t.y_mm * 1e-3;
      p.position.y = -t.x_mm * 1e-3;
      p.orientation.w = 1.0;
      pa.poses.push_back(p);
      v.data.push_back(static_cast<float>(t.v_cms * 1e-2));
    }
    radar_pub_->publish(pa);
    radar_vel_pub_->publish(v);
  }

  // ---------------- 발사대 ROS 쪽
  void on_enable(std::shared_ptr<rmw_request_id_t> h, std::shared_ptr<SetBool::Request> req) {
    if (enable_req_) {
      SetBool::Response r;
      r.success = false;
      r.message = "앞선 켜기·끄기 요청이 아직 끝나지 않음";
      enable_srv_->send_response(*h, r);
      return;
    }
    enable_req_ = h;   // 펌웨어 응답(LA,on / LA,off) 때 답한다
    const auto why = logic_->set_enabled(req->data, mono());
    if (!why.empty() && enable_req_) finish_enable(false, why);
  }
  void finish_enable(bool ok, const std::string& msg) {
    if (!enable_req_) return;
    SetBool::Response r;
    r.success = ok;
    r.message = msg;
    auto h = enable_req_;
    enable_req_.reset();
    enable_srv_->send_response(*h, r);
  }

  void on_goal(std::shared_ptr<GoalHandle> gh) {
    const auto mode = static_cast<LaunchMode>(gh->get_goal()->mode);
    if (goal_) {
      auto res = std::make_shared<Launch::Result>();
      res->success = false;
      res->message = "다른 발사대 동작이 진행 중";
      res->cocked = logic_->cocked();
      gh->abort(res);
      return;
    }
    goal_ = gh;
    const auto why = logic_->start_goal(mode, mono());
    if (!why.empty() && goal_ == gh) {
      goal_.reset();
      auto res = std::make_shared<Launch::Result>();
      res->success = false;
      res->message = "거절: " + why;
      res->cocked = logic_->cocked();
      gh->abort(res);
      RCLCPP_WARN(get_logger(), "발사대 동작 거절: %s", why.c_str());
      set_status("거절: " + why);
    }
  }
  void goal_feedback(const std::string& phase) {
    if (!goal_) return;
    auto fb = std::make_shared<Launch::Feedback>();
    fb->phase = phase;
    goal_->publish_feedback(fb);
  }
  void goal_done(GoalEnd e, const std::string& msg) {
    if (!goal_) return;
    auto gh = goal_;
    goal_.reset();
    auto res = std::make_shared<Launch::Result>();
    res->success = e == GoalEnd::kSucceeded;
    res->message = msg;
    res->cocked = logic_->cocked();
    if (e == GoalEnd::kSucceeded) gh->succeed(res);
    else if (e == GoalEnd::kCanceled && gh->is_canceling()) gh->canceled(res);
    else gh->abort(res);
    if (e == GoalEnd::kSucceeded) RCLCPP_INFO(get_logger(), "%s", msg.c_str());
    else RCLCPP_WARN(get_logger(), "%s", msg.c_str());
  }

  void publish_state() {
    if (!state_pub_) return;
    const std::string s = state_name(logic_->state());
    if (s != last_state_) {
      std_msgs::msg::String m;
      m.data = s;
      state_pub_->publish(m);
      last_state_ = s;
    }
    set_status(logic_->last_result());
  }
  void set_status(const std::string& text) {
    if (!status_pub_ || text.empty() || text == last_status_) return;
    std_msgs::msg::String m;
    m.data = text;
    status_pub_->publish(m);
    last_status_ = text;
    RCLCPP_INFO(get_logger(), "발사대: %s [%s]", text.c_str(), last_state_.c_str());
  }
  void publish_lever(double deg) {
    lever_deg_ = deg;
    if (!joint_pub_) return;
    sensor_msgs::msg::JointState j;   // 서보 명령 각(펌웨어 기록) — 잰 각이 아니다. URDF 에 이 관절은 없다(README)
    j.header.stamp = now();
    j.name = {lever_joint_};
    j.position = {deg * M_PI / 180.0};
    joint_pub_->publish(j);
  }

  // ---------------- 진단 (/diagnostics, 1 Hz)
  void diag_link(diagnostic_updater::DiagnosticStatusWrapper& st) {
    const auto t = std::chrono::steady_clock::now();
    const double dt = std::max(1e-3, std::chrono::duration<double>(t - last_rate_).count());
    last_rate_ = t;
    const double hb_age = (now() - last_hb_).seconds();
    if (fd_ < 0) st.summary(diagnostic_msgs::msg::DiagnosticStatus::ERROR, "포트가 열려 있지 않음");
    else if (hb_age > hb_timeout_) st.summary(diagnostic_msgs::msg::DiagnosticStatus::WARN, "살아 있음 줄이 끊김");
    else st.summary(diagnostic_msgs::msg::DiagnosticStatus::OK, "연결됨");
    st.add("port", port_);
    st.add("baud", baud_);
    st.add("heartbeat_age_s", hb_age);
    for (const char* k : {"US", "TH", "RD", "HB", "LA"}) {
      const size_t c = counts_[k];
      st.add(std::string("rate_") + k + "_hz", (c - prev_counts_[k]) / dt);
      prev_counts_[k] = c;
    }
    st.add("bad_lines_total", bad_);
    st.add("firmware_boots", launcher_on_ ? logic_->reboots() : 0);
  }
  void diag_launcher(diagnostic_updater::DiagnosticStatusWrapper& st) {
    const auto s = logic_->state();
    if (s == LauncherState::kFault) st.summary(diagnostic_msgs::msg::DiagnosticStatus::ERROR, "고장: " + logic_->fault());
    else if (s == LauncherState::kDisconnected) st.summary(diagnostic_msgs::msg::DiagnosticStatus::WARN, "펌웨어 발사대 응답 없음");
    else st.summary(diagnostic_msgs::msg::DiagnosticStatus::OK, state_name(s));
    st.add("state", state_name(s));
    st.add("cocked", logic_->cocked());
    st.add("lever_on", logic_->lever_on());
    st.add("lever_command_deg", lever_deg_);
    st.add("firing_enabled", logic_->firing_enabled());
    st.add("last_result", logic_->last_result());
  }

  std::string port_;
  int baud_;
  double hb_timeout_, fov_, min_, max_;
  std::vector<std::string> names_, frames_;
  std::string thermal_frame_, radar_frame_, lever_joint_;
  bool launcher_on_ = true;
  diagnostic_updater::Updater diag_;
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr thermal_pub_;
  rclcpp::Publisher<sensor_msgs::msg::Temperature>::SharedPtr thermal_max_pub_;
  rclcpp::Publisher<geometry_msgs::msg::PoseArray>::SharedPtr radar_pub_;
  rclcpp::Publisher<std_msgs::msg::Float32MultiArray>::SharedPtr radar_vel_pub_;
  std::vector<rclcpp::Publisher<sensor_msgs::msg::Range>::SharedPtr> pubs_;
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr state_pub_, status_pub_;
  rclcpp::Publisher<sensor_msgs::msg::JointState>::SharedPtr joint_pub_;
  rclcpp::Service<SetBool>::SharedPtr enable_srv_;
  rclcpp_action::Server<Launch>::SharedPtr action_;
  std::shared_ptr<GoalHandle> goal_;
  std::shared_ptr<rmw_request_id_t> enable_req_;
  std::unique_ptr<LauncherLogic> logic_;
  std::string last_state_, last_status_;
  double lever_deg_ = std::nan("");
  rclcpp::TimerBase::SharedPtr timer_;
  int fd_ = -1;
  std::string buf_;
  size_t bad_ = 0;
  std::map<std::string, size_t> counts_, prev_counts_;
  std::chrono::steady_clock::time_point last_rate_;
  rclcpp::Time last_hb_, last_try_{0, 0, RCL_ROS_TIME};
};

}  // namespace limo_fire_esp32_bridge

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<limo_fire_esp32_bridge::Esp32Bridge>());
  rclcpp::shutdown();
  return 0;
}
