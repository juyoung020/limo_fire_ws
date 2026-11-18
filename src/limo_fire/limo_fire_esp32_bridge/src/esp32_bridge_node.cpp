// ESP32 ↔ ROS 다리 — 지금은 초음파 HC-SR04 × 3 (연기 모드, docs/smoke_mode_plan.md 5 단계).
// USB 시리얼 줄(firmware/limo_fire_esp32, protocol.hpp)을 읽어 초음파마다 sensor_msgs/Range 를 낸다.
// 토픽·프레임은 시뮬(Webots LimoFire ultrasonic)과 같다: /ultrasonic/<이름>, ultrasonic_<이름>_link (장착 자리는 URDF ultrasonic.xacro).
//   메아리 없음(0 mm) → range = +inf (sensor_msgs/Range 규약: 고정 거리 안에 없음)
// 포트가 없거나 끊기면 알리고 1 초마다 다시 연다. 살아 있음 줄($HB)이 heartbeat_timeout 넘게 안 오면 알린다.
#include <fcntl.h>
#include <termios.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstring>
#include <limits>
#include <memory>
#include <string>
#include <vector>

#include "limo_fire_esp32_bridge/protocol.hpp"
#include "rclcpp/rclcpp.hpp"
#include "geometry_msgs/msg/pose_array.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "sensor_msgs/msg/range.hpp"
#include "std_msgs/msg/float32.hpp"
#include "std_msgs/msg/float32_multi_array.hpp"

namespace limo_fire_esp32_bridge {

class Esp32Bridge : public rclcpp::Node {
 public:
  Esp32Bridge() : Node("esp32_bridge") {
    port_ = declare_parameter<std::string>("port", "/dev/esp32");
    baud_ = declare_parameter<int>("baud", 115200);
    hb_timeout_ = declare_parameter<double>("heartbeat_timeout", 3.0);
    names_ = declare_parameter<std::vector<std::string>>("ultrasonic.names", {"front_left", "front_right"});
    fov_ = declare_parameter<double>("ultrasonic.field_of_view", 0.2618);
    min_ = declare_parameter<double>("ultrasonic.min_range", 0.02);
    max_ = declare_parameter<double>("ultrasonic.max_range", 4.0);
    thermal_frame_ = declare_parameter<std::string>("thermal.frame_id", "thermal_link");
    radar_frame_ = declare_parameter<std::string>("radar.frame_id", "radar_link");
    if (names_.empty() || fov_ <= 0.0 || min_ < 0.0 || max_ <= min_ || hb_timeout_ <= 0.0)
      throw std::invalid_argument("esp32_bridge: ultrasonic.names·field_of_view·min/max_range·heartbeat_timeout 확인");
    if (speed_of(baud_) == B0) throw std::invalid_argument("esp32_bridge: 지원하지 않는 baud " + std::to_string(baud_));
    for (const auto& n : names_) {
      pubs_.push_back(create_publisher<sensor_msgs::msg::Range>("/ultrasonic/" + n, rclcpp::SensorDataQoS()));
      frames_.push_back("ultrasonic_" + n + "_link");
    }
    thermal_pub_ = create_publisher<sensor_msgs::msg::Image>("/thermal/image", rclcpp::SensorDataQoS());
    thermal_max_pub_ = create_publisher<std_msgs::msg::Float32>("/thermal/max", rclcpp::SensorDataQoS());
    radar_pub_ = create_publisher<geometry_msgs::msg::PoseArray>("/radar/targets", rclcpp::SensorDataQoS());
    radar_vel_pub_ = create_publisher<std_msgs::msg::Float32MultiArray>("/radar/velocities", rclcpp::SensorDataQoS());
    last_hb_ = now();
    timer_ = create_wall_timer(std::chrono::milliseconds(5), [this] { poll(); });
    RCLCPP_INFO(get_logger(), "ESP32 %s @ %d, 초음파 %zu 개 · 열화상 · 레이더", port_.c_str(), baud_, names_.size());
  }
  ~Esp32Bridge() override { close_port(); }

 private:
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

  bool open_port() {
    fd_ = ::open(port_.c_str(), O_RDONLY | O_NOCTTY | O_NONBLOCK);
    if (fd_ < 0) return false;
    termios t{};
    if (tcgetattr(fd_, &t) != 0) { close_port(); return false; }
    cfmakeraw(&t);
    cfsetispeed(&t, speed_of(baud_));
    cfsetospeed(&t, speed_of(baud_));
    t.c_cflag |= CLOCAL | CREAD;
    if (tcsetattr(fd_, TCSANOW, &t) != 0) { close_port(); return false; }
    tcflush(fd_, TCIFLUSH);
    buf_.clear();
    RCLCPP_INFO(get_logger(), "ESP32 포트 열림: %s", port_.c_str());
    return true;
  }
  void close_port() {
    if (fd_ >= 0) ::close(fd_);
    fd_ = -1;
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
      if (n == 0 || (errno != EAGAIN && errno != EWOULDBLOCK)) {
        RCLCPP_ERROR(get_logger(), "ESP32 포트 끊김 (%s) — 다시 엶", n == 0 ? "EOF" : std::strerror(errno));
        close_port();
      }
      break;
    }
    size_t nl;
    while ((nl = buf_.find('\n')) != std::string::npos) {
      handle(buf_.substr(0, nl));
      buf_.erase(0, nl + 1);
    }
    if (buf_.size() > 16384) buf_.clear();   // 줄 끝이 안 오는 쓰레기
    if (fd_ >= 0 && (now() - last_hb_).seconds() > hb_timeout_)
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 5000, "ESP32 살아 있음 줄이 %.0f s 넘게 없음", hb_timeout_);
  }

  void handle(const std::string& raw) {
    const auto l = parse_line(raw);
    if (!l) { ++bad_; RCLCPP_DEBUG(get_logger(), "깨진 줄 버림 (%zu 번째)", bad_); return; }
    if (l->type == "HB") { last_hb_ = now(); return; }
    if (l->type == "TH") { handle_thermal(*l); return; }
    if (l->type == "RD") { handle_radar(*l); return; }
    const auto s = as_sonar(*l);
    if (!s) return;
    if (s->index >= static_cast<int>(pubs_.size())) {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 5000, "초음파 번호 %d 가 설정(%zu 개) 밖", s->index, pubs_.size());
      return;
    }
    sensor_msgs::msg::Range m;
    m.header.stamp = now();   // ESP32 시각(ms)은 시계가 달라 쓰지 않는다 — 받은 시각 (USB 지연 수 ms)
    m.header.frame_id = frames_[s->index];
    m.radiation_type = sensor_msgs::msg::Range::ULTRASOUND;
    m.field_of_view = static_cast<float>(fov_);
    m.min_range = static_cast<float>(min_);
    m.max_range = static_cast<float>(max_);
    m.range = s->mm == 0 ? std::numeric_limits<float>::infinity() : static_cast<float>(s->mm * 1e-3);
    pubs_[s->index]->publish(m);
  }

  void handle_thermal(const Line& l) {
    const auto t = as_thermal(l);
    if (!t) { ++bad_; RCLCPP_DEBUG(get_logger(), "열화상 줄 형식 틀림"); return; }
    sensor_msgs::msg::Image m;
    m.header.stamp = now();
    m.header.frame_id = thermal_frame_;
    m.height = kThermalRows;
    m.width = kThermalCols;
    m.encoding = "32FC1";   // 값 = °C
    m.is_bigendian = 0;
    m.step = kThermalCols * sizeof(float);
    m.data.resize(kThermalPoints * sizeof(float));
    std::memcpy(m.data.data(), t->c.data(), m.data.size());
    thermal_pub_->publish(m);
    std_msgs::msg::Float32 mx;
    mx.data = *std::max_element(t->c.begin(), t->c.end());
    thermal_max_pub_->publish(mx);
  }

  // LD2450 (x 오른쪽 + · y 앞, mm) → REP-103 (x 앞 · y 왼쪽, m): x_ros = y, y_ros = −x. 속도는 LD2450 원본 부호 그대로 m/s
  // (부호가 다가옴 +/멀어짐 + 중 어느 쪽인지는 실물에서 확인 전 — 출처마다 다르다)
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

  std::string port_;
  int baud_;
  double hb_timeout_, fov_, min_, max_;
  std::vector<std::string> names_, frames_;
  std::string thermal_frame_, radar_frame_;
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr thermal_pub_;
  rclcpp::Publisher<std_msgs::msg::Float32>::SharedPtr thermal_max_pub_;
  rclcpp::Publisher<geometry_msgs::msg::PoseArray>::SharedPtr radar_pub_;
  rclcpp::Publisher<std_msgs::msg::Float32MultiArray>::SharedPtr radar_vel_pub_;
  std::vector<rclcpp::Publisher<sensor_msgs::msg::Range>::SharedPtr> pubs_;
  rclcpp::TimerBase::SharedPtr timer_;
  int fd_ = -1;
  std::string buf_;
  size_t bad_ = 0;
  rclcpp::Time last_hb_, last_try_{0, 0, RCL_ROS_TIME};
};

}  // namespace limo_fire_esp32_bridge

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<limo_fire_esp32_bridge::Esp32Bridge>());
  rclcpp::shutdown();
  return 0;
}
