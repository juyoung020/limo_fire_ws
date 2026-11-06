// 초음파 → 점 몇 개짜리 스캔 (연기 모드 초음파 위치 추정, docs/smoke_mode_plan.md 3 단계 · docs/spec.md 연기 모드 "위치 보정 1").
// 초음파 Range 여럿(실물 ESP32 다리 · 시뮬 Webots DistanceSensor, 같은 토픽·프레임)을 base_link 기준 LaserScan 하나로 묶어 AMCL 에 준다.
//
// 만드는 법: 센서마다 최신 거리 r 이 유효(min_range < r < max_range)하면 원뿔(field_of_view) 폭에 광선 rays_per_cone 개를 고르게 펴서
//   끝점 r·(cos θ, sin θ) 를 base_link 로 옮기고, base_link 원점에서 본 각도 칸(bins 개, 360°)에 넣는다(같은 칸이면 가까운 값).
//   나머지 칸은 +inf — AMCL 은 최대 거리 이상인 빔을 버리므로 아는 곳만 쓴다. 메아리 없음(max_range)도 정보 없음으로 +inf.
//   원뿔 안 어디서 반사됐는지 모르므로 점을 폭에 펴고, AMCL 측정 잡음(sigma_hit)이 그 몫을 받는다(spec.md "측정 노이즈를 크게").
// 칸 수는 AMCL max_beams(60) 이하로 둔다 — AMCL 은 스캔을 고르게 솎아 max_beams 개만 쓰므로, 칸이 더 많으면 몇 안 되는 유효 칸이 버려진다.
// 오래된 거리(max_age 초과)는 안 쓴다. 하나도 없으면 내지 않는다(빈 스캔으로 AMCL 을 속이지 않게).
#include <cmath>
#include <limits>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/laser_scan.hpp"
#include "sensor_msgs/msg/range.hpp"
#include "tf2/LinearMath/Transform.h"
#include "tf2_geometry_msgs/tf2_geometry_msgs.hpp"
#include "tf2_ros/buffer.h"
#include "tf2_ros/transform_listener.h"

namespace limo_fire_navigation {

class SonarScanNode : public rclcpp::Node {
 public:
  SonarScanNode() : Node("sonar_scan") {
    topics_ = declare_parameter<std::vector<std::string>>(
        "range_topics", {"/ultrasonic/front", "/ultrasonic/left", "/ultrasonic/right"});
    base_frame_ = declare_parameter<std::string>("base_frame", "base_link");
    bins_ = declare_parameter<int>("bins", 60);
    rays_ = declare_parameter<int>("rays_per_cone", 3);
    rate_ = declare_parameter<double>("rate", 10.0);
    max_age_ = declare_parameter<double>("max_age", 0.5);
    if (topics_.empty() || bins_ < 4 || rays_ < 1 || rate_ <= 0.0 || max_age_ <= 0.0)
      throw std::invalid_argument("sonar_scan: range_topics·bins(≥4)·rays_per_cone(≥1)·rate·max_age 확인");

    tf_buffer_ = std::make_unique<tf2_ros::Buffer>(get_clock());
    tf_listener_ = std::make_unique<tf2_ros::TransformListener>(*tf_buffer_);
    pub_ = create_publisher<sensor_msgs::msg::LaserScan>("sonar_scan", rclcpp::SensorDataQoS());
    for (const auto& t : topics_) {
      subs_.push_back(create_subscription<sensor_msgs::msg::Range>(
          t, rclcpp::SensorDataQoS(), [this, t](sensor_msgs::msg::Range::SharedPtr m) { last_[t] = m; }));
    }
    timer_ = create_wall_timer(std::chrono::duration<double>(1.0 / rate_), [this] { tick(); });
    RCLCPP_INFO(get_logger(), "초음파 %zu 개 → sonar_scan (%d 칸, 원뿔마다 점 %d 개)", topics_.size(), bins_, rays_);
  }

 private:
  void tick() {
    const double inf = std::numeric_limits<double>::infinity();
    std::vector<float> ranges(bins_, static_cast<float>(inf));
    const double amin = -M_PI, inc = 2.0 * M_PI / bins_;
    double rmax = 0.0, rmin = inf;
    rclcpp::Time stamp(0, 0, get_clock()->get_clock_type());
    int used = 0;
    const rclcpp::Time now_t = now();
    for (const auto& [topic, m] : last_) {
      if (!m) continue;
      const rclcpp::Time t(m->header.stamp, get_clock()->get_clock_type());
      if ((now_t - t).seconds() > max_age_) continue;   // 끊긴 센서는 안 씀
      rmax = std::max(rmax, static_cast<double>(m->max_range));
      rmin = std::min(rmin, static_cast<double>(m->min_range));
      const double r = m->range;
      if (!std::isfinite(r) || r <= m->min_range || r >= 0.99 * m->max_range) continue;   // 메아리 없음 = 정보 없음
      tf2::Transform bs;
      try {
        tf2::fromMsg(tf_buffer_->lookupTransform(base_frame_, m->header.frame_id, tf2::TimePointZero).transform, bs);
      } catch (const tf2::TransformException& e) {
        RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 5000, "%s → %s TF 없음: %s", base_frame_.c_str(),
                             m->header.frame_id.c_str(), e.what());
        continue;
      }
      const double fov = m->field_of_view;
      for (int k = 0; k < rays_; ++k) {
        const double th = rays_ == 1 ? 0.0 : -0.5 * fov + fov * k / (rays_ - 1);
        const tf2::Vector3 p = bs * tf2::Vector3(r * std::cos(th), r * std::sin(th), 0.0);
        const double a = std::atan2(p.y(), p.x()), d = std::hypot(p.x(), p.y());
        int i = static_cast<int>(std::lround((a - amin) / inc));   // 가장 가까운 칸 (칸 각도 = angle_min + i·inc)
        i = (i % bins_ + bins_) % bins_;
        if (d < ranges[i]) ranges[i] = static_cast<float>(d);
      }
      if (t > stamp) stamp = t;
      ++used;
    }
    if (used == 0) return;
    sensor_msgs::msg::LaserScan s;
    s.header.stamp = stamp;
    s.header.frame_id = base_frame_;
    s.angle_min = amin;
    s.angle_increment = inc;
    s.angle_max = amin + inc * (bins_ - 1);
    s.scan_time = 1.0 / rate_;
    s.range_min = std::isfinite(rmin) ? rmin : 0.0;
    s.range_max = rmax;
    s.ranges = std::move(ranges);
    pub_->publish(s);
  }

  std::vector<std::string> topics_;
  std::string base_frame_;
  int bins_, rays_;
  double rate_, max_age_;
  std::map<std::string, sensor_msgs::msg::Range::SharedPtr> last_;
  std::unique_ptr<tf2_ros::Buffer> tf_buffer_;
  std::unique_ptr<tf2_ros::TransformListener> tf_listener_;
  rclcpp::Publisher<sensor_msgs::msg::LaserScan>::SharedPtr pub_;
  std::vector<rclcpp::Subscription<sensor_msgs::msg::Range>::SharedPtr> subs_;
  rclcpp::TimerBase::SharedPtr timer_;
};

}  // namespace limo_fire_navigation

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<limo_fire_navigation::SonarScanNode>());
  rclcpp::shutdown();
  return 0;
}
