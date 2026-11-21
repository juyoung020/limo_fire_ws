// 시뮬 초음파 → 실물 ESP32 다리와 같은 꼴로 다시 냄 (sim_robot.launch.xml 이 늘 띄움, 설정 config/sonar_sim*.yaml).
// Webots 드라이버가 /sim/ultrasonic/<이름> 으로 낸 Range 를 받아 /ultrasonic/<이름> 으로 낸다 — 연기 모드·탐사·BT 는 실물과 같은 토픽만 본다.
// 계산은 include/limo_fire_sim/sonar_sim.hpp (메아리 없음 +inf · 놓침 · 고장 · 치우침·잡음).
// 시각:
//   latency_s     — 측정한 값을 이만큼 늦게 내고, 시각은 낸 때로 찍음 (실물 다리는 줄을 받은 시각 now() 로 찍는다: 값은 그만큼 옛것)
//   stamp_ahead_s — 찍은 시각을 이만큼 더 앞으로 (실물에서 메아리 시각이 EKF odom TF 보다 앞서던 것을 시뮬에서 다시 보려고)
//   stamp_jitter_s — 위 둘에 [0, 이 값) 무작위를 더함
#include <deque>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/range.hpp>

#include "limo_fire_sim/sonar_sim.hpp"

class SonarSim : public rclcpp::Node
{
public:
  SonarSim()
  : Node("sonar_sim")
  {
    const auto names = declare_parameter<std::vector<std::string>>("names", {"front_left", "front_right"});
    const auto in_prefix = declare_parameter<std::string>("input_prefix", "/sim/ultrasonic/");
    const auto out_prefix = declare_parameter<std::string>("output_prefix", "/ultrasonic/");
    const auto dead = declare_parameter<std::vector<bool>>("dead", std::vector<bool>(names.size(), false));
    fov_ = declare_parameter<double>("field_of_view", 0.2618);
    base_.no_echo_ratio = declare_parameter<double>("no_echo_ratio", base_.no_echo_ratio);
    base_.min_range = declare_parameter<double>("min_range", base_.min_range);
    base_.max_range = declare_parameter<double>("max_range", base_.max_range);
    base_.drop_prob = declare_parameter<double>("drop_prob", base_.drop_prob);
    base_.bias_m = declare_parameter<double>("bias_m", base_.bias_m);
    base_.noise_sigma_m = declare_parameter<double>("noise_sigma_m", base_.noise_sigma_m);
    latency_ = declare_parameter<double>("latency_s", 0.0);
    ahead_ = declare_parameter<double>("stamp_ahead_s", 0.0);
    jitter_ = declare_parameter<double>("stamp_jitter_s", 0.0);
    const auto seed = declare_parameter<int>("seed", 0);
    if (names.empty() || dead.size() != names.size() || base_.no_echo_ratio <= 0.0 || base_.no_echo_ratio > 1.0 ||
      base_.min_range < 0.0 || base_.max_range <= base_.min_range || base_.drop_prob < 0.0 || base_.drop_prob > 1.0 ||
      base_.noise_sigma_m < 0.0 || latency_ < 0.0 || jitter_ < 0.0 || fov_ <= 0.0)
    {
      throw std::invalid_argument("sonar_sim: names·dead(같은 길이)·no_echo_ratio (0,1]·min/max_range·drop_prob [0,1]·σ·지연 ≥ 0 확인");
    }
    rng_.seed(seed > 0 ? static_cast<unsigned>(seed) : std::random_device{}());

    for (size_t i = 0; i < names.size(); ++i) {
      Effects e = base_;
      e.dead = dead[i];
      fx_.push_back(e);
      pubs_.push_back(create_publisher<sensor_msgs::msg::Range>(out_prefix + names[i], rclcpp::SensorDataQoS()));
      subs_.push_back(create_subscription<sensor_msgs::msg::Range>(in_prefix + names[i], rclcpp::SensorDataQoS(),
        [this, i](sensor_msgs::msg::Range::ConstSharedPtr m) {on_range(i, *m);}));
      if (e.dead) {RCLCPP_WARN(get_logger(), "초음파 %s 고장 흉내: 늘 메아리 없음", names[i].c_str());}
    }
    if (latency_ > 0.0) {
      // 지연 줄은 5 ms 마다 내보냄 (실물 다리 주기 약 14 Hz 보다 충분히 잘게). 벽시계 타이머라 Webots realtime 에서만 지연이 설정값과 맞다
      // (fast 모드면 그 사이 시뮬 시간이 더 흘러 실제 지연이 커짐) — 실물 흉내 설정은 mode:=realtime 으로
      timer_ = create_wall_timer(std::chrono::milliseconds(5), [this] {flush();});
    }
    RCLCPP_INFO(get_logger(), "초음파 %zu 개 %s* → %s*: 놓침 %.2f · 치우침 %.3f m · 잡음 %.3f m · 지연 %.3f s · 시각 앞섬 %.3f s (+%.3f)",
      names.size(), in_prefix.c_str(), out_prefix.c_str(), base_.drop_prob, base_.bias_m, base_.noise_sigma_m, latency_, ahead_, jitter_);
  }

private:
  using Effects = sonar_sim::Effects;
  struct Pending
  {
    size_t i;
    rclcpp::Time due;
    float range;
    std::string frame;
  };

  void on_range(size_t i, const sensor_msgs::msg::Range & m)
  {
    const float r = static_cast<float>(sonar_sim::apply(fx_[i], m.range, m.max_range, rng_));
    if (latency_ <= 0.0) {
      publish(i, r, m.header.frame_id);
      return;
    }
    q_.push_back({i, now() + rclcpp::Duration::from_seconds(latency_), r, m.header.frame_id});
  }

  void flush()
  {
    const auto t = now();
    while (!q_.empty() && q_.front().due <= t) {
      publish(q_.front().i, q_.front().range, q_.front().frame);
      q_.pop_front();
    }
  }

  void publish(size_t i, float r, const std::string & frame)
  {
    sensor_msgs::msg::Range o;
    double shift = ahead_;
    if (jitter_ > 0.0) {shift += std::uniform_real_distribution<double>(0.0, jitter_)(rng_);}
    o.header.stamp = now() + rclcpp::Duration::from_seconds(shift);
    o.header.frame_id = frame;   // Webots 드라이버가 URDF 와 같은 프레임으로 냄
    o.radiation_type = sensor_msgs::msg::Range::ULTRASOUND;
    o.field_of_view = static_cast<float>(fov_);
    o.min_range = static_cast<float>(fx_[i].min_range);
    o.max_range = static_cast<float>(fx_[i].max_range);
    o.range = r;
    pubs_[i]->publish(o);
  }

  Effects base_;
  std::vector<Effects> fx_;
  double fov_ = 0.2618, latency_ = 0.0, ahead_ = 0.0, jitter_ = 0.0;
  std::mt19937 rng_;
  std::deque<Pending> q_;
  std::vector<rclcpp::Publisher<sensor_msgs::msg::Range>::SharedPtr> pubs_;
  std::vector<rclcpp::Subscription<sensor_msgs::msg::Range>::SharedPtr> subs_;
  rclcpp::TimerBase::SharedPtr timer_;
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<SonarSim>());
  rclcpp::shutdown();
  return 0;
}
