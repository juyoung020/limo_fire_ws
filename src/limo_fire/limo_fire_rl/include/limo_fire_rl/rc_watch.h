// 리모컨 상태(/rc_state, limo_fire_base 가 sensor_msgs/Joy 를 RELIABLE · 깊이 10 · 약 40 Hz 로 냄) 받기를 튼튼하게 (low_rl):
//  - QoS: SensorData (best effort, 최근 5) — RELIABLE · BEST_EFFORT 발행자 모두와 맞는다 (구독이 best effort 면 늘 맞음, ROS 2 QoS 호환 표).
//  - 나이: 마지막으로 받은 때를 steady 시계로 (ROS 시계 · use_sim_time 과 무관).
//  - 다시 구독: 발행자가 있는데 지금 구독으로 resub_s 넘게 하나도 못 받거나 마지막 뒤 resub_s 가 지나면 구독을 새로 만든다
//    (2026-11-17 실물: 섀시 드라이버를 둔 채 low_rl 을 여러 번 다시 띄우자 새 low_rl 이 /rc_state 를 못 받음 — `ros2 topic echo` 엔 옴. 까닭은 못 가림)
#pragma once
#include <chrono>
#include <cstddef>

#include <rclcpp/qos.hpp>

namespace limo_fire_rl {

inline rclcpp::QoS rc_qos() { return rclcpp::SensorDataQoS().keep_last(5); }

class RcWatch {
 public:
  using clk = std::chrono::steady_clock;
  void subscribed(clk::time_point t) { t_sub_ = t; n_since_sub_ = 0; ++n_sub_; }
  void got(clk::time_point t) { t_last_ = t; ++n_; ++n_since_sub_; }
  // 마지막으로 받은 뒤 (s, 한 번도 못 받았으면 아주 큼)
  double age_s(clk::time_point t) const { return n_ ? secs(t - t_last_) : 1e9; }
  long count() const { return n_; }
  int resubs() const { return n_sub_ > 0 ? n_sub_ - 1 : 0; }
  bool want_resub(clk::time_point t, size_t publishers, double resub_s) const {
    if (publishers == 0 || secs(t - t_sub_) <= resub_s) return false;
    return n_since_sub_ == 0 || age_s(t) > resub_s;
  }

 private:
  static double secs(clk::duration d) { return std::chrono::duration<double>(d).count(); }
  clk::time_point t_sub_{}, t_last_{};
  long n_ = 0, n_since_sub_ = 0;
  int n_sub_ = 0;
};

}  // namespace limo_fire_rl
