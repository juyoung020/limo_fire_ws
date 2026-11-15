// 아래층 RL 실물 노드 — 시험(dry run)만: /scan 한 장 = 시뮬 걸음 하나(0.1 s)로 RobotEnv 아래층 모드(학습 시뮬과 같은 함수)에 쌓아
// 아래층 정책(순수 RL, network/policy_cpu.h LowPolicy)의 관측을 만들고, 정책이 낸 명령(v, ω)을 디버그 토픽(~/cmd_debug)에만 낸다.
// /cmd_vel 발행자는 아예 없다 — 로봇을 몰지 않는다 (dry_run:=false 면 시작하지 않음).
// 걸음 차례 (시뮬 env_ref.cpp step_env + low_prepare_env 와 같게 — limo_fire_rl/README "아래층 정책 관측"):
//   장 k: step(감각, 장 k) → 깊이 → [새 목적지면 set_goal] → low_obs → 정책 → low_act (명령 = 다음 걸음 명령, 사람 추적 포함) → ~/cmd_debug
// 목적지: ~/goal (geometry_msgs/PoseStamped, map 틀). 위치가 NaN 이면 목적지 없음. 목적지가 없으면 정책 환경이 아니라 명령 0 (시뮬은 규칙 추종기).
// 관측 설정(넓은 지도·움직이는 것 칸·깊이 층 주기·추적 값 …)은 학습처럼 환경변수 SIM_LOW_* 로 읽는다 (env_cfg.cpp) — 학습 판과 같은 값을 export 할 것.
// record:=<파일> 이면 받은 입력과 만든 관측·명령을 sim/tools/low_node_rec.h 꼴로 녹화 (low_node_replay 로 다시 돌려 같은지·시간).
#include <chrono>
#include <cmath>
#include <cstdio>
#include <memory>
#include <string>
#include <vector>

#include <geometry_msgs/msg/pose_stamped.hpp>
#include <geometry_msgs/msg/twist_stamped.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <sensor_msgs/msg/laser_scan.hpp>
#include <std_msgs/msg/string.hpp>
#include <tf2/LinearMath/Matrix3x3.h>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>

#include "limo_fire_rl/robot_env.h"
#include "low_node_rec.h"
#include "policy_cpu.h"

namespace limo_fire_rl {

class LowRlNode : public rclcpp::Node {
 public:
  LowRlNode() : Node("low_rl") {
    env_yaml_ = declare_parameter<std::string>("env_yaml", LIMO_FIRE_RL_ENV_YAML);
    ckpt_ = declare_parameter<std::string>("checkpoint", "");
    dry_run_ = declare_parameter<bool>("dry_run", true);
    map_frame_ = declare_parameter<std::string>("map_frame", "map");
    base_frame_ = declare_parameter<std::string>("base_frame", "base_link");
    tf_timeout_ = declare_parameter<double>("tf_timeout", 0.2);
    depth_timeout_ = declare_parameter<double>("depth_timeout", 0.5);
    record_ = declare_parameter<std::string>("record", "");
    const bool camera_layer = declare_parameter<bool>("camera_layer", true);
    const std::string scan_topic = declare_parameter<std::string>("scan_topic", "/scan");
    const std::string odom_topic = declare_parameter<std::string>("odom_topic", "/odom");
    const std::string imu_topic = declare_parameter<std::string>("imu_topic", "/imu");
    const std::string depth_topic = declare_parameter<std::string>("depth_topic", "/camera/depth/image_raw");
    const std::string info_topic = declare_parameter<std::string>("depth_info_topic", "/camera/depth/camera_info");
    if (!dry_run_) throw std::runtime_error("low_rl: 실제 주행(dry_run:=false)은 아직 없음 — 명령은 ~/cmd_debug 에만 낸다");
    if (ckpt_.empty()) throw std::runtime_error("checkpoint 파라미터 (아래층 체크포인트 .bin — train --low --gru, SIM_LOW_WIDE=1) 가 비었음");
    hc_ = sim::load_env_cfg(env_yaml_);
    if (!camera_layer) hc_.cfg.cam_on = 0;
    env_ = std::make_unique<RobotEnv>(hc_, true);
    pol_.load(ckpt_);
    if (pol_.n1d() != env_->low_1d_n())
      throw std::runtime_error("체크포인트 1D " + std::to_string(pol_.n1d()) + " 칸 ≠ 관측 설정 " + std::to_string(env_->low_1d_n()) +
                               " 칸 — 학습 판과 같은 SIM_LOW_OBS_MOV · SIM_LOW_MOV4 를 export");
    if (env_->low_img_n() != mdp::low::IMG_WIDE) throw std::runtime_error("넓은 지도 정책만 (SIM_LOW_WIDE=1 을 export)");
    const sim::EnvCfg& c = env_->cfg();
    RCLCPP_INFO(get_logger(), "아래층 정책 %s (반복 %d, GRU %d, 1D %d) · 넓은 지도 %d · 목적지만 %d · 움직이는 것 %d · 칸 넷 %d · 깊이 층 걸음마다 %d · 추적 놓침 %d · "
                "후진 없음 %d · 깊이 카메라 층 %s — dry run: 명령은 ~/cmd_debug 에만", ckpt_.c_str(), pol_.iteration(), pol_.hidden(), pol_.n1d(), c.lo_wide,
                c.lo_goalonly, c.lo_mov, c.lo_mov4, c.lo_cam_every, c.tk_miss, c.lg_noback, c.cam_on ? "켬" : "끔");
    if (!record_.empty()) rec_ = std::make_unique<lowrec::Writer>(record_, hc_, env_->low_img_n(), env_->low_1d_n());

    tf_buf_ = std::make_shared<tf2_ros::Buffer>(get_clock());
    tf_lis_ = std::make_shared<tf2_ros::TransformListener>(*tf_buf_);
    const auto sensor = rclcpp::SensorDataQoS();
    sub_scan_ = create_subscription<sensor_msgs::msg::LaserScan>(scan_topic, sensor, [this](sensor_msgs::msg::LaserScan::ConstSharedPtr m) { on_scan(m); });
    sub_odom_ = create_subscription<nav_msgs::msg::Odometry>(odom_topic, 20, [this](nav_msgs::msg::Odometry::ConstSharedPtr m) { odom_ = m; });
    sub_imu_ = create_subscription<sensor_msgs::msg::Imu>(imu_topic, 50, [this](sensor_msgs::msg::Imu::ConstSharedPtr m) { imu_ = m; });
    sub_depth_ = create_subscription<sensor_msgs::msg::Image>(depth_topic, sensor, [this](sensor_msgs::msg::Image::ConstSharedPtr m) { depth_ = m; });
    sub_info_ = create_subscription<sensor_msgs::msg::CameraInfo>(info_topic, sensor, [this](sensor_msgs::msg::CameraInfo::ConstSharedPtr m) { info_ = m; });
    sub_goal_ = create_subscription<geometry_msgs::msg::PoseStamped>("~/goal", 10, [this](geometry_msgs::msg::PoseStamped::ConstSharedPtr m) { on_goal(m); });
    pub_cmd_ = create_publisher<geometry_msgs::msg::TwistStamped>("~/cmd_debug", 10);
    pub_status_ = create_publisher<std_msgs::msg::String>("~/status", 10);
  }

 private:
  void on_goal(const geometry_msgs::msg::PoseStamped::ConstSharedPtr& m) {
    if (!m->header.frame_id.empty() && m->header.frame_id != map_frame_) {
      RCLCPP_WARN(get_logger(), "~/goal 틀 %s ≠ %s — 버림", m->header.frame_id.c_str(), map_frame_.c_str());
      return;
    }
    const double x = m->pose.position.x, y = m->pose.position.y;
    pending_goal_ = (std::isfinite(x) && std::isfinite(y)) ? std::make_pair((float)x, (float)y) : std::make_pair(NAN, NAN);
    have_pending_ = true;
  }

  Proprio proprio() const {
    Proprio p;
    p.wheel_v = (float)odom_->twist.twist.linear.x;
    p.wheel_vy = (float)odom_->twist.twist.linear.y;
    p.wheel_w = (float)odom_->twist.twist.angular.z;
    p.gyro_z = (float)imu_->angular_velocity.z;
    p.imu[0] = (float)imu_->linear_acceleration.x;
    p.imu[1] = (float)imu_->linear_acceleration.y;
    p.imu[2] = (float)imu_->linear_acceleration.z;
    p.imu[3] = (float)imu_->angular_velocity.x;
    p.imu[4] = (float)imu_->angular_velocity.y;
    tf2::Quaternion q(imu_->orientation.x, imu_->orientation.y, imu_->orientation.z, imu_->orientation.w);
    double r, pi, yw;
    tf2::Matrix3x3(q).getRPY(r, pi, yw);
    p.imu[5] = (float)r;
    p.imu[6] = (float)pi;
    return p;   // cmd_v·cmd_w 는 아래층 모드가 안 씀 (명령은 low_act 가 정함)
  }

  static double yaw_of(const geometry_msgs::msg::Quaternion& q) {
    return std::atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z));
  }

  void on_scan(const sensor_msgs::msg::LaserScan::ConstSharedPtr& m) {
    if (!odom_ || !imu_) {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000, "오도메트리·IMU 기다림");
      return;
    }
    geometry_msgs::msg::TransformStamped T;
    try {
      T = tf_buf_->lookupTransform(map_frame_, base_frame_, m->header.stamp, rclcpp::Duration::from_seconds(tf_timeout_));
    } catch (const std::exception& ex) {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000, "TF %s → %s (장 시각) 없음: %s — 이 장 건너뜀", map_frame_.c_str(), base_frame_.c_str(), ex.what());
      return;
    }
    Scan s;
    s.ranges.assign(m->ranges.begin(), m->ranges.end());
    s.intensities.assign(m->intensities.begin(), m->intensities.end());
    s.angle_min = m->angle_min;
    s.angle_increment = m->angle_increment;
    s.time_increment = m->time_increment;
    s.x = (float)T.transform.translation.x;
    s.y = (float)T.transform.translation.y;
    s.th = (float)yaw_of(T.transform.rotation);
    s.w = (float)imu_->angular_velocity.z;
    const Proprio p = proprio();
    using clk = std::chrono::steady_clock;
    const auto t0 = clk::now();
    if (!started_) {   // 판 시작 (시뮬 리셋 관측 — 장 없음): 관측·정책만, 이 장은 다음 걸음 것이 아님
      env_->reset(s.x, s.y, s.th, p, -2.f * s.angle_min);
      if (rec_) rec_->reset(s.x, s.y, s.th, p, -2.f * s.angle_min, RobotEnv::Window{env_->map().ox, env_->map().oy, env_->map().gw, env_->map().gh});
      started_ = true;
      RCLCPP_INFO(get_logger(), "시작 자세 (%.2f, %.2f, %.1f°)", s.x, s.y, s.th * 57.2958);
    } else {
      if (rec_) rec_->step(p, s);
      env_->step(p, s);
      if (env_->done() && env_->done() != last_done_)
        RCLCPP_WARN(get_logger(), "시뮬 규칙이면 판 끝 (코드 %d — 닿음·목표 수 등). 실물은 이어 감 (사람 추적·목표 상태는 시뮬처럼 비움)", env_->done());
      last_done_ = env_->done();
    }
    const double step_ms = std::chrono::duration<double, std::milli>(clk::now() - t0).count();
    // 깊이 (센서 입력): 최근 장 → 40 × 64 (없으면 0 — 구멍)
    std::vector<uint8_t> dep((size_t)mdp::obs::DEPTH_SIZE, 0);
    if (depth_ && info_ && (now() - rclcpp::Time(depth_->header.stamp)).seconds() < depth_timeout_ && depth_->encoding == "16UC1") {
      env_->depth_from_image(reinterpret_cast<const uint16_t*>(depth_->data.data()), (int)depth_->width, (int)depth_->height, (int)(depth_->step / 2),
                             (float)info_->k[0], (float)info_->k[4], (float)info_->k[2], (float)info_->k[5], dep.data());
    } else {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 5000, "깊이 영상 없음·오래됨·16UC1 아님 — 깊이 관측 0 (구멍)");
    }
    env_->set_depth(dep.data());
    if (rec_) rec_->depth(dep.data());
    if (have_pending_) {
      const int cell = std::isfinite(pending_goal_.first) ? env_->goal_of(pending_goal_.first, pending_goal_.second) : -1;
      if (std::isfinite(pending_goal_.first) && cell < 0) RCLCPP_WARN(get_logger(), "목적지 (%.2f, %.2f) 가 지도 창 밖 — 목적지 없음", pending_goal_.first, pending_goal_.second);
      env_->set_goal(cell);
      if (rec_) rec_->goal(cell);
      have_pending_ = false;
      RCLCPP_INFO(get_logger(), "목적지 칸 %d", cell);
    }
    const auto t1 = clk::now();
    env_->low_obs();
    if (rec_) rec_->obs(env_->low_img(), env_->low_1d());
    const auto t2 = clk::now();
    float a[mdp::low::ACT], v = 0.f, w = 0.f;
    pol_.act(env_->low_img(), env_->low_1d(), a);
    const auto t3 = clk::now();
    env_->low_act(a, &v, &w);
    if (rec_) rec_->act(a, v, w);
    const double obs_ms = std::chrono::duration<double, std::milli>(t2 - t1).count(), pol_ms = std::chrono::duration<double, std::milli>(t3 - t2).count();
    geometry_msgs::msg::TwistStamped cmd;
    cmd.header.stamp = m->header.stamp;
    cmd.header.frame_id = base_frame_;
    cmd.twist.linear.x = v;
    cmd.twist.angular.z = w;
    pub_cmd_->publish(cmd);
    sum_step_ += step_ms;
    sum_obs_ += obs_ms;
    sum_pol_ += pol_ms;
    if (++n_ % 10 == 0) {
      std_msgs::msg::String st;
      char buf[256];
      std::snprintf(buf, sizeof(buf), "걸음 %d · 목적지 %d · 명령 v %.2f ω %.2f · 평균 ms: 걸음 %.2f 관측 %.2f 정책 %.2f", env_->steps(), env_->low_cmd().target, v, w,
                    sum_step_ / n_, sum_obs_ / n_, sum_pol_ / n_);
      st.data = buf;
      pub_status_->publish(st);
    }
  }

  std::string env_yaml_, ckpt_, map_frame_, base_frame_, record_;
  bool dry_run_ = true, started_ = false, have_pending_ = false;
  double tf_timeout_ = 0.2, depth_timeout_ = 0.5, sum_step_ = 0, sum_obs_ = 0, sum_pol_ = 0;
  int n_ = 0, last_done_ = 0;
  std::pair<float, float> pending_goal_{NAN, NAN};
  sim::EnvHostCfg hc_;
  std::unique_ptr<RobotEnv> env_;
  xnet_cpu::LowPolicy pol_;
  std::unique_ptr<lowrec::Writer> rec_;
  std::shared_ptr<tf2_ros::Buffer> tf_buf_;
  std::shared_ptr<tf2_ros::TransformListener> tf_lis_;
  sensor_msgs::msg::Imu::ConstSharedPtr imu_;
  nav_msgs::msg::Odometry::ConstSharedPtr odom_;
  sensor_msgs::msg::Image::ConstSharedPtr depth_;
  sensor_msgs::msg::CameraInfo::ConstSharedPtr info_;
  rclcpp::Subscription<sensor_msgs::msg::LaserScan>::SharedPtr sub_scan_;
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr sub_odom_;
  rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr sub_imu_;
  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr sub_depth_;
  rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr sub_info_;
  rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr sub_goal_;
  rclcpp::Publisher<geometry_msgs::msg::TwistStamped>::SharedPtr pub_cmd_;
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr pub_status_;
};

}  // namespace limo_fire_rl

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  int rc = 0;
  try {
    rclcpp::spin(std::make_shared<limo_fire_rl::LowRlNode>());
  } catch (const std::exception& ex) {
    std::fprintf(stderr, "low_rl: %s\n", ex.what());
    rc = 1;
  }
  rclcpp::shutdown();
  return rc;
}
