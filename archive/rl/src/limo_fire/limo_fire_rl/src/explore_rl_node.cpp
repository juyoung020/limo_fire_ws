// 탐사 RL 실물 노드 (limo_fire_rl/README): /scan 한 장 = 시뮬 걸음 하나(0.1 s)로 RobotEnv(학습 시뮬과 같은 함수)에 쌓고,
// 결정 걸음(10 장 ≈ 1 s)마다 계층형 정책(CPU 정방향)이 프론티어 후보 8 곳 중 하나를 골라 Nav2 NavigateToPose 목표로 보낸다.
//   - 같은 후보를 다시 고르면 목표를 그대로 둔다 (시뮬 plan_cpu "같은 후보")
//   - 후보가 없으면 제자리 회전 (시뮬 path_follow "목표 없음" — Nav2 Spin)
//   - 지도 닫힘 + 지도 진척 시계 ≥ 10 s (시뮬 성공 판정 그대로, 1 s 마다) → 목표 취소·멈춤, ~/done = true
//   - policy:=nearest 면 정책 대신 BFS 거리 가장 짧은 후보 (같은 후보 계산 위 가장 가까운 프론티어 — 학습 비교 규칙 정책 slot0 꼴)
// 입력이 끊기면(라이다·TF·오도메트리) 목표를 취소하고 멈춘 채 알린다.
#include <cmath>
#include <cstdio>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include <geometry_msgs/msg/twist.hpp>
#include <nav2_msgs/action/navigate_to_pose.hpp>
#include <nav2_msgs/action/spin.hpp>
#include <nav_msgs/msg/occupancy_grid.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <rclcpp/rclcpp.hpp>
#include <rclcpp_action/rclcpp_action.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <sensor_msgs/msg/laser_scan.hpp>
#include <std_msgs/msg/bool.hpp>
#include <std_msgs/msg/string.hpp>
#include <std_srvs/srv/set_bool.hpp>
#include <tf2/LinearMath/Matrix3x3.h>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>
#include <visualization_msgs/msg/marker_array.hpp>

#include "limo_fire_rl/robot_env.h"
#include "policy_cpu.h"

namespace limo_fire_rl {

using NavTo = nav2_msgs::action::NavigateToPose;
using Spin = nav2_msgs::action::Spin;

class ExploreRlNode : public rclcpp::Node {
 public:
  ExploreRlNode() : Node("explore_rl") {
    env_yaml_ = declare_parameter<std::string>("env_yaml", LIMO_FIRE_RL_ENV_YAML);
    ckpt_ = declare_parameter<std::string>("checkpoint", "");
    policy_ = declare_parameter<std::string>("policy", "rl");   // rl | nearest
    dry_run_ = declare_parameter<bool>("dry_run", false);
    enabled_ = declare_parameter<bool>("enabled", true);
    map_frame_ = declare_parameter<std::string>("map_frame", "map");
    base_frame_ = declare_parameter<std::string>("base_frame", "base_link");
    tf_timeout_ = declare_parameter<double>("tf_timeout", 0.2);
    input_timeout_ = declare_parameter<double>("input_timeout", 1.0);
    depth_timeout_ = declare_parameter<double>("depth_timeout", 0.5);
    stop_on_no_progress_ = declare_parameter<bool>("stop_on_no_progress", false);
    spin_angle_ = declare_parameter<double>("spin_angle", 1.57);
    blacklist_time_ = declare_parameter<double>("blacklist_time", 10.0);
    blacklist_radius_ = declare_parameter<double>("blacklist_radius", 0.5);
    debug_dir_ = declare_parameter<std::string>("debug_dir", "");
    resend_min_dist_ = declare_parameter<double>("resend_min_dist", 0.25);
    continue_after_done_ = declare_parameter<bool>("continue_after_done", false);
    const std::string scan_topic = declare_parameter<std::string>("scan_topic", "/scan");
    const std::string odom_topic = declare_parameter<std::string>("odom_topic", "/odom");
    const std::string imu_topic = declare_parameter<std::string>("imu_topic", "/imu");
    const std::string cmd_topic = declare_parameter<std::string>("cmd_vel_topic", "/cmd_vel");
    const std::string depth_topic = declare_parameter<std::string>("depth_topic", "/camera/depth/image_raw");
    const std::string info_topic = declare_parameter<std::string>("depth_info_topic", "/camera/depth/camera_info");
    if (policy_ != "rl" && policy_ != "nearest") throw std::runtime_error("policy 는 rl · nearest");

    camera_layer_ = declare_parameter<bool>("camera_layer", true);
    hc_ = sim::load_env_cfg(env_yaml_);
    // 깊이 카메라 장애물 층: 학습과 같게 켬이 기본. 깊이 정렬·카메라 숙임이 학습 값(env.yaml camera_pitch, 정렬 켬)과 다르면 바닥을 장애물로 찍어
    // 후보가 사라지고 거짓 닫힘이 남 (README "확인" — 정렬 끈 1103 녹화). 그때만 false (후보 분포가 학습과 달라짐)
    if (!camera_layer_) hc_.cfg.cam_on = 0;
    env_ = std::make_unique<RobotEnv>(hc_);
    if (policy_ == "rl") {
      if (ckpt_.empty()) throw std::runtime_error("checkpoint 파라미터 (계층형 체크포인트 .bin 경로) 가 비었음 — policy:=nearest 면 없어도 됨");
      pol_.load(ckpt_);
      RCLCPP_INFO(get_logger(), "정책 %s (형식 %d, 반복 %d)", ckpt_.c_str(), pol_.format(), pol_.iteration());
    }
    RCLCPP_INFO(get_logger(), "env %s · 지도 %d×%d 칸 %.2f m · 결정 %d 걸음 · 깊이 카메라 층 %s · policy %s%s", env_yaml_.c_str(), env_->cfg().size,
                env_->cfg().size, env_->cfg().res, env_->cfg().decide_steps, env_->cfg().cam_on ? "켬" : "끔", policy_.c_str(), dry_run_ ? " (dry_run: 목표 안 보냄)" : "");

    tf_buf_ = std::make_shared<tf2_ros::Buffer>(get_clock());
    tf_lis_ = std::make_shared<tf2_ros::TransformListener>(*tf_buf_);
    const auto sensor = rclcpp::SensorDataQoS();
    sub_scan_ = create_subscription<sensor_msgs::msg::LaserScan>(scan_topic, sensor, [this](sensor_msgs::msg::LaserScan::ConstSharedPtr m) { on_scan(m); });
    sub_odom_ = create_subscription<nav_msgs::msg::Odometry>(odom_topic, 20, [this](nav_msgs::msg::Odometry::ConstSharedPtr m) { odom_ = m; });
    sub_imu_ = create_subscription<sensor_msgs::msg::Imu>(imu_topic, 50, [this](sensor_msgs::msg::Imu::ConstSharedPtr m) { imu_ = m; });
    sub_cmd_ = create_subscription<geometry_msgs::msg::Twist>(cmd_topic, 20, [this](geometry_msgs::msg::Twist::ConstSharedPtr m) {
      cmd_ = m;
      cmd_t_ = now();
    });
    sub_depth_ = create_subscription<sensor_msgs::msg::Image>(depth_topic, sensor, [this](sensor_msgs::msg::Image::ConstSharedPtr m) { depth_ = m; });
    sub_info_ = create_subscription<sensor_msgs::msg::CameraInfo>(info_topic, sensor, [this](sensor_msgs::msg::CameraInfo::ConstSharedPtr m) { info_ = m; });
    pub_done_ = create_publisher<std_msgs::msg::Bool>("~/done", rclcpp::QoS(1).transient_local());
    pub_status_ = create_publisher<std_msgs::msg::String>("~/status", 10);
    pub_markers_ = create_publisher<visualization_msgs::msg::MarkerArray>("~/candidates", 10);
    // RViz 확인용 (결정마다): 정책이 보는 로봇 지도 (0 빈 칸 · 100 막힘 p ≥ 0.65 · −1 모름) · 깊이 카메라 장애물 층 (100 = 장애물)
    pub_map_ = create_publisher<nav_msgs::msg::OccupancyGrid>("~/robot_map", rclcpp::QoS(1).transient_local());
    pub_cam_ = create_publisher<nav_msgs::msg::OccupancyGrid>("~/camera_layer", rclcpp::QoS(1).transient_local());
    srv_enable_ = create_service<std_srvs::srv::SetBool>("~/enable", [this](const std::shared_ptr<std_srvs::srv::SetBool::Request> q,
                                                                             std::shared_ptr<std_srvs::srv::SetBool::Response> r) {
      enabled_ = q->data;
      if (!enabled_) cancel_all("멈춤 (~/enable false)");
      r->success = true;
      r->message = enabled_ ? "탐사 켬" : "탐사 끔";
    });
    nav_ = rclcpp_action::create_client<NavTo>(this, "navigate_to_pose");
    spin_ = rclcpp_action::create_client<Spin>(this, "spin");
    watchdog_ = create_wall_timer(std::chrono::milliseconds(200), [this] { check_inputs(); });
    std_msgs::msg::Bool d;
    d.data = false;
    pub_done_->publish(d);
    if (!debug_dir_.empty()) {
      dbg_ = std::fopen((debug_dir_ + "/decisions.csv").c_str(), "w");
      obs_ = std::fopen((debug_dir_ + "/obs.bin").c_str(), "wb");
      if (dbg_) std::fprintf(dbg_, "t,step,x,y,th,slot,goal_x,goal_y,new_goal,n_valid,closed,idle_s,done,cand(x y valid logit)*8,step_ms,policy_ms\n");
    }
  }

  ~ExploreRlNode() override {
    if (dbg_) std::fclose(dbg_);
    if (obs_) std::fclose(obs_);
    save_map();
  }

 private:
  // ---- 라이다 한 장 = 걸음 하나
  void on_scan(const sensor_msgs::msg::LaserScan::ConstSharedPtr& m) {
    last_scan_t_ = now();
    if (finished_) return;
    if (!odom_ || !imu_) {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000, "오도메트리(/odom)·IMU(/imu) 기다림");
      return;
    }
    geometry_msgs::msg::TransformStamped T;
    try {
      T = tf_buf_->lookupTransform(map_frame_, base_frame_, m->header.stamp, rclcpp::Duration::from_seconds(tf_timeout_));
    } catch (const std::exception& ex) {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000, "TF %s → %s (장 시각) 없음: %s — 이 장 건너뜀", map_frame_.c_str(), base_frame_.c_str(), ex.what());
      return;
    }
    last_tf_t_ = now();
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
    Proprio p = proprio();
    const auto t0 = std::chrono::steady_clock::now();
    if (!started_) {
      env_->reset(s.x, s.y, s.th, p, -2.f * s.angle_min);
      started_ = true;
      RCLCPP_INFO(get_logger(), "시작 자세 (%.2f, %.2f, %.1f°) — 지도 창 %.1f m 가운데", s.x, s.y, s.th * 57.2958, env_->cfg().size * env_->cfg().res);
      decide(0.0);   // 판 시작 결정 (지도 빔 — 시뮬 리셋 관측과 같음: 후보 없음 → 제자리 회전)
    }
    env_->step(p, s);
    const double step_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    step_ms_sum_ += step_ms;
    ++n_steps_;
    const int dn = env_->done();
    if (dn == mdp::cur::kDeclareOk && !done_reported_) {
      finish("탐사 끝: 지도 닫힘 + 지도 진척 없음 10 s (시뮬 성공 판정)");
      if (!continue_after_done_) return;
    }
    if (dn == mdp::cur::kTruncated) {
      finish("안전 시간 1200 s 지남 — 멈춤");
      return;
    }
    if (dn == mdp::cur::kNoProgress) {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 5000, "지도 진척 없음 %.0f s (시뮬이면 판 끝 — 진행 없음)", env_->idle_s());
      if (stop_on_no_progress_) {
        finish("지도 진척 없음 60 s — stop_on_no_progress 로 멈춤");
        return;
      }
    }
    if (env_->decision_due()) decide(step_ms);
  }

  Proprio proprio() {
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
    {
      tf2::Quaternion q(imu_->orientation.x, imu_->orientation.y, imu_->orientation.z, imu_->orientation.w);
      double r, pi, y;
      tf2::Matrix3x3(q).getRPY(r, pi, y);
      p.imu[5] = (float)r;
      p.imu[6] = (float)pi;
    }
    if (cmd_ && (now() - cmd_t_).seconds() < 0.5) {
      p.cmd_v = (float)cmd_->linear.x;
      p.cmd_w = (float)cmd_->angular.z;
    }
    return p;
  }

  // ---- 결정 (1 s)
  void decide(double step_ms) {
    // 깊이: 최근 장 → 40 × 64 (없으면 모두 구멍 0)
    std::vector<uint8_t> dep((size_t)mdp::obs::DEPTH_SIZE, 0);
    if (depth_ && info_ && (now() - rclcpp::Time(depth_->header.stamp)).seconds() < depth_timeout_ && depth_->encoding == "16UC1") {
      env_->depth_from_image(reinterpret_cast<const uint16_t*>(depth_->data.data()), (int)depth_->width, (int)depth_->height, (int)(depth_->step / 2),
                             (float)info_->k[0], (float)info_->k[4], (float)info_->k[2], (float)info_->k[5], dep.data());
    } else {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 5000, "깊이 영상 없음·오래됨·16UC1 아님 — 깊이 관측 0 (구멍)");
    }
    if (depth_ && info_ && !intr_checked_) {   // 깊이 내부 파라미터가 학습 깊이 모델(정렬 켬)과 다른지 한 번 알림
      intr_checked_ = true;
      const sim::DepthSpec& D = env_->cfg().depth.spec;
      if (std::fabs(info_->k[0] - D.fx) > 0.02 * D.fx || std::fabs(info_->k[5] - D.cy) > 5.0 || info_->height < (unsigned)D.h)
        RCLCPP_WARN(get_logger(), "깊이 camera_info (fx %.1f, cy %.1f, %u×%u) 가 학습 깊이 모델(정렬 켬 fx %.1f, cy %.1f)과 다름 — 깊이 정렬(depth_registration)이 "
                    "꺼졌나? 카메라 장애물 층이 바닥을 장애물로 찍어 후보가 사라질 수 있음 (camera_layer:=false 로 끔)", info_->k[0], info_->k[5], info_->width,
                    info_->height, D.fx, D.cy);
    }
    env_->set_depth(dep.data());
    const auto t0 = std::chrono::steady_clock::now();
    float logits[mdp::hier::K], mask[mdp::hier::K];
    const float* o = env_->obs_1d();
    using namespace mdp::hier;
    int slot = 0;
    int n_valid = 0;
    for (int k = 0; k < K; ++k) {
      mask[k] = o[mdp::obs::POLICY_1D + k * FEAT + F_VALID] > 0.5f ? 1.f : 0.f;
      n_valid += mask[k] > 0.5f;
      logits[k] = 0.f;
    }
    if (policy_ == "rl") {
      slot = pol_.act(env_->obs_img(), o, logits, mask);
    } else {   // nearest: BFS 거리 가장 짧은 있는 후보
      float best = 1e9f;
      for (int k = 0; k < K; ++k)
        if (mask[k] > 0.5f && o[mdp::obs::POLICY_1D + k * FEAT + F_DIST] < best) { best = o[mdp::obs::POLICY_1D + k * FEAT + F_DIST]; slot = k; }
    }
    // Nav2 가 포기한 목표 둘레(blacklist_radius)는 blacklist_time 동안 피함 — 다음으로 좋은 있는 후보 (시뮬은 복구가 같은 일: hier_path.h path_recovery)
    if (n_valid > 0 && blacklisted(slot)) {
      int alt = -1;
      for (int k = 0; k < K; ++k)
        if (mask[k] > 0.5f && !blacklisted(k) && (alt < 0 || logits[k] > logits[alt] || (policy_ != "rl" && o[POLICY_1D_OFF(k)] < o[POLICY_1D_OFF(alt)]))) alt = k;
      if (alt >= 0) {
        RCLCPP_INFO(get_logger(), "후보 %d 는 Nav2 가 포기한 목표 둘레 — %d 로", slot, alt);
        slot = alt;
      }
    }
    const double pol_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    pol_ms_sum_ += pol_ms;
    ++n_dec_;
    const bool changed = env_->choose(slot);
    env_->camlayer_update();   // 다음 결정 묶음 앞 (시뮬 hier_camlayer_cpu 와 같은 때 — 이 결정 관측 깊이)
    if (obs_) {
      std::fwrite(env_->obs_img(), 1, (size_t)mdp::obs::IMG_SIZE, obs_);
      std::fwrite(env_->obs_1d(), sizeof(float), (size_t)mdp::hier::POLICY_1D, obs_);
    }
    float gx = NAN, gy = NAN;
    const bool has_goal = env_->cell_xy(env_->goal_cell(), &gx, &gy);
    const sim::EnvState& e = env_->state();
    if (enabled_ && !finished_ && !done_reported_) {
      if (!has_goal) {
        if (n_valid == 0) send_spin();
      } else if (!goal_active_ || (changed && std::hypot(gx - sent_x_, gy - sent_y_) >= resend_min_dist_)) {
        // 새 목표 칸이 보낸 목표에서 resend_min_dist 안이면 Nav2 목표는 그대로 (프론티어가 줄며 후보 칸이 몇 칸씩 옮겨 가는 것 — 다시 계획만 늘림)
        send_goal(gx, gy, e.x, e.y);
      }
    }
    publish_markers(logits, mask, slot, has_goal, gx, gy);
    publish_grids();
    char buf[256];
    std::snprintf(buf, sizeof(buf), "걸음 %d 후보 %d 고름 %d%s 닫힘 %d 진척없음 %.0fs 정방향 %.1fms 걸음 평균 %.1fms", e.step, n_valid, slot,
                  changed ? " (새 목표)" : "", e.map_closed, env_->idle_s(), pol_ms, n_steps_ ? step_ms_sum_ / n_steps_ : 0.0);
    std_msgs::msg::String st;
    st.data = buf;
    pub_status_->publish(st);
    RCLCPP_INFO(get_logger(), "%s", buf);
    if (dbg_) {
      std::fprintf(dbg_, "%.3f,%d,%.3f,%.3f,%.4f,%d,%.3f,%.3f,%d,%d,%d,%.1f,%d", now().seconds(), e.step, e.x, e.y, e.th, slot, gx, gy, changed ? 1 : 0, n_valid,
                   e.map_closed, env_->idle_s(), env_->done());
      for (int k = 0; k < K; ++k) {
        float cx = NAN, cy = NAN;
        env_->cand_xy(k, &cx, &cy);
        std::fprintf(dbg_, ",%.3f,%.3f,%d,%.4f", cx, cy, mask[k] > 0.5f ? 1 : 0, logits[k]);
      }
      std::fprintf(dbg_, ",%.3f,%.3f\n", step_ms, pol_ms);
      std::fflush(dbg_);
      if (n_dec_ % 10 == 0) save_map();
    }
  }
  static int POLICY_1D_OFF(int k) { return mdp::obs::POLICY_1D + k * mdp::hier::FEAT + mdp::hier::F_DIST; }

  bool blacklisted(int k) {
    float x, y;
    if (!env_->cand_xy(k, &x, &y)) return false;
    const rclcpp::Time t = now();
    for (const auto& b : blacklist_)
      if ((t - b.t).seconds() < blacklist_time_ && std::hypot(x - b.x, y - b.y) < blacklist_radius_) return true;
    return false;
  }

  // ---- Nav2
  void send_goal(float gx, float gy, float rx, float ry) {
    sent_x_ = gx;
    sent_y_ = gy;
    if (dry_run_) {
      goal_active_ = true;
      return;
    }
    if (!nav_->action_server_is_ready()) {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 3000, "Nav2 navigate_to_pose 서버 없음 — navigation.launch.xml localization:=false 를 켰나?");
      return;
    }
    NavTo::Goal g;
    g.pose.header.frame_id = map_frame_;
    g.pose.header.stamp = now();
    g.pose.pose.position.x = gx;
    g.pose.pose.position.y = gy;
    tf2::Quaternion q;
    q.setRPY(0, 0, std::atan2(gy - ry, gx - rx));   // 도착 방향 = 가는 방향 (시뮬 추종기는 방향을 안 맞춤 — 제자리 회전을 줄이려고)
    g.pose.pose.orientation.x = q.x();
    g.pose.pose.orientation.y = q.y();
    g.pose.pose.orientation.z = q.z();
    g.pose.pose.orientation.w = q.w();
    auto opt = rclcpp_action::Client<NavTo>::SendGoalOptions();
    const uint64_t id = ++goal_seq_;
    opt.goal_response_callback = [this, id](const rclcpp_action::ClientGoalHandle<NavTo>::SharedPtr& h) {
      if (!h) {
        RCLCPP_WARN(get_logger(), "Nav2 가 목표를 거절");
        if (id == goal_seq_) goal_active_ = false;
        return;
      }
      if (id == goal_seq_) goal_handle_ = h;
    };
    opt.result_callback = [this, id, gx, gy](const rclcpp_action::ClientGoalHandle<NavTo>::WrappedResult& r) {
      if (id != goal_seq_) return;   // 지난 목표 (새 목표로 바뀜)
      goal_active_ = false;
      goal_handle_.reset();
      if (r.code == rclcpp_action::ResultCode::SUCCEEDED) {
        RCLCPP_INFO(get_logger(), "목표 (%.2f, %.2f) 도착", gx, gy);
      } else if (r.code == rclcpp_action::ResultCode::ABORTED) {
        RCLCPP_WARN(get_logger(), "Nav2 가 목표 (%.2f, %.2f) 포기 — %.0f s 동안 둘레 %.1f m 후보 피함", gx, gy, blacklist_time_, blacklist_radius_);
        blacklist_.push_back({gx, gy, now()});
        env_->clear_goal();
      }
    };
    if (goal_handle_) nav_->async_cancel_goal(goal_handle_);   // Nav2 는 새 목표가 앞 목표를 바꾸지만 명시적으로
    goal_handle_.reset();
    goal_active_ = true;
    nav_->async_send_goal(g, opt);
    RCLCPP_INFO(get_logger(), "새 목표 (%.2f, %.2f)", gx, gy);
  }

  void send_spin() {
    if (dry_run_ || spinning_) return;
    if (goal_handle_) {
      nav_->async_cancel_goal(goal_handle_);
      goal_handle_.reset();
      goal_active_ = false;
    }
    if (!spin_->action_server_is_ready()) {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 3000, "후보 없음 — Nav2 spin 서버 없음, 제자리에서 기다림");
      return;
    }
    Spin::Goal g;
    g.target_yaw = (float)spin_angle_;
    g.time_allowance.sec = 10;
    auto opt = rclcpp_action::Client<Spin>::SendGoalOptions();
    opt.result_callback = [this](const rclcpp_action::ClientGoalHandle<Spin>::WrappedResult&) { spinning_ = false; };
    spinning_ = true;
    spin_->async_send_goal(g, opt);
    RCLCPP_INFO(get_logger(), "후보 없음 — 제자리 회전 %.2f rad (Nav2 spin)", spin_angle_);
  }

  void cancel_all(const char* why) {
    if (goal_handle_) nav_->async_cancel_goal(goal_handle_);
    goal_handle_.reset();
    goal_active_ = false;
    if (!dry_run_ && spin_->action_server_is_ready()) spin_->async_cancel_all_goals();
    spinning_ = false;
    env_->clear_goal();
    RCLCPP_WARN(get_logger(), "목표 취소: %s", why);
  }

  void finish(const char* why) {
    finished_ = !continue_after_done_;   // continue_after_done: 녹화 다시 돌리기에서 끝 뒤에도 결정 기록을 이어 감 (목표는 안 보냄)
    done_reported_ = true;
    cancel_all(why);
    std_msgs::msg::Bool d;
    d.data = true;
    pub_done_->publish(d);
    std_msgs::msg::String st;
    st.data = why;
    pub_status_->publish(st);
    RCLCPP_INFO(get_logger(), "%s (걸음 %d, 결정 %d, 걸음 평균 %.2f ms, 정방향 평균 %.2f ms)", why, env_->steps(), n_dec_, n_steps_ ? step_ms_sum_ / n_steps_ : 0.0,
                n_dec_ ? pol_ms_sum_ / n_dec_ : 0.0);
    save_map();
  }

  // 입력 끊김: 라이다·TF 가 input_timeout 넘게 없으면 목표 취소 (다시 오면 다음 결정에 새 목표)
  void check_inputs() {
    if (!started_ || finished_) return;
    const rclcpp::Time t = now();
    const double ds = (t - last_scan_t_).seconds(), dt = (t - last_tf_t_).seconds();
    if ((ds > input_timeout_ || dt > input_timeout_) && (goal_active_ || spinning_)) {
      char b[128];
      std::snprintf(b, sizeof(b), "입력 끊김 (라이다 %.1f s · TF %.1f s 전)", ds, dt);
      cancel_all(b);
    }
  }

  void publish_markers(const float* logits, const float* mask, int slot, bool has_goal, float gx, float gy) {
    visualization_msgs::msg::MarkerArray arr;
    for (int k = 0; k < mdp::hier::K; ++k) {
      visualization_msgs::msg::Marker m;
      m.header.frame_id = map_frame_;
      m.header.stamp = now();
      m.ns = "candidates";
      m.id = k;
      float x, y;
      if (!env_->cand_xy(k, &x, &y) || mask[k] < 0.5f) {
        m.action = visualization_msgs::msg::Marker::DELETE;
        arr.markers.push_back(m);
        continue;
      }
      m.type = visualization_msgs::msg::Marker::SPHERE;
      m.pose.position.x = x;
      m.pose.position.y = y;
      m.pose.orientation.w = 1.0;
      m.scale.x = m.scale.y = m.scale.z = k == slot ? 0.25 : 0.15;
      m.color.a = 1.0;
      m.color.r = k == slot ? 1.0 : 0.2;
      m.color.g = k == slot ? 0.3 : 0.6;
      m.color.b = k == slot ? 0.0 : 1.0;
      (void)logits;
      arr.markers.push_back(m);
    }
    visualization_msgs::msg::Marker g;
    g.header.frame_id = map_frame_;
    g.header.stamp = now();
    g.ns = "goal";
    g.id = 0;
    if (has_goal) {
      g.type = visualization_msgs::msg::Marker::CYLINDER;
      g.pose.position.x = gx;
      g.pose.position.y = gy;
      g.pose.orientation.w = 1.0;
      g.scale.x = g.scale.y = 0.3;
      g.scale.z = 0.05;
      g.color.a = 0.8;
      g.color.g = 1.0;
    } else {
      g.action = visualization_msgs::msg::Marker::DELETE;
    }
    arr.markers.push_back(g);
    pub_markers_->publish(arr);
  }

  void publish_grids() {
    const auto& c = env_->cfg();
    const sim::MapDev& m = env_->map();
    nav_msgs::msg::OccupancyGrid g;
    g.header.frame_id = map_frame_;
    g.header.stamp = now();
    g.info.resolution = c.res;
    g.info.width = (uint32_t)m.gw;
    g.info.height = (uint32_t)m.gh;
    g.info.origin.position.x = m.ox;
    g.info.origin.position.y = m.oy;
    g.info.origin.orientation.w = 1.0;
    g.data.resize((size_t)m.gw * m.gh);
    const int16_t* L = env_->lmap();
    for (int y = 0; y < m.gh; ++y)
      for (int x = 0; x < m.gw; ++x) {
        const int16_t l = L[(size_t)y * c.size + x];
        const uint8_t k = mdp::cell_class(l);
        g.data[(size_t)y * m.gw + x] = k == mdp::kUnk ? -1 : (k == mdp::kOcc ? 100 : 0);
      }
    pub_map_->publish(g);
    if (const uint8_t* V = env_->cam_layer()) {
      for (int y = 0; y < m.gh; ++y)
        for (int x = 0; x < m.gw; ++x) g.data[(size_t)y * m.gw + x] = V[(size_t)y * c.size + x] ? 100 : 0;
      pub_cam_->publish(g);
    }
  }

  // debug_dir 에 로봇 지도(로그 오즈 int16, size × size)와 창 원점
  void save_map() {
    if (debug_dir_.empty() || !env_ || !started_) return;
    const auto& c = env_->cfg();
    if (FILE* f = std::fopen((debug_dir_ + "/lmap.bin").c_str(), "wb")) {
      std::fwrite(env_->lmap(), sizeof(int16_t), (size_t)c.size * c.size, f);
      std::fclose(f);
    }
    if (env_->cam_layer())
      if (FILE* f = std::fopen((debug_dir_ + "/camlayer.bin").c_str(), "wb")) {   // 깊이 카메라 장애물 층 (칸마다 높이 칸 비트)
        std::fwrite(env_->cam_layer(), 1, (size_t)c.size * c.size, f);
        std::fclose(f);
      }
    if (FILE* f = std::fopen((debug_dir_ + "/lmap.txt").c_str(), "w")) {
      std::fprintf(f, "size %d res %.4f ox %.4f oy %.4f l_min %d l_max %d\n", c.size, c.res, env_->map().ox, env_->map().oy, c.l_min, c.l_max);
      std::fclose(f);
    }
  }

  static double yaw_of(const geometry_msgs::msg::Quaternion& q) {
    return std::atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z));
  }

  struct Black {
    float x, y;
    rclcpp::Time t;
  };
  std::string env_yaml_, ckpt_, policy_, map_frame_, base_frame_, debug_dir_;
  bool dry_run_ = false, enabled_ = true, stop_on_no_progress_ = false, camera_layer_ = true;
  double tf_timeout_, input_timeout_, depth_timeout_, spin_angle_, blacklist_time_, blacklist_radius_;
  sim::EnvHostCfg hc_;
  std::unique_ptr<RobotEnv> env_;
  xnet_cpu::HierPolicy pol_;
  std::shared_ptr<tf2_ros::Buffer> tf_buf_;
  std::shared_ptr<tf2_ros::TransformListener> tf_lis_;
  rclcpp::Subscription<sensor_msgs::msg::LaserScan>::SharedPtr sub_scan_;
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr sub_odom_;
  rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr sub_imu_;
  rclcpp::Subscription<geometry_msgs::msg::Twist>::SharedPtr sub_cmd_;
  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr sub_depth_;
  rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr sub_info_;
  rclcpp::Publisher<std_msgs::msg::Bool>::SharedPtr pub_done_;
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr pub_status_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr pub_markers_;
  rclcpp::Publisher<nav_msgs::msg::OccupancyGrid>::SharedPtr pub_map_, pub_cam_;
  rclcpp::Service<std_srvs::srv::SetBool>::SharedPtr srv_enable_;
  rclcpp_action::Client<NavTo>::SharedPtr nav_;
  rclcpp_action::Client<Spin>::SharedPtr spin_;
  rclcpp_action::ClientGoalHandle<NavTo>::SharedPtr goal_handle_;
  rclcpp::TimerBase::SharedPtr watchdog_;
  nav_msgs::msg::Odometry::ConstSharedPtr odom_;
  sensor_msgs::msg::Imu::ConstSharedPtr imu_;
  geometry_msgs::msg::Twist::ConstSharedPtr cmd_;
  sensor_msgs::msg::Image::ConstSharedPtr depth_;
  sensor_msgs::msg::CameraInfo::ConstSharedPtr info_;
  rclcpp::Time cmd_t_{0, 0, RCL_ROS_TIME}, last_scan_t_{0, 0, RCL_ROS_TIME}, last_tf_t_{0, 0, RCL_ROS_TIME};
  std::vector<Black> blacklist_;
  double resend_min_dist_ = 0.25;
  float sent_x_ = 1e9f, sent_y_ = 1e9f;
  bool continue_after_done_ = false, done_reported_ = false, intr_checked_ = false;
  bool started_ = false, finished_ = false, goal_active_ = false, spinning_ = false;
  uint64_t goal_seq_ = 0;
  int n_dec_ = 0, n_steps_ = 0;
  double step_ms_sum_ = 0, pol_ms_sum_ = 0;
  FILE* dbg_ = nullptr;
  FILE* obs_ = nullptr;
};

}  // namespace limo_fire_rl

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  try {
    auto node = std::make_shared<limo_fire_rl::ExploreRlNode>();
    rclcpp::spin(node);
  } catch (const std::exception& e) {
    std::fprintf(stderr, "explore_rl: %s\n", e.what());
    rclcpp::shutdown();
    return 1;
  }
  rclcpp::shutdown();
  return 0;
}
