// 합친 탐사 실물 노드 (limo_fire_rl/README "합친 탐사 low_rl"): 위층 fuel 이 로봇 자기 지도에서 목표를 고르고(1 s), 아래층 순수 RL 정책이 몬다.
// /scan 한 장 = 시뮬 걸음 하나(0.1 s). 시뮬 combine_run --upper fuel --policy 와 같은 코드·같은 값:
//   관측·지도·판 끝      RobotEnv 합친 탐사 모드 (sim 헤더 그대로 — env_ref.cpp step_env + low_prepare_env, combine_run.cpp:224-231 설정)
//   위층 결정·목표 넘기기  scripts/low_explore_ctl.h (combine_run.cpp:540-555 · 629-645 · 1086-1146 · 1326-1371 을 옮긴 것)
//   아래층 정책           network/policy_cpu.h LowPolicy (체크포인트 꼬리의 관측 설정 — mdp/ckpt_cfg.h)
// 걸음 차례 (장 k): step(감각, 장 k) → 판 끝·닿음 → 위층 걸음 뒤 → 깊이 → [결정 차례면 위층 결정 → set_cmd] → low_obs → 정책 → low_act → 명령
// 명령을 고치는 규칙 층은 없다(순수 RL). 명령을 0 으로 만드는 것은 아래 멈춤 조건뿐:
//   꺼짐(~/enable false · 처음) · 리모컨이 오토가 아님(/rc_state buttons[rc_button] ≠ rc_auto_value, 또는 rc_timeout 넘게 안 옴 — 꺼짐으로 바뀜, 다시 오토가 돼도
//   ~/enable true 를 따로 불러야 감) · 입력 끊김(라이다·TF·오도메트리 input_timeout 넘게 — 0, 돌아오면 이어 감) · 닿음(시뮬 contact.h sense_step 정의 —
//   멈추고 꺼짐, contact_resume_s > 0 이면 그만큼 뒤 저절로 켬) · 끝(지도 닫힘 + 진척 없음 10 s · 진척 없음 · 시간 — 시뮬 판 끝 규칙).
// 위층 목표가 없는 걸음(판 시작·프론티어 없음)은 시뮬과 같게 규칙 추종기의 제자리 회전 (env.h apply_action 계층형 가지 — RobotEnv::low_act).
// dry_run:=true (기본): /cmd_vel 발행자를 아예 안 만들고 같은 명령을 ~/cmd_debug 에만. 위층 목표는 upper:=topic 이면 ~/goal (시험용).
// lower:=rule: 아래층을 RL 대신 시뮬 규칙 추종기로 (combine_run 을 --policy 없이 — 로봇 지도 A* plan_ref + path_step_* 추종기, RobotEnv::low_act 계층형 가지).
//   체크포인트 없이 돈다 (관측 설정은 env.yaml 그대로 — combine_run 규칙과 같음). 목표 넘기기는 low_explore_ctl.h 규칙 가지. 멈춤 조건은 그대로
// upper:=decider: 위층을 로봇 안 llama-server 판단 모델에 (scripts/low_decider.h — decider_period 마다 비동기 요청, 기다리는 동안 지금 목표로 몰고, 답이 오면 바꿈).
#include <chrono>
#include <cmath>
#include <cstdio>
#include <memory>
#include <string>
#include <vector>

#include <geometry_msgs/msg/pose_stamped.hpp>
#include <geometry_msgs/msg/twist.hpp>
#include <geometry_msgs/msg/twist_stamped.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <sensor_msgs/msg/joy.hpp>
#include <sensor_msgs/msg/laser_scan.hpp>
#include <std_msgs/msg/bool.hpp>
#include <std_msgs/msg/string.hpp>
#include <std_srvs/srv/set_bool.hpp>
#include <tf2/LinearMath/Matrix3x3.h>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>
#include <visualization_msgs/msg/marker_array.hpp>

#include "ckpt_cfg.h"   // train/explore/mdp — 체크포인트 관측 설정 꼬리
#include "limo_fire_rl/decider_client.h"
#include "limo_fire_rl/robot_env.h"
#include "low_explore_ctl.h"
#include "low_node_rec.h"
#include "policy_cpu.h"

namespace limo_fire_rl {

class LowRlNode : public rclcpp::Node {
 public:
  LowRlNode() : Node("low_rl") {
    env_yaml_ = declare_parameter<std::string>("env_yaml", LIMO_FIRE_RL_ENV_YAML);
    ckpt_ = declare_parameter<std::string>("checkpoint", "");
    dry_run_ = declare_parameter<bool>("dry_run", true);
    enabled_ = declare_parameter<bool>("enabled", false);
    upper_ = declare_parameter<std::string>("upper", "fuel");
    lower_ = declare_parameter<std::string>("lower", "rl");
    map_frame_ = declare_parameter<std::string>("map_frame", "map");
    base_frame_ = declare_parameter<std::string>("base_frame", "base_link");
    tf_timeout_ = declare_parameter<double>("tf_timeout", 0.2);
    depth_timeout_ = declare_parameter<double>("depth_timeout", 0.5);
    input_timeout_ = declare_parameter<double>("input_timeout", 0.3);
    rc_timeout_ = declare_parameter<double>("rc_timeout", 0.5);
    rc_button_ = (int)declare_parameter<int>("rc_button", 1);
    rc_auto_ = (int)declare_parameter<int>("rc_auto_value", 2);
    require_rc_ = declare_parameter<bool>("require_rc", true);
    contact_resume_s_ = declare_parameter<double>("contact_resume_s", 0.0);
    max_speed_ = declare_parameter<double>("max_speed", -1.0);
    record_ = declare_parameter<std::string>("record", "");
    const bool camera_layer = declare_parameter<bool>("camera_layer", true);
    const bool close_lidar = declare_parameter<bool>("close_lidar", false);   // 닫힘을 라이다 지도만으로 (깊이 카메라 층 뺌 — combine_run --close-lidar, 기본 끔)
    const std::string scan_topic = declare_parameter<std::string>("scan_topic", "/scan");
    const std::string odom_topic = declare_parameter<std::string>("odom_topic", "/odom");
    const std::string imu_topic = declare_parameter<std::string>("imu_topic", "/imu");
    const std::string depth_topic = declare_parameter<std::string>("depth_topic", "/camera/depth/image_raw");
    const std::string info_topic = declare_parameter<std::string>("depth_info_topic", "/camera/depth/camera_info");
    const std::string cmd_topic = declare_parameter<std::string>("cmd_vel_topic", "/cmd_vel");
    const std::string rc_topic = declare_parameter<std::string>("rc_topic", "/rc_state");
    // 닿음 정의 (학습 run.json con_cfg 와 같게 — 시뮬 env_cfg.cpp SIM_CON_* 기본값 위에 덮음. 음수 = env.yaml·환경변수 그대로)
    const int con_mask = (int)declare_parameter<int>("con_mask", -1);
    const int con_stall = (int)declare_parameter<int>("con_stall", -1);
    const double con_acc = declare_parameter<double>("con_acc", -1.0);
    // 위층 fuel (combine_run --fuel-*, scripts/README "combine_run — 위층 fuel" 고른 조합)
    fu::Opt fo;
    fo.time = declare_parameter<bool>("fuel_time", true);
    fo.global = declare_parameter<bool>("fuel_global", true);
    fo.vp = declare_parameter<bool>("fuel_vp", true);
    fo.room = declare_parameter<bool>("fuel_room", false);
    fo.hyst = (float)declare_parameter<double>("fuel_hyst", 0.2);
    fo.v_eff = (float)declare_parameter<double>("fuel_v", 0.981);
    fo.w_eff = (float)declare_parameter<double>("fuel_w", 3.232);
    fo.t0 = (float)declare_parameter<double>("fuel_t0", 0.10);
    fo.vp_min = (float)declare_parameter<double>("fuel_vp_min", 0.75);
    fo.vp_clear = (float)declare_parameter<double>("fuel_vp_clear", 0.21);
    const bool slot_vp = declare_parameter<bool>("slot_vp", false);
    // 위층 판단 모델 (upper:=decider — scripts/low_decider.h). 주기 = 로봇에서 잰 답 시간(약 2.6 s) + 여유
    const std::string dec_url = declare_parameter<std::string>("decider_url", "http://127.0.0.1:8171");
    const double dec_period = declare_parameter<double>("decider_period", 3.0);
    const double dec_timeout = declare_parameter<double>("decider_timeout", 8.0);
    const std::string dec_input = declare_parameter<std::string>("decider_input", "both");
    lxd::Params dp;
    dp.px = (int)declare_parameter<int>("decider_px", 384);
    dp.prompt = (int)declare_parameter<int>("decider_prompt", 1);
    dp.input = dec::input_of(dec_input);
    dp.match_r = (float)declare_parameter<double>("decider_match_r", 1.5);
    if (upper_ != "fuel" && upper_ != "slot0" && upper_ != "topic" && upper_ != "decider") throw std::runtime_error("upper 는 fuel · slot0 · topic · decider");
    if (upper_ == "decider" && (dp.input < 0 || dp.input == dec::kAll)) throw std::runtime_error("decider_input 은 text · image · both (all 은 시뮬 장면 그림이라 실물에 없음)");
    if (lower_ != "rl" && lower_ != "rule") throw std::runtime_error("lower 는 rl · rule");
    rule_ = lower_ == "rule";
    if (rule_ && upper_ == "topic") throw std::runtime_error("lower:=rule 은 upper fuel · slot0 · decider 만");
    if (!rule_ && ckpt_.empty()) throw std::runtime_error("checkpoint 파라미터 (아래층 체크포인트 .bin) 가 비었음");
    if (!rule_) {
      std::string log;
      const ckcfg::Kv ck = ckcfg::apply(ckpt_, &log);   // load_env_cfg 앞에 — 체크포인트 관측 설정을 환경변수로 (다르게 명시한 환경변수면 멈춤)
      RCLCPP_INFO(get_logger(), "%s", log.c_str());
      const std::string w = ckcfg::hash_warning(ck, "env_yaml_fnv", env_yaml_, "설정 파일 env.yaml");
      if (!w.empty()) RCLCPP_WARN(get_logger(), "*** %s ***", w.c_str());
    }
    hc_ = sim::load_env_cfg(env_yaml_);
    if (close_lidar) hc_.cfg.low_explore = 2;   // RobotEnv 합친 탐사 모드가 이어받음 (녹화 머리 hc 에도 남아 다시 돌리기가 같은 규칙)
    if (!camera_layer) hc_.cfg.cam_on = 0;
    if (con_mask >= 0) hc_.cfg.con_mask = con_mask;
    if (con_stall >= 0) hc_.cfg.con_stall = con_stall;
    if (con_acc >= 0.0) hc_.cfg.con_acc = (float)con_acc;
    if (dry_run_ && (hc_.cfg.con_mask & sim::kConStall)) {   // dry run: 명령이 안 나가 로봇이 안 움직이므로 "명령은 가라는데 바퀴가 섬"(멈춤 신호)은 늘 참 — 빼고 셈
      hc_.cfg.con_mask &= ~sim::kConStall;
      RCLCPP_WARN(get_logger(), "dry run: 닿음 신호에서 멈춤(명령 대비 바퀴 정지)을 뺌 — 명령이 실제로 안 나가서");
    }
    env_ = std::make_unique<RobotEnv>(hc_, true, ctl_upper() || upper_ == "decider");
    if (!rule_) {
      pol_.load(ckpt_);
      if (pol_.n1d() != env_->low_1d_n())
        throw std::runtime_error("체크포인트 1D " + std::to_string(pol_.n1d()) + " 칸 ≠ 관측 설정 " + std::to_string(env_->low_1d_n()) + " 칸");
      if (env_->low_img_n() != mdp::low::IMG_WIDE) throw std::runtime_error("넓은 지도 정책만 (SIM_LOW_WIDE=1)");
    }
    const sim::EnvCfg& c = env_->cfg();
    if (!(max_speed_ > 0.0)) max_speed_ = c.act_v_fwd;   // 기본 = 학습 한도 (자르지 않음)
    lp_ = lxc::Params::from_cfg(c);
    lp_.fuel = fo;
    lp_.rule = rule_;   // 아래층 규칙: 목표 넘기기 규칙 가지 (combine_run --policy 없음)
    lp_.use_fuel = upper_ == "fuel" || (upper_ == "slot0" && slot_vp);   // slot0: 피하지 않는 첫 후보 · stick (slot_vp 면 그 덩어리의 시점 — fu::slot_view)
    if (upper_ == "slot0" && slot_vp) lp_.fuel = fu::slot_view(fo);
    ctl_ = std::make_unique<lxc::Ctl>(lp_);
    if (upper_ == "decider") {
      dp.period_steps = std::max(1, (int)std::lround(dec_period / dt()));
      dp.timeout_steps = std::max(1, (int)std::lround(dec_timeout / dt()));
      dec_up_ = std::make_unique<lxd::Upper>(dp);
      dec_cli_ = std::make_unique<DeciderClient>(dec_url, (int)std::lround(dec_timeout * 1000.0));
      RCLCPP_INFO(get_logger(), "위층 판단 모델 %s · 입력 %s · 프롬프트 %d · 주기 %.1f s (%d 걸음) · 한도 %.1f s — 기다리는 동안 지금 목표로, 목표가 죽으면 slot0 차례로 바로",
                  dec_url.c_str(), dec_input.c_str(), dp.prompt, dp.period_steps * dt(), dp.period_steps, dp.timeout_steps * dt());
    }
    const std::string lower_desc = rule_ ? std::string("규칙 추종기 (A* + path_step — combine_run --policy 없음)") : "RL 정책 " + ckpt_;
    RCLCPP_INFO(get_logger(),
                "아래층 %s (반복 %d, GRU %d, 1D %d) · 위층 %s (fuel 시간 %d 전역 %d 시점 %d v %.3f ω %.3f t0 %.2f) · 결정 %d 걸음 · 피하기 %.2f m %.0f s · 닿음 마스크 %d "
                "멈춤 %d 걸음 가속 %.1f m/s² · 최고 속도 %.2f m/s%s — %s",
                lower_desc.c_str(), rule_ ? 0 : pol_.iteration(), rule_ ? 0 : pol_.hidden(),
                rule_ ? 0 : pol_.n1d(), upper_.c_str(), fo.time, fo.global, fo.vp, fo.v_eff, fo.w_eff, fo.t0, lp_.decide_steps,
                lp_.bl_r, lp_.bl_steps * dt(), c.con_mask, c.con_stall, c.con_acc, max_speed_, max_speed_ < c.act_v_fwd ? " (학습 한도보다 낮춤)" : "",
                dry_run_ ? "dry run: /cmd_vel 안 냄, ~/cmd_debug 에만" : "실제 주행: /cmd_vel 냄");
    if (!record_.empty()) rec_ = std::make_unique<lowrec::Writer>(record_, hc_, env_->low_img_n(), env_->low_1d_n());

    tf_buf_ = std::make_shared<tf2_ros::Buffer>(get_clock());
    tf_lis_ = std::make_shared<tf2_ros::TransformListener>(*tf_buf_);
    const auto sensor = rclcpp::SensorDataQoS();
    sub_scan_ = create_subscription<sensor_msgs::msg::LaserScan>(scan_topic, sensor, [this](sensor_msgs::msg::LaserScan::ConstSharedPtr m) { on_scan(m); });
    sub_odom_ = create_subscription<nav_msgs::msg::Odometry>(odom_topic, 20, [this](nav_msgs::msg::Odometry::ConstSharedPtr m) {
      odom_ = m;
      odom_t_ = now();
    });
    sub_imu_ = create_subscription<sensor_msgs::msg::Imu>(imu_topic, 50, [this](sensor_msgs::msg::Imu::ConstSharedPtr m) { imu_ = m; });
    sub_depth_ = create_subscription<sensor_msgs::msg::Image>(depth_topic, sensor, [this](sensor_msgs::msg::Image::ConstSharedPtr m) { depth_ = m; });
    sub_info_ = create_subscription<sensor_msgs::msg::CameraInfo>(info_topic, sensor, [this](sensor_msgs::msg::CameraInfo::ConstSharedPtr m) { info_ = m; });
    sub_rc_ = create_subscription<sensor_msgs::msg::Joy>(rc_topic, 20, [this](sensor_msgs::msg::Joy::ConstSharedPtr m) { on_rc(m); });
    if (upper_ == "topic")
      sub_goal_ = create_subscription<geometry_msgs::msg::PoseStamped>("~/goal", 10, [this](geometry_msgs::msg::PoseStamped::ConstSharedPtr m) { on_goal(m); });
    if (!dry_run_) pub_cmd_vel_ = create_publisher<geometry_msgs::msg::Twist>(cmd_topic, 10);   // dry run 이면 발행자를 아예 안 만듦
    pub_cmd_ = create_publisher<geometry_msgs::msg::TwistStamped>("~/cmd_debug", 10);
    pub_status_ = create_publisher<std_msgs::msg::String>("~/status", 10);
    pub_state_ = create_publisher<std_msgs::msg::String>("~/state", rclcpp::QoS(1).transient_local());
    pub_done_ = create_publisher<std_msgs::msg::Bool>("~/done", rclcpp::QoS(1).transient_local());
    pub_goal_ = create_publisher<visualization_msgs::msg::MarkerArray>("~/goal_marker", 10);
    srv_enable_ = create_service<std_srvs::srv::SetBool>("~/enable", [this](const std::shared_ptr<std_srvs::srv::SetBool::Request> q,
                                                                             std::shared_ptr<std_srvs::srv::SetBool::Response> r) { on_enable(q->data, r.get()); });
    watchdog_ = create_wall_timer(std::chrono::milliseconds(50), [this] { check_inputs(); });
    if (enabled_ && require_rc_) {   // 켠 채로 띄워도 리모컨 오토를 한 번 본 뒤에만
      enabled_ = false;
      RCLCPP_WARN(get_logger(), "enabled:=true 지만 리모컨 오토를 아직 못 봄 — ~/enable true 로 켤 것");
    }
    set_state(enabled_ ? "running" : "idle", enabled_ ? "켬" : "꺼짐 — ~/enable true (RViz 탐사 패널) 로 켬");
    std_msgs::msg::Bool d;
    d.data = false;
    pub_done_->publish(d);
  }

 private:
  double dt() const { return (double)env_->cfg().substeps * env_->cfg().dt; }
  bool ctl_upper() const { return upper_ == "fuel" || upper_ == "slot0"; }   // 위층을 low_explore_ctl.h Ctl 이 정함 (fuel · slot0)

  // ---- 상태 (RViz 탐사 패널: ~/state running 이면 켜짐, ~/done, ~/status)
  void set_state(const std::string& st, const std::string& why) {
    if (st != state_ || why != why_) RCLCPP_INFO(get_logger(), "상태 %s — %s", st.c_str(), why.c_str());
    state_ = st;
    why_ = why;
    std_msgs::msg::String m;
    m.data = st;
    pub_state_->publish(m);
  }
  bool running() const { return enabled_ && !finished_; }

  void send_zero(const char* why) {   // 멈춤: 0 명령 (실제 주행이면 /cmd_vel 도)
    geometry_msgs::msg::TwistStamped c;
    c.header.stamp = now();
    c.header.frame_id = base_frame_;
    pub_cmd_->publish(c);
    if (pub_cmd_vel_) pub_cmd_vel_->publish(geometry_msgs::msg::Twist());
    (void)why;
  }
  void stop(const std::string& st, const std::string& why, bool disable) {
    if (disable) enabled_ = false;
    send_zero(why.c_str());
    set_state(st, why);
  }

  void on_enable(bool on, std_srvs::srv::SetBool::Response* r) {
    if (on) {
      if (require_rc_ && !rc_auto_now()) { r->success = false; r->message = "리모컨이 오토(SWB 위)가 아니거나 /rc_state 가 안 옴 — 켜지 않음"; return; }
      if (finished_) {   // 끝난 뒤 다시 켜면 새 판: 다음 장에서 지금 자리로 판 시작
        finished_ = false;
        started_ = false;
        std_msgs::msg::Bool d;
        d.data = false;
        pub_done_->publish(d);
      }
      enabled_ = true;
      contact_t_ = -1.0;
      ctl_->force_redo();   // 다시 켜면 바로 위층 결정 (지도·피하는 칸은 그대로)
      set_state("running", "켬");
      r->success = true;
      r->message = dry_run_ ? "켬 (dry run — ~/cmd_debug 에만)" : "켬 — 주행";
    } else {
      stop("idle", "꺼짐 (~/enable false)", true);
      r->success = true;
      r->message = "멈춤";
    }
  }

  bool rc_auto_now() const {
    if (!rc_) return false;
    if ((now() - rc_t_).seconds() > rc_timeout_) return false;
    return (int)rc_->buttons.size() > rc_button_ && rc_->buttons[rc_button_] == rc_auto_;
  }
  void on_rc(const sensor_msgs::msg::Joy::ConstSharedPtr& m) {
    rc_ = m;
    rc_t_ = now();
    rc_auto_ok_ = rc_auto_now();
    if (!rc_auto_ok_ && require_rc_ && enabled_) stop("paused", "리모컨이 오토가 아님 (SWB) — 다시 오토로 바꾸고 ~/enable true", true);
  }

  // 20 Hz: 입력 끊김 · 리모컨 끊김 · 닿음 뒤 저절로 켜기
  void check_inputs() {
    const rclcpp::Time t = now();
    if (require_rc_ && enabled_ && !rc_auto_now()) stop("paused", "/rc_state 가 안 옴(" + std::to_string((int)std::lround(rc_timeout_ * 1000)) + " ms 넘게) 또는 오토 아님 — ~/enable true 로 다시", true);
    const bool stale = (last_scan_t_.nanoseconds() == 0 || (t - last_scan_t_).seconds() > input_timeout_) ||
                       (last_tf_t_.nanoseconds() == 0 || (t - last_tf_t_).seconds() > input_timeout_) ||
                       (odom_t_.nanoseconds() == 0 || (t - odom_t_).seconds() > input_timeout_);
    if (running() && stale) {
      if (!stale_) RCLCPP_WARN(get_logger(), "입력 끊김 (라이다·TF·오도메트리 %.2f s 넘게) — 명령 0", input_timeout_);
      stale_ = true;
      send_zero("입력 끊김");
    } else if (stale_ && !stale) {
      stale_ = false;
      RCLCPP_INFO(get_logger(), "입력 돌아옴");
    }
    if (!enabled_ && contact_t_ >= 0.0 && contact_resume_s_ > 0.0 && t.seconds() - contact_t_ >= contact_resume_s_ && !finished_ && (!require_rc_ || rc_auto_now())) {
      enabled_ = true;
      contact_t_ = -1.0;
      ctl_->force_redo();
      set_state("running", "닿음 뒤 저절로 켬 (contact_resume_s)");
    }
  }

  void on_goal(const geometry_msgs::msg::PoseStamped::ConstSharedPtr& m) {   // upper:=topic (시험용)
    const double x = m->pose.position.x, y = m->pose.position.y;
    topic_goal_ = (std::isfinite(x) && std::isfinite(y)) ? env_->goal_of((float)x, (float)y) : -1;
    topic_new_ = true;
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
    return p;
  }
  static double yaw_of(const geometry_msgs::msg::Quaternion& q) {
    return std::atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z));
  }

  void on_scan(const sensor_msgs::msg::LaserScan::ConstSharedPtr& m) {
    last_scan_t_ = now();
    if (finished_) return;
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
    last_tf_t_ = now();
    // 꺼짐·멈춤 동안은 판을 세움: 판 시계(진척 없음·시간·닫힘)는 켠 동안만 감. 판은 처음 켤 때 시작
    if (!running()) return;
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
    if (!started_) {   // 판 시작 (시뮬 리셋 관측 — 장 없음): 관측·결정만
      env_->reset(s.x, s.y, s.th, p, -2.f * s.angle_min);
      const sim::MapDev& md = env_->map();
      world_.gw = md.gw; world_.gh = md.gh; world_.ox = md.ox; world_.oy = md.oy;
      ctl_->restart();   // 새 판: 위층 피하는 칸·목표도 처음부터
      if (dec_up_) dec_up_->restart();
      ctl_->set_geom(md.gw, env_->cfg().res);
      if (rec_) rec_->reset(s.x, s.y, s.th, p, -2.f * s.angle_min, RobotEnv::Window{md.ox, md.oy, md.gw, md.gh});
      started_ = true;
      RCLCPP_INFO(get_logger(), "시작 자세 (%.2f, %.2f, %.1f°) — 지도 창 %.1f m", s.x, s.y, s.th * 57.2958, md.gw * env_->cfg().res);
    } else {
      if (rec_) rec_->step(p, s);
      env_->step(p, s);
      const sim::EnvState& e = env_->state();
      // 닿음 (시뮬 contact.h sense_step — e.contact = 닿음 신호 & con_mask): 멈추고 끔
      if (e.contact && running()) {
        contact_t_ = now().seconds();
        char b[200];
        std::snprintf(b, sizeof b, "닿음 (신호 %d: 라이다 1 · 깊이 2 · 멈춤 4 · 지도 8 · 사람 16 · 바퀴·자이로 32 · 가속 64, 가속 %.1f m/s²) — %s", e.s_kind, e.a_h,
                      contact_resume_s_ > 0.0 ? "멈춤, 잠시 뒤 저절로 켬" : "멈춤, ~/enable true 로 다시");
        stop("paused", b, true);
      }
      // 판 끝 (시뮬 finish_step 의 합친 탐사 규칙 — combine_run.cpp:1270)
      const int dn = env_->done();
      if (dn == mdp::cur::kDeclareOk || dn == mdp::cur::kNoProgress || dn == mdp::cur::kTruncated) {
        finished_ = true;
        stop("done", dn == mdp::cur::kDeclareOk ? "끝: 지도 닫힘 + 진척 없음 10 s (시뮬 성공 판정)" : dn == mdp::cur::kNoProgress ? "끝: 지도 진척 없음 (시뮬 실패 판정)" : "끝: 시간 한도",
             true);
        std_msgs::msg::Bool d;
        d.data = true;
        pub_done_->publish(d);
        return;
      }
      if (ctl_upper() || upper_ == "decider") ctl_->after_step(e, env_->low_rl(), env_->low_ep());   // 판단 모델도 피하는 칸 · 다시 고름은 Ctl 이 셈
    }
    const double step_ms = std::chrono::duration<double, std::milli>(clk::now() - t0).count();
    // 깊이 (센서 입력)
    std::vector<uint8_t> dep((size_t)mdp::obs::DEPTH_SIZE, 0);
    if (depth_ && info_ && (now() - rclcpp::Time(depth_->header.stamp)).seconds() < depth_timeout_ && depth_->encoding == "16UC1") {
      env_->depth_from_image(reinterpret_cast<const uint16_t*>(depth_->data.data()), (int)depth_->width, (int)depth_->height, (int)(depth_->step / 2),
                             (float)info_->k[0], (float)info_->k[4], (float)info_->k[2], (float)info_->k[5], dep.data());
    } else {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 5000, "깊이 영상 없음·오래됨·16UC1 아님 — 깊이 관측 0 (구멍)");
    }
    env_->set_depth(dep.data());
    if (rec_) rec_->depth(dep.data());
    // 위층
    const auto t1 = clk::now();
    double dec_ms = -1.0;
    if (ctl_upper()) {
      if (running() && ctl_->due(env_->state())) {
        const lxc::View v{&env_->cfg(), &world_, env_->lmap(), env_->trail(), env_->cam_layer(), &env_->state(), env_->cand(), (uint32_t)env_->cand_cur()};
        int why = 0;
        const sim::LowCmd cmd = ctl_->decide(v, work_, &why);
        env_->set_cmd(cmd);
        why_dec_ = why;
        dec_ms = std::chrono::duration<double, std::milli>(clk::now() - t1).count();
        publish_goal(cmd.target);
      } else if (!running() && env_->low_cmd().follow != 0) {
        env_->set_cmd(sim::LowCmd{-1, 0});   // 꺼짐·멈춤: 목표 없음 (다시 켜면 바로 결정)
      }
    } else if (upper_ == "decider") {
      if (running()) {   // 기다리지 않음: 요청은 일꾼 스레드가, 답은 다음 걸음들에서 꺼냄
        const lxc::View v{&env_->cfg(), &world_, env_->lmap(), env_->trail(), env_->cam_layer(), &env_->state(), env_->cand(), (uint32_t)env_->cand_cur()};
        sim::LowCmd cmd;
        int why = 0;
        if (dec_up_->step(v, env_->obs_1d() + mdp::obs::POLICY_1D, *ctl_, *dec_cli_, &cmd, &why)) {
          env_->set_cmd(cmd);
          why_dec_ = why;
          publish_goal(cmd.target);
        }
        dec_ms = std::chrono::duration<double, std::milli>(clk::now() - t1).count();
      } else if (env_->low_cmd().follow != 0) {
        env_->set_cmd(sim::LowCmd{-1, 0});
      }
    } else if (topic_new_) {
      env_->set_goal(topic_goal_);
      topic_new_ = false;
      publish_goal(topic_goal_);
    }
    const auto t2 = clk::now();
    if (rule_) env_->set_slot(ctl_->slot());   // 규칙 추종기: 경로 계획 후보 (combine_run 의 act[0] — 다음 결정까지 그대로)
    env_->low_obs();
    if (rec_) rec_->obs(env_->low_img(), env_->low_1d());
    const auto t3 = clk::now();
    float a[mdp::low::ACT] = {0.f, 0.f, 0.f}, v = 0.f, w = 0.f;
    if (running() && !rule_) pol_.act(env_->low_img(), env_->low_1d(), a);   // 꺼짐·멈춤이면 정책을 안 돌리고 행동 0 (명령 0)
    const auto t4 = clk::now();
    if (running()) env_->low_act(a, &v, &w);
    else env_->hold();   // 멈춤: 명령 0
    if (rec_) rec_->act(a, v, w);
    if (!running() || stale_) { v = 0.f; w = 0.f; }
    if (std::fabs(v) > max_speed_) v = v > 0.f ? (float)max_speed_ : -(float)max_speed_;   // max_speed (기본 = 학습 한도라 안 자름)
    geometry_msgs::msg::TwistStamped cmd;
    cmd.header.stamp = m->header.stamp;
    cmd.header.frame_id = base_frame_;
    cmd.twist.linear.x = v;
    cmd.twist.angular.z = w;
    pub_cmd_->publish(cmd);
    if (pub_cmd_vel_ && running() && !stale_) pub_cmd_vel_->publish(cmd.twist);
    const double obs_ms = std::chrono::duration<double, std::milli>(t3 - t2).count(), pol_ms = std::chrono::duration<double, std::milli>(t4 - t3).count();
    sum_step_ += step_ms;
    sum_obs_ += obs_ms;
    sum_pol_ += pol_ms;
    if (dec_ms >= 0.0) { sum_dec_ += dec_ms; ++n_dec_; max_dec_ = std::max(max_dec_, dec_ms); }
    ++n_;
    if (n_ % 10 == 0) {
      std_msgs::msg::String st;
      char buf[512];
      const sim::LowCmd& lc = env_->low_cmd();
      std::snprintf(buf, sizeof(buf), "%s (%s) · 리모컨 %s · 걸음 %d · 목표 칸 %d%s · 명령 v %.2f ω %.2f · 결정 %ld 통과 %ld 막힘 %ld · 평균 ms: 걸음 %.2f 관측 %.2f 정책 %.2f 위층 %.1f (최대 %.1f)",
                    state_.c_str(), why_.c_str(), rc_auto_now() ? "오토" : "오토 아님", env_->steps(), lc.target, why_dec_ == 2 ? " (목표 없음)" : why_dec_ == 1 ? " (slot0)" : "", v, w,
                    ctl_->n_dec(), ctl_->n_reach(), ctl_->n_stall(), sum_step_ / n_, sum_obs_ / n_, sum_pol_ / n_, n_dec_ ? sum_dec_ / n_dec_ : 0.0, max_dec_);
      st.data = buf;
      if (dec_up_) {
        char b2[300];
        std::snprintf(b2, sizeof b2, " · 판단 모델: 보냄 %ld 답 %ld (마지막 %.0f ms) 시간 끝 %ld 실패 %ld%s%s 후보 안 맞음 %ld slot0 대신 %ld%s", dec_up_->n_sent(), dec_up_->n_ans(), dec_up_->last_ms(),
                      dec_up_->n_timeout(), dec_up_->n_fail(), dec_up_->last_err().empty() ? "" : " — ", dec_up_->last_err().c_str(), dec_up_->n_nomatch(), dec_up_->n_fallback(),
                      dec_up_->waiting() ? " · 기다리는 중" : "");
        st.data += b2;
      }
      pub_status_->publish(st);
      RCLCPP_INFO_THROTTLE(get_logger(), *get_clock(), 10000, "%s", st.data.c_str());
    }
  }

  void publish_goal(int cell) {
    visualization_msgs::msg::MarkerArray ma;
    visualization_msgs::msg::Marker mk;
    mk.header.frame_id = map_frame_;
    mk.header.stamp = now();
    mk.ns = "goal";
    mk.id = 0;
    float x = 0.f, y = 0.f;
    if (cell >= 0 && env_->cell_xy(cell, &x, &y)) {
      mk.type = visualization_msgs::msg::Marker::CYLINDER;
      mk.action = visualization_msgs::msg::Marker::ADD;
      mk.pose.position.x = x;
      mk.pose.position.y = y;
      mk.pose.orientation.w = 1.0;
      mk.scale.x = mk.scale.y = 0.3;
      mk.scale.z = 0.05;
      mk.color.g = 1.0f;
      mk.color.a = 0.8f;
    } else {
      mk.action = visualization_msgs::msg::Marker::DELETE;
    }
    ma.markers.push_back(mk);
    pub_goal_->publish(ma);
  }

  bool rule_ = false;   // lower:=rule
  std::string env_yaml_, ckpt_, upper_, lower_, map_frame_, base_frame_, record_, state_, why_;
  bool dry_run_ = true, enabled_ = false, started_ = false, finished_ = false, stale_ = false, require_rc_ = true, rc_auto_ok_ = false, topic_new_ = false;
  double tf_timeout_ = 0.2, depth_timeout_ = 0.5, input_timeout_ = 0.3, rc_timeout_ = 0.5, contact_resume_s_ = 0.0, max_speed_ = -1.0, contact_t_ = -1.0;
  double sum_step_ = 0, sum_obs_ = 0, sum_pol_ = 0, sum_dec_ = 0, max_dec_ = 0;
  long n_ = 0, n_dec_ = 0;
  int rc_button_ = 1, rc_auto_ = 2, why_dec_ = 0, topic_goal_ = -1;
  sim::EnvHostCfg hc_;
  sim::World world_;
  std::unique_ptr<RobotEnv> env_;
  xnet_cpu::LowPolicy pol_;
  lxc::Params lp_;
  std::unique_ptr<lxc::Ctl> ctl_;
  std::unique_ptr<lxd::Upper> dec_up_;
  std::unique_ptr<DeciderClient> dec_cli_;
  fu::Work work_;
  std::unique_ptr<lowrec::Writer> rec_;
  std::shared_ptr<tf2_ros::Buffer> tf_buf_;
  std::shared_ptr<tf2_ros::TransformListener> tf_lis_;
  sensor_msgs::msg::Imu::ConstSharedPtr imu_;
  nav_msgs::msg::Odometry::ConstSharedPtr odom_;
  sensor_msgs::msg::Image::ConstSharedPtr depth_;
  sensor_msgs::msg::CameraInfo::ConstSharedPtr info_;
  sensor_msgs::msg::Joy::ConstSharedPtr rc_;
  rclcpp::Time last_scan_t_{0, 0, RCL_ROS_TIME}, last_tf_t_{0, 0, RCL_ROS_TIME}, odom_t_{0, 0, RCL_ROS_TIME}, rc_t_{0, 0, RCL_ROS_TIME};
  rclcpp::Subscription<sensor_msgs::msg::LaserScan>::SharedPtr sub_scan_;
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr sub_odom_;
  rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr sub_imu_;
  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr sub_depth_;
  rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr sub_info_;
  rclcpp::Subscription<sensor_msgs::msg::Joy>::SharedPtr sub_rc_;
  rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr sub_goal_;
  rclcpp::Publisher<geometry_msgs::msg::Twist>::SharedPtr pub_cmd_vel_;
  rclcpp::Publisher<geometry_msgs::msg::TwistStamped>::SharedPtr pub_cmd_;
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr pub_status_, pub_state_;
  rclcpp::Publisher<std_msgs::msg::Bool>::SharedPtr pub_done_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr pub_goal_;
  rclcpp::Service<std_srvs::srv::SetBool>::SharedPtr srv_enable_;
  rclcpp::TimerBase::SharedPtr watchdog_;
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
