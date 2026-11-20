// 탁구공 발사대 접근·좌우 조준·(선택) 한 발 — 요청 한 번에 한 바퀴.
//   표적(/fire/observations) → 표적 둘레 정지 거리 고리의 후보 검사(차체 외곽·시선, costmap) → Nav2 경로 계획 → (motion_enabled) 접근
//   → 정지 확인 → 다시 본 표적으로 차체 좌우 정렬(Nav2 두 번째 자세 또는 제자리 /cmd_vel) → 정지 → 오차 확인
//   → (auto_fire_enabled) ESP32 다리 액션 /launcher/launch SHOT 한 번.
// 발사대 각은 기구로 15° 고정이라 상하 조준은 없다 — 맞추는 것은 거리(desired_distance_m)와 좌우뿐.
// 장착 위치·방향은 TF base_link → launcher_pivot (URDF limo_fire_description launcher_v5.xacro) 에서 읽는다.
// 상태: ~/state (짧은 영문, TRANSIENT_LOCAL) · ~/status (한글 한 줄). 서비스: ~/start · ~/cancel (std_srvs/Trigger).
// 예전 Python approach_aim.py 를 옮겼다. 안전 규칙(신선도·정지 확인·표적 이동 한계·취소 시 정지 확인)은 그대로, 상하각·탄도·각도 피드백은 뺐다.
#include <signal.h>

#include <atomic>
#include <chrono>
#include <cmath>
#include <functional>
#include <map>
#include <memory>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

#include "geometry_msgs/msg/pose_stamped.hpp"
#include "geometry_msgs/msg/twist.hpp"
#include "limo_fire_hardware/aim_geometry.hpp"
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
#include "tf2/exceptions.h"
#include "tf2_ros/buffer.h"
#include "tf2_ros/transform_listener.h"

namespace limo_fire_hardware {

using Fire = limo_fire_msgs::msg::FireObservation;
using Launch = limo_fire_msgs::action::Launch;
using Nav = nav2_msgs::action::NavigateToPose;
using Plan = nav2_msgs::action::ComputePathToPose;
using Trigger = std_srvs::srv::Trigger;

struct AimError : std::runtime_error {
  using std::runtime_error::runtime_error;
};

class ApproachAim : public rclcpp::Node {
 public:
  ApproachAim() : Node("approach_aim") {
    motion_ = p<bool>("motion_enabled", false);
    mount_calibrated_ = p<bool>("mount_calibrated", false);
    map_frame_ = p<std::string>("map_frame", "map");
    base_frame_ = p<std::string>("base_frame", "base_link");
    launcher_frame_ = p<std::string>("launcher_frame", "launcher_pivot");
    max_launcher_tilt_ = rad(p<double>("max_launcher_tilt_deg", 1.0));
    launcher_pitch_deg_ = p<double>("launcher_pitch_deg", 15.0);
    distance_ = p<double>("desired_distance_m", 1.2);
    distance_tol_ = p<double>("distance_tolerance_m", 0.15);
    for (double d : p<std::vector<double>>("candidate_offsets_deg", {0.0, 15.0, -15.0, 30.0, -30.0, 60.0, -60.0}))
      offsets_.push_back(rad(d));
    radius_ = p<double>("robot_radius_m", 0.22) + p<double>("footprint_margin_m", 0.03);
    blocked_cost_ = static_cast<int>(p<int64_t>("blocked_cost", 65));
    sight_margin_ = p<double>("sight_target_margin_m", 0.15);
    target_fresh_ = p<double>("target_freshness_s", 1.0);
    shift_limit_ = p<double>("target_shift_limit_m", 0.25);
    min_conf_ = p<double>("min_confidence", 0.5);
    sensor_fresh_ = p<double>("sensor_freshness_s", 0.5);
    costmap_fresh_ = p<double>("costmap_freshness_s", 3.0);
    tf_fresh_ = p<double>("tf_freshness_s", 0.5);
    max_base_tilt_ = rad(p<double>("max_base_tilt_deg", 5.0));
    plan_timeout_ = p<double>("plan_timeout_s", 5.0);
    nav_timeout_ = p<double>("navigation_timeout_s", 60.0);
    reacquire_timeout_ = p<double>("reacquire_timeout_s", 8.0);
    stop_dwell_ = p<double>("stop_dwell_s", 0.5);
    stop_lin_ = p<double>("stop_linear_mps", 0.01);
    stop_ang_ = p<double>("stop_angular_rps", 0.02);
    yaw_tol_ = rad(p<double>("yaw_tolerance_deg", 3.0));
    yaw_mode_ = p<std::string>("yaw_alignment_mode", "nav2");
    yaw_drive_verified_ = p<bool>("yaw_rate_drive_mode_verified", false);
    yaw_gain_ = p<double>("yaw_rate_gain", 1.2);
    yaw_min_ = p<double>("yaw_rate_min_rps", 0.05);
    yaw_max_ = p<double>("yaw_rate_max_rps", 0.25);
    yaw_timeout_ = p<double>("yaw_rate_timeout_s", 8.0);
    yaw_max_err_ = rad(p<double>("yaw_rate_max_error_deg", 45.0));
    auto_fire_ = p<bool>("auto_fire_enabled", false);
    shot_timeout_ = p<double>("shot_timeout_s", 12.0);
    planner_id_ = p<std::string>("planner_id", "GridBased");
    const auto yaw_topic = p<std::string>("yaw_velocity_topic", "/cmd_vel");
    const auto status_topic = p<std::string>("limo_status_topic", "/limo_status");
    validate();

    tf_ = std::make_shared<tf2_ros::Buffer>(get_clock());
    tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_);
    nav_ = rclcpp_action::create_client<Nav>(this, p<std::string>("navigate_action", "/navigate_to_pose"));
    planner_ = rclcpp_action::create_client<Plan>(this, p<std::string>("planner_action", "/compute_path_to_pose"));
    launch_ = rclcpp_action::create_client<Launch>(this, p<std::string>("launch_action", "/launcher/launch"));
    // 깊이 10: 빨리 이어지는 상태(planning → approaching)도 받는 쪽이 놓치지 않게. 늦게 붙은 쪽은 마지막 것이 지금 상태
    const auto latched = rclcpp::QoS(10).transient_local().reliable();
    state_pub_ = create_publisher<std_msgs::msg::String>("~/state", latched);
    status_pub_ = create_publisher<std_msgs::msg::String>("~/status", latched);
    yaw_pub_ = create_publisher<geometry_msgs::msg::Twist>(yaw_topic, 10);
    fire_sub_ = create_subscription<Fire>(p<std::string>("fire_topic", "/fire/observations"), 10,
                                          [this](Fire::SharedPtr m) { on_fire(m); });
    odom_sub_ = create_subscription<nav_msgs::msg::Odometry>(
        p<std::string>("odom_topic", "/odometry/filtered"), rclcpp::SensorDataQoS(),
        [this](nav_msgs::msg::Odometry::SharedPtr m) { odom_ = m; odom_at_ = mono(); });
    base_sub_ = create_subscription<limo_msgs::msg::LimoStatus>(
        status_topic, rclcpp::SensorDataQoS(),
        [this](limo_msgs::msg::LimoStatus::SharedPtr m) { base_ = m; base_at_ = mono(); });
    grid_sub_ = create_subscription<nav_msgs::msg::OccupancyGrid>(
        p<std::string>("costmap_topic", "/global_costmap/costmap"), rclcpp::QoS(1).transient_local().reliable(),
        [this](nav_msgs::msg::OccupancyGrid::SharedPtr m) { on_grid(m); });
    launcher_sub_ = create_subscription<std_msgs::msg::String>(
        p<std::string>("launcher_state_topic", "/launcher/state"), latched,
        [this](std_msgs::msg::String::SharedPtr m) { launcher_state_ = m->data; });
    start_srv_ = create_service<Trigger>("~/start", [this](Trigger::Request::SharedPtr, Trigger::Response::SharedPtr r) {
      start(*r);
    });
    cancel_srv_ = create_service<Trigger>("~/cancel", [this](Trigger::Request::SharedPtr, Trigger::Response::SharedPtr r) {
      if (active_) begin_cancel("cancelled", "취소 요청");
      r->success = true;
      r->message = active_ ? "취소 요청 — 정지 확인 중" : "진행 중인 임무 없음";
    });
    timer_ = create_wall_timer(std::chrono::milliseconds(100), [this] { tick(); });
    set_state("idle", std::string(get_fully_qualified_name()) + "/start 를 한 번 부르면 한 바퀴 (발사대 각 " +
                          num(launcher_pitch_deg_, 1) + "° 고정)");
  }

  // 끌 때: 진행 중 Nav2·발사 목표를 취소하고 0 속도
  void close() {
    ++epoch_;
    if (yaw_owned_) publish_yaw(0.0);
    for (auto& [id, cancel] : active_goals_) cancel();
  }
  bool goals_outstanding() const { return !active_goals_.empty() || pending_ > 0; }

 private:
  static double rad(double d) { return d * M_PI / 180.0; }
  static double deg(double r) { return r * 180.0 / M_PI; }
  static double mono() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
  }
  static std::string num(double v, int prec = 2) {
    std::ostringstream s;
    s.setf(std::ios::fixed);
    s.precision(prec);
    s << v;
    return s.str();
  }
  template <class T>
  T p(const std::string& name, const T& def) {
    rcl_interfaces::msg::ParameterDescriptor d;
    d.read_only = true;   // 장치 설정은 한 번 실행 동안 고정 — 바꾸려면 다시 켠다
    return declare_parameter<T>(name, def, d);
  }

  void validate() {
    auto need = [](bool ok, const char* what) {
      if (!ok) throw std::invalid_argument(std::string("approach_aim: ") + what);
    };
    need(!launcher_frame_.empty() && launcher_frame_ != base_frame_, "launcher_frame 은 base_frame 과 다른 이름");
    need(!map_frame_.empty() && !base_frame_.empty() && map_frame_ != base_frame_, "map_frame·base_frame 확인");
    for (double v : {distance_, distance_tol_, radius_, target_fresh_, sensor_fresh_, costmap_fresh_, tf_fresh_,
                     plan_timeout_, nav_timeout_, reacquire_timeout_, stop_dwell_, shift_limit_, yaw_tol_,
                     max_base_tilt_, max_launcher_tilt_, shot_timeout_, yaw_gain_, yaw_min_, yaw_max_, yaw_timeout_,
                     yaw_max_err_})
      need(std::isfinite(v) && v > 0, "시간·거리·각 한계는 양수");
    need(distance_tol_ < distance_, "distance_tolerance_m < desired_distance_m");
    need(blocked_cost_ > 0 && blocked_cost_ <= 100 && min_conf_ >= 0 && min_conf_ <= 1, "blocked_cost·min_confidence");
    need(sight_margin_ >= 0 && sight_margin_ < distance_ && stop_lin_ >= 0 && stop_ang_ >= 0, "시선 여유·정지 기준");
    need(!offsets_.empty(), "candidate_offsets_deg");
    need(yaw_mode_ == "nav2" || yaw_mode_ == "yaw_rate", "yaw_alignment_mode 는 nav2 또는 yaw_rate");
    need(!auto_fire_ || motion_, "auto_fire_enabled 는 motion_enabled 가 필요");
    need(yaw_min_ <= yaw_max_ && yaw_max_err_ > yaw_tol_, "yaw rate 범위");
    need(std::isfinite(launcher_pitch_deg_), "launcher_pitch_deg");
  }

  // ---------------- 입력
  bool stamp_fresh(const builtin_interfaces::msg::Time& st, double max_s) {
    const double age = (now() - rclcpp::Time(st, get_clock()->get_clock_type())).seconds();
    return age >= -0.1 && age <= max_s;
  }
  static Vec3 xyz(const Fire& f) { return {f.position.x, f.position.y, f.position.z}; }

  void on_fire(const Fire::SharedPtr& m) {
    if (phase_ == "cancelling" || phase_ == "stop_unconfirmed") return;
    const bool ok = m->position_valid && m->header.frame_id == map_frame_ && std::isfinite(m->confidence) &&
                    m->confidence >= min_conf_ && std::isfinite(m->position.x) && std::isfinite(m->position.y) &&
                    std::isfinite(m->position.z) && stamp_fresh(m->header.stamp, target_fresh_);
    if (!ok) return;
    if (active_ && target_) {
      if (m->id != target_->id) return;   // 진행 중엔 다른 표적을 받지 않는다
      const Vec3 a = xyz(*m);
      if (std::hypot(a.x - initial_.x, a.y - initial_.y, a.z - initial_.z) > shift_limit_) {
        fail("표적이 허용 거리보다 많이 움직임");
        return;
      }
      target_ = m;
    }
    latest_fire_ = m;
  }
  void on_grid(const nav_msgs::msg::OccupancyGrid::SharedPtr& m) {
    const auto& o = m->info.origin;
    if (m->header.frame_id == map_frame_ && stamp_fresh(m->header.stamp, costmap_fresh_) && m->info.width > 0 &&
        m->info.height > 0 && m->data.size() == static_cast<size_t>(m->info.width) * m->info.height &&
        std::isfinite(m->info.resolution) && m->info.resolution > 0 && std::isfinite(o.position.x) &&
        std::isfinite(o.position.y) && std::isfinite(o.orientation.z) && std::isfinite(o.orientation.w)) {
      grid_ = m;
      grid_at_ = mono();
    }
  }
  bool grid_fresh() {
    return grid_ && mono() - grid_at_ <= costmap_fresh_ && stamp_fresh(grid_->header.stamp, costmap_fresh_);
  }
  GridView grid_view() const {
    const auto& o = grid_->info.origin.orientation;
    GridView g;
    g.origin_x = grid_->info.origin.position.x;
    g.origin_y = grid_->info.origin.position.y;
    g.origin_yaw = std::atan2(2 * (o.w * o.z + o.x * o.y), 1 - 2 * (o.y * o.y + o.z * o.z));
    g.resolution = grid_->info.resolution;
    g.width = grid_->info.width;
    g.height = grid_->info.height;
    g.data = grid_->data.data();
    g.size = grid_->data.size();
    return g;
  }
  bool yaw_drive_ready() {
    // 제자리 회전은 차동(0)·메카넘(2) 에서만 (limo_fire_base: Ackermann 1 은 회전 명령이 조향각으로 감)
    return base_ && mono() - base_at_ <= sensor_fresh_ && stamp_fresh(base_->header.stamp, sensor_fresh_) &&
           (base_->motion_mode == 0 || base_->motion_mode == 2);
  }
  bool stopped() {
    if (!odom_ || mono() - odom_at_ > sensor_fresh_ || !stamp_fresh(odom_->header.stamp, sensor_fresh_)) return false;
    const auto& t = odom_->twist.twist;
    return std::isfinite(t.linear.x) && std::isfinite(t.linear.y) && std::isfinite(t.angular.z) &&
           std::hypot(t.linear.x, t.linear.y) <= stop_lin_ && std::fabs(t.angular.z) <= stop_ang_;
  }
  static void rpy(const geometry_msgs::msg::Quaternion& q, double& roll, double& pitch, double& yaw) {
    roll = std::atan2(2 * (q.w * q.x + q.y * q.z), 1 - 2 * (q.x * q.x + q.y * q.y));
    pitch = std::asin(std::max(-1.0, std::min(1.0, 2 * (q.w * q.y - q.z * q.x))));
    yaw = std::atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z));
  }
  static bool unit(const geometry_msgs::msg::Quaternion& q) {
    const double n = q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w;
    return std::isfinite(n) && std::fabs(n - 1.0) <= 0.01;
  }
  Pose pose() {
    const auto t = tf_->lookupTransform(map_frame_, base_frame_, tf2::TimePointZero);
    if (!stamp_fresh(t.header.stamp, tf_fresh_)) throw AimError("로봇 위치 TF 가 오래됨");
    const auto& q = t.transform.rotation;
    if (!unit(q)) throw AimError("위치 사원수가 틀림");
    double r, pt, y;
    rpy(q, r, pt, y);
    if (std::max(std::fabs(r), std::fabs(pt)) > max_base_tilt_) throw AimError("차체 기울기가 평면 조준 한계를 넘음");
    Pose out{t.transform.translation.x, t.transform.translation.y, t.transform.translation.z, y};
    require_finite({out.x, out.y, out.z, out.yaw}, "로봇 위치");
    return out;
  }
  void read_mount() {
    geometry_msgs::msg::TransformStamped t;
    try {
      t = tf_->lookupTransform(base_frame_, launcher_frame_, tf2::TimePointZero);
    } catch (const tf2::TransformException& e) {
      throw AimError("발사대 TF " + base_frame_ + " → " + launcher_frame_ +
                     " 가 없음 — 발사대 URDF 로 robot_state_publisher 를 켤 것 (" + e.what() + ")");
    }
    const auto& q = t.transform.rotation;
    if (!unit(q)) throw AimError("발사대 장착 사원수가 틀림");
    double r, pt, y;
    rpy(q, r, pt, y);
    if (std::max(std::fabs(r), std::fabs(pt)) > max_launcher_tilt_)
      throw AimError(launcher_frame_ + " 가 " + base_frame_ + " 에서 수평이어야 함(평면 조준)");
    mount_ = {t.transform.translation.x, t.transform.translation.y, t.transform.translation.z};
    mount_yaw_ = y;
    require_finite({mount_.x, mount_.y, mount_.z, mount_yaw_}, "발사대 장착");
  }
  bool launcher_usable() const { return launcher_state_ == "ready" || launcher_state_ == "off"; }

  // ---------------- 상태 알림
  void set_state(const std::string& phase, const std::string& reason = "") {
    phase_ = phase;
    phase_at_ = mono();
    std_msgs::msg::String s;
    s.data = phase;
    state_pub_->publish(s);
    std_msgs::msg::String t;
    t.data = reason.empty() ? phase : phase + ": " + reason;
    status_pub_->publish(t);
    RCLCPP_INFO(get_logger(), "%s", t.data.c_str());
  }

  // ---------------- 시작·취소
  void start(Trigger::Response& r) {
    if (active_) {
      r.success = false;
      r.message = "이미 진행 중이거나 취소(정지 확인) 중";
      return;
    }
    try {
      if (!latest_fire_ || !stamp_fresh(latest_fire_->header.stamp, target_fresh_))
        throw AimError("지도 좌표가 있는 최근 불 관측이 필요");
      if (!grid_fresh()) throw AimError("최근 global costmap 이 필요");
      const Pose now_pose = pose();
      read_mount();
      if (!stopped()) throw AimError("시작 전 정지한 최근 오도메트리가 필요");
      if (!planner_->action_server_is_ready()) throw AimError("Nav2 ComputePathToPose 가 없음");
      if (motion_) {
        if (!mount_calibrated_) throw AimError("발사대 장착값을 확인하고 mount_calibrated 를 켠 뒤 움직임 허용");
        if (yaw_mode_ == "yaw_rate" && !yaw_drive_verified_) throw AimError("제자리 회전 전에 차동·메카넘 모드를 확인할 것");
        if (yaw_mode_ == "yaw_rate" && !yaw_drive_ready()) throw AimError("최근 /limo_status 가 차동·메카넘 모드여야 함");
        if (!nav_->action_server_is_ready()) throw AimError("Nav2 NavigateToPose 가 없음");
        if (auto_fire_) {
          if (!launch_->action_server_is_ready()) throw AimError("발사대 액션(/launcher/launch) 이 없음 — ESP32 다리 확인");
          if (!launcher_usable())
            throw AimError("발사대 상태가 '" + launcher_state_ + "' — ready 또는 off 여야 함(장전·고장이면 먼저 확인)");
        }
      }
      target_ = latest_fire_;
      initial_ = xyz(*target_);
      candidates_ = approach_candidates(initial_, now_pose, distance_, mount_, mount_yaw_, offsets_);
      next_ = 0;
      active_ = true;
      ++epoch_;
      next_candidate();
      r.success = true;
      r.message = "받음 — ~/state·~/status 를 볼 것";
    } catch (const std::exception& e) {
      r.success = false;
      r.message = e.what();
    }
  }

  void fail(const std::string& why) { begin_cancel("failed", why); }
  void begin_cancel(const std::string& terminal, const std::string& why) {
    if (phase_ == "cancelling" || phase_ == "stop_unconfirmed") return;
    ++epoch_;
    if (yaw_owned_) publish_yaw(0.0);
    for (auto& [id, cancel] : active_goals_) cancel();   // Nav2 목표·발사 동작(다리가 x) 취소
    terminal_ = terminal;
    terminal_reason_ = why;
    stop_since_.reset();
    set_state("cancelling", why);
  }
  void finish(const std::string& phase, const std::string& why) {
    active_ = false;
    yaw_owned_ = false;
    set_state(phase, why);
  }

  // ---------------- 액션 보내기 (epoch 가 바뀌면 늦게 온 응답은 버리고 취소)
  template <class A>
  void send(typename rclcpp_action::Client<A>::SharedPtr client, const typename A::Goal& goal,
            std::function<void(const typename rclcpp_action::ClientGoalHandle<A>::WrappedResult*)> done) {
    using GH = rclcpp_action::ClientGoalHandle<A>;
    const int token = epoch_;
    ++pending_;
    typename rclcpp_action::Client<A>::SendGoalOptions opt;
    opt.goal_response_callback = [this, token, done, client](typename GH::SharedPtr h) {
      --pending_;
      if (!h) {
        if (token == epoch_) done(nullptr);
        return;
      }
      const std::string id = rclcpp_action::to_string(h->get_goal_id());
      active_goals_[id] = [client, h] { client->async_cancel_goal(h); };
      if (token != epoch_) client->async_cancel_goal(h);
    };
    opt.result_callback = [this, token, done](const typename GH::WrappedResult& w) {
      active_goals_.erase(rclcpp_action::to_string(w.goal_id));
      if (token != epoch_) return;
      try {
        done(&w);
      } catch (const std::exception& e) {
        fail(std::string("액션 결과 처리 실패: ") + e.what());
      }
    };
    client->async_send_goal(goal, opt);
  }

  geometry_msgs::msg::PoseStamped goal_pose(double x, double y, double yaw) {
    geometry_msgs::msg::PoseStamped ps;
    ps.header.frame_id = map_frame_;
    ps.header.stamp = now();
    ps.pose.position.x = x;
    ps.pose.position.y = y;
    ps.pose.orientation.z = std::sin(yaw / 2);
    ps.pose.orientation.w = std::cos(yaw / 2);
    return ps;
  }
  bool clear(const Candidate& c) { return grid_fresh() && candidate_clear(grid_view(), c, xyz(*target_), radius_, blocked_cost_, sight_margin_); }

  void next_candidate() {
    if (!grid_fresh()) return fail("global costmap 이 오래됨");
    while (next_ < candidates_.size()) {
      const Candidate c = candidates_[next_++];
      if (!clear(c)) continue;
      cand_ = c;
      Plan::Goal g;
      g.goal = goal_pose(c.base_x, c.base_y, c.base_yaw);
      g.planner_id = planner_id_;
      g.use_start = false;
      set_state("planning", "후보 x=" + num(c.base_x) + " y=" + num(c.base_y));
      send<Plan>(planner_, g, [this](const rclcpp_action::ClientGoalHandle<Plan>::WrappedResult* w) { planned(w); });
      return;
    }
    fail("차체 외곽·시선 검사와 경로 계획을 통과한 후보가 없음");
  }

  void planned(const rclcpp_action::ClientGoalHandle<Plan>::WrappedResult* w) {
    if (!w || w->code != rclcpp_action::ResultCode::SUCCEEDED || w->result->path.poses.empty()) return next_candidate();
    const auto& path = w->result->path;
    const auto& end = path.poses.back().pose.position;
    if (path.header.frame_id != map_frame_ || !std::isfinite(end.x + end.y) ||
        std::hypot(end.x - cand_.base_x, end.y - cand_.base_y) > distance_tol_)
      return next_candidate();
    if (!motion_) {
      return finish("preview_ready", "후보 x=" + num(cand_.base_x) + " y=" + num(cand_.base_y) +
                                         " yaw=" + num(deg(cand_.base_yaw), 1) + "° · 회전축 " + num(mount_.x, 4) + " " +
                                         num(mount_.y, 4) + " " + num(mount_.z, 4) + " m · 발사대 각 " +
                                         num(launcher_pitch_deg_, 1) + "° 고정" +
                                         (mount_calibrated_ ? "" : " · mount_calibrated=false(임시 장착값)"));
    }
    if (!stamp_fresh(target_->header.stamp, target_fresh_)) return fail("경로 계획 중 표적 관측이 오래됨");
    if (!clear(cand_)) return fail("경로 계획 중 후보 자리 costmap 이 바뀜");
    navigate(goal_pose(cand_.base_x, cand_.base_y, cand_.base_yaw), "approaching");
  }

  void navigate(const geometry_msgs::msg::PoseStamped& ps, const std::string& phase) {
    Nav::Goal g;
    g.pose = ps;
    set_state(phase);
    send<Nav>(nav_, g, [this](const rclcpp_action::ClientGoalHandle<Nav>::WrappedResult* w) {
      if (!w || w->code != rclcpp_action::ResultCode::SUCCEEDED) return fail("Nav2 이동 실패·거절");
      after_alignment_ = phase_ == "aligning";
      stop_since_.reset();
      set_state("wait_stop");
    });
  }

  void publish_yaw(double rate) {
    geometry_msgs::msg::Twist t;
    t.angular.z = rate;
    yaw_pub_->publish(t);
  }
  bool target_ready() {
    return target_ && stamp_fresh(target_->header.stamp, target_fresh_) &&
           rclcpp::Time(target_->header.stamp, get_clock()->get_clock_type()) >= reacquire_from_;
  }
  // 지금 자리: 회전축–표적 거리가 허용 안인지 보고 좌우 오차를 낸다
  double measure(Pose& out) {
    out = pose();
    const Vec3 t = xyz(*target_);
    const double d = pivot_distance(t, out, mount_);
    if (std::fabs(d - distance_) > distance_tol_) throw AimError("회전축–표적 거리 " + num(d, 3) + " m 가 허용 밖");
    return heading_error(t, out, mount_, mount_yaw_);
  }

  void tick() {
    if (!active_) return;
    if (yaw_owned_ && phase_ != "yaw_aligning") publish_yaw(0.0);
    const double elapsed = mono() - phase_at_;
    try {
      if (phase_ == "planning" && elapsed > plan_timeout_) {
        fail("경로 계획 시간 초과");
      } else if (phase_ == "approaching" || phase_ == "aligning") {
        if (elapsed > nav_timeout_) fail("Nav2 이동 시간 초과");
      } else if (phase_ == "wait_stop" || phase_ == "cancelling" || phase_ == "stop_unconfirmed") {
        bool settled = stopped();
        if (phase_ != "wait_stop") settled = settled && !goals_outstanding();
        if (settled) {
          if (!stop_since_) stop_since_ = mono();
        } else {
          stop_since_.reset();
        }
        if (stop_since_ && mono() - *stop_since_ >= stop_dwell_) {
          if (phase_ != "wait_stop") return finish(terminal_, terminal_reason_);
          yaw_owned_ = false;
          reacquire_from_ = now();
          set_state("reacquiring");
        } else if (elapsed > reacquire_timeout_) {
          if (phase_ == "wait_stop") fail("정지를 확인하지 못함");
          else if (phase_ == "cancelling") set_state("stop_unconfirmed", "로봇 정지와 남은 Nav2·발사 목표를 직접 확인");
        }
      } else if (phase_ == "reacquiring") {
        if (target_ready()) {
          Pose ps;
          const double err = measure(ps);
          if (!after_alignment_) {
            if (yaw_mode_ == "nav2") {
              navigate(goal_pose(ps.x, ps.y, body_yaw_for_target(xyz(*target_), ps, mount_, mount_yaw_)), "aligning");
            } else {
              if (!stopped() || goals_outstanding()) throw AimError("제자리 회전 전에 Nav2 가 끝나고 정지해 있어야 함");
              if (!yaw_drive_ready()) throw AimError("제자리 회전 전에 LIMO 주행 모드가 바뀜");
              if (std::fabs(err) > yaw_max_err_) throw AimError("처음 좌우 오차가 제자리 회전 한계를 넘음");
              if (!clear(candidate_at(ps, mount_, distance_))) throw AimError("제자리 회전 전에 costmap 여유가 바뀜");
              yaw_owned_ = true;
              set_state("yaw_aligning", "좌우 오차 " + num(deg(err), 1) + "°");
            }
          } else if (std::fabs(err) > yaw_tol_) {
            fail("정렬 뒤 좌우 오차 " + num(deg(err), 1) + "° 가 허용 밖");
          } else {
            aimed(ps, err);
          }
        } else if (elapsed > reacquire_timeout_) {
          fail("이동 뒤 표적을 다시 보지 못함");
        }
      } else if (phase_ == "yaw_aligning") {
        if (elapsed > yaw_timeout_) throw AimError("제자리 정렬 시간 초과");
        if (!target_ready() || !odom_ || mono() - odom_at_ > sensor_fresh_ || !stamp_fresh(odom_->header.stamp, sensor_fresh_))
          throw AimError("제자리 정렬 중 표적·오도메트리가 끊김");
        if (!yaw_drive_ready()) throw AimError("제자리 정렬 중 LIMO 주행 모드가 바뀜");
        Pose ps;
        const double err = measure(ps);
        if (!clear(candidate_at(ps, mount_, distance_))) throw AimError("제자리 정렬 중 costmap 여유가 바뀜");
        if (std::fabs(err) > yaw_max_err_) throw AimError("좌우 오차가 정렬 한계를 넘음");
        const double rate = yaw_rate_command(err, yaw_tol_, yaw_gain_, yaw_min_, yaw_max_);
        publish_yaw(rate);
        if (rate == 0.0) {
          after_alignment_ = true;
          stop_since_.reset();
          set_state("wait_stop", "좌우 오차 " + num(deg(err), 1) + "°");
        }
      } else if (phase_ == "wait_shot") {
        if (elapsed > shot_timeout_) {
          for (auto& [id, cancel] : active_goals_) cancel();
          ++epoch_;
          finish("shot_unconfirmed", "발사대 응답 시간 초과 — 다시 하기 전에 기구 확인");
        }
      }
    } catch (const std::exception& e) {
      fail(e.what());
    }
  }

  void aimed(const Pose& ps, double err) {
    if (!stopped()) throw AimError("조준 확인 전에 로봇이 움직임");
    if (!clear(candidate_at(ps, mount_, distance_))) throw AimError("조준 자리 costmap 여유를 확인하지 못함");
    if (!auto_fire_)
      return finish("aim_ready", "좌우 오차 " + num(deg(err), 1) + "° · 발사대 각 " + num(launcher_pitch_deg_, 1) +
                                     "° 고정 · 발사는 하지 않음");
    if (goals_outstanding()) throw AimError("발사 전에 Nav2 목표가 남아 있음");
    if (!launch_->action_server_is_ready()) throw AimError("발사대 액션이 없음");
    if (!launcher_usable()) throw AimError("발사대 상태가 '" + launcher_state_ + "' — 발사하지 않음");
    Launch::Goal g;
    g.mode = Launch::Goal::SHOT;
    set_state("wait_shot", "장전+발사 한 번 요청 (좌우 오차 " + num(deg(err), 1) + "°)");
    send<Launch>(launch_, g, [this](const rclcpp_action::ClientGoalHandle<Launch>::WrappedResult* w) {
      if (!w) return finish("shot_rejected", "발사대 액션이 목표를 받지 않음");
      const std::string msg = w->result ? w->result->message : "";
      if (w->code == rclcpp_action::ResultCode::SUCCEEDED && w->result->success)
        return finish("shot_done", "펌웨어가 장전·발사를 마쳤다고 응답 — 공의 비행·명중은 재지 않음");
      if (msg.rfind("거절", 0) == 0) return finish("shot_rejected", msg);
      finish("shot_unconfirmed", msg.empty() ? "발사대 동작이 끝나지 않음 — 기구 확인" : msg);
    });
  }

  // 설정
  bool motion_, mount_calibrated_, yaw_drive_verified_, auto_fire_;
  std::string map_frame_, base_frame_, launcher_frame_, yaw_mode_, planner_id_;
  double max_launcher_tilt_, launcher_pitch_deg_, distance_, distance_tol_, radius_, sight_margin_, target_fresh_,
      shift_limit_, min_conf_, sensor_fresh_, costmap_fresh_, tf_fresh_, max_base_tilt_, plan_timeout_, nav_timeout_,
      reacquire_timeout_, stop_dwell_, stop_lin_, stop_ang_, yaw_tol_, yaw_gain_, yaw_min_, yaw_max_, yaw_timeout_,
      yaw_max_err_, shot_timeout_;
  int blocked_cost_;
  std::vector<double> offsets_;
  // ROS
  std::shared_ptr<tf2_ros::Buffer> tf_;
  std::shared_ptr<tf2_ros::TransformListener> tf_listener_;
  rclcpp_action::Client<Nav>::SharedPtr nav_;
  rclcpp_action::Client<Plan>::SharedPtr planner_;
  rclcpp_action::Client<Launch>::SharedPtr launch_;
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr state_pub_, status_pub_;
  rclcpp::Publisher<geometry_msgs::msg::Twist>::SharedPtr yaw_pub_;
  rclcpp::Subscription<Fire>::SharedPtr fire_sub_;
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_sub_;
  rclcpp::Subscription<limo_msgs::msg::LimoStatus>::SharedPtr base_sub_;
  rclcpp::Subscription<nav_msgs::msg::OccupancyGrid>::SharedPtr grid_sub_;
  rclcpp::Subscription<std_msgs::msg::String>::SharedPtr launcher_sub_;
  rclcpp::Service<Trigger>::SharedPtr start_srv_, cancel_srv_;
  rclcpp::TimerBase::SharedPtr timer_;
  // 상태
  std::string phase_ = "idle", terminal_ = "cancelled", terminal_reason_, launcher_state_ = "unknown";
  bool active_ = false, after_alignment_ = false, yaw_owned_ = false;
  int epoch_ = 0, pending_ = 0;
  std::map<std::string, std::function<void()>> active_goals_;
  double phase_at_ = 0, odom_at_ = -1e9, base_at_ = -1e9, grid_at_ = -1e9;
  std::optional<double> stop_since_;
  Fire::SharedPtr latest_fire_, target_;
  Vec3 initial_, mount_;
  double mount_yaw_ = 0;
  std::vector<Candidate> candidates_;
  size_t next_ = 0;
  Candidate cand_{};
  nav_msgs::msg::Odometry::SharedPtr odom_;
  limo_msgs::msg::LimoStatus::SharedPtr base_;
  nav_msgs::msg::OccupancyGrid::SharedPtr grid_;
  rclcpp::Time reacquire_from_{0, 0, RCL_ROS_TIME};
};

}  // namespace limo_fire_hardware

namespace {
std::atomic<bool> g_stop{false};
void on_signal(int) { g_stop = true; }
}  // namespace

int main(int argc, char** argv) {
  // 우리 신호 처리: rclcpp 기본 처리기는 Ctrl+C 에 바로 컨텍스트를 닫아, 끄기 전 Nav2·발사 취소를 보낼 수 없다
  rclcpp::init(argc, argv, rclcpp::InitOptions(), rclcpp::SignalHandlerOptions::None);
  struct sigaction sa {};
  sa.sa_handler = on_signal;
  sigaction(SIGINT, &sa, nullptr);
  sigaction(SIGTERM, &sa, nullptr);
  int code = 0;
  try {
    auto node = std::make_shared<limo_fire_hardware::ApproachAim>();
    rclcpp::executors::SingleThreadedExecutor exec;
    exec.add_node(node);
    while (rclcpp::ok() && !g_stop) exec.spin_once(std::chrono::milliseconds(100));
    node->close();
    const auto end = std::chrono::steady_clock::now() + std::chrono::seconds(5);
    while (rclcpp::ok() && node->goals_outstanding() && std::chrono::steady_clock::now() < end)
      exec.spin_once(std::chrono::milliseconds(50));
    if (node->goals_outstanding())
      RCLCPP_ERROR(node->get_logger(), "끌 때 Nav2·발사 취소를 확인하지 못함 — 로봇 정지를 직접 확인");
  } catch (const std::exception& e) {
    RCLCPP_FATAL(rclcpp::get_logger("approach_aim"), "%s", e.what());
    code = 1;
  }
  if (rclcpp::ok()) rclcpp::shutdown();
  return code;
}
