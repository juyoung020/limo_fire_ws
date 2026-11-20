// 연기 모드 — 연기 속에서도 동작하는 센서(바퀴 오도메트리·IMU)만으로 저장 지도 위를 움직이게 한다.
// 설계와 근거: limo_fire_navigation/README.md "연기 모드".
//
// 이 노드가 맡는 것 (평상시·연기 모드 공통으로 늘 떠 있다)
//   map → odom TF 발행 주체. AMCL 은 tf_broadcast: false 로 두고 /amcl_pose 만 낸다.
//     평상시: AMCL 자세로 map → odom 을 계산해 낸다.
//     연기:   들어갈 때의 map → odom 을 붙잡아 계속 낸다 → 위치는 odom → base_link(EKF: 바퀴 + IMU)만 따라 움직인다.
//   라이다를 쓰는 곳으로 가는 길목.
//     scan_localization (AMCL): 평상시 /scan 그대로, 연기에선 끊는다(연기 반사로 위치가 튀지 않게).
//     scan_obstacles (costmap): 평상시 /scan 그대로, 연기에선 저장 지도에서 광선을 쏜 가상 스캔
//       → 장애물 지도에는 저장 지도의 벽만 남고, 연기가 찍은 가짜 장애물은 지워진다.
//   연기 모드 속도 한도 (Nav2 speed_limit), 연기 속 추정 자세 + 쌓인 불확실성, IMU 충격(부딪힘) 알림.
//
// 켜고 끄기: 서비스 smoke_mode/set (std_srvs/SetBool). 상태는 smoke_mode/active (std_msgs/Bool, 마지막 값 유지).
// 나올 때 지금 추정 자세를 쌓인 불확실성과 함께 /initialpose 로 AMCL 에 넘겨 다시 맞추게 한다.
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <deque>
#include <functional>
#include <vector>
#include <limits>
#include <memory>
#include <optional>
#include <string>

#include "geometry_msgs/msg/point_stamped.hpp"
#include "geometry_msgs/msg/pose_with_covariance_stamped.hpp"
#include "geometry_msgs/msg/transform_stamped.hpp"
#include "limo_fire_navigation/map_raycast.hpp"
#include "limo_fire_navigation/sonar_wall.hpp"
#include "nav2_msgs/msg/speed_limit.hpp"
#include "nav2_msgs/srv/clear_entire_costmap.hpp"
#include "nav_msgs/msg/occupancy_grid.hpp"
#include "rclcpp/create_timer.hpp"
#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/imu.hpp"
#include "sensor_msgs/msg/laser_scan.hpp"
#include "sensor_msgs/msg/range.hpp"
#include "std_msgs/msg/bool.hpp"
#include "std_msgs/msg/float64_multi_array.hpp"
#include "std_srvs/srv/set_bool.hpp"
#include "tf2/LinearMath/Quaternion.h"
#include "tf2/LinearMath/Transform.h"
#include "tf2/utils.h"
#include "tf2_geometry_msgs/tf2_geometry_msgs.hpp"
#include "tf2_ros/buffer.h"
#include "tf2_ros/transform_broadcaster.h"
#include "tf2_ros/transform_listener.h"

namespace limo_fire_navigation {

using std::placeholders::_1;
using std::placeholders::_2;

class SmokeModeNode : public rclcpp::Node {
 public:
  SmokeModeNode() : Node("smoke_mode") {
    global_frame_ = declare_parameter<std::string>("global_frame", "map");
    odom_frame_ = declare_parameter<std::string>("odom_frame", "odom");
    base_frame_ = declare_parameter<std::string>("base_frame", "base_link");
    publish_map_odom_ = declare_parameter<bool>("publish_map_odom", true);
    tf_rate_ = declare_parameter<double>("tf_rate", 20.0);
    transform_tolerance_ = declare_parameter<double>("transform_tolerance", 1.0);

    scan_frame_ = declare_parameter<std::string>("virtual_scan.frame_id", "laser_link");
    scan_rate_ = declare_parameter<double>("virtual_scan.rate", 10.0);
    scan_angle_min_ = declare_parameter<double>("virtual_scan.angle_min", -1.91986);
    scan_angle_max_ = declare_parameter<double>("virtual_scan.angle_max", 1.91986);
    scan_angle_inc_ = declare_parameter<double>("virtual_scan.angle_increment", 0.01544);
    scan_range_min_ = declare_parameter<double>("virtual_scan.range_min", 0.05);
    scan_range_max_ = declare_parameter<double>("virtual_scan.range_max", 4.0);
    occupied_threshold_ = declare_parameter<int>("virtual_scan.occupied_threshold", 65);

    speed_limit_ = declare_parameter<double>("speed_limit", 0.13);
    xy_sigma0_ = declare_parameter<double>("drift.xy_sigma0", 0.1);
    xy_per_m_ = declare_parameter<double>("drift.xy_per_m", 0.05);
    yaw_sigma0_ = declare_parameter<double>("drift.yaw_sigma0", 0.05);
    yaw_per_rad_ = declare_parameter<double>("drift.yaw_per_rad", 0.02);

    bump_accel_ = declare_parameter<double>("bump.accel", 4.0);
    bump_holdoff_ = declare_parameter<double>("bump.holdoff", 1.0);
    // 연기 속 위치: hold = 들어갈 때 map → odom 을 붙잡고 바퀴 + IMU 만 따라감 · wall = 거기에 초음파 메아리를 저장 지도 벽에 맞춰 조금씩 고침.
    // 연기 모드가 꺼져 있을 때만 바꿀 수 있다.
    // 앞 초음파가 slow.distance 보다 가까우면 연기 모드 속도 한도를 slow.speed 로 (장애물을 늦게 봐 상자를 못 피함 — 2026-11-06)
    slow_topics_ = declare_parameter<std::vector<std::string>>(
        "slow.topics", {"/ultrasonic/front_left", "/ultrasonic/front_right"});
    slow_dist_ = declare_parameter<double>("slow.distance", 0.6);
    slow_speed_ = declare_parameter<double>("slow.speed", 0.05);
    if (slow_dist_ <= 0.0 || slow_speed_ <= 0.0 || slow_topics_.empty())
      throw std::invalid_argument("smoke_mode: slow.* 파라미터 확인");
    // wall = 최근 window 초 초음파 메아리를 저장 지도 벽에 맞춰 map → odom 을 조금씩 고친다 (sonar_wall.hpp, AMCL 안 씀)
    wall_topics_ = declare_parameter<std::vector<std::string>>(
        "wall.range_topics", {"/ultrasonic/front_left", "/ultrasonic/front_right"});
    wall_window_ = declare_parameter<double>("wall.window", 4.0);
    wall_rate_ = declare_parameter<double>("wall.rate", 2.0);
    wall_min_hits_ = declare_parameter<int>("wall.min_hits", 8);
    wall_max_hits_ = declare_parameter<int>("wall.max_hits", 80);
    wall_blend_ = declare_parameter<double>("wall.blend", 0.5);
    wall_echo_ratio_ = declare_parameter<double>("wall.echo_max_ratio", 0.95);
    wall_dist_cap_ = declare_parameter<double>("wall.dist_cap", 1.0);
    wp_.rays = declare_parameter<int>("wall.rays", wp_.rays);
    wp_.max_range = declare_parameter<double>("wall.max_range", wp_.max_range);
    wp_.gate = declare_parameter<double>("wall.gate", wp_.gate);
    wp_.sigma = declare_parameter<double>("wall.sigma", wp_.sigma);
    wp_.huber = declare_parameter<double>("wall.huber", wp_.huber);
    wp_.min_normal = declare_parameter<double>("wall.min_normal", wp_.min_normal);
    // 추정 불확실성(칼만 꼴): 들어갈 때 init, 움직이면 q_* 만큼 커지고(분산, m²/m · rad²/rad), 맞출 때마다 줄어든다
    wall_init_xy_ = declare_parameter<double>("wall.init_xy", 0.05);
    wall_init_yaw_ = declare_parameter<double>("wall.init_yaw", 0.03);
    wall_q_xy_ = declare_parameter<double>("wall.q_xy", 0.0025);
    wall_q_yaw_ = declare_parameter<double>("wall.q_yaw", 0.0004);
    wp_.max_step_xy = declare_parameter<double>("wall.max_step_xy", wp_.max_step_xy);
    wp_.max_step_yaw = declare_parameter<double>("wall.max_step_yaw", wp_.max_step_yaw);
    wp_.iterations = declare_parameter<int>("wall.iterations", wp_.iterations);
    if (wall_topics_.empty() || wall_window_ <= 0.0 || wall_rate_ <= 0.0 || wall_min_hits_ < 1 ||
        wall_max_hits_ < wall_min_hits_ || wall_blend_ <= 0.0 || wall_blend_ > 1.0 || wall_echo_ratio_ <= 0.0 ||
        wall_echo_ratio_ > 1.0 || wall_dist_cap_ <= 0.0 || wp_.rays < 1 || wp_.max_range <= 0.0 || wp_.gate <= 0.0 ||
        wp_.sigma <= 0.0 || wp_.huber <= 0.0 || wall_init_xy_ <= 0.0 || wall_init_yaw_ <= 0.0 || wall_q_xy_ < 0.0 ||
        wall_q_yaw_ < 0.0 || wp_.max_step_xy <= 0.0 ||
        wp_.max_step_yaw <= 0.0 || wp_.iterations < 1)
      throw std::invalid_argument("smoke_mode: wall.* 파라미터 확인 (blend·echo_max_ratio 는 (0, 1])");
    // 지도에 없는 장애물 기억: 초음파가 저장 지도보다 min_gap 넘게 가까이 잰 곳을 두께 depth 만큼 덧그린 지도로 가상 스캔을 만든다
    // (초음파는 앞면만 봐서, 그대로면 지나가며 옆·뒤 모서리를 스친다 — 2026-11-20 상자 시험). 연기 모드 동안만 기억
    obs_enabled_ = declare_parameter<bool>("sonar_obstacles.enabled", true);
    obs_min_gap_ = declare_parameter<double>("sonar_obstacles.min_gap", 0.25);
    obs_max_range_ = declare_parameter<double>("sonar_obstacles.max_range", 1.5);
    obs_depth_ = declare_parameter<double>("sonar_obstacles.depth", 0.3);
    obs_min_hits_ = declare_parameter<int>("sonar_obstacles.min_hits", 2);
    // 빈 곳 지우기: 메아리 거리 r 이면 원뿔 안 r − carve_margin 까지는 비어 있다 → 그 칸의 찍힌 수를 줄이고, min_hits 밑이면 장애물에서 뺀다.
    // 없으면 원뿔 폭 때문에 옆으로 밀려 찍힌 가짜 장애물이 연기 모드 끝까지 남아 통로를 막았다(2026-11-20, 0.7 m/s 상자 시험 "no valid path")
    obs_carve_margin_ = declare_parameter<double>("sonar_obstacles.carve_margin", 0.1);
    // 이보다 가까운 메아리는 지도에 있는 벽이어도 찍는다: 연기 속 위치가 틀어지면(실물 바퀴 + IMU) 지도 벽이 실제와 어긋나
    // 가상 스캔만 믿고 실제 벽에 닿았다(2026-11-20 실물). 가까이 잰 것은 위치 오차와 상관없이 로봇에 대해 맞다
    obs_always_within_ = declare_parameter<double>("sonar_obstacles.always_within", 0.5);
    // 메아리 시각의 TF 가 아직 없을 때 가장 최근 TF 를 쓸 수 있는 시각 차 (실물: 다리가 받은 시각이 EKF TF 보다 앞섬)
    echo_tf_tolerance_ = declare_parameter<double>("echo_tf_tolerance", 0.3);
    if (obs_carve_margin_ < 0.0 || echo_tf_tolerance_ < 0.0)
      throw std::invalid_argument("smoke_mode: sonar_obstacles.carve_margin·echo_tf_tolerance 는 0 이상");
    if (obs_min_gap_ <= 0.0 || obs_max_range_ <= 0.0 || obs_depth_ < 0.0 || obs_min_hits_ < 1 || obs_min_hits_ > 255)
      throw std::invalid_argument("smoke_mode: sonar_obstacles.* 파라미터 확인");
    localization_ = declare_parameter<std::string>("smoke_localization", "hold");
    if (!valid_localization(localization_))
      throw std::invalid_argument("smoke_mode: smoke_localization 은 hold 또는 wall");
    param_cb_ = add_on_set_parameters_callback([this](const std::vector<rclcpp::Parameter>& ps) {
      rcl_interfaces::msg::SetParametersResult r;
      r.successful = true;
      for (const auto& p : ps) {
        if (p.get_name() != "smoke_localization") continue;
        const std::string v = p.get_type() == rclcpp::ParameterType::PARAMETER_STRING ? p.as_string() : "";
        if (smoke_) { r.successful = false; r.reason = "연기 모드 중에는 바꿀 수 없음"; }
        else if (!valid_localization(v)) { r.successful = false; r.reason = "hold 또는 wall"; }
        else localization_ = v;
      }
      return r;
    });

    if (tf_rate_ <= 0.0 || scan_rate_ <= 0.0 || scan_angle_inc_ <= 0.0 || scan_angle_max_ <= scan_angle_min_ ||
        scan_range_max_ <= scan_range_min_ || speed_limit_ <= 0.0) {
      throw std::invalid_argument("smoke_mode: 주기·스캔 각도·거리·속도 한도 파라미터가 잘못됨");
    }

    tf_buffer_ = std::make_unique<tf2_ros::Buffer>(get_clock());
    tf_listener_ = std::make_unique<tf2_ros::TransformListener>(*tf_buffer_);
    tf_broadcaster_ = std::make_unique<tf2_ros::TransformBroadcaster>(*this);

    const auto latched = rclcpp::QoS(1).transient_local().reliable();
    active_pub_ = create_publisher<std_msgs::msg::Bool>("smoke_mode/active", latched);
    pose_pub_ = create_publisher<geometry_msgs::msg::PoseWithCovarianceStamped>("smoke_mode/pose", 10);
    bump_pub_ = create_publisher<geometry_msgs::msg::PointStamped>("smoke_mode/bump", 10);
    speed_pub_ = create_publisher<nav2_msgs::msg::SpeedLimit>("speed_limit", 10);
    initialpose_pub_ = create_publisher<geometry_msgs::msg::PoseWithCovarianceStamped>("initialpose", 10);
    scan_loc_pub_ = create_publisher<sensor_msgs::msg::LaserScan>("scan_localization", rclcpp::SensorDataQoS());
    scan_obs_pub_ = create_publisher<sensor_msgs::msg::LaserScan>("scan_obstacles", rclcpp::SensorDataQoS());

    scan_sub_ = create_subscription<sensor_msgs::msg::LaserScan>(
        "scan", rclcpp::SensorDataQoS(), std::bind(&SmokeModeNode::on_scan, this, _1));
    // 앞쪽 초음파 중 하나라도 slow_distance 보다 가까우면 감속 (센서마다 가까움을 따로 들고 있다가 OR)
    close_.assign(slow_topics_.size(), false);
    for (size_t k = 0; k < slow_topics_.size(); ++k) {
      slow_subs_.push_back(create_subscription<sensor_msgs::msg::Range>(
          slow_topics_[k], rclcpp::SensorDataQoS(), [this, k](sensor_msgs::msg::Range::SharedPtr m) {
            close_[k] = std::isfinite(m->range) && m->range > m->min_range && m->range < slow_dist_;
            const bool any = std::any_of(close_.begin(), close_.end(), [](bool c) { return c; });
            if (any != front_close_) {
              front_close_ = any;
              if (smoke_) publish_speed_limit();
            }
          }));
    }
    for (const auto& t : wall_topics_) {
      range_subs_.push_back(create_subscription<sensor_msgs::msg::Range>(
          t, rclcpp::SensorDataQoS(), std::bind(&SmokeModeNode::on_range, this, _1)));
    }
    wall_pub_ = create_publisher<std_msgs::msg::Float64MultiArray>("smoke_mode/wall_fix", 10);
    imu_sub_ = create_subscription<sensor_msgs::msg::Imu>(
        "imu", rclcpp::SensorDataQoS(), std::bind(&SmokeModeNode::on_imu, this, _1));
    map_sub_ = create_subscription<nav_msgs::msg::OccupancyGrid>(
        "map", rclcpp::QoS(1).transient_local().reliable(), std::bind(&SmokeModeNode::on_map, this, _1));
    if (publish_map_odom_) {
      amcl_sub_ = create_subscription<geometry_msgs::msg::PoseWithCovarianceStamped>(
          "amcl_pose", rclcpp::QoS(10), std::bind(&SmokeModeNode::on_amcl_pose, this, _1));
      // 연기 속에서 사람이 RViz 2D Pose Estimate 로 고쳐 주면 그 자리로 옮긴다 (평상시엔 AMCL 이 받는다)
      initialpose_sub_ = create_subscription<geometry_msgs::msg::PoseWithCovarianceStamped>(
          "initialpose", rclcpp::QoS(10), std::bind(&SmokeModeNode::on_initialpose, this, _1));
    }

    clear_local_ = create_client<nav2_msgs::srv::ClearEntireCostmap>("local_costmap/clear_entirely_local_costmap");
    clear_global_ = create_client<nav2_msgs::srv::ClearEntireCostmap>("global_costmap/clear_entirely_global_costmap");

    set_srv_ = create_service<std_srvs::srv::SetBool>(
        "smoke_mode/set", std::bind(&SmokeModeNode::on_set, this, _1, _2));

    // 노드 시계(use_sim_time 이면 시뮬 시간)로 돈다 — 시뮬을 실시간보다 빠르게(Webots fast) 돌려도 주기가 시뮬 기준으로 같다
    const auto every = [this](double s, std::function<void()> f) {
      return rclcpp::create_timer(this, get_clock(), rclcpp::Duration::from_seconds(s), std::move(f));
    };
    tf_timer_ = every(1.0 / tf_rate_, [this] { on_tf_tick(); });
    scan_timer_ = every(1.0 / scan_rate_, [this] { on_scan_tick(); });
    speed_timer_ = every(1.0, [this] { publish_speed_limit(); });
    wall_timer_ = every(1.0 / wall_rate_, [this] { on_wall_tick(); });

    publish_active();
    RCLCPP_INFO(get_logger(), "연기 모드 준비 (map → odom 발행: %s)", publish_map_odom_ ? "예" : "아니오, SLAM 이 냄");
  }

 private:
  // ---- 켜고 끄기 ---------------------------------------------------------------------------------
  void on_set(const std::shared_ptr<std_srvs::srv::SetBool::Request> req,
              std::shared_ptr<std_srvs::srv::SetBool::Response> res) {
    if (req->data == smoke_) {
      res->success = true;
      res->message = smoke_ ? "이미 연기 모드" : "이미 평상시";
      return;
    }
    if (req->data) {
      std::string why;
      if (!enter(why)) {
        res->success = false;
        res->message = why;
        RCLCPP_ERROR(get_logger(), "연기 모드 못 들어감: %s", why.c_str());
        return;
      }
    } else {
      leave();
    }
    res->success = true;
    res->message = smoke_ ? "연기 모드 켬" : "평상시로 돌아옴";
  }

  bool enter(std::string& why) {
    if (!publish_map_odom_) {
      why = "SLAM(지도 만들기) 중에는 연기 모드를 쓸 수 없음 — 저장 지도 위에서만";
      return false;
    }
    if (!map_odom_) {
      why = "아직 위치를 모름 (AMCL 자세를 한 번도 못 받음)";
      return false;
    }
    if (!grid_) {
      why = "저장 지도를 아직 못 받음";
      return false;
    }
    const auto ob = odom_base();
    if (!ob) {
      why = odom_frame_ + " → " + base_frame_ + " TF 없음";
      return false;
    }
    smoke_ = true;
    last_odom_base_ = *ob;
    dist_m_ = 0.0;
    turn_rad_ = 0.0;
    echoes_.clear();
    if (grid_) {
      obs_grid_ = *grid_;
      obs_count_.assign(grid_->data.size(), 0);
    }
    obs_cells_ = 0;
    wall_tried_ = wall_applied_ = 0;
    reset_wall_cov();
    wall_last_ob_ = *ob;
    clear_costmaps();
    publish_active();
    publish_speed_limit();
    RCLCPP_WARN(get_logger(), "연기 모드 켬: 라이다·카메라를 쓰지 않고 %s, 속도 한도 %.2f m/s",
                localization_ == "wall" ? "바퀴 + IMU + 초음파(벽 맞추기)로 위치를 잡음" : "바퀴 + IMU 로만 움직임",
                speed_limit_);
    return true;
  }

  void leave() {
    // 지금 추정 자세를 쌓인 불확실성과 함께 AMCL 에 넘긴다. AMCL 은 다시 들어오는 스캔으로 이 근처에서 맞춘다.
    // 상태를 먼저 내린다: 이 노드도 /initialpose 를 받으므로(연기 속 사람 보정용) 자기 메시지를 사람 보정으로 받지 않게
    const auto est = estimate(now());
    smoke_ = false;
    if (est) initialpose_pub_->publish(*est);
    if (localization_ == "wall")
      RCLCPP_INFO(get_logger(), "연기 속 초음파 벽 맞추기: %d 번 중 %d 번 고침", wall_tried_, wall_applied_);
    if (obs_enabled_) RCLCPP_INFO(get_logger(), "연기 속 지도에 없는 장애물로 기억한 칸 %d 개 (나오며 지움)", obs_cells_);
    obs_grid_.reset();
    clear_costmaps();
    publish_active();
    publish_speed_limit();
    RCLCPP_WARN(get_logger(), "평상시로 돌아옴: 연기 속 이동 %.1f m · 회전 %.1f rad, 추정 자세를 AMCL 에 넘김", dist_m_,
                turn_rad_);
  }

  // ---- 위치 ----------------------------------------------------------------------------------------
  void on_amcl_pose(const geometry_msgs::msg::PoseWithCovarianceStamped::SharedPtr msg) {
    // 연기 속에서는 스캔을 끊어 AMCL 이 새 자세를 내지 않지만, 혹시 와도 쓰지 않는다
    if (smoke_) return;
    set_map_pose(*msg, rclcpp::Time(msg->header.stamp));
  }

  void on_initialpose(const geometry_msgs::msg::PoseWithCovarianceStamped::SharedPtr msg) {
    if (!smoke_) return;
    if (set_map_pose(*msg, rclcpp::Time(0, 0, get_clock()->get_clock_type()))) {
      dist_m_ = 0.0;
      turn_rad_ = 0.0;
      reset_wall_cov();
      RCLCPP_INFO(get_logger(), "연기 모드: 사람이 준 자세로 옮김, 쌓인 불확실성 초기화");
    }
  }

  // map 에서 본 base_link 자세 → map → odom = map_T_base · (odom_T_base)⁻¹ (같은 시각의 odom_T_base)
  bool set_map_pose(const geometry_msgs::msg::PoseWithCovarianceStamped& msg, const rclcpp::Time& stamp) {
    if (msg.header.frame_id != global_frame_) {
      RCLCPP_WARN(get_logger(), "자세 frame '%s' 가 '%s' 가 아니라 버림", msg.header.frame_id.c_str(),
                  global_frame_.c_str());
      return false;
    }
    tf2::Transform map_base;
    tf2::fromMsg(msg.pose.pose, map_base);
    geometry_msgs::msg::TransformStamped ob;
    try {
      ob = tf_buffer_->lookupTransform(odom_frame_, base_frame_, stamp, rclcpp::Duration::from_seconds(0.2));
    } catch (const tf2::TransformException& e) {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000, "자세 시각의 odom → base TF 없음: %s", e.what());
      return false;
    }
    tf2::Transform odom_base;
    tf2::fromMsg(ob.transform, odom_base);
    map_odom_ = map_base * odom_base.inverse();
    return true;
  }

  // ---- 초음파 벽 맞추기 (wall) ----------------------------------------------------------------------
  static bool valid_localization(const std::string& v) { return v == "hold" || v == "wall"; }

  // 메아리 하나를 그 시각 odom 에서 본 센서 자세와 함께 모은다. 메아리 없음(최대 거리 근처)은 버린다 — 비스듬한 벽은 메아리를 잃어 정보가 아니다
  void on_range(const sensor_msgs::msg::Range::SharedPtr m) {
    const bool want_wall = localization_ == "wall", want_obs = obs_enabled_ && obs_grid_ && map_odom_;
    if (!smoke_ || (!want_wall && !want_obs)) return;
    const double r = m->range;
    if (!std::isfinite(r) || r <= m->min_range || r >= wall_echo_ratio_ * m->max_range) return;
    const rclcpp::Time t(m->header.stamp, get_clock()->get_clock_type());
    geometry_msgs::msg::TransformStamped ts;
    try {
      ts = tf_buffer_->lookupTransform(odom_frame_, m->header.frame_id, t, rclcpp::Duration::from_seconds(0.05));
    } catch (const tf2::TransformException&) {
      // 실물: ESP32 다리가 받은 시각으로 찍어 TF(EKF)보다 조금 앞설 수 있다 → 그 시각 TF 가 아직 없으면 가장 최근 TF 를,
      // 둘의 차가 echo_tf_tolerance 안일 때만 쓴다 (2026-11-20 실물에서 메아리를 전부 버려 벽을 못 봤다)
      try {
        ts = tf_buffer_->lookupTransform(odom_frame_, m->header.frame_id, tf2::TimePointZero);
        const double lag = std::fabs((t - rclcpp::Time(ts.header.stamp, get_clock()->get_clock_type())).seconds());
        if (lag > echo_tf_tolerance_) {
          RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 5000, "초음파: odom → %s TF 가 메아리와 %.2f s 차, 버림",
                               m->header.frame_id.c_str(), lag);
          return;
        }
      } catch (const tf2::TransformException& e) {
        RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 5000, "초음파: odom → %s TF 없음, 버림: %s",
                             m->header.frame_id.c_str(), e.what());
        return;
      }
    }
    tf2::Transform os;
    tf2::fromMsg(ts.transform, os);
    if (want_obs) mark_obstacle(*map_odom_ * os, r, m->field_of_view);
    if (want_wall) echoes_.push_back({t, os, r, m->field_of_view});
  }

  // 메아리 하나가 저장 지도로 보여야 할 거리(원뿔 광선 중 가장 가까운 벽)보다 min_gap 넘게 가까우면 지도에 없는 장애물:
  // 가운데 광선의 잰 거리부터 depth 만큼 뒤까지 칸을 세고, min_hits 번 넘게 찍힌 칸을 가상 스캔용 지도에 벽으로 넣는다
  void mark_obstacle(const tf2::Transform& ms, double r, double fov) {
    const double sx = ms.getOrigin().x(), sy = ms.getOrigin().y(), syaw = tf2::getYaw(ms.getRotation());
    carve_free(sx, sy, syaw, r, fov);
    if (r > obs_max_range_) return;
    double r_exp = std::numeric_limits<double>::infinity();
    for (int k = 0; k < wp_.rays; ++k) {
      const double th = syaw + (wp_.rays == 1 ? 0.0 : -0.5 * fov + fov * k / (wp_.rays - 1));
      r_exp = std::min(r_exp, cast(*grid_, sx, sy, th, 0.0, r + obs_min_gap_ + 0.1, std::numeric_limits<double>::infinity()));
    }
    if (r_exp - r <= obs_min_gap_ && r > obs_always_within_) return;
    const double ux = std::cos(syaw), uy = std::sin(syaw), step = 0.5 * grid_->resolution;
    long last = -1;
    for (double d = 0.0; d <= obs_depth_ + 1e-9; d += step) {
      const long i = cell_index(*obs_grid_, sx + (r + d) * ux, sy + (r + d) * uy);
      if (i < 0 || i == last) continue;
      last = i;
      auto& c = obs_count_[static_cast<size_t>(i)];
      if (c < 255) ++c;
      if (c == obs_min_hits_) {
        obs_grid_->data[static_cast<size_t>(i)] = 100;
        ++obs_cells_;
      }
    }
  }

  void on_wall_tick() {
    if (!smoke_ || localization_ != "wall" || !dist_ || !map_odom_) return;
    const rclcpp::Time t = now();
    while (!echoes_.empty() && (t - echoes_.front().stamp).seconds() > wall_window_) echoes_.pop_front();
    if (static_cast<int>(echoes_.size()) < wall_min_hits_) return;
    const auto ob = odom_base();
    if (!ob) return;
    // 메아리가 max_hits 보다 많으면 창 전체에서 고르게 솎는다 (시간은 메아리 수에 비례)
    const size_t n = echoes_.size(), keep = std::min(n, static_cast<size_t>(wall_max_hits_));
    std::vector<SonarHit> hits;
    hits.reserve(keep);
    for (size_t i = 0; i < keep; ++i) {
      const auto& e = echoes_[i * n / keep];
      const tf2::Transform ms = *map_odom_ * e.odom_sensor;
      hits.push_back({ms.getOrigin().x(), ms.getOrigin().y(), tf2::getYaw(ms.getRotation()), e.r, e.fov});
    }
    const tf2::Transform mb = *map_odom_ * *ob;
    const double cx = mb.getOrigin().x(), cy = mb.getOrigin().y();
    // 지난 맞추기 뒤 움직인 만큼 불확실성을 키운다 (칼만 예측). 오차 상태 = 지금 자리 둘레 (dx, dy, dyaw) in map.
    // 방향 오차 δθ 는 그 뒤 움직인 만큼 자리 오차가 된다: δp += δθ × Δ (Δ = map 에서 본 이동) → P = F P Fᵀ + Q
    const tf2::Transform step = wall_last_ob_.inverse() * *ob;
    const tf2::Vector3 dm = tf2::Transform(map_odom_->getRotation()) * (ob->getOrigin() - wall_last_ob_.getOrigin());
    wall_last_ob_ = *ob;
    const double dist = std::hypot(step.getOrigin().x(), step.getOrigin().y());
    const std::array<double, 9> F = {1, 0, -dm.y(), 0, 1, dm.x(), 0, 0, 1};
    std::array<double, 9> FP{}, P{};
    for (int i = 0; i < 3; ++i)
      for (int j = 0; j < 3; ++j)
        for (int k = 0; k < 3; ++k) FP[3 * i + j] += F[3 * i + k] * wall_cov_[3 * k + j];
    for (int i = 0; i < 3; ++i)
      for (int j = 0; j < 3; ++j)
        for (int k = 0; k < 3; ++k) P[3 * i + j] += FP[3 * i + k] * F[3 * j + k];
    P[0] += wall_q_xy_ * dist;
    P[4] += wall_q_xy_ * dist;
    P[8] += wall_q_yaw_ * std::fabs(tf2::getYaw(step.getRotation()));
    wall_cov_ = P;
    WallParams p = wp_;
    p.prior_cov = wall_cov_;
    const double prior_xy = std::sqrt(0.5 * (P[0] + P[4])), prior_yaw = std::sqrt(P[8]);
    // 한 메아리가 창(window) 안에 있는 동안 window × rate 번 다시 쓰이므로 그만큼 덜 믿는다
    p.sigma = wp_.sigma * std::sqrt(std::max(1.0, wall_window_ * wall_rate_));
    const auto t0 = std::chrono::steady_clock::now();
    const WallResult m = match_walls(*dist_, *grid_, hits, p, cx, cy);
    const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    if (ms > 500.0 / wall_rate_)   // 주기의 절반을 넘으면 TF·가상 스캔이 밀린다
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 10000, "초음파 벽 맞추기가 느림: 메아리 %zu 개 %.0f ms", hits.size(), ms);
    ++wall_tried_;
    // 기록용: [쓴 메아리, 모은 메아리, 고치기 전 rms, 후 rms, dx, dy, dyaw, 고쳤나(1/0), 정보 작은 고윳값, 사전 σxy, 사전 σyaw]
    std_msgs::msg::Float64MultiArray rec;
    const bool apply = m.used >= wall_min_hits_ && m.rms <= m.rms0;
    rec.data = {static_cast<double>(m.used), static_cast<double>(hits.size()), m.rms0, m.rms, m.dx, m.dy, m.dyaw,
                apply ? 1.0 : 0.0, m.info_min, prior_xy, prior_yaw};
    wall_pub_->publish(rec);
    if (!apply) return;
    wall_cov_ = m.cov;
    const double k = wall_blend_;
    // map 에서 (cx, cy) 둘레로 k·dyaw 돌리고 k·(dx, dy) 옮김 → map → odom 앞에 곱한다
    tf2::Quaternion q;
    q.setRPY(0, 0, k * m.dyaw);
    const tf2::Transform rot(q, tf2::Vector3(0, 0, 0));
    const tf2::Transform to_c(tf2::Quaternion::getIdentity(), tf2::Vector3(cx, cy, 0));
    const tf2::Transform shift(tf2::Quaternion::getIdentity(), tf2::Vector3(k * m.dx, k * m.dy, 0));
    map_odom_ = shift * to_c * rot * to_c.inverse() * *map_odom_;
    ++wall_applied_;
  }

  void reset_wall_cov() {
    wall_cov_ = {wall_init_xy_ * wall_init_xy_, 0, 0, 0, wall_init_xy_ * wall_init_xy_, 0, 0, 0, wall_init_yaw_ * wall_init_yaw_};
  }

  // 원뿔 안 광선마다 센서에서 r − carve_margin 까지의 칸: 찍힌 수 −1, min_hits 밑으로 떨어지면 원래 지도 값으로
  void carve_free(double sx, double sy, double syaw, double r, double fov) {
    const double reach = r - obs_carve_margin_;
    if (reach <= 0.0) return;
    const int n = std::max(3, static_cast<int>(std::ceil(fov / 0.035)) + 1);   // 약 2° 간격
    const double step = 0.5 * grid_->resolution;
    std::vector<long> seen;
    for (int k = 0; k < n; ++k) {
      const double th = syaw - 0.5 * fov + fov * k / (n - 1), ux = std::cos(th), uy = std::sin(th);
      for (double d = step; d <= reach; d += step) {
        const long i = cell_index(*obs_grid_, sx + d * ux, sy + d * uy);
        if (i < 0 || std::find(seen.begin(), seen.end(), i) != seen.end()) continue;
        seen.push_back(i);
        auto& c = obs_count_[static_cast<size_t>(i)];
        if (c == 0) continue;
        if (c == obs_min_hits_) {
          obs_grid_->data[static_cast<size_t>(i)] = grid_->data[static_cast<size_t>(i)];
          --obs_cells_;
        }
        --c;
      }
    }
  }

  std::optional<tf2::Transform> odom_base() {
    try {
      const auto t = tf_buffer_->lookupTransform(odom_frame_, base_frame_, tf2::TimePointZero);
      tf2::Transform out;
      tf2::fromMsg(t.transform, out);
      return out;
    } catch (const tf2::TransformException&) {
      return std::nullopt;
    }
  }

  void on_tf_tick() {
    if (!publish_map_odom_ || !map_odom_) return;
    geometry_msgs::msg::TransformStamped t;
    // AMCL 과 같이 앞날짜로 낸다: 받는 쪽이 조금 늦은 시각을 물어도 찾게
    t.header.stamp = now() + rclcpp::Duration::from_seconds(transform_tolerance_);
    t.header.frame_id = global_frame_;
    t.child_frame_id = odom_frame_;
    t.transform = tf2::toMsg(*map_odom_);
    tf_broadcaster_->sendTransform(t);

    if (!smoke_) return;
    // 연기 속 이동량을 쌓아 불확실성을 키운다
    if (const auto ob = odom_base()) {
      const tf2::Transform d = last_odom_base_.inverse() * *ob;
      dist_m_ += std::hypot(d.getOrigin().x(), d.getOrigin().y());
      turn_rad_ += std::fabs(tf2::getYaw(d.getRotation()));
      last_odom_base_ = *ob;
    }
    if (const auto est = estimate(now())) pose_pub_->publish(*est);
  }

  // 연기 속 추정 자세: 붙잡은 map → odom · 지금 odom → base. 공분산은 이동·회전에 비례해 커진다 (잠정 계수, config)
  std::optional<geometry_msgs::msg::PoseWithCovarianceStamped> estimate(const rclcpp::Time& stamp) {
    if (!map_odom_) return std::nullopt;
    const auto ob = odom_base();
    if (!ob) return std::nullopt;
    const tf2::Transform map_base = *map_odom_ * *ob;
    geometry_msgs::msg::PoseWithCovarianceStamped p;
    p.header.stamp = stamp;
    p.header.frame_id = global_frame_;
    tf2::toMsg(map_base, p.pose.pose);
    const double sxy = xy_sigma0_ + xy_per_m_ * dist_m_;
    const double syaw = yaw_sigma0_ + yaw_per_rad_ * turn_rad_;
    p.pose.covariance[0] = sxy * sxy;     // x
    p.pose.covariance[7] = sxy * sxy;     // y
    p.pose.covariance[35] = syaw * syaw;  // yaw
    return p;
  }

  // ---- 라이다 길목 ---------------------------------------------------------------------------------
  void on_scan(const sensor_msgs::msg::LaserScan::SharedPtr msg) {
    if (smoke_) return;  // 연기 속 라이다는 쓰지 않는다. 장애물 지도에는 on_scan_tick 의 가상 스캔이 간다
    scan_loc_pub_->publish(*msg);
    scan_obs_pub_->publish(*msg);
  }

  void on_map(const nav_msgs::msg::OccupancyGrid::SharedPtr msg) {
    Grid g;
    g.width = static_cast<int>(msg->info.width);
    g.height = static_cast<int>(msg->info.height);
    g.resolution = msg->info.resolution;
    g.origin_x = msg->info.origin.position.x;
    g.origin_y = msg->info.origin.position.y;
    g.origin_yaw = tf2::getYaw(msg->info.origin.orientation);
    g.data = msg->data;
    g.occupied_threshold = occupied_threshold_;
    grid_ = std::move(g);
    dist_ = make_dist_field(*grid_, wall_dist_cap_);
    RCLCPP_INFO(get_logger(), "저장 지도 받음 %d × %d, %.2f m/칸", grid_->width, grid_->height, grid_->resolution);
  }

  // 연기 모드: 저장 지도에서 광선을 쏜 가상 스캔을 장애물 지도로. 실제 라이다가 와도 안 와도 같은 주기로 낸다
  void on_scan_tick() {
    if (!smoke_ || !grid_ || !map_odom_) return;
    tf2::Transform odom_laser;
    try {
      const auto t = tf_buffer_->lookupTransform(odom_frame_, scan_frame_, tf2::TimePointZero);
      tf2::fromMsg(t.transform, odom_laser);
    } catch (const tf2::TransformException& e) {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000, "가상 스캔: odom → %s TF 없음: %s", scan_frame_.c_str(),
                           e.what());
      return;
    }
    const tf2::Transform map_laser = *map_odom_ * odom_laser;
    const double lx = map_laser.getOrigin().x(), ly = map_laser.getOrigin().y();
    const double lyaw = tf2::getYaw(map_laser.getRotation());

    sensor_msgs::msg::LaserScan s;
    s.header.stamp = now();
    s.header.frame_id = scan_frame_;
    s.angle_min = scan_angle_min_;
    s.angle_increment = scan_angle_inc_;
    const int n = static_cast<int>(std::floor((scan_angle_max_ - scan_angle_min_) / scan_angle_inc_)) + 1;
    s.angle_max = scan_angle_min_ + (n - 1) * scan_angle_inc_;
    s.scan_time = 1.0 / scan_rate_;
    s.range_min = scan_range_min_;
    s.range_max = scan_range_max_;
    // 안 닿은 빔은 range_max 바로 안쪽: costmap 이 그 거리까지 비우고(raytrace), 장애물로는 안 찍는다(obstacle_max_range 밖)
    const double no_hit = scan_range_max_ - 1e-3;
    s.ranges.resize(n);
    const Grid& g = obs_grid_ ? *obs_grid_ : *grid_;   // 지도에 없는 장애물을 덧그린 지도
    for (int i = 0; i < n; ++i) {
      s.ranges[i] = static_cast<float>(
          cast(g, lx, ly, lyaw + s.angle_min + i * s.angle_increment, scan_range_min_, no_hit, no_hit));
    }
    scan_obs_pub_->publish(s);
  }

  void clear_costmaps() {
    for (auto* c : {&clear_local_, &clear_global_}) {
      if (!(*c)->service_is_ready()) {
        RCLCPP_WARN(get_logger(), "%s 서비스가 없어 장애물 지도를 못 비움", (*c)->get_service_name());
        continue;
      }
      (*c)->async_send_request(std::make_shared<nav2_msgs::srv::ClearEntireCostmap::Request>());
    }
  }

  // ---- 속도 한도·상태 -----------------------------------------------------------------------------
  // Nav2 controller_server 가 받는다. 늦게 뜬 컨트롤러도 받게 1 초마다 다시 낸다. 0 = 한도 없음(Nav2 규약)
  void publish_speed_limit() {
    nav2_msgs::msg::SpeedLimit m;
    m.header.stamp = now();
    m.header.frame_id = global_frame_;
    m.percentage = false;
    m.speed_limit = !smoke_ ? 0.0 : (front_close_ ? std::min(speed_limit_, slow_speed_) : speed_limit_);
    speed_pub_->publish(m);
  }

  void publish_active() {
    std_msgs::msg::Bool m;
    m.data = smoke_;
    active_pub_->publish(m);
  }

  // ---- IMU 충격 -------------------------------------------------------------------------------------
  // 연기 속에서는 앞을 못 보므로 부딪힘을 IMU 수평 가속도로 안다. 위치를 알려 막힌 곳 보고·BT 복구에 쓰게 한다
  void on_imu(const sensor_msgs::msg::Imu::SharedPtr msg) {
    if (!smoke_) return;
    const double a = std::hypot(msg->linear_acceleration.x, msg->linear_acceleration.y);
    if (a < bump_accel_) return;
    const rclcpp::Time t = now();
    if (last_bump_ && (t - *last_bump_).seconds() < bump_holdoff_) return;
    last_bump_ = t;
    geometry_msgs::msg::PointStamped p;
    p.header.stamp = t;
    p.header.frame_id = global_frame_;
    if (const auto est = estimate(t)) p.point = est->pose.pose.position;
    bump_pub_->publish(p);
    RCLCPP_WARN(get_logger(), "연기 모드: 충격 %.1f m/s² — 부딪힘 의심 (%.2f, %.2f)", a, p.point.x, p.point.y);
  }

  // 파라미터
  std::string global_frame_, odom_frame_, base_frame_, scan_frame_;
  bool publish_map_odom_ = true;
  double tf_rate_, transform_tolerance_;
  double scan_rate_, scan_angle_min_, scan_angle_max_, scan_angle_inc_, scan_range_min_, scan_range_max_;
  int occupied_threshold_;
  double speed_limit_, xy_sigma0_, xy_per_m_, yaw_sigma0_, yaw_per_rad_, bump_accel_, bump_holdoff_;

  // 상태
  bool smoke_ = false;
  std::optional<tf2::Transform> map_odom_;
  std::optional<Grid> grid_;
  std::optional<Grid> obs_grid_;          // 연기 모드 동안 grid_ + 지도에 없는 장애물 (가상 스캔용)
  std::vector<uint8_t> obs_count_;         // 칸마다 지도에 없는 장애물로 찍힌 수
  int obs_cells_ = 0;
  bool obs_enabled_ = true;
  double obs_min_gap_ = 0.25, obs_max_range_ = 1.5, obs_depth_ = 0.3;
  int obs_min_hits_ = 2;
  double obs_carve_margin_ = 0.1;
  double echo_tf_tolerance_ = 0.3;
  double obs_always_within_ = 0.5;
  tf2::Transform last_odom_base_;
  double dist_m_ = 0.0, turn_rad_ = 0.0;
  std::optional<rclcpp::Time> last_bump_;

  std::unique_ptr<tf2_ros::Buffer> tf_buffer_;
  std::unique_ptr<tf2_ros::TransformListener> tf_listener_;
  std::unique_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;
  rclcpp::Publisher<std_msgs::msg::Bool>::SharedPtr active_pub_;
  rclcpp::Publisher<geometry_msgs::msg::PoseWithCovarianceStamped>::SharedPtr pose_pub_, initialpose_pub_;
  rclcpp::Publisher<geometry_msgs::msg::PointStamped>::SharedPtr bump_pub_;
  rclcpp::Publisher<nav2_msgs::msg::SpeedLimit>::SharedPtr speed_pub_;
  rclcpp::Publisher<sensor_msgs::msg::LaserScan>::SharedPtr scan_loc_pub_, scan_obs_pub_;
  rclcpp::Subscription<sensor_msgs::msg::LaserScan>::SharedPtr scan_sub_;
  std::string localization_;
  double slow_dist_, slow_speed_;
  bool front_close_ = false;
  std::vector<std::string> slow_topics_;
  std::vector<bool> close_;
  std::vector<rclcpp::Subscription<sensor_msgs::msg::Range>::SharedPtr> slow_subs_;
  rclcpp::node_interfaces::OnSetParametersCallbackHandle::SharedPtr param_cb_;
  rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr imu_sub_;
  rclcpp::Subscription<nav_msgs::msg::OccupancyGrid>::SharedPtr map_sub_;
  rclcpp::Subscription<geometry_msgs::msg::PoseWithCovarianceStamped>::SharedPtr amcl_sub_, initialpose_sub_;
  rclcpp::Client<nav2_msgs::srv::ClearEntireCostmap>::SharedPtr clear_local_, clear_global_;
  rclcpp::Service<std_srvs::srv::SetBool>::SharedPtr set_srv_;
  rclcpp::TimerBase::SharedPtr tf_timer_, scan_timer_, speed_timer_, wall_timer_;
  // wall
  struct Echo {
    rclcpp::Time stamp;
    tf2::Transform odom_sensor;
    double r, fov;
  };
  std::vector<std::string> wall_topics_;
  double wall_window_, wall_rate_, wall_blend_, wall_echo_ratio_, wall_dist_cap_;
  double wall_init_xy_, wall_init_yaw_, wall_q_xy_, wall_q_yaw_;
  std::array<double, 9> wall_cov_{};   // wall 추정 불확실성 (dx, dy, dyaw) 3 × 3 행 우선
  tf2::Transform wall_last_ob_;
  int wall_min_hits_, wall_max_hits_, wall_tried_ = 0, wall_applied_ = 0;
  WallParams wp_;
  std::optional<DistField> dist_;
  std::deque<Echo> echoes_;
  std::vector<rclcpp::Subscription<sensor_msgs::msg::Range>::SharedPtr> range_subs_;
  rclcpp::Publisher<std_msgs::msg::Float64MultiArray>::SharedPtr wall_pub_;
};

}  // namespace limo_fire_navigation

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<limo_fire_navigation::SmokeModeNode>());
  rclcpp::shutdown();
  return 0;
}
