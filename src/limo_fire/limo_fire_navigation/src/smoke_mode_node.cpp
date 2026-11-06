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
#include <cmath>
#include <memory>
#include <optional>
#include <string>

#include "geometry_msgs/msg/point_stamped.hpp"
#include "geometry_msgs/msg/pose_with_covariance_stamped.hpp"
#include "geometry_msgs/msg/transform_stamped.hpp"
#include "limo_fire_navigation/map_raycast.hpp"
#include "nav2_msgs/msg/speed_limit.hpp"
#include "nav2_msgs/srv/clear_entire_costmap.hpp"
#include "nav_msgs/msg/occupancy_grid.hpp"
#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/imu.hpp"
#include "sensor_msgs/msg/laser_scan.hpp"
#include "sensor_msgs/msg/range.hpp"
#include "std_msgs/msg/bool.hpp"
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
    // 연기 속 위치: hold = 들어갈 때 map → odom 을 붙잡고 바퀴 + IMU 만 따라감 · sonar = AMCL 에 라이다 대신 초음파 스캔(sonar_scan 노드)을 넣어
    // AMCL 이 계속 고친다 (docs/smoke_mode_plan.md 3 단계). 연기 모드가 꺼져 있을 때만 바꿀 수 있다
    // sonar 일 때 AMCL 자세를 그대로 믿지 않는다(2026-11-06 측정: 그대로 믿으면 office·warehouse 에서 위치를 잃음). 바퀴 + IMU 추정과
    // gate_xy·gate_yaw 안으로 맞을 때만 blend 만큼 끌어당기고, reseed_after 번 잇달아 어긋나면 AMCL 을 지금 추정 자리로 다시 놓는다 (잠정 값)
    gate_xy_ = declare_parameter<double>("sonar.gate_xy", 0.5);
    gate_yaw_ = declare_parameter<double>("sonar.gate_yaw", 0.3);
    blend_ = declare_parameter<double>("sonar.blend", 0.3);
    reseed_after_ = declare_parameter<int>("sonar.reseed_after", 5);
    // 앞 초음파가 slow_distance 보다 가까우면 연기 모드 속도 한도를 slow_speed 로 (장애물을 늦게 봐 상자를 못 피함 — 2026-11-06)
    front_topic_ = declare_parameter<std::string>("sonar.front_topic", "/ultrasonic/front");
    slow_dist_ = declare_parameter<double>("sonar.slow_distance", 0.6);
    slow_speed_ = declare_parameter<double>("sonar.slow_speed", 0.05);
    if (gate_xy_ <= 0.0 || gate_yaw_ <= 0.0 || blend_ <= 0.0 || blend_ > 1.0 || reseed_after_ < 1 || slow_dist_ <= 0.0 ||
        slow_speed_ <= 0.0)
      throw std::invalid_argument("smoke_mode: sonar.* 파라미터 확인 (blend 는 (0, 1])");
    localization_ = declare_parameter<std::string>("smoke_localization", "hold");
    if (localization_ != "hold" && localization_ != "sonar")
      throw std::invalid_argument("smoke_mode: smoke_localization 은 hold 또는 sonar");
    param_cb_ = add_on_set_parameters_callback([this](const std::vector<rclcpp::Parameter>& ps) {
      rcl_interfaces::msg::SetParametersResult r;
      r.successful = true;
      for (const auto& p : ps) {
        if (p.get_name() != "smoke_localization") continue;
        const std::string v = p.get_type() == rclcpp::ParameterType::PARAMETER_STRING ? p.as_string() : "";
        if (smoke_) { r.successful = false; r.reason = "연기 모드 중에는 바꿀 수 없음"; }
        else if (v != "hold" && v != "sonar") { r.successful = false; r.reason = "hold 또는 sonar"; }
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
    sonar_sub_ = create_subscription<sensor_msgs::msg::LaserScan>(
        "sonar_scan", rclcpp::SensorDataQoS(), [this](sensor_msgs::msg::LaserScan::SharedPtr m) {
          if (smoke_ && localization_ == "sonar") scan_loc_pub_->publish(*m);   // 연기 속 AMCL 은 초음파만 본다
        });
    front_sub_ = create_subscription<sensor_msgs::msg::Range>(
        front_topic_, rclcpp::SensorDataQoS(), [this](sensor_msgs::msg::Range::SharedPtr m) {
          const bool close = std::isfinite(m->range) && m->range > m->min_range && m->range < slow_dist_;
          if (close != front_close_) {
            front_close_ = close;
            if (smoke_) publish_speed_limit();
          }
        });
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

    tf_timer_ = create_wall_timer(std::chrono::duration<double>(1.0 / tf_rate_), [this] { on_tf_tick(); });
    scan_timer_ = create_wall_timer(std::chrono::duration<double>(1.0 / scan_rate_), [this] { on_scan_tick(); });
    speed_timer_ = create_wall_timer(std::chrono::seconds(1), [this] { publish_speed_limit(); });

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
    clear_costmaps();
    publish_active();
    publish_speed_limit();
    RCLCPP_WARN(get_logger(), "연기 모드 켬: 라이다·카메라를 쓰지 않고 %s, 속도 한도 %.2f m/s",
                localization_ == "sonar" ? "바퀴 + IMU + 초음파(AMCL)로 위치를 잡음" : "바퀴 + IMU 로만 움직임", speed_limit_);
    return true;
  }

  void leave() {
    // 지금 추정 자세를 쌓인 불확실성과 함께 AMCL 에 넘긴다. AMCL 은 다시 들어오는 스캔으로 이 근처에서 맞춘다.
    // 상태를 먼저 내린다: 이 노드도 /initialpose 를 받으므로(연기 속 사람 보정용) 자기 메시지를 사람 보정으로 받지 않게
    const auto est = estimate(now());
    smoke_ = false;
    // sonar 면 AMCL 이 연기 속에서도 계속 위치를 고쳐 왔으므로 다시 넣지 않는다 (넣으면 입자를 흩뜨림)
    if (est && localization_ == "hold") initialpose_pub_->publish(*est);
    clear_costmaps();
    publish_active();
    publish_speed_limit();
    RCLCPP_WARN(get_logger(), "평상시로 돌아옴: 연기 속 이동 %.1f m · 회전 %.1f rad, 추정 자세를 AMCL 에 넘김", dist_m_,
                turn_rad_);
  }

  // ---- 위치 ----------------------------------------------------------------------------------------
  void on_amcl_pose(const geometry_msgs::msg::PoseWithCovarianceStamped::SharedPtr msg) {
    // hold: 연기 속에서는 스캔을 끊어 AMCL 이 새 자세를 내지 않지만, 혹시 와도 쓰지 않는다. sonar: 초음파로 고친 자세를 그대로 씀
    if (smoke_ && localization_ == "hold") return;
    if (smoke_) {   // sonar: 문 지킴 (위 파라미터 주석)
      gated_update(*msg);
      return;
    }
    set_map_pose(*msg, rclcpp::Time(msg->header.stamp));
  }

  void on_initialpose(const geometry_msgs::msg::PoseWithCovarianceStamped::SharedPtr msg) {
    if (!smoke_) return;
    if (reseeding_) {   // 방금 이 노드가 AMCL 을 다시 놓은 메시지
      reseeding_ = false;
      return;
    }
    if (set_map_pose(*msg, rclcpp::Time(0, 0, get_clock()->get_clock_type()))) {
      dist_m_ = 0.0;
      turn_rad_ = 0.0;
      RCLCPP_INFO(get_logger(), "연기 모드: 사람이 준 자세로 옮김, 쌓인 불확실성 초기화");
    }
  }

  // 연기 속 sonar: AMCL(초음파) 자세가 지금 추정(붙잡은 map → odom · 바퀴 + IMU)과 문 안이면 blend 만큼만 끌어당긴다
  void gated_update(const geometry_msgs::msg::PoseWithCovarianceStamped& msg) {
    if (msg.header.frame_id != global_frame_ || !map_odom_) return;
    geometry_msgs::msg::TransformStamped t;
    try {
      t = tf_buffer_->lookupTransform(odom_frame_, base_frame_, rclcpp::Time(msg.header.stamp), rclcpp::Duration::from_seconds(0.2));
    } catch (const tf2::TransformException& e) {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000, "자세 시각의 odom → base TF 없음: %s", e.what());
      return;
    }
    tf2::Transform ob, cand;
    tf2::fromMsg(t.transform, ob);
    tf2::fromMsg(msg.pose.pose, cand);
    const tf2::Transform cur = *map_odom_ * ob;
    const double dx = cand.getOrigin().x() - cur.getOrigin().x(), dy = cand.getOrigin().y() - cur.getOrigin().y();
    const double cy = tf2::getYaw(cur.getRotation());
    const double dyaw = std::atan2(std::sin(tf2::getYaw(cand.getRotation()) - cy), std::cos(tf2::getYaw(cand.getRotation()) - cy));
    if (std::hypot(dx, dy) > gate_xy_ || std::fabs(dyaw) > gate_yaw_) {
      ++rejects_;
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000, "초음파 AMCL 자세가 추정과 %.2f m · %.2f rad 어긋나 버림 (%d 번째)",
                           std::hypot(dx, dy), dyaw, rejects_);
      if (rejects_ >= reseed_after_) {   // AMCL 입자를 지금 추정 자리로 다시 모은다
        rejects_ = 0;
        geometry_msgs::msg::PoseWithCovarianceStamped p;
        p.header.stamp = msg.header.stamp;
        p.header.frame_id = global_frame_;
        tf2::toMsg(cur, p.pose.pose);
        p.pose.covariance[0] = p.pose.covariance[7] = 0.15 * 0.15;
        p.pose.covariance[35] = 0.1 * 0.1;
        reseeding_ = true;   // 이 노드도 /initialpose 를 받으므로 자기 메시지를 사람 보정으로 세지 않게
        initialpose_pub_->publish(p);
        RCLCPP_WARN(get_logger(), "초음파 AMCL 을 추정 자리 (%.2f, %.2f) 로 다시 놓음", cur.getOrigin().x(), cur.getOrigin().y());
      }
      return;
    }
    rejects_ = 0;
    tf2::Quaternion q;
    q.setRPY(0, 0, cy + blend_ * dyaw);
    const tf2::Transform next(q, tf2::Vector3(cur.getOrigin().x() + blend_ * dx, cur.getOrigin().y() + blend_ * dy, 0.0));
    map_odom_ = next * ob.inverse();
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
    for (int i = 0; i < n; ++i) {
      s.ranges[i] = static_cast<float>(
          cast(*grid_, lx, ly, lyaw + s.angle_min + i * s.angle_increment, scan_range_min_, no_hit, no_hit));
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
  rclcpp::Subscription<sensor_msgs::msg::LaserScan>::SharedPtr scan_sub_, sonar_sub_;
  std::string localization_;
  double gate_xy_, gate_yaw_, blend_, slow_dist_, slow_speed_;
  int reseed_after_, rejects_ = 0;
  bool reseeding_ = false, front_close_ = false;
  std::string front_topic_;
  rclcpp::Subscription<sensor_msgs::msg::Range>::SharedPtr front_sub_;
  rclcpp::node_interfaces::OnSetParametersCallbackHandle::SharedPtr param_cb_;
  rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr imu_sub_;
  rclcpp::Subscription<nav_msgs::msg::OccupancyGrid>::SharedPtr map_sub_;
  rclcpp::Subscription<geometry_msgs::msg::PoseWithCovarianceStamped>::SharedPtr amcl_sub_, initialpose_sub_;
  rclcpp::Client<nav2_msgs::srv::ClearEntireCostmap>::SharedPtr clear_local_, clear_global_;
  rclcpp::Service<std_srvs::srv::SetBool>::SharedPtr set_srv_;
  rclcpp::TimerBase::SharedPtr tf_timer_, scan_timer_, speed_timer_;
};

}  // namespace limo_fire_navigation

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<limo_fire_navigation::SmokeModeNode>());
  rclcpp::shutdown();
  return 0;
}
