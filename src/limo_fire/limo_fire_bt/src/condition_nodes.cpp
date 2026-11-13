// 조건 노드: 센서가 살아 있는가, 불·연기가 보이는가, 위치 추정이 안정한가, 같은 사건인가 …
// 조건은 한 틱 안에 답한다 (RUNNING 없음). 기다려야 하는 것(IsDepthAlignedToColor)은 액션으로 둔다
#include <cmath>
#include <memory>
#include <string>

#include <behaviortree_cpp/action_node.h>
#include <behaviortree_cpp/condition_node.h>
#include <geometry_msgs/msg/pose_with_covariance_stamped.hpp>
#include <nav_msgs/msg/occupancy_grid.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/joy.hpp>

#include "limo_fire_bt/nodes.hpp"

namespace limo_fire_bt
{

namespace
{
using BT::InputPort;
using BT::NodeStatus;
using BT::OutputPort;
using BT::PortsList;

class RosCondition : public BT::ConditionNode
{
public:
  RosCondition(const std::string & name, const BT::NodeConfig & cfg, std::shared_ptr<RosContext> ctx)
  : BT::ConditionNode(name, cfg), ctx_(std::move(ctx)) {}

protected:
  std::shared_ptr<RosContext> ctx_;
};

// 토픽이 window_s 동안 min_hz 이상으로 들어오는가
class IsTopicAlive : public RosCondition
{
public:
  // 트리를 만들 때 바로 구독해 둔다: 첫 점검에서 "아직 못 잼" 으로 실패하지 않게
  IsTopicAlive(const std::string & name, const BT::NodeConfig & cfg, std::shared_ptr<RosContext> ctx)
  : RosCondition(name, cfg, std::move(ctx))
  {
    if (const auto t = getInput<std::string>("topic")) {ctx_->watch(t.value());}
  }
  static PortsList providedPorts()
  {
    return {InputPort<std::string>("topic"), InputPort<double>("min_hz", "window_s 동안 평균 주기가 이 이상"),
      InputPort<double>("window_s", 1.0, "")};
  }
  NodeStatus tick() override
  {
    const auto topic = need<std::string>(*this, "topic");
    const double min_hz = ctx_->min_hz(topic, need<double>(*this, "min_hz"));
    const auto hz = ctx_->rate(topic, need<double>(*this, "window_s"));
    if (hz && *hz >= min_hz) {return NodeStatus::SUCCESS;}
    RCLCPP_WARN_THROTTLE(ctx_->logger(), *ctx_->node()->get_clock(), 5000, "%s: %.1f Hz < %.1f Hz",
      ctx_->resolve(topic).c_str(), hz.value_or(0.0), min_hz);
    return NodeStatus::FAILURE;
  }
};

// 리모컨이 켜져 있고 SWB 가 위(오토)인가. 시뮬처럼 리모컨이 없으면 설정 rc.required: false
class IsAutoMode : public RosCondition
{
public:
  using RosCondition::RosCondition;
  static PortsList providedPorts()
  {
    return {InputPort<std::string>("rc_topic", "/rc_state", "sensor_msgs/Joy, SWB 위 = 오토")};
  }
  NodeStatus tick() override
  {
    if (!ctx_->param<bool>("rc.required")) {return NodeStatus::SUCCESS;}
    rclcpp::Time stamp;
    const auto joy = ctx_->latest<sensor_msgs::msg::Joy>(need<std::string>(*this, "rc_topic"),
        rclcpp::SensorDataQoS(), &stamp);
    if (!joy || (ctx_->now() - stamp).seconds() > ctx_->param<double>("rc.timeout_s")) {return NodeStatus::FAILURE;}
    const auto i = static_cast<size_t>(ctx_->param<int>("rc.switch_index"));
    return i < joy->buttons.size() && joy->buttons[i] == ctx_->param<int>("rc.auto_value") ?
           NodeStatus::SUCCESS : NodeStatus::FAILURE;
  }
};

// 불·연기 감지. 탐지가 hit_window_s 안에 min_hits 장 이상이면 SUCCESS, 위치·크기를 낸다.
// 위치: use_depth 면 물체 지도(object_mapper ~/objects)에서 로봇에 가장 가까운 같은 종류,
//       아니거나 지도에 아직 없으면 카메라 방향으로 approx_distance_m 앞 (대략)
class IsHazardDetected : public RosCondition
{
public:
  IsHazardDetected(const std::string & name, const BT::NodeConfig & cfg, std::shared_ptr<RosContext> ctx,
    std::string cls)
  : RosCondition(name, cfg, std::move(ctx)), cls_(std::move(cls)) {}

  static PortsList ports(bool with_size)
  {
    PortsList p{
      InputPort<std::string>("detections_topic", "/yolo_detector/detections", "vision_msgs/Detection2DArray"),
      InputPort<std::string>("objects_topic", "/object_mapper/objects", "vision_msgs/Detection3DArray, 지도 위 물체"),
      InputPort<double>("min_score", 0.5, ""), InputPort<int>("min_hits", 5, ""),
      InputPort<bool>("use_depth", false, "false 면 로봇 위치 + 방향으로 대략 위치"),
      OutputPort<MapPoint>("location")};
    if (with_size) {p.insert(OutputPort<double>("size", "불 폭 (m)"));}
    return p;
  }

  NodeStatus tick() override
  {
    const double min_score = need<double>(*this, "min_score");
    DetectionFrame last;
    const int hits = ctx_->detection_hits(need<std::string>(*this, "detections_topic"), cls_, min_score,
        ctx_->param<double>("hit_window_s"), &last);
    if (hits < need<int>(*this, "min_hits")) {return NodeStatus::FAILURE;}

    std::optional<MapPoint> where;
    double size = 0;
    if (need<bool>(*this, "use_depth")) {
      const auto robot = ctx_->robot_pose();
      double best = 1e9;
      for (const auto & o : ctx_->objects(need<std::string>(*this, "objects_topic"), cls_, 0.0)) {
        const double d = robot ? std::hypot(o.p.x - robot->x, o.p.y - robot->y) : 0.0;
        if (d < best) {best = d; where = o.p; size = o.size;}
      }
    }
    if (!where) {
      // 마지막으로 찍힌 장에서 가장 확실한 상자 → 깊이로 잰 지도 위치 (깊이가 없으면 고정 거리로 대략)
      const vision_msgs::msg::BoundingBox2D * box = nullptr;
      double best = -1;
      for (const auto & d : last.msg->detections) {
        for (const auto & r : d.results) {
          if (r.hypothesis.class_id == cls_ && r.hypothesis.score > best) {
            best = r.hypothesis.score;
            box = &d.bbox;
          }
        }
      }
      const auto fix = box ? ctx_->locate_box(*box) : std::nullopt;
      if (!fix) {return NodeStatus::FAILURE;}   // 카메라 정보·TF 가 없으면 위치를 낼 수 없다
      if (!fix->measured) {
        RCLCPP_WARN_THROTTLE(ctx_->logger(), *ctx_->node()->get_clock(), 10000,
          "%s 거리를 깊이로 못 잼 — 카메라 앞 %.1f m 로 대략", cls_.c_str(), ctx_->param<double>("approx_distance_m"));
      }
      where = fix->p;
      if (size <= 0) {size = fix->width_m;}
    }
    setOutput("location", *where);
    if (config().output_ports.count("size")) {setOutput("size", size);}
    return NodeStatus::SUCCESS;
  }

private:
  std::string cls_;
};

class IsFireDetected : public IsHazardDetected
{
public:
  IsFireDetected(const std::string & name, const BT::NodeConfig & cfg, std::shared_ptr<RosContext> ctx)
  : IsHazardDetected(name, cfg, ctx, ctx->param<std::string>("classes.fire")) {}
  static PortsList providedPorts() {return ports(true);}
};

class IsSmokeDetected : public IsHazardDetected
{
public:
  IsSmokeDetected(const std::string & name, const BT::NodeConfig & cfg, std::shared_ptr<RosContext> ctx)
  : IsHazardDetected(name, cfg, ctx, ctx->param<std::string>("classes.smoke")) {}
  static PortsList providedPorts() {return ports(false);}
};

// 진압 확인: 불이 hold_s 동안 min_score 이상으로 안 보였는가. 탐지가 끊겼으면 확인할 수 없으므로 FAILURE
class IsFireGone : public RosCondition
{
public:
  using RosCondition::RosCondition;
  static PortsList providedPorts()
  {
    return {InputPort<std::string>("detections_topic", "/yolo_detector/detections", ""),
      InputPort<MapPoint>("target"), InputPort<double>("min_score", 0.35, ""), InputPort<double>("hold_s", 5.0, "")};
  }
  NodeStatus tick() override
  {
    const auto since = ctx_->since_last_detection(need<std::string>(*this, "detections_topic"),
        ctx_->param<std::string>("classes.fire"), need<double>(*this, "min_score"));
    return since && *since >= need<double>(*this, "hold_s") ? NodeStatus::SUCCESS : NodeStatus::FAILURE;
  }
};

// AMCL 공분산이 문턱 안인가
class IsLocalizationStable : public RosCondition
{
public:
  using RosCondition::RosCondition;
  static PortsList providedPorts()
  {
    return {InputPort<std::string>("pose_topic", "/amcl_pose", "geometry_msgs/PoseWithCovarianceStamped"),
      InputPort<double>("max_xy_std_m", 0.3, ""), InputPort<double>("max_yaw_std_rad", 0.2, "")};
  }
  NodeStatus tick() override
  {
    // AMCL 은 움직일 때만 내고 마지막 값을 transient local 로 둔다
    const auto p = ctx_->latest<geometry_msgs::msg::PoseWithCovarianceStamped>(
      need<std::string>(*this, "pose_topic"), rclcpp::QoS(1).reliable().transient_local());
    if (!p) {return NodeStatus::FAILURE;}
    const auto & c = p->pose.covariance;
    const double xy = std::sqrt(std::max(c[0], c[7]));
    const double yaw = std::sqrt(c[35]);
    return xy <= need<double>(*this, "max_xy_std_m") && yaw <= need<double>(*this, "max_yaw_std_rad") ?
           NodeStatus::SUCCESS : NodeStatus::FAILURE;
  }
};

// 지금 사건(incident)의 위치에서 radius_m 안인가
class IsSameIncident : public RosCondition
{
public:
  using RosCondition::RosCondition;
  static PortsList providedPorts()
  {
    return {InputPort<int>("incident"), InputPort<MapPoint>("location"), InputPort<double>("radius_m")};
  }
  NodeStatus tick() override
  {
    const auto at = ctx_->incident_location(need<int>(*this, "incident"));
    if (!at) {return NodeStatus::FAILURE;}
    const auto p = need<MapPoint>(*this, "location");
    return std::hypot(p.x - at->x, p.y - at->y) <= need<double>(*this, "radius_m") ?
           NodeStatus::SUCCESS : NodeStatus::FAILURE;
  }
};

// 이미 처리한 사건 자리의 불인가: incident 위치에서 radius_m 안이고, 마지막 처리 뒤 recheck_s 가 안 지났으면 SUCCESS.
// 순찰 중 같은 불을 볼 때마다 순찰을 멈추고 감시로 돌아가는 고리를 막는다 (recheck_s 가 지나면 다시 멈춰 감시)
class IsKnownHazard : public RosCondition
{
public:
  using RosCondition::RosCondition;
  static PortsList providedPorts()
  {
    return {InputPort<int>("incident"), InputPort<MapPoint>("location"), InputPort<double>("radius_m"),
      InputPort<double>("recheck_s", "처리 뒤 이 시간이 지나면 같은 자리 불도 다시 멈춰 감시")};
  }
  NodeStatus tick() override
  {
    const int id = need<int>(*this, "incident");
    const auto at = ctx_->incident_location(id);
    const auto since = ctx_->since_incident_handled(id);
    if (!at || !since) {return NodeStatus::FAILURE;}
    const auto p = need<MapPoint>(*this, "location");
    if (std::hypot(p.x - at->x, p.y - at->y) > need<double>(*this, "radius_m")) {return NodeStatus::FAILURE;}
    const double left = need<double>(*this, "recheck_s") - *since;
    if (left <= 0) {return NodeStatus::FAILURE;}
    RCLCPP_INFO_THROTTLE(ctx_->logger(), *ctx_->node()->get_clock(), 10000,
      "사건 %d 자리 불 (이미 처리함) — 순찰 계속, %.0f s 뒤 다시 감시", id, left);
    return NodeStatus::SUCCESS;
  }
};

// 서비스가 그래프에 있는가 (타입은 보지 않는다)
class IsServiceAvailable : public RosCondition
{
public:
  using RosCondition::RosCondition;
  static PortsList providedPorts() {return {InputPort<std::string>("service")};}
  NodeStatus tick() override
  {
    const auto all = ctx_->node()->get_service_names_and_types();
    return all.count(ctx_->resolve(need<std::string>(*this, "service"))) ? NodeStatus::SUCCESS : NodeStatus::FAILURE;
  }
};

// 불 폭이 max_size 이하인가. 폭을 모르면(0) 작다고 하지 않는다
class IsSmallFire : public BT::ConditionNode
{
public:
  using BT::ConditionNode::ConditionNode;
  static PortsList providedPorts() {return {InputPort<double>("max_size"), InputPort<double>("size")};}
  NodeStatus tick() override
  {
    const double s = need<double>(*this, "size");
    return s > 0 && s <= need<double>(*this, "max_size") ? NodeStatus::SUCCESS : NodeStatus::FAILURE;
  }
};

// 관리자가 요구조자 수색을 요청했는가 (~/request_survivor_search)
class IsSurvivorSearchRequested : public RosCondition
{
public:
  using RosCondition::RosCondition;
  static PortsList providedPorts() {return {};}
  NodeStatus tick() override
  {
    return ctx_->survivor_search_requested() ? NodeStatus::SUCCESS : NodeStatus::FAILURE;
  }
};

// 열화상 최고 온도가 최근 최고값보다 min_drop_c 이상 내려갔는가.
// 열화상 토픽은 잠정: sensor_msgs/Image 32FC1, 픽셀 = °C (부품 대기)
class IsTemperatureDropped : public RosCondition
{
public:
  using RosCondition::RosCondition;
  static PortsList providedPorts()
  {
    return {InputPort<MapPoint>("target"), InputPort<std::string>("thermal_topic", "/thermal/image_raw", "잠정"),
      InputPort<double>("min_drop_c")};
  }
  NodeStatus tick() override
  {
    const auto img = ctx_->latest<sensor_msgs::msg::Image>(need<std::string>(*this, "thermal_topic"),
        rclcpp::SensorDataQoS());
    if (!img || img->encoding != "32FC1" || img->data.empty()) {return NodeStatus::FAILURE;}
    const auto * px = reinterpret_cast<const float *>(img->data.data());
    float hot = -1e9f;
    for (size_t i = 0; i < img->data.size() / 4; ++i) {
      if (std::isfinite(px[i])) {hot = std::max(hot, px[i]);}
    }
    // 최고값은 사건이 바뀌어도 이 노드가 살아 있는 동안 이어진다: target 이 바뀌면 다시 잰다
    const auto t = need<MapPoint>(*this, "target");
    if (std::hypot(t.x - target_.x, t.y - target_.y) > 0.5) {target_ = t; peak_ = hot;}
    peak_ = std::max(peak_, static_cast<double>(hot));
    return peak_ - hot >= need<double>(*this, "min_drop_c") ? NodeStatus::SUCCESS : NodeStatus::FAILURE;
  }

private:
  MapPoint target_{1e9, 1e9, 0};
  double peak_ = -1e9;
};

// StartMissionTimer 이후 limit_sec 안인가
class IsWithinTimeLimit : public RosCondition
{
public:
  using RosCondition::RosCondition;
  static PortsList providedPorts() {return {InputPort<double>("limit_sec"), InputPort<double>("start_time")};}
  NodeStatus tick() override
  {
    return ctx_->now().seconds() - need<double>(*this, "start_time") <= need<double>(*this, "limit_sec") ?
           NodeStatus::SUCCESS : NodeStatus::FAILURE;
  }
};

// near 둘레 radius_m 안에 프론티어 칸(빈 칸 옆 미지 칸)이 min_cells 이상인가
class HasUnexploredFrontier : public RosCondition
{
public:
  using RosCondition::RosCondition;
  static PortsList providedPorts()
  {
    return {InputPort<MapPoint>("near"), InputPort<std::string>("map_topic", "/map", ""),
      InputPort<double>("radius_m", 5.0, ""), InputPort<int>("min_cells", 10, "")};
  }
  NodeStatus tick() override
  {
    const auto map = ctx_->latest<nav_msgs::msg::OccupancyGrid>(need<std::string>(*this, "map_topic"),
        rclcpp::QoS(1).reliable().transient_local());
    if (!map || map->info.resolution <= 0) {return NodeStatus::FAILURE;}
    const auto c = need<MapPoint>(*this, "near");
    const double r = need<double>(*this, "radius_m");
    const auto & info = map->info;
    const int w = static_cast<int>(info.width), h = static_cast<int>(info.height);
    const double res = info.resolution;
    const int cx = static_cast<int>((c.x - info.origin.position.x) / res);
    const int cy = static_cast<int>((c.y - info.origin.position.y) / res);
    const int rc = static_cast<int>(r / res);
    const int free_max = ctx_->param<int>("frontier.free_max");
    auto at = [&](int x, int y) {return map->data[static_cast<size_t>(y) * w + x];};
    int count = 0;
    for (int y = std::max(1, cy - rc); y < std::min(h - 1, cy + rc); ++y) {
      for (int x = std::max(1, cx - rc); x < std::min(w - 1, cx + rc); ++x) {
        if ((x - cx) * (x - cx) + (y - cy) * (y - cy) > rc * rc) {continue;}
        const int v = at(x, y);
        if (v < 0 || v > free_max) {continue;}
        if (at(x + 1, y) < 0 || at(x - 1, y) < 0 || at(x, y + 1) < 0 || at(x, y - 1) < 0) {++count;}
      }
    }
    return count >= need<int>(*this, "min_cells") ? NodeStatus::SUCCESS : NodeStatus::FAILURE;
  }
};

// 깊이 영상이 컬러에 정렬돼 있는가 (frame_id·크기). 첫 장을 wait_s 까지 기다린다 → 액션
class IsDepthAlignedToColor : public BT::StatefulActionNode
{
public:
  IsDepthAlignedToColor(const std::string & name, const BT::NodeConfig & cfg, std::shared_ptr<RosContext> ctx)
  : BT::StatefulActionNode(name, cfg), ctx_(std::move(ctx)) {}
  static PortsList providedPorts()
  {
    return {InputPort<std::string>("depth_topic", "/camera/depth/image_raw", ""),
      InputPort<std::string>("expected_frame", "camera_color_optical_frame", ""),
      InputPort<int>("expected_width", 640, ""), InputPort<int>("expected_height", 480, ""),
      InputPort<double>("wait_s", 10.0, "첫 깊이 영상을 기다리는 시간")};
  }
  NodeStatus onStart() override
  {
    start_ = ctx_->now();
    return onRunning();
  }
  NodeStatus onRunning() override
  {
    const auto img = ctx_->latest<sensor_msgs::msg::Image>(need<std::string>(*this, "depth_topic"),
        rclcpp::SensorDataQoS());
    if (!img) {
      if ((ctx_->now() - start_).seconds() < need<double>(*this, "wait_s")) {return NodeStatus::RUNNING;}
      RCLCPP_WARN(ctx_->logger(), "깊이 영상이 %.0f s 동안 안 옴", need<double>(*this, "wait_s"));
      return NodeStatus::FAILURE;
    }
    const bool ok = img->header.frame_id == need<std::string>(*this, "expected_frame") &&
      static_cast<int>(img->width) == need<int>(*this, "expected_width") &&
      static_cast<int>(img->height) == need<int>(*this, "expected_height");
    if (!ok) {
      RCLCPP_WARN(ctx_->logger(), "깊이 정렬 아님: %s %ux%u", img->header.frame_id.c_str(), img->width, img->height);
    }
    return ok ? NodeStatus::SUCCESS : NodeStatus::FAILURE;
  }
  void onHalted() override {}

private:
  std::shared_ptr<RosContext> ctx_;
  rclcpp::Time start_;
};

}  // namespace

void register_condition_nodes(BT::BehaviorTreeFactory & f, const std::shared_ptr<RosContext> & ctx)
{
  f.registerNodeType<IsTopicAlive>("IsTopicAlive", ctx);
  f.registerNodeType<IsAutoMode>("IsAutoMode", ctx);
  f.registerNodeType<IsFireDetected>("IsFireDetected", ctx);
  f.registerNodeType<IsSmokeDetected>("IsSmokeDetected", ctx);
  f.registerNodeType<IsFireGone>("IsFireGone", ctx);
  f.registerNodeType<IsLocalizationStable>("IsLocalizationStable", ctx);
  f.registerNodeType<IsSameIncident>("IsSameIncident", ctx);
  f.registerNodeType<IsKnownHazard>("IsKnownHazard", ctx);
  f.registerNodeType<IsServiceAvailable>("IsServiceAvailable", ctx);
  f.registerNodeType<IsSmallFire>("IsSmallFire");
  f.registerNodeType<IsSurvivorSearchRequested>("IsSurvivorSearchRequested", ctx);
  f.registerNodeType<IsTemperatureDropped>("IsTemperatureDropped", ctx);
  f.registerNodeType<IsWithinTimeLimit>("IsWithinTimeLimit", ctx);
  f.registerNodeType<HasUnexploredFrontier>("HasUnexploredFrontier", ctx);
  f.registerNodeType<IsDepthAlignedToColor>("IsDepthAlignedToColor", ctx);
}

}  // namespace limo_fire_bt
