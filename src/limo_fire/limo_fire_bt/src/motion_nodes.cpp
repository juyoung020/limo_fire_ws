// 주행 노드: 순찰점 읽기·고르기, Nav2 목표, 멈춤, 안전거리 감시, 연기 구역 훑기, 프론티어 탐사
#include <cmath>
#include <memory>
#include <string>

#include <behaviortree_cpp/action_node.h>
#include <geometry_msgs/msg/twist.hpp>
#include <sensor_msgs/msg/range.hpp>
#include <std_msgs/msg/string.hpp>
#include <yaml-cpp/yaml.h>

#include "limo_fire_bt/nodes.hpp"
#include "limo_fire_bt/ros_helpers.hpp"

namespace limo_fire_bt
{

namespace
{
using BT::InputPort;
using BT::NodeStatus;
using BT::OutputPort;
using BT::PortsList;

double wrap(double a) {return std::atan2(std::sin(a), std::cos(a));}

class RosAction : public BT::StatefulActionNode
{
public:
  RosAction(const std::string & name, const BT::NodeConfig & cfg, std::shared_ptr<RosContext> ctx)
  : BT::StatefulActionNode(name, cfg), ctx_(std::move(ctx)) {}

protected:
  std::shared_ptr<RosContext> ctx_;
};

// 순찰점 파일 읽기. 형식 (sim/webots/README.md "월드 데이터 파일" 과 같음):
//   patrol: [[x, y], ...]        지도 좌표 (m)
//   robot_start: {x, y, yaw}     있으면 patrol 은 세계 좌표 → 지도 원점 = robot_start 로 옮긴다 (시뮬 월드 yaml)
// 방향은 다음 순찰점 쪽
class LoadPatrolRoute : public BT::SyncActionNode
{
public:
  LoadPatrolRoute(const std::string & name, const BT::NodeConfig & cfg, std::shared_ptr<RosContext> ctx)
  : BT::SyncActionNode(name, cfg), ctx_(std::move(ctx)) {}
  static PortsList providedPorts()
  {
    return {InputPort<std::string>("file", "순찰점 yaml"), OutputPort<Route>("route")};
  }
  NodeStatus tick() override
  {
    const auto file = need<std::string>(*this, "file");
    Route route;
    try {
      const YAML::Node y = YAML::LoadFile(file);
      Pose2 start;
      if (const auto s = y["robot_start"]) {
        start = {s["x"].as<double>(), s["y"].as<double>(), s["yaw"].as<double>()};
      }
      const double c = std::cos(-start.yaw), s = std::sin(-start.yaw);
      for (const auto & p : y["patrol"]) {
        const double dx = p[0].as<double>() - start.x, dy = p[1].as<double>() - start.y;
        route.push_back({c * dx - s * dy, s * dx + c * dy, 0.0});
      }
    } catch (const std::exception & e) {
      RCLCPP_ERROR(ctx_->logger(), "순찰점 파일 %s 를 못 읽음: %s", file.c_str(), e.what());
      return NodeStatus::FAILURE;
    }
    if (route.empty()) {
      RCLCPP_ERROR(ctx_->logger(), "순찰점 파일 %s: patrol 이 비었음", file.c_str());
      return NodeStatus::FAILURE;
    }
    for (size_t i = 0; i < route.size(); ++i) {
      const auto & n = route[(i + 1) % route.size()];
      route[i].yaw = route.size() > 1 ? std::atan2(n.y - route[i].y, n.x - route[i].x) : 0.0;
    }
    RCLCPP_INFO(ctx_->logger(), "순찰점 %zu 개 (%s)", route.size(), file.c_str());
    setOutput("route", route);
    return NodeStatus::SUCCESS;
  }

private:
  std::shared_ptr<RosContext> ctx_;
};

// 순찰점을 차례로 (끝나면 처음부터). 다음 순찰을 이어서 하도록 위치는 노드가 기억한다
class GetNextWaypoint : public BT::SyncActionNode
{
public:
  GetNextWaypoint(const std::string & name, const BT::NodeConfig & cfg, std::shared_ptr<RosContext> ctx)
  : BT::SyncActionNode(name, cfg), ctx_(std::move(ctx)) {}
  static PortsList providedPorts() {return {InputPort<Route>("route"), OutputPort<Pose2>("goal")};}
  NodeStatus tick() override
  {
    const auto route = need<Route>(*this, "route");
    if (route.empty()) {return NodeStatus::FAILURE;}
    const size_t i = next_++ % route.size();
    setOutput("goal", route[i]);
    ctx_->set_status("patrol", std::to_string(i + 1) + "/" + std::to_string(route.size()) + " (" + to_text(route[i]) + ")");
    return NodeStatus::SUCCESS;
  }

private:
  std::shared_ptr<RosContext> ctx_;
  size_t next_ = 0;
};

// Nav2 navigate_to_pose 로 goal 까지
class NavigateToWaypoint : public RosAction
{
public:
  using RosAction::RosAction;
  static PortsList providedPorts()
  {
    return {InputPort<Pose2>("goal"),
      InputPort<std::string>("action_name", "/navigate_to_pose", "nav2_msgs/action/NavigateToPose")};
  }
  NodeStatus onStart() override
  {
    if (!nav_) {nav_ = std::make_unique<NavGoal>(*ctx_, need<std::string>(*this, "action_name"));}
    goal_ = need<Pose2>(*this, "goal");
    nav_->send(goal_);
    return NodeStatus::RUNNING;
  }
  NodeStatus onRunning() override
  {
    switch (nav_->poll()) {
      case CallState::Succeeded: return NodeStatus::SUCCESS;
      case CallState::Failed:
        RCLCPP_WARN(ctx_->logger(), "순찰점 (%s) 못 감: %s", to_text(goal_).c_str(), nav_->error().c_str());
        return NodeStatus::FAILURE;
      default: return NodeStatus::RUNNING;
    }
  }
  void onHalted() override {if (nav_) {nav_->cancel();}}

private:
  std::unique_ptr<NavGoal> nav_;
  Pose2 goal_;
};

// 순찰점에 못 간 것을 알리고 다음 순찰점으로
class LogPatrolIssue : public BT::SyncActionNode
{
public:
  LogPatrolIssue(const std::string & name, const BT::NodeConfig & cfg, std::shared_ptr<RosContext> ctx)
  : BT::SyncActionNode(name, cfg), ctx_(std::move(ctx)) {}
  static PortsList providedPorts() {return {InputPort<Pose2>("goal")};}
  NodeStatus tick() override
  {
    const auto g = getInput<Pose2>("goal");
    ctx_->event(std::string(R"({"event":"PATROL_POINT_SKIPPED","goal":")") +
      (g ? to_text(g.value()) : std::string()) + "\"}");
    return NodeStatus::SUCCESS;
  }

private:
  std::shared_ptr<RosContext> ctx_;
};

// 멈춤: Nav2 목표를 모두 취소하고 /cmd_vel 0 을 낸다
class StopRobot : public BT::SyncActionNode
{
public:
  StopRobot(const std::string & name, const BT::NodeConfig & cfg, std::shared_ptr<RosContext> ctx)
  : BT::SyncActionNode(name, cfg), ctx_(std::move(ctx)) {}
  static PortsList providedPorts()
  {
    return {InputPort<std::string>("cmd_vel_topic", "/cmd_vel", "geometry_msgs/Twist 0 발행"),
      InputPort<std::string>("action_name", "/navigate_to_pose", "목표를 취소할 Nav2 액션")};
  }
  NodeStatus tick() override
  {
    if (!nav_) {
      nav_ = std::make_unique<NavGoal>(*ctx_, need<std::string>(*this, "action_name"));
      pub_ = ctx_->node()->create_publisher<geometry_msgs::msg::Twist>(
        ctx_->resolve(need<std::string>(*this, "cmd_vel_topic")), 10);
    }
    nav_->cancel_all();
    pub_->publish(geometry_msgs::msg::Twist());
    return NodeStatus::SUCCESS;
  }

private:
  std::shared_ptr<RosContext> ctx_;
  std::unique_ptr<NavGoal> nav_;
  rclcpp::Publisher<geometry_msgs::msg::Twist>::SharedPtr pub_;
};

// 불에서 safe_distance_m 떨어진 곳(지금 로봇 쪽)으로 가서 불을 보며 머문다. 멈추게 하는 건 위의 Timeout
class MonitorFromSafeDistance : public RosAction
{
public:
  using RosAction::RosAction;
  static PortsList providedPorts()
  {
    return {InputPort<MapPoint>("location"), InputPort<double>("safe_distance_m", 2.0, ""),
      InputPort<std::string>("action_name", "/navigate_to_pose", ""),
      InputPort<std::string>("detections_topic", "/yolo_detector/detections", ""), InputPort<double>("min_score", 0.5, "")};
  }
  NodeStatus onStart() override
  {
    if (!nav_) {nav_ = std::make_unique<NavGoal>(*ctx_, need<std::string>(*this, "action_name"));}
    const auto fire = need<MapPoint>(*this, "location");
    const auto robot = ctx_->robot_pose();
    if (!robot) {return NodeStatus::FAILURE;}
    const double d = need<double>(*this, "safe_distance_m");
    double dx = robot->x - fire.x, dy = robot->y - fire.y;
    const double n = std::hypot(dx, dy);
    if (n < 1e-3) {dx = -std::cos(robot->yaw); dy = -std::sin(robot->yaw);} else {dx /= n; dy /= n;}
    nav_->send({fire.x + dx * d, fire.y + dy * d, std::atan2(-dy, -dx)});
    arrived_ = false;
    return NodeStatus::RUNNING;
  }
  NodeStatus onRunning() override
  {
    if (!arrived_) {
      const auto s = nav_->poll();
      if (s == CallState::Failed) {
        RCLCPP_WARN(ctx_->logger(), "안전거리 자리로 못 감: %s", nav_->error().c_str());
        return NodeStatus::FAILURE;
      }
      if (s == CallState::Succeeded) {
        arrived_ = true;
        RCLCPP_INFO(ctx_->logger(), "안전거리에서 감시 중");
      }
    }
    const int hits = ctx_->detection_hits(need<std::string>(*this, "detections_topic"),
        ctx_->param<std::string>("classes.fire"), need<double>(*this, "min_score"), ctx_->param<double>("hit_window_s"));
    RCLCPP_INFO_THROTTLE(ctx_->logger(), *ctx_->node()->get_clock(), 10000, "감시: 최근 %.1f s 불 탐지 %d 장",
      ctx_->param<double>("hit_window_s"), hits);
    return NodeStatus::RUNNING;
  }
  void onHalted() override {if (nav_) {nav_->cancel();}}

private:
  std::unique_ptr<NavGoal> nav_;
  bool arrived_ = false;
};

// 연기 구역 훑기: 연기 모드(smoke_mode/set)를 켜고 zone 둘레 sweep_points 곳을 차례로 간다.
// 초음파가 min_range_m 보다 가까우면 그 점을 건너뛴다. 절반 넘게 못 가면 FAILURE. 끝나거나 멈추면 연기 모드를 끈다
class SweepSmokeZone : public RosAction
{
public:
  using RosAction::RosAction;
  static PortsList providedPorts()
  {
    return {InputPort<MapPoint>("zone"),
      InputPort<std::string>("ultrasonic_topic", "/ultrasonic/front_left", "sensor_msgs/Range"),
      InputPort<double>("min_range_m", 0.25, ""), InputPort<double>("radius_m", 1.5, "zone 에서 훑는 반지름"),
      InputPort<int>("points", 4, "훑는 점 수"),
      InputPort<std::string>("smoke_service", "/smoke_mode/set", "std_srvs/SetBool"),
      InputPort<std::string>("action_name", "/navigate_to_pose", "")};
  }
  NodeStatus onStart() override
  {
    if (!nav_) {
      nav_ = std::make_unique<NavGoal>(*ctx_, need<std::string>(*this, "action_name"));
      smoke_ = std::make_unique<SetBoolCall>(*ctx_, need<std::string>(*this, "smoke_service"));
    }
    const auto zone = need<MapPoint>(*this, "zone");
    const double r = need<double>(*this, "radius_m");
    const int n = std::max(1, need<int>(*this, "points"));
    goals_.clear();
    for (int i = 0; i < n; ++i) {
      const double a = 2 * M_PI * i / n;
      goals_.push_back({zone.x + r * std::cos(a), zone.y + r * std::sin(a), wrap(a + M_PI)});
    }
    index_ = 0;
    failed_ = 0;
    smoke_on_ = false;
    smoke_->send(true);
    return NodeStatus::RUNNING;
  }
  NodeStatus onRunning() override
  {
    if (!smoke_on_) {
      const auto s = smoke_->poll();
      if (s == CallState::Failed) {
        RCLCPP_WARN(ctx_->logger(), "연기 모드 못 켬: %s", smoke_->error().c_str());
        return NodeStatus::FAILURE;
      }
      if (s != CallState::Succeeded) {return NodeStatus::RUNNING;}
      smoke_on_ = true;
      nav_->send(goals_[index_]);
    }
    const auto range = ctx_->latest<sensor_msgs::msg::Range>(need<std::string>(*this, "ultrasonic_topic"),
        rclcpp::SensorDataQoS());
    const bool blocked = range && std::isfinite(range->range) && range->range < need<double>(*this, "min_range_m");
    const auto s = nav_->poll();
    if (s == CallState::Active && !blocked) {return NodeStatus::RUNNING;}
    if (s == CallState::Waiting) {return NodeStatus::RUNNING;}
    if (s != CallState::Succeeded) {
      nav_->cancel();
      ++failed_;
      RCLCPP_WARN(ctx_->logger(), "연기 구역 점 %zu 건너뜀: %s", index_,
        blocked ? "초음파 가까움" : nav_->error().c_str());
    }
    if (++index_ < goals_.size()) {
      nav_->send(goals_[index_]);
      return NodeStatus::RUNNING;
    }
    finish();
    return 2 * failed_ > static_cast<int>(goals_.size()) ? NodeStatus::FAILURE : NodeStatus::SUCCESS;
  }
  void onHalted() override
  {
    if (nav_) {nav_->cancel();}
    finish();
  }

private:
  void finish()
  {
    if (smoke_ && !smoke_->fire_and_forget(false)) {
      RCLCPP_WARN(ctx_->logger(), "연기 모드 끄기 못 보냄 (서비스 없음)");
    }
  }

  std::unique_ptr<NavGoal> nav_;
  std::unique_ptr<SetBoolCall> smoke_;
  std::vector<Pose2> goals_;
  size_t index_ = 0;
  int failed_ = 0;
  bool smoke_on_ = false;
};

// 프론티어 탐사 노드(limo_fire_rl explore_frontier)를 켜고 state 가 done 이 될 때까지. 멈추면 끈다.
// survivors = 탐사하며 지도에 올라간 사람 수
class ExploreFrontiers : public RosAction
{
public:
  using RosAction::RosAction;
  static PortsList providedPorts()
  {
    return {InputPort<std::string>("explorer", "/explore_frontier", "탐사 노드 이름 (~/enable, ~/state)"),
      InputPort<std::string>("objects_topic", "/object_mapper/objects", ""),
      OutputPort<int>("survivors")};
  }
  NodeStatus onStart() override
  {
    const auto ns = need<std::string>(*this, "explorer");
    if (!enable_) {enable_ = std::make_unique<SetBoolCall>(*ctx_, ns + "/enable");}
    state_topic_ = ns + "/state";
    enabled_ = false;
    enable_->send(true);
    return NodeStatus::RUNNING;
  }
  NodeStatus onRunning() override
  {
    setOutput("survivors", static_cast<int>(ctx_->objects(need<std::string>(*this, "objects_topic"),
      ctx_->param<std::string>("classes.person"), 0.0).size()));
    if (!enabled_) {
      const auto s = enable_->poll();
      if (s == CallState::Failed) {
        RCLCPP_WARN(ctx_->logger(), "탐사 못 켬: %s", enable_->error().c_str());
        return NodeStatus::FAILURE;
      }
      if (s != CallState::Succeeded) {return NodeStatus::RUNNING;}
      enabled_ = true;
      started_ = ctx_->now();
    }
    const auto st = ctx_->latest<std_msgs::msg::String>(state_topic_, rclcpp::QoS(1).reliable().transient_local());
    // 켠 직후 남아 있던 done 을 끝으로 읽지 않게 잠깐 기다린다
    if (!st || (ctx_->now() - started_).seconds() < 1.0) {return NodeStatus::RUNNING;}
    if (st->data == "done") {
      enable_->fire_and_forget(false);
      return NodeStatus::SUCCESS;
    }
    if (st->data == "idle") {
      RCLCPP_WARN(ctx_->logger(), "탐사 노드가 멈춤(idle)");
      return NodeStatus::FAILURE;
    }
    return NodeStatus::RUNNING;
  }
  void onHalted() override
  {
    if (enable_ && !enable_->fire_and_forget(false)) {RCLCPP_WARN(ctx_->logger(), "탐사 끄기 못 보냄");}
  }

private:
  std::unique_ptr<SetBoolCall> enable_;
  std::string state_topic_;
  bool enabled_ = false;
  rclcpp::Time started_;
};

}  // namespace

void register_motion_nodes(BT::BehaviorTreeFactory & f, const std::shared_ptr<RosContext> & ctx)
{
  f.registerNodeType<LoadPatrolRoute>("LoadPatrolRoute", ctx);
  f.registerNodeType<GetNextWaypoint>("GetNextWaypoint", ctx);
  f.registerNodeType<NavigateToWaypoint>("NavigateToWaypoint", ctx);
  f.registerNodeType<LogPatrolIssue>("LogPatrolIssue", ctx);
  f.registerNodeType<StopRobot>("StopRobot", ctx);
  f.registerNodeType<MonitorFromSafeDistance>("MonitorFromSafeDistance", ctx);
  f.registerNodeType<SweepSmokeZone>("SweepSmokeZone", ctx);
  f.registerNodeType<ExploreFrontiers>("ExploreFrontiers", ctx);
}

}  // namespace limo_fire_bt
