// 부품 노드: 노즐 조준, 펌프, 대피 안내 방송, 사람 추적.
// 펌프·서보·스피커·레이더는 부품 대기라 토픽·서비스 이름과 타입이 "잠정"이다 (트리 XML 머리말, docs/bt_sim_plan.md).
// 없으면 고급 모드 진입 조건에서 걸러지고, 실행 중 사라지면 FAILURE → 베이직 복귀
#include <cmath>
#include <memory>
#include <string>

#include <behaviortree_cpp/action_node.h>
#include <geometry_msgs/msg/point_stamped.hpp>
#include <geometry_msgs/msg/pose_array.hpp>
#include <sensor_msgs/msg/joint_state.hpp>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>

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

class RosAction : public BT::StatefulActionNode
{
public:
  RosAction(const std::string & name, const BT::NodeConfig & cfg, std::shared_ptr<RosContext> ctx)
  : BT::StatefulActionNode(name, cfg), ctx_(std::move(ctx)) {}

protected:
  std::shared_ptr<RosContext> ctx_;
};

// 노즐 조준: 불 위치를 nozzle_frame 으로 옮겨 팬·틸트 각을 구하고 nozzle.command_topic 으로 낸 뒤,
// /joint_states 의 두 관절이 nozzle.tolerance_rad 안에 들어오면 SUCCESS. nozzle.timeout_s 넘으면 FAILURE.
// 각도는 직선 조준 (물줄기 포물선 보정은 조준 RL·실측 후)
class AimNozzle : public RosAction
{
public:
  using RosAction::RosAction;
  static PortsList providedPorts()
  {
    return {InputPort<MapPoint>("target", "불 위치 (map frame)"),
      InputPort<std::string>("joint_states_topic", "/joint_states", "노즐 서보 상태 (잠정)"),
      InputPort<std::string>("nozzle_frame", "nozzle_link", "URDF 의 노즐 링크")};
  }
  NodeStatus onStart() override
  {
    const auto t = need<MapPoint>(*this, "target");
    geometry_msgs::msg::PointStamped in, out;
    in.header.frame_id = ctx_->param<std::string>("map_frame");
    in.point.x = t.x;
    in.point.y = t.y;
    in.point.z = t.z;
    try {
      out = ctx_->tf().transform(in, need<std::string>(*this, "nozzle_frame"), tf2::durationFromSec(0.0));
    } catch (const tf2::TransformException & e) {
      RCLCPP_WARN(ctx_->logger(), "노즐 TF 없음: %s", e.what());
      return NodeStatus::FAILURE;
    }
    pan_ = std::atan2(out.point.y, out.point.x);
    tilt_ = std::atan2(out.point.z, std::hypot(out.point.x, out.point.y));
    if (!pub_) {
      pub_ = ctx_->node()->create_publisher<sensor_msgs::msg::JointState>(
        ctx_->param<std::string>("nozzle.command_topic"), 10);
    }
    sensor_msgs::msg::JointState cmd;
    cmd.header.stamp = ctx_->now();
    cmd.name = {ctx_->param<std::string>("nozzle.pan_joint"), ctx_->param<std::string>("nozzle.tilt_joint")};
    cmd.position = {pan_, tilt_};
    pub_->publish(cmd);
    start_ = ctx_->now();
    return NodeStatus::RUNNING;
  }
  NodeStatus onRunning() override
  {
    const auto js = ctx_->latest<sensor_msgs::msg::JointState>(need<std::string>(*this, "joint_states_topic"),
        rclcpp::QoS(10));
    const double tol = ctx_->param<double>("nozzle.tolerance_rad");
    if (js) {
      double pan = NAN, tilt = NAN;
      for (size_t i = 0; i < js->name.size() && i < js->position.size(); ++i) {
        if (js->name[i] == ctx_->param<std::string>("nozzle.pan_joint")) {pan = js->position[i];}
        if (js->name[i] == ctx_->param<std::string>("nozzle.tilt_joint")) {tilt = js->position[i];}
      }
      if (std::abs(pan - pan_) <= tol && std::abs(tilt - tilt_) <= tol) {return NodeStatus::SUCCESS;}
    }
    if ((ctx_->now() - start_).seconds() > ctx_->param<double>("nozzle.timeout_s")) {
      RCLCPP_WARN(ctx_->logger(), "노즐이 목표 각(%.2f, %.2f)에 못 감", pan_, tilt_);
      return NodeStatus::FAILURE;
    }
    return NodeStatus::RUNNING;
  }
  void onHalted() override {}

private:
  rclcpp::Publisher<sensor_msgs::msg::JointState>::SharedPtr pub_;
  double pan_ = 0, tilt_ = 0;
  rclcpp::Time start_;
};

// 펌프를 duration_ms 동안 켰다 끈다. 켜기가 거절되면 FAILURE, 멈추면 끈다
class SprayWater : public RosAction
{
public:
  using RosAction::RosAction;
  static PortsList providedPorts()
  {
    return {InputPort<std::string>("pump_service", "/pump/set", "std_srvs/SetBool (잠정)"),
      InputPort<int>("duration_ms")};
  }
  NodeStatus onStart() override
  {
    if (!pump_) {pump_ = std::make_unique<SetBoolCall>(*ctx_, need<std::string>(*this, "pump_service"));}
    pump_->send(true);
    on_ = false;
    return NodeStatus::RUNNING;
  }
  NodeStatus onRunning() override
  {
    if (!on_) {
      const auto s = pump_->poll();
      if (s == CallState::Failed) {
        RCLCPP_WARN(ctx_->logger(), "펌프 못 켬: %s", pump_->error().c_str());
        return NodeStatus::FAILURE;
      }
      if (s != CallState::Succeeded) {return NodeStatus::RUNNING;}
      on_ = true;
      since_ = ctx_->now();
    }
    if ((ctx_->now() - since_).seconds() * 1000.0 < need<int>(*this, "duration_ms")) {return NodeStatus::RUNNING;}
    pump_->fire_and_forget(false);
    return NodeStatus::SUCCESS;
  }
  void onHalted() override {if (pump_) {pump_->fire_and_forget(false);}}

private:
  std::unique_ptr<SetBoolCall> pump_;
  bool on_ = false;
  rclcpp::Time since_;
};

// 펌프 끄기 (응답을 기다리지 않음). 서비스가 없어도 SUCCESS — 이미 꺼진 것과 같다
class StopPump : public BT::SyncActionNode
{
public:
  StopPump(const std::string & name, const BT::NodeConfig & cfg, std::shared_ptr<RosContext> ctx)
  : BT::SyncActionNode(name, cfg), ctx_(std::move(ctx)) {}
  static PortsList providedPorts() {return {InputPort<std::string>("pump_service", "/pump/set", "")};}
  NodeStatus tick() override
  {
    if (!pump_) {pump_ = std::make_unique<SetBoolCall>(*ctx_, need<std::string>(*this, "pump_service"));}
    if (!pump_->fire_and_forget(false)) {RCLCPP_WARN(ctx_->logger(), "펌프 서비스 없음: 끄기 못 보냄");}
    return NodeStatus::SUCCESS;
  }

private:
  std::shared_ptr<RosContext> ctx_;
  std::unique_ptr<SetBoolCall> pump_;
};

// 대피 안내 방송: speaker.service(SetBool, 잠정)로 켜고, 멈출 때까지 RUNNING. 서비스가 없거나 사라지면 FAILURE
class PlayEvacuationGuidance : public RosAction
{
public:
  using RosAction::RosAction;
  static PortsList providedPorts() {return {};}
  NodeStatus onStart() override
  {
    if (!speaker_) {speaker_ = std::make_unique<SetBoolCall>(*ctx_, ctx_->param<std::string>("speaker.service"));}
    speaker_->send(true);
    return NodeStatus::RUNNING;
  }
  NodeStatus onRunning() override
  {
    const auto s = speaker_->poll();
    if (s == CallState::Failed) {
      RCLCPP_WARN(ctx_->logger(), "대피 안내 방송 실패: %s", speaker_->error().c_str());
      return NodeStatus::FAILURE;
    }
    if (s == CallState::Succeeded &&
      !ctx_->node()->get_service_names_and_types().count(ctx_->resolve(ctx_->param<std::string>("speaker.service"))))
    {
      return NodeStatus::FAILURE;   // 방송 중 스피커 노드가 죽음
    }
    return NodeStatus::RUNNING;
  }
  void onHalted() override {if (speaker_) {speaker_->fire_and_forget(false);}}

private:
  std::unique_ptr<SetBoolCall> speaker_;
};

// 레이더 사람 추적 (잠정: geometry_msgs/PoseArray, 사람 하나 = 자세 하나). 멈출 때까지 RUNNING, 끊기면 FAILURE
class TrackPeopleRadar : public RosAction
{
public:
  using RosAction::RosAction;
  static PortsList providedPorts()
  {
    return {InputPort<std::string>("radar_topic", "/radar/people", "잠정"), OutputPort<int>("people")};
  }
  NodeStatus onStart() override {return onRunning();}
  NodeStatus onRunning() override
  {
    rclcpp::Time stamp;
    const auto m = ctx_->latest<geometry_msgs::msg::PoseArray>(need<std::string>(*this, "radar_topic"),
        rclcpp::SensorDataQoS(), &stamp);
    if (!m || (ctx_->now() - stamp).seconds() > ctx_->param<double>("people.timeout_s")) {
      return NodeStatus::FAILURE;
    }
    setOutput("people", static_cast<int>(m->poses.size()));
    return NodeStatus::RUNNING;
  }
  void onHalted() override {}
};

// 카메라 사람 추적: 물체 지도에 올라간 사람 수. 멈출 때까지 RUNNING
class TrackPeopleCamera : public RosAction
{
public:
  using RosAction::RosAction;
  static PortsList providedPorts()
  {
    return {InputPort<std::string>("detections_topic", "/yolo_detector/detections", "class person"),
      InputPort<std::string>("objects_topic", "/object_mapper/objects", ""),
      InputPort<double>("min_score", 0.5, ""), OutputPort<int>("people")};
  }
  NodeStatus onStart() override {return onRunning();}
  NodeStatus onRunning() override
  {
    setOutput("people", static_cast<int>(ctx_->objects(need<std::string>(*this, "objects_topic"),
      ctx_->param<std::string>("classes.person"), need<double>(*this, "min_score")).size()));
    return NodeStatus::RUNNING;
  }
  void onHalted() override {}
};

}  // namespace

void register_hardware_nodes(BT::BehaviorTreeFactory & f, const std::shared_ptr<RosContext> & ctx)
{
  f.registerNodeType<AimNozzle>("AimNozzle", ctx);
  f.registerNodeType<SprayWater>("SprayWater", ctx);
  f.registerNodeType<StopPump>("StopPump", ctx);
  f.registerNodeType<PlayEvacuationGuidance>("PlayEvacuationGuidance", ctx);
  f.registerNodeType<TrackPeopleRadar>("TrackPeopleRadar", ctx);
  f.registerNodeType<TrackPeopleCamera>("TrackPeopleCamera", ctx);
}

}  // namespace limo_fire_bt
