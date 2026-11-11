// 비동기 ROS 호출을 BT 틱에 맞게 감싼 것. 틱마다 poll() 로 상태만 본다 (스핀은 mission_node 의 주 루프가 한다)
#pragma once

#include <memory>
#include <string>

#include <nav2_msgs/action/navigate_to_pose.hpp>
#include <rclcpp/rclcpp.hpp>
#include <rclcpp_action/rclcpp_action.hpp>
#include <std_srvs/srv/set_bool.hpp>

#include "limo_fire_bt/ros_context.hpp"

namespace limo_fire_bt
{

enum class CallState { Idle, Waiting, Active, Succeeded, Failed };

// Nav2 navigate_to_pose 목표 하나. 서버가 server_timeout_s 안에 안 뜨거나, 거절·실패하면 Failed
class NavGoal
{
public:
  using Action = nav2_msgs::action::NavigateToPose;

  NavGoal(RosContext & ctx, const std::string & action_name);

  void send(const Pose2 & goal);
  CallState poll();
  void cancel();
  const std::string & error() const {return error_;}

  // 서버의 모든 목표 취소 (누가 보냈든). StopRobot 이 쓴다
  void cancel_all();

private:
  void dispatch();

  RosContext & ctx_;
  std::string name_;
  rclcpp_action::Client<Action>::SharedPtr client_;
  std::shared_ptr<CallState> state_;
  std::shared_ptr<rclcpp_action::ClientGoalHandle<Action>::SharedPtr> handle_slot_;
  Pose2 goal_;
  rclcpp::Time asked_;
  std::string error_;
};

// std_srvs/SetBool 호출 하나. 서비스가 server_timeout_s 안에 없거나 success=false 면 Failed
class SetBoolCall
{
public:
  SetBoolCall(RosContext & ctx, const std::string & service);

  void send(bool value);
  CallState poll();
  const std::string & error() const {return error_;}
  // 응답을 기다리지 않고 보낸다 (멈춤·끄기용). 서비스가 없으면 false
  bool fire_and_forget(bool value);

private:
  RosContext & ctx_;
  std::string name_;
  rclcpp::Client<std_srvs::srv::SetBool>::SharedPtr client_;
  std::shared_ptr<CallState> state_;
  std::shared_ptr<std::string> message_;
  bool value_ = false;
  rclcpp::Time asked_;
  std::string error_;
};

}  // namespace limo_fire_bt
