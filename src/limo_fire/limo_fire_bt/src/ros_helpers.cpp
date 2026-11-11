#include "limo_fire_bt/ros_helpers.hpp"

#include <cmath>

namespace limo_fire_bt
{

NavGoal::NavGoal(RosContext & ctx, const std::string & action_name)
: ctx_(ctx), name_(ctx.resolve(action_name)), state_(std::make_shared<CallState>(CallState::Idle)),
  handle_slot_(std::make_shared<rclcpp_action::ClientGoalHandle<Action>::SharedPtr>())
{
  client_ = rclcpp_action::create_client<Action>(ctx_.node(), name_);
}

void NavGoal::send(const Pose2 & goal)
{
  cancel();
  goal_ = goal;
  asked_ = ctx_.now();
  error_.clear();
  state_ = std::make_shared<CallState>(CallState::Waiting);   // 앞 목표의 늦은 콜백이 덮어쓰지 않게 새 칸
  dispatch();
}

void NavGoal::dispatch()
{
  if (!client_->action_server_is_ready()) {return;}   // poll() 에서 다시
  Action::Goal g;
  g.pose.header.frame_id = ctx_.param<std::string>("map_frame");
  g.pose.header.stamp = ctx_.now();
  g.pose.pose.position.x = goal_.x;
  g.pose.pose.position.y = goal_.y;
  g.pose.pose.orientation.z = std::sin(goal_.yaw / 2);
  g.pose.pose.orientation.w = std::cos(goal_.yaw / 2);

  // 이 목표의 콜백만 상태를 바꾸게: 새 목표를 보내면 새 상태 칸을 쓴다
  auto state = std::make_shared<CallState>(CallState::Active);
  state_ = state;
  auto slot = std::make_shared<rclcpp_action::ClientGoalHandle<Action>::SharedPtr>();
  handle_slot_ = slot;
  rclcpp_action::Client<Action>::SendGoalOptions opt;
  opt.goal_response_callback = [state, slot](rclcpp_action::ClientGoalHandle<Action>::SharedPtr h) {
      if (!h) {*state = CallState::Failed;} else {*slot = h;}
    };
  opt.result_callback = [state](const rclcpp_action::ClientGoalHandle<Action>::WrappedResult & r) {
      *state = r.code == rclcpp_action::ResultCode::SUCCEEDED ? CallState::Succeeded : CallState::Failed;
    };
  client_->async_send_goal(g, opt);
}

CallState NavGoal::poll()
{
  if (*state_ == CallState::Waiting) {
    if ((ctx_.now() - asked_).seconds() > ctx_.param<double>("server_timeout_s")) {
      error_ = name_ + " 액션 서버 없음";
      *state_ = CallState::Failed;
    } else {
      dispatch();
    }
  }
  if (*state_ == CallState::Failed && error_.empty()) {error_ = name_ + " 목표 실패·거절";}
  return *state_;
}

void NavGoal::cancel()
{
  if (*state_ == CallState::Active && *handle_slot_) {client_->async_cancel_goal(*handle_slot_);}
  if (*state_ == CallState::Active || *state_ == CallState::Waiting) {*state_ = CallState::Idle;}
}

void NavGoal::cancel_all()
{
  cancel();
  if (client_->action_server_is_ready()) {client_->async_cancel_all_goals();}
}

SetBoolCall::SetBoolCall(RosContext & ctx, const std::string & service)
: ctx_(ctx), name_(ctx.resolve(service)), state_(std::make_shared<CallState>(CallState::Idle)),
  message_(std::make_shared<std::string>())
{
  client_ = ctx_.node()->create_client<std_srvs::srv::SetBool>(name_);
}

void SetBoolCall::send(bool value)
{
  value_ = value;
  asked_ = ctx_.now();
  error_.clear();
  state_ = std::make_shared<CallState>(CallState::Waiting);
  poll();
}

CallState SetBoolCall::poll()
{
  if (*state_ != CallState::Waiting) {
    if (*state_ == CallState::Failed && error_.empty()) {error_ = name_ + ": " + *message_;}
    return *state_;
  }
  if (client_->service_is_ready()) {
    auto req = std::make_shared<std_srvs::srv::SetBool::Request>();
    req->data = value_;
    auto state = state_;
    auto msg = message_;
    *state = CallState::Active;
    client_->async_send_request(req, [state, msg](rclcpp::Client<std_srvs::srv::SetBool>::SharedFuture f) {
        const auto r = f.get();
        *msg = r->message;
        *state = r->success ? CallState::Succeeded : CallState::Failed;
      });
  } else if ((ctx_.now() - asked_).seconds() > ctx_.param<double>("server_timeout_s")) {
    error_ = name_ + " 서비스 없음";
    *state_ = CallState::Failed;
  }
  return *state_;
}

bool SetBoolCall::fire_and_forget(bool value)
{
  if (!client_->service_is_ready()) {return false;}
  auto req = std::make_shared<std_srvs::srv::SetBool::Request>();
  req->data = value;
  client_->async_send_request(req, [](rclcpp::Client<std_srvs::srv::SetBool>::SharedFuture) {});
  return true;
}

}  // namespace limo_fire_bt
