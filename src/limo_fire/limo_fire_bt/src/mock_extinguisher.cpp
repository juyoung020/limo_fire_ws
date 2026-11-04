#include <limo_fire_msgs/action/extinguish.hpp>
#include <rclcpp/rclcpp.hpp>
#include <rclcpp_action/rclcpp_action.hpp>
class Mock : public rclcpp::Node {
  using A = limo_fire_msgs::action::Extinguish;
  using H = rclcpp_action::ServerGoalHandle<A>;
  rclcpp_action::Server<A>::SharedPtr server_;
  std::shared_ptr<H> active_;
  rclcpp::TimerBase::SharedPtr timer_;
  rclcpp::Time started_{0, 0, RCL_ROS_TIME};
  std::string outcome_;
  double delay_;

public:
  Mock() : Node("mock_extinguisher") {
    if (!get_parameter("use_sim_time").as_bool())
      throw std::runtime_error("simulation only");
    outcome_ = declare_parameter<std::string>("outcome");
    delay_ = declare_parameter<double>("delay_s");
    server_ = rclcpp_action::create_server<A>(
        this, declare_parameter<std::string>("action"),
        [this](auto, auto) {
          return active_ ? rclcpp_action::GoalResponse::REJECT
                         : rclcpp_action::GoalResponse::ACCEPT_AND_EXECUTE;
        },
        [](auto) { return rclcpp_action::CancelResponse::ACCEPT; },
        [this](auto h) {
          active_ = h;
          started_ = now();
        });
    timer_ = create_wall_timer(std::chrono::duration<double>(declare_parameter<double>("tick_s")), [this] {
      if (!active_)
        return;
      auto r = std::make_shared<A::Result>();
      if (active_->is_canceling()) {
        r->reason = "canceled";
        active_->canceled(r);
        active_.reset();
        return;
      }
      if (outcome_ == "timeout" || (now() - started_).seconds() < delay_)
        return;
      r->success = outcome_ == "success";
      r->reason = "simulated " + outcome_;
      if (r->success)
        active_->succeed(r);
      else
        active_->abort(r);
      active_.reset();
    });
  }
};
int main(int argc, char **argv) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<Mock>());
  rclcpp::shutdown();
}
