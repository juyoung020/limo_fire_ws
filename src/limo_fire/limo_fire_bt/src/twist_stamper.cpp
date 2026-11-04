#include <geometry_msgs/msg/twist.hpp>
#include <geometry_msgs/msg/twist_stamped.hpp>
#include <rclcpp/rclcpp.hpp>
class Stamper : public rclcpp::Node {
  rclcpp::Publisher<geometry_msgs::msg::TwistStamped>::SharedPtr pub_;
  rclcpp::Subscription<geometry_msgs::msg::Twist>::SharedPtr sub_;

public:
  Stamper() : Node("sim_twist_stamper") {
    if (!get_parameter("use_sim_time").as_bool())
      throw std::runtime_error("simulation only");
    pub_ = create_publisher<geometry_msgs::msg::TwistStamped>(
        declare_parameter<std::string>("output_topic"), 10);
    sub_ = create_subscription<geometry_msgs::msg::Twist>(
        declare_parameter<std::string>("input_topic"), 10,
        [this](geometry_msgs::msg::Twist::ConstSharedPtr m) {
          geometry_msgs::msg::TwistStamped t;
          t.header.stamp = now();
          t.twist = *m;
          pub_->publish(t);
        });
  }
};
int main(int argc, char **argv) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<Stamper>());
  rclcpp::shutdown();
}
