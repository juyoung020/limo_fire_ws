// Simulation-only evidence. This topic is never used as navigation
// localization.
#include <cmath>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <pluginlib/class_list_macros.hpp>
#include <unordered_map>
#include <webots/robot.h>
#include <webots/supervisor.h>
#include <webots_ros2_driver/PluginInterface.hpp>
#include <webots_ros2_driver/WebotsNode.hpp>
namespace limo_fire_sim {
class GroundTruthPose : public webots_ros2_driver::PluginInterface {
  webots_ros2_driver::WebotsNode *node_ = nullptr;
  rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr pub_;
  WbNodeRef robot_ = nullptr;
  double ox_ = 0, oy_ = 0, period_ = 0, next_ = 0;
  std::string frame_;

public:
  void init(webots_ros2_driver::WebotsNode *n,
            std::unordered_map<std::string, std::string> &p) override {
    if (!wb_robot_get_supervisor())
      return;
    node_ = n;
    robot_ = wb_supervisor_node_get_self();
    ox_ = std::stod(p.at("originX"));
    oy_ = std::stod(p.at("originY"));
    period_ = std::stod(p.at("periodMs")) * .001;
    frame_ = p.at("frameId");
    pub_ = n->create_publisher<geometry_msgs::msg::PoseStamped>(
        p.at("topicName"), 10);
  }
  void step() override {
    if (!robot_ || wb_robot_get_time() < next_)
      return;
    double t = wb_robot_get_time();
    next_ = t + period_;
    const auto p = wb_supervisor_node_get_position(robot_);
    const auto r = wb_supervisor_node_get_orientation(robot_);
    geometry_msgs::msg::PoseStamped m;
    m.header.frame_id = frame_;
    m.header.stamp.sec = int32_t(t);
    m.header.stamp.nanosec = uint32_t((t - int32_t(t)) * 1e9);
    m.pose.position.x = p[0] - ox_;
    m.pose.position.y = p[1] - oy_;
    m.pose.position.z = p[2];
    const auto yaw = std::atan2(r[3], r[0]);
    m.pose.orientation.z = std::sin(yaw / 2);
    m.pose.orientation.w = std::cos(yaw / 2);
    pub_->publish(m);
  }
};
} // namespace limo_fire_sim
PLUGINLIB_EXPORT_CLASS(limo_fire_sim::GroundTruthPose,
                       webots_ros2_driver::PluginInterface)
