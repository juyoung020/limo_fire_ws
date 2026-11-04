#include <limits>
#include <limo_fire_msgs/msg/danger_zones.hpp>
#include <mutex>
#include <nav2_costmap_2d/layer.hpp>
#include <pluginlib/class_list_macros.hpp>
#include <std_msgs/msg/u_int64.hpp>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
namespace limo_fire_bt {
class DangerLayer : public nav2_costmap_2d::Layer {
  std::mutex mutex_;
  limo_fire_msgs::msg::DangerZones zones_;
  rclcpp::Subscription<limo_fire_msgs::msg::DangerZones>::SharedPtr sub_;
  rclcpp_lifecycle::LifecyclePublisher<std_msgs::msg::UInt64>::SharedPtr ack_;

public:
  void onInitialize() override {
    auto n = node_.lock();
    declareParameter("topic", rclcpp::ParameterValue("/fire/danger_zones"));
    declareParameter("ack_topic",
                     rclcpp::ParameterValue("/fire/global_applied"));
    std::string topic, ack;
    n->get_parameter(name_ + ".topic", topic);
    n->get_parameter(name_ + ".ack_topic", ack);
    ack_ = n->create_publisher<std_msgs::msg::UInt64>(
        ack, rclcpp::QoS(1).transient_local());
    ack_->on_activate();
    sub_ = n->create_subscription<limo_fire_msgs::msg::DangerZones>(
        topic, rclcpp::QoS(1).transient_local(),
        [this](limo_fire_msgs::msg::DangerZones::ConstSharedPtr m) {
          if (m->centers.size() != m->radii.size() ||
              m->header.frame_id.empty())
            return;
          for (size_t i = 0; i < m->radii.size(); ++i)
            if (!std::isfinite(m->centers[i].x + m->centers[i].y +
                               m->radii[i]) ||
                m->radii[i] <= 0)
              return;
          std::lock_guard<std::mutex> lock(mutex_);
          zones_ = *m;
        });
    enabled_ = true;
    current_ = true;
  }
  void reset() override {
  } // hazard persists through Nav2 clear-costmap recovery
  bool isClearable() override { return false; }
  void updateBounds(double, double, double, double *x0, double *y0, double *x1,
                    double *y1) override {
    // Full bounds also ensure rolling windows reapply persistent zones.
    auto c = layered_costmap_->getCostmap();
    *x0 = std::min(*x0, c->getOriginX());
    *y0 = std::min(*y0, c->getOriginY());
    *x1 = std::max(*x1, c->getOriginX() + c->getSizeInMetersX());
    *y1 = std::max(*y1, c->getOriginY() + c->getSizeInMetersY());
  }
  void updateCosts(nav2_costmap_2d::Costmap2D &c, int x0, int y0, int x1,
                   int y1) override {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!zones_.revision)
      return;
    std::vector<geometry_msgs::msg::Point> centers;
    try {
      const auto frame = layered_costmap_->getGlobalFrameID();
      for (auto p : zones_.centers) {
        geometry_msgs::msg::PointStamped in, out;
        in.header = zones_.header;
        in.header.stamp = builtin_interfaces::msg::Time();
        in.point = p;
        out = tf_->transform(in, frame, tf2::durationFromSec(0.05));
        centers.push_back(out.point);
      }
    } catch (const tf2::TransformException &) {
      current_ = false;
      return;
    }
    for (int y = y0; y < y1; ++y)
      for (int x = x0; x < x1; ++x) {
        double wx, wy;
        c.mapToWorld(x, y, wx, wy);
        for (size_t i = 0; i < centers.size(); ++i) {
          // Intersecting cells, not only cell centers, are blocked.
          if (std::hypot(wx - centers[i].x, wy - centers[i].y) <=
              zones_.radii[i] + c.getResolution() * std::sqrt(0.5)) {
            c.setCost(x, y, nav2_costmap_2d::LETHAL_OBSTACLE);
            break;
          }
        }
      }
    current_ = true;
    std_msgs::msg::UInt64 a;
    a.data = zones_.revision;
    ack_->publish(a);
  }
};
} // namespace limo_fire_bt
PLUGINLIB_EXPORT_CLASS(limo_fire_bt::DangerLayer, nav2_costmap_2d::Layer)
