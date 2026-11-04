#include "../src/danger_layer.cpp"
#include <cassert>
#include <thread>
int main(int argc, char **argv) {
  rclcpp::init(argc, argv);
  auto n = std::make_shared<rclcpp_lifecycle::LifecycleNode>("layer_test");
  auto source = std::make_shared<rclcpp::Node>("zone_source");
  tf2_ros::Buffer tf(n->get_clock());
  nav2_costmap_2d::LayeredCostmap map("map", false, false);
  map.resizeMap(100, 100, .1, -5, -5);
  limo_fire_bt::DangerLayer layer;
  auto group =
      n->create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
  layer.initialize(&map, "fire_layer", &tf, n, group);
  auto pub = source->create_publisher<limo_fire_msgs::msg::DangerZones>(
      "/fire/danger_zones", rclcpp::QoS(1).transient_local());
  uint64_t applied = 0;
  auto ack = source->create_subscription<std_msgs::msg::UInt64>(
      "/fire/global_applied", rclcpp::QoS(1).transient_local(),
      [&](std_msgs::msg::UInt64::ConstSharedPtr m) { applied = m->data; });
  limo_fire_msgs::msg::DangerZones zones;
  zones.header.frame_id = "map";
  zones.revision = 1;
  zones.centers.emplace_back();
  zones.radii.push_back(1.0);
  pub->publish(zones);
  rclcpp::executors::SingleThreadedExecutor ex;
  ex.add_node(n->get_node_base_interface());
  ex.add_node(source);
  for (int i = 0; i < 100 && applied != 1; ++i) {
    ex.spin_some();
    layer.updateCosts(*map.getCostmap(), 0, 0, 100, 100);
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
  assert(applied == 1);
  assert(map.getCostmap()->getCost(50, 50) == nav2_costmap_2d::LETHAL_OBSTACLE);
  assert(map.getCostmap()->getCost(80, 80) != nav2_costmap_2d::LETHAL_OBSTACLE);
  map.getCostmap()->resetMap(0, 0, 100, 100);
  layer.reset();
  layer.updateCosts(*map.getCostmap(), 0, 0, 100, 100);
  assert(!layer.isClearable());
  assert(map.getCostmap()->getCost(50, 50) == nav2_costmap_2d::LETHAL_OBSTACLE);
  rclcpp::shutdown();
}
