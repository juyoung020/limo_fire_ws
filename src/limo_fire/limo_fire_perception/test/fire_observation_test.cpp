#include "../src/object_mapper_node.cpp"
#include <cassert>
#include <tf2_ros/static_transform_broadcaster.h>
#include <thread>
int main(int argc, char **argv) {
  rclcpp::init(argc, argv);
  rclcpp::NodeOptions opt;
  opt.parameter_overrides(
      {rclcpp::Parameter("class_names", std::vector<std::string>{"fire"}),
       rclcpp::Parameter("min_hits", 1),
       rclcpp::Parameter("missing_depth_timeout_s", .1)});
  auto mapper = std::make_shared<limo_fire_perception::ObjectMapperNode>(opt);
  auto n = std::make_shared<rclcpp::Node>("mapper_fixture");
  auto det = n->create_publisher<vision_msgs::msg::Detection2DArray>(
      "/yolo_detector/detections", 10);
  auto depth = n->create_publisher<sensor_msgs::msg::Image>(
      "/camera/depth/image_raw", 10);
  auto mask =
      n->create_publisher<sensor_msgs::msg::Image>("/yolo_detector/mask", 10);
  auto info = n->create_publisher<sensor_msgs::msg::CameraInfo>(
      "/camera/depth/camera_info", 10);
  bool valid = false, missing = false;
  auto obs = n->create_subscription<limo_fire_msgs::msg::FireObservation>(
      "/fire/observations", 10,
      [&](limo_fire_msgs::msg::FireObservation::ConstSharedPtr f) {
        if (f->size_valid) {
          assert(std::abs(f->width_m - .1) < .001);
          valid = true;
        } else if (!f->position_valid)
          missing = true;
      });
  tf2_ros::StaticTransformBroadcaster tf(n);
  geometry_msgs::msg::TransformStamped t;
  t.header.frame_id = "map";
  t.child_frame_id = "camera_optical";
  t.transform.rotation.w = 1;
  tf.sendTransform(t);
  rclcpp::executors::SingleThreadedExecutor ex;
  ex.add_node(n);
  ex.add_node(mapper);
  for (int i = 0; i < 180 && !missing; ++i) {
    sensor_msgs::msg::CameraInfo c;
    c.header.frame_id = "camera_optical";
    c.header.stamp = n->now();
    c.width = 64;
    c.height = 48;
    c.k[0] = 50;
    c.k[4] = 50;
    c.k[2] = 32;
    c.k[5] = 24;
    info->publish(c);
    vision_msgs::msg::Detection2DArray a;
    a.header = c.header;
    a.detections.emplace_back();
    auto &d = a.detections.back();
    d.bbox.center.position.x = 32;
    d.bbox.center.position.y = 24;
    d.bbox.size_x = 5;
    d.bbox.size_y = 5;
    d.results.emplace_back();
    d.results[0].hypothesis.class_id = "fire";
    d.results[0].hypothesis.score = .9;
    sensor_msgs::msg::Image m;
    m.header = c.header;
    m.width = 64;
    m.height = 48;
    m.step = 64;
    m.encoding = "mono8";
    m.data.assign(64 * 48, 1);
    mask->publish(m);
    // Publish masks before detections, as in the real detector.
    ex.spin_some();
    if (i < 70) {
      sensor_msgs::msg::Image z;
      z.header = c.header;
      z.width = 64;
      z.height = 48;
      z.step = 128;
      z.encoding = "16UC1";
      z.data.resize(128 * 48);
      for (size_t k = 0; k < z.data.size(); k += 2) {
        z.data[k] = 0xe8;
        z.data[k + 1] = 3;
      }
      depth->publish(z);
    }
    det->publish(a);
    ex.spin_some();
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
  assert(valid);
  assert(missing);
  rclcpp::shutdown();
}
