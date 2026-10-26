// YOLO 탐지 상자 + 정렬된 깊이 영상 → 물체의 지도(map) 좌표 → RViz 마커
// 같은 클래스가 merge_radius 안에서 다시 보이면 같은 물체로 보고 위치를 평균한다. min_hits 번 이상 본 것만 지도에 올린다.
// 깊이는 컬러 카메라에 정렬돼 있어야 한다 (실물 Orbbec align_mode HW, Webots 는 원래 같은 프레임)
#include <algorithm>
#include <cmath>
#include <memory>
#include <string>
#include <vector>

#include <message_filters/subscriber.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <message_filters/synchronizer.h>
#include <geometry_msgs/msg/point_stamped.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>
#include <vision_msgs/msg/detection2_d_array.hpp>
#include <visualization_msgs/msg/marker_array.hpp>

namespace limo_fire_perception
{

class ObjectMapperNode : public rclcpp::Node
{
public:
  ObjectMapperNode()
  : Node("object_mapper"), tf_buffer_(get_clock()), tf_listener_(tf_buffer_)
  {
    map_frame_ = declare_parameter<std::string>("map_frame", "map");
    min_depth_ = declare_parameter<double>("min_depth", 0.2);
    max_depth_ = declare_parameter<double>("max_depth", 5.0);
    box_shrink_ = declare_parameter<double>("box_shrink", 0.5);
    depth_percentile_ = declare_parameter<double>("depth_percentile", 0.3);
    merge_radius_ = declare_parameter<double>("merge_radius", 0.8);
    min_hits_ = declare_parameter<int>("min_hits", 3);
    min_score_ = declare_parameter<double>("min_score", 0.5);
    class_names_ = declare_parameter<std::vector<std::string>>("class_names");

    marker_pub_ = create_publisher<visualization_msgs::msg::MarkerArray>(
      "~/markers", rclcpp::QoS(1).transient_local());
    info_sub_ = create_subscription<sensor_msgs::msg::CameraInfo>(
      declare_parameter<std::string>("camera_info_topic", "/camera/depth/camera_info"), rclcpp::SensorDataQoS(),
      [this](sensor_msgs::msg::CameraInfo::ConstSharedPtr msg) {info_ = msg;});

    // 탐지와 깊이 영상은 시각이 정확히 같지 않으므로 가까운 것끼리 짝짓는다
    const auto qos = rclcpp::SensorDataQoS().get_rmw_qos_profile();
    det_sub_.subscribe(this, declare_parameter<std::string>("detections_topic", "/yolo_detector/detections"));
    depth_sub_.subscribe(this, declare_parameter<std::string>("depth_topic", "/camera/depth/image_raw"), qos);
    sync_ = std::make_shared<message_filters::Synchronizer<Policy>>(Policy(10), det_sub_, depth_sub_);
    // 회전 중에 다른 순간의 깊이와 짝지어지면 엉뚱한 곳에 찍히므로 시각 차이를 제한한다
    sync_->setMaxIntervalDuration(rclcpp::Duration::from_seconds(declare_parameter<double>("max_sync_dt", 0.05)));
    sync_->registerCallback(&ObjectMapperNode::on_pair, this);
  }

private:
  struct Landmark
  {
    int cls;
    double x, y, z;
    int hits;
    double best_score;
  };

  using Policy = message_filters::sync_policies::ApproximateTime<
    vision_msgs::msg::Detection2DArray, sensor_msgs::msg::Image>;

  int class_index(const std::string & name) const
  {
    const auto it = std::find(class_names_.begin(), class_names_.end(), name);
    return it == class_names_.end() ? -1 : static_cast<int>(it - class_names_.begin());
  }

  // 상자 가운데 부분 깊이의 하위 depth_percentile 값 (m). 얇은 물체 뒤의 벽이 섞여도 앞쪽(물체)을 고른다. 유효한 값이 없으면 NaN
  double box_depth(const sensor_msgs::msg::Image & depth, double cx, double cy, double w, double h) const
  {
    const int x0 = std::max(0, static_cast<int>(cx - w * box_shrink_ / 2));
    const int x1 = std::min(static_cast<int>(depth.width) - 1, static_cast<int>(cx + w * box_shrink_ / 2));
    const int y0 = std::max(0, static_cast<int>(cy - h * box_shrink_ / 2));
    const int y1 = std::min(static_cast<int>(depth.height) - 1, static_cast<int>(cy + h * box_shrink_ / 2));
    std::vector<float> vals;
    for (int y = y0; y <= y1; ++y) {
      const uint8_t * row = depth.data.data() + static_cast<size_t>(y) * depth.step;
      for (int x = x0; x <= x1; ++x) {
        float d;
        if (depth.encoding == "16UC1") {
          d = reinterpret_cast<const uint16_t *>(row)[x] * 0.001f;
        } else {
          d = reinterpret_cast<const float *>(row)[x];
        }
        if (std::isfinite(d) && d >= min_depth_ && d <= max_depth_) {vals.push_back(d);}
      }
    }
    if (vals.size() < 5) {return std::nan("");}
    const auto k = static_cast<size_t>(depth_percentile_ * (vals.size() - 1));
    std::nth_element(vals.begin(), vals.begin() + k, vals.end());
    return vals[k];
  }

  void on_pair(
    const vision_msgs::msg::Detection2DArray::ConstSharedPtr & dets,
    const sensor_msgs::msg::Image::ConstSharedPtr & depth)
  {
    if (!info_ || dets->detections.empty()) {return;}
    if (depth->encoding != "16UC1" && depth->encoding != "32FC1") {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 5000, "지원하지 않는 깊이 형식: %s", depth->encoding.c_str());
      return;
    }
    // 탐지 좌표는 컬러 영상 기준. 깊이 해상도가 다르면 비율로 맞춘다
    const double sx = static_cast<double>(depth->width) / info_->width;
    const double sy = static_cast<double>(depth->height) / info_->height;
    const double fx = info_->k[0], fy = info_->k[4], cx = info_->k[2], cy = info_->k[5];

    geometry_msgs::msg::TransformStamped tf;
    try {
      tf = tf_buffer_.lookupTransform(map_frame_, depth->header.frame_id, depth->header.stamp,
          rclcpp::Duration::from_seconds(0.2));
    } catch (const tf2::TransformException & e) {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 5000, "TF 없음: %s", e.what());
      return;
    }

    bool changed = false;
    for (const auto & d : dets->detections) {
      if (d.results.empty() || d.results[0].hypothesis.score < min_score_) {continue;}
      const int cls = class_index(d.results[0].hypothesis.class_id);
      if (cls < 0) {continue;}
      const double u = d.bbox.center.position.x, v = d.bbox.center.position.y;
      const double z = box_depth(*depth, u * sx, v * sy, d.bbox.size_x * sx, d.bbox.size_y * sy);
      if (!std::isfinite(z)) {continue;}

      geometry_msgs::msg::PointStamped p_cam, p_map;
      p_cam.point.x = (u - cx) * z / fx;
      p_cam.point.y = (v - cy) * z / fy;
      p_cam.point.z = z;
      tf2::doTransform(p_cam, p_map, tf);
      add(cls, p_map.point.x, p_map.point.y, p_map.point.z, d.results[0].hypothesis.score);
      changed = true;
    }
    if (changed) {publish();}
  }

  void add(int cls, double x, double y, double z, double score)
  {
    Landmark * best = nullptr;
    double best_d = merge_radius_;
    for (auto & l : landmarks_) {
      const double dist = std::hypot(l.x - x, l.y - y);
      if (l.cls == cls && dist < best_d) {best = &l; best_d = dist;}
    }
    if (!best) {
      landmarks_.push_back({cls, x, y, z, 1, score});
      return;
    }
    // 처음 몇 번은 평균, 그 뒤로는 천천히 따라간다 (멀리서 본 부정확한 값이 남지 않게)
    const double a = 1.0 / std::min(best->hits + 1, 20);
    best->x += a * (x - best->x);
    best->y += a * (y - best->y);
    best->z += a * (z - best->z);
    best->hits++;
    best->best_score = std::max(best->best_score, score);
  }

  void publish()
  {
    // 클래스마다 색 (RGB): 불 주황 · 연기 회색 · 소화기 빨강 · 유도등 초록 · 사람 파랑 · 문 갈색 · 계단 보라
    static const float rgb[7][3] = {{1.0f, 0.43f, 0.0f}, {0.55f, 0.55f, 0.55f}, {0.82f, 0.12f, 0.12f},
      {0.08f, 0.63f, 0.27f}, {0.12f, 0.37f, 0.75f}, {0.63f, 0.32f, 0.18f}, {0.49f, 0.31f, 0.63f}};
    visualization_msgs::msg::MarkerArray arr;
    visualization_msgs::msg::Marker clear;
    clear.action = visualization_msgs::msg::Marker::DELETEALL;
    arr.markers.push_back(clear);
    int id = 0;
    for (const auto & l : landmarks_) {
      if (l.hits < min_hits_) {continue;}
      visualization_msgs::msg::Marker m;
      m.header.frame_id = map_frame_;
      m.header.stamp = now();
      m.ns = class_names_[l.cls];
      m.id = id++;
      m.type = visualization_msgs::msg::Marker::CYLINDER;
      m.pose.position.x = l.x;
      m.pose.position.y = l.y;
      m.pose.position.z = 0.15;
      m.pose.orientation.w = 1.0;
      m.scale.x = m.scale.y = 0.3;
      m.scale.z = 0.3;
      const auto & c = rgb[l.cls % 7];
      m.color.r = c[0];
      m.color.g = c[1];
      m.color.b = c[2];
      m.color.a = 0.9f;
      arr.markers.push_back(m);

      m.ns = "label";
      m.type = visualization_msgs::msg::Marker::TEXT_VIEW_FACING;
      m.pose.position.z = 0.55;
      m.scale.z = 0.25;
      m.color.r = m.color.g = m.color.b = 1.0f;
      m.color.a = 1.0f;
      m.text = class_names_[l.cls];
      arr.markers.push_back(m);
      RCLCPP_INFO_ONCE(get_logger(), "지도에 첫 물체: %s (%.2f, %.2f)", class_names_[l.cls].c_str(), l.x, l.y);
    }
    marker_pub_->publish(arr);
  }

  tf2_ros::Buffer tf_buffer_;
  tf2_ros::TransformListener tf_listener_;
  std::string map_frame_;
  double min_depth_, max_depth_, box_shrink_, depth_percentile_, merge_radius_, min_score_;
  int min_hits_;
  std::vector<std::string> class_names_;
  std::vector<Landmark> landmarks_;
  sensor_msgs::msg::CameraInfo::ConstSharedPtr info_;

  message_filters::Subscriber<vision_msgs::msg::Detection2DArray> det_sub_;
  message_filters::Subscriber<sensor_msgs::msg::Image> depth_sub_;
  std::shared_ptr<message_filters::Synchronizer<Policy>> sync_;
  rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr info_sub_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr marker_pub_;
};

}  // namespace limo_fire_perception

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<limo_fire_perception::ObjectMapperNode>());
  rclcpp::shutdown();
  return 0;
}
