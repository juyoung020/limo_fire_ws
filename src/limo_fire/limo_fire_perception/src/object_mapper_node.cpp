// YOLO 탐지 상자 + 정렬된 깊이 영상 → 물체의 지도(map) 좌표 → RViz 마커 + 한글 현황판(오버레이)
// 같은 클래스가 merge_radius 안에서 다시 보이면 같은 물체로 보고 위치를 평균한다. min_hits 번 이상 본 것만 지도에 올린다.
// 깊이는 컬러 카메라에 정렬돼 있어야 한다 (실물 Orbbec align_mode HW, Webots 는 원래 같은 프레임)
#include <algorithm>
#include <array>
#include <cmath>
#include <memory>
#include <string>
#include <vector>

#include <message_filters/subscriber.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <message_filters/synchronizer.h>
#include <geometry_msgs/msg/point_stamped.hpp>
#include <rclcpp/rclcpp.hpp>
#include <rviz_2d_overlay_msgs/msg/overlay_text.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <std_srvs/srv/trigger.hpp>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>
#include <vision_msgs/msg/detection2_d_array.hpp>
#include <visualization_msgs/msg/marker_array.hpp>

namespace limo_fire_perception
{

namespace
{
using visualization_msgs::msg::Marker;

// 클래스마다 지도 표시: 지도 위 짧은 이름(RViz 글꼴은 영문만), 현황판 한글 이름, 색
struct Style
{
  const char * name;   // class_names 와 같은 이름
  const char * tag;    // 지도 라벨
  const char * ko;     // 현황판
  float r, g, b;
};
constexpr std::array<Style, 7> kStyles{{
  {"fire", "FIRE", "불", 1.00f, 0.36f, 0.10f},
  {"smoke", "SMOKE", "연기", 0.62f, 0.62f, 0.66f},
  {"fire_extinguisher", "EXT", "소화기", 0.90f, 0.12f, 0.16f},
  {"exit_sign", "EXIT", "비상구 유도등", 0.10f, 0.78f, 0.36f},
  {"person", "PERSON", "사람", 0.20f, 0.50f, 1.00f},
  {"door", "DOOR", "문", 0.72f, 0.48f, 0.26f},
  {"stairs", "STAIRS", "계단", 0.66f, 0.42f, 0.92f},
}};
constexpr Style kUnknown{"", "OBJ", "기타", 0.8f, 0.8f, 0.8f};

const Style & style_of(const std::string & name)
{
  for (const auto & s : kStyles) {
    if (name == s.name) {return s;}
  }
  return kUnknown;
}

std_msgs::msg::ColorRGBA rgba(float r, float g, float b, float a)
{
  std_msgs::msg::ColorRGBA c;
  c.r = r;
  c.g = g;
  c.b = b;
  c.a = a;
  return c;
}

std::string hex(const Style & s)
{
  char buf[8];
  std::snprintf(buf, sizeof(buf), "#%02x%02x%02x", static_cast<int>(s.r * 255), static_cast<int>(s.g * 255),
    static_cast<int>(s.b * 255));
  return buf;
}
}  // namespace

class ObjectMapperNode : public rclcpp::Node
{
public:
  ObjectMapperNode()
  : Node("object_mapper"), tf_buffer_(get_clock()), tf_listener_(tf_buffer_)
  {
    map_frame_ = declare_parameter<std::string>("map_frame", "map");
    robot_frame_ = declare_parameter<std::string>("robot_frame", "base_link");
    min_depth_ = declare_parameter<double>("min_depth", 0.2);
    max_depth_ = declare_parameter<double>("max_depth", 5.0);
    box_shrink_ = declare_parameter<double>("box_shrink", 0.5);
    depth_percentile_ = declare_parameter<double>("depth_percentile", 0.3);
    merge_radius_ = declare_parameter<double>("merge_radius", 0.8);
    min_hits_ = declare_parameter<int>("min_hits", 3);
    min_score_ = declare_parameter<double>("min_score", 0.5);
    fire_zone_radius_ = declare_parameter<double>("fire_zone_radius", 1.0);
    label_offset_ = declare_parameter<std::vector<double>>("label_offset", {-0.55, 0.0});
    icon_size_ = declare_parameter<double>("icon_size", 0.6);
    icon_yaw_ = declare_parameter<double>("icon_yaw", M_PI / 2);
    label_offset_.resize(2, 0.0);
    class_names_ = declare_parameter<std::vector<std::string>>("class_names");
    // 클래스별 병합 반경 (class_names 순서). 계단·문처럼 큰 물체는 넓게. 비우면 모두 merge_radius
    merge_radii_ = declare_parameter<std::vector<double>>("merge_radii", std::vector<double>{});
    merge_radii_.resize(class_names_.size(), merge_radius_);

    marker_pub_ = create_publisher<visualization_msgs::msg::MarkerArray>(
      "~/markers", rclcpp::QoS(1).transient_local());
    overlay_pub_ = create_publisher<rviz_2d_overlay_msgs::msg::OverlayText>(
      "~/summary", rclcpp::QoS(1).transient_local());
    info_sub_ = create_subscription<sensor_msgs::msg::CameraInfo>(
      declare_parameter<std::string>("camera_info_topic", "/camera/depth/camera_info"), rclcpp::SensorDataQoS(),
      [this](sensor_msgs::msg::CameraInfo::ConstSharedPtr msg) {info_ = msg;});
    // 지도에 올린 물체를 모두 지운다: ros2 service call /object_mapper/clear std_srvs/srv/Trigger
    clear_srv_ = create_service<std_srvs::srv::Trigger>("~/clear",
        [this](std_srvs::srv::Trigger::Request::SharedPtr, std_srvs::srv::Trigger::Response::SharedPtr res) {
          landmarks_.clear();
          counters_.fill(0);
          visualization_msgs::msg::MarkerArray arr;
          arr.markers.emplace_back().action = Marker::DELETEALL;
          marker_pub_->publish(arr);
          res->success = true;
          res->message = "지도 물체를 모두 지웠음";
        });

    // 탐지와 깊이 영상은 시각이 정확히 같지 않으므로 가까운 것끼리 짝짓는다
    const auto qos = rclcpp::SensorDataQoS().get_rmw_qos_profile();
    det_sub_.subscribe(this, declare_parameter<std::string>("detections_topic", "/yolo_detector/detections"));
    depth_sub_.subscribe(this, declare_parameter<std::string>("depth_topic", "/camera/depth/image_raw"), qos);
    sync_ = std::make_shared<message_filters::Synchronizer<Policy>>(Policy(10), det_sub_, depth_sub_);
    // 회전 중에 다른 순간의 깊이와 짝지어지면 엉뚱한 곳에 찍히므로 시각 차이를 제한한다
    sync_->setMaxIntervalDuration(rclcpp::Duration::from_seconds(declare_parameter<double>("max_sync_dt", 0.05)));
    sync_->registerCallback(&ObjectMapperNode::on_pair, this);

    // 불 위험 반경 깜빡임·로봇까지 거리 갱신 (벽시계 기준이라 시뮬이 멈춰도 표시는 돈다)
    timer_ = create_wall_timer(std::chrono::milliseconds(100), [this]() {publish();});
    publish_summary();
  }

private:
  struct Landmark
  {
    int cls;
    int number;      // 클래스 안 번호 (EXT 1, EXT 2 …), 지도 올라갈 때 매김
    double x, y, z;
    double yaw;      // 처음 본 방향 (문·유도등·계단 판을 카메라 쪽으로 세움)
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
      const double view_yaw = std::atan2(p_map.point.y - tf.transform.translation.y,
          p_map.point.x - tf.transform.translation.x);
      add(cls, p_map.point.x, p_map.point.y, p_map.point.z, view_yaw, d.results[0].hypothesis.score);
    }
  }

  void add(int cls, double x, double y, double z, double yaw, double score)
  {
    Landmark * best = nullptr;
    double best_d = merge_radii_[cls];
    for (auto & l : landmarks_) {
      const double dist = std::hypot(l.x - x, l.y - y);
      if (l.cls == cls && dist < best_d) {best = &l; best_d = dist;}
    }
    if (!best) {
      landmarks_.push_back({cls, 0, x, y, z, yaw, 1, score});
      return;
    }
    // 처음 몇 번은 평균, 그 뒤로는 천천히 따라간다 (멀리서 본 부정확한 값이 남지 않게)
    const double a = 1.0 / std::min(best->hits + 1, 20);
    best->x += a * (x - best->x);
    best->y += a * (y - best->y);
    best->z += a * (z - best->z);
    best->hits++;
    best->best_score = std::max(best->best_score, score);
    if (best->hits >= min_hits_ && best->number == 0) {
      best->number = ++counters_[std::min<size_t>(cls, counters_.size() - 1)];
      RCLCPP_INFO(get_logger(), "지도에 추가: %s %d (%.2f, %.2f)", style_of(class_names_[cls]).tag, best->number,
        best->x, best->y);
    }
  }

  // 물체 하나 = 바닥 원판 + 종류별 3D 모양 + 라벨. id 는 물체 번호 * 10 + 부품 번호로 고정해 깜빡임 없이 덮어쓴다
  void append_markers(const Landmark & l, size_t index, double t, visualization_msgs::msg::MarkerArray & arr) const
  {
    const Style & s = style_of(class_names_[l.cls]);
    const int base_id = static_cast<int>(index) * 10;
    auto make = [&](int part, int type, double z, double sx, double sy, double sz, std_msgs::msg::ColorRGBA c,
        double yaw = 0.0) {
        Marker m;
        m.header.frame_id = map_frame_;
        m.ns = s.name;
        m.id = base_id + part;
        m.type = type;
        m.pose.position.x = l.x;
        m.pose.position.y = l.y;
        m.pose.position.z = z;
        m.pose.orientation.z = std::sin(yaw / 2);
        m.pose.orientation.w = std::cos(yaw / 2);
        m.scale.x = sx;
        m.scale.y = sy;
        m.scale.z = sz;
        m.color = c;
        return m;
      };
    const bool is_fire = std::string(s.name) == "fire";
    const float pulse = 0.5f + 0.5f * static_cast<float>(std::sin(t * 4.0));

    if (is_fire) {
      // 위험 반경: 숨 쉬듯 진해졌다 옅어진다
      const double d = 2 * fire_zone_radius_;
      arr.markers.push_back(make(0, Marker::CYLINDER, 0.01, d, d, 0.02, rgba(1.0f, 0.15f, 0.05f, 0.10f + 0.16f * pulse)));
      arr.markers.push_back(make(1, Marker::CYLINDER, 0.02, d * 0.5, d * 0.5, 0.02, rgba(1.0f, 0.25f, 0.05f, 0.25f)));
    }
    // 아이콘 배지: icons/<클래스>.obj (납작한 원판 + 그림 텍스처). 그림 위쪽이 화면 위로 오게 icon_yaw 만큼 돌린다
    const double size = icon_size_ * (is_fire ? 1.0 + 0.1 * pulse : 1.0);
    Marker shadow = make(2, Marker::CYLINDER, 0.88, size * 1.08, size * 1.08, 0.01, rgba(0.0f, 0.0f, 0.0f, 0.35f));
    shadow.pose.position.x += 0.03 * std::sin(icon_yaw_);  // 화면 아래쪽으로 살짝 → 떠 있는 느낌
    shadow.pose.position.y -= 0.03 * std::cos(icon_yaw_);
    arr.markers.push_back(shadow);
    Marker icon = make(3, Marker::MESH_RESOURCE, 0.9, size, size, 1.0, rgba(0.0f, 0.0f, 0.0f, 0.0f), icon_yaw_);
    icon.mesh_resource = "package://limo_fire_perception/icons/" + std::string(s.name) + ".obj";
    icon.mesh_use_embedded_materials = true;  // 색 0 = 텍스처 그대로
    arr.markers.push_back(icon);

    // 라벨: 밝은 지도 위에서도 읽히게 어두운 바탕판(위에서 내려다볼 때 글자 뒤) + 클래스 색 테두리.
    // 아이콘을 가리지 않게 label_offset 만큼 비켜 둔다 (map 프레임 기준)
    const std::string text = std::string(s.tag) + "-" + std::to_string(l.number);  // RViz 글꼴은 띄어쓰기가 넓어 붙여 쓴다
    const double len = 0.15 * static_cast<double>(text.size()) + 0.12;
    auto shifted = [&](Marker m) {
        m.pose.position.x += label_offset_[0];
        m.pose.position.y += label_offset_[1];
        m.pose.orientation.z = 0.0;
        m.pose.orientation.w = 1.0;
        return m;
      };
    Marker edge = shifted(make(6, Marker::CUBE, 0.95, 0.36, len + 0.05, 0.01, rgba(s.r, s.g, s.b, 1.0f)));
    edge.ns = "labels";
    arr.markers.push_back(edge);
    Marker plate = shifted(make(7, Marker::CUBE, 0.97, 0.31, len, 0.01, rgba(0.08f, 0.09f, 0.11f, 0.92f)));
    plate.ns = "labels";
    arr.markers.push_back(plate);
    Marker label = shifted(make(5, Marker::TEXT_VIEW_FACING, 1.0, 0.0, 0.0, 0.24, rgba(1.0f, 1.0f, 1.0f, 1.0f)));
    label.ns = "labels";
    label.text = text;
    arr.markers.push_back(label);
  }

  void publish()
  {
    const double t = std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
    visualization_msgs::msg::MarkerArray arr;
    for (size_t i = 0; i < landmarks_.size(); ++i) {
      if (landmarks_[i].number > 0) {append_markers(landmarks_[i], i, t, arr);}
    }
    if (!arr.markers.empty()) {marker_pub_->publish(arr);}
    if (++tick_ % 5 == 0) {publish_summary();}
  }

  // RViz 왼쪽 위 현황판: 화재 경고 + 종류별 개수 + 로봇에서 가장 가까운 것까지 거리
  void publish_summary()
  {
    double rx = 0, ry = 0;
    bool have_robot = false;
    try {
      const auto tf = tf_buffer_.lookupTransform(map_frame_, robot_frame_, tf2::TimePointZero);
      rx = tf.transform.translation.x;
      ry = tf.transform.translation.y;
      have_robot = true;
    } catch (const tf2::TransformException &) {
    }

    struct Row {int count = 0; double nearest = 1e9; int nearest_no = 0;};
    std::vector<Row> rows(class_names_.size());
    for (const auto & l : landmarks_) {
      if (l.number == 0) {continue;}
      Row & r = rows[l.cls];
      r.count++;
      const double d = std::hypot(l.x - rx, l.y - ry);
      if (d < r.nearest) {r.nearest = d; r.nearest_no = l.number;}
    }
    auto dist_text = [&](const Row & r, const Style & s) {
        if (!have_robot) {return std::string();}
        char buf[64];
        std::snprintf(buf, sizeof(buf), "%s-%d · %.1f m", s.tag, r.nearest_no, r.nearest);
        return std::string(buf);
      };

    std::string html;
    int lines = 1;
    const int fire = class_index("fire");
    if (fire >= 0 && rows[fire].count > 0) {
      html += "<div style='color:#ff6b4a;font-size:large;'><b>⚠ 화재 " + std::to_string(rows[fire].count) +
        "곳 감지</b></div><div style='color:#ffb4a2;'>가장 가까운 불 " + dist_text(rows[fire], style_of("fire")) +
        "</div>";
      lines += 2;
    }
    html += "<table cellspacing='0' cellpadding='3' width='100%'>";
    html += "<tr style='color:#9aa0a6;'><td>종류</td><td align='right'>개수</td><td align='right'>가장 가까운 것</td></tr>";
    for (size_t c = 0; c < class_names_.size(); ++c) {
      const Style & s = style_of(class_names_[c]);
      const Row & r = rows[c];
      const bool has = r.count > 0;
      html += std::string("<tr style='color:") + (has ? "#ffffff" : "#6b7075") + ";'>" +
        "<td><span style='color:" + hex(s) + ";'>■</span> " + s.ko + "</td>" +
        "<td align='right'><b>" + std::to_string(r.count) + "</b></td>" +
        "<td align='right'>" + (has ? dist_text(r, s) : std::string("-")) + "</td></tr>";
      lines++;
    }
    html += "</table><div style='color:#9aa0a6;font-size:small;'>" + std::to_string(min_hits_) +
      "번 이상 확인된 것만 · 거리는 로봇 기준 직선</div>";

    rviz_2d_overlay_msgs::msg::OverlayText o;
    o.action = rviz_2d_overlay_msgs::msg::OverlayText::ADD;
    o.horizontal_alignment = rviz_2d_overlay_msgs::msg::OverlayText::LEFT;  // 지도는 RViz 설정에서 오른쪽으로 비켜 둠
    o.vertical_alignment = rviz_2d_overlay_msgs::msg::OverlayText::TOP;
    o.horizontal_distance = 12;
    o.vertical_distance = 12;
    o.width = 400;
    o.height = 60 + 27 * lines;
    o.text_size = 11.0;
    o.font = "Noto Sans CJK KR";
    o.bg_color = rgba(0.09f, 0.10f, 0.12f, 0.85f);
    o.fg_color = rgba(1.0f, 1.0f, 1.0f, 1.0f);
    o.text = html;
    overlay_pub_->publish(o);
  }

  tf2_ros::Buffer tf_buffer_;
  tf2_ros::TransformListener tf_listener_;
  std::string map_frame_, robot_frame_;
  double min_depth_, max_depth_, box_shrink_, depth_percentile_, merge_radius_, min_score_, fire_zone_radius_;
  std::vector<double> merge_radii_;
  std::vector<double> label_offset_;
  double icon_size_, icon_yaw_;
  int min_hits_;
  std::vector<std::string> class_names_;
  std::vector<Landmark> landmarks_;
  std::array<int, 16> counters_{};
  unsigned tick_ = 0;
  sensor_msgs::msg::CameraInfo::ConstSharedPtr info_;

  message_filters::Subscriber<vision_msgs::msg::Detection2DArray> det_sub_;
  message_filters::Subscriber<sensor_msgs::msg::Image> depth_sub_;
  std::shared_ptr<message_filters::Synchronizer<Policy>> sync_;
  rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr info_sub_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr marker_pub_;
  rclcpp::Publisher<rviz_2d_overlay_msgs::msg::OverlayText>::SharedPtr overlay_pub_;
  rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr clear_srv_;
  rclcpp::TimerBase::SharedPtr timer_;
};

}  // namespace limo_fire_perception

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<limo_fire_perception::ObjectMapperNode>());
  rclcpp::shutdown();
  return 0;
}
