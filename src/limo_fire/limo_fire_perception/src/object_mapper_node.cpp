// YOLO 탐지 상자 + 정렬된 깊이 영상 → 물체의 지도(map) 좌표 → RViz 마커 + 한글 현황판(오버레이)
// 같은 클래스가 merge_radius 안에서 다시 보이면 같은 물체로 보고 위치를 평균한다. min_hits 번 이상 본 것만 지도에 올린다.
// 깊이는 컬러 카메라에 정렬돼 있어야 한다 (실물 Orbbec align_mode HW, Webots 는 원래 같은 프레임)
// 분할 모델이면 같은 시각의 물체 번호 영상(yolo_detector/mask)으로 물체 픽셀만 골라 위치·거리·(불) 넓이를 잰다. 없으면 상자로
// 물체마다 가장 크게·확실하게 찍힌 장면(탐지 결과 영상을 상자 둘레만 잘라낸 것)을 두고, RViz Publish Point 로 아이콘을 누르면
// 그 사진을 ~/selected 로 낸다 (선택한 물체는 흰 고리, 현황판에 정보)
#include <algorithm>
#include <array>
#include <cmath>
#include <deque>
#include <memory>
#include <string>
#include <vector>

#include <message_filters/subscriber.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <message_filters/synchronizer.h>
#include <geometry_msgs/msg/point_stamped.hpp>
#include <rclcpp/rclcpp.hpp>
#include <rclcpp_components/register_node_macro.hpp>
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
  explicit ObjectMapperNode(const rclcpp::NodeOptions & options)
  : Node("object_mapper", options), tf_buffer_(get_clock()), tf_listener_(tf_buffer_)
  {
    map_frame_ = declare_parameter<std::string>("map_frame", "map");
    robot_frame_ = declare_parameter<std::string>("robot_frame", "base_link");
    min_depth_ = declare_parameter<double>("min_depth", 0.2);
    max_depth_ = declare_parameter<double>("max_depth", 5.0);
    box_shrink_ = declare_parameter<double>("box_shrink", 0.5);
    depth_percentile_ = declare_parameter<double>("depth_percentile", 0.3);
    mask_depth_percentile_ = declare_parameter<double>("mask_depth_percentile", 0.5);
    merge_radius_ = declare_parameter<double>("merge_radius", 0.8);
    min_hits_ = declare_parameter<int>("min_hits", 3);
    min_score_ = declare_parameter<double>("min_score", 0.5);
    fire_zone_radius_ = declare_parameter<double>("fire_zone_radius", 1.0);
    label_offset_ = declare_parameter<std::vector<double>>("label_offset", {0.0, 0.55});
    icon_size_ = declare_parameter<double>("icon_size", 0.6);
    view_angle_ = declare_parameter<double>("view_angle", 0.0);
    label_offset_.resize(2, 0.0);
    class_names_ = declare_parameter<std::vector<std::string>>("class_names");
    // 클래스별 병합 반경 (class_names 순서). 계단·문처럼 큰 물체는 넓게. 비우면 모두 merge_radius
    merge_radii_ = declare_parameter<std::vector<double>>("merge_radii", std::vector<double>{});
    merge_radii_.resize(class_names_.size(), merge_radius_);
    view_margin_ = declare_parameter<double>("view_margin", 0.3);
    select_radius_ = declare_parameter<double>("select_radius", 0.6);

    // 늦게 켠 RViz 도 받게 transient_local. 프로세스 안 전달(intra-process)은 volatile 만 되므로 이 둘은 끈다
    rclcpp::PublisherOptions latched;
    latched.use_intra_process_comm = rclcpp::IntraProcessSetting::Disable;
    marker_pub_ = create_publisher<visualization_msgs::msg::MarkerArray>(
      "~/markers", rclcpp::QoS(1).transient_local(), latched);
    overlay_pub_ = create_publisher<rviz_2d_overlay_msgs::msg::OverlayText>(
      "~/summary", rclcpp::QoS(1).transient_local(), latched);
    info_sub_ = create_subscription<sensor_msgs::msg::CameraInfo>(
      declare_parameter<std::string>("camera_info_topic", "/camera/depth/camera_info"), rclcpp::SensorDataQoS(),
      [this](sensor_msgs::msg::CameraInfo::ConstSharedPtr msg) {info_ = msg;});
    // 마스크는 탐지와 시각이 같다. 최근 몇 장만 두고 탐지가 오면 같은 시각 것을 찾는다
    mask_sub_ = create_subscription<sensor_msgs::msg::Image>(
      declare_parameter<std::string>("mask_topic", "/yolo_detector/mask"), 10,
      [this](sensor_msgs::msg::Image::ConstSharedPtr msg) {
        masks_.push_back(msg);
        if (masks_.size() > 10) {masks_.pop_front();}
      });
    // 대표 사진: 탐지 결과 영상(마스크 덧칠·상자·이름). 탐지보다 늦게 올 수 있어 최근 몇 장을 두고 시각으로 맞춘다
    view_sub_ = create_subscription<sensor_msgs::msg::Image>(
      declare_parameter<std::string>("view_topic", "/yolo_detector/image"), 5,
      [this](sensor_msgs::msg::Image::ConstSharedPtr msg) {
        views_.push_back(msg);
        if (views_.size() > 10) {views_.pop_front();}
        take_views();
      });
    // RViz Publish Point 로 누른 곳에서 가장 가까운 물체를 고른다
    selected_pub_ = create_publisher<sensor_msgs::msg::Image>("~/selected", rclcpp::QoS(1).transient_local(), latched);
    click_sub_ = create_subscription<geometry_msgs::msg::PointStamped>(
      declare_parameter<std::string>("clicked_topic", "/clicked_point"), 10,
      [this](geometry_msgs::msg::PointStamped::ConstSharedPtr msg) {select(*msg);});

    // 지도에 올린 물체를 모두 지운다: ros2 service call /object_mapper/clear std_srvs/srv/Trigger
    clear_srv_ = create_service<std_srvs::srv::Trigger>("~/clear",
        [this](std_srvs::srv::Trigger::Request::SharedPtr, std_srvs::srv::Trigger::Response::SharedPtr res) {
          landmarks_.clear();
          counters_.fill(0);
          pending_.clear();
          selected_ = -1;
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
    double area;     // m², 마스크로 잰 앞면 넓이 (불 크기). 마스크가 없으면 0
    sensor_msgs::msg::Image::SharedPtr view;  // 대표 사진 (view_quality 가 가장 큰 장면)
    double view_quality = 0;
  };

  // 대표 사진을 잘라 올 장면: 결과 영상이 오면 같은 시각 것을 잘라 물체에 붙인다
  struct PendingView
  {
    builtin_interfaces::msg::Time stamp;
    size_t index;
    int x0, y0, x1, y1;
    double quality;
  };

  // 탐지 하나를 지도에 올릴 값: 영상 좌표 (u, v), 거리 z, 앞면 넓이
  struct Measure
  {
    double u, v, z, area;
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

  // 마스크 픽셀의 무게중심과 그 픽셀들 깊이의 mask_depth_percentile 값. 유효한 깊이가 적으면 false
  bool mask_measure(const sensor_msgs::msg::Image & depth, const sensor_msgs::msg::Image & mask, uint8_t id,
    const vision_msgs::msg::BoundingBox2D & box, double sx, double sy, double fx, double fy, Measure & m) const
  {
    const int x0 = std::max(0, static_cast<int>(box.center.position.x - box.size_x / 2));
    const int x1 = std::min(static_cast<int>(mask.width) - 1, static_cast<int>(box.center.position.x + box.size_x / 2));
    const int y0 = std::max(0, static_cast<int>(box.center.position.y - box.size_y / 2));
    const int y1 = std::min(static_cast<int>(mask.height) - 1, static_cast<int>(box.center.position.y + box.size_y / 2));
    std::vector<float> vals;
    double su = 0, sv = 0;
    size_t n = 0;
    for (int y = y0; y <= y1; ++y) {
      const uint8_t * mrow = mask.data.data() + static_cast<size_t>(y) * mask.step;
      const int dy = std::min(static_cast<int>(y * sy), static_cast<int>(depth.height) - 1);
      const uint8_t * drow = depth.data.data() + static_cast<size_t>(dy) * depth.step;
      for (int x = x0; x <= x1; ++x) {
        if (mrow[x] != id) {continue;}
        su += x;
        sv += y;
        ++n;
        const int dx = std::min(static_cast<int>(x * sx), static_cast<int>(depth.width) - 1);
        const float d = depth.encoding == "16UC1" ?
          reinterpret_cast<const uint16_t *>(drow)[dx] * 0.001f : reinterpret_cast<const float *>(drow)[dx];
        if (std::isfinite(d) && d >= min_depth_ && d <= max_depth_) {vals.push_back(d);}
      }
    }
    if (vals.size() < 10) {return false;}
    const auto k = static_cast<size_t>(mask_depth_percentile_ * (vals.size() - 1));
    std::nth_element(vals.begin(), vals.begin() + k, vals.end());
    m.u = su / n;
    m.v = sv / n;
    m.z = vals[k];
    m.area = n * m.z * m.z / (fx * fy);   // 픽셀 하나 = (z/fx)·(z/fy) m²
    return true;
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

    // 같은 시각 마스크 (분할 모델일 때). 영상 크기가 탐지 기준(camera_info)과 같아야 쓴다
    sensor_msgs::msg::Image::ConstSharedPtr mask;
    for (const auto & mk : masks_) {
      if (mk->header.stamp == dets->header.stamp && mk->width == info_->width && mk->height == info_->height) {
        mask = mk;
      }
    }

    for (size_t i = 0; i < dets->detections.size(); ++i) {
      const auto & d = dets->detections[i];
      if (d.results.empty() || d.results[0].hypothesis.score < min_score_) {continue;}
      const int cls = class_index(d.results[0].hypothesis.class_id);
      if (cls < 0) {continue;}
      Measure m{d.bbox.center.position.x, d.bbox.center.position.y, 0.0, 0.0};
      if (!mask || i >= 255 || !mask_measure(*depth, *mask, static_cast<uint8_t>(i + 1), d.bbox, sx, sy, fx, fy, m)) {
        m.z = box_depth(*depth, m.u * sx, m.v * sy, d.bbox.size_x * sx, d.bbox.size_y * sy);
      }
      if (!std::isfinite(m.z)) {continue;}

      geometry_msgs::msg::PointStamped p_cam, p_map;
      p_cam.point.x = (m.u - cx) * m.z / fx;
      p_cam.point.y = (m.v - cy) * m.z / fy;
      p_cam.point.z = m.z;
      tf2::doTransform(p_cam, p_map, tf);
      const double view_yaw = std::atan2(p_map.point.y - tf.transform.translation.y,
          p_map.point.x - tf.transform.translation.x);
      const size_t idx = add(cls, p_map.point.x, p_map.point.y, p_map.point.z, view_yaw,
          d.results[0].hypothesis.score, m.area);
      // 크게(가까이)·확실하게 찍힐수록 좋은 사진. 지금 것보다 나을 때만 자른다
      const double quality = d.bbox.size_x * d.bbox.size_y * d.results[0].hypothesis.score;
      if (quality > landmarks_[idx].view_quality) {
        const double mx = view_margin_ * d.bbox.size_x, my = view_margin_ * d.bbox.size_y;
        pending_.push_back({dets->header.stamp, idx,
            static_cast<int>(d.bbox.center.position.x - d.bbox.size_x / 2 - mx),
            static_cast<int>(d.bbox.center.position.y - d.bbox.size_y / 2 - my * 2),  // 위쪽 이름표까지
            static_cast<int>(d.bbox.center.position.x + d.bbox.size_x / 2 + mx),
            static_cast<int>(d.bbox.center.position.y + d.bbox.size_y / 2 + my), quality});
      }
    }
    take_views();
  }

  // 기다리던 장면의 결과 영상이 왔으면 잘라 물체에 붙인다. 너무 오래된 요청은 버린다
  void take_views()
  {
    for (auto it = pending_.begin(); it != pending_.end(); ) {
      sensor_msgs::msg::Image::ConstSharedPtr img;
      for (const auto & v : views_) {
        if (v->header.stamp == it->stamp) {img = v;}
      }
      const bool stale = !views_.empty() &&
        rclcpp::Time(views_.back()->header.stamp) - rclcpp::Time(it->stamp) > rclcpp::Duration::from_seconds(1.0);
      if (img && it->index < landmarks_.size() && it->quality > landmarks_[it->index].view_quality) {
        landmarks_[it->index].view = crop(*img, it->x0, it->y0, it->x1, it->y1);
        landmarks_[it->index].view_quality = it->quality;
        if (static_cast<int>(it->index) == selected_) {selected_pub_->publish(*landmarks_[it->index].view);}
      }
      it = (img || stale) ? pending_.erase(it) : std::next(it);
    }
    while (pending_.size() > 200) {pending_.pop_front();}
  }

  static sensor_msgs::msg::Image::SharedPtr crop(const sensor_msgs::msg::Image & img, int x0, int y0, int x1, int y1)
  {
    x0 = std::clamp(x0, 0, static_cast<int>(img.width) - 1);
    x1 = std::clamp(x1, x0 + 1, static_cast<int>(img.width));
    y0 = std::clamp(y0, 0, static_cast<int>(img.height) - 1);
    y1 = std::clamp(y1, y0 + 1, static_cast<int>(img.height));
    const size_t px = img.step / img.width;  // 한 픽셀 바이트
    auto out = std::make_shared<sensor_msgs::msg::Image>();
    out->header = img.header;
    out->encoding = img.encoding;
    out->width = x1 - x0;
    out->height = y1 - y0;
    out->step = out->width * px;
    out->data.resize(static_cast<size_t>(out->step) * out->height);
    for (int y = y0; y < y1; ++y) {
      std::copy_n(img.data.data() + static_cast<size_t>(y) * img.step + x0 * px, out->step,
        out->data.data() + static_cast<size_t>(y - y0) * out->step);
    }
    return out;
  }

  // 누른 곳에서 select_radius 안의 가장 가까운 물체를 고르고 대표 사진을 낸다. 빈 곳을 누르면 선택 해제
  void select(const geometry_msgs::msg::PointStamped & click)
  {
    geometry_msgs::msg::PointStamped p = click;
    if (click.header.frame_id != map_frame_) {
      try {
        p = tf_buffer_.transform(click, map_frame_, tf2::durationFromSec(0.2));
      } catch (const tf2::TransformException & e) {
        RCLCPP_WARN(get_logger(), "누른 곳을 %s 로 못 바꿈: %s", map_frame_.c_str(), e.what());
        return;
      }
    }
    selected_ = -1;
    double best = select_radius_;
    for (size_t i = 0; i < landmarks_.size(); ++i) {
      const double d = std::hypot(landmarks_[i].x - p.point.x, landmarks_[i].y - p.point.y);
      if (landmarks_[i].number > 0 && d < best) {best = d; selected_ = static_cast<int>(i);}
    }
    if (selected_ >= 0 && landmarks_[selected_].view) {selected_pub_->publish(*landmarks_[selected_].view);}
    publish_summary();
  }

  // 물체 목록에 넣고 그 물체의 번호(landmarks_ 안 위치)를 돌려준다
  size_t add(int cls, double x, double y, double z, double yaw, double score, double area)
  {
    Landmark * best = nullptr;
    double best_d = merge_radii_[cls];
    for (auto & l : landmarks_) {
      const double dist = std::hypot(l.x - x, l.y - y);
      if (l.cls == cls && dist < best_d) {best = &l; best_d = dist;}
    }
    if (!best) {
      landmarks_.push_back({cls, 0, x, y, z, yaw, 1, score, area});
      return landmarks_.size() - 1;
    }
    // 처음 몇 번은 평균, 그 뒤로는 천천히 따라간다 (멀리서 본 부정확한 값이 남지 않게)
    const double a = 1.0 / std::min(best->hits + 1, 20);
    best->x += a * (x - best->x);
    best->y += a * (y - best->y);
    best->z += a * (z - best->z);
    if (area > 0) {best->area = best->area > 0 ? best->area + a * (area - best->area) : area;}
    best->hits++;
    best->best_score = std::max(best->best_score, score);
    if (best->hits >= min_hits_ && best->number == 0) {
      best->number = ++counters_[std::min<size_t>(cls, counters_.size() - 1)];
      RCLCPP_INFO(get_logger(), "지도에 추가: %s %d (%.2f, %.2f)", style_of(class_names_[cls]).tag, best->number,
        best->x, best->y);
    }
    return static_cast<size_t>(best - landmarks_.data());
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
    // 화면(RViz 위에서 보기, Angle = view_angle) 기준 오른쪽·위쪽이 map 에서 어느 방향인지
    const double right_x = std::cos(view_angle_), right_y = std::sin(view_angle_);
    const double up_x = -std::sin(view_angle_), up_y = std::cos(view_angle_);

    // 아이콘 배지: icons/<클래스>.obj (납작한 원판 + 그림 텍스처). 그림 위쪽이 화면 위로 오게 view_angle 만큼 돌린다
    const double size = icon_size_ * (is_fire ? 1.0 + 0.1 * pulse : 1.0);
    Marker shadow = make(2, Marker::CYLINDER, 0.88, size * 1.08, size * 1.08, 0.01, rgba(0.0f, 0.0f, 0.0f, 0.35f));
    shadow.pose.position.x -= 0.03 * up_x;  // 화면 아래쪽으로 살짝 → 떠 있는 느낌
    shadow.pose.position.y -= 0.03 * up_y;
    arr.markers.push_back(shadow);
    // 선택한 물체: 아이콘 둘레 흰 고리 (선택이 풀리면 지운다)
    Marker ring = make(8, Marker::CYLINDER, 0.87, size * 1.45, size * 1.45, 0.01, rgba(1.0f, 1.0f, 1.0f, 0.9f));
    if (static_cast<int>(index) != selected_) {ring.action = Marker::DELETE;}
    arr.markers.push_back(ring);
    Marker icon = make(3, Marker::MESH_RESOURCE, 0.9, size, size, 1.0, rgba(0.0f, 0.0f, 0.0f, 0.0f), view_angle_);
    icon.mesh_resource = "package://limo_fire_perception/icons/" + std::string(s.name) + ".obj";
    icon.mesh_use_embedded_materials = true;  // 색 0 = 텍스처 그대로
    arr.markers.push_back(icon);

    // 라벨: 밝은 지도 위에서도 읽히게 어두운 바탕판(위에서 내려다볼 때 글자 뒤) + 클래스 색 테두리.
    // 아이콘을 가리지 않게 label_offset(화면 기준 오른쪽·위쪽, m) 만큼 비켜 두고, 바탕판은 화면 가로로 길게
    const std::string text = std::string(s.tag) + "-" + std::to_string(l.number);  // RViz 글꼴은 띄어쓰기가 넓어 붙여 쓴다
    const double len = 0.15 * static_cast<double>(text.size()) + 0.12;
    auto shifted = [&](Marker m) {
        m.pose.position.x += label_offset_[0] * right_x + label_offset_[1] * up_x;
        m.pose.position.y += label_offset_[0] * right_y + label_offset_[1] * up_y;
        m.pose.orientation.z = std::sin(view_angle_ / 2);
        m.pose.orientation.w = std::cos(view_angle_ / 2);
        return m;
      };
    Marker edge = shifted(make(6, Marker::CUBE, 0.95, len + 0.05, 0.36, 0.01, rgba(s.r, s.g, s.b, 1.0f)));
    edge.ns = "labels";
    arr.markers.push_back(edge);
    Marker plate = shifted(make(7, Marker::CUBE, 0.97, len, 0.31, 0.01, rgba(0.08f, 0.09f, 0.11f, 0.92f)));
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

    struct Row {int count = 0; double nearest = 1e9; int nearest_no = 0; double nearest_area = 0;};
    std::vector<Row> rows(class_names_.size());
    for (const auto & l : landmarks_) {
      if (l.number == 0) {continue;}
      Row & r = rows[l.cls];
      r.count++;
      const double d = std::hypot(l.x - rx, l.y - ry);
      if (d < r.nearest) {r.nearest = d; r.nearest_no = l.number; r.nearest_area = l.area;}
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
      std::string size_text;
      if (rows[fire].nearest_area > 0) {
        char buf[48];
        std::snprintf(buf, sizeof(buf), " · 크기 약 %.2f m²", rows[fire].nearest_area);
        size_text = buf;
      }
      html += "<div style='color:#ff6b4a;font-size:large;'><b>⚠ 화재 " + std::to_string(rows[fire].count) +
        "곳 감지</b></div><div style='color:#ffb4a2;'>가장 가까운 불 " + dist_text(rows[fire], style_of("fire")) +
        size_text + "</div>";
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
      "번 이상 확인된 것만 · 거리는 로봇 기준 직선 · Publish Point 로 아이콘을 누르면 사진</div>";
    if (selected_ >= 0 && selected_ < static_cast<int>(landmarks_.size())) {
      const Landmark & l = landmarks_[selected_];
      const Style & s = style_of(class_names_[l.cls]);
      char buf[160];
      std::snprintf(buf, sizeof(buf), "선택: %s %s-%d · 신뢰도 %.0f%% · %d 번 봄%s", s.ko, s.tag, l.number,
        l.best_score * 100, l.hits, l.view ? "" : " · 사진 없음");
      html += std::string("<div style='color:") + hex(s) + ";'><b>" + buf + "</b></div>";
      lines++;
    }

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
  double min_depth_, max_depth_, box_shrink_, depth_percentile_, mask_depth_percentile_, merge_radius_, min_score_,
    fire_zone_radius_;
  std::vector<double> merge_radii_;
  std::vector<double> label_offset_;
  double icon_size_, view_angle_, view_margin_, select_radius_;
  int selected_ = -1;
  std::deque<PendingView> pending_;
  std::deque<sensor_msgs::msg::Image::ConstSharedPtr> views_;
  int min_hits_;
  std::vector<std::string> class_names_;
  std::vector<Landmark> landmarks_;
  std::array<int, 16> counters_{};
  unsigned tick_ = 0;
  sensor_msgs::msg::CameraInfo::ConstSharedPtr info_;
  std::deque<sensor_msgs::msg::Image::ConstSharedPtr> masks_;

  message_filters::Subscriber<vision_msgs::msg::Detection2DArray> det_sub_;
  message_filters::Subscriber<sensor_msgs::msg::Image> depth_sub_;
  std::shared_ptr<message_filters::Synchronizer<Policy>> sync_;
  rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr info_sub_;
  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr mask_sub_, view_sub_;
  rclcpp::Subscription<geometry_msgs::msg::PointStamped>::SharedPtr click_sub_;
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr selected_pub_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr marker_pub_;
  rclcpp::Publisher<rviz_2d_overlay_msgs::msg::OverlayText>::SharedPtr overlay_pub_;
  rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr clear_srv_;
  rclcpp::TimerBase::SharedPtr timer_;
};

}  // namespace limo_fire_perception

RCLCPP_COMPONENTS_REGISTER_NODE(limo_fire_perception::ObjectMapperNode)
