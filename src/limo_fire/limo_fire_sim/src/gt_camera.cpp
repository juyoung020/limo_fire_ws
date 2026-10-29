// 인식 벤치마크용 정답 카메라 (webots_ros2_driver 플러그인)
// 컬러 영상을 내는 그 카메라(rgb_camera)의 Recognition(분할)을 켜서, 영상과 같은 프레임의 물체별 정답을 낸다:
//   <gt>/detections  vision_msgs/Detection2DArray  상자 = 보이는 픽셀의 범위, class_id = 클래스 이름, id = 월드 데이터 파일의 물체 번호
//   <gt>/mask        sensor_msgs/Image mono8       0 = 배경, k = detections[k-1] (yolo_detector/mask 와 같은 형식)
// 물체는 월드에서 recognitionColors [ (클래스+1)/255, (물체 번호)/255, 1 ] 로 표시한다 (sim/webots/README.md "인식 정답").
// 컬러 영상은 교수님 limo_sensors 플러그인이 큰 렌더에서 잘라 내므로 같은 자리(crop)를 잘라 맞춘다.
// 받는 쪽이 있을 때만 켠다. 컬러 영상(syncTopic 의 camera_info)이 나오는 시각의 위상을 받아, 그 위상의 걸음에서
// 카메라·분할을 같이 다시 켜고 같은 걸음마다 낸다 — 컬러 영상과 정답이 같은 순간에 찍히고 시각 도장도 같다.
#include <array>
#include <cmath>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#include <pluginlib/class_list_macros.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <vision_msgs/msg/detection2_d_array.hpp>
#include <webots/camera.h>
#include <webots/camera_recognition_object.h>
#include <webots/robot.h>
#include <webots_ros2_driver/PluginInterface.hpp>
#include <webots_ros2_driver/WebotsNode.hpp>

namespace limo_fire_sim
{

class GtCamera : public webots_ros2_driver::PluginInterface
{
public:
  void init(webots_ros2_driver::WebotsNode * node, std::unordered_map<std::string, std::string> & params) override
  {
    node_ = node;
    auto get = [&](const std::string & key, const std::string & fallback) {
        const auto it = params.find(key);
        return it == params.end() ? fallback : it->second;
      };
    camera_ = wb_robot_get_device(get("cameraName", "rgb_camera").c_str());
    period_ms_ = std::stoi(get("periodMs", "30"));
    crop_top_ = std::stoi(get("cropTop", "0"));
    crop_left_ = std::stoi(get("cropLeft", "0"));
    width_ = std::stoi(get("width", "0"));
    height_ = std::stoi(get("height", "0"));
    min_pixels_ = std::stoi(get("minPixels", "30"));
    frame_id_ = get("frameId", "camera_color_optical_frame");
    std::stringstream names(get("classNames", ""));
    for (std::string s; std::getline(names, s, ',');) {class_names_.push_back(s);}
    const std::string topic = get("topicName", "/gt");
    det_pub_ = node->create_publisher<vision_msgs::msg::Detection2DArray>(topic + "/detections", 10);
    mask_pub_ = node->create_publisher<sensor_msgs::msg::Image>(topic + "/mask", 10);
    sync_topic_ = get("syncTopic", "/camera/color/camera_info");
    if (camera_ == 0) {
      RCLCPP_ERROR(node->get_logger(), "정답 카메라: 카메라 %s 없음", get("cameraName", "rgb_camera").c_str());
    }
  }

  void step() override
  {
    if (camera_ == 0) {return;}
    const bool wanted = det_pub_->get_subscription_count() + mask_pub_->get_subscription_count() > 0;
    const double t = wb_robot_get_time();
    const long ms = std::lround(t * 1000.0);
    // 컬러 영상 시각은 정답을 낼 때만 구독한다 (늘 구독하면 아무도 안 볼 때도 컬러 카메라가 렌더링된다)
    if (wanted && !sync_sub_) {
      sync_sub_ = node_->create_subscription<sensor_msgs::msg::CameraInfo>(sync_topic_, rclcpp::SensorDataQoS(),
          [this](sensor_msgs::msg::CameraInfo::ConstSharedPtr m) {
            const long ms = std::lround(m->header.stamp.sec * 1000.0 + m->header.stamp.nanosec / 1e6);
            phase_ms_ = ((ms % period_ms_) + period_ms_) % period_ms_;
          });
    }
    if (wanted && !enabled_) {
      // 컬러 영상이 나오는 위상의 걸음에서 카메라·분할을 같이 켠다 (컬러 영상이 아직 안 나오면 기다림)
      if (phase_ms_ < 0 || ms % period_ms_ != phase_ms_) {return;}
      wb_camera_enable(camera_, period_ms_);
      wb_camera_recognition_enable(camera_, period_ms_);
      wb_camera_recognition_enable_segmentation(camera_);
      enabled_ = true;
      next_ms_ = ms + period_ms_;
      RCLCPP_INFO(node_->get_logger(), "정답 카메라 켬 (%d ms)", period_ms_);
      return;
    }
    if (!wanted && enabled_) {
      wb_camera_recognition_disable_segmentation(camera_);
      wb_camera_recognition_disable(camera_);
      enabled_ = false;
      sync_sub_.reset();
      phase_ms_ = -1;
      return;
    }
    if (!enabled_ || ms < next_ms_) {return;}
    next_ms_ = ms + period_ms_;
    publish(t);
  }

private:
  void publish(double t)
  {
    const int rw = wb_camera_get_width(camera_), rh = wb_camera_get_height(camera_);
    const unsigned char * seg = wb_camera_recognition_get_segmentation_image(camera_);
    if (!seg) {return;}
    const int w = width_ > 0 ? width_ : rw, h = height_ > 0 ? height_ : rh;

    // 월드 물체 색 → 정답 번호. 색 코드가 아닌 것(B != 255)은 무시
    struct Obj {int cls, inst; int x0, y0, x1, y1, n;};
    std::vector<Obj> objs;
    std::unordered_map<uint32_t, int> index;  // RGB → objs 위치
    auto mask = std::make_unique<sensor_msgs::msg::Image>();
    mask->width = w;
    mask->height = h;
    mask->encoding = "mono8";
    mask->step = w;
    mask->data.assign(static_cast<size_t>(w) * h, 0);
    for (int y = 0; y < h; ++y) {
      const int sy = y + crop_top_;
      if (sy < 0 || sy >= rh) {continue;}
      for (int x = 0; x < w; ++x) {
        const int sx = x + crop_left_;
        if (sx < 0 || sx >= rw) {continue;}
        const unsigned char * p = seg + (static_cast<size_t>(sy) * rw + sx) * 4;  // BGRA
        if (p[0] != 255 || p[2] == 0) {continue;}
        const uint32_t key = (p[2] << 16) | (p[1] << 8) | p[0];
        auto it = index.find(key);
        if (it == index.end()) {
          if (objs.size() >= 255) {continue;}
          it = index.emplace(key, static_cast<int>(objs.size())).first;
          objs.push_back({p[2] - 1, p[1], x, y, x, y, 0});
        }
        Obj & o = objs[it->second];
        o.x0 = std::min(o.x0, x);
        o.y0 = std::min(o.y0, y);
        o.x1 = std::max(o.x1, x);
        o.y1 = std::max(o.y1, y);
        o.n++;
        mask->data[static_cast<size_t>(y) * w + x] = static_cast<uint8_t>(it->second + 1);
      }
    }

    // 너무 작게 보이는 것은 정답에서 뺀다(번호 영상에서도 지움). 남은 것의 순서대로 번호를 다시 매긴다
    std::array<uint8_t, 256> remap{};
    auto dets = std::make_unique<vision_msgs::msg::Detection2DArray>();
    for (size_t i = 0; i < objs.size(); ++i) {
      const Obj & o = objs[i];
      if (o.n < min_pixels_ || o.cls < 0 || o.cls >= static_cast<int>(class_names_.size())) {continue;}
      remap[i + 1] = static_cast<uint8_t>(dets->detections.size() + 1);
      vision_msgs::msg::Detection2D d;
      d.bbox.center.position.x = (o.x0 + o.x1 + 1) / 2.0;
      d.bbox.center.position.y = (o.y0 + o.y1 + 1) / 2.0;
      d.bbox.size_x = o.x1 - o.x0 + 1;
      d.bbox.size_y = o.y1 - o.y0 + 1;
      d.id = std::to_string(o.inst);
      vision_msgs::msg::ObjectHypothesisWithPose hyp;
      hyp.hypothesis.class_id = class_names_[o.cls];
      hyp.hypothesis.score = 1.0;
      d.results.push_back(hyp);
      dets->detections.push_back(d);
    }
    for (auto & v : mask->data) {v = remap[v];}

    builtin_interfaces::msg::Time stamp;
    stamp.sec = static_cast<int32_t>(std::floor(t));
    stamp.nanosec = static_cast<uint32_t>(std::lround((t - std::floor(t)) * 1e9) % 1000000000);
    dets->header.stamp = stamp;
    dets->header.frame_id = frame_id_;
    mask->header = dets->header;
    for (auto & d : dets->detections) {d.header = dets->header;}
    det_pub_->publish(std::move(dets));
    mask_pub_->publish(std::move(mask));
  }

  webots_ros2_driver::WebotsNode * node_ = nullptr;
  WbDeviceTag camera_ = 0;
  int period_ms_ = 30, crop_top_ = 0, crop_left_ = 0, width_ = 0, height_ = 0, min_pixels_ = 30;
  std::string frame_id_, sync_topic_;
  std::vector<std::string> class_names_;
  bool enabled_ = false;
  long next_ms_ = 0;
  long phase_ms_ = -1;
  rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr sync_sub_;
  rclcpp::Publisher<vision_msgs::msg::Detection2DArray>::SharedPtr det_pub_;
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr mask_pub_;
};

}  // namespace limo_fire_sim

PLUGINLIB_EXPORT_CLASS(limo_fire_sim::GtCamera, webots_ros2_driver::PluginInterface)
