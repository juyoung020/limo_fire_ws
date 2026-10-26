// 카메라 영상 → TensorRT YOLO → vision_msgs/Detection2DArray + 상자 그린 영상
// 파라미터는 config/*.yaml (엔진 경로·문턱·클래스 이름·토픽)
#include <chrono>
#include <memory>
#include <string>
#include <vector>

#include <opencv2/imgproc.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <vision_msgs/msg/detection2_d_array.hpp>

#include "limo_fire_perception/trt_yolo.hpp"

namespace limo_fire_perception
{

class YoloDetectorNode : public rclcpp::Node
{
public:
  YoloDetectorNode()
  : Node("yolo_detector")
  {
    const auto engine = declare_parameter<std::string>("engine_path");
    const auto conf = declare_parameter<double>("conf_threshold", 0.35);
    const auto iou = declare_parameter<double>("nms_iou", 0.5);
    class_names_ = declare_parameter<std::vector<std::string>>("class_names");
    publish_image_ = declare_parameter<bool>("publish_image", true);
    const auto image_topic = declare_parameter<std::string>("image_topic", "/camera/color/image_raw");

    yolo_ = std::make_unique<TrtYolo>(engine, static_cast<float>(conf), static_cast<float>(iou));
    RCLCPP_INFO(get_logger(), "엔진 %s: 입력 %dx%d, 출력 %s, 클래스 이름 %zu 개",
      engine.c_str(), yolo_->input_width(), yolo_->input_height(),
      yolo_->end2end() ? "end2end(NMS 없음)" : "raw(+NMS)", class_names_.size());
    if (!yolo_->end2end() && yolo_->num_classes() != static_cast<int>(class_names_.size())) {
      RCLCPP_WARN(get_logger(), "모델 클래스 %d 개 != class_names %zu 개", yolo_->num_classes(), class_names_.size());
    }

    det_pub_ = create_publisher<vision_msgs::msg::Detection2DArray>("~/detections", 10);
    img_pub_ = create_publisher<sensor_msgs::msg::Image>("~/image", rclcpp::SensorDataQoS());
    // 카메라 드라이버(실물 Orbbec, Webots 플러그인) 둘 다 받도록 SensorData(BEST_EFFORT)
    sub_ = create_subscription<sensor_msgs::msg::Image>(
      image_topic, rclcpp::SensorDataQoS(),
      [this](sensor_msgs::msg::Image::ConstSharedPtr msg) {on_image(msg);});
  }

private:
  std::string name_of(int c) const
  {
    return (c >= 0 && c < static_cast<int>(class_names_.size())) ? class_names_[c] : std::to_string(c);
  }

  void on_image(const sensor_msgs::msg::Image::ConstSharedPtr & msg)
  {
    // cv_bridge 는 쓰지 않는다: ROS 의 cv_bridge 는 Ubuntu OpenCV 4.5, 이 노드는 JetPack OpenCV 4.8 이라 섞이면 위험
    cv::Mat bgr;
    if (msg->encoding == "bgr8" || msg->encoding == "rgb8") {
      cv::Mat view(msg->height, msg->width, CV_8UC3, const_cast<uint8_t *>(msg->data.data()), msg->step);
      if (msg->encoding == "rgb8") {cv::cvtColor(view, bgr, cv::COLOR_RGB2BGR);} else {bgr = view.clone();}
    } else if (msg->encoding == "bgra8" || msg->encoding == "rgba8") {
      cv::Mat view(msg->height, msg->width, CV_8UC4, const_cast<uint8_t *>(msg->data.data()), msg->step);
      cv::cvtColor(view, bgr, msg->encoding == "bgra8" ? cv::COLOR_BGRA2BGR : cv::COLOR_RGBA2BGR);
    } else {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 5000, "지원하지 않는 영상 형식: %s", msg->encoding.c_str());
      return;
    }

    const auto t0 = std::chrono::steady_clock::now();
    const auto dets = yolo_->infer(bgr);
    const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    ms_avg_ = ms_avg_ <= 0 ? ms : 0.9 * ms_avg_ + 0.1 * ms;
    RCLCPP_INFO_THROTTLE(get_logger(), *get_clock(), 5000, "추론 %.1f ms (평균), 탐지 %zu 개", ms_avg_, dets.size());

    vision_msgs::msg::Detection2DArray arr;
    arr.header = msg->header;
    for (const auto & d : dets) {
      vision_msgs::msg::Detection2D det;
      det.header = msg->header;
      det.bbox.center.position.x = d.box.x + d.box.width / 2;
      det.bbox.center.position.y = d.box.y + d.box.height / 2;
      det.bbox.size_x = d.box.width;
      det.bbox.size_y = d.box.height;
      vision_msgs::msg::ObjectHypothesisWithPose h;
      h.hypothesis.class_id = name_of(d.class_id);
      h.hypothesis.score = d.score;
      det.results.push_back(h);
      arr.detections.push_back(det);
    }
    det_pub_->publish(arr);

    if (publish_image_ && img_pub_->get_subscription_count() > 0) {
      cv::Mat vis = bgr;
      for (const auto & d : dets) {
        const cv::Scalar col = color_of(d.class_id);
        cv::rectangle(vis, d.box, col, 2);
        const std::string label = name_of(d.class_id) + " " + std::to_string(static_cast<int>(d.score * 100)) + "%";
        int base = 0;
        const cv::Size ts = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &base);
        const cv::Point p(static_cast<int>(d.box.x), std::max(static_cast<int>(d.box.y), ts.height + 4));
        cv::rectangle(vis, cv::Rect(p.x, p.y - ts.height - 4, ts.width + 4, ts.height + 4), col, cv::FILLED);
        cv::putText(vis, label, cv::Point(p.x + 2, p.y - 3), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);
      }
      cv::putText(vis, cv::format("%.0f ms", ms_avg_), cv::Point(8, 20), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 255), 2);
      sensor_msgs::msg::Image out;
      out.header = msg->header;
      out.height = vis.rows;
      out.width = vis.cols;
      out.encoding = "bgr8";
      out.step = static_cast<uint32_t>(vis.cols * 3);
      out.data.assign(vis.data, vis.data + vis.total() * 3);
      img_pub_->publish(out);
    }
  }

  static cv::Scalar color_of(int c)
  {
    // 클래스마다 눈에 띄는 색 (BGR). 0 불 주황 · 1 연기 회색 · 2 소화기 빨강 · 3 유도등 초록 · 4 사람 파랑 · 5 문 갈색 · 6 계단 보라
    static const cv::Scalar table[] = {{0, 110, 255}, {140, 140, 140}, {30, 30, 210}, {70, 160, 20},
      {190, 95, 30}, {45, 82, 160}, {160, 80, 125}};
    return table[(c % 7 + 7) % 7];
  }

  std::unique_ptr<TrtYolo> yolo_;
  std::vector<std::string> class_names_;
  bool publish_image_;
  double ms_avg_ = 0;
  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr sub_;
  rclcpp::Publisher<vision_msgs::msg::Detection2DArray>::SharedPtr det_pub_;
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr img_pub_;
};

}  // namespace limo_fire_perception

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<limo_fire_perception::YoloDetectorNode>());
  rclcpp::shutdown();
  return 0;
}
