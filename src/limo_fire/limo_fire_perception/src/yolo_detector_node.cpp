// 카메라 영상 → TensorRT YOLO → vision_msgs/Detection2DArray + 상자 그린 영상
// 분할(-seg) 엔진이면 물체 번호 영상(~/mask, mono8, 값 = detections 안 순서 + 1, 0 = 배경)도 같은 시각으로 낸다
// 파라미터는 config/*.yaml (엔진 경로·문턱·클래스 이름·토픽)
//
// 구독 콜백은 가장 최근 영상만 남기고 바로 돌아온다. 추론은 작업 스레드 하나가 한다:
//   결과 기다림(CPU 잠듦) → 다음 영상이 와 있으면 GPU 에 먼저 걸고 → 앞 장 결과 발행 (GPU 와 겹침)
// 밀리면 오래된 영상은 버린다 (늦은 탐지보다 최신 탐지)
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
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
    RCLCPP_INFO(get_logger(), "엔진 %s: 입력 %dx%d %s, 출력 %s, 클래스 이름 %zu 개",
      engine.c_str(), yolo_->input_width(), yolo_->input_height(),
      yolo_->uint8_input() ? "uint8 RGB(전처리 엔진 안)" : "float",
      yolo_->end2end() ? "end2end(NMS 없음)" : "raw(+NMS)", class_names_.size());
    if (!yolo_->end2end() && yolo_->num_classes() != static_cast<int>(class_names_.size())) {
      RCLCPP_WARN(get_logger(), "모델 클래스 %d 개 != class_names %zu 개", yolo_->num_classes(), class_names_.size());
    }

    det_pub_ = create_publisher<vision_msgs::msg::Detection2DArray>("~/detections", 10);
    if (yolo_->has_masks()) {
      mask_pub_ = create_publisher<sensor_msgs::msg::Image>("~/mask", 10);
      RCLCPP_INFO(get_logger(), "분할 모델: 물체 마스크를 ~/mask 로 냄");
    }
    // 신뢰(RELIABLE) 발행이면 RViz·rqt 의 신뢰/최선 구독 둘 다 받는다
    img_pub_ = create_publisher<sensor_msgs::msg::Image>("~/image", rclcpp::QoS(1));
    // 카메라 드라이버(실물 Orbbec, Webots 플러그인) 둘 다 받도록 SensorData(BEST_EFFORT)
    sub_ = create_subscription<sensor_msgs::msg::Image>(
      image_topic, rclcpp::SensorDataQoS(),
      [this](sensor_msgs::msg::Image::ConstSharedPtr msg) {
        {
          std::lock_guard<std::mutex> lk(mtx_);
          if (latest_) {++dropped_;}
          latest_ = std::move(msg);
        }
        cv_.notify_one();
      });
    worker_ = std::thread([this] {run();});
  }

  ~YoloDetectorNode() override
  {
    {
      std::lock_guard<std::mutex> lk(mtx_);
      stop_ = true;
    }
    cv_.notify_one();
    if (worker_.joinable()) {worker_.join();}
  }

private:
  using Clock = std::chrono::steady_clock;

  struct Frame
  {
    sensor_msgs::msg::Image::ConstSharedPtr msg;
    cv::Mat rgb;       // msg 를 그대로 보거나(rgb8) 바꾼 것
    int slot = -1;
    Clock::time_point t_submit;
  };

  // 가장 최근 영상 하나를 가져온다. wait 이면 올 때까지 잔다
  sensor_msgs::msg::Image::ConstSharedPtr take_latest(bool wait)
  {
    std::unique_lock<std::mutex> lk(mtx_);
    if (wait) {
      cv_.wait(lk, [this] {return stop_ || latest_;});
    }
    return std::move(latest_);
  }

  bool to_rgb(const sensor_msgs::msg::Image::ConstSharedPtr & msg, cv::Mat & rgb)
  {
    // cv_bridge 는 쓰지 않는다: ROS 의 cv_bridge 는 Ubuntu OpenCV 4.5, 이 노드는 JetPack OpenCV 4.8 이라 섞이면 위험
    auto * data = const_cast<uint8_t *>(msg->data.data());
    if (msg->encoding == "rgb8") {
      rgb = cv::Mat(msg->height, msg->width, CV_8UC3, data, msg->step);
    } else if (msg->encoding == "bgr8") {
      cv::cvtColor(cv::Mat(msg->height, msg->width, CV_8UC3, data, msg->step), rgb, cv::COLOR_BGR2RGB);
    } else if (msg->encoding == "bgra8" || msg->encoding == "rgba8") {
      cv::cvtColor(cv::Mat(msg->height, msg->width, CV_8UC4, data, msg->step), rgb,
        msg->encoding == "bgra8" ? cv::COLOR_BGRA2RGB : cv::COLOR_RGBA2RGB);
    } else {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 5000, "지원하지 않는 영상 형식: %s", msg->encoding.c_str());
      return false;
    }
    return true;
  }

  // 영상을 GPU 에 건다. 크기가 앞 장과 다르면 앞 장 결과를 다 쓴 뒤에만 걸 수 있어 (버퍼를 새로 만듦) 미뤄 둔다
  bool submit(sensor_msgs::msg::Image::ConstSharedPtr msg, Frame & f, const Frame * in_use)
  {
    if (!to_rgb(msg, f.rgb)) {
      return false;
    }
    if (in_use && in_use->rgb.size() != f.rgb.size()) {
      std::lock_guard<std::mutex> lk(mtx_);
      if (!latest_) {latest_ = msg;}
      return false;
    }
    f.msg = std::move(msg);
    f.t_submit = Clock::now();
    f.slot = yolo_->submit(f.rgb);
    return true;
  }

  void run()
  {
    Frame cur, next;
    bool have_cur = false;
    while (rclcpp::ok()) {
      if (!have_cur) {
        auto msg = take_latest(true);
        if (stop_) {break;}
        have_cur = msg && submit(std::move(msg), cur, nullptr);
        continue;
      }
      YoloResult res;
      try {
        res = yolo_->collect(cur.slot);
      } catch (const std::exception & e) {
        RCLCPP_ERROR(get_logger(), "추론 실패: %s", e.what());
        have_cur = false;
        continue;
      }
      const double ms = std::chrono::duration<double, std::milli>(Clock::now() - cur.t_submit).count();
      // 다음 장을 먼저 GPU 에 걸어 두고 이 장을 발행한다
      bool have_next = false;
      if (auto msg = take_latest(false)) {
        try {
          have_next = submit(std::move(msg), next, &cur);
        } catch (const std::exception & e) {
          RCLCPP_ERROR(get_logger(), "GPU 에 걸기 실패: %s", e.what());
        }
      }
      publish(cur, res, ms);
      if (have_next) {
        std::swap(cur, next);
      }
      have_cur = have_next;
    }
  }

  void publish(const Frame & f, const YoloResult & res, double ms)
  {
    const auto & dets = res.dets;
    const auto & header = f.msg->header;
    ms_avg_ = ms_avg_ <= 0 ? ms : 0.9 * ms_avg_ + 0.1 * ms;
    ++frames_;
    const auto now = Clock::now();
    if (now - t_log_ > std::chrono::seconds(5)) {
      const double sec = std::chrono::duration<double>(now - t_log_).count();
      size_t dropped;
      {
        std::lock_guard<std::mutex> lk(mtx_);
        dropped = dropped_;
        dropped_ = 0;
      }
      RCLCPP_INFO(get_logger(), "추론 %.1f ms (평균, 걸기→결과), %.1f 장/초, 버린 영상 %zu, 탐지 %zu 개%s",
        ms_avg_, t_log_.time_since_epoch().count() ? frames_ / sec : 0.0, dropped, dets.size(),
        yolo_->cuda_graph() ? "" : " (CUDA 그래프 없음)");
      t_log_ = now;
      frames_ = 0;
    }

    // 마스크를 탐지보다 먼저 낸다 (object_mapper 가 탐지를 받을 때 같은 시각 마스크가 이미 와 있게)
    if (mask_pub_ && !res.ids.empty()) {
      sensor_msgs::msg::Image out;
      out.header = header;
      out.height = res.ids.rows;
      out.width = res.ids.cols;
      out.encoding = "mono8";
      out.step = res.ids.cols;
      out.data.assign(res.ids.data, res.ids.data + res.ids.total());
      mask_pub_->publish(out);
    }

    vision_msgs::msg::Detection2DArray arr;
    arr.header = header;
    for (const auto & d : dets) {
      vision_msgs::msg::Detection2D det;
      det.header = header;
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
      publish_vis(f, res);
    }
  }

  // 사람이 보는 영상 (RGB 그대로 그려 rgb8 로 낸다)
  void publish_vis(const Frame & f, const YoloResult & res)
  {
    sensor_msgs::msg::Image out;
    out.header = f.msg->header;
    out.height = f.rgb.rows;
    out.width = f.rgb.cols;
    out.encoding = "rgb8";
    out.step = static_cast<uint32_t>(f.rgb.cols * 3);
    out.data.resize(static_cast<size_t>(out.step) * out.height);
    cv::Mat vis(f.rgb.size(), CV_8UC3, out.data.data());
    f.rgb.copyTo(vis);
    // 마스크는 클래스 색을 반투명하게 덧칠
    for (size_t i = 0; i < res.dets.size() && !res.ids.empty(); ++i) {
      const auto & d = res.dets[i];
      if (d.mask_roi.empty()) {continue;}
      cv::Mat m = res.ids(d.mask_roi) == static_cast<int>(i + 1);
      cv::Mat roi = vis(d.mask_roi);
      cv::Mat tint(roi.size(), roi.type(), color_of(d.class_id));
      cv::Mat blended;
      cv::addWeighted(roi, 0.55, tint, 0.45, 0.0, blended);
      blended.copyTo(roi, m);
    }
    for (const auto & d : res.dets) {
      const cv::Scalar col = color_of(d.class_id);
      cv::rectangle(vis, d.box, col, 2);
      const std::string label = name_of(d.class_id) + " " + std::to_string(static_cast<int>(d.score * 100)) + "%";
      int base = 0;
      const cv::Size ts = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &base);
      const cv::Point p(static_cast<int>(d.box.x), std::max(static_cast<int>(d.box.y), ts.height + 4));
      cv::rectangle(vis, cv::Rect(p.x, p.y - ts.height - 4, ts.width + 4, ts.height + 4), col, cv::FILLED);
      cv::putText(vis, label, cv::Point(p.x + 2, p.y - 3), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);
    }
    cv::putText(vis, cv::format("%.0f ms", ms_avg_), cv::Point(8, 20), cv::FONT_HERSHEY_SIMPLEX, 0.6,
      cv::Scalar(255, 255, 0), 2);
    img_pub_->publish(out);
  }

  std::string name_of(int c) const
  {
    return (c >= 0 && c < static_cast<int>(class_names_.size())) ? class_names_[c] : std::to_string(c);
  }

  static cv::Scalar color_of(int c)
  {
    // 클래스마다 눈에 띄는 색 (RGB). 0 불 주황 · 1 연기 회색 · 2 소화기 빨강 · 3 유도등 초록 · 4 사람 파랑 · 5 문 갈색 · 6 계단 보라
    static const cv::Scalar table[] = {{255, 110, 0}, {140, 140, 140}, {210, 30, 30}, {20, 160, 70},
      {30, 95, 190}, {160, 82, 45}, {125, 80, 160}};
    return table[(c % 7 + 7) % 7];
  }

  std::unique_ptr<TrtYolo> yolo_;
  std::vector<std::string> class_names_;
  bool publish_image_;
  double ms_avg_ = 0;
  size_t frames_ = 0;
  Clock::time_point t_log_{};

  std::mutex mtx_;
  std::condition_variable cv_;
  sensor_msgs::msg::Image::ConstSharedPtr latest_;
  size_t dropped_ = 0;
  std::atomic<bool> stop_{false};
  std::thread worker_;

  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr sub_;
  rclcpp::Publisher<vision_msgs::msg::Detection2DArray>::SharedPtr det_pub_;
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr img_pub_;
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr mask_pub_;
};

}  // namespace limo_fire_perception

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<limo_fire_perception::YoloDetectorNode>());
  rclcpp::shutdown();
  return 0;
}
