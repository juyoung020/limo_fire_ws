// 카메라 영상 → TensorRT YOLO → vision_msgs/Detection2DArray + 상자 그린 영상
// 분할(-seg) 엔진이면 물체 번호 영상(~/mask, mono8, 값 = detections 안 순서 + 1, 0 = 배경)도 같은 시각으로 낸다 (물체가 있을 때만)
// 파라미터는 config/*.yaml (엔진 경로·문턱·클래스 이름·토픽)
//
// 영상은 실행기(executor) 콜백이 아니라 작업 스레드 하나가 WaitSet 으로 직접 받고 추론까지 한다:
//   결과 기다림(CPU 잠듦) → 다음 영상이 와 있으면 GPU 에 먼저 걸고 → 앞 장 결과 발행 (GPU 와 겹침)
// 그래서 한 스레드 컨테이너에서 다른 노드(object_map 등) 콜백이 길어져도 영상 받기가 밀리지 않는다.
// 밀리면 오래된 영상은 버린다 (늦은 탐지보다 최신 탐지). 받은 영상 메시지는 몇 개를 돌려 써서 장마다 큰 할당을 하지 않는다
// 결과 영상(~/image)은 구독자가 있을 때만 만들고, vis_release_s 초 동안 없으면 그 GPU 버퍼·그래프를 푼다
// 컴포넌트라 카메라 드라이버·object_map 과 한 프로세스에 올리면 영상·탐지·마스크를 복사 없이 주고받는다
#include <atomic>
#include <chrono>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include <opencv2/imgproc.hpp>
#include <ament_index_cpp/get_package_share_directory.hpp>
#include <rclcpp/rclcpp.hpp>
#include <rclcpp_components/register_node_macro.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <vision_msgs/msg/detection2_d_array.hpp>

#include "limo_fire_perception/trt_yolo.hpp"

namespace limo_fire_perception
{

class ObjectDetectorNode : public rclcpp::Node
{
public:
  explicit ObjectDetectorNode(const rclcpp::NodeOptions & options)
  : Node("object_detector", options)
  {
    // 절대 경로가 아니면 이 패키지 설치 폴더의 models/ 에서 찾는다 (엔진을 src/…/models 에 두고 colcon build)
    auto engine = declare_parameter<std::string>("engine_path");
    if (!engine.empty() && engine[0] != '/') {
      engine = ament_index_cpp::get_package_share_directory("limo_fire_perception") + "/models/" + engine;
    }
    const auto conf = declare_parameter<double>("conf_threshold", 0.35);
    class_names_ = declare_parameter<std::vector<std::string>>("class_names");
    publish_image_ = declare_parameter<bool>("publish_image", true);
    vis_release_ = std::chrono::duration<double>(declare_parameter<double>("vis_release_s", 5.0));
    const auto image_topic = declare_parameter<std::string>("image_topic", "/camera/color/image_raw");

    yolo_ = std::make_unique<TrtYolo>(engine, static_cast<float>(conf));
    std::vector<cv::Vec3b> colors;
    for (int c = 0; c < 7; ++c) {
      const cv::Scalar s = color_of(c);
      colors.emplace_back(s[0], s[1], s[2]);
    }
    yolo_->set_class_colors(colors);
    RCLCPP_INFO(get_logger(), "엔진 %s: 입력 %dx%d %s, 출력 end2end(NMS 없음), 클래스 이름 %zu 개, TensorRT 작업 메모리 %.1f MB",
      engine.c_str(), yolo_->input_width(), yolo_->input_height(),
      yolo_->uint8_input() ? "uint8 RGB(전처리 엔진 안)" : "float", class_names_.size(), yolo_->activation_bytes() / 1048576.0);

    det_pub_ = create_publisher<vision_msgs::msg::Detection2DArray>("detections", 10);
    if (yolo_->has_masks()) {
      mask_pub_ = create_publisher<sensor_msgs::msg::Image>("mask", 10);
      RCLCPP_INFO(get_logger(), "분할 모델: 물체 마스크를 ~/mask 로 냄");
    }
    // 신뢰(RELIABLE) 발행이면 RViz·rqt 의 신뢰/최선 구독 둘 다 받는다
    img_pub_ = create_publisher<sensor_msgs::msg::Image>("detection_image", rclcpp::QoS(1));
    // 카메라 드라이버(실물 Orbbec, Webots 플러그인) 둘 다 받도록 SensorData(BEST_EFFORT).
    // 실행기에 안 올리는 콜백 그룹에 두고 작업 스레드가 WaitSet 으로 꺼낸다 (카메라는 다른 프로세스라 프로세스 안 전달은 끔)
    rclcpp::SubscriptionOptions so;
    so.callback_group = create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive, false);
    so.use_intra_process_comm = rclcpp::IntraProcessSetting::Disable;
    sub_ = create_subscription<sensor_msgs::msg::Image>(
      image_topic, rclcpp::SensorDataQoS(), [](sensor_msgs::msg::Image::ConstSharedPtr) {}, so);
    wait_set_.add_subscription(sub_);
    worker_ = std::thread([this] {run();});
  }

  ~ObjectDetectorNode() override
  {
    stop_ = true;   // 작업 스레드는 WaitSet 을 100 ms 마다 깨어 본다
    if (worker_.joinable()) {worker_.join();}
  }

private:
  using Clock = std::chrono::steady_clock;

  struct Frame
  {
    sensor_msgs::msg::Image::ConstSharedPtr msg;
    cv::Mat rgb;       // msg 를 그대로 보거나(rgb8) 바꾼 것
    int slot = -1;
    bool vis = false;  // 결과 영상을 GPU 에서 칠해 받음
    Clock::time_point t_submit;
  };

  // 다른 곳(Frame)이 안 쥐고 있는 받기용 메시지 (데이터 버퍼를 다시 써서 장마다 921 KB 할당을 안 함)
  std::shared_ptr<sensor_msgs::msg::Image> spare_msg()
  {
    for (auto & m : pool_) {
      if (m.use_count() == 1) {return m;}
    }
    pool_.push_back(std::make_shared<sensor_msgs::msg::Image>());
    return pool_.back();
  }

  // 와 있는 영상을 모두 꺼내 가장 최근 것 하나를 돌려준다 (앞 것은 버린 수로 셈). wait 이면 올 때까지 잔다 (멈춤 확인하려고 100 ms 마다 깸)
  sensor_msgs::msg::Image::ConstSharedPtr take_latest(bool wait)
  {
    std::shared_ptr<sensor_msgs::msg::Image> latest = std::move(held_);
    held_.reset();
    while (wait && !latest && !stop_ && rclcpp::ok()) {
      if (wait_set_.wait(std::chrono::milliseconds(100)).kind() == rclcpp::WaitResultKind::Ready) {break;}
    }
    for (;;) {
      auto m = spare_msg();
      rclcpp::MessageInfo info;
      if (!sub_->take(*m, info)) {break;}
      if (latest) {++dropped_;}
      latest = std::move(m);
    }
    return latest;
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

  bool image_wanted() const
  {
    return publish_image_ &&
           img_pub_->get_subscription_count() + img_pub_->get_intra_process_subscription_count() > 0;
  }

  // 영상을 GPU 에 건다. 크기가 앞 장과 다르면 앞 장 결과를 다 쓴 뒤에만 걸 수 있어 (버퍼를 새로 만듦) 미뤄 둔다
  bool submit(sensor_msgs::msg::Image::ConstSharedPtr msg, Frame & f, const Frame * in_use)
  {
    if (!to_rgb(msg, f.rgb)) {
      return false;
    }
    if (in_use && in_use->rgb.size() != f.rgb.size()) {
      held_ = std::const_pointer_cast<sensor_msgs::msg::Image>(msg);   // 앞 장을 다 쓴 뒤 다시 건다
      return false;
    }
    f.msg = std::move(msg);
    f.t_submit = Clock::now();
    f.vis = image_wanted();
    if (f.vis) {t_vis_wanted_ = f.t_submit;}
    const bool had_vis = yolo_->vis_enabled();
    f.slot = yolo_->submit(f.rgb, f.vis);
    if (!had_vis && yolo_->vis_enabled()) {RCLCPP_INFO(get_logger(), "결과 영상 구독자가 생겨 GPU 덧칠을 켬 — %s", yolo_->memory_report().c_str());}
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
      if (!logged_memory_) {
        RCLCPP_INFO(get_logger(), "%s", yolo_->memory_report().c_str());
        logged_memory_ = true;
      }
      // 결과 영상을 vis_release_s 동안 아무도 안 보면 그 GPU 버퍼·그래프를 푼다 (이 장은 결과 영상이 없고 다른 슬롯도 비어 있을 때)
      if (res.vis.empty() && yolo_->vis_enabled() && Clock::now() - t_vis_wanted_ > vis_release_ && yolo_->release_vis()) {
        RCLCPP_INFO(get_logger(), "결과 영상 구독자가 없어 GPU 덧칠 버퍼를 풂");
      }
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
      const size_t dropped = dropped_;
      dropped_ = 0;
      static const char * modes[] = {"그래프 없음", "CUDA 그래프", "CUDA 그래프 + 조건 노드"};
      RCLCPP_INFO(get_logger(), "추론 %.1f ms (평균, 걸기→결과), %.1f 장/초, 버린 영상 %zu, 탐지 %zu 개, %s",
        ms_avg_, t_log_.time_since_epoch().count() ? frames_ / sec : 0.0, dropped, dets.size(),
        modes[yolo_->graph_mode()]);
      t_log_ = now;
      frames_ = 0;
    }

    // 마스크를 탐지보다 먼저 낸다 (object_map 이 탐지를 받을 때 같은 시각 마스크가 이미 와 있게).
    // 물체가 없으면 쓸 곳이 없으니 내지 않는다
    if (mask_pub_ && !res.ids.empty() && !dets.empty()) {
      auto out = std::make_unique<sensor_msgs::msg::Image>();
      out->header = header;
      out->height = res.ids.rows;
      out->width = res.ids.cols;
      out->encoding = "mono8";
      out->step = res.ids.cols;
      out->data.assign(res.ids.data, res.ids.data + res.ids.total());
      mask_pub_->publish(std::move(out));
    }

    auto arr = std::make_unique<vision_msgs::msg::Detection2DArray>();
    arr->header = header;
    arr->detections.reserve(dets.size());
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
      arr->detections.push_back(std::move(det));
    }
    det_pub_->publish(std::move(arr));

    if (f.vis || (publish_image_ && res.vis.empty() && image_wanted())) {
      publish_vis(f, res);
    }
  }

  // 사람이 보는 영상 (RGB 그대로 그려 rgb8 로 낸다). 마스크 덧칠은 GPU 가 해 둔 것(res.vis)을 쓰고 상자·글자만 여기서
  void publish_vis(const Frame & f, const YoloResult & res)
  {
    auto out = std::make_unique<sensor_msgs::msg::Image>();
    out->header = f.msg->header;
    out->height = f.rgb.rows;
    out->width = f.rgb.cols;
    out->encoding = "rgb8";
    out->step = static_cast<uint32_t>(f.rgb.cols * 3);
    out->data.resize(static_cast<size_t>(out->step) * out->height);
    cv::Mat vis(f.rgb.size(), CV_8UC3, out->data.data());
    (res.vis.empty() ? f.rgb : res.vis).copyTo(vis);
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
    img_pub_->publish(std::move(out));
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
  std::chrono::duration<double> vis_release_{5.0};
  Clock::time_point t_vis_wanted_{};
  bool logged_memory_ = false;
  double ms_avg_ = 0;
  size_t frames_ = 0;
  Clock::time_point t_log_{};

  // 작업 스레드만 쓰는 것
  rclcpp::WaitSet wait_set_;
  std::vector<std::shared_ptr<sensor_msgs::msg::Image>> pool_;   // 받기용 메시지 (돌려 씀)
  std::shared_ptr<sensor_msgs::msg::Image> held_;   // 크기가 바뀌어 다음에 걸 영상
  size_t dropped_ = 0;
  std::atomic<bool> stop_{false};

  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr sub_;
  rclcpp::Publisher<vision_msgs::msg::Detection2DArray>::SharedPtr det_pub_;
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr img_pub_;
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr mask_pub_;
  std::thread worker_;   // 맨 뒤: 위 멤버가 다 만들어진 뒤 시작, 먼저 멈춤(소멸자)
};

}  // namespace limo_fire_perception

RCLCPP_COMPONENTS_REGISTER_NODE(limo_fire_perception::ObjectDetectorNode)
