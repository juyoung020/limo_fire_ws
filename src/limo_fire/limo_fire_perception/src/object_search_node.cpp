// 물체 설명 글로 찾기 (/perception/object_search) — "빨간 컵", "소화기 옆 의자" 같은 글이 지도 위 물체마다 얼마나 맞는지 확률로.
// 기본은 꺼짐: launch search:=true 일 때만 이 노드가 올라가고 object_map 이 물체 사진을 낸다 (README "설명 글로 물체 찾기").
//
// 입력: object_map 이 낸 물체별 원본 사진(views_topic, frame_id = 물체 이름 "chair-1") → Embedder(backend) 그림 벡터 →
//       물체마다 픽셀 수가 큰(가까이·크게 찍힌) 사진 views_per_object 장만 둔다.
// 글: query_topic (std_msgs/String) → prompt 틀 "a photo of {}." 에 넣어 글 벡터 한 번 (글이 바뀔 때만 — 다시 셈은 둔 벡터로). 빈 글 = 지움.
// 결과: query_result_topic (vision_msgs/Detection3DArray, map, transient_local 깊이 1) — objects_topic 의 물체마다 하나, 점수순.
//       id = 물체 이름, results[0].hypothesis.class_id = 클래스, score = sem_score.hpp object_prob (사진 없으면 kNoViews = −1), 위치·크기는 objects_topic 그대로.
//       글이 살아 있는 동안 rescore_period 마다 물체 목록·사진 벡터가 바뀌었으면 다시 내서 새로 올라간 물체도 점수를 받는다.
// 사진 벡터는 embeddings_file 에 저장(~/save · ~/save_to)·켤 때 읽기 (물체 지도 파일과 같이 — 저장 지도로 다시 돌 때 사진을 다시 안 찍어도 찾음).
#include <algorithm>
#include <chrono>
#include <deque>
#include <fstream>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include <ament_index_cpp/get_package_share_directory.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <rclcpp/rclcpp.hpp>
#include <rclcpp_components/register_node_macro.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <std_msgs/msg/string.hpp>
#include <std_srvs/srv/trigger.hpp>
#include <vision_msgs/msg/detection3_d_array.hpp>

#include "limo_fire_perception/embedder.hpp"
#include "limo_fire_perception/sem_score.hpp"

namespace limo_fire_perception
{

class ObjectSearchNode : public rclcpp::Node
{
public:
  explicit ObjectSearchNode(const rclcpp::NodeOptions & options)
  : Node("object_search", options)
  {
    // 모델 파일: 이름만 쓰면 model_dir(비우면 이 패키지 설치 폴더 models/)에서 — src/…/models 에 두고 colcon build
    std::string model_dir = declare_parameter<std::string>("model_dir", "");
    if (model_dir.empty()) {model_dir = ament_index_cpp::get_package_share_directory("limo_fire_perception") + "/models";}
    // 판마다의 파라미터는 설정 파일 형(정수·글)과 상관없이 글로 받아 판에 넘긴다
    rcl_interfaces::msg::ParameterDescriptor dyn;
    dyn.dynamic_typing = true;
    net_ = make_embedder(declare_parameter<std::string>("backend", "siglip2_trt"), model_dir,
        [this, &dyn](const std::string & name, const std::string & def) {
          return rclcpp::to_string(declare_parameter(name, rclcpp::ParameterValue(def), dyn));
        });
    // google/siglip2-base-patch16-224: exp(logit_scale) = 112.67, logit_bias = −16.77 (README 변환 스크립트가 출력)
    scale_ = declare_parameter<double>("logit_scale", 112.66889953613281);
    bias_ = declare_parameter<double>("logit_bias", -16.771724700927734);
    per_obj_ = static_cast<size_t>(std::max<int64_t>(1, declare_parameter<int>("views_per_object", 4)));
    queue_max_ = static_cast<size_t>(std::max<int64_t>(1, declare_parameter<int>("queue_size", 32)));
    prompt_ = declare_parameter<std::string>("prompt", "a photo of {}.");
    emb_file_ = declare_parameter<std::string>("embeddings_file", "");
    if (!emb_file_.empty()) {load(emb_file_);}

    // 결과·물체 목록은 transient_local 이라 프로세스 안 전달(volatile 만 됨)을 끈다 (object_map 과 같음)
    rclcpp::PublisherOptions po;
    po.use_intra_process_comm = rclcpp::IntraProcessSetting::Disable;
    result_pub_ = create_publisher<vision_msgs::msg::Detection3DArray>(
      declare_parameter<std::string>("query_result_topic", "/perception/object_query_result"), rclcpp::QoS(1).transient_local(), po);
    rclcpp::SubscriptionOptions so;
    so.use_intra_process_comm = rclcpp::IntraProcessSetting::Disable;
    objects_sub_ = create_subscription<vision_msgs::msg::Detection3DArray>(
      declare_parameter<std::string>("objects_topic", "/perception/objects"), rclcpp::QoS(1).transient_local(),
      [this](vision_msgs::msg::Detection3DArray::ConstSharedPtr m) {objects_ = m; dirty_ = true;}, so);
    views_sub_ = create_subscription<sensor_msgs::msg::Image>(
      declare_parameter<std::string>("views_topic", "/perception/object_views"), 20,
      [this](sensor_msgs::msg::Image::ConstSharedPtr m) {
        queue_.push_back(m);
        if (queue_.size() > queue_max_) {queue_.pop_front();}
      });
    query_sub_ = create_subscription<std_msgs::msg::String>(
      declare_parameter<std::string>("query_topic", "/perception/object_query"), 10,
      [this](std_msgs::msg::String::ConstSharedPtr m) {query(m->data);});
    save_srv_ = create_service<std_srvs::srv::Trigger>("~/save",
      [this](std_srvs::srv::Trigger::Request::SharedPtr, std_srvs::srv::Trigger::Response::SharedPtr r) {
        r->success = !emb_file_.empty() && save(emb_file_);
        r->message = r->success ? "저장: " + emb_file_ : "embeddings_file 이 비었거나 쓰기 실패";
      });
    save_to_sub_ = create_subscription<std_msgs::msg::String>("~/save_to", 10,
      [this](std_msgs::msg::String::ConstSharedPtr m) {save(m->data);});
    // 사진 묶음은 타이머에서 max_batch 장씩 (받기 콜백을 막지 않게)
    embed_timer_ = create_wall_timer(std::chrono::duration<double>(declare_parameter<double>("embed_period", 0.2)), [this]() {drain();});
    // 글이 살아 있으면 물체 목록·사진 벡터가 바뀐 만큼 다시 셈 (새로 지도에 올라간 물체도 점수)
    rescore_timer_ = create_wall_timer(std::chrono::duration<double>(declare_parameter<double>("rescore_period", 1.0)),
        [this]() {if (active_ && dirty_) {publish_result(false);}});
    RCLCPP_INFO(get_logger(), "물체 찾기 준비: %s, 물체마다 사진 %zu 장, 글 틀 '%s'", net_->describe().c_str(), per_obj_, prompt_.c_str());
  }

private:
  struct View {double q; std::vector<float> e;};   // 품질(픽셀 수) · 단위 벡터

  void drain()
  {
    std::vector<cv::Mat> imgs;
    std::vector<std::string> ids;
    while (!queue_.empty() && static_cast<int>(imgs.size()) < net_->max_batch()) {
      const auto m = queue_.front();
      queue_.pop_front();
      if (m->width < 8 || m->height < 8 || m->header.frame_id.empty()) {continue;}
      const cv::Mat raw(static_cast<int>(m->height), static_cast<int>(m->width), CV_8UC3, const_cast<uint8_t *>(m->data.data()), m->step);
      cv::Mat rgb;
      if (m->encoding == "rgb8") {rgb = raw.clone();} else if (m->encoding == "bgr8") {cv::cvtColor(raw, rgb, cv::COLOR_BGR2RGB);} else {
        RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 5000, "모르는 사진 형식 %s (rgb8·bgr8 만)", m->encoding.c_str());
        continue;
      }
      imgs.push_back(rgb);
      ids.push_back(m->header.frame_id);
    }
    if (imgs.empty()) {return;}
    const auto t0 = std::chrono::steady_clock::now();
    std::vector<std::vector<float>> embs;
    try {   // 같은 프로세스(/app)의 탐사를 같이 죽이지 않게 실패는 로그만
      embs = net_->embed_images(imgs);
    } catch (const std::exception & e) {
      RCLCPP_ERROR_THROTTLE(get_logger(), *get_clock(), 5000, "사진 벡터 실패 (%zu 장 버림): %s", imgs.size(), e.what());
      return;
    }
    const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    for (size_t k = 0; k < embs.size(); ++k) {add_view(ids[k], {static_cast<double>(imgs[k].total()), embs[k]});}
    dirty_ = true;
    RCLCPP_DEBUG(get_logger(), "사진 %zu 장 벡터 %.1f ms", embs.size(), ms);
  }

  // 물체마다 품질(픽셀 수 — 크게 찍힌 사진) 높은 per_obj_ 장만
  void add_view(const std::string & id, View v)
  {
    auto & vs = views_[id];
    vs.push_back(std::move(v));
    std::stable_sort(vs.begin(), vs.end(), [](const View & a, const View & b) {return a.q > b.q;});
    if (vs.size() > per_obj_) {vs.resize(per_obj_);}
  }

  void query(const std::string & text)
  {
    if (text.empty()) {   // 지움: 빈 결과를 남겨 둠 (transient_local — 늦게 붙은 쪽도 지운 것을 받음)
      active_ = false;
      text_.clear();
      q_.clear();
      vision_msgs::msg::Detection3DArray out;
      out.header.stamp = now();
      out.header.frame_id = "map";
      result_pub_->publish(out);
      RCLCPP_INFO(get_logger(), "글 지움");
      return;
    }
    std::string p = prompt_;
    const auto at = p.find("{}");
    p = at == std::string::npos ? text : p.replace(at, 2, text);
    const auto t0 = std::chrono::steady_clock::now();
    try {
      q_ = net_->embed_text(p);
    } catch (const std::exception & e) {
      RCLCPP_ERROR(get_logger(), "글 벡터 실패 '%s': %s", text.c_str(), e.what());
      return;
    }
    text_ = text;
    active_ = true;
    RCLCPP_INFO(get_logger(), "글 '%s' 벡터 %.1f ms", text.c_str(),
      std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
    publish_result(true);
  }

  // objects_ 의 물체마다 하나 (점수순). 물체 목록이 아직 없으면 빈 결과
  void publish_result(bool log)
  {
    dirty_ = false;
    vision_msgs::msg::Detection3DArray out;
    out.header.stamp = now();
    out.header.frame_id = objects_ ? objects_->header.frame_id : "map";
    std::vector<std::vector<float>> es;
    if (objects_) {
      for (const auto & d : objects_->detections) {
        es.clear();
        const auto it = views_.find(d.id);
        if (it != views_.end()) {for (const auto & v : it->second) {es.push_back(v.e);}}
        auto o = d;
        o.results.resize(1);
        o.results[0].hypothesis.score = sem::object_prob(es, q_, scale_, bias_);
        out.detections.push_back(std::move(o));
      }
    }
    std::stable_sort(out.detections.begin(), out.detections.end(),
      [](const vision_msgs::msg::Detection3D & a, const vision_msgs::msg::Detection3D & b) {
        return a.results[0].hypothesis.score > b.results[0].hypothesis.score;
      });
    result_pub_->publish(out);
    if (log) {
      std::string top;
      for (size_t k = 0; k < out.detections.size() && k < 3; ++k) {
        char b[96];
        std::snprintf(b, sizeof b, " %s %.3f", out.detections[k].id.c_str(), out.detections[k].results[0].hypothesis.score);
        top += b;
      }
      RCLCPP_INFO(get_logger(), "글 '%s' → 물체 %zu 개 (사진 있는 물체 %zu) 위:%s", text_.c_str(), out.detections.size(), views_.size(),
        top.c_str());
    }
  }

  // 파일: 줄마다 "물체이름 품질 v0 v1 …"
  bool save(const std::string & path)
  {
    std::ofstream f(path);
    if (!f) {RCLCPP_WARN(get_logger(), "사진 벡터 저장 실패: %s", path.c_str()); return false;}
    size_t n = 0;
    for (const auto & kv : views_) {
      for (const auto & v : kv.second) {
        f << kv.first << ' ' << v.q;
        for (float x : v.e) {f << ' ' << x;}
        f << '\n';
        ++n;
      }
    }
    RCLCPP_INFO(get_logger(), "사진 벡터 저장: %s (%zu 개)", path.c_str(), n);
    return static_cast<bool>(f);
  }
  void load(const std::string & path)
  {
    std::ifstream f(path);
    if (!f) {RCLCPP_WARN(get_logger(), "사진 벡터 파일 없음: %s", path.c_str()); return;}
    std::string id;
    double q;
    while (f >> id >> q) {
      View v{q, std::vector<float>(static_cast<size_t>(net_->dim()))};
      for (auto & x : v.e) {f >> x;}
      if (!f) {break;}
      add_view(id, std::move(v));
    }
    RCLCPP_INFO(get_logger(), "사진 벡터 읽음: %s (물체 %zu 개)", path.c_str(), views_.size());
  }

  std::unique_ptr<Embedder> net_;
  double scale_, bias_;
  size_t per_obj_, queue_max_;
  std::string prompt_, emb_file_, text_;
  std::vector<float> q_;          // 지금 글 벡터 (글이 바뀔 때만 셈)
  bool active_ = false, dirty_ = false;
  std::map<std::string, std::vector<View>> views_;
  std::deque<sensor_msgs::msg::Image::ConstSharedPtr> queue_;
  vision_msgs::msg::Detection3DArray::ConstSharedPtr objects_;
  rclcpp::Publisher<vision_msgs::msg::Detection3DArray>::SharedPtr result_pub_;
  rclcpp::Subscription<vision_msgs::msg::Detection3DArray>::SharedPtr objects_sub_;
  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr views_sub_;
  rclcpp::Subscription<std_msgs::msg::String>::SharedPtr query_sub_, save_to_sub_;
  rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr save_srv_;
  rclcpp::TimerBase::SharedPtr embed_timer_, rescore_timer_;
};

}  // namespace limo_fire_perception

RCLCPP_COMPONENTS_REGISTER_NODE(limo_fire_perception::ObjectSearchNode)
