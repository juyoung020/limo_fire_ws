// 물체 설명 글로 찾기 (SigLIP2) — "빨간 컵", "소화기 옆 의자" 같은 글에 가장 맞는 지도 위 물체를 점수순으로.
// object_mapper 가 낸 물체별 원본 사진(~/object_views, frame_id = "chair-1")을 SigLIP2 그림 벡터로 바꿔 물체마다 최근 좋은 사진 views_per_object 장을 둔다.
// 글이 오면(query_topic) 글 벡터와 견줘 물체 점수 = 그 물체 사진들 중 가장 높은 sigmoid(logit_scale · cos + logit_bias) — 여러 방향 사진을 평균하면
// 오히려 나빠진다는 보고가 있어(README 출처) 최댓값을 쓴다. 결과: query_result_topic (vision_msgs/Detection3DArray, 점수순, 위치는 /object_mapper/objects 에서).
// 벡터는 embeddings_file 에 저장·읽기 (물체 지도 파일과 같이 — 저장 지도로 다시 돌 때 사진을 다시 안 찍어도 찾음).
#include <algorithm>
#include <chrono>
#include <cmath>
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

#include "limo_fire_perception/siglip_tokenizer.hpp"
#include "limo_fire_perception/trt_siglip.hpp"

namespace limo_fire_perception
{

class ObjectSearchNode : public rclcpp::Node
{
public:
  explicit ObjectSearchNode(const rclcpp::NodeOptions & options)
  : Node("object_search", options)
  {
    // 파일 이름만 쓰면 model_dir(비우면 이 패키지 설치 폴더 models/)에서 찾는다 — 엔진·어휘 파일은 src/…/models 에 두고 colcon build
    std::string share = declare_parameter<std::string>("model_dir", "");
    if (share.empty()) {share = ament_index_cpp::get_package_share_directory("limo_fire_perception") + "/models";}
    auto path = [&](const std::string & p) {return p.empty() || p[0] == '/' ? p : share + "/" + p;};
    tok_ = std::make_unique<SiglipTokenizer>(path(declare_parameter<std::string>("vocab_file", "siglip2_vocab.txt")),
        path(declare_parameter<std::string>("merges_file", "siglip2_merges.txt")));
    net_ = std::make_unique<TrtSiglip>(path(declare_parameter<std::string>("vision_engine", "siglip2_vision.engine")),
        path(declare_parameter<std::string>("text_engine", "siglip2_text_body.engine")),
        path(declare_parameter<std::string>("token_table", "siglip2_tok_emb.f16")));
    // google/siglip2-base-patch16-224: exp(logit_scale) = 112.67, logit_bias = −16.77 (README "SigLIP2" 변환이 출력)
    scale_ = declare_parameter<double>("logit_scale", 112.66889953613281);
    bias_ = declare_parameter<double>("logit_bias", -16.771724700927734);
    per_obj_ = declare_parameter<int>("views_per_object", 4);
    prompt_ = declare_parameter<std::string>("prompt", "a photo of {}.");
    emb_file_ = declare_parameter<std::string>("embeddings_file", "");
    if (!emb_file_.empty()) {load(emb_file_);}

    result_pub_ = create_publisher<vision_msgs::msg::Detection3DArray>(declare_parameter<std::string>("query_result_topic", "/object_mapper/query_result"), 10);
    objects_sub_ = create_subscription<vision_msgs::msg::Detection3DArray>(
      declare_parameter<std::string>("objects_topic", "/object_mapper/objects"), rclcpp::QoS(1).transient_local(),
      [this](vision_msgs::msg::Detection3DArray::ConstSharedPtr m) {objects_ = m;});
    views_sub_ = create_subscription<sensor_msgs::msg::Image>(
      declare_parameter<std::string>("views_topic", "/object_mapper/object_views"), 20,
      [this](sensor_msgs::msg::Image::ConstSharedPtr m) {queue_.push_back(m); if (queue_.size() > 32) {queue_.pop_front();}});
    query_sub_ = create_subscription<std_msgs::msg::String>(declare_parameter<std::string>("query_topic", "/object_mapper/query"), 10,
      [this](std_msgs::msg::String::ConstSharedPtr m) {query(m->data);});
    save_srv_ = create_service<std_srvs::srv::Trigger>("~/save",
      [this](std_srvs::srv::Trigger::Request::SharedPtr, std_srvs::srv::Trigger::Response::SharedPtr r) {
        r->success = !emb_file_.empty() && save(emb_file_);
        r->message = r->success ? "저장: " + emb_file_ : "embeddings_file 이 비었거나 쓰기 실패";
      });
    save_to_sub_ = create_subscription<std_msgs::msg::String>("~/save_to", 10, [this](std_msgs::msg::String::ConstSharedPtr m) {save(m->data);});
    // 사진 묶음은 작업 타이머에서 한 번에 (콜백을 막지 않게, 장 수가 많으면 max_batch 씩)
    timer_ = create_wall_timer(std::chrono::milliseconds(200), [this]() {drain();});
    RCLCPP_INFO(get_logger(), "SigLIP2 준비: 벡터 %d 차원, 물체마다 사진 %d 장, 글 틀 '%s', TensorRT 작업 메모리 %.1f MB (그림·글 같이 씀, 따로면 %.1f MB)",
      net_->dim(), per_obj_, prompt_.c_str(), net_->shared_scratch_bytes() / 1048576.0, net_->separate_scratch_bytes() / 1048576.0);
  }

private:
  struct View {double q; std::vector<float> e;};

  void drain()
  {
    if (queue_.empty()) {return;}
    std::vector<cv::Mat> imgs;
    std::vector<std::string> ids;
    while (!queue_.empty() && static_cast<int>(imgs.size()) < net_->max_batch()) {
      const auto m = queue_.front();
      queue_.pop_front();
      if (m->width < 8 || m->height < 8) {continue;}
      cv::Mat rgb;
      const cv::Mat raw(static_cast<int>(m->height), static_cast<int>(m->width), CV_8UC3, const_cast<uint8_t *>(m->data.data()), m->step);
      if (m->encoding == "rgb8") {rgb = raw.clone();} else if (m->encoding == "bgr8") {cv::cvtColor(raw, rgb, cv::COLOR_BGR2RGB);} else {continue;}
      imgs.push_back(rgb);
      ids.push_back(m->header.frame_id);
    }
    if (imgs.empty()) {return;}
    const auto t0 = std::chrono::steady_clock::now();
    const auto embs = net_->embed_images(imgs);
    const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    for (size_t k = 0; k < embs.size(); ++k) {
      auto & v = views_[ids[k]];
      v.push_back({static_cast<double>(imgs[k].total()), embs[k]});   // 품질 = 픽셀 수 (크게 찍힌 사진 우선)
      std::sort(v.begin(), v.end(), [](const View & a, const View & b) {return a.q > b.q;});
      if (static_cast<int>(v.size()) > per_obj_) {v.resize(static_cast<size_t>(per_obj_));}
    }
    RCLCPP_DEBUG(get_logger(), "사진 %zu 장 벡터 %.1f ms", embs.size(), ms);
  }

  void query(const std::string & text)
  {
    std::string p = prompt_;
    const auto at = p.find("{}");
    p = at == std::string::npos ? text : p.replace(at, 2, text);
    const auto t0 = std::chrono::steady_clock::now();
    const auto q = net_->embed_text(tok_->encode(p));
    std::vector<std::pair<double, std::string>> rank;
    for (const auto & kv : views_) {
      double best = -1e9;
      for (const auto & v : kv.second) {
        double c = 0;
        for (size_t i = 0; i < q.size(); ++i) {c += q[i] * v.e[i];}
        best = std::max(best, 1.0 / (1.0 + std::exp(-(scale_ * c + bias_))));
      }
      rank.push_back({best, kv.first});
    }
    std::sort(rank.rbegin(), rank.rend());
    vision_msgs::msg::Detection3DArray out;
    out.header.stamp = now();
    out.header.frame_id = "map";
    std::string log;
    for (const auto & r : rank) {
      if (!objects_) {break;}
      for (const auto & d : objects_->detections) {
        if (d.id != r.second) {continue;}
        auto o = d;
        o.results.resize(1);
        o.results[0].hypothesis.score = r.first;
        out.detections.push_back(o);
        if (out.detections.size() <= 3) {char b[64]; std::snprintf(b, sizeof b, " %s %.3f", d.id.c_str(), r.first); log += b;}
      }
    }
    result_pub_->publish(out);
    const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    RCLCPP_INFO(get_logger(), "글 '%s' → 물체 %zu 개 중 위:%s (%.1f ms)", text.c_str(), out.detections.size(), log.c_str(), ms);
  }

  // 파일: 줄마다 "물체이름 품질 v0 v1 …"
  bool save(const std::string & path)
  {
    std::ofstream f(path);
    if (!f) {return false;}
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
      views_[id].push_back(v);
    }
    RCLCPP_INFO(get_logger(), "사진 벡터 읽음: %s (물체 %zu 개)", path.c_str(), views_.size());
  }

  std::unique_ptr<SiglipTokenizer> tok_;
  std::unique_ptr<TrtSiglip> net_;
  double scale_, bias_;
  int per_obj_;
  std::string prompt_, emb_file_;
  std::map<std::string, std::vector<View>> views_;
  std::deque<sensor_msgs::msg::Image::ConstSharedPtr> queue_;
  vision_msgs::msg::Detection3DArray::ConstSharedPtr objects_;
  rclcpp::Publisher<vision_msgs::msg::Detection3DArray>::SharedPtr result_pub_;
  rclcpp::Subscription<vision_msgs::msg::Detection3DArray>::SharedPtr objects_sub_;
  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr views_sub_;
  rclcpp::Subscription<std_msgs::msg::String>::SharedPtr query_sub_, save_to_sub_;
  rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr save_srv_;
  rclcpp::TimerBase::SharedPtr timer_;
};

}  // namespace limo_fire_perception

RCLCPP_COMPONENTS_REGISTER_NODE(limo_fire_perception::ObjectSearchNode)
