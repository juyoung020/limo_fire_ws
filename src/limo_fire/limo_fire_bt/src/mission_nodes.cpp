// 임무 기록 노드: 사건 만들기, 관리자 알림, 119 신고, 위치·사진 기록, 임무 시계
#include <chrono>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <memory>
#include <sstream>
#include <string>

#include <behaviortree_cpp/action_node.h>
#include <sensor_msgs/msg/image.hpp>

#include "limo_fire_bt/nodes.hpp"

namespace limo_fire_bt
{

namespace
{
using BT::InputPort;
using BT::NodeStatus;
using BT::OutputPort;
using BT::PortsList;

class RosSync : public BT::SyncActionNode
{
public:
  RosSync(const std::string & name, const BT::NodeConfig & cfg, std::shared_ptr<RosContext> ctx)
  : BT::SyncActionNode(name, cfg), ctx_(std::move(ctx)) {}

protected:
  std::shared_ptr<RosContext> ctx_;
};

std::string json_escape(const std::string & s)
{
  std::string o;
  for (char c : s) {
    if (c == '"' || c == '\\') {o += '\\';}
    if (c == '\n') {o += "\\n"; continue;}
    o += c;
  }
  return o;
}

// 연결된 포트만 JSON 필드로
class JsonFields
{
public:
  JsonFields(const BT::TreeNode & n) : node_(n) {}
  template<class T>
  void add(const std::string & port, const std::string & key)
  {
    const auto v = node_.getInput<T>(port);
    if (!v) {return;}
    out_ << ",\"" << key << "\":";
    if constexpr (std::is_same_v<T, std::string>) {
      out_ << '"' << json_escape(v.value()) << '"';
    } else if constexpr (std::is_same_v<T, MapPoint>) {
      out_ << "[" << v.value().x << "," << v.value().y << "," << v.value().z << "]";
    } else {
      out_ << v.value();
    }
  }
  std::string str() const {return out_.str();}

private:
  const BT::TreeNode & node_;
  std::ostringstream out_;
};

// 새 사건 번호를 매기고 위치를 기억한다
class CreateIncident : public RosSync
{
public:
  using RosSync::RosSync;
  static PortsList providedPorts() {return {OutputPort<int>("incident"), InputPort<MapPoint>("location")};}
  NodeStatus tick() override
  {
    const auto at = need<MapPoint>(*this, "location");
    const int id = ctx_->create_incident(at);
    setOutput("incident", id);
    RCLCPP_INFO(ctx_->logger(), "새 사건 %d (%s)", id, to_text(at).c_str());
    return NodeStatus::SUCCESS;
  }
};

// 관리자 알림. 지금은 /mission/events 한 줄 JSON (웹 앱 limo_fire_report 가 받을 자리)
class NotifyAdmin : public RosSync
{
public:
  using RosSync::RosSync;
  static PortsList providedPorts()
  {
    return {InputPort<std::string>("event"), InputPort<std::string>("history"), InputPort<int>("incident"),
      InputPort<MapPoint>("location"), InputPort<std::string>("mode"), InputPort<int>("people"),
      InputPort<std::string>("reason")};
  }
  NodeStatus tick() override
  {
    JsonFields f(*this);
    f.add<int>("incident", "incident");
    f.add<MapPoint>("location", "location");
    f.add<std::string>("mode", "mode");
    f.add<std::string>("reason", "reason");
    f.add<std::string>("history", "history");
    f.add<int>("people", "people");
    ctx_->event("{\"event\":\"" + json_escape(need<std::string>(*this, "event")) + "\"" + f.str() + "}");
    return NodeStatus::SUCCESS;
  }
};

// 119 신고. call119.enabled: false(기본)면 실제로 보내지 않고 로그만 남긴다 (시험 중 오신고 방지).
// true 면 call119.topic 에 내는데, 받는 쪽(신고 중계)이 없으면 FAILURE → 트리가 다시 시도
class Call119 : public RosSync
{
public:
  using RosSync::RosSync;
  static PortsList providedPorts() {return {InputPort<int>("incident"), InputPort<MapPoint>("location")};}
  NodeStatus tick() override
  {
    JsonFields f(*this);
    f.add<int>("incident", "incident");
    f.add<MapPoint>("location", "location");
    const std::string json = "{\"event\":\"CALL_119\"" + f.str() + "}";
    if (!ctx_->param<bool>("call119.enabled")) {
      RCLCPP_WARN(ctx_->logger(), "[119 신고 막힘: call119.enabled=false] %s", json.c_str());
      return NodeStatus::SUCCESS;
    }
    if (!pub_) {
      pub_ = ctx_->node()->create_publisher<std_msgs::msg::String>(ctx_->param<std::string>("call119.topic"),
          rclcpp::QoS(10).reliable());
    }
    if (pub_->get_subscription_count() == 0) {
      RCLCPP_ERROR(ctx_->logger(), "119 신고 받을 곳 없음 (%s)", pub_->get_topic_name());
      return NodeStatus::FAILURE;
    }
    std_msgs::msg::String m;
    m.data = json;
    pub_->publish(m);
    ctx_->event(json);
    return NodeStatus::SUCCESS;
  }

private:
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr pub_;
};

// 사건 위치(yaml)와 탐지 결과 영상 한 장(ppm)을 record_dir 에 남긴다
class RecordFireLocation : public RosSync
{
public:
  using RosSync::RosSync;
  static PortsList providedPorts()
  {
    return {InputPort<int>("incident"), InputPort<MapPoint>("location"), InputPort<std::string>("map_frame", "map", ""),
      InputPort<std::string>("image_topic", "/yolo_detector/image", "")};
  }
  NodeStatus tick() override
  {
    const auto img = ctx_->latest<sensor_msgs::msg::Image>(need<std::string>(*this, "image_topic"),
        rclcpp::SensorDataQoS());
    const int id = need<int>(*this, "incident");
    const auto at = need<MapPoint>(*this, "location");
    const auto dir = std::filesystem::path(ctx_->param<std::string>("record_dir"));
    std::error_code ec;
    std::filesystem::create_directories(dir, ec);
    const std::time_t t = std::time(nullptr);
    char stamp[32];
    std::strftime(stamp, sizeof(stamp), "%Y%m%d_%H%M%S", std::localtime(&t));
    // 같은 초에 두 번 기록해도(신고 직후 + 베이직 복귀) 덮어쓰지 않게 순번을 붙인다
    const std::string base = "incident_" + std::to_string(id) + "_" + stamp + "_" + std::to_string(++count_);

    std::ofstream y(dir / (base + ".yaml"));
    if (!y) {
      RCLCPP_ERROR(ctx_->logger(), "기록 못 씀: %s", (dir / (base + ".yaml")).c_str());
      return NodeStatus::FAILURE;
    }
    y << "incident: " << id << "\nframe: " << need<std::string>(*this, "map_frame") << "\nlocation: [" << at.x << ", "
      << at.y << ", " << at.z << "]\nros_time: " << ctx_->now().seconds() << "\n";
    if (img && (img->encoding == "rgb8" || img->encoding == "bgr8") && img->step >= img->width * 3) {
      std::ofstream p(dir / (base + ".ppm"), std::ios::binary);
      p << "P6\n" << img->width << " " << img->height << "\n255\n";
      const bool bgr = img->encoding == "bgr8";
      for (uint32_t r = 0; r < img->height; ++r) {
        const uint8_t * row = img->data.data() + static_cast<size_t>(r) * img->step;
        for (uint32_t c = 0; c < img->width; ++c) {
          const uint8_t * px = row + c * 3;
          const char rgb[3] = {static_cast<char>(px[bgr ? 2 : 0]), static_cast<char>(px[1]),
            static_cast<char>(px[bgr ? 0 : 2])};
          p.write(rgb, 3);
        }
      }
      y << "image: " << base << ".ppm\n";
    } else {
      y << "image: null  # 탐지 영상 없음\n";
    }
    RCLCPP_INFO(ctx_->logger(), "사건 %d 기록: %s", id, (dir / base).c_str());
    return NodeStatus::SUCCESS;
  }


private:
  static inline int count_ = 0;   // 트리 안 RecordFireLocation 노드 모두가 함께 센다
};

// 지금 ROS 시각(s)을 남긴다 (IsWithinTimeLimit 가 씀)
class StartMissionTimer : public RosSync
{
public:
  using RosSync::RosSync;
  static PortsList providedPorts() {return {OutputPort<double>("start_time")};}
  NodeStatus tick() override
  {
    setOutput("start_time", ctx_->now().seconds());
    return NodeStatus::SUCCESS;
  }
};

}  // namespace

void register_mission_nodes(BT::BehaviorTreeFactory & f, const std::shared_ptr<RosContext> & ctx)
{
  f.registerNodeType<CreateIncident>("CreateIncident", ctx);
  f.registerNodeType<NotifyAdmin>("NotifyAdmin", ctx);
  f.registerNodeType<Call119>("Call119", ctx);
  f.registerNodeType<RecordFireLocation>("RecordFireLocation", ctx);
  f.registerNodeType<StartMissionTimer>("StartMissionTimer", ctx);
}

}  // namespace limo_fire_bt
