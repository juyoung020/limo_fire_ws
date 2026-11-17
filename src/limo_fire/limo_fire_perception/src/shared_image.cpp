// shared_image.hpp 의 구현: 프로세스에 하나인 표 (토픽 → 구독 하나 + 받는 노드들)
#include "limo_fire_perception/shared_image.hpp"

#include <algorithm>
#include <map>
#include <mutex>
#include <utility>
#include <vector>

#include <rclcpp/message_memory_strategy.hpp>
#include <rclcpp/rclcpp.hpp>

namespace limo_fire_perception
{

namespace
{
using Image = sensor_msgs::msg::Image;

// 받기 메시지를 돌려 쓰는 메모리 전략: 아무도(받는 노드들의 줄) 안 쥔 메시지에 다음 장을 역직렬화한다 → 영상 버퍼를 장마다 새로 할당하지 않음.
// 다 쥐고 있으면(최대 max 개) 새로 만든다. 실행기 스레드에서만 부른다
class PooledImages : public rclcpp::message_memory_strategy::MessageMemoryStrategy<Image>
{
public:
  explicit PooledImages(size_t max) : max_(max) {}
  std::shared_ptr<Image> borrow_message() override
  {
    for (auto & m : pool_) {
      if (m.use_count() == 1) {return m;}
    }
    if (pool_.size() < max_) {
      pool_.push_back(std::make_shared<Image>());
      return pool_.back();
    }
    return std::make_shared<Image>();
  }
  void return_message(std::shared_ptr<Image> & msg) override {msg.reset();}

private:
  size_t max_;
  std::vector<std::shared_ptr<Image>> pool_;
};
constexpr size_t kPool = 32;   // 물체 지도 동기화 줄 10 + 작업 줄 10 + 탐사 줄 몇 장 + 여유
}  // namespace

struct SharedImageHub
{
  struct Topic
  {
    rclcpp::Subscription<Image>::SharedPtr sub;
    rclcpp::Node * owner = nullptr;                 // 구독을 만든 노드
    std::vector<SharedImageSubscription *> clients; // 붙은 차례
  };
  std::mutex mu;
  std::map<std::string, Topic> topics;

  static SharedImageHub & get()
  {
    static SharedImageHub hub;
    return hub;
  }

  // mu 를 쥔 채 부른다
  void subscribe_on(const std::string & name, Topic & t, rclcpp::Node * node)
  {
    t.owner = node;
    t.sub = node->create_subscription<Image>(
      name, rclcpp::SensorDataQoS(),
      [this, name](Image::ConstSharedPtr msg) {
        std::vector<SharedImageSubscription::Callback> cbs;
        {
          std::lock_guard<std::mutex> lk(mu);
          const auto it = topics.find(name);
          if (it == topics.end()) {return;}
          for (auto * c : it->second.clients) {cbs.push_back(c->cb_);}
        }
        for (const auto & cb : cbs) {cb(msg);}
      },
      rclcpp::SubscriptionOptions(), std::make_shared<PooledImages>(kPool));
  }
};

std::shared_ptr<SharedImageSubscription> SharedImageSubscription::create(
  rclcpp::Node * node, const std::string & topic, Callback cb)
{
  const std::string name = node->get_node_topics_interface()->resolve_topic_name(topic);
  std::shared_ptr<SharedImageSubscription> h(new SharedImageSubscription(node, name, std::move(cb)));
  auto & hub = SharedImageHub::get();
  std::lock_guard<std::mutex> lk(hub.mu);
  auto & t = hub.topics[name];
  t.clients.push_back(h.get());
  if (!t.sub) {
    hub.subscribe_on(name, t, node);
    RCLCPP_INFO(node->get_logger(), "영상 %s: 이 프로세스의 구독 하나를 만듦 (같은 토픽을 받는 다른 노드는 이것을 나눠 씀)", name.c_str());
  } else {
    RCLCPP_INFO(node->get_logger(), "영상 %s: 이 프로세스에 이미 있는 구독(%s)을 나눠 씀 — 받는 노드 %zu 개, 역직렬화 한 번",
      name.c_str(), t.owner->get_fully_qualified_name(), t.clients.size());
  }
  return h;
}

SharedImageSubscription::SharedImageSubscription(rclcpp::Node * node, std::string topic, Callback cb)
: node_(node), topic_(std::move(topic)), cb_(std::move(cb)) {}

SharedImageSubscription::~SharedImageSubscription()
{
  auto & hub = SharedImageHub::get();
  std::lock_guard<std::mutex> lk(hub.mu);
  const auto it = hub.topics.find(topic_);
  if (it == hub.topics.end()) {return;}
  auto & t = it->second;
  t.clients.erase(std::remove(t.clients.begin(), t.clients.end(), this), t.clients.end());
  // 이 노드에 구독이 있었으면 거두고, 남은 노드가 있으면 그 노드에 다시 만든다 (그 노드가 실행기에 있으니 계속 받음)
  const bool remaining_on_this_node = std::any_of(t.clients.begin(), t.clients.end(),
      [this](const SharedImageSubscription * c) {return c->node_ == node_;});
  if (t.owner == node_ && !remaining_on_this_node) {
    t.sub.reset();
    t.owner = nullptr;
    if (!t.clients.empty()) {hub.subscribe_on(topic_, t, t.clients.front()->node_);}
  }
  if (t.clients.empty()) {hub.topics.erase(it);}
}

}  // namespace limo_fire_perception
