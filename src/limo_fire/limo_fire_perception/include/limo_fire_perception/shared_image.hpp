// 한 프로세스(컨테이너 /app) 안 여러 노드가 같은 영상 토픽을 받을 때 구독을 하나만 만든다:
// 다른 프로세스(카메라)에서 온 영상은 구독마다 역직렬화되므로(614 KB 깊이 × 30 장/초), 한 번 받아 같은 ConstSharedPtr 을 차례로 넘긴다.
// ROS 토픽을 새로 내지 않는다(중계 노드 아님) — 프로세스 안 함수 부름이다.
//
//   depth_ = limo_fire_perception::SharedImageSubscription::create(this, "/camera/depth/image_raw",
//       [this](const sensor_msgs::msg::Image::ConstSharedPtr & m) {...});
//
// - QoS 는 SensorDataQoS (BEST_EFFORT / VOLATILE / 5) 하나. 받기 메시지는 32 개를 돌려 써서 장마다 큰 할당을 하지 않는다.
// - 구독은 처음 부른 노드에 만들어진다. 그 노드가 빠지면(핸들이 사라지면) 남은 노드에 다시 만든다.
// - 콜백은 붙은 차례대로 같은 실행기 스레드에서 부른다. 콜백이 메시지를 고치면 안 된다(const).
// 이 라이브러리는 공유 라이브러리 하나(liblimo_fire_shared_image.so)라 한 프로세스에 표가 하나다.
#pragma once

#include <functional>
#include <memory>
#include <string>

#include <rclcpp/node.hpp>
#include <sensor_msgs/msg/image.hpp>

namespace limo_fire_perception
{

class SharedImageSubscription
{
public:
  using Callback = std::function<void (const sensor_msgs::msg::Image::ConstSharedPtr &)>;

  // node 가 핸들보다 오래 살아야 한다 (노드 멤버로 쥐면 됨)
  static std::shared_ptr<SharedImageSubscription> create(rclcpp::Node * node, const std::string & topic, Callback cb);
  ~SharedImageSubscription();

  SharedImageSubscription(const SharedImageSubscription &) = delete;
  SharedImageSubscription & operator=(const SharedImageSubscription &) = delete;

private:
  SharedImageSubscription(rclcpp::Node * node, std::string topic, Callback cb);
  rclcpp::Node * node_;
  std::string topic_;   // 풀어 쓴 이름 (/camera/depth/image_raw)
  Callback cb_;
  friend struct SharedImageHub;
};

}  // namespace limo_fire_perception
