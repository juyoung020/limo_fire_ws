// RViz 운용 화면 임무 현황판: 지금 모드·단계·실행 중인 노드·순찰점·사건·실패 사유·최근 알림을
// rviz_2d_overlay_msgs/OverlayText 로 낸다 (/mission/overlay, 마지막 값 유지). 트리는 읽기만 한다
#pragma once

#include <memory>
#include <string>

#include <behaviortree_cpp/bt_factory.h>
#include <rviz_2d_overlay_msgs/msg/overlay_text.hpp>

#include "limo_fire_bt/ros_context.hpp"

namespace limo_fire_bt
{

class StatusOverlay
{
public:
  StatusOverlay(std::shared_ptr<RosContext> ctx, BT::Tree & tree);
  // 주 루프가 부른다. overlay.period_s 마다 한 번 낸다
  void update();

private:
  std::string bb_text(const std::string & key) const;
  std::string render() const;

  std::shared_ptr<RosContext> ctx_;
  BT::Tree & tree_;   // rootBlackboard() 가 const 가 아니라서
  rclcpp::Publisher<rviz_2d_overlay_msgs::msg::OverlayText>::SharedPtr pub_;
  std::chrono::steady_clock::time_point next_;
};

}  // namespace limo_fire_bt
