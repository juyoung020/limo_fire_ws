// 운용 화면 순찰 패널: 임무 BT 순찰을 버튼으로 켜고 끄고, RViz 에서 찍은 순찰점을 지운다.
//   순찰 버튼 → /mission/set_patrol (std_srvs/SetBool). 켜면 임무 BT 가 찍은 점을 차례로 돌고(탐사는 BT 가 끔), 끄면 가던 목표를 취소하고 선다.
//   상태 ← /mission/patrol_active (Bool, transient local) · 순찰점 ← /patrol/route (nav_msgs/Path, 찍어서 순찰 click_patrol 이 냄)
//   점 찍기는 RViz 위쪽 Publish Point — 지우기 버튼 → /click_patrol/undo · /click_patrol/clear (std_srvs/Trigger)
#pragma once
#include <memory>
#include <string>

#include <QLabel>
#include <QPushButton>
#include <QTimer>
#include <nav_msgs/msg/path.hpp>
#include <rclcpp/rclcpp.hpp>
#include <rviz_common/panel.hpp>
#include <std_msgs/msg/bool.hpp>
#include <std_srvs/srv/set_bool.hpp>
#include <std_srvs/srv/trigger.hpp>

namespace limo_fire_rviz {

class PatrolPanel : public rviz_common::Panel {
  Q_OBJECT

 public:
  explicit PatrolPanel(QWidget* parent = nullptr);
  void onInitialize() override;

 private Q_SLOTS:
  void onToggle();
  void onUndo();
  void onClear();
  void refresh();

 private:
  void callTrigger(const rclcpp::Client<std_srvs::srv::Trigger>::SharedPtr& c, const char* name);

  QPushButton* button_ = nullptr;
  QPushButton* undo_ = nullptr;
  QPushButton* clear_ = nullptr;
  QLabel* state_label_ = nullptr;
  QLabel* points_label_ = nullptr;
  QLabel* detail_label_ = nullptr;
  QTimer* timer_ = nullptr;

  rclcpp::Node::SharedPtr node_;
  rclcpp::Client<std_srvs::srv::SetBool>::SharedPtr set_patrol_;
  rclcpp::Client<std_srvs::srv::Trigger>::SharedPtr undo_client_, clear_client_;
  rclcpp::Subscription<std_msgs::msg::Bool>::SharedPtr sub_active_;
  rclcpp::Subscription<nav_msgs::msg::Path>::SharedPtr sub_route_;
  // 콜백(RViz 노드 스레드)은 값만 바꾸고 화면은 refresh(Qt 타이머)가 그린다
  bool known_ = false;     // /mission/patrol_active 를 한 번이라도 받음
  bool active_ = false;
  bool pending_ = false;
  int points_ = -1;        // 찍은 순찰점 수, -1 = 아직 모름
  std::string message_, error_;
};

}  // namespace limo_fire_rviz
