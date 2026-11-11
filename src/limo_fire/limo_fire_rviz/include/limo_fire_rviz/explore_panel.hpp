// 운용 화면 탐사 패널: "Auto explore ON/OFF" 버튼 하나로 탐사 노드를 켜고 끈다.
//   버튼 → <노드>/enable (std_srvs/SetBool) — 프론티어 /explore_frontier (limo_fire_rl explore_frontier) · RL /explore_rl (limo_fire_rl explore_rl)
//   상태 ← <노드>/state (idle · running · done, 프론티어만) · <노드>/done (Bool, 둘 다) · <노드>/status (한 줄)
#pragma once
#include <memory>
#include <string>

#include <QComboBox>
#include <QLabel>
#include <QPushButton>
#include <QTimer>
#include <rclcpp/rclcpp.hpp>
#include <rviz_common/panel.hpp>
#include <std_msgs/msg/bool.hpp>
#include <std_msgs/msg/string.hpp>
#include <std_srvs/srv/set_bool.hpp>

namespace limo_fire_rviz {

class ExplorePanel : public rviz_common::Panel {
  Q_OBJECT

 public:
  explicit ExplorePanel(QWidget* parent = nullptr);
  void onInitialize() override;
  void load(const rviz_common::Config& config) override;
  void save(rviz_common::Config config) const override;

 private Q_SLOTS:
  void onToggle();
  void onMode(int index);
  void refresh();

 private:
  void connectNode();
  std::string ns() const;
  bool running() const;

  QComboBox* mode_ = nullptr;
  QPushButton* button_ = nullptr;
  QLabel* state_label_ = nullptr;
  QLabel* detail_label_ = nullptr;
  QTimer* timer_ = nullptr;

  rclcpp::Node::SharedPtr node_;
  rclcpp::Client<std_srvs::srv::SetBool>::SharedPtr client_;
  rclcpp::Subscription<std_msgs::msg::String>::SharedPtr sub_state_, sub_status_;
  rclcpp::Subscription<std_msgs::msg::Bool>::SharedPtr sub_done_;
  std::string state_;          // 노드가 낸 상태 (프론티어). 비면 아래 둘로 셈
  bool done_ = false;          // <노드>/done
  bool enabled_ = false;       // 마지막으로 성공한 요청 (RL 노드는 상태 토픽이 없어 이것으로)
  bool pending_ = false;       // 요청 보내고 답 기다림
  std::string status_, error_;
};

}  // namespace limo_fire_rviz
