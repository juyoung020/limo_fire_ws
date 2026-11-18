// 운용 화면 연기 모드 패널: 버튼 하나로 연기 모드(limo_fire_navigation smoke_mode)를 켜고 끈다.
//   버튼 → /smoke_mode/set (std_srvs/SetBool). 켜면 라이다(위치 추정·장애물)와 카메라를 주행에 쓰지 않고 바퀴 + IMU 로만 저장 지도 위를 간다.
//   상태 ← /smoke_mode/active (Bool, transient local) · /smoke_mode/pose (연기 속 추정 자세와 쌓인 불확실성)
#pragma once
#include <memory>
#include <string>

#include <QLabel>
#include <QPushButton>
#include <QTimer>
#include <geometry_msgs/msg/pose_with_covariance_stamped.hpp>
#include <rclcpp/rclcpp.hpp>
#include <rviz_common/panel.hpp>
#include <std_msgs/msg/bool.hpp>
#include <std_srvs/srv/set_bool.hpp>

namespace limo_fire_rviz {

class SmokePanel : public rviz_common::Panel {
  Q_OBJECT

 public:
  explicit SmokePanel(QWidget* parent = nullptr);
  void onInitialize() override;

 private Q_SLOTS:
  void onToggle();
  void refresh();

 private:
  QPushButton* button_ = nullptr;
  QLabel* state_label_ = nullptr;
  QLabel* sensor_label_ = nullptr;
  QLabel* detail_label_ = nullptr;
  QTimer* timer_ = nullptr;

  rclcpp::Node::SharedPtr node_;
  rclcpp::Client<std_srvs::srv::SetBool>::SharedPtr client_;
  rclcpp::Subscription<std_msgs::msg::Bool>::SharedPtr sub_active_;
  rclcpp::Subscription<geometry_msgs::msg::PoseWithCovarianceStamped>::SharedPtr sub_pose_;
  // 콜백(RViz 노드 스레드)은 값만 바꾸고 화면은 refresh(Qt 타이머)가 그린다
  bool known_ = false;     // /smoke_mode/active 를 한 번이라도 받음
  bool active_ = false;
  bool pending_ = false;   // 요청 보내고 답 기다림
  double sigma_xy_ = -1.0, sigma_yaw_ = -1.0;   // 연기 속 추정 불확실성 (1σ, m · rad). 음수 = 아직 없음
  std::string message_, error_;
};

}  // namespace limo_fire_rviz
