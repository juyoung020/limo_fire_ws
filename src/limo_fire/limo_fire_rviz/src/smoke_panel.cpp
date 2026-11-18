#include "limo_fire_rviz/smoke_panel.hpp"

#include <cmath>
#include <cstdio>

#include <QVBoxLayout>
#include <pluginlib/class_list_macros.hpp>
#include <rviz_common/display_context.hpp>

namespace limo_fire_rviz {

namespace {
// 노드 이름·인터페이스는 limo_fire_navigation smoke_mode 와 같게 (README "연기 모드")
const char* const kSet = "/smoke_mode/set";
const char* const kActive = "/smoke_mode/active";
const char* const kPose = "/smoke_mode/pose";
}  // namespace

SmokePanel::SmokePanel(QWidget* parent) : rviz_common::Panel(parent) {
  button_ = new QPushButton;
  button_->setCheckable(true);
  button_->setMinimumHeight(36);
  state_label_ = new QLabel("상태: -");
  sensor_label_ = new QLabel;
  sensor_label_->setWordWrap(true);
  detail_label_ = new QLabel;
  detail_label_->setWordWrap(true);
  detail_label_->setStyleSheet("color: gray;");
  auto* lay = new QVBoxLayout;
  lay->addWidget(button_);
  lay->addWidget(state_label_);
  lay->addWidget(sensor_label_);
  lay->addWidget(detail_label_);
  lay->addStretch(1);
  setLayout(lay);
  connect(button_, &QPushButton::clicked, this, &SmokePanel::onToggle);
  timer_ = new QTimer(this);
  connect(timer_, &QTimer::timeout, this, &SmokePanel::refresh);
}

void SmokePanel::onInitialize() {
  node_ = getDisplayContext()->getRosNodeAbstraction().lock()->get_raw_node();
  client_ = node_->create_client<std_srvs::srv::SetBool>(kSet);
  sub_active_ = node_->create_subscription<std_msgs::msg::Bool>(
      kActive, rclcpp::QoS(1).transient_local().reliable(), [this](std_msgs::msg::Bool::ConstSharedPtr m) {
        known_ = true;
        if (active_ != m->data) sigma_xy_ = sigma_yaw_ = -1.0;
        active_ = m->data;
      });
  sub_pose_ = node_->create_subscription<geometry_msgs::msg::PoseWithCovarianceStamped>(
      kPose, 10, [this](geometry_msgs::msg::PoseWithCovarianceStamped::ConstSharedPtr m) {
        const auto& c = m->pose.covariance;   // 6×6 (x y z rx ry rz) 행 우선
        sigma_xy_ = std::sqrt(std::max(0.0, std::max(c[0], c[7])));
        sigma_yaw_ = std::sqrt(std::max(0.0, c[35]));
      });
  timer_->start(500);
  refresh();
}

void SmokePanel::onToggle() {
  if (!client_ || !client_->service_is_ready()) {
    error_ = std::string(kSet) + " 서비스 없음 — 저장 지도로 띄웠나? (SLAM 중에는 연기 모드 노드가 위치를 못 붙잡음)";
    refresh();
    return;
  }
  auto req = std::make_shared<std_srvs::srv::SetBool::Request>();
  req->data = !active_;
  pending_ = true;
  error_.clear();
  client_->async_send_request(req, [this](rclcpp::Client<std_srvs::srv::SetBool>::SharedFuture f) {
    pending_ = false;
    const auto r = f.get();
    // 상태는 /smoke_mode/active 가 바꾼다 — 여기서는 노드의 답만 보여 준다
    if (r->success) message_ = r->message;
    else error_ = "거절: " + r->message;
  });
  refresh();
}

void SmokePanel::refresh() {
  const bool ready = client_ && client_->service_is_ready();
  const bool on = known_ && active_;
  if (!ready) state_label_->setText("상태: 노드 없음 (/smoke_mode)");
  else if (!known_) state_label_->setText("상태: 상태 토픽을 기다림");
  else state_label_->setText(on ? "상태: 연기 모드 — 바퀴 + IMU 로만 주행" : "상태: 평상시 — 라이다로 위치 추정(AMCL)");

  sensor_label_->setText(on ? "라이다: 위치 추정에 안 씀 · 장애물 지도는 저장 지도 벽(가상 스캔)\n"
                              "카메라: 주행에 안 씀\n"
                              "속도 한도 걸림 · 나올 때 추정 자세를 AMCL 에 넘김"
                            : "");

  button_->blockSignals(true);
  button_->setChecked(on);
  button_->blockSignals(false);
  button_->setText(on ? "연기 모드 ON  (누르면 평상시로)" : "연기 모드 OFF  (누르면 켬)");
  button_->setStyleSheet(on ? "background-color: #616161; color: white; font-weight: bold;" : "font-weight: bold;");
  button_->setEnabled(ready && known_ && !pending_);

  std::string d = error_;
  if (d.empty() && on && sigma_xy_ >= 0.0) {
    char buf[96];
    std::snprintf(buf, sizeof(buf), "추정 불확실성 1σ: 위치 %.2f m · 방향 %.1f°", sigma_xy_, sigma_yaw_ * 180.0 / M_PI);
    d = buf;
  }
  if (d.empty()) d = message_;
  detail_label_->setText(QString::fromStdString(d));
}

}  // namespace limo_fire_rviz

PLUGINLIB_EXPORT_CLASS(limo_fire_rviz::SmokePanel, rviz_common::Panel)
