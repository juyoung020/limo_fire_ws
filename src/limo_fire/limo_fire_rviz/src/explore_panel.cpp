#include "limo_fire_rviz/explore_panel.hpp"

#include <QVBoxLayout>
#include <pluginlib/class_list_macros.hpp>
#include <rviz_common/display_context.hpp>

namespace limo_fire_rviz {

namespace {
// 합친 탐사 노드 /explore/explorer 의 입출력 앞머리 — 주 노드라 이름공간 /explore 바로 아래 (docs/conventions.md "이름")
const char* const kNs = "/explore";
}  // namespace

ExplorePanel::ExplorePanel(QWidget* parent) : rviz_common::Panel(parent) {
  button_ = new QPushButton("Auto explore OFF");
  button_->setCheckable(true);
  button_->setMinimumHeight(36);
  state_label_ = new QLabel("상태: -");
  detail_label_ = new QLabel;
  detail_label_->setWordWrap(true);
  detail_label_->setStyleSheet("color: gray;");
  auto* lay = new QVBoxLayout;
  lay->addWidget(new QLabel("탐사 (/explore/explorer)"));
  lay->addWidget(button_);
  lay->addWidget(state_label_);
  lay->addWidget(detail_label_);
  lay->addStretch(1);
  setLayout(lay);
  connect(button_, &QPushButton::clicked, this, &ExplorePanel::onToggle);
  timer_ = new QTimer(this);
  connect(timer_, &QTimer::timeout, this, &ExplorePanel::refresh);
}

void ExplorePanel::onInitialize() {
  node_ = getDisplayContext()->getRosNodeAbstraction().lock()->get_raw_node();
  connectNode();
  timer_->start(500);
}

void ExplorePanel::connectNode() {
  if (!node_) return;
  state_.clear();
  status_.clear();
  error_.clear();
  done_ = enabled_ = pending_ = false;
  const std::string n = kNs;
  client_ = node_->create_client<std_srvs::srv::SetBool>(n + "/enable");
  const auto latched = rclcpp::QoS(1).transient_local();
  // 콜백은 RViz 가 노드를 돌리는 스레드에서 오므로 값만 바꾸고 화면은 refresh(Qt 타이머)가 그린다
  sub_state_ = node_->create_subscription<std_msgs::msg::String>(n + "/state", latched, [this](std_msgs::msg::String::ConstSharedPtr m) { state_ = m->data; });
  sub_done_ = node_->create_subscription<std_msgs::msg::Bool>(n + "/done", latched, [this](std_msgs::msg::Bool::ConstSharedPtr m) { done_ = m->data; });
  sub_status_ = node_->create_subscription<std_msgs::msg::String>(n + "/status", 10, [this](std_msgs::msg::String::ConstSharedPtr m) { status_ = m->data; });
  refresh();
}

bool ExplorePanel::running() const {
  if (!state_.empty()) return state_ == "running";
  return enabled_ && !done_;
}

void ExplorePanel::onToggle() {
  if (!client_ || !client_->service_is_ready()) {
    error_ = std::string(kNs) + "/enable 서비스 없음 — 탐사 노드를 띄웠나? (README)";
    refresh();
    return;
  }
  auto req = std::make_shared<std_srvs::srv::SetBool::Request>();
  req->data = !running();
  pending_ = true;
  error_.clear();
  const bool want = req->data;
  client_->async_send_request(req, [this, want](rclcpp::Client<std_srvs::srv::SetBool>::SharedFuture f) {
    pending_ = false;
    const auto r = f.get();
    if (r->success) {
      enabled_ = want;
      if (want) done_ = false;
    } else {
      error_ = "거절: " + r->message;
    }
  });
  refresh();
}

void ExplorePanel::refresh() {
  const bool ready = client_ && client_->service_is_ready();
  const bool on = running();
  std::string st;
  if (!ready) st = "노드 없음";
  else if (!state_.empty()) st = state_;
  else st = done_ ? "done" : (enabled_ ? "running" : "idle");
  const char* ko = st == "running" ? "탐사 중" : st == "done" ? "끝 (다시 누르면 처음부터)" : st == "idle" ? "멈춤" : st == "paused" ? "멈춤 (리모컨·닿음 — 상세 보기)" : "서비스를 기다림";
  state_label_->setText(QString::fromStdString("상태: " + st + " — ") + QString::fromUtf8(ko));
  button_->blockSignals(true);
  button_->setChecked(on);
  button_->blockSignals(false);
  button_->setText(on ? "Auto explore ON  (누르면 멈춤)" : "Auto explore OFF  (누르면 시작)");
  button_->setStyleSheet(on ? "background-color: #2e7d32; color: white; font-weight: bold;" : "font-weight: bold;");
  button_->setEnabled(ready && !pending_);
  const std::string d = !error_.empty() ? error_ : status_;
  detail_label_->setText(QString::fromStdString(d));
}

}  // namespace limo_fire_rviz

PLUGINLIB_EXPORT_CLASS(limo_fire_rviz::ExplorePanel, rviz_common::Panel)
