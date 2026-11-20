#include "limo_fire_rviz/patrol_panel.hpp"

#include <QHBoxLayout>
#include <QVBoxLayout>
#include <pluginlib/class_list_macros.hpp>
#include <rviz_common/display_context.hpp>

namespace limo_fire_rviz {

namespace {
// 이름은 limo_fire_bt 임무 노드(/mission)·찍어서 순찰(/click_patrol)과 같게 (limo_fire_bt README)
const char* const kSetPatrol = "/mission/set_patrol";
const char* const kActive = "/mission/patrol_active";
const char* const kRoute = "/patrol/route";
const char* const kUndo = "/click_patrol/undo";
const char* const kClear = "/click_patrol/clear";
}  // namespace

PatrolPanel::PatrolPanel(QWidget* parent) : rviz_common::Panel(parent) {
  button_ = new QPushButton;
  button_->setCheckable(true);
  button_->setMinimumHeight(36);
  undo_ = new QPushButton("마지막 점 지우기");
  clear_ = new QPushButton("점 다 지우기");
  state_label_ = new QLabel("상태: -");
  points_label_ = new QLabel;
  points_label_->setWordWrap(true);
  detail_label_ = new QLabel;
  detail_label_->setWordWrap(true);
  detail_label_->setStyleSheet("color: gray;");
  auto* row = new QHBoxLayout;
  row->addWidget(undo_);
  row->addWidget(clear_);
  auto* lay = new QVBoxLayout;
  lay->addWidget(button_);
  lay->addWidget(state_label_);
  lay->addWidget(points_label_);
  lay->addLayout(row);
  lay->addWidget(detail_label_);
  lay->addStretch(1);
  setLayout(lay);
  connect(button_, &QPushButton::clicked, this, &PatrolPanel::onToggle);
  connect(undo_, &QPushButton::clicked, this, &PatrolPanel::onUndo);
  connect(clear_, &QPushButton::clicked, this, &PatrolPanel::onClear);
  timer_ = new QTimer(this);
  connect(timer_, &QTimer::timeout, this, &PatrolPanel::refresh);
}

void PatrolPanel::onInitialize() {
  node_ = getDisplayContext()->getRosNodeAbstraction().lock()->get_raw_node();
  set_patrol_ = node_->create_client<std_srvs::srv::SetBool>(kSetPatrol);
  undo_client_ = node_->create_client<std_srvs::srv::Trigger>(kUndo);
  clear_client_ = node_->create_client<std_srvs::srv::Trigger>(kClear);
  const auto latched = rclcpp::QoS(1).transient_local().reliable();
  sub_active_ = node_->create_subscription<std_msgs::msg::Bool>(kActive, latched, [this](std_msgs::msg::Bool::ConstSharedPtr m) {
    known_ = true;
    active_ = m->data;
  });
  sub_route_ = node_->create_subscription<nav_msgs::msg::Path>(kRoute, latched, [this](nav_msgs::msg::Path::ConstSharedPtr m) {
    points_ = static_cast<int>(m->poses.size());
  });
  timer_->start(500);
  refresh();
}

void PatrolPanel::onToggle() {
  if (!set_patrol_ || !set_patrol_->service_is_ready()) {
    error_ = std::string(kSetPatrol) + " 서비스 없음 — 임무 BT 를 띄웠나? (real.launch.xml bt:=true)";
    refresh();
    return;
  }
  auto req = std::make_shared<std_srvs::srv::SetBool::Request>();
  req->data = !active_;
  pending_ = true;
  error_.clear();
  set_patrol_->async_send_request(req, [this](rclcpp::Client<std_srvs::srv::SetBool>::SharedFuture f) {
    pending_ = false;
    const auto r = f.get();
    // 상태는 /mission/patrol_active 가 바꾼다 — 여기서는 답만 보여 준다
    if (r->success) message_ = r->message;
    else error_ = "거절: " + r->message;
  });
  refresh();
}

void PatrolPanel::callTrigger(const rclcpp::Client<std_srvs::srv::Trigger>::SharedPtr& c, const char* name) {
  if (!c || !c->service_is_ready()) {
    error_ = std::string(name) + " 서비스 없음 — 찍어서 순찰(click_patrol)을 띄웠나?";
    refresh();
    return;
  }
  error_.clear();
  c->async_send_request(std::make_shared<std_srvs::srv::Trigger::Request>(),
                        [this](rclcpp::Client<std_srvs::srv::Trigger>::SharedFuture f) { message_ = f.get()->message; });
}

void PatrolPanel::onUndo() { callTrigger(undo_client_, kUndo); }
void PatrolPanel::onClear() { callTrigger(clear_client_, kClear); }

void PatrolPanel::refresh() {
  const bool ready = set_patrol_ && set_patrol_->service_is_ready();
  const bool on = known_ && active_;
  if (!ready) state_label_->setText("상태: 임무 BT 없음 (/mission)");
  else if (!known_) state_label_->setText("상태: 상태 토픽을 기다림");
  else state_label_->setText(on ? "상태: 순찰 중 — 찍은 점을 차례로 (불을 보면 멈추고 신고)" : "상태: 순찰 꺼짐 — 로봇은 서 있음 (불 감지는 계속)");

  if (points_ < 0) points_label_->setText("순찰점: 찍어서 순찰(click_patrol) 없음");
  else if (points_ == 0) points_label_->setText("순찰점: 없음 — RViz 위쪽 Publish Point 로 지도를 찍기");
  else points_label_->setText(QString("순찰점: %1 개 (지도의 주황 점·번호 — 근처를 다시 찍으면 지움)").arg(points_));

  button_->blockSignals(true);
  button_->setChecked(on);
  button_->blockSignals(false);
  button_->setText(on ? "순찰 ON  (누르면 멈춤)" : "순찰 OFF  (누르면 시작)");
  button_->setStyleSheet(on ? "background-color: #2e7d32; color: white; font-weight: bold;" : "font-weight: bold;");
  // 점이 없을 때 켜도 BT 는 기다릴 뿐이지만, 헷갈리지 않게 점이 있어야 시작 버튼을 살린다 (끄기는 언제나)
  button_->setEnabled(ready && known_ && !pending_ && (on || points_ > 0));
  const bool cp = undo_client_ && undo_client_->service_is_ready();
  undo_->setEnabled(cp && points_ > 0);
  clear_->setEnabled(cp && points_ > 0);

  detail_label_->setText(QString::fromStdString(error_.empty() ? message_ : error_));
}

}  // namespace limo_fire_rviz

PLUGINLIB_EXPORT_CLASS(limo_fire_rviz::PatrolPanel, rviz_common::Panel)
