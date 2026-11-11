#include "limo_fire_bt/status_overlay.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <map>
#include <vector>

namespace limo_fire_bt
{

namespace
{
// 서브트리 ID → 현황판 이름. 없는 ID 는 그대로 보인다
const std::map<std::string, std::string> kStageNames{
  {"PreflightSensors", "센서 점검"},
  {"CheckDepthAlignment", "깊이 정렬 확인"},
  {"BasicPatrol", "순찰"},
  {"DetectHazard", "불·연기 감시"},
  {"IncidentManager", "사건 판별"},
  {"BasicEmergencyReport", "신고·기록"},
  {"AdvancedModes", "고급 모드"},
  {"SelectAdvancedMode", "고급 모드 고르기"},
  {"SmokeMode", "연기 모드 (대피 안내·훑기)"},
  {"WaterMode", "물 모드 (조준·분사)"},
  {"FrontierMode", "탐사 모드 (요구조자 수색)"},
  {"BasicFallback", "베이직 복귀 (안전거리 감시)"},
};

// 모드 → (한글, 색)
struct ModeStyle {const char * ko; const char * color;};
ModeStyle mode_style(const std::string & m)
{
  if (m == "SMOKE") {return {"연기", "#b0b4bc"};}
  if (m == "WATER") {return {"물", "#4aa3ff"};}
  if (m == "FRONTIER") {return {"탐사", "#c08cff"};}
  return {"베이직", "#5fd38d"};
}

std::string esc(const std::string & s)
{
  std::string o;
  for (char c : s) {
    if (c == '<') {o += "&lt;";} else if (c == '>') {o += "&gt;";} else if (c == '&') {o += "&amp;";} else {o += c;}
  }
  return o;
}

// JSON 한 줄에서 "key":"값" 또는 "key":값 을 꺼낸다 (우리가 낸 알림 형식만)
std::string json_field(const std::string & json, const std::string & key)
{
  const std::string k = "\"" + key + "\":";
  const auto p = json.find(k);
  if (p == std::string::npos) {return {};}
  size_t b = p + k.size();
  if (b < json.size() && json[b] == '"') {
    const auto e = json.find('"', b + 1);
    return json.substr(b + 1, e == std::string::npos ? std::string::npos : e - b - 1);
  }
  const auto e = json.find_first_of(",}", b);
  return json.substr(b, e == std::string::npos ? std::string::npos : e - b);
}
}  // namespace

StatusOverlay::StatusOverlay(std::shared_ptr<RosContext> ctx, BT::Tree & tree)
: ctx_(std::move(ctx)), tree_(tree), next_(std::chrono::steady_clock::now())
{
  // 늦게 켠 RViz 도 바로 보게 마지막 값 유지
  pub_ = ctx_->node()->create_publisher<rviz_2d_overlay_msgs::msg::OverlayText>(
    ctx_->param<std::string>("overlay.topic"), rclcpp::QoS(1).transient_local());
}

std::string StatusOverlay::bb_text(const std::string & key) const
{
  const auto bb = tree_.rootBlackboard();
  if (!bb) {return {};}
  try {
    const auto locked = bb->getAnyLocked(key);
    if (!locked || locked->empty()) {return {};}
    BT::Any copy = *locked.get();   // castPtr 는 const 가 아닌 Any 에만 있다
    BT::Any * any = &copy;
    if (any->isString()) {return any->cast<std::string>();}
    if (const auto * p = any->castPtr<MapPoint>()) {return to_text(*p);}
    if (const auto * b = any->castPtr<bool>()) {return *b ? "true" : "false";}
    if (any->isNumber()) {
      char b[32];
      std::snprintf(b, sizeof(b), "%g", any->cast<double>());
      return b;
    }
  } catch (const std::exception &) {
  }
  return {};
}

std::string StatusOverlay::render() const
{
  // 실행 중인 서브트리(바깥 → 안)와 잎 노드
  std::vector<std::string> stages;
  bool show_reason = false;   // fail_reason 은 사건 판별 때만 지워지므로 실패가 의미 있는 단계에서만 보인다
  std::vector<std::string> leaves;
  for (const auto & st : tree_.subtrees) {
    if (st->nodes.empty()) {continue;}
    if (st->tree_ID != "MainTree" && st->nodes.front()->status() == BT::NodeStatus::RUNNING) {
      const auto it = kStageNames.find(st->tree_ID);
      stages.push_back(it == kStageNames.end() ? st->tree_ID : it->second);
      static const std::vector<std::string> kFailStages{"PreflightSensors", "AdvancedModes", "BasicFallback"};
      if (std::find(kFailStages.begin(), kFailStages.end(), st->tree_ID) != kFailStages.end()) {show_reason = true;}
    }
    for (const auto & n : st->nodes) {
      const auto t = n->type();
      if (n->status() == BT::NodeStatus::RUNNING && (t == BT::NodeType::ACTION || t == BT::NodeType::CONDITION)) {
        leaves.push_back(n->registrationName());
      }
    }
  }

  const auto mode = mode_style(bb_text("active_mode"));
  std::string h = "<div style='font-size:large;'><b>임무 · <span style='color:" + std::string(mode.color) + ";'>" +
    mode.ko + " 모드</span></b></div>";
  h += "<table cellspacing='0' cellpadding='2' width='100%'>";
  auto row = [&h](const std::string & k, const std::string & v, const char * color = "#ffffff") {
      h += "<tr><td style='color:#9aa0a6;'>" + k + "</td><td style='color:" + color + ";'>" + v + "</td></tr>";
    };

  std::string stage;
  for (size_t i = 0; i < stages.size(); ++i) {stage += (i ? " › " : "") + esc(stages[i]);}
  row("단계", stage.empty() ? "-" : stage);
  std::string leaf;
  for (size_t i = 0; i < leaves.size() && i < 3; ++i) {leaf += (i ? ", " : "") + esc(leaves[i]);}
  row("실행 중", leaf.empty() ? "-" : leaf, "#ffd27f");
  const auto patrol = ctx_->status("patrol");
  row("순찰점", patrol.empty() ? "-" : esc(patrol));
  const auto incident = bb_text("incident_id");
  if (!incident.empty() && incident != "0") {
    row("사건", incident + " · 위치 " + esc(bb_text("fire_location")) + (bb_text("reported_119") == "true" || bb_text("reported_119") == "1" ? " · 신고함" : ""),
      "#ff8a6b");
  }
  const auto reason = bb_text("fail_reason");
  if (show_reason && !reason.empty()) {row("실패 사유", esc(reason), "#ff6b6b");}
  const auto history = bb_text("fail_history");
  if (!history.empty()) {row("실패 이력", esc(history), "#ff9e9e");}
  h += "</table>";

  // 최근 알림 (새것부터)
  const auto & ev = ctx_->recent_events();
  const size_t shown = static_cast<size_t>(std::max(0, ctx_->param<int>("overlay.events")));
  if (!ev.empty() && shown > 0) {
    h += "<div style='color:#9aa0a6;margin-top:4px;'>최근 알림</div>";
    const double t_now = ctx_->now().seconds();
    size_t k = 0;
    for (auto it = ev.rbegin(); it != ev.rend() && k < shown; ++it, ++k) {
      const auto name = json_field(it->second, "event");
      const auto why = json_field(it->second, "reason");
      const bool bad = name.find("FAULT") != std::string::npos || name.find("FALLBACK") != std::string::npos ||
        name == "HAZARD_DETECTED";
      char ago[32];
      std::snprintf(ago, sizeof(ago), "%.0f s 전", std::max(0.0, t_now - it->first));
      h += std::string("<div style='color:") + (bad ? "#ffb4a2" : "#d0d4da") + ";'>" + esc(name) +
        (why.empty() ? "" : " — " + esc(why)) + " <span style='color:#6b7075;'>(" + ago + ")</span></div>";
    }
  }
  return h;
}

void StatusOverlay::update()
{
  const auto t = std::chrono::steady_clock::now();
  if (t < next_) {return;}
  next_ = t + std::chrono::duration_cast<std::chrono::steady_clock::duration>(
    std::chrono::duration<double>(ctx_->param<double>("overlay.period_s")));

  rviz_2d_overlay_msgs::msg::OverlayText o;
  o.action = rviz_2d_overlay_msgs::msg::OverlayText::ADD;
  // 물체 현황판(/object_mapper/summary)이 왼쪽 위라 겹치지 않게 왼쪽 아래
  o.horizontal_alignment = rviz_2d_overlay_msgs::msg::OverlayText::LEFT;
  o.vertical_alignment = rviz_2d_overlay_msgs::msg::OverlayText::BOTTOM;
  o.horizontal_distance = 12;
  o.vertical_distance = 12;
  o.width = ctx_->param<int>("overlay.width");
  o.height = ctx_->param<int>("overlay.height");
  o.text_size = 11.0;
  o.font = "Noto Sans CJK KR";
  o.bg_color.r = 0.09f; o.bg_color.g = 0.10f; o.bg_color.b = 0.12f; o.bg_color.a = 0.85f;
  o.fg_color.r = 1.0f; o.fg_color.g = 1.0f; o.fg_color.b = 1.0f; o.fg_color.a = 1.0f;
  o.text = render();
  pub_->publish(o);
}

}  // namespace limo_fire_bt
