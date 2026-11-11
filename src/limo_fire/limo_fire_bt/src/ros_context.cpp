#include "limo_fire_bt/ros_context.hpp"

#include <algorithm>
#include <cmath>

#include <geometry_msgs/msg/point_stamped.hpp>
#include <std_srvs/srv/set_bool.hpp>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>

namespace limo_fire_bt
{

namespace
{
// "a:=b" 목록 → 표
std::map<std::string, std::string> parse_pairs(const std::vector<std::string> & items, const rclcpp::Logger & log)
{
  std::map<std::string, std::string> out;
  for (const auto & s : items) {
    if (s.empty()) {continue;}   // 빈 목록 자리 ([""]: ROS 파라미터는 빈 배열 타입을 못 정함)
    const auto k = s.find(":=");
    if (k == std::string::npos) {
      RCLCPP_ERROR(log, "\"%s\": \"이름:=값\" 형식이 아님 — 무시", s.c_str());
      continue;
    }
    out[s.substr(0, k)] = s.substr(k + 2);
  }
  return out;
}
}  // namespace

RosContext::RosContext(rclcpp::Node::SharedPtr node)
: node_(std::move(node))
{
  tf_buffer_ = std::make_shared<tf2_ros::Buffer>(node_->get_clock());
  tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_, node_, false);

  remap_ = parse_pairs(node_->declare_parameter<std::vector<std::string>>("topic_remap", {""}), logger());
  remap_.erase("");
  for (const auto & [k, v] : parse_pairs(
      node_->declare_parameter<std::vector<std::string>>("min_hz_override", {""}), logger()))
  {
    if (!k.empty()) {min_hz_[k] = std::stod(v);}
  }
  map_frame_ = node_->declare_parameter<std::string>("map_frame", "map");
  robot_frame_ = node_->declare_parameter<std::string>("robot_frame", "base_link");
  history_s_ = node_->declare_parameter<double>("detection_history_s", 15.0);
  // 탐지 좌표(컬러 영상) → 방향. 깊이를 못 믿을 때 대략 위치에 쓴다
  node_->declare_parameter<std::string>("camera_info_topic", "/camera/color/camera_info");

  event_pub_ = node_->create_publisher<std_msgs::msg::String>(
    node_->declare_parameter<std::string>("event_topic", "/mission/events"), rclcpp::QoS(50).reliable());

  // 요구조자 수색 요청: ros2 service call /mission/request_survivor_search std_srvs/srv/SetBool "{data: true}"
  survivor_srv_ = node_->create_service<std_srvs::srv::SetBool>("~/request_survivor_search",
      [this](std_srvs::srv::SetBool::Request::SharedPtr q, std_srvs::srv::SetBool::Response::SharedPtr r) {
        survivor_search_ = q->data;
        r->success = true;
        r->message = q->data ? "요구조자 수색 요청됨" : "요구조자 수색 요청 취소";
        RCLCPP_INFO(logger(), "%s", r->message.c_str());
      });

  // 발행자가 늦게 뜬 토픽도 첫 점검 전에 구독되게
  watch_timer_ = node_->create_wall_timer(std::chrono::seconds(1), [this]() {
      for (const auto & [name, mon] : rates_) {
        if (!mon.sub) {subscribe_rate(name);}
      }
    });
}

std::string RosContext::resolve(const std::string & name) const
{
  const auto it = remap_.find(name);
  return it == remap_.end() ? name : it->second;
}

double RosContext::min_hz(const std::string & name, double tree_value) const
{
  const auto it = min_hz_.find(name);
  return it == min_hz_.end() ? tree_value : it->second;
}

void RosContext::watch(const std::string & topic)
{
  const std::string name = resolve(topic);
  if (rates_.count(name)) {return;}
  rates_.emplace(name, RateMonitor{});
  subscribe_rate(name);
}

bool RosContext::subscribe_rate(const std::string & name)
{
  RateMonitor & mon = rates_.at(name);
  if (mon.sub) {return true;}
  // 타입은 그래프에서 찾는다. 아직 발행자가 없으면 watch_timer_ 가 다시 시도
  const auto all = node_->get_topic_names_and_types();
  const auto t = all.find(name);
  if (t == all.end() || t->second.empty()) {return false;}
  RateMonitor * m = &mon;
  rclcpp::Node * n = node_.get();
  // best effort 구독은 reliable·best effort 발행자 모두와 맞는다
  mon.sub = node_->create_generic_subscription(name, t->second.front(), rclcpp::SensorDataQoS(),
      [m, n](std::shared_ptr<const rclcpp::SerializedMessage>) {
        m->stamps.push_back(n->now());
        if (m->stamps.size() > 2000) {m->stamps.pop_front();}
      });
  return true;
}

std::optional<double> RosContext::rate(const std::string & topic, double window_s)
{
  watch(topic);
  const std::string name = resolve(topic);
  if (!subscribe_rate(name)) {return std::nullopt;}
  auto & st = rates_.at(name).stamps;
  const rclcpp::Time t_now = now();
  const rclcpp::Time from = t_now - rclcpp::Duration::from_seconds(window_s);
  while (!st.empty() && st.front() < t_now - rclcpp::Duration::from_seconds(std::max(window_s, 10.0))) {
    st.pop_front();
  }
  const auto n = std::count_if(st.begin(), st.end(), [&](const rclcpp::Time & s) {return s >= from;});
  return static_cast<double>(n) / window_s;
}

bool RosContext::is_hit(const vision_msgs::msg::Detection2DArray & msg, const std::string & cls, double min_score,
  double * u, double * w)
{
  double best = -1;
  for (const auto & d : msg.detections) {
    for (const auto & r : d.results) {
      if (r.hypothesis.class_id == cls && r.hypothesis.score >= min_score && r.hypothesis.score > best) {
        best = r.hypothesis.score;
        if (u) {*u = d.bbox.center.position.x;}
        if (w) {*w = d.bbox.size_x;}
      }
    }
  }
  return best >= 0;
}

int RosContext::detection_hits(const std::string & topic, const std::string & class_name, double min_score,
  double window_s, DetectionFrame * last)
{
  const std::string name = resolve(topic);
  auto it = detections_.find(name);
  if (it == detections_.end()) {
    it = detections_.emplace(name, DetectionHistory{}).first;
    DetectionHistory * h = &it->second;
    rclcpp::Node * n = node_.get();
    const double keep = history_s_;
    h->sub = node_->create_subscription<vision_msgs::msg::Detection2DArray>(name, rclcpp::SensorDataQoS(),
        [h, n, keep](vision_msgs::msg::Detection2DArray::ConstSharedPtr m) {
          const rclcpp::Time t = n->now();
          h->frames.push_back({t, m});
          while (!h->frames.empty() && (t - h->frames.front().stamp).seconds() > keep) {h->frames.pop_front();}
        });
  }
  const rclcpp::Time from = now() - rclcpp::Duration::from_seconds(window_s);
  int hits = 0;
  for (const auto & f : it->second.frames) {
    if (f.stamp >= from && is_hit(*f.msg, class_name, min_score)) {
      ++hits;
      if (last) {*last = f;}
    }
  }
  return hits;
}

std::optional<double> RosContext::since_last_detection(const std::string & topic, const std::string & class_name,
  double min_score)
{
  detection_hits(topic, class_name, min_score, 0.0);   // 구독 보장
  const auto & frames = detections_.at(resolve(topic)).frames;
  if (frames.empty()) {return std::nullopt;}
  // 탐지가 지금도 들어오는지: 마지막 장이 1 s 넘게 묵었으면 "안 보임" 을 믿을 수 없다
  if ((now() - frames.back().stamp).seconds() > 1.0) {return std::nullopt;}
  for (auto f = frames.rbegin(); f != frames.rend(); ++f) {
    if (is_hit(*f->msg, class_name, min_score)) {return (now() - f->stamp).seconds();}
  }
  return (now() - frames.front().stamp).seconds();   // 기록 안에서는 한 번도 안 보임
}

std::vector<MapObject> RosContext::objects(const std::string & topic, const std::string & class_name,
  double min_score)
{
  std::vector<MapObject> out;
  const auto msg = latest<vision_msgs::msg::Detection3DArray>(topic, rclcpp::QoS(1).reliable().transient_local());
  if (!msg) {return out;}
  for (const auto & d : msg->detections) {
    if (d.results.empty()) {continue;}
    const auto & h = d.results.front().hypothesis;
    if (h.class_id != class_name || h.score < min_score) {continue;}
    MapObject o;
    o.cls = h.class_id;
    o.id = d.id;
    o.score = h.score;
    o.p = {d.bbox.center.position.x, d.bbox.center.position.y, d.bbox.center.position.z};
    o.size = d.bbox.size.y;
    out.push_back(o);
  }
  return out;
}

std::optional<Pose2> RosContext::robot_pose()
{
  try {
    const auto tf = tf_buffer_->lookupTransform(map_frame_, robot_frame_, tf2::TimePointZero);
    const auto & q = tf.transform.rotation;
    return Pose2{tf.transform.translation.x, tf.transform.translation.y,
      std::atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))};
  } catch (const tf2::TransformException & e) {
    RCLCPP_WARN_THROTTLE(logger(), *node_->get_clock(), 5000, "로봇 위치 없음: %s", e.what());
    return std::nullopt;
  }
}

std::optional<MapPoint> RosContext::approx_location(double u, double distance_m)
{
  const auto info = latest<sensor_msgs::msg::CameraInfo>(
    param<std::string>("camera_info_topic"), rclcpp::SensorDataQoS());
  if (!info || info->k[0] <= 0) {return std::nullopt;}
  geometry_msgs::msg::PointStamped cam, map;
  cam.header.frame_id = info->header.frame_id;
  cam.point.x = (u - info->k[2]) * distance_m / info->k[0];
  cam.point.z = distance_m;
  try {
    map = tf_buffer_->transform(cam, map_frame_);
  } catch (const tf2::TransformException & e) {
    RCLCPP_WARN_THROTTLE(logger(), *node_->get_clock(), 5000, "카메라 → 지도 TF 없음: %s", e.what());
    return std::nullopt;
  }
  return MapPoint{map.point.x, map.point.y, map.point.z};
}

std::optional<double> RosContext::approx_width(double width_px, double distance_m)
{
  const auto info = latest<sensor_msgs::msg::CameraInfo>(
    param<std::string>("camera_info_topic"), rclcpp::SensorDataQoS());
  if (!info || info->k[0] <= 0) {return std::nullopt;}
  return width_px * distance_m / info->k[0];
}

int RosContext::create_incident(const MapPoint & where)
{
  const int id = next_incident_++;
  incidents_[id] = where;
  return id;
}

std::optional<MapPoint> RosContext::incident_location(int id) const
{
  const auto it = incidents_.find(id);
  if (it == incidents_.end()) {return std::nullopt;}
  return it->second;
}

void RosContext::event(const std::string & json)
{
  std_msgs::msg::String m;
  m.data = json;
  event_pub_->publish(m);
  RCLCPP_INFO(logger(), "[알림] %s", json.c_str());
}

}  // namespace limo_fire_bt
