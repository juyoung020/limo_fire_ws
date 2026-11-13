// BT 노드들이 함께 쓰는 ROS 연결: 노드·TF·토픽 구독(처음 쓸 때 만들고 노드끼리 나눔)·사건 기록·관리자 알림.
// 트리 XML 의 토픽 이름은 실물 기준이다. 시뮬 차이는 설정의 topic_remap·min_hz_override 로 준다 (docs/bt_sim_plan.md 3 절)
#pragma once

#include <deque>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

#include <nav_msgs/msg/occupancy_grid.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <std_msgs/msg/string.hpp>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>
#include <vision_msgs/msg/detection2_d_array.hpp>
#include <vision_msgs/msg/detection3_d_array.hpp>

#include "limo_fire_bt/types.hpp"

namespace limo_fire_bt
{

// 마지막으로 받은 메시지 하나와 받은 시각
template<class T>
struct Latest
{
  typename T::ConstSharedPtr msg;
  rclcpp::Time stamp;
  typename rclcpp::Subscription<T>::SharedPtr sub;
};

// 탐지 한 장 = 받은 시각 + 메시지
struct DetectionFrame
{
  rclcpp::Time stamp;
  vision_msgs::msg::Detection2DArray::ConstSharedPtr msg;
};

// 탐지 하나 (지도 위 물체 목록 /object_mapper/objects 에서)
struct MapObject
{
  std::string cls;
  std::string id;
  double score = 0;
  MapPoint p;
  double size = 0;   // 앞면 넓이의 제곱근 (m). 모르면 0
};

class RosContext
{
public:
  explicit RosContext(rclcpp::Node::SharedPtr node);

  rclcpp::Node::SharedPtr node() const {return node_;}
  rclcpp::Time now() const {return node_->now();}
  rclcpp::Logger logger() const {return node_->get_logger();}

  // 트리에 적힌 이름 → 실제 토픽·서비스·액션 이름 (topic_remap)
  std::string resolve(const std::string & name) const;
  // 트리에 적힌 min_hz 를 설정이 덮어쓰면 그 값 (min_hz_override)
  double min_hz(const std::string & name, double tree_value) const;

  // ---- 토픽 생존: 타입을 몰라도 받는다. 토픽이 아직 없으면 nullopt (다음에 다시 시도) ----
  // window_s 동안 받은 메시지 수 / window_s
  std::optional<double> rate(const std::string & topic, double window_s);
  // 트리를 만들 때 미리 구독해 둔다 (첫 점검 전에 주기를 잴 수 있게). 발행자가 아직 없으면 1 초마다 다시
  void watch(const std::string & topic);

  // ---- 타입이 있는 마지막 메시지 ----
  template<class T>
  typename T::ConstSharedPtr latest(const std::string & topic, const rclcpp::QoS & qos, rclcpp::Time * stamp = nullptr);

  // ---- 탐지 ----
  // window_s 안에 class_name 이 min_score 이상으로 찍힌 장 수와 그중 마지막 장
  int detection_hits(const std::string & topic, const std::string & class_name, double min_score, double window_s,
    DetectionFrame * last = nullptr);
  // class_name 이 min_score 이상으로 마지막에 찍힌 뒤 지난 시간 (s). 탐지 토픽이 hold 동안 안 들어왔으면 nullopt
  std::optional<double> since_last_detection(const std::string & topic, const std::string & class_name,
    double min_score);
  // 지도 위 물체 목록 (object_mapper ~/objects)
  std::vector<MapObject> objects(const std::string & topic, const std::string & class_name, double min_score);

  // ---- 위치 ----
  std::optional<Pose2> robot_pose();
  // 탐지 상자 가운데를 지도 좌표로: 카메라에서 distance_m 앞의 점 (깊이를 못 믿을 때 대략 위치)
  std::optional<MapPoint> approx_location(double u, double distance_m);
  // 상자 폭(px)을 distance_m 에서의 실제 폭(m)으로
  std::optional<double> approx_width(double width_px, double distance_m);

  // ---- 임무 상태 ----
  int create_incident(const MapPoint & where);
  std::optional<MapPoint> incident_location(int id) const;
  // 사건을 처리했다고 적는다 (IncidentManager). 순찰 중 같은 자리 불을 recheck_s 동안 다시 멈추지 않는 데 쓴다
  void mark_incident_handled(int id);
  // 마지막 처리 뒤 지난 시간 (s). 처리한 적 없으면 nullopt
  std::optional<double> since_incident_handled(int id) const;
  bool survivor_search_requested() const {return survivor_search_;}

  // 관리자 알림: /mission/events 에 한 줄 JSON + 로그
  void event(const std::string & json);
  // 최근 알림 (현황판용): (ROS 시각 s, JSON)
  const std::deque<std::pair<double, std::string>> & recent_events() const {return events_;}
  // 노드가 현황판에 띄울 한 줄 (예: "patrol" → "4/13")
  void set_status(const std::string & key, const std::string & value) {status_[key] = value;}
  std::string status(const std::string & key) const
  {
    const auto it = status_.find(key);
    return it == status_.end() ? std::string() : it->second;
  }

  // 설정 값 읽기 (선언은 생성자에서)
  template<class T>
  T param(const std::string & name) const {return node_->get_parameter(name).get_value<T>();}

  tf2_ros::Buffer & tf() {return *tf_buffer_;}

private:
  struct RateMonitor
  {
    rclcpp::GenericSubscription::SharedPtr sub;
    std::deque<rclcpp::Time> stamps;
  };
  bool subscribe_rate(const std::string & name);

  struct DetectionHistory
  {
    rclcpp::Subscription<vision_msgs::msg::Detection2DArray>::SharedPtr sub;
    std::deque<DetectionFrame> frames;
  };

  static bool is_hit(const vision_msgs::msg::Detection2DArray & msg, const std::string & cls, double min_score,
    double * u = nullptr, double * w = nullptr);

  rclcpp::Node::SharedPtr node_;
  std::shared_ptr<tf2_ros::Buffer> tf_buffer_;
  std::shared_ptr<tf2_ros::TransformListener> tf_listener_;
  std::map<std::string, std::string> remap_;
  std::map<std::string, double> min_hz_;
  std::map<std::string, RateMonitor> rates_;
  std::map<std::string, DetectionHistory> detections_;
  std::map<std::string, std::shared_ptr<void>> latest_;
  std::map<int, MapPoint> incidents_;
  std::map<int, rclcpp::Time> incident_handled_;
  int next_incident_ = 1;
  std::deque<std::pair<double, std::string>> events_;
  std::map<std::string, std::string> status_;
  bool survivor_search_ = false;
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr event_pub_;
  rclcpp::ServiceBase::SharedPtr survivor_srv_;
  rclcpp::TimerBase::SharedPtr watch_timer_;
  std::string map_frame_, robot_frame_;
  double history_s_;
};

template<class T>
typename T::ConstSharedPtr RosContext::latest(const std::string & topic, const rclcpp::QoS & qos,
  rclcpp::Time * stamp)
{
  const std::string name = resolve(topic);
  auto it = latest_.find(name);
  if (it == latest_.end()) {
    auto slot = std::make_shared<Latest<T>>();
    std::weak_ptr<Latest<T>> weak = slot;
    rclcpp::Node * n = node_.get();
    slot->sub = node_->create_subscription<T>(name, qos,
        [weak, n](typename T::ConstSharedPtr m) {
          if (auto s = weak.lock()) {s->msg = m; s->stamp = n->now();}
        });
    it = latest_.emplace(name, slot).first;
  }
  auto slot = std::static_pointer_cast<Latest<T>>(it->second);
  if (stamp && slot->msg) {*stamp = slot->stamp;}
  return slot->msg;
}

}  // namespace limo_fire_bt
