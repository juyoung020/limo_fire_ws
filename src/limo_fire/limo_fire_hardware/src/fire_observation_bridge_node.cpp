// 물체 지도(/perception/objects, vision_msgs/Detection3DArray, map 틀) → 조준 입력 /fire/observations (limo_fire_msgs/FireObservation).
// 최근 max_seen_age_s 안에 카메라로 다시 본 'fire' 하나만 넘긴다. 저장된 물체를 다시 낸 것은 새 관측이 아니다(aim_geometry.hpp select_live_fire).
// 관측 시각 = 물체 지도 메시지 시각 − 마지막으로 본 뒤 지난 초. 불이 여럿이면 target_id 로 고르게 하고 넘기지 않는다.
#include <cmath>
#include <string>

#include "limo_fire_hardware/aim_geometry.hpp"
#include "limo_fire_msgs/msg/fire_observation.hpp"
#include "rclcpp/rclcpp.hpp"
#include "vision_msgs/msg/detection3_d_array.hpp"

namespace limo_fire_hardware {

class FireObservationBridge : public rclcpp::Node {
 public:
  FireObservationBridge() : Node("fire_observation_bridge") {
    target_id_ = declare_parameter<std::string>("target_id", "");
    min_conf_ = declare_parameter<double>("min_confidence", 0.5);
    max_age_ = declare_parameter<double>("max_seen_age_s", 0.5);
    max_delay_ = declare_parameter<double>("max_message_delay_s", 0.2);
    map_frame_ = declare_parameter<std::string>("map_frame", "map");
    fire_class_ = declare_parameter<std::string>("fire_class", "fire");
    if (!(min_conf_ >= 0 && min_conf_ <= 1 && max_age_ > 0 && max_delay_ > 0))
      throw std::invalid_argument("fire_observation_bridge: min_confidence·max_seen_age_s·max_message_delay_s 확인");
    pub_ = create_publisher<limo_fire_msgs::msg::FireObservation>(
        declare_parameter<std::string>("fire_topic", "/fire/observations"), 10);
    sub_ = create_subscription<vision_msgs::msg::Detection3DArray>(
        declare_parameter<std::string>("objects_topic", "/perception/objects"), 10,
        [this](vision_msgs::msg::Detection3DArray::SharedPtr m) { on_objects(*m); });
  }

 private:
  void on_objects(const vision_msgs::msg::Detection3DArray& msg) {
    if (msg.header.frame_id != map_frame_) return warn_once("물체 지도 틀이 " + map_frame_ + " 가 아님");
    const rclcpp::Time stamp(msg.header.stamp, get_clock()->get_clock_type());
    const double delay = (now() - stamp).seconds();
    if (!std::isfinite(delay) || delay < -0.1 || delay > max_delay_) return warn_once("물체 지도 메시지가 늦었거나 미래 시각");
    std::vector<MapObject> objs;
    for (const auto& d : msg.detections) {
      MapObject o;
      o.id = d.id;
      if (!d.results.empty()) {
        o.class_id = d.results[0].hypothesis.class_id;
        o.score = d.results[0].hypothesis.score;
        o.seen_age_s = d.results[0].pose.covariance[35];
      }
      o.position = {d.bbox.center.position.x, d.bbox.center.position.y, d.bbox.center.position.z};
      objs.push_back(o);
    }
    const auto pick = select_live_fire(objs, target_id_, min_conf_, max_age_, fire_class_);
    if (pick.index < 0) return warn_once(pick.reason);
    const auto& d = msg.detections[pick.index];
    limo_fire_msgs::msg::FireObservation o;
    o.header.frame_id = map_frame_;
    o.header.stamp = stamp - rclcpp::Duration::from_seconds(pick.age_s);
    o.id = d.id;
    o.position = d.bbox.center.position;
    o.position_valid = true;
    o.width_m = d.bbox.size.x;
    o.size_valid = std::isfinite(o.width_m) && o.width_m > 0;
    o.confidence = d.results[0].hypothesis.score;
    pub_->publish(o);
    last_issue_.clear();
  }
  void warn_once(const std::string& m) {
    if (m != last_issue_) RCLCPP_WARN(get_logger(), "%s", m.c_str());
    last_issue_ = m;
  }

  std::string target_id_, map_frame_, fire_class_, last_issue_;
  double min_conf_, max_age_, max_delay_;
  rclcpp::Publisher<limo_fire_msgs::msg::FireObservation>::SharedPtr pub_;
  rclcpp::Subscription<vision_msgs::msg::Detection3DArray>::SharedPtr sub_;
};

}  // namespace limo_fire_hardware

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<limo_fire_hardware::FireObservationBridge>());
  rclcpp::shutdown();
  return 0;
}
