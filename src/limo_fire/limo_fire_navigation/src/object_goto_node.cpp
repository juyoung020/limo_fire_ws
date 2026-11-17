// "○○ 로 가" (저장 지도 주행 — navigation.launch.xml: map_server + AMCL + Nav2): 물체 지도(/perception/objects)에서 물체를 찾아 그 옆 설 자리로 Nav2 NavigateToPose.
//
//   ros2 topic pub --once /explore/goal std_msgs/msg/String "{data: '소화기'}"     # 탐사 중(app.launch.xml)과 같은 토픽 — 둘을 같이 띄우지 않는다
//
// 찾기: 글 → 클래스는 explorer 와 같은 코드(limo_fire_explore/class_alias.h)·같은 별칭(limo_fire_explore config/explorer.yaml 의 /explore/** aliases).
//   같은 클래스가 여럿이면 로봇에서 가장 가까운 것. 지도에 없으면 물체 지도가 바뀔 때마다 다시 찾아 나타나는 즉시 그리로 간다.
// 설 자리: 물체 둘레 stand_dist ~ stand_dist + stand_band 고리의 지도 칸 가운데, 로봇 반경(robot_radius) 안이 모두 빈 칸이고
//   로봇에서 빈 칸으로 이어지는(BFS) 칸 중 로봇에서 길이 가장 짧은 칸. 방향은 물체를 봄.
// 빈 글을 주면 지시를 지움. 상태: /explore/goal_status (std_msgs/String, 한글 한 줄)
// (탐사 중 "○○ 로 가" 는 explorer 가 직접 받는다 — 2026-11-18 에 이 노드의 탐사 모드를 explorer 로 옮김)
#include <algorithm>
#include <chrono>
#include <cmath>
#include <deque>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include "geometry_msgs/msg/pose_stamped.hpp"
#include "nav2_msgs/action/navigate_to_pose.hpp"
#include "nav_msgs/msg/occupancy_grid.hpp"
#include "rclcpp/rclcpp.hpp"
#include "rclcpp_action/rclcpp_action.hpp"
#include "std_msgs/msg/string.hpp"
#include "tf2_ros/buffer.h"
#include "tf2_ros/transform_listener.h"
#include "limo_fire_explore/class_alias.h"
#include "vision_msgs/msg/detection3_d_array.hpp"

namespace {

using Nav = nav2_msgs::action::NavigateToPose;

class ObjectGoto : public rclcpp::Node {
 public:
  ObjectGoto() : Node("object_goto") {
    map_frame_ = declare_parameter<std::string>("map_frame", "map");
    robot_frame_ = declare_parameter<std::string>("robot_frame", "base_link");
    stand_dist_ = declare_parameter<double>("stand_dist", 0.6);
    stand_band_ = declare_parameter<double>("stand_band", 0.6);
    robot_radius_ = declare_parameter<double>("robot_radius", 0.25);
    occ_threshold_ = declare_parameter<int>("occupied_threshold", 50);
    aliases_ = limo_fire_explore::parse_aliases(declare_parameter<std::vector<std::string>>("aliases", std::vector<std::string>{}));
    status_pub_ = create_publisher<std_msgs::msg::String>("/explore/goal_status", rclcpp::QoS(1).transient_local());
    objects_sub_ = create_subscription<vision_msgs::msg::Detection3DArray>(
      "/perception/objects", rclcpp::QoS(1).transient_local(),
      [this](vision_msgs::msg::Detection3DArray::ConstSharedPtr m) {
        const size_t before = objects_ ? objects_->detections.size() : 0;
        objects_ = m;
        if (searching_ && m->detections.size() != before) {try_name();}   // 새 물체가 올라오면 바로 다시 찾음
      });
    goal_sub_ = create_subscription<std_msgs::msg::String>("/explore/goal", 10,
      [this](std_msgs::msg::String::ConstSharedPtr m) {on_goal(m->data);});
    tf_buffer_ = std::make_unique<tf2_ros::Buffer>(get_clock());
    tf_listener_ = std::make_unique<tf2_ros::TransformListener>(*tf_buffer_);
    map_sub_ = create_subscription<nav_msgs::msg::OccupancyGrid>(
      "/map", rclcpp::QoS(1).transient_local().reliable(),
      [this](nav_msgs::msg::OccupancyGrid::ConstSharedPtr m) {map_ = m;});
    nav_ = rclcpp_action::create_client<Nav>(this, "/navigate_to_pose");
    say("대기 — /explore/goal 에 지시를 주세요 (예: 소화기)");
  }

 private:
  void say(const std::string & s) {
    std_msgs::msg::String m;
    m.data = s;
    status_pub_->publish(m);
    RCLCPP_INFO(get_logger(), "%s", s.c_str());
  }

  std::optional<std::pair<double, double>> robot_xy() {
    try {
      const auto tf = tf_buffer_->lookupTransform(map_frame_, robot_frame_, tf2::TimePointZero);
      return std::make_pair(tf.transform.translation.x, tf.transform.translation.y);
    } catch (const tf2::TransformException &) {return std::nullopt;}
  }

  std::string class_of(const std::string & text) const {
    std::vector<std::string> known;
    if (objects_) {
      for (const auto & d : objects_->detections) {
        if (!d.results.empty()) {known.push_back(d.results[0].hypothesis.class_id);}
      }
    }
    return limo_fire_explore::class_of(text, aliases_, known);
  }

  void on_goal(const std::string & text) {
    goal_text_ = text;
    searching_ = false;
    if (text.empty()) {
      say("지시 지움");
      return;
    }
    if (!try_name()) {
      searching_ = true;
      say("'" + goal_text_ + "' — 지도에 없음. 물체 지도에 나타나면 바로 감");
    }
  }

  // 클래스 이름·별칭으로 찾아 가면 true
  bool try_name() {
    const std::string cls = class_of(goal_text_);
    if (!cls.empty() && objects_) {
      const auto r = robot_xy();
      const vision_msgs::msg::Detection3D * best = nullptr;
      double bd = std::numeric_limits<double>::max();
      for (const auto & d : objects_->detections) {
        if (d.results.empty() || d.results[0].hypothesis.class_id != cls) {continue;}
        const double dd = r ? std::hypot(d.bbox.center.position.x - r->first, d.bbox.center.position.y - r->second) : 0.0;
        if (dd < bd) {bd = dd; best = &d;}
      }
      if (best) {searching_ = false; go_to(*best, "클래스 " + cls); return true;}
    }
    return false;
  }

  // 물체 둘레 고리에서 설 자리를 고른다 (로봇에서 빈 칸으로 이어지고 반경 안이 빈 칸). 못 찾으면 nullopt
  std::optional<std::pair<double, double>> stand_spot(double ox, double oy) {
    if (!map_) {return std::nullopt;}
    const auto & info = map_->info;
    const int W = static_cast<int>(info.width), H = static_cast<int>(info.height);
    const double res = info.resolution, x0 = info.origin.position.x, y0 = info.origin.position.y;
    auto free_cell = [&](int c) {const int8_t v = map_->data[static_cast<size_t>(c)]; return v >= 0 && v < occ_threshold_;};
    const int rr = static_cast<int>(std::ceil(robot_radius_ / res));
    auto clear = [&](int cx, int cy) {
      for (int dy = -rr; dy <= rr; ++dy) {
        for (int dx = -rr; dx <= rr; ++dx) {
          if (dx * dx + dy * dy > rr * rr) {continue;}
          const int x = cx + dx, y = cy + dy;
          if (x < 0 || y < 0 || x >= W || y >= H || !free_cell(y * W + x)) {return false;}
        }
      }
      return true;
    };
    const auto r = robot_xy();
    if (!r) {return std::nullopt;}
    const int rx = static_cast<int>((r->first - x0) / res), ry = static_cast<int>((r->second - y0) / res);
    if (rx < 0 || ry < 0 || rx >= W || ry >= H) {return std::nullopt;}
    // 로봇에서 빈 칸 BFS (로봇 반경을 지킬 수 있는 칸만; 출발 칸은 예외)
    std::vector<int> dist(static_cast<size_t>(W) * H, -1);
    std::deque<int> q;
    dist[static_cast<size_t>(ry * W + rx)] = 0;
    q.push_back(ry * W + rx);
    while (!q.empty()) {
      const int c = q.front();
      q.pop_front();
      const int cx = c % W, cy = c / W;
      static const int d4[4][2] = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}};
      for (const auto & d : d4) {
        const int x = cx + d[0], y = cy + d[1];
        if (x < 0 || y < 0 || x >= W || y >= H) {continue;}
        const int n = y * W + x;
        if (dist[static_cast<size_t>(n)] >= 0 || !clear(x, y)) {continue;}
        dist[static_cast<size_t>(n)] = dist[static_cast<size_t>(c)] + 1;
        q.push_back(n);
      }
    }
    int best = -1, bd = std::numeric_limits<int>::max();
    const double r_in = stand_dist_, r_out = stand_dist_ + stand_band_;
    const int ext = static_cast<int>(std::ceil(r_out / res));
    const int ocx = static_cast<int>((ox - x0) / res), ocy = static_cast<int>((oy - y0) / res);
    for (int dy = -ext; dy <= ext; ++dy) {
      for (int dx = -ext; dx <= ext; ++dx) {
        const double rd = std::hypot(dx, dy) * res;
        if (rd < r_in || rd > r_out) {continue;}
        const int x = ocx + dx, y = ocy + dy;
        if (x < 0 || y < 0 || x >= W || y >= H) {continue;}
        // 물체에 가까운 고리(0.1 m 띠)부터, 같은 띠면 로봇에서 길이 짧은 칸 — 갈 수 있으면 물체 바로 옆에 선다
        const int dd = dist[static_cast<size_t>(y * W + x)];
        const int key = static_cast<int>((rd - r_in) / 0.1) * 1000000 + dd;
        if (dd >= 0 && key < bd) {bd = key; best = y * W + x;}
      }
    }
    if (best < 0) {return std::nullopt;}
    return std::make_pair(x0 + (best % W + 0.5) * res, y0 + (best / W + 0.5) * res);
  }

  void go_to(const vision_msgs::msg::Detection3D & d, const std::string & how) {
    searching_ = false;   // 찾았으면 다시 찾기를 멈춤 (도착·새 지시까지 "지도에 없음" 으로 안 돌아감)
    const double ox = d.bbox.center.position.x, oy = d.bbox.center.position.y;
    const auto spot = stand_spot(ox, oy);
    if (!spot) {
      say("'" + goal_text_ + "' → " + d.id + " (" + how + ") 를 찾았지만 지도·위치가 없거나 설 자리가 없음");
      return;
    }
    char b[256];
    std::snprintf(b, sizeof b, "'%s' → %s (%s) 물체 (%.2f, %.2f) · 설 자리 (%.2f, %.2f) 로 Nav2 주행", goal_text_.c_str(), d.id.c_str(),
      how.c_str(), ox, oy, spot->first, spot->second);
    say(b);
    if (!nav_->wait_for_action_server(std::chrono::seconds(2))) {say("Nav2 navigate_to_pose 가 없음 (navigation.launch.xml 을 켜 두세요)"); return;}
    Nav::Goal g;
    g.pose.header.frame_id = map_frame_;
    g.pose.header.stamp = now();
    g.pose.pose.position.x = spot->first;
    g.pose.pose.position.y = spot->second;
    const double yaw = std::atan2(oy - spot->second, ox - spot->first);   // 물체를 봄
    g.pose.pose.orientation.z = std::sin(yaw / 2);
    g.pose.pose.orientation.w = std::cos(yaw / 2);
    rclcpp_action::Client<Nav>::SendGoalOptions o;
    const std::string id = d.id;
    o.result_callback = [this, id](const rclcpp_action::ClientGoalHandle<Nav>::WrappedResult & r) {
      say(std::string(r.code == rclcpp_action::ResultCode::SUCCEEDED ? "도착: " : "주행 못 끝냄: ") + id);
    };
    nav_->async_send_goal(g, o);
  }

  std::unique_ptr<tf2_ros::Buffer> tf_buffer_;
  std::unique_ptr<tf2_ros::TransformListener> tf_listener_;
  std::string map_frame_, robot_frame_, goal_text_;
  double stand_dist_, stand_band_, robot_radius_;
  int occ_threshold_;
  limo_fire_explore::Aliases aliases_;
  bool searching_ = false;
  vision_msgs::msg::Detection3DArray::ConstSharedPtr objects_;
  nav_msgs::msg::OccupancyGrid::ConstSharedPtr map_;
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr status_pub_;
  rclcpp::Subscription<vision_msgs::msg::Detection3DArray>::SharedPtr objects_sub_;
  rclcpp::Subscription<nav_msgs::msg::OccupancyGrid>::SharedPtr map_sub_;
  rclcpp::Subscription<std_msgs::msg::String>::SharedPtr goal_sub_;
  rclcpp_action::Client<Nav>::SharedPtr nav_;
};

}  // namespace

int main(int argc, char ** argv) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<ObjectGoto>());
  rclcpp::shutdown();
  return 0;
}
