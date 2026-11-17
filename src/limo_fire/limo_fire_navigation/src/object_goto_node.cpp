// "○○ 로 가" — 물체 지도(limo_fire_perception object_mapper ~/objects)에서 물체를 찾아 그 옆 설 자리로 간다.
//
//   ros2 topic pub --once /object_goto/goal std_msgs/msg/String "{data: '소화기'}"     # 이름(한글 별칭·영문 클래스) 또는 설명 글
//
// 찾기 (차례대로):
//   1. 클래스 이름: 글이 class 이름(영문, '_' 대신 띄어쓰기도)이나 aliases 의 한글 이름과 같으면 그 클래스
//   2. 설명 글 (SigLIP2, object_mapper 에 글 검색이 켜져 있을 때): query_topic 에 글을 내고 query_result_topic(점수순 Detection3DArray)을 기다림
//   같은 클래스가 여럿이면 로봇에서 가장 가까운 것.
// 설 자리: 물체 둘레 stand_dist ~ stand_dist + stand_band 고리의 지도 칸 가운데, 로봇 반경(robot_radius) 안이 모두 빈 칸이고
//   로봇에서 빈 칸으로 이어지는(BFS) 칸 중 로봇에서 길이 가장 짧은 칸. 방향은 물체를 봄.
// 가는 길 (mode):
//   nav2 — 저장 지도 위 주행(navigation.launch.xml: map_server + AMCL + Nav2): NavigateToPose 액션
//   explore — 탐사 중(explore_fuel_rl.launch.xml: Cartographer + low_rl, Nav2 없음): 지시 글(~/instruction)과 풀어 낸 클래스(~/target_class)만 low_rl 에 넘기고,
//             판단 모델이 후보 목록(그 물체 옆 설 자리 + 프론티어)에서 고른다 (low_decider.h 물체 후보). direct: true 면 예전처럼 설 자리를 ~/goto 로 바로
// 지도에 없으면: explore_topic(low_rl ~/instruction)에 글을 넘겨 탐사로 찾게 하고(판단 모델 글에 지시 + 아는 물체가 들어감),
//   물체 지도가 바뀔 때마다(이름) · requery_s 마다(설명 검색) 다시 찾아 나타나는 즉시 그리로 간다. 빈 글을 주면 지시를 지움.
// 상태: ~/status (std_msgs/String, 한글 한 줄)
#include <algorithm>
#include <chrono>
#include <cmath>
#include <deque>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "geometry_msgs/msg/point_stamped.hpp"
#include "geometry_msgs/msg/pose_stamped.hpp"
#include "nav2_msgs/action/navigate_to_pose.hpp"
#include "nav_msgs/msg/occupancy_grid.hpp"
#include "rclcpp/rclcpp.hpp"
#include "rclcpp_action/rclcpp_action.hpp"
#include "std_msgs/msg/string.hpp"
#include "tf2_ros/buffer.h"
#include "tf2_ros/transform_listener.h"
#include "vision_msgs/msg/detection3_d_array.hpp"
#include "visualization_msgs/msg/marker_array.hpp"
#include "rviz_2d_overlay_msgs/msg/overlay_text.hpp"

namespace {

using Nav = nav2_msgs::action::NavigateToPose;

std::string norm(std::string s)
{
  std::string o;
  for (char c : s) {
    if (c == '_' || c == '-') {c = ' ';}
    o += static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
  }
  while (!o.empty() && o.back() == ' ') {o.pop_back();}
  while (!o.empty() && o.front() == ' ') {o.erase(o.begin());}
  return o;
}

class ObjectGoto : public rclcpp::Node {
 public:
  ObjectGoto() : Node("object_goto"), tf_buffer_(get_clock()), tf_listener_(tf_buffer_) {
    mode_ = declare_parameter<std::string>("mode", "nav2");
    direct_ = declare_parameter<bool>("direct", false);
    map_frame_ = declare_parameter<std::string>("map_frame", "map");
    robot_frame_ = declare_parameter<std::string>("robot_frame", "base_link");
    stand_dist_ = declare_parameter<double>("stand_dist", 0.6);
    stand_band_ = declare_parameter<double>("stand_band", 0.6);
    robot_radius_ = declare_parameter<double>("robot_radius", 0.25);
    occ_threshold_ = declare_parameter<int>("occupied_threshold", 50);
    query_wait_ = declare_parameter<double>("query_wait", 3.0);
    query_min_score_ = declare_parameter<double>("query_min_score", 0.05);
    requery_s_ = declare_parameter<double>("requery_s", 5.0);
    // "한글=영문클래스" (영문은 object_mapper class_names 이름)
    for (const auto & a : declare_parameter<std::vector<std::string>>("aliases", std::vector<std::string>{})) {
      const auto eq = a.find('=');
      if (eq != std::string::npos) {aliases_.push_back({a.substr(0, eq), a.substr(eq + 1)});}
    }

    status_pub_ = create_publisher<std_msgs::msg::String>("~/status", rclcpp::QoS(1).transient_local());
    // RViz: 지금 지시·고른 물체(흰 고리 + "GOAL 이름")·설 자리, 왼쪽 아래 글상자
    target_pub_ = create_publisher<visualization_msgs::msg::MarkerArray>("~/target", rclcpp::QoS(1).transient_local());
    overlay_pub_ = create_publisher<rviz_2d_overlay_msgs::msg::OverlayText>("~/overlay", rclcpp::QoS(1).transient_local());
    query_pub_ = create_publisher<std_msgs::msg::String>(declare_parameter<std::string>("query_topic", "/object_mapper/query"), 10);
    explore_pub_ = create_publisher<std_msgs::msg::String>(declare_parameter<std::string>("explore_topic", "/low_rl/instruction"), 10);
    goto_pub_ = create_publisher<geometry_msgs::msg::PointStamped>(declare_parameter<std::string>("explore_goto_topic", "/low_rl/goto"), 10);
    face_pub_ = create_publisher<geometry_msgs::msg::PointStamped>(declare_parameter<std::string>("explore_face_topic", "/low_rl/goto_face"), 10);
    target_pub_cls_ = create_publisher<std_msgs::msg::String>(declare_parameter<std::string>("explore_target_topic", "/low_rl/target_class"), 10);
    objects_sub_ = create_subscription<vision_msgs::msg::Detection3DArray>(
      declare_parameter<std::string>("objects_topic", "/object_mapper/objects"), rclcpp::QoS(1).transient_local(),
      [this](vision_msgs::msg::Detection3DArray::ConstSharedPtr m) {
        const size_t before = objects_ ? objects_->detections.size() : 0;
        objects_ = m;
        if (searching_ && !done_ && m->detections.size() != before) {try_name();}   // 새 물체가 올라오면 바로 다시 찾음
      });
    result_sub_ = create_subscription<vision_msgs::msg::Detection3DArray>(
      declare_parameter<std::string>("query_result_topic", "/object_mapper/query_result"), 10,
      [this](vision_msgs::msg::Detection3DArray::ConstSharedPtr m) {on_result(m);});
    map_sub_ = create_subscription<nav_msgs::msg::OccupancyGrid>(
      declare_parameter<std::string>("map_topic", "/map"), rclcpp::QoS(1).transient_local().reliable(),
      [this](nav_msgs::msg::OccupancyGrid::ConstSharedPtr m) {map_ = m;});
    // 탐사 노드가 도착을 알리면(low_rl ~/goto_status "도착…") 이 목표는 끝 — 새 목표가 올 때까지 다시 찾지 않음
    arrived_sub_ = create_subscription<std_msgs::msg::String>(declare_parameter<std::string>("explore_status_topic", "/low_rl/goto_status"), 10,
      [this](std_msgs::msg::String::ConstSharedPtr m) {
        if (m->data.rfind("도착", 0) == 0 && !goal_text_.empty() && !done_) {
          done_ = true;
          searching_ = false;
          waiting_query_ = false;
          say("끝: " + m->data + " — 새 목표를 줄 때까지 다시 찾지 않음");
        }
      });
    goal_sub_ = create_subscription<std_msgs::msg::String>("~/goal", 10,
      [this](std_msgs::msg::String::ConstSharedPtr m) {on_goal(m->data);});
    nav_ = rclcpp_action::create_client<Nav>(this, declare_parameter<std::string>("nav_action", "navigate_to_pose"));
    timer_ = create_wall_timer(std::chrono::milliseconds(200), [this]() {tick();});
    say("대기 — /object_goto/goal 에 물체 이름이나 설명을 주세요");
  }

 private:
  void say(const std::string & s) {
    std_msgs::msg::String m;
    m.data = s;
    status_pub_->publish(m);
    RCLCPP_INFO(get_logger(), "%s", s.c_str());
    rviz_2d_overlay_msgs::msg::OverlayText o;
    o.action = rviz_2d_overlay_msgs::msg::OverlayText::ADD;
    o.horizontal_alignment = rviz_2d_overlay_msgs::msg::OverlayText::LEFT;
    o.vertical_alignment = rviz_2d_overlay_msgs::msg::OverlayText::BOTTOM;
    o.horizontal_distance = 12;
    o.vertical_distance = 12;
    o.width = 520;
    o.height = 70;
    o.text_size = 11.0;
    o.font = "Noto Sans CJK KR";
    o.bg_color.r = 0.09f; o.bg_color.g = 0.10f; o.bg_color.b = 0.12f; o.bg_color.a = 0.85f;
    o.fg_color.r = o.fg_color.g = o.fg_color.b = o.fg_color.a = 1.0f;
    o.text = "<b>지시</b> " + (goal_text_.empty() ? std::string("없음") : goal_text_) + "<br>" + s;
    overlay_pub_->publish(o);
  }

  // 고른 물체(흰 고리 + 이름) · 설 자리(작은 초록 원). clear 면 지움
  void show_target(const vision_msgs::msg::Detection3D * d, double sx, double sy) {
    visualization_msgs::msg::MarkerArray a;
    visualization_msgs::msg::Marker base;
    base.header.frame_id = map_frame_;
    base.header.stamp = now();
    base.ns = "goto";
    base.pose.orientation.w = 1.0;
    for (int k = 0; k < 3; ++k) {
      visualization_msgs::msg::Marker m = base;
      m.id = k;
      if (!d) {m.action = visualization_msgs::msg::Marker::DELETE; a.markers.push_back(m); continue;}
      const double ox = d->bbox.center.position.x, oy = d->bbox.center.position.y;
      if (k == 0) {m.type = m.CYLINDER; m.pose.position.x = ox; m.pose.position.y = oy; m.pose.position.z = 0.05; m.scale.x = m.scale.y = 0.9; m.scale.z = 0.02;
        m.color.r = m.color.g = m.color.b = 1.0f; m.color.a = 0.5f;}
      if (k == 1) {m.type = m.TEXT_VIEW_FACING; m.pose.position.x = ox; m.pose.position.y = oy - 0.55; m.pose.position.z = 1.0; m.scale.z = 0.26;
        m.color.r = m.color.g = m.color.b = m.color.a = 1.0f; m.text = "GOAL " + d->id;}
      if (k == 2) {m.type = m.CYLINDER; m.pose.position.x = sx; m.pose.position.y = sy; m.pose.position.z = 0.05; m.scale.x = m.scale.y = 0.25; m.scale.z = 0.03;
        m.color.g = 1.0f; m.color.a = 0.9f;}
      a.markers.push_back(m);
    }
    target_pub_->publish(a);
  }

  std::optional<std::pair<double, double>> robot_xy() {
    try {
      const auto tf = tf_buffer_.lookupTransform(map_frame_, robot_frame_, tf2::TimePointZero);
      return std::make_pair(tf.transform.translation.x, tf.transform.translation.y);
    } catch (const tf2::TransformException &) {return std::nullopt;}
  }

  // 글 → 클래스 이름 (별칭 · 클래스 이름 그대로). 없으면 빈 글
  std::string class_of(const std::string & text) const {
    const std::string t = norm(text);
    for (const auto & a : aliases_) {
      if (norm(a.first) == t) {return a.second;}
    }
    if (objects_) {
      for (const auto & d : objects_->detections) {
        if (!d.results.empty() && norm(d.results[0].hypothesis.class_id) == t) {return d.results[0].hypothesis.class_id;}
      }
    }
    return "";
  }

  void on_goal(const std::string & text) {
    goal_text_ = text;
    waiting_query_ = false;
    searching_ = false;
    done_ = false;
    if (text.empty()) {
      std_msgs::msg::String m;
      explore_pub_->publish(m);   // 탐사 노드 지시도 지움
      show_target(nullptr, 0, 0);
      say("지시 지움");
      return;
    }
    if (mode_ == "explore" && !direct_) {   // 판단 모델이 고르게: 클래스와 지시만 넘김
      const std::string cls = class_of(text);
      std_msgs::msg::String c, t;
      c.data = cls;
      t.data = text;
      target_pub_cls_->publish(c);
      explore_pub_->publish(t);
      int n = 0;
      if (objects_ && !cls.empty()) {
        for (const auto & d : objects_->detections) {n += !d.results.empty() && d.results[0].hypothesis.class_id == cls;}
      }
      say("'" + text + "' → " + (cls.empty() ? std::string("클래스 모름(설명 검색은 다음 단계)") : "클래스 " + cls) + " · 지도에 " + std::to_string(n) +
        " 개 — 판단 모델이 후보(그 물체 옆 · 프론티어)에서 고름");
      return;
    }
    if (try_name()) {return;}
    // 이름으로 못 찾음 → 설명 글 검색 (SigLIP2). 답이 안 오면 탐사로 넘김
    ask_query();
    say("'" + text + "' — 지도 이름에 없어 설명 검색(SigLIP2) 중");
  }

  void ask_query() {
    std_msgs::msg::String q;
    q.data = goal_text_;
    query_pub_->publish(q);
    waiting_query_ = true;
    query_t0_ = now();
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

  void on_result(const vision_msgs::msg::Detection3DArray::ConstSharedPtr & m) {
    if (!waiting_query_ || done_) {return;}
    waiting_query_ = false;
    if (!m->detections.empty() && !m->detections[0].results.empty() &&
        m->detections[0].results[0].hypothesis.score >= query_min_score_) {
      char b[64];
      std::snprintf(b, sizeof b, "설명 검색 1 위 %.2f", m->detections[0].results[0].hypothesis.score);
      searching_ = false;
      go_to(m->detections[0], b);
      return;
    }
    if (!searching_) {not_found();}
  }

  void not_found() {
    if (done_) {return;}
    std_msgs::msg::String m;
    m.data = goal_text_;
    explore_pub_->publish(m);
    searching_ = true;
    say("'" + goal_text_ + "' — 지도에 없음. 탐사하며 찾는 중 (물체 지도에 나타나면 바로 감)");
  }

  void tick() {
    if (done_) {return;}
    if (waiting_query_ && (now() - query_t0_).seconds() > query_wait_) {
      waiting_query_ = false;
      if (!searching_) {not_found();}
    }
    if (searching_ && !waiting_query_ && (now() - query_t0_).seconds() > requery_s_) {ask_query();}
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
    waiting_query_ = false;   // 찾았으면 기다리던 설명 검색·다시 찾기를 멈춤 (도착·새 지시까지 "지도에 없음" 으로 안 돌아감)
    searching_ = false;
    const double ox = d.bbox.center.position.x, oy = d.bbox.center.position.y;
    const auto spot = stand_spot(ox, oy);
    if (!spot) {
      say("'" + goal_text_ + "' → " + d.id + " (" + how + ") 를 찾았지만 지도·위치가 없거나 설 자리가 없음");
      return;
    }
    char b[256];
    std::snprintf(b, sizeof b, "'%s' → %s (%s) 물체 (%.2f, %.2f) · 설 자리 (%.2f, %.2f) 로 %s", goal_text_.c_str(), d.id.c_str(),
      how.c_str(), ox, oy, spot->first, spot->second, mode_ == "nav2" ? "Nav2 주행" : "탐사 노드에 목표");
    say(b);
    show_target(&d, spot->first, spot->second);
    if (mode_ == "nav2") {
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
    } else {
      geometry_msgs::msg::PointStamped f;   // 도착 뒤 바라볼 물체 자리 (low_rl ~/goto_face) — ~/goto 보다 먼저
      f.header.frame_id = map_frame_;
      f.header.stamp = now();
      f.point.x = ox;
      f.point.y = oy;
      face_pub_->publish(f);
      geometry_msgs::msg::PointStamped p;
      p.header.frame_id = map_frame_;
      p.header.stamp = now();
      p.point.x = spot->first;
      p.point.y = spot->second;
      goto_pub_->publish(p);
    }
  }

  tf2_ros::Buffer tf_buffer_;
  tf2_ros::TransformListener tf_listener_;
  std::string mode_, map_frame_, robot_frame_, goal_text_;
  double stand_dist_, stand_band_, robot_radius_, query_wait_, query_min_score_;
  int occ_threshold_;
  std::vector<std::pair<std::string, std::string>> aliases_;
  bool waiting_query_ = false, searching_ = false, direct_ = false, done_ = false;
  rclcpp::Subscription<std_msgs::msg::String>::SharedPtr arrived_sub_;
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr target_pub_cls_;
  double requery_s_;
  rclcpp::Time query_t0_{0, 0, RCL_ROS_TIME};
  vision_msgs::msg::Detection3DArray::ConstSharedPtr objects_;
  nav_msgs::msg::OccupancyGrid::ConstSharedPtr map_;
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr status_pub_, query_pub_, explore_pub_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr target_pub_;
  rclcpp::Publisher<rviz_2d_overlay_msgs::msg::OverlayText>::SharedPtr overlay_pub_;
  rclcpp::Publisher<geometry_msgs::msg::PointStamped>::SharedPtr goto_pub_, face_pub_;
  rclcpp::Subscription<vision_msgs::msg::Detection3DArray>::SharedPtr objects_sub_, result_sub_;
  rclcpp::Subscription<nav_msgs::msg::OccupancyGrid>::SharedPtr map_sub_;
  rclcpp::Subscription<std_msgs::msg::String>::SharedPtr goal_sub_;
  rclcpp_action::Client<Nav>::SharedPtr nav_;
  rclcpp::TimerBase::SharedPtr timer_;
};

}  // namespace

int main(int argc, char ** argv) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<ObjectGoto>());
  rclcpp::shutdown();
  return 0;
}
