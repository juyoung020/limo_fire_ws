// 찍어서 순찰 (임무 BT 와 따로 도는 시험 도구): RViz "Publish Point" 로 지도를 찍은 점들을 Nav2 navigate_to_pose 로
// 차례로 돌고 끝나면 처음부터 다시 돈다.
//   - 찍으면 순찰점 끝에 더함. 이미 찍은 점 근처(remove_radius_m 안)를 다시 찍으면 그 점을 지움
//   - ~/start 로 출발, ~/stop 으로 멈춤(Nav2 목표 취소), ~/undo 마지막 점 지우기, ~/clear 다 지우기 (std_srvs/Trigger)
//   - 바뀔 때마다 out_file 에 patrol: [[x, y], ...] 로 저장 (임무 BT patrol_file 과 같은 형식)
//   - ~/markers: 번호·순서 선, 지금 가는 점은 초록
// 설정: config/click_patrol.yaml. 실행: ros2 launch limo_fire_bt click_patrol.launch.xml
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

#include <geometry_msgs/msg/point_stamped.hpp>
#include <nav2_msgs/action/navigate_to_pose.hpp>
#include <rclcpp/rclcpp.hpp>
#include <rclcpp_action/rclcpp_action.hpp>
#include <std_srvs/srv/trigger.hpp>
#include <visualization_msgs/msg/marker_array.hpp>
#include <yaml-cpp/yaml.h>

namespace
{
using Nav = nav2_msgs::action::NavigateToPose;
using Trigger = std_srvs::srv::Trigger;

struct Point2
{
  double x, y;
};

class ClickPatrol : public rclcpp::Node
{
public:
  ClickPatrol()
  : Node("click_patrol")
  {
    frame_ = declare_parameter<std::string>("frame", "map");
    remove_radius_ = declare_parameter<double>("remove_radius_m", 0.3);
    retry_wait_s_ = declare_parameter<double>("retry_wait_s", 3.0);
    loop_ = declare_parameter<bool>("loop", true);
    const auto file = declare_parameter<std::string>("file", "");
    out_file_ = declare_parameter<std::string>("out_file", "");
    if (out_file_.empty()) {
      const char * home = std::getenv("HOME");
      out_file_ = std::string(home ? home : "/tmp") + "/limo_fire_patrol.yaml";
    }
    nav_ = rclcpp_action::create_client<Nav>(this, declare_parameter<std::string>("action_name", "/navigate_to_pose"));
    marker_pub_ = create_publisher<visualization_msgs::msg::MarkerArray>("~/markers",
        rclcpp::QoS(1).reliable().transient_local());
    click_sub_ = create_subscription<geometry_msgs::msg::PointStamped>(
      declare_parameter<std::string>("clicked_topic", "/clicked_point"), 10,
      [this](geometry_msgs::msg::PointStamped::ConstSharedPtr m) {on_click(*m);});
    srvs_.push_back(service("~/start", [this](Trigger::Response & r) {
        if (points_.empty()) {r.message = "순찰점이 없음 — RViz Publish Point 로 먼저 찍기"; return;}
        if (!running_) {running_ = true; next_ = 0; send_next();}
        r.success = true;
        r.message = std::to_string(points_.size()) + " 개 순찰 시작";
      }));
    srvs_.push_back(service("~/stop", [this](Trigger::Response & r) {
        stop("멈춤 요청");
        r.success = true;
      }));
    srvs_.push_back(service("~/undo", [this](Trigger::Response & r) {
        r.success = !points_.empty();
        if (r.success) {points_.pop_back(); changed("마지막 점 지움");}
        r.message = std::to_string(points_.size()) + " 개";
      }));
    srvs_.push_back(service("~/clear", [this](Trigger::Response & r) {
        points_.clear();
        changed("다 지움");
        r.success = true;
      }));

    if (!file.empty()) {
      try {
        for (const auto & p : YAML::LoadFile(file)["patrol"]) {points_.push_back({p[0].as<double>(), p[1].as<double>()});}
        RCLCPP_INFO(get_logger(), "순찰점 %zu 개를 %s 에서 읽음 — 이어서 찍으면 끝에 더함", points_.size(), file.c_str());
      } catch (const std::exception & e) {
        RCLCPP_ERROR(get_logger(), "순찰점 파일 %s 를 못 읽음: %s — 빈 순찰로 시작", file.c_str(), e.what());
      }
    }
    draw();
    RCLCPP_INFO(get_logger(), "RViz 위쪽 'Publish Point' 로 지도를 찍으면 순찰점 (저장: %s). "
      "출발: ros2 service call /click_patrol/start std_srvs/srv/Trigger", out_file_.c_str());
    if (declare_parameter<bool>("auto_start", false) && !points_.empty()) {
      running_ = true;
      send_next();
    }
  }

  ~ClickPatrol() override {if (goal_) {nav_->async_cancel_goal(goal_);}}

private:
  rclcpp::Service<Trigger>::SharedPtr service(const std::string & name, std::function<void(Trigger::Response &)> fn)
  {
    return create_service<Trigger>(name, [fn](Trigger::Request::SharedPtr, Trigger::Response::SharedPtr r) {fn(*r);});
  }

  void on_click(const geometry_msgs::msg::PointStamped & m)
  {
    if (m.header.frame_id != frame_) {
      RCLCPP_WARN(get_logger(), "찍은 점의 frame 이 %s — RViz Fixed Frame 을 %s 로", m.header.frame_id.c_str(), frame_.c_str());
      return;
    }
    for (size_t i = 0; i < points_.size(); ++i) {
      if (std::hypot(points_[i].x - m.point.x, points_[i].y - m.point.y) < remove_radius_) {
        points_.erase(points_.begin() + static_cast<long>(i));
        if (next_ > i) {--next_;}
        changed(std::to_string(i + 1) + " 번 점 지움");
        return;
      }
    }
    points_.push_back({m.point.x, m.point.y});
    char b[96];
    std::snprintf(b, sizeof(b), "%zu 번 점 (%.2f, %.2f)", points_.size(), m.point.x, m.point.y);
    changed(b);
  }

  void changed(const std::string & what)
  {
    if (points_.empty()) {stop("순찰점 없음");}
    draw();
    std::ofstream f(out_file_);
    f << "# RViz 에서 찍은 순찰점 (click_patrol). 지도 좌표 m, 순서대로 돌고 처음으로\npatrol:" << (points_.empty() ? " []" : "") << "\n";
    char b[64];
    for (const auto & p : points_) {std::snprintf(b, sizeof(b), "  - [%.2f, %.2f]\n", p.x, p.y); f << b;}
    RCLCPP_INFO(get_logger(), "%s → 순찰점 %zu 개%s", what.c_str(), points_.size(), f ? "" : " (저장 못 함)");
  }

  void stop(const std::string & why)
  {
    if (!running_) {return;}
    running_ = false;
    retry_.reset();
    if (goal_) {nav_->async_cancel_goal(goal_); goal_.reset();}
    RCLCPP_INFO(get_logger(), "순찰 멈춤 (%s)", why.c_str());
    draw();
  }

  // 순서 그대로 다음 점 (끝나면 처음, loop=false 면 멈춤). 방향은 그다음 점 쪽
  void send_next()
  {
    if (!running_ || points_.empty()) {return;}
    if (next_ >= points_.size()) {
      if (!loop_) {stop("한 바퀴 끝"); return;}
      next_ = 0;
    }
    if (!nav_->wait_for_action_server(std::chrono::seconds(0))) {
      RCLCPP_WARN(get_logger(), "Nav2(navigate_to_pose) 가 아직 없음 — %.0f s 뒤 다시", retry_wait_s_);
      wait_then_next(false);
      return;
    }
    const auto & p = points_[next_];
    const auto & n = points_[(next_ + 1) % points_.size()];
    const double yaw = points_.size() > 1 ? std::atan2(n.y - p.y, n.x - p.x) : 0.0;
    Nav::Goal g;
    g.pose.header.frame_id = frame_;
    g.pose.header.stamp = now();
    g.pose.pose.position.x = p.x;
    g.pose.pose.position.y = p.y;
    g.pose.pose.orientation.z = std::sin(yaw / 2);
    g.pose.pose.orientation.w = std::cos(yaw / 2);
    const size_t index = next_;
    const int token = ++token_;
    rclcpp_action::Client<Nav>::SendGoalOptions o;
    o.goal_response_callback = [this, token](rclcpp_action::ClientGoalHandle<Nav>::SharedPtr h) {
        if (token != token_) {return;}
        if (!h) {RCLCPP_WARN(get_logger(), "Nav2 가 목표를 거절함"); wait_then_next(true); return;}
        goal_ = h;
      };
    o.result_callback = [this, token, index](const rclcpp_action::ClientGoalHandle<Nav>::WrappedResult & r) {
        if (token != token_ || !running_) {return;}
        goal_.reset();
        if (r.code == rclcpp_action::ResultCode::SUCCEEDED) {
          RCLCPP_INFO(get_logger(), "%zu 번 점 도착", index + 1);
          ++next_;
          send_next();
        } else {
          RCLCPP_WARN(get_logger(), "%zu 번 점 못 감 — %.0f s 뒤 다음 점", index + 1, retry_wait_s_);
          wait_then_next(true);
        }
      };
    RCLCPP_INFO(get_logger(), "%zu/%zu 번 점 (%.2f, %.2f) 로", index + 1, points_.size(), p.x, p.y);
    nav_->async_send_goal(g, o);
    draw();
  }

  void wait_then_next(bool skip)
  {
    retry_ = create_wall_timer(std::chrono::duration<double>(retry_wait_s_), [this, skip]() {
        retry_->cancel();
        if (skip) {++next_;}
        send_next();
      });
  }

  void draw()
  {
    visualization_msgs::msg::MarkerArray arr;
    visualization_msgs::msg::Marker m;
    m.action = visualization_msgs::msg::Marker::DELETEALL;
    arr.markers.push_back(m);
    m.action = visualization_msgs::msg::Marker::ADD;
    m.header.frame_id = frame_;
    m.header.stamp = now();
    m.pose.orientation.w = 1.0;
    if (points_.size() > 1) {
      auto line = m;
      line.ns = "line";
      line.type = visualization_msgs::msg::Marker::LINE_STRIP;
      line.scale.x = 0.04;
      line.color.r = 1.0; line.color.g = 0.55; line.color.a = 0.8;
      for (size_t i = 0; i <= points_.size(); ++i) {
        geometry_msgs::msg::Point q;
        q.x = points_[i % points_.size()].x;
        q.y = points_[i % points_.size()].y;
        line.points.push_back(q);
      }
      arr.markers.push_back(line);
    }
    for (size_t i = 0; i < points_.size(); ++i) {
      auto dot = m;
      dot.ns = "point";
      dot.id = static_cast<int>(i);
      dot.type = visualization_msgs::msg::Marker::SPHERE;
      dot.pose.position.x = points_[i].x;
      dot.pose.position.y = points_[i].y;
      dot.scale.x = dot.scale.y = dot.scale.z = 0.2;
      const bool target = running_ && i == next_;
      dot.color.r = target ? 0.1 : 1.0; dot.color.g = target ? 0.9 : 0.55; dot.color.b = 0.1; dot.color.a = 1.0;
      arr.markers.push_back(dot);
      auto label = dot;
      label.ns = "number";
      label.type = visualization_msgs::msg::Marker::TEXT_VIEW_FACING;
      label.text = std::to_string(i + 1);
      label.pose.position.z = 0.3;
      label.scale.x = label.scale.y = 0.0;
      label.scale.z = 0.3;
      label.color.r = label.color.g = label.color.b = 1.0;
      arr.markers.push_back(label);
    }
    marker_pub_->publish(arr);
  }

  std::string frame_, out_file_;
  double remove_radius_, retry_wait_s_;
  bool loop_, running_ = false;
  std::vector<Point2> points_;
  size_t next_ = 0;
  int token_ = 0;
  rclcpp_action::Client<Nav>::SharedPtr nav_;
  rclcpp_action::ClientGoalHandle<Nav>::SharedPtr goal_;
  rclcpp::TimerBase::SharedPtr retry_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr marker_pub_;
  rclcpp::Subscription<geometry_msgs::msg::PointStamped>::SharedPtr click_sub_;
  std::vector<rclcpp::Service<Trigger>::SharedPtr> srvs_;
};
}  // namespace

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<ClickPatrol>());
  rclcpp::shutdown();
  return 0;
}
