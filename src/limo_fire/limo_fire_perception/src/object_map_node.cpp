// YOLO 탐지 상자 + 정렬된 깊이 영상 → 물체의 지도(map) 좌표 → RViz 마커 + 한글 현황판(오버레이)
// 같은 클래스가 merge_radius 안에서 다시 보이면 같은 물체로 보고 위치를 평균한다. min_hits 번 이상 본 것만 지도에 올린다.
// 깊이는 컬러 카메라에 정렬돼 있어야 한다 (실물 Orbbec align_mode HW, Webots 는 원래 같은 프레임)
// 분할 모델이면 같은 시각의 물체 번호 영상(/perception/mask)으로 물체 픽셀만 골라 위치·거리·(불) 넓이를 잰다. 없으면 상자로
//
// 한 스레드 컨테이너에서 다른 콜백을 막지 않게: 탐지·깊이 짝이 오면 받기 콜백은 작업 줄(max_pending 장, 넘치면 가장 오래된 장을 버림 —
// 늘 최신 쪽이 남음)에 넣고 바로 돌아오고, 처리는 process_period 타이머가 그 시각 TF 가 있는 장부터 한다(tf_wait 지나도 없으면 버림).
// TF 도 이 노드의 콜백으로 받는다(따로 스레드 없음). 깊이 영상 받기 메시지는 몇 개를 돌려 써서 장마다 큰 할당을 하지 않는다.
// stats_period 마다 짝 못 지은 탐지 비율(탐지↔깊이 max_sync_dt)·줄 지연·처리 시간을 로그로 낸다
#include <algorithm>
#include <array>
#include <chrono>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <deque>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include <message_filters/pass_through.h>
#include <message_filters/subscriber.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <message_filters/synchronizer.h>
#include <geometry_msgs/msg/point_stamped.hpp>
#include <ament_index_cpp/get_package_share_directory.hpp>
#include <rclcpp/message_memory_strategy.hpp>
#include <rclcpp/rclcpp.hpp>
#include <rclcpp_components/register_node_macro.hpp>
#include <rviz_2d_overlay_msgs/msg/overlay_text.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <std_msgs/msg/string.hpp>
#include <std_srvs/srv/trigger.hpp>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>
#include <vision_msgs/msg/detection2_d_array.hpp>
#include <vision_msgs/msg/detection3_d_array.hpp>
#include <visualization_msgs/msg/marker_array.hpp>

namespace limo_fire_perception
{

namespace
{
using visualization_msgs::msg::Marker;

// 받기 메시지를 돌려 쓰는 메모리 전략: 아무도(동기화 줄·작업 줄) 안 쥔 메시지에 다음 장을 역직렬화한다 → 영상 버퍼를 장마다 새로 할당하지 않음.
// 다 쥐고 있으면(최대 max 개) 새로 만든다. 실행기 스레드에서만 부른다
template<class MsgT>
class PooledMessages : public rclcpp::message_memory_strategy::MessageMemoryStrategy<MsgT>
{
public:
  explicit PooledMessages(size_t max) : max_(max) {}
  std::shared_ptr<MsgT> borrow_message() override
  {
    for (auto & m : pool_) {
      if (m.use_count() == 1) {return m;}
    }
    if (pool_.size() < max_) {
      pool_.push_back(std::make_shared<MsgT>());
      return pool_.back();
    }
    return std::make_shared<MsgT>();
  }
  void return_message(std::shared_ptr<MsgT> & msg) override {msg.reset();}

private:
  size_t max_;
  std::vector<std::shared_ptr<MsgT>> pool_;
};

// 클래스마다 지도 표시: 지도 위 짧은 이름(RViz 글꼴은 영문만), 현황판 한글 이름, 색
struct Style
{
  const char * name;   // class_names 와 같은 이름
  const char * tag;    // 지도 라벨
  const char * ko;     // 현황판
  float r, g, b;
};
constexpr std::array<Style, 7> kStyles{{
  {"fire", "FIRE", "불", 1.00f, 0.36f, 0.10f},
  {"smoke", "SMOKE", "연기", 0.62f, 0.62f, 0.66f},
  {"fire_extinguisher", "EXT", "소화기", 0.90f, 0.12f, 0.16f},
  {"exit_sign", "EXIT", "비상구 유도등", 0.10f, 0.78f, 0.36f},
  {"person", "PERSON", "사람", 0.20f, 0.50f, 1.00f},
  {"door", "DOOR", "문", 0.72f, 0.48f, 0.26f},
  {"stairs", "STAIRS", "계단", 0.66f, 0.42f, 0.92f},
}};
// 화재 7 종 밖의 물체(COCO 80 종 등): 지도 라벨 = 이름 대문자(12 자까지), 현황판 = 이름 그대로, 색 = 이름 해시로 고른 밝은 색.
// 이 문자열들은 노드가 살아 있는 동안 남아야 해서(Style 은 const char *) 한 번 만들어 둔다
// 설정 class_colors 로 준 색 (노드가 켤 때 채움). 있으면 정해진 색보다 앞섬
std::unordered_map<std::string, std::array<float, 3>> & color_override()
{
  static std::unordered_map<std::string, std::array<float, 3>> m;
  return m;
}

const Style & style_of(const std::string & name)
{
  const auto co = color_override().find(name);
  if (co == color_override().end()) {
    for (const auto & s : kStyles) {
      if (name == s.name) {return s;}
    }
  }
  static std::deque<std::array<std::string, 3>> names;
  static std::deque<Style> made;
  for (const auto & s : made) {
    if (name == s.name) {return s;}
  }
  for (const auto & s : kStyles) {   // 정해진 이름표·한글 이름은 두고 색만 바꿈
    if (name == s.name) {
      made.push_back({s.name, s.tag, s.ko, co->second[0], co->second[1], co->second[2]});
      return made.back();
    }
  }
  std::string tag;
  for (char c : name) {tag += (c == ' ') ? '_' : static_cast<char>(std::toupper(static_cast<unsigned char>(c)));}
  if (tag.size() > 12) {tag.resize(12);}
  names.push_back({name, tag, name});
  uint32_t h = 2166136261u;
  for (char c : name) {h = (h ^ static_cast<uint8_t>(c)) * 16777619u;}
  const float hue = (h % 360) / 60.0f, x = 1.0f - std::fabs(std::fmod(hue, 2.0f) - 1.0f);
  float r = 0, g = 0, b = 0;
  switch (static_cast<int>(hue)) {
    case 0: r = 1; g = x; break;
    case 1: r = x; g = 1; break;
    case 2: g = 1; b = x; break;
    case 3: g = x; b = 1; break;
    case 4: r = x; b = 1; break;
    default: r = 1; b = x; break;
  }
  if (co != color_override().end()) {r = co->second[0]; g = co->second[1]; b = co->second[2];} else {
    r = 0.35f + 0.6f * r; g = 0.35f + 0.6f * g; b = 0.35f + 0.6f * b;
  }
  made.push_back({names.back()[0].c_str(), names.back()[1].c_str(), names.back()[2].c_str(), r, g, b});
  return made.back();
}

// 배지 아이콘 파일(icons/<클래스>.obj)이 설치돼 있는지 (화재 7 종 + COCO 80 종 — icons/README). 없으면 클래스 색 원판
bool has_icon(const std::string & name)
{
  static std::unordered_map<std::string, bool> cache;
  const auto it = cache.find(name);
  if (it != cache.end()) {return it->second;}
  bool ok = false;
  try {
    ok = static_cast<bool>(std::ifstream(ament_index_cpp::get_package_share_directory("limo_fire_perception") + "/icons/" + name + ".obj"));
  } catch (const std::exception &) {}
  cache[name] = ok;
  return ok;
}

std_msgs::msg::ColorRGBA rgba(float r, float g, float b, float a)
{
  std_msgs::msg::ColorRGBA c;
  c.r = r;
  c.g = g;
  c.b = b;
  c.a = a;
  return c;
}

std::string hex(const Style & s)
{
  char buf[8];
  std::snprintf(buf, sizeof(buf), "#%02x%02x%02x", static_cast<int>(s.r * 255), static_cast<int>(s.g * 255),
    static_cast<int>(s.b * 255));
  return buf;
}
}  // namespace

class ObjectMapNode : public rclcpp::Node
{
public:
  explicit ObjectMapNode(const rclcpp::NodeOptions & options)
  : Node("object_map", options), tf_buffer_(get_clock()),
    tf_listener_(tf_buffer_, this, false, tf2_ros::DynamicListenerQoS(), tf2_ros::StaticListenerQoS(), tf_sub_options(false),
      tf_sub_options(true))
  {
    map_frame_ = declare_parameter<std::string>("map_frame", "map");
    robot_frame_ = declare_parameter<std::string>("robot_frame", "base_link");
    min_depth_ = declare_parameter<double>("min_depth", 0.2);
    max_depth_ = declare_parameter<double>("max_depth", 5.0);
    box_shrink_ = declare_parameter<double>("box_shrink", 0.5);
    depth_percentile_ = declare_parameter<double>("depth_percentile", 0.3);
    mask_depth_percentile_ = declare_parameter<double>("mask_depth_percentile", 0.5);
    merge_radius_ = declare_parameter<double>("merge_radius", 0.8);
    min_hits_ = declare_parameter<int>("min_hits", 3);
    min_score_ = declare_parameter<double>("min_score", 0.5);
    fire_zone_radius_ = declare_parameter<double>("fire_zone_radius", 1.0);
    label_offset_ = declare_parameter<std::vector<double>>("label_offset", {0.0, 0.55});
    icon_size_ = declare_parameter<double>("icon_size", 0.6);
    view_angle_ = declare_parameter<double>("view_angle", 0.0);
    label_offset_.resize(2, 0.0);
    class_names_ = declare_parameter<std::vector<std::string>>("class_names");
    // 클래스별 병합 반경 (class_names 순서). 계단·문처럼 큰 물체는 넓게. 비우면 모두 merge_radius
    merge_radii_ = declare_parameter<std::vector<double>>("merge_radii", std::vector<double>{});
    merge_radii_.resize(class_names_.size(), merge_radius_);
    // 클래스별 지도 색 "#rrggbb" (class_names 순서). 비우면 정해진 색(화재 7 종) 또는 이름에서 고른 색
    const auto colors = declare_parameter<std::vector<std::string>>("class_colors", std::vector<std::string>{});
    for (size_t k = 0; k < colors.size() && k < class_names_.size(); ++k) {
      unsigned rr, gg, bb;
      if (std::sscanf(colors[k].c_str(), "#%02x%02x%02x", &rr, &gg, &bb) == 3) {
        color_override()[class_names_[k]] = {rr / 255.0f, gg / 255.0f, bb / 255.0f};
      }
    }
    counters_.assign(class_names_.size(), 0);
    // 지도에 안 올릴 클래스 (실내 로봇에 없는 것 — COCO 의 비행기·버스·트럭은 실험실 큰 가구를 잘못 본 것이 대부분)
    for (const auto & n : declare_parameter<std::vector<std::string>>("ignore_classes", std::vector<std::string>{})) {ignored_.push_back(n);}
    // 물체 지도 파일 (클래스·위치·크기·본 횟수·마지막 본 시각). 있으면 켤 때 읽는다 — 저장한 지도(map_saver)와 같이 다시 쓰려고
    objects_file_ = declare_parameter<std::string>("objects_file", "");

    // 늦게 켠 RViz 도 받게 transient_local. 프로세스 안 전달(intra-process)은 volatile 만 되므로 이 둘은 끈다
    rclcpp::PublisherOptions latched;
    latched.use_intra_process_comm = rclcpp::IntraProcessSetting::Disable;
    marker_pub_ = create_publisher<visualization_msgs::msg::MarkerArray>(
      "markers", rclcpp::QoS(1).transient_local(), latched);
    overlay_pub_ = create_publisher<rviz_2d_overlay_msgs::msg::OverlayText>(
      "summary", rclcpp::QoS(1).transient_local(), latched);
    // 다른 노드(임무 BT)가 읽는 물체 목록: 지도에 올라간 것만, map frame. 마커는 표시용이라 형식이 바뀔 수 있다
    objects_pub_ = create_publisher<vision_msgs::msg::Detection3DArray>(
      "objects", rclcpp::QoS(1).transient_local(), latched);
    info_sub_ = create_subscription<sensor_msgs::msg::CameraInfo>(
      declare_parameter<std::string>("camera_info_topic", "/camera/depth/camera_info"), rclcpp::SensorDataQoS(),
      [this](sensor_msgs::msg::CameraInfo::ConstSharedPtr msg) {info_ = msg;});
    // 마스크는 탐지와 시각이 같다. 최근 몇 장만 두고 탐지가 오면 같은 시각 것을 찾는다
    mask_sub_ = create_subscription<sensor_msgs::msg::Image>(
      declare_parameter<std::string>("mask_topic", "/perception/mask"), 10,
      [this](sensor_msgs::msg::Image::ConstSharedPtr msg) {
        masks_.push_back(msg);
        if (masks_.size() > 10) {masks_.pop_front();}
      });

    // 지도에 올린 물체를 모두 지운다: ros2 service call /perception/clear_objects std_srvs/srv/Trigger
    clear_srv_ = create_service<std_srvs::srv::Trigger>("clear_objects",
        [this](std_srvs::srv::Trigger::Request::SharedPtr, std_srvs::srv::Trigger::Response::SharedPtr res) {
          landmarks_.clear();
          std::fill(counters_.begin(), counters_.end(), 0);
          visualization_msgs::msg::MarkerArray arr;
          arr.markers.emplace_back().action = Marker::DELETEALL;
          marker_pub_->publish(arr);
          res->success = true;
          res->message = "지도 물체를 모두 지웠음";
        });

    // 물체 지도 저장: ~/save (objects_file 에) · ~/save_to (std_msgs/String 경로 — save_map.launch.xml 이 지도 옆에) · ~/load
    save_srv_ = create_service<std_srvs::srv::Trigger>("save_objects",
        [this](std_srvs::srv::Trigger::Request::SharedPtr, std_srvs::srv::Trigger::Response::SharedPtr res) {
          res->success = !objects_file_.empty() && save(objects_file_);
          res->message = res->success ? "저장: " + objects_file_ : "objects_file 이 비었거나 쓰기 실패";
        });
    load_srv_ = create_service<std_srvs::srv::Trigger>("load_objects",
        [this](std_srvs::srv::Trigger::Request::SharedPtr, std_srvs::srv::Trigger::Response::SharedPtr res) {
          res->success = !objects_file_.empty() && load(objects_file_);
          res->message = res->success ? "읽음: " + objects_file_ : "objects_file 이 비었거나 읽기 실패";
        });
    save_to_sub_ = create_subscription<std_msgs::msg::String>("save_objects_to", 10,
        [this](std_msgs::msg::String::ConstSharedPtr m) {save(m->data);});
    if (!objects_file_.empty()) {load(objects_file_);}

    // 탐지와 깊이 영상은 시각이 정확히 같지 않으므로 가까운 것끼리 짝짓는다
    det_sub_.subscribe(this, declare_parameter<std::string>("detections_topic", "/perception/detections"));
    det_sub_.registerCallback([this](const vision_msgs::msg::Detection2DArray::ConstSharedPtr &) {++st_.dets;});
    // 깊이: 카메라 드라이버(다른 프로세스)가 내므로 역직렬화한다 — 받기 메시지를 돌려 써서(동기화 줄 10 + 작업 줄 + 여유) 장마다 할당을 안 함
    depth_sub_ = create_subscription<sensor_msgs::msg::Image>(
      declare_parameter<std::string>("depth_topic", "/camera/depth/image_raw"), rclcpp::SensorDataQoS(),
      [this](sensor_msgs::msg::Image::ConstSharedPtr msg) {++st_.depths; depth_in_.add(msg);}, rclcpp::SubscriptionOptions(),
      std::make_shared<PooledMessages<sensor_msgs::msg::Image>>(32));
    const double sync_dt = declare_parameter<double>("max_sync_dt", 0.05);
    // 탐지 시각의 TF 를 기다리는 한도(s), 작업 줄 장 수 한도 (넘으면 가장 오래된 장을 버림)
    tf_wait_ = declare_parameter<double>("tf_wait", 0.2);
    max_pending_ = static_cast<size_t>(std::max<int64_t>(1, declare_parameter<int>("max_pending", 10)));
    sync_ = std::make_shared<message_filters::Synchronizer<Policy>>(Policy(10), det_sub_, depth_in_);
    // 회전 중에 다른 순간의 깊이와 짝지어지면 엉뚱한 곳에 찍히므로 시각 차이를 제한한다
    sync_->setMaxIntervalDuration(rclcpp::Duration::from_seconds(sync_dt));
    sync_->registerCallback(std::bind(&ObjectMapNode::on_pair, this, std::placeholders::_1, std::placeholders::_2));
    cell_size_ = declare_parameter<double>("cell_size", 0.05);
    // 지도 갱신: 보여야 하는데 neg_misses 장 연속 못 보면 뺌 (시야 가장자리 neg_margin 비율 · neg_max_depth m 밖 · neg_occl_m 넘게 앞이 가린 것은 셈 안 함)
    neg_misses_ = declare_parameter<int>("neg_misses", 20);
    neg_max_depth_ = declare_parameter<double>("neg_max_depth", 3.0);
    neg_margin_ = declare_parameter<double>("neg_margin", 0.1);
    neg_occl_m_ = declare_parameter<double>("neg_occl_m", 0.3);
    // 움직이는 클래스(사람·동물): 위치·발자국은 가장 최근 장 것(dynamic_alpha 1 = 그대로 — 평균하면 걷는 사람을 늦게 따라감),
    // 같은 물체로 볼 거리는 merge 반경 + dynamic_speed × 못 본 시간 (dynamic_max_radius 까지 — 잠깐 가려졌다 다시 나와도 같은 번호),
    // 시야 안에서 dynamic_neg_misses 번 못 보면 · 시야 밖이면 dynamic_ttl 초 안 보이면 뺌
    dynamic_ = declare_parameter<std::vector<std::string>>("dynamic_classes", std::vector<std::string>{"person", "cat", "dog", "bird"});
    dynamic_ttl_ = declare_parameter<double>("dynamic_ttl", 3.0);
    dynamic_alpha_ = std::clamp(declare_parameter<double>("dynamic_alpha", 1.0), 0.05, 1.0);
    dynamic_speed_ = declare_parameter<double>("dynamic_speed", 1.5);
    dynamic_max_radius_ = declare_parameter<double>("dynamic_max_radius", 2.0);
    dynamic_neg_misses_ = declare_parameter<int>("dynamic_neg_misses", 5);
    // 정지 물체가 merge 반경 안에서 옮겨짐: 지금 자리에서 max(move_dist, 너비/2) 넘게 떨어진 측정이 move_frames 번 이어지면
    // 그 자리로 옮기고 발자국·밑점을 새로 쌓는다 (평균만 하면 천천히 끌려가고 옛 자리 발자국이 남음). 0 이면 끔
    move_dist_ = declare_parameter<double>("move_dist", 0.5);
    move_frames_ = declare_parameter<int>("move_frames", 10);
    // 바닥 광선 거리 (가상 깊이). base_height = base_link 의 바닥 위 높이 (limo_pro_base.xacro 0.145 m)
    base_height_ = declare_parameter<double>("base_height", 0.145);
    base_rows_frac_ = declare_parameter<double>("base_rows_frac", 0.05);
    fuse_tol_m_ = declare_parameter<double>("fuse_tol_m", 0.3);
    fuse_tol_frac_ = declare_parameter<double>("fuse_tol_frac", 0.15);
    prefer_floor_ = declare_parameter<std::vector<std::string>>("prefer_floor_classes", std::vector<std::string>{"fire", "smoke"});
    footprint_stride_ = declare_parameter<int>("footprint_stride", 3);
    footprint_min_count_ = declare_parameter<int>("footprint_min_count", 2);
    footprint_alpha_ = static_cast<float>(declare_parameter<double>("footprint_alpha", 0.25));   // 발자국 칠 진하기 (아이콘 밑에 옅게)
    footprint_max_z_ = declare_parameter<double>("footprint_max_z", 2.0);

    // 10 Hz: ~/objects(물체 목록 — 바뀌지 않아도 늘 다시 냄, 받는 쪽은 마지막 것만 씀) · 마커(불 위험 반경 깜빡임·로봇까지 거리) — 벽시계라 시뮬이 멈춰도 표시는 돈다
    timer_ = create_wall_timer(std::chrono::milliseconds(100), [this]() {publish();});
    // 작업 줄 처리: 노드 시계 타이머 (시뮬이면 시뮬 시간 — 받은 시각·tf_wait 와 같은 시계)
    proc_timer_ = rclcpp::create_timer(this, get_clock(),
        rclcpp::Duration::from_seconds(declare_parameter<double>("process_period", 0.02)), [this]() {drain_jobs();});
    stats_period_ = declare_parameter<double>("stats_period", 10.0);
    publish_summary();
  }

private:
  // TF 리스너 구독: /tf_static 은 transient_local 이라 프로세스 안 전달(volatile 만 됨)을 끈다
  static rclcpp::SubscriptionOptions tf_sub_options(bool is_static)
  {
    auto o = is_static ? tf2_ros::detail::get_default_transform_listener_static_sub_options<>() :
      tf2_ros::detail::get_default_transform_listener_sub_options<>();
    o.use_intra_process_comm = rclcpp::IntraProcessSetting::Disable;
    return o;
  }

  // TF 를 기다리는 탐지·깊이 짝 (마스크·camera_info 는 받은 때 것으로 묶어 둠)
  struct Job
  {
    vision_msgs::msg::Detection2DArray::ConstSharedPtr dets;
    sensor_msgs::msg::Image::ConstSharedPtr depth, mask;
    sensor_msgs::msg::CameraInfo::ConstSharedPtr info;
    rclcpp::Time t;   // 받은 때 (노드 시계 — 시뮬 시계면 시뮬 시간으로 기다림)
    std::chrono::steady_clock::time_point t_in;   // 받은 때 (줄 지연 재기)
  };

  // 받기·처리 통계 (log_stats 가 stats_period 마다 내고 비움)
  struct Stats
  {
    size_t dets = 0, depths = 0, pairs = 0, n = 0, no_tf = 0, overflow = 0;
    double q_sum = 0, q_max = 0, p_sum = 0, p_max = 0, age_sum = 0, age_max = 0;
    std::chrono::steady_clock::time_point t0{};
    void add_job(double q_ms, double p_ms, double age_ms)
    {
      ++n;
      q_sum += q_ms;
      q_max = std::max(q_max, q_ms);
      p_sum += p_ms;
      p_max = std::max(p_max, p_ms);
      age_sum += age_ms;
      age_max = std::max(age_max, age_ms);
    }
  };

  struct Landmark
  {
    int cls;
    int number;      // 클래스 안 번호 (EXT 1, EXT 2 …), 지도 올라갈 때 매김
    double x, y, z;
    double yaw;      // 처음 본 방향 (문·유도등·계단 판을 카메라 쪽으로 세움)
    int hits;
    double best_score;
    double area;     // m², 마스크로 잰 앞면 넓이 (불 크기). 마스크가 없으면 0
    double ext = 0;  // m, 물체 너비 (상자 너비 × 거리 / fx, 평균) — 지도 위 발자국 크기
    double last_seen = 0;  // s (노드 시계), 마지막으로 본 때
    std::unordered_map<int64_t, uint16_t> cells;
    double bx = 0, by = 0, bz = 0;   // 밑점: 마스크 3D 점 중 가장 낮은 곳들(아래 10 %)의 평균 (map) — 불이면 불 밑동. 마스크가 없으면 0
    int base_n = 0;   // 바닥 발자국: 분할 마스크 픽셀의 3D 점을 지도 칸(cell_size)에 내린 것 — 칸 → 본 횟수
    int misses = 0;          // 보여야 할 때 못 본 수 (지도 갱신 — 다시 보면 0)
    int moved = 0;           // 처음 자리(ax, ay)에서 멀리 잰 측정이 이어진 수 (정지 물체 옮겨짐 판정)
    double ax = 0, ay = 0;   // 그 이어짐이 시작될 때의 자리 (평균이 끌려가도 이 자리와 견줌)
    double mx = 0, my = 0, mz = 0;   // 그 측정들의 평균 (옮겨진 자리)
    bool gone = false;       // 없어짐 (못 본 수 neg_misses 번 · 움직이는 클래스는 dynamic_ttl 초) — 표시·목록·저장에서 뺌, 같은 자리 탐지는 새 물체로
    bool deleted = false;    // 없어진 물체 마커를 지웠음
  };

  // 탐지 하나를 지도에 올릴 값: 영상 좌표 (u, v), 거리 z, 앞면 넓이
  struct Measure
  {
    double u, v, z, area;
  };

  using Policy = message_filters::sync_policies::ApproximateTime<
    vision_msgs::msg::Detection2DArray, sensor_msgs::msg::Image>;


  int class_index(const std::string & name) const
  {
    const auto it = std::find(class_names_.begin(), class_names_.end(), name);
    return it == class_names_.end() ? -1 : static_cast<int>(it - class_names_.begin());
  }

  // 깊이 한 픽셀 (m). u16 = 16UC1(mm), 아니면 32FC1(m)
  static float depth_px(const uint8_t * row, int x, bool u16)
  {
    return u16 ? reinterpret_cast<const uint16_t *>(row)[x] * 0.001f : reinterpret_cast<const float *>(row)[x];
  }

  // 마스크 한 줄 [x0, x1] 에 id 픽셀이 있는지 — 분기 없는 OR 모음이라 컴파일러가 SIMD(NEON·SSE)로 16 칸씩 본다.
  // 물체가 없는 줄을 통째로 건너뛰려고
  static bool row_has(const uint8_t * row, int x0, int x1, uint8_t id)
  {
    unsigned acc = 0;
    for (int x = x0; x <= x1; ++x) {acc |= static_cast<unsigned>(row[x] == id);}
    return acc != 0;
  }

  // 마스크 열 x → 깊이 영상 열 (min(x · sx, 깊이 너비 − 1)). 픽셀마다 곱·변환하지 않게 크기가 바뀔 때만 다시 만든다
  const int * depth_cols(int mask_w, double sx, int depth_w)
  {
    if (dxs_.size() != static_cast<size_t>(mask_w) || dxs_sx_ != sx || dxs_dw_ != depth_w) {
      dxs_.resize(static_cast<size_t>(mask_w));
      for (int x = 0; x < mask_w; ++x) {dxs_[static_cast<size_t>(x)] = std::min(static_cast<int>(x * sx), depth_w - 1);}
      dxs_sx_ = sx;
      dxs_dw_ = depth_w;
    }
    return dxs_.data();
  }

  // 상자 가운데 부분 깊이의 하위 depth_percentile 값 (m). 얇은 물체 뒤의 벽이 섞여도 앞쪽(물체)을 고른다. 유효한 값이 없으면 NaN
  double box_depth(const sensor_msgs::msg::Image & depth, double cx, double cy, double w, double h)
  {
    const int x0 = std::max(0, static_cast<int>(cx - w * box_shrink_ / 2));
    const int x1 = std::min(static_cast<int>(depth.width) - 1, static_cast<int>(cx + w * box_shrink_ / 2));
    const int y0 = std::max(0, static_cast<int>(cy - h * box_shrink_ / 2));
    const int y1 = std::min(static_cast<int>(depth.height) - 1, static_cast<int>(cy + h * box_shrink_ / 2));
    const bool u16 = depth.encoding == "16UC1";
    auto & vals = vals_;
    vals.clear();
    for (int y = y0; y <= y1; ++y) {
      const uint8_t * row = depth.data.data() + static_cast<size_t>(y) * depth.step;
      for (int x = x0; x <= x1; ++x) {
        const float d = depth_px(row, x, u16);
        if (std::isfinite(d) && d >= min_depth_ && d <= max_depth_) {vals.push_back(d);}
      }
    }
    if (vals.size() < 5) {return std::nan("");}
    const auto k = static_cast<size_t>(depth_percentile_ * (vals.size() - 1));
    std::nth_element(vals.begin(), vals.begin() + k, vals.end());
    return vals[k];
  }

  // 분할 마스크에서 한 물체(id)의 픽셀을 상자 안에서 한 번만 훑어, 위치·거리(mask_measure)·밑동(floor_ray)·발자국(footprint)이 같이 쓴다
  struct MaskScan
  {
    int x0, x1, y0, y1;            // 상자 안 픽셀 범위
    size_t n = 0;                  // 물체 픽셀 수
    double su = 0, sv = 0;         // 픽셀 좌표 합 (무게중심)
    int ymin = -1, ymax = -1;
    // vals_ = 물체 픽셀의 유효 깊이 전부, pts_ = 그중 footprint_stride 칸마다의 (x, y, 깊이)
  };
  void scan_mask(const sensor_msgs::msg::Image & depth, const sensor_msgs::msg::Image & mask, uint8_t id,
    const vision_msgs::msg::BoundingBox2D & box, double sx, double sy, MaskScan & sc)
  {
    sc = MaskScan{};
    sc.x0 = std::max(0, static_cast<int>(box.center.position.x - box.size_x / 2));
    sc.x1 = std::min(static_cast<int>(mask.width) - 1, static_cast<int>(box.center.position.x + box.size_x / 2));
    sc.y0 = std::max(0, static_cast<int>(box.center.position.y - box.size_y / 2));
    sc.y1 = std::min(static_cast<int>(mask.height) - 1, static_cast<int>(box.center.position.y + box.size_y / 2));
    const int x0 = sc.x0, x1 = sc.x1;
    const int st = std::max(1, footprint_stride_);
    const bool u16 = depth.encoding == "16UC1";
    const int * dxs = depth_cols(static_cast<int>(mask.width), sx, static_cast<int>(depth.width));
    vals_.clear();
    pts_.clear();
    for (int y = sc.y0; y <= sc.y1; ++y) {
      const uint8_t * mrow = mask.data.data() + static_cast<size_t>(y) * mask.step;
      if (!row_has(mrow, x0, x1, id)) {continue;}
      const int dy = std::min(static_cast<int>(y * sy), static_cast<int>(depth.height) - 1);
      const uint8_t * drow = depth.data.data() + static_cast<size_t>(dy) * depth.step;
      const bool grid_row = (y - sc.y0) % st == 0;
      if (sc.ymin < 0) {sc.ymin = y;}
      sc.ymax = y;
      for (int x = x0; x <= x1; ++x) {
        if (mrow[x] != id) {continue;}
        sc.su += x;
        sc.sv += y;
        ++sc.n;
        const float d = depth_px(drow, dxs[x], u16);
        if (!(std::isfinite(d) && d >= min_depth_ && d <= max_depth_)) {continue;}
        vals_.push_back(d);
        if (grid_row && (x - x0) % st == 0) {pts_.push_back({static_cast<float>(x), static_cast<float>(y), d});}
      }
    }
  }

  // 마스크 픽셀의 무게중심과 그 픽셀들 깊이의 mask_depth_percentile 값. 유효한 깊이가 적으면 false
  bool mask_measure(const MaskScan & sc, double fx, double fy, Measure & m)
  {
    auto & vals = vals_;
    if (vals.size() < 10) {return false;}
    const auto k = static_cast<size_t>(mask_depth_percentile_ * (vals.size() - 1));
    std::nth_element(vals.begin(), vals.begin() + k, vals.end());
    m.u = sc.su / sc.n;
    m.v = sc.sv / sc.n;
    m.z = vals[k];
    m.area = sc.n * m.z * m.z / (fx * fy);   // 픽셀 하나 = (z/fx)·(z/fy) m²
    return true;
  }

  // 탐지·깊이 짝: TF 를 기다리지 않고 작업 줄에 넣는다 (같은 시각 마스크·camera_info 는 지금 것으로 묶음)
  void on_pair(
    const vision_msgs::msg::Detection2DArray::ConstSharedPtr & dets,
    const sensor_msgs::msg::Image::ConstSharedPtr & depth)
  {
    if (!info_) {return;}
    if (depth->encoding != "16UC1" && depth->encoding != "32FC1") {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 5000, "지원하지 않는 깊이 형식: %s", depth->encoding.c_str());
      return;
    }
    // 같은 시각 마스크 (분할 모델일 때). 영상 크기가 탐지 기준(camera_info)과 같아야 쓴다
    sensor_msgs::msg::Image::ConstSharedPtr mask;
    for (const auto & mk : masks_) {
      if (mk->header.stamp == dets->header.stamp && mk->width == info_->width && mk->height == info_->height) {
        mask = mk;
      }
    }
    ++st_.pairs;
    jobs_.push_back({dets, depth, mask, info_, now(), std::chrono::steady_clock::now()});
    if (jobs_.size() > max_pending_) {
      jobs_.pop_front();
      ++st_.overflow;
      ++n_overflow_;
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 5000, "작업 줄이 %zu 장을 넘어 오래된 장을 버림 (지금까지 %zu 장)",
        max_pending_, n_overflow_);
    }
  }

  // 타이머: 앞에서부터 그 시각 TF 가 있는 장을 처리한다. 없으면 tf_wait 까지 기다리고(뒤 장도 차례를 지켜 기다림), 지나면 버린다
  void drain_jobs()
  {
    while (!jobs_.empty()) {
      const Job & j = jobs_.front();
      std::string err;
      if (!tf_buffer_.canTransform(map_frame_, j.depth->header.frame_id, tf2_ros::fromMsg(j.depth->header.stamp), &err)) {
        if ((now() - j.t).seconds() < tf_wait_) {break;}
        ++n_no_tf_;
        ++st_.no_tf;
        RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 5000, "TF 없음 (지금까지 %zu 장): %s", n_no_tf_, err.c_str());
        jobs_.pop_front();
        continue;
      }
      const Job job = std::move(jobs_.front());
      jobs_.pop_front();
      geometry_msgs::msg::TransformStamped tf;
      try {
        // tf2::TimePoint 판(BufferCore)으로: rclcpp::Time 판은 기다림 없는 부름에도 "따로 스레드 없음" 오류를 찍는다
        tf = tf_buffer_.lookupTransform(map_frame_, job.depth->header.frame_id, tf2_ros::fromMsg(job.depth->header.stamp));
      } catch (const tf2::TransformException & e) {
        ++n_no_tf_;
        ++st_.no_tf;
        RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 5000, "TF 없음 (지금까지 %zu 장): %s", n_no_tf_, e.what());
        continue;
      }
      const auto t0 = std::chrono::steady_clock::now();
      process(job, tf);
      const auto t1 = std::chrono::steady_clock::now();
      st_.add_job(std::chrono::duration<double, std::milli>(t0 - job.t_in).count(),
        std::chrono::duration<double, std::milli>(t1 - t0).count(),
        (now() - rclcpp::Time(job.dets->header.stamp, get_clock()->get_clock_type())).seconds() * 1e3);
    }
    log_stats();
  }

  // stats_period 마다 한 줄: 탐지↔깊이 짝(ApproximateTime, max_sync_dt) 못 지은 탐지 비율, 줄 지연(받음 → 처리 시작, steady 시계)·
  // 탐지 시각부터 처리 끝까지 나이, 장당 처리 시간, 버린 장
  void log_stats()
  {
    const auto t = std::chrono::steady_clock::now();
    if (st_.t0 == std::chrono::steady_clock::time_point{}) {st_.t0 = t; return;}
    const double sec = std::chrono::duration<double>(t - st_.t0).count();
    if (stats_period_ <= 0 || sec < stats_period_) {return;}
    const double miss = st_.dets > 0 ? 100.0 * (1.0 - std::min<double>(1.0, double(st_.pairs) / st_.dets)) : 0.0;
    RCLCPP_INFO(get_logger(),
      "%.0f s: 탐지 %zu · 깊이 %zu 장 → 짝 %zu (짝 못 지은 탐지 %.1f %%), 처리 %zu 장 — 줄 지연 평균 %.1f · 최대 %.1f ms, "
      "처리 평균 %.2f · 최대 %.2f ms, 탐지 시각부터 나이 평균 %.0f · 최대 %.0f ms, 버림: TF 없음 %zu · 줄 넘침 %zu",
      sec, st_.dets, st_.depths, st_.pairs, miss, st_.n, st_.n ? st_.q_sum / st_.n : 0.0, st_.q_max,
      st_.n ? st_.p_sum / st_.n : 0.0, st_.p_max, st_.n ? st_.age_sum / st_.n : 0.0, st_.age_max, st_.no_tf, st_.overflow);
    st_ = Stats{};
    st_.t0 = t;
  }

  void process(const Job & job, const geometry_msgs::msg::TransformStamped & tf)
  {
    const auto & dets = job.dets;
    const auto & depth = job.depth;
    const auto & mask = job.mask;
    const auto & info = *job.info;
    t_obs_ = job.t.seconds();   // 이 장의 시각 = 받은 때 (처리가 타이머로 늦어져도 마지막 본 시각·못 본 시간은 받은 때 기준)
    // 탐지 좌표는 컬러 영상 기준. 깊이 해상도가 다르면 비율로 맞춘다
    const double sx = static_cast<double>(depth->width) / info.width;
    const double sy = static_cast<double>(depth->height) / info.height;
    const double fx = info.k[0], fy = info.k[4], cx = info.k[2], cy = info.k[5];

    auto & seen_now = seen_now_;   // 이 장에서 다시 본 물체 (못 봄 셈에서 뺌)
    seen_now.clear();
    for (size_t i = 0; i < dets->detections.size(); ++i) {
      const auto & d = dets->detections[i];
      if (d.results.empty() || d.results[0].hypothesis.score < min_score_) {continue;}
      const int cls = class_index(d.results[0].hypothesis.class_id);
      if (cls < 0 || std::find(ignored_.begin(), ignored_.end(), class_names_[cls]) != ignored_.end()) {continue;}
      Measure m{d.bbox.center.position.x, d.bbox.center.position.y, 0.0, 0.0};
      const bool scanned = mask && i < 255;
      MaskScan sc;
      if (scanned) {scan_mask(*depth, *mask, static_cast<uint8_t>(i + 1), d.bbox, sx, sy, sc);}
      const bool mask_ok = scanned && mask_measure(sc, fx, fy, m);
      if (!mask_ok) {
        m.z = box_depth(*depth, m.u * sx, m.v * sy, d.bbox.size_x * sx, d.bbox.size_y * sy);
      }
      // 바닥 광선 거리 (가상 깊이): 마스크 밑동 픽셀을 지나는 카메라 광선이 바닥면과 만나는 곳까지의 광축 거리.
      // 불꽃처럼 깊이 카메라가 비거나 틀리는 물체를 위해 — 바닥에 선 물체에서만 맞음
      Base b;
      const bool base_ok = scanned && floor_ray(*mask, static_cast<uint8_t>(i + 1), sc, fx, fy, cx, cy, tf, b);
      if (base_ok) {
        const bool prefer_floor = std::find(prefer_floor_.begin(), prefer_floor_.end(), class_names_[cls]) != prefer_floor_.end();
        const bool disagree = std::isfinite(m.z) && std::fabs(m.z - b.z) > fuse_tol_m_ + fuse_tol_frac_ * b.z;
        if (!std::isfinite(m.z) || (disagree && prefer_floor)) {m.z = b.z;}
      }
      if (!std::isfinite(m.z)) {continue;}

      geometry_msgs::msg::PointStamped p_cam, p_map;
      p_cam.point.x = (m.u - cx) * m.z / fx;
      p_cam.point.y = (m.v - cy) * m.z / fy;
      p_cam.point.z = m.z;
      tf2::doTransform(p_cam, p_map, tf);
      const double view_yaw = std::atan2(p_map.point.y - tf.transform.translation.y,
          p_map.point.x - tf.transform.translation.x);
      const size_t idx = add_seen(seen_now, cls, p_map.point.x, p_map.point.y, p_map.point.z, view_yaw,
          d.results[0].hypothesis.score, m.area, d.bbox.size_x * m.z / fx);
      if (scanned) {footprint(fx, fy, cx, cy, tf, landmarks_[idx]);}   // scan_mask 의 pts_ (그 뒤 box_depth 만 vals_ 를 씀)
    }
    negative_evidence(*depth, info, tf, sx, sy, fx, fy, cx, cy, seen_now);
  }

  // 물체 목록에 넣고 그 물체의 번호(landmarks_ 안 위치)를 돌려준다
  // 움직이는 클래스(dynamic_classes)인지 (클래스 번호로 기억)
  bool is_dynamic(int cls)
  {
    while (dyn_cls_.size() < class_names_.size()) {
      const auto & n = class_names_[dyn_cls_.size()];
      dyn_cls_.push_back(std::find(dynamic_.begin(), dynamic_.end(), n) != dynamic_.end());
    }
    return dyn_cls_[static_cast<size_t>(cls)];
  }

  size_t add(int cls, double x, double y, double z, double yaw, double score, double area, double ext)
  {
    const double t_now = t_obs_;   // 이 장을 받은 때 (process)
    const bool dyn = is_dynamic(cls);
    Landmark * best = nullptr;
    double best_d = merge_radii_[cls];
    for (auto & l : landmarks_) {
      // 움직이는 것은 뺀 지 dynamic_ttl 안이면 다시 살린다 (잠깐 못 잡아 빠졌다 다시 보여도 같은 번호)
      const bool revivable = dyn && l.gone && l.number > 0 && t_now - l.last_seen < dynamic_ttl_;
      if ((l.gone && !revivable) || l.cls != cls) {continue;}
      const double dist = std::hypot(l.x - x, l.y - y);
      if (dyn) {   // 움직이는 것: 못 본 동안 움직였을 만큼 넓게, 그중 가장 가까운 것
        const double r = std::min(merge_radii_[cls] + dynamic_speed_ * std::max(0.0, t_now - l.last_seen), dynamic_max_radius_);
        if (dist < r && (!best || dist < best_d)) {best = &l; best_d = dist;}
      } else if (dist < best_d) {
        best = &l;
        best_d = dist;
      }
    }
    if (!best) {
      landmarks_.push_back({cls, 0, x, y, z, yaw, 1, score, area, ext, t_now});
      return landmarks_.size() - 1;
    }
    // 처음 몇 번은 평균, 그 뒤로는 천천히 따라간다 (멀리서 본 부정확한 값이 남지 않게). 움직이는 것은 최근 값(dynamic_alpha)
    const double a = dyn ? dynamic_alpha_ : 1.0 / std::min(best->hits + 1, 20);
    const double move_thr = std::max(move_dist_, 0.5 * best->ext);
    if (!dyn && move_frames_ > 0 && best->moved == 0 && best_d > move_thr) {
      best->ax = best->x;
      best->ay = best->y;
    }
    if (!dyn && move_frames_ > 0 && (best_d > move_thr || best->moved > 0) && std::hypot(x - best->ax, y - best->ay) > move_thr) {
      // 처음 자리에서 멀리 잰 측정이 이어지면 옮겨진 것: 그 측정들의 평균 자리로 옮기고 발자국·밑점을 새로 쌓음
      const double k = 1.0 / (best->moved + 1);
      best->mx += k * (x - best->mx);
      best->my += k * (y - best->my);
      best->mz += k * (z - best->mz);
      if (++best->moved >= move_frames_) {
        RCLCPP_INFO(get_logger(), "옮겨짐: %s-%d (%.2f, %.2f) → (%.2f, %.2f)", class_names_[cls].c_str(), best->number, best->x, best->y,
          best->mx, best->my);
        best->x = best->mx;
        best->y = best->my;
        best->z = best->mz;
        best->cells.clear();
        best->base_n = 0;
        best->moved = 0;
        best->last_seen = t_now;
        return static_cast<size_t>(best - landmarks_.data());
      }
    } else {
      best->moved = 0;
    }
    if (best->gone) {
      best->gone = false;
      best->deleted = false;
      RCLCPP_INFO(get_logger(), "다시 봄: %s-%d", class_names_[cls].c_str(), best->number);
    }
    best->x += a * (x - best->x);
    best->y += a * (y - best->y);
    best->z += a * (z - best->z);
    if (area > 0) {best->area = best->area > 0 ? best->area + a * (area - best->area) : area;}
    if (ext > 0) {best->ext = best->ext > 0 ? best->ext + a * (ext - best->ext) : ext;}
    best->last_seen = t_now;
    best->hits++;
    best->best_score = std::max(best->best_score, score);
    const int need = min_hits_;
    if (best->hits >= need && best->number == 0) {
      best->number = ++counters_[cls];
      RCLCPP_INFO(get_logger(), "지도에 추가: %s %d (%.2f, %.2f)", style_of(class_names_[cls]).tag, best->number,
        best->x, best->y);
    }
    return static_cast<size_t>(best - landmarks_.data());
  }

  struct Base
  {
    double u = 0, v = 0;          // 밑동 픽셀
    double z = 0;                 // 바닥 광선 광축 거리 (m)
  };
  // 마스크 맨 아래 base_rows_frac 줄의 가운데 열 → 카메라 광선 → 바닥면(map 높이 floor_z: 로봇 base_link 높이 − base_height) 과 만나는 점
  bool floor_ray(const sensor_msgs::msg::Image & mask, uint8_t id, const MaskScan & sc,
    double fx, double fy, double cx, double cy, const geometry_msgs::msg::TransformStamped & tf, Base & b)
  {
    const int x0 = sc.x0, x1 = sc.x1;
    const int ymin = sc.ymin, ymax = sc.ymax;
    if (ymax < 0 || ymax - ymin < 3) {return false;}
    if (ymax >= static_cast<int>(mask.height) - 2) {return false;}   // 밑동이 영상 아래 끝에 잘리면 진짜 밑동이 아님
    const int ylo = ymax - std::max(1, static_cast<int>(base_rows_frac_ * (ymax - ymin)));
    auto & xs = xs_;
    xs.clear();
    for (int y = ylo; y <= ymax; ++y) {
      const uint8_t * row = mask.data.data() + static_cast<size_t>(y) * mask.step;
      for (int x = x0; x <= x1; ++x) {if (row[x] == id) {xs.push_back(x);}}
    }
    std::nth_element(xs.begin(), xs.begin() + xs.size() / 2, xs.end());
    b.u = xs[xs.size() / 2];
    b.v = ymax;
    tf2::Transform T;
    tf2::fromMsg(tf.transform, T);
    const tf2::Vector3 o = T.getOrigin();
    const tf2::Vector3 dir = T.getBasis() * tf2::Vector3((b.u - cx) / fx, (b.v - cy) / fy, 1.0);   // 광축 거리 1 m 당
    // 바닥 높이: 로봇(base_link) 높이 − base_height (map 이 2D 라 보통 base_link z = 0)
    double floor_z = -base_height_;
    try {
      floor_z = tf_buffer_.lookupTransform(map_frame_, robot_frame_, tf2::TimePointZero).transform.translation.z - base_height_;   // 기다리지 않음 (높이는 거의 안 바뀜)
    } catch (const tf2::TransformException &) {}
    if (dir.z() > -1e-3) {return false;}   // 광선이 바닥으로 안 내려감
    const double t = (floor_z - o.z()) / dir.z();
    if (t <= min_depth_ || t > max_depth_) {return false;}
    b.z = t;
    return true;
  }

  static int64_t cell_key(int64_t gx, int64_t gy)
  {
    return static_cast<int64_t>((static_cast<uint64_t>(static_cast<uint32_t>(gx)) << 32) | static_cast<uint32_t>(gy));
  }
  static void cell_of(int64_t k, int32_t & gx, int32_t & gy)
  {
    gx = static_cast<int32_t>(static_cast<uint64_t>(k) >> 32);
    gy = static_cast<int32_t>(static_cast<uint64_t>(k) & 0xffffffffu);
  }

  // 분할 마스크 픽셀(footprint_stride 칸마다)을 깊이로 3D 점 → map → 바닥 칸(cell_size)에 센다. 물체 깊이(가운데 값) ± 0.5 m 밖 점(배경)은 버림
  void footprint(double fx, double fy, double cx, double cy, const geometry_msgs::msg::TransformStamped & tf, Landmark & l)
  {
    const auto & pts = pts_;
    auto & zs = zs_;
    zs.clear();
    for (const auto & p : pts) {zs.push_back(p[2]);}
    if (zs.size() < 5) {return;}
    std::nth_element(zs.begin(), zs.begin() + zs.size() / 2, zs.end());
    const float zmed = zs[zs.size() / 2];
    tf2::Transform T;
    tf2::fromMsg(tf.transform, T);
    auto & kept = kept_;
    kept.clear();
    const bool dyn = is_dynamic(l.cls);
    if (dyn) {l.cells.clear();}   // 움직이는 것: 발자국은 이 장 것만 (지나온 자리가 남지 않게)
    // 점마다 나누지 않게 역수를 곱한다 (마지막 비트가 다를 수 있음 — README "스레드·메모리" 에 확인 결과)
    const double ifx = 1.0 / fx, ify = 1.0 / fy, icell = 1.0 / cell_size_;
    for (const auto & p : pts) {
      if (std::fabs(p[2] - zmed) > 0.5f) {continue;}
      const tf2::Vector3 v = T * tf2::Vector3((p[0] - cx) * p[2] * ifx, (p[1] - cy) * p[2] * ify, p[2]);
      kept.push_back(v);
      if (v.z() > footprint_max_z_) {continue;}
      const int64_t gx = static_cast<int64_t>(std::floor(v.x() * icell)), gy = static_cast<int64_t>(std::floor(v.y() * icell));
      uint16_t & c = l.cells[cell_key(gx, gy)];
      if (c < 65535) {++c;}
    }
    // 밑점: 높이 아래 10 % 점의 평균. 여러 번 본 값은 평균 (처음 몇 번 평균 → 천천히 따라감, 위치와 같은 규칙)
    if (kept.size() >= 10) {
      std::sort(kept.begin(), kept.end(), [](const tf2::Vector3 & a, const tf2::Vector3 & b) {return a.z() < b.z();});
      const size_t n = std::max<size_t>(1, kept.size() / 10);
      double x = 0, y = 0, z = 0;
      for (size_t k = 0; k < n; ++k) {x += kept[k].x(); y += kept[k].y(); z += kept[k].z();}
      x /= n; y /= n; z /= n;
      const double a = dyn ? dynamic_alpha_ : 1.0 / std::min(l.base_n + 1, 20);
      l.bx += a * (x - l.bx);
      l.by += a * (y - l.by);
      l.bz += a * (z - l.bz);
      ++l.base_n;
    }
  }

  template<typename ... A>
  size_t add_seen(std::vector<size_t> & seen, A ... a)
  {
    const size_t i = add(a ...);
    seen.push_back(i);
    landmarks_[i].misses = 0;
    return i;
  }

  // 지도 갱신 (못 본 증거): 지도에 올라간 물체 가운데 이 장에서 보여야 하는데(카메라 시야 안 · neg_max_depth 안 · 그 픽셀 깊이가 물체보다 앞에 가리는 게 없음)
  // 다시 못 본 것은 misses 를 늘리고, neg_misses 번이면 없어짐. 움직이는 클래스(dynamic_classes)는 dynamic_ttl 초 안 보이면 없어짐.
  void negative_evidence(const sensor_msgs::msg::Image & depth, const sensor_msgs::msg::CameraInfo & info,
    const geometry_msgs::msg::TransformStamped & tf, double sx, double sy, double fx, double fy, double cx, double cy,
    const std::vector<size_t> & seen)
  {
    const bool u16 = depth.encoding == "16UC1";
    tf2::Transform T;
    tf2::fromMsg(tf.transform, T);
    const tf2::Transform inv = T.inverse();
    const double t_now = t_obs_;   // 이 장을 받은 때 (process)
    for (size_t i = 0; i < landmarks_.size(); ++i) {
      Landmark & l = landmarks_[i];
      if (l.gone || l.number == 0) {continue;}
      const bool dyn = is_dynamic(l.cls);
      if (dyn && t_now - l.last_seen > dynamic_ttl_) {
        l.gone = true;
        RCLCPP_INFO(get_logger(), "지도에서 뺌 (움직이는 클래스 %.0f s 안 보임): %s-%d", t_now - l.last_seen, class_names_[l.cls].c_str(), l.number);
        continue;
      }
      if (std::find(seen.begin(), seen.end(), i) != seen.end()) {continue;}
      const tf2::Vector3 pc = inv * tf2::Vector3(l.x, l.y, l.z);
      if (pc.z() < min_depth_ || pc.z() > neg_max_depth_) {continue;}
      const double u = fx * pc.x() / pc.z() + cx, v = fy * pc.y() / pc.z() + cy;
      const double mu = neg_margin_ * info.width, mv = neg_margin_ * info.height;
      if (u < mu || v < mv || u >= info.width - mu || v >= info.height - mv) {continue;}
      // 그 픽셀 둘레 깊이 가운데 값: 물체보다 neg_occl_m 넘게 앞이면 가려짐(셈 안 함), 비었으면 모름
      auto & ds = vals_;
      ds.clear();
      for (int dy = -3; dy <= 3; ++dy) {
        for (int dx = -3; dx <= 3; ++dx) {
          const int px = std::clamp(static_cast<int>((u + dx * 2) * sx), 0, static_cast<int>(depth.width) - 1);
          const int py = std::clamp(static_cast<int>((v + dy * 2) * sy), 0, static_cast<int>(depth.height) - 1);
          const uint8_t * row = depth.data.data() + static_cast<size_t>(py) * depth.step;
          const float d = depth_px(row, px, u16);
          if (std::isfinite(d) && d > 0.05f) {ds.push_back(d);}
        }
      }
      if (ds.size() < 10) {continue;}
      std::nth_element(ds.begin(), ds.begin() + ds.size() / 2, ds.end());
      if (ds[ds.size() / 2] < pc.z() - neg_occl_m_) {continue;}
      if (++l.misses >= (dyn ? dynamic_neg_misses_ : neg_misses_)) {
        l.gone = true;
        RCLCPP_INFO(get_logger(), "지도에서 뺌 (보여야 하는데 %d 번 못 봄): %s-%d (%.2f, %.2f)", l.misses, class_names_[l.cls].c_str(), l.number, l.x, l.y);
      }
    }
  }

  // 물체 지도 파일: 머리 줄 + 물체마다 한 줄 "클래스 번호 x y z yaw 본횟수 최고점수 넓이 너비 마지막본시각" (map 좌표, m · rad · s).
  // 지도에 올라간 것(number > 0)만 쓴다. 대표 사진은 쓰지 않는다
  bool save(const std::string & path)
  {
    std::ofstream f(path);
    if (!f) {RCLCPP_WARN(get_logger(), "물체 지도 저장 실패: %s", path.c_str()); return false;}
    f << "# limo_fire object map v1 (frame " << map_frame_ << ", cell " << cell_size_ << " m): class number x y z yaw hits best_score area ext last_seen n_cells gx,gy,count ...\n";
    int n = 0;
    for (const auto & l : landmarks_) {
      if (l.number == 0 || l.gone) {continue;}
      char buf[256];
      std::snprintf(buf, sizeof(buf), "%s %d %.3f %.3f %.3f %.3f %d %.3f %.4f %.3f %.1f %zu", class_names_[l.cls].c_str(), l.number,
        l.x, l.y, l.z, l.yaw, l.hits, l.best_score, l.area, l.ext, l.last_seen, l.cells.size());
      f << buf;
      // 발자국 칸은 (gx, gy) 순으로 (해시 표 순서는 정해져 있지 않아 같은 지도도 파일이 달라짐)
      std::vector<std::array<int32_t, 3>> cs;
      cs.reserve(l.cells.size());
      for (const auto & kv : l.cells) {
        int32_t gx, gy;
        cell_of(kv.first, gx, gy);
        cs.push_back({gx, gy, kv.second});
      }
      std::sort(cs.begin(), cs.end());
      for (const auto & c : cs) {
        f << ' ' << c[0] << ',' << c[1] << ',' << c[2];
      }
      f << '\n';
      ++n;
    }
    RCLCPP_INFO(get_logger(), "물체 지도 저장: %s (%d 개)", path.c_str(), n);
    return static_cast<bool>(f);
  }

  // 저장한 물체 지도를 읽어 지금 목록에 더한다 (같은 클래스 merge 반경 안이면 합침). 모르는 클래스는 class_names 뒤에 붙인다
  bool load(const std::string & path)
  {
    std::ifstream f(path);
    if (!f) {RCLCPP_WARN(get_logger(), "물체 지도 파일 없음: %s", path.c_str()); return false;}
    std::string line;
    int n = 0;
    while (std::getline(f, line)) {
      if (line.empty() || line[0] == '#') {continue;}
      std::istringstream is(line);
      std::string name;
      int number, hits;
      double x, y, z, yaw, score, area, ext, last;
      if (!(is >> name >> number >> x >> y >> z >> yaw >> hits >> score >> area >> ext >> last)) {continue;}
      int cls = class_index(name);
      if (cls < 0) {class_names_.push_back(name); merge_radii_.push_back(merge_radius_); counters_.push_back(0); cls = class_index(name);}
      Landmark l{cls, 0, x, y, z, yaw, std::max(hits, min_hits_), score, area, ext, last};
      size_t nc = 0;
      if (is >> nc) {
        std::string tok;
        for (size_t k = 0; k < nc && (is >> tok); ++k) {
          int gx, gy, c;
          if (std::sscanf(tok.c_str(), "%d,%d,%d", &gx, &gy, &c) == 3) {l.cells[cell_key(gx, gy)] = static_cast<uint16_t>(c);}
        }
      }
      l.number = number > 0 ? number : counters_[cls] + 1;   // 저장한 번호 그대로 (물체 이름 "chair-3" 이 사진 묶음 파일과 맞게)
      counters_[cls] = std::max(counters_[cls], l.number);
      landmarks_.push_back(l);
      ++n;
    }
    RCLCPP_INFO(get_logger(), "물체 지도 읽음: %s (%d 개)", path.c_str(), n);
    return true;
  }

  // 물체 하나 = 바닥 원판 + 종류별 3D 모양 + 라벨. id 는 물체 번호 * 10 + 부품 번호로 고정해 깜빡임 없이 덮어쓴다
  // dx, dy: 배지·이름표만 옮기는 표시 오프셋 (겹침 피하기 — 물체 위치·발자국은 그대로)
  void append_markers(const Landmark & l, size_t index, double t, visualization_msgs::msg::MarkerArray & arr, double dx = 0.0, double dy = 0.0)
  {
    const bool dyn_fp = is_dynamic(l.cls);
    const Style & s = style_of(class_names_[l.cls]);
    const int base_id = static_cast<int>(index) * 10;
    auto make = [&](int part, int type, double z, double sx, double sy, double sz, std_msgs::msg::ColorRGBA c,
        double yaw = 0.0) {
        Marker m;
        m.header.frame_id = map_frame_;
        m.ns = s.name;
        m.id = base_id + part;
        m.type = type;
        m.pose.position.x = l.x + dx;
        m.pose.position.y = l.y + dy;
        m.pose.position.z = z;
        m.pose.orientation.z = std::sin(yaw / 2);
        m.pose.orientation.w = std::cos(yaw / 2);
        m.scale.x = sx;
        m.scale.y = sy;
        m.scale.z = sz;
        m.color = c;
        return m;
      };
    const bool is_fire = std::string(s.name) == "fire";
    const float pulse = 0.5f + 0.5f * static_cast<float>(std::sin(t * 4.0));

    if (is_fire) {
      // 위험 반경: 숨 쉬듯 진해졌다 옅어진다
      const double d = 2 * fire_zone_radius_;
      Marker z0 = make(0, Marker::CYLINDER, 0.01, d, d, 0.02, rgba(1.0f, 0.15f, 0.05f, 0.10f + 0.16f * pulse));
      Marker z1 = make(1, Marker::CYLINDER, 0.02, d * 0.5, d * 0.5, 0.02, rgba(1.0f, 0.25f, 0.05f, 0.25f));
      z0.pose.position.x = z1.pose.position.x = l.x;   // 위험 반경은 표시 오프셋 없이 물체 자리에
      z0.pose.position.y = z1.pose.position.y = l.y;
      arr.markers.push_back(z0);
      arr.markers.push_back(z1);
    }
    // 화면(RViz 위에서 보기, Angle = view_angle) 기준 오른쪽·위쪽이 map 에서 어느 방향인지
    const double right_x = std::cos(view_angle_), right_y = std::sin(view_angle_);
    const double up_x = -std::sin(view_angle_), up_y = std::cos(view_angle_);

    // 아이콘 배지: icons/<클래스>.obj (납작한 원판 + 그림 텍스처). 그림 위쪽이 화면 위로 오게 view_angle 만큼 돌린다
    const double size = icon_size_ * (is_fire ? 1.0 + 0.1 * pulse : 1.0);
    Marker shadow = make(2, Marker::CYLINDER, 0.88, size * 1.08, size * 1.08, 0.01, rgba(0.0f, 0.0f, 0.0f, 0.35f));
    shadow.pose.position.x -= 0.03 * up_x;  // 화면 아래쪽으로 살짝 → 떠 있는 느낌
    shadow.pose.position.y -= 0.03 * up_y;
    arr.markers.push_back(shadow);
    if (has_icon(s.name)) {
      Marker icon = make(3, Marker::MESH_RESOURCE, 0.9, size, size, 1.0, rgba(0.0f, 0.0f, 0.0f, 0.0f), view_angle_);
      icon.mesh_resource = "package://limo_fire_perception/icons/" + std::string(s.name) + ".obj";
      icon.mesh_use_embedded_materials = true;  // 색 0 = 텍스처 그대로
      arr.markers.push_back(icon);
    } else {   // 아이콘이 없는 물체(COCO 80 종 등): 클래스 색 원판
      arr.markers.push_back(make(3, Marker::CYLINDER, 0.9, size * 0.7, size * 0.7, 0.02, rgba(s.r, s.g, s.b, 0.95f)));
    }
    // 바닥 발자국: 분할 마스크로 쌓은 칸(footprint_min_count 번 넘게 본 칸)을 클래스 색으로 칠함 — 지도(2D)에서 물체 모양.
    // 칸이 없으면(상자만 내는 탐지기) 잰 너비(없으면 0.2 m) 지름의 반투명 원
    Marker cells = make(9, Marker::CUBE_LIST, 0.01, cell_size_, cell_size_, 0.01, rgba(s.r, s.g, s.b, footprint_alpha_));
    cells.pose.position.x = 0.0;
    cells.pose.position.y = 0.0;
    for (const auto & kv : l.cells) {
      if (kv.second < (dyn_fp ? 1 : footprint_min_count_)) {continue;}   // 움직이는 것은 이 장 칸뿐이라 한 번 본 칸도
      int32_t gx, gy;
      cell_of(kv.first, gx, gy);
      geometry_msgs::msg::Point pt;
      pt.x = (gx + 0.5) * cell_size_;
      pt.y = (gy + 0.5) * cell_size_;
      cells.points.push_back(pt);
    }
    const double fp = std::clamp(l.ext > 0 ? l.ext : 0.2, 0.1, 3.0);
    Marker disc = make(4, Marker::CYLINDER, 0.03, fp, fp, 0.02, rgba(s.r, s.g, s.b, 0.45f));
    if (cells.points.empty()) {cells.action = Marker::DELETE;} else {disc.action = Marker::DELETE;}
    arr.markers.push_back(cells);
    arr.markers.push_back(disc);

    // 라벨: 밝은 지도 위에서도 읽히게 어두운 바탕판(위에서 내려다볼 때 글자 뒤) + 클래스 색 테두리.
    // 아이콘을 가리지 않게 label_offset(화면 기준 오른쪽·위쪽, m) 만큼 비켜 두고, 바탕판은 화면 가로로 길게
    const std::string text = std::string(s.tag) + "-" + std::to_string(l.number);  // RViz 글꼴은 띄어쓰기가 넓어 붙여 쓴다
    const double len = 0.15 * static_cast<double>(text.size()) + 0.12;
    auto shifted = [&](Marker m) {
        m.pose.position.x += label_offset_[0] * right_x + label_offset_[1] * up_x;
        m.pose.position.y += label_offset_[0] * right_y + label_offset_[1] * up_y;
        m.pose.orientation.z = std::sin(view_angle_ / 2);
        m.pose.orientation.w = std::cos(view_angle_ / 2);
        return m;
      };
    Marker edge = shifted(make(6, Marker::CUBE, 0.95, len + 0.05, 0.36, 0.01, rgba(s.r, s.g, s.b, 1.0f)));
    edge.ns = "labels";
    arr.markers.push_back(edge);
    Marker plate = shifted(make(7, Marker::CUBE, 0.97, len, 0.31, 0.01, rgba(0.08f, 0.09f, 0.11f, 0.92f)));
    plate.ns = "labels";
    arr.markers.push_back(plate);
    Marker label = shifted(make(5, Marker::TEXT_VIEW_FACING, 1.0, 0.0, 0.0, 0.24, rgba(1.0f, 1.0f, 1.0f, 1.0f)));
    label.ns = "labels";
    label.text = text;
    arr.markers.push_back(label);
  }

  void publish()
  {
    const double t = std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
    // 배지가 겹치면 서로 밀어 표시 자리만 벌린다 (배지 + 이름표 높이만큼, 몇 번 반복). 물체 자리에서 너무 멀어지지 않게 icon_size 까지만
    std::vector<size_t> on;
    for (size_t i = 0; i < landmarks_.size(); ++i) {if (landmarks_[i].number > 0 && !landmarks_[i].gone) {on.push_back(i);}}
    std::vector<double> px(on.size()), py(on.size());
    for (size_t k = 0; k < on.size(); ++k) {px[k] = landmarks_[on[k]].x; py[k] = landmarks_[on[k]].y;}
    const double need = icon_size_ * 1.15 + std::fabs(label_offset_[1]) * 0.6;
    for (int it = 0; it < 20; ++it) {
      bool moved = false;
      for (size_t a = 0; a < on.size(); ++a) {
        for (size_t b = a + 1; b < on.size(); ++b) {
          double ex = px[b] - px[a], ey = py[b] - py[a], d = std::hypot(ex, ey);
          if (d >= need) {continue;}
          if (d < 1e-6) {ex = 1.0; ey = 0.0; d = 1.0;} else {ex /= d; ey /= d;}
          const double push = (need - d) / 2;
          px[a] -= ex * push; py[a] -= ey * push;
          px[b] += ex * push; py[b] += ey * push;
          moved = true;
        }
      }
      for (size_t k = 0; k < on.size(); ++k) {   // 물체 자리에서 icon_size 넘게는 안 감
        const double ox = px[k] - landmarks_[on[k]].x, oy = py[k] - landmarks_[on[k]].y, od = std::hypot(ox, oy);
        if (od > icon_size_) {px[k] = landmarks_[on[k]].x + ox / od * icon_size_; py[k] = landmarks_[on[k]].y + oy / od * icon_size_;}
      }
      if (!moved) {break;}
    }
    visualization_msgs::msg::MarkerArray arr;
    for (size_t i = 0; i < landmarks_.size(); ++i) {   // 없어진 물체의 마커를 한 번 지움 (id = 번호 × 10 + 부품)
      Landmark & l = landmarks_[i];
      if (!l.gone || l.deleted) {continue;}
      l.deleted = true;
      for (int part = 0; part < 10; ++part) {
        for (const char * ns : {style_of(class_names_[l.cls]).name, "labels"}) {
          Marker m;
          m.header.frame_id = map_frame_;
          m.ns = ns;
          m.id = static_cast<int>(i) * 10 + part;
          m.action = Marker::DELETE;
          arr.markers.push_back(m);
        }
      }
    }
    for (size_t k = 0; k < on.size(); ++k) {
      append_markers(landmarks_[on[k]], on[k], t, arr, px[k] - landmarks_[on[k]].x, py[k] - landmarks_[on[k]].y);
    }
    if (!arr.markers.empty()) {marker_pub_->publish(arr);}
    publish_objects();
    if (++tick_ % 5 == 0) {publish_summary();}
  }

  // ~/objects: 물체 하나 = Detection3D (id "fire-1", 클래스·최고 점수, 위치, 크기 y = 앞면 넓이의 제곱근, 모르면 0)
  void publish_objects()
  {
    vision_msgs::msg::Detection3DArray out;
    out.header.stamp = now();
    out.header.frame_id = map_frame_;
    for (const auto & l : landmarks_) {
      if (l.number == 0 || l.gone) {continue;}
      auto & d = out.detections.emplace_back();
      d.header = out.header;
      d.id = class_names_[l.cls] + "-" + std::to_string(l.number);
      auto & h = d.results.emplace_back();
      h.hypothesis.class_id = class_names_[l.cls];
      h.hypothesis.score = l.best_score;
      d.bbox.center.position.x = l.x;
      d.bbox.center.position.y = l.y;
      d.bbox.center.position.z = l.z;
      d.bbox.center.orientation.w = 1.0;
      d.bbox.size.y = l.area > 0 ? std::sqrt(l.area) : 0.0;
      d.bbox.size.x = l.ext;   // 물체 너비 (m, 모르면 0)
      // 밑점 (마스크 3D 점의 가장 낮은 10 % 평균, map) — results[0].pose. 마스크로 못 잰 물체는 가운데와 같게
      h.pose.pose.position.x = l.base_n > 0 ? l.bx : l.x;
      h.pose.pose.position.y = l.base_n > 0 ? l.by : l.y;
      h.pose.pose.position.z = l.base_n > 0 ? l.bz : l.z;
      h.pose.pose.orientation.w = 1.0;
      h.pose.covariance[35] = std::max(0.0, now().seconds() - l.last_seen);   // 마지막으로 본 뒤 지난 초 (공분산 칸을 빌려 씀 — explorer 물체 후보 seen_s)
    }
    objects_pub_->publish(out);
  }

  // RViz 왼쪽 위 현황판: 화재 경고 + 종류별 개수 + 로봇에서 가장 가까운 것까지 거리
  void publish_summary()
  {
    double rx = 0, ry = 0;
    bool have_robot = false;
    try {
      const auto tf = tf_buffer_.lookupTransform(map_frame_, robot_frame_, tf2::TimePointZero);
      rx = tf.transform.translation.x;
      ry = tf.transform.translation.y;
      have_robot = true;
    } catch (const tf2::TransformException &) {
    }

    struct Row {int count = 0; double nearest = 1e9; int nearest_no = 0; double nearest_area = 0;};
    std::vector<Row> rows(class_names_.size());
    for (const auto & l : landmarks_) {
      if (l.number == 0 || l.gone) {continue;}
      Row & r = rows[l.cls];
      r.count++;
      const double d = std::hypot(l.x - rx, l.y - ry);
      if (d < r.nearest) {r.nearest = d; r.nearest_no = l.number; r.nearest_area = l.area;}
    }
    auto dist_text = [&](const Row & r, const Style & s) {
        if (!have_robot) {return std::string();}
        char buf[64];
        std::snprintf(buf, sizeof(buf), "%s-%d · %.1f m", s.tag, r.nearest_no, r.nearest);
        return std::string(buf);
      };

    std::string html;
    int lines = 1;
    const int fire = class_index("fire");
    if (fire >= 0 && rows[fire].count > 0) {
      std::string size_text;
      if (rows[fire].nearest_area > 0) {
        char buf[48];
        std::snprintf(buf, sizeof(buf), " · 크기 약 %.2f m²", rows[fire].nearest_area);
        size_text = buf;
      }
      html += "<div style='color:#ff6b4a;font-size:large;'><b>⚠ 화재 " + std::to_string(rows[fire].count) +
        "곳 감지</b></div><div style='color:#ffb4a2;'>가장 가까운 불 " + dist_text(rows[fire], style_of("fire")) +
        size_text + "</div>";
      lines += 2;
    }
    html += "<table cellspacing='0' cellpadding='3' width='100%'>";
    html += "<tr style='color:#9aa0a6;'><td>종류</td><td align='right'>개수</td><td align='right'>가장 가까운 것</td></tr>";
    for (size_t c = 0; c < class_names_.size(); ++c) {
      const Style & s = style_of(class_names_[c]);
      const Row & r = rows[c];
      const bool has = r.count > 0;
      html += std::string("<tr style='color:") + (has ? "#ffffff" : "#6b7075") + ";'>" +
        "<td><span style='color:" + hex(s) + ";'>■</span> " + s.ko + "</td>" +
        "<td align='right'><b>" + std::to_string(r.count) + "</b></td>" +
        "<td align='right'>" + (has ? dist_text(r, s) : std::string("-")) + "</td></tr>";
      lines++;
    }
    html += "</table><div style='color:#9aa0a6;font-size:small;'>" + std::to_string(min_hits_) +
      "번 이상 확인된 것만 · 거리는 로봇 기준 직선</div>";

    rviz_2d_overlay_msgs::msg::OverlayText o;
    o.action = rviz_2d_overlay_msgs::msg::OverlayText::ADD;
    o.horizontal_alignment = rviz_2d_overlay_msgs::msg::OverlayText::LEFT;  // 지도는 RViz 설정에서 오른쪽으로 비켜 둠
    o.vertical_alignment = rviz_2d_overlay_msgs::msg::OverlayText::TOP;
    o.horizontal_distance = 12;
    o.vertical_distance = 12;
    o.width = 400;
    o.height = 60 + 27 * lines;
    o.text_size = 11.0;
    o.font = "Noto Sans CJK KR";
    o.bg_color = rgba(0.09f, 0.10f, 0.12f, 0.85f);
    o.fg_color = rgba(1.0f, 1.0f, 1.0f, 1.0f);
    o.text = html;
    overlay_pub_->publish(o);
  }

  tf2_ros::Buffer tf_buffer_;
  tf2_ros::TransformListener tf_listener_;
  std::string map_frame_, robot_frame_;
  double min_depth_, max_depth_, box_shrink_, depth_percentile_, mask_depth_percentile_, merge_radius_, min_score_,
    fire_zone_radius_;
  std::vector<double> merge_radii_;
  std::vector<double> label_offset_;
  double icon_size_, view_angle_;
  int min_hits_;
  std::vector<std::string> class_names_;
  std::vector<Landmark> landmarks_;
  std::vector<int> counters_;
  std::string objects_file_;
  rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr save_srv_, load_srv_;
  rclcpp::Subscription<std_msgs::msg::String>::SharedPtr save_to_sub_;
  int neg_misses_ = 20;
  double neg_max_depth_ = 3.0, neg_margin_ = 0.1, neg_occl_m_ = 0.3, dynamic_ttl_ = 10.0;
  double dynamic_alpha_ = 1.0, dynamic_speed_ = 1.5, dynamic_max_radius_ = 2.0;
  int dynamic_neg_misses_ = 5;
  double move_dist_ = 0.5;
  int move_frames_ = 10;
  std::vector<std::string> dynamic_;
  std::vector<bool> dyn_cls_;   // 클래스 번호 → 움직이는 클래스
  double base_height_ = 0.145, base_rows_frac_ = 0.05, fuse_tol_m_ = 0.3, fuse_tol_frac_ = 0.15;
  std::vector<std::string> prefer_floor_;
  std::vector<std::string> ignored_;
  double cell_size_ = 0.05, footprint_max_z_ = 2.0;
  int footprint_stride_ = 3, footprint_min_count_ = 2;
  float footprint_alpha_ = 0.25f;
  unsigned tick_ = 0;
  sensor_msgs::msg::CameraInfo::ConstSharedPtr info_;
  std::deque<sensor_msgs::msg::Image::ConstSharedPtr> masks_;
  std::deque<Job> jobs_;
  Stats st_;
  double stats_period_ = 10.0;
  double t_obs_ = 0;   // 처리 중인 장을 받은 때 (s, 노드 시계)
  double tf_wait_ = 0.2;
  size_t max_pending_ = 10;
  size_t n_no_tf_ = 0, n_overflow_ = 0;   // TF 가 없어·줄이 넘쳐 버린 장 수
  // 장마다 다시 쓰는 작업 버퍼 (할당을 줄이려고 — 한 스레드에서만 씀)
  std::vector<float> vals_, zs_;
  std::vector<int> xs_;
  std::vector<size_t> seen_now_;
  std::vector<std::array<float, 3>> pts_;
  std::vector<tf2::Vector3> kept_;
  std::vector<int> dxs_;   // 마스크 열 → 깊이 열 (depth_cols)
  double dxs_sx_ = 0;
  int dxs_dw_ = 0;

  message_filters::Subscriber<vision_msgs::msg::Detection2DArray> det_sub_;
  message_filters::PassThrough<sensor_msgs::msg::Image> depth_in_;
  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr depth_sub_;
  std::shared_ptr<message_filters::Synchronizer<Policy>> sync_;
  rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr info_sub_;
  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr mask_sub_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr marker_pub_;
  rclcpp::Publisher<rviz_2d_overlay_msgs::msg::OverlayText>::SharedPtr overlay_pub_;
  rclcpp::Publisher<vision_msgs::msg::Detection3DArray>::SharedPtr objects_pub_;
  rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr clear_srv_;
  rclcpp::TimerBase::SharedPtr timer_, proc_timer_;
};

}  // namespace limo_fire_perception

RCLCPP_COMPONENTS_REGISTER_NODE(limo_fire_perception::ObjectMapNode)
