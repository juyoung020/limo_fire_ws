// 실물 녹화 → 다시 돌리기 파일(.lfrec): 장마다 라이다 값과 그때 로봇 자세(Cartographer map → base_link)·속도(EKF).
// 학습 시뮬 replay_check 가 같은 자세·시각에서 시뮬 라이다를 쏴 실물과 빔마다 비교한다 (시뮬 = 현실 확인).
//   bag_export <녹화 폴더> <출력.lfrec> [--scan /scan] [--twist /odometry/filtered]
// 녹화에 /map 이 있으면 마지막 지도를 <출력>.map.pgm · <출력>.map.yaml (Nav2 map_server 형식, scale: 칸 확률 그대로)로도 쓴다
// → 학습 시뮬 map2scene (막힘 판단 문턱은 거기서)
// .lfrec (리틀 엔디언): "LFRC" u32 버전 1 | u32 장 수 | 장마다 { f64 stamp, f32 angle_min, f32 inc, f32 time_inc, f32 scan_time,
//   f32 x, y, yaw (map 에서 base_link), f32 v, w, u32 n, f32 ranges[n] } — 자세를 못 구한 장은 뺀다
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include <nav_msgs/msg/occupancy_grid.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <rclcpp/serialization.hpp>
#include <rosbag2_cpp/reader.hpp>
#include <sensor_msgs/msg/laser_scan.hpp>
#include <tf2_msgs/msg/tf_message.hpp>

namespace
{
double stamp(const builtin_interfaces::msg::Time & t) {return t.sec + t.nanosec * 1e-9;}

struct P { double t, x, y, yaw; };

// 시각 순 목록에서 t 의 자세 (선형, 각은 짧은 쪽). 범위 밖이면 false
bool interp(const std::vector<P> & v, double t, P & out)
{
  if (v.empty() || t < v.front().t || t > v.back().t) {return false;}
  size_t lo = 0, hi = v.size() - 1;
  while (hi - lo > 1) {
    const size_t m = (lo + hi) / 2;
    (v[m].t <= t ? lo : hi) = m;
  }
  const P & a = v[lo], & b = v[hi];
  const double f = b.t > a.t ? (t - a.t) / (b.t - a.t) : 0;
  const double dy = std::remainder(b.yaw - a.yaw, 2 * M_PI);
  out = {t, a.x + f * (b.x - a.x), a.y + f * (b.y - a.y), a.yaw + f * dy};
  return true;
}
}  // namespace

int main(int argc, char ** argv)
{
  if (argc < 3) {
    std::fprintf(stderr, "bag_export <녹화 폴더> <출력.lfrec> [--scan /scan] [--twist /odometry/filtered]\n");
    return 2;
  }
  std::string scan_topic = "/scan", twist_topic = "/odometry/filtered";
  for (int i = 3; i + 1 < argc; i += 2) {
    const std::string k = argv[i];
    if (k == "--scan") {scan_topic = argv[i + 1];} else if (k == "--twist") {twist_topic = argv[i + 1];}
  }
  rosbag2_cpp::Reader reader;
  reader.open(argv[1]);
  rclcpp::Serialization<sensor_msgs::msg::LaserScan> scan_ser;
  rclcpp::Serialization<nav_msgs::msg::Odometry> odom_ser;
  rclcpp::Serialization<tf2_msgs::msg::TFMessage> tf_ser;
  rclcpp::Serialization<nav_msgs::msg::OccupancyGrid> map_ser;
  nav_msgs::msg::OccupancyGrid last_map;
  bool have_map = false;
  std::vector<sensor_msgs::msg::LaserScan> scans;
  std::vector<P> map_odom, odom_base, twist;   // twist: x = v, y = w
  while (reader.has_next()) {
    const auto m = reader.read_next();
    rclcpp::SerializedMessage raw(*m->serialized_data);
    if (m->topic_name == scan_topic) {
      sensor_msgs::msg::LaserScan s;
      scan_ser.deserialize_message(&raw, &s);
      scans.push_back(std::move(s));
    } else if (m->topic_name == twist_topic) {
      nav_msgs::msg::Odometry o;
      odom_ser.deserialize_message(&raw, &o);
      twist.push_back({stamp(o.header.stamp), o.twist.twist.linear.x, o.twist.twist.angular.z, 0});
    } else if (m->topic_name == "/map") {
      map_ser.deserialize_message(&raw, &last_map);
      have_map = true;
    } else if (m->topic_name == "/tf") {
      tf2_msgs::msg::TFMessage tf;
      tf_ser.deserialize_message(&raw, &tf);
      for (const auto & t : tf.transforms) {
        const auto & q = t.transform.rotation;
        const P p{stamp(t.header.stamp), t.transform.translation.x, t.transform.translation.y,
          std::atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))};
        if (t.header.frame_id == "map" && t.child_frame_id == "odom") {map_odom.push_back(p);}
        if (t.header.frame_id == "odom" && t.child_frame_id == "base_link") {odom_base.push_back(p);}
      }
    }
  }
  auto by_time = [](std::vector<P> & v) {std::sort(v.begin(), v.end(), [](const P & a, const P & b) {return a.t < b.t;});};
  by_time(map_odom);
  by_time(odom_base);
  by_time(twist);
  FILE * f = std::fopen(argv[2], "wb");
  if (!f) {std::perror(argv[2]); return 1;}
  auto w = [&](const void * p, size_t n) {std::fwrite(p, 1, n, f);};
  const uint32_t ver = 1;
  uint32_t count = 0;
  w("LFRC", 4);
  w(&ver, 4);
  const long count_pos = std::ftell(f);
  w(&count, 4);
  for (const auto & s : scans) {
    const double t = stamp(s.header.stamp);
    P mo, ob, tw;
    if (!interp(map_odom, t, mo) || !interp(odom_base, t, ob) || !interp(twist, t, tw)) {continue;}
    const double c = std::cos(mo.yaw), sn = std::sin(mo.yaw);
    const float pose[3] = {float(mo.x + c * ob.x - sn * ob.y), float(mo.y + sn * ob.x + c * ob.y),
      float(std::remainder(mo.yaw + ob.yaw, 2 * M_PI))};
    const float meta[4] = {s.angle_min, s.angle_increment, s.time_increment, s.scan_time};
    const float vel[2] = {float(tw.x), float(tw.y)};
    const uint32_t n = static_cast<uint32_t>(s.ranges.size());
    std::vector<float> r(s.ranges.begin(), s.ranges.end());
    for (float & x : r) {if (!std::isfinite(x)) {x = 0;}}
    w(&t, 8); w(meta, 16); w(pose, 12); w(vel, 8); w(&n, 4); w(r.data(), 4 * n);
    ++count;
  }
  std::fseek(f, count_pos, SEEK_SET);
  w(&count, 4);
  std::fclose(f);
  std::printf("%s: 스캔 %zu 장 중 자세 있는 %u 장 → %s (map→odom %zu, odom→base %zu, 속도 %zu)\n", argv[1], scans.size(), count,
    argv[2], map_odom.size(), odom_base.size(), twist.size());
  if (have_map) {
    // Nav2 map_server scale: 픽셀 = 255 − 확률(%)·255/100 (0 막힘 ~ 255 빈 곳), 모름 205. PGM 은 위 행이 y 가 큰 쪽
    const std::string base = std::string(argv[2]) + ".map";
    const auto & info = last_map.info;
    FILE * pg = std::fopen((base + ".pgm").c_str(), "wb");
    std::fprintf(pg, "P5\n%u %u\n255\n", info.width, info.height);
    for (int y = static_cast<int>(info.height) - 1; y >= 0; --y) {
      for (uint32_t x = 0; x < info.width; ++x) {
        const int8_t v = last_map.data[y * info.width + x];
        std::fputc(v < 0 ? 205 : 255 - (v * 255) / 100, pg);
      }
    }
    std::fclose(pg);
    const auto & q = info.origin.orientation;
    FILE * ym = std::fopen((base + ".yaml").c_str(), "w");
    const std::string img = base.substr(base.find_last_of('/') + 1) + ".pgm";
    std::fprintf(ym, "image: %s\nmode: scale\nresolution: %.4f\norigin: [%.4f, %.4f, %.4f]\nnegate: 0\noccupied_thresh: 0.65\nfree_thresh: 0.25\n",
      img.c_str(), info.resolution, info.origin.position.x, info.origin.position.y, 2 * std::atan2(q.z, q.w));
    std::fclose(ym);
    std::printf("마지막 지도 %u×%u (%.2f m) → %s.pgm · .yaml\n", info.width, info.height, info.resolution, base.c_str());
  }
  return 0;
}
