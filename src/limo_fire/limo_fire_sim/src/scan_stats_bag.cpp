// 녹화(rosbag2)의 라이다 오차 통계 → YAML. 실물·Webots 녹화 어느 쪽이든 같은 표가 나와 학습 시뮬(noise_check)과 비교한다 (V3).
//   scan_stats_bag <출력.yaml> <녹화 폴더> [녹화 폴더 …] [--scan /scan] [--odom /odom] [--margin 0.5]
// 녹화마다 정지 구간 하나: /odom 속도가 0 인 가장 긴 구간(앞뒤 margin 초 뺌). /odom 이 없으면(센서만 켜고 세워 둔 녹화,
// robot.launch.xml base:=false) 녹화 전체. 그 안의 가장 흔한 빔 수 장으로 빔별 통계를 내고, 여러 녹화(장소)는 빔 단위로 모은다.
// 한 바퀴 점 수(2π / angle_increment) 분위는 모든 녹화의 모든 장에서.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include <nav_msgs/msg/odometry.hpp>
#include <rclcpp/serialization.hpp>
#include <rosbag2_cpp/reader.hpp>
#include <sensor_msgs/msg/laser_scan.hpp>

#include "limo_fire_sim/scan_stats.hpp"

namespace
{
double stamp(const builtin_interfaces::msg::Time & t) {return t.sec + t.nanosec * 1e-9;}

struct Opts
{
  std::string scan_topic = "/scan", odom_topic = "/odom";
  double margin = 0.5;
};

// 녹화 하나 → 정지 구간 묶음. n_rev 에 모든 장의 한 바퀴 점 수를 더하고 설명 줄을 note 에
bool read_bag(const std::string & path, const Opts & o, scan_stats::Set & set, std::vector<float> & n_rev, std::string & note)
{
  rosbag2_cpp::Reader reader;
  reader.open(path);
  rclcpp::Serialization<sensor_msgs::msg::LaserScan> scan_ser;
  rclcpp::Serialization<nav_msgs::msg::Odometry> odom_ser;
  struct Scan { double t; float inc; std::vector<float> r; };
  std::vector<Scan> scans;
  std::vector<std::pair<double, bool>> still;   // (시각, 멈춤)
  while (reader.has_next()) {
    const auto m = reader.read_next();
    rclcpp::SerializedMessage raw(*m->serialized_data);
    if (m->topic_name == o.scan_topic) {
      sensor_msgs::msg::LaserScan s;
      scan_ser.deserialize_message(&raw, &s);
      std::vector<float> r(s.ranges.begin(), s.ranges.end());
      for (float & x : r) {if (!std::isfinite(x)) {x = 0;}}
      scans.push_back({stamp(s.header.stamp), s.angle_increment, std::move(r)});
    } else if (m->topic_name == o.odom_topic) {
      nav_msgs::msg::Odometry od;
      odom_ser.deserialize_message(&raw, &od);
      still.emplace_back(stamp(od.header.stamp),
        std::fabs(od.twist.twist.linear.x) < 1e-3 && std::fabs(od.twist.twist.angular.z) < 1e-3);
    }
  }
  if (scans.empty()) {
    std::fprintf(stderr, "%s: %s 가 녹화에 없음\n", path.c_str(), o.scan_topic.c_str());
    return false;
  }
  // 가장 긴 정지 구간 (오도메트리가 없으면 녹화 전체)
  double best0 = scans.front().t, best1 = scans.back().t, s0 = -1;
  if (!still.empty()) {
    best0 = best1 = 0;
    for (size_t i = 0; i < still.size(); ++i) {
      if (still[i].second && s0 < 0) {s0 = still[i].first;}
      if ((!still[i].second || i + 1 == still.size()) && s0 >= 0) {
        if (still[i].first - s0 > best1 - best0) {best0 = s0; best1 = still[i].first;}
        s0 = -1;
      }
    }
  }
  std::vector<std::vector<float>> fixed;
  for (const auto & s : scans) {
    n_rev.push_back(2 * M_PI / s.inc);
    if (s.t > best0 + o.margin && s.t < best1 - o.margin) {fixed.push_back(s.r);}
  }
  fixed = scan_stats::mode_length(fixed);
  if (fixed.empty()) {
    std::fprintf(stderr, "%s: 정지 구간에 스캔이 없음\n", path.c_str());
    return false;
  }
  double inc_sum = 0;
  int k = 0;
  for (const auto & s : scans) {
    if (s.t > best0 + o.margin && s.t < best1 - o.margin && s.r.size() == fixed[0].size()) {inc_sum += s.inc; ++k;}
  }
  set.scans = std::move(fixed);
  set.inc = inc_sum / std::max(k, 1);
  char buf[512];
  std::snprintf(buf, sizeof(buf), "%s (전체 %zu 장, 정지 %.1f s%s, 빔 %zu 개 장 %zu)", path.c_str(), scans.size(), best1 - best0,
    still.empty() ? " = 녹화 전체(오도메트리 없음)" : "", set.scans[0].size(), set.scans.size());
  note = buf;
  return true;
}
}  // namespace

int main(int argc, char ** argv)
{
  Opts o;
  std::vector<std::string> bags;
  std::string out;
  for (int i = 1; i < argc; ++i) {
    const std::string k = argv[i];
    if (k == "--scan" && i + 1 < argc) {
      o.scan_topic = argv[++i];
    } else if (k == "--odom" && i + 1 < argc) {
      o.odom_topic = argv[++i];
    } else if (k == "--margin" && i + 1 < argc) {
      o.margin = std::stod(argv[++i]);
    } else if (out.empty()) {
      out = k;
    } else {
      bags.push_back(k);
    }
  }
  if (out.empty() || bags.empty()) {
    std::fprintf(stderr, "scan_stats_bag <출력.yaml> <녹화 폴더> [녹화 폴더 …] [--scan /scan] [--odom /odom] [--margin 0.5]\n");
    return 2;
  }
  std::vector<scan_stats::Set> sets;
  std::vector<float> n_rev;
  std::string source;
  for (const auto & b : bags) {
    scan_stats::Set set;
    std::string note;
    if (!read_bag(b, o, set, n_rev, note)) {return 1;}
    sets.push_back(std::move(set));
    std::printf("%s\n", note.c_str());
    source += (source.empty() ? "" : "; ") + note;
  }
  auto sum = scan_stats::summarize(sets);
  sum.n_rev_q = scan_stats::n_rev_quantiles(n_rev);
  FILE * f = std::fopen(out.c_str(), "w");
  if (!f) {std::perror(out.c_str()); return 1;}
  scan_stats::write_yaml(f, sum, source);
  std::fclose(f);
  return 0;
}
