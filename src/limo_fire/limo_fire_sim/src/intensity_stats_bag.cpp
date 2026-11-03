// 녹화(rosbag2)의 라이다 세기 통계 → YAML (intensity_stats.hpp). 학습 시뮬 intensity_check 가 세기 모델을 맞추고 확인한다.
//   intensity_stats_bag <출력.yaml> <녹화 폴더> [녹화 폴더 …] [--margin 0.5]
// 녹화마다 정지 구간 하나 (scan_stats_bag 과 같음: /odom 속도 0 인 가장 긴 구간, 없으면 녹화 전체), 그 안의 가장 흔한 빔 수 장.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include <nav_msgs/msg/odometry.hpp>
#include <rclcpp/serialization.hpp>
#include <rosbag2_cpp/reader.hpp>
#include <sensor_msgs/msg/laser_scan.hpp>

#include "limo_fire_sim/intensity_stats.hpp"

namespace
{
double stamp(const builtin_interfaces::msg::Time & t) {return t.sec + t.nanosec * 1e-9;}
}

int main(int argc, char ** argv)
{
  std::string out;
  std::vector<std::string> bags;
  double margin = 0.5;
  for (int i = 1; i < argc; ++i) {
    const std::string k = argv[i];
    if (k == "--margin" && i + 1 < argc) {margin = std::stod(argv[++i]);} else if (out.empty()) {out = k;} else {bags.push_back(k);}
  }
  if (out.empty() || bags.empty()) {
    std::fprintf(stderr, "intensity_stats_bag <출력.yaml> <녹화 폴더> [녹화 폴더 …] [--margin 0.5]\n");
    return 2;
  }
  std::vector<intensity_stats::Set> sets;
  std::string source;
  for (const auto & path : bags) {
    rosbag2_cpp::Reader reader;
    reader.open(path);
    rclcpp::Serialization<sensor_msgs::msg::LaserScan> scan_ser;
    rclcpp::Serialization<nav_msgs::msg::Odometry> odom_ser;
    struct Scan { double t; float inc; std::vector<float> r, i; };
    std::vector<Scan> scans;
    std::vector<std::pair<double, bool>> still;
    while (reader.has_next()) {
      const auto m = reader.read_next();
      rclcpp::SerializedMessage raw(*m->serialized_data);
      if (m->topic_name == "/scan") {
        sensor_msgs::msg::LaserScan s;
        scan_ser.deserialize_message(&raw, &s);
        std::vector<float> r(s.ranges.begin(), s.ranges.end());
        for (float & x : r) {if (!std::isfinite(x)) {x = 0;}}
        std::vector<float> in(s.intensities.begin(), s.intensities.end());
        in.resize(r.size(), 0.f);
        scans.push_back({stamp(s.header.stamp), s.angle_increment, std::move(r), std::move(in)});
      } else if (m->topic_name == "/odom") {
        nav_msgs::msg::Odometry od;
        odom_ser.deserialize_message(&raw, &od);
        still.emplace_back(stamp(od.header.stamp), std::fabs(od.twist.twist.linear.x) < 1e-3 && std::fabs(od.twist.twist.angular.z) < 1e-3);
      }
    }
    if (scans.empty()) {std::fprintf(stderr, "%s: /scan 없음\n", path.c_str()); return 1;}
    double b0 = scans.front().t, b1 = scans.back().t, s0 = -1;
    if (!still.empty()) {
      b0 = b1 = 0;
      for (size_t i = 0; i < still.size(); ++i) {
        if (still[i].second && s0 < 0) {s0 = still[i].first;}
        if ((!still[i].second || i + 1 == still.size()) && s0 >= 0) {
          if (still[i].first - s0 > b1 - b0) {b0 = s0; b1 = still[i].first;}
          s0 = -1;
        }
      }
    }
    // 가장 흔한 빔 수
    std::vector<int> cnt(1024, 0);
    for (const auto & s : scans) {if (s.t > b0 + margin && s.t < b1 - margin && s.r.size() < cnt.size()) {++cnt[s.r.size()];}}
    const size_t nb = std::max_element(cnt.begin(), cnt.end()) - cnt.begin();
    intensity_stats::Set set;
    double inc = 0;
    for (const auto & s : scans) {
      if (s.t > b0 + margin && s.t < b1 - margin && s.r.size() == nb) {
        set.range.scans.push_back(s.r);
        set.inten.push_back(s.i);
        inc += s.inc;
      }
    }
    if (set.inten.empty()) {std::fprintf(stderr, "%s: 정지 구간에 스캔이 없음\n", path.c_str()); return 1;}
    set.range.inc = inc / set.inten.size();
    char buf[256];
    std::snprintf(buf, sizeof(buf), "%s (정지 %.1f s, 빔 %zu 개 장 %zu)", path.c_str(), b1 - b0, nb, set.inten.size());
    std::printf("%s\n", buf);
    source += (source.empty() ? "" : "; ") + std::string(buf);
    sets.push_back(std::move(set));
  }
  const auto sum = intensity_stats::summarize(sets);
  FILE * f = std::fopen(out.c_str(), "w");
  if (!f) {std::perror(out.c_str()); return 1;}
  intensity_stats::write_yaml(f, sum, source);
  std::fclose(f);
  return 0;
}
