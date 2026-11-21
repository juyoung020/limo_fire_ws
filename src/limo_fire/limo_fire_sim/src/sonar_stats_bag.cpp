// 녹화(rosbag2)의 초음파 통계 → YAML (sonar_stats.hpp). 실물 값으로 시뮬 초음파 흉내(config/sonar_sim_real.yaml)를 정한다.
//   sonar_stats_bag <출력.yaml> <녹화 폴더> [녹화 폴더 …] [--topics /ultrasonic/front_left,/ultrasonic/front_right]
//                   [--odom /odom] [--static] [--min-seg 5] [--truth 1.00] [--no-echo-ratio 0.99]
// 거리: /odom 바퀴 v·ω 가 0.5 s 넘게 0 인 구간(--min-seg 초 이상)마다. --static 이면 녹화 전체를 한 구간으로(로봇을 손으로 세워 둔 녹화).
//   --truth: 줄자로 잰 센서 앞면 → 벽 거리 (m). 주면 구간마다 치우침(중앙값 − 줄자)을 낸다. 거리·각도마다 녹화를 따로 해 따로 돌린다
//   (data_plan.md 초음파 절 — 0.3·0.5·1·2 m × 벽과 0·15·30°).
// 시각(녹화 전체, 움직일 때 포함): 메시지 시각 − 그때까지 받은 마지막 odom → base_link TF 시각 — 양수면 연기 모드가 그 메아리 시각의 TF 를 아직 못 가짐
//   (smoke_mode echo_tf_tolerance 를 넘으면 버림). 메시지 시각 − 녹화기가 받은 시각도 같이.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#include <nav_msgs/msg/odometry.hpp>
#include <rclcpp/serialization.hpp>
#include <rosbag2_cpp/reader.hpp>
#include <sensor_msgs/msg/range.hpp>
#include <tf2_msgs/msg/tf_message.hpp>

#include "limo_fire_sim/sonar_stats.hpp"

namespace
{

double sec(const builtin_interfaces::msg::Time & t) {return t.sec + t.nanosec * 1e-9;}

void put(FILE * f, const char * ind, const char * key, const sonar_stats::Summary & s, double scale = 1.0)
{
  std::fprintf(f, "%s%s: {n: %zu, mean: %.4f, sd: %.4f, p05: %.4f, p50: %.4f, p95: %.4f, min: %.4f, max: %.4f}\n", ind, key, s.n,
    s.mean * scale, s.sd * scale, s.p05 * scale, s.p50 * scale, s.p95 * scale, s.min * scale, s.max * scale);
}

}  // namespace

int main(int argc, char ** argv)
{
  std::string out, odom_topic = "/odom", tf_parent = "odom", tf_child = "base_link";
  std::vector<std::string> bags, topics = {"/ultrasonic/front_left", "/ultrasonic/front_right"};
  bool is_static = false;
  double min_seg = 5.0, truth = NAN, no_echo_ratio = 0.99;
  for (int i = 1; i < argc; ++i) {
    const std::string k = argv[i];
    if (k == "--topics" && i + 1 < argc) {
      topics.clear();
      std::stringstream ss(argv[++i]);
      for (std::string t; std::getline(ss, t, ',');) {if (!t.empty()) {topics.push_back(t);}}
    } else if (k == "--odom" && i + 1 < argc) {
      odom_topic = argv[++i];
    } else if (k == "--static") {
      is_static = true;
    } else if (k == "--min-seg" && i + 1 < argc) {
      min_seg = std::stod(argv[++i]);
    } else if (k == "--truth" && i + 1 < argc) {
      truth = std::stod(argv[++i]);
    } else if (k == "--no-echo-ratio" && i + 1 < argc) {
      no_echo_ratio = std::stod(argv[++i]);
    } else if (out.empty()) {
      out = k;
    } else {
      bags.push_back(k);
    }
  }
  if (out.empty() || bags.empty() || topics.empty() || !(no_echo_ratio > 0.0 && no_echo_ratio <= 1.0)) {
    std::fprintf(stderr, "sonar_stats_bag <출력.yaml> <녹화 폴더> [녹화 폴더 …] [--topics a,b] [--odom /odom] [--static] [--min-seg 5] [--truth 1.00] [--no-echo-ratio (0,1]]\n");
    return 2;
  }
  FILE * f = std::fopen(out.c_str(), "w");
  if (!f) {
    std::fprintf(stderr, "출력 파일을 못 엶: %s\n", out.c_str());
    return 1;
  }
  std::fprintf(f, "# sonar_stats_bag — 거리 m, 시각 ms. echo_ratio = 메아리 있는 메시지 비율 (1 − 놓침)\n");
  if (std::isfinite(truth)) {std::fprintf(f, "truth_m: %.3f\n", truth);}
  std::fprintf(f, "bags:\n");

  rclcpp::Serialization<sensor_msgs::msg::Range> range_ser;
  rclcpp::Serialization<nav_msgs::msg::Odometry> odom_ser;
  rclcpp::Serialization<tf2_msgs::msg::TFMessage> tf_ser;
  for (const auto & b : bags) {
    rosbag2_cpp::Reader reader;
    reader.open(b);
    sonar_stats::StillDetector still;
    bool have_odom = false;
    double t_start = NAN, last_tf = NAN;
    std::map<std::string, std::vector<sonar_stats::Segment>> segs;   // 토픽 → 구간들
    std::map<std::string, sonar_stats::Segment> cur;
    std::map<std::string, std::vector<double>> tf_ahead, rx_ahead;
    bool was_still = false;
    auto close_all = [&]() {
        for (auto & [t, s] : cur) {
          if (s.msgs && (s.t1 - s.t0) >= min_seg) {segs[t].push_back(s);}
        }
        cur.clear();
      };
    while (reader.has_next()) {
      const auto m = reader.read_next();
      const double rx = m->time_stamp * 1e-9;
      if (!std::isfinite(t_start)) {t_start = rx;}
      rclcpp::SerializedMessage raw(*m->serialized_data);
      if (m->topic_name == odom_topic) {
        nav_msgs::msg::Odometry o;
        odom_ser.deserialize_message(&raw, &o);
        have_odom = true;
        const bool s = still.update(rx, o.twist.twist.linear.x, o.twist.twist.angular.z);
        if (was_still && !s && !is_static) {close_all();}
        was_still = s;
      } else if (m->topic_name == "/tf") {
        tf2_msgs::msg::TFMessage tf;
        tf_ser.deserialize_message(&raw, &tf);
        for (const auto & t : tf.transforms) {
          if (t.header.frame_id == tf_parent && t.child_frame_id == tf_child) {last_tf = std::max(std::isfinite(last_tf) ? last_tf : 0.0, sec(t.header.stamp));}
        }
      } else if (std::find(topics.begin(), topics.end(), m->topic_name) != topics.end()) {
        sensor_msgs::msg::Range r;
        range_ser.deserialize_message(&raw, &r);
        const double st = sec(r.header.stamp);
        rx_ahead[m->topic_name].push_back(st - rx);
        if (std::isfinite(last_tf)) {tf_ahead[m->topic_name].push_back(st - last_tf);}
        if (is_static || (have_odom && was_still)) {
          auto & seg = cur[m->topic_name];
          seg.no_echo_ratio = no_echo_ratio;
          seg.add(rx - t_start, r.range, r.min_range, r.max_range);
        }
      }
    }
    close_all();
    if (!is_static && !have_odom) {std::fprintf(stderr, "%s: %s 없음 — 정지 구간을 못 찾음 (--static 이나 --odom 확인)\n", b.c_str(), odom_topic.c_str());}

    std::fprintf(f, "  - bag: %s\n    topics:\n", b.c_str());
    for (const auto & t : topics) {
      std::fprintf(f, "      %s:\n", t.c_str());
      put(f, "        ", "stamp_minus_tf_ms", sonar_stats::summarize(tf_ahead[t]), 1e3);
      put(f, "        ", "stamp_minus_rx_ms", sonar_stats::summarize(rx_ahead[t]), 1e3);
      std::fprintf(f, "        segments:\n");
      if (segs[t].empty()) {std::fprintf(f, "          []   # 정지 구간 없음 (--min-seg %.1f s)\n", min_seg);}
      for (const auto & s : segs[t]) {
        const auto e = sonar_stats::summarize(s.echo);
        const auto g = sonar_stats::summarize(s.gap);
        std::fprintf(f, "          - {t0: %.1f, t1: %.1f, msgs: %zu, echo_ratio: %.3f, rate_hz: %.2f}\n", s.t0, s.t1, s.msgs, s.echo_ratio(),
          g.n ? 1.0 / g.p50 : NAN);
        put(f, "            ", "range", e);
        if (std::isfinite(truth) && e.n) {std::fprintf(f, "            bias_m: %.4f   # 중앙값 − 줄자\n", e.p50 - truth);}
      }
      std::printf("%s %s: 정지 구간 %zu 개, 시각 − TF 중앙 %.1f ms (p95 %.1f)\n", b.c_str(), t.c_str(), segs[t].size(),
        sonar_stats::summarize(tf_ahead[t]).p50 * 1e3, sonar_stats::summarize(tf_ahead[t]).p95 * 1e3);
    }
  }
  std::fclose(f);
  return 0;
}
