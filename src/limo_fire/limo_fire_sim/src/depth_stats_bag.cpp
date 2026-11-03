// 녹화(rosbag2)의 깊이 영상 오차 통계 → YAML (depth_stats.hpp). 학습 시뮬 depth_check 가 이 YAML 로 깊이 모델을 맞추고 확인한다.
//   depth_stats_bag <출력.yaml> <녹화 폴더> [녹화 폴더 …] [--topic /camera/depth/image_raw] [--static] [--every 1] [--cam-h 0.205] [--cam-pitch 0]
//                   [--dump 첫장.pgm]
// 픽셀별 시간 σ: --static 이면 녹화 전체를 한 구간으로, 아니면 /odom 바퀴 v·ω 가 0.5 s 넘게 0 인 구간(3 s 이상)마다.
// 내부값은 녹화의 /camera/depth/camera_info (정렬 안 한 640 × 400). --cam-h·--cam-pitch: 바닥 구멍 통계에 쓰는 카메라 광축 높이 (m)·아래로 숙인 각 (rad). 출력 floor_plane 이 녹화로 잰 값.
#include <algorithm>
#include <cstdio>
#include <string>
#include <vector>

#include <nav_msgs/msg/odometry.hpp>
#include <rclcpp/serialization.hpp>
#include <rosbag2_cpp/reader.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <sensor_msgs/msg/image.hpp>

#include "limo_fire_sim/depth_stats.hpp"

int main(int argc, char ** argv)
{
  std::string out, topic = "/camera/depth/image_raw", dump;
  std::vector<std::string> bags;
  bool is_static = false;
  int every = 1;
  float cam_h = 0.205f, cam_pitch = 0.f;
  for (int i = 1; i < argc; ++i) {
    const std::string k = argv[i];
    if (k == "--topic" && i + 1 < argc) {
      topic = argv[++i];
    } else if (k == "--static") {
      is_static = true;
    } else if (k == "--every" && i + 1 < argc) {
      every = std::stoi(argv[++i]);
    } else if (k == "--cam-pitch" && i + 1 < argc) {
      cam_pitch = std::stof(argv[++i]);
    } else if (k == "--cam-h" && i + 1 < argc) {
      cam_h = std::stof(argv[++i]);
    } else if (k == "--dump" && i + 1 < argc) {
      dump = argv[++i];
    } else if (out.empty()) {
      out = k;
    } else {
      bags.push_back(k);
    }
  }
  if (out.empty() || bags.empty()) {
    std::fprintf(stderr, "depth_stats_bag <출력.yaml> <녹화 폴더> [녹화 폴더 …] [--topic ..] [--static] [--every 1] [--cam-h 0.205] [--dump ..]\n");
    return 2;
  }
  depth_stats::Acc acc;
  depth_stats::Temporal tmp;
  std::string source;
  constexpr int kMinSegFrames = 90;   // 3 s (30 Hz)
  for (const auto & b : bags) {
    rosbag2_cpp::Reader reader;
    reader.open(b);
    rclcpp::Serialization<sensor_msgs::msg::Image> img_ser;
    rclcpp::Serialization<sensor_msgs::msg::CameraInfo> info_ser;
    rclcpp::Serialization<nav_msgs::msg::Odometry> odom_ser;
    depth_stats::Intr K{0, 0, 0, 0, 0, 0, cam_h, cam_pitch};
    bool have_k = false, have_odom = false, still = false;
    double still_since = 0;
    long n = 0, used = 0, bad = 0;
    std::vector<sensor_msgs::msg::Image> early;   // 내부값보다 먼저 온 영상
    auto use = [&](const sensor_msgs::msg::Image & im) {
        if (im.encoding != "16UC1" || static_cast<int>(im.width) != K.w || static_cast<int>(im.height) != K.h || im.step != im.width * 2) {
          ++bad;
          return;
        }
        if (n++ % every) {return;}
        const auto * p = reinterpret_cast<const uint16_t *>(im.data.data());
        if (!dump.empty() && used == 0) {
          // 첫 장을 PGM 으로 (0 = 검정 구멍, 0.3~3.5 m → 밝음~어두움) — 구멍이 어디 몰렸는지 사람이 보려고. 원값(mm, u16)은 <dump>.raw
          FILE * pg = std::fopen(dump.c_str(), "wb");
          std::fprintf(pg, "P5\n%d %d\n255\n", K.w, K.h);
          for (int i = 0; i < K.w * K.h; ++i) {std::fputc(p[i] ? 255 - std::min(250, std::max(0, (p[i] - 300) * 250 / 3200)) : 0, pg);}
          FILE * rw = std::fopen((dump + ".raw").c_str(), "wb");
          std::fwrite(p, 2, K.w * K.h, rw);
          std::fclose(rw);
          std::fclose(pg);
        }
        depth_stats::add_frame(acc, p, K);
        const double t = im.header.stamp.sec + im.header.stamp.nanosec * 1e-9;
        if (is_static || (still && t - still_since > 0.5)) {
          tmp.add(p, K.w * K.h);
        } else {
          tmp.flush(kMinSegFrames);
        }
        ++used;
      };
    while (reader.has_next()) {
      const auto m = reader.read_next();
      rclcpp::SerializedMessage raw(*m->serialized_data);
      if (m->topic_name == "/camera/depth/camera_info" && !have_k) {
        sensor_msgs::msg::CameraInfo ci;
        info_ser.deserialize_message(&raw, &ci);
        K = {static_cast<float>(ci.k[0]), static_cast<float>(ci.k[4]), static_cast<float>(ci.k[2]), static_cast<float>(ci.k[5]),
          static_cast<int>(ci.width), static_cast<int>(ci.height), cam_h, cam_pitch};
        have_k = true;
        for (const auto & im : early) {use(im);}
        early.clear();
      } else if (m->topic_name == "/odom") {
        nav_msgs::msg::Odometry od;
        odom_ser.deserialize_message(&raw, &od);
        const bool s = od.twist.twist.linear.x == 0 && od.twist.twist.angular.z == 0;
        if (s && !still) {still_since = od.header.stamp.sec + od.header.stamp.nanosec * 1e-9;}
        still = s;
        have_odom = true;
      } else if (m->topic_name == topic) {
        sensor_msgs::msg::Image im;
        img_ser.deserialize_message(&raw, &im);
        if (have_k) {use(im);} else if (early.size() < 60) {early.push_back(std::move(im));}
      }
    }
    tmp.flush(kMinSegFrames);
    if (!have_k) {std::fprintf(stderr, "%s: /camera/depth/camera_info 없음\n", b.c_str()); return 1;}
    if (!is_static && !have_odom) {std::fprintf(stderr, "%s: /odom 없음 — 시간 σ 는 --static 일 때만\n", b.c_str());}
    char buf[256];
    std::snprintf(buf, sizeof(buf), "%s (깊이 %ld 장 중 %ld 장, fx %.1f %d×%d%s)", b.c_str(), n, used, K.fx, K.w, K.h,
      bad ? ", 형식 다른 장 있음" : "");
    std::printf("%s\n", buf);
    source += (source.empty() ? "" : "; ") + std::string(buf);
  }
  FILE * f = std::fopen(out.c_str(), "w");
  if (!f) {std::perror(out.c_str()); return 1;}
  depth_stats::write_yaml(f, acc, tmp.segments ? &tmp : nullptr, source);
  std::fclose(f);
  return 0;
}
