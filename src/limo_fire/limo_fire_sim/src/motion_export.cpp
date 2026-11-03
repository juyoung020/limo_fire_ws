// 실물 녹화 → 구동 응답 표(CSV): 바퀴 오도메트리(/odom, 50 Hz) 한 줄마다 그때의 명령(리모컨 스틱 또는 /cmd_vel)·IMU 자이로·조종 모드.
// 학습 시뮬 motion_check 가 같은 명령을 시뮬 구동 모델에 넣어 실물 속도와 비교·맞춘다 (시뮬 = 현실 확인).
//   motion_export <녹화 폴더> <출력.csv>
// 열: t (s, 첫 줄 0), mode (바닥판 조종 모드: 3 리모컨 · 1 ROS 명령 · 0 대기), rc_v (왼 스틱 세로 −1~1), rc_w (오른 스틱 가로, 부호를
//     ω 와 같게 뒤집음), cmd_v, cmd_w (/cmd_vel), odom_v, odom_w (바퀴), gyro_w (IMU z), stamp (절대 시각 s, 다른 표와 맞출 때),
//     acc_x (IMU 앞쪽 가속도 m/s², 정지 때 평균을 뺀 값 — 기울기·바이어스 몫)
// 명령·자이로는 그 시각 직전 값(명령) · 선형 보간(자이로)
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include <geometry_msgs/msg/twist.hpp>
#include <limo_msgs/msg/limo_status.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <rclcpp/serialization.hpp>
#include <rosbag2_cpp/reader.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <sensor_msgs/msg/joy.hpp>

namespace
{
double stamp(const builtin_interfaces::msg::Time & t) {return t.sec + t.nanosec * 1e-9;}

struct S { double t; double a, b; };

// t 직전 값 (없으면 0)
const S * before(const std::vector<S> & v, double t)
{
  auto it = std::upper_bound(v.begin(), v.end(), t, [](double x, const S & s) {return x < s.t;});
  return it == v.begin() ? nullptr : &*(it - 1);
}
double lerp(const std::vector<S> & v, double t, bool second = false)
{
  auto it = std::upper_bound(v.begin(), v.end(), t, [](double x, const S & s) {return x < s.t;});
  auto val = [&](const S & s) {return second ? s.b : s.a;};
  if (it == v.begin()) {return v.empty() ? 0 : val(v.front());}
  if (it == v.end()) {return val(v.back());}
  const S & p = *(it - 1), & q = *it;
  return val(p) + (val(q) - val(p)) * (t - p.t) / (q.t - p.t);
}
}  // namespace

int main(int argc, char ** argv)
{
  if (argc < 3) {
    std::fprintf(stderr, "motion_export <녹화 폴더> <출력.csv>\n");
    return 2;
  }
  rosbag2_cpp::Reader reader;
  reader.open(argv[1]);
  rclcpp::Serialization<nav_msgs::msg::Odometry> odom_ser;
  rclcpp::Serialization<sensor_msgs::msg::Imu> imu_ser;
  rclcpp::Serialization<sensor_msgs::msg::Joy> joy_ser;
  rclcpp::Serialization<geometry_msgs::msg::Twist> twist_ser;
  rclcpp::Serialization<limo_msgs::msg::LimoStatus> status_ser;
  std::vector<S> odom, gyro, rc, cmd, mode;
  while (reader.has_next()) {
    const auto m = reader.read_next();
    rclcpp::SerializedMessage raw(*m->serialized_data);
    const double tb = m->time_stamp * 1e-9;   // 받은 시각 (/cmd_vel 은 stamp 가 없다)
    if (m->topic_name == "/odom") {
      nav_msgs::msg::Odometry o;
      odom_ser.deserialize_message(&raw, &o);
      odom.push_back({stamp(o.header.stamp), o.twist.twist.linear.x, o.twist.twist.angular.z});
    } else if (m->topic_name == "/imu") {
      sensor_msgs::msg::Imu i;
      imu_ser.deserialize_message(&raw, &i);
      gyro.push_back({stamp(i.header.stamp), i.angular_velocity.z, i.linear_acceleration.x});
    } else if (m->topic_name == "/rc_state") {
      sensor_msgs::msg::Joy j;
      joy_ser.deserialize_message(&raw, &j);
      if (j.axes.size() >= 4) {rc.push_back({stamp(j.header.stamp), j.axes[2], -j.axes[0]});}
    } else if (m->topic_name == "/cmd_vel") {
      geometry_msgs::msg::Twist c;
      twist_ser.deserialize_message(&raw, &c);
      cmd.push_back({tb, c.linear.x, c.angular.z});
    } else if (m->topic_name == "/limo_status") {
      limo_msgs::msg::LimoStatus s;
      status_ser.deserialize_message(&raw, &s);
      mode.push_back({tb, double(s.control_mode), 0});   // stamp 은 원본 드라이버 버그로 못 씀 → 받은 시각
    }
  }
  if (odom.empty()) {
    std::fprintf(stderr, "/odom 이 녹화에 없음\n");
    return 1;
  }
  for (auto * v : {&odom, &gyro, &rc, &cmd, &mode}) {
    std::sort(v->begin(), v->end(), [](const S & a, const S & b) {return a.t < b.t;});
  }
  FILE * f = std::fopen(argv[2], "w");
  if (!f) {std::perror(argv[2]); return 1;}
  // 가속도 x 의 정지 평균 (바퀴가 멈춘 줄): 기울기로 섞인 중력·바이어스
  double acc0 = 0;
  size_t n0 = 0;
  for (const S & o : odom) {
    if (std::fabs(o.a) < 1e-3 && std::fabs(o.b) < 1e-3) {acc0 += lerp(gyro, o.t, true); ++n0;}
  }
  acc0 = n0 ? acc0 / n0 : 0;
  std::fprintf(f, "t,mode,rc_v,rc_w,cmd_v,cmd_w,odom_v,odom_w,gyro_w,stamp,acc_x\n");
  const double t0 = odom.front().t;
  for (const S & o : odom) {
    const S * r = before(rc, o.t), * c = before(cmd, o.t), * md = before(mode, o.t);
    std::fprintf(f, "%.4f,%d,%.3f,%.3f,%.3f,%.3f,%.4f,%.4f,%.5f,%.4f,%.4f\n", o.t - t0, md ? int(md->a) : -1, r ? r->a : 0.0,
      r ? r->b : 0.0, c ? c->a : 0.0, c ? c->b : 0.0, o.a, o.b, lerp(gyro, o.t), o.t, lerp(gyro, o.t, true) - acc0);
  }
  std::fclose(f);
  std::printf("%s: 오도메트리 %zu 줄 (%.1f s), 리모컨 %zu, /cmd_vel %zu, 자이로 %zu → %s\n", argv[1], odom.size(),
    odom.back().t - t0, rc.size(), cmd.size(), gyro.size(), argv[2]);
  return 0;
}
