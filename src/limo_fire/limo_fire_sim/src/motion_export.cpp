// 실물 녹화 → 구동 응답 표(CSV): 바퀴 오도메트리(/odom, 50 Hz) 한 줄마다 그때의 명령(리모컨 스틱 또는 /cmd_vel)·IMU 자이로·조종 모드.
// 학습 시뮬 motion_check 가 같은 명령을 시뮬 구동 모델에 넣어 실물 속도와 비교·맞춘다 (시뮬 = 현실 확인).
//   motion_export <녹화 폴더> <출력.csv>
// 열: t (s, 첫 줄 0), mode (섀시 조종 모드: 3 리모컨 · 1 ROS 명령 · 0 대기), rc_v (왼 스틱 세로 −1~1), rc_w (오른 스틱 가로, 부호를
//     ω 와 같게 뒤집음), cmd_v, cmd_w (/cmd_vel), odom_v, odom_w (바퀴), gyro_w (IMU z), stamp (절대 시각 s, 다른 표와 맞출 때),
//     acc_x (IMU 앞쪽 가속도 m/s², 정지 때 평균을 뺀 값 — 기울기·바이어스 몫)
//   IMU 원값 (imu_stats.hpp 가 잡음·진동 통계를 낸다 — 보간하면 잡음이 줄어서 그 시각 직전 메시지 그대로):
//     ax, ay, az (m/s², imu_link = base_link 방향), gx, gy (rad/s), roll, pitch (rad, 쿼터니언의 ZYX 오일러각 그대로).
//     드라이버 소스는 바이트 필드 이름이 roll·pitch 뒤바뀌어 있고 setRPY(pitch, roll, yaw) 로 다시 뒤바꿔 넣어, 나오는 쿼터니언은 REP-103 대로다
//     (실측 8 녹화 정지 평균: roll = atan2(ay, az), pitch = −atan2(ax, az) 와 1 mrad 안, 자이로 x 적분은 roll 변화와 상관 —
//     train/explore/sim/analysis/real_limo.md IMU 절). imu_n (지금까지 받은 /imu 수 — 같은 메시지가 두 줄에 걸쳤는지)
//   지도 자세 (Cartographer map → odom ∘ EKF odom → base_link, 그 시각 보간): map_x, map_y, map_yaw,
//     mo_x, mo_y, mo_yaw (map → odom 그 시각 직전 값 — 스캔 매칭 보정이 계단으로 보인다), mo_n (지금까지 받은 map → odom 수). 없으면 nan
//   메카넘 옆·모터: cmd_vy (/cmd_vel linear.y, 직전 값), odom_vy (바퀴로 계산한 옆 속도), m1~m4 (/motor_states 모터 속도, 직전 값)
//   라이다 몸 자세 (scan_match.hpp 스캔 맞추기, 녹화 처음 기준 base_link — 바퀴와 따로 잰 몸 이동 참값): lo_x, lo_y, lo_yaw (그 시각 보간, 없으면 nan)
// 명령·자이로는 그 시각 직전 값(명령) · 선형 보간(자이로)
#include <algorithm>
#include <map>
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
#include <sensor_msgs/msg/joint_state.hpp>
#include <sensor_msgs/msg/joy.hpp>
#include <sensor_msgs/msg/laser_scan.hpp>
#include <tf2_msgs/msg/tf_message.hpp>

#include "limo_fire_sim/scan_match.hpp"

namespace
{
double stamp(const builtin_interfaces::msg::Time & t) {return t.sec + t.nanosec * 1e-9;}

struct S { double t; double a, b; };
// IMU 원값 한 메시지 (REP-103 roll·pitch)
struct Imu { double t; double ax, ay, az, gx, gy, roll, pitch; };
// 평면 자세
struct P { double t, x, y, yaw; };

// 시각 순 목록에서 t 의 자세 (선형, 각은 짧은 쪽). 범위 밖이면 false
bool interp(const std::vector<P> & v, double t, P & out)
{
  if (v.empty() || t < v.front().t || t > v.back().t) {return false;}
  auto it = std::upper_bound(v.begin(), v.end(), t, [](double x, const P & p) {return x < p.t;});
  if (it == v.end()) {out = v.back(); return true;}
  const P & a = *(it - 1), & b = *it;
  const double f = b.t > a.t ? (t - a.t) / (b.t - a.t) : 0;
  out = {t, a.x + f * (b.x - a.x), a.y + f * (b.y - a.y), a.yaw + f * std::remainder(b.yaw - a.yaw, 2 * M_PI)};
  return true;
}

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
  rclcpp::Serialization<tf2_msgs::msg::TFMessage> tf_ser;
  std::vector<S> odom, gyro, rc, cmd, mode;
  std::vector<Imu> imu;
  std::vector<P> map_odom, odom_base;
  std::vector<S> odom_vy, cmd_vy;   // 메카넘 옆 (바퀴로 계산한 vy · 명령 vy)
  struct Motor { double t; double w[4]; };
  std::vector<Motor> motors;
  struct Scan { double t; std::string frame; sensor_msgs::msg::LaserScan s; };
  std::vector<Scan> scans;
  struct Static { std::string parent; scan_match::Pose p; };
  std::map<std::string, Static> statics;   // 자식 → (부모, 평면 자세) — /tf_static
  rclcpp::Serialization<sensor_msgs::msg::LaserScan> scan_ser;
  rclcpp::Serialization<sensor_msgs::msg::JointState> joint_ser;
  while (reader.has_next()) {
    const auto m = reader.read_next();
    rclcpp::SerializedMessage raw(*m->serialized_data);
    const double tb = m->time_stamp * 1e-9;   // 받은 시각 (/cmd_vel 은 stamp 가 없다)
    if (m->topic_name == "/odom") {
      nav_msgs::msg::Odometry o;
      odom_ser.deserialize_message(&raw, &o);
      odom.push_back({stamp(o.header.stamp), o.twist.twist.linear.x, o.twist.twist.angular.z});
      odom_vy.push_back({stamp(o.header.stamp), o.twist.twist.linear.y, 0});
    } else if (m->topic_name == "/imu") {
      sensor_msgs::msg::Imu i;
      imu_ser.deserialize_message(&raw, &i);
      gyro.push_back({stamp(i.header.stamp), i.angular_velocity.z, i.linear_acceleration.x});
      // 쿼터니언 → 오일러 (ZYX). REP-103 대로 나온다 (위 머리말)
      const auto & q = i.orientation;
      const double q_roll = std::atan2(2 * (q.w * q.x + q.y * q.z), 1 - 2 * (q.x * q.x + q.y * q.y));
      const double q_pitch = std::asin(std::max(-1.0, std::min(1.0, 2 * (q.w * q.y - q.z * q.x))));
      imu.push_back({stamp(i.header.stamp), i.linear_acceleration.x, i.linear_acceleration.y, i.linear_acceleration.z,
        i.angular_velocity.x, i.angular_velocity.y, q_roll, q_pitch});
    } else if (m->topic_name == "/rc_state") {
      sensor_msgs::msg::Joy j;
      joy_ser.deserialize_message(&raw, &j);
      if (j.axes.size() >= 4) {rc.push_back({stamp(j.header.stamp), j.axes[2], -j.axes[0]});}
    } else if (m->topic_name == "/cmd_vel") {
      geometry_msgs::msg::Twist c;
      twist_ser.deserialize_message(&raw, &c);
      cmd.push_back({tb, c.linear.x, c.angular.z});
      cmd_vy.push_back({tb, c.linear.y, 0});
    } else if (m->topic_name == "/limo_status") {
      limo_msgs::msg::LimoStatus s;
      status_ser.deserialize_message(&raw, &s);
      mode.push_back({tb, double(s.control_mode), 0});   // stamp 은 원본 드라이버 버그로 못 씀 → 받은 시각
    } else if (m->topic_name == "/scan") {
      Scan sc;
      scan_ser.deserialize_message(&raw, &sc.s);
      sc.t = stamp(sc.s.header.stamp);
      sc.frame = sc.s.header.frame_id;
      scans.push_back(std::move(sc));
    } else if (m->topic_name == "/motor_states") {
      sensor_msgs::msg::JointState j;
      joint_ser.deserialize_message(&raw, &j);
      if (j.velocity.size() >= 4) {motors.push_back({stamp(j.header.stamp), {j.velocity[0], j.velocity[1], j.velocity[2], j.velocity[3]}});}
    } else if (m->topic_name == "/tf_static") {
      tf2_msgs::msg::TFMessage tf;
      tf_ser.deserialize_message(&raw, &tf);
      for (const auto & t : tf.transforms) {
        const auto & q = t.transform.rotation;
        statics[t.child_frame_id] = {t.header.frame_id, {t.transform.translation.x, t.transform.translation.y,
            std::atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))}};
      }
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
  if (odom.empty()) {
    std::fprintf(stderr, "/odom 이 녹화에 없음\n");
    return 1;
  }
  for (auto * v : {&odom, &gyro, &rc, &cmd, &mode, &odom_vy, &cmd_vy}) {
    std::sort(v->begin(), v->end(), [](const S & a, const S & b) {return a.t < b.t;});
  }
  std::sort(imu.begin(), imu.end(), [](const Imu & a, const Imu & b) {return a.t < b.t;});
  for (auto * v : {&map_odom, &odom_base}) {
    std::sort(v->begin(), v->end(), [](const P & a, const P & b) {return a.t < b.t;});
  }
  std::sort(motors.begin(), motors.end(), [](const Motor & a, const Motor & b) {return a.t < b.t;});
  std::sort(scans.begin(), scans.end(), [](const Scan & a, const Scan & b) {return a.t < b.t;});
  // 라이다 오도메트리 (scan_match.hpp): 몸이 실제로 움직인 자세 lo_* — 바퀴(/odom)와 따로. 빔마다 시각(time_increment)만큼
  // 몸 움직임(바퀴 v·vy, 자이로 ω)을 되돌려 한 시각으로 모은 뒤, 스캔 프레임 → base_link 는 /tf_static.
  // 빔 시각은 **마지막 빔이 먼저** (라이다가 시계 방향으로 돌고 드라이버가 각도 순으로 담음): 1103_M_cmd1 제자리 ±1·±2 rad/s 회전 네 번을
  // 자이로 적분과 비교하면 이 순서로 0.97~1.01, 빔 순서 = 시각이면 0.93~1.08 (방향마다 반대로 틀림), 안 되돌리면 0.97~1.05
  std::vector<P> lo;
  size_t lo_fail = 0;
  {
    auto to_base = [&](const std::string & frame, scan_match::Pose & out) {
        out = {0, 0, 0};
        std::string cur = frame;
        for (int k = 0; k < 8 && cur != "base_link"; ++k) {
          const auto it = statics.find(cur);
          if (it == statics.end()) {return false;}
          out = scan_match::compose(it->second.p, out);
          cur = it->second.parent;
        }
        return cur == "base_link";
      };
    scan_match::Odom od;
    double unwrapped = 0, last_th = 0;
    for (size_t k = 0; k < scans.size(); ++k) {
      const auto & s = scans[k].s;
      scan_match::Pose mount;
      if (!to_base(scans[k].frame, mount)) {
        std::fprintf(stderr, "/tf_static 에 base_link → %s 없음 — 라이다 몸 자세 안 냄\n", scans[k].frame.c_str());
        lo.clear();
        break;
      }
      const double t = scans[k].t, vx = lerp(odom, t), vy = lerp(odom_vy, t), w = lerp(gyro, t);
      std::vector<scan_match::Pt> pts;
      for (size_t i = 0; i < s.ranges.size(); ++i) {
        const double r = s.ranges[i];
        if (!std::isfinite(r) || r < std::max(0.12, double(s.range_min)) || r > std::min(8.0, double(s.range_max))) {continue;}
        const double a = s.angle_min + i * s.angle_increment, ti = double(s.ranges.size() - 1 - i) * s.time_increment;
        const scan_match::Pt pb = scan_match::apply(mount, {r * std::cos(a), r * std::sin(a)});
        pts.push_back(scan_match::apply({vx * ti, vy * ti, w * ti}, pb));   // 장 시작 시각 몸 좌표로
      }
      // 지난 장에서 이번 장까지 바퀴 v·vy · 자이로로 짐작
      scan_match::Pose g{0, 0, 0};
      if (k > 0) {
        const double t_prev = scans[k - 1].t;
        const int n = 10;
        for (int j = 0; j < n; ++j) {
          const double tm = t_prev + (j + 0.5) * (t - t_prev) / n, dt = (t - t_prev) / n;
          g = scan_match::compose(g, {lerp(odom, tm) * dt, lerp(odom_vy, tm) * dt, lerp(gyro, tm) * dt});
        }
      }
      if (!od.add(pts, g)) {++lo_fail;}
      unwrapped += std::remainder(od.pose.th - last_th, 2 * M_PI);
      last_th = od.pose.th;
      lo.push_back({t, od.pose.x, od.pose.y, unwrapped});
    }
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
  std::fprintf(f, "t,mode,rc_v,rc_w,cmd_v,cmd_w,odom_v,odom_w,gyro_w,stamp,acc_x,ax,ay,az,gx,gy,roll,pitch,imu_n,map_x,map_y,map_yaw,mo_x,mo_y,mo_yaw,mo_n,cmd_vy,odom_vy,m1,m2,m3,m4,lo_x,lo_y,lo_yaw\n");
  const double t0 = odom.front().t;
  for (const S & o : odom) {
    const S * r = before(rc, o.t), * c = before(cmd, o.t), * md = before(mode, o.t);
    std::fprintf(f, "%.4f,%d,%.3f,%.3f,%.3f,%.3f,%.4f,%.4f,%.5f,%.4f,%.4f", o.t - t0, md ? int(md->a) : -1, r ? r->a : 0.0,
      r ? r->b : 0.0, c ? c->a : 0.0, c ? c->b : 0.0, o.a, o.b, lerp(gyro, o.t), o.t, lerp(gyro, o.t, true) - acc0);
    // IMU 그 시각 직전 메시지, 지도 자세
    const auto ii = std::upper_bound(imu.begin(), imu.end(), o.t, [](double x, const Imu & s) {return x < s.t;}) - imu.begin();
    const Imu im = ii > 0 ? imu[ii - 1] : Imu{0, NAN, NAN, NAN, NAN, NAN, NAN, NAN};
    std::fprintf(f, ",%.3f,%.3f,%.3f,%.5f,%.5f,%.5f,%.5f,%ld", im.ax, im.ay, im.az, im.gx, im.gy, im.roll, im.pitch, long(ii));
    P mo, ob;
    const auto mi = std::upper_bound(map_odom.begin(), map_odom.end(), o.t, [](double x, const P & p) {return x < p.t;}) - map_odom.begin();
    if (mi > 0 && interp(odom_base, o.t, ob) && interp(map_odom, o.t, mo)) {
      const double c = std::cos(mo.yaw), sn = std::sin(mo.yaw);
      const P & last = map_odom[mi - 1];
      std::fprintf(f, ",%.5f,%.5f,%.6f,%.5f,%.5f,%.6f,%ld", mo.x + c * ob.x - sn * ob.y, mo.y + sn * ob.x + c * ob.y,
        std::remainder(mo.yaw + ob.yaw, 2 * M_PI), last.x, last.y, last.yaw, long(mi));
    } else {
      std::fprintf(f, ",nan,nan,nan,nan,nan,nan,%ld", long(mi));
    }
    // 메카넘 옆·모터·라이다 몸 자세
    const S * cy = before(cmd_vy, o.t);
    const auto mt = std::upper_bound(motors.begin(), motors.end(), o.t, [](double x, const Motor & s) {return x < s.t;}) - motors.begin();
    const Motor mm = mt > 0 ? motors[mt - 1] : Motor{0, {NAN, NAN, NAN, NAN}};
    std::fprintf(f, ",%.3f,%.4f,%.3f,%.3f,%.3f,%.3f", cy ? cy->a : 0.0, lerp(odom_vy, o.t), mm.w[0], mm.w[1], mm.w[2], mm.w[3]);
    P lp;
    if (interp(lo, o.t, lp)) {std::fprintf(f, ",%.5f,%.5f,%.6f\n", lp.x, lp.y, lp.yaw);} else {std::fprintf(f, ",nan,nan,nan\n");}
  }
  std::fclose(f);
  std::printf("%s: 오도메트리 %zu 줄 (%.1f s), 리모컨 %zu, /cmd_vel %zu, 자이로 %zu, 라이다 %zu 장 (맞추기 실패 %zu) → %s\n", argv[1], odom.size(),
    odom.back().t - t0, rc.size(), cmd.size(), gyro.size(), lo.size(), lo_fail, argv[2]);
  return 0;
}
