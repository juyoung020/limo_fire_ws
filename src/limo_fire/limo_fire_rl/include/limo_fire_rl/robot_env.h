// 실물 쪽 탐사 RL 환경 (ROS 없음): 실물 센서 값(라이다 장·지도 자세·바퀴·IMU·명령·깊이)을 학습 시뮬과 같은 함수로 쌓아 계층형 관측을 만든다.
// 시뮬 한 걸음(0.1 s, env_ref.cpp step_env)의 차례를 그대로 따르고, 센서를 흉내 내는 부분(물리·광선)만 실물 값으로 바꾼다:
//   path_last_act(/cmd_vel) → 바퀴·자이로·IMU 값 → scan_begin_map(지도 자세) → 빔마다 scan_beam_map(세기 = /scan intensities) → scan_apply → known_box
//   → map_progress → finish_step(보상은 버림, 닫힘 성공·진행 없음 판정은 그대로) → ring_advance → push_bins → write_obs(깊이 빼고 — 깊이는 set_depth)
// 지도·관측·후보·닫힘은 모두 sim/include(env.h · obs_ref.h)·mdp/ 의 함수 그대로라 같은 입력이면 시뮬과 비트까지 같다
// (train/explore/sim/tools/rl_node_check 가 CPU 시뮬 걸음과 비교). 정답 격자는 없다(MapDev::grid 는 0 — 덮음·벽 칸은 실물에서 0).
// 빌드할 때 -ffp-contract=off (시뮬 CPU 참조판과 같은 부동소수점 — aarch64 g++ 기본은 곱셈·덧셈을 합쳐 값이 달라짐).
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "env_cfg.h"   // train/explore/sim/include
#include "obs_ref.h"

namespace limo_fire_rl {

// 라이다 한 장 (sensor_msgs/LaserScan 꼴) + 그 장 첫 빔 시각의 지도 자세 (base_link, Cartographer map 틀)
struct Scan {
  std::vector<float> ranges;       // m (0·inf·nan = 빠짐)
  std::vector<float> intensities;  // 없으면 비움 (세기 0)
  float angle_min = 0.f, angle_increment = 0.f, time_increment = 0.f;
  float x = 0.f, y = 0.f, th = 0.f;   // 지도 자세 (base_link)
  float w = 0.f;                      // 몸통 각속도 (rad/s) — 빔 시각만큼 펴기 (시뮬 sh.mw)
  float n_rev = 0.f;                  // 시험용: > 0 이면 시뮬 장의 한 바퀴 점 수·한 바퀴 시간을 그대로 (실물은 0 — angle_increment·time_increment 로)
  float scan_time = 0.f;
};

// 그 걸음의 고유 감각 (시뮬 EnvState 의 같은 칸)
struct Proprio {
  float wheel_v = 0.f, wheel_vy = 0.f, wheel_w = 0.f;   // 바퀴 오도메트리 (/odom twist) — e.drive.wv·wvy·ww
  float gyro_z = 0.f;                                   // /imu 각속도 z — e.gyro
  float imu[7] = {0, 0, 0, 0, 0, 0, 0};                 // ax, ay, az, gx, gy, roll, pitch — e.imu.out (REP-103, az 는 중력 포함)
  float cmd_v = 0.f, cmd_w = 0.f;                       // 이번 걸음 명령 (/cmd_vel — Nav2 velocity_smoother 뒤)
};

class RobotEnv {
 public:
  explicit RobotEnv(const sim::EnvHostCfg& hc);
  RobotEnv(const RobotEnv&) = delete;
  RobotEnv& operator=(const RobotEnv&) = delete;

  // 지도 창 (시뮬 MapDev 의 ox, oy, gw, gh). 실물은 시작점 가운데 size × size 칸
  struct Window {
    float ox, oy;
    int gw, gh;
  };
  // 새 판: 시작 지도 자세·그때 감각(바퀴·IMU — 리셋 관측의 숫자 칸)·라이다 시야(rad, −2·angle_min — 닫힘 규칙의 본 칸).
  // 리셋 관측(장 없음)까지 만든다. win = nullptr 이면 시작점 가운데 창
  void reset(float x0, float y0, float th0, const Proprio& p0, float fov, const Window* win = nullptr);
  void refresh_obs() { obs(); }   // 상태를 밖에서 바꾼 뒤 (시험·목표 버림) 관측 다시
  // 0.1 s 걸음 하나 (라이다 장 하나). 끝에 관측을 다시 쓴다 (깊이 칸은 set_depth 가 넣은 값 그대로)
  void step(const Proprio& p, const Scan& s);
  // 결정 걸음인가 (시뮬: decide_steps 걸음마다 정책이 고름 — 판 시작 걸음 0 포함)
  bool decision_due() const { return e_.step % b_.cfg.decide_steps == 0; }
  // 관측 깊이 칸 (mdp::obs::DEPTH_SIZE 바이트, observations.h depth_u8 값)
  void set_depth(const uint8_t* d);
  // 깊이 → 관측 칸: 실물 깊이 영상(16UC1 mm, 내부 파라미터 fx fy cx cy, 크기 w·h) → 40 × 64. 시뮬 관측 픽셀 (u, v)(depth_uv) 의 광선 방향을
  // 실물 카메라 내부 파라미터로 다시 투영한 가장 가까운 픽셀의 z (정렬 켬이면 같은 내부 파라미터라 그 픽셀 그대로) → valid_rect 밖 0 → depth_u8
  void depth_from_image(const uint16_t* mm, int w, int h, int step_px, float fx, float fy, float cx, float cy, uint8_t* out) const;
  // 결정 뒤: 깊이 카메라 장애물 층 갱신 (시뮬: 다음 결정 묶음 앞에 hier_camlayer_cpu — 그 결정 관측의 깊이로)
  void camlayer_update();
  // 고른 슬롯 → 목표 칸. 같은 덩어리를 다시 고르면(cand_cur 비트, 시뮬 plan_cpu "같은 후보") 목표 그대로 = false, 새 목표면 true.
  // 슬롯이 빈 칸이면 목표 없음(goal_cell −1, 시뮬은 제자리 회전) = true
  bool choose(int slot);
  void clear_goal() { e_.goal_cell = -1; }   // Nav2 실패 등으로 목표를 버림 (다음 결정에 새로)

  // 읽기
  const uint8_t* obs_img() const { return b_.obs_img; }
  const float* obs_1d() const { return b_.obs_1d; }
  int cand_cell(int k) const { return b_.cand[k]; }
  bool cand_xy(int k, float* x, float* y) const;   // 후보 칸 가운데 (지도 틀)
  bool cell_xy(int cell, float* x, float* y) const;
  int cand_cur() const { return b_.cand_cur[0]; }
  int goal_cell() const { return e_.goal_cell; }
  int done() const { return done_; }               // 지난 걸음 mdp::cur::Done (0 진행 · 1 닫힘 성공 · 5 진행 없음 · 3 끊김 …)
  bool closed() const { return e_.map_closed != 0; }
  float idle_s() const { return (float)e_.since_occ * dt_act(); }
  int steps() const { return e_.step; }
  float dt_act() const { return (float)b_.cfg.substeps * b_.cfg.dt; }
  const sim::EnvCfg& cfg() const { return b_.cfg; }
  const sim::EnvState& state() const { return e_; }
  const sim::MapDev& map() const { return m_; }
  const int16_t* lmap() const { return b_.lmap; }
  const uint8_t* cam_layer() const { return b_.cvox; }

  // 시험용 (rl_node_check): 시뮬과 같은 판 시작 상태를 넣음 — reset 뒤에 부름
  sim::EnvState& state_mut() { return e_; }
  const sim::EnvBatch& batch() const { return b_; }

 private:
  void obs();
  sim::EnvBatch b_{};
  sim::MapDev m_{};
  sim::EnvState e_{};
  sim::Shared sh_{};
  int done_ = 0;
  // 메모리 (EnvBatch 포인터가 가리킴)
  std::vector<uint8_t> grid_;
  std::vector<int16_t> lmap_;
  std::vector<uint32_t> marks_, trail_;
  std::vector<uint8_t> img_, cvox_;
  std::vector<float> o1d_, priv_, ring_, vprev_, lint_, rew_, terms_, cpts_;
  std::vector<int32_t> done_buf_, cand_, cand_cur_;
  std::vector<sim::EpInfo> ep_;
};

}  // namespace limo_fire_rl
