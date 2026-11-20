// 실물 쪽 탐사 환경 (ROS 없음) — 실물 합친 탐사 노드(explorer_node.cpp)가 쓰는 시뮬 한 환경 몫.
// 실물 센서 값(라이다 장·지도 자세·바퀴·IMU·깊이)을 시뮬과 같은 함수로 쌓아 지도·후보·닫힘을 만들고, 아래층 규칙 추종기(A* 경로 + 추종)로 명령을 낸다.
// 시뮬 한 걸음(0.1 s, env_ref.cpp step_env + low_prepare_env)의 차례를 그대로 따르고, 센서를 흉내 내는 부분(물리·광선)만 실물 값으로 바꾼다:
//   low_act(규칙 추종기 → 명령) → step(감각, 라이다 장: scan_begin_map → 빔마다 scan_beam_map → scan_apply → known_box → map_progress → sense_step
//   → finish_step(닫힘 성공·진행 없음 판정) → ring_advance → push_bins → write_obs(프론티어 요약·후보, 깊이 빼고)) → set_depth(깊이) → [set_cmd(위층 목표)] → low_prepare(계획)
// 지도·관측·후보·닫힘·경로는 모두 sim/include(env.h · obs_ref.h · plan_ref.h)·mdp/ 의 함수 그대로라 같은 입력이면 시뮬과 비트까지 같다
// (train/explore/sim/tools/low_explore_check 가 CPU 시뮬 걸음과 비교). 정답 격자는 없다(MapDev::grid 는 0 — 덮음·벽 칸은 실물에서 0).
// 빌드할 때 -ffp-contract=off (시뮬 CPU 참조판과 같은 부동소수점 — aarch64 g++ 기본은 곱셈·덧셈을 합쳐 값이 달라짐).
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "env_cfg.h"   // train/explore/sim/include
#include "obs_ref.h"
#include "plan_ref.h"   // 경로 계획 CPU 판 (합친 탐사 모드 — 시뮬 env_ref.cpp 와 같은 함수)

namespace limo_fire_explore {

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
  // 합친 탐사 (combine_run 과 같은 설정: 아래층 걸음 길 · low_explore 1 — 판 끝은 탐사 규칙, 정체는 막힘 신호만). 닫힘을 라이다 지도만으로 하려면 hc.cfg.low_explore = 2
  explicit RobotEnv(const sim::EnvHostCfg& hc);
  RobotEnv(const RobotEnv&) = delete;
  RobotEnv& operator=(const RobotEnv&) = delete;

  // 지도 창 (시뮬 MapDev 의 ox, oy, gw, gh). 실물은 시작점 가운데 size × size 칸
  struct Window {
    float ox, oy;
    int gw, gh;
  };
  // 새 판: 시작 지도 자세·그때 감각(바퀴·IMU — 리셋 관측의 숫자 칸)·라이다 시야(rad, −2·angle_min — 닫힘 규칙의 본 칸).
  // 리셋 관측(장 없음 — 프론티어 요약·후보)까지 만든다. win = nullptr 이면 시작점 가운데 창
  void reset(float x0, float y0, float th0, const Proprio& p0, float fov, const Window* win = nullptr);
  // 0.1 s 걸음 하나 (라이다 장 하나). 끝에 프론티어 요약·후보를 다시 낸다 (깊이 칸은 set_depth 가 넣은 값 그대로)
  void step(const Proprio& p, const Scan& s);
  // 관측 깊이 칸 (mdp::obs::DEPTH_SIZE 바이트, observations.h depth_u8 값)
  void set_depth(const uint8_t* d);
  // 깊이 → 관측 칸: 실물 깊이 영상(16UC1 mm, 내부 파라미터 fx fy cx cy, 크기 w·h) → 40 × 64. 시뮬 관측 픽셀 (u, v)(depth_uv) 의 광선 방향을
  // 실물 카메라 내부 파라미터로 다시 투영한 가장 가까운 픽셀의 z (정렬 켬이면 같은 내부 파라미터라 그 픽셀 그대로) → valid_rect 밖 0 → depth_u8
  void depth_from_image(const uint16_t* mm, int w, int h, int step_px, float fx, float fy, float cx, float cy, uint8_t* out) const;
  // 깊이 카메라 층 전용 격자 (cfg.cam_lw × cam_lh, env.yaml camera_obstacles.layer_grid — 0 이면 없음: 층은 관측 깊이를 씀).
  // layer_n() 바이트를 영상에서 (depth_from_image 와 같은 다시 투영, 픽셀 자리만 sim::cam_uv) → set_layer_depth 로 넣고 low_prepare 앞에
  int layer_n() const { return b_.cfg.cam_lw > 0 ? sim::cam_npix(b_.cfg) : 0; }
  void layer_from_image(const uint16_t* mm, int w, int h, int step_px, float fx, float fy, float cx, float cy, uint8_t* out) const;
  void set_layer_depth(const uint8_t* d);

  // ---- 아래층 (규칙 추종기) — 한 걸음 차례 (시뮬 env_ref.cpp step_env + low_prepare_env 와 같게):
  //   step(감각, 라이다 장) → set_depth(깊이) → [set_cmd(위층 목표, 바뀔 때만)] → set_slot · low_prepare() → low_act(명령) → (명령을 내보냄)
  //   low_prepare = 시뮬 준비 단계 (Env::low_prepare): 계획 차례면 깊이 카메라 층 · 경로 계획(plan_ref)
  //   low_act     = 시뮬 apply_action 의 계층형 가지 (lowrl_begin · path_step_a·b·c — 목표 없으면 제자리 회전, 속도 다듬기·충돌 감시까지 같은 함수)
  void set_cmd(const sim::LowCmd& cmd);   // 합친 탐사 모드: 위층 명령 (combine_run 의 env.set_low_cmd)
  // 합친 탐사 모드: 준비 단계 행동 칸 (combine_run 의 act[0] — 정책 환경이 아닐 때 경로 계획 후보 번호). 기본 −1 (목표 없음). 아래층 규칙이 lxc::Ctl::slot() 을 넘김
  void set_slot(int k) { slot_ = (float)k; }
  const sim::LowEp& low_ep() const { return lowep_[0]; }
  void low_prepare();
  void low_act(float* v, float* w);   // 규칙 추종기 명령 → 내보낼 v (m/s) · ω (rad/s)
  // 실물 멈춤(꺼짐·리모컨·닿음) 걸음: 명령 0 (low_act 대신 — 시뮬에는 없는 걸음)
  void hold();
  const sim::LowRl& low_rl() const { return lowrl_[0]; }
  const sim::LowCmd& low_cmd() const { return lowcmd_[0]; }

  // 읽기
  bool cell_xy(int cell, float* x, float* y) const;
  int cand_cur() const { return b_.cand_cur[0]; }
  int done() const { return done_; }               // 지난 걸음 mdp::cur::Done (0 진행 · 1 닫힘 성공 · 5 진행 없음 · 3 끊김 …)
  int steps() const { return e_.step; }
  const sim::EnvCfg& cfg() const { return b_.cfg; }
  const sim::EnvState& state() const { return e_; }
  const sim::MapDev& map() const { return m_; }
  const int16_t* lmap() const { return b_.lmap; }
  const uint8_t* cam_layer() const { return b_.cvox; }
  const uint32_t* trail() const { return b_.trail; }   // 지나간 칸·본 칸 판 (fuel 위층 입력)
  const int32_t* cand() const { return b_.cand; }     // 후보 칸 K 개

  const sim::EnvBatch& batch() const { return b_; }

  // ---- 주행 목표 모드 (explorer_node 의 NavigateToPose 서버 — 탐사·시뮬 비교에는 안 씀, 부르지 않으면 동작이 바이트 같음)
  // 저장 지도(/map, nav_msgs/OccupancyGrid 값 −1 · 0~100)를 로봇 지도의 아직 모르는 칸에만 채움: 막힘(≥ occ_thresh) → 막힘 로그 오즈 l_occ,
  // 빔(≤ free_thresh) → 빈 칸 l_free. 라이다가 본 칸은 그대로 (본 것이 지도보다 먼저). 칸 가운데가 들어가는 /map 칸 값으로. 채운 칸 수를 돌려줌
  int prime_map(const int8_t* occ, int w, int h, float res, float ox, float oy, int occ_thresh, int free_thresh, int16_t l_occ, int16_t l_free);
  // 이번 걸음 아래층 명령(low_act) 앞에 깊이 열 점(cpts — 지역 계획기 거리 장 · 충돌 감시가 같이 씀) 뒤에 덧붙일 점 (지도 좌표 x, y 차례로 n 개).
  // 초음파 메아리 자리를 넣음. low_prepare 뒤 · low_act 앞에 부름. 깊이 층이 꺼져 있으면(cpts 없음) 아무것도 안 함 — 넣은 점 수를 돌려줌
  int add_points(const float* xy, int n);
  // 아래층 복구(hier_path.h path_recovery)가 포기해 피하는 목표 칸 목록을 비움 — 주행 목표가 새로 오면 (옛 목표 둘레를 계속 피하지 않게)
  void clear_gaveup();

  // 걸음마다 계층형 후보 8 칸(cand() · cand_cur())을 셀지 — 시뮬 기준선 위층 fuel · slot0 만 씀 (low_explore_check · low_explore_replay — 노드는 끔). 끄면 빈 칸(−1).
  // 기본 켬 (시뮬과 같은 후보 — low_explore_check 가 비트 비교)
  void set_obs(bool hier) { obs_hier_ = hier; }
  // 이 걸음 관측의 프론티어 찾기 결과 (후보 뽑기 cand::build 가 같은 입력이면 다시 안 찾음). 깊이 층이 바뀌면(low_prepare) 무효 — nullptr
  const sim::ReachKeep* reach_keep() const { return keep_.valid ? &keep_ : nullptr; }
  // 걸음 사이 프론티어 칸 판정 기억 (mdp/frontier_cache.h — 기본 켬, 값은 같음). every > 0: 그 걸음마다 기억 전체를 처음부터 다시 센 값과 견줌 (다르면 n_bad · 전부 다시)
  void set_frontier_cache(bool on, int check_every = 0) { keep_.use_cache = on; keep_.cache.check_every = check_every; }
  const mdp::FrontierCache& frontier_cache() const { return keep_.cache; }

 private:
  void obs();
  sim::EnvBatch b_{};
  sim::MapDev m_{};
  sim::EnvState e_{};
  sim::Shared sh_{};
  int done_ = 0;
  float slot_ = -1.f;
  bool cam_ran_ = false;          // 이번 low_prepare 에서 깊이 층(cpts)을 새로 셈
  int extra_at_ = -1, extra_n_ = 0;   // add_points 가 지난번 넣은 자리 · 수
  bool obs_hier_ = true;
  sim::ReachKeep keep_;
  // 메모리 (EnvBatch 포인터가 가리킴)
  std::vector<uint8_t> grid_;
  std::vector<int16_t> lmap_;
  std::vector<uint32_t> marks_, trail_;
  std::vector<uint8_t> depth_, cvox_, cldep_;
  std::vector<float> ring_, cpts_;
  std::vector<int32_t> done_buf_, cand_, cand_cur_;
  std::vector<sim::EpInfo> ep_;
  // 아래층 걸음 길 (env_host.cpp 가 cfg.low_rl 일 때 잡는 칸과 같은 것)
  std::vector<sim::LowRl> lowrl_;
  std::vector<sim::LowCmd> lowcmd_;
  std::vector<sim::LowEp> lowep_;
};

}  // namespace limo_fire_explore
