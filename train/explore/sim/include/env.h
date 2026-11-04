// GPU 학습 환경: 첫 지도 만들기(탐사) 한 판의 물리·센서·지도·보상 전부. 환경 하나 = 스레드 블록 하나(kThreads),
// 커널 한 번 = 행동 하나(substeps × dt, 속도 행동 0.1 s · 목표점 행동 0.5 s). GPU 커널(src/env_kernel.cu)과 CPU 참조판(src/env_ref.cpp)이 이 단계 함수들을 같이 불러
// 비트까지 같은 값을 낸다(tools/env_verify). 블록 안 합치기는 정수만(순서와 무관) — 결정적.
//
// 한 행동 (env_step):
//   0. 끝난 판이면 리셋: 맵·시작 칸·방향·라이다 오차 무작위화·지도 비움
//   1. 행동: 속도 행동이면 (v, vy, ω, 선언) 정규 단위 → 한도·바퀴 단위 자르기 → 안전층(앞 길 라이다 < 문턱이면 앞 명령 0) → /cmd_vel.
//      목표점 행동이면 로봇(지도 자세) 기준 방향 n_dir × 거리,
//      정답 충돌 격자로 막히기 전까지 자름 (Nav2 가 피해 가는 몫 대신)
//   2. 물리 substeps 번 (dt, limo_base 50 Hz):
//        (목표점이면 Nav2 Regulated Pure Pursuit 식 → velocity_smoother 가속 한도) → 섀시 구동 모델(drive.h, 실물로 맞춤:
//        지연·가속·1 차 응답, 미끄러짐은 판마다 뽑은 바닥 재질) → 참값 자세 적분 → 몸통 사각형이 충돌 격자에 닿으면 되돌리고 판 끝.
//        IMU 측정(imu.h: 가속도·roll·pitch·자이로 x·y — 몸통 v·ω·가속 + 실측 잡음·진동), 지도 자세 오차(OU, Cartographer 가 남기는 몫)
//      lidar_every 걸음마다: 실물 오차 라이다(lidar_real.h, 스캔 중 움직임 포함) → Cartographer 와 같은 확률 지도 갱신
//        (지도 자세로, 빔 시각만큼 돌려 펴기(unwarp), max_range 넘으면 missing_ray 만큼만 빈 칸, 한 장에 칸마다 한 번, 맞음 우선)
//        + 덮음 표시: 빔 끝점(참 광선 · 잡음 섞인 거리, max_range 안)이 떨어진 벽 면 칸(아래 wall_credit), 빔이 지나간 바닥 칸
//   3. 보상 항(mdp/rewards.h)·끝 판정(mdp/curriculums.h: 선언·충돌 = 종료, 안전 시간 = 끊김)·단계 오르내림,
//      관측(mdp/observations.h — mdp/obs_layout.h 배치 그대로)
//
// 덮음(coverage) = 본 벽 면 칸 / 벽 면 칸 (env_host.cpp build_world: 출발점에서 갈 수 있는 바닥에 4 이웃으로 맞닿은 막힌 칸).
// 바닥 덮음(floor_cov) = 빔이 한 번이라도 지나간(지도 자세의 Bresenham, 끝 칸 빼고) 갈 수 있는 바닥 칸 비율 — 보조 지표, 보상에 안 씀.
#pragma once
#include <cstdint>

#include "curriculums.h"   // mdp/ — 판 끝 코드·단계
#include "drive.h"
#include "imu.h"
#include "frontier.h"      // mdp/ — 칸 종류 (안 넓이·프론티어 요약)
#include "lidar_real.h"
#include "obs_layout.h"    // mdp/ — 관측·행동 크기
#include "rewards.h"       // mdp/ — 보상 항
#include "smoke.h"         // mdp/ — 연기 모드 (센서 가림·속도 한도·추측 항법 오차)
#include "people.h"
#include "sensor_model.h"
#include "sensors.h"

namespace sim {

constexpr int kThreads = 256;
constexpr int16_t kUnknown = -32768;   // 지도 칸 아직 모름
constexpr int kMaxFloors = 8;
enum ActionMode : int32_t { kActGoal = 0, kActVelocity = 1 };

struct EnvCfg {
  float dt;
  int substeps, lidar_every;
  // 추종기 (Nav2 RPP · velocity_smoother · goal checker)
  float v_des, lookahead, min_lookahead, rot_w, rot_min_angle, approach_dist, approach_min_v, goal_tol;
  float v_max, w_max, a_max, alpha_max;
  // 몸통
  float half_len, half_wid;
  DriveModel drive;
  int n_floors;
  float floor_cum[kMaxFloors];          // 무게 누적 (0~1)
  Floor floor_lo[kMaxFloors], floor_hi[kMaxFloors];   // 바닥 재질 값 범위 (판마다 이 안에서 고르게)
  float mount_x, mount_y, lidar_z;
  // 지도 (Cartographer)
  float res, inv_res;        // inv_res = 1 / res (칸 번호를 나눗셈 없이 — dm::floor_div)
  int size;                 // 환경마다 지도 size × size 칸 (맵 격자는 이 안에)
  int16_t l_hit, l_miss, l_min, l_max;   // 로그 오즈 × 1000
  float map_max_range, missing_ray;
  // 지도 자세 오차 (OU)
  float pe_xy, pe_yaw, pe_tau;
  // 행동 (mdp/README "행동"): 속도 행동은 환경마다 float mdp::act::DIM 개 = (v, vy, ω, 끝 선언)
  int action_mode;
  float act_v_max, act_w_max;   // 물리 단위 행동(act_raw)의 한도 = drive.max
  int act_raw;              // 1: 행동이 물리 단위 (v, vy, ω, 선언) — 실물 녹화 다시 돌리기·기준선 도구용, 바퀴 자르기 없음
  float act_v_fwd, act_v_back, act_vy_max;   // 정규 단위 [−1, 1] → v ∈ [−back, fwd], vy ∈ ±vy_max, ω ∈ ±act_w_max
  float wheel_k, wheel_max;  // 메카넘 바퀴 선속도 = v ± vy ± k·ω, |·| ≤ wheel_max 넘으면 (v, vy, ω) 를 같은 비로 줄임
  int safety;               // 안전층 켬 (mdp/README "안전층")
  float safety_dist;        // 앞 범퍼에서 이 거리 안(몸통 폭 + 5 cm 길)에 직전 라이다 점 → 앞으로 가는 명령 0
  int curriculum;           // 1: 환경마다 단계(맵 묶음·오차 배율), 0: 모든 맵 고르게·오차 배율 1 (기준선·도구)
  int max_delay;            // 관측 지연 최대 (행동 걸음)
  int n_dir, n_dist;
  float dist[4];
  float goal_margin;        // 목표점을 막힌 칸 이만큼 앞에서 자름
  // IMU 자이로 z (관측용, 실측 analysis/real_limo.md)
  float gyro_sigma, gyro_bias;
  // IMU 나머지 (가속도 x·y·z, roll·pitch, 자이로 x·y): 실물에 맞춘 모델 (imu.h, sensor_model.yaml imu)
  ImuModel imu;
  int max_steps;            // 안전 시간 (끊김) — mdp::cur::kSafeTime
  // 깊이 카메라 (관측용 줄인 영상): 실물 모델(sensor_model.yaml depth), 장착 (base_link 기준 x, y, 바닥에서 높이)
  DepthModel depth;
  float cam_x, cam_y, cam_z;
  float cam_cp, cam_sp;     // 카메라를 아래로 숙인 각의 cos·sin (env.yaml body.camera_pitch, 실측 바닥 면 — 호스트가 계산)
  int depth_w, depth_h;     // 관측 영상 크기 — 원본 칸을 고르게 건너뛴 가운데 광선
  PeopleCfg people;         // 움직이는 사람 (people.h)
};

// 칸 격자 비트 (MapDev::grid, World::grid — env_host.cpp build_world)
constexpr uint8_t kCellBlocked = 1;   // 몸통 높이(collision_band)에 형상이 걸린 칸 — 충돌
constexpr uint8_t kCellFloor = 2;     // 로봇이 출발점에서 갈 수 있는 바닥 칸 (바닥 덮음 분모)
constexpr uint8_t kCellWall = 4;      // 그 바닥에 4 이웃으로 맞닿은 막힌 칸 = 벽 면 칸 (덮음 분모)

// 맵 하나 (장면 + 정답 격자). 격자는 지도와 같은 칸(res), 원점 (ox, oy)
struct MapDev {
  SceneView scene;
  const uint8_t* grid;     // 칸마다 kCell* 비트
  const int32_t* spawn;    // 시작 칸 번호들 (몸통이 넉넉히 들어감)
  const uint8_t* clear;    // 칸마다 가장 가까운 막힌 칸까지 (cm) — 사람이 설 수 있나 (people_sim.h person_free)
  int gw, gh, n_spawn, n_wall, n_floor;
  float ox, oy;
  float t_ref;             // 기준선 속도로 다 덮는 시간 (mdp::rew::t_ref) — 초당 감점 = kTime / t_ref
  int stage;               // 이 맵의 커리큘럼 단계 (넓이로)
};

constexpr int kLidarRing = mdp::obs::LIDAR_FRAMES + 1;   // 라이다 칸 링 (지금 + 직전 3 장 + 지연 1 장)
constexpr int kProgRing = 20;                             // 탐사 진행 속도: 최근 20 걸음(2 s) 새로 안 칸 수
constexpr int kBinNone = 1 << 30;                         // 라이다 칸에 점 없음

// 판 끝 정보 (러너·평가용, done ≠ 0 인 걸음에만 새로 씀)
struct EpInfo {
  float coverage, t95, t;  // 끝날 때 벽 덮음, 95 % 처음 닿은 시각 (s, 못 닿으면 −1), 판 길이 (s)
  int32_t done, collided, stage, map;
};

struct EnvState {
  float x, y, th, v, w;        // 참값 자세·속도 (세계)
  float vs, ws;                // velocity_smoother 출력 (목표점 행동의 명령)
  float cmd_v, cmd_w;          // 속도 행동의 명령 (/cmd_vel)
  DriveState drive;            // 섀시 구동 (바퀴 속도 = 오도메트리)
  Floor ground;                // 이 판 바닥 (미끄러짐·마찰)
  int32_t floor, tick;         // 바닥 번호, 물리 걸음 수
  float ex, ey, eth;           // 지도 자세 = 참값 + 오차
  float gx, gy;                // 목표점 (지도 자세 기준 세계 좌표)
  uint64_t rng, world_seed;
  int32_t map, step, seen_wall, seen_floor, done;   // 본 벽 면 칸·지나간 바닥 칸 수. done: mdp::cur::Done (0 진행, 1 선언 성공, 2 충돌, 3 끊김, 4 선언 실패)
  int32_t last_action, collided, declared;
  float cmd_vy, vy;            // 옆 이동 명령·몸통 옆 속도 (로봇 기준)
  float x0, y0, th0;           // 시작 자세 (판 시작 때 참값 = 지도 자세, 시작점 기준 관측의 원점)
  float last_act[mdp::act::DIM * 4];   // 직전 행동 4 걸음 (정규 단위로 자른 값, 최근 것 먼저)
  float ekf_v, ekf_w;          // EKF 속도 자리: 바퀴 v·자이로 ω 의 1 차 필터 (잠정 — 실물 robot_localization 은 📏)
  int32_t known_free, prog[kProgRing], prog_sum, since_new;   // 지도에서 갈 수 있음 칸 수, 걸음마다 새로 안 칸 수 링·합, 마지막 새 칸 뒤 걸음
  int32_t since_wall;   // 마지막 새 벽 면 칸(덮음 증가) 뒤 걸음 — 진행 없음 끝 (curriculums.h kIdleTime)
  int32_t kb_x0, kb_y0, kb_x1, kb_y1;   // 지도에서 건드린 칸 범위 (프론티어 요약을 이 안에서만)
  int32_t lid_head, delay;     // 라이다 링 최신 칸, 이 판 관측 지연 (걸음)
  int32_t stage, cur_n;        // 커리큘럼 단계, 최근 판 기록 수
  uint32_t cur_hist;           // 최근 판 끝 덮음 ×1000 합 (curriculums.h on_episode_end)
  int32_t t95_step;            // 덮음 95 % 처음 닿은 걸음 (−1 아직)
  float err_scale;             // 이 판 자세 오차·자이로 잡음 배율 (단계)
  float reward, coverage, floor_cov;   // coverage = seen_wall / n_wall, floor_cov = seen_floor / n_floor
  float gyro, gyro_bias;       // 자이로 측정 (몸통 ω + 이 판 바이어스 + 잡음)
  ImuState imu;                // IMU 나머지 측정 (imu.out: ax, ay, az, gx, gy, roll, pitch — REP-103, 물리 걸음마다)
  float slip_acc;              // 이번 행동 동안 |바퀴 ω − 몸통 ω| · dt 합 (헛돎 감점)
  LidarNoise noise;            // 이 판 라이다 오차 (무작위화)
  People people;              // 이 판 움직이는 사람 (people.h)
  int32_t smoke;               // 연기 모드 (mdp/smoke.h): 라이다·깊이 가림, 지도 고정, 추측 항법 자세, 안전층 대신 속도 한도. Env::set_smoke, 리셋이 0 으로
  int32_t smoke_steps;         // 연기가 낀 뒤 걸음 수 (켜고 끌 때 0). 구조체 끝을 8 바이트로 채워 패딩이 없게(env_verify 가 memcmp)
};

// 블록 안 공유 (GPU __shared__, CPU 는 지역 변수)
struct Shared {
  ScanIn scan;
  int n_beams;
  float mx, my, mth, mw;       // 지도 자세의 라이다 위치·방향·각속도 (지도 넣기용)
  float inc;
  int bx0, by0, bx1, by1;      // 이번 장에 건드린 칸 범위
  int part[kThreads];          // obs_cells 스레드마다 프론티어 칸 수
  int bins[mdp::obs::LIDAR_BINS];   // 이번 장 각도 칸마다 가장 가까운 거리 (mm)
  int scanned;                 // 이번 행동에 라이다 장이 있었나
  unsigned long long fr_min;   // 프론티어 요약: summary_key 최솟값
  float vec[mdp::obs::VEC];    // 이번 걸음 숫자 관측 (지연 전)
  // 한 행동 안 진행 (GPU 는 물리·빔·지도 갱신을 커널로 나눠 돌려 환경마다 장치 메모리 EnvBatch::scratch 에 둔다)
  int k_next, tick0, pending;  // 다음 물리 걸음 번호, 행동 시작 tick, 빔을 아직 안 쏜 장이 있음
  int new_wall, new_floor, d_known, n_new;   // 이번 행동 동안 합 (scan_apply)
  // 먼 지도 다시 그릴 곳 (GPU): 이번 행동에 지도·지나간 곳이 바뀐 칸 범위, 지난번 그린 판(시작 자세·맵)
  int dx0, dy0, dx1, dy1, far_ok, far_map;
  float far_key[3];
};

// scan_apply 가 스레드 하나 몫으로 센 수 (합은 순서와 무관한 정수 합)
struct ApplyCount {
  int wall, floor, known, fresh;   // 새로 본 벽 면 칸, 새로 지나간 바닥 칸, 갈 수 있음 칸 변화, 모름 → 앎 칸
};

// 장치·호스트 공용 묶음 (포인터는 GPU 면 장치, CPU 면 호스트)
struct EnvBatch {
  EnvCfg cfg;
  const MapDev* maps;
  int n_maps;
  LidarModel lidar;
  EnvState* st;
  int16_t* lmap;               // [env · size² + cy · size + cx]
  uint32_t* marks;             // [env · size²/4]  칸마다 한 바이트: 이번 장 bit0 지나감 · bit1 맞음 · bit2 끝점이 이 벽 면 칸에,
                               //                   판 동안 남음 bit3 본 벽 면 칸 · bit4 지나간 바닥 칸 (Env::get_seen)
  // 관측 (mdp/obs_layout.h, mdp/observations.h)
  uint8_t* obs_img;            // [env · IMG_SIZE]  가까운 지도 · 먼 지도 · 깊이 (uint8)
  float* obs_1d;               // [env · POLICY_1D]  라이다 4 장 · 세기 · 숫자
  float* priv;                 // [env · PRIV]  가치 전용 참값
  float* lidar_ring;           // [env · kLidarRing · LIDAR_BINS]  각도 칸 거리 (m, 없음 = map_max_range)
  float* vec_prev;             // [env · VEC]  직전 걸음 숫자 관측 (지연용)
  float* lidar_int;            // [env · LIDAR_BINS]  지금 장 칸 세기 (가장 가까운 빔, 점 없음 = 0)
  float* reward;               // [env] 이번 행동 보상 (= reward_terms 합)
  float* reward_terms;         // [env · mdp::rew::kTerms]
  int32_t* done;               // [env] mdp::cur::Done. 끝난 판은 바로 리셋돼 관측은 새 판 것
  EpInfo* ep;                  // [env] 끝난 판 정보 (done ≠ 0 인 걸음에 씀)
  // 커리큘럼: 단계 s 에서 고를 맵 = stage_maps[stage_off[s] .. stage_off[s + 1])
  const int32_t* stage_maps;
  int stage_off[mdp::cur::kNumStages + 1];
  int max_stage;
  int n_env;
  Shared* scratch;             // [env] GPU 판: 커널 사이에 넘기는 장 준비·한 행동 합 (CPU 판은 안 씀)
  // GPU 판 깊이 광선 묶음 (env_kernel.cu): 픽셀마다 가운데 광선 거리·삼각형, 옆 광선 큐(픽셀 · 2 + 왼0/오른1)·그 거리, 원자 셈 칸
  float* dep_z;                // [env · DEPTH_SIZE]
  int32_t* dep_tri;            // [env · DEPTH_SIZE]
  float* dep_side;             // [env · DEPTH_SIZE · 2]
  int32_t* dep_jobs;           // [env · DEPTH_SIZE · 2]
  // 라이다 빔 광선 묶음: 빔(env · kMaxBeams + j)마다 가운데 광선 거리·삼각형, 옆 광선 큐(빔 · 4 + k)·그 거리
  float* beam_t;
  int32_t* beam_tri;
  float* beam_side;
  int32_t* beam_jobs;
  int32_t* counters;           // [kCounters] (env_kernel.cu)
  // RT 코어 (OptiX, 선택): 있으면 깊이 가운데 광선을 하드웨어로 (include/optix_rt.h). 광선 준비 칸 [env · DEPTH_SIZE]
  void* rt;                    // OptixRt* (호스트 객체, 없으면 nullptr)
  void* rt_o;                  // float4
  void* rt_d;                  // float4
  int32_t* rt_tri0;
  float* rt_t;                 // 결과 거리 (번호는 rt_tri0 자리에 덮어씀)
};

#include "people_sim.h"   // 사람 동작 (EnvState·MapDev 뒤)

// ---------- 작은 도구 ----------

DEV float wrapf(float a) { return dm::wrap_pi(a); }

DEV int cell_of(const EnvCfg& c, const MapDev& m, float x, float y, int* cx, int* cy) {
  *cx = dm::floor_div(x - m.ox, c.res, c.inv_res);
  *cy = dm::floor_div(y - m.oy, c.res, c.inv_res);
  return (*cx >= 0 && *cy >= 0 && *cx < m.gw && *cy < m.gh) ? (*cy * c.size + *cx) : -1;
}

DEV bool blocked(const EnvCfg& c, const MapDev& m, float x, float y) {
  int cx, cy;
  if (cell_of(c, m, x, y, &cx, &cy) < 0) return true;
  return (m.grid[cy * m.gw + cx] & 1) != 0;
}

// 몸통 사각형(가장자리 12 점 + 가운데)이 막힌 칸에 닿나
DEV bool body_hits(const EnvCfg& c, const MapDev& m, float x, float y, float th) {
  float s, co;
  dm::sincosf_d(th, &s, &co);
  for (int i = 0; i < 13; ++i) {
    float u, v;
    if (i == 12) { u = 0.f; v = 0.f; }
    else {
      const int side = i / 3, k = i % 3;
      const float t = (float)(k - 1);   // −1, 0, 1
      u = side < 2 ? (side == 0 ? c.half_len : -c.half_len) : t * c.half_len;
      v = side < 2 ? t * c.half_wid : (side == 2 ? c.half_wid : -c.half_wid);
    }
    const float px = (x + co * u) - s * v, py = (y + s * u) + co * v;
    if (blocked(c, m, px, py)) return true;
  }
  return false;
}

// ---------- 0. 리셋 ----------

// 스레드 0: 상태
DEV void reset_state(const EnvBatch& b, EnvState& e) {
  const EnvCfg& c = b.cfg;
  if (c.curriculum) {
    // 단계에 맞는 맵 묶음에서 (판 끝에 on_episode_end 가 단계를 고침 — finish_step)
    const int s = e.stage < 0 ? 0 : (e.stage > b.max_stage ? b.max_stage : e.stage);
    e.stage = s;
    const int n = b.stage_off[s + 1] - b.stage_off[s];
    e.map = b.stage_maps[b.stage_off[s] + (int)(dm::splitmix64(e.rng) % (uint64_t)n)];
    e.err_scale = mdp::cur::err_scale_of(s);
  } else {
    e.map = (int)(dm::splitmix64(e.rng) % (uint64_t)b.n_maps);
    e.err_scale = 1.f;
  }
  const MapDev& m = b.maps[e.map];
  const int k = m.spawn[dm::splitmix64(e.rng) % (uint64_t)m.n_spawn];
  e.x = m.ox + ((float)(k % m.gw) + 0.5f) * c.res;
  e.y = m.oy + ((float)(k / m.gw) + 0.5f) * c.res;
  e.th = dm::rand_range(e.rng, -3.14159265f, 3.14159265f);
  e.v = e.w = e.vs = e.ws = e.cmd_v = e.cmd_w = 0.f;
  drive_reset(e.drive);
  const float u = dm::rand01(e.rng);
  e.floor = 0;
  while (e.floor < c.n_floors - 1 && u >= c.floor_cum[e.floor]) ++e.floor;
  const Floor& lo = c.floor_lo[e.floor];
  const Floor& hi = c.floor_hi[e.floor];
  e.ground.slip_v = dm::rand_range(e.rng, lo.slip_v, hi.slip_v);
  e.ground.slip_w = dm::rand_range(e.rng, lo.slip_w, hi.slip_w);
  e.ground.trac_v = dm::rand_range(e.rng, lo.trac_v, hi.trac_v);
  e.ground.trac_w = dm::rand_range(e.rng, lo.trac_w, hi.trac_w);
  e.ground.tau_v = dm::rand_range(e.rng, lo.tau_v, hi.tau_v);
  e.ground.tau_w = dm::rand_range(e.rng, lo.tau_w, hi.tau_w);
  e.tick = 0;
  e.gyro = 0.f;
  e.gyro_bias = dm::rand_range(e.rng, -c.gyro_bias, c.gyro_bias);
  imu_reset(c.imu, e.imu, e.rng);
  e.slip_acc = 0.f;
  e.ex = e.ey = e.eth = 0.f;
  e.gx = e.x;
  e.gy = e.y;
  e.world_seed = dm::splitmix64(e.rng);
  e.noise = randomize(b.lidar, e.rng);
  e.step = 0;
  e.seen_wall = 0;
  e.seen_floor = 0;
  e.done = 0;
  e.last_action = -1;
  e.collided = 0;
  e.declared = 0;
  e.reward = 0.f;
  e.coverage = 0.f;
  e.floor_cov = 0.f;
  e.cmd_vy = e.vy = 0.f;
  e.x0 = e.x;
  e.y0 = e.y;
  e.th0 = e.th;
  for (int i = 0; i < mdp::act::DIM * 4; ++i) e.last_act[i] = 0.f;
  e.ekf_v = e.ekf_w = 0.f;
  e.known_free = 0;
  for (int i = 0; i < kProgRing; ++i) e.prog[i] = 0;
  e.prog_sum = 0;
  e.since_new = 0;
  e.since_wall = 0;
  e.kb_x0 = e.kb_y0 = 1 << 30;
  e.kb_x1 = e.kb_y1 = -1;
  e.lid_head = 0;
  // 관측 지연: 판마다 0~150 ms 고르게 (실물 라이다 지연 0.10 s + 추론). 관측은 행동마다(0.1 s) 만들므로 걸음 단위로 내림 → 0 걸음(2/3)·1 걸음(1/3)
  e.delay = (int)(dm::rand01(e.rng) * 150.f) / 100;
  if (e.delay > c.max_delay) e.delay = c.max_delay;
  e.t95_step = -1;
  e.smoke = 0;   // 판은 연기 없이 시작 (켜는 쪽: Env::set_smoke)
  e.smoke_steps = 0;
  if (c.curriculum) e.people.k_max = mdp::cur::people_of(e.stage);   // 단계별 사람 수 (curriculums.h)
  people_reset(b, e);   // 사람 놓기 (로봇 자세·world_seed 뒤)
}

// 모든 스레드: 지도·표시 비우기
DEV void reset_map(const EnvBatch& b, int env, int tid) {
  const int n = b.cfg.size * b.cfg.size;
  int16_t* L = b.lmap + (size_t)env * n;
  uint32_t* M = b.marks + (size_t)env * (n / 4);
  for (int i = tid; i < n; i += kThreads) L[i] = kUnknown;
  for (int i = tid; i < n / 4; i += kThreads) M[i] = 0u;
  float* R = b.lidar_ring + (size_t)env * kLidarRing * mdp::obs::LIDAR_BINS;
  for (int i = tid; i < kLidarRing * mdp::obs::LIDAR_BINS; i += kThreads) R[i] = b.cfg.map_max_range;   // 아직 장 없음 = 없음
  float* I = b.lidar_int + (size_t)env * mdp::obs::LIDAR_BINS;
  for (int i = tid; i < mdp::obs::LIDAR_BINS; i += kThreads) I[i] = 0.f;
}

// ---------- 1. 행동 (스레드 0) ----------

DEV void set_goal(const EnvBatch& b, EnvState& e, int action);

DEV float clip1(float a) { return a != a ? 0.f : dm::clampf(a, -1.f, 1.f); }   // NaN → 0

// 안전층: 직전 라이다 장(링 최신 칸)에서 몸통 폭(+5 cm) 길 안 앞쪽 점까지 앞 범퍼로부터 거리 (m). 없으면 큰 값.
// 실물 정지 감시(limo_fire_rl)도 같은 식 — /scan 을 같은 248 각도 칸으로 모아 같은 문턱.
DEV float front_gap(const EnvBatch& b, const EnvState& e, int env) {
  const EnvCfg& c = b.cfg;
  const float* R = b.lidar_ring + ((size_t)env * kLidarRing + e.lid_head) * mdp::obs::LIDAR_BINS;
  const float half = c.half_wid + 0.05f, bumper = c.half_len - c.mount_x;
  float best = 1e9f;
  for (int k = 0; k < mdp::obs::LIDAR_BINS; ++k) {
    const float r = R[k];
    if (r >= c.map_max_range) continue;
    const float a = mdp::obs::kLidarMin + ((float)k + 0.5f) * mdp::obs::kLidarBin;
    float s, co;
    dm::sincosf_d(a, &s, &co);
    const float fx = r * co, fy = r * s;
    if (fx <= 0.f || dm::absf(fy) > half) continue;
    const float g = fx - bumper;
    best = g < best ? g : best;
  }
  return best;
}

// a: 환경마다 float mdp::act::DIM 개. 속도 행동이면 (v, vy, ω, 선언) — 정규 단위 [−1, 1] (act_raw 면 물리 단위),
// 목표점 행동이면 a[0] = 행동 번호 (a[3] 선언은 같음)
DEV void apply_action(const EnvBatch& b, EnvState& e, int env, const float* a) {
  const EnvCfg& c = b.cfg;
  e.people.hit_person = 0;
  e.declared = a[3] > 0.5f ? 1 : 0;   // 베르누이 표본 0/1
  if (c.action_mode == kActVelocity) {
    float v, vy, w;
    float n0, n1, n2;
    if (c.act_raw) {
      v = dm::maxf(-c.act_v_max, dm::minf(c.act_v_max, a[0]));
      vy = dm::maxf(-c.act_v_max, dm::minf(c.act_v_max, a[1]));
      w = dm::maxf(-c.act_w_max, dm::minf(c.act_w_max, a[2]));
      n0 = v / (v >= 0.f ? c.act_v_fwd : c.act_v_back);
      n1 = vy / c.act_vy_max;
      n2 = w / c.act_w_max;
    } else {
      n0 = clip1(a[0]);
      n1 = clip1(a[1]);
      n2 = clip1(a[2]);
      v = n0 >= 0.f ? n0 * c.act_v_fwd : n0 * c.act_v_back;
      vy = n1 * c.act_vy_max;
      w = n2 * c.act_w_max;
      // 바퀴 단위로 자름: 가장 빠른 바퀴 |v| + |vy| + k|ω| 가 한도를 넘으면 같은 비로 줄임 (방향 유지)
      const float wheel = (dm::absf(v) + dm::absf(vy)) + c.wheel_k * dm::absf(w);
      if (wheel > c.wheel_max) {
        const float k = c.wheel_max / wheel;
        v = v * k;
        vy = vy * k;
        w = w * k;
      }
    }
    // 연기 속: 라이다 안전층 대신 속도 한도 (mdp/smoke.h — 실물 노드도 같은 규칙)
    mdp::smoke::cap_command(e.smoke, &v, &vy, &w);
    if (c.safety && mdp::smoke::lidar_safety(e.smoke) && v > 0.f && front_gap(b, e, env) < c.safety_dist) v = 0.f;
    e.cmd_v = v;
    e.cmd_vy = vy;
    e.cmd_w = w;
    e.last_action = -1;
    for (int i = mdp::act::DIM * 4 - 1; i >= mdp::act::DIM; --i) e.last_act[i] = e.last_act[i - mdp::act::DIM];
    e.last_act[0] = dm::clampf(n0, -1.f, 1.f);
    e.last_act[1] = dm::clampf(n1, -1.f, 1.f);
    e.last_act[2] = dm::clampf(n2, -1.f, 1.f);
    e.last_act[3] = (float)e.declared;
  } else {
    set_goal(b, e, (int)a[0]);
  }
}

DEV void set_goal(const EnvBatch& b, EnvState& e, int action) {
  const EnvCfg& c = b.cfg;
  const MapDev& m = b.maps[e.map];
  const int dir = action % c.n_dir, di = (action / c.n_dir) % c.n_dist;
  float s, co;
  dm::sincosf_d(e.eth + (float)dir * (6.28318531f / (float)c.n_dir), &s, &co);
  // 지도 자세에서 그 방향으로, 정답 격자에서 막히기 goal_margin 앞까지
  const float px = e.x + e.ex, py = e.y + e.ey;
  const float step = 0.5f * c.res;
  float d = 0.f;
  const float dmax = c.dist[di];
  while (d + step <= dmax + c.goal_margin) {
    const float t = d + step;
    if (blocked(c, m, px + co * t, py + s * t)) break;
    d = t;
  }
  d = dm::maxf(0.f, dm::minf(dmax, d - c.goal_margin));
  e.gx = px + co * d;
  e.gy = py + s * d;
  e.last_action = action;
}

// ---------- 2. 물리 한 걸음 (스레드 0) ----------

DEV float approach(float cur, float target, float max_step) {
  const float d = target - cur;
  return cur + (d > max_step ? max_step : (d < -max_step ? -max_step : d));
}

DEV void physics_step(const EnvBatch& b, EnvState& e) {
  const EnvCfg& c = b.cfg;
  const MapDev& m = b.maps[e.map];
  people_step(b, e);   // 사람 먼저 움직임
  float cmd_v, cmd_w;
  if (c.action_mode == kActVelocity) {
    cmd_v = e.cmd_v;
    cmd_w = e.cmd_w;
  } else {
    // 추종기: 지도 자세로 목표점을 본다 (실물은 Nav2 가 map 좌표로 따라간다)
    const float px = e.x + e.ex, py = e.y + e.ey, pth = e.th + e.eth;
    const float dx = e.gx - px, dy = e.gy - py;
    const float d = sqrtf(dx * dx + dy * dy);
    float vc = 0.f, wc = 0.f;
    if (d > c.goal_tol) {
      const float err = wrapf(dm::atan2f_d(dy, dx) - pth);
      if (dm::absf(err) > c.rot_min_angle) {
        wc = err > 0.f ? c.rot_w : -c.rot_w;
      } else {
        vc = c.v_des;
        if (d < c.approach_dist) vc = dm::maxf(c.v_des * d / c.approach_dist, c.approach_min_v);
        const float ld = dm::maxf(c.min_lookahead, dm::minf(c.lookahead, d));
        float se, ce;
        dm::sincosf_d(err, &se, &ce);
        wc = vc * (2.f * se / ld);
        wc = dm::maxf(-c.w_max, dm::minf(c.w_max, wc));
      }
    }
    // velocity_smoother 가속 한도
    e.vs = approach(e.vs, dm::maxf(-c.v_max, dm::minf(c.v_max, vc)), c.a_max * c.dt);
    e.ws = approach(e.ws, wc, c.alpha_max * c.dt);
    cmd_v = e.vs;
    cmd_w = e.ws;
  }
  // 섀시 구동: 명령 → 바퀴 → 몸통 (바닥 미끄러짐)
  const float v_prev = e.v;
  const float x_prev = e.x, y_prev = e.y, th_prev = e.th;   // 연기 속 추측 항법 오차 (이번 걸음 참 이동)
  e.vy = drive_lat(c.drive, e.drive, c.action_mode == kActVelocity ? e.cmd_vy : 0.f, c.dt, e.ground);
  drive_step(c.drive, e.drive, cmd_v, cmd_w, c.dt, e.ground, &e.v, &e.w);
  const float acc = (e.v - v_prev) / c.dt;   // 몸통 직진 가속 (IMU)
  // 적분 (가운데 방향). 옆 속도 vy 는 로봇 기준 왼쪽
  const float thm = e.th + 0.5f * e.w * c.dt;
  float s, co;
  dm::sincosf_d(thm, &s, &co);
  const float nx = e.x + (e.v * co - e.vy * s) * c.dt, ny = e.y + (e.v * s + e.vy * co) * c.dt, nth = wrapf(e.th + e.w * c.dt);
  const bool hit_p = people_gap(c, e.people, nx, ny, nth) < 0.f;   // 몸통이 사람과 겹침 (사람이 다가와 겹쳐도)
  if (body_hits(c, m, nx, ny, nth) || hit_p) {
    e.v = e.w = e.vs = e.ws = e.vy = 0.f;
    drive_reset(e.drive);   // 부딪히면 바퀴·몸통 모두 멈춤 (밀린 명령도 비움)
    e.collided = 1;
    e.people.hit_person |= (int32_t)hit_p;
  } else {
    e.x = nx;
    e.y = ny;
    e.th = nth;
  }
  e.people.near_d = people_gap(c, e.people, e.x, e.y, e.th);
  // 자이로 측정·헛돎 (실물 정책도 바퀴 오도메트리 ω 와 IMU 자이로를 본다)
  e.gyro = (e.w + e.gyro_bias) + (c.gyro_sigma * e.err_scale) * gauss(e.rng);
  // IMU 나머지. 옆 이동(vy)의 옆 가속은 아직 안 넣음 — imu.h 는 2D 몸통 v·ω·직진 가속만 받고(ay = k_ay·v·ω + g·roll), vy 명령 녹화가 없어
  // 옆 가속 배율을 못 맞춤 📏 (vy 계단 녹화로 imu_check 를 늘린 뒤 넣는다)
  imu_step(c.imu, e.imu, e.v, e.w, acc, e.rng);   // 부딪혀 멈춘 걸음은 v·ω 0 (충돌 충격은 모델에 없음)
  e.slip_acc = e.slip_acc + dm::absf(e.drive.ww - e.w) * c.dt;
  if (e.smoke) {
    // 연기 속 지도 자세 = 들어갈 때 지도 자세 ∘ 바퀴 + IMU 이동: 오차가 이동량만큼 쌓인다 (mdp/smoke.h drift_step, 잠정 계수)
    const float g0 = gauss(e.rng), g1 = gauss(e.rng), g2 = gauss(e.rng), g3 = gauss(e.rng);
    mdp::smoke::drift_step(&e.ex, &e.ey, &e.eth, e.x - x_prev, e.y - y_prev, wrapf(e.th - th_prev), e.err_scale, g0, g1, g2, g3);
    return;
  }
  // 지도 자세 오차: OU (평균 0, 표준편차 pe_*, 상관 시간 pe_tau)
  const float a = c.dt / c.pe_tau, q = sqrtf(2.f * a);
  const float sxy = c.pe_xy * e.err_scale, syaw = c.pe_yaw * e.err_scale;   // 단계 오차 배율 (curriculums.h)
  e.ex = e.ex - e.ex * a + sxy * q * gauss(e.rng);
  e.ey = e.ey - e.ey * a + sxy * q * gauss(e.rng);
  e.eth = e.eth - e.eth * a + syaw * q * gauss(e.rng);
}

// ---------- 라이다 + 지도 ----------

// 스레드 0: 이번 장 준비
DEV void scan_begin(const EnvBatch& b, const EnvState& e, Shared& sh) {
  const EnvCfg& c = b.cfg;
  uint64_t r = e.rng ^ ((uint64_t)(e.tick + 17) * 0xA24BAED4963EE407ull);   // 장마다 다른 난수 (물리 걸음 번호)
  float s, co;
  dm::sincosf_d(e.th, &s, &co);
  ScanIn& in = sh.scan;
  in.pose = Pose{{co, -s, 0.f, s, co, 0.f, 0.f, 0.f, 1.f},
                 V3{(e.x + co * c.mount_x) - s * c.mount_y, (e.y + s * c.mount_x) + co * c.mount_y, c.lidar_z}};
  in.v = e.v;
  in.w = e.w;
  in.n_rev = sample_n_rev(e.noise, dm::rand01(r));
  in.phase = phase_of(e.noise, dm::rand01(r));
  in.seed = dm::splitmix64(r);
  in.world_seed = e.world_seed;
  in.noise = 0;
  in.pad = 0;
  sh.n_beams = scan_beams(e.noise, in.n_rev);
  sh.inc = kTwoPi / in.n_rev;
  const float pth = e.th + e.eth;
  float ps, pc;
  dm::sincosf_d(pth, &ps, &pc);
  sh.mx = (e.x + e.ex + pc * c.mount_x) - ps * c.mount_y;
  sh.my = (e.y + e.ey + ps * c.mount_x) + pc * c.mount_y;
  sh.mth = pth;
  sh.mw = e.w;
  sh.bx0 = sh.by0 = 1 << 30;
  sh.bx1 = sh.by1 = -1;
  for (int k = 0; k < mdp::obs::LIDAR_BINS; ++k) sh.bins[k] = kBinNone;
  sh.scanned = 1;
}

// 관측 라이다 칸: ROS 가 적는 빔 각도(−fov/2 + j·inc — 실물 /scan 과 같은 각도, 펴기 전)로 ±110° 를 248 칸에 나누고 칸마다 가장 가까운 거리.
// 빔 없거나 안 맞은 칸 = 없음(kBinNone → map_max_range). 실물 노드도 같은 규칙.
// 칸 값 = 거리 mm << 16 | 세기(0~65535) — 가장 가까운 빔의 세기가 같이 남는다(같은 거리면 작은 세기, 순서와 무관).
// 세기는 맞은 빔만(lidar_intensity, 실측 맞춤). 점 없는 칸 = 거리 없음·세기 0 (실물 노드도 같은 규칙)
DEV void bin_beam(Shared& sh, float a_ros, float r, float max_r, float inten) {
  const float u = (a_ros - mdp::obs::kLidarMin) / mdp::obs::kLidarBin;
  if (!(u >= 0.f) || u >= (float)mdp::obs::LIDAR_BINS) return;
  const int k = (int)u;
  const float rr = r < max_r ? r : max_r;
  const int it = inten < 0.f ? 0 : (inten > 65535.f ? 65535 : (int)inten);
  const int mm = ((int)(rr * 1000.f) << 16) | it;
#ifdef __CUDA_ARCH__
  atomicMin(&sh.bins[k], mm);
#else
  if (mm < sh.bins[k]) sh.bins[k] = mm;
#endif
}

DEV void mark(uint32_t* M, int size, int cx, int cy, uint32_t bits) {
  const int i = cy * size + cx;
#ifdef __CUDA_ARCH__
  atomicOr(&M[i >> 2], bits << (8 * (i & 3)));
#else
  M[i >> 2] |= bits << (8 * (i & 3));
#endif
}

DEV void atomic_min_i(int* p, int v) {
#ifdef __CUDA_ARCH__
  atomicMin(p, v);
#else
  if (v < *p) *p = v;
#endif
}
DEV void atomic_max_i(int* p, int v) {
#ifdef __CUDA_ARCH__
  atomicMax(p, v);
#else
  if (v > *p) *p = v;
#endif
}

// 칸 (cx, cy) 안의 점 (px, py) 에서 방향 (vx, vy) 로 광선이 다음에 들어가는 이웃 칸 (Amanatides–Woo 한 걸음). 경계까지 거리가 같으면 y 쪽
DEV void ray_next_cell(const EnvCfg& c, const MapDev& m, float px, float py, float vx, float vy, int* cx, int* cy) {
  const float tx = vx > 0.f ? ((m.ox + (float)(*cx + 1) * c.res) - px) / vx : (vx < 0.f ? ((m.ox + (float)*cx * c.res) - px) / vx : 3e38f);
  const float ty = vy > 0.f ? ((m.oy + (float)(*cy + 1) * c.res) - py) / vy : (vy < 0.f ? ((m.oy + (float)*cy * c.res) - py) / vy : 3e38f);
  if (tx < ty) *cx += vx > 0.f ? 1 : -1;
  else *cy += vy > 0.f ? 1 : -1;
}

// 벽 덮음: 빔 끝점(참 광선 o + d · 잡음 섞인 거리 r)이 떨어진 벽 면 칸 하나를 고른다. 없으면 false.
//   끝점 칸 → 아니면 광선을 거슬러 바로 앞 칸(센서 쪽) → 아니면 광선 따라 바로 뒤 칸.
// 근거: 참 광선의 맞은 점은 장면 표면 위이고, 그 표면이 지나는 칸이 막힌 칸(build_world 가 삼각형을 칸에 찍음)이다. 끝점은 그 점에서
//   광선 방향으로만 거리 잡음만큼 벗어난다 — σ = (sigma0 + sigma1·r)·물체 배율 ≈ 4~12 mm (sensor_model.yaml lidar.sigma, 실측 표면 빔 σ
//   중앙 4.7 mm · p90 12 mm)에 양자화 1 mm 라 칸(5 cm)의 반도 안 된다. 그래서 끝점 칸이 벽 면 칸이 아니면 표면 칸은 광선 위 바로 앞(길게 재서
//   벽 두께 안쪽으로 넘어감) 또는 바로 뒤(짧게 재서 바닥 칸에 떨어짐) 칸이다. 앞을 먼저 보는 까닭: 둘 다 벽 면 칸일 수 있는 건 1 칸 두께 벽
//   너머 다른 면인데, 그쪽은 이 빔이 본 면이 아니다. 두 칸 넘게 떨어진 끝점(모서리 혼합 점 — 두 면 사이 값)은 덮음으로 안 친다.
//   지도 자세 오차(OU)는 덮음에 안 들어간다(참 광선) — 덮음은 "센서가 실제로 그 벽을 맞혔나" 이고 지도 품질은 따로 본다.
DEV bool wall_credit(const EnvCfg& c, const MapDev& m, const BeamRay& ray, float r, int* ox, int* oy) {
  const float px = ray.o.x + ray.d.x * r, py = ray.o.y + ray.d.y * r;
  int cx, cy;
  if (cell_of(c, m, px, py, &cx, &cy) < 0) return false;
  const auto wall = [&](int x, int y) { return x >= 0 && y >= 0 && x < m.gw && y < m.gh && (m.grid[y * m.gw + x] & kCellWall) != 0; };
  if (wall(cx, cy)) { *ox = cx; *oy = cy; return true; }
  int bx = cx, by = cy;
  ray_next_cell(c, m, px, py, -ray.d.x, -ray.d.y, &bx, &by);
  if (wall(bx, by)) { *ox = bx; *oy = by; return true; }
  int fx = cx, fy = cy;
  ray_next_cell(c, m, px, py, ray.d.x, ray.d.y, &fx, &fy);
  if (wall(fx, fy)) { *ox = fx; *oy = fy; return true; }
  return false;
}

// 빔 j: 재고, 지도 자세로 펴서 지나간 칸·맞은 칸 표시 (Bresenham), 끝점이 떨어진 벽 면 칸 표시 (wall_credit)
// 잰 뒤 (r = lidar_beam_real, ray = 그 참 광선) — GPU 는 광선을 묶어 쏜 뒤 이것만 부른다
DEV void scan_beam_mark(const EnvBatch& b, const EnvState& e, int env, Shared& sh, int j, float r, const BeamRay& ray) {
  const EnvCfg& c = b.cfg;
  const MapDev& m = b.maps[e.map];
  if (r <= 0.f) return;   // 빠짐: Cartographer 는 min_range 아래로 버린다
  // ROS 가 적는 빔 각도(격자) + 빔 시각만큼 돈 몫 (Cartographer 가 펴 줌)
  const float t = (float)j * (e.noise.scan_time / sh.scan.n_rev);
  const float a_ros = -0.5f * e.noise.fov + (float)j * sh.inc;
  {
    uint64_t si = sh.scan.seed ^ ((uint64_t)(j + 1) * 0xD6E8FEB86659FD93ull);   // 세기 흔들림 난수 (빔마다)
    bin_beam(sh, a_ros, r, c.map_max_range, lidar_intensity(e.noise, e.world_seed, ray.obj, r, ray.cosi, false, si));
  }
  const float a = sh.mth + a_ros + sh.mw * t;
  float s, co;
  dm::sincosf_d(a, &s, &co);
  const bool hit = r <= c.map_max_range;
  const float len = hit ? r : c.missing_ray;
  int x0, y0, x1, y1;
  cell_of(c, m, sh.mx, sh.my, &x0, &y0);
  cell_of(c, m, sh.mx + co * len, sh.my + s * len, &x1, &y1);
  uint32_t* M = b.marks + (size_t)env * (c.size * c.size / 4);
  // 끝 칸을 맵 격자 안으로 (밖이면 그 앞까지)
  const int ddx = x1 > x0 ? x1 - x0 : x0 - x1, ddy = y1 > y0 ? y1 - y0 : y0 - y1;
  const int sx = x0 < x1 ? 1 : -1, sy = y0 < y1 ? 1 : -1;
  int err = ddx - ddy, x = x0, y = y0;
  const auto inside = [&](int cx, int cy) { return cx >= 0 && cy >= 0 && cx < m.gw && cy < m.gh; };
  int lox = 1 << 30, loy = 1 << 30, hix = -1, hiy = -1;
  const auto grow = [&](int cx, int cy) {
    lox = cx < lox ? cx : lox;
    loy = cy < loy ? cy : loy;
    hix = cx > hix ? cx : hix;
    hiy = cy > hiy ? cy : hiy;
  };
  while (true) {
    const bool last = x == x1 && y == y1;
    if (!inside(x, y)) break;
    mark(M, c.size, x, y, last && hit ? 2u : 1u);
    grow(x, y);
    if (last) break;
    const int e2 = 2 * err;
    if (e2 > -ddy) { err -= ddy; x += sx; }
    if (e2 < ddx) { err += ddx; y += sy; }
  }
  // 벽 덮음은 지도와 같은 거리 한도(max_range 안 = 지도에 맞음으로 들어가는 빔)만
  int wx, wy;
  if (hit && ray.obj != kObjPerson && wall_credit(c, m, ray, r, &wx, &wy)) {   // 사람을 맞은 빔은 덮음에 안 셈
    mark(M, c.size, wx, wy, 4u);
    grow(wx, wy);   // scan_apply 가 이 칸을 지나가게 (참 광선 끝점은 지도 자세 Bresenham 범위 밖일 수 있음)
  }
  if (hix >= 0) {
    atomic_min_i(&sh.bx0, lox);
    atomic_min_i(&sh.by0, loy);
    atomic_max_i(&sh.bx1, hix);
    atomic_max_i(&sh.by1, hiy);
  }
}

DEV void scan_beam(const EnvBatch& b, const EnvState& e, int env, Shared& sh, int j) {
  BeamRay ray;
  const float r = lidar_beam_real(with_people(b.maps[e.map].scene, e.people), e.noise, sh.scan, j, &ray);
  scan_beam_mark(b, e, env, sh, j, r, ray);
}

// 스레드 tid / nthreads: 표시된 칸을 한 번씩 갱신(맞음 우선)하고 이번 장 표시(bit0~2)를 지운다. 판 동안 남는 표시(bit3 본 벽 면 칸, bit4 지나간 바닥 칸)를
// 처음 켤 때 센다. 칸 낱말(4 칸)마다 스레드 하나만 만지므로 나누는 방식과 무관하게 지도·합이 같다
DEV ApplyCount scan_apply(const EnvBatch& b, const EnvState& e, int env, const Shared& sh, int tid, int nthreads) {
  const EnvCfg& c = b.cfg;
  const MapDev& m = b.maps[e.map];
  int n_wall = 0, n_floor = 0, d_known = 0, n_new = 0;
  // 연기 속: 이번 장 표시(bit0~2)는 지우되 지도·덮음은 갱신하지 않는다 (mdp/smoke.h — 라이다를 안 씀, 연기 전 지도를 그대로 둠)
  const bool upd = mdp::smoke::map_update(e.smoke);
  if (sh.bx1 >= 0) {
    const int n = c.size * c.size;
    int16_t* L = b.lmap + (size_t)env * n;
    uint32_t* M = b.marks + (size_t)env * (n / 4);
    const int w0 = sh.bx0 >> 2, w1 = sh.bx1 >> 2, nw = w1 - w0 + 1, rows = sh.by1 - sh.by0 + 1;
    for (int k = tid; k < nw * rows; k += nthreads) {
      const int cy = sh.by0 + k / nw, wx = w0 + k % nw;
      const int wi = (cy * c.size >> 2) + wx;   // size 는 4 의 배수
      const uint32_t word = M[wi];
      if ((word & 0x07070707u) == 0u) continue;
      uint32_t keep = word & 0x38383838u;   // 판 동안 남는 bit3·4·5
      for (int q = 0; q < 4; ++q) {
        const uint32_t mk = (word >> (8 * q)) & 0xffu;
        if (!(mk & 7u)) continue;
        const int cx = wx * 4 + q, ci = cy * c.size + cx;
        if (upd && (mk & 3u)) {
          const int16_t obs = (mk & 2u) ? c.l_hit : c.l_miss;
          const int16_t old = L[ci];
          if (old == kUnknown) {
            L[ci] = obs;
            ++n_new;
          } else {
            const int v = (int)old + (int)obs;
            L[ci] = (int16_t)(v < c.l_min ? c.l_min : (v > c.l_max ? c.l_max : v));
          }
          d_known += (int)(mdp::cell_class(L[ci]) == mdp::kFree) - (int)(mdp::cell_class(old) == mdp::kFree);
        }
        if (upd && (mk & 1u) && !(mk & 16u) && cx < m.gw && (m.grid[cy * m.gw + cx] & kCellFloor)) {
          keep |= 16u << (8 * q);
          ++n_floor;
        }
        if (upd && (mk & 4u) && !(mk & 8u)) {
          keep |= 8u << (8 * q);
          ++n_wall;
        }
      }
      M[wi] = keep;
    }
  }
  return ApplyCount{n_wall, n_floor, d_known, n_new};
}

// 스레드 0: 이번 장에 건드린 칸 범위를 판 동안 범위에 합침 (프론티어 요약을 이 안에서만 — observations.h)
DEV void known_box(const EnvBatch& b, EnvState& e, const Shared& sh) {
  (void)b;
  if (sh.bx1 < 0) return;
  e.kb_x0 = sh.bx0 < e.kb_x0 ? sh.bx0 : e.kb_x0;
  e.kb_y0 = sh.by0 < e.kb_y0 ? sh.by0 : e.kb_y0;
  e.kb_x1 = sh.bx1 > e.kb_x1 ? sh.bx1 : e.kb_x1;
  e.kb_y1 = sh.by1 > e.kb_y1 ? sh.by1 : e.kb_y1;
}

// ---------- 3. 보상·끝·관측 ----------

// 스레드 0. 이번 행동 동안 새로 본 벽 면 칸·새로 지나간 바닥 칸·갈 수 있음 칸 변화·새로 안 칸. 보상 항 → terms[kTerms]
DEV void finish_step(const EnvBatch& b, EnvState& e, int env, int new_wall, int new_floor, int d_known, int n_new) {
  const EnvCfg& c = b.cfg;
  const MapDev& m = b.maps[e.map];
  const float dt_act = (float)c.substeps * c.dt;
  e.seen_wall += new_wall;
  e.seen_floor += new_floor;
  const float cov = (float)e.seen_wall / (float)m.n_wall;
  mdp::rew::StepIn in;
  in.cov_gain = e.step == 0 ? 0.f : cov - e.coverage;   // 출발 자리 첫 장의 덮음은 보상 안 함 (rewards.h "제자리")
  in.dt = dt_act;
  in.t_ref = m.t_ref;
  in.collided_wall = e.collided && !e.people.hit_person;
  in.collided_person = e.people.hit_person;
  {
    const float sp = sqrtf(e.v * e.v + e.vy * e.vy);
    in.near_person = e.people.near_d < mdp::rew::kNearDist ? sp * dt_act : 0.f;   // 행동 끝 간격·속도로 (잠정)
  }
  in.declared = e.declared && !e.collided;
  in.cov = cov;
  in.slip = e.slip_acc;
  // 진행 없음 (curriculums.h kIdleTime): 새 벽 면 칸 없는 걸음 수 — 충돌·선언이 먼저.
  // 지도 새 칸(since_new)이 아니라 벽 덮음으로 셈: 제자리에서도 지도 새 칸은 가끔 생겨(reward_probe still 8 판 중 1~3 판이 1,200 s 까지 감) 버티기를 못 막음
  // 연기 속에서는 벽을 볼 수 없어(라이다 안 씀) 세지 않고 멈춰 둔다 — 연기가 걷히면 그 자리부터 다시 셈
  e.since_wall = new_wall > 0 ? 0 : (e.smoke ? e.since_wall : e.since_wall + 1);
  in.idle = !e.smoke && !e.collided && !in.declared && (float)e.since_wall * dt_act >= mdp::cur::kIdleTime;
  float* T = b.reward_terms + (size_t)env * mdp::rew::kTerms;
  e.reward = mdp::rew::terms(in, T);
  e.slip_acc = 0.f;
  e.coverage = cov;
  e.floor_cov = (float)e.seen_floor / (float)m.n_floor;
  e.step += 1;
  if (e.smoke) e.smoke_steps += 1;
  if (e.t95_step < 0 && cov >= mdp::cur::kSuccessCov) e.t95_step = e.step;
  // 지도 진행
  e.known_free += d_known;
  const int slot = e.step % kProgRing;
  e.prog_sum += n_new - e.prog[slot];
  e.prog[slot] = n_new;
  e.since_new = n_new > 0 ? 0 : e.since_new + 1;
  // EKF 속도 자리: 1 차 필터 (시간상수 0.1 s → 행동 0.1 s 마다 반씩). 잠정
  e.ekf_v = e.ekf_v + 0.5f * (e.drive.wv - e.ekf_v);
  e.ekf_w = e.ekf_w + 0.5f * (e.gyro - e.ekf_w);
  // 지나간 곳 (지도 자세, 몸통 반폭 원 — 관측 "지나간 곳" 채널). 판 동안 남는 bit5
  {
    int rx, ry;
    cell_of(c, m, e.x + e.ex, e.y + e.ey, &rx, &ry);
    uint32_t* M = b.marks + (size_t)env * (c.size * c.size / 4);
    for (int dy = -2; dy <= 2; ++dy)
      for (int dx = -2; dx <= 2; ++dx) {
        const int u = rx + dx, v = ry + dy;
        if (dx * dx + dy * dy <= 5 && u >= 0 && v >= 0 && u < m.gw && v < m.gh) mark(M, c.size, u, v, 32u);
      }
  }
  // 판 끝: 충돌 > 선언 > 진행 없음 > 끊김
  if (e.collided) e.done = e.people.hit_person ? mdp::cur::kCollisionPerson : mdp::cur::kCollision;
  else if (in.declared) e.done = cov >= mdp::cur::kSuccessCov ? mdp::cur::kDeclareOk : mdp::cur::kDeclareShort;
  else if (in.idle) e.done = mdp::cur::kNoProgress;
  else if (e.step >= c.max_steps) e.done = mdp::cur::kTruncated;
  if (e.done) {
    EpInfo& I = b.ep[env];
    I.coverage = cov;
    I.t95 = e.t95_step < 0 ? -1.f : (float)e.t95_step * dt_act;
    I.t = (float)e.step * dt_act;
    I.done = e.done;
    I.collided = e.collided;
    I.stage = e.stage;
    I.map = e.map;
    if (c.curriculum) mdp::cur::on_episode_end(&e.stage, &e.cur_hist, &e.cur_n, cov, b.max_stage);
  }
}

// 모든 스레드: 이번 장 라이다 칸을 링 다음 칸에 (lid_head 는 스레드 0 이 먼저 올림)
DEV void push_bins(const EnvBatch& b, const EnvState& e, int env, const Shared& sh, int tid) {
  float* R = b.lidar_ring + ((size_t)env * kLidarRing + e.lid_head) * mdp::obs::LIDAR_BINS;
  for (int k = tid; k < mdp::obs::LIDAR_BINS; k += kThreads)
    R[k] = sh.bins[k] == kBinNone ? b.cfg.map_max_range : (float)(sh.bins[k] >> 16) * 0.001f;
  float* I = b.lidar_int + (size_t)env * mdp::obs::LIDAR_BINS;   // 세기: 지금 장만 (관측 INTENSITY)
  for (int k = tid; k < mdp::obs::LIDAR_BINS; k += kThreads) I[k] = sh.bins[k] == kBinNone ? 0.f : (float)(sh.bins[k] & 0xffff);
}

}  // namespace sim

#include "observations.h"   // mdp/ — 관측 만들기 (EnvState·EnvBatch 를 씀)
