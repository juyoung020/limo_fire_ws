// GPU 학습 환경: 첫 지도 만들기(탐사) 한 판의 물리·센서·지도·보상 전부. 환경 하나 = 스레드 블록 하나(kThreads),
// 커널 한 번 = 행동 하나(substeps × dt, 속도 행동 0.1 s · 목표점 행동 0.5 s). GPU 커널(src/env_kernel.cu)과 CPU 참조판(src/env_ref.cpp)이 이 단계 함수들을 같이 불러
// 비트까지 같은 값을 낸다(tools/env_verify). 블록 안 합치기는 정수만(순서와 무관) — 결정적.
//
// 한 행동 (env_step):
//   0. 끝난 판이면 리셋: 맵·시작 칸·방향·라이다 오차 무작위화·지도 비움
//   1. 행동: 속도 행동이면 (v, ω) 가 그대로 /cmd_vel. 목표점 행동이면 로봇(지도 자세) 기준 방향 n_dir × 거리,
//      정답 충돌 격자로 막히기 전까지 자름 (Nav2 가 피해 가는 몫 대신)
//   2. 물리 substeps 번 (dt, limo_base 50 Hz):
//        (목표점이면 Nav2 Regulated Pure Pursuit 식 → velocity_smoother 가속 한도) → 바닥판 구동 모델(drive.h, 실물로 맞춤:
//        지연·가속·1 차 응답, 미끄러짐은 판마다 뽑은 바닥 재질) → 참값 자세 적분 → 몸통 사각형이 충돌 격자에 닿으면 되돌리고 판 끝.
//        지도 자세 오차(OU, Cartographer 가 남기는 몫)
//      lidar_every 걸음마다: 실물 오차 라이다(lidar_real.h, 스캔 중 움직임 포함) → Cartographer 와 같은 확률 지도 갱신
//        (지도 자세로, 빔 시각만큼 돌려 펴기(unwarp), max_range 넘으면 missing_ray 만큼만 빈 칸, 한 장에 칸마다 한 번, 맞음 우선)
//        + 덮음 표시: 빔 끝점(참 광선 · 잡음 섞인 거리, max_range 안)이 떨어진 벽 면 칸(아래 wall_credit), 빔이 지나간 바닥 칸
//   3. 보상(새로 덮은 벽 면 칸 비율·시간·충돌·성공)·끝 판정, 로봇 중심 관측
//
// 덮음(coverage) = 본 벽 면 칸 / 벽 면 칸 (env_host.cpp build_world: 출발점에서 갈 수 있는 바닥에 4 이웃으로 맞닿은 막힌 칸).
// 바닥 덮음(floor_cov) = 빔이 한 번이라도 지나간(지도 자세의 Bresenham, 끝 칸 빼고) 갈 수 있는 바닥 칸 비율 — 보조 지표, 보상에 안 씀.
#pragma once
#include <cstdint>

#include "drive.h"
#include "lidar_real.h"
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
  float res;
  int size;                 // 환경마다 지도 size × size 칸 (맵 격자는 이 안에)
  int16_t l_hit, l_miss, l_min, l_max;   // 로그 오즈 × 1000
  float map_max_range, missing_ray;
  // 지도 자세 오차 (OU)
  float pe_xy, pe_yaw, pe_tau;
  // 행동
  int action_mode;
  float act_v_max, act_w_max;   // 속도 행동 한도 = drive.max
  int n_dir, n_dist;
  float dist[4];
  float goal_margin;        // 목표점을 막힌 칸 이만큼 앞에서 자름
  // 보상·끝
  float r_cov, r_success, r_time, r_coll, r_slip;
  // IMU 자이로 z (관측용, 실측 analysis/real_limo.md)
  float gyro_sigma, gyro_bias;
  int max_steps;
  float success_cov;
  // 관측
  int obs_size;
  float obs_cell;
  // 깊이 카메라 (관측용 줄인 영상): 실물 모델(sensor_model.yaml depth), 장착 (base_link 기준 x, y, 바닥에서 높이)
  DepthModel depth;
  float cam_x, cam_y, cam_z;
  int depth_w, depth_h;     // 관측 영상 크기 — 원본 칸을 고르게 건너뛴 가운데 광선
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
  int gw, gh, n_spawn, n_wall, n_floor;
  float ox, oy;
};

struct EnvState {
  float x, y, th, v, w;        // 참값 자세·속도 (세계)
  float vs, ws;                // velocity_smoother 출력 (목표점 행동의 명령)
  float cmd_v, cmd_w;          // 속도 행동의 명령 (/cmd_vel)
  DriveState drive;            // 바닥판 구동 (바퀴 속도 = 오도메트리)
  Floor ground;                // 이 판 바닥 (미끄러짐·마찰)
  int32_t floor, tick;         // 바닥 번호, 물리 걸음 수
  float ex, ey, eth;           // 지도 자세 = 참값 + 오차
  float gx, gy;                // 목표점 (지도 자세 기준 세계 좌표)
  uint64_t rng, world_seed;
  int32_t map, step, seen_wall, seen_floor, done;   // 본 벽 면 칸·지나간 바닥 칸 수. done: 0 진행, 1 성공, 2 충돌, 3 시간 끝
  int32_t last_action, collided;
  float reward, coverage, floor_cov;   // coverage = seen_wall / n_wall, floor_cov = seen_floor / n_floor
  float gyro, gyro_bias;       // 자이로 측정 (몸통 ω + 이 판 바이어스 + 잡음)
  float slip_acc;              // 이번 행동 동안 |바퀴 ω − 몸통 ω| · dt 합 (헛돎 감점)
  LidarNoise noise;            // 이 판 라이다 오차 (무작위화)
};

// 블록 안 공유 (GPU __shared__, CPU 는 지역 변수)
struct Shared {
  ScanIn scan;
  int n_beams;
  float mx, my, mth, mw;       // 지도 자세의 라이다 위치·방향·각속도 (지도 넣기용)
  float inc;
  int bx0, by0, bx1, by1;      // 이번 장에 건드린 칸 범위
  int part[kThreads];          // scan_apply 스레드마다 새로 본 벽 면 칸 수
  int part_floor[kThreads];    // 새로 지나간 바닥 칸 수
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
  uint8_t* obs;                // [env · obs_size²]  0 모름, 1~255 = 로그 오즈 낮음→높음
  float* obs_num;              // [env · kObsNum]
  uint8_t* obs_depth;          // [env · depth_w · depth_h]  0 없음(범위 밖), 1~255 = min_z → max_z
  float* reward;               // [env] 이번 행동 보상
  int32_t* done;               // [env] 이번 행동으로 끝났나 (0 진행, 1 성공, 2 충돌, 3 시간 끝). 끝난 판은 바로 리셋돼 관측은 새 판 것
  int n_env;
};
constexpr int kObsNum = 7;      // 덮은 비율, 바퀴 v, 자이로 ω, 남은 시간, 직전 명령 (v, ω) 또는 직전 목표 (x, y), 헛돎(바퀴 ω − 자이로 ω)

// ---------- 작은 도구 ----------

DEV float wrapf(float a) { return dm::wrap_pi(a); }

DEV int cell_of(const EnvCfg& c, const MapDev& m, float x, float y, int* cx, int* cy) {
  *cx = (int)floorf((x - m.ox) / c.res);
  *cy = (int)floorf((y - m.oy) / c.res);
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
  e.map = (int)(dm::splitmix64(e.rng) % (uint64_t)b.n_maps);
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
  e.reward = 0.f;
  e.coverage = 0.f;
  e.floor_cov = 0.f;
}

// 모든 스레드: 지도·표시 비우기
DEV void reset_map(const EnvBatch& b, int env, int tid) {
  const int n = b.cfg.size * b.cfg.size;
  int16_t* L = b.lmap + (size_t)env * n;
  uint32_t* M = b.marks + (size_t)env * (n / 4);
  for (int i = tid; i < n; i += kThreads) L[i] = kUnknown;
  for (int i = tid; i < n / 4; i += kThreads) M[i] = 0u;
}

// ---------- 1. 행동 (스레드 0) ----------

DEV void set_goal(const EnvBatch& b, EnvState& e, int action);

// a: 환경마다 float 2 개. 속도 행동이면 (v, ω), 목표점 행동이면 a[0] = 행동 번호
DEV void apply_action(const EnvBatch& b, EnvState& e, const float* a) {
  const EnvCfg& c = b.cfg;
  if (c.action_mode == kActVelocity) {
    e.cmd_v = dm::maxf(-c.act_v_max, dm::minf(c.act_v_max, a[0]));
    e.cmd_w = dm::maxf(-c.act_w_max, dm::minf(c.act_w_max, a[1]));
    e.last_action = -1;
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
  // 바닥판 구동: 명령 → 바퀴 → 몸통 (바닥 미끄러짐)
  drive_step(c.drive, e.drive, cmd_v, cmd_w, c.dt, e.ground, &e.v, &e.w);
  // 적분 (가운데 방향)
  const float thm = e.th + 0.5f * e.w * c.dt;
  float s, co;
  dm::sincosf_d(thm, &s, &co);
  const float nx = e.x + e.v * co * c.dt, ny = e.y + e.v * s * c.dt, nth = wrapf(e.th + e.w * c.dt);
  if (body_hits(c, m, nx, ny, nth)) {
    e.v = e.w = e.vs = e.ws = 0.f;
    drive_reset(e.drive);   // 부딪히면 바퀴·몸통 모두 멈춤 (밀린 명령도 비움)
    e.collided = 1;
  } else {
    e.x = nx;
    e.y = ny;
    e.th = nth;
  }
  // 자이로 측정·헛돎 (실물 정책도 바퀴 오도메트리 ω 와 IMU 자이로를 본다)
  e.gyro = (e.w + e.gyro_bias) + c.gyro_sigma * gauss(e.rng);
  e.slip_acc = e.slip_acc + dm::absf(e.drive.ww - e.w) * c.dt;
  // 지도 자세 오차: OU (평균 0, 표준편차 pe_*, 상관 시간 pe_tau)
  const float a = c.dt / c.pe_tau, q = sqrtf(2.f * a);
  e.ex = e.ex - e.ex * a + c.pe_xy * q * gauss(e.rng);
  e.ey = e.ey - e.ey * a + c.pe_xy * q * gauss(e.rng);
  e.eth = e.eth - e.eth * a + c.pe_yaw * q * gauss(e.rng);
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
DEV void scan_beam(const EnvBatch& b, const EnvState& e, int env, Shared& sh, int j) {
  const EnvCfg& c = b.cfg;
  const MapDev& m = b.maps[e.map];
  BeamRay ray;
  const float r = lidar_beam_real(m.scene, e.noise, sh.scan, j, &ray);
  if (r <= 0.f) return;   // 빠짐: Cartographer 는 min_range 아래로 버린다
  // ROS 가 적는 빔 각도(격자) + 빔 시각만큼 돈 몫 (Cartographer 가 펴 줌)
  const float t = (float)j * (e.noise.scan_time / sh.scan.n_rev);
  const float a = sh.mth + (-0.5f * e.noise.fov + (float)j * sh.inc) + sh.mw * t;
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
  if (hit && wall_credit(c, m, ray, r, &wx, &wy)) {
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

// 모든 스레드: 표시된 칸을 한 번씩 갱신(맞음 우선)하고 이번 장 표시(bit0~2)를 지운다. 판 동안 남는 표시(bit3 본 벽 면 칸, bit4 지나간 바닥 칸)를
// 처음 켤 때 센다 — 새로 본 벽 면 칸 수를 part[tid], 새로 지나간 바닥 칸 수를 part_floor[tid] 에
DEV void scan_apply(const EnvBatch& b, const EnvState& e, int env, Shared& sh, int tid) {
  const EnvCfg& c = b.cfg;
  const MapDev& m = b.maps[e.map];
  int n_wall = 0, n_floor = 0;
  if (sh.bx1 >= 0) {
    const int n = c.size * c.size;
    int16_t* L = b.lmap + (size_t)env * n;
    uint32_t* M = b.marks + (size_t)env * (n / 4);
    const int w0 = sh.bx0 >> 2, w1 = sh.bx1 >> 2, nw = w1 - w0 + 1, rows = sh.by1 - sh.by0 + 1;
    for (int k = tid; k < nw * rows; k += kThreads) {
      const int cy = sh.by0 + k / nw, wx = w0 + k % nw;
      const int wi = (cy * c.size >> 2) + wx;   // size 는 4 의 배수
      const uint32_t word = M[wi];
      if ((word & 0x07070707u) == 0u) continue;
      uint32_t keep = word & 0x18181818u;
      for (int q = 0; q < 4; ++q) {
        const uint32_t mk = (word >> (8 * q)) & 0xffu;
        if (!(mk & 7u)) continue;
        const int cx = wx * 4 + q, ci = cy * c.size + cx;
        if (mk & 3u) {
          const int16_t obs = (mk & 2u) ? c.l_hit : c.l_miss;
          const int16_t old = L[ci];
          if (old == kUnknown) {
            L[ci] = obs;
          } else {
            const int v = (int)old + (int)obs;
            L[ci] = (int16_t)(v < c.l_min ? c.l_min : (v > c.l_max ? c.l_max : v));
          }
        }
        if ((mk & 1u) && !(mk & 16u) && cx < m.gw && (m.grid[cy * m.gw + cx] & kCellFloor)) {
          keep |= 16u << (8 * q);
          ++n_floor;
        }
        if ((mk & 4u) && !(mk & 8u)) {
          keep |= 8u << (8 * q);
          ++n_wall;
        }
      }
      M[wi] = keep;
    }
  }
  sh.part[tid] = n_wall;
  sh.part_floor[tid] = n_floor;
}

// ---------- 3. 보상·끝·관측 ----------

// 스레드 0. new_wall·new_floor: 이번 행동 동안 새로 본 벽 면 칸·새로 지나간 바닥 칸
DEV void finish_step(const EnvBatch& b, EnvState& e, int new_wall, int new_floor) {
  const EnvCfg& c = b.cfg;
  const MapDev& m = b.maps[e.map];
  e.seen_wall += new_wall;
  e.seen_floor += new_floor;
  const float cov = (float)e.seen_wall / (float)m.n_wall;
  float r = c.r_cov * (cov - e.coverage) + c.r_time * ((float)c.substeps * c.dt)   // 시간 감점은 초당
            + c.r_slip * e.slip_acc;                                                 // 헛돈 각도(rad)당
  e.slip_acc = 0.f;
  e.coverage = cov;
  e.floor_cov = (float)e.seen_floor / (float)m.n_floor;
  e.step += 1;
  if (e.collided) { r += c.r_coll; e.done = 2; }
  else if (cov >= c.success_cov) { r += c.r_success; e.done = 1; }
  else if (e.step >= c.max_steps) { e.done = 3; }
  e.reward = r;
}

// 모든 스레드: 로봇(지도 자세) 중심 obs_size² 칸, 앞 = 위쪽 행
DEV void write_obs(const EnvBatch& b, const EnvState& e, int env, int tid) {
  const EnvCfg& c = b.cfg;
  const MapDev& m = b.maps[e.map];
  const int S = c.obs_size;
  const int16_t* L = b.lmap + (size_t)env * c.size * c.size;
  uint8_t* O = b.obs + (size_t)env * S * S;
  const float px = e.x + e.ex, py = e.y + e.ey;
  float s, co;
  dm::sincosf_d(e.th + e.eth, &s, &co);
  for (int i = tid; i < S * S; i += kThreads) {
    const int row = i / S, col = i % S;
    const float fwd = ((float)(S / 2 - row) - 0.5f) * c.obs_cell, left = ((float)(S / 2 - col) - 0.5f) * c.obs_cell;
    const float wx = (px + co * fwd) - s * left, wy = (py + s * fwd) + co * left;
    int cx, cy;
    const int ci = cell_of(c, m, wx, wy, &cx, &cy);
    uint8_t v = 0;
    if (ci >= 0 && L[ci] != kUnknown) v = (uint8_t)(1 + ((int)L[ci] - c.l_min) * 254 / (c.l_max - c.l_min));
    O[i] = v;
  }
  // 깊이: 참값 자세의 카메라에서 (실물도 카메라는 실제 자리에서 본다)
  {
    const DepthModel& M = c.depth;
    const int W = c.depth_w, H = c.depth_h;
    uint8_t* Dp = b.obs_depth + (size_t)env * W * H;
    float ts, tc;
    dm::sincosf_d(e.th, &ts, &tc);
    const Pose cam{{tc, -ts, 0.f, ts, tc, 0.f, 0.f, 0.f, 1.f},
                   V3{(e.x + tc * c.cam_x) - ts * c.cam_y, (e.y + ts * c.cam_x) + tc * c.cam_y, c.cam_z}};
    const float su = (float)M.spec.w / (float)W, sv = (float)M.spec.h / (float)H;
    for (int i = tid; i < W * H; i += kThreads) {
      const float u = ((float)(i % W) + 0.5f) * su - 0.5f, v = ((float)(i / W) + 0.5f) * sv - 0.5f;
      const V3 d = rotate(cam, depth_ray(M.spec, u, v));
      const uint64_t seed = e.rng ^ ((uint64_t)(i + 1) * 0x9E3779B97F4A7C15ull) ^ ((uint64_t)(e.tick + 1) * 0xD1B54A32D192ED03ull);
      const float z = depth_sample(m.scene, M, cam.p, d, seed);
      Dp[i] = z <= 0.f ? (uint8_t)0 : (uint8_t)(1 + (int)((z - M.spec.min_z) * 254.f / (M.spec.max_z - M.spec.min_z)));
    }
  }
  if (tid == 0) {
    float* N = b.obs_num + (size_t)env * kObsNum;
    const float gx = e.gx - px, gy = e.gy - py;
    N[0] = e.coverage;
    N[1] = e.drive.wv;   // 실물 정책도 바퀴 오도메트리 v 와 자이로 ω 를 본다
    N[2] = e.gyro;
    N[3] = 1.f - (float)e.step / (float)c.max_steps;
    N[4] = c.action_mode == kActVelocity ? e.cmd_v : co * gx + s * gy;   // 직전 명령 또는 직전 목표점(로봇 기준)
    N[5] = c.action_mode == kActVelocity ? e.cmd_w : co * gy - s * gx;
    N[6] = e.drive.ww - e.gyro;   // 헛돎: 바퀴가 몸통보다 더 돈 만큼
  }
}

}  // namespace sim
