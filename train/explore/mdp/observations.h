// 탐사 시뮬 관측 만들기: 깊이 관측(깊이 카메라 장애물 층·충돌 감시가 읽음)과 계층형 후보 칸. 호스트·장치 공용.
// sim/include/env.h 끝에서 include 된다(EnvState·EnvBatch·Shared 를 씀). GPU 커널(env_kernel.cu)과 CPU 참조판(env_ref.cpp · obs_ref.h)이 같은 함수를 부른다.
// 2026-11-18: 그만둔 강화학습 정책 관측(가까운·먼 지도 그림, 라이다 4 장·세기 1D, 숫자 64 칸과 그 지연, 후보 특징 칸)을 지움 (archive/rl).
//
// 프론티어 덩어리 요약 = mdp/frontier.h reach_summary (기준선 frontier_search: 로봇 반경 kRobotRadius 로 막힘 팽창, 크기 문턱 kFrontMinSize) — 닫힘(EnvState::map_closed)·후보.
//   CPU 참조판은 obs_ref.h reach_obs 가, GPU 는 env_kernel.cu reach_block 이 같은 값을 블록 병렬로 낸다.
// 깊이 1 × 40 × 64 (0 = 구멍, 1~255 = min_z→max_z).
#pragma once
#include "env.h"

namespace mdp::obsgen {

// 기준선과 같은 값 (baselines/frontier/config/frontier.yaml min_frontier_size, nav2_params.yaml robot_radius — frontier_check 가 같은지 본다)
constexpr float kFrontMinSize = 0.75f, kRobotRadius = 0.2f;
DEV mdp::Robot reach_robot(const sim::EnvCfg& c) { return mdp::robot_cells(kRobotRadius, c.res); }
DEV int reach_min_cells(const sim::EnvCfg& c) { return (int)ceilf(kFrontMinSize / c.res - 1e-4f); }   // frontier_explorer.cpp make_plan 과 같은 식
// 메카넘 바퀴 4 개 (바퀴 선속도 = v ∓ vy ∓ k·ω): k = 0.200 m — 실측 (1104 녹화 /motor_states 모터 rad/s × 바퀴 반지름 0.0476 m 를
// 바퀴 v·ω 에 회귀, 섀시 펌웨어 값도 0.2). URDF 치수로 셈한 (축간 0.2 + 윤거 0.13) / 2 = 0.165 와 다름 — sim/analysis/real_limo.md 구동 절 (env_cfg.cpp wheel_k)
constexpr float kWheelK = 0.200f;

using sim::EnvBatch;
using sim::EnvState;
using sim::Shared;

// 깊이 픽셀 하나 (i = 행 · DEPTH_W + 열): 참값 자세의 카메라에서 (실물도 카메라는 실제 자리에서 본다). 픽셀끼리 독립 — CPU 참조판은 obs_depth 로
// 차례로, GPU 는 같은 단계 함수(depth_camera·depth_uv·depth_seed·sensor_model.h 의 depth_*)로 광선을 묶어 쏜다. 판 끝 리셋 뒤 상태(새 판)로 그린다
// 몸통 방향(z 축 yaw) · 카메라 숙임(y 축 pitch, 아래로 +, env.yaml body.camera_pitch 실측)
// 늦음(e.cam_lag > 0, sensor_model.yaml depth.latency): 그만큼 앞 물리 걸음의 참값 자세에서 — 실물 깊이 영상은 라이다 장 자세보다 늦게 찍힌 것이 온다
DEV sim::Pose depth_camera(const sim::EnvCfg& c, const EnvState& e) {
  const int k = e.cam_lag - 1;
  const float x = k >= 0 ? e.cam_hx[k] : e.x, y = k >= 0 ? e.cam_hy[k] : e.y, th = k >= 0 ? e.cam_hth[k] : e.th;
  float ts, tc;
  dm::sincosf_d(th, &ts, &tc);
  return sim::Pose{{tc * c.cam_cp, -ts, tc * c.cam_sp, ts * c.cam_cp, tc, ts * c.cam_sp, -c.cam_sp, 0.f, c.cam_cp},
                   sim::V3{(x + tc * c.cam_x) - ts * c.cam_y, (y + ts * c.cam_x) + tc * c.cam_y, c.cam_z}};
}
// 관측 해상도(64 × 40)에서만 그린다: 원본 640 × 400 칸을 고르게 건너뛴 가운데 광선 하나씩 (모서리 판정 광선 2 개 포함 = 픽셀당 최대 3 광선)
DEV void depth_uv(const sim::DepthModel& DM, int i, float* u, float* v) {
  using namespace mdp::obs;
  const float su = (float)DM.spec.w / (float)DEPTH_W, sv = (float)DM.spec.h / (float)DEPTH_H;
  *u = ((float)(i % DEPTH_W) + 0.5f) * su - 0.5f;
  *v = ((float)(i / DEPTH_W) + 0.5f) * sv - 0.5f;
}
DEV uint64_t depth_seed(const EnvState& e, int i) {
  return e.rng ^ ((uint64_t)(i + 1) * 0x9E3779B97F4A7C15ull) ^ ((uint64_t)(e.tick + 1) * 0xD1B54A32D192ED03ull);
}
// 깊이 → 관측 칸: 0 = 구멍, 1~255 = min_z → obs_max_z. 센서는 obs_max_z(3 m) 넘는 값도 내고(실물 정렬 켬 5 m 까지) 그건 255 로 자른다 —
// 실물 깊이도 이 함수 그대로 (시뮬·실물 같은 규칙 — RobotEnv::depth_from_image)
DEV uint8_t depth_u8(const sim::DepthModel& DM, float z) {
  if (z <= 0.f) return (uint8_t)0;
  const float zc = z < DM.spec.obs_max_z ? z : DM.spec.obs_max_z;
  return (uint8_t)(1 + (int)((zc - DM.spec.min_z) * 254.f / (DM.spec.obs_max_z - DM.spec.min_z)));
}
DEV uint8_t* depth_out(const EnvBatch& b, int env) { return b.depth + (size_t)env * mdp::obs::DEPTH_SIZE; }

DEV void obs_depth(const EnvBatch& b, const EnvState& e, int env, int i) {
  const sim::EnvCfg& c = b.cfg;
  const sim::MapDev& m = b.maps[e.map];
  float u, v;
  depth_uv(c.depth, i, &u, &v);
  const float z = sim::depth_pixel_real(sim::with_people(m.scene, e.people), c.depth, depth_camera(c, e), u, v, depth_seed(e, i), e.depth_floor);   // 사람 원기둥도 보임
  depth_out(b, env)[i] = depth_u8(c.depth, z);
}

// 깊이 카메라 층 전용 격자 픽셀 i (EnvCfg::cam_lw · cam_lh, sim::cam_uv) — 관측 깊이와 같은 카메라·모델, 다른 난수 줄 (관측 픽셀 번호 뒤)
DEV void cam_layer_depth(const EnvBatch& b, const EnvState& e, int env, int i) {
  const sim::EnvCfg& c = b.cfg;
  const sim::MapDev& m = b.maps[e.map];
  float u, v;
  sim::cam_uv(c, i, &u, &v);
  const float z = sim::depth_pixel_real(sim::with_people(m.scene, e.people), c.depth, depth_camera(c, e), u, v, depth_seed(e, mdp::obs::DEPTH_SIZE + i), e.depth_floor);
  b.cldep[(size_t)env * sim::cam_npix(c) + i] = depth_u8(c.depth, z);
}

// 계층형 후보 칸 (mdp::hier, 스레드 하나): 후보 요약 → 환경이 결정 때 읽는 후보 칸(b.cand)·지금 목표 비트(b.cand_cur). 빈 칸은 −1.
// CPU 참조판(obs_ref.h reach_obs)과 GPU env_reach_kernel 이 같은 요약(mdp::cand_pick == reach_block)으로 부른다
DEV void obs_cand(const EnvBatch& b, int env, const mdp::CandSummary& cs) {
  using namespace mdp::hier;
  int32_t* C = b.cand + (size_t)env * K;
  int cur = 0;
  for (int k = 0; k < K; ++k) {
    const mdp::CandTop& t = cs.c[k];
    C[k] = t.n > 0 ? t.cell : -1;
    if (t.n > 0 && t.cur) cur |= 1 << k;
  }
  b.cand_cur[env] = cur;
}

}  // namespace mdp::obsgen
