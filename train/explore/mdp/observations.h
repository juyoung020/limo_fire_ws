// 탐사 RL 관측 만들기: 시뮬 상태 → 정책 입력 (mdp/obs_layout.h 배치 그대로) + 가치 전용 참값. 호스트·장치 공용.
// sim/include/env.h 끝에서 include 된다(EnvState·EnvBatch·Shared 를 씀). GPU 커널(env_kernel.cu)과 CPU 참조판(env_ref.cpp)이
// 같은 두 단계를 부른다 — obs_cells(모든 스레드: 칸마다 따로) → obs_finish(스레드 0: 합치기·숫자). 합치기는 정수만(결정적).
//
// 정규화 상수 (관측 만들 때 곱함, 신경망은 그대로 받음)
//   kRangeNorm 8 m   : 라이다 거리·없음 = 1. 지도와 같은 최대 거리(env.yaml map.max_range = Cartographer max_range 8 m) → [0, 1]
//   kPosNorm 12.8 m  : 시작점 기준 위치·시작점 거리. 먼 지도 창(25.6 m)의 반 → 창 안이면 [−1, 1], 지도 한도(25.6 m) 안이면 |·| ≤ 2
//   kFrontNorm 25.6 m: 프론티어 덩어리까지 BFS 길 거리(설 수 있는 칸 4 이웃 걸음 × 5 cm). 지도 한 변 — 돌아가는 길이면 넘을 수 있어 자름 2
//   kSpeedNorm 1 m/s : 바퀴·EKF·바퀴 4 개 속도. 데이터시트 최고 속도 1 m/s (실측 1.11 → 1.11)
//   kRateNorm 3.14 rad/s: 자이로·EKF ω·헛돎. drive.max ω 실측 3.14 → [−1, 1] (헛돎은 순간 더 클 수 있음)
//   지난 시간 / 안전 시간(1200 s) → [0, 1]. 안 넓이 / 655.36 m²(지도 25.6²) → [0, 1].
//   프론티어 덩어리 크기: 칸 수 × 5 cm / kFrontSizeNorm 10 m (자름 1) — 문턱 0.75 m 면 0.075. 덩어리 수 / 10 (자름 1).
//   번진 곳 안 프론티어 칸 수 / 4000 (자름 1) — 집 하나 프론티어는 수백~수천 칸(5 cm).
//   새로 안 칸(최근 2 s) 넓이 / 20 m² (자름 2) — 1 m/s 로 8 m 시야가 새 곳을 쓸면 2 s 에 수십 m².
//   진행 없음 시계 / kIdleTime 60 s (자름 1) — 판 끝 진행 없음과 같은 시계 (env.h since_occ: 새 막힘 칸 ≥ kIdleOccMin 인 마지막 행동 뒤).
//   새 막힘 칸(최근 2 s) 수 / kOccGainNorm (자름 2).
//   가치 전용: 자세 오차 / (0.1 m, 0.1 m, 0.05 rad) — env.yaml pose_error σ(0.03 m, 0.017 rad)의 약 3 배 → 대개 [−1, 1].
//              남은 벽 면 칸 / 5000 (작은 집 2594·서점 3206 칸).
//              정답 벽 진척 시계 (마지막 새 벽 면 칸 뒤 시간 / kIdleTime, 자름 1) — 정답 벽이라 가치에만 (판 끝은 지도 시계).
// 프론티어 덩어리 요약 = mdp/frontier.h reach_summary (기준선 frontier_search: 로봇 반경 kRobotRadius 로 막힘 팽창, 크기 문턱 kFrontMinSize).
//   CPU 참조판은 env_ref.cpp 가 그대로 부르고, GPU 는 env_kernel.cu reach_block 이 같은 값을 블록 병렬로 — 둘 다 sh.reach 에 넣는다.
//
// 2D (uint8)
//   가까운 지도 6 × 64 × 64, 칸 10 cm, 로봇(지도 자세) 중심·로봇 방향, 앞 = 위 행: 0 확률(0 모름, 1~255 로그 오즈 낮음→높음),
//     1 지나간 곳(255), 2~5 물체 4 (📏 0). 칸 가운데 점의 지도 칸(5 cm) 하나를 읽는다.
//   먼 지도 7 × 32 × 32, 칸 0.8 m, 시작 자세 기준(시작점 가운데, 시작 방향 = 위 행) 25.6 m 창:
//     0 확률(칸 안 8 × 8 점(0.1 m) 중 아는 점의 최댓값 — 벽이 칸 평균에 묻히지 않게), 1 지나간 곳(아무 점이라도), 2 지금 위치(255), 3~6 물체 (📏 0).
//     실물: Cartographer map 틀 = 첫 자세 → 같은 창.
//   깊이 1 × 40 × 64 (env.h depth: 0 = 구멍, 1~255 = min_z→max_z).
// 1D (float): 라이다 4 장(지금 → 직전 3 장, 지연 반영) · 세기(지금 장, 칸마다 가장 가까운 빔 — 지연은 아직 안 넣음) · 숫자 VEC (obs_layout.h 오프셋)
// 관측 지연: 판마다 e.delay 걸음(0 또는 1, reset_state) — 라이다 장과 숫자(직전 행동·지난 시간 빼고)를 그만큼 늦은 것으로.
//   지도·깊이는 지연 안 함(지도는 실물도 쌓는 데 시간이 걸려 늦지만 그 몫은 📏).
#pragma once
#include "env.h"

namespace mdp::obsgen {

constexpr float kRangeNorm = 8.f, kPosNorm = 12.8f, kFrontNorm = 25.6f, kSpeedNorm = 1.f, kRateNorm = 3.14f;
constexpr float kAreaNorm = 655.36f, kFrontCountNorm = 4000.f, kProgAreaNorm = 20.f;
constexpr float kFrontSizeNorm = 10.f, kFrontNumNorm = 10.f, kOccGainNorm = 100.f;
// 기준선과 같은 값 (baselines/frontier/config/frontier.yaml min_frontier_size, nav2_params.yaml robot_radius — frontier_check 가 같은지 본다)
constexpr float kFrontMinSize = 0.75f, kRobotRadius = 0.2f;
DEV mdp::Robot reach_robot(const sim::EnvCfg& c) { return mdp::robot_cells(kRobotRadius, c.res); }
DEV int reach_min_cells(const sim::EnvCfg& c) { return (int)ceilf(kFrontMinSize / c.res - 1e-4f); }   // frontier_explorer.cpp make_plan 과 같은 식
constexpr float kErrXyNorm = 0.1f, kErrThNorm = 0.05f, kRemainNorm = 5000.f;
// IMU·세기 (S1 실측 맞춤, sensor_model.yaml imu·lidar.intensity)
//   가속도 / 10 m/s²: 몸통 가속 ≲ 5 m/s²(구동 가속 한도) + 주행 진동 σ ≤ 1.3 m/s² → 대개 ±0.6. az 는 중력 9.81 을 빼고 (실측 g 10.03 이라 +0.02 치우침)
//   roll·pitch / 0.1 rad: 실측 바닥 기울기 σ 0.0025, 주행 몫 ≲ 0.03 rad → 대개 ±0.3
//   자이로 x·y / 0.2 rad/s: 주행 진동 σ 0.01~0.06 rad/s → 대개 ±0.3
//   세기 / 100: 실측 평균 15~53 (0.75 m 에서 가장 밝음), 물체마다 σ 6 → [0, 1] 안
constexpr float kAccNorm = 10.f, kGravity = 9.81f, kTiltNorm = 0.1f, kGyroXyNorm = 0.2f, kIntensityNorm = 100.f;
constexpr float kNearCell = 0.10f, kFarCell = 0.8f;
static_assert(kFarCell == sim::kFarCellM, "먼 지도 칸 크기 (env.h far_mark_cell)");
constexpr int kFarSub = 8;   // 먼 지도 칸 안 점 8 × 8 (0.1 m)
// 메카넘 바퀴 4 개 (바퀴 선속도 = v ∓ vy ∓ k·ω): k = 0.200 m — 실측 (1104 녹화 /motor_states 모터 rad/s × 바퀴 반지름 0.0476 m 를
// 바퀴 v·ω 에 회귀, 섀시 펌웨어 값도 0.2). URDF 치수로 셈한 (축간 0.2 + 윤거 0.13) / 2 = 0.165 와 다름 — sim/analysis/real_limo.md 구동 절
constexpr float kWheelK = 0.200f;

using sim::EnvBatch;
using sim::EnvState;
using sim::Shared;

DEV uint8_t prob_u8(const sim::EnvCfg& c, int16_t l) {
  return l == sim::kUnknown ? (uint8_t)0 : (uint8_t)(1 + ((int)l - c.l_min) * 254 / (c.l_max - c.l_min));
}
// 프론티어 칸인가 (frontier.h frontier_cell 과 같은 값): GPU 는 지도 갱신 때 고쳐 둔 프론티어 비트 지도(EnvBatch::fmap, env_kernel.cu frontier_fix)를 읽고,
// CPU 참조판(fmap 없음)은 칸과 4 이웃을 직접 본다 — env_verify 가 관측 GPU == CPU 로 같은 값임을 확인
DEV bool frontier_at(const uint32_t* FM, int ws, const mdp::MapView& mv, int x, int y) {
  if (FM) return (FM[y * ws + (x >> 5)] >> (x & 31)) & 1u;
  return mdp::frontier_cell(mv, x, y);
}
DEV bool visited(const uint32_t* M, int ci) { return ((M[ci >> 2] >> (8 * (ci & 3))) & 32u) != 0u; }

// 모든 스레드: 2D 관측·라이다 장 복사 (프론티어 요약 sh.reach 는 부르는 쪽이 — 위 머리말)
// 먼 지도 확률·지나간 곳 채널을 다시 그릴 곳: all 이면 전부, 아니면 칸 가운데가 세계 상자 [x0, x1] × [y0, y1] 안인 칸만
// (나머지 칸은 지난 걸음 값 그대로 — 그 칸이 읽는 지도 칸이 안 바뀌었으므로 같은 값. GPU 가 쓴다, CPU 참조판은 늘 전부)
struct FarDirty {
  int all;
  float x0, y0, x1, y1;          // 지나간 곳 표시 상자 (세계)
  const uint32_t* mask;          // 로그 오즈가 바뀐 먼 칸 (비트, nullptr = 없음)
};

// 먼 지도 다시 그릴 곳 (GPU 마무리 커널·CPU 참조판이 같은 규칙으로) — g 는 환경마다 걸음을 넘어 남는 칸(EnvBatch::scratch)
DEV void far_dirty_clear(Shared& g) {   // 행동 시작
  g.dx0 = g.dy0 = 1 << 30;
  g.dx1 = g.dy1 = -1;
  for (int k = 0; k < mdp::obs::FAR_H * mdp::obs::FAR_W / 32; ++k) g.far_mask[k] = 0u;
}
DEV void far_dirty_add(Shared& g, int x0, int y0, int x1, int y1) {   // 바뀐 지도 칸 범위 (장 갱신 상자 · 지나간 곳 표시)
  g.dx0 = x0 < g.dx0 ? x0 : g.dx0;
  g.dy0 = y0 < g.dy0 ? y0 : g.dy0;
  g.dx1 = x1 > g.dx1 ? x1 : g.dx1;
  g.dy1 = y1 > g.dy1 ? y1 : g.dy1;
}
DEV void far_dirty_robot(const EnvBatch& b, const EnvState& e, Shared& g) {   // finish_step 의 지나간 곳 표시 (로봇 칸 ±2)
  int rx, ry;
  sim::cell_of(b.cfg, b.maps[e.map], e.x + e.ex, e.y + e.ey, &rx, &ry);
  far_dirty_add(g, rx - 2, ry - 2, rx + 2, ry + 2);
}
// 관측 직전 (리셋 뒤 상태 e): 판이 바뀌었거나(리셋·처음·밖에서 상태를 바꿈) 아니면 바뀐 칸 범위 둘레 0.6 m 만
DEV FarDirty far_dirty_take(const EnvBatch& b, const EnvState& e, Shared& g, bool was_done) {
  const bool same = g.far_ok && g.far_map == e.map && g.far_key[0] == e.x0 && g.far_key[1] == e.y0 && g.far_key[2] == e.th0;
  const sim::MapDev& m = b.maps[e.map];
  const float res = b.cfg.res, pad = 0.6f;
  FarDirty fd;
  fd.all = (was_done || !same) ? 1 : 0;
  fd.x0 = (m.ox + (float)g.dx0 * res) - pad;
  fd.y0 = (m.oy + (float)g.dy0 * res) - pad;
  fd.x1 = (m.ox + (float)(g.dx1 + 1) * res) + pad;
  fd.y1 = (m.oy + (float)(g.dy1 + 1) * res) + pad;
  if (g.dx1 < g.dx0) fd.x1 = fd.x0 - 1.f;   // 바뀐 칸 없음
  fd.mask = g.far_mask;
  g.far_ok = 1;
  g.far_map = e.map;
  g.far_key[0] = e.x0;
  g.far_key[1] = e.y0;
  g.far_key[2] = e.th0;
  return fd;
}

// 먼 지도 칸 (시작 자세 기준 0.8 m, 점 8 × 8). 칸마다 따로라 CPU 는 차례로, GPU 는 다시 그릴 칸만 모아 점 줄(kr)마다 나눠 부른다
struct FarFrame {
  float s0, c0, half;
  int prow, pcol;   // 지금 위치 칸
};
DEV void far_frame(const EnvState& e, float px, float py, FarFrame* f) {
  using namespace mdp::obs;
  dm::sincosf_d(e.th0, &f->s0, &f->c0);
  const float dx = px - e.x0, dy = py - e.y0;
  const float rf = f->c0 * dx + f->s0 * dy, rl = -f->s0 * dx + f->c0 * dy;
  f->half = 0.5f * (float)FAR_H * kFarCell;
  f->prow = (int)floorf((f->half - rf) / kFarCell);
  f->pcol = (int)floorf((f->half - rl) / kFarCell);
}
// 칸 i 의 지금 위치 채널(과 statics 면 물체 📏 0 채널)을 쓰고, 확률·지나간 곳을 다시 그려야 하면 true
DEV bool far_cell_head(const EnvState& e, const FarDirty& fd, const FarFrame& f, uint8_t* F, int i, bool statics = true) {
  using namespace mdp::obs;
  const int row = i / FAR_W, col = i % FAR_W;
  F[2 * FAR_H * FAR_W + i] = (row == f.prow && col == f.pcol) ? (uint8_t)255 : (uint8_t)0;
  if (statics)
    for (int ch = 3; ch < FAR_FRONTIER_CH; ++ch) F[ch * FAR_H * FAR_W + i] = 0;   // 물체 📏 (마지막 칸은 프론티어 — far_cell_put)
  if (fd.all) return true;
  // 칸 가운데 (점 64 개는 가운데에서 앞·옆 0.35 m 안 → 반지름 0.5 m 안, fd 상자는 0.6 m 넓혀 둠)
  const float fwd_c = f.half - ((float)row + 0.5f) * kFarCell, left_c = f.half - ((float)col + 0.5f) * kFarCell;
  const float wcx = (e.x0 + f.c0 * fwd_c) - f.s0 * left_c, wcy = (e.y0 + f.s0 * fwd_c) + f.c0 * left_c;
  const bool marked = fd.mask && ((fd.mask[i >> 5] >> (i & 31)) & 1u);
  return marked || (wcx >= fd.x0 && wcx <= fd.x1 && wcy >= fd.y0 && wcy <= fd.y1);
}
// 칸 i 의 점 줄 kr (8 점): 아는 점의 로그 오즈 최댓값·지나간 곳을 lmax·vis 에 합침.
// 확률 칸 값 = 아는 점의 prob_u8 최댓값. prob_u8 은 로그 오즈에 단조(모름 = kUnknown 이 가장 작음)라 로그 오즈 최댓값 하나만 바꾼다
DEV void far_cell_row(const sim::EnvCfg& c, const sim::MapDev& m, const int16_t* L, const uint32_t* M, const uint32_t* FM, const EnvState& e,
                      const FarFrame& f, int i, int kr, int* lmax, int* vis, int* fr) {
  const mdp::MapView mv{L, c.size, m.gw, m.gh};
  const int ws = sim::fmap_row(c);
  using namespace mdp::obs;
  const int row = i / FAR_W, col = i % FAR_W;
  const float fwd = f.half - ((float)row + ((float)kr + 0.5f) / (float)kFarSub) * kFarCell;
  const float ax = e.x0 + f.c0 * fwd, ay = e.y0 + f.s0 * fwd;
  for (int kc = 0; kc < kFarSub; ++kc) {
    const float left = f.half - ((float)col + ((float)kc + 0.5f) / (float)kFarSub) * kFarCell;
    const float wx = ax - f.s0 * left, wy = ay + f.c0 * left;
    int cx, cy;
    const int ci = sim::cell_of(c, m, wx, wy, &cx, &cy);
    if (ci < 0) continue;
    const int l = L[ci];
    *lmax = l > *lmax ? l : *lmax;
    *vis |= visited(M, ci) ? 1 : 0;
    *fr |= frontier_at(FM, ws, mv, cx, cy) ? 1 : 0;
  }
}
DEV void far_cell_put(const sim::EnvCfg& c, uint8_t* F, int i, int lmax, int vis, int fr) {
  using namespace mdp::obs;
  F[i] = prob_u8(c, (int16_t)lmax);
  F[FAR_H * FAR_W + i] = vis ? (uint8_t)255 : (uint8_t)0;
  F[FAR_FRONTIER_CH * FAR_H * FAR_W + i] = fr ? (uint8_t)255 : (uint8_t)0;
}

// whole = false: 먼 지도는 부르는 쪽이 (GPU env_kernel.cu obs_far_block — 같은 값).
// statics = false: 늘 0 인 물체 채널(📏)을 안 씀 — GPU 는 판이 시작할 때(리셋)만 쓰고 그 뒤 걸음은 그대로 둔다 (관측 버퍼는 이 함수만 씀, 같은 값)
DEV void obs_cells(const EnvBatch& b, const EnvState& e, int env, Shared& sh, int tid, const FarDirty& fd, bool whole = true, bool statics = true) {
  using namespace mdp::obs;
  const sim::EnvCfg& c = b.cfg;
  const sim::MapDev& m = b.maps[e.map];
  const int16_t* L = b.lmap + (size_t)env * c.size * c.size;
  const uint32_t* M = b.marks + (size_t)env * (c.size * c.size / 4);
  uint8_t* I = b.obs_img + (size_t)env * IMG_SIZE;
  const float px = e.x + e.ex, py = e.y + e.ey;
  float s, co;
  dm::sincosf_d(e.th + e.eth, &s, &co);
  // 가까운 지도
  for (int i = tid; i < NEAR_H * NEAR_W; i += sim::kThreads) {
    const int row = i / NEAR_W, col = i % NEAR_W;
    const float fwd = ((float)(NEAR_H / 2 - row) - 0.5f) * kNearCell, left = ((float)(NEAR_W / 2 - col) - 0.5f) * kNearCell;
    const float wx = (px + co * fwd) - s * left, wy = (py + s * fwd) + co * left;
    int cx, cy;
    const int ci = sim::cell_of(c, m, wx, wy, &cx, &cy);
    I[i] = ci >= 0 ? prob_u8(c, L[ci]) : (uint8_t)0;
    I[NEAR_H * NEAR_W + i] = ci >= 0 && visited(M, ci) ? (uint8_t)255 : (uint8_t)0;
    if (statics)
      for (int ch = 2; ch < NEAR_FRONTIER_CH; ++ch) I[ch * NEAR_H * NEAR_W + i] = 0;   // 물체 📏
    // 프론티어 채널: 칸(10 cm) 안 2 × 2 점(지도 5 cm 칸 간격) 중 하나라도 프론티어 칸이면 255
    {
      const mdp::MapView mv{L, c.size, m.gw, m.gh};
      const uint32_t* FM = b.fmap ? b.fmap + (size_t)env * c.size * sim::fmap_row(c) : nullptr;
      const int ws = sim::fmap_row(c);
      int fr = 0;
      for (int q = 0; q < 4; ++q) {
        const float f2 = fwd + ((q & 1) ? -0.25f : 0.25f) * kNearCell, l2 = left + ((q & 2) ? -0.25f : 0.25f) * kNearCell;
        const float qx = (px + co * f2) - s * l2, qy = (py + s * f2) + co * l2;
        int qx_c, qy_c;
        if (sim::cell_of(c, m, qx, qy, &qx_c, &qy_c) >= 0) fr |= frontier_at(FM, ws, mv, qx_c, qy_c) ? 1 : 0;
      }
      I[NEAR_FRONTIER_CH * NEAR_H * NEAR_W + i] = fr ? (uint8_t)255 : (uint8_t)0;
    }
  }
  // 먼 지도 (시작 자세 기준). GPU 는 다시 그릴 칸만 모아 칸마다 8 스레드(점 줄)가 나눠 한다 (env_kernel.cu obs_far_block — 같은 함수, 최댓값·OR 이라 순서 무관)
  if (whole) {
    uint8_t* F = I + NEAR_SIZE;
    FarFrame ff;
    far_frame(e, px, py, &ff);
    for (int i = tid; i < FAR_H * FAR_W; i += sim::kThreads) {
      if (!far_cell_head(e, fd, ff, F, i)) continue;
      int lmax = sim::kUnknown, vis = 0, fr = 0;
      const uint32_t* FM = b.fmap ? b.fmap + (size_t)env * c.size * sim::fmap_row(c) : nullptr;
      for (int kr = 0; kr < kFarSub; ++kr) far_cell_row(c, m, L, M, FM, e, ff, i, kr, &lmax, &vis, &fr);
      far_cell_put(c, F, i, lmax, vis, fr);
    }
  }
  // 깊이는 픽셀마다 따로 (obs_depth — GPU 는 픽셀 하나 = 스레드 하나인 커널)
  // 라이다 4 장 (지연 반영) + 세기
  {
    float* O = b.obs_1d + (size_t)env * b.o1d;   // 한 환경 칸 수 o1d (계층형이면 뒤에 후보 칸)
    const float* R = b.lidar_ring + (size_t)env * sim::kLidarRing * LIDAR_BINS;
    const float* IT = b.lidar_int + (size_t)env * LIDAR_BINS;
    for (int i = tid; i < LIDAR + INTENSITY; i += sim::kThreads) {
      if (i >= LIDAR) { O[i] = IT[i - LIDAR] / kIntensityNorm; continue; }   // 세기: 지금 장 (지연 안 함 — 아래 주석)
      const int f = i / LIDAR_BINS, k = i % LIDAR_BINS;
      const int slot = ((e.lid_head - e.delay - f) % sim::kLidarRing + sim::kLidarRing) % sim::kLidarRing;
      O[i] = R[slot * LIDAR_BINS + k] / kRangeNorm;
    }
  }
}

// 깊이 픽셀 하나 (i = 행 · DEPTH_W + 열): 참값 자세의 카메라에서 (실물도 카메라는 실제 자리에서 본다). 픽셀끼리 독립 — CPU 참조판은 obs_depth 로
// 차례로, GPU 는 같은 단계 함수(depth_camera·depth_uv·depth_seed·sensor_model.h 의 depth_*)로 광선을 묶어 쏜다. 판 끝 리셋 뒤 상태(새 판)로 그린다
// 몸통 방향(z 축 yaw) · 카메라 숙임(y 축 pitch, 아래로 +, env.yaml body.camera_pitch 실측)
DEV sim::Pose depth_camera(const sim::EnvCfg& c, const EnvState& e) {
  float ts, tc;
  dm::sincosf_d(e.th, &ts, &tc);
  return sim::Pose{{tc * c.cam_cp, -ts, tc * c.cam_sp, ts * c.cam_cp, tc, ts * c.cam_sp, -c.cam_sp, 0.f, c.cam_cp},
                   sim::V3{(e.x + tc * c.cam_x) - ts * c.cam_y, (e.y + ts * c.cam_x) + tc * c.cam_y, c.cam_z}};
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
// 실물 정책 입력도 이 함수 그대로 (시뮬·실물 같은 규칙)
DEV uint8_t depth_u8(const sim::DepthModel& DM, float z) {
  if (z <= 0.f) return (uint8_t)0;
  const float zc = z < DM.spec.obs_max_z ? z : DM.spec.obs_max_z;
  return (uint8_t)(1 + (int)((zc - DM.spec.min_z) * 254.f / (DM.spec.obs_max_z - DM.spec.min_z)));
}
DEV uint8_t* depth_out(const EnvBatch& b, int env) { return b.obs_img + (size_t)env * mdp::obs::IMG_SIZE + mdp::obs::NEAR_SIZE + mdp::obs::FAR_SIZE; }

DEV void obs_depth(const EnvBatch& b, const EnvState& e, int env, int i) {
  const sim::EnvCfg& c = b.cfg;
  const sim::MapDev& m = b.maps[e.map];
  float u, v;
  depth_uv(c.depth, i, &u, &v);
  const float z = sim::depth_pixel_real(sim::with_people(m.scene, e.people), c.depth, depth_camera(c, e), u, v, depth_seed(e, i), e.depth_floor);   // 사람 원기둥도 보임
  depth_out(b, env)[i] = depth_u8(c.depth, z);
}

// 프론티어 덩어리 칸 (mdp::ReachSummary → V): BFS 거리 짧은 순 3 개 — 거리, 그 칸(덩어리에서 BFS 거리 가장 짧은 칸) 방향 sin·cos, 크기. 없으면 0.
// px, py, s, co = 지도 자세 위치·sin·cos. 이 칸들만 따로 쓰는 GPU 커널(env_kernel.cu env_reach_kernel)과 obs_finish 가 같이 부른다
DEV bool reach_slot(int i) {
  using namespace mdp::obs;
  return (i >= FRONTIER && i < FRONTIER + 4) || (i >= FRONTIER2 && i < FRONTIER3 + 4) || i == FRONT_N || i == CLOSED || i == FRONT_CELLS;
}
DEV void reach_vec(const sim::EnvCfg& c, const sim::MapDev& m, float px, float py, float s, float co, const mdp::ReachSummary& R, float* V) {
  using namespace mdp::obs;
  const int slot[mdp::kReachTop] = {FRONTIER, FRONTIER2, FRONTIER3};
  for (int k = 0; k < mdp::kReachTop; ++k) {
    const mdp::ReachTop& t = R.top[k];
    if (t.n <= 0) continue;
    const float fx = m.ox + ((float)(t.cell % m.gw) + 0.5f) * c.res, fy = m.oy + ((float)(t.cell / m.gw) + 0.5f) * c.res;
    const float ddx = fx - px, ddy = fy - py;
    const float d = sqrtf(ddx * ddx + ddy * ddy);
    V[slot[k] + 0] = dm::minf(2.f, (float)t.dist * c.res / kFrontNorm);
    if (d > 1e-6f) {
      V[slot[k] + 1] = dm::clampf((co * ddy - s * ddx) / d, -1.f, 1.f);   // 로봇 기준 sin
      V[slot[k] + 2] = dm::clampf((co * ddx + s * ddy) / d, -1.f, 1.f);   // cos
    }
    V[slot[k] + 3] = dm::minf(1.f, (float)t.n * c.res / kFrontSizeNorm);
  }
  V[FRONT_N] = dm::minf(1.f, (float)R.n_found / kFrontNumNorm);
  V[CLOSED] = R.closed ? 1.f : 0.f;
  V[FRONT_CELLS] = dm::minf(1.f, (float)R.n_cells / kFrontCountNorm);
}

// 스레드 0: 숫자 관측(지연 반영)·가치 전용 참값
// with_reach = false: 프론티어 덩어리 칸(reach_slot)은 안 쓴다 — GPU 는 env_reach_kernel 이 obs_reach 로 따로 (같은 값·같은 지연 규칙)
DEV void obs_finish(const EnvBatch& b, EnvState& e, int env, Shared& sh, bool with_reach = true) {
  using namespace mdp::obs;
  const sim::EnvCfg& c = b.cfg;
  const sim::MapDev& m = b.maps[e.map];
  float* V = sh.vec;
  for (int i = 0; i < VEC; ++i) V[i] = 0.f;
  const float px = e.x + e.ex, py = e.y + e.ey, pth = e.th + e.eth;
  float s, co, s0, c0;
  dm::sincosf_d(pth, &s, &co);
  dm::sincosf_d(e.th0, &s0, &c0);
  const float dx = px - e.x0, dy = py - e.y0;
  V[POSE + 0] = (c0 * dx + s0 * dy) / kPosNorm;
  V[POSE + 1] = (-s0 * dx + c0 * dy) / kPosNorm;
  {
    float a, b2;
    dm::sincosf_d(pth - e.th0, &a, &b2);
    V[POSE + 2] = a;
    V[POSE + 3] = b2;
  }
  // MAP_VEL 3 칸 📏 0
  V[WHEEL_V] = e.drive.wv / kSpeedNorm;
  V[GYRO_Z] = e.gyro / kRateNorm;
  V[SLIP] = (e.drive.ww - e.gyro) / kRateNorm;
  V[EKF + 0] = e.ekf_v / kSpeedNorm;
  V[EKF + 1] = e.ekf_w / kRateNorm;
  // IMU (imu.h, 실측 맞춤 ✅ — imu_check): e.imu.out = ax, ay, az, gx, gy, roll, pitch
  V[ACCEL + 0] = e.imu.out[0] / kAccNorm;
  V[ACCEL + 1] = e.imu.out[1] / kAccNorm;
  V[ACCEL + 2] = (e.imu.out[2] - kGravity) / kAccNorm;
  V[IMU_TILT + 0] = e.imu.out[5] / kTiltNorm;
  V[IMU_TILT + 1] = e.imu.out[6] / kTiltNorm;
  V[IMU_TILT + 2] = e.imu.out[3] / kGyroXyNorm;
  V[IMU_TILT + 3] = e.imu.out[4] / kGyroXyNorm;
  {
    const float v = e.drive.wv, vy = e.drive.wvy, w = kWheelK * e.drive.ww;
    V[WHEELS4 + 0] = ((v - vy) - w) / kSpeedNorm;   // 앞 왼
    V[WHEELS4 + 1] = ((v + vy) + w) / kSpeedNorm;   // 앞 오른
    V[WHEELS4 + 2] = ((v + vy) - w) / kSpeedNorm;   // 뒤 왼
    V[WHEELS4 + 3] = ((v - vy) + w) / kSpeedNorm;   // 뒤 오른
  }
  const float dt_act = (float)c.substeps * c.dt;
  V[KNOWN_AREA] = (float)e.known_free * (c.res * c.res) / kAreaNorm;
  if (with_reach) reach_vec(c, m, px, py, s, co, sh.reach, V);
  V[PROGRESS + 0] = dm::minf(2.f, (float)e.prog_sum * (c.res * c.res) / kProgAreaNorm);
  V[PROGRESS + 1] = dm::minf(1.f, (float)e.since_occ * dt_act / mdp::cur::kIdleTime);   // 진행 없음 시계 = 판 끝 시계 (env.h map_progress)
  V[OCC_GAIN] = dm::minf(2.f, (float)e.occ_sum / kOccGainNorm);
  {
    const float hx = e.x0 - px, hy = e.y0 - py;
    const float d = sqrtf(hx * hx + hy * hy);
    V[HOME + 0] = d / kPosNorm;
    if (d > 1e-6f) {
      V[HOME + 1] = dm::clampf((co * hy - s * hx) / d, -1.f, 1.f);
      V[HOME + 2] = dm::clampf((co * hx + s * hy) / d, -1.f, 1.f);
    }
  }
  // 지연: 직전 걸음 숫자 (판 첫 걸음은 지금 것). 직전 행동·지난 시간은 정책·시계가 아는 값이라 지연 안 함
  float* O = b.obs_1d + (size_t)env * b.o1d + LIDAR + INTENSITY;
  float* P = b.vec_prev + (size_t)env * VEC;
  const bool late = e.delay > 0 && e.step > 0;
  for (int i = 0; i < VEC; ++i) {
    if (!with_reach && reach_slot(i)) continue;
    const float now = V[i];
    O[i] = late ? P[i] : now;
    P[i] = now;
  }
  for (int i = 0; i < mdp::act::DIM * 4; ++i) O[LAST_ACT + i] = e.last_act[i];
  O[ELAPSED] = (float)e.step * dt_act / mdp::cur::kSafeTime;
  // 가치 전용 참값
  float* Q = b.priv + (size_t)env * PRIV;
  for (int i = 0; i < PRIV; ++i) Q[i] = 0.f;
  {
    const float tx = e.x - e.x0, ty = e.y - e.y0;
    Q[PRIV_TRUE_POSE + 0] = (c0 * tx + s0 * ty) / kPosNorm;
    Q[PRIV_TRUE_POSE + 1] = (-s0 * tx + c0 * ty) / kPosNorm;
    float a, b2;
    dm::sincosf_d(e.th - e.th0, &a, &b2);
    Q[PRIV_TRUE_POSE + 2] = a;
    Q[PRIV_TRUE_POSE + 3] = b2;
    Q[PRIV_POSE_ERR + 0] = e.ex / kErrXyNorm;
    Q[PRIV_POSE_ERR + 1] = e.ey / kErrXyNorm;
    Q[PRIV_POSE_ERR + 2] = e.eth / kErrThNorm;
    Q[PRIV_COVERAGE] = e.coverage;
    Q[PRIV_REMAIN] = (float)(m.n_wall - e.seen_wall) / kRemainNorm;
    // PRIV_RESERVED: 몸통~가장 가까운 사람 간격 / 5 m (자름 1, 사람 없으면 1, 겹치면 음수 그대로) — people.h near_d
    Q[PRIV_RESERVED] = dm::minf(1.f, e.people.near_d / 5.f);
    Q[PRIV_IDLE] = dm::minf(1.f, (float)e.since_wall * dt_act / mdp::cur::kIdleTime);   // 정답 벽 진척 시계 (가치에만, 판 끝엔 안 씀)
  }
}

// 프론티어 덩어리 칸만 (GPU env_reach_kernel): obs_finish(with_reach = false) 뒤 같은 상태로 — 지연 규칙도 obs_finish 와 같다
DEV void obs_reach(const EnvBatch& b, const EnvState& e, int env, const mdp::ReachSummary& R) {
  using namespace mdp::obs;
  const sim::MapDev& m = b.maps[e.map];
  float V[VEC];
  for (int i = 0; i < VEC; ++i) V[i] = 0.f;
  float s, co;
  dm::sincosf_d(e.th + e.eth, &s, &co);
  reach_vec(b.cfg, m, e.x + e.ex, e.y + e.ey, s, co, R, V);
  float* O = b.obs_1d + (size_t)env * b.o1d + LIDAR + INTENSITY;
  float* P = b.vec_prev + (size_t)env * VEC;
  const bool late = e.delay > 0 && e.step > 0;
  for (int i = 0; i < VEC; ++i) {
    if (!reach_slot(i)) continue;
    const float now = V[i];
    O[i] = late ? P[i] : now;
    P[i] = now;
  }
}

// 계층형 후보 칸 (mdp::hier, 스레드 하나): 후보 요약 → 1D 뒤 CAND_1D 칸 + 환경이 결정 때 읽는 후보 칸(b.cand)·지금 목표 비트(b.cand_cur).
// 지연 없음 — 지도에서 나온 값이고(지도·깊이처럼), 정책이 본 후보와 환경이 결정 때 쓰는 후보 칸이 같아야 해서. 빈 칸은 7 칸 모두 0.
// CPU 참조판(env_ref.cpp)과 GPU env_reach_kernel 이 같은 요약(mdp::cand_pick == reach_block)으로 부른다
DEV void obs_cand(const EnvBatch& b, const EnvState& e, int env, const mdp::CandSummary& cs) {
  using namespace mdp::hier;
  const sim::EnvCfg& c = b.cfg;
  const sim::MapDev& m = b.maps[e.map];
  float* O = b.obs_1d + (size_t)env * b.o1d + mdp::obs::POLICY_1D;
  int32_t* C = b.cand + (size_t)env * K;
  const float px = e.x + e.ex, py = e.y + e.ey;
  float s, co;
  dm::sincosf_d(e.th + e.eth, &s, &co);
  int cur = 0;
  for (int k = 0; k < K; ++k) {
    float* F = O + k * FEAT;
    for (int q = 0; q < FEAT; ++q) F[q] = 0.f;
    const mdp::CandTop& t = cs.c[k];
    C[k] = t.n > 0 ? t.cell : -1;
    if (t.n <= 0) continue;
    const float fx = m.ox + ((float)(t.cell % m.gw) + 0.5f) * c.res, fy = m.oy + ((float)(t.cell / m.gw) + 0.5f) * c.res;
    const float ddx = fx - px, ddy = fy - py;
    const float d = sqrtf(ddx * ddx + ddy * ddy);
    F[F_DIST] = dm::minf(2.f, (float)t.dist * c.res / kFrontNorm);
    if (d > 1e-6f) {
      F[F_SIN] = dm::clampf((co * ddy - s * ddx) / d, -1.f, 1.f);
      F[F_COS] = dm::clampf((co * ddx + s * ddy) / d, -1.f, 1.f);
    }
    F[F_SIZE] = dm::minf(1.f, (float)t.n * c.res / kFrontSizeNorm);
    F[F_UTIL] = dm::minf(1.f, (float)t.util / (c.cand_v2 ? (float)mdp::cand_gain_norm(mdp::cand_gain_cells(c.map_max_range, c.res)) : kUtilNorm));
    F[F_CUR] = t.cur ? 1.f : 0.f;
    F[F_VALID] = 1.f;
    cur |= t.cur ? (1 << k) : 0;
  }
  b.cand_cur[env] = cur;
}

}  // namespace mdp::obsgen
