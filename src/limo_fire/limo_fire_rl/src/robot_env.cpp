// 실물 쪽 탐사 RL 환경 — robot_env.h 설명
#include "limo_fire_rl/robot_env.h"

#include <cmath>
#include <cstring>
#include <stdexcept>

namespace limo_fire_rl {

RobotEnv::RobotEnv(const sim::EnvHostCfg& hc) {
  b_.cfg = hc.cfg;
  sim::EnvCfg& c = b_.cfg;
  c.action_mode = sim::kActPath;   // 계층형 (train --hier 와 같음)
  c.curriculum = 0;                // 단계·오차 배율 없음 (평가와 같음)
  const int n = c.size * c.size;
  const int nf = c.size * sim::fmap_row(c);
  // 맵: 정답 격자 없음 (0 — 덮음·벽·바닥 칸 없음). 지도 창 = size × size 칸 전부
  grid_.assign((size_t)n, 0);
  m_.grid = grid_.data();
  m_.gw = m_.gh = c.size;
  m_.n_wall = m_.n_floor = 1;   // 덮음 분모 (실물엔 정답 벽이 없어 덮음은 0 — 보상·덮음 칸은 안 씀)
  m_.spawn = nullptr;
  m_.clear = nullptr;
  m_.n_spawn = 0;
  m_.stage = 0;
  b_.maps = &m_;
  b_.n_maps = 1;
  b_.n_env = 1;
  lmap_.assign((size_t)n, sim::kUnknown);
  marks_.assign((size_t)n / 4, 0u);
  trail_.assign((size_t)2 * nf, 0u);
  img_.assign((size_t)mdp::obs::IMG_SIZE, 0);
  o1d_.assign((size_t)mdp::hier::POLICY_1D, 0.f);
  ring_.assign((size_t)sim::kLidarRing * mdp::obs::LIDAR_BINS, 0.f);
  vprev_.assign((size_t)mdp::obs::VEC, 0.f);
  lint_.assign((size_t)mdp::obs::LIDAR_BINS, 0.f);
  rew_.assign(1, 0.f);
  terms_.assign((size_t)mdp::rew::kTerms, 0.f);
  done_buf_.assign(1, 0);
  ep_.assign(1, sim::EpInfo{});
  cand_.assign((size_t)mdp::hier::K, -1);
  cand_cur_.assign(1, 0);
  b_.lmap = lmap_.data();
  b_.marks = marks_.data();
  b_.trail = trail_.data();
  b_.obs_img = img_.data();
  b_.obs_1d = o1d_.data();
  b_.o1d = mdp::hier::POLICY_1D;
  b_.lidar_ring = ring_.data();
  b_.vec_prev = vprev_.data();
  b_.lidar_int = lint_.data();
  b_.reward = rew_.data();
  b_.reward_terms = terms_.data();
  b_.done = done_buf_.data();
  b_.ep = ep_.data();
  b_.cand = cand_.data();
  b_.cand_cur = cand_cur_.data();
  if (c.cam_on) {   // 깊이 카메라 장애물 층 (후보 찾기가 계획기와 같은 막힘으로 봄 — obs_ref.h reach_obs sb.cam)
    cvox_.assign((size_t)n, 0);
    cpts_.assign((size_t)mdp::obs::DEPTH_W * 2, 3e38f);
    b_.cvox = cvox_.data();
    b_.cpts = cpts_.data();
  }
  // GPU 전용 칸은 없음 (CPU 참조판과 같음)
  b_.fmap = nullptr;
  b_.rbits = nullptr;
  b_.scratch = nullptr;
}

void RobotEnv::reset(float x0, float y0, float th0, const Proprio& p0, float fov, const Window* win) {
  const sim::EnvCfg& c = b_.cfg;
  // 지도 창: 시작점 가운데 (시뮬은 맵 격자가 창 안 어디든 — 창 크기 같음)
  if (win) {
    if (win->gw > c.size || win->gh > c.size) throw std::runtime_error("RobotEnv::reset: 창이 지도 크기보다 큼");
    m_.ox = win->ox;
    m_.oy = win->oy;
    m_.gw = win->gw;
    m_.gh = win->gh;
  } else {
    m_.ox = x0 - 0.5f * (float)c.size * c.res;
    m_.oy = y0 - 0.5f * (float)c.size * c.res;
    m_.gw = m_.gh = c.size;
  }
  for (int tid = 0; tid < sim::kThreads; ++tid) sim::reset_map(b_, 0, tid);
  // 상태: 시뮬 reset_state 중 관측·판 끝에 쓰는 칸만 (무작위 뽑기 — 맵·바닥·오차·사람 — 는 실물에 없음)
  std::memset(&e_, 0, sizeof(e_));
  e_.map = 0;
  e_.x = e_.x0 = x0;
  e_.y = e_.y0 = y0;
  e_.th = e_.th0 = th0;
  e_.err_scale = 1.f;
  e_.kb_x0 = e_.kb_y0 = 1 << 30;
  e_.kb_x1 = e_.kb_y1 = -1;
  e_.delay = 0;              // 실물 지연은 센서 값 자체에 들어 있음 (시뮬은 0 또는 1 걸음을 판마다 뽑음)
  e_.t95_step = -1;
  e_.goal_cell = -1;
  e_.p_slot = -1;
  for (int i = 0; i < sim::kRecBl; ++i) { e_.p_bl_cell[i] = -1; e_.p_bl_until[i] = 0; }
  e_.people.n = 0;
  e_.people.near_d = 1e9f;
  e_.noise.fov = fov;   // 첫 장 전 trail_mark 시야 (장마다 /scan 값으로)
  e_.noise.scan_time = 0.1f;
  e_.drive.wv = p0.wheel_v;
  e_.drive.wvy = p0.wheel_vy;
  e_.drive.ww = p0.wheel_w;
  e_.gyro = p0.gyro_z;
  for (int i = 0; i < 7; ++i) e_.imu.out[i] = p0.imu[i];
  done_ = 0;
  sh_ = sim::Shared{};
  obs();
}

void RobotEnv::step(const Proprio& p, const Scan& s) {
  const sim::EnvCfg& c = b_.cfg;
  // 1. 행동 자리: 이번 걸음 명령 (시뮬 apply_action 계층형 끝)
  e_.collided = 0;
  e_.done = 0;
  e_.people.hit_person = 0;
  e_.declared = 0;
  e_.cmd_v = p.cmd_v;
  e_.cmd_vy = 0.f;
  e_.cmd_w = p.cmd_w;
  sim::path_last_act(c, e_, p.cmd_v, p.cmd_w);
  // 2. 물리 자리: 실물 감각 (시뮬 physics_step 이 쓰는 칸)
  e_.drive.wv = p.wheel_v;
  e_.drive.wvy = p.wheel_vy;
  e_.drive.ww = p.wheel_w;
  e_.gyro = p.gyro_z;
  for (int i = 0; i < 7; ++i) e_.imu.out[i] = p.imu[i];
  e_.x = s.x;
  e_.y = s.y;
  e_.th = s.th;
  e_.w = s.w;
  e_.v = p.wheel_v;
  e_.tick += c.substeps;
  // 라이다 장 (시뮬 step_env: scan_begin → 빔 → scan_apply → known_box)
  const int nb = (int)s.ranges.size();
  if (!(s.angle_increment > 0.f) || nb < 2) throw std::runtime_error("RobotEnv::step: 라이다 각 간격·빔 수");
  e_.noise.fov = -2.f * s.angle_min;   // 빔 j 각도 = −fov/2 + j·inc (ROS angle_min + j·angle_increment)
  if (s.n_rev > 0) {   // 시뮬 장을 그대로 넣을 때 (rl_node_check) — 한 바퀴 점 수·한 바퀴 시간 그대로
    sh_.scan.n_rev = s.n_rev;
    e_.noise.scan_time = s.scan_time;
  } else {
    sh_.scan.n_rev = sim::kTwoPi / s.angle_increment;   // 시뮬 inc = kTwoPi / n_rev 의 거꿀
    e_.noise.scan_time = s.time_increment * (float)sh_.scan.n_rev;   // 빔 시각 j · scan_time / n_rev ≈ j · time_increment
  }
  sim::scan_begin_map(c, e_, s.w, sh_);
  sh_.inc = s.angle_increment;
  sh_.n_beams = nb;
  for (int j = 0; j < nb; ++j) {
    float r = s.ranges[j];
    if (!(r > 0.f) || std::isinf(r)) r = 0.f;   // 빠짐 (드라이버 0 · inf · nan) — scan_beam_map 이 버림
    const float it = j < (int)s.intensities.size() ? s.intensities[j] : 0.f;
    sim::scan_beam_map(b_, e_, 0, sh_, j, r, it, nullptr);
  }
  const sim::ApplyCount a = sim::scan_apply(b_, e_, 0, sh_, 0, 1, nullptr);
  sim::known_box(b_, e_, sh_);
  // 3. 판 끝·진행 (시뮬 step_env 끝과 같은 차례)
  sim::map_progress(e_, a.occ);
  sim::sense_step(b_, e_, 0, sh_);   // 간격·닿음 (센서 — 시뮬 contact.h 와 같은 함수)
  sim::finish_step(b_, e_, 0, a.wall, a.floor, a.known, a.fresh);
  done_ = e_.done;
  sim::ring_advance(e_);
  for (int tid = 0; tid < sim::kThreads; ++tid) sim::push_bins(b_, e_, 0, sh_, tid);
  obs();
}

void RobotEnv::obs() {
  sim::Shared sh{};
  sim::write_obs(b_, e_, 0, sh, mdp::obsgen::FarDirty{1, 0.f, 0.f, 0.f, 0.f, nullptr}, false);
}

void RobotEnv::set_depth(const uint8_t* d) {
  std::memcpy(mdp::obsgen::depth_out(b_, 0), d, (size_t)mdp::obs::DEPTH_SIZE);
}

void RobotEnv::depth_from_image(const uint16_t* mm, int w, int h, int step_px, float fx, float fy, float cx, float cy, uint8_t* out) const {
  const sim::DepthModel& DM = b_.cfg.depth;
  const sim::DepthSpec& D = DM.spec;
  for (int i = 0; i < mdp::obs::DEPTH_SIZE; ++i) {
    float u, v;
    mdp::obsgen::depth_uv(DM, i, &u, &v);
    if (!sim::depth_valid(D, u, v)) { out[i] = 0; continue; }   // 시뮬과 같은 유효 영역 (정렬 켬: 위 20 행·왼 6 열 늘 0)
    // 시뮬 픽셀 광선 (1, −(u−cx)/fx, −(v−cy)/fy) → 실물 카메라 픽셀
    const float ru = cx + (u - D.cx) * (fx / D.fx), rv = cy + (v - D.cy) * (fy / D.fy);
    const int pu = (int)std::floor(ru + 0.5f), pv = (int)std::floor(rv + 0.5f);
    if (pu < 0 || pv < 0 || pu >= w || pv >= h) { out[i] = 0; continue; }
    const uint16_t z = mm[(size_t)pv * step_px + pu];
    out[i] = mdp::obsgen::depth_u8(DM, z == 0 ? 0.f : (float)z * 0.001f);
  }
}

void RobotEnv::camlayer_update() { sim::camlayer_ref(b_, e_, 0); }

bool RobotEnv::choose(int slot) {
  // 시뮬 plan_cpu: tgt = cand[slot]; 같은 후보(cand_cur 비트)이고 목표가 있으면 그대로, 아니면 goal_cell = tgt
  e_.p_slot = slot;
  const int tgt = (slot >= 0 && slot < mdp::hier::K) ? b_.cand[slot] : -1;
  if (tgt < 0) {   // 빈 칸: 목표 없음 (시뮬은 제자리 회전)
    e_.goal_cell = -1;
    return true;
  }
  if (((b_.cand_cur[0] >> slot) & 1) && e_.goal_cell >= 0) return false;
  e_.goal_cell = tgt;
  return true;
}

bool RobotEnv::cell_xy(int cell, float* x, float* y) const {
  if (cell < 0) return false;
  *x = m_.ox + ((float)(cell % m_.gw) + 0.5f) * b_.cfg.res;
  *y = m_.oy + ((float)(cell / m_.gw) + 0.5f) * b_.cfg.res;
  return true;
}

bool RobotEnv::cand_xy(int k, float* x, float* y) const { return cell_xy(b_.cand[k], x, y); }

}  // namespace limo_fire_rl
