// 실물 쪽 탐사 환경 — robot_env.h 설명
#include "limo_fire_explore/robot_env.h"

#include <cmath>
#include <algorithm>
#include <cstring>
#include <stdexcept>

namespace limo_fire_explore {

RobotEnv::RobotEnv(const sim::EnvHostCfg& hc) {
  keep_.use_cache = true;   // 걸음 사이 프론티어 칸 판정 기억 (set_frontier_cache)
  b_.cfg = hc.cfg;
  sim::EnvCfg& c = b_.cfg;
  c.action_mode = sim::kActPath;   // 계층형 (후보·경로 계획)
  // 합친 탐사 (scripts/combine_run.cpp 와 같게): 아래층 걸음 길 · 목표는 호스트(위층) · 탐사 판 끝 규칙 · 정체는 막힘 신호만.
  // 닫힘을 라이다 지도만으로 하려면 부르는 쪽이 low_explore 2 로 줌 (env.h EnvCfg::low_explore)
  const int close_lidar = hc.cfg.low_explore == 2 ? 2 : 1;
  c.low_stat = c.low_task = c.low_rl = 1;
  c.low_explore = close_lidar;
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
  b_.maps = &m_;
  b_.n_maps = 1;
  b_.n_env = 1;
  lmap_.assign((size_t)n, sim::kUnknown);
  marks_.assign((size_t)n / 4, 0u);
  trail_.assign((size_t)2 * nf, 0u);
  depth_.assign((size_t)mdp::obs::DEPTH_SIZE, 0);
  ring_.assign((size_t)sim::kLidarRing * mdp::obs::LIDAR_BINS, 0.f);
  done_buf_.assign(1, 0);
  ep_.assign(1, sim::EpInfo{});
  cand_.assign((size_t)mdp::hier::K, -1);
  cand_cur_.assign(1, 0);
  b_.lmap = lmap_.data();
  b_.marks = marks_.data();
  b_.trail = trail_.data();
  b_.depth = depth_.data();
  b_.lidar_ring = ring_.data();
  b_.done = done_buf_.data();
  b_.ep = ep_.data();
  b_.cand = cand_.data();
  b_.cand_cur = cand_cur_.data();
  if (c.cam_on) {   // 깊이 카메라 장애물 층 (후보 찾기가 계획기와 같은 막힘으로 봄 — obs_ref.h reach_obs sb.cam)
    cvox_.assign((size_t)n, 0);
    cpts_.assign((size_t)mdp::obs::DEPTH_W * 2, 3e38f);
    b_.cvox = cvox_.data();
    b_.cpts = cpts_.data();
    if (layer_n() > 0) { cldep_.assign((size_t)layer_n(), 0); b_.cldep = cldep_.data(); }   // 층 전용 깊이 (set_layer_depth)
  }
  // 아래층 걸음 길 (env_host.cpp 가 cfg.low_rl 일 때 잡는 칸)
  lowrl_.assign(1, sim::LowRl{});
  sim::lowrl_clear(lowrl_[0]);
  lowrl_[0].stall = 0;
  lowrl_[0].stall_tgt = -1;
  lowcmd_.assign(1, sim::LowCmd{-1, 0});
  lowep_.assign(1, sim::LowEp{});
  b_.lowrl = lowrl_.data();
  b_.lowcmd = lowcmd_.data();
  b_.low = lowep_.data();
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
  e_.kb_x0 = e_.kb_y0 = 1 << 30;
  e_.kb_x1 = e_.kb_y1 = -1;
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
  sim::lowrl_clear(lowrl_[0]);   // 시뮬 판 시작: lowrl_clear (막힘 신호는 그대로) · 목표 없음
  lowcmd_[0] = sim::LowCmd{-1, 0};
  obs();
}

void RobotEnv::step(const Proprio& p, const Scan& s) {
  const sim::EnvCfg& c = b_.cfg;
  // 1. 행동 자리: 명령·직전 행동은 low_act 가 정했음 (시뮬 apply_action 계층형 가지)
  e_.collided = 0;
  e_.done = 0;
  e_.people.hit_person = 0;
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
  if (s.n_rev > 0) {   // 시뮬 장을 그대로 넣을 때 (low_explore_check · room_run) — 한 바퀴 점 수·한 바퀴 시간 그대로
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
  sim::map_progress(e_, a.occ, a.fresh);
  sim::sense_step(b_, e_, 0, sh_);   // 간격·닿음 (센서 — 시뮬 contact.h 와 같은 함수)
  sim::finish_step(b_, e_, 0, a.wall, a.floor, a.known, a.fresh);
  done_ = e_.done;
  sim::ring_advance(e_);
  for (int tid = 0; tid < sim::kThreads; ++tid) sim::push_bins(b_, e_, 0, sh_, tid);
  obs();
}

void RobotEnv::obs() {
  sim::Shared sh{};
  sim::write_obs(b_, e_, 0, sh, false, obs_hier_, &keep_);   // 깊이는 set_depth 가 (실물 영상)
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

void RobotEnv::set_layer_depth(const uint8_t* d) {
  if (!cldep_.empty()) std::memcpy(cldep_.data(), d, cldep_.size());
}

void RobotEnv::layer_from_image(const uint16_t* mm, int w, int h, int step_px, float fx, float fy, float cx, float cy, uint8_t* out) const {
  const sim::DepthModel& DM = b_.cfg.depth;
  const sim::DepthSpec& D = DM.spec;
  for (int i = 0, n = layer_n(); i < n; ++i) {
    float u, v;
    sim::cam_uv(b_.cfg, i, &u, &v);
    if (!sim::depth_valid(D, u, v)) { out[i] = 0; continue; }
    const float ru = cx + (u - D.cx) * (fx / D.fx), rv = cy + (v - D.cy) * (fy / D.fy);
    const int pu = (int)std::floor(ru + 0.5f), pv = (int)std::floor(rv + 0.5f);
    if (pu < 0 || pv < 0 || pu >= w || pv >= h) { out[i] = 0; continue; }
    const uint16_t z = mm[(size_t)pv * step_px + pu];
    out[i] = mdp::obsgen::depth_u8(DM, z == 0 ? 0.f : (float)z * 0.001f);
  }
}

bool RobotEnv::cell_xy(int cell, float* x, float* y) const {
  if (cell < 0) return false;
  *x = m_.ox + ((float)(cell % m_.gw) + 0.5f) * b_.cfg.res;
  *y = m_.oy + ((float)(cell / m_.gw) + 0.5f) * b_.cfg.res;
  return true;
}

void RobotEnv::set_cmd(const sim::LowCmd& cmd) {
  lowcmd_[0] = cmd;   // 시뮬 Env::set_low_cmd 와 같음 (LowRl 은 안 건드림 — 새 목표면 lowrl_due 가 tgt_seen 으로 계획 차례를 앞당김)
}

void RobotEnv::low_act(float* v, float* w) {
  // 시뮬 step_env 첫머리 + apply_action 계층형 가지 (env.h): collided 0 → lowrl_begin → 추종기 (목표 없으면 제자리 회전, 속도 다듬기·충돌 감시까지 같은 함수)
  e_.collided = 0;
  sim::lowrl_begin(b_, e_, 0);
  sim::PathStep ps;
  sim::path_step_a(b_, e_, 0, nullptr, ps);
  const bool imm = ps.want_imm && sim::path_imminent(b_, e_, 0, ps.vc, ps.wc, ps.cd);
  sim::path_step_b(b_, e_, 0, imm, nullptr, ps);
  sim::path_step_c(b_, e_, 0, sim::path_monitor_scale(b_, e_, 0, ps.v, ps.w), ps);
  *v = e_.cmd_v;
  *w = e_.cmd_w;
}

void RobotEnv::hold() {
  e_.collided = 0;
  sim::lowrl_begin(b_, e_, 0);
  e_.cmd_v = e_.cmd_vy = e_.cmd_w = 0.f;
}

void RobotEnv::low_prepare() {
  // 시뮬 env_ref.cpp low_prepare_env 와 같은 차례: 계획 차례면 깊이 카메라 층 · 경로 계획(행동 칸 = set_slot — 위층이 고른 후보 번호, fuel 이면 0, −1 목표 없음)
  cam_ran_ = sim::lowrl_cam_due(b_, e_, 0) || sim::lowrl_due(b_, e_, 0);
  if (sim::lowrl_cam_due(b_, e_, 0) || sim::lowrl_due(b_, e_, 0)) keep_.valid = false;   // 깊이 층이 바뀜 → 이 걸음 찾기 결과는 더 안 맞음
  if (sim::lowrl_cam_due(b_, e_, 0) && !sim::lowrl_due(b_, e_, 0)) sim::camlayer_ref(b_, e_, 0);   // cam_every: 계획 차례가 아닌 걸음에도 층만 (시뮬 low_prepare_env 와 같음)
  if (sim::lowrl_due(b_, e_, 0)) {
    sim::camlayer_ref(b_, e_, 0);
    sim::plan_ref(b_, e_, 0, sim::slot_of(b_, slot_));
    sim::lowrl_planned(b_, e_, 0);
  }
}

int RobotEnv::prime_map(const int8_t* occ, int w, int h, float res, float ox, float oy, int occ_thresh, int free_thresh, int16_t l_occ, int16_t l_free) {
  const sim::EnvCfg& c = b_.cfg;
  int n = 0;
  for (int cy = 0; cy < m_.gh; ++cy) {
    const float y = m_.oy + ((float)cy + 0.5f) * c.res;
    const int my = (int)std::floor((y - oy) / res);
    if (my < 0 || my >= h) continue;
    for (int cx = 0; cx < m_.gw; ++cx) {
      int16_t& L = lmap_[(size_t)cy * c.size + cx];
      if (L != sim::kUnknown) continue;
      const float x = m_.ox + ((float)cx + 0.5f) * c.res;
      const int mx = (int)std::floor((x - ox) / res);
      if (mx < 0 || mx >= w) continue;
      const int v = occ[(size_t)my * w + mx];
      if (v < 0) continue;
      if (v >= occ_thresh) L = l_occ;
      else if (v <= free_thresh) L = l_free;
      else continue;
      ++n;
    }
  }
  if (n > 0) {   // 칸 판정 기억(프론티어)은 바뀐 칸을 모르므로 처음부터 다시 (값은 같음 — 다시 셀 뿐)
    keep_.valid = false;
    const int every = keep_.cache.check_every;
    keep_.cache = mdp::FrontierCache{};
    keep_.cache.check_every = every;
  }
  return n;
}

void RobotEnv::set_wall_ring(const int8_t* occ, int w, int h, float res, float ox, float oy, int occ_thresh, float r) {
  const sim::EnvCfg& c = b_.cfg;
  if (r <= 0.f) { wall_ring_.clear(); return; }
  wall_ring_.assign((size_t)c.size * c.size, 0);
  const int k = (int)std::ceil(r / c.res);
  for (int cy = 0; cy < m_.gh; ++cy)
    for (int cx = 0; cx < m_.gw; ++cx) {
      const float x = m_.ox + ((float)cx + 0.5f) * c.res, y = m_.oy + ((float)cy + 0.5f) * c.res;
      const int mx = (int)std::floor((x - ox) / res), my = (int)std::floor((y - oy) / res);
      if (mx < 0 || my < 0 || mx >= w || my >= h || occ[(size_t)my * w + mx] < occ_thresh) continue;
      for (int dy = -k; dy <= k; ++dy)
        for (int dx = -k; dx <= k; ++dx) {
          const int u = cx + dx, v = cy + dy;
          if (u < 0 || v < 0 || u >= m_.gw || v >= m_.gh || (float)(dx * dx + dy * dy) * c.res * c.res > r * r) continue;
          wall_ring_[(size_t)v * c.size + u] = 1;
        }
    }
}

int RobotEnv::mask_cam_wall() {
  if (cvox_.empty() || wall_ring_.empty()) return 0;
  int n = 0;
  for (size_t i = 0; i < cvox_.size(); ++i)
    if (wall_ring_[i] && cvox_[i]) { cvox_[i] = 0; ++n; }
  if (n > 0) keep_.valid = false;
  return n;
}

int RobotEnv::mask_cam_near_occ(float r, float win) {
  if (cvox_.empty() || r <= 0.f) return 0;
  const sim::EnvCfg& c = b_.cfg;
  const int k = (int)std::ceil(r / c.res), W = (int)std::ceil(win / c.res);
  int rcx = (int)std::floor((e_.x - m_.ox) / c.res), rcy = (int)std::floor((e_.y - m_.oy) / c.res);
  int n = 0;
  for (int cy = std::max(0, rcy - W); cy <= std::min(m_.gh - 1, rcy + W); ++cy)
    for (int cx = std::max(0, rcx - W); cx <= std::min(m_.gw - 1, rcx + W); ++cx) {
      uint8_t& v = cvox_[(size_t)cy * c.size + cx];
      if (!v) continue;
      bool near = false;
      for (int dy = -k; dy <= k && !near; ++dy)
        for (int dx = -k; dx <= k && !near; ++dx) {
          const int u = cx + dx, w = cy + dy;
          if (u < 0 || w < 0 || u >= m_.gw || w >= m_.gh || (float)(dx * dx + dy * dy) * c.res * c.res > r * r) continue;
          near = mdp::cell_class(lmap_[(size_t)w * c.size + u]) == mdp::kOcc;
        }
      if (near) { v = 0; ++n; }
    }
  if (n > 0) keep_.valid = false;
  return n;
}

int RobotEnv::clear_cam_cells(const float* xy, int n) {
  if (cvox_.empty()) return 0;
  const sim::EnvCfg& c = b_.cfg;
  int k = 0;
  for (int i = 0; i < n; ++i) {
    const int cx = (int)std::floor((xy[2 * i] - m_.ox) / c.res), cy = (int)std::floor((xy[2 * i + 1] - m_.oy) / c.res);
    if (cx < 0 || cy < 0 || cx >= m_.gw || cy >= m_.gh) continue;
    uint8_t& v = cvox_[(size_t)cy * c.size + cx];
    if (v) { v = 0; ++k; }
  }
  if (k > 0) keep_.valid = false;
  return k;
}

void RobotEnv::clear_gaveup() {
  for (int i = 0; i < sim::kRecBl; ++i) {
    e_.p_bl_cell[i] = -1;
    e_.p_bl_until[i] = 0;
  }
  e_.p_blk = 0;
  e_.p_blk_now = 0;
}

int RobotEnv::add_points(const float* xy, int n) {
  if (cpts_.empty()) return 0;
  const int cap = (int)(cpts_.size() / 2);
  int k = 0;
  if (cam_ran_ || extra_at_ < 0) {   // 깊이 층이 새로 셈 → 그 점 끝(첫 빈 칸)부터
    while (k < cap && cpts_[2 * k] != sim::kCamNone) ++k;
  } else {   // 이번 걸음 깊이 층은 그대로 → 지난번 넣은 자리를 고쳐 씀
    k = extra_at_;
  }
  for (int i = k; i < std::min(cap, k + extra_n_) && !cam_ran_; ++i) cpts_[2 * i] = cpts_[2 * i + 1] = sim::kCamNone;
  int m = 0;
  for (; m < n && k + m < cap; ++m) {
    cpts_[2 * (k + m)] = xy[2 * m];
    cpts_[2 * (k + m) + 1] = xy[2 * m + 1];
  }
  extra_at_ = k;
  extra_n_ = m;
  cam_ran_ = false;
  return m;
}

}  // namespace limo_fire_explore
