// 사람 따라가기 (판단 모델이 kind follow 후보를 고르면 — low_decider.h): 같은 사람을 계속 잡고(지도 번호 → 없으면 예측 자리 둘레 문(gate)
// 안 가장 가까운 사람), 사람에서 dist m 떨어진 로봇 쪽 설 자리를 아래층 목표로 준다. 실물 노드(low_rl)와 시뮬(low_explore_check --follow)이
// 같은 코드를 쓴다 — 입력은 사람 목록(지도 좌표 · 마지막으로 본 뒤 초)과 로봇 자세, 출력은 sim::LowCmd 에 넣을 값.
//
//  - 몰기: 목표 = 사람 + dist × (로봇 − 사람) 방향 단위 (부르는 쪽 spot() 이 갈 수 있는 칸으로 맞춤). 새 목표가 지금 목표에서 hyst m 안이면 그대로
//    (목표가 떨리지 않게), replan_s 마다만 다시. 속도 한도 v_max (LowCmd::vmax).
//  - 섬: 로봇이 사람에서 dist + hyst 안이거나 stop 보다 가까우면 목표 없이 사람 쪽으로 돌아 섬 (LowCmd::face) — stop 안으로는 몰지 않음.
//  - 놓침: lost_s 넘게 못 보면 먼저 마지막 목표(마지막으로 본 자리 앞)까지 가고, 거기서 사람이 가던 쪽으로 천천히 돌며 찾음 (LowCmd::w_idle = ±search_w).
//    같은 사람(예측 자리 둘레 문 안)이 다시 보이면 이어 감.
//    search_s 넘게 못 찾으면 포기 (kGiveUp — 부르는 쪽이 멈추고 "놓침" 을 알림).
//  - 다시 묻기: 고른 뒤 사람이 reask_m 넘게 옮겨 가면 reask = true 한 번 (부르는 쪽이 판단 모델에 다시 물음 — 계속 따라갈지 판단 모델이 정함).
#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <string>
#include <vector>

#include "frontier.h"   // mdp::cell_class (지도 칸 종류)
#include "low_task.h"   // sim::LowCmd

namespace ofl {

struct Params {
  float dist = 1.0f;       // 사람과 둘 거리 (m)
  float stop = 0.7f;       // 이보다 가까우면 앞으로 몰지 않음 (m, 로봇 가운데 ~ 사람 자리)
  float hyst = 0.3f;       // 목표 바꿈 문턱 (m) · 섬 ↔ 몰기 띠
  float replan_s = 0.3f;   // 목표 다시 셈 간격 (s) — 약 3 Hz
  float gate = 0.8f;       // 같은 사람 문 (m, 예측 자리에서) — 못 본 시간 × gate_v 만큼 넓힘
  float gate_v = 1.2f;     // 사람 걸음 상한 (m/s)
  float gate_max = 3.0f;   // 문 상한 (m)
  float lost_s = 1.5f;     // 이만큼 못 보면 놓침 (s)
  float search_s = 12.0f;  // 놓친 뒤 이만큼 찾고 포기 (s)
  float search_w = 0.5f;   // 찾기 제자리 회전 ω (rad/s)
  float v_max = 0.6f;      // 따라가는 동안 속도 한도 (m/s)
  float reask_m = 1.5f;    // 고른 뒤 사람이 이만큼 옮기면 판단 모델에 다시 물음 (m)
  float clear = 0.25f;     // 설 자리 둘레 빈 칸 반경 (m) — object_goto robot_radius 와 같은 뜻
  float search = 0.6f;     // 바라는 자리에서 설 자리를 찾는 반경 (m)
};

struct Person {
  std::string oid;     // 물체 지도 번호 ("person-2")
  float x = 0, y = 0;  // 지도 좌표 (m)
  float seen_s = 0;    // 마지막으로 본 뒤 (s)
};

enum Mode : int { kOff = 0, kDrive = 1, kHold = 2, kSearch = 3, kGiveUp = 4 };

struct Out {
  int mode = kOff;
  float gx = 0, gy = 0;    // kDrive: 목표 자리
  bool new_goal = false;   // 이번에 목표가 바뀜
  float px = 0, py = 0;    // 사람 자리 (마지막으로 본 곳)
  float w_idle = 0;        // kSearch: 제자리 회전 ω
  bool reask = false;
  float d = 0;             // 로봇 ~ 사람 (m)
  bool searching = false;  // 놓쳐서 찾는 중 (마지막 자리로 가는 kDrive 포함)
};

class Follower {
 public:
  explicit Follower(const Params& p = Params()) : p_(p) {}
  const Params& params() const { return p_; }
  bool active() const { return on_; }
  const std::string& oid() const { return oid_; }
  void start(const std::string& oid, float x, float y, double t) {
    on_ = true;
    oid_ = oid;
    px_ = x; py_ = y; vx_ = vy_ = 0;
    t_seen_ = t; t_plan_ = -1e9; ax_ = x; ay_ = y;
    have_goal_ = false;
    mode_ = kDrive;
  }
  void stop() { on_ = false; mode_ = kOff; }
  // spot(sx, sy, px, py, &gx, &gy): 바라는 자리 (사람 px py) → 갈 수 있는 자리 (못 찾으면 false — spot_on_map). t = 지금 (s)
  template <class Spot>
  Out step(double t, float rx, float ry, const std::vector<Person>& ps, Spot spot) {
    Out o;
    if (!on_) return o;
    // 같은 사람 찾기: 번호가 같고 최근에 본 것 → 아니면 예측 자리 둘레 문 안 가장 가까운 (최근에 본) 사람
    const double since = t - t_seen_;
    const float qx = px_ + vx_ * (float)std::min(since, 1.0), qy = py_ + vy_ * (float)std::min(since, 1.0);
    const float g = std::min(p_.gate_max, p_.gate + p_.gate_v * (float)since);
    const Person* hit = nullptr;
    for (const auto& q : ps)
      if (q.oid == oid_ && q.seen_s <= p_.lost_s && std::hypot(q.x - qx, q.y - qy) <= g) hit = &q;
    if (!hit) {
      float bd = g;
      for (const auto& q : ps) {
        if (q.seen_s > p_.lost_s) continue;
        const float dd = std::hypot(q.x - qx, q.y - qy);
        if (dd <= bd) { bd = dd; hit = &q; }
      }
    }
    if (hit) {
      const double ts = t - hit->seen_s;
      if (ts > t_seen_ + 1e-3) {   // 새 관측: 속도 어림 (낮은 통과)
        const float dt = (float)(ts - t_seen_);
        if (dt < 1.0f) { vx_ = 0.6f * vx_ + 0.4f * (hit->x - px_) / dt; vy_ = 0.6f * vy_ + 0.4f * (hit->y - py_) / dt; }
        else vx_ = vy_ = 0;
      }
      oid_ = hit->oid;
      px_ = hit->x; py_ = hit->y;
      t_seen_ = std::max(t_seen_, ts);
    }
    o.px = px_; o.py = py_;
    o.d = std::hypot(rx - px_, ry - py_);
    const double lost = t - t_seen_;
    if (lost > p_.lost_s) {
      if (lost > p_.lost_s + p_.search_s) { mode_ = kGiveUp; on_ = false; o.mode = kGiveUp; return o; }
      o.searching = true;
      // 먼저 마지막으로 본 자리 쪽(모퉁이를 돈 사람)으로 감: 지금 목표가 있고 아직 그 자리에서 dist + hyst 밖이면 그대로 몲
      if (have_goal_ && o.d > p_.dist + p_.hyst) { mode_ = kDrive; o.mode = kDrive; o.gx = gx_; o.gy = gy_; return o; }
      mode_ = kSearch;
      o.mode = kSearch;
      // 그다음 사람이 가던 쪽으로 천천히 돎 (로봇에서 본 사람 걸음의 가로 방향 부호)
      o.w_idle = search_dir_ * p_.search_w;
      have_goal_ = false;
      return o;
    }
    {   // 찾기 방향 = 로봇에서 본 사람 걸음 방향 (가로 성분 부호)
      const float bx = px_ - rx, by = py_ - ry;
      const float cr = bx * vy_ - by * vx_;
      if (std::fabs(cr) > 1e-3f) search_dir_ = cr > 0 ? 1.f : -1.f;
    }
    if (std::hypot(px_ - ax_, py_ - ay_) > p_.reask_m) { o.reask = true; ax_ = px_; ay_ = py_; }
    // 섬 ↔ 몰기 (띠): 가까우면 섬, dist + 2 hyst 넘게 멀어지면 다시 몲
    if (o.d <= p_.stop || (mode_ == kHold && o.d <= p_.dist + 2.f * p_.hyst) || (mode_ != kHold && o.d <= p_.dist + p_.hyst)) {
      mode_ = kHold;
      o.mode = kHold;
      have_goal_ = false;
      return o;
    }
    mode_ = kDrive;
    o.mode = kDrive;
    if (have_goal_ && t - t_plan_ < p_.replan_s) { o.gx = gx_; o.gy = gy_; return o; }
    t_plan_ = t;
    const float ux = (rx - px_) / std::max(o.d, 1e-3f), uy = (ry - py_) / std::max(o.d, 1e-3f);
    float gx = 0, gy = 0;
    if (!spot(px_ + p_.dist * ux, py_ + p_.dist * uy, px_, py_, &gx, &gy)) {   // 갈 자리 없음: 사람 쪽을 보고 섬
      mode_ = kHold;
      o.mode = kHold;
      have_goal_ = false;
      return o;
    }
    if (!have_goal_ || std::hypot(gx - gx_, gy - gy_) > p_.hyst) { gx_ = gx; gy_ = gy; have_goal_ = true; o.new_goal = true; }
    o.gx = gx_; o.gy = gy_;
    return o;
  }

 private:
  Params p_;
  bool on_ = false, have_goal_ = false;
  int mode_ = kOff;
  std::string oid_;
  float px_ = 0, py_ = 0, vx_ = 0, vy_ = 0, gx_ = 0, gy_ = 0, ax_ = 0, ay_ = 0, search_dir_ = 1.f;
  double t_seen_ = 0, t_plan_ = -1e9;
};

// 지도(lmap: 로그 오즈 × 1000, stride 줄 간격, 창 gw × gh, 원점 ox oy, 칸 res)에서 바라는 자리 (x, y) 둘레 search_m 안 칸 가운데
// 둘레 clear_m 이 모두 아는 빈 칸이고 사람 (px, py) 에서 min_d 이상인 가장 가까운 칸 (가운데 좌표). 없으면 false
inline bool spot_on_map(const int16_t* lmap, int stride, int gw, int gh, float ox, float oy, float res, float x, float y, float px, float py, float min_d,
                        float clear_m, float search_m, float* gx, float* gy) {
  const int cx = (int)std::floor((x - ox) / res), cy = (int)std::floor((y - oy) / res);
  const int R = (int)std::ceil(search_m / res), C = (int)std::ceil(clear_m / res);
  float bd = 1e9f;
  for (int dy = -R; dy <= R; ++dy)
    for (int dx = -R; dx <= R; ++dx) {
      const float dd = std::hypot((float)dx, (float)dy) * res;
      if (dd > search_m || dd >= bd) continue;
      const int qx = cx + dx, qy = cy + dy;
      const float wx = ox + (qx + 0.5f) * res, wy = oy + (qy + 0.5f) * res;
      if (std::hypot(wx - px, wy - py) < min_d) continue;
      bool ok = qx - C >= 0 && qy - C >= 0 && qx + C < gw && qy + C < gh;
      for (int ey = -C; ok && ey <= C; ++ey)
        for (int ex = -C; ok && ex <= C; ++ex)
          if (ex * ex + ey * ey <= C * C && mdp::cell_class(lmap[(size_t)(qy + ey) * stride + (qx + ex)]) != mdp::kFree) ok = false;
      if (ok) { bd = dd; *gx = wx; *gy = wy; }
    }
  return bd < 1e9f;
}

// 따라가기 출력 → 아래층 명령 (cell(x, y) = 탐사 지도 칸 번호, 창 밖 −1). 노드·시뮬이 같은 함수
template <class Cell>
sim::LowCmd cmd_of(const Out& o, const Params& p, Cell cell) {
  sim::LowCmd c{-1, 1};
  c.vmax = p.v_max;
  if (o.mode == kDrive) c.target = cell(o.gx, o.gy);
  else if (o.mode == kHold) c.face = cell(o.px, o.py);
  else if (o.mode == kSearch) c.w_idle = o.w_idle;
  else return sim::LowCmd{-1, 0};
  return c;
}

}  // namespace ofl
