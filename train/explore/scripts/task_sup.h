// 지시 일 감독 (판단 모델이 물체 후보를 골랐을 때 · ~/goto): 설 자리로 가기 → 닿으면 물체 쪽으로 돌아 섬 → 끝, 못 가면 실패를 알리고 그 물체를 잠시 뺌,
// 한 지시에서 실패가 잇따르거나 시간이 넘으면 포기. 실물 노드(low_rl)와 시뮬(low_explore_check --dec-fake obj)이 같은 코드 — 입력은 시각 · 지도 자세 ·
// 아래층 상태(경로 없음), 출력은 sim::LowCmd 와 사건.
//
//  - 감: LowCmd{설 자리 칸, follow 1}. 설 자리에서 arrive_m 안이면 닿음 → 바라보기: LowCmd{−1, 1, face = 물체 칸} (아래층 hier_path.h path_idle_w 가 돎),
//    남은 각 < face_tol 이거나 face_timeout_s 넘으면 kArrived. 바라볼 물체가 없으면 바로 kArrived.
//  - 실패 (kFailed): stall_s 동안 설 자리에 stall_m 더 못 다가감 (막힘 · 돌기만 함) · 경로 없음이 nopath_s 이어짐. 뒤로 물러나지 않음 — 부르는 쪽이
//    그 물체를 Fails 에 넣고(fail_ttl_s 동안 후보에서 뺌) 판단 모델에 다시 묻는다.
//  - 지시 감독 (Instr): 지시마다 실패 수 · 시작 시각. max_fail 번 실패하면 "못 감", instr_limit_s 넘도록 못 끝내면 "못 찾음" (0 = 끔).
#pragma once
#include <algorithm>
#include <cmath>
#include <string>
#include <vector>

#include "env.h"        // sim::EnvState
#include "low_task.h"   // sim::LowCmd

namespace tsk {

struct Params {
  float arrive_m = 0.4f;        // 설 자리 닿음 (m) — low_rl goto_arrive_m
  float arrive_far_m = 1.0f;    // 아래층 경로가 끝났으면(설 자리가 막혀 계획기가 둘레 허용 칸으로 바꿈) 이 안이면 닿음으로 (m)
  float face_tol = 0.15f;       // 바라보기 끝 남은 각 (rad) — env.yaml action.path.face_tol 과 같게
  float face_timeout_s = 5.f;   // 바라보기 시간 한도 (s)
  float stall_s = 10.f;         // 이 동안
  float stall_m = 0.3f;         // 이만큼 더 못 다가가면 실패 (m)
  float nopath_s = 3.f;         // 경로 없음이 이만큼 이어지면 실패 (s)
  int max_fail = 3;             // 한 지시에서 물체 가기 실패가 이만큼이면 "못 감"
  float fail_ttl_s = 120.f;     // 실패한 물체를 후보에서 빼는 시간 (s)
  float fail_r = 0.5f;          // 실패한 물체 자리에서 이 안의 같은 클래스 물체도 뺌 (m — 지도 번호가 바뀌어도)
  float instr_limit_s = 300.f;  // 지시 하나 시간 한도 (s, 0 = 끔) — 넘으면 "못 찾음"
};

enum Ev : int { kNone = 0, kArrived = 1, kFailed = 2 };

class Goto {
 public:
  explicit Goto(const Params& p = Params()) : p_(p) {}
  const Params& params() const { return p_; }
  void start(const std::string& label, const std::string& oid, const std::string& cls, float gx, float gy, bool face, float fx, float fy, double t) {
    on_ = true; facing_ = false;
    label_ = label; oid_ = oid; cls_ = cls;
    gx_ = gx; gy_ = gy; face_ = face; fx_ = fx; fy_ = fy;
    best_ = 1e9f; t_best_ = t; t_nopath_ = -1;
  }
  void cancel() { on_ = false; facing_ = false; }
  bool active() const { return on_; }
  bool facing() const { return on_ && facing_; }
  const std::string& label() const { return label_; }
  const std::string& oid() const { return oid_; }
  const std::string& cls() const { return cls_; }
  float gx() const { return gx_; }
  float gy() const { return gy_; }
  float fx() const { return fx_; }
  float fy() const { return fy_; }
  const char* why() const { return why_; }
  // no_path: 아래층이 목표 칸은 받았는데 경로 점이 0 (못 짬) · at_end: 아래층 경로 끝에 닿음 (계획기가 고른 마지막 점 — path_end() )
  Ev step(double t, float rx, float ry, float rth, bool no_path, bool at_end = false) {
    if (!on_) return kNone;
    if (facing_) {
      const float err = std::remainder(std::atan2(fy_ - ry, fx_ - rx) - rth, 6.2831853f);
      if (std::fabs(err) < p_.face_tol || t - t_face_ > p_.face_timeout_s) { on_ = facing_ = false; why_ = "arrived"; return kArrived; }
      return kNone;
    }
    const float d = std::hypot(gx_ - rx, gy_ - ry);
    if (d <= p_.arrive_m || (at_end && d <= p_.arrive_far_m)) {
      if (!face_) { on_ = false; why_ = "arrived"; return kArrived; }
      facing_ = true;
      t_face_ = t;
      return kNone;
    }
    if (d < best_ - p_.stall_m) { best_ = d; t_best_ = t; }
    if (t - t_best_ > p_.stall_s) { on_ = false; why_ = "no progress"; return kFailed; }
    if (no_path) {
      if (t_nopath_ < 0) t_nopath_ = t;
      if (t - t_nopath_ > p_.nopath_s) { on_ = false; why_ = "no path"; return kFailed; }
    } else t_nopath_ = -1;
    return kNone;
  }
  // 아래층 명령 (cell(x, y) = 탐사 지도 칸, 창 밖 −1)
  template <class Cell>
  sim::LowCmd cmd(Cell cell) const {
    sim::LowCmd c{-1, 1};
    if (!on_) return c;
    if (facing_) c.face = cell(fx_, fy_);
    else c.target = cell(gx_, gy_);
    return c;
  }

 private:
  Params p_;
  bool on_ = false, facing_ = false, face_ = false;
  std::string label_, oid_, cls_;
  float gx_ = 0, gy_ = 0, fx_ = 0, fy_ = 0, best_ = 1e9f;
  double t_best_ = 0, t_nopath_ = -1, t_face_ = 0;
  const char* why_ = "";
};

// 아래층 경로 끝에 닿음: 경로 점이 있고 로봇이 마지막 점에서 tol 안 (지도 창 원점 ox oy, 칸 res — EnvState::wp 는 창 칸)
inline bool path_end(const sim::EnvState& e, float ox, float oy, float res, float tol) {
  if (e.p_n <= 0) return false;
  const float lx = ox + ((float)e.wp[2 * (e.p_n - 1)] + 0.5f) * res, ly = oy + ((float)e.wp[2 * (e.p_n - 1) + 1] + 0.5f) * res;
  return std::hypot(lx - (e.x + e.ex), ly - (e.y + e.ey)) <= tol;
}

// 실패한 물체 (지시 하나 동안 · fail_ttl_s): 번호가 같거나 같은 클래스로 fail_r 안이면 뺌
class Fails {
 public:
  explicit Fails(const Params& p = Params()) : p_(p) {}
  void add(const std::string& oid, const std::string& cls, float x, float y, double t) { f_.push_back({oid, cls, x, y, t}); }
  bool blocked(const std::string& oid, const std::string& cls, float x, float y, double t) const {
    for (const auto& f : f_)
      if (t - f.t <= p_.fail_ttl_s && (f.oid == oid || (f.cls == cls && std::hypot(f.x - x, f.y - y) <= p_.fail_r))) return true;
    return false;
  }
  void clear() { f_.clear(); }
  int n() const { return (int)f_.size(); }

 private:
  struct F { std::string oid, cls; float x, y; double t; };
  Params p_;
  std::vector<F> f_;
};

// 지시 하나의 감독: 시작 시각 · 실패 수 → 포기 까닭 (없으면 nullptr)
class Instr {
 public:
  explicit Instr(const Params& p = Params()) : p_(p) {}
  void start(double t) { on_ = true; t0_ = t; fails_ = 0; }
  void stop() { on_ = false; }
  bool active() const { return on_; }
  void failed() { ++fails_; }
  int fails() const { return fails_; }
  double age(double t) const { return on_ ? t - t0_ : 0.0; }
  const char* give_up(double t) const {
    if (!on_) return nullptr;
    if (fails_ >= p_.max_fail) return "could not reach";
    if (p_.instr_limit_s > 0.f && t - t0_ > p_.instr_limit_s) return "not found in time";
    return nullptr;
  }

 private:
  Params p_;
  bool on_ = false;
  double t0_ = 0;
  int fails_ = 0;
};

}  // namespace tsk
