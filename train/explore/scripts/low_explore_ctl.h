// 합친 탐사(위층 fuel + 아래층 RL)의 위층 결정·목표 넘기기 — 로봇 하나 몫. 실물 노드(limo_fire_rl low_rl)와 그 확인 도구(sim/tools/low_node_check --explore)가 같은 코드로 돌린다.
// combine_run.cpp (--upper fuel --policy) 의 환경 하나 몫을 그대로 옮긴 것 — 줄 번호는 2026-11-16 origin/main 1bd79b5 기준:
//   피하는 칸 고리 kBl 8 · 반경 rec_bl_r · rec_bl_steps 걸음            combine_run.cpp:159-164 (Avoid) · 276-277 · 345-359 (avoided · avoid_add)
//   결정할 때: 걸음 % decide_steps == 0 또는 redo · 피하지 않는 후보 차례  combine_run.cpp:540-555
//   fuel 이 목표 칸을 냄 (로봇 지도·지나간 길·깊이 층·지도 자세)           combine_run.cpp:629-645 (fu::decide — scripts/fuel_upper.h)
//   목표 넘기기: fuel 칸 → LowCmd{칸, 2}. fuel 이 못 내면 slot0 차례(피하지 않는 첫 후보) — stick(같은 덩어리·경로 있음이면 그대로) — 후보도 없으면 LowCmd{−1, 0}
//                                                                          combine_run.cpp:1086-1146 (hold 0 · rule_near 0 · 갈래 없음 — 기본값)
//   걸음 뒤: 막힘 신호(LowRl::stall 올라감) → 그 칸 피함 · fuel 기억 비움 · 다시 결정 combine_run.cpp:1326-1335
//            통과(reach_step) · 경로 끝 밖(hold) · 경로 없음 → fuel 기억 비움 · (경로 끝 밖·경로 없음은 그 칸 피함) · 다시 결정  combine_run.cpp:1351-1371
// 판 끝(닫힘 + 진척 없음 10 s 성공 · 진척 없음 · 시간 · 닿음)은 환경(finish_step — low_explore 합친 탐사 규칙)이 정하고 부르는 쪽이 읽는다 (combine_run.cpp:1270-1272).
// LowCmd{−1, 0} 인 걸음은 두 쪽 다 시뮬 규칙 추종기(env.h apply_action 계층형 가지 — 행동 칸이 −1 이면 제자리 회전).
// 아래층 규칙(rule — combine_run 을 --policy 없이): 아래층이 RL 대신 규칙 추종기(A* plan_ref + path_step_*). 목표 넘기기만 다름 (combine_run.cpp:1124-1126 · 1143-1146):
//   fuel 칸 → LowCmd{칸, 1} · 행동 칸 0. 그 밖(slot0 · 판단 모델 · fuel 이 못 냄) → LowCmd{−1, 0} · 행동 칸 = 고른 후보 번호(없으면 −1), stick 없음.
//   행동 칸은 다음 결정까지 그대로 (combine_run 의 act 배열) — slot() 으로 꺼내 준비 단계(경로 계획 후보)에 넘김. 걸음 뒤 통과·경로 끝 밖·경로 없음은 follow 1 에서도 셈
#pragma once
#include <cmath>
#include <cstdint>
#include <vector>

#include "fuel_upper.h"
#include "low_task.h"

namespace lxc {

struct Params {
  int decide_steps = 10;    // EnvCfg::decide_steps (1 s)
  float bl_r = 0.5f;        // 피하는 칸 반경 (m) — EnvCfg::rec_bl_r (0 이면 0.5)
  int bl_steps = 100;       // 피하는 시간 (걸음) — EnvCfg::rec_bl_steps (0 이면 10 s)
  bool avoid = true, redecide = true, stick = true;   // combine_run 기본 (--no-avoid · --no-redecide · --no-stick 로 끔)
  bool rule = false;        // 아래층 규칙 추종기 (combine_run 을 --policy 없이 — 위 머리말)
  bool use_fuel = true;     // false = combine_run --upper slot0 (fuel 을 안 부르고 늘 아래 slot0 차례 — 피하지 않는 첫 후보 · stick)
  fu::Opt fuel;             // --fuel-* (부르는 쪽이 채움)
  static Params from_cfg(const sim::EnvCfg& c) {
    Params p;
    p.decide_steps = c.decide_steps;
    const double dt = (double)c.substeps * c.dt;
    p.bl_r = c.rec_bl_r > 0.f ? c.rec_bl_r : 0.5f;
    p.bl_steps = c.rec_bl_steps > 0 ? c.rec_bl_steps : (int)std::lround(10.0 / dt);
    return p;
  }
};

// 결정 하나에 쓸 로봇 쪽 값 (RobotEnv · 시뮬 Env 어느 쪽이든 같은 꼴)
struct View {
  const sim::EnvCfg* c;
  const sim::World* W;           // gw · gh · ox · oy 만 씀
  const int16_t* lmap;           // 로봇 지도 [size²]
  const uint32_t* trail;         // 지나간 칸·본 칸 판
  const uint8_t* cam;            // 깊이 카메라 층 (없으면 nullptr)
  const sim::EnvState* s;
  const int32_t* cand;           // 후보 칸 K 개 (−1 빈 칸)
  uint32_t cand_cur;             // 후보마다 "그 덩어리가 지금 목표 칸을 품음" 비트
};

class Ctl {
 public:
  explicit Ctl(const Params& p) : p_(p) {
    for (int k = 0; k < kBl; ++k) { av_cell_[k] = -1; av_until_[k] = 0; }
  }
  // 이번 걸음에 결정하나 (combine_run.cpp:544)
  bool due(const sim::EnvState& s) const { return s.step % p_.decide_steps == 0 || redo_; }
  // 결정 → 아래층 명령 (combine_run.cpp:546-555 · 629-645 · 1086-1146). why: 0 fuel · 1 slot0 (fuel 대신 또는 use_fuel 끔) · 2 목표 없음
  sim::LowCmd decide(const View& v, fu::Work& work, int* why = nullptr) {
    const sim::EnvState& s = *v.s;
    const int K = mdp::hier::K;
    int pick = -1;   // slot0: 피하지 않는 첫 후보
    for (int j = 0; j < K && pick < 0; ++j) {
      const int cell = v.cand[j];
      if (cell >= 0 && !avoided(cell, s.step)) pick = j;
    }
    const fu::In in{v.c, v.W, v.lmap, v.trail, v.cam, v.s, v.cand, [this, &s](int cell) { return avoided(cell, s.step); }};
    const int tcell = p_.use_fuel ? fu::decide(in, p_.fuel, fmem_, work) : -1;
    redo_ = false;
    last_dec_ = s.step;
    ++n_dec_;
    int tgt = pick >= 0 ? v.cand[pick] : -1;
    int w = 1;
    if (tcell >= 0 && p_.rule) {   // 규칙 추종기 + fuel: 목표 칸을 그대로 따라감 (combine_run.cpp:1124-1126)
      tgt = tcell;
      w = 0;
      cmd_ = sim::LowCmd{tgt, 1};
      slot_ = 0;
    } else if (p_.rule) {          // 규칙 추종기: 후보 번호 그대로 (combine_run.cpp:1149-1151)
      cmd_ = sim::LowCmd{-1, 0};
      slot_ = pick;
      w = pick >= 0 ? 1 : 2;
    } else if (tcell >= 0) {   // fuel 칸 그대로 (combine_run.cpp:1118-1123)
      tgt = tcell;
      w = 0;
      if (tgt != cmd_.target || cmd_.follow != 2) sw_step_ = s.step;
      cmd_ = sim::LowCmd{tgt, 2};
    } else {            // fuel 이 못 냄: slot0 차례 (combine_run.cpp:1127-1146)
      if (p_.stick && pick >= 0 && ((v.cand_cur >> pick) & 1u) && cmd_.follow == 2 && cmd_.target >= 0 && cmd_.target == s.goal_cell && s.p_n > 0 &&
          !avoided(cmd_.target, s.step))
        tgt = cmd_.target;
      if (tgt >= 0) {
        if (tgt != cmd_.target || cmd_.follow != 2) sw_step_ = s.step;
        cmd_ = sim::LowCmd{tgt, 2};
      } else {
        cmd_ = sim::LowCmd{-1, 0};
        w = 2;
      }
    }
    if (why) *why = w;
    return cmd_;
  }
  // 위층 판단 모델 (upper decider — low_decider.h): 후보 want 를 고른 결정. why: 0 판단 모델 후보 · 1 slot0 차례 · 2 목표 없음. want < 0 이거나 피하는 칸 · 빈 칸이면 slot0 차례(피하지 않는 첫 후보).
  // 목표 넘기기는 decide 의 slot0 가지 그대로 (combine_run.cpp:1127-1146 — 판단 모델 · slot0 위층이 같은 길)
  sim::LowCmd decide_slot(const View& v, int want, int* why = nullptr) {
    const sim::EnvState& s = *v.s;
    const int K = mdp::hier::K;
    int pick = -1;
    if (want >= 0 && want < K && v.cand[want] >= 0 && !avoided(v.cand[want], s.step)) pick = want;
    for (int j = 0; j < K && pick < 0; ++j)
      if (v.cand[j] >= 0 && !avoided(v.cand[j], s.step)) pick = j;
    redo_ = false;
    last_dec_ = s.step;
    ++n_dec_;
    if (p_.rule) {   // 규칙 추종기: 후보 번호 그대로 · stick 없음 (combine_run.cpp:1149-1151)
      cmd_ = sim::LowCmd{-1, 0};
      slot_ = pick;
      if (why) *why = pick >= 0 ? 1 : 2;
      return cmd_;
    }
    int tgt = pick >= 0 ? v.cand[pick] : -1;
    if (p_.stick && pick >= 0 && ((v.cand_cur >> pick) & 1u) && live(s)) tgt = cmd_.target;
    int w = want >= 0 && pick == want ? 0 : 1;   // 0 = 판단 모델이 고른 후보 · 1 = slot0 차례
    if (tgt >= 0) {
      if (tgt != cmd_.target || cmd_.follow != 2) sw_step_ = s.step;
      cmd_ = sim::LowCmd{tgt, 2};
    } else {
      cmd_ = sim::LowCmd{-1, 0};
      w = 2;
    }
    if (why) *why = w;
    return cmd_;
  }
  // 지금 목표가 살아 있나 (아래층이 그 칸으로 가는 경로가 있고 피하는 칸이 아님)
  bool live(const sim::EnvState& s) const {
    if (p_.rule) return slot_ >= 0 && s.goal_cell >= 0 && s.p_n > 0 && !avoided(s.goal_cell, s.step);   // 규칙: 고른 후보로 가는 경로가 있음
    return cmd_.follow == 2 && cmd_.target >= 0 && cmd_.target == s.goal_cell && s.p_n > 0 && !avoided(cmd_.target, s.step);
  }
  bool redo_pending() const { return redo_; }   // 막힘 · 통과 · 경로 끝 밖 · 경로 없음 뒤 바로 다시 결정해야 함
  // 걸음 뒤 (환경 step 끝 상태 · LowRl · LowEp) — combine_run.cpp:1326-1371
  void after_step(const sim::EnvState& s, const sim::LowRl& rl, const sim::LowEp& L) {
    if (rl.stall && !sig_prev_) {
      ++n_stall_;
      avoid_add(rl.stall_tgt, s.step);
      fmem_ = fu::Mem();
      if (p_.avoid) redo_ = true;
    }
    sig_prev_ = rl.stall;
    if (cmd_.follow != 0) {   // RL follow 2 · 규칙 + fuel follow 1 (combine_run.cpp:1354)
      const bool reached = L.reach_step == s.step && L.reached > 0;
      const bool off = !reached && L.hold != 0;
      const bool nopath = !reached && !off && s.step > last_dec_ && (s.goal_cell != cmd_.target || s.p_n <= 0);
      if (reached) ++n_reach_;
      if (reached || off || nopath) fmem_ = fu::Mem();
      if (p_.redecide && (reached || off || nopath)) {
        if (off) avoid_add(cmd_.target, s.step);
        if (nopath) avoid_add(cmd_.target, s.step);
        redo_ = true;
      }
    }
  }
  // 판을 새로 (실물: 노드를 다시 켤 때) — 피하는 칸·기억·명령을 비움
  void restart() { *this = Ctl(p_); }
  void force_redo() { redo_ = true; }   // 실물: 다시 켤 때 바로 결정 (시뮬에는 없음 — 판 중간에 멈추지 않으므로)
  const sim::LowCmd& cmd() const { return cmd_; }
  int slot() const { return slot_; }   // 준비 단계 행동 칸 (규칙: 고른 후보 번호 · fuel 이면 0 · 없으면 −1. RL 은 늘 −1 — 정책 환경은 계획 칸 0)
  long n_dec() const { return n_dec_; }
  long n_reach() const { return n_reach_; }
  long n_stall() const { return n_stall_; }
  const fu::Mem& mem() const { return fmem_; }
  bool avoided(int cell, int step) const {
    if (!p_.avoid || cell < 0) return false;
    const float q = p_.bl_r / res_;
    for (int k = 0; k < kBl; ++k) {
      if (av_cell_[k] < 0 || step >= av_until_[k]) continue;
      const int dx = cell % gw_ - av_cell_[k] % gw_, dy = cell / gw_ - av_cell_[k] / gw_;
      if ((float)(dx * dx + dy * dy) <= q * q) return true;
    }
    return false;
  }
  void set_geom(int gw, float res) { gw_ = gw; res_ = res; }   // 칸 번호 → 칸 좌표 (지도 창 너비 · 칸 크기)

 private:
  static constexpr int kBl = 8;
  Params p_;
  int av_cell_[kBl], av_until_[kBl], av_n_ = 0;
  int gw_ = 1;
  float res_ = 0.05f;
  bool redo_ = true;   // 첫 걸음에 결정 (combine_run.cpp:331 redo(N, 1))
  int sig_prev_ = 0, last_dec_ = 0, sw_step_ = 0;
  long n_dec_ = 0, n_reach_ = 0, n_stall_ = 0;
  sim::LowCmd cmd_{-1, 0};
  int slot_ = -1;
  fu::Mem fmem_;
  void avoid_add(int cell, int step) {
    if (!p_.avoid || cell < 0) return;
    av_cell_[av_n_ % kBl] = cell;
    av_until_[av_n_ % kBl] = step + p_.bl_steps;
    av_n_ += 1;
  }
};

}  // namespace lxc
