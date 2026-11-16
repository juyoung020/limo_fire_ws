// 합친 탐사(위층 fuel · slot0 · 판단 모델 + 아래층 규칙 추종기)의 위층 결정·목표 넘기기 — 로봇 하나 몫. 실물 노드(limo_fire_rl low_rl)와
// 그 확인 도구(sim/tools/low_explore_check · low_explore_replay · room_run)가 같은 코드로 돌린다. combine_run.cpp 의 환경 하나 몫을 그대로 옮긴 것:
//   피하는 칸 고리 kBl 8 · 반경 rec_bl_r · rec_bl_steps 걸음 (combine_run Avoid · avoided · avoid_add)
//   결정할 때: 걸음 % decide_steps == 0 또는 redo · 피하지 않는 후보 차례
//   fuel 이 목표 칸을 냄 (로봇 지도·지나간 길·깊이 층·지도 자세 — scripts/fuel_upper.h fu::decide)
//   목표 넘기기: fuel 칸 → LowCmd{칸, 1} · 행동 칸 0 (목표 칸을 그대로 따라감). 그 밖(slot0 · 판단 모델 · fuel 이 못 냄) → LowCmd{−1, 0} ·
//                행동 칸 = 고른 후보 번호(피하지 않는 첫 후보 — 없으면 −1). fuel 이 "시점 자격 되는 덩어리 없음"(fu::kNone)이면 목표 없음(행동 칸 −1)
//   행동 칸은 다음 결정까지 그대로 (combine_run 의 act 배열) — slot() 으로 꺼내 준비 단계(경로 계획 후보)에 넘김
//   걸음 뒤: 막힘 신호(LowRl::stall 올라감) → 그 칸 피함 · fuel 기억 비움 · 다시 결정
//            follow 1 이면 통과(reach_step) · 경로 끝 밖(hold) · 경로 없음 → fuel 기억 비움 · (경로 끝 밖·경로 없음은 그 칸 피함) · 다시 결정
// 판 끝(닫힘 + 진척 없음 10 s 성공 · 진척 없음 · 시간 · 닿음)은 환경(finish_step — low_explore 합친 탐사 규칙)이 정하고 부르는 쪽이 읽는다.
// 행동 칸이 −1 인 걸음은 두 쪽 다 시뮬 규칙 추종기의 제자리 회전 (env.h apply_action 계층형 가지).
// (2026-11-16 아래층 RL 을 그만두며 RL 목표 넘기기 가지 — follow 2 · stick — 를 뺐다. archive/rl/README.md)
#pragma once
#include <cmath>
#include <cstdint>
#include <vector>

#include "fuel_upper.h"
#include "low_task.h"
#include "cand_view.h"

namespace lxc {

struct Params {
  int decide_steps = 10;    // EnvCfg::decide_steps (1 s)
  float bl_r = 0.5f;        // 피하는 칸 반경 (m) — EnvCfg::rec_bl_r (0 이면 0.5)
  int bl_steps = 100;       // 피하는 시간 (걸음) — EnvCfg::rec_bl_steps (0 이면 10 s)
  bool avoid = true, redecide = true, stick = true;   // combine_run 기본 (--no-avoid · --no-redecide · --no-stick 로 끔)
  bool use_fuel = true;     // false = combine_run --upper slot0 (fuel 을 안 부르고 늘 아래 slot0 차례 — 피하지 않는 첫 후보)
  fu::Opt fuel;             // --fuel-* (부르는 쪽이 채움)
  bool use_cand = false;    // 위층 후보 = scripts/cand_view.h (서는 자리 ≤ 8 개, 안정 번호). decide 는 후보 고르기(cand_pick)로, 판단 모델은 build_cands → go_cand
  cand::Opt cand;
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
  // 이번 걸음에 결정하나
  bool due(const sim::EnvState& s) const { return s.step % p_.decide_steps == 0 || redo_; }
  // 결정 → 아래층 명령. why: 0 fuel · 1 slot0 (fuel 대신 또는 use_fuel 끔) · 2 목표 없음
  sim::LowCmd decide(const View& v, fu::Work& work, int* why = nullptr) {
    const sim::EnvState& s = *v.s;
    if (p_.use_cand) { build_cands(v); return go_cand(v, -1, why); }
    const int K = mdp::hier::K;
    int pick = -1;   // slot0: 피하지 않는 첫 후보
    for (int j = 0; j < K && pick < 0; ++j) {
      const int cell = v.cand[j];
      if (cell >= 0 && !avoided(cell, s.step)) pick = j;
    }
    fu::In in{v.c, v.W, v.lmap, v.trail, v.cam, v.s, v.cand, [this, &s](int cell) { return avoided(cell, s.step); }};
    in.diag = diag_;
    const int tcell = p_.use_fuel ? fu::decide(in, p_.fuel, fmem_, work) : -1;
    redo_ = false;
    last_dec_ = s.step;
    ++n_dec_;
    int w = 1;
    if (tcell == fu::kNone) {   // 프론티어는 있으나 시점 자격 되는 덩어리 없음: 목표 없음 (combine_run 과 같음 — slot0 로 안 떨어짐)
      cmd_ = sim::LowCmd{-1, 0};
      w = 2;
      slot_ = -1;
    } else if (tcell >= 0) {   // fuel: 목표 칸을 그대로 따라감
      w = 0;
      cmd_ = sim::LowCmd{tcell, 1};
      slot_ = 0;
    } else {                   // slot0 차례: 후보 번호 그대로
      cmd_ = sim::LowCmd{-1, 0};
      slot_ = pick;
      w = pick >= 0 ? 1 : 2;
    }
    if (why) *why = w;
    return cmd_;
  }
  // 위층 판단 모델 (upper decider — low_decider.h): 후보 want 를 고른 결정. why: 0 판단 모델 후보 · 1 slot0 차례 · 2 목표 없음. want < 0 이거나 피하는 칸 · 빈 칸이면 slot0 차례(피하지 않는 첫 후보).
  // 목표 넘기기는 decide 의 slot0 가지 그대로 (판단 모델 · slot0 위층이 같은 길)
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
    cmd_ = sim::LowCmd{-1, 0};
    slot_ = pick;
    if (why) *why = pick >= 0 ? 1 : 2;   // 판단 모델 후보를 따른 결정도 1 (RL 을 빼기 전 규칙 가지 그대로 — 0 은 안 냄)
    return cmd_;
  }
  // ---- 후보 모드 (use_cand — scripts/cand_view.h)
  // 결정 때 후보를 만듦 (판단 모델은 이 목록을 보고 go_cand 로 고름). 기억(cand_mem)은 판 동안 이어짐
  const cand::Out& build_cands(const View& v) {
    const sim::EnvState& s = *v.s;
    const cand::In in{v.c, v.W, v.lmap, v.trail, v.cam, v.s, [this, &s](int cell) { return avoided(cell, s.step); }, cmd_.follow != 0 ? cmd_.target : -1};
    cout_ = cand::build(in, p_.cand, cmem_, cwork_);
    return cout_;
  }
  // 후보 idx 로 감 (−1 = 규칙: 지금 목표 번호가 목록에 있고 점수가 0 번의 keep 배 이상이면 그대로, 아니면 0 번). why: 0 후보로 감 · 2 목표 없음
  sim::LowCmd go_cand(const View& v, int idx, int* why = nullptr) {
    const sim::EnvState& s = *v.s;
    redo_ = false;
    last_dec_ = s.step;
    ++n_dec_;
    const int n = (int)cout_.c.size();
    if (idx < 0 || idx >= n) idx = cand_pick();
    if (idx < 0) {
      cmd_ = sim::LowCmd{-1, 0};
      slot_ = -1;
      if (why) *why = 2;
      return cmd_;
    }
    const cand::Cand& c = cout_.c[idx];
    cmem_.note_target(c.id, s.step, v.W->ox + (c.vp % v.W->gw + 0.5f) * v.c->res, v.W->oy + (c.vp / v.W->gw + 0.5f) * v.c->res);
    cur_id_ = c.id;
    cmd_ = sim::LowCmd{c.vp, 1};   // 규칙 추종기: 목표 칸 그대로 (fuel 과 같은 길)
    slot_ = 0;
    if (why) *why = 0;
    return cmd_;
  }
  int cand_pick() const {
    if (cout_.c.empty()) return -1;
    if (cur_id_ >= 0 && cmd_.follow != 0)
      for (int i = 0; i < (int)cout_.c.size(); ++i)
        if (cout_.c[i].id == cur_id_ && !cout_.c[i].in_avoid && cout_.c[i].score >= p_.cand.keep * cout_.c[0].score) return i;
    return 0;
  }
  int cand_index(int id) const { for (int i = 0; i < (int)cout_.c.size(); ++i) if (cout_.c[i].id == id) return i; return -1; }
  const cand::Out& cands() const { return cout_; }
  const cand::Mem& cand_mem() const { return cmem_; }
  int cur_id() const { return cur_id_; }
  // 지금 목표가 살아 있나 (고른 후보로 가는 경로가 있고 피하는 칸이 아님)
  bool live(const sim::EnvState& s) const { return slot_ >= 0 && s.goal_cell >= 0 && s.p_n > 0 && !avoided(s.goal_cell, s.step); }
  bool redo_pending() const { return redo_; }   // 막힘 · 통과 · 경로 끝 밖 · 경로 없음 뒤 바로 다시 결정해야 함
  // 걸음 뒤 (환경 step 끝 상태 · LowRl · LowEp)
  void after_step(const sim::EnvState& s, const sim::LowRl& rl, const sim::LowEp& L) {
    if (p_.use_cand) cmem_.note_pose(s.step, s.x + s.ex, s.y + s.ey, s.th + s.eth);
    if (rl.stall && !sig_prev_) {
      if (p_.use_cand && cmd_.follow != 0) cmem_.note_end(cur_id_, s.step, cand::kStall);
      ++n_stall_;
      avoid_add(rl.stall_tgt, s.step);
      fmem_ = fu::Mem();
      if (p_.avoid) redo_ = true;
    }
    sig_prev_ = rl.stall;
    if (cmd_.follow != 0) {   // fuel follow 1
      const bool reached = L.reach_step == s.step && L.reached > 0;
      const bool off = !reached && L.hold != 0;
      const bool nopath = !reached && !off && s.step > last_dec_ && (s.goal_cell != cmd_.target || s.p_n <= 0);
      if (reached) ++n_reach_;
      if (p_.use_cand) {
        if (reached) cmem_.note_end(cur_id_, s.step, cand::kReached);
        else if (off) cmem_.note_end(cur_id_, s.step, cand::kOffPath);
        else if (nopath) cmem_.note_end(cur_id_, s.step, cand::kNoPath);
      }
      if (reached || off || nopath) fmem_ = fu::Mem();
      if (p_.redecide && (reached || off || nopath)) {
        if (off) avoid_add(cmd_.target, s.step);
        if (nopath) avoid_add(cmd_.target, s.step);
        redo_ = true;
      }
    }
  }
  // 판을 새로 (실물: 노드를 다시 켤 때) — 피하는 칸·기억·명령을 비움
  void restart() { fu::Diag* d = diag_; *this = Ctl(p_); diag_ = d; }
  void force_redo() { redo_ = true; }   // 실물: 다시 켤 때 바로 결정 (시뮬에는 없음 — 판 중간에 멈추지 않으므로)
  const sim::LowCmd& cmd() const { return cmd_; }
  int slot() const { return slot_; }   // 준비 단계 행동 칸 (고른 후보 번호 · fuel 이면 0 · 없으면 −1)
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
  void set_diag(fu::Diag* d) { diag_ = d; }   // 결정 속 적기 (재기만 — 다시 돌리기 도구)
  void set_geom(int gw, float res) { gw_ = gw; res_ = res; }   // 칸 번호 → 칸 좌표 (지도 창 너비 · 칸 크기)

 private:
  static constexpr int kBl = 8;
  Params p_;
  int av_cell_[kBl], av_until_[kBl], av_n_ = 0;
  int gw_ = 1;
  float res_ = 0.05f;
  bool redo_ = true;   // 첫 걸음에 결정 (combine_run redo(N, 1))
  int sig_prev_ = 0, last_dec_ = 0;
  long n_dec_ = 0, n_reach_ = 0, n_stall_ = 0;
  sim::LowCmd cmd_{-1, 0};
  int slot_ = -1;
  fu::Mem fmem_;
  fu::Diag* diag_ = nullptr;
  cand::Mem cmem_;
  cand::Work cwork_;
  cand::Out cout_;
  int cur_id_ = -1;
  void avoid_add(int cell, int step) {
    if (!p_.avoid || cell < 0) return;
    av_cell_[av_n_ % kBl] = cell;
    av_until_[av_n_ % kBl] = step + p_.bl_steps;
    av_n_ += 1;
  }
};

}  // namespace lxc
