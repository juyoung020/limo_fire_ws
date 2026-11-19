// 합친 탐사(위층 fuel · slot0 · 판단 모델 + 아래층 규칙 추종기)의 위층 결정·목표 넘기기 — 로봇 하나 몫. 실물 노드(limo_fire_explore explorer)와
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
  int rough_steps = 100;    // 후보 모드 "대충 다 앎" 끝: 후보 0 개이고 남은 덩어리가 모두 구멍인 상태가 이 걸음 수 이어지면 (EnvCfg::close_idle_s — 닫힘 끝과 같은 10 s)
  // 목표에 다가가지 못함: 같은 목표(fail_m 안)를 따라가는 동안 남은 경로 길이가 approach_m 넘게 줄지 않고 approach_steps 걸음이 지나면 실패(시간 넘김)로 보고
  // 그 칸 피함 · 다시 결정. 막힘 신호(정체 — 0.3 m 안에 10 s)와 달리 움직이는데 목표에 못 닿는 것: 촘촘한 깊이 층(layer_grid)에서 두 갈래 길이 번갈아 막혀
  // 복도를 오가던 것 (hssd_103997718 씨앗 4 — sim/analysis/real_limo.md "층 전용 격자"). Nav2 progress_checker(움직임만 봄)에 목표 쪽 진척을 더한 것. 0 = 끔.
  int approach_steps = 200;   // 20 s (잠정)
  float approach_m = 0.5f;    // m (Nav2 progress_checker required_movement_radius 와 같은 값)
  int cand_commit = 0;      // 후보 0 번 규칙의 붙들기: 0 = 고른 차례 점수(겹침 뺀 값)로 keep 배 · 1 = 겹침 안 뺀 점수(넓이/시간)로 keep 배 (FUEL 처럼 새 것이 margin 넘게 좋을 때만 바꿈)
  int blk_abort_steps = 0;  // 지금 목표로 가며 충돌 검사 막힘(kLowImm) 걸음이 이만큼이면 그 목표 실패 · 피함 · 다시 결정 (EnvCfg::rec_abort — env.yaml recovery.goal_abort, 0 = 끔)
  bool use_cand = false;    // 위층 후보 = scripts/cand_view.h (서는 자리 ≤ 8 개, 안정 번호). decide 는 후보 고르기(cand_pick)로, 판단 모델은 build_cands → go_cand
  cand::Opt cand;
  static Params from_cfg(const sim::EnvCfg& c) {
    Params p;
    p.decide_steps = c.decide_steps;
    const double dt = (double)c.substeps * c.dt;
    p.bl_r = c.rec_bl_r > 0.f ? c.rec_bl_r : 0.5f;
    p.bl_steps = c.rec_bl_steps > 0 ? c.rec_bl_steps : (int)std::lround(10.0 / dt);
    p.rough_steps = c.rough_wait > 0 ? c.rough_wait : std::max(1, (int)std::lround(c.close_idle_s / dt));
    p.approach_steps = (int)std::lround(20.0 / dt);
    p.blk_abort_steps = c.rec_abort;
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
  const sim::ReachKeep* reach = nullptr;   // 같은 걸음 관측의 프론티어 찾기 (RobotEnv::reach_keep — 후보 뽑기가 같은 입력이면 다시 안 찾음)
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
    if (cand_step_ == s.step) return cout_;   // 같은 걸음에 다시 부르면 그대로 (번호 나이 age 가 부른 수로 늘지 않게 · 시간 아낌)
    cand_step_ = s.step;
    cand::In in{v.c, v.W, v.lmap, v.trail, v.cam, v.s, [this, &s](int cell) { return avoided(cell, s.step); }, cmd_.follow != 0 ? cmd_.target : -1};
    in.reach = v.reach;
    in.avoid_s = (float)(p_.bl_steps * (double)v.c->substeps * v.c->dt);   // 실패 뒤 피하는 시간과 같게
    cout_ = cand::build(in, p_.cand, cmem_, cwork_);
    // 대충 다 앎 판정: 후보 0 · 남은 덩어리 모두 구멍(대충 아는 굵은 칸 안)이거나 버림(같은 자리 실패 give_up 번) — 덩어리 0 개도
    if (cout_.c.empty() && cout_.start_free && cout_.n_pocket + cout_.n_gaveup == cout_.n_dormant) { if (rough_since_ < 0) rough_since_ = s.step; }
    else rough_since_ = -1;
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
    if (c.recent_fail) ++n_refail_;   // 최근 실패한 후보로 감 (판단 모델이 골랐을 때만 생겨야 — 규칙은 안 고름)
    cmem_.note_target(c.id, s.step, v.W->ox + (c.vp % v.W->gw + 0.5f) * v.c->res, v.W->oy + (c.vp / v.W->gw + 0.5f) * v.c->res);
    if (c.id != cur_id_ || cmd_.follow == 0) blk_n_ = 0;   // 목표가 바뀌면 막힘 셈을 새로
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
        if (cout_.c[i].id == cur_id_ && !cout_.c[i].in_avoid && !cout_.c[i].recent_fail) {
          if (p_.cand_commit == 0) { if (cout_.c[i].score >= p_.cand.keep * cout_.c[0].score) return i; break; }
          // 1: 겹침을 안 뺀 넓이/시간 — 지금 목표가 차례에서 밀려 겹침 몫이 빠져도 바꾸지 않게. 가장 좋은 것보다 1/keep 배 넘게 나쁠 때만 바꿈
          const auto full = [&](const cand::Cand& c) { return c.gain_m2 / (c.t_s + p_.cand.t0); };
          float best = 0.f;
          for (const cand::Cand& c : cout_.c) best = std::max(best, full(c));
          if (full(cout_.c[i]) >= p_.cand.keep * best) return i;
          break;
        }
    // 규칙(대신 고르기)은 최근 실패·피하는 칸 후보를 안 고름 — 목록에서 그런 것은 맨 뒤라 앞에서 처음 나오는 성한 것. 모두 실패면 목표 없음 (제자리 회전)
    const auto ok = [&](const cand::Cand& c) { return !c.recent_fail && !c.in_avoid; };
    if (p_.cand_commit == 1) {   // 바꿀 때도 겹침 안 뺀 넓이/시간이 가장 큰 것
      int bi = -1;
      for (int i = 0; i < (int)cout_.c.size(); ++i)
        if (ok(cout_.c[i]) && (bi < 0 || cout_.c[i].gain_m2 / (cout_.c[i].t_s + p_.cand.t0) > cout_.c[bi].gain_m2 / (cout_.c[bi].t_s + p_.cand.t0))) bi = i;
      return bi;
    }
    for (int i = 0; i < (int)cout_.c.size(); ++i) if (ok(cout_.c[i])) return i;
    return -1;
  }
  int cand_index(int id) const { for (int i = 0; i < (int)cout_.c.size(); ++i) if (cout_.c[i].id == id) return i; return -1; }
  const cand::Out& cands() const { return cout_; }
  const cand::Mem& cand_mem() const { return cmem_; }
  int cur_id() const { return cur_id_; }
  long n_refail() const { return n_refail_; }   // 최근 실패 후보로 간 수 (재기 — low_explore_check)
  // 대충 다 앎 (사용자 목표 — 공간을 대충 빨리 앎): 후보 0 개이고 남은 덩어리가 모두 구멍인 상태가 rough_steps 이어짐. 노드·확인 도구가 판 끝으로 씀 (시뮬 == 실물 같은 판정)
  bool rough_done(int step) const { return p_.use_cand && rough_since_ >= 0 && step - rough_since_ >= p_.rough_steps; }
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
    if (p_.blk_abort_steps > 0 && cmd_.follow != 0 && !redo_) {   // 충돌 검사에 오래 막힘 → 그 목표 버림 (blk_abort_steps)
      if (L.kind == sim::kLowImm) ++blk_n_;
      if (blk_n_ >= p_.blk_abort_steps) {
        if (p_.use_cand) cmem_.note_end(cur_id_, s.step, cand::kStall);
        ++n_blk_;
        avoid_add(cmd_.target, s.step);
        fmem_ = fu::Mem();
        redo_ = true;
        blk_n_ = 0;
      }
    }
    const bool far_ = approach_check(s);   // 목표에 다가가지 못함 (approach_steps)
    if (far_) {
      if (p_.use_cand) cmem_.note_end(cur_id_, s.step, cand::kTimeout);
      ++n_far_;
      avoid_add(cmd_.target, s.step);
      fmem_ = fu::Mem();
      redo_ = true;
    }
    if (cmd_.follow != 0 && !far_) {   // fuel follow 1
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
  // 위층 밖에서 정한 목표 칸을 그대로 (시험 도구 room_run --follow — 실물 궤적 따라가기). LowCmd{칸, 1} 이면 규칙 추종기 목표 칸 길(fuel 과 같음), 아니면 목표 없음
  void force_cmd(const sim::LowCmd& c) { cmd_ = c; slot_ = c.target >= 0 ? 0 : -1; }
  void restart() { fu::Diag* d = diag_; *this = Ctl(p_); diag_ = d; }
  void force_redo() { redo_ = true; }   // 실물: 다시 켤 때 바로 결정 (시뮬에는 없음 — 판 중간에 멈추지 않으므로)
  const sim::LowCmd& cmd() const { return cmd_; }
  int slot() const { return slot_; }   // 준비 단계 행동 칸 (고른 후보 번호 · fuel 이면 0 · 없으면 −1)
  long n_dec() const { return n_dec_; }
  long n_reach() const { return n_reach_; }
  long n_stall() const { return n_stall_; }
  long n_far() const { return n_far_; }   // 목표에 다가가지 못해 버린 수 (approach_steps)
  long n_blk() const { return n_blk_; }   // 충돌 검사 막힘으로 버린 목표 수 (blk_abort_steps)
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
  long n_refail_ = 0;
  int cand_step_ = -1;   // build_cands 를 마지막으로 부른 걸음
  int rough_since_ = -1;   // 후보 0 · 모두 구멍인 결정이 이어진 첫 걸음 (아니면 −1)
  int ap_tgt_ = -1, ap_step_ = 0;   // 다가감 재기: 기준 목표 칸 · 남은 길이가 마지막으로 줄어든 걸음
  float ap_best_ = 0.f;             // 그때 남은 경로 길이 (m)
  long n_far_ = 0;
  int blk_n_ = 0;     // 지금 목표로 가며 충돌 검사에 막힌 걸음
  long n_blk_ = 0;
  // 목표에 다가가지 못함: 목표가 fail_m(1 m) 넘게 옮기면 새로 잼. 남은 길이 = 지금 구간 다음 경로 점들 사이 길이 합 (칸 × 칸 크기) — 경로가 있을 때만 줄었나 봄
  bool approach_check(const sim::EnvState& s) {
    if (p_.approach_steps <= 0 || cmd_.follow == 0 || cmd_.target < 0) { ap_tgt_ = -1; return false; }
    if (ap_tgt_ < 0 || std::hypot((float)(cmd_.target % gw_ - ap_tgt_ % gw_), (float)(cmd_.target / gw_ - ap_tgt_ / gw_)) * res_ > 1.0f) {
      ap_tgt_ = cmd_.target;
      ap_step_ = s.step;
      ap_best_ = 3e38f;
    }
    if (s.goal_cell == cmd_.target && s.p_n > 0) {
      float rem = 0.f;
      for (int j = std::max(0, s.p_i); j + 1 < s.p_n; ++j)
        rem += std::hypot((float)(s.wp[2 * j + 2] - s.wp[2 * j]), (float)(s.wp[2 * j + 3] - s.wp[2 * j + 1]));
      rem *= res_;
      if (rem < ap_best_ - p_.approach_m || ap_best_ == 3e38f) { ap_best_ = rem; ap_step_ = s.step; }
    }
    if (s.step - ap_step_ < p_.approach_steps) return false;
    ap_tgt_ = -1;   // 버린 뒤 새 목표에서 새로
    return true;
  }
  void avoid_add(int cell, int step) {
    if (!p_.avoid || cell < 0) return;
    av_cell_[av_n_ % kBl] = cell;
    av_until_[av_n_ % kBl] = step + p_.bl_steps;
    av_n_ += 1;
  }
};

}  // namespace lxc
