// 합친 탐사의 위층을 판단 모델(llama-server /v1/systemone, zero-shot)에 맡기는 길 — 느린 주기 · 비동기. 실물 노드(limo_fire_explore explorer upper:=decider)와
// 시뮬(low_explore_check --upper decider --dec-period · --dec-latency, dec_rollout)이 같은 코드로 돈다 (sim = real).
//
// 요청 (render_cand): 후보 = scripts/cand_view.h (Ctl::build_cands — 서는 자리, 안정 번호), 글자 A … = 후보 차례(지시가 있으면 물체 후보가 앞 글자),
//   지도 사진 = 로봇 자기 지도 + 글자 동그라미 + 지나온 길 · 지난 목표. 옛 hier 8 칸 후보 요청(관측 1D 후보 특징)은 2026-11-17 지움.
// 걸음마다 (Upper::step, 걸음 = 라이다 한 장 0.1 s):
//   1. 답이 왔으면: 보낸 지 timeout_steps 안이고 답 꼴이 맞으면, 그 안정 번호의 지금 후보(없으면 서는 자리 match_r 안)로 → Ctl::go_cand (못 찾으면 다시 물음)
//   2. 보낸 지 timeout_steps 넘도록 답이 없으면 버림 (지금 목표 그대로, 늦게 온 답은 번호가 달라 버림)
//   3. 지금 목표가 죽었으면: 답이 오는 중이면 (지연 어림 + hold_extra) 걸음까지 기다리고, 아니면 바로 묻고, 답 실패가 fail_max 번 잇따르면 규칙(cand_pick)이 바로 대신
//      — 그 밖의 1 s 결정 차례에는 목표를 바꾸지 않는다 (판단 모델 답이 올 때만 바꿈 — fuel 의 매초 목표 바뀜을 피함)
//   4. 묻는 때: event (기본 — 다가감 · 실패 · 후보 크게 바뀜 · 최대 간격) 또는 periodic (period_steps 마다 — 시뮬 비교용)
// 보내기·받기는 Transport 가 맡는다: 실물 = 비차단 소켓 HTTP — poll 이 한 단계씩 (limo_fire_explore/decider_client.h, 스레드 없음), 시뮬 = 정해진 걸음 뒤에 답을 내놓는 줄 (low_explore_check · dec_rollout).
#pragma once
#include <algorithm>
#include <cmath>
#include <deque>
#include <map>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "decider.h"
#include "low_explore_ctl.h"

namespace lxd {

struct Params {
  int period_steps = 30;    // 요청 주기 (걸음 — 3 s)
  int timeout_steps = 80;   // 답 기다림 한도 (걸음 — 8 s)
  int px = 288;             // 지도 사진 한 변 (px) — 288 이면 그림 약 81 토큰 (LFM2-VL: (px/32)²). 로봇 지연 표 upper_lora/README "요청 꼴과 지연"
  int input = dec::kBoth;   // dec::kText · kImage · kBoth
  bool hint = true;         // 후보 모드 글의 "Rules of thumb" (가까운 쪽으로 쏠리게 함 — 평가에서 빼고도 잼)
  bool mem_image = true;    // 후보 모드 지도 사진에 지나온 길 · 지난 목표 표시 (꼴 A)
  int mode = 0;             // 0 = periodic (period_steps 마다) · 1 = event: 목표에 다가감(남은 길 ≤ approach_m — 미리 물어 닿기 전에 답) · 목표 실패 · 후보가 크게 바뀜
                            //   (지금 목표 덩어리가 사라짐 · 점수가 switch_ratio 배 넘는 새 1 순위) · 마지막으로 보낸 지 max_steps 넘음 일 때만 물음 (후보 모드)
  float approach_m = 1.0f;
  // 답이 늦어 규칙이 대신 고르는 것 줄이기 (2026-11-17 — 시뮬 지연 1.6 s 에서 대신 고른 까닭의 대부분이 "답이 오기 전에 목표에 닿음"):
  float ask_lead_s = 1.0f;  // event: 남은 길 / ask_v ≤ 답 지연(걸음으로 잰 것의 이동 평균) + 이것 이면 미리 물음 (approach_m 과 함께 — 먼저 닿는 쪽)
  float ask_v = 0.98f;      // m/s — 남은 길 → 남은 시간 (fuel · cand 가는 시간 v_eff 와 같은 잰 값)
  int delay0_steps = 20;    // 답 지연 첫 어림 (걸음 — 2 s, 로봇 Orin v1 꼴 1.5–2.2 s)
  int fail_max = 3;         // 답 실패(시간 끝 · 꼴 틀림 · 후보 사라짐)가 이만큼 잇따르면 기다리지 않고 규칙이 바로 대신 (서버 꺼짐에 매달리지 않게) — 답이 하나 맞으면 0 으로
  int hold_extra = 5;       // 목표가 죽었는데 답이 오는 중이면 (보낸 뒤 지연 어림 + 이것) 걸음까지 기다림 — 규칙이 대신 고르지 않음. < 0 = 바로 대신 (옛 동작)
  float switch_ratio = 1.5f;
  int max_steps = 100;      // event 모드 최대 간격 (걸음 — 10 s)
  int log_k = 4;
  bool compact = true;      // 후보 모드 글: 짧은 키 10 개 (t · len · bear · new · w · room · cur · fail · fail_s · seen) · 힌트를 후보 앞 · 최근 목표 목록 뺌 — 기본 (로봇 약 2 s, 미세조정 v1 꼴). 끄면 긴 꼴
  bool mem_fields = true;   // 후보 모드 글: 기억 필드(고른·실패 수 · 까닭 · 실패 뒤 초 · 실패 자리 · 피하는 곳 · 가 본 곳)
  int max_cands = 6;        // 후보 모드: 요청에 넣는 후보 수 상한 (0 = 다). 기본 6 — 위층 비교(upper_lora/README, 빌드 1f275bf)에서 AUC180 이 후보 다 보다 +0.026, 토큰도 적음
  int order = 0;            // 후보 모드 글자 차례: 0 = 후보 점수 순 (cand::build) · 1 = 길 거리 순            // 후보 모드 결정 기록: 최근 K 개 결정(언제 · 고른 번호의 지금 글자 · 이어 감/바꿈 · 결과)을 글로, 그 목표 자리를 지도에 번호(1 = 가장 최근)로 — 0 = 끔
  int image = 0;            // 후보 모드 사진 꼴: 0 = A (지금 지도 한 장) · 1 = B (2 × 2 칸: 지금 · 5 · 15 · 30 s 전, 같은 범위 — 토큰 수 같음, 칸마다 해상도 반)
  // 사용자 지시 (실물 노드 ~/instruction — 아무 글: "가장 먼 의자로", "사람 따라가", "탐사해", "TV 옆으로"). 비면 글이 예전과 바이트까지 같음 —
  // 차면 지시를 그대로 "instruction" 칸에, 물체 후보(아래)·"known_objects" 를 넣고 질문을 "지시를 가장 잘 따르는 후보"로. 무엇을 할지는 판단 모델이 고름
  std::string instruction;
  std::string class_hint;   // 선택: 지시에서 풀어 낸 클래스 (object_goto 별칭·설명 검색). 그 클래스 물체를 물체 후보 칸 앞에 — 비어도 됨 (모든 클래스가 후보)
  int obj_max = 3;          // 물체 후보 칸 수 (따라가기 후보 포함, 합쳐 max_cands — 프론티어는 남은 칸)
  bool obj_always = false;  // 지시가 없어도 물체 후보를 넣음 (기본 끔 — 지시 없는 탐사 요청은 예전과 바이트 같음)
  float vis_s = 1.0f;       // 이 안에 본 물체 = 지금 보임 (vis)
  std::vector<std::string> known_objects;   // "cup 1.2 m ahead-left" 꼴 짧은 글 (노드가 물체 지도에서 만듦)
  // 물체 후보 (지시가 있을 때만): 지시에 맞는 지도 위 물체의 설 자리. 후보 목록 앞 칸에 들어가 프론티어 후보와 섞임 (합쳐 max_cands)
  struct Obj {
    std::string oid, cls;   // "chair-1" · "chair"
    float x = 0, y = 0;     // 물체 자리 (지도 m)
    float seen_s = 0;       // 마지막으로 본 뒤 (s)
    float match = 0;        // 1 = class_hint 클래스 (물체 후보 칸 앞에)
    bool follow = false;    // 따라가기 후보: 지금 보이는 사람 (노드가 실시간 탐지에서) — 고르면 노드가 그 사람을 약 1 m 뒤에서 따라감
    int vp = -1;            // 설 자리 칸 (build_cand 가 채움)
    float len_m = 0;        // 로봇에서 설 자리까지 길 (m, 빈 칸 BFS)
  };
  std::vector<Obj> objects;
  float obj_ring_in = 0.6f, obj_ring_out = 1.2f, obj_clear = 0.25f;   // 설 자리 고리 (물체에서 m) · 둘레 빈 칸 반경 (m) — object_goto 와 같은 뜻
  float match_r = 1.5f;     // 답의 후보 안정 번호가 사라졌을 때 그 서는 자리 → 지금 후보 (m)
};

// 보낸 요청 하나
struct Req {
  long id = 0;
  int step = 0;
  std::string body;
  std::vector<int> cells;   // 글자 차례의 후보 서는 자리 칸
  std::vector<int> ids;     // 글자 차례의 안정 번호
  std::string why;          // 묻는 까닭 (period · start · goal failed or lost · approaching goal · current goal's frontier gone · much better candidate · max interval)
};
// 위층 결정 하나 (후보 모드 — 판단 모델 답을 따른 것 · 규칙으로 대신 고른 것 모두)
struct Dec {
  int step;
  int id;          // 고른 후보 안정 번호 (−1 = 목표 없음)
  float x, y;      // 그 서는 자리 (지도 m)
  int kept;        // 1 = 지금 목표를 이어 감 · 0 = 바꿈
  int src;         // 0 = 판단 모델 답 · 1 = 규칙 (목표가 죽어 대신)
};
struct Answer {
  long id = 0;
  bool ok = false;
  std::string choice;   // 고른 글자 ("A" …)
  std::string err;
  double ms = 0;
};
class Transport {
 public:
  virtual ~Transport() = default;
  virtual void send(const Req& r) = 0;    // 기다리지 않음
  virtual bool poll(Answer* a) = 0;       // 와 있는 답 하나 (없으면 false) — 기다리지 않음
};

// 답 고르기: answers.pick.choice (llama-server /v1/systemone 의 choice 답)
inline Answer parse_answer(long id, const std::string& raw) {
  Answer a;
  a.id = id;
  const auto j = nlohmann::json::parse(raw, nullptr, false);
  if (!j.is_object() || !j.contains("answers") || !j["answers"].is_object() || !j["answers"].contains("pick") || !j["answers"]["pick"].is_object() ||
      !j["answers"]["pick"].contains("choice") || !j["answers"]["pick"]["choice"].is_string()) {
    a.err = "answer shape";
    return a;
  }
  a.choice = j["answers"]["pick"]["choice"].get<std::string>();
  a.ok = a.choice.size() == 1 && a.choice[0] >= 'A' && a.choice[0] <= 'Z';
  if (!a.ok) a.err = "unknown option " + a.choice;
  return a;
}

class Upper {
 public:
  explicit Upper(const Params& p) : p_(p) {}
  // 걸음 하나. 목표가 바뀌었으면 true 와 *cmd
  bool step(const lxc::View& v, lxc::Ctl& ctl, Transport& tr, sim::LowCmd* cmd, int* why = nullptr) {
    const sim::EnvState& s = *v.s;
    if (p_.image == 1 && v.lmap && s.step % 10 == 0) {   // 꼴 B: 10 걸음마다 지도를 남김 (최근 31 장 = 30 s)
      hist_.emplace_back(s.step, std::vector<int16_t>(v.lmap, v.lmap + (size_t)v.c->size * v.c->size));
      if (hist_.size() > 31) hist_.pop_front();
    }
    bool changed = false, picked_obj = false;   // picked_obj: 이번 걸음 답이 물체 후보 — 노드가 그 자리로 가므로 목표가 죽은 게 아님
    Answer a;
    while (tr.poll(&a)) {
      if (!wait_ || a.id != req_.id) { ++n_late_; continue; }
      wait_ = false;
      last_ms_ = a.ms;
      delay_ema_ = n_delay_ == 0 ? (double)(s.step - req_.step) : 0.7 * delay_ema_ + 0.3 * (double)(s.step - req_.step);   // 걸음으로 잰 답 지연
      ++n_delay_;
      if (s.step - req_.step > p_.timeout_steps) { ++n_timeout_; ++streak_; continue; }
      if (!a.ok) { ++n_fail_; ++streak_; last_err_ = a.err; if (streak_ < p_.fail_max) ask_ = "previous answer was invalid"; continue; }
      const int k = a.choice[0] - 'A';
      if (k < 0 || k >= (int)req_.cells.size()) { ++n_fail_; ++streak_; last_err_ = "letter out of range"; if (streak_ < p_.fail_max) ask_ = "previous answer was invalid"; continue; }
      ++n_ans_;
      streak_ = 0;
      last_choice_ = a.choice;   // 그림용 (RViz ~/candidates): 마지막 답 글자 · 그 후보 번호 · 걸음
      last_pick_id_ = req_.ids[(size_t)k];
      last_ans_step_ = s.step;
      if (req_.ids[(size_t)k] <= kObjId0) {   // 물체 후보: 노드가 그 설 자리로 감 (take_object_pick)
        const size_t oi = (size_t)(kObjId0 - req_.ids[(size_t)k]);
        if (oi < obj_req_.size()) { obj_last_ = obj_req_[oi]; obj_pick_ = true; picked_obj = true; }
        continue;
      }
      // 지금 목록을 새로 만들고 안정 번호로, 없으면 서는 자리 match_r 안
      ctl.build_cands(v);
      int idx = ctl.cand_index(req_.ids[(size_t)k]);
      if (idx < 0) idx = match_cand(v, ctl, req_.cells[(size_t)k]);
      if (idx < 0) { ++n_nomatch_; ++streak_; if (streak_ < p_.fail_max) ask_ = "chosen candidate is gone"; continue; }   // 답의 후보가 사라짐 — 옛 자리로 안 가고 다시 물음
      { const cand::Cand& q = ctl.cands().c[(size_t)idx]; n_pick_failed_ += q.aborts > 0 || q.in_avoid; n_pick_top_ += idx == 0; }   // 평가 셈: 실패 기록 후보 · 규칙 1 순위
      { const int prev = ctl.cur_id(); *cmd = ctl.go_cand(v, idx, why); note_dec(v, ctl, prev, 0); }
      changed = true;
    }
    if (wait_ && s.step - req_.step > p_.timeout_steps) { wait_ = false; ++n_timeout_; ++streak_; }
    if (ext_) {   // 목표를 밖(노드 — 사람 따라가기)이 쥠: 답만 받고, 목표 죽음·사건 물음은 안 함. 밖이 ask() 로 부탁한 물음만
      if (!wait_ && !ask_.empty() && ctl.due(s)) { const std::string w = ask_; ask_.clear(); send_req(v, ctl, tr, w); }
      return changed;
    }
    bool urgent = false;
    const bool dead = !changed && !picked_obj && ctl.due(s) && (ctl.redo_pending() || !ctl.live(s));
    // 목표가 죽었는데 오는 답이 없으면 지금 묻고 (판 처음 포함), 답이 오는 중이면 보낸 뒤 지연 어림 + hold_extra 걸음까지 기다림: 목표는 판단 모델 답이 정함.
    // 그 사이 아래층은 지금 명령 그대로 (닿았으면 제자리 · 막혔으면 막힘 처리 그대로). 요청을 못 만들면 (후보 0 · 1 개) 바로 규칙
    // 답이 fail_max 번 잇따라 틀렸으면(서버 꺼짐 · 느림 · 꼴 틀림) 기다리지 않고 바로 규칙 — 요청은 사건 때 계속 보내 서버가 살아나면 다시 판단 모델로
    const bool can_hold = p_.hold_extra >= 0 && streak_ < p_.fail_max;
    if (dead && !wait_ && can_hold) send_req(v, ctl, tr, last_send_ < 0 ? "start" : "goal failed or lost");
    const bool hold = dead && can_hold && wait_ && s.step - req_.step <= (int)(delay_ema() + 0.5) + p_.hold_extra;
    if (hold) ++n_hold_;
    if (dead && !hold) {   // 기다리지 않고 slot0 차례로 (combine_run 의 slot0 위층과 같은 고름)
      {   // 셈: 왜 목표가 죽었나 (끝난 목표의 까닭 cand::Why · 9 = 끝 기록 없이 경로·목표 칸 사라짐) · 답이 오는 중이었나
        int w9 = 9;
        if (ctl.redo_pending() && !ctl.cand_mem().goals.empty() && ctl.cand_mem().goals.back().end >= 0) w9 = ctl.cand_mem().goals.back().why;
        ++n_fb_why_[w9 >= 0 && w9 < 10 ? w9 : 9];
        n_fb_wait_ += wait_;
      }
      if (fallback_object(v, ctl)) {   // 지시가 있고 힌트 클래스 물체에 갈 수 있으면 규칙도 그리로 (가장 가까운 것 — 노드가 take_object_pick 으로 감)
        ++n_fallback_;
        ++n_fb_obj_;
        return changed;
      }
      { ctl.build_cands(v); const int prev = ctl.cur_id(); *cmd = ctl.go_cand(v, -1, why); note_dec(v, ctl, prev, 1); }   // 규칙(cand_pick)
      changed = true;
      urgent = true;
      ++n_fallback_;
    }
    std::string trig;   // 이번에 묻는 까닭 (RViz · 상태 글)
    if (!wait_) {
      if (urgent) trig = "goal failed or lost";
      else if (last_send_ < 0) trig = "start";
      else if (instr_new_ && ctl.due(s)) trig = "new instruction";
      else if (!ask_.empty() && ctl.due(s)) { trig = ask_; ask_.clear(); }   // 밖(노드)이 부탁한 물음   // 지시가 바뀜: 지금 목표로 몰면서 바로 물음 (노드는 이 걸음에 물체 목록을 새로 채움)
      else if (p_.mode == 0) { if (s.step - last_send_ >= p_.period_steps) trig = "period"; }
      else trig = event_trigger(v, ctl);
    }
    if (!trig.empty()) send_req(v, ctl, tr, trig);
    return changed;
  }
  // 규칙 대신 고름의 지시 쪽: 지시 · 힌트 클래스가 있고 그 클래스 물체 설 자리가 있으면 길이 가장 짧은 것을 물체 답처럼 (따라가기 후보는 안 고름 — 판단 모델만)
  bool fallback_object(const lxc::View& v, lxc::Ctl& ctl) {
    if (p_.instruction.empty() || p_.class_hint.empty() || p_.objects.empty()) return false;
    const cand::Out& co = ctl.build_cands(v);
    const Params::Obj* best = nullptr;
    const std::vector<Params::Obj> sp = object_spots(v, co);
    for (const auto& o : sp)
      if (!o.follow && o.match >= 1.f && (!best || o.len_m < best->len_m)) best = &o;
    if (!best) return false;
    obj_last_ = *best;
    obj_pick_ = true;
    return true;
  }
  // 요청 하나 만들어 보냄 (못 만들면 — 후보 0 · 1 개 — 안 보냄, wait_ 그대로 거짓)
  void send_req(const lxc::View& v, lxc::Ctl& ctl, Transport& tr, const std::string& trig) {
    const sim::EnvState& s = *v.s;
    last_send_ = s.step;
    Req r;
    if (build_cand(v, ctl, &r)) {
      instr_new_ = false;
      r.id = ++next_id_;
      r.step = s.step;
      r.why = trig;
      last_trig_ = trig;
      ++n_trig_[trig];
      req_ = r;
      wait_ = true;
      ++n_sent_;
      tr.send(req_);
    }
  }
  // event 모드: 물을 까닭 (없으면 빈 글)
  std::string event_trigger(const lxc::View& v, lxc::Ctl& ctl) {
    const sim::EnvState& s = *v.s;
    if (!ctl.due(s)) return "";   // 1 s 결정 차례에만 봄 (갈래 굴리기는 이 걸음에만 지도를 꺼냄)
    if (s.step - last_send_ >= p_.max_steps) return "max interval";
    if (ctl.live(s) && ctl.cmd().target >= 0 && approach_goal_ != ctl.cmd().target && s.p_n > 0) {   // 남은 길 (지금 웨이포인트부터 끝까지)
      const float rx = (s.x + s.ex - v.W->ox) / v.c->res, ry = (s.y + s.ey - v.W->oy) / v.c->res;
      const int pi = std::max(0, std::min((int)s.p_i, (int)s.p_n - 1));
      double len = std::hypot(s.wp[2 * pi] + 0.5 - rx, s.wp[2 * pi + 1] + 0.5 - ry);
      for (int k = pi; k + 1 < s.p_n; ++k) len += std::hypot((double)(s.wp[2 * k + 2] - s.wp[2 * k]), (double)(s.wp[2 * k + 3] - s.wp[2 * k + 1]));
      const double dt_s = (double)v.c->substeps * v.c->dt;   // 걸음 시간 (s)
      const bool soon = p_.ask_lead_s >= 0.f && len * v.c->res / p_.ask_v <= delay_ema() * dt_s + p_.ask_lead_s;   // 답보다 먼저 닿을 듯
      if (len * v.c->res <= p_.approach_m || soon) { approach_goal_ = ctl.cmd().target; return "approaching goal"; }
    }
    if (v.lmap) {   // 1 s 결정 차례마다 후보를 새로 만들어 견줌
      const cand::Out& co = ctl.build_cands(v);
      const int cur = ctl.cur_id();
      if (cur >= 0 && ctl.cmd().follow != 0 && ctl.cand_index(cur) < 0 && gone_id_ != cur) { gone_id_ = cur; return "current goal's frontier gone"; }
      const int ci = cur >= 0 ? ctl.cand_index(cur) : -1;
      if (!co.c.empty() && ci > 0 && co.c[0].id != better_id_ && co.c[0].score >= p_.switch_ratio * co.c[(size_t)ci].score) {
        better_id_ = co.c[0].id;
        return "much better candidate";
      }
    }
    return "";
  }
  const std::string& last_trigger() const { return last_trig_; }
  long n_pick_failed() const { return n_pick_failed_; }   // 답이 고른 후보가 실패 기록(aborts > 0 · 피하는 곳)이 있던 수
  long n_pick_top() const { return n_pick_top_; }         // 답이 후보 점수 1 순위(규칙과 같은 고름)였던 수
  const std::map<std::string, long>& trigger_counts() const { return n_trig_; }
  // 후보 모드 요청 (scripts/cand_view.h 후보 — 필드 이름은 cand::Cand 그대로 뜻을 풀어 씀). 기억: 후보마다 기록 + 최근 목표 5 개 + 사진의 지나온 길 · 지난 목표
  bool build_cand(const lxc::View& v, lxc::Ctl& ctl, Req* r) {
    const cand::Out& co = ctl.build_cands(v);
    const sim::EnvState& s = *v.s;
    if ((p_.instruction.empty() && !p_.obj_always) || p_.objects.empty()) {   // 지시 없음: 예전과 같은 요청 (바이트 같음)
      obj_req_.clear();
      if (co.c.size() < 2) return false;
      return render_cand(p_, co.c, ctl.cand_mem(), decs_, hist_, s.step, s.x + s.ex, s.y + s.ey, s.th + s.eth, *v.c, *v.W, v.lmap, r);
    }
    Params P = p_;
    P.objects = pick_objects(object_spots(v, co), p_.obj_max);
    obj_req_ = P.objects;
    if (co.c.size() + P.objects.size() < 2) return false;
    return render_cand(P, co.c, ctl.cand_mem(), decs_, hist_, s.step, s.x + s.ex, s.y + s.ey, s.th + s.eth, *v.c, *v.W, v.lmap, r);
  }
  // 물체마다 설 자리: 물체 둘레 obj_ring_in ~ obj_ring_out 고리에서, 이 결정의 후보 뽑기 BFS 가 닿은 칸(계획 몸통 반경·깊이 층 — 벽 뒤·가구 속 아님) 가운데
  // 물체에 가까운 띠(0.1 m)부터, 같은 띠면 BFS 길이 짧은 칸. 못 가는 물체는 뺀다. 물체 하나 = 후보 하나 (물체 지도가 이미 합친 것)
  std::vector<Params::Obj> object_spots(const lxc::View& v, const cand::Out& co) const {
    std::vector<Params::Obj> out;
    if (!v.lmap || !co.reach_flag || !co.reach_dist) return out;
    const sim::EnvCfg& c = *v.c;
    const sim::World& W = *v.W;
    const int gw = W.gw, gh = W.gh;
    // 갈 수 있는 칸 = 이 결정의 후보 뽑기와 같은 BFS (계획 몸통 반경 · 깊이 카메라 층) — 프론티어 후보와 같은 잣대
    const std::vector<int> dist = [&] { std::vector<int> d((size_t)gw * gh, -1); for (int i = 0; i < gw * gh; ++i) if ((co.reach_flag[i] & 1) && co.reach_dist[i] >= 0) d[(size_t)i] = co.reach_dist[i]; return d; }();
    for (const auto& o : p_.objects) {
      const int ox = (int)std::floor((o.x - W.ox) / c.res), oy = (int)std::floor((o.y - W.oy) / c.res);
      const int ext = (int)std::ceil(p_.obj_ring_out / c.res) + 1;
      int best = -1, bd = 1 << 30;
      for (int dy = -ext; dy <= ext; ++dy)
        for (int dx = -ext; dx <= ext; ++dx) {
          const float rd = std::hypot((float)dx, (float)dy) * c.res;
          const int x = ox + dx, y = oy + dy;
          if (rd < p_.obj_ring_in || rd > p_.obj_ring_out || x < 0 || y < 0 || x >= gw || y >= gh) continue;
          const int dd = dist[(size_t)y * gw + x];   // 물체에 가까운 고리(0.1 m 띠)부터, 같은 띠면 길 짧은 칸 (object_goto 와 같은 규칙)
          const int key = (int)((rd - p_.obj_ring_in) / 0.1f) * 1000000 + dd;
          if (dd >= 0 && key < bd) { bd = key; best = y * gw + x; }
        }
      if (best < 0) continue;
      Params::Obj e = o;
      e.vp = best;
      e.len_m = (bd % 1000000) * c.res;
      out.push_back(e);
    }
    return out;
  }
  // 물체 후보 칸 고르기 (k 칸): 따라가기 후보(가까운 사람부터) → class_hint 클래스 → 나머지 가까운 것. 한 무리가 남은 칸보다 많으면 길 거리 순으로
  // 고르게 벌려 뽑음 (가장 가까운 것과 가장 먼 것을 늘 넣음 — "가장 먼 의자" 같은 지시도 답이 후보 안에 있게)
  static std::vector<Params::Obj> pick_objects(std::vector<Params::Obj> in, int k) {
    std::vector<Params::Obj> out;
    std::stable_sort(in.begin(), in.end(), [](const Params::Obj& a, const Params::Obj& b) { return a.len_m < b.len_m; });
    for (int g = 0; g < 3 && (int)out.size() < k; ++g) {
      std::vector<Params::Obj> grp;
      for (const auto& o : in) if ((g == 0 && o.follow) || (g == 1 && !o.follow && o.match >= 1.f) || (g == 2 && !o.follow && o.match < 1.f)) grp.push_back(o);
      const int room = k - (int)out.size(), n = (int)grp.size();
      if (n <= room) { out.insert(out.end(), grp.begin(), grp.end()); continue; }
      for (int i = 0; i < room; ++i) out.push_back(grp[(size_t)(room == 1 ? 0 : (int)std::lround((double)i * (n - 1) / (room - 1)))]);
    }
    return out;
  }
  // 요청 본문 그리기 (후보 모드) — 노드·시뮬(build_cand)과 라벨 다시 그리기(upper_lora/rollout/req_render: 갈래 굴리기가 남긴 원시 상태)가 같은 함수를 씀.
  // C = 후보(cand::build 차례, 기억 필드 포함) · M = 판 기억 · decs = 지금까지 결정 · (rx, ry, rth) = 지도 자세 · lmap = 지도 (c.size 줄 간격)
  static constexpr const char* kHint = "Rules of thumb: prefer a spot that is close and roughly ahead; finish nearby small frontiers before leaving an area because coming back later is "
                                       "expensive; keep the current target unless another candidate is clearly better; avoid spots that failed recently.";
  static bool render_cand(const Params& P, const std::vector<cand::Cand>& C, const cand::Mem& M, const std::vector<Dec>& decs,
                          const std::deque<std::pair<int, std::vector<int16_t>>>& hist, int step, float rx, float ry, float rth,
                          const sim::EnvCfg& c, const sim::World& W, const int16_t* lmap, Req* r) {
    const double dt = (double)c.substeps * c.dt;
    using oj = nlohmann::ordered_json;
    const char* why_txt[] = {"none", "no path", "stuck or blocked", "contact", "timeout", "path ended off target", "reached but nothing new seen", "reached", "switched"};
    const auto rnd = [](double x, double q) { const double k = std::round(1.0 / q); return std::round(x * k) / k; };   // JSON 에 0.30000000000000004 꼴이 안 찍히게
    const auto rel = [&](float x, float y) {   // 로봇 기준 거리 · 방향
      const float dx = x - rx, dy = y - ry;
      const float b = std::remainder(std::atan2(dy, dx) - rth, 6.2831853f);
      return std::make_pair(std::hypot(dx, dy), (int)std::lround(b * 57.29578f));
    };
    std::vector<dec::Cand> dc;
    std::vector<int> cells;
    oj cl = oj::array();
    // 물체 후보 (지시가 있을 때만, build_cand 가 설 자리를 채움): 앞 글자들. 프론티어 후보는 남은 칸에 (합쳐 max_cands)
    const size_t n_obj = P.instruction.empty() && !P.obj_always ? 0 : std::min(P.objects.size(), (size_t)std::max(0, P.obj_max));
    for (size_t i = 0; i < n_obj; ++i) {
      const Params::Obj& ob = P.objects[i];
      dec::Cand d;
      d.key = (char)('A' + (int)i);
      d.slot = (int)i;
      d.object = true;
      dc.push_back(d);
      cells.push_back(ob.vp % W.gw);
      cells.push_back(ob.vp / W.gw);
      r->cells.push_back(ob.vp);
      r->ids.push_back(kObjId0 - (int)i);
      const auto [dist, ang] = rel(ob.x, ob.y);
      oj o;
      o["id"] = std::string(1, d.key);
      o["kind"] = ob.follow ? "follow" : "object";
      o["object"] = ob.cls;
      o["oid"] = ob.oid;
      if (P.compact) { o["len"] = rnd(ob.len_m, 0.1); o["bear"] = ang; o["dist"] = rnd(dist, 0.1); o["seen_s"] = rnd(ob.seen_s, 1.0); o["vis"] = ob.seen_s <= P.vis_s; }
      else { o["path_length_m"] = rnd(ob.len_m, 0.1); o["bearing_deg_left_positive"] = ang; o["distance_m"] = rnd(dist, 0.1); o["seconds_since_seen"] = rnd(ob.seen_s, 1.0); o["visible_now"] = ob.seen_s <= P.vis_s; }
      cl.push_back(o);
    }
    std::vector<size_t> ord;   // 글자 차례 → C 번호
    for (size_t i = 0; i < C.size() && i < 26; ++i) ord.push_back(i);
    if (P.order == 1) std::stable_sort(ord.begin(), ord.end(), [&](size_t a, size_t b) { return C[a].len_m < C[b].len_m; });
    const int room = P.max_cands > 0 ? P.max_cands - (int)n_obj : (int)ord.size();
    if (room >= 0 && (int)ord.size() > room) ord.resize((size_t)room);   // 점수 순 · 거리 순 어느 쪽이든 앞에서
    if (ord.size() + n_obj < 2) return false;
    std::map<int, char> id_let;
    for (size_t i = 0; i < ord.size(); ++i) id_let[C[ord[i]].id] = (char)('A' + (int)(n_obj + i));
    for (size_t i = 0; i < ord.size(); ++i) {
      const cand::Cand& q = C[ord[i]];
      dec::Cand d;
      d.key = (char)('A' + (int)(n_obj + i));
      d.slot = (int)(n_obj + i);
      dc.push_back(d);
      cells.push_back(q.vp % W.gw);
      cells.push_back(q.vp / W.gw);
      r->cells.push_back(q.vp);
      r->ids.push_back(q.id);
      oj o;
      o["id"] = std::string(1, d.key);
      if (n_obj > 0) o["kind"] = "frontier";
      if (P.compact) {   // 짧은 키 (candidate_fields 에 뜻)
        o["t"] = rnd(q.t_s, 0.1); o["len"] = rnd(q.len_m, 0.1); o["bear"] = (int)std::lround(q.bearing * 57.29578f); o["new"] = rnd(q.gain_m2, 1.0);
        o["w"] = rnd(q.open_m, 0.1); o["room"] = q.same_room != 0; o["cur"] = q.cur != 0;
        if (P.mem_fields) { o["fail"] = q.aborts; o["fail_s"] = q.since_fail_s < 0 ? oj(nullptr) : oj(rnd(q.since_fail_s, 1.0)); o["seen"] = q.visited != 0; }
        cl.push_back(o);
        continue;
      }
      o["travel_time_s"] = rnd(q.t_s, 0.1);
      o["path_length_m"] = rnd(q.len_m, 0.1);
      o["bearing_deg_left_positive"] = (int)std::lround(q.bearing * 57.29578f);
      o["new_area_m2"] = rnd(q.gain_m2, 0.1);
      o["new_area_not_seen_from_other_candidates_m2"] = rnd(q.marg_m2, 0.1);
      o["frontier_width_m"] = rnd(q.open_m, 0.1);
      o["free_space_around_spot_m"] = rnd(q.clear_m, 0.01);
      o["narrowest_gap_on_way_m"] = rnd(q.path_clear_m, 0.01);
      o["same_room_as_robot"] = q.same_room != 0;
      o["other_frontiers_in_its_room"] = q.room_left;
      o["is_current_target"] = q.cur != 0;
      if (P.mem_fields) {
        o["times_chosen"] = q.chosen;
        o["times_failed"] = q.aborts;
        o["last_failure"] = why_txt[std::clamp(q.last_fail, 0, 8)];
        o["seconds_since_failure"] = q.since_fail_s < 0 ? oj(nullptr) : oj(rnd(q.since_fail_s, 1.0));
        o["failed_spots_nearby"] = q.fail_near;
        o["inside_avoid_zone"] = q.in_avoid != 0;
        o["robot_was_here_before"] = q.visited != 0;
      }
      cl.push_back(o);
    }
    // 최근 목표 (cand::Mem::goals, 새것부터 5 개)
    oj gl = oj::array();
    // 결정 기록 (새것부터 log_k 개): "12s ago: chose C (switched) - reached" 꼴 한 줄 — 고른 번호가 지금 목록에 없으면 gone. 결과는 그 결정 뒤 그 번호 목표 기록에서
    oj dl = oj::array();
    for (int i = (int)decs.size() - 1, n = 0; i >= 0 && n < P.log_k; --i, ++n) {
      const Dec& d = decs[(size_t)i];
      std::string let = "none";
      if (d.id >= 0) { const auto it = id_let.find(d.id); let = it != id_let.end() ? std::string(1, it->second) : "gone"; }
      std::string out = "";
      for (const auto& g : M.goals)
        if (g.id == d.id && g.start <= d.step + 1 && (g.end < 0 || g.end >= d.step)) out = g.end < 0 ? "still going" : why_txt[std::clamp(g.why, 0, 8)];
      if (out.empty()) out = d.id < 0 ? "no goal" : "replaced";
      char b[160];
      std::snprintf(b, sizeof b, "#%d %.0fs ago: %s %s%s - %s", n + 1, (step - d.step) * dt, d.kept ? "kept" : "chose", let.c_str(), d.src ? " (fallback rule)" : "", out.c_str());
      dl.push_back(std::string(b));
    }
    dec::MapMem mm;
    mm.clear_tags = true;
    const auto cxy = [&](float x, float y, float* cx, float* cy) { *cx = (x - W.ox) / c.res; *cy = (y - W.oy) / c.res; };
    for (int i = (int)M.goals.size() - 1, n = 0; i >= 0 && n < 5; --i, ++n) {
      const cand::Mem::Goal& g = M.goals[(size_t)i];
      const auto [dist, ang] = rel(g.x, g.y);
      oj o;
      o["distance_from_robot_m"] = rnd(dist, 0.1);
      o["bearing_deg_left_positive"] = ang;
      o["outcome"] = g.end < 0 ? "in progress" : why_txt[std::clamp(g.why, 0, 8)];
      o["started_s_ago"] = rnd((step - g.start) * dt, 1.0);
      if (g.end >= 0) o["ended_s_ago"] = rnd((step - g.end) * dt, 1.0);
      gl.push_back(o);
    }
    if (P.mem_image) {
      for (const auto& q : M.trail) {
        dec::MapMem::P pp;
        cxy(q.x, q.y, &pp.cx, &pp.cy);
        pp.age_s = (float)((step - q.step) * dt);
        mm.trail.push_back(pp);
      }
      for (size_t gi = M.goals.size() > 10 ? M.goals.size() - 10 : 0; gi < M.goals.size(); ++gi) {   // 최근 10 개만 (사진이 어지럽지 않게)
        const cand::Mem::Goal& g = M.goals[gi];
        dec::MapMem::G gg;
        cxy(g.x, g.y, &gg.cx, &gg.cy);
        gg.outcome = g.end < 0 ? 0 : g.why == cand::kReached ? 1 : g.why == cand::kSwitched ? 3 : 2;
        mm.goals.push_back(gg);
      }
      for (int i = (int)decs.size() - 1, n = 0; i >= 0 && n < P.log_k; --i, ++n) {   // 결정 기록 번호 (1 = 가장 최근)
        if (decs[(size_t)i].id < 0) continue;
        dec::MapMem::N nn;
        cxy(decs[(size_t)i].x, decs[(size_t)i].y, &nn.cx, &nn.cy);
        nn.num = n + 1;
        mm.nums.push_back(nn);
      }
    }
    oj st;
    st["task"] = P.instruction.empty()
                     ? "A small wheeled robot is exploring an unknown indoor floor. Its job is to map the whole floor in the shortest total time. It drives at about 1 m/s and "
                       "turning around costs a few seconds. A frontier is the edge between mapped free space and unmapped space; each candidate is a spot from which a frontier can be seen."
                     : "A small wheeled robot drives on an indoor floor that it maps as it goes, and carries out the user's instruction (field instruction, written by the user, "
                       "any language). Exploring means mapping the whole floor in the shortest total time. It drives at about 1 m/s and turning around costs a few seconds. "
                       "A frontier is the edge between mapped free space and unmapped space; a frontier candidate is a spot from which a frontier can be seen.";
    std::vector<uint8_t> png;
    if (P.input != dec::kText) {
      int kb[4] = {1 << 30, 1 << 30, -1, -1};
      for (int y = 0; y < W.gh; ++y)
        for (int x = 0; x < W.gw; ++x)
          if (lmap[(size_t)y * c.size + x] != sim::kUnknown) {
            kb[0] = std::min(kb[0], x); kb[1] = std::min(kb[1], y); kb[2] = std::max(kb[2], x); kb[3] = std::max(kb[3], y);
          }
      if (P.image == 1) {   // 꼴 B: 지난 지도 셋 (5 · 15 · 30 s 전 쯤 — 10 걸음 칸에서 가장 가까운 것), 그때 자세는 기억의 자취에서
        std::vector<const int16_t*> past;
        std::vector<float> age, pose;
        for (int ago : {50, 150, 300}) {
          const std::pair<int, std::vector<int16_t>>* best = nullptr;
          for (const auto& q : hist) if (q.first <= step - ago && (!best || q.first > best->first)) best = &q;
          if (!best && !hist.empty()) best = &hist.front();
          if (!best) continue;
          past.push_back(best->second.data());
          age.push_back((float)((step - best->first) * dt));
          float px_ = (rx - W.ox) / c.res, py_ = (ry - W.oy) / c.res, pth = rth;
          int bd = 1 << 30;
          for (const auto& t : M.trail)
            if (std::abs(t.step - best->first) < bd) { bd = std::abs(t.step - best->first); px_ = (t.x - W.ox) / c.res; py_ = (t.y - W.oy) / c.res; pth = t.th; }
          pose.push_back(px_); pose.push_back(py_); pose.push_back(pth);
        }
        png = dec::render_png_tiles(lmap, past, age, pose, c.size, W.gw, W.gh, sim::kUnknown, mdp::kOccL, kb, (rx - W.ox) / c.res, (ry - W.oy) / c.res, rth, dc,
                                    cells, P.px, mm);
      } else {
        png = dec::render_png_mem(lmap, c.size, W.gw, W.gh, sim::kUnknown, mdp::kOccL, kb, (rx - W.ox) / c.res, (ry - W.oy) / c.res, rth, dc, cells, P.px, mm);
      }
      st["map_image"] = std::string("The image is the robot's current occupancy map seen from above: white = free space, black = wall or obstacle, gray = not mapped yet. "
                                    "The blue arrow is the robot, pointing where it is heading. Each candidate spot is a small yellow dot; its letter is in a colored circle "
                                    "joined to that dot by a line of the same color (each letter has its own color).") +
                        (P.mem_image ? " Purple dots are where the robot drove in the last 60 seconds: dark purple = recent, faint purple = older. Past goals are marked: "
                                        "green check = reached, red X = failed (the robot got stuck or had no path), gray ring = left for another goal, blue ring = current goal."
                                        " Small black numbers in white boxes mark the targets of the last decisions (1 = most recent), matching decision_log_newest_first."
                                      : "") +
                        (P.image == 1 ? " The image is split into four panels showing the same map area at the same scale, so positions line up: top-left = now (with the "
                                         "candidates, the path and the goals), top-right, bottom-left and bottom-right = the map a few seconds ago (the time is written in each "
                                         "panel, e.g. -5s); in those older panels the blue arrow is where the robot was then."
                                       : "");
    }
    if (P.input != dec::kImage) {
      if (P.compact)
        st["candidate_fields"] = std::string("t = driving time s incl. turning; len = path m; bear = bearing deg (left +); new = unmapped area visible from the spot m2; ") +
                                 "w = frontier opening width m; room = in the robot's room; cur = current target" +
                                 (P.mem_fields ? "; fail = times the robot failed to reach it; fail_s = seconds since that failure; seen = the robot already stood near it." : ".");
      else st["candidate_fields"] = "travel_time_s = estimated driving time including turning; new_area_m2 = unmapped area visible from the spot; "
                               "new_area_not_seen_from_other_candidates_m2 = part of it that no better-ranked candidate also sees; frontier_width_m = width of the opening; "
                               "free_space_around_spot_m = distance from the spot to the nearest obstacle; narrowest_gap_on_way_m = tightest squeeze on the way; "
                               "times_failed / last_failure / seconds_since_failure = what happened when the robot tried this spot before; failed_spots_nearby = failed goals within 1 m; "
                               "inside_avoid_zone = near a spot the robot is currently avoiding after a failure; robot_was_here_before = the robot already stood near it.";
      // 짧은 꼴: 바뀌지 않는 글(힌트)을 바뀌는 후보 앞에 — 서버 앞부분 캐시(llama-server decision checkpoint · images-last 패치)가 더 길게 이어짐.
      // 최근 목표 목록은 빼고(지도 사진의 ✓ · × 표시와 결정 기록이 같은 것을 말함) 결정 기록만 — 로봇 Orin 에서 요청 하나 약 2 s 안 (upper_lora/README "요청 꼴과 지연")
      if (P.compact && P.hint) st["hint"] = P.instruction.empty() ? std::string(kHint) : std::string("When exploring: ") + kHint;
      st["candidates"] = cl;
      if (!P.compact) st["recent_goals_newest_first"] = gl;
      if (P.log_k > 0) st["decision_log_newest_first"] = dl;
    }
    if (P.hint && !(P.compact && P.input != dec::kImage)) st["hint"] = P.instruction.empty() ? std::string(kHint) : std::string("When exploring: ") + kHint;
    oj body;
    if (!P.instruction.empty()) {
      st["instruction"] = P.instruction;   // 사용자 글 그대로
      if (!P.class_hint.empty()) st["class_hint"] = P.class_hint;
      st["known_objects"] = P.known_objects;
    }
    if (!P.instruction.empty() || n_obj > 0)
      st["candidate_kinds"] = n_obj > 0 ? std::string("kind = object: a spot next to an object on the robot's object map built from its camera (object = class, oid = map id, ") +
                                              (P.compact ? "len = path m, bear = bearing deg left +, dist = straight-line m, seen_s = seconds since the camera last saw it, vis = seen right now"
                                                         : "visible_now = seen by the camera right now") +
                                              "); choosing it drives there, turns to the object and stops. kind = follow: a person seen right now; choosing it makes the robot keep "
                                              "following that person at about 1 m. kind = frontier: a spot that maps unknown space (exploring). On the map image object and follow "
                                              "candidates are cyan squares, frontier candidates yellow dots."
                                        : "No object on the robot's object map is a candidate right now; all candidates are frontier spots (exploring).";
    body["state"] = st;
    if (!png.empty()) body["images"] = oj::array({dec::png_uri(png)});
    oj crit = oj::object();
    for (const auto& d : dc) crit[std::string(1, d.key)] = nullptr;
    body["questions"]["pick"] = oj{{"type", "choice"}, {"instructions", P.instruction.empty() ? "Which candidate spot should the robot drive to next so that the whole floor gets mapped in the least total time?"
                                                                                               : "Which candidate best carries out the user's instruction next? If the instruction is to explore, or what it asks "
                                                                                                 "for is not among the candidates, pick the spot that maps the floor (and so most likely reveals it) fastest."},
                                   {"criteria", crit}};
    r->body = body.dump();
    return true;
  }
  int match_cand(const lxc::View& v, const lxc::Ctl& ctl, int vp) const {
    const float q = p_.match_r / v.c->res;
    int best = -1;
    float bd = q * q;
    const auto& L = ctl.cands().c;
    for (int j = 0; j < (int)L.size(); ++j) {
      const float dx = (float)(L[(size_t)j].vp % v.W->gw - vp % v.W->gw), dy = (float)(L[(size_t)j].vp / v.W->gw - vp / v.W->gw), d = dx * dx + dy * dy;
      if (d <= bd) { bd = d; best = j; }
    }
    return best;
  }
  // 새 판: 답 지연 어림은 서버 성질이라 이어 감 (판마다 첫 어림 delay0 으로 돌아가지 않게)
  void restart() { const Params p = p_; const double de = delay_ema_; const long nd = n_delay_; *this = Upper(p); delay_ema_ = de; n_delay_ = nd; }
  bool waiting() const { return wait_; }
  // 지시가 바뀌면 다음 결정 차례에 바로 물음 ("new instruction"). 옛 지시로 만든 요청을 기다리는 중이면 그 답은 버림 (늦은 답으로 셈)
  void set_instruction(const std::string& t) {
    if (t == p_.instruction) return;
    p_.instruction = t;
    instr_new_ = true;
    wait_ = false;
  }
  void set_known_objects(const std::vector<std::string>& k) { p_.known_objects = k; }
  void set_class_hint(const std::string& c) { p_.class_hint = c; }
  // 밖(노드)이 목표를 쥠 (사람 따라가기): 켜면 목표 죽음 → 규칙 · 사건 물음을 안 하고 답만 받음. 고른 답이 프론티어면 step 이 명령을 돌려줌(밖은 따라가기를 끝냄)
  void set_external(bool on) { ext_ = on; }
  bool external() const { return ext_; }
  // 다음 결정 차례에 이 까닭으로 물음 (기다리는 답이 있으면 그 뒤)
  void ask(const std::string& why) { ask_ = why; }
  void set_objects(const std::vector<Params::Obj>& o) { p_.objects = o; }
  // 판단 모델이 물체 후보를 골랐으면 그 설 자리 (한 번 꺼내면 지움) — 노드는 ~/goto 처럼 그 자리로 감
  bool take_object_pick(Params::Obj* o) { if (!obj_pick_) return false; *o = obj_last_; obj_pick_ = false; return true; }
  const std::vector<Params::Obj>& last_objects() const { return obj_req_; }   // 마지막 요청의 물체 후보 (글자 A … 차례, RViz 용)
  int last_send() const { return last_send_; }   // 마지막으로 보낸 걸음 (−1 = 아직)
  long n_sent() const { return n_sent_; }
  long n_ans() const { return n_ans_; }
  long n_timeout() const { return n_timeout_; }
  long n_fail() const { return n_fail_; }
  long n_nomatch() const { return n_nomatch_; }
  long n_fallback() const { return n_fallback_; }
  long n_fb_why(int k) const { return n_fb_why_[k]; }   // 규칙이 대신 고른 까닭별 수 (cand::Why · 9 = 경로·목표 칸 사라짐)
  long n_fb_wait() const { return n_fb_wait_; }         // 그중 답이 오는 중이던 수
  long n_src(int k) const { return n_src_[k]; }        // 후보 모드 결정 수: 0 판단 모델 답 · 1 규칙
  long n_hold() const { return n_hold_; }
  long n_fb_obj() const { return n_fb_obj_; }         // 규칙 대신 고름 중 지시 물체로 간 수
  int streak() const { return streak_; }              // 잇따른 답 실패 수 (fail_max 넘으면 규칙이 바로 대신)               // 목표가 죽었지만 답을 기다린 걸음 수
  double delay_ema() const { return n_delay_ ? delay_ema_ : (double)p_.delay0_steps; }   // 답 지연 어림 (걸음)
  long n_late() const { return n_late_; }
  double last_ms() const { return last_ms_; }
  const std::string& last_choice() const { return last_choice_; }   // 마지막으로 받아 쓴 답 글자 (없으면 빈 글)
  int last_pick_id() const { return last_pick_id_; }                // 그 답의 후보 안정 번호 (없으면 −1)
  int last_ans_step() const { return last_ans_step_; }              // 그 답을 받은 걸음 (없으면 −1)
  const std::string& last_err() const { return last_err_; }
  const Params& params() const { return p_; }
  void set_input(int input) { p_.input = input; }
  const std::vector<Dec>& decisions() const { return decs_; }   // 판 동안 모든 결정 (갈래 원시 상태 — 렌더 때 K 를 바꿀 수 있게)
  void note_dec(const lxc::View& v, const lxc::Ctl& ctl, int prev_id, int src) {
    Dec d{v.s->step, ctl.cur_id(), 0.f, 0.f, 0, src};
    if (ctl.cmd().follow != 0 && ctl.cmd().target >= 0) {
      d.x = v.W->ox + (ctl.cmd().target % v.W->gw + 0.5f) * v.c->res;
      d.y = v.W->oy + (ctl.cmd().target / v.W->gw + 0.5f) * v.c->res;
    } else d.id = -1;
    d.kept = d.id >= 0 && d.id == prev_id;
    decs_.push_back(d);
    ++n_src_[src ? 1 : 0];
  }

 private:
  Params p_;
  static constexpr int kObjId0 = -1000;   // 물체 후보 안정 번호 = kObjId0 − 차례 (프론티어 후보 번호는 0 이상)
  std::vector<Params::Obj> obj_req_;      // 마지막 요청의 물체 후보 (번호 차례)
  Params::Obj obj_last_;
  bool obj_pick_ = false;
  Req req_;
  bool wait_ = false, instr_new_ = false, ext_ = false;
  int streak_ = 0;
  long n_fb_obj_ = 0;
  std::string ask_;
  int last_send_ = -1;
  int approach_goal_ = -1, gone_id_ = -1, better_id_ = -1;
  long n_pick_failed_ = 0, n_pick_top_ = 0;
  std::string last_trig_;
  std::map<std::string, long> n_trig_;
  std::vector<Dec> decs_;
  std::deque<std::pair<int, std::vector<int16_t>>> hist_;   // 꼴 B 지난 지도 (걸음, 지도)
  long n_fb_why_[10] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0}, n_fb_wait_ = 0, n_src_[2] = {0, 0}, n_hold_ = 0, n_delay_ = 0;
  double delay_ema_ = 0;
  long next_id_ = 0, n_sent_ = 0, n_ans_ = 0, n_timeout_ = 0, n_fail_ = 0, n_nomatch_ = 0, n_fallback_ = 0, n_late_ = 0;
  double last_ms_ = 0;
  std::string last_choice_;
  int last_pick_id_ = -1, last_ans_step_ = -1;
  std::string last_err_;
};

}  // namespace lxd
