// 합친 탐사의 위층을 판단 모델(llama-server /v1/systemone, zero-shot)에 맡기는 길 — 느린 주기 · 비동기. 실물 노드(limo_fire_rl low_rl upper:=decider)와
// 시뮬(combine_run --upper decider --decider-period · --decider-latency)이 같은 코드로 돈다 (sim = real).
//
// 요청 (combine_run --upper decider --decider-input both 와 같은 꼴 — decider.h): 유효 후보(빈 칸 아님 · 피하는 칸 아님) = hier 후보 8 칸 중 차례대로,
//   글자 A … = 유효 후보 차례, 후보 특징 = 위층 관측 mdp::hier::F_* (거리 · 방위 · 크기 · 예상 관측량 · 지금 목표), 지도 사진 = 로봇 자기 지도 + 글자 동그라미.
//   카메라 그림(입력 all)은 시뮬에서만 된다 — 시뮬 장면에 광선을 쏴 칠한 그림이라 실물에는 같은 것이 없음 (decider.h 머리말).
// 걸음마다 (Upper::step, 걸음 = 라이다 한 장 0.1 s):
//   1. 답이 왔으면: 보낸 지 timeout_steps 안이고 답 꼴이 맞으면, 고른 후보 칸에서 match_r 안의 지금 유효 후보로 → Ctl::decide_slot (못 찾으면 지금 목표 그대로)
//   2. 보낸 지 timeout_steps 넘도록 답이 없으면 버림 (지금 목표 그대로, 늦게 온 답은 번호가 달라 버림)
//   3. 위층 결정 차례(Ctl::due)인데 다시 고를 일(막힘 · 통과 · 경로 끝 밖 · 경로 없음)이 있거나 지금 목표가 죽었으면: 바로 slot0 차례로 목표를 잡고(기다리지 않음) 곧 새 요청
//      — 그 밖의 1 s 결정 차례에는 목표를 바꾸지 않는다 (판단 모델 답이 올 때만 바꿈 — fuel 의 매초 목표 바뀜을 피함)
//   4. 기다리는 요청이 없고 마지막으로 보낸 지 period_steps 지났으면(또는 3 에서 급하면) 새 요청 (유효 후보 2 개 이상일 때만)
// 보내기·받기는 Transport 가 맡는다: 실물 = 소켓 HTTP 일꾼 스레드 (limo_fire_rl/decider_client.h), 시뮬 = 정해진 걸음 뒤에 답을 내놓는 줄 (combine_run).
#pragma once
#include <algorithm>
#include <cmath>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "decider.h"
#include "low_explore_ctl.h"
#include "observations.h"

namespace lxd {

struct Params {
  int period_steps = 30;    // 요청 주기 (걸음 — 3 s)
  int timeout_steps = 80;   // 답 기다림 한도 (걸음 — 8 s)
  int px = 384;             // 지도 사진 한 변 (combine_run --decider-px)
  int input = dec::kBoth;   // dec::kText · kImage · kBoth (kAll 은 시뮬 장면 그림이 필요해 여기서는 안 됨)
  int prompt = 1;           // combine_run --decider-prompt
  float match_r = 1.5f;     // 답의 후보 칸 → 지금 후보 (m). hier 후보 칸은 덩어리에서 로봇에 가장 가까운 칸이라 답을 기다리는 동안 옮겨감 — 0.5 m 면 답의 40 ~ 55 % 를 버림 (low_explore_check, 지연 2.6 s)
};

// 보낸 요청 하나
struct Req {
  long id = 0;
  int step = 0;
  std::string body;
  std::vector<int> cells;   // 글자 차례의 후보 칸
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

// 답 고르기: answers.pick.choice (decider.cpp Client::ask 의 choice 가지와 같은 꼴)
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
  // 걸음 하나. F = 위층 관측의 후보 특징 (obs_1d + mdp::obs::POLICY_1D, [K][FEAT]). 목표가 바뀌었으면 true 와 *cmd
  bool step(const lxc::View& v, const float* F, lxc::Ctl& ctl, Transport& tr, sim::LowCmd* cmd, int* why = nullptr) {
    const sim::EnvState& s = *v.s;
    bool changed = false;
    Answer a;
    while (tr.poll(&a)) {
      if (!wait_ || a.id != req_.id) { ++n_late_; continue; }
      wait_ = false;
      last_ms_ = a.ms;
      if (s.step - req_.step > p_.timeout_steps) { ++n_timeout_; continue; }
      if (!a.ok) { ++n_fail_; last_err_ = a.err; continue; }
      const int k = a.choice[0] - 'A';
      if (k < 0 || k >= (int)req_.cells.size()) { ++n_fail_; last_err_ = "letter out of range"; continue; }
      const int slot = match(v, ctl, req_.cells[k]);
      ++n_ans_;
      if (slot < 0) { ++n_nomatch_; continue; }
      *cmd = ctl.decide_slot(v, slot, why);
      changed = true;
    }
    if (wait_ && s.step - req_.step > p_.timeout_steps) { wait_ = false; ++n_timeout_; }
    bool urgent = false;
    if (!changed && ctl.due(s) && (ctl.redo_pending() || !ctl.live(s))) {   // 기다리지 않고 slot0 차례로 (combine_run 의 slot0 위층과 같은 고름)
      *cmd = ctl.decide_slot(v, -1, why);
      changed = true;
      urgent = true;
      ++n_fallback_;
    }
    if (!wait_ && (urgent || last_send_ < 0 || s.step - last_send_ >= p_.period_steps)) {
      last_send_ = s.step;
      Req r;
      if (build(v, F, ctl, &r)) {
        r.id = ++next_id_;
        r.step = s.step;
        req_ = r;
        wait_ = true;
        ++n_sent_;
        tr.send(req_);
      }
    }
    return changed;
  }
  // 요청 본문 (combine_run.cpp 판단 모델 가지 — 후보 특징 → dec::Cand, render_png, build_body)
  bool build(const lxc::View& v, const float* F, const lxc::Ctl& ctl, Req* r) const {
    const sim::EnvState& s = *v.s;
    const sim::EnvCfg& c = *v.c;
    std::vector<dec::Cand> cs;
    std::vector<int> cells;
    for (int j = 0; j < mdp::hier::K; ++j) {
      const int cell = v.cand[j];
      if (cell < 0 || ctl.avoided(cell, s.step)) continue;
      const float* f = F + j * mdp::hier::FEAT;
      dec::Cand cd;
      cd.key = (char)('A' + (int)cs.size());
      cd.slot = j;
      cd.path_m = f[mdp::hier::F_DIST] * mdp::obsgen::kFrontNorm;
      cd.bearing = std::atan2(f[mdp::hier::F_SIN], f[mdp::hier::F_COS]);
      cd.size_m = f[mdp::hier::F_SIZE] * mdp::obsgen::kFrontSizeNorm;
      cd.util = c.cand_v2 ? f[mdp::hier::F_UTIL] : f[mdp::hier::F_UTIL] * mdp::hier::kUtilNorm;
      cd.cur = f[mdp::hier::F_CUR] > 0.5f;
      cs.push_back(cd);
      cells.push_back(cell % v.W->gw);
      cells.push_back(cell / v.W->gw);
      r->cells.push_back(cell);
    }
    if (cs.size() < 2) { r->cells.clear(); return false; }
    std::vector<uint8_t> png;
    if (p_.input != dec::kText) {
      // 자를 범위 = 지도에서 아는 칸(모름이 아닌 칸)의 범위. EnvState::kb_* 를 안 쓰는 까닭: RobotEnv 와 시뮬의 kb_x1 이 한 칸 다를 때가 있음
      // (low_explore_check small-house 씨앗 2 걸음 70 — 관측에는 영향 없음), 지도는 비트 같음
      int kb[4] = {1 << 30, 1 << 30, -1, -1};
      for (int y = 0; y < v.W->gh; ++y)
        for (int x = 0; x < v.W->gw; ++x)
          if (v.lmap[(size_t)y * c.size + x] != sim::kUnknown) {
            kb[0] = std::min(kb[0], x); kb[1] = std::min(kb[1], y); kb[2] = std::max(kb[2], x); kb[3] = std::max(kb[3], y);
          }
      png = dec::render_png(v.lmap, c.size, v.W->gw, v.W->gh, sim::kUnknown, mdp::kOccL, kb, (s.x + s.ex - v.W->ox) / c.res, (s.y + s.ey - v.W->oy) / c.res, s.th + s.eth, cs, cells, p_.px);
    }
    r->body = dec::build_body(p_.input, p_.prompt, cs, png, c.cand_v2 != 0);
    return true;
  }
  // 답의 후보 칸 → 지금 유효 후보 번호 (그 칸에서 match_r 안, 가장 가까운 것). 없으면 −1
  int match(const lxc::View& v, const lxc::Ctl& ctl, int cell) const {
    const float q = p_.match_r / v.c->res;
    int best = -1;
    float bd = q * q;
    for (int j = 0; j < mdp::hier::K; ++j) {
      const int c2 = v.cand[j];
      if (c2 < 0 || ctl.avoided(c2, v.s->step)) continue;
      const float dx = (float)(c2 % v.W->gw - cell % v.W->gw), dy = (float)(c2 / v.W->gw - cell / v.W->gw), d = dx * dx + dy * dy;
      if (d <= bd) { bd = d; best = j; }
    }
    return best;
  }
  void restart() { const Params p = p_; *this = Upper(p); }
  bool waiting() const { return wait_; }
  long n_sent() const { return n_sent_; }
  long n_ans() const { return n_ans_; }
  long n_timeout() const { return n_timeout_; }
  long n_fail() const { return n_fail_; }
  long n_nomatch() const { return n_nomatch_; }
  long n_fallback() const { return n_fallback_; }
  long n_late() const { return n_late_; }
  double last_ms() const { return last_ms_; }
  const std::string& last_err() const { return last_err_; }
  const Params& params() const { return p_; }

 private:
  Params p_;
  Req req_;
  bool wait_ = false;
  int last_send_ = -1;
  long next_id_ = 0, n_sent_ = 0, n_ans_ = 0, n_timeout_ = 0, n_fail_ = 0, n_nomatch_ = 0, n_fallback_ = 0, n_late_ = 0;
  double last_ms_ = 0;
  std::string last_err_;
};

}  // namespace lxd
