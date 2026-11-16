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
#include <deque>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "cam_memory.h"
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
  bool cand = false;        // 후보 = scripts/cand_view.h (Ctl use_cand — build_cands · go_cand, 답은 안정 번호로 맞춤). 끄면 hier 8 칸 (예전 길)
  bool hint = true;         // 후보 모드 글의 "Rules of thumb" (가까운 쪽으로 쏠리게 함 — 평가에서 빼고도 잼)
  bool mem_image = true;    // 후보 모드 지도 사진에 지나온 길 · 지난 목표 표시 (꼴 A)
  int cam = 0;              // 후보 모드 둘째 사진: 0 = 없음 · n = 앞 n 후보마다 그 자리를 가장 잘 본 지난 카메라 장면 (cam_memory.h — 2 × 2 격자, 512 × 384)
  int image = 0;            // 후보 모드 사진 꼴: 0 = A (지금 지도 한 장) · 1 = B (2 × 2 칸: 지금 · 5 · 15 · 30 s 전, 같은 범위 — 토큰 수 같음, 칸마다 해상도 반)
  float match_r = 1.5f;     // 답의 후보 칸 → 지금 후보 (m). hier 후보 칸은 덩어리에서 로봇에 가장 가까운 칸이라 답을 기다리는 동안 옮겨감 — 0.5 m 면 답의 40 ~ 55 % 를 버림 (low_explore_check, 지연 2.6 s)
};

// 보낸 요청 하나
struct Req {
  long id = 0;
  int step = 0;
  std::string body;
  std::vector<int> cells;   // 글자 차례의 후보 칸 (후보 모드: 서는 자리 vp)
  std::vector<int> ids;     // 후보 모드: 글자 차례의 안정 번호
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
    if (p_.cand && p_.image == 1 && v.lmap && s.step % 10 == 0) {   // 꼴 B: 10 걸음마다 지도를 남김 (최근 31 장 = 30 s)
      hist_.emplace_back(s.step, std::vector<int16_t>(v.lmap, v.lmap + (size_t)v.c->size * v.c->size));
      if (hist_.size() > 31) hist_.pop_front();
    }
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
      ++n_ans_;
      if (p_.cand) {   // 후보 모드: 지금 목록을 새로 만들고 안정 번호로, 없으면 서는 자리 match_r 안
        ctl.build_cands(v);
        int idx = ctl.cand_index(req_.ids[(size_t)k]);
        if (idx < 0) idx = match_cand(v, ctl, req_.cells[(size_t)k]);
        if (idx < 0) { ++n_nomatch_; continue; }
        *cmd = ctl.go_cand(v, idx, why);
        changed = true;
        continue;
      }
      const int slot = match(v, ctl, req_.cells[k]);
      if (slot < 0) { ++n_nomatch_; continue; }
      *cmd = ctl.decide_slot(v, slot, why);
      changed = true;
    }
    if (wait_ && s.step - req_.step > p_.timeout_steps) { wait_ = false; ++n_timeout_; }
    bool urgent = false;
    if (!changed && ctl.due(s) && (ctl.redo_pending() || !ctl.live(s))) {   // 기다리지 않고 slot0 차례로 (combine_run 의 slot0 위층과 같은 고름)
      if (p_.cand) { ctl.build_cands(v); *cmd = ctl.go_cand(v, -1, why); }   // 후보 모드 규칙(cand_pick)
      else *cmd = ctl.decide_slot(v, -1, why);
      changed = true;
      urgent = true;
      ++n_fallback_;
    }
    if (!wait_ && (urgent || last_send_ < 0 || s.step - last_send_ >= p_.period_steps)) {
      last_send_ = s.step;
      Req r;
      if (p_.cand ? build_cand(v, ctl, &r) : build(v, F, ctl, &r)) {
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
  // 후보 모드 요청 (scripts/cand_view.h 후보 — 필드 이름은 cand::Cand 그대로 뜻을 풀어 씀). 기억: 후보마다 기록 + 최근 목표 5 개 + 사진의 지나온 길 · 지난 목표
  bool build_cand(const lxc::View& v, lxc::Ctl& ctl, Req* r) const {
    const cand::Out& co = ctl.build_cands(v);
    if (co.c.size() < 2) return false;
    const sim::EnvState& s = *v.s;
    const sim::EnvCfg& c = *v.c;
    const double dt = (double)c.substeps * c.dt;
    const float rx = s.x + s.ex, ry = s.y + s.ey, rth = s.th + s.eth;
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
    for (size_t i = 0; i < co.c.size() && i < 26; ++i) {
      const cand::Cand& q = co.c[i];
      dec::Cand d;
      d.key = (char)('A' + (int)i);
      d.slot = (int)i;
      dc.push_back(d);
      cells.push_back(q.vp % v.W->gw);
      cells.push_back(q.vp / v.W->gw);
      r->cells.push_back(q.vp);
      r->ids.push_back(q.id);
      oj o;
      o["id"] = std::string(1, d.key);
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
      o["times_chosen"] = q.chosen;
      o["times_failed"] = q.aborts;
      o["last_failure"] = why_txt[std::clamp(q.last_fail, 0, 8)];
      o["seconds_since_failure"] = q.since_fail_s < 0 ? oj(nullptr) : oj(rnd(q.since_fail_s, 1.0));
      o["failed_spots_nearby"] = q.fail_near;
      o["inside_avoid_zone"] = q.in_avoid != 0;
      o["robot_was_here_before"] = q.visited != 0;
      cl.push_back(o);
    }
    // 최근 목표 (cand::Mem::goals, 새것부터 5 개)
    const cand::Mem& M = ctl.cand_mem();
    oj gl = oj::array();
    dec::MapMem mm;
    mm.clear_tags = true;
    const auto cxy = [&](float x, float y, float* cx, float* cy) { *cx = (x - v.W->ox) / c.res; *cy = (y - v.W->oy) / c.res; };
    for (int i = (int)M.goals.size() - 1, n = 0; i >= 0 && n < 5; --i, ++n) {
      const cand::Mem::Goal& g = M.goals[(size_t)i];
      const auto [dist, ang] = rel(g.x, g.y);
      oj o;
      o["distance_from_robot_m"] = rnd(dist, 0.1);
      o["bearing_deg_left_positive"] = ang;
      o["outcome"] = g.end < 0 ? "in progress" : why_txt[std::clamp(g.why, 0, 8)];
      o["started_s_ago"] = rnd((s.step - g.start) * dt, 1.0);
      if (g.end >= 0) o["ended_s_ago"] = rnd((s.step - g.end) * dt, 1.0);
      gl.push_back(o);
    }
    if (p_.mem_image) {
      for (const auto& q : M.trail) {
        dec::MapMem::P pp;
        cxy(q.x, q.y, &pp.cx, &pp.cy);
        pp.age_s = (float)((s.step - q.step) * dt);
        mm.trail.push_back(pp);
      }
      for (size_t gi = M.goals.size() > 10 ? M.goals.size() - 10 : 0; gi < M.goals.size(); ++gi) {   // 최근 10 개만 (사진이 어지럽지 않게)
        const cand::Mem::Goal& g = M.goals[gi];
        dec::MapMem::G gg;
        cxy(g.x, g.y, &gg.cx, &gg.cy);
        gg.outcome = g.end < 0 ? 0 : g.why == cand::kReached ? 1 : g.why == cand::kSwitched ? 3 : 2;
        mm.goals.push_back(gg);
      }
    }
    oj st;
    st["task"] = "A small wheeled robot is exploring an unknown indoor floor. Its job is to map the whole floor in the shortest total time. It drives at about 1 m/s and "
                 "turning around costs a few seconds. A frontier is the edge between mapped free space and unmapped space; each candidate is a spot from which a frontier can be seen.";
    std::vector<uint8_t> png;
    if (p_.input != dec::kText) {
      int kb[4] = {1 << 30, 1 << 30, -1, -1};
      for (int y = 0; y < v.W->gh; ++y)
        for (int x = 0; x < v.W->gw; ++x)
          if (v.lmap[(size_t)y * c.size + x] != sim::kUnknown) {
            kb[0] = std::min(kb[0], x); kb[1] = std::min(kb[1], y); kb[2] = std::max(kb[2], x); kb[3] = std::max(kb[3], y);
          }
      if (p_.image == 1) {   // 꼴 B: 지난 지도 셋 (5 · 15 · 30 s 전 쯤 — 10 걸음 칸에서 가장 가까운 것), 그때 자세는 기억의 자취에서
        std::vector<const int16_t*> past;
        std::vector<float> age, pose;
        for (int ago : {50, 150, 300}) {
          const std::pair<int, std::vector<int16_t>>* best = nullptr;
          for (const auto& q : hist_) if (q.first <= s.step - ago && (!best || q.first > best->first)) best = &q;
          if (!best && !hist_.empty()) best = &hist_.front();
          if (!best) continue;
          past.push_back(best->second.data());
          age.push_back((float)((s.step - best->first) * dt));
          float px_ = (rx - v.W->ox) / c.res, py_ = (ry - v.W->oy) / c.res, pth = rth;
          int bd = 1 << 30;
          for (const auto& t : M.trail)
            if (std::abs(t.step - best->first) < bd) { bd = std::abs(t.step - best->first); px_ = (t.x - v.W->ox) / c.res; py_ = (t.y - v.W->oy) / c.res; pth = t.th; }
          pose.push_back(px_); pose.push_back(py_); pose.push_back(pth);
        }
        png = dec::render_png_tiles(v.lmap, past, age, pose, c.size, v.W->gw, v.W->gh, sim::kUnknown, mdp::kOccL, kb, (rx - v.W->ox) / c.res, (ry - v.W->oy) / c.res, rth, dc,
                                    cells, p_.px, mm);
      } else {
        png = dec::render_png_mem(v.lmap, c.size, v.W->gw, v.W->gh, sim::kUnknown, mdp::kOccL, kb, (rx - v.W->ox) / c.res, (ry - v.W->oy) / c.res, rth, dc, cells, p_.px, mm);
      }
      st["map_image"] = std::string("The image is the robot's current occupancy map seen from above: white = free space, black = wall or obstacle, gray = not mapped yet. "
                                    "The blue arrow is the robot, pointing where it is heading. Each candidate spot is a small yellow dot; its letter is in a colored circle "
                                    "joined to that dot by a line of the same color (each letter has its own color).") +
                        (p_.mem_image ? " Purple dots are where the robot drove in the last 60 seconds: dark purple = recent, faint purple = older. Past goals are marked: "
                                        "green check = reached, red X = failed (the robot got stuck or had no path), gray ring = left for another goal, blue ring = current goal."
                                      : "") +
                        (p_.image == 1 ? " The image is split into four panels showing the same map area at the same scale, so positions line up: top-left = now (with the "
                                         "candidates, the path and the goals), top-right, bottom-left and bottom-right = the map a few seconds ago (the time is written in each "
                                         "panel, e.g. -5s); in those older panels the blue arrow is where the robot was then."
                                       : "");
    }
    if (p_.input != dec::kImage) {
      st["candidate_fields"] = "travel_time_s = estimated driving time including turning; new_area_m2 = unmapped area visible from the spot; "
                               "new_area_not_seen_from_other_candidates_m2 = part of it that no better-ranked candidate also sees; frontier_width_m = width of the opening; "
                               "free_space_around_spot_m = distance from the spot to the nearest obstacle; narrowest_gap_on_way_m = tightest squeeze on the way; "
                               "times_failed / last_failure / seconds_since_failure = what happened when the robot tried this spot before; failed_spots_nearby = failed goals within 1 m; "
                               "inside_avoid_zone = near a spot the robot is currently avoiding after a failure; robot_was_here_before = the robot already stood near it.";
      st["candidates"] = cl;
      st["recent_goals_newest_first"] = gl;
    }
    if (p_.hint)
      st["hint"] = "Rules of thumb: prefer a spot that is close and roughly ahead; finish nearby small frontiers before leaving an area because coming back later is "
                   "expensive; keep the current target unless another candidate is clearly better; avoid spots that failed recently.";
    std::vector<uint8_t> cpng;
    if (p_.cam > 0 && cam_ && p_.input != dec::kText) {   // 둘째 사진: 후보별 지난 장면 (기하 검색)
      std::vector<dec::CamTile> tiles;
      const int n = std::min<int>(p_.cam, (int)co.c.size());
      for (int i = 0; i < n; ++i) {
        const cand::Cand& q = co.c[(size_t)i];
        const float wx = v.W->ox + (q.vp % v.W->gw + 0.5f) * c.res, wy = v.W->oy + (q.vp / v.W->gw + 0.5f) * c.res;
        dec::CamTile t;
        t.key = (char)('A' + i);
        camm::Hit h;
        if (cam_->find(wx, wy, s.step, dt, &h)) {
          const camm::Frame& fr = cam_->frames()[(size_t)h.idx];
          t.rgb = fr.rgb.data(); t.w = cam_->in.w; t.h = cam_->in.h; t.u = h.u; t.v = h.v; t.age_s = h.age_s;
        }
        tiles.push_back(t);
      }
      const int cols = n <= 1 ? 1 : 2, rows = (n + 1) / 2;
      cpng = dec::render_cam_grid(tiles, 512, 192 * rows, cols, rows);
      st["camera_image"] = "The second image shows real camera photos from the robot's front camera, one panel per candidate (the first candidates by letter): each panel is "
                           "the closest recent photo in which that candidate spot was visible, labelled with the candidate letter (top-left) and how many seconds ago it was "
                           "taken (top-right); the yellow dot with a black ring marks the candidate spot. A plain gray panel means the camera has not seen that spot yet.";
    }
    oj body;
    body["state"] = st;
    if (!png.empty()) body["images"] = oj::array({dec::png_uri(png)});
    if (!cpng.empty()) body["images"].push_back(dec::png_uri(cpng));
    oj crit = oj::object();
    for (const auto& d : dc) crit[std::string(1, d.key)] = nullptr;
    body["questions"]["pick"] = oj{{"type", "choice"}, {"instructions", "Which candidate spot should the robot drive to next so that the whole floor gets mapped in the least total time?"},
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
  void restart() { const Params p = p_; const camm::Memory* c = cam_; *this = Upper(p); cam_ = c; }
  bool waiting() const { return wait_; }
  int last_send() const { return last_send_; }   // 마지막으로 보낸 걸음 (−1 = 아직)
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
  void set_input(int input) { p_.input = input; }
  void set_cam(const camm::Memory* m) { cam_ = m; }   // 카메라 키프레임 기억 (부르는 쪽이 걸음마다 채움 — 실물 노드)   // 요청 꼴만 바꿈 (갈래 굴리기: 기준 응답기는 그림을 안 봄 — 지도 없이 글만)

 private:
  Params p_;
  Req req_;
  bool wait_ = false;
  int last_send_ = -1;
  const camm::Memory* cam_ = nullptr;
  std::deque<std::pair<int, std::vector<int16_t>>> hist_;   // 꼴 B 지난 지도 (걸음, 지도)
  long next_id_ = 0, n_sent_ = 0, n_ans_ = 0, n_timeout_ = 0, n_fail_ = 0, n_nomatch_ = 0, n_fallback_ = 0, n_late_ = 0;
  double last_ms_ = 0;
  std::string last_err_;
};

}  // namespace lxd
