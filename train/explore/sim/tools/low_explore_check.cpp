// 실물 합친 탐사 노드(위층 + 아래층 규칙 추종기 — limo_fire_explore explorer) == CPU 시뮬 합친 탐사 (combine_run --upper fuel|slot0 와 같은 설정 — 판단 모델 위층 --upper decider 는 이 도구 · dec_rollout 에만).
// 실물 노드의 위층은 판단 모델(decider)과 그 대신 고르는 규칙(cand) 뿐 — fuel · slot0 은 시뮬 기준선 (같은 Ctl 코드라 비교는 그대로 뜻이 있음).
// 시뮬(환경 하나, CPU 참조판)과 RobotEnv(합친 탐사 모드)가 각자 자기 상태로 위층 결정(scripts/low_explore_ctl.h — combine_run 의 환경 하나 몫)을 하고,
// 걸음마다 시뮬 라이다 장·감각·깊이를 RobotEnv 에 넣어 두 쪽의 위층 명령(목표 칸·follow)·준비 뒤 경로(목표 칸·경로 점)·명령(v, ω)·판 끝을 비트 비교한다.
// 아래층은 두 쪽 다 규칙 추종기 (준비 단계 행동 칸 = 위층 Ctl::slot() — combine_run 의 act[0], RobotEnv::low_act 계층형 가지). 관측 설정은 env.yaml 그대로.
//   low_explore_check <장면.scene> … [--steps 1500] [--seed 1] [--people 0] [--upper fuel|slot0|decider] [--fuel-v 0.981 --fuel-w 3.232 --fuel-t0 0.10] [--fuel-time 1 --fuel-global 1 --fuel-vp 1]
//   --upper slot0: combine_run --upper slot0 (피하지 않는 첫 후보 · stick — 기준선, 실물 노드에는 없음)
//   fuel 기본 = 실물 노드가 fuel 을 갖던 때의 config/explorer.yaml (시간 + 전역 + 시점, 붙들기 0.2, 시점 자격 0.75 m · 0.21 m). --fuel-hyst · --fuel-vp-min · --fuel-vp-clear · --fuel-first 로 바꿈
//   --contact-log 파일: 판이 닿음(2)으로 끝나면 몸통 8 cm 안 막힌 칸(정답 격자)이 라이다 높이 형상인지 · 로봇 지도·깊이 층이 알았는지 · 그 형상 물체·높이 · 마지막 2 s 자취,
//     진행 없음(6)으로 끝나면 마지막 65 s 를 1 s 마다 (자세·목표·경로·시계·닫힘·지나온 거리). LOW_CHECK_NNEW=파일: 걸음마다 새로 안 칸·새 막힘·속도·경로·계획 단계·검사 멈춤
//   --log 파일: 시뮬 쪽 결정마다 한 줄 (판·시각·목표·바뀜·뜀·목표 없음 까닭·닫힘) — 실물 녹화 low_explore_replay --log 와 같은 꼴로 견줌
//   --upper decider [--dec-period 3.0] [--dec-latency 2.6] [--dec-input both] [--dec-cand 1] [--dec-hint 1] [--dec-mem-image 1] [--dec-image A|B] [--dec-log-k 8]: 실물 노드 위층 (scripts/low_decider.h) — 판단 모델 대신 정해진 가짜 응답기
//     (요청 본문의 FNV 해시로 글자를 고르고 --dec-latency 초 뒤 걸음에 답을 냄). 두 쪽 요청 본문도 비트 비교. 시뮬이 판단 모델 답을 몇 걸음 뒤에 받는 것과 같은 길
//     --dec-url http://127.0.0.1:8171: 가짜 대신 진짜 llama-server (실물 노드와 같은 비차단 소켓 limo_fire_explore/decider_client.h). 시뮬 시계는 답을 기다리는 동안 멈추고,
//     답은 --dec-latency 초 뒤 걸음에 줌 (실물 답 시간을 시뮬 걸음으로). 실물 쪽은 같은 답을 같은 차례로 받음 — 판단 모델 위층을 시뮬에서 실물과 같은 코드로 평가
// 지도 자세 오차는 끈다 (실물 노드는 지도 자세를 그대로 받으므로). 판이 끝나면 둘 다 새 판 (위층 기억도 비움).
#include <chrono>
#include <cmath>
#include <memory>
#include <thread>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "env_api.h"
#include "limo_fire_explore/decider_client.h"
#include "limo_fire_explore/robot_env.h"
#include "coverage.h"
#include "low_decider.h"
#include "obj_follow.h"
#include "task_sup.h"
#include "low_explore_ctl.h"
#include "cand_view.h"
#include "cand_quality.h"

namespace {
constexpr int kRoughDone = 11;   // --ep-log 끝 코드: 후보 모드 "대충 다 앎" (Ctl::rough_done)
// 대충 앎 넓이 (사용자 목표 잣대): 정답 바닥 칸이 든 0.5 m 굵은 칸 가운데 그 바닥 칸의 20 % 이상을 로봇 지도가 아는 칸 몫 (판단 모델 담당 scripts/coverage.h 의 "0.5 m @ 20 %" 와 같은 뜻)
double rough_cover(const int16_t* L, int stride, const uint8_t* grid, int gw, int gh, float res) {
  const int B = std::max(1, (int)std::lround(0.5f / res)), bw = (gw + B - 1) / B, bh = (gh + B - 1) / B;
  std::vector<int> fl((size_t)bw * bh, 0), kn((size_t)bw * bh, 0);
  for (int y = 0; y < gh; ++y)
    for (int x = 0; x < gw; ++x)
      if (grid[(size_t)y * gw + x] & sim::kCellFloor) {
        const int q = (y / B) * bw + x / B;
        fl[q] += 1;
        kn[q] += L[(size_t)y * stride + x] != mdp::kUnknownL;
      }
  int nb = 0, nk = 0;
  for (size_t q = 0; q < fl.size(); ++q) if (fl[q] > 0) { ++nb; nk += kn[q] * 5 >= fl[q]; }
  return nb ? (double)nk / nb : 0.0;
}
// 실물 쪽 걸음 시간 (이 기기 한 코어) — RobotEnv::step(라이다 장 → 지도·관측·닫힘) · low_prepare · 위층 결정(Ctl::decide)
double t_step = 0, t_obs = 0, t_dec = 0, t_dec_max = 0;
long n_reuse = 0;   // 실물 쪽 후보 뽑기가 같은 걸음 관측의 프론티어 찾기를 쓴 결정 수 (RobotEnv::reach_keep)
long n_step = 0, n_tdec = 0;
// 가짜 판단 모델: 본문 해시로 글자, lat 걸음 뒤 답 (now 는 부르는 쪽이 걸음마다 맞춤)
struct FakeModel : lxd::Transport {
  int lat = 26, now = 0;
  bool first = false;   // --dec-fake A: 늘 글자 A (후보 0 번 = 규칙 1 순위) — 지연 · 주기만의 비용을 잼
  bool follow = false;  // --dec-fake follow: kind follow 후보가 있으면 그 첫 글자 (따라가기 시험), 없으면 해시
  int fault = 0;        // --dec-fake down: 바로 실패(서버 꺼짐) · garbled: 지연 뒤 꼴 틀린 답 · oor: 없는 글자 'Z' · obj: kind object 첫 글자(없으면 해시)
  bool obj = false;
  bool stopk = false;   // --dec-fake stop: kind stop 후보가 있으면 그 글자 (멈춤 시험), 없으면 해시
  struct P { int ready; lxd::Answer a; std::string body; };
  std::vector<P> q;
  std::vector<std::string> sent;   // 보낸 본문 (두 쪽 비교)
  limo_fire_explore::DeciderClient* real = nullptr;   // 진짜 서버 (시뮬 쪽만 — 답을 기다림)
  std::vector<lxd::Answer>* log = nullptr;       // 진짜 답 기록 (시뮬 쪽이 씀, 실물 쪽이 같은 차례로 읽음)
  size_t replay = 0;
  FILE* blog = nullptr;   // DEC_BODY_LOG=파일: 시뮬 쪽 요청마다 "걸음 본문해시 까닭" 한 줄 (빌드 사이 요청 비교)
  void send(const lxd::Req& r) override {
    if (blog) {
      uint64_t hb = 1469598103934665603ull;
      for (unsigned char ch : r.body) { hb ^= ch; hb *= 1099511628211ull; }
      std::fprintf(blog, "%d %016llx %s\n", r.step, (unsigned long long)hb, r.why.c_str());
    }
    if (real || log) {
      lxd::Answer a;
      if (real) {
        real->send(r);
        while (!real->poll(&a)) std::this_thread::sleep_for(std::chrono::milliseconds(2));
        log->push_back(a);
      } else if (replay < log->size()) {
        a = (*log)[replay++];
      } else {
        a.err = "답 기록 없음";
      }
      a.id = r.id;
      q.push_back({now + lat, a, r.body});
      sent.push_back(r.body);
      return;
    }
    uint64_t h = 1469598103934665603ull;
    for (unsigned char ch : r.body) { h ^= ch; h *= 1099511628211ull; }
    lxd::Answer a;
    a.id = r.id;
    if (fault == 1) { a.ok = false; a.err = "connect: refused"; q.push_back({now, a, r.body}); sent.push_back(r.body); return; }
    if (fault == 2) { a.ok = false; a.err = "answer shape"; q.push_back({now + lat, a, r.body}); sent.push_back(r.body); return; }
    if (fault == 3) { a.ok = true; a.choice = "Z"; q.push_back({now + lat, a, r.body}); sent.push_back(r.body); return; }
    a.ok = true;
    a.choice = std::string(1, (char)('A' + (first ? 0 : (int)(h % (uint64_t)(r.cells.size())))));
    if (follow || obj || stopk) {
      const std::string kind = follow ? "follow" : obj ? "object" : "stop";
      const auto j = nlohmann::json::parse(r.body, nullptr, false);
      if (j.is_object() && j.contains("state") && j["state"].is_object() && j["state"].contains("candidates")) {
        for (const auto& c : j["state"]["candidates"])
          if (c.contains("kind") && c["kind"] == kind) { a.choice = c["id"].get<std::string>(); break; }
      } else if (j.is_object() && j.contains("state") && j["state"].is_string()) {   // 꼴 2: 후보 줄 "글자 종류 …"
        const std::string t = j["state"].get<std::string>();
        for (size_t i = t.find("\nCandidates:"); i != std::string::npos && (i = t.find('\n', i + 1)) != std::string::npos;) {
          if (t.compare(i + 2, kind.size() + 1, " " + kind) == 0 && (i + 3 + kind.size() >= t.size() || t[i + 3 + kind.size()] == ' ' || t[i + 3 + kind.size()] == '\n')) {
            a.choice = t.substr(i + 1, 1);
            break;
          }
          if (i + 1 < t.size() && t[i + 1] == '\n') break;   // 후보 줄 끝 (빈 줄)
        }
      }
    }
    q.push_back({now + lat, a, r.body});
    sent.push_back(r.body);
  }
  bool poll(lxd::Answer* a) override {
    if (q.empty() || q.front().ready > now) return false;
    *a = q.front().a;
    q.erase(q.begin());
    return true;
  }
};
void write_text(const std::string& p, const std::string& t) { std::FILE* f = std::fopen(p.c_str(), "wb"); if (f) { std::fwrite(t.data(), 1, t.size(), f); std::fclose(f); } }
struct Tap : sim::CpuTap {
  limo_fire_explore::RobotEnv* robot = nullptr;
  limo_fire_explore::Scan sc;
  bool have = false;
  void scan(int, const sim::EnvState& e, const sim::Shared& sh, const std::vector<float>& r, const std::vector<float>& inten) override {
    sc.ranges = r;
    sc.intensities = inten;
    sc.angle_min = -0.5f * e.noise.fov;
    sc.angle_increment = sh.inc;
    sc.n_rev = sh.scan.n_rev;
    sc.scan_time = e.noise.scan_time;
    sc.time_increment = e.noise.scan_time / sh.scan.n_rev;
    sc.x = e.x + e.ex;
    sc.y = e.y + e.ey;
    sc.th = e.th + e.eth;
    sc.w = e.w;
    have = true;
  }
  void step_end(int, const sim::EnvState& e) override {
    if (!have) throw std::runtime_error("걸음에 라이다 장이 없음");
    limo_fire_explore::Proprio p;
    p.wheel_v = e.drive.wv;
    p.wheel_vy = e.drive.wvy;
    p.wheel_w = e.drive.ww;
    p.gyro_z = e.gyro;
    for (int i = 0; i < 7; ++i) p.imu[i] = e.imu.out[i];
    { const auto t0 = std::chrono::steady_clock::now(); robot->step(p, sc); t_step += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count(); ++n_step; }
    have = false;
  }
};
}  // namespace

int main(int argc, char** argv) {
  setenv("SIM_EXACT", "1", 0);
  std::vector<std::string> scenes;
  std::string env_path = SIM_CONFIG_DIR "/env.yaml", model_path = SIM_CONFIG_DIR "/sensor_model.yaml";
  int steps = 1500, people = 0;
  uint64_t seed = 1;
  bool use_fuel = true, use_dec = false, use_cand = false, dec_hint = true, dec_mimg = true, node_obs = false, fc_on = true;
  int fc_check = 0;
  int dec_img = 0, dec_logk = lxd::Params{}.log_k;   // 기본은 노드와 같은 lxd::Params
  double dec_period = 3.0, dec_lat = 2.6, dec_match = -1.0;
  int dec_hold = lxd::Params{}.hold_extra;
  float dec_lead = lxd::Params{}.ask_lead_s;
  int dec_mode = 1, dec_max = 6, dec_order = 0;   // 실물 노드 기본과 같게 (decider_mode event · decider_max_cands 6)
  bool dec_compact = lxd::Params{}.compact, dec_memf = true;
  int dec_fmt = lxd::Params{}.fmt;
  bool dec_stop = lxd::Params{}.stop_cand;
  int dec_px = lxd::Params{}.px;
  std::string dec_input = "both", dec_url;
  std::string ep_log, lat_log, contact_log;   // --lat-log: 판 끝마다 (판 · 끝 코드 · 끝 시각 · 규칙 대신 · 답 기다리는 중 닿음 · 보냄 · 답 적용 · 실패 기록 후보 고름 · 1 순위 고름 · 정답 바닥 칸 수 · 덮음 지표 9 개(가는 칸, 굵은 0.5 m × 문턱 0.1/0.2/0.3/0.5, 굵은 1.0 m × 같은 문턱 — scripts/coverage.h)를 1 s 마다: 지표 사이 ; 값 사이 ,)
  bool dec_fake_a = false, dec_fake_follow = false, dec_fake_obj = false, dec_fake_stop = false, follow_on = false, goto_on = false;
  tsk::Params tsp;
  int obj_move_step = -1, instr_at_step = -1;
  std::string instr_at_text;
  float obj_move_dx = 0.f, obj_move_dy = 0.f;
  int dec_fault = 0;
  float follow_fov = 1.159f, follow_range = 5.0f;   // --ep-log: 판 끝마다 한 줄 (판 · 끝 코드 · 시각 s · 위층 결정 · 목표 바뀜) — --max-ep N 판에서 멈춤
  std::string dec_instr;   // --dec-instr "글": 판단 모델 물체 찾기 지시 (실물 노드 ~/instruction) — 요청 본문 비트 비교 확인용
  std::vector<std::string> dec_objs;
  std::string dec_hint_cls;
  bool dec_obj_always = false;   // --dec-obj "클래스,x,y" (여러 번): 지시가 있을 때 물체 후보 (실물 노드는 물체 지도에서) — 요청 비트 비교 확인용
  int max_ep = 0;
  bool close_lidar = false;
  bool cand_on = false, open_scene = false;
  std::string cand_dump;
  int cand_every = 20;
  cand::Opt co;
  int cand_commit = 0;   // --cand-* 로 바꿈 (기본 = cand_view.h)
  std::string logp;   // --log: 시뮬 쪽 결정마다 한 줄 (판 · 시각 · 목표 · 바뀜 · 뜀 · 목표 없음 까닭 · 닫힘) — 실물 녹화 low_explore_replay --log 와 견줌
  fu::Opt fo;
  fo.time = fo.global = fo.vp = true;   // 고른 조합 (scripts/README "combine_run — 위층 fuel": 시간 + 전역 + 시점)
  fo.v_eff = 0.981f;
  fo.w_eff = 3.232f;
  fo.t0 = 0.10f;
  fo.hyst = 0.2f;      // 실물 노드가 fuel 위층을 갖던 때(~2026-11-17)의 기본 fuel_hyst · fuel_vp_min · fuel_vp_clear
  fo.vp_min = 0.75f;
  fo.vp_clear = 0.21f;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    auto nx = [&]() -> std::string { if (i + 1 >= argc) throw std::runtime_error(a + " 뒤에 값"); return argv[++i]; };
    if (a == "--steps") steps = std::stoi(nx());
    else if (a == "--seed") seed = std::stoull(nx());
    else if (a == "--people") people = std::stoi(nx());
    else if (a == "--env") env_path = nx();
    else if (a == "--log") logp = nx();
    else if (a == "--open-scene") open_scene = true;   // 닫히지 않은 장면도 (실물 방 재현 room_run 장면 — 판 끝은 로봇 지도만 봄)
    else if (a == "--cand") cand_on = true;
    else if (a == "--cand-dump") { cand_on = true; cand_dump = nx(); }   // 결정 --cand-every 째마다 지도·후보 그림 재료 (cand_quality.h dump)
    else if (a == "--cand-every") cand_every = std::stoi(nx());
    else if (a == "--cand-gain-m") co.gain_m = std::stof(nx());
    else if (a == "--cand-algo") {   // 후보 만드는 법 (cand::Algo — 기본 view)
      const std::string v = nx();
      const char* nm[] = {"view", "center", "nearest", "tour", "decay", "sector"};
      co.algo = -1;
      for (int k = 0; k < 6; ++k) if (v == nm[k]) co.algo = k;
      if (co.algo < 0) throw std::runtime_error("--cand-algo 는 view · center · nearest · tour · decay · sector");
    }
    else if (a == "--cand-k") co.k = std::stoi(nx());
    else if (a == "--cand-vp-min") co.vp_min = std::stof(nx());
    else if (a == "--cand-margin") co.margin = std::stof(nx());
    else if (a == "--cand-coarse") co.coarse_m = std::stof(nx());
    else if (a == "--cand-commit") cand_commit = std::stoi(nx());
    else if (a == "--cand-keep") co.keep = std::stof(nx());
    else if (a == "--cand-known") co.known_frac = std::stof(nx());
    // 위층 결정 때마다 후보 생성(scripts/cand_view.h)을 두 쪽에서 — 비트 비교 · 시간 · 안정성 (주행은 그대로)
    else if (a == "--close-lidar") close_lidar = true;   // 닫힘을 라이다 지도만으로 (EnvCfg::low_explore 2 — 실물 노드 close_lidar)
    else if (a == "--node-obs") node_obs = true;   // 실물 쪽 RobotEnv 를 노드처럼 (set_obs — cand_view 후보 위층이면 계층형 후보 끔) — 그래도 시뮬과 같은지
    else if (a == "--fc-check") fc_check = std::stoi(nx());   // 실물 쪽 걸음 사이 프론티어 칸 판정 기억을 N 걸음마다 처음부터 다시 센 값과 견줌 (frontier_cache.h)
    else if (a == "--no-fc") fc_on = false;                   // 실물 쪽 기억 끔 (걸음마다 처음부터)
    else if (a == "--dec-hint") dec_hint = std::stoi(nx()) != 0;
    else if (a == "--dec-mem-image") dec_mimg = std::stoi(nx()) != 0;
    else if (a == "--dec-image") dec_img = nx() == "B" ? 1 : 0;
    else if (a == "--dec-log-k") dec_logk = std::stoi(nx());
    else if (a == "--upper") { const std::string u = nx(); if (u != "fuel" && u != "slot0" && u != "decider" && u != "cand") throw std::runtime_error("--upper 는 fuel · slot0 · decider · cand"); use_fuel = u == "fuel"; use_dec = u == "decider"; use_cand = u == "cand"; }
    else if (a == "--dec-period") dec_period = std::stod(nx());
    else if (a == "--dec-mode") dec_mode = nx() == "event" ? 1 : 0;
    else if (a == "--dec-compact") dec_compact = std::stoi(nx()) != 0;
    else if (a == "--dec-fmt") dec_fmt = std::stoi(nx());             // 요청 꼴 1 (JSON 객체) · 2 (글 하나 — lxd::Params::fmt)
    else if (a == "--dec-stop") dec_stop = std::stoi(nx()) != 0;      // 지시가 있을 때 멈춤 후보 (lxd::Params::stop_cand)
    else if (a == "--dec-px") dec_px = std::stoi(nx());
    else if (a == "--dec-memf") dec_memf = std::stoi(nx()) != 0;
    else if (a == "--dec-max") dec_max = std::stoi(nx());
    else if (a == "--dec-order") dec_order = nx() == "dist" ? 1 : 0;
    else if (a == "--dec-latency") dec_lat = std::stod(nx());
    else if (a == "--dec-hold") dec_hold = std::stoi(nx());     // 목표가 죽었는데 답이 오는 중이면 기다릴 덤 걸음 (lxd::Params::hold_extra, −1 = 바로 규칙 — 옛 동작)
    else if (a == "--dec-lead") dec_lead = std::stof(nx());     // event: 남은 시간이 답 지연 + 이것(s) 이하면 미리 물음 (Params::ask_lead_s, −1 = 끔 — 옛 동작)
    else if (a == "--dec-input") dec_input = nx();
    else if (a == "--dec-match") dec_match = std::stod(nx());
    else if (a == "--dec-url") dec_url = nx();
    else if (a == "--ep-log") ep_log = nx();
    else if (a == "--contact-log") contact_log = nx();
    else if (a == "--lat-log") lat_log = nx();
    else if (a == "--dec-fake") { const std::string f = nx(); dec_fake_a = f == "A"; dec_fake_follow = f == "follow"; dec_fake_obj = f == "obj"; dec_fake_stop = f == "stop";
                                  dec_fault = f == "down" ? 1 : f == "garbled" ? 2 : f == "oor" ? 3 : 0; }
    else if (a == "--follow") follow_on = true;
    else if (a == "--goto") goto_on = true;
    else if (a == "--instr-at") { const std::string v = nx(); const size_t c = v.find(':'); if (c == std::string::npos) throw std::runtime_error("--instr-at 걸음:글");
                                  instr_at_step = std::stoi(v.substr(0, c)); instr_at_text = v.substr(c + 1); }   // 시험: 이 걸음에 지시를 바꿈 (빈 글 = 지움)                     // 물체 답이면 노드처럼 그 설 자리로 감 (task_sup.h — 닿음 · 못 감 · 지시 포기를 셈)
    else if (a == "--goto-stall-s") tsp.stall_s = std::stof(nx());     // 시험: 못 감 판정 시간 (작게 하면 실패 → 빼기 → 다시 물음 → 포기 길을 탐)
    else if (a == "--instr-limit") tsp.instr_limit_s = std::stof(nx());  // 지시 시간 한도 (s)
    else if (a == "--obj-move") { if (std::sscanf(nx().c_str(), "%d,%f,%f", &obj_move_step, &obj_move_dx, &obj_move_dy) != 3) throw std::runtime_error("--obj-move 걸음,dx,dy"); }                 // 시뮬 사람을 물체 지도 사람처럼(카메라 시야·가림) 판단 모델 follow 후보로, 고르면 obj_follow.h 로 따라감 (--people 과 같이)
    else if (a == "--follow-fov") follow_fov = std::stof(nx()); // 카메라 가로 화각 (rad) — docs/hardware.md 컬러 66.4°
    else if (a == "--follow-range") follow_range = std::stof(nx());   // 사람 보는 거리 (m) — 물체 지도(object_map) max_depth 5.0
    else if (a == "--dec-instr") dec_instr = nx();
    else if (a == "--dec-obj") dec_objs.push_back(nx());
    else if (a == "--dec-class-hint") dec_hint_cls = nx();          // 지시의 클래스 힌트 (Params::class_hint — 그 클래스 물체가 물체 후보 칸 앞)
    else if (a == "--dec-obj-always") dec_obj_always = true;  // 지시 없어도 물체 후보 (Params::obj_always)
    else if (a == "--max-ep") max_ep = std::stoi(nx());
    else if (a == "--fuel-v") fo.v_eff = std::stof(nx());
    else if (a == "--fuel-w") fo.w_eff = std::stof(nx());
    else if (a == "--fuel-t0") fo.t0 = std::stof(nx());
    else if (a == "--fuel-hyst") fo.hyst = std::stof(nx());
    else if (a == "--fuel-vp-min") fo.vp_min = std::stof(nx());
    else if (a == "--fuel-vp-clear") fo.vp_clear = std::stof(nx());
    else if (a == "--fuel-first") fo.first = std::stoi(nx()) != 0;
    else if (a == "--fuel-time") fo.time = std::stoi(nx()) != 0;
    else if (a == "--fuel-global") fo.global = std::stoi(nx()) != 0;
    else if (a == "--fuel-vp") fo.vp = std::stoi(nx()) != 0;
    else scenes.push_back(a);
  }
  if (scenes.empty()) {
    std::fprintf(stderr, "low_explore_check <장면.scene> … [--upper fuel|slot0|decider] [--steps 1500] [--seed 1] [--people 0] [--fuel-*]\n");
    return 2;
  }
  try {
    sim::EnvHostCfg hc = sim::load_env_cfg(env_path);
    // combine_run.cpp:224-231 와 같은 설정 (max_steps 는 판 길이라 관측과 무관 — 여기서는 env.yaml 그대로)
    hc.cfg.action_mode = sim::kActPath;
    hc.cfg.low_stat = hc.cfg.low_task = hc.cfg.low_rl = 1;
    hc.cfg.low_explore = close_lidar ? 2 : 1;
    hc.cfg.pe_xy = hc.cfg.pe_yaw = 0.f;
    const sim::LidarModel lidar = sim::load_lidar_model(model_path);
    std::vector<sim::World> worlds;
    for (const auto& p : scenes) worlds.push_back(sim::build_world(p, sim::load_scene(p), hc, !open_scene));
    // --contact-log: 판이 닿음(끝 코드 2)으로 끝나면 무엇에 닿았는지 — 몸통 둘레 막힌 칸(정답 격자)마다 라이다 높이 형상이 있는지 · 로봇 지도·깊이 층이 알았는지, 마지막 2 s 자취
    std::vector<std::vector<uint8_t>> lid_geo;   // 세계마다 라이다 높이(lidar_z ± 1 cm) 형상이 지나는 칸
    if (!contact_log.empty())
      for (const auto& W : worlds) {
        std::vector<uint8_t> g((size_t)W.gw * W.gh, 0);
        for (const sim::Tri& t : W.scene.tris) {
          const float lo = std::min(t.a.z, std::min(t.b.z, t.c.z)), hi = std::max(t.a.z, std::max(t.b.z, t.c.z));
          if (lo > hc.cfg.lidar_z + 0.01f || hi < hc.cfg.lidar_z - 0.01f) continue;
          const float x0 = std::min(t.a.x, std::min(t.b.x, t.c.x)), x1 = std::max(t.a.x, std::max(t.b.x, t.c.x));
          const float y0 = std::min(t.a.y, std::min(t.b.y, t.c.y)), y1 = std::max(t.a.y, std::max(t.b.y, t.c.y));
          for (int y = (int)std::floor((y0 - W.oy) / hc.cfg.res); y <= (int)std::floor((y1 - W.oy) / hc.cfg.res); ++y)
            for (int x = (int)std::floor((x0 - W.ox) / hc.cfg.res); x <= (int)std::floor((x1 - W.ox) / hc.cfg.res); ++x)
              if (x >= 0 && y >= 0 && x < W.gw && y < W.gh) g[(size_t)y * W.gw + x] = 1;   // 삼각형 상자 (넉넉히)
        }
        lid_geo.push_back(std::move(g));
      }
    struct Trace { int step; float x, y, th, v, w, cv, cw; int goal, p_n, p_i, since_occ, closed, why; };
    std::vector<Trace> trace;
    sim::set_cpu_threads(1);
    sim::Env cpu(hc, worlds, lidar, 1, seed, false);
    if (people > 0) cpu.set_people_max(std::vector<int>(1, people));
    cpu.reset();
    std::vector<sim::EnvState> st;
    cpu.get_state(st);
    limo_fire_explore::RobotEnv robot(hc);
    const int AW = mdp::act::DIM, K = mdp::hier::K;
    lxc::Params lp = lxc::Params::from_cfg(hc.cfg);
    lp.fuel = fo;
    lp.use_fuel = use_fuel;
    lp.use_cand = use_cand || use_dec;
    if (node_obs) robot.set_obs(!lp.use_cand);
    robot.set_frontier_cache(fc_on, fc_check);   // 실물 노드(explorer_node.cpp)와 같은 규칙 (obs_all false)
    lp.cand = co;
    lp.cand_commit = cand_commit;
    lxc::Ctl cs(lp), cr(lp);
    const double dt = (double)hc.cfg.substeps * hc.cfg.dt;
    lxd::Params dp;
    dp.period_steps = std::max(1, (int)std::lround(dec_period / dt));
    dp.mode = dec_mode;
    dp.hold_extra = dec_hold;
    dp.obj_always = dec_obj_always;
    dp.ask_lead_s = dec_lead;
    dp.fmt = dec_fmt; dp.stop_cand = dec_stop;
    dp.compact = dec_compact; dp.px = dec_px; dp.mem_fields = dec_memf; dp.max_cands = dec_max; dp.order = dec_order;
    dp.input = dec::input_of(dec_input);
    if (dec_match > 0.0) dp.match_r = (float)dec_match;
    dp.hint = dec_hint; dp.mem_image = dec_mimg; dp.image = dec_img; dp.log_k = dec_logk;
    if (use_dec && dp.input < 0) throw std::runtime_error("--dec-input text · image · both");
    lxd::Upper us(dp), ur(dp);
    std::vector<lxd::Params::Obj> base_objs, live_objs;   // --dec-obj (따라가기 시험은 결정마다 사람을 더해 다시 넣음) · 그중 못 간 것 뺀 지금 목록
    tsk::Goto gs_(tsp), gr_(tsp);
    tsk::Fails gfail_s(tsp);
    tsk::Instr ins_s(tsp);
    if (!dec_instr.empty()) ins_s.start(0.0);
    long g_stop = 0, g_start = 0, g_arrive = 0, g_nofail = 0, g_gone = 0, g_moved = 0, g_giveup_fail = 0, g_giveup_time = 0, g_instr_change = 0;
    if (!dec_instr.empty()) { us.set_instruction(dec_instr); ur.set_instruction(dec_instr); }
    us.set_class_hint(dec_hint_cls); ur.set_class_hint(dec_hint_cls);
    if (!dec_objs.empty()) {
      std::vector<lxd::Params::Obj> ob;
      for (const auto& t : dec_objs) {
        lxd::Params::Obj o;
        char cls[64];
        if (std::sscanf(t.c_str(), "%63[^,],%f,%f", cls, &o.x, &o.y) != 3) throw std::runtime_error("--dec-obj 클래스,x,y");
        o.cls = cls;
        o.oid = o.cls + "-" + std::to_string(ob.size() + 1);
        o.match = !dec_hint_cls.empty() && o.cls == dec_hint_cls ? 1.f : 0.f;
        ob.push_back(o);
      }
      us.set_objects(ob);
      ur.set_objects(ob);
      base_objs = ob;
    }
    FakeModel ms, mr;
    ms.lat = mr.lat = std::max(0, (int)std::lround(dec_lat / dt));
    ms.first = mr.first = dec_fake_a;
    ms.follow = mr.follow = dec_fake_follow;
    ms.obj = mr.obj = dec_fake_obj;
    ms.stopk = mr.stopk = dec_fake_stop;
    ms.fault = mr.fault = dec_fault;
    // 따라가기 (--follow): 두 쪽 따라가기 (같은 입력 → 같은 명령이어야), 사람별 마지막으로 본 때 · 자리, 셈
    ofl::Follower fs, fr;
    double ps_t[sim::kMaxPeople];
    float ps_x[sim::kMaxPeople], ps_y[sim::kMaxPeople];
    for (int q = 0; q < sim::kMaxPeople; ++q) ps_t[q] = -1e9;
    long f_start = 0, f_giveup = 0, f_diff = 0, f_steps = 0, f_mode[5] = {0, 0, 0, 0, 0}, f_hit = 0, f_reask = 0, f_hit_f = 0, f_near = 0, f_band = 0, f_close = 0;
    double f_dsum = 0, f_dmin = 1e9;
    if (const char* bl = std::getenv("DEC_BODY_LOG")) ms.blog = std::fopen(bl, "a");
    long l_fb0 = 0, l_rw = 0, l_sent0 = 0, l_pf0 = 0, l_pt0 = 0, l_ans0 = 0;
    std::vector<std::vector<float>> l_cov;   // 덮음 값들, 1 s 마다 (판 끝까지)
    covm::Coverage cov;
    int cov_map = -1;
    std::unique_ptr<limo_fire_explore::DeciderClient> real;
    std::vector<lxd::Answer> real_log;
    double real_ms = 0, real_max = 0;
    if (use_dec && !dec_url.empty()) {
      real = std::make_unique<limo_fire_explore::DeciderClient>(dec_url, 60000);
      ms.real = real.get();
      ms.log = mr.log = &real_log;
    }
    long d_body = 0, n_ans = 0, n_sent_all = 0, n_nomatch_all = 0, n_fb_all = 0;
    long fbw_all[10] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0}, fb_wait_all = 0, src_all[2] = {0, 0}, hold_all = 0;   // 판단 모델: 규칙이 대신 고른 까닭 · 답 오는 중 · 결정 출처
    fu::Diag dgs;
    cs.set_diag(&dgs);
    FILE* lf = logp.empty() ? nullptr : std::fopen(logp.c_str(), "w");
    if (lf) std::fprintf(lf, "ep\tt_s\tstep\tcmd_why\ttgt\tchanged\tjump_m\trobot_tgt_m\tn_clu\tn_node\tno_goal_why\tclosed\n");
    int prev_cell = -2;
    fu::Work ws, wr;
    Tap tap;
    tap.robot = &robot;
    std::vector<int16_t> lm;
    std::vector<uint32_t> tr;
    std::vector<uint8_t> vox, img, ldep;
    std::vector<float> pts;
    std::vector<sim::EnvState> st_prep;
    std::vector<int32_t> cand, ccur;
    std::vector<sim::LowRl> R;
    std::vector<sim::LowEp> L;
    sim::World rw;   // RobotEnv 지도 창 (gw · gh · ox · oy 만)
    // --cand: 후보 생성 두 쪽 (시뮬 Env 값 · RobotEnv 값) · 안정성 (3 결정 뒤 — 판단 모델 답 지연 2.6 s 쯤)
    cand::Mem cms, cmr;
    cand::Work cws, cwr;
    struct Snap { std::vector<int> id, vp; };
    std::vector<Snap> hist, hhist;   // 새 후보 · 환경 hier 후보 (최근 결정들)
    candq::Acc qacc;
    long g_chosen = 0, g_fail[9] = {0, 0, 0, 0, 0, 0, 0, 0, 0}, g_reach = 0, g_end[9] = {0, 0, 0, 0, 0, 0, 0, 0, 0}, g_refail = 0;
    int c_sig = 0, c_last_dec = 0;
    long c_pocket = 0, c_dec = 0, c_diff = 0, c_n = 0, c_rot = 0, c_none = 0, c_dorm = 0, c_near = 0, c_kept = 0;
    double c_ms = 0, c_ms_max = 0, c_part[6] = {0, 0, 0, 0, 0, 0};
    long s_n = 0, s_same = 0, h_n = 0, h_same = 0;   // 3 결정 뒤에도 남음 (새: 같은 번호 · 같은 칸 / hier: 1.5 m 안 후보 있음)
    double s_move = 0, h_move = 0;
    long s_alive = 0, h_alive = 0, s_m15 = 0;
    long ep_dec0 = 0, ep_sw0 = 0;
    double ep_path = 0, ep_px = 0, ep_py = 0;   // --ep-log: 이 판 지나온 길 (m, 참 자세)
    long ep_gsw = 0;                            // --ep-log: 이 판 경로 목표 칸(EnvState::goal_cell)이 다른 칸으로 바뀐 수 (위층 방식과 무관한 같은 잣대)
    int ep_goal = -1;   // --ep-log: 이 판 시작 때 셈
    long n_obs = 0, d_path = 0, d_cmd = 0, d_dec = 0, n_dec = 0, n_ep = 0, why_n[3] = {0, 0, 0}, n_sw = 0;
    int first_bad = -1, prev_tgt = -2;
    int st_ep_map = 0;   // 이 판의 장면 번호 (판 끝 대충 앎 넓이)
    const auto start = [&]() {
      cpu.get_state(st);
      const sim::EnvState& e = st[0];
      st_ep_map = e.map;
      const sim::World& w = worlds[(size_t)e.map];
      limo_fire_explore::Proprio p0;
      p0.wheel_v = e.drive.wv; p0.wheel_vy = e.drive.wvy; p0.wheel_w = e.drive.ww; p0.gyro_z = e.gyro;
      for (int i = 0; i < 7; ++i) p0.imu[i] = e.imu.out[i];
      const limo_fire_explore::RobotEnv::Window win{w.ox, w.oy, w.gw, w.gh};
      robot.reset(e.x0, e.y0, e.th0, p0, e.noise.fov, &win);
      rw.gw = w.gw; rw.gh = w.gh; rw.ox = w.ox; rw.oy = w.oy;
      if (use_cand) {   // 후보 모드 목표 끝 셈 (판마다 Ctl 이 새로 — 그 전에 더함)
        const cand::Mem& m = cr.cand_mem();
        for (const auto& h : m.hist) g_chosen += h.chosen;
        for (const auto& f : m.fails) g_fail[f.why < 9 && f.why >= 0 ? f.why : 0] += 1;
        g_reach += cr.n_reach();
        for (int q = 0; q < 9; ++q) g_end[q] += m.n_end[q];
        g_refail += cr.n_refail();
      }
      cs.restart(); cr.restart();
      cms = cand::Mem(); cmr = cand::Mem(); hist.clear(); hhist.clear();
      n_sent_all += us.n_sent(); n_nomatch_all += us.n_nomatch(); n_fb_all += us.n_fallback();
      for (int q = 0; q < 10; ++q) fbw_all[q] += us.n_fb_why(q);
      fb_wait_all += us.n_fb_wait(); src_all[0] += us.n_src(0); src_all[1] += us.n_src(1); hold_all += us.n_hold();
      us.restart(); ur.restart();
      fs.stop(); fr.stop();
      gs_.cancel(); gr_.cancel();
      for (int q = 0; q < sim::kMaxPeople; ++q) ps_t[q] = -1e9;
      prev_cell = -2;
      cs.set_geom(w.gw, hc.cfg.res); cr.set_geom(w.gw, hc.cfg.res);
      ++n_ep;
    };
    start();
    sim::set_cpu_tap(&tap);
    std::vector<float> act((size_t)AW, 0.f);
    std::vector<int32_t> dn;
    for (int t = 0; t < steps; ++t) {
      cpu.get_state(st);
      if (use_dec) {   // 판단 모델 위층 (걸음마다 — 보내기 · 받기 · 목표 잃으면 slot0 차례)
        ms.now = mr.now = t;
        cpu.get_map(lm); cpu.get_trail(tr); cpu.get_camlayer(vox, pts); cpu.get_cand(cand); cpu.get_cand_cur(ccur);
        const sim::World& w = worlds[(size_t)st[0].map];
        const lxc::View vs{&hc.cfg, &w, lm.data(), tr.data(), vox.empty() ? nullptr : vox.data(), &st[0], cand.data(), (uint32_t)ccur[0]};
        sim::LowCmd c1;
        int why = 0;
        const long a0 = us.n_ans();
        std::vector<ofl::Person> persons;   // 따라가기: 지금 (마지막으로 본) 사람들 — 실물 노드의 물체 지도 사람 항목 자리
        if (t == instr_at_step) {   // 노드 ~/instruction 콜백과 같은 차례: 따라가기·물체로 가기 끝 → 새 지시
          fs.stop(); fr.stop(); gs_.cancel(); gr_.cancel(); us.set_external(false); ur.set_external(false);
          us.set_instruction(instr_at_text); ur.set_instruction(instr_at_text);
          if (instr_at_text.empty()) ins_s.stop(); else ins_s.start(t * dt);
          ++g_instr_change;
        }
        if (goto_on && obj_move_step >= 0 && t == obj_move_step) {   // 시험: 물체를 옮김 (고른 뒤 지도에서 옮겨진 물체)
          for (auto& o : base_objs) { o.x += obj_move_dx; o.y += obj_move_dy; }
          ++g_moved;
        }
        if (goto_on && !follow_on && st[0].step % hc.cfg.decide_steps == 0) {   // 노드처럼 결정 차례마다 물체 후보 재료 (못 간 물체는 뺌)
          live_objs.clear();
          for (const auto& o : base_objs) if (!gfail_s.blocked(o.oid, o.cls, o.x, o.y, t * dt)) live_objs.push_back(o);
          us.set_objects(live_objs); ur.set_objects(live_objs);
        }
        if (follow_on) {
          const double tn = t * dt;
          const sim::EnvState& e = st[0];
          const float rx = e.x + e.ex, ry = e.y + e.ey, rth = e.th + e.eth;
          for (int q = 0; q < e.people.n; ++q) {
            const sim::Cyl& c = e.people.cyl[q];
            const float dx = c.x - rx, dy = c.y - ry, d = std::hypot(dx, dy);
            bool vis = d <= follow_range && std::fabs(std::remainder(std::atan2(dy, dx) - rth, 6.2831853f)) <= 0.5f * follow_fov;
            for (float u = 0.f; vis && u < d - c.r - 0.15f; u += 0.5f * hc.cfg.res) {   // 가림: 로봇 지도 막힌 칸 (사람 둘레는 빼고)
              const int cx = (int)std::floor((rx + dx / d * u - w.ox) / hc.cfg.res), cy = (int)std::floor((ry + dy / d * u - w.oy) / hc.cfg.res);
              if (cx >= 0 && cy >= 0 && cx < w.gw && cy < w.gh && lm[(size_t)cy * hc.cfg.size + cx] >= mdp::kOccL) vis = false;
            }
            if (vis) { ps_t[q] = tn; ps_x[q] = c.x; ps_y[q] = c.y; }
            if (ps_t[q] > -1e8) persons.push_back({"person-" + std::to_string(q + 1), ps_x[q], ps_y[q], (float)(tn - ps_t[q])});
          }
          if (st[0].step % hc.cfg.decide_steps == 0) {   // 노드처럼 결정 차례마다 물체 후보 재료: --dec-obj 물체 + 지금 보이는 사람(follow)
            std::vector<lxd::Params::Obj> ob = base_objs;
            for (const auto& pp : persons) {
              if (pp.seen_s > dp.vis_s) continue;
              lxd::Params::Obj o;
              o.oid = pp.oid; o.cls = "person"; o.x = pp.x; o.y = pp.y; o.seen_s = pp.seen_s; o.follow = true;
              ob.push_back(o);
            }
            us.set_objects(ob); ur.set_objects(ob);
          }
        }
        const bool ch_s = us.step(vs, cs, ms, &c1, &why);
        if (ch_s) {
          why_n[why] += 1;
          cpu.set_low_cmd(std::vector<sim::LowCmd>{c1});
          ++n_dec;
          if (c1.target != prev_tgt) { ++n_sw; prev_tgt = c1.target; }
        }
        n_ans += us.n_ans() - a0;
        if (const char* tp = std::getenv("DEC_TRACE")) {   // 진단: 1 s 마다 자세·목표·진행 없음 시계, 결정·요청마다 한 줄/파일
          static FILE* tf = std::fopen((std::string(tp) + ".tsv").c_str(), "w");
          static size_t n_req = 0, n_d = 0;
          const auto wx = [&](int cell) { return w.ox + (cell % w.gw + 0.5) * hc.cfg.res; };
          const auto wy = [&](int cell) { return w.oy + (cell / w.gw + 0.5) * hc.cfg.res; };
          while (n_req < ms.sent.size()) { char nm[64]; std::snprintf(nm, sizeof nm, "_req%03zu_t%d.json", n_req, t); write_text(std::string(tp) + nm, ms.sent[n_req]); ++n_req; }
          while (n_d < us.decisions().size()) {
            const auto& d = us.decisions()[n_d++];
            std::fprintf(tf, "D\t%d\t%d\t%.2f\t%.2f\t%d\t%d\t%s\n", d.step, d.id, d.x, d.y, d.kept, d.src, us.last_trigger().c_str());
          }
          if (t % 10 == 0) {
            const int tg = cs.cmd().target;
            std::fprintf(tf, "P\t%d\t%.2f\t%.2f\t%.2f\t%.2f\t%.2f\t%d\t%d\t%d\n", t, st[0].x, st[0].y, st[0].th, tg >= 0 ? wx(tg) : -99.0, tg >= 0 ? wy(tg) : -99.0, cs.cmd().follow, st[0].since_occ, st[0].collided);
          }
          std::fflush(tf);
        }
        const lxc::View vr{&robot.cfg(), &rw, robot.lmap(), robot.trail(), robot.cam_layer(), &robot.state(), robot.cand(), (uint32_t)robot.cand_cur(), robot.reach_keep()};
        sim::LowCmd c2;
        const bool ch_r = ur.step(vr, cr, mr, &c2);
        if (ch_r) robot.set_cmd(c2);
        if (follow_on || goto_on) {   // 실물 노드의 물체 답 가지와 같은 차례: 물체 답 꺼내기 → 프론티어 답이면 끝 → 따라가기 · 물체로 가기 걸음 → 아래층 명령
          const double tn = t * dt;
          lxd::Params::Obj pk_s, pk_r;
          const bool gs = us.take_object_pick(&pk_s), gr = ur.take_object_pick(&pk_r);
          if (gs != gr || (gs && (pk_s.oid != pk_r.oid || pk_s.follow != pk_r.follow))) ++f_diff;
          const bool ss = us.take_stop_pick(), sr = ur.take_stop_pick();
          if (ss != sr) ++f_diff;
          if (ss) {   // 노드: 따라가기 · 물체로 가기 끝 → 멈춤(꺼짐). 여기선 다음 지시까지 제자리 (목표 없음 · 판단 모델은 답만 기다림)
            fs.stop(); fr.stop(); gs_.cancel(); gr_.cancel(); ins_s.stop();
            us.set_external(true); ur.set_external(true);
            cpu.set_low_cmd(std::vector<sim::LowCmd>{sim::LowCmd{-1, 1}}); robot.set_cmd(sim::LowCmd{-1, 1});   // 도착 멈춤과 같은 "목표 없음 — 제자리"
            ++g_stop;
          }
          if (goto_on) {   // 물체로 가기 (task_sup.h — 노드 explorer_node.cpp 와 같은 차례)
            const sim::EnvState& e = st[0];
            const sim::EnvState& er = robot.state();
            if (gs_.active() && (ch_s || ch_r)) { gs_.cancel(); gr_.cancel(); us.set_external(false); ur.set_external(false); }
            if (gs && !pk_s.follow) {
              const lxd::Params::Obj* cur = nullptr;
              for (const auto& o : live_objs) if (o.oid == pk_s.oid) cur = &o;
              if (!cur || std::hypot(cur->x - pk_s.x, cur->y - pk_s.y) > 1.0) { us.ask("chosen object is gone or moved"); ur.ask("chosen object is gone or moved"); ++g_gone; }
              else {
                const auto cxy = [&](int vp, const sim::World& ww, float res, float* x, float* y) { *x = ww.ox + (vp % ww.gw + 0.5f) * res; *y = ww.oy + (vp / ww.gw + 0.5f) * res; };
                float sx, sy, rx2, ry2;
                cxy(pk_s.vp, w, hc.cfg.res, &sx, &sy); cxy(pk_r.vp, rw, robot.cfg().res, &rx2, &ry2);
                gs_.start(pk_s.oid, pk_s.oid, pk_s.cls, sx, sy, true, pk_s.x, pk_s.y, tn);
                gr_.start(pk_r.oid, pk_r.oid, pk_r.cls, rx2, ry2, true, pk_r.x, pk_r.y, tn);
                us.set_external(true); ur.set_external(true);
                ++g_start;
              }
            }
            if (gs_.active()) {
              const tsk::Ev es = gs_.step(tn, e.x + e.ex, e.y + e.ey, e.th + e.eth, e.goal_cell >= 0 && e.p_n <= 0, tsk::path_end(e, w.ox, w.oy, hc.cfg.res, hc.cfg.goal_tol));
              const tsk::Ev er2 = gr_.step(tn, er.x + er.ex, er.y + er.ey, er.th + er.eth, er.goal_cell >= 0 && er.p_n <= 0,
                                           tsk::path_end(er, rw.ox, rw.oy, robot.cfg().res, robot.cfg().goal_tol));
              if (es != er2) ++f_diff;
              if (std::getenv("GOTO_TRACE") && t % 5 == 0)
                std::printf("  걸음 %d 감 %s 설 자리 (%.2f, %.2f) 로봇 (%.2f, %.2f) 거리 %.2f 목표 칸 %d 경로 %d/%d v %.2f 사건 %d\n", t, gs_.oid().c_str(), gs_.gx(), gs_.gy(), e.x + e.ex, e.y + e.ey,
                            std::hypot(gs_.gx() - e.x - e.ex, gs_.gy() - e.y - e.ey), e.goal_cell, e.p_i, e.p_n, e.v, (int)es);
              if (es == tsk::kArrived) {   // 노드: 도착 → 멈춤 (goto_resume false). 여기선 지시를 지우고 탐사를 이어 감 (셈만)
                ++g_arrive;
                ins_s.stop();
                us.set_external(false); ur.set_external(false); us.set_instruction(""); ur.set_instruction("");
                cpu.set_low_cmd(std::vector<sim::LowCmd>{sim::LowCmd{-1, 1}}); robot.set_cmd(sim::LowCmd{-1, 1});
              } else if (es == tsk::kFailed) {
                ++g_nofail;
                gfail_s.add(gs_.oid(), gs_.cls(), gs_.fx(), gs_.fy(), tn);
                ins_s.failed();
                us.set_external(false); ur.set_external(false);
                us.ask("could not reach " + gs_.oid()); ur.ask("could not reach " + gs_.oid());
                cpu.set_low_cmd(std::vector<sim::LowCmd>{sim::LowCmd{-1, 1}}); robot.set_cmd(sim::LowCmd{-1, 1});
              } else {
                const auto cell_of = [](const sim::World& ww, float res) { return [&ww, res](float x, float y) {
                  const int cx = (int)std::floor((x - ww.ox) / res), cy = (int)std::floor((y - ww.oy) / res);
                  return cx < 0 || cy < 0 || cx >= ww.gw || cy >= ww.gh ? -1 : cy * ww.gw + cx; }; };
                const sim::LowCmd ks = gs_.cmd(cell_of(w, hc.cfg.res)), kr = gr_.cmd(cell_of(rw, robot.cfg().res));
                if (std::memcmp(&ks, &kr, sizeof ks) != 0) ++f_diff;
                cpu.set_low_cmd(std::vector<sim::LowCmd>{ks}); robot.set_cmd(kr);
              }
            }
            if (const char* g = ins_s.give_up(tn)) {   // 노드: 멈추고 "못 감 / 못 찾음". 여기선 지시를 지우고 셈
              ++(std::strcmp(g, "could not reach") == 0 ? g_giveup_fail : g_giveup_time);
              ins_s.stop();
              gs_.cancel(); gr_.cancel();
              us.set_external(false); ur.set_external(false); us.set_instruction(""); ur.set_instruction("");
            }
          }
          if (fs.active() && (ch_s || ch_r)) { fs.stop(); fr.stop(); us.set_external(false); ur.set_external(false); }   // 판단 모델이 프론티어를 고름
          if (gs && pk_s.follow) {
            fs.start(pk_s.oid, pk_s.x, pk_s.y, tn); fr.start(pk_r.oid, pk_r.x, pk_r.y, tn);
            us.set_external(true); ur.set_external(true);
            ++f_start;
          }
          if (fs.active()) {
            const sim::EnvState& e = st[0];
            const sim::EnvState& er = robot.state();
            const ofl::Params& fp = fs.params();
            const auto spot_s = [&](float x, float y, float px, float py, float* gx, float* gy) {
              return ofl::spot_on_map(lm.data(), hc.cfg.size, w.gw, w.gh, w.ox, w.oy, hc.cfg.res, x, y, px, py, fp.stop, fp.clear, fp.search, gx, gy); };
            const auto spot_r = [&](float x, float y, float px, float py, float* gx, float* gy) {
              return ofl::spot_on_map(robot.lmap(), robot.cfg().size, rw.gw, rw.gh, rw.ox, rw.oy, robot.cfg().res, x, y, px, py, fp.stop, fp.clear, fp.search, gx, gy); };
            const auto cell_s = [&](float x, float y) { const int cx = (int)std::floor((x - w.ox) / hc.cfg.res), cy = (int)std::floor((y - w.oy) / hc.cfg.res);
              return cx < 0 || cy < 0 || cx >= w.gw || cy >= w.gh ? -1 : cy * w.gw + cx; };
            const auto cell_r = [&](float x, float y) { const int cx = (int)std::floor((x - rw.ox) / robot.cfg().res), cy = (int)std::floor((y - rw.oy) / robot.cfg().res);
              return cx < 0 || cy < 0 || cx >= rw.gw || cy >= rw.gh ? -1 : cy * rw.gw + cx; };
            const ofl::Out os = fs.step(tn, e.x + e.ex, e.y + e.ey, persons, spot_s);
            const ofl::Out orr = fr.step(tn, er.x + er.ex, er.y + er.ey, persons, spot_r);
            const sim::LowCmd ks = ofl::cmd_of(os, fp, cell_s), kr = ofl::cmd_of(orr, fp, cell_r);
            if (std::memcmp(&ks, &kr, sizeof ks) != 0 || os.mode != orr.mode) ++f_diff;
            if (std::getenv("LOW_CHECK_VERBOSE")) std::printf("  걸음 %d 따라가기 %d 목표 %d/%d 얼굴 %d/%d ω %.2f 사람 (%.2f, %.2f) 거리 %.2f\n", t, os.mode, ks.target, kr.target, ks.face, kr.face, ks.w_idle, os.px, os.py, os.d);
            cpu.set_low_cmd(std::vector<sim::LowCmd>{ks});
            robot.set_cmd(kr);
            if (os.reask) { us.ask("followed person moved"); ur.ask("followed person moved"); ++f_reask; }
            if (os.mode == ofl::kGiveUp) {
              us.set_external(false); ur.set_external(false); us.ask("person lost"); ur.ask("person lost");
              cpu.set_low_cmd(std::vector<sim::LowCmd>{sim::LowCmd{-1, 1}}); robot.set_cmd(sim::LowCmd{-1, 1});   // 목표 없음 (제자리 회전) — 판단 모델 답을 기다림
              ++f_giveup;
            }
            ++f_steps; ++f_mode[os.searching ? 3 : os.mode];
            {   // 참 거리 (따르는 사람 원기둥 가운데 ~ 로봇 참 자리)
              const int q = std::atoi(fs.oid().c_str() + 7) - 1;
              if (q >= 0 && q < e.people.n) {
                const double d = std::hypot(e.people.cyl[q].x - e.x, e.people.cyl[q].y - e.y);
                f_dsum += d; f_dmin = std::min(f_dmin, d); ++f_near; f_band += d >= 0.6 && d <= 1.8; f_close += d < 0.6;
              }
            }
          }
          f_hit += st[0].people.hit_person;
          f_hit_f += st[0].people.hit_person && fs.active();
        }
        if (std::getenv("DEC_SAVE") && ms.sent.size() == (size_t)std::atoi(std::getenv("DEC_SAVE_N") ? std::getenv("DEC_SAVE_N") : "10") && !ms.sent.empty())
          write_text(std::getenv("DEC_SAVE"), ms.sent.back());   // 요청 하나를 파일로 (눈으로 보기)
        if (ms.sent.size() != mr.sent.size()) { ++d_body; if (first_bad < 0) first_bad = t; }
        else if (!ms.sent.empty() && ms.sent.back() != mr.sent.back()) {
          if (d_body == 0 && std::getenv("DEC_DUMP")) {
            const sim::EnvState& A = st[0]; const sim::EnvState& B = robot.state();
            std::fprintf(stderr, "kb sim %d %d %d %d rob %d %d %d %d | xy sim %.6f %.6f %.6f ex %.6f %.6f %.6f rob %.6f %.6f %.6f ex %.6f %.6f %.6f | win sim %d %d %.3f %.3f rob %d %d %.3f %.3f\n",
                         A.kb_x0, A.kb_y0, A.kb_x1, A.kb_y1, B.kb_x0, B.kb_y0, B.kb_x1, B.kb_y1, A.x, A.y, A.th, A.ex, A.ey, A.eth, B.x, B.y, B.th, B.ex, B.ey, B.eth,
                         worlds[(size_t)A.map].gw, worlds[(size_t)A.map].gh, worlds[(size_t)A.map].ox, worlds[(size_t)A.map].oy, rw.gw, rw.gh, rw.ox, rw.oy); write_text(std::string(std::getenv("DEC_DUMP")) + ".sim.json", ms.sent.back()); write_text(std::string(std::getenv("DEC_DUMP")) + ".robot.json", mr.sent.back()); }
          ++d_body; if (first_bad < 0) first_bad = t; }
      }
      // 위층 (시뮬 쪽: combine_run 과 같은 입력 — Env 에서 꺼낸 지도·지나간 길·깊이 층·후보)
      if (!use_dec && cs.due(st[0])) {
        cpu.get_map(lm); cpu.get_trail(tr); cpu.get_camlayer(vox, pts); cpu.get_cand(cand); cpu.get_cand_cur(ccur);
        const sim::World& w = worlds[(size_t)st[0].map];
        const lxc::View vs{&hc.cfg, &w, lm.data(), tr.data(), vox.empty() ? nullptr : vox.data(), &st[0], cand.data(), (uint32_t)ccur[0]};
        if (cand_on) {
          const cand::In ci{&hc.cfg, &w, lm.data(), tr.data(), vox.empty() ? nullptr : vox.data(), &st[0], [&](int cell) { return cs.avoided(cell, st[0].step); }, cs.cmd().target};
          const cand::In ri{&robot.cfg(), &rw, robot.lmap(), robot.trail(), robot.cam_layer(), &robot.state(), [&](int cell) { return cr.avoided(cell, robot.state().step); }, cr.cmd().target};
          const cand::Out a = cand::build(ci, co, cms, cws);
          const cand::Out b = cand::build(ri, co, cmr, cwr);
          ++c_dec;
          c_ms += b.ms; c_ms_max = std::max(c_ms_max, b.ms);
          for (int q = 0; q < 6; ++q) c_part[q] += b.ms_part[q];
          bool same = a.c.size() == b.c.size() && a.rotate == b.rotate && a.n_clu == b.n_clu && a.n_dormant == b.n_dormant && a.n_gaveup == b.n_gaveup;
          for (size_t i = 0; same && i < a.c.size(); ++i) same = std::memcmp(&a.c[i], &b.c[i], sizeof(cand::Cand)) == 0;
          if (!same) { ++c_diff; if (first_bad < 0) first_bad = t; }
          c_n += (long)b.c.size(); c_rot += b.rotate; c_none += b.c.empty(); c_dorm += b.n_dormant; c_near += b.n_near; c_pocket += b.n_pocket;
          for (const cand::Cand& x : b.c) c_kept += x.kept;
          c_last_dec = robot.state().step;
          candq::eval(ri, co, cwr, b, robot.cand(), &w, robot.state().map_closed != 0, qacc, &cmr);
          if (!cand_dump.empty() && (c_dec - 1) % cand_every == 0)
            candq::dump(cand_dump + "_ep" + std::to_string(n_ep) + "_s" + std::to_string(robot.state().step), ri, b, robot.cand());
          Snap sn, hs;
          for (const cand::Cand& x : b.c) { sn.id.push_back(x.id); sn.vp.push_back(x.vp); }
          for (int k = 0; k < K; ++k) if (robot.cand()[k] >= 0) { hs.id.push_back(k); hs.vp.push_back(robot.cand()[k]); }
          hist.push_back(sn); hhist.push_back(hs);
          const auto dist_m = [&](int a1, int b1) { return std::hypot(a1 % rw.gw - b1 % rw.gw, a1 / rw.gw - b1 / rw.gw) * hc.cfg.res; };
          if (hist.size() > 3) {
            const Snap& o = hist[hist.size() - 4];
            for (size_t i = 0; i < o.id.size(); ++i) {
              ++s_n;
              for (size_t j = 0; j < sn.id.size(); ++j)
                if (sn.id[j] == o.id[i]) { ++s_alive; s_move += dist_m(sn.vp[j], o.vp[i]); s_same += sn.vp[j] == o.vp[i]; }
              double bd = 1e9;   // hier 와 같은 잣대: 1.5 m 안 후보 있음
              for (size_t j = 0; j < sn.vp.size(); ++j) bd = std::min(bd, dist_m(sn.vp[j], o.vp[i]));
              s_m15 += bd <= 1.5;
            }
            const Snap& ho = hhist[hhist.size() - 4];
            for (size_t i = 0; i < ho.vp.size(); ++i) {
              ++h_n;
              double bd = 1e9;
              for (size_t j = 0; j < hs.vp.size(); ++j) bd = std::min(bd, dist_m(hs.vp[j], ho.vp[i]));
              if (bd <= 1.5) { ++h_alive; h_move += bd; h_same += bd == 0.0; }
            }
          }
        }
        int why = 0;
        dgs = fu::Diag();
        const sim::LowCmd c1 = cs.decide(vs, ws, &why);
        why_n[why] += 1;
        if (lf) {
          const double dt = (double)hc.cfg.substeps * hc.cfg.dt;
          const auto wx = [&](int cell) { return w.ox + (cell % w.gw + 0.5) * hc.cfg.res; };
          const auto wy = [&](int cell) { return w.oy + (cell / w.gw + 0.5) * hc.cfg.res; };
          const bool chg = prev_cell != -2 && c1.target != prev_cell;
          const double jump = chg && c1.target >= 0 && prev_cell >= 0 ? std::hypot(wx(c1.target) - wx(prev_cell), wy(c1.target) - wy(prev_cell)) : 0.0;
          const double rt = c1.target >= 0 ? std::hypot(wx(c1.target) - st[0].x, wy(c1.target) - st[0].y) : -1.0;
          std::fprintf(lf, "%ld\t%.1f\t%d\t%s\t%d\t%d\t%.2f\t%.2f\t%d\t%d\t%d\t%d\n", n_ep, st[0].step * dt, st[0].step, why == 0 ? "fuel" : why == 1 ? "slot0" : "none", c1.target, chg ? 1 : 0,
                       jump, rt, dgs.n_clu, dgs.n_node, why == 2 ? dgs.why : -1, (int)st[0].map_closed);
          prev_cell = c1.target;
        }
        cpu.set_low_cmd(std::vector<sim::LowCmd>{c1});
        ++n_dec;
        if (c1.target != prev_tgt) { ++n_sw; prev_tgt = c1.target; }
      }
      // 위층 (실물 쪽: RobotEnv 자기 값)
      if (!use_dec && cr.due(robot.state())) {
        const lxc::View vr{&robot.cfg(), &rw, robot.lmap(), robot.trail(), robot.cam_layer(), &robot.state(), robot.cand(), (uint32_t)robot.cand_cur(), robot.reach_keep()};
        const auto td0 = std::chrono::steady_clock::now();
        const sim::LowCmd c2 = cr.decide(vr, wr);
        n_reuse += cr.cands().reused;
        { const double d = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - td0).count(); t_dec += d; t_dec_max = std::max(t_dec_max, d); ++n_tdec; }
        robot.set_cmd(c2);
      }
      if (cs.cmd().target != cr.cmd().target || cs.cmd().follow != cr.cmd().follow || cs.slot() != cr.slot()) { ++d_dec; if (first_bad < 0) first_bad = t; }
      // 준비 단계 (시뮬 Env::low_prepare · 실물 low_prepare) — 목표 없음이면 행동 칸 −1 (combine_run.cpp:1144-1146)
      std::fill(act.begin(), act.end(), 0.f);
      const bool ext_goal = fs.active() || gs_.active();   // 따라가기 · 물체로 가기: 목표 칸은 밖이 줌 — 행동 칸 0 (low_follow 의 목표 칸 하나, 노드와 같음)
      act[0] = ext_goal ? 0.f : (float)cs.slot();   // combine_run 규칙: 결정 때 정한 행동 칸 (다음 결정까지 그대로)
      robot.set_slot(ext_goal ? 0 : cr.slot());
      cpu.low_prepare(act.data());
      cpu.get_depth(img);
      robot.set_depth(img.data());
      if (robot.layer_n() > 0) {   // 깊이 층 전용 격자: 시뮬이 준비 단계에서 그린 것을 실물 쪽에 그대로 (실물 노드는 영상에서 layer_from_image)
        cpu.get_layer_depth(ldep);
        robot.set_layer_depth(ldep.data());
      }
      { const auto t0 = std::chrono::steady_clock::now(); robot.low_prepare(); t_obs += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count(); }
      cpu.get_state(st_prep);
      ++n_obs;
      {   // 준비 뒤 경로 (목표 칸 · 경로 점 · 지금 구간 · 고른 후보) — 두 쪽 같아야
        const sim::EnvState& es = st_prep[0];
        const sim::EnvState& er = robot.state();
        const bool dp = es.goal_cell != er.goal_cell || es.p_n != er.p_n || es.p_i != er.p_i || es.p_slot != er.p_slot ||
                        std::memcmp(es.wp, er.wp, sizeof(es.wp[0]) * 2 * (size_t)std::max(0, es.p_n)) != 0;
        if (dp && std::getenv("LOW_CHECK_VERBOSE")) std::printf("  걸음 %d 경로 다름: 목표 %d/%d 점 %d/%d\n", t, es.goal_cell, er.goal_cell, es.p_n, er.p_n);
        d_path += dp;
        if (dp && first_bad < 0) first_bad = t;
      }
      if (std::getenv("LOW_CHECK_VERBOSE")) {
        cpu.get_state(st);
        if (std::memcmp(&st[0].ws, &robot.state().ws, 4) != 0 || std::memcmp(&st[0].vs, &robot.state().vs, 4) != 0)
          std::printf("  걸음 %d 속도 다듬기 상태 시뮬 (%g, %g) 실물 (%g, %g) · 시뮬 몸통 닿음 %d\n", t, st[0].vs, st[0].ws, robot.state().vs, robot.state().ws, st[0].collided);
      }
      float v, w;
      robot.low_act(&v, &w);
      std::vector<sim::EnvState> st_pre;
      if (!contact_log.empty()) cpu.get_state(st_pre);
      cpu.step(act.data());
      cpu.get_state(st);
      cpu.get_done(dn);
      if (const char* nn = std::getenv("LOW_CHECK_NNEW")) {   // 걸음마다 새로 안 칸 수(prog 링) · 새 막힘(occ 링) · 몸통 v ω — 진행 없음 문턱 재기
        static FILE* nf = std::fopen(nn, "w");
        const sim::EnvState& e = st[0];
        if (nf && e.step > 0) {   // 걸음 · 새로 안 칸 · 새 막힘 · 몸통 v ω · x y · 목표 칸 · 경로 점 i/n · 계획 단계(p_mode) · 검사 멈춤 누계(p_stops) · 명령 v ω
          const int k = (e.step - 1) % sim::kProgRing;
          std::fprintf(nf, "%d\t%d\t%d\t%.3f\t%.3f\t%.3f\t%.3f\t%d\t%d\t%d\t%d\t%d\t%.3f\t%.3f\n", e.step, e.prog[k], e.occ[k], e.v, e.w, e.x, e.y, e.goal_cell, e.p_i, e.p_n, e.p_mode, e.p_stops,
                       e.cmd_v, e.cmd_w);
        }
      }
      if (!contact_log.empty()) {
        const sim::EnvState& e = st_pre[0];
        trace.push_back(Trace{e.step, e.x, e.y, e.th, e.v, e.w, e.cmd_v, e.cmd_w, e.goal_cell, e.p_n, e.p_i, e.since_occ, (int)e.map_closed, cs.cmd().target});
        if (trace.size() > 650) trace.erase(trace.begin());
        if (dn[0] == 6) {   // 진행 없음: 마지막 65 s 를 1 s 마다 (자세 · 몸통 v · 위층 목표 · 경로 · 진척 시계 · 닫힘)
          if (FILE* cf = std::fopen(contact_log.c_str(), "a")) {
            const sim::World& W = worlds[(size_t)e.map];
            std::fprintf(cf, "판 %ld 진행 없음 %.1f s 장면 %s\n", n_ep, e.step * dt, W.name.c_str());
            float px = trace.front().x, py = trace.front().y, path = 0.f;
            for (size_t qi = 0; qi < trace.size(); ++qi) {
              const Trace& q = trace[qi];
              path += std::hypot(q.x - px, q.y - py); px = q.x; py = q.y;
              if (qi % 10) continue;
              const float gx = q.goal >= 0 ? W.ox + (q.goal % W.gw + 0.5f) * hc.cfg.res : NAN, gy = q.goal >= 0 ? W.oy + (q.goal / W.gw + 0.5f) * hc.cfg.res : NAN;
              std::fprintf(cf, "  %5.1f s (%.2f, %.2f) v %5.2f ω %5.2f 목표 %d (%.2f, %.2f) 거리 %.2f 경로 %d/%d 진척시계 %.1f s 닫힘 %d 지나온 %.1f m\n", q.step * dt, q.x, q.y, q.v, q.w, q.goal, gx, gy,
                           q.goal >= 0 ? std::hypot(gx - q.x, gy - q.y) : -1.f, q.p_i, q.p_n, q.since_occ * dt, q.closed, path);
            }
            std::fclose(cf);
          }
        }
        if (dn[0] == 2) {   // 닿음: 리셋 전 마지막 자세(st_pre)에서
          FILE* cf = std::fopen(contact_log.c_str(), "a");
          if (cf) {
            const sim::World& W = worlds[(size_t)e.map];
            const float res = hc.cfg.res, c0 = std::cos(e.th), s0 = std::sin(e.th);
            int nb = 0, n_lid = 0, n_map = 0, n_cam = 0, n_unk = 0;
            float dmin = 9.f;
            const int R = (int)std::ceil(0.45f / res);
            const int cx = (int)std::floor((e.x - W.ox) / res), cy = (int)std::floor((e.y - W.oy) / res);
            for (int dy = -R; dy <= R; ++dy)
              for (int dx = -R; dx <= R; ++dx) {
                const int x = cx + dx, y = cy + dy;
                if (x < 0 || y < 0 || x >= W.gw || y >= W.gh || !(W.grid[(size_t)y * W.gw + x] & sim::kCellBlocked)) continue;
                const float wx = W.ox + (x + 0.5f) * res - e.x, wy = W.oy + (y + 0.5f) * res - e.y;
                const float lx = c0 * wx + s0 * wy, ly = -s0 * wx + c0 * wy;
                const float gx = std::max(0.f, std::fabs(lx) - hc.cfg.half_len), gy = std::max(0.f, std::fabs(ly) - hc.cfg.half_wid);
                const float gap = std::hypot(gx, gy);
                if (gap > 0.08f) continue;   // 몸통에서 8 cm 안 막힌 칸만
                ++nb;
                dmin = std::min(dmin, gap);
                n_lid += lid_geo[(size_t)e.map][(size_t)y * W.gw + x];
                const int16_t l = robot.lmap()[(size_t)y * hc.cfg.size + x];
                n_map += l != mdp::kUnknownL && l >= mdp::kOccL;
                n_unk += l == mdp::kUnknownL;
                n_cam += robot.cam_layer() && robot.cam_layer()[(size_t)y * hc.cfg.size + x];
              }
            {   // 닿은 칸에 걸친 형상: 물체 이름 · 높이 범위 (몸통 띠 안 삼각형만)
              std::vector<std::pair<int, std::pair<float, float>>> objs;
              float zin_lo = 9.f, zin_hi = -9.f;
              for (size_t ti = 0; ti < W.scene.tris.size(); ++ti) {
                const sim::Tri& t = W.scene.tris[ti];
                const float lo = std::min(t.a.z, std::min(t.b.z, t.c.z)), hi = std::max(t.a.z, std::max(t.b.z, t.c.z));
                if (hi < hc.band_lo || lo > hc.band_hi) continue;
                const float mx = (t.a.x + t.b.x + t.c.x) / 3.f - e.x, my = (t.a.y + t.b.y + t.c.y) / 3.f - e.y;
                const float x0 = std::min(t.a.x, std::min(t.b.x, t.c.x)) - e.x, x1 = std::max(t.a.x, std::max(t.b.x, t.c.x)) - e.x;
                const float y0 = std::min(t.a.y, std::min(t.b.y, t.c.y)) - e.y, y1 = std::max(t.a.y, std::max(t.b.y, t.c.y)) - e.y;
                (void)mx; (void)my;
                if (x1 < -0.3f || x0 > 0.3f || y1 < -0.3f || y0 > 0.3f) continue;   // 몸통 둘레 0.3 m 상자
                {   // 몸통(지금 자세) 8 cm 안에 든 삼각형 높이만: 몸통 좌표로 세 꼭짓점 중 하나라도 몸통 + 8 cm 상자 안
                  bool in = false;
                  for (const sim::V3* q : {&t.a, &t.b, &t.c}) {
                    const float qx = q->x - e.x, qy = q->y - e.y, lx = c0 * qx + s0 * qy, ly = -s0 * qx + c0 * qy;
                    in = in || (std::fabs(lx) < hc.cfg.half_len + 0.08f && std::fabs(ly) < hc.cfg.half_wid + 0.08f);
                  }
                  if (in) { zin_lo = std::min(zin_lo, lo); zin_hi = std::max(zin_hi, hi); }
                }
                const int id = W.scene.obj.empty() ? -1 : (int)W.scene.obj[ti];
                bool f = false;
                for (auto& o : objs) if (o.first == id) { o.second.first = std::min(o.second.first, lo); o.second.second = std::max(o.second.second, hi); f = true; }
                if (!f) objs.push_back({id, {lo, hi}});
              }
              std::fprintf(cf, "  둘레 0.3 m 몸통 띠 형상:");
              for (const auto& o : objs)
                std::fprintf(cf, " [%s z %.2f~%.2f]", o.first >= 0 && o.first < (int)W.scene.obj_names.size() ? W.scene.obj_names[(size_t)o.first].c_str() : "?", o.second.first, o.second.second);
              std::fprintf(cf, " | 몸통 8 cm 안 꼭짓점이 있는 몸통 띠 삼각형 높이 %.2f~%.2f\n", zin_lo, zin_hi);
            }
            std::fprintf(cf, "판 %ld 닿음 %.1f s 장면 %s 자세 (%.2f, %.2f, %.0f°) 몸통 v %.2f ω %.2f | 몸통 8 cm 안 막힌 칸 %d (가장 가까움 %.2f m): 라이다 높이 형상 %d · 로봇 지도 막힘 %d · 모름 %d · 깊이 층 %d\n",
                         n_ep, e.step * dt, W.name.c_str(), e.x, e.y, e.th * 57.2958f, e.v, e.w, nb, dmin, n_lid, n_map, n_unk, n_cam);
            for (size_t qi = trace.size() > 20 ? trace.size() - 20 : 0; qi < trace.size(); ++qi) {
              const Trace& q = trace[qi];
              std::fprintf(cf, "  %5.1f s (%.2f, %.2f, %4.0f°) 몸통 v %5.2f ω %5.2f 명령 v %5.2f ω %5.2f 목표 %d 경로 %d/%d\n", q.step * dt, q.x, q.y, q.th * 57.2958f, q.v, q.w, q.cv, q.cw, q.goal, q.p_i, q.p_n);
            }
            std::fclose(cf);
          }
        }
      }
      int end_code = dn[0];
      if (!end_code && cs.rough_done(st[0].step)) {   // 후보 모드 "대충 다 앎" 끝 (Ctl::rough_done — 실물 노드와 같은 판정) — 두 쪽이 같은 걸음에 끝나야
        end_code = kRoughDone;
        if (!cr.rough_done(robot.state().step)) { ++d_cmd; if (first_bad < 0) first_bad = t; }
      } else if (!end_code && cr.rough_done(robot.state().step)) { ++d_cmd; if (first_bad < 0) first_bad = t; }
      if (end_code != 0) {   // 판 끝 (시뮬은 리셋됨) — 실물 쪽 판 끝 코드가 같은지
        if (dn[0] != 0 && robot.done() != dn[0]) {
          ++d_cmd;
          if (first_bad < 0) first_bad = t;
          if (std::getenv("LOW_CHECK_VERBOSE")) std::printf("  걸음 %d 판 끝 시뮬 %d 실물 %d\n", t, dn[0], robot.done());
        }
        if (!lat_log.empty()) {
          if (FILE* lf2 = std::fopen(lat_log.c_str(), "a")) {
            const long fb = (n_fb_all + us.n_fallback()) - l_fb0;
            const long sent = (n_sent_all + us.n_sent()) - l_sent0;
            std::fprintf(lf2, "%ld\t%d\t%.1f\t%ld\t%ld\t%ld\t%ld\t%ld\t%ld", n_ep, end_code, robot.state().step * dt, fb, l_rw, sent, n_ans - l_ans0, us.n_pick_failed() - l_pf0,
                         us.n_pick_top() - l_pt0);
            std::fprintf(lf2, "\t%d\t", worlds[(size_t)st[0].map].n_floor);
            for (int m = 0; m < cov.n_values(); ++m) {   // 지표마다 1 s 값들을 쉼표로, 지표 사이는 ;
              if (m) std::fprintf(lf2, ";");
              for (size_t q = 0; q < l_cov.size(); ++q) std::fprintf(lf2, "%s%.4f", q ? "," : "", l_cov[q][(size_t)m]);
            }
            std::fprintf(lf2, "\n");
            std::fclose(lf2);
          }
          l_cov.clear();
          l_rw = 0;
          l_fb0 = n_fb_all + us.n_fallback();
          l_sent0 = n_sent_all + us.n_sent(); l_ans0 = n_ans; l_pf0 = 0; l_pt0 = 0;   // us 는 판마다 새로 (restart) — 고름 셈도 0 부터
        }
        if (!ep_log.empty()) {
          if (FILE* ef = std::fopen(ep_log.c_str(), "a")) {
            std::fprintf(ef, "%ld\t%d\t%.1f\t%ld\t%ld\t%.4f\t%.2f\t%ld\n", n_ep, end_code, robot.state().step * (double)hc.cfg.substeps * hc.cfg.dt, n_dec - ep_dec0, n_sw - ep_sw0,
                         rough_cover(robot.lmap(), hc.cfg.size, worlds[(size_t)st_ep_map].grid.data(), rw.gw, rw.gh, hc.cfg.res), ep_path, ep_gsw);
            std::fclose(ef);
          }
          ep_dec0 = n_dec;
          ep_sw0 = n_sw;
          ep_path = 0;
          ep_gsw = 0;
          ep_goal = -1;
        }
        if (max_ep > 0 && n_ep >= max_ep) break;
        if (dn[0] == 0) cpu.reset();   // 대충 다 앎으로 끝낸 판: 시뮬도 새 판
        cpu.get_state(st);
        start();
        continue;
      }
      if ((std::memcmp(&st[0].cmd_v, &v, 4) != 0 || std::memcmp(&st[0].cmd_w, &w, 4) != 0)) {
        ++d_cmd;
        if (first_bad < 0) first_bad = t;
        if (std::getenv("LOW_CHECK_VERBOSE"))
          std::printf("  걸음 %d 명령 시뮬 (%g, %g) 실물 (%g, %g) · 위층 목표 %d follow %d · 경로 칸 시뮬 %d 실물 %d · 목표 칸 시뮬 %d 실물 %d\n", t, st[0].cmd_v, st[0].cmd_w, v, w, cs.cmd().target,
                      cs.cmd().follow, (int)st[0].p_n, (int)robot.state().p_n, st[0].goal_cell, robot.state().goal_cell);
        if (std::getenv("LOW_CHECK_VERBOSE")) {   // 두 쪽 지도·깊이 층·지나간 칸이 같은지 (어디서 갈라졌나)
          cpu.get_map(lm); cpu.get_trail(tr); cpu.get_camlayer(vox, pts);
          long dm = 0, dc = 0, dt = 0;
          for (size_t i = 0; i < lm.size(); ++i) dm += lm[i] != robot.lmap()[i];
          for (size_t i = 0; i < vox.size(); ++i) dc += vox[i] != robot.cam_layer()[i];
          for (size_t i = 0; i < tr.size(); ++i) dt += tr[i] != robot.trail()[i];
          std::printf("    지도 다른 칸 %ld · 깊이 층 다른 칸 %ld · 지나간 칸 다른 낱말 %ld · 닫힘 시뮬 %d 실물 %d · 자세 시뮬 (%.4f %.4f %.4f) 실물 (%.4f %.4f %.4f)\n", dm, dc, dt, st[0].map_closed,
                      robot.state().map_closed, st[0].x, st[0].y, st[0].th, robot.state().x, robot.state().y, robot.state().th);
        }
      }
      cpu.get_low_rl(R); cpu.get_low(L);
      if (st[0].step > 0) ep_path += std::hypot(st[0].x - ep_px, st[0].y - ep_py);
      ep_px = st[0].x; ep_py = st[0].y;
      if (st[0].goal_cell >= 0 && st[0].goal_cell != ep_goal) { ep_gsw += ep_goal >= 0; ep_goal = st[0].goal_cell; }
      cs.after_step(st[0], R[0], L[0]);
      if (!lat_log.empty()) {
        if (use_dec && L[0].reach_step == st[0].step && L[0].reached > 0 && us.waiting()) ++l_rw;
        if (st[0].step % 10 == 0) {   // 덮음 (1 s 마다, scripts/coverage.h — 가는 칸 · 굵은 칸 0.5 / 1.0 m × 문턱 10 · 20 · 30 · 50 %)
          if (!use_dec) cpu.get_map(lm);
          if (cov_map != st[0].map) { cov.init(worlds[(size_t)st[0].map]); cov_map = st[0].map; }
          l_cov.push_back(cov.eval(lm.data(), hc.cfg.size));
        }
      }
      if (cand_on) {   // 후보 기억: 자취 · 실패 자리 (Ctl::after_step 과 같은 판정 — 막힘 신호 · 경로 끝 밖 · 경로 없음)
        const sim::EnvState& e = robot.state();
        const sim::LowRl& rl = robot.low_rl();
        const sim::LowEp& le = robot.low_ep();
        const sim::LowCmd cm = cr.cmd();
        const auto fail = [&](int cell, int why) {
          if (cell < 0) return;
          const float fx = rw.ox + (cell % rw.gw + 0.5f) * hc.cfg.res, fy = rw.oy + (cell / rw.gw + 0.5f) * hc.cfg.res;
          cms.note_fail_at(fx, fy, why, e.step); cmr.note_fail_at(fx, fy, why, e.step);
        };
        if (rl.stall && !c_sig) fail(rl.stall_tgt, cand::kStall);
        c_sig = rl.stall;
        if (cm.follow != 0 && cm.target >= 0) {
          const bool reached = le.reach_step == e.step && le.reached > 0;
          const bool off = !reached && le.hold != 0;
          const bool nopath = !reached && !off && e.step > c_last_dec && (e.goal_cell != cm.target || e.p_n <= 0);
          if (off) fail(cm.target, cand::kOffPath);
          else if (nopath) fail(cm.target, cand::kNoPath);
        }
        cms.note_pose(e.step, e.x + e.ex, e.y + e.ey, e.th + e.eth); cmr.note_pose(e.step, e.x + e.ex, e.y + e.ey, e.th + e.eth);
      }
      cr.after_step(robot.state(), robot.low_rl(), robot.low_ep());
    }
    sim::set_cpu_tap(nullptr);
    if (lf) std::fclose(lf);
    for (const auto& x : real_log) { real_ms += x.ms; real_max = std::max(real_max, x.ms); }
    if (real) std::printf("진짜 판단 모델 %s: 답 %zu 개 · 평균 %.0f ms · 최대 %.0f ms (시뮬 시계에는 --dec-latency %.1f s 로)\n", dec_url.c_str(), real_log.size(),
                          real_log.empty() ? 0.0 : real_ms / real_log.size(), real_max, dec_lat);
    std::printf("위층 %s · 아래층 %s · ", use_dec ? "decider (가짜 응답기)" : use_cand ? "cand (후보 0 번 · 붙듦)" : use_fuel ? "fuel" : "slot0", "규칙 추종기");
    if (use_dec)
      std::printf("판단 모델: 주기 %d 걸음 · 지연 %d 걸음 · 보냄 %ld · 답 적용 %ld · 후보 안 맞음 %ld · slot0 대신 %ld · 본문 다름 %ld | ", dp.period_steps, ms.lat, n_sent_all + us.n_sent(), n_ans,
                  n_nomatch_all + us.n_nomatch(), n_fb_all + us.n_fallback(), d_body);
    if (dec_stop) std::fprintf(stderr, "STOP 멈춤 고름 %ld\n", g_stop);
    if (goto_on)
      std::fprintf(stderr, "GOTO 감 %ld · 닿음 %ld · 못 감 %ld · 고른 물체 사라짐·옮김 %ld (옮김 주입 %ld) · 포기(못 감 %ld · 시간 %ld) · 규칙이 지시 물체로 %ld · 잇따른 실패 끝 %d · 지시 바뀜 %ld · 새 지시 물음 %ld · 두 쪽 다름 %ld\n",
                   g_start, g_arrive, g_nofail, g_gone, g_moved, g_giveup_fail, g_giveup_time, us.n_fb_obj(), us.streak(), g_instr_change,
                   us.trigger_counts().count("new instruction") ? us.trigger_counts().at("new instruction") : 0L, f_diff);
    if (follow_on)
      std::fprintf(stderr, "FOLLOW 시작 %ld · 걸음 %ld (몲 %ld 섬 %ld 찾음 %ld) · 포기 %ld · 다시 물음 %ld · 참 거리 평균 %.2f 최소 %.2f m · 0.6~1.8 m 몫 %.2f · 0.6 m 안 %ld 걸음 · 사람 닿음 %ld (따라가는 중 %ld) · 두 쪽 다름 %ld\n",
                   f_start, f_steps, f_mode[1], f_mode[2], f_mode[3], f_giveup, f_reask, f_near ? f_dsum / f_near : 0.0, f_dmin < 1e8 ? f_dmin : -1.0, f_near ? (double)f_band / f_near : 0.0, f_close, f_hit, f_hit_f, f_diff);
    if (use_dec) {
      for (int q = 0; q < 10; ++q) fbw_all[q] += us.n_fb_why(q);
      fb_wait_all += us.n_fb_wait(); src_all[0] += us.n_src(0); src_all[1] += us.n_src(1);
      std::fprintf(stderr, "DECSRC 답 %ld 규칙 %ld · 기다림 %ld 걸음 · 대신 까닭: 닿음 %ld 막힘 %ld 다가가지못함 %ld 경로끝밖 %ld 경로없음 %ld 그밖 %ld · 답 오는 중 %ld\n", src_all[0], src_all[1], hold_all + us.n_hold(),
                   fbw_all[cand::kReached], fbw_all[cand::kStall], fbw_all[cand::kTimeout], fbw_all[cand::kOffPath], fbw_all[cand::kNoPath],
                   fbw_all[9] + fbw_all[cand::kNone] + fbw_all[cand::kContact] + fbw_all[cand::kNoGain] + fbw_all[cand::kSwitched], fb_wait_all);
    }
    std::printf(use_dec ? "관측 %ld 번 · 판 %ld · 위층 결정 %ld (판단 모델 후보 %ld · slot0 차례 %ld · 목표 없음 %ld) · 목표 바뀜 %ld · 통과 %ld · 막힘 신호 %ld · 다가가지 못함 %ld\n" : "관측 %ld 번 · 판 %ld · 위층 결정 %ld (fuel %ld · slot0 대신 %ld · 목표 없음 %ld) · 목표 바뀜 %ld · 통과 %ld · 막힘 신호 %ld · 다가가지 못함 %ld\n", n_obs, n_ep, n_dec, why_n[0], why_n[1],
                why_n[2], n_sw, cs.n_reach(), cs.n_stall(), cs.n_far());
    if (cand_on && c_dec) {
      std::printf("후보 생성 (scripts/cand_view.h): 결정 %ld · 두 쪽 다름 %ld · 평균 후보 %.2f 개 · 후보 0 %ld (제자리 회전 %ld) · 쉬는 덩어리 평균 %.2f (로봇 곁 %.2f · 구멍 %.2f) · 지난 칸 그대로 %.2f · 시간 평균 %.2f ms 최대 %.2f ms\n",
                  c_dec, c_diff, (double)c_n / c_dec, c_none, c_rot, (double)c_dorm / c_dec, (double)c_near / c_dec, (double)c_pocket / c_dec, c_n ? (double)c_kept / c_n : 0.0, c_ms / c_dec, c_ms_max);
      std::printf("  나눠 본 시간 (평균 ms): 찾기 %.2f · 거리 지도·방 %.2f · 덩어리 %.2f · 로봇 거리 %.2f · 시점 %.2f · 고르기 %.2f\n", c_part[0] / c_dec, c_part[1] / c_dec, c_part[2] / c_dec,
                  c_part[3] / c_dec, c_part[4] / c_dec, c_part[5] / c_dec);
      candq::print(qacc, "시뮬 RobotEnv 쪽");
      std::printf("3 결정 뒤 1.5 m 안 후보 있음 (판단 모델 답 맞춤 반경): 새 %.3f\n", s_n ? (double)s_m15 / s_n : 0.0);
      std::printf("3 결정 뒤 (판단 모델 답 지연쯤): 새 후보 — 같은 번호 남음 %.3f · 그중 같은 칸 %.3f · 옮긴 거리 평균 %.2f m | 환경 hier 후보 — 1.5 m 안 후보 있음 %.3f · 같은 칸 %.3f · 거리 평균 %.2f m\n",
                  s_n ? (double)s_alive / s_n : 0.0, s_alive ? (double)s_same / s_alive : 0.0, s_alive ? s_move / s_alive : 0.0, h_n ? (double)h_alive / h_n : 0.0,
                  h_alive ? (double)h_same / h_alive : 0.0, h_alive ? h_move / h_alive : 0.0);
    }
    if (use_cand) {
      const cand::Mem& m = cr.cand_mem();   // 마지막 판 몫
      for (const auto& h : m.hist) g_chosen += h.chosen;
      for (const auto& f : m.fails) g_fail[f.why < 9 && f.why >= 0 ? f.why : 0] += 1;
      g_reach += cr.n_reach();
      for (int q = 0; q < 9; ++q) g_end[q] += m.n_end[q];
      g_refail += cr.n_refail();
      std::printf("후보 모드 목표: 보냄 %ld · 통과 %ld · 실패 — 경로 없음 %ld · 막힘·정체 %ld · 경로 끝 밖 %ld\n", g_chosen, g_reach, g_fail[cand::kNoPath], g_fail[cand::kStall], g_fail[cand::kOffPath]);
      std::printf("후보 모드 목표 끝 까닭: 닿음 %ld · 다른 후보로 바꿈 %ld · 경로 없음 %ld · 막힘 %ld · 경로 끝 밖 %ld\n", g_end[cand::kReached], g_end[cand::kSwitched], g_end[cand::kNoPath],
                  g_end[cand::kStall], g_end[cand::kOffPath]);
      std::printf("최근 실패 후보로 다시 감 %ld 번%s\n", g_refail, use_cand && g_refail > 0 ? " — 규칙(upper cand)은 0 이어야: 틀림" : "");
    }
    std::printf("실물 쪽 시간 (이 기기 한 코어): step 평균 %.2f ms · low_prepare %.2f ms · 위층 결정 평균 %.2f ms (최대 %.2f, %ld 번 · 후보 찾기 재사용 %ld · 칸 기억 전체 %ld 점검 %ld 다름 %ld)\n", n_step ? t_step / n_step : 0.0,
                n_obs ? t_obs / n_obs : 0.0, n_tdec ? t_dec / n_tdec : 0.0, t_dec_max, n_tdec, n_reuse, robot.frontier_cache().n_full, robot.frontier_cache().n_check,
                robot.frontier_cache().n_bad);
    std::printf("RobotEnv vs CPU 시뮬: 위층 명령 다름 %ld · 경로 다름 %ld · 아래층 명령·판 끝 다름 %ld\n", d_dec, d_path, d_cmd);
    const bool ok = d_dec == 0 && d_path == 0 && d_cmd == 0 && d_body == 0 && f_diff == 0 && !(use_cand && g_refail > 0);   // 후보 규칙이 최근 실패 후보를 다시 고르면 틀림
    std::printf("%s%s\n", ok ? "통과: 실물 합친 탐사 노드 == 시뮬 (비트)" : "다름", first_bad >= 0 ? (" — 처음 다른 걸음 " + std::to_string(first_bad)).c_str() : "");
    return ok ? 0 : 1;
  } catch (const std::exception& ex) {
    std::fprintf(stderr, "low_explore_check: %s\n", ex.what());
    return 1;
  }
}
