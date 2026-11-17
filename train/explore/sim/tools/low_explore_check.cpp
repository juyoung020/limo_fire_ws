// 실물 합친 탐사 노드(위층 + 아래층 규칙 추종기 — limo_fire_rl low_rl) == CPU 시뮬 합친 탐사 (combine_run --upper fuel|slot0|decider 와 같은 설정).
// 시뮬(환경 하나, CPU 참조판)과 RobotEnv(합친 탐사 모드)가 각자 자기 상태로 위층 결정(scripts/low_explore_ctl.h — combine_run 의 환경 하나 몫)을 하고,
// 걸음마다 시뮬 라이다 장·감각·깊이를 RobotEnv 에 넣어 두 쪽의 위층 명령(목표 칸·follow)·준비 뒤 경로(목표 칸·경로 점)·명령(v, ω)·판 끝을 비트 비교한다.
// 아래층은 두 쪽 다 규칙 추종기 (준비 단계 행동 칸 = 위층 Ctl::slot() — combine_run 의 act[0], RobotEnv::low_act 계층형 가지). 관측 설정은 env.yaml 그대로.
//   low_explore_check <장면.scene> … [--steps 1500] [--seed 1] [--people 0] [--upper fuel|slot0|decider] [--fuel-v 0.981 --fuel-w 3.232 --fuel-t0 0.10] [--fuel-time 1 --fuel-global 1 --fuel-vp 1]
//   --upper slot0: combine_run --upper slot0 (피하지 않는 첫 후보 · stick — 실물 노드 upper:=slot0)
//   fuel 기본 = 실물 노드 config/low_rl.yaml (시간 + 전역 + 시점, 붙들기 0.2, 시점 자격 0.75 m · 0.21 m). --fuel-hyst · --fuel-vp-min · --fuel-vp-clear · --fuel-first 로 바꿈
//   --contact-log 파일: 판이 닿음(2)으로 끝나면 몸통 8 cm 안 막힌 칸(정답 격자)이 라이다 높이 형상인지 · 로봇 지도·깊이 층이 알았는지 · 그 형상 물체·높이 · 마지막 2 s 자취,
//     진행 없음(6)으로 끝나면 마지막 65 s 를 1 s 마다 (자세·목표·경로·시계·닫힘·지나온 거리). LOW_CHECK_NNEW=파일: 걸음마다 새로 안 칸·새 막힘·속도·경로·계획 단계·검사 멈춤
//   --log 파일: 시뮬 쪽 결정마다 한 줄 (판·시각·목표·바뀜·뜀·목표 없음 까닭·닫힘) — 실물 녹화 low_explore_replay --log 와 같은 꼴로 견줌
//   --upper decider [--dec-period 3.0] [--dec-latency 2.6] [--dec-input both] [--dec-cand 1] [--dec-hint 1] [--dec-mem-image 1] [--dec-image A|B] [--dec-log-k 8]: 실물 노드 upper:=decider (scripts/low_decider.h) — 판단 모델 대신 정해진 가짜 응답기
//     (요청 본문의 FNV 해시로 글자를 고르고 --dec-latency 초 뒤 걸음에 답을 냄). 두 쪽 요청 본문도 비트 비교. 시뮬이 판단 모델 답을 몇 걸음 뒤에 받는 것과 같은 길
//     --dec-url http://127.0.0.1:8171: 가짜 대신 진짜 llama-server (실물 노드와 같은 비차단 소켓 limo_fire_rl/decider_client.h). 시뮬 시계는 답을 기다리는 동안 멈추고,
//     답은 --dec-latency 초 뒤 걸음에 줌 (실물 답 시간을 시뮬 걸음으로). 실물 쪽은 같은 답을 같은 차례로 받음 — 판단 모델 위층을 시뮬에서 실물과 같은 코드로 평가
// 지도 자세 오차·관측 지연은 끈다 (실물 노드는 지도 자세를 그대로 받으므로). 판이 끝나면 둘 다 새 판 (위층 기억도 비움).
#include <chrono>
#include <cmath>
#include <memory>
#include <thread>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "env_api.h"
#include "limo_fire_rl/decider_client.h"
#include "limo_fire_rl/robot_env.h"
#include "coverage.h"
#include "low_decider.h"
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
  struct P { int ready; lxd::Answer a; std::string body; };
  std::vector<P> q;
  std::vector<std::string> sent;   // 보낸 본문 (두 쪽 비교)
  limo_fire_rl::DeciderClient* real = nullptr;   // 진짜 서버 (시뮬 쪽만 — 답을 기다림)
  std::vector<lxd::Answer>* log = nullptr;       // 진짜 답 기록 (시뮬 쪽이 씀, 실물 쪽이 같은 차례로 읽음)
  size_t replay = 0;
  void send(const lxd::Req& r) override {
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
    a.ok = true;
    a.choice = std::string(1, (char)('A' + (first ? 0 : (int)(h % (uint64_t)(r.cells.size())))));
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
  limo_fire_rl::RobotEnv* robot = nullptr;
  limo_fire_rl::Scan sc;
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
    limo_fire_rl::Proprio p;
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
  bool use_fuel = true, use_dec = false, use_cand = false, dec_cand = true, dec_hint = true, dec_mimg = true, node_obs = false, fc_on = true;
  int fc_check = 0;
  int dec_img = 0, dec_logk = lxd::Params{}.log_k;   // 기본은 노드와 같은 lxd::Params
  double dec_period = 3.0, dec_lat = 2.6, dec_match = -1.0;
  int dec_mode = 1, dec_max = 6, dec_order = 0;   // 실물 노드 기본과 같게 (decider_mode event · decider_max_cands 6)
  bool dec_compact = lxd::Params{}.compact, dec_memf = true;
  int dec_px = lxd::Params{}.px;
  std::string dec_input = "both", dec_url;
  std::string ep_log, lat_log, contact_log;   // --lat-log: 판 끝마다 (판 · 끝 코드 · 끝 시각 · 규칙 대신 · 답 기다리는 중 닿음 · 보냄 · 답 적용 · 실패 기록 후보 고름 · 1 순위 고름 · 정답 바닥 칸 수 · 덮음 지표 9 개(가는 칸, 굵은 0.5 m × 문턱 0.1/0.2/0.3/0.5, 굵은 1.0 m × 같은 문턱 — scripts/coverage.h)를 1 s 마다: 지표 사이 ; 값 사이 ,)
  bool dec_fake_a = false;   // --ep-log: 판 끝마다 한 줄 (판 · 끝 코드 · 시각 s · 위층 결정 · 목표 바뀜) — --max-ep N 판에서 멈춤
  std::string dec_instr;   // --dec-instr "글": 판단 모델 물체 찾기 지시 (실물 노드 ~/instruction) — 요청 본문 비트 비교 확인용
  std::vector<std::string> dec_objs;   // --dec-obj "클래스,x,y" (여러 번): 지시가 있을 때 물체 후보 (실물 노드는 물체 지도에서) — 요청 비트 비교 확인용
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
  fo.hyst = 0.2f;      // 실물 노드 기본 (limo_fire_rl config/low_rl.yaml fuel_hyst · fuel_vp_min · fuel_vp_clear)
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
    else if (a == "--cand-vp-min") co.vp_min = std::stof(nx());
    else if (a == "--cand-margin") co.margin = std::stof(nx());
    else if (a == "--cand-coarse") co.coarse_m = std::stof(nx());
    else if (a == "--cand-commit") cand_commit = std::stoi(nx());
    else if (a == "--cand-keep") co.keep = std::stof(nx());
    else if (a == "--cand-known") co.known_frac = std::stof(nx());
    // 위층 결정 때마다 후보 생성(scripts/cand_view.h)을 두 쪽에서 — 비트 비교 · 시간 · 안정성 (주행은 그대로)
    else if (a == "--close-lidar") close_lidar = true;   // 닫힘을 라이다 지도만으로 (EnvCfg::low_explore 2 — 실물 노드 close_lidar)
    else if (a == "--node-obs") node_obs = true;
    else if (a == "--fc-check") fc_check = std::stoi(nx());   // 실물 쪽 걸음 사이 프론티어 칸 판정 기억을 N 걸음마다 처음부터 다시 센 값과 견줌 (frontier_cache.h)
    else if (a == "--no-fc") fc_on = false;                   // 실물 쪽 기억 끔 (걸음마다 처음부터)   // 실물 쪽 RobotEnv 를 노드처럼 쓰는 관측만 (set_obs — 그림 끔, cand_view 후보 위층이면 계층형 후보도 끔) — 그래도 시뮬과 같은지
    else if (a == "--dec-cand") dec_cand = std::stoi(nx()) != 0;   // 판단 모델 후보: 1 = cand_view.h (기본) · 0 = hier 8 칸
    else if (a == "--dec-hint") dec_hint = std::stoi(nx()) != 0;
    else if (a == "--dec-mem-image") dec_mimg = std::stoi(nx()) != 0;
    else if (a == "--dec-image") dec_img = nx() == "B" ? 1 : 0;
    else if (a == "--dec-log-k") dec_logk = std::stoi(nx());
    else if (a == "--upper") { const std::string u = nx(); if (u != "fuel" && u != "slot0" && u != "decider" && u != "cand") throw std::runtime_error("--upper 는 fuel · slot0 · decider · cand"); use_fuel = u == "fuel"; use_dec = u == "decider"; use_cand = u == "cand"; }
    else if (a == "--dec-period") dec_period = std::stod(nx());
    else if (a == "--dec-mode") dec_mode = nx() == "event" ? 1 : 0;
    else if (a == "--dec-compact") dec_compact = std::stoi(nx()) != 0;
    else if (a == "--dec-px") dec_px = std::stoi(nx());
    else if (a == "--dec-memf") dec_memf = std::stoi(nx()) != 0;
    else if (a == "--dec-max") dec_max = std::stoi(nx());
    else if (a == "--dec-order") dec_order = nx() == "dist" ? 1 : 0;
    else if (a == "--dec-latency") dec_lat = std::stod(nx());
    else if (a == "--dec-input") dec_input = nx();
    else if (a == "--dec-match") dec_match = std::stod(nx());
    else if (a == "--dec-url") dec_url = nx();
    else if (a == "--ep-log") ep_log = nx();
    else if (a == "--contact-log") contact_log = nx();
    else if (a == "--lat-log") lat_log = nx();
    else if (a == "--dec-fake") dec_fake_a = nx() == "A";
    else if (a == "--dec-instr") dec_instr = nx();
    else if (a == "--dec-obj") dec_objs.push_back(nx());
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
    const auto no_delay = [&]() { cpu.get_state(st); st[0].delay = 0; cpu.set_state(0, st[0]); };
    no_delay();
    limo_fire_rl::RobotEnv robot(hc);
    const int AW = mdp::act::DIM, K = mdp::hier::K;
    lxc::Params lp = lxc::Params::from_cfg(hc.cfg);
    lp.fuel = fo;
    lp.use_fuel = use_fuel;
    lp.use_cand = use_cand || (use_dec && dec_cand);
    if (node_obs) robot.set_obs(false, !lp.use_cand);
    robot.set_frontier_cache(fc_on, fc_check);   // low_rl_node 와 같은 규칙 (obs_all false)
    lp.cand = co;
    lp.cand_commit = cand_commit;
    lxc::Ctl cs(lp), cr(lp);
    const double dt = (double)hc.cfg.substeps * hc.cfg.dt;
    lxd::Params dp;
    dp.period_steps = std::max(1, (int)std::lround(dec_period / dt));
    dp.mode = dec_mode;
    dp.compact = dec_compact; dp.px = dec_px; dp.mem_fields = dec_memf; dp.max_cands = dec_max; dp.order = dec_order;
    dp.input = dec::input_of(dec_input);
    if (dec_match > 0.0) dp.match_r = (float)dec_match;
    dp.cand = dec_cand; dp.hint = dec_hint; dp.mem_image = dec_mimg; dp.image = dec_img; dp.log_k = dec_logk;
    if (use_dec && (dp.input < 0 || dp.input == dec::kAll)) throw std::runtime_error("--dec-input text · image · both");
    lxd::Upper us(dp), ur(dp);
    if (!dec_instr.empty()) { us.set_instruction(dec_instr); ur.set_instruction(dec_instr); }
    if (!dec_objs.empty()) {
      std::vector<lxd::Params::Obj> ob;
      for (const auto& t : dec_objs) {
        lxd::Params::Obj o;
        char cls[64];
        if (std::sscanf(t.c_str(), "%63[^,],%f,%f", cls, &o.x, &o.y) != 3) throw std::runtime_error("--dec-obj 클래스,x,y");
        o.cls = cls;
        o.oid = o.cls + "-" + std::to_string(ob.size() + 1);
        ob.push_back(o);
      }
      us.set_objects(ob);
      ur.set_objects(ob);
    }
    FakeModel ms, mr;
    ms.lat = mr.lat = std::max(0, (int)std::lround(dec_lat / dt));
    ms.first = mr.first = dec_fake_a;
    long l_fb0 = 0, l_rw = 0, l_sent0 = 0, l_pf0 = 0, l_pt0 = 0, l_ans0 = 0;
    std::vector<std::vector<float>> l_cov;   // 덮음 값들, 1 s 마다 (판 끝까지)
    covm::Coverage cov;
    int cov_map = -1;
    std::unique_ptr<limo_fire_rl::DeciderClient> real;
    std::vector<lxd::Answer> real_log;
    double real_ms = 0, real_max = 0;
    if (use_dec && !dec_url.empty()) {
      real = std::make_unique<limo_fire_rl::DeciderClient>(dec_url, 60000);
      ms.real = real.get();
      ms.log = mr.log = &real_log;
    }
    long d_body = 0, n_ans = 0, n_sent_all = 0, n_nomatch_all = 0, n_fb_all = 0;
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
    std::vector<float> pts, o1d;
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
      limo_fire_rl::Proprio p0;
      p0.wheel_v = e.drive.wv; p0.wheel_vy = e.drive.wvy; p0.wheel_w = e.drive.ww; p0.gyro_z = e.gyro;
      for (int i = 0; i < 7; ++i) p0.imu[i] = e.imu.out[i];
      const limo_fire_rl::RobotEnv::Window win{w.ox, w.oy, w.gw, w.gh};
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
      us.restart(); ur.restart();
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
        cpu.get_map(lm); cpu.get_trail(tr); cpu.get_camlayer(vox, pts); cpu.get_cand(cand); cpu.get_cand_cur(ccur); cpu.get_obs(img, o1d);
        const sim::World& w = worlds[(size_t)st[0].map];
        const lxc::View vs{&hc.cfg, &w, lm.data(), tr.data(), vox.empty() ? nullptr : vox.data(), &st[0], cand.data(), (uint32_t)ccur[0]};
        sim::LowCmd c1;
        int why = 0;
        const long a0 = us.n_ans();
        if (us.step(vs, o1d.data() + mdp::obs::POLICY_1D, cs, ms, &c1, &why)) {
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
        if (ur.step(vr, robot.obs_1d() + mdp::obs::POLICY_1D, cr, mr, &c2)) robot.set_cmd(c2);
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
      act[0] = (float)cs.slot();   // combine_run 규칙: 결정 때 정한 행동 칸 (다음 결정까지 그대로)
      robot.set_slot(cr.slot());
      cpu.low_prepare(act.data());
      cpu.get_obs(img, o1d);
      robot.set_depth(img.data() + mdp::obs::NEAR_SIZE + mdp::obs::FAR_SIZE);
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
        no_delay();
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
    const bool ok = d_dec == 0 && d_path == 0 && d_cmd == 0 && d_body == 0 && !(use_cand && g_refail > 0);   // 후보 규칙이 최근 실패 후보를 다시 고르면 틀림
    std::printf("%s%s\n", ok ? "통과: 실물 합친 탐사 노드 == 시뮬 (비트)" : "다름", first_bad >= 0 ? (" — 처음 다른 걸음 " + std::to_string(first_bad)).c_str() : "");
    return ok ? 0 : 1;
  } catch (const std::exception& ex) {
    std::fprintf(stderr, "low_explore_check: %s\n", ex.what());
    return 1;
  }
}
