// combine_run — 두 층을 한 판에서 합쳐 탐사를 끝까지 돌린다: 위층 = 가장 가까운 후보 규칙(slot0, 1 Hz) 또는 fuel, 아래층 = 규칙 추종기(A* + 경로 추종).
// (아래층 RL 정책 --policy 는 2026-11-16 에 그만둠 — archive/rl/README.md.) 시뮬은 아래층 걸음(EnvCfg::low_rl, step 한 번 = 0.1 s) + 합친 탐사 스위치(EnvCfg::low_explore — 탐사 판 끝 규칙 그대로, 정체는 막힘 신호만)로 돈다.
//
//   combine_run --scenes a.scene [b.scene …] [--episodes 256] [--seed 1] [--people 0] [--eval-seed] [--out <접두사>]
//               [--no-avoid] [--no-redecide] [--no-stick] [--stall-s <초>] [--close-lidar] [--env …] [--model …] [--frontier …]
//               [--upper slot0|fuel] [--fuel-* …] [--rule-cell] [--leg-log <접두사>] [--gt-bound] [--gt-tile <m>]
//   아래층은 규칙 추종기: 후보 번호를 그대로(follow 0), fuel · --rule-cell 이면 목표 칸을 그대로(follow 1).
//   판·씨앗·판 길이 상한은 frontier_run --modes slot0 와 같다 (장면 i 의 씨앗 seed · 1000003 + i · 7919, --eval-seed 면 seed · 7919 + 17 + i).
//   옛 판단 모델 실험 길(--upper decider · random · cand_tsp · rich, --decider-* · --cand-* · --branch-* · --video · --gt/--gt-base)은 2026-11-18 지움 —
//   판단 모델 라벨은 upper_lora/rollout/dec_rollout, 실물과 같은 판단 모델 위층의 시뮬 확인은 sim/tools/low_explore_check --upper decider.
//
// 위층 규칙 (환경마다, 결정 = 10 걸음마다 · 아래 "바로 다시 고름" 때):
//   · 후보 0 번부터 차례로, 피하는 칸(아래) 둘레가 아닌 첫 후보. 없으면 목표 없음(규칙 추종기의 제자리 회전 — 지도 닫힘을 기다림)
//   · 막힘 신호(sim::LowRl::stall — 10 s 동안 0.3 m 를 못 벗어남, env.yaml follow.stall_*)가 서면 그때 가던 칸(stall_tgt)을 피하는 칸에 넣고 바로 다시 고른다.
//     피하는 반경·시간은 규칙 추종기 복구와 같은 값(env.yaml recovery.same_goal_radius · blacklist_time — 둘 다 잠정). explore_lite 의 progress_timeout →
//     frontier_blacklist 와 같은 꼴 (--no-avoid 로 끔)
//   · 목표 칸을 그대로 따라가는 판(fuel · --rule-cell)만: 목표 통과 반경에 들었거나(pass_radius) 경로 끝이 목표 밖이거나 경로를 못 짰으면 다음 걸음에 바로 다시 고른다 (--no-redecide 로 끔).
//     경로 끝이 목표 밖·경로 없음은 그 칸도 피하는 칸에 넣는다
//   · --no-stick 은 첫 요약 줄의 "목표 이어 가기" 표시만 바꾼다 (이 도구 안에서 고르기에 쓰는 곳 없음)
//
// 위층 고르기 (--upper, 위 규칙의 첫 줄만 바꿈 — 피하는 칸·바로 다시 고름은 그대로):
//   slot0 : 유효 후보(빈 슬롯·피하는 칸이 아닌 후보) 가운데 번호가 가장 작은 것 — 기준선
//   fuel  : 프론티어 덩어리 순회의 첫 시점 (scripts/fuel_upper.h, --fuel-* — scripts/README "combine_run — 위층 fuel")
//   --gt-bound: 장면마다 정답 격자를 다 아는 순회의 시간 (scripts/cand_rules.h gt_tour, 타일 변 --gt-tile — 시뮬 전용 진단, 판에는 안 씀)
//
// 출력: 장면마다 성공률·실패 종류·닫힘까지 시간·평균 속도·이동 거리 + 원인 분석용 걸음 통계. --out 을 주면 <접두사>_<장면>.tsv 에 판마다 한 줄
#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <thread>
#include <vector>

#include "cand_rules.h"
#include "env_api.h"
#include "frontier.h"
#include "frontier_explorer.h"
#include "fuel_upper.h"

namespace {

struct Args {
  std::vector<std::string> scenes;
  int episodes = 256, people = 0;
  uint64_t seed = 1;
  std::string out;
  std::string env = SIM_CONFIG_DIR "/env.yaml", model = SIM_CONFIG_DIR "/sensor_model.yaml", frontier = FRONTIER_CONFIG;
  bool eval_seed = false, avoid = true, redecide = true, stick = true;
  bool close_lidar = false;   // --close-lidar: 합친 탐사 닫힘·후보 번지기를 라이다 지도만으로 (깊이 카메라 층 뺌, EnvCfg::low_explore 2)
  double stall_s = 0.0;   // > 0 이면 정체 판정 시간(env.yaml follow.stall_time)을 이 값으로 (실험용)
  std::string upper = "slot0";   // 위층 고르기: slot0 · fuel
  cr::Opt gt;                    // --gt-bound 의 순회: --gt-tile (scripts/cand_rules.h)
  fu::Opt fuel;                  // fuel: --fuel-* (scripts/fuel_upper.h)
  std::string leg_log;           // --leg-log <파일>: 목표 구간마다 (계획 길이 · 첫 회전각 · 걸린 시간) — fuel 의 v_eff · ω_eff 맞춤용 (기록만)
  bool gt_bound = false;         // --gt-bound: 장면마다 정답 격자를 다 아는 순회의 시간 (시뮬 전용 진단 — 정답 지도를 씀, 판에는 안 씀)
  bool rule_cell = false;        // --rule-cell: 규칙 추종기에 후보 번호 대신 목표 칸을 줌 (follow 1 — fuel 과 같은 길, slot0 견줌용)
};

std::string stem(const std::string& p) {
  const auto s = p.find_last_of('/');
  std::string b = s == std::string::npos ? p : p.substr(s + 1);
  const auto k = b.rfind('.');
  return k == std::string::npos ? b : b.substr(0, k);
}

enum Upper { kSlot0, kFuel };

constexpr int kBl = 8;   // 환경마다 피하는 칸 수 (고리)
struct Avoid {
  int cell[kBl];
  int until[kBl];
  int n = 0;
  Avoid() { for (int i = 0; i < kBl; ++i) { cell[i] = -1; until[i] = 0; } }
};

int run(const Args& a) {
  sim::EnvHostCfg hc = sim::load_env_cfg(a.env);
  const fb::Params P = fb::load_params(a.frontier, a.env, "fast");   // frontier_run run_hier 와 같은 판 길이 상한
  hc.cfg.action_mode = sim::kActPath;
  hc.cfg.max_steps = (int)std::ceil(P.t_safe / (hc.cfg.substeps * hc.cfg.dt)) + 100;
  hc.cfg.low_stat = hc.cfg.low_task = hc.cfg.low_rl = 1;
  if (a.stall_s > 0.0) hc.cfg.low_idle_steps = std::max(1, (int)std::lround(a.stall_s / (hc.cfg.substeps * hc.cfg.dt)));
  hc.cfg.low_explore = a.close_lidar ? 2 : 1;   // 합친 탐사: 탐사 판 끝 규칙 그대로 · 정체는 막힘 신호만 (2 = 닫힘을 라이다 지도만으로 — mdp/frontier.h)
  const sim::LidarModel lidar = sim::load_lidar_model(a.model);
  const sim::EnvCfg& c = hc.cfg;
  const int N = a.episodes, AW = mdp::act::DIM, K = mdp::hier::K, DS = c.decide_steps;
  const double dt = (double)c.substeps * c.dt;
  const int upper = a.upper == "slot0" ? kSlot0 : a.upper == "fuel" ? kFuel : -1;
  const int n_thr = std::max(1u, std::min(12u, std::thread::hardware_concurrency()));
  if (upper < 0) throw std::runtime_error("--upper 는 slot0 · fuel (decider · random · cand_tsp · rich 는 2026-11-18 지움)");
  if (upper == kFuel && (!(a.fuel.v_eff > 0.f) || !(a.fuel.w_eff > 0.f) || a.fuel.max_nodes < 1)) throw std::runtime_error("--fuel-v · --fuel-w > 0, --fuel-nodes ≥ 1");
  const float bl_r = c.rec_bl_r > 0.f ? c.rec_bl_r : 0.5f;
  const int bl_steps = c.rec_bl_steps > 0 ? c.rec_bl_steps : (int)std::lround(10.0 / dt);
  std::printf("아래층: 규칙 추종기 (A* + 경로 추종)\n");
  std::printf("위층 고르기: %s\n", upper == kFuel ? "프론티어 덩어리 순회의 첫 시점 (fuel, 로봇 지도만)" : "유효 후보 가운데 번호가 가장 작은 것 (slot0)");
  std::printf("위층: 0 번 후보 규칙 · 결정 %d 걸음마다 · 막힘 신호 피하기 %s (반경 %.2f m · %.0f s) · 바로 다시 고름 %s · 목표 이어 가기 %s | 정체 = %.0f s 동안 %.2f m · 통과 반경 %.2f m · 판 상한 %.0f s · 사람 %d\n",
              DS, a.avoid ? "켬" : "끔", bl_r, bl_steps * dt, a.redecide ? "켬" : "끔", a.stick ? "켬" : "끔", c.low_idle_steps * dt, c.low_idle_d, c.low_pass_r, c.max_steps * dt, a.people);
  if (upper == kFuel)
    std::printf("fuel: 시간 비용 %s (v_eff %.2f m/s · ω_eff %.2f rad/s · 첫 회전 배 %.2f · w_c %.2f) · 전역 순회 %s (덩어리 상한 %d · 합침 %.2f m · 나눔 %.2f m) · 시점 고르기 %s (t0 %.2f s) · 붙들기 여유 %.2f · 방 마무리 %s (떠남 벌점 %.1f s · 문 %.2f m)\n",
                a.fuel.time ? "켬" : "끔", a.fuel.v_eff, a.fuel.w_eff, a.fuel.turn_w, a.fuel.wc, a.fuel.global ? "켬" : "끔", a.fuel.max_nodes, a.fuel.merge_m, a.fuel.split_m, a.fuel.vp ? "켬" : "끔", a.fuel.t0, a.fuel.hyst,
                a.fuel.room ? "켬" : "끔", a.fuel.room_pen, a.fuel.door_m);
  float* d_act;
  cudaMalloc(&d_act, sizeof(float) * AW * N);
  for (size_t wi = 0; wi < a.scenes.size(); ++wi) {
    std::vector<sim::World> one{sim::build_world(stem(a.scenes[wi]), sim::load_scene(a.scenes[wi]), hc)};
    const sim::World& W = one[0];
    const uint64_t seed = a.eval_seed ? a.seed * 7919ull + 17 + wi : a.seed * 1000003ull + wi * 7919ull;
    if (a.gt_bound) {   // 시뮬 전용 진단 (정답 지도): cand_rules.h gt_tour — 1.5 m 타일 닻 탐욕 덮기 + 최근접 이웃·2-opt, 360° · 지도 사거리, 시간 = 길이 / v_eff + 회전 / ω_eff (fuel 맞춤 값)
      const cr::GtMap gm = cr::build_gt(W, hc.cfg, a.gt);
      const cr::GtTour t = cr::gt_tour(W, hc.cfg, gm, a.gt);
      std::printf("[%s] 정답 지도를 다 아는 순회 (시뮬 전용 진단 · 정답 사용): 시점 %d 개 · 덮음 %.3f · 길이 %.1f m · 회전 %.1f rad → 시간 %.1f s (v %.2f m/s · ω %.2f rad/s)\n", W.name.c_str(), t.n_view, t.cover,
                  t.len_m, t.turn_rad, t.len_m / a.fuel.v_eff + t.turn_rad / a.fuel.w_eff, a.fuel.v_eff, a.fuel.w_eff);
      std::fflush(stdout);
    }
    sim::Env env(hc, one, lidar, N, seed, true);
    env.reset();
    env.sync();
    if (a.people > 0) {   // frontier_run 과 같은 차례 (people 0 이면 안 건드림)
      env.set_people_max(std::vector<int>(N, a.people));
      env.reset();
      env.sync();
    }
    std::vector<float> act((size_t)AW * N, 0.f);
    std::vector<sim::LowCmd> cmd(N, sim::LowCmd{-1, 0});
    std::vector<int32_t> cand, done;
    std::vector<sim::EnvState> st;
    std::vector<sim::EpInfo> ep, first(N);
    std::vector<sim::LowEp> low;
    std::vector<sim::LowRl> rl;
    std::vector<Avoid> av(N);
    std::vector<int> fin(N, 0), redo(N, 1), sig_prev(N, 0), nsteps(N, 0), last_dec(N, 0), goal_prev(N, -1), sw_step(N, 0), end_since(N, -1);
    std::vector<float> end_v(N, 0.f), end_d(N, 0.f), end_w(N, 0.f);
    constexpr int kH = 50;   // 끝 직전 5 s 의 자리·후진 (충돌 원인 분석)
    std::vector<float> hx((size_t)N * kH, 0.f), hy((size_t)N * kH, 0.f);
    std::vector<unsigned char> hb((size_t)N * kH, 0);
    std::vector<double> dist(N, 0.0), chord(N, 0.0), px(N), py(N), cx(N), cy(N);
    // 판마다 걸음 통계 (원인 분석)
    std::vector<int> fc_step(N, -1), t_n(N, 0), t_sum(N, 0), t_cell(N, -1), t_start(N, 0), t_prev(N, -1);   // 처음 닫힌 걸음 · 60° 이상 전환 (수 · 걸음 합 · 재는 중인 목표 칸 · 시작 걸음 · 직전 목표 칸)
    std::vector<int> n_dec(N, 0), n_sw(N, 0), n_stall(N, 0), n_reach(N, 0), n_off(N, 0), n_nopath(N, 0), n_back(N, 0), n_slow(N, 0), n_nogoal(N, 0), n_avoided(N, 0), n_early(N, 0);
    std::vector<double> tgt_d(N, 0.0), w_abs(N, 0.0);
    long dbin[4] = {0, 0, 0, 0};   // 결정 때 목표까지 직선 거리: < 1 · 1 ~ 3 · 3 ~ 15 · ≥ 15 m
    env.get_state(st);
    for (int i = 0; i < N; ++i) { px[i] = cx[i] = st[i].x; py[i] = cy[i] = st[i].y; }
    const auto avoided = [&](int i, int cell, int step) {
      if (!a.avoid || cell < 0) return false;
      const float q = bl_r / c.res;
      for (int k = 0; k < kBl; ++k) {
        if (av[i].cell[k] < 0 || step >= av[i].until[k]) continue;
        const int dx = cell % W.gw - av[i].cell[k] % W.gw, dy = cell / W.gw - av[i].cell[k] / W.gw;
        if ((float)(dx * dx + dy * dy) <= q * q) return true;
      }
      return false;
    };
    const auto avoid_add = [&](int i, int cell, int step) {
      if (!a.avoid || cell < 0) return;
      av[i].cell[av[i].n % kBl] = cell;
      av[i].until[av[i].n % kBl] = step + bl_steps;
      av[i].n += 1;
    };
    // 위층 고르기 (--upper): 걸음마다 결정할 환경의 유효 후보를 모으고(need · vslot · nv), slot0 차례를 pick 에 둔다. fuel 은 목표 칸을 tcell 에
    std::vector<int> need(N, 0), nv(N, 0), vslot((size_t)N * K, 0), pick(N, -1), ask, n_asked(N, 0);
    std::vector<int16_t> r_map; std::vector<uint32_t> r_trail; std::vector<uint8_t> r_cam; std::vector<float> r_pts;   // fuel 만
    std::vector<fu::Mem> fmem(N); std::vector<fu::Work> f_works(n_thr); std::vector<int> tcell(N, -1); long f_fall = 0;   // fuel 만
    std::vector<int> lg_tgt(N, -1), lg_step(N, 0);   // --leg-log: 지금 구간 목표 · 시작 걸음 · 첫 회전각 · 시작 직선 거리
    std::vector<float> lg_dth(N, -1.f), lg_d0(N, 0.f), lg_len(N, -1.f);
    std::FILE* flg = nullptr;
    if (!a.leg_log.empty()) {
      flg = std::fopen((a.leg_log + "_" + W.name + ".tsv").c_str(), "w");
      if (!flg) throw std::runtime_error("--leg-log 파일을 못 엶");
      std::fputs("env\tstep0\tsteps\tt_s\tplan_len_m\tturn0_rad\tstraight_m\n", flg);
    }
    long u_ask = 0, u_first = 0;
    long u_rank[mdp::hier::K] = {0}, u_slot[mdp::hier::K] = {0}, u_nv = 0;
    int left = N;
    for (long t = 0; left > 0 && t < (long)c.max_steps + 10 * DS; ++t) {
      env.get_cand(cand);
      ask.clear();
      for (int i = 0; i < N; ++i) {   // 결정할 환경과 유효 후보
        need[i] = 0;
        if (fin[i]) continue;
        const sim::EnvState& s = st[i];
        if (!(s.step % DS == 0 || redo[i])) continue;
        need[i] = 1;
        nv[i] = 0;
        for (int j = 0; j < K; ++j) {
          const int cell = cand[(size_t)i * K + j];
          if (cell < 0) continue;
          if (avoided(i, cell, s.step)) { if (nv[i] == 0) n_avoided[i] += 1; continue; }   // 셈은 첫 유효 후보 앞의 것만 (slot0 규칙이 건너뛴 수)
          vslot[(size_t)i * K + nv[i]++] = j;
        }
        pick[i] = nv[i] > 0 ? vslot[(size_t)i * K] : -1;
        if (upper != kSlot0 && nv[i] >= 2) ask.push_back(i);
      }
      if (upper == kFuel) {   // fuel (scripts/fuel_upper.h): 결정하는 모든 환경 (후보 수와 무관 — 전역 순회는 후보 밖 덩어리도 봄)
        bool any = false;
        for (int i = 0; i < N; ++i) any = any || need[i];
        if (any) { env.get_map(r_map); env.get_trail(r_trail); env.get_camlayer(r_cam, r_pts); }
        const size_t tws = (size_t)(c.size + 31) / 32, per_trail = 2 * c.size * tws, per_map = (size_t)c.size * c.size;
        std::atomic<int> nx{0};   // 환경마다 따로 (스레드마다 버퍼) — 결정은 환경끼리 독립이라 결과는 스레드 수와 무관
        std::vector<std::thread> th;
        for (int t = 0; t < n_thr; ++t)
          th.emplace_back([&, t]() {
            for (int i; (i = nx.fetch_add(1)) < N;) {
              if (!need[i]) continue;
              const int step_i = st[i].step;
              const fu::In in{&c, &W, &r_map[i * per_map], &r_trail[i * per_trail], r_cam.empty() ? nullptr : &r_cam[i * per_map], &st[i], &cand[(size_t)i * K],
                              [&, i, step_i](int cell) { return avoided(i, cell, step_i); }};
              tcell[i] = fu::decide(in, a.fuel, fmem[i], f_works[t]);
            }
          });
        for (auto& x : th) x.join();
        for (int i = 0; i < N; ++i) if (need[i] && tcell[i] < 0) f_fall += 1;
      }
      for (int i : ask) {   // 고른 후보의 자리 (유효 후보 가운데 몇 째 · 후보 번호)
        int rank = 0;
        while (rank < nv[i] && vslot[(size_t)i * K + rank] != pick[i]) ++rank;
        u_ask += 1;
        n_asked[i] += 1;
        u_nv += nv[i];
        u_rank[std::min(rank, K - 1)] += 1;
        u_slot[pick[i]] += 1;
        u_first += rank == 0;
      }
      for (int i = 0; i < N; ++i) {
        if (fin[i]) { cmd[i] = sim::LowCmd{-1, 0}; act[(size_t)AW * i] = 0.f; continue; }   // 끝난 환경: 규칙 탐사로 흘려 둠 (안 셈)
        const sim::EnvState& s = st[i];
        if (!need[i]) continue;
        if (s.step % DS != 0) n_early[i] += 1;
        redo[i] = 0;
        last_dec[i] = s.step;
        const int k = pick[i];
        n_dec[i] += 1;
        int tgt = k >= 0 ? cand[(size_t)i * K + k] : -1;
        const bool fz = upper == kFuel && tcell[i] >= 0;   // fuel 이 낸 목표 칸 (못 내면 slot0 차례 그대로)
        const bool fnone = upper == kFuel && tcell[i] == fu::kNone;   // 프론티어는 있으나 시점 자격 되는 덩어리 없음: 목표 없음 (slot0 로 안 떨어짐)
        if (fz) tgt = tcell[i];
        if (fnone) tgt = -1;
        if (fz || (a.rule_cell && tgt >= 0)) {   // 규칙 추종기 + fuel (또는 --rule-cell): 단독 과제 꼴(follow 1 — 목표 칸을 그대로 따라감)
          cmd[i] = sim::LowCmd{tgt, 1};
          act[(size_t)AW * i] = 0.f;
        } else {
          cmd[i] = sim::LowCmd{-1, 0};
          act[(size_t)AW * i] = fnone ? -1.f : (float)k;   // fuel 시점 자격 없음: 규칙 추종기도 목표 없음 (low_explore_ctl.h slot_ −1)
        }
        if (tgt < 0) n_nogoal[i] += 1;
        else {
          const double d = std::hypot(W.ox + ((double)(tgt % W.gw) + 0.5) * c.res - (s.x + s.ex), W.oy + ((double)(tgt / W.gw) + 0.5) * c.res - (s.y + s.ey));
          tgt_d[i] += d;
          dbin[d < 1.0 ? 0 : d < 3.0 ? 1 : d < 15.0 ? 2 : 3] += 1;
        }
      }
      cudaMemcpy(d_act, act.data(), sizeof(float) * act.size(), cudaMemcpyHostToDevice);
      env.set_low_cmd(cmd);
      env.low_prepare(d_act);
      env.sync();
      env.step(d_act);
      env.sync();
      env.get_done(done);
      env.get_ep(ep);
      env.get_state(st);
      env.get_low(low);
      env.get_low_rl(rl);
      for (int i = 0; i < N; ++i) {
        if (fin[i]) continue;
        nsteps[i] += 1;
        if (done[i]) {
          fin[i] = 1;
          first[i] = ep[i];
          first[i].done = done[i];
          --left;
          continue;
        }
        const sim::EnvState& s = st[i];
        dist[i] += std::hypot(s.x - px[i], s.y - py[i]);
        px[i] = s.x;
        py[i] = s.y;
        if (s.step % DS == 0) {   // frontier_run 과 같은 셈: 결정(1 s) 사이 직선 거리의 합
          chord[i] += std::hypot(s.x - cx[i], s.y - cy[i]);
          cx[i] = s.x;
          cy[i] = s.y;
        }
        if (s.goal_cell != goal_prev[i]) {
          if (goal_prev[i] >= 0 && s.goal_cell >= 0) n_sw[i] += 1;
          sw_step[i] = s.step;
          goal_prev[i] = s.goal_cell;
        }
        if (s.map_closed && fc_step[i] < 0) fc_step[i] = s.step;
        {   // 60° 이상 전환: 목표 칸이 바뀐 걸음에 새 목표 쪽 방위가 60° 이상이면 재기 시작, 방위가 30° 안에 들면 끝 (그 전에 목표가 또 바뀌면 버림)
          const auto bearing = [&](int cell) {
            const double dx = W.ox + ((double)(cell % W.gw) + 0.5) * c.res - s.x, dy = W.oy + ((double)(cell / W.gw) + 0.5) * c.res - s.y;
            return std::fabs(std::remainder(std::atan2(dy, dx) - (double)s.th, 6.283185307179586)) * 57.29577951308232;
          };
          if (t_cell[i] >= 0) {
            if (s.goal_cell != t_cell[i]) t_cell[i] = -1;
            else if (bearing(t_cell[i]) < 30.0) { t_n[i] += 1; t_sum[i] += s.step - t_start[i]; t_cell[i] = -1; }
          }
          if (s.goal_cell != t_prev[i]) {
            if (t_prev[i] >= 0 && s.goal_cell >= 0 && bearing(s.goal_cell) >= 60.0) { t_cell[i] = s.goal_cell; t_start[i] = s.step; }
            t_prev[i] = s.goal_cell;
          }
        }
        end_v[i] = s.v;
        end_w[i] = s.w;
        hx[(size_t)i * kH + nsteps[i] % kH] = s.x;
        hy[(size_t)i * kH + nsteps[i] % kH] = s.y;
        hb[(size_t)i * kH + nsteps[i] % kH] = s.cmd_v < 0.f;
        end_since[i] = s.step - sw_step[i];
        end_d[i] = s.goal_cell >= 0 ? (float)std::hypot(W.ox + ((double)(s.goal_cell % W.gw) + 0.5) * c.res - s.x, W.oy + ((double)(s.goal_cell / W.gw) + 0.5) * c.res - s.y) : -1.f;
        n_back[i] += s.cmd_v < 0.f;
        n_slow[i] += std::fabs(s.v) < 0.1f;
        w_abs[i] += std::fabs(s.w);
        // 막힘 신호 → 그 칸을 피하고 바로 다시 고름
        if (rl[i].stall && !sig_prev[i]) {
          n_stall[i] += 1;
          avoid_add(i, rl[i].stall_tgt, s.step);
          fmem[i] = fu::Mem();
          if (a.avoid) redo[i] = 1;
        }
        sig_prev[i] = rl[i].stall;
        if (lg_tgt[i] >= 0 && lg_dth[i] < 0.f && s.goal_cell == lg_tgt[i] && s.p_n > 0) {   // --leg-log: 처음 짜인 경로의 0.5 m 앞 칸 방향과 머리 방향 차
          const double ex = s.x + s.ex, ey = s.y + s.ey;
          const int pi = std::max(0, std::min((int)s.p_i, (int)s.p_n - 1));
          int q = pi;
          for (; q < s.p_n - 1; ++q)
            if (std::hypot(W.ox + (s.wp[2 * q] + 0.5) * c.res - ex, W.oy + (s.wp[2 * q + 1] + 0.5) * c.res - ey) >= 0.5) break;
          const double dir = std::atan2(W.oy + (s.wp[2 * q + 1] + 0.5) * c.res - ey, W.ox + (s.wp[2 * q] + 0.5) * c.res - ex);
          lg_dth[i] = (float)std::fabs(std::remainder(dir - (double)(s.th + s.eth), 6.283185307179586));
          double len = std::hypot(W.ox + (s.wp[2 * pi] + 0.5) * c.res - ex, W.oy + (s.wp[2 * pi + 1] + 0.5) * c.res - ey);   // 처음 짜인 경로 길이: 로봇 → 지금 웨이포인트 → … → 끝
          for (int r = pi; r + 1 < s.p_n; ++r) len += std::hypot((double)(s.wp[2 * r + 2] - s.wp[2 * r]), (double)(s.wp[2 * r + 3] - s.wp[2 * r + 1])) * c.res;
          lg_len[i] = (float)len;
        }
        if (cmd[i].follow != 0 && cmd[i].target != lg_tgt[i]) {
          lg_tgt[i] = cmd[i].target; lg_step[i] = s.step; lg_dth[i] = -1.f;
          lg_d0[i] = (float)std::hypot(W.ox + ((double)(cmd[i].target % W.gw) + 0.5) * c.res - (s.x + s.ex), W.oy + ((double)(cmd[i].target / W.gw) + 0.5) * c.res - (s.y + s.ey));
        }
        if (cmd[i].follow == 1) {
          const sim::LowEp& L = low[i];
          const bool reached = L.reach_step == s.step && L.reached > 0;
          const bool off = !reached && L.hold != 0;
          const bool nopath = !reached && !off && s.step > last_dec[i] && (s.goal_cell != cmd[i].target || s.p_n <= 0);
          if (reached) n_reach[i] += 1;
          if (reached && flg && lg_dth[i] >= 0.f)
            std::fprintf(flg, "%d\t%d\t%d\t%.1f\t%.3f\t%.3f\t%.3f\n", i, lg_step[i], s.step - lg_step[i], (s.step - lg_step[i]) * dt, lg_len[i], lg_dth[i], lg_d0[i]);
          if (reached || off || nopath) fmem[i] = fu::Mem();
          if (a.redecide && (reached || off || nopath)) {
            if (off) { n_off[i] += 1; avoid_add(i, cmd[i].target, s.step); }
            if (nopath) { n_nopath[i] += 1; avoid_add(i, cmd[i].target, s.step); }
            redo[i] = 1;
          }
        }
      }
    }
    // 장면 요약
    double n = N, succ = 0, cw = 0, cp = 0, idle = 0, trunc = 0, ts = 0, t_all = 0, d_all = 0, ch_all = 0, d_succ = 0, cov_s = 0;
    double s_dec = 0, s_sw = 0, s_stall = 0, s_reach = 0, s_off = 0, s_nopath = 0, s_back = 0, s_slow = 0, s_nogoal = 0, s_avoided = 0, s_early = 0, s_steps = 0, s_tgt = 0, s_w = 0, ep_stall = 0;
    double fc_s = 0, s_tn = 0, s_tsum = 0;
    std::FILE* f = nullptr;
    if (!a.out.empty()) {
      f = std::fopen((a.out + "_" + W.name + ".tsv").c_str(), "w");
      if (!f) throw std::runtime_error("--out 파일을 못 엶");
      std::fputs("env\tdone\tt\tcoverage\tdist_m\tchord_m\tdecisions\tswitches\tstalls\treached\toff\tnopath\tback_steps\tslow_steps\tnogoal_dec\tend_v\tend_steps_since_goal_change\tend_goal_dist\tend_w\tend_5s_span_m\tend_5s_back_steps\tfirst_closed_t\tturn60_n\tturn60_steps\n", f);
    }
    for (int i = 0; i < N; ++i) {
      const int dd = fin[i] ? first[i].done : mdp::cur::kTruncated;
      const double ti = fin[i] ? first[i].t : nsteps[i] * dt;
      succ += dd == mdp::cur::kDeclareOk;
      cw += dd == mdp::cur::kCollision;
      cp += dd == mdp::cur::kCollisionPerson;
      idle += dd == mdp::cur::kNoProgress;
      trunc += dd == mdp::cur::kTruncated;
      t_all += ti;
      d_all += dist[i];
      ch_all += chord[i];
      if (dd == mdp::cur::kDeclareOk) { ts += first[i].t; d_succ += dist[i]; cov_s += first[i].coverage; }
      s_dec += n_dec[i]; s_sw += n_sw[i]; s_stall += n_stall[i]; s_reach += n_reach[i]; s_off += n_off[i]; s_nopath += n_nopath[i];
      s_back += n_back[i]; s_slow += n_slow[i]; s_nogoal += n_nogoal[i]; s_avoided += n_avoided[i]; s_early += n_early[i]; s_steps += nsteps[i]; s_tgt += tgt_d[i]; s_w += w_abs[i];
      ep_stall += n_stall[i] > 0;
      float span = 0.f;
      int nb = 0;
      if (nsteps[i] > kH) {
        const size_t o = (size_t)i * kH;
        const int last = (nsteps[i] - 1) % kH;   // 끝난 걸음 바로 앞
        for (int q = 0; q < kH; ++q) {
          span = std::max(span, (float)std::hypot(hx[o + q] - hx[o + last], hy[o + q] - hy[o + last]));
          nb += hb[o + q];
        }
      }
      if (f)
        std::fprintf(f, "%d\t%d\t%.1f\t%.4f\t%.2f\t%.2f\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%.2f\t%d\t%.2f\t%.2f\t%.2f\t%d\t%.1f\t%d\t%d", i, dd, ti, fin[i] ? first[i].coverage : 0.f, dist[i], chord[i], n_dec[i], n_sw[i], n_stall[i], n_reach[i], n_off[i],
                     n_nopath[i], n_back[i], n_slow[i], n_nogoal[i], end_v[i], end_since[i], end_d[i], end_w[i], span, nb, fc_step[i] >= 0 ? fc_step[i] * dt : -1.0, t_n[i], t_sum[i]);
      if (f) std::fputs("\n", f);
      if (dd == mdp::cur::kDeclareOk) fc_s += fc_step[i] >= 0 ? fc_step[i] * dt : first[i].t;
      s_tn += t_n[i]; s_tsum += t_sum[i];
    }
    if (f) std::fclose(f);
    const char* tag = "rule";
    std::printf("[%s %s] 판 %d: 성공 %.3f · 벽 충돌 %.3f · 사람 충돌 %.3f · 진행 없음 %.3f · 끊김 %.3f | 닫힘까지 %.1f s (성공 판 평균, 정답 덮음 %.3f) · 판 평균 길이 %.1f s\n", tag, W.name.c_str(), N, succ / n,
                cw / n, cp / n, idle / n, trunc / n, succ > 0 ? ts / succ : 0.0, succ > 0 ? cov_s / succ : 0.0, t_all / n);
    std::printf("[%s %s] 평균 속도 %.3f m/s (걸음 거리 합 / 시간) · %.3f m/s (1 s 사이 직선 거리 합 / 시간 — frontier_run 과 같은 셈) · 이동 거리 모든 판 %.1f m · 성공 판 %.1f m\n", tag, W.name.c_str(),
                t_all > 0 ? d_all / t_all : 0.0, t_all > 0 ? ch_all / t_all : 0.0, d_all / n, succ > 0 ? d_succ / succ : 0.0);
    const double nd = std::max(1.0, s_dec), ng = std::max(1.0, s_dec - s_nogoal), nst = std::max(1.0, s_steps);
    std::printf("[%s %s] 위층: 판당 결정 %.1f 개 (그중 10 걸음 주기 밖 %.3f) · 목표 없는 결정 %.3f · 목표 칸 바뀜 판당 %.1f 번 (%.1f s 에 한 번) · 피하는 칸이라 건너뛴 후보 %.0f 개 | 결정 때 목표까지 직선 거리 평균 %.2f m — "
                "1 m 안 %.3f · 1~3 m %.3f · 3~15 m %.3f · 15 m 밖 %.3f\n", tag, W.name.c_str(), s_dec / n, s_early / nd, s_nogoal / nd, s_sw / n, s_sw > 0 ? t_all / s_sw : 0.0, s_avoided, s_tgt / ng,
                dbin[0] / ng, dbin[1] / ng, dbin[2] / ng, dbin[3] / ng);
    std::printf("[%s %s] 아래층: 막힘 신호 %.0f 번 (선 판 비율 %.3f) · 통과 %.0f · 경로 끝이 목표 밖 %.0f · 경로 없음 %.0f | 걸음 비율 — 뒤로 %.3f · 0.1 m/s 아래 %.3f · 평균 |ω| %.2f rad/s\n",
                tag, W.name.c_str(), s_stall, ep_stall / n, s_reach, s_off, s_nopath, s_back / nst, s_slow / nst, s_w / nst);
    std::printf("[%s %s] 더: 지도가 처음 닫힌 시각 %.1f s (성공 판 평균) · 뒤로 걸음 비율 %.4f · 60° 이상 전환 판당 %.1f 번 · 평균 %.2f s (목표 쪽 방위 30° 안까지; 그 전에 목표가 바뀐 것은 뺌)\n",
                tag, W.name.c_str(), succ > 0 ? fc_s / succ : 0.0, s_back / nst, s_tn / n, s_tn > 0 ? s_tsum * dt / s_tn : 0.0);
    std::printf("[%s %s] 벽 충돌 %.2f 건 / 100 m (벽 충돌 판 %.0f · 모든 판 이동 거리 합 %.0f m)\n", tag, W.name.c_str(), d_all > 0 ? cw / d_all * 100.0 : 0.0, cw, d_all);
    if (upper != kSlot0) {
      const double na = std::max(1L, u_ask);
      // 고른 결정 · 몇 째 · 번호 분포: 옛 비교 줄 꼴 그대로 (fuel 판의 pick 은 slot0 차례라 늘 0 째 — 판단 모델 쪽 셈은 2026-11-18 지워 0)
      std::printf("[%s %s] 위층 고르기 %s: 고른 결정(유효 후보 2 개 이상) %ld 개 (모든 결정의 %.3f, 유효 후보 평균 %.2f 개) · slot0 로 떨어짐 0 (연결·시간 끝 0 · HTTP 0 · 답 꼴 0) · 특징 어긋남 0\n", tag, W.name.c_str(),
                  a.upper.c_str(), u_ask, u_ask / nd, u_nv / na);
      std::printf("[%s %s] 고른 후보가 유효 후보 가운데 몇 째인가 (0 = slot0 규칙과 같은 것): ", tag, W.name.c_str());
      for (int k = 0; k < K; ++k) std::printf("%d: %.3f%s", k, u_rank[k] / na, k + 1 < K ? " · " : "\n");
      std::printf("[%s %s] 고른 후보 번호 분포: ", tag, W.name.c_str());
      for (int k = 0; k < K; ++k) std::printf("%d: %.3f%s", k, u_slot[k] / na, k + 1 < K ? " · " : "\n");
    }
    if (upper == kFuel) {
      fu::Stat F;
      for (fu::Work& fw : f_works) { F.add(fw.st); fw.st = fu::Stat(); }
      const double fd = std::max(1L, F.dec);
      std::printf("[%s %s] fuel: 결정 %ld 개 · 노드 평균 %.1f 개 · 정확한 순회 %.3f · 지금 목표 덩어리를 찾은 결정 중 그대로 %ld · 바꿈 %ld · slot0 로 떨어짐 %ld | CPU %.2f ms/결정 (최대 %.1f ms, 한 스레드)\n", tag, W.name.c_str(), F.dec, F.nodes / fd,
                  F.exact / fd, F.kept, F.sw, f_fall, F.ms / fd, F.ms_max);
      if (F.g_dec > 0) {
        const double gd = (double)F.g_dec;
        std::printf("[%s %s] fuel 후보 모음 (전역 결정 %ld 개): 원래 덩어리 평균 %.1f (최대 %ld) → 합침·나눔 뒤 %.1f (최대 %ld) | 상한 %d 에 걸린 결정 %.3f · 빠진 덩어리 %ld 개 — 길 평균 %.1f m (최대 %.1f) · 크기 평균 %.2f m (최대 %.2f)"
                    " | 후보 상한 %d 에 걸린 결정 %.3f · 빠진 후보 %ld 개 — 길 평균 %.1f m (최대 %.1f) · 틈 평균 %.2f m (최대 %.2f)\n", tag, W.name.c_str(), F.g_dec, F.g_raw / gd, F.g_rawmax, F.g_mid / gd, F.g_midmax,
                    a.fuel.max_nodes, F.cap_dec / gd, F.cap_drop, F.cap_drop ? F.cap_d / F.cap_drop : 0.0, F.cap_dmax, F.cap_drop ? F.cap_s / F.cap_drop : 0.0, F.cap_smax,
                    fu::In{}.cand_cap, F.cc_dec / gd, F.cc_drop, F.cc_drop ? F.cc_d / F.cc_drop : 0.0, F.cc_dmax, F.cc_drop ? F.cc_s / F.cc_drop : 0.0, F.cc_smax);
      }
      std::printf("[%s %s] fuel 목표를 후보 칸으로 나타낼 수 있나 (시뮬 후보가 있는 결정 %ld 개): 고른 덩어리가 후보 %d 개 안 %.3f · 그 후보 칸 0.5 m 안이 목표 %.3f · 목표–후보 칸 평균 거리 %.2f m\n", tag, W.name.c_str(),
                  F.slot_n, K, F.slot_n ? (double)F.slot_in / F.slot_n : 0.0, F.slot_n ? (double)F.slot_vp / F.slot_n : 0.0, F.slot_in ? F.slot_d / F.slot_in : 0.0);
    }
    if (flg) std::fclose(flg);
    {
      double qa = 0, qs = 0;
      for (int i = 0; i < N; ++i) { qa += n_asked[i]; if (fin[i] && first[i].done == mdp::cur::kDeclareOk) qs += n_asked[i]; }
      if (upper != kSlot0) std::printf("[%s %s] 판당 물은 결정 %.1f 개 (성공 판 평균 %.1f 개)\n", tag, W.name.c_str(), qa / n, succ > 0 ? qs / succ : 0.0);
    }
    std::fflush(stdout);
  }
  cudaFree(d_act);
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  setenv("SIM_EXACT", "1", 0);   // frontier_run 과 같은 광선 길 (정확한 길)
  Args a;
  for (int i = 1; i < argc; ++i) {
    const std::string k = argv[i];
    auto nx = [&]() -> std::string {
      if (i + 1 >= argc) throw std::runtime_error(k + " 뒤에 값");
      return argv[++i];
    };
    try {
      if (k == "--scenes") { while (i + 1 < argc && argv[i + 1][0] != '-') a.scenes.push_back(argv[++i]); }
      else if (k == "--episodes") a.episodes = std::stoi(nx());
      else if (k == "--seed") a.seed = std::stoull(nx());
      else if (k == "--people") a.people = std::stoi(nx());
      else if (k == "--out") a.out = nx();
      else if (k == "--env") a.env = nx();
      else if (k == "--model") a.model = nx();
      else if (k == "--frontier") a.frontier = nx();
      else if (k == "--eval-seed") a.eval_seed = true;
      else if (k == "--no-avoid") a.avoid = false;
      else if (k == "--no-redecide") a.redecide = false;
      else if (k == "--no-stick") a.stick = false;
      else if (k == "--stall-s") a.stall_s = std::stod(nx());
      else if (k == "--upper") a.upper = nx();
      else if (k == "--fuel-time") a.fuel.time = std::stoi(nx()) != 0;
      else if (k == "--fuel-global") a.fuel.global = std::stoi(nx()) != 0;
      else if (k == "--fuel-vp") a.fuel.vp = std::stoi(nx()) != 0;
      else if (k == "--fuel-room") a.fuel.room = std::stoi(nx()) != 0;
      else if (k == "--fuel-hyst") a.fuel.hyst = std::stof(nx());
      else if (k == "--fuel-vp-min") a.fuel.vp_min = std::stof(nx());
      else if (k == "--fuel-vp-clear") a.fuel.vp_clear = std::stof(nx());
      else if (k == "--fuel-first") a.fuel.first = std::stoi(nx()) != 0;
      else if (k == "--close-lidar") a.close_lidar = true;
      else if (k == "--fuel-v") a.fuel.v_eff = std::stof(nx());
      else if (k == "--fuel-w") a.fuel.w_eff = std::stof(nx());
      else if (k == "--fuel-t0") a.fuel.t0 = std::stof(nx());
      else if (k == "--fuel-turn") a.fuel.turn_w = std::stof(nx());
      else if (k == "--fuel-wc") a.fuel.wc = std::stof(nx());
      else if (k == "--fuel-nodes") a.fuel.max_nodes = std::stoi(nx());
      else if (k == "--fuel-merge") a.fuel.merge_m = std::stof(nx());
      else if (k == "--fuel-split") a.fuel.split_m = std::stof(nx());
      else if (k == "--fuel-room-pen") a.fuel.room_pen = std::stof(nx());
      else if (k == "--fuel-door") a.fuel.door_m = std::stof(nx());
      else if (k == "--leg-log") a.leg_log = nx();
      else if (k == "--rule-cell") a.rule_cell = true;
      else if (k == "--gt-bound") a.gt_bound = true;
      else if (k == "--gt-tile") a.gt.tile_m = std::stof(nx());
      else { std::fprintf(stderr, "모르는 인자 %s\n", k.c_str()); return 2; }
    } catch (const std::exception& e) {
      std::fprintf(stderr, "combine_run: %s\n", e.what());
      return 2;
    }
  }
  if (a.scenes.empty()) {
    std::fprintf(stderr, "combine_run --scenes a.scene [b.scene …] [--episodes 256] [--seed 1] [--people 0] [--eval-seed] [--out <접두사>] [--no-avoid] [--no-redecide] [--no-stick]\n");
    return 2;
  }
  try {
    return run(a);
  } catch (const std::exception& e) {
    std::fprintf(stderr, "combine_run: %s\n", e.what());
    return 1;
  }
}
