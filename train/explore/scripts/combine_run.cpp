// combine_run — 두 층을 한 판에서 합쳐 탐사를 끝까지 돌린다: 위층 = 가장 가까운 후보 규칙(slot0, 1 Hz), 아래층 = 학습한 RL 정책(10 Hz) 또는 규칙 추종기.
// 시뮬은 아래층 RL 걸음(EnvCfg::low_rl, step 한 번 = 0.1 s) + 합친 탐사 스위치(EnvCfg::low_explore — 탐사 판 끝 규칙 그대로, 정체는 막힘 신호만)로 돈다.
//
//   combine_run --scenes a.scene [b.scene …] [--policy <아래층 체크포인트>] [--episodes 256] [--seed 1] [--people 0] [--eval-seed] [--out <접두사>]
//               [--no-avoid] [--no-redecide] [--no-stick] [--hold <걸음>] [--stall-s <초>] [--rule-near <m>] [--env …] [--model …] [--frontier …]
//   --policy 를 주면 아래층 RL(LowCmd::follow 2 — 목표 칸 = 고른 후보 칸), 안 주면 같은 걸음 루프에서 규칙 추종기(follow 0 — 후보 번호를 그대로, 대조).
//   판·씨앗·판 길이 상한은 frontier_run --modes slot0 와 같다 (장면 i 의 씨앗 seed · 1000003 + i · 7919, --eval-seed 면 seed · 7919 + 17 + i).
//
// 위층 규칙 (환경마다, 결정 = 10 걸음마다 · 아래 "바로 다시 고름" 때):
//   · 후보 0 번부터 차례로, 피하는 칸(아래) 둘레가 아닌 첫 후보. 없으면 목표 없음(규칙 추종기의 제자리 회전 — 지도 닫힘을 기다림)
//   · 고른 후보의 덩어리가 지금 목표 칸을 품으면(Env::get_cand_cur — hier_path.h act_same 과 같은 뜻) 목표 칸을 그대로 둔다 (--no-stick 으로 끔)
//   · 막힘 신호(sim::LowRl::stall — 10 s 동안 0.3 m 를 못 벗어남, env.yaml follow.stall_*)가 서면 그때 가던 칸(stall_tgt)을 피하는 칸에 넣고 바로 다시 고른다.
//     피하는 반경·시간은 규칙 추종기 복구와 같은 값(env.yaml recovery.same_goal_radius · blacklist_time — 둘 다 잠정). explore_lite 의 progress_timeout →
//     frontier_blacklist 와 같은 꼴 (--no-avoid 로 끔)
//   · 아래층 RL 만: 목표 통과 반경에 들었거나(pass_radius) 경로 끝이 목표 밖이거나 경로를 못 짰으면 다음 걸음에 바로 다시 고른다 (--no-redecide 로 끔).
//     경로 끝이 목표 밖·경로 없음은 그 칸도 피하는 칸에 넣는다 (정책 환경은 규칙 추종기의 복구가 없어서)
//
// 출력: 장면마다 성공률·실패 종류·닫힘까지 시간·평균 속도·이동 거리 + 원인 분석용 걸음 통계. --out 을 주면 <접두사>_<장면>.tsv 에 판마다 한 줄
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <string>
#include <vector>

#include "actor_critic.h"
#include "env_api.h"
#include "frontier.h"
#include "frontier_explorer.h"
#include "ppo.h"

namespace {

struct Args {
  std::vector<std::string> scenes;
  int episodes = 256, people = 0;
  uint64_t seed = 1;
  std::string policy, out;
  std::string env = SIM_CONFIG_DIR "/env.yaml", model = SIM_CONFIG_DIR "/sensor_model.yaml", frontier = FRONTIER_CONFIG;
  bool eval_seed = false, avoid = true, redecide = true, stick = true;
  double rule_near = 0.0;   // > 0 이면 아래층 RL 판에서도 목표까지 직선 거리가 이 값 아래인 결정은 규칙 추종기에 맡김 (실험용)
  double stall_s = 0.0;   // > 0 이면 정체 판정 시간(env.yaml follow.stall_time)을 이 값으로 (실험용)
  int hold = 0;   // 아래층 RL: 목표 칸을 정한 뒤 이 걸음 수 동안은 주기 결정에서 안 바꿈 (0 = 끔 — 실험용)
};

std::string stem(const std::string& p) {
  const auto s = p.find_last_of('/');
  std::string b = s == std::string::npos ? p : p.substr(s + 1);
  const auto k = b.rfind('.');
  return k == std::string::npos ? b : b.substr(0, k);
}

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
  hc.cfg.curriculum = 0;
  hc.cfg.max_steps = (int)std::ceil(P.t_safe / (hc.cfg.substeps * hc.cfg.dt)) + 100;
  hc.cfg.low_stat = hc.cfg.low_task = hc.cfg.low_rl = 1;
  hc.cfg.low_auto = 0;      // 목표는 호스트(위층 규칙)가 — Env::low_prepare
  hc.cfg.low_act_w = 0;     // 행동 줄 폭 mdp::act::DIM — 규칙 추종 환경이 후보 번호를 읽음
  if (a.stall_s > 0.0) hc.cfg.low_idle_steps = std::max(1, (int)std::lround(a.stall_s / (hc.cfg.substeps * hc.cfg.dt)));
  hc.cfg.low_explore = 1;   // 합친 탐사: 탐사 판 끝 규칙 그대로 · 정체는 막힘 신호만
  const sim::LidarModel lidar = sim::load_lidar_model(a.model);
  const sim::EnvCfg& c = hc.cfg;
  const int N = a.episodes, AW = mdp::act::DIM, PW = mdp::low::ACT, K = mdp::hier::K, DS = c.decide_steps;
  const double dt = (double)c.substeps * c.dt;
  const bool use_pol = !a.policy.empty();
  const float bl_r = c.rec_bl_r > 0.f ? c.rec_bl_r : 0.5f;
  const int bl_steps = c.rec_bl_steps > 0 ? c.rec_bl_steps : (int)std::lround(10.0 / dt);
  std::unique_ptr<xppo::PPO> ppo;
  if (use_pol) {
    xppo::PPOConfig pc;
    pc.num_envs = N;
    ppo.reset(new xppo::PPO(pc, xnet::explore_spec_low(), 1));
    ppo->load(a.policy);
    std::printf("아래층: RL 정책 %s (반복 %d) — 결정적(평균 행동)\n", a.policy.c_str(), ppo->iteration());
  } else {
    std::printf("아래층: 규칙 추종기 (같은 걸음 루프 — 대조)\n");
  }
  std::printf("위층: 0 번 후보 규칙 · 결정 %d 걸음마다 · 막힘 신호 피하기 %s (반경 %.2f m · %.0f s) · 바로 다시 고름 %s · 목표 이어 가기 %s · 목표 붙들기 %d 걸음 | 정체 = %.0f s 동안 %.2f m · 통과 반경 %.2f m · 판 상한 %.0f s · 사람 %d\n",
              DS, a.avoid ? "켬" : "끔", bl_r, bl_steps * dt, a.redecide ? "켬" : "끔", a.stick ? "켬" : "끔", a.hold, c.low_idle_steps * dt, c.low_idle_d, c.low_pass_r, c.max_steps * dt, a.people);
  float *d_act, *d_pol;
  cudaMalloc(&d_act, sizeof(float) * AW * N);
  cudaMalloc(&d_pol, sizeof(float) * PW * N);
  for (size_t wi = 0; wi < a.scenes.size(); ++wi) {
    std::vector<sim::World> one{sim::build_world(stem(a.scenes[wi]), sim::load_scene(a.scenes[wi]), hc)};
    const sim::World& W = one[0];
    const uint64_t seed = a.eval_seed ? a.seed * 7919ull + 17 + wi : a.seed * 1000003ull + wi * 7919ull;
    sim::Env env(hc, one, lidar, N, seed, true);
    env.reset();
    env.sync();
    if (a.people > 0) {   // frontier_run 과 같은 차례 (people 0 이면 안 건드림)
      env.set_people_max(std::vector<int>(N, a.people));
      env.reset();
      env.sync();
    }
    std::vector<float> act((size_t)AW * N, 0.f), pol((size_t)PW * N, 0.f), rew;
    std::vector<sim::LowCmd> cmd(N, sim::LowCmd{-1, 0});
    std::vector<int32_t> cand, cur, done;
    std::vector<sim::EnvState> st;
    std::vector<sim::EpInfo> ep, first(N);
    std::vector<sim::LowEp> low;
    std::vector<sim::LowRl> rl;
    std::vector<Avoid> av(N);
    std::vector<int> fin(N, 0), redo(N, 1), slot(N, -1), sig_prev(N, 0), nsteps(N, 0), last_dec(N, 0), goal_prev(N, -1), sw_step(N, 0), end_since(N, -1), end_safe(N, 0);
    std::vector<float> end_v(N, 0.f), end_d(N, 0.f), end_w(N, 0.f);
    constexpr int kH = 50;   // 끝 직전 5 s 의 자리·후진 (충돌 원인 분석)
    std::vector<float> hx((size_t)N * kH, 0.f), hy((size_t)N * kH, 0.f);
    std::vector<unsigned char> hb((size_t)N * kH, 0);
    std::vector<double> dist(N, 0.0), chord(N, 0.0), px(N), py(N), cx(N), cy(N);
    // 판마다 걸음 통계 (원인 분석)
    std::vector<int> n_dec(N, 0), n_sw(N, 0), n_stall(N, 0), n_reach(N, 0), n_off(N, 0), n_nopath(N, 0), n_safe(N, 0), n_back(N, 0), n_slow(N, 0), n_nogoal(N, 0), n_avoided(N, 0), n_early(N, 0), n_pathless(N, 0);
    std::vector<double> tgt_d(N, 0.0), w_abs(N, 0.0);
    std::vector<int> n_rule(N, 0);
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
    int left = N;
    for (long t = 0; left > 0 && t < (long)c.max_steps + 10 * DS; ++t) {
      env.get_cand(cand);
      env.get_cand_cur(cur);
      for (int i = 0; i < N; ++i) {
        if (fin[i]) { cmd[i] = sim::LowCmd{-1, 0}; act[(size_t)AW * i] = 0.f; continue; }   // 끝난 환경: 규칙 탐사로 흘려 둠 (안 셈)
        const sim::EnvState& s = st[i];
        if (!(s.step % DS == 0 || redo[i])) continue;
        if (s.step % DS != 0) n_early[i] += 1;
        const bool forced = redo[i] != 0 && n_dec[i] > 0;
        redo[i] = 0;
        last_dec[i] = s.step;
        int k = -1;
        for (int j = 0; j < K && k < 0; ++j) {
          const int cell = cand[(size_t)i * K + j];
          if (cell < 0) continue;
          if (avoided(i, cell, s.step)) { n_avoided[i] += 1; continue; }
          k = j;
        }
        n_dec[i] += 1;
        slot[i] = k;
        int tgt = k >= 0 ? cand[(size_t)i * K + k] : -1;
        if (use_pol) {
          const bool live = cmd[i].follow == 2 && cmd[i].target >= 0 && cmd[i].target == s.goal_cell && s.p_n > 0 && !avoided(i, cmd[i].target, s.step);
          if (a.hold > 0 && !forced && live && s.step - sw_step[i] < a.hold) tgt = cmd[i].target;
          else if (a.stick && k >= 0 && ((cur[i] >> k) & 1) && cmd[i].follow == 2 && cmd[i].target >= 0 && cmd[i].target == s.goal_cell && s.p_n > 0 && !avoided(i, cmd[i].target, s.step)) tgt = cmd[i].target;
          const double dn = tgt >= 0 ? std::hypot(W.ox + ((double)(tgt % W.gw) + 0.5) * c.res - (s.x + s.ex), W.oy + ((double)(tgt / W.gw) + 0.5) * c.res - (s.y + s.ey)) : 0.0;
          if (tgt >= 0 && dn < a.rule_near) {   // 가까운 목표: 규칙 추종기 (후보 번호 그대로)
            cmd[i] = sim::LowCmd{-1, 0};
            act[(size_t)AW * i] = (float)k;
            n_rule[i] += 1;
          } else if (tgt >= 0) {
            if (tgt != cmd[i].target || cmd[i].follow != 2) sw_step[i] = s.step;
            cmd[i] = sim::LowCmd{tgt, 2};
          } else {   // 후보 없음: 규칙 추종기의 "목표 없음"(제자리 회전)
            cmd[i] = sim::LowCmd{-1, 0};
            act[(size_t)AW * i] = -1.f;
          }
        } else {
          cmd[i] = sim::LowCmd{-1, 0};
          act[(size_t)AW * i] = (float)k;
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
      if (use_pol) {
        const sim::StepOut o = env.outputs();
        ppo->act_inference(o.low_img, o.low_1d, d_pol, N, -1);
        cudaDeviceSynchronize();
        cudaMemcpy(pol.data(), d_pol, sizeof(float) * pol.size(), cudaMemcpyDeviceToHost);
        for (int i = 0; i < N; ++i) {
          if (cmd[i].follow != 2) continue;
          act[(size_t)AW * i] = pol[(size_t)PW * i];
          act[(size_t)AW * i + 1] = pol[(size_t)PW * i + 1];
        }
        cudaMemcpy(d_act, act.data(), sizeof(float) * act.size(), cudaMemcpyHostToDevice);
      }
      env.step(d_act);
      env.sync();
      env.get_reward(rew, done);
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
          if (!use_pol) sw_step[i] = s.step;
          goal_prev[i] = s.goal_cell;
        }
        end_v[i] = s.v;
        end_w[i] = s.w;
        hx[(size_t)i * kH + nsteps[i] % kH] = s.x;
        hy[(size_t)i * kH + nsteps[i] % kH] = s.y;
        hb[(size_t)i * kH + nsteps[i] % kH] = s.cmd_v < 0.f;
        end_since[i] = s.step - sw_step[i];
        end_safe[i] = rl[i].safe != 0;
        end_d[i] = s.goal_cell >= 0 ? (float)std::hypot(W.ox + ((double)(s.goal_cell % W.gw) + 0.5) * c.res - s.x, W.oy + ((double)(s.goal_cell / W.gw) + 0.5) * c.res - s.y) : -1.f;
        n_safe[i] += rl[i].safe != 0;
        n_back[i] += s.cmd_v < 0.f;
        n_slow[i] += std::fabs(s.v) < 0.1f;
        w_abs[i] += std::fabs(s.w);
        if (use_pol && cmd[i].follow == 2 && (s.goal_cell != cmd[i].target || s.p_n <= 0)) n_pathless[i] += 1;
        // 막힘 신호 → 그 칸을 피하고 바로 다시 고름
        if (rl[i].stall && !sig_prev[i]) {
          n_stall[i] += 1;
          avoid_add(i, rl[i].stall_tgt, s.step);
          if (a.avoid) redo[i] = 1;
        }
        sig_prev[i] = rl[i].stall;
        if (use_pol && cmd[i].follow == 2) {
          const sim::LowEp& L = low[i];
          const bool reached = L.reach_step == s.step && L.reached > 0;
          const bool off = !reached && L.hold != 0;
          const bool nopath = !reached && !off && s.step > last_dec[i] && (s.goal_cell != cmd[i].target || s.p_n <= 0);
          if (reached) n_reach[i] += 1;
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
    double s_dec = 0, s_sw = 0, s_stall = 0, s_reach = 0, s_off = 0, s_nopath = 0, s_safe = 0, s_back = 0, s_slow = 0, s_nogoal = 0, s_avoided = 0, s_early = 0, s_steps = 0, s_tgt = 0, s_w = 0, s_pathless = 0, ep_stall = 0, s_rule = 0;
    std::FILE* f = nullptr;
    if (!a.out.empty()) {
      f = std::fopen((a.out + "_" + W.name + ".tsv").c_str(), "w");
      if (!f) throw std::runtime_error("--out 파일을 못 엶");
      std::fputs("env\tdone\tt\tcoverage\tdist_m\tchord_m\tdecisions\tswitches\tstalls\treached\toff\tnopath\tsafe_steps\tback_steps\tslow_steps\tnogoal_dec\tend_v\tend_steps_since_goal_change\tend_goal_dist\tend_safe\tend_w\tend_5s_span_m\tend_5s_back_steps\n", f);
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
      s_dec += n_dec[i]; s_sw += n_sw[i]; s_stall += n_stall[i]; s_reach += n_reach[i]; s_off += n_off[i]; s_nopath += n_nopath[i]; s_safe += n_safe[i];
      s_back += n_back[i]; s_slow += n_slow[i]; s_nogoal += n_nogoal[i]; s_avoided += n_avoided[i]; s_early += n_early[i]; s_steps += nsteps[i]; s_tgt += tgt_d[i]; s_w += w_abs[i];
      s_pathless += n_pathless[i];
      s_rule += n_rule[i];
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
        std::fprintf(f, "%d\t%d\t%.1f\t%.4f\t%.2f\t%.2f\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%.2f\t%d\t%.2f\t%d\t%.2f\t%.2f\t%d\n", i, dd, ti, fin[i] ? first[i].coverage : 0.f, dist[i], chord[i], n_dec[i], n_sw[i], n_stall[i], n_reach[i], n_off[i],
                     n_nopath[i], n_safe[i], n_back[i], n_slow[i], n_nogoal[i], end_v[i], end_since[i], end_d[i], end_safe[i], end_w[i], span, nb);
    }
    if (f) std::fclose(f);
    const char* tag = use_pol ? "rl" : "rule";
    std::printf("[%s %s] 판 %d: 성공 %.3f · 벽 충돌 %.3f · 사람 충돌 %.3f · 진행 없음 %.3f · 끊김 %.3f | 닫힘까지 %.1f s (성공 판 평균, 정답 덮음 %.3f) · 판 평균 길이 %.1f s\n", tag, W.name.c_str(), N, succ / n,
                cw / n, cp / n, idle / n, trunc / n, succ > 0 ? ts / succ : 0.0, succ > 0 ? cov_s / succ : 0.0, t_all / n);
    std::printf("[%s %s] 평균 속도 %.3f m/s (걸음 거리 합 / 시간) · %.3f m/s (1 s 사이 직선 거리 합 / 시간 — frontier_run 과 같은 셈) · 이동 거리 모든 판 %.1f m · 성공 판 %.1f m\n", tag, W.name.c_str(),
                t_all > 0 ? d_all / t_all : 0.0, t_all > 0 ? ch_all / t_all : 0.0, d_all / n, succ > 0 ? d_succ / succ : 0.0);
    const double nd = std::max(1.0, s_dec), ng = std::max(1.0, s_dec - s_nogoal), nst = std::max(1.0, s_steps);
    std::printf("[%s %s] 위층: 판당 결정 %.1f 개 (그중 10 걸음 주기 밖 %.3f) · 목표 없는 결정 %.3f · 규칙 추종기에 맡긴 결정 %.3f · 목표 칸 바뀜 판당 %.1f 번 (%.1f s 에 한 번) · 피하는 칸이라 건너뛴 후보 %.0f 개 | 결정 때 목표까지 직선 거리 평균 %.2f m — "
                "1 m 안 %.3f · 1~3 m %.3f · 3~15 m %.3f · 15 m 밖 %.3f\n", tag, W.name.c_str(), s_dec / n, s_early / nd, s_nogoal / nd, s_rule / nd, s_sw / n, s_sw > 0 ? t_all / s_sw : 0.0, s_avoided, s_tgt / ng,
                dbin[0] / ng, dbin[1] / ng, dbin[2] / ng, dbin[3] / ng);
    std::printf("[%s %s] 아래층: 막힘 신호 %.0f 번 (선 판 비율 %.3f) · 통과 %.0f · 경로 끝이 목표 밖 %.0f · 경로 없음 %.0f | 걸음 비율 — 안전층이 고침 %.3f · 뒤로 %.3f · 0.1 m/s 아래 %.3f · 경로 없이 몬 걸음 %.3f · 평균 |ω| %.2f rad/s\n",
                tag, W.name.c_str(), s_stall, ep_stall / n, s_reach, s_off, s_nopath, s_safe / nst, s_back / nst, s_slow / nst, s_pathless / nst, s_w / nst);
    std::fflush(stdout);
  }
  cudaFree(d_act);
  cudaFree(d_pol);
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
      else if (k == "--policy") a.policy = nx();
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
      else if (k == "--hold") a.hold = std::stoi(nx());
      else if (k == "--stall-s") a.stall_s = std::stod(nx());
      else if (k == "--rule-near") a.rule_near = std::stod(nx());
      else { std::fprintf(stderr, "모르는 인자 %s\n", k.c_str()); return 2; }
    } catch (const std::exception& e) {
      std::fprintf(stderr, "combine_run: %s\n", e.what());
      return 2;
    }
  }
  if (a.scenes.empty()) {
    std::fprintf(stderr, "combine_run --scenes a.scene [b.scene …] [--policy <아래층 체크포인트>] [--episodes 256] [--seed 1] [--people 0] [--eval-seed] [--out <접두사>] [--no-avoid] [--no-redecide] [--no-stick]\n");
    return 2;
  }
  try {
    return run(a);
  } catch (const std::exception& e) {
    std::fprintf(stderr, "combine_run: %s\n", e.what());
    return 1;
  }
}
