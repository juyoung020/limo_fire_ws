// hybrid_eval — 재학습 없이 "RL 정책 + 프론티어 감시자" 하이브리드를 시험 장면에서 잰다 (평가 전용, 학습 안 함).
// 시뮬·판 끝·성공 판정은 학습 러너 평가(train.cpp evaluate)와 같다: 같은 env.yaml(커리큘럼 끔), 장면 i 의 씨앗 seed · 7919 + 17 + i,
// 환경 --episodes 개, 환경마다 첫 판. --policy rl 은 train --eval-ckpt 와 같은 판·같은 행동(결정 = 평균, --sample = 같은 표본 번호).
//
//   hybrid_eval --ckpt model.bin --scenes bookstore.scene small-house.scene [--episodes 64] [--seed 1] [--sample]
//               [--policy rl|frontier|hybrid] [--T 15] [--small 1] [--hold-fr 5] [--hold-rl 2] [--cooldown 20] [--fmode fast]
//               [--threads 4] [--out eps.jsonl] [--env sim/config/env.yaml] [--model sim/config/sensor_model.yaml] [--frontier frontier.yaml]
//
// 하이브리드(감시자) — 환경마다 조종자 RL / 프론티어(baselines/frontier Explorer, --fmode 판)를 바꾼다:
//   RL → 프론티어:  (a) 지도 진척 시계(env.h since_occ = 판 끝 진행 없음 시계) ≥ T s  (T ≤ 0 이면 안 씀)
//                   (b) --small 1: 1 s 마다 로봇 지도에서 mdp::frontier_search(설 수 있는 칸, 로봇 반경 0.2 m) — 도달 가능한 프론티어 칸이 있는데
//                       덩어리가 전부 0.75 m(관측 요약 kFrontMinSize, explore_lite min_frontier_size) 미만
//                   RL 로 돌아온 뒤 --hold-rl s 동안은 안 넘김. 프론티어가 포기(all_blacklisted 등)하고 돌려준 뒤엔 --cooldown s 동안 (b) 안 씀.
//   프론티어 → RL: 넘긴 뒤 --hold-fr s 지났고, 지도 진척이 다시 생겼고(since_occ < 1 s), 1 s 마다 보는 찾기에서 0.75 m 이상 덩어리가 있음.
//                   Explorer 가 지도 닫힘(closed)으로 끝내면 멈춰 시뮬 성공(굵은 격자 닫힘 + 시계 10 s)을 기다림, 다른 까닭으로 끝내면 RL 로.
//   넘길 때마다 Explorer 를 새로 만든다(속도 smoother·경로·복구 상태가 옛 것이 아니게).
//   프론티어 행동 (v, ω) 물리 단위 → 정규 단위(v / V_FWD 또는 V_BACK, ω / act_w_max). 시뮬의 바퀴 자르기·앞 안전층·(켜면) 쓸림 안전층은 그대로 거친다.
// --policy frontier: 처음부터 끝까지 프론티어 (RL 환경 설정에서의 기준선 — frontier_run 은 물리 단위·안전층 끔·안전 시간 1800 s 라 조금 다름).
//
// 출력: 판마다 한 줄 JSON(--out), 장면마다 요약(stderr). 판 시간은 시뮬 EpInfo.t, 프론티어 몫 = 첫 판 동안 프론티어가 조종한 걸음 비율.
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include "curriculums.h"
#include "env_api.h"
#include "frontier.h"
#include "frontier_explorer.h"
#include "observations.h"
#include "ppo.h"

namespace {

struct Args {
  std::string ckpt, policy = "hybrid", fmode = "fast", out;
  std::vector<std::string> scenes;
  int episodes = 64, threads = 4, small = 1;
  uint64_t seed = 1;
  bool sample = false;
  float T = 15.f, hold_fr = 5.f, hold_rl = 2.f, cooldown = 20.f;
  std::string env = SIM_CONFIG_DIR "/env.yaml", model = SIM_CONFIG_DIR "/sensor_model.yaml", frontier = FRONTIER_CONFIG;
};

std::string stem(const std::string& p) {
  std::string b = p.substr(p.find_last_of('/') + 1);
  const auto k = b.rfind('.');
  return k == std::string::npos ? b : b.substr(0, k);
}

const char* outcome_of(int32_t d) {
  switch (d) {
    case mdp::cur::kDeclareOk: return "success";
    case mdp::cur::kCollision:
    case mdp::cur::kCollisionPerson: return "collision";
    case mdp::cur::kNoProgress: return "no_progress";
    case mdp::cur::kTruncated: return "truncated";
    default: return "other";
  }
}

enum Ctrl { kRL = 0, kFR = 1 };

// 환경 하나의 감시자 상태
struct Sup {
  bool live = true;            // 첫 판 진행 중
  int ctrl = kRL;
  double t_switch = -1e9;      // 마지막 조종자 바뀐 시각
  double t_cool = -1e9;        // 이 시각까지 (b) 안 씀
  bool fr_wait = false;        // Explorer 가 닫힘으로 끝냄 → 멈춰 기다림
  int big = -1;                // 마지막 찾기: 0.75 m 이상 덩어리 수 (−1 = 아직)
  int cells = 0;               // 마지막 찾기: 도달 가능한 프론티어 칸 수
  long fr_steps = 0, steps = 0;
  int n_to_fr = 0, n_clock = 0, n_small = 0, n_giveup = 0, ctrl_end = 0;
  float fv = 0, fw = 0;        // 이번 걸음 프론티어 행동 (물리 단위)
  std::unique_ptr<fb::Explorer> ex;
  // 찾기 버퍼
  std::vector<uint8_t> flag;
  std::vector<int32_t> qa, qb;
  std::vector<mdp::Frontier> fr;
  std::vector<int16_t> L;      // 이 환경 지도 (stride = cfg.size)
  // 결과
  int32_t done = 0;
  float t = 0, coverage = 0;
};

int run_scene(const Args& a, size_t ti, const std::string& scene, xppo::PPO& ppo, FILE* fo) {
  sim::EnvHostCfg hc = sim::load_env_cfg(a.env);
  hc.cfg.curriculum = 0;   // train.cpp evaluate 와 같음
  const sim::EnvCfg& c = hc.cfg;
  if (c.action_mode != sim::kActVelocity || c.act_raw) throw std::runtime_error("env.yaml: 속도 행동·정규 단위여야 함");
  const sim::LidarModel lidar = sim::load_lidar_model(a.model);
  std::vector<sim::World> one{sim::build_world(stem(scene), sim::load_scene(scene), hc)};
  const sim::World& W = one[0];
  const int E = a.episodes, S = c.size;
  const float dt = c.substeps * c.dt;
  sim::Env env(hc, one, lidar, E, a.seed * 7919ull + 17 + ti, true);
  env.reset();
  env.sync();
  const bool use_rl = a.policy != "frontier", use_fr = a.policy != "rl";
  fb::Params P{};
  if (use_fr) {
    P = fb::load_params(a.frontier, a.env, a.fmode);
    if (std::fabs(P.ctrl_dt - dt) > 1e-6f) throw std::runtime_error("프론티어 ctrl_dt 와 시뮬 행동 간격이 다름");
  }
  const mdp::Robot rb = mdp::obsgen::reach_robot(c);
  const int min_cells = mdp::obsgen::reach_min_cells(c);
  std::vector<Sup> sup(E);
  for (auto& s : sup) {
    if (!use_fr) continue;
    const size_t n = (size_t)W.gw * W.gh;
    s.flag.resize(n); s.qa.resize(n); s.qb.resize(n); s.fr.resize(n / 4 + 16);
    s.L.resize((size_t)S * S);
    if (a.policy == "frontier") { s.ctrl = kFR; s.t_switch = 0; s.ex.reset(new fb::Explorer(P, W.gw, W.gh, S, W.ox, W.oy)); s.n_to_fr = 1; }
  }
  float* d_act;
  cudaMalloc(&d_act, sizeof(float) * mdp::act::DIM * E);
  cudaMemset(d_act, 0, sizeof(float) * mdp::act::DIM * E);
  std::vector<float> act((size_t)mdp::act::DIM * E), reward;
  std::vector<int32_t> done;
  std::vector<sim::EnvState> st;
  std::vector<sim::EpInfo> ep;
  const int16_t* d_lmap = env.batch().lmap;
  std::vector<int> need_map(E), need_search(E);
  const int NT = (int)a.scenes.size();
  int n_live = E;
  const int trace = std::getenv("HYB_TRACE") ? std::atoi(std::getenv("HYB_TRACE")) : -1;
  auto par_for = [&](auto&& fn) {
    const int nt = std::max(1, a.threads);
    std::vector<std::thread> th;
    for (int k = 0; k < nt; ++k) th.emplace_back([&, k]() { for (int i = k; i < E; i += nt) fn(i); });
    for (auto& t : th) t.join();
  };
  for (int step = 0; step < c.max_steps + 10 && n_live > 0; ++step) {
    if (use_rl) {
      const sim::StepOut o = env.outputs();
      ppo.act_inference(o.obs_img, o.obs_1d, d_act, E, a.sample ? (int64_t)step * (int64_t)NT + (int64_t)ti : -1);
      cudaDeviceSynchronize();   // PPO 스트림 → 시뮬(기본 스트림) 차례
    }
    if (use_fr) {
      cudaDeviceSynchronize();
      env.get_state(st);
      if (use_rl) cudaMemcpy(act.data(), d_act, sizeof(float) * act.size(), cudaMemcpyDeviceToHost);
      else std::fill(act.begin(), act.end(), 0.f);
      // 1) 지도가 필요한 환경: 프론티어 조종 중이거나 1 s 찾기 차례
      const bool tick = step % 10 == 0;
      for (int i = 0; i < E; ++i) {
        Sup& s = sup[i];
        need_map[i] = need_search[i] = 0;
        if (!s.live) continue;
        need_search[i] = tick && (a.small || s.ctrl == kFR);
        need_map[i] = s.ctrl == kFR || need_search[i];
        if (need_map[i]) cudaMemcpy(s.L.data(), d_lmap + (size_t)i * S * S, sizeof(int16_t) * S * S, cudaMemcpyDeviceToHost);
      }
      // 2) 환경마다 감시자 (서로 독립 — 차례와 무관하게 같은 결과)
      par_for([&](int i) {
        Sup& s = sup[i];
        if (!s.live) return;
        const sim::EnvState& e = st[i];
        const double t = (double)e.step * dt;
        const float px = e.x + e.ex, py = e.y + e.ey, pth = e.th + e.eth;
        const mdp::MapView mv{s.L.data(), S, W.gw, W.gh};
        if (need_search[i]) {
          const float fx = (px - W.ox) / c.res, fy = (py - W.oy) / c.res;
          const int rx = (int)std::floor(fx), ry = (int)std::floor(fy);
          const mdp::SearchResult r = mdp::frontier_search(mv, rx, ry, rb, min_cells, mdp::SearchBuf{s.flag.data(), s.qa.data(), s.qb.data()},
                                                           s.fr.data(), (int)s.fr.size());
          s.big = r.n_found;
          s.cells = r.n_cells;
        }
        const double idle = (double)e.since_occ * dt;
        if (a.policy == "hybrid") {
          if (s.ctrl == kRL) {
            const bool clock = a.T > 0 && idle >= a.T;
            const bool small = a.small && need_search[i] && t >= s.t_cool && s.cells > 0 && s.big == 0;
            if (t - s.t_switch >= a.hold_rl && (clock || small)) {
              s.ctrl = kFR;
              s.t_switch = t;
              s.fr_wait = false;
              s.ex.reset(new fb::Explorer(P, W.gw, W.gh, S, W.ox, W.oy));
              s.n_to_fr++;
              s.n_clock += clock;
              s.n_small += !clock && small;
              if (!need_map[i]) cudaMemcpy(s.L.data(), d_lmap + (size_t)i * S * S, sizeof(int16_t) * S * S, cudaMemcpyDeviceToHost);
            }
          } else if (!s.fr_wait && t - s.t_switch >= a.hold_fr && idle < 1.0 && need_search[i] && s.big > 0) {
            s.ctrl = kRL;
            s.t_switch = t;
            s.ex.reset();
          }
        }
        if (s.ctrl == kFR) {
          s.fv = s.fw = 0;
          if (!s.fr_wait) {
            const fb::Action ac = s.ex->step(s.L.data(), px, py, pth, t);
            if (s.ex->phase() == fb::Phase::Done) {
              if (s.ex->end_reason() == "closed" || a.policy == "frontier") s.fr_wait = true;   // 멈춰 시뮬 판정 기다림 (기준선 단독이면 포기도 멈춤)
              else if (a.policy == "hybrid") {   // 포기 → RL 로 돌려줌
                s.ctrl = kRL;
                s.t_switch = t;
                s.t_cool = t + a.cooldown;
                s.n_giveup++;
                s.ex.reset();
              }
            } else {
              s.fv = ac.v;
              s.fw = ac.w;
            }
          }
        }
        if (s.ctrl == kFR) {
          float* q = act.data() + (size_t)mdp::act::DIM * i;
          q[0] = s.fv >= 0.f ? s.fv / c.act_v_fwd : s.fv / c.act_v_back;
          q[1] = 0.f;
          q[2] = s.fw / c.act_w_max;
          q[3] = 0.f;
          s.fr_steps++;
        }
        s.steps++;
        if (trace == i && need_search[i])   // HYB_TRACE=환경 번호: 1 s 마다 감시자 상태
          std::fprintf(stderr, "  [trace %s env %d] t %.0f 시계 %.1f 조종 %s 큰 덩어리 %d 프론티어 칸 %d 기다림 %d 행동 (%.2f, %.2f) %s\n", W.name.c_str(), i, t, idle,
                       s.ctrl == kFR ? "FR" : "RL", s.big, s.cells, (int)s.fr_wait, s.fv, s.fw, s.ex ? s.ex->end_reason().c_str() : "");
      });
      cudaMemcpy(d_act, act.data(), sizeof(float) * act.size(), cudaMemcpyHostToDevice);
    }
    env.step(d_act);
    env.sync();
    env.get_reward(reward, done);
    bool any = false;
    for (int i = 0; i < E; ++i) any = any || (sup[i].live && done[i]);
    if (!use_fr) for (int i = 0; i < E; ++i) if (sup[i].live) sup[i].steps++;
    if (any) {
      env.get_ep(ep);
      for (int i = 0; i < E; ++i) {
        Sup& s = sup[i];
        if (!s.live || !done[i]) continue;
        s.live = false;
        --n_live;
        s.done = done[i];
        s.t = ep[i].t;
        s.coverage = ep[i].coverage;
        s.ctrl_end = s.ctrl;
        s.ex.reset();
      }
    }
  }
  cudaFree(d_act);
  // 끝까지 안 끝난 판 = 끊김
  double n = E, succ = 0, coll = 0, idle = 0, trunc = 0, ts = 0, frs = 0, stp = 0;
  std::vector<float> tsv;
  for (int i = 0; i < E; ++i) {
    Sup& s = sup[i];
    if (s.live) { s.done = mdp::cur::kTruncated; s.t = (float)s.steps * dt; }
    const char* oc = outcome_of(s.done);
    succ += s.done == mdp::cur::kDeclareOk;
    coll += s.done == mdp::cur::kCollision || s.done == mdp::cur::kCollisionPerson;
    idle += s.done == mdp::cur::kNoProgress;
    trunc += s.done == mdp::cur::kTruncated || s.live;
    if (s.done == mdp::cur::kDeclareOk) { ts += s.t; tsv.push_back(s.t); }
    frs += s.fr_steps;
    stp += s.steps;
    if (fo)
      std::fprintf(fo,
                   "{\"scene\":\"%s\",\"env\":%d,\"policy\":\"%s\",\"sample\":%d,\"T\":%g,\"small\":%d,\"sweep\":%d,\"outcome\":\"%s\",\"done\":%d,"
                   "\"t\":%.3f,\"coverage\":%.4f,\"steps\":%ld,\"fr_steps\":%ld,\"to_fr\":%d,\"by_clock\":%d,\"by_small\":%d,\"giveup\":%d,\"ctrl_end\":\"%s\"}\n",
                   W.name.c_str(), i, a.policy.c_str(), (int)a.sample, a.T, a.small, c.sweep_guard, oc, s.done, s.t, s.coverage, s.steps, s.fr_steps,
                   s.n_to_fr, s.n_clock, s.n_small, s.n_giveup, s.ctrl_end == kFR ? "fr" : "rl");
  }
  if (fo) std::fflush(fo);
  std::sort(tsv.begin(), tsv.end());
  const double med = tsv.empty() ? 0 : (tsv.size() % 2 ? tsv[tsv.size() / 2] : 0.5 * (tsv[tsv.size() / 2 - 1] + tsv[tsv.size() / 2]));
  std::fprintf(stderr, "%s [%s%s T %g small %d sweep %d]: 성공 %.0f/%d 충돌 %.0f 정체 %.0f 끊김 %.0f, 성공 시간 평균 %.1f 중앙 %.1f s, 프론티어 몫 %.3f\n",
               W.name.c_str(), a.policy.c_str(), a.sample ? " 표본" : "", a.T, a.small, c.sweep_guard, succ, E, coll, idle, trunc,
               succ > 0 ? ts / succ : 0.0, med, stp > 0 ? frs / stp : 0.0);
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  Args a;
  for (int i = 1; i < argc; ++i) {
    const std::string k = argv[i];
    auto nx = [&]() -> std::string {
      if (i + 1 >= argc) throw std::runtime_error(k + " 뒤에 값");
      return argv[++i];
    };
    if (k == "--scenes") { while (i + 1 < argc && argv[i + 1][0] != '-') a.scenes.push_back(argv[++i]); }
    else if (k == "--ckpt") a.ckpt = nx();
    else if (k == "--policy") a.policy = nx();
    else if (k == "--fmode") a.fmode = nx();
    else if (k == "--episodes") a.episodes = std::stoi(nx());
    else if (k == "--seed") a.seed = std::stoull(nx());
    else if (k == "--sample") a.sample = true;
    else if (k == "--T") a.T = std::stof(nx());
    else if (k == "--small") a.small = std::stoi(nx());
    else if (k == "--hold-fr") a.hold_fr = std::stof(nx());
    else if (k == "--hold-rl") a.hold_rl = std::stof(nx());
    else if (k == "--cooldown") a.cooldown = std::stof(nx());
    else if (k == "--threads") a.threads = std::stoi(nx());
    else if (k == "--out") a.out = nx();
    else if (k == "--env") a.env = nx();
    else if (k == "--model") a.model = nx();
    else if (k == "--frontier") a.frontier = nx();
    else { std::fprintf(stderr, "모르는 인자 %s\n", k.c_str()); return 2; }
  }
  if (a.scenes.empty() || (a.policy != "frontier" && a.ckpt.empty()) || (a.policy != "rl" && a.policy != "frontier" && a.policy != "hybrid")) {
    std::fprintf(stderr, "hybrid_eval --ckpt model.bin --scenes a.scene [b.scene …] [--policy rl|frontier|hybrid] [--episodes 64] [--sample] [--T 15] [--small 1] …\n");
    return 2;
  }
  try {
    xppo::PPOConfig pc;
    pc.num_envs = a.episodes;
    xppo::PPO ppo(pc, xnet::explore_spec(), a.seed);
    if (a.policy != "frontier") {
      ppo.load(a.ckpt);
      std::fprintf(stderr, "체크포인트 %s 반복 %d\n", a.ckpt.c_str(), ppo.iteration());
    }
    FILE* fo = a.out.empty() ? nullptr : std::fopen(a.out.c_str(), "a");
    for (size_t ti = 0; ti < a.scenes.size(); ++ti) run_scene(a, ti, a.scenes[ti], ppo, fo);
    if (fo) std::fclose(fo);
  } catch (const std::exception& e) {
    std::fprintf(stderr, "hybrid_eval: %s\n", e.what());
    return 1;
  }
  return 0;
}
