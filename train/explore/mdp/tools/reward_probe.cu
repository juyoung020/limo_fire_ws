// 보상 해킹 시험 (mdp/README "보상 항", archive/explore_roadmap_1104 ROADMAP S4): 정해진 행동으로 판 하나씩 끝까지 돌려 판 합 보상·끝 코드·항마다 합을 잰다.
//   still      : 제자리(행동 0) → 안전 시간 끊김까지. 기대: 판 합 < 0
//   spin       : 제자리 돌기(ω = 0.5 정규 = 1.57 rad/s) → 끊김까지. 참고(해킹 아님 — 돌기도 탐사의 일부)
//   declare0   : 첫 걸음에 선언. 옛 규칙: done 4(선언 실패)·판 합 < 0. 닫힘 성공 규칙(기본): 선언은 무시 — done 4 가 없어야 함
//   crash      : 안전층 끄고 앞으로 최고 속도 직진 → 벽에 박음. 기대: 판 합 < frontier 판 합(정상 선언)
//   crash_safe : 안전층 켜고 같은 직진 → 안전층이 막아 박지 않는지(충돌 수)
//   frontier   : 프론티어 기준선(baselines/frontier, --mode)이 모는 대로(물리 → 정규 단위), 기준선이 끝나면 그 걸음에 선언. 기대: 판 합 > 0
// 닫힘 성공 규칙(sim EnvCfg::success_closed)에서는 덧붙여: 지도 닫힘으로 성공한 판의 정답 덮음(이 도구 안에서만 쓰는 확인값)이 0.8 아래인 판 = 0
// (거짓 닫힘), 제자리·제자리 돌기의 덮음 항(로봇 지도 처음 막힘 칸 보상) < 기준선 따라가기의 1/4 (잡음·자세 흐름으로 칸을 키우지 못함)
// 속도 행동 모드의 촘촘한 항(rewards.h "직접 제어 촘촘한 항": approach·front_cells·proximity·smooth)을 노리는 행동 — 기대: 넷 다 판 합 평균 < frontier
//   osc        : 앞뒤 왕복 (앞 0.3 m/s 1 s ↔ 뒤 0.3 m/s 1 s) — approach 를 왕복으로 못 범
//   hug        : 벽 붙어 다니기 (안전층 끔. 근접 항이 0 이면 앞으로 0.2 m/s, 근접 항이 나오면 벽 곁에서 옆으로 0.3 m/s 1 s 씩 왕복)
//   dither     : 걸음마다 (v, vy, ω) 를 +1 ↔ −1 로 뒤집기 (가장 큰 행동 변화)
//   spin 도 같은 검사에 넣는다
// --hier: 계층형 행동(결정 1 s = 속도 행동 10 걸음, 보상 = 결정 동안 합)에서 결정 단위로 — 정책은 있는 후보만 고를 수 있으므로:
//   h_spin   : 빈 후보 번호(−1) → 환경이 제자리 회전만 (계층형의 "제자리"). 기대: 모든 판 진행 없음·판 합 < 0
//   h_slot0  : 늘 0 번(가장 가까운) 후보. 기대: 판 합 평균 > 0, > h_spin
//   h_far    : 늘 마지막(가장 먼) 있는 후보 · h_random: 있는 후보 무작위 (참고)
//   h_toggle : 결정마다 0 번 ↔ 마지막 후보 (왕복 진동). 기대: 판 합 평균 ≤ h_slot0 (진동으로 보상을 더 얻지 못함)
//   일부러 박기는 계층형에선 정책이 할 수 없음(추종기가 몰고 충돌 검사·안전층) — h_* 들의 충돌 수를 참고로 찍음
// reward_probe <장면.scene> [장면 …] [--envs 8] [--mode fast] [--only 이름,…]
// 모두 GPU 환경(학습과 같은 커널). 커리큘럼 끔(맵 하나씩·오차 배율 1).
#include <cmath>
#include <cstdio>
#include <cstring>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#include "env_api.h"
#include "frontier_explorer.h"

namespace {

struct Result {
  std::vector<double> ret;
  std::vector<int> done, steps;
  std::vector<float> cov;
  double terms[mdp::rew::kTerms] = {};
  std::vector<long> sub;                       // 계층형: 판 동안 더한 시뮬 걸음 수 (시간 항으로 셈)
  long n_dec = 0, bad_sum = 0, bad_steps = 0, bad_len = 0;   // 행동(결정) 수 · 보상 ≠ 항 합 · 결정 걸음 수 어긋남 · 판 길이 어긋남
};

double mean(const std::vector<double>& v) {
  double s = 0;
  for (double x : v) s += x;
  return v.empty() ? 0 : s / v.size();
}

}  // namespace

int main(int argc, char** argv) {
  std::vector<std::string> scenes;
  int n_env = 8;
  std::string mode = "fast", only;
  bool hier = false;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    if (a == "--envs" && i + 1 < argc) n_env = std::stoi(argv[++i]);
    else if (a == "--mode" && i + 1 < argc) mode = argv[++i];
    else if (a == "--only" && i + 1 < argc) only = argv[++i];
    else if (a == "--hier") hier = true;
    else scenes.push_back(a);
  }
  if (scenes.empty()) { std::fprintf(stderr, "reward_probe <장면.scene> [장면 …] [--envs 8] [--mode fast] [--only still,…]\n"); return 2; }
  sim::EnvHostCfg hc0 = sim::load_env_cfg(SIM_CONFIG_DIR "/env.yaml");
  hc0.cfg.curriculum = 0;
  const sim::LidarModel lidar = sim::load_lidar_model(SIM_CONFIG_DIR "/sensor_model.yaml");
  const fb::Params P = fb::load_params(FRONTIER_CONFIG, SIM_CONFIG_DIR "/env.yaml", mode);
  if (hier) hc0.cfg.action_mode = sim::kActPath;
  const std::vector<std::string> names = hier ? std::vector<std::string>{"h_spin", "h_slot0", "h_far", "h_toggle", "h_random"}
                                              : std::vector<std::string>{"still", "spin", "osc", "hug", "dither", "declare0", "crash", "crash_safe", "frontier"};
  const int dec = hier ? hc0.cfg.decide_steps : 1;   // 행동 하나 = 시뮬 걸음 dec 개
  constexpr int A = mdp::act::DIM, K = mdp::rew::kTerms;
  bool all_ok = true;
  std::printf("가중치: 덮음 %.1f · 시간 %.2f/T_ref(초당) · 벽 충돌 %.1f · 선언 성공 %+.1f / 실패 %.1f~%.1f · 헛돎 %.2f/rad · 안전 시간 %.0f s\n",
              mdp::rew::kCov, mdp::rew::kTime, mdp::rew::kCollWall, mdp::rew::kDeclareSuccess, mdp::rew::kDeclareFail, mdp::rew::kDeclareNear,
              mdp::rew::kSlip, mdp::cur::kSafeTime);
  for (const auto& path : scenes) {
    const sim::World W = sim::build_world(path, sim::load_scene(path), hc0);
    std::printf("== 맵 %s: 벽 면 칸 %d, T_ref %.0f s, 환경 %d\n", path.c_str(), W.n_wall, mdp::rew::t_ref(W.n_wall), n_env);
    std::map<std::string, Result> res;
    for (const auto& nm : names) {
      if (!only.empty() && ("," + only + ",").find("," + nm + ",") == std::string::npos) continue;
      sim::EnvHostCfg hc = hc0;
      if (nm == "crash" || nm == "frontier" || nm == "hug") hc.cfg.safety = 0;   // 기준선은 Nav2 감속을 스스로 함
      const std::vector<sim::World> one{W};
      sim::Env env(hc, one, lidar, n_env, 11, true);
      env.reset();
      env.sync();
      const sim::StepOut out = env.outputs();
      float* d_act;
      cudaMalloc(&d_act, sizeof(float) * A * n_env);
      std::vector<float> act((size_t)A * n_env, 0.f), rew, terms;
      std::vector<int32_t> done;
      std::vector<sim::EnvState> st;
      std::vector<int16_t> L;
      std::vector<sim::EpInfo> ep;
      std::vector<fb::Explorer> ex;
      if (nm == "frontier")
        for (int i = 0; i < n_env; ++i) ex.emplace_back(P, W.gw, W.gh, hc.cfg.size, W.ox, W.oy);
      Result& R = res[nm];
      R.ret.assign(n_env, 0.0);
      R.done.assign(n_env, 0);
      R.steps.assign(n_env, 0);
      R.cov.assign(n_env, 0.f);
      R.sub.assign(n_env, 0);
      std::vector<char> active(n_env, 1), stopped(n_env, 0);
      int n_active = n_env;
      std::vector<uint8_t> h_img;
      std::vector<float> h_o1;
      uint64_t hrng = 12345;
      for (int t = 0; n_active > 0 && t <= hc.cfg.max_steps / dec + 1; ++t) {
        std::fill(act.begin(), act.end(), 0.f);
        if (hier) env.get_obs(h_img, h_o1);
        if (nm == "frontier") {
          env.get_state(st);
          env.get_map(L);
        }
        for (int i = 0; i < n_env; ++i) {
          float* a = &act[(size_t)A * i];
          if (!active[i]) continue;
          if (hier) {   // 후보 칸 (obs_layout.h mdp::hier): 있는 후보 번호들
            const float* O = h_o1.data() + (size_t)i * env.batch().o1d + mdp::obs::POLICY_1D;
            int nv = 0, valid[mdp::hier::K];
            for (int k = 0; k < mdp::hier::K; ++k)
              if (O[k * mdp::hier::FEAT + mdp::hier::F_VALID] > 0.5f) valid[nv++] = k;
            int s = -1;
            if (nm == "h_slot0") s = 0;
            else if (nm == "h_far" && nv > 0) s = valid[nv - 1];
            else if (nm == "h_toggle" && nv > 0) s = (t & 1) ? valid[nv - 1] : valid[0];
            else if (nm == "h_random" && nv > 0) s = valid[dm::splitmix64(hrng) % (uint64_t)nv];
            a[0] = (float)s;
          } else if (nm == "spin") a[2] = 0.5f;
          else if (nm == "osc") a[0] = (t / 10) % 2 == 0 ? 0.3f : -1.f;   // 앞 0.3 m/s (정규 0.3) ↔ 뒤 한도 0.3 m/s (정규 −1), 1 s 씩
          else if (nm == "hug") {   // 직전 걸음 근접 항으로 벽 곁인지 봄
            const bool near = !terms.empty() && terms[(size_t)K * i + mdp::rew::kTermProx] < 0.f;
            if (near) a[1] = (t / 10) % 2 == 0 ? 0.3f : -0.3f;
            else a[0] = 0.2f;
          } else if (nm == "dither") a[0] = a[1] = a[2] = (t & 1) ? -1.f : 1.f;
          else if (nm == "declare0") a[3] = 1.f;
          else if (nm == "crash" || nm == "crash_safe") a[0] = 1.f;
          else if (nm == "frontier") {
            const sim::EnvState& s = st[i];
            const fb::Action ac = ex[i].step(L.data() + (size_t)i * hc.cfg.size * hc.cfg.size, s.x + s.ex, s.y + s.ey, s.th + s.eth,
                                             (double)s.step * hc.cfg.substeps * hc.cfg.dt);
            if (ex[i].phase() == fb::Phase::Done) {
              if (ac.declare) sim::to_normalized(hc.cfg, 0.f, 0.f, 0.f, 1.f, a);
              else stopped[i] = 1;   // 기준선이 선언 없이 멈춤(안전 시간) → 끊김으로 셈
            } else {
              sim::to_normalized(hc.cfg, ac.v, 0.f, ac.w, 0.f, a);
            }
          }
        }
        cudaMemcpy(d_act, act.data(), sizeof(float) * A * n_env, cudaMemcpyHostToDevice);
        env.step(d_act);
        env.sync();
        env.get_reward(rew, done);
        env.get_terms(terms);
        env.get_ep(ep);
        for (int i = 0; i < n_env; ++i) {
          if (!active[i]) continue;
          R.ret[i] += rew[i];
          double tsum = 0;
          for (int k = 0; k < K; ++k) { R.terms[k] += terms[(size_t)K * i + k]; tsum += terms[(size_t)K * i + k]; }
          // 보상 = 항 합 (행동 하나 — 계층형이면 결정 하나의 묶음 합)
          R.n_dec += 1;
          if (std::fabs(tsum - rew[i]) > 1e-4 * std::max(1.0, std::fabs((double)rew[i]))) R.bad_sum += 1;
          if (hier) {   // 시간 항 / 걸음당 시간 항 = 이 결정에 더한 시뮬 걸음 수: 판이 안 끝났으면 decide_steps, 끝났으면 1 ~ decide_steps
            const double per = -(double)mdp::rew::kTime * hc.cfg.substeps * hc.cfg.dt / mdp::rew::t_ref(W.n_wall);
            const double ns = terms[(size_t)K * i + mdp::rew::kTermTime] / per;
            const int ni = (int)std::lround(ns);
            if (std::fabs(ns - ni) > 1e-3 || (done[i] == 0 ? ni != dec : (ni < 1 || ni > dec))) R.bad_steps += 1;
            R.sub[i] += ni;
            if (done[i] != 0 && std::fabs(R.sub[i] * hc.cfg.substeps * hc.cfg.dt - ep[i].t) > 1e-3) R.bad_len += 1;   // 판 길이 = 더한 걸음 수 × 0.1 s
          }
          if (done[i] != 0 || stopped[i]) {
            active[i] = 0;
            --n_active;
            R.done[i] = done[i] != 0 ? done[i] : mdp::cur::kTruncated;
            R.steps[i] = t + 1;
            R.cov[i] = done[i] != 0 ? ep[i].coverage : st[i].coverage;
          }
        }
      }
      cudaFree(d_act);
      (void)out;
      int codes[mdp::cur::kNumDone] = {};
      double cv = 0, len = 0;
      for (int i = 0; i < n_env; ++i) { codes[R.done[i]]++; cv += R.cov[i]; len += R.steps[i]; }
      double mn = 1e9, mx = -1e9;
      for (double r : R.ret) { mn = std::min(mn, r); mx = std::max(mx, r); }
      std::printf("  %-10s 판 합 평균 %+8.3f (최소 %+.3f, 최대 %+.3f) · 끝 [진행 %d, 성공 %d, 충돌 %d, 끊김 %d, 선언 실패 %d, 사람 충돌 %d, 진행 없음 %d] · 길이 %.1f s · 끝 덮음 %.3f\n",
                  nm.c_str(), mean(R.ret), mn, mx, codes[0], codes[1], codes[2], codes[3], codes[4], codes[5], codes[6], len / n_env * dec * hc.cfg.substeps * hc.cfg.dt,
                  cv / n_env);
      std::printf("             항 평균:");
      for (int k = 0; k < K; ++k) std::printf(" %s %+.3f", mdp::rew::term_name(k), R.terms[k] / n_env);
      std::printf("\n");
      if (hier)
        std::printf("             결정 %ld 개: 보상 ≠ 항 합 %ld · 결정 걸음 수 어긋남 %ld · 판 길이 ≠ 더한 걸음 × 0.1 s %ld\n", R.n_dec, R.bad_sum, R.bad_steps, R.bad_len);
    }
    auto has = [&](const char* k) { return res.count(k) > 0; };
    auto check = [&](bool ok, const char* what) {
      std::printf("  [%s] %s\n", ok ? "통과" : "실패", what);
      all_ok = all_ok && ok;
    };
    {
      long bs = 0, bst = 0, bl = 0;
      for (auto& kv : res) { bs += kv.second.bad_sum; bst += kv.second.bad_steps; bl += kv.second.bad_len; }
      check(bs == 0, "행동(결정)마다 보상 = 항 합");
      if (hier) {
        check(bst == 0 && bl == 0, "계층형 결정 보상 = 묶음 걸음 합: 안 끝난 결정은 10 걸음, 끝난 결정은 1~10 걸음, 판 길이 = 더한 걸음 × 0.1 s");
        double sl = 0, tu = 0;
        for (auto& kv : res) { sl += std::fabs(kv.second.terms[mdp::rew::kTermSlip]); tu += std::fabs(kv.second.terms[mdp::rew::kTermTurn]); }
        check(sl == 0 && tu == 0, "계층형: 헛돎·회전 항 0 (rewards.h 근거 \"계층형\")");
      }
    }
    if (has("still")) {
      double mx = -1e9;
      for (double r : res["still"].ret) mx = std::max(mx, r);
      check(mx < 0, "제자리 → 모든 판 합 < 0");
    }
    if (has("declare0")) {
      bool ok = true;
      if (hc0.cfg.success_closed) {
        for (int i = 0; i < n_env; ++i) ok = ok && res["declare0"].done[i] != mdp::cur::kDeclareShort;
        check(ok, "닫힘 성공 규칙: 선언은 무시 → 선언 실패(done 4) 판 0");
      } else {
        for (int i = 0; i < n_env; ++i) ok = ok && res["declare0"].done[i] == mdp::cur::kDeclareShort && res["declare0"].ret[i] < 0;
        check(ok, "0 걸음 선언 → 모든 판 done 4·판 합 < 0");
      }
    }
    if (hc0.cfg.success_closed) {
      int n_cl = 0, n_low = 0;
      for (auto& kv : res)
        for (int i = 0; i < n_env; ++i)
          if (kv.second.done[i] == mdp::cur::kDeclareOk) { ++n_cl; n_low += kv.second.cov[i] < 0.8f; }
      std::printf("  [참고] 지도 닫힘 성공 %d 판 (모든 시험 행동), 그중 정답 덮음 < 0.8 %d 판\n", n_cl, n_low);
      check(n_low == 0, "거짓 닫힘 없음: 닫힘 성공 판의 정답 덮음 ≥ 0.8");
      if (has("still") && has("spin") && has("frontier")) {
        const double fc = res["frontier"].terms[0] / n_env, sc = res["still"].terms[0] / n_env, pc = res["spin"].terms[0] / n_env;
        check(sc < 0.25 * fc && pc < 0.25 * fc, "덮음 항(로봇 지도 처음 막힘 칸): 제자리·제자리 돌기 < 기준선 따라가기의 1/4 (칸 키우기 없음)");
      }
    }
    if (has("still") && has("crash") && has("frontier")) {
      // 진행 없음 끝 (D1): 버티다 끝나는 판이 박기보다 덜 나쁘고(박기로 도망 안 감), 기준선 따라 선언보다 나빠야 함
      bool all6 = true;
      for (int d : res["still"].done) all6 = all6 && d == mdp::cur::kNoProgress;
      check(all6, "제자리 → 모든 판 done 6(진행 없음)");
      check(mean(res["still"].ret) > mean(res["crash"].ret), "제자리(진행 없음 끝) 판 합 평균 > 일부러 박기 판 합 평균");
      check(mean(res["still"].ret) < mean(res["frontier"].ret), "제자리(진행 없음 끝) 판 합 평균 < 기준선 따라 선언 판 합 평균");
    }
    if (has("crash") && has("frontier")) check(mean(res["crash"].ret) < mean(res["frontier"].ret), "일부러 박기 판 합 평균 < 기준선 따라 선언 판 합 평균");
    if (has("frontier")) check(mean(res["frontier"].ret) > 0, "기준선 따라가기 → 판 합 평균 > 0");
    if (has("frontier"))   // 촘촘한 항을 노리는 행동은 탐사(기준선 따라가기)보다 못 받아야 함
      for (const char* k : {"spin", "osc", "hug", "dither"}) {
        if (!has(k)) continue;
        const Result& r = res[k];
        const double sh = (r.terms[mdp::rew::kTermApproach] + r.terms[mdp::rew::kTermFrontCells] + r.terms[mdp::rew::kTermProx] + r.terms[mdp::rew::kTermSmooth]) / n_env;
        std::printf("  [참고] %s 촘촘한 항 넷 합 평균 %+.3f (기준선 %+.3f)\n", k, sh,
                    (res["frontier"].terms[mdp::rew::kTermApproach] + res["frontier"].terms[mdp::rew::kTermFrontCells] + res["frontier"].terms[mdp::rew::kTermProx] +
                     res["frontier"].terms[mdp::rew::kTermSmooth]) / n_env);
        const std::string what = std::string(k) + " 판 합 평균 < 기준선 따라 선언 판 합 평균";
        check(mean(r.ret) < mean(res["frontier"].ret), what.c_str());
      }
    if (has("h_spin")) {
      double mx = -1e9;
      bool all6 = true;
      for (int i = 0; i < n_env; ++i) { mx = std::max(mx, res["h_spin"].ret[i]); all6 = all6 && res["h_spin"].done[i] == mdp::cur::kNoProgress; }
      // 계층형엔 "서 있기" 행동이 없다 — 빈 후보는 제자리 회전(기본 판의 spin 과 같은 '참고' 행동: 첫 바퀴에 보이는 벽은 정당한 덮음). 그래서 기준은
      // 평균 < 0 · 모두 진행 없음 끝 · 가장 좋은 판도 늘 0 번 후보 평균보다 아래 (2026-11-08 첫 측정에서 넓은 맵 한 판이 +0.57 이라 "모든 판 < 0" 에서 바꿈)
      std::printf("  [참고] 제자리 회전 판 합 최대 %+.3f\n", mx);
      check(mean(res["h_spin"].ret) < 0 && all6 && (!has("h_slot0") || mx < mean(res["h_slot0"].ret)),
            "계층형 제자리(빈 후보 → 제자리 회전) → 판 합 평균 < 0, 모두 진행 없음, 최대도 늘 0 번 후보 평균 아래");
    }
    if (has("h_slot0") && has("h_spin")) {
      check(mean(res["h_slot0"].ret) > 0 && mean(res["h_slot0"].ret) > mean(res["h_spin"].ret), "늘 0 번 후보 → 판 합 평균 > 0 이고 > 제자리");
      const double fc = res["h_slot0"].terms[0] / n_env, sc = res["h_spin"].terms[0] / n_env;
      check(sc < 0.25 * fc, "덮음 항: 제자리 회전 < 늘 0 번 후보의 1/4");
    }
    if (has("h_toggle") && has("h_slot0")) check(mean(res["h_toggle"].ret) <= mean(res["h_slot0"].ret), "왕복 진동(0 번 ↔ 마지막) 판 합 평균 ≤ 늘 0 번 후보");
    if (hier) {
      std::printf("  [참고] 충돌 판:");
      for (auto& kv : res) { int c = 0; for (int d : kv.second.done) c += d == mdp::cur::kCollision || d == mdp::cur::kCollisionPerson; std::printf(" %s %d/%d", kv.first.c_str(), c, n_env); }
      std::printf("\n");
    }
    if (has("crash_safe")) {
      int c = 0;
      for (int d : res["crash_safe"].done) c += d == mdp::cur::kCollision;
      std::printf("  [참고] 안전층 켜고 앞으로 직진: 충돌 %d / %d 판\n", c, n_env);
    }
  }
  std::printf("%s\n", all_ok ? "보상 해킹 시험 통과" : "보상 해킹 시험 실패");
  return all_ok ? 0 : 1;
}
