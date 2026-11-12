// low_run — 아래층(경로 추종기) 기준선 측정 (mdp/README "아래층 계측·단독 과제", sim/include/low_task.h). 시뮬은 env_api.h 로만 쓴다.
//
//   low_run --mode stat   --scenes a.scene [b.scene …] [--people K] [--episodes 64] [--seeds 1,2,3,4] [--no-stat]
//       탐사 판(계층형 규칙 정책 — 늘 0 번 후보, frontier_run --modes slot0 과 같은 설정·씨앗·판) 안에서 걸음마다 "왜 그 속도였나" 를 분류해
//       시간 손실(= 걸린 시간 − 이동 거리 ÷ 최고 속도)을 나눈다. --no-stat: 계측을 끄고 같은 판 (판 지문이 같아야 — 계측이 판을 안 바꿈)
//   low_run --mode follow --scenes a.scene [...] [--people K] [--episodes 128] [--per-env 8] [--seed 1] [--warm 10,60] [--len 3,6,10,15]
//       아래층 단독 과제: 판 앞을 탐사(0 번 후보)로 --warm 초(고르게 뽑음) 돌려 로봇 지도를 만든 뒤, 그 지도에서 닿을 수 있는 목표 칸을
//       경로 길이 구간이 고르게 뽑아 추종기에 준다(통과하면 다음 목표). 환경 --episodes 개 × 환경당 판 --per-env 개.
//   low_run --mode verify --scenes a.scene [--people K] [--episodes 8] [--per-env 2] [--warm 3,8]
//       단독 과제를 GPU 판·CPU 참조판에 같은 명령으로 나란히 돌려 결정마다 상태·지도·관측·보상·끝·계측 누적을 비트 비교
//   low_policy_run --mode follow --policy <체크포인트> --scenes a.scene [...] [--people 0,2] (그 밖은 follow 와 같음)   ← scripts/CMakeLists.txt 의 다른 타깃
//       같은 단독 과제를 아래층 RL 정책(train --low 의 체크포인트)으로: 걸음(0.1 s)마다 정책이 (v, ω) 를 낸다. 같은 표 + ROADMAP "2 단계 통과 조건" 판정.
//       --policy 대신 --step 이면 같은 걸음 루프에서 규칙 추종기 (기준선이 다시 나오는지 보는 대조)
//   공통: [--env sim/config/env.yaml] [--model sim/config/sensor_model.yaml] [--frontier baselines/frontier/config/frontier.yaml] [--cpu]
//         [--time-limit S] [--max-goals G] (단독 과제 판 끝 규칙을 설정 파일 대신)
//
// 자기 일관성 검사(판마다, 어긋나면 끝에 "자기 일관성 실패" 와 종료 코드 1): 분류 걸음 합 = 분류한 걸음 수 = 판 걸음 수, 분류 거리 합 = 이동 거리,
// 가속 걸음의 원인별 합 = 가속 분류, 버린 구간 걸음 ≤ 판 걸음.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <queue>
#include <sstream>
#include <string>
#include <vector>

#include "env_api.h"
#include "frontier.h"
#include "frontier_explorer.h"
#ifdef LOW_RUN_POLICY   // low_policy_run (scripts/CMakeLists.txt) — 정책 측정
#include <memory>

#include "actor_critic.h"
#include "ppo.h"
#endif

namespace {

using sim::kLowKinds;
using sim::LowEp;

const char* kKind[kLowKinds] = {"제 속도", "가속 중", "곡률 감속", "장애물 근접 감속", "경로 끝 감속", "충돌 감시 감속", "도착 뒤 회전", "도착 뒤 섬",
                                "회전: 목적지 바뀜", "회전: 다시 계획", "회전: 가던 중", "회전: 목적지 없음", "경로 없음", "충돌 검사 멈춤", "복구 물러나기",
                                "회전 막힘", "안전층"};
// 누구 몫: 0 못 줄임 · 1 아래층 · 2 위층 (제 속도·가속은 따로 셈)
const int kOwner[kLowKinds] = {0, 0, 1, 1, 2, 1, 2, 2, 2, 1, 1, 2, 1, 1, 1, 1, 1};
const char* kAng[sim::kLowAng] = {"0-30", "30-60", "60-90", "90-135", "135-180", "미정"};
const char* kDst[sim::kLowDst] = {"<0.5m", "0.5-2m", ">=2m"};

struct Args {
  std::string mode = "stat";
  std::vector<std::string> scenes;
  std::vector<uint64_t> seeds{1};
  int episodes = 64, per_env = 8, people = 0;
  bool cpu = false, no_stat = false;
  float warm_lo = 10.f, warm_hi = 60.f;          // 예열 탐사 시간 (s) — 잠정
  float len[4] = {3.f, 6.f, 10.f, 15.f};         // 경로 길이 구간 경계 (m) — 설계 문서 3~15 m 를 셋으로 (잠정)
  float time_limit = -1.f;
  int max_goals = -1;
  std::string policy;             // low_policy_run: 아래층 정책 체크포인트
  bool step = false;              // low_policy_run: 정책 없이 걸음 루프(규칙 추종기)
  std::vector<int> people_list;   // --people 에 쉼표로 여럿 (low_policy_run 이 차례로 돈다)
  std::string diag;               // low_policy_run: 충돌 진단 TSV 를 쓸 파일 (없으면 끔 — 판·표·지문은 그대로)
  std::string env = SIM_CONFIG_DIR "/env.yaml", model = SIM_CONFIG_DIR "/sensor_model.yaml", frontier = FRONTIER_CONFIG;
};

std::vector<std::string> split(const std::string& s, char c) {
  std::vector<std::string> out;
  std::stringstream ss(s);
  std::string t;
  while (std::getline(ss, t, c)) if (!t.empty()) out.push_back(t);
  return out;
}
std::string stem(const std::string& p) {
  const auto s = p.find_last_of('/');
  std::string b = s == std::string::npos ? p : p.substr(s + 1);
  const auto k = b.rfind('.');
  return k == std::string::npos ? b : b.substr(0, k);
}
uint64_t mix(uint64_t& s) {
  uint64_t z = (s += 0x9E3779B97F4A7C15ull);
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
  z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
  return z ^ (z >> 31);
}
double urand(uint64_t& s) { return (double)(mix(s) >> 11) * (1.0 / 9007199254740992.0); }
void fnv(uint64_t& h, const void* p, size_t n) {
  const unsigned char* b = (const unsigned char*)p;
  for (size_t i = 0; i < n; ++i) { h ^= b[i]; h *= 1099511628211ull; }
}

// 판 누적의 합
struct Agg {
  double n[kLowKinds] = {}, dist[kLowKinds] = {}, cmd[kLowKinds] = {}, acc_n[kLowKinds + 1] = {}, acc_dist[kLowKinds + 1] = {};
  double steps = 0, dist_all = 0, sw[2][sim::kLowDst][sim::kLowAng] = {}, ab_n[sim::kLowAng] = {}, ab_dist[sim::kLowAng] = {}, left_sum = 0, n_goal = 0;
  double eps = 0;
  void add(const LowEp& L) {
    for (int k = 0; k < kLowKinds; ++k) { n[k] += L.n[k]; dist[k] += L.dist[k]; cmd[k] += L.cmd[k]; }
    for (int k = 0; k <= kLowKinds; ++k) { acc_n[k] += L.acc_n[k]; acc_dist[k] += L.acc_dist[k]; }
    steps += L.steps;
    dist_all += L.dist_all;
    for (int a = 0; a < 2; ++a)
      for (int d = 0; d < sim::kLowDst; ++d)
        for (int g = 0; g < sim::kLowAng; ++g) sw[a][d][g] += L.sw[a][d][g];
    for (int g = 0; g < sim::kLowAng; ++g) { ab_n[g] += L.ab_n[g]; ab_dist[g] += L.ab_dist[g]; }
    left_sum += L.left_sum;
    n_goal += L.n_goal;
    eps += 1;
  }
};

// 판 하나의 자기 일관성. steps_want < 0 이면 판 걸음 수 비교는 안 함. 반환 = 어긋난 항목 수
int check_ep(const LowEp& L, int steps_want, const char* tag, int env) {
  long sn = 0, sa = 0, sab = 0;
  double sd = 0, sad = 0;
  for (int k = 0; k < kLowKinds; ++k) { sn += L.n[k]; sd += L.dist[k]; }
  for (int k = 0; k <= kLowKinds; ++k) { sa += L.acc_n[k]; sad += L.acc_dist[k]; }
  for (int g = 0; g < sim::kLowAng; ++g) sab += L.ab_n[g];
  int bad = 0;
  bad += sn != L.steps;
  bad += steps_want >= 0 && L.steps != steps_want;
  bad += std::fabs(sd - L.dist_all) > 1e-3 * std::max(1.0, (double)L.dist_all);
  bad += sa != L.n[sim::kLowAccel];
  bad += std::fabs(sad - L.dist[sim::kLowAccel]) > 1e-3 * std::max(1.0, (double)L.dist[sim::kLowAccel]);
  bad += sab > L.steps;
  if (bad) std::fprintf(stderr, "자기 일관성 어긋남 [%s env %d]: 분류 합 %ld · 분류한 걸음 %d · 판 걸음 %d · 거리 합 %.4f · 이동 거리 %.4f · 가속 원인 합 %ld · 가속 %d\n",
                        tag, env, sn, L.steps, steps_want, sd, (double)L.dist_all, sa, L.n[sim::kLowAccel]);
  return bad;
}

// 분해 표 + 누구 몫
void print_decomp(const Agg& A, const sim::EnvCfg& c, const std::string& title) {
  const double dt = (double)c.substeps * c.dt, vmax = c.path_v;
  const double T = A.steps * dt, D = A.dist_all, loss_all = T - D / vmax;
  std::printf("\n== 분해: %s — 판 %.0f, 시간 %.1f s, 거리 %.1f m, 평균 속도 %.3f m/s (최고 %.2f), 손실 %.1f s (시간의 %.1f %%)\n", title.c_str(), A.eps, T, D,
              T > 0 ? D / T : 0.0, vmax, loss_all, T > 0 ? 100.0 * loss_all / T : 0.0);
  std::printf("%-26s %9s %7s %9s %9s %9s %8s\n", "분류", "걸음", "시간%", "실제m/s", "명령m/s", "손실 s", "손실%");
  for (int k = 0; k < kLowKinds; ++k) {
    if (A.n[k] <= 0) continue;
    const double t = A.n[k] * dt, loss = t - A.dist[k] / vmax;
    std::printf("%-26s %9.0f %7.2f %9.3f %9.3f %9.1f %8.2f\n", kKind[k], A.n[k], 100.0 * t / T, A.dist[k] / t, A.cmd[k] / t, loss, loss_all > 0 ? 100.0 * loss / loss_all : 0.0);
  }
  // 가속 중 걸음을 직전 감속 원인별로
  double own[3] = {0, 0, 0};   // 손실 s: 못 줄임 · 아래층 · 위층
  for (int k = 0; k < kLowKinds; ++k) {
    if (k == sim::kLowFull || k == sim::kLowAccel) continue;
    own[kOwner[k]] += A.n[k] * dt - A.dist[k] / vmax;
  }
  const double full_loss = A.n[sim::kLowFull] * dt - A.dist[sim::kLowFull] / vmax;
  double acc_own[3] = {0, 0, 0};
  std::printf("가속 중(직전 감속 원인별 손실 s):");
  for (int k = 0; k <= kLowKinds; ++k) {
    if (A.acc_n[k] <= 0) continue;
    const double loss = A.acc_n[k] * dt - A.acc_dist[k] / vmax;
    acc_own[k == kLowKinds ? 0 : kOwner[k]] += loss;
    std::printf(" %s %.1f", k == kLowKinds ? "판 시작" : kKind[k], loss);
  }
  std::printf("\n");
  const auto pc = [&](double x) { return loss_all > 0 ? 100.0 * x / loss_all : 0.0; };
  std::printf("누구 몫(손실의 %%): 아래층 %.1f (감속·멈춤·회전 %.1f + 그 뒤 가속 %.1f) · 위층 %.1f (도착·바뀜·끝 감속·목적지 없음 %.1f + 그 뒤 가속 %.1f) · "
              "못 줄임 %.1f (제 속도 구간의 명령-실제 차 %.1f + 판 시작 가속 %.1f)\n",
              pc(own[1] + acc_own[1]), pc(own[1]), pc(acc_own[1]), pc(own[2] + acc_own[2]), pc(own[2]), pc(acc_own[2]), pc(full_loss + acc_own[0]), pc(full_loss),
              pc(acc_own[0]));
  // 목적지 바뀜
  double n_sw = 0, n_arr = 0, n_ab = 0, abn = 0, abd = 0, abn_back = 0, abd_back = 0;
  for (int a = 0; a < 2; ++a)
    for (int d = 0; d < sim::kLowDst; ++d)
      for (int g = 0; g < sim::kLowAng; ++g) { n_sw += A.sw[a][d][g]; (a ? n_arr : n_ab) += A.sw[a][d][g]; }
  for (int g = 0; g < sim::kLowAng; ++g) { abn += A.ab_n[g]; abd += A.ab_dist[g]; }
  for (int g = 3; g <= 4; ++g) { abn_back += A.ab_n[g]; abd_back += A.ab_dist[g]; }
  std::printf("목적지 바뀜: %.0f 번 = 분당 %.2f (도착 뒤 %.0f · 도착 전 %.0f), 목적지 받은 수 %.0f\n", n_sw, T > 0 ? n_sw / (T / 60.0) : 0.0, n_arr, n_ab, A.n_goal);
  for (int a = 0; a < 2; ++a) {
    std::printf("  %s — 방향 변화(도):", a ? "도착 뒤" : "도착 전");
    for (int g = 0; g < sim::kLowAng; ++g) {
      double s = 0;
      for (int d = 0; d < sim::kLowDst; ++d) s += A.sw[a][d][g];
      std::printf(" %s %.0f", kAng[g], s);
    }
    std::printf(" | 옮겨간 거리:");
    for (int d = 0; d < sim::kLowDst; ++d) {
      double s = 0;
      for (int g = 0; g < sim::kLowAng; ++g) s += A.sw[a][d][g];
      std::printf(" %s %.0f", kDst[d], s);
    }
    std::printf("\n");
  }
  std::printf("  도착 전에 버린 구간: 걸음 %.0f (시간의 %.1f %%) · 거리 %.1f m (거리의 %.1f %%) · 그때 옛 목적지까지 남은 거리 평균 %.2f m\n", abn, T > 0 ? 100.0 * abn * dt / T : 0.0,
              abd, D > 0 ? 100.0 * abd / D : 0.0, n_ab > 0 ? A.left_sum / n_ab : 0.0);
  std::printf("  그중 새 목적지가 90 도 넘게 틀어진 것(되돌아감): 걸음 %.0f (시간의 %.1f %%) · 거리 %.1f m (거리의 %.1f %%)\n", abn_back, T > 0 ? 100.0 * abn_back * dt / T : 0.0,
              abd_back, D > 0 ? 100.0 * abd_back / D : 0.0);
}

struct Sim {
  sim::EnvHostCfg hc;
  sim::LidarModel lidar;
};

// 환경 하나의 로봇 지도에서 목표 칸을 뽑는다: 계획 반경으로 설 수 있는 칸 위 8 이웃 최단 거리(대각은 두 옆 칸이 모두 설 수 있을 때)가
// 구간 [len0, len3) 인 칸 — 칸이 있는 구간 가운데 하나를 고르게, 그 안 칸을 고르게. 없으면 −1
int sample_goal(const sim::EnvCfg& c, const sim::World& W, const std::vector<int16_t>& L, const sim::EnvState& s, const float* len, uint64_t& rng, float* est) {
  const mdp::MapView mv{L.data(), c.size, W.gw, W.gh};
  const mdp::Robot rb = sim::path_rb_plan(c, mdp::obsgen::reach_robot(c));
  const int gw = W.gw, gh = W.gh;
  std::vector<uint8_t> ok((size_t)gw * gh);
  for (int y = 0; y < gh; ++y)
    for (int x = 0; x < gw; ++x) ok[(size_t)y * gw + x] = mdp::standable(mv, x, y, rb);
  const int rx = (int)std::floor((s.x + s.ex - W.ox) / c.res), ry = (int)std::floor((s.y + s.ey - W.oy) / c.res);
  std::vector<int> dist((size_t)gw * gh, 1 << 30);
  typedef std::pair<int, int> QE;
  std::priority_queue<QE, std::vector<QE>, std::greater<QE>> q;
  for (int dy = -2; dy <= 2; ++dy)   // 로봇이 선 칸이 계획 반경으로는 못 서는 칸일 수 있어 둘레 2 칸 안에서 시작
    for (int dx = -2; dx <= 2; ++dx) {
      const int x = rx + dx, y = ry + dy;
      if (x < 0 || y < 0 || x >= gw || y >= gh || !ok[(size_t)y * gw + x]) continue;
      const int d0 = (int)std::lround(10.0 * std::sqrt((double)(dx * dx + dy * dy)));
      dist[(size_t)y * gw + x] = d0;
      q.push({d0, y * gw + x});
    }
  while (!q.empty()) {
    const QE t = q.top();
    q.pop();
    if (t.first != dist[t.second]) continue;
    const int x = t.second % gw, y = t.second / gw;
    for (int dy = -1; dy <= 1; ++dy)
      for (int dx = -1; dx <= 1; ++dx) {
        if (!dx && !dy) continue;
        const int u = x + dx, v = y + dy;
        if (u < 0 || v < 0 || u >= gw || v >= gh || !ok[(size_t)v * gw + u]) continue;
        if (dx && dy && (!ok[(size_t)y * gw + u] || !ok[(size_t)v * gw + x])) continue;
        const int nd = t.first + (dx && dy ? 14 : 10);
        if (nd < dist[(size_t)v * gw + u]) { dist[(size_t)v * gw + u] = nd; q.push({nd, v * gw + u}); }
      }
  }
  std::vector<int> cells[3];
  for (int i = 0; i < gw * gh; ++i) {
    if (dist[i] == (1 << 30)) continue;
    const float d = (float)dist[i] * 0.1f * c.res;
    for (int k = 0; k < 3; ++k)
      if (d >= len[k] && d < len[k + 1]) cells[k].push_back(i);
  }
  int have[3], nh = 0;
  for (int k = 0; k < 3; ++k)
    if (!cells[k].empty()) have[nh++] = k;
  if (!nh) return -1;
  const std::vector<int>& pick = cells[have[std::min(nh - 1, (int)(urand(rng) * nh))]];
  const int cell = pick[std::min(pick.size() - 1, (size_t)(urand(rng) * pick.size()))];
  *est = (float)dist[cell] * 0.1f * c.res;
  return cell;
}

float plan_len(const sim::EnvCfg& c, const sim::EnvState& s) {
  double l = 0;
  for (int i = 0; i + 1 < s.p_n; ++i) l += std::hypot((double)(s.wp[2 * i + 2] - s.wp[2 * i]), (double)(s.wp[2 * i + 3] - s.wp[2 * i + 1]));
  return (float)(l * c.res);
}

sim::EnvHostCfg base_cfg(const Args& a) {
  sim::EnvHostCfg hc = sim::load_env_cfg(a.env);
  hc.cfg.action_mode = sim::kActPath;
  hc.cfg.curriculum = 0;
  return hc;
}

// ---------------------------------------------------------------------------------------------------------------- stat
int run_stat(const Args& a) {
  sim::EnvHostCfg hc = base_cfg(a);
  const fb::Params P = fb::load_params(a.frontier, a.env, "fast");   // frontier_run run_hier 와 같은 판 길이 상한
  hc.cfg.max_steps = (int)std::ceil(P.t_safe / (hc.cfg.substeps * hc.cfg.dt)) + 100;
  hc.cfg.low_stat = a.no_stat ? 0 : 1;
  const sim::LidarModel lidar = sim::load_lidar_model(a.model);
  const sim::EnvCfg& c = hc.cfg;
  const double dt = (double)c.substeps * c.dt;
  const int N = a.episodes;
  int bad = 0;
  Agg total;
  uint64_t h_all = 1469598103934665603ull;
  for (size_t wi = 0; wi < a.scenes.size(); ++wi) {
    std::vector<sim::World> one{sim::build_world(stem(a.scenes[wi]), sim::load_scene(a.scenes[wi]), hc)};
    Agg agg;
    double succ = 0, wall = 0, person = 0, idle = 0, trunc = 0, t_sum = 0, cov = 0, n_ep = 0, pf = 0, pb = 0;
    for (uint64_t seed : a.seeds) {
      sim::Env env(hc, one, lidar, N, seed * 1000003ull + wi * 7919ull, !a.cpu);
      env.reset();
      env.sync();
      if (a.people > 0) {
        env.set_people_max(std::vector<int>(N, a.people));
        env.reset();
        env.sync();
      }
      float* d_act = nullptr;
      std::vector<float> act((size_t)mdp::act::DIM * N, 0.f);
      if (!a.cpu) {
        cudaMalloc(&d_act, sizeof(float) * act.size());
        cudaMemcpy(d_act, act.data(), sizeof(float) * act.size(), cudaMemcpyHostToDevice);
      }
      std::vector<int> fin(N, 0);
      std::vector<sim::EpInfo> ep;
      std::vector<float> rew;
      std::vector<int32_t> done;
      std::vector<LowEp> low;
      int left = N;
      uint64_t h = 1469598103934665603ull;
      for (int d = 0; left > 0 && d < c.max_steps / c.decide_steps + 10; ++d) {
        env.step(a.cpu ? act.data() : d_act);
        env.sync();
        env.get_reward(rew, done);
        bool any = false;
        for (int i = 0; i < N; ++i) any = any || (!fin[i] && done[i]);
        if (!any) continue;
        env.get_ep(ep);
        env.get_low(low);
        for (int i = 0; i < N; ++i) {
          if (fin[i] || !done[i]) continue;
          fin[i] = 1;
          --left;
          const int dd = done[i];
          succ += dd == mdp::cur::kDeclareOk;
          wall += dd == mdp::cur::kCollision;
          person += dd == mdp::cur::kCollisionPerson;
          idle += dd == mdp::cur::kNoProgress;
          trunc += dd == mdp::cur::kTruncated;
          t_sum += ep[i].t;
          cov += ep[i].coverage;
          n_ep += 1;
          fnv(h, &i, 4); fnv(h, &dd, 4); fnv(h, &ep[i].t, 4); fnv(h, &ep[i].coverage, 4); fnv(h, &ep[i].t95, 4);
          if (!low.empty()) {
            bad += check_ep(low[i], (int)std::lround(ep[i].t / dt), one[0].name.c_str(), i);
            agg.add(low[i]);
            total.add(low[i]);
            pf += low[i].hit_dir == 1;
            pb += low[i].hit_dir == 2;
          }
        }
      }
      if (left > 0) { std::fprintf(stderr, "%s 씨앗 %llu: 판 %d 개가 안 끝남\n", one[0].name.c_str(), (unsigned long long)seed, left); bad += left; }
      if (d_act) cudaFree(d_act);
      std::printf("판 지문 %s 씨앗 %llu 사람 %d: %016llx\n", one[0].name.c_str(), (unsigned long long)seed, a.people, (unsigned long long)h);
      fnv(h_all, &h, 8);
    }
    std::printf("\n#### %s · 사람 상한 %d · 씨앗 %zu 개 × %d 판: 성공 %.3f · 벽 충돌 %.3f · 사람 충돌 %.3f (앞 %0.f · 밖 %.0f) · 진행 없음 %.3f · 끊김 %.3f · 평균 판 길이 %.1f s · 평균 덮음 %.3f\n",
                one[0].name.c_str(), a.people, a.seeds.size(), N, succ / n_ep, wall / n_ep, person / n_ep, pf, pb, idle / n_ep, trunc / n_ep, t_sum / n_ep, cov / n_ep);
    if (!a.no_stat) print_decomp(agg, c, one[0].name + " 사람 " + std::to_string(a.people));
  }
  if (!a.no_stat && a.scenes.size() > 1) print_decomp(total, c, "모든 장면 사람 " + std::to_string(a.people));
  std::printf("\n전체 지문 %016llx\n", (unsigned long long)h_all);
  std::printf(bad ? "자기 일관성 실패 %d 건\n" : "자기 일관성 통과\n", bad);
  return bad ? 1 : 0;
}

// -------------------------------------------------------------------------------------------------------------- follow
enum { kOutReach = 0, kOutWall, kOutPersonFront, kOutPersonBack, kOutStall, kOutTime, kOutGiveUp, kOuts };
// 결과 이름 (차례대로): "도달", "벽 충돌", "사람 충돌(앞)", "사람 충돌(밖)", "정체", "시간 끝(진행 중)", "포기(경로 끝이 목표 밖·복구 포기)"

struct BinAgg {
  double out[kOuts] = {}, t = 0, plen = 0, dist = 0;
  std::vector<float> ratio;
};

int run_follow(const Args& a, bool verify) {
  sim::EnvHostCfg hc = base_cfg(a);
  hc.cfg.low_stat = 1;
  hc.cfg.low_task = 1;
  if (a.time_limit > 0) hc.cfg.low_time_steps = (int)std::lround(a.time_limit / (hc.cfg.substeps * hc.cfg.dt));
  if (a.max_goals > 0) hc.cfg.low_goals = a.max_goals;
  const double dt = (double)hc.cfg.substeps * hc.cfg.dt, dec_s = dt * hc.cfg.decide_steps;
  const int warm_max = (int)std::ceil(a.warm_hi / dec_s) + 1;
  hc.cfg.max_steps = (warm_max + 200) * hc.cfg.decide_steps + hc.cfg.low_time_steps;   // 예열 + 과제 구간이 판 길이 상한에 안 걸리게 (예열 다시 뽑기 여유 포함)
  const sim::LidarModel lidar = sim::load_lidar_model(a.model);
  const sim::EnvCfg& c = hc.cfg;
  const int N = a.episodes, M = a.per_env;
  int bad = 0;
  size_t v_state = 0, v_map = 0, v_low = 0, v_rew = 0, v_obs = 0, v_cam = 0;
  int v_note = 0;
  std::vector<int> v_hold(N, 0), v_tgt(N, -1);
  long v_dec = 0;
  uint64_t h_all = 1469598103934665603ull;
  std::printf("단독 과제 설정: 통과 반경 %.2f m · 정체 %d 걸음 동안 %.2f m · 시간 %d 걸음 · 목표 %d 개 · 앞 시야 ±%.1f 도 · 예열 %.0f~%.0f s · 경로 길이 구간 %.0f/%.0f/%.0f/%.0f m\n",
              c.low_pass_r, c.low_idle_steps, c.low_idle_d, c.low_time_steps, c.low_goals, c.low_front * 57.29578f, a.warm_lo, a.warm_hi, a.len[0], a.len[1], a.len[2], a.len[3]);
  for (size_t wi = 0; wi < a.scenes.size(); ++wi) {
    std::vector<sim::World> one{sim::build_world(stem(a.scenes[wi]), sim::load_scene(a.scenes[wi]), hc)};
    const sim::World& W = one[0];
    const uint64_t seed = a.seeds[0] * 1000003ull + wi * 7919ull;
    std::vector<sim::Env*> envs;
    envs.push_back(new sim::Env(hc, one, lidar, N, seed, verify ? true : !a.cpu));
    if (verify) envs.push_back(new sim::Env(hc, one, lidar, N, seed, false));
    for (sim::Env* e : envs) {
      e->reset();
      e->sync();
      if (a.people > 0) {
        e->set_people_max(std::vector<int>(N, a.people));
        e->reset();
        e->sync();
      }
    }
    const bool gpu0 = verify || !a.cpu;
    float* d_act = nullptr;
    std::vector<float> act((size_t)mdp::act::DIM * N, 0.f);
    if (gpu0) {
      cudaMalloc(&d_act, sizeof(float) * act.size());
      cudaMemcpy(d_act, act.data(), sizeof(float) * act.size(), cudaMemcpyHostToDevice);
    }
    uint64_t rng = a.seeds[0] * 0x9E3779B97F4A7C15ull + wi * 104729ull + 31;
    const auto warm_draw = [&]() { return std::max(1, (int)std::lround((a.warm_lo + urand(rng) * (a.warm_hi - a.warm_lo)) / dec_s)); };
    // 환경마다 호스트 상태
    std::vector<int> phase(N, 0), warm(N), eps(N, 0), gstate(N, 0), bin(N, 0), step0(N, 0), reached0(N, 0), off0(N, 0);
    std::vector<float> plen(N, 0.f), est(N, 0.f), dist0(N, 0.f);
    std::vector<sim::LowCmd> cmd(N, sim::LowCmd{-1, 0});
    for (int i = 0; i < N; ++i) warm[i] = warm_draw();
    std::vector<sim::EnvState> st, st2;
    std::vector<float> rew, rew2;
    std::vector<int32_t> done, done2;
    std::vector<LowEp> low, low2;
    std::vector<int16_t> lm, map1, map2;
    BinAgg B[3];
    Agg agg;
    double end_n[6] = {}, n_ep = 0, warm_fail = 0, no_sample = 0, redraw_nopath = 0, redraw_len = 0, n_trunc = 0, hit_front = 0, hit_back = 0, n_assign = 0, quick = 0;
    double est_sum = 0, plen_sum = 0, n_valid = 0;
    uint64_t h = 1469598103934665603ull;
    int active = N;
    const auto assign = [&](int i) {   // 새 목표를 뽑아 넣음 (지금 지도·자세에서). 못 뽑으면 목표 없음(−1)으로 두고 다음 결정에 다시
      envs[0]->get_map_env(i, lm);
      const int cell = sample_goal(c, W, lm, st[i], a.len, rng, &est[i]);
      cmd[i].target = cell;
      cmd[i].follow = 1;
      gstate[i] = cell >= 0 ? 0 : -1;
      if (cell < 0) no_sample += 1;
      else n_assign += 1;
      step0[i] = st[i].step;
    };
    const auto record = [&](int i, int out, float t, float d) {
      BinAgg& b = B[bin[i]];
      b.out[out] += 1;
      if (out == kOutReach) {
        b.t += t;
        b.plen += plen[i];
        b.dist += d;
        b.ratio.push_back(t / (plen[i] / c.path_v));
      }
      const float rec[4] = {(float)(bin[i] * 16 + out), plen[i], t, d};
      fnv(h, rec, sizeof(rec));
    };
    const long max_dec = (long)M * (warm_max + 200 + c.low_time_steps / c.decide_steps + 2) + 50;
    for (long d = 0; active > 0 && d < max_dec; ++d) {
      for (size_t k = 0; k < envs.size(); ++k) {
        envs[k]->set_low_cmd(cmd);
        envs[k]->step(k == 0 && gpu0 ? d_act : act.data());
        envs[k]->sync();
      }
      envs[0]->get_reward(rew, done);
      envs[0]->get_state(st);
      envs[0]->get_low(low);
      if (verify) {
        envs[1]->get_reward(rew2, done2);
        envs[1]->get_state(st2);
        envs[1]->get_low(low2);
        envs[0]->get_map(map1);
        envs[1]->get_map(map2);
        for (int i = 0; i < N; ++i) {
          v_state += std::memcmp(&st[i], &st2[i], sizeof(sim::EnvState)) != 0;
          v_low += std::memcmp(&low[i], &low2[i], sizeof(LowEp)) != 0;
          v_rew += std::memcmp(&rew[i], &rew2[i], 4) != 0 || done[i] != done2[i];
        }
        for (int i = 0; i < N; ++i) {   // 환경마다 — 처음 몇 번은 어디서 갈렸는지 적는다
          const size_t cells = map1.size() / N;
          size_t nd = 0;
          for (size_t k = 0; k < cells; ++k) nd += map1[cells * i + k] != map2[cells * i + k];
          v_map += nd;
          if (nd && v_note++ < 4) {
            std::printf("  지도 다름: 결정 %ld 환경 %d 칸 %zu — 과제 중 %d · 쉼 %d · 끝 %d · 판 끝 %d · 걸음 %d · 도달 %d · 앞 결정 쉼 %d · 목표 %d (앞 %d)\n", d, i, nd, low[i].following,
                        low[i].hold, low[i].end, (int)done[i], st[i].step, low[i].reached, v_hold[i], cmd[i].target, v_tgt[i]);
            int shown = 0;
            for (size_t k = 0; k < cells && shown < 6; ++k)
              if (map1[cells * i + k] != map2[cells * i + k]) {
                std::printf("    칸 (%zu, %zu) 첫째 %d 둘째 %d | 로봇 (%.2f, %.2f) 목표 칸 %d\n", k % c.size, k / c.size, map1[cells * i + k], map2[cells * i + k], st[i].x, st[i].y,
                            st[i].goal_cell);
                ++shown;
              }
          }
          v_hold[i] = low[i].hold;
          v_tgt[i] = cmd[i].target;
        }
        std::vector<uint8_t> im1, im2, cv1, cv2;
        std::vector<float> o1, o2, p1, p2, cp1, cp2;
        envs[0]->get_obs(im1, o1, p1);
        envs[1]->get_obs(im2, o2, p2);
        v_obs += im1 != im2;
        v_obs += std::memcmp(o1.data(), o2.data(), 4 * o1.size()) != 0;
        v_obs += std::memcmp(p1.data(), p2.data(), 4 * p1.size()) != 0;
        envs[0]->get_camlayer(cv1, cp1);
        envs[1]->get_camlayer(cv2, cp2);
        v_cam += cv1 != cv2;
        v_cam += cp1.size() != cp2.size() || std::memcmp(cp1.data(), cp2.data(), 4 * cp1.size()) != 0;
        ++v_dec;
      }
      for (int i = 0; i < N; ++i) {
        if (phase[i] == 2) continue;
        const LowEp& L = low[i];
        if (phase[i] == 0) {   // 예열 탐사
          if (done[i]) { warm_fail += 1; warm[i] = warm_draw(); continue; }   // 예열 중 충돌로 판이 끝남 — 버리고 새 판
          if (--warm[i] > 0) continue;
          assign(i);
          if (cmd[i].target < 0) { cmd[i].follow = 0; warm[i] = 5; continue; }   // 닿을 목표가 없음(지도가 아직 좁음) — 5 결정 더 탐사
          phase[i] = 1;
          reached0[i] = off0[i] = 0;
          dist0[i] = 0.f;
          continue;
        }
        // 단독 과제 구간
        bool next = false;
        if (L.reached > reached0[i]) {   // 통과 반경에 듦
          if (gstate[i] == 0 && !done[i] && st[i].goal_cell == cmd[i].target && st[i].p_n > 0) {   // 첫 결정 안에 닿음 (드묾)
            plen[i] = plan_len(c, st[i]);
            gstate[i] = plen[i] >= a.len[0] && plen[i] < a.len[3] ? 1 : 0;
            for (int k = 0; k < 3; ++k) if (plen[i] >= a.len[k]) bin[i] = k;
          }
          if (gstate[i] == 1) record(i, kOutReach, (float)((L.reach_step - step0[i]) * dt), L.dist_all - dist0[i]);
          else quick += 1;
          reached0[i] = L.reached;
          next = true;
        }
        if (done[i]) {   // 판 끝
          int e = L.end;
          if (e == sim::kLowEndNone) e = sim::kLowEndTime;   // 판 길이 상한 (안 옴 — max_steps 를 넉넉히 둠)
          end_n[e] += 1;
          hit_front += e == sim::kLowEndPerson && L.hit_dir == 1;
          hit_back += e == sim::kLowEndPerson && L.hit_dir == 2;
          if (e != sim::kLowEndGoals && gstate[i] == 1) {
            const int out = e == sim::kLowEndWall ? kOutWall : e == sim::kLowEndPerson ? (L.hit_dir == 1 ? kOutPersonFront : kOutPersonBack) : e == sim::kLowEndStall ? kOutStall : kOutTime;
            record(i, out, 0.f, 0.f);
          }
          bad += check_ep(L, -1, W.name.c_str(), i);
          agg.add(L);
          fnv(h, &L, sizeof(LowEp));
          n_ep += 1;
          eps[i] += 1;
          cmd[i] = sim::LowCmd{-1, 0};
          if (eps[i] >= M) { phase[i] = 2; --active; }
          else { phase[i] = 0; warm[i] = warm_draw(); }
          continue;
        }
        if (!next) {
          if (gstate[i] < 0) next = true;   // 지난번에 못 뽑음 — 다시
          else if (gstate[i] == 0) {        // 첫 결정 뒤: 계획기가 그 칸까지 길을 짰나, 길이가 구간 안인가 — 아니면 벌 없이 다시 뽑음
            if (st[i].goal_cell != cmd[i].target || st[i].p_n <= 0) { redraw_nopath += 1; next = true; }
            else {
              plen[i] = st[i].p_trunc ? est[i] : plan_len(c, st[i]);   // 웨이포인트 상한에 잘린 경로는 호스트 최단 거리로
              n_trunc += st[i].p_trunc != 0;
              if (plen[i] < a.len[0] || plen[i] >= a.len[3]) { redraw_len += 1; next = true; }
              else {
                gstate[i] = 1;
                for (int k = 0; k < 3; ++k) if (plen[i] >= a.len[k]) bin[i] = k;
                est_sum += est[i];
                plen_sum += plen[i];
                n_valid += 1;
              }
            }
          } else if (L.off > off0[i] || st[i].goal_cell != cmd[i].target) {   // 가던 중: 경로 끝에 섰는데 목표 밖 · 복구가 목표를 포기
            record(i, kOutGiveUp, 0.f, 0.f);
            next = true;
          }
        }
        off0[i] = L.off;
        if (next) {
          assign(i);
          dist0[i] = L.dist_all;
        }
      }
    }
    if (active > 0) { std::fprintf(stderr, "%s: 환경 %d 개가 판 수를 못 채움\n", W.name.c_str(), active); bad += active; }
    if (d_act) cudaFree(d_act);
    for (sim::Env* e : envs) delete e;
    fnv(h_all, &h, 8);
    // 보고
    const double T = agg.steps * dt, D = agg.dist_all;
    std::printf("\n#### 단독 과제 %s · 사람 상한 %d · 환경 %d × 판 %d = %.0f 판 (지문 %016llx)\n", W.name.c_str(), a.people, N, M, n_ep, (unsigned long long)h);
    std::printf("판 끝: 목표 %d 개 %.3f · 시간 %.3f · 정체 %.3f · 벽 충돌 %.3f · 사람 충돌 %.3f\n", c.low_goals, end_n[sim::kLowEndGoals] / n_ep, end_n[sim::kLowEndTime] / n_ep,
                end_n[sim::kLowEndStall] / n_ep, end_n[sim::kLowEndWall] / n_ep, end_n[sim::kLowEndPerson] / n_ep);
    std::printf("과제 구간 전체: 시간 %.0f s · 거리 %.0f m · 평균 속도 %.3f m/s | 100 m 당: 벽 충돌 %.3f · 사람 충돌 앞 %.3f · 밖 %.3f · 정체 %.3f\n", T, D, T > 0 ? D / T : 0.0,
                D > 0 ? 100.0 * end_n[sim::kLowEndWall] / D : 0.0, D > 0 ? 100.0 * hit_front / D : 0.0, D > 0 ? 100.0 * hit_back / D : 0.0,
                D > 0 ? 100.0 * end_n[sim::kLowEndStall] / D : 0.0);
    std::printf("목표 뽑기: 넣은 목표 %.0f · 유효(계획 경로 %.0f~%.0f m) %.0f · 다시 뽑음: 경로 없음·포기 %.0f, 길이 구간 밖 %.0f · 못 뽑음 %.0f · 예열 중 판 끝 %.0f · "
                "웨이포인트 잘림 %.0f · 첫 결정 안 통과(뺌) %.0f · 호스트 최단 거리 평균 %.2f m 대 계획 경로 %.2f m\n",
                n_assign, a.len[0], a.len[3], n_valid, redraw_nopath, redraw_len, no_sample, warm_fail, n_trunc, quick, n_valid > 0 ? est_sum / n_valid : 0.0,
                n_valid > 0 ? plen_sum / n_valid : 0.0);
    std::printf("%-10s %7s %7s %9s %9s %9s %9s | %6s %6s %6s %6s %6s %6s\n", "경로 길이", "목표", "도달률", "도달률*", "속도m/s", "시간비", "시간비중앙", "벽", "사람앞", "사람밖", "정체", "시간끝", "포기");
    BinAgg all;
    for (int k = 0; k < 4; ++k) {
      BinAgg& b = k < 3 ? B[k] : all;
      if (k < 3) {
        for (int o = 0; o < kOuts; ++o) all.out[o] += b.out[o];
        all.t += b.t; all.plen += b.plen; all.dist += b.dist;
        all.ratio.insert(all.ratio.end(), b.ratio.begin(), b.ratio.end());
      }
      double n = 0;
      for (int o = 0; o < kOuts; ++o) n += b.out[o];
      std::sort(b.ratio.begin(), b.ratio.end());
      const double med = b.ratio.empty() ? 0.0 : b.ratio[b.ratio.size() / 2];
      char name[32];
      if (k < 3) std::snprintf(name, sizeof(name), "%.0f-%.0f m", a.len[k], a.len[k + 1]);
      else std::snprintf(name, sizeof(name), "모두");
      const double decided = n - b.out[kOutTime];
      std::printf("%-10s %7.0f %7.3f %9.3f %9.3f %9.3f %9.3f | %6.0f %6.0f %6.0f %6.0f %6.0f %6.0f\n", name, n, n > 0 ? b.out[kOutReach] / n : 0.0,
                  decided > 0 ? b.out[kOutReach] / decided : 0.0, b.t > 0 ? b.dist / b.t : 0.0, b.plen > 0 ? b.t / (b.plen / c.path_v) : 0.0, med, b.out[kOutWall],
                  b.out[kOutPersonFront], b.out[kOutPersonBack], b.out[kOutStall], b.out[kOutTime], b.out[kOutGiveUp]);
    }
    std::printf("  (도달률 = 도달 ÷ 유효 목표 모두, 도달률* = 시간 끝으로 끊긴 목표를 뺀 것. 속도 = 도달한 목표의 이동 거리 ÷ 걸린 시간, 시간비 = 걸린 시간 ÷ (계획 경로 길이 ÷ 최고 속도) — 합의 비·중앙값)\n");
    print_decomp(agg, c, "단독 과제 " + W.name + " 사람 " + std::to_string(a.people));
  }
  std::printf("\n전체 지문 %016llx\n", (unsigned long long)h_all);
  if (verify) {
    std::printf("결정 %ld 번 비교: 상태 다름 %zu, 계측 누적 다름 %zu, 보상·끝 다름 %zu, 관측 다름 %zu, 지도 칸 다름 %zu, 카메라 층 다름 %zu\n", v_dec, v_state, v_low, v_rew, v_obs, v_map,
                v_cam);
    const bool same = !(v_state || v_low || v_rew || v_obs || v_map || v_cam);
    std::printf("%s\n", same ? "단독 과제 통과: GPU == CPU (비트 같음)" : "다름");
    if (!same) return 1;
  }
  std::printf(bad ? "자기 일관성 실패 %d 건\n" : "자기 일관성 통과\n", bad);
  return bad ? 1 : 0;
}

#ifdef LOW_RUN_POLICY
// -------------------------------------------------------------------------------------------------------------- 정책 (low_policy_run)
// 단독 과제를 아래층 RL 걸음(cfg.low_rl — env.step 한 번 = 0.1 s 걸음 하나)으로 돌린다. 예열·목표 뽑기·유효 판정·기록·표는 run_follow 와 같고,
// 과제 구간을 모는 것만 정책(LowCmd::follow 2, --policy) 또는 규칙 추종기(follow 1, --step — 같은 루프에서 기준선이 다시 나오는지 보는 대조)다.
// 다른 점: 호스트가 결정(1 s)이 아니라 걸음마다 보므로 통과한 바로 다음 걸음에 새 목표가 들어간다 (기준선은 결정 끝까지 쉼 — 쉰 걸음은 어느 쪽도 과제 시간에 안 셈)
// ---------- 진단 (--diag 파일): 정책 환경의 충돌 걸음마다(와 견줄 보통 걸음 표본) 걸음 앞 상태·행동·관측에 그 장애물이 있었나를 한 줄씩 (TSV). 끄면 아무것도 안 함 ----------
struct DiagHist { float v, w, a0, a1; int safe; };
constexpr int kDiagH = 16;
struct Diag {
  std::FILE* f = nullptr;
  std::vector<sim::EnvState> st;       // 걸음 앞 상태
  std::vector<uint8_t> img;            // 걸음 앞 정책 관측
  std::vector<float> o1d;
  std::vector<sim::LowRl> rl, rl_post;
  std::vector<DiagHist> hist;          // [N][kDiagH] 고리
  std::vector<int> hn;
  long n_ref = 0;
};
// 가장 가까운 막힌 칸(정답 격자): 걸음 앞 자세에서 (v, ω) 로 0.1 s 나아간 몸통 사각형에 가장 가까운 칸 → 걸음 앞 몸통 좌표 (u 앞, v 왼쪽)·몸통까지 거리
bool diag_nearest(const sim::EnvCfg& c, const sim::World& W, const sim::EnvState& e, float* hu, float* hv, float* hd) {
  const float dtk = (float)c.substeps * c.dt;
  const float thm = e.th + 0.5f * e.w * dtk, x1 = e.x + e.v * std::cos(thm) * dtk, y1 = e.y + e.v * std::sin(thm) * dtk, th1 = e.th + e.w * dtk;
  const float s0 = std::sin(e.th), c0 = std::cos(e.th), s1 = std::sin(th1), c1 = std::cos(th1);
  const int cx = (int)std::floor((e.x - W.ox) / c.res), cy = (int)std::floor((e.y - W.oy) / c.res);
  float best = 1e9f;
  bool got = false;
  for (int y = cy - 14; y <= cy + 14; ++y)
    for (int x = cx - 14; x <= cx + 14; ++x) {
      if (x < 0 || y < 0 || x >= W.gw || y >= W.gh || !(W.grid[(size_t)y * W.gw + x] & 1)) continue;
      const float wx = W.ox + ((float)x + 0.5f) * c.res, wy = W.oy + ((float)y + 0.5f) * c.res;
      const float u1 = c1 * (wx - x1) + s1 * (wy - y1), v1 = c1 * (wy - y1) - s1 * (wx - x1);
      const float u0 = c0 * (wx - e.x) + s0 * (wy - e.y), v0 = c0 * (wy - e.y) - s0 * (wx - e.x);
      const float d1 = std::hypot(std::max(std::fabs(u1) - c.half_len, 0.f), std::max(std::fabs(v1) - c.half_wid, 0.f));
      const float d0 = std::hypot(std::max(std::fabs(u0) - c.half_len, 0.f), std::max(std::fabs(v0) - c.half_wid, 0.f));
      const float score = std::max(d1, 0.03f) * 1000.f + d0;
      if (score < best) { best = score; *hu = u0; *hv = v0; *hd = d0; got = true; }
    }
  return got;
}
void diag_row(Diag& D, const char* kind, const sim::EnvCfg& c, const sim::World& W, int people, int i, const sim::LowEp& L, const sim::LowCmd& cmd, int safe_now, int goal_age) {
  using namespace mdp::obs;
  const sim::EnvState& e = D.st[i];
  const uint8_t* I = D.img.data() + (size_t)i * sim::kLowImg;
  const float* O = D.o1d.data() + (size_t)i * sim::kLow1d;
  float hu = 0, hv = 0, hd = 9;
  const bool got = diag_nearest(c, W, e, &hu, &hv, &hd);
  // 부딪힌 곳 (몸통 기준): F 앞면 · FC 앞 모서리 · S 옆면 · RC 뒤 모서리 · R 뒷면
  const char* reg = "-";
  if (got) {
    const bool fu = hu > c.half_len, ru = hu < -c.half_len, sv = std::fabs(hv) > c.half_wid;
    reg = fu ? (sv ? "FC" : "F") : ru ? (sv ? "RC" : "R") : "S";
  }
  int near0 = -1, near0m = -1, cam = -1, fov = 0;
  float geom = 0, lr[LIDAR_FRAMES] = {0, 0, 0, 0};
  const auto lid = [&](int bin, int f) {   // 라이다 그림 → m (lowrl_img 의 Z 차례 거꾸로)
    const int z = 4 * bin + f;
    int row = 0, col = 0;
    for (int k = 0; k < 5; ++k) { col |= ((z >> (2 * k)) & 1) << k; row |= ((z >> (2 * k + 1)) & 1) << k; }
    const float q = (float)I[sim::kLowLidarOff + row * sim::kLowLidarHW + col] / 255.f;
    return q * q * c.map_max_range;
  };
  if (got) {
    const int row = (int)std::lround(NEAR_H / 2 - 0.5f - hu / 0.1f), col = (int)std::lround(NEAR_W / 2 - 0.5f - hv / 0.1f);
    near0m = 0; cam = 0;
    for (int dy = -1; dy <= 1; ++dy)
      for (int dx = -1; dx <= 1; ++dx) {
        const int r = row + dy, q = col + dx;
        if (r < 0 || q < 0 || r >= NEAR_H || q >= NEAR_W) continue;
        const int p0 = I[r * NEAR_W + q], p2 = I[sim::kLowCamCh * NEAR_H * NEAR_W + r * NEAR_W + q];
        if (dy == 0 && dx == 0) near0 = p0;
        near0m = std::max(near0m, p0);
        cam = std::max(cam, p2);
      }
    const float lx = hu - c.mount_x, ang = std::atan2(hv, lx);
    geom = std::hypot(lx, hv);
    const int bin = (int)std::floor((ang - kLidarMin) / kLidarBin);
    fov = bin >= 0 && bin < LIDAR_BINS;
    for (int f = 0; f < LIDAR_FRAMES; ++f) {
      float m = 99.f;
      if (fov) for (int k = std::max(0, bin - 3); k <= std::min(LIDAR_BINS - 1, bin + 3); ++k) m = std::min(m, lid(k, f));
      lr[f] = m;
    }
  }
  float gap = 99.f;   // 관측 라이다 최신 장으로 센 앞 틈 (env.h front_gap 과 같은 식 — 그림 눈금만큼 거침)
  {
    const float half = c.half_wid + 0.05f, bumper = c.half_len - c.mount_x;
    for (int k = 0; k < LIDAR_BINS; ++k) {
      const float r = lid(k, 0);
      if (r >= c.map_max_range * 0.999f) continue;
      const float a = kLidarMin + ((float)k + 0.5f) * kLidarBin, fx = r * std::cos(a), fy = r * std::sin(a);
      if (fx <= 0.f || std::fabs(fy) > half) continue;
      gap = std::min(gap, fx - bumper);
    }
  }
  const int cx = (int)std::floor((e.x - W.ox) / c.res), cy = (int)std::floor((e.y - W.oy) / c.res);
  const int clr = cx >= 0 && cy >= 0 && cx < W.gw && cy < W.gh ? (int)W.clear[(size_t)cy * W.gw + cx] : -1;
  float gd = -1.f;
  if (cmd.target >= 0) {
    const float gx = W.ox + ((float)(cmd.target % W.gw) + 0.5f) * c.res - (e.x + e.ex), gy = W.oy + ((float)(cmd.target / W.gw) + 0.5f) * c.res - (e.y + e.ey);
    gd = std::hypot(gx, gy);
  }
  const DiagHist* H = D.hist.data() + (size_t)i * kDiagH;
  const int hn = D.hn[i];
  const auto h = [&](int back) -> const DiagHist& { return H[((hn - 1 - back) % kDiagH + kDiagH) % kDiagH]; };   // back 0 = 이번 걸음
  float v5 = -9, v10 = -9, wmax5 = 0, a0m5 = 0;
  int safe10 = 0, k5 = 0;
  for (int k = 0; k < std::min(hn, 10); ++k) {
    safe10 += h(k).safe;
    if (k < 5) { wmax5 = std::max(wmax5, std::fabs(h(k).w)); a0m5 += h(k).a0; ++k5; }
  }
  if (hn > 5) v5 = h(5).v;
  if (hn > 10) v10 = h(10).v;
  const DiagHist& now = h(0);
  std::fprintf(D.f, "%s\t%s\t%d\t%d\t%d\t%d\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%d\t%.3f\t%.3f\t%.3f\t%.3f\t%d\t%.3f\t%.3f\t%.3f\t%s\t%d\t%d\t%d\t%d\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%d\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%d\t%d\t%d\n",
               kind, W.name.c_str(), people, i, e.step - L.f_start, goal_age, e.x, e.y, e.th, e.v, e.w, now.a0, now.a1, safe_now, v5, v10, wmax5, k5 ? a0m5 / k5 : 0.f, safe10,
               hu, hv, hd, reg, near0, near0m, cam, fov, geom, lr[0], lr[1], lr[2], lr[3], gap, clr, gd, O[sim::kLoRem] * sim::kLrGoalNorm, O[sim::kLoSub] * sim::kLrSubNorm,
               O[sim::kLoSub + 1], O[sim::kLoSub + 2], (int)O[sim::kLoHasPath], e.step - D.rl[i].p_step, e.delay);
  if (kind[0] != 'r') std::fflush(D.f);
}
const char* kDiagHead = "kind\tscene\tpeople\tenv\ttask_step\tgoal_age\tx\ty\tth\tv\tw\ta_v\ta_w\tsafe\tv5\tv10\twmax5\ta_v_mean5\tsafe10\thu\thv\thd\treg\tnear0\tnear0max\tcam\tfov\tgeom\tlid0\tlid1\tlid2\tlid3\tgap\tclr_cm\tgoal_d\trem\tsub_d\tsub_sin\tsub_cos\thas_path\tplan_age\tdelay\n";

struct PolRes {
  std::string scene;
  int people;
  double speed, wall100, front100, reach, reach_x, goals;
};

int run_policy(const Args& a) {
  sim::EnvHostCfg hc = base_cfg(a);
  hc.cfg.low_stat = hc.cfg.low_task = hc.cfg.low_rl = 1;
  hc.cfg.low_auto = 0;    // 목표는 호스트가 (Env::low_prepare)
  hc.cfg.low_act_w = 0;   // 행동 줄 폭 mdp::act::DIM — 예열 환경이 후보 번호를 읽음
  if (a.time_limit > 0) hc.cfg.low_time_steps = (int)std::lround(a.time_limit / (hc.cfg.substeps * hc.cfg.dt));
  if (a.max_goals > 0) hc.cfg.low_goals = a.max_goals;
  const int DS = hc.cfg.decide_steps;
  const double dt = (double)hc.cfg.substeps * hc.cfg.dt, dec_s = dt * DS;
  const int warm_max = (int)std::ceil(a.warm_hi / dec_s) + 1;
  hc.cfg.max_steps = (warm_max + 200) * DS + hc.cfg.low_time_steps;
  const sim::LidarModel lidar = sim::load_lidar_model(a.model);
  const sim::EnvCfg& c = hc.cfg;
  const int N = a.episodes, M = a.per_env, AW = mdp::act::DIM, PW = mdp::low::ACT;
  const bool use_pol = !a.policy.empty();
  const int fol = use_pol ? 2 : 1;
  std::unique_ptr<xppo::PPO> ppo;
  if (use_pol) {
    xppo::PPOConfig pc;
    pc.num_envs = N;
    ppo.reset(new xppo::PPO(pc, xnet::explore_spec_low(), 1));
    ppo->load(a.policy);
    std::printf("정책: %s (반복 %d, 변수 %ld) — 결정적(평균 행동)\n", a.policy.c_str(), ppo->iteration(), (long)ppo->ac().n_params);
  } else {
    std::printf("정책 없음(--step): 같은 걸음 루프에서 규칙 추종기\n");
  }
  std::printf("단독 과제 설정: 통과 반경 %.2f m · 정체 %d 걸음 동안 %.2f m · 시간 %d 걸음 · 목표 %d 개 · 앞 시야 ±%.1f 도 · 예열 %.0f~%.0f s · 경로 길이 구간 %.0f/%.0f/%.0f/%.0f m\n",
              c.low_pass_r, c.low_idle_steps, c.low_idle_d, c.low_time_steps, c.low_goals, c.low_front * 57.29578f, a.warm_lo, a.warm_hi, a.len[0], a.len[1], a.len[2], a.len[3]);
  const std::vector<int> people_set = a.people_list.empty() ? std::vector<int>{0, 2} : a.people_list;
  std::vector<PolRes> res;
  int bad = 0;
  uint64_t h_all = 1469598103934665603ull;
  float *d_act, *d_zero, *d_pol;
  cudaMalloc(&d_act, sizeof(float) * AW * N);
  cudaMalloc(&d_zero, sizeof(float) * AW * N);
  cudaMalloc(&d_pol, sizeof(float) * PW * N);
  cudaMemset(d_zero, 0, sizeof(float) * AW * N);
  cudaMemset(d_act, 0, sizeof(float) * AW * N);
  Diag dg;
  if (!a.diag.empty()) {
    dg.f = std::fopen(a.diag.c_str(), "w");
    if (!dg.f) throw std::runtime_error("--diag 파일을 못 엶: " + a.diag);
    std::fputs(kDiagHead, dg.f);
  }
  for (int people : people_set)
  for (size_t wi = 0; wi < a.scenes.size(); ++wi) {
    std::vector<sim::World> one{sim::build_world(stem(a.scenes[wi]), sim::load_scene(a.scenes[wi]), hc)};
    const sim::World& W = one[0];
    const uint64_t seed = a.seeds[0] * 1000003ull + wi * 7919ull;
    sim::Env env(hc, one, lidar, N, seed, true);
    env.reset();
    env.sync();
    if (people > 0) {
      env.set_people_max(std::vector<int>(N, people));
      env.reset();
      env.sync();
    }
    std::vector<float> act((size_t)AW * N, 0.f), pol((size_t)PW * N, 0.f);
    uint64_t rng = a.seeds[0] * 0x9E3779B97F4A7C15ull + wi * 104729ull + 31;
    const auto warm_draw = [&]() { return DS * std::max(1, (int)std::lround((a.warm_lo + urand(rng) * (a.warm_hi - a.warm_lo)) / dec_s)); };
    std::vector<int> phase(N, 0), warm(N), eps(N, 0), gstate(N, 0), bin(N, 0), step0(N, 0), reached0(N, 0), off0(N, 0);
    std::vector<float> plen(N, 0.f), est(N, 0.f), dist0(N, 0.f);
    std::vector<sim::LowCmd> cmd(N, sim::LowCmd{-1, 0});
    for (int i = 0; i < N; ++i) warm[i] = warm_draw();
    std::vector<sim::EnvState> st;
    std::vector<float> rew;
    std::vector<int32_t> done;
    std::vector<LowEp> low;
    std::vector<int16_t> lm;
    std::vector<sim::LowRlStat> rs;
    env.take_low_stat(rs);
    BinAgg B[3];
    Agg agg;
    double end_n[6] = {}, n_ep = 0, warm_fail = 0, no_sample = 0, redraw_nopath = 0, redraw_len = 0, n_trunc = 0, hit_front = 0, hit_back = 0, n_assign = 0, quick = 0;
    double est_sum = 0, plen_sum = 0, n_valid = 0;
    uint64_t h = 1469598103934665603ull;
    int active = N;
    const auto assign = [&](int i) {
      env.get_map_env(i, lm);
      const int cell = sample_goal(c, W, lm, st[i], a.len, rng, &est[i]);
      cmd[i].target = cell;
      cmd[i].follow = fol;
      gstate[i] = cell >= 0 ? 0 : -1;
      if (cell < 0) no_sample += 1;
      else n_assign += 1;
      step0[i] = st[i].step;
    };
    const auto record = [&](int i, int out, float t, float d) {
      BinAgg& b = B[bin[i]];
      b.out[out] += 1;
      if (out == kOutReach) {
        b.t += t;
        b.plen += plen[i];
        b.dist += d;
        b.ratio.push_back(t / (plen[i] / c.path_v));
      }
      const float rec[4] = {(float)(bin[i] * 16 + out), plen[i], t, d};
      fnv(h, rec, sizeof(rec));
    };
    const long max_t = (long)M * ((long)(warm_max + 200) * DS + c.low_time_steps + 2 * DS) + 50 * DS;
    if (dg.f) { dg.hist.assign((size_t)N * kDiagH, DiagHist{}); dg.hn.assign(N, 0); }
    for (long t = 0; active > 0 && t < max_t; ++t) {
      env.set_low_cmd(cmd);
      env.low_prepare(d_zero);
      env.sync();
      if (dg.f) {   // 걸음 앞 상태·관측 (진단)
        env.get_state(dg.st);
        env.get_low_obs(dg.img, dg.o1d);
        env.get_low_rl(dg.rl);
      }
      if (use_pol) {
        const sim::StepOut o = env.outputs();
        ppo->act_inference(o.low_img, o.low_1d, d_pol, N, -1);
        cudaMemcpy(pol.data(), d_pol, sizeof(float) * pol.size(), cudaMemcpyDeviceToHost);
        for (int i = 0; i < N; ++i) {   // 정책 환경만 (v, ω) — 예열 환경은 0 번 후보
          const bool p = cmd[i].follow == 2;
          act[(size_t)AW * i] = p ? pol[(size_t)PW * i] : 0.f;
          act[(size_t)AW * i + 1] = p ? pol[(size_t)PW * i + 1] : 0.f;
        }
        cudaMemcpy(d_act, act.data(), sizeof(float) * act.size(), cudaMemcpyHostToDevice);
      }
      env.step(d_act);
      env.sync();
      env.get_reward(rew, done);
      env.get_state(st);
      env.get_low(low);
      if (dg.f) env.get_low_rl(dg.rl_post);
      for (int i = 0; i < N; ++i) {
        if (phase[i] == 2) continue;
        const LowEp& L = low[i];
        if (dg.f && phase[i] == 1 && cmd[i].follow == 2) {   // 진단: 정책이 몬 걸음
          const int safe_now = done[i] ? (L.kind == sim::kLowSafety) : dg.rl_post[i].safe;
          dg.hist[(size_t)i * kDiagH + dg.hn[i] % kDiagH] = DiagHist{dg.st[i].v, dg.st[i].w, pol[(size_t)PW * i], pol[(size_t)PW * i + 1], safe_now};
          dg.hn[i] += 1;
          const int e_ = done[i] ? L.end : 0;
          if (e_ == sim::kLowEndWall || e_ == sim::kLowEndPerson)
            diag_row(dg, e_ == sim::kLowEndWall ? "wall" : (L.hit_dir == 1 ? "personF" : "personB"), c, W, people, i, L, cmd[i], safe_now, dg.st[i].step - step0[i]);
          else if (!done[i] && (dg.n_ref++ % 97) == 0) diag_row(dg, "ref", c, W, people, i, L, cmd[i], safe_now, dg.st[i].step - step0[i]);
          if (done[i]) dg.hn[i] = 0;
        }
        if (phase[i] == 0) {   // 예열 탐사
          if (done[i]) { warm_fail += 1; warm[i] = warm_draw(); continue; }
          if (--warm[i] > 0) continue;
          assign(i);
          if (cmd[i].target < 0) { cmd[i].follow = 0; warm[i] = 5 * DS; continue; }
          phase[i] = 1;
          reached0[i] = off0[i] = 0;
          dist0[i] = 0.f;
          continue;
        }
        bool next = false;
        if (L.reached > reached0[i]) {
          if (gstate[i] == 0 && !done[i] && st[i].goal_cell == cmd[i].target && st[i].p_n > 0) {
            plen[i] = plan_len(c, st[i]);
            gstate[i] = plen[i] >= a.len[0] && plen[i] < a.len[3] ? 1 : 0;
            for (int k = 0; k < 3; ++k) if (plen[i] >= a.len[k]) bin[i] = k;
          }
          if (gstate[i] == 1) record(i, kOutReach, (float)((L.reach_step - step0[i]) * dt), L.dist_all - dist0[i]);
          else quick += 1;
          reached0[i] = L.reached;
          next = true;
        }
        if (done[i]) {
          int e = L.end;
          if (e == sim::kLowEndNone) e = sim::kLowEndTime;
          end_n[e] += 1;
          hit_front += e == sim::kLowEndPerson && L.hit_dir == 1;
          hit_back += e == sim::kLowEndPerson && L.hit_dir == 2;
          if (e != sim::kLowEndGoals && gstate[i] == 1) {
            const int out = e == sim::kLowEndWall ? kOutWall : e == sim::kLowEndPerson ? (L.hit_dir == 1 ? kOutPersonFront : kOutPersonBack) : e == sim::kLowEndStall ? kOutStall : kOutTime;
            record(i, out, 0.f, 0.f);
          }
          bad += check_ep(L, -1, W.name.c_str(), i);
          agg.add(L);
          fnv(h, &L, sizeof(LowEp));
          n_ep += 1;
          eps[i] += 1;
          cmd[i] = sim::LowCmd{-1, 0};
          if (eps[i] >= M) { phase[i] = 2; --active; }
          else { phase[i] = 0; warm[i] = warm_draw(); }
          continue;
        }
        if (!next) {
          if (gstate[i] < 0) next = true;
          else if (gstate[i] == 0) {   // 목표를 넣은 뒤 첫 걸음: 계획기가 그 칸까지 길을 짰나, 길이가 구간 안인가
            if (st[i].goal_cell != cmd[i].target || st[i].p_n <= 0) { redraw_nopath += 1; next = true; }
            else {
              plen[i] = st[i].p_trunc ? est[i] : plan_len(c, st[i]);
              n_trunc += st[i].p_trunc != 0;
              if (plen[i] < a.len[0] || plen[i] >= a.len[3]) { redraw_len += 1; next = true; }
              else {
                gstate[i] = 1;
                for (int k = 0; k < 3; ++k) if (plen[i] >= a.len[k]) bin[i] = k;
                est_sum += est[i];
                plen_sum += plen[i];
                n_valid += 1;
              }
            }
          } else if (L.off > off0[i] || st[i].goal_cell != cmd[i].target) {
            record(i, kOutGiveUp, 0.f, 0.f);
            next = true;
          }
        }
        off0[i] = L.off;
        if (next) {
          assign(i);
          dist0[i] = L.dist_all;
        }
      }
    }
    if (active > 0) { std::fprintf(stderr, "%s: 환경 %d 개가 판 수를 못 채움\n", W.name.c_str(), active); bad += active; }
    fnv(h_all, &h, 8);
    env.take_low_stat(rs);
    double ps = 0, safe = 0, back = 0;
    for (const sim::LowRlStat& s : rs) { ps += s.steps; safe += s.safe_steps; back += s.back_steps; }
    const double T = agg.steps * dt, D = agg.dist_all;
    std::printf("\n#### 단독 과제(%s) %s · 사람 상한 %d · 환경 %d × 판 %d = %.0f 판 (지문 %016llx)\n", use_pol ? "정책" : "규칙 추종기·걸음 루프", W.name.c_str(), people, N, M, n_ep,
                (unsigned long long)h);
    std::printf("판 끝: 목표 %d 개 %.3f · 시간 %.3f · 정체 %.3f · 벽 충돌 %.3f · 사람 충돌 %.3f\n", c.low_goals, end_n[sim::kLowEndGoals] / n_ep, end_n[sim::kLowEndTime] / n_ep,
                end_n[sim::kLowEndStall] / n_ep, end_n[sim::kLowEndWall] / n_ep, end_n[sim::kLowEndPerson] / n_ep);
    std::printf("과제 구간 전체: 시간 %.0f s · 거리 %.0f m · 평균 속도 %.3f m/s | 100 m 당: 벽 충돌 %.3f · 사람 충돌 앞 %.3f · 밖 %.3f · 정체 %.3f\n", T, D, T > 0 ? D / T : 0.0,
                D > 0 ? 100.0 * end_n[sim::kLowEndWall] / D : 0.0, D > 0 ? 100.0 * hit_front / D : 0.0, D > 0 ? 100.0 * hit_back / D : 0.0,
                D > 0 ? 100.0 * end_n[sim::kLowEndStall] / D : 0.0);
    if (use_pol) std::printf("정책 걸음 %.0f: 안전층이 고친 걸음 %.4f · 후진 명령 걸음 %.4f\n", ps, ps > 0 ? safe / ps : 0.0, ps > 0 ? back / ps : 0.0);
    std::printf("목표 뽑기: 넣은 목표 %.0f · 유효(계획 경로 %.0f~%.0f m) %.0f · 다시 뽑음: 경로 없음·포기 %.0f, 길이 구간 밖 %.0f · 못 뽑음 %.0f · 예열 중 판 끝 %.0f · "
                "웨이포인트 잘림 %.0f · 첫 걸음 안 통과(뺌) %.0f · 호스트 최단 거리 평균 %.2f m 대 계획 경로 %.2f m\n",
                n_assign, a.len[0], a.len[3], n_valid, redraw_nopath, redraw_len, no_sample, warm_fail, n_trunc, quick, n_valid > 0 ? est_sum / n_valid : 0.0,
                n_valid > 0 ? plen_sum / n_valid : 0.0);
    std::printf("%-10s %7s %7s %9s %9s %9s %9s | %6s %6s %6s %6s %6s %6s\n", "경로 길이", "목표", "도달률", "도달률*", "속도m/s", "시간비", "시간비중앙", "벽", "사람앞", "사람밖", "정체", "시간끝", "포기");
    BinAgg all;
    PolRes R{W.name, people, T > 0 ? D / T : 0.0, D > 0 ? 100.0 * end_n[sim::kLowEndWall] / D : 0.0, D > 0 ? 100.0 * hit_front / D : 0.0, 0, 0, 0};
    for (int k = 0; k < 4; ++k) {
      BinAgg& b = k < 3 ? B[k] : all;
      if (k < 3) {
        for (int o = 0; o < kOuts; ++o) all.out[o] += b.out[o];
        all.t += b.t; all.plen += b.plen; all.dist += b.dist;
        all.ratio.insert(all.ratio.end(), b.ratio.begin(), b.ratio.end());
      }
      double n = 0;
      for (int o = 0; o < kOuts; ++o) n += b.out[o];
      std::sort(b.ratio.begin(), b.ratio.end());
      const double med = b.ratio.empty() ? 0.0 : b.ratio[b.ratio.size() / 2];
      char name[32];
      if (k < 3) std::snprintf(name, sizeof(name), "%.0f-%.0f m", a.len[k], a.len[k + 1]);
      else std::snprintf(name, sizeof(name), "모두");
      const double decided = n - b.out[kOutTime];
      std::printf("%-10s %7.0f %7.3f %9.3f %9.3f %9.3f %9.3f | %6.0f %6.0f %6.0f %6.0f %6.0f %6.0f\n", name, n, n > 0 ? b.out[kOutReach] / n : 0.0,
                  decided > 0 ? b.out[kOutReach] / decided : 0.0, b.t > 0 ? b.dist / b.t : 0.0, b.plen > 0 ? b.t / (b.plen / c.path_v) : 0.0, med, b.out[kOutWall],
                  b.out[kOutPersonFront], b.out[kOutPersonBack], b.out[kOutStall], b.out[kOutTime], b.out[kOutGiveUp]);
      if (k == 3) { R.goals = n; R.reach = n > 0 ? b.out[kOutReach] / n : 0.0; R.reach_x = decided > 0 ? b.out[kOutReach] / decided : 0.0; }
    }
    res.push_back(R);
    std::fflush(stdout);
  }
  if (dg.f) std::fclose(dg.f);
  cudaFree(d_act);
  cudaFree(d_zero);
  cudaFree(d_pol);
  // ROADMAP "2 단계 통과 조건" (2026-11-12 고정) — 값은 과제 구간 전체 평균. 한도는 시험 맵 둘에만 있다
  struct Lim { const char* scene; double v0, wall0, front2, v2, reach; };
  const Lim lims[2] = {{"bookstore", 0.775, 0.014, 0.061, 0.676, 0.921}, {"small-house", 0.880, 0.008, 0.083, 0.768, 0.968}};
  std::printf("\n#### 2 단계 통과 조건 (ROADMAP — 사람 0·2, 맵마다)\n");
  int n_row = 0, n_ok = 0;
  for (const PolRes& r : res) {
    const Lim* l = nullptr;
    for (const Lim& x : lims) if (r.scene == x.scene) l = &x;
    if (!l) { std::printf("%s 사람 %d: 속도 %.3f · 벽 %.3f · 사람 앞 %.3f · 도달률* %.3f (한도 없는 맵)\n", r.scene.c_str(), r.people, r.speed, r.wall100, r.front100, r.reach_x); continue; }
    const auto row = [&](const char* name, double v, double lim, bool ge) {
      const bool ok = ge ? v >= lim : v <= lim;
      std::printf("  %-12s 사람 %d | %-34s %.3f (%s %.3f) %s\n", r.scene.c_str(), r.people, name, v, ge ? "≥" : "≤", lim, ok ? "통과" : "불통과");
      ++n_row;
      n_ok += ok;
    };
    if (r.people == 0) {
      row("평균 속도 m/s", r.speed, l->v0, true);
      row("벽 충돌 /100 m", r.wall100, l->wall0, false);
      row("도달률*(시간 끝 뺌) ≥ 기준선 − 0.02", r.reach_x, l->reach - 0.02, true);
    } else if (r.people == 2) {
      row("앞 220° 사람 충돌 /100 m", r.front100, l->front2, false);
      row("평균 속도 m/s", r.speed, l->v2, true);
      std::printf("  %-12s 사람 %d | 도달률* %.3f (참고 — 한도는 사람 0 줄에)\n", r.scene.c_str(), r.people, r.reach_x);
    }
  }
  std::printf("통과 조건 %d 줄 가운데 %d 줄 통과 (시험 맵 둘 × 사람 0·2 를 모두 돌렸을 때 10 줄 = 표 네 줄 × 맵 둘 + 도달률 둘)\n", n_row, n_ok);
  std::printf("\n전체 지문 %016llx\n", (unsigned long long)h_all);
  std::printf(bad ? "자기 일관성 실패 %d 건\n" : "자기 일관성 통과\n", bad);
  return bad ? 1 : 0;
}
#endif   // LOW_RUN_POLICY

}  // namespace

int main(int argc, char** argv) {
  Args a;
  for (int i = 1; i < argc; ++i) {
    const std::string k = argv[i];
    const auto val = [&]() -> std::string { return i + 1 < argc ? argv[++i] : ""; };
    if (k == "--mode") a.mode = val();
    else if (k == "--scenes") { while (i + 1 < argc && std::strncmp(argv[i + 1], "--", 2) != 0) a.scenes.push_back(argv[++i]); }
    else if (k == "--seeds" || k == "--seed") { a.seeds.clear(); for (const std::string& s : split(val(), ',')) a.seeds.push_back(std::stoull(s)); }
    else if (k == "--episodes") a.episodes = std::stoi(val());
    else if (k == "--per-env") a.per_env = std::stoi(val());
    else if (k == "--people") { a.people_list.clear(); for (const std::string& s : split(val(), ',')) a.people_list.push_back(std::stoi(s)); if (!a.people_list.empty()) a.people = a.people_list[0]; }
    else if (k == "--policy") a.policy = val();
    else if (k == "--step") a.step = true;
    else if (k == "--diag") a.diag = val();
    else if (k == "--warm") { const auto v = split(val(), ','); if (v.size() == 2) { a.warm_lo = std::stof(v[0]); a.warm_hi = std::stof(v[1]); } }
    else if (k == "--len") { const auto v = split(val(), ','); if (v.size() == 4) for (int j = 0; j < 4; ++j) a.len[j] = std::stof(v[j]); }
    else if (k == "--time-limit") a.time_limit = std::stof(val());
    else if (k == "--max-goals") a.max_goals = std::stoi(val());
    else if (k == "--env") a.env = val();
    else if (k == "--model") a.model = val();
    else if (k == "--frontier") a.frontier = val();
    else if (k == "--cpu") a.cpu = true;
    else if (k == "--no-stat") a.no_stat = true;
    else { std::fprintf(stderr, "모르는 인자 %s\n", k.c_str()); return 2; }
  }
  if (a.scenes.empty() || a.seeds.empty()) { std::fprintf(stderr, "쓰는 법은 파일 머리말 (scripts/low_run.cpp)\n"); return 2; }
  try {
    if (a.mode == "stat") return run_stat(a);
#ifdef LOW_RUN_POLICY
    if (a.mode == "follow" && (!a.policy.empty() || a.step)) return run_policy(a);
#else
    if (!a.policy.empty() || a.step) { std::fprintf(stderr, "--policy · --step 은 low_policy_run 에만 있음 (scripts/CMakeLists.txt)\n"); return 2; }
#endif
    if (a.mode == "follow") return run_follow(a, false);
    if (a.mode == "verify") return run_follow(a, true);
    std::fprintf(stderr, "모르는 모드 %s\n", a.mode.c_str());
    return 2;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "low_run: %s\n", e.what());
    return 1;
  }
}
