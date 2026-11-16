// 환경 한 걸음 CPU 참조판: GPU 커널(src/env_kernel.cu)의 블록 하나를 스레드 순서대로 흉내 낸다. 단계 순서는 커널과 한 줄씩 같다
#include <algorithm>
#include <thread>
#include <vector>

#include "env_api.h"
#include "obs_ref.h"
#include "plan_ref.h"

namespace sim {
namespace {

int g_cpu_threads = 1;
CpuTap* g_tap = nullptr;

// 환경 [0, n) 을 스레드 g_cpu_threads 개에 고르게 나눠 f(env) — 환경끼리 독립(상태·지도·관측이 환경마다 따로)이라 결과는 스레드 수와 무관
template <class F>
void for_envs(int n, F f) {
  const int T = std::max(1, std::min(g_cpu_threads, n));
  if (T == 1) {
    for (int env = 0; env < n; ++env) f(env);
    return;
  }
  std::vector<std::thread> th;
  for (int t = 0; t < T; ++t)
    th.emplace_back([&, t] {
      for (int env = t; env < n; env += T) f(env);   // 번갈아 나눔: 맵 크기가 섞여 짐이 고름
    });
  for (auto& x : th) x.join();
}

void reset_env(const EnvBatch& b, EnvState& e, int env) {
  for (int tid = 0; tid < kThreads; ++tid) reset_map(b, env, tid);
  reset_state(b, e);
}


// 계층형 결정: include/plan_ref.h plan_ref (실물 노드와 같은 함수)
void plan_cpu(const EnvBatch& b, EnvState& e, int env, int k) { plan_ref(b, e, env, k); }

// 환경 하나 행동 하나. sub < 0: 기본, sub ≥ 0: 계층형 묶음 sub 번째 걸음 (GPU env_finish_kernel 의 sub 와 같은 규칙)
void step_env(const EnvBatch& b, const float* actions, int env, int sub) {
  const EnvCfg& c = b.cfg;
  if (sub > 0 && b.done[env] != 0) return;   // 계층형: 이 묶음 앞 걸음에서 판이 끝나 쉼
  if (gdec_idle(b, env, sub)) return;          // 그래프 모드: 결정이 앞 걸음에서 끝남
  Shared sh;
  Shared& g = b.scratch[env];   // 걸음을 넘어 남는 칸 (먼 지도 다시 그릴 곳 — GPU 와 같은 증분)
  mdp::obsgen::far_dirty_clear(g);
  EnvState e = b.st[env];
  e.collided = 0;
  if (sub == 0) gdec_begin(b, e, env);
  apply_action(b, e, env, actions + (size_t)lowrl_act_w(b) * env);
  int new_wall = 0, new_floor = 0, d_known = 0, n_new = 0, n_occ = 0;
  sh.scanned = 0;
  const int tick0 = e.tick;
  for (int k = 0; k < c.substeps; ++k) {
    if (!e.collided) physics_step(b, e);
    e.tick += 1;
    if ((tick0 + k + 1) % c.lidar_every != 0) continue;
    scan_begin(b, e, sh);
    if (CpuTap* tap = g_tap) {   // 시험 탭: 이 장을 밖으로 (같은 값을 실물 노드 RobotEnv 에 넣어 비교 — tools/rl_node_check)
      std::vector<float> rr((size_t)sh.n_beams), it((size_t)sh.n_beams);
      for (int j = 0; j < sh.n_beams; ++j) scan_beam(b, e, env, sh, j, &rr[(size_t)j], &it[(size_t)j]);
      tap->scan(env, e, sh, rr, it);
    } else {
      for (int j = 0; j < sh.n_beams; ++j) scan_beam(b, e, env, sh, j);
    }
    int s = 0, f = 0, kn = 0, nn = 0, no = 0;
    for (int tid = 0; tid < kThreads; ++tid) {
      const ApplyCount a = scan_apply(b, e, env, sh, tid, kThreads, g.far_mask);
      s += a.wall;
      f += a.floor;
      kn += a.known;
      nn += a.fresh;
      no += a.occ;
    }
    new_wall += s;
    new_floor += f;
    d_known += kn;
    n_new += nn;
    n_occ += no;
    known_box(b, e, sh);
  }
  map_progress(e, n_occ);   // 진행 없음 시계 — finish_step 판 끝이 씀
  sense_step(b, e, env, sh);   // 로봇이 재는 간격·닿음 (contact.h) — 보상·판 끝이 씀
  finish_step(b, e, env, new_wall, new_floor, d_known, n_new);
  mdp::obsgen::far_dirty_robot(b, e, g);
  b.reward[env] = e.reward;
  b.done[env] = e.done;
  if (sub >= 0) {   // 계층형: 결정 동안 합
    const float* T = b.reward_terms + (size_t)env * mdp::rew::kTerms;
    float* A = b.tacc + (size_t)env * mdp::rew::kTerms;
    if (sub == 0) {
      b.racc[env] = e.reward;
      for (int k = 0; k < mdp::rew::kTerms; ++k) A[k] = T[k];
    } else {
      b.racc[env] += e.reward;
      for (int k = 0; k < mdp::rew::kTerms; ++k) A[k] += T[k];
    }
  }
  gdec_step_ref(b, e, env, sub);   // 그래프 모드: 결정 끝 판정 (GPU env_finish_kernel 과 같은 자리)
  if (CpuTap* tap = g_tap) tap->step_end(env, e);   // 시험 탭: 걸음 끝 상태 (판 끝 리셋·관측 전)
  if (sh.scanned) ring_advance(e);
  if (sh.scanned)
    for (int tid = 0; tid < kThreads; ++tid) push_bins(b, e, env, sh, tid);
  const bool was_done = e.done != 0;
  if (was_done && sub >= 0) {   // 계층형: 리셋·관측은 묶음 끝에
    b.st[env] = e;
    return;
  }
  if (was_done) reset_env(b, e, env);
  write_obs(b, e, env, sh, mdp::obsgen::far_dirty_take(b, e, g, was_done),
            sub < 0 || (b.gdec ? b.gdec[env].reason != kGdRun : sub == c.decide_steps - 1) || (b.lowcmd && b.low[env].hold) || b.lowrl);   // 깊이는 묶음의 마지막 걸음에만 (아래층 단독 과제: 결정이 끝난 걸음)
  shape_obs(b, e, env, sh.reach);   // approach·front_cells 보상 (GPU env_reach_kernel 과 같은 때)
  b.st[env] = e;
}

}  // namespace

void set_cpu_threads(int n) { g_cpu_threads = std::max(1, n); }
void set_cpu_tap(CpuTap* t) { g_tap = t; }

void hier_plan_cpu(const EnvBatch& b, EnvState& e, int env, int slot) { plan_cpu(b, e, env, slot); }
// 깊이 카메라 장애물 층 갱신 — include/obs_ref.h camlayer_ref (실물 노드 limo_fire_rl 도 같은 함수)
void hier_camlayer_cpu(const EnvBatch& b, const EnvState& e, int env) { camlayer_ref(b, e, env); }

static_assert(kLrNearCell == mdp::obsgen::kNearCell, "low_rl.h 가까운 지도 픽셀 크기");
// 아래층 RL 준비 단계 (include/low_rl.h 머리말 — GPU env_low_prepare_gpu 와 같은 차례): [새 목표] → 깊이 카메라 층·경로 계획(차례일 때만) → 정책 관측
// 목적지만 목표 뽑기의 로봇 덩어리 판 [size][fmap_row] (goal_reach.h): P = 상자(낱말 범위) 안 설 수 있음(계획 반경) & 깊이 카메라 층 아님, 시작 칸에서 4 이웃 큐 BFS
static std::vector<uint32_t> goal_reach_cpu(const EnvBatch& b, const EnvState& e, int env, mdp::Robot rb) {
  const EnvCfg& c = b.cfg;
  const MapDev& m = b.maps[e.map];
  const int ws = fmap_row(c);
  std::vector<uint32_t> R((size_t)c.size * ws, 0u);
  int by0, by1, bw0, bw1;
  if (!goal_box(e, &by0, &by1, &bw0, &bw1)) return R;
  const mdp::MapView mv{b.lmap + (size_t)env * c.size * c.size, c.size, m.gw, m.gh};
  const auto pass = [&](int x, int y) -> bool {
    if (x < 0 || y < 0 || x >= m.gw || y >= m.gh) return false;
    if (y < by0 || y > by1 || (x >> 5) < bw0 || (x >> 5) > bw1) return false;
    return mdp::standable(mv, x, y, rb) && !cam_lethal(b, env, x, y);
  };
  int rx, ry;
  if (cell_of(c, m, e.x + e.ex, e.y + e.ey, &rx, &ry) < 0) return R;
  std::vector<int32_t> qa(mdp::kReachCap), qb(mdp::kReachCap);
  const int s = goal_start(m.gw, m.gh, rx, ry, pass, qa.data(), qb.data(), mdp::kReachCap);
  if (s < 0) return R;
  std::vector<int> q{s};
  R[(size_t)(s / m.gw) * ws + ((s % m.gw) >> 5)] |= 1u << ((s % m.gw) & 31);
  for (size_t h = 0; h < q.size(); ++h) {
    const int x = q[h] % m.gw, y = q[h] / m.gw;
    for (int k = 0; k < 4; ++k) {
      int dx, dy;
      mdp::nb4(k, &dx, &dy);
      const int u = x + dx, v = y + dy;
      if (!pass(u, v)) continue;
      uint32_t& w = R[(size_t)v * ws + (u >> 5)];
      if ((w >> (u & 31)) & 1u) continue;
      w |= 1u << (u & 31);
      q.push_back(v * m.gw + u);
    }
  }
  return R;
}

static void low_prepare_env(const EnvBatch& b, const float* actions, int env) {
  const EnvCfg& c = b.cfg;
  EnvState e = b.st[env];
  if (c.low_auto == 2 && low_xenv(b, env)) {   // 탐사 판 학습: 위층 0 번 후보 규칙
    lowrl_goal_explore(b, e, env);
  } else if (c.low_auto && b.lowrl[env].need_goal) {
    const mdp::Robot rb = path_rb_plan(c, mdp::obsgen::reach_robot(c));
    std::vector<uint32_t> reach;
    if (c.lo_goalonly && c.low_auto == 1) reach = goal_reach_cpu(b, e, env, rb);   // 목적지만 · 학습: 로봇과 같은 덩어리 (goal_reach.h — GPU goal_reach_gpu 와 같은 칸 집합)
    uint64_t key = ~0ull;
    for (int j = 0; j < kLrProbes; ++j) {
      const uint64_t k = lowrl_goal_probe(b, e, env, j, rb, reach.empty() ? nullptr : reach.data());
      key = k < key ? k : key;
    }
    lowrl_goal_set(b, e, env, key);
  }
  if (lowrl_cam_due(b, e, env) && !lowrl_due(b, e, env)) hier_camlayer_cpu(b, e, env);   // lo_cam_every: 계획 차례가 아닌 걸음에도 층만 (env_camlayer_kernel 과 같은 규칙)
  if (lowrl_due(b, e, env)) {
    hier_camlayer_cpu(b, e, env);
    if (lowrl_replan(b, e, env))   // 목적지만은 경로를 안 짬 (env_kernel.cu env_plan_kernel 과 같은 규칙)
      plan_cpu(b, e, env, lowrl_policy(b, env) ? 0 : (actions && c.low_auto != 2 ? slot_of(b, actions[(size_t)lowrl_act_w(b) * env]) : -1));
    else if (lowrl_go_policy(b, env) && lowrl_goal_new(b, env))   // 목적지만 정책 환경: 새 목표면 goal_cell 만 (goal_reach.h goal_only_issue)
      goal_only_issue(b, e, env);
    lowrl_planned(b, e, env);
  }
  lowrl_pack(b, e, env);
  lowrl_body_clear(b, e, env);   // 몸 둘레 간격 (켰을 때만 — GPU 는 env_low_pack_kernel 이 블록으로 나눠)
  const int n = low_img_n(c);
  for (int i = 0; i < kLowImg; ++i) b.low_img[(size_t)env * n + i] = lowrl_img(b, e, env, i);
  if (c.lo_wide) {
    for (int k = 0; k < (c.size / 4) * (c.size / 4); ++k) lowrl_wred(b, e, env, k);
    for (int k = 0; k < kLowWideHW * kLowWideHW; ++k) lowrl_wide(b, e, env, k, b.low_img + (size_t)env * n + kLowWideOff);
  }
  b.st[env] = e;
}
void env_low_prepare_cpu(const EnvBatch& b, const float* actions) {
  if (!b.lowrl) return;
  for_envs(b.n_env, [&](int env) { low_prepare_env(b, actions, env); });
}

void env_step_cpu(const EnvBatch& b, const float* actions) {
  const EnvCfg& c = b.cfg;
  if (c.action_mode != kActPath) {
    for_envs(b.n_env, [&](int env) { step_env(b, actions, env, -1); });
    return;
  }
  if (b.lowrl) {   // 아래층 RL: 한 번 = 걸음 하나 (계획은 준비 단계가 — 걸음 끝, 관측 앞)
    for_envs(b.n_env, [&](int env) {
      step_env(b, actions, env, 0);
      b.reward[env] = b.racc[env];
      for (int k = 0; k < mdp::rew::kTerms; ++k) b.reward_terms[(size_t)env * mdp::rew::kTerms + k] = b.tacc[(size_t)env * mdp::rew::kTerms + k];
      if (b.done[env] != 0) {
        Shared sh;
        Shared& g = b.scratch[env];
        EnvState e = b.st[env];
        if (c.end_hold) { e.done = 0; e.halt = 1; }   // 판 끝 뒤 멈춤: 리셋 대신 표시만 (GPU env_bundle_end_kernel 과 같게)
        else reset_env(b, e, env);
        write_obs(b, e, env, sh, mdp::obsgen::far_dirty_take(b, e, g, true));
        b.st[env] = e;
      }
      if (c.low_auto) low_prepare_env(b, actions, env);
    });
    return;
  }
  // 계층형: GPU env_step_gpu 와 같은 차례 — 결정(경로 계획) → decide_steps 걸음 → 묶음 끝
  for_envs(b.n_env, [&](int env) {
    EnvState e = b.st[env];
    hier_camlayer_cpu(b, e, env);   // 깊이 카메라 장애물 층 (cam_on 일 때만) — 계획 앞, 지난 묶음 끝 깊이 장으로
    plan_cpu(b, e, env, slot_of(b, actions[(size_t)mdp::act::DIM * env]));
    b.st[env] = e;
    if (b.gdec) {   // 그래프 모드: 결정 하나 = 노드 하나 이동 — 닿거나 실패·시간 초과까지 (graph_env.h gdec_step)
      for (int s = 0; s < c.graph_max_steps && (s == 0 || b.gdec[env].reason == kGdRun); ++s) step_env(b, actions, env, s);
      // 묶음 끝 (GPU env_graph_kernel gate 1 과 같은 차례): 안 끝난 환경은 다음 결정 관측 + 결정 결과·전작 방식 판 끝, 걸음에서 끝난 환경은 결과만
      if (b.done[env] == 0) graph_ref(b, b.st[env], env, true);
      else graph_finish(b, b.st[env], env, -1, 0);
    } else {
      for (int s = 0; s < c.decide_steps; ++s) step_env(b, actions, env, s);
    }
    b.reward[env] = b.racc[env];
    for (int k = 0; k < mdp::rew::kTerms; ++k) b.reward_terms[(size_t)env * mdp::rew::kTerms + k] = b.tacc[(size_t)env * mdp::rew::kTerms + k];
    if (b.done[env] == 0) return;
    Shared sh;
    Shared& g = b.scratch[env];
    e = b.st[env];
    reset_env(b, e, env);
    write_obs(b, e, env, sh, mdp::obsgen::far_dirty_take(b, e, g, true));
    b.st[env] = e;
    graph_ref(b, b.st[env], env);
  });
}

void env_reset_cpu(const EnvBatch& b) {
  for_envs(b.n_env, [&](int env) {
    Shared sh;
    EnvState e = b.st[env];
    reset_env(b, e, env);
    write_obs(b, e, env, sh, mdp::obsgen::FarDirty{1, 0.f, 0.f, 0.f, 0.f});
    shape_obs(b, e, env, sh.reach);   // 판 첫 관측: 값만 적음
    b.scratch[env].far_ok = 0;   // GPU 리셋 커널과 같게: 다음 걸음은 다시 전부
    b.st[env] = e;
    graph_ref(b, b.st[env], env);
    b.reward[env] = 0.f;
    b.done[env] = 0;
    for (int k = 0; k < mdp::rew::kTerms; ++k) b.reward_terms[(size_t)env * mdp::rew::kTerms + k] = 0.f;
    if (b.lowrl) {
      lowrl_clear(b.lowrl[env]);
      if (b.cfg.low_auto) low_prepare_env(b, nullptr, env);
    }
  });
}

void env_graph_cpu(const EnvBatch& b) {
  for_envs(b.n_env, [&](int env) { graph_ref(b, b.st[env], env); });
}

}  // namespace sim
