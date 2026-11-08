// 환경 한 걸음 CPU 참조판: GPU 커널(src/env_kernel.cu)의 블록 하나를 스레드 순서대로 흉내 낸다. 단계 순서는 커널과 한 줄씩 같다
#include <algorithm>
#include <thread>
#include <vector>

#include "env_api.h"

namespace sim {
namespace {

int g_cpu_threads = 1;

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

// 도달 가능한 프론티어 요약 (mdp/frontier.h reach_summary = 기준선 frontier_search 그대로). GPU 는 env_kernel.cu reach_block — 같은 값
void reach_obs(const EnvBatch& b, const EnvState& e, int env, Shared& sh) {
  const EnvCfg& c = b.cfg;
  const MapDev& m = b.maps[e.map];
  mdp::reach_clear(&sh.reach);
  if (e.kb_x1 < 0) return;   // 아는 칸 없음
  const int16_t* L = b.lmap + (size_t)env * c.size * c.size;
  const mdp::MapView mv{L, c.size, m.gw, m.gh};
  int n_tot = 0;   // 지도 전체 프론티어 칸 (모두 건드린 범위 안)
  for (int y = e.kb_y0; y <= e.kb_y1; ++y)
    for (int x = e.kb_x0; x <= e.kb_x1; ++x) n_tot += mdp::frontier_cell(mv, x, y) ? 1 : 0;
  thread_local std::vector<uint8_t> flag;
  thread_local std::vector<int32_t> qa, qb, dist;
  thread_local std::vector<mdp::Frontier> out;
  const size_t n = (size_t)m.gw * m.gh;
  if (flag.size() < n) { flag.resize(n); qa.resize(n); qb.resize(n); dist.resize(n); }
  if (out.size() < (size_t)mdp::kReachCap + 1) out.resize(mdp::kReachCap + 1);
  int rx, ry;
  cell_of(c, m, e.x + e.ex, e.y + e.ey, &rx, &ry);
  mdp::reach_summary(mv, rx, ry, mdp::obsgen::reach_robot(c), mdp::obsgen::reach_min_cells(c), mdp::SearchBuf{flag.data(), qa.data(), qb.data(), dist.data()},
                     out.data(), (int)out.size(), n_tot, &sh.reach);
}

void write_obs(const EnvBatch& b, EnvState& e, int env, Shared& sh, const mdp::obsgen::FarDirty& fd) {
  reach_obs(b, e, env, sh);
  for (int tid = 0; tid < kThreads; ++tid) mdp::obsgen::obs_cells(b, e, env, sh, tid, fd);
  for (int i = 0; i < mdp::obs::DEPTH_SIZE; ++i) mdp::obsgen::obs_depth(b, e, env, i);
  mdp::obsgen::obs_finish(b, e, env, sh);
}

}  // namespace

void set_cpu_threads(int n) { g_cpu_threads = std::max(1, n); }

void env_step_cpu(const EnvBatch& b, const float* actions) {
  const EnvCfg& c = b.cfg;
  for_envs(b.n_env, [&](int env) {
    Shared sh;
    Shared& g = b.scratch[env];   // 걸음을 넘어 남는 칸 (먼 지도 다시 그릴 곳 — GPU 와 같은 증분)
    mdp::obsgen::far_dirty_clear(g);
    EnvState e = b.st[env];
    e.collided = 0;
    apply_action(b, e, env, actions + mdp::act::DIM * env);
    int new_wall = 0, new_floor = 0, d_known = 0, n_new = 0, n_occ = 0;
    sh.scanned = 0;
    const int tick0 = e.tick;
    for (int k = 0; k < c.substeps; ++k) {
      if (!e.collided) physics_step(b, e);
      e.tick += 1;
      if ((tick0 + k + 1) % c.lidar_every != 0) continue;
      scan_begin(b, e, sh);
      for (int j = 0; j < sh.n_beams; ++j) scan_beam(b, e, env, sh, j);
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
    finish_step(b, e, env, new_wall, new_floor, d_known, n_new);
    mdp::obsgen::far_dirty_robot(b, e, g);
    b.reward[env] = e.reward;
    b.done[env] = e.done;
    if (sh.scanned) e.lid_head = (e.lid_head + 1) % kLidarRing;
    if (sh.scanned)
      for (int tid = 0; tid < kThreads; ++tid) push_bins(b, e, env, sh, tid);
    const bool was_done = e.done != 0;
    if (was_done) reset_env(b, e, env);
    write_obs(b, e, env, sh, mdp::obsgen::far_dirty_take(b, e, g, was_done));
    b.st[env] = e;
  });
}

void env_reset_cpu(const EnvBatch& b) {
  for_envs(b.n_env, [&](int env) {
    Shared sh;
    EnvState e = b.st[env];
    reset_env(b, e, env);
    write_obs(b, e, env, sh, mdp::obsgen::FarDirty{1, 0.f, 0.f, 0.f, 0.f});
    b.scratch[env].far_ok = 0;   // GPU 리셋 커널과 같게: 다음 걸음은 다시 전부
    b.st[env] = e;
    b.reward[env] = 0.f;
    b.done[env] = 0;
    for (int k = 0; k < mdp::rew::kTerms; ++k) b.reward_terms[(size_t)env * mdp::rew::kTerms + k] = 0.f;
  });
}

}  // namespace sim
