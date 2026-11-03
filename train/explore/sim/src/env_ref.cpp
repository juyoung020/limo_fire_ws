// 환경 한 걸음 CPU 참조판: GPU 커널(src/env_kernel.cu)의 블록 하나를 스레드 순서대로 흉내 낸다. 단계 순서는 커널과 한 줄씩 같다
#include "env_api.h"

namespace sim {
namespace {

void reset_env(const EnvBatch& b, EnvState& e, int env) {
  for (int tid = 0; tid < kThreads; ++tid) reset_map(b, env, tid);
  reset_state(b, e);
}

void write_obs(const EnvBatch& b, EnvState& e, int env, Shared& sh) {
  sh.fr_min = mdp::kNoFrontier;
  const mdp::obsgen::FarDirty all{1, 0.f, 0.f, 0.f, 0.f};
  for (int tid = 0; tid < kThreads; ++tid) mdp::obsgen::obs_cells(b, e, env, sh, tid, all);
  for (int i = 0; i < mdp::obs::DEPTH_SIZE; ++i) mdp::obsgen::obs_depth(b, e, env, i);
  mdp::obsgen::obs_finish(b, e, env, sh);
}

}  // namespace

void env_step_cpu(const EnvBatch& b, const float* actions) {
  const EnvCfg& c = b.cfg;
  Shared sh;
  for (int env = 0; env < b.n_env; ++env) {
    EnvState e = b.st[env];
    e.collided = 0;
    apply_action(b, e, env, actions + mdp::act::DIM * env);
    int new_wall = 0, new_floor = 0, d_known = 0, n_new = 0;
    sh.scanned = 0;
    const int tick0 = e.tick;
    for (int k = 0; k < c.substeps; ++k) {
      if (!e.collided) physics_step(b, e);
      e.tick += 1;
      if ((tick0 + k + 1) % c.lidar_every != 0) continue;
      scan_begin(b, e, sh);
      for (int j = 0; j < sh.n_beams; ++j) scan_beam(b, e, env, sh, j);
      int s = 0, f = 0, kn = 0, nn = 0;
      for (int tid = 0; tid < kThreads; ++tid) {
        const ApplyCount a = scan_apply(b, e, env, sh, tid, kThreads);
        s += a.wall;
        f += a.floor;
        kn += a.known;
        nn += a.fresh;
      }
      new_wall += s;
      new_floor += f;
      d_known += kn;
      n_new += nn;
      known_box(b, e, sh);
    }
    finish_step(b, e, env, new_wall, new_floor, d_known, n_new);
    b.reward[env] = e.reward;
    b.done[env] = e.done;
    if (sh.scanned) e.lid_head = (e.lid_head + 1) % kLidarRing;
    if (sh.scanned)
      for (int tid = 0; tid < kThreads; ++tid) push_bins(b, e, env, sh, tid);
    if (e.done) reset_env(b, e, env);
    write_obs(b, e, env, sh);
    b.st[env] = e;
  }
}

void env_reset_cpu(const EnvBatch& b) {
  Shared sh;
  for (int env = 0; env < b.n_env; ++env) {
    EnvState e = b.st[env];
    reset_env(b, e, env);
    write_obs(b, e, env, sh);
    b.st[env] = e;
    b.reward[env] = 0.f;
    b.done[env] = 0;
    for (int k = 0; k < mdp::rew::kTerms; ++k) b.reward_terms[(size_t)env * mdp::rew::kTerms + k] = 0.f;
  }
}

}  // namespace sim
