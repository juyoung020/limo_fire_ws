// 환경 한 걸음 CPU 참조판: GPU 커널(src/env_kernel.cu)의 블록 하나를 스레드 순서대로 흉내 낸다. 단계 순서는 커널과 한 줄씩 같다
#include "env_api.h"

namespace sim {
namespace {

void reset_env(const EnvBatch& b, EnvState& e, int env) {
  for (int tid = 0; tid < kThreads; ++tid) reset_map(b, env, tid);
  reset_state(b, e);
}

}  // namespace

void env_step_cpu(const EnvBatch& b, const float* actions) {
  const EnvCfg& c = b.cfg;
  Shared sh;
  for (int env = 0; env < b.n_env; ++env) {
    EnvState e = b.st[env];
    e.collided = 0;
    apply_action(b, e, actions + 2 * env);
    int new_wall = 0, new_floor = 0;
    const int tick0 = e.tick;
    for (int k = 0; k < c.substeps; ++k) {
      if (!e.collided) physics_step(b, e);
      e.tick += 1;
      if ((tick0 + k + 1) % c.lidar_every != 0) continue;
      scan_begin(b, e, sh);
      for (int j = 0; j < sh.n_beams; ++j) scan_beam(b, e, env, sh, j);
      for (int tid = 0; tid < kThreads; ++tid) scan_apply(b, e, env, sh, tid);
      int s = 0, f = 0;
      for (int i = 0; i < kThreads; ++i) { s += sh.part[i]; f += sh.part_floor[i]; }
      new_wall += s;
      new_floor += f;
    }
    finish_step(b, e, new_wall, new_floor);
    b.reward[env] = e.reward;
    b.done[env] = e.done;
    if (e.done) reset_env(b, e, env);
    for (int tid = 0; tid < kThreads; ++tid) write_obs(b, e, env, tid);
    b.st[env] = e;
  }
}

void env_reset_cpu(const EnvBatch& b) {
  for (int env = 0; env < b.n_env; ++env) {
    EnvState e = b.st[env];
    reset_env(b, e, env);
    for (int tid = 0; tid < kThreads; ++tid) write_obs(b, e, env, tid);
    b.st[env] = e;
    b.reward[env] = 0.f;
    b.done[env] = 0;
  }
}

}  // namespace sim
