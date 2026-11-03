// 환경 한 걸음 GPU 커널: 블록 하나 = 환경 하나. 단계 순서는 src/env_ref.cpp 와 한 줄씩 같아야 한다 (비트 동일)
#include "env_api.h"

namespace sim {
namespace {

__device__ void reset_env(const EnvBatch& b, EnvState& e, int env, int tid) {
  reset_map(b, env, tid);
  if (tid == 0) reset_state(b, e);
  __syncthreads();
}

__global__ void env_step_kernel(EnvBatch b, const float* actions) {
  const int env = blockIdx.x, tid = threadIdx.x;
  __shared__ EnvState e;
  __shared__ Shared sh;
  __shared__ int new_known;
  const EnvCfg& c = b.cfg;
  if (tid == 0) {
    e = b.st[env];
    e.collided = 0;
    apply_action(b, e, actions + 2 * env);
    new_known = 0;
  }
  __syncthreads();
  const int tick0 = e.tick;   // 모든 스레드가 같은 값으로 라이다 걸음을 정한다 (스레드 0 이 e.tick 을 바꾸는 중에 읽지 않게)
  for (int k = 0; k < c.substeps; ++k) {
    if (tid == 0) {
      if (!e.collided) physics_step(b, e);
      e.tick += 1;
    }
    __syncthreads();
    if ((tick0 + k + 1) % c.lidar_every != 0) continue;
    if (tid == 0) scan_begin(b, e, sh);
    __syncthreads();
    for (int j = tid; j < sh.n_beams; j += kThreads) scan_beam(b, e, env, sh, j);
    __syncthreads();
    scan_apply(b, e, env, sh, tid);
    __syncthreads();
    if (tid == 0) {
      int s = 0;
      for (int i = 0; i < kThreads; ++i) s += sh.part[i];
      new_known += s;
    }
    __syncthreads();
  }
  if (tid == 0) {
    finish_step(b, e, sh, new_known);
    b.reward[env] = e.reward;
    b.done[env] = e.done;
  }
  __syncthreads();
  if (e.done) reset_env(b, e, env, tid);
  write_obs(b, e, env, tid);
  if (tid == 0) b.st[env] = e;
}

__global__ void env_reset_kernel(EnvBatch b) {
  const int env = blockIdx.x, tid = threadIdx.x;
  __shared__ EnvState e;
  if (tid == 0) e = b.st[env];
  __syncthreads();
  reset_env(b, e, env, tid);
  write_obs(b, e, env, tid);
  if (tid == 0) {
    b.st[env] = e;
    b.reward[env] = 0.f;
    b.done[env] = 0;
  }
}

}  // namespace

void env_step_gpu(const EnvBatch& b, const float* actions, cudaStream_t st) {
  env_step_kernel<<<b.n_env, kThreads, 0, st>>>(b, actions);
}

void env_reset_gpu(const EnvBatch& b, cudaStream_t st) { env_reset_kernel<<<b.n_env, kThreads, 0, st>>>(b); }

}  // namespace sim
