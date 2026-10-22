#include <cuda_runtime.h>

#include <stdexcept>

#include "fenv_api.h"

namespace fe {

namespace {
__global__ void k_init(Core* core, uint64_t seed, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) init_one(core[i], seed, i);
}
__global__ void k_reset(Core* core, uint8_t* know, uint32_t stride, fm::View v, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  Core c = core[i];
  reset(c, v, know + (size_t)i * stride);
  core[i] = c;
}
__global__ void k_step(Core* core, uint8_t* know, uint32_t stride, fm::View v, const int32_t* act, float* rew, uint32_t* done, EpisodeEnd* ends, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  Core c = core[i];
  step_one(c, act[i], v, know + (size_t)i * stride, rew + i, done + i, ends + i);
  core[i] = c;
}
void check(cudaError_t e, const char* what) {
  if (e != cudaSuccess) throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(e));
}
constexpr int TB = 128;
}  // namespace

GpuEnvs gpu_create(const fm::View& dv, uint32_t max_cells, int n, uint64_t seed) {
  GpuEnvs e;
  e.n = n; e.know_stride = max_cells; e.view = dv;
  check(cudaMalloc(&e.core, sizeof(Core) * n), "core");
  check(cudaMalloc(&e.know, (size_t)max_cells * n), "know");
  check(cudaMalloc(&e.action, sizeof(int32_t) * n), "action");
  check(cudaMalloc(&e.reward, sizeof(float) * n), "reward");
  check(cudaMalloc(&e.done, sizeof(uint32_t) * n), "done");
  check(cudaMalloc(&e.ends, sizeof(EpisodeEnd) * n), "ends");
  cudaMemset(e.know, 0, (size_t)max_cells * n);
  cudaMemset(e.action, 0, sizeof(int32_t) * n);
  k_init<<<(n + TB - 1) / TB, TB>>>(e.core, seed, n);
  check(cudaGetLastError(), "init");
  return e;
}
void gpu_reset_all(GpuEnvs& e) {
  k_reset<<<(e.n + TB - 1) / TB, TB>>>(e.core, e.know, e.know_stride, e.view, e.n);
  check(cudaGetLastError(), "reset");
}
void gpu_step(GpuEnvs& e) {
  k_step<<<(e.n + TB - 1) / TB, TB>>>(e.core, e.know, e.know_stride, e.view, e.action, e.reward, e.done, e.ends, e.n);
  check(cudaGetLastError(), "step");
}
void gpu_destroy(GpuEnvs& e) {
  cudaFree(e.core); cudaFree(e.know); cudaFree(e.action); cudaFree(e.reward); cudaFree(e.done); cudaFree(e.ends);
  e = GpuEnvs{};
}

}  // namespace fe
