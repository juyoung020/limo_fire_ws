// 처리량: fenv_bench <fmap 폴더> [맵 수=1000] [환경 수=16384] [행동 수=200]
#include <cuda_runtime.h>

#include <chrono>
#include <cstdio>
#include <string>
#include <vector>

#include "fenv_api.h"

__global__ void k_rand_actions(int32_t* a, int n, uint64_t seed) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  uint64_t s = seed ^ (0xD1B54A32D192ED03ull * (uint64_t)(i + 1));
  a[i] = (int32_t)(dm::splitmix64(s) % fe::K::n_actions);
}

int main(int argc, char** argv) {
  if (argc < 2) { std::fprintf(stderr, "fenv_bench <fmap 폴더> [맵 수] [환경 수] [행동 수]\n"); return 2; }
  size_t n_maps = argc > 2 ? std::stoul(argv[2]) : 1000;
  int n = argc > 3 ? std::stoi(argv[3]) : 16384;
  int T = argc > 4 ? std::stoi(argv[4]) : 200;
  fm::Bank bank = fm::load_bank(argv[1], "", n_maps);
  fm::DeviceBank d = fm::upload(bank);
  fe::GpuEnvs g = fe::gpu_create(d.view, bank.max_cells, n, 1);
  fe::gpu_reset_all(g);
  cudaDeviceSynchronize();
  auto t0 = std::chrono::steady_clock::now();
  for (int t = 0; t < T; ++t) {
    k_rand_actions<<<(n + 127) / 128, 128>>>(g.action, n, (uint64_t)t * 7919);
    fe::gpu_step(g);
  }
  cudaDeviceSynchronize();
  double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  std::printf("맵 %zu, 환경 %d, 행동 %d: %.2f s → %.3g env-step/s (1 env-step = 0.5 s 시뮬 = 서브스텝 10 + 라이다 248 빔), 아는 지도 %.2f GB\n",
              bank.names.size(), n, T, s, (double)n * T / s, (double)bank.max_cells * n / 1e9);
  fe::gpu_destroy(g);
  fm::free_device(d);
  return 0;
}
