// GPU == CPU 비트 동일 검증 + 판 통계.
// fenv_verify <fmap 폴더> [맵 수=300] [환경 수=512] [행동 수=400] [--negative]
// 행동은 무작위(같은 난수열을 양쪽에). --negative 는 GPU 쪽 행동 하나를 바꿔서 일부러 어긋나게 한다(검사가 틀림을 잡는지).
#include <cuda_runtime.h>

#include <cstdio>
#include <cstring>
#include <string>

#include "fenv_api.h"

int main(int argc, char** argv) {
  if (argc < 2) { std::fprintf(stderr, "fenv_verify <fmap 폴더> [맵 수] [환경 수] [행동 수] [--negative]\n"); return 2; }
  std::string dir = argv[1];
  size_t n_maps = argc > 2 ? std::stoul(argv[2]) : 300;
  int n = argc > 3 ? std::stoi(argv[3]) : 512;
  int T = argc > 4 ? std::stoi(argv[4]) : 400;
  bool negative = false;
  for (int i = 1; i < argc; ++i) if (!std::strcmp(argv[i], "--negative")) negative = true;

  // 세 소스를 고루
  fm::Bank bank = fm::load_bank(dir, "houseexpo", n_maps / 3);
  for (const char* s : {"procthor", "duct"}) {
    fm::Bank b2 = fm::load_bank(dir, s, n_maps / 3);
    size_t base_idx = bank.idx.size();
    uint64_t base_cell = bank.cell.size();
    for (size_t m = 0; m < b2.names.size(); ++m) {
      bank.names.push_back(b2.names[m]);
      bank.off.push_back(b2.off[m] + base_cell);
      bank.w.push_back(b2.w[m]); bank.h.push_back(b2.h[m]); bank.ox.push_back(b2.ox[m]); bank.oy.push_back(b2.oy[m]);
      bank.free_off.push_back(b2.free_off[m] + (uint32_t)base_idx); bank.free_n.push_back(b2.free_n[m]);
      bank.start_off.push_back(b2.start_off[m] + (uint32_t)base_idx); bank.start_n.push_back(b2.start_n[m]);
      bank.n_target.push_back(b2.n_target[m]); bank.area.push_back(b2.area[m]);
    }
    bank.cell.insert(bank.cell.end(), b2.cell.begin(), b2.cell.end());
    bank.idx.insert(bank.idx.end(), b2.idx.begin(), b2.idx.end());
    bank.max_cells = std::max(bank.max_cells, b2.max_cells);
  }
  std::printf("맵 %zu 개, 칸 %.1f M, 환경 %d, 행동 %d, 아는 지도 %.1f MB\n", bank.names.size(), bank.cell.size() / 1e6, n, T,
              (double)bank.max_cells * n / 1e6);

  fm::DeviceBank dbank = fm::upload(bank);
  fe::GpuEnvs g = fe::gpu_create(dbank.view, bank.max_cells, n, 12345);
  fe::CpuEnvs c = fe::cpu_create(bank.host_view(), bank.max_cells, n, 12345);
  fe::gpu_reset_all(g);
  fe::cpu_reset_all(c);

  std::vector<fe::Core> gc(n);
  std::vector<float> gr(n);
  std::vector<uint32_t> gd(n);
  std::vector<fe::EpisodeEnd> ge(n);
  uint64_t rng = 777;
  long long mismatch = 0;
  long long ends[4] = {0, 0, 0, 0};
  double cov_sum = 0, fire_seen = 0, fire_total = 0;
  for (int t = 0; t < T; ++t) {
    for (int i = 0; i < n; ++i) c.action[i] = (int32_t)(dm::splitmix64(rng) % fe::K::n_actions);
    std::vector<int32_t> ga = c.action;
    if (negative && t == T / 2) ga[0] = (ga[0] + 1) % fe::K::n_actions;
    cudaMemcpy(g.action, ga.data(), sizeof(int32_t) * n, cudaMemcpyHostToDevice);
    fe::gpu_step(g);
    fe::cpu_step(c);
    cudaMemcpy(gr.data(), g.reward, sizeof(float) * n, cudaMemcpyDeviceToHost);
    cudaMemcpy(gd.data(), g.done, sizeof(uint32_t) * n, cudaMemcpyDeviceToHost);
    cudaMemcpy(gc.data(), g.core, sizeof(fe::Core) * n, cudaMemcpyDeviceToHost);
    cudaMemcpy(ge.data(), g.ends, sizeof(fe::EpisodeEnd) * n, cudaMemcpyDeviceToHost);
    for (int i = 0; i < n; ++i) {
      if (std::memcmp(&gr[i], &c.reward[i], 4) || gd[i] != c.done[i] || std::memcmp(&gc[i], &c.core[i], sizeof(fe::Core))) ++mismatch;
      if (c.done[i]) {
        ++ends[c.done[i]];
        cov_sum += c.ends[i].coverage;
        fire_seen += c.ends[i].fires_seen;
        fire_total += c.ends[i].n_fire;
      }
    }
  }
  std::vector<uint8_t> gk((size_t)bank.max_cells * n);
  cudaMemcpy(gk.data(), g.know, gk.size(), cudaMemcpyDeviceToHost);
  long long know_diff = 0;
  for (size_t k = 0; k < gk.size(); ++k) know_diff += gk[k] != c.know[k];
  long long total = ends[1] + ends[2] + ends[3];
  std::printf("끝난 판 %lld: 성공 %lld / 충돌 %lld / 시간초과 %lld, 평균 덮은 비율 %.3f, 불 본 비율 %.3f\n", total, ends[1], ends[2], ends[3],
              total ? cov_sum / total : 0.0, fire_total ? fire_seen / fire_total : 0.0);
  std::printf("불일치: 행동 단위 %lld, 아는 지도 칸 %lld\n", mismatch, know_diff);
  bool same = mismatch == 0 && know_diff == 0;
  fe::gpu_destroy(g);
  fm::free_device(dbank);
  if (negative) { std::printf(same ? "음성 대조 실패: 어긋남을 못 잡음\n" : "음성 대조 통과: 어긋남을 잡음\n"); return same ? 1 : 0; }
  std::printf(same ? "통과: GPU == CPU 비트 동일\n" : "실패: 비트가 다름\n");
  return same ? 0 : 1;
}
