// 빌드 (젯슨, ~/decider_test 처럼 따로 둔 폴더에서): g++ -O2 -std=c++17 -I<network> -I<explore> neon_bench.cpp -o bench_neon · 스칼라 판은 -DXNET_CPU_SCALAR 를 더해 bench_scalar
//   bench_scalar 정책.bin s.bin 30 && bench_neon 정책.bin n.bin 30 && cmp s.bin n.bin   (2026-11-15 Orin Nano: 6.94 → 2.84 ms/act, 출력 비트 같음)
// 실물 CPU 정책(policy_cpu.h) 시간·출력 덤프 — 같은 무작위 관측(씨앗 고정)에서 로짓을 파일로. 스칼라(-DXNET_CPU_SCALAR) 판과 NEON 판 덤프를 cmp 로 견줌
#include <chrono>
#include <cstdio>
#include <random>
#include <vector>
#include "policy_cpu.h"
int main(int argc, char** argv) {
  if (argc < 3) { std::fprintf(stderr, "neon_bench ckpt out.bin [n]\n"); return 2; }
  const int n = argc > 3 ? std::atoi(argv[3]) : 50;
  xnet_cpu::HierPolicy p;
  p.load(argv[1]);
  std::mt19937 rng(7);
  std::uniform_int_distribution<int> U8(0, 255);
  std::normal_distribution<float> G(0.f, 1.f);
  std::vector<uint8_t> img(mdp::obs::IMG_SIZE);
  std::vector<float> o1(mdp::hier::POLICY_1D);
  std::vector<float> logits(mdp::hier::K), mask(mdp::hier::K);
  FILE* f = std::fopen(argv[2], "wb");
  double t = 0;
  int sum = 0;
  for (int r = 0; r < n; ++r) {
    for (auto& v : img) v = (uint8_t)(U8(rng) % 3 == 0 ? 0 : U8(rng));
    for (auto& v : o1) v = G(rng);
    for (int k = 0; k < mdp::hier::K; ++k) o1[mdp::obs::POLICY_1D + k * mdp::hier::FEAT + mdp::hier::FEAT - 1] = (r + k) % 3 ? 1.f : 0.f;
    const auto t0 = std::chrono::steady_clock::now();
    const int a = p.act(img.data(), o1.data(), logits.data(), mask.data());
    t += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    sum += a;
    std::fwrite(logits.data(), 4, logits.size(), f);
  }
  std::fclose(f);
  std::printf("%s: %d 번 평균 %.3f ms/act (결정 합 %d)\n",
#ifdef XNET_CPU_NEON
              "NEON",
#else
              "스칼라",
#endif
              n, t / n, sum);
  return 0;
}
