// 빌드 (젯슨, ~/decider_test 처럼 따로 둔 폴더에서): g++ -O2 -std=c++17 -I<network> -I<explore> low_bench.cpp -o low_neon · 스칼라 판은 -DXNET_CPU_SCALAR 를 더해 low_scalar
//   low_scalar 정책.bin s.bin 40 && low_neon 정책.bin n.bin 40 && cmp s.bin n.bin
// 실물 아래층 CPU 정책(policy_cpu.h LowPolicy) 시간·출력 덤프 — 같은 무작위 관측(씨앗 고정, 10 걸음마다 판 끝 → h 초기화)에서 행동 출력과 h 를 파일로
#include <chrono>
#include <cstdio>
#include <random>
#include <vector>
#include "policy_cpu.h"
int main(int argc, char** argv) {
  if (argc < 3) { std::fprintf(stderr, "low_bench ckpt out.bin [n]\n"); return 2; }
  const int n = argc > 3 ? std::atoi(argv[3]) : 40;
  xnet_cpu::LowPolicy p;
  p.load(argv[1]);
  std::mt19937 rng(7);
  std::uniform_int_distribution<int> U8(0, 255);
  std::normal_distribution<float> G(0.f, 1.f);
  std::vector<uint8_t> img(mdp::low::IMG_WIDE);
  std::vector<float> o1(mdp::low::N1D), out(mdp::low::ACT);
  FILE* f = std::fopen(argv[2], "wb");
  double t = 0;
  for (int r = 0; r < n; ++r) {
    if (r % 10 == 0) p.reset();
    for (auto& v : img) v = (uint8_t)(U8(rng) % 3 == 0 ? 0 : U8(rng));
    for (auto& v : o1) v = G(rng);
    const auto t0 = std::chrono::steady_clock::now();
    p.act(img.data(), o1.data(), out.data());
    t += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    std::fwrite(out.data(), 4, out.size(), f);
    std::fwrite(p.h().data(), 4, p.h().size(), f);
  }
  std::fclose(f);
#ifdef XNET_CPU_NEON
  const char* kind = "NEON";
#else
  const char* kind = "스칼라";
#endif
  std::printf("%s: %d 걸음 평균 %.3f ms/act (GRU %d)\n", kind, n, t / n, p.hidden());
  return 0;
}
