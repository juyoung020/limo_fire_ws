// 합성곱 융합 커널: 텐서 코어 판 vs fp32 판 — 여러 n(조각 타일이 덜 찬 경우)·색인 읽기에서 출력·기울기 상대 오차, 가지 밖 열을 안 건드리는지.
//   cnn_check   (통과 기준: 상대 < 3e-2 = bf16 반올림 크기, 모든 출력 칸이 쓰임, 바깥 0)
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>
#include "cnn_fused.cuh"
#include "../../mdp/obs_layout.h"
using namespace xnet;
namespace o = mdp::obs;
int main() {
  long feat = 3840 + 1336;
  CnnBranch br[3] = {{6, 64, 64, 16, 32, 32, 0, o::IMG_SIZE, feat, 0},
                     {7, 32, 32, 16, 32, 32, o::NEAR_SIZE, o::IMG_SIZE, feat, 2048},
                     {1, 40, 64, 16, 32, 32, o::NEAR_SIZE + o::FAR_SIZE, o::IMG_SIZE, feat, 2560}};
  int NMAX = 1100;
  uint8_t* img; float *X0a, *X0b, *dX0, *P, *Ga, *Gb, *part; int* idx;
  cudaMalloc(&img, (size_t)NMAX * o::IMG_SIZE);
  cudaMalloc(&X0a, NMAX * feat * 4); cudaMalloc(&X0b, NMAX * feat * 4); cudaMalloc(&dX0, NMAX * feat * 4);
  cudaMalloc(&P, 7000 * 4); cudaMalloc(&Ga, 7000 * 4); cudaMalloc(&Gb, 7000 * 4); cudaMalloc(&idx, NMAX * 4);
  long pf = 0; for (auto& b : br) pf = std::max(pf, cnn_fused_part_floats(b));
  cudaMalloc(&part, pf * 4);
  std::mt19937 rng(5);
  std::vector<uint8_t> hi((size_t)NMAX * o::IMG_SIZE); for (auto& v : hi) v = rng() % 3 == 0 ? 0 : rng() & 255;
  cudaMemcpy(img, hi.data(), hi.size(), cudaMemcpyHostToDevice);
  std::normal_distribution<float> N(0, 1.f);
  std::vector<float> hd(NMAX * feat); for (auto& v : hd) v = N(rng);
  cudaMemcpy(dX0, hd.data(), hd.size() * 4, cudaMemcpyHostToDevice);
  std::vector<int> hx(NMAX); for (int i = 0; i < NMAX; ++i) hx[i] = (i * 37 + 11) % NMAX;
  cudaMemcpy(idx, hx.data(), NMAX * 4, cudaMemcpyHostToDevice);
  int worst = 0; double wmax = 0;
  for (int b = 0; b < 3; ++b) {
    // torch 기본 초기화 크기 U(±1/√fan_in)
    std::vector<float> hp(7000);
    int K[3] = {4 * br[b].cin, 64, 128}, C[3] = {16, 32, 32}; int off = 0;
    for (int l = 0; l < 3; ++l) { float bd = 1 / std::sqrt((float)K[l]); std::uniform_real_distribution<float> U(-bd, bd);
      for (int e = 0; e < C[l] * K[l] + C[l]; ++e) hp[off + e] = U(rng); off += C[l] * K[l] + C[l]; }
    cudaMemcpy(P, hp.data(), off * 4, cudaMemcpyHostToDevice);
    for (int n : {1, 3, 7, 33, 1029}) for (int ui = 0; ui < 2; ++ui) {
      const int* ix = ui ? idx : nullptr;
      cudaMemset(X0a, 0, NMAX * feat * 4); cudaMemset(X0b, 0, NMAX * feat * 4);
      cudaMemset(Ga, 0, 7000 * 4); cudaMemset(Gb, 0, 7000 * 4);
      cnn_fused_forward(br[b], P, img, ix, n, X0a, 0, false);
      cnn_fused_forward(br[b], P, img, ix, n, X0b, 0, true);
      cnn_fused_backward(br[b], P, img, ix, n, dX0, Ga, part, 0, false, false);
      cnn_fused_backward(br[b], P, img, ix, n, dX0, Gb, part, 0, false, true);
      std::vector<float> a(n * feat), c(n * feat), ga(off), gb(off);
      cudaMemcpy(a.data(), X0a, n * feat * 4, cudaMemcpyDeviceToHost); cudaMemcpy(c.data(), X0b, n * feat * 4, cudaMemcpyDeviceToHost);
      cudaMemcpy(ga.data(), Ga, off * 4, cudaMemcpyDeviceToHost); cudaMemcpy(gb.data(), Gb, off * 4, cudaMemcpyDeviceToHost);
      double na = 0, d = 0, outside = 0; long cnt = 0;
      int w = b == 0 ? 2048 : (b == 1 ? 512 : 1280);
      for (int i = 0; i < n; ++i) for (long j = 0; j < feat; ++j) {
        double x = a[i * feat + j], y = c[i * feat + j];
        bool in = j >= br[b].flat_off && j < br[b].flat_off + w;
        if (in) { na += x * x; d += (x - y) * (x - y); cnt += x != 0; } else outside += std::fabs(x) + std::fabs(y);
      }
      double ng = 0, dg = 0; for (int e = 0; e < off; ++e) { ng += ga[e] * ga[e]; dg += (ga[e] - gb[e]) * (ga[e] - gb[e]); }
      double eo = std::sqrt(d / na), eg = std::sqrt(dg / ng);
      printf("가지 %d n %4d 색인 %d: 출력 상대 %.2e (0 아닌 %ld/%ld) 바깥 %g, 기울기 상대 %.2e\n", b, n, ui, eo, cnt, (long)n * w, outside, eg);
      if (eo > wmax) wmax = eo; if (eg > wmax) wmax = eg;
      if (!(eo < 3e-2 && eg < 3e-2 && outside == 0 && cnt == (long)n * w)) worst = 1;
    }
  }
  cudaError_t e = cudaGetLastError(); if (e) { printf("오류 %s\n", cudaGetErrorString(e)); return 1; }
  printf("%s (최대 상대 %.2e)\n", worst ? "실패" : "통과", wmax);
  return worst;
}
