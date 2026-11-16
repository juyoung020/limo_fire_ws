// 합성곱 융합 커널만 재기: 가지 3 개, 정방향·역방향 ms 와 TFLOPS (역방향은 재계산 포함 연산 수).
// 출력은 학습과 같은 가지 버퍼 F3 [n][h3·w3·32] (가지마다 따로).
//   cnn_bench [1 = 텐서 코어 판(기본) / 0 = fp32 판] [n = 12288 (학습 덩이 = 미니배치 하나)]
#include <cstdio>
#include <random>
#include <vector>
#include "cnn_fused.cuh"
#include "../../mdp/obs_layout.h"
using namespace xnet;
namespace o = mdp::obs;
int main(int argc, char** argv) {
  bool tc = argc > 1 ? atoi(argv[1]) : 1;
  int n = argc > 2 ? atoi(argv[2]) : 12288, reps = 20;
  const long feat = 2048;  // 가장 큰 가지의 F3 행 (버퍼 크기용)
  CnnBranch br[3] = {{6, 64, 64, 16, 32, 32, 0, o::IMG_SIZE, 2048, 0},
                     {7, 32, 32, 16, 32, 32, o::NEAR_SIZE, o::IMG_SIZE, 512, 0},
                     {1, 40, 64, 16, 32, 32, o::NEAR_SIZE + o::FAR_SIZE, o::IMG_SIZE, 1280, 0}};
  double macs[3] = {1179648, 311296, 532480};  // 정방향 MAC/샘플
  double bmacs[3];
  double l2[3] = {262144, 65536, 163840}, l0[3] = {393216, 114688, 40960};
  for (int b = 0; b < 3; ++b) bmacs[b] = (macs[b] - l2[b]) + macs[b] + (macs[b] - l0[b]);
  uint8_t* img; float *X0, *dX0, *P, *G, *part;
  cudaMalloc(&img, (size_t)n * o::IMG_SIZE);
  cudaMalloc(&X0, (size_t)n * feat * 4); cudaMalloc(&dX0, (size_t)n * feat * 4);
  cudaMalloc(&P, 20000 * 4 * 3); cudaMalloc(&G, 20000 * 4 * 3);
  long pf = 0; for (auto& b : br) pf = std::max(pf, cnn_fused_part_floats(b));
  cudaMalloc(&part, pf * 4);
  std::mt19937 rng(1);
  std::vector<uint8_t> hi((size_t)n * o::IMG_SIZE); for (auto& v : hi) v = rng() & 255;
  cudaMemcpy(img, hi.data(), hi.size(), cudaMemcpyHostToDevice);
  std::vector<float> hp(60000); std::normal_distribution<float> N(0, 0.1f); for (auto& v : hp) v = N(rng);
  cudaMemcpy(P, hp.data(), hp.size() * 4, cudaMemcpyHostToDevice);
  std::vector<float> hd((size_t)n * feat); for (auto& v : hd) v = N(rng);
  cudaMemcpy(dX0, hd.data(), hd.size() * 4, cudaMemcpyHostToDevice);
  cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
  double tf = 0, tb = 0;
  for (int b = 0; b < 3; ++b) {
    float* pb = P + b * 20000;
    for (int w = 0; w < 2; ++w) cnn_fused_forward(br[b], pb, img, nullptr, n, X0, 0, tc);
    cudaEventRecord(e0);
    for (int r = 0; r < reps; ++r) cnn_fused_forward(br[b], pb, img, nullptr, n, X0, 0, tc);
    cudaEventRecord(e1); cudaEventSynchronize(e1);
    float f; cudaEventElapsedTime(&f, e0, e1); f /= reps;
    cnn_fused_backward(br[b], pb, img, nullptr, n, dX0, G, part, 0, false, tc);
    cudaEventRecord(e0);
    for (int r = 0; r < reps; ++r) cnn_fused_backward(br[b], pb, img, nullptr, n, dX0, G, part, 0, false, tc);
    cudaEventRecord(e1); cudaEventSynchronize(e1);
    float g; cudaEventElapsedTime(&g, e0, e1); g /= reps;
    printf("가지 %d: 정방향 %.3f ms (%.1f TFLOPS)  역방향 %.3f ms (%.1f TFLOPS)\n", b, f, 2 * macs[b] * n / f / 1e9, g,
           2 * bmacs[b] * n / g / 1e9);
    tf += f, tb += g;
  }
  double tm = 0, tbm = 0; for (int b = 0; b < 3; ++b) tm += macs[b], tbm += bmacs[b];
  printf("합: 정방향 %.3f ms (%.1f TFLOPS)  역방향 %.3f ms (%.1f TFLOPS)\n", tf, 2 * tm * n / tf / 1e9, tb, 2 * tbm * n / tb / 1e9);
  cudaError_t e = cudaGetLastError(); if (e) printf("오류 %s\n", cudaGetErrorString(e));
}
