// MLP GEMM 모양별 시간·오차: cuBLAS sgemm(옛 길) vs 직접 짠 융합 GEMM (fp32 정확·bf16). 덩이 1024, actor 첫 층 크기.
//   gemm_bench [--n 1024] [--reps 20]   (시간 = 호출마다 L2 를 비우고 이벤트로 잰 최솟값)
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <random>
#include <string>
#include <vector>

#include "common.cuh"
#include "gemm_fused.cuh"

using namespace xnet;

int main(int argc, char** argv) {
  int n = 1024, reps = 20;
  float dys = 1e-3f;  // dY 크기 (학습 중 실제 값은 1e-6 이하로 작다)
  for (int i = 1; i + 1 < argc; i += 2) {
    std::string a = argv[i];
    if (a == "--n") n = std::stoi(argv[i + 1]);
    if (a == "--reps") reps = std::stoi(argv[i + 1]);
    if (a == "--dys") dys = std::stof(argv[i + 1]);
  }
  cublasHandle_t h;
  CKB(cublasCreate(&h));
  cudaStream_t st;
  CK(cudaStreamCreate(&st));
  CKB(cublasSetStream(h, st));
  struct Shape {
    const char* name;
    int in, out;
  };
  Shape shapes[] = {{"층0 5144→512", 5144, 512}, {"층1 512→256", 512, 256}, {"층2 256→128", 256, 128}, {"층3 128→4", 128, 4}};
  std::mt19937 rng(1);
  std::normal_distribution<float> G;
  auto dev = [&](long cnt, float sc) {
    std::vector<float> hv(cnt);
    for (auto& v : hv) v = sc * G(rng);
    float* d;
    CK(cudaMalloc(&d, cnt * 4));
    CK(cudaMemcpy(d, hv.data(), cnt * 4, cudaMemcpyHostToDevice));
    return d;
  };
  auto rel = [&](const float* a, const float* b, long cnt) {
    std::vector<float> x(cnt), y(cnt);
    CK(cudaMemcpy(x.data(), a, cnt * 4, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(y.data(), b, cnt * 4, cudaMemcpyDeviceToHost));
    double d = 0, s = 0;
    for (long i = 0; i < cnt; ++i) d += (x[i] - y[i]) * (double)(x[i] - y[i]), s += (double)y[i] * y[i];
    return std::sqrt(d / std::max(s, 1e-300));
  };
  // 호출마다 이벤트로 재고 가장 짧은 값 — 다른 작업이 GPU 를 나눠 쓸 때도 커널 자체 속도에 가깝다
  constexpr size_t kFlush = 32u << 20;
  void* flush;
  CK(cudaMalloc(&flush, kFlush));
  cudaEvent_t e0, e1;
  CK(cudaEventCreate(&e0));
  CK(cudaEventCreate(&e1));
  auto timeit = [&](auto f) {
    f();
    CK(cudaStreamSynchronize(st));
    float best = 1e30f;
    for (int r = 0; r < reps; ++r) {
      CK(cudaMemsetAsync(flush, r & 0xff, kFlush, st));  // L2(3 MB) 비우기 — 학습 중처럼 차가운 캐시에서
      CK(cudaEventRecord(e0, st));
      f();
      CK(cudaEventRecord(e1, st));
      CK(cudaEventSynchronize(e1));
      float ms;
      CK(cudaEventElapsedTime(&ms, e0, e1));
      best = std::min(best, ms);
    }
    return (double)best;
  };
  double tot[3] = {0, 0, 0};
  for (auto& s : shapes) {
    float* X = dev((long)n * s.in, 1.f);
    float* W = dev((long)s.out * s.in, 0.02f);
    float* dY = dev((long)n * s.out, dys);
    float* Y = dev((long)n * s.out, 0);
    float* Y2 = dev((long)n * s.out, 0);
    float* dX = dev((long)n * s.in, 0);
    float* dX2 = dev((long)n * s.in, 0);
    float* dW = dev((long)s.out * s.in, 0);
    float* dW2 = dev((long)s.out * s.in, 0);
    float* bg = dev(s.out, 0);
    double gfl = 2.0 * n * s.in * s.out / 1e9;
    // 옛 길
    double tf0 = timeit([&] { gemm(h, false, true, n, s.out, s.in, 1.f, X, s.in, W, s.in, 0.f, Y, s.out); });
    double tx0 = timeit([&] { gemm(h, false, false, n, s.in, s.out, 1.f, dY, s.out, W, s.in, 0.f, dX, s.in); });
    double tw0 = timeit([&] { gemm(h, true, false, s.out, s.in, n, 1.f, dY, s.out, X, s.in, 0.f, dW, s.in); });
    std::printf("%-14s cuBLAS      fwd %.3f  dX %.3f  dW %.3f ms  (%.1f / %.1f / %.1f TFLOPS)\n", s.name, tf0, tx0, tw0,
                gfl / tf0, gfl / tx0, gfl / tw0);
    tot[0] += tf0 + tx0 + tw0;
    unsigned short* Wh;  // 가중치 bf16 사본 (학습 경로와 같게 정방향·dX 가 읽음)
    CK(cudaMalloc(&Wh, (long)s.out * s.in * 2));
    to_bf16(W, Wh, (long)s.out * s.in, st);
    for (int lowp = 0; lowp < 2; ++lowp) {
      FGemm f;
      f.M = n, f.N = s.out, f.K = s.in, f.A = X, f.lda = s.in, f.B = W, f.ldb = s.in, f.tb = true, f.C = Y2, f.ldc = s.out;
      FGemm d;
      d.M = n, d.N = s.in, d.K = s.out, d.A = dY, d.lda = s.out, d.B = W, d.ldb = s.in, d.C = dX2, d.ldc = s.in;
      FGemm w;
      w.M = s.out, w.N = s.in, w.K = n, w.A = dY, w.lda = s.out, w.ta = true, w.B = X, w.ldb = s.in, w.C = dW2,
      w.ldc = s.in, w.bias_grad = bg, w.epi = FEpi::Accum;
      if (lowp) f.Bh = Wh, d.Bh = Wh;
      double tf = timeit([&] { fgemm(f, lowp, st); });
      double tx = timeit([&] { fgemm(d, lowp, st); });
      // dW 는 누적이라 시간 잰 뒤 한 번 0 으로 놓고 다시
      double tw = timeit([&] { fgemm(w, lowp, st); });
      CK(cudaMemsetAsync(dW2, 0, (long)s.out * s.in * 4, st));
      CK(cudaMemsetAsync(bg, 0, s.out * 4, st));
      fgemm(w, lowp, st);
      CK(cudaStreamSynchronize(st));
      double bge = 0;  // 편향 기울기(Σ_샘플 dY) vs host 합 — split-K 조각 합까지 확인
      {
        std::vector<float> hy((long)n * s.out), hb(s.out);
        CK(cudaMemcpy(hy.data(), dY, hy.size() * 4, cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(hb.data(), bg, s.out * 4, cudaMemcpyDeviceToHost));
        double d = 0, nn = 0;
        for (int m = 0; m < s.out; ++m) {
          double r = 0;
          for (int i = 0; i < n; ++i) r += hy[(long)i * s.out + m];
          d += (hb[m] - r) * (hb[m] - r), nn += r * r;
        }
        bge = std::sqrt(d / std::max(nn, 1e-300));
      }
      std::printf("%-14s %-11s fwd %.3f  dX %.3f  dW %.3f ms  (%.1f / %.1f / %.1f TFLOPS)  오차 %.1e %.1e %.1e 편향 %.1e\n", s.name,
                  lowp ? "융합 bf16" : "융합 fp32", tf, tx, tw, gfl / tf, gfl / tx, gfl / tw, rel(Y2, Y, (long)n * s.out),
                  rel(dX2, dX, (long)n * s.in), rel(dW2, dW, (long)s.out * s.in), bge);
      tot[1 + lowp] += tf + tx + tw;
    }
    for (float* p : {X, W, dY, Y, Y2, dX, dX2, dW, dW2, bg}) cudaFree(p);
    cudaFree(Wh);
  }
  std::printf("합 (fwd+dX+dW, 세 층): cuBLAS %.3f ms, 융합 fp32 %.3f ms, 융합 bf16 %.3f ms\n", tot[0], tot[1], tot[2]);
  return 0;
}
