// CPU 참조 (double 누적): 캐시한 28 층 입력 h[5][D] → 28 · 29 층(인과 shortconv 커널 3 + SwiGLU FFN) → 출력 RMS 정규화 → 라벨 토큰 로짓.
// LoRA (선택): 각 층 in_proj · out_proj · gate · up · down 에 W + (α/r)·B·A. 학습 커널(lora_cuda)의 기울기 검사 기준.
#pragma once
#include <cmath>
#include <vector>

#include "d1_weights.h"

namespace d1 {
enum Mat { kIn = 0, kOut, kGate, kUp, kDown, kMats };
inline int mat_in(int m) { return m == kDown ? kF : kD; }
inline int mat_out(int m) { return m == kIn ? 3 * kD : m == kOut ? kD : m == kDown ? kD : kF; }
struct Lora {   // 층 2 × 행렬 5: A [r][in], B [out][r] (비면 끔)
  int r = 0;
  float scale = 1.f;   // α / r
  std::vector<double> A[2][kMats], B[2][kMats];
  bool on(int l, int m) const { return r > 0 && !A[l][m].empty(); }
};
inline void rms(const double* x, const std::vector<float>& w, float eps, double* y) {
  double s = 0;
  for (int i = 0; i < kD; ++i) s += x[i] * x[i];
  const double inv = 1.0 / std::sqrt(s / kD + eps);
  for (int i = 0; i < kD; ++i) y[i] = x[i] * inv * w[i];
}
// y = (W + s B A) x — W [out][in]
inline void mv(const std::vector<float>& W, int out, int in, const Lora* lo, int l, int m, const double* x, double* y) {
  for (int o = 0; o < out; ++o) {
    const float* w = &W[(size_t)o * in];
    double a = 0;
    for (int i = 0; i < in; ++i) a += (double)w[i] * x[i];
    y[o] = a;
  }
  if (lo && lo->on(l, m)) {
    std::vector<double> t(lo->r, 0.0);
    for (int k = 0; k < lo->r; ++k) { double a = 0; for (int i = 0; i < in; ++i) a += lo->A[l][m][(size_t)k * in + i] * x[i]; t[k] = a; }
    for (int o = 0; o < out; ++o) { double a = 0; for (int k = 0; k < lo->r; ++k) a += lo->B[l][m][(size_t)o * lo->r + k] * t[k]; y[o] += lo->scale * a; }
  }
}
// 층 하나: 입력 n 줄 (인과 — 줄 t 의 conv 는 t−2 … t 를 봄, 앞이 없으면 0). 출력은 뒤 n − 2 줄 (앞 두 줄은 과거가 모자라 버림)
inline std::vector<double> layer(const Weights& W, int l, const Lora* lo, const std::vector<double>& X, int n) {
  const Layer& L = W.L[l];
  std::vector<double> nx(kD), bcx((size_t)3 * kD * n), bx((size_t)kD * n);
  for (int t = 0; t < n; ++t) {
    rms(&X[(size_t)t * kD], L.attn_norm, W.eps, nx.data());
    mv(L.in_proj, 3 * kD, kD, lo, l, kIn, nx.data(), &bcx[(size_t)t * 3 * kD]);
    for (int c = 0; c < kD; ++c) bx[(size_t)t * kD + c] = bcx[(size_t)t * 3 * kD + c] * bcx[(size_t)t * 3 * kD + 2 * kD + c];
  }
  const int no = n - 2;
  std::vector<double> out((size_t)kD * no), y(kD), o(kD), h(kD), f(kD), g(kF), u(kF), d(kD);
  for (int q = 0; q < no; ++q) {
    const int t = q + 2;
    for (int c = 0; c < kD; ++c) {
      double cv = 0;
      for (int k = 0; k < 3; ++k) cv += bx[(size_t)(t - 2 + k) * kD + c] * L.conv[(size_t)c * 3 + k];
      y[c] = bcx[(size_t)t * 3 * kD + kD + c] * cv;
    }
    mv(L.out_proj, kD, kD, lo, l, kOut, y.data(), o.data());
    for (int c = 0; c < kD; ++c) h[c] = X[(size_t)t * kD + c] + o[c];
    rms(h.data(), L.ffn_norm, W.eps, f.data());
    mv(L.gate, kF, kD, lo, l, kGate, f.data(), g.data());
    mv(L.up, kF, kD, lo, l, kUp, f.data(), u.data());
    for (int j = 0; j < kF; ++j) g[j] = g[j] / (1.0 + std::exp(-g[j])) * u[j];
    mv(L.down, kD, kF, lo, l, kDown, g.data(), d.data());
    for (int c = 0; c < kD; ++c) out[(size_t)q * kD + c] = h[c] + d[c];
  }
  return out;
}
// 라벨 토큰 로짓 (E = 머리 줄 [nt][D])
inline std::vector<double> logits(const Weights& W, const Lora* lo, const float* h5, const std::vector<float>& E, int nt) {
  std::vector<double> X((size_t)kT * kD);
  for (size_t k = 0; k < X.size(); ++k) X[k] = h5[k];
  const std::vector<double> a = layer(W, 0, lo, X, kT);   // 3 줄
  const std::vector<double> b = layer(W, 1, lo, a, 3);    // 1 줄 = 마지막 토큰
  std::vector<double> z(kD);
  rms(b.data(), W.out_norm, W.eps, z.data());
  std::vector<double> lg(nt);
  for (int k = 0; k < nt; ++k) { double s = 0; for (int c = 0; c < kD; ++c) s += (double)E[(size_t)k * kD + c] * z[c]; lg[k] = s; }
  return lg;
}
}  // namespace d1
