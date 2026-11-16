// d1-3B (lfm2) 27 층 = 마지막 attention 층 (학습을 27 층부터 할 때 — README "확장 설계" (a)).
// 계산은 llama.cpp src/models/lfm2.cpp build_attn_block 과 같게: RMS(attn_norm) → q · k · v 투영 → 머리마다 RMS(q_norm · k_norm, 머리 64) → RoPE(NEOX, 밑 1e6)
// → 인과 GQA (q 32 머리 · kv 8 머리, 크기 1/8) → out_proj → 잔차 → RMS(ffn_norm) → SwiGLU FFN → 잔차.
// 입력은 d1_cache --full-from 27 의 l_out-26 (시퀀스 전체, 자리 0 … T−1). 28 층은 27 층 출력의 마지막 5 줄만 보므로 q · FFN 은 마지막 5 줄만, k · v 는 전체 줄.
#pragma once
#include <algorithm>
#include <cmath>
#include <string>
#include <vector>

#include "d1_ref.h"

namespace d1 {
constexpr int kH = 32, kHkv = 8, kHd = 64, kKV = kHkv * kHd;   // 머리 수 · kv 머리 수 · 머리 크기 · kv 너비 (512)
constexpr double kRopeBase = 1e6;
enum AMat { kQ = 0, kK, kV, kO, kAG, kAU, kAD, kAMats };
inline int amat_in(int m) { return m == kAD ? kF : kD; }
inline int amat_out(int m) { return (m == kK || m == kV) ? kKV : m == kAG || m == kAU ? kF : kD; }
struct ALayer {
  std::vector<float> attn_norm, ffn_norm, q_norm, k_norm;   // [D] · [D] · [64] · [64]
  std::vector<float> W[kAMats];                             // [out][in]
};
inline ALayer load27(const Gguf& G) {
  ALayer A;
  const std::string p = "blk.27.";
  A.attn_norm = G.all(p + "attn_norm.weight", kD, 1);
  A.ffn_norm = G.all(p + "ffn_norm.weight", kD, 1);
  A.q_norm = G.all(p + "attn_q_norm.weight", kHd, 1);
  A.k_norm = G.all(p + "attn_k_norm.weight", kHd, 1);
  const char* nm[kAMats] = {"attn_q", "attn_k", "attn_v", "attn_output", "ffn_gate", "ffn_up", "ffn_down"};
  for (int m = 0; m < kAMats; ++m) A.W[m] = G.all(p + nm[m] + ".weight", amat_in(m), amat_out(m));
  return A;
}
struct ALora {   // 27 층 행렬 7 개: A [r][in] · B [out][r] (비면 끔)
  int r = 0;
  float scale = 1.f;
  std::vector<double> A[kAMats], B[kAMats];
  bool on(int m) const { return r > 0 && !A[m].empty(); }
};
inline void amv(const std::vector<float>& W, int out, int in, const ALora* lo, int m, const double* x, double* y) {
  for (int o = 0; o < out; ++o) {
    const float* w = &W[(size_t)o * in];
    double a = 0;
    for (int i = 0; i < in; ++i) a += (double)w[i] * x[i];
    y[o] = a;
  }
  if (lo && lo->on(m)) {
    std::vector<double> t(lo->r, 0.0);
    for (int k = 0; k < lo->r; ++k) { double a = 0; for (int i = 0; i < in; ++i) a += lo->A[m][(size_t)k * in + i] * x[i]; t[k] = a; }
    for (int o = 0; o < out; ++o) { double a = 0; for (int k = 0; k < lo->r; ++k) a += lo->B[m][(size_t)o * lo->r + k] * t[k]; y[o] += lo->scale * a; }
  }
}
// 머리마다 RMS + NEOX RoPE (자리 pos) — 제자리
inline void head_norm_rope(double* x, int nh, const std::vector<float>& w, float eps, int pos) {
  for (int h = 0; h < nh; ++h) {
    double* v = x + (size_t)h * kHd;
    double s = 0;
    for (int i = 0; i < kHd; ++i) s += v[i] * v[i];
    const double inv = 1.0 / std::sqrt(s / kHd + eps);
    for (int i = 0; i < kHd; ++i) v[i] = v[i] * inv * w[i];
    for (int i = 0; i < kHd / 2; ++i) {
      const double th = pos * std::pow(kRopeBase, -2.0 * i / kHd), c = std::cos(th), sn = std::sin(th);
      const double a = v[i], b = v[i + kHd / 2];
      v[i] = a * c - b * sn;
      v[i + kHd / 2] = a * sn + b * c;
    }
  }
}
// 27 층: 입력 X [T][D] (f32) → 출력 마지막 kT 줄 [kT][D]
inline std::vector<double> layer27(const ALayer& L, float eps, const ALora* lo, const float* X, int T) {
  std::vector<double> nx(kD), K((size_t)T * kKV), V((size_t)T * kKV), x(kD);
  for (int t = 0; t < T; ++t) {
    for (int c = 0; c < kD; ++c) x[c] = X[(size_t)t * kD + c];
    rms(x.data(), L.attn_norm, eps, nx.data());
    amv(L.W[kK], kKV, kD, lo, kK, nx.data(), &K[(size_t)t * kKV]);
    amv(L.W[kV], kKV, kD, lo, kV, nx.data(), &V[(size_t)t * kKV]);
    head_norm_rope(&K[(size_t)t * kKV], kHkv, L.k_norm, eps, t);
  }
  std::vector<double> out((size_t)kT * kD), q(kD), att(kD), o(kD), h(kD), f(kD), g(kF), u(kF), d(kD), p(T);
  for (int i = 0; i < kT; ++i) {
    const int t = T - kT + i;
    for (int c = 0; c < kD; ++c) x[c] = X[(size_t)t * kD + c];
    rms(x.data(), L.attn_norm, eps, nx.data());
    amv(L.W[kQ], kD, kD, lo, kQ, nx.data(), q.data());
    head_norm_rope(q.data(), kH, L.q_norm, eps, t);
    for (int hh = 0; hh < kH; ++hh) {
      const int kv = hh / (kH / kHkv);
      double mx = -1e300;
      for (int j = 0; j <= t; ++j) {
        double s = 0;
        for (int e = 0; e < kHd; ++e) s += q[(size_t)hh * kHd + e] * K[(size_t)j * kKV + kv * kHd + e];
        p[j] = s / std::sqrt((double)kHd);
        mx = std::max(mx, p[j]);
      }
      double den = 0;
      for (int j = 0; j <= t; ++j) { p[j] = std::exp(p[j] - mx); den += p[j]; }
      for (int e = 0; e < kHd; ++e) {
        double a = 0;
        for (int j = 0; j <= t; ++j) a += p[j] * V[(size_t)j * kKV + kv * kHd + e];
        att[(size_t)hh * kHd + e] = a / den;
      }
    }
    amv(L.W[kO], kD, kD, lo, kO, att.data(), o.data());
    for (int c = 0; c < kD; ++c) h[c] = x[c] + o[c];
    rms(h.data(), L.ffn_norm, eps, f.data());
    amv(L.W[kAG], kF, kD, lo, kAG, f.data(), g.data());
    amv(L.W[kAU], kF, kD, lo, kAU, f.data(), u.data());
    for (int j = 0; j < kF; ++j) g[j] = g[j] / (1.0 + std::exp(-g[j])) * u[j];
    amv(L.W[kAD], kD, kF, lo, kAD, g.data(), d.data());
    for (int c = 0; c < kD; ++c) out[(size_t)i * kD + c] = h[c] + d[c];
  }
  return out;
}
// 27 → 28 → 29 → 라벨 로짓
inline std::vector<double> logits_full(const Weights& W, const ALayer& A, const Lora* lo, const ALora* alo, const float* X, int T, const std::vector<float>& E, int nt) {
  const std::vector<double> h5 = layer27(A, W.eps, alo, X, T);
  const std::vector<double> a = layer(W, 0, lo, h5, kT);
  const std::vector<double> b = layer(W, 1, lo, a, 3);
  std::vector<double> z(kD);
  rms(b.data(), W.out_norm, W.eps, z.data());
  std::vector<double> lg(nt);
  for (int k = 0; k < nt; ++k) { double s = 0; for (int c = 0; c < kD; ++c) s += (double)E[(size_t)k * kD + c] * z[c]; lg[k] = s; }
  return lg;
}
}  // namespace d1
