// lora_train — 위층 판단 모델(d1-3B) 마지막 shortconv 두 층(28·29)에 LoRA 를 붙여 선택지 고르기를 학습 (upper_lora/README.md).
// 입력은 d1_cache 표본(28 층 입력의 마지막 5 토큰 은닉 · 선택지 글자 토큰 묶음 · 정답 번호). 앞 층은 다시 안 돌린다(얼린 앞 층 출력 그대로).
// 계산은 d1_ref.h(CPU, double) 와 같은 식을 GPU(f32, cuBLAS)로: 층마다 RMS → in_proj → b·x → 인과 conv(커널 3) → c· → out_proj → 잔차 → RMS → SwiGLU FFN → 잔차,
// 끝에 출력 RMS → 묶인 머리(token_embd 의 라벨 토큰 줄). 선택지 점수 = 묶음 안 로짓 최댓값(서버 server-context.cpp 와 같음), 손실 = 교차 엔트로피.
// LoRA: 층마다 in_proj · out_proj · gate · up · down 중 --mats 로 고른 것에 W + (α/r)·B·A (A 작은 무작위 · B 0 으로 시작 — Hu et al. 2021).
//
//   lora_train --model d1.gguf --gradcheck cache.bin                         CPU 참조(double, 유한 차분)와 GPU 해석 기울기 견줌
//   lora_train --model d1.gguf --train a.bin --val b.bin --out adapter.bin [--r 8] [--alpha 16] [--lr 1e-3] [--epochs 5] [--batch 16] [--mats in,out,gate,up,down]
//              [--shuffle-aug]  선택지 차례 섞기 증강은 캐시 단계에서(다른 차례로 만든 요청을 같이 캐시) — 여기서는 표본을 그대로 씀
#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <map>
#include <random>
#include <string>
#include <vector>

#include "d1_ref.h"

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { std::fprintf(stderr, "CUDA %s @%d\n", cudaGetErrorString(e_), __LINE__); std::exit(1); } } while (0)
#define CB(x) do { cublasStatus_t e_ = (x); if (e_ != CUBLAS_STATUS_SUCCESS) { std::fprintf(stderr, "cuBLAS %d @%d\n", (int)e_, __LINE__); std::exit(1); } } while (0)

namespace {
using d1::kD;
using d1::kF;
using d1::kT;

struct Sample {
  int no = 0, target = -1;
  int win = -1, lose = -1;   // 선호 쌍 (DPO): 이긴 · 진 선택지 (−1 이면 교차 엔트로피 표본)
  double dref = 0;           // 기준 모델(시작 어댑터)의 (log p_win − log p_lose)
  double beta = 0.1;
  std::vector<int> grp, tok;
  std::vector<float> h;   // [5][D]
};
std::vector<Sample> read_cache(const std::string& path) {
  std::vector<Sample> S;
  std::FILE* f = std::fopen(path.c_str(), "rb");
  if (!f) throw std::runtime_error("캐시를 못 엶: " + path);
  for (;;) {
    uint32_t no, nt;
    int32_t tg;
    if (std::fread(&no, 4, 1, f) != 1 || std::fread(&nt, 4, 1, f) != 1 || std::fread(&tg, 4, 1, f) != 1) break;
    Sample s;
    s.no = (int)no; s.target = tg;
    std::vector<uint32_t> g(no); std::vector<int32_t> t(nt); std::vector<float> lg(nt);
    s.h.resize((size_t)kT * kD);
    if (std::fread(g.data(), 4, no, f) != no || std::fread(t.data(), 4, nt, f) != nt || std::fread(lg.data(), 4, nt, f) != nt || std::fread(s.h.data(), 4, s.h.size(), f) != s.h.size()) break;
    s.grp.assign(g.begin(), g.end()); s.tok.assign(t.begin(), t.end());
    S.push_back(std::move(s));
  }
  std::fclose(f);
  return S;
}

// ---- 커널
__global__ void k_rms_fwd(const float* x, const float* w, float* y, float* inv, int rows, float eps) {
  const int r = blockIdx.x;
  if (r >= rows) return;
  __shared__ float red[256];
  float s = 0;
  for (int i = threadIdx.x; i < kD; i += blockDim.x) { const float v = x[(size_t)r * kD + i]; s += v * v; }
  red[threadIdx.x] = s;
  __syncthreads();
  for (int k = blockDim.x / 2; k > 0; k >>= 1) { if (threadIdx.x < k) red[threadIdx.x] += red[threadIdx.x + k]; __syncthreads(); }
  const float iv = rsqrtf(red[0] / kD + eps);
  if (threadIdx.x == 0) inv[r] = iv;
  for (int i = threadIdx.x; i < kD; i += blockDim.x) y[(size_t)r * kD + i] = x[(size_t)r * kD + i] * iv * w[i];
}
// dx += inv·w·dy − x·inv³/D·Σ(w·dy·x)
__global__ void k_rms_bwd(const float* x, const float* w, const float* inv, const float* dy, float* dx, int rows) {
  const int r = blockIdx.x;
  if (r >= rows) return;
  __shared__ float red[256];
  float s = 0;
  for (int i = threadIdx.x; i < kD; i += blockDim.x) s += w[i] * dy[(size_t)r * kD + i] * x[(size_t)r * kD + i];
  red[threadIdx.x] = s;
  __syncthreads();
  for (int k = blockDim.x / 2; k > 0; k >>= 1) { if (threadIdx.x < k) red[threadIdx.x] += red[threadIdx.x + k]; __syncthreads(); }
  const float iv = inv[r], c = red[0] * iv * iv * iv / kD;
  for (int i = threadIdx.x; i < kD; i += blockDim.x) dx[(size_t)r * kD + i] += iv * w[i] * dy[(size_t)r * kD + i] - x[(size_t)r * kD + i] * c;
}
// conv: 표본마다 입력 n 줄, 출력 n − 2 줄. bcx [B·n][3D] → y [B·(n−2)][D], conv [B·(n−2)][D] (역전파용)
__global__ void k_conv_fwd(const float* bcx, const float* cw, float* y, float* cv, int B, int n) {
  const int no = n - 2;
  const size_t tot = (size_t)B * no * kD;
  for (size_t e = blockIdx.x * (size_t)blockDim.x + threadIdx.x; e < tot; e += (size_t)gridDim.x * blockDim.x) {
    const int c = (int)(e % kD);
    const size_t rq = e / kD;
    const int s = (int)(rq / no), q = (int)(rq % no);
    float a = 0;
    for (int k = 0; k < 3; ++k) {
      const size_t rr = (size_t)s * n + q + k;
      a += bcx[rr * 3 * kD + c] * bcx[rr * 3 * kD + 2 * kD + c] * cw[c * 3 + k];
    }
    cv[e] = a;
    y[e] = bcx[((size_t)s * n + q + 2) * 3 * kD + kD + c] * a;
  }
}
// dbcx 를 0 에서 채움 (한 스레드가 칸 하나의 모든 기여를 더해 경쟁 없음)
__global__ void k_conv_bwd(const float* bcx, const float* cw, const float* cv, const float* dy, float* dbcx, int B, int n) {
  const int no = n - 2;
  const size_t tot = (size_t)B * n * kD;
  for (size_t e = blockIdx.x * (size_t)blockDim.x + threadIdx.x; e < tot; e += (size_t)gridDim.x * blockDim.x) {
    const int c = (int)(e % kD);
    const size_t rr = e / kD;
    const int s = (int)(rr / n), t = (int)(rr % n);
    const float b = bcx[rr * 3 * kD + c], x = bcx[rr * 3 * kD + 2 * kD + c];
    float dbx = 0, dc = 0;
    for (int k = 0; k < 3; ++k) {   // t = q + k → q = t − k
      const int q = t - k;
      if (q < 0 || q >= no) continue;
      const size_t o = ((size_t)s * no + q) * kD + c;
      const float cc = bcx[((size_t)s * n + q + 2) * 3 * kD + kD + c];
      dbx += dy[o] * cc * cw[c * 3 + k];
    }
    if (t >= 2) { const size_t o = ((size_t)s * no + t - 2) * kD + c; dc = dy[o] * cv[o]; }
    dbcx[rr * 3 * kD + c] = dbx * x;
    dbcx[rr * 3 * kD + kD + c] = dc;
    dbcx[rr * 3 * kD + 2 * kD + c] = dbx * b;
  }
}
// 표본마다 줄 고르기: dst[s·m + j] = src[s·n + off + j]
__global__ void k_pick(const float* src, float* dst, int B, int n, int m, int off) {
  const size_t tot = (size_t)B * m * kD;
  for (size_t e = blockIdx.x * (size_t)blockDim.x + threadIdx.x; e < tot; e += (size_t)gridDim.x * blockDim.x) {
    const int c = (int)(e % kD);
    const size_t r = e / kD;
    const int s = (int)(r / m), j = (int)(r % m);
    dst[e] = src[((size_t)s * n + off + j) * kD + c];
  }
}
__global__ void k_pick_add_back(const float* d, float* dsrc, int B, int n, int m, int off) {
  const size_t tot = (size_t)B * m * kD;
  for (size_t e = blockIdx.x * (size_t)blockDim.x + threadIdx.x; e < tot; e += (size_t)gridDim.x * blockDim.x) {
    const int c = (int)(e % kD);
    const size_t r = e / kD;
    const int s = (int)(r / m), j = (int)(r % m);
    dsrc[((size_t)s * n + off + j) * kD + c] += d[e];
  }
}
__global__ void k_add(const float* a, const float* b, float* y, size_t n) {
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) y[i] = a[i] + b[i];
}
__global__ void k_swiglu(const float* g, const float* u, float* a, size_t n) {
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) { const float x = g[i]; a[i] = x / (1.f + __expf(-x)) * u[i]; }
}
__global__ void k_swiglu_bwd(const float* g, const float* u, const float* da, float* dg, float* du, size_t n) {
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) {
    const float x = g[i], sg = 1.f / (1.f + __expf(-x)), si = x * sg;
    du[i] = da[i] * si;
    dg[i] = da[i] * u[i] * (sg + x * sg * (1.f - sg));
  }
}
__global__ void k_adam(float* p, const float* g, float* m, float* v, size_t n, float lr, float b1, float b2, float c1, float c2, float eps) {
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) {
    m[i] = b1 * m[i] + (1 - b1) * g[i];
    v[i] = b2 * v[i] + (1 - b2) * g[i] * g[i];
    p[i] -= lr * (m[i] / c1) / (sqrtf(v[i] / c2) + eps);
  }
}
inline int nb(size_t n) { return (int)std::min<size_t>(4096, (n + 255) / 256); }

// ---- 장치 가중·LoRA
struct DMat { float* W = nullptr; int out = 0, in = 0; float *A = nullptr, *B = nullptr, *gA = nullptr, *gB = nullptr, *mA = nullptr, *vA = nullptr, *mB = nullptr, *vB = nullptr; };
struct DLayer { float *attn_norm, *ffn_norm, *conv; DMat M[d1::kMats]; };
struct Net {
  cublasHandle_t h;
  DLayer L[2];
  float* out_norm;
  int r = 0;
  float scale = 1.f;
  float eps = 1e-5f;
};
float* up(const std::vector<float>& v) { float* p; CK(cudaMalloc(&p, sizeof(float) * v.size())); CK(cudaMemcpy(p, v.data(), sizeof(float) * v.size(), cudaMemcpyHostToDevice)); return p; }
float* zeros(size_t n) { float* p; CK(cudaMalloc(&p, sizeof(float) * n)); CK(cudaMemset(p, 0, sizeof(float) * n)); return p; }

// Y[rows][out] (=, 또는 += beta 1) X[rows][in] · Wᵀ + s·(X Aᵀ) Bᵀ. t = X Aᵀ [rows][r] 는 역전파용으로 남김
void mm_fwd(Net& N, const DMat& M, const float* X, int rows, float* Y, float* t) {
  const float one = 1.f, zero = 0.f;
  CB(cublasSgemm(N.h, CUBLAS_OP_T, CUBLAS_OP_N, M.out, rows, M.in, &one, M.W, M.in, X, M.in, &zero, Y, M.out));
  if (M.A) {
    CB(cublasSgemm(N.h, CUBLAS_OP_T, CUBLAS_OP_N, N.r, rows, M.in, &one, M.A, M.in, X, M.in, &zero, t, N.r));
    CB(cublasSgemm(N.h, CUBLAS_OP_T, CUBLAS_OP_N, M.out, rows, N.r, &N.scale, M.B, N.r, t, N.r, &one, Y, M.out));
  }
}
// dX (+=) dY · W + LoRA 몫, gA · gB 에 더함 (dt 는 작업 버퍼 [rows][r])
void mm_bwd(Net& N, DMat& M, const float* X, const float* t, const float* dY, int rows, float* dX, bool accumulate, float* dt) {
  const float one = 1.f, zero = 0.f, beta = accumulate ? 1.f : 0.f;
  CB(cublasSgemm(N.h, CUBLAS_OP_N, CUBLAS_OP_N, M.in, rows, M.out, &one, M.W, M.in, dY, M.out, &beta, dX, M.in));
  if (M.A) {
    CB(cublasSgemm(N.h, CUBLAS_OP_N, CUBLAS_OP_T, N.r, M.out, rows, &N.scale, t, N.r, dY, M.out, &one, M.gB, N.r));   // gB_cm (r × out)
    CB(cublasSgemm(N.h, CUBLAS_OP_N, CUBLAS_OP_N, N.r, rows, M.out, &N.scale, M.B, N.r, dY, M.out, &zero, dt, N.r));   // dt (r × rows)
    CB(cublasSgemm(N.h, CUBLAS_OP_N, CUBLAS_OP_T, M.in, N.r, rows, &one, X, M.in, dt, N.r, &one, M.gA, M.in));         // gA_cm (in × r)
    CB(cublasSgemm(N.h, CUBLAS_OP_N, CUBLAS_OP_N, M.in, rows, N.r, &one, M.A, M.in, dt, N.r, &one, dX, M.in));          // dX += A dt
  }
}

// 층 하나의 활성 (역전파용)
struct Act {
  int B = 0, n = 0;
  float *X, *nx, *inv1, *bcx, *tin, *cv, *y, *tout, *o, *Xs, *h, *f, *inv2, *g, *u, *a, *tg, *tu, *d, *tdown, *out;
};
Act alloc_act(int B, int n, int r) {
  Act A;
  A.B = B; A.n = n;
  const size_t R = (size_t)B * n, Q = (size_t)B * (n - 2);
  A.X = zeros(R * kD); A.nx = zeros(R * kD); A.inv1 = zeros(R); A.bcx = zeros(R * 3 * kD); A.tin = zeros(R * std::max(r, 1));
  A.cv = zeros(Q * kD); A.y = zeros(Q * kD); A.tout = zeros(Q * std::max(r, 1)); A.o = zeros(Q * kD); A.Xs = zeros(Q * kD); A.h = zeros(Q * kD);
  A.f = zeros(Q * kD); A.inv2 = zeros(Q); A.g = zeros(Q * kF); A.u = zeros(Q * kF); A.a = zeros(Q * kF); A.tg = zeros(Q * std::max(r, 1)); A.tu = zeros(Q * std::max(r, 1));
  A.d = zeros(Q * kD); A.tdown = zeros(Q * std::max(r, 1)); A.out = zeros(Q * kD);
  return A;
}
void layer_fwd(Net& N, int l, Act& A) {
  DLayer& L = N.L[l];
  const int R = A.B * A.n, Q = A.B * (A.n - 2);
  k_rms_fwd<<<R, 256>>>(A.X, L.attn_norm, A.nx, A.inv1, R, N.eps);
  mm_fwd(N, L.M[d1::kIn], A.nx, R, A.bcx, A.tin);
  k_conv_fwd<<<nb((size_t)Q * kD), 256>>>(A.bcx, L.conv, A.y, A.cv, A.B, A.n);
  mm_fwd(N, L.M[d1::kOut], A.y, Q, A.o, A.tout);
  k_pick<<<nb((size_t)Q * kD), 256>>>(A.X, A.Xs, A.B, A.n, A.n - 2, 2);
  k_add<<<nb((size_t)Q * kD), 256>>>(A.Xs, A.o, A.h, (size_t)Q * kD);
  k_rms_fwd<<<Q, 256>>>(A.h, L.ffn_norm, A.f, A.inv2, Q, N.eps);
  mm_fwd(N, L.M[d1::kGate], A.f, Q, A.g, A.tg);
  mm_fwd(N, L.M[d1::kUp], A.f, Q, A.u, A.tu);
  k_swiglu<<<nb((size_t)Q * kF), 256>>>(A.g, A.u, A.a, (size_t)Q * kF);
  mm_fwd(N, L.M[d1::kDown], A.a, Q, A.d, A.tdown);
  k_add<<<nb((size_t)Q * kD), 256>>>(A.h, A.d, A.out, (size_t)Q * kD);
}
// dout [Q][D] → dX [R][D] (덮어씀). 작업 버퍼 W*
struct Work { float *dh, *da, *dg, *du, *df, *dy, *dbcx, *dnx, *dt; };
Work alloc_work(int B, int n, int r) {
  Work w;
  const size_t R = (size_t)B * n, Q = (size_t)B * (n - 2), mx = std::max(R, Q);
  w.dh = zeros(Q * kD); w.da = zeros(Q * kF); w.dg = zeros(Q * kF); w.du = zeros(Q * kF); w.df = zeros(Q * kD); w.dy = zeros(Q * kD);
  w.dbcx = zeros(R * 3 * kD); w.dnx = zeros(R * kD); w.dt = zeros(mx * std::max(r, 1));
  return w;
}
void layer_bwd(Net& N, int l, Act& A, const float* dout, float* dX, Work& w) {
  DLayer& L = N.L[l];
  const int R = A.B * A.n, Q = A.B * (A.n - 2);
  CK(cudaMemcpy(w.dh, dout, sizeof(float) * (size_t)Q * kD, cudaMemcpyDeviceToDevice));   // 잔차 (out = h + d)
  mm_bwd(N, L.M[d1::kDown], A.a, A.tdown, dout, Q, w.da, false, w.dt);
  k_swiglu_bwd<<<nb((size_t)Q * kF), 256>>>(A.g, A.u, w.da, w.dg, w.du, (size_t)Q * kF);
  mm_bwd(N, L.M[d1::kGate], A.f, A.tg, w.dg, Q, w.df, false, w.dt);
  mm_bwd(N, L.M[d1::kUp], A.f, A.tu, w.du, Q, w.df, true, w.dt);
  k_rms_bwd<<<Q, 256>>>(A.h, L.ffn_norm, A.inv2, w.df, w.dh, Q);
  mm_bwd(N, L.M[d1::kOut], A.y, A.tout, w.dh, Q, w.dy, false, w.dt);
  k_conv_bwd<<<nb((size_t)R * kD), 256>>>(A.bcx, L.conv, A.cv, w.dy, w.dbcx, A.B, A.n);
  mm_bwd(N, L.M[d1::kIn], A.nx, A.tin, w.dbcx, R, w.dnx, false, w.dt);
  CK(cudaMemset(dX, 0, sizeof(float) * (size_t)R * kD));
  k_rms_bwd<<<R, 256>>>(A.X, L.attn_norm, A.inv1, w.dnx, dX, R);
  k_pick_add_back<<<nb((size_t)Q * kD), 256>>>(w.dh, dX, A.B, A.n, A.n - 2, 2);   // 잔차 (h = X[t] + o)
}

// 전체: 표본 B 개 → 손실 (그리고 backward 면 LoRA 기울기 누적)
struct Model {
  Net N;
  Act a0, a1;
  Work w0, w1;
  float *z, *zinv, *lg, *dlg, *dz, *dfin, *E;   // E: 머리 줄 [nE][D]
  int Bmax = 0, nE = 0;
  std::map<int, int> erow;                      // 토큰 → E 줄
  std::vector<double>* last_d = nullptr;        // DPO 기준값 재기: 표본마다 s_w − s_l 을 여기에
};
double run(Model& M, const std::vector<const Sample*>& S, bool backward, int* correct) {
  const int B = (int)S.size();
  std::vector<float> hx((size_t)B * kT * kD);
  for (int s = 0; s < B; ++s) std::memcpy(&hx[(size_t)s * kT * kD], S[s]->h.data(), sizeof(float) * kT * kD);
  M.a0.B = B; M.a1.B = B;
  CK(cudaMemcpy(M.a0.X, hx.data(), sizeof(float) * hx.size(), cudaMemcpyHostToDevice));
  layer_fwd(M.N, 0, M.a0);
  CK(cudaMemcpy(M.a1.X, M.a0.out, sizeof(float) * (size_t)B * 3 * kD, cudaMemcpyDeviceToDevice));
  layer_fwd(M.N, 1, M.a1);
  k_rms_fwd<<<B, 256>>>(M.a1.out, M.N.out_norm, M.z, M.zinv, B, M.N.eps);
  const float one = 1.f, zero = 0.f;
  CB(cublasSgemm(M.N.h, CUBLAS_OP_T, CUBLAS_OP_N, M.nE, B, kD, &one, M.E, kD, M.z, kD, &zero, M.lg, M.nE));   // lg [B][nE]
  std::vector<float> lg((size_t)B * M.nE), dlg((size_t)B * M.nE, 0.f);
  CK(cudaMemcpy(lg.data(), M.lg, sizeof(float) * lg.size(), cudaMemcpyDeviceToHost));
  double loss = 0;
  int ok = 0;
  for (int s = 0; s < B; ++s) {   // 선택지 점수 = 묶음 최댓값, 교차 엔트로피
    const Sample& x = *S[s];
    std::vector<double> sc(x.no);
    std::vector<int> am(x.no);
    size_t i = 0;
    for (int o = 0; o < x.no; ++o) {
      double best = -1e30; int bi = -1;
      for (int j = 0; j < x.grp[o]; ++j, ++i) { const double v = lg[(size_t)s * M.nE + M.erow.at(x.tok[i])]; if (v > best) { best = v; bi = M.erow.at(x.tok[i]); } }
      sc[o] = best; am[o] = bi;
    }
    const double mx = *std::max_element(sc.begin(), sc.end());
    double den = 0;
    for (double v : sc) den += std::exp(v - mx);
    if (x.win >= 0) {   // DPO (Rafailov et al. 2023): L = −log σ(β((s_w − s_l) − Δ_ref)) — log p 차는 점수 차와 같음(정규화 항이 지워짐)
      const double d = (sc[x.win] - sc[x.lose]) - x.dref, z = x.beta * d, sg = 1.0 / (1.0 + std::exp(-z));
      loss += -std::log(std::max(1e-12, sg));
      ok += d > 0;
      const double g = -x.beta * (1.0 - sg) / B;
      dlg[(size_t)s * M.nE + am[x.win]] += (float)g;
      dlg[(size_t)s * M.nE + am[x.lose]] -= (float)g;
      if (M.last_d) (*M.last_d)[s] = sc[x.win] - sc[x.lose];
      continue;
    }
    loss += -(sc[x.target] - mx - std::log(den));
    ok += std::max_element(sc.begin(), sc.end()) - sc.begin() == x.target;
    for (int o = 0; o < x.no; ++o) dlg[(size_t)s * M.nE + am[o]] += (float)((std::exp(sc[o] - mx) / den - (o == x.target)) / B);
  }
  if (correct) *correct = ok;
  if (!backward) return loss / B;
  CK(cudaMemcpy(M.dlg, dlg.data(), sizeof(float) * dlg.size(), cudaMemcpyHostToDevice));
  CB(cublasSgemm(M.N.h, CUBLAS_OP_N, CUBLAS_OP_N, kD, B, M.nE, &one, M.E, kD, M.dlg, M.nE, &zero, M.dz, kD));   // dz [B][D]
  CK(cudaMemset(M.dfin, 0, sizeof(float) * (size_t)B * kD));
  k_rms_bwd<<<B, 256>>>(M.a1.out, M.N.out_norm, M.zinv, M.dz, M.dfin, B);
  static float* d0 = nullptr;   // 1 층 입력(= 0 층 출력) 기울기
  if (!d0) d0 = zeros((size_t)M.Bmax * 3 * kD);
  layer_bwd(M.N, 1, M.a1, M.dfin, d0, M.w1);
  static float* dX0 = nullptr;
  if (!dX0) dX0 = zeros((size_t)M.Bmax * kT * kD);
  layer_bwd(M.N, 0, M.a0, d0, dX0, M.w0);
  return loss / B;
}

void zero_grads(Net& N) {
  for (auto& L : N.L)
    for (auto& m : L.M)
      if (m.A) { CK(cudaMemset(m.gA, 0, sizeof(float) * (size_t)N.r * m.in)); CK(cudaMemset(m.gB, 0, sizeof(float) * (size_t)N.r * m.out)); }
}

Model build(const d1::Weights& W, const d1::Gguf& G, const std::vector<int>& toks, int r, float alpha, const std::vector<bool>& use, int Bmax, uint64_t seed) {
  Model M;
  CB(cublasCreate(&M.N.h));
  M.N.r = r; M.N.scale = r > 0 ? alpha / r : 0.f; M.N.eps = W.eps;
  std::mt19937_64 rng(seed);
  std::normal_distribution<float> nd(0.f, 1.f);
  for (int l = 0; l < 2; ++l) {
    const d1::Layer& L = W.L[l];
    DLayer& D = M.N.L[l];
    D.attn_norm = up(L.attn_norm); D.ffn_norm = up(L.ffn_norm); D.conv = up(L.conv);
    const std::vector<float>* src[d1::kMats] = {&L.in_proj, &L.out_proj, &L.gate, &L.up, &L.down};
    for (int m = 0; m < d1::kMats; ++m) {
      DMat& X = D.M[m];
      X.W = up(*src[m]); X.in = d1::mat_in(m); X.out = d1::mat_out(m);
      if (r > 0 && use[m]) {   // A ~ N(0, 1/in) (작게), B = 0
        std::vector<float> a((size_t)r * X.in);
        for (auto& v : a) v = nd(rng) / std::sqrt((float)X.in);
        X.A = up(a); X.B = zeros((size_t)X.out * r);
        X.gA = zeros(a.size()); X.gB = zeros((size_t)X.out * r);
        X.mA = zeros(a.size()); X.vA = zeros(a.size()); X.mB = zeros((size_t)X.out * r); X.vB = zeros((size_t)X.out * r);
      }
    }
  }
  M.N.out_norm = up(W.out_norm);
  M.Bmax = Bmax;
  M.a0 = alloc_act(Bmax, kT, r); M.a1 = alloc_act(Bmax, 3, r);
  M.w0 = alloc_work(Bmax, kT, r); M.w1 = alloc_work(Bmax, 3, r);
  M.z = zeros((size_t)Bmax * kD); M.zinv = zeros(Bmax); M.dz = zeros((size_t)Bmax * kD); M.dfin = zeros((size_t)Bmax * kD);
  M.nE = (int)toks.size();
  for (int k = 0; k < M.nE; ++k) M.erow[toks[k]] = k;
  M.E = up(d1::head_rows(G, toks));
  M.lg = zeros((size_t)Bmax * M.nE); M.dlg = zeros((size_t)Bmax * M.nE);
  return M;
}
// 장치 LoRA ↔ CPU 참조 Lora (기울기 검사)
d1::Lora to_ref(const Model& M) {
  d1::Lora lo;
  lo.r = M.N.r; lo.scale = M.N.scale;
  for (int l = 0; l < 2; ++l)
    for (int m = 0; m < d1::kMats; ++m) {
      const DMat& X = M.N.L[l].M[m];
      if (!X.A) continue;
      std::vector<float> a((size_t)lo.r * X.in), b((size_t)X.out * lo.r);
      CK(cudaMemcpy(a.data(), X.A, sizeof(float) * a.size(), cudaMemcpyDeviceToHost));
      CK(cudaMemcpy(b.data(), X.B, sizeof(float) * b.size(), cudaMemcpyDeviceToHost));
      lo.A[l][m].assign(a.begin(), a.end());
      // 장치 B 는 열 우선 (r × out) = 행 우선 [out][r] 와 같은 배치
      lo.B[l][m].assign(b.begin(), b.end());
    }
  return lo;
}
double ref_loss(const d1::Weights& W, const d1::Lora& lo, const Sample& s, const std::vector<float>& E) {
  const std::vector<double> lg = d1::logits(W, &lo, s.h.data(), E, (int)s.tok.size());
  std::vector<double> sc(s.no);
  size_t i = 0;
  for (int o = 0; o < s.no; ++o) { double b = -1e30; for (int j = 0; j < s.grp[o]; ++j, ++i) b = std::max(b, lg[i]); sc[o] = b; }
  const double mx = *std::max_element(sc.begin(), sc.end());
  double den = 0;
  for (double v : sc) den += std::exp(v - mx);
  return -(sc[s.target] - mx - std::log(den));
}
}  // namespace

int main(int argc, char** argv) {
  std::string model, gc, train, val, out = "adapter.bin", mats = "in,out,gate,up,down", init, pairs, vpairs;
  float beta = 0.1f;
  int r = 8, epochs = 5, batch = 16;
  float alpha = 16.f, lr = 1e-3f;
  for (int i = 1; i < argc; ++i) {
    const std::string k = argv[i];
    auto nx = [&]() { return std::string(argv[++i]); };
    if (k == "--model") model = nx(); else if (k == "--gradcheck") gc = nx(); else if (k == "--train") train = nx(); else if (k == "--val") val = nx();
    else if (k == "--out") out = nx(); else if (k == "--r") r = std::stoi(nx()); else if (k == "--alpha") alpha = std::stof(nx()); else if (k == "--lr") lr = std::stof(nx());
    else if (k == "--epochs") epochs = std::stoi(nx());
    else if (k == "--init") init = nx(); else if (k == "--pairs") pairs = nx(); else if (k == "--val-pairs") vpairs = nx(); else if (k == "--beta") beta = std::stof(nx()); else if (k == "--batch") batch = std::stoi(nx()); else if (k == "--mats") mats = nx();
    else { std::fprintf(stderr, "모르는 인자 %s\n", k.c_str()); return 2; }
  }
  std::vector<bool> use(d1::kMats, false);
  const char* names[d1::kMats] = {"in", "out", "gate", "up", "down"};
  for (int m = 0; m < d1::kMats; ++m) use[m] = ("," + mats + ",").find(std::string(",") + names[m] + ",") != std::string::npos;
  d1::Gguf G(model);
  const d1::Weights W = d1::load(G);
  if (!gc.empty()) {   // ---- 기울기 검사: 작은 r, 무작위 A·B, 표본 2 개
    std::vector<Sample> S = read_cache(gc);
    S.resize(std::min<size_t>(2, S.size()));
    for (auto& s : S) if (s.target < 0) s.target = 0;
    std::vector<int> toks;
    for (auto& s : S) for (int t : s.tok) if (std::find(toks.begin(), toks.end(), t) == toks.end()) toks.push_back(t);
    Model M = build(W, G, toks, 2, 4.f, std::vector<bool>(d1::kMats, true), (int)S.size(), 7);
    std::mt19937_64 rng(11);
    std::normal_distribution<float> nd(0.f, 0.02f);
    for (auto& L : M.N.L) for (auto& m : L.M) {   // B 도 0 아닌 작은 값 (A 쪽 기울기가 0 이 아니게)
      std::vector<float> b((size_t)m.out * M.N.r);
      for (auto& v : b) v = nd(rng);
      CK(cudaMemcpy(m.B, b.data(), sizeof(float) * b.size(), cudaMemcpyHostToDevice));
    }
    std::vector<const Sample*> P;
    for (auto& s : S) P.push_back(&s);
    zero_grads(M.N);
    const double lgpu = run(M, P, true, nullptr);
    CK(cudaDeviceSynchronize());
    d1::Lora lo = to_ref(M);
    std::vector<std::vector<float>> E;
    for (auto& s : S) E.push_back(d1::head_rows(G, s.tok));
    const auto L = [&](const d1::Lora& x) { double t = 0; for (size_t k = 0; k < S.size(); ++k) t += ref_loss(W, x, S[k], E[k]); return t / S.size(); };
    const double lcpu = L(lo);
    std::printf("손실 GPU %.6f · CPU 참조 %.6f\n", lgpu, lcpu);
    double worst = 0;
    int nchk = 0;
    for (int l = 0; l < 2; ++l)
      for (int m = 0; m < d1::kMats; ++m)
        for (int which = 0; which < 2; ++which) {   // A · B 에서 칸 하나씩
          DMat& X = M.N.L[l].M[m];
          const size_t n = which == 0 ? (size_t)M.N.r * X.in : (size_t)X.out * M.N.r;
          const size_t idx = (size_t)(rng() % n);
          float g;
          CK(cudaMemcpy(&g, (which == 0 ? X.gA : X.gB) + idx, sizeof(float), cudaMemcpyDeviceToHost));
          std::vector<double>& V = which == 0 ? lo.A[l][m] : lo.B[l][m];
          // 장치 gA 는 열 우선 (in × r): 칸 idx = k·in + i ↔ A[k][i] 와 같은 배치. gB (r × out): idx = o·r + k ↔ B[o][k]
          const double h = 1e-3, v0 = V[idx];
          V[idx] = v0 + h; const double lp = L(lo);
          V[idx] = v0 - h; const double lm = L(lo);
          V[idx] = v0;
          const double num = (lp - lm) / (2 * h), rel = std::fabs(num - g) / std::max(1e-6, std::fabs(num) + std::fabs(g));
          std::printf("층 %d %-5s %s[%zu]: 해석 %+.6e · 수치 %+.6e · 상대 차 %.2e\n", 28 + l, names[m], which ? "B" : "A", idx, (double)g, num, rel);
          worst = std::max(worst, rel);
          ++nchk;
        }
    std::printf("기울기 검사 %d 칸 · 가장 큰 상대 차 %.2e (%s)\n", nchk, worst, worst < 2e-2 ? "통과" : "실패");
    return worst < 2e-2 ? 0 : 1;
  }
  // ---- 학습
  std::vector<Sample> TR = read_cache(train), VA = val.empty() ? std::vector<Sample>() : read_cache(val);
  // 선호 쌍 (--pairs: 줄마다 "<캐시 표본 번호> <이긴 선택지> <진 선택지>") — 쌍마다 표본을 하나씩 만듦
  const auto to_pairs = [&](std::vector<Sample>& V, const std::string& pf) {
    if (pf.empty()) return;
    std::vector<Sample> P;
    std::FILE* f = std::fopen(pf.c_str(), "r");
    if (!f) throw std::runtime_error("쌍 파일을 못 엶: " + pf);
    int ix, w, l;
    while (std::fscanf(f, "%d %d %d", &ix, &w, &l) == 3)
      if (ix >= 0 && ix < (int)V.size() && w >= 0 && l >= 0 && w < V[ix].no && l < V[ix].no && w != l) { Sample s = V[ix]; s.win = w; s.lose = l; s.beta = beta; P.push_back(s); }
    std::fclose(f);
    V.swap(P);
  };
  to_pairs(TR, pairs);
  to_pairs(VA, vpairs);
  const auto bad = [](const Sample& s) { return s.win < 0 && (s.target < 0 || s.target >= s.no); };
  TR.erase(std::remove_if(TR.begin(), TR.end(), bad), TR.end());
  VA.erase(std::remove_if(VA.begin(), VA.end(), bad), VA.end());
  std::vector<int> toks;
  for (auto* V : {&TR, &VA}) for (auto& s : *V) for (int t : s.tok) if (std::find(toks.begin(), toks.end(), t) == toks.end()) toks.push_back(t);
  Model M = build(W, G, toks, r, alpha, use, batch, 1);
  if (!init.empty()) {   // 시작 어댑터 (BC 결과) — 같은 r · 행렬이어야 함
    std::FILE* fi = std::fopen(init.c_str(), "rb");
    int r0; float a0;
    if (!fi || std::fread(&r0, 4, 1, fi) != 1 || std::fread(&a0, 4, 1, fi) != 1 || r0 != r) throw std::runtime_error("--init 어댑터가 r 이 다름");
    for (auto& L : M.N.L) for (auto& m : L.M) {
      int on = 0;
      if (std::fread(&on, 4, 1, fi) != 1) throw std::runtime_error("--init 짧음");
      if (on != (m.A ? 1 : 0)) throw std::runtime_error("--init 행렬 고름이 다름");
      if (!on) continue;
      std::vector<float> a((size_t)r * m.in), b((size_t)m.out * r);
      if (std::fread(a.data(), 4, a.size(), fi) != a.size() || std::fread(b.data(), 4, b.size(), fi) != b.size()) throw std::runtime_error("--init 짧음");
      CK(cudaMemcpy(m.A, a.data(), sizeof(float) * a.size(), cudaMemcpyHostToDevice));
      CK(cudaMemcpy(m.B, b.data(), sizeof(float) * b.size(), cudaMemcpyHostToDevice));
    }
    std::fclose(fi);
  }
  for (auto* V : {&TR, &VA}) {   // DPO 기준값: 시작 어댑터로 s_w − s_l
    std::vector<Sample*> P;
    for (auto& s : *V) if (s.win >= 0) P.push_back(&s);
    for (size_t b0 = 0; b0 < P.size(); b0 += batch) {
      std::vector<const Sample*> Q;
      for (size_t k = b0; k < std::min(P.size(), b0 + batch); ++k) Q.push_back(P[k]);
      std::vector<double> d(Q.size());
      M.last_d = &d;
      for (auto* q : Q) const_cast<Sample*>(q)->dref = 0;
      run(M, Q, false, nullptr);
      M.last_d = nullptr;
      for (size_t k = 0; k < Q.size(); ++k) P[b0 + k]->dref = d[k];
    }
  }
  const auto eval = [&](const std::vector<Sample>& V) {
    double ls = 0; int ok = 0;
    for (size_t b0 = 0; b0 < V.size(); b0 += batch) {
      std::vector<const Sample*> P;
      for (size_t k = b0; k < std::min(V.size(), b0 + batch); ++k) P.push_back(&V[k]);
      int c = 0;
      ls += run(M, P, false, &c) * P.size();
      ok += c;
    }
    return std::make_pair(ls / std::max<size_t>(1, V.size()), (double)ok / std::max<size_t>(1, V.size()));
  };
  auto e0 = eval(VA);
  std::printf("학습 %zu · 검증 %zu 표본 · r %d · α %.0f · 행렬 %s | 시작 검증 손실 %.4f · 맞춤 %.3f\n", TR.size(), VA.size(), r, alpha, mats.c_str(), e0.first, e0.second);
  std::mt19937_64 rng(3);
  std::vector<size_t> ord(TR.size());
  for (size_t k = 0; k < ord.size(); ++k) ord[k] = k;
  int step = 0;
  const float b1 = 0.9f, b2 = 0.999f;
  for (int ep = 0; ep < epochs; ++ep) {
    std::shuffle(ord.begin(), ord.end(), rng);
    double tl = 0; int tok_ = 0;
    for (size_t b0 = 0; b0 + batch <= ord.size(); b0 += batch) {
      std::vector<const Sample*> P;
      for (int k = 0; k < batch; ++k) P.push_back(&TR[ord[b0 + k]]);
      zero_grads(M.N);
      int c = 0;
      tl += run(M, P, true, &c);
      tok_ += c;
      ++step;
      const float c1 = 1 - std::pow(b1, (float)step), c2 = 1 - std::pow(b2, (float)step);
      for (auto& L : M.N.L) for (auto& m : L.M) if (m.A) {
        k_adam<<<nb((size_t)r * m.in), 256>>>(m.A, m.gA, m.mA, m.vA, (size_t)r * m.in, lr, b1, b2, c1, c2, 1e-8f);
        k_adam<<<nb((size_t)r * m.out), 256>>>(m.B, m.gB, m.mB, m.vB, (size_t)r * m.out, lr, b1, b2, c1, c2, 1e-8f);
      }
    }
    const auto e = eval(VA);
    const size_t nbt = std::max<size_t>(1, ord.size() / batch);
    std::printf("에폭 %d: 학습 손실 %.4f · 학습 맞춤 %.3f | 검증 손실 %.4f · 맞춤 %.3f\n", ep + 1, tl / nbt, (double)tok_ / (nbt * batch), e.first, e.second);
    std::fflush(stdout);
  }
  // 어댑터 저장 (간단한 바이너리 — GGUF LoRA 로 바꾸는 것은 lora_export 몫)
  std::FILE* f = std::fopen(out.c_str(), "wb");
  std::fwrite(&r, 4, 1, f); std::fwrite(&alpha, 4, 1, f);
  for (int l = 0; l < 2; ++l) for (int m = 0; m < d1::kMats; ++m) {
    const DMat& X = M.N.L[l].M[m];
    const int on = X.A ? 1 : 0;
    std::fwrite(&on, 4, 1, f);
    if (!on) continue;
    std::vector<float> a((size_t)r * X.in), b((size_t)X.out * r);
    CK(cudaMemcpy(a.data(), X.A, sizeof(float) * a.size(), cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(b.data(), X.B, sizeof(float) * b.size(), cudaMemcpyDeviceToHost));
    std::fwrite(a.data(), 4, a.size(), f); std::fwrite(b.data(), 4, b.size(), f);
  }
  std::fclose(f);
  std::printf("어댑터 %s\n", out.c_str());
  return 0;
}
