#include <algorithm>
#include <cfloat>

#include "kern.cuh"
#include "vec.cuh"

namespace kern {
namespace {
constexpr int TPB = 256;
inline int nb(int64_t n) { return cdiv(n, TPB); }
__device__ __forceinline__ float silu(float x) { return x / (1.f + __expf(-x)); }
__device__ __forceinline__ float dsilu(float x) {
  float sg = 1.f / (1.f + __expf(-x));
  return sg * (1.f + x * (1.f - sg));
}

// 채널별 합 두 개를 double 로 모으는 공통 틀: 블록 = 32 채널 × 8 행 묶음
template <class F>
__global__ void chan_reduce2_k(int64_t M, int C, int rows_per_block, F f, double* acc) {
  int c = blockIdx.x * 32 + threadIdx.x;
  int64_t r0 = int64_t(blockIdx.y) * rows_per_block;
  int64_t r1 = min(M, r0 + rows_per_block);
  float a = 0.f, b = 0.f;
  if (c < C)
    for (int64_t r = r0 + threadIdx.y; r < r1; r += 8) {
      float2 v = f(r, c);
      a += v.x;
      b += v.y;
    }
  __shared__ float sa[8][33], sb[8][33];
  sa[threadIdx.y][threadIdx.x] = a;
  sb[threadIdx.y][threadIdx.x] = b;
  __syncthreads();
  if (threadIdx.y == 0 && c < C) {
    for (int i = 1; i < 8; ++i) {
      a += sa[i][threadIdx.x];
      b += sb[i][threadIdx.x];
    }
    atomicAdd(&acc[c], (double)a);
    atomicAdd(&acc[C + c], (double)b);
  }
}
template <class F>
void chan_reduce2(int64_t M, int C, F f, double* acc, cudaStream_t s) {
  CK(cudaMemsetAsync(acc, 0, sizeof(double) * 2 * C, s));
  int rpb = 256;
  int64_t gy = (M + rpb - 1) / rpb;
  while (gy > 4096) {
    rpb *= 2;
    gy = (M + rpb - 1) / rpb;
  }
  dim3 g(cdiv(C, 32), (unsigned)gy), b(32, 8);
  chan_reduce2_k<<<g, b, 0, s>>>(M, C, rpb, f, acc);
}

struct StatF {
  const act_t* z;
  int C;
  __device__ float2 operator()(int64_t r, int c) const {
    float v = to_f(z[r * C + c]);
    return make_float2(v, v * v);
  }
};
__global__ void bn_final_k(const double* acc, int64_t M, int C, float* mean, float* istd, float* rm, float* rv,
                           float mom, float eps) {
  int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= C) return;
  double m = acc[c] / M, var = acc[C + c] / M - m * m;
  if (var < 0) var = 0;
  mean[c] = (float)m;
  istd[c] = (float)(1.0 / sqrt(var + eps));
  if (rm) {
    double unb = M > 1 ? var * M / (M - 1) : var;
    rm[c] = (float)((1.0 - mom) * rm[c] + mom * m);
    rv[c] = (float)((1.0 - mom) * rv[c] + mom * unb);
  }
}
__global__ void bn_eval_k(int C, const float* rm, const float* rv, float eps, float* mean, float* istd) {
  int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= C) return;
  mean[c] = rm[c];
  istd[c] = rsqrtf(rv[c] + eps);
}
__global__ void bn_apply_k(const act_t* z, int64_t n, int C, const float* mean, const float* istd, const float* g,
                           const float* b, int act, act_t* y) {
  int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
  if (i >= n) return;
  int c = int(i % C);
  float v = (to_f(z[i]) - mean[c]) * istd[c] * g[c] + b[c];
  if (act) v = silu(v);
  y[i] = from_f<act_t>(v);
}
struct BwdF {
  const act_t *z, *dy;
  const float *mean, *istd, *g, *b;
  int C, act, ldy;
  __device__ float2 operator()(int64_t r, int c) const {
    float xh = (to_f(z[r * C + c]) - mean[c]) * istd[c];
    float d = to_f(dy[r * ldy + c]);
    if (act) d *= dsilu(xh * g[c] + b[c]);
    return make_float2(d, d * xh);
  }
};
// 채널 계수: k0 = γ·istd, sg = Σg/M, sgx = Σg·x̂/M  → coef [3C]
__global__ void bn_param_grad_k(const double* acc, int64_t M, int C, const float* gamma, const float* istd, float* dg,
                                float* db, float* coef) {
  int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= C) return;
  db[c] += (float)acc[c];
  dg[c] += (float)acc[C + c];
  coef[c] = gamma[c] * istd[c];
  coef[C + c] = (float)(acc[c] / (double)M);
  coef[2 * C + c] = (float)(acc[C + c] / (double)M);
}
// 행 하나 = 블록 y, 채널 = 스레드(32 비트 첨자)
__global__ void bn_dz_k(const act_t* z, const act_t* dy, int lddy, int M, int C, const float* mean, const float* istd,
                        const float* g, const float* b, int act, const float* coef, act_t* dz) {
  int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= C) return;
  float mu = mean[c], is = istd[c], gg = g[c], bb = b[c], k0 = coef[c], sg = coef[C + c], sgx = coef[2 * C + c];
  for (int r = blockIdx.y; r < M; r += gridDim.y) {
    int i = r * C + c;
    float xh = (to_f(z[i]) - mu) * is;
    float d = to_f(dy[r * lddy + c]);
    if (act) d *= dsilu(xh * gg + bb);
    dz[i] = from_f<act_t>(k0 * (d - sg - xh * sgx));
  }
}
__global__ void bn_apply2_k(const act_t* z, int M, int C, const float* mean, const float* istd, const float* g,
                            const float* b, int act, act_t* y, int ldy) {
  int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= C) return;
  float k = istd[c] * g[c], o = b[c] - mean[c] * k;
  for (int r = blockIdx.y; r < M; r += gridDim.y) {
    float v = to_f(z[r * C + c]) * k + o;
    if (act) v = silu(v);
    y[r * ldy + c] = from_f<act_t>(v);
  }
}
inline dim3 rows_grid(int64_t M, int C, int tx) {
  int gx = cdiv(C, tx);
  int64_t gy = std::min<int64_t>(M, std::max<int64_t>(1, 65535));
  gy = std::min<int64_t>(gy, std::max<int64_t>(1, (int64_t)2048 * 32 / gx));
  return dim3(gx, (unsigned)gy);
}
}  // namespace

void bn_stats(const act_t* z, int64_t M, int C, double* acc, float* mean, float* istd, float* run_mean,
              float* run_var, float momentum, float eps, cudaStream_t s) {
  if (vec_ok(C))
    bn_reduce_vec(false, z, nullptr, C, M, C, nullptr, nullptr, nullptr, nullptr, 0, acc, s);
  else
    chan_reduce2(M, C, StatF{z, C}, acc, s);
  bn_final_k<<<cdiv(C, 128), 128, 0, s>>>(acc, M, C, mean, istd, run_mean, run_var, momentum, eps);
}
void bn_eval_stats(int C, const float* rm, const float* rv, float eps, float* mean, float* istd, cudaStream_t s) {
  bn_eval_k<<<cdiv(C, 128), 128, 0, s>>>(C, rm, rv, eps, mean, istd);
}
void bn_apply(const act_t* z, int64_t M, int C, const float* mean, const float* istd, const float* gamma,
              const float* beta, int act, act_t* y, int ldy, cudaStream_t s) {
  if (vec_ok(C) && vec_ok(ldy)) return bn_apply_vec(z, M, C, mean, istd, gamma, beta, act, y, ldy, s);
  int tx = std::min(C, 128);
  bn_apply2_k<<<rows_grid(M, C, tx), tx, 0, s>>>(z, (int)M, C, mean, istd, gamma, beta, act, y, ldy);
}
void bn_backward(const act_t* z, const act_t* dy, int lddy, int64_t M, int C, const float* mean, const float* istd,
                 const float* gamma, const float* beta, int act, double* acc, act_t* dz, float* dgamma, float* dbeta,
                 cudaStream_t s) {
  bool v = vec_ok(C) && vec_ok(lddy);
  if (v)
    bn_reduce_vec(true, z, dy, lddy, M, C, mean, istd, gamma, beta, act, acc, s);
  else
    chan_reduce2(M, C, BwdF{z, dy, mean, istd, gamma, beta, C, act, lddy}, acc, s);
  float* coef = reinterpret_cast<float*>(acc + 2 * C);  // acc 뒤쪽 공간(3C float)
  bn_param_grad_k<<<cdiv(C, 128), 128, 0, s>>>(acc, M, C, gamma, istd, dgamma, dbeta, coef);
  if (v) return bn_dz_vec(z, dy, lddy, M, C, mean, istd, gamma, beta, act, coef, dz, s);
  int tx = std::min(C, 128);
  bn_dz_k<<<rows_grid(M, C, tx), tx, 0, s>>>(z, dy, lddy, (int)M, C, mean, istd, gamma, beta, act, coef, dz);
}

// ---- 모양 ----
namespace {
__global__ void copy_ch_k(const act_t* src, int sC, int s0, act_t* dst, int dC, int d0, int n, int64_t M,
                          bool accum) {
  int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
  if (i >= M * n) return;
  int64_t r = i / n;
  int c = int(i % n);
  act_t v = src[r * sC + s0 + c];
  act_t* p = dst + r * dC + d0 + c;
  *p = accum ? from_f<act_t>(to_f(*p) + to_f(v)) : v;
}
__global__ void add_k(const act_t* a, int lda, const act_t* b, int ldb, act_t* y, int ldy, int M, int C) {
  int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
  if (i >= (int64_t)M * C) return;
  int r = int(i / C), c = int(i % C);
  y[(int64_t)r * ldy + c] = from_f<act_t>(to_f(a[(int64_t)r * lda + c]) + to_f(b[(int64_t)r * ldb + c]));
}
__global__ void up2_k(const act_t* x, int ldx, act_t* y, int ldy, int N, int H, int W, int C) {
  int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
  int64_t n = int64_t(N) * 4 * H * W * C;
  if (i >= n) return;
  int c = int(i % C);
  int64_t t = i / C;
  int ox = int(t % (2 * W));
  t /= 2 * W;
  int oy = int(t % (2 * H));
  int b = int(t / (2 * H));
  y[((int64_t(b) * 2 * H + oy) * 2 * W + ox) * ldy + c] = x[((int64_t(b) * H + oy / 2) * W + ox / 2) * ldx + c];
}
__global__ void up2_bwd_k(const act_t* dy, int lddy, act_t* dx, int lddx, int N, int H, int W, int C) {
  int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
  int64_t n = int64_t(N) * H * W * C;
  if (i >= n) return;
  int c = int(i % C);
  int64_t t = i / C;
  int x = int(t % W);
  t /= W;
  int y = int(t % H);
  int b = int(t / H);
  float s = 0.f;
  for (int dy2 = 0; dy2 < 2; ++dy2)
    for (int dx2 = 0; dx2 < 2; ++dx2)
      s += to_f(dy[((int64_t(b) * 2 * H + 2 * y + dy2) * 2 * W + 2 * x + dx2) * lddy + c]);
  act_t* p = dx + ((int64_t(b) * H + y) * W + x) * lddx + c;
  *p = from_f<act_t>(to_f(*p) + s);
}
__global__ void maxpool_k(const act_t* x, int ldx, act_t* y, int ldy, uint8_t* arg, int N, int H, int W, int C,
                          int k) {
  int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
  int64_t n = int64_t(N) * H * W * C;
  if (i >= n) return;
  int c = int(i % C);
  int64_t t = i / C;
  int xx = int(t % W);
  t /= W;
  int yy = int(t % H);
  int b = int(t / H);
  int p = k / 2;
  float best = -FLT_MAX;
  int bi = 0;
  for (int u = 0; u < k; ++u) {
    int sy = yy + u - p;
    if (sy < 0 || sy >= H) continue;
    for (int v = 0; v < k; ++v) {
      int sx = xx + v - p;
      if (sx < 0 || sx >= W) continue;
      float val = to_f(x[((int64_t(b) * H + sy) * W + sx) * ldx + c]);
      if (val > best) {  // 처음 만난 최댓값(PyTorch 와 같음)
        best = val;
        bi = u * k + v;
      }
    }
  }
  y[(i / C) * ldy + c] = from_f<act_t>(best);
  arg[i] = (uint8_t)bi;
}
__global__ void maxpool_bwd_k(const act_t* dy, int lddy, const uint8_t* arg, act_t* dx, int lddx, int N, int H, int W,
                              int C, int k) {
  // 입력 칸마다 자기를 고른 출력들을 모은다(경합 없음)
  int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
  int64_t n = int64_t(N) * H * W * C;
  if (i >= n) return;
  int c = int(i % C);
  int64_t t = i / C;
  int xx = int(t % W);
  t /= W;
  int yy = int(t % H);
  int b = int(t / H);
  int p = k / 2;
  float s = 0.f;
  for (int u = 0; u < k; ++u) {
    int oy = yy - u + p;
    if (oy < 0 || oy >= H) continue;
    for (int v = 0; v < k; ++v) {
      int ox = xx - v + p;
      if (ox < 0 || ox >= W) continue;
      int64_t o = (int64_t(b) * H + oy) * W + ox;
      if (arg[o * C + c] == u * k + v) s += to_f(dy[o * lddy + c]);
    }
  }
  act_t* q = dx + (i / C) * lddx + c;
  *q = from_f<act_t>(to_f(*q) + s);
}
__global__ void add_bias_k(act_t* y, int64_t n, int C, const float* b) {
  int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
  if (i < n) y[i] = from_f<act_t>(to_f(y[i]) + b[i % C]);
}
struct SumF {
  const act_t* x;
  int C;
  __device__ float2 operator()(int64_t r, int c) const { return make_float2(to_f(x[r * C + c]), 0.f); }
};
__global__ void bias_final_k(const double* acc, int C, float* db) {
  int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c < C) db[c] += (float)acc[c];
}
__global__ void f2a_k(const float* x, act_t* y, int64_t n) {
  int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
  if (i < n) y[i] = from_f<act_t>(x[i]);
}
__global__ void a2f_k(const act_t* x, float* y, int64_t n, bool accum) {
  int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
  if (i < n) y[i] = accum ? y[i] + to_f(x[i]) : to_f(x[i]);
}
__global__ void scale_k(act_t* x, float a, int64_t n) {
  int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
  if (i < n) x[i] = from_f<act_t>(to_f(x[i]) * a);
}
}  // namespace

void copy_ch(const act_t* src, int sC, int s0, act_t* dst, int dC, int d0, int n, int64_t M, bool accum,
             cudaStream_t s) {
  if (vec_ok(sC) && vec_ok(dC) && vec_ok(s0) && vec_ok(d0) && vec_ok(n))
    return copy_ch_vec(src, sC, s0, dst, dC, d0, n, M, accum, s);
  copy_ch_k<<<nb(M * n), TPB, 0, s>>>(src, sC, s0, dst, dC, d0, n, M, accum);
}
void add(const act_t* a, int lda, const act_t* b, int ldb, act_t* y, int ldy, int64_t M, int C, cudaStream_t s) {
  if (vec_ok(C) && vec_ok(lda) && vec_ok(ldb) && vec_ok(ldy)) return add_vec(a, lda, b, ldb, y, ldy, M, C, s);
  add_k<<<nb(M * C), TPB, 0, s>>>(a, lda, b, ldb, y, ldy, (int)M, C);
}
void acc(const act_t* x, int ldx, act_t* y, int ldy, int64_t M, int C, cudaStream_t s) {
  add(x, ldx, y, ldy, y, ldy, M, C, s);
}
void up2(const act_t* x, int ldx, act_t* y, int ldy, int N, int H, int W, int C, cudaStream_t s) {
  if (vec_ok(C) && vec_ok(ldx) && vec_ok(ldy)) return up2_vec(x, ldx, y, ldy, N, H, W, C, s);
  up2_k<<<nb(int64_t(N) * 4 * H * W * C), TPB, 0, s>>>(x, ldx, y, ldy, N, H, W, C);
}
void up2_bwd(const act_t* dy, int lddy, act_t* dx, int lddx, int N, int H, int W, int C, cudaStream_t s) {
  if (vec_ok(C) && vec_ok(lddy) && vec_ok(lddx)) return up2_bwd_vec(dy, lddy, dx, lddx, N, H, W, C, s);
  up2_bwd_k<<<nb(int64_t(N) * H * W * C), TPB, 0, s>>>(dy, lddy, dx, lddx, N, H, W, C);
}
void maxpool(const act_t* x, int ldx, act_t* y, int ldy, uint8_t* arg, int N, int H, int W, int C, int k,
             cudaStream_t s) {
  maxpool_k<<<nb(int64_t(N) * H * W * C), TPB, 0, s>>>(x, ldx, y, ldy, arg, N, H, W, C, k);
}
void maxpool_bwd(const act_t* dy, int lddy, const uint8_t* arg, act_t* dx, int lddx, int N, int H, int W, int C, int k,
                 cudaStream_t s) {
  maxpool_bwd_k<<<nb(int64_t(N) * H * W * C), TPB, 0, s>>>(dy, lddy, arg, dx, lddx, N, H, W, C, k);
}
void add_bias(act_t* y, int64_t M, int C, const float* b, cudaStream_t s) {
  add_bias_k<<<nb(M * C), TPB, 0, s>>>(y, M * C, C, b);
}
void bias_grad(const act_t* dy, int64_t M, int C, double* acc, float* db, cudaStream_t s) {
  chan_reduce2(M, C, SumF{dy, C}, acc, s);
  bias_final_k<<<cdiv(C, 128), 128, 0, s>>>(acc, C, db);
}
// ---- depthwise 3×3 ----
namespace {
// 블록: 채널 32 × 행 8. 스레드 하나가 (픽셀, 채널) 하나
__global__ void dw3_k(const act_t* x, const act_t* w, act_t* y, int N, int H, int W, int C, int flip, int accum) {
  int c = blockIdx.x * 32 + threadIdx.x;
  if (c >= C) return;
  float k[9];
  for (int j = 0; j < 9; ++j) k[j] = to_f(w[c * 9 + (flip ? 8 - j : j)]);
  int P = N * H * W;
  for (int p = blockIdx.y * 8 + threadIdx.y; p < P; p += gridDim.y * 8) {
    int xx = p % W, t = p / W, yy = t % H, n = t / H;
    float s = 0.f;
    for (int ky = 0; ky < 3; ++ky) {
      int sy = yy + ky - 1;
      if (sy < 0 || sy >= H) continue;
      for (int kx = 0; kx < 3; ++kx) {
        int sx = xx + kx - 1;
        if (sx < 0 || sx >= W) continue;
        s += to_f(x[((n * H + sy) * W + sx) * C + c]) * k[ky * 3 + kx];
      }
    }
    act_t* o = y + int64_t(p) * C + c;
    *o = from_f<act_t>(accum ? to_f(*o) + s : s);
  }
}
__global__ void dw3_wgrad_k(const act_t* x, const act_t* dy, float* tmp, int N, int H, int W, int C) {
  int c = blockIdx.x * 32 + threadIdx.x;
  float a[9] = {0, 0, 0, 0, 0, 0, 0, 0, 0};
  int P = N * H * W;
  if (c < C)
    for (int p = blockIdx.y * 8 + threadIdx.y; p < P; p += gridDim.y * 8) {
      int xx = p % W, t = p / W, yy = t % H, n = t / H;
      float g = to_f(dy[int64_t(p) * C + c]);
      if (g == 0.f) continue;
      for (int ky = 0; ky < 3; ++ky) {
        int sy = yy + ky - 1;
        if (sy < 0 || sy >= H) continue;
        for (int kx = 0; kx < 3; ++kx) {
          int sx = xx + kx - 1;
          if (sx < 0 || sx >= W) continue;
          a[ky * 3 + kx] += g * to_f(x[((n * H + sy) * W + sx) * C + c]);
        }
      }
    }
  __shared__ float sh[8][9][33];
  for (int j = 0; j < 9; ++j) sh[threadIdx.y][j][threadIdx.x] = a[j];
  __syncthreads();
  if (threadIdx.y == 0 && c < C)
    for (int j = 0; j < 9; ++j) {
      float v = 0.f;
      for (int r = 0; r < 8; ++r) v += sh[r][j][threadIdx.x];
      atomicAdd(&tmp[c * 9 + j], v);
    }
}
__global__ void f2a_plain_k(const float* x, act_t* y, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) y[i] = from_f<act_t>(x[i]);
}
inline dim3 dw_grid(int P, int C) { return dim3(cdiv(C, 32), (unsigned)std::min(cdiv(P, 8), std::max(1, 4096 / cdiv(C, 32)))); }
}  // namespace
void dw3_fwd(const act_t* x, const act_t* w, act_t* y, int N, int H, int W, int C, cudaStream_t s) {
  dw3_k<<<dw_grid(N * H * W, C), dim3(32, 8), 0, s>>>(x, w, y, N, H, W, C, 0, 0);
}
void dw3_dgrad(const act_t* dy, const act_t* w, act_t* dx, int N, int H, int W, int C, cudaStream_t s) {
  // stride 1 · pad 1 이면 입력 기울기 = 뒤집은 커널로 합성곱
  dw3_k<<<dw_grid(N * H * W, C), dim3(32, 8), 0, s>>>(dy, w, dx, N, H, W, C, 1, 1);
}
void dw3_wgrad(const act_t* x, const act_t* dy, act_t* dw, float* tmp, int N, int H, int W, int C, cudaStream_t s) {
  CK(cudaMemsetAsync(tmp, 0, sizeof(float) * 9 * C, s));
  dw3_wgrad_k<<<dw_grid(N * H * W, C), dim3(32, 8), 0, s>>>(x, dy, tmp, N, H, W, C);
  f2a_plain_k<<<cdiv(9 * C, 256), 256, 0, s>>>(tmp, dw, 9 * C);
}

void f32_to_act(const float* x, act_t* y, int64_t n, cudaStream_t s) { f2a_k<<<nb(n), TPB, 0, s>>>(x, y, n); }
void act_to_f32(const act_t* x, float* y, int64_t n, bool accum, cudaStream_t s) {
  a2f_k<<<nb(n), TPB, 0, s>>>(x, y, n, accum);
}
void scale_act(act_t* x, float a, int64_t n, cudaStream_t s) { scale_k<<<nb(n), TPB, 0, s>>>(x, a, n); }

}  // namespace kern
