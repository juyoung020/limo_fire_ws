#include <cfloat>

#include "kern.cuh"

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
  int C, act;
  __device__ float2 operator()(int64_t r, int c) const {
    float xh = (to_f(z[r * C + c]) - mean[c]) * istd[c];
    float d = to_f(dy[r * C + c]);
    if (act) d *= dsilu(xh * g[c] + b[c]);
    return make_float2(d, d * xh);
  }
};
__global__ void bn_dz_k(const act_t* z, const act_t* dy, int64_t n, int64_t M, int C, const float* mean,
                        const float* istd, const float* g, const float* b, int act, const double* acc, act_t* dz) {
  int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
  if (i >= n) return;
  int c = int(i % C);
  float xh = (to_f(z[i]) - mean[c]) * istd[c];
  float d = to_f(dy[i]);
  if (act) d *= dsilu(xh * g[c] + b[c]);
  float sg = (float)(acc[c] / M), sgx = (float)(acc[C + c] / M);
  dz[i] = from_f<act_t>(g[c] * istd[c] * (d - sg - xh * sgx));
}
__global__ void bn_param_grad_k(const double* acc, int C, float* dg, float* db) {
  int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= C) return;
  db[c] += (float)acc[c];
  dg[c] += (float)acc[C + c];
}
}  // namespace

void bn_stats(const act_t* z, int64_t M, int C, double* acc, float* mean, float* istd, float* run_mean,
              float* run_var, float momentum, float eps, cudaStream_t s) {
  chan_reduce2(M, C, StatF{z, C}, acc, s);
  bn_final_k<<<cdiv(C, 128), 128, 0, s>>>(acc, M, C, mean, istd, run_mean, run_var, momentum, eps);
}
void bn_eval_stats(int C, const float* rm, const float* rv, float eps, float* mean, float* istd, cudaStream_t s) {
  bn_eval_k<<<cdiv(C, 128), 128, 0, s>>>(C, rm, rv, eps, mean, istd);
}
void bn_apply(const act_t* z, int64_t M, int C, const float* mean, const float* istd, const float* gamma,
              const float* beta, int act, act_t* y, cudaStream_t s) {
  int64_t n = M * C;
  bn_apply_k<<<nb(n), TPB, 0, s>>>(z, n, C, mean, istd, gamma, beta, act, y);
}
void bn_backward(const act_t* z, const act_t* dy, int64_t M, int C, const float* mean, const float* istd,
                 const float* gamma, const float* beta, int act, double* acc, act_t* dz, float* dgamma, float* dbeta,
                 cudaStream_t s) {
  chan_reduce2(M, C, BwdF{z, dy, mean, istd, gamma, beta, C, act}, acc, s);
  bn_param_grad_k<<<cdiv(C, 128), 128, 0, s>>>(acc, C, dgamma, dbeta);
  int64_t n = M * C;
  bn_dz_k<<<nb(n), TPB, 0, s>>>(z, dy, n, M, C, mean, istd, gamma, beta, act, acc, dz);
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
__global__ void add_k(const act_t* a, const act_t* b, act_t* y, int64_t n) {
  int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
  if (i < n) y[i] = from_f<act_t>(to_f(a[i]) + to_f(b[i]));
}
__global__ void acc_k(const act_t* x, act_t* y, int64_t n) {
  int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
  if (i < n) y[i] = from_f<act_t>(to_f(y[i]) + to_f(x[i]));
}
__global__ void up2_k(const act_t* x, act_t* y, int N, int H, int W, int C) {
  int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
  int64_t n = int64_t(N) * 4 * H * W * C;
  if (i >= n) return;
  int c = int(i % C);
  int64_t t = i / C;
  int ox = int(t % (2 * W));
  t /= 2 * W;
  int oy = int(t % (2 * H));
  int b = int(t / (2 * H));
  y[i] = x[((int64_t(b) * H + oy / 2) * W + ox / 2) * C + c];
}
__global__ void up2_bwd_k(const act_t* dy, act_t* dx, int N, int H, int W, int C) {
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
      s += to_f(dy[((int64_t(b) * 2 * H + 2 * y + dy2) * 2 * W + 2 * x + dx2) * C + c]);
  dx[i] = from_f<act_t>(to_f(dx[i]) + s);
}
__global__ void maxpool_k(const act_t* x, act_t* y, uint8_t* arg, int N, int H, int W, int C, int k) {
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
      float val = to_f(x[((int64_t(b) * H + sy) * W + sx) * C + c]);
      if (val > best) {  // 처음 만난 최댓값(PyTorch 와 같음)
        best = val;
        bi = u * k + v;
      }
    }
  }
  y[i] = from_f<act_t>(best);
  arg[i] = (uint8_t)bi;
}
__global__ void maxpool_bwd_k(const act_t* dy, const uint8_t* arg, act_t* dx, int N, int H, int W, int C, int k) {
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
      int64_t o = ((int64_t(b) * H + oy) * W + ox) * C + c;
      if (arg[o] == u * k + v) s += to_f(dy[o]);
    }
  }
  dx[i] = from_f<act_t>(to_f(dx[i]) + s);
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
  copy_ch_k<<<nb(M * n), TPB, 0, s>>>(src, sC, s0, dst, dC, d0, n, M, accum);
}
void add(const act_t* a, const act_t* b, act_t* y, int64_t n, cudaStream_t s) { add_k<<<nb(n), TPB, 0, s>>>(a, b, y, n); }
void acc(const act_t* x, act_t* y, int64_t n, cudaStream_t s) { acc_k<<<nb(n), TPB, 0, s>>>(x, y, n); }
void up2(const act_t* x, act_t* y, int N, int H, int W, int C, cudaStream_t s) {
  up2_k<<<nb(int64_t(N) * 4 * H * W * C), TPB, 0, s>>>(x, y, N, H, W, C);
}
void up2_bwd(const act_t* dy, act_t* dx, int N, int H, int W, int C, cudaStream_t s) {
  up2_bwd_k<<<nb(int64_t(N) * H * W * C), TPB, 0, s>>>(dy, dx, N, H, W, C);
}
void maxpool(const act_t* x, act_t* y, uint8_t* arg, int N, int H, int W, int C, int k, cudaStream_t s) {
  maxpool_k<<<nb(int64_t(N) * H * W * C), TPB, 0, s>>>(x, y, arg, N, H, W, C, k);
}
void maxpool_bwd(const act_t* dy, const uint8_t* arg, act_t* dx, int N, int H, int W, int C, int k, cudaStream_t s) {
  maxpool_bwd_k<<<nb(int64_t(N) * H * W * C), TPB, 0, s>>>(dy, arg, dx, N, H, W, C, k);
}
void add_bias(act_t* y, int64_t M, int C, const float* b, cudaStream_t s) {
  add_bias_k<<<nb(M * C), TPB, 0, s>>>(y, M * C, C, b);
}
void bias_grad(const act_t* dy, int64_t M, int C, double* acc, float* db, cudaStream_t s) {
  chan_reduce2(M, C, SumF{dy, C}, acc, s);
  bias_final_k<<<cdiv(C, 128), 128, 0, s>>>(acc, C, db);
}
void f32_to_act(const float* x, act_t* y, int64_t n, cudaStream_t s) { f2a_k<<<nb(n), TPB, 0, s>>>(x, y, n); }
void act_to_f32(const act_t* x, float* y, int64_t n, bool accum, cudaStream_t s) {
  a2f_k<<<nb(n), TPB, 0, s>>>(x, y, n, accum);
}
void scale_act(act_t* x, float a, int64_t n, cudaStream_t s) { scale_k<<<nb(n), TPB, 0, s>>>(x, a, n); }

}  // namespace kern
