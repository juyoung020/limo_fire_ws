// BatchNorm·모양 커널의 벡터판(채널 수가 V 의 배수일 때). kern.cu 가 고른다.
//  - 통계·역방향 합: 블록 = (채널 묶음 × 행 갈래), 스레드마다 채널 V 개를 레지스터에 모아 공유 메모리로 합친 뒤
//    채널마다 double atomic 하나.
//  - 적용·dz: 16 B 벡터마다 한 번, 채널 계수는 공유 메모리.
#include "kern.cuh"
#include "vec.cuh"

namespace kern {
namespace {
__device__ __forceinline__ float silu(float x) { return x / (1.f + __expf(-x)); }
__device__ __forceinline__ float dsilu(float x) {
  float sg = 1.f / (1.f + __expf(-x));
  return sg * (1.f + x * (1.f - sg));
}

// BWD=false: Σz, Σz²   BWD=true: Σd, Σd·x̂ (d = dy·act'(·))
template <bool BWD>
__global__ void __launch_bounds__(256) reduce_v(const act_t* z, const act_t* dy, int lddy, int M, int C,
                                                 const float* mean,
                                                 const float* istd, const float* g, const float* b, int act, int rpb,
                                                 float* part) {
  const int groups = C / V, gpb = min(groups, 256), lanes = 256 / gpb;
  int gi = blockIdx.x * gpb + threadIdx.x % gpb, lane = threadIdx.x / gpb;
  bool ok = gi < groups && lane < lanes;
  int c0 = gi * V;
  float sa[V], sq[V], mu[V], is[V], gg[V], bb[V];
#pragma unroll
  for (int j = 0; j < V; ++j) sa[j] = sq[j] = 0.f;
  if (ok && BWD)
#pragma unroll
    for (int j = 0; j < V; ++j) mu[j] = mean[c0 + j], is[j] = istd[c0 + j], gg[j] = g[c0 + j], bb[j] = b[c0 + j];
  int r0 = blockIdx.y * rpb, r1 = min(M, r0 + rpb);
  if (ok)
    for (int r = r0 + lane; r < r1; r += lanes) {
      float zz[V];
      vld(z + (size_t)r * C + c0, zz);
      if (!BWD) {
#pragma unroll
        for (int j = 0; j < V; ++j) sa[j] += zz[j], sq[j] += zz[j] * zz[j];
      } else {
        float dd[V];
        vld(dy + (size_t)r * lddy + c0, dd);
#pragma unroll
        for (int j = 0; j < V; ++j) {
          float xh = (zz[j] - mu[j]) * is[j], d = dd[j];
          if (act) d *= dsilu(xh * gg[j] + bb[j]);
          sa[j] += d;
          sq[j] += d * xh;
        }
      }
    }
  __shared__ float sh[2][256 * V];
#pragma unroll
  for (int j = 0; j < V; ++j) {
    sh[0][threadIdx.x * V + j] = sa[j];
    sh[1][threadIdx.x * V + j] = sq[j];
  }
  __syncthreads();
  // 채널(블록 안 gpb·V 개)마다 갈래 합
  for (int t = threadIdx.x; t < gpb * V; t += 256) {
    int gl = t / V, j = t % V, gi2 = blockIdx.x * gpb + gl;
    if (gi2 >= groups) continue;
    float a = 0.f, q = 0.f;
    for (int l = 0; l < lanes; ++l) {
      int src = l * gpb + gl;
      a += sh[0][src * V + j];
      q += sh[1][src * V + j];
    }
    int c = gi2 * V + j;
    part[(size_t)blockIdx.y * 2 * C + c] = a;
    part[(size_t)blockIdx.y * 2 * C + C + c] = q;
  }
}
// 블록별 부분합 [gy][2C] → acc [2C] (double)
__global__ void reduce_part_k(const float* part, int gy, int C, double* acc) {
  // 블록 = 채널 32 × 갈래 8
  int c = blockIdx.x * 32 + threadIdx.x;
  float s = 0.f;
  if (c < 2 * C)
    for (int y = threadIdx.y; y < gy; y += 8) s += part[(size_t)y * 2 * C + c];
  __shared__ float sh[8][33];
  sh[threadIdx.y][threadIdx.x] = s;
  __syncthreads();
  if (threadIdx.y == 0 && c < 2 * C) {
    double t = 0;
    for (int i = 0; i < 8; ++i) t += sh[i][threadIdx.x];
    acc[c] = t;
  }
}

__global__ void apply_v(const act_t* z, int M, int C, const float* mean, const float* istd, const float* g,
                        const float* b, int act, act_t* y, int ldy) {
  extern __shared__ float sm[];
  float *k = sm, *o = sm + C;
  for (int c = threadIdx.x; c < C; c += blockDim.x) {
    k[c] = istd[c] * g[c];
    o[c] = b[c] - mean[c] * k[c];
  }
  __syncthreads();
  int groups = C / V;
  int n = M * groups;
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
    int c0 = (i % groups) * V;
    float v[V];
    vld(z + (size_t)i * V, v);
#pragma unroll
    for (int j = 0; j < V; ++j) {
      float t = v[j] * k[c0 + j] + o[c0 + j];
      v[j] = act ? silu(t) : t;
    }
    vst(y + (size_t)(i / groups) * ldy + c0, v);
  }
}

__global__ void dz_v(const act_t* z, const act_t* dy, int lddy, int M, int C, const float* mean, const float* istd,
                     const float* g, const float* b, int act, const float* coef, act_t* dz) {
  extern __shared__ float sm[];
  float *mu = sm, *is = sm + C, *gg = sm + 2 * C, *bb = sm + 3 * C, *k0 = sm + 4 * C, *sg = sm + 5 * C,
        *sgx = sm + 6 * C;
  for (int c = threadIdx.x; c < C; c += blockDim.x) {
    mu[c] = mean[c], is[c] = istd[c], gg[c] = g[c], bb[c] = b[c];
    k0[c] = coef[c], sg[c] = coef[C + c], sgx[c] = coef[2 * C + c];
  }
  __syncthreads();
  int groups = C / V;
  int n = M * groups;
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
    int c0 = (i % groups) * V;
    float zz[V], dd[V];
    vld(z + (size_t)i * V, zz);
    vld(dy + (size_t)(i / groups) * lddy + c0, dd);
#pragma unroll
    for (int j = 0; j < V; ++j) {
      int c = c0 + j;
      float xh = (zz[j] - mu[c]) * is[c], d = dd[j];
      if (act) d *= dsilu(xh * gg[c] + bb[c]);
      dd[j] = k0[c] * (d - sg[c] - xh * sgx[c]);
    }
    vst(dz + (size_t)i * V, dd);
  }
}

__global__ void copy_v(const act_t* src, int sC, int s0, act_t* dst, int dC, int d0, int nv, int M, int accum) {
  int n = M * nv;
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
    int r = i / nv, cv = i % nv;
    float a[V];
    vld(src + (size_t)r * sC + s0 + cv * V, a);
    act_t* p = dst + (size_t)r * dC + d0 + cv * V;
    if (accum) {
      float b[V];
      vld(p, b);
#pragma unroll
      for (int j = 0; j < V; ++j) a[j] += b[j];
    }
    vst(p, a);
  }
}
__global__ void add_v(const act_t* a, int lda, const act_t* b, int ldb, act_t* y, int ldy, int M, int C) {
  int g = C / V, n = M * g;
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
    int r = i / g, c = (i % g) * V;
    float x[V], z[V];
    vld(a + (size_t)r * lda + c, x);
    vld(b + (size_t)r * ldb + c, z);
#pragma unroll
    for (int j = 0; j < V; ++j) x[j] += z[j];
    vst(y + (size_t)r * ldy + c, x);
  }
}
__global__ void up2_v(const act_t* x, int ldx, act_t* y, int ldy, int N, int H, int W, int C) {
  int cv = C / V, n = N * 4 * H * W * cv;
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
    int c = i % cv, t = i / cv, ox = t % (2 * W);
    t /= 2 * W;
    int oy = t % (2 * H), bb = t / (2 * H);
    *reinterpret_cast<uint4*>(y + ((size_t)(bb * 2 * H + oy) * 2 * W + ox) * ldy + c * V) =
        *reinterpret_cast<const uint4*>(x + ((size_t)(bb * H + oy / 2) * W + ox / 2) * ldx + c * V);
  }
}
__global__ void up2_bwd_v(const act_t* dy, int lddy, act_t* dx, int lddx, int N, int H, int W, int C) {
  int cv = C / V, n = N * H * W * cv;
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
    int c = i % cv, t = i / cv, x = t % W;
    t /= W;
    int y = t % H, bb = t / H;
    act_t* p = dx + ((size_t)(bb * H + y) * W + x) * lddx + c * V;
    float s[V], q[V];
    vld(p, s);
    for (int a = 0; a < 2; ++a)
      for (int e = 0; e < 2; ++e) {
        vld(dy + ((size_t)(bb * 2 * H + 2 * y + a) * 2 * W + 2 * x + e) * lddy + c * V, q);
#pragma unroll
        for (int j = 0; j < V; ++j) s[j] += q[j];
      }
    vst(p, s);
  }
}
inline int grid_for(int64_t n) { return (int)std::min<int64_t>((n + 255) / 256, 70 * 16); }
}  // namespace

bool vec_ok(int C) { return C % V == 0; }

static float* g_part = nullptr;
static size_t g_part_n = 0;
void bn_reduce_vec(bool bwd, const act_t* z, const act_t* dy, int lddy, int64_t M, int C, const float* mean,
                   const float* istd, const float* g, const float* b, int act, double* acc, cudaStream_t s) {
  int groups = C / V, gpb = std::min(groups, 256), gx = cdiv(groups, gpb);
  int target = 560 / gx + 1;  // SM 70 × 8
  int rpb = std::max<int>(1, (int)cdiv(M, target));
  int gy = cdiv(M, rpb);
  size_t need = (size_t)gy * 2 * C;
  if (need > g_part_n) {  // 처음 몇 번만(그래프 잡기 전 첫 반복에서 다 잡힌다)
    if (g_part) CK(cudaFree(g_part));
    g_part_n = std::max(need, (size_t)4 << 20);
    g_part = dalloc<float>(g_part_n);
  }
  dim3 grid(gx, gy);
  if (bwd)
    reduce_v<true><<<grid, 256, 0, s>>>(z, dy, lddy, (int)M, C, mean, istd, g, b, act, rpb, g_part);
  else
    reduce_v<false><<<grid, 256, 0, s>>>(z, dy, lddy, (int)M, C, mean, istd, g, b, act, rpb, g_part);
  reduce_part_k<<<cdiv(2 * C, 32), dim3(32, 8), 0, s>>>(g_part, gy, C, acc);
}
void bn_apply_vec(const act_t* z, int64_t M, int C, const float* mean, const float* istd, const float* g,
                  const float* b, int act, act_t* y, int ldy, cudaStream_t s) {
  apply_v<<<grid_for(M * C / V), 256, 2 * C * sizeof(float), s>>>(z, (int)M, C, mean, istd, g, b, act, y, ldy);
}
void bn_dz_vec(const act_t* z, const act_t* dy, int lddy, int64_t M, int C, const float* mean, const float* istd,
               const float* g, const float* b, int act, const float* coef, act_t* dz, cudaStream_t s) {
  dz_v<<<grid_for(M * C / V), 256, 7 * C * sizeof(float), s>>>(z, dy, lddy, (int)M, C, mean, istd, g, b, act, coef,
                                                               dz);
}
void copy_ch_vec(const act_t* src, int sC, int s0, act_t* dst, int dC, int d0, int n, int64_t M, bool accum,
                 cudaStream_t s) {
  copy_v<<<grid_for(M * n / V), 256, 0, s>>>(src, sC, s0, dst, dC, d0, n / V, (int)M, accum);
}
void add_vec(const act_t* a, int lda, const act_t* b, int ldb, act_t* y, int ldy, int64_t M, int C, cudaStream_t s) {
  add_v<<<grid_for(M * C / V), 256, 0, s>>>(a, lda, b, ldb, y, ldy, (int)M, C);
}
void up2_vec(const act_t* x, int ldx, act_t* y, int ldy, int N, int H, int W, int C, cudaStream_t s) {
  up2_v<<<grid_for((int64_t)N * 4 * H * W * C / V), 256, 0, s>>>(x, ldx, y, ldy, N, H, W, C);
}
void up2_bwd_vec(const act_t* dy, int lddy, act_t* dx, int lddx, int N, int H, int W, int C, cudaStream_t s) {
  up2_bwd_v<<<grid_for((int64_t)N * H * W * C / V), 256, 0, s>>>(dy, lddy, dx, lddx, N, H, W, C);
}
}  // namespace kern
