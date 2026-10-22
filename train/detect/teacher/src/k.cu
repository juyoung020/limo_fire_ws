#include <cfloat>
#include <cmath>
#include <map>
#include <mutex>
#include <tuple>
#include <type_traits>

#include "k.cuh"

namespace k {

namespace {
template <typename T>
__device__ __forceinline__ float tof(T x) {
  if constexpr (std::is_same_v<T, float>)
    return x;
  else if constexpr (std::is_same_v<T, f16>)
    return __half2float(x);
  else
    return __bfloat162float(x);
}
template <typename T>
__device__ __forceinline__ T fromf(float x) {
  if constexpr (std::is_same_v<T, float>)
    return x;
  else if constexpr (std::is_same_v<T, f16>)
    return __float2half(x);
  else
    return __float2bfloat16(x);
}
template <typename T>
constexpr cudaDataType_t cuda_type() {
  return std::is_same_v<T, float> ? CUDA_R_32F : std::is_same_v<T, f16> ? CUDA_R_16F : CUDA_R_16BF;
}
__device__ __forceinline__ float gelu(float x) { return 0.5f * x * (1.f + erff(x * 0.70710678118654752f)); }
inline int nblk(long long n, int t) { return (int)((n + t - 1) / t); }
}  // namespace

// ---------------- GEMM ----------------
template <typename T>
void linear(cublasHandle_t h, const T* X, int ldx, const T* W, int ldw, float* Y, int ldy, int M, int N, int K,
            float alpha, float beta) {
  // 행 우선 Y = X W^T  ⇔  열 우선 Y^T = W X^T
  CB(cublasGemmEx(h, CUBLAS_OP_T, CUBLAS_OP_N, N, M, K, &alpha, W, cuda_type<T>(), ldw, X, cuda_type<T>(), ldx, &beta,
                  Y, CUDA_R_32F, ldy, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
}
template <typename T>
void linear_batched(cublasHandle_t h, const T* X, int ldx, long long sx, const T* W, int ldw, long long sw, float* Y,
                    int ldy, long long sy, int M, int N, int K, int batch, float beta) {
  float alpha = 1.f;
  CB(cublasGemmStridedBatchedEx(h, CUBLAS_OP_T, CUBLAS_OP_N, N, M, K, &alpha, W, cuda_type<T>(), ldw, sw, X,
                                cuda_type<T>(), ldx, sx, &beta, Y, CUDA_R_32F, ldy, sy, batch, CUBLAS_COMPUTE_32F,
                                CUBLAS_GEMM_DEFAULT));
}


// ---------------- cuBLASLt ----------------
#define CL(x)                                                                    \
  do {                                                                           \
    cublasStatus_t st_cl_ = (x);                                                 \
    if (st_cl_ != CUBLAS_STATUS_SUCCESS) {                                       \
      std::fprintf(stderr, "cuBLASLt %s:%d 상태 %d\n", __FILE__, __LINE__, (int)st_cl_); \
      std::abort();                                                              \
    }                                                                            \
  } while (0)
void Lt::init(cudaStream_t st) {
  s = st;
  CL(cublasLtCreate(&h));
  CK(cudaMalloc(&ws, wsz));
}
void Lt::destroy() {
  if (h) cublasLtDestroy(h);
  if (ws) cudaFree(ws);
  h = nullptr;
  ws = nullptr;
}
namespace {
struct LtPlan {
  cublasLtMatmulDesc_t op;
  cublasLtMatrixLayout_t a, b, c;
  cublasLtMatmulAlgo_t algo;
};
std::map<std::tuple<void*, int, int, int, int, int, int, int, int, int, int>, LtPlan> g_plans;
std::mutex g_plans_mu;
}  // namespace
template <typename T, typename To>
void lt_linear(Lt& lt, const T* X, int ldx, const T* W, int ldw, To* Dm, int ldd, int M, int N, int K, const To* bias,
               int epi, float beta) {
  // 열 우선으로 보면 D^T[N, M] = W[N, K] (op T) · X^T (op N)
  cudaDataType_t ti = cuda_type<T>(), to = cuda_type<To>();
  auto key = std::make_tuple((void*)lt.h, M, N, K, ldx, ldw, ldd, (int)ti, (int)to, epi, beta != 0.f ? 1 : 0);
  LtPlan* pl;
  {
    std::lock_guard<std::mutex> lk(g_plans_mu);
    auto it = g_plans.find(key);
    if (it == g_plans.end()) {
      LtPlan p;
      CL(cublasLtMatmulDescCreate(&p.op, CUBLAS_COMPUTE_32F, CUDA_R_32F));
      cublasOperation_t tA = CUBLAS_OP_T, tB = CUBLAS_OP_N;
      CL(cublasLtMatmulDescSetAttribute(p.op, CUBLASLT_MATMUL_DESC_TRANSA, &tA, sizeof(tA)));
      CL(cublasLtMatmulDescSetAttribute(p.op, CUBLASLT_MATMUL_DESC_TRANSB, &tB, sizeof(tB)));
      cublasLtEpilogue_t e = epi == 2 ? CUBLASLT_EPILOGUE_RELU_BIAS : CUBLASLT_EPILOGUE_BIAS;
      CL(cublasLtMatmulDescSetAttribute(p.op, CUBLASLT_MATMUL_DESC_EPILOGUE, &e, sizeof(e)));
      cudaDataType_t bt = to;
      CL(cublasLtMatmulDescSetAttribute(p.op, CUBLASLT_MATMUL_DESC_BIAS_DATA_TYPE, &bt, sizeof(bt)));
      const void* bp = bias;
      CL(cublasLtMatmulDescSetAttribute(p.op, CUBLASLT_MATMUL_DESC_BIAS_POINTER, &bp, sizeof(bp)));
      CL(cublasLtMatrixLayoutCreate(&p.a, ti, K, N, ldw));
      CL(cublasLtMatrixLayoutCreate(&p.b, ti, K, M, ldx));
      CL(cublasLtMatrixLayoutCreate(&p.c, to, N, M, ldd));
      cublasLtMatmulPreference_t pref;
      CL(cublasLtMatmulPreferenceCreate(&pref));
      CL(cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &lt.wsz,
                                              sizeof(lt.wsz)));
      cublasLtMatmulHeuristicResult_t res;
      int n = 0;
      CL(cublasLtMatmulAlgoGetHeuristic(lt.h, p.op, p.a, p.b, p.c, p.c, pref, 1, &res, &n));
      cublasLtMatmulPreferenceDestroy(pref);
      if (n == 0) {
        std::fprintf(stderr, "cuBLASLt 알고리즘 없음 M=%d N=%d K=%d\n", M, N, K);
        std::abort();
      }
      p.algo = res.algo;
      it = g_plans.emplace(key, p).first;
    }
    pl = &it->second;
  }
  const void* bp = bias;
  CL(cublasLtMatmulDescSetAttribute(pl->op, CUBLASLT_MATMUL_DESC_BIAS_POINTER, &bp, sizeof(bp)));
  float alpha = 1.f;
  CL(cublasLtMatmul(lt.h, pl->op, &alpha, W, pl->a, X, pl->b, &beta, Dm, pl->c, Dm, pl->c, &pl->algo, lt.ws, lt.wsz,
                    lt.s));
}
template void lt_linear<f16, f16>(Lt&, const f16*, int, const f16*, int, f16*, int, int, int, int, const f16*, int,
                                  float);
template void lt_linear<f16, float>(Lt&, const f16*, int, const f16*, int, float*, int, int, int, int, const float*,
                                    int, float);
template void lt_linear<bf16, bf16>(Lt&, const bf16*, int, const bf16*, int, bf16*, int, int, int, int, const bf16*,
                                    int, float);
template void lt_linear<bf16, float>(Lt&, const bf16*, int, const bf16*, int, float*, int, int, int, int,
                                     const float*, int, float);

template <typename T>
__global__ void gelu_inplace_k(T* y, long long n) {
  long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x;
  if (i < n) y[i] = fromf<T>(gelu(tof(y[i])));
}
template <typename T>
void gelu_inplace(T* y, long long n, cudaStream_t s) {
  gelu_inplace_k<<<nblk(n, 256), 256, 0, s>>>(y, n);
}
__global__ void window_scatter_k(const float* y, float* x, int h, int w, int C, int ws) {
  long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x;
  if (i >= (long long)h * w * C) return;
  int c = (int)(i % C), t = (int)(i / C);
  int nwx = w / ws, win = t / (ws * ws), rr = t % (ws * ws);
  int yy = (win / nwx) * ws + rr / ws, xx = (win % nwx) * ws + rr % ws;
  x[((long long)yy * w + xx) * C + c] = y[i];
}
void window_scatter(const float* y, float* x, int h, int w, int C, int ws, cudaStream_t s) {
  window_scatter_k<<<nblk((long long)h * w * C, 256), 256, 0, s>>>(y, x, h, w, C, ws);
}
template <typename T>
__global__ void layernorm_pos_k(const float* x, const float* g, const float* b, float eps, const float* pos, int npos,
                                T* y1, T* y2, int C) {
  const float* xr = x + (long long)blockIdx.x * C;
  __shared__ float red[32];
  float v[8];
  int n = 0;
  float s = 0.f;
  for (int c = threadIdx.x; c < C; c += blockDim.x) {
    v[n] = xr[c];
    s += v[n++];
  }
  auto block_sum = [&](float a) {
    for (int o = 16; o > 0; o >>= 1) a += __shfl_xor_sync(0xffffffff, a, o);
    if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = a;
    __syncthreads();
    float t = 0.f;
    for (int i = 0; i < (int)(blockDim.x >> 5); ++i) t += red[i];
    __syncthreads();
    return t;
  };
  float mean = block_sum(s) / C;
  float q = 0.f;
  for (int i = 0; i < n; ++i) {
    float d = v[i] - mean;
    q += d * d;
  }
  float rstd = rsqrtf(block_sum(q) / C + eps);
  const float* pr = pos + (long long)(blockIdx.x % npos) * C;
  n = 0;
  for (int c = threadIdx.x; c < C; c += blockDim.x) {
    float yv = (v[n++] - mean) * rstd * g[c] + b[c];
    y1[(long long)blockIdx.x * C + c] = fromf<T>(yv);
    y2[(long long)blockIdx.x * C + c] = fromf<T>(yv + pr[c]);
  }
}
template <typename T>
void layernorm_pos(const float* x, const float* g, const float* b, float eps, const float* pos, int npos, T* y1, T* y2,
                   int rows, int C, cudaStream_t s) {
  layernorm_pos_k<T><<<rows, 128, 0, s>>>(x, g, b, eps, pos, npos, y1, y2, C);
}

// ---------------- 원소별 ----------------
template <typename T>
__global__ void cast_k(const float* x, T* y, long long n) {
  long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x;
  if (i < n) y[i] = fromf<T>(x[i]);
}
template <typename T>
void cast(const float* x, T* y, long long n, cudaStream_t s) {
  cast_k<<<nblk(n, 256), 256, 0, s>>>(x, y, n);
}

template <typename T>
__global__ void bias_act_k(const float* x, int ldx, const float* b, T* y, int ldy, int rows, int cols, int act) {
  long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x;
  if (i >= (long long)rows * cols) return;
  int r = (int)(i / cols), c = (int)(i % cols);
  float v = x[(long long)r * ldx + c] + (b ? b[c] : 0.f);
  if (act == 1) v = gelu(v);
  if (act == 2) v = fmaxf(v, 0.f);
  y[(long long)r * ldy + c] = fromf<T>(v);
}
template <typename T>
void bias_act(const float* x, int ldx, const float* b, T* y, int ldy, int rows, int cols, int act, cudaStream_t s) {
  bias_act_k<<<nblk((long long)rows * cols, 256), 256, 0, s>>>(x, ldx, b, y, ldy, rows, cols, act);
}

__global__ void bias_residual_k(const float* x, int ldx, const float* b, float* r, int ldr, int rows, int cols) {
  long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x;
  if (i >= (long long)rows * cols) return;
  int row = (int)(i / cols), c = (int)(i % cols);
  // PyTorch 와 같은 순서: (x + b) 를 먼저 만들고 잔차에 더한다
  float v = x[(long long)row * ldx + c] + (b ? b[c] : 0.f);
  r[(long long)row * ldr + c] = r[(long long)row * ldr + c] + v;
}
void bias_residual(const float* x, int ldx, const float* b, float* r, int ldr, int rows, int cols, cudaStream_t s) {
  bias_residual_k<<<nblk((long long)rows * cols, 256), 256, 0, s>>>(x, ldx, b, r, ldr, rows, cols);
}

template <typename T>
__global__ void add_to_k(const float* a, const float* b, T* y, long long n) {
  long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x;
  if (i < n) y[i] = fromf<T>(a[i] + b[i]);
}
template <typename T>
void add_to(const float* a, const float* b, T* y, long long n, cudaStream_t s) {
  add_to_k<<<nblk(n, 256), 256, 0, s>>>(a, b, y, n);
}

// ---------------- LayerNorm ----------------
template <typename T>
__global__ void layernorm_k(const float* x, int ldx, const float* g, const float* b, float eps, T* y, int ldy, int C) {
  // 블록(128) 하나가 한 행. 두 번 읽기(평균 → 분산)
  const float* xr = x + (long long)blockIdx.x * ldx;
  __shared__ float red[32];
  float v[8];
  int n = 0;
  float s = 0.f;
  for (int c = threadIdx.x; c < C; c += blockDim.x) {
    v[n] = xr[c];
    s += v[n++];
  }
  auto block_sum = [&](float a) {
    for (int o = 16; o > 0; o >>= 1) a += __shfl_xor_sync(0xffffffff, a, o);
    if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = a;
    __syncthreads();
    float t = 0.f;
    for (int i = 0; i < (int)(blockDim.x >> 5); ++i) t += red[i];
    __syncthreads();
    return t;
  };
  float mean = block_sum(s) / C;
  float q = 0.f;
  for (int i = 0; i < n; ++i) {
    float d = v[i] - mean;
    q += d * d;
  }
  float rstd = rsqrtf(block_sum(q) / C + eps);
  T* yr = y + (long long)blockIdx.x * ldy;
  n = 0;
  for (int c = threadIdx.x; c < C; c += blockDim.x) yr[c] = fromf<T>((v[n++] - mean) * rstd * g[c] + b[c]);
}
template <typename T>
void layernorm(const float* x, int ldx, const float* g, const float* b, float eps, T* y, int ldy, int rows, int C,
               cudaStream_t s) {
  layernorm_k<T><<<rows, 128, 0, s>>>(x, ldx, g, b, eps, y, ldy, C);
}

// ---------------- GroupNorm(NHWC) + ReLU ----------------
__global__ void groupnorm_relu_k(float* x, int HW, int C, int G, const float* g, const float* b, float eps) {
  int bi = blockIdx.x / G, gi = blockIdx.x % G, cg = C / G;
  float* xb = x + (long long)bi * HW * C + gi * cg;
  long long n = (long long)HW * cg;
  __shared__ double red[32];
  auto block_sum = [&](double a) {
    for (int o = 16; o > 0; o >>= 1) a += __shfl_xor_sync(0xffffffff, a, o);
    if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = a;
    __syncthreads();
    double t = 0.;
    for (int i = 0; i < (int)(blockDim.x >> 5); ++i) t += red[i];
    __syncthreads();
    return t;
  };
  double s = 0.;
  for (long long i = threadIdx.x; i < n; i += blockDim.x) s += xb[(i / cg) * C + i % cg];
  float mean = (float)(block_sum(s) / n);
  double q = 0.;
  for (long long i = threadIdx.x; i < n; i += blockDim.x) {
    float d = xb[(i / cg) * C + i % cg] - mean;
    q += (double)d * d;
  }
  float rstd = rsqrtf((float)(block_sum(q) / n) + eps);
  for (long long i = threadIdx.x; i < n; i += blockDim.x) {
    int c = (int)(i % cg);
    float* p = xb + (i / cg) * C + c;
    *p = fmaxf((*p - mean) * rstd * g[gi * cg + c] + b[gi * cg + c], 0.f);
  }
}
void groupnorm_relu(float* x, int B, int HW, int C, int G, const float* g, const float* b, float eps, cudaStream_t s) {
  groupnorm_relu_k<<<B * G, 1024, 0, s>>>(x, HW, C, G, g, b, eps);
}

// ---------------- ViT ----------------
template <typename T>
__global__ void patchify_k(const float* img, T* out, int H, int W, int P) {
  int wp = W / P, hp = H / P, K = 3 * P * P;
  long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x;
  if (i >= (long long)hp * wp * K) return;
  int col = (int)(i % K), t = (int)(i / K);
  int py = t / wp, px = t % wp;
  int c = col / (P * P), ky = (col / P) % P, kx = col % P;
  out[i] = fromf<T>(img[((long long)c * H + py * P + ky) * W + px * P + kx]);
}
template <typename T>
void patchify(const float* img, T* out, int H, int W, int P, cudaStream_t s) {
  long long n = (long long)(H / P) * (W / P) * 3 * P * P;
  patchify_k<<<nblk(n, 256), 256, 0, s>>>(img, out, H, W, P);
}

__global__ void add_tiled_pos_k(float* x, const float* pos, int h, int w, int ph, int pw, int C) {
  long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x;
  if (i >= (long long)h * w * C) return;
  int c = (int)(i % C), t = (int)(i / C), y = t / w, xx = t % w;
  x[i] += pos[((long long)(y % ph) * pw + (xx % pw)) * C + c];
}
void add_tiled_pos(float* x, const float* pos, int h, int w, int ph, int pw, int C, cudaStream_t s) {
  add_tiled_pos_k<<<nblk((long long)h * w * C, 256), 256, 0, s>>>(x, pos, h, w, ph, pw, C);
}

template <typename T>
__global__ void window_gather_k(const T* x, T* y, int h, int w, int C, int ws) {
  long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x;
  if (i >= (long long)h * w * C) return;
  int c = (int)(i % C), t = (int)(i / C);
  int nwx = w / ws, win = t / (ws * ws), r = t % (ws * ws);
  int yy = (win / nwx) * ws + r / ws, xx = (win % nwx) * ws + r % ws;
  y[i] = x[((long long)yy * w + xx) * C + c];
}
template <typename T>
void window_gather(const T* x, T* y, int h, int w, int C, int ws, cudaStream_t s) {
  window_gather_k<<<nblk((long long)h * w * C, 256), 256, 0, s>>>(x, y, h, w, C, ws);
}
__global__ void window_scatter_residual_k(const float* y, const float* b, float* r, int h, int w, int C, int ws) {
  long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x;
  if (i >= (long long)h * w * C) return;
  int c = (int)(i % C), t = (int)(i / C);
  int nwx = w / ws, win = t / (ws * ws), rr = t % (ws * ws);
  int yy = (win / nwx) * ws + rr / ws, xx = (win % nwx) * ws + rr % ws;
  float* p = r + ((long long)yy * w + xx) * C + c;
  *p = *p + (y[i] + b[c]);
}
void window_scatter_residual(const float* y, const float* b, float* r, int h, int w, int C, int ws, cudaStream_t s) {
  window_scatter_residual_k<<<nblk((long long)h * w * C, 256), 256, 0, s>>>(y, b, r, h, w, C, ws);
}

template <typename T>
__global__ void qkv_rope_k(const float* qkv, const float* bias, const float* pos, int N, int C, int heads,
                           float theta, T* q, T* kk, T* v) {
  // 스레드 하나 = (토큰, 채널 쌍)
  long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x;
  int half = C / 2;
  if (i >= (long long)N * half) return;
  int n = (int)(i / half), pr = (int)(i % half);
  int c0 = 2 * pr, d = C / heads, p = (c0 % d) / 2;  // 헤드 안 쌍 번호 0..d/2-1
  int nf = d / 4;                                     // 축마다 주파수 수
  int fi = p < nf ? p : p - nf;
  float coord = p < nf ? pos[2 * n] : pos[2 * n + 1];
  float inv = 1.f / powf(theta, (float)(2 * fi) / (float)(d / 2));
  float ang = coord * inv, cs = cosf(ang), sn = sinf(ang);
  const float* row = qkv + (long long)n * 3 * C;
  float q0 = row[c0] + bias[c0], q1 = row[c0 + 1] + bias[c0 + 1];
  float k0 = row[C + c0] + bias[C + c0], k1 = row[C + c0 + 1] + bias[C + c0 + 1];
  q[(long long)n * C + c0] = fromf<T>(q0 * cs + (-q1) * sn);
  q[(long long)n * C + c0 + 1] = fromf<T>(q1 * cs + q0 * sn);
  kk[(long long)n * C + c0] = fromf<T>(k0 * cs + (-k1) * sn);
  kk[(long long)n * C + c0 + 1] = fromf<T>(k1 * cs + k0 * sn);
  v[(long long)n * C + c0] = fromf<T>(row[2 * C + c0] + bias[2 * C + c0]);
  v[(long long)n * C + c0 + 1] = fromf<T>(row[2 * C + c0 + 1] + bias[2 * C + c0 + 1]);
}
template <typename T>
void qkv_rope(const float* qkv, const float* bias, const float* pos, int N, int C, int heads, float theta, T* q, T* kk,
              T* v, cudaStream_t s) {
  qkv_rope_k<<<nblk((long long)N * C / 2, 256), 256, 0, s>>>(qkv, bias, pos, N, C, heads, theta, q, kk, v);
}

// ---------------- 어텐션(일반판: 스레드 하나가 질의 하나, 온라인 softmax) ----------------
template <typename T, int D>
__global__ void attention_k(Attn<T> a) {
  constexpr int TK = 32;
  __shared__ float ks[TK][D], vs[TK][D];
  __shared__ float bys[TK];  // 쓰지 않음(자리 맞춤)
  (void)bys;
  int bh = blockIdx.y, b = bh / a.H, h = bh % a.H;
  int qi = blockIdx.x * blockDim.x + threadIdx.x;
  bool active = qi < a.Lq;
  float q[D], acc[D];
  if (active) {
    const T* qp = a.q + b * a.qb + (long long)qi * a.qr + h * D;
#pragma unroll
    for (int d = 0; d < D; ++d) {
      q[d] = tof(qp[d]);
      acc[d] = 0.f;
    }
  }
  float m = -INFINITY, l = 0.f;
  const float* byq = nullptr;
  const float* bxq = nullptr;
  if (a.by && active) {
    byq = a.by + (((long long)b * a.Lq + qi) * a.bh) * a.H;
    bxq = a.bx + (((long long)b * a.Lq + qi) * a.bw) * a.H;
  }
  int kend = a.Lk;
  if (a.causal) kend = min(a.Lk, (int)(blockIdx.x * blockDim.x + blockDim.x));
  for (int k0 = 0; k0 < kend; k0 += TK) {
    __syncthreads();
    for (int e = threadIdx.x; e < TK * D; e += blockDim.x) {
      int j = e / D, d = e % D, kj = k0 + j;
      float kv = 0.f, vv = 0.f;
      if (kj < a.Lk) {
        kv = tof(a.k[b * a.kb + (long long)kj * a.kr + h * D + d]);
        vv = tof(a.v[b * a.vb + (long long)kj * a.vr + h * D + d]);
      }
      ks[j][d] = kv;
      vs[j][d] = vv;
    }
    __syncthreads();
    if (!active) continue;
    for (int j = 0; j < TK; ++j) {
      int kj = k0 + j;
      if (kj >= a.Lk) break;
      if (a.causal && kj > qi) break;
      if (a.kmask && !a.kmask[b * a.kmb + kj]) continue;
      float sc = 0.f;
#pragma unroll
      for (int d = 0; d < D; ++d) sc += q[d] * ks[j][d];
      sc *= a.scale;
      if (byq) sc += byq[(kj / a.bw) * a.H + h] + bxq[(kj % a.bw) * a.H + h];
      float mn = fmaxf(m, sc), corr = __expf(m - mn), p = __expf(sc - mn);
      l = l * corr + p;
#pragma unroll
      for (int d = 0; d < D; ++d) acc[d] = acc[d] * corr + p * vs[j][d];
      m = mn;
    }
  }
  if (active) {
    T* op = a.o + b * a.ob + (long long)qi * a.orr + h * D;
    float il = 1.f / l;
#pragma unroll
    for (int d = 0; d < D; ++d) op[d] = fromf<T>(acc[d] * il);
  }
}
template <typename T>
void flash_attention(const Attn<T>& a, cudaStream_t s);
template <typename T>
void attention(const Attn<T>& a, cudaStream_t s) {
  if constexpr (!std::is_same_v<T, float>) {
    if (a.D == 64 || a.D == 32) return flash_attention<T>(a, s);
  }
  dim3 g(nblk(a.Lq, 64), a.B * a.H);
  if (a.D == 64)
    attention_k<T, 64><<<g, 64, 0, s>>>(a);
  else if (a.D == 32)
    attention_k<T, 32><<<g, 64, 0, s>>>(a);
  else {
    std::fprintf(stderr, "어텐션 헤드 크기 %d 미지원\n", a.D);
    std::abort();
  }
}

// ---------------- 합성곱 ----------------
template <typename T>
__global__ void convt2_shuffle_k(const float* Y, const float* b, T* out, int h, int w, int Cout, int act) {
  long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x;
  long long n = (long long)4 * h * w * Cout;
  if (i >= n) return;
  int co = (int)(i % Cout), t = (int)(i / Cout);
  int W2 = 2 * w, oy = t / W2, ox = t % W2;
  int y = oy / 2, dy = oy % 2, x = ox / 2, dx = ox % 2;
  float v = Y[((long long)y * w + x) * Cout * 4 + co * 4 + dy * 2 + dx] + b[co];
  if (act == 1) v = gelu(v);
  out[i] = fromf<T>(v);
}
template <typename T>
void convt2_shuffle(const float* Y, const float* b, T* out, int h, int w, int Cout, int act, cudaStream_t s) {
  convt2_shuffle_k<<<nblk((long long)4 * h * w * Cout, 256), 256, 0, s>>>(Y, b, out, h, w, Cout, act);
}

template <typename T>
__global__ void pad1_k(const T* x, T* p, int h, int w, int C) {
  long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x;
  long long n = (long long)(h + 2) * (w + 2) * C;
  if (i >= n) return;
  int c = (int)(i % C), t = (int)(i / C), y = t / (w + 2) - 1, xx = t % (w + 2) - 1;
  p[i] = (y < 0 || y >= h || xx < 0 || xx >= w) ? fromf<T>(0.f) : x[((long long)y * w + xx) * C + c];
}
template <typename T>
void pad1(const T* x, T* p, int h, int w, int C, cudaStream_t s) {
  pad1_k<<<nblk((long long)(h + 2) * (w + 2) * C, 256), 256, 0, s>>>(x, p, h, w, C);
}

__global__ void crop_bias_k(const float* Y, const float* b, float* out, int h, int w, int C) {
  long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x;
  if (i >= (long long)h * w * C) return;
  int c = (int)(i % C), t = (int)(i / C), y = t / w, x = t % w;
  out[i] = Y[((long long)y * (w + 2) + x) * C + c] + b[c];
}
void crop_bias(const float* Y, const float* b, float* out, int h, int w, int C, cudaStream_t s) {
  crop_bias_k<<<nblk((long long)h * w * C, 256), 256, 0, s>>>(Y, b, out, h, w, C);
}

template <typename T>
__global__ void upsample2_add_k(const float* x, const float* skip, T* out, int h, int w, int C, long long n) {
  long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x;
  if (i >= n) return;
  long long per = (long long)4 * h * w * C;
  int bi = (int)(i / per);
  long long r = i % per;
  int c = (int)(r % C), t = (int)(r / C), oy = t / (2 * w), ox = t % (2 * w);
  out[i] = fromf<T>(x[(long long)bi * h * w * C + ((long long)(oy / 2) * w + ox / 2) * C + c] + skip[r]);
}
template <typename T>
void upsample2_add(const float* x, const float* skip, T* out, int B, int h, int w, int C, cudaStream_t s) {
  long long n = (long long)B * 4 * h * w * C;
  upsample2_add_k<<<nblk(n, 256), 256, 0, s>>>(x, skip, out, h, w, C, n);
}

// ---------------- 명시적 인스턴스 ----------------
#define INST(T)                                                                                                     \
  template void linear<T>(cublasHandle_t, const T*, int, const T*, int, float*, int, int, int, int, float, float); \
  template void linear_batched<T>(cublasHandle_t, const T*, int, long long, const T*, int, long long, float*, int,  \
                                  long long, int, int, int, int, float);                                            \
  template void cast<T>(const float*, T*, long long, cudaStream_t);                                                 \
  template void bias_act<T>(const float*, int, const float*, T*, int, int, int, int, cudaStream_t);                 \
  template void add_to<T>(const float*, const float*, T*, long long, cudaStream_t);                                 \
  template void layernorm<T>(const float*, int, const float*, const float*, float, T*, int, int, int, cudaStream_t); \
  template void patchify<T>(const float*, T*, int, int, int, cudaStream_t);                                         \
  template void window_gather<T>(const T*, T*, int, int, int, int, cudaStream_t);                                   \
  template void qkv_rope<T>(const float*, const float*, const float*, int, int, int, float, T*, T*, T*,             \
                            cudaStream_t);                                                                          \
  template void attention<T>(const Attn<T>&, cudaStream_t);                                                         \
  template void convt2_shuffle<T>(const float*, const float*, T*, int, int, int, int, cudaStream_t);                \
  template void pad1<T>(const T*, T*, int, int, int, cudaStream_t);                                                 \
  template void upsample2_add<T>(const float*, const float*, T*, int, int, int, int, cudaStream_t);             \
  template void gelu_inplace<T>(T*, long long, cudaStream_t);                                                       \
  template void layernorm_pos<T>(const float*, const float*, const float*, float, const float*, int, T*, T*, int, int, \
                                 cudaStream_t);
INST(float)
INST(bf16)
INST(f16)

}  // namespace k
