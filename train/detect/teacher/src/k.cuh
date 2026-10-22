// SAM 3 커널. 텐서는 모두 토큰(화소)이 행, 채널이 열인 행 우선 [N, C] (영상은 NHWC).
// T = GEMM·어텐션 입력 자료형(float: 정합성 확인, half·bf16: 텐서코어 빠른 경로). 잔차 흐름은 늘 float.
#pragma once
#include <cublasLt.h>
#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>

#define CK(x)                                                                            \
  do {                                                                                   \
    cudaError_t e_ = (x);                                                                \
    if (e_ != cudaSuccess) {                                                             \
      std::fprintf(stderr, "CUDA %s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e_)); \
      std::abort();                                                                      \
    }                                                                                    \
  } while (0)
#define CB(x)                                                                    \
  do {                                                                           \
    cublasStatus_t st_cb_ = (x);                                                     \
    if (st_cb_ != CUBLAS_STATUS_SUCCESS) {                                           \
      std::fprintf(stderr, "cuBLAS %s:%d 상태 %d\n", __FILE__, __LINE__, (int)st_cb_); \
      std::abort();                                                              \
    }                                                                            \
  } while (0)

using bf16 = __nv_bfloat16;
using f16 = __half;

namespace k {

// ---- GEMM (cuBLAS) ----
// Y[M,N] (float, ldy) = alpha * X[M,K] (T, ldx) · W[N,K]^T (T, ldw) + beta * Y
template <typename T>
void linear(cublasHandle_t h, const T* X, int ldx, const T* W, int ldw, float* Y, int ldy, int M, int N, int K,
            float alpha = 1.f, float beta = 0.f);
// 배치 GEMM: 배치마다 X, Y 를 stride 만큼 옮긴다(W 공유이면 sw = 0)
template <typename T>
void linear_batched(cublasHandle_t h, const T* X, int ldx, long long sx, const T* W, int ldw, long long sw, float* Y,
                    int ldy, long long sy, int M, int N, int K, int batch, float beta = 0.f);

// cuBLASLt 빠른 GEMM(half·bf16): D[M,N] (To, ldd) = epi(X W^T + bias) + beta·D. bias 는 To 자료형.
// epi 0 = 편향, 2 = 편향 + ReLU. beta = 1 이면 D 에 더한다(잔차).
struct Lt {
  cublasLtHandle_t h = nullptr;
  void* ws = nullptr;
  size_t wsz = 32u << 20;
  cudaStream_t s = nullptr;
  void init(cudaStream_t st);
  void destroy();
};
template <typename T, typename To>
void lt_linear(Lt& lt, const T* X, int ldx, const T* W, int ldw, To* Dm, int ldd, int M, int N, int K, const To* bias,
               int epi, float beta);

// ---- 원소별 ----
template <typename T>
void gelu_inplace(T* y, long long n, cudaStream_t s);
// 창 순서 [창, ws*ws, C] → 원래 [h, w, C] (float)
void window_scatter(const float* y, float* x, int h, int w, int C, int ws, cudaStream_t s);
// y1 = LN(x), y2 = LN(x) + pos[r % npos]
template <typename T>
void layernorm_pos(const float* x, const float* g, const float* b, float eps, const float* pos, int npos, T* y1, T* y2,
                   int rows, int C, cudaStream_t s);
template <typename T>
void cast(const float* x, T* y, long long n, cudaStream_t s);
// y[r, c] (T) = act(x[r, c] + b[c]); act 0 없음 1 GELU(erf) 2 ReLU
template <typename T>
void bias_act(const float* x, int ldx, const float* b, T* y, int ldy, int rows, int cols, int act, cudaStream_t s);
// r[i, c] += x[i, c] + b[c]  (잔차 더하기)
void bias_residual(const float* x, int ldx, const float* b, float* r, int ldr, int rows, int cols, cudaStream_t s);
// y = a + b (T 로)
template <typename T>
void add_to(const float* a, const float* b, T* y, long long n, cudaStream_t s);

// ---- 정규화 ----
// y[r] = LN(x[r]) * g + b. C ≤ 1024 의 4 의 배수
template <typename T>
void layernorm(const float* x, int ldx, const float* g, const float* b, float eps, T* y, int ldy, int rows, int C,
               cudaStream_t s);
// NHWC GroupNorm(G 묶음) + ReLU, 제자리. 배치 B 장
void groupnorm_relu(float* x, int B, int HW, int C, int G, const float* g, const float* b, float eps, cudaStream_t s);

// ---- ViT ----
// 1008×1008×3 float CHW → [5184, 588] 패치 행렬 (c, ky, kx 순)
template <typename T>
void patchify(const float* img, T* out, int H, int W, int P, cudaStream_t s);
// x[y*w+x, c] += pos[(y%ph)*pw + (x%pw), c]
void add_tiled_pos(float* x, const float* pos, int h, int w, int ph, int pw, int C, cudaStream_t s);
// 창 나누기: [h, w, C] → [창, ws*ws, C] (나누어떨어지는 경우만)
template <typename T>
void window_gather(const T* x, T* y, int h, int w, int C, int ws, cudaStream_t s);
// 창 되돌리기 + 잔차: r[원래 자리] += y[창 순서] + b
void window_scatter_residual(const float* y, const float* b, float* r, int h, int w, int C, int ws, cudaStream_t s);
// qkv[N, 3C] (float, 편향 전) → q, k, v (T, [N, C]); q·k 에 2D RoPE (헤드 크기 64). pos: [N, 2] (열 0 = x, 열 1 = y)
template <typename T>
void qkv_rope(const float* qkv, const float* bias, const float* pos, int N, int C, int heads, float theta, T* q, T* kk,
              T* v, cudaStream_t s);

// ---- 어텐션 ----
template <typename T>
struct Attn {
  const T *q, *k, *v;
  T* o;
  int B, H, Lq, Lk, D;
  long long qb, qr, kb, kr, vb, vr, ob, orr;  // 배치·행 stride (원소)
  const int* kmask = nullptr;                 // [B, Lk] 1 = 유효 (kmb = 배치 stride)
  long long kmb = 0;
  bool causal = false;
  // 상자 상대 위치 편향(BoxRPB): bias = by[b, q, ky, h] + bx[b, q, kx, h], key = ky*bw + kx
  const float *by = nullptr, *bx = nullptr;
  int bh = 0, bw = 0;
  float scale = 1.f;
};
template <typename T>
void attention(const Attn<T>& a, cudaStream_t s);

// ---- 합성곱(NHWC) ----
// 전치 합성곱 k2 s2 의 GEMM 결과 Y[h*w, Cout*4 (co,dy,dx)] → out[2h, 2w, Cout] + b, act 0/1(GELU)
template <typename T>
void convt2_shuffle(const float* Y, const float* b, T* out, int h, int w, int Cout, int act, cudaStream_t s);
// 3×3 합성곱용 0 덧대기: x[h, w, C] → p[h+2, w+2, C]
template <typename T>
void pad1(const T* x, T* p, int h, int w, int C, cudaStream_t s);
// 덧댄 폭으로 계산한 Y[h*(w+2), C] → out[h, w, C] + b
void crop_bias(const float* Y, const float* b, float* out, int h, int w, int C, cudaStream_t s);
// out[2h, 2w, C] (T) = up_nearest(x[h, w, C]) + skip[2h, 2w, C]
template <typename T>
void upsample2_add(const float* x, const float* skip, T* out, int B, int h, int w, int C, cudaStream_t s);

}  // namespace k
