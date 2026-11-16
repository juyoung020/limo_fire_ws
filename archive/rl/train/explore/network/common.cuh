// 탐사 RL 신경망·PPO 공통: 오류 검사, cuBLAS 행 우선(row-major) GEMM, 결정적(원자 연산 없는) 축소 커널.
// 모든 축소는 고정 순서(블록 안 트리 → 블록 순서대로)로 해서 같은 입력이면 비트 단위로 같은 결과가 나온다.
#pragma once

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>

#define CK(x)                                                                                   \
  do {                                                                                          \
    cudaError_t e_ = (x);                                                                       \
    if (e_ != cudaSuccess) {                                                                    \
      std::fprintf(stderr, "CUDA 오류 %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e_)); \
      std::abort();                                                                             \
    }                                                                                           \
  } while (0)

#define CKB(x)                                                                         \
  do {                                                                                 \
    cublasStatus_t s_ = (x);                                                           \
    if (s_ != CUBLAS_STATUS_SUCCESS) {                                                 \
      std::fprintf(stderr, "cuBLAS 오류 %s:%d: %d\n", __FILE__, __LINE__, (int)s_);    \
      std::abort();                                                                    \
    }                                                                                  \
  } while (0)

// 실수형: 학습·추론은 fp32. XNET_FP64 로 같은 소스를 fp64 로 한 벌 더 빌드해 기울기 검사의 기준으로 쓴다(이름공간이 달라 한 실행 파일에 같이 링크된다).
#ifdef XNET_FP64
#define XNET_NS xnet64
#define XPPO_NS xppo64
#else
#define XNET_NS xnet
#define XPPO_NS xppo
#endif

namespace XNET_NS {

#ifdef XNET_FP64
using real = double;
#else
using real = float;
#endif

inline int nblk(long n, int t = 256) { return (int)((n + t - 1) / t); }

// 행 우선 C[M×N] = alpha·op(A)·op(B) + beta·C. cuBLAS 는 열 우선이라 C^T = op(B)^T·op(A)^T 로 부른다.
inline void gemm(cublasHandle_t h, bool tA, bool tB, int M, int N, int K, real alpha, const real* A, int lda,
                 const real* B, int ldb, real beta, real* C, int ldc) {
#ifdef XNET_FP64
  CKB(cublasDgemm(h, tB ? CUBLAS_OP_T : CUBLAS_OP_N, tA ? CUBLAS_OP_T : CUBLAS_OP_N, N, M, K, &alpha, B, ldb, A, lda,
                  &beta, C, ldc));
#else
  CKB(cublasSgemm(h, tB ? CUBLAS_OP_T : CUBLAS_OP_N, tA ? CUBLAS_OP_T : CUBLAS_OP_N, N, M, K, &alpha, B, ldb, A, lda,
                  &beta, C, ldc));
#endif
}

// [R][C] (행 간격 ld) 의 열 합을 out 에 더하거나(acc) 쓴다. 두 단계(행 묶음 부분합 → 블록 순서대로) — 결정적.
void col_sum(const real* x, long R, int C, int ld, real* out, bool acc, cudaStream_t s, const real* a_out = nullptr,
             bool broken = false);
// a_out 을 주면 x 를 제자리에서 ELU'(a_out) 배 한 뒤 더한다 (ld == C 일 때만).
// 전체 합 (double 로 누적) → host 로 돌려준다. 고정 순서.
double sum_all(const real* x, long n, cudaStream_t s);
// 제곱합 (double)
double sumsq_all(const real* x, long n, cudaStream_t s);

}  // namespace XNET_NS
