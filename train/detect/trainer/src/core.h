// 공통: 오류 검사, 활성값 자료형, 장치 메모리.
// 활성값·기울기는 NHWC. 학습은 fp16(+ 동적 loss scale, PyTorch AMP 와 같음), 정합성 시험 빌드는 fp32(-DYT_FP32).
#pragma once
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cudnn.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#define CK(x)                                                                                   \
  do {                                                                                          \
    cudaError_t e_ = (x);                                                                       \
    if (e_ != cudaSuccess) {                                                                    \
      std::fprintf(stderr, "CUDA %s:%d %s: %s\n", __FILE__, __LINE__, #x, cudaGetErrorString(e_)); \
      std::abort();                                                                             \
    }                                                                                           \
  } while (0)
#define CDN(x)                                                                                    \
  do {                                                                                            \
    cudnnStatus_t e_ = (x);                                                                       \
    if (e_ != CUDNN_STATUS_SUCCESS) {                                                             \
      std::fprintf(stderr, "cuDNN %s:%d %s: %s\n", __FILE__, __LINE__, #x, cudnnGetErrorString(e_)); \
      std::abort();                                                                               \
    }                                                                                             \
  } while (0)
#define CBL(x)                                                                          \
  do {                                                                                  \
    cublasStatus_t e_ = (x);                                                            \
    if (e_ != CUBLAS_STATUS_SUCCESS) {                                                  \
      std::fprintf(stderr, "cuBLAS %s:%d %s: 상태 %d\n", __FILE__, __LINE__, #x, (int)e_); \
      std::abort();                                                                     \
    }                                                                                   \
  } while (0)
#define YT_ASSERT(c, ...)                                               \
  do {                                                                  \
    if (!(c)) {                                                         \
      std::fprintf(stderr, "확인 실패 %s:%d %s: ", __FILE__, __LINE__, #c); \
      std::fprintf(stderr, __VA_ARGS__);                                \
      std::fprintf(stderr, "\n");                                       \
      std::abort();                                                     \
    }                                                                   \
  } while (0)

#if defined(YT_FP32)
using act_t = float;
constexpr cudnnDataType_t kActDnn = CUDNN_DATA_FLOAT;
constexpr cudaDataType_t kActCuda = CUDA_R_32F;
#else
using act_t = __half;
constexpr cudnnDataType_t kActDnn = CUDNN_DATA_HALF;
constexpr cudaDataType_t kActCuda = CUDA_R_16F;
#endif

__host__ __device__ inline float to_f(float x) { return x; }
__host__ __device__ inline float to_f(__half x) { return __half2float(x); }
template <class T>
__host__ __device__ inline T from_f(float x);
template <>
__host__ __device__ inline float from_f<float>(float x) {
  return x;
}
template <>
__host__ __device__ inline __half from_f<__half>(float x) {
  return __float2half(x);
}

// 장치 메모리: 시작할 때 잡고 학습 중에는 잡지 않는다.
inline size_t& dev_total() {
  static size_t t = 0;
  return t;
}
template <class T>
T* dalloc(size_t n) {
  void* p = nullptr;
  if (n) CK(cudaMalloc(&p, ((n * sizeof(T)) + 255) & ~size_t(255)));
  dev_total() += n * sizeof(T);
  return static_cast<T*>(p);
}
inline int cdiv(int64_t a, int64_t b) { return int((a + b - 1) / b); }
