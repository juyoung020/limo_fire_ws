// 16 바이트 벡터 읽기·쓰기(act_t 8 개 = half, 4 개 = float). 채널 수가 V 의 배수인 텐서에 쓴다.
#pragma once
#include "core.h"

constexpr int V = 16 / sizeof(act_t);

__device__ __forceinline__ void vld(const act_t* p, float* f) {
  uint4 r = *reinterpret_cast<const uint4*>(p);
#if defined(YT_FP32)
  f[0] = __uint_as_float(r.x), f[1] = __uint_as_float(r.y), f[2] = __uint_as_float(r.z), f[3] = __uint_as_float(r.w);
#else
  const __half2* h = reinterpret_cast<const __half2*>(&r);
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    float2 t = __half22float2(h[i]);
    f[2 * i] = t.x;
    f[2 * i + 1] = t.y;
  }
#endif
}
__device__ __forceinline__ void vst(act_t* p, const float* f) {
  uint4 r;
#if defined(YT_FP32)
  r.x = __float_as_uint(f[0]), r.y = __float_as_uint(f[1]), r.z = __float_as_uint(f[2]), r.w = __float_as_uint(f[3]);
#else
  __half2* h = reinterpret_cast<__half2*>(&r);
#pragma unroll
  for (int i = 0; i < 4; ++i) h[i] = __floats2half2_rn(f[2 * i], f[2 * i + 1]);
#endif
  *reinterpret_cast<uint4*>(p) = r;
}
