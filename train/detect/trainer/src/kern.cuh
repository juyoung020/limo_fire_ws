// 합성곱 밖의 커널: BatchNorm(학습 통계) + SiLU, 채널 조각 복사(concat·split), 더하기, 업샘플, maxpool, 편향.
// 모든 텐서는 NHWC 를 [M = N·H·W, C] 행렬로 본다. 역방향 커널은 입력 기울기에 더한다(+=). 기울기 버퍼는 스텝마다
// 0 으로 지운 뒤 쓴다.
#pragma once
#include "core.h"

namespace kern {

// ---- BatchNorm ----
// 학습: 배치 평균·분산(편향) → mean, istd. running 통계는 momentum 으로(분산은 불편 추정, PyTorch 와 같음).
// acc: 장치 double [2C] 작업 공간.
void bn_stats(const act_t* z, int64_t M, int C, double* acc, float* mean, float* istd, float* run_mean,
              float* run_var, float momentum, float eps, cudaStream_t s);
// 평가: running 통계로 mean, istd
void bn_eval_stats(int C, const float* run_mean, const float* run_var, float eps, float* mean, float* istd,
                   cudaStream_t s);
// y = act(γ·(z − μ)·istd + β). act: 0 없음, 1 SiLU
// y 의 행 간격 ldy(채널 C 보다 클 수 있음: concat 버퍼 안의 자리)
void bn_apply(const act_t* z, int64_t M, int C, const float* mean, const float* istd, const float* gamma,
              const float* beta, int act, act_t* y, int ldy, cudaStream_t s);
// 역방향: dy → dz(덮어쓰기), dγ·dβ(+=, float). acc: double [4C] (뒤쪽은 채널 계수 자리)
void bn_backward(const act_t* z, const act_t* dy, int lddy, int64_t M, int C, const float* mean, const float* istd,
                 const float* gamma, const float* beta, int act, double* acc, act_t* dz, float* dgamma, float* dbeta,
                 cudaStream_t s);

// ---- 모양 ----
// dst[:, d0:d0+n] (=|+=) src[:, s0:s0+n]
void copy_ch(const act_t* src, int sC, int s0, act_t* dst, int dC, int d0, int n, int64_t M, bool accum,
             cudaStream_t s);
// (행 간격 ld 를 받는 판) y = a + b, dst += src  — [M, C]
void add(const act_t* a, int lda, const act_t* b, int ldb, act_t* y, int ldy, int64_t M, int C, cudaStream_t s);
void acc(const act_t* src, int lds, act_t* dst, int ldd, int64_t M, int C, cudaStream_t s);
// 최근접 2 배: x [N,H,W,C] → y [N,2H,2W,C]; 역방향 dx += Σ 2×2
void up2(const act_t* x, int ldx, act_t* y, int ldy, int N, int H, int W, int C, cudaStream_t s);
void up2_bwd(const act_t* dy, int lddy, act_t* dx, int lddx, int N, int H, int W, int C, cudaStream_t s);
// maxpool k×k stride 1 pad k/2 (PyTorch 와 같이 -inf 덧대기). arg: 창 안 최댓값 위치(u8, [M, C] 꽉 참)
void maxpool(const act_t* x, int ldx, act_t* y, int ldy, uint8_t* arg, int N, int H, int W, int C, int k,
             cudaStream_t s);
void maxpool_bwd(const act_t* dy, int lddy, const uint8_t* arg, act_t* dx, int lddx, int N, int H, int W, int C, int k,
                 cudaStream_t s);

// ---- depthwise 3×3 (stride 1, pad 1), NHWC, 가중치 [C, 3, 3] (act_t) ----
void dw3_fwd(const act_t* x, const act_t* w, act_t* y, int N, int H, int W, int C, cudaStream_t s);
// dx (+=)
void dw3_dgrad(const act_t* dy, const act_t* w, act_t* dx, int N, int H, int W, int C, cudaStream_t s);
// dw (덮어쓰기, act_t [C·9]). tmp: float [9C]
void dw3_wgrad(const act_t* x, const act_t* dy, act_t* dw, float* tmp, int N, int H, int W, int C, cudaStream_t s);

// ---- 편향·형 ----
void add_bias(act_t* y, int64_t M, int C, const float* b, cudaStream_t s);
// db (+=) Σ_rows dy. acc: double [C]
void bias_grad(const act_t* dy, int64_t M, int C, double* acc, float* db, cudaStream_t s);
void f32_to_act(const float* x, act_t* y, int64_t n, cudaStream_t s);
void act_to_f32(const act_t* x, float* y, int64_t n, bool accum, cudaStream_t s);
void scale_act(act_t* x, float a, int64_t n, cudaStream_t s);

// 벡터판(bnv.cu) — 채널 수가 V(16 B) 의 배수일 때 위 함수들이 고른다
bool vec_ok(int C);
void bn_reduce_vec(bool bwd, const act_t* z, const act_t* dy, int lddy, int64_t M, int C, const float* mean,
                   const float* istd, const float* g, const float* b, int act, double* acc, cudaStream_t s);
void bn_apply_vec(const act_t* z, int64_t M, int C, const float* mean, const float* istd, const float* g,
                  const float* b, int act, act_t* y, int ldy, cudaStream_t s);
void bn_dz_vec(const act_t* z, const act_t* dy, int lddy, int64_t M, int C, const float* mean, const float* istd,
               const float* g, const float* b, int act, const float* coef, act_t* dz, cudaStream_t s);
void copy_ch_vec(const act_t* src, int sC, int s0, act_t* dst, int dC, int d0, int n, int64_t M, bool accum,
                 cudaStream_t s);
void add_vec(const act_t* a, int lda, const act_t* b, int ldb, act_t* y, int ldy, int64_t M, int C, cudaStream_t s);
void up2_vec(const act_t* x, int ldx, act_t* y, int ldy, int N, int H, int W, int C, cudaStream_t s);
void up2_bwd_vec(const act_t* dy, int lddy, act_t* dx, int lddx, int N, int H, int W, int C, cudaStream_t s);

}  // namespace kern
