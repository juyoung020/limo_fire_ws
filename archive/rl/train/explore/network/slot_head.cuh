// 계층형 정책 머리 (mdp/README "계층형"): 후보 슬롯마다 같은 MLP 로 점수 — 점수_k = w2 · ELU(W1 [z ; f_k] + b1) (상수 편향은 두지 않음 — 범주 분포는 모든 로짓에 같은 값을 더해도 같아 기울기가 늘 0).
// z = actor MLP 출력(상태 임베딩, E 칸), f_k = 후보 k 의 정규화된 특징(C 칸, 관측 1D 의 후보 칸). 슬롯 순서와 무관한 가중치라
// 후보 몇 번 칸에 오든 같은 점수(ARiADNE·Cao 2024 의 후보 노드 점수 꼴을 MLP 로). 출력 = 슬롯 로짓 [n][K] (가림은 PPO 가 분포에서).
// 모두 fp32/fp64(real) 정확 경로 — 기울기는 결정적: 샘플마다 값 → cuBLAS GEMM·col_sum(고정 순서)으로 변수 기울기.
#pragma once

#include "common.cuh"

namespace XNET_NS {

struct SlotHead {
  int E = 0, C = 0, K = 0, H = 0;     // 임베딩·특징·슬롯·은닉 수
  long w1_off = 0, b1_off = 0, w2_off = 0;   // d_params 안 자리. W1 [H][E + C] 행 우선
  // 작업 공간 (max_n 샘플)
  real *F = nullptr, *A = nullptr, *Hs = nullptr, *dA = nullptr, *dU = nullptr;
  int max_n = 0;
  void alloc(int max_n_);
  void release();
  // z [n][ldz] (앞 E 칸), feat [n][ldf] 의 열 foff 부터 K·C 칸 (슬롯 k = foff + k·C …) → logits [n][K]
  void forward(cudaStream_t st, int n, const real* z, int ldz, const real* feat, int ldf, int foff, const real* P, real* logits);
  // 직전 forward 와 같은 n. G [n][K] = dL/d로짓 → dz [n][ldz] 에 쓰고(앞 E 칸), 변수 기울기를 grads 에 더한다
  void backward(cublasHandle_t h, cudaStream_t st, int n, const real* z, int ldz, const real* P, real* grads, const real* G_, real* dz, int lddz);
};

}  // namespace XNET_NS
