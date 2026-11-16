// 합성곱 넷째 층(꼬리, 2×2 stride 2 + 덧댐 + 편향 + ELU)을 커널 하나로 — 정방향·역방향. fp32 학습 빌드 전용 (fp64 검사 빌드는 기준 경로).
//
// 입력 F3 = 셋째 층 출력 [n][h·w][C] (래스터, 융합 합성곱 커널이 씀). 출력 픽셀 p = (py, px) 는 F3 의 (2py+dy, 2px+dx) 네 칸에 기대고
// 격자 밖(홀수 행·열을 덧댄 자리)은 0. 행 = (샘플 i, p), k = 자식(dy·2+dx)·C + c → 덧댄 im2col 행렬을 메모리에 만들지 않고 타일을 공유 메모리로 바로 읽는다.
// 계산은 fp32 FMA (기준 경로의 cuBLAS sgemm 과 같은 정밀도, 더하는 순서만 다름).
//   정방향: Y[i·ldy + p·cout + co] = ELU(b[co] + Σ_k X(row, k)·W[co][k]) — Y 는 MLP 입력 [F | 1D] 의 이 가지 열(복사 없음)
//   역방향: dZ = (g1 + g2) · ELU'(Y) (두 MLP 의 특징 기울기를 읽으면서 고정 순서로 더함), dF3 = dZ·W (덧댄 자리는 버림),
//           dW = Σ_행 dZᵀ·X, db = Σ_행 dZ — 블록(고정 개수)마다 행 타일을 정해진 순서로 맡아 부분합 → 블록 순서대로 더함(원자 연산 없음 → 결정적)
#pragma once

#include <cuda_runtime.h>

namespace xnet {

struct TailShape {
  int n, h, w, C;   // 샘플 수, 셋째 층 격자·채널
  int oh, ow, cout; // 넷째 층 격자(덧댄 뒤 ÷2)·채널
};

// 지원: cout ∈ {4, 32}, C % 4 == 0, 4C ≤ 128 (스레드 하나가 dW 4×4 칸 하나)
bool cnn_tail_supported(const TailShape& s);
long cnn_tail_part_floats(const TailShape& s);  // 역방향 블록별 부분합 작업 공간
// W [cout][4][C] 와 b [cout] 은 이어져 있다 (체크포인트 배치). Y = 출력 첫 열(이 가지 평탄화 시작) 위치.
// Yh: 주면 같은 자리(같은 행 간격)에 bf16 사본도 쓴다 (MLP 첫 층 bf16 GEMM 입력, 반올림 = __float2bfloat16_rn)
void cnn_tail_forward(const TailShape& s, const float* F3, const float* Wb, float* Y, long ldy, cudaStream_t st,
                      unsigned short* Yh = nullptr);
// g1·g2 = 특징 기울기 (이 가지 첫 열 위치, g2 = nullptr 이면 g1 만). gWb = dW·db 자리(누적 +=). part = 작업 공간.
// z_out: dF3 대신 dZ3 = dF3 · ELU'(F3) 를 쓴다 (앞 세 층 융합 커널이 받는 꼴 — F3 를 다시 읽지 않게; 곱은 그 커널이 하던 fp32 식 그대로)
void cnn_tail_backward(const TailShape& s, const float* F3, const float* Wb, const float* Y, long ldy, const float* g1, long ld1,
                       const float* g2, long ld2, float* dF3, float* gWb, float* part, bool broken_elu, cudaStream_t st,
                       bool z_out = false);

}  // namespace xnet
