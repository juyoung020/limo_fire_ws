// 직접 짠 GEMM + 에필로그 융합 (MLP 용, fp32 빌드 전용). 설명은 network/README.md "계산 방식".
//
//   C[M×N] = op(A)[M×K] · op(B)[K×N] 뒤에 에필로그를 같은 커널에서:
//     Store    C = acc
//     Bias     C = acc + b[열]
//     BiasElu  C = ELU(acc + b[열])                     (정방향 은닉 층 — 편향·ELU 커널 따로 없음)
//     EluBack  C = acc · ELU'(aux[행][열])  (aux = 그 층 입력 활성값)  (역방향 dX — ELU 미분 커널 따로 없음)
//     Accum    C += acc                                   (역방향 dW 누적)
//   bias_grad 를 주면(ta 일 때만) 같은 커널이 A 의 K 방향 합(= 편향 기울기 Σ_샘플 dY)을 고정 순서로 더한다 — 원자 연산 없음.
//   ta: A 가 [K][M] 로 저장(M 연속), 아니면 [M][K]. tb: B 가 [N][K] 로 저장(K 연속), 아니면 [K][N].
//   lowp: bf16 텐서 코어(WMMA m16n16k16, 입력 bf16 반올림, fp32 누산). 아니면 같은 타일·같은 에필로그의 fp32 FMA(정확 모드, 기울기 검사용).
//   타일·누산 순서가 고정이라 같은 입력이면 비트 단위로 같은 결과(결정적).
#pragma once

#include <cuda_runtime.h>

namespace xnet {

enum class FEpi : int { Store = 0, Bias = 1, BiasElu = 2, EluBack = 3, Accum = 4 };

struct FGemm {
  int M = 0, N = 0, K = 0;
  const float* A = nullptr;
  int lda = 0;
  bool ta = false;
  const float* B = nullptr;
  int ldb = 0;
  bool tb = false;
  float* C = nullptr;
  int ldc = 0;
  FEpi epi = FEpi::Store;
  const float* bias = nullptr;   // Bias·BiasElu
  const float* aux = nullptr;    // EluBack: 활성값 [M][ldaux]
  int ldaux = 0;
  bool broken = false;           // EluBack: ELU' = 1 (기울기 검사 --negative)
  float* bias_grad = nullptr;    // ta 일 때: bias_grad[m] += Σ_k A(m,k)
  const unsigned short* Bh = nullptr;  // B 의 bf16 사본(같은 배치·ld). lowp 이고 !ta 면 B 를 여기서 읽는다(가중치 — 바꿀 때마다 to_bf16 로 새로)
};

void fgemm(const FGemm& g, bool lowp, cudaStream_t st);
void to_bf16(const float* x, unsigned short* y, long n, cudaStream_t st);  // 반올림은 GEMM 로더와 같음(__float2bfloat16_rn)

}  // namespace xnet
