// GRU 한 층 (아래층 순환 정책 — network/README "순환 핵"). PyTorch nn.GRU 와 같은 식 (문 차례 r, z, n):
//   r = σ(Wx_r x + bx_r + Wh_r h + bh_r),  z = σ(Wx_z x + bx_z + Wh_z h + bh_z),  n = tanh(Wx_n x + bx_n + r ⊙ (Wh_n h + bh_n)),  h' = (1 − z) ⊙ n + z ⊙ h
// 입력 x = 한 행 [F (nf 열) | … | 1D (nd 열, off_d 부터)] — MLP 입력 버퍼의 행을 그대로 읽는다 (가운데 h 열은 안 읽음 — 이 층의 출력 자리).
// 변수 (블록 이름 <pfx>.wx · bx · wh · bh): Wx [3H][nf + nd] (선형 층과 같은 [out][in]), bx [3H], Wh 는 [H][3H] (k 먼저 — 문 커널이 이웃 스레드 = 이웃 출력으로 읽게), bh [3H].
// 초기화: PyTorch nn.GRU 기본 U(±1/√H).
// 판 경계: reset[행] = 1 이면 그 걸음의 이전 h 를 0 으로 (판이 그 걸음에 새로 시작). 0 걸음 행의 이전 h = h0.
// 계산: 입력 쪽은 GEMM 두 번(F 열 · 1D 열)으로 모든 행을 한 번에, 순환 쪽은 걸음마다 직접 짠 문 커널 하나(앞)·둘(뒤). 원자 연산 없음 — 같은 입력이면 비트 같음.
// 모든 작업 공간은 bind 에서 미리 잡는다 (호스트 왕복 없음).
#pragma once

#include <cstdint>
#include <random>
#include <string>
#include <vector>

#include "common.cuh"

namespace XNET_NS {

struct ParamBlock;

class Gru {
 public:
  int nf = 0, nd = 0, H = 0;
  long wx_off = 0, bx_off = 0, wh_off = 0, bh_off = 0;
  // 변수 자리를 등록 (ParamBlock 목록·커서)
  void build(int nf_, int nd_, int H_, std::vector<ParamBlock>& reg, long& cur, const std::string& pfx);
  // 버퍼 묶기·작업 공간: max_rows = 한 번에 넣는 행(걸음 × 환경) 수 상한
  void bind(real* params, real* grads, int max_rows);
  void free_ws();
  void init(std::mt19937_64& rng, std::vector<real>& host) const;
  // 한 걸음 (롤아웃·평가): 행 B 개. h [B][H] 는 들어올 때 이전 h (부르는 쪽이 판 경계에서 0 으로), 나갈 때 새 h.
  // 새 h 를 out (행 간격 ldo) 에도 쓴다 (outh: 같은 자리 bf16 사본, nullptr 이면 안 씀)
  void step(cublasHandle_t cb, cudaStream_t st, int B, const real* x, long ldx, int off_d, real* h, real* out, long ldo, uint16_t* outh);
  // 순서 정방향 (갱신): 행 n = T · B 개, 걸음 먼저 차례 (행 t·B + b). h0 [B][H], reset [n] (0/1). 새 h → out. 역방향용으로 문 값을 남긴다
  void forward_seq(cublasHandle_t cb, cudaStream_t st, int T, int B, const real* x, long ldx, int off_d, const real* h0, const real* reset, real* out,
                   long ldo, uint16_t* outh);
  // 직전 forward_seq 와 같은 입력으로. dout [n] (행 간격 lddo) = 손실의 출력 h 기울기 → 변수 기울기에 더함, dxf [n][nf] = F 열 입력 기울기 (덮어씀)
  void backward_seq(cublasHandle_t cb, cudaStream_t st, int T, int B, const real* x, long ldx, int off_d, const real* reset, const real* dout, long lddo,
                    real* dxf);
  real* dxf_buf() { return dXF_; }   // backward_seq 의 dxf 로 쓸 수 있는 작업 버퍼 [max_rows][nf]

 private:
  real *P_ = nullptr, *G_ = nullptr;
  int max_rows_ = 0;
  real *GX_ = nullptr, *HP_ = nullptr, *R_ = nullptr, *Z_ = nullptr, *N_ = nullptr, *GHN_ = nullptr;   // [rows][3H] · [rows][H] × 5
  real *DGX_ = nullptr, *DGH_ = nullptr, *DC_ = nullptr, *DIR_ = nullptr, *dXF_ = nullptr;           // [rows][3H] × 2 · [rows][H] × 2 · [rows][nf]
  real* GH_ = nullptr;   // 순환 쪽 GEMM 결과 h·Wh [rows][3H] (걸음마다 GEMM 한 번 → 문 커널이 읽음)
  bool ref_ = false;     // GRU_REF=1: 옛 문 커널(블록 안 직접 곱) — A/B 확인용
  void input_gemm(cublasHandle_t cb, cudaStream_t st, int n, const real* x, long ldx, int off_d);
};

}  // namespace XNET_NS
