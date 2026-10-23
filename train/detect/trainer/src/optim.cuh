// 옵티마이저: Ultralytics 8.4 trainer(optimizer=auto → 반복 > 1 만이면 MuSGD) 와 같은 규칙.
//  - 무리: 합성곱 가중치(Muon + SGD, 감쇠 있음) / BN γ(SGD, 감쇠 없음) / 편향·BN β(SGD, 감쇠 없음).
//    Detect 의 분류 가지(cv3, one2one_cv3) 파라미터는 lr ×3.
//  - MuSGD: Muon 몫 = lr·0.2·NS5(nesterov 모멘텀(β = momentum)), SGD 몫 = lr·1.0·nesterov SGD(감쇠 포함).
//    NS5 = Newton–Schulz 5 회(bf16, 계수 3.4445, −4.7750, 2.0315), 가중치를 [K, C·R·S] 행렬로.
//  - 워밍업(ni ≤ nw): lr 0 → lr0·lf(epoch), momentum 0.8 → 0.937. lf = 선형 (1 − e/E)(1 − lrf) + lrf.
//  - 기울기: loss scale 로 나누고 전체 L2 노름 10 으로 자름. inf/nan 이면 그 스텝을 건너뜀(GradScaler).
//    loss scale: 65536 에서 시작, 2000 스텝 연속 정상이면 ×2, inf 면 ×0.5.
//  - EMA: d = 0.9999·(1 − exp(−updates/2000)), 파라미터와 BN running 통계 모두(건너뛴 스텝에도 갱신).
// 모두 장치에서 돈다(inf 판정도 장치 플래그) — 호스트 동기 없음.
#pragma once
#include <cuda_bf16.h>

#include "net.h"

struct OptCfg {
  float lr0 = 0.01f, lrf = 0.01f, momentum = 0.9f, warm_momentum_end = 0.937f, weight_decay = 5e-4f;
  float warmup_epochs = 3.f, warmup_momentum = 0.8f;
  float muon = 0.2f, sgd = 1.0f, head_lr_mult = 3.f;
  int epochs = 100, batch = 16, nbs = 64;
  int64_t nb = 1;  // epoch 당 마이크로 배치 수
  float clip = 10.f, ema_decay = 0.9999f, ema_tau = 2000.f;
  float scale_init = 65536.f;
  int growth_interval = 2000;
  bool loss_scaling = true;
};

struct MatInfo {  // Muon 행렬 하나
  int64_t off;     // vals 안 위치
  int tr;          // 1 이면 [K, D] (K > D) 를 전치해서 [D, K] 로 직교화
  float lr_mult;   // 머리 분류 가지 ×3
};

struct OptDev {  // 장치 상태
  float scale;
  int good;
  int found_inf;
  int64_t updates;
  int64_t skipped;
  float grad_norm;
};

class Optimizer {
 public:
  void init(const OptCfg& c, Net* net, cublasHandle_t blas, cudaStream_t s);
  // 지금 반복(마이크로 배치 번호 ni, epoch)에 맞는 누적 수. 이 값만큼 마이크로 배치를 모은 뒤 step.
  int accumulate(int64_t ni) const;
  void step(int64_t ni, int epoch);  // 기울기(grads, scale 곱해진 값) → 파라미터 갱신, 기울기 0 으로
  float* ema() const { return ema_; }
  const float* scale() const { return &dev_->scale; }
  OptDev read_state() const;
  // 이어 하기: 모멘텀 두 벌·EMA·장치 상태
  void save(std::FILE* f) const;
  void load(std::FILE* f);
  OptCfg cfg;
  float lr_now[3] = {0, 0, 0}, mom_now = 0;  // 마지막 step 의 값(기록용): 합성곱·BN γ·편향

 private:
  Net* net_ = nullptr;
  cublasHandle_t blas_ = nullptr;
  cudaStream_t s_ = nullptr;
  OptDev* dev_ = nullptr;
  float* mom_ = nullptr;   // Muon 모멘텀 [total]
  float* buf_ = nullptr;   // SGD 모멘텀 [total]
  float* ema_ = nullptr;   // [total]
  float* part_ = nullptr;  // 노름 부분합
  uint8_t* kind_ = nullptr;  // 원소별 무리
  float wd_ = 0.f;
  int64_t nw_ = 0;
  // 파라미터별 무리: 0 합성곱(Muon) 1 BN γ 2 편향·BN β, 머리 분류 가지 여부
  std::vector<int> group_;
  std::vector<char> boost_;
  // NS5: 같은 모양(행 ≤ 열 로 돌린 뒤) 행렬끼리 묶어 한 번에
  struct NsGroup {
    int rows = 0, cols = 0, n = 0;
    MatInfo* mats = nullptr;
    __nv_bfloat16 *X = nullptr, *A = nullptr, *B = nullptr, *T = nullptr;
  };
  std::vector<NsGroup> ns_;
  void newton_schulz(NsGroup& g);
};
