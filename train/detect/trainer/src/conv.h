// cuDNN 합성곱(NHWC, 묶음 합성곱 포함). 정방향·입력 기울기·가중치 기울기 알고리즘을 시작 때 한 번 재서 고른다.
// 가중치 배치는 cuDNN NHWC 필터 = [K, R, S, C/g] (PyTorch [K, C/g, R, S] 에서 바꿔 둔다).
#pragma once
#include "core.h"

struct ConvGeom {
  int N, H, W, C;  // 입력
  int K, R, S;     // 출력 채널, 커널
  int stride = 1, pad = 0, groups = 1;
  int P() const { return (H + 2 * pad - R) / stride + 1; }
  int Q() const { return (W + 2 * pad - S) / stride + 1; }
  int64_t in_elems() const { return int64_t(N) * H * W * C; }
  int64_t out_elems() const { return int64_t(N) * P() * Q() * K; }
  int64_t w_elems() const { return int64_t(K) * R * S * (C / groups); }
};

class ConvOp {
 public:
  void init(cudnnHandle_t h, const ConvGeom& g);
  // 실제 버퍼로 알고리즘을 재고, 필요한 작업 공간 크기를 돌려준다. need_dgrad=false 면 입력 기울기는 재지 않는다.
  size_t tune(act_t* x, act_t* w, act_t* y, act_t* dx, act_t* dw, act_t* dy, void* ws, size_t ws_limit,
              bool need_dgrad);
  void fwd(const act_t* x, const act_t* w, act_t* y, void* ws, size_t wsz, float beta = 0.f) const;
  void dgrad(const act_t* dy, const act_t* w, act_t* dx, void* ws, size_t wsz, float beta) const;
  void wgrad(const act_t* x, const act_t* dy, act_t* dw, void* ws, size_t wsz, float beta = 0.f) const;
  const ConvGeom& geom() const { return g_; }
  float ms[3] = {0, 0, 0};  // 잰 시간(정방향, 입력 기울기, 가중치 기울기)

 private:
  cudnnHandle_t h_ = nullptr;
  ConvGeom g_{};
  cudnnTensorDescriptor_t xd_{}, yd_{};
  cudnnFilterDescriptor_t wd_{};
  cudnnConvolutionDescriptor_t cd_{};
  cudnnConvolutionFwdAlgo_t fa_{};
  cudnnConvolutionBwdDataAlgo_t da_{};
  cudnnConvolutionBwdFilterAlgo_t wa_{};
};
