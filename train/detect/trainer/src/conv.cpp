#include "conv.h"

void ConvOp::init(cudnnHandle_t h, const ConvGeom& g) {
  h_ = h;
  g_ = g;
  CDN(cudnnCreateTensorDescriptor(&xd_));
  CDN(cudnnCreateTensorDescriptor(&yd_));
  CDN(cudnnCreateFilterDescriptor(&wd_));
  CDN(cudnnCreateConvolutionDescriptor(&cd_));
  CDN(cudnnSetTensor4dDescriptor(xd_, CUDNN_TENSOR_NHWC, kActDnn, g.N, g.C, g.H, g.W));
  CDN(cudnnSetTensor4dDescriptor(yd_, CUDNN_TENSOR_NHWC, kActDnn, g.N, g.K, g.P(), g.Q()));
  CDN(cudnnSetFilter4dDescriptor(wd_, kActDnn, CUDNN_TENSOR_NHWC, g.K, g.C / g.groups, g.R, g.S));
  CDN(cudnnSetConvolution2dDescriptor(cd_, g.pad, g.pad, g.stride, g.stride, 1, 1, CUDNN_CROSS_CORRELATION,
                                      CUDNN_DATA_FLOAT));
  CDN(cudnnSetConvolutionGroupCount(cd_, g.groups));
#if defined(YT_FP32)
  CDN(cudnnSetConvolutionMathType(cd_, CUDNN_FMA_MATH));  // TF32 끔(정합성)
#else
  CDN(cudnnSetConvolutionMathType(cd_, CUDNN_TENSOR_OP_MATH));
#endif
}

size_t ConvOp::tune(act_t* x, act_t* w, act_t* y, act_t* dx, act_t* dw, act_t* dy, void* ws, size_t ws_limit,
                    bool need_dgrad) {
  size_t need = 0;
  int n = 0;
  {
    cudnnConvolutionFwdAlgoPerf_t p[CUDNN_CONVOLUTION_FWD_ALGO_COUNT];
    CDN(cudnnFindConvolutionForwardAlgorithmEx(h_, xd_, x, wd_, w, cd_, yd_, y, CUDNN_CONVOLUTION_FWD_ALGO_COUNT, &n,
                                               p, ws, ws_limit));
    int i = 0;
    while (i < n && (p[i].status != CUDNN_STATUS_SUCCESS || p[i].memory > ws_limit)) ++i;
    YT_ASSERT(i < n, "정방향 알고리즘 없음");
    fa_ = p[i].algo;
    ms[0] = p[i].time;
    need = std::max(need, p[i].memory);
  }
  if (need_dgrad) {
    cudnnConvolutionBwdDataAlgoPerf_t p[CUDNN_CONVOLUTION_BWD_DATA_ALGO_COUNT];
    CDN(cudnnFindConvolutionBackwardDataAlgorithmEx(h_, wd_, w, yd_, dy, cd_, xd_, dx,
                                                    CUDNN_CONVOLUTION_BWD_DATA_ALGO_COUNT, &n, p, ws, ws_limit));
    int i = 0;
    while (i < n && (p[i].status != CUDNN_STATUS_SUCCESS || p[i].memory > ws_limit)) ++i;
    YT_ASSERT(i < n, "입력 기울기 알고리즘 없음");
    da_ = p[i].algo;
    ms[1] = p[i].time;
    need = std::max(need, p[i].memory);
  }
  {
    cudnnConvolutionBwdFilterAlgoPerf_t p[CUDNN_CONVOLUTION_BWD_FILTER_ALGO_COUNT];
    CDN(cudnnFindConvolutionBackwardFilterAlgorithmEx(h_, xd_, x, yd_, dy, cd_, wd_, dw,
                                                      CUDNN_CONVOLUTION_BWD_FILTER_ALGO_COUNT, &n, p, ws, ws_limit));
    int i = 0;
    while (i < n && (p[i].status != CUDNN_STATUS_SUCCESS || p[i].memory > ws_limit)) ++i;
    YT_ASSERT(i < n, "가중치 기울기 알고리즘 없음");
    wa_ = p[i].algo;
    ms[2] = p[i].time;
    need = std::max(need, p[i].memory);
  }
  return need;
}

void ConvOp::fwd(const act_t* x, const act_t* w, act_t* y, void* ws, size_t wsz, float beta) const {
  float a = 1.f;
  CDN(cudnnConvolutionForward(h_, &a, xd_, x, wd_, w, cd_, fa_, ws, wsz, &beta, yd_, y));
}
void ConvOp::dgrad(const act_t* dy, const act_t* w, act_t* dx, void* ws, size_t wsz, float beta) const {
  float a = 1.f;
  CDN(cudnnConvolutionBackwardData(h_, &a, wd_, w, yd_, dy, cd_, da_, ws, wsz, &beta, xd_, dx));
}
void ConvOp::wgrad(const act_t* x, const act_t* dy, act_t* dw, void* ws, size_t wsz, float beta) const {
  float a = 1.f;
  CDN(cudnnConvolutionBackwardFilter(h_, &a, xd_, x, yd_, dy, cd_, wa_, ws, wsz, &beta, wd_, dw));
}
