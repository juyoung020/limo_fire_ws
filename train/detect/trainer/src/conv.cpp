#include "conv.h"

#include <cudnn_frontend.h>

#include <chrono>
#include <cstdlib>
#include <unordered_map>

namespace fe = cudnn_frontend;
using TA = std::shared_ptr<fe::graph::Tensor_attributes>;

struct FeConv {
  std::shared_ptr<fe::graph::Graph> g[3];
  TA in[3][2], out[3];  // 연산마다 입력 둘, 출력 하나
  int64_t wsz[3] = {0, 0, 0};
  int64_t plan[3] = {-1, -1, -1};  // 고른 실행 계획 번호
};

namespace {
fe::DataType_t fe_type() {
#if defined(YT_FP32)
  return fe::DataType_t::FLOAT;
#else
  return fe::DataType_t::HALF;
#endif
}
// op 0 정방향(x, w → y), 1 입력 기울기(dy, w → dx), 2 가중치 기울기(dy, x → dw). 못 만들면 false
bool fe_build(cudnnHandle_t h, const ConvGeom& c, int op, FeConv& f, int64_t ws_limit) {
  int64_t n = c.N, C = c.C, H = c.H, W = c.W, K = c.K, R = c.R, S = c.S, P = c.P(), Q = c.Q();
  auto g = std::make_shared<fe::graph::Graph>();
  g->set_io_data_type(fe_type()).set_intermediate_data_type(fe::DataType_t::FLOAT).set_compute_data_type(fe::DataType_t::FLOAT);
  auto X = g->tensor(fe::graph::Tensor_attributes().set_name("x").set_dim({n, C, H, W}).set_stride({C * H * W, 1, C * W, C}));
  auto Wt = g->tensor(fe::graph::Tensor_attributes().set_name("w").set_dim({K, C, R, S}).set_stride({C * R * S, 1, C * S, C}));
  auto Y = g->tensor(fe::graph::Tensor_attributes().set_name("y").set_dim({n, K, P, Q}).set_stride({K * P * Q, 1, K * Q, K}));
  TA O;
  std::vector<int64_t> pad = {c.pad, c.pad}, st = {c.stride, c.stride}, dil = {1, 1};
  if (op == 0) {
    O = g->conv_fprop(X, Wt, fe::graph::Conv_fprop_attributes().set_padding(pad).set_stride(st).set_dilation(dil));
    f.in[0][0] = X, f.in[0][1] = Wt;
  } else if (op == 1) {
    O = g->conv_dgrad(Y, Wt, fe::graph::Conv_dgrad_attributes().set_padding(pad).set_stride(st).set_dilation(dil));
    O->set_dim({n, C, H, W}).set_stride({C * H * W, 1, C * W, C});
    f.in[1][0] = Y, f.in[1][1] = Wt;
  } else {
    O = g->conv_wgrad(Y, X, fe::graph::Conv_wgrad_attributes().set_padding(pad).set_stride(st).set_dilation(dil));
    O->set_dim({K, C, R, S}).set_stride({C * R * S, 1, C * S, C});
    f.in[2][0] = Y, f.in[2][1] = X;
  }
  O->set_output(true);
  auto e = g->validate();
  if (e.is_good()) e = g->build_operation_graph(h);
  if (e.is_good()) e = g->create_execution_plans({fe::HeurMode_t::A, fe::HeurMode_t::FALLBACK});
  if (e.is_good()) {
    // FFT 엔진은 autotune 때 버퍼 밖에 쓰는 일이 있어 빼고, 작업 공간 상한을 넘는 엔진도 뺀다
    g->deselect_numeric_notes({fe::NumericalNote_t::FFT});
    g->deselect_workspace_greater_than(ws_limit);
    e = g->check_support();
  }
  if (e.is_good()) e = g->build_plans(fe::BuildPlanPolicy_t::ALL);
  if (!e.is_good()) return false;
  f.g[op] = g;
  f.out[op] = O;
  return true;
}
bool fe_run(cudnnHandle_t h, FeConv& f, int op, int64_t plan, const void* a, const void* b, void* o, void* ws) {
  std::unordered_map<TA, void*> vp = {{f.in[op][0], const_cast<void*>(a)}, {f.in[op][1], const_cast<void*>(b)}, {f.out[op], o}};
  return f.g[op]->execute_plan_at_index(h, vp, ws, plan).is_good();
}
void fe_exec(cudnnHandle_t h, FeConv& f, int op, const void* a, const void* b, void* o, void* ws) {
  YT_ASSERT(fe_run(h, f, op, f.plan[op], a, b, o, ws), "backend 실행 실패(연산 %d)", op);
}
// 시험 입력: [-1, 1) 무작위(해시)
__global__ void fill_k(act_t* p, int64_t n, uint32_t seed) {
  int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
  if (i >= n) return;
  uint32_t x = (uint32_t)i * 2654435761u ^ seed;
  x ^= x >> 15;
  x *= 2246822519u;
  x ^= x >> 13;
  p[i] = from_f<act_t>((x & 0xffff) / 32768.f - 1.f);
}
__global__ void maxdiff_k(const act_t* a, const act_t* b, int64_t n, float* out) {
  float d = 0.f, m = 0.f;
  for (int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x; i < n; i += (int64_t)gridDim.x * blockDim.x) {
    float x = to_f(a[i]), y = to_f(b[i]);
    d = fmaxf(d, fabsf(x - y));
    m = fmaxf(m, fabsf(x));
  }
  atomicMax(reinterpret_cast<int*>(&out[0]), __float_as_int(d));
  atomicMax(reinterpret_cast<int*>(&out[1]), __float_as_int(m));
}
}  // namespace

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
                    bool need_dgrad, bool dgrad_overwrite) {
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
  // backend 엔진 후보(묶음 합성곱 제외): 계획마다 직접 재고, legacy 결과와 같은지 확인
  if (g_.groups == 1 && !std::getenv("YT_NO_BACKEND")) {
    fe_ = std::make_shared<FeConv>();
    cudaStream_t st;
    CDN(cudnnGetStream(h_, &st));
    int64_t nx = g_.in_elems(), nw = g_.w_elems(), ny = g_.out_elems();
    fill_k<<<cdiv(nx, 256), 256, 0, st>>>(x, nx, 1);
    fill_k<<<cdiv(nw, 256), 256, 0, st>>>(w, nw, 2);
    fill_k<<<cdiv(ny, 256), 256, 0, st>>>(dy, ny, 3);
    act_t* ref = nullptr;
    float* md = nullptr;
    CK(cudaMalloc(&ref, std::max({nx, nw, ny}) * sizeof(act_t)));
    CK(cudaMalloc(&md, 8));
    cudaEvent_t e0, e1;
    CK(cudaEventCreate(&e0));
    CK(cudaEventCreate(&e1));
    for (int op = 0; op < 3; ++op) {
      if (op == 1 && !(need_dgrad && dgrad_overwrite)) continue;
      if (!fe_build(h_, g_, op, *fe_, (int64_t)ws_limit)) continue;
      const void* a = op == 0 ? (const void*)x : (const void*)dy;
      const void* b = op == 2 ? (const void*)x : (const void*)w;
      act_t* o = op == 0 ? y : op == 1 ? dx : dw;
      int64_t no = op == 0 ? ny : op == 1 ? nx : nw;
      // legacy 정답
      if (op == 0) fwd(x, w, o, ws, ws_limit);
      if (op == 1) {
        CK(cudaMemsetAsync(o, 0, no * sizeof(act_t), st));
        float al = 1.f, be = 0.f;
        CDN(cudnnConvolutionBackwardData(h_, &al, wd_, w, yd_, dy, cd_, da_, ws, ws_limit, &be, xd_, o));
      }
      if (op == 2) {
        float al = 1.f, be = 0.f;
        CDN(cudnnConvolutionBackwardFilter(h_, &al, xd_, x, yd_, dy, cd_, wa_, ws, ws_limit, &be, wd_, o));
      }
      CK(cudaMemcpyAsync(ref, o, no * sizeof(act_t), cudaMemcpyDeviceToDevice, st));
      int64_t np = fe_->g[op]->get_execution_plan_count();
      float best = ms[op] * 0.97f;
      for (int64_t pi = 0; pi < np; ++pi) {
        int64_t wsz = 0;
        if (!fe_->g[op]->get_workspace_size_plan_at_index(pi, wsz).is_good() || (size_t)wsz > ws_limit) continue;
        if (!fe_run(h_, *fe_, op, pi, a, b, o, ws) || cudaStreamSynchronize(st) != cudaSuccess) {
          cudaGetLastError();
          continue;
        }
        CK(cudaMemsetAsync(md, 0, 8, st));
        maxdiff_k<<<256, 256, 0, st>>>(ref, o, no, md);
        float hd[2];
        CK(cudaMemcpy(hd, md, 8, cudaMemcpyDeviceToHost));
#if defined(YT_FP32)
        const float tol = 1e-4f;
#else
        const float tol = 2e-2f;
#endif
        if (!(hd[0] <= tol * std::max(hd[1], 1e-6f))) continue;  // 틀린 결과(또는 NaN) 버림
        CK(cudaEventRecord(e0, st));
        for (int it = 0; it < 5; ++it) fe_run(h_, *fe_, op, pi, a, b, o, ws);
        CK(cudaEventRecord(e1, st));
        CK(cudaEventSynchronize(e1));
        float t;
        CK(cudaEventElapsedTime(&t, e0, e1));
        t /= 5;
        if (t < best) {
          best = t;
          fe_->plan[op] = pi;
          fe_->wsz[op] = wsz;
        }
      }
      if (fe_->plan[op] >= 0) {
        use_be[op] = true;
        ms[op] = best;
        need = std::max(need, (size_t)fe_->wsz[op]);
      }
    }
    cudaEventDestroy(e0);
    cudaEventDestroy(e1);
    CK(cudaFree(ref));
    CK(cudaFree(md));
  }
  return need;
}

void ConvOp::fwd(const act_t* x, const act_t* w, act_t* y, void* ws, size_t wsz, float beta) const {
  if (use_be[0] && beta == 0.f) return fe_exec(h_, *fe_, 0, x, w, y, ws);
  float a = 1.f;
  CDN(cudnnConvolutionForward(h_, &a, xd_, x, wd_, w, cd_, fa_, ws, wsz, &beta, yd_, y));
}
void ConvOp::dgrad(const act_t* dy, const act_t* w, act_t* dx, void* ws, size_t wsz, float beta) const {
  // backend 를 고른 것은 처음 쓰는 연산(버퍼가 0)뿐이라 덮어써도 더하기와 같다
  if (use_be[1]) return fe_exec(h_, *fe_, 1, dy, w, dx, ws);
  float a = 1.f;
  CDN(cudnnConvolutionBackwardData(h_, &a, wd_, w, yd_, dy, cd_, da_, ws, wsz, &beta, xd_, dx));
}
void ConvOp::wgrad(const act_t* x, const act_t* dy, act_t* dw, void* ws, size_t wsz, float beta) const {
  if (use_be[2] && beta == 0.f) return fe_exec(h_, *fe_, 2, dy, x, dw, ws);
  float a = 1.f;
  CDN(cudnnConvolutionBackwardFilter(h_, &a, xd_, x, yd_, dy, cd_, wa_, ws, wsz, &beta, wd_, dw));
}
