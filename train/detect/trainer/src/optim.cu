#include <cuda_bf16.h>

#include <algorithm>
#include <cmath>
#include <map>

#include "optim.cuh"

namespace {
// 원소별 종류: 하위 2 비트 무리(0 합성곱, 1 BN γ, 2 편향, 3 학습 안 함), 비트 2 = 머리 분류 가지
__global__ void unscale_k(float* g, const uint8_t* kind, int64_t n, OptDev* st, float* part) {
  float inv = 1.f / st->scale;
  float s = 0.f;
  bool bad = false;
  for (int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x; i < n; i += (int64_t)gridDim.x * blockDim.x) {
    if ((kind[i] & 3) == 3) continue;
    float v = g[i] * inv;
    g[i] = v;
    if (!isfinite(v)) bad = true;
    s += v * v;
  }
  __shared__ float sh[256];
  sh[threadIdx.x] = s;
  __syncthreads();
  for (int o = 128; o; o >>= 1) {
    if (threadIdx.x < o) sh[threadIdx.x] += sh[threadIdx.x + o];
    __syncthreads();
  }
  if (threadIdx.x == 0) part[blockIdx.x] = sh[0];
  if (bad) st->found_inf = 1;
}
__global__ void norm_k(const float* part, int n, OptDev* st) {
  __shared__ double sh[256];
  double s = 0;
  for (int i = threadIdx.x; i < n; i += blockDim.x) s += part[i];
  sh[threadIdx.x] = s;
  __syncthreads();
  for (int o = 128; o; o >>= 1) {
    if (threadIdx.x < o) sh[threadIdx.x] += sh[threadIdx.x + o];
    __syncthreads();
  }
  if (threadIdx.x == 0) st->grad_norm = (float)sqrt(sh[0]);
}
__device__ inline float clip_coef(const OptDev* st, float maxn) {
  return fminf(maxn / (st->grad_norm + 1e-6f), 1.f);
}
// Muon 모멘텀(nesterov) → X(bf16, 행렬 방향 맞춤)
__global__ void muon_in_k(const float* g, float* m, const MatInfo* mats, int nm, int rows, int cols, float beta,
                          OptDev* st, float maxn, __nv_bfloat16* X) {
  if (st->found_inf) return;
  int64_t per = int64_t(rows) * cols;
  int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
  if (i >= per * nm) return;
  int k = int(i / per);
  int64_t e = i % per;  // 파라미터 안 평면 위치(PyTorch [K, C·R·S])
  MatInfo mi = mats[k];
  int64_t pi = mi.off + e;
  float gv = g[pi] * clip_coef(st, maxn);
  float mv = beta * m[pi] + (1.f - beta) * gv;
  m[pi] = mv;
  float u = beta * mv + (1.f - beta) * gv;
  // 파라미터 행렬 [R0, C0] (R0 = K). 넣는 행렬은 tr 이면 [C0, R0]
  // 파라미터 [K, D] 의 (k, d) 는 tr 이면 X [D, K] 의 (d, k)
  int64_t xi = mi.tr ? int64_t(k) * per + (e % rows) * cols + e / rows : int64_t(k) * per + e;
  X[xi] = __float2bfloat16(u);
}
// 행렬마다 X /= (‖X‖_F + 1e-7)
__global__ void ns_norm_k(__nv_bfloat16* X, int64_t per, OptDev* st) {
  if (st->found_inf) return;
  __nv_bfloat16* x = X + int64_t(blockIdx.x) * per;
  float s = 0.f;
  for (int64_t i = threadIdx.x; i < per; i += blockDim.x) {
    float v = __bfloat162float(x[i]);
    s += v * v;
  }
  __shared__ float sh[256];
  sh[threadIdx.x] = s;
  __syncthreads();
  for (int o = 128; o; o >>= 1) {
    if (threadIdx.x < o) sh[threadIdx.x] += sh[threadIdx.x + o];
    __syncthreads();
  }
  float inv = 1.f / (__bfloat162float(__float2bfloat16(sqrtf(sh[0]))) + 1e-7f);
  for (int64_t i = threadIdx.x; i < per; i += blockDim.x) x[i] = __float2bfloat16(__bfloat162float(x[i]) * inv);
}
__global__ void muon_out_k(float* p, const MatInfo* mats, int nm, int rows, int cols, const __nv_bfloat16* X,
                           float lr_muon, OptDev* st) {
  if (st->found_inf) return;
  int64_t per = int64_t(rows) * cols;
  int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
  if (i >= per * nm) return;
  int k = int(i / per);
  int64_t e = i % per;
  MatInfo mi = mats[k];
  int64_t xi;
  if (mi.tr) {
    int64_t kk = e / rows, dd = e % rows;
    xi = int64_t(k) * per + dd * cols + kk;
  } else {
    xi = int64_t(k) * per + e;
  }
  p[mi.off + e] -= lr_muon * mi.lr_mult * __bfloat162float(X[xi]);
}
// SGD(nesterov, 감쇠) 모든 학습 파라미터
__global__ void sgd_k(float* p, const float* g, float* buf, const uint8_t* kind, int64_t n, float lr0, float lr1,
                      float lr2, float mult, float mom, float wd, OptDev* st, float maxn) {
  if (st->found_inf) return;
  float c = clip_coef(st, maxn);
  for (int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x; i < n; i += (int64_t)gridDim.x * blockDim.x) {
    int k = kind[i], grp = k & 3;
    if (grp == 3) continue;
    float lr = grp == 0 ? lr0 : (grp == 1 ? lr1 : lr2);
    if (k & 4) lr *= mult;
    float gv = g[i] * c + (grp == 0 ? wd * p[i] : 0.f);
    float b = mom * buf[i] + gv;
    buf[i] = b;
    p[i] -= lr * (gv + mom * b);
  }
}
__global__ void scaler_k(OptDev* st, int growth_interval, bool scaling) {
  if (scaling) {
    if (st->found_inf) {
      st->scale *= 0.5f;
      st->good = 0;
      st->skipped++;
    } else if (++st->good >= growth_interval) {
      st->scale *= 2.f;
      st->good = 0;
    }
  } else if (st->found_inf) {
    st->skipped++;
  }
  st->found_inf = 0;
  st->updates++;
}
__global__ void ema_k(float* ema, const float* v, int64_t n, const OptDev* st, float decay, float tau) {
  float d = decay * (1.f - expf(-(float)st->updates / tau));
  for (int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x; i < n; i += (int64_t)gridDim.x * blockDim.x)
    ema[i] = ema[i] * d + (1.f - d) * v[i];
}
// 행 우선 묶음 GEMM(bf16)
void gemm_bf16(cublasHandle_t h, bool tA, bool tB, int M, int N, int K, float alpha, const __nv_bfloat16* A, int lda,
               long long sa, const __nv_bfloat16* B, int ldb, long long sb, float beta, __nv_bfloat16* C, int ldc,
               long long sc, int batch) {
  CBL(cublasGemmStridedBatchedEx(h, tB ? CUBLAS_OP_T : CUBLAS_OP_N, tA ? CUBLAS_OP_T : CUBLAS_OP_N, N, M, K, &alpha,
                                 B, CUDA_R_16BF, ldb, sb, A, CUDA_R_16BF, lda, sa, &beta, C, CUDA_R_16BF, ldc, sc,
                                 batch, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
}
}  // namespace

void Optimizer::init(const OptCfg& c, Net* net, cublasHandle_t blas, cudaStream_t s) {
  cfg = c;
  net_ = net;
  blas_ = blas;
  s_ = s;
  auto& P = net->params;
  int accumulate = std::max((int)std::lround((double)cfg.nbs / cfg.batch), 1);
  wd_ = cfg.weight_decay * cfg.batch * accumulate / cfg.nbs;
  nw_ = cfg.warmup_epochs > 0 ? (int64_t)std::lround(std::min<double>(cfg.warmup_epochs, std::max(cfg.epochs - 1, 0)) * cfg.nb) : 0;
  dev_ = dalloc<OptDev>(1);
  OptDev h{cfg.scale_init, 0, 0, 0, 0, 0.f};
  if (!cfg.loss_scaling) h.scale = 1.f;
  CK(cudaMemcpy(dev_, &h, sizeof(h), cudaMemcpyHostToDevice));
  mom_ = dalloc<float>(P.total);
  buf_ = dalloc<float>(P.total);
  ema_ = dalloc<float>(P.total);
  part_ = dalloc<float>(1024);
  CK(cudaMemset(mom_, 0, P.total * 4));
  CK(cudaMemset(buf_, 0, P.total * 4));
  CK(cudaMemcpy(ema_, P.vals, P.total * 4, cudaMemcpyDeviceToDevice));
  // 원소별 종류
  std::vector<uint8_t> kind(P.total, 3);
  std::map<std::pair<int, int>, NsGroup> groups;
  std::map<std::pair<int, int>, std::vector<MatInfo>> infos;
  for (auto& p : P.list) {
    int grp = p.kind == P_CONV ? 0 : p.kind == P_BN_W ? 1 : (p.kind == P_BN_B || p.kind == P_BIAS) ? 2 : 3;
    bool boost = p.name.rfind("model.23.cv3.", 0) == 0 || p.name.rfind("model.23.one2one_cv3.", 0) == 0;
    for (int64_t i = 0; i < p.n; ++i) kind[p.off + i] = uint8_t(grp | (boost ? 4 : 0));
    if (grp == 0) {
      int R0 = (int)p.shape[0], C0 = (int)(p.n / p.shape[0]);
      bool tr = R0 > C0;
      int r = tr ? C0 : R0, cc = tr ? R0 : C0;
      auto& g = groups[{r, cc}];
      g.rows = r;
      g.cols = cc;
      infos[{r, cc}].push_back({p.off, tr ? 1 : 0, boost ? cfg.head_lr_mult : 1.f});
    }
  }
  kind_ = dalloc<uint8_t>(P.total);
  CK(cudaMemcpy(kind_, kind.data(), P.total, cudaMemcpyHostToDevice));
  for (auto& [k, g] : groups) {
    auto& inf = infos[k];
    g.n = (int)inf.size();
    int64_t per = int64_t(g.rows) * g.cols;
    g.X = dalloc<__nv_bfloat16>(per * g.n);
    g.T = dalloc<__nv_bfloat16>(per * g.n);
    g.A = dalloc<__nv_bfloat16>(int64_t(g.rows) * g.rows * g.n);
    g.B = dalloc<__nv_bfloat16>(int64_t(g.rows) * g.rows * g.n);
    g.mats = dalloc<MatInfo>(g.n);
    CK(cudaMemcpy(g.mats, inf.data(), sizeof(MatInfo) * g.n, cudaMemcpyHostToDevice));
    ns_.push_back(g);
  }
}

int Optimizer::accumulate(int64_t ni) const {
  double full = (double)cfg.nbs / cfg.batch;
  if (nw_ > 0 && ni <= nw_) return std::max(1, (int)std::lround(1.0 + (full - 1.0) * (double)ni / nw_));
  return std::max((int)std::lround(full), 1);
}

void Optimizer::newton_schulz(NsGroup& g) {
  const float a = 3.4445f, b = -4.7750f, c = 2.0315f;
  int r = g.rows, cc = g.cols, n = g.n;
  long long sx = (long long)r * cc, sa = (long long)r * r;
  ns_norm_k<<<n, 256, 0, s_>>>(g.X, sx, dev_);
  for (int it = 0; it < 5; ++it) {
    gemm_bf16(blas_, false, true, r, r, cc, 1.f, g.X, cc, sx, g.X, cc, sx, 0.f, g.A, r, sa, n);       // A = X Xᵀ
    CK(cudaMemcpyAsync(g.B, g.A, sizeof(__nv_bfloat16) * sa * n, cudaMemcpyDeviceToDevice, s_));
    gemm_bf16(blas_, false, false, r, r, r, c, g.A, r, sa, g.A, r, sa, b, g.B, r, sa, n);             // B = bA + cA²
    CK(cudaMemcpyAsync(g.T, g.X, sizeof(__nv_bfloat16) * sx * n, cudaMemcpyDeviceToDevice, s_));
    gemm_bf16(blas_, false, false, r, cc, r, 1.f, g.B, r, sa, g.T, cc, sx, a, g.X, cc, sx, n);        // X = aX + BX
  }
}

void Optimizer::step(int64_t ni, int epoch) {
  auto& P = net_->params;
  double lf = (1.0 - (double)epoch / cfg.epochs) * (1.0 - cfg.lrf) + cfg.lrf;
  double base = cfg.lr0 * lf, mom = cfg.momentum;
  double lr = base;
  if (nw_ > 0 && ni <= nw_) {
    double t = (double)ni / nw_;
    lr = base * t;  // 워밍업: 0 → lr0·lf (optimizer=auto 는 warmup_bias_lr = 0)
    mom = cfg.warmup_momentum + (cfg.warm_momentum_end - cfg.warmup_momentum) * t;
  } else if (nw_ > 0) {
    mom = cfg.warm_momentum_end;  // 워밍업 뒤에는 마지막 값(args.momentum) 그대로
  }
  lr_now[0] = lr_now[1] = lr_now[2] = (float)lr;
  mom_now = (float)mom;
  int nblk = 1024;
  unscale_k<<<nblk, 256, 0, s_>>>(P.grads, kind_, P.total, dev_, part_);
  norm_k<<<1, 256, 0, s_>>>(part_, nblk, dev_);
  // Muon 몫
  for (auto& g : ns_) {
    int64_t per = int64_t(g.rows) * g.cols;
    muon_in_k<<<cdiv(per * g.n, 256), 256, 0, s_>>>(P.grads, mom_, g.mats, g.n, g.rows, g.cols, (float)mom, dev_,
                                                     cfg.clip, g.X);
    newton_schulz(g);
    muon_out_k<<<cdiv(per * g.n, 256), 256, 0, s_>>>(P.vals, g.mats, g.n, g.rows, g.cols, g.X,
                                                      (float)(lr * cfg.muon), dev_);
  }
  // SGD 몫(합성곱 무리는 lr·sgd)
  sgd_k<<<nblk, 256, 0, s_>>>(P.vals, P.grads, buf_, kind_, P.total, (float)(lr * cfg.sgd), (float)lr, (float)lr,
                              cfg.head_lr_mult, (float)mom, wd_, dev_, cfg.clip);
  scaler_k<<<1, 1, 0, s_>>>(dev_, cfg.growth_interval, cfg.loss_scaling);
  CK(cudaMemsetAsync(P.grads, 0, P.total * 4, s_));
  ema_k<<<nblk, 256, 0, s_>>>(ema_, P.vals, P.total, dev_, cfg.ema_decay, cfg.ema_tau);
  P.sync_w16(s_);
}

OptDev Optimizer::read_state() const {
  OptDev h;
  CK(cudaMemcpy(&h, dev_, sizeof(h), cudaMemcpyDeviceToHost));
  return h;
}

void Optimizer::save(std::FILE* f) const {
  int64_t n = net_->params.total;
  std::vector<float> h(n);
  for (float* p : {mom_, buf_, ema_}) {
    CK(cudaMemcpy(h.data(), p, n * 4, cudaMemcpyDeviceToHost));
    std::fwrite(h.data(), 4, n, f);
  }
  OptDev d = read_state();
  std::fwrite(&d, sizeof(d), 1, f);
}
void Optimizer::load(std::FILE* f) {
  int64_t n = net_->params.total;
  std::vector<float> h(n);
  for (float* p : {mom_, buf_, ema_}) {
    YT_ASSERT(std::fread(h.data(), 4, n, f) == (size_t)n, "상태 파일이 짧음");
    CK(cudaMemcpy(p, h.data(), n * 4, cudaMemcpyHostToDevice));
  }
  OptDev d;
  YT_ASSERT(std::fread(&d, sizeof(d), 1, f) == 1, "상태 파일이 짧음");
  CK(cudaMemcpy(dev_, &d, sizeof(d), cudaMemcpyHostToDevice));
}
