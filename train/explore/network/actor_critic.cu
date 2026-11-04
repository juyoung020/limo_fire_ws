// 탐사 RL 정책·가치 신경망 — 합성곱(2×2 stride 2, 계층 배치로 GEMM 하나), 완전연결(cuBLAS), ELU, 정규화. 설명은 actor_critic.h·README.md.
//
// 합성곱 배치: 2×2 stride 2 는 조각이 안 겹치므로, 픽셀을 "부모 조각 순서"(쿼드트리)로 늘어놓고 채널을 맨 뒤에 두면
// 층 l 의 입력 [n·P_out][4·cin] 이 그대로 im2col 행렬이 된다. 출력 [n·P_out][cout] 은 다음 층 입력과 같은 배치.
// → im2col·col2im 복사 없이 정방향·역방향이 GEMM 하나씩. u8 입력만 처음에 한 번 이 배치로 바꾼다(/255).
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <stdexcept>

#include "../mdp/obs_layout.h"
#include "actor_critic.h"
#ifndef XNET_FP64
#include "gemm_fused.cuh"
#endif

namespace XNET_NS {

bool g_break_elu_grad = false;

namespace {
constexpr int KK = 4;  // 2×2

// u8 [n][stride] (채널 [c][h][w]) → real [n][계층 픽셀][c] /255. L = 합성곱 층 수, h·w 는 2^L 의 배수.
// 계층 색인 = ((래스터(y>>L, x>>L)·4 + 자식(L−1))·4 + …)·4 + 자식(0), 자식(s) = ((y>>s)&1)·2 + ((x>>s)&1)
// 목적지 순서로 돌아 쓰기를 연속으로 (읽기는 u8 이라 흩어져도 싸다)
__global__ void k_reorder_u8(const uint8_t* src, long sstride, int c, int h, int w, int L, int n, real* dst) {
  long HW = (long)h * w;
  long idx = blockIdx.x * (long)blockDim.x + threadIdx.x;
  if (idx >= (long)n * c * HW) return;
  int ci = (int)(idx % c);
  long p = (idx / c) % HW;
  long i = idx / (HW * c);
  int y = 0, x = 0;
  for (int s = 0; s < L; ++s) {  // 아래 자식 비트부터
    int ch = (int)(p & 3);
    y |= (ch >> 1) << s;
    x |= (ch & 1) << s;
    p >>= 2;
  }
  int wL = w >> L;
  y |= (int)(p / wL) << L;
  x |= (int)(p % wL) << L;
  dst[idx] = src[i * sstride + ((long)ci * h + y) * w + x] * (real)(1.0 / 255.0);
}

__device__ __forceinline__ real elu(real z) { return z > 0.f ? z : expm1(z); }

// Y [rows][cols] 에 열 편향 더하고 (act 면) ELU
__global__ void k_bias_act(real* Y, long rows, int cols, const real* b, bool act) {
  long idx = blockIdx.x * (long)blockDim.x + threadIdx.x;
  if (idx >= rows * cols) return;
  real z = Y[idx] + b[idx % cols];
  Y[idx] = act ? elu(z) : z;
}

// dA *= ELU'(z) — 출력 a 로: a>0 → 1, 아니면 a+1 (= e^z)
__global__ void k_elu_back(real* dA, const real* A, long n, bool broken) {
  long idx = blockIdx.x * (long)blockDim.x + threadIdx.x;
  if (idx >= n) return;
  real a = A[idx];
  real d = (a > 0.f || broken) ? 1.f : a + 1.f;
  dA[idx] *= d;
}

// 행 복사: dst[i·ldd + off + j] = src[i·lds + j] (j < d)
__global__ void k_copy_rows(const real* src, int lds, int n, int d, real* dst, long ldd, long off) {
  long idx = blockIdx.x * (long)blockDim.x + threadIdx.x;
  if (idx >= (long)n * d) return;
  long i = idx / d;
  int j = (int)(idx % d);
  dst[i * ldd + off + j] = src[i * lds + j];
}


// 정규화 갱신: 블록 하나가 차원 32 개(레인 = 차원, 읽기가 이어짐), 행 묶음 NG 개가 샘플을 나눠 double 로 더한 뒤 묶음 순서대로 합침 — 결정적.
// (예전: 차원마다 스레드 하나가 2048 샘플을 차례로 — 스레드 1336 개뿐이라 act 마다 0.4 ms)
constexpr int kNormG = 16;
__global__ void __launch_bounds__(32 * kNormG) k_norm_update(const real* x, int n, int ld, int D, real* mean, real* var,
                                                             double count_old, double count_new) {
  __shared__ double sh[kNormG][32];
  __shared__ double mx_s[32];
  const int lane = threadIdx.x, g = threadIdx.y;
  const int j = blockIdx.x * 32 + lane;
  const bool ok = j < D;
  double s = 0;
  if (ok)
    for (int i = g; i < n; i += kNormG) s += x[(long)i * ld + j];
  sh[g][lane] = s;
  __syncthreads();
  if (g == 0) {
    double t = 0;
    for (int k = 0; k < kNormG; ++k) t += sh[k][lane];
    mx_s[lane] = t / n;
  }
  __syncthreads();
  const double mx = mx_s[lane];
  double v = 0;
  if (ok)
    for (int i = g; i < n; i += kNormG) {
      double d = x[(long)i * ld + j] - mx;
      v += d * d;
    }
  __syncthreads();
  sh[g][lane] = v;
  __syncthreads();
  if (g != 0 || !ok) return;
  double vv0 = 0;
  for (int k = 0; k < kNormG; ++k) vv0 += sh[k][lane];
  vv0 /= n;  // unbiased=False (rsl_rl normalizer.py:67)
  double rate = (double)n / count_new;
  double m = mean[j], vv = var[j];
  double dm = mx - m;
  m += rate * dm;
  vv += rate * (vv0 - vv + dm * (mx - m));
  mean[j] = (real)m;
  var[j] = (real)vv;
}

__global__ void k_norm_apply(const real* x, int n, int ldx, int D, const real* mean, const real* var, real* y,
                             int ldy, real eps) {
  long idx = blockIdx.x * (long)blockDim.x + threadIdx.x;
  if (idx >= (long)n * D) return;
  int i = (int)(idx / D), j = (int)(idx % D);
  y[(long)i * ldy + j] = (x[(long)i * ldx + j] - mean[j]) / (sqrt(var[j]) + eps);
}
// 넷째 층(꼬리) im2col: F3 [n][h·w][C] (래스터) → Tc [n·oh·ow][자식 2×2][C], 격자 밖(덧댐)은 0
__global__ void k_im2col_pad(const real* F3, int n, int h, int w, int C, int oh, int ow, real* Tc) {
  long k = blockIdx.x * (long)blockDim.x + threadIdx.x;
  const long P4 = (long)oh * ow;
  if (k >= (long)n * P4 * 4 * C) return;
  int c = (int)(k % C);
  long r = k / C;
  int child = (int)(r % 4);
  r /= 4;
  int p = (int)(r % P4);
  long i = r / P4;
  int y = 2 * (p / ow) + (child >> 1), x = 2 * (p % ow) + (child & 1);
  Tc[k] = (y < h && x < w) ? F3[i * ((long)h * w * C) + ((long)y * w + x) * C + c] : (real)0;
}
// 그 역: dF3 [n][h·w][C] = dTc 의 해당 칸 (칸마다 하나 — 2×2 s2 는 안 겹침)
__global__ void k_col2im(const real* dTc, int n, int h, int w, int C, int oh, int ow, real* dF3) {
  long k = blockIdx.x * (long)blockDim.x + threadIdx.x;
  if (k >= (long)n * h * w * C) return;
  int c = (int)(k % C);
  long r = k / C;
  int x = (int)(r % w);
  r /= w;
  int y = (int)(r % h);
  long i = r / h;
  long row = i * ((long)oh * ow) + (long)(y / 2) * ow + (x / 2);
  dF3[k] = dTc[(row * 4 + (y & 1) * 2 + (x & 1)) * C + c];
}
}  // namespace

// ---------------- Net ----------------
void Net::build(const NetSpec& s, std::vector<ParamBlock>& reg, long& cur, const std::string& pfx) {
  s_ = s;
  br_.clear();
  lin_.clear();
  feat_img_ = 0;
  const int L = (int)s.conv_ch.size();
  if (!s.imgs.empty() && (L < 1 || L > 4)) throw std::runtime_error("합성곱 층 수는 1~4");
  const int L3 = std::min(L, 3);  // 앞 세 층 = 계층 배치(융합 커널), 넷째 층 = 덧댄 꼬리
  for (size_t b = 0; b < s.imgs.size(); ++b) {
    Branch B;
    B.in = s.imgs[b];
    int c = B.in.c, h = B.in.h, w = B.in.w;
    if (h % (1 << L3) || w % (1 << L3))
      throw std::runtime_error("CNN 입력 " + std::to_string(h) + "×" + std::to_string(w) + " 이 2^층수 의 배수가 아님");
    for (int l = 0; l < L3; ++l) {
      ConvL C{c, h, w, s.conv_ch[l], h / 2, w / 2, 0, 0};
      std::string nm = pfx + ".cnn" + std::to_string(b) + ".conv" + std::to_string(l);
      C.w_off = cur;  // W [cout][자식 4][cin]
      reg.push_back({cur, (long)C.cout * C.cin * KK, PKind::ConvW, nm + ".w"});
      cur += (long)C.cout * C.cin * KK;
      C.b_off = cur;
      reg.push_back({cur, C.cout, PKind::ConvB, nm + ".b"});
      cur += C.cout;
      B.L.push_back(C);
      c = C.cout, h = C.oh, w = C.ow;
    }
    if (L == 4) {
      ConvL C{c, h, w, s.conv_ch[3], (h + 1) / 2, (w + 1) / 2, 0, 0};
      std::string nm = pfx + ".cnn" + std::to_string(b) + ".conv3";
      C.w_off = cur;
      reg.push_back({cur, (long)C.cout * C.cin * KK, PKind::ConvW, nm + ".w"});
      cur += (long)C.cout * C.cin * KK;
      C.b_off = cur;
      reg.push_back({cur, C.cout, PKind::ConvB, nm + ".b"});
      cur += C.cout;
      B.tail = true;
      B.T = C;
      c = C.cout, h = C.oh, w = C.ow;
    }
    B.flat_off = feat_img_;
    feat_img_ += (long)c * h * w;
    br_.push_back(B);
  }
  feat_ = feat_img_ + s.in1d;
  grad_cols_ = s.grad_cols < 0 ? (int)feat_img_ : s.grad_cols;
  int in = (int)feat_;
  std::vector<int> dims = s.hidden;
  if (s.out > 0) dims.push_back(s.out);  // 출력 0 = MLP 없음 (공유 인코더: 특징 X0 가 곧 출력)
  for (size_t l = 0; l < dims.size(); ++l) {
    LinL Lw{in, dims[l], 0, 0};
    std::string nm = pfx + ".mlp" + std::to_string(l);
    Lw.w_off = cur;
    reg.push_back({cur, (long)Lw.out * Lw.in, PKind::LinW, nm + ".w"});
    cur += (long)Lw.out * Lw.in;
    Lw.b_off = cur;
    reg.push_back({cur, Lw.out, PKind::LinB, nm + ".b"});
    cur += Lw.out;
    lin_.push_back(Lw);
    in = Lw.out;
  }
}

CnnBranch Net::desc(const Branch& B) const {
  // 꼬리가 있으면 앞 세 층 출력은 가지 버퍼 F3 [n][h3·w3·c] 로
  const auto& C3 = B.L.back();
  const long f3 = (long)C3.oh * C3.ow * C3.cout;
  return CnnBranch{B.in.c, B.in.h, B.in.w, B.L[0].cout, B.L.size() > 1 ? B.L[1].cout : 0,
                   B.L.size() > 2 ? B.L[2].cout : 0, B.in.off, s_.img_stride, B.tail ? f3 : feat_, B.tail ? 0 : B.flat_off};
}

void Net::bind(real* params, real* grads, int max_n) {
  P_ = params;
  G_ = grads;
  max_n_ = max_n;
  ws_bytes_ = 0;
  auto mk = [&](long nfl) {
    real* p;
    CK(cudaMalloc(&p, nfl * sizeof(real)));
    ws_bytes_ += nfl * sizeof(real);
    return p;
  };
#ifndef XNET_FP64
  // 융합 커널: 지원하는 모양이면 쓴다. XNET_CNN_REF=1 이면 기준 경로(합성곱 = cuBLAS GEMM) — TODO(제거): 전후 A/B 실측용
  static const bool force_ref = std::getenv("XNET_CNN_REF") != nullptr;
  long part = 0;
  for (auto& B : br_) {
    B.fused = !force_ref && cnn_fused_supported(desc(B), (int)B.L.size());
    if (B.fused) part = std::max(part, cnn_fused_part_floats(desc(B)));
  }
  if (part) part_ = mk(part);
#endif
  for (auto& B : br_) {
    if (!B.fused) {  // 융합 가지는 중간 활성값을 전역에 두지 않음
      B.X = mk((long)max_n * B.in.c * B.in.h * B.in.w);
      for (auto& C : B.L) {
        long a = (long)max_n * C.oh * C.ow * C.cout;
        B.A.push_back(mk(a));
        B.dA.push_back(mk(a));
      }
    }
    if (B.tail) {
      const auto& C3 = B.L.back();
      const long f3 = (long)max_n * C3.oh * C3.ow * C3.cout, p4 = (long)max_n * B.T.oh * B.T.ow;
      B.F3 = B.fused ? mk(f3) : B.A.back();  // 기준 경로는 셋째 층 출력(맨 위 단계 = 래스터)이 곧 F3
      B.dF3 = B.fused ? mk(f3) : B.dA.back();
      B.Tc = mk(p4 * KK * B.T.cin);
      B.dTc = mk(p4 * KK * B.T.cin);
      B.Ty = mk(p4 * B.T.cout);
      B.dTy = mk(p4 * B.T.cout);
    }
  }
  X0_ = mk(feat_ * max_n);
  dX0_ = mk(feat_ * max_n);
  for (size_t l = 0; l + 1 < lin_.size(); ++l) {
    H_.push_back(mk((long)lin_[l].out * max_n));
    dH_.push_back(mk((long)lin_[l].out * max_n));
  }
}

void Net::free_ws() {
  for (auto& B : br_) {
    if (B.tail) {
      if (B.fused) {
        cudaFree(B.F3);
        cudaFree(B.dF3);
      }
      for (real* p : {B.Tc, B.dTc, B.Ty, B.dTy}) cudaFree(p);
      B.F3 = B.dF3 = B.Tc = B.dTc = B.Ty = B.dTy = nullptr;
    }
    cudaFree(B.X);
    B.X = nullptr;
    for (auto p : B.A) cudaFree(p);
    for (auto p : B.dA) cudaFree(p);
    B.A.clear();
    B.dA.clear();
  }
  for (auto p : H_) cudaFree(p);
  for (auto p : dH_) cudaFree(p);
  H_.clear();
  dH_.clear();
  cudaFree(X0_);
  cudaFree(dX0_);
  cudaFree(part_);
  X0_ = dX0_ = part_ = nullptr;
}

void Net::init(std::mt19937_64& rng, std::vector<real>& host) const {
  auto uni = [&](long off, long n, float bound) {  // fp64 빌드도 같은 값이 나오게 float 로 뽑는다
    std::uniform_real_distribution<float> U(-bound, bound);
    for (long i = 0; i < n; ++i) host[off + i] = U(rng);
  };
  for (auto& B : br_) {
    std::vector<ConvL> Ls = B.L;
    if (B.tail) Ls.push_back(B.T);
    for (auto& C : Ls) {
      float bd = 1.f / std::sqrt((float)C.cin * KK);
      uni(C.w_off, (long)C.cout * C.cin * KK, bd);
      uni(C.b_off, C.cout, bd);
    }
  }
  for (auto& Lw : lin_) {
    float bd = 1.f / std::sqrt((float)Lw.in);
    uni(Lw.w_off, (long)Lw.out * Lw.in, bd);
    uni(Lw.b_off, Lw.out, bd);
  }
}

bool Net::cnn_reads_index() const {
  for (auto& B : br_)
    if (!B.fused) return false;
  return true;
}

void Net::forward(cublasHandle_t h, cudaStream_t st, int n, const uint8_t* img, const real* x1d, int ld1d,
                  real* out, const int* img_idx) {
  if (img_idx && !cnn_reads_index()) throw std::runtime_error("Net::forward: 이 구조는 img_idx 를 못 읽음 (cnn_reads_index)");
  if (n > max_n_) throw std::runtime_error("Net::forward: n > max_n");
  for (auto& B : br_) {
#ifndef XNET_FP64
    if (B.fused) {
      cnn_fused_forward(desc(B), P_ + B.L[0].w_off, img, img_idx, n, B.tail ? B.F3 : X0_, st, cnn_bf16);
      if (B.tail) tail_forward(h, st, n, B);
      continue;
    }
#endif
    long tot = (long)n * B.in.c * B.in.h * B.in.w;
    k_reorder_u8<<<nblk(tot), 256, 0, st>>>(img + B.in.off, s_.img_stride, B.in.c, B.in.h, B.in.w, (int)B.L.size(), n, B.X);
    for (size_t l = 0; l < B.L.size(); ++l) {
      auto& C = B.L[l];
      long rows = (long)n * C.oh * C.ow;
      int K = C.cin * KK;
      const real* X = l == 0 ? B.X : B.A[l - 1];
      gemm(h, false, true, (int)rows, C.cout, K, 1.f, X, K, P_ + C.w_off, K, 0.f, B.A[l], C.cout);
      k_bias_act<<<nblk(rows * C.cout), 256, 0, st>>>(B.A[l], rows, C.cout, P_ + C.b_off, true);
    }
    if (B.tail) {
      tail_forward(h, st, n, B);
      continue;
    }
    auto& C = B.L.back();
    int d = C.oh * C.ow * C.cout;  // 샘플마다 연속 [픽셀][채널]
    k_copy_rows<<<nblk((long)n * d), 256, 0, st>>>(B.A.back(), d, n, d, X0_, feat_, B.flat_off);
  }
  if (s_.in1d > 0) k_copy_rows<<<nblk((long)n * s_.in1d), 256, 0, st>>>(x1d, ld1d, n, s_.in1d, X0_, feat_, feat_img_);
  const real* x = X0_;
#ifndef XNET_FP64
  if (mlp_mode != 0) {  // 직접 짠 GEMM: 편향(+ELU)을 같은 커널에서
    for (size_t l = 0; l < lin_.size(); ++l) {
      auto& Lw = lin_[l];
      bool last = l + 1 == lin_.size();
      real* y = last ? out : H_[l];
      FGemm g;
      g.M = n, g.N = Lw.out, g.K = Lw.in;
      g.A = x, g.lda = Lw.in, g.ta = false;
      g.B = P_ + Lw.w_off, g.ldb = Lw.in, g.tb = true;
      g.Bh = Ph_ ? Ph_ + Lw.w_off : nullptr;
      g.C = y, g.ldc = Lw.out;
      g.epi = last ? FEpi::Bias : FEpi::BiasElu;
      g.bias = P_ + Lw.b_off;
      fgemm(g, mlp_mode == 2, st);
      x = y;
    }
    return;
  }
#endif
  for (size_t l = 0; l < lin_.size(); ++l) {
    auto& Lw = lin_[l];
    bool last = l + 1 == lin_.size();
    real* y = last ? out : H_[l];
    gemm(h, false, true, n, Lw.out, Lw.in, 1.f, x, Lw.in, P_ + Lw.w_off, Lw.in, 0.f, y, Lw.out);
    k_bias_act<<<nblk((long)n * Lw.out), 256, 0, st>>>(y, n, Lw.out, P_ + Lw.b_off, !last);
    x = y;
  }
}

void Net::backward(cublasHandle_t h, cudaStream_t st, int n, const uint8_t* img, const real* dout, const int* img_idx) {
  if (img_idx && !cnn_reads_index()) throw std::runtime_error("Net::backward: 이 구조는 img_idx 를 못 읽음 (cnn_reads_index)");
  // 기준 경로의 첫 층 입력은 forward 때 바꿔 둔 B.X 를 쓰고, 융합 경로는 img 를 다시 읽는다
  // MLP
  const real* dy = dout;
#ifndef XNET_FP64
  if (mlp_mode != 0) {  // dW 누적 + 편향 기울기(같은 커널), dX + ELU 미분(같은 커널). 첫 층 dX 는 CNN 열만(1D 입력 기울기는 안 씀)
    for (int l = (int)lin_.size() - 1; l >= 0; --l) {
      auto& Lw = lin_[l];
      const real* x = l == 0 ? X0_ : H_[l - 1];
      FGemm w;
      w.M = Lw.out, w.N = Lw.in, w.K = n;
      w.A = dy, w.lda = Lw.out, w.ta = true;
      w.B = x, w.ldb = Lw.in, w.tb = false;
      w.C = G_ + Lw.w_off, w.ldc = Lw.in, w.epi = FEpi::Accum;
      w.bias_grad = G_ + Lw.b_off;
      fgemm(w, mlp_mode == 2, st);
      FGemm d;
      d.M = n, d.K = Lw.out;
      d.A = dy, d.lda = Lw.out, d.ta = false;
      d.B = P_ + Lw.w_off, d.ldb = Lw.in, d.tb = false;
      d.Bh = Ph_ ? Ph_ + Lw.w_off : nullptr;
      if (l > 0) {
        d.N = Lw.in, d.C = dH_[l - 1], d.ldc = Lw.in;
        d.epi = FEpi::EluBack, d.aux = H_[l - 1], d.ldaux = Lw.in, d.broken = g_break_elu_grad;
      } else {
        d.N = grad_cols_, d.C = dX0_, d.ldc = (int)feat_;
        d.epi = FEpi::Store;
      }
      if (d.N > 0) fgemm(d, mlp_mode == 2, st);
      dy = l > 0 ? dH_[l - 1] : nullptr;
    }
  } else
#endif
  for (int l = (int)lin_.size() - 1; l >= 0; --l) {
    auto& Lw = lin_[l];
    const real* x = l == 0 ? X0_ : H_[l - 1];
    gemm(h, true, false, Lw.out, Lw.in, n, 1.f, dy, Lw.out, x, Lw.in, 1.f, G_ + Lw.w_off, Lw.in);  // dW += dYᵀ X
    col_sum(dy, n, Lw.out, Lw.out, G_ + Lw.b_off, true, st);
    real* dx = l == 0 ? dX0_ : dH_[l - 1];
    int ncol = l == 0 ? grad_cols_ : Lw.in;  // 첫 층은 앞 grad_cols 열만 (합성곱·인코더 특징 — 1D 입력 기울기는 안 씀)
    if (ncol > 0)
      gemm(h, false, false, n, ncol, Lw.out, 1.f, dy, Lw.out, P_ + Lw.w_off, Lw.in, 0.f, dx, l == 0 ? (int)feat_ : Lw.in);  // dX = dY W
    if (l > 0) k_elu_back<<<nblk((long)n * Lw.in), 256, 0, st>>>(dx, H_[l - 1], (long)n * Lw.in, g_break_elu_grad);
    dy = dx;
  }
  // CNN — MLP 없는 망(공유 인코더)은 dout 이 곧 특징 기울기 [n][feat]
  const real* dsrc = lin_.empty() ? dout : dX0_;
  for (auto& B : br_) {
    if (B.tail) tail_backward(h, st, n, B, dsrc);  // → B.dF3 (기준 경로는 B.dA[2] 와 같은 버퍼)
#ifndef XNET_FP64
    if (B.fused) {
      cnn_fused_backward(desc(B), P_ + B.L[0].w_off, img, img_idx, n, B.tail ? B.F3 : X0_, B.tail ? B.dF3 : dsrc,
                         G_ + B.L[0].w_off, part_, st, g_break_elu_grad, cnn_bf16);
      continue;
    }
#endif
    if (!B.tail) {
      auto& C = B.L.back();
      int d = C.oh * C.ow * C.cout;
      k_copy_rows<<<nblk((long)n * d), 256, 0, st>>>(dsrc + B.flat_off, (int)feat_, n, d, B.dA.back(), d, 0);
    }
    for (int l = (int)B.L.size() - 1; l >= 0; --l) {
      auto& C = B.L[l];
      long rows = (long)n * C.oh * C.ow;
      int K = C.cin * KK;
      static const bool unfused = std::getenv("XNET_UNFUSED") != nullptr;  // TODO(제거): 융합 전후 A/B 실측용
      if (unfused) {
        k_elu_back<<<nblk(rows * C.cout), 256, 0, st>>>(B.dA[l], B.A[l], rows * C.cout, g_break_elu_grad);
        col_sum(B.dA[l], rows, C.cout, C.cout, G_ + C.b_off, true, st);
      } else {
        col_sum(B.dA[l], rows, C.cout, C.cout, G_ + C.b_off, true, st, B.A[l], g_break_elu_grad);  // ELU 미분 + 편향 기울기 한 번에
      }
      const real* X = l == 0 ? B.X : B.A[l - 1];
      gemm(h, true, false, C.cout, K, (int)rows, 1.f, B.dA[l], C.cout, X, K, 1.f, G_ + C.w_off, K);  // dW += dZᵀ X
      if (l > 0)  // dX = dZ W → 앞 층 출력 기울기 (같은 배치)
        gemm(h, false, false, (int)rows, K, C.cout, 1.f, B.dA[l], C.cout, P_ + C.w_off, K, 0.f, B.dA[l - 1], K);
    }
  }
}

// 넷째 층(꼬리): F3 → 덧댄 im2col → GEMM → 편향 + ELU → X0 의 가지 열
void Net::tail_forward(cublasHandle_t h, cudaStream_t st, int n, Branch& B) {
  const auto& T = B.T;
  const long rows = (long)n * T.oh * T.ow;
  const int K = T.cin * KK;
  k_im2col_pad<<<nblk(rows * K), 256, 0, st>>>(B.F3, n, T.h, T.w, T.cin, T.oh, T.ow, B.Tc);
  gemm(h, false, true, (int)rows, T.cout, K, 1.f, B.Tc, K, P_ + T.w_off, K, 0.f, B.Ty, T.cout);
  k_bias_act<<<nblk(rows * T.cout), 256, 0, st>>>(B.Ty, rows, T.cout, P_ + T.b_off, true);
  const int d = T.oh * T.ow * T.cout;
  k_copy_rows<<<nblk((long)n * d), 256, 0, st>>>(B.Ty, d, n, d, X0_, feat_, B.flat_off);
}
void Net::tail_backward(cublasHandle_t h, cudaStream_t st, int n, Branch& B, const real* dsrc) {
  const auto& T = B.T;
  const long rows = (long)n * T.oh * T.ow;
  const int K = T.cin * KK, d = T.oh * T.ow * T.cout;
  k_copy_rows<<<nblk((long)n * d), 256, 0, st>>>(dsrc + B.flat_off, (int)feat_, n, d, B.dTy, d, 0);
  col_sum(B.dTy, rows, T.cout, T.cout, G_ + T.b_off, true, st, B.Ty, g_break_elu_grad);  // ELU 미분 + 편향 기울기
  gemm(h, true, false, T.cout, K, (int)rows, 1.f, B.dTy, T.cout, B.Tc, K, 1.f, G_ + T.w_off, K);  // dW += dZᵀ Tc
  gemm(h, false, false, (int)rows, K, T.cout, 1.f, B.dTy, T.cout, P_ + T.w_off, K, 0.f, B.dTc, K);  // dTc = dZ W
  k_col2im<<<nblk((long)n * T.h * T.w * T.cin), 256, 0, st>>>(B.dTc, n, T.h, T.w, T.cin, T.oh, T.ow, B.dF3);
}

// ---------------- Normalizer ----------------
void Normalizer::init(int d) {
  dim = d;
  CK(cudaMalloc(&d_mean, d * sizeof(real)));
  CK(cudaMalloc(&d_var, d * sizeof(real)));
  CK(cudaMemset(d_mean, 0, d * sizeof(real)));
  std::vector<real> one(d, 1.f);
  CK(cudaMemcpy(d_var, one.data(), d * sizeof(real), cudaMemcpyHostToDevice));
  count = 0;
}
void Normalizer::release() {
  cudaFree(d_mean);
  cudaFree(d_var);
  d_mean = d_var = nullptr;
}
void Normalizer::update(const real* x, int n, int ld, cudaStream_t st) {
  double cn = count + n;
  k_norm_update<<<(dim + 31) / 32, dim3(32, kNormG), 0, st>>>(x, n, ld, dim, d_mean, d_var, count, cn);
  count = cn;
}
void Normalizer::apply(const real* x, int n, int ldx, real* y, int ldy, cudaStream_t st, int d) const {
  if (d < 0) d = dim;
  k_norm_apply<<<nblk((long)n * d), 256, 0, st>>>(x, n, ldx, d, d_mean, d_var, y, ldy, kEps);
}

// ---------------- 기본 구조 ----------------
ACSpec explore_spec() {
  namespace o = mdp::obs;
  ACSpec s;
  NetSpec e;
  e.imgs = {{0, o::NEAR_C, o::NEAR_H, o::NEAR_W},
            {o::NEAR_SIZE, o::FAR_C, o::FAR_H, o::FAR_W},
            {o::NEAR_SIZE + o::FAR_SIZE, o::DEPTH_C, o::DEPTH_H, o::DEPTH_W}};
  e.img_stride = o::IMG_SIZE;
  e.conv_ch = {16, 32, 32, 32};  // 4 층: 평탄화 near 4×4·far 2×2·depth 3×4 (5 행을 6 으로 덧댐) × 32 = 1024 (network/README)
  e.in1d = 0;
  e.out = 0;
  s.enc = e;
  NetSpec a;
  a.in1d = o::POLICY_1D;
  a.hidden = {512, 256, 128};
  a.out = mdp::act::DIM;
  NetSpec c = a;
  c.in1d = o::CRITIC_1D;
  c.out = 1;
  s.actor = a;
  s.critic = c;
  s.n_cont = mdp::act::CONT;
  s.norm_dim = o::CRITIC_1D;
  return s;
}

// ---------------- ActorCritic ----------------
ActorCritic::ActorCritic(const ACSpec& sp, uint64_t seed, int mc) : spec(sp), max_chunk(mc) {
  if (sp.actor.out != sp.n_cont + 1 || sp.critic.out != 1) throw std::runtime_error("ACSpec 출력 크기");
  if (sp.actor.in1d > sp.norm_dim || sp.critic.in1d != sp.norm_dim) throw std::runtime_error("ACSpec 1D 크기");
  if (!sp.actor.imgs.empty() || !sp.critic.imgs.empty() || sp.enc.in1d != 0 || sp.enc.out != 0 || !sp.enc.hidden.empty())
    throw std::runtime_error("ACSpec: 합성곱은 enc 에만, actor·critic 은 MLP 만");
  long cur = 0;
  enc.build(sp.enc, blocks, cur, "enc");
  nf = (int)enc.feat_img();
  NetSpec as = sp.actor, cs = sp.critic;  // 입력 = [F | 1D], 앞 nf 열 기울기를 인코더로
  as.in1d += nf;
  cs.in1d += nf;
  as.grad_cols = cs.grad_cols = nf;
  actor.build(as, blocks, cur, "actor");
  critic.build(cs, blocks, cur, "critic");
  logstd_off = cur;
  blocks.push_back({cur, sp.n_cont, PKind::LogStd, "log_std"});
  cur += sp.n_cont;
  n_params = cur;
  CK(cudaMalloc(&d_params, n_params * sizeof(real)));
  CK(cudaMalloc(&d_grads, n_params * sizeof(real)));
  CK(cudaStreamCreate(&stream));
  CKB(cublasCreate(&cublas));
  CKB(cublasSetStream(cublas, stream));
  CK(cudaStreamCreateWithFlags(&stream2, cudaStreamNonBlocking));
  CKB(cublasCreate(&cublas2));
  CKB(cublasSetStream(cublas2, stream2));
  CKB(cublasSetMathMode(cublas2, CUBLAS_DEFAULT_MATH));
  CK(cudaEventCreateWithFlags(&ev_fork_, cudaEventDisableTiming));
  CK(cudaEventCreateWithFlags(&ev_join_, cudaEventDisableTiming));
  CKB(cublasSetMathMode(cublas, CUBLAS_DEFAULT_MATH));  // fp32 그대로 (TF32 안 씀)
  enc.bind(d_params, d_grads, mc);
  actor.bind(d_params, d_grads, mc);
  critic.bind(d_params, d_grads, mc);
  std::vector<real> host(n_params);
  std::mt19937_64 rng(seed);
  enc.init(rng, host);
  actor.init(rng, host);
  critic.init(rng, host);
  for (int j = 0; j < sp.n_cont; ++j) host[logstd_off + j] = sp.init_log_std;
  // 선언 로짓 = actor 마지막 층 편향의 마지막 원소
  const ParamBlock* last_b = nullptr;
  for (auto& b : blocks)
    if (b.name.rfind("actor.mlp", 0) == 0 && b.kind == PKind::LinB) last_b = &b;  // actor 의 마지막 선형 층 편향
  host[last_b->off + last_b->n - 1] = sp.declare_bias;
  CK(cudaMemcpy(d_params, host.data(), n_params * sizeof(real), cudaMemcpyHostToDevice));
  zero_grad();
  norm.init(sp.norm_dim);
#ifndef XNET_FP64
  CK(cudaMalloc(&d_params_h, n_params * sizeof(unsigned short)));
  actor.Ph_ = critic.Ph_ = d_params_h;
  params_changed();
#endif
}

ActorCritic::~ActorCritic() {
  enc.free_ws();
  actor.free_ws();
  critic.free_ws();
  norm.release();
  cudaFree(d_params);
  cudaFree(d_grads);
  cudaFree(d_params_h);
  cublasDestroy(cublas);
  cublasDestroy(cublas2);
  cudaEventDestroy(ev_fork_);
  cudaEventDestroy(ev_join_);
  cudaStreamDestroy(stream);
  cudaStreamDestroy(stream2);
}

void ActorCritic::fork() {
  CK(cudaEventRecord(ev_fork_, stream));
  CK(cudaStreamWaitEvent(stream2, ev_fork_, 0));
}
void ActorCritic::join() {
  CK(cudaEventRecord(ev_join_, stream2));
  CK(cudaStreamWaitEvent(stream, ev_join_, 0));
}

void ActorCritic::params_changed() {
#ifndef XNET_FP64
  to_bf16(d_params, d_params_h, n_params, stream);
#endif
}

void ActorCritic::zero_grad() { CK(cudaMemsetAsync(d_grads, 0, n_params * sizeof(real), stream)); }

}  // namespace XNET_NS
