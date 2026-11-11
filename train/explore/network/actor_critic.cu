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
#include <cuda_bf16.h>

#include "cnn_tail.cuh"
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

#ifndef XNET_FP64
// 행 복사 bf16: dst[i·ldd + off + j] = bf16(src[i·lds + j]) (기준 경로 넷째 층의 bf16 사본)
__global__ void k_copy_rows_h(const float* src, int lds, int n, int d, unsigned short* dst, long ldd, long off) {
  long idx = blockIdx.x * (long)blockDim.x + threadIdx.x;
  if (idx >= (long)n * d) return;
  long i = idx / d;
  int j = (int)(idx % d);
  dst[i * ldd + off + j] = __bfloat16_as_ushort(__float2bfloat16_rn(src[i * lds + j]));
}
#endif
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
#ifndef XNET_FP64
// cuBLAS 층 정방향 에필로그: Y = ELU(Y + b) (융합 GEMM 에필로그와 같은 식) 와 그 bf16 사본
__global__ void k_bias_elu_h(float* Y, unsigned short* Yh, long rows, int cols, const float* b) {
  long idx = blockIdx.x * (long)blockDim.x + threadIdx.x;
  if (idx >= rows * cols) return;
  // 열 번호: 원소 수가 2^32 안이면 32 비트 나머지 (64 비트 나머지는 명령 수십 개)
  const int c = rows * cols <= 0xffffffffL ? (int)((unsigned)idx % (unsigned)cols) : (int)(idx % cols);
  const float z = Y[idx] + b[c];
  const float v = z > 0.f ? z : expm1f(z);
  Y[idx] = v;
  Yh[idx] = __bfloat16_as_ushort(__float2bfloat16_rn(v));
}
// cuBLAS 층 역방향 입력: 블록 (행 묶음 g, 열 256 개), 스레드 = 열 하나가 묶음의 행을 차례로 —
// (aux 면 dy ·= ELU'(aux) 제자리, 융합 dX 에필로그와 같은 식) → bf16 사본, 열 부분합 part[g][c]
__global__ void k_dy_prep(float* dy, const float* aux, bool broken, long R, int C, long rpb, unsigned short* dyh, float* part) {
  const int c = blockIdx.y * blockDim.x + threadIdx.x;
  if (c >= C) return;
  const long r0 = blockIdx.x * rpb, r1 = r0 + rpb < R ? r0 + rpb : R;
  float s = 0.f;
  long r = r0;
  for (; r + 4 <= r1; r += 4) {  // 네 행을 먼저 다 읽고(지연 겹침) 같은 순서로 더한다
    float v[4], a[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      v[i] = dy[(r + i) * C + c];
      a[i] = aux ? aux[(r + i) * C + c] : 1.f;
    }
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const long o = (r + i) * C + c;
      if (aux) {
        v[i] *= (a[i] > 0.f || broken) ? 1.f : a[i] + 1.f;
        dy[o] = v[i];
      }
      dyh[o] = __bfloat16_as_ushort(__float2bfloat16_rn(v[i]));
      s += v[i];
    }
  }
  for (; r < r1; ++r) {
    const long o = r * C + c;
    float v = dy[o];
    if (aux) {
      const float a = aux[o];
      v *= (a > 0.f || broken) ? 1.f : a + 1.f;
      dy[o] = v;
    }
    dyh[o] = __bfloat16_as_ushort(__float2bfloat16_rn(v));
    s += v;
  }
  part[(long)blockIdx.x * C + c] = s;
}
// 부분합을 묶음 순서대로 더해 편향 기울기에 누적
__global__ void k_dy_prep_fin(const float* part, int nb, int C, float* g) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= C) return;
  float s = 0.f;
  for (int b = 0; b < nb; ++b) s += part[(long)b * C + c];
  g[c] += s;
}
#endif
// 넷째 층 역방향 입력: dst [n][d] = g1[i·ld1 + j] (+ g2[i·ld2 + j]) — 두 MLP 의 특징 기울기를 고정 순서(g1 + g2)로 더함
__global__ void k_add_rows(const real* g1, long ld1, const real* g2, long ld2, int n, int d, real* dst) {
  long idx = blockIdx.x * (long)blockDim.x + threadIdx.x;
  if (idx >= (long)n * d) return;
  long i = idx / d;
  int j = (int)(idx % d);
  real v = g1[i * ld1 + j];
  if (g2) v = v + g2[i * ld2 + j];
  dst[idx] = v;
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

#ifndef XNET_FP64
namespace {
// 가지의 넷째 층 모양 (Net::Branch 는 private 라 템플릿으로 받음)
template <class Br>
TailShape tail_shape(const Br& B, int n) {
  return TailShape{n, B.T.h, B.T.w, B.T.cin, B.T.oh, B.T.ow, B.T.cout};
}
}  // namespace
#endif

// ---------------- Net ----------------
void Net::build(const NetSpec& s, std::vector<ParamBlock>& reg, long& cur, const std::string& pfx) {
  s_ = s;
  br_.clear();
  lin_.clear();
  feat_img_ = 0;
  // 합성곱은 4 층: 앞 세 층 = 계층 배치(융합 커널), 넷째 층 = 덧댄 꼬리 (3 층 구조 = 체크포인트 형식 1 은 불러오기부터 거부)
  if (!s.imgs.empty() && s.conv_ch.size() != 4) throw std::runtime_error("합성곱 층 수는 4");
  constexpr int L3 = 3;
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
    {
      ConvL C{c, h, w, s.conv_ch[3], (h + 1) / 2, (w + 1) / 2, 0, 0};
      std::string nm = pfx + ".cnn" + std::to_string(b) + ".conv3";
      C.w_off = cur;
      reg.push_back({cur, (long)C.cout * C.cin * KK, PKind::ConvW, nm + ".w"});
      cur += (long)C.cout * C.cin * KK;
      C.b_off = cur;
      reg.push_back({cur, C.cout, PKind::ConvB, nm + ".b"});
      cur += C.cout;
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
  // 앞 세 층 출력은 가지 버퍼 F3 [n][h3·w3·c] 로 (넷째 층이 읽음)
  const auto& C3 = B.L.back();
  const long f3 = (long)C3.oh * C3.ow * C3.cout;
  return CnnBranch{B.in.c, B.in.h, B.in.w, B.L[0].cout, B.L[1].cout, B.L[2].cout, B.in.off, s_.img_stride, f3, 0};
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
    B.tail_fused = !force_ref && cnn_tail_supported(tail_shape(B, 0));
    if (B.tail_fused) part = std::max(part, cnn_tail_part_floats(tail_shape(B, 0)));
    if (B.fused) part = std::max(part, cnn_fused_part_floats(desc(B)));
    // 융합 합성곱 역방향은 넷째 층 융합 역방향이 쓰는 dZ3 (= dF3 · ELU'(F3)) 를 받는다
    if (B.fused && !B.tail_fused) throw std::runtime_error("Net::bind: 융합 합성곱에는 융합 넷째 층이 필요");
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
    {
      const auto& C3 = B.L.back();
      const long f3 = (long)max_n * C3.oh * C3.ow * C3.cout, p4 = (long)max_n * B.T.oh * B.T.ow;
      B.F3 = B.fused ? mk(f3) : B.A.back();  // 기준 경로는 셋째 층 출력(맨 위 단계 = 래스터)이 곧 F3
      B.dF3 = B.fused ? mk(f3) : B.dA.back();
      if (!B.tail_fused) {  // 융합 꼬리는 im2col·출력 사본을 전역에 두지 않음
        B.Tc = mk(p4 * KK * B.T.cin);
        B.dTc = mk(p4 * KK * B.T.cin);
        B.Ty = mk(p4 * B.T.cout);
        B.dTy = mk(p4 * B.T.cout);
      }
    }
  }
  // X0: 합성곱 출력(+ 1D). 합성곱 없는 망(MLP 만)은 forward 의 x1d 를 바로 읽어 X0 가 없다
  ldx0_ = feat_;
  own_x0_ = true;
  if (!br_.empty()) X0_ = mk(feat_ * max_n);
  if (!lin_.empty()) dX0_ = mk(feat_ * max_n);
#ifndef XNET_FP64
  for (size_t l = 0; l + 1 < lin_.size(); ++l) {  // cuBLAS bf16 층의 은닉 출력·출력 기울기 bf16 사본, 편향 기울기 부분합
    const size_t b = (size_t)max_n * lin_[l].out * sizeof(unsigned short);
    unsigned short *a, *d;
    CK(cudaMalloc(&a, b));
    CK(cudaMalloc(&d, b));
    Hh_.push_back(a);
    dyh_.push_back(d);
    ws_bytes_ += 2 * b;
  }
  if (lin_.size() > 1) {
    int cmax = 0;
    for (auto& Lw : lin_) cmax = std::max(cmax, Lw.out);
    CK(cudaMalloc(&bpart_, (size_t)kPrepRows * cmax * sizeof(real)));
  }
#endif
  for (size_t l = 0; l + 1 < lin_.size(); ++l) {
    H_.push_back(mk((long)lin_[l].out * max_n));
    dH_.push_back(mk((long)lin_[l].out * max_n));
  }
}

void Net::set_x0(real* buf, long ld, unsigned short* bufh) {
  if (br_.empty() || !lin_.empty()) throw std::runtime_error("Net::set_x0: MLP 없는 합성곱 망(인코더)만");
  if (ld < feat_) throw std::runtime_error("Net::set_x0: 행 간격 < 특징 수");
  if (own_x0_) cudaFree(X0_);
  X0_ = buf;
  ldx0_ = ld;
  own_x0_ = false;
  X0h_ = bufh;
}

void Net::dy_prep(real* dy, const real* aux, int n, int C, unsigned short* dyh, real* bias_grad, cudaStream_t st) {
#ifdef XNET_FP64
  throw std::runtime_error("dy_prep: fp32 빌드만");
#else
  if (n <= 0) return;
  const long rpb = (n + kPrepRows - 1) / kPrepRows;
  const int nb = (int)((n + rpb - 1) / rpb);
  k_dy_prep<<<dim3(nb, (C + 255) / 256), 256, 0, st>>>(dy, aux, g_break_elu_grad, n, C, rpb, dyh, bpart_);
  k_dy_prep_fin<<<nblk(C, 128), 128, 0, st>>>(bpart_, nb, C, bias_grad);
#endif
}

bool Net::cublas_ok(int l) const {  // 텐서 코어 정렬: 행 간격·폭 8 의 배수, W bf16 사본 16 B 정렬
  const auto& Lw = lin_[l];
  return Lw.in % 8 == 0 && Lw.out % 8 == 0 && (Lw.w_off - wh_lo_) % 8 == 0 && (l > 0 || ldxin_ % 8 == 0);
}

void Net::mlp_range(long& lo, long& n) const {
  if (lin_.empty()) {
    lo = n = 0;
    return;
  }
  lo = lin_.front().w_off;
  n = lin_.back().b_off + lin_.back().out - lo;
}

void Net::free_ws() {
  for (auto& B : br_) {
    if (B.fused) {
      cudaFree(B.F3);
      cudaFree(B.dF3);
    }
    for (real* p : {B.Tc, B.dTc, B.Ty, B.dTy}) cudaFree(p);
    B.F3 = B.dF3 = B.Tc = B.dTc = B.Ty = B.dTy = nullptr;
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
  if (own_x0_) cudaFree(X0_);
  cudaFree(dX0_);
  for (auto p : Hh_) cudaFree(p);
  for (auto p : dyh_) cudaFree(p);
  Hh_.clear();
  dyh_.clear();
  cudaFree(bpart_);
  bpart_ = nullptr;
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
    Ls.push_back(B.T);
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
                  real* out, const int* img_idx, const unsigned short* x1d_h) {
  if (img_idx && !cnn_reads_index()) throw std::runtime_error("Net::forward: 이 구조는 img_idx 를 못 읽음 (cnn_reads_index)");
  if (n > max_n_) throw std::runtime_error("Net::forward: n > max_n");
  for (auto& B : br_) {
#ifndef XNET_FP64
    if (B.fused) {
      cnn_fused_forward(desc(B), P_ + B.L[0].w_off, img, img_idx, n, B.F3, st, cnn_bf16);
      tail_forward(h, st, n, B);
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
    tail_forward(h, st, n, B);
  }
  if (lin_.empty()) return;
  // MLP 첫 층 입력: 합성곱 없는 망은 x1d 를 그대로(복사 없음), 아니면 X0 = [합성곱 평탄화 | x1d]
  xinh_ = nullptr;
  if (br_.empty()) {
    xin_ = x1d, ldxin_ = ld1d;
    xinh_ = x1d_h;
  }
  ncub_ = 0;
#ifndef XNET_FP64
  // 앞에서부터 cuBLAS bf16 로 돌 층 수 (마지막 출력 층은 폭이 작아 융합 커널). 조건이 안 맞는 층부터는 융합 커널
  if (xinh_ && mlp_mode == 2 && Wh_)
    while (ncub_ + 1 < (int)lin_.size() && cublas_ok(ncub_)) ++ncub_;
#endif
  if (xinh_ && ncub_ == 0) throw std::runtime_error("Net::forward: bf16 입력 사본을 줬지만 첫 층 cuBLAS bf16 조건이 안 맞음 (mlp 2·행 간격 8 의 배수)");
  if (!br_.empty()) {
    if (s_.in1d > 0) k_copy_rows<<<nblk((long)n * s_.in1d), 256, 0, st>>>(x1d, ld1d, n, s_.in1d, X0_, ldx0_, feat_img_);
    xin_ = X0_, ldxin_ = ldx0_;
  }
  const real* x = xin_;
#ifndef XNET_FP64
  if (mlp_mode != 0) {  // 직접 짠 GEMM: 편향(+ELU)을 같은 커널에서
    for (size_t l = 0; l < lin_.size(); ++l) {
      auto& Lw = lin_[l];
      bool last = l + 1 == lin_.size();
      real* y = last ? out : H_[l];
      if ((int)l < ncub_) {  // cuBLAS bf16 (fp32 누산) 뒤 편향 + ELU — 융합 커널 에필로그와 같은 식 (z = 누산 + b, ELU = expm1f), 다음 층용 bf16 사본도
        const float one = 1.f, zero = 0.f;
        const unsigned short* xh = l == 0 ? xinh_ : Hh_[l - 1];
        CKB(cublasGemmEx(h, CUBLAS_OP_T, CUBLAS_OP_N, Lw.out, n, Lw.in, &one, ph(Lw.w_off), CUDA_R_16BF, Lw.in, xh, CUDA_R_16BF,
                         l == 0 ? (int)ldxin_ : Lw.in, &zero, y, CUDA_R_32F, Lw.out, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
        k_bias_elu_h<<<nblk((long)n * Lw.out), 256, 0, st>>>(y, Hh_[l], n, Lw.out, P_ + Lw.b_off);
        x = y;
        continue;
      }
      FGemm g;
      g.M = n, g.N = Lw.out, g.K = Lw.in;
      g.A = x, g.lda = l == 0 ? (int)ldxin_ : Lw.in, g.ta = false;
      g.B = P_ + Lw.w_off, g.ldb = Lw.in, g.tb = true;
      g.Bh = ph(Lw.w_off);
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
    gemm(h, false, true, n, Lw.out, Lw.in, 1.f, x, l == 0 ? (int)ldxin_ : Lw.in, P_ + Lw.w_off, Lw.in, 0.f, y, Lw.out);
    k_bias_act<<<nblk((long)n * Lw.out), 256, 0, st>>>(y, n, Lw.out, P_ + Lw.b_off, !last);
    x = y;
  }
}

void Net::backward(cublasHandle_t h, cudaStream_t st, int n, const uint8_t* img, const real* dout, const int* img_idx) {
  if (lin_.empty()) {  // MLP 없는 망(인코더): dout 이 곧 특징 기울기 [n][feat]
    backward_feat(h, st, n, img, img_idx, dout, feat_, nullptr, 0);
    return;
  }
  // MLP — 첫 층 입력은 직전 forward 의 xin_ (합성곱 없는 망은 그때의 x1d)
  const real* dy = dout;
#ifndef XNET_FP64
  if (mlp_mode != 0) {  // dW 누적 + 편향 기울기(같은 커널), dX + ELU 미분(같은 커널). 첫 층 dX 는 CNN 열만(1D 입력 기울기는 안 씀)
    for (int l = (int)lin_.size() - 1; l >= 0; --l) {
      auto& Lw = lin_[l];
      const real* x = l == 0 ? xin_ : H_[l - 1];
      if (l < ncub_) {  // cuBLAS bf16: dY 를 bf16 으로(융합 커널도 같은 반올림) → dW·dX, 편향 기울기는 fp32 dY 열 합
        const float one = 1.f, zero = 0.f;
        // 위 층이 cuBLAS 였으면 dy 는 날 dX — ELU'(이 층 출력) 을 여기서 곱한다 (융합 커널은 dX 에필로그에서 곱함, 같은 식)
        const bool raw = l + 1 < ncub_;
        dy_prep(const_cast<real*>(dy), raw ? H_[l] : nullptr, n, Lw.out, dyh_[l], G_ + Lw.b_off, st);  // raw 면 dy = dH_[l] (자기 버퍼)
        const unsigned short* xh = l == 0 ? xinh_ : Hh_[l - 1];
        CKB(cublasGemmEx(h, CUBLAS_OP_N, CUBLAS_OP_T, Lw.in, Lw.out, n, &one, xh, CUDA_R_16BF, l == 0 ? (int)ldxin_ : Lw.in, dyh_[l],
                         CUDA_R_16BF, Lw.out, &one, G_ + Lw.w_off, CUDA_R_32F, Lw.in, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));  // dW += dYᵀ X
        const int ncol = l == 0 ? grad_cols_ : Lw.in;  // 첫 층은 앞 grad_cols 열만
        real* dx = l == 0 ? dX0_ : dH_[l - 1];
        if (ncol > 0)  // dX = dY W (날값 — 아래 층이 ELU' 을 곱함)
          CKB(cublasGemmEx(h, CUBLAS_OP_N, CUBLAS_OP_N, ncol, n, Lw.out, &one, ph(Lw.w_off), CUDA_R_16BF, Lw.in, dyh_[l], CUDA_R_16BF,
                           Lw.out, &zero, dx, CUDA_R_32F, l == 0 ? (int)feat_ : Lw.in, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
        dy = l > 0 ? dH_[l - 1] : nullptr;
        continue;
      }
      FGemm w;
      w.M = Lw.out, w.N = Lw.in, w.K = n;
      w.A = dy, w.lda = Lw.out, w.ta = true;
      w.B = x, w.ldb = l == 0 ? (int)ldxin_ : Lw.in, w.tb = false;
      w.C = G_ + Lw.w_off, w.ldc = Lw.in, w.epi = FEpi::Accum;
      w.bias_grad = G_ + Lw.b_off;
      fgemm(w, mlp_mode == 2, st);
      FGemm d;
      d.M = n, d.K = Lw.out;
      d.A = dy, d.lda = Lw.out, d.ta = false;
      d.B = P_ + Lw.w_off, d.ldb = Lw.in, d.tb = false;
      d.Bh = ph(Lw.w_off);
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
    const real* x = l == 0 ? xin_ : H_[l - 1];
    gemm(h, true, false, Lw.out, Lw.in, n, 1.f, dy, Lw.out, x, l == 0 ? (int)ldxin_ : Lw.in, 1.f, G_ + Lw.w_off, Lw.in);  // dW += dYᵀ X
    col_sum(dy, n, Lw.out, Lw.out, G_ + Lw.b_off, true, st);
    real* dx = l == 0 ? dX0_ : dH_[l - 1];
    int ncol = l == 0 ? grad_cols_ : Lw.in;  // 첫 층은 앞 grad_cols 열만 (합성곱·인코더 특징 — 1D 입력 기울기는 안 씀)
    if (ncol > 0)
      gemm(h, false, false, n, ncol, Lw.out, 1.f, dy, Lw.out, P_ + Lw.w_off, Lw.in, 0.f, dx, l == 0 ? (int)feat_ : Lw.in);  // dX = dY W
    if (l > 0) k_elu_back<<<nblk((long)n * Lw.in), 256, 0, st>>>(dx, H_[l - 1], (long)n * Lw.in, g_break_elu_grad);
    dy = dx;
  }
  if (!br_.empty()) backward_feat(h, st, n, img, img_idx, dX0_, feat_, nullptr, 0);
}

void Net::backward_feat(cublasHandle_t h, cudaStream_t st, int n, const uint8_t* img, const int* img_idx, const real* g1, long ld1,
                        const real* g2, long ld2) {
  if (img_idx && !cnn_reads_index()) throw std::runtime_error("Net::backward: 이 구조는 img_idx 를 못 읽음 (cnn_reads_index)");
  // 기준 경로의 첫 층 입력은 forward 때 바꿔 둔 B.X 를 쓰고, 융합 경로는 img 를 다시 읽는다
  for (auto& B : br_) {
    tail_backward(h, st, n, B, g1, ld1, g2, ld2);  // → B.dF3 (융합 합성곱이면 dZ3 = dF3·ELU′(F3); 기준 경로는 B.dA[2] 와 같은 버퍼)
#ifndef XNET_FP64
    if (B.fused) {
      cnn_fused_backward(desc(B), P_ + B.L[0].w_off, img, img_idx, n, B.dF3, G_ + B.L[0].w_off, part_, st, g_break_elu_grad,
                         cnn_bf16);
      continue;
    }
#endif
    for (int l = (int)B.L.size() - 1; l >= 0; --l) {
      auto& C = B.L[l];
      long rows = (long)n * C.oh * C.ow;
      int K = C.cin * KK;
      col_sum(B.dA[l], rows, C.cout, C.cout, G_ + C.b_off, true, st, B.A[l], g_break_elu_grad);  // ELU 미분 + 편향 기울기 한 번에
      const real* X = l == 0 ? B.X : B.A[l - 1];
      gemm(h, true, false, C.cout, K, (int)rows, 1.f, B.dA[l], C.cout, X, K, 1.f, G_ + C.w_off, K);  // dW += dZᵀ X
      if (l > 0)  // dX = dZ W → 앞 층 출력 기울기 (같은 배치)
        gemm(h, false, false, (int)rows, K, C.cout, 1.f, B.dA[l], C.cout, P_ + C.w_off, K, 0.f, B.dA[l - 1], K);
    }
  }
}

// 넷째 층(꼬리): F3 → 덧댄 im2col → GEMM → 편향 + ELU → X0 의 가지 열. fp32 빌드는 융합 커널 하나(cnn_tail), 아래 im2col·cuBLAS 는 기준 경로(fp64 검사·XNET_CNN_REF)
void Net::tail_forward(cublasHandle_t h, cudaStream_t st, int n, Branch& B) {
  const auto& T = B.T;
#ifndef XNET_FP64
  if (B.tail_fused) {
    cnn_tail_forward(tail_shape(B, n), B.F3, P_ + T.w_off, X0_ + B.flat_off, ldx0_, st, X0h_ ? X0h_ + B.flat_off : nullptr);
    return;
  }
#endif
  const long rows = (long)n * T.oh * T.ow;
  const int K = T.cin * KK;
  k_im2col_pad<<<nblk(rows * K), 256, 0, st>>>(B.F3, n, T.h, T.w, T.cin, T.oh, T.ow, B.Tc);
  gemm(h, false, true, (int)rows, T.cout, K, 1.f, B.Tc, K, P_ + T.w_off, K, 0.f, B.Ty, T.cout);
  k_bias_act<<<nblk(rows * T.cout), 256, 0, st>>>(B.Ty, rows, T.cout, P_ + T.b_off, true);
  const int d = T.oh * T.ow * T.cout;
  k_copy_rows<<<nblk((long)n * d), 256, 0, st>>>(B.Ty, d, n, d, X0_, ldx0_, B.flat_off);
#ifndef XNET_FP64
  if (X0h_) k_copy_rows_h<<<nblk((long)n * d), 256, 0, st>>>(B.Ty, d, n, d, X0h_, ldx0_, B.flat_off);
#endif
}
void Net::tail_backward(cublasHandle_t h, cudaStream_t st, int n, Branch& B, const real* g1, long ld1, const real* g2, long ld2) {
  const auto& T = B.T;
#ifndef XNET_FP64
  if (B.tail_fused) {
    cnn_tail_backward(tail_shape(B, n), B.F3, P_ + T.w_off, X0_ + B.flat_off, ldx0_, g1 + B.flat_off, ld1, g2 ? g2 + B.flat_off : nullptr, ld2,
                      B.dF3, G_ + T.w_off, part_, g_break_elu_grad, st, /*z_out=*/B.fused);
    return;
  }
#endif
  const long rows = (long)n * T.oh * T.ow;
  const int K = T.cin * KK, d = T.oh * T.ow * T.cout;
  k_add_rows<<<nblk((long)n * d), 256, 0, st>>>(g1 + B.flat_off, ld1, g2 ? g2 + B.flat_off : nullptr, ld2, n, d, B.dTy);
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
  s.split_enc = true;   // 2026-11-08: actor·critic 인코더 따로 (체크포인트 형식 3)
  return s;
}

ACSpec explore_spec_hier() {
  namespace h = mdp::hier;
  ACSpec s = explore_spec();
  s.actor.in1d = h::POLICY_1D;
  s.actor.out = 64;   // 상태 임베딩 (잠정 — 마지막 은닉 128 의 반)
  s.critic.in1d = h::CRITIC_1D;
  s.norm_dim = h::CRITIC_1D;
  s.n_cont = 0;
  s.n_slots = h::K;
  s.slot_feat = h::FEAT;
  s.slot_off = mdp::obs::POLICY_1D;
  s.slot_hidden = 64;   // 잠정
  return s;
}

ACSpec explore_spec_graph() {
  namespace g = mdp::graph;
  ACSpec s;
  s.graph = true;
  s.gspec.gw = g::GW;
  s.gspec.gh = g::GH;
  s.gspec.feat = g::FEAT;
  s.graph_glob = g::GLOB;
  NetSpec c;
  c.in1d = mdp::obs::PRIV;
  c.hidden = {128};   // 잠정 (전작의 Q 머리는 선형 한 층 — README "논문과 다른 곳")
  c.out = 1;
  s.critic = c;
  s.n_cont = 0;
  s.norm_dim = mdp::obs::PRIV;
  return s;
}

// ---------------- ActorCritic ----------------
void ActorCritic::alloc_common() {
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
}

// 그래프 어텐션 정책: 정책 몸통 + 포인터 머리, 가치 몸통(따로 또는 공유) + 가치 머리 MLP. 계산은 한 스트림에서 차례로 (stream2 안 씀)
void ActorCritic::init_graph(uint64_t seed) {
  const ACSpec& sp = spec;
  if (sp.n_cont != 0 || sp.n_slots != 0 || sp.critic.out != 1 || !sp.critic.imgs.empty() || sp.critic.in1d != sp.norm_dim || sp.norm_dim < 1 ||
      sp.graph_glob < 0)
    throw std::runtime_error("ACSpec(그래프): n_cont 0, critic = 참값 MLP (in1d = norm_dim ≥ 1, out 1)");
  if (sp.graph_rows < (long)sp.gspec.gw * sp.gspec.gh) throw std::runtime_error("ACSpec(그래프): graph_rows 가 그래프 하나(gw·gh)보다 작음");
  const int d = sp.gspec.d;
  long cur = 0;
  gbody.build(sp.gspec, blocks, cur, "gactor");
  ptr.build(d, sp.graph_glob, blocks, cur, "gactor");
  if (!sp.graph_share) gbody_c.build(sp.gspec, blocks, cur, "gcritic");
  NetSpec cs = sp.critic;   // 입력 = [E | Hc | 참값], 앞 2d 열 기울기를 몸통으로
  cs.in1d += 2 * d;
  cs.grad_cols = 2 * d;
  critic.build(cs, blocks, cur, "gcritic.v");
  logstd_off = cur;
  n_params = cur;
  alloc_common();
  gbody.bind(d_params, d_grads, sp.graph_rows, max_chunk);
  ptr.alloc(max_chunk);
  if (!sp.graph_share) gbody_c.bind(d_params, d_grads, sp.graph_rows, max_chunk);
  critic.bind(d_params, d_grads, max_chunk);
  critic.mlp_mode = 0;
  std::vector<real> host(n_params);
  std::mt19937_64 rng(seed);
  gbody.init(rng, host);
  ptr.init(rng, host);
  if (!sp.graph_share) gbody_c.init(rng, host);
  critic.init(rng, host);
  CK(cudaMemcpy(d_params, host.data(), n_params * sizeof(real), cudaMemcpyHostToDevice));
  zero_grad();
  norm.init(sp.norm_dim);
}

ActorCritic::ActorCritic(const ACSpec& sp, uint64_t seed, int mc) : spec(sp), max_chunk(mc) {
  if (sp.graph) {
    init_graph(seed);
    return;
  }
  if (sp.n_slots > 0 ? (sp.n_cont != 0 || sp.actor.out < 1 || sp.slot_feat < 1 || sp.slot_off + sp.n_slots * sp.slot_feat > sp.actor.in1d)
                     : sp.actor.out != sp.n_cont + 1)
    throw std::runtime_error("ACSpec 출력 크기");
  if (sp.critic.out != 1) throw std::runtime_error("ACSpec 출력 크기");
  if (sp.actor.in1d > sp.norm_dim || sp.critic.in1d != sp.norm_dim) throw std::runtime_error("ACSpec 1D 크기");
  if (!sp.actor.imgs.empty() || !sp.critic.imgs.empty() || sp.enc.in1d != 0 || sp.enc.out != 0 || !sp.enc.hidden.empty())
    throw std::runtime_error("ACSpec: 합성곱은 enc 에만, actor·critic 은 MLP 만");
  long cur = 0;
  enc.build(sp.enc, blocks, cur, "enc");
  nf = (int)enc.feat_img();
  split = sp.split_enc;
  if (split) enc_c.build(sp.enc, blocks, cur, "enc_c");
  NetSpec as = sp.actor, cs = sp.critic;  // 입력 = [F | 1D], 앞 nf 열 기울기를 인코더로
  as.in1d += nf;
  cs.in1d += nf;
  as.grad_cols = cs.grad_cols = nf;
  actor.build(as, blocks, cur, "actor");
  critic.build(cs, blocks, cur, "critic");
  if (sp.n_slots > 0) {   // 슬롯 머리 (slot_head.cuh): W1 [H][E + C], b1 [H], w2 [H]
    head.E = sp.actor.out;
    head.C = sp.slot_feat;
    head.K = sp.n_slots;
    head.H = sp.slot_hidden;
    const long nw1 = (long)head.H * (head.E + head.C);
    head.w1_off = cur; blocks.push_back({cur, nw1, PKind::LinW, "head.w1"}); cur += nw1;
    head.b1_off = cur; blocks.push_back({cur, head.H, PKind::LinB, "head.b1"}); cur += head.H;
    head.w2_off = cur; blocks.push_back({cur, head.H, PKind::LinW, "head.w2"}); cur += head.H;
  }
  logstd_off = cur;
  if (sp.n_cont > 0) blocks.push_back({cur, sp.n_cont, PKind::LogStd, "log_std"});
  cur += sp.n_cont;
  n_params = cur;
  alloc_common();
  enc.bind(d_params, d_grads, mc);
  if (split) enc_c.bind(d_params, d_grads, mc);
  actor.bind(d_params, d_grads, mc);
  critic.bind(d_params, d_grads, mc);
  std::vector<real> host(n_params);
  std::mt19937_64 rng(seed);
  enc.init(rng, host);
  if (split) enc_c.init(rng, host);
  actor.init(rng, host);
  critic.init(rng, host);
  for (int j = 0; j < sp.n_cont; ++j) host[logstd_off + j] = sp.init_log_std;
  if (sp.n_slots > 0) {   // torch nn.Linear 기본 초기화와 같은 꼴 U(±1/√fan_in), float 로 뽑음 (fp64 빌드도 같은 값)
    auto uni = [&](long off, long n, float bound) {
      std::uniform_real_distribution<float> U(-bound, bound);
      for (long i = 0; i < n; ++i) host[off + i] = U(rng);
    };
    const float b1 = 1.f / std::sqrt((float)(head.E + head.C)), b2 = 1.f / std::sqrt((float)head.H);
    uni(head.w1_off, (long)head.H * (head.E + head.C), b1);
    uni(head.b1_off, head.H, b1);
    uni(head.w2_off, head.H, b2);
  }
  // 선언 로짓 = actor 마지막 층 편향의 마지막 원소
  const ParamBlock* last_b = nullptr;
  for (auto& b : blocks)
    if (b.name.rfind("actor.mlp", 0) == 0 && b.kind == PKind::LinB) last_b = &b;  // actor 의 마지막 선형 층 편향
  if (sp.n_slots == 0) host[last_b->off + last_b->n - 1] = sp.declare_bias;
  CK(cudaMemcpy(d_params, host.data(), n_params * sizeof(real), cudaMemcpyHostToDevice));
  zero_grad();
  if (sp.n_slots > 0) head.alloc(mc);
  norm.init(sp.norm_dim);
#ifndef XNET_FP64
  for (Net* m : {&actor, &critic}) {  // bf16 사본은 망마다 따로 (첫 층 W 가 cudaMalloc 정렬 = 16 B 정렬)
    m->mlp_range(m->wh_lo_, m->wh_n_);
    CK(cudaMalloc(&m->Wh_, m->wh_n_ * sizeof(unsigned short)));
  }
  params_changed();
#endif
}

ActorCritic::~ActorCritic() {
  if (spec.n_slots > 0) head.release();
  gbody.free_ws();
  gbody_c.free_ws();
  ptr.release();
  enc.free_ws();
  if (split) enc_c.free_ws();
  actor.free_ws();
  critic.free_ws();
  norm.release();
  cudaFree(d_params);
  cudaFree(d_grads);
  cudaFree(actor.Wh_);
  cudaFree(critic.Wh_);
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
  for (Net* m : {&actor, &critic})
    if (m->Wh_) to_bf16(d_params + m->wh_lo_, m->Wh_, m->wh_n_, stream);   // 그래프 모드는 bf16 사본 없음
#endif
}

void ActorCritic::zero_grad() { CK(cudaMemsetAsync(d_grads, 0, n_params * sizeof(real), stream)); }

}  // namespace XNET_NS
