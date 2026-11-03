// 합성곱 가지 융합 커널 (정방향·역방향). 설명은 cnn_fused.cuh·README.md "계산 방식".
//
// 조각(patch) = 마지막 층 출력 픽셀 하나 = 입력 8×8. 조각 안 배치(기존 계층 배치와 같음):
//   층 0 출력 행 r0 = 조각·16 + q1·4 + c1,  열 = 채널        (입력 행 r0 의 K0 = [자식 c0][cin])
//   층 1 출력 행 r1 = 조각·4 + q1,          열 = 채널        (입력 = 층 0 출력 [r1][c1·C0 + ch], K1 = 4·C0)
//   층 2 출력 행 = 조각,                     열 = 채널        (입력 = 층 1 출력 [조각][q1·C1 + ch], K2 = 4·C1)
//   자식 번호 = (y 비트)·2 + (x 비트): q1 은 조각 안 4×4 칸, c1 은 2×2 칸, c0 은 픽셀.
// 그래서 공유 메모리 안 활성값은 행 우선 [행][열] 하나로 두 층의 입력·출력 모양을 다 쓴다(복사 없음).
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <type_traits>

#include <stdexcept>

#include "cnn_fused.cuh"

namespace xnet {

namespace {

constexpr int NT = 256;  // 블록 스레드 수
constexpr int TP = 16;   // 블록 타일 = 조각 수

// ELU: expm1f 는 느린 길(분기·정밀 보정)이라 층 0 출력 원소마다 부르면 커널 시간의 ~40 % 였다.
// |z| < 1/16 은 4 차 테일러(절단 오차 |z|⁵/120 < 1e-8, 상대 ~1e-7), 그 밖은 __expf(z) − 1 (빼기 상쇄가 작은 범위).
// 분기 없이 둘 다 계산해 고른다 (워프 안 갈림 없음).
__device__ __forceinline__ float elu_f(float z) {
  const float t = z * (1.f + z * (0.5f + z * (0.16666667f + z * 0.041666668f)));
  const float e = __expf(fminf(z, 0.f)) - 1.f;
  return z > 0.f ? z : (z > -0.0625f ? t : e);
}
__device__ __forceinline__ float elu_d(float a, bool broken) { return (a > 0.f || broken) ? 1.f : a + 1.f; }

__device__ __forceinline__ float4 ld4(const float* p) { return *reinterpret_cast<const float4*>(p); }
__device__ __forceinline__ float4 ld4(const uint8_t* p) {
  uchar4 u = *reinterpret_cast<const uchar4*>(p);
  return make_float4(u.x, u.y, u.z, u.w);
}

// 블록 GEMM, 레인이 열을 맡음: epi(r, c, Σ_k A[r][k]·B[k][c]). A [R][K] 행 우선, B [K][C] 행 우선 (둘 다 공유 메모리).
// 스레드마다 RM 행 × CJ 열(c = 레인 + 32·j). A 는 같은 행을 여러 레인이 읽어 브로드캐스트, B 는 레인마다 연속.
// k 순서로 더하므로 RM 등 타일 모양과 무관하게 같은 값 (정방향·역방향 재계산이 비트 단위로 같음).
template <int R, int C, int K, int RM, typename TA, typename Epi>
__device__ __forceinline__ void gemm_lc(const TA* __restrict__ A, const float* __restrict__ B, Epi epi) {
  static_assert(C < 32 || C % 32 == 0, "C");
  static_assert(K % 4 == 0, "K");
  constexpr int CW = C < 32 ? C : 32, LPR = 32 / CW, CJ = (C + 31) / 32, RPP = LPR * RM;
  constexpr int NPASS = (R + RPP - 1) / RPP;
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  const int cl = lane % CW, rs = lane / CW;
  if (rs >= LPR) return;
  for (int pass = warp; pass < NPASS; pass += NT / 32) {
    float acc[RM][CJ];
#pragma unroll
    for (int i = 0; i < RM; ++i)
#pragma unroll
      for (int j = 0; j < CJ; ++j) acc[i][j] = 0.f;
    const int rb = pass * RPP + rs;
    int ro[RM];
#pragma unroll
    for (int i = 0; i < RM; ++i) {
      int r = rb + LPR * i;
      ro[i] = (r < R ? r : R - 1) * K;
    }
#pragma unroll 2
    for (int k = 0; k < K; k += 4) {
      float b[4][CJ];
#pragma unroll
      for (int kk = 0; kk < 4; ++kk)
#pragma unroll
        for (int j = 0; j < CJ; ++j) b[kk][j] = B[(k + kk) * C + cl + 32 * j];
#pragma unroll
      for (int i = 0; i < RM; ++i) {
        float4 a = ld4(A + ro[i] + k);
#pragma unroll
        for (int j = 0; j < CJ; ++j) {
          float s = acc[i][j];
          s = fmaf(a.x, b[0][j], s);
          s = fmaf(a.y, b[1][j], s);
          s = fmaf(a.z, b[2][j], s);
          s = fmaf(a.w, b[3][j], s);
          acc[i][j] = s;
        }
      }
    }
#pragma unroll
    for (int i = 0; i < RM; ++i) {
      int r = rb + LPR * i;
      if (r < R)
#pragma unroll
        for (int j = 0; j < CJ; ++j) epi(r, cl + 32 * j, acc[i][j]);
    }
  }
}

// 가중치 기울기 누적 dW[c][k] += Σ_r Z[r][c]·X[r][k], 편향 db[c] += Σ_r Z[r][c]. 블록 수명 내내 레지스터에.
// 워프가 c 묶음(CPW 개), 레인이 k (k = 레인 + 32·j). 행 순서 고정 → 결정적.
template <int C, int K>
struct DWAcc {
  static constexpr int NWP = NT / 32, CPW = (C + NWP - 1) / NWP, KJ = (K + 31) / 32;
  float w[CPW][KJ];
  float b[CPW];
  __device__ __forceinline__ void zero() {
#pragma unroll
    for (int i = 0; i < CPW; ++i) {
      b[i] = 0.f;
#pragma unroll
      for (int j = 0; j < KJ; ++j) w[i][j] = 0.f;
    }
  }
  template <int R, typename TX>
  __device__ __forceinline__ void add(const float* __restrict__ Z, const TX* __restrict__ X) {
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    if (warp * CPW >= C) return;
#pragma unroll 4
    for (int r = 0; r < R; ++r) {
      float x[KJ], z[CPW];
#pragma unroll
      for (int j = 0; j < KJ; ++j) {
        int k = lane + 32 * j;
        x[j] = (K % 32 == 0 || k < K) ? (float)X[r * K + k] : 0.f;
      }
#pragma unroll
      for (int i = 0; i < CPW; ++i) {
        int c = warp * CPW + i;
        z[i] = (C % NWP == 0 || c < C) ? Z[r * C + c] : 0.f;
      }
#pragma unroll
      for (int i = 0; i < CPW; ++i) {
        b[i] += z[i];
#pragma unroll
        for (int j = 0; j < KJ; ++j) w[i][j] = fmaf(z[i], x[j], w[i][j]);
      }
    }
  }
  __device__ __forceinline__ void store(float* part, int ow, int ob, float wscale) const {
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
#pragma unroll
    for (int i = 0; i < CPW; ++i) {
      int c = warp * CPW + i;
      if (c >= C) continue;
#pragma unroll
      for (int j = 0; j < KJ; ++j) {
        int k = lane + 32 * j;
        if (k < K) part[ow + c * K + k] = w[i][j] * wscale;
      }
      if (lane == 0) part[ob + c] = b[i];
    }
  }
};

template <int CIN, int C0, int C1, int C2>
struct Shape {
  static_assert(C1 % 4 == 0 && C2 % 4 == 0, "채널");
  static constexpr int K0 = 4 * CIN, K1 = 4 * C0, K2 = 4 * C1;
  static constexpr int OW0 = 0, OB0 = C0 * K0, OW1 = OB0 + C0, OB1 = OW1 + C1 * K1, OW2 = OB1 + C1,
                       OB2 = OW2 + C2 * K2, NP = OB2 + C2;
};

__host__ __device__ constexpr int r4(int x) { return (x + 3) & ~3; }

struct Args {
  const float* P;
  const uint8_t* img;
  long img_off;
  int img_stride, H, W, n;
  const int* idx;  // 샘플 i 의 입력 행 = idx[i] (nullptr 이면 i) — 미니배치를 복사 없이 롤아웃에서 바로 읽기
  long feat, flat_off;
  float* X0;
  const float* dX0;
  float* part;
  bool broken;
};

// 가중치를 공유 메모리로 (전치: Wt[k][c] = W[c][k])
template <int C, int K>
__device__ __forceinline__ void load_wt(const float* __restrict__ W, float* Wt) {
  for (int e = threadIdx.x; e < C * K; e += NT) {
    int c = e / K, k = e % K;
    Wt[k * C + c] = W[e];
  }
}
__device__ __forceinline__ void load_v(const float* __restrict__ src, float* dst, int n) {
  for (int e = threadIdx.x; e < n; e += NT) dst[e] = src[e];
}

// u8 타일 → Xs [TP·16][K0] (조각 안 계층 배치). 범위 밖 조각은 0.
template <int CIN>
__device__ __forceinline__ void load_x(const Args& a, long g0, long total, uint8_t* Xs) {
  constexpr int K0 = 4 * CIN;
  const int WP = a.W >> 3, PPS = (a.H >> 3) * WP;
  for (int it = threadIdx.x; it < TP * CIN * 8; it += NT) {
    int y = it & 7, ci = (it >> 3) % CIN, p = (it >> 3) / CIN;
    long g = g0 + p;
    uint2 v = make_uint2(0u, 0u);
    if (g < total) {
      long s = g / PPS;
      int pix = (int)(g % PPS), py = pix / WP, px = pix % WP;
      if (a.idx) s = a.idx[s];
      const uint8_t* src = a.img + s * a.img_stride + a.img_off + ((long)ci * a.H + py * 8 + y) * a.W + px * 8;
      v = *reinterpret_cast<const uint2*>(src);
    }
    const int q1y = (y >> 2) & 1, c1y = (y >> 1) & 1, c0y = y & 1;
#pragma unroll
    for (int x = 0; x < 8; ++x) {
      uint32_t word = x < 4 ? v.x : v.y;
      uint8_t byte = (uint8_t)(word >> (8 * (x & 3)));
      int q1 = q1y * 2 + ((x >> 2) & 1), c1 = c1y * 2 + ((x >> 1) & 1), c0 = c0y * 2 + (x & 1);
      Xs[(p * 16 + q1 * 4 + c1) * K0 + c0 * CIN + ci] = byte;
    }
  }
}

__device__ __forceinline__ long out_idx(const Args& a, long g, int c, int C2) {
  const int PPS = (a.H >> 3) * (a.W >> 3);
  return (g / PPS) * a.feat + a.flat_off + (g % PPS) * C2 + c;
}

// 층 0·1 정방향 (타일 하나): Xs → A0 → A1
template <int CIN, int C0, int C1>
__device__ __forceinline__ void fwd01(const uint8_t* Xs, const float* Wt0, const float* b0, const float* Wt1,
                                      const float* b1, float* A0, float* A1) {
  constexpr int K0 = 4 * CIN, K1 = 4 * C0;
  constexpr float inv = 1.f / 255.f;
  gemm_lc<TP * 16, C0, K0, 8>(Xs, Wt0, [&](int r, int c, float s) { A0[r * C0 + c] = elu_f(fmaf(s, inv, b0[c])); });
  __syncthreads();
  gemm_lc<TP * 4, C1, K1, 4>(A0, Wt1, [&](int r, int c, float s) { A1[r * C1 + c] = elu_f(s + b1[c]); });
  __syncthreads();
}

template <int CIN, int C0, int C1, int C2>
__global__ void __launch_bounds__(NT) k_cnn_fwd(Args a) {
  using S = Shape<CIN, C0, C1, C2>;
  constexpr int K0 = S::K0, K1 = S::K1, K2 = S::K2;
  extern __shared__ float4 smem4[];
  float* sm = reinterpret_cast<float*>(smem4);
  float* Wt0 = sm;
  float* b0 = Wt0 + r4(K0 * C0);
  float* Wt1 = b0 + r4(C0);
  float* b1 = Wt1 + r4(K1 * C1);
  float* Wt2 = b1 + r4(C1);
  float* b2 = Wt2 + r4(K2 * C2);
  float* A0 = b2 + r4(C2);
  float* A1 = A0 + r4(TP * 16 * C0);
  uint8_t* Xs = reinterpret_cast<uint8_t*>(A1 + r4(TP * 4 * C1));
  load_wt<C0, K0>(a.P + S::OW0, Wt0);
  load_v(a.P + S::OB0, b0, C0);
  load_wt<C1, K1>(a.P + S::OW1, Wt1);
  load_v(a.P + S::OB1, b1, C1);
  load_wt<C2, K2>(a.P + S::OW2, Wt2);
  load_v(a.P + S::OB2, b2, C2);
  const long total = (long)a.n * (a.H >> 3) * (a.W >> 3);
  const long ntiles = (total + TP - 1) / TP;
  for (long t = blockIdx.x; t < ntiles; t += gridDim.x) {
    const long g0 = t * TP;
    load_x<CIN>(a, g0, total, Xs);
    __syncthreads();
    fwd01<CIN, C0, C1>(Xs, Wt0, b0, Wt1, b1, A0, A1);
    gemm_lc<TP, C2, K2, 2>(A1, Wt2, [&](int r, int c, float s) {
      long g = g0 + r;
      if (g < total) a.X0[out_idx(a, g, c, C2)] = elu_f(s + b2[c]);
    });
  }
}

template <int CIN, int C0, int C1, int C2>
__global__ void __launch_bounds__(NT) k_cnn_bwd(Args a) {
  using S = Shape<CIN, C0, C1, C2>;
  constexpr int K0 = S::K0, K1 = S::K1, K2 = S::K2;
  extern __shared__ float4 smem4[];
  float* sm = reinterpret_cast<float*>(smem4);
  float* Wt0 = sm;
  float* b0 = Wt0 + r4(K0 * C0);
  float* Wt1 = b0 + r4(C0);
  float* b1 = Wt1 + r4(K1 * C1);
  float* W1 = b1 + r4(C1);       // [C1][K1] (dA0 = Z1·W1)
  float* W2 = W1 + r4(C1 * K1);  // [C2][K2] (dA1 = Z2·W2)
  float* A0 = W2 + r4(C2 * K2);  // 층 0 출력 → 역방향 뒤 Z0 로 덮음
  float* A1 = A0 + r4(TP * 16 * C0);
  float* Z1 = A1 + r4(TP * 4 * C1);
  float* Z2 = Z1 + r4(TP * 4 * C1);
  uint8_t* Xs = reinterpret_cast<uint8_t*>(Z2 + r4(TP * C2));
  load_wt<C0, K0>(a.P + S::OW0, Wt0);
  load_v(a.P + S::OB0, b0, C0);
  load_wt<C1, K1>(a.P + S::OW1, Wt1);
  load_v(a.P + S::OB1, b1, C1);
  load_v(a.P + S::OW1, W1, C1 * K1);
  load_v(a.P + S::OW2, W2, C2 * K2);
  DWAcc<C0, K0> g0acc;
  DWAcc<C1, K1> g1acc;
  DWAcc<C2, K2> g2acc;
  g0acc.zero();
  g1acc.zero();
  g2acc.zero();
  const long total = (long)a.n * (a.H >> 3) * (a.W >> 3);
  const long ntiles = (total + TP - 1) / TP;
  const bool broken = a.broken;
  for (long t = blockIdx.x; t < ntiles; t += gridDim.x) {
    const long g0 = t * TP;
    load_x<CIN>(a, g0, total, Xs);
    for (int e = threadIdx.x; e < TP * C2; e += NT) {  // Z2 = dX0 · ELU'(출력)
      int r = e / C2, c = e % C2;
      long g = g0 + r;
      float z = 0.f;
      if (g < total) {
        long ix = out_idx(a, g, c, C2);
        z = a.dX0[ix] * elu_d(a.X0[ix], broken);
      }
      Z2[e] = z;
    }
    __syncthreads();
    fwd01<CIN, C0, C1>(Xs, Wt0, b0, Wt1, b1, A0, A1);
    // 층 2: dW2 += Z2ᵀ A1, dA1 = Z2 · W2 → Z1 = dA1 · ELU'(A1)
    g2acc.template add<TP>(Z2, A1);
    gemm_lc<TP, K2, C2, 2>(Z2, W2, [&](int r, int c, float s) { Z1[r * K2 + c] = s * elu_d(A1[r * K2 + c], broken); });
    __syncthreads();
    // 층 1: dW1 += Z1ᵀ A0 (행 r1), dA0 = Z1 · W1 → A0 자리에 Z0 = dA0 · ELU'(A0)
    g1acc.template add<TP * 4>(Z1, A0);
    __syncthreads();
    gemm_lc<TP * 4, K1, C1, 4>(Z1, W1, [&](int r, int c, float s) {
      float* p = A0 + r * K1 + c;
      *p = s * elu_d(*p, broken);
    });
    __syncthreads();
    // 층 0: dW0 += Z0ᵀ X (u8, /255 는 저장 때)
    g0acc.template add<TP * 16>(A0, Xs);
    __syncthreads();
  }
  float* part = a.part + (long)blockIdx.x * S::NP;
  g0acc.store(part, S::OW0, S::OB0, 1.f / 255.f);
  g1acc.store(part, S::OW1, S::OB1, 1.f);
  g2acc.store(part, S::OW2, S::OB2, 1.f);
}

// 블록 부분합을 블록 순서대로 더해 기울기에 누적
__global__ void k_part_fin(const float* __restrict__ part, int nb, int np, float* __restrict__ G) {
  int e = blockIdx.x * blockDim.x + threadIdx.x;
  if (e >= np) return;
  float s = 0.f;
  for (int b = 0; b < nb; ++b) s += part[(long)b * np + e];
  G[e] += s;
}

template <int CIN, int C0, int C1, int C2>
size_t smem_fwd() {
  using S = Shape<CIN, C0, C1, C2>;
  return 4 * (size_t)(r4(S::K0 * C0) + r4(C0) + r4(S::K1 * C1) + r4(C1) + r4(S::K2 * C2) + r4(C2) +
                      r4(TP * 16 * C0) + r4(TP * 4 * C1)) +
         (size_t)TP * 16 * S::K0;
}
template <int CIN, int C0, int C1, int C2>
size_t smem_bwd() {
  using S = Shape<CIN, C0, C1, C2>;
  return 4 * (size_t)(r4(S::K0 * C0) + r4(C0) + r4(S::K1 * C1) + r4(C1) + r4(C1 * S::K1) + r4(C2 * S::K2) +
                      r4(TP * 16 * C0) + 2 * r4(TP * 4 * C1) + r4(TP * C2)) +
         (size_t)TP * 16 * S::K0;
}

int num_sms() {
  static int n = [] {
    int d, v;
    CK(cudaGetDevice(&d));
    CK(cudaDeviceGetAttribute(&v, cudaDevAttrMultiProcessorCount, d));
    return v;
  }();
  return n;
}

#include "cnn_fused_tc.cuh"

// 인스턴스 하나: 커널 설정·격자 크기 (기기마다 고정 → 결정적)
struct Inst {
  int cin, c0, c1, c2;
  void (*fwd)(Args);
  void (*bwd)(Args);
  size_t sf, sb;
  int np;
  void (*fwd_tc)(Args) = nullptr;  // 텐서 코어 판 (채널이 16 의 배수일 때만)
  void (*bwd_tc)(Args) = nullptr;
  size_t sf_tc = 0, sb_tc = 0;
  int h = 0, w = 0;  // 텐서 코어 판이 컴파일된 입력 크기
  int gf = 0, gb = 0, gf_tc = 0, gb_tc = 0;  // 상주 블록 수 (처음 쓸 때)
  static int resident(void (*k)(Args), size_t sm, int nt = NT) {
    CK(cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sm));
    int o = 0;
    CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&o, k, nt, sm));
    if (o < 1) throw std::runtime_error("cnn_fused: 공유 메모리 부족");
    return o * num_sms();
  }
  void prepare() {
    if (gb) return;
    gf = resident(fwd, sf);
    gb = resident(bwd, sb);
    if (fwd_tc) {
      gf_tc = resident(fwd_tc, sf_tc, tc::NTT);
      gb_tc = resident(bwd_tc, sb_tc, tc::NTT);
    }
  }
};

#define XNET_CNN_INST(CI, A, B, C)                                                                      \
  Inst {                                                                                                \
    CI, A, B, C, k_cnn_fwd<CI, A, B, C>, k_cnn_bwd<CI, A, B, C>, smem_fwd<CI, A, B, C>(),               \
        smem_bwd<CI, A, B, C>(), Shape<CI, A, B, C>::NP                                                 \
  }
#define XNET_CNN_INST_TC(CI, A, B, C, H, W)                                                                    \
  Inst {                                                                                                \
    CI, A, B, C, k_cnn_fwd<CI, A, B, C>, k_cnn_bwd<CI, A, B, C>, smem_fwd<CI, A, B, C>(),               \
        smem_bwd<CI, A, B, C>(), Shape<CI, A, B, C>::NP, tc::k_cnn_fwd_tc<CI, A, B, C, H, W>,           \
        tc::k_cnn_bwd_tc<CI, A, B, C, H, W>, tc::smem_fwd_tc<CI, A, B, C>(), tc::smem_bwd_tc<CI, A, B, C>(), \
        H, W                                                                                            \
  }

// 실제 구조(가까운 지도 6·먼 지도 7·깊이 1 채널, 16-32-32)와 기울기 검사의 작은 구조(3-4-4)
Inst* find_inst(int cin, int c0, int c1, int c2) {
  static Inst tab[] = {XNET_CNN_INST_TC(6, 16, 32, 32, 64, 64), XNET_CNN_INST_TC(7, 16, 32, 32, 32, 32),
                       XNET_CNN_INST_TC(1, 16, 32, 32, 40, 64),
                       XNET_CNN_INST(2, 3, 4, 4),    XNET_CNN_INST(3, 3, 4, 4),    XNET_CNN_INST(1, 3, 4, 4)};
  for (auto& t : tab)
    if (t.cin == cin && t.c0 == c0 && t.c1 == c1 && t.c2 == c2) return &t;
  return nullptr;
}

Inst& get(const CnnBranch& b) {
  Inst* t = find_inst(b.cin, b.c0, b.c1, b.c2);
  if (!t) throw std::runtime_error("cnn_fused: 지원 안 하는 채널");
  t->prepare();
  return *t;
}

Args make_args(const CnnBranch& b, const float* P, const uint8_t* img, const int* idx, int n) {
  Args a{};
  a.idx = idx;
  a.P = P;
  a.img = img;
  a.img_off = b.img_off;
  a.img_stride = b.img_stride;
  a.H = b.h;
  a.W = b.w;
  a.n = n;
  a.feat = b.feat;
  a.flat_off = b.flat_off;
  return a;
}

int grid_for(const CnnBranch& b, int n, int gmax, int tp = TP) {
  long tiles = ((long)n * (b.h / 8) * (b.w / 8) + tp - 1) / tp;
  return (int)std::max<long>(1, std::min<long>(tiles, gmax));
}

}  // namespace

bool cnn_fused_supported(const CnnBranch& b, int nlayers) {
  return nlayers == 3 && b.h % 8 == 0 && b.w % 8 == 0 && (b.img_off % 8) == 0 && (b.img_stride % 8) == 0 &&
         find_inst(b.cin, b.c0, b.c1, b.c2) != nullptr;
}

long cnn_fused_part_floats(const CnnBranch& b) {
  Inst& t = get(b);
  return (long)std::max(t.gb, t.gb_tc) * t.np;
}

bool cnn_fused_tc_supported(const CnnBranch& b) {
  Inst* t = find_inst(b.cin, b.c0, b.c1, b.c2);
  return t && t->fwd_tc && t->h == b.h && t->w == b.w;
}

void cnn_fused_forward(const CnnBranch& b, const float* params, const uint8_t* img, const int* idx, int n, float* X0,
                       cudaStream_t st, bool tc) {
  if (n <= 0) return;
  Inst& t = get(b);
  Args a = make_args(b, params, img, idx, n);
  a.X0 = X0;
  if (tc && t.fwd_tc && t.h == b.h && t.w == b.w)
    t.fwd_tc<<<grid_for(b, n, t.gf_tc, tc::TPT), tc::NTT, t.sf_tc, st>>>(a);
  else
    t.fwd<<<grid_for(b, n, t.gf), NT, t.sf, st>>>(a);
  CK(cudaGetLastError());
}

void cnn_fused_backward(const CnnBranch& b, const float* params, const uint8_t* img, const int* idx, int n, const float* X0,
                        const float* dX0, float* grads, float* part, cudaStream_t st, bool broken_elu, bool tc) {
  if (n <= 0) return;
  Inst& t = get(b);
  Args a = make_args(b, params, img, idx, n);
  a.X0 = const_cast<float*>(X0);
  a.dX0 = dX0;
  a.part = part;
  a.broken = broken_elu;
  const bool use_tc = tc && t.bwd_tc && t.h == b.h && t.w == b.w;
  int g = use_tc ? grid_for(b, n, t.gb_tc, tc::TPT) : grid_for(b, n, t.gb);
  if (use_tc)
    t.bwd_tc<<<g, tc::NTT, t.sb_tc, st>>>(a);
  else
    t.bwd<<<g, NT, t.sb, st>>>(a);
  CK(cudaGetLastError());
  k_part_fin<<<nblk(t.np, 256), 256, 0, st>>>(part, g, t.np, grads);
  CK(cudaGetLastError());
}

}  // namespace xnet
