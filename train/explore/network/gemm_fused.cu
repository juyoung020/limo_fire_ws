// 직접 짠 GEMM + 에필로그 융합. 인터페이스·뜻은 gemm_fused.cuh.
//
// 블록 = C 타일 BM×BN, K 를 BK 씩. 전역 fp32 → 레지스터(다음 타일 미리 읽기) → 공유 메모리(bf16 또는 fp32, 두 칸 번갈아) → 계산.
// 공유 메모리 배치는 전역 배치를 그대로 따른다(전치 복사 없음) — WMMA 조각의 row/col_major 로 맞춘다.
#include <mma.h>
#include <cuda_bf16.h>

#include <algorithm>
#include <cstdlib>
#include <map>
#include <stdexcept>
#include <type_traits>

#include "common.cuh"
#include "gemm_fused.cuh"

namespace xnet {

namespace {
using namespace nvcuda;


// R 행 × CC 열(열이 연속) 타일 한 칸을 float4 로. 범위 밖은 0.
template <int R, int CC, int NT>
struct TileLd {
  static constexpr int NV = R * CC / 4 / NT;
  float4 v[NV];
  __device__ __forceinline__ void load(const float* src, int ld, int r0, int c0, int rlim, int clim, bool vec) {
#pragma unroll
    for (int i = 0; i < NV; ++i) {
      int g = threadIdx.x + i * NT;
      int r = g / (CC / 4), c = (g % (CC / 4)) * 4;
      int gr = r0 + r, gc = c0 + c;
      float4 x = make_float4(0.f, 0.f, 0.f, 0.f);
      if (gr < rlim) {
        const float* p = src + (long)gr * ld + gc;
        if (vec && gc + 3 < clim) {
          x = __ldg(reinterpret_cast<const float4*>(p));
        } else {
          if (gc < clim) x.x = __ldg(p);
          if (gc + 1 < clim) x.y = __ldg(p + 1);
          if (gc + 2 < clim) x.z = __ldg(p + 2);
          if (gc + 3 < clim) x.w = __ldg(p + 3);
        }
      }
      v[i] = x;
    }
  }
  template <typename T, int LD>
  __device__ __forceinline__ void store(T* s) const {
#pragma unroll
    for (int i = 0; i < NV; ++i) {
      int g = threadIdx.x + i * NT;
      int r = g / (CC / 4), c = (g % (CC / 4)) * 4;
      T* d = s + r * LD + c;
      if constexpr (std::is_same_v<T, float>) {
        *reinterpret_cast<float4*>(d) = v[i];
      } else {
        __nv_bfloat162 a = __floats2bfloat162_rn(v[i].x, v[i].y), b = __floats2bfloat162_rn(v[i].z, v[i].w);
        uint2 u;
        u.x = *reinterpret_cast<uint32_t*>(&a);
        u.y = *reinterpret_cast<uint32_t*>(&b);
        *reinterpret_cast<uint2*>(d) = u;
      }
    }
  }
};

// bf16 전역 → 레지스터(uint2 = 4 개) → 공유 메모리 그대로. TileLd 와 같은 배치
template <int R, int CC, int NT>
struct TileLd16 {
  static constexpr int NV = R * CC / 4 / NT;
  uint2 v[NV];
  __device__ __forceinline__ void load(const unsigned short* src, int ld, int r0, int c0, int rlim, int clim, bool vec) {
#pragma unroll
    for (int i = 0; i < NV; ++i) {
      int g = threadIdx.x + i * NT;
      int r = g / (CC / 4), c = (g % (CC / 4)) * 4;
      int gr = r0 + r, gc = c0 + c;
      uint2 x = make_uint2(0u, 0u);
      if (gr < rlim) {
        const unsigned short* p = src + (long)gr * ld + gc;
        if (vec && gc + 3 < clim) {
          x = __ldg(reinterpret_cast<const uint2*>(p));
        } else {
          unsigned short e[4] = {0, 0, 0, 0};
#pragma unroll
          for (int q = 0; q < 4; ++q)
            if (gc + q < clim) e[q] = __ldg(p + q);
          x.x = e[0] | ((unsigned)e[1] << 16);
          x.y = e[2] | ((unsigned)e[3] << 16);
        }
      }
      v[i] = x;
    }
  }
  template <typename T, int LD>
  __device__ __forceinline__ void store(T* s) const {
#pragma unroll
    for (int i = 0; i < NV; ++i) {
      int g = threadIdx.x + i * NT;
      int r = g / (CC / 4), c = (g % (CC / 4)) * 4;
      *reinterpret_cast<uint2*>(s + r * LD + c) = v[i];
    }
  }
};

__global__ void k_to_bf16(const float* x, __nv_bfloat16* y, long n) {
  long i = blockIdx.x * (long)blockDim.x + threadIdx.x;
  if (i < n) y[i] = __float2bfloat16_rn(x[i]);
}

__device__ __forceinline__ void epi_one(const FGemm& g, int r, int c, float v) {
  if (r >= g.M || c >= g.N) return;
  long o = (long)r * g.ldc + c;
  switch (g.epi) {
    case FEpi::Store: g.C[o] = v; break;
    case FEpi::Bias: g.C[o] = v + g.bias[c]; break;
    case FEpi::BiasElu: {
      float z = v + g.bias[c];
      g.C[o] = z > 0.f ? z : expm1f(z);
      break;
    }
    case FEpi::EluBack: {
      float a = g.aux[(long)r * g.ldaux + c];
      g.C[o] = v * ((a > 0.f || g.broken) ? 1.f : a + 1.f);
      break;
    }
    case FEpi::Accum: g.C[o] += v; break;
  }
}

// 블록 타일 BM×BN, 워프 WM×WN 개(워프 하나 = (BM/WM)×(BN/WN)). fp32 정확 모드는 64×64·워프 2×2 만.
template <int BM, int BN, int BK, int WM, int WN, bool TA, bool TB, bool LOWP, bool BH = false>
// split-K: blockIdx.z 가 K 타일 [z·kps, (z+1)·kps) 를 맡고, ws 가 있으면 에필로그 대신 날값을 ws[z][M][N] 에 쓴다(k_splitk_epi 가 z 순서대로 더함).
__global__ void __launch_bounds__(32 * WM * WN) k_fgemm(FGemm g, bool va, bool vb, int kps, float* ws) {
  constexpr int NT = 32 * WM * WN;
  constexpr int FM = BM / WM / 16, FN = BN / WN / 16;  // 워프 하나의 16×16 조각 수
  static_assert(LOWP || (BM == 64 && BN == 64 && NT == 128), "fp32 정확 모드 타일");
  using T = std::conditional_t<LOWP, __nv_bfloat16, float>;
  constexpr int PAD = LOWP ? 8 : 4;
  constexpr int AR = TA ? BK : BM, AC = TA ? BM : BK, ALD = AC + PAD;
  constexpr int BR = TB ? BN : BK, BC = TB ? BK : BN, BLD = BC + PAD;
  // 동적 공유 메모리 (BK 64 · 128×64 는 48 KB 를 넘는다)
  extern __shared__ __align__(128) unsigned char smem_raw[];
  constexpr int ASZ = AR * ALD, BSZ = BR * BLD;
  T(*As)[ASZ] = reinterpret_cast<T(*)[ASZ]>(smem_raw);
  T(*Bs)[BSZ] = reinterpret_cast<T(*)[BSZ]>(smem_raw + 2 * ASZ * sizeof(T));

  const int n0 = blockIdx.x * BN, m0 = blockIdx.y * BM;
  const int t0 = blockIdx.z * kps;
  const int nk = min((g.K + BK - 1) / BK, t0 + kps);
  float* bias_out = g.bias_grad;  // split-K 면 z 조각마다 ws 뒤쪽 [S][M] 에 날값
  const bool bg_acc = ws == nullptr;
  if (ws) {
    bias_out = ws + (long)gridDim.z * g.M * g.N + (long)blockIdx.z * g.M;
    g.C = ws + (long)blockIdx.z * g.M * g.N;
    g.ldc = g.N;
    g.epi = FEpi::Store;
  }
  TileLd<AR, AC, NT> la;
  std::conditional_t<BH, TileLd16<BR, BC, NT>, TileLd<BR, BC, NT>> lb;
  static_assert(!BH || (LOWP && !TA), "bf16 사본은 lowp·!ta 만");
  // 편향 기울기 (ta): 이 스레드가 맡는 열(m) 4 개는 모든 타일에서 같다 (NT 가 AC/4 의 배수)
  static_assert(!TA || NT % (AC / 4) == 0, "열 고정");
  const bool do_bg = TA && g.bias_grad != nullptr && blockIdx.x == 0;
  float bg[4] = {0.f, 0.f, 0.f, 0.f};

  auto load = [&](int t) {
    int k0 = t * BK;
    if (TA)
      la.load(g.A, g.lda, k0, m0, g.K, g.M, va);
    else
      la.load(g.A, g.lda, m0, k0, g.M, g.K, va);
    if constexpr (BH) {
      if (TB)
        lb.load(g.Bh, g.ldb, n0, k0, g.N, g.K, vb);
      else
        lb.load(g.Bh, g.ldb, k0, n0, g.K, g.N, vb);
    } else {
      if (TB)
        lb.load(g.B, g.ldb, n0, k0, g.N, g.K, vb);
      else
        lb.load(g.B, g.ldb, k0, n0, g.K, g.N, vb);
    }
    if (TA && do_bg) {
#pragma unroll
      for (int i = 0; i < TileLd<AR, AC, NT>::NV; ++i) {
        bg[0] += la.v[i].x;
        bg[1] += la.v[i].y;
        bg[2] += la.v[i].z;
        bg[3] += la.v[i].w;
      }
    }
  };

  // ---- 누산기
  const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
  const int wm = warp / WN, wn = warp % WN;
  constexpr int WTM = BM / WM, WTN = BN / WN;
  wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[FM][FN];
  float sacc[8][4];
  if constexpr (LOWP) {
#pragma unroll
    for (int i = 0; i < FM; ++i)
#pragma unroll
      for (int j = 0; j < FN; ++j) wmma::fill_fragment(acc[i][j], 0.f);
  } else {
#pragma unroll
    for (int i = 0; i < 8; ++i)
#pragma unroll
      for (int j = 0; j < 4; ++j) sacc[i][j] = 0.f;
  }

  load(t0);
  la.template store<T, ALD>(As[0]);
  lb.template store<T, BLD>(Bs[0]);
  __syncthreads();
  for (int t = t0; t < nk; ++t) {
    const int cb = (t - t0) & 1;
    if (t + 1 < nk) load(t + 1);
    const T* a_s = As[cb];
    const T* b_s = Bs[cb];
    if constexpr (LOWP) {
      using LA = std::conditional_t<TA, wmma::col_major, wmma::row_major>;
      using LB = std::conditional_t<TB, wmma::col_major, wmma::row_major>;
#pragma unroll
      for (int kk = 0; kk < BK; kk += 16) {
        wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, LA> fa[FM];
        wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, LB> fb[FN];
#pragma unroll
        for (int i = 0; i < FM; ++i) {
          int m = wm * WTM + i * 16;
          const T* p = TA ? a_s + kk * ALD + m : a_s + m * ALD + kk;
          wmma::load_matrix_sync(fa[i], p, ALD);
        }
#pragma unroll
        for (int j = 0; j < FN; ++j) {
          int n = wn * WTN + j * 16;
          const T* p = TB ? b_s + n * BLD + kk : b_s + kk * BLD + n;
          wmma::load_matrix_sync(fb[j], p, BLD);
        }
#pragma unroll
        for (int i = 0; i < FM; ++i)
#pragma unroll
          for (int j = 0; j < FN; ++j) wmma::mma_sync(acc[i][j], fa[i], fb[j], acc[i][j]);
      }
    } else {
      // 스레드 하나 = 행 8 × 열 4 (행 = (tid/16)·8 + i, 열 = (tid%16)·4 + j). 정확 모드라 속도보다 단순함.
      const int tr = (threadIdx.x / 16) * 8, tc = (threadIdx.x % 16) * 4;
#pragma unroll 4
      for (int k = 0; k < BK; ++k) {
        float a[8], b[4];
#pragma unroll
        for (int i = 0; i < 8; ++i) a[i] = TA ? a_s[k * ALD + tr + i] : a_s[(tr + i) * ALD + k];
#pragma unroll
        for (int j = 0; j < 4; ++j) b[j] = TB ? b_s[(tc + j) * BLD + k] : b_s[k * BLD + tc + j];
#pragma unroll
        for (int i = 0; i < 8; ++i)
#pragma unroll
          for (int j = 0; j < 4; ++j) sacc[i][j] = fmaf(a[i], b[j], sacc[i][j]);
      }
    }
    if (t + 1 < nk) {
      la.template store<T, ALD>(As[cb ^ 1]);
      lb.template store<T, BLD>(Bs[cb ^ 1]);
    }
    __syncthreads();
  }

  // ---- 에필로그
  if constexpr (LOWP) {
    // 본 루프 뒤라 As 칸을 워프별 16×16 fp32 임시 칸으로 다시 쓴다
    static_assert(2 * ASZ * sizeof(T) >= NT / 32 * 256 * sizeof(float), "임시 칸");
    float* s = reinterpret_cast<float*>(&As[0][0]) + warp * 256;
#pragma unroll
    for (int i = 0; i < FM; ++i)
#pragma unroll
      for (int j = 0; j < FN; ++j) {
        wmma::store_matrix_sync(s, acc[i][j], 16, wmma::mem_row_major);
        __syncwarp();
        int r0 = m0 + wm * WTM + i * 16, c0 = n0 + wn * WTN + j * 16;
#pragma unroll
        for (int e = lane; e < 256; e += 32) epi_one(g, r0 + e / 16, c0 + e % 16, s[e]);
        __syncwarp();
      }
  } else {
    const int tr = (threadIdx.x / 16) * 8, tc = (threadIdx.x % 16) * 4;
#pragma unroll
    for (int i = 0; i < 8; ++i)
#pragma unroll
      for (int j = 0; j < 4; ++j) epi_one(g, m0 + tr + i, n0 + tc + j, sacc[i][j]);
  }

  // ---- 편향 기울기: 같은 열을 맡은 스레드들의 부분합을 고정 순서로
  if (TA && do_bg) {
    static_assert(2 * BSZ * sizeof(T) >= NT * 4 * sizeof(float), "합 칸");
    auto red = reinterpret_cast<float(*)[4]>(&Bs[0][0]);  // Bs 칸을 다시 씀
#pragma unroll
    for (int q = 0; q < 4; ++q) red[threadIdx.x][q] = bg[q];
    __syncthreads();
    constexpr int CG = AC / 4;  // 열 묶음 수
    if (threadIdx.x < CG) {
#pragma unroll
      for (int q = 0; q < 4; ++q) {
        float s = 0.f;
        for (int r = 0; r < NT / CG; ++r) s += red[r * CG + threadIdx.x][q];
        int m = m0 + threadIdx.x * 4 + q;
        if (m < g.M) bias_out[m] = bg_acc ? bias_out[m] + s : s;
      }
    }
  }
}

// split-K 마무리: z 순서대로 더하고(결정적) 에필로그
__global__ void k_splitk_epi(FGemm g, int S, const float* ws) {
  long i = blockIdx.x * (long)blockDim.x + threadIdx.x;
  long MN = (long)g.M * g.N;
  if (g.bias_grad && i < g.M) {  // 편향 기울기 조각도 z 순서대로
    float b = 0.f;
    for (int z = 0; z < S; ++z) b += ws[S * MN + (long)z * g.M + i];
    g.bias_grad[i] += b;
  }
  if (i >= MN) return;
  float v = 0.f;
  for (int z = 0; z < S; ++z) v += ws[z * MN + i];
  epi_one(g, (int)(i / g.N), (int)(i % g.N), v);
}

// ---- 큰 dW 전용: cp.async 2 단 fp32 적재 → 공유 메모리 안에서 bf16 변환 → WMMA (블록 128×128, BK 16, 워프 4 개 64×64)
// 실측(gemm_bench --n 4096, 첫 층 dW): 레지스터 미리 읽기 판 1.46 → 1.29 ms. BK 32·워프 4 개(공유 81 KB, SM 당 1 블록)는 2.27, 워프 8 개 1.90 ms 로 느렸다.
// 레지스터 미리 읽기가 없어 128×128 에서도 레지스터가 넘치지 않고, 타일 t 를 계산하는 동안 t+1·t+2 가 들어온다.
// 원소마다 K 순서·mma 순서가 기존 dW 커널과 같다(타일 크기만 다름) → 결과 비트 같음.
__device__ __forceinline__ void cp16(void* smem, const void* gmem, int bytes) {
  unsigned sa = (unsigned)__cvta_generic_to_shared(smem);
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(sa), "l"(gmem), "r"(bytes));
}
__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;\n" ::); }
template <int N>
__device__ __forceinline__ void cp_wait() { asm volatile("cp.async.wait_group %0;\n" ::"n"(N)); }

constexpr int DW_BM = 128, DW_BN = 128, DW_BK = 16, DW_NT = 128, DW_LD = 136, DW_WN = 2, DW_FN = DW_BN / DW_WN / 16;  // 워프 2×2, 워프당 64×64. BK 16 이라 공유 메모리 41 KB → SM 당 블록 2 개
constexpr size_t DW_SMEM = (size_t)2 * 2 * DW_BK * 128 * sizeof(float) + (size_t)2 * DW_BK * DW_LD * sizeof(__nv_bfloat16);

__global__ void __launch_bounds__(DW_NT) k_dw_async(FGemm g) {
  extern __shared__ __align__(128) unsigned char smem_raw[];
  float* sA = reinterpret_cast<float*>(smem_raw);       // [2][BK][BM] fp32 (cp.async 목적지)
  float* sB = sA + 2 * DW_BK * DW_BM;                    // [2][BK][BN]
  __nv_bfloat16* bA = reinterpret_cast<__nv_bfloat16*>(sB + 2 * DW_BK * DW_BN);  // [BK][LD]
  __nv_bfloat16* bB = bA + DW_BK * DW_LD;
  const int tid = threadIdx.x;
  const int n0 = blockIdx.x * DW_BN, m0 = blockIdx.y * DW_BM;
  const int nk = (g.K + DW_BK - 1) / DW_BK;
  const bool do_bg = g.bias_grad != nullptr && blockIdx.x == 0;
  float bg[4] = {0.f, 0.f, 0.f, 0.f};

  auto issue = [&](int t, int st) {
    const int k0 = t * DW_BK;
#pragma unroll
    for (int i = 0; i < DW_BK * 32 / DW_NT; ++i) {
      int q = tid + i * DW_NT;
      int r = q / 32, c = (q % 32) * 4;
      int gr = k0 + r;
      int ga = m0 + c, gb = n0 + c;
      int ba = (gr < g.K && ga < g.M) ? min(4, g.M - ga) * 4 : 0;
      int bb = (gr < g.K && gb < g.N) ? min(4, g.N - gb) * 4 : 0;
      cp16(sA + st * DW_BK * DW_BM + r * DW_BM + c, ba ? g.A + (long)gr * g.lda + ga : g.A, ba);
      cp16(sB + st * DW_BK * DW_BN + r * DW_BN + c, bb ? g.B + (long)gr * g.ldb + gb : g.B, bb);
    }
  };

  const int warp = tid / 32, lane = tid % 32;
  const int wm = warp / DW_WN, wn = warp % DW_WN;
  wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[4][DW_FN];
#pragma unroll
  for (int i = 0; i < 4; ++i)
#pragma unroll
    for (int j = 0; j < DW_FN; ++j) wmma::fill_fragment(acc[i][j], 0.f);

  issue(0, 0);
  cp_commit();
  if (nk > 1) issue(1, 1);
  cp_commit();
  for (int t = 0; t < nk; ++t) {
    const int st = t & 1;
    cp_wait<1>();
    __syncthreads();  // 타일 t 도착, 앞 계산이 bA·bB 를 다 읽음
#pragma unroll
    for (int i = 0; i < DW_BK * 32 / DW_NT; ++i) {  // fp32 → bf16 (스레드마다 같은 열 → 편향 기울기 합 순서 고정)
      int q = tid + i * DW_NT;
      int r = q / 32, c = (q % 32) * 4;
      float4 a = *reinterpret_cast<const float4*>(sA + st * DW_BK * DW_BM + r * DW_BM + c);
      float4 b = *reinterpret_cast<const float4*>(sB + st * DW_BK * DW_BN + r * DW_BN + c);
      if (do_bg) {
        bg[0] += a.x;
        bg[1] += a.y;
        bg[2] += a.z;
        bg[3] += a.w;
      }
      __nv_bfloat162 a0 = __floats2bfloat162_rn(a.x, a.y), a1 = __floats2bfloat162_rn(a.z, a.w);
      __nv_bfloat162 b0 = __floats2bfloat162_rn(b.x, b.y), b1 = __floats2bfloat162_rn(b.z, b.w);
      uint2 ua, ub;
      ua.x = *reinterpret_cast<uint32_t*>(&a0), ua.y = *reinterpret_cast<uint32_t*>(&a1);
      ub.x = *reinterpret_cast<uint32_t*>(&b0), ub.y = *reinterpret_cast<uint32_t*>(&b1);
      *reinterpret_cast<uint2*>(bA + r * DW_LD + c) = ua;
      *reinterpret_cast<uint2*>(bB + r * DW_LD + c) = ub;
    }
    __syncthreads();  // bf16 타일 준비, 단 st 비었음
    if (t + 2 < nk) issue(t + 2, st);
    cp_commit();
#pragma unroll
    for (int kk = 0; kk < DW_BK; kk += 16) {
      wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::col_major> fa[4];
      wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::row_major> fb[DW_FN];
#pragma unroll
      for (int i = 0; i < 4; ++i) wmma::load_matrix_sync(fa[i], bA + kk * DW_LD + wm * 64 + i * 16, DW_LD);
#pragma unroll
      for (int j = 0; j < DW_FN; ++j) wmma::load_matrix_sync(fb[j], bB + kk * DW_LD + wn * (DW_BN / DW_WN) + j * 16, DW_LD);
#pragma unroll
      for (int i = 0; i < 4; ++i)
#pragma unroll
        for (int j = 0; j < DW_FN; ++j) wmma::mma_sync(acc[i][j], fa[i], fb[j], acc[i][j]);
    }
  }
  cp_wait<0>();
  __syncthreads();
  float* stg = sA + warp * 256;  // 적재 칸은 이제 비었음
#pragma unroll
  for (int i = 0; i < 4; ++i)
#pragma unroll
    for (int j = 0; j < DW_FN; ++j) {
      wmma::store_matrix_sync(stg, acc[i][j], 16, wmma::mem_row_major);
      __syncwarp();
      int r0 = m0 + wm * 64 + i * 16, c0 = n0 + wn * (DW_BN / DW_WN) + j * 16;
#pragma unroll
      for (int e = lane; e < 256; e += 32) epi_one(g, r0 + e / 16, c0 + e % 16, stg[e]);
      __syncwarp();
    }
  if (do_bg) {
    auto red = reinterpret_cast<float(*)[4]>(sB);
#pragma unroll
    for (int q = 0; q < 4; ++q) red[tid][q] = bg[q];
    __syncthreads();
    if (tid < 32) {
#pragma unroll
      for (int q = 0; q < 4; ++q) {
        float sum = 0.f;
        for (int r = 0; r < DW_NT / 32; ++r) sum += red[r * 32 + tid][q];
        int m = m0 + tid * 4 + q;
        if (m < g.M) g.bias_grad[m] += sum;
      }
    }
  }
}

// ---- 큰 정방향·dX 전용 (!ta, W 는 bf16 사본): A(fp32)는 cp.async 2 단 → 공유 메모리 안 bf16 변환, B(W bf16)는 cp.async 로 바로 3 칸 돌림.
// 블록 128×128, BK 16, 워프 4 개 64×64. K 순서·mma 순서가 기존 커널과 같아 결과 비트 같음.
constexpr int FX_ALD = DW_BK + 8;  // bf16 A [BM][24]
template <bool TB>
struct FxB {  // B bf16 타일: TB 면 [BN][BK] (K 연속), 아니면 [BK][BN]
  static constexpr int R = TB ? DW_BN : DW_BK, C = TB ? DW_BK : DW_BN, LD = C + 8, SZ = R * LD;
};
template <bool TB>
constexpr size_t fx_smem() {
  return (size_t)2 * DW_BM * DW_BK * sizeof(float) + (size_t)DW_BM * FX_ALD * 2 + (size_t)3 * FxB<TB>::SZ * 2;
}

template <bool TB>
__global__ void __launch_bounds__(DW_NT) k_fx_async(FGemm g) {
  using BL = FxB<TB>;
  extern __shared__ __align__(128) unsigned char smem_raw[];
  float* sA = reinterpret_cast<float*>(smem_raw);                                // [2][BM][BK] fp32
  __nv_bfloat16* bA = reinterpret_cast<__nv_bfloat16*>(sA + 2 * DW_BM * DW_BK);  // [BM][ALD]
  __nv_bfloat16* bB = bA + DW_BM * FX_ALD;                                       // [3][R][LD]
  const unsigned short* Bh = g.Bh;
  const int tid = threadIdx.x;
  const int n0 = blockIdx.x * DW_BN, m0 = blockIdx.y * DW_BM;
  const int nk = (g.K + DW_BK - 1) / DW_BK;

  auto issue = [&](int t) {
    const int k0 = t * DW_BK, sa = t & 1, sb = t % 3;
    // A: [BM][BK] fp32 = 128 행 × 4 조각(16 B)
#pragma unroll
    for (int i = 0; i < DW_BM * DW_BK / 4 / DW_NT; ++i) {
      int q = tid + i * DW_NT;
      int r = q / (DW_BK / 4), c = (q % (DW_BK / 4)) * 4;
      int gm = m0 + r, gk = k0 + c;
      int b = (gm < g.M && gk < g.K) ? min(4, g.K - gk) * 4 : 0;
      cp16(sA + sa * DW_BM * DW_BK + r * DW_BK + c, b ? g.A + (long)gm * g.lda + gk : g.A, b);
    }
    // B: bf16, 16 B = 8 원소
#pragma unroll
    for (int i = 0; i < BL::R * BL::C / 8 / DW_NT; ++i) {
      int q = tid + i * DW_NT;
      int r = q / (BL::C / 8), c = (q % (BL::C / 8)) * 8;
      int gr = (TB ? n0 : k0) + r, gc = (TB ? k0 : n0) + c;
      int rl = TB ? g.N : g.K, cl = TB ? g.K : g.N;
      int b = (gr < rl && gc < cl) ? min(8, cl - gc) * 2 : 0;
      cp16(bB + sb * BL::SZ + r * BL::LD + c, b ? Bh + (long)gr * g.ldb + gc : Bh, b);
    }
  };

  const int warp = tid / 32, lane = tid % 32;
  const int wm = warp / DW_WN, wn = warp % DW_WN;
  wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[4][DW_FN];
#pragma unroll
  for (int i = 0; i < 4; ++i)
#pragma unroll
    for (int j = 0; j < DW_FN; ++j) wmma::fill_fragment(acc[i][j], 0.f);

  issue(0);
  cp_commit();
  if (nk > 1) issue(1);
  cp_commit();
  for (int t = 0; t < nk; ++t) {
    cp_wait<1>();
    __syncthreads();
#pragma unroll
    for (int i = 0; i < DW_BM * DW_BK / 4 / DW_NT; ++i) {
      int q = tid + i * DW_NT;
      int r = q / (DW_BK / 4), c = (q % (DW_BK / 4)) * 4;
      float4 a = *reinterpret_cast<const float4*>(sA + (t & 1) * DW_BM * DW_BK + r * DW_BK + c);
      __nv_bfloat162 a0 = __floats2bfloat162_rn(a.x, a.y), a1 = __floats2bfloat162_rn(a.z, a.w);
      uint2 u;
      u.x = *reinterpret_cast<uint32_t*>(&a0), u.y = *reinterpret_cast<uint32_t*>(&a1);
      *reinterpret_cast<uint2*>(bA + r * FX_ALD + c) = u;
    }
    __syncthreads();
    if (t + 2 < nk) issue(t + 2);
    cp_commit();
    const __nv_bfloat16* b_s = bB + (t % 3) * BL::SZ;
    using LB = std::conditional_t<TB, wmma::col_major, wmma::row_major>;
    wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> fa[4];
    wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, LB> fb[DW_FN];
#pragma unroll
    for (int i = 0; i < 4; ++i) wmma::load_matrix_sync(fa[i], bA + (wm * 64 + i * 16) * FX_ALD, FX_ALD);
#pragma unroll
    for (int j = 0; j < DW_FN; ++j) {
      int n = wn * (DW_BN / DW_WN) + j * 16;
      wmma::load_matrix_sync(fb[j], TB ? b_s + n * BL::LD : b_s + n, BL::LD);
    }
#pragma unroll
    for (int i = 0; i < 4; ++i)
#pragma unroll
      for (int j = 0; j < DW_FN; ++j) wmma::mma_sync(acc[i][j], fa[i], fb[j], acc[i][j]);
  }
  cp_wait<0>();
  __syncthreads();
  float* stg = sA + warp * 256;
#pragma unroll
  for (int i = 0; i < 4; ++i)
#pragma unroll
    for (int j = 0; j < DW_FN; ++j) {
      wmma::store_matrix_sync(stg, acc[i][j], 16, wmma::mem_row_major);
      __syncwarp();
      int r0 = m0 + wm * 64 + i * 16, c0 = n0 + wn * (DW_BN / DW_WN) + j * 16;
#pragma unroll
      for (int e = lane; e < 256; e += 32) epi_one(g, r0 + e / 16, c0 + e % 16, stg[e]);
      __syncwarp();
    }
}

bool aligned16(const void* p) { return (reinterpret_cast<uintptr_t>(p) & 15) == 0; }

// split-K 작업 공간: 스트림마다 따로 (actor·critic 두 스트림)
float* splitk_ws(cudaStream_t st, long n) {
  struct W {
    float* p = nullptr;
    long cap = 0;
  };
  static std::map<cudaStream_t, W> ws;
  W& w = ws[st];
  if (n > w.cap) {
    if (w.p) CK(cudaFree(w.p));
    CK(cudaMalloc(&w.p, n * sizeof(float)));
    w.cap = n;
  }
  return w.p;
}

template <int BM, int BN, int BK, int WM, int WN, bool LOWP>
void launch(const FGemm& g, cudaStream_t st, int S = 1) {
  constexpr int NT = 32 * WM * WN;
  int nkt = (g.K + BK - 1) / BK;
  int kps = (nkt + S - 1) / S;
  S = (nkt + kps - 1) / kps;
  float* ws = S > 1 ? splitk_ws(st, (long)S * g.M * g.N + (long)S * g.M) : nullptr;
  // 블록 순서(M 타일 먼저 vs N 타일 먼저)는 실측(nsys, 학습 중)에서 차이 없음 — dW 735 vs 730 µs
  dim3 grid((g.N + BN - 1) / BN, (g.M + BM - 1) / BM, S);
  bool va = g.lda % 4 == 0 && aligned16(g.A), vb = g.ldb % 4 == 0 && aligned16(g.B);
  using T = std::conditional_t<LOWP, __nv_bfloat16, float>;
  constexpr int PAD = LOWP ? 8 : 4;
  // 공유 메모리 크기: A·B 두 칸씩 (배치는 ta·tb 에 따라 다르지만 크기는 BM·BK, BN·BK 로 같음 — 큰 쪽으로)
  constexpr size_t SM = 2 * sizeof(T) * ((size_t)std::max(BM * (BK + PAD), BK * (BM + PAD)) + std::max(BN * (BK + PAD), BK * (BN + PAD)));
  auto setattr = [](auto kern) {  // 48 KB 넘는 동적 공유 메모리 허용 (커널마다; 세 커널이 같은 함수 포인터 형이라 static 으로 한 번만 하면 안 됨)
    if (SM > 48 * 1024) CK(cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)SM));
  };
  const bool bh = LOWP && g.Bh != nullptr;
  if (bh) vb = g.ldb % 4 == 0 && (reinterpret_cast<uintptr_t>(g.Bh) & 7) == 0;
  if (!g.ta && g.tb && bh)
    setattr(k_fgemm<BM, BN, BK, WM, WN, false, true, LOWP, LOWP>), k_fgemm<BM, BN, BK, WM, WN, false, true, LOWP, LOWP><<<grid, NT, SM, st>>>(g, va, vb, kps, ws);
  else if (!g.ta && !g.tb && bh)
    setattr(k_fgemm<BM, BN, BK, WM, WN, false, false, LOWP, LOWP>), k_fgemm<BM, BN, BK, WM, WN, false, false, LOWP, LOWP><<<grid, NT, SM, st>>>(g, va, vb, kps, ws);
  else if (!g.ta && g.tb)
    setattr(k_fgemm<BM, BN, BK, WM, WN, false, true, LOWP>), k_fgemm<BM, BN, BK, WM, WN, false, true, LOWP><<<grid, NT, SM, st>>>(g, va, vb, kps, ws);
  else if (!g.ta && !g.tb)
    setattr(k_fgemm<BM, BN, BK, WM, WN, false, false, LOWP>), k_fgemm<BM, BN, BK, WM, WN, false, false, LOWP><<<grid, NT, SM, st>>>(g, va, vb, kps, ws);
  else if (g.ta && !g.tb)
    setattr(k_fgemm<BM, BN, BK, WM, WN, true, false, LOWP>), k_fgemm<BM, BN, BK, WM, WN, true, false, LOWP><<<grid, NT, SM, st>>>(g, va, vb, kps, ws);
  else
    throw std::runtime_error("fgemm: ta && tb 조합은 없음");
  CK(cudaGetLastError());
  if (S > 1) {
    long MN = (long)g.M * g.N;
    k_splitk_epi<<<(int)((std::max<long>(MN, g.M) + 255) / 256), 256, 0, st>>>(g, S, ws);
    CK(cudaGetLastError());
  }
}
int g_tile = -1;  // TODO(제거): 타일 고르기 실험용 (XNET_TILE)
}  // namespace

void to_bf16(const float* x, unsigned short* y, long n, cudaStream_t st) {
  k_to_bf16<<<(int)((n + 255) / 256), 256, 0, st>>>(x, reinterpret_cast<__nv_bfloat16*>(y), n);
  CK(cudaGetLastError());
}

void fgemm(const FGemm& g, bool lowp, cudaStream_t st) {
  if (g.M <= 0 || g.N <= 0) return;
  if (g.bias_grad && !g.ta) throw std::runtime_error("fgemm: bias_grad 는 ta 일 때만");
  if (g.bias_grad && g.epi != FEpi::Accum) throw std::runtime_error("fgemm: bias_grad 는 Accum(dW) 과 같이");
  if (!lowp) return launch<64, 64, 32, 2, 2, false>(g, st);
  if (g_tile < 0) g_tile = std::getenv("XNET_TILE") ? std::atoi(std::getenv("XNET_TILE")) : 0;
  // 2026-11-04 gemm_bench 실측(RTX 3060, 덩이 1024, 이벤트 최솟값): 128×64 블록이 4 물결(SM 28 × 4) 이상 나오면 128×64, 아니면 64×64.
  // 첫 층 dX·dW 는 128×64 (0.40·0.38 vs 64×64 0.57·0.56 ms). 첫 층 정방향(1024×512)은 128×64 로 64 블록뿐이라
  // split-K 4 (256 블록, z 순서로 더함): 0.39 ms (64×64 0.43, 128×64 0.50, split 2 0.41). 128×128 타일·BK 64 는 더 느렸다(0.68 ms 등).
  long big = (long)((g.M + 127) / 128) * ((g.N + 63) / 64);
  static int split_env = std::getenv("XNET_SPLIT") ? std::atoi(std::getenv("XNET_SPLIT")) : -1;  // TODO(제거): 실험용 (XNET_SPLIT)
  int t = g_tile ? g_tile : (big >= 112 ? 3 : 1);
  if (big < 112 && g.K >= 1024) {  // 블록이 모자라고 K 가 길면 split-K: 첫 층 정방향(덩이 1024), 작은 층 dW(K = 샘플 수)
    // 정방향은 128×64(실측 0.39 ms), dW 는 64×64 블록으로 나눈다. 블록 × 조각 ≥ 224 (SM 28 × 8), 조각마다 K 타일 4 개 이상
    const bool wide = !g.bias_grad;
    long blocks = wide ? big : (long)((g.M + 63) / 64) * ((g.N + 63) / 64);
    int S = split_env > 0 ? split_env : (int)std::min<long>(16, (224 + blocks - 1) / blocks);
    S = std::min(S, std::max(1, g.K / (32 * 4)));
    if (S > 1) return wide ? launch<128, 64, 32, 2, 2, true>(g, st, S) : launch<64, 64, 32, 2, 2, true>(g, st, S);
  }
  // 큰 dW(ta)는 128×128·워프 2×2(워프당 64×64): 덩이 4096 첫 층 dW 1.63 → 1.45 ms. 정방향·dX 는 이 타일이 더 느림(1.31 → 2.02, 1.51 → 2.75 ms)
  static int dw_async = std::getenv("XNET_DWASYNC") ? std::atoi(std::getenv("XNET_DWASYNC")) : 1;  // TODO(제거): A/B 실험용 끄개
  if (t == 3 && g.ta && g_tile == 0 && dw_async && !g.tb && g.epi == FEpi::Accum && g.lda % 4 == 0 && g.ldb % 4 == 0 &&
      aligned16(g.A) && aligned16(g.B)) {
    CK(cudaFuncSetAttribute(k_dw_async, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)DW_SMEM));
    dim3 grid((g.N + DW_BN - 1) / DW_BN, (g.M + DW_BM - 1) / DW_BM);
    k_dw_async<<<grid, DW_NT, DW_SMEM, st>>>(g);
    CK(cudaGetLastError());
    return;
  }
  if (t == 3 && g.ta && g_tile == 0) return launch<128, 128, 32, 2, 2, true>(g, st);
  // W bf16 사본을 읽는 큰 정방향·dX 는 128×128 (B 레지스터가 반이라 넘침 없음): 덩이 4096 첫 층 정방향 1.29 → 1.18, dX 1.43 → 1.30 ms.
  // 작은 층(512→256 등)은 128×64 가 빨라 K·N 이 1024 이상일 때만
  long big2 = (long)((g.M + 127) / 128) * ((g.N + 127) / 128);
  if (t == 3 && g_tile == 0 && g.Bh && !g.ta && big2 >= 112 && (g.K >= 1024 || g.N >= 1024)) {
    static int fx_async = std::getenv("XNET_FXASYNC") ? std::atoi(std::getenv("XNET_FXASYNC")) : 1;  // TODO(제거): A/B 실험용 끄개
    // dX(!tb) 만: 1.31 → 1.19 ms. 정방향(tb)은 1.20 → 1.22 ms 로 이득 없어 기존 커널
    if (fx_async && !g.tb && g.lda % 4 == 0 && aligned16(g.A) && g.ldb % 8 == 0 && aligned16(g.Bh)) {
      dim3 grid((g.N + DW_BN - 1) / DW_BN, (g.M + DW_BM - 1) / DW_BM);
      if (g.tb) {
        CK(cudaFuncSetAttribute(k_fx_async<true>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)fx_smem<true>()));
        k_fx_async<true><<<grid, DW_NT, fx_smem<true>(), st>>>(g);
      } else {
        CK(cudaFuncSetAttribute(k_fx_async<false>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)fx_smem<false>()));
        k_fx_async<false><<<grid, DW_NT, fx_smem<false>(), st>>>(g);
      }
      CK(cudaGetLastError());
      return;
    }
    return launch<128, 128, 32, 2, 2, true>(g, st);
  }
  if (t == 3) return launch<128, 64, 32, 2, 2, true>(g, st);
  launch<64, 64, 32, 2, 2, true>(g, st);
}

}  // namespace xnet
