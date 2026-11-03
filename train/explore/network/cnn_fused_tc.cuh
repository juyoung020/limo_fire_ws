// 합성곱 가지 융합 커널의 텐서 코어 판 (bf16 입력, fp32 누산: mma.sync m16n8k16). cnn_fused.cu 안(익명 이름공간)에서만 include.
//
// 구조는 fp32 판과 같다(조각 타일, 재계산 역방향, 블록 부분합). 다른 점:
//  - 공유 메모리 활성값·가중치는 bf16. u8 입력(0~255)은 bf16 에 정확히 들어가고 /255 는 누산 뒤에 곱한다.
//  - GEMM 은 워프마다 16×16 조각을 ldmatrix 로 읽어 mma. 전치가 필요한 곱(dW = Zᵀ·X, dA = Z·W)은 ldmatrix.trans.
//  - 가중치 기울기는 워프가 맡은 출력 조각의 누산기를 블록 수명 내내 레지스터에 둔다. 편향 기울기는 스레드별 고정 행 합.
//  - 공유 메모리 배치: 활성값은 "칸"(가장 잘게 쪼갠 행, 너비 = 채널) 단위, 칸 간격 SS = 채널 + 8(16 B 덧댐),
//    칸 4 개(위 층 한 행)마다 SK 를 더 띄운다 → ldmatrix 8 행이 서로 다른 뱅크 묶음에 걸리게.
#pragma once

// (cuda_bf16.h·algorithm·type_traits 는 cnn_fused.cu 맨 위에서)

namespace tc {

constexpr int TPT = 32;   // 블록 타일 = 조각 수 (텐서 코어 판)
constexpr int NTT = 512;  // 블록 스레드 수 (텐서 코어 판)

typedef __nv_bfloat16 bf;


__device__ __forceinline__ uint32_t sa(const void* p) { return (uint32_t)__cvta_generic_to_shared(p); }
__device__ __forceinline__ void ldsm4(uint32_t (&r)[4], const void* p) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
               : "r"(sa(p)));
}
__device__ __forceinline__ void ldsm4t(uint32_t (&r)[4], const void* p) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
               : "r"(sa(p)));
}
__device__ __forceinline__ void mma(float (&c)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

// 워프 하나의 출력 단위 (MT 개 16 행 조각 × NP 쌍의 16 열) 를 k0 부터 k 걸음 하나(16) 누산.
// A_T: A 를 저장 S[k][m] 에서 전치로 (aptr(k, m)), 아니면 aptr(m, k). B_T: 저장 S[k][n] (bptr(k, n)), 아니면 S[n][k] (bptr(n, k)).
template <int MT, int NP, bool A_T, bool B_T, class AP, class BP>
__device__ __forceinline__ void kstep(float (&acc)[MT][2 * NP][4], int m0, int n0, int k0, AP aptr, BP bptr) {
  const int l = threadIdx.x & 31, i = l & 7, j = l >> 3;
  uint32_t a[MT][4], b[NP][4];
#pragma unroll
  for (int t = 0; t < MT; ++t) {
    const int mb = m0 + 16 * t;
    if (A_T)
      ldsm4t(a[t], aptr(k0 + 8 * (j >> 1) + i, mb + 8 * (j & 1)));
    else
      ldsm4(a[t], aptr(mb + i + 8 * (j & 1), k0 + 8 * (j >> 1)));
  }
#pragma unroll
  for (int u = 0; u < NP; ++u) {
    const int nb = n0 + 16 * u;
    if (B_T)
      ldsm4t(b[u], bptr(k0 + i + 8 * (j & 1), nb + 8 * (j >> 1)));
    else
      ldsm4(b[u], bptr(nb + i + 8 * (j >> 1), k0 + 8 * (j & 1)));
  }
#pragma unroll
  for (int t = 0; t < MT; ++t)
#pragma unroll
    for (int u = 0; u < NP; ++u) {
      mma(acc[t][2 * u], a[t], b[u][0], b[u][1]);
      mma(acc[t][2 * u + 1], a[t], b[u][2], b[u][3]);
    }
}

// 누산기 → epi(m, n, v0, v1) (열 n, n+1)
template <int MT, int NP, class Epi>
__device__ __forceinline__ void emit(const float (&acc)[MT][2 * NP][4], int m0, int n0, Epi epi) {
  const int l = threadIdx.x & 31, g = l >> 2, q = l & 3;
#pragma unroll
  for (int t = 0; t < MT; ++t)
#pragma unroll
    for (int s = 0; s < 2 * NP; ++s) {
      int m = m0 + 16 * t + g, n = n0 + 8 * s + 2 * q;
      epi(m, n, acc[t][s][0], acc[t][s][1]);
      epi(m + 8, n, acc[t][s][2], acc[t][s][3]);
    }
}

// 블록 GEMM: D[M][N] = Σ_k A·B, 단위를 워프에 돌려 맡김. 끝나면 epi.
template <int M, int N, int K, int MT, int NP, bool A_T, bool B_T, class AP, class BP, class Epi>
__device__ __forceinline__ void bgemm(AP aptr, BP bptr, Epi epi) {
  static_assert(M % (16 * MT) == 0 && N % (16 * NP) == 0 && K % 16 == 0, "bgemm 모양");
  constexpr int UM = M / (16 * MT), UN = N / (16 * NP), U = UM * UN;
  const int w = threadIdx.x >> 5;
  for (int u = w; u < U; u += NTT / 32) {
    const int m0 = (u / UN) * 16 * MT, n0 = (u % UN) * 16 * NP;
    float acc[MT][2 * NP][4];
#pragma unroll
    for (int t = 0; t < MT; ++t)
#pragma unroll
      for (int s = 0; s < 2 * NP; ++s)
#pragma unroll
        for (int e = 0; e < 4; ++e) acc[t][s][e] = 0.f;
#pragma unroll 4
    for (int k0 = 0; k0 < K; k0 += 16) kstep<MT, NP, A_T, B_T>(acc, m0, n0, k0, aptr, bptr);
    emit<MT, NP>(acc, m0, n0, epi);
  }
}

// 단위 수 = 워프 수인 블록 GEMM 을 둘로: compute 가 누산기를 레지스터에 두고, emit 은 나중에(동기화 뒤) 쓴다.
// 그 사이에 다른 곱(dW 누적)이 같은 단계에서 돌 수 있다 — 쓰기가 다른 스레드의 읽기와 겹칠 때 동기화 하나로 묶기 위함.
template <int M, int N, int K, int MT, int NP, bool A_T, bool B_T>
struct Held {
  static constexpr int UM = M / (16 * MT), UN = N / (16 * NP);
  static_assert(UM * UN == NTT / 32, "Held: 단위 수 = 워프 수");
  float acc[MT][2 * NP][4];
  int m0, n0;
  template <class AP, class BP>
  __device__ __forceinline__ void compute(AP aptr, BP bptr) {
    const int u = threadIdx.x >> 5;
    m0 = (u / UN) * 16 * MT;
    n0 = (u % UN) * 16 * NP;
#pragma unroll
    for (int t = 0; t < MT; ++t)
#pragma unroll
      for (int s2 = 0; s2 < 2 * NP; ++s2)
#pragma unroll
        for (int e = 0; e < 4; ++e) acc[t][s2][e] = 0.f;
#pragma unroll
    for (int k0 = 0; k0 < K; k0 += 16) kstep<MT, NP, A_T, B_T>(acc, m0, n0, k0, aptr, bptr);
  }
  template <class Epi>
  __device__ __forceinline__ void emit_to(Epi epi) const { emit<MT, NP>(acc, m0, n0, epi); }
};

// split-K 블록 GEMM: 단위 U 개 × k 묶음 KS 개 = 워프 수. 묶음마다 scr[ks][M][N] (fp32) 에 쓰고 묶음 순서대로 더해 epi(m, n, v).
// 끝에 __syncthreads 가 두 번 (scr 쓰기 뒤, 읽기 뒤는 부르는 쪽 다음 동기화가 맡음).
template <int M, int N, int K, int MT, int NP, class AP, class BP, class Epi>
__device__ __forceinline__ void bgemm_splitk(AP aptr, BP bptr, float* scr, Epi epi) {
  constexpr int UM = M / (16 * MT), UN = N / (16 * NP), U = UM * UN, KS = (NTT / 32) / U;
  static_assert(M % (16 * MT) == 0 && N % (16 * NP) == 0 && (NTT / 32) % U == 0, "split-K 모양");
  static_assert(K % (16 * KS) == 0, "split-K k");
  constexpr int KL = K / KS;
  const int w = threadIdx.x >> 5, u = w % U, ks = w / U;
  const int m0 = (u / UN) * 16 * MT, n0 = (u % UN) * 16 * NP;
  float acc[MT][2 * NP][4];
#pragma unroll
  for (int t = 0; t < MT; ++t)
#pragma unroll
    for (int s2 = 0; s2 < 2 * NP; ++s2)
#pragma unroll
      for (int e = 0; e < 4; ++e) acc[t][s2][e] = 0.f;
#pragma unroll
  for (int k0 = ks * KL; k0 < (ks + 1) * KL; k0 += 16) kstep<MT, NP, false, false>(acc, m0, n0, k0, aptr, bptr);
  float* sp = scr + ks * M * N;
  emit<MT, NP>(acc, m0, n0, [&](int m, int n, float v0, float v1) {
    *reinterpret_cast<float2*>(sp + m * N + n) = make_float2(v0, v1);
  });
  __syncthreads();
  for (int e = threadIdx.x; e < M * N; e += NTT) {
    float v = 0.f;
#pragma unroll
    for (int k = 0; k < KS; ++k) v += scr[k * M * N + e];
    epi(e / N, e % N, v);
  }
}

// 블록 수명 내내 쌓는 가중치 기울기 dW[M][N] = Σ_k Aᵀ…: 단위 U 개를 워프에 고정, 남는 워프는 k 걸음을 나눠 맡음(G 묶음).
template <int M, int N, int MT, int NP>
struct DWTc {
  static constexpr int UM = M / (16 * MT), UN = N / (16 * NP), U = UM * UN;
  static_assert(U <= NTT / 32 && (NTT / 32) % U == 0, "DWTc 단위");
  static constexpr int G = (NTT / 32) / U;
  float acc[MT][2 * NP][4];
  __device__ __forceinline__ void zero() {
#pragma unroll
    for (int t = 0; t < MT; ++t)
#pragma unroll
      for (int s = 0; s < 2 * NP; ++s)
#pragma unroll
        for (int e = 0; e < 4; ++e) acc[t][s][e] = 0.f;
  }
  template <int K, class AP, class BP>
  __device__ __forceinline__ void add(AP aptr, BP bptr) {
    const int w = threadIdx.x >> 5, u = w % U, grp = w / U;
    const int m0 = (u / UN) * 16 * MT, n0 = (u % UN) * 16 * NP;
#pragma unroll 2
    for (int k0 = 16 * grp; k0 < K; k0 += 16 * G) kstep<MT, NP, true, true>(acc, m0, n0, k0, aptr, bptr);
  }
  // 묶음별로 scratch [G][M][N] 에 쓴다 (호출 뒤 __syncthreads, 묶음 순서대로 더하기는 부르는 쪽)
  __device__ __forceinline__ void dump(float* scratch) const {
    const int w = threadIdx.x >> 5, u = w % U, grp = w / U;
    const int m0 = (u / UN) * 16 * MT, n0 = (u % UN) * 16 * NP;
    float* s = scratch + grp * M * N;
    emit<MT, NP>(acc, m0, n0, [&](int m, int n, float v0, float v1) {
      s[m * N + n] = v0;
      s[m * N + n + 1] = v1;
    });
  }
};

// 편향 기울기: 스레드 t 가 열 t % C, 행 묶음 t / C 를 블록 수명 내내 더함
template <int C>
struct DBTc {
  static constexpr int G = NTT / C;
  float s = 0.f;
  template <int R, class ZP>
  __device__ __forceinline__ void add(ZP zptr) {
    const int c = threadIdx.x % C, grp = threadIdx.x / C;
    if (grp >= G) return;
    for (int r = grp; r < R; r += G) s += __bfloat162float(*zptr(r, c));
  }
};

__device__ __forceinline__ void st2(bf* p, float v0, float v1) {
  *reinterpret_cast<__nv_bfloat162*>(p) = __floats2bfloat162_rn(v0, v1);
}
__device__ __forceinline__ float2 ld2(const bf* p) {
  return __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(p));
}

template <int CIN, int C0, int C1, int C2>
struct TcShape {
  static_assert(C0 % 16 == 0 && C1 % 16 == 0 && C2 % 16 == 0, "텐서 코어 판 채널은 16 의 배수");
  static constexpr int K0 = 4 * CIN, K0P = (K0 + 15) / 16 * 16, K1 = 4 * C0, K2 = 4 * C1;
  static constexpr bool ONES = K0 < K0P;  // 입력 덧댄 열 하나를 1 로 두어 db0 를 dW0 mma 로
  static constexpr int SSX = K0P + 8, SS0 = C0 + 8, SS1 = C1 + 8, SSZ2 = C2 + 8, SK = 8;
  static constexpr int SSW0 = K0P + 8, SSW1 = K1 + 8, SSW2 = K2 + 8;
  // bf16 원소 수
  static constexpr int NX = TPT * 16 * SSX;
  static constexpr int NA0 = TPT * 16 * SS0 + TPT * 4 * SK;
  static constexpr int NA1 = TPT * 4 * SS1 + TPT * SK;
  static constexpr int NZ2 = TPT * SSZ2;
  static constexpr int NW0 = C0 * SSW0, NW1 = C1 * SSW1, NW2 = C2 * SSW2;
  static constexpr int NBIAS = C0 + C1 + C2;  // float
  __device__ static int seg0(int r0) { return r0 * SS0 + (r0 >> 2) * SK; }  // 층 0 출력 칸 (r0)
  __device__ static int seg1(int r1) { return r1 * SS1 + (r1 >> 2) * SK; }  // 층 1 출력 칸 (r1)
};

// 가중치 (fp32, [cout][K]) → bf16 [cout][SS] (K 뒤는 0)
template <int C, int K, int KP, int SS>
__device__ __forceinline__ void load_w_bf(const float* __restrict__ W, bf* Ws) {
  for (int e = threadIdx.x; e < C * KP; e += NTT) {
    int c = e / KP, k = e % KP;
    Ws[c * SS + k] = __float2bfloat16_rn(k < K ? W[c * K + k] : 0.f);
  }
}

// 타일 입력을 레지스터로 미리 읽어 두고(다음 타일 계산 동안 전역 지연을 숨김) 공유 메모리에 쓴다.
// 반 줄 하나 = (조각, 조각 안 행 y, 왼·오른 4 픽셀): 채널마다 4 바이트를 읽어, 이웃 두 픽셀(x 짝)의 모든 채널 = Xs 행 안 연속 2·CIN 원소를 32 비트씩 쓴다.
// H·W 는 컴파일 때 상수 (조각 번호 나눗셈이 싸게).
template <int CIN, int SSX, bool DZ, int C2, int H, int W>
struct Prefetch {
  static constexpr int WP = W / 8, PPS = (H / 8) * WP;
  static constexpr int NL = TPT * 16;                        // 타일의 반 줄 수 (조각, 행 y, 왼·오른 4 픽셀)
  static constexpr int NI = (NL + NTT - 1) / NTT;            // 스레드마다 반 줄 수
  static constexpr int NZ = (TPT * C2 / 2 + NTT - 1) / NTT;  // 스레드마다 (dX0, X0) 쌍 수
  uint32_t v[NI][CIN];
  float2 dx[DZ ? NZ : 1], xo[DZ ? NZ : 1];
  __device__ static long oidx(const Args& a, long g, int c) { return (g / PPS) * a.feat + a.flat_off + (g % PPS) * C2 + c; }
  __device__ __forceinline__ void fetch(const Args& a, long g0, long total) {
#pragma unroll
    for (int i = 0; i < NI; ++i) {
      int it = threadIdx.x + i * NTT;
      int hx = it & 1, y = (it >> 1) & 7, p = it >> 4;
      long g = g0 + p;
      bool ok = (NL % NTT == 0 || it < NL) && g < total;
      const uint8_t* src = nullptr;
      if (ok) {
        long s = g / PPS;
        int pix = (int)(g % PPS), py = pix / WP, px = pix % WP;
        if (a.idx) s = __ldg(a.idx + s);
        src = a.img + s * a.img_stride + a.img_off + (py * 8 + y) * W + px * 8 + hx * 4;
      }
#pragma unroll
      for (int ci = 0; ci < CIN; ++ci) v[i][ci] = ok ? __ldg(reinterpret_cast<const uint32_t*>(src + ci * H * W)) : 0u;
    }
    if (DZ)
#pragma unroll
      for (int i = 0; i < NZ; ++i) {
        int e = threadIdx.x + i * NTT;
        int r = e / (C2 / 2), c = 2 * (e % (C2 / 2));
        long g = g0 + r;
        dx[i] = xo[i] = make_float2(0.f, 0.f);
        if (e < TPT * C2 / 2 && g < total) {
          long ix = oidx(a, g, c);
          dx[i] = __ldg(reinterpret_cast<const float2*>(a.dX0 + ix));
          xo[i] = __ldg(reinterpret_cast<const float2*>(a.X0 + ix));
        }
      }
  }
  // u8 → bf16 비트: 2^23 + v 의 float 에서 2^23 을 빼면 v (I2F 없이), 0~255 는 위 16 비트만으로 정확
  __device__ static uint32_t bfbits(uint32_t byte) { return __float_as_uint(__uint_as_float(0x4B000000u | byte) - 8388608.f) >> 16; }
  __device__ __forceinline__ void store(bf* Xs, bf* Z2, int ssz2, bool broken) const {
#pragma unroll
    for (int i = 0; i < NI; ++i) {
      int it = threadIdx.x + i * NTT;
      if (NL % NTT != 0 && it >= NL) break;
      int hx = it & 1, y = (it >> 1) & 7, p = it >> 4;
      const int q1y = (y >> 2) & 1, c1y = (y >> 1) & 1, c0y = y & 1;
#pragma unroll
      for (int xh = 0; xh < 2; ++xh) {  // 반 줄 안 x 짝 (x = 4·hx + 2·xh, +1)
        const int q1 = q1y * 2 + hx, c1 = c1y * 2 + xh;
        uint32_t* dst = reinterpret_cast<uint32_t*>(Xs + (p * 16 + q1 * 4 + c1) * SSX + c0y * 2 * CIN);
        uint32_t e[2 * CIN];  // [x 짝 안 x][채널]
#pragma unroll
        for (int ci = 0; ci < CIN; ++ci) {
          int sh = 16 * xh;
          e[ci] = bfbits((v[i][ci] >> sh) & 255u);
          e[CIN + ci] = bfbits((v[i][ci] >> (sh + 8)) & 255u);
        }
#pragma unroll
        for (int h = 0; h < CIN; ++h) dst[h] = e[2 * h] | (e[2 * h + 1] << 16);
      }
    }
    if (DZ)
#pragma unroll
      for (int i = 0; i < NZ; ++i) {  // Z2 = dX0 · ELU'(출력)
        int e = threadIdx.x + i * NTT;
        if (e >= TPT * C2 / 2) break;
        int r = e / (C2 / 2), c = 2 * (e % (C2 / 2));
        st2(Z2 + r * ssz2 + c, dx[i].x * elu_d(xo[i].x, broken), dx[i].y * elu_d(xo[i].y, broken));
      }
  }
};

// 층 0·1 정방향: Xs → A0 → A1 (bf16, ELU 뒤)
template <class S, int C0, int C1>
__device__ __forceinline__ void fwd01_tc(const bf* Xs, const bf* W0s, const bf* W1s, const float* b0, const float* b1,
                                         bf* A0, bf* A1) {
  constexpr float inv = 1.f / 255.f;
  bgemm<TPT * 16, C0, S::K0P, 2, 1, false, false>(
      [&](int m, int k) { return Xs + m * S::SSX + k; }, [&](int n, int k) { return W0s + n * S::SSW0 + k; },
      [&](int m, int n, float v0, float v1) {
        st2(A0 + S::seg0(m) + n, elu_f(fmaf(v0, inv, b0[n])), elu_f(fmaf(v1, inv, b0[n + 1])));
      });
  __syncthreads();
  bgemm<TPT * 4, C1, S::K1, 1, 1, false, false>(
      [&](int m, int k) { return A0 + S::seg0(m * 4 + k / C0) + k % C0; },
      [&](int n, int k) { return W1s + n * S::SSW1 + k; },
      [&](int m, int n, float v0, float v1) { st2(A1 + S::seg1(m) + n, elu_f(v0 + b1[n]), elu_f(v1 + b1[n + 1])); });
  __syncthreads();
}

template <int CIN, int C0, int C1, int C2, int H, int W>
__global__ void __launch_bounds__(NTT) k_cnn_fwd_tc(Args a) {
  using S = TcShape<CIN, C0, C1, C2>;
  using P = Shape<CIN, C0, C1, C2>;
  extern __shared__ float4 smem4[];
  float* bias = reinterpret_cast<float*>(smem4);
  bf* W0s = reinterpret_cast<bf*>(bias + r4(S::NBIAS));
  bf* W1s = W0s + S::NW0;
  bf* W2s = W1s + S::NW1;
  bf* Xs = W2s + S::NW2;
  bf* A0 = Xs + S::NX;
  bf* A1 = A0 + S::NA0;
  float *b0 = bias, *b1 = bias + C0, *b2 = bias + C0 + C1;
  load_v(a.P + P::OB0, b0, C0);
  load_v(a.P + P::OB1, b1, C1);
  load_v(a.P + P::OB2, b2, C2);
  load_w_bf<C0, S::K0, S::K0P, S::SSW0>(a.P + P::OW0, W0s);
  load_w_bf<C1, S::K1, S::K1, S::SSW1>(a.P + P::OW1, W1s);
  load_w_bf<C2, S::K2, S::K2, S::SSW2>(a.P + P::OW2, W2s);
  for (int e = threadIdx.x; e < S::NX; e += NTT) Xs[e] = __float2bfloat16_rn(0.f);  // 덧댄 열 0
  __syncthreads();
  const long total = (long)a.n * ((H / 8) * (W / 8));
  const long ntiles = (total + TPT - 1) / TPT;
  using PF = Prefetch<CIN, S::SSX, false, C2, H, W>;
  PF pf;
  pf.fetch(a, (long)blockIdx.x * TPT, total);
  for (long t = blockIdx.x; t < ntiles; t += gridDim.x) {
    const long g0 = t * TPT;
    pf.store(Xs, nullptr, 0, false);
    __syncthreads();
    pf.fetch(a, (t + gridDim.x) * TPT, total);  // 다음 타일 (범위 밖이면 0)
    fwd01_tc<S, C0, C1>(Xs, W0s, W1s, b0, b1, A0, A1);
    // 층 2: 출력이 작아(TPT×C2) 단위가 워프 수보다 적으므로 k 를 나눠(split-K) 모든 워프가 맡고, A0 자리(이제 안 씀)에 부분합
    float* scr = reinterpret_cast<float*>(A0);
    bgemm_splitk<TPT, C2, S::K2, 1, 1>(
        [&](int m, int k) { return A1 + S::seg1(m * 4 + k / C1) + k % C1; },
        [&](int n, int k) { return W2s + n * S::SSW2 + k; }, scr, [&](int m, int n, float v) {
          long g = g0 + m;
          if (g < total) a.X0[PF::oidx(a, g, n)] = elu_f(v + b2[n]);
        });
  }
}

template <int CIN, int C0, int C1, int C2, int H, int W>
__global__ void __launch_bounds__(NTT) k_cnn_bwd_tc(Args a) {
  using S = TcShape<CIN, C0, C1, C2>;
  using P = Shape<CIN, C0, C1, C2>;
  extern __shared__ float4 smem4[];
  float* bias = reinterpret_cast<float*>(smem4);
  bf* W0s = reinterpret_cast<bf*>(bias + r4(S::NBIAS));
  bf* W1s = W0s + S::NW0;
  bf* W2s = W1s + S::NW1;
  bf* Xs = W2s + S::NW2;
  bf* A0 = Xs + S::NX;  // 층 0 출력 → 역방향에서 Z0 로 덮음
  bf* A1 = A0 + S::NA0;
  bf* Z1 = A1;  // dA1 은 A1 을 제자리에서 덮는다 (같은 원소를 읽고 씀)
  bf* Z2 = A1 + S::NA1;
  float *b0 = bias, *b1 = bias + C0;
  load_v(a.P + P::OB0, b0, C0);
  load_v(a.P + P::OB1, b1, C1);
  load_w_bf<C0, S::K0, S::K0P, S::SSW0>(a.P + P::OW0, W0s);
  load_w_bf<C1, S::K1, S::K1, S::SSW1>(a.P + P::OW1, W1s);
  load_w_bf<C2, S::K2, S::K2, S::SSW2>(a.P + P::OW2, W2s);
  for (int e = threadIdx.x; e < S::NX; e += NTT)  // 덧댄 열 0, 열 K0 = 1 → dW0 의 그 열이 db0 = Σ Z0 (mma 가 편향 기울기까지)
    Xs[e] = __float2bfloat16_rn(S::ONES && e % S::SSX == S::K0 ? 1.f : 0.f);
  __syncthreads();
  DWTc<C2, S::K2, 2, 1> dw2;  // 32×128: 단위 8
  DWTc<C1, S::K1, 2, 1> dw1;  // 32×64: 단위 4, 묶음 2
  DWTc<C0, S::K0P, 1, 1> dw0; // 16×K0P: 단위 1~2
  DBTc<C0> db0;
  DBTc<C1> db1;
  DBTc<C2> db2;
  dw2.zero();
  dw1.zero();
  dw0.zero();
  const long total = (long)a.n * ((H / 8) * (W / 8));
  const long ntiles = (total + TPT - 1) / TPT;
  const bool broken = a.broken;
  using PF = Prefetch<CIN, S::SSX, true, C2, H, W>;
  PF pf;
  pf.fetch(a, (long)blockIdx.x * TPT, total);
  for (long t = blockIdx.x; t < ntiles; t += gridDim.x) {
    pf.store(Xs, Z2, S::SSZ2, broken);
    __syncthreads();
    pf.fetch(a, (t + gridDim.x) * TPT, total);
    fwd01_tc<S, C0, C1>(Xs, W0s, W1s, b0, b1, A0, A1);
    // 층 2: dW2 += Z2ᵀ A1 (k = 조각), db2, dA1 = Z2 · W2 → Z1 = dA1 · ELU'(A1)
    dw2.template add<TPT>([&](int k, int m) { return Z2 + k * S::SSZ2 + m; },
                          [&](int k, int n) { return A1 + S::seg1(k * 4 + n / C1) + n % C1; });
    db2.template add<TPT>([&](int r, int c) { return Z2 + r * S::SSZ2 + c; });
    {
      Held<TPT, S::K2, C2, 1, 1, false, true> da1;  // dA1 = Z2 · W2 (같은 단계에서 계산, 쓰기는 dW2 가 A1 을 다 읽은 뒤)
      da1.compute([&](int m, int k) { return Z2 + m * S::SSZ2 + k; }, [&](int k, int n) { return W2s + k * S::SSW2 + n; });
      __syncthreads();
      da1.emit_to([&](int m, int n, float v0, float v1) {
        int o = S::seg1(m * 4 + n / C1) + n % C1;
        float2 av = ld2(A1 + o);
        st2(Z1 + o, v0 * elu_d(av.x, broken), v1 * elu_d(av.y, broken));
      });
    }
    __syncthreads();
    // 층 1: dW1 += Z1ᵀ A0 (k = r1), db1, dA0 = Z1 · W1 → A0 자리에 Z0
    dw1.template add<TPT * 4>([&](int k, int m) { return Z1 + S::seg1(k) + m; },
                              [&](int k, int n) { return A0 + S::seg0(k * 4 + n / C0) + n % C0; });
    db1.template add<TPT * 4>([&](int r, int c) { return Z1 + S::seg1(r) + c; });
    {
      Held<TPT * 4, S::K1, C1, 1, 2, false, true> da0;  // dA0 = Z1 · W1 → A0 자리에 Z0 (dW1 이 A0 을 다 읽은 뒤)
      da0.compute([&](int m, int k) { return Z1 + S::seg1(m) + k; }, [&](int k, int n) { return W1s + k * S::SSW1 + n; });
      __syncthreads();
      da0.emit_to([&](int m, int n, float v0, float v1) {
        bf* p = A0 + S::seg0(m * 4 + n / C0) + n % C0;
        float2 av = ld2(p);
        st2(p, v0 * elu_d(av.x, broken), v1 * elu_d(av.y, broken));
      });
    }
    __syncthreads();
    // 층 0: dW0 += Z0ᵀ X (k = r0), db0
    dw0.template add<TPT * 16>([&](int k, int m) { return A0 + S::seg0(k) + m; },
                               [&](int k, int n) { return Xs + k * S::SSX + n; });
    if constexpr (!S::ONES) db0.template add<TPT * 16>([&](int r, int c) { return A0 + S::seg0(r) + c; });
    __syncthreads();
  }
  // 블록 부분합: 묶음 순서대로 더해 전역 part 로 (공유 메모리를 scratch 로 다시 씀)
  float* part = a.part + (long)blockIdx.x * P::NP;
  float* scr = reinterpret_cast<float*>(smem4);
  auto flush = [&](auto& dw, int M, int N, int K, int ow, float scale, int ob) {
    __syncthreads();
    dw.dump(scr);
    __syncthreads();
    for (int e = threadIdx.x; e < M * K; e += NTT) {
      int m = e / K, k = e % K;
      float s = 0.f;
      for (int gi = 0; gi < std::remove_reference_t<decltype(dw)>::G; ++gi) s += scr[gi * M * N + m * N + k];
      part[ow + e] = s * scale;
    }
    if (ob >= 0)  // 1 열(k = K) = 편향 기울기
      for (int m = threadIdx.x; m < M; m += NTT) {
        float s = 0.f;
        for (int gi = 0; gi < std::remove_reference_t<decltype(dw)>::G; ++gi) s += scr[gi * M * N + m * N + K];
        part[ob + m] = s;
      }
  };
  flush(dw0, C0, S::K0P, S::K0, P::OW0, 1.f / 255.f, S::ONES ? P::OB0 : -1);
  flush(dw1, C1, S::K1, S::K1, P::OW1, 1.f, -1);
  flush(dw2, C2, S::K2, S::K2, P::OW2, 1.f, -1);
  auto flushb = [&](auto& db, int C, int ob) {
    __syncthreads();
    scr[threadIdx.x] = db.s;
    __syncthreads();
    if (threadIdx.x < C) {
      float s = 0.f;
      for (int gi = 0; gi < std::remove_reference_t<decltype(db)>::G; ++gi) s += scr[gi * C + threadIdx.x];
      part[ob + threadIdx.x] = s;
    }
  };
  if (!S::ONES) flushb(db0, C0, P::OB0);
  flushb(db1, C1, P::OB1);
  flushb(db2, C2, P::OB2);
}

template <int CIN, int C0, int C1, int C2>
size_t smem_fwd_tc() {
  using S = TcShape<CIN, C0, C1, C2>;
  return 4 * (size_t)r4(S::NBIAS) + 2 * (size_t)(S::NW0 + S::NW1 + S::NW2 + S::NX + S::NA0 + S::NA1);
}
template <int CIN, int C0, int C1, int C2>
size_t smem_bwd_tc() {
  using S = TcShape<CIN, C0, C1, C2>;
  size_t b = 4 * (size_t)r4(S::NBIAS) + 2 * (size_t)(S::NW0 + S::NW1 + S::NW2 + S::NX + S::NA0 + S::NA1 + S::NZ2);
  size_t scr = 4 * (size_t)std::max({DWTc<C2, S::K2, 2, 1>::G * C2 * S::K2, DWTc<C1, S::K1, 2, 1>::G * C1 * S::K1,
                                      DWTc<C0, S::K0P, 1, 1>::G * C0 * S::K0P, NTT});  // flush 때 scratch
  return std::max(b, scr);
}

}  // namespace tc
