// 합성곱 가지 융합 커널의 텐서 코어 판 — 워프 하나가 자기 몫을 레지스터 안에서 끝까지 (bf16 입력, fp32 누산: mma.sync m16n8k16).
// cnn_fused.cu 안(익명 이름공간)에서만 include. 모양은 채널 16-32-32 고정, 입력 채널 CIN 1~8.
//
// 일 단위 = 조각 16 개(셋째 층 출력 행 16 개 = mma 행 16 개 하나). 행 순서를 이렇게 잡으면 앞 층 누산기(C 조각)가
// 그대로 다음 층 A 조각이 된다(FlashAttention 의 P → A 와 같은 수 — 공유 메모리·동기화 없음):
//   층 0 행 묶음 (q1, c1) = 16 조각 각각의 층 0 픽셀 (q1, c1)   → 두 n8 출력(채널 16) = 층 1 의 k 묶음 c1
//   층 1 행 묶음 q1       = 16 조각 각각의 층 1 픽셀 q1        → 네 n8 출력(채널 32) = 층 2 의 k 묶음 (q1, 앞·뒤 16)
//   층 2 행 묶음          = 16 조각
// 층 0 의 k 순서는 [cin][dy][dx] (가중치를 공유 메모리에 올릴 때 바꿈) — 그러면 A 조각 원소 쌍(k, k+1)이 입력 한 줄의
// 이웃 두 바이트라 u8 입력을 u32 로 바로 읽어 조각을 만든다(공유 메모리 거치지 않음).
// 정방향: 워프가 일 단위 하나를 층 0 → 1 → 2 까지. 블록 동기화는 시작 때 가중치 올리기 한 번뿐.
// 역방향: 워프가 (일 단위, q1) 하나 — 층 2 출력 기울기는 조각마다(전역 F3·dF3) 있으므로 q1 넷이 서로 기대지 않는다.
//   워프는 dW2 의 자기 q1 열(32×32)·dW1 전체(32×64)·dW0(16×K0) 를 블록 수명 내내 레지스터에 쌓고,
//   전치가 필요한 곱(dW = Zᵀ·X)은 movmatrix.trans 로 C 조각을 B·A 조각으로 바꾼다. 끝에 워프 순서대로 더해 블록 부분합(결정적).
// 반올림은 옛 판(공유 메모리 bf16)과 같은 자리: 활성값·Z 는 bf16 로 반올림해 곱하고, ELU′ 는 반올림된 활성값에서.
#pragma once

namespace wr {

constexpr int NT = 256;  // 블록 스레드 (워프 8 개)
constexpr int NW = NT / 32;

__device__ __forceinline__ void mma(float (&c)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
  asm("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
__device__ __forceinline__ void mma(float (&c)[4], const uint32_t (&a)[4], uint2 b) { mma(c, a, b.x, b.y); }
// k 8 판: A 조각 앞 두 레지스터(k 0~7)·B 조각 앞 레지스터만 — k 묶음 뒤 절반이 덧댄 0 일 때 (일이 반)
__device__ __forceinline__ void mma8(float (&c)[4], uint32_t a0, uint32_t a1, uint32_t b0) {
  asm("mma.sync.aligned.m16n8k8.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};\n"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a0), "r"(a1), "r"(b0));
}
// 8×8 bf16 전치 (워프 전체): 스레드가 가진 (행 g, 열 2q·2q+1) → 전치 행렬의 같은 자리
__device__ __forceinline__ uint32_t tr(uint32_t v) {
  uint32_t d;
  asm("movmatrix.sync.aligned.m8n8.trans.b16 %0, %1;\n" : "=r"(d) : "r"(v));
  return d;
}
__device__ __forceinline__ uint32_t pk(float lo, float hi) {
  __nv_bfloat162 h = __floats2bfloat162_rn(lo, hi);
  return *reinterpret_cast<uint32_t*>(&h);
}
__device__ __forceinline__ float2 upk(uint32_t v) {
  return __bfloat1622float2(*reinterpret_cast<__nv_bfloat162*>(&v));
}
// u32 안 바이트 2X·2X+1 (u8 0~255) → bf16 두 개: 2^23 + v 의 float 에서 2^23 을 빼면 정확히 v, 위 16 비트가 bf16
template <int X>
__device__ __forceinline__ uint32_t u8pair(uint32_t v) {
  const float f0 = __uint_as_float(__byte_perm(v, 0x4Bu, 0x4550 + 2 * X)) - 8388608.f;
  const float f1 = __uint_as_float(__byte_perm(v, 0x4Bu, 0x4551 + 2 * X)) - 8388608.f;
  return __byte_perm(__float_as_uint(f0), __float_as_uint(f1), 0x7632);
}

// cp.async: 전역 → 공유 메모리 (레지스터 안 거침). n = 0 이면 0 으로 채움
template <int B>
__device__ __forceinline__ void cpa(void* dst, const void* src, bool ok) {
  asm volatile("cp.async.ca.shared.global [%0], [%1], %2, %3;\n" ::"r"((uint32_t)__cvta_generic_to_shared(dst)), "l"(src), "n"(B),
               "r"(ok ? B : 0));
}
__device__ __forceinline__ void cpa_commit() { asm volatile("cp.async.commit_group;\n" ::); }
__device__ __forceinline__ void cpa_wait() { asm volatile("cp.async.wait_group 0;\n" ::: "memory"); }

template <int CIN>
struct WShape {
  static_assert(CIN >= 1 && CIN <= 8, "CIN");
  static constexpr int K0 = 4 * CIN, NKB0 = (K0 + 15) / 16, NT0 = (K0 + 7) / 8;  // 층 0 k 묶음 수, dW0 의 n8 수
  // 공유 메모리 B 조각 번호 (조각 = 레인마다 uint2)
  static constexpr int FW0 = 0, FW1 = FW0 + 2 * NKB0, FW2 = FW1 + 16, FW2T = FW2 + 32, FW1T = FW2T + 32, NF = FW1T + 16;
};

// B 조각 하나: 레인 (g, q) 의 b0 = (B[k0+2q][n], B[k0+2q+1][n]), b1 = k0+8 쪽, n = n0 + g. Bv(k, n) 이 값.
template <class BV>
__device__ __forceinline__ uint2 bfrag(int lane, int k0, int n0, BV Bv) {
  const int g = lane >> 2, q = lane & 3, n = n0 + g, k = k0 + 2 * q;
  return make_uint2(pk(Bv(k, n), Bv(k + 1, n)), pk(Bv(k + 8, n), Bv(k + 9, n)));
}

// 가중치 → 공유 메모리 B 조각 (fp32 [cout][K] 를 읽어 bf16). 정방향은 FW0·FW1·FW2, 역방향은 FW0·FW1·FW2T·FW1T.
template <int CIN, bool BWD>
__device__ void build_frags(const float* __restrict__ P, uint2* Fr) {
  using S = WShape<CIN>;
  using PS = Shape<CIN, 16, 32, 32>;
  const float *W0 = P + PS::OW0, *W1 = P + PS::OW1, *W2 = P + PS::OW2;
  for (int e = threadIdx.x; e < S::NF * 32; e += NT) {
    const int f = e >> 5, l = e & 31;
    uint2 v = make_uint2(0u, 0u);
    if (f < S::FW1) {  // 층 0: B[k][n] = W0 (k 순서 [cin][자식]) — 원래 열 = 자식·CIN + cin
      const int kb = (f - S::FW0) >> 1, j = (f - S::FW0) & 1;
      v = bfrag(l, 16 * kb, 8 * j, [&](int k, int n) { return k < S::K0 ? W0[n * S::K0 + (k & 3) * CIN + (k >> 2)] : 0.f; });
    } else if (f < S::FW2) {  // 층 1 정방향: B[k1][n] = W1[n][k1]
      const int c1 = (f - S::FW1) >> 2, j = (f - S::FW1) & 3;
      v = bfrag(l, 16 * c1, 8 * j, [&](int k, int n) { return W1[n * 64 + k]; });
    } else if (f < S::FW2T) {  // 층 2 정방향: B[k2][n] = W2[n][k2], k 묶음 = q1·2 + h
      if (BWD) continue;
      const int kb = (f - S::FW2) >> 2, j = (f - S::FW2) & 3;
      v = bfrag(l, 16 * kb, 8 * j, [&](int k, int n) { return W2[n * 128 + k]; });
    } else if (f < S::FW1T) {  // dA1(q1) = Z2·W2[:, q1 열]: B[ch2][n] = W2[ch2][32·q1 + n]
      if (!BWD) continue;
      const int q1 = (f - S::FW2T) >> 3, kb = ((f - S::FW2T) >> 2) & 1, j = (f - S::FW2T) & 3;
      v = bfrag(l, 16 * kb, 8 * j, [&](int k, int n) { return W2[k * 128 + 32 * q1 + n]; });
    } else {  // dA0 = Z1·W1: B[ch1][k1] = W1[ch1][k1]
      if (!BWD) continue;
      const int kb = (f - S::FW1T) >> 3, jj = (f - S::FW1T) & 7;
      v = bfrag(l, 16 * kb, 8 * jj, [&](int k, int n) { return W1[k * 64 + n]; });
    }
    Fr[e] = v;
  }
}

// (일 단위, q1) 의 입력: 레인 (g, q) 가 조각 g·g+8 (r), cy, k 묶음 kb, 반 h 마다 u32 하나 =
// 입력 행 y = qy·4 + cy·2 + (q & 1), 채널 cin = 4kb + 2h + (q >> 1), 열 qx·4 .. +3 (바이트 2cx·2cx+1 = 자식 c1 = (cy, cx) 의 dx 0·1)
template <int CIN, int H, int W>
struct In {
  using S = WShape<CIN>;
  static constexpr int WP = W / 8, PPS = (H / 8) * WP;
  uint32_t v[2][2][S::NKB0][2];
  __device__ __forceinline__ void load(const Args& a, long u, int q1, long total) {
    const int l = threadIdx.x & 31, g = l >> 2, q = l & 3, qy = q1 >> 1, qx = q1 & 1;
#pragma unroll
    for (int r = 0; r < 2; ++r) {
      const long gp = u * 16 + g + 8 * r;
      const bool ok = gp < total;
      const uint8_t* base = a.img;
      if (ok) {
        long s = gp / PPS;
        const int pix = (int)(gp % PPS), py = pix / WP, px = pix % WP;
        if (a.idx) s = __ldg(a.idx + s);
        base = a.img + s * a.img_stride + a.img_off + (py * 8 + qy * 4 + (q & 1)) * W + px * 8 + qx * 4 + (q >> 1) * H * W;
      }
#pragma unroll
      for (int cy = 0; cy < 2; ++cy)
#pragma unroll
        for (int kb = 0; kb < S::NKB0; ++kb)
#pragma unroll
          for (int h = 0; h < 2; ++h) {
            const int c0 = 4 * kb + 2 * h;  // 이 레인의 cin = c0 + (q >> 1)
            uint32_t x = 0u;
            if (c0 < CIN && ok && c0 + (q >> 1) < CIN)
              x = __ldg(reinterpret_cast<const uint32_t*>(base + c0 * H * W + cy * 2 * W));
            v[r][cy][kb][h] = x;
          }
    }
  }
  // 같은 값을 cp.async 로 레인 자기 칸 slot[i·32 + 레인] 에 (읽는 것도 그 레인뿐이라 워프 동기화가 필요 없다)
  static constexpr int NSLOT = 8 * S::NKB0;
  __device__ __forceinline__ static void issue(const Args& a, long u, int q1, long total, uint32_t* slot) {
    const int l = threadIdx.x & 31, g = l >> 2, q = l & 3, qy = q1 >> 1, qx = q1 & 1;
#pragma unroll
    for (int r = 0; r < 2; ++r) {
      const long gp = u * 16 + g + 8 * r;
      const bool ok = gp < total;
      const uint8_t* base = a.img;
      if (ok) {
        long s = gp / PPS;
        const int pix = (int)(gp % PPS), py = pix / WP, px = pix % WP;
        if (a.idx) s = __ldg(a.idx + s);
        base = a.img + s * a.img_stride + a.img_off + (py * 8 + qy * 4 + (q & 1)) * W + px * 8 + qx * 4 + (q >> 1) * H * W;
      }
#pragma unroll
      for (int cy = 0; cy < 2; ++cy)
#pragma unroll
        for (int kb = 0; kb < S::NKB0; ++kb)
#pragma unroll
          for (int h = 0; h < 2; ++h) {
            const int c0 = 4 * kb + 2 * h, i = ((r * 2 + cy) * S::NKB0 + kb) * 2 + h;
            if (c0 < CIN) {
              const bool v = ok && c0 + (q >> 1) < CIN;
              cpa<4>(slot + i * 32 + l, v ? base + c0 * H * W + cy * 2 * W : a.img, v);
            }
          }
    }
  }
  __device__ __forceinline__ void fetch(const uint32_t* slot) {
    const int l = threadIdx.x & 31;
#pragma unroll
    for (int r = 0; r < 2; ++r)
#pragma unroll
      for (int cy = 0; cy < 2; ++cy)
#pragma unroll
        for (int kb = 0; kb < S::NKB0; ++kb)
#pragma unroll
          for (int h = 0; h < 2; ++h) {
            const int i = ((r * 2 + cy) * S::NKB0 + kb) * 2 + h;
            v[r][cy][kb][h] = 4 * kb + 2 * h < CIN ? slot[i * 32 + l] : 0u;
          }
  }
  // 층 0 행 묶음 (q1, c1 = cy·2 + cx) 의 k 묶음 kb A 조각
  template <int CX>
  __device__ __forceinline__ void frag(uint32_t (&a)[4], int cy, int kb) const {
    a[0] = u8pair<CX>(v[0][cy][kb][0]);
    a[1] = u8pair<CX>(v[1][cy][kb][0]);
    a[2] = u8pair<CX>(v[0][cy][kb][1]);
    a[3] = u8pair<CX>(v[1][cy][kb][1]);
  }
  __device__ __forceinline__ void frag(uint32_t (&a)[4], int c1, int kb) const {
    if (c1 & 1)
      frag<1>(a, c1 >> 1, kb);
    else
      frag<0>(a, c1 >> 1, kb);
  }
};

__device__ __forceinline__ long oidx(const Args& a, long gp, int pps, int c) {
  return (gp / pps) * a.feat + a.flat_off + (gp % pps) * 32 + c;
}

// 층 0 (q1, c1): 입력 A 조각 → 누산 → /255 + 편향 + ELU → bf16 = 층 1 의 A 조각 (k 묶음 c1)
template <int CIN, int H, int W>
__device__ __forceinline__ void layer0(const In<CIN, H, W>& in, int c1, const uint2* Fr, const float (&b0c)[2][2],
                                       uint32_t (&af)[4]) {
  using S = WShape<CIN>;
  const int l = threadIdx.x & 31;
  float acc[2][4] = {};
#pragma unroll
  for (int kb = 0; kb < S::NKB0; ++kb) {
    uint32_t xa[4];
    in.frag(xa, c1, kb);
#pragma unroll
    for (int j = 0; j < 2; ++j) {
      const uint2 b = Fr[(S::FW0 + kb * 2 + j) * 32 + l];
      if (4 * kb + 2 >= CIN)  // 이 k 묶음의 뒤 8 (cin ≥ 4kb + 2)은 모두 덧댐
        mma8(acc[j], xa[0], xa[1], b.x);
      else
        mma(acc[j], xa, b);
    }
  }
  constexpr float inv = 1.f / 255.f;
#pragma unroll
  for (int j = 0; j < 2; ++j) {
    af[2 * j] = pk(elu_f(fmaf(acc[j][0], inv, b0c[j][0])), elu_f(fmaf(acc[j][1], inv, b0c[j][1])));
    af[2 * j + 1] = pk(elu_f(fmaf(acc[j][2], inv, b0c[j][0])), elu_f(fmaf(acc[j][3], inv, b0c[j][1])));
  }
}

// 편향을 레인의 열(8j + 2q, +1)로
template <int NJ>
__device__ __forceinline__ void load_bias(const float* __restrict__ b, float (&o)[NJ][2]) {
  const int q = threadIdx.x & 3;
#pragma unroll
  for (int j = 0; j < NJ; ++j) {
    o[j][0] = __ldg(b + 8 * j + 2 * q);
    o[j][1] = __ldg(b + 8 * j + 2 * q + 1);
  }
}

template <int CIN, int H, int W>
__global__ void __launch_bounds__(NT) k_fwd(Args a) {
  using S = WShape<CIN>;
  using PS = Shape<CIN, 16, 32, 32>;
  using I = In<CIN, H, W>;
  extern __shared__ float4 smem4[];
  uint2* Fr = reinterpret_cast<uint2*>(smem4);
  build_frags<CIN, false>(a.P, Fr);
  float b0c[2][2], b1c[4][2], b2c[4][2];
  load_bias(a.P + PS::OB0, b0c);
  load_bias(a.P + PS::OB1, b1c);
  load_bias(a.P + PS::OB2, b2c);
  __syncthreads();
  const int l = threadIdx.x & 31, g = l >> 2, q = l & 3;
  const long total = (long)a.n * I::PPS, nu = (total + 15) / 16;
  for (long u = (long)blockIdx.x * NW + (threadIdx.x >> 5); u < nu; u += (long)gridDim.x * NW) {
    float acc2[4][4] = {};
    I in;
    in.load(a, u, 0, total);
#pragma unroll 1
    for (int q1 = 0; q1 < 4; ++q1) {
      I cur = in;
      if (q1 < 3) in.load(a, u, q1 + 1, total);  // 다음 사분면 미리 읽기
      float acc1[4][4] = {};
#pragma unroll
      for (int c1 = 0; c1 < 4; ++c1) {
        uint32_t af[4];
        layer0(cur, c1, Fr, b0c, af);
#pragma unroll
        for (int j = 0; j < 4; ++j) mma(acc1[j], af, Fr[(S::FW1 + c1 * 4 + j) * 32 + l]);
      }
#pragma unroll
      for (int h = 0; h < 2; ++h) {
        uint32_t af[4];
#pragma unroll
        for (int t = 0; t < 2; ++t) {
          const int j = 2 * h + t;
          af[2 * t] = pk(elu_f(acc1[j][0] + b1c[j][0]), elu_f(acc1[j][1] + b1c[j][1]));
          af[2 * t + 1] = pk(elu_f(acc1[j][2] + b1c[j][0]), elu_f(acc1[j][3] + b1c[j][1]));
        }
#pragma unroll
        for (int j = 0; j < 4; ++j) mma(acc2[j], af, Fr[(S::FW2 + (q1 * 2 + h) * 4 + j) * 32 + l]);
      }
    }
#pragma unroll
    for (int r = 0; r < 2; ++r) {
      const long gp = u * 16 + g + 8 * r;
      if (gp < total)
#pragma unroll
        for (int j = 0; j < 4; ++j)
          *reinterpret_cast<float2*>(a.X0 + oidx(a, gp, I::PPS, 8 * j + 2 * q)) =
              make_float2(elu_f(acc2[j][2 * r] + b2c[j][0]), elu_f(acc2[j][2 * r + 1] + b2c[j][1]));
    }
  }
}

template <int CIN, int H, int W>
__global__ void __launch_bounds__(NT, 1) k_bwd(Args a) {
  using S = WShape<CIN>;
  using PS = Shape<CIN, 16, 32, 32>;
  using I = In<CIN, H, W>;
  extern __shared__ float4 smem4[];
  uint2* Fr = reinterpret_cast<uint2*>(smem4);
  build_frags<CIN, true>(a.P, Fr);
  float b0c[2][2], b1c[4][2];
  load_bias(a.P + PS::OB0, b0c);
  load_bias(a.P + PS::OB1, b1c);
  __syncthreads();
  const int l = threadIdx.x & 31, g = l >> 2, q = l & 3, w = threadIdx.x >> 5, q1 = w & 3, grp = w >> 2;
  constexpr int G = NW / 4;
  const bool broken = a.broken;
  float dW2[2][4][4] = {}, dW1[2][8][4] = {}, dW0[S::NT0][4] = {};
  float db0[2][2] = {}, db1[4][2] = {}, db2[4][2] = {};
  const long total = (long)a.n * I::PPS, nu = (total + 15) / 16;
  // 다음 일의 입력·dZ3 를 cp.async 로 워프 자기 칸에 미리 (레지스터 안 씀): [입력 NSLOT][dZ3 8 (float2)] × 32 레인
  uint32_t* slot = reinterpret_cast<uint32_t*>(Fr + S::NF * 32) + w * (I::NSLOT + 16) * 32;
  float2* zslot = reinterpret_cast<float2*>(slot + I::NSLOT * 32);
  auto issue = [&](long u) {
    if (u >= nu) return;
    I::issue(a, u, q1, total, slot);
#pragma unroll
    for (int r = 0; r < 2; ++r) {
      const long gp = u * 16 + g + 8 * r;
#pragma unroll
      for (int kb = 0; kb < 2; ++kb)
#pragma unroll
        for (int hh = 0; hh < 2; ++hh) {
          const int i = (r * 2 + kb) * 2 + hh;
          const bool ok = gp < total;
          const long ix = ok ? oidx(a, gp, I::PPS, 16 * kb + 8 * hh + 2 * q) : 0;
          cpa<8>(zslot + i * 32 + l, a.dX0 + ix, ok);
        }
    }
  };
  const long ustep = (long)gridDim.x * G;
  issue((long)blockIdx.x * G + grp);
  cpa_commit();
  for (long u = (long)blockIdx.x * G + grp; u < nu; u += ustep) {
    cpa_wait();
    I in;
    in.fetch(slot);
    // Z2 = dZ3 (= dF3 · ELU′(F3), 넷째 층 역방향이 곱해 둠) 의 bf16, A 조각 (행 조각, k = ch2): z2[kb][hh·2 + r]
    uint32_t z2[2][4];
#pragma unroll
    for (int r = 0; r < 2; ++r)
#pragma unroll
      for (int kb = 0; kb < 2; ++kb)
#pragma unroll
        for (int hh = 0; hh < 2; ++hh) {
          const int i = (r * 2 + kb) * 2 + hh;
          const float2 d = zslot[i * 32 + l];
          z2[kb][hh * 2 + r] = pk(d.x, d.y);
        }
    issue(u + ustep);  // 칸을 다 읽었으니 다음 일을 바로 걸어 둠
    cpa_commit();
    if (q1 == 0)  // db2 는 조각마다 한 번 (q1 0 워프만)
#pragma unroll
      for (int kb = 0; kb < 2; ++kb)
#pragma unroll
        for (int i = 0; i < 4; ++i) {
          const float2 z = upk(z2[kb][i]);
          db2[2 * kb + (i >> 1)][0] += z.x;
          db2[2 * kb + (i >> 1)][1] += z.y;
        }
    // 층 0·1 다시 계산 (이 q1 만)
    uint32_t a0f[4][4];
    float acc1[4][4] = {};
#pragma unroll
    for (int c1 = 0; c1 < 4; ++c1) {
      layer0(in, c1, Fr, b0c, a0f[c1]);
#pragma unroll
      for (int j = 0; j < 4; ++j) mma(acc1[j], a0f[c1], Fr[(S::FW1 + c1 * 4 + j) * 32 + l]);
    }
    uint32_t a1h[4][2];
#pragma unroll
    for (int j = 0; j < 4; ++j)
#pragma unroll
      for (int r = 0; r < 2; ++r)
        a1h[j][r] = pk(elu_f(acc1[j][2 * r] + b1c[j][0]), elu_f(acc1[j][2 * r + 1] + b1c[j][1]));
    // dW2[:, q1 열] += Z2ᵀ · A1(q1)
    {
      uint32_t z2t[2][4];
#pragma unroll
      for (int mt = 0; mt < 2; ++mt) {
        z2t[mt][0] = tr(z2[mt][0]);
        z2t[mt][1] = tr(z2[mt][2]);
        z2t[mt][2] = tr(z2[mt][1]);
        z2t[mt][3] = tr(z2[mt][3]);
      }
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        const uint32_t b0 = tr(a1h[j][0]), b1 = tr(a1h[j][1]);
#pragma unroll
        for (int mt = 0; mt < 2; ++mt) mma(dW2[mt][j], z2t[mt], b0, b1);
      }
    }
    // dA1(q1) = Z2 · W2[:, q1 열] → Z1 = dA1 · ELU′(A1)
    uint32_t z1h[4][2];
    {
      float d1[4][4] = {};
#pragma unroll
      for (int kb = 0; kb < 2; ++kb)
#pragma unroll
        for (int j = 0; j < 4; ++j) mma(d1[j], z2[kb], Fr[(S::FW2T + (q1 * 2 + kb) * 4 + j) * 32 + l]);
#pragma unroll
      for (int j = 0; j < 4; ++j)
#pragma unroll
        for (int r = 0; r < 2; ++r) {
          const float2 av = upk(a1h[j][r]);
          z1h[j][r] = pk(d1[j][2 * r] * elu_d(av.x, broken), d1[j][2 * r + 1] * elu_d(av.y, broken));
          const float2 z = upk(z1h[j][r]);
          db1[j][0] += z.x;
          db1[j][1] += z.y;
        }
    }
    // dW1 += Z1ᵀ · A0(q1)
    {
      uint32_t z1t[2][4];
#pragma unroll
      for (int mt = 0; mt < 2; ++mt) {
        z1t[mt][0] = tr(z1h[2 * mt][0]);
        z1t[mt][1] = tr(z1h[2 * mt + 1][0]);
        z1t[mt][2] = tr(z1h[2 * mt][1]);
        z1t[mt][3] = tr(z1h[2 * mt + 1][1]);
      }
#pragma unroll
      for (int jj = 0; jj < 8; ++jj) {
        const int c1 = jj >> 1, hh = jj & 1;
        const uint32_t b0 = tr(a0f[c1][2 * hh]), b1 = tr(a0f[c1][2 * hh + 1]);
#pragma unroll
        for (int mt = 0; mt < 2; ++mt) mma(dW1[mt][jj], z1t[mt], b0, b1);
      }
    }
    // dA0 = Z1 · W1 → Z0 = dA0 · ELU′(A0), dW0 += Z0ᵀ · X
    float d0[8][4] = {};
#pragma unroll
    for (int kb = 0; kb < 2; ++kb) {
      const uint32_t af[4] = {z1h[2 * kb][0], z1h[2 * kb][1], z1h[2 * kb + 1][0], z1h[2 * kb + 1][1]};
#pragma unroll
      for (int jj = 0; jj < 8; ++jj) mma(d0[jj], af, Fr[(S::FW1T + kb * 8 + jj) * 32 + l]);
    }
#pragma unroll
    for (int c1 = 0; c1 < 4; ++c1) {
      uint32_t z0[4];
#pragma unroll
      for (int hh = 0; hh < 2; ++hh)
#pragma unroll
        for (int r = 0; r < 2; ++r) {
          const int jj = 2 * c1 + hh, i = 2 * hh + r;
          const float2 av = upk(a0f[c1][i]);
          z0[i] = pk(d0[jj][2 * r] * elu_d(av.x, broken), d0[jj][2 * r + 1] * elu_d(av.y, broken));
          const float2 z = upk(z0[i]);
          db0[hh][0] += z.x;
          db0[hh][1] += z.y;
        }
      const uint32_t z0t[4] = {tr(z0[0]), tr(z0[2]), tr(z0[1]), tr(z0[3])};
#pragma unroll
      for (int kb = 0; kb < S::NKB0; ++kb) {
        uint32_t xa[4];
        in.frag(xa, c1, kb);
#pragma unroll
        for (int t2 = 0; t2 < 2; ++t2)
          if (kb * 2 + t2 < S::NT0) mma(dW0[kb * 2 + t2], z0t, tr(xa[2 * t2]), tr(xa[2 * t2 + 1]));
      }
    }
  }
  // 편향 기울기: 같은 q 레인끼리 (g 8 개) 고정 순서 나비 합
  auto red = [](float& v) {
#pragma unroll
    for (int m = 4; m < 32; m <<= 1) v += __shfl_xor_sync(0xffffffffu, v, m);
  };
#pragma unroll
  for (int j = 0; j < 2; ++j) red(db0[j][0]), red(db0[j][1]);
#pragma unroll
  for (int j = 0; j < 4; ++j) red(db1[j][0]), red(db1[j][1]), red(db2[j][0]), red(db2[j][1]);
  // 블록 부분합: 워프 순서대로 공유 메모리에 더한 뒤 part 로 (결정적)
  float* acc = reinterpret_cast<float*>(smem4);
  __syncthreads();
  for (int e = threadIdx.x; e < PS::NP; e += NT) acc[e] = 0.f;
  __syncthreads();
  for (int ww = 0; ww < NW; ++ww) {
    if (w == ww) {
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        const int row = g + 8 * (e >> 1), col = 2 * q + (e & 1);
#pragma unroll
        for (int mt = 0; mt < 2; ++mt) {
#pragma unroll
          for (int j = 0; j < 4; ++j) acc[PS::OW2 + (16 * mt + row) * 128 + 32 * q1 + 8 * j + col] += dW2[mt][j][e];
#pragma unroll
          for (int jj = 0; jj < 8; ++jj) acc[PS::OW1 + (16 * mt + row) * 64 + 8 * jj + col] += dW1[mt][jj][e];
        }
#pragma unroll
        for (int t = 0; t < S::NT0; ++t) {
          const int k = 8 * t + col;
          if (k < S::K0) acc[PS::OW0 + row * S::K0 + (k & 3) * CIN + (k >> 2)] += dW0[t][e];
        }
      }
      if (g == 0) {
#pragma unroll
        for (int c = 0; c < 2; ++c) {
#pragma unroll
          for (int j = 0; j < 2; ++j) acc[PS::OB0 + 8 * j + 2 * q + c] += db0[j][c];
#pragma unroll
          for (int j = 0; j < 4; ++j) {
            acc[PS::OB1 + 8 * j + 2 * q + c] += db1[j][c];
            acc[PS::OB2 + 8 * j + 2 * q + c] += db2[j][c];
          }
        }
      }
    }
    __syncthreads();
  }
  float* part = a.part + (long)blockIdx.x * PS::NP;
  for (int e = threadIdx.x; e < PS::NP; e += NT) part[e] = e < PS::OB0 ? acc[e] * (1.f / 255.f) : acc[e];
}

template <int CIN>
size_t smem_fwd() {
  return (size_t)WShape<CIN>::NF * 32 * sizeof(uint2);
}
template <int CIN>
size_t smem_bwd() {
  const size_t slots = (size_t)NW * (8 * WShape<CIN>::NKB0 + 16) * 32 * sizeof(uint32_t);
  return std::max((size_t)WShape<CIN>::NF * 32 * sizeof(uint2) + slots, (size_t)Shape<CIN, 16, 32, 32>::NP * sizeof(float));
}

}  // namespace wr
