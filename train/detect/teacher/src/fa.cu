// 텐서코어 플래시 어텐션(mma.sync m16n8k16, half·bf16 입력, float 누적).
// 블록 = 워프 4 개 = 질의 64 줄, 키·값 64 개씩 공유 메모리로 돌며 온라인 softmax.
// 지원: 키 패딩 마스크, 인과 마스크, 상자 상대 위치 편향(BoxRPB), 길이 아무거나.
#include <cfloat>
#include <type_traits>

#include "k.cuh"

namespace k {

namespace {
template <typename T>
__device__ __forceinline__ uint32_t pack2(float a, float b) {
  if constexpr (std::is_same_v<T, f16>) {
    __half2 h = __floats2half2_rn(a, b);
    return *reinterpret_cast<uint32_t*>(&h);
  } else {
    __nv_bfloat162 h = __floats2bfloat162_rn(a, b);
    return *reinterpret_cast<uint32_t*>(&h);
  }
}
template <typename T>
__device__ __forceinline__ void mma(float* c, const uint32_t* a, const uint32_t* b) {
  if constexpr (std::is_same_v<T, f16>)
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
        "{%0,%1,%2,%3};\n"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
  else
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
        "{%0,%1,%2,%3};\n"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}
}  // namespace

__device__ __forceinline__ float ex2(float x) {
  float y;
  asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
  return y;
}

// NW = 워프 수(질의 16 × NW 줄), MASK·CAUSAL·BIAS 는 컴파일 때 정한다
template <typename T, int D, int NW, bool MASK, bool CAUSAL, bool BIAS>
__global__ void __launch_bounds__(NW * 32) flash_k(Attn<T> a) {
  constexpr int BQ = 16 * NW, BK = 64, KS = D / 16, NT = BK / 8, DT = D / 8;
  constexpr int PADK = D + 8, PADV = BK + 8;  // 뱅크 충돌 줄이기
  __shared__ __align__(16) T ks[BK][PADK];
  __shared__ __align__(16) T vt[D][PADV];  // V 전치: vt[d][key]
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, t4 = lane & 3;
  const int bh = blockIdx.y, b = bh / a.H, h = bh % a.H;
  const int q0 = blockIdx.x * BQ + warp * 16;
  const int r0 = q0 + g, r1 = q0 + g + 8;  // 이 스레드가 맡은 두 질의 줄

  // Q 조각(A 행렬): KS 단계 × 4 레지스터
  uint32_t qa[KS][4];
  {
    const T* qp = a.q + b * a.qb + h * D;
    auto ld = [&](int r, int c) -> uint32_t {
      if (r >= a.Lq) return 0u;
      return *reinterpret_cast<const uint32_t*>(qp + (long long)r * a.qr + c);
    };
#pragma unroll
    for (int s = 0; s < KS; ++s) {
      qa[s][0] = ld(r0, s * 16 + t4 * 2);
      qa[s][1] = ld(r1, s * 16 + t4 * 2);
      qa[s][2] = ld(r0, s * 16 + 8 + t4 * 2);
      qa[s][3] = ld(r1, s * 16 + 8 + t4 * 2);
    }
  }
  float o[DT][4];
#pragma unroll
  for (int j = 0; j < DT; ++j) o[j][0] = o[j][1] = o[j][2] = o[j][3] = 0.f;
  float m0 = -INFINITY, m1 = -INFINITY, l0 = 0.f, l1 = 0.f;
  const float sl2 = a.scale * 1.4426950408889634f;  // exp → exp2
  const float* by0 = nullptr;
  const float* bx0 = nullptr;
  const float* by1 = nullptr;
  const float* bx1 = nullptr;
  if (BIAS) {
    int rr0 = min(r0, a.Lq - 1), rr1 = min(r1, a.Lq - 1);
    by0 = a.by + (((long long)b * a.Lq + rr0) * a.bh) * a.H + h;
    bx0 = a.bx + (((long long)b * a.Lq + rr0) * a.bw) * a.H + h;
    by1 = a.by + (((long long)b * a.Lq + rr1) * a.bh) * a.H + h;
    bx1 = a.bx + (((long long)b * a.Lq + rr1) * a.bw) * a.H + h;
  }
  int kend = a.Lk;
  if (CAUSAL) kend = min(a.Lk, blockIdx.x * BQ + BQ);
  const T* kp = a.k + b * a.kb + h * D;
  const T* vp = a.v + b * a.vb + h * D;

  for (int k0 = 0; k0 < kend; k0 += BK) {
    __syncthreads();
    // K, V 타일 읽기 (8 개 원소 = 16 B 씩)
    for (int e = threadIdx.x; e < BK * D / 8; e += blockDim.x) {
      int j = e / (D / 8), c = (e % (D / 8)) * 8, kj = k0 + j;
      uint4 kv = make_uint4(0, 0, 0, 0), vv = make_uint4(0, 0, 0, 0);
      if (kj < a.Lk) {
        kv = *reinterpret_cast<const uint4*>(kp + (long long)kj * a.kr + c);
        vv = *reinterpret_cast<const uint4*>(vp + (long long)kj * a.vr + c);
      }
      *reinterpret_cast<uint4*>(&ks[j][c]) = kv;
      const T* vs = reinterpret_cast<const T*>(&vv);
#pragma unroll
      for (int u = 0; u < 8; ++u) vt[c + u][j] = vs[u];
    }
    __syncthreads();

    // S = Q K^T : 16 × 64 (NT 조각)
    float s[NT][4];
#pragma unroll
    for (int j = 0; j < NT; ++j) {
      s[j][0] = s[j][1] = s[j][2] = s[j][3] = 0.f;
#pragma unroll
      for (int kk = 0; kk < KS; ++kk) {
        uint32_t bb[2];
        bb[0] = *reinterpret_cast<const uint32_t*>(&ks[j * 8 + g][kk * 16 + t4 * 2]);
        bb[1] = *reinterpret_cast<const uint32_t*>(&ks[j * 8 + g][kk * 16 + 8 + t4 * 2]);
        mma<T>(s[j], qa[kk], bb);
      }
    }
    // 눈금·편향·마스크 (log2 단위)
    float mx0 = -INFINITY, mx1 = -INFINITY;
    const bool edge = k0 + BK > a.Lk;
#pragma unroll
    for (int j = 0; j < NT; ++j)
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        int kj = k0 + j * 8 + t4 * 2 + (e & 1);
        float v = s[j][e] * sl2;
        if (BIAS) {
          const float* yy = e < 2 ? by0 : by1;
          const float* xx = e < 2 ? bx0 : bx1;
          int kc = min(kj, a.Lk - 1);
          v += (yy[(kc / a.bw) * a.H] + xx[(kc % a.bw) * a.H]) * 1.4426950408889634f;
        }
        bool ok = true;
        if (edge) ok = kj < a.Lk;
        if (CAUSAL) ok = ok && kj <= (e < 2 ? r0 : r1);
        if (MASK) ok = ok && kj < a.Lk && a.kmask[b * a.kmb + kj];
        v = ok ? v : -INFINITY;
        s[j][e] = v;
        if (e < 2)
          mx0 = fmaxf(mx0, v);
        else
          mx1 = fmaxf(mx1, v);
      }
#pragma unroll
    for (int o2 = 1; o2 < 4; o2 <<= 1) {
      mx0 = fmaxf(mx0, __shfl_xor_sync(0xffffffff, mx0, o2));
      mx1 = fmaxf(mx1, __shfl_xor_sync(0xffffffff, mx1, o2));
    }
    float mn0 = fmaxf(m0, mx0), mn1 = fmaxf(m1, mx1);
    float c0 = mn0 == -INFINITY ? 1.f : ex2(m0 - mn0), c1 = mn1 == -INFINITY ? 1.f : ex2(m1 - mn1);
    float base0 = mn0 == -INFINITY ? 0.f : mn0, base1 = mn1 == -INFINITY ? 0.f : mn1;
    float ps0 = 0.f, ps1 = 0.f;
#pragma unroll
    for (int j = 0; j < NT; ++j) {
      s[j][0] = ex2(s[j][0] - base0);
      s[j][1] = ex2(s[j][1] - base0);
      s[j][2] = ex2(s[j][2] - base1);
      s[j][3] = ex2(s[j][3] - base1);
      ps0 += s[j][0] + s[j][1];
      ps1 += s[j][2] + s[j][3];
    }
    l0 = l0 * c0 + ps0;  // 줄 합은 끝에서 4 스레드를 모은다
    l1 = l1 * c1 + ps1;
    m0 = mn0;
    m1 = mn1;
#pragma unroll
    for (int j = 0; j < DT; ++j) {
      o[j][0] *= c0;
      o[j][1] *= c0;
      o[j][2] *= c1;
      o[j][3] *= c1;
    }
    // O += P V : P 를 A 조각으로 바꿔 키 16 개씩
#pragma unroll
    for (int st = 0; st < BK / 16; ++st) {
      uint32_t pa[4];
      pa[0] = pack2<T>(s[2 * st][0], s[2 * st][1]);
      pa[1] = pack2<T>(s[2 * st][2], s[2 * st][3]);
      pa[2] = pack2<T>(s[2 * st + 1][0], s[2 * st + 1][1]);
      pa[3] = pack2<T>(s[2 * st + 1][2], s[2 * st + 1][3]);
#pragma unroll
      for (int j = 0; j < DT; ++j) {
        uint32_t bb[2];
        bb[0] = *reinterpret_cast<const uint32_t*>(&vt[j * 8 + g][st * 16 + t4 * 2]);
        bb[1] = *reinterpret_cast<const uint32_t*>(&vt[j * 8 + g][st * 16 + 8 + t4 * 2]);
        mma<T>(o[j], pa, bb);
      }
    }
  }
#pragma unroll
  for (int o2 = 1; o2 < 4; o2 <<= 1) {
    l0 += __shfl_xor_sync(0xffffffff, l0, o2);
    l1 += __shfl_xor_sync(0xffffffff, l1, o2);
  }
  float i0 = l0 > 0.f ? 1.f / l0 : 0.f, i1 = l1 > 0.f ? 1.f / l1 : 0.f;
  T* op = a.o + b * a.ob + h * D;
#pragma unroll
  for (int j = 0; j < DT; ++j) {
    int c = j * 8 + t4 * 2;
    if (r0 < a.Lq) *reinterpret_cast<uint32_t*>(op + (long long)r0 * a.orr + c) = pack2<T>(o[j][0] * i0, o[j][1] * i0);
    if (r1 < a.Lq) *reinterpret_cast<uint32_t*>(op + (long long)r1 * a.orr + c) = pack2<T>(o[j][2] * i1, o[j][3] * i1);
  }
}

template <typename T, int D, int NW>
static void launch(const Attn<T>& a, cudaStream_t s) {
  dim3 g((a.Lq + 16 * NW - 1) / (16 * NW), a.B * a.H);
  bool m = a.kmask != nullptr, c = a.causal, bi = a.by != nullptr;
  if (bi)
    flash_k<T, D, NW, false, false, true><<<g, NW * 32, 0, s>>>(a);
  else if (m && c)
    flash_k<T, D, NW, true, true, false><<<g, NW * 32, 0, s>>>(a);
  else if (m)
    flash_k<T, D, NW, true, false, false><<<g, NW * 32, 0, s>>>(a);
  else
    flash_k<T, D, NW, false, false, false><<<g, NW * 32, 0, s>>>(a);
}
template <typename T>
void flash_attention(const Attn<T>& a, cudaStream_t s) {
  if (a.by && (a.kmask || a.causal)) {
    std::fprintf(stderr, "편향 + 마스크 조합은 없음\n");
    std::abort();
  }
  if (a.D == 64)
    launch<T, 64, 4>(a, s);
  else if (a.Lq >= 4096)
    launch<T, 32, 8>(a, s);
  else
    launch<T, 32, 4>(a, s);
}
template void flash_attention<f16>(const Attn<f16>&, cudaStream_t);
template void flash_attention<bf16>(const Attn<bf16>&, cudaStream_t);

}  // namespace k
