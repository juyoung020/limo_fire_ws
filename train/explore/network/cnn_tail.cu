// 합성곱 넷째 층(꼬리) 융합 커널 — 설명은 cnn_tail.cuh.
//
// 둘 다 채널 수(C = 셋째 층 출력, CO = 넷째 층 출력)를 컴파일 상수로 (실제 구조 32·32, 검사용 작은 구조 4·4).
// 행(= 샘플 i, 출력 픽셀 p) 마다 F3 의 왼쪽 위 칸 위치와 덧댄 행·열 여부를 한 번만 구해 네 자식 읽기·dF3 쓰기에 같이 쓴다.
// 읽기·쓰기는 채널 4 개씩(float4, C % 4 == 0).
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <stdexcept>

#include "cnn_tail.cuh"
#include "common.cuh"

namespace xnet {

namespace {
constexpr int kNT = 256;   // 블록 스레드
constexpr int kTRF = 128;  // 정방향 블록 행 수 (스레드 = 4 행 × 4 채널, 채널 묶음 CO/4 개 × 행 묶음 32 개)
constexpr int kTRB = 32;   // 역방향 타일 행 수 (스레드 = 행 묶음 8 개 × 레인 32 개)

struct TailArgs {
  const float* F3;
  int n, h, w, oh, ow, P4;
  long rows;
  const float* Wb;  // W [CO][K] 다음 b [CO]
  float* Y;
  __nv_bfloat16* Yh;  // nullptr 이면 안 씀
  long ldy;
  const float *g1, *g2;
  long ld1, ld2;
  float* dF3;
  float* part;
  bool broken;
};

// 행 → F3 의 왼쪽 위 칸 오프셋(float), 덧댐 비트 (1 = 아래 행 있음, 2 = 오른쪽 열 있음). 행이 범위 밖이면 −1.
template <int C>
__device__ __forceinline__ int row_base(const TailArgs& a, long row, int& flags, long& i_out, int& p_out) {
  if (row >= a.rows) return -1;
  const long i = row / a.P4;
  const int p = (int)(row - i * a.P4);
  const int py = p / a.ow, px = p - py * a.ow;
  const int y = 2 * py, x = 2 * px;
  flags = (y + 1 < a.h ? 1 : 0) | (x + 1 < a.w ? 2 : 0);
  i_out = i, p_out = p;
  return (int)(((i * a.h + y) * a.w + x) * C);
}
// 자식(dy, dx) 칸 오프셋 (덧댄 자리면 −1)
__device__ __forceinline__ int child_off(int base, int flags, int child, int w, int C) {
  if (base < 0) return -1;
  const int dy = child >> 1, dx = child & 1;
  if ((dy && !(flags & 1)) || (dx && !(flags & 2))) return -1;
  return base + (dy * w + dx) * C;
}

__device__ __forceinline__ float elu_f(float z) { return z > 0.f ? z : expm1f(z); }

// 정방향: 블록 = 행 kTRF 개. 자식 하나(C 채널)씩 공유 메모리 sX [행][C+1] 로 받아 누산.
template <int CO, int C>
__global__ void __launch_bounds__(kNT) k_tail_fwd(TailArgs a) {
  constexpr int K = 4 * C, CG = CO / 4, LDX = C + 1, Q = C / 4;
  static_assert(CG * (kTRF / 4) <= kNT * 4, "정방향 스레드 배치");
  __shared__ __align__(16) float sW[K * CO];  // [K][CO] (W 전치)
  __shared__ float sX[kTRF * LDX];
  __shared__ int sBase[kTRF], sFlag[kTRF];
  const int t = threadIdx.x;
  const long r0 = (long)blockIdx.x * kTRF;
  for (int e = t; e < K * CO; e += kNT) {
    const int co = e / K, k = e - co * K;
    sW[k * CO + co] = a.Wb[e];
  }
  for (int rr = t; rr < kTRF; rr += kNT) {
    int fl = 0, p;
    long i;
    sBase[rr] = row_base<C>(a, r0 + rr, fl, i, p);
    sFlag[rr] = fl;
  }
  // 계산 스레드: (cg, rg), 4 행 × 4 채널. CO = 4 면 행 묶음이 kNT 개라 앞 kTRF/4 개만 일함
  const int cg = t % CG, rg = t / CG;
  const bool comp = rg < kTRF / 4;
  float acc[4][4];
#pragma unroll
  for (int r = 0; r < 4; ++r)
#pragma unroll
    for (int c = 0; c < 4; ++c) acc[r][c] = 0.f;
  for (int child = 0; child < 4; ++child) {
    __syncthreads();
    for (int e = t; e < kTRF * Q; e += kNT) {
      const int q = e % Q, rr = e / Q;
      const int o = child_off(sBase[rr], sFlag[rr], child, a.w, C);
      float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
      if (o >= 0) v = __ldg(reinterpret_cast<const float4*>(a.F3 + o + q * 4));
      float* d = sX + rr * LDX + q * 4;
      d[0] = v.x, d[1] = v.y, d[2] = v.z, d[3] = v.w;
    }
    __syncthreads();
    if (comp) {
#pragma unroll 8
      for (int c = 0; c < C; ++c) {
        const float4 w = *reinterpret_cast<const float4*>(sW + (child * C + c) * CO + cg * 4);
        const float wc[4] = {w.x, w.y, w.z, w.w};
#pragma unroll
        for (int r = 0; r < 4; ++r) {
          const float x = sX[(rg * 4 + r) * LDX + c];
#pragma unroll
          for (int j = 0; j < 4; ++j) acc[r][j] = fmaf(x, wc[j], acc[r][j]);
        }
      }
    }
  }
  if (!comp) return;
  const float* b = a.Wb + CO * K;
#pragma unroll
  for (int r = 0; r < 4; ++r) {
    const long row = r0 + rg * 4 + r;
    if (row >= a.rows) continue;
    const long i = row / a.P4;
    const int p = (int)(row - i * a.P4);
    const long o = i * a.ldy + (long)p * CO + cg * 4;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const float v = elu_f(acc[r][j] + b[cg * 4 + j]);
      a.Y[o + j] = v;
      if (a.Yh) a.Yh[o + j] = __float2bfloat16_rn(v);
    }
  }
}

// 역방향: 상주 블록이 행 타일 kTRB 개씩 (blockIdx, +gridDim, …) 차례로, 다음 타일 입력을 레지스터에 미리 읽어 둔다.
// 스레드 (rg = t/32, 레인) 가 행 rg·4 … +3 의 X 칸 4 개(k = 레인·4 …, 레인 < C) 와 dZ 칸(채널 = 레인, 레인 < CO) 을 맡는다 —
// 읽은 X 와 같은 위치에 dF3 를 쓴다. dW 칸은 스레드 (cg, kg) 가 4 채널 × 4 k 를 레지스터에 쌓는다.
template <int CO, int C>
__global__ void __launch_bounds__(kNT) k_tail_bwd(TailArgs a) {
  constexpr int K = 4 * C, CG = CO / 4, KG = C;  // KG = K/4
  static_assert(C <= 32 && CO <= 32 && CG * KG <= kNT, "역방향 스레드 배치");
  __shared__ __align__(16) float sW[CO * K];     // [CO][K] (저장 배치 그대로)
  __shared__ __align__(16) float sX[kTRB * K];   // [행][K]
  __shared__ __align__(16) float sDZ[kTRB * CO]; // [행][CO]
  const int t = threadIdx.x, lane = t & 31, rg = t >> 5;
  for (int e = t; e < CO * K; e += kNT) sW[e] = a.Wb[e];
  const int wcg = t % CG, wkg = t / CG;
  const bool wown = wkg < KG;
  float acc[4][4], bacc[4];
#pragma unroll
  for (int c = 0; c < 4; ++c) {
    bacc[c] = 0.f;
#pragma unroll
    for (int j = 0; j < 4; ++j) acc[c][j] = 0.f;
  }
  const int child = (lane * 4) / C, cofs = (lane * 4) % C;
  const bool xl = lane < KG, zl = lane < CO;
  // 한 타일 입력 (레지스터)
  float4 xv[4];
  float gv[4], yv[4];
  int xo[4];
  auto load = [&](long tile) {
    const long r0 = tile * kTRB;
#pragma unroll
    for (int r = 0; r < 4; ++r) {
      int fl = 0, p = 0;
      long i = 0;
      const int base = row_base<C>(a, r0 + rg * 4 + r, fl, i, p);
      xo[r] = xl ? child_off(base, fl, child, a.w, C) : -1;
      if (xo[r] >= 0) xo[r] += cofs;
      xv[r] = xo[r] >= 0 ? __ldg(reinterpret_cast<const float4*>(a.F3 + xo[r])) : make_float4(0.f, 0.f, 0.f, 0.f);
      gv[r] = 0.f, yv[r] = 0.f;
      if (base >= 0 && zl) {
        const long col = (long)p * CO + lane;
        float g = __ldg(a.g1 + i * a.ld1 + col);
        if (a.g2) g = g + __ldg(a.g2 + i * a.ld2 + col);
        gv[r] = g;
        yv[r] = __ldg(a.Y + i * a.ldy + col);
      }
    }
  };
  const long ntile = (a.rows + kTRB - 1) / kTRB;
  long tile = blockIdx.x;
  if (tile < ntile) load(tile);
  for (; tile < ntile; tile += gridDim.x) {
    __syncthreads();  // 앞 타일을 다 읽음 (첫 번엔 sW 쓰기 끝)
    int xo_cur[4];
#pragma unroll
    for (int r = 0; r < 4; ++r) {
      const int rr = rg * 4 + r;
      if (xl) *reinterpret_cast<float4*>(sX + rr * K + lane * 4) = xv[r];
      if (zl) sDZ[rr * CO + lane] = gv[r] * ((yv[r] > 0.f || a.broken) ? 1.f : yv[r] + 1.f);
      xo_cur[r] = xo[r];
    }
    __syncthreads();
    if (tile + gridDim.x < ntile) load(tile + gridDim.x);  // 다음 타일 읽기를 계산과 겹침
    if (wown) {
#pragma unroll 4
      for (int rr = 0; rr < kTRB; ++rr) {
        const float4 d = *reinterpret_cast<const float4*>(sDZ + rr * CO + wcg * 4);
        const float4 x = *reinterpret_cast<const float4*>(sX + rr * K + wkg * 4);
        const float dc[4] = {d.x, d.y, d.z, d.w}, xj[4] = {x.x, x.y, x.z, x.w};
#pragma unroll
        for (int c = 0; c < 4; ++c) {
#pragma unroll
          for (int j = 0; j < 4; ++j) acc[c][j] = fmaf(dc[c], xj[j], acc[c][j]);
          bacc[c] += dc[c];
        }
      }
    }
    if (xl) {  // dF3 = dZ·W (행 4 개 × k 4 개), 읽은 X 와 같은 자리에
      float s[4][4];
#pragma unroll
      for (int r = 0; r < 4; ++r)
#pragma unroll
        for (int j = 0; j < 4; ++j) s[r][j] = 0.f;
#pragma unroll 8
      for (int co = 0; co < CO; ++co) {
        const float4 w = *reinterpret_cast<const float4*>(sW + co * K + lane * 4);
        const float wj[4] = {w.x, w.y, w.z, w.w};
#pragma unroll
        for (int r = 0; r < 4; ++r) {
          const float d = sDZ[(rg * 4 + r) * CO + co];
#pragma unroll
          for (int j = 0; j < 4; ++j) s[r][j] = fmaf(d, wj[j], s[r][j]);
        }
      }
#pragma unroll
      for (int r = 0; r < 4; ++r)
        if (xo_cur[r] >= 0) *reinterpret_cast<float4*>(a.dF3 + xo_cur[r]) = make_float4(s[r][0], s[r][1], s[r][2], s[r][3]);
    }
  }
  // 블록 부분합: [블록][CO·K (dW) | CO (db)]
  float* P = a.part + (long)blockIdx.x * (CO * K + CO);
  if (wown) {
#pragma unroll
    for (int c = 0; c < 4; ++c) {
      const int co = wcg * 4 + c;
      *reinterpret_cast<float4*>(P + (long)co * K + wkg * 4) = make_float4(acc[c][0], acc[c][1], acc[c][2], acc[c][3]);
      if (wkg == 0) P[CO * K + co] = bacc[c];
    }
  }
}

// 블록 순서대로 더해 기울기에 누적
__global__ void k_tail_fin(const float* part, int nb, int len, float* g) {
  const int j = blockIdx.x * blockDim.x + threadIdx.x;
  if (j >= len) return;
  float s = 0.f;
  for (int b = 0; b < nb; ++b) s += part[(long)b * len + j];
  g[j] += s;
}

struct Kern {
  void (*fwd)(TailArgs);
  void (*bwd)(TailArgs);
};
Kern kern(const TailShape& s) {
  if (s.cout == 32 && s.C == 32) return {k_tail_fwd<32, 32>, k_tail_bwd<32, 32>};
  if (s.cout == 4 && s.C == 4) return {k_tail_fwd<4, 4>, k_tail_bwd<4, 4>};
  return {nullptr, nullptr};
}

// 역방향 블록 수 = SM 수 × 상주 블록 수 (기기마다 고정 → 같은 기기·같은 n 이면 부분합 나누기가 같아 결정적)
int grid_bwd(void (*k)(TailArgs)) {
  static int g32 = 0, g4 = 0;
  int& g = k == k_tail_bwd<32, 32> ? g32 : g4;
  if (!g) {
    int dev = 0, sms = 0, o = 0;
    CK(cudaGetDevice(&dev));
    CK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev));
    CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&o, k, kNT, 0));
    if (o < 1) throw std::runtime_error("cnn_tail: 상주 블록 없음");
    g = o * sms;
  }
  return g;
}

TailArgs make_args(const TailShape& s) {
  TailArgs a{};
  a.n = s.n, a.h = s.h, a.w = s.w, a.oh = s.oh, a.ow = s.ow;
  a.P4 = s.oh * s.ow;
  a.rows = (long)s.n * a.P4;
  return a;
}
}  // namespace

bool cnn_tail_supported(const TailShape& s) {
  return kern(s).fwd && s.oh == (s.h + 1) / 2 && s.ow == (s.w + 1) / 2 && (long)s.n * s.h * s.w * s.C < (1L << 31);
}

long cnn_tail_part_floats(const TailShape& s) {
  const Kern k = kern(s);
  if (!k.bwd) return 0;
  return (long)grid_bwd(k.bwd) * (s.cout * 4 * s.C + s.cout);
}

void cnn_tail_forward(const TailShape& s, const float* F3, const float* Wb, float* Y, long ldy, cudaStream_t st, unsigned short* Yh) {
  if (!cnn_tail_supported(s)) throw std::runtime_error("cnn_tail_forward: 지원 안 하는 모양");
  TailArgs a = make_args(s);
  if (a.rows == 0) return;
  a.F3 = F3, a.Wb = Wb, a.Y = Y, a.ldy = ldy;
  a.Yh = reinterpret_cast<__nv_bfloat16*>(Yh);
  kern(s).fwd<<<(int)((a.rows + kTRF - 1) / kTRF), kNT, 0, st>>>(a);
  CK(cudaGetLastError());
}

void cnn_tail_backward(const TailShape& s, const float* F3, const float* Wb, const float* Y, long ldy, const float* g1, long ld1,
                       const float* g2, long ld2, float* dF3, float* gWb, float* part, bool broken_elu, cudaStream_t st) {
  if (!cnn_tail_supported(s)) throw std::runtime_error("cnn_tail_backward: 지원 안 하는 모양");
  TailArgs a = make_args(s);
  if (a.rows == 0) return;
  a.F3 = F3, a.Wb = Wb, a.Y = const_cast<float*>(Y), a.ldy = ldy;
  a.g1 = g1, a.ld1 = ld1, a.g2 = g2, a.ld2 = ld2;
  a.dF3 = dF3, a.part = part, a.broken = broken_elu;
  const Kern k = kern(s);
  const int grid = grid_bwd(k.bwd);
  k.bwd<<<grid, kNT, 0, st>>>(a);
  CK(cudaGetLastError());
  const int len = s.cout * 4 * s.C + s.cout;
  k_tail_fin<<<nblk(len, 128), 128, 0, st>>>(part, grid, len, gWb);
  CK(cudaGetLastError());
}

}  // namespace xnet
