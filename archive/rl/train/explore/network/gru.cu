// GRU 한 층 — gru.cuh 머리말. 문 커널: 블록 하나 = 행 kRB 개 × 출력 칸 H (스레드 = 출력 칸 j), 이전 h 를 공유 메모리에.
#include <cuda_bf16.h>

#include <cmath>
#include <cstdlib>
#include <stdexcept>

#include "actor_critic.h"
#include "gru.cuh"

namespace XNET_NS {
namespace {

constexpr int kRB = 8;    // 앞 문 커널 블록 하나의 행 수
constexpr int kRBb = 4;   // 뒤 h 커널 블록 하나의 행 수 (공유 메모리 kRBb · 3H — fp64 · H 256 에서 24 KB)

__device__ __forceinline__ real sigm(real a) { return (real)1 / ((real)1 + exp(-a)); }

// 앞 걸음 하나: 행 r0 … r0 + B − 1 (r0 = t·B). hp_src(b) = 이전 h — t == 0 이면 h0[b], 아니면 직전 걸음 out 행 (seq) 또는 h[b] (step).
// 출력: h' → out 행·hbuf (있으면), 역방향용 문 값 (있으면)
__global__ void k_gru_fwd(int B, int H, const real* __restrict__ GX, const real* __restrict__ WhT, const real* __restrict__ bh,
                          const real* __restrict__ hprev, long ldp, const real* __restrict__ reset, real* __restrict__ out, long ldo,
                          uint16_t* __restrict__ outh, real* __restrict__ hbuf, real* HP, real* R, real* Z, real* Nn, real* GHN) {
  extern __shared__ unsigned char smem_[];
  real* sh = (real*)smem_;   // [kRB][H]
  const int j = threadIdx.x, b0 = blockIdx.x * kRB;
  for (int r = 0; r < kRB; ++r) {
    const int b = b0 + r;
    real v = 0;
    if (b < B && !(reset && reset[b] != (real)0)) v = hprev[(long)b * ldp + j];
    sh[r * H + j] = v;
  }
  __syncthreads();
  real ar[kRB], az[kRB], an[kRB];
  for (int r = 0; r < kRB; ++r) ar[r] = az[r] = an[r] = 0;
  for (int k = 0; k < H; ++k) {
    const real w0 = WhT[(long)k * 3 * H + j], w1 = WhT[(long)k * 3 * H + H + j], w2 = WhT[(long)k * 3 * H + 2 * H + j];
#pragma unroll
    for (int r = 0; r < kRB; ++r) {
      const real hv = sh[r * H + k];
      ar[r] += w0 * hv;
      az[r] += w1 * hv;
      an[r] += w2 * hv;
    }
  }
  for (int r = 0; r < kRB; ++r) {
    const int b = b0 + r;
    if (b >= B) break;
    const real* gx = GX + (long)b * 3 * H;
    const real ghn = an[r] + bh[2 * H + j];
    const real rr = sigm(gx[j] + ar[r] + bh[j]);
    const real zz = sigm(gx[H + j] + az[r] + bh[H + j]);
    const real nn = tanh(gx[2 * H + j] + rr * ghn);
    const real hp = sh[r * H + j];
    const real hn = ((real)1 - zz) * nn + zz * hp;
    out[(long)b * ldo + j] = hn;
    if (outh) outh[(long)b * ldo + j] = __bfloat16_as_ushort(__float2bfloat16_rn((float)hn));
    if (hbuf) hbuf[(long)b * H + j] = hn;
    if (HP) {
      const long o = (long)b * H + j;
      HP[o] = hp;
      R[o] = rr;
      Z[o] = zz;
      Nn[o] = nn;
      GHN[o] = ghn;
    }
  }
}

// 입력 쪽 편향을 더한다 (GEMM 뒤): GX[i][c] += bx[c]
__global__ void k_add_bias(real* GX, long n, int C, const real* bx) {
  const long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n * C) GX[i] += bx[i % C];
}

// 뒤 걸음 하나 (1): 문 기울기. dh = dout[행] + DC[b] (뒤 걸음에서 온 기울기, t = T−1 이면 0)
__global__ void k_gru_bwd_gates(int B, int H, const real* __restrict__ dout, long lddo, const real* __restrict__ DC, const real* HP, const real* R,
                                const real* Z, const real* Nn, const real* GHN, real* DGX, real* DGH, real* DIR) {
  const long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= (long)B * H) return;
  const long b = i / H;
  const int j = (int)(i % H);
  const real dh = dout[b * lddo + j] + (DC ? DC[i] : (real)0);
  const real z = Z[i], n = Nn[i], r = R[i], hp = HP[i];
  const real dn = dh * ((real)1 - z), dz = dh * (hp - n);
  const real dan = dn * ((real)1 - n * n);
  const real dr = dan * GHN[i];
  const real dar = dr * r * ((real)1 - r), daz = dz * z * ((real)1 - z);
  real* gx = DGX + b * 3 * H;
  real* gh = DGH + b * 3 * H;
  gx[j] = dar;
  gx[H + j] = daz;
  gx[2 * H + j] = dan;
  gh[j] = dar;
  gh[H + j] = daz;
  gh[2 * H + j] = dan * r;
  DIR[i] = dh * z;
}

// 뒤 걸음 하나 (2): 이전 h 기울기 DCn[b][k] = (DIR[b][k] + Σ_j WhT[k][j] · DGH[b][j]) · (1 − reset[b]) — 스레드 = (b, k), j 고정 차례
__global__ void k_gru_bwd_h(int B, int H, const real* __restrict__ WhT, const real* __restrict__ DGH, const real* __restrict__ DIR,
                            const real* __restrict__ reset, real* DCn) {
  extern __shared__ unsigned char smem_[];
  real* sg = (real*)smem_;   // [kRBb][3H]
  const int k = threadIdx.x, b0 = blockIdx.x * kRBb;
  for (int r = 0; r < kRBb; ++r) {
    const int b = b0 + r;
    for (int c = k; c < 3 * H; c += blockDim.x) sg[r * 3 * H + c] = b < B ? DGH[(long)b * 3 * H + c] : (real)0;
  }
  __syncthreads();
  real acc[kRBb];
  for (int r = 0; r < kRBb; ++r) acc[r] = 0;
  const real* w = WhT + (long)k * 3 * H;
  for (int c = 0; c < 3 * H; ++c) {
    const real wv = w[c];
#pragma unroll
    for (int r = 0; r < kRBb; ++r) acc[r] += wv * sg[r * 3 * H + c];
  }
  for (int r = 0; r < kRBb; ++r) {
    const int b = b0 + r;
    if (b >= B) break;
    const real keep = reset && reset[b] != (real)0 ? (real)0 : (real)1;
    DCn[(long)b * H + k] = (DIR[(long)b * H + k] + acc[r]) * keep;
  }
}

// ---- 빠른 길 (2026-11-15): 걸음마다 순환 곱을 cuBLAS GEMM 하나로 (모든 SM), 나머지는 원소별 커널 하나 ----
// 앞: HP[b] = reset[b] ? 0 : hprev[b] (t = 0 걸음 — 다음 걸음부터는 문 커널이 미리 씀)
__global__ void k_gru_hp0(int B, int H, const real* __restrict__ hprev, long ldp, const real* __restrict__ reset, real* __restrict__ HP) {
  const long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= (long)B * H) return;
  const long b = i / H;
  const int j = (int)(i % H);
  HP[i] = (reset && reset[b] != (real)0) ? (real)0 : hprev[b * ldp + j];
}
// 앞 문 (원소별): GH = HP·Wh (GEMM 결과). h' → out (·outh·hbuf), 역방향용 R·Z·N·GHN, 다음 걸음 HP (reset_next 면 0)
__global__ void k_gru_gate(int B, int H, const real* __restrict__ GX, const real* __restrict__ GH, const real* __restrict__ bh, const real* __restrict__ HP,
                           real* __restrict__ out, long ldo, uint16_t* __restrict__ outh, real* __restrict__ hbuf, real* R, real* Z, real* Nn, real* GHN,
                           real* HPnext, const real* __restrict__ reset_next) {
  const long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= (long)B * H) return;
  const long b = i / H;
  const int j = (int)(i % H);
  const real* gx = GX + b * 3 * H;
  const real* gh = GH + b * 3 * H;
  const real ghn = gh[2 * H + j] + bh[2 * H + j];
  const real rr = sigm(gx[j] + gh[j] + bh[j]);
  const real zz = sigm(gx[H + j] + gh[H + j] + bh[H + j]);
  const real nn = tanh(gx[2 * H + j] + rr * ghn);
  const real hp = HP[i];
  const real hn = ((real)1 - zz) * nn + zz * hp;
  out[b * ldo + j] = hn;
  if (outh) outh[b * ldo + j] = __bfloat16_as_ushort(__float2bfloat16_rn((float)hn));
  if (hbuf) hbuf[i] = hn;
  if (R) {
    R[i] = rr;
    Z[i] = zz;
    Nn[i] = nn;
    GHN[i] = ghn;
  }
  if (HPnext) HPnext[i] = (reset_next && reset_next[b] != (real)0) ? (real)0 : hn;
}
// 뒤 문 (원소별): dh = dout + DC·keep (DC = 뒤 걸음에서 온 이전 h 기울기, keep = 1 − reset[뒤 걸음]). DC 칸(= 이 걸음의 DIR)에 dh·z 를 써 두면
// 이전 h 기울기 GEMM (DGH·Whᵀ, β = 1) 이 그 위에 더한다
__global__ void k_gru_bgate(int B, int H, const real* __restrict__ dout, long lddo, const real* __restrict__ DC, const real* __restrict__ reset_next,
                            const real* HP, const real* R, const real* Z, const real* Nn, const real* GHN, real* DGX, real* DGH, real* DIRo) {
  const long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= (long)B * H) return;
  const long b = i / H;
  const int j = (int)(i % H);
  real dh = dout[b * lddo + j];
  if (DC && !(reset_next && reset_next[b] != (real)0)) dh += DC[i];
  const real z = Z[i], n = Nn[i], r = R[i], hp = HP[i];
  const real dn = dh * ((real)1 - z), dz = dh * (hp - n);
  const real dan = dn * ((real)1 - n * n);
  const real dr = dan * GHN[i];
  const real dar = dr * r * ((real)1 - r), daz = dz * z * ((real)1 - z);
  real* gx = DGX + b * 3 * H;
  real* gh = DGH + b * 3 * H;
  gx[j] = dar;
  gx[H + j] = daz;
  gx[2 * H + j] = dan;
  gh[j] = dar;
  gh[H + j] = daz;
  gh[2 * H + j] = dan * r;
  DIRo[i] = dh * z;
}

}  // namespace

void Gru::build(int nf_, int nd_, int H_, std::vector<ParamBlock>& reg, long& cur, const std::string& pfx) {
  nf = nf_, nd = nd_, H = H_;
  if (H < 32 || H > 1024 || H % 32) throw std::runtime_error("GRU 은닉 크기는 32 의 배수, 32 ~ 1024");
  const long nx = (long)3 * H * (nf + nd);
  wx_off = cur;
  reg.push_back({cur, nx, PKind::LinW, pfx + ".wx"});
  cur += nx;
  bx_off = cur;
  reg.push_back({cur, 3L * H, PKind::LinB, pfx + ".bx"});
  cur += 3L * H;
  wh_off = cur;
  reg.push_back({cur, 3L * H * H, PKind::LinW, pfx + ".wh"});
  cur += 3L * H * H;
  bh_off = cur;
  reg.push_back({cur, 3L * H, PKind::LinB, pfx + ".bh"});
  cur += 3L * H;
}

void Gru::bind(real* params, real* grads, int max_rows) {
  P_ = params;
  G_ = grads;
  max_rows_ = max_rows;
  const long R = max_rows;
  auto al = [&](real** p, long n) { CK(cudaMalloc(p, (size_t)n * sizeof(real))); };
  al(&GX_, R * 3 * H);
  al(&HP_, R * H);
  al(&R_, R * H);
  al(&Z_, R * H);
  al(&N_, R * H);
  al(&GHN_, R * H);
  al(&DGX_, R * 3 * H);
  al(&DGH_, R * 3 * H);
  al(&DC_, R * H);
  al(&DIR_, R * H);
  al(&dXF_, R * nf);
  al(&GH_, R * 3 * H);
  ref_ = std::getenv("GRU_REF") != nullptr && std::atoi(std::getenv("GRU_REF")) != 0;
}

void Gru::free_ws() {
  for (real* p : {GX_, HP_, R_, Z_, N_, GHN_, DGX_, DGH_, DC_, DIR_, dXF_, GH_}) cudaFree(p);
  GX_ = HP_ = R_ = Z_ = N_ = GHN_ = DGX_ = DGH_ = DC_ = DIR_ = dXF_ = GH_ = nullptr;
}

void Gru::init(std::mt19937_64& rng, std::vector<real>& host) const {
  const float bd = 1.f / std::sqrt((float)H);   // PyTorch nn.GRU reset_parameters
  std::uniform_real_distribution<float> U(-bd, bd);
  for (long o : {wx_off, bx_off, wh_off, bh_off}) {
    const long n = o == wx_off ? 3L * H * (nf + nd) : o == wh_off ? 3L * H * H : 3L * H;
    for (long i = 0; i < n; ++i) host[o + i] = U(rng);
  }
}

void Gru::input_gemm(cublasHandle_t cb, cudaStream_t st, int n, const real* x, long ldx, int off_d) {
  CKB(cublasSetStream(cb, st));
  const int ld = nf + nd;
  gemm(cb, false, true, n, 3 * H, nf, (real)1, x, (int)ldx, P_ + wx_off, ld, (real)0, GX_, 3 * H);
  if (nd > 0) gemm(cb, false, true, n, 3 * H, nd, (real)1, x + off_d, (int)ldx, P_ + wx_off + nf, ld, (real)1, GX_, 3 * H);
  k_add_bias<<<nblk((long)n * 3 * H), 256, 0, st>>>(GX_, n, 3 * H, P_ + bx_off);
}

void Gru::step(cublasHandle_t cb, cudaStream_t st, int B, const real* x, long ldx, int off_d, real* h, real* out, long ldo, uint16_t* outh) {
  if (B > max_rows_) throw std::runtime_error("Gru::step: 행 수 > max_rows");
  input_gemm(cb, st, B, x, ldx, off_d);
  if (!ref_) {   // 빠른 길: HP = h → GH = HP·Wh (GEMM) → 문 (h 제자리)
    k_gru_hp0<<<nblk((long)B * H), 256, 0, st>>>(B, H, h, H, nullptr, HP_);
    gemm(cb, false, false, B, 3 * H, H, (real)1, HP_, H, P_ + wh_off, 3 * H, (real)0, GH_, 3 * H);
    k_gru_gate<<<nblk((long)B * H), 256, 0, st>>>(B, H, GX_, GH_, P_ + bh_off, HP_, out, ldo, outh, h, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr);
    CK(cudaGetLastError());
    return;
  }
  const size_t sm = (size_t)kRB * H * sizeof(real);
  // h 를 읽고 같은 자리에 새 h 를 쓰면 다른 블록이 아직 읽는 중일 수 있다 — 블록 하나는 자기 행만 읽고 쓰므로 안전 (읽기는 공유 메모리로 먼저 끝남)
  k_gru_fwd<<<nblk(B, kRB), H, sm, st>>>(B, H, GX_, P_ + wh_off, P_ + bh_off, h, H, nullptr, out, ldo, outh, h, nullptr, nullptr, nullptr, nullptr,
                                         nullptr);
  CK(cudaGetLastError());
}

void Gru::forward_seq(cublasHandle_t cb, cudaStream_t st, int T, int B, const real* x, long ldx, int off_d, const real* h0, const real* reset, real* out,
                      long ldo, uint16_t* outh) {
  const int n = T * B;
  if (n > max_rows_) throw std::runtime_error("Gru::forward_seq: 행 수 > max_rows");
  input_gemm(cb, st, n, x, ldx, off_d);
  if (!ref_) {   // 빠른 길: 걸음마다 GEMM (HP_t · Wh → GH_t) + 문 커널 (다음 걸음 HP 까지 씀)
    k_gru_hp0<<<nblk((long)B * H), 256, 0, st>>>(B, H, h0, H, reset, HP_);
    for (int t = 0; t < T; ++t) {
      const long r0 = (long)t * B;
      gemm(cb, false, false, B, 3 * H, H, (real)1, HP_ + r0 * H, H, P_ + wh_off, 3 * H, (real)0, GH_ + r0 * 3 * H, 3 * H);
      k_gru_gate<<<nblk((long)B * H), 256, 0, st>>>(B, H, GX_ + r0 * 3 * H, GH_ + r0 * 3 * H, P_ + bh_off, HP_ + r0 * H, out + r0 * ldo, ldo,
                                                    outh ? outh + r0 * ldo : nullptr, nullptr, R_ + r0 * H, Z_ + r0 * H, N_ + r0 * H, GHN_ + r0 * H,
                                                    t + 1 < T ? HP_ + r0 * H + (long)B * H : nullptr, t + 1 < T ? reset + r0 + B : nullptr);
    }
    CK(cudaGetLastError());
    return;
  }
  const size_t sm = (size_t)kRB * H * sizeof(real);
  for (int t = 0; t < T; ++t) {
    const long r0 = (long)t * B;
    const real* hp = t == 0 ? h0 : out + (r0 - B) * ldo;
    const long ldp = t == 0 ? H : ldo;
    k_gru_fwd<<<nblk(B, kRB), H, sm, st>>>(B, H, GX_ + r0 * 3 * H, P_ + wh_off, P_ + bh_off, hp, ldp, reset + r0, out + r0 * ldo, ldo,
                                           outh ? outh + r0 * ldo : nullptr, nullptr, HP_ + r0 * H, R_ + r0 * H, Z_ + r0 * H, N_ + r0 * H,
                                           GHN_ + r0 * H);
  }
  CK(cudaGetLastError());
}

void Gru::backward_seq(cublasHandle_t cb, cudaStream_t st, int T, int B, const real* x, long ldx, int off_d, const real* reset, const real* dout,
                       long lddo, real* dxf) {
  const int n = T * B;
  const size_t sm = (size_t)kRBb * 3 * H * sizeof(real);
  // DC_ 를 두 칸(번갈아)으로: 앞 절반 · 뒤 절반 — max_rows ≥ 2 B 이어야 (T ≥ 2). T == 1 이면 다음 칸이 필요 없음
  real* dc[2] = {DC_, DC_ + (long)B * H};
  if (T > 1 && 2 * B > max_rows_) throw std::runtime_error("Gru::backward_seq: 작업 공간 부족");
  CKB(cublasSetStream(cb, st));
  if (!ref_) {   // 빠른 길: 걸음마다 문 커널 (DIR → dc 칸) + 이전 h 기울기 GEMM (dc += DGH·Whᵀ). 판 경계(reset)는 다음 문 커널이 dc 를 버림
    for (int t = T - 1; t >= 0; --t) {
      const long r0 = (long)t * B;
      const real* dcin = t == T - 1 ? nullptr : dc[(t + 1) & 1];
      real* dcout = dc[t & 1];
      k_gru_bgate<<<nblk((long)B * H), 256, 0, st>>>(B, H, dout + r0 * lddo, lddo, dcin, t == T - 1 ? nullptr : reset + r0 + B, HP_ + r0 * H, R_ + r0 * H,
                                                     Z_ + r0 * H, N_ + r0 * H, GHN_ + r0 * H, DGX_ + r0 * 3 * H, DGH_ + r0 * 3 * H, dcout);
      if (t > 0) gemm(cb, false, true, B, H, 3 * H, (real)1, DGH_ + r0 * 3 * H, 3 * H, P_ + wh_off, 3 * H, (real)1, dcout, H);
    }
    CK(cudaGetLastError());
  } else
  for (int t = T - 1; t >= 0; --t) {
    const long r0 = (long)t * B;
    const real* dcin = t == T - 1 ? nullptr : dc[(t + 1) & 1];
    k_gru_bwd_gates<<<nblk((long)B * H), 256, 0, st>>>(B, H, dout + r0 * lddo, lddo, dcin, HP_ + r0 * H, R_ + r0 * H, Z_ + r0 * H, N_ + r0 * H,
                                                       GHN_ + r0 * H, DGX_ + r0 * 3 * H, DGH_ + r0 * 3 * H, DIR_ + r0 * H);
    if (t > 0)
      k_gru_bwd_h<<<nblk(B, kRBb), H, sm, st>>>(B, H, P_ + wh_off, DGH_ + r0 * 3 * H, DIR_ + r0 * H, reset + r0, dc[t & 1]);
  }
  CK(cudaGetLastError());
  CKB(cublasSetStream(cb, st));
  const int ld = nf + nd;
  // 변수 기울기 (모든 걸음 한 번에): dWhT [H][3H] += HP^T · DGH, dWx [3H][nf | nd] += DGX^T · X, 편향 = 열 합
  gemm(cb, true, false, H, 3 * H, n, (real)1, HP_, H, DGH_, 3 * H, (real)1, G_ + wh_off, 3 * H);
  gemm(cb, true, false, 3 * H, nf, n, (real)1, DGX_, 3 * H, x, (int)ldx, (real)1, G_ + wx_off, ld);
  if (nd > 0) gemm(cb, true, false, 3 * H, nd, n, (real)1, DGX_, 3 * H, x + off_d, (int)ldx, (real)1, G_ + wx_off + nf, ld);
  col_sum(DGH_, n, 3 * H, 3 * H, G_ + bh_off, true, st);
  col_sum(DGX_, n, 3 * H, 3 * H, G_ + bx_off, true, st);
  // F 열 입력 기울기 (인코더로): dxf [n][nf] = DGX · Wx[:, :nf]
  if (dxf) gemm(cb, false, false, n, nf, 3 * H, (real)1, DGX_, 3 * H, P_ + wx_off, ld, (real)0, dxf, nf);
}

}  // namespace XNET_NS
