// 계층형 정책 머리 — slot_head.cuh 설명
#include "slot_head.cuh"

namespace XNET_NS {

namespace {
__device__ __forceinline__ real elu_s(real z) { return z > (real)0 ? z : expm1(z); }

// 특징 모으기: F [n·K][C] = feat 의 슬롯 칸
__global__ void k_slot_gather(const real* feat, int ldf, int foff, int n, int K, int C, real* F) {
  long t = blockIdx.x * (long)blockDim.x + threadIdx.x;
  if (t >= (long)n * K * C) return;
  const int i = (int)(t / (K * C)), r = (int)(t % (K * C));
  F[t] = feat[(long)i * ldf + foff + r];
}
// 슬롯 (i, k) 마다: A = W1 [z ; f] + b1 (h 마다), Hs = ELU(A), 로짓 = w2 · Hs. 더하는 순서 고정
__global__ void k_slot_fwd(const real* z, int ldz, const real* F, const real* P, long w1, long b1, long w2, int n, int K, int E,
                           int C, int H, real* A, real* Hs, real* logits) {
  long t = blockIdx.x * (long)blockDim.x + threadIdx.x;
  if (t >= (long)n * K) return;
  const int i = (int)(t / K);
  const real* zi = z + (long)i * ldz;
  const real* f = F + t * C;
  real s = 0;
  for (int h = 0; h < H; ++h) {
    const real* W = P + w1 + (long)h * (E + C);
    real a = P[b1 + h];
    for (int e = 0; e < E; ++e) a += W[e] * zi[e];
    for (int c = 0; c < C; ++c) a += W[E + c] * f[c];
    const real y = elu_s(a);
    A[t * H + h] = a;
    Hs[t * H + h] = y;
    s += P[w2 + h] * y;
  }
  logits[t] = s;
}
// dA[i,k,h] = G[i,k] · w2[h] · ELU'(A) ;  dU[i,h] = Σ_k dA (k 차례)
__global__ void k_slot_bwd1(const real* G, const real* A, const real* P, long w2, int n, int K, int H, real* dA, real* dU) {
  long t = blockIdx.x * (long)blockDim.x + threadIdx.x;
  if (t >= (long)n * H) return;
  const int i = (int)(t / H), h = (int)(t % H);
  real u = 0;
  for (int k = 0; k < K; ++k) {
    const long q = ((long)i * K + k) * H + h;
    const real a = A[q];
    const real d = G[(long)i * K + k] * P[w2 + h] * (a > (real)0 ? (real)1 : exp(a));
    dA[q] = d;
    u += d;
  }
  dU[t] = u;
}
// dz[i, e] = Σ_h dU[i, h] · W1[h][e] (h 차례)
__global__ void k_slot_dz(const real* dU, const real* P, long w1, int n, int E, int C, int H, real* dz, int lddz) {
  long t = blockIdx.x * (long)blockDim.x + threadIdx.x;
  if (t >= (long)n * E) return;
  const int i = (int)(t / E), e = (int)(t % E);
  real s = 0;
  for (int h = 0; h < H; ++h) s += dU[(long)i * H + h] * P[w1 + (long)h * (E + C) + e];
  dz[(long)i * lddz + e] = s;
}
}  // namespace

void SlotHead::alloc(int mn) {
  max_n = mn;
  CK(cudaMalloc(&F, sizeof(real) * (size_t)mn * K * C));
  CK(cudaMalloc(&A, sizeof(real) * (size_t)mn * K * H));
  CK(cudaMalloc(&Hs, sizeof(real) * (size_t)mn * K * H));
  CK(cudaMalloc(&dA, sizeof(real) * (size_t)mn * K * H));
  CK(cudaMalloc(&dU, sizeof(real) * (size_t)mn * H));
}
void SlotHead::release() {
  for (real* p : {F, A, Hs, dA, dU}) cudaFree(p);
  F = A = Hs = dA = dU = nullptr;
}

void SlotHead::forward(cudaStream_t st, int n, const real* z, int ldz, const real* feat, int ldf, int foff, const real* P, real* logits) {
  k_slot_gather<<<nblk((long)n * K * C), 256, 0, st>>>(feat, ldf, foff, n, K, C, F);
  k_slot_fwd<<<nblk((long)n * K, 128), 128, 0, st>>>(z, ldz, F, P, w1_off, b1_off, w2_off, n, K, E, C, H, A, Hs, logits);
}

void SlotHead::backward(cublasHandle_t h, cudaStream_t st, int n, const real* z, int ldz, const real* P, real* grads, const real* G_, real* dz,
                        int lddz) {
  k_slot_bwd1<<<nblk((long)n * H), 256, 0, st>>>(G_, A, P, w2_off, n, K, H, dA, dU);
  k_slot_dz<<<nblk((long)n * E), 256, 0, st>>>(dU, P, w1_off, n, E, C, H, dz, lddz);
  CKB(cublasSetStream(h, st));
  // dW1 앞 E 열 += dU^T · Z,  뒤 C 열 += dA^T · F   (행 우선 [H][E + C])
  gemm(h, true, false, H, E, n, (real)1, dU, H, z, ldz, (real)1, grads + w1_off, E + C);
  gemm(h, true, false, H, C, n * K, (real)1, dA, H, F, C, (real)1, grads + w1_off + E, E + C);
  col_sum(dU, n, H, H, grads + b1_off, true, st);
  // dw2 += G^T · Hs (G [n·K][1])
  gemm(h, true, false, 1, H, n * K, (real)1, G_, 1, Hs, H, (real)1, grads + w2_off, H);
}

}  // namespace XNET_NS
