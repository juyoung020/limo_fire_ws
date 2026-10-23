// PSABlock 의 어텐션 핵심(Ultralytics Attention.forward 와 같음):
//   qkv [B, N, nh·(2kd+hd)] (헤드마다 q kd | k kd | v hd 채널) →
//   o[n, h·hd + d] = Σ_m softmax_m(scale·q_n·k_m) v[m, d],  vf[n, h·hd + d] = v[n, d] (pe 합성곱 입력)
// GEMM 은 cuBLAS 묶음(영상 × 헤드), softmax 는 줄마다 워프 하나.
#include <cfloat>
#include <vector>

#include "kern.cuh"
#include "net.h"

namespace {
__global__ void softmax_k(const float* S, act_t* P, int rows, int N) {
  int r = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32, lane = threadIdx.x & 31;
  if (r >= rows) return;
  const float* s = S + int64_t(r) * N;
  float m = -FLT_MAX;
  for (int j = lane; j < N; j += 32) m = fmaxf(m, s[j]);
  for (int o = 16; o; o >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffff, m, o));
  float z = 0.f;
  for (int j = lane; j < N; j += 32) z += __expf(s[j] - m);
  for (int o = 16; o; o >>= 1) z += __shfl_xor_sync(0xffffffff, z, o);
  float iz = 1.f / z;
  for (int j = lane; j < N; j += 32) P[int64_t(r) * N + j] = from_f<act_t>(__expf(s[j] - m) * iz);
}
// dS = P ⊙ (dP − Σ_j P·dP)
__global__ void softmax_bwd_k(const act_t* P, const float* dP, act_t* dS, int rows, int N) {
  int r = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32, lane = threadIdx.x & 31;
  if (r >= rows) return;
  const act_t* p = P + int64_t(r) * N;
  const float* d = dP + int64_t(r) * N;
  float t = 0.f;
  for (int j = lane; j < N; j += 32) t += to_f(p[j]) * d[j];
  for (int o = 16; o; o >>= 1) t += __shfl_xor_sync(0xffffffff, t, o);
  for (int j = lane; j < N; j += 32) dS[int64_t(r) * N + j] = from_f<act_t>(to_f(p[j]) * (d[j] - t));
}

// 행 우선 C[M,N] = α·op(A)·op(B) + β·C, 묶음(포인터 배열)
void gemm_rm(cublasHandle_t h, bool tA, bool tB, int M, int N, int K, float alpha, const void* const* A, int lda,
             cudaDataType_t ta, const void* const* B, int ldb, cudaDataType_t tb, float beta, void* const* C, int ldc,
             cudaDataType_t tc, int batch) {
  CBL(cublasGemmBatchedEx(h, tB ? CUBLAS_OP_T : CUBLAS_OP_N, tA ? CUBLAS_OP_T : CUBLAS_OP_N, N, M, K, &alpha, B, tb, ldb,
                          A, ta, lda, &beta, C, tc, ldc, batch, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
}

class OpAttn : public Op {
 public:
  Tensor *qkv, *o, *vf;
  int nh, kd, hd, N, B, Cq, C;
  float scale;
  cublasHandle_t blas;
  cudaStream_t s;
  float* S;   // [B·nh, N, N] 점수, 역방향에서는 dP
  act_t* P;   // softmax
  act_t* dS;  // 역방향
  // 포인터 배열(장치): 0 Q 1 K 2 V 3 S 4 P 5 O 6 dO 7 dQ 8 dK 9 dV 10 dS
  void** ptr[11];
  OpAttn(Tensor* q, Tensor* out, Tensor* v, int nh_, int kd_, int hd_, cublasHandle_t b, cudaStream_t st)
      : qkv(q), o(out), vf(v), nh(nh_), kd(kd_), hd(hd_), blas(b), s(st) {
    N = q->H * q->W;
    B = q->N;
    Cq = q->C;
    C = o->C;
    scale = 1.f / std::sqrt((float)kd);
    int64_t nn = int64_t(B) * nh * N * N;
    S = dalloc<float>(nn);
    P = dalloc<act_t>(nn);
    dS = dalloc<act_t>(nn);
    for (auto& p : ptr) p = dalloc<void*>(B * nh);
    std::vector<void*> h[11];
    int qs = 2 * kd + hd;
    for (int b = 0; b < B; ++b)
      for (int hh = 0; hh < nh; ++hh) {
        act_t* qb = q->x + int64_t(b) * N * Cq + hh * qs;
        h[0].push_back(qb);
        h[1].push_back(qb + kd);
        h[2].push_back(qb + 2 * kd);
        int64_t so = (int64_t(b) * nh + hh) * N * N;
        h[3].push_back(S + so);
        h[4].push_back(P + so);
        h[5].push_back(out->x + int64_t(b) * N * C + hh * hd);
        h[10].push_back(dS + so);
      }
    for (int k : {0, 1, 2, 3, 4, 5, 10})
      CK(cudaMemcpy(ptr[k], h[k].data(), sizeof(void*) * B * nh, cudaMemcpyHostToDevice));
  }
  // 기울기 주소는 Net::finalize 뒤에 정해지므로 처음 역방향 때 만든다
  bool gptr_ready = false;
  void make_gptr() {
    std::vector<void*> h[11];
    int qs = 2 * kd + hd;
    for (int b = 0; b < B; ++b)
      for (int hh = 0; hh < nh; ++hh) {
        h[6].push_back(o->g + int64_t(b) * N * C + hh * hd);
        act_t* qg = qkv->g + int64_t(b) * N * Cq + hh * qs;
        h[7].push_back(qg);
        h[8].push_back(qg + kd);
        h[9].push_back(qg + 2 * kd);
      }
    for (int k : {6, 7, 8, 9}) CK(cudaMemcpy(ptr[k], h[k].data(), sizeof(void*) * B * nh, cudaMemcpyHostToDevice));
    gptr_ready = true;
  }
  void fwd(bool) override {
    int bt = B * nh;
    gemm_rm(blas, false, true, N, N, kd, scale, (const void* const*)ptr[0], Cq, kActCuda, (const void* const*)ptr[1],
            Cq, kActCuda, 0.f, ptr[3], N, CUDA_R_32F, bt);
    int rows = bt * N;
    softmax_k<<<cdiv(rows, 8), 256, 0, s>>>(S, P, rows, N);
    gemm_rm(blas, false, false, N, hd, N, 1.f, (const void* const*)ptr[4], N, kActCuda, (const void* const*)ptr[2], Cq,
            kActCuda, 0.f, ptr[5], C, kActCuda, bt);
    for (int hh = 0; hh < nh; ++hh)
      kern::copy_ch(qkv->x, Cq, hh * (2 * kd + hd) + 2 * kd, vf->x, C, hh * hd, hd, qkv->M(), false, s);
  }
  void bwd() override {
    if (!gptr_ready) make_gptr();
    int bt = B * nh, rows = bt * N;
    for (int hh = 0; hh < nh; ++hh)
      kern::copy_ch(vf->g, C, hh * hd, qkv->g, Cq, hh * (2 * kd + hd) + 2 * kd, hd, qkv->M(), true, s);
    // dP = dO·Vᵀ (S 버퍼에)
    gemm_rm(blas, false, true, N, N, hd, 1.f, (const void* const*)ptr[6], C, kActCuda, (const void* const*)ptr[2], Cq,
            kActCuda, 0.f, ptr[3], N, CUDA_R_32F, bt);
    // dV += Pᵀ·dO
    gemm_rm(blas, true, false, N, hd, N, 1.f, (const void* const*)ptr[4], N, kActCuda, (const void* const*)ptr[6], C,
            kActCuda, 1.f, ptr[9], Cq, kActCuda, bt);
    softmax_bwd_k<<<cdiv(rows, 8), 256, 0, s>>>(P, S, dS, rows, N);
    // dQ += scale·dS·K,  dK += scale·dSᵀ·Q
    gemm_rm(blas, false, false, N, kd, N, scale, (const void* const*)ptr[10], N, kActCuda, (const void* const*)ptr[1],
            Cq, kActCuda, 1.f, ptr[7], Cq, kActCuda, bt);
    gemm_rm(blas, true, false, N, kd, N, scale, (const void* const*)ptr[10], N, kActCuda, (const void* const*)ptr[0],
            Cq, kActCuda, 1.f, ptr[8], Cq, kActCuda, bt);
  }
  std::string desc() const override { return "attention"; }
};
}  // namespace

std::unique_ptr<Op> make_attn_op(Tensor* qkv, Tensor* o, Tensor* vf, int nh, int kd, int hd, cublasHandle_t blas,
                                 cudaStream_t s) {
  return std::make_unique<OpAttn>(qkv, o, vf, nh, kd, hd, blas, s);
}
