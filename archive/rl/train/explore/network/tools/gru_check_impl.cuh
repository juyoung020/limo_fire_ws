// gru_check 정밀도별 실행 (XNET_FP64 를 정의하고 include 하면 fp64 판)
#include <cuda_runtime.h>

#include <vector>

#include "../actor_critic.h"
#include "gru_check.h"

namespace XNET_NS {
using namespace gruchk;
namespace {
template <class V>
void up(real* dst, const V& h) {
  std::vector<real> t(h.begin(), h.end());
  CK(cudaMemcpy(dst, t.data(), t.size() * sizeof(real), cudaMemcpyHostToDevice));
}
std::vector<double> down(const real* s, size_t n) {
  std::vector<real> t(n);
  CK(cudaMemcpy(t.data(), s, n * sizeof(real), cudaMemcpyDeviceToHost));
  return std::vector<double>(t.begin(), t.end());
}
}  // namespace
Out gru_layout() {
  Gru g;
  std::vector<ParamBlock> bl;
  long n = 0;
  g.build(NF, ND, H, bl, n, "g");
  Out o;
  o.n = n;
  o.wx = g.wx_off, o.bx = g.bx_off, o.wh = g.wh_off, o.bh = g.bh_off;
  for (auto& b : bl) o.blocks.push_back({b.name, b.off, b.n});
  return o;
}
Out gru_run(const Data& d, bool negative) {
  Out o = gru_layout();
  Gru g;
  std::vector<ParamBlock> bl;
  long n = 0;
  g.build(NF, ND, H, bl, n, "g");
  real *P, *G, *X, *H0, *RS, *OUT, *DO, *DXF;
  CK(cudaMalloc(&P, n * sizeof(real)));
  CK(cudaMalloc(&G, n * sizeof(real)));
  CK(cudaMalloc(&X, (size_t)NR * LD * sizeof(real)));
  CK(cudaMalloc(&H0, (size_t)B * H * sizeof(real)));
  CK(cudaMalloc(&RS, NR * sizeof(real)));
  CK(cudaMalloc(&OUT, (size_t)NR * LD * sizeof(real)));
  CK(cudaMalloc(&DO, (size_t)NR * LD * sizeof(real)));
  CK(cudaMalloc(&DXF, (size_t)NR * NF * sizeof(real)));
  g.bind(P, G, NR);
  cublasHandle_t cb;
  cudaStream_t st;
  CKB(cublasCreate(&cb));
  CK(cudaStreamCreate(&st));
  up(P, d.params);
  up(X, d.x);
  up(H0, d.h0);
  up(RS, d.reset);
  std::vector<double> dw((size_t)NR * LD, 0.0);
  for (int r = 0; r < NR; ++r)
    for (int j = 0; j < H; ++j) dw[(size_t)r * LD + NF + j] = d.w[(size_t)r * H + j] * (negative && r / B == 1 ? -1.0 : 1.0);
  up(DO, dw);
  CK(cudaMemset(OUT, 0, (size_t)NR * LD * sizeof(real)));
  g.forward_seq(cb, st, T, B, X, LD, OFFD, H0, RS, OUT + NF, LD, nullptr);
  CK(cudaStreamSynchronize(st));
  auto ob = down(OUT, (size_t)NR * LD);
  o.h.resize((size_t)NR * H);
  for (int r = 0; r < NR; ++r)
    for (int j = 0; j < H; ++j) o.h[(size_t)r * H + j] = ob[(size_t)r * LD + NF + j];
  CK(cudaMemset(G, 0, n * sizeof(real)));
  g.backward_seq(cb, st, T, B, X, LD, OFFD, RS, DO + NF, LD, DXF);
  CK(cudaStreamSynchronize(st));
  o.grad = down(G, n);
  o.dxf = down(DXF, (size_t)NR * NF);
  // step 걸음마다 (판 경계는 부르는 쪽이 h 를 0 으로)
  {
    std::vector<double> hb(d.h0);
    real *Hd, *Od;
    CK(cudaMalloc(&Hd, (size_t)B * H * sizeof(real)));
    CK(cudaMalloc(&Od, (size_t)B * LD * sizeof(real)));
    double es = 0;
    for (int t = 0; t < T; ++t) {
      for (int b = 0; b < B; ++b)
        if (d.reset[t * B + b] != 0)
          for (int j = 0; j < H; ++j) hb[(size_t)b * H + j] = 0;
      up(Hd, hb);
      g.step(cb, st, B, X + (long)t * B * LD, LD, OFFD, Hd, Od + NF, LD, nullptr);
      CK(cudaStreamSynchronize(st));
      hb = down(Hd, (size_t)B * H);
      for (int b = 0; b < B; ++b)
        for (int j = 0; j < H; ++j) es = std::max(es, std::fabs(hb[(size_t)b * H + j] - o.h[(size_t)(t * B + b) * H + j]));
    }
    o.step_diff = es;
    cudaFree(Hd);
    cudaFree(Od);
  }
  g.free_ws();
  for (real* p : {P, G, X, H0, RS, OUT, DO, DXF}) cudaFree(p);
  cublasDestroy(cb);
  cudaStreamDestroy(st);
  return o;
}
}  // namespace XNET_NS
