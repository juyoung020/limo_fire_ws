// tools/graph_net_check 가 쓰는 GPU 쪽: 그래프 몸통 + 포인터 머리를 PPO 없이 직접 돌린다 (fp32·fp64 같은 소스, 같은 씨앗 → 같은 가중치).
#pragma once

#include <cmath>
#include <cstring>
#include <random>
#include <vector>

#include "actor_critic.h"
#include "gc_api.h"

namespace XPPO_NS {

using namespace XNET_NS;

GkOut gk_run(const GkSpec& gs, const GkCase& c, const std::vector<double>& coef, bool perturb) {
  const int n = c.n, N = c.gw * c.gh, d = gs.d;
  ACSpec spec;
  spec.graph = true;
  spec.gspec = GraphSpec{c.gw, c.gh, gs.feat, gs.d, gs.heads, gs.layers, gs.ff};
  spec.graph_glob = gs.glob;
  spec.graph_share = true;   // 가치 쪽은 안 돌린다 (가치 몸통은 같은 GraphBody 코드, 가치 머리는 기존 MLP — 기울기 검사가 본다)
  spec.graph_rows = (long)n * N;
  NetSpec cr;
  cr.hidden = {4};
  cr.out = 1;
  spec.critic = cr;
  spec.n_cont = 0;
  ActorCritic ac(spec, 4321, n);
  const long NP = ac.n_params;
  std::vector<real> params(NP);
  CK(cudaMemcpy(params.data(), ac.d_params, NP * sizeof(real), cudaMemcpyDeviceToHost));
  if (perturb) {
    std::mt19937 rng(99);
    std::normal_distribution<float> G(0.f, 1.f);
    for (auto& v : params) {
      float f = (float)v;
      f += 0.15f * G(rng) * (std::fabs(f) + 0.05f);
      v = f;
    }
    CK(cudaMemcpy(ac.d_params, params.data(), NP * sizeof(real), cudaMemcpyHostToDevice));
    ac.params_changed();
  }
  uint8_t* df;
  uint32_t* de;
  int *dc, *dcnt;
  real *dg = nullptr, *dl, *dcoef;
  CK(cudaMalloc(&df, c.feat.size()));
  CK(cudaMalloc(&de, c.edge.size() * 4));
  CK(cudaMalloc(&dc, n * 4));
  CK(cudaMalloc(&dcnt, n * 4));
  CK(cudaMalloc(&dl, (size_t)n * kGNb * sizeof(real)));
  CK(cudaMalloc(&dcoef, (size_t)n * kGNb * sizeof(real)));
  CK(cudaMemcpy(df, c.feat.data(), c.feat.size(), cudaMemcpyHostToDevice));
  CK(cudaMemcpy(de, c.edge.data(), c.edge.size() * 4, cudaMemcpyHostToDevice));
  CK(cudaMemcpy(dc, c.cur.data(), n * 4, cudaMemcpyHostToDevice));
  if (gs.glob) {
    std::vector<real> g(c.glob.begin(), c.glob.end());
    CK(cudaMalloc(&dg, g.size() * sizeof(real)));
    CK(cudaMemcpy(dg, g.data(), g.size() * sizeof(real), cudaMemcpyHostToDevice));
  }
  const GraphIn in{df, de, dc, nullptr};
  std::vector<int> cnt(n);
  auto st = ac.stream;
  graph_count(spec.gspec, st, n, in, dcnt, cnt.data());
  ac.gbody.forward(ac.cublas, st, n, in, cnt.data());
  ac.ptr.forward(ac.cublas, st, n, ac.gbody.E(), ac.gbody.Hc(), dg, gs.glob, ac.gbody.Hn(), ac.d_params, dl);
  CK(cudaStreamSynchronize(st));
  GkOut o;
  o.rows = ac.gbody.rows();
  std::vector<real> hl((size_t)n * kGNb), hE((size_t)n * d), hH((size_t)n * d);
  std::vector<int> nb((size_t)n * kGNb);
  CK(cudaMemcpy(hl.data(), dl, hl.size() * sizeof(real), cudaMemcpyDeviceToHost));
  CK(cudaMemcpy(hE.data(), ac.gbody.E(), hE.size() * sizeof(real), cudaMemcpyDeviceToHost));
  CK(cudaMemcpy(hH.data(), ac.gbody.Hc(), hH.size() * sizeof(real), cudaMemcpyDeviceToHost));
  CK(cudaMemcpy(nb.data(), ac.gbody.nbrow(), nb.size() * 4, cudaMemcpyDeviceToHost));
  o.logits.assign(hl.begin(), hl.end());
  o.E.assign(hE.begin(), hE.end());
  o.Hc.assign(hH.begin(), hH.end());
  for (int v : nb) o.mask.push_back(v >= 0 ? 1.0 : 0.0);
  for (const auto* v : {&hl, &hE, &hH}) {
    const unsigned char* b = (const unsigned char*)v->data();
    o.bits.insert(o.bits.end(), b, b + v->size() * sizeof(real));
  }
  o.params.assign(params.begin(), params.end());
  for (auto& b : ac.blocks) o.blocks.push_back({b.off, b.n, (int)b.kind, b.name});
  if (!coef.empty()) {
    std::vector<real> hc(coef.begin(), coef.end());
    CK(cudaMemcpy(dcoef, hc.data(), hc.size() * sizeof(real), cudaMemcpyHostToDevice));
    ac.zero_grad();
    ac.ptr.backward(ac.cublas, st, n, ac.gbody.Hn(), ac.d_params, ac.d_grads, dcoef);
    ac.gbody.backward(ac.cublas, st, n, ac.ptr.d_E(), ac.ptr.d_Hc(), ac.ptr.ld(), ac.ptr.d_Hn());
    CK(cudaStreamSynchronize(st));
    std::vector<real> g(NP);
    CK(cudaMemcpy(g.data(), ac.d_grads, NP * sizeof(real), cudaMemcpyDeviceToHost));
    o.grad.assign(g.begin(), g.end());
  }
  for (void* p : std::initializer_list<void*>{df, de, dc, dcnt, dg, dl, dcoef}) cudaFree(p);
  return o;
}

}  // namespace XPPO_NS
