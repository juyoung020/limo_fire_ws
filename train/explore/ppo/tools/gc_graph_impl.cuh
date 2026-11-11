// 그래프 어텐션 정책(ACSpec::graph)의 기울기 검사 한 벌 — gc_impl.cuh 와 같은 틀(fp32·fp64 가 같은 소스, 같은 씨앗 → 같은 입력·가중치).
// 작은 구조(같은 코드 길): 격자 6×5, d 8, 머리 2, 인코더 2 층, FF 12, 전역 3 칸, 참값 4 칸. 유효 노드·간선은 무작위(간선은 한쪽 방향만 있는 것도 섞임),
// 무효 노드엔 쓰레기 특징, 이웃이 하나도 없는 샘플(고를 것 없음) 하나. 묶음: 샘플 4 개 또는 유효 노드 합 60 → 행 상한으로 나뉘는 길도 지난다.
#pragma once

#include <algorithm>
#include <cmath>
#include <random>
#include <vector>

#include "../../mdp/obs_layout.h"
#include "gc_api.h"
#include "ppo.h"

namespace XPPO_NS {

using namespace XNET_NS;

GcOut gc_run_graph(bool share, bool negative, bool do_fd, int K) {
  ACSpec spec;
  spec.graph = true;
  spec.gspec = GraphSpec{6, 5, 8, 8, 2, 2, 12};
  spec.graph_glob = 3;
  spec.graph_share = share;
  spec.graph_rows = 60;
  NetSpec c;
  c.in1d = 4;
  c.hidden = {6};
  c.out = 1;
  spec.critic = c;
  spec.n_cont = 0;
  spec.norm_dim = 4;
  PPOConfig cfg;
  cfg.num_envs = 6;
  cfg.steps_per_env = 2;
  cfg.minibatches = 1;
  cfg.chunk = 4;
  cfg.tf32 = false;
  PPO ppo(cfg, spec, 1234);
  auto& ac = ppo.ac();
  const int N = cfg.num_envs, GN = spec.gspec.gw * spec.gspec.gh, F = spec.gspec.feat, GG = spec.graph_glob, GP = spec.norm_dim;

  std::mt19937 rng(7);
  std::uniform_int_distribution<int> U8(0, 255);
  std::normal_distribution<float> G(0.f, 1.f);
  std::vector<uint8_t> hf((size_t)N * GN * F);
  std::vector<uint32_t> he((size_t)N * GN);
  std::vector<int> hc(N), hd(N);
  std::vector<real> hg((size_t)N * GG), hp((size_t)N * GP), hr(N);
  uint8_t* df;
  uint32_t* de;
  int *dc, *dd;
  real *dg, *dp, *dact, *dr;
  CK(cudaMalloc(&df, hf.size()));
  CK(cudaMalloc(&de, he.size() * 4));
  CK(cudaMalloc(&dc, N * 4));
  CK(cudaMalloc(&dd, N * 4));
  CK(cudaMalloc(&dg, hg.size() * sizeof(real)));
  CK(cudaMalloc(&dp, hp.size() * sizeof(real)));
  CK(cudaMalloc(&dact, (size_t)N * mdp::act::DIM * sizeof(real)));
  CK(cudaMalloc(&dr, N * sizeof(real)));
  int fill_t = 0;
  auto fill_obs = [&]() {
    ++fill_t;
    for (int i = 0; i < N; ++i) {
      std::vector<int> valid;
      for (int j = 0; j < GN; ++j) {
        uint8_t* f = &hf[((size_t)i * GN + j) * F];
        for (int k = 0; k < F; ++k) f[k] = (uint8_t)U8(rng);   // 무효 노드에도 쓰레기 값
        f[0] = U8(rng) < 150 ? 1 : 0;
        if (f[0]) valid.push_back(j);
        he[(size_t)i * GN + j] = ((uint32_t)U8(rng) | ((uint32_t)U8(rng) << 8) | ((uint32_t)U8(rng) << 16) | ((uint32_t)U8(rng) << 24)) &
                                 ((uint32_t)U8(rng) | ((uint32_t)U8(rng) << 8) | ((uint32_t)U8(rng) << 16) | 0xFF000000u);   // 비트의 1/4 쯤, 윗 8 비트는 쓰레기
      }
      if (valid.empty()) {
        hf[((size_t)i * GN) * F] = 1;
        valid.push_back(0);
      }
      hc[i] = valid[(size_t)U8(rng) % valid.size()];
      if (i == 4 && fill_t == 1) he[(size_t)i * GN + hc[i]] = 0;   // 고를 이웃이 없는 샘플
    }
    for (auto& v : hg) v = G(rng);
    for (auto& v : hp) v = 2.f * G(rng) + 0.5f;
    CK(cudaMemcpy(df, hf.data(), hf.size(), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(de, he.data(), he.size() * 4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dc, hc.data(), N * 4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dg, hg.data(), hg.size() * sizeof(real), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dp, hp.data(), hp.size() * sizeof(real), cudaMemcpyHostToDevice));
  };
  const GraphObs obs{df, de, dc, dg, dp};
  for (int t = 0; t < cfg.steps_per_env; ++t) {
    fill_obs();
    ppo.act(obs, dact);
    for (int i = 0; i < N; ++i) {
      hr[i] = G(rng);
      hd[i] = (i % 5 == 1) ? 1 : (i % 5 == 3 ? 3 : 0);
    }
    CK(cudaMemcpy(dr, hr.data(), N * sizeof(real), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dd, hd.data(), N * sizeof(int), cudaMemcpyHostToDevice));
    ppo.store(dr, dd);
  }
  fill_obs();
  ppo.compute_returns(obs);

  // 롤아웃 때와 다른 정책으로 (비 ≠ 1, 자르기 쪽 샘플 섞임). 전역 벡터 가중치(0 으로 시작)도 0 이 아니게 된다. float 로 계산 → 두 정밀도 같은 값
  const long NP = ac.n_params;
  std::vector<real> params(NP);
  CK(cudaMemcpy(params.data(), ac.d_params, NP * sizeof(real), cudaMemcpyDeviceToHost));
  for (auto& v : params) {
    float f = (float)v;
    f += 0.15f * G(rng) * (std::fabs(f) + 0.05f);
    v = f;
  }
  CK(cudaMemcpy(ac.d_params, params.data(), NP * sizeof(real), cudaMemcpyHostToDevice));

  const int S = N * cfg.steps_per_env;
  std::vector<int> hidx(S);
  for (int i = 0; i < S; ++i) hidx[i] = i;
  int* didx;
  CK(cudaMalloc(&didx, S * 4));
  CK(cudaMemcpy(didx, hidx.data(), S * 4, cudaMemcpyHostToDevice));

  GcOut out;
  g_break_graph_grad = negative;
  out.loss = ppo.loss_and_grad(didx, S, true);
  g_break_graph_grad = false;
  std::vector<real> g(NP);
  CK(cudaMemcpy(g.data(), ac.d_grads, NP * sizeof(real), cudaMemcpyDeviceToHost));
  out.ga.assign(g.begin(), g.end());
  for (auto& b : ac.blocks) out.blocks.push_back({b.off, b.n, (int)b.kind, b.name});
  out.n_params = NP;
  if (do_fd) {
    for (auto& b : ac.blocks) {
      std::vector<long> ids(b.n);
      for (long i = 0; i < b.n; ++i) ids[i] = b.off + i;
      const long k = std::min<long>(K, b.n);
      std::partial_sort(ids.begin(), ids.begin() + k, ids.end(), [&](long x, long y) { return std::fabs(out.ga[x]) > std::fabs(out.ga[y]); });
      ids.resize(k);
      for (long id : ids) {
        const real o = params[id];
        const real h = (real)1e-6 * std::max((real)1, std::fabs(o));
        const real vp = o + h, vm = o - h;
        CK(cudaMemcpy(ac.d_params + id, &vp, sizeof(real), cudaMemcpyHostToDevice));
        const double Lp = ppo.loss_and_grad(didx, S, false);
        CK(cudaMemcpy(ac.d_params + id, &vm, sizeof(real), cudaMemcpyHostToDevice));
        const double Lm = ppo.loss_and_grad(didx, S, false);
        CK(cudaMemcpy(ac.d_params + id, &o, sizeof(real), cudaMemcpyHostToDevice));
        out.fd_ids.push_back(id);
        out.fd.push_back((Lp - Lm) / ((double)vp - (double)vm));
      }
    }
  }
  out.proj_ok = true;   // σ 없음
  for (void* p : std::initializer_list<void*>{df, de, dc, dd, dg, dp, dact, dr, didx}) cudaFree(p);
  return out;
}

}  // namespace XPPO_NS
