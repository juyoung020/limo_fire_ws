// 기울기 검사 한 벌 (fp32·fp64 빌드가 같은 소스로). gradcheck.cu(fp32) 와 gradcheck64.cu(XNET_FP64) 가 include 한다.
// 같은 씨앗 → 같은 입력·같은 초기 가중치(float 로 뽑음) → 두 정밀도의 기울기를 원소끼리 비교할 수 있다.
#pragma once

#include <algorithm>
#include <cmath>
#include <random>
#include <vector>

#include "gc_api.h"
#include "ppo.h"

namespace XPPO_NS {

using namespace XNET_NS;

GcOut gc_run(bool full, bool negative, bool do_fd, int K, int mlp) {
  ACSpec spec;
  if (full) {
    spec = explore_spec();
  } else {
    NetSpec a;
    a.imgs = {{0, 2, 8, 8}, {128, 3, 8, 8}, {320, 1, 8, 16}};
    a.img_stride = 448;
    a.conv_ch = {3, 4, 4};
    a.in1d = 5;
    a.hidden = {6, 5};
    a.out = 4;
    NetSpec c = a;
    c.in1d = 7;
    c.out = 1;
    spec.actor = a;
    spec.critic = c;
    spec.norm_dim = 7;
  }
  spec.declare_bias = 0.f;  // 검사에서는 선언 확률을 중간으로 (−7 이면 베르누이 기울기가 거의 0)
  PPOConfig cfg;
  cfg.num_envs = 6;
  cfg.steps_per_env = 2;
  cfg.minibatches = 1;
  cfg.chunk = 4;  // S=12 를 4 개씩 3 덩이 → 덩이 누적도 검사
  cfg.tf32 = false;  // fp32 정확 모드로 검사
  cfg.cnn_bf16 = false;
  cfg.mlp = mlp;
  PPO ppo(cfg, spec, 1234);
  auto& ac = ppo.ac();
  int N = cfg.num_envs, IMG = spec.actor.img_stride, P1 = spec.actor.in1d, PR = spec.norm_dim - P1, A = spec.n_cont + 1;

  std::mt19937 rng(7);
  std::uniform_int_distribution<int> U8(0, 255);
  std::normal_distribution<float> G(0.f, 1.f);
  std::vector<uint8_t> himg((size_t)N * IMG);
  std::vector<real> h1((size_t)N * P1), hp((size_t)N * PR), hr(N);
  std::vector<int> hd(N);
  uint8_t* d_img;
  real *d1, *dp, *dact, *dr;
  int* dd;
  CK(cudaMalloc(&d_img, himg.size()));
  CK(cudaMalloc(&d1, h1.size() * sizeof(real)));
  CK(cudaMalloc(&dp, std::max<size_t>(1, hp.size()) * sizeof(real)));
  CK(cudaMalloc(&dact, (size_t)N * A * sizeof(real)));
  CK(cudaMalloc(&dr, N * sizeof(real)));
  CK(cudaMalloc(&dd, N * sizeof(int)));
  auto fill_obs = [&]() {
    for (auto& v : himg) v = (uint8_t)U8(rng);
    for (auto& v : h1) v = 2.f * G(rng) + 0.5f;
    for (auto& v : hp) v = G(rng);
    CK(cudaMemcpy(d_img, himg.data(), himg.size(), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d1, h1.data(), h1.size() * sizeof(real), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dp, hp.data(), hp.size() * sizeof(real), cudaMemcpyHostToDevice));
  };
  for (int t = 0; t < cfg.steps_per_env; ++t) {
    fill_obs();
    ppo.act(d_img, d1, dp, dact);
    for (int i = 0; i < N; ++i) {
      hr[i] = G(rng);
      hd[i] = (i % 5 == 1) ? 1 : (i % 5 == 3 ? 3 : 0);  // 진행·종료·끊김 섞기
    }
    CK(cudaMemcpy(dr, hr.data(), N * sizeof(real), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dd, hd.data(), N * sizeof(int), cudaMemcpyHostToDevice));
    ppo.store(dr, dd);
  }
  fill_obs();
  ppo.compute_returns(d_img, d1, dp);

  // 롤아웃 때와 다른 정책으로 → ratio ≠ 1, 일부 샘플이 자르기 쪽, 가치도 자르기 쪽에 걸리게.
  // 변형은 float 로 계산해 두 정밀도에서 같은 값이 되게 한다.
  long NP = ac.n_params;
  std::vector<real> params(NP);
  CK(cudaMemcpy(params.data(), ac.d_params, NP * sizeof(real), cudaMemcpyDeviceToHost));
  for (auto& v : params) {
    float f = (float)v;
    f += 0.15f * G(rng) * (std::fabs(f) + 0.05f);
    v = f;
  }
  for (int j = 0; j < spec.n_cont; ++j) params[ac.logstd_off + j] = -0.3f + 0.2f * j;
  CK(cudaMemcpy(ac.d_params, params.data(), NP * sizeof(real), cudaMemcpyHostToDevice));

  int S = N * cfg.steps_per_env;
  std::vector<int> hidx(S);
  for (int i = 0; i < S; ++i) hidx[i] = i;
  int* didx;
  CK(cudaMalloc(&didx, S * 4));
  CK(cudaMemcpy(didx, hidx.data(), S * 4, cudaMemcpyHostToDevice));

  GcOut out;
  g_break_elu_grad = negative;
  out.loss = ppo.loss_and_grad(didx, S, true);
  g_break_elu_grad = false;
  std::vector<real> g(NP);
  CK(cudaMemcpy(g.data(), ac.d_grads, NP * sizeof(real), cudaMemcpyDeviceToHost));
  out.ga.assign(g.begin(), g.end());
  for (auto& b : ac.blocks) out.blocks.push_back({b.off, b.n, (int)b.kind, b.name});
  out.n_params = NP;
  if (do_fd) {
    // fp64 에서 중앙 차분 h = 1e-6·max(1,|p|): 반올림 ~1e-16/1e-6 = 1e-10, 절단 ~h² — 자르기 꺾임을 건널 확률도 무시할 만함
    for (auto& b : ac.blocks) {
      std::vector<long> ids(b.n);
      for (long i = 0; i < b.n; ++i) ids[i] = b.off + i;
      long k = std::min<long>(K, b.n);
      std::partial_sort(ids.begin(), ids.begin() + k, ids.end(),
                        [&](long x, long y) { return std::fabs(out.ga[x]) > std::fabs(out.ga[y]); });
      ids.resize(k);
      for (long id : ids) {
        real o = params[id];
        real h = (real)1e-6 * std::max((real)1, std::fabs(o));
        real vp = o + h, vm = o - h;
        CK(cudaMemcpy(ac.d_params + id, &vp, sizeof(real), cudaMemcpyHostToDevice));
        double Lp = ppo.loss_and_grad(didx, S, false);
        CK(cudaMemcpy(ac.d_params + id, &vm, sizeof(real), cudaMemcpyHostToDevice));
        double Lm = ppo.loss_and_grad(didx, S, false);
        CK(cudaMemcpy(ac.d_params + id, &o, sizeof(real), cudaMemcpyHostToDevice));
        out.fd_ids.push_back(id);
        out.fd.push_back((Lp - Lm) / ((double)vp - (double)vm));
      }
    }
  }
  cudaFree(d_img);
  cudaFree(d1);
  cudaFree(dp);
  cudaFree(dact);
  cudaFree(dr);
  cudaFree(dd);
  cudaFree(didx);
  return out;
}

}  // namespace XPPO_NS
