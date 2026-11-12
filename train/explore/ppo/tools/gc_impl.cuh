// 기울기 검사 한 벌 (fp32·fp64 빌드가 같은 소스로). gradcheck.cu(fp32) 와 gradcheck64.cu(XNET_FP64) 가 include 한다.
// 같은 씨앗 → 같은 입력·같은 초기 가중치(float 로 뽑음) → 두 정밀도의 기울기를 원소끼리 비교할 수 있다.
#pragma once

#include <algorithm>
#include <cstdlib>
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
    // 공유 인코더 4 층: 넷째 층 덧댐(셋째 층 격자 홀수 행)까지 들어가게 — 16×16 → 2×2 → 1×1, 8×16 → 1×2 → 1×1, 24×16 → 3×2 → 2×1
    NetSpec e;
    e.imgs = {{0, 2, 16, 16}, {512, 3, 8, 16}, {896, 1, 24, 16}};
    e.img_stride = 1280;
    e.conv_ch = {3, 4, 4, 4};
    spec.enc = e;
    NetSpec a;
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
  // 계층형 (GC_HIER=1, gradcheck --hier): 슬롯 머리 + 가린 범주 분포 — 작은 구조는 슬롯 3 개 × 특징 7 (1D 앞 5 칸 뒤), 임베딩 4, 은닉 5
  const bool hier = std::getenv("GC_HIER") != nullptr;
  if (hier && full) spec = explore_spec_hier();
  if (hier && !full) {
    spec.actor.in1d = 5 + 3 * 7;
    spec.actor.out = 4;
    spec.critic.in1d = spec.actor.in1d + 2;
    spec.norm_dim = spec.critic.in1d;
    spec.n_cont = 0;
    spec.n_slots = 3;
    spec.slot_feat = 7;
    spec.slot_off = 5;
    spec.slot_hidden = 5;
  }
  // 아래층 RL (GC_LOW=1, gradcheck --low): 연속 행동 2 개 + 선언 자리. 실제 크기는 explore_spec_low (라이다 그림·깊이 가지), 작은 구조는 같은 인코더에 행동 폭만
  if (std::getenv("GC_LOW") != nullptr) {
    if (full) spec = explore_spec_low();
    else { spec.n_cont = 2; spec.actor.out = 3; }
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
  // 평균 행동 경계 손실 검사: GC_BOUND_SOFT 로 경계를 좁히면(예 0.02) 작은 무작위 μ 도 경계 밖이라 그 항의 기울기가 차분과 비교된다
  if (const char* e = std::getenv("GC_BOUND_SOFT")) cfg.bound_soft = (float)std::atof(e);
  // 슬롯 로짓 공통 성분 벌점(ppo.h slot_cm_coef): 기본 1e-3 은 작은 구조에서 기울기에 거의 안 보여 검사에선 0.05 로 키운다 (GC_SLOT_CM 로 바꿈)
  cfg.slot_cm_coef = 0.05f;
  if (const char* e = std::getenv("GC_SLOT_CM")) cfg.slot_cm_coef = (float)std::atof(e);
  PPO ppo(cfg, spec, 1234);
  auto& ac = ppo.ac();
  int N = cfg.num_envs, IMG = spec.enc.img_stride, P1 = spec.actor.in1d, PR = spec.norm_dim - P1, A = std::max(spec.n_cont + 1, 4);

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
  int fill_t = 0;
  auto fill_obs = [&]() {
    ++fill_t;
    for (auto& v : himg) v = (uint8_t)U8(rng);
    for (auto& v : h1) v = 2.f * G(rng) + 0.5f;
    if (spec.n_slots > 0)   // 후보 있음 칸(특징 마지막) 0/1: 샘플 i 슬롯 k 는 (i + k) % 3 ≠ 0 이면 있음, i % 5 == 4 는 모두 빈 칸(고를 것 없음)
      for (int i = 0; i < N; ++i)
        for (int k = 0; k < spec.n_slots; ++k)
          h1[(size_t)i * P1 + spec.slot_off + k * spec.slot_feat + spec.slot_feat - 1] = (i % 5 != 4 && (i + k + fill_t) % 3 != 0) ? 1.f : 0.f;
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
  // log σ = −0.3, −0.1, 0 — 마지막 성분은 정확히 상한 ln std_max = 0 위 (투영은 Adam 뒤에만이라 손실 자체는 clamp 없음 → 상한 위 기울기도 매끈한 값)
  for (int j = 0; j < spec.n_cont; ++j) params[ac.logstd_off + j] = std::min(-0.3f + 0.2f * j, std::log(cfg.std_max));
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
  // σ 범위 투영: log σ 를 범위 밖(위·아래)에 두고 update 한 번(Adam 5 걸음) → 모두 [ln std_min, ln std_max] 안이어야
  {
    const float bad[] = {3.f, -10.f, 0.5f, -4.f};
    for (int j = 0; j < spec.n_cont; ++j) params[ac.logstd_off + j] = bad[j % 4];
    CK(cudaMemcpy(ac.d_params, params.data(), NP * sizeof(real), cudaMemcpyHostToDevice));
    ac.params_changed();
    ppo.update(d_img, d1, dp);
    std::vector<real> ls(spec.n_cont);
    CK(cudaMemcpy(ls.data(), ac.log_std(), spec.n_cont * sizeof(real), cudaMemcpyDeviceToHost));
    const double lo = std::log((double)cfg.std_min), hi = std::log((double)cfg.std_max);
    out.proj_ok = true;
    for (real l : ls) {
      out.proj_ls.push_back((double)l);
      if (!((double)l >= lo - 1e-6 && (double)l <= hi + 1e-6)) out.proj_ok = false;
    }
    out.proj_lo = lo;
    out.proj_hi = hi;
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
