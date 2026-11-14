// 계층형 PPO 검증 (mdp/README "계층형" 단계 3): 장난감 환경 "맞는 슬롯 고르기" — 가린 범주 분포·슬롯 머리가 학습되는지, 가린 칸 확률이 정확히 0 인지,
// 같은 씨앗이면 가중치가 비트까지 같은지.
//   환경(판 길이 1, 문맥 띠): 슬롯 K = 8, 슬롯마다 특징 7 칸(정규 난수 6 + 마지막 = 있음 0/1, 있는 칸 1~8 개 무작위). 맞는 슬롯 = 있는 칸 중
//   특징 0 이 가장 큰 칸 → 보상 1, 아니면 0. 판마다 끝(done 1). 1D 앞 5 칸은 잡음, 이미지 0.
//   toy_slot [--iters 200] [--envs 256] [--seed 1]                    학습 → 결정 정확도(새 표본 4096) > 0.9 이면 통과
//   toy_slot --check-mask                                              가린 칸을 고른 샘플 수 = 0 (학습 샘플 · 평가 샘플 · 결정)
//   toy_slot --check-determinism [--iters 30]                          같은 씨앗 두 번 → 가중치·Adam 비트 같음
//   toy_slot --check-common-mode [--iters 400]                          슬롯 로짓 공통 성분 |l̄| (ppo/README "슬롯 로짓 공통 성분"): 학습률 상한 1e-2·목표 KL 0.05 로
//                                                                       벌점 끔(0) vs 기본(PPOConfig::slot_cm_coef) — 기본에서 마지막 |l̄| < 10 이고 정확도 > 0.9 이면 통과
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include "ppo.h"

using namespace xnet;

namespace {
constexpr int K = 8, F = 7, BASE = 5, P1 = BASE + K * F;

xnet::ACSpec toy_spec() {
  ACSpec s;
  NetSpec e;   // gradcheck 작은 구조와 같은 인코더 (이미지는 늘 0)
  e.imgs = {{0, 2, 16, 16}, {512, 3, 8, 16}, {896, 1, 24, 16}};
  e.img_stride = 1280;
  e.conv_ch = {3, 4, 4, 4};
  s.enc = e;
  NetSpec a;
  a.in1d = P1;
  a.hidden = {64, 64};
  a.out = 16;
  NetSpec c = a;
  c.out = 1;
  s.actor = a;
  s.critic = c;
  s.split_enc = true;
  s.n_cont = 0;
  s.n_slots = K;
  s.slot_feat = F;
  s.slot_off = BASE;
  s.slot_hidden = 32;
  return s;
}

struct Toy {
  int n;
  std::mt19937 rng;
  std::vector<float> o1;
  std::vector<int> best;
  explicit Toy(int n_, uint64_t seed) : n(n_), rng((uint32_t)seed), o1((size_t)n_ * P1), best(n_) {}
  void sample() {
    std::normal_distribution<float> G(0.f, 1.f);
    std::uniform_int_distribution<int> U(1, K);
    for (int i = 0; i < n; ++i) {
      float* o = &o1[(size_t)i * P1];
      for (int j = 0; j < BASE; ++j) o[j] = G(rng);
      const int nv = U(rng);
      std::vector<int> perm(K);
      for (int k = 0; k < K; ++k) perm[k] = k;
      std::shuffle(perm.begin(), perm.end(), rng);
      int b = -1;
      for (int k = 0; k < K; ++k) {
        float* f = o + BASE + k * F;
        for (int j = 0; j < F - 1; ++j) f[j] = G(rng);
        const bool v = std::find(perm.begin(), perm.begin() + nv, k) != perm.begin() + nv;
        f[F - 1] = v ? 1.f : 0.f;
        if (!v) for (int j = 0; j < F - 1; ++j) f[j] = 0.f;   // 빈 칸은 0 (환경과 같음)
        if (v && (b < 0 || f[0] > o[BASE + b * F])) b = k;
      }
      best[i] = b;
    }
  }
};

struct Dev {
  uint8_t* img;
  float *o1, *act, *rew;
  int* done;
  Dev(int n) {
    CK(cudaMalloc(&img, (size_t)n * 1280));
    CK(cudaMemset(img, 0, (size_t)n * 1280));
    CK(cudaMalloc(&o1, sizeof(float) * n * P1));
    CK(cudaMalloc(&act, sizeof(float) * n * 4));
    CK(cudaMalloc(&rew, sizeof(float) * n));
    CK(cudaMalloc(&done, sizeof(int) * n));
  }
  void up(const Toy& t) {
    CK(cudaMemcpy(o1, t.o1.data(), sizeof(float) * t.o1.size(), cudaMemcpyHostToDevice));
  }
};

// 학습 iters 반복. 반환 = 마지막 결정 정확도 (새 표본 4096). bad_mask 에 가린 칸을 고른 수를 더함
struct Opt {
  float cm_coef = -1, lr_max = -1, desired_kl = -1;   // < 0 = PPOConfig 기본
  bool cm_log = false;                              // 반복마다(50 번째) 롤아웃 로짓 공통 성분 찍기
  double last_cm = 0;
};
Opt g_opt;

double train(int iters, int n, uint64_t seed, const std::string& save, long* bad_mask, bool verbose) {
  xppo::PPOConfig cfg;
  if (g_opt.cm_coef >= 0) cfg.slot_cm_coef = g_opt.cm_coef;
  if (g_opt.lr_max > 0) cfg.lr_max = g_opt.lr_max;
  if (g_opt.desired_kl > 0) cfg.desired_kl = g_opt.desired_kl;
  cfg.num_envs = n;
  cfg.steps_per_env = 8;
  cfg.minibatches = 4;
  cfg.cnn_bf16 = false;
  cfg.mlp = 1;
  cfg.gamma = 0.f;   // 판 길이 1
  xppo::PPO ppo(cfg, toy_spec(), seed);
  Toy env(n, seed * 7 + 3);
  Dev d(n);
  std::vector<float> act((size_t)n * 4), rew(n);
  std::vector<int> done(n, 1);
  CK(cudaMemcpy(d.done, done.data(), sizeof(int) * n, cudaMemcpyHostToDevice));
  env.sample();
  d.up(env);
  for (int it = 0; it < iters; ++it) {
    double hit = 0;
    for (int t = 0; t < cfg.steps_per_env; ++t) {
      ppo.act(d.img, d.o1, d.act);
      CK(cudaMemcpy(act.data(), d.act, sizeof(float) * act.size(), cudaMemcpyDeviceToHost));
      for (int i = 0; i < n; ++i) {
        const int a = (int)act[(size_t)i * 4];
        const bool valid = env.o1[(size_t)i * P1 + BASE + a * F + F - 1] > 0.5f;
        if (!valid) ++*bad_mask;
        rew[i] = a == env.best[i] ? 1.f : 0.f;
        hit += rew[i];
      }
      CK(cudaMemcpy(d.rew, rew.data(), sizeof(float) * n, cudaMemcpyHostToDevice));
      ppo.store(d.rew, d.done);
      env.sample();
      d.up(env);
    }
    const auto us = ppo.update(d.img, d.o1);
    g_opt.last_cm = us.logit_cm;
    if (g_opt.cm_log && (it % 50 == 0 || it == iters - 1))
      std::printf("  반복 %3d: 공통 성분 |l̄| %.4g, 있는 칸 표준편차 %.3g, max|l| %.4g, 기울기 노름 %.3g, kl0 %.2g, kl %.4f, lr %.2e\n", it, us.logit_cm,
                  us.logit_spread, us.logit_absmax, us.grad_norm, us.kl0, us.kl, us.lr);
    if (verbose && (it % 20 == 0 || it == iters - 1))
      std::printf("반복 %d: 학습 정확도 %.3f, 엔트로피 %.3f, kl %.4f, value %.4f, lr %.2e\n", it, hit / (n * cfg.steps_per_env), us.entropy, us.kl, us.value, us.lr);
  }
  // 결정 평가 (새 표본) + 평가 샘플의 가림 검사
  Toy ev(4096 > n ? n : 4096, seed * 13 + 5);
  double ok = 0, tot = 0;
  for (int r = 0; r < (4096 + n - 1) / n; ++r) {
    ev.sample();
    d.up(ev);
    ppo.act_inference(d.img, d.o1, d.act, ev.n);
    CK(cudaMemcpy(act.data(), d.act, sizeof(float) * ev.n * 4, cudaMemcpyDeviceToHost));
    for (int i = 0; i < ev.n; ++i) { ok += (int)act[(size_t)i * 4] == ev.best[i]; tot += 1; }
    ppo.act_inference(d.img, d.o1, d.act, ev.n, r);
    CK(cudaMemcpy(act.data(), d.act, sizeof(float) * ev.n * 4, cudaMemcpyDeviceToHost));
    for (int i = 0; i < ev.n; ++i)
      if (ev.o1[(size_t)i * P1 + BASE + (int)act[(size_t)i * 4] * F + F - 1] <= 0.5f) ++*bad_mask;
  }
  if (!save.empty()) ppo.save(save);
  return ok / tot;
}
}  // namespace

int main(int argc, char** argv) {
  int iters = 200, n = 256;
  uint64_t seed = 1;
  bool det = false, mask = false, cmode = false;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    if (a == "--iters" && i + 1 < argc) iters = std::atoi(argv[++i]);
    else if (a == "--envs" && i + 1 < argc) n = std::atoi(argv[++i]);
    else if (a == "--seed" && i + 1 < argc) seed = std::strtoull(argv[++i], nullptr, 10);
    else if (a == "--check-determinism") det = true;
    else if (a == "--check-mask") mask = true;
    else if (a == "--check-common-mode") { cmode = true; if (iters == 200) iters = 400; }
  }
  if (cmode) {
    g_opt.lr_max = 1e-2f;
    g_opt.desired_kl = 0.05f;
    g_opt.cm_log = true;
    long b0 = 0, b1 = 0;
    std::printf("벌점 끔 (slot_cm_coef 0):\n");
    g_opt.cm_coef = 0;
    const double a0 = train(iters, n, seed, "", &b0, false);
    const double cm0 = g_opt.last_cm;
    std::printf("벌점 기본 (slot_cm_coef %g):\n", xppo::PPOConfig{}.slot_cm_coef);
    g_opt.cm_coef = -1;
    const double a1 = train(iters, n, seed, "", &b1, false);
    const double cm1 = g_opt.last_cm;
    const bool ok = cm1 < 10 && a1 > 0.9 && b1 == 0;
    std::printf("공통 성분 마지막 |l̄|: 끔 %.4g (정확도 %.3f) → 기본 %.4g (정확도 %.3f) — %s (기준: 기본 |l̄| < 10, 정확도 > 0.9)\n", cm0, a0, cm1, a1,
                ok ? "통과" : "실패");
    return ok ? 0 : 1;
  }
  long bad = 0;
  if (det) {
    const std::string pa = "toy_slot_a.bin", pb = "toy_slot_b.bin";   // 지금 폴더에 잠깐
    train(iters, n, seed, pa, &bad, false);
    train(iters, n, seed, pb, &bad, false);
    FILE *fa = std::fopen(pa.c_str(), "rb"), *fb = std::fopen(pb.c_str(), "rb");
    std::vector<char> A, B;
    for (int c; (c = std::fgetc(fa)) != EOF;) A.push_back((char)c);
    for (int c; (c = std::fgetc(fb)) != EOF;) B.push_back((char)c);
    std::fclose(fa);
    std::fclose(fb);
    const bool same = A == B && !A.empty();
    std::printf("결정성 (같은 씨앗 %d 반복 두 번, 체크포인트 %zu 바이트): %s\n", iters, A.size(), same ? "비트 같음 — 통과" : "다름 — 실패");
    std::remove(pa.c_str());
    std::remove(pb.c_str());
    return same ? 0 : 1;
  }
  if (mask) {
    const double acc = train(iters < 30 ? iters : 30, n, seed, "", &bad, false);
    std::printf("가린 칸을 고른 샘플 %ld (학습 샘플 %d 반복 × %d 걸음 × %d 환경 + 평가 샘플), 정확도 %.3f — %s\n", bad, iters < 30 ? iters : 30, 8, n, acc,
                bad == 0 ? "통과" : "실패");
    return bad == 0 ? 0 : 1;
  }
  const double acc = train(iters, n, seed, "", &bad, true);
  std::printf("결정 정확도 %.3f (무작위로 고르면 있는 칸 수의 역수 평균 ≈ 0.34), 가린 칸 고름 %ld — %s (기준 > 0.9)\n", acc, bad, acc > 0.9 && bad == 0 ? "통과" : "실패");
  return acc > 0.9 && bad == 0 ? 0 : 1;
}
