// 실물 아래층 CPU 정방향(network/policy_cpu.h LowPolicy — 넓은 지도 가지 + 순환 핵) == 학습기 act_inference(GPU, 결정적) 확인.
// 같은 무작위 관측을 환경 E 개 · T 걸음 넣고 (몇 걸음마다 판 끝 → 두 쪽 모두 h 초기화) 행동 출력 3 칸(v, ω 평균 · 선언 로짓)의 최대 차를 잰다.
// 학습기 정방향 두 가지: 정확 경로(합성곱 fp32 융합 · MLP fp32 융합 — 기준 1e-4)와 학습·평가 기본(bf16 텐서 코어 — 차이만 보고).
//   low_cpu_check --ckpt 정책.bin [--envs 8] [--steps 30] [--tol 1e-4] [--perturb 0.05]
//   --perturb s: 모든 변수에 s · N(0,1) · (|w| + 0.05) 를 더한 체크포인트(<ckpt>.perturbed)로 검사 — 이어 학습 시작점처럼 h 열이 0 이면 순환 핵 길이 출력에 안 보이므로
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <random>
#include <string>
#include <vector>

#include "../../mdp/obs_layout.h"
#include "../network/policy_cpu.h"
#include "ppo.h"

int main(int argc, char** argv) {
  std::string ckpt;
  int E = 8, T = 30;
  double tol = 1e-4, perturb = 0;
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    if (a == "--ckpt" && i + 1 < argc) ckpt = argv[++i];
    else if (a == "--envs" && i + 1 < argc) E = std::stoi(argv[++i]);
    else if (a == "--steps" && i + 1 < argc) T = std::stoi(argv[++i]);
    else if (a == "--tol" && i + 1 < argc) tol = std::stod(argv[++i]);
    else if (a == "--perturb" && i + 1 < argc) perturb = std::stod(argv[++i]);
  }
  if (ckpt.empty()) { std::fprintf(stderr, "low_cpu_check --ckpt 정책.bin\n"); return 2; }
  if (perturb > 0) {   // 흔든 체크포인트를 만들어 그걸로
    xnet_cpu::LowPolicy probe;
    probe.load(ckpt);
    xppo::PPOConfig pc;
    pc.num_envs = E;
    pc.steps_per_env = 1;
    pc.minibatches = 1;
    xppo::PPO ppo(pc, xnet::explore_spec_low(true, probe.hidden()), 1);
    ppo.load(ckpt);
    auto& ac = ppo.ac();
    std::vector<float> p(ac.n_params);
    cudaMemcpy(p.data(), ac.d_params, sizeof(float) * p.size(), cudaMemcpyDeviceToHost);
    std::mt19937 rr(3);
    std::normal_distribution<float> N(0.f, 1.f);
    for (long i = 0; i < ac.n_params; ++i)
      if (i < ac.logstd_off || i >= ac.logstd_off + 2) p[i] += (float)perturb * N(rr) * (std::fabs(p[i]) + 0.05f);
    cudaMemcpy(ac.d_params, p.data(), sizeof(float) * p.size(), cudaMemcpyHostToDevice);
    ckpt += ".perturbed";
    ppo.save(ckpt);
  }
  std::vector<xnet_cpu::LowPolicy> cpu(E);
  for (auto& p : cpu) p.load(ckpt);
  const int H = cpu[0].hidden(), IMG = mdp::low::IMG_WIDE, D = mdp::low::N1D, A = mdp::low::ACT;
  std::printf("체크포인트 %s (반복 %d), GRU %d, 환경 %d × 걸음 %d\n", ckpt.c_str(), cpu[0].iteration(), H, E, T);
  int fail = 0;
  for (int exact = 1; exact >= 0; --exact) {
    xppo::PPOConfig pc;
    pc.num_envs = E;
    pc.steps_per_env = 1;
    pc.minibatches = 1;
    pc.cnn_bf16 = !exact;
    pc.mlp = exact ? 1 : 2;
    xppo::PPO ppo(pc, xnet::explore_spec_low(true, H), 1);
    ppo.load(ckpt);
    for (auto& p : cpu) p.reset();
    ppo.inference_reset_all();
    uint8_t* d_img;
    float *d1, *dact;
    int* ddone;
    cudaMalloc(&d_img, (size_t)E * IMG);
    cudaMalloc(&d1, sizeof(float) * E * D);
    cudaMalloc(&dact, sizeof(float) * E * A);
    cudaMalloc(&ddone, sizeof(int) * E);
    std::mt19937 rng(11);
    std::uniform_int_distribution<int> U8(0, 255);
    std::normal_distribution<float> G(0.f, 1.f);
    std::vector<uint8_t> img((size_t)E * IMG);
    std::vector<float> o1((size_t)E * D), ga((size_t)E * A), ca(A);
    std::vector<int> done(E);
    double worst = 0;
    int dec_diff = 0;
    for (int t = 0; t < T; ++t) {
      for (auto& v : img) v = (uint8_t)(U8(rng) % 3 == 0 ? 0 : U8(rng));
      for (auto& v : o1) v = G(rng);
      cudaMemcpy(d_img, img.data(), img.size(), cudaMemcpyHostToDevice);
      cudaMemcpy(d1, o1.data(), sizeof(float) * o1.size(), cudaMemcpyHostToDevice);
      ppo.act_inference(d_img, d1, dact, E, -1);
      cudaMemcpy(ga.data(), dact, sizeof(float) * ga.size(), cudaMemcpyDeviceToHost);
      // GPU act_inference 의 행동 = [평균 2, 선언 0/1] → 선언 로짓은 안 나오므로 선언은 0/1 로 견줌
      for (int i = 0; i < E; ++i) {
        const bool d = cpu[i].act(img.data() + (size_t)i * IMG, o1.data() + (size_t)i * D, ca.data());
        for (int j = 0; j < A - 1; ++j) worst = std::max(worst, (double)std::fabs(ca[j] - ga[(size_t)i * A + j]));
        dec_diff += (d ? 1.f : 0.f) != ga[(size_t)i * A + A - 1];
      }
      for (int i = 0; i < E; ++i) done[i] = (t + i) % 7 == 6;   // 판 끝 섞기 → 다음 걸음 h = 0
      cudaMemcpy(ddone, done.data(), sizeof(int) * E, cudaMemcpyHostToDevice);
      ppo.inference_reset(ddone, E);
      for (int i = 0; i < E; ++i)
        if (done[i]) cpu[i].reset();
    }
    cudaDeviceSynchronize();
    const bool ok = worst <= tol && dec_diff == 0;
    std::printf("%s: 행동 평균 최대 차 %.3e · 선언 다름 %d / %d%s\n", exact ? "정확 경로 (fp32)" : "학습 기본 (bf16)", worst, dec_diff, E * T,
                exact ? (ok ? "  통과" : "  실패") : "  (보고만)");
    if (exact && !ok) fail = 1;
    cudaFree(d_img); cudaFree(d1); cudaFree(dact); cudaFree(ddone);
  }
  return fail;
}
