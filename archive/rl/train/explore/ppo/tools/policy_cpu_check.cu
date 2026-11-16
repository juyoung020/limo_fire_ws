// 실물 노드의 CPU 정방향(network/policy_cpu.h) == 학습기 act_inference(GPU) 확인: 같은 관측에서 같은 결정 슬롯, 슬롯 로짓 차 ≤ 1e-4.
// 관측 파일 = 기록마다 이미지 mdp::obs::IMG_SIZE 바이트 + 1D mdp::hier::POLICY_1D float (sim/tools/rl_node_check --dump · limo_fire_rl obs_dump).
// 학습기 정방향 두 가지를 다 본다: 정확 경로(합성곱 fp32 융합·MLP fp32 융합 = 기울기 검사 경로)와 학습·평가 기본(합성곱·MLP bf16 텐서 코어, fp32 누산).
//   policy_cpu_check --ckpt 정책.bin 관측.bin [관측.bin …] [--tol 1e-4]
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

#include "../../mdp/obs_layout.h"
#include "../network/policy_cpu.h"
#include "ppo.h"

int main(int argc, char** argv) {
  std::string ckpt;
  std::vector<std::string> files;
  double tol = 1e-4;
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    if (a == "--ckpt" && i + 1 < argc) ckpt = argv[++i];
    else if (a == "--tol" && i + 1 < argc) tol = std::stod(argv[++i]);
    else files.push_back(a);
  }
  if (ckpt.empty() || files.empty()) {
    std::fprintf(stderr, "policy_cpu_check --ckpt 정책.bin 관측.bin …\n");
    return 2;
  }
  constexpr int IMG = mdp::obs::IMG_SIZE, D1 = mdp::hier::POLICY_1D, K = mdp::hier::K;
  std::vector<uint8_t> img;
  std::vector<float> o1;
  for (auto& f : files) {
    std::ifstream in(f, std::ios::binary);
    if (!in) { std::fprintf(stderr, "없음: %s\n", f.c_str()); return 1; }
    std::vector<uint8_t> bi(IMG);
    std::vector<float> bo(D1);
    while (in.read((char*)bi.data(), IMG) && in.read((char*)bo.data(), D1 * 4)) {
      img.insert(img.end(), bi.begin(), bi.end());
      o1.insert(o1.end(), bo.begin(), bo.end());
    }
  }
  const int N = (int)(o1.size() / D1);
  std::printf("관측 %d 개 (%zu 파일)\n", N, files.size());
  // CPU
  xnet_cpu::HierPolicy pol;
  pol.load(ckpt);
  std::vector<float> lc((size_t)N * K), mk((size_t)N * K);
  std::vector<int> sc(N);
  for (int i = 0; i < N; ++i) sc[i] = pol.act(&img[(size_t)i * IMG], &o1[(size_t)i * D1], &lc[(size_t)i * K], &mk[(size_t)i * K]);
  // GPU
  uint8_t* d_img;
  float *d_o1, *d_act;
  cudaMalloc(&d_img, img.size());
  cudaMalloc(&d_o1, o1.size() * 4);
  cudaMalloc(&d_act, (size_t)N * mdp::act::DIM * 4);
  cudaMemcpy(d_img, img.data(), img.size(), cudaMemcpyHostToDevice);
  cudaMemcpy(d_o1, o1.data(), o1.size() * 4, cudaMemcpyHostToDevice);
  int fail = 0;
  for (int mode = 0; mode < 2; ++mode) {
    xppo::PPOConfig pc;
    pc.num_envs = N;
    pc.cnn_bf16 = mode == 1;
    pc.mlp = mode == 1 ? 2 : 1;
    xppo::PPO ppo(pc, xnet::explore_spec_hier(), 1);
    ppo.load(ckpt);
    ppo.act_inference(d_img, d_o1, d_act, N);
    cudaDeviceSynchronize();
    std::vector<float> act((size_t)N * mdp::act::DIM), lg((size_t)N * K);
    cudaMemcpy(act.data(), d_act, act.size() * 4, cudaMemcpyDeviceToHost);
    cudaMemcpy(lg.data(), ppo.last_logits(), lg.size() * 4, cudaMemcpyDeviceToHost);
    double maxd = 0, maxv = 0;
    int same = 0, nv = 0;
    for (int i = 0; i < N; ++i) {
      same += (int)act[(size_t)i * mdp::act::DIM] == sc[i];
      for (int k = 0; k < K; ++k) {
        if (mk[(size_t)i * K + k] < 0.5f) continue;   // 가린 칸은 행동에 안 씀
        ++nv;
        maxd = std::max(maxd, (double)std::fabs(lg[(size_t)i * K + k] - lc[(size_t)i * K + k]));
        maxv = std::max(maxv, (double)std::fabs(lc[(size_t)i * K + k]));
      }
    }
    const bool ok = same == N && maxd <= tol;
    std::printf("%s (반복 %d): 결정 슬롯 같음 %d/%d, 있는 칸 로짓 %d 개 최대 차 %.3g (로짓 크기 최대 %.3g) — %s\n",
                mode == 0 ? "학습기 정확 경로 (fp32)" : "학습기 기본 경로 (bf16 텐서 코어)", ppo.iteration(), same, N, nv, maxd, maxv,
                ok ? "통과" : (mode == 0 ? "다름" : "허용 밖 (bf16 반올림 — 참고)"));
    if (mode == 0 && !ok) fail = 1;
  }
  return fail;
}
