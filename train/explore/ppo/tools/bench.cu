// 처리량·메모리: 환경 N 개(기본 2048), obs_layout.h 실제 크기, 무작위 관측으로 act 한 번·update 한 번 시간과 GPU 메모리.
//   bench [--envs N] [--steps T] [--chunk C]
#include <chrono>
#include <cstdio>
#include <random>
#include <string>
#include <vector>

#include "../../mdp/obs_layout.h"
#include "ppo.h"

using namespace xnet;
namespace o = mdp::obs;

int main(int argc, char** argv) {
  xppo::PPOConfig cfg;
  for (int i = 1; i + 1 < argc; i += 2) {
    std::string a = argv[i];
    if (a == "--envs") cfg.num_envs = std::stoi(argv[i + 1]);
    if (a == "--steps") cfg.steps_per_env = std::stoi(argv[i + 1]);
    if (a == "--chunk") cfg.chunk = std::stoi(argv[i + 1]);
    if (a == "--mlp") cfg.mlp = std::stoi(argv[i + 1]);
  }
  int N = cfg.num_envs;
  size_t free0, total;
  CK(cudaMemGetInfo(&free0, &total));
  xppo::PPO ppo(cfg, explore_spec(), 1);
  auto& ac = ppo.ac();
  // 환경 쪽 버퍼 (러너가 줄 것과 같은 크기)
  uint8_t* img;
  float *o1, *pv, *act, *rew;
  int* done;
  CK(cudaMalloc(&img, (size_t)N * o::IMG_SIZE));
  CK(cudaMalloc(&o1, (size_t)N * o::POLICY_1D * 4));
  CK(cudaMalloc(&pv, (size_t)N * o::PRIV * 4));
  CK(cudaMalloc(&act, (size_t)N * mdp::act::DIM * 4));
  CK(cudaMalloc(&rew, N * 4));
  CK(cudaMalloc(&done, N * 4));
  {
    std::mt19937 rng(3);
    std::vector<uint8_t> hi((size_t)N * o::IMG_SIZE);
    for (auto& v : hi) v = (uint8_t)(rng() & 255);
    CK(cudaMemcpy(img, hi.data(), hi.size(), cudaMemcpyHostToDevice));
    std::normal_distribution<float> G;
    std::vector<float> h1((size_t)N * o::POLICY_1D), hp((size_t)N * o::PRIV), hr(N);
    for (auto& v : h1) v = G(rng);
    for (auto& v : hp) v = G(rng);
    for (auto& v : hr) v = G(rng);
    CK(cudaMemcpy(o1, h1.data(), h1.size() * 4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(pv, hp.data(), hp.size() * 4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(rew, hr.data(), N * 4, cudaMemcpyHostToDevice));
    CK(cudaMemset(done, 0, N * 4));
  }
  auto now = [] { return std::chrono::steady_clock::now(); };
  auto ms = [](auto a, auto b) { return std::chrono::duration<double, std::milli>(b - a).count(); };
  size_t free1;
  double t_act = 0, t_upd = 0, t_inf = 0;
  int reps = 2;
  for (int r = 0; r < reps + 1; ++r) {  // 첫 번은 데우기
    double ta = 0;
    for (int t = 0; t < cfg.steps_per_env; ++t) {
      CK(cudaDeviceSynchronize());
      auto a0 = now();
      ppo.act(img, o1, pv, act);
      CK(cudaDeviceSynchronize());
      ta += ms(a0, now());
      ppo.store(rew, done);
    }
    auto u0 = now();
    ppo.update(img, o1, pv);
    CK(cudaDeviceSynchronize());
    double tu = ms(u0, now());
    auto i0 = now();
    ppo.act_inference(img, o1, act);
    CK(cudaDeviceSynchronize());
    double ti = ms(i0, now());
    if (r > 0) t_act += ta / cfg.steps_per_env, t_upd += tu, t_inf += ti;
  }
  CK(cudaMemGetInfo(&free1, &total));
  long S = (long)N * cfg.steps_per_env;
  std::printf("환경 %d, 걸음/반복 %d (샘플 %ld), 덩이 %d, 에포크 %d × 미니배치 %d, 변수 %ld 개 (%.1f MB)\n", N,
              cfg.steps_per_env, S, cfg.chunk, cfg.epochs, cfg.minibatches, ac.n_params, ac.n_params * 4 / 1e6);
  std::printf("act 한 번 (환경 %d 개, 샘플·기록·정규화 갱신): %.2f ms\n", N, t_act / reps);
  std::printf("act_inference 한 번: %.2f ms\n", t_inf / reps);
  std::printf("update 한 번 (GAE + %d 에포크 × %d 미니배치, 샘플 %ld): %.0f ms → %.0f 샘플·에포크/s\n", cfg.epochs,
              cfg.minibatches, S, t_upd / reps, (double)S * cfg.epochs / (t_upd / reps / 1000));
  std::printf("GPU 메모리: 이 프로세스가 더 쓴 양 %.2f GB (롤아웃·Adam·작업 버퍼 %.2f GB, 신경망 작업 공간 actor %.2f + critic %.2f GB, "
              "환경 쪽 관측 버퍼 %.2f GB, 나머지 cuBLAS·CUDA 문맥)\n",
              (free0 - free1) / 1e9, ppo.storage_bytes() / 1e9, ac.actor.ws_bytes() / 1e9, ac.critic.ws_bytes() / 1e9,
              ((size_t)N * (o::IMG_SIZE + 4 * (o::POLICY_1D + o::PRIV + mdp::act::DIM + 2))) / 1e9);
  return 0;
}
