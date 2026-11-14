// 그래프 어텐션 정책 처리량·메모리: 환경 N 개(기본 2048), explore_spec_graph (mdp::graph 격자 32×32, HEADER 크기), 그래프마다 유효 노드 약 --nodes 개
// (이어진 덩어리 하나, 5×5 스텐실 안 유효 이웃과 0.5 확률로 양방향 간선) 로 act 한 번·update 한 번 시간과 GPU 메모리. 기존 CNN+MLP 는 tools/bench.
//   graph_bench [--envs 2048] [--steps 24] [--nodes 150] [--share 1] [--rows R] [--chunk C] [--epochs E] [--tf32 1] [--layers L]
// 데우기는 act_inference 한 번, 잰 것은 한 반복(롤아웃 steps 걸음 + update 한 번).
#include <chrono>
#include <cstdio>
#include <random>
#include <string>
#include <vector>

#include "../../mdp/obs_layout.h"
#include "ppo.h"

using namespace xnet;
namespace g = mdp::graph;

int main(int argc, char** argv) {
  xppo::PPOConfig cfg;
  ACSpec spec = explore_spec_graph();
  int nodes = 150;
  for (int i = 1; i + 1 < argc; i += 2) {
    std::string a = argv[i];
    if (a == "--envs") cfg.num_envs = std::stoi(argv[i + 1]);
    if (a == "--steps") cfg.steps_per_env = std::stoi(argv[i + 1]);
    if (a == "--chunk") cfg.chunk = std::stoi(argv[i + 1]);
    if (a == "--epochs") cfg.epochs = std::stoi(argv[i + 1]);
    if (a == "--tf32") cfg.tf32 = std::stoi(argv[i + 1]) != 0;
    if (a == "--nodes") nodes = std::stoi(argv[i + 1]);
    if (a == "--share") spec.graph_share = std::stoi(argv[i + 1]) != 0;
    if (a == "--rows") spec.graph_rows = std::stol(argv[i + 1]);
    if (a == "--layers") spec.gspec.layers = std::stoi(argv[i + 1]);
  }
  const int N = cfg.num_envs, GN = g::N, F = g::FEAT, GW = g::GW, GH = g::GH;
  size_t free0, total;
  CK(cudaMemGetInfo(&free0, &total));
  xppo::PPO ppo(cfg, spec, 1);
  auto& ac = ppo.ac();
  uint8_t* df;
  uint32_t* de;
  int *dc, *done;
  float *dg, *act, *rew;
  CK(cudaMalloc(&df, (size_t)N * GN * F));
  CK(cudaMalloc(&de, (size_t)N * GN * 4));
  CK(cudaMalloc(&dc, (size_t)N * 4));
  CK(cudaMalloc(&done, (size_t)N * 4));
  CK(cudaMalloc(&dg, (size_t)N * g::GLOB * 4));
  CK(cudaMalloc(&act, (size_t)N * mdp::act::DIM * 4));
  CK(cudaMalloc(&rew, (size_t)N * 4));
  double mean_nodes = 0, mean_deg = 0;
  {
    std::mt19937 rng(3);
    std::vector<uint8_t> hf((size_t)N * GN * F);
    std::vector<uint32_t> he((size_t)N * GN, 0);
    std::vector<int> hc(N);
    std::uniform_int_distribution<int> U8(0, 255);
    for (int i = 0; i < N; ++i) {
      uint8_t* f = &hf[(size_t)i * GN * F];
      uint32_t* e = &he[(size_t)i * GN];
      for (size_t k = 0; k < (size_t)GN * F; ++k) f[k] = (uint8_t)U8(rng);
      for (int j = 0; j < GN; ++j) f[(size_t)j * F] = 0;
      // 이어진 덩어리: 가운데 근처에서 시작해 상하좌우로 무작위로 자람
      std::vector<int> blob{(GH / 2) * GW + GW / 2};
      f[(size_t)blob[0] * F] = 1;
      const int want = nodes + (int)(rng() % 21) - 10;
      while ((int)blob.size() < want) {
        const int a = blob[rng() % blob.size()], dir = (int)(rng() % 4);
        const int x = a % GW + (dir == 0) - (dir == 1), y = a / GW + (dir == 2) - (dir == 3);
        if (x < 0 || x >= GW || y < 0 || y >= GH || f[(size_t)(y * GW + x) * F]) continue;
        f[(size_t)(y * GW + x) * F] = 1;
        blob.push_back(y * GW + x);
      }
      for (int a : blob)
        for (int k = 0; k < g::NB; ++k) {
          const int i5 = k >= 12 ? k + 1 : k, x = a % GW + i5 % 5 - 2, y = a / GW + i5 / 5 - 2;
          if (x < 0 || x >= GW || y < 0 || y >= GH) continue;
          const int b = y * GW + x;
          if (b < a || !f[(size_t)b * F] || (rng() & 1)) continue;
          const int j5 = 24 - i5, kr = j5 > 12 ? j5 - 1 : j5;   // 반대 방향
          e[a] |= 1u << k;
          e[b] |= 1u << kr;
        }
      hc[i] = blob[rng() % blob.size()];
      mean_nodes += blob.size();
      for (int a : blob) mean_deg += __builtin_popcount(e[a]);
    }
    mean_deg /= mean_nodes;
    mean_nodes /= N;
    std::normal_distribution<float> G;
    std::vector<float> hg((size_t)N * g::GLOB), hr(N);
    for (auto& v : hg) v = G(rng);
    for (auto& v : hr) v = G(rng);
    CK(cudaMemcpy(df, hf.data(), hf.size(), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(de, he.data(), he.size() * 4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dc, hc.data(), hc.size() * 4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dg, hg.data(), hg.size() * 4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(rew, hr.data(), N * 4, cudaMemcpyHostToDevice));
    CK(cudaMemset(done, 0, N * 4));
  }
  const xppo::GraphObs obs{df, de, dc, dg};
  auto now = [] { return std::chrono::steady_clock::now(); };
  auto ms = [](auto a, auto b) { return std::chrono::duration<double, std::milli>(b - a).count(); };
  ppo.act_inference(obs, act);   // 데우기
  CK(cudaDeviceSynchronize());
  double t_act = 0;
  for (int t = 0; t < cfg.steps_per_env; ++t) {
    CK(cudaDeviceSynchronize());
    auto a0 = now();
    ppo.act(obs, act);
    CK(cudaDeviceSynchronize());
    t_act += ms(a0, now());
    ppo.store(rew, done);
  }
  t_act /= cfg.steps_per_env;
  auto u0 = now();
  ppo.update(obs);
  CK(cudaDeviceSynchronize());
  const double t_upd = ms(u0, now());
  auto i0 = now();
  ppo.act_inference(obs, act);
  CK(cudaDeviceSynchronize());
  const double t_inf = ms(i0, now());
  size_t free1;
  CK(cudaMemGetInfo(&free1, &total));
  const long S = (long)N * cfg.steps_per_env;
  const auto& gs = ac.spec.gspec;
  std::printf("환경 %d, 걸음/반복 %d (샘플 %ld), 그래프마다 유효 노드 평균 %.0f / %d (노드마다 간선 평균 %.1f), 가치 몸통 %s, d %d · 머리 %d · 인코더 %d 층 · FF %d\n", N,
              cfg.steps_per_env, S, mean_nodes, GN, mean_deg, ac.spec.graph_share ? "공유" : "따로", gs.d, gs.heads, gs.layers, gs.ff);
  std::printf("묶음: 샘플 ≤ %d, 유효 노드 합 ≤ %ld. 에포크 %d × 미니배치 %d, 변수 %ld 개 (%.1f MB), TF32 %d\n", ppo.config().chunk, ac.spec.graph_rows, cfg.epochs,
              cfg.minibatches, ac.n_params, ac.n_params * 4 / 1e6, (int)cfg.tf32);
  std::printf("act 한 번 (환경 %d 개 — 정책 + 가치 정방향, 샘플·기록): %.1f ms\n", N, t_act);
  std::printf("act_inference 한 번 (정책만): %.1f ms\n", t_inf);
  std::printf("update 한 번 (GAE + %d 에포크 × %d 미니배치, 샘플 %ld): %.0f ms → %.0f 샘플·에포크/s\n", cfg.epochs, cfg.minibatches, S, t_upd,
              (double)S * cfg.epochs / (t_upd / 1000));
  std::printf("GPU 메모리: 이 프로세스가 더 쓴 양 %.2f GB (롤아웃·Adam·작업 버퍼 %.2f GB, 몸통 작업 공간 정책 %.2f + 가치 %.2f GB, 환경 쪽 관측 버퍼 %.2f GB, 나머지 cuBLAS·CUDA 문맥)\n",
              (free0 - free1) / 1e9, ppo.storage_bytes() / 1e9, ac.gbody.ws_bytes() / 1e9, ac.spec.graph_share ? 0.0 : ac.gbody_c.ws_bytes() / 1e9,
              ((size_t)N * (GN * (F + 4) + 4 * (g::GLOB + mdp::act::DIM + 3))) / 1e9);
  return 0;
}
