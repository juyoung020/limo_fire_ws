// 환경 속도: 환경 N 개를 무작위 행동으로 돌려 초당 행동 수와 덮은 비율을 잰다.
// env_bench <장면.scene> [장면 …] [--envs 1024] [--steps 50] [--env config/env.yaml] [--model config/sensor_model.yaml]
#include <chrono>
#include <cstdio>
#include <string>
#include <vector>

#include "env_api.h"

int main(int argc, char** argv) {
  std::vector<std::string> scenes;
  std::string env_path = SIM_CONFIG_DIR "/env.yaml", model_path = SIM_CONFIG_DIR "/sensor_model.yaml";
  int n_env = 1024, n_steps = 50;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    if (a == "--envs" && i + 1 < argc) n_env = std::stoi(argv[++i]);
    else if (a == "--steps" && i + 1 < argc) n_steps = std::stoi(argv[++i]);
    else if (a == "--env" && i + 1 < argc) env_path = argv[++i];
    else if (a == "--model" && i + 1 < argc) model_path = argv[++i];
    else scenes.push_back(a);
  }
  if (scenes.empty()) { std::fprintf(stderr, "env_bench <장면.scene> [장면 …] [--envs 1024] [--steps 50]\n"); return 2; }
  const sim::EnvHostCfg hc = sim::load_env_cfg(env_path);
  const sim::LidarModel lidar = sim::load_lidar_model(model_path);
  std::vector<sim::World> worlds;
  for (const auto& p : scenes) worlds.push_back(sim::build_world(p, sim::load_scene(p), hc));
  sim::Env env(hc, worlds, lidar, n_env, 1, true);
  env.reset();
  std::vector<float> acts((size_t)2 * n_env * n_steps);
  uint64_t rng = 5;
  for (size_t i = 0; i < acts.size(); i += 2) sim::random_action(hc.cfg, rng, &acts[i]);
  float* d_act;
  cudaMalloc(&d_act, sizeof(float) * acts.size());
  cudaMemcpy(d_act, acts.data(), sizeof(float) * acts.size(), cudaMemcpyHostToDevice);
  env.step(d_act);   // 데우기
  env.sync();
  const auto t0 = std::chrono::steady_clock::now();
  for (int t = 0; t < n_steps; ++t) env.step(d_act + (size_t)2 * t * n_env);
  env.sync();
  const double sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  std::vector<sim::EnvState> st;
  env.get_state(st);
  double cov = 0;
  for (const auto& e : st) cov += e.coverage;
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, 0);
  const double sim_sec = (double)n_env * n_steps * hc.cfg.substeps * hc.cfg.dt;
  std::printf("GPU %s: 환경 %d × 행동 %d = %.1f ms/걸음, 초당 행동 %.0f (시뮬 시간 %.0f 배속), 라이다 %.0f 장/초\n", prop.name, n_env,
              n_steps, sec / n_steps * 1e3, n_env * n_steps / sec, sim_sec / sec,
              (double)n_env * n_steps * (hc.cfg.substeps / hc.cfg.lidar_every) / sec);
  std::printf("무작위 행동 %d 걸음 뒤 평균 덮은 비율 %.3f\n", n_steps + 1, cov / n_env);
  return 0;
}
