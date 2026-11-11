// 환경 속도: 환경 N 개를 무작위 행동으로 돌려 초당 행동 수와 덮은 비율을 잰다.
// env_bench <장면.scene> [장면 …] [--list 목록] [--envs 1024] [--steps 50] [--env config/env.yaml] [--model config/sensor_model.yaml]
//           [--cpu] [--threads N]   CPU 참조판(같은 단계 함수·같은 BVH 순회)을 스레드 N 개로 — 같은 맵·행동열·환경 수로 GPU 와 비교.
//           [--hier] 계층형(행동 = 후보 번호, 걸음 하나 = 결정 하나)  [--graph] 그래프 모드(행동 = 이웃 번호) — 그래프 관측만 다시 내는 시간도 잰다
//   끝에 상태 해시를 찍는다: 같은 인자면 GPU 와 CPU 가 같은 값이어야 한다(비트 같음)
#include <chrono>
#include <cstdlib>
#include <cstdio>
#include <string>
#include <vector>

#include "env_api.h"

int main(int argc, char** argv) {
  std::vector<std::string> scenes;
  std::string env_path = SIM_CONFIG_DIR "/env.yaml", model_path = SIM_CONFIG_DIR "/sensor_model.yaml", list;
  int n_env = 1024, n_steps = 50;
  int people = 0, curriculum = -1, threads = 1;
  bool cpu = false, hier = false, graph = false;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    if (a == "--envs" && i + 1 < argc) n_env = std::stoi(argv[++i]);
    else if (a == "--steps" && i + 1 < argc) n_steps = std::stoi(argv[++i]);
    else if (a == "--env" && i + 1 < argc) env_path = argv[++i];
    else if (a == "--curriculum" && i + 1 < argc) curriculum = std::stoi(argv[++i]);
    else if (a == "--people" && i + 1 < argc) people = std::stoi(argv[++i]);
    else if (a == "--model" && i + 1 < argc) model_path = argv[++i];
    else if (a == "--list" && i + 1 < argc) list = argv[++i];
    else if (a == "--cpu") cpu = true;
    else if (a == "--hier") hier = true;
    else if (a == "--graph") hier = graph = true;
    else if (a == "--threads" && i + 1 < argc) threads = std::stoi(argv[++i]);
    else scenes.push_back(a);
  }
  if (scenes.empty() && list.empty()) { std::fprintf(stderr, "env_bench <장면.scene> [장면 …] [--envs 1024] [--steps 50]\n"); return 2; }
  sim::EnvHostCfg hc = sim::load_env_cfg(env_path);
  if (curriculum >= 0) hc.cfg.curriculum = curriculum;   // 0: 모든 맵 고르게 (사람 수도 --people 그대로)
  if (hier) hc.cfg.action_mode = sim::kActPath;
  if (graph) hc.cfg.graph_on = 1;
  const sim::LidarModel lidar = sim::load_lidar_model(model_path);
  std::vector<sim::World> worlds;
  if (!list.empty()) worlds = sim::load_scene_list(list, hc, nullptr);
  for (const auto& p : scenes) worlds.push_back(sim::build_world(p, sim::load_scene(p), hc));
  sim::set_cpu_threads(threads);
  sim::Env env(hc, worlds, lidar, n_env, 1, !cpu);
  env.set_people_max(std::vector<int>(n_env, people));   // 환경마다 사람 0~people 명
  env.reset();
  std::vector<float> acts((size_t)mdp::act::DIM * n_env * n_steps);
  uint64_t rng = 5;
  for (size_t i = 0; i < acts.size(); i += mdp::act::DIM) sim::random_action(hc.cfg, rng, &acts[i]);
  float* d_act = acts.data();   // CPU 판은 호스트 행동
  if (!cpu) {
    cudaMalloc(&d_act, sizeof(float) * acts.size());
    cudaMemcpy(d_act, acts.data(), sizeof(float) * acts.size(), cudaMemcpyHostToDevice);
  }
  env.step(d_act);   // 데우기
  env.sync();
  const auto t0 = std::chrono::steady_clock::now();
  for (int t = 0; t < n_steps; ++t) env.step(d_act + (size_t)mdp::act::DIM * t * n_env);
  env.sync();
  const double sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  double graph_ms = 0;
  bool graph_same = true;
  if (graph) {   // 그래프 관측만 kRep 번 다시 (같은 상태 — 값이 그대로인지도 봄)
    constexpr int kRep = 20;
    std::vector<uint8_t> f0, f1;
    std::vector<uint32_t> e0, e1;
    std::vector<int32_t> c0, c1, k0, k1;
    std::vector<float> g0, g1;
    env.get_graph(f0, e0, c0, g0, k0);
    const auto g_t0 = std::chrono::steady_clock::now();
    for (int r = 0; r < kRep; ++r) env.graph_refresh();
    env.sync();
    graph_ms = std::chrono::duration<double>(std::chrono::steady_clock::now() - g_t0).count() / kRep * 1e3;
    env.get_graph(f1, e1, c1, g1, k1);
    graph_same = f0 == f1 && e0 == e1 && c0 == c1 && g0 == g1 && k0 == k1;
  }
  std::vector<sim::EnvState> st;
  env.get_state(st);
  double cov = 0;
  for (const auto& e : st) cov += e.coverage;
  uint64_t h = 1469598103934665603ull;   // 상태 해시 (FNV-1a)
  {
    const uint8_t* p = reinterpret_cast<const uint8_t*>(st.data());
    for (size_t i = 0; i < st.size() * sizeof(sim::EnvState); ++i) { h ^= p[i]; h *= 1099511628211ull; }
  }
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, 0);
  const double sim_sec = (double)n_env * n_steps * hc.cfg.substeps * hc.cfg.dt;
  char dev[300];
  if (cpu) std::snprintf(dev, sizeof(dev), "CPU 참조판 %d 스레드", threads);
  else std::snprintf(dev, sizeof(dev), "GPU %s%s", prop.name, std::getenv("SIM_NO_OPTIX") ? " (소프트 순회)" : "");
  std::printf("%s: 환경 %d × 행동 %d = %.1f ms/걸음, 초당 행동 %.0f (시뮬 시간 %.0f 배속), 라이다 %.0f 장/초\n", dev, n_env,
              n_steps, sec / n_steps * 1e3, n_env * n_steps / sec, sim_sec / sec,
              (double)n_env * n_steps * (hc.cfg.substeps / hc.cfg.lidar_every) / sec);
  if (hier) std::printf("%s: 결정 하나 = 걸음 %d 개 → 초당 걸음 %.0f\n", graph ? "그래프 모드" : "계층형", hc.cfg.decide_steps, (double)n_env * n_steps * hc.cfg.decide_steps / sec);
  if (graph) std::printf("그래프 관측만: 환경 %d 개 한 번 %.3f ms (20 번 평균), 다시 낸 값 %s\n", n_env, graph_ms, graph_same ? "그대로" : "달라짐");
  std::printf("무작위 행동 %d 걸음 뒤 평균 덮은 비율 %.3f, 상태 해시 %016llx\n", n_steps + 1, cov / n_env, (unsigned long long)h);
  return 0;
}
