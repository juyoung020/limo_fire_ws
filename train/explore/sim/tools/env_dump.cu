// 옛 빌드 == 새 빌드 비교용: 같은 장면·씨앗·행동열로 GPU 환경을 돌려 걸음마다 모든 출력(상태·보상·항·끝·판 정보·관측 2D·1D·가치 전용)과
// 끝에 지도·덮음 표시의 해시(FNV-1a 64)를 한 줄씩 찍는다. 커널을 고친 뒤 두 빌드의 출력을 diff 하면 시뮬 결과가 비트까지 같은지 안다.
// env_dump <장면.scene> [장면 …] [--list 목록] [--envs 256] [--steps 60] [--people 0] [--declare 0.01] [--seed 7]
#include <cstdio>
#include <string>
#include <vector>

#include "env_api.h"

namespace {
uint64_t fnv(const void* p, size_t n, uint64_t h = 1469598103934665603ull) {
  const uint8_t* b = (const uint8_t*)p;
  for (size_t i = 0; i < n; ++i) { h ^= b[i]; h *= 1099511628211ull; }
  return h;
}
template <class T>
uint64_t fnv_v(const std::vector<T>& v) { return fnv(v.data(), v.size() * sizeof(T)); }
}  // namespace

int main(int argc, char** argv) {
  std::vector<std::string> scenes;
  std::string env_path = SIM_CONFIG_DIR "/env.yaml", model_path = SIM_CONFIG_DIR "/sensor_model.yaml", list;
  int n_env = 256, n_steps = 60, people = 0;
  float p_declare = 0.01f;
  uint64_t seed = 7;
  int img_step = -1;
  std::string img_file;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    if (a == "--envs" && i + 1 < argc) n_env = std::stoi(argv[++i]);
    else if (a == "--steps" && i + 1 < argc) n_steps = std::stoi(argv[++i]);
    else if (a == "--people" && i + 1 < argc) people = std::stoi(argv[++i]);
    else if (a == "--declare" && i + 1 < argc) p_declare = std::stof(argv[++i]);
    else if (a == "--seed" && i + 1 < argc) seed = std::stoull(argv[++i]);
    else if (a == "--list" && i + 1 < argc) list = argv[++i];
    else if (a == "--img" && i + 2 < argc) { img_step = std::stoi(argv[++i]); img_file = argv[++i]; }   // 그 걸음 관측 영상을 파일로 (다른 칸 세기)
    else scenes.push_back(a);
  }
  sim::EnvHostCfg hc = sim::load_env_cfg(env_path);
  if (people > 0) hc.cfg.curriculum = 0;
  const sim::LidarModel lidar = sim::load_lidar_model(model_path);
  std::vector<sim::World> worlds;
  if (!list.empty()) worlds = sim::load_scene_list(list, hc, nullptr);
  for (const auto& p : scenes) worlds.push_back(sim::build_world(p, sim::load_scene(p), hc));
  sim::Env gpu(hc, worlds, lidar, n_env, seed, true);
  gpu.set_people_max(std::vector<int>(n_env, people));
  gpu.reset();
  constexpr int A = mdp::act::DIM;
  float* d_act;
  cudaMalloc(&d_act, sizeof(float) * A * n_env);
  uint64_t rng = 99 + seed;
  std::vector<sim::EnvState> st;
  std::vector<float> r, t, o1, pv;
  std::vector<int32_t> d;
  std::vector<sim::EpInfo> ep;
  std::vector<uint8_t> img;
  uint64_t all = 0;
  for (int s = 0; s < n_steps; ++s) {
    std::vector<float> act((size_t)A * n_env);
    for (int i = 0; i < n_env; ++i) sim::random_action(hc.cfg, rng, &act[(size_t)A * i], p_declare);
    cudaMemcpy(d_act, act.data(), sizeof(float) * A * n_env, cudaMemcpyHostToDevice);
    gpu.step(d_act);
    gpu.sync();
    gpu.get_state(st);
    gpu.get_reward(r, d);
    gpu.get_terms(t);
    gpu.get_ep(ep);
    gpu.get_obs(img, o1, pv);
    if (s == img_step) { FILE* f = std::fopen(img_file.c_str(), "wb"); std::fwrite(img.data(), 1, img.size(), f); std::fclose(f); }
    // 판 정보는 끝난 환경만 유효
    uint64_t he = 1469598103934665603ull;
    for (int i = 0; i < n_env; ++i)
      if (d[i] != 0) he = fnv(&ep[i], sizeof(sim::EpInfo), he);
    const uint64_t h[7] = {fnv_v(st), fnv_v(r), fnv_v(d), fnv_v(t), he, fnv_v(img), fnv(o1.data(), o1.size() * 4, fnv_v(pv))};
    std::printf("%d st %016llx r %016llx d %016llx t %016llx ep %016llx img %016llx 1d %016llx\n", s, (unsigned long long)h[0],
                (unsigned long long)h[1], (unsigned long long)h[2], (unsigned long long)h[3], (unsigned long long)h[4],
                (unsigned long long)h[5], (unsigned long long)h[6]);
    all = fnv(h, sizeof(h), all);
  }
  std::vector<int16_t> map;
  std::vector<uint8_t> seen;
  gpu.get_map(map);
  gpu.get_seen(seen);
  std::printf("map %016llx seen %016llx\n", (unsigned long long)fnv_v(map), (unsigned long long)fnv_v(seen));
  std::printf("all %016llx\n", (unsigned long long)fnv(&all, 8, fnv_v(map) ^ fnv_v(seen)));
  return 0;
}
