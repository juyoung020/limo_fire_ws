// V0·V1 (환경): 같은 맵·씨앗·행동열이면 GPU 커널과 CPU 참조판의 상태·보상·지도·관측이 걸음마다 비트까지 같은지.
// env_verify <장면.scene> [장면 …] [--envs 8] [--steps 80] [--negative] [--env config/env.yaml] [--model config/sensor_model.yaml]
//            [--dump 지도.pgm]   (env 0 의 마지막 지도: 0 모름 128, 빈 칸 흼, 막힘 검음)
//   --negative: GPU 쪽 행동 하나를 바꿔 일부러 어긋나게 한다 (검사가 틀림을 잡는지)
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "env_api.h"

int main(int argc, char** argv) {
  std::vector<std::string> scenes;
  std::string env_path = SIM_CONFIG_DIR "/env.yaml", model_path = SIM_CONFIG_DIR "/sensor_model.yaml", dump;
  int n_env = 8, n_steps = 80;
  bool negative = false;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    if (a == "--envs" && i + 1 < argc) n_env = std::stoi(argv[++i]);
    else if (a == "--steps" && i + 1 < argc) n_steps = std::stoi(argv[++i]);
    else if (a == "--negative") negative = true;
    else if (a == "--env" && i + 1 < argc) env_path = argv[++i];
    else if (a == "--model" && i + 1 < argc) model_path = argv[++i];
    else if (a == "--dump" && i + 1 < argc) dump = argv[++i];
    else scenes.push_back(a);
  }
  if (scenes.empty()) { std::fprintf(stderr, "env_verify <장면.scene> [장면 …] [--envs 8] [--steps 80] [--negative]\n"); return 2; }
  const sim::EnvHostCfg hc = sim::load_env_cfg(env_path);
  const sim::LidarModel lidar = sim::load_lidar_model(model_path);
  std::vector<sim::World> worlds;
  for (const auto& p : scenes) {
    worlds.push_back(sim::build_world(p, sim::load_scene(p), hc));
    const auto& w = worlds.back();
    std::printf("맵 %s: %d×%d 칸, 덮을 빈 칸 %d (%.1f m²), 시작 후보 %zu\n", p.c_str(), w.gw, w.gh, w.n_gt,
                w.n_gt * hc.cfg.res * hc.cfg.res, w.spawn.size());
  }
  sim::Env gpu(hc, worlds, lidar, n_env, 7, true), cpu(hc, worlds, lidar, n_env, 7, false);
  gpu.reset();
  cpu.reset();
  gpu.sync();
  float* d_act;
  cudaMalloc(&d_act, sizeof(float) * 2 * n_env);
  uint64_t rng = 99;
  size_t bad_state = 0, bad_reward = 0;
  int first_bad = -1, ends[4] = {0, 0, 0, 0};
  double reward_sum = 0;
  std::vector<sim::EnvState> sg, sc;
  std::vector<float> rg, rc;
  std::vector<int32_t> dg, dc;
  for (int t = 0; t < n_steps; ++t) {
    std::vector<float> act(2 * n_env);
    for (int i = 0; i < n_env; ++i) sim::random_action(hc.cfg, rng, &act[2 * i]);
    std::vector<float> gact = act;
    if (negative && t == n_steps / 2) gact[0] = hc.cfg.action_mode == sim::kActVelocity ? -gact[0] : (float)(((int)gact[0] + 1) % (hc.cfg.n_dir * hc.cfg.n_dist));   // 속도 행동 1 m/s 더 · 목표점 행동 다음 번호
    cudaMemcpy(d_act, gact.data(), sizeof(float) * 2 * n_env, cudaMemcpyHostToDevice);
    gpu.step(d_act);
    cpu.step(act.data());
    gpu.sync();
    gpu.get_state(sg);
    cpu.get_state(sc);
    gpu.get_reward(rg, dg);
    cpu.get_reward(rc, dc);
    for (int i = 0; i < n_env; ++i) {
      const bool s_ok = std::memcmp(&sg[i], &sc[i], sizeof(sim::EnvState)) == 0;
      const bool r_ok = std::memcmp(&rg[i], &rc[i], 4) == 0 && dg[i] == dc[i];
      bad_state += !s_ok;
      bad_reward += !r_ok;
      if ((!s_ok || !r_ok) && first_bad < 0) first_bad = t;
      ends[dc[i]]++;
      reward_sum += rc[i];
    }
  }
  std::vector<int16_t> mg, mc;
  gpu.get_map(mg);
  cpu.get_map(mc);
  size_t bad_map = 0, known = 0;
  for (size_t i = 0; i < mg.size(); ++i) { bad_map += mg[i] != mc[i]; known += mc[i] != sim::kUnknown; }
  std::vector<uint8_t> og, oc, dep_g, dcp;
  std::vector<float> ng, nc;
  gpu.get_obs(og, ng, &dep_g);
  cpu.get_obs(oc, nc, &dcp);
  size_t bad_obs = 0, depth_valid = 0;
  for (size_t i = 0; i < og.size(); ++i) bad_obs += og[i] != oc[i];
  for (size_t i = 0; i < dep_g.size(); ++i) { bad_obs += dep_g[i] != dcp[i]; depth_valid += dcp[i] > 0; }
  for (size_t i = 0; i < ng.size(); ++i) bad_obs += std::memcmp(&ng[i], &nc[i], 4) != 0;

  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, 0);
  std::printf("GPU %s (sm_%d%d), 환경 %d × 행동 %d%s\n", prop.name, prop.major, prop.minor, n_env, n_steps,
              negative ? " (일부러 어긋나게)" : "");
  std::printf("판 끝: 성공 %d · 충돌 %d · 시간 %d, 행동당 평균 보상 %.4f, 지도에서 아는 칸 %zu\n", ends[1], ends[2], ends[3],
              reward_sum / (n_env * n_steps), known);
  std::printf("상태 다름 %zu, 보상·끝 다름 %zu (처음 걸음 %d), 지도 칸 다름 %zu, 관측 다름 %zu (깊이 관측 값 있음 %.1f %%)\n", bad_state,
              bad_reward, first_bad, bad_map, bad_obs, 100.0 * depth_valid / std::max<size_t>(1, dcp.size()));
  if (!dump.empty()) {
    const int S = hc.cfg.size;
    FILE* f = std::fopen(dump.c_str(), "wb");
    std::fprintf(f, "P5\n%d %d\n255\n", S, S);
    for (int y = S - 1; y >= 0; --y)
      for (int x = 0; x < S; ++x) {
        const int16_t v = mc[(size_t)y * S + x];
        const uint8_t p = v == sim::kUnknown ? 128 : (uint8_t)(255 - (v - hc.cfg.l_min) * 255 / (hc.cfg.l_max - hc.cfg.l_min));
        std::fputc(p, f);
      }
    std::fclose(f);
  }
  const bool same = bad_state == 0 && bad_reward == 0 && bad_map == 0 && bad_obs == 0;
  std::printf("%s\n", same ? "환경 V0 통과: GPU == CPU (비트 같음)" : "다름");
  return negative ? (same ? 1 : 0) : (same ? 0 : 1);
}
