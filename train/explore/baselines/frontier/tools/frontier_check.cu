// frontier_check — mdp/frontier.h 가 CPU 와 GPU 에서 같은 결과를 내는지(비트 비교).
//   1. 문턱 상수: kOccL 이 Cartographer 막힘 문턱(0.65)의 로그 오즈 × 1000 정수 경계와 같은지, kUnknownL == sim::kUnknown
//   2. 진짜 시뮬 지도(장면마다 GPU 환경을 무작위 속도 행동으로 돌려 여러 시점에 꺼냄)마다
//      a. 칸마다 프론티어 판정 + 칸 수: GPU 칸 하나 = 스레드 하나, atomicAdd  vs  CPU 순차
//      b. frontier_search(닫힘 + 덩어리 + BFS 거리, 로봇 반경 = nav2 robot_radius): GPU 환경 하나 = 스레드 하나  vs  CPU — 결과·덩어리 배열 바이트 비교
//   3. RL 관측 프론티어 요약(mdp/observations.h)의 크기 문턱·로봇 반경이 기준선 설정과 같은지
//   (관측 요약의 GPU 블록 병렬판 == CPU frontier_search 는 sim/tools/env_verify 가 관측 비트로 본다)
//   frontier_check <장면.scene> [장면 …] [--envs 4] [--steps 300] [--every 100] [--negative]
//     --negative: GPU 쪽 지도 칸 하나를 바꿔 넣어 검사가 틀림을 잡는지 (그때는 다름이 나와야 끝 코드 0)
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "env_api.h"
#include "frontier.h"
#include "frontier_explorer.h"

namespace {

constexpr int kMaxOut = 4096;

__global__ void k_cells(mdp::MapView m, uint8_t* mask, int* count) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= m.w * m.h) return;
  const int x = i % m.w, y = i / m.w;
  const bool f = mdp::frontier_cell(m, x, y);
  mask[i] = f;
  if (f) atomicAdd(count, 1);
}

__global__ void k_search(mdp::MapView m, int rx, int ry, mdp::Robot rb, int min_cells, mdp::SearchBuf b, mdp::Frontier* out, mdp::SearchResult* res) {
  if (blockIdx.x != 0 || threadIdx.x != 0) return;
  *res = mdp::frontier_search(m, rx, ry, rb, min_cells, b, out, kMaxOut);
}

void need(cudaError_t e, const char* w) {
  if (e != cudaSuccess) { std::fprintf(stderr, "%s: %s\n", w, cudaGetErrorString(e)); std::exit(1); }
}

}  // namespace

int main(int argc, char** argv) {
  std::vector<std::string> scenes;
  int n_env = 4, n_steps = 300, every = 100;
  bool negative = false;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    if (a == "--envs" && i + 1 < argc) n_env = std::stoi(argv[++i]);
    else if (a == "--steps" && i + 1 < argc) n_steps = std::stoi(argv[++i]);
    else if (a == "--every" && i + 1 < argc) every = std::stoi(argv[++i]);
    else if (a == "--negative") negative = true;
    else scenes.push_back(a);
  }
  if (scenes.empty()) { std::fprintf(stderr, "frontier_check <장면.scene> [장면 …] [--envs 4] [--steps 300] [--every 100] [--negative]\n"); return 2; }

  // 1. 상수
  const int occ = (int)std::ceil(1000.0 * std::log(mdp::kOccP / (1.0 - mdp::kOccP)));
  const bool const_ok = occ == mdp::kOccL && mdp::kUnknownL == sim::kUnknown;
  std::printf("문턱: 막힘 p > %.3f ⇔ l ≥ %d (상수 %d), 모름 %d == sim::kUnknown %d → %s\n", mdp::kOccP, occ, mdp::kOccL, mdp::kUnknownL,
              sim::kUnknown, const_ok ? "같음" : "다름");

  const sim::EnvHostCfg hc = sim::load_env_cfg(SIM_CONFIG_DIR "/env.yaml");
  const sim::LidarModel lidar = sim::load_lidar_model(SIM_CONFIG_DIR "/sensor_model.yaml");
  const int S = hc.cfg.size;
  const int min_cells = 15;   // explore.launch min_frontier_size 0.75 m / 0.05 m
  const fb::Params fp = fb::load_params(FRONTIER_CONFIG, SIM_CONFIG_DIR "/env.yaml", "nav2");
  const mdp::Robot rb = mdp::robot_cells(fp.robot_radius, hc.cfg.res);
  // 3. 관측 요약 상수 == 기준선 설정
  const bool obs_ok = mdp::obsgen::kRobotRadius == fp.robot_radius && mdp::obsgen::kFrontMinSize == fp.min_frontier_size &&
                      mdp::obsgen::reach_min_cells(hc.cfg) == min_cells;
  std::printf("관측 요약 상수: 로봇 반경 %.3f (기준선 %.3f), 크기 문턱 %.3f m = %d 칸 (기준선 %.3f m = %d 칸) → %s\n", mdp::obsgen::kRobotRadius,
              fp.robot_radius, mdp::obsgen::kFrontMinSize, mdp::obsgen::reach_min_cells(hc.cfg), fp.min_frontier_size, min_cells, obs_ok ? "같음" : "다름");
  size_t n_maps = 0, bad_cells = 0, bad_sum = 0, bad_search = 0, bad_dist = 0, total_frontier_cells = 0, total_clusters = 0, total_small = 0, n_closed = 0;
  for (const auto& sp : scenes) {
    std::vector<sim::World> worlds{sim::build_world(sp, sim::load_scene(sp), hc)};
    const sim::World& W = worlds[0];
    const int n = W.gw * W.gh;
    sim::Env env(hc, worlds, lidar, n_env, 11, true);
    env.reset();
    float* d_act;
    need(cudaMalloc(&d_act, sizeof(float) * mdp::act::DIM * n_env), "malloc");
    int16_t* dL;
    uint8_t *dmask, *dflag;
    int32_t *dqa, *dqb, *ddist;
    int* dcount;
    mdp::Frontier* dout;
    mdp::SearchResult* dres;
    need(cudaMalloc(&dL, sizeof(int16_t) * S * S), "malloc");
    need(cudaMalloc(&dmask, n), "malloc");
    need(cudaMalloc(&dflag, n), "malloc");
    need(cudaMalloc(&dqa, sizeof(int32_t) * n), "malloc");
    need(cudaMalloc(&dqb, sizeof(int32_t) * n), "malloc");
    need(cudaMalloc(&ddist, sizeof(int32_t) * n), "malloc");
    need(cudaMalloc(&dcount, sizeof(int)), "malloc");
    need(cudaMalloc(&dout, sizeof(mdp::Frontier) * kMaxOut), "malloc");
    need(cudaMalloc(&dres, sizeof(mdp::SearchResult)), "malloc");
    std::vector<uint8_t> flag(n), mask_c(n), mask_g(n);
    std::vector<int32_t> qa(n), qb(n), dist(n);
    std::vector<mdp::Frontier> out_c(kMaxOut), out_g(kMaxOut);
    uint64_t rng = 5;
    std::vector<sim::EnvState> st;
    std::vector<int16_t> L;
    for (int t = 1; t <= n_steps; ++t) {
      std::vector<float> act((size_t)mdp::act::DIM * n_env);
      for (int i = 0; i < n_env; ++i) sim::random_action(hc.cfg, rng, &act[(size_t)mdp::act::DIM * i]);
      cudaMemcpy(d_act, act.data(), sizeof(float) * mdp::act::DIM * n_env, cudaMemcpyHostToDevice);
      env.step(d_act);
      env.sync();
      if (t % every != 0) continue;
      env.get_state(st);
      env.get_map(L);
      for (int i = 0; i < n_env; ++i) {
        std::vector<int16_t> Lc(L.begin() + (size_t)i * S * S, L.begin() + (size_t)(i + 1) * S * S), Lg = Lc;
        int rx = (int)std::floor((st[i].x + st[i].ex - W.ox) / hc.cfg.res), ry = (int)std::floor((st[i].y + st[i].ey - W.oy) / hc.cfg.res);
        if (negative && i == 0) {   // 로봇에서 1 m 떨어진 갈 수 있는 칸 하나를 모름으로 (번진 칸 안 → 찾기도 달라져야 함)
          int best = -1, bd = 1 << 30;
          for (int k = 0; k < n; ++k) {
            if (mdp::cell_class(Lg[(size_t)(k / W.gw) * S + k % W.gw]) != mdp::kFree) continue;
            const int dx = k % W.gw - rx, dy = k / W.gw - ry, d = std::abs(dx * dx + dy * dy - 400);
            if (d < bd) { bd = d; best = k; }
          }
          if (best >= 0) Lg[(size_t)(best / W.gw) * S + best % W.gw] = mdp::kUnknownL;
        }
        const mdp::MapView mc{Lc.data(), S, W.gw, W.gh};
        // CPU
        int cnt_c = 0;
        for (int k = 0; k < n; ++k) {
          const int x = k % W.gw, y = k / W.gw;
          mask_c[k] = mdp::frontier_cell(mc, x, y);
          cnt_c += mask_c[k];
        }
        std::memset(out_c.data(), 0, sizeof(mdp::Frontier) * kMaxOut);
        const mdp::SearchResult rc =
            mdp::frontier_search(mc, rx, ry, rb, min_cells, mdp::SearchBuf{flag.data(), qa.data(), qb.data(), dist.data()}, out_c.data(), kMaxOut);
        // 거리를 켜도 기준선 결과(덩어리·닫힘)는 같아야 함: dist 없이 다시 찾아 거리 칸만 빼고 비교
        std::vector<mdp::Frontier> out_n(kMaxOut);
        std::memset(out_n.data(), 0, sizeof(mdp::Frontier) * kMaxOut);
        const mdp::SearchResult rn = mdp::frontier_search(mc, rx, ry, rb, min_cells, mdp::SearchBuf{flag.data(), qa.data(), qb.data()}, out_n.data(), kMaxOut);
        bool dist_same = std::memcmp(&rc, &rn, sizeof(rc)) == 0;
        for (int q = 0; q < std::min(kMaxOut, rc.n_found); ++q) {
          mdp::Frontier a = out_c[q], b2 = out_n[q];
          a.near_dist = b2.near_dist = 0;
          a.near_cell = b2.near_cell = 0;
          dist_same = dist_same && std::memcmp(&a, &b2, sizeof(a)) == 0 && out_c[q].near_dist >= 0;
        }
        bad_dist += !dist_same;
        // GPU
        cudaMemcpy(dL, Lg.data(), sizeof(int16_t) * S * S, cudaMemcpyHostToDevice);
        const mdp::MapView mg{dL, S, W.gw, W.gh};
        int zero = 0;
        cudaMemcpy(dcount, &zero, sizeof(int), cudaMemcpyHostToDevice);
        cudaMemset(dout, 0, sizeof(mdp::Frontier) * kMaxOut);
        k_cells<<<(n + 255) / 256, 256>>>(mg, dmask, dcount);
        k_search<<<1, 1>>>(mg, rx, ry, rb, min_cells, mdp::SearchBuf{dflag, dqa, dqb, ddist}, dout, dres);
        need(cudaDeviceSynchronize(), "kernel");
        int cnt_g;
        mdp::SearchResult rg;
        cudaMemcpy(&cnt_g, dcount, sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(mask_g.data(), dmask, n, cudaMemcpyDeviceToHost);
        cudaMemcpy(&rg, dres, sizeof(rg), cudaMemcpyDeviceToHost);
        cudaMemcpy(out_g.data(), dout, sizeof(mdp::Frontier) * kMaxOut, cudaMemcpyDeviceToHost);
        size_t bc = 0;
        for (int k = 0; k < n; ++k) bc += mask_c[k] != mask_g[k];
        bad_cells += bc;
        bad_sum += cnt_c != cnt_g;
        const bool sr = std::memcmp(&rc, &rg, sizeof(rc)) == 0 &&
                        std::memcmp(out_c.data(), out_g.data(), sizeof(mdp::Frontier) * std::min(kMaxOut, rc.n_found)) == 0;
        bad_search += !sr;
        ++n_maps;
        total_frontier_cells += cnt_c;
        total_clusters += rc.n_found;
        total_small += rc.n_small;
        n_closed += mdp::closed(rc);
        int near_d = -1;   // 문턱 넘는 덩어리 중 가장 가까운 BFS 거리
        for (int q = 0; q < std::min(kMaxOut, rc.n_found); ++q)
          if (near_d < 0 || out_c[q].near_dist < near_d) near_d = out_c[q].near_dist;
        std::printf("  %s 걸음 %d 환경 %d: 프론티어 칸 %d, 덩어리 %d (작아서 버림 %d, 가장 가까운 BFS %.2f m), 번진 칸 %d 안 프론티어 칸 %d, 닫힘 %d — 칸 다름 %zu, 칸 수 %s, 찾기 %s, 거리 켬/끔 %s\n",
                    W.name.c_str(), t, i, cnt_c, rc.n_found, rc.n_small, near_d < 0 ? -1.0 : near_d * hc.cfg.res, rc.n_reach, rc.n_cells, (int)mdp::closed(rc), bc,
                    cnt_c == cnt_g ? "같음" : "다름", sr ? "같음" : "다름", dist_same ? "같음" : "다름");
      }
    }
    for (void* p : std::initializer_list<void*>{d_act, dL, dmask, dflag, dqa, dqb, ddist, dcount, dout, dres}) cudaFree(p);
  }
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, 0);
  const bool same = bad_cells == 0 && bad_sum == 0 && bad_search == 0;   // GPU == CPU (--negative 가 어긋나게 하는 것)
  std::printf("GPU %s (sm_%d%d), 지도 %zu 장: 프론티어 칸 합 %zu, 덩어리 합 %zu (작은 것 %zu), 닫힌 지도 %zu\n", prop.name, prop.major, prop.minor, n_maps,
              total_frontier_cells, total_clusters, total_small, n_closed);
  std::printf("칸 판정 다름 %zu, 칸 수 다름 %zu, 찾기 다름 %zu%s, 거리 켬/끔 다름 %zu\n", bad_cells, bad_sum, bad_search, negative ? " (일부러 어긋나게)" : "",
              bad_dist);
  const bool ok = same && const_ok && obs_ok && bad_dist == 0;
  std::printf("%s\n", ok ? "프론티어 통과: GPU == CPU (비트 같음)" : "다름");
  if (negative) return same ? 1 : 0;
  return ok ? 0 : 1;
}
