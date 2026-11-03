// V3: 학습 시뮬 라이다 오차가 실물과 같은 분포인지. 장면 안 여러 자리에 세워 두고 장을 많이 만들어, 실물 녹화와 같은 통계 코드
// (limo_fire_sim/scan_stats.hpp, 실물은 scan_stats_bag)로 재서 비교한다. 허용 차이는 sensor_model.yaml check.
// 통과 기준은 센서 물리(표면 흔들림·모서리·비스듬함)만. 재질 섞임은 장소마다 달라 표로만 낸다.
// noise_check <장면.scene> [--model config/sensor_model.yaml] [--real analysis/real_scan_stats.yaml]
//             [--poses 24] [--scans 600] [--seed 2026] [--out 시뮬통계.yaml]
#include <cuda_runtime.h>
#include <yaml-cpp/yaml.h>

#include <cmath>
#include <cstdio>
#include <cstring>
#include <map>
#include <string>
#include <vector>

#include "limo_fire_sim/scan_stats.hpp"
#include "scene_host.h"
#include "sensor_model.h"
#include "sensors_api.h"

#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { std::fprintf(stderr, "CUDA %s: %s\n", #x, cudaGetErrorString(e)); return 1; } } while (0)

namespace {

template <class T>
T* upload(const std::vector<T>& v) {
  T* d = nullptr;
  cudaMalloc(&d, v.size() * sizeof(T));
  cudaMemcpy(d, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice);
  return d;
}

sim::Pose yaw_pose(float x, float y, float z, float yaw) {
  const float c = std::cos(yaw), s = std::sin(yaw);
  sim::Pose T{{c, -s, 0, s, c, 0, 0, 0, 1}, sim::V3{x, y, z}};
  return T;
}

std::vector<float> floats(const YAML::Node& n) {
  std::vector<float> v;
  for (const auto& x : n) v.push_back(x.as<float>());
  return v;
}

struct Row {
  std::string name;
  double sim, real, diff, tol;
  bool ratio;
};

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    std::fprintf(stderr, "noise_check <장면.scene> [--model ..] [--real ..] [--poses 24] [--scans 600] [--out ..]\n");
    return 2;
  }
  std::string model_path = SIM_CONFIG_DIR "/sensor_model.yaml", real_path = SIM_ANALYSIS_DIR "/real_scan_stats.yaml", out_path;
  int n_poses = 24, n_scans = 600;
  uint64_t rng = 2026;
  for (int i = 2; i + 1 < argc; i += 2) {
    if (!std::strcmp(argv[i], "--model")) model_path = argv[i + 1];
    else if (!std::strcmp(argv[i], "--real")) real_path = argv[i + 1];
    else if (!std::strcmp(argv[i], "--poses")) n_poses = std::stoi(argv[i + 1]);
    else if (!std::strcmp(argv[i], "--scans")) n_scans = std::stoi(argv[i + 1]);
    else if (!std::strcmp(argv[i], "--out")) out_path = argv[i + 1];
    else if (!std::strcmp(argv[i], "--seed")) rng = std::stoull(argv[i + 1]);
  }
  const sim::Scene scene = sim::load_scene(argv[1]);
  const sim::LidarModel model = sim::load_lidar_model(model_path);
  const YAML::Node real = YAML::LoadFile(real_path);
  const YAML::Node tol = YAML::LoadFile(model_path)["check"];

  // 자리 고르기: 참값 라이다가 거의 다 맞고(≥ 95 %) 너무 붙지 않은(≥ 0.25 m) 곳 — 실물 녹화처럼 방 안
  const sim::LidarSpec spec;
  const auto& root = scene.nodes[0];
  std::vector<sim::Pose> poses;
  for (int tries = 0; (int)poses.size() < n_poses && tries < 200000; ++tries) {
    const float x = dm::rand_range(rng, root.lo[0], root.hi[0]), y = dm::rand_range(rng, root.lo[1], root.hi[1]);
    const sim::Pose T = yaw_pose(x, y, 0.111f, dm::rand_range(rng, -3.14159f, 3.14159f));
    int hit = 0;
    float mn = 1e9f;
    for (int j = 0; j < spec.n; ++j) {
      const float r = sim::lidar_beam(scene.view(), spec, T, j);
      if (r > 0) { ++hit; mn = std::min(mn, r); }
    }
    if (hit >= 0.95 * spec.n && mn >= 0.25f) poses.push_back(T);
  }
  if ((int)poses.size() < n_poses) { std::fprintf(stderr, "자리를 %zu 곳만 찾음\n", poses.size()); return 1; }

  // 자리마다 판 하나(표면 성질 고정), 장마다 점 수·위상·난수만 바뀜. 오차 배율은 실측 그대로(무작위화 안 함)
  std::vector<sim::ScanIn> scans((size_t)n_poses * n_scans);
  for (int p = 0; p < n_poses; ++p) {
    const uint64_t world = dm::splitmix64(rng);
    for (int k = 0; k < n_scans; ++k) {
      sim::ScanIn& in = scans[(size_t)p * n_scans + k];
      in.pose = poses[p];
      in.v = in.w = 0.f;
      in.n_rev = sim::sample_n_rev(model.nominal, dm::rand01(rng));
      in.phase = sim::phase_of(model.nominal, dm::rand01(rng));
      in.seed = dm::splitmix64(rng);
      in.world_seed = world;
      in.noise = 0;
      in.pad = 0;
    }
  }
  std::vector<sim::Node> nodes = scene.nodes;
  const sim::SceneView dev{upload(nodes), upload(scene.tris), upload(scene.obj), upload(scene.obj_size)};
  const std::vector<sim::LidarNoise> noise{model.nominal};
  sim::LidarNoise* d_noise = upload(noise);
  sim::ScanIn* d_scans = upload(scans);
  float* d_out;
  CK(cudaMalloc(&d_out, sizeof(float) * scans.size() * sim::kMaxBeams));
  sim::lidar_real_gpu(dev, d_noise, d_scans, (int)scans.size(), d_out);
  CK(cudaDeviceSynchronize());
  std::vector<float> out(scans.size() * sim::kMaxBeams);
  CK(cudaMemcpy(out.data(), d_out, out.size() * sizeof(float), cudaMemcpyDeviceToHost));

  // 실물과 같게: 자리마다 가장 흔한 빔 수의 장만, 빔 간격은 그 장들의 평균
  std::vector<scan_stats::Set> sets;
  std::vector<float> n_rev_all;
  for (int p = 0; p < n_poses; ++p) {
    std::map<int, scan_stats::Set> by_n;
    std::map<int, double> inc_sum;
    for (int k = 0; k < n_scans; ++k) {
      const size_t i = (size_t)p * n_scans + k;
      const int nb = sim::scan_beams(model.nominal, scans[i].n_rev);
      by_n[nb].scans.emplace_back(out.begin() + i * sim::kMaxBeams, out.begin() + i * sim::kMaxBeams + nb);
      inc_sum[nb] += sim::kTwoPi / scans[i].n_rev;
      n_rev_all.push_back(scans[i].n_rev);
    }
    auto best = by_n.begin();
    for (auto it = by_n.begin(); it != by_n.end(); ++it)
      if (it->second.scans.size() > best->second.scans.size()) best = it;
    best->second.inc = inc_sum[best->first] / best->second.scans.size();
    sets.push_back(std::move(best->second));
  }
  scan_stats::Summary sim_s = scan_stats::summarize(sets);
  sim_s.n_rev_q = scan_stats::n_rev_quantiles(n_rev_all);
  if (!out_path.empty()) {
    FILE* f = std::fopen(out_path.c_str(), "w");
    scan_stats::write_yaml(f, sim_s, std::string("학습 시뮬 noise_check ") + argv[1]);
    std::fclose(f);
  }

  // 비교: 센서 물리 (통과 기준)
  std::vector<Row> rows;
  auto absrow = [&](const char* name, double s, double r, const char* key) {
    rows.push_back({name, s, r, std::fabs(s - r), tol[key].as<double>(), false});
  };
  auto ratiorow = [&](const char* name, double s, double r, const char* key) {
    rows.push_back({name, s, r, std::fabs(s / r - 1), tol[key].as<double>(), true});
  };
  ratiorow("표면 빔 σ 중앙 (m)", sim_s.surface_sigma_med, real["surface"]["sigma_med"].as<double>(), "surface_sigma_ratio");
  ratiorow("표면 빔 σ p90 (m)", sim_s.surface_sigma_p90, real["surface"]["sigma_p90"].as<double>(), "surface_sigma_p90_ratio");
  absrow("모서리 빔 빠짐률", sim_s.edge_drop, real["edge"]["drop"].as<double>(), "edge_drop");
  absrow("모서리 빔 σ>3cm 비율", sim_s.edge_noisy, real["edge"]["noisy"].as<double>(), "edge_noisy");
  // 비스듬함: 입사 60~90° 와 0~30° 빠짐률 차 (재질 몫은 양쪽에 같이 들어 빠진다)
  const auto ri = floats(real["inc_drop"]);
  absrow("비스듬할 때 는 빠짐 (60~90° − 0~30°)", sim_s.inc_drop[2] - sim_s.inc_drop[0], ri[2] - ri[0], "graze_rise");

  std::printf("장면 %s: 자리 %d × 장 %d (빔 %d 개 모음), 실측 %s\n\n", argv[1], n_poses, n_scans, sim_s.beams, real_path.c_str());
  std::printf("| 항목 | 시뮬 | 실물 | 차이 | 허용 | |\n|---|---|---|---|---|---|\n");
  bool pass = true;
  for (const auto& r : rows) {
    const bool ok = std::isfinite(r.diff) && r.diff <= r.tol;
    pass = pass && ok;
    std::printf("| %s | %.4f | %.4f | %s%.3f | %.3f | %s |\n", r.name.c_str(), r.sim, r.real, r.ratio ? "×" : "", r.diff, r.tol,
                ok ? "통과" : "**넘음**");
  }
  // 재질 섞임 (참고)
  auto line = [](const char* name, const std::vector<float>& sv, const std::vector<float>& rv) {
    std::printf("| %s | ", name);
    for (float x : sv) std::printf("%.3f ", x);
    std::printf("| ");
    for (float x : rv) std::printf("%.3f ", x);
    std::printf("|\n");
  };
  std::printf("\n재질 섞임 — 장소마다 다름, 참고 (판마다 무작위화 폭 안에 들면 됨)\n\n| 항목 | 시뮬 | 실물 |\n|---|---|---|\n");
  line("빔별 유효율 분포 (0~5 %, 5~20, 20~50, 50~80, 80~95, 95~99, 99~100)", sim_s.valid_hist, floats(real["valid_hist"]));
  line("표면 빔 빠짐률", {sim_s.surface_drop}, {real["surface"]["drop"].as<float>()});
  line("입사각 0~30°·30~60°·60~90° 빠짐률", sim_s.inc_drop, floats(real["inc_drop"]));
  line("거리 0~0.5·~1.5·~3·~6 m 빠짐률", sim_s.range_drop, floats(real["range_drop"]));
  std::printf("\n%s\n", pass ? "V3 라이다 통과: 센서 물리 오차가 실측 허용 안" : "V3 라이다 넘음");
  return pass ? 0 : 1;
}
