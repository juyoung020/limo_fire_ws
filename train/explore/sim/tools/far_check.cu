// 라이다 먼 거리 빠짐 vs 실물: 실물 정지 묶음(같은 자리에서 여러 장)과 같은 자리·같은 장 수로 시뮬 라이다(lidar_real_gpu)를 쏴
// 거리 구간별 "면 빔이 장마다 돌아오는 몫"을 같은 방식으로 잰다. 실측 표·자세는 analysis/real_far_stats.yaml (만든 방법은 그 파일 머리).
//   far_check <장면.scene> [--real analysis/real_far_stats.yaml] [--model config/sensor_model.yaml] [--role fit|validate] [--seeds 8] [--fit]
// 장면은 실측 묶음의 돌아온 점으로 세운 지도(map2scene) — 실측 "정답 거리"(돌아온 칸은 그 값, 안 돌아온 칸은 이 지도에 쏜 거리)와 같은 기하.
// 잼 (실측과 같은 규칙): 빔을 드라이버 각도(−110° + j·간격)로 2π/405 칸에 넣고, 칸마다 정답 거리 = 지도에 쏜 거리,
//   이웃 칸 정답 거리가 max(0.15, 0.04·r) 넘게 다르거나 없으면 모서리라 뺌. 구간마다 Σ돌아온 장 / Σ장 · 칸별 유효율 5~95 % 인 칸 몫.
// 시뮬은 같은 장(같은 난수)을 먼 거리 빠짐을 끈 모델(far.keep 모두 1)로도 쏴서 "먼 거리 몫" = 켬 / 끔 을 낸다 — 지도로 세운 장면은
//   벽이 조각나 작은 물체 재질 빠짐이 실물보다 많아(5 m 안 돌아옴도 실물보다 낮음) 절대값보다 이 몫을 실측 상대값(3~5 m 대비)과 견준다.
// --fit: 실측(그 역할) 3~5 m 대비 상대 돌아옴의 단조 감소 회귀(빔 수 가중) → sensor_model.yaml lidar.dropout.far 마디 후보
#include <cuda_runtime.h>
#include <yaml-cpp/yaml.h>

#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "scene_host.h"
#include "sensor_model.h"
#include "sensors_api.h"

namespace {

template <class T>
T* upload(const std::vector<T>& v) {
  T* d = nullptr;
  cudaMalloc(&d, v.size() * sizeof(T));
  cudaMemcpy(d, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice);
  return d;
}

std::vector<double> nums(const YAML::Node& n) {
  std::vector<double> v;
  for (const auto& x : n) v.push_back(x.as<double>());
  return v;
}

constexpr float kLidarZ = 0.111f;                    // env.yaml body.lidar_mount 높이
const float kInc0 = 2.f * (float)M_PI / 405.f;       // 각도 칸 (실측과 같게)
const float kAmin = -110.f * (float)M_PI / 180.f;

struct Acc {
  std::vector<double> k, n, beams, mid;
  explicit Acc(size_t nb) : k(nb), n(nb), beams(nb), mid(nb) {}
};

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    std::fprintf(stderr, "far_check <장면.scene> [--real ..] [--model ..] [--role fit|validate] [--seeds 8] [--fit]\n");
    return 2;
  }
  std::string real_path = SIM_ANALYSIS_DIR "/real_far_stats.yaml", model_path = SIM_CONFIG_DIR "/sensor_model.yaml", role = "fit";
  int n_seeds = 8;
  bool fit = false;
  for (int i = 2; i < argc; ++i) {
    if (!std::strcmp(argv[i], "--real") && i + 1 < argc) real_path = argv[++i];
    else if (!std::strcmp(argv[i], "--model") && i + 1 < argc) model_path = argv[++i];
    else if (!std::strcmp(argv[i], "--role") && i + 1 < argc) role = argv[++i];
    else if (!std::strcmp(argv[i], "--seeds") && i + 1 < argc) n_seeds = std::stoi(argv[++i]);
    else if (!std::strcmp(argv[i], "--fit")) fit = true;
    else { std::fprintf(stderr, "모르는 인자 %s\n", argv[i]); return 2; }
  }
  const sim::Scene scene = sim::load_scene(argv[1]);
  const sim::LidarModel model = sim::load_lidar_model(model_path);
  const sim::LidarNoise& N = model.nominal;
  const YAML::Node real = YAML::LoadFile(real_path);
  const std::vector<double> edges = nums(real["bins"]);
  const size_t nb = edges.size() - 1;
  if (!real[role]) { std::fprintf(stderr, "%s: %s 표 없음\n", real_path.c_str(), role.c_str()); return 2; }
  const std::vector<double> r_beams = nums(real[role]["beams"]), r_ret = nums(real[role]["ret"]), r_mid = nums(real[role]["mid"]);
  const sim::SceneView view = scene.view();
  auto bin_of = [&](float r) { for (size_t b = 0; b < nb; ++b) if (r >= edges[b] && r < edges[b + 1]) return (int)b; return -1; };

  std::vector<sim::Node> nodes = scene.nodes;
  const sim::SceneView dev{upload(nodes), upload(scene.tris), upload(scene.obj), upload(scene.obj_size)};
  sim::LidarNoise off = N;
  for (int i = 0; i < sim::kFarNodes; ++i) off.far_keep[i] = 1.f;
  const std::vector<sim::LidarNoise> nz{N, off};
  sim::LidarNoise* d_noise = upload(nz);
  Acc acc(nb), acc0(nb);
  uint64_t rng = 2027;
  int n_sets = 0, n_scans = 0;
  const int K = (int)std::ceil(220.f * (float)M_PI / 180.f / kInc0) + 2;
  for (const auto& st : real["sets"]) {
    if (st["role"].as<std::string>() != role) continue;
    ++n_sets;
    const std::vector<double> p = nums(st["pose"]);
    const int ns = st["scans"].as<int>();
    const float c = std::cos((float)p[2]), s = std::sin((float)p[2]);
    const sim::Pose T{{c, -s, 0, s, c, 0, 0, 0, 1}, sim::V3{(float)p[0], (float)p[1], kLidarZ}};
    // 칸마다 정답 거리 (드라이버 각도 방향, 지도에 쏨)
    std::vector<float> gt(K);
    for (int b = 0; b < K; ++b) {
      const float a = kAmin + b * kInc0;
      gt[b] = sim::raycast(view, T.p, sim::rotate(T, sim::V3{std::cos(a), std::sin(a), 0.f}), N.max_range, nullptr);
    }
    std::vector<int> bin(K, -1);
    for (int b = 1; b + 1 < K; ++b) {
      if (!(gt[b] < sim::kInf)) continue;
      bool edge = false;
      for (int o : {b - 1, b + 1}) edge = edge || !(gt[o] < sim::kInf) || std::fabs(gt[o] - gt[b]) > std::max(0.15f, 0.04f * gt[b]);
      if (!edge) bin[b] = bin_of(gt[b]);
    }
    for (int seed = 0; seed < n_seeds; ++seed) {
      const uint64_t world = dm::splitmix64(rng);   // 판(표면 성질)마다 다른 반사 순위
      std::vector<sim::ScanIn> scans(2 * ns);   // 앞 ns 장 켬, 뒤 ns 장 같은 난수로 끔
      for (auto& in : scans) {
        in = sim::ScanIn{};
        in.pose = T;
        in.n_rev = sim::sample_n_rev(N, dm::rand01(rng));
        in.phase = sim::phase_of(N, dm::rand01(rng));
        in.seed = dm::splitmix64(rng);
        in.world_seed = world;
      }
      for (int i = 0; i < ns; ++i) { scans[ns + i] = scans[i]; scans[ns + i].noise = 1; }
      float* d_out;
      cudaMalloc(&d_out, sizeof(float) * 2 * ns * sim::kMaxBeams);
      sim::ScanIn* d_scans = upload(scans);
      sim::lidar_real_gpu(dev, d_noise, d_scans, 2 * ns, d_out);
      std::vector<float> out((size_t)2 * ns * sim::kMaxBeams);
      cudaMemcpy(out.data(), d_out, out.size() * sizeof(float), cudaMemcpyDeviceToHost);
      cudaFree(d_out);
      cudaFree(d_scans);
      n_scans += ns;
      for (int half = 0; half < 2; ++half) {
        Acc& A = half ? acc0 : acc;
        std::vector<int> kk(K, 0), nn(K, 0);
        for (int i = 0; i < ns; ++i) {
          const float inc = sim::kTwoPi / scans[i].n_rev;
          const int n = sim::scan_beams(N, scans[i].n_rev);
          std::vector<char> seen(K, 0);
          for (int j = 0; j < n; ++j) {
            const float a = -0.5f * N.fov + j * inc;   // 드라이버가 붙이는 각도
            const int b = (int)std::lround((a - kAmin) / kInc0);
            if (b < 0 || b >= K || seen[b] || std::fabs(a - kAmin - b * kInc0) >= 0.5f * inc) continue;
            seen[b] = 1;
            ++nn[b];
            kk[b] += out[((size_t)half * ns + i) * sim::kMaxBeams + j] > 0.f;
          }
        }
        for (int b = 0; b < K; ++b) {
          if (bin[b] < 0 || nn[b] == 0) continue;
          const double vr = (double)kk[b] / nn[b];
          A.k[bin[b]] += kk[b];
          A.n[bin[b]] += nn[b];
          A.beams[bin[b]] += 1.0 / n_seeds;
          A.mid[bin[b]] += (vr >= 0.05 && vr <= 0.95) ? 1.0 / n_seeds : 0.0;
        }
      }
    }
  }
  std::printf("far_check %s — %s 묶음 %d 곳 × 판 %d (시뮬 %d 장), 실측 %s\n\n", argv[1], role.c_str(), n_sets, n_seeds, n_scans, real_path.c_str());
  std::printf("| 거리 m | 면 빔 실물 · 시뮬 | 돌아옴 실물 | 시뮬 (먼 빠짐 끔) | 실물 상대 (3~5 m 대비) | 시뮬 먼 거리 몫 (켬 / 끔) | 차 | 깜빡이는 빔(5~95 %%) 실물 · 시뮬 |\n|---|---|---|---|---|---|---|---|\n");
  const double rb = r_ret[0];
  for (size_t b = 0; b < nb; ++b) {
    const double sr = acc.n[b] > 0 ? acc.k[b] / acc.n[b] : NAN, s0 = acc0.n[b] > 0 ? acc0.k[b] / acc0.n[b] : NAN;
    const double sm = acc.beams[b] > 0 ? acc.mid[b] / acc.beams[b] : NAN, rr = r_beams[b] > 0 ? r_ret[b] / rb : NAN;
    std::printf("| %.1f~%.1f | %.0f · %.0f | %.3f | %.3f (%.3f) | %.3f | %.3f | %+.3f | %.2f · %.2f |\n", edges[b], edges[b + 1], r_beams[b], acc.beams[b],
                r_beams[b] > 0 ? r_ret[b] : NAN, sr, s0, rr, sr / s0, sr / s0 - rr, r_beams[b] > 0 ? r_mid[b] : NAN, sm);
  }
  if (fit) {
    // 실측 상대 돌아옴(3~5 m 대비, 1 로 자름)의 단조 감소 회귀 — 빔 수 가중 PAVA
    struct Blk { double y, w; size_t b0, b1; };
    std::vector<Blk> bl;
    for (size_t b = 1; b < nb; ++b)
      if (r_beams[b] > 0) bl.push_back({std::min(1.0, r_ret[b] / rb), r_beams[b], b, b});
    for (size_t i = 0; i + 1 < bl.size();) {
      if (bl[i].y < bl[i + 1].y) {
        bl[i] = {(bl[i].y * bl[i].w + bl[i + 1].y * bl[i + 1].w) / (bl[i].w + bl[i + 1].w), bl[i].w + bl[i + 1].w, bl[i].b0, bl[i + 1].b1};
        bl.erase(bl.begin() + i + 1);
        i = i > 0 ? i - 1 : 0;
      } else {
        ++i;
      }
    }
    std::printf("\n맞춘 보이는 몫 F(r) (%s, 3~5 m 돌아옴 %.3f 대비 단조 감소):\n", role.c_str(), rb);
    for (const auto& x : bl) std::printf("  %.1f~%.1f m: %.3f (빔 %.0f)\n", edges[x.b0], edges[x.b1 + 1], x.y, x.w);
  }
  return 0;
}
