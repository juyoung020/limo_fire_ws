// GPU 환경 = 실물 (전체): 실물 녹화를 그 녹화의 공간(Cartographer 지도 → map2scene)에서 다시 돌려 비교한다.
//   구동 (ROS 명령 녹화만): 실물 /cmd_vel 을 50 Hz 그대로 GPU 환경 커널(속도 행동·한 걸음씩)에 넣어 바퀴 v·ω · 몸통 ω(자이로) · 궤적
//   라이다 (모든 녹화): 같은 시각 실물 Cartographer 자세에서 실물 오차 라이다(lidar_real_gpu)를 쏴 실물 /scan 과 빔마다,
//     빠짐은 조건(모서리·입사각·물체 크기·거리·회전 속도)별로 실물과 모델 기대값(같은 기하 코드 beam_geom)을 나란히
//   replay_check --set analysis/calibration.yaml fit|validate [--fit-dropout] [--env ..] [--model ..]
//   replay_check <장면.scene> <구동.csv>[@뺄 구간] <녹화.lfrec> [--map 지도.yaml] [--from 초] [--to 초] [--floor office] …
//   --fit-dropout: 빠짐 매개변수(점 비율 · 작은 물체 재질 배율)를 실물 빠짐에 맞춘다 (log-loss 최소, 확실한 벽의 모서리 아닌 빔만).
//     모서리·비스듬함·무작위 빠짐은 실물 스캔 자체로 모서리·면 방향을 정하는 정지 검사(noise_check)가 정한다 — 지도로 세운 장면은
//     Cartographer 가 모서리를 몇 cm 흐리게 하고 5 cm 칸 등고선의 면 방향이 흔들려 이 둘을 가르지 못한다
// 빠짐 맞추기·비교는 지도에서 확실한 칸(확률 ≥ 80 %)을 맞힌 빔만 — 녹화 중 지나간 사람처럼 지도에 남은 가짜 장애물을 뺀다.
#include <cuda_runtime.h>
#include <yaml-cpp/yaml.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <functional>
#include <sstream>
#include <string>
#include <vector>

#include "calibration.h"
#include "env_api.h"
#include "sensors_api.h"

namespace {

struct Row {
  double t, stamp;
  int mode;
  float cmd_v, cmd_w, odom_v, odom_w, gyro;
  bool skip;
};

struct Scan {
  double stamp;
  float angle_min, inc, time_inc, scan_time, x, y, yaw, v, w;
  std::vector<float> r;
};

struct Job {
  std::string name, scene, csv, lfrec, map, floor;
  std::vector<std::pair<double, double>> skip;
  double from = -1, to = 1e18;
};

std::vector<Row> load_csv(const std::string& path, const std::vector<std::pair<double, double>>& skips) {
  std::ifstream f(path);
  if (!f) throw std::runtime_error("못 읽음: " + path);
  std::string line;
  std::getline(f, line);
  if (line.find("stamp") == std::string::npos) throw std::runtime_error(path + ": stamp 열 없음 — motion_export 를 다시");
  std::vector<Row> rows;
  while (std::getline(f, line)) {
    std::stringstream ss(line);
    Row r{};
    char c;
    float rc_v, rc_w;
    ss >> r.t >> c >> r.mode >> c >> rc_v >> c >> rc_w >> c >> r.cmd_v >> c >> r.cmd_w >> c >> r.odom_v >> c >> r.odom_w >> c >> r.gyro >> c >>
        r.stamp;
    for (const auto& [a, b] : skips) r.skip = r.skip || (r.t >= a && r.t <= b);
    rows.push_back(r);
  }
  return rows;
}

std::vector<Scan> load_lfrec(const std::string& path) {
  FILE* f = std::fopen(path.c_str(), "rb");
  if (!f) throw std::runtime_error("못 읽음: " + path);
  char magic[4];
  uint32_t ver, n;
  auto rd = [&](void* p, size_t k) { if (std::fread(p, 1, k, f) != k) throw std::runtime_error(path + ": 잘림"); };
  rd(magic, 4);
  rd(&ver, 4);
  if (std::memcmp(magic, "LFRC", 4) != 0 || ver != 1) throw std::runtime_error(path + ": .lfrec 아님");
  rd(&n, 4);
  std::vector<Scan> scans(n);
  for (auto& s : scans) {
    float meta[4], pose[3], vel[2];
    uint32_t k;
    rd(&s.stamp, 8); rd(meta, 16); rd(pose, 12); rd(vel, 8); rd(&k, 4);
    s.angle_min = meta[0]; s.inc = meta[1]; s.time_inc = meta[2]; s.scan_time = meta[3];
    s.x = pose[0]; s.y = pose[1]; s.yaw = pose[2]; s.v = vel[0]; s.w = vel[1];
    s.r.resize(k);
    rd(s.r.data(), 4 * k);
  }
  std::fclose(f);
  return scans;
}

// scale 지도 (bag_export): 칸 확률 = (255 − 픽셀) / 255, 모름 205
struct ProbMap {
  int w = 0, h = 0;
  float res = 0, ox = 0, oy = 0;
  std::vector<uint8_t> px;   // 위 행이 y 가 큰 쪽
  bool ok() const { return w > 0; }
  // 맞은 점 둘레 3×3 칸 중 가장 높은 확률 (모름은 0)
  float prob(float x, float y) const {
    const int cx = (int)std::floor((x - ox) / res), cy = (int)std::floor((y - oy) / res);
    float best = 0;
    for (int dy = -1; dy <= 1; ++dy)
      for (int dx = -1; dx <= 1; ++dx) {
        const int X = cx + dx, Y = cy + dy;
        if (X < 0 || Y < 0 || X >= w || Y >= h) continue;
        const uint8_t v = px[(size_t)(h - 1 - Y) * w + X];
        if (v != 205) best = std::max(best, (255 - v) / 255.f);
      }
    return best;
  }
};

ProbMap load_map(const std::string& yaml_path) {
  ProbMap m;
  const YAML::Node y = YAML::LoadFile(yaml_path);
  if (y["mode"].as<std::string>() != "scale") throw std::runtime_error(yaml_path + ": scale 지도가 아님 (bag_export 를 다시)");
  m.res = y["resolution"].as<float>();
  m.ox = y["origin"][0].as<float>();
  m.oy = y["origin"][1].as<float>();
  const std::string dir = yaml_path.substr(0, yaml_path.find_last_of('/') + 1);
  FILE* f = std::fopen((dir + y["image"].as<std::string>()).c_str(), "rb");
  if (!f) throw std::runtime_error("지도 그림 못 읽음");
  char magic[3] = {};
  int maxv;
  if (std::fscanf(f, "%2s %d %d %d", magic, &m.w, &m.h, &maxv) != 4) throw std::runtime_error("PGM 머리");
  std::fgetc(f);
  m.px.resize((size_t)m.w * m.h);
  if (std::fread(m.px.data(), 1, m.px.size(), f) != m.px.size()) throw std::runtime_error("PGM 잘림");
  std::fclose(f);
  return m;
}

float median(std::vector<float> v) {
  if (v.empty()) return NAN;
  std::nth_element(v.begin(), v.begin() + v.size() / 2, v.end());
  return v[v.size() / 2];
}
float quant(std::vector<float> v, float q) {
  if (v.empty()) return NAN;
  std::sort(v.begin(), v.end());
  return v[(size_t)(q * (v.size() - 1))];
}

// 빔 하나의 빠짐 조건 (분석·맞추기)
struct Feat {
  sim::BeamGeom g;
  float size, w;
  bool real_drop;
};

template <class T>
T* upload(const std::vector<T>& v) {
  T* d = nullptr;
  cudaMalloc(&d, v.size() * sizeof(T));
  cudaMemcpy(d, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice);
  return d;
}

// 녹화 하나: 구동(ROS 명령이면)·궤적·라이다 비교를 찍고, 확실한 벽 빔의 빠짐 조건을 feats 에 더한다
void run_job(const Job& job, const sim::EnvHostCfg& hc_in, const sim::LidarModel& lidar, const YAML::Node& floors,
             std::vector<Feat>& feats) {
  sim::EnvHostCfg hc = hc_in;
  hc.cfg.action_mode = sim::kActVelocity;
  hc.cfg.substeps = 1;
  hc.cfg.max_steps = 1 << 30;
  sim::Floor ground{};
  bool found = false;
  for (const auto& f : floors)
    if (f["name"].as<std::string>() == job.floor) {
      auto mid = [&](const char* key, int k) { return 0.5f * (f[key][k][0].as<float>() + f[key][k][1].as<float>()); };
      ground = sim::Floor{mid("slip", 0), mid("slip", 1), mid("traction", 0), mid("traction", 1), mid("lag", 0), mid("lag", 1)};
      found = true;
    }
  if (!found) throw std::runtime_error("env.yaml floors 에 " + job.floor + " 없음");
  const std::vector<Row> rows = load_csv(job.csv, job.skip);
  const std::vector<Scan> scans = load_lfrec(job.lfrec);
  const ProbMap pmap = job.map.empty() ? ProbMap{} : load_map(job.map);
  size_t n_rc = 0;
  for (const auto& r : rows) n_rc += r.mode == 3;
  const bool rc = n_rc * 2 > rows.size();
  const sim::Scene scene = sim::load_scene(job.scene);
  std::printf("\n## %s (%s)\n", job.name.c_str(), rc ? "리모컨 — 구동은 학습 경로가 아니라 라이다만" : "ROS 명령");

  // 구간
  size_t i0 = 0;
  if (job.from < 0) {
    while (i0 < rows.size() && std::fabs(rows[i0].cmd_v) + std::fabs(rows[i0].cmd_w) == 0.f && !rc) ++i0;
    i0 = i0 >= 50 ? i0 - 50 : 0;
  } else {
    while (i0 < rows.size() && rows[i0].t < job.from) ++i0;
  }
  double st0 = rows[i0].stamp, st1 = rows.back().stamp;
  for (const auto& r : rows) if (r.t <= job.to) st1 = r.stamp;

  if (!rc) {
    auto pose_at = [&](double st, float* x, float* y, float* yaw) {
      auto it = std::lower_bound(scans.begin(), scans.end(), st, [](const Scan& s, double t) { return s.stamp < t; });
      if (it == scans.begin() || it == scans.end()) return false;
      const Scan& b = *it;
      const Scan& a = *(it - 1);
      const double f = (st - a.stamp) / (b.stamp - a.stamp);
      *x = (float)(a.x + f * (b.x - a.x));
      *y = (float)(a.y + f * (b.y - a.y));
      *yaw = (float)(a.yaw + f * std::remainder(b.yaw - a.yaw, 2 * M_PI));
      return true;
    };
    float x0, y0, th0;
    if (!pose_at(rows[i0].stamp, &x0, &y0, &th0)) throw std::runtime_error(job.name + ": 시작 시각에 Cartographer 자세 없음 (replay_from)");
    // 실물 녹화 지도는 일부만 그려 닫히지 않을 수 있다 — 여기선 덮음을 안 쓰므로 열린 지도도 받는다(알리기만)
    const std::vector<sim::World> worlds{sim::build_world(job.scene, scene, hc, false)};
    if (!worlds[0].closed) std::printf("(덮음 안 씀) %s\n", worlds[0].leak.c_str());
    sim::Env env(hc, worlds, lidar, 1, 3, true);
    env.reset();
    env.sync();
    std::vector<sim::EnvState> st;
    env.get_state(st);
    sim::EnvState e = st[0];
    e.x = x0;
    e.y = y0;
    e.th = th0;
    e.ex = e.ey = e.eth = 0.f;
    e.ground = ground;
    e.noise = lidar.nominal;
    sim::drive_reset(e.drive);
    env.set_state(0, e);
    float* d_act;
    cudaMalloc(&d_act, sizeof(float) * mdp::act::DIM);
    double ev = 0, ew = 0, eg = 0, sv = 0, mv = 0, sw = 0, mw = 0, sg = 0, mg = 0;
    size_t n = 0;
    struct P { double stamp; float x, y, th; };
    std::vector<P> path;
    bool collided = false;
    for (size_t i = i0; i < rows.size() && rows[i].t <= job.to; ++i) {
      const float act[mdp::act::DIM] = {rows[i].cmd_v, 0.f, rows[i].cmd_w, 0.f};
      cudaMemcpy(d_act, act, sizeof(act), cudaMemcpyHostToDevice);
      env.step(d_act);
      env.sync();
      env.get_state(st);
      if (st[0].tick == 0 || st[0].collided) { collided = true; break; }
      path.push_back({rows[i].stamp, st[0].x, st[0].y, st[0].th});
      if (rows[i].skip) continue;
      const Row& r = rows[i];
      ev += (st[0].drive.wv - r.odom_v) * (st[0].drive.wv - r.odom_v);
      ew += (st[0].drive.ww - r.odom_w) * (st[0].drive.ww - r.odom_w);
      eg += (st[0].w - r.gyro) * (st[0].w - r.gyro);
      sv += r.odom_v * r.odom_v; mv += r.odom_v;
      sw += r.odom_w * r.odom_w; mw += r.odom_w;
      sg += r.gyro * r.gyro; mg += r.gyro;
      ++n;
    }
    cudaFree(d_act);
    auto r2 = [&](double e2, double s2, double m) { return 1 - e2 / n / (s2 / n - (m / n) * (m / n)); };
    std::printf("%.1f~%.1f s, GPU 커널 %zu 걸음%s\n\n", rows[i0].t, rows[i0].t + (path.empty() ? 0 : path.back().stamp - path.front().stamp),
                path.size(), collided ? " — 시뮬에서 부딪혀 멈춤" : "");
    std::printf("| 구동 (GPU 커널) | RMSE | R² |\n|---|---|---|\n");
    std::printf("| 바퀴 v | %.3f m/s | %.3f |\n| 바퀴 ω | %.3f rad/s | %.3f |\n| 몸통 ω vs 자이로 | %.3f rad/s | %.3f |\n", std::sqrt(ev / n),
                r2(ev, sv, mv), std::sqrt(ew / n), r2(ew, sw, mw), std::sqrt(eg / n), r2(eg, sg, mg));
    std::vector<float> pe, he;
    for (const auto& s : scans) {
      if (path.empty() || s.stamp < path.front().stamp || s.stamp > path.back().stamp) continue;
      auto it = std::lower_bound(path.begin(), path.end(), s.stamp, [](const P& p, double t) { return p.stamp < t; });
      pe.push_back(std::hypot(it->x - s.x, it->y - s.y));
      he.push_back(std::fabs((float)std::remainder(it->th - s.yaw, 2 * M_PI)));
    }
    if (!pe.empty())
      std::printf("\n| 궤적 vs Cartographer | 중앙 | 끝 | 최대 |\n|---|---|---|---|\n| 위치 (m) | %.3f | %.3f | %.3f |\n| 방향 (°) | %.1f | %.1f | %.1f |\n",
                  median(pe), pe.back(), quant(pe, 1.f), median(he) * 57.2958f, he.back() * 57.2958f, quant(he, 1.f) * 57.2958f);
    if (!path.empty()) { st0 = path.front().stamp; st1 = path.back().stamp; }
  }

  // 라이다: 같은 시각 실물 Cartographer 자세에서 (자세 오차와 떼어 라이다만)
  std::vector<sim::ScanIn> ins;
  std::vector<const Scan*> used;
  uint64_t rng = 11;
  for (const auto& s : scans) {
    if (s.stamp < st0 || s.stamp > st1) continue;
    const float c = std::cos(s.yaw), sn = std::sin(s.yaw);
    sim::ScanIn in{};
    in.pose = sim::Pose{{c, -sn, 0, sn, c, 0, 0, 0, 1},
                        sim::V3{s.x + c * hc.cfg.mount_x - sn * hc.cfg.mount_y, s.y + sn * hc.cfg.mount_x + c * hc.cfg.mount_y, hc.cfg.lidar_z}};
    in.v = s.v;
    in.w = s.w;
    in.n_rev = std::round(2 * M_PI / s.inc);
    in.phase = sim::phase_of(lidar.nominal, dm::rand01(rng));
    in.seed = dm::splitmix64(rng);
    in.world_seed = 77;
    ins.push_back(in);
    used.push_back(&s);
  }
  if (ins.empty()) return;
  std::vector<sim::Node> nodes = scene.nodes;
  const sim::SceneView dev{upload(nodes), upload(scene.tris), upload(scene.obj), upload(scene.obj_size)};
  const std::vector<sim::LidarNoise> nz{lidar.nominal};
  float* d_out;
  cudaMalloc(&d_out, sizeof(float) * ins.size() * sim::kMaxBeams);
  sim::lidar_real_gpu(dev, upload(nz), upload(ins), (int)ins.size(), d_out);
  std::vector<float> out(ins.size() * sim::kMaxBeams);
  cudaMemcpy(out.data(), d_out, out.size() * sizeof(float), cudaMemcpyDeviceToHost);
  std::vector<float> diff;
  size_t both = 0, real_only = 0, sim_only = 0, none = 0, beams = 0, phantom = 0;
  const sim::SceneView view = scene.view();
  for (size_t k = 0; k < ins.size(); ++k) {
    const Scan& s = *used[k];
    const int nb = std::min<int>(sim::scan_beams(lidar.nominal, ins[k].n_rev), (int)s.r.size());
    for (int j = 0; j < nb; ++j) {
      const float a = s.r[j], b = out[k * sim::kMaxBeams + j];
      ++beams;
      if (a > 0 && b > 0) { ++both; diff.push_back(std::fabs(a - b)); }
      else if (a > 0) ++real_only;
      else if (b > 0) ++sim_only;
      else ++none;
      const sim::BeamGeom g = sim::beam_geom(view, lidar.nominal, ins[k], j);
      if (!g.hit || g.r < lidar.nominal.min_range || g.r >= lidar.nominal.max_range) continue;
      if (pmap.ok()) {
        const sim::V3 p = sim::add(g.o, sim::scale(g.d, g.r));
        if (pmap.prob(p.x, p.y) < 0.8f) { ++phantom; continue; }   // 지도에서 확실하지 않은 칸 (지나간 사람·흐릿한 것)
      }
      feats.push_back({g, scene.obj_size[g.obj], std::fabs(s.w), !(a > 0)});
    }
  }
  size_t near = 0;
  for (float d : diff) near += d < 0.05f;
  std::printf("\n| 라이다 (실물 자세에서, %zu 장 · 빔 %zu) | 값 |\n|---|---|\n", ins.size(), beams);
  std::printf("| 둘 다 값 | %.1f %% — 거리 차 중앙 %.3f m, p90 %.3f m, 5 cm 안 %.1f %% |\n", 100.0 * both / beams, median(diff), quant(diff, 0.9f),
              100.0 * near / std::max<size_t>(1, diff.size()));
  std::printf("| 실물만 값 | %.1f %% |\n| 시뮬만 값 | %.1f %% |\n| 둘 다 없음 | %.1f %% |\n| 빠짐 분석에서 뺀 빔 (지도 확률 < 80 %%) | %zu |\n",
              100.0 * real_only / beams, 100.0 * sim_only / beams, 100.0 * none / beams, phantom);
}

void drop_table(const std::vector<Feat>& fs, const sim::LidarNoise& N, const char* title) {
  struct Bin { const char* name; std::function<bool(const Feat&)> in; };
  const std::vector<Bin> bins = {
      {"전체", [](const Feat&) { return true; }},
      {"모서리", [](const Feat& f) { return f.g.edge; }},
      {"면 · 입사 0~30°", [](const Feat& f) { return !f.g.edge && f.g.cosi >= 0.866f; }},
      {"면 · 입사 30~60°", [](const Feat& f) { return !f.g.edge && f.g.cosi >= 0.5f && f.g.cosi < 0.866f; }},
      {"면 · 입사 60~90°", [](const Feat& f) { return !f.g.edge && f.g.cosi < 0.5f; }},
      {"작은 물체", [&](const Feat& f) { return f.size < N.structure_size; }},
      {"큰 구조물", [&](const Feat& f) { return f.size >= N.structure_size; }},
      {"거리 < 1 m", [](const Feat& f) { return f.g.r < 1.f; }},
      {"거리 1~3 m", [](const Feat& f) { return f.g.r >= 1.f && f.g.r < 3.f; }},
      {"거리 ≥ 3 m", [](const Feat& f) { return f.g.r >= 3.f; }},
      {"회전 < 0.3 rad/s", [](const Feat& f) { return f.w < 0.3f; }},
      {"회전 0.3~1.5", [](const Feat& f) { return f.w >= 0.3f && f.w < 1.5f; }},
      {"회전 ≥ 1.5", [](const Feat& f) { return f.w >= 1.5f; }},
  };
  std::printf("\n| 빠짐 %s | 빔 | 실물 | 모델 기대 | 차 |\n|---|---|---|---|---|\n", title);
  for (const auto& b : bins) {
    size_t n = 0, d = 0;
    double e = 0;
    for (const Feat& f : fs) if (b.in(f)) { ++n; d += f.real_drop; e += sim::expected_drop(N, f.g, f.size); }
    if (n) std::printf("| %s | %zu | %.3f | %.3f | %+.3f |\n", b.name, n, (double)d / n, e / n, e / n - (double)d / n);
  }
}

}  // namespace

int main(int argc, char** argv) {
  std::vector<std::string> pos;
  std::string env_path = SIM_CONFIG_DIR "/env.yaml", model_path = SIM_CONFIG_DIR "/sensor_model.yaml", floor_name = "office", map_path;
  std::string set_path, set_role;
  double t_from = -1, t_to = 1e18;
  bool fit_dropout = false;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    if (a == "--env" && i + 1 < argc) env_path = argv[++i];
    else if (a == "--model" && i + 1 < argc) model_path = argv[++i];
    else if (a == "--floor" && i + 1 < argc) floor_name = argv[++i];
    else if (a == "--map" && i + 1 < argc) map_path = argv[++i];
    else if (a == "--from" && i + 1 < argc) t_from = std::stod(argv[++i]);
    else if (a == "--to" && i + 1 < argc) t_to = std::stod(argv[++i]);
    else if (a == "--fit-dropout") fit_dropout = true;
    else if (a == "--set" && i + 2 < argc) { set_path = argv[i + 1]; set_role = argv[i + 2]; i += 2; }
    else pos.push_back(a);
  }
  std::vector<Job> jobs;
  if (!set_path.empty()) {
    for (const auto& c : sim::load_calibration(set_path, set_role))
      jobs.push_back({c.name, c.scene, c.csv, c.lfrec, c.map, c.floor, c.skip, c.replay_from, 1e18});
  } else if (pos.size() == 3) {
    Job j{pos[1], pos[0], pos[1], pos[2], map_path, floor_name, {}, t_from, t_to};
    const size_t at = pos[1].find('@');
    if (at != std::string::npos) {
      j.csv = pos[1].substr(0, at);
      std::stringstream ss(pos[1].substr(at + 1));
      std::string part;
      while (std::getline(ss, part, ',')) {
        const size_t c = part.find(':');
        j.skip.emplace_back(std::stod(part.substr(0, c)), std::stod(part.substr(c + 1)));
      }
    }
    jobs.push_back(j);
  } else {
    std::fprintf(stderr, "replay_check --set calibration.yaml fit|validate [--fit-dropout] | <장면.scene> <구동.csv>[@뺄 구간] <녹화.lfrec> [--map 지도.yaml] …\n");
    return 2;
  }
  sim::EnvHostCfg hc = sim::load_env_cfg(env_path);
  // 실물 /cmd_vel 을 그대로: 물리 단위 행동, 안전층·커리큘럼 끔 (녹화 재현이 목적)
  hc.cfg.act_raw = 1;
  hc.cfg.safety = 0;
  hc.cfg.curriculum = 0;
  const sim::LidarModel lidar = sim::load_lidar_model(model_path);
  const YAML::Node floors = YAML::LoadFile(env_path)["floors"];
  std::vector<Feat> feats;
  for (const auto& j : jobs) run_job(j, hc, lidar, floors, feats);

  std::printf("\n## 라이다 빠짐 — %zu 녹화의 확실한 벽 빔 %zu 개 모음\n", jobs.size(), feats.size());
  drop_table(feats, lidar.nominal, "(지금 설정)");
  if (fit_dropout) {
    sim::LidarNoise best = lidar.nominal;
    double best_ll = 1e300;
    std::vector<const Feat*> flat;
    for (const Feat& f : feats) if (!f.g.edge) flat.push_back(&f);
    for (float fs : {0.03f, 0.04f, 0.05f, 0.06f, 0.07f, 0.08f, 0.09f, 0.1f})
      for (float fsm : {0.f, 0.02f, 0.04f, 0.06f, 0.08f, 0.1f, 0.15f, 0.2f})
          for (float km : {0.f, 0.25f, 0.5f, 0.75f, 1.f, 1.5f, 2.f}) {
            sim::LidarNoise N = lidar.nominal;
            N.f_spot = fs;
            N.f_spot_mid = fsm;
            N.f_dark = lidar.nominal.f_dark * km;
            N.f_mid = lidar.nominal.f_mid * km;
            double ll = 0;
            for (const Feat* f : flat) {
              const double p = std::min(0.999, std::max(0.001, (double)sim::expected_drop(N, f->g, f->size)));
              ll -= f->real_drop ? std::log(p) : std::log(1 - p);
            }
            if (ll < best_ll) { best_ll = ll; best = N; }
          }
    const float km = lidar.nominal.f_dark > 0 ? best.f_dark / lidar.nominal.f_dark : 0.f;
    std::printf("\n맞춘 빠짐 (sensor_model.yaml lidar.dropout, 모서리 아닌 빔 %zu 개): spot [%.2f, %.3f, %.3f], "
                "작은 물체 재질 배율 %.2f → dark %.3f, mid %.3f\n", flat.size(), best.spot_size, best.f_spot, best.f_spot_mid, km, best.f_dark, best.f_mid);
    drop_table(feats, best, "(맞춘 값)");
  }
  return 0;
}
