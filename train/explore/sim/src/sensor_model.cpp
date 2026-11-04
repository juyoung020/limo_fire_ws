#include "sensor_model.h"

#include <yaml-cpp/yaml.h>

#include <stdexcept>

namespace sim {

namespace {
Range2 range2(const YAML::Node& n) { return Range2{n[0].as<float>(), n[1].as<float>()}; }
}  // namespace

LidarModel load_lidar_model(const std::string& path) {
  const YAML::Node l = YAML::LoadFile(path)["lidar"];
  if (!l) throw std::runtime_error(path + ": lidar 항목 없음");
  LidarModel m;
  LidarNoise& N = m.nominal;
  N.fov = l["fov"].as<float>();
  N.min_range = l["range"][0].as<float>();
  N.max_range = l["range"][1].as<float>();
  N.scan_time = l["scan_time"].as<float>();
  N.phase_jitter = l["phase_jitter"].as<float>();
  const YAML::Node q = l["n_rev_quantiles"];
  if (q.size() != 17) throw std::runtime_error(path + ": n_rev_quantiles 는 17 개");
  for (int i = 0; i < 17; ++i) N.n_rev_q[i] = q[i].as<float>();
  N.sigma0 = l["sigma"][0].as<float>();
  N.sigma1 = l["sigma"][1].as<float>();
  N.quantum = l["quantum"].as<float>();
  const YAML::Node d = l["dropout"];
  N.p_random = d["random"].as<float>();
  N.f_dark = d["dark"][0].as<float>();
  N.p_dark = d["dark"][1].as<float>();
  N.f_mid = d["mid"][0].as<float>();
  N.mid_lo = d["mid"][1].as<float>();
  N.mid_hi = d["mid"][2].as<float>();
  N.mid_sigma = d["mid"][3].as<float>();
  N.sigma_spread = l["sigma_spread"].as<float>();
  N.structure_size = d["structure_size"].as<float>();
  N.patch_size = d["graze"][0].as<float>();
  N.p_graze = d["graze"][1].as<float>();
  N.graze_flicker = d["graze"][2].as<float>();
  N.spot_size = d["spot"][0].as<float>();
  N.f_spot = d["spot"][1].as<float>();
  N.f_spot_mid = d["spot"][2].as<float>();
  const YAML::Node e = l["edge"];
  N.footprint = e["footprint"].as<float>();
  N.edge_gap = e["gap"].as<float>();
  N.p_edge_drop = e["p_drop"].as<float>();
  N.p_mix = e["p_mix"].as<float>();
  const YAML::Node it = l["intensity"];
  if (!it) throw std::runtime_error(path + ": lidar.intensity 항목 없음");
  if (it["range_table"].size() != 5) throw std::runtime_error(path + ": lidar.intensity.range_table 는 5 개 (거리 0.25·0.75·1.5·3·6 m)");
  for (int i = 0; i < 5; ++i) N.int_r[i] = it["range_table"][i].as<float>();
  N.int_c = it["cos_slope"].as<float>();
  N.int_obj = it["object_sigma"].as<float>();
  N.int_sigma = it["sigma"].as<float>();
  N.int_drop = it["dropped"].as<float>();
  m.latency = l["latency"].as<float>();
  const YAML::Node r = l["randomize"];
  m.r_sigma = range2(r["sigma"]);
  m.r_dropout = range2(r["dropout"]);
  m.r_edge_drop = range2(r["edge_drop"]);
  return m;
}

DepthModel load_depth_model(const std::string& path) {
  const YAML::Node d = YAML::LoadFile(path)["depth"];
  if (!d) throw std::runtime_error(path + ": depth 항목 없음");
  DepthModel m;
  DepthSpec& D = m.spec;
  D.w = d["size"][0].as<int>();
  D.h = d["size"][1].as<int>();
  D.fx = d["intrinsics"][0].as<float>();
  D.fy = d["intrinsics"][1].as<float>();
  D.cx = d["intrinsics"][2].as<float>();
  D.cy = d["intrinsics"][3].as<float>();
  D.min_z = d["range"][0].as<float>();
  D.max_z = d["range"][1].as<float>();
  D.obs_max_z = d["obs_max"].as<float>();
  if (!(D.obs_max_z > D.min_z && D.obs_max_z <= D.max_z)) throw std::runtime_error(path + ": depth.obs_max 는 range 안");
  const YAML::Node vr = d["valid_rect"];   // [row0, row1, col0, col1) — 실물 가장자리 늘 0 인 띠
  D.valid_row0 = vr[0].as<int>();
  D.valid_row1 = vr[1].as<int>();
  D.valid_col0 = vr[2].as<int>();
  D.valid_col1 = vr[3].as<int>();
  m.k = d["sigma_at_1m"].as<float>();
  m.near_x = d["sigma_near"][0].as<float>();
  m.near_z0 = d["sigma_near"][1].as<float>();
  m.near_z1 = d["sigma_near"][2].as<float>();
  m.quantum = d["quantum"].as<float>();
  const YAML::Node h = d["holes"];
  auto fill = [&](const char* key, float* out, int n) {
    if (!h[key] || (int)h[key].size() != n) throw std::runtime_error(path + ": depth.holes." + key + " 는 " + std::to_string(n) + " 개");
    for (int i = 0; i < n; ++i) out[i] = h[key][i].as<float>();
  };
  fill("floor_cos", m.floor_cos, kDepthFloorBins + 1);
  fill("floor", m.floor_hole, kDepthFloorBins);
  fill("floor_hi", m.floor_hole_hi, kDepthFloorBins);
  for (int i = 0; i < kDepthFloorBins; ++i)
    if (m.floor_hole_hi[i] < m.floor_hole[i]) throw std::runtime_error(path + ": depth.holes.floor_hi 는 floor 보다 작지 않게");
  fill("edge_z", m.edge_z, kDepthEdgeBins + 1);
  fill("edge", m.edge_hole, kDepthEdgeBins);
  m.other_hole = h["other"].as<float>();
  m.edge_px = h["edge_px"].as<float>();
  return m;
}

}  // namespace sim

namespace sim {

ImuModel load_imu_model(const std::string& path, float dt) {
  const YAML::Node d = YAML::LoadFile(path)["imu"];
  if (!d) throw std::runtime_error(path + ": imu 항목 없음");
  ImuModel m{};
  m.g = d["g"].as<float>();
  m.k_ax = d["k"][0].as<float>();
  m.k_ay = d["k"][1].as<float>();
  m.b_pitch = d["tilt"][0].as<float>();
  m.b_roll = d["tilt"][1].as<float>();
  m.tilt_sigma = d["tilt_sigma"].as<float>();
  m.gyro_bias = d["gyro_bias"].as<float>();
  const char* names[kImuCh] = {"ax", "ay", "az", "gx", "gy", "roll", "pitch"};
  for (int k = 0; k < kImuCh; ++k) {
    const YAML::Node c = d["noise"][names[k]];
    if (!c || c.size() != 5) throw std::runtime_error(path + ": imu.noise." + names[k] + " 는 [s0, q1, qv, qw, tau]");
    m.ch[k] = ImuChannel{c[0].as<float>(), c[1].as<float>(), c[2].as<float>(), c[3].as<float>(), c[4].as<float>(), 0.f};
  }
  imu_set_dt(m, dt);
  return m;
}

}  // namespace sim
