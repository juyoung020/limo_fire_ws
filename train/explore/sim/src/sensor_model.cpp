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
  D.valid_row0 = 0;
  D.valid_row1 = D.h;
  D.valid_col0 = 0;
  D.valid_col1 = D.w;
  m.k = d["sigma_at_1m"].as<float>();
  m.quantum = d["quantum"].as<float>();
  return m;
}

}  // namespace sim
