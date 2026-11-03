// 시뮬 = 현실 (깊이 카메라): 실물 녹화 깊이 통계(limo_fire_sim depth_stats_bag → YAML)로 학습 시뮬 깊이 모델(sensor_model.yaml depth,
// sensor_model.h depth_sample)을 맞추고(--fit, fit 녹화), 다른 녹화(validate)에서 모델 예측과 실측을 비교한다.
//   depth_check [--set analysis/calibration.yaml] [--model config/sensor_model.yaml] [--fit]
// calibration.yaml depth: 목록의 녹화마다 data 폴더에 <bag>.depth.yaml 이 있어야 한다:
//   depth_stats_bag <data>/<bag>.depth.yaml ~/limo_bags/<bag> [--static] --cam-h <h> --cam-pitch <p>
// 비교 (허용 차이는 sensor_model.yaml check.depth_*):
//   카메라 자세 (바닥 면으로 잰 광축 높이·숙인 각) vs env.yaml body.camera_mount·camera_pitch 와 같은 값인 depth.mount 확인값
//   유효 영역 (늘 0 인 가장자리 띠), 시간 σ(z) = k·z² (거리 구간 가운데), 바닥 구멍 비율(입사 cos 구간), 모서리 구멍 비율(거리 구간),
//   벽·물체(바닥 아님, 모서리 아님) 구멍 비율
#include <yaml-cpp/yaml.h>

#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include "sensor_model.h"

namespace {

// depth_stats_bag 은 표본이 없는 칸을 nan 으로 쓴다 (yaml-cpp 는 .nan 만 읽음)
double num(const YAML::Node& x) { return x.as<std::string>() == "nan" ? NAN : x.as<double>(); }

std::vector<double> vec(const YAML::Node& n) {
  std::vector<double> v;
  for (const auto& x : n) v.push_back(num(x));
  return v;
}

struct Stat {
  std::string name;
  YAML::Node y;
};

// 여러 녹화 통계를 표본 수로 가중 평균 (구간 값 · 수)
std::vector<double> pooled(const std::vector<Stat>& s, const char* val, const char* cnt) {
  std::vector<double> sum, n;
  for (const auto& x : s) {
    if (!x.y[val]) continue;
    const auto v = vec(x.y[val]), c = vec(x.y[cnt]);
    if (sum.empty()) { sum.assign(v.size(), 0); n.assign(v.size(), 0); }
    for (size_t i = 0; i < v.size(); ++i)
      if (std::isfinite(v[i]) && c[i] > 0) { sum[i] += v[i] * c[i]; n[i] += c[i]; }
  }
  for (size_t i = 0; i < sum.size(); ++i) sum[i] = n[i] > 0 ? sum[i] / n[i] : NAN;
  return sum;
}
std::vector<double> pooled_n(const std::vector<Stat>& s, const char* cnt) {
  std::vector<double> n;
  for (const auto& x : s) {
    if (!x.y[cnt]) continue;
    const auto c = vec(x.y[cnt]);
    if (n.empty()) n.assign(c.size(), 0);
    for (size_t i = 0; i < c.size(); ++i) n[i] += c[i];
  }
  return n;
}

// 바닥 아님·모서리 아님 구멍 비율 (hole [z][cos] 전체)
double other_hole(const std::vector<Stat>& s) {
  double h = 0, n = 0;
  for (const auto& x : s)
    for (size_t z = 0; z < x.y["hole"].size(); ++z)
      for (size_t c = 0; c < x.y["hole"][z].size(); ++c) {
        const double r = num(x.y["hole"][z][c]), k = num(x.y["hole_n"][z][c]);
        if (std::isfinite(r)) { h += r * k; n += k; }
      }
  return n > 0 ? h / n : NAN;
}

}  // namespace

int main(int argc, char** argv) {
  std::string set_path = SIM_ANALYSIS_DIR "/calibration.yaml", model_path = SIM_CONFIG_DIR "/sensor_model.yaml";
  bool do_fit = false;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    if (a == "--set" && i + 1 < argc) set_path = argv[++i];
    else if (a == "--model" && i + 1 < argc) model_path = argv[++i];
    else if (a == "--fit") do_fit = true;
    else { std::fprintf(stderr, "depth_check [--set calibration.yaml] [--model sensor_model.yaml] [--fit]\n"); return 2; }
  }
  const YAML::Node y = YAML::LoadFile(set_path);
  const std::string dir = set_path.substr(0, set_path.find_last_of('/') + 1);
  const std::string data = dir + y["data"].as<std::string>() + "/";
  const std::string role = do_fit ? "fit" : "validate";
  std::vector<Stat> st;
  for (const auto& r : y["depth"]) {
    if (r["role"].as<std::string>() != role) continue;
    const std::string bag = r["bag"].as<std::string>();
    st.push_back({bag, YAML::LoadFile(data + bag + ".depth.yaml")});
    std::printf("%s %s: %s 장\n", role.c_str(), bag.c_str(), st.back().y["frames"].as<std::string>().c_str());
  }
  if (st.empty()) { std::fprintf(stderr, "depth 목록에 %s 녹화가 없음\n", role.c_str()); return 1; }
  const auto ze = vec(st[0].y["z_edges"]), fce = vec(st[0].y["floor_cos_edges"]);
  const auto tsd = pooled(st, "temporal_sigma", "temporal_n"), tn = pooled_n(st, "temporal_n");
  const auto fh = pooled(st, "floor_hole", "floor_n"), fn = pooled_n(st, "floor_n");
  const auto eh = pooled(st, "edge_hole", "edge_n"), en = pooled_n(st, "edge_n");
  const double oh = other_hole(st);
  double h = 0, p = 0;
  for (const auto& x : st) { h += x.y["floor_plane"][0].as<double>() / st.size(); p += x.y["floor_plane"][1].as<double>() / st.size(); }
  const auto rect = vec(st[0].y["valid_rect"]);

  if (do_fit) {
    // σ = k·z²: 거리 구간 가운데 z 로 최소제곱 (픽셀 수 가중)
    double a = 0, b = 0;
    for (size_t i = 0; i < tsd.size(); ++i) {
      if (!std::isfinite(tsd[i])) continue;
      const double z = 0.5 * (ze[i] + ze[i + 1]), z2 = z * z;
      a += tn[i] * tsd[i] * z2;
      b += tn[i] * z2 * z2;
    }
    std::printf("\n# sensor_model.yaml depth 에 넣을 값 (depth_check --fit)\n  sigma_at_1m: %.5f\n  valid_rect: [%.0f, %.0f, %.0f, %.0f]\n", a / b, rect[0],
                rect[1], rect[2], rect[3]);
    std::printf("  holes:\n    floor_cos: [");
    for (size_t i = 0; i < fce.size(); ++i) std::printf("%s%.2f", i ? ", " : "", fce[i]);
    std::printf("]\n    floor: [");
    for (size_t i = 0; i < fh.size(); ++i) std::printf("%s%.4f", i ? ", " : "", std::isfinite(fh[i]) ? fh[i] : 0.0);
    std::printf("]\n    other: %.4f\n    edge_z: [", oh);
    for (size_t i = 0; i < ze.size(); ++i) std::printf("%s%.2f", i ? ", " : "", ze[i]);
    std::printf("]\n    edge: [");
    for (size_t i = 0; i < eh.size(); ++i) std::printf("%s%.4f", i ? ", " : "", std::isfinite(eh[i]) ? eh[i] : 0.0);
    std::printf("]\n# env.yaml body: camera 광축 높이 %.4f m, 숙인 각 %.4f rad\n", h, p);
  }

  // 모델 (설정 파일) vs 이 녹화 묶음
  const sim::DepthModel M = sim::load_depth_model(model_path);
  const YAML::Node tol = YAML::LoadFile(model_path)["check"];
  const double t_sd = tol["depth_sigma_ratio"].as<double>(), t_hole = tol["depth_hole"].as<double>(), t_edge = tol["depth_edge_hole"].as<double>(),
               t_h = tol["depth_mount_h"].as<double>(), t_p = tol["depth_mount_pitch"].as<double>(), t_px = tol["depth_rect_px"].as<double>();
  const YAML::Node dm_ = YAML::LoadFile(model_path)["depth"];
  bool pass = true;
  std::printf("\n| 항목 | 시뮬 모델 | 실물 | 차이 | 허용 | |\n|---|---|---|---|---|---|\n");
  auto row = [&](const std::string& name, double s, double r, bool ratio, double t) {
    const double d = ratio ? std::fabs(s / r - 1) : std::fabs(s - r);
    const bool ok = std::isfinite(d) && d <= t;
    pass = pass && ok;
    std::printf("| %s | %.4f | %.4f | %s%.3f | %.3f | %s |\n", name.c_str(), s, r, ratio ? "×" : "", d, t, ok ? "통과" : "**넘음**");
  };
  row("카메라 광축 높이 (m)", dm_["mount_check"][0].as<double>(), h, false, t_h);
  row("카메라 숙인 각 (rad)", dm_["mount_check"][1].as<double>(), p, false, t_p);
  const int mr[4] = {M.spec.valid_row0, M.spec.valid_row1, M.spec.valid_col0, M.spec.valid_col1};
  const char* rn[4] = {"유효 영역 row0", "유효 영역 row1", "유효 영역 col0", "유효 영역 col1"};
  for (int i = 0; i < 4; ++i) row(rn[i], mr[i], rect[i], false, t_px);
  char buf[128];
  for (size_t i = 0; i < tsd.size(); ++i) {
    if (!std::isfinite(tsd[i]) || tn[i] < 5000) continue;
    const double z = 0.5 * (ze[i] + ze[i + 1]);
    std::snprintf(buf, sizeof(buf), "시간 σ z %.1f~%.1f m (픽셀 %.0f)", ze[i], ze[i + 1], tn[i]);
    row(buf, M.k * z * z, tsd[i], true, t_sd);
  }
  for (size_t i = 0; i < fh.size(); ++i) {
    if (!std::isfinite(fh[i]) || fn[i] < 1e5) continue;
    std::snprintf(buf, sizeof(buf), "바닥 구멍 cos %.2f~%.2f", fce[i], fce[i + 1]);
    row(buf, M.floor_hole[i], fh[i], false, t_hole);
  }
  for (size_t i = 0; i < eh.size(); ++i) {
    if (!std::isfinite(eh[i]) || en[i] < 1e4) continue;
    std::snprintf(buf, sizeof(buf), "모서리 구멍 z %.1f~%.1f m", ze[i], ze[i + 1]);
    row(buf, M.edge_hole[i], eh[i], false, t_edge);
  }
  row("벽·물체 구멍 (바닥·모서리 아님)", M.other_hole, oh, false, t_hole);
  std::printf("\n참고 (모델에 없음): 둘레로도 거리를 모르는 격자 픽셀 %.3f (범위 밖·큰 구멍 덩어리), 전체 유효율 %.3f, 유효 값 최소 분위 %s m\n",
              st[0].y["unknown"].as<double>(), st[0].y["valid"].as<double>(), YAML::Dump(st[0].y["valid_z_quantiles"]).c_str());
  std::printf("\n%s\n", pass ? "깊이 통과: 모델이 실측 허용 안" : "깊이 넘음");
  return pass ? 0 : 1;
}
