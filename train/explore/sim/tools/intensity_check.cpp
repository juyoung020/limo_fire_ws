// 시뮬 = 현실 (라이다 세기): 실물 정지 녹화 세기 통계(limo_fire_sim intensity_stats_bag → YAML)로 학습 시뮬 세기 모델
// (lidar_real.h lidar_intensity, sensor_model.yaml lidar.intensity)을 맞추고(--fit) 다른 녹화(validate)에서 모델과 실측을 비교한다.
//   intensity_check [--set analysis/calibration.yaml] [--model config/sensor_model.yaml] [--fit]
// calibration.yaml intensity: {fit: <통계 파일>, validate: <통계 파일>} (data 폴더 안). 만들기:
//   intensity_stats_bag <data>/intensity_fit.yaml ~/limo_bags/static_1103_floor ~/limo_bags/1103_R1_spot1 ~/limo_bags/1103_R1_spot2
// 비교 (허용 차이는 sensor_model.yaml check.intensity_*): 거리·입사 cos 구간(빔 5 개 넘는 칸)마다 모델 평균(구간 가운데, intensity_mean) vs 실측 빔 평균,
//   빔 사이 σ(재질 차) vs 모델 물체 σ, 장마다 흔들림 σ, 빠진 빔 세기.
#include <yaml-cpp/yaml.h>

#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include "sensor_model.h"

namespace {
double num(const YAML::Node& x) { return x.as<std::string>() == "nan" ? NAN : x.as<double>(); }
}  // namespace

int main(int argc, char** argv) {
  std::string set_path = SIM_ANALYSIS_DIR "/calibration.yaml", model_path = SIM_CONFIG_DIR "/sensor_model.yaml";
  bool do_fit = false;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    if (a == "--set" && i + 1 < argc) set_path = argv[++i];
    else if (a == "--model" && i + 1 < argc) model_path = argv[++i];
    else if (a == "--fit") do_fit = true;
    else { std::fprintf(stderr, "intensity_check [--set ..] [--model ..] [--fit]\n"); return 2; }
  }
  const YAML::Node y = YAML::LoadFile(set_path);
  const std::string dir = set_path.substr(0, set_path.find_last_of('/') + 1);
  const std::string data = dir + y["data"].as<std::string>() + "/";
  const std::string role = do_fit ? "fit" : "validate";
  const YAML::Node st = YAML::LoadFile(data + y["intensity"][role].as<std::string>());
  std::printf("%s: %s\n", role.c_str(), y["intensity"][role].as<std::string>().c_str());
  const std::vector<double> rm{0.25, 0.75, 1.5, 3.0, 6.0}, cm{0.25, 0.685, 0.935};
  // 빔 사이 σ: 칸마다 σ² 를 빔 수로 가중 평균 (빔 5 개 넘는 칸)
  double sp = 0, spn = 0;
  for (size_t r = 0; r < rm.size(); ++r)
    for (size_t c = 0; c < cm.size(); ++c) {
      const double n = num(st["n"][r][c]), s = num(st["spread"][r][c]);
      if (n > 5 && std::isfinite(s)) { sp += n * s * s; spn += n; }
    }
  const double spread = std::sqrt(sp / spn);
  if (do_fit) {
    // 입사 cos 기울기 c = 표면 빔 최소제곱(a + b·r + c·cos)의 c. 거리 표 = 거리 구간마다 (빔 평균 − c·(cos 가운데 − 0.8)) 의 빔 수 가중 평균
    const double c = st["fit"][2].as<double>();
    std::printf("\n# sensor_model.yaml lidar.intensity 에 넣을 값 (intensity_check --fit)\n    range_table: [");
    for (size_t r = 0; r < rm.size(); ++r) {
      double s = 0, n = 0;
      for (size_t k = 0; k < cm.size(); ++k) {
        const double nn = num(st["n"][r][k]), m = num(st["mean"][r][k]);
        if (nn > 0 && std::isfinite(m)) { s += nn * (m - c * (cm[k] - 0.8)); n += nn; }
      }
      std::printf("%s%.2f", r ? ", " : "", n > 0 ? s / n : NAN);
    }
    std::printf("]\n    cos_slope: %.3f\n    object_sigma: %.3f\n    sigma: %.3f\n    dropped: %.2f\n", c, spread, st["temporal_sigma"].as<double>(),
                st["drop_intensity"].as<double>());
  }
  const sim::LidarModel M = sim::load_lidar_model(model_path);
  const sim::LidarNoise& N = M.nominal;
  const YAML::Node tol = YAML::LoadFile(model_path)["check"];
  const double t_mean = tol["intensity_mean"].as<double>(), t_ratio = tol["intensity_sigma_ratio"].as<double>();
  bool pass = true;
  std::printf("\n| 항목 | 시뮬 모델 | 실물 | 차이 | 허용 | |\n|---|---|---|---|---|---|\n");
  auto row = [&](const std::string& name, double s, double r, bool ratio, double t) {
    const double d = ratio ? std::fabs(s / r - 1) : std::fabs(s - r);
    const bool ok = std::isfinite(d) && d <= t;
    pass = pass && ok;
    std::printf("| %s | %.2f | %.2f | %s%.3f | %.2f | %s |\n", name.c_str(), s, r, ratio ? "×" : "", d, t, ok ? "통과" : "**넘음**");
  };
  char buf[128];
  for (size_t r = 0; r < rm.size(); ++r)
    for (size_t c = 0; c < cm.size(); ++c) {
      const double n = num(st["n"][r][c]), m = num(st["mean"][r][c]);
      if (!(n > 5) || !std::isfinite(m)) continue;
      std::snprintf(buf, sizeof(buf), "평균 세기 거리 %.2g m · 입사 cos %.2g (빔 %.0f)", rm[r], cm[c], n);
      row(buf, sim::intensity_mean(N, (float)rm[r], (float)cm[c]), m, false, t_mean);
    }
  row("빔 사이 σ (물체·재질 차)", N.int_obj, spread, true, t_ratio);
  row("장마다 흔들림 σ", N.int_sigma, st["temporal_sigma"].as<double>(), true, t_ratio);
  row("빠진 빔 세기", N.int_drop, st["drop_intensity"].as<double>(), false, t_mean);
  std::printf("\n참고: 실측 표면 빔 평균 세기 분위 0.05·0.5·0.95 = %.1f · %.1f · %.1f\n", st["quantiles"][0].as<double>(),
              st["quantiles"][2].as<double>(), st["quantiles"][4].as<double>());
  std::printf("\n%s\n", pass ? "세기 통과: 모델이 실측 허용 안" : "세기 넘음");
  return pass ? 0 : 1;
}
