// 실물 측정 CSV → 시뮬·연기 모드 파라미터 후보. 값을 직접 고치지는 않는다 — 나온 후보를 보고 config 주석에 근거(날짜·녹화)와 함께 옮긴다.
//   real_calib echo  <파일.csv> [--max-angle 10]   반사 각도별 메아리 비율·거리 오차, 정면 거리 잡음
//   real_calib slip  <파일.csv>                     줄자 이동 / odom 이동 비율 (바퀴 반지름 배율 후보)
//   real_calib drift <파일.csv>                     IMU yaw 드리프트 (deg/min)
// 파일 형식은 real_calib.hpp 머리말. 맞추기(fit)와 확인(validate) 녹화는 따로 둔다(analysis/calibration.yaml).
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>

#include "real_calib.hpp"

using namespace real_calib;

namespace {
int usage() {
  std::fprintf(stderr,
               "사용: real_calib echo|slip|drift <파일.csv> [--max-angle 도]\n"
               "  echo : angle_deg,true_m,range_m[,material]\n  slip : true_m,odom_m\n  drift: t_s,yaw_deg\n");
  return 2;
}

int do_echo(std::istream& in, double max_angle) {
  size_t bad = 0;
  const auto rows = read_echo(in, &bad);
  if (rows.empty()) { std::fprintf(stderr, "읽은 줄이 없음\n"); return 1; }
  const auto bins = by_angle(rows);
  std::printf("# 반사 각도별 (줄 %zu, 건너뜀 %zu)\n각도°   n  메아리%%  편향 m  σ m  σ/참\n", rows.size(), bad);
  for (const auto& b : bins)
    std::printf("%5.1f %4zu %7.0f  %+.3f  %.3f  %.1f%%\n", b.angle, b.n, 100.0 * b.rate(), b.bias, b.sd, 100.0 * b.sd_rel);
  const auto sh = echo_shape(bins);
  std::printf("\n# 시뮬 반사 모양 후보 (잠정 기준: hard 22.5°, soft 25~55° 로 줄어듦 — isaac_sim_cross_check.md)\n");
  std::printf("항상 메아리(≥95%%)인 마지막 각도: %s\n", std::isnan(sh.full_deg) ? "없음(0° 근처도 95% 미만 — 마운트·벽 확인)" : (std::to_string(sh.full_deg) + "°").c_str());
  std::printf("거의 없음(≤5%%)이 되는 첫 각도: %s\n", std::isnan(sh.none_deg) ? "시험한 각도에서는 없음(더 큰 각도 필요)" : (std::to_string(sh.none_deg) + "°").c_str());
  const auto db = by_distance(rows, max_angle);
  if (!db.empty()) {
    std::printf("\n# 정면(|각도| ≤ %.0f°) 거리 잡음\n참 m    n  편향 m   σ m   σ/참\n", max_angle);
    std::vector<double> rel;
    for (const auto& d : db) {
      std::printf("%5.2f %4zu  %+.3f  %.3f  %.1f%%\n", d.true_m, d.n, d.bias, d.sd, 100.0 * d.sd / d.true_m);
      rel.push_back(d.sd / d.true_m);
    }
    std::printf("σ/참 평균 %.2f%% (시뮬 가정 1%%). 편향이 거리에 따라 늘면 음속(온도) 가정 확인(펌웨어 343 m/s, 20 °C).\n", 100.0 * stat(rel).mean);
  }
  return 0;
}

int do_slip(std::istream& in) {
  size_t used = 0;
  const auto s = read_slip(in, &used);
  if (!s) { std::fprintf(stderr, "읽은 줄이 없음 (true_m,odom_m)\n"); return 1; }
  std::printf("# 줄자 이동 / odom 이동 (줄 %zu)\n평균 %.4f  σ %.4f  (σ/√n %.4f)\n", used, s->ratio.mean, s->ratio.sd,
              used > 1 ? s->ratio.sd / std::sqrt(static_cast<double>(used)) : 0.0);
  std::printf("시뮬 가정 1.03: 실제 이동이 odom 의 1.03 배(odom 이 2.9%% 적게 셈). 값 > 1 이면 odom 이 실제보다 적게 셈.\n");
  return 0;
}

int do_drift(std::istream& in) {
  const auto d = read_drift(in);
  if (!d) { std::fprintf(stderr, "읽은 줄이 3 개 미만 (t_s,yaw_deg)\n"); return 1; }
  std::printf("# IMU yaw (줄 %zu)\n드리프트 %+.3f °/min, 직선에서 벗어남 σ %.3f°\n", d->n, d->deg_per_min, d->resid_sd_deg);
  std::printf("연기 모드 drift.yaw_per_rad·wall.q_yaw 근거로 쓴다(정지 기록과 주행 기록을 따로 재서 비교).\n");
  return 0;
}
}  // namespace

int main(int argc, char** argv) {
  if (argc < 3) return usage();
  const std::string mode = argv[1];
  double max_angle = 10.0;
  for (int i = 3; i + 1 < argc; i += 2)
    if (std::strcmp(argv[i], "--max-angle") == 0) max_angle = std::atof(argv[i + 1]);
  std::ifstream f(argv[2]);
  if (!f) { std::fprintf(stderr, "파일을 못 엶: %s\n", argv[2]); return 1; }
  if (mode == "echo") return do_echo(f, max_angle);
  if (mode == "slip") return do_slip(f);
  if (mode == "drift") return do_drift(f);
  return usage();
}
