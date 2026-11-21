// 초음파 오차 통계 (sonar_stats_bag 이 녹화에서 모은다, ROS 없이 시험). 결과로 sonar_sim_real.yaml 값을 정한다.
//   거리: 정지 구간마다 메아리 비율(= 1 − 놓침) · 평균 · 중앙값 · σ · 줄자 대비 치우침
//   주기: 메시지 사이 간격 (실물 다리 약 14 Hz, 2026-11-20)
//   시각: 메시지 시각 − 그때까지 받은 마지막 odom → base_link TF 시각 (양수 = 메아리가 TF 보다 앞섬, 2026-11-20 연기 모드가 메아리를 버린 까닭)
//         메시지 시각 − 녹화기가 받은 시각 (양수면 시계가 어긋남)
#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

namespace sonar_stats
{

// 값 모음 → 요약. 비면 n = 0, 나머지 NaN
struct Summary
{
  std::size_t n = 0;
  double mean = NAN, sd = NAN, p05 = NAN, p50 = NAN, p95 = NAN, min = NAN, max = NAN;
};

inline double quantile(std::vector<double> v, double q)   // 정렬한 사본에서 선형 보간
{
  if (v.empty()) {return NAN;}
  std::sort(v.begin(), v.end());
  const double x = q * static_cast<double>(v.size() - 1);
  const auto i = static_cast<std::size_t>(std::floor(x));
  const auto j = std::min(i + 1, v.size() - 1);
  return v[i] + (v[j] - v[i]) * (x - static_cast<double>(i));
}

inline Summary summarize(const std::vector<double> & v)
{
  Summary s;
  s.n = v.size();
  if (v.empty()) {return s;}
  double sum = 0;
  for (double x : v) {sum += x;}
  s.mean = sum / static_cast<double>(v.size());
  double ss = 0;
  for (double x : v) {ss += (x - s.mean) * (x - s.mean);}
  s.sd = v.size() > 1 ? std::sqrt(ss / static_cast<double>(v.size() - 1)) : 0.0;
  s.p05 = quantile(v, 0.05);
  s.p50 = quantile(v, 0.50);
  s.p95 = quantile(v, 0.95);
  s.min = *std::min_element(v.begin(), v.end());
  s.max = *std::max_element(v.begin(), v.end());
  return s;
}

// 센서 하나의 한 구간
struct Segment
{
  double t0 = 0, t1 = 0;           // 녹화 시각 (s, 녹화 처음 기준)
  std::size_t msgs = 0;            // 받은 메시지
  std::vector<double> echo;        // 메아리 있는 값 (m)
  std::vector<double> gap;         // 메시지 간격 (s)
  double last_rx = NAN;
  double no_echo_ratio = 0.99;     // 값 ≥ 이 비율 × max_range 면 메아리 없음 (sonar_sim.yaml no_echo_ratio 와 같게)

  void add(double rx, double range, double min_range, double max_range)
  {
    if (msgs == 0) {t0 = rx;}
    t1 = rx;
    if (std::isfinite(last_rx)) {gap.push_back(rx - last_rx);}
    last_rx = rx;
    ++msgs;
    // 다리 규약: 메아리 없음 = +inf. 시뮬·옛 판이 max_range 를 내도 메아리 없음으로 (no_echo_ratio 배 이상)
    if (std::isfinite(range) && range > min_range && range < no_echo_ratio * max_range) {echo.push_back(range);}
  }
  double echo_ratio() const {return msgs ? static_cast<double>(echo.size()) / static_cast<double>(msgs) : NAN;}
};

// 정지 판정: 바퀴 |v| · |ω| 가 문턱 아래인 채 hold 초 넘게 이어지면 정지
struct StillDetector
{
  double v_eps = 0.005, w_eps = 0.01, hold = 0.5;
  bool still = false;
  double since = NAN;

  // 반환: 지금 정지인가
  bool update(double t, double v, double w)
  {
    const bool quiet = std::fabs(v) < v_eps && std::fabs(w) < w_eps;
    if (!quiet) {
      since = NAN;
      still = false;
    } else {
      if (!std::isfinite(since)) {since = t;}
      still = (t - since) >= hold;
    }
    return still;
  }
};

}  // namespace sonar_stats
