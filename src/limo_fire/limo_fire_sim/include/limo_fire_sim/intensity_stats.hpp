// 라이다 세기(intensity) 통계: 세워 둔 로봇의 스캔 여러 장(거리·세기) → 빔마다 평균 세기·시간 σ, 거리·입사각 구간별 요약.
// 빔 기하(평균 거리·입사각·모서리)는 scan_stats.hpp 와 같은 코드. 실물 녹화(intensity_stats_bag)와 학습 시뮬(intensity_check)이 같이 쓴다. ROS 없음.
#pragma once
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include "limo_fire_sim/scan_stats.hpp"

namespace intensity_stats
{

inline const std::vector<float> kREdges{0.f, 0.5f, 1.f, 2.f, 4.f, 8.f};   // m
inline const std::vector<float> kCEdges{0.f, 0.5f, 0.87f, 1.01f};          // 입사 cos (60° 넘음 · 30~60° · 30° 안)

// 한 자리 묶음: 장마다 거리·세기 (빔 수 같음)
struct Set
{
  scan_stats::Set range;
  std::vector<std::vector<float>> inten;
};

struct Summary
{
  int beams = 0;
  std::vector<double> mean, spread, n;   // [r][c] 빔 평균 세기의 평균·빔 사이 σ·빔 수 (표면 빔: 모서리 아님, 유효율 > 0.5, 입사각 앎)
  double temporal = 0;                   // 표면 빔 시간 σ 중앙
  double drop_intensity = 0;             // 빠진(거리 0) 빔이 내는 세기 평균
  std::vector<float> q;                  // 표면 빔 평균 세기 분위 0.05 · 0.25 · 0.5 · 0.75 · 0.95
  double slope_r = 0, slope_c = 0, icept = 0;   // 표면 빔 평균 세기 = icept + slope_r·r + slope_c·cos (최소제곱)
};

// 빔 하나 표본 (요약 전)
struct Beam { float r, cosi, mean, sd; };

inline void beams_of(const Set & s, std::vector<Beam> & out, double & drop_sum, double & drop_n)
{
  scan_stats::Beams b;
  scan_stats::beam_stats(s.range, b);
  const int n = static_cast<int>(b.vr.size());
  for (int j = 0; j < n; ++j) {
    double sum = 0, sq = 0;
    int k = 0;
    for (size_t t = 0; t < s.inten.size(); ++t) {
      const float r = s.range.scans[t][j], i = s.inten[t][j];
      if (r > 0) {sum += i; sq += double(i) * i; ++k;} else {drop_sum += i; drop_n += 1;}
    }
    if (b.edge[j] || b.vr[j] <= 0.5f || !std::isfinite(b.inc_deg[j]) || k < 5) {continue;}
    const double m = sum / k;
    out.push_back({b.mean[j], std::cos(b.inc_deg[j] * float(M_PI) / 180.f), float(m), float(std::sqrt(std::max(0.0, sq / k - m * m)))});
  }
}

inline Summary summarize(const std::vector<Set> & sets)
{
  Summary s;
  std::vector<Beam> bs;
  double ds = 0, dn = 0;
  for (const auto & x : sets) {if (!x.inten.empty()) {beams_of(x, bs, ds, dn);}}
  s.beams = static_cast<int>(bs.size());
  s.drop_intensity = dn > 0 ? ds / dn : NAN;
  const int nr = static_cast<int>(kREdges.size()) - 1, nc = static_cast<int>(kCEdges.size()) - 1;
  s.mean.assign(nr * nc, 0);
  s.spread.assign(nr * nc, 0);
  s.n.assign(nr * nc, 0);
  std::vector<float> sds, ms;
  double a[3][3] = {}, y[3] = {};
  for (const auto & b : bs) {
    const int ri = scan_stats::bucket(kREdges, b.r), ci = scan_stats::bucket(kCEdges, b.cosi);
    sds.push_back(b.sd);
    ms.push_back(b.mean);
    const double x[3] = {1.0, b.r, b.cosi};
    for (int p = 0; p < 3; ++p) {y[p] += x[p] * b.mean; for (int q = 0; q < 3; ++q) {a[p][q] += x[p] * x[q];}}
    if (ri < 0 || ci < 0) {continue;}
    const int k = ri * nc + ci;
    s.mean[k] += b.mean;
    s.spread[k] += double(b.mean) * b.mean;
    s.n[k] += 1;
  }
  for (size_t k = 0; k < s.n.size(); ++k) {
    if (s.n[k] > 0) {
      const double m = s.mean[k] / s.n[k];
      s.spread[k] = std::sqrt(std::max(0.0, s.spread[k] / s.n[k] - m * m));
      s.mean[k] = m;
    } else {
      s.mean[k] = s.spread[k] = NAN;
    }
  }
  s.temporal = scan_stats::quantile(sds, 0.5f);
  for (float q : {0.05f, 0.25f, 0.5f, 0.75f, 0.95f}) {s.q.push_back(scan_stats::quantile(ms, q));}
  // 3 × 3 정규방정식
  for (int p = 0; p < 3; ++p) {
    for (int q = p + 1; q < 3; ++q) {
      const double f = a[q][p] / a[p][p];
      for (int j = p; j < 3; ++j) {a[q][j] -= f * a[p][j];}
      y[q] -= f * y[p];
    }
  }
  double c[3];
  for (int p = 2; p >= 0; --p) {
    double v = y[p];
    for (int j = p + 1; j < 3; ++j) {v -= a[p][j] * c[j];}
    c[p] = v / a[p][p];
  }
  s.icept = c[0];
  s.slope_r = c[1];
  s.slope_c = c[2];
  return s;
}

inline void write_yaml(FILE * f, const Summary & s, const std::string & source)
{
  std::fprintf(f, "# 라이다 세기 통계 (intensity_stats.hpp). 출처: %s\nbeams: %d   # 표면 빔 (모서리 아님, 유효율 > 0.5, 입사각 앎)\n", source.c_str(), s.beams);
  std::fprintf(f, "r_edges: [0, 0.5, 1, 2, 4, 8]\ncos_edges: [0, 0.5, 0.87, 1]\n# 빔 평균 세기의 평균 [r][cos]\nmean:\n");
  const int nc = static_cast<int>(kCEdges.size()) - 1, nr = static_cast<int>(kREdges.size()) - 1;
  auto table = [&](const std::vector<double> & v, const char * fmt) {
      for (int r = 0; r < nr; ++r) {
        std::fprintf(f, "  - [");
        for (int c = 0; c < nc; ++c) {std::fprintf(f, c ? ", " : ""); std::fprintf(f, fmt, v[r * nc + c]);}
        std::fprintf(f, "]\n");
      }
    };
  table(s.mean, "%.2f");
  std::fprintf(f, "spread:   # 같은 구간 빔 사이 σ (재질 차)\n");
  table(s.spread, "%.2f");
  std::fprintf(f, "n:\n");
  table(s.n, "%.0f");
  std::fprintf(f, "temporal_sigma: %.3f   # 표면 빔 시간 σ 중앙\ndrop_intensity: %.3f   # 빠진 빔(거리 0)의 세기 평균\n", s.temporal, s.drop_intensity);
  std::fprintf(f, "quantiles: [%.2f, %.2f, %.2f, %.2f, %.2f]   # 표면 빔 평균 세기 0.05 · 0.25 · 0.5 · 0.75 · 0.95\n", s.q[0], s.q[1], s.q[2], s.q[3], s.q[4]);
  std::fprintf(f, "fit: [%.3f, %.4f, %.3f]   # 평균 세기 = a + b·거리 + c·입사 cos (최소제곱)\n", s.icept, s.slope_r, s.slope_c);
}

}  // namespace intensity_stats
