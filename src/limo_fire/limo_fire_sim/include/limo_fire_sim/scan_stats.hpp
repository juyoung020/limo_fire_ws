// 라이다 오차 통계: 세워 둔 로봇의 스캔 여러 장 → 빔마다 유효율·평균·흔들림 → 표면·모서리·입사각·거리별 요약.
// 실물 녹화(scan_stats_bag)와 학습 시뮬(train/explore/sim noise_check)이 같은 코드로 재서 비교한다 (V3). ROS 없음.
#pragma once
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

namespace scan_stats
{

// 거리·입사각 구간 (요약 표의 열)
inline const std::vector<float> kRangeEdges{0.f, 0.5f, 1.5f, 3.f, 6.f};
inline const std::vector<float> kIncEdges{0.f, 30.f, 60.f, 90.f};   // 도, 0 = 정면
inline const std::vector<float> kValidEdges{0.f, .05f, .2f, .5f, .8f, .95f, .99f, 1.001f};
constexpr float kEdgeJump = 0.15f;   // 이웃 빔 평균 거리 차가 이보다 크면 모서리 빔

struct Summary
{
  int scans = 0, beams = 0;
  std::vector<float> valid_hist;              // 빔별 유효율 구간(kValidEdges)에 든 빔 비율
  int surface_n = 0, edge_n = 0;
  float surface_drop = 0, surface_sigma_med = 0, surface_sigma_p90 = 0;   // 표면 빔(모서리 아님, 유효율 > 0.5)
  float edge_drop = 0, edge_noisy = 0;        // 모서리 빔: 빠짐률, σ > 3 cm 비율
  std::vector<float> range_drop, range_sigma; // 거리 구간별 빠짐률(유효율 > 0 인 빔) · 표면 빔 σ 중앙
  std::vector<int> range_n;
  std::vector<float> inc_drop;                // 입사각 구간별 빠짐률 (면을 알 수 있는 빔)
  std::vector<int> inc_n;
  std::vector<float> n_rev_q;                 // 한 바퀴 점 수 분위(0, 1/16, …, 1) — 모든 장(주행 포함)
};

inline float quantile(std::vector<float> v, float q)
{
  if (v.empty()) {return NAN;}
  std::sort(v.begin(), v.end());
  const float x = q * (v.size() - 1);
  const size_t i = static_cast<size_t>(x);
  return i + 1 < v.size() ? v[i] + (x - i) * (v[i + 1] - v[i]) : v[i];
}

inline int bucket(const std::vector<float> & edges, float x)
{
  for (size_t k = 0; k + 1 < edges.size(); ++k) {
    if (x >= edges[k] && x < edges[k + 1]) {return static_cast<int>(k);}
  }
  return -1;
}

// 한 자리에 세워 두고 잰 스캔 묶음 (빔 수가 같은 장들, 0 = 빠짐). inc: 빔 간격(rad)
struct Set
{
  std::vector<std::vector<float>> scans;
  float inc = 0;
};

// 빔마다: 유효율·평균·흔들림·모서리 여부·입사각(도, 모르면 NaN)
struct Beams
{
  std::vector<float> vr, mean, sd, inc_deg;
  std::vector<bool> edge;
};

inline void beam_stats(const Set & set, Beams & b)
{
  const auto & scans = set.scans;
  const int n = static_cast<int>(scans[0].size()), ns = static_cast<int>(scans.size());
  std::vector<float> vr(n), mean(n, NAN), sd(n, NAN);
  for (int j = 0; j < n; ++j) {
    double sum = 0, sq = 0;
    int k = 0;
    for (const auto & sc : scans) {
      if (sc[j] > 0) {sum += sc[j]; sq += double(sc[j]) * sc[j]; ++k;}
    }
    vr[j] = float(k) / ns;
    if (k > 0) {
      mean[j] = sum / k;
      sd[j] = std::sqrt(std::max(0.0, sq / k - (sum / k) * (sum / k)));
    }
  }
  // 모서리: 이웃과 평균 거리가 크게 다름
  std::vector<bool> edge(n, false);
  for (int j = 0; j + 1 < n; ++j) {
    if (std::isfinite(mean[j]) && std::isfinite(mean[j + 1]) && std::fabs(mean[j] - mean[j + 1]) > kEdgeJump) {
      edge[j] = edge[j + 1] = true;
    }
  }
  // 입사각: 앞뒤 2 빔(모서리 없이 이어진 면)에 맞춘 직선과 빔 방향 사이
  std::vector<float> inc_deg(n, NAN);
  for (int j = 2; j + 2 < n; ++j) {
    float px[5], py[5];
    int m = 0;
    bool flat = std::isfinite(mean[j]);
    for (int k = -2; k <= 2 && flat; ++k) {
      if (!std::isfinite(mean[j + k])) {continue;}
      if (std::fabs(mean[j + k] - mean[j]) > kEdgeJump) {flat = false; break;}
      const float a = (j + k) * set.inc;
      px[m] = mean[j + k] * std::cos(a);
      py[m] = mean[j + k] * std::sin(a);
      ++m;
    }
    if (!flat || m < 4) {continue;}
    float cx = 0, cy = 0;
    for (int i = 0; i < m; ++i) {cx += px[i] / m; cy += py[i] / m;}
    float sxx = 0, sxy = 0, syy = 0;
    for (int i = 0; i < m; ++i) {
      sxx += (px[i] - cx) * (px[i] - cx); sxy += (px[i] - cx) * (py[i] - cy); syy += (py[i] - cy) * (py[i] - cy);
    }
    const float th = 0.5f * std::atan2(2 * sxy, sxx - syy);   // 면 방향
    const float a = j * set.inc;
    // 입사각(면 법선과 빔 사이) = 90° − (면 방향과 빔 사이 각)
    const float c = std::fabs(std::cos(th) * std::cos(a) + std::sin(th) * std::sin(a));
    inc_deg[j] = 90.f - std::acos(std::min(1.f, c)) * 180.f / float(M_PI);
  }
  b.vr.insert(b.vr.end(), vr.begin(), vr.end());
  b.mean.insert(b.mean.end(), mean.begin(), mean.end());
  b.sd.insert(b.sd.end(), sd.begin(), sd.end());
  b.inc_deg.insert(b.inc_deg.end(), inc_deg.begin(), inc_deg.end());
  b.edge.insert(b.edge.end(), edge.begin(), edge.end());
}

// 여러 자리의 묶음을 빔 단위로 모아 요약
inline Summary summarize(const std::vector<Set> & sets)
{
  Summary s;
  Beams bs;
  for (const auto & set : sets) {
    if (set.scans.empty()) {continue;}
    s.scans += static_cast<int>(set.scans.size());
    beam_stats(set, bs);
  }
  const int n = static_cast<int>(bs.vr.size());
  s.beams = n;
  if (n == 0) {return s;}
  const auto & vr = bs.vr;
  const auto & mean = bs.mean;
  const auto & sd = bs.sd;
  const auto & edge = bs.edge;
  const auto & inc_deg = bs.inc_deg;
  s.valid_hist.assign(kValidEdges.size() - 1, 0.f);
  for (float v : vr) {s.valid_hist[bucket(kValidEdges, v)] += 1.f / n;}

  std::vector<float> surf_sd, surf_vr, edge_vr, edge_sd;
  for (int j = 0; j < n; ++j) {
    if (vr[j] <= 0.5f) {
      if (edge[j]) {edge_vr.push_back(vr[j]);}
      continue;
    }
    if (edge[j]) {edge_vr.push_back(vr[j]); edge_sd.push_back(sd[j]);} else {surf_vr.push_back(vr[j]); surf_sd.push_back(sd[j]);}
  }
  auto drop_of = [](const std::vector<float> & v) {
      float a = 0;
      for (float x : v) {a += 1 - x;}
      return v.empty() ? NAN : a / v.size();
    };
  s.surface_n = static_cast<int>(surf_vr.size());
  s.edge_n = static_cast<int>(edge_vr.size());
  s.surface_drop = drop_of(surf_vr);
  s.surface_sigma_med = quantile(surf_sd, 0.5f);
  s.surface_sigma_p90 = quantile(surf_sd, 0.9f);
  s.edge_drop = drop_of(edge_vr);
  int noisy = 0;
  for (float x : edge_sd) {noisy += x > 0.03f;}
  s.edge_noisy = edge_sd.empty() ? NAN : float(noisy) / edge_sd.size();

  const int nr = static_cast<int>(kRangeEdges.size()) - 1, ni = static_cast<int>(kIncEdges.size()) - 1;
  std::vector<std::vector<float>> rv(nr), rs(nr), iv(ni);
  for (int j = 0; j < n; ++j) {
    if (!std::isfinite(mean[j])) {continue;}
    const int b = bucket(kRangeEdges, mean[j]);
    if (b >= 0) {
      rv[b].push_back(vr[j]);
      if (!edge[j] && vr[j] > 0.5f) {rs[b].push_back(sd[j]);}
    }
    const int c = std::isfinite(inc_deg[j]) ? bucket(kIncEdges, inc_deg[j]) : -1;
    if (c >= 0) {iv[c].push_back(vr[j]);}
  }
  for (int b = 0; b < nr; ++b) {
    s.range_drop.push_back(drop_of(rv[b]));
    s.range_sigma.push_back(quantile(rs[b], 0.5f));
    s.range_n.push_back(static_cast<int>(rv[b].size()));
  }
  for (int c = 0; c < ni; ++c) {
    s.inc_drop.push_back(drop_of(iv[c]));
    s.inc_n.push_back(static_cast<int>(iv[c].size()));
  }
  return s;
}

// 가장 흔한 빔 수의 장만 남긴다 (실물은 장마다 빔 수가 바뀐다)
inline std::vector<std::vector<float>> mode_length(const std::vector<std::vector<float>> & scans)
{
  std::vector<int> cnt(1024, 0);
  for (const auto & s : scans) {if (s.size() < cnt.size()) {++cnt[s.size()];}}
  const size_t n = std::max_element(cnt.begin(), cnt.end()) - cnt.begin();
  std::vector<std::vector<float>> out;
  for (const auto & s : scans) {if (s.size() == n) {out.push_back(s);}}
  return out;
}

inline std::vector<float> n_rev_quantiles(std::vector<float> n_rev)
{
  std::vector<float> q;
  for (int k = 0; k <= 16; ++k) {q.push_back(quantile(n_rev, k / 16.f));}
  return q;
}

// YAML 로 쓴다 (noise_check 가 읽음)
inline void write_yaml(FILE * f, const Summary & s, const std::string & source)
{
  auto list = [&](const char * key, const auto & v, const char * fmt) {
      std::fprintf(f, "%s: [", key);
      for (size_t i = 0; i < v.size(); ++i) {std::fprintf(f, fmt, double(v[i])); std::fprintf(f, i + 1 < v.size() ? ", " : "");}
      std::fprintf(f, "]\n");
    };
  std::fprintf(f, "# 라이다 오차 통계 (scan_stats.hpp). 출처: %s\n", source.c_str());
  std::fprintf(f, "scans: %d\nbeams: %d   # 모든 자리의 빔 합\n", s.scans, s.beams);
  list("valid_hist", s.valid_hist, "%.4f");
  std::fprintf(f, "surface: {n: %d, drop: %.4f, sigma_med: %.5f, sigma_p90: %.5f}\n", s.surface_n, s.surface_drop,
    s.surface_sigma_med, s.surface_sigma_p90);
  std::fprintf(f, "edge: {n: %d, drop: %.4f, noisy: %.4f}\n", s.edge_n, s.edge_drop, s.edge_noisy);
  list("range_drop", s.range_drop, "%.4f");
  list("range_sigma", s.range_sigma, "%.5f");
  list("range_n", s.range_n, "%.0f");
  list("inc_drop", s.inc_drop, "%.4f");
  list("inc_n", s.inc_n, "%.0f");
  list("n_rev_quantiles", s.n_rev_q, "%.2f");
}

}  // namespace scan_stats
