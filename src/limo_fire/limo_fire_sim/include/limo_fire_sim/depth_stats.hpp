// 깊이 카메라 오차 통계: 16UC1 mm 영상 → 구멍(0) 비율을 거리·입사각·모서리별로, 공간 잔차 σ 를 거리별로, 세워 둔 녹화면 픽셀별 시간 σ.
// 실물 녹화(depth_stats_bag)와 학습 시뮬(train/explore/sim depth_check)이 같은 코드로 재서 비교한다. ROS 없음.
//
// 구멍인 픽셀은 거리를 모르므로 둘레로 정한다: 4 칸 격자의 픽셀마다 7 × 7 이웃(가운데 뺌)의 유효 값 중앙 = z_ref (유효 12 개 이상일 때만).
//   모서리: 이웃 유효 값 p90 − p10 > 0.05·z_ref + 0.02 m (깊이가 끊김 — 스테레오 가림 구멍이 생기는 곳)
//   입사 cos: (u ± 3, v), (u, v ± 3) 네 점(유효)으로 세운 면 법선과 광선 사이
// 공간 잔차: 모서리 아닌 픽셀에서 z(u) − (z(u−2) + z(u+2))/2 (셋 다 유효), 거리 구간마다 1.4826·MAD.
#pragma once
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

namespace depth_stats
{

inline const std::vector<float> kZEdges{0.3f, 0.5f, 0.8f, 1.2f, 1.8f, 2.5f, 3.2f};   // m
inline const std::vector<float> kCosEdges{0.f, 0.5f, 0.8f, 1.01f};                   // 입사 cos (1 = 정면)
constexpr int kStep = 4, kHalf = 3, kMinValid = 12;

struct Intr { float fx, fy, cx, cy; int w, h; float cam_h, cam_pitch; };   // 광축 높이 (m)·아래로 숙인 각 (rad) — 바닥 구멍 통계용

// 바닥 구멍: 행 v 광선이 바닥에 닿는 깊이(아래 add_frame)와 입사 cos (바닥 법선과 광선 사이).
// 그 행의 픽셀이 바닥 값(±8 %)이면 바닥 유효, 0 이면 바닥 구멍(가까운 물체 뒤 구멍도 섞임 — 바닥이 보이는 장만 쓰려고 행의 바닥 유효가
// 절반 넘는 행만), 다른 값이면 물체(뺌). 입사 cos 구간마다 구멍 비율
inline const std::vector<float> kFloorCos{0.f, 0.1f, 0.15f, 0.2f, 0.25f, 0.3f, 0.4f, 1.01f};
constexpr float kFloorZMax = 2.8f;   // 이보다 먼 바닥은 범위(3 m) 끝이라 뺌

inline int bucket(const std::vector<float> & e, float x)
{
  for (size_t k = 0; k + 1 < e.size(); ++k) {if (x >= e[k] && x < e[k + 1]) {return static_cast<int>(k);}}
  return -1;
}

// 모은 값 (여러 장·여러 녹화를 더한다)
struct Acc
{
  int nz = static_cast<int>(kZEdges.size()) - 1, nc = static_cast<int>(kCosEdges.size()) - 1;
  std::vector<double> hole, tot;          // [z][cos] 모서리 아님
  std::vector<double> ehole, etot;        // [z] 모서리
  std::vector<std::vector<float>> resid;  // [z] 공간 잔차 (표본)
  double frames = 0, px = 0, valid = 0, grid = 0, unknown = 0;   // 전체 유효율, 격자 픽셀 중 둘레로도 거리를 모르는 수
  std::vector<double> zhist;              // 유효 값 1 cm 구간 0~4 m
  std::vector<double> fhole, ftot;        // [바닥 cos 구간]
  std::vector<double> col_valid, row_valid;   // 열·행별 유효 수 (가장자리 늘 0 띠)
  std::vector<float> plane_x, plane_y;        // 바닥 면 맞추기 표본
  Acc() : hole(nz * nc, 0), tot(nz * nc, 0), ehole(nz, 0), etot(nz, 0), resid(nz), zhist(400, 0), fhole(kFloorCos.size() - 1, 0), ftot(kFloorCos.size() - 1, 0) {}
};

// 한 장 (mm, 행 우선)
inline void add_frame(Acc & a, const uint16_t * img, const Intr & K)
{
  a.frames += 1;
  const int W = K.w, H = K.h;
  auto z_at = [&](int u, int v) {return img[v * W + u] * 1e-3f;};
  if (a.col_valid.empty()) {a.col_valid.assign(W, 0); a.row_valid.assign(H, 0);}
  for (int v = 0; v < H; ++v) {
    for (int u = 0; u < W; ++u) {
      const float z = z_at(u, v);
      a.px += 1;
      if (z > 0) {a.col_valid[u] += 1; a.row_valid[v] += 1;}
      if (z > 0) {a.valid += 1; const int b = static_cast<int>(z * 100); if (b < 400) {a.zhist[b] += 1;}}
    }
  }
  // 바닥 면 맞추기 표본: 10 장마다, 지평선 30 행 아래 행마다 가운데 열들의 유효 값 중앙 → (x = (v − cy)/fy, 1/z)
  if (static_cast<long>(a.frames) % 10 == 1) {
    std::vector<float> row;
    for (int v = static_cast<int>(K.cy) + 30; v < H; v += 2) {
      row.clear();
      for (int u = W / 6; u < W - W / 6; ++u) {const float z = z_at(u, v); if (z > 0) {row.push_back(z);}}
      if (row.size() < static_cast<size_t>(W / 3)) {continue;}
      std::nth_element(row.begin(), row.begin() + row.size() / 2, row.end());
      a.plane_x.push_back((v - K.cy) / K.fy);
      a.plane_y.push_back(1.f / row[row.size() / 2]);
    }
  }
  // 바닥 행: 카메라가 pitch(아래로 +)만큼 숙이고 광축 높이 cam_h 면 행 v 의 바닥 깊이 z = cam_h / ((v − cy)/fy·cos + sin)
  const float cp = std::cos(K.cam_pitch), sp = std::sin(K.cam_pitch);
  for (int v = static_cast<int>(K.cy) + 3; v < H; ++v) {
    const float y = (v - K.cy) / K.fy, den = y * cp + sp;
    if (den <= 0) {continue;}
    const float zf = K.cam_h / den;
    if (zf > kFloorZMax) {continue;}
    double fv = 0, fh = 0, other = 0;
    for (int u = 10; u < W - 2; u += 2) {
      const float z = z_at(u, v);
      if (z <= 0) {fh += 1;} else if (std::fabs(z - zf) < 0.08f * zf) {fv += 1;} else {other += 1;}
    }
    if (fv < 0.5 * (fv + fh + other) || fv + fh < 20) {continue;}
    const float c = den / std::sqrt(1.f + y * y);   // 가운데 열 광선과 바닥 법선 (0, cos, sin) 사이 cos
    const int b = bucket(kFloorCos, c);
    if (b >= 0) {a.fhole[b] += fh; a.ftot[b] += fh + fv;}
  }
  float nb[48];
  for (int v = kHalf; v < H - kHalf; v += kStep) {
    for (int u = kHalf; u < W - kHalf; u += kStep) {
      a.grid += 1;
      int n = 0;
      for (int dv = -kHalf; dv <= kHalf; ++dv) {
        for (int du = -kHalf; du <= kHalf; ++du) {
          if (!du && !dv) {continue;}
          const float z = z_at(u + du, v + dv);
          if (z > 0) {nb[n++] = z;}
        }
      }
      if (n < kMinValid) {a.unknown += 1; continue;}
      std::sort(nb, nb + n);
      const float zr = nb[n / 2], p10 = nb[n / 10], p90 = nb[(9 * n) / 10];
      const int zb = bucket(kZEdges, zr);
      if (zb < 0) {continue;}
      const bool hole = z_at(u, v) <= 0;
      if (p90 - p10 > 0.05f * zr + 0.02f) {a.etot[zb] += 1; a.ehole[zb] += hole; continue;}
      // 입사 cos
      const float zl = z_at(u - kHalf, v), zrr = z_at(u + kHalf, v), zu = z_at(u, v - kHalf), zd = z_at(u, v + kHalf);
      if (!(zl > 0 && zrr > 0 && zu > 0 && zd > 0)) {continue;}
      auto P = [&](float uu, float vv, float z) {
          return std::array<float, 3>{(uu - K.cx) / K.fx * z, (vv - K.cy) / K.fy * z, z};
        };
      const auto pl = P(u - kHalf, v, zl), pr = P(u + kHalf, v, zrr), pu = P(u, v - kHalf, zu), pd = P(u, v + kHalf, zd);
      const float ax = pr[0] - pl[0], ay = pr[1] - pl[1], az = pr[2] - pl[2];
      const float bx = pd[0] - pu[0], by = pd[1] - pu[1], bz = pd[2] - pu[2];
      const float nx = ay * bz - az * by, ny = az * bx - ax * bz, nz = ax * by - ay * bx;
      const float rx = (u - K.cx) / K.fx, ry = (v - K.cy) / K.fy, rz = 1.f;
      const float c = std::fabs(nx * rx + ny * ry + nz * rz) /
        std::sqrt((nx * nx + ny * ny + nz * nz) * (rx * rx + ry * ry + rz * rz) + 1e-20f);
      const int cb = bucket(kCosEdges, c);
      if (cb < 0) {continue;}
      a.tot[zb * a.nc + cb] += 1;
      a.hole[zb * a.nc + cb] += hole;
      // 공간 잔차
      if (!hole) {
        const float z0 = z_at(u, v), zm = z_at(u - 2, v), zp = z_at(u + 2, v);
        if (zm > 0 && zp > 0 && a.resid[zb].size() < 200000) {a.resid[zb].push_back(z0 - 0.5f * (zm + zp));}
      }
    }
  }
}

inline float robust_sigma(std::vector<float> v)
{
  if (v.size() < 30) {return NAN;}
  std::nth_element(v.begin(), v.begin() + v.size() / 2, v.end());
  const float med = v[v.size() / 2];
  for (float & x : v) {x = std::fabs(x - med);}
  std::nth_element(v.begin(), v.begin() + v.size() / 2, v.end());
  return 1.4826f * v[v.size() / 2];
}

// 세워 둔 구간(정지 녹화 전체, 또는 주행 녹화의 바퀴가 멈춘 구간)마다 픽셀별 시간 σ (그 구간 유효율 > 0.5, 10 장 이상) →
// 구간들을 모아 거리 구간마다 σ 중앙
struct Temporal
{
  std::vector<double> sum, sq, n;
  int frames = 0, segments = 0;
  std::vector<std::vector<float>> pooled = std::vector<std::vector<float>>(kZEdges.size() - 1);
  void add(const uint16_t * img, int npx)
  {
    if (sum.empty()) {sum.assign(npx, 0); sq.assign(npx, 0); n.assign(npx, 0);}
    ++frames;
    for (int i = 0; i < npx; ++i) {
      if (img[i]) {const double z = img[i] * 1e-3; sum[i] += z; sq[i] += z * z; n[i] += 1;}
    }
  }
  // 지금 구간을 닫는다 (min_frames 장 안 되면 버림)
  void flush(int min_frames)
  {
    if (frames >= min_frames) {
      ++segments;
      for (size_t i = 0; i < sum.size(); ++i) {
        if (n[i] < 0.5 * frames || n[i] < 10) {continue;}
        const double m = sum[i] / n[i], sd = std::sqrt(std::max(0.0, sq[i] / n[i] - m * m));
        const int b = bucket(kZEdges, static_cast<float>(m));
        if (b >= 0) {pooled[b].push_back(static_cast<float>(sd));}
      }
    }
    std::fill(sum.begin(), sum.end(), 0);
    std::fill(sq.begin(), sq.end(), 0);
    std::fill(n.begin(), n.end(), 0);
    frames = 0;
  }
  void summary(std::vector<float> & med, std::vector<int> & cnt) const
  {
    for (auto x : pooled) {
      cnt.push_back(static_cast<int>(x.size()));
      if (x.empty()) {med.push_back(NAN); continue;}
      std::nth_element(x.begin(), x.begin() + x.size() / 2, x.end());
      med.push_back(x[x.size() / 2]);
    }
  }
};

inline void write_yaml(FILE * f, const Acc & a, const Temporal * t, const std::string & source)
{
  std::fprintf(f, "# 깊이 오차 통계 (depth_stats.hpp). 출처: %s\n", source.c_str());
  std::fprintf(f, "frames: %.0f\nvalid: %.4f        # 전체 픽셀 유효율\nunknown: %.4f      # 격자 픽셀 중 둘레로도 거리를 모름\n", a.frames,
    a.valid / std::max(1.0, a.px), a.unknown / std::max(1.0, a.grid));
  std::fprintf(f, "z_edges: [");
  for (size_t i = 0; i < kZEdges.size(); ++i) {std::fprintf(f, "%s%.2f", i ? ", " : "", kZEdges[i]);}
  std::fprintf(f, "]\ncos_edges: [0.0, 0.5, 0.8, 1.0]\n# 구멍 비율 [z][cos] (모서리 아님), 표본 수\nhole:\n");
  for (int z = 0; z < a.nz; ++z) {
    std::fprintf(f, "  - [");
    for (int c = 0; c < a.nc; ++c) {
      const double t2 = a.tot[z * a.nc + c];
      std::fprintf(f, "%s%.4f", c ? ", " : "", t2 > 0 ? a.hole[z * a.nc + c] / t2 : NAN);
    }
    std::fprintf(f, "]\n");
  }
  std::fprintf(f, "hole_n:\n");
  for (int z = 0; z < a.nz; ++z) {
    std::fprintf(f, "  - [");
    for (int c = 0; c < a.nc; ++c) {std::fprintf(f, "%s%.0f", c ? ", " : "", a.tot[z * a.nc + c]);}
    std::fprintf(f, "]\n");
  }
  std::fprintf(f, "edge_hole: [");
  for (int z = 0; z < a.nz; ++z) {std::fprintf(f, "%s%.4f", z ? ", " : "", a.etot[z] > 0 ? a.ehole[z] / a.etot[z] : NAN);}
  std::fprintf(f, "]\nedge_n: [");
  for (int z = 0; z < a.nz; ++z) {std::fprintf(f, "%s%.0f", z ? ", " : "", a.etot[z]);}
  std::fprintf(f, "]\nspatial_sigma: [   # 거리 구간마다 공간 잔차 σ (m)\n  ");
  for (int z = 0; z < a.nz; ++z) {std::fprintf(f, "%s%.5f", z ? ", " : "", robust_sigma(a.resid[z]));}
  std::fprintf(f, "]\nspatial_n: [");
  for (int z = 0; z < a.nz; ++z) {std::fprintf(f, "%s%zu", z ? ", " : "", a.resid[z].size());}
  // 유효 값 분위 (최소 거리)
  double tot = 0;
  for (double x : a.zhist) {tot += x;}
  auto q = [&](double p) {
      double c = 0;
      for (size_t i = 0; i < a.zhist.size(); ++i) {c += a.zhist[i]; if (c >= p * tot) {return i * 0.01;}}
      return 4.0;
    };
  std::fprintf(f, "]\nvalid_z_quantiles: [%.2f, %.2f, %.2f, %.2f, %.2f]   # 유효 값 0.0001 · 0.001 · 0.01 · 0.5 · 0.99 분위 (m)\n", q(1e-4),
    q(1e-3), q(1e-2), q(0.5), q(0.99));
  std::fprintf(f, "floor_cos_edges: [");
  for (size_t i = 0; i < kFloorCos.size(); ++i) {std::fprintf(f, "%s%.2f", i ? ", " : "", std::min(1.f, kFloorCos[i]));}
  std::fprintf(f, "]\nfloor_hole: [");
  for (size_t i = 0; i + 1 < kFloorCos.size(); ++i) {std::fprintf(f, "%s%.4f", i ? ", " : "", a.ftot[i] > 0 ? a.fhole[i] / a.ftot[i] : NAN);}
  std::fprintf(f, "]   # 바닥 입사 cos 구간마다 구멍 비율\nfloor_n: [");
  for (size_t i = 0; i + 1 < kFloorCos.size(); ++i) {std::fprintf(f, "%s%.0f", i ? ", " : "", a.ftot[i]);}
  // 가장자리 띠: 유효율 2 % 아래인 열·행이 가장자리에서 이어진 만큼 → 유효 영역 [row0, row1) × [col0, col1)
  auto band = [&](const std::vector<double> & cnt, double per, int & lo, int & hi) {
      const int n = static_cast<int>(cnt.size());
      lo = 0; hi = n;
      while (lo < n && cnt[lo] < 0.02 * per) {++lo;}
      while (hi > lo && cnt[hi - 1] < 0.02 * per) {--hi;}
    };
  int c0 = 0, c1 = 0, r0 = 0, r1 = 0;
  if (!a.col_valid.empty()) {
    band(a.col_valid, a.frames * a.row_valid.size(), c0, c1);
    band(a.row_valid, a.frames * a.col_valid.size(), r0, r1);
  }
  std::fprintf(f, "]\nvalid_rect: [%d, %d, %d, %d]   # 늘 유효 2 %% 아래인 가장자리 띠를 뺀 [row0, row1, col0, col1)\n", r0, r1, c0, c1);
  // 바닥 면: 1/z = x·cos/h + sin/h 를 최소제곱, 5 % 넘게 벗어난 점(바닥 아닌 행)을 빼고 세 번 → 카메라 높이 h·숙인 각
  {
    std::vector<char> keep(a.plane_x.size(), 1);
    double A = 0, B = 0;
    size_t used = 0;
    for (int it = 0; it < 4; ++it) {
      double sx = 0, sy = 0, sxx = 0, sxy = 0, n = 0;
      for (size_t i = 0; i < keep.size(); ++i) {
        if (!keep[i]) {continue;}
        sx += a.plane_x[i]; sy += a.plane_y[i]; sxx += double(a.plane_x[i]) * a.plane_x[i]; sxy += double(a.plane_x[i]) * a.plane_y[i]; n += 1;
      }
      if (n < 10) {break;}
      A = (n * sxy - sx * sy) / (n * sxx - sx * sx);
      B = (sy - A * sx) / n;
      used = 0;
      for (size_t i = 0; i < keep.size(); ++i) {
        keep[i] = std::fabs(a.plane_y[i] - (A * a.plane_x[i] + B)) < 0.05 * a.plane_y[i];
        used += keep[i];
      }
    }
    const double th = std::atan2(B, A), h = std::cos(th) / A;
    std::fprintf(f, "floor_plane: [%.4f, %.4f]   # 바닥 면으로 잰 카메라 광축 높이 (m)·아래로 숙인 각 (rad), 행 표본 %zu / %zu\n", h, th, used,
      a.plane_x.size());
  }
  if (t) {
    std::vector<float> med;
    std::vector<int> cnt;
    t->summary(med, cnt);
    std::fprintf(f, "temporal_sigma: [");
    for (size_t i = 0; i < med.size(); ++i) {std::fprintf(f, "%s%.5f", i ? ", " : "", med[i]);}
    std::fprintf(f, "]   # 거리 구간마다 픽셀별 시간 σ 중앙 (세워 둔 구간 %d 개)\ntemporal_n: [", t->segments);
    for (size_t i = 0; i < cnt.size(); ++i) {std::fprintf(f, "%s%d", i ? ", " : "", cnt[i]);}
    std::fprintf(f, "]\n");
  }
}

}  // namespace depth_stats
