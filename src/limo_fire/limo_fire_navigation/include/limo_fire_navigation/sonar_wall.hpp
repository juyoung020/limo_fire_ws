// 초음파 메아리로 저장 지도의 벽에 맞춰 연기 속 위치를 조금씩 고친다 — 연기 모드 smoke_localization: wall.
// ROS 의존 없음(테스트용). 설계: limo_fire_navigation/README.md "연기 모드" · docs/smoke_mode_plan.md.
//
// 왜 AMCL(sonar)이 아닌가: 초음파 2 개면 한 번에 점이 몇 개뿐이라, AMCL 이 지도 전체에서 비슷한 복도의 엉뚱한 자리를 고른다
// (benchmarks 결과 5·6). 여기서는 "지금 추정 근처"만 본다.
//
// 방법 (점-면 맞추기, 가우스-뉴턴 몇 번)
//   최근 몇 초 메아리를 각각 그 시각 odom 자세와 함께 모은다(바퀴 + IMU 는 몇 초 동안은 거의 맞으므로 서로 단단히 묶여 있다).
//   메아리마다: 지금 추정 자세에서 원뿔 폭 광선을 지도에 쏴 "보여야 할 거리" r_exp 를 구한다(가장 가까운 광선 = 반사한 곳).
//     |잰 거리 − r_exp| > gate 이면 버린다 — 지도에 없는 물체·다른 벽.
//     끝점(가장 가까운 광선 방향 × 잰 거리)이 벽 면에서 떨어진 부호 거리 e 와 벽 면 법선 n 을 거리 지도에서 읽는다.
//   고침 δ = (dx, dy, dyaw) 는 Σ w·(e + n·J·δ)² + 사전(δ²/σ²) 를 최소로. 한 벽만 보이면 그 벽에 수직인 방향만 정보가 있어
//   → 복도를 따라가는 방향은 사전 때문에 그대로 둔다(복도 착각이 구조적으로 안 생김).
//   잔차 큰 메아리는 Huber 무게로 덜 믿는다.
//   사전 σ 는 부르는 쪽이 준다(칼만 꼴): 지금 추정의 불확실성 — 움직인 만큼 커지고 맞출 때마다 줄어든다(결과의 var_*).
//   이렇게 해야 지도와 조금 다른 물체가 같은 쪽으로 계속 당겨도 추정이 끝없이 밀려가지 않는다.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

#include "limo_fire_navigation/map_raycast.hpp"

namespace limo_fire_navigation {

// 벽 면까지 부호 거리 지도 (m, ±cap 에서 자름). 빈 칸은 +(가장 가까운 벽 칸까지 − 반 칸), 벽 칸은 −(가장 가까운 빈 칸까지 − 반 칸).
// 칸 가운데 값을 겹선형 보간 → 벽 면(칸 경계)에서 0 을 지난다. 거리는 8 방향 두 번 훑기(chamfer, 1 · √2).
struct DistField {
  int width = 0, height = 0;
  double resolution = 0.05, origin_x = 0.0, origin_y = 0.0, origin_yaw = 0.0, cap = 1.0;
  std::vector<float> d;

  double cell(int ix, int iy) const {
    ix = std::clamp(ix, 0, width - 1);   // 지도 밖은 가장자리 값 그대로 (기울기 0)
    iy = std::clamp(iy, 0, height - 1);
    return d[static_cast<size_t>(iy) * width + ix];
  }
  // 지도 좌표 → 격자 좌표 (칸 가운데 = 정수)
  void to_grid(double x, double y, double& u, double& v) const {
    const double c = std::cos(-origin_yaw), s = std::sin(-origin_yaw);
    const double dx = x - origin_x, dy = y - origin_y;
    u = (c * dx - s * dy) / resolution - 0.5;
    v = (s * dx + c * dy) / resolution - 0.5;
  }
  // 부호 거리와 지도 좌표 기울기(∂d/∂x, ∂d/∂y) — 벽 면 법선(빈 쪽을 가리킴)
  double at(double x, double y, double* gx = nullptr, double* gy = nullptr) const {
    double u, v;
    to_grid(x, y, u, v);
    const int ix = static_cast<int>(std::floor(u)), iy = static_cast<int>(std::floor(v));
    const double fu = u - ix, fv = v - iy;
    const double a = cell(ix, iy), b = cell(ix + 1, iy), c = cell(ix, iy + 1), e = cell(ix + 1, iy + 1);
    if (gx && gy) {
      const double du = ((1 - fv) * (b - a) + fv * (e - c)) / resolution;   // 격자 축 기울기
      const double dv = ((1 - fu) * (c - a) + fu * (e - b)) / resolution;
      const double co = std::cos(origin_yaw), si = std::sin(origin_yaw);   // 격자 축 → 지도 축
      *gx = co * du - si * dv;
      *gy = si * du + co * dv;
    }
    return (1 - fv) * ((1 - fu) * a + fu * b) + fv * ((1 - fu) * c + fu * e);
  }
};

namespace detail {
inline std::vector<float> chamfer(const std::vector<char>& seed, int w, int h, double res) {
  const float big = std::numeric_limits<float>::max() / 4;
  std::vector<float> d(seed.size(), big);
  for (size_t i = 0; i < d.size(); ++i)
    if (seed[i]) d[i] = 0.0f;
  const float a = static_cast<float>(res), b = static_cast<float>(res * std::sqrt(2.0));
  auto relax = [&](int x, int y, int nx, int ny, float wgt) {
    if (nx < 0 || ny < 0 || nx >= w || ny >= h) return;
    float& v = d[static_cast<size_t>(y) * w + x];
    v = std::min(v, d[static_cast<size_t>(ny) * w + nx] + wgt);
  };
  for (int y = 0; y < h; ++y)
    for (int x = 0; x < w; ++x) {
      relax(x, y, x - 1, y, a);
      relax(x, y, x, y - 1, a);
      relax(x, y, x - 1, y - 1, b);
      relax(x, y, x + 1, y - 1, b);
    }
  for (int y = h - 1; y >= 0; --y)
    for (int x = w - 1; x >= 0; --x) {
      relax(x, y, x + 1, y, a);
      relax(x, y, x, y + 1, a);
      relax(x, y, x + 1, y + 1, b);
      relax(x, y, x - 1, y + 1, b);
    }
  return d;
}
}  // namespace detail

inline DistField make_dist_field(const Grid& g, double cap) {
  DistField f;
  f.width = g.width;
  f.height = g.height;
  f.resolution = g.resolution;
  f.origin_x = g.origin_x;
  f.origin_y = g.origin_y;
  f.origin_yaw = g.origin_yaw;
  f.cap = cap;
  const size_t n = static_cast<size_t>(g.width) * g.height;
  std::vector<char> wall(n), free_(n);
  for (size_t i = 0; i < n; ++i) {
    wall[i] = g.data[i] >= g.occupied_threshold;
    free_[i] = !wall[i];
  }
  const auto to_wall = detail::chamfer(wall, g.width, g.height, g.resolution);
  const auto to_free = detail::chamfer(free_, g.width, g.height, g.resolution);
  f.d.resize(n);
  const double half = 0.5 * g.resolution;
  for (size_t i = 0; i < n; ++i) {
    const double v = std::min((wall[i] ? to_free[i] : to_wall[i]) - half, cap);
    f.d[i] = static_cast<float>(wall[i] ? -v : v);
  }
  return f;
}

// 메아리 하나: 그 시각 센서 자세(map 좌표 — 지금 map → odom 으로 옮긴 값)와 잰 거리
struct SonarHit {
  double x, y, yaw;   // 센서 자리·방향 (map)
  double r, fov;      // 잰 거리 (m) · 원뿔 폭 (rad)
};

struct WallParams {
  int rays = 7;                 // 원뿔 폭에 편 광선 수 (보여야 할 거리 = 가장 가까운 광선)
  double max_range = 4.0;       // 광선 최대 (m)
  double gate = 0.25;           // |잰 거리 − 보여야 할 거리| 이보다 크면 버림 (m)
  double sigma = 0.05;          // 끝점 ↔ 벽 면 거리 잡음 (m)
  double huber = 0.08;          // 이보다 큰 잔차는 무게를 줄임 (m)
  double min_normal = 0.5;      // 거리 지도 기울기 크기가 이보다 작으면(모서리·벽 속) 버림
  double prior_xy = 0.05;       // 사전 σ (부르는 쪽이 지금 추정 불확실성으로 줌) — 정보 없는 방향은 그대로 (m)
  double prior_yaw = 0.03;      // (rad)
  double max_step_xy = 0.10;    // 한 번에 고치는 최대 (m)
  double max_step_yaw = 0.05;   // (rad)
  int iterations = 3;           // 가우스-뉴턴 횟수
};

struct WallResult {
  double dx = 0.0, dy = 0.0, dyaw = 0.0;   // map 에서 (cx, cy) 를 중심으로 한 고침
  int used = 0;                            // 맞추기에 쓴 메아리 수 (마지막 반복)
  double rms0 = 0.0, rms = 0.0;            // 고치기 전·후 쓴 메아리의 잔차 제곱 평균 제곱근 (m)
  double info_min = 0.0;                   // 자료 정보 행렬(xy) 의 작은 고윳값 — 0 에 가까우면 한 방향(한 벽)만 봄
  double var_x = 0.0, var_y = 0.0, var_yaw = 0.0;   // 고친 뒤 불확실성 (정규 행렬 역의 대각, m² · rad²). 메아리를 못 쓰면 사전 그대로
};

namespace detail {
struct Row {
  double e, jx, jy, jt;   // 잔차 · 야코비안 (dx, dy, dyaw)
};

// 후보 고침 (dx, dy, dyaw, 중심 c) 에서 쓸 수 있는 메아리의 잔차·야코비안
inline std::vector<Row> rows(const DistField& f, const Grid& g, const std::vector<SonarHit>& hits, const WallParams& p,
                             double cx, double cy, double dx, double dy, double dyaw) {
  std::vector<Row> out;
  out.reserve(hits.size());
  const double cs = std::cos(dyaw), sn = std::sin(dyaw);
  for (const auto& h : hits) {
    const double rx = h.x - cx, ry = h.y - cy;
    const double sx = cx + cs * rx - sn * ry + dx, sy = cy + sn * rx + cs * ry + dy, syaw = h.yaw + dyaw;
    // 보여야 할 거리: 원뿔 폭 광선 중 가장 가까운 벽
    double r_exp = std::numeric_limits<double>::infinity(), th_best = syaw;
    for (int k = 0; k < p.rays; ++k) {
      const double th = syaw + (p.rays == 1 ? 0.0 : -0.5 * h.fov + h.fov * k / (p.rays - 1));
      const double r = cast(g, sx, sy, th, 0.0, p.max_range, std::numeric_limits<double>::infinity());
      if (r < r_exp) {
        r_exp = r;
        th_best = th;
      }
    }
    if (!std::isfinite(r_exp) || std::fabs(h.r - r_exp) > p.gate) continue;
    const double px = sx + h.r * std::cos(th_best), py = sy + h.r * std::sin(th_best);
    double gx = 0.0, gy = 0.0;
    const double e = f.at(px, py, &gx, &gy);
    const double gn = std::hypot(gx, gy);
    if (gn < p.min_normal) continue;
    const double nx = gx / gn, ny = gy / gn;
    // 끝점 p(δ) = R(dyaw)(s − c) + c + (dx, dy) + r·u(θ + dyaw) → ∂p/∂dyaw = (p − c) 를 90° 돌린 것
    out.push_back({e, nx, ny, nx * (-(py - cy)) + ny * (px - cx)});
  }
  return out;
}

inline double huber_w(double e, double k) { return std::fabs(e) <= k ? 1.0 : k / std::fabs(e); }

inline double rms(const std::vector<Row>& r) {
  if (r.empty()) return 0.0;
  double s = 0.0;
  for (const auto& x : r) s += x.e * x.e;
  return std::sqrt(s / r.size());
}

// 3 × 3 대칭 양정치 풀기 (가우스 소거)
inline bool solve3(double A[3][3], double b[3], double x[3]) {
  for (int i = 0; i < 3; ++i) {
    int piv = i;
    for (int k = i + 1; k < 3; ++k)
      if (std::fabs(A[k][i]) > std::fabs(A[piv][i])) piv = k;
    if (std::fabs(A[piv][i]) < 1e-12) return false;
    if (piv != i) {
      for (int j = 0; j < 3; ++j) std::swap(A[i][j], A[piv][j]);
      std::swap(b[i], b[piv]);
    }
    for (int k = i + 1; k < 3; ++k) {
      const double m = A[k][i] / A[i][i];
      for (int j = i; j < 3; ++j) A[k][j] -= m * A[i][j];
      b[k] -= m * b[i];
    }
  }
  for (int i = 2; i >= 0; --i) {
    double s = b[i];
    for (int j = i + 1; j < 3; ++j) s -= A[i][j] * x[j];
    x[i] = s / A[i][i];
  }
  return true;
}
}  // namespace detail

// hits 는 지금 map → odom 으로 옮긴 센서 자세. (cx, cy) = 로봇 지금 자리(map) — 회전 중심
inline WallResult match_walls(const DistField& f, const Grid& g, const std::vector<SonarHit>& hits, const WallParams& p,
                              double cx, double cy) {
  WallResult res;
  res.var_x = res.var_y = p.prior_xy * p.prior_xy;
  res.var_yaw = p.prior_yaw * p.prior_yaw;
  const auto r0 = detail::rows(f, g, hits, p, cx, cy, 0, 0, 0);
  res.rms0 = detail::rms(r0);
  double x[3] = {0, 0, 0};
  const double ws = 1.0 / (p.sigma * p.sigma);
  for (int it = 0; it < p.iterations; ++it) {
    const auto r = it == 0 ? r0 : detail::rows(f, g, hits, p, cx, cy, x[0], x[1], x[2]);
    if (r.empty()) break;
    // 사전은 고침 전체(x)에 건다: 정규 방정식 (JᵀWJ + P) Δ = −JᵀWe − P x
    double A[3][3] = {{1 / (p.prior_xy * p.prior_xy), 0, 0}, {0, 1 / (p.prior_xy * p.prior_xy), 0},
                      {0, 0, 1 / (p.prior_yaw * p.prior_yaw)}};
    double b[3] = {-A[0][0] * x[0], -A[1][1] * x[1], -A[2][2] * x[2]};
    double Ixx = 0, Ixy = 0, Iyy = 0;
    for (const auto& row : r) {
      const double w = ws * detail::huber_w(row.e, p.huber);
      const double J[3] = {row.jx, row.jy, row.jt};
      for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) A[i][j] += w * J[i] * J[j];
        b[i] -= w * J[i] * row.e;
      }
      Ixx += row.jx * row.jx;
      Ixy += row.jx * row.jy;
      Iyy += row.jy * row.jy;
    }
    // 고친 뒤 불확실성 = A⁻¹ 의 대각 (A 는 solve3 가 바꾸므로 먼저 단위 벡터로 푼다)
    double var[3];
    for (int k = 0; k < 3; ++k) {
      double Ak[3][3], e[3] = {0, 0, 0}, col[3];
      for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) Ak[i][j] = A[i][j];
      e[k] = 1.0;
      var[k] = detail::solve3(Ak, e, col) ? col[k] : 0.0;
    }
    double d[3];
    if (!detail::solve3(A, b, d)) break;
    res.var_x = var[0];
    res.var_y = var[1];
    res.var_yaw = var[2];
    for (int i = 0; i < 3; ++i) x[i] += d[i];
    res.used = static_cast<int>(r.size());
    const double tr = Ixx + Iyy, det = Ixx * Iyy - Ixy * Ixy;
    res.info_min = 0.5 * tr - std::sqrt(std::max(0.0, 0.25 * tr * tr - det));
  }
  // 한 번에 고치는 양 제한
  const double s = std::hypot(x[0], x[1]);
  if (s > p.max_step_xy) {
    x[0] *= p.max_step_xy / s;
    x[1] *= p.max_step_xy / s;
  }
  x[2] = std::clamp(x[2], -p.max_step_yaw, p.max_step_yaw);
  res.dx = x[0];
  res.dy = x[1];
  res.dyaw = x[2];
  res.rms = detail::rms(detail::rows(f, g, hits, p, cx, cy, x[0], x[1], x[2]));
  return res;
}

}  // namespace limo_fire_navigation
