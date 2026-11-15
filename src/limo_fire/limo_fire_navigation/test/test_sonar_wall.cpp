// 초음파 벽 맞추기: 벽에 수직인 방향만 고치고 복도를 따라가는 방향은 그대로, 지도에 없는 물체는 버림
#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <vector>

#include "limo_fire_navigation/sonar_wall.hpp"

using namespace limo_fire_navigation;

namespace {
constexpr double kRes = 0.05;

// 8 m × 3 m, 0.05 m/칸. 위·아래 벽(y < 0.5 m, y ≥ 2.5 m) = x 방향 복도 폭 2 m. with_end 면 x ≥ 7.5 m 막힘
Grid corridor(bool with_end) {
  Grid g;
  g.width = 160;
  g.height = 60;
  g.resolution = kRes;
  g.data.assign(static_cast<size_t>(g.width) * g.height, 0);
  for (int y = 0; y < g.height; ++y)
    for (int x = 0; x < g.width; ++x) {
      const double cy = (y + 0.5) * kRes, cx = (x + 0.5) * kRes;
      if (cy < 0.5 || cy >= 2.5 || (with_end && cx >= 7.5)) g.data[static_cast<size_t>(y) * g.width + x] = 100;
    }
  return g;
}

// 참 자세(x, y, yaw)에서 앞 좌우 ±angle 센서가 보는 거리(원뿔 가장 가까운 광선)를 자세 오차(ex, ey, eyaw)만큼 틀린 추정으로 옮긴 메아리
std::vector<SonarHit> make_hits(const Grid& g, const std::vector<std::array<double, 3>>& truth, double angle,
                                double ex, double ey, double eyaw) {
  std::vector<SonarHit> out;
  const double fov = 15.0 * M_PI / 180.0;
  for (const auto& p : truth) {
    for (double s : {angle, -angle}) {
      const double sx = p[0] + 0.18 * std::cos(p[2]), sy = p[1] + 0.18 * std::sin(p[2]), syaw = p[2] + s;
      double r = 1e9;
      for (int k = 0; k < 7; ++k) r = std::min(r, cast(g, sx, sy, syaw - fov / 2 + fov * k / 6, 0.0, 4.0, 1e9));
      if (r > 4.0) continue;   // 메아리 없음
      // 추정 자세 = 참 + 오차 (센서 자리도 같이 틀림, 로봇 자리 둘레로 eyaw)
      const double bx = p[0] + ex, by = p[1] + ey, byaw = p[2] + eyaw;
      out.push_back({bx + 0.18 * std::cos(byaw), by + 0.18 * std::sin(byaw), byaw + s, r, fov});
    }
  }
  return out;
}
}  // namespace

TEST(SonarWall, DistFieldSignAndNormal) {
  const Grid g = corridor(false);
  const DistField f = make_dist_field(g, 1.0);
  double gx, gy;
  EXPECT_NEAR(f.at(4.0, 0.8, &gx, &gy), 0.3, 0.03);   // 아래 벽 면 y = 0.5 에서 0.3 m
  EXPECT_NEAR(gy, 1.0, 0.1);                           // 법선은 빈 쪽(+y)
  EXPECT_NEAR(gx, 0.0, 0.1);
  EXPECT_LT(f.at(4.0, 0.4), 0.0);                      // 벽 속은 음수
}

// 복도 가운데보다 0.15 m 위로 틀리고 앞으로 0.3 m 틀린 추정: 옆(y)은 고치고 앞뒤(x)는 그대로
TEST(SonarWall, CorrectsAcrossCorridorOnly) {
  const Grid g = corridor(false);
  const DistField f = make_dist_field(g, 1.0);
  std::vector<std::array<double, 3>> truth;
  // 복도를 따라 지그재그로 (좌우 벽을 번갈아 봄)
  for (int i = 0; i < 12; ++i) truth.push_back({2.0 + 0.1 * i, 1.5, (i % 2 ? 1 : -1) * 0.5});
  const auto hits = make_hits(g, truth, 20.0 * M_PI / 180.0, 0.3, 0.15, 0.0);
  ASSERT_GE(hits.size(), 8u);
  WallParams p;
  p.max_step_xy = 1.0;
  const auto r = match_walls(f, g, hits, p, 2.6 + 0.3, 1.5 + 0.15);
  EXPECT_GT(r.used, 8);
  EXPECT_NEAR(r.dy, -0.15, 0.04);   // 옆 오차를 거의 다 고침
  EXPECT_NEAR(r.dx, 0.0, 0.02);     // 복도 방향은 정보가 없어 그대로
  EXPECT_LT(r.rms, r.rms0);
  // 불확실성: 옆(y)은 줄고, 앞뒤(x)는 사전 그대로
  EXPECT_LT(r.var_y, 0.5 * p.prior_xy * p.prior_xy);
  EXPECT_NEAR(r.var_x, p.prior_xy * p.prior_xy, 0.1 * p.prior_xy * p.prior_xy);
}

// 막다른 벽을 보면 앞뒤도 고친다
TEST(SonarWall, CorrectsAlongWhenEndWallSeen) {
  const Grid g = corridor(true);
  const DistField f = make_dist_field(g, 1.0);
  std::vector<std::array<double, 3>> truth;
  for (int i = 0; i < 10; ++i) truth.push_back({5.5 + 0.05 * i, 1.5, 0.0});
  const auto hits = make_hits(g, truth, 20.0 * M_PI / 180.0, -0.12, 0.0, 0.0);
  WallParams p;
  p.max_step_xy = 1.0;
  const auto r = match_walls(f, g, hits, p, 5.95 - 0.12, 1.5);
  EXPECT_NEAR(r.dx, 0.12, 0.04);
}

// 방향 오차: 옆 벽 둘을 번갈아 보면 yaw 를 고친다
TEST(SonarWall, CorrectsYaw) {
  const Grid g = corridor(false);
  const DistField f = make_dist_field(g, 1.0);
  std::vector<std::array<double, 3>> truth;
  for (int i = 0; i < 16; ++i) truth.push_back({2.0 + 0.1 * i, 1.5, (i % 2 ? 1 : -1) * 0.6});
  const auto hits = make_hits(g, truth, 20.0 * M_PI / 180.0, 0.0, 0.0, 0.04);
  WallParams p;
  const auto r = match_walls(f, g, hits, p, 3.5, 1.5);
  EXPECT_LT(r.dyaw, -0.01);   // 부호가 맞는 쪽으로
  EXPECT_LT(r.rms, r.rms0);
}

// 지도에 없는 물체(가까운 메아리)는 gate 로 버려 고침을 끌지 않는다
TEST(SonarWall, IgnoresUnmappedObstacle) {
  const Grid g = corridor(false);
  const DistField f = make_dist_field(g, 1.0);
  std::vector<std::array<double, 3>> truth;
  for (int i = 0; i < 12; ++i) truth.push_back({2.0 + 0.1 * i, 1.5, (i % 2 ? 1 : -1) * 0.5});
  auto hits = make_hits(g, truth, 20.0 * M_PI / 180.0, 0.0, 0.0, 0.0);
  const size_t good = hits.size();
  for (int i = 0; i < 6; ++i) hits.push_back({3.0, 1.5, 0.0, 0.4, 0.26});   // 앞 0.4 m 상자 (지도엔 없음)
  WallParams p;
  const auto r = match_walls(f, g, hits, p, 3.0, 1.5);
  EXPECT_LE(r.used, static_cast<int>(good));
  EXPECT_NEAR(r.dx, 0.0, 0.02);
  EXPECT_NEAR(r.dy, 0.0, 0.02);
}

TEST(SonarWall, StepIsClamped) {
  const Grid g = corridor(false);
  const DistField f = make_dist_field(g, 1.0);
  std::vector<std::array<double, 3>> truth;
  for (int i = 0; i < 12; ++i) truth.push_back({2.0 + 0.1 * i, 1.5, (i % 2 ? 1 : -1) * 0.5});
  const auto hits = make_hits(g, truth, 20.0 * M_PI / 180.0, 0.0, 0.2, 0.0);
  WallParams p;
  p.max_step_xy = 0.05;
  const auto r = match_walls(f, g, hits, p, 2.6, 1.7);
  EXPECT_LE(std::hypot(r.dx, r.dy), 0.05 + 1e-9);
}
