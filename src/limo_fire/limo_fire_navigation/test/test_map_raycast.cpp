// 가상 스캔 광선 추적: 벽까지 거리, 지도 밖·모름 통과, 지도 원점 회전
#include <gtest/gtest.h>

#include "limo_fire_navigation/map_raycast.hpp"

using limo_fire_navigation::cast;
using limo_fire_navigation::Grid;

namespace {
// 10 × 10 칸, 0.1 m/칸, x = 0.7 m 열(칸 7)이 벽
Grid wall_grid() {
  Grid g;
  g.width = 10;
  g.height = 10;
  g.resolution = 0.1;
  g.data.assign(100, 0);
  for (int y = 0; y < 10; ++y) g.data[y * 10 + 7] = 100;
  return g;
}
}  // namespace

TEST(MapRaycast, HitsWall) {
  const Grid g = wall_grid();
  const double r = cast(g, 0.25, 0.55, 0.0, 0.0, 3.0, 99.0);
  EXPECT_NEAR(r, 0.45, 0.5 * g.resolution + 1e-9);  // 벽 칸 모서리 0.7 − 0.25
}

TEST(MapRaycast, NoHitAwayFromWall) {
  const Grid g = wall_grid();
  EXPECT_EQ(cast(g, 0.25, 0.55, M_PI, 0.0, 3.0, 99.0), 99.0);  // 뒤로는 지도 밖으로 나감
}

TEST(MapRaycast, UnknownIsNotWall) {
  Grid g = wall_grid();
  for (int y = 0; y < 10; ++y) g.data[y * 10 + 7] = -1;
  EXPECT_EQ(cast(g, 0.25, 0.55, 0.0, 0.0, 3.0, 99.0), 99.0);
}

TEST(MapRaycast, RangeMaxCuts) {
  const Grid g = wall_grid();
  EXPECT_EQ(cast(g, 0.25, 0.55, 0.0, 0.0, 0.3, 99.0), 99.0);
}

TEST(MapRaycast, RotatedOrigin) {
  // 원점을 90° 돌리면 격자의 x 축이 지도의 y 축 → 벽은 지도 y = 0.7 에
  Grid g = wall_grid();
  g.origin_yaw = M_PI / 2;
  const double r = cast(g, -0.55, 0.25, M_PI / 2, 0.0, 3.0, 99.0);
  EXPECT_NEAR(r, 0.45, 0.5 * g.resolution + 1e-9);
}
