// 접근·좌우 조준 계산, costmap 후보 검사, 물체 지도에서 불 고르기 (ROS 없음). 예전 Python 시험을 옮김 — 상하·탄도 시험은 뺌(발사대 15° 고정)
#include <gtest/gtest.h>

#include <cmath>
#include <vector>

#include "limo_fire_hardware/aim_geometry.hpp"

using namespace limo_fire_hardware;

namespace {
const std::vector<double> kOffsets = {0, 15 * M_PI / 180, -15 * M_PI / 180, 30 * M_PI / 180, -30 * M_PI / 180,
                                      60 * M_PI / 180, -60 * M_PI / 180};
double r(double d) { return d * M_PI / 180.0; }
}  // namespace

TEST(AimGeometry, CenteredTarget) {
  const auto c = approach_candidates({2.0, 0.0, 0.5}, {}, 1.0, {}, 0.0, kOffsets);
  ASSERT_EQ(c.size(), 7u);
  EXPECT_NEAR(c[0].base_x, 1.0, 1e-9);
  EXPECT_NEAR(c[0].base_y, 0.0, 1e-9);
  EXPECT_NEAR(c[0].base_yaw, 0.0, 1e-9);
}

TEST(AimGeometry, LateralMountAndAxisYaw) {
  const Vec3 target{2.7, -0.4, 0.85}, mount{0.3, -0.18, 0.25};
  const Pose current{0.2, 0.5, 0.1, 0.4};
  const double beta = r(23.0), dist = 1.4;
  const auto cs = approach_candidates(target, current, dist, mount, beta, kOffsets);
  ASSERT_EQ(cs.size(), 7u);
  for (const auto& c : cs) {
    const Pose p{c.base_x, c.base_y, current.z, c.base_yaw};
    EXPECT_NEAR(heading_error(target, p, mount, beta), 0.0, 1e-9);
    EXPECT_NEAR(std::hypot(target.x - c.pivot_x, target.y - c.pivot_y), dist, 1e-9);
    EXPECT_NEAR(pivot_distance(target, p, mount), dist, 1e-9);
    EXPECT_NEAR(c.pivot_z, 0.35, 1e-9);
  }
}

TEST(AimGeometry, AlreadyAimedPoseIsKept) {
  const Pose cur{0.7, -0.2, 0.1, 0.6};
  const Vec3 m{0.25, 0.12, 0.3};
  const double beta = -0.15, dist = 1.1;
  const double px = cur.x + std::cos(cur.yaw) * m.x - std::sin(cur.yaw) * m.y;
  const double py = cur.y + std::sin(cur.yaw) * m.x + std::cos(cur.yaw) * m.y;
  const Vec3 t{px + dist * std::cos(cur.yaw + beta), py + dist * std::sin(cur.yaw + beta), cur.z + m.z + 0.2};
  const auto c = approach_candidates(t, cur, dist, m, beta, kOffsets)[0];
  EXPECT_NEAR(c.base_x, cur.x, 1e-9);
  EXPECT_NEAR(c.base_y, cur.y, 1e-9);
  EXPECT_NEAR(c.base_yaw, cur.yaw, 1e-9);
}

TEST(AimGeometry, HeadingWrapAndSign) {
  EXPECT_NEAR(heading_error({0, 1, 0}, {}), M_PI / 2, 1e-12);
  EXPECT_NEAR(heading_error({1, 0, 0}, {0, 0, 0, 2 * M_PI}), 0.0, 1e-12);
  EXPECT_NEAR(heading_error({1, -1, 0}, {}), -M_PI / 4, 1e-12);
}

TEST(AimGeometry, BoundedYawRateAndDeadband) {
  auto cmd = [](double e) { return yaw_rate_command(r(e), r(3), 1.2, 0.05, 0.25); };
  EXPECT_EQ(cmd(2), 0.0);
  EXPECT_EQ(cmd(-2), 0.0);
  EXPECT_NEAR(cmd(30), 0.25, 1e-12);
  EXPECT_NEAR(cmd(-30), -0.25, 1e-12);
  EXPECT_GT(cmd(4), 0.05);
  EXPECT_LT(cmd(4), 0.25);
  EXPECT_THROW(yaw_rate_command(std::nan(""), 0.05, 1.2, 0.05, 0.25), std::invalid_argument);
  EXPECT_THROW(yaw_rate_command(0.1, 0.05, 1.2, 0.3, 0.25), std::invalid_argument);
}

TEST(AimGeometry, BodyYawAccountsForLateralOffset) {
  const Vec3 t{2.0, 0.0, 0.5}, m{0.3, 0.2, 0.25};
  const Pose b{0, 0, 0, 0.4};
  const double yaw = body_yaw_for_target(t, b, m);
  EXPECT_NEAR(yaw, -std::asin(0.2 / 2.0), 1e-12);
  EXPECT_NEAR(heading_error(t, {b.x, b.y, b.z, yaw}, m), 0.0, 1e-9);
}

TEST(AimGeometry, BodyYawAccountsForMountYaw) {
  const Vec3 t{1.5, -0.7, 0.6}, m{0.23, -0.12, 0.25};
  const Pose b{-0.2, 0.4, 0.1, -0.4};
  const double yaw = body_yaw_for_target(t, b, m, 0.3);
  EXPECT_NEAR(heading_error(t, {b.x, b.y, b.z, yaw}, m, 0.3), 0.0, 1e-9);
}

TEST(AimGeometry, BodyYawRejectsUnreachable) {
  EXPECT_THROW(body_yaw_for_target({0.1, 0, 0.3}, {}, {0, 0.2, 0}), std::invalid_argument);
  EXPECT_THROW(body_yaw_for_target({0.1, 0, 0.3}, {}, {0.2, 0, 0}), std::invalid_argument);
  EXPECT_THROW(body_yaw_for_target({0, 0, 0.3}, {}), std::invalid_argument);
}

TEST(AimGeometry, CoincidentPivotUsesCurrentAxis) {
  const auto c = approach_candidates({0, 0, 0.2}, {0, 0, 0, 0.3}, 1, {}, 0.0, kOffsets);
  EXPECT_NEAR(c[0].base_yaw, 0.3, 1e-12);
}

TEST(AimGeometry, InvalidGeometry) {
  for (double d : std::vector<double>{0.0, -1.0, std::nan(""), INFINITY})
    EXPECT_THROW(approach_candidates({1, 0, 0.2}, {}, d, {}, 0.0, kOffsets), std::invalid_argument) << d;
  EXPECT_THROW(heading_error({std::nan(""), 0, 1}, {}), std::invalid_argument);
  EXPECT_THROW(heading_error({1, INFINITY, 1}, {}), std::invalid_argument);
  EXPECT_THROW(heading_error({0, 0, 1}, {}), std::invalid_argument);
  EXPECT_THROW(heading_error({1, 0, 0.2}, {0, 0, 0, std::nan("")}), std::invalid_argument);
  EXPECT_THROW(heading_error({1, 0, 0.2}, {}, {}, INFINITY), std::invalid_argument);
  EXPECT_THROW(heading_error({1, 0, 0.2}, {}, {0, std::nan(""), 0}), std::invalid_argument);
}

TEST(AimGeometry, InvalidOffsets) {
  for (const auto& o : std::vector<std::vector<double>>{{}, {0, 0}, {0, 2 * M_PI}, {0, std::nan("")}})
    EXPECT_THROW(approach_candidates({1, 0, 0.2}, {}, 1, {}, 0.0, o), std::invalid_argument);
}

// ---- costmap 검사
namespace {
struct Grid {
  std::vector<int8_t> data = std::vector<int8_t>(40 * 40, 0);
  GridView view(double ox = 0, double oy = 0, double yaw = 0) {
    GridView g;
    g.origin_x = ox;
    g.origin_y = oy;
    g.origin_yaw = yaw;
    g.resolution = 0.1;
    g.width = g.height = 40;
    g.data = data.data();
    g.size = data.size();
    return g;
  }
  void set(int col, int row, int8_t v) { data[row * 40 + col] = v; }
};
const Candidate kCand{0.85, 0.85, 0.0, 0.85, 0.85, 0.2, 2.0};
const Vec3 kTarget{2.85, 0.85, 0.4};
}  // namespace

TEST(NavigationChecks, FreeFootprintAndSight) {
  Grid g;
  EXPECT_TRUE(candidate_clear(g.view(), kCand, kTarget, 0.2, 65, 0.15));
}
TEST(NavigationChecks, UnknownOrOccupiedFootprintRejected) {
  Grid g;
  g.set(9, 9, -1);
  EXPECT_FALSE(candidate_clear(g.view(), kCand, kTarget, 0.2, 65, 0.15));
  Grid h;
  h.set(8, 9, 100);
  EXPECT_FALSE(candidate_clear(h.view(), kCand, kTarget, 0.2, 65, 0.15));
}
TEST(NavigationChecks, OutOfBoundsFootprintRejected) {
  Grid g;
  EXPECT_FALSE(candidate_clear(g.view(), {0.1, 0.85, 0, 0.1, 0.85, 0.2, 2.75}, kTarget, 0.2, 65, 0.15));
}
TEST(NavigationChecks, SightLineAndTargetMargin) {
  Grid g;
  g.set(18, 8, -1);
  EXPECT_FALSE(candidate_clear(g.view(), kCand, kTarget, 0.2, 65, 0.15));
  Grid h;
  h.set(24, 8, 100);
  EXPECT_FALSE(candidate_clear(h.view(), kCand, kTarget, 0.2, 65, 0.25));
  Grid k;
  k.set(28, 8, 100);   // 표적 자체(끝 여유 안)는 허용
  EXPECT_TRUE(candidate_clear(k.view(), kCand, kTarget, 0.2, 65, 0.25));
  EXPECT_FALSE(candidate_clear(k.view(), kCand, kTarget, 0.2, 65, 0.0));
}
TEST(NavigationChecks, CostThresholdIsInclusive) {
  Grid g;
  g.set(18, 8, 64);
  EXPECT_TRUE(candidate_clear(g.view(), kCand, kTarget, 0.2, 65, 0.15));
  g.set(18, 8, 65);
  EXPECT_FALSE(candidate_clear(g.view(), kCand, kTarget, 0.2, 65, 0.15));
}
TEST(NavigationChecks, RotatedOrigin) {
  Grid g;
  g.set(2, 3, 37);
  const auto v = g.view(5.0, -1.0, M_PI / 2);
  ASSERT_TRUE(grid_cell(v, 4.65, -0.75));
  EXPECT_EQ(*grid_cell(v, 4.65, -0.75), 37);
  EXPECT_FALSE(grid_cell(v, 5.2, -1.0));
  Grid h;
  const Candidate c{4.15, -0.15, M_PI / 2, 4.15, -0.15, 0.2, 2.0};
  EXPECT_TRUE(candidate_clear(h.view(5.0, -1.0, M_PI / 2), c, {4.15, 1.85, 0.4}, 0.2, 65, 0.15));
  h.set(18, 8, 100);
  EXPECT_FALSE(candidate_clear(h.view(5.0, -1.0, M_PI / 2), c, {4.15, 1.85, 0.4}, 0.2, 65, 0.15));
}
TEST(NavigationChecks, InvalidResolutionAndTruncatedData) {
  Grid g;
  auto v = g.view();
  v.resolution = 0.0;
  EXPECT_FALSE(candidate_clear(v, kCand, kTarget, 0.2, 65, 0.15));
  v = g.view();
  v.size = 0;
  EXPECT_FALSE(candidate_clear(v, kCand, kTarget, 0.2, 65, 0.15));
}

// ---- 물체 지도에서 방금 본 불 하나
namespace {
MapObject fire(const std::string& id = "fire-1", double score = 0.9, double age = 0.1) {
  MapObject o;
  o.id = id;
  o.class_id = "fire";
  o.score = score;
  o.seen_age_s = age;
  o.position = {1.0, 2.0, 0.3};
  return o;
}
}  // namespace

TEST(FireSelection, StaleRepublishedLandmarkIsIgnored) {
  const auto p = select_live_fire({fire("fire-1", 0.9, 2.0)}, "", 0.5, 0.5);
  EXPECT_EQ(p.index, -1);
  EXPECT_NE(p.reason.find("최근"), std::string::npos);
}
TEST(FireSelection, TwoFiresNeedExplicitTarget) {
  const std::vector<MapObject> pair = {fire("fire-1"), fire("fire-2")};
  EXPECT_EQ(select_live_fire(pair, "", 0.5, 0.5).index, -1);
  EXPECT_EQ(select_live_fire(pair, "fire-2", 0.5, 0.5).index, 1);
}
TEST(FireSelection, FreshFireIsSelected) {
  const auto p = select_live_fire({fire()}, "", 0.5, 0.5);
  EXPECT_EQ(p.index, 0);
  EXPECT_DOUBLE_EQ(p.age_s, 0.1);
  EXPECT_TRUE(p.reason.empty());
}
TEST(FireSelection, LowConfidenceAndOtherClassesIgnored) {
  auto person = fire();
  person.class_id = "person";
  EXPECT_EQ(select_live_fire({fire("fire-1", 0.3), person}, "", 0.5, 0.5).index, -1);
}
