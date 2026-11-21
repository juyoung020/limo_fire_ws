// sonar_stats.hpp: 녹화 통계
#include <gtest/gtest.h>

#include <cmath>
#include <limits>
#include <vector>

#include "limo_fire_sim/sonar_stats.hpp"

TEST(SonarStats, Summary)
{
  const auto s = sonar_stats::summarize({1.0, 2.0, 3.0, 4.0, 5.0});
  EXPECT_EQ(s.n, 5u);
  EXPECT_DOUBLE_EQ(s.mean, 3.0);
  EXPECT_DOUBLE_EQ(s.p50, 3.0);
  EXPECT_NEAR(s.sd, std::sqrt(2.5), 1e-12);
  EXPECT_DOUBLE_EQ(s.min, 1.0);
  EXPECT_DOUBLE_EQ(s.max, 5.0);
  EXPECT_EQ(sonar_stats::summarize({}).n, 0u);
}

TEST(SonarStats, SegmentCountsNoEchoBothConventions)   // +inf(다리) · max_range(옛 시뮬) 둘 다 메아리 없음
{
  sonar_stats::Segment s;
  const double inf = std::numeric_limits<double>::infinity();
  s.add(0.0, 1.0, 0.02, 4.0);
  s.add(0.1, inf, 0.02, 4.0);
  s.add(0.2, 4.0, 0.02, 4.0);
  s.add(0.3, 1.1, 0.02, 4.0);
  EXPECT_EQ(s.msgs, 4u);
  EXPECT_EQ(s.echo.size(), 2u);
  EXPECT_DOUBLE_EQ(s.echo_ratio(), 0.5);
  ASSERT_EQ(s.gap.size(), 3u);
  EXPECT_NEAR(s.gap[0], 0.1, 1e-12);
  EXPECT_DOUBLE_EQ(s.t1 - s.t0, 0.3);
}

TEST(SonarStats, StillNeedsHold)
{
  sonar_stats::StillDetector d;   // hold 0.5 s
  EXPECT_FALSE(d.update(0.0, 0.0, 0.0));
  EXPECT_FALSE(d.update(0.4, 0.0, 0.0));
  EXPECT_TRUE(d.update(0.6, 0.0, 0.0));
  EXPECT_FALSE(d.update(0.7, 0.2, 0.0));   // 움직이면 바로 끝
  EXPECT_FALSE(d.update(1.0, 0.0, 0.0));
}
