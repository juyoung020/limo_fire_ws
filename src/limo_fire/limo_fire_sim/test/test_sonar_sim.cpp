// sonar_sim.hpp: 시뮬 초음파 → 실물 다리 꼴
#include <gtest/gtest.h>

#include <cmath>
#include <random>

#include "limo_fire_sim/sonar_sim.hpp"

using sonar_sim::Effects;

TEST(SonarSim, PassThroughKeepsEcho)
{
  Effects e;
  std::mt19937 rng(1);
  EXPECT_DOUBLE_EQ(sonar_sim::apply(e, 1.234, 4.0, rng), 1.234);
}

TEST(SonarSim, WebotsNoEchoBecomesInf)   // Webots 4.0 m · 메시지 max 3.96 m 둘 다 → 다리처럼 +inf
{
  Effects e;
  std::mt19937 rng(1);
  EXPECT_TRUE(std::isinf(sonar_sim::apply(e, 4.0, 4.0, rng)));
  EXPECT_TRUE(std::isinf(sonar_sim::apply(e, 4.0, 3.96, rng)));
  EXPECT_TRUE(std::isinf(sonar_sim::apply(e, NAN, 4.0, rng)));
}

TEST(SonarSim, OutOfDatasheetRangeIsNoEcho)   // 펌웨어: 범위 밖 = 0 → +inf
{
  Effects e;
  e.bias_m = -0.05;
  std::mt19937 rng(1);
  EXPECT_TRUE(std::isinf(sonar_sim::apply(e, 0.04, 4.0, rng)));
}

TEST(SonarSim, DeadSensorNeverEchoes)
{
  Effects e;
  e.dead = true;
  std::mt19937 rng(1);
  for (int i = 0; i < 100; ++i) {EXPECT_TRUE(std::isinf(sonar_sim::apply(e, 1.0, 4.0, rng)));}
}

TEST(SonarSim, DropProbabilityRoughlyHolds)
{
  Effects e;
  e.drop_prob = 0.2;
  std::mt19937 rng(7);
  int lost = 0;
  const int n = 20000;
  for (int i = 0; i < n; ++i) {lost += std::isinf(sonar_sim::apply(e, 1.0, 4.0, rng)) ? 1 : 0;}
  EXPECT_NEAR(static_cast<double>(lost) / n, 0.2, 0.01);
}

TEST(SonarSim, BiasAndNoise)
{
  Effects e;
  e.bias_m = 0.03;
  e.noise_sigma_m = 0.01;
  std::mt19937 rng(3);
  double sum = 0, ss = 0;
  const int n = 20000;
  for (int i = 0; i < n; ++i) {
    const double r = sonar_sim::apply(e, 1.0, 4.0, rng);
    sum += r;
    ss += r * r;
  }
  const double mean = sum / n;
  EXPECT_NEAR(mean, 1.03, 0.001);
  EXPECT_NEAR(std::sqrt(ss / n - mean * mean), 0.01, 0.001);
}
