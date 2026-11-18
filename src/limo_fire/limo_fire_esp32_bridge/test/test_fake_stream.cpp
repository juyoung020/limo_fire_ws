// 가짜 ESP32 줄 만들기: 다리가 쓰는 해석기(parse_line)를 거쳐 장면마다 기대한 대로 나오는지
#include <gtest/gtest.h>

#include <algorithm>

#include "limo_fire_esp32_bridge/fake_stream.hpp"
#include "limo_fire_esp32_bridge/protocol.hpp"

using namespace limo_fire_esp32_bridge;

namespace {
struct Count { int sonar = 0, hb = 0, bad = 0; uint32_t min_mm = 1u << 30, max_mm = 0; };
Count run(Scenario sc, int steps, const FakeParams& p = {}) {
  Count c;
  for (int s = 0; s < steps; ++s)
    for (const auto& l : lines_at(sc, s, p)) {
      const auto parsed = parse_line(l);
      if (!parsed) { ++c.bad; continue; }
      if (const auto u = as_sonar(*parsed)) {
        ++c.sonar;
        c.min_mm = std::min(c.min_mm, u->mm);
        c.max_mm = std::max(c.max_mm, u->mm);
      } else if (parsed->type == "HB") ++c.hb;
    }
  return c;
}
}  // namespace

TEST(FakeStream, SweepStaysInRangeAndAllLinesParse) {
  const auto c = run(Scenario::Sweep, 600);   // 21 s
  EXPECT_EQ(c.bad, 0);
  EXPECT_EQ(c.sonar, 600);
  EXPECT_GE(c.min_mm, 250u);
  EXPECT_LE(c.max_mm, 3700u);
  EXPECT_GE(c.hb, 20);   // 1 s 마다
}

TEST(FakeStream, NoEchoIsAllZero) {
  const auto c = run(Scenario::NoEcho, 100);
  EXPECT_EQ(c.sonar, 100);
  EXPECT_EQ(c.max_mm, 0u);
}

TEST(FakeStream, CorruptLinesAreRejectedByParser) {
  const auto c = run(Scenario::Corrupt, 90);
  EXPECT_EQ(c.sonar, 90);   // 정상 줄은 전부 살아남고
  EXPECT_EQ(c.bad, 30 * 3); // 3 걸음마다 나쁜 줄 3 개(체크섬 틀림·반쪽·쓰레기)가 전부 버려짐
}

TEST(FakeStream, StallSendsNothingInWindow) {
  FakeParams p;
  p.stall_after = 1.0;
  p.stall_len = 2.0;
  EXPECT_FALSE(lines_at(Scenario::Stall, 10, p).empty());   // 0.35 s
  EXPECT_TRUE(lines_at(Scenario::Stall, 50, p).empty());    // 1.75 s
  EXPECT_FALSE(lines_at(Scenario::Stall, 100, p).empty());  // 3.5 s
}

TEST(FakeStream, ParseScenario) {
  Scenario s;
  EXPECT_TRUE(parse_scenario("corrupt", s));
  EXPECT_FALSE(parse_scenario("nope", s));
}

TEST(FakeStream, SensorsSceneThermalAndRadarParse) {
  int th = 0, rd = 0, bad = 0;
  float max_c = 0;
  size_t max_targets = 0;
  for (int s = 0; s < 300; ++s)   // 10.5 s
    for (const auto& l : lines_at(Scenario::Sensors, s)) {
      const auto parsed = parse_line(l);
      if (!parsed) { ++bad; continue; }
      if (const auto t = as_thermal(*parsed)) {
        ++th;
        ASSERT_EQ(t->c.size(), static_cast<size_t>(kThermalPoints));
        max_c = std::max(max_c, *std::max_element(t->c.begin(), t->c.end()));
        EXPECT_NEAR(t->ambient_c, 22.0, 1e-6);
      } else if (const auto r = as_radar(*parsed)) {
        ++rd;
        max_targets = std::max(max_targets, r->targets.size());
      }
    }
  EXPECT_EQ(bad, 0);
  EXPECT_NEAR(th, 10.5 * 4, 3);    // 약 4 Hz
  EXPECT_NEAR(rd, 10.5 * 10, 5);   // 약 10 Hz
  EXPECT_NEAR(max_c, 80.0, 1e-4);
  EXPECT_EQ(max_targets, 2u);      // 셋째 칸은 비어 있어 뺀다
}
