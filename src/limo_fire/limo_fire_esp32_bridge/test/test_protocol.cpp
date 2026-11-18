// 시리얼 줄 해석: 체크섬·형식·초음파 필드
#include <gtest/gtest.h>

#include <cstdio>
#include <cstring>
#include <string>

#include "limo_fire_esp32_bridge/protocol.hpp"

using namespace limo_fire_esp32_bridge;

namespace {
// 펌웨어 frame_line 과 같은 방식으로 만든 줄
std::string frame(const char* body) {
  uint8_t c = 0;
  for (const char* p = body; *p; ++p) c ^= static_cast<uint8_t>(*p);
  char out[128];
  std::snprintf(out, sizeof out, "$%s*%02X\r\n", body, c);
  return out;
}
}  // namespace

TEST(Protocol, SonarLine) {
  const auto l = parse_line(frame("US,2,1234,5678"));
  ASSERT_TRUE(l);
  EXPECT_EQ(l->type, "US");
  const auto s = as_sonar(*l);
  ASSERT_TRUE(s);
  EXPECT_EQ(s->index, 2);
  EXPECT_EQ(s->mm, 1234u);
  EXPECT_EQ(s->ms, 5678u);
}

TEST(Protocol, NoEchoIsZero) {
  const auto s = as_sonar(*parse_line(frame("US,0,0,10")));
  ASSERT_TRUE(s);
  EXPECT_EQ(s->mm, 0u);
}

TEST(Protocol, BadChecksumDropped) {
  std::string f = frame("US,1,500,1");
  f[f.find('*') + 1] = f[f.find('*') + 1] == '0' ? '1' : '0';
  EXPECT_FALSE(parse_line(f));
}

TEST(Protocol, GarbageBeforeDollarIgnored) {
  EXPECT_TRUE(parse_line("\x01\x02junk" + frame("HB,42")));
}

TEST(Protocol, HalfLineDropped) {
  EXPECT_FALSE(parse_line("$US,1,50"));
  EXPECT_FALSE(parse_line("US,1,500,1*00"));
}

TEST(Protocol, BadSonarFields) {
  EXPECT_FALSE(as_sonar(*parse_line(frame("US,1,abc,1"))));
  EXPECT_FALSE(as_sonar(*parse_line(frame("US,1,500"))));
  EXPECT_FALSE(as_sonar(*parse_line(frame("US,-1,500,1"))));
  EXPECT_FALSE(as_sonar(*parse_line(frame("HB,1"))));
}

namespace {
std::string frame_long(const std::string& body) {
  uint8_t c = 0;
  for (char ch : body) c ^= static_cast<uint8_t>(ch);
  char tail[8];
  std::snprintf(tail, sizeof tail, "*%02X\r\n", c);
  return "$" + body + tail;
}
std::string thermal_body(int n_points, const std::string& first = "250") {
  std::string b = "TH,1234,215";
  for (int i = 0; i < n_points; ++i) b += "," + (i == 0 ? first : std::string("200"));
  return b;
}
}  // namespace

TEST(Protocol, ThermalLine) {
  const auto l = parse_line(frame_long(thermal_body(kThermalPoints)));
  ASSERT_TRUE(l);
  const auto t = as_thermal(*l);
  ASSERT_TRUE(t);
  EXPECT_EQ(t->ms, 1234u);
  EXPECT_NEAR(t->ambient_c, 21.5, 1e-9);
  ASSERT_EQ(t->c.size(), static_cast<size_t>(kThermalPoints));
  EXPECT_NEAR(t->c[0], 25.0, 1e-4);
  EXPECT_NEAR(t->c[767], 20.0, 1e-4);
}

TEST(Protocol, ThermalWrongLengthOrNumberRejected) {
  EXPECT_FALSE(as_thermal(*parse_line(frame_long(thermal_body(kThermalPoints - 1)))));
  EXPECT_FALSE(as_thermal(*parse_line(frame_long(thermal_body(kThermalPoints, "2x0")))));
}

TEST(Protocol, RadarLineSkipsEmptyTargets) {
  const auto r = as_radar(*parse_line(frame_long("RD,500,-782,1713,-12,0,0,0,300,2500,5")));
  ASSERT_TRUE(r);
  EXPECT_EQ(r->ms, 500u);
  ASSERT_EQ(r->targets.size(), 2u);
  EXPECT_EQ(r->targets[0].x_mm, -782);
  EXPECT_EQ(r->targets[0].y_mm, 1713);
  EXPECT_EQ(r->targets[0].v_cms, -12);
  EXPECT_EQ(r->targets[1].y_mm, 2500);
}

TEST(Protocol, RadarBadFieldCountRejected) {
  EXPECT_FALSE(as_radar(*parse_line(frame_long("RD,500,1,2,3"))));
  EXPECT_FALSE(as_radar(*parse_line(frame_long("RD,500,1,2,3,4,5,6,7,8,x"))));
}
