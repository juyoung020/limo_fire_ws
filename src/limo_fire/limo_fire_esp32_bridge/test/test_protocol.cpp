// 시리얼 줄 해석: 체크섬·형식·초음파 필드
#include <gtest/gtest.h>

#include <cstdio>
#include <cstring>

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
