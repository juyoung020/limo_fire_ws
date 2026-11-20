// 시리얼 줄 해석: 체크섬·형식·초음파·열화상·레이더·발사대 필드
#include <gtest/gtest.h>

#include <cstdio>
#include <cstring>
#include <string>
#include <utility>
#include <vector>

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

// ---- 발사대 응답 $LA (펌웨어 include/launcher.h)
TEST(Protocol, LauncherSimpleReplies) {
  const std::pair<const char*, LauncherReply::Kind> cases[] = {
      {"LA,boot", LauncherReply::kBoot},     {"LA,on", LauncherReply::kOn},       {"LA,off", LauncherReply::kOff},
      {"LA,cocked", LauncherReply::kCocked}, {"LA,fired", LauncherReply::kFired}, {"LA,home", LauncherReply::kHome}};
  for (const auto& [body, kind] : cases) {
    const auto l = parse_line(frame(body));
    ASSERT_TRUE(l) << body;
    const auto r = as_launcher(*l);
    ASSERT_TRUE(r) << body;
    EXPECT_EQ(r->kind, kind) << body;
  }
}

TEST(Protocol, LauncherStateAndLever) {
  auto r = as_launcher(*parse_line(frame("LA,state,100.0,on,1")));
  ASSERT_TRUE(r);
  EXPECT_EQ(r->kind, LauncherReply::kState);
  EXPECT_DOUBLE_EQ(r->lever_deg, 100.0);
  EXPECT_TRUE(r->on);
  EXPECT_TRUE(r->busy);
  r = as_launcher(*parse_line(frame("LA,state,30.5,off,0")));
  ASSERT_TRUE(r);
  EXPECT_FALSE(r->on);
  EXPECT_FALSE(r->busy);
  r = as_launcher(*parse_line(frame("LA,lever,42.5")));
  ASSERT_TRUE(r);
  EXPECT_EQ(r->kind, LauncherReply::kLever);
  EXPECT_DOUBLE_EQ(r->lever_deg, 42.5);
}

TEST(Protocol, LauncherBlockedAndErrors) {
  auto r = as_launcher(*parse_line(frame("LA,blocked,off")));
  ASSERT_TRUE(r);
  EXPECT_EQ(r->kind, LauncherReply::kBlocked);
  EXPECT_EQ(r->detail, "off");
  for (const char* d : {"busy", "bad", "attach"}) {
    const std::string body = std::string("LA,error,") + d;
    r = as_launcher(*parse_line(frame(body.c_str())));
    ASSERT_TRUE(r);
    EXPECT_EQ(r->kind, LauncherReply::kError);
    EXPECT_EQ(r->detail, d);
  }
}

TEST(Protocol, LauncherMalformedRejected) {
  for (const char* body : {"LA", "LA,", "LA,nope", "LA,cocked,1", "LA,lever", "LA,lever,abc", "LA,lever,181", "LA,lever,-1",
                           "LA,state,100,on", "LA,state,100,maybe,0", "LA,state,100,on,2", "LA,state,x,on,0", "LA,error,",
                           "LA,blocked"}) {
    const auto l = parse_line(frame(body));
    if (!l) continue;   // 종류가 비면 줄 자체가 거부됨
    EXPECT_FALSE(as_launcher(*l)) << body;
  }
  // 다른 종류의 줄은 발사대 응답이 아니다
  EXPECT_FALSE(as_launcher(*parse_line(frame("US,0,100,5"))));
}

// 열화상 줄(≈ 3 KB)과 발사대 응답이 한 버퍼에 섞여 와도 둘 다 온전히 나온다 (다리의 줄 나누기와 같게)
TEST(Protocol, LongThermalLineThenLauncherReply) {
  std::string th = "TH,1000,250";
  for (int i = 0; i < kThermalPoints; ++i) th += ",215";
  uint8_t c = 0;
  for (char ch : th) c ^= static_cast<uint8_t>(ch);
  char tail[8];
  std::snprintf(tail, sizeof tail, "*%02X\r\n", c);
  const std::string buf = "$" + th + tail + frame("LA,cocked") + frame("HB,1001");
  ASSERT_GT(buf.size(), 3000u);
  std::vector<std::string> types;
  size_t start = 0, nl;
  while ((nl = buf.find('\n', start)) != std::string::npos) {
    const auto l = parse_line(buf.substr(start, nl - start));
    ASSERT_TRUE(l);
    types.push_back(l->type);
    if (l->type == "TH") {
      EXPECT_TRUE(as_thermal(*l));
    }
    if (l->type == "LA") {
      EXPECT_EQ(as_launcher(*l)->kind, LauncherReply::kCocked);
    }
    start = nl + 1;
  }
  EXPECT_EQ(types, (std::vector<std::string>{"TH", "LA", "HB"}));
}
