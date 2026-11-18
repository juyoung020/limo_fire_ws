// 호스트 시험: g++ -std=c++17 -Wall -Wextra -Iinclude test/test_sensor_frames.cpp -o /tmp/tsf && /tmp/tsf   (test/run.sh)
#include <cstdio>
#include <vector>

#include "sensor_frames.h"

static int fails = 0;
#define CHECK(c) do { if (!(c)) { std::printf("실패 %s:%d  %s\n", __FILE__, __LINE__, #c); ++fails; } } while (0)

constexpr uint8_t frames::Thermal::kHead[4];
constexpr uint8_t frames::Radar::kHead[4];

static std::vector<uint8_t> thermal_frame(int16_t base, int16_t ambient, bool good_sum = true) {
  std::vector<uint8_t> f(frames::kThermalBytes, 0);
  f[0] = f[1] = 0x5A; f[2] = 0x02; f[3] = 0x06;
  for (int i = 0; i < frames::kThermalPoints; ++i) {
    const int16_t v = static_cast<int16_t>(base + i);
    f[4 + 2 * i] = v & 0xFF; f[5 + 2 * i] = (v >> 8) & 0xFF;
  }
  f[1540] = ambient & 0xFF; f[1541] = (ambient >> 8) & 0xFF;
  uint16_t sum = 0;
  for (int i = 0; i < 1542; i += 2) sum += static_cast<uint16_t>(f[i] | (f[i + 1] << 8));
  if (!good_sum) sum += 1;
  f[1542] = sum & 0xFF; f[1543] = sum >> 8;
  return f;
}

static void put_radar(std::vector<uint8_t>& f, int t, int x, int y, int v) {
  auto enc = [](int a) { return static_cast<uint16_t>(a >= 0 ? (0x8000 | a) : -a); };
  uint8_t* p = f.data() + 4 + 8 * t;
  const uint16_t ex = enc(x), ey = enc(y), ev = enc(v);
  p[0] = ex & 0xFF; p[1] = ex >> 8; p[2] = ey & 0xFF; p[3] = ey >> 8; p[4] = ev & 0xFF; p[5] = ev >> 8;
  p[6] = 360 & 0xFF; p[7] = 360 >> 8;
}
static std::vector<uint8_t> radar_frame() {
  std::vector<uint8_t> f(frames::kRadarBytes, 0);
  f[0] = 0xAA; f[1] = 0xFF; f[2] = 0x03; f[3] = 0x00; f[28] = 0x55; f[29] = 0xCC;
  return f;
}

int main() {
  {  // 열화상: 앞에 쓰레기, 정상 프레임, 체크섬 틀린 프레임, 정상 프레임
    frames::Thermal th;
    std::vector<uint8_t> in = {0x00, 0x5A, 0x11, 0x5A};   // 머리가 어긋난 쓰레기(5A 뒤에 5A 가 아님)
    for (auto b : thermal_frame(2000, 2345)) in.push_back(b);
    for (auto b : thermal_frame(100, 0, false)) in.push_back(b);
    for (auto b : thermal_frame(-50, -123)) in.push_back(b);
    int got = 0;
    int16_t first = 0, last_amb = 0;
    for (uint8_t b : in)
      if (th.push(b)) {
        if (++got == 1) { first = th.point_cC(0); CHECK(th.point_cC(767) == 2767); CHECK(th.ambient_cC() == 2345); }
        last_amb = th.ambient_cC();
      }
    CHECK(got == 2);
    CHECK(first == 2000);
    CHECK(last_amb == -123);
    CHECK(th.ok == 2 && th.bad == 1);
  }
  {  // 레이더: 부호 규칙(맨 윗비트 1 = 양수), 빈 표적, 머리 겹침(AA AA FF …)
    frames::Radar r;
    auto f = radar_frame();
    put_radar(f, 0, -782, 1713, -12);
    put_radar(f, 1, 300, 2500, 0);
    std::vector<uint8_t> in = {0xAA};   // 앞 프레임 꼬리가 AA 로 끝난 경우: 다음 AA FF… 를 놓치면 안 된다
    for (auto b : f) in.push_back(b);
    int got = 0;
    for (uint8_t b : in)
      if (r.push(b)) {
        ++got;
        const auto t0 = r.target(0), t1 = r.target(1), t2 = r.target(2);
        CHECK(!t0.none && t0.x_mm == -782 && t0.y_mm == 1713 && t0.v_cms == -12);
        CHECK(!t1.none && t1.x_mm == 300 && t1.y_mm == 2500 && t1.v_cms == 0);
        CHECK(t2.none);
      }
    CHECK(got == 1);
    auto bad = radar_frame();
    bad[29] = 0x00;
    for (uint8_t b : bad) CHECK(!r.push(b));
    CHECK(r.bad == 1 && r.ok == 1);
  }
  std::printf(fails ? "실패 %d 건\n" : "통과\n", fails);
  return fails ? 1 : 0;
}
