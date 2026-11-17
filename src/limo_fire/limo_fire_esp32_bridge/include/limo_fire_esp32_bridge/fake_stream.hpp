// 가짜 ESP32 줄 만들기 (실물 ESP32·HC-SR04 없이 다리를 시험). ROS 의존 없음 — fake_esp32 도구와 test_fake_stream 이 같이 쓴다.
// 줄 형식은 protocol.hpp 와 같다: $US,<번호>,<mm>,<ms>*XX · $HB,<ms>*XX
// 장면(scenario):
//   sweep   앞 센서는 0.3 → 3.5 m 를 오가고 뒤 센서는 2.0 m 고정 (+ 잡음 1 %)
//   noecho  메아리 없음(0 mm) 만 — 다리가 +inf 를 내는지
//   corrupt 정상 줄 사이에 체크섬이 틀린 줄·반쪽 줄·쓰레기를 섞음 — 다리가 정상 줄만 내는지
//   stall   처음 stall_after 초 뒤 stall_len 초 동안 아무것도 안 보냄(펌웨어 멈춤) — 살아 있음 경고·복구
#pragma once

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

namespace limo_fire_esp32_bridge {

inline std::string frame_line(const std::string& body) {
  uint8_t c = 0;
  for (char ch : body) c ^= static_cast<uint8_t>(ch);
  char tail[8];
  std::snprintf(tail, sizeof tail, "*%02X\r\n", c);
  return "$" + body + tail;
}

enum class Scenario { Sweep, NoEcho, Corrupt, Stall };

inline bool parse_scenario(const std::string& s, Scenario& out) {
  if (s == "sweep") out = Scenario::Sweep;
  else if (s == "noecho") out = Scenario::NoEcho;
  else if (s == "corrupt") out = Scenario::Corrupt;
  else if (s == "stall") out = Scenario::Stall;
  else return false;
  return true;
}

struct FakeParams {
  int sensors = 2;
  double stall_after = 5.0, stall_len = 6.0;   // s
  double slot_s = 0.035;                       // 펌웨어 kSlotMs: 센서마다 35 ms 번갈아
};

// 시각 t(s) 에 보낼 줄들. step 은 0, 1, 2, … (슬롯 번호, 센서 = step % sensors, t = step × slot_s).
// 살아 있음 줄은 1 s 마다(슬롯 28 번마다). 같은 step 이면 같은 줄이 나온다(잡음은 step 에서 정해짐).
inline std::vector<std::string> lines_at(Scenario sc, int64_t step, const FakeParams& p = {}) {
  std::vector<std::string> out;
  const double t = static_cast<double>(step) * p.slot_s;
  const int idx = static_cast<int>(step % p.sensors);
  const uint32_t ms = static_cast<uint32_t>(t * 1000.0);
  if (sc == Scenario::Stall && t >= p.stall_after && t < p.stall_after + p.stall_len) return out;
  double dist = 0.0;   // m, 0 = 메아리 없음
  switch (sc) {
    case Scenario::Sweep: {
      if (idx == 0) {
        const double ph = std::fmod(t / 10.0, 2.0);   // 10 s 에 한 번 오감
        dist = 0.3 + 3.2 * (ph < 1.0 ? ph : 2.0 - ph);
      } else {
        dist = 2.0;
      }
      const double n = 0.01 * dist * std::sin(static_cast<double>(step) * 12.9898);   // 결정적 잡음(재현)
      dist += n;
      break;
    }
    case Scenario::NoEcho: dist = 0.0; break;
    case Scenario::Corrupt:
    case Scenario::Stall: dist = 1.0 + 0.1 * idx; break;
  }
  const uint32_t mm = static_cast<uint32_t>(std::lround(dist * 1000.0));
  char body[64];
  std::snprintf(body, sizeof body, "US,%d,%u,%u", idx, mm, ms);
  out.push_back(frame_line(body));
  if (sc == Scenario::Corrupt && step % 3 == 1) {
    out.push_back("$US,0,1500,100*00\r\n");                      // 체크섬 틀림
    out.push_back(frame_line("US,1,2500,100").substr(0, 9));      // 반쪽 줄(줄 끝도 없음)
    out.push_back("\x01\x02garbage\r\n");                         // 쓰레기
  }
  if (step % 28 == 0) {
    std::snprintf(body, sizeof body, "HB,%u", ms);
    out.push_back(frame_line(body));
  }
  return out;
}

}  // namespace limo_fire_esp32_bridge
