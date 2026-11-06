// ESP32 → PC 시리얼 줄 형식 (README "시리얼 형식"). 브리지(limo_fire_esp32_bridge protocol.hpp)와 같아야 한다.
//   $<종류>,<필드>,...*<XOR 체크섬 16 진 2 자리>\r\n   — 체크섬 = '$' 와 '*' 사이 글자의 XOR (NMEA 0183 꼴)
//   $US,<번호>,<mm>,<ms>    초음파 번호(0 부터) 거리(mm, 0 = 메아리 없음) ESP32 시각(ms)
//   $HB,<ms>                살아 있음
#pragma once
#include <stdint.h>
#include <stdio.h>
#include <string.h>

inline uint8_t nmea_checksum(const char* body, size_t n) {
  uint8_t c = 0;
  for (size_t i = 0; i < n; ++i) c ^= static_cast<uint8_t>(body[i]);
  return c;
}

// body("US,0,1234,5678") → out("$US,0,1234,5678*4F\r\n"). 돌려주는 값 = 길이, 버퍼가 모자라면 0
inline size_t frame_line(const char* body, char* out, size_t cap) {
  const size_t n = strlen(body);
  if (n + 7 > cap) return 0;
  const int w = snprintf(out, cap, "$%s*%02X\r\n", body, nmea_checksum(body, n));
  return w > 0 ? static_cast<size_t>(w) : 0;
}
