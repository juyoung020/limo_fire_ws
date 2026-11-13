// 배선·주기 설정 (잠정 — 실물 장착 때 확인해 고친다). 값의 근거는 README "초음파".
#pragma once
#include <stdint.h>

// HC-SR04 × 2 (ESP32-S3 개발 보드, docs/wiring.md 핀표): 순서 = ROS 쪽 이름 순서(esp32_bridge.yaml ultrasonic.names: front_left, front_right)
// Echo 는 5 V 라 전압 분배(1 kΩ / 2 kΩ, docs/wiring.md "초음파 ECHO 전압 분배") 뒤 ESP32 에
constexpr uint8_t kNumSonar = 2;
constexpr uint8_t kTrigPin[kNumSonar] = {6, 15};
constexpr uint8_t kEchoPin[kNumSonar] = {7, 16};

// 번갈아 쏜다(동시에 쏘면 서로의 메아리를 받음 — docs/wiring.md, 60 ms 이상). 센서 하나당 35 ms → 2 개면 한 바퀴 70 ms ≈ 14 Hz
constexpr uint32_t kSlotMs = 35;
// 메아리 기다림 한도: 4 m 왕복 = 8 m / 343 m/s ≈ 23.3 ms → 25 ms. 넘으면 "메아리 없음"(0 mm 로 보냄)
constexpr uint32_t kEchoTimeoutUs = 25000;
// 음속 (m/s, 20 °C). 데이터시트 식은 340 m/s — 20 °C 공기 343 m/s 를 쓴다. 온도에 따라 약 0.6 m/s/°C 달라진다
constexpr uint32_t kSoundSpeedMmPerMs = 343;
// 데이터시트 측정 범위 2~400 cm. 밖이면 메아리 없음으로 보낸다
constexpr uint32_t kMinMm = 20, kMaxMm = 4000;

constexpr uint32_t kBaud = 115200;
constexpr uint32_t kHeartbeatMs = 1000;   // 살아 있음 줄 (브리지가 끊김을 안다)
