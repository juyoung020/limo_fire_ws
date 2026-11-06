// LIMO 화재 대응 ESP32 펌웨어 — 지금은 초음파 HC-SR04 × 3 (연기 모드 위치 추정·장애물, docs/smoke_mode_plan.md 5 단계).
// 하나씩 차례로 트리거(10 µs 이상 HIGH, 데이터시트) → Echo HIGH 길이 = 왕복 시간 → 거리. 결과를 USB 시리얼 한 줄씩 (include/protocol.h).
#include <Arduino.h>

#include "config.h"
#include "protocol.h"

namespace {

uint8_t next_sonar = 0;
uint32_t next_slot_ms = 0, next_hb_ms = 0;

void send(const char* body) {
  char line[64];
  const size_t n = frame_line(body, line, sizeof line);
  if (n) Serial.write(reinterpret_cast<const uint8_t*>(line), n);
}

// 거리 mm, 메아리 없음·범위 밖 = 0
uint32_t measure(uint8_t i) {
  digitalWrite(kTrigPin[i], LOW);
  delayMicroseconds(2);
  digitalWrite(kTrigPin[i], HIGH);
  delayMicroseconds(10);   // 데이터시트: 10 µs 이상
  digitalWrite(kTrigPin[i], LOW);
  const uint32_t us = pulseIn(kEchoPin[i], HIGH, kEchoTimeoutUs);
  if (us == 0) return 0;
  const uint32_t mm = us * kSoundSpeedMmPerMs / 2000;   // 왕복 µs × mm/ms ÷ 2 ÷ 1000
  return (mm < kMinMm || mm > kMaxMm) ? 0 : mm;
}

}  // namespace

void setup() {
  Serial.begin(kBaud);
  for (uint8_t i = 0; i < kNumSonar; ++i) {
    pinMode(kTrigPin[i], OUTPUT);
    digitalWrite(kTrigPin[i], LOW);
    pinMode(kEchoPin[i], INPUT);
  }
  next_slot_ms = next_hb_ms = millis();
}

void loop() {
  const uint32_t now = millis();
  if (static_cast<int32_t>(now - next_slot_ms) >= 0) {
    next_slot_ms += kSlotMs;
    const uint32_t mm = measure(next_sonar);
    char body[40];
    snprintf(body, sizeof body, "US,%u,%lu,%lu", next_sonar, static_cast<unsigned long>(mm), static_cast<unsigned long>(now));
    send(body);
    next_sonar = (next_sonar + 1) % kNumSonar;
  }
  if (static_cast<int32_t>(now - next_hb_ms) >= 0) {
    next_hb_ms += kHeartbeatMs;
    char body[24];
    snprintf(body, sizeof body, "HB,%lu", static_cast<unsigned long>(now));
    send(body);
  }
}
