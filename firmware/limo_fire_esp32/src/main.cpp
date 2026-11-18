// LIMO 화재 대응 ESP32 펌웨어 — 초음파 HC-SR04 × 2 (연기 모드 위치 추정·장애물) · 열화상 GY-MCU90640 · 레이더 HLK-LD2450 (docs/wiring.md).
// 하나씩 차례로 트리거(10 µs 이상 HIGH, 데이터시트) → Echo HIGH 길이 = 왕복 시간 → 거리. 결과를 USB 시리얼 한 줄씩 (include/protocol.h).
#include <Arduino.h>

#include "config.h"
#include "protocol.h"
#include "sensor_frames.h"

namespace {

uint8_t next_sonar = 0;
uint32_t next_slot_ms = 0, next_hb_ms = 0;
#if LF_THERMAL
frames::Thermal thermal;
char thermal_body[6000], thermal_line[6100];   // 768 점 × 최대 7 글자
#endif
#if LF_RADAR
frames::Radar radar;
#endif

void send(const char* body) {
  char line[128];
  const size_t n = frame_line(body, line, sizeof line);
  if (n) Serial.write(reinterpret_cast<const uint8_t*>(line), n);
}

#if LF_THERMAL
void send_thermal(uint32_t now) {
  int w = snprintf(thermal_body, sizeof thermal_body, "TH,%lu,%d", static_cast<unsigned long>(now), thermal.ambient_cC() / 10);
  for (int i = 0; i < frames::kThermalPoints && w > 0 && w < static_cast<int>(sizeof thermal_body) - 8; ++i)
    w += snprintf(thermal_body + w, sizeof thermal_body - w, ",%d", thermal.point_cC(i) / 10);
  const size_t n = frame_line(thermal_body, thermal_line, sizeof thermal_line);
  if (n) Serial.write(reinterpret_cast<const uint8_t*>(thermal_line), n);
}

#endif
#if LF_RADAR
void send_radar(uint32_t now) {
  char body[128];
  int w = snprintf(body, sizeof body, "RD,%lu", static_cast<unsigned long>(now));
  for (int t = 0; t < frames::kRadarTargets; ++t) {
    const frames::RadarTarget r = radar.target(t);
    w += snprintf(body + w, sizeof body - w, ",%d,%d,%d", r.x_mm, r.y_mm, r.v_cms);
  }
  send(body);
}
#endif

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
#if LF_THERMAL
  {
    Serial1.setRxBufferSize(4096);   // 프레임 1544 B 가 134 ms — 초음파 pulseIn(최대 25 ms) 동안 쌓이는 걸 받는다
    Serial1.begin(kThermalBaud, SERIAL_8N1, kThermalRxPin, kThermalTxPin);
    delay(500);
    const uint8_t auto_mode[4] = {0xA5, 0x35, 0x02, 0xDC};   // 자동(연속) 출력, flash 저장은 안 함 (매뉴얼)
    Serial1.write(auto_mode, 4);
  }
#endif
#if LF_RADAR
  Serial2.setRxBufferSize(1024);
  Serial2.begin(kRadarBaud, SERIAL_8N1, kRadarRxPin, kRadarTxPin);
#endif
  next_slot_ms = next_hb_ms = millis();
}

void loop() {
  const uint32_t now = millis();
#if LF_THERMAL
  while (Serial1.available()) if (thermal.push(Serial1.read())) send_thermal(now);
#endif
#if LF_RADAR
  while (Serial2.available()) if (radar.push(Serial2.read())) send_radar(now);
#endif
  if (static_cast<int32_t>(now - next_slot_ms) >= 0) {
    // 열화상 줄을 쓰느라 늦어졌으면 몰아 쏘지 않고 한 칸만 건너뛴다(센서 사이 간격 kSlotMs 를 지킨다)
    next_slot_ms = (static_cast<int32_t>(now - next_slot_ms) > static_cast<int32_t>(kSlotMs)) ? now + kSlotMs : next_slot_ms + kSlotMs;
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
