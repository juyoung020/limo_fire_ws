// 초음파 HC-SR04 × 2 단독 확인 (Arduino Uno) — 로봇 펌웨어 firmware/limo_fire_esp32 와 같은 줄($US·$HB)을 USB 115200 으로 낸다.
// 배선: 센서 VCC → 5V, GND → GND, Trig·Echo → 아래 핀 (Uno 는 5 V 입력이라 Echo 전압 분배가 필요 없다)
// 거리 = Echo HIGH 길이(왕복 µs) × 343 m/s ÷ 2. 25 ms 안에 메아리가 없거나 2~400 cm 밖이면 0 (HC-SR04 데이터시트, 로봇 펌웨어와 같은 규칙)
#include <Arduino.h>
#include "protocol.h"

constexpr uint8_t kNum = 2;
constexpr uint8_t kTrig[kNum] = {9, 11};    // 0 front_left, 1 front_right (브리지 names 순서)
constexpr uint8_t kEcho[kNum] = {10, 12};
constexpr unsigned long kTimeoutUs = 25000;
constexpr unsigned long kSlotMs = 35;       // 센서마다 35 ms, 번갈아 쏜다(동시에 쏘면 서로 메아리를 받음)

static void send(const char* body) {
  char line[48];
  if (frame_line(body, line, sizeof(line))) Serial.print(line);
}

void setup() {
  Serial.begin(115200);
  for (uint8_t i = 0; i < kNum; ++i) {
    pinMode(kTrig[i], OUTPUT);
    digitalWrite(kTrig[i], LOW);
    pinMode(kEcho[i], INPUT);
  }
}

void loop() {
  static unsigned long last_hb = 0;
  char body[32];
  for (uint8_t i = 0; i < kNum; ++i) {
    const unsigned long t0 = millis();
    digitalWrite(kTrig[i], LOW);
    delayMicroseconds(2);
    digitalWrite(kTrig[i], HIGH);
    delayMicroseconds(10);
    digitalWrite(kTrig[i], LOW);
    const unsigned long us = pulseIn(kEcho[i], HIGH, kTimeoutUs);
    unsigned long mm = us * 343UL / 2000UL;
    if (mm < 20 || mm > 4000) mm = 0;
    snprintf(body, sizeof(body), "US,%u,%lu,%lu", i, mm, millis());
    send(body);
    while (millis() - t0 < kSlotMs) {
    }
  }
  if (millis() - last_hb >= 1000) {
    last_hb = millis();
    snprintf(body, sizeof(body), "HB,%lu", last_hb);
    send(body);
  }
}
