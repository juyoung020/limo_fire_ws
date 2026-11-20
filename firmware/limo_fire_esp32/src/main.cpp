// LIMO 화재 대응 ESP32 펌웨어 — 초음파 HC-SR04 × 2 (연기 모드 위치 추정·장애물) · 열화상 GY-MCU90640 · 레이더 HLK-LD2450 · 탁구공 발사대 레버 서보 (docs/wiring.md, include/launcher.h).
// 하나씩 차례로 트리거(10 µs 이상 HIGH, 데이터시트) → Echo HIGH 길이 = 왕복 시간 → 거리. 결과를 USB 시리얼 한 줄씩 (include/protocol.h).
#include <Arduino.h>

#include "config.h"
#if LF_LAUNCHER
#include "launcher.h"
#endif
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

#if LF_LAUNCHER
// 레버 서보 PWM: 50 Hz(20 ms), 14 비트 (Arduino-ESP32 2.x 는 채널 0 을 직접 고른다)
constexpr int kPwmFreq = 50, kPwmRes = 14;
bool pwm_attach(int pin) {
#if ESP_ARDUINO_VERSION_MAJOR >= 3
  return ledcAttach(pin, kPwmFreq, kPwmRes);
#else
  ledcSetup(0, kPwmFreq, kPwmRes);
  ledcAttachPin(pin, 0);
  return true;
#endif
}
void pwm_detach_low(int pin) {
#if ESP_ARDUINO_VERSION_MAJOR >= 3
  ledcDetach(pin);
#else
  ledcDetachPin(pin);
#endif
  pinMode(pin, OUTPUT);
  digitalWrite(pin, LOW);
}
void pwm_write_us(int pin, int us) {
  const uint32_t duty = static_cast<uint32_t>(static_cast<uint64_t>(us) * ((1u << kPwmRes) - 1) / 20000u);
#if ESP_ARDUINO_VERSION_MAJOR >= 3
  ledcWrite(pin, duty);
#else
  (void)pin;
  ledcWrite(0, duty);
#endif
}
launcher::Launcher launch({pwm_attach, pwm_detach_low, pwm_write_us, send});
char cmd_buf[32];
size_t cmd_len = 0;
bool cmd_overflow = false;

// 호스트 → ESP32 명령 줄 (발사대). 너무 긴 줄은 통째로 버린다
void read_commands() {
  while (Serial.available()) {
    const char ch = static_cast<char>(Serial.read());
    if (ch == '\n' || ch == '\r') {
      if (cmd_len && !cmd_overflow) { cmd_buf[cmd_len] = '\0'; launch.line(cmd_buf); }
      cmd_len = 0;
      cmd_overflow = false;
    } else if (cmd_len + 1 < sizeof cmd_buf) {
      cmd_buf[cmd_len++] = ch;
    } else {
      cmd_overflow = true;
    }
  }
}
#endif

}  // namespace

void setup() {
  for (uint8_t i = 0; i < kNumServoHold; ++i) {   // 서보 신호선을 맨 먼저 LOW 로 (떠 있으면 서보가 돎 — config.h)
    pinMode(kServoHoldPin[i], OUTPUT);
    digitalWrite(kServoHoldPin[i], LOW);
  }
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
#if LF_LAUNCHER
  launch.boot();   // $LA,boot — 다리가 칩이 다시 켜진 것(레버 꺼짐)을 안다
#endif
}

void loop() {
  const uint32_t now = millis();
#if LF_LAUNCHER
  read_commands();
  launch.update(now);
#endif
#if LF_THERMAL
  while (Serial1.available()) if (thermal.push(Serial1.read())) send_thermal(now);
#endif
#if LF_RADAR
  while (Serial2.available()) if (radar.push(Serial2.read())) send_radar(now);
#endif
  if (LF_SONAR && static_cast<int32_t>(now - next_slot_ms) >= 0) {
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
