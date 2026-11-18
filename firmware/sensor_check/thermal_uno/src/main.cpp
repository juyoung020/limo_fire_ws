// 열화상 GY-MCU90640 단독 확인 (Arduino Uno) — 모듈 UART 프레임을 받아 체크섬을 검사하고 온도를 USB 시리얼로 한 줄씩 낸다.
// 출력은 ESP32-S3 시험 코드(thermal_gymcu90640)와 같다: "F,ta,t1,...,t768" (0.1 °C 정수) · "S,ok=N,bad=M" 한 장마다.
//   → tools/sensor_view 가 그대로 그린다 (실행: sensor_view /dev/ttyUSB0 --baud 115200, m 키로 좌우 반전).
// 배선: 모듈 TX → Uno 0 번 핀(하드웨어 UART RX) · VIN 5V · GND 공통 · 모듈 RX·PS 는 비워 둔다. 올릴 때는 0 번 핀 선을 뽑는다(USB 와 같은 선).
// 왜 하드웨어 UART 인가: AltSoftSerial(8 번 핀)로 115200 을 받으면 실물(2026-11-18)에서 거의 모든 장이 체크섬에서 깨졌다(ok 0~50 %, 흔들림).
//   PJRC 문서도 16 MHz AVR 에서 115200 은 다른 인터럽트가 거의 없을 때만 된다고 한다.
// 대가: 모듈과 USB 가 같은 115200 을 쓴다. 한 장(약 3.2 KB 글)을 내는 280 ms 동안 오는 다음 장은 버퍼가 넘쳐 깨지고,
//   그다음 머리에서 다시 맞춘다 → 제대로 받는 장은 초당 약 1.2 장(실물). 빠르게 보려면 ESP32-S3(thermal_gymcu90640, 약 4 장/초).
// 메모리: 프레임 1544 B 를 통째로 받아야 체크섬을 안다 → Uno SRAM 2 KB 중 대부분을 쓴다(빌드 때 사용량을 본다).
#include <Arduino.h>

#include "sensor_frames.h"

constexpr uint32_t kBaud = 115200;   // 모듈 기본 = USB

static frames::Thermal th;

void setup() { Serial.begin(kBaud); }

void loop() {
  while (Serial.available()) {
    if (!th.push(static_cast<uint8_t>(Serial.read()))) continue;
    Serial.print(F("F,"));
    Serial.print(th.ambient_cC() / 10);
    for (int i = 0; i < frames::kThermalPoints; ++i) {
      Serial.print(',');
      Serial.print(th.point_cC(i) / 10);
    }
    Serial.println();
    Serial.print(F("S,ok="));
    Serial.print(th.ok);
    Serial.print(F(",bad="));
    Serial.println(th.bad);
  }
}
