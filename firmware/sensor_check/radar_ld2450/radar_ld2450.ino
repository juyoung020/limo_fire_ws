// 레이더 HLK-LD2450 단독 확인 (ESP32-C3 SuperMini) — 모듈 UART 프레임을 받아 표적 3 개를 USB 시리얼로 한 줄씩 낸다.
// 근거: HLK-LD2450 Instruction manual V1.00 6 장 — 256000 8N1, 10 프레임/s, 프레임 30 바이트:
//   AA FF 03 00 | 표적 3 × 8 바이트 (x·y mm, 속도 cm/s, 거리 해상도 mm) | 55 CC
//   x·y·속도는 int16 이 아니라 "맨 윗비트 1 = 양수, 0 = 음수, 나머지 15 비트 = 크기" (매뉴얼 예: x = −782 mm, y = 1713 mm)
// 출력: "T,x1,y1,v1,x2,y2,v2,x3,y3,v3" (없는 표적은 0) 프레임마다
#include <Arduino.h>

constexpr int kRxPin = 4;           // 모듈 TX → ESP 받기
constexpr int kTxPin = 3;           // 모듈 RX ← ESP 보내기
constexpr uint32_t kModuleBaud = 256000;

static int sval(uint16_t raw) { return (raw & 0x8000) ? static_cast<int>(raw & 0x7FFF) : -static_cast<int>(raw & 0x7FFF); }

void setup() {
  Serial.begin(115200);
  Serial1.begin(kModuleBaud, SERIAL_8N1, kRxPin, kTxPin);
}

void loop() {
  static uint8_t buf[30];
  static int n = 0;
  static const uint8_t head[4] = {0xAA, 0xFF, 0x03, 0x00};
  while (Serial1.available()) {
    const uint8_t b = Serial1.read();
    if (n < 4 && b != head[n]) { n = (b == head[0]) ? 1 : 0; if (n) buf[0] = b; continue; }
    buf[n++] = b;
    if (n < 30) continue;
    n = 0;
    if (buf[28] != 0x55 || buf[29] != 0xCC) continue;
    Serial.print('T');
    for (int t = 0; t < 3; ++t) {
      const uint8_t* p = buf + 4 + 8 * t;
      const uint16_t rx = p[0] | (p[1] << 8), ry = p[2] | (p[3] << 8), rv = p[4] | (p[5] << 8);
      const bool none = (rx == 0 && ry == 0);
      Serial.printf(",%d,%d,%d", none ? 0 : sval(rx), none ? 0 : sval(ry), none ? 0 : sval(rv));
    }
    Serial.println();
  }
}
