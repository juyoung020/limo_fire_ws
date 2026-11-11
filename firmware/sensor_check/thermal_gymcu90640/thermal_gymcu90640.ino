// 열화상 GY-MCU90640 단독 확인 (ESP32-S3) — 모듈 UART 프레임을 받아 체크섬을 검사하고 온도를 USB 시리얼로 한 줄씩 낸다.
// 근거: GYMCU90640 模块使用手册 V1.0 — 115200 8N1(기본), 프레임 1544 바이트:
//   5A 5A | 02 06(데이터 수 0x0602) | 768 점 × int16 (°C × 100, 작은 바이트 먼저) | 모듈 자체 온도 | 체크섬(앞 771 워드 16 bit 합)
//   점 순서 (Col1,Row1)…(Col32,Row1),(Col1,Row2)… / 명령 A5 35 02 DC = 자동(연속) 출력. PS 핀이 GND 면 UART 가 꺼진다(I2C 모드).
// 출력: "F,ta,t1,...,t768" (0.1 °C 정수) 프레임마다 · "S,ok=N,bad=M" 2 초마다(프레임이 안 오면 이것만 보임)
#include <Arduino.h>

constexpr int kRxPin = 10;          // 모듈 TX → ESP 받기
constexpr int kTxPin = 11;          // 모듈 RX ← ESP 보내기
constexpr uint32_t kModuleBaud = 115200;
constexpr uint32_t kUsbBaud = 921600;
constexpr int kFrame = 1544;

static uint8_t buf[kFrame];
static int n = 0;
static uint32_t ok = 0, bad = 0;

void setup() {
  Serial.begin(kUsbBaud);
  Serial1.setRxBufferSize(4096);
  Serial1.begin(kModuleBaud, SERIAL_8N1, kRxPin, kTxPin);
  delay(500);
  const uint8_t autoMode[4] = {0xA5, 0x35, 0x02, 0xDC};   // flash 저장은 안 함
  Serial1.write(autoMode, 4);
}

void loop() {
  while (Serial1.available()) {
    const uint8_t b = Serial1.read();
    if (n < 2 && b != 0x5A) { n = 0; continue; }
    if (n == 2 && b != 0x02) { n = (b == 0x5A) ? 2 : 0; continue; }
    buf[n++] = b;
    if (n < kFrame) continue;
    n = 0;
    uint16_t sum = 0;
    for (int i = 0; i < 1542; i += 2) sum += static_cast<uint16_t>(buf[i] | (buf[i + 1] << 8));
    if (sum != static_cast<uint16_t>(buf[1542] | (buf[1543] << 8))) { ++bad; continue; }
    ++ok;
    const int16_t ta = static_cast<int16_t>(buf[1540] | (buf[1541] << 8));
    Serial.printf("F,%d", ta / 10);
    for (int i = 0; i < 768; ++i) {
      const int16_t t = static_cast<int16_t>(buf[4 + 2 * i] | (buf[5 + 2 * i] << 8));
      Serial.printf(",%d", t / 10);
    }
    Serial.println();
  }
  static uint32_t last = 0;
  if (millis() - last > 2000) {
    Serial.printf("S,ok=%lu,bad=%lu\n", static_cast<unsigned long>(ok), static_cast<unsigned long>(bad));
    last = millis();
  }
}
