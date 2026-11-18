// 열화상 GY-MCU90640 · 레이더 HLK-LD2450 모듈 UART 프레임 조립·검사. Arduino 의존 없음(호스트에서 test/ 로 시험).
// 근거: firmware/sensor_check/{thermal_gymcu90640,radar_ld2450} 의 단독 시험(2026-11-11)과 두 모듈 매뉴얼.
//   열화상: 115200 8N1, 프레임 1544 B = 5A 5A 02 06 | 768 점 × int16 (°C×100, 작은 바이트 먼저) | 모듈 온도 int16 | 체크섬(앞 771 워드의 16 bit 합)
//   레이더: 256000 8N1, 프레임 30 B = AA FF 03 00 | 표적 3 × 8 B (x·y mm, 속도 cm/s, 거리 해상도 mm) | 55 CC
//           x·y·속도는 int16 이 아니라 "맨 윗비트 1 = 양수, 0 = 음수, 나머지 15 비트 = 크기"
#pragma once
#include <stdint.h>

namespace frames {

// 바이트를 하나씩 먹여 프레임 머리를 맞추고 길이를 채운다. 다 차면 push 가 true.
template <int N, int H>
struct Assembler {
  uint8_t buf[N];
  int n = 0;
  const uint8_t* head;
  explicit Assembler(const uint8_t* h) : head(h) {}
  bool push(uint8_t b) {
    if (n < H) {
      if (b == head[n]) { buf[n++] = b; return false; }
      // 머리가 어긋남: 지금까지 받은 것 + b 의 꼬리 중 머리의 앞부분과 같은 가장 긴 것으로 다시 맞춘다
      uint8_t t[H + 1];
      for (int i = 0; i < n; ++i) t[i] = buf[i];
      t[n] = b;
      const int len = n + 1;
      for (int k = (len < H ? len : H - 1); k > 0; --k) {
        bool same = true;
        for (int i = 0; i < k; ++i) if (t[len - k + i] != head[i]) { same = false; break; }
        if (same) { for (int i = 0; i < k; ++i) buf[i] = head[i]; n = k; return false; }
      }
      n = 0;
      return false;
    }
    buf[n++] = b;
    if (n < N) return false;
    n = 0;
    return true;
  }
};

constexpr int kThermalBytes = 1544, kThermalPoints = 768;
constexpr int kRadarBytes = 30, kRadarTargets = 3;

inline int16_t le16(const uint8_t* p) { return static_cast<int16_t>(static_cast<uint16_t>(p[0] | (p[1] << 8))); }

struct Thermal {
  static constexpr uint8_t kHead[4] = {0x5A, 0x5A, 0x02, 0x06};
  Assembler<kThermalBytes, 4> a{kHead};
  uint32_t ok = 0, bad = 0;
  // 프레임이 다 찼고 체크섬이 맞으면 true (틀리면 bad 만 센다)
  bool push(uint8_t b) {
    if (!a.push(b)) return false;
    uint16_t sum = 0;
    for (int i = 0; i < kThermalBytes - 2; i += 2) sum += static_cast<uint16_t>(a.buf[i] | (a.buf[i + 1] << 8));
    if (sum != static_cast<uint16_t>(a.buf[kThermalBytes - 2] | (a.buf[kThermalBytes - 1] << 8))) { ++bad; return false; }
    ++ok;
    return true;
  }
  int16_t point_cC(int i) const { return le16(a.buf + 4 + 2 * i); }       // 0.01 °C
  int16_t ambient_cC() const { return le16(a.buf + kThermalBytes - 4); }  // 모듈 자체 온도, 0.01 °C
};

struct RadarTarget {
  int x_mm = 0, y_mm = 0, v_cms = 0;
  bool none = true;
};

inline int radar_signed(uint16_t raw) { return (raw & 0x8000) ? static_cast<int>(raw & 0x7FFF) : -static_cast<int>(raw & 0x7FFF); }

struct Radar {
  static constexpr uint8_t kHead[4] = {0xAA, 0xFF, 0x03, 0x00};
  Assembler<kRadarBytes, 4> a{kHead};
  uint32_t ok = 0, bad = 0;
  bool push(uint8_t b) {
    if (!a.push(b)) return false;
    if (a.buf[28] != 0x55 || a.buf[29] != 0xCC) { ++bad; return false; }
    ++ok;
    return true;
  }
  RadarTarget target(int t) const {
    const uint8_t* p = a.buf + 4 + 8 * t;
    const uint16_t rx = static_cast<uint16_t>(p[0] | (p[1] << 8)), ry = static_cast<uint16_t>(p[2] | (p[3] << 8)),
                   rv = static_cast<uint16_t>(p[4] | (p[5] << 8));
    RadarTarget r;
    if (rx == 0 && ry == 0) return r;   // 없는 표적
    r.none = false;
    r.x_mm = radar_signed(rx);
    r.y_mm = radar_signed(ry);
    r.v_cms = radar_signed(rv);
    return r;
  }
};

}  // namespace frames
