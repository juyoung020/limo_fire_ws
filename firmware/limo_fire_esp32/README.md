# limo_fire_esp32 — ESP32 펌웨어

LIMO 에 더하는 모듈을 ESP32 하나로 다루는 펌웨어(Arduino 프레임워크, PlatformIO). **지금은 초음파 HC-SR04 × 3 만** 있다
(연기 모드 위치 추정·장애물, [docs/smoke_mode_plan.md](../../docs/smoke_mode_plan.md) 5 단계). 레이더·열화상·서보·펌프는 같은 시리얼 형식에 줄 종류를 더한다.
ROS 쪽은 [limo_fire_esp32_bridge](../../src/limo_fire/limo_fire_esp32_bridge/README.md).

```bash
pip install platformio            # 한 번
cd firmware/limo_fire_esp32
pio run                           # 빌드
pio run -t upload                 # USB 로 올리기
pio device monitor                # 115200, 줄이 나오는지 보기
```

## 초음파 배선 (잠정 — `include/config.h`, 실물 장착 때 확인)

| 센서 (ROS 이름) | Trig | Echo |
|---|---|---|
| 0 front | GPIO 25 | GPIO 34 |
| 1 left | GPIO 26 | GPIO 35 |
| 2 right | GPIO 27 | GPIO 32 |

- HC-SR04 는 5 V 로 돌고 **Echo 도 5 V** 다 — ESP32(3.3 V) 에 바로 꽂지 말고 전압 분배(예: 1 kΩ / 2 kΩ) 뒤 꽂는다(docs/hardware.md). GND 는 모두 공통.
- 셋을 **하나씩 차례로** 쏜다(센서마다 35 ms, 한 바퀴 105 ms ≈ 9.5 Hz) — 동시에 쏘면 서로의 메아리를 받는다(docs/spec.md).
- 거리 = Echo HIGH 길이(왕복 µs) × 343 m/s ÷ 2. 데이터시트는 340 m/s 로 셈 — 20 °C 공기 343 m/s 를 썼다. 25 ms 안에 메아리가 없거나 2~400 cm 밖이면 0(메아리 없음).
- 근거: HC-SR04 데이터시트(ElecFreaks) — 측정 2~400 cm, 감지 각 15°, 트리거 10 µs 이상, 거리 = 펄스 폭 × 음속 / 2.

## 시리얼 형식 (115200 baud, 한 줄씩)

```
$<종류>,<필드>,...*<체크섬>\r\n     체크섬 = '$' 와 '*' 사이 글자의 XOR, 16 진 2 자리 (NMEA 0183 꼴)
$US,<번호>,<mm>,<ms>                 초음파 번호(0 부터) 거리 mm(0 = 메아리 없음) ESP32 시각 ms
$HB,<ms>                             살아 있음 (1 s 마다)
```

체크섬이 틀리거나 반쪽인 줄은 다리가 버린다. 형식 정의는 `include/protocol.h` 와 다리의 `protocol.hpp` 두 곳 — 바꾸면 같이 바꾼다.

## 확인한 것 (2026-11-06)

- `pio run` 빌드 성공(경고 0, RAM 6.8 %, Flash 22.3 %). **실물 ESP32·HC-SR04 에서는 아직 안 돌려 봤다.**
