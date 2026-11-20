# limo_fire_esp32 — ESP32 펌웨어

LIMO 에 더하는 모듈을 ESP32 하나로 다루는 **하나뿐인** 펌웨어(Arduino 프레임워크, PlatformIO): 초음파 HC-SR04 × 2 · 열화상 · 레이더 · 탁구공 발사대 레버 서보 (ESP32-S3 개발 보드, 핀은 [docs/wiring.md](../../docs/wiring.md)).
예전 발사대 시험 펌웨어(`launcher_test`)는 여기로 합쳤다 — 센서 줄을 내보내는 채로 발사대 명령을 받는다
(연기 모드 위치 추정·장애물, [docs/smoke_mode_plan.md](../../docs/smoke_mode_plan.md) 5 단계). 열화상(UART1)·레이더(UART2)도 읽는다(ESP32-S3 만 — C3 는 UART 가 모자라 초음파만). 발사대 응답은 같은 시리얼 형식의 줄 종류 `LA` 다.
ROS 쪽은 [limo_fire_esp32_bridge](../../src/limo_fire/limo_fire_esp32_bridge/README.md).

```bash
pip install platformio            # 한 번
cd firmware/limo_fire_esp32
pio run                           # 빌드
pio run -t upload                 # USB 로 올리기
pio device monitor                # 921600, 줄이 나오는지 보기
test/run.sh                       # 호스트 시험(보드 없이): 열화상·레이더 프레임, 발사대 명령 + 발사 다리 시험용 /tmp/launcher_host_sim 빌드
pio run -e esp32s3_nosonar -t upload   # 초음파 끔(열화상·레이더만), 시리얼은 S3 USB 직결 포트로
```

## 초음파 배선 (잠정 — `include/config.h`, 실물 장착 때 확인)

| 센서 (ROS 이름) | Trig | Echo |
|---|---|---|
| 0 front_left (앞 왼쪽 +20°) | GPIO 6 | GPIO 7 |
| 1 front_right (앞 오른쪽 −20°) | GPIO 15 | GPIO 16 |

| 센서 (ROS 이름) | ESP32-C3 SuperMini (`pio run -e esp32c3`, **잠정·실물 배선 때 확인**) |
|---|---|
| 0 front_left | Trig GPIO 3 · Echo GPIO 4 |
| 1 front_right | Trig GPIO 5 · Echo GPIO 6 |

C3 에는 GPIO 15·16 이 없어 `config.h` 가 칩에 따라 핀을 고른다. 부팅 핀(2·8·9)·USB(18·19)는 피했다. USB 시리얼은 칩 안의 USB-CDC(`ARDUINO_USB_CDC_ON_BOOT`) — 포트 이름이 `/dev/ttyACM*` 일 수 있어 udev 규칙을 따로 본다.

- HC-SR04 는 5 V 로 돌고 **Echo 도 5 V** 다 — ESP32(3.3 V) 에 바로 꽂지 말고 전압 분배(예: 1 kΩ / 2 kΩ) 뒤 꽂는다(docs/hardware.md). GND 는 모두 공통.
- 둘을 **번갈아** 쏜다(센서마다 35 ms, 한 바퀴 70 ms ≈ 14 Hz) — 동시에 쏘면 서로의 메아리를 받는다(docs/wiring.md, 60 ms 이상).
- 거리 = Echo HIGH 길이(왕복 µs) × 343 m/s ÷ 2. 데이터시트는 340 m/s 로 셈 — 20 °C 공기 343 m/s 를 썼다. 25 ms 안에 메아리가 없거나 2~400 cm 밖이면 0(메아리 없음).
- 근거: HC-SR04 데이터시트(ElecFreaks) — 측정 2~400 cm, 감지 각 15°, 트리거 10 µs 이상, 거리 = 펄스 폭 × 음속 / 2.

## 시리얼 형식 (921600 baud, 한 줄씩)

```
$<종류>,<필드>,...*<체크섬>\r\n     체크섬 = '$' 와 '*' 사이 글자의 XOR, 16 진 2 자리 (NMEA 0183 꼴)
$US,<번호>,<mm>,<ms>                 초음파 번호(0 부터) 거리 mm(0 = 메아리 없음) ESP32 시각 ms
$TH,<ms>,<ta>,<t1>…<t768>             열화상: 모듈 온도 ta, 768 점(32×24, 행 먼저), 모두 0.1 °C 정수, 약 4 Hz
$RD,<ms>,x1,y1,v1,x2,y2,v2,x3,y3,v3   레이더 표적 3 개(없으면 0,0,0): x 오른쪽 +·y 앞 mm, v cm/s(LD2450 원본 부호), 약 10 Hz
$HB,<ms>                             살아 있음 (1 s 마다)
$LA,<몸통>                           발사대 응답 (아래 "발사대")
```

## 발사대 (`include/launcher.h`, 값은 `config.h`)

**레버 서보(GPIO 4) 하나만** 움직인다. 각도 서보는 뗐고 발사대 각은 기구로 15° 고정(2026-11-20) — GPIO 5 는 부팅부터 끝까지 LOW, 각도 명령은 없다.
호스트 → ESP32 는 글자 한 줄씩(같은 포트·같은 921600, 사람이 시리얼 모니터에서 쳐도 됨). 응답은 센서 줄과 같은 `$…*체크섬` 틀의 `$LA,…` 줄이다.
ROS 쪽은 [limo_fire_esp32_bridge](../../src/limo_fire/limo_fire_esp32_bridge/README.md) 가 이 포트를 혼자 열고 액션 `/launcher/launch`·서비스 `/launcher/enable` 로 낸다.

| 명령 | 하는 일 | 응답 (`$LA,…`) |
|---|---|---|
| `e` | 레버 PWM 켜기 — 기록된 자리(처음엔 쉼 100°)로 | `on` (실패 `error,attach`) |
| `x` | 끄기(핀 LOW). **움직이는 중에도 바로** 듣고 쌓인 명령을 버림 | `off` |
| `c` · `f` · `s` | 장전 · 발사 · 장전+발사 (끝났을 때 응답) | `cocked` · `fired` · `cocked` 다음 `fired` |
| `h` · `l<0..180>` | 쉼 자리로 · 레버 서보 각(각 찾을 때, 60°/s) | `home` · `lever,<각>` |
| `?` | 상태 | `state,<레버 각>,<on\|off>,<busy 0\|1>` |
| (부팅) | 칩이 켜짐 — 레버 꺼짐 | `boot` |
| (거절) | 꺼진 채 움직임 명령 · 쌓을 자리(4 개) 넘침 · 모르는 명령·범위 밖 | `blocked,off` · `error,busy` · `error,bad` |

- 레버 각(`config.h`, 2026-11-16 실물에서 맞춤): 쉼 100° · 장전 30° · 발사 125°. 장전은 300°/s 로 가서 1.0 s 버티고 150°/s 로 돌아와 0.3 s 쉼(≈ 2.0 s), 발사 ≈ 0.45 s, 장전+발사 ≈ 2.4 s (`test/run.sh` 의 호스트 시험이 잼).
- 서보는 20 ms 마다 한 걸음씩 움직여 **센서 읽기를 막지 않는다**. 움직이는 동안 온 명령은 4 개까지 쌓았다가 차례로.
- 응답은 펌웨어가 명령을 끝냈다는 뜻이다. 레버·공의 실제 움직임을 잰 것이 아니다.
- **포트를 열 때 칩이 다시 켜진다**(COM 포트의 DTR/RTS 자동 리셋) → `$LA,boot`, 레버 꺼짐. 그 순간 핀이 잠깐 떠 서보가 튈 수 있다 — 서보 6 V 를 켜기 전에 포트를 열거나, 신호선–GND 에 10 kΩ 풀다운을 단다.
- ESP32-C3 판은 GPIO 4 가 초음파 Echo 라 발사대를 넣지 않는다(`LF_LAUNCHER 0`).

열화상 줄이 3 KB 쯤이라 115200 으로는 모자라서 USB 속도를 921600 으로 했다(`config.h kBaud`, 다리 `esp32_bridge.yaml baud` 와 같게). 모듈 프레임(열화상 1544 B·레이더 30 B) 해석은 `include/sensor_frames.h`.

**열화상·레이더 배선** (`docs/wiring.md` 핀표, S3): 열화상 TX → GPIO 10 · RX ← 11 (UART1, 115200) / 레이더 TX → GPIO 17 · RX ← 18 (UART2, 256000, **전원 5 V**).

체크섬이 틀리거나 반쪽인 줄은 다리가 버린다. 형식 정의는 `include/protocol.h` 와 다리의 `protocol.hpp` 두 곳 — 바꾸면 같이 바꾼다.

## 확인한 것 (2026-11-20, `esp32s3`, Jetson 에 CH343(COM) 포트로)

- 10 초 동안 `$HB` 1 초마다, 열화상 `$TH` 3.8 Hz(모듈 30.2 °C, 화소 21.3~25.3 °C), 레이더 `$RD` 11 Hz(표적 0), 초음파 왼쪽(0) 0.75~1.70 m.
- **초음파 오른쪽(1)은 140 번 모두 0(메아리 없음)** — GPIO 15·16 배선·전압 분배를 볼 것.
- 서보 신호선이 이 펌웨어에서 떠 있어 레버 서보(DS3218)가 계속 돌던 것 → 부팅 때 GPIO 4·5 를 LOW 로 잡게 고침(`config.h kServoHoldPin`).
- 칩 내장 USB 포트로 꽂으면 기본 `esp32s3` 는 아무것도 안 나온다(시리얼이 UART0) — COM 포트로 꽂는다.

## 확인한 것 (2026-11-18, `esp32s3_nosonar`)

- S3 USB 직결 포트(Windows COM5)로 올려 `$HB` 가 1 초마다 나옴. 열화상 `$TH` 는 0 줄 — 진단 코드로 GPIO 10·11 에서 신호 변화 0, 115200·9600·57600·460800 과 TX/RX 바꿔서도 0 바이트(GPIO 10 은 풀다운에 떨어져 떠 있음). 모듈 전원·GND 접촉을 먼저 볼 것. 레이더는 꽂지 않음
- 기본 `esp32s3` 는 시리얼이 UART0(CH343 COM 포트)로 나가서 USB 직결 포트에는 아무것도 안 나온다 — 그래서 이 환경에서 USB 직결로 돌렸다

## 확인한 것 (2026-11-18)

- `pio run` 열화상·레이더 포함 빌드: `esp32s3` RAM 10.9 %·Flash 9.0 %, `esp32c3`(초음파만) RAM 4.4 %·Flash 21.7 %, 둘 다 경고 0. `test/run.sh`(프레임 조립·체크섬·레이더 부호, 어긋난 머리 다시 맞추기) 통과.
- **열화상·레이더를 실물 ESP32 에 꽂아 돌려 보지는 않았다** (센서 단독 시험은 `firmware/sensor_check`, 2026-11-11). 초음파 때문에 열화상 줄을 쓰는 동안 초음파 간격이 늘 수 있다 — 늦어지면 몰아 쏘지 않고 한 칸 건너뛴다.

## 확인한 것 (2026-11-06)

- `pio run` 빌드 성공(경고 0, RAM 6.8 %, Flash 22.3 %). 2026-11-17 다시: platform espressif32 8.0.0 에서 `esp32s3`(Flash 8.9 %)·`esp32c3`(RAM 4.4 %, Flash 21.7 %) 둘 다 경고 0 으로 성공(Docker python:3.12 + PlatformIO 6.2). **실물 ESP32·HC-SR04 에서는 아직 안 돌려 봤다.**
