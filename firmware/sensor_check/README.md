# sensor_check — 모듈 하나씩 ESP32 에 꽂아 확인하는 시험 코드

로봇 펌웨어([limo_fire_esp32](../limo_fire_esp32/README.md))에 넣기 전에, 새 모듈이 데이터를 맞게 내는지 모듈마다 따로 보는 Arduino 스케치.
결과와 알게 된 것은 [docs/hardware.md "열화상·레이더 단독 연결 시험"](../../docs/hardware.md#열화상레이더-단독-연결-시험-2026-11-11).

| 폴더 | 모듈 | 보드 | 출력 (USB 시리얼, 한 줄씩) |
|---|---|---|---|
| `thermal_gymcu90640/` | 열화상 GY-MCU90640 (MLX90640, 32×24) | ESP32-S3-N16R8 개발 보드 | `F,ta,t1,…,t768` (0.1 °C 정수, 921600 baud) · 2 초마다 `S,ok=…,bad=…` |
| `radar_ld2450/` | 레이더 HLK-LD2450 (24 GHz, 표적 3) | ESP32-C3 SuperMini | `T,x1,y1,v1,x2,y2,v2,x3,y3,v3` (mm, cm/s, 없으면 0, 115200 baud) |

## 배선

열화상 → ESP32-S3 (모듈 아래 줄 4 핀만)

| 모듈 | ESP32-S3 |
|---|---|
| VIN | 3V3 (3~5 V 모두 됨) |
| GND | GND |
| TX | GPIO 10 |
| RX | GPIO 11 |
| SCL · SDA · NC · PS | 꽂지 않음 — **PS 가 GND 에 닿으면 UART 가 꺼진다** |

레이더 → ESP32-C3 SuperMini (5V 가 있는 줄만)

| 모듈 | ESP32-C3 |
|---|---|
| 5V | **5V** — 3.3 V 로는 안 켜진다 |
| GND | G |
| TX | GPIO 4 |
| RX | GPIO 3 |

핀 번호는 각 스케치 맨 위 `kRxPin`·`kTxPin` 에 있다. TX 는 상대 RX 로 엇갈려 꽂는다.

## 빌드·올리기 (arduino-cli)

```bash
arduino-cli core install esp32:esp32      # 한 번 (보드 주소: https://espressif.github.io/arduino-esp32/package_esp32_index.json)

# 열화상 → S3. 보드의 USB 포트에 꽂았으면 CDCOnBoot=cdc, COM(UART) 포트면 CDCOnBoot=default
arduino-cli compile --fqbn esp32:esp32:esp32s3:FlashSize=16M,PSRAM=opi,CDCOnBoot=cdc firmware/sensor_check/thermal_gymcu90640
arduino-cli upload -p /dev/ttyACM0 --fqbn esp32:esp32:esp32s3:FlashSize=16M,PSRAM=opi,CDCOnBoot=cdc firmware/sensor_check/thermal_gymcu90640
arduino-cli monitor -p /dev/ttyACM0 -c baudrate=921600

# 레이더 → C3
arduino-cli compile --fqbn esp32:esp32:esp32c3:CDCOnBoot=cdc firmware/sensor_check/radar_ld2450
arduino-cli upload -p /dev/ttyACM0 --fqbn esp32:esp32:esp32c3:CDCOnBoot=cdc firmware/sensor_check/radar_ld2450
arduino-cli monitor -p /dev/ttyACM0 -c baudrate=115200
```

- 리눅스는 `dialout` 그룹이어야 포트를 연다(`sudo usermod -aG dialout $USER` 뒤 다시 로그인).
- 다시 꽂으면 포트 번호가 `ttyACM1` 등으로 바뀔 수 있다(`ls /dev/ttyACM*`).
- 열화상 `S,ok=0,bad=0` 만 나오면 프레임이 안 오는 것: 핀 번호(스케치 vs 실제 배선)·PS 핀·전원 순으로 본다.

## 데이터 형식 근거

- 열화상: GYMCU90640 模块使用手册 V1.0 — 115200 8N1, 프레임 1544 바이트 `5A 5A 02 06` + 768 점(°C × 100, int16, 작은 바이트 먼저) + 모듈 온도 + 체크섬(앞 771 워드 합).
  점 순서 (Col1,Row1)…(Col32,Row1),(Col1,Row2)…, (Col1,Row1) 은 매뉴얼 그림의 오른쪽 위.
- 레이더: HLK-LD2450 Instruction manual V1.00 6 장 — 256000 8N1, 10 프레임/s, `AA FF 03 00` + 표적 3 × 8 바이트 + `55 CC`.
  x·y·속도는 "맨 윗비트 1 = 양수, 나머지 15 비트 = 크기" (매뉴얼 예 x = −782 mm, y = 1713 mm 와 같게 풂).

화면으로 보려면 [tools/sensor_view](../../tools/sensor_view/README.md) (열화상 색 지도·레이더 부채꼴). 로봇에서는 이 줄들을 `limo_fire_esp32_bridge` 가 ROS 토픽으로 바꿀 예정이다.
