# limo_fire_esp32_bridge

ESP32(펌웨어 [firmware/limo_fire_esp32](../../../firmware/limo_fire_esp32/README.md)) USB 시리얼 → ROS. 초음파 HC-SR04 × 2(앞 범퍼 아래 좌우 ±20°) · 열화상 GY-MCU90640 · 레이더 HLK-LD2450 ([docs/wiring.md](../../../docs/wiring.md)).

| 토픽 | 형 | 프레임 |
|---|---|---|
| `/ultrasonic/front_left` · `/ultrasonic/front_right` | `sensor_msgs/Range` (ULTRASOUND, 시야 15°, 0.02~4.0 m) | `ultrasonic_<이름>_link` (URDF `ultrasonic.xacro`) |
| `/thermal/image` | `sensor_msgs/Image` (`32FC1`, 32×24, 값 = °C, 약 4 Hz) | `thermal_link` (URDF, 자리 잠정) |
| `/thermal/max` | `std_msgs/Float32` (그 프레임 최고 온도 °C) | — |
| `/radar/targets` | `geometry_msgs/PoseArray` (있는 표적만, 최대 3, 위치 m, 약 10 Hz) | `radar_link` (URDF, 자리 잠정) |
| `/radar/velocities` | `std_msgs/Float32MultiArray` (`/radar/targets` 와 같은 순서, m/s) | — |

- 레이더 좌표: LD2450 은 x 오른쪽 +·y 앞(mm, [ESPHome 문서](https://esphome.io/components/sensor/ld2450/)) → REP-103 `x_ros = y`, `y_ros = −x`. 속도는 LD2450 원본 부호 그대로 — 다가올 때 +인지 −인지 출처마다 달라 **실물에서 걸어 보고 확인한 뒤** 여기에 적는다.
- 열화상 줄이 길다(약 3 KB × 4 Hz) — USB 속도를 921600 으로 올렸다(`esp32_bridge.yaml baud`, 펌웨어 `kBaud` 와 같아야 한다).

- 시뮬(Webots `LimoFire` `ultrasonic TRUE`)과 같은 토픽·프레임이라 연기 모드(`limo_fire_navigation` `smoke_mode` — 장애물·감속·`smoke_localization: wall`)가 그대로 쓴다.
- 메아리 없음 → `range = +inf` (sensor_msgs/Range 규약: 고정 거리 안에 물체 없음).
- 체크섬이 틀리거나 반쪽인 줄은 버린다. 포트가 없거나 끊기면 알리고 1 초마다 다시 연다. 살아 있음 줄이 3 s 넘게 없으면 알린다.

```bash
ros2 launch limo_fire_bringup robot.launch.xml esp32:=true      # 실물: 센서·구동과 같이
ros2 run limo_fire_esp32_bridge esp32_bridge --ros-args --params-file <share>/config/esp32_bridge.yaml -p port:=/dev/ttyUSB1   # 따로
```

## udev — `/dev/esp32`

설정의 포트는 `/dev/esp32` 다. 라이다(`/dev/ydlidar`)도 CP210x 일 수 있어 **칩 종류(VID:PID)만으로 고르지 말고 시리얼 번호로** 고른다.

```bash
udevadm info -q property -n /dev/ttyUSB1 | grep -E "ID_VENDOR_ID|ID_MODEL_ID|ID_SERIAL_SHORT"   # ESP32 를 꽂고
# /etc/udev/rules.d/99-limo-esp32.rules
SUBSYSTEM=="tty", ATTRS{idVendor}=="<VID>", ATTRS{idProduct}=="<PID>", ATTRS{serial}=="<시리얼>", SYMLINK+="esp32"
sudo udevadm control --reload-rules && sudo udevadm trigger
```

## 가짜 ESP32 로 시험 (실물 없이)

`fake_esp32` 가 가상 시리얼(pty)에 펌웨어와 같은 줄을 흘린다 (장면: `sweep` 오가는 거리 · `noecho` 메아리 없음 · `corrupt` 깨진 줄 섞음 · `stall` 중간에 멈춤 · `sensors` 초음파 + 열화상(80 °C 열점이 가로로 움직임) + 레이더(표적 둘). 만드는 쪽은 `include/.../fake_stream.hpp`).

```bash
lib=$(ros2 pkg prefix limo_fire_esp32_bridge)/lib/limo_fire_esp32_bridge
$lib/fake_esp32 --link /tmp/fake_esp32 --scenario sweep &
ros2 run limo_fire_esp32_bridge esp32_bridge --ros-args --params-file <share>/config/esp32_bridge.yaml -p port:=/tmp/fake_esp32
$lib/fake_bridge_check.sh      # 위를 장면마다 자동으로 돌려 토픽 값을 확인 (다리 먼저 켜기·끊김 복구 포함)
```

## 확인한 것 (2026-11-18, 열화상·레이더, 가짜 ESP32)

- `colcon test` 18 개 통과(열화상·레이더 줄 해석, 길이·숫자 틀린 줄 거부, `sensors` 장면). `fake_bridge_check.sh` 의 `sensors` 장면: `/thermal/image` 폭 32 · `/thermal/max` ≈ 80 °C · `/radar/targets` · `/radar/velocities` 나옴 (Docker `limo-fire:humble-deps`).
- 이 확인은 줄 해석·토픽 모양까지다. **실물 열화상·레이더·ESP32 와의 연결, 좌표 부호·속도 부호는 안 봤다.**

## 확인한 것 (2026-11-17, 가짜 ESP32)

- `colcon test`: 줄 해석 + 가짜 줄 만들기 단위 시험 통과. `fake_bridge_check.sh`: sweep(0.2~4 m, front_right ≈ 2 m)·noecho(`inf`)·corrupt(정상 줄만)·가짜가 죽었다 다시 떠도 이어짐 전부 통과 (Docker `limo-fire:humble`).
- **살아 있음 경고(`stall`)는 줄 만들기 단위 시험만**, 다리 경고 로그까지는 안 봤다. 실물 ESP32·HC-SR04 는 여전히 안 돌려 봤다.

## 확인한 것 (2026-11-06, 데스크톱 Jazzy)

- 줄 해석 단위 테스트 통과 (`colcon test --packages-select limo_fire_esp32_bridge`: 체크섬·반쪽 줄·앞쪽 쓰레기·필드 오류).
- 가짜 ESP32(pty 에 펌웨어 형식 줄)로 끝까지: 1234 mm → 0 번 센서 1.234 m, 0 mm → 1 번 센서 +inf (11-06, 그때 이름 front·left), 체크섬 틀린 줄 → 안 냄. 포트 없음 → 1 초마다 다시 엶.
- **실물 ESP32·HC-SR04 에서는 아직 안 돌려 봤다.**
