# limo_fire_esp32_bridge

ESP32(펌웨어 [firmware/limo_fire_esp32](../../../firmware/limo_fire_esp32/README.md)) USB 시리얼 → ROS. 지금은 초음파 HC-SR04 × 2 만 (앞 범퍼 아래 좌우 ±20°, [docs/wiring.md](../../../docs/wiring.md)).

| 토픽 | 형 | 프레임 |
|---|---|---|
| `/ultrasonic/front_left` · `/ultrasonic/front_right` | `sensor_msgs/Range` (ULTRASOUND, 시야 15°, 0.02~4.0 m) | `ultrasonic_<이름>_link` (URDF `ultrasonic.xacro`) |

- 시뮬(Webots `LimoFire` `ultrasonic TRUE`)과 같은 토픽·프레임이라 연기 모드(`limo_fire_navigation` `sonar_scan` · `smoke_mode` `smoke_localization: sonar`)가 그대로 쓴다.
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

## 확인한 것 (2026-11-06, 데스크톱 Jazzy)

- 줄 해석 단위 테스트 통과 (`colcon test --packages-select limo_fire_esp32_bridge`: 체크섬·반쪽 줄·앞쪽 쓰레기·필드 오류).
- 가짜 ESP32(pty 에 펌웨어 형식 줄)로 끝까지: 1234 mm → 0 번 센서 1.234 m, 0 mm → 1 번 센서 +inf (11-06, 그때 이름 front·left), 체크섬 틀린 줄 → 안 냄. 포트 없음 → 1 초마다 다시 엶.
- **실물 ESP32·HC-SR04 에서는 아직 안 돌려 봤다.**
