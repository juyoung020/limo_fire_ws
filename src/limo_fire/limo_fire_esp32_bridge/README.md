# limo_fire_esp32_bridge

ESP32(펌웨어 [firmware/limo_fire_esp32](../../../firmware/limo_fire_esp32/README.md) — 센서와 발사대가 **한 펌웨어**) USB 시리얼 ↔ ROS.
초음파 HC-SR04 × 2(앞 범퍼 아래 좌우 ±20°) · 열화상 GY-MCU90640 · 레이더 HLK-LD2450 · 탁구공 발사대 레버 서보(GPIO 4) ([docs/wiring.md](../../../docs/wiring.md)).
**포트는 이 노드 하나만 연다**(921600 baud) — 센서 줄과 발사대 명령·응답이 같은 포트로 오간다. 발사대 각은 기구로 15° 고정이라 각도 명령은 없다.

## 인터페이스

머리 시각(`header.stamp`)은 **다리가 줄을 받은 시각**이다(ESP32 ms 시계는 호스트와 달라 쓰지 않음, USB 지연 수 ms). 단위는 SI(REP-103: m·rad·°C), 프레임 이름은 URDF 링크와 같다.
센서 토픽은 `SensorDataQoS`(best effort, 깊이 5), 상태 토픽은 TRANSIENT_LOCAL(늦게 붙어도 지금 상태를 받음).

| 종류 | 이름 | 형 | QoS | 프레임·뜻 |
|---|---|---|---|---|
| 토픽 | `/ultrasonic/front_left` · `/ultrasonic/front_right` | `sensor_msgs/Range` (`ULTRASOUND`, 시야 15°, 0.02~4.0 m, 메아리 없음 = `+inf`) | 센서 | `ultrasonic_<이름>_link` (URDF `ultrasonic.xacro`) |
| 토픽 | `/thermal/image` | `sensor_msgs/Image` (`32FC1`, 32×24, 값 = °C, 약 4 Hz) | 센서 | `thermal_link` (URDF, 자리 잠정) |
| 토픽 | `/thermal/max` | `sensor_msgs/Temperature` (그 프레임 화소 최고 °C, `variance` 0 = 모름) | 센서 | `thermal_link` |
| 토픽 | `/radar/targets` | `geometry_msgs/PoseArray` (있는 표적만, 최대 3, 위치 m, 약 10 Hz) | 센서 | `radar_link` (URDF, 자리 잠정) |
| 토픽 | `/radar/velocities` | `std_msgs/Float32MultiArray` (`/radar/targets` 와 같은 차례, m/s, LD2450 원본 부호) | 센서 | — |
| 액션 | `/launcher/launch` | `limo_fire_msgs/action/Launch` — 목표 `mode` `COCK`·`FIRE`·`SHOT`, 피드백 `phase` (`enabling`·`cocking`·`firing`), 결과 `success`·`message`·`cocked` | 기본 | 장전·발사·장전+발사 (≈ 2.0 · 0.45 · 2.4 s, 펌웨어 응답 기준). **취소 = 펌웨어 `x`** |
| 서비스 | `/launcher/enable` | `std_srvs/SetBool` | 기본 | `true` 레버 PWM 켜기(쉼 자리로) · `false` 끄기(늘 받음, 고장 풀기). 펌웨어 응답(`LA,on`·`LA,off`)을 받은 뒤 답한다 |
| 토픽 | `/launcher/state` | `std_msgs/String` | TRANSIENT_LOCAL | `disconnected`(포트 없음·펌웨어 응답 아직) · `off` · `ready`(레버 켜짐) · `busy` · `cocked` · `fault` |
| 토픽 | `/launcher/status` | `std_msgs/String` | TRANSIENT_LOCAL | 한글 한 줄: 마지막 결과·거절·고장 까닭 |
| 토픽 | `/launcher/joint_states` | `sensor_msgs/JointState` — `launcher_lever_joint` 위치(rad) | 기본 | 레버 서보 **명령 각**(펌웨어가 알려 준 값, 잰 각 아님). 쉴 때 1 s 마다·동작 뒤. URDF 에 이 관절은 없다(아래) |
| 토픽 | `/diagnostics` | `diagnostic_msgs/DiagnosticArray` (`diagnostic_updater`, 1 Hz) | 기본 | `esp32_bridge: link` — 포트·baud·살아 있음 나이·줄 종류별 빈도(`rate_US_hz` `rate_TH_hz` `rate_RD_hz` `rate_HB_hz` `rate_LA_hz`)·깨진 줄 수·펌웨어 부팅 수. `esp32_bridge: launcher` — 상태·장전·레버·마지막 결과(고장이면 ERROR) |

이름은 그대로 두었다: `/ultrasonic/*` 는 연기 모드·Nav2 초음파 층·임무 BT·녹화가 쓰고, `/thermal/*`·`/radar/*` 는 아직 쓰는 패키지가 없다(임무 BT 는 잠정 이름 `/thermal/image_raw`·`/radar/people` 을 적어 둠 — 맞출 때 그쪽에서).
`/thermal/max` 는 2026-11-20 `std_msgs/Float32` → `sensor_msgs/Temperature` 로 바꿨다(머리 시각·프레임이 생김, 쓰는 곳 없었음).

발사대 상태 기계(`include/limo_fire_esp32_bridge/launcher_logic.hpp`, 예전 Python `pitch_serial_bridge` 의 안전 규칙):
- `launcher.firing_enabled` 기본 `true`(2026-11-20 사용자 결정, `robot.launch.xml launcher_firing` 도 기본 true). `false` 면 켜기·장전·발사를 받지 않는다(끄기는 받음).
- 장전된 채 `COCK`·`SHOT` 거절, 장전 응답 없이 `FIRE` 거절, 다른 동작 중이면 거절 — 거절은 목표를 받은 뒤 `abort` 하고 `message` 에 까닭(`거절: …`).
- 레버가 꺼져 있으면 동작 전에 `e` → `LA,on` 을 받고 동작 명령을 보낸다.
- 응답 기한(`ack_timeout_s` 3 s, 동작 `action_timeout_s` 8 s)을 넘기면 `fault` — `x` 로 멈추게만 하고 **다시 시도하지 않는다**.
- 레버가 켜져 있거나 동작 중일 때 `LA,boot`(펌웨어 재시작)·포트 끊김·막힘/오류 응답·순서 틀린 응답 → `fault`. 꺼진 채 쉴 때의 `LA,boot`(포트를 열면 칩이 다시 켜짐)는 알리기만.
- 취소 → `x` → `LA,off` → 목표 `canceled`, 장전 여부를 모르므로 `fault`. `fault` 는 사람이 확인하고 `/launcher/enable false` 로 푼다. 고장이 나면 장전 기억도 버린다.
- 펌웨어를 아직 모르면(포트를 연 직후) 1 s 마다 `?` 로 묻는다. 열화상 줄(≈ 3 KB)과 발사대 응답이 섞여도 줄 단위로 나눠 읽는다.

선택한 것과 남긴 것:
- 레버는 URDF 관절이 아니다. 기구가 레버 각을 공 위치로 바꾸는 모양이 모델에 없어서 관절을 지어내지 않았다(TF 발행자도 만들지 않음). `/launcher/joint_states` 의 `launcher_lever_joint` 는 기록·화면용이다.
- 발사대 각(`launcher_pitch_joint`)은 서보가 없어 이 다리가 내지 않는다. `limo_fire_description` `description.launch.xml` 의 `joint_state_publisher` 가 고정값 15°(`zeros`)로 그린다.
- 레이더: [`radar_msgs`](https://github.com/ros-perception/radar_msgs)(`RadarTracks`)가 Humble 에 있지만(`ros-humble-radar-msgs`) 로봇에 깔려 있는지 확인하지 않아 지금은 `PoseArray` + 속도 배열을 그대로 둔다. 바꾸면 쓰는 쪽과 같이.
- `diagnostic_updater` 는 Humble 패키지(`ros-humble-diagnostic-updater`)다. 로봇에 없으면 `sudo apt install ros-humble-diagnostic-updater`.

- 레이더 좌표: LD2450 은 x 오른쪽 +·y 앞(mm, [ESPHome 문서](https://esphome.io/components/sensor/ld2450/)) → REP-103 `x_ros = y`, `y_ros = −x`. 속도는 LD2450 원본 부호 그대로 — 다가올 때 +인지 −인지 출처마다 달라 **실물에서 걸어 보고 확인한 뒤** 여기에 적는다.
- 열화상 줄이 길다(약 3 KB × 4 Hz) — USB 속도를 921600 으로 올렸다(`esp32_bridge.yaml baud`, 펌웨어 `kBaud` 와 같아야 한다).
- 시뮬(Webots `LimoFire` `ultrasonic TRUE`)과 같은 토픽·프레임이라 연기 모드(`limo_fire_navigation` `smoke_mode`)가 그대로 쓴다.
- 체크섬이 틀리거나 반쪽인 줄은 버린다(진단의 깨진 줄 수). 포트가 없거나 끊기면 알리고 1 초마다 다시 연다. 살아 있음 줄이 3 s 넘게 없으면 알린다.

```bash
ros2 launch limo_fire_bringup robot.launch.xml esp32:=true [esp32_port:=/dev/serial/by-id/…] [launcher_firing:=true]   # 실물: 센서·구동과 같이
ros2 run limo_fire_esp32_bridge esp32_bridge --ros-args --params-file <share>/config/esp32_bridge.yaml -p port:=/dev/ttyUSB1   # 따로
```

## udev — `/dev/esp32`

설정의 포트는 `/dev/esp32` 다. 라이다(`/dev/ydlidar`)도 CP210x 일 수 있어 **칩 종류(VID:PID)만으로 고르지 말고 시리얼 번호로** 고른다.
지금 로봇의 ESP32-S3 보드 COM 포트(CH343): `1a86:55d3`, 시리얼 `5CE6033354` (2026-11-20, `/dev/serial/by-id/usb-1a86_USB_Single_Serial_5CE6033354-if00`).

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

## 발사대 시험 (실물 없이)

```bash
firmware/limo_fire_esp32/test/run.sh     # /tmp/launcher_host_sim = 펌웨어 원본(src/main.cpp)을 PC 가상 시리얼에
colcon test --packages-select limo_fire_esp32_bridge
ros2 action send_goal /launcher/launch limo_fire_msgs/action/Launch '{mode: 3}' --feedback   # 손으로: SHOT
ros2 service call /launcher/enable std_srvs/srv/SetBool '{data: false}'
```

- `test_launcher_logic`: 상태 기계를 시간을 넣어 가며(켜기 먼저·거절·기한·재시작·취소·순서 틀림·끊김·쓰기 실패).
- `test_bridge_host_sim`: 실행 파일 `esp32_bridge` ↔ 펌웨어 원본 호스트 시뮬. SHOT(레버 꺼진 채 → 켜고 장전·발사, 2~6 s), 장전 없이 FIRE 거절, COCK → `cocked` → 다시 COCK·SHOT 거절 → FIRE, 끄기, SHOT 도중 취소 → `canceled`·`fault` → 끄기로 풀림, 같은 포트로 초음파 줄 계속, 레버 관절·진단 나옴, 레버 PWM 장전(< 900 µs)·발사(> 1850 µs), 각도 서보 핀은 켜지지 않음.

## 확인한 것 (2026-11-20, 발사대, 펌웨어 호스트 시뮬)

- `colcon test` gtest 37 개 통과(줄 해석 15 · 가짜 줄 6 · 상태 기계 15 · 호스트 시뮬 끝까지 1, 이 PC Humble), 빌드 경고 0. 세 번 되풀이해도 같음.
- **실물 ESP32 에 올린 펌웨어·레버 서보로는 아직 안 돌려 봤다.** 호스트 시뮬은 펌웨어 코드·타이밍은 같지만 USB 자동 리셋(DTR/RTS)은 흉내 내지 않는다.

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
