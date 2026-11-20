# ESP32-S3 실물 장전·발사와 별도 터미널

대상: 센서와 한 펌웨어인 `firmware/limo_fire_esp32`(레버 서보 GPIO 4 하나, **발사대 각은 기구로 15° 고정**).
시리얼은 ESP32 다리(`limo_fire_esp32_bridge`, `robot.launch.xml esp32:=true`)가 혼자 연다 — 921600 baud, 센서 줄과 발사대 응답 `$LA,…` 이 같은 포트로 온다.
발사대 동작 끝(`shot_done`, 액션 결과 `success`)은 펌웨어가 명령을 마쳤다는 뜻이며 공의 비행·명중을 재지 않는다.
**C++ 판(2026-11-20)은 아직 실물에서 돌려 보지 않았다.** 아래 순서는 그 첫 실물 시험용이다.

## 실물 데모 실행 순서

단계마다 **기대 결과가 나와야만** 다음으로 간다. 로봇이 움직이는 단계에서는 사람이 리모컨을 들고 곁에 있는다(리모컨 SWB 를 내리면 섀시가 ROS 명령을 받지 않음). 한 번에 움직이는 거리는 0.6 m 안.

**모든 터미널 공통**

```bash
source /home/wego/limo_fire_ws/env.sh
export ROS_DOMAIN_ID=0 ROS_LOCALHOST_ONLY=1
ros2 pkg prefix limo_fire_hardware     # 기대: /home/wego/limo_fire_ws/install/limo_fire_hardware
```

### 0. 한 번만 하는 준비

1. **빌드**: `colcon build --packages-select limo_fire_msgs limo_fire_esp32_bridge limo_fire_hardware limo_fire_description limo_fire_bringup` 후 `source install/setup.bash`.
2. **시리얼 권한**: 2026-11-20 기준 `wego` 는 `dialout` 그룹이 아니었다. `sudo usermod -aG dialout wego` → 다시 로그인 → `groups` 에 `dialout`.
3. **ESP32 장치 경로**: 꽂기 전후 `ls -l /dev/serial/by-id/` 를 견줘 새로 생긴 것만 쓴다. `usb-Silicon_Labs_CP2102…`(`/dev/ydlidar`)는 라이다다. 아래에서 `ESP=/dev/serial/by-id/…`.
   udev 로 `/dev/esp32` 를 만들었으면 그것을 쓴다([다리 README](../limo_fire_esp32_bridge/README.md#udev--devesp32)).
4. **펌웨어 업로드**(서보 전원 끈 채): `cd firmware/limo_fire_esp32 && pio run -t upload --upload-port $ESP`.
   예전 발사대 전용 이미지(`launcher_s3_arduino_3.3.12_20261119.merged.bin`)는 각도 서보·옛 명령을 쓰는 펌웨어라 **쓰지 않는다**.
5. **장착값**: URDF `launcher_xyz`·`yaw` (근거는 [REAL_AIM_TEST.md](REAL_AIM_TEST.md) 1 절). 조준 설정을 복사해 `mount_calibrated: true`:
   ```bash
   cp $(ros2 pkg prefix limo_fire_hardware)/share/limo_fire_hardware/config/approach_aim.yaml /home/wego/launcher_aim_measured.yaml
   sed -i -E "s/mount_calibrated: *(true|false)/mount_calibrated: true/" /home/wego/launcher_aim_measured.yaml
   ```

### 1. 로봇·지도·Nav2 (차체 움직임 없음)

충전기를 빼고 섀시 전원·리모컨을 켠다(SWB 위). 표적(YOLO `fire` 로 잡히는 물체)의 실제 위치를 줄자로 재어 둔다.

```bash
# 터미널 1 — 로봇 + ESP32 다리(발사 허용은 아직 끔)
ros2 launch limo_fire_bringup robot.launch.xml base:=true viz:=false max_clocks:=false esp32:=true esp32_port:=$ESP
# 터미널 2 — Cartographer + Nav2 + 꺼진 프론티어 탐사
ros2 launch limo_fire_explore frontier_explorer.launch.xml stack:=true rviz:=true
```

확인(터미널 4):
```bash
ros2 topic echo /limo_status --once                     # motion_mode 기록 (0 차동 · 1 Ackermann · 2 메카넘)
ros2 topic hz /odometry/filtered                        # 수십 Hz
ros2 run tf2_ros tf2_echo map base_link                 # Translation 이 나옴
ros2 run tf2_ros tf2_echo base_link launcher_pivot      # URDF 값(지금 -0.0399 0 0.065), 회전 0
ros2 topic echo /explore/frontier_explorer/state --once # idle
ros2 topic echo /launcher/state --once                  # off (펌웨어를 알아봄). disconnected 면 포트·펌웨어 확인
ros2 topic echo /diagnostics --once                     # esp32_bridge: link 가 OK, rate_US_hz·rate_HB_hz 가 0 이 아님
```

### 2. 발사기 연결 확인 (공·고무줄 뺌, 차체 정지)

터미널 1 을 `launcher_firing:=true` 로 다시 켠다(발사 허용은 다리 설정이라 로봇 launch 인자다).

```bash
# 터미널 1
ros2 launch limo_fire_bringup robot.launch.xml base:=true viz:=false max_clocks:=false esp32:=true esp32_port:=$ESP launcher_firing:=true
# 터미널 4
ros2 run limo_fire_hardware launcher_terminal --log /home/wego/launcher-commands.jsonl
#   c → completed (레버 장전) / f → completed (레버 발사) / s → completed / x → 레버 꺼짐 / q
```

- 레버가 꺼져 있으면 다리가 먼저 `e` 로 켠다(서보가 쉼 자리 100° 로 감). 장전된 채 `c`·`s`, 장전 없이 `f` 는 다리가 거절한다.
- 레버가 반대로 돌거나 걸리면 `x`(또는 터미널 1 Ctrl+C — 다리가 끌 때 `x`)를 보내고 서보 전원을 내린다. 펌웨어 `config.h` 레버 각(쉼 100°·장전 30°·발사 125°)을 맞춘 뒤 다시.
- `/launcher/state` 가 `fault` 면 `/launcher/status` 의 까닭을 보고, 기구를 확인한 뒤 터미널 `x` 로 푼다. 응답 기한을 넘긴 명령은 다시 보내지 않는다.

### 3. 카메라 표적 확인과 미리보기 (차체 정지)

```bash
# 터미널 3
ros2 launch limo_fire_hardware real_camera_aim.launch.xml config:=/home/wego/launcher_aim_measured.yaml motion_enabled:=false
# 터미널 4
ros2 topic echo /fire/observations --once                 # frame_id map, position_valid true, confidence ≥ 0.5
ros2 service call /launcher/approach_aim/start std_srvs/srv/Trigger '{}'
ros2 topic echo /launcher/approach_aim/status --once      # preview_ready: 후보 x·y·yaw, 회전축 값
```

`/fire/observations` 좌표를 1 단계 줄자값과 견준다. 30 cm 넘게 다르면 멈춘다. 처음에는 표적을 로봇 앞 약 1.8 m 에 둔다(정지 거리 1.2 m → 이동 약 0.6 m).

### 4. 무장전 자동 임무 리허설 (공·고무줄 뺌, 차체 이동)

```bash
# 터미널 3
ros2 launch limo_fire_hardware real_camera_aim.launch.xml config:=/home/wego/launcher_aim_measured.yaml \
  motion_enabled:=true auto_fire_enabled:=true
# 터미널 4
ros2 run limo_fire_hardware launcher_terminal --log /home/wego/launcher-commands.jsonl
#   a  (한 번)
```

기대 순서(표적이 안 보이면 먼저 탐사):
`planning → approaching → wait_stop → reacquiring → aligning → wait_stop → reacquiring → wait_shot → shot_done`

멈춘 자리에서 표적까지 거리·좌우 오차를 줄자로 기록한다. 이상하면 터미널 4 에서 Ctrl+C(탐사 정지·임무 취소·발사 동작 취소)와 함께 리모컨을 수동으로. 다시 하려면 `r` → `a`.

### 5. 실제 장전·발사 한 번

4 단계가 기대대로 끝났을 때만. 공이 날아갈 쪽에 사람·깨질 물건이 없어야 한다. 공·고무줄을 끼우고 촬영을 시작한다. 로봇을 4 단계 출발점에 다시 두고 터미널 4 에서 `r`, `a` 를 한 번씩.

기록: 커밋, 장착값, `ESP` 경로, `motion_mode`, 터미널 기록(`/home/wego/launcher-commands.jsonl`), 끝 상태, 영상에서 본 공의 비행·착탄과 표적 사이 오차, 정지 거리. `shot_done` 만으로는 발사 성공이 아니다.

## 알아 둘 것

- **포트는 하나**: 다리가 포트를 연 채로 `pio device monitor`·다른 시리얼 터미널을 같이 열지 않는다. 포트를 열 때 칩이 다시 켜진다(DTR/RTS) — 다리는 `$LA,boot` 로 알고 레버를 꺼진 것으로 본다. 레버가 켜져 있거나 동작 중에 재시작되면 `fault`.
- 발사 동작은 중간 취소(`x`)가 된다 — 펌웨어가 움직이던 것을 바로 멈추고 쌓인 명령을 버린다. 장전이 어디까지 됐는지 모르므로 다리는 `fault` 로 두고, 확인한 뒤 `x` 로 풀어야 다음 명령을 받는다.
- 터미널 `a` 는 launch 의 `auto_fire_enabled:=true` 를 켜는 스위치가 아니라, 켜진 자동 임무를 한 번 시작하는 명령이다. `/launcher/approach_aim/start` 를 직접 부르면 탐색 단계는 거치지 않는다.
- 상태: `/explore/frontier_explorer/state`, `/launcher/approach_aim/state`·`status`, `/launcher/state`·`status`, `/diagnostics`.

## 예전 기록 (Python 판, 2026-11-19~20)

C++ 로 옮기기 전 Python 패키지로 한 확인이다. 지금 코드에는 그대로 옮겨지지 않는다(각도 서보·옛 펌웨어 기준).
- Wego 가짜 Nav2·차체·표적 시험 83~84 개 통과, 펌웨어 호스트 시뮬 ↔ Python 다리 연결 확인(그때는 각도 서보 포함).
- Wego 센서 점검: 컬러 17.5 Hz, 깊이 23 Hz, 라이다 10 Hz. 섀시 드라이버가 `/dev/ttylimo` 를 열었지만 섀시 데이터가 안 와서 `/limo_status`·오도메트리·`map` TF 가 없었다. 표적이 없어 탐지 0.
- CP2102 `ttyUSB0` 은 라이다(`/dev/ydlidar`). 그때 ESP32-S3 는 연결돼 있지 않았다. 실물의 탐색·이동·조준·장전·발사 연속 실행은 한 번도 하지 않았다.
- 실물 launch 결함 하나를 고쳤다: 인식 launch 에 넘긴 `config:=yolo_fire.yaml` 이 새어 조준 노드가 실측 설정 대신 받던 것 → 인식 include 를 범위 제한 그룹으로 감쌈(지금 `real_camera_aim.launch.xml` 도 같다).
