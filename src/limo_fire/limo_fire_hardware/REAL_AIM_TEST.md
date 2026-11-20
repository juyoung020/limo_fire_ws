# 실물 LIMO 발사기 장착값·좌우 조준 시험 (무장전)

**장전·발사는 이 절차에 없다**(`auto_fire_enabled:=false`, 로봇 `launcher_firing:=false`). 공·고무줄을 빼고 회전 범위를 비운 채로 한다.
발사대 각은 기구로 **15° 고정**이라 상하 조준은 없다 — 이 시험은 정지 거리와 좌우만 본다.
실물 표적은 카메라 RGB 에서 탐지하고 깊이·카메라 보정·TF 로 지도 좌표를 구한다. 손으로 넣은 표적 좌표는 쓰지 않는다.

## 1. 장착값

`base_link` 는 LIMO 모델에서 바닥 위 14.5 cm, 앞뒤 바퀴 축 가운데, 좌우 중심선. 회전축 `launcher_pivot` 은 발사대 받침의 회전축 중심(지금은 돌지 않음).
값은 URDF `limo_fire_description/urdf/limo_fire.urdf.xacro` 의 `launcher_xyz`(m, `base_link` 기준)와 `yaw` 에 넣는다. 조준은 TF `base_link → launcher_pivot` 으로 읽는다.

| 값 | 재는 곳·부호 | 지금 URDF 값 |
|---|---|---|
| x | 바퀴 앞뒤 축 가운데 → 회전축, 앞 + | −3.99 cm, CAD 값(재지 않음) |
| y | 차체 좌우 중심선 → 회전축, 왼쪽 + | 0, CAD 값(재지 않음) |
| 회전축 높이 | 바닥 → 회전축 중심. URDF z = 높이 − 14.5 cm | 21.0 cm 실측(2026-11-20) → z 0.065 m. CAD 는 20.25 cm |
| yaw | 차체 앞(+x) → 발사기 수평 방향, 왼쪽 회전 + | 0°, CAD 값(재지 않음) |
| 발사대 각 | 수평 → 총신, 위 + | **15° 고정**(각도 서보 뗌). `description.launch.xml` 이 `launcher_pitch_joint` 를 15° 로 그린다 |
| 총구 | 회전축 기준 | URDF `launcher_muzzle`(CAD: 0° 에서 앞 6.57 cm, 위 4.5 cm). 계산에는 안 씀 |

좌우로 빗나가면 y·yaw 를, 거리가 틀리면 x 와 `desired_distance_m` 를 먼저 의심한다. 실물 설정 파일에서 `mount_calibrated: true`(저장소 기본은 안전을 위해 `false`).

```bash
ros2 run tf2_ros tf2_echo base_link launcher_pivot    # 로봇을 켠 뒤. 기대: URDF 에 넣은 x y z, 회전 0
```

## 2. 보드와 로봇 확인 (읽기만)

ESP32-S3 GPIO 4 = 레버 서보. GPIO 5 는 쓰지 않고 부팅 때부터 LOW. 서보 외부 전원 GND 와 ESP32 GND 를 공통으로. 장치 경로는 [REAL_FIRE_TEST.md](REAL_FIRE_TEST.md) 0 절대로 고르고, 라이다(CP2102, `/dev/ydlidar`)에 쓰지 않는다.

```bash
source /home/wego/limo_fire_ws/env.sh
export ROS_DOMAIN_ID=0 ROS_LOCALHOST_ONLY=1
ls -l /dev/serial/by-id/
ros2 topic echo /limo_status --once
ros2 action list -t
```

제자리 요레이트 정렬은 `/limo_status.motion_mode` 가 차동 `0`·메카넘 `2` 일 때만. Ackermann `1` 이면 `yaw_alignment_mode:=nav2`. 리모컨은 오토 모드여야 섀시가 `/cmd_vel` 을 받는다.

## 3. 카메라 탐지·거리와 경로 미리보기

인식 경로: Orbbec RGB `/camera/color/image_raw` → YOLO `/perception/detections` → 깊이 `/camera/depth/image_raw`·`camera_info`·TF → `/perception/objects`(`map` 좌표) → `fire_observation_bridge` → `/fire/observations`.
물체 지도는 깊이 픽셀·분할 마스크·바닥 광선 추정을 함께 쓰므로 거리를 줄자값과 견준다. `fire_observation_bridge` 는 **최근 0.5 s 안에 다시 본 `fire`** 만 넘긴다. 불이 여럿이면 `target_id:=fire-1` 처럼 하나를 준다.

```bash
# 터미널 1, 2: 로봇·Nav2 (REAL_FIRE_TEST.md 1 절). RViz 2D Pose Estimate 로 위치를 맞춘다
# 터미널 3
ros2 launch limo_fire_hardware real_camera_aim.launch.xml config:=/home/wego/launcher_aim_measured.yaml motion_enabled:=false
# 터미널 4
ros2 topic info -v /fire/observations       # 발행자가 fire_observation_bridge 하나
ros2 topic echo /perception/objects --once
ros2 topic echo /fire/observations --once   # map 좌표, position_valid true, confidence ≥ 0.5
ros2 service call /launcher/approach_aim/start std_srvs/srv/Trigger '{}'
ros2 topic echo /launcher/approach_aim/status
```

`preview_ready` 에 후보 자리·차체 yaw·회전축 값이 나오고, 경로가 공·사람·장애물을 지나지 않아야 한다. `map` TF 가 없으면 지도 좌표가 없어 조준할 수 없다.

## 4. 무장전 실물 조준

```bash
ros2 launch limo_fire_hardware real_camera_aim.launch.xml config:=/home/wego/launcher_aim_measured.yaml \
  motion_enabled:=true yaw_alignment_mode:=yaw_rate yaw_rate_drive_mode_verified:=true   # 0·2 를 확인했을 때만
ros2 service call /launcher/approach_aim/start std_srvs/srv/Trigger '{}'   # 한 번
```

기대 순서: `planning → approaching → wait_stop → reacquiring → yaw_aligning → wait_stop → reacquiring → aim_ready`.
이미 좌우가 맞으면 `yaw_aligning` 에서 바로 넘어갈 수 있다. 발사하지 않는다. 방향이 반대로 돌거나 표적·오도메트리가 끊기면 바로 취소하고 정지를 확인한다.

```bash
ros2 service call /launcher/approach_aim/cancel std_srvs/srv/Trigger '{}'
ros2 topic echo /launcher/approach_aim/state
```

기록: 커밋, 장착값, ESP32 경로, `motion_mode`, 끝 상태, 멈춘 자리에서 표적까지 거리·실제 좌우 오차.

## 5. 지금 확인 범위

- C++ 판: PC 에서 가짜 Nav2·차체·표적 + 펌웨어 호스트 시뮬로 시험 통과(README "확인한 것"). **실물 카메라 표적·Nav2 주행·차체 회전은 아직 안 함.**
- 예전 Python 판(2026-11-19~20): Wego 에서 Orbbec RGB·깊이 수신, YOLO 약 30 장/s, 인식·매핑·조준 토픽 생성까지. 표적이 없고 `map` TF 가 없어 실제 탐지·지도 좌표는 확인 못 함.
