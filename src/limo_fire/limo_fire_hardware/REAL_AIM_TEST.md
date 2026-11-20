# 실물 LIMO 발사기 좌우·상하 조준 시험

2026-11-20 측정값 입력을 위한 실행 절차. **장전·발사 명령은 이 절차에 없다.** 처음에는 공과 고무줄 하중을 제거하고 회전 범위를 확보한다. ESP32-S3에는 수정된 `launcher_test/src/main.cpp`를 올려야 한다. 기존 Uno 펌웨어나 주영 저장소 원본 ESP32 펌웨어를 이 설정과 혼동하지 않는다. 실물 표적은 카메라 RGB 영상에서 탐지하고 깊이 영상·카메라 보정값·TF로 지도 좌표를 구한다. 수동 입력 표적 좌표는 이 실물 절차에서 사용하지 않는다.

## 1. 기록할 실측값

`base_link`는 현재 LIMO 모델에서 바닥 위 14.5cm이며 앞·뒤 바퀴 축의 중간, 좌우 중심선에 있다. 회전축은 상하 서보가 발사기를 돌리는 **물리적 축 중심**이다. 값은 URDF `limo_fire_description/urdf/limo_fire.urdf.xacro`의 `launcher_xyz`(m, `base_link` 기준)와 `yaw`에 넣는다. 조준 노드는 이 값을 TF `base_link → launcher_pivot`으로 읽는다.

| 값 | 측정 위치·부호 | 지금 URDF 값 |
|---|---|---|
| x | 바퀴 앞뒤 축 중간 → 회전축, 전방 + | −3.99cm, CAD 값 사용(측정 안 함) |
| y | 차체 좌우 중심선 → 회전축, 로봇 왼쪽 + | 0, CAD 값 사용(측정 안 함) |
| 회전축 높이 | 바닥 → 회전축 중심. URDF z = 높이 − 14.5cm | 21.0cm 실측(2026-11-20) → z 0.065m. CAD는 20.25cm |
| yaw | 차체 전방(+x) → 발사기 수평 방향, 왼쪽 회전 + | 0°, CAD 값 사용(측정 안 함) |
| 총구 위치 | 회전축 기준 | URDF `launcher_muzzle`(CAD: 앞 6.57cm, 위 4.5cm, 0°에서). 탄도 계산에는 아직 미반영 |

발사 직후 속도는 파란 공 약 3.9m/s, 노란 공 약 4.2m/s의 사용자 제공 근삿값이다. 현재 기본 각도 계산은 파란 공 3.9m/s를 사용한다. 옆으로 휘는 양과 실제 발사 위치 차이는 아직 보정되지 않았다.

## 2. 보드와 로봇 확인

ESP32-S3 펌웨어의 GPIO4는 레버, GPIO5는 270° 상하 서보다. 개인 저장소 배치는 `hardware/launcher_test`, 주영 저장소 배치는 `hardware/modeling/탁구공발사대/firmware/launcher_test`가 소스 폴더다. 11월 19일 노트북의 Arduino ESP32 코어 3.3.12로 S3 소스를 **컴파일**했고 Wego에 4MB 통합 이미지를 복사했다. Wego의 PlatformIO 보드 도구 다운로드는 디스크 여유가 줄어 중단하고 방금 받은 캐시를 정리했다. 따라서 금요일에는 아래의 사전 빌드 이미지를 사용한다. 업로드와 실물 서보 반응은 아직 검증되지 않았다. 서보 외부 전원 GND와 ESP32 GND를 공통으로 연결한다. 브리지는 `e5`만 보내며 GPIO4 레버와 발사 명령은 건드리지 않는다. `e5` 순간 1500µs로 모터가 움직일 수 있다.

ESP32-S3 보드가 연결된 뒤 **장치 경로가 이 보드인지 확인하고**, 다음 명령으로 이미지를 업로드한다. 2026-11-20 확인: Wego에 보이는 CP2102(`10c4:ea60`, `ttyUSB0`)는 udev 규칙상 `/dev/ydlidar`, 즉 **라이다**다. 이 포트에 esptool을 쓰거나 발사 브리지를 연결하지 않는다. 당시 ESP32-S3는 연결돼 있지 않았다(Espressif `303a` 장치 없음). ESP32를 꽂기 전후로 `ls -l /dev/serial/by-id/`를 비교해 새로 생긴 장치만 사용한다.

```bash
sha256sum /home/wego/launcher_aim_friday_20261120/firmware/launcher_s3_arduino_3.3.12_20261119.merged.bin
# 기대 SHA-256: accec093a16b8806322f877c77dc0ba34fb1285bed00dcc25df7e092d053862e
/home/wego/.venvs/launcher-pio/bin/python3 -m esptool --chip esp32s3 \
  --port /dev/serial/by-id/<확인한_ESP32_장치명> write-flash 0x0 \
  /home/wego/launcher_aim_friday_20261120/firmware/launcher_s3_arduino_3.3.12_20261119.merged.bin
```

Wego의 기존 `/home/wego/limo_fire_ws` 작업공간에는 별도 실험 변경이 있으므로, 이번 시험의 ROS 패키지 두 개는 `/home/wego/launcher_aim_friday_20261120` 오버레이에 따로 빌드해 두었다. 기존 Nav2·로봇 실행 환경을 먼저 불러오고 시험 오버레이를 그다음 불러온다. Wego에서 각 장치와 구동 모드를 읽기 전용으로 확인한다.

```bash
source /home/wego/limo_fire_ws/env.sh
export ROS_DOMAIN_ID=0 ROS_LOCALHOST_ONLY=1
ls -l /dev/serial/by-id/
ros2 topic echo /limo_status --once
ros2 action list -t
```

제자리 요레이트 조준은 `/limo_status.motion_mode`가 차동 `0` 또는 메카넘 `2`일 때만 사용한다. Ackermann `1`이면 `yaw_alignment_mode:=nav2`를 사용한다. 리모컨은 로봇이 `/cmd_vel`을 받는 오토 모드여야 한다.

## 3. 장착값을 URDF에 넣고 TF로 확인하기

실측한 값으로 `limo_fire.urdf.xacro`의 `launcher_xyz`(`real` 항목)와 `yaw`를 고친다. 1절 표처럼 근거(실측/CAD)를 주석에 남긴다. 그다음 시험 오버레이에서 URDF와 조준 패키지를 함께 빌드한다. **로봇 실행(`robot.launch.xml`)도 이 오버레이를 source한 터미널에서 켜야** URDF에 발사대가 들어간다. 기존 작업공간의 URDF에는 발사대가 없어서 조준 노드가 실행을 거부한다.

```bash
source /home/wego/limo_fire_ws/env.sh
cd <시험 오버레이> && colcon build --packages-select limo_fire_msgs limo_fire_description limo_fire_hardware --symlink-install
source install/setup.bash
ros2 run tf2_ros tf2_echo base_link launcher_pivot    # 로봇을 켠 뒤. 기대: URDF 에 넣은 x y z, 회전 0
```

조준 설정은 `config/approach_aim.yaml`을 `/home/wego/launcher_aim_measured.yaml`로 복사해 쓴다. 2026-11-20 결정: x·y·yaw는 줄자로 재지 않고 CAD 값을 장착값으로 쓰고, 높이는 실측값을 쓴다. 이 URDF 값으로 실물 시험을 하므로 실물 설정 파일에서 `mount_calibrated: true`로 둔다(저장소 기본값은 안전을 위해 `false`). 좌우로 빗나가면 y·yaw를, 거리가 틀리면 x를 먼저 의심한다. 예전 `mount_xyz_m`·`mount_yaw_deg` 줄은 더 이상 읽지 않으므로 지운다.

## 4. 카메라 탐지·거리 계산과 경로 미리보기

Wego에서 `robot.launch.xml`과 `navigation.launch.xml`을 켜고 RViz의 **2D Pose Estimate**로 실제 위치를 맞춘다. 실제 표적을 카메라에 보이게 둔다. 주영 인식 경로는 Orbbec RGB `/camera/color/image_raw` → YOLO `/perception/detections` → 깊이 `/camera/depth/image_raw`·`/camera/depth/camera_info`·TF → `/perception/objects`의 `map` 좌표다. 주영 매퍼는 깊이 픽셀 외에 분할 마스크와 바닥 광선 추정도 사용할 수 있으므로, 산출 거리는 현장 줄자값과 대조한다. `mapper_observation_bridge`는 **최근 0.5초 안에 본 `fire`**만 `/fire/observations`로 전달한다. 저장된 물체 지도는 반복 발행되더라도 새 탐지로 취급하지 않는다. 여러 불이 보이면 `target_id:=fire-1`처럼 하나를 명시한다. 개인 저장소의 인식 패키지를 쓰는 경우에는 `yolo.launch.xml`의 기존 `/fire/observations` 출력을 직접 사용한다.

터미널 1에서 실물 로봇 센서와 베이스를, 터미널 2에서 실제 지도와 Nav2를 켠다. 로봇 주변을 비우고 리모컨을 확인한다. 각 명령은 해당 터미널에서 별도로 실행한다.

```bash
source /home/wego/limo_fire_ws/env.sh
export ROS_DOMAIN_ID=0 ROS_LOCALHOST_ONLY=1
ros2 launch limo_fire_bringup robot.launch.xml base:=true viz:=false max_clocks:=false
```

```bash
source /home/wego/limo_fire_ws/env.sh
export ROS_DOMAIN_ID=0 ROS_LOCALHOST_ONLY=1
ros2 launch limo_fire_bringup navigation.launch.xml rviz:=true
```

RViz에서 위치를 맞춘 뒤 `map → base_link` TF와 RGB·깊이 영상의 실제 수신을 확인한다. `map` TF가 없으면 YOLO가 2D 탐지를 하더라도 지도 좌표와 거리 기반 조준은 진행할 수 없다.

다음 명령은 인식·매핑·브리지를 함께 켜되 차체와 서보는 움직이지 않는다. Wego 오버레이에는 주영 인식 패키지가 빌드되어 있어야 한다. `publish_test_target.py`는 소프트웨어 시험 전용이며 실물 시험 중 실행하지 않는다.

```bash
source /home/wego/limo_fire_ws/env.sh
source /home/wego/launcher_aim_friday_20261120/install/local_setup.bash
export ROS_DOMAIN_ID=0 ROS_LOCALHOST_ONLY=1
ros2 launch limo_fire_hardware real_camera_aim.launch.py \
  config:=/home/wego/launcher_aim_measured.yaml motion_enabled:=false serial_enabled:=false
```

다른 터미널에서 RGB·깊이 영상, 탐지 결과, 지도 좌표와 조준 입력을 확인한다. `/fire/observations`에 실제 표적의 `map` 좌표, `position_valid: true`, 신뢰도 0.5 이상이 나와야 한다. 카메라로 본 표적의 방향과 줄자로 잰 대략의 거리를 비교한다. `ros2 topic info -v /fire/observations`에서 발행자가 실제 인식 경로 하나인지 확인한다. 이 조건이 없으면 아래 서비스 호출을 하지 않는다.

```bash
source /home/wego/limo_fire_ws/env.sh
source /home/wego/launcher_aim_friday_20261120/install/local_setup.bash
export ROS_DOMAIN_ID=0 ROS_LOCALHOST_ONLY=1
ros2 topic info -v /fire/observations
ros2 topic echo /perception/detections --once
ros2 topic echo /perception/objects --once
ros2 topic echo /fire/observations --once
```

실물 표적 좌표가 확인된 뒤에만 아래 요청을 실행한다. `PREVIEW_READY`에 후보 위치·차체 yaw·상하각이 나오고, 공·사람·장애물을 지나는 경로가 아니어야 한다. 미리보기 노드를 종료한 뒤 다음 단계로 간다.

```bash
source /home/wego/limo_fire_ws/env.sh
source /home/wego/launcher_aim_friday_20261120/install/local_setup.bash
export ROS_DOMAIN_ID=0 ROS_LOCALHOST_ONLY=1
ros2 service call /launcher/approach_and_aim std_srvs/srv/Trigger '{}'
ros2 topic echo /launcher/approach_aim_status
```

## 5. 무장전 실물 조준 시험

장치 경로와 차동·메카넘 모드를 다시 확인한 뒤 실행한다. 아래의 `serial_port` 전체 경로를 실제 ESP32의 `/dev/serial/by-id/...`로 바꾼다. `yaw_rate_drive_mode_verified:=true`는 모드 `0` 또는 `2`를 확인한 경우에만 준다.

```bash
source /home/wego/limo_fire_ws/env.sh
source /home/wego/launcher_aim_friday_20261120/install/local_setup.bash
export ROS_DOMAIN_ID=0 ROS_LOCALHOST_ONLY=1
ros2 launch limo_fire_hardware real_camera_aim.launch.py \
  config:=/home/wego/launcher_aim_measured.yaml \
  motion_enabled:=true serial_enabled:=true \
  serial_port:=/dev/serial/by-id/<확인한_ESP32_장치명> \
  yaw_alignment_mode:=yaw_rate yaw_rate_drive_mode_verified:=true
```

상태가 `ready`가 된 뒤 `/launcher/approach_and_aim` 서비스를 **한 번** 호출한다. 기대 순서는 `PLANNING → APPROACHING → WAIT_STOP → REACQUIRING → YAW_ALIGNING → WAIT_STOP → REACQUIRING → WAIT_PITCH → PITCH_ACKNOWLEDGED`다. 이미 좌우가 맞으면 `YAW_ALIGNING`에서 0 속도로 바로 넘어갈 수 있다. `PITCH_ACKNOWLEDGED`는 펌웨어 응답이며 실제 각도 측정이 아니다. 이 과정은 발사하지 않는다.

방향이 반대로 돌거나 표적·오도메트리·상태가 사라지면 즉시 취소하고 로봇 정지를 확인한다.

```bash
source /home/wego/limo_fire_ws/env.sh
source /home/wego/launcher_aim_friday_20261120/install/local_setup.bash
export ROS_DOMAIN_ID=0 ROS_LOCALHOST_ONLY=1
ros2 service call /launcher/cancel std_srvs/srv/Trigger '{}'
ros2 topic echo /launcher/approach_aim_status
```

## 6. 현재 검증 범위

2026-11-19에 Wego 임시 복사본에서 `limo_fire_msgs`·`limo_fire_hardware` 두 패키지 빌드와 가짜 Nav2·차체·표적·시리얼을 사용한 **기존 단위·통합 시험 60개**가 통과했다. 금요일용 오버레이에서 새 브리지 시험을 포함한 전체 **63개**도 통과했다. 금요일용 별도 Wego 오버레이에서는 주영 `limo_fire_perception`과 조준 패키지 빌드, 통합 launch 인자 확인, Orbbec RGB·깊이 영상 수신, YOLO 약 30장/초 추론과 인식·매핑·조준 토픽 생성이 확인됐다. 당시 실제 불 표적은 없어 탐지는 0개였고, 센서 전용 구동이라 `map` TF가 없어 지도 좌표를 만들지 못했다. 합성 ROS 물체 메시지를 브리지로 전달하는 시험은 통과했으나 카메라 표적 시험은 아니다. ESP32-S3 소스의 Arduino 코어 3.3.12 컴파일과 Wego 통합 이미지의 SHA-256 대조도 통과했다. 실제 `fire` 표적의 탐지·거리, ESP32 업로드, 상하 모터 각도, 차체 회전과 발사 성공은 확인하지 않았다. 실물 결과를 기록할 때는 사용한 브랜치·커밋, 네 측정값, ESP32 장치 경로, `/limo_status.motion_mode`, 종료 상태, 실제 좌우 오차와 상하각을 함께 남긴다.
