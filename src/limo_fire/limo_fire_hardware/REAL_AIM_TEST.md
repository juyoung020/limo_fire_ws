# 실물 LIMO 발사기 좌우·상하 조준 시험

2026-11-20 측정값 입력을 위한 실행 절차. **장전·발사 명령은 이 절차에 없다.** 처음에는 공과 고무줄 하중을 제거하고 회전 범위를 확보한다. ESP32-S3에는 수정된 `launcher_test/src/main.cpp`를 올려야 한다. 기존 Uno 펌웨어나 주영 저장소 원본 ESP32 펌웨어를 이 설정과 혼동하지 않는다.

## 1. 기록할 실측값

`base_link`는 현재 LIMO 모델에서 바닥 위 14.5cm이며 앞·뒤 바퀴 축의 중간, 좌우 중심선에 있다. 실제 로봇/TF와 다르면 모델을 먼저 바로잡거나 아래 `--base-link-height-cm`를 실측값으로 지정한다. 회전축은 상하 서보가 발사기를 돌리는 **물리적 축 중심**이다.

| 값 | 측정 위치·부호 | 실측 기록 |
|---|---|---|
| `x_cm` | 바퀴 앞뒤 축 중간 → 회전축, 전방 + | 미측정 |
| `y_cm` | 차체 좌우 중심선 → 회전축, 로봇 왼쪽 + | 미측정 |
| `pivot_height_cm` | 바닥 → 회전축 중심 | 미측정 |
| `mount_yaw_deg` | 차체 전방(+x) → 발사기 수평 방향, 왼쪽 회전 + | 미측정 |
| 총구 출구 중심 높이·회전축과의 거리 | 탄도 오차 기록용; 현재 계산에는 미반영 | 미측정 |

발사 직후 속도는 파란 공 약 3.9m/s, 노란 공 약 4.2m/s의 사용자 제공 근삿값이다. 현재 기본 각도 계산은 파란 공 3.9m/s를 사용한다. 옆으로 휘는 양과 실제 발사 위치 차이는 아직 보정되지 않았다.

## 2. 보드와 로봇 확인

ESP32-S3 펌웨어의 GPIO4는 레버, GPIO5는 270° 상하 서보다. 개인 저장소 배치는 `hardware/launcher_test`, 주영 저장소 배치는 `hardware/modeling/탁구공발사대/firmware/launcher_test`가 소스 폴더다. 11월 19일 노트북의 Arduino ESP32 코어 3.3.12로 S3 소스를 **컴파일**했고 Wego에 4MB 통합 이미지를 복사했다. Wego의 PlatformIO 보드 도구 다운로드는 디스크 여유가 줄어 중단하고 방금 받은 캐시를 정리했다. 따라서 금요일에는 아래의 사전 빌드 이미지를 사용한다. 업로드와 실물 서보 반응은 아직 검증되지 않았다. 서보 외부 전원 GND와 ESP32 GND를 공통으로 연결한다. 브리지는 `e5`만 보내며 GPIO4 레버와 발사 명령은 건드리지 않는다. `e5` 순간 1500µs로 모터가 움직일 수 있다.

ESP32-S3 보드가 연결된 뒤 **장치 경로가 이 보드인지 확인하고**, 다음 명령으로 이미지를 업로드한다. 현재 보이는 CP2102 장치는 ESP32로 확인되지 않았으므로 사용하지 않는다.

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

## 3. 측정값으로 설정 파일 만들기

다음 네 숫자를 실제 측정값으로 바꾼다. 이 명령은 새 파일만 만들고 로봇을 움직이지 않는다. 출력은 `motion_enabled: false`로 시작한다.

```bash
cd /home/wego/launcher_aim_friday_20261120
python3 src/limo_fire/limo_fire_hardware/tools/write_mount_config.py \
  --x-cm <앞쪽_플러스_cm> --y-cm <왼쪽_플러스_cm> \
  --pivot-height-cm <바닥에서_회전축_cm> --mount-yaw-deg <왼쪽_플러스_deg> \
  --measurements-confirmed --output /home/wego/launcher_aim_measured.yaml
```

위 꺾쇠 자리는 실제 숫자로 교체한다. 도구는 `z = (pivot_height_cm - 14.5) / 100`을 계산하고 `mount_xyz_m`, `mount_yaw_deg`, `mount_calibrated`를 채운다. 출력 파일을 열어 부호와 단위를 재확인한다. 시험 오버레이의 `limo_fire_msgs`와 `limo_fire_hardware`는 11월 19일 빌드했다. Wego를 바꾸거나 코드를 새로 받았다면 기존 작업공간을 source한 뒤 시험 오버레이에서 `colcon build --packages-select limo_fire_msgs limo_fire_hardware --symlink-install`로 다시 빌드한다.

## 4. 표적 좌표와 경로 미리보기

`robot.launch.xml`과 `navigation.launch.xml`을 켜고 RViz의 **2D Pose Estimate**로 실제 위치를 맞춘다. 개인 저장소의 감지·매핑 노드를 사용할 때는 `/fire/observations`에 최신 `map` 좌표와 `position_valid=true`가 나오는지 확인한다. 주영 저장소의 이번 브랜치만 사용할 때는 자동 표적 좌표 발행기가 없으므로, 지도에서 위치를 측정한 **고정 시험 표식**을 다음 도구로 10Hz 발행할 수 있다. 이 도구는 카메라 탐지 결과가 아니다. 자동 탐지 노드와 동시에 실행하지 않는다.

```bash
python3 /home/wego/launcher_aim_friday_20261120/src/limo_fire/limo_fire_hardware/tools/publish_test_target.py \
  --x <지도_x_m> --y <지도_y_m> --z <표식_높이_m>
```

새 터미널에서 먼저 구동 없이 경로만 계산한다.

```bash
source /home/wego/limo_fire_ws/env.sh
source /home/wego/launcher_aim_friday_20261120/install/local_setup.bash
export ROS_DOMAIN_ID=0 ROS_LOCALHOST_ONLY=1
ros2 launch limo_fire_hardware approach_aim.launch.py \
  config:=/home/wego/launcher_aim_measured.yaml motion_enabled:=false serial_enabled:=false
```

또 다른 터미널에서 아래 요청과 상태 확인을 실행한다. `PREVIEW_READY`에 후보 위치·차체 yaw·상하각이 나오고, 공·사람·장애물을 지나는 경로가 아니어야 한다. 미리보기 노드를 종료한 뒤 다음 단계로 간다.

```bash
ros2 service call /launcher/approach_and_aim std_srvs/srv/Trigger '{}'
ros2 topic echo /launcher/approach_aim_status
```

## 5. 무장전 실물 조준 시험

장치 경로와 차동·메카넘 모드를 다시 확인한 뒤 실행한다. 아래의 `serial_port` 전체 경로를 실제 ESP32의 `/dev/serial/by-id/...`로 바꾼다. `yaw_rate_drive_mode_verified:=true`는 모드 `0` 또는 `2`를 확인한 경우에만 준다.

```bash
source /home/wego/limo_fire_ws/env.sh
source /home/wego/launcher_aim_friday_20261120/install/local_setup.bash
export ROS_DOMAIN_ID=0 ROS_LOCALHOST_ONLY=1
ros2 launch limo_fire_hardware approach_aim.launch.py \
  config:=/home/wego/launcher_aim_measured.yaml \
  motion_enabled:=true serial_enabled:=true \
  serial_port:=/dev/serial/by-id/<확인한_ESP32_장치명> \
  yaw_alignment_mode:=yaw_rate yaw_rate_drive_mode_verified:=true
```

상태가 `ready`가 된 뒤 `/launcher/approach_and_aim` 서비스를 **한 번** 호출한다. 기대 순서는 `PLANNING → APPROACHING → WAIT_STOP → REACQUIRING → YAW_ALIGNING → WAIT_STOP → REACQUIRING → WAIT_PITCH → PITCH_ACKNOWLEDGED`다. 이미 좌우가 맞으면 `YAW_ALIGNING`에서 0 속도로 바로 넘어갈 수 있다. `PITCH_ACKNOWLEDGED`는 펌웨어 응답이며 실제 각도 측정이 아니다. 이 과정은 발사하지 않는다.

방향이 반대로 돌거나 표적·오도메트리·상태가 사라지면 즉시 취소하고 로봇 정지를 확인한다.

```bash
ros2 service call /launcher/cancel std_srvs/srv/Trigger '{}'
ros2 topic echo /launcher/approach_aim_status
```

## 6. 현재 검증 범위

2026-11-19에 Wego 임시 복사본에서 `limo_fire_msgs`·`limo_fire_hardware` 두 패키지 빌드와 가짜 Nav2·차체·표적·시리얼을 사용한 **전체 단위·통합 시험 60개**가 통과했다. 금요일용 별도 Wego 오버레이에서도 두 패키지 빌드와 `ros2 pkg prefix limo_fire_hardware` 경로 확인이 통과했다. ESP32-S3 소스의 Arduino 코어 3.3.12 컴파일도 통과했고 통합 이미지의 SHA-256을 Wego에서 대조했다. 실제 ESP32 업로드, 상하 모터 각도, 차체 회전, 표적 탐지와 발사 성공은 확인하지 않았다. 실물 결과를 기록할 때는 사용한 브랜치·커밋, 네 측정값, ESP32 장치 경로, `/limo_status.motion_mode`, 종료 상태, 실제 좌우 오차와 상하각을 함께 남긴다.
