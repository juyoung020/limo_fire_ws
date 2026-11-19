# 실물 LIMO 거리 확보와 좌우·상하 조준

표적 탐지 → 사거리 후보 검사 → Nav2 접근 → 차체 좌우 정렬 → 정지·재탐지 → 상하 각도 명령을 한 번씩 실행한다.

금요일 실물 치수 입력과 무장전 시험 순서는 [REAL_AIM_TEST.md](REAL_AIM_TEST.md)를 따른다.

발사 결과 기록과 오프라인 보정 후보 학습은 [AIM_LEARNING_PREP.md](AIM_LEARNING_PREP.md)를 따른다. 현재 조준 실행에는 학습 모델이 연결되지 않는다.

ESP32-S3의 장전·발사 브리지, 사용자가 조작하는 별도 터미널, 조준 후 자동 한 발 요청은 [REAL_FIRE_TEST.md](REAL_FIRE_TEST.md)를 따른다. 모두 기본 비활성화이며 펌웨어 `done`은 실제 탄착 측정이 아니다.

## 확인한 발사장치 구성

2026-11-18에는 Arduino Uno로 분리 모터 진단을 했다. 이번 통합 시험의 대상은 **ESP32-S3 GPIO4 레버·GPIO5 상하**다. 상하 모터는 사용자 제공 DSSERVO 20kg, 270° 위치형 가정이며 실물 각도·영점은 아직 측정되지 않았다.

| 역할 | 설계 모터 | Arduino Uno | ESP32-S3 | ESP32-C3 |
|---|---|---|---|---|
| 좌우 조준 | LIMO 차체 회전 | Nav2 | Nav2/요레이트 | Nav2/요레이트 |
| 상하 조준 | DSSERVO 20kg, 270° 가정 | D5 | GPIO5 | GPIO3 |
| 장전·발사 레버 | DS3218 PRO | D4 | GPIO4 | GPIO4 |

11월 16일 실물 기록에는 레버 자리에 MG996R을 사용한 시험도 있다. GPIO4 시험 스케치와 발사대 전용 펌웨어의 명령이 겹치므로, 브리지는 `launcher_test ready` 배너 또는 전용 상태 응답을 확인한 뒤 각도 명령을 보낸다. 이번 ESP32 펌웨어는 부팅 시 두 PWM을 끄고 GPIO를 LOW로 둔다. USB 포트를 열면 보드가 재시작될 수 있다.

원본은 [발사대 펌웨어](https://github.com/juyoung020/limo_fire_ws/blob/main/hardware/modeling/탁구공발사대/firmware/launcher_test/src/main.cpp)와 [노션 모델링 문서](https://www.notion.so/415454b08de2814f9851c1b030da46c7)다. 이 패키지의 브리지는 상태 조회 `?\n`, 새 ESP32 펌웨어의 상하 활성화 `e5\n`, 각도 명령 `a<각도>\n`를 사용한다.

수정된 ESP32 소스는 [launcher_test/src/main.cpp](../../../hardware/modeling/탁구공발사대/firmware/launcher_test/src/main.cpp)다. 통신 속도는 115200 baud다. 브리지는 Uno 상태 응답도 식별할 수 있지만, 이 브랜치의 금요일 실물 시험은 ESP32-S3를 대상으로 한다. `READY`만으로는 장치를 신뢰하지 않으며, 확인 후 보드가 재시작되면 새 각도 명령을 차단한다.

이 패키지의 `pitch_serial_bridge`는 명시적으로 `firing_enabled`를 켰을 때만 ESP32-S3 장전·발사 명령 `c`/`f`/`s`를 보낸다. `launcher_control_terminal`은 같은 ROS 브리지를 사용한다. `auto_fire_enabled`를 따로 켜면 조준 완료 후 `s`를 한 번 요청한다. 파일 배치와 소프트웨어 시험은 실제 보드 업로드·서보 동작·발사 검증을 대신하지 않는다.

## 빌드

Wego의 `/home/wego/limo_fire_ws`에서 실행한다.

```bash
source /home/wego/limo_fire_ws/env.sh
cd /home/wego/limo_fire_ws
colcon build --packages-select limo_fire_hardware --symlink-install
source install/local_setup.bash
```

ROS 의존성은 `package.xml`에 있다. `import serial`이 실패하면 Ubuntu의 `python3-serial` 패키지를 설치한다.

## 준비할 실측값

`config/approach_aim.yaml`을 사용자 설정 파일로 복사해 다음 값을 채운다.

- `mount_xyz_m`은 `base_link` 기준 상하 회전축 중심의 위치다. 단위는 m다.
- `mount_yaw_deg`는 차체 전방에서 발사기 전방이 수평으로 돌아간 각도다.
- 실제 장착 방향, 서보 영점과 회전 부호는 발사대 펌웨어의 `kTiltServoAt15`, `kTiltSign`에 맞춘다.
- `desired_distance_m`은 상하 회전축에서 표적까지의 수평 거리다. 기본 1.2m는 위치 제어 예시다. 실물 사거리 표를 측정해 설정한다.
- `mount_calibrated`는 실측 설정을 채운 후 `true`로 둔다.
- `robot_radius_m + footprint_margin_m`은 실제 차체 외곽을 포함해야 한다.

기하 계산은 평면 주행을 가정한다. TF의 차체 roll/pitch가 `max_base_tilt_deg`를 넘으면 중단한다. 초기 구동·마찰·가감속 설정은 기존 실물 Nav2 설정을 사용한다.

## 센서와 Nav2

각 터미널에서 환경을 연다. Wego 대시보드와 같은 장치를 사용하는 launch가 이미 켜져 있으면 해당 스택을 재사용하거나 종료 후 이 구성을 시작한다.

터미널 1에서 센서와 베이스를 켠다.

```bash
source /home/wego/limo_fire_ws/env.sh
export ROS_DOMAIN_ID=0 ROS_LOCALHOST_ONLY=1
ros2 launch limo_fire_bringup robot.launch.xml base:=true camera_pitch:=0.1152 viz:=false
```

터미널 2에서 현재 공간에 맞는 지도와 Nav2를 켠다. 기본 지도는 저장된 실험실 지도다. RViz의 `2D Pose Estimate`로 현재 위치를 맞춘다.

```bash
source /home/wego/limo_fire_ws/env.sh
export ROS_DOMAIN_ID=0 ROS_LOCALHOST_ONLY=1
ros2 launch limo_fire_bringup navigation.launch.xml rviz:=true
```

주영 인식 패키지의 Orbbec RGB·깊이 입력과 YOLO·물체 지도를 실물 표적 소스로 쓴다. `/perception/objects`에는 저장된 물체도 주기적으로 나오므로 `mapper_observation_bridge`가 마지막 관측 나이·신뢰도·표적 수를 검사한 뒤 `/fire/observations`로 보낸다. 실물 시험에서 `publish_test_target.py`를 실행하지 않는다. 센서·거리 확인 및 2026-11-20 실행 순서는 [REAL_AIM_TEST.md](REAL_AIM_TEST.md)를 따른다.

## 후보와 경로 미리보기

터미널 3에서 카메라 인식·매핑·조준을 묶은 launch를 실행한다. 기본값은 `motion_enabled=false`, `serial_enabled=false`다.

```bash
source /home/wego/limo_fire_ws/env.sh
source /home/wego/launcher_aim_friday_20261120/install/local_setup.bash
export ROS_DOMAIN_ID=0 ROS_LOCALHOST_ONLY=1
ros2 launch limo_fire_hardware real_camera_aim.launch.py
```

다른 터미널에서 한 번의 계산을 요청한다.

```bash
source /home/wego/limo_fire_ws/env.sh
source /home/wego/launcher_aim_friday_20261120/install/local_setup.bash
export ROS_DOMAIN_ID=0 ROS_LOCALHOST_ONLY=1
ros2 service call /launcher/approach_and_aim std_srvs/srv/Trigger '{}'
ros2 topic echo /launcher/approach_aim_status
```

미리보기는 실제 costmap과 Nav2 경로 계획을 이용한다. `PREVIEW_READY` 메시지에 후보 위치, 차체 yaw와 상하 각도를 표시한다. `mount_calibrated=false`라면 후보도 임시 영점 기준이다.

후보는 표적 주변의 각도 순서로 검사한다. 차체 외곽을 포함하는 정사각 영역, 지도상의 표적 시야와 각도 한계를 검사하고, Nav2가 요청 위치까지 경로를 만들 수 있는 첫 후보를 사용한다. 마지막 표적 근처 `sight_target_margin_m` 구간은 과녁 자체가 점유된 경우를 위해 시야 검사에서 제외한다. 높이에 따른 실제 가림 여부는 재탐지 단계에서 확인한다.

## 실제 접근과 상하 명령

실측한 사용자 설정 파일과 보드 장치 경로를 지정한다. `/dev/serial/by-id/`에서 확인한 전체 경로를 사용한다. 아래 꺾쇠 자리 두 곳을 실제 경로로 바꾼다.

```bash
source /home/wego/limo_fire_ws/env.sh
source /home/wego/launcher_aim_friday_20261120/install/local_setup.bash
export ROS_DOMAIN_ID=0 ROS_LOCALHOST_ONLY=1
ros2 launch limo_fire_hardware real_camera_aim.launch.py \
  config:=<실측한_설정파일의_절대경로> \
  motion_enabled:=true serial_enabled:=true \
  serial_port:=<확인한_보드의_전체_장치경로>
```

실제 `/fire/observations`의 `map` 좌표가 카메라 표적과 줄자 거리와 맞고 브리지 상태가 `ready`가 된 뒤 `/launcher/approach_and_aim` 서비스를 호출한다. 무장전 시험에서 공과 고무줄 하중을 제거해 상하 구동 및 혼 방향을 먼저 확인한다.

ESP32-S3의 수정된 `hardware/modeling/탁구공발사대/firmware/launcher_test/src/main.cpp`를 쓰면 브리지는 펌웨어 신원을 확인하고 `e5`를 보내 상하 GPIO5 PWM의 `tilt PWM ON` 응답을 받은 뒤에만 `ready`가 된다. 따라서 이후 `a<각도>`가 거부되지 않는다. 이 단계에서 1500µs 신호로 모터가 움직일 수 있다. 기본 `firing_enabled=false`에서는 장전·발사 레버 GPIO4를 활성화하지 않으며 `s`·`c`·`f` 명령도 보내지 않는다. 서보 영점과 회전 부호, 발사기 장착 치수는 실물에서 아직 확인해야 한다.

실행 상태는 `PLANNING → APPROACHING → WAIT_STOP → REACQUIRING → ALIGNING → WAIT_STOP → REACQUIRING → WAIT_PITCH` 순서다. 이동 후 최신 표적과 거리·정지를 다시 확인하고, 차체 방향 오차가 허용 범위일 때 상하 명령을 보낸다. 불 검출은 `FireObservation`의 `position_valid`, confidence와 시각을 확인한다. 진행 중 다른 표적 ID는 채택하지 않고, 선택 표적의 누적 위치 변화가 허용치를 넘으면 취소한다.

좌우를 차체의 **요레이트**로 미세 조준하려면, 먼저 `ros2 topic echo /limo_status --once`에서 `motion_mode`가 차동 `0` 또는 메카넘 `2`인지 확인한다. Ackermann `1`은 제자리 회전을 지원하지 않는다. 확인한 뒤 같은 launch에 `yaw_alignment_mode:=yaw_rate yaw_rate_drive_mode_verified:=true`를 추가한다. 접근 Nav2 목표가 끝나고 로봇이 정지하면, 최신 표적 좌표와 차체 TF에서 좌우 오차를 다시 계산해 `/cmd_vel`의 `angular.z`를 최대 0.25 rad/s로 제한하여 보낸다. 3° 이내에 들면 0 속도를 반복 전송하고 정지·재탐지를 확인한 다음 상하각 명령을 보낸다. 회전 중에도 최신 `/limo_status`가 0 또는 2인지 확인한다. 표적·오도메트리·차체 상태·costmap이 오래되거나, 오차가 45°를 넘거나, 8초 안에 정렬하지 못하면 0 속도를 보내고 중단한다. 기본 `nav2` 모드에서는 기존 두 번째 자세 목표를 그대로 사용한다.

이 분기는 주영 저장소 `src/limo_fire/limo_fire_base/src/limo_driver.cpp`의 `/cmd_vel` 처리에 맞춘다. 차동·메카넘 모드에서는 `angular.z`가 차체에 전달되지만 Ackermann 모드의 회전 명령은 조향각 계산으로 들어가므로 제자리 좌우 조준에 사용하지 않는다. 주영 저장소의 `AimNozzle`은 별도의 물 노즐 팬·틸트 기능이므로 공 발사기 조준에 연결하지 않았다.

`PITCH_ACKNOWLEDGED`는 펌웨어가 명령 각도를 회신했다는 뜻이며 `measured=false`로 기록한다. 현재 270° 서보도 이 프로토콜로 실제 각도를 읽지 않는다. 별도 각도 센서가 있을 때 `require_pitch_feedback=true`로 설정하고 `/launcher/joint_states`에 `launcher_pitch_joint`의 실측 rad 값을 발행한다. 명령 이후의 최신 피드백이 허용 오차 안에 유지돼야 `AIM_VERIFIED`가 된다. 목표 각도를 그대로 복사한 joint state는 이 입력으로 사용하지 않는다.

취소는 다음 서비스를 사용한다. 이 노드가 보낸 Nav2 목표의 취소와 정지 odom을 확인한다. `STOP_UNCONFIRMED` 상태는 정지 확인이 필요한 상태이며, 확인될 때까지 새 요청을 받지 않는다.

```bash
ros2 service call /launcher/cancel std_srvs/srv/Trigger '{}'
```

## 발사각 모델

2026-11-18 사용자가 실물 공의 발사 직후 속도를 약 **3.9m/s(14.04km/h)**로 확인했고, 현재 보정값으로 사용하기로 했다. 기본 launch 설정을 `aiming_model=ballistic`, `measured_launch_speed_mps=3.9`로 반영했다. 사용자는 공이 옆으로 약간 휜다고 보고했으며 횡방향 보정량은 아직 측정하지 않았다. 이 값은 사용자 제공 실측값이고 이번 작업에서 재측정하지 않았다.

공별 발사 직후 속도는 다음과 같이 기록한다. 두 값 모두 사용자가 제공한 근삿값이다.

| 공 | 속도 | 시속 환산 | 관측·적용 상태 |
|---|---|---|---|
| 파란 공 | 약 3.9m/s | 약 14.04km/h | 현재 기본 보정값. 옆으로 약간 휘며 보정량 미측정 |
| 노란 공 | 약 4.2m/s | 약 15.12km/h | 추가 기록값. 공별 보정에 사용할 때 속도 파라미터를 4.2로 설정 |

노란 공의 횡방향 편차와 반복 발사 속도 편차는 아직 확인되지 않았다. 공을 바꿀 때는 해당 공의 속도로 각도를 다시 계산한다.

탄도 모델은 상하 회전축을 발사 출발점으로 근사해 중력 아래 저각 해를 계산한다. 회전축과 실제 총구의 위치 차이, 탁구공 공기저항, 옆 휘어짐과 고무줄 반복 편차는 실측 탄착 오차로 보정해야 한다. `line_of_sight`를 선택하면 표적을 향하는 기하학적 상하 각도를 계산한다. 선택한 모델의 계산 각도가 5–30° 범위를 벗어나거나 도달 불가능하면 해당 후보를 채택하지 않는다.

## 소프트웨어 검증

ROS가 없는 PC에서는 순수 계산과 프로토콜 테스트를 실행할 수 있다.

```bash
cd /home/wego/limo_fire_ws/src/limo_fire/limo_fire_hardware
python3 -m unittest discover -s test -p 'test_aim_geometry.py' -v
python3 -m unittest discover -s test -p 'test_pitch_protocol.py' -v
```

ROS 통합 fixture는 별도 도메인에서 가상 Nav2·센서·펌웨어 응답을 사용한다. 실제 로봇 성능은 실물 시험 로그로 기록한다.

```bash
source /home/wego/limo_fire_ws/env.sh
source /home/wego/limo_fire_ws/install/local_setup.bash
cd /home/wego/limo_fire_ws/src/limo_fire/limo_fire_hardware
ROS_DOMAIN_ID=194 ROS_LOCALHOST_ONLY=1 python3 -m unittest discover -s test -p 'test_approach_aim_ros.py' -v
```

시리얼 브리지는 Linux 가상 터미널 포트로 실제 송수신 코드를 검사한다.

```bash
ROS_DOMAIN_ID=195 ROS_LOCALHOST_ONLY=1 python3 -m unittest discover -s test -p 'test_pitch_bridge_ros.py' -v
```

2026-11-18 기준 순수 계산·프로토콜 시험 34개, Wego ROS 통합 fixture 9개와 가상 시리얼 fixture 3개가 통과했다. 실물 모터의 각도 정확도와 실제 접근 위치 오차는 실측 시험 항목으로 남아 있다.

2026-11-19 변경분은 Wego의 `/tmp/codex_launcher_aim_20261119` 임시 복사본에서 `limo_fire_hardware` 패키지 빌드, ROS 접근·조준 fixture **13개**, 가상 시리얼 fixture **6개**가 통과했다. Windows의 순수 계산 시험 15개와 프로토콜 시험 13개도 통과했다. ROS 시험에서는 가짜 Nav2·표적·차체·ESP32 응답을 사용해 12° 좌우 오차의 회전, 정지 후 상하 명령, 회전 중 취소와 주행 모드 변경 시 정지를 확인했다. 실제 Wego 차체를 움직이거나 ESP32 모터를 구동한 결과는 아니다.
