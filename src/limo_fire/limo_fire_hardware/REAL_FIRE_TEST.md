# ESP32-S3 실물 장전·발사 연결과 별도 터미널

이 절차는 GPIO4 레버·GPIO5 상하 서보의 `launcher_test` 펌웨어를 대상으로 한다. 코드는 조준 후 `s` 한 번으로 장전·발사를 요청할 수 있다. 응답 `done`은 펌웨어가 명령을 마쳤다는 뜻이며 공의 비행이나 명중을 측정하지 않는다. ESP32 업로드, 실제 장착 치수, 실제 표적 거리·방향, 서보 움직임과 발사는 아직 이 통합 코드로 시험하지 않았다.

## 공통 준비

1. [실물 조준 절차](REAL_AIM_TEST.md)의 로봇·Nav2·카메라·발사기 장착 치수 확인을 먼저 끝낸다. ESP32-S3 펌웨어를 올리고 GPIO4/5 및 외부 서보 전원·공통 GND를 확인한다.
2. `/dev/serial/by-id/`에서 ESP32 장치를 확인한다. 발사 브리지가 포트를 단독 소유하므로 기존 원시 시리얼 터미널과 동시에 열지 않는다. Wego의 CP2102(`usb-Silicon_Labs_CP2102...`, `/dev/ydlidar`)는 라이다이므로 ESP32 경로로 쓰지 않는다. ESP32를 꽂기 전후 목록을 비교해 새로 생긴 장치만 쓴다.
3. 공의 비행 방향에 사람과 깨질 물건이 없는 시험 공간에서만 `firing_enabled:=true`를 지정한다. 발사 명령은 중간에 취소할 수 없다. 통신 시간 초과 후 자동 재시도하지 않는다.

새 Git 커밋을 받은 Wego 작업공간에서는 먼저 패키지를 빌드하고 새 설치 환경을 불러온다. 기존 시험 오버레이가 먼저 로드되면 옛 실행 파일이 선택될 수 있으므로 `ros2 pkg prefix limo_fire_hardware`로 확인한다.

```bash
cd /home/wego/limo_fire_ws
source /opt/ros/humble/setup.bash
colcon build --packages-select limo_fire_hardware --symlink-install
source install/local_setup.bash
ros2 pkg prefix limo_fire_hardware
```

## 사용자가 별도 터미널에서 장전·발사

첫 터미널에서 로봇 구동 없이 브리지와 조준 노드를 띄운다. 장전·발사 기능은 명시적으로 켜지만 자동 발사는 꺼 둔다.

```bash
source /opt/ros/humble/setup.bash
source /home/wego/limo_fire_ws/install/local_setup.bash
ros2 launch limo_fire_hardware approach_aim.launch.py \
  motion_enabled:=false serial_enabled:=true firing_enabled:=true \
  auto_fire_enabled:=false serial_port:=/dev/serial/by-id/<ESP32_확인한_장치>
```

둘째 터미널을 직접 열거나 GUI에서 아래 명령으로 새 터미널 창을 연다.

```bash
bash /home/wego/limo_fire_ws/tools/hardware_check/open_ros_launcher_terminal.sh
# GUI 창이 없으면 현재 터미널에서:
ros2 run limo_fire_hardware launcher_control_terminal --log /home/wego/launcher-commands.jsonl
```

터미널에서 `c`는 장전, `f`는 앞선 `c`의 응답을 받은 뒤 발사, `s`는 장전·발사 한 번, `?`는 마지막 상태, `r`은 공 재장전 확인(다음 자동 임무 허용), `q`는 종료다. `a`는 아래의 자동 발사 launch가 켜진 경우에만 탐지·이동·조준·발사 임무를 시작한다. 브리지는 `e4` 레버 활성화 응답과 명령별 펌웨어 응답을 확인한다. 서비스의 `accepted`는 완료가 아니므로 터미널에서 `completed`를 확인한다. 연결 오류나 시간 초과가 나면 하드웨어 상태를 확인한 뒤 브리지를 다시 시작한다.

GUI 터미널이 없으면 위의 `ros2 run` 명령을 직접 사용한다. 이미 장전된 상태에서 `c`나 `s`를 다시 보내지 못하도록 브리지가 막는다. 먼저 상태를 확인하고 `f` 또는 종료를 선택한다.

## 탐색·탐지·이동·조준 후 자동 장전·발사

실물 카메라 표적의 지도 좌표와 거리, 장착 치수, Nav2 도착·차체 정지, 좌우 오차와 상하각을 확인한 뒤 실행한다. `a`를 누를 때 최근 표적이 없으면 주영의 프론티어 탐사를 켠다. 지도 좌표가 있는 같은 표적을 서로 다른 영상 시각으로 세 번 확인하면 탐사를 끄고, `idle` 상태와 차체 정지 0.5초를 확인한 뒤 기존 접근·조준·발사 임무를 시작한다. 180초 안에 표적이 없거나 지도가 닫혀 탐사가 끝나면 발사하지 않는다. 탐사 서비스가 없거나 다른 탐사가 이미 진행 중이어도 시작하지 않는다.

Wego의 기존 작업공간에는 `limo_fire_explore`가 설치돼 있지 않아, 2026-11-19에 `/home/wego/launcher_auto_fire_20261119` 시험 오버레이에 프론티어 실행 파일만 빌드했다. 이 빌드는 `-DLIMO_FIRE_FRONTIER_ONLY=ON`을 사용했고, 기존 인식·Nav2 설치본은 바꾸지 않았다. `ros2 pkg prefix limo_fire_explore`가 이 오버레이를 가리키는지 확인한다. 새 저장소 전체를 빌드할 때는 기본 옵션으로 전체 탐사 컴포넌트를 빌드할 수 있다.

표적을 탐색하며 지도를 새로 만들 때는 첫 터미널에서 로봇을, 둘째 터미널에서 Cartographer·Nav2·멈춘 프론티어 탐사 노드를 실행한다. 각 터미널의 ROS 도메인을 맞추고 RViz에서 위치·지도·장애물을 확인한다. 저장된 지도에 미지 영역이 없으면 프론티어 탐사는 곧 `done`이 된다.

```bash
source /home/wego/limo_fire_ws/env.sh
export ROS_DOMAIN_ID=0 ROS_LOCALHOST_ONLY=1
ros2 launch limo_fire_bringup robot.launch.xml base:=true viz:=false max_clocks:=false
```

```bash
source /home/wego/limo_fire_ws/env.sh
source /home/wego/launcher_auto_fire_20261119/install/local_setup.bash
export ROS_DOMAIN_ID=0 ROS_LOCALHOST_ONLY=1
ros2 launch limo_fire_explore frontier_explorer.launch.xml stack:=true rviz:=true
```

셋째 터미널에서 카메라 인식·관측 브리지·조준·발사 브리지를 실행한다. 아래 실측 YAML과 ESP32 장치 경로는 현장에서 확인한 값으로 바꾼다. `fire_on_pitch_ack:=true`는 각도 센서 대신 펌웨어 명령 응답으로 진행하겠다는 명시적 선택이다.

```bash
source /opt/ros/humble/setup.bash
source /home/wego/limo_fire_ws/install/local_setup.bash
source /home/wego/launcher_auto_fire_20261119/install/local_setup.bash
export ROS_DOMAIN_ID=0 ROS_LOCALHOST_ONLY=1
ros2 launch limo_fire_hardware real_camera_aim.launch.py \
  config:=/home/wego/launcher_aim_measured.yaml \
  motion_enabled:=true serial_enabled:=true firing_enabled:=true \
  auto_fire_enabled:=true fire_on_pitch_ack:=true \
  serial_port:=/dev/serial/by-id/<ESP32_확인한_장치>
```

넷째 터미널에서 같은 ROS 환경을 불러온 뒤 `ros2 run limo_fire_hardware launcher_control_terminal`을 실행하고 `a`를 한 번 입력한다. `--search-timeout-s 180`은 기본 탐색 제한 시간이며 `--no-search`는 탐색 없이 현재 보이는 표적만 사용한다. GUI 새 창은 `LIMO_FIRE_AIM_OVERLAY=/home/wego/launcher_auto_fire_20261119 bash /home/wego/launcher_auto_fire_20261119/tools/hardware_check/open_ros_launcher_terminal.sh`로 연다. `a`는 launch의 `auto_fire_enabled:=true`를 켜는 스위치가 아니라 현재 설정된 자동 임무를 한 번 시작하는 명령이다. 직접 `/launcher/approach_and_aim` 서비스를 호출하면 탐색 단계는 거치지 않는다.

`a`를 누르면 터미널은 최소 1.2초, 최대 5초 동안 조준 노드의 새 상태를 기다린다. 받지 못하면 시작하지 않는다. 자동 임무가 발사를 요청했거나 결과를 확인하지 못하면 이후의 `a`는 거부된다. 미리 입력해 둔 `a`도 마찬가지다. 공과 발사기를 확인한 뒤 `r`을 입력해야 다음 `a` 한 번이 허용된다. 탐색 중 Ctrl+C를 누르면 `/explore/frontier_explorer/enable false`를 보낸 뒤 종료한다. 이동·조준 중 Ctrl+C를 누르면 `/launcher/cancel`을 호출한 뒤 종료한다. 두 경우 모두 응답을 받지 못할 수 있으므로 로봇 정지는 현장에서 직접 확인한다.

실제 `/fire/observations`와 `map → base_link` TF를 확인한다. 상태는 `/explore/frontier_explorer/state`, `/launcher/approach_aim_status`, `/launcher/pitch_status`, `/launcher/shot_status`에서 확인한다. 예상 종료 상태 `SHOT_COMMAND_ACKNOWLEDGED`는 펌웨어 응답이며, 공의 비행과 명중은 별도 촬영·관측이 필요하다. 탐색을 중간에 멈추려면 `/explore/frontier_explorer/enable`에 `false`를 보내고 로봇 정지를 확인한다. 터미널에서 Ctrl+C가 발생하면 탐사 정지를 요청하지만, 정지 응답을 받지 못하면 현장에서 직접 상태를 확인한다.

## 검증 기록

**2026-11-20 Wego ROS 시험(가짜 Nav2·차체·표적·발사 서비스, 격리 도메인 `ROS_DOMAIN_ID=91`)**
- 시험용 새 오버레이 `/home/wego/launcher_auto_fire_20261120`에 `limo_fire_hardware`만 빌드했다. `limo_fire_explore`는 11-19 오버레이, `limo_fire_msgs`는 기존 작업공간의 것을 사용했다. 기존 `limo_fire_ws`와 11-19 오버레이는 바꾸지 않았다.
- 최종 코드로 전체 83개 시험을 3회 실행해 3회 모두 통과했다. 자동 발사·탐색 시험군 13개를 5회 더 반복해 모두 통과했다. 로그는 `test_run_4~6.log`, `autofire_repeat_1~5.log`다.
- 확인 항목은 다음과 같다.
  - **실제 탐사 노드 인계:** 설치된 실제 `frontier_explorer` 실행 파일의 `~/enable` 서비스와 `~/state`(idle→running→idle)를 거쳐 1회 발사 요청까지 이어졌다.
  - **탐색 취소·시간 초과:** 탐색 중 Ctrl+C를 누르면 탐사를 끄고 발사 0회였다. 제한 시간 초과에서도 탐사를 끄고 발사 0회였다.
  - **오탐 방지:** 신뢰도 0.3 탐지는 인계되지 않았다. 표적 없이 탐사가 `done`이 되면 발사 0회였다. 이미 다른 탐사가 실행 중이면 인수하지 않았다.
  - **중복 발사 방지:** 미리 입력한 두 번째 `a`는 거부됐다. `r` 뒤의 `a` 한 번만 추가로 허용됐다.
- 수정 전 시험에서 두 가지 시작 시점 문제가 각각 1회씩 나타났다. 하나는 조준 상태를 1.2초 안에 받지 못한 경우, 다른 하나는 탐사 상태를 1초 안에 받지 못한 경우였다. 두 경우 모두 발사 없이 임무를 거부했다. 대기를 "받을 때까지 최대 5초"로 바꾼 뒤에는 위 반복 시험에서 다시 나타나지 않았다.
- 이 시험은 실물 시험이 아니다. 실제 카메라 표적, Nav2 주행, 서보, ESP32 장전·발사는 포함되지 않는다.

**2026-11-20 실물 준비 점검(읽기 전용)**
- CP2102 `ttyUSB0`은 라이다(`/dev/ydlidar`)다. ESP32-S3는 연결돼 있지 않았다.
- `/home/wego/launcher_aim_measured.yaml`은 아직 없다. 장착 x/y/회전축 높이/방향은 측정되지 않았다.
- 사전 빌드 펌웨어 이미지의 SHA-256은 문서의 기대값과 일치한다.
- 로봇·Nav2·카메라 스택은 꺼져 있었다. 실제 표적의 map 좌표와 거리도 아직 확인하지 않았다.
- 따라서 실물의 탐색·이동·조준·장전·발사 연속 실행은 아직 하지 않았다.

`fire_on_pitch_ack:=true`는 현재 각도 센서가 없으므로 펌웨어의 상하각 **명령 응답**을 실제 각도 대신 허용하는 명시적 선택이다. 실제 각도 센서를 연결하면 `require_pitch_feedback: true`를 설정하고 `fire_on_pitch_ack:=false`로 시험한다. 기본값은 `firing_enabled:=false`, `auto_fire_enabled:=false`, `fire_on_pitch_ack:=false`다. 상하 피드백이 없고 `fire_on_pitch_ack`도 꺼져 있으면 자동 발사는 시작하지 않는다.
