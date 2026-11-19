# ESP32-S3 실물 장전·발사 연결과 별도 터미널

이 절차는 GPIO4 레버·GPIO5 상하 서보의 `launcher_test` 펌웨어를 대상으로 한다. 코드는 조준 후 `s` 한 번으로 장전·발사를 요청할 수 있다. 응답 `done`은 펌웨어가 명령을 마쳤다는 뜻이며 공의 비행이나 명중을 측정하지 않는다. ESP32 업로드, 실제 장착 치수, 실제 표적 거리·방향, 서보 움직임과 발사는 아직 이 통합 코드로 시험하지 않았다.

## 공통 준비

1. [실물 조준 절차](REAL_AIM_TEST.md)의 로봇·Nav2·카메라·발사기 장착 치수 확인을 먼저 끝낸다. ESP32-S3 펌웨어를 올리고 GPIO4/5 및 외부 서보 전원·공통 GND를 확인한다.
2. `/dev/serial/by-id/`에서 ESP32 장치를 확인한다. 발사 브리지가 포트를 단독 소유하므로 기존 원시 시리얼 터미널과 동시에 열지 않는다.
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

터미널에서 `c`는 장전, `f`는 앞선 `c`의 응답을 받은 뒤 발사, `s`는 장전·발사 한 번, `?`는 마지막 상태, `q`는 종료다. 브리지는 `e4` 레버 활성화 응답과 명령별 펌웨어 응답을 확인한다. 서비스의 `accepted`는 완료가 아니므로 터미널에서 `completed`를 확인한다. 연결 오류나 시간 초과가 나면 하드웨어 상태를 확인한 뒤 브리지를 다시 시작한다.

GUI 터미널이 없으면 위의 `ros2 run` 명령을 직접 사용한다. 이미 장전된 상태에서 `c`나 `s`를 다시 보내지 못하도록 브리지가 막는다. 먼저 상태를 확인하고 `f` 또는 종료를 선택한다.

## 탐지·이동·조준 후 자동 장전·발사

실물 카메라 표적의 지도 좌표와 거리, 장착 치수, Nav2 도착·차체 정지, 좌우 오차와 상하각을 확인한 뒤 실행한다. 별도 `launcher_control_terminal`은 열지 않는다.

```bash
source /opt/ros/humble/setup.bash
source /home/wego/limo_fire_ws/install/local_setup.bash
ros2 launch limo_fire_hardware real_camera_aim.launch.py \
  config:=/home/wego/launcher_aim_measured.yaml \
  motion_enabled:=true serial_enabled:=true firing_enabled:=true \
  auto_fire_enabled:=true fire_on_pitch_ack:=true \
  serial_port:=/dev/serial/by-id/<ESP32_확인한_장치>
```

다른 터미널에서 실제 `/fire/observations`와 `map → base_link` TF를 확인한 다음 `ros2 service call /launcher/approach_and_aim std_srvs/srv/Trigger '{}'`를 한 번 실행한다. 상태는 `/launcher/approach_aim_status`, `/launcher/pitch_status`, `/launcher/shot_status`에서 확인한다. 예상 종료 상태는 `SHOT_COMMAND_ACKNOWLEDGED`다. 공의 비행과 명중은 별도 촬영·관측이 필요하다.

`fire_on_pitch_ack:=true`는 현재 각도 센서가 없으므로 펌웨어의 상하각 **명령 응답**을 실제 각도 대신 허용하는 명시적 선택이다. 실제 각도 센서를 연결하면 `require_pitch_feedback: true`를 설정하고 `fire_on_pitch_ack:=false`로 시험한다. 기본값은 `firing_enabled:=false`, `auto_fire_enabled:=false`, `fire_on_pitch_ack:=false`다. 상하 피드백이 없고 `fire_on_pitch_ack`도 꺼져 있으면 자동 발사는 시작하지 않는다.
