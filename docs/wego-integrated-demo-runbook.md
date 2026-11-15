# Wego 통합 시뮬레이션 실행 안내

Webots LIMO 이동, Nav2, YOLO/물체 지도, FireManager, MuJoCo 발사 시뮬레이션을 함께 실행하는 절차다. Wego Ubuntu + ROS 2 Humble에서 실행한다. 이 안내는 기능 브랜치 `codex/wego-ackermann-printer-world` 기준이다.

## 1. GUI 통합 실행

Wego Ubuntu에 접속하고 환경 선택 메뉴에서 Global 환경을 선택한다. 이전 domain 63 / Webots 포트 1256 실행이 있으면 먼저 종료한다. 첫 터미널에서 다음 준비 명령을 실행한다:

```bash
source /opt/ros/humble/setup.bash
source /home/wego/limo_fire_ws/install/setup.bash
source /home/wego/limo_fire_ws_v5/install/setup.bash
export AMENT_PREFIX_PATH=/home/wego/limo_fire_ws_v5/install/limo_fire_bt:/home/wego/limo_fire_ws_v5/install/limo_fire_bridge:/home/wego/limo_fire_ws_v5/install/limo_fire_msgs:$AMENT_PREFIX_PATH
export ROS_DOMAIN_ID=63
export ROS_LOCALHOST_ONLY=1
export DISPLAY=:1
export XAUTHORITY=/run/user/1000/gdm/Xauthority
export WEBOTS_HOME=/home/wego/webots_build/webots

ros2 launch limo_fire_bt limo_fire_printer_fire.launch.xml \
  port:=1256 \
  world_dir:=/home/wego/limo_fire_ws_v5/sim/webots/worlds \
  map:=/home/wego/limo_fire_ws_v5/src/limo_fire/limo_fire_navigation/maps/sim/limo_fire_printer_obstacles.yaml \
  yolo_engine_path:=/home/wego/limo_fire_ws/src/limo_fire/limo_fire_perception/models/yolo26s_seg_fire_u8_480.engine \
  mujoco_root:=/home/wego/limo_fire_ws_v5/sim/mujoco_ball_mvp \
  mujoco_results_dir:=/home/wego/limo_fire_ws_v5/sim/mujoco_ball_mvp/experiments/wego-nav-rerun \
  rviz:=false
```

YOLO TensorRT 엔진 경로는 검증된 실행에서 사용한 기존 공유 작업공간 파일이다. Webots와 MuJoCo GUI를 띄우는 동안 첫 터미널을 켜 둔다. `DISPLAY`, `XAUTHORITY`, `WEBOTS_HOME`은 GUI launch용이므로 첫 터미널에 설정한다.

## 2. Nav2 목표 보내기

Webots와 MuJoCo GUI가 뜨면 Wego에 두 번째 터미널로 접속한다. 별도의 셸이므로 액션 명령을 입력할 터미널에도 아래 준비 명령을 다시 실행한다.

```bash
source /opt/ros/humble/setup.bash
source /home/wego/limo_fire_ws/install/setup.bash
source /home/wego/limo_fire_ws_v5/install/setup.bash
export AMENT_PREFIX_PATH=/home/wego/limo_fire_ws_v5/install/limo_fire_bt:/home/wego/limo_fire_ws_v5/install/limo_fire_bridge:/home/wego/limo_fire_ws_v5/install/limo_fire_msgs:$AMENT_PREFIX_PATH
export ROS_DOMAIN_ID=63
export ROS_LOCALHOST_ONLY=1
```

그 다음 액션 서버를 확인한다.

```bash
ros2 action list -t | grep mission_navigate
```

다음 항목이 보여야 한다.

```text
/fire/mission_navigate_to_pose [nav2_msgs/action/NavigateToPose]
```

아래는 시연에서 사용한 수동 경유·접근 목표다. 표적 좌표를 접근 목표로 자동 변환하는 기능은 아직 연결되지 않았다. 목표를 한 번에 하나씩 보내고, 매번 `SUCCEEDED` 확인 후 다음으로 진행한다.

```bash
ros2 action send_goal /fire/mission_navigate_to_pose nav2_msgs/action/NavigateToPose "{pose: {header: {frame_id: map}, pose: {position: {x: 2.8, y: 3.2, z: 0.0}, orientation: {z: 0.0, w: 1.0}}}}"
```

```bash
ros2 action send_goal /fire/mission_navigate_to_pose nav2_msgs/action/NavigateToPose "{pose: {header: {frame_id: map}, pose: {position: {x: 3.31, y: 0.50, z: 0.0}, orientation: {z: 0.049979, w: 0.99875}}}}"
```

```bash
ros2 action send_goal /fire/mission_navigate_to_pose nav2_msgs/action/NavigateToPose "{pose: {header: {frame_id: map}, pose: {position: {x: 2.95, y: 0.65, z: 0.0}, orientation: {z: 0.0, w: 1.0}}}}"
```

세 번째 목표 뒤에 표적 탐지 및 FireManager의 MuJoCo 발사 처리를 관찰한다. 표적 지도 좌표는 YOLO와 object mapper가 계산하지만, 위 이동 목표는 수동 입력이다.

## 3. 결과 및 종료

- FireManager 상태에서 탐지·진압 결과를 확인한다.
- MuJoCo 결과 폴더에서 새 summary와 CSV를 확인한다.
- 전도 성공은 기존 기준대로 기울기 **45° 이상을 0.2초 유지**해야 한다.
- 녹화할 때 Webots와 MuJoCo 창을 배치하고, MuJoCo 창에서 `Tab`, `Shift+Tab`으로 패널을 숨긴다.
- 액션 서버가 목록에 없으면 목표를 보내지 말고, 두 터미널의 ROS 도메인이 63인지와 launch 오류를 확인한다.
- 종료할 때는 launch 터미널에서 `Ctrl+C`를 누른다. 같은 domain/port가 사용 중이면 재사용하지 않는다.
