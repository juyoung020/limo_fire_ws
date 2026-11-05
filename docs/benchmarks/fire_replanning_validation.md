# 화재 대응 검증 기록 — 2026-11-04

**Jazzy + 실제 Webots + 실제 Nav2 전체 시나리오 실행 및 독립 위치 검사 PASS.**
Humble 빌드와 실제 YOLO 추론은 아직 검증하지 않았다. 실제 모터·펌프·ESP32는 사용하지 않았다.

## 환경

Ubuntu 24.04, ROS Jazzy, Webots R2025a. 기존 아래 작업공간을 재사용했다.

- `~/webots_ros2_ws/install`: Jazzy로 빌드된 Webots driver/control (CMakeCache의 rclcpp_DIR 확인)
- `~/robot_sim_ws/install`: 기존 LIMO 실행 의존성. 현재 저장소의 webots_ros2_limo도 별도 빌드했다.
- `~/limo_fire_ws/install`: 이번 구현, Nav2, description, sim 플러그인

처음 `/opt/ros`만 불러왔을 때 Package not found였지만, 이후 위 overlay를 발견했다.
'Webots ROS 패키지가 이 컴퓨터에 전혀 없다'는 초기 판단은 정정한다. Humble은 실제로 없다.
CUDA/nvcc와 TensorRT가 없어 perception은 `BUILD_YOLO=OFF`로 CPU 매퍼만 빌드했다.
rviz_2d_overlay_msgs는 Debian 패키지를 `/tmp/limo_build_deps/root`에 추출해 사용했다.
시스템 설치나 `src/external` 소스 수정은 하지 않았다.

## 실제 Webots 결과

| 시험 | 기대 / 실제 결과 | 증거 |
| --- | --- | --- |
| 정상 전체 임무 | 작은불 정지→가상 소화→재개→큰불 구역 적용→Smac 우회→최종 도착 PASS | [로그](fire_replanning_webots/success_gt.log), [CSV](fire_replanning_webots/success_gt.csv) |
| 우회 경로 없음 | 반경 4.0m + 차체 여유가 8m 폭을 막음. 실제 Smac 실패→SAFE_STOP, 실제 정지 유지 PASS | [로그](fire_replanning_webots/no_path.log), [CSV](fire_replanning_webots/no_path.csv) |
| 가상 소화 실패 | 실패 Action 응답→구역 등록→시작점이 위험 구역에 있어 안전 경로 없음→SAFE_STOP PASS | [로그](fire_replanning_webots/extinguish_failure.log), [CSV](fire_replanning_webots/extinguish_failure.csv) |
| 가상 소화 무응답 | 20초 시뮬 타임아웃→소화 취소 요청→위험 등록→SAFE_STOP, 재개 없음 PASS | [로그](fire_replanning_webots/extinguish_timeout.log), [CSV](fire_replanning_webots/extinguish_timeout.csv) |

실패 대응 시험의 전체 임무 CSV는 **FAIL / mission_success=0**가 맞다. 이를 성공 임무로 바꾸지 않는다.
별도의 안전 판정은 Action 실패 + SAFE_STOP + 독립 위치에서 정지 유지가 모두 맞을 때만
`PASS negative safety check`와 `EXPECTED_SAFE_STOP_VERIFIED`로 기록한다.
작은불 실패로 구역이 이미 정지한 로봇을 포함하면, 그 안에서 정지 유지하는 것만 허용한다.
그 상태에서 움직이면 침범 실패다. 정상 성공 시나리오의 침범 기준은 완화하지 않았다.

정상 독립 검증의 첫 PASS 측정값:

| 지표 | 값 / 정의 |
| --- | --- |
| 전체 임무 | 1회 PASS. 반복 신뢰도나 일반적인 성공률을 입증하는 표본은 아님 |
| 감지→정지 | 0.24초, 시뮬 시계·odometry 기준. 소화 요청 시 실제 정답 위치 속도도 별도 확인 |
| 재계획 | 시뮬 시계 표시 0초 = 10ms 시계 분해능 미만. 계산이 물리적으로 0초라는 뜻 아님 |
| 새 경로 길이 | 8.8813m, 실제 Smac ComputePathToPose 결과 |
| 실제 총 이동 | 10.6561m, Webots supervisor 위치 누적 |
| 위험 구역 침범 | 없음. 차체 여유를 포함한 1.25m 원과 연속 위치 선분 검사 |
| 최종 실제 위치 | map (9.70893, 0.04229), 목표 (10,0), 오차 0.294125m < 사전 기준 0.30m |
| 소화 후 재개 | 실제 이동 재개 및 최종 Action 성공 확인 |

정답 위치는 전용 월드의 supervisor-enabled LIMO에서 읽으며, **Nav2 위치 추정 입력으로 사용하지 않는다.**
로봇 높이 0.10~0.20m, 정답 위치 신선도, 소화 직전 실제 정지, 실제 이동량, 최종 오차를 검사한다.
Nav2는 기존 AMCL/라이다/odom으로 동작한다. 초기 기준 TF-only 성공은 참고용이며,
현재 PASS는 독립 정답 검사에 근거한다.

## 자동 로직·통신 시험

`limo_fire_bt` 11/11, `limo_fire_perception` 1/1, 총 **12/12 PASS**.
[최종 실행 원문](fire_replanning_protocol.log).

- 폭 0.20/0.30 경계, 중간·NaN·음수·invalid flag, 선분-원 교차 검사
- 실제 Nav2 Costmap2D lethal 셀, revision 적용 ACK, clear/reset 후 위험 유지
- 실제 ROS Action 통신의 작은불 성공·실패·큰불·경로 없음·잘못된 입력·깊이 누락·중복·Nav2 실패·소화 타임아웃
- 합성 Detection/Depth/Mask/CameraInfo/TF의 0.10m 폭 추정 및 연속 탐지 중 depth 중단 보고

Action 시험의 Nav2·계획·소화·odom·ACK는 대역이다. 실제 Nav2/Webots 결과와 혼동하지 않는다.
시험마다 ROS_DOMAIN_ID를 분리해 앞 시험이나 지각 시험·GUI 데모의 메시지가 섞이지 않게 했다.
`/clock` 전달 순서 때문에 관측 stamp가 잠깐 앞설 수 있어 YAML에서 0.10초 시계 차이를 허용한다.
그보다 큰 미래 시각과 오래된 입력, 잘못된 좌표·신뢰도는 계속 거부한다.

## 실행 명령

빌드된 기존 Webots 환경을 불러온다:

```bash
cd ~/limo_fire_ws
source /opt/ros/jazzy/setup.bash
source ~/webots_ros2_ws/install/local_setup.bash
source ~/robot_sim_ws/install/local_setup.bash
source install/local_setup.bash
export WEBOTS_HOME=/usr/local/webots
export ROS_DOMAIN_ID=174

ros2 launch limo_fire_bt demo.launch.xml jazzy:=true rviz:=true
ros2 launch limo_fire_bt demo.launch.xml jazzy:=true rviz:=true \
  scenario_params:=$HOME/limo_fire_ws/install/limo_fire_bt/share/limo_fire_bt/config/demo_blocked.yaml \
  report:=/tmp/fire_demo_blocked.csv
ros2 launch limo_fire_bt demo.launch.xml jazzy:=true rviz:=true outcome:=failure \
  scenario_params:=$HOME/limo_fire_ws/install/limo_fire_bt/share/limo_fire_bt/config/demo_negative.yaml \
  report:=/tmp/fire_demo_failure.csv
ros2 launch limo_fire_bt demo.launch.xml jazzy:=true rviz:=true outcome:=timeout \
  scenario_params:=$HOME/limo_fire_ws/install/limo_fire_bt/share/limo_fire_bt/config/demo_negative.yaml \
  report:=/tmp/fire_demo_timeout.csv
```

각 실행이 끝나 완전히 종료된 뒤 다음 것을 실행한다. `jazzy` 기본값은 false로 Humble 설정을 유지한다.
실행된 주요 빌드/시험:

```bash
colcon build --packages-select limo_fire_msgs limo_fire_bt --cmake-args -DBUILD_TESTING=ON
colcon build --base-paths src/limo_fire/limo_fire_sim --packages-select limo_fire_sim --cmake-args -DBUILD_BAG_TOOLS=OFF
colcon build --base-paths src/external/webots_ros2_limo --packages-select webots_ros2_limo
colcon build --base-paths src/limo_fire/limo_fire_bringup src/limo_fire/limo_fire_description --packages-select limo_fire_bringup limo_fire_description
# CPU 매퍼 빌드에는 /tmp/limo_build_deps/root/opt/ros/jazzy를 CMAKE_PREFIX_PATH/AMENT_PREFIX_PATH에 추가
colcon build --packages-select limo_fire_perception --cmake-args -DBUILD_YOLO=OFF -DBUILD_TESTING=ON
# 실행에는 추출한 overlay_msgs lib 경로를 LD_LIBRARY_PATH에 추가
colcon test --packages-select limo_fire_bt limo_fire_perception --event-handlers console_direct+
colcon test-result --test-result-base build/limo_fire_bt --verbose
colcon test-result --test-result-base build/limo_fire_perception --verbose
```

## 발견·수정한 문제와 남은 한계

- Jazzy BT 기본 노드 중복 등록: 공용 원본 설정에서 plugin_lib_names만 빼서 설치 시 생성한다. 별도 전체 Nav2 설정 복사본을 유지하지 않는다.
- Jazzy Smac 플러그인 이름과 TwistStamped 입력: 전용 YAML 및 시뮬 어댑터로 처리한다.
  [Jazzy diff-drive 인터페이스 근거](https://control.ros.org/jazzy/doc/ros2_controllers/diff_drive_controller/doc/userdoc.html).
- controller_manager가 받는 robot_description에 제어 태그가 없던 문제: Webots 모드에서 기존 driver URDF를 xacro로 포함한다.
- 시작 순서: demo가 bt_navigator lifecycle active·신선한 odom·정답 위치를 확인하고 목표를 보낸다.
- 설치된 Webots에 기본 Floor/Wall PROTO가 없어 유효 라이다가 없었다. 전용 월드를 기본 Solid/Box 충돌 지형으로 바꾸자 248개 빔이 유효해졌다.
  GPU 소프트웨어 렌더링 시험만으로는 해결되지 않았고, 최종 성공은 일반 렌더링으로 실행했다.
- 초기 TF-only PASS 뒤 독립 검사에서는 실제 도착 오차가 기준을 만족하지 않았다. 기준을 넓히지 않고 Nav2 수락 오차를 0.25→0.10m로 엄격하게 했다.
  [초기 실패 원본](fire_replanning_webots/initial_gt_failure.txt)은 이전 CSV 형식 그대로 보존한 텍스트다.
- Humble 실행, YOLO 추론, 열화상 소화 확인, 실물 안전 거리·장치 검증은 미실행. 자동 데모의 화재 이벤트는 알려진 좌표/폭을 주입한다.
- 최종 오차는 기준에 가까워 반복 회차·여러 초기 자세·여러 지도에서 추가 평가가 필요하다.
- RViz overlay 플러그인 미설치/지도 GLSL 경고가 남아 있지만 Nav2와 Webots 주행은 실행됐다.
- 일부 중단 실행에서 활성 Action 종료 중 예외가 났다. 이번 성공·안전 정지 완료 후 종료와 강제 중단 복구는 구분한다.
- CPU 매퍼의 기존 초기화/anonymous namespace 경고, Jazzy bag 변환기의 time_stamp API 차이는 남아 있다.
  시뮬 빌드는 BUILD_BAG_TOOLS=OFF로 불필요한 오프라인 변환기를 제외했다. 기본값 ON은 유지한다.
- 상태·구역의 디스크 영속화, 다중 목표 preemption, 프로세스만 재시작하는 복구는 후속 범위다.

인터페이스와 파일 목록: [개발 문서](../fire_replanning.md).

## 정상 시나리오 재현 확인

도착 허용 오차를 0.10m로 엄격하게 한 뒤 독립 검증 정상 실행 두 회가 모두 PASS했다.
두 회의 최종 오차는 0.294125m, 0.264441m였다. 두 번째는 시계 허용 차이와 실패 대응 시험 중재 수정까지 반영된 최종 바이너리다.
[재실행 로그](fire_replanning_webots/success_gt_repeat.log), [재실행 CSV](fire_replanning_webots/success_gt_repeat.csv).
두 회 모두 실제 침범 없음·실제 소화 전 정지·재개·목표 도착을 확인했다.
개발 중 앞선 실패를 제외해 일반 임무 성공률 100%라고 주장하지 않는다. 위 두 회는 같은 독립 통과 기준의 재현 확인이다.

## 종료 결함 (미해결)

최종 데모는 결과 기록 후 navigation/localization lifecycle manager에 SHUTDOWN을 순차 요청하고
응답을 확인한다. [종료 시험 로그](fire_replanning_webots/shutdown_check.log)에서 이 응답과
실제 안전 정지는 확인됐지만, **그 뒤 Nav2 component_container_isolated 언로드 중 SIGSEGV가 여전히 발생했다.**
전체 주행의 CSV PASS는 유지되지만 `ros2 launch` 종료 코드는 1일 수 있다. 이를 깨끗한 프로세스 종료 성공으로 간주하지 않는다.
원인은 아직 특정하지 못했으며, 조합형 컨테이너의 종료 순서와 플러그인 수명 분석이 다음 우선순위다.
강제 중단 시 활성 Action 종료 예외와 외부 ros2_supervisor의 SIGINT(-2)도 별도 정리가 필요하다.

## MuJoCo Extinguish Action 종단 간 연결 — 2026-11-04

별도 MuJoCo MVP의 `/fire/mujoco_extinguish` Action 서버를 overlay에 포함해 `use_mujoco_extinguisher:=true`로 실행했다. 대상은 실제 YOLO가 아니라 Fire Demo의 알려진 좌표 입력이며, `demo_mujoco.yaml`에서 작은불을 `(2.0,0.25)m`에 배치한 정면 연결 시연이다.

- Webots/Nav2 LIMO 정지 후 MuJoCo action target `base_link=(1.673,0.250)m`
- MuJoCo `contact/direct_hit/toppled=true` → Fire Manager `VERIFY_RESULT` → `RESUME_NAVIGATION`
- 이후 큰불을 감지해 danger zone 적용 및 Smac 재계획(경로 8.949764m), 위험 침범 없음
- 최종 위치 `(9.74337,0.01020)m`, 목표 `(10,0)m`, 오차 `0.256836m`; 전체 임무 CSV `mission_success=1`
- 원시 실행 기록: MuJoCo 프로젝트 `results/ros2_bridge_verification/fire_aware_mujoco.md`

기본 좌표 `(1.0,0.8)m`를 사용한 첫 조합에서는 실제 정지 후 방위/상자 앞면 오프셋으로 조준각 제한 또는 전도 실패가 발생했다. 이 실패를 보존한 뒤, scene 상자 중심→힌지의 측정된 0.15m offset과 전용 정면 fixture를 적용했다. 물리 파라미터는 조작하지 않았다. 임의 좌표 성공률은 이 1회 성공으로 입증하지 않는다.

임무 자체와 lifecycle shutdown 응답은 PASS였으나 이후 Nav2 컨테이너 언로드 SIGSEGV와 launch의 ROS adapter 종료 예외가 다시 발생했다. 이는 기존 종료 결함으로 남아 있다. Webots 불 오브젝트에 MuJoCo 충돌 자세가 전달되는 것은 아니다.

## 최종 바이너리 재검증

종료 절차를 포함한 최종 바이너리에서도 실제 주행 PASS와 lifecycle 종료 응답을 확인했다.
[최종 로그](fire_replanning_webots/success_gt_final.log), [최종 CSV](fire_replanning_webots/success_gt_final.csv).
도착 오차를 엄격하게 한 뒤 정상 독립 실행은 총 3회 PASS이며, 마지막 오차는 아래 CSV 원문을 따른다.
최종 자동 회귀 시험도 11+1개 모두 PASS했다. 언로드 SIGSEGV는 여전히 별도 미해결이며 주행 PASS로 숨기지 않는다.
로그는 가독성과 git 공백 검사를 위해 줄 끝 공백만 제거했다.

## Wego Humble GUI·Action 실행 점검 — 2026-11-05

이번 실행은 전체 임무 성공 검증이 아니라 Wego Humble의 GUI·Action 배선 진단이다. SSH 셸에서 `DISPLAY`가 비어 있으면 Webots와 RViz가 `xcb` 초기화 오류로 종료한다. GUI 시험은 Wego 데스크톱 터미널 또는 동작하는 X11 forwarding 세션에서 한다. Fast DDS `Failed init_port`는 단독으로 치명 오류라 판단하지 않는다.

Wego `env.sh`에서 Conda를 deactivate한 뒤 `PYTHONHOME`과 `PYTHONPATH`도 unset하도록 보강했다. 이로써 ROS 2 Humble Python이 Conda의 NumPy/rclpy 패키지를 먼저 읽는 경로를 제거한다. Humble `controller_manager` import와 `spawner --help`를 확인했다.

처음 빌드하거나 소스를 수정한 경우 Wego에서 한 번 빌드한다. 이미 설치된 소스를 그대로 시험한다면 재빌드는 생략한다.

```bash
source /opt/ros/humble/setup.bash
source ~/limo_fire_ws/env.sh
cd ~/limo_fire_ws
colcon build --symlink-install \
  --packages-up-to limo_fire_bt limo_fire_bridge webots_ros2_limo limo_fire_perception
source ~/limo_fire_ws/install/setup.bash
```

GUI가 기동된 후 확인된 상태:

- `/odom` 발행 중, `diffdrive_controller` active
- Nav2 계획 경로 표시 및 `/fire/cmd_vel_input` 약 `-0.05 m/s`
- `/fire/cmd_vel_safe`는 0
- `/fire/mission_navigate_to_pose`: server `/fire_manager`, client 0
- `/navigate_to_pose`: server `/bt_navigator`, client에 `/fire_demo` 포함

원인은 `fire_demo`가 Fire Manager mission action을 통하지 않고 Nav2 action에 직접 목표를 보낸 것이다. Fire Manager는 활성 mission handle이 없는 상태에서 안전 출력 `/fire/cmd_vel_safe`를 0으로 유지한다. 아래처럼 명시적으로 mission action을 지정해 재실행한다.

터미널 1:

```bash
source /opt/ros/humble/setup.bash
source ~/limo_fire_ws/env.sh
source ~/limo_fire_ws/install/setup.bash
export WEBOTS_HOME=/usr/local/webots

cat > /tmp/fire_manager_action_fix.yaml <<'EOF'
fire_manager:
  ros__parameters:
    backend_action: /navigate_to_pose
    mission_action: /fire/mission_navigate_to_pose
EOF

ros2 launch limo_fire_bt fire_sim.launch.xml \
  world:=limo_fire_replan rviz:=true perception:=true \
  use_mujoco_extinguisher:=true show_mujoco_gui:=true \
  scenario_params:=/tmp/fire_manager_action_fix.yaml
```

터미널 2:

```bash
source /opt/ros/humble/setup.bash
source ~/limo_fire_ws/env.sh
source ~/limo_fire_ws/install/setup.bash
ros2 run limo_fire_bt fire_demo --ros-args \
  --params-file "$HOME/limo_fire_ws/install/limo_fire_bt/share/limo_fire_bt/config/fire_manager.yaml" \
  --params-file /tmp/fire_manager_action_fix.yaml \
  -p use_sim_time:=true \
  -p mission_action:=/fire/mission_navigate_to_pose
```

실행 중 `ros2 action info /fire/mission_navigate_to_pose`에서 client 1개가 보여야 한다. 이 수정 인자를 적용한 뒤 로봇 이동·발사·명중·전도는 아직 확인하지 않았다. `perception:=true`는 YOLO를 켜지만, 고정 좌표 Fire Demo 자체는 실제 카메라 검출 성공을 뜻하지 않는다. 엔진 파일과 그래픽 디스플레이가 준비되어야 한다.
