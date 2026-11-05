# Fire-Aware Dynamic Replanning — 시뮬 전용

2026-11-04. **Jazzy + 실제 Webots + 실제 Nav2에서 전체 임무 및 독립 위치 검사 PASS.**
작은불 정지·가상 소화·재개, 큰불 우회·최종 도착을 확인했고, 경로 없음·소화 실패·타임아웃 안전 정지도 검증했다.
Humble 빌드·실제 YOLO 추론·실물 장치 동작은 검증하지 않았다.
상세 명령·측정값·실패 원인·로그는 [검증 기록](benchmarks/fire_replanning_validation.md).

## 저장소 분석과 배치

로컬 대상 저장소가 없어 GitHub `juyoung020/limo_fire_ws`를 clone했다. 기준 HEAD는
`fd4a15de58834ac9aec245e620c79e2408d07ff5`. 깨끗한 main에서 pull 후
`feature/fire-aware-replanning` 브랜치를 만들었다. 두 초안 ZIP은 홈에서 찾지 못했다.
CLAUDE.md, conventions, spec, nav2_params, sim/navigation launch, yolo_fire, object_mapper를 읽었다.

규약의 역할별 패키지에 맞춰 별도 replanner 대신 **limo_fire_bt**에 임무 중재를 통합했다.
제어 타입은 limo_fire_msgs, 독립 Webots 정답 플러그인은 limo_fire_sim에 둔다.
기존 Smac 2D, RPP, Nav2 기본 replanning BT, Object Mapper의 depth/TF 계산을 재사용한다.
SLAM·YOLO·경로 계획기를 복제하지 않고 `src/external`은 수정하지 않았다.

## 상태와 안전 게이트

1. `/navigate_to_pose` 목표를 보존하고 backend `/fire/nav2_navigate_to_pose`로 위임한다.
2. 화재 수신 시 FIRE_DETECTED→CLASSIFY_FIRE→SMALL_FIRE/LARGE_FIRE/UNKNOWN→STOPPING.
   속도 게이트를 닫고, 목표별 취소 응답과 terminal result, 최신 odometry의 연속 정지를 확인한다.
3. 작은불은 위치·정지 거리·주변 scan 여유가 유효할 때만 Extinguish Action을 요청한다.
   성공 result 확인 후 VERIFY_RESULT→RESUME_NAVIGATION. 실패는 구역 등록 후 우회 또는 정지.
4. 큰불/불확실은 UPDATE_DANGER_ZONE→양쪽 Costmap 적용 ACK→REPLANNING.
   Smac 결과의 모든 선분이 금지 원을 피할 때만 FOLLOW_NEW_PATH로 간다.
5. 경로 없음·Action 실패·취소/장치 타임아웃은 SAFE_STOP과 이유를 보고한다. 타임아웃을 성공으로 간주하지 않는다.

실제 모터·펌프 노드를 실행하지 않는다. fire_manager/mock/stamper는 use_sim_time이 아니면 시작을 거부한다.
Nav2 출력→`/fire/cmd_vel_input`→manager→`/fire/cmd_vel_safe`→Webots.
Jazzy는 시뮬 전용 stamper를 거쳐 `/fire/cmd_vel_stamped`로 연결한다. `/cmd_vel`은 쓰지 않는다.
RL 탐사도 이 launch에서 시작하지 않는다.

동일 ID 재관측은 중복 처리하지 않고, 분류 상승·위치 이동은 다시 처리한다.
소화 중 큰불로 상승하면 장치 취소와 안전 정지를 요청한다. 구역은 보수적으로 누적하고 자동 해제하지 않는다.
실패는 래치되며 원인을 확인한 뒤 시뮬 전체를 재시작한다. 프로세스만 재시작하는 복구·이력 영속화·다중 임무 preemption은 미구현이다.

## 제어용 지각·소화 계약

| 인터페이스 | 계약 |
| --- | --- |
| `/fire/observations` FireObservation | 촬영 stamp, 안정적 ID, map 위치, metric width, position/size validity, confidence |
| `/fire/mock_extinguish` Extinguish Action | fire_id와 PointStamped target. 물리적으로 소화가 확인된 뒤에만 success를 반환해야 함 |
| `/fire/danger_zones` DangerZones | 전체 원 집합과 revision. 반경은 차체 여유 포함 |
| `/fire/global_applied`, `/fire/local_applied` UInt64 | 해당 revision을 실제 Costmap 셀에 적용한 뒤 ACK |
| `/fire/state` String | `STATE|reason`, 현재 상태·실패 이유 |
| `/fire/ground_truth` PoseStamped | 독립 검증용. Nav2의 위치 추정 입력으로 사용 금지 |

Object Mapper의 동기화된 Detection2DArray/Depth/CameraInfo/TF에서
`width = bbox.size_x * depth_m / fx`를 구한다. 면적이나 RViz Marker를 제어 입력으로 읽지 않는다.
분할 깊이가 없고 상자 깊이만 있으면 뒤 벽일 수 있어 크기를 valid로 승인하지 않는다.
유효 추정 없이 화재 탐지가 계속되면 위치 미상의 UNKNOWN을 보고한다. 영상 길이/stride/encoding과 유한성을 검사한다.
폭 ≤0.20m 작은불, ≥0.30m 큰불, 중간/실패 UNKNOWN. 임계값·토픽·시간은 YAML이다.
`/clock` 수신 순서 차이만 0.10초 범위에서 허용하며 큰 미래/오래된 시각은 거부한다.

분할 깊이도 불꽃 뒤 배경일 수 있다. 실물에는 받침 depth·라이다 교차 검증·거리 불확도·열화상 확인이 필요하다.
매퍼 clear는 ID epoch를 바꾸지만 매퍼만 재시작하면 ID 재사용 가능하므로 전체 시뮬을 재시작한다.
가상 소화 `success|failure|timeout`은 장치 계약 시험이다. 실제 열화상 판정이나 Webots 불꽃 삭제를 뜻하지 않는다.

## Costmap과 좌표

DangerLayer는 전역·지역 Costmap에 실제 lethal 셀을 쓰고 inflation 전에 실행한다.
clear/reset으로 화재를 지우지 않으며, 지역 odom 프레임은 TF로 변환한다.
기본 위험 반경 1.0m + 차체·위치 여유 0.25m(잠정). 구역 revision을 양쪽에서 적용해야 계획한다.
기존 경로의 침범을 검사하고 모든 화재에서 보수적으로 정지한다. 새 경로, 후속 `/plan`, 현재 위치도 검사한다.
시작점이 금지 구역이면 억지 탈출하지 않고 정지한다.
공식 [Humble Layer API](https://github.com/ros-navigation/navigation2/blob/humble/nav2_costmap_2d/include/nav2_costmap_2d/layer.hpp)를 확인해
동적 원·적용 ACK·clear 방지에 맞는 커스텀 layer를 선택했다. Keepout Filter와 중복 적용하지 않는다.

office 월드의 fire_floor 폭은 0.70m, Candle 불꽃은 0.07×0.8=0.056m (메타데이터 0.06m는 반올림).
이름이 fire_small인 다른 불도 폭 0.30m라 큰불 기준이다. 사무실의 2m 복도는 위험 구역으로 완전히 막힐 수 있다.
따라서 실제 통로가 있는 전용 `limo_fire_replan` 월드를 만들었다. 바닥·벽은 설치 자산 다운로드가 필요 없는 기본 Solid/Box다.
세계 시작 (-5,0), 지도 시작 (0,0)이므로 이 전용 초기 자세에서만 `x_map=x_world+5, y_map=y_world`다.
일반 지각 입력은 TF를 사용한다. 정답 플러그인은 supervisor-enabled 전용 월드에서만 활성화된다.

## 실행과 PASS 조건

[Jazzy 실행 절차와 실패 대응 시험 명령](benchmarks/fire_replanning_validation.md#실행-명령)을 따른다.
Humble 기본 경로는 `source env.sh` 후 기존 Webots 의존성·메시지·패키지를 빌드하고
`ros2 launch limo_fire_bt demo.launch.xml`이다. 2026-11-05 Wego Humble에서는 GUI·Nav2 Action 경로의 시작 문제를 진단했으나,
수정 Action 경로로 이동·발사까지 재실행하지 않아 Humble 전체 임무 성공은 아직 검증되지 않았다.
재현 명령과 확인된 상태는 [Wego Humble 실행 점검](benchmarks/fire_replanning_validation.md#wego-humble-guiaction-실행-점검--2026-11-05)을 참고한다.

자동 데모는 알려진 map 위치/폭의 화재 관측을 주입한다. YOLO 정확도 시험이 아니다.
PASS는 Action 성공만으로 내지 않는다. 독립 정답 위치의 높이·신선도·이동거리·침범·최종 오차 <0.30m,
정지 후 소화, 실제 재개, 전 상태 전환, Costmap ACK와 안전한 경로 결과를 함께 요구한다.
최종 오차 기준은 유지했고 Nav2 수락 오차만 0.25→0.10m로 더 엄격하게 했다.
실패 대응 시험은 임무 실패를 그대로 기록하고, Action 실패+SAFE_STOP+정지 유지의 안전 판정을 별도로 낸다.

## 파일 목록

`src/limo_fire/` 아래:

- `limo_fire_msgs`: package.xml, CMakeLists.txt, msg/{FireObservation,DangerZones}.msg, action/Extinguish.action
- `limo_fire_bt`: package.xml, CMakeLists.txt, danger_plugins.xml, include/limo_fire_bt/safety.hpp,
  src/{fire_manager,mock_extinguisher,danger_layer,fire_demo,twist_stamper}.cpp,
  config/{fire_manager,nav2_fire,nav2_fire_jazzy,demo_blocked,demo_negative}.yaml,
  launch/{fire_sim,demo}.launch.xml, test/{safety_test,protocol_test,danger_layer_test}.cpp
- `limo_fire_bringup/launch`: sim.launch.xml, navigation.launch.xml, sim_robot.launch.xml
- `limo_fire_description/urdf/limo_fire.urdf.xacro`: Webots 모드에서 동일 제어 URDF 포함
- `limo_fire_navigation/maps/sim/limo_fire_replan.{yaml,pgm}`
- `limo_fire_perception`: CMakeLists.txt, package.xml, config/yolo_fire.yaml, src/object_mapper_node.cpp,
  test/fire_observation_test.cpp
- `limo_fire_sim`: CMakeLists.txt, limo_fire_sim_plugins.xml, resource/limo_webots.urdf, src/ground_truth_pose.cpp

설치 시 기존 Nav2 원본에서 BT 목록만 제거한 nav2_base_jazzy.yaml을 생성하며 전체 설정을 중복 관리하지 않는다.
루트 sim/webots/worlds/limo_fire_replan.{wbt,yaml}, docs/conventions.md, 이 문서, 검증 문서와 증거 로그/CSV,
CHANGELOG.md도 변경했다.

## 팀 연결과 다음 우선순위

### MuJoCo 가상 발사 연결 (코드 통합 및 지정 시나리오 검증)

별도 MuJoCo MVP 저장소의 `limo_fire_bridge/mujoco_extinguisher` Action 서버 소스는 `src/external/mujoco_ball_mvp/ros2/limo_fire_bridge`에 원본 그대로 포함되어 mock 서버를 대체할 수 있다. Fire manager는 `use_mujoco_extinguisher:=true`일 때 정지 후 대상 위치를 보내며, MuJoCo의 접촉·공중 직접 명중·전도가 모두 확인된 경우에만 작은불 소화 성공을 받는다. MuJoCo 실행기 자체는 별도 `mujoco_ball_mvp` 체크아웃의 `mujoco_root`에서 실행한다. 기존 통합 검증은 임시 Jazzy overlay에서 수행했다. Webots/Nav2 전체 시나리오의 소화·재개·큰불 재계획·최종 위치 검증은 통과했으나, 이 결과를 이번 브랜치의 새 Wego 실행 결과로 해석하지 않는다. 초기 기본 화재 좌표 실패와 Nav2 언로드 SIGSEGV는 별도 기록되어 있다. GUI 결과와 제약은 MuJoCo 프로젝트 `ROS2_INTEGRATION.md`와 `results/ros2_bridge_verification/fire_aware_mujoco_gui.md`에 있다.

이 Action은 별도 MuJoCo 장면에서 계산되며 Webots 화재 상자나 로봇 물리 상태를 바꾸지 않는다. Jazzy 임시 overlay 빌드와 지정 시나리오의 성공 Action/전체 임무를 검증했다. 임의 target 조건 및 정상적인 Nav2 컨테이너 언로드는 여전히 미검증이다.

### Webots YOLO 표적 지도 → MuJoCo live map

`limo_fire_bringup sim.launch.xml perception:=true`는 카메라 YOLO 탐지와 depth/TF 기반 `object_mapper` 지도를 켠다. 이때 `yolo_engine_path`로 기기별 TensorRT 엔진을 지정할 수 있다. `src/external/mujoco_ball_mvp/ros2/limo_fire_mujoco`의 `live_map.launch.py map_yaml:=<같은 Nav2 map YAML>`은 저장된 지도와 MuJoCo viewer, ROS pose bridge를 함께 실행한다. 브리지는 `/odom`과 `/object_mapper/markers`를 `map` 프레임으로 정합해, 검출된 fire landmark를 MuJoCo 표적으로 표시한다.

이 연결은 구현했고 map/odom 좌표 변환과 UDP packet은 합성 메시지로 확인했다. Wego에서 실제 fire 표식의 `CAMERA FIRE TARGET SYNCED`와 MuJoCo GUI 표시도 앞서 확인했다. 다만 최신 Wego 재실행에서는 Webots 외부 컨트롤러와 `odom` TF가 기동되지 않아 자동 접근·발사는 수행되지 않았다. 지속 SLAM 중인 occupancy map을 실시간 충돌 geom으로 갱신하지 않으며 저장된 YAML/PGM map을 불러온다. MuJoCo 물리 실행기와 `.venv`는 별도 `robot-programming` 체크아웃에 필요하며, 경로가 기본값과 다르면 launch의 `mujoco_root`와 `python` 인자를 맞춘다. 자세한 실행과 환경 제한은 MuJoCo MVP `POSE_SYNC.md`에 기록했다.

### 단일 표적 자동 접근 → MuJoCo GUI 발사 (선택 실행)

`limo_fire_mujoco`에는 가장 가까운 `fire` mesh 표식을 선택해 Nav2 목표를 보내고, 도착 뒤 거리·방향·정지 조건을 확인한 경우에만 MuJoCo GUI 발사를 요청하는 `auto_approach_fire` 노드가 있다. 이 경로는 한 번 접근하고 한 번 발사하는 시연용이며, Fire Manager의 임무·위험 구역 재계획 동작을 대신하지 않는다. `fire_sim.launch.xml`의 Fire Manager와 함께 실행하면 Nav2 목표가 중복될 수 있으므로 두 경로를 동시에 켜지 않는다.

팀 워크스페이스에서 패키지를 빌드한다:

```bash
cd ~/limo_fire_ws
source /opt/ros/humble/setup.bash
colcon build --symlink-install --packages-up-to limo_fire_bridge limo_fire_mujoco
source install/setup.bash
```

별도 터미널에서 평상시 Webots/Nav2와 perception을 실행한다. 이 자동 접근 시연에서는 `fire_sim.launch.xml` 대신 기본 bringup을 사용한다:

```bash
source /opt/ros/humble/setup.bash
source ~/limo_fire_ws/install/setup.bash
ros2 launch limo_fire_bringup sim.launch.xml world:=limo_fire_world perception:=true rviz:=true
```

MuJoCo 지도 화면과 marker/pose bridge:

```bash
source /opt/ros/humble/setup.bash
source ~/limo_fire_ws/install/setup.bash
ros2 launch limo_fire_mujoco live_map.launch.py map_yaml:=$HOME/limo_fire_ws/src/limo_fire/limo_fire_navigation/maps/sim/limo_fire_world.yaml
```

세 번째 터미널에서 한 번의 자동 접근·발사를 시작한다:

```bash
source /opt/ros/humble/setup.bash
source ~/limo_fire_ws/install/setup.bash
ros2 run limo_fire_mujoco auto_approach_fire --ros-args -p use_sim_time:=true -p standoff_m:=1.2
```

발사 전제는 `/object_mapper/markers`의 `fire` mesh 표식, `/odom`, `map → odom` TF, Nav2 `/navigate_to_pose` action, 그리고 살아 있는 MuJoCo GUI/UDP bridge다. 목표나 TF가 없거나 Nav2가 실패하면 로봇을 움직이거나 발사하지 않는다. 발사 요청은 GUI에서 유효한 표적이 조준 가능하고 일시정지/준비 상태일 때만 실행된다. ObjectMapper 표식은 map에서 지속될 수 있으므로 오래된 표적이 있으면 시연 전에 지도를 지우고 다시 만들거나 ObjectMapper를 초기화한다. `standoff_m`, 허용거리와 정지 속도는 시뮬레이션 설정값이며 LIMO의 실측 제동거리로 검증되지 않았다. 이번 Wego 재실행에서 자동 접근부터 전도까지 종단 간 성공은 아직 확인하지 않았다.

1. Humble 대상 PC에서 동일 빌드·실행, 여러 회차·초기 자세·지도에서 반복 신뢰도와 최종 오차 측정.
2. 지각 팀은 FireObservation의 ID/stamp/width validity 계약에 맞춰 실제 YOLO/Depth 품질을 검증.
3. 소화 팀은 Extinguish success 전에 실제 소멸 확인, cancel 후 장치 정지 확인을 보장. mock을 실물로 바로 remap하지 말 것.
4. 임무 팀은 목표를 manager의 `/navigate_to_pose`로 보내고 backend를 직접 호출하지 않는다.
5. 보고 팀은 `/fire/state`와 DangerZones를 구독한다. 검증 전용 ground truth는 운영 판단에 쓰지 않는다.

종료 시 lifecycle manager 응답을 확인하도록 했지만 Nav2 컨테이너 언로드 충돌은 남아 있다.
**주행 시연 PASS와 프로세스 종료 정상 여부를 구분해야 한다.** 상세 종료 로그는 검증 기록을 본다.
