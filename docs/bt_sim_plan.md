# 임무 BT 시뮬 확인 계획

`src/limo_fire/limo_fire_bt/behavior_trees/limo_fire_bt.xml` 이 Webots 에서 뜻대로 도는지 **눈으로** 확인하기까지의 순서. 2026-11-08 작성.
백로그는 [PB-11·PB-12](backlog.md), 토픽은 [sensor_topics.md](sensor_topics.md), 시뮬 실행은 [simulation.md](simulation.md), 연기 모드는 [smoke_mode_plan.md](smoke_mode_plan.md).

## 1. 지금 상태

| 항목 | 상태 |
|---|---|
| 트리 XML | 노드 포트를 코드에 맞춤 (`<TreeNodesModel>` 은 `mission --model` 로 뽑음) |
| BT 노드 구현 | 2026-11-11 전부 C++ 로 (limo_fire_bt README). Jazzy 에서 가짜 센서로 S2·S3·S4 분기 확인 |
| 실행 파일·launch | `mission`, `bt.launch.xml [sim:=true]`, `limo_fire_bt.yaml` + `limo_fire_bt_sim.yaml` |
| 시뮬 실행 | 2026-11-13~14 Jazzy PC 에서 S1(순찰)·S2(큰 불 → 신고 → 베이직 복귀 → 순찰 재개)·S3(같은 불 = 같은 사건) 확인. S4~S7 은 아직 (limo_fire_bt README "시뮬에서 확인한 것") |

## 2. 먼저 정할 것

| # | 질문 | 제안 | 까닭 |
|---|---|---|---|
| D1 | spec.md 는 "Nav2 BT(BehaviorTree.CPP v3) 플러그인"인데 트리는 v4 형식이다 | 따로 도는 임무 노드 + v4, 주행은 `navigate_to_pose` 호출. 정해지면 spec.md 를 고친다 | 임무 트리와 주행 트리가 나뉘고, Script·`_onFailure` 와 Groot2 모니터를 쓴다. 로봇 Jetson(Humble)에 4.x 가 있는지는 **아직 확인 안 함**. 안 되면 v3 형식으로 옮긴다 |
| D2 | 불 위치를 어떻게 받나 | **정함**: `/perception/objects` (`vision_msgs/Detection3DArray`) 를 더해 BT 가 읽는다 | 마커는 표시용이라 형식이 바뀌면 BT 가 깨진다 |
| D3 | 시뮬에서 불을 어떻게 "보게" 하나 | **정함**: `limo_fire_bt_sim.yaml` 이 `/perception/detections` 를 `/gt/detections` 로 바꿔 읽음 | 시뮬 불·연기는 YOLO 가 거의 못 잡는다(simulation.md). 정답 카메라가 같은 타입 `vision_msgs/Detection2DArray` 를 낸다. 점수·클래스 값이 BT 문턱과 맞는지는 확인 필요 |

## 3. 시뮬에서 그대로는 안 되는 것

토픽 이름·문턱이 XML 에 적혀 있어 지금 트리를 Webots 에 넣으면 `PreflightSensors` 에서 멈춘다.

| 항목 | 실물 | Webots | 할 일 |
|---|---|---|---|
| 융합 오도메트리 | `/odometry/filtered` | 없음, `/odom` 만 (`nav2_sim.yaml`) | 토픽을 설정으로 |
| 리모컨 오토 모드 | `/rc_state` | 없음 | 시뮬에서 검사 끄기 |
| 컬러·깊이 주기 | 약 30 Hz, 문턱 20 | 약 20 장/초 | 시뮬 문턱 낮추기, 시뮬 시계로 잴지 정하기 |
| 탐지 입력 | `/perception/detections` | YOLO 는 불을 거의 못 봄 | D3 |
| 초음파 | esp32_bridge | 월드 `LimoFire ultrasonic TRUE` 일 때만 | 연기 시나리오용 월드 사본 |
| 펌프·서보·열화상·레이더 | 부품 대기 | 없음 | 물 모드는 가짜 노드 필요 |
| 순찰점 | 미정 | 월드 yaml `patrol` | `LoadPatrolRoute` 입력을 지도 파일 대신 순찰점 파일로 |

- 규약(주기·임계값은 `config/*.yaml`)에 맞춰 토픽·문턱을 설정으로 빼고, 시뮬 차이는 덮어쓰기 파일(`limo_fire_bt_sim.yaml`) 하나로 준다.
- 물 모드 진입의 `/joint_states` 검사는 바퀴 관절만 나와도 통과할 수 있다. 서보 관절 이름까지 보게 한다.
- `fire_small` 크기 0.30 m 는 `IsSmallFire max_size` 0.3 과 딱 같다. 어느 쪽으로 판정할지 정하고 시험한다.
- 저장 지도 위에서는 `HasUnexploredFrontier` 가 대개 거짓일 것이므로, 탐사 모드 시험은 요청 신호(`IsSurvivorSearchRequested`)로 켠다.

## 4. 순서

| 단계 | 할 일 | 끝 기준 |
|---|---|---|
| 0 | D1~D3 정하기, Humble 에 BT.CPP 설치 확인 | 이 문서에 결정을 적음 |
| 1 | 가짜 노드 실행 파일로 ROS·Webots 없이 트리를 돌리고 Groot2 로 보기 | 5 절 S1~S3 분기가 Groot2 에서 차례로 보임 |
| 2 | 토픽·문턱을 설정으로 빼고 `limo_fire_bt_sim.yaml`, `bt.launch.xml`, `sim.launch.xml bt:=true` | 시뮬에서 `PreflightSensors` 통과 |
| 3 | 실제 노드: `IsTopicAlive` `IsAutoMode` `IsDepthAlignedToColor` `LoadPatrolRoute` `GetNextWaypoint` `NavigateToWaypoint` `StopRobot` | S1 순찰 |
| 4 | 실제 노드: `IsFireDetected` `IsSmokeDetected` 사건 판별, 신고는 로그로 대신 | S2·S3 |
| 5 | 고장 주입: 센서 차단 노드(연기 모드에서 만든 것)로 토픽 끊기 | S4 |
| 6 | 연기 모드: `SweepSmokeZone` 이 `/smoke_mode/set` 을 켜고 주행. 탐사 모드 연결 | S5·S6 |
| 7 | 물 모드: `limo_fire_sim` 에 가짜 펌프·서보 | S7 |

BT 노드는 C++ 로 `limo_fire_bt` 에 둔다. 임시 코드에는 `TODO(제거):` 를 단다.

## 5. 눈으로 볼 시나리오

월드는 `limo_fire_maze`(불 `fire_floor` 0.6 m, `fire_small` 0.3 m). 실행은 저장소 기준 환경인 Humble 로봇 Jetson 에서 하고, 다른 사람의 시뮬과 겹치면 포트·`ROS_DOMAIN_ID` 를 나눈다.

| # | 상황 | 보여야 하는 것 |
|---|---|---|
| S1 | 불 없이 순찰 | 순찰점을 차례로 돌고 트리가 순찰에 머문다 |
| S2 | 순찰 중 큰 불 | 순찰 멈춤 → 새 사건 → 신고 1 회 → 고를 고급 모드 없음 → `BasicFallback`(안전거리 감시 60 s) → 순찰 재개 |
| S3 | 같은 불을 다시 봄 | 같은 사건으로 판별, 119 신고를 되풀이하지 않음 |
| S4 | 순찰 중 라이다 끊기 | 순찰 멈춤, `SENSOR_FAULT`(`LIDAR_TOPIC_LOST`), 복구하면 재개 |
| S5 | 연기 + 초음파 켠 월드 | `SMOKE` 선택, 라이다가 끊겨도 계속, 초음파 끊기면 `ULTRASONIC_FAULT` → 베이직 복귀 |
| S6 | 수색 요청 | `FRONTIER` 선택, 300 s 넘으면 `EXPLORATION_TIMEOUT` → 베이직 복귀 |
| S7 | 작은 불 + 가짜 펌프 | `WATER` 선택, 펌프 고장 주입 시 `PUMP_ERROR` → 펌프 정지 → 베이직 복귀, 같은 사건에서 물 모드 재진입 안 함 |

**보는 화면**
- Webots: 로봇이 실제로 움직이는지.
- RViz 운용 화면(`viewer.launch.xml`): 경로·물체 지도.
- Groot2 Monitor: 지금 켜진 노드. 전체 트리가 160 개 넘는 노드라 무료판 20 개 제한에 걸린다 → 요약 트리 + 서브트리별 포트로 나눠 본다.
- 판마다 BT 기록 파일을 남겨 Groot2 로 다시 본다. 중간평가 영상(PB-22)에도 쓴다.

## 6. 실물로 옮기기 전에

- 시뮬 덮어쓰기 없이 `limo_fire_bt.yaml` 로 뜨는지 확인. 실물 문턱은 실측으로 다시 본다.
- 깊이 정렬(sensor_topics.md 주의 1): 정렬이 꺼져 있으면 `CheckDepthAlignment` 에 걸려 물 모드에 들어가지 않는다. `depth_registration` 을 정한다.
- 신고(`Call119`)는 시험 중 실제로 나가지 않게 막는 설정을 둔다.
- 주행 전 CLAUDE.md "실물 로봇" 안전 확인(충전기·주변·사람·이동 거리·오토 모드).

## 7. 실물 순찰 (2026-11-20) — 겪은 것과 다음 계획

실험실(`lab_1116b` 저장 지도 + AMCL)에서 `real.launch.xml bt:=true` 로 임무 BT 순찰을 처음 돌렸다.

**고친 것** (같은 날 코드에 반영)

| 겪은 것 | 원인 | 고침 |
|---|---|---|
| 리모컨을 오토로 올리자 로봇이 혼자 4 m 넘게 감 | 실물 기본 순찰점 파일이 지도 원점 기준이었다 | 실물은 기본 순찰점 없음 — RViz 에서 찍은 점(click_patrol → `/patrol/route`)만 순찰, 운용 화면 **순찰 패널** 버튼으로 켤 때까지 순찰 안 함 |
| 0.6 m 왕복이 제자리에서 맴돎 | 도착 허용 0.25 m · 순찰점마다 다음 점 쪽으로 180° 회전 | 점을 직접 찍게 함 (1.5 m 넘게 띄울 것) |
| 카메라 30 Hz 인데 YOLO 18.8 장/초 · 깊이 짝 실패 44 % | 팀 env.sh 대신 옛 셸로 띄워 Fast DDS 공유 메모리 칸 4 MB 설정이 빠짐 | `real.launch.xml` 이 그 설정 없으면 바로 멈춤 → 29.9 장/초 · 짝 실패 3 % |
| 순찰점에 물건이 있으면 못 가고 건너뛰기만 | 목표 칸이 막힘 | 플래너 허용 0.25 → 0.5 m, BT 는 1 m 안까지 오면 들른 것으로 침(`IsNearGoal`) |
| 순찰 중 탐사로 빠져 5 분 서 있음 | 지도를 아직 못 받은 순간 "모르는 곳" 으로 판정 | 지도 없으면 아는 곳으로 치고, 트리 만들 때 미리 구독 |
| 순찰·탐사가 서로 목표를 끊을 수 있음 | 둘 다 주행 목표를 냄 | 순찰을 켜면 BT 가 탐사를 끄고, 밖에서 탐사를 켜면 순찰을 끔 |

**다음 계획 — 막히면 비켜 가는 경로 추종 (아직 안 함, 팀원과 상의)**

- 지금 Nav2 경로 추종기는 RPP 다. 라이다로 앞 물체를 보면(`RegulatedPurePursuitController detected collision ahead!`) **비켜 가지 않고 선다.**
  돌아가는 길은 A\*(`SmacPlanner2D`, 비용 고려 2D A\*)가 1 초마다 다시 짜야 생기는데, 로봇에서 재계획이 0.8 Hz 로 밀려(부하 평균 7~10, RViz·RustDesk 만 1.3 코어)
  머뭇거리거나 물체에 닿았다.
- 탐사 쪽 지역 계획기(`train/explore/sim/include/local_planner.h`, 2026-11-19)가 막히면 비켜 가는 방식이지만 탐사 노드 안에 박혀 있어 Nav2 순찰에 바로 못 쓴다.
  그 설계가 "Nav2 DWB 처럼 궤적 묶음, 비용은 MPPI 꼴" 이라 **Nav2 MPPI 제어기**가 같은 일을 한다. 로봇에 `ros-humble-nav2-mppi-controller`·`nav2-dwb-controller` 가 이미 설치돼 있다.
- 기본값을 바꾸지 않는 까닭: RPP 는 연기 모드에 맞춰 0.70 m/s 로 손본 값이고(시뮬 상자 16 판·순찰 8 판), WeGo DWB 는 미로 U 자에서 제자리 회전만 했다(`nav2_params.yaml` 주석). MPPI 는 계산이 무거워 Jetson 부하가 더 오른다.
- 할 순서:
  1. MPPI 를 선택으로 더함 — `real.launch.xml controller:=mppi` 일 때만 `nav2_mppi.yaml` 로 FollowPath 를 바꿈 (기본 RPP 그대로, 속도 한도 0.70 m/s·로봇 크기 같게).
  2. 시뮬 미로에서 U 자 구간·상자 피하기를 RPP 와 견줌 (같은 판 수).
  3. 실물 순찰에서 견줌 — 물체 앞 머뭇거린 시간·닿음 횟수·Jetson 부하.
  4. 좋으면 연기 모드 담당과 상의해 기본값으로 할지 정함.
- 같이 볼 것: 경로 위에 막힘이 생기면 바로 다시 짜기(Nav2 경로 유효성 검사 — 지금은 1 초마다만), 지역 장애물 지도 갱신 2 → 5 Hz,
  "출발점이 장애물 안" 으로 막히면 0.2 m 물러나 다시 짜기(지금은 사람이 리모컨으로 빼냄), RViz 를 로봇이 아닌 노트북에서 띄우기(로봇 부하).
