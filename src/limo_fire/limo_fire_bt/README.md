# limo_fire_bt — 임무 행동 트리

순찰 → 불·연기 발견 → 신고 → 고급 모드(연기·물·탐사) → 실패하면 베이직 복귀. 트리는 `behavior_trees/limo_fire_bt.xml`(BehaviorTree.CPP 4.x),
노드는 이 패키지의 C++ 이고 실행 파일 `mission` 이 트리를 돌린다. 주행은 Nav2 `navigate_to_pose` 액션을 부른다(Nav2 bt_navigator 와 따로).
시뮬 확인 순서는 [docs/bt_sim_plan.md](../../../docs/bt_sim_plan.md).

```bash
# 실물 (순찰점 파일은 아직 없음)
ros2 launch limo_fire_bt bt.launch.xml patrol_file:=<순찰점 yaml>
# Webots: sim.launch.xml perception:=true 를 먼저 띄운 뒤
ros2 launch limo_fire_bt bt.launch.xml sim:=true patrol_file:=$HOME/limo_fire_ws/sim/webots/worlds/limo_fire_maze.yaml
ros2 topic echo /mission/events                      # 관리자 알림 (JSON 한 줄)
ros2 service call /mission/request_survivor_search std_srvs/srv/SetBool "{data: true}"   # 탐사 모드 요청
```

Groot2 Monitor 는 `localhost:1667`(설정 `groot_port`). `btlog_dir:=<폴더>` 면 판마다 `.btlog` 를 남겨 Groot2 로 다시 본다.

RViz 운용 화면(`limo_fire_description` `viewer.launch.xml`) 왼쪽 아래 **Mission** 현황판(`/mission/overlay`)에 지금 모드·단계(실행 중인 서브트리)·실행 중인 노드·순찰점·사건·실패 사유·최근 알림 3 개가 뜬다. 0.5 초마다 갱신하고, 크기·줄 수는 설정 `overlay.*`.

## 구조

| 파일 | 무엇 |
|---|---|
| `src/mission_node.cpp` | 설정 읽기 → 트리 만들기 → `tick_hz` 로 틱. 멈출 때 진행 중 노드를 멈춰 Nav2 목표·펌프·연기 모드·탐사를 끈다. `mission --model` 은 `<TreeNodesModel>` 을 코드에서 뽑아 낸다 |
| `src/ros_context.cpp` | 노드끼리 나눠 쓰는 ROS 연결: 토픽 구독(처음 쓸 때 만들어 공유), 토픽 주기, 탐지 기록, 물체 목록, 로봇 위치(TF), 사건 기록, 알림 |
| `src/status_overlay.cpp` | RViz 임무 현황판: 트리에서 실행 중인 서브트리·노드와 블랙보드 값을 읽어 `OverlayText` 로 냄 (트리는 바꾸지 않음) |
| `src/ros_helpers.cpp` | Nav2 목표·`SetBool` 서비스 호출을 틱마다 상태만 보게 감싼 것. 서버가 `server_timeout_s` 안에 없으면 실패 |
| `src/condition_nodes.cpp` | `IsTopicAlive` `IsAutoMode` `IsFireDetected` `IsSmokeDetected` `IsFireGone` `IsLocalizationStable` `IsSameIncident` `IsServiceAvailable` `IsSmallFire` `IsSurvivorSearchRequested` `IsTemperatureDropped` `IsWithinTimeLimit` `HasUnexploredFrontier` `IsDepthAlignedToColor` |
| `src/motion_nodes.cpp` | `LoadPatrolRoute` `GetNextWaypoint` `NavigateToWaypoint` `LogPatrolIssue` `StopRobot` `MonitorFromSafeDistance` `SweepSmokeZone` `ExploreFrontiers` |
| `src/mission_nodes.cpp` | `CreateIncident` `NotifyAdmin` `Call119` `RecordFireLocation` `StartMissionTimer` |
| `src/hardware_nodes.cpp` | `AimNozzle` `SprayWater` `StopPump` `PlayEvacuationGuidance` `TrackPeopleRadar` `TrackPeopleCamera` |
| `config/limo_fire_bt.yaml` | 실물 설정. 시뮬은 위에 `limo_fire_bt_sim.yaml` (토픽 바꿔 읽기·주기 문턱·리모컨 검사 끔) |

## 노드가 기대는 것

| 무엇 | 어디서 | 상태 |
|---|---|---|
| 센서 주기 | 트리의 `IsTopicAlive` (타입 상관없이 받음). 시뮬 차이는 `topic_remap`·`min_hz_override` | 있음 |
| 불·연기·사람 탐지 | `/yolo_detector/detections` (시뮬은 `/gt/detections`), `hit_window_s` 안에 `min_hits` 장 | 있음 |
| 불 위치·크기·사람 수 | `/object_mapper/objects` (`use_depth` 일 때). 없으면 정렬된 깊이 영상에서 탐지 상자 가운데 거리(`depth.*`), 깊이도 없으면 카메라 앞 `approx_distance_m`(잠정) | 있음 |
| 순찰점 | `patrol_file` yaml `patrol: [[x, y], ...]`. `robot_start` 가 있으면 세계 좌표(시뮬 월드 yaml)로 보고 지도로 옮김 | 시뮬만 |
| 연기 모드 | `/smoke_mode/set` (limo_fire_navigation) | 있음 |
| 탐사 | `/explore_frontier/enable`·`/state` (limo_fire_rl) | 있음 |
| 관리자 알림 | `/mission/events` 에 냄. 받는 웹 앱(limo_fire_report)은 아직 없음 | 내기만 |
| 119 신고 | `call119.enabled: false`(기본)면 로그만. 켜면 `/mission/call119` 에 내고 받는 곳이 없으면 실패 | 중계 없음 |
| 펌프·노즐·스피커·레이더·열화상 | `/pump/set`, `/nozzle/command`+`/joint_states`, `/speaker/evacuation`, `/radar/people`, `/thermal/image_raw` — 이름·타입 잠정 | 부품 대기 |

부품이 없으면 그 고급 모드는 진입 조건에서 걸러진다. 연기 모드는 스피커(`/speaker/evacuation`)가 없으면 `SPEAKER_FAULT` 로 끝나므로 시뮬 S5 는 가짜 스피커가 있어야 한다.

## 확인한 것 (2026-11-11, Jazzy, Nav2·시뮬 없이 가짜 센서)

- 센서가 없으면 `SENSOR_FAULT` 알림 후 5 초마다 다시 점검. 다 들어오면 순찰 시작, Nav2 가 없으면 순찰점마다 `PATROL_POINT_SKIPPED`.
- 불 탐지를 켜면 순찰 멈춤 → 사건 1 → `HAZARD_DETECTED` 한 번, 119 는 막힘 로그 → 고급 모드 없음 → `FALLBACK_TO_BASIC` → 다시 봐도 같은 사건(신고 되풀이 없음).
- 라이다를 끊으면 `SENSOR_FAULT` `LIDAR_TOPIC_LOST`.
- Webots·Nav2 위에서는 아직 안 돌려 봄. 로봇 Jetson(Humble)에 BehaviorTree.CPP 4.x 가 있는지도 아직 확인 안 함.

## 이미 처리한 불

순찰 중 불을 보면 멈추는데, 이미 처리한 사건 자리(3 m 안)의 불은 처리(`MarkIncidentHandled`) 뒤 `recheck_s`(300 s) 동안 멈추지 않고 순찰을 계속한다(`IsKnownHazard`). 300 s 가 지나도 보이면 다시 멈춰 감시한다. 이것이 없으면 꺼지지 않은 불 앞에서 감시 → 순찰 시작 → 같은 불 → 감시만 되풀이한다.
