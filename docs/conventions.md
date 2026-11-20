# 규약 — 폴더·패키지·언어·URDF/TF·메시지

코드를 어디에, 무엇으로, 어떤 이름으로 두는지 한 곳에 모은다. 매 작업에 지켜야 할 핵심 규칙은 [CLAUDE.md](../CLAUDE.md) 에 있고, 여기는 그 세부다.

## 폴더 구조

ROS 2 관례를 따른다.

```
limo_fire_ws/                  워크스페이스 루트 (여기서 colcon build)
├── src/
│   ├── external/              남의 코드 — 고치지 않음, 출처·커밋은 README 표에
│   └── limo_fire/             우리 패키지
│       ├── limo_fire_description/   URDF/xacro, 메시, RViz 설정
│       ├── limo_fire_bringup/       launch(실물·시뮬), config/*.yaml 파라미터
│       ├── limo_fire_base/          섀시(구동·IMU·리모컨·모터 상태) 시리얼 드라이버 — WeGo limo_base 그대로 + 확장(C++)
│       ├── limo_fire_msgs/          우리 메시지·서비스·액션
│       ├── limo_fire_esp32_bridge/  ESP32 ↔ ROS (센서 + 발사대 레버, 시리얼을 혼자 엶)
│       ├── limo_fire_hardware/      탁구공 발사대: 불 관측 전달·접근·좌우 조준·발사 터미널(C++)
│       ├── limo_fire_perception/    불 탐지·크기 추정·소방 시설 인식
│       ├── limo_fire_people/        레이더 추적·구역 인원
│       ├── limo_fire_navigation/    Nav2 파라미터·지도·연기 모드
│       ├── limo_fire_sim/           Webots 우리 판: 드라이버 설정, 인식 벤치마크 정답 카메라 플러그인(C++)
│       ├── limo_fire_bt/            BT 플러그인 + 트리 XML
│       ├── limo_fire_explore/       탐사 노드: 합친 탐사(판단 모델 + A*) 하나 (2026-11-18 전 이름 limo_fire_rl)
│       ├── limo_fire_rviz/          운용 화면 RViz 패널 플러그인 (탐사 켜기·끄기)
│       └── limo_fire_report/        웹 앱(rosbridge)·음성 안내
├── sim/                       Isaac Sim 확장·USD, Webots 월드, maps/, 독립 MuJoCo 데모  [COLCON_IGNORE]
├── train/                     학습 (C++/CUDA/Rust, ROS 아님)  [COLCON_IGNORE]
│   ├── maps/                  맵 변환기 — 탐사·조준 공용
│   ├── explore/               탐사: sim/ mdp/ scripts/ baselines/ upper_lora/ viewer/ runs/(git 제외) — 강화학습은 archive/rl
│   └── detect/                YOLO 탐지: trainer/ tools/ data/(git 제외)
├── firmware/                  ESP32 코드
├── hardware/wiring/           추가 장치 회로도(KiCad)·배선(WireViz)·배선판 납땜 그림 [COLCON_IGNORE]
├── hardware/mounts/           센서·전장 고정대 출력 파일(STL)·그림 [COLCON_IGNORE]
├── tools/repo_check/          커밋·푸시 전 규칙 검사(Rust) — Claude Code 훅이 씀 [COLCON_IGNORE]
├── tools/io_table/            실행 중인 노드 입출력(ros2 node info) ↔ docs/runtime_io.md 생성 칸 견주기·새로 쓰기(Rust) [COLCON_IGNORE]
├── tools/sensor_view/         ESP32 시험 코드(firmware/sensor_check) 화면: 열화상·레이더 (C++/Qt5) [COLCON_IGNORE]
├── docs/                      문서
├── refs/                      참고 자료                    [COLCON_IGNORE]
└── archive/                   안 쓰는 것                   [COLCON_IGNORE]
```

- 패키지는 역할마다 하나, 이름은 `limo_fire_<역할>`. 새 패키지는 위 목록에 맞춰 만들고, 목록에 없는 역할이면 이 목록부터 고친다.
- 지금 있는 패키지는 description·bringup·base·navigation·perception·sim·esp32_bridge·hardware·bt·explore·rviz 다 (explore 는 2026-11-18 rl 에서 이름을 바꿈 — 강화학습을 그만둬서). 나머지는 만들 때 위 이름을 쓴다.
- 패키지 안: `include/<패키지명>/`, `src/`, `launch/`, `config/`, `test/` (ROS 2 기본 배치).
- 실행 파일·파라미터를 바꾸고 싶은 외부 패키지는 고치지 말고 `limo_fire_bringup` 에 우리 판을 만든다. 코드(플러그인)가 필요한 Webots 쪽 우리 판은 `limo_fire_sim`.

## 이름 (노드·토픽·서비스·launch) — 2026-11-18

누가 봐도 무엇인지 알게 이름 짓고, 모듈 하나 = 이름공간 하나로 묶는다. ROS 2 규칙을 따른다:
토픽·서비스 이름은 글자·숫자·`_`·`/` 만, 숫자로 시작하지 않고 `__`·끝 `/` 금지, `~/` 는 "노드 이름 아래"
([ROS 2 design "Topic and Service name mapping"](https://design.ros2.org/articles/topic_and_service_names.html)).
패키지 이름은 소문자·숫자·`_` 만, 글자로 시작 ([REP 144](https://reps.openrobotics.org/rep-0144/)).

- **snake_case, 하는 일을 말하는 이름.** 그만둔 방식·옛 실험 이름(rl · fuel · low · hier · slot0 · yolo …)은 노드·토픽·launch 이름에 쓰지 않는다.
- **모듈 이름공간**: 모듈의 주 노드는 이름공간을 쓰고 토픽은 이름공간 바로 아래(`/explore/status`). 같은 이름공간의 다른 노드는 자기 이름 아래(`~/` → `/explore/<노드>/<이름>`).
- **센서·드라이버·표준 토픽은 그대로**: `/scan` `/odom` `/imu` `/camera/*` `/cmd_vel` `/map` `/tf` `/tf_static` `/rc_state`.
- **팀원 모듈 이름은 그 사람이 정한다** (임무 BT `/mission/*`, 연기 모드, ESP32 다리) — 우리 이름을 바꿀 때는 그쪽의 참조만 고친다.
- **launch 파일 = 띄우는 것의 이름**: `robot.launch.xml`(로봇 센서·섀시) · `app.launch.xml`(우리 노드 전부 — 프로세스 `/app` 하나) · `perception.launch.xml`(인식만, 시뮬) · `navigation.launch.xml`(Nav2).
  노드 하나만 띄우는 launch 는 노드 이름(`approach_aim.launch.xml`). 설정 파일도 노드 이름(`config/explorer.yaml`).

| 모듈 (패키지) | 노드 | 주요 입출력 | launch |
|---|---|---|---|
| 탐사 (`limo_fire_explore`) | `/explore/explorer` | `/explore/status` `/explore/state` `/explore/done` `/explore/goal` `/explore/goal_status`, 서비스 `/explore/enable` (debug: `/explore/goal_marker` `/explore/candidates` `/explore/candidates_text` `/explore/decider_image` `/explore/cmd_debug`) | `app.launch.xml` (컨테이너 `/app`)
| 주행 (`limo_fire_navigation`) | `/explore/object_goto` (저장 지도만) | `/explore/goal` `/explore/goal_status` | `navigation.launch.xml` |
| 인식 (`limo_fire_perception`) | `/perception/object_detector` | `/perception/detections` `/perception/mask` `/perception/detection_image` | `app.launch.xml perception:=true` · `perception.launch.xml`(시뮬) |
| ESP32 (`limo_fire_esp32_bridge`) | `/esp32_bridge` | `/ultrasonic/*` `/thermal/image` `/thermal/max` `/radar/*` `/diagnostics`, 발사대: 액션 `/launcher/launch`, 서비스 `/launcher/enable`, `/launcher/state` `/launcher/status` `/launcher/joint_states` | `robot.launch.xml esp32:=true` |
| 발사대 조준 (`limo_fire_hardware`) | `/launcher/approach_aim` · `/fire_observation_bridge` | `/launcher/approach_aim/{state,status}`, 서비스 `…/start` `…/cancel`, `/fire/observations` | `approach_aim.launch.xml` · `real_camera_aim.launch.xml` |
| 인식 (`limo_fire_perception`) | `/perception/object_map` | `/perception/objects` `/perception/markers` `/perception/summary`, 서비스 `/perception/{clear,save,load}_objects`, 토픽 `/perception/save_objects_to` | `app.launch.xml perception:=true` · `perception.launch.xml`(시뮬) |

옛 이름 → 새 이름 표 (2026-11-18 바꿈):

| 옛 | 새 |
|---|---|
| 패키지 `limo_fire_rl` | `limo_fire_explore` |
| 노드 `/low_rl` (실행 파일 `low_rl`) | `/explore/explorer` (`explorer`) |
| 토픽·서비스 `/low_rl/<이름>` | `/explore/<이름>` (같은 끝 이름: status · state · done · enable · goal_marker · candidates · candidates_text · decider_image · instruction · target_class · goto_status · cmd_debug) |
| 노드 `/explore_frontier` | `/explore/frontier_explorer` (토픽·서비스 `/explore_frontier/x` → `/explore/frontier_explorer/x`) — 2026-11-20 노드째 지움, 탐사는 `/explore/explorer` 하나 |
| 노드 `/object_goto` | `/explore/object_goto` (`/object_goto/x` → `/explore/object_goto/x`) |
| 노드 `/yolo_detector` | `/perception/object_detector` |
| `/yolo_detector/detections` · `/mask` · `/image` | `/perception/detections` · `/perception/mask` · `/perception/detection_image` |
| 노드 `/object_mapper` | `/perception/object_map` |
| `/object_mapper/objects` · `/markers` · `/summary` | `/perception/objects` · `/perception/markers` · `/perception/summary` |
| 서비스 `/object_mapper/clear` · `/save` · `/load`, 토픽 `/object_mapper/save_to` | `/perception/clear_objects` · `/perception/save_objects` · `/perception/load_objects`, `/perception/save_objects_to` |
| launch `explore_fuel_rl.launch.xml` · `low_rl.launch.xml` · `explore_frontier.launch.xml` | `explore.launch.xml` · `explorer.launch.xml` · `frontier_explorer.launch.xml`(2026-11-20 지움) |
| launch `yolo.launch.xml` (limo_fire_perception) | `perception.launch.xml` |
| (2026-11-18 한 프로세스) launch `explore.launch.xml` · `explorer.launch.xml` | `app.launch.xml` |
| (2026-11-18) `/explore/object_goto/goal` · `/explore/instruction` · `/explore/target_class` | `/explore/goal` (explorer 가 직접 받음) |
| (2026-11-18) `/explore/goto_status` · `/explore/object_goto/status` | `/explore/goal_status` (`object_goto/target` · `overlay` 는 지움) |
| 설정 `low_rl.yaml` · `explore_frontier.yaml` | `explorer.yaml` · `frontier_explorer.yaml`(2026-11-20 지움) |
| RViz `explore_fuel_rl.rviz` · (2026-11-18) `explore.rviz` | `limo_fire_description/rviz/limo_fire.rviz` 하나로 합침 |

## 언어

새로 쓰는 코드는 Rust 또는 C++ 로만 쓴다. Python 은 쓰지 않는다.

- ROS 2 노드: C++(`rclcpp`, `ament_cmake`)가 기본. Rust(`rclrs`)는 ROS 의존이 없는 라이브러리·도구에 우선 쓴다.
- launch 파일: Python 대신 XML 또는 YAML launch(`*.launch.xml`, `*.launch.yaml`).
- ESP32 펌웨어(`firmware/`): C++ (Arduino 프레임워크 / PlatformIO).
- GPU 학습 환경: C++/CUDA.
- 예외: `src/external/` 의 남의 코드는 원래 언어 그대로 두고 고치지 않는다.

## URDF·TF·센서

- 좌표·단위는 REP-103(미터·라디안, x 앞·y 왼쪽·z 위), 프레임은 REP-105: `map → odom → base_link`.
- 프레임 이름은 실물 LIMO 드라이버와 같게: `odom`, `base_link`, `laser_link`, `imu_link`, `camera_link`. `base_footprint` 는 쓰지 않는다(실물에 없음).
- 센서·노즐 장착 위치는 전부 URDF(xacro) 한 곳에. launch 에 `static_transform_publisher` 를 흩어 두지 않는다. 모듈마다 xacro 매크로 파일 하나.
- 카메라류는 `<이름>_link`(장착)와 `<이름>_optical_frame`(z 앞) 을 따로 둔다.
- 팬·틸트는 revolute 관절, 서보 각도는 `/joint_states` 로 낸다.
- TF 하나는 발행 주체 하나: `odom → base_link` 는 EKF 만, `map → odom` 은 위치 추정 하나만.
- 주기·포트·임계값은 `config/*.yaml` 파라미터로. 코드에 박지 않는다.

## 메시지

- 표준 타입 우선(`sensor_msgs/Image`, `sensor_msgs/Range`, `sensor_msgs/JointState`, `std_srvs/SetBool` …). 맞는 게 없을 때만 `limo_fire_msgs` 에 만든다.
- 헤더 `frame_id` 는 URDF 링크 이름과 같게.
- 노드 사이 상태 알림은 사람이 읽는 한글 한 줄 `std_msgs/String`(`~/status`, `~/goal_status`), 켜짐 상태는 짧은 영문 값 `~/state`(`idle`·`running`·`paused`·`done`, TRANSIENT_LOCAL)로 낸다. 다른 노드가 글을 보고 갈래를 정할 때는 앞말만 본다(예: `object_goto` 는 `도착` 으로 시작하는 글).
- 지도 위 물체는 `vision_msgs/Detection3DArray`(`map` 틀): `id` = `<클래스>-<번호>`, `results[0].hypothesis.class_id` = 클래스 이름 글, `results[0].pose.covariance[35]` = 마지막으로 본 뒤 지난 초. 칸마다의 뜻은 [runtime_io.md](runtime_io.md#object_mapper-limo_fire_perceptionsrcobject_mapper_nodecpp-같은-설정-파일의-object_mapper-칸).
- 실행 중인 노드·스레드가 주고받는 토픽 전체(타입·QoS·주기·frame_id·상태 값)는 [실행 구조 — 스레드·토픽 입출력](runtime_io.md) 한 곳에 둔다. 토픽을 더하거나 바꾸면 그 표도 같이 고친다.
