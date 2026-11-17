# ROS 시험 공부 — 돌아가는 시스템 전체 지도

LIMO Pro 화재 대응 로봇이 **실제로 돌 때** 어떤 launch 가 어떤 프로세스를 띄우고, 그 안에 어떤 노드·스레드가 있고, 무엇을 받아 무엇을 내는지 한 곳에 모은다.
우리 코드뿐 아니라 남의 코드(WeGo 드라이버, Nav2, Cartographer, robot_localization EKF, robot_state_publisher, Webots, RViz)와 팀원 코드(임무 BT, 연기 모드, ESP32 다리)까지 다룬다.
끝의 (f) 는 교수님이 물을 "현직자 질문" 40 개와 이 프로젝트에서 실제로 겪은 일로 쓴 답이다.

2026-11-18 · `main` 6c85897 기준 · 자세한 표는 [실행 구조](../runtime_io.md)·[센서 토픽](../sensor_topics.md)·[규약](../conventions.md) 에도 있고, 이 문서는 그것들을 **시험용으로 한눈에** 다시 묶고 실측을 더한 것이다.

## 0. 먼저 읽기

### 근거 등급 — 표마다 어디서 온 값인지

| 표시 | 뜻 | 어디 |
|---|---|---|
| 🧪 시뮬 실측 | 이 PC 에서 Webots 시뮬을 실제로 띄우고 `ros2 node/topic/service/action/param/lifecycle/component` 와 `ps -T` 로 뽑은 값 | [`raw/sim_A_slam_explore/`](raw/sim_A_slam_explore) · [`raw/sim_B_lowrl_explore/`](raw/sim_B_lowrl_explore) · [`raw/sim_C_patrol/`](raw/sim_C_patrol) |
| 🤖 실물 녹화 | 실물 로봇 녹화(`ros2 bag info`)·기동 로그에서 센 값 | [`raw/real_bags/`](raw/real_bags) |
| 📄 코드 기준 | 이 PC 에서 못 띄워 launch·config·소스로만 정한 값 | 각 줄에 파일 |

**이 PC 에서 못 띄운 것 (📄 코드 기준)**
- 실물 드라이버 `limo_fire_base`(섀시 UART)·`ydlidar_ros2_driver`·Orbbec 카메라·EKF(`robot_localization` 은 `geographic_msgs` 가 없어 빌드 안 됨) — 시뮬에서는 Webots 플러그인이 같은 이름·토픽으로 대신 낸다.
- 인식(`object_detector`·`object_map`) — TensorRT 와 Jetson CUDA 12 기준 코드라 이 PC(CUDA 13)에서 빌드 안 됨.
- 임무 BT(`mission`) — BehaviorTree.CPP 4.x 가 이 Humble PC 에 없음(팀원이 Jazzy PC 에서 확인).
- 판단 모델 서버(llama-server) — 다른 작업이 GPU 를 쓰고 있어 탐사 위층은 규칙 `upper:=slot0` 로 대신 띄움(노드 구조·토픽은 같음).

**시뮬 실측의 한계**: 화면 없는 Xvfb + 소프트웨어 렌더에, 다른 작업이 CPU 를 많이 쓰는 중(load 15)이라 시뮬이 실시간보다 느렸다(sim A 약 0.8 배, sim B 약 0.3 배 — `rtf.txt`). 그래서 **벽시계 Hz 는 실물보다 낮다.** 주기는 🤖 실물 녹화 값을 먼저 보고, 🧪 값은 "구조가 이렇게 이어져 있다" 의 증거로 본다.

### 우리 모듈 이름 (2026-11-18 바꿈)

이름 규칙과 옛 이름 → 새 이름 표는 [conventions.md "이름"](../conventions.md). `docs/exam/raw/` 의 덤프는 바꾸기 전에 뜬 것이라 옛 이름(`/low_rl`, `/yolo_detector`, `/object_mapper`, `/object_goto`, `/explore_frontier`)과
옛 구조(탐사·`object_goto`·인식이 따로 프로세스 셋, YOLO 작업 스레드)가 그대로 있다. **2026-11-18 부터 우리 노드는 프로세스 하나 `/app`, 실행기 스레드 하나** — 지금 구조는 [runtime_io.md](../runtime_io.md) 가 기준.

| 모듈 | 노드 | 토픽 |
|---|---|---|
| 탐사 (`limo_fire_explore`) | `/explore/explorer` (합친 탐사: 판단 모델 + A*, 물체로 가기 지시도) — 컨테이너 `/app` | `/explore/status` · `state` · `done` · `goal`(지시) · `goal_status`, 서비스 `/explore/enable` (debug: `candidates` · `candidates_text` · `decider_image` · `goal_marker` · `cmd_debug`) |
| 탐사 | `/explore/frontier_explorer` (Nav2 판 프론티어 탐사) | `/explore/frontier_explorer/*` |
| 주행 (`limo_fire_navigation`) | `/explore/object_goto` (저장 지도 "○○ 로 가", Nav2) | `/explore/goal` · `/explore/goal_status` |
| 인식 (`limo_fire_perception`) | `/perception/object_detector` (YOLO) — 컨테이너 `/app` | `/perception/detections` · `mask` · `detection_image` |
| 인식 | `/perception/object_map` (물체 지도) — 컨테이너 `/app` | `/perception/objects` · `markers` · `summary` |
| 그대로 | `/smoke_mode`, `/mission`, `/esp32_bridge`, `/limo/limo_fire_base` | |

### 누가 만든 것인가 (소유)

| 소유 | 모듈 |
|---|---|
| **우리(김주영)** | 섀시 드라이버 확장 `limo_fire_base`, 탐사 `explorer`·`frontier_explorer`, 인식 `object_detector`·`object_map`, 저장 지도 `object_goto`, RViz 탐사 패널, launch·config(`limo_fire_bringup`), URDF(`limo_fire_description`), 시뮬 우리 판(`limo_fire_sim`) |
| **팀원** | 임무 BT `mission`(`limo_fire_bt`), 연기 모드 `smoke_mode`, ESP32 다리 `esp32_bridge`, 펌웨어 |
| **남의 코드(base)** | WeGo 드라이버(Orbbec·YDLIDAR·`robot_localization`), Nav2(Humble apt), Cartographer(`cartographer_ros`), `robot_state_publisher`·`joint_state_publisher`, Webots + `webots_ros2_driver` + 교수님 `webots_ros2_limo`, RViz2, llama-server(llama.cpp + 우리 패치) |

### 세 가지 임무 모드

| 모드 | 언제 | 띄우는 것 | 주행 |
|---|---|---|---|
| ① 첫 지도 탐사 | 처음 한 번 | `robot.launch.xml` + `app.launch.xml stack:=true [perception:=true]` + llama-server | `explorer` 의 A* 가 `/cmd_vel` 직접 (Nav2 없음) |
| ② 순찰 | 매일 밤, 저장 지도 위 | `robot.launch.xml` + `navigation.launch.xml` + `perception.launch.xml objects_file:=…` + `bt.launch.xml` | Nav2 (`navigate_to_pose`) |
| ③ 화재 대응 | 불을 봤을 때 | ② 와 같은 프로세스. BT 가 사건 → 신고 → 고급 모드(연기·물·탐사) 로 감 | Nav2, 연기 모드면 `smoke_mode` 가 위치 추정을 바꿈 |

---

## (a) 전체 지도 — launch → 프로세스 → 노드 → 스레드

### a-1. 실물 로봇 (📄 코드 기준 + 🤖 2026-11-17 기동 로그)

```
robot.launch.xml  (항상 먼저)
├─ jetson_clocks            한 번 돌고 끝 (클럭 고정)
├─ robot_state_publisher    URDF → /tf_static, /joint_states → 바퀴 /tf
├─ joint_state_publisher    바퀴 관절 0 (실물엔 바퀴 각도 낼 노드가 없어서)
├─ limo_fire_base  (ns limo) 섀시 UART: /odom /imu /rc_state … , /cmd_vel 받음
├─ camera_container         컴포넌트 컨테이너 1 개 — /camera/camera (Orbbec)
├─ ydlidar_ros2_driver_node /scan
├─ ekf_filter_node_odom     /odom + /imu → /odometry/filtered, TF odom→base_link
└─ (esp32:=true) esp32_bridge, (viz:=true) rviz2

① app.launch.xml stack:=true perception:=true
├─ mapping.launch.xml: cartographer_node, cartographer_occupancy_grid_node
├─ /app (component_container — 프로세스 하나, 실행기 스레드 하나, intra-process)
│     ← explorer (+ perception:=true 면 object_detector + object_map)   ↔ HTTP llama-server :8171 (ROS 밖, 손으로 먼저 띄움)
└─ (rviz:=true) rviz2 (explore.rviz)

② navigation.launch.xml  (+ perception.launch.xml, bt.launch.xml)
├─ nav2_container (component_container_isolated) ← controller_server, smoother_server, planner_server,
│     behavior_server, bt_navigator, waypoint_follower, velocity_smoother, lifecycle_manager_navigation,
│     map_server, amcl, lifecycle_manager_localization
├─ smoke_mode            (늘 뜸: 스캔 중계 + map→odom)
├─ object_goto (저장 지도 "○○ 로 가", /explore/goal)
├─ (rviz:=true) rviz2 (limo_fire.rviz)
└─ bt.launch.xml: mission (BT 4.x, 한 스레드 실행기 + Groot2 :1667)

save_map.launch.xml : map_saver_cli × 2 (src·install) + /perception/save_objects_to 두 번 → 끝나면 사라짐
record.launch.xml   : ros2 bag record (늘 같은 토픽 묶음)
```

🤖 실물 기동 로그(2026-11-17, [`raw/real_bags/robot_launch_1117.txt`](raw/real_bags/robot_launch_1117.txt)): `jetson_clocks`·`robot_state_publisher`·`joint_state_publisher`·`limo_fire_base`·`component_container`(카메라)·`ydlidar_ros2_driver_node`·`ekf_node` 7 프로세스. 탐사 쪽([`explore_launch_1117.txt`](raw/real_bags/explore_launch_1117.txt))은 `cartographer_node`·`cartographer_occupancy_grid_node`·`component_container_mt`(그날은 멀티스레드 — 다음 날 한 스레드로 바꿈, Q6)·`object_goto`·`explorer`, 컨테이너에 `/perception/object_detector`·`/perception/object_map` 적재.

#### 프로세스 → 스레드 (실물 노드)

| 프로세스 | 스레드 (실행기 + 그 밖) | 하는 일 |
|---|---|---|
| `limo_fire_base` | 시리얼 읽기 `std::thread` + 실행기 1 (`rclcpp::spin`) + DDS | 읽기 스레드가 프레임을 풀어 **그 자리에서** `/odom` 50 Hz·`/imu` 100 Hz·`/rc_state` 40 Hz… 발행(stamp = 받은 순간 `now()`), 실행기는 `/cmd_vel` → UART 쓰기 |
| `ydlidar_ros2_driver_node` | main 고리(`WallRate 20`) + SDK 읽기 스레드 | 한 바퀴 모아 `/scan` 10 Hz (SensorDataQoS) |
| `camera_container` | 실행기 1 + Orbbec SDK 프레임 콜백 스레드 + 컬러 스레드 | HW 깊이→컬러 정렬, `/camera/*` 30 Hz + TF |
| `ekf_filter_node_odom` | 실행기 1 (30 Hz 타이머) | 융합 → `/odometry/filtered` + TF `odom→base_link` |
| `cartographer_node` | 실행기 + Cartographer 뒤 작업 스레드(최적화) | SLAM → TF `map→odom`(50 Hz 설정), `/submap_list` |
| `cartographer_occupancy_grid_node` | 실행기 | `/submap_list` → `/map` 1 Hz |
| **`/app`** (2026-11-18 ~) | 실행기 **1** — `object_detector` · `object_map` · `explorer` 의 모든 콜백·타이머, TF 리스너 스레드 없음 (+ DDS·CUDA 안 스레드) | 영상 → GPU 에 걸기(비동기) · 2 ms 타이머가 끝났나 묻고 탐지 발행 → 물체 지도 → `/scan` 한 장 = 탐사 걸음 0.1 s(비차단 HTTP) → `/cmd_vel` |
| `object_goto` (저장 지도) | 실행기 1 + TF 리스너 스레드 | 글 → 클래스 → 설 자리 → `navigate_to_pose` |
| `nav2_container` | **컴포넌트마다 실행기 스레드**(isolated) + 각 서버 안 TF 리스너 스레드 | 아래 Nav2 표 |
| `mission` | `SingleThreadedExecutor` 를 틱 사이 `spin_some` + Groot2 ZMQ 스레드 | BT 10 Hz 틱 |
| `rviz2` | Qt 화면 스레드 + ROS 실행 스레드 | 표시, 탐사 패널 서비스 부름 |
| 모든 ROS 프로세스 | Fast DDS 받기·이벤트·발견 스레드(약 10 개) | 역직렬화해 실행기 줄에 넣음 (Q8) |

### a-2. Webots 시뮬 (🧪 실측)

`sim.launch.xml` 은 실물 `robot.launch.xml` 자리에 `sim_robot.launch.xml` 을 쓰고, 그 뒤 지도·주행·인식 launch 는 **실물과 같은 파일**을 인자만 바꿔 include 한다(`use_sim_time:=true`, `odom_topic:=/odom`, `params_override:=nav2_sim.yaml`).

```
sim.launch.xml world:=limo_fire_maze [slam:=true] [nav:=true] [explore:=$(slam)] [perception] [bt] [rviz]
├─ sim_robot.launch.xml
│   ├─ webots (→ webots-bin, 물리·렌더)                 33 스레드, CPU 310~380 %
│   ├─ Ros2Supervisor (python) → /clock
│   ├─ webots_controller_LIMO (→ driver, C++)          28 스레드. 안에 노드 /LIMO, /limo/limo_base, /camera/camera,
│   │                                                    /ydlidar_ros2_driver_node, /controller_manager, /diffdrive_controller,
│   │                                                    /joint_state_broadcaster (실물과 같은 이름·토픽)
│   ├─ spawner × 2 (diffdrive_controller, joint_state_broadcaster 켜고 끝)
│   └─ robot_state_publisher (우리 URDF, platform:=webots)
├─ slam:=true  → mapping.launch.xml (cartographer × 2)
├─ nav:=true   → navigation.launch.xml (nav2_container + smoke_mode + object_goto nav2, slam 이면 map_server·amcl 없음)
├─ explore     → frontier_explorer
├─ perception  → perception.launch.xml (이 PC 는 못 띄움)
├─ rviz        → viewer.launch.xml (limo_fire_rviz)
└─ bt          → bt.launch.xml sim:=true (이 PC 는 못 띄움)
```

#### 🧪 sim A — `slam:=true` (Cartographer + Nav2 지도 없이 + 프론티어 탐사 + RViz)

[`raw/sim_A_slam_explore/threads.txt`](raw/sim_A_slam_explore/threads.txt), [`nodes.txt`](raw/sim_A_slam_explore/nodes.txt), [`components.txt`](raw/sim_A_slam_explore/components.txt)

| 프로세스 | 스레드 | RSS | 안의 노드 |
|---|---|---|---|
| `webots-bin` | 33 | 792 MB | (ROS 노드 아님) |
| `driver` (webots_ros2_driver) | 28 | 191 MB | `/LIMO` `/limo/limo_base` `/camera/camera` `/ydlidar_ros2_driver_node` `/controller_manager` `/diffdrive_controller` `/joint_state_broadcaster` |
| `ros2_supervisor.py` | 11 | 70 MB | `/Ros2Supervisor/Ros2Supervisor` |
| `robot_state_publisher` | 12 | 33 MB | `/robot_state_publisher` |
| `cartographer_node` | 18 | 55 MB | `/cartographer_node` |
| `cartographer_occupancy_grid_node` | 12 | 38 MB | `/cartographer_occupancy_grid_node` |
| `component_container_isolated` | **47** | 102 MB | `/nav2_container` + 컴포넌트 8: controller·smoother·planner·behavior·bt_navigator·waypoint_follower·velocity_smoother·lifecycle_manager_navigation (+ 숨은 노드 `global_costmap/global_costmap`·`local_costmap/local_costmap`·`bt_navigator_navigate_*_rclcpp_node`) |
| `smoke_mode` | 14 | 34 MB | `/smoke_mode` |
| `object_goto` | 14 | 34 MB | `/object_goto` (nav2 모드) |
| `frontier_explorer` | 14 | 35 MB | `/explore/frontier_explorer` |
| `rviz2` | 43 | 692 MB | `/limo_fire_rviz` (소프트웨어 렌더 llvmpipe 스레드 12 개 포함) |
| 숨은 노드 | | | `/launch_ros_<pid>`(launch 자신), `/transform_listener_impl_*` × 9 (TF 리스너마다 하나) |

노드 38 개, 토픽 99, 서비스 248(그중 파라미터 서비스 6 × 28 노드 = 168, 수명 주기 서비스 45), 액션 11 — `nodes.txt`·`topics.txt`·`services.txt`·`actions.txt`.

#### 🧪 sim B — 모드 ① 그대로: Cartographer + `explorer` (Nav2 없음) + `object_goto` explore

[`raw/sim_B_lowrl_explore/`](raw/sim_B_lowrl_explore). `sim.launch.xml slam:=true nav:=false explore:=false rviz:=false` + `explorer.launch.xml use_sim_time:=true require_rc:=false upper:=slot0 dry_run:=false` + `object_goto mode:=explore`.

| 프로세스 | 스레드 | RSS | 비고 |
|---|---|---|---|
| `explorer` | 12 | 50 MB | 실행기 1 + DDS. `~/enable` 로 켬 → `/cmd_vel` 직접 |
| `object_goto` | 12 | 29 MB | explore 모드: `/map`·TF·Nav2 클라이언트를 안 만듦 → sim A(nav2 모드) 14 스레드보다 2 개 적음 |
| `cartographer_node` | 18 | 48 MB | |

#### 🧪 sim C — 모드 ② 그대로: 저장 지도 + AMCL + Nav2 + 연기 모드 + object_goto nav2

[`raw/sim_C_patrol/`](raw/sim_C_patrol). `sim.launch.xml rviz:=false` (slam 기본 false → `map_server`·`amcl`·`lifecycle_manager_localization` 이 같은 컨테이너에 더 올라감, 로그 순서: localization `Managed nodes are active` 가 navigation 보다 10 s 먼저).

| 프로세스 | 스레드 | RSS | 비고 |
|---|---|---|---|
| `component_container_isolated` | **54** | 118 MB | sim A(47)보다 map_server·amcl·lifecycle_manager_localization 만큼 많음 |
| `smoke_mode` | 14 | 34 MB | 이제 `map → odom` 을 실제로 냄(AMCL 자세로) |
| `object_goto` | 14 | 33 MB | nav2 모드 |

한 일: 첫 순찰점(지도 3.8, 0.25)으로 `ros2 action send_goal --feedback /navigate_to_pose …` → 받아들임 → 피드백 여러 개 → `SUCCEEDED`(회복 0 번, 시뮬 15.9 s, 결과 `{}`) — [`nav_goal.txt`](raw/sim_C_patrol/nav_goal.txt). 주행 중 `/cmd_vel`·`/cmd_vel_nav` 모두 20.0 Hz — [`hz_during_goal.txt`](raw/sim_C_patrol/hz_during_goal.txt). `/smoke_mode/set true` → `success=True, message='연기 모드 켬'`, `/smoke_mode/active` true, `/speed_limit` 0.13 → `false` → `'평상시로 돌아옴'` — [`smoke_set.txt`](raw/sim_C_patrol/smoke_set.txt).

---

## (b) 노드별 — 역할·입력·출력·스레드·소유

QoS 는 `신뢰/지속` 으로: R = RELIABLE, BE = BEST_EFFORT, V = VOLATILE, TL = TRANSIENT_LOCAL. 🧪 표시는 sim 실측 `edges.txt`(발행 노드 → 토픽 → 구독 노드, QoS 포함 — rqt_graph 와 같은 정보)에서 확인한 것.

### b-1. 드라이버·기반 (남의 코드 + 우리 섀시)

| 노드 | 소유 | 받음 | 냄 | 파라미터 파일 | 스레드 |
|---|---|---|---|---|---|
| `/limo/limo_fire_base` | 우리(WeGo `limo_base` + 확장) | `/cmd_vel` Twist R | `/odom` R 50 Hz (frame odom→base_link, **TF 안 냄** `pub_odom_tf: false`), `/imu` R 100 Hz (`imu_link`), `/limo_status` 2 Hz, `/rc_state` Joy R 40 Hz, `/motor_states` 20 Hz, `/wheel_odometry` 50 Hz | `limo_fire_bringup/config/limo_base.yaml` | 시리얼 읽기 스레드 + 실행기 |
| `/ydlidar_ros2_driver_node` | WeGo | — | `/scan` LaserScan **BE** 10 Hz `laser_link` ±110°; 서비스 `/start_scan` `/stop_scan` (Empty) | `config/ydlidar.yaml` | main 고리 + SDK |
| `/camera/camera` (컨테이너 `/camera/camera_container`) | Orbbec | — | `/camera/color/image_raw` rgb8 640×480 R 30 Hz, `/camera/depth/image_raw` 16UC1 mm (컬러에 정렬) 30 Hz, 두 `camera_info`, TF `camera_link → camera_{color,depth}_frame → _optical_frame`, `/camera/depth_to_color` (TL) | `config/orbbec_dabai_dc1.yaml` | 실행기 + SDK 콜백 |
| `/ekf_filter_node_odom` | robot_localization (base) | `/odom`(x,y,yaw,vx,vy,vyaw, differential), `/imu`(yaw, 각속도, 가속도 x·y, 중력 제거) | `/odometry/filtered` 30 Hz, **TF `odom→base_link`(유일한 발행자)** | `config/ekf.yaml` (`frequency 30`, `two_d_mode`) | 실행기 1 |
| `/robot_state_publisher` | base | `/robot_description`(파라미터), `/joint_states` | `/tf_static`(R/TL): `base_link→{laser_link, imu_link, camera_link, inertial_link, ultrasonic_*}`, `/tf`: 바퀴 4 개, `/robot_description` (TL) | `limo_fire_description/urdf/limo_fire.urdf.xacro` | 실행기 (🧪 12 스레드) |
| `/joint_state_publisher` | base | — | `/joint_states` (실물 바퀴 0) | — | |
| `/esp32_bridge` | 팀원 | USB 시리얼 `/dev/esp32` 115200 (5 ms 타이머로 읽음) | `/ultrasonic/front_left`·`front_right` Range **BE**, frame `ultrasonic_*_link` | `limo_fire_esp32_bridge/config/esp32_bridge.yaml` | 실행기 1 (벽시계 5 ms 타이머) |

🧪 시뮬에서는 같은 이름의 노드가 Webots `driver` 프로세스 안에 있다: `/limo/limo_base`(냄 `/odom` `/imu` `/tf` — **시뮬에선 이것이 `odom→base_link` 를 냄, EKF 없음**), `/ydlidar_ros2_driver_node`(냄 `/scan` BE), `/camera/camera`(냄 컬러·깊이·IR·포인트 클라우드·압축, 구독자가 있을 때만 렌더), 바퀴는 `/diffdrive_controller`(ros2_control, 받음 `/cmd_vel` **BE**)·`/joint_state_broadcaster`(냄 `/joint_states`). 정답 카메라 `/LIMO` 가 `/gt/detections`·`/gt/mask`·`/sim/ground_truth` 를 낸다(우리 `limo_fire_sim` 플러그인).

### b-2. 지도·위치 (Cartographer · Nav2 위치 추정 · 연기 모드)

| 노드 | 소유 | 받음 | 냄 | 서비스 | 파라미터 |
|---|---|---|---|---|---|
| `/cartographer_node` | base | `/scan` BE, `/imu` BE, `odom` → remap `/odometry/filtered`(실물)·`/odom`(시뮬) BE, TF | TF `map→odom`(`published_frame = odom`, `provide_odom_frame = false`), `/submap_list`, `/trajectory_node_list`, `/constraint_list`, `/landmark_poses_list`, `/scan_matched_points2` | `/finish_trajectory` `/start_trajectory` `/write_state` `/get_trajectory_states` `/submap_query` `/trajectory_query` `/read_metrics` `/tf2_frames` | `limo_fire_navigation/config/cartographer_2d.lua` (`tracking_frame imu_link`, `pose_publish_period_sec 0.02`, 거리 0.1~8 m, IMU 씀) |
| `/cartographer_occupancy_grid_node` | base | `/submap_list` | `/map` OccupancyGrid **R/TL** 1 Hz (`-resolution 0.05 -publish_period_sec 1.0`) | — | 실행 인자 |
| `/map_server` (모드 ②) | Nav2 | 파일 `maps/*.yaml` | `/map` R/TL (한 번) | `/map_server/load_map`, 수명 주기 | `navigation.launch.xml` `map:=` |
| `/amcl` (모드 ②) | Nav2 | `scan_localization`(smoke_mode 가 넘김), `/map`, `/initialpose`, TF | `/amcl_pose`, `/particle_cloud` — **TF 는 안 냄** (`tf_broadcast: false`) | 수명 주기 | `nav2_params.yaml` amcl |
| `/smoke_mode` | 팀원 | `/scan` BE, `/imu`, `/map`, `/ultrasonic/*`, `/amcl_pose`, `/initialpose` | **TF `map→odom`**(모드 ②, `publish_map_odom`), `scan_localization`·`scan_obstacles` BE, `/speed_limit`, `/smoke_mode/active`(TL)·`pose`·`bump`·`wall_fix`, `/initialpose`(연기 모드에서 나올 때) | 서버 `/smoke_mode/set` SetBool, 클라이언트 `*/clear_entirely_*_costmap` | `limo_fire_navigation/config/smoke_mode.yaml` (타이머 4 개: TF·스캔·속도 1 s·벽 맞추기, 모두 벽시계) |

🧪 sim A 에서 `/scan` 구독자 = `cartographer_node`·`smoke_mode`·`limo_fire_rviz` (모두 BE), `/scan_obstacles`(smoke_mode BE) → local·global costmap, `/map`(occupancy_grid_node R/TL) → global_costmap·smoke_mode·object_goto·RViz(TL), **`frontier_explorer` 만 R/V**(Q3).
🧪 sim C(저장 지도): `/map` 은 `map_server`(R/TL, 한 번) → global_costmap·amcl·smoke_mode·object_goto, `/scan_localization` 은 `smoke_mode`(BE) → `amcl`, `/amcl_pose` 는 `amcl`(R/TL) → `smoke_mode` — 그래서 `map → odom` 을 smoke_mode 가 계산해 낸다. 벽시계로 `/scan` 6.5 Hz 와 `/scan_localization`·`/scan_obstacles` 6.5 Hz 가 같다 = 중계가 장마다 하나씩 넘김([`raw/sim_C_patrol/hz.txt`](raw/sim_C_patrol/hz.txt)).

### b-3. Nav2 (모드 ② ③, 그리고 Nav2 판 프론티어 탐사) — base, 한 컨테이너

`navigation.launch.xml` → `nav2_container` (`component_container_isolated`). 파라미터: `nav2_params.yaml` → `nav2_$ROS_DISTRO.yaml` → `params_override`(시뮬 `nav2_sim.yaml`). 🧪 수명 주기 모두 `active [3]` ([`lifecycle.txt`](raw/sim_A_slam_explore/lifecycle.txt)).

| 노드 | 받음 | 냄 | 서버 (액션·서비스) | 핵심 값 |
|---|---|---|---|---|
| `bt_navigator` | `/goal_pose`(RViz), `odom_topic`, TF | `/behavior_tree_log` | 액션 `/navigate_to_pose`, `/navigate_through_poses` | 기본 트리 `navigate_to_pose_w_replanning_and_recovery.xml` (1 Hz 다시 계획 + 회복) |
| `planner_server` | global costmap | `/plan` | 액션 `/compute_path_to_pose`, `/compute_path_through_poses`, 서비스 `/is_path_valid` | `nav2_smac_planner/SmacPlanner2D` |
| `controller_server` | local costmap, `odom_topic`, `/speed_limit` | `cmd_vel` → remap **`/cmd_vel_nav`**, `/local_plan`, `/lookahead_point`, `/lookahead_collision_arc`, `/received_global_plan` | 액션 `/follow_path` | RPP(Regulated Pure Pursuit) `desired_linear_vel 0.26`, lookahead 0.3~0.9, **20 Hz** |
| `smoother_server` | global costmap | `/plan_smoothed` | 액션 `/smooth_path` | SimpleSmoother |
| `behavior_server` | local costmap raw·footprint | **`/cmd_vel` 직접** (🧪 발행자 4 개) | 액션 `/spin` `/backup` `/drive_on_heading` `/wait` | 회복 동작 |
| `velocity_smoother` | `/cmd_vel_nav` | **`/cmd_vel`** | — | `smoothing_frequency 20`, 최대 [0.26, 0, 1.0] |
| `waypoint_follower` | — | — | 액션 `/follow_waypoints` | WaitAtWaypoint |
| `global_costmap/global_costmap` | `/map`(정적 층), `/scan_obstacles`(장애물 층), `/ultrasonic/*`(Range 층) | `/global_costmap/costmap`(+ raw·updates·footprint) | `clear_*`, `get_costmap` | `map` 틀, 0.5 Hz, 0.05 m, `robot_radius 0.2`, 팽창 0.45 m |
| `local_costmap/local_costmap` | `/scan_obstacles`, `/ultrasonic/*` | `/local_costmap/costmap` … | `clear_*`, `get_costmap` | `odom` 틀, 2 Hz, footprint 0.32 × 0.22 m |
| `lifecycle_manager_navigation` | `/bond`, `/diagnostics` | `/bond` | `manage_nodes`(ManageLifecycleNodes), `is_active`(Trigger) | `autostart: true`, 차례 controller → … → velocity_smoother |
| `lifecycle_manager_localization` (모드 ②) | | | 같음 | `map_server, amcl` |

🧪 측정(sim A): `/local_costmap/costmap` 1.0 Hz, `/global_costmap/costmap` 0.25 Hz (벽시계, 시뮬 0.8 배) — [`hz.txt`](raw/sim_A_slam_explore/hz.txt).

### b-4. 우리 탐사·인식·운용 노드

| 노드 | 받음 | 냄 | 서비스·액션 | 파라미터 | 스레드 |
|---|---|---|---|---|---|
| **`/explore/explorer`** (모드 ①) | `/scan` BE (= 걸음 시계), `/odom`(섀시 원값, EKF 아님), `/imu`, `/camera/depth/image_raw`+info BE, `/rc_state` BE(깊이 5), `/perception/objects` R/TL, `~/goal`(지시 글), TF `map→base_link` (스캔 stamp 시각) | `/cmd_vel` 10 Hz (**`dry_run:=false` 일 때만 발행자 생성**), `~/status` 1 Hz, `~/state`·`~/done`·`~/goal_status` (TL), debug 일 때만 `~/cmd_debug`(dry run 이면 늘)·`~/goal_marker`·`~/candidates`(+`_text`)·`~/decider_image` | 서버 `~/enable` SetBool (리모컨 오토일 때만 켜짐), HTTP `POST /v1/systemone` → llama-server | `limo_fire_explore/config/explorer.yaml` + 시뮬 공용 `train/explore/sim/config/env.yaml` | 실행기 1, 타이머 50 ms(입력 감시)·5 ms(TF 기다림 장) |
| `/explore/frontier_explorer` (Nav2 판 탐사, 기준선·BT `ExploreFrontiers`) | `/map` R/V, TF | `~/state`·`~/done` (TL), `~/status`, `~/frontiers` | 서버 `~/enable` SetBool, 클라이언트 액션 `/navigate_to_pose`·`/spin` | `config/frontier_explorer.yaml` (`occupied_thresh 65`) | 실행기 1 + TF 리스너 스레드, 200 ms 벽시계 타이머 |
| `/perception/object_detector` 📄 | `/camera/color/image_raw` BE (실행기 콜백, GPU 가 일하는 동안 최근 장만 쥠) | `~/detections` Detection2DArray, `~/mask` mono8(먼저), `~/image`(구독자 있을 때만) | — | `limo_fire_perception/config/yolo_coco.yaml`·`yolo_fire.yaml` | `/app` 실행기 1 (2 ms 타이머가 GPU 끝났나 물음 — 2026-11-18 전엔 GPU 작업 스레드) |
| `/perception/object_map` 📄 | `/perception/detections`+`/camera/depth/image_raw` (ApproximateTime 0.05 s), `~/mask`, depth `camera_info`, TF, `/clicked_point`, `~/save_to` | `~/objects` Detection3DArray **R/TL 10 Hz** (map 틀), `~/markers`, `~/summary`, `~/selected` | 서버 `~/clear` `~/save` `~/load` (Trigger) | 같은 파일 `object_map:` | 컨테이너 실행기 1, 20 ms 처리 타이머 |
| `/explore/object_goto` (저장 지도만) | `/explore/goal` String(사람 지시), `/perception/objects`, `/map` + TF | `/explore/goal_status`(TL) | 클라이언트 액션 `/navigate_to_pose` | `limo_fire_navigation/config/object_goto.yaml` + 별칭 `explorer.yaml` `/explore/**` | 실행기 1 + TF 스레드 |
| `/limo_fire_rviz` (RViz2) | 지도·스캔·경로·물체·상태 글 | `/goal_pose`(2D Goal Pose), `/initialpose`(2D Pose Estimate), `/clicked_point`(Publish Point) | 클라이언트 `/explore/frontier_explorer/enable` 또는 `/explore/enable` (탐사 패널) | `limo_fire_description/rviz/limo_fire.rviz`, `limo_fire_explore/rviz/explore.rviz` | Qt + ROS 스레드 |
| `/mission` 📄 (팀원) | `IsTopicAlive` 로 `/scan /odom /imu /odometry/filtered /camera/* /perception/detections`(generic 구독, SensorData), `/rc_state`, `/perception/objects`(R/TL), `/map`, `/amcl_pose`, TF | `/mission/events`(JSON 한 줄, R 깊이 50), `/mission/overlay`, `/cmd_vel`(StopRobot 0), `/nozzle/command`, `/mission/call119`(꺼짐) | 서버 `/mission/request_survivor_search` SetBool; 클라이언트 액션 `/navigate_to_pose`, 서비스 `/lifecycle_manager_*/is_active`(Trigger)·`/smoke_mode/set`·`/explore/frontier_explorer/enable`·`/pump/set`·`/speaker/evacuation` | `limo_fire_bt/config/limo_fire_bt.yaml` (+ `_sim.yaml`), 트리 `behavior_trees/limo_fire_bt.xml` | 한 스레드, 10 Hz 틱 |

🧪 (11-18 한 프로세스 전 옛 구조) sim B `explorer` 실측([`raw/sim_B_lowrl_explore/node_info.txt`](raw/sim_B_lowrl_explore/node_info.txt)), 지시 흐름([`instruction_flow.txt`](raw/sim_B_lowrl_explore/instruction_flow.txt)): `/explore/object_goto/goal "의자로 가"` → `/explore/object_goto/status "'의자로 가' → 클래스 chair · 지도에 0 개 — 판단 모델이 후보(그 물체 옆 · 프론티어)에서 고름"` → `/explore/goto_status "지시: 의자로 가 — …"`.

### b-5. ROS 밖 프로세스

| 프로세스 | 무엇 | 잇는 법 |
|---|---|---|
| llama-server (`127.0.0.1:8171`) | 판단 모델 d1-3B Q8 (LiquidAI), 우리 패치 2 개 | `explorer` 이 HTTP POST, 비차단 소켓, 한도 8 s. `-lm dio -ngl 99 -c 4096 -np 1`, `LLAMA_DECISION_IMAGES_LAST=1` |
| Groot2 | BT 모니터 | `mission` 이 ZMQ `:1667` 로 냄 |
| Webots (`webots-bin`) | 물리·센서 렌더 | `webots_ros2_driver` 가 IPC(`ipc://<port>/LIMO`)로 붙음 |

---

## (c) 서비스와 액션 — 서버·클라이언트·필드·언제 쓰나

### c-1. 액션 (오래 걸리는 일 — 목표·피드백·결과·취소)

| 액션 (타입) | 서버 | 클라이언트 | 목표 / 결과 / 피드백 | 언제 |
|---|---|---|---|---|
| `/navigate_to_pose` (`nav2_msgs/action/NavigateToPose`) | `bt_navigator` | 임무 BT `NavigateToWaypoint`·`SweepSmokeZone`, `object_goto`(nav2 모드), `frontier_explorer`, RViz 2D Goal Pose(`/goal_pose` 토픽 → bt_navigator 가 **자기 자신에게** `/navigate_to_pose` 목표를 보냄 — 🧪 sim C `node_info` 에서 bt_navigator 가 이 액션의 서버이자 클라이언트) | 목표 `pose`(PoseStamped, map 틀), `behavior_tree`(빈 글 = 기본 트리) / 결과 `std_msgs/Empty` (Humble — 성공·실패는 액션 상태 코드로) / 피드백 `current_pose`, `navigation_time`, `estimated_time_remaining`, `number_of_recoveries`, `distance_remaining` | 순찰점마다, 물체 옆 설 자리로, 프론티어로 |
| `/navigate_through_poses` | `bt_navigator` | (안 씀) | 목표 `poses[]` | |
| `/compute_path_to_pose` (`ComputePathToPose`) | `planner_server` | bt_navigator 의 트리 | 목표 `goal`, `start`, `planner_id`, `use_start` / 결과 `path`, `planning_time` | 1 Hz 다시 계획 |
| `/follow_path` (`FollowPath`) | `controller_server` | bt_navigator | 목표 `path`, `controller_id`, `goal_checker_id` / 피드백 `distance_to_goal`, `speed` | 경로 추종 20 Hz |
| `/smooth_path` | `smoother_server` | bt_navigator | | |
| `/spin` (`Spin`) | `behavior_server` | bt_navigator 회복, `frontier_explorer`(시작 때 프론티어 없으면) | 목표 `target_yaw`, `time_allowance` / 결과 `total_elapsed_time` / 피드백 `angular_distance_traveled` | 회복·둘러보기 |
| `/backup` `/drive_on_heading` `/wait` | `behavior_server` | bt_navigator 회복 트리 | | 🧪 sim A 에서 `backup failed`·`spin failed`(앞 막힘) 2 번씩 |
| `/follow_waypoints` | `waypoint_follower` | (안 씀 — BT 가 순찰점을 하나씩 줌) | | |

🧪 sim C 에서 순찰점으로 `ros2 action send_goal --feedback /navigate_to_pose …` 를 보내 받은 피드백·결과: [`raw/sim_C_patrol/nav_goal.txt`](raw/sim_C_patrol/nav_goal.txt).

### c-2. 서비스 (짧은 요청·응답)

| 서비스 (타입) | 서버 | 클라이언트 | 요청 / 응답 | 언제 |
|---|---|---|---|---|
| `/explore/enable` (`std_srvs/SetBool`) | `explorer` | RViz 탐사 패널, 사람 | `data` / `success`, `message` (리모컨 오토 아님·`/rc_state` 없음이면 `success: false` + 까닭) | 탐사 켜기·끄기 (늘 꺼진 채 뜸) |
| `/explore/frontier_explorer/enable` (SetBool) | `frontier_explorer` | RViz 패널, 임무 BT `ExploreFrontiers` | 같음. 끄면 Nav2 목표 취소 | Nav2 판 탐사 |
| `/smoke_mode/set` (SetBool) | `smoke_mode` | 임무 BT `SweepSmokeZone`, 사람 | `data: true` 연기 모드 / 저장 지도·AMCL 자세·TF 없으면 거절 + 까닭 | 연기 속 주행 |
| `/mission/request_survivor_search` (SetBool) | `mission` | 사람·웹 | `data` → 블랙보드 | 탐사 모드 요청 |
| `/lifecycle_manager_navigation/is_active` (`std_srvs/Trigger`) | lifecycle manager | 임무 BT `WaitForLifecycleActive` | — / `success` | 시작 때 Nav2 준비 확인 |
| `/lifecycle_manager_*/manage_nodes` (`nav2_msgs/ManageLifecycleNodes`) | lifecycle manager | 사람 | `command` STARTUP 0·PAUSE 1·RESUME 2·RESET 3·SHUTDOWN 4 / `success` | Nav2 전체 켜고 끄기 |
| `/<노드>/get_state` `change_state` … (`lifecycle_msgs`) | Nav2 서버마다 | lifecycle manager, `ros2 lifecycle` | | 수명 주기 |
| `/global_costmap/clear_entirely_global_costmap` (`nav2_msgs/ClearEntireCostmap`) 외 `clear_around`·`clear_except`·`get_costmap` | costmap | bt_navigator 회복 트리, `smoke_mode`(모드 바꿀 때) | | 장애물 지우기 |
| `/is_path_valid` (`nav2_msgs/IsPathValid`) | planner_server | | | |
| `/set_initial_pose` (`nav2_msgs/SetInitialPose`), `/reinitialize_global_localization`·`/request_nomotion_update` (`std_srvs/Empty`) 🧪 sim C | `amcl` | 사람(토픽 `/initialpose` 가 보통 길) | 자세+공분산 | AMCL 초기화·입자 다시 뿌리기 |
| `/map_server/load_map` (`nav2_msgs/LoadMap`: `map_url` → `map`, `result`), `/map_server/map` (`nav_msgs/GetMap`) 🧪 sim C | `map_server` | 사람 | | 지도 바꿔 읽기 |
| `/perception/save_objects` `load` `clear` (Trigger) 📄 | `object_map` | 사람. 저장은 `save_map.launch.xml` 이 토픽 `/perception/save_objects_to`(String 경로)로도 | | 물체 지도 파일 |
| `/map_saver/save_map` (`nav2_msgs/SaveMap`) | (map_saver_server 를 띄우면) | — 우리는 `map_saver_cli` 실행 파일을 씀 | `map_topic`, `map_url`, `image_format`, `map_mode`, `free_thresh`, `occupied_thresh` / `result` | 우리 `save_map.launch.xml` 은 CLI 두 번(src·install), 대기 10 s |
| Cartographer `/finish_trajectory` (`FinishTrajectory`: `trajectory_id` → `status`), `/write_state` (`filename`, `include_unfinished_submaps` → `status`, `.pbstream` 저장), `/start_trajectory`, `/get_trajectory_states`, `/submap_query`, `/trajectory_query`, `/read_metrics`, `/tf2_frames` | `cartographer_node` | 사람 (우리 launch 는 안 부름) | | 궤적 끝내기·상태 저장(나중에 위치 추정만 할 때). 우리는 `/map` 을 `map_saver_cli` 로 PGM 저장 |
| `/start_scan` `/stop_scan` (`std_srvs/Empty`) 📄 | YDLIDAR | 사람 | | 라이다 모터 |
| `/controller_manager/*` (list·load·switch …) 🧪 | Webots ros2_control | spawner | | 시뮬 바퀴 제어기 켜기 |
| `/Ros2Supervisor/spawn_urdf_robot` 등 🧪 | Webots supervisor | | | 시뮬 |
| 파라미터 서비스 6 종 (`get/set/list/describe_parameters`, `get_parameter_types`, `set_parameters_atomically`) | **모든 노드** | `ros2 param`, rqt | | 🧪 sim A 28 노드 × 6 = 168 개 |

**계획만 있고 받는 쪽 없는 것** (임무 BT, 📄): `/pump/set`(SetBool), `/speaker/evacuation`(SetBool), `/nozzle/command`(JointState 토픽), `/radar/people`, `/thermal/image_raw`, `/mission/call119` — 부품 대기, 이름·타입 잠정.

---

## (d) TF 나무 — 누가 내고 얼마나 자주

```
map ──(Cartographer: 모드 ① · ②의 지도 만들기 / smoke_mode: 모드 ② 저장 지도, AMCL 자세로 계산)──▶ odom
odom ──(EKF ekf_filter_node_odom 만 · 시뮬은 Webots /limo/limo_base)──▶ base_link
base_link ──(robot_state_publisher, /tf_static)──▶ laser_link · imu_link · camera_link · inertial_link · ultrasonic_front_left_link · ultrasonic_front_right_link
base_link ──(robot_state_publisher, /tf ← /joint_states)──▶ front/rear_left/right_wheel_link
camera_link ──(Orbbec 드라이버 · 시뮬 /camera/camera)──▶ camera_color_frame → camera_color_optical_frame
                                                     ──▶ camera_depth_frame → camera_depth_optical_frame
                                                     ──▶ (시뮬만) camera_ir_frame → camera_ir_optical_frame
```

| 변환 | 발행자 | 설정 주기 | 🧪 sim A 실측 (stamp 기준 = 시뮬 시각 Hz / 벽시계 Hz) | 🧪 sim B |
|---|---|---|---|---|
| `map → odom` | Cartographer (`pose_publish_period_sec 0.02`) · 모드 ② 는 `smoke_mode` | 50 Hz | 24.9 / 19.7 | 38.6 / 11.3 |
| `odom → base_link` | 실물 EKF 30 Hz · 시뮬 Webots `/limo/limo_base` | 30 (실물) · 50 (시뮬) | **50.0** / 39.5, 같은 stamp 중복 0 | 50.0 / 14.6 |
| `base_link → *_wheel_link` | robot_state_publisher | `/joint_states` 따라 | 12.9 / 10.2 | 12.4 / 3.6 |
| `camera_link → camera_*_frame`, `→ _optical_frame` | 카메라 드라이버 | 실물 `publish_tf` | 10.0 / 7.9 | 10.0 / 2.9 |
| `base_link → laser_link · imu_link · camera_link · inertial_link · ultrasonic_*` | robot_state_publisher `/tf_static` (R/TL, 한 번) | — | 6 개 확인 | 같음 |

- 근거: [`raw/sim_A_slam_explore/tf_frames.txt`](raw/sim_A_slam_explore/tf_frames.txt) (`/tf` 를 12 s 받아 부모→자식별로 셈). `odom → base_link` 같은 stamp 중복 0 = **발행자가 하나**뿐이다(🧪 `/tf` 발행자 목록엔 `diffdrive_controller` 도 있지만 이 변환은 안 낸다).
- `tf2_monitor` 는 ROS 2 에서 발행자 이름(authority)을 "no authority available" 로만 보여 준다([`tf2_monitor.txt`](raw/sim_A_slam_explore/tf2_monitor.txt)) — 그래서 위처럼 따로 셌다.
- 🤖 실물 녹화 `/tf` 9076 개 / 92.5 s = 98 Hz(모든 변환 합), `/odometry/filtered` 29.1 Hz.
- 🧪 sim C(저장 지도)의 `map → odom` 은 `smoke_mode` 가 낸다: 벽시계 19.6 Hz(설정 `tf_rate` 20, 벽시계 타이머), stamp 는 `now() + 1.0 s`(앞날짜, `transform_tolerance`) — 느린 시뮬 시계 때문에 같은 stamp 가 150/235 번 되풀이됨 ([`raw/sim_C_patrol/tf_frames.txt`](raw/sim_C_patrol/tf_frames.txt)).
- Cartographer 의 `map→odom` 이 설정 50 Hz 보다 적게 나온 것은 이 PC 부하(load 15) 때문으로 본다 — 실물 값은 따로 재지 않았다(⚠).

---

## (e) 세 임무 모드의 데이터 흐름 — 차례대로

### e-1. 모드 ① 첫 지도 탐사 (판단 모델 + A*, Nav2 없음)

| # | 무엇이 | 토픽·서비스·액션 |
|---|---|---|
| 0 | 사람이 `robot.launch.xml` → llama-server(`-lm dio`) → `app.launch.xml stack:=true perception:=true [dry_run:=false]` | |
| 1 | 섀시·라이다·카메라가 센서를 냄 | `/odom` 50 · `/imu` 100 · `/rc_state` 40 · `/scan` 10 · `/camera/color/image_raw`·`/camera/depth/image_raw` 30 Hz |
| 2 | EKF 가 `/odom`+`/imu` 를 합쳐 TF 를 냄 | `/odometry/filtered` 30 Hz, TF `odom→base_link` |
| 3 | Cartographer 가 스캔 맞추기로 위치를 고침 | 받음 `/scan` `/imu` `/odometry/filtered` → TF `map→odom` 50 Hz, `/submap_list` → `/map` 1 Hz(RViz·저장용) |
| 4 | `explorer` 이 늘 꺼진 채 뜸 (`~/state = idle`) | |
| 5 | 사람이 리모컨 SWB 를 위로(오토), RViz 패널 버튼 | `/explore/enable` SetBool `true` → `success` (오토가 아니면 거절) |
| 6 | 걸음마다(0.1 s, 스캔 한 장): 스캔 stamp 시각 TF `map→base_link` 를 얻으면 → 라이다·깊이로 자기 지도 갱신 → 프론티어 후보 최대 8 곳(판단 모델엔 6) | 받음 `/scan` `/odom` `/imu` 깊이, TF |
| 7 | 목표 1 m 앞·실패·후보 크게 바뀜·최대 10 s 때만 위층에 물음 | HTTP `POST /v1/systemone` (지도 PNG 288 px + 후보 글) → 답 `answers.pick.choice` = 글자 하나 (약 1.6 s) |
| 8 | 아래층이 로봇 지도 위 A* 경로를 따라 속도 명령 | `/cmd_vel` 10 Hz (`dry_run:=false` 일 때만), `~/cmd_debug`(dry run·debug 일 때) |
| 9 | 섀시는 SWB 오토일 때만 따름 | UART |
| 10 | (perception) YOLO → 물체 지도 | `/perception/mask`·`detections`(프로세스 안) → `object_map` ← 깊이 → `/perception/objects` R/TL 10 Hz → `explorer` (모두 `/app` 한 스레드 안) |
| 11 | (지시) 사람이 글을 줌 | `/explore/goal "의자로 가"` → explorer 가 별칭으로 클래스 chair 를 풀어 → 판단 모델 후보에 물체 옆 설 자리 → `/explore/goal_status "지시: … → 클래스 chair"` … `"도착: …"` |
| 12 | 끝: 대충 다 앎·지도 닫힘·진척 없음·시간 한도 | `~/state = done`, `~/done = true` |
| 13 | 저장 | `save_map.launch.xml name:=<이름>` → `map_saver_cli` 가 `/map` 을 받아 PGM+YAML, `/perception/save_objects_to` 로 `<이름>_objects.txt` |

안전: `/rc_state` 0.5 s 끊김·SWB 내림 → 0 한 번 내고 `paused`. 라이다·TF·오도메트리 0.3 s 끊김 → 명령만 0, 돌아오면 이어 감.

### e-2. 모드 ② 순찰 (저장 지도 + AMCL + Nav2 + 임무 BT)

| # | 무엇이 | 토픽·서비스·액션 |
|---|---|---|
| 0 | `robot.launch.xml` + `navigation.launch.xml map:=…` + `perception.launch.xml objects_file:=…` + `bt.launch.xml patrol_file:=…` | |
| 1 | `map_server` 가 지도를 한 번 냄, lifecycle manager 가 map_server·amcl → Nav2 서버들을 configure → activate | `/map` R/TL, `manage_nodes`, bond |
| 2 | 사람이 RViz 2D Pose Estimate | `/initialpose` → `amcl` |
| 3 | `smoke_mode` 가 스캔을 넘기고 `map→odom` 을 냄 | `/scan` → `scan_localization`(AMCL), `scan_obstacles`(costmap); `/amcl_pose` → TF `map→odom` |
| 4 | BT 시작: 깊이 정렬 확인 → Nav2 준비 확인 → 센서 주기 확인 → 리모컨 오토 확인 | `IsDepthAlignedToColor`, `/lifecycle_manager_*/is_active`, `IsTopicAlive`(`/scan` ≥ 7 Hz, `/odom` ≥ 35, `/imu` ≥ 70, `/odometry/filtered` ≥ 20, 영상 ≥ 20, 탐지 ≥ 10), `IsAutoMode`(`/rc_state`) — 실패면 `SENSOR_FAULT` 알림, 5 s 뒤 다시 |
| 5 | 순찰점을 하나씩 | `LoadPatrolRoute` → `GetNextWaypoint` → `NavigateToWaypoint` = 액션 `/navigate_to_pose` |
| 6 | bt_navigator 트리: 1 Hz 다시 계획 + 추종 + 회복 | `/compute_path_to_pose` → `/plan` → `/smooth_path` → `/follow_path` → `controller_server` → `/cmd_vel_nav` → `velocity_smoother` → `/cmd_vel`; 막히면 `clear_*costmap` → `/spin` → `/wait` → `/backup` (회복은 `/cmd_vel` 직접) |
| 7 | 순찰 중 늘 불·연기 탐지 확인 | `IsFireDetected`/`IsSmokeDetected`: `/perception/detections` 1 s 안 5 장 이상(score ≥ 0.5) + 위치는 `/perception/objects` |

### e-3. 모드 ③ 화재 대응 (② 위에서 BT 가 갈래를 탐)

| # | 무엇이 | 토픽·서비스·액션 |
|---|---|---|
| 1 | 불·연기 확인 → 순찰 멈춤 | Nav2 목표 취소, `StopRobot` → `/cmd_vel` 0 |
| 2 | 사건 판별: 이미 처리한 자리(3 m 안, 300 s 안)면 같은 사건 | `IsSameIncident`, `/perception/objects` |
| 3 | 새 사건이면 기록·알림·신고 | `CreateIncident`, `RecordFireLocation`(사진·위치 `/tmp/limo_fire_incidents`), `NotifyAdmin` → `/mission/events` JSON `HAZARD_DETECTED`, `Call119`(`call119.enabled: false` 면 로그만) |
| 4 | 고급 모드 고르기 | 연기 + 초음파 살아 있음 → 연기 모드; 작은불 + 펌프 서비스 + 서보 관절 → 물 모드; 불 근처 미탐사 + 위치 안정(`/amcl_pose` σ) → 탐사 모드 |
| 4a | 연기 모드 | `SweepSmokeZone`: `/smoke_mode/set true` → (AMCL 스캔 끊김, costmap 가상 스캔, `map→odom` 붙잡음, `/speed_limit` 0.13 m/s) → 불 둘레 4 곳 `/navigate_to_pose` → 사람 세기(`/radar/people` 없으면 카메라 탐지) → `/smoke_mode/set false`(나올 때 `/initialpose` 로 AMCL 다시 맞춤) |
| 4b | 물 모드 | `AimNozzle`(`/nozzle/command`, `/joint_states`) → `SprayWater`(`/pump/set` 3 s) → 열화상 온도 떨어짐 확인 → `StopPump` — 부품 대기 |
| 4c | 탐사 모드 | `ExploreFrontiers`: `/explore/frontier_explorer/enable true` → `~/state == done` 까지 (Nav2 판 프론티어 탐사) |
| 5 | 고급 모드가 없거나 실패 | `BasicFallback`: 안전거리 감시 60 s(`MonitorFromSafeDistance`) → `FALLBACK_TO_BASIC` 알림 → 순찰 재개 |

📄 BT 는 팀원이 Jazzy PC 에서 S1(순찰)·S2(큰불 → 신고 → 베이직 복귀 → 순찰 재개)·S3(같은 불 = 같은 사건)를 확인했다(2026-11-13~14, `limo_fire_bt/README.md`). S4~S7 은 아직.

---

## 주기 한눈에 — 🤖 실물 녹화

`ros2 bag info ~/limo_bags/1116_fuel_rl_1` (2026-11-16, 92.5 s, 모드 ① 실물 탐사) — [`raw/real_bags/1116_fuel_rl_1.info.txt`](raw/real_bags/1116_fuel_rl_1.info.txt)

| 토픽 | 개수 | Hz | 토픽 | 개수 | Hz |
|---|---|---|---|---|---|
| `/imu` | 8981 | 97.1 | `/camera/depth/image_raw` | 2698 | 29.2 |
| `/tf` (모든 변환 합) | 9076 | 98.2 | `/camera/color/image_raw/compressed` | 2687 | 29.1 |
| `/odom` | 4490 | 48.6 | `/odometry/filtered` | 2693 | 29.1 |
| `/wheel_odometry` | 4488 | 48.5 | `/motor_states` | 1796 | 19.4 |
| `/rc_state` | 3593 | 38.9 | `/joint_states` | 924 | 10.0 |
| `/trajectory_node_list` | 3034 | 32.8 | `/scan` | 900 | 9.7 |
| `/cmd_vel` = `/explore/cmd_debug` | 391 | 4.2 (켠 동안만 10 Hz) | `/limo_status` | 179 | 1.9 |
| `/map` | 101 | 1.09 | `/explore/status` | 39 | 0.42 (켠 동안 1 Hz) |

🧪 sim 벽시계 Hz(부하·소프트웨어 렌더로 실시간보다 느림): [`raw/sim_A_slam_explore/hz.txt`](raw/sim_A_slam_explore/hz.txt), [`raw/sim_B_lowrl_explore/hz.txt`](raw/sim_B_lowrl_explore/hz.txt), [`raw/sim_C_patrol/hz.txt`](raw/sim_C_patrol/hz.txt).

---

## (f) 현직자 질문 — 문제 → 원인 → 해결 → 다른 선택지의 문제

교수님 질문은 "X 를 어떻게 풀겠나", "Y 가 잘못되면 무엇이 깨지나" 꼴이다. 답은 **이 프로젝트에서 실제로 겪은 일**과 그 근거 파일로 한다.
표기: 📁 = 근거 파일·커밋, 🧪 = 이번 시뮬 실측(`raw/`), ⚠ = 기록에서 확인 못 한 것.

### 통신·QoS(Quality of Service, 서비스 품질)

**Q1. 라이다를 구독했는데 콜백이 한 번도 안 불린다. 무엇부터 보나?**
- 문제: `/scan` 은 오는데(`ros2 topic hz` 는 됨) 내 노드만 못 받는다.
- 원인: QoS 불일치. YDLIDAR 드라이버는 `/scan` 을 `SensorDataQoS`(BEST_EFFORT, 최선 전달) 로 낸다. 구독자가 RELIABLE(신뢰 전달) 을 요구하면 DDS 가 짝을 안 지어 조용히 아무것도 안 온다. 반대(발행 RELIABLE, 구독 BEST_EFFORT)는 맞는다.
- 해결: `ros2 topic info -v /scan` 으로 양쪽 Reliability·Durability 를 본다. 센서 구독은 `rclcpp::SensorDataQoS()` 로. 이 저장소의 `/scan` 구독자(Cartographer·smoke_mode·RViz·explorer)는 모두 BE 로 붙어 있다 🧪 `raw/sim_A_slam_explore/edges.txt`.
- 다른 선택지의 문제: 발행 쪽을 RELIABLE 로 바꾸면 Wi-Fi·부하에서 재전송이 쌓여 지연이 커진다. 센서는 "늦은 값보다 새 값" 이 맞다.
- ⚠ `/scan` QoS 로 실제 못 받은 사고 기록은 없다(원리 + 설정으로 답). 📁 `docs/sensor_topics.md` 라이다 절.

**Q2. 노드를 몇 번 껐다 켰더니 `/rc_state` 가 안 들어온다. `ros2 topic echo` 에는 나온다.**
- 문제: 2026-11-17 실물. 섀시 드라이버는 둔 채 `explorer` 을 여러 번 다시 띄우자 새 `explorer` 이 `/rc_state`(리모컨, 40 Hz) 를 못 받음 → 안전 규칙(0.5 s 못 받으면 멈춤)으로 로봇이 안 켜짐.
- 원인: **끝내 못 가림** (기록에 그대로 적음). DDS 발견·짝짓기가 재시작 사이에 꼬인 것으로 의심.
- 해결(`rc_watch.h`): ① 구독을 RELIABLE 깊이 20 → `SensorDataQoS().keep_last(5)`(BE 구독은 R·BE 발행자 모두와 맞음) ② 나이를 `steady_clock`(벽시계와 무관한 단조 시계) 으로 잼 — `use_sim_time` 과 무관 ③ **발행자가 있는데 `rc_resubscribe_s`(2 s) 동안 못 받으면 구독을 지우고 새로 만듦** ④ 못 받는 동안 `~/enable` 거절, `~/status` 에 나이·받은 수·발행자 수·다시 구독 수. 시험 `test/test_rc_watch.cpp`(받는 노드 5 번 다시 만들기).
- 다른 선택지의 문제: "그냥 다시 띄우기" 는 실물 시연 중 운에 맡김. 안전 타임아웃을 늘리면 리모컨을 내려도 로봇이 늦게 섬.
- 📁 커밋 2072d35, `src/limo_fire/limo_fire_explore/include/limo_fire_explore/rc_watch.h`, `limo_fire_explore/README.md` 실패 표.

**Q3. 나중에 켠 노드가 지도·상태를 못 받는다(늦게 온 구독자).**
- 원리: Durability TRANSIENT_LOCAL(TL, 마지막 값 보관) 발행자 + TL 구독자면 늦게 와도 마지막 값을 받는다. VOLATILE 구독자는 못 받는다.
- 이 프로젝트: `/map`(Cartographer occupancy grid, map_server), `/robot_description`, `~/state`·`~/done`·`~/goal_status`, `/perception/objects` 가 R/TL/1. RViz 탐사 패널은 켜자마자 `~/state` 를 알아야 버튼 색을 정하므로 TL 로 구독한다(`explore_panel.cpp`).
- 🧪 `frontier_explorer` 는 `/map` 을 `QoS(1).reliable()` = VOLATILE 로 구독한다(edges: `/explore/frontier_explorer[R,V]`). Cartographer 가 1 s 마다 다시 내므로 괜찮지만, 한 번만 내는 map_server `/map` 이었다면 켠 뒤 지도를 영영 못 받는다.
- 대가: TL 은 프로세스 안 전달(intra-process)과 같이 못 쓴다 → `/perception/objects` 는 같은 컨테이너라도 직렬화된다(`docs/runtime_io.md` 복사 표 5).

**Q4. 탐지(YOLO) 결과와 깊이 영상을 어떻게 짝짓나? 시각이 정확히 같지 않다.**
- 문제: 컬러와 깊이는 카메라 안에서 따로 찍혀 stamp 가 다르고, 탐지는 처리 시간만큼 늦게 나온다.
- 해결: `object_map` 가 `message_filters` ApproximateTime(근사 시각 맞춤), 줄 10, **시각 차 ≤ `max_sync_dt` 0.05 s**. 마스크는 탐지와 같은 컬러 header 를 쓰므로 최근 10 장에서 **같은 stamp** 를 찾는다(YOLO 가 `~/mask` 를 `~/detections` 보다 먼저 냄).
- 결과: 결정적 재생 `1116_fuel_rl_1` 에서 2687 장 중 58 장(2.2 %)만 짝 못 지음. 부하 걸린 PC 실시간 재생에서는 23~49 %(입력이 12~18 fps 로 듬성듬성) — 로봇에서 다시 잴 것.
- 다른 선택지의 문제: ExactTime 은 거의 영원히 짝이 안 남. 창을 넓히면(0.2 s) 로봇이 돌 때 다른 순간의 깊이를 써서 물체가 옆으로 번진다. 30 Hz 면 한 장 33 ms 라 0.05 s ≈ 1.5 장.
- 📁 커밋 1059059, `limo_fire_perception/README.md` "스레드·메모리", `config/yolo_coco.yaml` `max_sync_dt`.

**Q5. 처리보다 입력이 빠르면? 줄(queue) 은 어떻게 두나?**
- 원칙: 센서·인식은 **가장 최근 것만**. 밀린 장을 다 처리하면 지연이 계속 쌓인다(지연 = 줄 길이 × 처리 시간).
- 이 프로젝트: `object_detector` 는 GPU 가 일하는 동안 **가장 최근 장 하나만** 쥐고, 밀린 장은 버림. `object_map` 작업 줄 `max_pending` 10, 넘치면 **가장 오래된 장** 버림. 구독 QoS 깊이도 작게(SensorData 5).
- 다른 선택지의 문제: QoS 깊이를 크게(100) 하면 지연이 수 초가 되고 메모리도 큼. 반대로 깊이 1 + RELIABLE 은 느린 구독자가 발행자를 막을 수 있다.
- 📁 `limo_fire_perception/README.md`, 설계안 `limo_upper_keep/wt_rust/docs/onboard_rust.md`("가장 최근 하나" 칸).

### 실행기(executor)·스레드·TF

**Q6. 콜백 안에서 TF 를 기다리면 왜 안 되나? (단일 스레드 실행기에서 일어난 일)**
- 문제: 2026-10-28 인식 컨테이너를 한 스레드(`component_container`)로 돌리자 5 s 에 18~20 장을 버림 → 멀티스레드(`component_container_mt`)로 바꿨더니 0 장. 2026-11-17 실물 기동 로그에도 `component_container_mt` 가 찍혀 있다 📁 `raw/real_bags/explore_launch_1117.txt`.
- 진짜 원인(나중에 찾음): `object_map` 가 **콜백 안에서 TF 를 최대 0.2 s 기다림**. 한 스레드 실행기는 그동안 다른 콜백을 하나도 못 돌린다. 1116 재생에서 2500 콜백 중 71 번이 33 ms 넘게 막힘(최악 232 ms).
- 해결(1059059): 다시 한 스레드. 받기 콜백은 짝을 작업 줄에 **넣기만**, 20 ms 노드 시계 타이머가 **그 시각 TF 가 이미 있는 장부터** 처리(`canTransform` 비차단), 0.2 s 넘으면 버림. TF 리스너 스레드도 없앰. RSS 85 → 52 MB, 스레드 16 → 15(멀티스레드판은 112 MB·27 스레드).
- 다른 선택지의 문제: 멀티스레드 실행기는 증상만 숨긴다 — 콜백 그룹·뮤텍스가 필요하고, 막힌 스레드가 여전히 있으며 메모리·스레드가 는다. `lookupTransform(…, timeout)` 같은 기다림 API 는 TF 를 채울 별도 스레드가 있어야 동작한다(같은 스레드면 영원히 못 채움).
- 📁 커밋 05f3a3b → 1059059, `limo_fire_perception/README.md`.

**Q7. 탐사 노드(`explorer`)는 왜 한 스레드인가? TF 는 언제 보나?**
- 구조: `rclcpp::spin` 한 스레드. `/scan` 한 장 = 걸음 하나(0.1 s). 장과 그때의 `/odom`·`/imu`·깊이를 줄에 두고, **장 시각의** TF `map → base_link` 가 오면 처리(5 ms 벽시계 타이머가 비차단 `BufferCore::canTransform` 으로 다시 봄), `tf_timeout` 0.2 s 넘으면 그 장은 건너뜀. `TransformListener(..., spin_thread=false)` — 리스너 스레드 없음.
- 판단 모델 HTTP 도 막지 않음: 비차단 소켓을 걸음마다 한 단계씩(연결 → 보내기 → 받기). 예전 작업 스레드 + 뮤텍스 판을 이것으로 바꿈.
- 결과(커밋 4541e4c): 걸음 수 같음(889=889), TF 건너뜀 0, 평균 걸음 9.65 → 5.53 ms, 스레드 12 → 11.
- 왜 좋은가: 한 스레드면 경쟁 상태가 없고, 녹화를 넣으면 **같은 결과가 비트 단위로** 나와 시뮬(`low_explore_check`)과 실물 노드를 비교할 수 있다.
- 2026-11-18: 한 걸음 더 — 우리 노드 전부(탐사·YOLO·물체 지도)를 **프로세스 하나 `/app`, 실행기 스레드 하나**에. YOLO 도 스레드 없이: 영상 콜백이 GPU 에 걸고(비동기), GPU 에 건 장이 있을 때만 도는 2 ms 타이머가 `cudaEventQuery` 로 끝났나만 묻는다(기다리지 않음).
  PC 녹화 재생: 프로세스 3 → 1, 스레드 37 → 14, CPU 9.6 → 8.5 %, RSS 128 → 76 MB, 결정은 같음 📁 `docs/runtime_io.md` "앞 · 뒤".
- 📁 `limo_fire_explore/src/explore/explorer_node.cpp`(174·245·407·416 줄 근처), `docs/runtime_io.md` 스레드 표.

**Q8. 한 노드인데 스레드가 왜 14 개인가?**
- 🧪 `raw/sim_A_slam_explore/threads.txt`: `object_goto`·`frontier_explorer`·`smoke_mode` 처럼 실행기 하나짜리 노드도 14 스레드, Nav2 컨테이너(isolated) 47, RViz 43, Cartographer 18.
- 까닭: rclcpp 실행기 1 개 외에 **Fast DDS 가 받기·이벤트·발견·타이머 스레드를 따로** 띄운다(+ rcl 로깅). `component_container_isolated` 는 컴포넌트마다 실행기 스레드를 하나씩 더 둔다. TF 리스너는 기본으로 스레드 하나(+ 숨은 노드 `transform_listener_impl_*` — 🧪 `nodes.txt` 에 9 개).
- 그래서 "한 스레드 노드" 는 "콜백이 한 스레드에서 차례로 돈다" 는 뜻이지 프로세스 스레드가 1 개라는 뜻이 아니다.

**Q9. TF 가 "extrapolation into the future/past" 를 낸다.**
- 원인 후보: ① 요청 시각의 TF 가 아직 안 옴(발행 주기보다 최신을 물음) ② 두 컴퓨터 시계가 다름(PC 에서 RViz) ③ `use_sim_time` 이 노드마다 다름(벽시계 vs `/clock`) ④ TF 버퍼보다 오래된 시각.
- 이 프로젝트의 선택: 섀시 메시지 stamp 는 **시리얼을 받은 순간 `now()`**, 라이다는 드라이버 시각. `explorer` 은 **스캔 stamp 의 TF** 를 기다렸다 쓴다(최신 TF 를 쓰면 회전 중 스캔이 휘어 지도에 번짐). Cartographer `pose_publish_period_sec` 5 ms → 20 ms(50 Hz)로 TF 트래픽을 줄임(02ec95e).
- 🧪 sim B 에서 실제로 났다: `explorer` 이 `TF map → base_link (장 시각) 없음: Lookup would require extrapolation into the future. Requested time 25.1 but the latest data is at time 24.9 — 이 장 건너뜀` 을 9 번 냄(시뮬이 실시간의 0.3 배로 느려 Cartographer `map→odom` 이 스캔보다 늦게 옴). 설계대로 그 장만 건너뛰고 다음 장으로 감 — 막고 기다리지 않음 📁 `raw/sim_B_lowrl_explore/launch_log_summary.txt`.
- 반대 방향의 해법도 있다: `smoke_mode` 는 `map → odom` 을 **stamp = now() + 1.0 s 로 앞날짜**를 찍어 20 Hz 로 냄(`transform_tolerance` 1.0, AMCL 과 같은 방식) — 늦게 묻는 노드도 extrapolation 없이 받게. 🧪 sim C 에서 이 변환은 같은 stamp 가 반복(dup 150/235)되는데, 벽시계 20 Hz 타이머가 느린 시뮬 시계보다 자주 돌아서다 📁 `raw/sim_C_patrol/tf_frames.txt`, `smoke_mode_node.cpp` 407 줄.
- PC·로봇이면 `timedatectl` 로 NTP(시간 동기) 확인 — 로봇 NTP 설정은 ⚠ "확인 필요" 로 남아 있음. 📁 `limo_fire_bringup/README.md` (가) 절.

**Q10. 시뮬에서 Nav2 가 스캔을 다 버리거나 로봇이 0.125 m/s 로만 간다.**
- 원인: `use_sim_time` 을 Nav2 에 안 주면 costmap 이 `/clock` 시각 스캔을 "아주 옛날 것" 으로 보고 버림. `odom_topic` 이 실물 값(`/odometry/filtered`)이면 시뮬엔 그 토픽이 없어 컨트롤러가 속도 0 으로 보고 출력을 0.125 m/s 로 묶음.
- 해결: `nav2_sim.yaml` 이 `use_sim_time: True` 와 `odom_topic: /odom` 만 덮어씀(실물 파일은 그대로). 임무 BT 는 `/clock` 이 오기 전에는 틱하지 않음(시각 0 에서 시작한 기다림이 첫 `/clock` 에 수십 초 건너뛰어 바로 시간 초과가 되므로) 📁 `limo_fire_bt/src/mission_node.cpp`.
- 🧪 이번 시뮬은 Xvfb(화면 없는 X) + 소프트웨어 렌더라 실시간의 약 0.7 배로 돌았고(`/scan` 벽시계 7.1 Hz = 시뮬 10 Hz × 0.7), `use_sim_time` 덕에 Nav2·Cartographer 는 그대로 동작했다.

### 프로세스 안 전달·성능

**Q11. 같은 프로세스에 올리면 복사가 없어지나? (intra-process, zero-copy)**
- 조건: 같은 프로세스 + `use_intra_process_comms: true` + `unique_ptr` 로 publish + VOLATILE + KEEP_LAST. 그러면 포인터만 넘긴다.
- 이 프로젝트: `/app` 안 `object_detector → object_map` 의 `~/detections`·`~/mask` 만 복사 없음. 카메라 영상은 카메라가 **다른 컨테이너**라 DDS 로 온다.
  `/perception/objects` 는 같은 `/app` 안 `explorer` 로 가도 TL(임무 BT 가 TL 로 받음)이라 프로세스 안 전달을 끄고 직렬화한다(작은 글 10 Hz). 깊이는 `object_map`·`explorer` 가 따로 받아 두 번 역직렬화 —
  중계 노드로 한 번 받아 나눠도 Humble rclcpp 는 밖에서 온 메시지를 프로세스 안으로 넘길 때 한 번 복사해 이득이 없다.
- 카메라를 같이 못 올린 까닭: Orbbec 드라이버가 TRANSIENT_LOCAL 발행자(`/camera/depth_to_color` 등)를 만들어, intra-process 를 켜면 rclcpp 가 노드 생성을 거부. 또 YOLO 가 죽으면 카메라도 같이 죽는다(장애 격리). 📁 `robot.launch.xml` 주석, 커밋 05f3a3b.
- 다른 선택지의 문제: 다 한 컨테이너 = 한 노드가 죽으면 다 죽음, TL 토픽은 어차피 직렬화.

**Q12. 탐사 노드 계산은 48 µs 인데 왜 느린가?**
- 측정: 같은 녹화 재생에서 토픽 전달·역직렬화(deserialization, 풀기) **1 675 ms** 대 지도 계산 **48 µs/장**. 계산 자체는 C++ 51~56 µs, Rust 52~55 µs 로 같음.
- 결론: 병목은 계산이 아니라 **프로세스 사이 직렬화·DDS**. 그래서 "탐사·인식을 Rust 한 프로세스 + 실행기 한 스레드 + 최근 하나 칸" 설계를 따로 진행 중(카메라 SDK·llama-server·Cartographer·EKF 는 따로 프로세스로 남김).
- 다른 선택지의 문제: 알고리즘 최적화는 48 µs 를 줄여도 전체가 안 바뀜. 영상 640×480 깊이 614 KB × 30 장/초 × 구독 프로세스 수 만큼 복사된다(`docs/runtime_io.md` 복사 표).
- 📁 `limo_upper_keep/wt_rust/docs/onboard_rust.md`, 커밋 a3d4ea8. ⚠ 1 675 ms 원 측정 커밋은 못 찾음.

**Q13. Jetson 에서 판단 모델 서버가 "NvMap error 12" 로 죽는다. 메모리는 남아 보이는데?**
- 원인: llama-server 기본 mmap 이면 모델 파일 2.9 GB 가 **page cache** 에 남은 채 같은 내용을 GPU 버퍼로 또 잡는다. Jetson(Orin, CPU·GPU 메모리 공유)은 **GPU 할당 때 page cache 를 못 비워** KV 캐시 할당이 실패.
- 해결: `-lm dio`(DirectIO, page cache 우회) 로 띄우고, free 3 GB 밑이면 먼저 `sync; echo 3 | sudo tee /proc/sys/vm/drop_caches`. 인식 쪽도 줄였지만(GPU 4~7 MB·CPU 20~35 MB) "OOM 을 막기엔 작다" 고 기록 — RViz 를 로봇 밖에서 띄우고 TensorRT 작업 메모리를 줄이는 게 더 큼.
- 📁 커밋 f211b2b, `limo_fire_explore/README.md` "위층 판단 모델 띄우기".

**Q14. 판단 모델 답이 5 s 걸려 로봇이 서 있다. 어떻게 1.6 s 로 줄였나?**
- 측정: 예전 긴 꼴(384 px 지도 + 카메라 2×2, 2789 토큰) 5.0~5.6 s → 짧은 꼴(1140 토큰, 지도 288 px, 카메라 없음, 결정 기록 4) **처음 2.2 s / 이어서 1.58 s**.
- 원인 분석: 앞부분 처리(prefill) 가 토큰당 약 1.6 ms 라 **글 길이가 지배**. 그림은 (px/32)² 토큰(288 px → 81)이라 줄여도 이득 작음. 서버 패치: 그림을 글 뒤로(`LLAMA_DECISION_IMAGES_LAST=1`) + 결정 체크포인트 → 요청 사이 공통 앞부분을 다시 계산 안 함(0.6 s 절약).
- 양자화: Q4_K_M·Q5_K_M 이 Q8_0 보다 **느림**(prefill 전용이라 K-quant 풀기 비용) → Q8 유지.
- 기다리는 동안: 아래층은 지금 목표로 계속 감, 목표 1 m 앞에서 미리 물음(`decider_ask_lead_s`).
- 📁 커밋 6e22f93, `train/explore/upper_lora/README.md` "요청 꼴과 지연".

**Q15. 판단 모델 서버가 죽으면 로봇도 서나?**
- 아니다. 답 실패(시간 끝 8 s·꼴 틀림·없는 글자·후보 사라짐)가 `fail_max` 3 번 잇따르면 규칙(점수 1 등 후보)이 바로 대신 고르고 계속 탐사, 요청은 계속 보내 살아나면 다시 판단 모델로. 시뮬 주입 11 가지 × 2 장면 × 씨앗 2 = 44 판 확인.
- llama-server 를 ROS 노드가 아니라 **따로 프로세스(HTTP)** 로 둔 까닭: 요청 1.5~2.2 s 에 HTTP 비용은 ms 수준이고, 서버가 죽어도 탐사 노드는 안 죽는다.
- 📁 커밋 2072d35, `limo_fire_explore/README.md` 실패 표.

### 탐사·주행 설계

**Q16. 탐사는 왜 강화학습(RL)을 버리고 A* 로 갔나? (sim2real gap, 시뮬-실물 격차)**
- 문제: 아래층 RL 정책(low_w3 36000)이 실물에서 자주 서거나 제자리에서 돎 — `1116_fuel_rl_1` 걸음의 **77 % 가 v ≈ 0**. 시뮬에서도 기준 미달(사람 충돌 0.337 /100 m, 한도 0.10).
- 같은 날(2026-11-16) 실물 네 조합(fuel+RL · 가까운 후보+RL · 판단 모델+RL · 판단 모델+A*) 중 **판단 모델 + A*** 가 가장 잘 됨. 위층 RL 도 단순 규칙을 못 이김(157.4 s/0.740 대 150.2 s/0.771).
- 결정: 위층 = 판단 모델(LLM, 거대 언어 모델), 아래층 = 로봇 지도 위 A* 경로 추종. RL 코드는 `archive/rl/`.
- ⚠ 실물 RL 대 A* 를 같은 표로 잰 수치는 기록에 없음. A* 실물 값은 `1116_decider_rule_2`: v≈0 몫 0.35, |v| 0.42 m/s, 아는 넓이 65/93/102/111 m² (10/20/40/70 s) — `train/explore/sim/analysis/real_limo.md`.
- 다른 선택지의 문제: 시뮬을 더 맞추는(domain randomization) 길은 시연까지 시간이 없음, A* 는 실물에서 바로 검증 가능.

**Q17. 뒤로 물러나는 회복 동작이 많다. (물러나기 몫)**
- 기록: `real_obj_1` 64 s 중 **20 %** 가 −0.05 m/s 물러나기(복구 BackUp 6 s 한 번 + RPP 충돌 검사에 걸릴 때마다 물러남), 그때 라이다 최소 거리 0.53~0.90 m. ⚠ "24 %" 라는 수치는 기록에서 못 찾음(가까운 것: 시뮬에서 RL 이 닫힘 시간 23~24 % 짧았음 — `archive/rl/train/explore/mdp/README.md`).
- 🧪 이번 Webots 실행에서도 Nav2 회복이 그대로 보였다: `RegulatedPurePursuitController detected collision ahead!` 38 번, `Starting point in lethal space` 10 번, `spin failed`·`backup failed`(앞이 막혀 Collision Ahead 로 끝) 각 2 번 — `raw/sim_A_slam_explore/launch_log_summary.txt`.
- 🧪 그 결과 Nav2 판 프론티어 탐사(`frontier_explorer`)는 이번 실행에서 `탐사 끝 (all_blacklisted) — 214.9 s, 목표 29 · 도착 0 · 포기 5`, 지도 29.6 m² 로 끝났다 — 목표마다 Nav2 가 포기해 블랙리스트에 쌓임 📁 `raw/sim_A_slam_explore/explore/frontier_explorer_log.txt`. 같은 미로에서 저장 지도 순찰 목표(sim C)는 회복 0 번으로 15.9 s(시뮬 시각)에 SUCCEEDED 📁 `raw/sim_C_patrol/nav_goal.txt` — SLAM 중(지도가 계속 바뀌고 미지 칸이 많음)이 더 어렵다.
- 우리 노드도 같은 미로에서 막혔다: 🧪 sim B `explorer` 이 켠 지 30 s 만에 `paused — 닿음 (신호 4: 멈춤)` — 명령을 내는데 바퀴·자이로가 안 움직이면 "닿음" 으로 보고 멈춘 뒤 사람이 다시 켜야 함(`contact_resume_s` 0). 시뮬이 느린 탓도 있어 실물 판단은 아님 📁 `raw/sim_B_lowrl_explore/launch_log_summary.txt`.
- 원인: 좁은 덕트 미로 + 팽창 반경 0.45 m(`inflation_radius`)·`cost_scaling` 때문에 로봇 중심이 치명 칸(lethal) 에 들어가 계획이 실패 → 회복 트리(clear costmap → spin → wait → backup).
- 해결 방향: 우리 A* 추종기는 "막히면 뒤로 물러나지 않고 그 목표를 빼고 다시 묻기"(`task_sup.h` Goto: stall 이면 못 감). Nav2 쪽은 팽창·footprint 값을 미로 폭에 맞춰야 함.
- 다른 선택지의 문제: 팽창을 너무 줄이면 벽을 긁고, 회복을 끄면 막힌 채 영원히 섬.

**Q18. Nav2 를 쓰지 왜 자체 추종기를 만들었나? 언제 Nav2 를 쓰나?**
- Nav2 를 쓰는 곳: **저장 지도 순찰·화재 출동**(모드 ②③) — `map_server` + AMCL + costmap + 계획·추종·회복, 임무 BT 가 `navigate_to_pose` 로 부름. 물체로 가기 nav2 모드, 프론티어 탐사 노드(`frontier_explorer`)도 Nav2 를 부름.
- 자체 A* 를 쓰는 곳: **첫 지도 탐사**(모드 ①, `explorer`). 까닭: 시뮬(`train/explore`)과 **같은 C++ 코드**라 녹화를 넣으면 시뮬 결과와 비트 단위로 같음 → 시뮬에서 수천 판 잰 것을 실물에 그대로 믿을 수 있음. 판단 모델 후보(프론티어 6 곳)·물체 후보·사람 따라가기를 같은 지도에서 다룸. 깊이 장애물을 0.1 s 마다 지도에 더해 라이다보다 낮은 가구도 피함.
- 다른 선택지의 문제: Nav2 를 쓰면 시뮬 쪽에 Nav2 를 흉내 내야 하고 비트 비교가 불가능. 반대로 자체 추종기는 Nav2 의 회복·costmap 층·속도 평활을 다시 만들어야 함.
- 둘을 **동시에 켜면 `/cmd_vel` 발행자가 둘** → 명령이 섞임. 그래서 `app.launch.xml` 은 Nav2 를 안 띄운다(Q34).

**Q19. 탐사 노드는 Cartographer `/map` 을 왜 안 읽나?**
- `explorer` 은 라이다·깊이로 **자기 지도**를 만들고 Cartographer 에서는 TF `map → odom`(위치 보정)만 받는다. `/odom` 도 EKF 결과가 아니라 섀시 원래 값.
- 까닭: 시뮬 관측과 같게(시뮬에는 Cartographer 가 없음). 1 Hz `/map` 을 기다리면 0.1 s 걸음과 안 맞음. 깊이 장애물 층을 더해야 함.
- 대가: 지도가 두 벌(Cartographer = 저장·RViz 용, 노드 지도 = 결정용). 저장은 Cartographer 쪽(`save_map.launch.xml`).
- 📁 `docs/runtime_io.md` 첫 절.

**Q20. 프론티어 탐사(Nav2 판)에서 로봇이 벽을 무시하고 7 분 넘게 돌았다.**
- 원인: Cartographer occupancy 값이 벽에서도 최대 87 정도인데 costmap 정적 층 `lethal_cost_threshold` 기본 100 → **전역 costmap 에 벽이 하나도 없음**.
- 해결: 문턱 65(= Cartographer `occupied_thresh` 0.65 와 같게, 2026-11-11). `frontier_explorer.yaml` `occupied_thresh: 65` 도 같은 값.
- 교훈: 지도 만든 쪽과 읽는 쪽의 "막힘" 정의를 맞춰야 한다. 📁 `docs/simulation.md` Nav2 표.

### TF·좌표

**Q21. `odom → base_link` 를 두 노드가 내면?**
- 문제: TF 는 "한 변환 = 발행자 하나" 가 규칙(REP-105). 둘이면 버퍼에 번갈아 들어가 로봇이 순간이동하듯 튀고 AMCL·Cartographer 가 흔들린다.
- 이 프로젝트: 실물 `odom → base_link` 는 **EKF 만**(`limo_fire_base` `pub_odom_tf: false`). `map → odom` 은 Cartographer(지도 만들기) **또는** `smoke_mode`(저장 지도, AMCL `tf_broadcast: false` 로 두고 `/amcl_pose` 로 계산) 하나만. 시뮬은 Webots 플러그인 `/limo/limo_base` 가 EKF 자리.
- 🧪 `raw/sim_A_slam_explore/edges.txt` 의 `/tf` 발행자: `robot_state_publisher`·`cartographer_node`·`smoke_mode`·`/limo/limo_base`·`/camera/camera`·`/diffdrive_controller`. smoke_mode 는 SLAM 때 `publish_map_odom: false` 라 실제로 안 낸다(발행자만 있음). `diffdrive_controller` 가 `odom→base_link` 를 같이 내는지는 `/tf` 를 직접 세어 확인했다 — 같은 stamp 중복 0 이라 발행자는 하나((d) 표). sim C 에서는 `amcl` 도 `/tf` 발행자 목록에 있지만 `tf_broadcast: false` 라 실제로는 안 내고, `map → odom` 은 `smoke_mode` 혼자 낸다. **"발행자 목록에 있다" ≠ "그 변환을 낸다"** — 발행자는 노드가 broadcaster 를 만들기만 해도 생긴다.
- 확인법: `ros2 run tf2_ros tf2_monitor`(변환별 발행자·주기), `ros2 run tf2_tools view_frames`.

**Q22. 센서 위치를 launch 에 static_transform_publisher 로 흩어 두면?**
- 문제: WeGo 원본은 카메라 틸트용 static TF 3 개를 launch 에 두었다. 위치 값이 여러 곳 → 틀려도 못 찾음.
- 해결: 장착 위치는 전부 URDF(xacro) 한 곳, `robot_state_publisher` 가 `/tf_static`(TL) 로 냄. 카메라 `camera_link` 아래 color·depth 프레임만 Orbbec 드라이버가 공장 값으로 냄(`camera_driver_tf:=true`). `base_link → camera_link` 은 실측 (0.15, −0.005, 0.06), 틸트 `camera_pitch` 0.1152 rad.
- 📁 `limo_fire_bringup/README.md` "WeGo 와 다른 점", `limo_fire_description/urdf/limo_fire.urdf.xacro`.

**Q23. 교수님 Webots launch 로 띄우면 두 번에 한 번 TF 가 안 나왔다.**
- 원인: 드라이버가 로봇 모델(URDF)을 `robot_state_publisher` 에 **1 초만 기다려** 넘김 → 늦으면 RSP 가 모델 없이 떠 TF 없음(시작 순서 경쟁).
- 해결: 우리 `sim_robot.launch.xml` 이 Webots·드라이버를 직접 띄우고 `set_robot_state_publisher:=false`, 우리 URDF 로 RSP 를 띄움. Nav2 는 TF·`/clock`·지도가 올 때까지 스스로 기다리므로 **시간 지연(sleep)을 넣지 않음**.
- 🧪 이번 실행에서도 Nav2 가 처음 몇 초 `Timed out waiting for transform from base_link to odom` 을 반복하다 Webots 가 뜨자 `Managed nodes are active` — 기다림이 정상 동작.
- 📁 커밋 4698491, `docs/simulation.md`.

### Nav2 수명 주기·파라미터·launch

**Q24. Nav2 노드는 왜 떴는데 일을 안 하나? (lifecycle, 수명 주기 노드)**
- Nav2 서버는 managed node(관리형 노드): unconfigured → `configure` → inactive → `activate` → active. `lifecycle_manager_navigation`(`autostart: true`)이 정해진 차례(controller → smoother → planner → behavior → bt_navigator → waypoint → velocity_smoother)로 올리고 bond(살아 있음 신호)로 감시, 하나가 죽으면 전부 내림.
- 🧪 `raw/sim_A_slam_explore/lifecycle.txt`: 모두 `active [3]`. 로그 `Managed nodes are active`.
- 임무 BT 는 `WaitForLifecycleActive`(`/lifecycle_manager_*/is_active`, std_srvs/Trigger) 로 확인한 뒤 시작 — 이것이 없을 때 시작마다 `LIDAR_TOPIC_LOST` 한 번·첫 순찰점 거절(커밋 a1fcce7).
- 다른 선택지의 문제: 고정 sleep 은 PC·부하마다 틀림.

**Q25. Nav2 를 한 프로세스에 몰아넣었다. 장단점은? (composition, 조합)**
- `navigation.launch.xml`: `component_container_isolated` 하나에 controller·smoother·planner·behavior·bt_navigator·waypoint_follower·velocity_smoother·lifecycle_manager(+ 저장 지도면 map_server·amcl). 🧪 `components.txt` 8 개.
- isolated = 컴포넌트마다 실행기 스레드 하나 → 한 서버의 긴 콜백이 다른 서버를 안 막음(🧪 47 스레드).
- 장점: 프로세스 수·메모리 감소, 같은 프로세스 토픽은 intra-process 가능. 단점: 하나가 segfault 면 Nav2 전체가 죽음, 로그가 한 줄기로 섞임.
- 주의: launch 에서 `load_composable_node` 는 `if` 를 무시 → `<group if=…>` 로 감쌈. Nav2 컨테이너는 `/tf`→`tf` remap 을 컴포넌트마다 줌.

**Q26. 실물·시뮬·ROS 판(Humble/Jazzy)마다 파라미터가 다르다. 파일을 어떻게 나누나?**
- 층: `nav2_params.yaml`(실물 기준, WeGo 값) → `nav2_$(env ROS_DISTRO).yaml`(판 차이: Jazzy 는 플러그인 이름 `::`, TwistStamped `/cmd_vel`, 초음파 층 뺌) → `params_override`(시뮬 `nav2_sim.yaml`: `use_sim_time`·`odom_topic`·AMCL 초기 자세만). 뒤 파일이 앞을 덮는다.
- 임무 BT 도 같은 방식: `limo_fire_bt.yaml` 위에 `limo_fire_bt_sim.yaml`(토픽 바꿔 읽기 `/odometry/filtered:=/odom`, `/perception/detections:=/gt/detections`, 주기 문턱 8 Hz, 리모컨 검사 끔).
- 규칙: 주기·포트·임계값은 `config/*.yaml`, 코드에 박지 않음(CLAUDE.md). 📁 커밋 fc84923.

**Q27. XML launch 에서 include 인자가 다른 include 로 새어 나갔다.**
- 문제: Humble 에서 `<include>` 에 준 `<arg>` 가 바깥 범위로 새어 나와 **다음 include 의 같은 이름 인자 값을 바꿈**(예: `rviz:=false` 가 다른 곳까지).
- 해결: include 를 하나씩 `<group>` 으로 가둠(`sim.launch.xml` 주석). 📁 커밋 571f1a0.

**Q28. `ros2 launch` 한 번에 무엇이 뜨는지 어떻게 아나?**
- `ros2 launch … --show-args`, `--print`(설명), 실행 중엔 `ros2 node list`·`ros2 component list`·`ps -T`. 이 문서의 (a) 표와 `raw/*/nodes.txt`·`components.txt`·`threads.txt` 가 그 결과다.
- 숨은 노드: `launch_ros_<pid>`(launch 자신), `transform_listener_impl_*`(TF 리스너), `bt_navigator_navigate_to_pose_rclcpp_node`(BT 플러그인용 보조 노드)는 파라미터 서비스가 없어 `ros2 param list` 가 시간 초과난다.

### 하드웨어·드라이버·환경

**Q29. 깊이로 잰 물체 위치가 엉뚱하다(카메라 설정을 하나 껐을 뿐인데).**
- 원인: 포인트 클라우드를 끄자(02ec95e) Orbbec 드라이버가 `depth_registration` 도 같이 꺼 **깊이가 컬러에 정렬 안 됨**: 640×480/`camera_color_optical_frame`/fx 489.39 → 640×400/`camera_depth_optical_frame`/fx 478.02. 탐지 상자(컬러 좌표)가 깊이의 엉뚱한 줄을 읽고, 마스크 측정은 크기가 달라 조용히 꺼짐. 탐사 노드에선 바닥이 장애물로 찍혀 33 s 에 거짓 "탐사 끝".
- 해결: `depth_registration: true`(정렬만 켬). 대가: 깊이 29.9 → 29.0 Hz, 드라이버 CPU 15 → 19 %. 임무 BT 는 시작 때 `IsDepthAlignedToColor` 로 확인.
- 📁 커밋 4207e70, `docs/sensor_topics.md` 주의 1.

**Q30. 지도 저장이 가끔 실패한다(대시보드 "맵 저장").**
- 원인: `map_saver_cli` 기본 대기 2 s, `/map` 은 1 s 마다(Cartographer) → 발견·첫 메시지까지 2 s 를 넘기면 실패.
- 해결: `save_map_timeout` 10 s, src·install 두 곳에 저장(다시 빌드 없이 navigation 이 새 지도 읽음), 물체 지도도 `/perception/save_objects_to` 로 같이 저장. 📁 `save_map.launch.xml`, 커밋 d5068ff.

**Q31. 카메라가 30 Hz 인데 YOLO 가 21 ms 나 걸린다(장을 쉬지 않고 연달아 넣으면 10.3 ms).**
- 원인: 33 ms 마다 한 장이면 장 사이에 GPU 클럭이 내려감(DVFS, 동적 클럭).
- 해결: `robot.launch.xml` 이 `sudo -n jetson_clocks` 로 클럭 고정 → 10.8 ms. 대가: 대기 전력. `max_clocks:=false` 로 끔. 📁 커밋 05f3a3b.

**Q32. 깊이 주기가 19~25 Hz 로 떨어진다.**
- 원인: 로봇 CPU 부하(Webots·RViz 와 같이, 정렬 켬). 안 쓰는 포인트 클라우드 2 개·IR 을 끄자 19 → 30 Hz, 드라이버 CPU 42 → 35 %.
- 대응: 쓰는 스트림만 켬, RViz 는 PC 에서(젯슨 RViz 106 % → 10 fps 제한 76 %), 구독은 BE·최근 것만, BT 의 `IsTopicAlive` 문턱(실물 20 Hz, 시뮬 8 Hz) 으로 감시. 📁 `docs/sensor_topics.md`, `docs/simulation.md`.
- 받는 쪽에서 빠지는 것도 있다: Fast DDS 공유 메모리 칸 기본 512 KB 가 영상 한 장(깊이 614 KB · 컬러 921 KB)보다 작아, 받는 쪽이 늦으면 칸이 덮여 그 장이 버려진다.
  이 PC 합성 시험에서 기본 칸은 깊이 1~3 % 빠지고, 두 프로세스를 코어 2 개에 몰면 컬러가 14.5~18.5 Hz, 1 개면 거의 0 장. 칸 4 MB(`limo_fire_bringup/config/fastdds.xml`, `env.sh` 가 고름)면 모두 30 Hz.
  보내는 쪽 칸이 기준이라 카메라도 `env.sh` 셸에서 띄워야 한다. `ROS_LOCALHOST_ONLY=1` 을 같이 켜면 rmw 가 기본 칸을 더 붙여 섞인다 → 로봇은 UDP 를 127.0.0.1 로 묶은 `fastdds_localhost.xml`. 📁 `docs/runtime_io.md` "DDS 설정".

**Q33. PC 에서 로봇 토픽이 하나도 안 보인다.**
- 차례로: ① 로봇 env.sh 가 이 컴퓨터 안에서만 통하는 DDS 설정(`fastdds_localhost.xml` = WeGo 대시보드의 `ROS_LOCALHOST_ONLY=1` 과 같은 범위)을 고름 → 로봇 터미널마다 `FASTRTPS_DEFAULT_PROFILES_FILE` 을 `fastdds.xml` 로 하고 노드 다시 켜기 ② `ROS_DOMAIN_ID` 같게(이번 시험도 다른 사람 시뮬과 안 섞이게 93 을 씀 🧪 `00_when.txt`) ③ 방화벽(UDP 7400 + 250×도메인) ④ 같은 서브넷 — Tailscale 은 멀티캐스트를 안 넘겨 Fast DDS 발견이 안 됨 → Discovery Server(`fastdds discovery`, `ROS_DISCOVERY_SERVER`) ⚠ 아직 안 해 봄.
- 📁 `limo_fire_bringup/README.md` (가) 절.

**Q34. 리모컨·탐사 노드·Nav2 가 모두 `/cmd_vel` 을 내면? 로봇을 어떻게 안전하게 세우나?**
- 구조: 섀시는 리모컨 SWB 가 위(오토)일 때만 `/cmd_vel` 을 따름(하드웨어 쪽 안전). `explorer` 은 `dry_run: true` 가 기본이라 `/cmd_vel` 발행자 자체를 안 만듦(`~/cmd_debug` 에만), 켜기는 `~/enable`(SetBool) 이고 리모컨 오토가 아니면 거절, `/rc_state` 0.5 s 끊기면 멈춤(0 한 번)·꺼짐.
- Nav2 판: controller → `cmd_vel_nav` → velocity_smoother → `/cmd_vel`. 🧪 그런데 회복 동작(behavior_server 의 spin·backup)은 **smoother 를 거치지 않고 `/cmd_vel` 에 바로** 낸다(edges: `/cmd_vel` 발행자 behavior_server × 4 + velocity_smoother).
- 겹침 방지: 탐사 launch 는 Nav2 를 안 띄움, 임무 BT 는 멈출 때 `haltTree()` 로 Nav2 목표 취소·`StopRobot` 0 발행. 정석 대안은 `twist_mux`(우선순위 섞기) — 이 저장소엔 없음.

**Q35. conda 를 켠 터미널에서 Webots 시뮬 `/clock` 이 사라졌다.**
- 원인: conda 의 Python 3.14 가 ROS 2 Humble 의 `rclpy`(python3.10) 를 깨뜨려 Webots supervisor(파이썬 노드)가 죽음.
- 해결: `env.sh` 가 conda 를 PATH 에서 뺌. 📁 커밋 c26f42b, `docs/progress.md`.

**Q36. 드라이버 원본 버그는 왜 안 고쳤나?**
- `/limo_status` stamp 가 1970 년(초를 나노초 칸에), IMU 바이트 이름이 roll·pitch 뒤바뀜(그러나 `setRPY(pitch, roll, yaw)` 로 다시 뒤바꿔 **나오는 쿼터니언은 REP-103 대로**, 8 녹화 1 mrad 안 확인).
- 원칙: `src/external/` 는 고치지 않는다(출처·커밋 표). 바꿀 게 있으면 우리 판(`limo_fire_base` = WeGo limo_base + 리모컨·모터 토픽 더함)을 만든다. 쓰는 쪽이 영향 없으면 기록만. 📁 커밋 6f59512.

### 팀원 모듈(BT·연기 모드)·운용

**Q37. 임무 BT 는 왜 Nav2 bt_navigator 플러그인이 아니라 따로 된 노드인가?**
- 결정(D1, `docs/bt_sim_plan.md`): 임무 트리(순찰·사건·신고·고급 모드)와 주행 트리(Nav2 내부 계획·회복)를 나눔. `mission` 실행 파일이 BehaviorTree.CPP 4.x 트리를 `tick_hz` 10 Hz 로 틱, 주행은 `navigate_to_pose` 액션 **클라이언트**로 부름. Groot2 모니터(포트 1667)·`.btlog` 재생.
- 실행 구조: `SingleThreadedExecutor` 로 `spin_some()` 후 `tickOnce()`, 틱 사이에도 5 ms 씩 콜백을 받음. 액션·서비스 호출은 틱마다 상태만 보는 비차단 래퍼(`ros_helpers.cpp`, 서버가 `server_timeout_s` 5 s 안에 없으면 실패). 멈출 때 `haltTree()` → Nav2 목표 취소·펌프·연기 모드·탐사 끔.
- 센서 건강: `IsTopicAlive` 가 **타입 상관없이** generic subscription(일반 구독)으로 주기를 잼(`/scan` 7 Hz, `/odom` 35, `/imu` 70, `/odometry/filtered` 20, 영상 20 …).
- ⚠ 이 PC(Humble)에 BehaviorTree.CPP 4.x 가 없어 이번엔 못 띄움 — 확인은 팀원이 Jazzy PC 에서 S1~S3(2026-11-13~14). 로봇 Jetson 에 4.x 있는지도 아직 확인 안 함.

**Q38. 연기 속에서는 라이다를 못 믿는다. 위치 추정을 어떻게 바꾸나? (팀원 연기 모드)**
- `smoke_mode` 노드가 **늘 떠서 길목**을 쥔다: 평상시 `/scan` → `scan_localization`(AMCL)·`scan_obstacles`(costmap) 그대로 넘기고, `map → odom` 을 `/amcl_pose` 로 계산해 냄. 연기 모드(`/smoke_mode/set` true)면 AMCL 스캔을 끊고, costmap 에는 저장 지도에서 광선을 쏜 가상 스캔을 주고, `map → odom` 을 붙잡아(`hold`) 위치는 EKF(바퀴+IMU)만 따라감, 속도 한도 0.13 m/s(`/speed_limit`), 나올 때 `/initialpose` 로 AMCL 에 넘김.
- 왜 remap 이 아니라 중계 노드인가: 실행 중에 입력원을 바꿔야 하는데 remap 은 시작 때 고정. 대가: 스캔 한 번 더 복사(🧪 `/scan_obstacles` 발행자 smoke_mode).
- 시뮬 결과: `hold` 최대 0.29 m 벌어짐, 초음파 벽 맞추기 `wall` 0.20 m. 📁 `limo_fire_navigation/README.md` "연기 모드".

**Q39. 실물 녹화를 왜 늘 같은 토픽으로 담나? 재생해서 무엇을 하나?**
- `record.launch.xml` 이 `/scan /imu /odom /odometry/filtered /limo_status /rc_state /motor_states /wheel_odometry /cmd_vel /tf /tf_static /joint_states /map /trajectory_node_list` + 카메라 info·압축 컬러·깊이 원본(초당 약 15 MB). 시뮬 맞추기(fit)와 확인(validate) 녹화를 나눔.
- 탐사 노드는 자기 입력을 따로 녹화(`record` → `*_node.bin`)해 `low_explore_replay` 로 다시 돌려 **결정 기록이 바이트 단위로 같은지** 확인 — 코드 정리 커밋마다 이것으로 동작이 안 바뀐 것을 증명.
- 🧪 실물 녹화 주기(`raw/real_bags/1116_fuel_rl_1.info.txt`, 92.5 s): `/scan` 900 개 = 9.7 Hz, `/imu` 97 Hz, `/odom` 48.5 Hz, `/rc_state` 38.9 Hz, `/odometry/filtered` 29.1 Hz, 깊이 29.2 Hz, `/map` 1.1 Hz.

**Q40. 이 시스템을 처음 보는 사람이 "왜 로봇이 안 움직이지?" 를 풀 때 차례는?**
1. 리모컨 SWB 위(오토)? `ros2 topic echo /rc_state --field buttons`(buttons[1] == 2).
2. 탐사가 켜졌나? `/explore/state`(또는 `/explore/frontier_explorer/state`) == running, `~/status` 한 줄의 까닭.
3. `dry_run:=false` 인가(아니면 `/cmd_vel` 발행자 없음 — `ros2 topic info /cmd_vel`).
4. 입력이 오나: `/scan` 10 Hz, TF `map→base_link`(`ros2 run tf2_ros tf2_echo map base_link`).
5. 판단 모델: `~/status` 의 "판단 모델: 보냄·답·시간 끝".
6. Nav2 판이면 `ros2 lifecycle get /bt_navigator`, `/navigate_to_pose` 피드백, costmap 에 로봇이 치명 칸 안인지.
7. 다른 `/cmd_vel` 발행자가 0 을 계속 내고 있지 않은지.

---

## raw/ 안에 무엇이 있나

| 파일 | 뽑은 명령 |
|---|---|
| `nodes.txt` · `topics.txt` · `services.txt` · `actions.txt` · `components.txt` | `ros2 node list` · `ros2 topic list -t` · `ros2 service list -t` · `ros2 action list -t` · `ros2 component list` |
| `node_info.txt` | 노드마다 `ros2 node info` |
| `topic_info_v.txt` | 토픽마다 `ros2 topic info -v` (발행·구독 노드와 QoS) |
| `edges.txt` | `topic_info_v.txt` 를 "발행 노드[QoS] → 토픽 → 구독 노드[QoS]" 한 줄씩으로 — rqt_graph 와 같은 정보 |
| `params.txt` · `param_values.yaml` | 노드마다 `ros2 param list` · `ros2 param dump` (숨은 보조 노드는 파라미터 서비스가 없어 건너뜀) |
| `lifecycle.txt` | `ros2 lifecycle nodes` + `ros2 lifecycle get` |
| `hz.txt` · `rtf.txt` | 주요 토픽 `ros2 topic hz -w 100` 10 s (벽시계), `/clock` 으로 잰 시뮬 속도 |
| `tf_frames.txt` · `tf2_monitor.txt` | `/tf` 를 12 s 받아 변환별 개수·Hz·중복 / `tf2_monitor` |
| `threads.txt` | `ps -eo nlwp,rss,pcpu` + `ps -T` (프로세스별 스레드) |
| `launch_log_summary.txt` | launch 로그의 프로세스 시작 줄과 WARN/ERROR 종류별 횟수 |
| `real_bags/*.info.txt` · `*_1117.txt` | 실물 녹화 `ros2 bag info`, 실물 기동 로그에서 뽑은 줄 |

다시 뽑기: 같은 launch 를 띄운 뒤 위 명령들. 이번엔 다른 사람 시뮬과 안 섞이게 `ROS_DOMAIN_ID=93`, Webots 포트 1293~1295, 화면 없는 `xvfb-run` 으로 띄웠다.
