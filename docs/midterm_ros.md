# 중간고사 대비 — 우리 코드베이스의 ROS 2 구조

서술형 시험용 정리. 이 문서만 보고 **폴더 구조 · 전체 구조 · 노드/토픽/주기 · TF · 액션 주는 법 · 내가 한 일**을 손으로 그리고 적을 수 있게 썼다.
값은 전부 우리 코드(`src/limo_fire`)·설정(`config/*.yaml`)·실측(11-03 녹화 bag 의 `metadata.yaml` 메시지 수 ÷ 길이)에서 왔다. 2026-11-04 기준.
센서별 세부 근거는 [sensor_topics.md](sensor_topics.md), 시뮬 구조는 [simulation.md](simulation.md).

---

## 0. 외울 것 한 장

| 무엇 | 값 |
|---|---|
| 로봇 | AgileX LIMO Pro (WeGo 판), 컴퓨터 Jetson Orin Nano, Ubuntu 22.04 + **ROS 2 Humble** |
| 센서 | 라이다 YDLIDAR T-mini Plus · 깊이 카메라 Orbbec Dabai DC1 · IMU(섀시 안) · 바퀴 엔코더 4개 |
| 핵심 흐름 | 센서 → **EKF**(odom→base_link) → **Cartographer / AMCL**(map→odom) → **Nav2** → `/cmd_vel` → 섀시 |
| 인식 흐름 | 컬러 → **yolo_detector**(TensorRT YOLO26s-seg, 7 클래스) → **object_mapper**(깊이 + TF) → 지도 위 물체 |
| 주기 | `/imu` 100 · `/odom` 50 · `/wheel_odometry` 50 · `/rc_state` 40 · 카메라 30 · `/odometry/filtered` 30 · `/motor_states` 20 · Nav2 제어 20 · `/scan` 10 · `/joint_states` 10 · `/limo_status` 2 · `/map` 1 Hz |
| TF | `map →(Cartographer 또는 AMCL) odom →(EKF) base_link →(URDF) laser_link · imu_link · camera_link →(카메라 드라이버) color/depth 프레임` |
| 액션 | RViz **2D Goal Pose** → `/goal_pose` → bt_navigator 가 **`navigate_to_pose`** 액션 실행 → 안에서 `compute_path_to_pose` · `follow_path` · `spin` · `backup` · `wait` |
| 언어 규칙 | 노드는 C++(rclcpp), 도구는 Rust, launch 는 **XML**, Python 안 씀 |

---

## ★ 시험장에서 적을 답 (주영 답지 — A4 두 장, 약 25 분)

이 문서 전체(1~14 절)는 공부·근거용이다. 시험에서는 **그림 2 개 + 문단 7 개**를 적는다. 다 쓰면 약 7,000 자라 손으로는 시간 안에 다 못 쓸 수 있으니 **먼저 그림 1 → 그림 2 → 문단 4 → 3 → 6 → 5 (핵심, 약 5,000 자)** 순서로 쓰고, 남는 시간에 문단 2 → 7 → 1. 커밋 이야기(문단 3·5·6)는 빼지 않는다(교수님이 커밋 기록과 대조). 숫자를 외워 옮기기보다 결정마다 "무엇이 문제 → 무엇을 재고 → 무엇을 바꿨나"를 자기 말로 쓴다.

**그림 1 — 노드·토픽** (실행 중 `ros2 node info` 로 확인한 연결)
```
limo_fire_base ─/odom 50Hz, /imu 100Hz─► EKF ─/odometry/filtered 30Hz─► Cartographer · controller · bt_navigator
               (/imu 는 Cartographer 도)   EKF ─TF odom→base_link─► 모두 (AMCL 은 토픽이 아니라 이 TF)
라이다 ─/scan 10Hz (BEST_EFFORT)─► Cartographer | AMCL, 전역·지역 costmap
지도 만들 때: cartographer_node ─/submap_list 0.3s마다─► occupancy_grid_node ─/submap_query(서비스)─► cartographer_node
              occupancy_grid_node ─/map 1Hz─► RViz·map_saver_cli,  cartographer_node ─TF map→odom, /trajectory_node_list ~30Hz─►
주행 때: map_server ─/map(TRANSIENT_LOCAL, 한 번)─► AMCL·전역 costmap,  AMCL(0.25 m·0.2 rad 움직이면 갱신) ─TF map→odom, /amcl_pose, /particle_cloud─►
RViz ─/initialpose─► AMCL    RViz ─/goal_pose(토픽)─► bt_navigator =액션=► planner · controller · behavior_server
controller ─/cmd_vel_nav 20Hz─► velocity_smoother ─/cmd_vel 20Hz(설정)─► limo_fire_base ─UART 0x111─► 섀시
behavior_server(spin·backup) ─/cmd_vel 직접─► limo_fire_base
카메라 ─color 30Hz─► yolo_detector ─detections, mask, image─► object_mapper ◄─depth 30Hz, camera_info─ 카메라
RViz ─/clicked_point─► object_mapper ─/object_mapper/markers 10Hz, /summary 2Hz, /selected─► RViz
```

**그림 2 — TF**: `map →(Cartographer|AMCL) odom →(EKF) base_link →(URDF, robot_state_publisher) laser_link · imu_link · camera_link →(Orbbec 드라이버) color/depth optical`

**문단 1 — 전체 구조·폴더.** ROS 2 Humble 워크스페이스. 남의 코드는 `src/external`(WeGo LIMO, 교수님 Webots)에 두고 직접 고치지 않으며(바꿀 것은 우리 패키지로 가져와 고침 — 섀시 드라이버가 그 예), 우리 코드는 역할마다 패키지 하나 `src/limo_fire/limo_fire_<역할>`: base(섀시 드라이버), bringup(launch·config), description(URDF→TF), navigation(Cartographer·Nav2 설정·지도), perception(YOLO·물체 지도), sim(Webots 플러그인·녹화 분석). `train/`·`sim/` 은 ROS 가 아니라 COLCON_IGNORE. 노드는 C++, launch 는 XML, 설정은 yaml. 실행 순서: `robot.launch.xml`(센서만은 `base:=false`) → 지도 `mapping.launch.xml` → `save_map.launch.xml`, 또는 주행 `navigation.launch.xml` → 인식 `yolo.launch.xml`. 시뮬은 `sim.launch.xml world:= slam:= perception:=` 이 같은 파일들을 include 해서 "로봇 켜기"만 다르다.

**문단 2 — 센서와 주기.** limo_fire_base 가 UART 460800 bps 로 받은 프레임을 ID 별로 나눠 `/imu` 100 Hz, `/odom` 50 Hz, `/wheel_odometry` 50 Hz, `/rc_state` 40 Hz, `/motor_states` 20 Hz, `/limo_status` 2 Hz 로 내고, `/cmd_vel` 이 오면 바로 0x111 구동 프레임을 보낸다(리모컨 SWB 오토 모드일 때만 섀시가 받음). 라이다 `/scan` 10 Hz(±110°), 카메라 컬러·깊이 각 30 Hz. `/joint_states` 10 Hz 는 센서가 아니라 joint_state_publisher 가 URDF 바퀴 관절 값(실물은 0)을 내는 것. 센서 주기는 녹화 bag 의 메시지 수 ÷ 시간 실측, Nav2(제어 20, 재계획 1, costmap 0.5·2 Hz)는 설정값.

**문단 3 — 위치 추정.** EKF 가 `/odom`(x, y, yaw, 속도)과 `/imu`(yaw, 각속도, 가속도)를 30 Hz 로 합쳐 `/odometry/filtered` 와 TF odom→base_link 를 낸다(섀시 TF 는 꺼서 발행자 하나). 바퀴 위치는 미끄러짐이 쌓이니 differential(앞 값과의 차이)로 쓰고, 회전은 IMU 자이로를 같이 믿는다. 누적 오차는 지도 만들 때 Cartographer(`/scan`·`/imu`·`/odometry/filtered` 로 스캔 정합 + 루프 닫기), 주행할 때 AMCL(입자 500~2000, `/scan` 을 `/map` 에 맞춤)이 TF map→odom 으로 보정한다. `/map` 은 0.05 m 격자. 내가 `05f3a3b` 에서 Cartographer 회전 필터를 0.1°→0.5° 로 바꿔 노드가 덜 생겨 CPU 를 20 % 줄였고(궤적 차 1.5~1.8 cm < 지도 칸 5 cm), 자세 발행 주기를 5→20 ms 로 늘렸다(map→odom TF 가 /tf 의 가장 큰 몫이라). 그래서 /tf 가 10-26 녹화 245 Hz 에서 11-03 녹화 91~98 Hz 로 줄었다.

**문단 4 — 액션으로 주행.** 2D Goal Pose 는 액션이 아니라 `/goal_pose` 토픽이고, bt_navigator 가 이를 받아 자기 `navigate_to_pose` 액션 서버에 goal 로 넣는다(명령줄: `ros2 action send_goal /navigate_to_pose nav2_msgs/action/NavigateToPose …`). bt_navigator 는 이 액션의 서버이면서, 행동 트리가 `compute_path_to_pose`(planner_server, Smac 2D, 1 Hz 재계획) · `follow_path`(controller_server, RPP 20 Hz) · `spin`·`backup`·`wait`(behavior_server) 액션의 클라이언트이고, costmap `clear_entirely_*` 서비스의 클라이언트다. 경로는 토픽이 아니라 액션으로 넘어간다: compute_path_to_pose 의 result 경로를 BT 가 follow_path 의 goal 로 준다(`/plan` 토픽은 RViz 표시용). BT 는 100 ms(bt_loop_duration)마다 돌며 feedback 을 낸다. 실패하면 Nav2 기본 BT 의 복구(costmap 지우기 → spin → wait → backup)를 RoundRobin 으로 실패할 때마다 다음 것으로 돌린다 — 같은 복구만 되풀이하지 않으려고. 복구 동작은 behavior_server 가 velocity_smoother 를 거치지 않고 `/cmd_vel` 로 바로 낸다. 타입 `nav2_msgs/action/NavigateToPose`: goal = PoseStamped(+behavior_tree), feedback = current_pose·navigation_time·estimated_time_remaining·number_of_recoveries·distance_remaining, result 는 비어 있어 성공·실패·취소 상태로 끝을 안다. 액션은 서비스와 달리 피드백과 취소가 된다. 시뮬 맵 4 개의 순찰점 46 개는 월드 파일의 `patrol` 목록을 읽어 navigate_to_pose 액션 클라이언트(임시 스크립트, 저장소에 없음)로 하나씩 보내고 결과를 기다려 돌았다. 여러 점을 한 번에 주는 정식 방법은 `follow_waypoints`(waypoint_follower 가 navigate_to_pose 클라이언트).

**문단 5 — 인식과 내 실수.** yolo_detector 가 컬러 영상을 TensorRT YOLO26s-seg(7 클래스: 불·연기·소화기·유도등·사람·문·계단)로 한 장 10.3 ms 에 처리해 `/yolo_detector/detections`(Detection2DArray)·`mask` 를 낸다. object_mapper 는 탐지와 깊이를 ApproximateTime(0.05 s)으로 짝짓고, 깊이 + camera_info(fx, fy, cx, cy)로 3D 역투영 → 영상 시각의 TF(map ← 깊이 카메라)를 `lookupTransform` 으로 찾아 map 좌표 → 같은 클래스 근처면 합치고 5 번 이상 본 것만 MarkerArray 10 Hz 로 낸다. 내 실수: `05f3a3b` 에서 안 쓰는 포인트 클라우드를 껐는데, Orbbec 드라이버는 컬러 포인트 클라우드가 켜져야만 depth_registration 을 켜서 깊이가 컬러에 정렬되지 않게 됐다(640×400 대 640×480). 팀원이 `7c208c1` 에서 찾았다. 그래서 실물은 마스크 대신 상자 깊이로 재고 위치가 어긋날 수 있다. 고칠 방법은 `depth_registration: true` 로 정렬만 켜는 것. 11-04 실물에서 켜 보니 드라이버가 `ALIGN_D2C_HW_MODE` 로 깊이를 640×480·`camera_color_optical_frame`·fx 489.39(컬러와 같음)로 내는 것까지 확인했다. 깊이 값과 주기("19→30 Hz" 가 정렬을 끈 덕분인지)는 카메라 앞이 0.3 m 안이라 아직 못 쟀고, 확인 전이라 설정은 되돌려 뒀다. 같이 바꿀 것: 탐사 RL 시뮬 깊이 모델(지금 640×400·fx 478 기준).

**문단 6 — 내가 한 것 (커밋).** 구조 `759c357`, URDF `10e621b`, 실물 XML launch `d5068ff`(WeGo 대시보드 Python 실행을 옮김), YOLO 노드 `bf15281`·`09da2f2`, object_mapper `fdda7f8`, Jetson 실시간화·컴포넌트 `05f3a3b`, Webots 맵·sim.launch·Nav2 `571f1a0`, 인식 벤치마크 `4698491`, 섀시 드라이버 `6f59512`(버려지던 리모컨·모터·바퀴 거리 토픽), 녹화 launch·bag 분석 도구는 탐사 RL 커밋 `eb2670c` 에 같이 넣음(탐사 RL 자체는 ROS 가 아닌 GPU 학습, `40ecee2` 까지). 팀원: 실물 TF 를 URDF 하나로 `773d838`(woojoo1116), 센서 토픽 문서 `7c208c1`(leeyechan). `571f1a0` 에서 Webots 새 맵 3 개와 교수님 맵을 실물과 같은 launch 로 돌리며 Nav2 를 바꿨다: DWB 는 U 자 구간에서 제자리 회전만 하다 멈춰 RPP 로, NavFn 은 미로 끝까지 경로를 못 내 Smac 2D 로, 팽창 0.05·0.13 m 는 로봇 반경 0.2 m 보다 작아 벽에 붙어서 0.45 m 로, velocity_smoother 5 Hz 는 명령이 0.2 초마다만 나가 20 Hz 로(컨트롤러는 원래 20 Hz). 4 맵 순찰점 46 개 전부 도착. EKF·AMCL 값은 WeGo 그대로다. main 은 "작업 하나 = 커밋 하나" 규칙(`62b091d`)으로 합쳤고 세부 커밋은 태그 v0.1.0~v0.6.0 에 남아 있다.

**문단 7 — ROS 2 기능.** Nav2 는 lifecycle 노드라 lifecycle_manager 두 개(localization: map_server·amcl / navigation: 나머지 7 개)가 autostart 로 각 노드의 `/<노드>/change_state` 서비스를 불러 configure → activate 하고, `/bond` 로 살아 있는지 봐서 하나가 죽으면 전체를 내린다. 재시작·정지는 `…/manage_nodes` 서비스. 모두 `component_container_isolated`(nav2_container) 한 프로세스에 `load_composable_node` 로 올렸다. 인식 두 노드는 `component_container_mt` 에 `use_intra_process_comms` 로 복사 없이 넘기고, yolo_detector 는 작업 스레드가 최신 영상만 처리한다. 파라미터는 노드별 yaml 을 `<param from>` 으로 넣는다. QoS: `/scan` BEST_EFFORT, 카메라는 RELIABLE 로 나오고 우리 노드는 SensorDataQoS 로 받음, `/map`·`/tf_static`·마커는 TRANSIENT_LOCAL. 내 서비스 `/object_mapper/clear`(std_srvs/Trigger). 시뮬은 Ros2Supervisor 가 `/clock`(모든 노드 use_sim_time), 바퀴는 ros2_control `diffdrive_controller`, EKF 자리는 교수님 플러그인이 대신하고, `nav2_sim.yaml` 이 use_sim_time·odom_topic·AMCL 처음 위치만 덮는다. 교수님 robot_launch.py 는 URDF 를 1 초만 기다려 두 번에 한 번 TF 가 안 나와 `sim_robot.launch.xml` 을 따로 짰다.

---

## 1. 폴더 구조

ROS 2 워크스페이스 관례(`src/` 아래 패키지, 루트에서 `colcon build`)를 따른다. ROS 가 아닌 폴더에는 `COLCON_IGNORE` 를 두어 빌드에서 뺀다.

```
limo_fire_ws/                       워크스페이스 루트 (colcon build)
├── src/
│   ├── external/                   남의 코드 — 고치지 않음
│   │   ├── wego_limo_pro/          WeGo LIMO Pro 드라이버·예제 (라이다·카메라 드라이버 포함)
│   │   └── webots_ros2_limo/       교수님 Webots LIMO (센서 플러그인 limo_sensors)
│   └── limo_fire/                  ★ 우리 패키지 (이름 규칙 limo_fire_<역할>)
│       ├── limo_fire_base/         섀시 UART 드라이버 (C++) — /odom /imu /cmd_vel …
│       ├── limo_fire_bringup/      launch 와 config (실물·시뮬·녹화)
│       ├── limo_fire_description/  URDF/xacro, RViz 설정, 로봇 모델 → TF
│       ├── limo_fire_navigation/   Cartographer·Nav2 파라미터, 저장 지도
│       ├── limo_fire_perception/   YOLO 탐지·물체 지도·인식 벤치마크 (C++/CUDA, TensorRT)
│       └── limo_fire_sim/          Webots 정답 카메라 플러그인, bag 분석 도구 (C++)
├── sim/          Webots 월드·PROTO, 맵 원본, Isaac Sim 자료        [COLCON_IGNORE]
├── train/        학습 (C++/CUDA/Rust, ROS 아님)                    [COLCON_IGNORE]
│   ├── detect/   YOLO 학습기·데이터 라벨러
│   ├── explore/  탐사 강화학습 GPU 시뮬·MDP·PPO
│   └── maps/     맵 변환기 (Rust)
├── tools/repo_check/  커밋 규칙 검사 (Rust)                         [COLCON_IGNORE]
├── docs/  문서 · refs/  참고 자료 · archive/  안 쓰는 코드 · firmware/  ESP32 (예정)
├── env.sh        환경 (conda 빼고 ROS 2 Humble + install 소싱)
├── CLAUDE.md     작업 규칙 · CHANGELOG.md  변경 기록
```

패키지 안은 ROS 2 기본 배치: `include/<패키지>/`, `src/`, `launch/`, `config/`, `package.xml`, `CMakeLists.txt`(ament_cmake).

### 왜 이렇게 나눴나 (서술형 포인트)
- **역할마다 패키지 하나** — 드라이버·설정·모델·주행·인식·시뮬이 서로 인터페이스(토픽·TF)로만 이어져서 하나를 바꿔도 나머지는 그대로.
- **남의 코드는 `external/` 에 그대로** — 바꿀 게 있으면 우리 판을 만든다. 예: WeGo `limo_base` 를 고치지 않고 `limo_fire_base` 로 가져와 확장, WeGo Python launch 대신 `limo_fire_bringup` 의 XML launch.
- **설정은 코드 밖 yaml** — 주기·토픽 이름·임계값을 다시 빌드 없이 바꾼다.
- **실물과 시뮬이 같은 launch 를 쓴다** — 다른 것은 "로봇 켜기" 하나(`robot.launch.xml` ↔ `sim_robot.launch.xml`).

---

## 2. 패키지 · 노드 · 실행 파일

| 패키지 | 노드 (실행 이름) | 하는 일 |
|---|---|---|
| `limo_fire_base` | `/limo/limo_fire_base` | 섀시와 UART(`/dev/ttylimo`, 460800 bps)로 통신. 받은 프레임 → `/odom` `/imu` `/limo_status` `/rc_state` `/motor_states` `/wheel_odometry`, `/cmd_vel` → 구동 명령 프레임 |
| (external) ydlidar_ros2_driver | `/ydlidar_ros2_driver_node` | 라이다 → `/scan` |
| (external) orbbec_camera | `/camera/camera` (컴포넌트, `camera_container`) | 카메라 → `/camera/color/*` `/camera/depth/*`, 카메라 프레임 TF |
| robot_localization | `/ekf_filter_node_odom` | `/odom` + `/imu` → `/odometry/filtered`, TF odom→base_link |
| `limo_fire_description` | `robot_state_publisher`, `joint_state_publisher` | URDF → `/tf_static`(센서 장착 위치), 바퀴 관절 → `/tf` |
| cartographer_ros | `cartographer_node`, `cartographer_occupancy_grid_node` | SLAM: `/scan` `/imu` `/odometry/filtered` → TF map→odom, `/map` |
| nav2 (`nav2_container` 한 프로세스) | `map_server` `amcl` `planner_server` `controller_server` `smoother_server` `behavior_server` `bt_navigator` `waypoint_follower` `velocity_smoother` `lifecycle_manager_*` | 저장 지도로 위치 추정 + 목표 주행 |
| `limo_fire_perception` (`perception_container`) | `yolo_detector`, `object_mapper` | 탐지 → 지도 위 물체 |
| `limo_fire_perception` | `perception_benchmark` | 시뮬 정답과 비교해 인식 성능 표 (시뮬 전용) |
| `limo_fire_sim` | `gt_camera` (Webots 플러그인), `bag_export` `scan_stats_bag` `motion_export` `sensor_probe` | 시뮬 정답 `/gt/*`, 녹화 분석 도구 |
| rviz2 | `limo_fire_rviz` | 운용 화면 하나(`limo_fire.rviz`) |

---

## 3. launch 구조 — 무엇을 켜면 무엇이 뜨나

```
[실물] robot.launch.xml ─┬─ description.launch.xml (robot_state_publisher + joint_state_publisher)
                         ├─ limo_fire_base          (base:=false 면 끔)
                         ├─ camera_container / Orbbec
                         ├─ ydlidar_ros2_driver_node
                         ├─ ekf_node               (base:=false 면 끔)
                         └─ jetson_clocks          (클럭 고정)

[시뮬] sim_robot.launch.xml ─┬─ webots (+ Ros2Supervisor → /clock)
                             ├─ webots_controller_LIMO (교수님 센서 플러그인 + ros2_control)
                             ├─ spawner diffdrive_controller, joint_state_broadcaster
                             └─ description.launch.xml (platform:=webots)

그 위에 공통으로:
  mapping.launch.xml     Cartographer 2 노드 + RViz          ← 지도 만들 때
  save_map.launch.xml    map_saver_cli 로 pgm + yaml 저장
  navigation.launch.xml  nav2_container + map_server + AMCL + RViz  ← 주행할 때
  yolo.launch.xml        perception_container (yolo_detector + object_mapper)
  record.launch.xml      ros2 bag record (늘 같은 토픽 묶음)
  lane_tracking.launch.xml, dl_tracking.launch.xml   WeGo 대시보드 예제(차선·딥러닝 주행)를 XML 로 옮긴 것
  sim.launch.xml world:=<맵> [slam:=true] [perception:=true]  = 시뮬 전부 한 번에
```

- 실물 순서: `robot.launch.xml` → (`mapping.launch.xml` → `save_map.launch.xml`) 또는 `navigation.launch.xml` → `yolo.launch.xml`.
- Cartographer 와 AMCL 은 같이 켜지 않는다(둘 다 map→odom 을 내므로).
- Nav2 노드는 `component_container_isolated` 하나에 컴포넌트로 올리고 `lifecycle_manager` 가 `autostart` 로 configure → activate 한다.

---

## 4. 전체 노드-토픽 그림 (손으로 옮겨 그릴 것)

```
 ┌──────────────┐ /odom 50Hz      ┌──────────┐ /odometry/filtered 30Hz  ┌───────────────────┐
 │limo_fire_base├────────────────►│   EKF    ├─────────────────────────►│ Cartographer      │
 │ (UART 섀시)├─/imu 100Hz─────►│          │ TF odom→base_link        │  또는 AMCL(TF 로) │
 │              │                 │          │  (AMCL 은 토픽 대신 TF)  │ + map_server(주행)│
 └──▲───────────┘        │        └──────────┘                          │ TF map→odom, /map │
    │ /cmd_vel           └──────────────────────────────────────────────►│                   │
    │                                                                    └────────┬──────────┘
    │                    ┌──────────┐ /scan 10Hz ─────────────────────────────────┤
    │                    │  라이다   ├────────────────┐                           │ /map, TF
    │                    └──────────┘                ▼                           ▼
    │   ┌──────────────────┐ cmd_vel_nav ┌──────────────────────────────────────────────┐
    └───┤velocity_smoother │◄────────────┤ controller_server (RPP 20Hz, 지역 costmap)    │
        └──────────────────┘             │ planner_server (Smac2D, 전역 costmap)        │
                                         │ bt_navigator ◄── /goal_pose (RViz)           │
                                         └──────────────────────────────────────────────┘

 ┌────────────┐ /camera/color/image_raw 30Hz ┌──────────────┐ /yolo_detector/detections ┌──────────────┐
 │ Orbbec 카메라├───────────────────────────►│ yolo_detector├──────────────────────────►│ object_mapper│
 │            ├── /camera/depth/image_raw ───┼──────────────┼── /yolo_detector/mask ───►│ (+ TF map)   │
 │            ├── /camera/depth/camera_info ─┼──────────────┼──────────────────────────►│              │
 └────────────┘                              └──────────────┘                           └──────┬───────┘
                                                                     /object_mapper/markers ▼  RViz
```

---

## 5. 토픽 전체 — 누가 내고 누가 받고 몇 Hz 인가

주기 "실측"은 11-03 녹화(`1103_M_drive3` 95 s, `1103_M_cmd1` 227 s, `static_1103_floor` 61 s)의 메시지 수 ÷ 녹화 길이. 녹화에 없는 토픽은 설정값.

### 5-1. 센서·섀시

| 토픽 | 메시지 타입 | 주기 (실측) | frame_id | 발행 | 구독 |
|---|---|---|---|---|---|
| `/imu` | `sensor_msgs/Imu` | **99.8 Hz** | `imu_link` | limo_fire_base | EKF, Cartographer |
| `/odom` | `nav_msgs/Odometry` | **50.0 Hz** | `odom` → `base_link` | limo_fire_base | EKF |
| `/wheel_odometry` | `sensor_msgs/JointState` (좌·우 누적 m) | **50.0 Hz** | — | limo_fire_base | 녹화·분석 |
| `/rc_state` | `sensor_msgs/Joy` (스틱 축, 스위치 SWA~SWD) | **40.0 Hz** | — | limo_fire_base | 녹화·분석 |
| `/motor_states` | `sensor_msgs/JointState` (rad/s, A, 엔코더) | **20.0 Hz** | — | limo_fire_base | 녹화·분석 |
| `/limo_status` | `limo_msgs/LimoStatus` (전압·모드·오류) | **2.0 Hz** | — | limo_fire_base | 녹화 |
| `/scan` | `sensor_msgs/LaserScan` (±110°, 0.0155 rad 간격, 0.01~64 m) | **10.0 Hz** | `laser_link` | ydlidar 드라이버 | Cartographer, AMCL, 전역·지역 costmap |
| `/camera/color/image_raw` | `sensor_msgs/Image` rgb8 640×480 | **29.9 Hz** | `camera_color_optical_frame` | Orbbec | yolo_detector |
| `/camera/color/camera_info` | `sensor_msgs/CameraInfo` | 29.9 Hz | 위와 같음 | Orbbec | (perception_benchmark) |
| `/camera/depth/image_raw` | `sensor_msgs/Image` 16UC1(mm) 640×400 | **30.0 Hz** | `camera_depth_optical_frame` | Orbbec | object_mapper |
| `/camera/depth/camera_info` | `sensor_msgs/CameraInfo` (fx 478.0, cx 323.4, cy 199.3) | 30.0 Hz | 위와 같음 | Orbbec | object_mapper |

### 5-2. 위치·지도

| 토픽 | 타입 | 주기 | 발행 | 구독 |
|---|---|---|---|---|
| `/odometry/filtered` | `nav_msgs/Odometry` | **30.0 Hz** (설정 `frequency: 30`) | EKF | Cartographer(`odom` 리맵), bt_navigator, controller_server (velocity_smoother 는 `feedback: OPEN_LOOP` 라 안 받음, AMCL 은 토픽 대신 TF) |
| `/tf` | `tf2_msgs/TFMessage` | 약 91~98 Hz (모든 동적 TF 합) | EKF, Cartographer/AMCL, robot_state_publisher, Orbbec | 모두 |
| `/tf_static` | `tf2_msgs/TFMessage` | 시작 때 한 번 (TRANSIENT_LOCAL) | robot_state_publisher, Orbbec | 모두 |
| `/joint_states` | `sensor_msgs/JointState` | **10 Hz** | joint_state_publisher (실물은 0), joint_state_broadcaster (시뮬) | robot_state_publisher |
| `/map` | `nav_msgs/OccupancyGrid` 0.05 m 칸 | 지도 만들 때 **약 1 Hz** (`publish_period_sec 1.0`, 실측 0.8~1.05) / 주행 때 map_server 가 한 번(TRANSIENT_LOCAL) | cartographer_occupancy_grid_node 또는 map_server | AMCL, 전역 costmap, RViz |
| `/trajectory_node_list` | `visualization_msgs/MarkerArray` | 약 28~33 Hz | cartographer_node | RViz |
| `/amcl_pose` | `geometry_msgs/PoseWithCovarianceStamped` | 스캔마다(0.25 m·0.2 rad 움직였을 때 갱신) | AMCL | RViz |
| `/initialpose` | `geometry_msgs/PoseWithCovarianceStamped` | 누를 때 | RViz 2D Pose Estimate | AMCL |

### 5-3. 주행

| 토픽 | 타입 | 주기 | 발행 | 구독 |
|---|---|---|---|---|
| `/goal_pose` | `geometry_msgs/PoseStamped` | 누를 때 | RViz 2D Goal Pose | bt_navigator |
| `/plan` | `nav_msgs/Path` | 1 Hz 재계획 | planner_server | controller(액션으로 받음), RViz |
| `cmd_vel_nav` | `geometry_msgs/Twist` | 20 Hz (`controller_frequency`) | controller_server (리맵 `cmd_vel → cmd_vel_nav`) | velocity_smoother |
| `/cmd_vel` | `geometry_msgs/Twist` (linear.x, angular.z) | Nav2 일 때 20 Hz (`smoothing_frequency`, 설정값 — Nav2 녹화는 아직 없음) | velocity_smoother (또는 teleop) | limo_fire_base (시뮬: diffdrive_controller) |
| `global_costmap/costmap` | `nav_msgs/OccupancyGrid` | 0.5 Hz | planner_server 안 costmap | RViz |
| `local_costmap/costmap` | `nav_msgs/OccupancyGrid` | 2 Hz | controller_server 안 costmap | RViz, behavior_server(raw) |

### 5-4. 인식

| 토픽 | 타입 | 주기 | 발행 | 구독 |
|---|---|---|---|---|
| `/yolo_detector/detections` | `vision_msgs/Detection2DArray` (상자·클래스·점수) | 컬러 영상마다 (~30 Hz, 한 장 10.3 ms) | yolo_detector | object_mapper |
| `/yolo_detector/mask` | `sensor_msgs/Image` (물체 번호) | 위와 같음 | yolo_detector | object_mapper |
| `/yolo_detector/image` | `sensor_msgs/Image` (마스크 덧칠) | 위와 같음 | yolo_detector | object_mapper(사진 저장), RViz |
| `/object_mapper/markers` | `visualization_msgs/MarkerArray` (map 프레임) | **10 Hz** (100 ms wall timer, 물체가 있을 때), TRANSIENT_LOCAL | object_mapper | RViz |
| `/object_mapper/summary` | `rviz_2d_overlay_msgs/OverlayText` (한글 현황판) | 2 Hz, TRANSIENT_LOCAL | object_mapper | RViz |
| `/object_mapper/selected` | `sensor_msgs/Image` (고른 물체 사진) | 누를 때 | object_mapper | RViz |
| `/clicked_point` | `geometry_msgs/PointStamped` | 누를 때 | RViz Publish Point | object_mapper |

### 5-5. 실행 중 노드마다 주고받는 것 전부 (실측 `ros2 node info`)

2026-11-04 Webots 에서 `sim.launch.xml world:=limo_fire_maze perception:=true`(저장 지도 + AMCL + Nav2 + YOLO·물체 지도)를 띄우고 모든 노드에 `ros2 node info` 를 돌린 결과. 모든 노드에 공통인 `/parameter_events`·`/rosout`·`/clock`·lifecycle 서비스(`change_state` 등)·파라미터 서비스·`/bond` 는 뺐다. 시뮬에서는 Webots 플러그인이 실물 드라이버와 같은 이름(`/limo/limo_base`, `/ydlidar_ros2_driver_node`, `/camera/camera`)으로 낸다. 실물은 섀시가 `/rc_state` `/motor_states` `/wheel_odometry` `/limo_status` 를 더 내고 `/cmd_vel` 을 받으며, odom→base_link 는 EKF 가 낸다. bt_navigator·controller 의 `/odom` 구독은 실물에서 `/odometry/filtered`.

| 노드 | 구독 | 발행 | 서비스 서버 | 액션 |
|---|---|---|---|---|
| amcl | `/scan` LaserScan, `/map` OccupancyGrid, `/initialpose` PoseWithCovarianceStamped | `/amcl_pose`, `/particle_cloud` (nav2_msgs/ParticleCloud), `/tf` (map→odom) | `/set_initial_pose`, `/reinitialize_global_localization`, `/request_nomotion_update` | — |
| map_server | — | `/map` | `/map_server/load_map`, `/map_server/map` (GetMap) | — |
| bt_navigator | `/goal_pose` PoseStamped, `/odom`, `/tf` | — | — | 서버 `navigate_to_pose`, `navigate_through_poses` · 행동 트리 클라이언트 `compute_path_to_pose`, `follow_path`, `spin`, `backup`, `wait` + 서비스 클라이언트 `clear_entirely_global/local_costmap` |
| planner_server (+ global_costmap) | `/map`, `/scan`, `/global_costmap/footprint` | `/plan`, `/unsmoothed_plan`, `/global_costmap/costmap`(+`_raw`, `_updates`, `published_footprint`) | `/is_path_valid`, `/global_costmap/clear_entirely_global_costmap`, `clear_around_*`, `clear_except_*`, `get_costmap` | 서버 `compute_path_to_pose`, `compute_path_through_poses` |
| controller_server (+ local_costmap) | `/odom`, `/speed_limit`, `/scan` | `/cmd_vel_nav`, `/received_global_plan`, `/lookahead_point`, `/lookahead_collision_arc`, `/local_costmap/costmap`(+`_raw`, `_updates`, `published_footprint`) | `/local_costmap/clear_entirely_local_costmap` 등 | 서버 `follow_path` |
| behavior_server | `/local_costmap/costmap_raw`, `/local_costmap/published_footprint` | **`/cmd_vel`** (복구 동작은 velocity_smoother 를 거치지 않고 바로) | — | 서버 `spin`, `backup`, `drive_on_heading`, `wait` |
| smoother_server | `/global_costmap/costmap_raw`, `published_footprint` | `/plan_smoothed` | — | 서버 `smooth_path` (기본 BT 는 안 부름) |
| velocity_smoother | `/cmd_vel_nav` | `/cmd_vel` | — | — |
| waypoint_follower | — | — | — | 서버 `follow_waypoints`, 클라이언트 `navigate_to_pose` |
| lifecycle_manager_localization / _navigation | — | `/diagnostics` | `…/manage_nodes` (nav2_msgs/ManageLifecycleNodes), `…/is_active` (Trigger) | — |
| yolo_detector | `/camera/color/image_raw` | `/yolo_detector/detections` (Detection2DArray), `/yolo_detector/mask`, `/yolo_detector/image` | — | — |
| object_mapper | `/yolo_detector/detections`, `/yolo_detector/mask`, `/yolo_detector/image`, `/camera/depth/image_raw`, `/camera/depth/camera_info`, `/clicked_point` | `/object_mapper/markers`, `/object_mapper/summary` (OverlayText), `/object_mapper/selected` | `/object_mapper/clear` (Trigger) | — |
| robot_state_publisher | `/joint_states` | `/tf`, `/tf_static`, `/robot_description` | — | — |
| (시뮬) diffdrive_controller | `/cmd_vel` | `/diffdrive_controller/odom`, `/tf` | — | — |
| (시뮬) joint_state_broadcaster | — | `/joint_states`, `/dynamic_joint_states` | — | — |
| (시뮬) controller_manager | — | — | `load/switch/list_controllers` 등 (spawner 가 부름) | — |
| (시뮬) Ros2Supervisor | — | `/clock` | `spawn_urdf_robot` 등 | — |
| (시뮬) LIMO 정답 카메라 `gt_camera` | `/camera/color/camera_info` (동기용) | `/gt/detections`, `/gt/mask` | — | — |

실행 중 액션 11 개(`ros2 action list -t`): `navigate_to_pose`, `navigate_through_poses`, `compute_path_to_pose`, `compute_path_through_poses`, `follow_path`, `smooth_path`, `follow_waypoints`, `spin`, `backup`, `drive_on_heading`, `wait` — 전부 `nav2_msgs/action/*`.

---

## 6. TF 트리 (그릴 것)

```
map
 └─(Cartographer 또는 AMCL)─ odom
     └─(EKF)─ base_link
         ├─(URDF 고정)─ laser_link      (0.103, 0, −0.034)
         ├─(URDF 고정)─ imu_link        (0, 0, −0.1)
         ├─(URDF 고정)─ inertial_link
         ├─(URDF 고정)─ camera_link     (0.15, −0.005, 0.06)
         │    ├─(Orbbec 드라이버)─ camera_color_frame ── camera_color_optical_frame
         │    └─(Orbbec 드라이버)─ camera_depth_frame ── camera_depth_optical_frame
         └─(joint_states)─ front/rear_left/right_wheel_link (continuous 관절 4개)
```

- 규약: REP-103 (미터·라디안, x 앞·y 왼쪽·z 위), REP-105 (map → odom → base_link). `base_footprint` 는 실물 드라이버에 없어서 안 쓴다.
- **TF 하나 = 발행자 하나**. 섀시도 odom→base_link 를 낼 수 있지만 `pub_odom_tf: false` 로 끄고 EKF 만 낸다. 둘이 내면 번갈아 덮어써 위치가 튄다.
- **map→odom 의 뜻**: odom→base_link 는 매끄럽지만 바퀴 미끄러짐으로 누적 오차가 생긴다. 라이다를 지도에 맞춰 찾은 "진짜 위치"와의 차이를 map→odom 으로 내서 보정한다. 그래서 로봇 위치 = map→odom × odom→base_link.
- 센서 장착 위치는 URDF(`limo_fire_description/urdf/*.xacro`) 한 곳에만 둔다. launch 에 static_transform_publisher 를 흩어 두지 않는다. 카메라는 `orbbec_dabai.xacro` 모듈.

---

## 7. 노드별 자세히

### 7-1. 섀시 드라이버 `limo_fire_base` (C++, 내가 확장)
- 시리얼 `/dev/ttylimo`(→ `/dev/ttyTHS1`), **460800 bps**. 읽기 전용 스레드가 바이트를 받아 프레임으로 맞추고(`processRxData` → `parseFrame`), **프레임 ID** 로 나눈다:

| 프레임 ID | 내용 | 나가는 토픽 |
|---|---|---|
| `0x221` 움직임 상태 | 선속도·각속도·횡속도·조향각 (×0.001) | `/odom` (속도를 적분해 위치) |
| `0x321` `0x322` `0x323` | IMU 가속도·각속도·오일러각 | `/imu` |
| `0x211` 시스템 상태 | 차량 상태, 제어 모드, 배터리 전압(×0.1 V), 오류 코드, 구동 모드 | `/limo_status` |
| `0x241` 리모컨 | 스위치 SWA~SWD, 스틱 | `/rc_state` ★ 추가 |
| `0x251`~`0x254` 모터 | rpm, 전류, 엔코더 펄스 | `/motor_states` ★ 추가 |
| `0x311` 바퀴 거리 | 좌·우 누적 mm | `/wheel_odometry` ★ 추가 |
| `0x111` (보냄) | 구동 명령 | ← `/cmd_vel` |

- `/cmd_vel` 콜백이 오면 바로 `0x111` 프레임을 보낸다(타이머가 아니라 이벤트). 구동 모드에 따라 쓰는 값이 다르다: 4WD 차동은 `linear.x`·`angular.z`, 아커만은 `angular.z` 로 조향각 계산, 메카넘은 `linear.y` 도.
- 리모컨 스위치 SWB 가 위(오토 모드)여야 섀시가 ROS 명령을 받는다. 리모컨 조종은 ROS 를 거치지 않고 섀시가 바로 처리한다.
- 토픽을 절대 이름(`"/odom"`)으로 만들어서 노드가 `/limo` 이름공간이어도 토픽은 `/odom`. 코드 안 노드 이름은 `limo_base` 이고 launch 의 `name=` 이 `limo_fire_base` 로 바꾼다.
- `control_rate` 파라미터는 선언만 되고 쓰이지 않는다(WeGo 원본 그대로). 주기는 섀시가 보내는 프레임 주기가 정한다.
- WeGo 원본은 리모컨·모터·바퀴 거리 프레임을 받고 버렸다 → `limo_fire_base` 로 가져와 토픽으로 낸다(원본은 `archive/limo_base`).

### 7-2. EKF (`robot_localization`, `config/ekf.yaml`)
- 30 Hz, `two_d_mode: true`(평면 로봇), `world_frame: odom` → odom→base_link 를 낸다.
- 입력 `odom0: odom` — x, y, yaw, vx, vy, vyaw 사용. 입력 `imu0: imu` — yaw, 각속도 x·y·z, 가속도 x·y 사용(중력 제거).
- 왜: 바퀴는 회전할 때 미끄러져 yaw 가 틀리고, IMU 자이로는 짧은 시간 회전을 잘 잰다. 둘을 칼만 필터로 합친다.

### 7-3. Cartographer (지도 만들기, `cartographer_2d.lua`)
- 입력: `/scan`(0.1~8 m 만 씀), `/imu`(`use_imu_data = true`, `tracking_frame = imu_link`), 오도메트리는 `odom` 을 `/odometry/filtered` 로 리맵.
- 출력: TF map→odom (`published_frame = odom`, `provide_odom_frame = false` — odom 은 EKF 가 이미 냄), `/map` 은 `cartographer_occupancy_grid_node` 가 0.05 m 칸·1 초마다.
- 원리: 스캔을 서브맵에 맞추고(scan matching), 루프를 닫으면 포즈 그래프 최적화로 전체를 바로잡는다.
- 저장: `save_map.launch.xml` → `map_saver_cli` 가 `map.pgm` + `map.yaml` 을 src 와 install 두 곳에.

### 7-4. Nav2 (주행, `nav2_params.yaml`)
| 서버 | 설정 | 하는 일 |
|---|---|---|
| map_server | `map.yaml` | 저장 지도 → `/map` |
| AMCL | 입자 500~2000, likelihood_field, 빔 60개, 0.25 m·0.2 rad 움직이면 갱신, `tf_broadcast: true` | 입자 필터로 지도 위 위치 → TF map→odom |
| bt_navigator | 행동 트리 `navigate_to_pose_w_replanning_and_recovery.xml`, 100 ms 주기 | `navigate_to_pose` 액션 서버 |
| planner_server | **SmacPlanner2D**, 허용 0.25 m | 전역 경로 (`compute_path_to_pose` 액션) |
| controller_server | **Regulated Pure Pursuit**, 20 Hz, 0.26 m/s, 앞보기 0.3~0.9 m, 45° 넘게 틀어지면 제자리 회전, 도착 허용 0.25 m·0.25 rad | 경로 추종 (`follow_path` 액션) → `cmd_vel_nav` |
| velocity_smoother | 20 Hz, 최대 ±0.26 m/s · ±1.0 rad/s, 가속 2.5 m/s² · 3.2 rad/s², `feedback: OPEN_LOOP`(오도메트리 안 받음) | remap `cmd_vel → cmd_vel_nav`(입력), `cmd_vel_smoothed → cmd_vel`(출력) |
| behavior_server | spin, backup, drive_on_heading, wait | 복구 동작 (각각 액션) |
| 전역 costmap | map 프레임, 0.5 Hz, 정적 지도 + 라이다 장애물(3 m) + 부풀림 0.45 m, 로봇 반경 0.2 m | |
| 지역 costmap | odom 프레임, 2 Hz, 3×3 m 굴러가는 창, 라이다 장애물(1 m) + 부풀림 0.45 m, 발자국 0.32×0.22 m | |

- WeGo 값(DWB·NavFn·부풀림 0.05 m·5 Hz)으로는 Webots 덕트호스 미로 U 자에서 막혀서 RPP·Smac 2D·0.45 m·20 Hz 로 바꿨다.

### 7-5. 인식 (`limo_fire_perception`, C++/CUDA)
- **yolo_detector**: `/camera/color/image_raw` 를 SensorData QoS 로 받아 TensorRT 엔진(YOLO26s-seg, 입력 480×640 uint8, 전처리도 엔진 안)으로 탐지+분할. 클래스 7개: fire, smoke, fire_extinguisher, exit_sign, person, door, stairs. 점수 0.35 이상, NMS IoU 0.5. Jetson 한 장 **10.3 ms**.
- **object_mapper**: 탐지와 깊이를 `message_filters` **ApproximateTime** 으로 짝짓고(0.05 s 안), 마스크 픽셀 깊이의 가운데 값(마스크 없으면 상자 가운데 50 %의 하위 30 % 깊이)을 쓴다. 0.2~5 m 만. camera_info(fx, fy, cx, cy)로 역투영해 3D → 영상 시각의 TF(map ← 깊이 카메라)로 지도 좌표 → 같은 클래스가 반경(0.8 m, 연기 1.2, 문 1.0, 계단 1.6) 안이면 합침 → 5번 이상 본 것만 표시.
- 두 노드는 `component_container_mt` 하나(`perception_container`)에 컴포넌트로 올린다.
- 서비스 `/object_mapper/clear` (`std_srvs/Trigger`): 지도에 올린 물체를 모두 지운다.
- 한계: 지금 실물 깊이는 컬러에 정렬하지 않은 640×400 이다. object_mapper 는 깊이 영상과 깊이 camera_info 의 비율(640/640, 400/400 = 1)로 맞추므로 컬러 640×480 기준 탐지 좌표가 그대로 들어가 세로가 어긋나고, 마스크(640×480)는 camera_info(640×400)와 크기가 달라 버려져 상자 깊이로 잰다. 시뮬은 정렬돼 있어 맞다. 고칠 방법: `orbbec_dabai_dc1.yaml` 에 `depth_registration: true` ([sensor_topics.md 주의 1](sensor_topics.md#주의-1-깊이가-컬러에-정렬되지-않게-바뀐다)).

---

## 8. 액션 — 어떻게 주나

### 8-1. 토픽·서비스·액션 차이
| | 방식 | 우리 예 |
|---|---|---|
| 토픽 | 발행/구독, 한쪽으로 계속 흐름 | 센서 값, `/cmd_vel` |
| 서비스 | 요청 → 응답 한 번 | 라이다 `/stop_scan`·`/start_scan`, costmap `clear_entirely_*`, lifecycle 전환 |
| 액션 | 목표 → **피드백**(진행 중) → 결과, **취소 가능**. 안에서는 goal·cancel·result 서비스 + feedback·status 토픽 | Nav2 `navigate_to_pose` |

### 8-2. 목표를 주는 세 가지 방법
1. **RViz 2D Goal Pose**: 지도에서 누르고 끌어 위치·방향 → `/goal_pose`(PoseStamped) → bt_navigator 가 받아 `navigate_to_pose` 액션을 스스로 시작.
2. **명령줄**:
   ```bash
   ros2 action list                                # /navigate_to_pose 등
   ros2 action info /navigate_to_pose              # 서버: bt_navigator
   ros2 action send_goal /navigate_to_pose nav2_msgs/action/NavigateToPose \
     "{pose: {header: {frame_id: map}, pose: {position: {x: 1.0, y: 0.5}, orientation: {w: 1.0}}}}" --feedback
   ```
   피드백으로 현재 위치·남은 거리·복구 횟수가 온다.
3. **여러 점 순찰**: `follow_waypoints` 액션(waypoint_follower) — 점마다 `navigate_to_pose` 를 차례로 부른다.

### 8-3. `navigate_to_pose` 안에서 일어나는 일 (행동 트리)
```
NavigateRecovery (재시도 6번)
├── PipelineSequence
│   ├── RateController 1 Hz → ComputePathToPose  ── 액션 compute_path_to_pose → planner_server
│   │                          실패 시 전역 costmap 지우기 (서비스)
│   └── FollowPath           ── 액션 follow_path → controller_server (20 Hz 로 cmd_vel_nav)
│                              실패 시 지역 costmap 지우기
└── 복구 (번갈아 하나씩): costmap 둘 다 지우기 → Spin 1.57 rad → Wait 5 s → BackUp 0.30 m (0.05 m/s)
```
- 1 Hz 로 경로를 다시 짜면서 따라간다. 막히면 복구 동작도 각각 액션(`spin`, `backup`, `wait`)으로 behavior_server 에 요청한다.
- smoother_server 는 떠 있지만 기본 행동 트리에 SmoothPath 가 없어서 지금은 안 쓴다.

---

## 9. QoS · 파라미터 · 시간

- **QoS**: `/scan` 은 BEST_EFFORT (늦은 스캔은 버리는 게 낫다). 카메라는 RELIABLE 로 내고 우리 노드는 SensorData(BEST_EFFORT)로 받는다 — 받는 쪽이 더 느슨하면 연결된다. `/map`·`/tf_static`·물체 마커는 TRANSIENT_LOCAL — 늦게 켠 RViz 도 마지막 값을 받는다(녹화 bag 의 QoS 기록으로 확인).
- **파라미터**: 노드마다 yaml(`limo_base.yaml`, `ydlidar.yaml`, `orbbec_dabai_dc1.yaml`, `ekf.yaml`, `nav2_params.yaml`, `yolo_fire.yaml`)을 launch 의 `<param from=…>` 로 넣는다. 시뮬은 `nav2_sim.yaml` 로 다른 것만 덮는다.
- **시간**: 실물은 벽시계. 시뮬은 Webots 가 `/clock` 을 내고 모든 노드에 `use_sim_time: true`. 틀리면 costmap 이 스캔을 "오래됐다"며 버린다.

---

## 10. 시뮬레이션 (Webots) — 실물과 무엇이 같고 다른가

- 교수님 센서 플러그인(`webots_ros2_limo` `limo_sensors`)이 실물과 **같은 이름·타입·프레임**으로 `/scan`(10 Hz) `/imu`(100 Hz) `/odom`(50 Hz) `/camera/*`(약 33 Hz)를 낸다. 그래서 Cartographer·Nav2·YOLO launch 를 그대로 쓴다.
- 다른 점: 바퀴는 ros2_control `diffdrive_controller` 가 받는다(`/diffdrive_controller/cmd_vel_unstamped` 를 `/cmd_vel` 로 리맵). odom→base_link 는 EKF 대신 플러그인이 낸다. `/clock` 은 Ros2Supervisor.
- 교수님 `robot_launch.py` 를 쓰지 않는 까닭: 드라이버가 로봇 모델을 1 초만 기다려 두 번에 한 번꼴로 TF 가 안 나왔다 → Webots·컨트롤러를 직접 띄우고 우리 URDF 로 TF.
- 맵 4개(교수님 방, 덕트호스 미로, 사무실, 창고) — `sim.launch.xml world:=<이름>`. 순찰점 전부 도착(미로 13·사무실 14·창고 13·교수님 방 6).
- `limo_fire_sim` 의 `gt_camera` 플러그인이 정답 `/gt/detections`·`/gt/mask` 를 내고, `perception_benchmark` 가 YOLO 결과와 비교해 P·R·F1·mAP 표를 쓴다.

---

## 11. 녹화와 분석
- `record.launch.xml name:=MMDD_상황` — 늘 같은 토픽(`/scan /imu /odom /odometry/filtered /limo_status /rc_state /motor_states /wheel_odometry /cmd_vel /tf /tf_static /joint_states /map /trajectory_node_list` + 카메라)을 `~/limo_bags/` 에. 깊이 원본은 초당 약 15 MB.
- bag 분석 도구(C++, `limo_fire_sim`): `scan_stats_bag`(정지 라이다 오차 통계), `bag_export`(스캔·속도 내보내기 → 시뮬과 빔마다 비교), `motion_export`(명령 → 실제 속도 응답).
- 녹화 목록·조건: `train/explore/sim/analysis/bags.md`. 녹화 파일은 GitHub Release `bags-1103`.

---

## 12. 내가 한 것 (juyoung020 커밋, 날짜순)

| 날짜 | 한 일 |
|---|---|
| 10-22 | 워크스페이스 시작: ROS 2 패키지 구조·작업 규칙·`env.sh`, WeGo·교수님 Webots 코드를 `src/external` 에 |
| 10-23 | `limo_fire_description`: LIMO Pro URDF·RViz launch · 맵 변환기(Rust) · GPU 학습 환경 커널(CUDA) · YOLO 데이터(SAM 3 자동 라벨 6.8만 장)·C++/CUDA YOLO 학습기 |
| 10-26 | **실물 실행**: WeGo 대시보드의 센서·매핑·맵 저장·Nav2 를 XML launch(`limo_fire_bringup`·`limo_fire_navigation`)로 · Webots 시험 월드(불·연기·소화기 등) · TensorRT YOLO C++ 탐지 노드 · Jetson Webots 라이다 inf 버그(aarch64) 패치 |
| 10-27 | **object_mapper**: 탐지 → 깊이·TF 로 지도 좌표, 아이콘·현황판 · YOLO26s-seg(탐지+분할) 학습과 노드 적용 |
| 10-28 | YOLO 전처리·마스크 GPU 로(한 장 10.3 ms), 컴포넌트로 묶음, jetson_clocks, 깊이 19 → 30 Hz · CHANGELOG · 규약 문서 |
| 10-29 | Webots 맵 3개와 `sim.launch.xml world:=` · RViz 운용 화면 하나 · Nav2 를 RPP·Smac 2D 로 · 인식 벤치마크(정답 카메라 플러그인 + `perception_benchmark`) |
| 11-03 | **`limo_fire_base`**: 섀시 드라이버를 가져와 리모컨·모터·바퀴 거리 토픽 추가, `base:=false` · 녹화 launch · 실물 녹화로 맞춘 탐사 RL GPU 시뮬 |
| 11-04 | 커밋 규칙 검사 도구(Rust)·훅 · 탐사 RL 덮음·끝 판정, 프론티어 기준선 |

팀원: woojoo1116 — 실물 TF 를 우리 URDF 하나로(카메라 장착 실측), leeyechan — `sensor_topics.md`, Luke — teleop 사용법·`.gitignore`.

---

## 13. 서술형 예상 문제와 답

**Q1. 로봇은 자기 위치를 어떻게 아는가?**
섀시가 바퀴 속도를 적분한 `/odom`(50 Hz)과 IMU `/imu`(100 Hz)를 robot_localization EKF 가 합쳐 `/odometry/filtered`(30 Hz)와 TF odom→base_link 를 낸다. 이것은 매끄럽지만 미끄러짐으로 누적 오차가 생긴다. 지도 만들 때는 Cartographer 가 `/scan` 을 서브맵에 맞추고 루프를 닫아, 주행할 때는 AMCL 이 입자 필터로 `/scan` 을 저장 지도에 맞춰 TF map→odom 을 낸다. 최종 위치는 TF 체인 map→odom→base_link 로 얻는다.

**Q2. 목표를 찍으면 바퀴가 돌기까지 데이터가 어떻게 흐르나?**
RViz 2D Goal Pose → `/goal_pose` → bt_navigator 가 `navigate_to_pose` 액션 시작 → 행동 트리가 1 Hz 로 `compute_path_to_pose` 액션을 planner_server(Smac 2D, 전역 costmap)에 보내 경로를 받고, `follow_path` 액션으로 controller_server(RPP, 20 Hz, 지역 costmap)가 따라간다 → `cmd_vel_nav` → velocity_smoother(가속 제한) → `/cmd_vel`(Twist) → limo_fire_base 가 UART `0x111` 프레임으로 섀시에 → 모터. 위치는 TF 와 `/odometry/filtered` 로 계속 읽고, 막히면 costmap 지우기·Spin·Wait·BackUp 으로 복구한다.

**Q3. 라이다는 어디에 쓰이나?**
`/scan`(10 Hz, ±110°)은 Cartographer(SLAM), AMCL(위치 추정), 전역·지역 costmap 장애물 층(3 m·1 m 안 장애물 표시, 4 m·1.5 m 까지 지우기)에서 쓴다. QoS 는 BEST_EFFORT.

**Q4. 카메라로 본 불을 지도에 어떻게 올리나?**
yolo_detector 가 컬러 영상에서 상자·마스크를 낸다 → object_mapper 가 같은 시각 깊이 영상과 짝짓고(ApproximateTime 0.05 s) 마스크 안 깊이와 camera_info 내부 파라미터로 핀홀 역투영 → 그 영상 시각의 TF 로 map 좌표 → 같은 클래스 근처면 합치고 5번 이상 본 것만 MarkerArray 로 표시. TF 에 시각이 있어 움직이는 중에도 찍은 순간의 자세를 쓴다. 단 실물은 지금 깊이가 컬러에 정렬되지 않아 마스크를 못 쓰고 위치가 어긋날 수 있다(시뮬은 맞음, `depth_registration: true` 로 고칠 예정).

**Q5. TF 를 왜 한 발행자만 내게 하나?**
같은 변환을 둘이 내면 번갈아 덮어써 로봇 위치가 튄다. 그래서 odom→base_link 는 EKF 만(섀시 `pub_odom_tf: false`), map→odom 은 Cartographer 와 AMCL 중 하나만 켠다.

**Q6. 실물과 시뮬을 어떻게 같은 코드로 돌리나?**
교수님 센서 플러그인이 실물과 같은 토픽·프레임으로 내서, "로봇 켜기" launch 만 다르고 mapping·navigation·yolo launch 는 같다. 차이는 `use_sim_time`·`odom_topic`·초기 위치 같은 파라미터를 `nav2_sim.yaml` 로 덮는 것뿐이다.

**Q7. 컴포넌트 컨테이너를 왜 쓰나?**
여러 노드를 한 프로세스에 올려 프로세스 수와 메모리를 줄이고, 같은 프로세스 안 전달이 가능하다. Nav2 전체를 `nav2_container` 하나에, 카메라는 `camera_container`, YOLO·object_mapper 는 `perception_container` 에 올렸다.

---

## 14. 직접 확인하는 명령

```bash
source ~/limo_fire_ws/env.sh
ros2 launch limo_fire_bringup robot.launch.xml        # 먼저

rqt_graph                                  # 노드 ↔ 토픽 그림
ros2 run rqt_topic rqt_topic               # 토픽 목록 + 메시지 항목 트리 + Hz
ros2 run rqt_tf_tree rqt_tf_tree           # TF 트리
ros2 node list ; ros2 node info /ekf_filter_node_odom
ros2 topic list -t ; ros2 topic hz /scan ; ros2 topic info -v /scan
ros2 interface show nav_msgs/msg/Odometry
ros2 action list -t ; ros2 service list
ros2 bag info ~/limo_bags/1103_M_drive3    # 토픽별 메시지 수 → 주기
```
