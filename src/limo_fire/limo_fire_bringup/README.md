# limo_fire_bringup

실물 LIMO Pro 켜기. WeGo 대시보드 버튼이 하던 일(`~/limo_ros2_start_shortcut/bashes/*.sh` → `wego` 패키지 launch)을 우리 XML launch 로 옮겼다. 값은 WeGo 그대로다.

```bash
source ~/limo_fire_ws/env.sh                                   # 실물이면 ~/wego_ws 드라이버 + ROS_LOCALHOST_ONLY=1
ros2 launch limo_fire_bringup robot.launch.xml [viz:=true]     # 1. 센서·구동 (항상 먼저)
ros2 launch limo_fire_bringup mapping.launch.xml               # 2-a. 지도 만들기 (Cartographer + RViz)
ros2 launch limo_fire_bringup save_map.launch.xml [name:=map]  #      지도 저장
ros2 launch limo_fire_bringup navigation.launch.xml            # 2-b. 저장한 지도로 주행 (AMCL + Nav2 + RViz)
```

매핑과 네비게이션은 동시에 켜지 않는다(둘 다 `/map` 을 낸다). 네비게이션은 RViz 에서 **2D Pose Estimate** 로 처음 위치를 찍어야 시작한다.

## 키보드로 조종 (teleop)

바퀴만 움직이면 되면 `limo_base` 만 켠다. 센서도 필요하면 대신 `robot.launch.xml` 을 켠다(둘 다 `/dev/ttylimo` 를 열어서 같이 켜면 안 된다).

```bash
source ~/limo_fire_ws/env.sh
ros2 launch limo_base limo_base.launch.py      # 구동 드라이버 (/cmd_vel 받음)

# 새 터미널 — 키 입력을 받으므로 이 창이 선택돼 있어야 한다
source ~/limo_fire_ws/env.sh
ros2 run teleop_twist_keyboard teleop_twist_keyboard --ros-args -p speed:=0.2 -p turn:=0.5
```

기본 속도(0.5 m/s, 1.0 rad/s)는 실내에서 빨라 낮춰 시작한다. `i`/`,` 앞뒤, `j`/`l` 제자리 회전, `u` `o` `m` `.` 대각선, **`k` 정지**, `q`/`z` 속도 ±10 %, `Ctrl+C` 종료. 다른 창을 누르면 마지막 명령이 남을 수 있으니 먼저 `k`.

## WeGo 와 대응

| 대시보드 버튼 | WeGo | 우리 |
|---|---|---|
| 센서 | `wego/teleop_launch.py` | `robot.launch.xml` |
| 맵 | `wego/cartographer_launch.py` | `mapping.launch.xml` |
| 맵 저장 | `save_map.sh` | `save_map.launch.xml` |
| 네비게이션 | `wego/navigation_diff_launch.py` | `navigation.launch.xml` |
| 차선 인식 | `limo_ros2_application/track_application_launch.py` | `lane_tracking.launch.xml` (로봇이 움직임) |
| 딥러닝 주행 | `dl_ros2_application/dl_track_application.launch.py` | `dl_tracking.launch.xml` (로봇이 움직임, `conda activate dl_env` 필요) |

나머지 버튼은 명령 한 줄이라 launch 를 따로 만들지 않았다.

| 버튼 | 명령 |
|---|---|
| 사진 찍기 | `ros2 run dl_ros2_application take_a_picture` |
| 사진 저장 (TensorFlow용 / YOLO용) | `ros2 service call /photoshoot dl_ros2_msgs/srv/PhotoShoot "{chalkak: true}"` / `false` |
| YOLO 라벨링 | `ros2 run dl_ros2_application yolo_labeler` |
| TensorFlow → TensorRT | `ros2 run dl_ros2_application tf_to_trt <.h5>` (dl_env) |
| YOLO → TensorRT | `ros2 run dl_ros2_application yolo_to_trt <.pt>` (dl_env) |

| 파일 | 원본 |
|---|---|
| `config/ydlidar.yaml` | `ydlidar_ros2_driver/params/ydlidar.yaml` |
| `config/ekf.yaml` | `robot_localization/params/limo_ekf.yaml` |
| `config/orbbec_dabai_dc1.yaml` | `orbbec_camera/launch/astra_stereo_u3.launch.py` 의 인자 기본값 |
| `limo_fire_navigation/config/nav2_params.yaml` | `wego_2d_nav/params/diff_navigation_params.yaml` |
| `limo_fire_navigation/config/cartographer_2d.lua` | `wego/config/limo_lds_2d.lua` |
| `limo_fire_description/rviz/{sensors,mapping,navigation}.rviz` | `wego/rviz/{display,cartographer,navigation}.rviz` |
| `limo_fire_navigation/maps/map.*` | 2026-10-26 실험실에서 만든 지도 (9.4 × 6.1 m) |

WeGo 와 다른 점 (성능 측정은 limo_fire_perception README·2026-10-28 측정):
- `robot.launch.xml` 이 시작할 때 `sudo -n jetson_clocks` 로 CPU·GPU·메모리 클럭을 최대로 고정한다 (`max_clocks:=false` 로 끔, 재부팅하면 풀림).
  카메라처럼 33 ms 마다 한 장이면 장 사이에 GPU 클럭이 내려가 YOLO 가 21 ms 걸리고, 고정하면 11 ms. 대신 대기 전력이 는다
- `orbbec_dabai_dc1.yaml`: 아무도 안 쓰는 포인트 클라우드 2 개·IR 영상을 껐다. 드라이버 CPU 42 → 35 %, 깊이 영상 19 → 30 Hz (카메라만 띄워 30 초씩 잼)
- `cartographer_2d.lua`: 회전 움직임 필터 0.1° → 0.5° (CPU -20 %, 궤적 차 1.5~1.8 cm), 자세 발행 5 → 20 ms (TF 트래픽). 파일 머리 주석에 비교 방법
- `save_map.launch.xml` 의 지도 대기 시간을 2 초 → 10 초로 늘렸다. 2 초로는 `/map`(1 초마다 발행)을 못 받고 실패하는 일이 있었다 — 대시보드 "맵 저장"이 파일을 안 바꾼 원인으로 보인다. 저장 위치는 WeGo 처럼 src 와 install 두 곳이라 다시 빌드하지 않아도 네비게이션이 새 지도를 연다.

드라이버(`limo_base`, `orbbec_camera`, `ydlidar_ros2_driver`, `robot_localization`, `limo_description`)는 로봇의 `~/wego_ws` 에 빌드된 것을 쓴다. 대시보드로 켠 것과 동시에 켜면 같은 장치를 두 번 열어 실패하니 한쪽만 쓴다.

## 검증 (2026-10-26, 실물)

- `robot.launch.xml`: WeGo 와 같은 노드 12 개, `/scan` 10 Hz · `/imu` 94~100 Hz · `/odom` 50 Hz · `/odometry/filtered` 30 Hz · 컬러 29 Hz · 깊이 28 Hz, `base_link → camera_link` (0.23, 0.05, 0.06) 로 WeGo 와 같음. 기동 때 찍히는 오류 두 줄(EKF `differential`·`relative`, 카메라 `Failed to setup devices`)도 WeGo 와 같고 동작에는 지장 없음.
- `navigation.launch.xml`: Nav2 노드 11 개 컨테이너에 올라감, 처음 위치를 주면 전부 active, `map → base_link` 나옴.
- `mapping.launch.xml`: `/map` 1 Hz. `save_map.launch.xml`: 두 곳 모두 저장 성공.

## 그대로 옮겨서 남은 것

WeGo 값을 바꾸지 않았기 때문에 아래는 아직 그대로다 ([hardware.md](../../../docs/hardware.md) 참고).

- 카메라 위치는 WeGo 틸트 TF 값 (0.23, 0.05, 0.06). 실측으로는 약 (0.15, −0.005) — `limo_fire_description` 으로 옮길 때 고친다.
- 카메라 TF 를 launch 의 `static_transform_publisher` 로 낸다(CLAUDE.md 는 URDF 한 곳에 두라고 함).
- EKF 설정 충돌(`odom0_differential`·`odom0_relative` 둘 다 true), IMU 공분산 1e-6.
