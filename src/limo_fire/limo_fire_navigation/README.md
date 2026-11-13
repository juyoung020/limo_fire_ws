# limo_fire_navigation

Nav2·Cartographer 파라미터(`config/`), 저장 지도(`maps/`), 연기 모드 노드(`smoke_mode`).
실행은 `limo_fire_bringup` 의 `navigation.launch.xml`·`sim.launch.xml` 이 한다.

## 연기 모드

연기 속에서 저장 지도 위를 **바퀴 오도메트리 + IMU 만으로** 움직인다. 라이다·카메라는 쓰지 않는다.

```bash
ros2 service call /smoke_mode/set std_srvs/srv/SetBool "{data: true}"    # 켬
ros2 service call /smoke_mode/set std_srvs/srv/SetBool "{data: false}"   # 평상시로
ros2 topic echo /smoke_mode/active                                        # 지금 상태 (마지막 값 유지)
```

저장 지도 위치 추정(`navigation.launch.xml`, 기본 `localization:=true`)일 때만 켜진다. 지도 만들기(SLAM) 중이거나,
아직 AMCL 자세·지도·`odom → base_link` TF 가 없으면 켜기를 거절하고 까닭을 돌려준다.

### 어떤 센서를 쓰나

지금 로봇에 달린 센서([hardware.md](../../../docs/hardware.md)) 가운데 연기 속에서 동작하는 것만 쓴다.

| 센서 | 연기 속 | 연기 모드에서 |
|---|---|---|
| 바퀴 오도메트리 | 빛을 안 써서 영향 없음 | **씀** — 이동량 |
| IMU | 빛을 안 써서 영향 없음 | **씀** — 방향(EKF), 부딪힘 감지 |
| 2D 라이다 YDLIDAR T-mini Plus | 연기 입자에 반사돼 거리가 틀어짐 | 안 씀 |
| 깊이·컬러 카메라 Orbbec Dabai DC1 | 보이는 빛·적외선 패턴이 연기에 가림 | 안 씀 |
| 초음파 HC-SR04 × 2 (앞 범퍼 아래 좌우 ±20°) | 빛이 아니라 소리라 덜 막힐 것으로 본다(잠정 — Starr & Lattimer 2014 시험 대상에 sonar 2 종이 있으나 결과 수치는 원문 확인 필요, 열기·기류 영향도 확인 필요) | **`smoke_localization: sonar` 일 때 씀** — AMCL 위치 보정, 장애물. 시뮬엔 있음, 실물은 장착 전([limo_fire_esp32_bridge](../limo_fire_esp32_bridge/README.md)) |
| 레이더·열화상 | — | 아직 안 달아서 다루지 않음 (부품 대기, [spec.md](../../../docs/spec.md#연기-모드)) |

근거: Starr & Lattimer, "Evaluation of Navigation Sensors in Fire Smoke Environments", *Fire Technology* 50, 2014
([doi:10.1007/s10694-013-0356-3](https://link.springer.com/article/10.1007/s10694-013-0356-3)) — 항법 센서 11 종을 연기 속에서
비교해 레이더·적외선(열) 카메라가 가장 좋았고, LIDAR 는 시정 4 m 아래에서 거리 오차가 생겼다. 깊이 카메라(Kinect)와
가시광 카메라도 시험 대상이었다. 연기의 농도에 따라 라이다가 부분적으로 쓸 만할 수는 있지만, 어디까지 믿을지 잴 수 있는
센서가 지금은 없어서 연기 모드에서는 통째로 뺐다. Dabai DC1 은 근적외선 패턴을 쏘는 깊이 카메라라 시험된 Kinect 와 같은 방식이다(열화상 카메라와 다르다).

### 동작

`smoke_mode` 노드는 평상시에도 늘 떠 있고, 모드에 따라 아래를 바꾼다.

| 할 일 | 평상시 | 연기 모드 |
|---|---|---|
| `map → odom` TF | AMCL 자세(`/amcl_pose`)로 계산해 냄 | `hold`: 들어갈 때 값을 붙잡아 냄 → 위치는 EKF(바퀴 + IMU) 만 따라감 · `sonar`: 바퀴 + IMU 추정과 가까운 AMCL 자세만 조금씩 반영(`sonar.gate_*`·`blend`), 잇달아 어긋나면 AMCL 을 추정 자리로 다시 놓음 |
| AMCL 이 받는 스캔 `scan_localization` | `/scan` 그대로 | `hold`: 끊음 · `sonar`: 초음파 스캔(`sonar_scan`) |
| 장애물 지도가 받는 라이다 `scan_obstacles` | `/scan` 그대로 | 저장 지도에서 광선을 쏜 **가상 스캔** — 지도의 벽만 남고 연기가 찍은 가짜 장애물은 지워짐 |
| 최고 속도 (`speed_limit`) | 한도 없음 | 0.13 m/s, 앞 초음파 0.6 m 안이면 0.05 m/s (잠정) |
| 추정 자세 `smoke_mode/pose` | — | 자세 + 이동·회전에 비례해 커지는 불확실성 |
| 부딪힘 `smoke_mode/bump` | — | IMU 수평 가속도가 4 m/s² (잠정) 넘으면 그 지도 위치 |

- 연기 속 위치 방식 `smoke_localization`(`config/smoke_mode.yaml`, 기본 `hold`)은 연기 모드가 꺼져 있을 때만 바꿀 수 있다: `ros2 param set /smoke_mode smoke_localization sonar`.
- `sonar_scan` 노드: 초음파 2 개 Range → base_link 기준 60 칸 스캔(원뿔 15° 폭에 점 3 개, 나머지 칸 +inf = AMCL 이 안 씀). 칸 수가 AMCL `max_beams`(60) 이하라야 AMCL 이 몇 안 되는 점을 버리지 않는다.
- 초음파는 장애물 지도(전역·지역)의 `range_layer`(`nav2_costmap_2d::RangeSensorLayer`, 확률로 찍고 지움)로도 들어간다 — 연기 속 지도에 없는 장애물(4 단계).
- `map → odom` 은 이 노드 하나만 낸다. 그래서 AMCL 은 `tf_broadcast: false` 다(`nav2_params.yaml`).
- 들어갈 때·나올 때 장애물 지도를 비운다(연기 전·후의 라이다 흔적을 지움).
- 나올 때 지금 추정 자세를 쌓인 불확실성과 함께 `/initialpose` 로 AMCL 에 넘긴다. AMCL 은 다시 들어오는 라이다로 그 근처에서 맞춘다.
- 연기 속에서 사람이 RViz `2D Pose Estimate` 로 자세를 고쳐 주면 그 자리로 옮기고 불확실성을 처음 값으로 되돌린다.
- 부딪힘은 알리기만 한다. 멈추기·후진·다른 목표·보고는 BT(`limo_fire_bt`, 아직 없음)가 이 토픽을 받아 한다.

### 한계

- 라이다·카메라가 없으니 새 장애물(쓰러진 물건, 사람)은 초음파로만 본다. 초음파가 없으면 장애물 지도에는 저장 지도의 벽만 있어 부딪힌 뒤에야 IMU 로 안다. 초음파를 켜도 시뮬 상자에 3 판 중 1 판은 닿았다.
- 위치는 이동할수록 흐려진다. 오차가 얼마나 빨리 커지는지는 바퀴 오도메트리 주행 시험(1 m 직진·360° 회전, hardware.md "확인할 것") 뒤에 `drift.*` 를 맞춘다.
- 연기를 스스로 알아채지 않는다. 켜고 끄기는 서비스로 한다(시연 버튼, 나중에 BT).
- 잠정 값(`config/smoke_mode.yaml`): 속도 한도, 불확실성 계수, 부딪힘 문턱. 메카넘 바퀴 주행 진동을 재서 부딪힘 문턱을 다시 정한다.

### 확인한 것

- 2026-11-04 (데스크톱, Jazzy): 빌드 경고 0, 가상 스캔 광선 단위 테스트 통과. 노드만 띄워(map_server + 고정 TF + 가짜 `/amcl_pose`)
  `map → odom` 계산, 켜기 거절(지도 없음), 연기 속 `scan_localization` 끊김, 속도 한도 0.13 → 나올 때 0, 가상 스캔, 나올 때 `/initialpose` 확인.
- 2026-11-05 (로봇 Jetson, Humble + Webots R2025a, `nav_benchmark`): Webots 맵 4 개 순찰점 46 개를 (a) 라이다 켬 · (b) 연기 모드(바퀴 + IMU 만)
  두 조건으로 다 도착. 결과와 한계는 [docs/benchmarks "주행 벤치마크"](../../../docs/benchmarks/README.md#주행-벤치마크-연기-모드).
- 부딪힘 감지는 미로 맵 (b) 에서 15 번 떴고 모두 오탐(실제 충돌 0) — 제자리 회전·출발 가속에 걸린다. 문턱(잠정 4 m/s²)을 다시 정해야 한다.
- 2026-11-13: 초음파를 실물 받침대대로 2 개(앞 좌우 ±20°)로 바꾸니 `hold` 는 네 맵 다 도착, `sonar` 는 더 나빠짐(maze·office 에서 위치를 잃음) — [benchmarks 결과 5](../../../docs/benchmarks/README.md).
- 2026-11-06 (로봇 Jetson, 거리 −3 % · 초음파 켬): **`smoke_localization: sonar` 가 `hold` 보다 네 맵 모두 나빴다**(office·warehouse 에서 위치를 잃음),
  지도에 없는 상자는 초음파를 켜도 못 피했다 → [docs/benchmarks "결과 2"](../../../docs/benchmarks/README.md). `sonar` 는 실험 단계 — 연기 모드 기본은 `hold`.
- 2026-11-06 오후 (같은 조건, 고친 뒤): `hold` 는 네 맵 다 끝까지 갔다. `sonar` 는 나아졌지만 여전히 네 맵 모두 `hold` 보다 부정확했다(warehouse 11/13, 최대 1.70 m).
  상자는 초음파 끔 3/3 닿음 → 초음파 켬 1/3 닿음(`hold`·`sonar` 같음). 물체가 있어도 초음파 메아리가 없는 때가 21~39 % 다 → [docs/benchmarks "결과 3"](../../../docs/benchmarks/README.md). 다음 할 일: [smoke_mode_plan.md "다음에 볼 것"](../../../docs/smoke_mode_plan.md).
- **아직 안 함**: 실물 주행, 실물 수준 오도메트리 오차에서의 측정.

## 주행 벤치마크 `nav_benchmark` (시뮬 전용)

월드 데이터 파일(`sim/webots/worlds/<월드>.yaml`)의 순찰점을 Nav2 로 차례로 가며 Webots 정답 자세와 비교해 `docs/benchmarks/` 에 표(md)·CSV 를 쓴다.
정답은 `limo_fire_sim` `GroundTruthPose`(월드에서 LIMO `supervisor TRUE`) 의 `/sim/ground_truth` — 평가에만 쓰고 주행·위치 추정에는 안 쓴다.

```bash
ros2 launch limo_fire_bringup sim.launch.xml world:=limo_fire_maze rviz:=false        # 먼저 시뮬 (Nav2 가 다 켜질 때까지)
ros2 run limo_fire_navigation nav_benchmark --ros-args -p use_sim_time:=true \
  -p world_file:=sim/webots/worlds/limo_fire_maze.yaml -p label:=wheel_imu -p smoke:=true -p report_dir:=docs/benchmarks
```

- `smoke:=true` 면 처음 위치를 라이다(AMCL)로 잡은 상태에서 연기 모드를 켜고 출발한다.
- 목표마다: 결과(성공·중단·시간 초과), 걸린 시간(시뮬), 도착 오차 |정답 − 목표|, 추정 오차 |Nav2 자세 − 정답|, 정답으로 잰 이동 거리.
