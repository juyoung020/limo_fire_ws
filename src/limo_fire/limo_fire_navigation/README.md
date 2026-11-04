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
| 레이더·초음파·열화상 | — | 아직 안 달아서 다루지 않음 (부품 대기, [spec.md](../../../docs/spec.md#연기-모드)) |

근거: Starr & Lattimer, "Evaluation of Navigation Sensors in Fire Smoke Environments", *Fire Technology* 50, 2014
([doi:10.1007/s10694-013-0356-3](https://link.springer.com/article/10.1007/s10694-013-0356-3)) — 항법 센서 11 종을 연기 속에서
비교해 레이더·적외선(열) 카메라가 가장 좋았고, LIDAR 는 시정 4 m 아래에서 거리 오차가 생겼다. 깊이 카메라(Kinect)와
가시광 카메라도 시험 대상이었다. 연기의 농도에 따라 라이다가 부분적으로 쓸 만할 수는 있지만, 어디까지 믿을지 잴 수 있는
센서가 지금은 없어서 연기 모드에서는 통째로 뺐다. Dabai DC1 은 근적외선 패턴을 쏘는 깊이 카메라라 시험된 Kinect 와 같은 방식이다(열화상 카메라와 다르다).

### 동작

`smoke_mode` 노드는 평상시에도 늘 떠 있고, 모드에 따라 아래를 바꾼다.

| 할 일 | 평상시 | 연기 모드 |
|---|---|---|
| `map → odom` TF | AMCL 자세(`/amcl_pose`)로 계산해 냄 | 들어갈 때 값을 붙잡아 냄 → 위치는 EKF(바퀴 + IMU) 만 따라감 |
| AMCL 이 받는 라이다 `scan_localization` | `/scan` 그대로 | 끊음 |
| 장애물 지도가 받는 라이다 `scan_obstacles` | `/scan` 그대로 | 저장 지도에서 광선을 쏜 **가상 스캔** — 지도의 벽만 남고 연기가 찍은 가짜 장애물은 지워짐 |
| 최고 속도 (`speed_limit`) | 한도 없음 | 0.13 m/s (잠정) |
| 추정 자세 `smoke_mode/pose` | — | 자세 + 이동·회전에 비례해 커지는 불확실성 |
| 부딪힘 `smoke_mode/bump` | — | IMU 수평 가속도가 4 m/s² (잠정) 넘으면 그 지도 위치 |

- `map → odom` 은 이 노드 하나만 낸다. 그래서 AMCL 은 `tf_broadcast: false` 다(`nav2_params.yaml`).
- 들어갈 때·나올 때 장애물 지도를 비운다(연기 전·후의 라이다 흔적을 지움).
- 나올 때 지금 추정 자세를 쌓인 불확실성과 함께 `/initialpose` 로 AMCL 에 넘긴다. AMCL 은 다시 들어오는 라이다로 그 근처에서 맞춘다.
- 연기 속에서 사람이 RViz `2D Pose Estimate` 로 자세를 고쳐 주면 그 자리로 옮기고 불확실성을 처음 값으로 되돌린다.
- 부딪힘은 알리기만 한다. 멈추기·후진·다른 목표·보고는 BT(`limo_fire_bt`, 아직 없음)가 이 토픽을 받아 한다.

### 한계

- 연기 속에서는 새 장애물(쓰러진 물건, 사람)을 미리 못 본다. 장애물 지도에는 저장 지도의 벽만 있다. 부딪힌 뒤에야 IMU 로 안다.
- 위치는 이동할수록 흐려진다. 오차가 얼마나 빨리 커지는지는 바퀴 오도메트리 주행 시험(1 m 직진·360° 회전, hardware.md "확인할 것") 뒤에 `drift.*` 를 맞춘다.
- 연기를 스스로 알아채지 않는다. 켜고 끄기는 서비스로 한다(시연 버튼, 나중에 BT).
- 잠정 값(`config/smoke_mode.yaml`): 속도 한도, 불확실성 계수, 부딪힘 문턱. 메카넘 바퀴 주행 진동을 재서 부딪힘 문턱을 다시 정한다.

### 확인한 것 (2026-11-04)

- 빌드 경고 0, 가상 스캔 광선 단위 테스트 통과 (`colcon test --packages-select limo_fire_navigation`).
- ROS 2 Jazzy 데스크톱에서 노드만 띄워(map_server + 고정 TF + 가짜 `/amcl_pose`) 확인: `map → odom` 계산, 켜기 거절(지도 없음),
  연기 속 `scan_localization` 끊김, 속도 한도 0.13 → 나올 때 0, 가상 스캔 249 빔(덕트호스 미로 맵에서 232 빔이 벽에 닿음),
  나올 때 `/initialpose`.
- **아직 안 함**: Humble 로봇·Webots 에서 Nav2 와 같이 주행, 연기 속 목표 도착 오차.
