# Isaac Sim 교차 검증 (연기 모드 초음파)

연기 모드 검증 현황의 "다른 시뮬레이터와 교차 검증" 항목을 한 문서로 정리한다. 무엇을·왜·어떻게 했는지, 어디까지 믿을 수 있는지를 적는다.
숫자 결과는 [benchmarks/README.md](benchmarks/README.md) "결과 4"(초음파 3 개, 옛 설계) 와 "결과 6"(**현재 설계 2 개 재실행**), [20261113_0650_nav_isaac_cross_check.csv](benchmarks/20261113_0650_nav_isaac_cross_check.csv)·[20261114_1724_nav_isaac_sonar2_cross_check.csv](benchmarks/20261114_1724_nav_isaac_sonar2_cross_check.csv) 에 있다.

> **2026-11-14 갱신**: 아래 "환경·교차 방법·시험 조합"은 처음 돌린 **초음파 3 개** 배치 기준이다. 현재 실물 설계는 2 개(앞 좌우 ±20°·위 5°, `/ultrasonic/front_left`·`front_right`)라 이 배치로 office·warehouse 시드 0~2(18 판)를 다시 돌렸고, 결과 요약은 아래 "2 개 배치 재실행"에 있다. 3 개 배치 결과는 옛 설계 기록으로만 남긴다.

## 한눈에

- 질문: Webots 에서 나온 "초음파로 위치를 고치는 `sonar` 가 바퀴 + IMU 만 쓰는 `hold` 보다 나쁘다" 가 실물 초음파의 성질인가, Webots 센서 모델의 성질인가.
- 방법: 다른 시뮬레이터(Isaac Sim)에 같은 로봇·맵·Nav2·`smoke_mode`·벤치마크를 올리고, 초음파가 메아리를 돌려주는 기준만 `hard`(22.5° 에서 딱 끊음, Webots 와 같게)와 `soft`(25~55° 에서 서서히 줄어듦)로 바꿔 같은 시험을 했다.
- 결과: `hard` 에서는 office·warehouse 에서 `sonar` 가 `hold` 보다 나빴다(Webots 결과 재현). `soft` 에서는 `hold` 와 비슷한 수준으로 돌아왔다. 판정 규칙대로 "좋아짐" 쪽이므로, 결론을 믿기 전에 실물 초음파의 반사 각도를 먼저 재야 한다.
- 이 결과는 "`soft` 모양이 맞다" 는 뜻이 아니다. `soft` 의 25~55° 모양은 잠정값이고, 실물 반사 각도는 아직 안 쟀다.

## 왜 Isaac Sim 으로 교차 검증했나

1. 의심한 원인이 시뮬레이터 안에 있다. 결과 3 의 "물체가 있어도 초음파 메아리가 21~39 % 없다" 는 Webots `DistanceSensor` 가 입사각 22.5° 넘는 광선을 최대 거리로 돌려주기 때문일 가능성이 컸다. 같은 시뮬레이터만 다시 돌리면 그 가정을 못 가른다. 검증 현황에도 "시뮬레이터도 Webots 하나" 가 한계로 적혀 있다.
2. Isaac Sim 에는 초음파 센서가 없다. 그래서 거리를 PhysX 광선으로 직접 계산하고 반사 기준을 우리가 정한다. 반사 기준을 바꿔 가며 같은 시험을 하려는 이 실험에 맞다. Webots `DistanceSensor` 의 반사 기준은 센서 내부 동작이라 우리가 바꿔 본 적이 없다.
3. 물리 엔진·센서 구현·ROS 연결이 Webots 와 다르다. 같은 Nav2 를 다른 시뮬에 붙이면 "연기 모드 처리가 특정 시뮬레이터에서만 되는 것이 아닌지" 도 같이 본다.
4. 로봇 모델이 있다. WeGo LIMO USD(`limo_diff_thin.usd`, 팀 위키 첨부)가 있어서 로봇을 새로 만들 필요가 없었다.

한계도 있다. Isaac 쪽 초음파 모델도 우리가 만든 것이라 **실물 증거가 아니다.** 이 교차 검증이 말해 주는 것은 "결과가 반사 기준에 따라 바뀐다" 까지다. 어느 기준이 실물과 맞는지는 실물 측정으로만 가려진다.

## 환경

| 항목 | 값 |
|---|---|
| Isaac Sim | 6.2.0 (`6.2.0-rc.14+release.52805`), 창 없이(headless) 실행, `SimulationApp`, Python 3.12 |
| PC | 노트북 ROG Zephyrus G14, GPU RTX 5060 Laptop 8 GB (드라이버 580.178), Ubuntu 26.04 |
| 설치 위치 | 별도 파티션 `/mnt/isaac` (루트 용량 부족으로 분리) |
| ROS 2 연결 | Isaac Sim 확장 `isaacsim.ros2.bridge` + Isaac 에 들어 있는 Humble 라이브러리(`LD_LIBRARY_PATH` 로 추가), 시뮬 안에서 `rclpy` 노드로 발행·구독 |
| Nav2·`smoke_mode`·`nav_benchmark` | Docker 컨테이너(Ubuntu 22.04, ROS 2 Humble) 안, 저장소를 읽기 전용으로 붙여서 실행 |
| 호스트 ↔ 컨테이너 통신 | Fast DDS 를 UDP 만 쓰게 한 설정(기본 설정에서는 컨테이너가 `/clock` 토픽은 보여도 데이터를 못 받았음) |
| 시간 | Isaac 이 `/clock` 을 발행, 모든 ROS 노드 `use_sim_time:=true`. 프레임·물리 1/60 s 고정(물리 100 Hz 는 떨림이 더 커져서 안 씀) |
| 로봇 | WeGo LIMO USD 에 초음파 마운트 3 개(앞·왼·오른쪽)를 얹은 덮개 파일. 마운트 자리는 `limo_fire.urdf.xacro`·`LimoFire.proto` 와 같은 값 |

호환성 확인: Isaac Sim 6.2 ↔ ROS 2 Humble 은 Isaac 내장 Humble 라이브러리로 맞췄고, 빈 장면에서 `/clock`·토픽 교환을 먼저 확인한 뒤 본 시험을 했다.

## 교차 방법

### 무엇을 같게, 무엇을 다르게

| | Webots 판 | Isaac 판 |
|---|---|---|
| 로봇·맵·순찰점·시작 자세 | 월드 데이터 파일 | 같은 파일에서 가져옴(순찰점·`robot_start` 그대로) |
| 위치 추정·주행 소프트웨어 | 저장소의 Nav2·`smoke_mode`·`sonar_scan`·`nav_benchmark` | **같은 코드, 같은 설정** |
| 오도메트리 오차 | 바퀴 반지름 ×1.03 (`wheelRadiusScale`) | 같은 정의로 주입 |
| 초음파 사양 | 0.02~4 m, 15°, 광선 5 개, 잡음 1 % | 같은 값(잠정) |
| 장면 | `.wbt` 월드(벽·물체 원래 모양) | 저장한 지도(pgm)의 점유 칸을 높이 2.4 m 박스로 변환 |
| 구동 | 바퀴 물리 | 차체 속도를 직접 주는 운동학 구동 |
| 초음파 거리 | Webots `DistanceSensor` | PhysX 광선 + 우리 반사 모델 |
| 라이다 | Webots 라이다 | PhysX 광선 248 개(같은 시야·거리·잡음 값) |

### 1. 장면

`pgm_to_usda` 변환기(임시)가 `maps/sim/limo_fire_<맵>.pgm`·`.yaml` 의 점유 칸을 합쳐 직사각형 박스로 만들고, 지도 좌표를 `robot_start` 로 옮겨 월드 좌표에 놓는다. 지도 좌표 = 월드 좌표를 `robot_start` 로 옮기고 돌린 것이므로 `p_world = start + R(yaw)·p_map`. 높이는 지도에 없어 2.4 m 로 통일했다.
(처음에는 `.wbt` 의 벽(`Wall`)만 박스로 옮겼으나, Nav2 가 쓰는 지도와 실제 장면을 맞추려고 지도에서 만든 장면을 본 시험에 썼다.)

### 2. 구동과 슬립 주입

`/cmd_vel` 을 받으면 차체 속도(선속도 한도 0.5 m/s, 명령 시간 제한 0.5 s)를 직접 적용한다. 바퀴 마찰·벽 마찰이 속도를 깎지 않게 마찰 0 재질을 지도와 로봇에 묶었고, 충돌 반력은 PhysX 가 그대로 낸다.
바퀴를 물리로만 굴리면 회전이 비선형·불감대가 커서 오도메트리 시험이 안 됐다(마찰·솔버 설정을 바꿔도 같았다). 운동학 구동에서는 `0.5 rad/s × 6 s` 회전이 약 171°(이론 172°)로 맞았다.
**슬립 주입**: 오도메트리는 공칭 바퀴 반지름(0.045 m)으로 센 이동량이고, 실제 차체 속도는 `k = 1.03` 배로 준다. 실제 이동 ×1.03, 오도메트리 거리 1/1.03 (−2.9 %). Webots `wheelRadiusScale 1.03` 과 같은 정의다.

### 3. 센서와 TF

- `/odom`(50 Hz): Webots `limo_sensors.py` 와 같은 방식 — 거리는 바퀴 엔코더 이동량 평균, 헤딩은 IMU. `odom → base_link` TF 도 시뮬이 낸다(Webots 시뮬이 EKF 자리에서 내는 것과 같음).
- `/imu`(100 Hz): 물리 엔진의 루트 속도는 정지에서도 떨려(가속도 표준편차 약 5 m/s²) 쓰지 않았다. 적용한 평면 운동에서 이상적인 IMU 값을 만들고, 실물 측정값(`docs/hardware.md`: 자이로 0.002 rad/s·0.1°/s 단위, 가속도 배율 1.023 등)의 잡음을 얹었다. 벽에 막힌 충격 가속도는 안 나온다.
- `/scan`(10 Hz): 광선 248 개, 시야·거리 0.1~12 m·잡음 0.3 % 는 `LimoFire.proto` 와 같은 값.
- `/ultrasonic/{front,left,right}`(`sensor_msgs/Range`, 센서마다 10 Hz): 메아리 없음은 `+inf`.
- `/sim/ground_truth`: 평가 전용 정답(Webots 월드 좌표). 주행·위치 추정에는 안 쓰고 `nav_benchmark` 의 도착·추정 오차 계산에만 쓴다.

### 4. 초음파 반사 모델 (실험의 핵심)

센서 프림 앞 방향 1 개 + 가장자리 4 개(15° 원뿔의 절반 각)로 광선 5 개를 쏘고, 각 광선의 첫 충돌(로봇 자신 제외)에서 **입사각**(광선과 면 법선의 각)을 구한다.

- `hard`: 입사각 ≤ 22.5° 일 때만 메아리. Webots 판과 같은 기준.
- `soft`: 입사각 ≤ 25° 는 항상, 25~55° 는 확률이 1 에서 0 으로 직선으로 줄고(`p = 1 − (각 − 25)/30`), 55° 넘으면 없음. **모양·문턱은 잠정값**이다.

메아리가 온 광선들 중 가장 가까운 거리에 1 % 상대 잡음(가우시안)을 얹고, 0.02 m 미만은 `-inf`, 4 m 초과·메아리 없음은 `+inf` 로 낸다.

### 5. 실행 절차 (판마다)

1. Isaac 시뮬을 `--map <맵> --pose <robot_start> --wheel-radius-scale 1.03 --model hard|soft --seed <n>` 으로 켠다.
2. 컨테이너에서 `description` launch(로봇 TF) → `navigation.launch.xml`(지도 서버·AMCL·컨트롤러, 시뮬용 설정 `nav2_sim.yaml`) → 25 초 기다린 뒤 `nav_benchmark`(`smoke:=true`, `smoke_localization:=hold|sonar`) 순서로 켠다.
3. `nav_benchmark` 가 월드 데이터 파일의 순찰점을 차례로 주고, 목표마다 도착 오차(정답 위치 대비)·추정 오차(`smoke_mode` 추정 대비 정답)를 `.csv`·`.md` 로 쓴다.
4. 판이 끝나면 집계(임시 스크립트)로 맵·조건·시드별 표를 만든다.

### 6. 시험 조합과 판정

| 맵 | 조건 | 시드 |
|---|---|---|
| office, warehouse | `hold` / `sonar hard` / `sonar soft` | 0~4 (5 판씩, 총 30 판) |
| maze, world | 같은 3 조건 | 0 (1 판씩, 총 6 판) |

총 36 판. 시드는 초음파 잡음·반사 추첨·IMU 잡음의 난수만 바꾼다(순찰점·시작 자세 동일).
판정 규칙(검증 현황): 반사 기준을 부드럽게 해도 `sonar` 가 `hold` 보다 나쁘면 결론을 신뢰하고, 좋아지면 실물 반사 각도부터 측정한다.
노트북이 뜨거워 GPU 72 ℃·시스템 82 ℃ 를 넘으면 쉬었다 이어 가는 보호를 두고 판을 하나씩 순서대로 돌렸다(한 판 약 8~20 분).

### 7. 영상

office 맵 `sonar hard`·`soft` 를 한 판씩 위에서 내려다본 영상: [assets/office_sonar_hard_vs_soft.mp4](assets/office_sonar_hard_vs_soft.mp4) · [GIF](assets/office_sonar_hard_vs_soft.gif).
파란 원 = 순찰점, 주황 선 = 실제 이동 궤적, 빨간 원 = 로봇. 24 배속.

## 2 개 배치 재실행 (현재 실물 설계, 2026-11-14)

- 로봇 덮개 파일의 초음파를 2 개로 바꿨다: `ultrasonic_front_left` (0.18, +0.025, −0.06)·`ultrasonic_front_right` (0.18, −0.025, −0.06), 좌우 ±20°, 위 5°(`limo_fire.urdf.xacro` 와 같은 값, 잠정). 토픽은 `/ultrasonic/front_left`·`front_right`. 사양(0.02~4 m, 15°, 광선 5 개, 잡음 1 %)은 같다.
- 시험은 줄여서 office·warehouse × (`hold`/`sonar hard`/`sonar soft`) × 시드 0~2, 18 판. 두 판(warehouse 시드 2 의 `hard`·`soft`)은 시뮬 시간 한도에 걸려 일부 목표만 시도했다.
- 결과: `hold` 는 모두 전 목표 성공. `sonar` 는 `hold` 보다 나쁘다 — office `hard` 32/42·`soft` 40/42, warehouse `hard` 12/37·`soft` 8/38. 3 개 배치와 달리 warehouse 에서는 `soft` 도 `hold` 수준으로 돌아오지 않았다. 자세한 표는 [결과 6](benchmarks/README.md).
- 이 결과로 위 "한눈에" 의 `soft` 문장(`hold` 와 비슷)은 3 개 배치에만 맞고, 현재 설계에서는 office 에서만 조금 낫다.

## 어디까지 믿을 수 있나

- 믿을 수 있는 것: 같은 소프트웨어·같은 맵에서 반사 기준만 바꿨을 때 `sonar` 결과가 크게 달라진다는 것(office·warehouse 에서 `hard` 5 판 중 4 판이 나빴고 `soft` 5 판은 모두 `hold` 수준).
- 믿으면 안 되는 것:
  - 절대 수치. Webots 와 다르다(예: `hard` warehouse 추정 오차 최대 4.87 m vs 결과 3 의 1.70 m). 장면·구동이 다르기 때문이니 맵·조건 사이의 **순서만** 비교한다.
  - `soft` 가 실물 모양이라는 것. 잠정값이다.
  - maze·world 의 순서. 한 판씩이라 가르지 못한다.
  - 같은 시드의 완전한 재현. Nav2 처리 시점이 판마다 조금씩 달라 같은 시드도 숫자가 조금 다르다(영상 판 `hard` 11/14 ↔ 표의 시드 0 `hard` 8/14, 방향은 같음).
- 장면 한계: 둥근 덕트 호스 등은 계단 모양 면이 되어 입사각이 실제와 다르다. 높이 2.4 m 통일. 구동은 운동학이라 바퀴 미끄러짐의 비선형이 없고, IMU 는 충돌 반응이 없다. 시작 직후 로봇이 −1.7° 정도 틀어진 채 자리 잡는다(정답 위치는 매 판 실제 값을 쓴다).

## 재현할 때

시험 스크립트(Isaac 실행, 지도→USD 변환, 집계, 영상 녹화)는 **Python 이라 이 저장소에 없다**(저장소 규칙: C++/Rust). 위 표의 값과 절차로 다시 만들어야 한다. 판별 실행 기록은 [20261113_isaac_run_logs.txt](benchmarks/20261113_isaac_run_logs.txt)·[20261113_isaac_chain_status.txt](benchmarks/20261113_isaac_chain_status.txt) 에 있다.

## 다음에 할 것

1. 실물 HC-SR04 가 벽에 비스듬히 와도 메아리가 오는 각도를 재서 `soft` 모양을 실측으로 바꾼다.
2. 오도메트리 오차 값(1.03 하나만 시험)을 바꿔 가며 민감도를 본다.
3. Webots 와 같은 장면(`.wbt` 원래 모양)으로 Isaac 장면을 다시 만들어 장면 차이를 줄인다.
