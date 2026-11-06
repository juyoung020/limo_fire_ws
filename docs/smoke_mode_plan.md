# 연기 모드 1 주차 데모 계획 (2026-11-04 ~ 11-08)

라이다·카메라에 천을 씌워도(= 토픽이 끊겨도) 다른 센서로 저장 지도 위를 주행해 목표에 닿는다. 요구사항은
[spec.md "연기 모드"](spec.md#연기-모드), 백로그는 [PB-17·PB-18](backlog.md) (에픽 E6 "연기 모드 주행").
이번 주는 Webots 시뮬에서 재고 검증하는 것이 목표다. 실물은 부품이 준비되는 대로 같은 토픽·프레임으로 잇는다.

## 1. 정한 것

- **이번 데모의 주행은 Nav2.** 강화학습 연기 모드(`train/explore/mdp/smoke.h`)는 연기 속 과제·보상이 아직 없고 CUDA 빌드·학습 전이라 이번 주에 데모까지 못 간다.
  초음파 위치 추정은 Nav2·강화학습이 같이 쓰는 부품이라 먼저 만든다(강화학습 연결은 6 단계).
- **쓰는 센서**: 바퀴 오도메트리 + IMU(지금 달림, 연기 영향 없음) + 초음파 HC-SR04 × 3(키트 보유, 아직 안 달림).
  라이다·깊이·컬러 카메라는 쓰지 않는다(천을 씌운 상황 = 토픽 끊김). 레이더·열화상은 이번 범위가 아니다.
- **센서를 끄는 방법**: 시뮬·실물 공용 "센서 차단 노드"(spec.md 의 센서 열화 주입 노드). Webots 월드는 고치지 않고 토픽을 끊는다.
- **기준선부터 잰다**: 초음파 없이(바퀴 + IMU 만) 얼마나 가는지 먼저 재야 초음파가 얼마나 도움이 되는지 말할 수 있다.

## 2. 우선순위

| 순서 | 할 일 | 결과물 | 끝 기준 | 목표일 |
|---|---|---|---|---|
| **1** | **기준 측정: 라이다·카메라 끄고 바퀴 + IMU 만으로 주행** (Webots) | 센서 차단 노드(C++, `limo_fire_navigation`) — 처음 위치를 AMCL 로 잡은 뒤 라이다·카메라를 끊고, `map → odom` 을 붙잡아 위치는 바퀴 + IMU(EKF) 만 따라가게. 장애물 지도는 저장 지도만. 맵 4 개 × 목표 여러 개의 도착 오차·성공률 표 | 표가 나옴 (`docs/benchmarks/`) | 11-05 |
| **2** | **Webots 로봇에 초음파 3 개** | HC-SR04 모델: 2~400 cm, 감지 각 15° 원뿔(광선 여러 개 중 가장 가까운 값), 거리 잡음, 하나씩 차례로 측정(간섭 — spec.md) → `sensor_msgs/Range`. 장착 자리는 URDF xacro 한 곳에(모듈마다 파일 하나 — conventions.md) | 시뮬에서 벽까지 거리가 맞게 나옴 | 11-06 |
| **3** | **초음파로 내 위치 측정** | 초음파 3 개 → 점 몇 개짜리 스캔(원뿔 폭에 점 여럿, 측정 잡음 크게) → AMCL (spec.md 연기 모드 "위치 보정 1"). 아는 출발 위치에서 따라가기 | 1 단계와 같은 시험에서 도착 오차가 줄어듦. 통로 여유 20~30 cm 안 (spec.md) | 11-07 |
| 4 | 초음파로 장애물 피하기 | 초음파를 Nav2 장애물 지도(range sensor layer)에. 주행 중 상자 놓기 시험 (idea.md 시연 "무너진 장애물") | 상자를 돌아감 | 11-08 (여유 있으면) |
| 5 | 실물에 달기 | HC-SR04 × 3 + ESP32 펌웨어(`firmware/`) + `limo_fire_esp32_bridge` → 시뮬과 같은 토픽·프레임. Echo 5 V 는 전압 분배(hardware.md). 라이다·카메라에 천을 씌우고 같은 시험 | 실물로 목표 도착 | 부품·배선 준비되는 대로 |
| 6 | 강화학습 연기 모드에 연결 | 초음파 값을 정책 관측 예약 칸에(`obs_layout.h RESERVED`), 연기 속 과제·보상 정하기 | — | 데모 뒤 |

이번 주 데모 = 1~3 (시뮬). 4 는 여유가 있으면. 5 는 하드웨어 일정에 따른다.

## 진행 (2026-11-05)

- **1 단계 끝**: `smoke_mode`(Nav2 연기 모드) 다시 넣음, `nav_benchmark` · 정답 자세 플러그인 · 월드 supervisor. 맵 4 개 46 목표 (a)·(b) 모두 도착.
  다만 시뮬 오도메트리가 거의 완벽해 (b) 가 (a) 보다 정확했다 → [benchmarks "주행 벤치마크"](benchmarks/README.md#주행-벤치마크-연기-모드).
- **시뮬 오도메트리 오차(잠정)**: 실물 주행 오차를 재기 전까지 `LimoFire.proto wheelRadiusScale`(거리 오차, 기본 1.0 = 없음)과 월드 바퀴 마찰(미끄러짐)로 넣는다.
  방향 오차는 Webots IMU 에 쌓이는 치우침이 없어 실측 전에는 넣지 않는다. 첫 판: 거리 +3 % 미로 (b) 추정 오차 최대 0.25 m.
- 실행은 Humble 인 로봇 Jetson 에서 한다(이 저장소 기준 환경). Ubuntu 24.04 + Jazzy 데스크톱은 Nav2 플러그인 이름·바퀴 제어기 한도·속도 메시지 형식이 달라 그대로는 안 돈다.
- 다음: 마찰 단계별 실험 → 2 단계(Webots 초음파 3 개).

## 진행 (2026-11-06) — 2~5 단계 코드 다 만듦, 첫 시뮬 측정: 3·4 단계 효과 없음

코드·설정과 시뮬 없이 되는 시험을 한 뒤, 로봇 Jetson 에서 팀원 시뮬과 같이(포트 1235) 첫 측정을 돌렸다.
결과: [benchmarks "결과 2"](benchmarks/README.md) —
**초음파 위치 추정(`sonar`)이 바퀴 + IMU(`hold`)보다 네 맵 모두 나빴고(office·warehouse 에서 위치를 잃음), 상자는 초음파를 켜도 못 피했다.**

| 단계 | 만든 것 | 확인한 것 | 남은 것 |
|---|---|---|---|
| 2 초음파 3 개 (시뮬) | `LimoFire.proto` `ultrasonic`(기본 FALSE) — Webots sonar 3 개(15°, 2~400 cm, 잡음 1 % 잠정), `/ultrasonic/front·left·right` (`sensor_msgs/Range`, 10 Hz), `urdf/ultrasonic.xacro` 프레임 (잠정 자리: 앞 0.16 m, 왼·오른 옆면 0.11 m, 바닥 위 약 0.115 m) | URDF 링크 3 개, 시뮬에서 토픽이 나와 3 단계가 씀 | 값 확인(확인 스크립트 오류로 숫자 못 얻음) |
| 3 초음파 위치 추정 | `sonar_scan` 노드(초음파 → base_link 60 칸 스캔, 원뿔에 점 3 개), `smoke_mode` `smoke_localization: sonar`(연기 속 AMCL 에 초음파), `nav_benchmark` `smoke_localization` | 가짜 Range 로: 앞 1 m → −6°·0°·+6° 칸 1.16 m, 왼 0.5 m → 90° 근처 0.61 m. 맵 4 개 측정: **(c) 가 (b) 보다 나쁨** | 아래 "고쳐서 다시 잴 것" 1 |
| 4 초음파 장애물 | 전역·지역 장애물 지도에 `sonar` 관측원(찍기만), `nav_benchmark` `obstacle_box`(정답으로 상자까지 가장 가까운 거리, 0.12 m 미만 = 부딪힘, 잠정) | 상자 2 판: **둘 다 닿음(초음파로 못 피함)**, 부딪힘 기준이 틀렸음 | 아래 2·3·4 |
| 5 실물 초음파 | `firmware/limo_fire_esp32`(HC-SR04 × 3 차례로, 체크섬 줄), `limo_fire_esp32_bridge`(시리얼 → 시뮬과 같은 토픽·프레임), `robot.launch.xml esp32:=true` | 펌웨어 빌드(경고 0), 다리 단위 테스트, 가짜 ESP32(pty) 끝까지 | 실물 배선·장착·천 씌우고 주행 |

공통: `sim.launch.xml port:=` (한 컴퓨터에 시뮬 둘), 데스크톱 빌드 경고 0.

### 다음에 돌릴 것 (로봇 Jetson, Humble — 아무도 시뮬을 안 쓸 때)

로봇 `~/smoke_ws2` 가 이 작업판(main + 위 코드)이다. 남의 시뮬과 안 겹치게 **Webots 포트 1235 · `ROS_DOMAIN_ID=57`**.

1. 빌드: `cd ~/smoke_ws2 && colcon build --symlink-install --packages-select limo_fire_description limo_fire_navigation limo_fire_bringup limo_fire_sim --packages-ignore <실물 드라이버들>`
2. 실험 월드: 원본 월드에서 `LimoFire` 에 `wheelRadiusScale 1.03`(거리 +3 %) · `ultrasonic TRUE` 를 넣은 사본을 같은 폴더에(상대 EXTERNPROTO 그대로). 4 단계는 고정 상자
   `Solid { translation -2.3 -2.45 0.15  boundingObject Box { size 0.4 0.5 0.3 } … }` 를 미로 월드 끝에 더한다(물리 없음, 출발 직후 통로, 위로 0.65 m 틈).
   로봇의 `~/smoke_run2/mkworld.sh <월드> 1.03 1.0 TRUE [x,y,가로,세로]` 가 이것을 만든다.
3. 시뮬: `ros2 launch limo_fire_bringup sim.launch.xml world:=<실험 월드> port:=1235 rviz:=false map:=<원본 월드 지도>.yaml`
4. 측정: `ros2 run limo_fire_navigation nav_benchmark --ros-args -p use_sim_time:=true -p world_file:=sim/webots/worlds/<원본 월드>.yaml -p smoke:=true -p smoke_localization:=<hold|sonar> -p label:=<이름> -p report_dir:=docs/benchmarks`
   (4 단계는 `-p max_goals:=1 -p "obstacle_box:=[-2.3,-2.45,0.4,0.5]"`)
   로봇의 `~/smoke_run2/run_one.sh` 가 3·4 를 한 번에, `when_free.sh` 가 아래 목록 전부를 (빈 때를 기다렸다가) 돈다.

| 순서 | 무엇 | 월드 | 조건 | 기대 |
|---|---|---|---|---|
| 1 | 2 단계 확인 | maze, 초음파 켬 | — | 초음파 ≈ 같은 방향 라이다 빔 (앞은 장착 차이 0.057 m) |
| 2~9 | 3 단계 | 맵 4 개, 거리 +3 % · 초음파 켬 | (b) `hold` · (c) `sonar` | (c) 추정 오차 < (b) |
| 10 | 4 단계 | maze + 상자, 초음파 켬 | `sonar`, 첫 목표만 | 상자를 돌아 도착, 안 부딪힘 |
| 11 | 4 단계 기준 | maze + 상자, 초음파 끔 | `hold`, 첫 목표만 | 상자를 못 봄 (부딪힘 예상) |

걸리는 시간: 맵마다 시뮬 켜기 약 75 s + 순찰 5~10 분 → 전부 약 1 시간 40 분.

### 다음에 고쳐서 다시 잴 것

원인 후보는 확인 전이다. 하나씩 바꾸고 위 표를 다시 돈다(로봇 `~/smoke_run2/run_now.sh` = 기다림 없이 전부, 약 1 시간 40 분).

1. **초음파 AMCL 설정** — 지금은 라이다용 AMCL 값(sigma_hit 0.2, z_hit 0.5·z_rand 0.5, likelihood_max_dist 2.0) 그대로 점 9 개짜리 초음파 스캔을 믿어
   엉뚱한 자리로 끌려간다(office·warehouse). 초음파일 때 측정 잡음을 크게(sigma_hit↑·z_rand↑), 갱신을 덜 믿게(update_min_d↑) 한 AMCL 판을 쓰거나,
   AMCL 대신 바퀴 + IMU 를 기본으로 두고 초음파는 벽까지 거리가 지도와 많이 다를 때만 작게 고치는 방식을 비교한다.
2. **초음파 장애물이 안 지워짐** — `nav2_params.yaml` `sonar` 관측원이 찍기만 해서 잡음이 장애물 지도에 쌓였을 수 있다(office (b) 가 위치는 맞는데 끝에서 경로를 못 냄).
   Nav2 의 초음파용 층(`nav2_costmap_2d::RangeSensorLayer`, 확률로 찍고 지움)으로 바꾸거나 지우기를 켠다.
3. **장애물을 늦게 봄** — 앞 초음파 하나(15°, 10 Hz)로 0.13 m/s 에서도 못 피했다. 연기 모드에서 앞 초음파 거리가 짧으면 속도를 줄이거나 멈추는 규칙,
   초음파 개수·방향(앞 대각선 2 개 등)을 비교한다.
4. **측정 도구 부딪힘 기준** — 가운데에서 거리(0.12 m)가 아니라 몸통 사각형(0.32 × 0.22 m, 자세 방향)과 상자가 겹치는지로 센다. 먼저 고친다.
5. **2 단계 확인 스크립트** — `ros2 topic echo --full-length` 로 배열을 줄이지 않게 하거나 C++ 확인 도구로.

### 남은 것

- 위 "고쳐서 다시 잴 것" 1~5 → 다시 측정 → 결과를 `docs/benchmarks/` 와 이 문서에.
- 실물: 초음파 3 개 장착 자리 정하기(지금 값 잠정) → 배선(Echo 5 V 전압 분배) → `esp32:=true` 로 토픽 확인 → 라이다·카메라에 천을 씌우고 같은 목표 주행.
- 실물 주행 오차(1 m 직진·360° 회전) 측정 → 시뮬 `wheelRadiusScale`·마찰을 실측값으로.
- 부딪힘 감지 문턱(잠정 4 m/s²) 다시 정하기 — 1 단계에서 오탐 15 번.
- 로봇 `~/smoke_ws` 의 `nav_benchmark`·`smoke_mode` 에 누가 검사를 더했는지 확인(11-05 05:44 접속, 06:02~06:05 수정) — 이 저장소에는 안 들어가 있다.

## 3. 시험 (1·3 단계 공통)

- 맵: Webots 4 개 (`limo_fire_maze`·`office`·`warehouse`·`world`, 저장 지도 `limo_fire_navigation/maps/sim/`).
- 출발: 맵 시작 자리(지도 원점), 라이다로 AMCL 위치를 잡은 뒤 센서를 끊는다.
- 목표: 맵마다 가까운 곳·먼 곳(통로 꺾임 포함) 몇 개. 같은 목표 목록을 1·3 단계에 똑같이 쓴다.
- 잴 것: 목표 도착 여부, 도착 때 참 위치(Webots)와 목표 사이 오차, 추정 위치와 참 위치의 차이(이동 거리별), 충돌.
- 비교: (a) 라이다 켬(평소) · (b) 바퀴 + IMU 만 · (c) 바퀴 + IMU + 초음파. 표는 `docs/benchmarks/` 에.

## 4. 알고 갈 한계

- 초음파 3 개로는 처음 위치를 스스로 찾지 못한다(점이 너무 적다). 출발 위치를 아는 상태에서 오차를 고쳐 가는 용도.
- 장착 방향이 정확도를 정한다. 앞·왼·오른쪽이면 복도에서 좌우 위치는 잘 잡지만, 앞뒤 위치는 앞 센서 하나에 기댄다. 장착 방향은 2 단계 시작 전에 정한다(잠정: 앞·왼·오른).
- 초음파는 벽에 비스듬히 쏘면 거울처럼 튕겨 값이 틀릴 수 있다 → 시뮬 잡음 모델에 넣고 실물로 맞춘다.
- HC-SR04 정확도 3 mm 는 주로 판매처 숫자라 실측으로 확인한다. 거리·각도 값은 ElecFreaks 데이터시트 기준.
- 바퀴 오도메트리 주행 오차는 아직 안 쟀다(hardware.md "확인할 것"). 1 단계 결과가 시뮬 구동 모델 기준이라는 것을 표에 적는다.

## 5. 출처

- HC-SR04 데이터시트 (ElecFreaks): 거리 2~400 cm, 감지 각 15°, 트리거 10 µs 이상, 거리 = 펄스 폭 × 음속 / 2 — https://www.radiolocman.com/datasheet/pdf.html?di=168043 ·
  https://www-users.cs.york.ac.uk/~mjf/roach/Datasheets/HCSR04_Manual.pdf
- 연기 속 센서 비교: Starr & Lattimer, "Evaluation of Navigation Sensors in Fire Smoke Environments", Fire Technology 50, 2014 — https://link.springer.com/article/10.1007/s10694-013-0356-3
