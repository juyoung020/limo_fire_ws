# 실물 이식 체크리스트 (연기 모드)

Isaac Sim·Webots 결과를 실물 LIMO 에 반영하기 전에 준비할 것. 시뮬 결과는 [isaac_sim_cross_check.md](isaac_sim_cross_check.md), 연기 모드 계획은 [smoke_mode_plan.md](smoke_mode_plan.md), 부품은 [hardware.md](hardware.md)·[wiring.md](wiring.md).

## 원칙

- 교차 검증이 말해 주는 것은 "초음파 반사 기준에 따라 결과가 크게 바뀐다" 까지다. `soft` 모양(25~55°)과 오도메트리 ×1.03 은 **잠정값**이다. 코드에 박지 말고 **파라미터**로 두고, 실측 뒤 yaml 에서 바꾼다.
- 절대 수치는 시뮬과 다르다. 맵·조건 사이의 **순서**만 비교한다.
- 연기 속 위치 방식은 `hold`(바퀴 + IMU) · `wall`(초음파를 저장 지도 벽에 맞춰 보정, `config/smoke_mode.yaml` 의 `wall.*`) 둘이고 기본은 `hold` 다. 옛 `sonar` 는 지워졌다. 실물에서도 `ros2 param set /smoke_mode smoke_localization wall` 로 바꾼다(연기 모드가 꺼져 있을 때만).
- `wall.*` 의 값(`sigma`·`gate`·`q_xy` 등)은 전부 시뮬 HC-SR04 모델 기준 **잠정**이다. 실물 측정은 아래 §3 도구(`tools/real_calib`)로 후보를 뽑아 yaml 주석에 근거(날짜·녹화)와 함께 옮긴다.

## 0. 지금 상태 (2026-11-17 확인)

| 항목 | 상태 |
|---|---|
| 초음파 HC-SR04 × 2 | 키트 보유, **로봇 미장착** |
| 펌웨어 `firmware/limo_fire_esp32` | `pio run` 빌드 통과(S3·C3 둘 다, 2026-11-17, C3 핀은 잠정), **실물 ESP32·HC-SR04 에서 안 돌려 봄** |
| ESP32 브리지 `limo_fire_esp32_bridge` | 줄 해석·가짜 ESP32 장면 시험 통과(2026-11-17, 단위 13 개) + 가짜 ESP32(pty)로 끝까지 통과(`scripts/fake_bridge_check.sh`: 값·`+inf`·깨진 줄·재연결). **실물 ESP32 연결은 아직** |
| 핀 배치(`include/config.h`) | 잠정 — front_left Trig 6/Echo 7, front_right Trig 15/Echo 16 (ESP32-S3 기준) |
| Echo 5 V → 3.3 V | 전압 분배(예 1 kΩ/2 kΩ) 필요 |
| 마운트 자리·각도 | 잠정(앞 좌우 ±10°, 위 5° — 3D 출력 초음파 막대 CAD 값) |

## 1. 하드웨어 준비

- [ ] HC-SR04 2 개를 앞 범퍼 아래 좌우에 장착. 위치·각도를 줄자·각도기로 실측해 `limo_fire.urdf.xacro` 와 비교
- [ ] Echo 라인 전압 분배 회로 만들기, GND 공통
- [ ] ESP32 보드 핀: S3 는 `config.h` 그대로, C3 SuperMini 는 `pio run -e esp32c3` 의 잠정 핀(Trig 3·5 / Echo 4·6)으로 배선하고 실물에서 확인 — 배선하면 `docs/wiring.md` 에도 반영
- [ ] USB 시리얼 `/dev/esp32` udev 규칙
- [ ] 배터리·전원: 초음파 5 V 공급원, 모터 노이즈가 센서에 끼는지 확인

## 2. 펌웨어·브리지 단독 확인 (로봇 안 움직이고)

실물 ESP32 가 오기 전에 가짜 ESP32 로 다리부터 확인해 둘 수 있다(2026-11-17 통과):

```bash
source env.sh && colcon build --packages-select limo_fire_esp32_bridge && source install/setup.bash
lib=$(ros2 pkg prefix limo_fire_esp32_bridge)/lib/limo_fire_esp32_bridge
$lib/fake_bridge_check.sh        # sweep·noecho·corrupt 장면 + 다리를 먼저 켜도 이어짐·끊김 복구
$lib/fake_esp32 --link /tmp/fake_esp32 --scenario stall   # 손으로: 멈춤(살아 있음 경고) 장면
```

- [ ] `pio run -t upload` 후 `pio device monitor` 에서 `$US`, `$HB` 줄과 체크섬이 나오는지
- [ ] 센서를 번갈아 쏘는지(센서마다 35 ms, 한 바퀴 70 ms). 동시에 쏘면 서로 메아리를 받는다
- [ ] 줄자로 0.3 / 1 / 2 / 3 m 거리 확인. 오차 표 기록
- [ ] 브리지가 `/ultrasonic/front_left`, `/ultrasonic/front_right` 를 `sensor_msgs/Range` 로 내는지 `ros2 topic echo`·`hz`
- [ ] 메아리 없음 규칙: 펌웨어 0 → ROS `+inf` 로 바뀌는지(시뮬과 같게). 0.02 m 미만 `-inf`, 4 m 초과 `+inf`
- [ ] `frame_id` 가 URDF 의 초음파 프레임과 같은지 (`tf2_echo`)
- [ ] 시리얼을 뽑았다 꽂았을 때 브리지가 복구되는지 (가짜로는 통과, 실물 USB 로도)

## 3. 실물 측정 (파라미터 확정용) — 가장 중요

| 측정 | 방법 | 쓰이는 곳 |
|---|---|---|
| **반사 각도** | 벽에 입사각 0, 15, 30, 45, 60° 로 대고 각도마다 100 회. 메아리 비율과 거리 오차 | `wall`/`soft` 반사 모양 (문서 "다음에 할 것" 1번) |
| 재질별 | 합판·벽지·철판·유리·천 | 시연장 벽 재질 대표 |
| 감지 각(원뿔) | 작은 물체를 옆으로 옮기며 메아리가 끊기는 각 | 시뮬 15°·광선 5 개 가정 |
| 거리 잡음 | 고정 벽, 0.5 / 1 / 2 / 3 m 에서 표준편차 | 시뮬 1 % 가정 |
| 오도메트리 슬립 | 직선 3~5 m, 360° 회전을 각 10 회. 실제 거리 대비 `/odom` 비율 | 시뮬 ×1.03 가정 (문서 "다음에 할 것" 2번) |
| IMU yaw 드리프트 | 정지 5 분, 주행 중 | 헤딩 신뢰도 |
| 모터 켜졌을 때 초음파 | 바퀴 구동 중 잡음·오탐 | 시뮬엔 없음 |

측정값을 csv 로 저장하고(`train/explore/sim/analysis/bags.md` 에 조건 한 줄) `tools/real_calib` 로 후보를 뽑는다. 맞추기(fit)와 확인(validate) 측정은 나눈다(`calibration.yaml`).

```bash
cmake -S tools/real_calib -B /tmp/real_calib_build && cmake --build /tmp/real_calib_build && ctest --test-dir /tmp/real_calib_build
/tmp/real_calib_build/real_calib echo  echo.csv    # angle_deg,true_m,range_m[,material] — 각도별 메아리 비율·편향·σ, 정면 거리 잡음
/tmp/real_calib_build/real_calib slip  slip.csv    # true_m,odom_m — 줄자 이동 / odom 비율 (시뮬 1.03 과 비교)
/tmp/real_calib_build/real_calib drift imu.csv     # t_s,yaw_deg — IMU yaw 드리프트 °/min
```

초음파·wall 보정·AMCL 자세는 `record.launch.xml` 이 같이 녹화한다. 나온 후보는 값만 옮기지 말고 날짜·녹화 이름을 config 주석에 남긴다.

## 4. 소프트웨어 준비

- [ ] Jetson(Ubuntu 22.04, Humble)에서 `colcon build` 가 되는지 (시뮬은 Docker 였다)
- [ ] `use_sim_time` 시뮬 true / 실물 false 를 launch 인자로 분리
- [ ] Nav2 설정 `nav2_sim.yaml` 과 실물용 설정 분리. 실물 속도·가속도 한도는 낮게 시작 (`progress.md` 의 "Nav2 설정은 실물 확인 남음")
- [ ] 센서 차단 노드: 실물에서도 라이다·카메라 토픽을 끊는 방식이 시뮬과 같은지
- [ ] 시작 순서: 라이다로 AMCL 초기 자세 → 라이다·카메라 차단 → `smoke_mode` 시작 (시뮬 절차와 같게)
- [ ] 실물 `/odom`·`/imu` 의미 확인: 거리는 엔코더 평균, 헤딩은 IMU. 시뮬 `limo_sensors.py` 와 같은지
- [ ] 비상정지: 리모컨 모드 스위치(SWB)가 하드웨어 우선(리모컨이 오토 모드 위여야 `/cmd_vel` 이 섀시에 가고, 내리면 ROS 명령이 끊김). 연기 모드 속도 한도(`speed_limit` 0.70 m/s, 시뮬 기준 — 실물은 낮게 시작)와 초음파 0.6 m 안 감속(`slow`)은 이미 있음. **별도 소프트웨어 비상정지 노드는 아직 없음**

## 5. 지도·평가 준비

- [ ] 실물 지도: Cartographer 로 시연장 미로 지도 저장 (PB-05). 시뮬 맵은 pgm 박스 2.4 m, 실물은 높이·둥근 면이 다르다
- [x] 정답 위치 대용: `/sim/ground_truth` 가 없으니 `nav_benchmark` 에 실물 모드(`real:=true`)를 만들었다. 시작 자리와 목표마다 멈춘 뒤 줄자로 잰 지도 좌표를 `/real/measured_pose` 로 알려 주면 시뮬과 같은 csv·md 표가 나온다(잰 값이 오기를 `real_wait` 초 기다림).
- [x] 실물 안전: 실물 모드는 시작 자리에서 목표들까지 직선 구간이 `max_leg`(기본 0.6 m, CLAUDE.md) 를 넘으면 시작 전에 막는다. 올릴 때는 주변 비움·사람 곁을 확인한 뒤 `-p max_leg:=<m>`.
- [ ] 순찰점 목록을 실물 지도 좌표로 다시 정의

## 6. 실물 시험 순서

1. `hold`(바퀴 + IMU)만 실물에서 먼저 — 기준선. 목표마다 3 회 이상
2. 초음파 단독 읽기(로봇 정지, 움직이는 사람·벽 앞) 확인
3. `wall`(초음파 켠 위치 보정)을 `hold` 와 같은 목표로 비교. 3 회 이상
4. 반사 각도 실측으로 바꾼 파라미터로 다시 시뮬(Isaac) 돌려 실물 결과와 순서가 맞는지 확인
5. 순서가 다르면 시뮬 모델(반사·잡음·슬립)을 고치고 반복

판정은 시뮬과 같다: `wall` 이 `hold` 보다 나쁘면 초음파 보정을 쓰지 않고 `hold` 로 간다.

## 7. Isaac 슬립 민감도 (2026-11-17, office·시드 0 하나)

바퀴 반지름 배율 k(오도메트리 슬립)를 바꿔 `hold` 와 `wall`(hard 반사)을 같은 목표 14 개로 돌렸다. 원본 표는 [benchmarks/isaac_slip_k/](benchmarks/isaac_slip_k/).

| k | hold 성공 / 평균 도착 오차 | wall 성공 / 평균 도착 오차 |
|---|---|---|
| 0.97 | 14/14 · 0.34 m | 14/14 · 0.32 m |
| 1.00 | 14/14 · 0.29 m | 14/14 · 0.26 m |
| 1.03 | 14/14 · 0.26 m | 14/14 · 0.24 m |
| 1.06 | 11/14 · 0.83 m | 14/14 · 0.23 m |
| 1.10 | 7/14 · 2.47 m | 11/14 · 3.76 m |

`wall`+soft 반사는 k=1.03 에서 14/14 · 0.24 m, k=1.10 에서 14/14 · 0.21 m(최대 추정 오차 0.15 m).

- 읽을 수 있는 것: k 가 ±3 % 안이면 둘 다 거의 같고, 6 % 부근에서 `hold` 만 무너진다. 10 % 에서는 hard 모양 `wall` 도 일부 실패(11/14)하는데 soft 에서는 14/14 였다 — 반사 모양 가정이 결과를 바꾼다는 교차 검증 결론과 같은 방향.
- 한계: office 한 곳·시드 하나·판마다 1 회. 1.10 처럼 크게 갈리는 구간은 판 사이 흔들림일 수 있어 순서만 믿는다. 실물에서 슬립(`real_calib slip`)을 먼저 재서 어느 줄에 해당하는지 본다.

## 8. 알려진 위험

- 초음파 두 개뿐이라 옆·뒤는 못 본다. 코너·비스듬한 벽에서 메아리가 안 올 수 있다(옛 `sonar` 방식이 시뮬 warehouse 에서 크게 나빴던 곳 — `wall` 은 같은 곳에서 8 판 중 7 판에서 `hold` 보다 오차가 작았다, benchmarks 결과 7).
- 모터·USB 전원 노이즈, 5 V Echo 레벨, 온도에 따른 음속(20 °C 343 m/s 가정).
- 시뮬 IMU 에 충돌 충격이 없어서, 실물에서 벽에 부딪히면 헤딩이 흔들릴 수 있다.
- 처음 몇 번은 사람이 옆에서 비상정지 가능한 상태로 시험한다.
