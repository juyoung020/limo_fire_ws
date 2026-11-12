# limo_fire_rl — 탐사 노드 (RL 정책 · 프론티어)

두 노드가 있다. 둘 다 Nav2 `NavigateToPose` 로 목표를 보내고, `~/enable`(std_srvs/SetBool)로 켜고 끄며, 운용 화면 RViz 의 **탐사 패널**
("Auto explore ON/OFF" 버튼, [limo_fire_rviz](../limo_fire_rviz/README.md))이 이 서비스를 부른다. 실물에서 RViz 를 켜는 법은
[limo_fire_bringup README "실물에서 운용 화면(RViz) 켜기"](../limo_fire_bringup/README.md#실물에서-운용-화면rviz-켜기).

| 노드 | 무엇 | 켜기 |
|---|---|---|
| `explore_frontier` | 프론티어 탐사 (아래 "프론티어 탐사") — 탐사 기준선과 같은 찾기·고르기 | 멈춘 채 뜸 → 버튼 |
| `explore_rl` | 학습한 계층형 정책 (아래부터) | 기본 바로 켜짐 (`enabled:=false` 면 버튼) |

## 프론티어 탐사 (`explore_frontier`)

Cartographer `/map` 에서 프론티어(아는 빈 칸과 모르는 칸의 경계) 덩어리를 찾아 하나를 Nav2 목표로 보내고, 지도가 닫히면 멈춘다. 알고리즘은 새로 쓰지 않았다:

| 차례 | 코드 (탐사 기준선과 같은 것) |
|---|---|
| 찾기·닫힘 | `train/explore/mdp/frontier.h` `frontier_search` · `closed` — 설 수 있는 칸(로봇 반경 = `nav2_params.yaml` `robot_radius`)으로 번진 곳 안의 덩어리. 막힘 = `/map` 값 > 65 (p > 0.65) |
| 고르기 | `train/explore/baselines/frontier/include/explore_lite.h` — explore_lite 비용(거리·크기) 오름차순, 블랙리스트 아닌 첫 덩어리의 중심. 기준선 `Explorer::make_plan` 도 이 함수를 쓴다 |
| 값 | 기준선 설정 `baselines/frontier/config/frontier.yaml` 그대로 (3 초마다 다시 고름, 같은 목표 30 초 진척 없음 → 블랙리스트, 0.75 m 미만 덩어리는 다른 게 없을 때만) |
| 가기 | Nav2 (기준선의 A*·RPP 흉내 대신 진짜 Nav2). 도착·포기 때 바로 다시 고름, 포기(ABORTED)한 목표는 블랙리스트 |
| 시작 | 프론티어가 안 보이면 Nav2 `spin` (30 초 안에 안 보이면 끝) |
| 끝 | 닫힘 · 남은 덩어리가 다 블랙리스트 → 목표 취소, `~/done` true, `~/state` done |

- **끄기**: `~/enable false` → 보낸 Nav2 목표·회전을 취소한다(로봇이 선다). **켜기**: 블랙리스트를 비우고 지금 지도에서 다시 고른다(끝난 뒤 다시 켜도 처음부터).
- 토픽: `~/state`(idle · running · done, transient local) · `~/done`(Bool) · `~/status`(한 줄) · `~/frontiers`(MarkerArray: 파랑 후보, 빨강 목표, 회색 블랙리스트 — 운용 화면 "Frontiers").
- 설정: `config/explore_frontier.yaml` (ROS 쪽만). 저장 지도·AMCL 위에서도 돌지만 지도가 이미 닫혀 있으면 바로 끝난다 — Cartographer(지도 만들기)와 같이 쓴다.

```bash
# 시뮬: slam:=true 면 노드가 멈춘 채 같이 뜬다 (explore:=false 로 끔). RViz 탐사 패널에서 "Auto explore OFF (누르면 시작)"
ros2 launch limo_fire_bringup sim.launch.xml world:=limo_fire_maze slam:=true
# 실물: robot.launch.xml 을 켠 뒤 (Cartographer + Nav2 + RViz 까지 한 번에)
ros2 launch limo_fire_rl explore_frontier.launch.xml stack:=true rviz:=true
# 버튼 대신 명령으로 (버튼이 부르는 것과 같음)
ros2 service call /explore_frontier/enable std_srvs/srv/SetBool "{data: true}"    # false 면 멈춤
ros2 topic echo /explore_frontier/state
```

### 확인 (2026-11-11, 이 PC Webots R2025a 화면 없이 · 덕트호스 미로 `slam:=true`)

| 확인 | 결과 |
|---|---|
| 끄기 (버튼과 같은 `~/enable false`, 0.26 m/s 로 가는 중) | 0.08 s 뒤 `/cmd_vel` 0, Nav2 목표 취소(CANCELED), 끈 뒤 미끄러짐 2 cm · 그 뒤 10 s 동안 0 |
| 다시 켜기 | 블랙리스트를 비우고 이어서 탐사 |
| 끝까지 | 켬 60 s → 끔 10 s → 켬 65.3 s 에 닫힘으로 끝(`~/state` done). 탐사한 시간 125 s(시뮬 시간), 지도 49.5 m², 보낸 목표 18 · 포기 0 |
| 운용 화면 패널 | `rviz2 -d limo_fire.rviz` 가 패널을 불러옴. 상태 idle → running → done, 버튼 색·글자가 바뀜 |

- 프론티어 중심이 지도가 자랄 때마다 조금씩 옮겨 가서 목표를 자주 다시 보낸다(3 초마다, explore_lite 와 같음) — Nav2 로그에 앞 목표 취소가 많이 찍힌다.
- 처음엔 전역 costmap 이 Cartographer 벽을 안 넣어(막힘 문턱 100, 아래 문서) 라이다 3 m 밖 덕트를 뚫는 경로로 7 분 넘게 맴돌았다 —
  `nav2_params.yaml` `lethal_cost_threshold: 65` 로 고친 뒤 위 결과. [docs/simulation.md "Nav2 설정"](../../../docs/simulation.md).
- 시뮬은 화면 없는 X 의 소프트웨어 렌더링이라 실시간의 약 0.55 배. 실물·젯슨에서는 아직 안 돌림.

## RL 정책 (`explore_rl`)

학습한 계층형 탐사 정책(`train --hier`)을 실물 LIMO 에서 돌린다. 1 s 마다 프론티어 후보 8 곳 중 하나를 골라 Nav2 `NavigateToPose` 목표로 보내고,
지도가 닫히면(시뮬 성공 판정과 같음) 멈추고 `~/done` 에 `true` 를 낸다.
주행을 Nav2 대신 학습한 아래층 정책이 맡는 구조는 만드는 중이다([train/explore/ROADMAP.md](../../../train/explore/ROADMAP.md)).

**관측은 학습 시뮬과 같은 코드다.** `/scan` 한 장을 시뮬 걸음 하나(0.1 s)로 보고, 시뮬이 쓰는 함수(`train/explore/sim/include/env.h`·`obs_ref.h`, `mdp/`)에
실물 값을 그대로 넣는다. 관측 지도도 Cartographer `/map` 을 바꿔 쓰지 않고, 시뮬과 같은 로그 오즈 갱신으로 `/scan` + Cartographer 자세(TF `map → base_link`)에서
쌓는다. 그래서 같은 센서 값이면 관측이 시뮬과 비트까지 같다(아래 "확인").

## 실행 (로봇)

빌드 (젯슨에서 한 번, 2~3 분):

```bash
cd ~/limo_fire_ws && git pull --rebase --autostash
source env.sh
colcon build --packages-select limo_fire_rl          # Release·-ffp-contract=off 는 CMakeLists 가 넣음. CUDA 필요 없음
source install/local_setup.bash
```

체크포인트는 학습 PC 의 `train/explore/runs/<실험>/ckpt/model_latest.bin` 등 `.bin` 하나를 로봇으로 복사해 경로를 준다(정책 부분만 읽음, 형식 3).

터미널 셋 (모두 `source ~/limo_fire_ws/env.sh` 뒤):

```bash
# 1. 센서·섀시 (대시보드 limo_base 가 떠 있으면 켜지 않는다 — 포트 충돌)
ros2 launch limo_fire_bringup robot.launch.xml

# 2. 먼저 목표를 안 보내고(dry_run) 후보·목표가 맞는지 RViz 로 본다 — Cartographer + Nav2 도 같이 (stack:=true)
ros2 launch limo_fire_rl explore_rl.launch.xml checkpoint:=$HOME/ckpt/h4_2511.bin stack:=true dry_run:=true rviz:=true
#    RViz: Map /map, MarkerArray /explore_rl/candidates (파랑 = 후보, 빨강 = 고른 후보, 초록 원판 = 목표)
#    ros2 topic echo /explore_rl/status   → "걸음 … 후보 … 고름 … 닫힘 … 진척없음 …"

# 3. 괜찮으면 끄고 실제로 (리모컨 오토 모드 SWB 위, 충전기 뺌, 주변 비움, 사람 곁에)
ros2 launch limo_fire_rl explore_rl.launch.xml checkpoint:=$HOME/ckpt/h4_2511.bin stack:=true
ros2 topic echo /explore_rl/done        # true 면 탐사 끝
ros2 service call /explore_rl/enable std_srvs/srv/SetBool "{data: false}"   # 언제든 멈춤 (목표 취소)
```

- Cartographer·Nav2 를 따로 띄웠으면 `stack:=false`(기본). Nav2 는 저장 지도 없이 `navigation.launch.xml localization:=false`.
- 정책 없이 같은 후보 계산 위 "가장 가까운 프론티어" 로 돌리려면 `policy:=nearest` (체크포인트 필요 없음). explore_lite 식 프론티어 탐사는 위 `explore_frontier`.
- 시험 기록: `debug_dir:=/폴더` 면 결정마다 `decisions.csv`(자세·후보 8 곳·로짓·고른 칸·걸음·정방향 시간), 관측 `obs.bin`, 로봇 지도 `lmap.bin` 을 쓴다.

## 한 결정 (1 s)

| 차례 | 실물 노드 | 시뮬 (`--hier`) |
|---|---|---|
| 걸음 (0.1 s) | `/scan` 한 장마다: TF `map → base_link`(장 시각) · `/odom` · `/imu` · `/cmd_vel` → `RobotEnv::step` | `env_ref.cpp step_env` — 같은 함수 차례 |
| 지도 | `scan_begin_map` → 빔마다 `scan_beam_map`(세기 = `/scan` intensities) → `scan_apply` | 같은 함수 (세기만 시뮬 모델) |
| 관측 | `write_obs` (깊이 빼고) + 깊이 영상 → `depth_from_image` | `write_obs` + `obs_depth` |
| 정책 | CPU 정방향 `network/policy_cpu.h` (결정 = 가린 로짓 최대) | `act_inference` (GPU) |
| 목표 | 같은 덩어리면 그대로(`RobotEnv::choose` — 시뮬 `plan_cpu` "같은 후보"), 새 칸이 보낸 목표에서 0.25 m 넘게 옮기면 Nav2 에 다시 | 경로 계획·RPP 추종 |
| 후보 없음 | Nav2 `spin` 1.57 rad | 제자리 회전 |
| 끝 | 지도 닫힘 + 지도 진척 시계 ≥ 10 s (1 s 마다) → 목표 취소, `~/done` | 같은 판정 (`finish_step`) |
| 진척 없음 60 s | 알리고 계속 (`stop_on_no_progress:=true` 면 멈춤) | 판 끝 (실패) |
| 입력 끊김 | 라이다·TF 1 s 없으면 목표 취소 | — |
| Nav2 실패 | 그 목표 둘레 0.5 m 후보를 10 s 피함 (다음 로짓) | 복구(물러나기·목표 포기, `hier_path.h`) |

파라미터는 `config/explore_rl.yaml` (토픽 이름은 [sensor_topics.md](../../../docs/sensor_topics.md)). 관측·결정 주기·닫힘 판정 값은 학습 설정
`train/explore/sim/config/env.yaml`(+ 그 안의 `nav2_params.yaml`·`sensor_model.yaml`)을 그대로 읽는다 — 노드 파라미터 `env_yaml` 기본값이 이 저장소의 그 파일.

## 확인 (2026-11-11, 이 PC: Ryzen 5 5600X · ROS 2 Humble x86_64 — 젯슨에서는 아직 안 돌려 봄)

| 확인 | 방법 | 결과 |
|---|---|---|
| 관측 == 시뮬 | `train/explore/sim/tools/rl_node_check`: CPU 시뮬(계층형) 걸음마다 시뮬 라이다 장·지도 자세·바퀴·IMU·명령을 `RobotEnv` 에 넣고 결정마다 비교 | 시험 맵 bookstore·small-house, 자세 오차 끔·켬, 정책·가장 가까운 규칙·무작위가 몬 판 120 개 · 결정 17,545 번: 관측 2D·1D·후보 칸·로봇 지도·닫힘 길·상태 칸 **모두 비트 같음**, 판 끝 코드(닫힘 성공·진척 없음)도 같음 |
| 정방향 == 학습기 | `train/explore/ppo/tools/policy_cpu_check` (GPU) | 학습이 GPU 를 쓰는 동안은 못 돌림 — 아래 "남은 것". 대신 같은 체크포인트(h4 반복 2511)를 CPU 정방향으로 CPU 시뮬에서 몰면: 학습 평가 조건(자세 오차·관측 지연 0/1) 성공 32/32 (학습기 평가 it2500: 0.94·1.00), 노드 조건(자세 오차·지연 0) 31/32, 로짓 크기도 학습 로그와 같음(있는 칸 평균 \|l̄\| 0.89 vs 0.87~0.90). 자세 오차를 끈(완벽한 지도 — 학습에 없던) 시뮬에서는 bookstore 7/16 으로 떨어짐(bf16 흉내도 6/16 — 수치 탓 아님) |
| 녹화 다시 돌리기 | `ros2 bag play ~/limo_bags/1103_M_drive1 --clock` + 노드 `dry_run` (녹화의 Cartographer TF·`/map` 그대로) | 로봇 지도가 Cartographer `/map` 과 벽 자리가 맞음. 깊이 카메라 층 끄면 결정 75 번·새 목표 46 개: 44 개가 모름 칸 0.15 m 안(프론티어), 43 개가 Nav2 가 짤 수 있는 자리(막힘 0.2 m 팽창·모름 지나감 — `allow_unknown`) |
| 깊이 카메라 층 | 같은 녹화, 층 켬 | **정렬 끈 깊이(1103 녹화, 640 × 400 fx 478, 숙임이 학습 값과 1.4° 다름)는 바닥을 장애물로 찍어(1,887 칸) 로봇을 둘러싸고 33 s 에 거짓 닫힘.** 정렬 켬 녹화(1104_M_cmd_mix, `map_frame:=odom`)는 529 칸·후보 그대로 |
| Nav2 액션 | 같은 녹화 + 가짜 `navigate_to_pose`·`spin` 서버(4 s 뒤 성공, 3 번째마다 실패) | 75 s 동안 목표 48 개: 새 목표로 바뀌며 취소 42 · 도착 3 · 실패 3 → 그 둘레 후보를 피해 다른 후보로 8 번 · 시작 때 후보 없어 spin 1 번, 오류 없음. 실제 Nav2·Cartographer 와는 아직 안 돌림 |
| 시간 | 노드 기록 | 걸음(라이다 장 하나 → 관측) 평균 2.2~5.1 ms(최대 8.1), 정방향 평균 3.5 ms(최대 4.6) — 이 PC 한 코어. 젯슨 Orin Nano CPU 는 안 잼 |

```bash
# 관측 == 시뮬 (학습 PC, CPU 만)
cmake -S train/explore/sim -B build_sim -DCMAKE_CUDA_COMPILER=/usr/local/cuda-13.0/bin/nvcc -DCMAKE_CUDA_ARCHITECTURES=86 && cmake --build build_sim -j --target rl_node_check
build_sim/rl_node_check train/explore/scenes/test/small-house.scene --decisions 400 --seed 1 --ckpt <정책.bin> [--pose-error] [--dump obs.bin]
# 정방향 == 학습기 act_inference (GPU 여유 3 GB 이상일 때 · 젯슨에서도 -DCMAKE_CUDA_ARCHITECTURES=87)
cmake -S train/explore/ppo -B build_ppo -DCMAKE_CUDA_COMPILER=/usr/local/cuda-13.0/bin/nvcc && cmake --build build_ppo -j --target policy_cpu_check
build_ppo/policy_cpu_check --ckpt <정책.bin> obs.bin
# 녹화 다시 돌리기 (한 터미널씩, ROS_DOMAIN_ID 를 다른 것과 겹치지 않게)
ros2 run limo_fire_rl explore_rl --ros-args -p use_sim_time:=true -p dry_run:=true -p checkpoint:=<정책.bin> -p debug_dir:=/tmp/rl_replay
ros2 bag play ~/limo_bags/1103_M_drive1 --clock
```

### 깊이 카메라 층을 볼 것

학습 정책은 깊이 카메라 장애물 층(env.yaml `camera_obstacles`)이 막은 곳을 후보·닫힘에서 뺀 채로 배웠다. 실물 깊이가 학습 모델과 다르면(정렬 꺼짐·카메라 숙임 바뀜)
바닥이 장애물로 찍혀 후보가 사라지고 **아직 안 본 곳이 있는데 "탐사 끝"** 이 난다. 노드는 `camera_info` 가 학습 모델(정렬 켬 fx 489.39)과 다르면 경고한다.
`dry_run` 때 RViz 에 `/explore_rl/camera_layer`(OccupancyGrid)를 띄워 빈 바닥에 점이 깔리는지 보고, 깔리면 `camera_layer:=false` 로 끈다
(후보가 학습 때와 조금 달라짐 — 정책 입력 관측은 그대로).

## 시뮬과 다른 곳 (알고 쓰는 차이)

- **주행 속도**: 정책은 시뮬 추종기 1 m/s(`env.yaml action.path.v`, 기준선 fast 판)로 배웠다. 실물 Nav2 RPP 는 `nav2_params.yaml` 0.26 m/s. 한 결정(1 s) 동안 덜 가므로
  같은 목표를 더 여러 번 고르게 된다(목표는 그대로라 동작은 같음). 지도 진척 60 s 시계는 느린 만큼 더 쉽게 찬다 → 기본은 멈추지 않고 알림.
- **지연**: 시뮬은 판마다 0 또는 1 걸음 늦은 관측을 뽑는다. 실물은 센서 지연이 값 안에 있어 노드는 지연 0.
- **빔 펴기 각속도**: 시뮬은 참 몸통 ω, 실물은 `/imu` 각속도 z.
- **깊이**: 시뮬 깊이 모델은 정렬 켬 640 × 480 의 위 400 행(fx 489.39). 노드는 시뮬 관측 픽셀 광선을 실물 `camera_info` 로 다시 투영해 가장 가까운 픽셀을 읽는다
  (정렬 켬이면 같은 픽셀, 반 픽셀 반올림 차이). 정렬을 끈 녹화(1103, 640 × 400 fx 478)는 깊이·컬러 사이 수 cm·숙임 1.4° 차이를 무시한다.
- **지도 자세 오차**: 시뮬은 오차 모델로 흔든 자세로 쌓고, 실물은 Cartographer 자세 그대로. Cartographer 가 나중에 고리를 닫아도 이미 쌓은 로봇 지도는 안 고친다(시뮬도 같음).
- **지도 창**: 시작점 가운데 25.6 m × 25.6 m (`map.size` 512 × 5 cm). 밖은 모름으로 남는다.
- **복구·목표 방향**: 시뮬 하위 층(경로 계획·RPP·복구)은 Nav2 가 맡는다. Nav2 목표 방향은 로봇 → 목표 방향으로 준다(시뮬은 방향을 안 맞춤).
- **덮음·보상**: 정답 벽이 없어 0. 관측·판 끝에는 안 쓰인다.

## 남은 것 (2026-11-11 기준)

- **정방향 == 학습기 GPU 비교**: `policy_cpu_check` 를 못 돌림(학습 explore_h5 가 GPU 메모리를 거의 다 씀). 학습이 끝나거나 젯슨에서
  `-DCMAKE_CUDA_ARCHITECTURES=87` 로 빌드해 `obs.bin`(노드 `debug_dir` 또는 `rl_node_check --dump`)으로 돌린다. 기준: 정확 경로 슬롯 같음·로짓 차 ≤ 1e-4.
- **젯슨 빌드·시간**: x86 에서만 빌드·시험함. 젯슨 Orin Nano 에서 `colcon build` 와 노드 로그의 "걸음 평균·정방향" ms 를 본다(장 하나 100 ms 안이면 됨).
- **실물 깊이 정렬**: 지금 로봇 설정(`depth_registration: true`, 640 × 480)이 학습 깊이 모델과 같은지 `dry_run` 에서 `camera_layer` 를 보고 확인.

