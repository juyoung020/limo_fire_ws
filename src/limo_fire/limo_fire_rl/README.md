# limo_fire_rl — 탐사 노드 (RL 정책 · 프론티어)

노드가 셋 있다. `explore_frontier`·`explore_rl` 은 Nav2 `NavigateToPose` 로 목표를 보내고 `low_rl` 은 아래층 RL 정책이 바로 `/cmd_vel` 을 낸다. 셋 다 `~/enable`(std_srvs/SetBool)로 켜고 끄며, 운용 화면 RViz 의 **탐사 패널**
("Auto explore ON/OFF" 버튼, [limo_fire_rviz](../limo_fire_rviz/README.md))이 이 서비스를 부른다. 실물에서 RViz 를 켜는 법은
[limo_fire_bringup README "실물에서 운용 화면(RViz) 켜기"](../limo_fire_bringup/README.md#실물에서-운용-화면rviz-켜기).

| 노드 | 무엇 | 켜기 |
|---|---|---|
| `explore_frontier` | 프론티어 탐사 (아래 "프론티어 탐사") — 탐사 기준선과 같은 찾기·고르기 | 멈춘 채 뜸 → 버튼 |
| `explore_rl` | 학습한 계층형 정책 (아래 "RL 정책") | 기본 바로 켜짐 (`enabled:=false` 면 버튼) |
| `low_rl` | 위층 fuel + 아래층 순수 RL 정책 (아래 "합친 탐사") — Nav2 없음 | 멈춘 채 뜸 → 버튼 (리모컨 오토일 때만) |

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

## 합친 탐사 (`low_rl` — 위층 fuel + 아래층 순수 RL, 2026-11-16)

시뮬 `combine_run --upper fuel --policy` 와 **같은 코드·같은 값**으로 로봇이 탐사한다. 위층 fuel(`train/explore/scripts/fuel_upper.h`)이 1 s 마다 로봇이 쌓은 지도에서
목표 칸을 고르고, 아래층 순수 RL 정책(체크포인트)이 0.1 s 마다 `/cmd_vel` 을 낸다. Nav2 는 쓰지 않는다. Cartographer 는 지도 자세(TF `map → base_link`)만 준다.

| 차례 | 실물 노드 | 시뮬 (`train/explore/scripts/combine_run.cpp`, 1bd79b5 기준 줄) |
|---|---|---|
| 설정 | `RobotEnv` 합친 탐사 모드 — 목표는 위층, 탐사 판 끝 규칙, 막힘은 신호만 | 224-231 (`low_auto 0` · `low_explore 1`) |
| 결정 때 | 걸음 % 10 == 0 또는 다시 결정 | 544 |
| 목표 | fuel (시간 + 전역 + 시점, v 0.981 · ω 3.232 · t0 0.10 — `config/low_rl.yaml`) | 629-645 · 1118-1123 |
| fuel 이 못 냄 | 피하지 않는 첫 후보(slot0), 같은 덩어리·경로 있음이면 그대로(stick) | 546-555 · 1127-1143 |
| `upper:=slot0` | fuel 을 안 부르고 늘 위 slot0 차례 (시뮬 `combine_run --upper slot0 --policy` — fuel 과 A/B 용, launch 인자 `upper`) | 546-555 · 1127-1146 |
| `upper:=decider` | 판단 모델(d1-3B, zero-shot)이 hier 후보 8 개 중 고름 — `scripts/low_decider.h` (시뮬 `combine_run --upper decider --decider-input both` 와 같은 요청: 지도 사진 + 후보 글). `decider_period`(기본 3 s — 로봇 답 약 2.6 s + 여유)마다 비동기로 묻고, 기다리는 동안 지금 목표로 몰며, 답이 오면(그 후보가 아직 있으면) 바꿈. 목표가 죽으면(막힘·통과·경로 없음) 기다리지 않고 slot0 차례로 바로. `decider_timeout`(8 s) 넘은 답은 버림 | 546-555 · 1127-1146 |
| 목표 없음 | 시뮬 규칙 추종기와 같은 제자리 회전 (`RobotEnv::low_act` → env.h `path_step_*`) | 1144-1146 |
| 다시 결정 | 통과 · 경로 끝 밖 · 경로 없음 · 막힘 신호 → fuel 기억 비움(경로 끝 밖·없음·막힘은 그 칸 0.5 m 를 10 s 피함) | 1326-1335 · 1351-1371 · 345-359 |
| 끝 | 지도 닫힘 + 지도 진척 없음 10 s (성공) · 진척 없음 · 시간 → 명령 0, `~/done` true | 1270 (`finish_step`) |

위층 결정·목표 넘기기는 `train/explore/scripts/low_explore_ctl.h` 하나를 노드와 확인 도구가 같이 쓴다. 관측 설정은 체크포인트 꼬리에서 읽는다.

**멈춤 조건 (이것만 명령을 0 으로 — 정책 명령을 고치는 규칙 층은 없다)**

| 조건 | 동작 |
|---|---|
| 처음 · `~/enable false` | 명령 0 (`idle`). `~/enable true` 로 켬 — **리모컨이 오토일 때만 켜진다** |
| 리모컨이 오토가 아님 (`/rc_state` `buttons[1]` ≠ 2 — SWB 위가 2) 또는 `/rc_state` 0.5 s 넘게 안 옴 | 명령 0 · 꺼짐 (`paused`). **다시 오토로 바꿔도 저절로 안 감 — `~/enable true` 를 다시** |
| 라이다·TF·오도메트리 0.3 s 넘게 안 옴 | 명령 0, 돌아오면 이어 감 |
| 닿음 (`train/explore/sim/include/contact.h` `sense_step` — 라이다·깊이·멈춤·IMU 수평 가속 > 15 m/s², 마스크 71 · 멈춤 2 걸음, low_w3 학습 `run.json` `con_cfg`) | 명령 0 · 꺼짐. `contact_resume_s:=1.0` 이면 1 s 뒤 저절로 켬 (기본 0 = `~/enable` 따로) |
| 끝 (위 표) | 명령 0 · `done` |

- `dry_run:=true`(기본): `/cmd_vel` 발행자를 아예 안 만들고 같은 명령을 `~/cmd_debug`(TwistStamped)에만. dry run 에서는 닿음의 "멈춤" 신호(명령 대비 바퀴 정지)를 뺀다 — 명령이 안 나가 늘 참.
- `max_speed`: 기본 = 학습 한도(자르지 않음). 토픽: `~/state`(idle · running · paused · done) · `~/status`(한 줄 — 까닭·리모컨·목표·명령·시간) · `~/done` · `~/goal_marker` · `~/cmd_debug`.
- RViz 탐사 패널: 고르기에서 "fuel + 아래층 RL (/low_rl)". `paused` 면 상세 줄에 까닭.
- 리모컨 칸 확인: 녹화 `1103_M_drive3`(리모컨으로 몬 것)의 `/rc_state` buttons 는 모두 `[3, 1, 3, 3]`, `1104_M_cmd_mix`(ROS 명령으로 몬 것)는 모두 `[3, 2, 3, 3]`.

**체크포인트**: low_w3 반복 36000 (`w3_model_36000.bin`, md5 `a43b5f95…`) — 검증 세트 2 고친 계측 옛 물리 벽 0.197 · 사람 0.337 · 도달 0.922 · 속도 0.661,
실물형 물리(slip1) 벽 0.178 · 사람 0.381. 상위 후보 가운데 두 물리를 다 잰 것 (판정은 모두 기준 미달 — 사람 기준 0.10).

| 확인 (이 PC, 2026-11-16) | 결과 |
|---|---|
| `train/explore/sim/tools/low_explore_check` — CPU 시뮬과 `RobotEnv` 가 각자 위층 결정, 시뮬 라이다 장·감각·깊이를 `RobotEnv` 에 넣고 걸음마다 비교 | bookstore · small-house, 씨앗 1~3, 사람 0 · 2, 800~2500 걸음: 위층 명령 · 아래층 관측(그림·1D) · 명령 · 판 끝 **비트 같음** (목표 없는 걸음 포함) |
| 녹화 `1104_M_cmd_mix` (깊이 정렬) 에 `explore_fuel_rl.launch.xml` dry run (`map_frame:=odom`, `use_sim_time:=true`) | 발행자에 `/cmd_vel` 없음. 켬 뒤 결정 141 · 통과 24. x86 한 코어: 걸음 7.3 · 관측 2.1 · 정책 7.7 · 위층 평균 12 ms (최대 22) |
| 리모컨 (녹화 대신 손으로 낸 `/rc_state`) | 수동(1) → paused · 다시 오토(2) → 그대로 paused · `~/enable` → running · 안 옴 → paused · 안 오는데 `~/enable` → 거절 |

젯슨 Orin Nano 빌드·시간·정지 dry run 은 아직 이 README 에 적지 않음 (그쪽 확인 뒤).

### 실행 (로봇)

`WS` = 이 저장소를 받은 워크스페이스 (예: `export WS=$HOME/limo_fire_ws`). 체크포인트는 로봇의 `~/ckpt/low_w3_36000.bin` (다르면 `checkpoint:=`).

```bash
# 빌드 (한 번)
cd $WS && git pull --rebase --autostash && source env.sh
colcon build --packages-select limo_fire_rl limo_fire_rviz && source install/local_setup.bash
mkdir -p ~/ckpt   # 학습 PC 에서: scp <...>/w3_model_36000.bin wego@<로봇>:ckpt/low_w3_36000.bin

# 터미널 1 — 센서·섀시·EKF (대시보드 limo_base 가 떠 있으면 켜지 않는다 — 섀시 포트는 하나만)
ros2 launch limo_fire_bringup robot.launch.xml

# 터미널 2 — 1) dry run 먼저: Cartographer + low_rl + RViz, /cmd_vel 안 냄
ros2 launch limo_fire_rl explore_fuel_rl.launch.xml stack:=true rviz:=true dry_run:=true
#   RViz 탐사 패널 "fuel + 아래층 RL (/low_rl)" → Auto explore ON. 확인할 것:
ros2 topic echo /low_rl/status          # running · 리모컨 오토 · 목표 칸 · 명령 · 시간
ros2 topic echo /low_rl/cmd_debug       # 정책이 낼 명령 (v ≤ 1.0 m/s, ω ≤ 3.14 rad/s)
#   리모컨 SWB 를 내렸다 올림 → ~/state paused 로 바뀌고, 올려도 paused 그대로인지 (다시 버튼을 눌러야 running)

# 2) 점검표를 다 확인한 뒤에만 — 실제 주행
ros2 launch limo_fire_rl explore_fuel_rl.launch.xml stack:=true rviz:=true dry_run:=false
ros2 service call /low_rl/enable std_srvs/srv/SetBool "{data: true}"    # 버튼과 같음
ros2 topic echo /low_rl/done            # true 면 끝 (지도 닫힘)
```

**멈추는 법** (어느 것이든): 리모컨 SWB 를 내림(오토 해제 — 섀시가 ROS 명령을 안 받고 노드도 꺼짐) · RViz 버튼 OFF ·
`ros2 service call /low_rl/enable std_srvs/srv/SetBool "{data: false}"` · 터미널 2 Ctrl+C (마지막 명령이 남지 않게 SWB 를 먼저 내린다).

**실제 주행 전 점검표**
- [ ] 충전기를 뺐다 (꽂힌 채 움직이지 않는다)
- [ ] 주행할 곳이 비었다 — 사람·깨질 것 치움, 지도가 닫히는 공간(문 닫음)
- [ ] 사람이 로봇 곁에서 리모컨을 들고 있다 (SWB 를 바로 내릴 수 있게)
- [ ] 리모컨이 켜졌고 오토 모드 (SWB 위) — `ros2 topic echo /rc_state --field buttons` 의 둘째 값이 2
- [ ] dry run 에서 `~/status` 가 running, 목표·`~/cmd_debug` 가 말이 되고, SWB 내림 → paused 를 봤다
- [ ] 다른 주행 노드(Nav2·`explore_rl`·`explore_frontier`·teleop)가 `/cmd_vel` 을 안 낸다 — `ros2 topic info /cmd_vel` 의 발행자가 없거나 low_rl 뿐
- [ ] 젯슨 시간: `~/status` 의 평균 걸음 + 관측 + 정책 + 위층이 100 ms 안 (넘으면 멈추고 알림)

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

### 아래층 정책 관측 · `low_rl` 시험 노드 (2026-11-16 — 로봇을 몰지 않음)

`RobotEnv(hc, true)` (아래층 모드)가 학습(`train --low`, 목적지만 `SIM_LOW_GOALONLY=1`)과 **같은 함수로** 아래층 관측 전부를 만든다:
그림(가까운 지도 6 채널 — 깊이 카메라 층 채널 포함 · 라이다 4 장 그림 · 깊이 · 넓은 지도 6 채널 — 막힘·빈 칸·모름·지나간 곳·목적지 점·깊이 층)과
1D(바퀴·자이로·헛돎·EKF · 목적지 · 직전 행동 4 걸음 · 움직이는 것 칸 · 칸 넷 · 몸 둘레 간격 8 칸 — `SIM_LOW_BODY`, `lowrl_pack` 뒤 `lowrl_body_clear`). 시뮬 걸음 차례(`env_ref.cpp` `step_env` + `low_prepare_env`) 그대로:
`low_act`(정책 출력 → `lowrl_begin`·`lowrl_act`: 바퀴 한도·직전 행동·사람 추적) → `step`(라이다 장 — 지도·위층 관측, `finish_step` 안의 `low_finish`·`lowrl_finish` 포함)
→ `set_depth` → `set_goal`(위층이 새 목적지를 줄 때 — 시뮬 `lowrl_goal_set` 의 칸) → `low_obs`(깊이 카메라 층 `lowrl_cam_due` · 새 목표면 `goal_only_issue` ·
`lowrl_pack`·`lowrl_img`·`lowrl_wred`·`lowrl_wide`). 경로 계획은 없다(학습 판 정책 환경도 안 짬). 보조 예측 재료(`SIM_LOW_AUX`)는 학습 전용이라 안 만든다.

`low_rl` 노드는 이 관측으로 아래층 정책(`LowPolicy`)을 돌린다 — 지금은 위층 fuel 과 합친 탐사 노드다(아래 "합친 탐사"). `upper:=topic` 이면 목적지를
`~/goal`(PoseStamped, `map` 틀, 위치 NaN = 없음)로 받는다(시험용). 관측 설정(`SIM_LOW_*`)은 **체크포인트 꼬리에서 읽는다**
(학습기가 적음 — ppo/README "체크포인트 설정 꼬리"). 환경변수를 다른 값으로 명시하면 시작하지 않는다. 꼬리 없는 옛 체크포인트는 환경변수 그대로라
학습 판과 같은 값을 export (low_w3 옛 체크포인트: `SIM_LOW_WIDE=1 SIM_LOW_GOALONLY=1 SIM_LOW_OBS_MOV=1 SIM_LOW_MOV4=1 SIM_LOW_CAM_EVERY=1 SIM_LOW_TRACK_MISS=20`); 체크포인트 1D 칸 수가 설정과 다르면 멈춘다.
`record:=<파일>` 이면 입력·관측·명령을 녹화한다(`low_node_replay` 로 다시 돌려 같은지·시간).

| 확인 | 결과 |
|---|---|
| `sim/tools/low_node_check` — 학습 판 CPU 시뮬(환경 하나, 자세 오차·관측 지연 끔) 걸음마다 라이다 장·감각·깊이·시뮬이 뽑은 목적지를 `RobotEnv` 에 넣고 아래층 관측·명령 비교, `--gpu` 면 GPU 시뮬과도 직접 | 위 export 로 bookstore·small-house, 실물 CPU 정책(low_w3)이 몬 1,500 걸음 판 묶음 — CPU 비교 4 번(씨앗 1~4, 사람 8) · GPU 비교 2 번: 그림·1D·명령 **다름 0 — CPU 와 GPU 둘 다** (움직이는 것 찬 관측 85~526 번, 둘째 칸까지 찬 관측 최대 97 번). 칸 넷 끈 32 칸 정책(w1g)도 같음. 깊이 바이트 하나를 바꾸면 잡음 |
| `low_node_replay` — 위 녹화를 `RobotEnv` 만으로 다시 돌림 | x86 · **Orin(aarch64, g++ 11 -O2 -ffp-contract=off) 둘 다 비트 같음** |
| 녹화 bag(`1103_M_drive1`, 74 s)에 노드를 dry run 으로 | `~/cmd_debug` 만 나옴(노드 발행자: cmd_debug·status), 노드 녹화를 x86·Orin 에서 다시 돌려 같음 |
| Orin 시간(한 코어, 다른 일로 부하 평균 ≈ 6) | 걸음 하나: `step` 2.4 ms · `low_obs` 2.7~3.0 ms · `low_act` 0.02 ms · 정책 NEON 5.0 ms → **합 ≈ 10.4 ms / 100 ms** (99 % 값 ≈ 15 ms) |
| 몸 둘레 간격 8 칸 (`SIM_LOW_BODY=1`, 1D 64, 2026-11-16) | `low_node_check` CPU 4 번 · `--gpu` 2 번(사람 8, 1,500 걸음): 다름 0 — 몸 둘레 1 m 안에 막힘이 있는 관측 1,357~1,493 번. 녹화 다시 돌리기 x86 · Orin 비트 같음. Orin(부하 ≈ 0.5): `low_obs` 2.90 ms (몸 둘레 없는 녹화 2.58 ms — 다른 녹화라 차이 ≈ 0.3 ms 는 잠정) · `step` 2.78 · 정책 NEON 4.07 ms |

시뮬과 다른 곳: 시뮬은 판 끝(닿음·목표 수 등 — `low_finish`)에 환경을 리셋하지만 실물은 이어 간다(같은 함수가 사람 추적·목표 상태를 비우는 것만 같음 — 로그에 알림).
목적지가 없으면 시뮬은 규칙 추종기가 몰지만 실물 노드는 명령 0.

### 아래층 움직이는 것 칸 (2026-11-16 — `SIM_LOW_OBS_MOV`·`SIM_LOW_MOV4` 체크포인트)

아래층 정책 1D 의 움직이는 것 칸(kLoMov 6 칸 · 칸 넷 kLoMov4 24 칸, 1D 32 → 56)은 `include/limo_fire_rl/low_mov.h` `LowMov` 가 만든다 — 시뮬과 같은 함수
(`low_rl.h` 라이다 덩어리 칼만 추적 `lowrl_trk_step` · `lowrl_trk_movers` · 칸 채우기 `lowrl_mov_slots`)를 RobotEnv 가 이미 들고 있는 라이다 칸 링·자세로 부른다.
`RobotEnv::set_mov_track(true)` 면 `step` 첫머리(새 장을 넣기 전 — 시뮬 `lowrl_act` 자리)에 추적하고, 관측 때 `mov_slots(mov, mov4)`. 정책은
`network/policy_cpu.h` `LowPolicy` 가 1D 칸 수(32·56)를 체크포인트 정규화 차원에서 읽는다. `-ffp-contract=off` 빌드(이 패키지 기본)에서 비트까지 같다:

| 확인 | 결과 |
|---|---|
| GPU 녹화 다시 돌리기: `env_verify --low --mov-dump` (SIM_EXACT=1, 넓은 지도·목적지만·OBS_MOV·MOV4, 사람 상한 4) → `sim/tools/low_mov_replay` | 환경 16 × 600 걸음: 추적 한 걸음 9,503 번 · 칸 9,616 번 다름 0 (움직이는 것 1 개 찬 걸음 515 · 2 개 60 · 3 개 3). x86 과 **Orin(aarch64) 둘 다 비트 같음**. 칸 넷 끈 녹화(8 × 400)도 같음. 사람 상한 8, 16 × 800 걸음: 추적 12,669 번 · 칸 12,816 번 다름 0 (2 개 찬 걸음 118 · 3 개 15 — 넷 다 찬 걸음은 아직 녹화에 없음), x86·Orin 같음. 일부러 라이다 거리에 3 mm 더한 녹화는 추적 다름 38 으로 잡음 |
| 차례: `rl_node_check --mov --people 6` (CPU 시뮬 장을 RobotEnv 에 넣고 걸음마다 RobotEnv 추적 vs 시뮬 링 추적) | bookstore·small-house 씨앗 1~5 걸음 약 1.1 만 번 다름 0, 다른 관측도 그대로 비트 같음 |
| Orin 시간 (`~/decider_test`) | 추적 한 걸음 8~9 µs · 칸 채우기 0.1 µs. 정책 정방향 56 칸 NEON 4.8~4.9 ms (스칼라 10.2 ms, NEON 출력 == 스칼라 바이트 같음) |

```bash
SIM_EXACT=1 SIM_LOW_WIDE=1 SIM_LOW_GOALONLY=1 SIM_LOW_OBS_MOV=1 SIM_LOW_MOV4=1 build_sim/env_verify --list train/explore/scenes/train/list.tsv --low --envs 16 --steps 600 --people 4 --mov-dump rec.bin
build_sim/low_mov_replay rec.bin   # 젯슨: g++ -O2 -std=c++17 -ffp-contract=off -I<explore>/sim/include -I<explore>/mdp -I<이 패키지>/include low_mov_replay.cpp
```

노드(`explore_rl`)는 아직 아래층 정책을 몰지 않는다 — 이 칸은 아래층 정책을 붙일 때 쓰는 부품이다. 추적은 `sense_step`(닿음 구분)에는 안 넣었다(RobotEnv 가 쓰는 닿음 값은 그대로).

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



### 위층 판단 모델 (`upper:=decider`) 띄우기

로봇 안에서 llama-server(판단 모델 d1-3B, `/v1/systemone` 이 있는 빌드)를 먼저 띄운다. 메모리가 빠듯하면(가용 4 GB 아래) 다른 무거운 것을 끄고 띄운다:

```bash
cd ~/decider_test && nohup ./llama.cpp/build/bin/llama-server -m models/d1-3B-Q8_0.gguf --mmproj models/mmproj-d1-3B-Q8_0.gguf \
  --host 127.0.0.1 --port 8171 -ngl 99 -c 4096 -np 1 > server_dec.log 2>&1 &
# 뜰 때까지 (모델 읽기 약 1 분): grep -m1 "listening" server_dec.log
ros2 launch limo_fire_rl explore_fuel_rl.launch.xml upper:=decider [decider_period:=3.0] [dry_run:=false] [rviz:=true]
```

상태는 `~/status` 끝에 "판단 모델: 보냄 · 답 (마지막 ms) · 시간 끝 · 실패 · 후보 안 맞음 · slot0 대신". 서버가 없거나 답이 늦으면 slot0 차례로 계속 돈다 (멈추지 않음).
카메라 그림(시뮬 `--decider-input all`)은 시뮬 장면에 광선을 쏴 칠한 그림이라 실물에는 같은 것이 없어 쓰지 않는다 (`decider_input` text · image · both).
