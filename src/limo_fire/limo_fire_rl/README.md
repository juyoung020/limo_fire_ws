# limo_fire_rl — 탐사 노드 (합친 탐사 · 프론티어)

노드가 둘 있다. `explore_frontier` 는 Nav2 `NavigateToPose` 로 목표를 보내고, `low_rl` 은 위층이 고른 목표까지 아래층 A* 경로 추종이 바로 `/cmd_vel` 을 낸다.
둘 다 `~/enable`(std_srvs/SetBool)로 켜고 끄며, 운용 화면 RViz 의 **탐사 패널**
("Auto explore ON/OFF" 버튼, [limo_fire_rviz](../limo_fire_rviz/README.md))이 이 서비스를 부른다. 실물에서 RViz 를 켜는 법은
[limo_fire_bringup README "실물에서 운용 화면(RViz) 켜기"](../limo_fire_bringup/README.md#실물에서-운용-화면rviz-켜기).

| 노드 | 무엇 | 켜기 |
|---|---|---|
| `explore_frontier` | 프론티어 탐사 (아래 "프론티어 탐사") — 탐사 기준선과 같은 찾기·고르기 | 멈춘 채 뜸 → 버튼 |
| `low_rl` | 위층 판단 모델(또는 cand · fuel · slot0) + 아래층 A* 경로 추종 (아래 "합친 탐사") — Nav2 없음 | 멈춘 채 뜸 → 버튼 (리모컨 오토일 때만) |

탐사 강화학습(RL 정책 노드 `explore_rl`, 아래층 RL 정책 `lower:=rl`)은 2026-11-16 에 그만두고 [archive/rl](../../../archive/rl/README.md) 로 옮겼다
(실물에서 시뮬과의 격차가 컸고, A* 경로 추종 아래층이 훨씬 빠르고 잘 됐다). 그때의 이 README 는 `archive/rl/src/limo_fire/limo_fire_rl/README.md`.
패키지·노드 이름(`limo_fire_rl`, `low_rl`)은 launch·녹화·RViz 설정이 쓰고 있어 그대로 둔다.

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

## 합친 탐사 (`low_rl` — 위층 판단 모델 + 아래층 A*, 2026-11-16)

> **확정 구성 (2026-11-16 실물 시험):** 위층 판단 모델(`upper:=decider`, 기본) + 아래층 A* 경로 추종(규칙 추종기 — 유일한 아래층).
> 같은 날 실물에서 fuel+RL · 가까운 후보+RL · 판단 모델+RL · 판단 모델+A* 를 돌려 판단 모델+A* 가 가장 잘 됐다(녹화 `~/limo_bags/1116_*`).
> 실행: ① 로봇 page cache 비우기(`sync && echo 3 | sudo -n tee /proc/sys/vm/drop_caches` — Orin GPU 메모리 확보) → ② 아래 "판단 모델" 절의 llama-server →
> ③ `robot.launch.xml` → ④ `ros2 launch limo_fire_rl explore_fuel_rl.launch.xml stack:=true rviz:=true dry_run:=false` → 리모컨 SWB 위 + 패널 버튼. 멈춤은 SWB 내림.

시뮬(`low_explore_check` · `combine_run`)과 **같은 코드·같은 값**으로 로봇이 탐사한다. 위층이 로봇이 쌓은 지도에서 목표를 고르고(결정 차례 1 s — 판단 모델은 그중 사건 때만 묻는다), 아래층 규칙 추종기(로봇 지도 A* `plan_ref.h` + 경로 추종 `path_step_*` — `RobotEnv::low_act` 계층형 가지)가 0.1 s 마다 `/cmd_vel` 을 낸다.
Nav2 는 쓰지 않는다. Cartographer 는 지도 자세(TF `map → base_link`)만 준다. 관측 설정은 `train/explore/sim/config/env.yaml` 그대로.

| 차례 | 실물 노드 | 시뮬 (`train/explore/sim/tools/low_explore_check` · `scripts/combine_run`) |
|---|---|---|
| 설정 | `RobotEnv` 합친 탐사 모드 — 목표는 위층, 탐사 판 끝 규칙, 막힘은 신호만 | `low_rl 1` · `low_explore 1` |
| 결정 때 | 걸음 % 10 == 0 또는 다시 결정 | 같음 |
| `upper:=decider` (기본) | 판단 모델(d1-3B, zero-shot)이 `scripts/cand_view.h` 후보 앞 6 개 중 고름 — `scripts/low_decider.h` (요청: 지도 사진 + 후보 글). 사건 때만(`decider_mode: event` — 목표에 1 m 다가감 · 목표 실패 · 후보 크게 바뀜 · 10 s 최대 간격) 비동기로 묻고, 기다리는 동안 지금 목표로 몰며, 답이 오면(그 후보가 아직 있으면) 바꿈. 목표가 죽으면 기다리지 않고 slot0 차례로 바로. `decider_timeout`(8 s) 넘은 답은 버림. `decider_mode: periodic` 이면 `decider_period`(3 s)마다 | `low_explore_check --upper decider` |
| `upper:=cand` | `cand_view.h` 0 번 후보, 지금 목표 번호가 있으면 붙듦 — 시뮬 위층 비교(`train/explore/upper_lora/README.md` "위층 비교")에서 판단 모델보다 빠름 | `low_explore_check --upper cand` |
| `upper:=fuel` | fuel (시간 + 전역 + 시점, v 0.981 · ω 3.232 · t0 0.10, 붙들기 0.2, 시점은 로봇에서 0.75 m 밖·모름에서 0.21 m — `config/low_rl.yaml`, `scripts/README` "실물 1116"). 덩어리 없음: 피하지 않는 첫 후보(slot0). 시점 자격 되는 덩어리가 없음: 목표 없음 | `--upper fuel` |
| `upper:=slot0` | 피하지 않는 첫 후보, 같은 덩어리면 그대로(stick) — A/B 용. `slot_vp: true` 면 그 덩어리의 시점(`fu::slot_view`) — 시뮬에서 나빠 기본 끔 | `--upper slot0` |
| 아래층 | 규칙 추종기: 후보 번호(slot0 · 판단 모델) 또는 목표 칸(fuel)까지 A* 경로를 짜 따라감, 막히면 복구(물러나기·목표 포기 — `hier_path.h`) | 같은 함수 |
| 목표 없음 | 규칙 추종기의 제자리 회전 | 같음 |
| 다시 결정 | 통과 · 경로 끝 밖 · 경로 없음 · 막힘 신호 → fuel 기억 비움(경로 끝 밖·없음·막힘은 그 칸 0.5 m 를 10 s 피함) | 같음 |
| 끝 | 지도 닫힘 + 지도 진척 없음 10 s (성공) · 진척 없음 · 시간 → 명령 0, `~/done` true. `close_lidar: true` 면 닫힘을 라이다 지도만으로 — 깊이 카메라 층에만 막힌 프론티어는 안 닫힘 (`mdp/frontier.h`, 시뮬에서 나빠 기본 끔) | `finish_step` |

위층 결정·목표 넘기기는 `train/explore/scripts/low_explore_ctl.h` 하나를 노드와 확인 도구가 같이 쓴다.
launch 인자는 `upper` · `dry_run` · `record` · `decider_url` · `decider_period` 등뿐이고, 나머지 값(아래 `camera_layer` · `close_lidar` · `slot_vp` · `contact_resume_s` · `decider_mode` …)은
`config/low_rl.yaml` 에서 바꾼다.

**멈춤 조건 (명령을 0 으로)**

| 조건 | 동작 |
|---|---|
| 처음 · `~/enable false` | 명령 0 (`idle`). `~/enable true` 로 켬 — **리모컨이 오토일 때만 켜진다** |
| 리모컨이 오토가 아님 (`/rc_state` `buttons[1]` ≠ 2 — SWB 위가 2) 또는 `/rc_state` 0.5 s 넘게 안 옴 | 명령 0 · 꺼짐 (`paused`). **다시 오토로 바꿔도 저절로 안 감 — `~/enable true` 를 다시** |
| 라이다·TF·오도메트리 0.3 s 넘게 안 옴 | 명령 0, 돌아오면 이어 감 |
| 닿음 (`train/explore/sim/include/contact.h` `sense_step` — 라이다·깊이·멈춤·IMU 수평 가속 > 15 m/s², 마스크 71 · 멈춤 2 걸음 — 그만둔 RL 학습 low_w3 의 값) | 명령 0 · 꺼짐. `contact_resume_s: 1.0` 이면 1 s 뒤 저절로 켬 (기본 0 = `~/enable` 따로) |
| 끝 (위 표) | 명령 0 · `done` |

- `dry_run:=true`(기본): `/cmd_vel` 발행자를 아예 안 만들고 같은 명령을 `~/cmd_debug`(TwistStamped)에만. dry run 에서는 닿음의 "멈춤" 신호(명령 대비 바퀴 정지)를 뺀다 — 명령이 안 나가 늘 참.
- `max_speed`: 기본 = 시뮬 한도(자르지 않음). 토픽: `~/state`(idle · running · paused · done) · `~/status`(한 줄 — 까닭·리모컨·목표·명령·시간) · `~/done` · `~/goal_marker` · `~/cmd_debug` · `~/candidates`.
- `~/candidates` (MarkerArray, map 틀, 결정마다 — `upper:=cand` · `decider`): 새 후보(`train/explore/scripts/cand_view.h`)마다 구(서는 자리)·글(판단 모델 글자 · 안정 번호 · 예상 넓이/시간)·덩어리 대표 칸까지 선·덩어리 프론티어 칸, 색 초록 보통 · 파랑 지금 목표 · 주황 실패 기억 · 회색 쉬는 덩어리, 로봇 자취(60 s), 판단 모델 답 수·마지막 답 시간. RViz `explore_fuel_rl.rviz` 의 Candidates 표시. `upper:=cand` 는 그 후보 0 번으로 감(지금 목표 번호가 있으면 붙듦 — `low_explore_check --upper cand`).
- RViz 탐사 패널: 고르기에서 "합친 탐사 (/low_rl)". `paused` 면 상세 줄에 까닭.
- 리모컨 칸 확인: 녹화 `1103_M_drive3`(리모컨으로 몬 것)의 `/rc_state` buttons 는 모두 `[3, 1, 3, 3]`, `1104_M_cmd_mix`(ROS 명령으로 몬 것)는 모두 `[3, 2, 3, 3]`.

| 확인 (이 PC, 2026-11-16) | 결과 |
|---|---|
| `train/explore/sim/tools/low_explore_check` — CPU 시뮬과 `RobotEnv` 가 각자 위층 결정, 시뮬 라이다 장·감각·깊이를 `RobotEnv` 에 넣고 걸음마다 비교 | bookstore · small-house, 씨앗 1~3, 사람 0 · 2, 800~2500 걸음: 위층 명령 · 준비 뒤 경로(목표 칸 · 경로 점) · 명령 · 판 끝 **비트 같음** (목표 없는 걸음 포함). 위층 slot0 · fuel · cand · 판단 모델(가짜 응답기) 모두. `--node-obs` 면 실물 쪽을 노드와 같은 관측 설정으로 |
| `low_explore_check … --upper decider [--dec-latency 2.6] [--dec-mode event] [--dec-max 6] [--dec-match 1.5] [--dec-url http://…]` — 판단 모델 위층(`scripts/low_decider.h`)을 가짜 응답기(본문 해시로 글자, 지연 뒤 걸음에 답)로 두 쪽이 같이; `--dec-url` 이면 진짜 llama-server 답을 지연 걸음 뒤에 (시뮬로 평가) | bookstore · small-house, 씨앗 1~3, 사람 0 · 2, 지연 0 · 2.6 · 7.5 s, 1200~2500 걸음: 요청 본문 · 위층 명령 · 아래층 관측 · 명령 · 판 끝 **비트 같음**. 답의 후보 칸 맞춤 반경 0.5 m 면 답의 40 ~ 55 % 를 버려 기본 1.5 m (버림 10 ~ 25 %) |
| `train/explore/sim/tools/low_explore_replay` — 노드 녹화(`record:=`)를 `RobotEnv` + 위층만으로 다시 돌려 결정마다 까닭(주기·통과·경로 없음·막힘)·목표·뜀·목표 없음 까닭·닫힘을 적음, 지도 그림(`--pgm`) | 그때 설정(`--fuel-hyst 0 --fuel-vp-min 0 --fuel-vp-clear 0`)으로 `1116_fuel_rl_1` 659 / 659 · `1116_slot0_1`(`--upper slot0`) 86 / 86 관측 비트 같음 (531e7fc 위). 녹화 꼴은 판 4(`sim/tools/low_node_rec.h`), 옛 판 1~3 도 읽지만 그 뒤 위층·깊이 층이 바뀌어 1116 녹화의 명령은 지금 코드와 다름 |
| 녹화 `1104_M_cmd_mix` (깊이 정렬) 에 `explore_fuel_rl.launch.xml` dry run (`map_frame:=odom`, `use_sim_time:=true`, 그때 위층 fuel + 아래층 RL) | 발행자에 `/cmd_vel` 없음. 켬 뒤 결정 141 · 통과 24. x86 한 코어: 걸음 7.3 · 관측 2.1 · 위층 평균 12 ms (최대 22) — 그 뒤(2026-11-17, 한 스레드 · 안 쓰는 관측 끔) `1116_fuel_rl_1` dry run 걸음 5.53 · 위층 8.9 ms |
| 리모컨 (녹화 대신 손으로 낸 `/rc_state`) | 수동(1) → paused · 다시 오토(2) → 그대로 paused · `~/enable` → running · 안 옴 → paused · 안 오는데 `~/enable` → 거절 |

젯슨 Orin Nano (2026-11-17, 한 스레드): 걸음 약 5.9 ms · 위층 결정 약 9.5 ms — 0.1 s 걸음 안에 넉넉히 듦.

### 실행 (로봇)

`WS` = 이 저장소를 받은 워크스페이스 (예: `export WS=$HOME/limo_fire_ws`). 판단 모델 서버는 아래 "위층 판단 모델 띄우기" 로 먼저 띄운다.

```bash
# 빌드 (한 번)
cd $WS && git pull --rebase --autostash && source env.sh
colcon build --packages-select limo_fire_rl limo_fire_rviz && source install/local_setup.bash

# 터미널 1 — 센서·섀시·EKF (대시보드 limo_base 가 떠 있으면 켜지 않는다 — 섀시 포트는 하나만)
ros2 launch limo_fire_bringup robot.launch.xml

# 터미널 2 — 1) dry run 먼저: Cartographer + low_rl + RViz, /cmd_vel 안 냄
ros2 launch limo_fire_rl explore_fuel_rl.launch.xml stack:=true rviz:=true dry_run:=true
#   RViz 탐사 패널 "합친 탐사 (/low_rl)" → Auto explore ON. 확인할 것:
ros2 topic echo /low_rl/status          # running · 리모컨 오토 · 목표 칸 · 명령 · 시간
ros2 topic echo /low_rl/cmd_debug       # 낼 명령 (v ≤ 1.0 m/s, ω ≤ 3.14 rad/s)
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
- [ ] 다른 주행 노드(Nav2·`explore_frontier`·teleop)가 `/cmd_vel` 을 안 낸다 — `ros2 topic info /cmd_vel` 의 발행자가 없거나 low_rl 뿐
- [ ] 젯슨 시간: `~/status` 의 평균 걸음 + 관측 + 위층이 100 ms 안 (넘으면 멈추고 알림)

### 탐사 중 "○○ 로 가" (물체 지도 · `perception:=true`)

`explore_fuel_rl.launch.xml perception:=true` 면 YOLO 분할 + 물체 지도(limo_fire_perception) 와 `object_goto`(limo_fire_navigation, `mode explore`)가 같이 뜬다.

```bash
ros2 topic pub -1 /object_goto/goal std_msgs/msg/String "{data: '의자'}"   # 한글 별칭(config/object_goto.yaml) 또는 영문 클래스 이름. 빈 글 = 지시 지움
ros2 topic echo /object_goto/status        # 무엇을 찾았고 어디로 가는지 · /low_rl/goto_status: 가는 중 · 도착
```

- 물체 지도에 있으면: 물체 둘레 0.6~1.2 m 고리에서 로봇과 빈 칸으로 이어지는 가장 가까운 설 자리를 골라 `/low_rl/goto` 로 준다. `low_rl` 은 위층(판단 모델) 대신 그 칸을 목표로 같은 A* 아래층으로 가고,
  `goto_arrive_m`(0.4 m) 안에 들면 "도착" 을 알리고 멈춘다 (기본 `goto_resume: false`, true 면 탐사를 이어 감). 도착하면 물체 쪽으로 제자리에서 돌아(`~/goto_face`, 남은 각 < 0.15 rad 또는 5 s) 물체를 바라보고 끝낸다. object_goto 는 도착을 받으면 그 목표를 끝내고 새 목표까지 다시 찾지 않는다. 가는 동안은 탐사 끝 판정을 하지 않는다.
- 없으면: 글을 `/low_rl/instruction` 으로 넘겨 판단 모델 요청에 지시(`instruction`)와 아는 물체 10 개(`known_objects`, 거리 · 방향)를 넣고 질문을 "빨리 찾기" 로 바꾼다 — 물체 찾기 모드.
  물체 지도에 그 클래스가 올라오는 즉시 그리로 간다. 지시가 없으면 요청 본문은 예전과 바이트까지 같다 (`low_explore_check --dec-instr` 로 지시가 있을 때도 시뮬 == 실물 비트 확인).
- RViz: 고른 물체에 흰 고리 + "GOAL 이름", 설 자리 초록 원, 왼쪽 아래 글상자(`/object_goto/overlay`).
- 설명 글 검색(SigLIP2, "빨간 컵" 같은 글)은 다음 단계 — 지금은 클래스 이름·별칭만.

### 깊이 카메라 층을 볼 것

깊이 카메라 장애물 층(env.yaml `camera_obstacles`, 걸음 0.1 s 마다 갱신 — `update_every_step`)은 후보·닫힘에서 막힌 곳을 뺀다. 층 전용 촘촘한 깊이 격자(`layer_grid`)는 시뮬에서 끝 무렵 길이 더 막혀 기본 끔. 실물 깊이가 시뮬 모델과 다르면(정렬 꺼짐·카메라 숙임 바뀜)
바닥이 장애물로 찍혀 후보가 사라지고 **아직 안 본 곳이 있는데 "탐사 끝"** 이 난다 (옛 `explore_rl` 노드로 1103 녹화에서 확인 — 정렬 끈 깊이는 33 s 에 거짓 닫힘).
의심되면 `camera_layer: false` 로 끈다.

### 위층 판단 모델 (`upper:=decider`) 띄우기

로봇 안에서 llama-server(판단 모델 d1-3B, `/v1/systemone` 이 있는 빌드)를 먼저 띄운다. `-lm dio` 를 꼭 준다: 기본(mmap)이면 모델 파일(2.9 GB)이 page cache 에 남은 채
같은 내용을 GPU 버퍼로 또 잡아 Orin Nano 8 GB 에서 KV 캐시 할당이 실패한다(NvMap error 12 — Jetson 은 GPU 할당 때 page cache 를 못 비움). DirectIO 로 읽으면 page cache 를 안 거친다.
이미 page cache 가 차 있으면(free 3 GB 아래) 한 번 비우고 띄운다 (`sync; echo 3 | sudo tee /proc/sys/vm/drop_caches`):

```bash
cd ~/decider_test && LLAMA_DECISION_IMAGES_LAST=1 nohup ./llama.cpp/build/bin/llama-server -m models/d1-3B-Q8_0.gguf --mmproj models/mmproj-d1-3B-Q8_0.gguf \
  --host 127.0.0.1 --port 8171 -ngl 99 -c 4096 -np 1 -lm dio > server_dec.log 2>&1 &
# 뜰 때까지 (모델 읽기 약 1 분): grep -m1 "listening" server_dec.log
ros2 launch limo_fire_rl explore_fuel_rl.launch.xml upper:=decider [dry_run:=false] [rviz:=true]
```

서버는 `train/explore/upper_lora/patches/` 의 두 패치(판단 요청 checkpoint · 그림을 글 뒤로)를 넣은 빌드에 `LLAMA_DECISION_IMAGES_LAST=1` 로 띄운다 — 결정마다 같은 앞부분(템플릿 · 설명 글 · 힌트)을 다시 계산하지 않음. 미세조정 v1(LoRA)도 이 차례로 학습한다.
요청 꼴 기본은 짧은 꼴(`decider_compact: true`, 지도 288 px, 결정 기록 4, 카메라 없음 `decider_cam: 0`): 로봇 Orin 에서 요청 하나 약 1.6 s(앞 요청 뒤) · 2.2 s(처음) — 예전 기본(긴 꼴 + 카메라 2 × 2)은 5.0–5.6 s (`upper_lora/README.md` "요청 꼴과 지연").
기본은 사건 때만 묻고(`decider_mode: event`) 요청에 후보를 6 개까지 넣는다(`decider_max_cands: 6`) — 시뮬 위층 비교(`train/explore/upper_lora/README.md` "위층 비교")에서 가장 빨랐고, 라벨 굴리기(`dec_rollout`)·`low_explore_check --upper decider` 기본도 같은 값.
상태는 `~/status` 끝에 "판단 모델: 보냄 · 답 (마지막 ms) · 시간 끝 · 실패 · 후보 안 맞음 · slot0 대신". 서버가 없거나 답이 늦으면 slot0 차례로 계속 돈다 (멈추지 않음).
카메라 그림(시뮬 `--decider-input all`)은 시뮬 장면에 광선을 쏴 칠한 그림이라 실물에는 같은 것이 없어 쓰지 않는다 (`decider_input` text · image · both).
