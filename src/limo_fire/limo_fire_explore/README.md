# limo_fire_explore — 탐사 노드 (합친 탐사)

탐사 노드는 하나다: `explorer`(`/explore/explorer`, `app.launch.xml`). 위층이 고른 목표까지 아래층 A* 경로 추종이 바로 `/cmd_vel` 을 낸다 — Nav2 는 쓰지 않는다.
`/explore/enable`(std_srvs/SetBool)로 켜고 끄며, 운용 화면 RViz 의 **탐사 패널**
("Auto explore ON/OFF" 버튼, [limo_fire_rviz](../limo_fire_rviz/README.md))이 이 서비스를 부른다. 실물에서 RViz 를 켜는 법은
[limo_fire_bringup README "실물에서 운용 화면(RViz) 켜기"](../limo_fire_bringup/README.md#실물에서-운용-화면rviz-켜기).

| 노드 | 무엇 | 켜기 |
|---|---|---|
| `explorer` | 위층 판단 모델 + 아래층 A* 경로 추종 (아래 "합친 탐사") — 시뮬과 같은 코드 | 멈춘 채 뜸 → 버튼 (리모컨 오토일 때만) |

탐사 강화학습(RL 정책 노드 `explore_rl`, 아래층 RL 정책 `lower:=rl`)은 2026-11-16 에 그만두고 [archive/rl](../../../archive/rl/README.md) 로 옮겼다
(실물에서 시뮬과의 격차가 컸고, A* 경로 추종 아래층이 훨씬 빠르고 잘 됐다). 그때의 이 README 는 `archive/rl/src/limo_fire/limo_fire_explore/README.md`.
패키지·노드 이름(`limo_fire_explore`, `explorer`)은 launch·녹화·RViz 설정이 쓰고 있어 그대로 둔다.

옛 구성: Nav2 `NavigateToPose` 로 프론티어 목표를 보내던 노드(`/explore/frontier_explorer`)는 2026-11-20 에 지웠다 — 탐사가 두 벌이 되지 않게.
그 코드·확인 기록(시뮬 2026-11-11, 실물 2026-11-18: Nav2 명령 0.26 m/s 에 바퀴 0.118 m/s, 라이다 한 평면이라 의자 좌판을 못 봄)은 커밋 `ae62588` 의 이 README.
시뮬 견줌 기준선(프론티어)은 `train/explore/baselines/frontier` 에 그대로 있다 — 로봇에서는 안 쓴다.

## 주행 목표 서버 (`/explore/navigate_to_pose`, 2026-11-20)

탐사 노드가 Nav2 와 같은 **`nav2_msgs/NavigateToPose` 액션**을 낸다. 목표 하나를 받으면 위층(탐사)은 쉬고 아래층
(A* + 지역 계획기 + 깊이 카메라 층 + 닿음 감지 + 막힘 복구 — 탐사와 같은 코드)이 그 자리까지 간다. 순찰을 이것으로 몰면 회피·경로 계획이 탐사와 한 벌이 된다.

```bash
# 실물: 탐사 노드는 늘 떠 있음 — 순찰(임무 BT 순찰 · click_patrol)만 이 서버로
ros2 launch limo_fire_bringup real.launch.xml bt:=true nav_action:=/explore/navigate_to_pose
# 시뮬
ros2 launch limo_fire_bringup sim.launch.xml world:=limo_fire_obstacles bt:=true explore:=true nav_action:=/explore/navigate_to_pose
# 손으로 한 점
ros2 action send_goal /explore/navigate_to_pose nav2_msgs/action/NavigateToPose "{pose: {header: {frame_id: map}, pose: {position: {x: 1.0, y: 0.5}, orientation: {w: 1.0}}}}"
```

- **기본은 Nav2 그대로** (`nav_action:=/navigate_to_pose`). 불 대응·연기 구역 훑기는 늘 Nav2 — 연기 모드가 Nav2 장애물 지도·속도 한도에 기댄다.
- 받는 것: 목표 틀은 `map` 만 (아니면 거절). dry run · 리모컨 오토 아님(require_rc) 이면 거절. 다른 목표가 오면 앞 목표는 못 감(abort)으로 끝내고 새 목표.
- 상태 `/explore/state` 가 `navigating` (끝나면 `idle`) — 임무 BT 는 `running` 만 "밖에서 탐사를 켬" 으로 보므로 순찰이 꺼지지 않는다.
- 끝: 목표 0.25 m 안 + 목표 방향 0.25 rad (`nav_arrive_m` · `nav_yaw_tol`) → 성공. 못 감 → 15 s 동안 0.5 m 못 움직임(`nav_progress_*`, Nav2 진척 검사 꼴) ·
  경로 없음 5 s · 아래층 막힘 복구(다시 계획 → 물러나기)를 다 쓰고 포기 · 닿음 · 리모컨 · 끔. 탐사 버튼을 누르면 목표는 못 감으로 끝나고 탐사가 켜짐.
- 저장 지도: `/map`(map_server 또는 Cartographer)을 로봇 지도의 **아직 모르는 칸에만** 채운 뒤 계획 — 본 칸은 라이다 값이 먼저. 지도 창(25.6 m, 판 시작 자리 가운데) 밖 목표는 거절.
- 초음파(`/ultrasonic/front_left|front_right`): 1.0 m 안 메아리를 깊이 열 점(지역 계획기 거리 장·충돌 감시가 같이 쓰는 점)에 덧붙임. 원뿔(15°)이 넓어
  메아리 자리를 축에 찍으면 옆 틈(문)을 가짜로 막으므로 **라이다가 원뿔 안 그 거리 앞에서 이미 본 메아리는 뺀다**(`sonar_skip_lidar_seen`) — 라이다 높이 아래 물체·유리만 초음파 점이 됨.
- 막힘 처리 (2 차, 2026-11-20): 아래층 결정이 끝나 쉬면 같은 목표로 다시 짬(1 s 에 한 번 — 예전에는 목표를 한 걸음 비워 그 걸음에 제자리 회전이 나와 맴돎),
  경로 없음 · 충돌 검사 막힘 · 회전 막힘이 1 s 이어지면 물러나기(목표 하나에 3 번, `nav_nopath_backup_s` · `nav_backups` — 가구 모서리·문틀에 붙어 끼임),
  깊이 층 걷어내기: 저장 지도 벽 둘레 0.15 m 안(`nav_wall_ring_m` — 자세 흔들림으로 벽 깊이 표시가 문틈으로 번져 0.70 m 문이 약 0.45 m 로 보였음),
  바닥이 보인 칸(`nav_floor_tol` 0.02 m · `nav_floor_range` 2.5 m — 깊이 층은 띠 안 점으로만 지워서 지나간 사람 자취가 열린 바닥에 남아 길을 막았음).
  못 간 자리 둘레 지도 그림: `nav_dump_dir`. 계획 반경은 env.yaml `plan_margin` 0 (0.293 → 0.222 m, 탐사와 같은 값).
- 시뮬(Webots) 덮어쓰기 `config/explorer_sim.yaml`: 깊이 카메라 자리(Webots 모델은 실물보다 낮고 수평 — 실측 TF), 깊이 32FC1 도 받음.
- 시뮬 확인 (2026-11-20, `limo_fire_obstacles` — 문 0.70 m + 문 앞 상자 · 낮은 상자 · 막다른 칸 · 구석 · 뜬 판 · 걷는 사람, 3 판씩, 보고서 `limo_upper_keep/reports/obstacle_nav.md`):
  | | 도착 | 형상 닿음 | 사람 닿음 | 10 s 갇힘 (그중 구석) | 맴돎 | 문 앞 상자 머뭇거림 |
  |---|---|---|---|---|---|---|
  | Nav2 (기본, RPP) | 22/27 | 1 (낮은 상자) | 4 | 7 (4) | 0 | 34.6 · 204.8 · 5.0 s |
  | 이 서버 1 차 (c26a208) | 11/27 | 0 | 0 | 2 (0) | 2 | 38.3 · 26.9 · 47.1 s |
  | 이 서버 2 차 (계획 반경 0.222 · 깊이 층 걷어내기 · 끼임 물러나기) | 18/27 | 0 | 8 (한 판에서 7) | 1 (0) | 1 | 3.4 · 20.4 · 5.4 s |

  2 차는 판마다 2/9 · 7/9 · 9/9 (2/9 판은 멈춰 선 로봇에 걷는 사람이 걸어 들어와 닿음 감지로 멈춘 뒤 다음 목표들이 곧바로 닿음으로 끝남 — 시뮬 사람은 안 비킴).
  막다른 칸·구석은 두 판에서 모두 들어갔다 나옴. 문 앞 0.60 m 대각 틈은 3 판 중 2 판 지남. **Nav2(22/27)에 못 미쳐 기본값은 Nav2 그대로**.

## 합친 탐사 (`explorer` — 위층 판단 모델 + 아래층 A*, 2026-11-16)

> **확정 구성 (2026-11-16 실물 시험):** 위층 판단 모델 + 아래층 A* 경로 추종(규칙 추종기 — 유일한 아래층). 노드에는 이 위층 하나만 있다 —
> 기준선 위층(cand · fuel · slot0)은 2026-11-18 노드에서 빼고 시뮬 도구(`low_explore_check --upper …` · `combine_run`)에만 남겼다.
> 같은 날 실물에서 fuel+RL · 가까운 후보+RL · 판단 모델+RL · 판단 모델+A* 를 돌려 판단 모델+A* 가 가장 잘 됐다(녹화 `~/limo_bags/1116_*`).
> 실행: ① 로봇 page cache 비우기(`sync && echo 3 | sudo -n tee /proc/sys/vm/drop_caches` — Orin GPU 메모리 확보) → ② 아래 "판단 모델" 절의 llama-server →
> ③ `robot.launch.xml` → ④ `ros2 launch limo_fire_explore app.launch.xml stack:=true rviz:=true dry_run:=false [perception:=true]` → 리모컨 SWB 위 + 패널 버튼. 멈춤은 SWB 내림.

**한 프로세스 · 한 스레드 (2026-11-18)**: `app.launch.xml` 이 우리 노드 전부(`/explore/explorer` + `perception:=true` 면 `/perception/object_detector` · `/perception/object_map`)를
컨테이너 `/app` 하나(`component_container` — 실행기 스레드 하나)에 올린다. 그 스레드가 차례로: 영상을 GPU 에 걸고(기다리지 않음) → 끝난 탐지를 물체 지도에 → `/scan` 한 장마다 걸음 하나
(`on_scan_tf`: 감각 `sense` → 깊이 `read_depth` → 위층 `decide` → A* 추종 → `publish_cmd`). 오래 걸리는 것(GPU · 판단 모델 HTTP · TF)은 모두 묻기만 하고 기다리지 않는다.
프로세스·스레드·토픽 표는 [docs/runtime_io.md](../../../docs/runtime_io.md).

시뮬(`low_explore_check` · `combine_run`)과 **같은 코드·같은 값**으로 로봇이 탐사한다. 위층이 로봇이 쌓은 지도에서 목표를 고르고(결정 차례 1 s — 판단 모델은 그중 사건 때만 묻는다), 아래층 규칙 추종기(로봇 지도 A* `plan_ref.h` + 경로 추종 `path_step_*` — `RobotEnv::low_act` 계층형 가지)가 0.1 s 마다 `/cmd_vel` 을 낸다.
Nav2 는 쓰지 않는다. Cartographer 는 지도 자세(TF `map → base_link`)만 준다. 관측 설정은 `train/explore/sim/config/env.yaml` 그대로.

| 차례 | 실물 노드 | 시뮬 (`train/explore/sim/tools/low_explore_check` · `scripts/combine_run`) |
|---|---|---|
| 설정 | `RobotEnv` 합친 탐사 모드 — 목표는 위층, 탐사 판 끝 규칙, 막힘은 신호만 | `low_rl 1` · `low_explore 1` |
| 결정 때 | 걸음 % 10 == 0 또는 다시 결정 | 같음 |
| 위층 판단 모델 | 판단 모델(d1-3B, zero-shot)이 `scripts/cand_view.h` 후보 앞 6 개 중 고름 — `scripts/low_decider.h` (요청: 지도 사진 + 후보 글). 사건 때만(`decider_mode: event` — 목표에 1 m 다가감 · 목표 실패 · 후보 크게 바뀜 · 10 s 최대 간격) 비동기로 묻고, 기다리는 동안 지금 목표로 몰며, 답이 오면(그 후보가 아직 있으면) 바꿈. 목표가 죽으면(판 처음·켠 직후 포함) 바로 묻고 답을 (걸음으로 잰 답 지연 이동 평균 + `decider_hold_extra` 0.5 s)까지 기다림 — 그 사이 아래층은 지금 명령 그대로(목표 없으면 제자리), 그래도 안 오면 규칙(cand_pick)이 대신. 남은 길이 답 지연 + `decider_ask_lead_s` 안에 끝날 듯하면 미리 물음. 지시(`/explore/instruction`)가 바뀌면 다음 결정 차례에 바로 물음("new instruction"), 옛 지시 요청의 답은 버림. 둘 다 −1 이면 옛 동작(요청 바이트 같음). `decider_timeout`(8 s) 넘은 답은 버림. 정해진 주기로 묻기(periodic)는 시뮬 비교에서 덮음이 같아 노드에서 뺐다(시뮬 `--dec-mode periodic` 은 남음) | `low_explore_check --upper decider` |
| 대신 고르는 규칙 | 답을 기다리다 시간이 넘거나 답이 잇따라 틀리면 `cand_view.h` 0 번 후보, 지금 목표 번호가 있으면 붙듦(`Ctl::cand_pick`). `decider: false`(`app.launch.xml decider:=false` — 판단 모델 서버를 안 띄울 때)면 늘 이 규칙: 결정 차례마다 `Ctl::decide`, 판단 모델 요청·입력 지도를 안 만들고 지시(`/explore/goal`)를 안 받음 | `low_explore_check --upper cand` |
| 아래층 | 규칙 추종기: 고른 후보의 서는 자리까지 A* 경로를 짜 따라감, 막히면 복구(물러나기·목표 포기 — `hier_path.h`) | 같은 함수 |
| 목표 없음 | 규칙 추종기의 제자리 회전 | 같음 |
| 다시 결정 | 통과 · 경로 끝 밖 · 경로 없음 · 막힘 신호 → 경로 끝 밖·없음·막힘은 그 칸 0.5 m 를 10 s 피함 | 같음 |
| 끝 | 지도 닫힘 + 지도 진척 없음 10 s (성공) · 진척 없음 · 시간 → 명령 0, `/explore/done` true | `finish_step` |

위층 결정·목표 넘기기는 `train/explore/scripts/low_explore_ctl.h` 하나를 노드와 확인 도구가 같이 쓴다.
launch 인자는 `dry_run` · `record` · `decider`(판단 모델 씀 · false = 규칙만) 등뿐이고, 나머지 값(`camera_layer` · `contact_resume_s` · `decider_max_cands` …)은
`config/explorer.yaml` 에서 바꾼다. 2026-11-17 에 실물에서 안 쓰던 값(`decider_mode`·`decider_period`·`decider_input`·`decider_prompt`·`decider_cand`(옛 hier 8 칸 후보)·
`decider_cam`·`cam_base_h`·`color_topic`(카메라 키프레임 기억)·`close_lidar`·`slot_vp`·`obs_all`·`frontier_cache`·`frontier_check_every`)과 `~/goto`·`~/goto_face` 를,
2026-11-18 에 기준선 위층(`upper`·`fuel_*`)을 지웠다 —
시뮬 비교·점검 쪽 옵션은 `low_explore_check` 에 남아 있다.

**멈춤 조건 (명령을 0 으로)**

| 조건 | 동작 |
|---|---|
| 처음 · `/explore/enable false` | 명령 0 (`idle`). `/explore/enable true` 로 켬 — **리모컨이 오토일 때만 켜진다** |
| 리모컨이 오토가 아님 (`/rc_state` `buttons[1]` ≠ 2 — SWB 위가 2) 또는 `/rc_state` 0.5 s 넘게 안 옴 | 명령 0 · 꺼짐 (`paused`). **다시 오토로 바꿔도 저절로 안 감 — `/explore/enable true` 를 다시** |
| 라이다·TF·오도메트리 0.3 s 넘게 안 옴 | 명령 0, 돌아오면 이어 감 |
| 닿음 (`train/explore/sim/include/contact.h` `sense_step` — 라이다·깊이·멈춤·IMU 수평 가속 > 15 m/s², 마스크 71 · 멈춤 2 걸음 — 그만둔 RL 학습 low_w3 의 값) | 명령 0 · 꺼짐. `contact_resume_s: 1.0` 이면 1 s 뒤 저절로 켬 (기본 0 = `/explore/enable` 따로) |
| 끝 (위 표) | 명령 0 · `done` |

- `dry_run:=true`(기본): `/cmd_vel` 발행자를 아예 안 만들고 같은 명령을 `/explore/cmd_debug`(TwistStamped)에만. dry run 에서는 닿음의 "멈춤" 신호(명령 대비 바퀴 정지)를 뺀다 — 명령이 안 나가 늘 참.
- `max_speed`: 기본 = 시뮬 한도(자르지 않음). 토픽: `/explore/state`(idle · running · paused · done) · `/explore/status`(한 줄 — 까닭·리모컨·목표·명령·시간) · `/explore/done` · `/explore/goal`(지시 글) · `/explore/goal_status`.
  `debug:=true`(`app.launch.xml` 은 `rviz:=true` 면 켬)일 때만 RViz 표시 토픽 `/explore/goal_marker` · `/explore/candidates` · `/explore/candidates_text` · `/explore/decider_image`, `cmd_debug` 는 dry run 이거나 debug 일 때.
  `/explore/decider_image`(판단 모델에 보낸 지도 그림)는 `decider: true` 일 때만 만들고, 서버가 한 번이라도 답한 뒤부터 낸다(서버가 없으면 빈 회색 판도 안 냄).
  운용 화면의 "판단 모델 입력 지도" 표시는 기본 꺼짐 — 판단 모델을 쓸 때 RViz Displays 에서 켠다.
- `/explore/candidates` (MarkerArray, map 틀, 결정마다): 새 후보(`train/explore/scripts/cand_view.h`)마다 구(서는 자리)·글(판단 모델 글자 · 안정 번호 · 예상 넓이/시간)·덩어리 대표 칸까지 선·덩어리 프론티어 칸, 색 초록 보통 · 파랑 지금 목표 · 주황 실패 기억 · 회색 쉬는 덩어리, 로봇 자취(60 s), 판단 모델 답 수·마지막 답 시간. 운용 화면 `limo_fire.rviz` 의 Candidates 표시.
- RViz 탐사 패널: 고르기에서 "합친 탐사 (/explore)". `paused` 면 상세 줄에 까닭.
- 리모컨 칸 확인: 녹화 `1103_M_drive3`(리모컨으로 몬 것)의 `/rc_state` buttons 는 모두 `[3, 1, 3, 3]`, `1104_M_cmd_mix`(ROS 명령으로 몬 것)는 모두 `[3, 2, 3, 3]`.

| 확인 (이 PC, 2026-11-16) | 결과 |
|---|---|
| `train/explore/sim/tools/low_explore_check` — CPU 시뮬과 `RobotEnv` 가 각자 위층 결정, 시뮬 라이다 장·감각·깊이를 `RobotEnv` 에 넣고 걸음마다 비교 | bookstore · small-house, 씨앗 1~3, 사람 0 · 2, 800~2500 걸음: 위층 명령 · 준비 뒤 경로(목표 칸 · 경로 점) · 명령 · 판 끝 **비트 같음** (목표 없는 걸음 포함). 위층 slot0 · fuel · cand · 판단 모델(가짜 응답기) 모두. `--node-obs` 면 실물 쪽을 노드와 같은 관측 설정으로 |
| `low_explore_check … --upper decider [--dec-latency 2.6] [--dec-mode event] [--dec-max 6] [--dec-match 1.5] [--dec-url http://…]` — 판단 모델 위층(`scripts/low_decider.h`)을 가짜 응답기(본문 해시로 글자, 지연 뒤 걸음에 답)로 두 쪽이 같이; `--dec-url` 이면 진짜 llama-server 답을 지연 걸음 뒤에 (시뮬로 평가) | bookstore · small-house, 씨앗 1~3, 사람 0 · 2, 지연 0 · 2.6 · 7.5 s, 1200~2500 걸음: 요청 본문 · 위층 명령 · 아래층 관측 · 명령 · 판 끝 **비트 같음**. 답의 후보 칸 맞춤 반경 0.5 m 면 답의 40 ~ 55 % 를 버려 기본 1.5 m (버림 10 ~ 25 %) |
| `train/explore/sim/tools/low_explore_replay` — 노드 녹화(`record:=`)를 `RobotEnv` + 위층만으로 다시 돌려 결정마다 까닭(주기·통과·경로 없음·막힘)·목표·뜀·목표 없음 까닭·닫힘을 적음, 지도 그림(`--pgm`) | 그때 설정(`--fuel-hyst 0 --fuel-vp-min 0 --fuel-vp-clear 0`)으로 `1116_fuel_rl_1` 659 / 659 · `1116_slot0_1`(`--upper slot0`) 86 / 86 관측 비트 같음 (531e7fc 위). 녹화 꼴은 판 4(`sim/tools/low_node_rec.h`), 옛 판 1~3 도 읽지만 그 뒤 위층·깊이 층이 바뀌어 1116 녹화의 명령은 지금 코드와 다름 |
| 녹화 `1104_M_cmd_mix` (깊이 정렬) 에 `explore.launch.xml` dry run (`map_frame:=odom`, `use_sim_time:=true`, 그때 위층 fuel + 아래층 RL) | 발행자에 `/cmd_vel` 없음. 켬 뒤 결정 141 · 통과 24. x86 한 코어: 걸음 7.3 · 관측 2.1 · 위층 평균 12 ms (최대 22) — 그 뒤(2026-11-17, 한 스레드 · 안 쓰는 관측 끔) `1116_fuel_rl_1` dry run 걸음 5.53 · 위층 8.9 ms |
| 리모컨 (녹화 대신 손으로 낸 `/rc_state`) | 수동(1) → paused · 다시 오토(2) → 그대로 paused · `/explore/enable` → running · 안 옴 → paused · 안 오는데 `/explore/enable` → 거절 |

젯슨 Orin Nano (2026-11-17, 한 스레드): 걸음 약 5.9 ms · 위층 결정 약 9.5 ms — 0.1 s 걸음 안에 넉넉히 듦.

### 실행 (로봇)

`WS` = 이 저장소를 받은 워크스페이스 (예: `export WS=$HOME/limo_fire_ws`). 판단 모델 서버는 아래 "위층 판단 모델 띄우기" 로 먼저 띄운다.

```bash
# 빌드 (한 번)
cd $WS && git pull --rebase --autostash && source env.sh
colcon build --packages-select limo_fire_explore limo_fire_navigation limo_fire_perception limo_fire_rviz && source install/local_setup.bash

# 터미널 1 — 센서·섀시·EKF (대시보드 limo_base 가 떠 있으면 켜지 않는다 — 섀시 포트는 하나만)
ros2 launch limo_fire_bringup robot.launch.xml

# 터미널 2 — 1) dry run 먼저: Cartographer + /app(explorer [+ 인식]) + RViz, /cmd_vel 안 냄
ros2 launch limo_fire_explore app.launch.xml stack:=true rviz:=true dry_run:=true [perception:=true]
#   RViz 탐사 패널 "합친 탐사 (/explore)" → Auto explore ON. 확인할 것:
ros2 topic echo /explore/status          # running · 리모컨 오토 · 목표 칸 · 명령 · 시간
ros2 topic echo /explore/cmd_debug       # 낼 명령 (v ≤ 1.0 m/s, ω ≤ 3.14 rad/s)
#   리모컨 SWB 를 내렸다 올림 → /explore/state paused 로 바뀌고, 올려도 paused 그대로인지 (다시 버튼을 눌러야 running)

# 2) 점검표를 다 확인한 뒤에만 — 실제 주행
ros2 launch limo_fire_explore app.launch.xml stack:=true rviz:=true dry_run:=false [perception:=true]
ros2 service call /explore/enable std_srvs/srv/SetBool "{data: true}"    # 버튼과 같음
ros2 topic echo /explore/done            # true 면 끝 (지도 닫힘)
```

**멈추는 법** (어느 것이든): 리모컨 SWB 를 내림(오토 해제 — 섀시가 ROS 명령을 안 받고 노드도 꺼짐) · RViz 버튼 OFF ·
`ros2 service call /explore/enable std_srvs/srv/SetBool "{data: false}"` · 터미널 2 Ctrl+C (마지막 명령이 남지 않게 SWB 를 먼저 내린다).

**실제 주행 전 점검표**
- [ ] 충전기를 뺐다 (꽂힌 채 움직이지 않는다)
- [ ] 주행할 곳이 비었다 — 사람·깨질 것 치움, 지도가 닫히는 공간(문 닫음)
- [ ] 사람이 로봇 곁에서 리모컨을 들고 있다 (SWB 를 바로 내릴 수 있게)
- [ ] 리모컨이 켜졌고 오토 모드 (SWB 위) — `ros2 topic echo /rc_state --field buttons` 의 둘째 값이 2
- [ ] dry run 에서 `/explore/status` 가 running, 목표·`/explore/cmd_debug` 가 말이 되고, SWB 내림 → paused 를 봤다
- [ ] 다른 주행 노드(Nav2 목표·click_patrol·임무 BT·teleop)가 `/cmd_vel` 을 안 낸다 — `ros2 topic info /cmd_vel` 의 발행자가 없거나 explorer 뿐
- [ ] 젯슨 시간: `/explore/status` 의 평균 걸음 + 관측 + 위층이 100 ms 안 (넘으면 멈추고 알림)

### 탐사 중 "○○ 로 가" (물체 지도 · `perception:=true`)

`app.launch.xml perception:=true` 면 YOLO 분할 + 물체 지도(limo_fire_perception)가 같은 프로세스 `/app` 에 같이 뜬다. 지시는 explorer 가 직접 받는다.

```bash
ros2 topic pub -1 /explore/goal std_msgs/msg/String "{data: '앉을 수 있는 곳으로 가'}"   # 아무 글 (한글 · 영어). 빈 글 = 지시 지움
ros2 topic echo /explore/goal_status        # 푼 클래스 · 고른 물체 · 가는 중 · 도착
```

- 판단 모델이 물체 후보를 고르면 `explorer` 이 그 설 자리로 같은 A* 아래층으로 가고, `goto_arrive_m`(0.4 m) 안에 들면 물체 쪽으로 제자리에서 돌아(남은 각 < `goto_face_tol` 0.15 rad 또는 5 s)
  "도착" 을 알리고 멈춘다 (기본 `goto_resume: false`, true 면 탐사를 이어 감). 가는 동안은 탐사 끝 판정을 하지 않는다.
- RViz: `/explore/candidates_text` 끝 줄(지시 · 마지막 진행) · `/explore/candidates` 의 물체 후보 글자.
- **지시 클래스도 판단 모델이 고름** (`include/limo_fire_explore/class_ask.h`): 지시를 받으면 explorer 가 판단 모델(`decider_url`)에 choice 질문을 한 번 보낸다 —
  "이 지시가 가리키는 탐지기 클래스는?" 선택지 = 인식 설정의 `class_names` 전체(`class_files`, app.launch.xml 은 `perception_config` — 지금 물체 지도에 있는 것만이 아님) + `none`.
  답을 기다리는 동안은 지시 없는 탐사 그대로, 답이 오면(none · 실패 · `decider_timeout` 이면 클래스 없음) 그 클래스를 힌트(`class_hint`)로, 지시 글을 위층에 넘긴다. 진행은 `/explore/goal_status`.
- **판단 모델이 고름**: 무엇을 할지는 판단 모델이 지시 글을 그대로 보고 고른다. 지시가 있으면 요청 후보 목록 앞 `decider_obj_max`(3) 칸에
  물체 지도의 **힌트 클래스** 물체 설 자리가 물체 후보(`kind: object` — `object` 클래스 · `oid` 지도 번호 · `len` 길 m · `bear` · `dist` 직선 m · `seen_s` · `vis` 지금 보임)로 들어가
  프론티어 후보와 섞인다. 칸보다 많으면 길 거리 순으로 고르게 벌려(가장 가까운 것·가장 먼 것 늘 포함 — "가장 먼 의자" 같은 지시).
  힌트 클래스가 아닌 물체로는 칸을 채우지 않는다(`decider_obj_fill: false` — 판단 모델이 첫 물체 글자로 쏠려 엉뚱한 물체로 가지 않게). 클래스가 없으면 물체 칸은 비고 프론티어만.
  글에는 지시 원문(`instruction`) · 힌트(`class_hint`) · 아는 물체 10 개(`known_objects`: "chair-3 chair 4.1 m front-left seen 2 s visible") · 후보 종류 설명(`candidate_kinds`)이 들어가고
  질문은 "지시를 가장 잘 따르는 후보 (탐사 지시거나 후보에 없으면 가장 빨리 지도를 넓히는 곳)". 지시가 바뀌면 하던 물체로 가기를 그만두고 바로 다시 묻는다.
  설 자리는 이번 결정의 후보 뽑기 BFS(계획 몸통 반경 · 깊이 층 — 프론티어 후보와 같은 잣대)가 닿은 칸 중 물체에 가까운 0.6~1.2 m 고리부터. 오래(600 s) 안 본 물체는 뺀다.
  지도 그림에서 물체 후보는 하늘색 네모, 프론티어 후보는 노란 점. 판단 모델이 물체 후보를 고르면 그 설 자리로 가서 물체를 바라보고 멈춘다. RViz /explore/candidates 에 물체 후보 글자(ns cand_obj).
  지시가 없으면 요청 본문은 예전과 바이트까지 같다(`decider_obj_always: false` 기본 — 켜면 지시 없어도 물체 후보). 확인: `low_explore_check --dec-instr 글 --dec-class-hint 클래스 --dec-obj 클래스,x,y`
  (·`--dec-obj-always`, 클래스 묻기까지는 `--class-file 인식설정.yaml [--cls-fake 답 | --dec-url …]`) 로 시뮬 == 실물 비트, 지시 없을 때 요청(DEC_BODY_LOG 걸음·본문 해시) 앞 커밋과 같음.
  지시가 있으면 프론티어 후보는 `decider_front_max_instr`(3) 개까지 (이름 물체 ≤ 3 + 프론티어 ≤ 3).
- **사람 따라가기** (`scripts/obj_follow.h` — 노드와 시뮬이 같은 코드): 지시가 있을 때 지금 보이는(`decider_vis_s`) `follow_classes`(person) 물체는 `kind: follow` 후보로도 들어간다.
  판단 모델이 고르면 노드가 목표를 쥔다: 같은 사람을 지도 번호로, 없으면 예측 자리 둘레 문(`follow_gate` + 못 본 시간 × `follow_gate_v`, 3 m 까지) 안 가장 가까운 사람으로 잡고,
  사람에서 `follow_dist`(1.0 m) 떨어진 로봇 쪽 자리(둘레 `follow_clear` 빈 칸, `follow_stop` 이상)를 `follow_replan_s` 마다 다시 세되 `follow_hyst` 안이면 그대로(떨림 없음),
  아래층은 같은 A* 추종기에 속도 한도 `follow_max_speed`(0.6 m/s — `LowCmd::vmax`). 가까우면(≤ dist + hyst, `follow_stop` 안) 서서 사람 쪽으로 돈다(`LowCmd::face`).
  `follow_lost_s` 넘게 못 보면 마지막 목표까지 간 뒤 사람이 가던 쪽으로 `follow_search_w` 로 돌며 찾고, `follow_search_s` 넘으면 멈추고 "놓침" 을 알린다.
  사람이 `follow_reask_m` 넘게 옮기면 판단 모델에 다시 묻고(그 사이 따라가기는 계속), 답이 탐사 자리면 따라가기를 끝낸다. 새 지시·지시 지움도 끝. RViz /explore/candidates ns follow (주황 원기둥 + 모드·거리).
  시뮬 확인: `low_explore_check --upper decider --people 3 --follow --dec-fake follow --dec-instr "사람 따라가"` (시뮬 사람을 카메라 화각 66.4°·5 m·지도 가림으로 본 것만 사람 목록에) —
  두 쪽 따라가기 명령·경로 비트 같음, 따라간 거리·모드·놓침·사람 닿음을 `FOLLOW` 줄로.

### 실패 상황 표 (판단 모델 · 지시 · 따라가기)

정상 길만이 아니라 실패를 먼저 보고 짰다. 지시 일 감독은 `scripts/task_sup.h`, 따라가기는 `scripts/obj_follow.h`, 판단 모델 답 처리는 `scripts/low_decider.h` — 노드와 시뮬이 같은 코드.
시뮬 확인은 `low_explore_check --upper decider --node-obs --dec-instr "의자로 가" --dec-class-hint chair --dec-obj … --goto` 에 아래 주입을 더해 bookstore · small-house × 씨앗 1 · 2 (1200 걸음),
모든 판 시뮬 == 실물 비트 (2026-11-17, 결과는 GOTO · DECSRC · FOLLOW 줄).

| 상황 | 하는 일 | 시뮬 주입 · 결과 |
|---|---|---|
| 판단 모델 서버 꺼짐 | 답 실패가 `fail_max`(3) 번 잇따르면 기다리지 않고 규칙이 바로 대신 — 지시 · 힌트 클래스 물체에 갈 수 있으면 가장 가까운 그 물체로, 아니면 탐사 규칙. 요청은 사건 때 계속 보내 살아나면 다시 판단 모델로. /explore/status 에 실패 수 | `--dec-fake down`: 4 판 모두 규칙이 지시 물체로 가 닿음, 지시 없을 때도 탐사 이어 감(멈춤 없음) |
| 답이 느림 (8 s 넘음) | 늦은 답은 버리고 같은 길 | `--dec-latency 9`: 4 판 중 3 판 닿음, 1 판은 못 감 1 번 뒤 닿음 |
| 답 꼴이 틀림 · 없는 글자 | 바로 다시 물음(잇따라 3 번까지), 그 뒤 위와 같이 규칙 | `--dec-fake garbled` · `oor`: 8 판 중 6 판 닿음, 2 판(bookstore 씨앗 1 — 그 장면에서 갈 수 없는 의자 자리)은 못 감 3 번 뒤 포기 |
| 답의 후보가 사라짐 · 옮김 | 옛 자리로 안 가고 다시 물음 ("chosen candidate is gone" · 물체는 지도에서 빠지거나 1 m 넘게 옮겼으면 "chosen object is gone or moved") | `--obj-move 15,3,0`: 4 판 모두 다시 물은 뒤 닿음 |
| 설 자리에 못 감 (막힘 · 경로 없음) | `goto_stall_s` 동안 `goto_stall_m` 더 못 다가가거나 경로 없음 `goto_nopath_s` → 못 감: 뒤로 물러나지 않고 그 물체를 `goto_fail_ttl_s` 빼고 다시 물음, 한 지시에서 `goto_max_fail` 번이면 멈추고 "못 감" | `--goto-stall-s 1`: 3 판 못 감 3 번 → 포기, 1 판은 닿음 |
| 지시한 물체가 끝내 없음 | 탐사하며 찾고 `instruction_limit_s` 넘으면 멈추고 "못 찾음". 다 탐사("대충 다 앎")했는데 지시가 남으면 판을 끝내지 않고 물체 후보로 다시 묻고, 맞는 물체도 없으면 "못 찾음" | `--instr-limit 30 --dec-obj chair,60,60`: 4 판 모두 30 s 에 포기. 다 탐사한 경우는 노드에만 (시뮬 판 끝은 시뮬 규칙) |
| 지시 바뀜 · 지움 | 따라가기 · 물체로 가기를 바로 끝내고 새 지시로 바로 물음 (옛 지시 요청의 답은 버림) | `--instr-at 12:탐사해` · `--instr-at 12:`: 8 판 비트 같음 |
| 사람을 놓침 · 가림 · 여러 명 | 같은 사람만(지도 번호 → 예측 자리 문), `follow_lost_s` 넘으면 마지막 자리 앞까지 간 뒤 천천히 돌며 찾고 `follow_search_s` 넘으면 멈추고 "놓침" | `--people 3 --follow --dec-fake follow`: 따라간 판에서 최소 거리 0.98 m, 따라가는 중 사람 닿음 0, 놓쳐 포기 1 번씩(2 판) |
| 리모컨 상태가 안 옴 | best effort 구독(RELIABLE 발행자와 맞음), 나이는 steady 시계, 발행자가 있는데 `rc_resubscribe_s` 못 받으면 구독을 새로 만듦, 켜기 거절 · /explore/status 에 나이 · 받은 수 · 발행자 수 | `colcon test --packages-select limo_fire_explore` (test_rc_watch: 다시 구독 판단 · RELIABLE/BEST_EFFORT 발행자 · 받는 노드 5 번 다시 만들기) |

### 깊이 카메라 층을 볼 것

깊이 카메라 장애물 층(env.yaml `camera_obstacles`, 걸음 0.1 s 마다 갱신 — `update_every_step`)은 후보·닫힘에서 막힌 곳을 뺀다. 층 전용 촘촘한 깊이 격자(`layer_grid`)는 시뮬에서 끝 무렵 길이 더 막혀 기본 끔. 실물 깊이가 시뮬 모델과 다르면(정렬 꺼짐·카메라 숙임 바뀜)
바닥이 장애물로 찍혀 후보가 사라지고 **아직 안 본 곳이 있는데 "탐사 끝"** 이 난다 (옛 `explore_rl` 노드로 1103 녹화에서 확인 — 정렬 끈 깊이는 33 s 에 거짓 닫힘).
의심되면 `camera_layer: false` 로 끈다.

### 깊이 카메라 숙임 (2026-11-20 실물 — 제자리 회전 · '대충 다 앎' 끝)

증상: 켜자마자 `목표 칸 -1 (목표 없음) · 명령 v 0.00 ω 1.80`(제자리 회전)이 이어지다 `끝: 대충 다 앎 (후보 0 …)`. 원인은 카메라 마운트(손 고정)가
설정(env.yaml `body.camera_pitch` 5.2°)보다 2° 더 숙여져 있던 것(바닥 면 실측 7.2°) — 그러면 0.9 m 밖 바닥이 띠 아래 끝(3 cm)보다 높게 계산돼
깊이 층에 **로봇을 둘러싼 장애물 고리**로 찍히고, 갈 수 있는 칸이 로봇 곁에 갇혀 지도 닫힘 → 후보 0 → 제자리 회전 → 5 s 뒤 '대충 다 앎'.
녹화 `app_1120_1842` 를 다시 돌리면 설정 0.0902 에서 3 s 째 닫힘·후보 0 (노드 로그와 같은 8.0 s 에 끝), 0.1250 에서는 닫힘 없이 후보 6~8 개
(근거·그림 `limo_upper_keep/reports/real_spin.md`). 그래서 env.yaml 숙임을 0.1250 으로 다시 맞췄다(시뮬도 같은 값).

- **노드가 스스로 잼**: 1 s 마다 깊이 영상 한 장의 바닥 점을 모아 숙임·높이를 맞추고(`train/explore/sim/include/cam_floor.h` — 녹화 도구와 같은 함수),
  설정과 `camera_pitch_check_deg`(1°) 넘게 다르면 경고 로그와 `/explore/status` 에 `깊이 카메라 숙임 실측 7.2° · 설정 5.2° …`. 켜기 전(idle)에도 잰다. 판단·명령은 안 바꾼다(설정 그대로 — 시뮬과 같은 값).
- **녹화로 재기**: `low_explore_replay <녹화> --floor-fit` → 숙임·높이·잔차와, 그 바닥이 녹화 설정으로는 몇 cm 로 보이는지(3 cm 넘으면 층에 찍힘).
  `--cam-pitch <rad>` 로 숙임만 바꿔 다시 돌리면 후보·닫힘이 어떻게 바뀌는지 본다 (`--upper cand` = 판단 모델이 없을 때의 노드와 같은 위층).
- 경고가 나면: 마운트를 고치거나 `--floor-fit` 값으로 env.yaml `camera_pitch` 를 고치고(analysis/calibration.yaml `depth_pitch` 에 맞춤·확인 녹화), 노드를 다시 빌드·실행.
- 발사대(launcher_v5)는 카메라 시야에 없다: 축이 base_link 뒤 0.04 m · 총구가 앞 0.01 m 쯤이고 카메라는 앞 0.15 m 에서 앞을 봄. 녹화에서도 늘 같은 픽셀에 찍히는 점이 없다(`--depth-pix`).

### 위층 판단 모델 띄우기

로봇 안에서 llama-server(판단 모델 d1-3B, `/v1/systemone` 이 있는 빌드)를 먼저 띄운다. `-lm dio` 를 꼭 준다: 기본(mmap)이면 모델 파일(2.9 GB)이 page cache 에 남은 채
같은 내용을 GPU 버퍼로 또 잡아 Orin Nano 8 GB 에서 KV 캐시 할당이 실패한다(NvMap error 12 — Jetson 은 GPU 할당 때 page cache 를 못 비움). DirectIO 로 읽으면 page cache 를 안 거친다.
이미 page cache 가 차 있으면(free 3 GB 아래) 한 번 비우고 띄운다 (`sync; echo 3 | sudo tee /proc/sys/vm/drop_caches`):

```bash
cd ~/decider_test && LLAMA_DECISION_IMAGES_LAST=1 nohup ./llama.cpp/build/bin/llama-server -m models/d1-3B-Q8_0.gguf --mmproj models/mmproj-d1-3B-Q8_0.gguf \
  --host 127.0.0.1 --port 8171 -ngl 99 -c 4096 -np 1 -lm dio > server_dec.log 2>&1 &
# 뜰 때까지 (모델 읽기 약 1 분): grep -m1 "listening" server_dec.log
ros2 launch limo_fire_explore app.launch.xml [dry_run:=false] [rviz:=true]
```

서버는 `train/explore/upper_lora/patches/` 의 두 패치(판단 요청 checkpoint · 그림을 글 뒤로)를 넣은 빌드에 `LLAMA_DECISION_IMAGES_LAST=1` 로 띄운다 — 결정마다 같은 앞부분(템플릿 · 설명 글 · 힌트)을 다시 계산하지 않음. 미세조정 v1(LoRA)도 이 차례로 학습한다.
요청 꼴 기본은 짧은 꼴(`decider_compact: true`, 지도 288 px, 결정 기록 4, 카메라 없음 — 카메라 키프레임 사진은 v1 학습에 없어 2026-11-17 지움): 로봇 Orin 에서 요청 하나 약 1.6 s(앞 요청 뒤) · 2.2 s(처음) — 예전 기본(긴 꼴 + 카메라 2 × 2)은 5.0–5.6 s (`upper_lora/README.md` "요청 꼴과 지연").
사건 때만 묻고(event) 요청에 후보를 6 개까지 넣는다(`decider_max_cands: 6`) — 시뮬 위층 비교(`train/explore/upper_lora/README.md` "위층 비교")에서 가장 빨랐고, 라벨 굴리기(`dec_rollout`)·`low_explore_check --upper decider` 기본도 같은 값.
상태는 `/explore/status` 끝에 "판단 모델: 보냄 · 답 (마지막 ms) · 시간 끝 · 실패 · 후보 안 맞음 · 규칙 대신". 서버가 없거나 답이 잇따라 틀리면 규칙(cand_pick)이 대신 고르며 계속 돈다 (멈추지 않음).
요청 입력은 지도 사진 + 후보 글(시뮬 `--dec-input both`)로 고정 — 옛 시뮬 장면 카메라 그림(`combine_run --decider-input all`, 2026-11-18 지움)은 실물에 같은 것이 없었다.
