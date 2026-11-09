# baselines/frontier — 프론티어 탐사 기준선

탐사 RL 정책이 이겨야 할 비교 대상. 성공 기준 "95 % 도달 시간이 프론티어보다 짧음"의 프론티어가 이것이다.
GPU 시뮬(`sim/`)을 **`env_api.h` 로만** 쓰고(시뮬 코드는 안 고침), 정책처럼 **지도(`get_map`)와 지도 자세(참값 + 오차)만** 본다.

```
baselines/frontier/
├── config/frontier.yaml          explore_lite 값 · Nav2 BT 복구 값 · 두 속도 판(nav2 / fast) · 안전 시간
├── include/frontier_explorer.h   Explorer: 지도 + 지도 자세 → (v, ω), 끝 선언
├── src/frontier_explorer.cpp
├── tools/frontier_check.cu       mdp/frontier.h CPU == GPU 비트 비교
└── CMakeLists.txt                sim/ 을 add_subdirectory 로 가져와 frontier_run·frontier_check 빌드
../../mdp/frontier.h              칸 종류·닫힘·프론티어 찾기 (호스트·장치 공용, RL 관측도 같은 함수)
../../scripts/frontier_run.cpp    장면 × 판을 돌려 학습 뷰어 실행 폴더로
```


## 한 걸음 (0.1 s = 시뮬 속도 행동 한 번)

1. **시작**: 설 수 있는 칸에서 번져 프론티어 칸이 하나 보일 때까지 제자리 회전(최대 30 s). explore_lite 는 지도가 생긴 뒤 켠다는 전제라 처음부터 돌리면 바로 멈춘다(`explore.cpp:190-193`).
2. **고르기 (explore_lite `makePlan`)**: 0.33 Hz, 그리고 목표 도착·실패 때 바로. `mdp::frontier_search`(아래 칸 종류·닫힘) → 비용 = `potential_scale × 거리(m) × 해상도 − gain_scale × 크기(칸) × 해상도` 오름차순 → 블랙리스트 아닌 첫 덩어리의 **중심**이 목표. 크기 문턱(0.75 m)을 넘는 덩어리가 없고 작은 것만 남으면 작은 것들을 후보로. 같은 목표면 다시 안 보냄. 30 s 동안 진척(덩어리까지 거리 줄어듦) 없으면 블랙리스트.
3. **가기 (Nav2 식)**: 비용 지도 = 지도 막힘 칸 + 장애물 층 흉내(로봇 3 m 안의 `l > 0` 칸) → Nav2 InflationLayer 식 팽창(내접 0.2 m 253, 0.45 m 까지 `252·e^{−3(d−0.2)}`). Smac 2D 와 같은 비용의 8 이웃 A*(모름 칸 지남, tolerance 0.25 m). 1 Hz 또는 앞 경로가 막히면 다시 짬. 못 짜면 ABORTED → 블랙리스트 → 바로 다시 고름.
4. **추종 (RPP)**: 경로에서 lookahead 0.5 m 이상 떨어진 첫 점. 45° 넘게 틀어지면 제자리 회전 1 rad/s, 아니면 `ω = v · 2 sin(α) / ld`. 속도는 RPP `applyConstraints` 대로 곡률 감속(`use_regulated_linear_velocity_scaling: true`, 반지름 < 0.9 m 면 `v × r / 0.9`, 아래 한도 0.1 m/s)과 (판에 따라) 장애물 근접 감속 중 작은 쪽 → 가까우면 다가가기 감속 → velocity_smoother 가속 한도 → (v, ω).
5. **충돌 검사·복구**: RPP `isCollisionImminent` 처럼 발자국(local_costmap footprint + padding 0.02 → 0.36 × 0.26 m)을 명령으로 1 s(당근까지) 굴려 보고, 닿으면 Nav2 기본 BT 의 복구를 RoundRobin 으로 하나씩(비우기 · Spin 1.57 · Wait 5 s · BackUp 0.30 m), 6 번 넘으면 ABORTED.
6. **끝**: 지도가 닫히면(`closed`) 또는 남은 덩어리가 다 블랙리스트면(`all_blacklisted`) 끝을 선언. 안전 시간 1800 s 는 끊김.

값은 전부 파일에서 읽는다: RPP·velocity_smoother·goal checker·팽창·Smac·장애물 층·발자국은 `env.yaml` 이 가리키는 `src/limo_fire/limo_fire_navigation/config/nav2_params.yaml`(시뮬과 같은 읽기 `sim::load_env_cfg`), explore_lite·BT 값·판별 RPP 스위치는 `config/frontier.yaml`.

| 판 | 직진 | 그 밖 |
|---|---|---|
| `nav2` | `desired_linear_vel` 0.26 m/s, smoother `max_velocity` 0.26 | nav2_params 그대로 |
| `fast` | 1.0 m/s (LIMO 데이터시트 최고 속도, `mdp/README` 행동 표) | RPP 장애물 근접 감속(`use_cost_regulated_linear_velocity_scaling`, Nav2 기본 `cost_scaling_dist` 0.6 · `gain` 1.0)을 켬(아래 충돌 원인). 회전 1 rad/s·가속·lookahead·팽창은 nav2 값 그대로 |

### 칸 종류·닫힘 (`mdp/frontier.h`, 정의는 `mdp/README` "덮음·닫힘·끝")

| 종류 | 규칙 | 근거 |
|---|---|---|
| 모름 | 한 번도 안 건드린 칸 | Cartographer `/map` 의 −1 = 관측 안 됨(`cartographer_ros/msg_conversion.cc:407-410`) |
| 막힘 | p > 0.65 ⇔ 로그 오즈 × 1000 ≥ 620 (맞음 4 번) | `cartographer_ros/ros_map.cc:44-45` `occupied_thresh: 0.65` |
| 갈 수 있음 | 그 밖의 건드린 칸 | 빈 칸 문턱 p < 0.196(지나감 36 번)은 안 씀 — 95 % 뒤 긴 꼬리의 원인이었다 |

- 설 수 있는 칸 = 갈 수 있음 + `robot_radius`(0.2 m) 안에 막힘 없음. 프론티어 칸 = 갈 수 있음 + 4 이웃에 모름.
- 찾기는 로봇에서 가장 가까운 설 수 있는 칸부터 설 수 있는 칸으로만 4 이웃 번지고, 번진 칸 안의 프론티어 칸을 8 이웃으로 묶는다. **닫힘** = 번진 칸 안에 프론티어 칸이 없음.

## 출처 (원본 소스를 `gh api` 로 열어 확인)

| 무엇 | 파일:줄 (저장소 @ 커밋) |
| 프론티어 찾기 BFS·덩어리·크기 문턱 | `explore/src/frontier_search.cpp:27-95, 97-170, 172-189` ([hrnr/m-explore](https://github.com/hrnr/m-explore) @ `712bdd41027b645c9c876a4c0071478f090825ea`) |
| 비용 식 | 같은 파일 `:191-196` |
| 이웃 차례·가장 가까운 빈 칸 | `explore/include/explore/costmap_tools.h` `nhood4`·`nhood8`·`nearestCell` |
| 고르기·블랙리스트·진척 시간·같은 목표 | `explore/src/explore.cpp:179-245, 247-262, 264-281` |
| 값 (potential 3.0, gain 1.0, min_frontier_size 0.75, planner_frequency 0.33, progress_timeout 30) | `explore/launch/explore.launch` |
| 팽창 비용 | `nav2_costmap_2d/include/nav2_costmap_2d/inflation_layer.hpp:148-162` ([ros-navigation/navigation2](https://github.com/ros-navigation/navigation2) humble @ `3c3db59d6969d8ecee8e68468693d006397f4a0c`) |
| A* 한 칸 비용 | `nav2_smac_planner/src/node_2d.cpp:64-80` |
| 당근 점·충돌 검사 | `nav2_regulated_pure_pursuit_controller/src/regulated_pure_pursuit_controller.cpp:442-454, 475-580` |
| 1 Hz 다시 짜기·복구 | `nav2_bt_navigator/behavior_trees/navigate_to_pose_w_replanning_and_recovery.xml` |
| 지도 문턱 | `cartographer_ros/cartographer_ros/ros_map.cc:44-45`, `msg_conversion.cc:407-410` ([cartographer-project/cartographer_ros](https://github.com/cartographer-project/cartographer_ros) @ `c138034db0c47fe0ea5a2abe516acae02190dbf5`) |

| 곡률·장애물 근접·다가가기 감속, 속도 비례 lookahead | `regulated_pure_pursuit_controller.cpp:636-672, 264-276`, 기본값 `:70-101` |
| 1 Hz 다시 짜기·복구 | `nav2_bt_navigator/behavior_trees/navigate_to_pose_w_replanning_and_recovery.xml` |
| 지도 문턱 | `cartographer_ros/cartographer_ros/ros_map.cc:44-45`, `msg_conversion.cc:407-410` ([cartographer-project/cartographer_ros](https://github.com/cartographer-project/cartographer_ros) @ `c138034db0c47fe0ea5a2abe516acae02190dbf5`) |

원본과 다른 점(의도): ① 덩어리 첫 칸도 중심 합·최소 거리에 넣음(원본은 크기에만 넣어 중심이 세계 원점 쪽으로 치우침). ② 시작 대기(위 1). ③ 장애물 층은 스캔 대신 지도의 `l > 0` 칸(시뮬 API 가 스캔을 안 내줌). ④ 시작 칸이 내접 반경 안이면 그 칸들을 정규 비용 20 으로 지나가게(Nav2 는 복구 동작으로 빠져나옴, `inscribed_escape_cost` 잠정). ⑤ 추종은 10 Hz(Nav2 controller 20 Hz). 장애물 근접 감속의 거리는 로봇 칸 비용을 되돌리는 대신 지도에서 바로 잼(같은 값), 속도 비례 lookahead 의 속도는 smoother 출력. ⑥ 프론티어 칸·번지기·끝은 우리 닫힘 정의(위) — explore_lite 는 "빈 칸에 닿은 모름 칸"을 빈 칸 전체로 번져 찾고 큰 덩어리가 없으면 끝낸다. ⑦ 시뮬 goal 모드 추종기(`env.h physics_step`)엔 곡률 감속이 아직 없다(기준선만).

## 빌드·실행

```bash
export PATH=/usr/local/cuda-13.0/bin:$PATH
cmake -S train/explore/baselines/frontier -B build/frontier -DCMAKE_CUDA_ARCHITECTURES=86   # RTX 3060. 5070 Ti 는 120
cmake --build build/frontier -j                                                              # → frontier_run, frontier_check, sim/ 도구들

# 시험 장면 (git 제외, train/explore/scenes/). --occupied 230: map_saver 지도의 모름(205)도 막힘으로 — 집 밖 모름 영역을 막는다.
# 출발점은 기본 지도 원점(0, 0) — 덮음 분모를 여기서 flood fill 로 정한다(집이 안 닫히면 build_world 가 새는 곳 좌표와 함께 실패)
for m in bookstore small-house; do
  build/frontier/sim/map2scene sim/maps/aws_robomaker/$m/maps/turtlebot3_waffle_pi/map.yaml train/explore/scenes/$m.scene --occupied 230
done

cd train/explore
../../build/frontier/frontier_run --scenes scenes/small-house.scene scenes/bookstore.scene --modes nav2,fast --episodes 8 --seed 1 --root runs --name frontier
viewer/target/release/runcheck runs/frontier_nav2 runs/frontier_fast
viewer/target/release/trainview --root runs --port 8097
../../build/frontier/frontier_check scenes/bookstore.scene scenes/small-house.scene             # + --negative 로 검사가 틀림을 잡는지
```

`frontier_run` 옵션: `--episodes N`(장면마다 GPU 환경 N 개 = 판 N 개), `--obs-every K`(재생 관측 지도 K 걸음마다, 기본 5 = 0.5 s), `--cpu`(CPU 참조 시뮬), `--env`·`--model`·`--frontier`(설정 파일). `FRONTIER_DEBUG=1` 이면 고를 때마다·충돌 직전 상태를 stderr 에. `--people K`(계층형 `slot0` 만): 판마다 사람 0 ~ K 명 — 끝 지표에 벽 충돌·사람 충돌·충돌 때 명령(앞·제자리 회전·뒤·멈춤)·판 평균 길이·충돌 감시 비율이 따로 나온다(`SIM_HIER_AVOID=0` 이면 사람 피하기 끔, mdp/README "계층형").


## 실행 폴더 (`viewer/include/runfolder.h`)

`runs/frontier_<판>/`: `run.json`(`kind: "baseline"`, `mode`, `group: frontier_<판>`, 쓴 값 전부 `params`, 장면(벽 면 칸 수·바닥 넓이·닫힘), 덮음·바닥 덮음·끝 정의, 시뮬 덮어쓰기, 출처), `progress.jsonl`(장면마다 한 줄: `rollout/*` 지표 + `eval/frontier_t95_mean`), `episodes.jsonl`, `evals/frontier.json`(장면마다 `t95`·`ref_t95`·`tail_s`·`floor_coverage`), `replays/ep_*.xrp`(정답 맵, 관측 지도, **덮인 벽 면 칸**(`Env::get_seen`), 참·지도 자세 궤적, 라이다, 지령·바퀴 속도, 덮음·안 넓이, 사건 비트).

판 지표(episodes 한 줄)

| 키 | 뜻 |
|---|---|
| `coverage` | 끝 덮은 비율 = 벽 면 칸 덮음(시뮬 `env.h`, 정의는 `mdp/README` "덮음·닫힘·끝") |
| `floor_coverage` | 빔이 한 번이라도 지나간 갈 수 있는 바닥 칸 비율 — 보조 지표 |
| `t95` · `tail_s` | 덮음 ≥ 0.95 가 처음 된 시각(s), 그때부터 판 끝까지(s). 못 닿으면 키 없음 |
| `area_m2` · `n_wall` | 갈 수 있는 바닥 넓이, 벽 면 칸 수(덮음 분모) |
| `path_len` | 참 자세 이동 거리(m) |
| `collided` · `outcome` · `end_reason` | 충돌(시뮬 판 끝) / 선언 — 덮음 ≥ 0.95 면 `success`, 아니면 `declared_short` / 안전 시간 `truncated`. 까닭 `closed`·`all_blacklisted`·`no_frontier_at_start`·`collision`·`t_safe` |
| `known_m2` | 지도에서 갈 수 있음 칸 넓이 |
| `baseline.*` | 고르기·목표·도착·실패(ABORTED)·진척 시간 초과·A*·막힘 다시 짜기·충돌 검사 멈춤·복구 수·블랙리스트 크기 |

시뮬 덮어쓰기: `env.yaml episode.success_coverage`(0.95 에서 자동 성공 끝)를 끄고 `max_steps` 를 안전 시간보다 길게 — 판 끝은 기준선 선언·충돌·안전 시간만.

## 확인 (2026-11-04, RTX 3060 sm_86)

| 무엇 | 결과 |
|---|---|
| `env_verify`(두 장면, 8 환경 × 80·400 행동, 속도·목표점 행동) | 상태·보상·지도·**덮음 표시**·관측 GPU == CPU 비트 같음, `--negative` 는 다름을 잡음 |
| `frontier_check`(두 장면, 4 환경 × 3 시점 = 지도 24 장) | 칸 판정·요약·찾기 다름 0. `--negative`(로봇 1 m 앞 갈 수 있는 칸 하나를 모름으로) 칸 22 · 요약 10 · 찾기 6 다름. 이 24 장엔 닫힌 지도가 없다 |
| `runcheck runs/frontier_nav2 runs/frontier_fast` | 둘 다 OK (판 16 · 재생 16 · 평가 표 1) |

### 지표 (판 8 개씩, 씨앗 1)

장면: 작은 집 벽 면 칸 2 594 (갈 수 있는 바닥 147.8 m²), 서점 3 206 (145.3 m²), 둘 다 닫힘.

| 판 | 장면 | 성공 | 충돌 | 기준 미달 선언 | 끝 덮음 평균 | 95 % 도달 | t95 평균 s | 꼬리(95 % → 끝) 평균 s |
|---|---|---|---|---|---|---|---|---|
| nav2 | 작은 집 | 3 | 1 | 4 | 0.936 | 3/8 | 325.9 | 76.8 |
| nav2 | 서점 | 3 | 2 | 3 | 0.803 | 3/8 | 619.8 | 213.8 |
| fast | 작은 집 | 2 | 1 | 5 | 0.926 | 2/8 | 146.1 | 58.0 |
| fast | 서점 | 6 | 1 | 1 | 0.958 | 6/8 | 191.7 | 36.0 |

덮음 정의가 바뀌어(바닥 → 벽 면 칸) t95 는 옛 표와 바로 비교할 수 없다. 꼬리는 옛 정의(빈 칸 확정 문턱으로 끝 판정)에서 nav2 458.7·568.0 s, fast 243.0·107.0 s 였다.

## 충돌 원인 (2026-11-04, 재서 확정)

충돌마다 직전 상태에서 몸통이 닿은 참 막힌 칸, 그 칸의 지도 값, 그 칸을 지나는 장면 물체와 이 판의 라이다 재질을 쟀다(진단 도구는 저장소 밖, 같은 씨앗으로 `frontier_run` 과 판 결과가 같음을 확인). 두 장면 × 두 판 × 씨앗 3 × 판 8 = 판 96 개씩.

| 설정 | 충돌 | 안 보이는 재질 물체 | 지도가 아는 벽 | 성공 |
|---|---|---|---|---|
| 옛 설정 (곡률 감속 없음) | 18 | 10 | 7 (모두 fast) | 54 |
| + 곡률 감속 (nav2_params 값) | 17 | 11 | 6 | 54 |
| + `robot_radius` 0.25 | 9 | 8 | 1 | 31 |
| fast + 장애물 근접 감속 (채택) | fast 4 / 48 | 4 | **0** | fast 31 / 48 |

1. **안 보이는 재질의 작은 물체(가장 많음)**: 닿은 칸의 물체가 판마다 뽑는 "안 보이는 재질"(`sensor_model.yaml lidar.dark`, 빠짐 확률 1.0 — 실물 녹화에서 맞춘 값)이라 라이다가 한 번도 못 맞히고 지도엔 빈 칸(l −80 ~ −2197)이다. 계획기 설정으로는 못 고친다. 최종 실행의 충돌 5 개는 모두 이것.
2. **fast 판의 제동 거리 (c)**: 지도가 아는 벽에 부딪힌 7 개는 모두 fast 판, 명령은 이미 0 인데 바퀴가 0.3~0.5 m/s 로 밀고 들어감. 구동 모델로 잰 제동 거리는 0.26 m/s 에서 0.044 m, 1 m/s 에서 0.317 m(smoother 2.5 m/s² 거침) — RPP 충돌 검사는 발자국 여유 0.02 m 로 당근(≤ 0.5 m)까지만 보니 1 m/s 에선 늦다. Nav2 RPP 의 장애물 근접 감속을 켜자 판 48 개에서 0. 속도 비례 lookahead(0.9 m)는 2 로 덜 줄었다.
3. **`robot_radius` 0.2 (a)**: 0.25 로 키우면 아는 벽 충돌이 줄지만 문·좁은 곳이 막혀 성공이 54 → 31 로 떨어진다. nav2 판(0.26 m/s)은 0.2 그대로 아는 벽 충돌이 판 48 개에 0~1 이라 바꾸지 않았다.
4. **곡률 감속 (d)**: nav2_params 가 `use_regulated_linear_velocity_scaling: true` 인데 빠져 있었다 — 실물 설정과 같게 넣었다. 충돌 수는 거의 같다(18 → 17).
5. **벽 좌표 (b)**: `map2scene` 의 등고선(`--occupied 230` → iso 25)은 빈 픽셀(254)과 검정 픽셀 사이에서 빈 쪽 9.4 % 지점이라, 장면 벽 표면이 검정 픽셀 앞 경계보다 2.0 cm 방 쪽(벽 면 칸 안 0.0297 m 깊이, 작은 집 2 659 · 서점 3 294 광선 모두 같음)에 선다. 그래서 충돌 격자의 벽 면 칸은 100 % 지도 PGM 에서 빈 픽셀(검정 픽셀 한 칸 앞)이다. 다만 라이다 지도도 같은 장면으로 쌓이므로 라이다 지도 벽과 충돌 격자는 같은 칸이고(충돌 칸의 지도 값은 막힘 l 1248~2197), 충돌 원인은 아니다. 표면이 칸 앞쪽 59 % 를 비워 두어 스치는 빔이 지나감으로 쳐서, 판 끝 벽 면 칸의 18~21 % 가 지도에서 빈 칸이다.

실물 위험(같은 `nav2_params.yaml`): Nav2 장애물 층이 `/scan` 하나만 보므로 라이다가 못 보는 물체에는 실물도 같은 위험이 있다(깊이 카메라를 장애물 층에 넣을지 따로 정할 것). `robot_radius` 0.2 는 몸통 반대각 0.194 m 에 칸 반 대각(0.035 m)을 못 더한 값이라 경로가 벽 0.2~0.25 m 까지 붙는다 — 0.26 m/s 에선 제동 거리 4 cm 라 시뮬에선 아는 벽 충돌이 거의 없었다. 1 m/s 로 올릴 거면 장애물 근접 감속을 켜야 한다.

## 남은 문제 (있는 그대로)

- **95 % 미달 선언이 많다**: 판 32 개 중 13 개 — 12 개는 덮음 0.86~0.947 에서 `all_blacklisted` 로 끝냄(목표 도착 0~3 번 — 중심이 모름 칸 쪽이거나 막힌 곳이라 진척 없음·경로 실패로 블랙리스트). 닫혀서 끝난 판은 1 개뿐이고 그것도 잘못 닫힘(아래).
- **잘못 닫힘**: nav2 서점 env 1 은 128 s, 덮음 0.403 에서 `closed` — 로봇이 설 수 있는 칸 덩어리가 막힘 팽창으로 갇혀 프론티어에 안 닿음(로봇 반경 팽창 닫힘 정의의 약점). 원인은 아직 안 잼.
- 판 끝에 남는 프론티어 칸(번진 칸 안)은 참 바닥·벽 바깥 칸이 대부분 — 벽 면 칸이 스치는 빔으로 빈 칸이 되어 벽 안쪽 모름과 닿는 곳이 있다(위 5).
- 재생 라이다는 참 자세에서 정적 장면에 다시 쏜 깨끗한 광선이다(시뮬 안의 오차 스캔은 API 로 안 나옴). 관측 지도는 0.5 s 마다(재생 파일 nav2 판 2.4~13.3 MB).
- progress 는 장면마다 한 줄이라 Training 탭 곡선의 가로축은 시간이 아니라 장면 차례다(머리 카드는 마지막 장면 값). 장면별 값은 평가 표 `evals/frontier.json` 에.
- 기준선은 호스트에서 환경마다 순서대로 돈다(두 장면 × 판 8 개: nav2 124 s, fast 51 s). 같은 씨앗이면 판 결과가 같다.
