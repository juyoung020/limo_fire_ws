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
../../mdp/frontier.h              프론티어 칸 판정·요약·explore_lite 찾기 (호스트·장치 공용, RL 관측도 같은 함수)
../../scripts/frontier_run.cpp    장면 × 판을 돌려 학습 뷰어 실행 폴더로
```

## 한 걸음 (0.1 s = 시뮬 속도 행동 한 번)

1. **시작**: 아는 빈 칸에서 이어진, 크기 문턱을 넘는 프론티어가 하나 생길 때까지 제자리 회전(최대 30 s). explore_lite 는 지도가 생긴 뒤 켠다는 전제라 처음부터 돌리면 바로 멈춘다(`explore.cpp:190-193`).
2. **고르기 (explore_lite `makePlan`)**: 0.33 Hz, 그리고 목표 도착·실패 때 바로. `mdp::frontier_search` → 비용 = `potential_scale × 거리(m) × 해상도 − gain_scale × 크기(칸) × 해상도` 오름차순 → 블랙리스트 아닌 첫 덩어리의 **중심**이 목표. 같은 목표면 다시 안 보냄. 30 s 동안 진척(덩어리까지 거리 줄어듦) 없으면 블랙리스트.
3. **가기 (Nav2 식)**: 비용 지도 = 지도 막힘 칸 + 장애물 층 흉내(로봇 3 m 안의 `l > 0` 칸) → Nav2 InflationLayer 식 팽창(내접 0.2 m 253, 0.45 m 까지 `252·e^{−3(d−0.2)}`). Smac 2D 와 같은 비용의 8 이웃 A*(모름 칸 지남, tolerance 0.25 m). 1 Hz 또는 앞 경로가 막히면 다시 짬. 못 짜면 ABORTED → 블랙리스트 → 바로 다시 고름.
4. **추종 (RPP)**: 경로에서 lookahead 0.5 m 이상 떨어진 첫 점. 속도·곡률은 **시뮬 goal 모드 추종기(`env.h physics_step`)와 같은 식**(45° 넘게 틀어지면 제자리 회전 1 rad/s, 가까우면 감속, `ω = v · 2 sin(α) / ld`) → velocity_smoother 가속 한도 → (v, ω).
5. **충돌 검사·복구**: RPP `isCollisionImminent` 처럼 발자국(local_costmap footprint + padding 0.02 → 0.36 × 0.26 m)을 명령으로 1 s(당근까지) 굴려 보고, 닿으면 Nav2 기본 BT 의 복구를 RoundRobin 으로 하나씩(비우기 · Spin 1.57 · Wait 5 s · BackUp 0.30 m), 6 번 넘으면 ABORTED.
6. **끝**: 남은 프론티어가 없거나(`no_frontier`) 다 블랙리스트면(`all_blacklisted`) 끝을 선언. 안전 시간 1800 s 는 끊김.

값은 전부 파일에서 읽는다: RPP·velocity_smoother·goal checker·팽창·Smac·장애물 층·발자국은 `env.yaml` 이 가리키는 `src/limo_fire/limo_fire_navigation/config/nav2_params.yaml`(시뮬과 같은 읽기 `sim::load_env_cfg`), explore_lite·BT 값은 `config/frontier.yaml`.

| 판 | 직진 | 그 밖 |
|---|---|---|
| `nav2` | `desired_linear_vel` 0.26 m/s, smoother `max_velocity` 0.26 | nav2_params 그대로 |
| `fast` | 1.0 m/s (LIMO 데이터시트 최고 속도, `mdp/README` 행동 표) | 회전 1 rad/s·가속·lookahead·팽창은 nav2 값 그대로 |

### 칸 종류 문턱 (`mdp/frontier.h`)

| 종류 | 규칙 | 근거 |
|---|---|---|
| 모름 | 한 번도 관측 안 됨, 또는 관측했지만 아래 둘 사이 | Cartographer `/map` 의 −1 = 관측 안 됨(`cartographer_ros/msg_conversion.cc:407-410`), map_server 는 문턱 사이를 −1 로 읽음 |
| 막힘 | p > 0.65 ⇔ 로그 오즈 × 1000 ≥ 620 | `cartographer_ros/ros_map.cc:44-45` `occupied_thresh: 0.65` (map_saver 상수) |
| 아는 빈 칸 | p < 0.196 ⇔ ≤ −1412 | 같은 줄 `free_thresh: 0.196` |

시뮬 갱신값(hit 0.55 → +201, miss 0.49 → −40)으로는 막힘 = 맞음 4 번, 빈 칸 = 지나감 36 번. 빈 칸 쪽이 엄격해서 "저장할 지도에 모름이 안 남을 때까지" 탐사한다 — 95 % 뒤 꼬리가 긴 까닭(아래 결과).

## 출처 (원본 소스를 `gh api` 로 열어 확인)

| 무엇 | 파일:줄 (저장소 @ 커밋) |
|---|---|
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

원본과 다른 점(의도): ① 덩어리 첫 칸도 중심 합·최소 거리에 넣음(원본은 크기에만 넣어 중심이 세계 원점 쪽으로 치우침). ② 시작 대기(위 1). ③ 장애물 층은 스캔 대신 지도의 `l > 0` 칸(시뮬 API 가 스캔을 안 내줌). ④ 시작 칸이 내접 반경 안이면 그 칸들을 정규 비용 20 으로 지나가게(Nav2 는 복구 동작으로 빠져나옴, `inscribed_escape_cost` 잠정). ⑤ 추종은 10 Hz(Nav2 controller 20 Hz), RPP 의 곡률 감속(`use_regulated_linear_velocity_scaling`)은 시뮬 추종기 식에 없어서 뺌.

## 빌드·실행

```bash
export PATH=/usr/local/cuda-13.0/bin:$PATH
cmake -S train/explore/baselines/frontier -B build/frontier -DCMAKE_CUDA_ARCHITECTURES=86   # RTX 3060. 5070 Ti 는 120
cmake --build build/frontier -j                                                              # → frontier_run, frontier_check, sim/ 도구들

# 시험 장면 (git 제외, train/explore/scenes/). --occupied 230: map_saver 지도의 모름(205)도 막힘으로 — 집 밖 모름 영역을 막는다
for m in bookstore small-house; do
  build/frontier/sim/map2scene sim/maps/aws_robomaker/$m/maps/turtlebot3_waffle_pi/map.yaml train/explore/scenes/$m.scene --occupied 230
done

cd train/explore
../../build/frontier/frontier_run --scenes scenes/small-house.scene scenes/bookstore.scene --modes nav2,fast --episodes 8 --seed 1 --root runs --name frontier
viewer/target/release/runcheck runs/frontier_nav2 runs/frontier_fast
viewer/target/release/trainview --root runs --port 8097
../../build/frontier/frontier_check scenes/bookstore.scene scenes/small-house.scene             # + --negative 로 검사가 틀림을 잡는지
```

`frontier_run` 옵션: `--episodes N`(장면마다 GPU 환경 N 개 = 판 N 개), `--obs-every K`(재생 관측 지도 K 걸음마다, 기본 5 = 0.5 s), `--cpu`(CPU 참조 시뮬), `--no-interior`, `--env`·`--model`·`--frontier`(설정 파일). `FRONTIER_DEBUG=1` 이면 고를 때마다·충돌 직전 상태를 stderr 에.

**덮을 칸 다시 고르기**: `sim::build_world` 는 "시작 후보가 가장 많은 빈 칸 덩어리"를 덮을 칸으로 고르는데, map_saver 지도에선 그것이 집 밖(지도 바깥 모름 영역)이었다(작은 집 457 m², 로봇이 집 밖 들판에서 시작). `frontier_run` 은 `World` 를 Env 에 넘기기 전에 "맵 테두리에 안 닿는 가장 큰 빈 칸 덩어리"로 다시 고른다(작은 집 148.3 m², 서점 145.6 m²). 시뮬 쪽 수정은 하지 않았다(아래 남은 문제).

## 실행 폴더 (`viewer/include/runfolder.h`)

`runs/frontier_<판>/`: `run.json`(`kind: "baseline"`, `mode`, `group: frontier_<판>`, 쓴 값 전부 `params`, 장면, 덮음 정의와 **"덮음 정의 미정"**, 시뮬 덮어쓰기, 출처), `progress.jsonl`(장면마다 한 줄: `rollout/*` 지표 + `eval/frontier_t95_mean`), `episodes.jsonl`, `evals/frontier.json`(장면마다 `t95`·`ref_t95`), `replays/ep_*.xrp`(정답 맵, 관측 지도, 덮인 칸, 참·지도 자세 궤적, 라이다, 지령·바퀴 속도, 덮음·안 넓이, 사건 비트).

판 지표(episodes 한 줄)

| 키 | 뜻 |
|---|---|
| `coverage` | 끝 덮은 비율 = 지금 시뮬 정의 `known_gt`(정답 빈 칸 중 지도에서 한 번이라도 관측된 칸). **덮음 정의 미정**(`mdp/README` 정할 것 1) |
| `t95` | 덮음 ≥ 0.95 가 처음 된 시각(s). 못 닿으면 키 없음 |
| `path_len` | 참 자세 이동 거리(m) |
| `collided` · `outcome` · `end_reason` | 충돌(시뮬 판 끝) / 선언 — 덮음 ≥ 0.95 면 `success`, 아니면 `declared_short` / 안전 시간 `truncated`. 까닭 `no_frontier`·`all_blacklisted`·`no_frontier_at_start`·`collision`·`t_safe` |
| `known_m2` | 지도에서 아는 빈 칸 넓이 |
| `baseline.*` | 고르기·목표·도착·실패(ABORTED)·진척 시간 초과·A*·막힘 다시 짜기·충돌 검사 멈춤·복구 수·블랙리스트 크기 |

시뮬 덮어쓰기: `env.yaml episode.success_coverage`(0.95 에서 자동 성공 끝)를 끄고 `max_steps` 를 안전 시간보다 길게 — 판 끝은 기준선 선언·충돌·안전 시간만.

## 확인 (2026-11-04, RTX 3060 sm_86)

| 무엇 | 결과 |
|---|---|
| `env_verify`(두 장면, 8 환경 × 80 행동) | GPU == CPU 비트 같음, `--negative` 는 다름을 잡음 |
| `frontier_check`(두 장면, 4 환경 × 3 시점 = 지도 24 장, 프론티어 칸 10 505 · 덩어리 40) | 칸 판정·요약·찾기 다름 0, 문턱 상수 == 식. `--negative`(빈 칸 하나를 모름으로) 칸 14 · 요약 5 · 찾기 5 다름 |
| `runcheck runs/frontier_nav2 runs/frontier_fast` | 둘 다 OK (판 16 · 재생 16 · 평가 표 1) |
| 뷰어 Training·Replay·Compare | 콘솔 오류 0. Replay 에서 관측 지도가 프론티어를 따라 채워짐 |

### 지표 (판 8 개씩, 씨앗 1)

| 판 | 장면(덮을 넓이) | 성공 | 충돌 | 기준 미달 선언 | 끝 덮음 평균 | 95 % 도달 | t95 평균 (최소~최대) s | 이동 m | 판 길이 s |
|---|---|---|---|---|---|---|---|---|---|
| nav2 | 작은 집 (148.3 m²) | 5 | 3 | 0 | 0.997 | 8/8 | 116.7 (67.5~224.8) | 118.8 | 575.3 |
| nav2 | 서점 (145.6 m²) | 7 | 1 | 0 | 0.918 | 7/8 | 229.0 (182.5~269.6) | 144.0 | 705.2 |
| fast | 작은 집 | 4 | 4 | 0 | 0.999 | 8/8 | 70.3 (55.3~96.3) | 146.8 | 313.3 |
| fast | 서점 | 1 | 6 | 1 | 0.767 | 5/8 | 162.9 (126.4~204.7) | 84.9 | 184.5 |

충돌한 판도 그 전에 95 % 에 닿았으면 t95 에 들어 있다(충돌은 판 끝일 뿐 t95 를 지우지 않음). RL 기준으로 쓸 때 충돌 판을 뺄지는 정할 것.

## 남은 문제 (있는 그대로)

- **충돌이 많다**: nav2 4/16, fast 10/16. 직전 상태를 보면 로봇 가운데에서 가장 가까운 정답 벽 칸까지 0.15~0.31 m — Nav2 `robot_radius` 0.2 가 발자국 반대각(0.194 m) + 칸 반 대각(0.035 m)보다 작고, 정답 충돌 격자의 벽 칸이 지도에선 강한 빈 칸(l −2197)인 경우가 많다(장면 벽면을 칸에 찍는 것과 라이다 지도가 한 칸쯤 어긋나는 듯 — 추정, 안 잼). fast 는 nav2 용 lookahead·팽창으로 1 m/s 를 내서 더 잦다. RPP 곡률 감속을 뺀 것도 원인일 수 있다.
- **95 % 뒤 꼬리가 길다**: 빈 칸 문턱(p < 0.196, 지나감 36 번)이 엄격해서 nav2 판은 t95 뒤 평균 450 s 넘게 작은 프론티어를 쫓는다. 끝은 대부분 `all_blacklisted`(목표 도착 0~1 번 — 중심이 모름 칸이라 경로 끝이 목표에서 멀거나, 지도가 바뀌어 목표가 계속 바뀜).
- **이른 끝**: fast 서점 env 6 은 37 s, 덮음 0.386 에서 `no_frontier` 로 끝냄 — 빈 칸 문턱 때문에 빈 칸 덩어리가 끊겨 BFS 가 못 가는 프론티어가 있음(explore_lite 는 빈 칸으로 이어진 것만 봄).
- **시뮬 장면**: `build_world` 가 map_saver 지도에서 집 밖을 덮을 칸으로 고른다(위). `frontier_run` 에서 우회했고, 시뮬(`env_host.cpp build_world`)이나 `map2scene` 을 고칠지는 사용자 판단.
- 재생 라이다는 참 자세에서 정적 장면에 다시 쏜 깨끗한 광선이다(시뮬 안의 오차 스캔은 API 로 안 나옴). 관측 지도는 0.5 s 마다(재생 파일 크기: nav2 판 1.6~13.7 MB).
- progress 는 장면마다 한 줄이라 Training 탭 곡선의 가로축은 시간이 아니라 장면 차례다(머리 카드는 마지막 장면 값). 장면별 값은 평가 표 `evals/frontier.json` 에.
- 덮음 정의가 정해지면(`mdp/README` 정할 것 1) 다시 돌린다. 기준선은 호스트에서 환경마다 순서대로 돌린다(두 장면 × 판 8 개: nav2 150 s, fast 66 s 걸림). 같은 씨앗으로 다시 돌리면 판 결과가 그대로 같다(확인함).
