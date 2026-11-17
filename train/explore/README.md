# explore — 탐사 시뮬과 위층·아래층 측정

지금 탐사 구성(2026-11-16 실물 시험으로 정함)은 **위층 판단 모델(d1-3B) + 아래층 A* 경로 추종** 이다. 실물 노드는
[`limo_fire_explore` explorer](../../src/limo_fire/limo_fire_explore/README.md), 같은 코드를 시뮬에서 수천 판 돌려 재는 것이 여기다.

```
explore/
├── sim/         시뮬레이터: 물리·센서·지도 (상태 EnvState). 실측 값은 sim/config/env.yaml, 근거는 sim/analysis
├── mdp/         관측 + 크기 · 판 끝 판정 · 프론티어·닫힘 (시뮬과 실물 노드가 같이 씀)
├── scripts/     combine_run(두 층 합친 탐사 측정) · frontier_run(프론티어 기준선) · 위층 규칙(fuel_upper.h · low_explore_ctl.h · low_decider.h)
├── baselines/   비교 기준선: frontier/ = 프론티어 탐사(explore_lite + Nav2 식)
├── upper_lora/  위층 판단 모델 LoRA 학습 (지도 학습)
├── viewer/      실행 폴더 뷰어(trainview): 판 표·평가 표·2D 재생·실행 비교 + 러너가 쓰는 실행 폴더 헤더 include/runfolder.h
├── runs/        실행 폴더 (git 제외)
└── scenes/      시험 장면(.scene, git 제외 — 만드는 명령은 scenes/README · baselines/frontier/README)
```

## 구조 한눈에

실물과 시뮬이 **같은 코드**를 같은 차례로 부른다. 다른 것은 센서 값이 어디서 오느냐뿐이다.

```
실물 센서 (/scan · map→base_link · /odom · /imu · 깊이 영상)              시뮬 센서 (sim/: 장면 광선 추적 + 실측 오차 모델)
        │  limo_fire_explore explorer_node.cpp                                        │  sim/src/env_ref.cpp (CPU) · env_kernel.cu (GPU)
        ▼                                                                      ▼
  RobotEnv (limo_fire_explore/robot_env.h) ─────── 같은 함수 ───────  Env (sim/include/env.h · env_api.h)
    지도 갱신(Cartographer 꼴) · 닫힘 판정 · 깊이 카메라 장애물 층(걸음 0.1 s 마다)      (mdp/frontier.h · observations.h · sim/include/obs_ref.h)
        │
        ▼
  후보: cand · 판단 모델 = scripts/cand_view.h (덩어리마다 서는 자리 ≤ 8 곳, 판단 모델 요청엔 앞 6 곳) · slot0 = 환경 hier 후보 (mdp/frontier.h) · fuel = 자기 덩어리 시점
        │
        ▼
  위층 (scripts/low_explore_ctl.h lxc::Ctl — 결정 차례 1 s · 실패 때 바로, 막힌 칸 피하기·목표 이어 가기)
    decider  판단 모델 (scripts/low_decider.h · decider.cpp — 사진·후보 표를 llama-server 에, 비동기 · 실물 기본).
             사건 때만 묻는다(event: 목표에 다가감 · 실패 · 후보 크게 바뀜 · 10 s 최대 간격), 답을 기다리는 동안 지금 목표로
    cand     cand_view 0 번 후보 (규칙 — 시뮬 위층 비교에서 가장 빠름, upper_lora/README "위층 비교")
    fuel     시간 비용 순회의 첫 목적지 (scripts/fuel_upper.h)
    slot0    피하지 않는 첫 후보 (기준선)
        │  LowCmd (목표 칸 · follow)
        ▼
  아래층 규칙 추종기 (sim/include/hier_path.h — A* 경로(plan_ref.h) + Nav2 RPP 식 추종 · 충돌 감시, low_rl.h 계획 차례 · 막힘 신호)
        │
        ▼
  /cmd_vel (실물)  ·  구동 모델 drive.h (시뮬)
```

확인 도구 (sim/tools):
- `low_explore_check` — 같은 판을 CPU 시뮬과 RobotEnv 로 나란히 돌려 위층 결정·준비 뒤 경로·명령·판 끝이 비트까지 같은지 (위층 fuel · slot0 · cand · decider 가짜 응답기).
- `low_explore_replay` · `node_stats` — 실물 노드 녹화(`record:=`, 꼴은 `sim/tools/low_node_rec.h`)를 RobotEnv + 위층으로 다시 돌려 결정 까닭·통계.
- `room_run` — 실물 방 장면에서 노드와 같은 차례로 돌려 노드 꼴로 녹화 (실물 녹화와 같은 잣대로 견줌).
- `env_verify` — GPU == CPU 비트 비교 (`--low --low-explore` 가 합친 탐사 길).
- 많은 판 측정은 `scripts/combine_run`(GPU), 판단 모델 라벨용 갈래 굴리기는 `upper_lora/rollout/dec_rollout`.

탐사 강화학습(신경망 `network/`, PPO `ppo/`, 학습기 `scripts/train.cpp`, 두 층 계획 `ROADMAP.md`)은 2026-11-16 에 그만두고
[archive/rl](../../archive/rl/README.md) 로 옮겼다 — 실물 LIMO 에서 시뮬과의 격차가 컸고 A* 아래층이 훨씬 빠르고 잘 됐다.
시뮬에 남아 있던 아래층 정책 길(정책 명령·관측·자동 목표·사람 추적)과 `mdp/` 의 보상·커리큘럼·그래프 모드도 2026-11-17 에 걷어냈다
(판 끝 코드·끝 조건은 `mdp/episode.h`, 옮긴 목록은 archive/rl README).
