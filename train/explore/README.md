# explore — 탐사 시뮬과 위층·아래층 측정

지금 탐사 구성(2026-11-16 실물 시험으로 정함)은 **위층 판단 모델(d1-3B) + 아래층 A* 경로 추종** 이다. 실물 노드는
[`limo_fire_rl` low_rl](../../src/limo_fire/limo_fire_rl/README.md), 같은 코드를 시뮬에서 수천 판 돌려 재는 것이 여기다.

```
explore/
├── sim/         시뮬레이터: 물리·센서·지도 (상태 EnvState). 실측 값은 sim/config/env.yaml, 근거는 sim/analysis
├── mdp/         관측 + 크기 · 판 끝 판정 · 프론티어·닫힘 (시뮬과 실물 노드가 같이 씀)
├── scripts/     combine_run(두 층 합친 탐사 측정) · frontier_run · oracle_run · 위층 규칙(fuel_upper.h · low_explore_ctl.h · low_decider.h)
├── baselines/   비교 기준선: frontier/ = 프론티어 탐사(explore_lite + Nav2 식)
├── upper_lora/  위층 판단 모델 LoRA 학습 (지도 학습)
├── viewer/      실행 폴더 뷰어(trainview): 곡선·표·2D 재생 + 러너가 쓰는 실행 폴더 헤더 include/runfolder.h
├── runs/        실행 폴더 (git 제외)
└── scenes/      시험 장면(.scene, git 제외 — 만드는 명령은 scenes/README · baselines/frontier/README)
```

탐사 강화학습(신경망 `network/`, PPO `ppo/`, 학습기 `scripts/train.cpp`, 두 층 계획 `ROADMAP.md`)은 2026-11-16 에 그만두고
[archive/rl](../../archive/rl/README.md) 로 옮겼다 — 실물 LIMO 에서 시뮬과의 격차가 컸고 A* 아래층이 훨씬 빠르고 잘 됐다.
`mdp/` 의 보상·커리큘럼 코드는 시뮬 걸음(`sim/include/env.h`)이 아직 같이 계산해 그대로 두었다(판 끝 코드·관측과 한 몸이라 떼면 시뮬 결과가 바뀜).
