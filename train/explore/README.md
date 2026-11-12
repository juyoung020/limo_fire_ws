# explore — 탐사 RL

시뮬레이터(`sim`) → 무엇을 배우나(`mdp`) → 신경망(`network`) → 어떻게 배우나(`ppo`).

```
explore/
├── sim/        시뮬레이터: 물리·센서·지도 (상태 EnvState). 실측 값은 sim/config/env.yaml
├── mdp/        관측 + 크기 · 보상 식 + 가중치 · 판 끝 판정 + 커리큘럼 단계
├── network/    CNN 혼합 정책·가치 (2D 마다 CNN + MLP)
├── ppo/        롤아웃·GAE · PPO 손실·갱신·Adam + 하이퍼파라미터 · 학습 루프
├── scripts/    train · play · frontier_run(프론티어 기준선을 돌려 실행 폴더로)
├── baselines/  비교 기준선: frontier/ = 프론티어 탐사(explore_lite + Nav2 식). RL 이 이겨야 할 95 % 도달 시간
├── viewer/     학습 뷰어(trainview): 곡선·표·2D 재생·비교 + 러너가 쓰는 실행 폴더 헤더 include/runfolder.h
├── runs/       실험마다 체크포인트·로그·그때의 mdp/·ppo.h 사본·커밋 해시 (git 제외)
└── scenes/     시험 장면(.scene, git 제외 — 만드는 명령은 baselines/frontier/README)
```

지금 방향(위층 = 목적지 고르기, 아래층 = 조작)과 두 층의 입력·출력·단계는 [ROADMAP.md](ROADMAP.md).
학습을 보는 화면은 [viewer/](viewer/README.md): 러너가 `viewer/include/runfolder.h` 로 `runs/<실험>/` 에 쓰면 `trainview --root train/explore/runs` 가 곡선·판 재생을 보여 준다.
구성은 학습기 쪽은 rsl_rl(algorithms·modules·storage·runners), 과제 쪽은 Isaac Lab(`<과제>/mdp`) 을 줄여 따랐다.
보상·관측 크기·끝 조건·PPO 하이퍼파라미터는 yaml 이 아니라 코드 안에 둔다. 바꾸면 다시 컴파일하고 체크포인트에서 이어 학습한다.
