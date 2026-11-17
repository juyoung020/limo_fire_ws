# archive/rl — 그만둔 탐사 강화학습 (2026-11-16)

탐사를 강화학습(PPO) 두 층으로 하려던 코드·문서. **2026-11-16 에 그만뒀다.**

## 왜 그만뒀나

- 실물 LIMO 에서 시뮬과의 격차(sim-to-real)가 컸다. 아래층 RL 정책(low_w3 36000)은 실물 방에서 멈춤·제자리 회전이 잦았다
  (`1116_fuel_rl_1` — 걸음의 77 % 가 v ≈ 0, `train/explore/scripts/README.md` "실물 1116 진단").
- 아래층을 A* 경로 추종(시뮬 규칙 추종기와 같은 코드)으로 바꾸니 훨씬 빠르고 잘 됐다.
- 같은 날 실물에서 네 조합(fuel+RL · 가까운 후보+RL · 판단 모델+RL · 판단 모델+A*)을 돌려 **위층 판단 모델(`upper:=decider`) + 아래층 A*** 가
  가장 잘 됐다(녹화 `~/limo_bags/1116_*`). 이것이 현장 구성이 됐다 — `src/limo_fire/limo_fire_rl/README.md` "합친 탐사".

## 마지막 상태·결과

| 무엇 | 결과 | 출처 (이 폴더 안) |
|---|---|---|
| 아래층 RL low_w3 반복 36000 (`runs/low_w3/ckpt/model_36000.bin`, md5 `a43b5f95…` — 로봇에는 `~/ckpt/low_w3_36000.bin`) | 검증 세트 2: 옛 물리 벽 0.197 · 사람 0.337 /100 m · 도달 0.922 · 속도 0.661 m/s, 실물형 물리(slip1) 벽 0.178 · 사람 0.381. 통과 기준(벽 ≤ 0.20 · 사람 ≤ 0.10 · 도달 ≥ 0.90 · 속도 ≥ 0.676) **미달** (사람·속도) | `src/limo_fire/limo_fire_rl/README.md`, `train/explore/scripts/README.md` "아래층 판정" |
| 시뮬 합친 탐사 위층 fuel + 아래층 low_w3 36000 (시험 4 장면 × 씨앗 4 · 5 × 12 판) | 성공 0.792 · 끝 시간 134.9 s (fuel 고치기 전 0.708) | 지금 `train/explore/scripts/README.md` "실물 1116 진단과 고침" |
| 위층 RL explore_h4 반복 2511 (실물 노드 `explore_rl`) | 시험 맵 bookstore · small-house 성공 32/32 (학습기 평가 it2500: 0.94 · 1.00) | `src/limo_fire/limo_fire_rl/README.md` "확인 (2026-11-11)" |
| 위층 RL explore_h5 반복 3350 vs 가장 가까운 후보 규칙(slot0) | 157.4 s · 성공 0.740 대 slot0 150.2 s · 0.771 — 규칙을 못 넘음 | 지금 `train/explore/scripts/README.md` |
| 실물 관측 == 시뮬 | `rl_node_check` · `low_node_check` · `low_node_replay` · `low_mov_replay` 모두 비트 같음 (x86 · Orin) | `src/limo_fire/limo_fire_rl/README.md` |

계획·단계·통과 조건은 `train/explore/ROADMAP.md`(두 층, 2026-11-12), 첫 계획은 `../explore_roadmap_1104/`.
학습기·신경망·PPO 세부는 `train/explore/ppo/README.md` · `train/explore/network/README.md`, 그때 실행기 설명은 `train/explore/scripts/README.md`.

## 옮긴 것 (원래 경로 = 이 폴더 안 경로에서 `archive/rl/` 를 뗀 것)

| 경로 | 무엇 |
|---|---|
| `train/explore/ppo/` | PPO 학습기 · 하이퍼파라미터 · 기울기 검사 · 체크포인트 도구(`ckpt_add_*`, `ckpt_drop_priv`) · CUDA 그래프 갱신 |
| `train/explore/network/` | 정책·가치 신경망(CNN·GRU·슬롯 머리·그래프 어텐션), 실물 CPU 정방향 `policy_cpu.h`(LowPolicy) · `graph_policy_cpu.h`, ONNX 내보내기 |
| `train/explore/scripts/train.cpp` · `low_run.cpp`(+ `low_policy_run`) · `hybrid_eval.cpp` · `ckpt_meta.h` · `tools/judge_low.py` | 학습 러너 · 아래층 단독 과제 측정 · RL+프론티어 감시 평가 · 아래층 판정 |
| `train/explore/mdp/ckpt_cfg.h` · `sim/include/low_ckpt_cfg.h` | 체크포인트 관측 설정 꼬리 |
| `train/explore/mdp/tools/reward_probe.cu` | 보상 해킹 시험 |
| `train/explore/sim/tools/rl_node_check.cpp` · `low_node_check.cpp` · `low_node_replay.cpp` · `contact_eval.cpp` | 실물 RL 노드 관측 == 시뮬 확인, 닿음 센서 일치도 |
| `train/explore/ROADMAP.md` | 두 층 RL 계획 |
| `src/limo_fire/limo_fire_rl/src/explore_rl_node.cpp` · `config/explore_rl.yaml` · `launch/explore_rl.launch.xml` | 위층 RL 실물 노드 (`/explore_rl`) |
| `src/limo_fire/limo_fire_rl/README.md` | RL 을 빼기 전 패키지 README (explore_rl · 아래층 RL · 확인 표) |
| `docs/assets/explore_rl_*.gif` | 위층 RL 정책 시뮬 영상 (README 에 있던 것) |

2026-11-17 정리(2 단계 — 되살리려면 그 앞 커밋 `e5a7c74`)에서 더 옮긴 것:

| 경로 | 무엇 |
|---|---|
| `train/explore/mdp/graph.h` · `sim/include/graph_env.h` · `sim/tools/graph_check.cu` | 그래프 모드(HEADER·ARiADNE 식 노드 그래프 관측·결정) — 시뮬 커널·참조판·`env_verify --graph`·`env_bench --graph`·env.yaml `action.graph` 도 뺌 |
| `train/explore/scripts/oracle_run.cpp` | 정답 지도를 아는 허용 상한 (커리큘럼 문턱 측정) |
| `train/explore/mdp/rewards.h` · `curriculums.h` | 보상 항·커리큘럼(단계 오르내림) — 판 끝 코드·끝 조건은 `mdp/episode.h` 로 남김. 시뮬의 보상 버퍼·`Env::get_reward`·`get_terms`·단계별 맵 고르기·오차 배율도 뺌 (`get_done` 으로) |
| `sim/include/low_rl.h` 의 아래층 관측(`lowrl_pack` · `lowrl_img`, `mdp::low`) | 그만둔 정책의 입력 꼴 — 실물 노드 녹화(`sim/tools/low_node_rec.h`)는 판 2 로 바꿈 (지금 판 4 — 판 1 녹화는 새 코드가 RL 칸을 건너뛰며 읽음, 그때 코드로 읽으려면 `e5a7c74`). EnvCfg 의 빈 칸(`gone_rl_*`)·커리큘럼·그래프·단독 과제 칸도 지움 |
| (코드에서만 뺌) | `frontier_run` 의 계층형 규칙 정책 판(`slot0` · `graph_*` · `cand_*` — 기준선 `nav2` · `fast` 는 그대로), 아래층 단독 과제 모드(`low_explore 0` — 목표 수·시간으로 판을 끝내던 학습 판) |

같은 커밋에서 지금 코드에서 뺀 것: `low_rl` 노드의 `lower:=rl` · `checkpoint` · `upper:=topic`(RL 아래층에서만 돌던 시험용), `combine_run --policy`(와 그것만 쓰던
`--hold` · `--rule-near` · slot0 갈래), `low_explore_check --ckpt/--lower`, `room_run --ckpt`, RViz 탐사 패널의 `/explore_rl` 항목, 위층 목표 넘기기의 RL 가지(`low_explore_ctl.h`).

**제자리에 남긴 것** (지금 코드가 쓰므로): 시뮬(`train/explore/sim`)과 그 안의 걸음 차례 `low_rl.h`(이름만 RL — 경로 계획 차례 · 막힘 신호 · 닿음),
학습 뷰어(`train/explore/viewer` — 프론티어 기준선 실행 폴더도 봄), 위층 판단 모델 LoRA(`train/explore/upper_lora`, 지도 학습 — DPO 부분 포함).
처음(2026-11-16)에는 `mdp/` 보상·커리큘럼과 `RobotEnv` 아래층 관측 모드 · `low_mov.h` 도 남겼으나 2026-11-17 정리(위 표 · `e5a7c74`)에서 뺐다.

## 되살리기

옮기기 바로 앞 커밋은 **`b48688a`** (origin/main, 2026-11-16). 그 판에서는 그대로 빌드·실행된다:

```bash
git worktree add ../limo_rl b48688a          # 또는 git checkout b48688a -- <경로>
cmake -S ../limo_rl/train/explore/scripts -B build_rl -DCMAKE_CUDA_COMPILER=/usr/local/cuda-13.0/bin/nvcc -DCMAKE_CUDA_ARCHITECTURES=86
cmake --build build_rl -j8 --target train low_policy_run combine_run
```

이 폴더의 파일은 원래 자리의 상대 경로(`../../../cmake/cuda_strict.cmake`, `../network` 등)를 그대로 가리켜 여기서는 빌드되지 않는다(`archive/` 는 빌드에서 빠짐).

## 저장소 밖 데이터 (git 제외 — 옮기지 않음)

- 학습 실행 폴더: `train/explore/runs/` (이 PC 약 38 GB — `explore_h*` 위층, `low_*` 아래층, `low_w3/ckpt/model_*.bin` 등 체크포인트·로그·평가)
- 로봇의 체크포인트: `~/ckpt/low_w3_36000.bin`
- 실물 녹화: `~/limo_bags/1116_*` (RL 아래층 노드 녹화 `1116_fuel_rl_1_node.bin` · `1116_slot0_1_node.bin` 은 지금 `low_explore_replay` 로도 읽히지만 명령까지 그때와 같게 다시 돌리려면 `531e7fc` 판으로)
