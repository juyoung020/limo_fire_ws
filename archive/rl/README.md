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
| `train/aim/README.md` | 조준 강화학습 계획(시작 안 함) |
| `src/limo_fire/limo_fire_rl/src/explore_rl_node.cpp` · `config/explore_rl.yaml` · `launch/explore_rl.launch.xml` | 위층 RL 실물 노드 (`/explore_rl`) |
| `src/limo_fire/limo_fire_rl/README.md` | RL 을 빼기 전 패키지 README (explore_rl · 아래층 RL · 확인 표) |
| `docs/assets/explore_rl_*.gif` | 위층 RL 정책 시뮬 영상 (README 에 있던 것) |

같은 커밋에서 지금 코드에서 뺀 것: `low_rl` 노드의 `lower:=rl` · `checkpoint` · `upper:=topic`(RL 아래층에서만 돌던 시험용), `combine_run --policy`(와 그것만 쓰던
`--hold` · `--rule-near` · slot0 갈래), `low_explore_check --ckpt/--lower`, `room_run --ckpt`, RViz 탐사 패널의 `/explore_rl` 항목, 위층 목표 넘기기의 RL 가지(`low_explore_ctl.h`).

**제자리에 남긴 것** (지금 코드가 쓰므로): 시뮬(`train/explore/sim`)과 그 안의 아래층 RL 걸음(`low_rl.h` — 규칙 추종기·막힘 신호·닿음이 같은 구조체를 씀),
`mdp/` 의 보상·커리큘럼(시뮬 걸음이 같이 계산 — 떼면 시뮬 결과가 바뀜), `RobotEnv` 의 아래층 관측 모드와 `low_mov.h`(합친 탐사 노드가 같은 걸음 차례로 부름 —
시뮬과 비트 같음을 지키려고 그대로), 학습 뷰어(`train/explore/viewer` — 프론티어 기준선 실행 폴더도 봄), 위층 판단 모델 LoRA(`train/explore/upper_lora`, 지도 학습 — DPO 부분 포함).

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
- 실물 녹화: `~/limo_bags/1116_*` (RL 아래층 노드 녹화 `1116_fuel_rl_1_node.bin` · `1116_slot0_1_node.bin` 은 녹화 꼴이 같은 `531e7fc` 판의 `low_explore_replay` 로 다시 돌린다 — `2a17a5a` 부터 구조체 크기가 달라 안 읽힘)
