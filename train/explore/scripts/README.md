# scripts — 실행 파일

| 파일 | 내용 |
|---|---|
| `train.cpp` | 학습 러너: 장면 목록으로 환경 N 개 → 24 걸음 롤아웃 → PPO update 반복, 뷰어 실행 폴더(`run.json kind: explore_ppo`)·체크포인트·시험 맵 평가. 빌드·옵션은 아래 |
| `play.cpp` | 학습한 정책 돌려 보기 |
| `frontier_run.cpp` | 프론티어 탐사 기준선을 장면 × 판으로 돌려 뷰어 실행 폴더(`run.json kind: baseline`)로 쓴다. 빌드·옵션은 [baselines/frontier](../baselines/frontier/README.md) · `--modes slot0`(·`slot0_nosafe`) 는 계층형 규칙 정책(늘 0 번 후보)을 같은 판으로 |
| `oracle_run.cpp` | **측정 도구 — 정답을 씀.** 학습 맵 단계마다 정답 격자 경로·참 자세·정답 사람 위치로 모는 oracle(`--policy oracle`, 목표는 로봇 지도 굵은 프론티어, 성공 정의 그대로)과 slot0 규칙 정책을 같은 판으로 돌려 성공 상한을 잰다(`--stage 0..3 --people on\|off --out 판.tsv`). 시뮬은 env_api 로만 몰고 정답은 러너 안에서만 — 학습 환경·관측과 무관. 빌드는 baselines/frontier, 결과는 [mdp/README](../mdp/README.md) "커리큘럼" |

## train (`train.cpp`, 빌드 `CMakeLists.txt`)

```bash
cmake -S train/explore/scripts -B <빌드> -DCMAKE_CUDA_COMPILER=/usr/local/cuda-13.0/bin/nvcc -DCMAKE_CUDA_ARCHITECTURES=86
cmake --build <빌드> -j8 --target train
# 저장소 뿌리에서 (기본 경로가 train/explore/… 상대)
<빌드>/train --name explore_s1 --seed 1 --minutes 120 --eval-every 50 --ckpt-every 50
<빌드>/train --name explore_s1 --resume --minutes 60      # ckpt/model_latest.bin 에서 이어 (실행 폴더도 그 반복으로 되감음)
```

| 옵션 | 기본 | 뜻 |
|---|---|---|
| `--envs` | 2048 | 학습 환경 수 (2048 에서 GPU 5.6 GB) |
| `--iters` · `--minutes` | 0 · 0 | 반복 수·시간 한계 (둘 다 0 이면 1000 반복). 끝·SIGINT·SIGTERM 이면 체크포인트 저장 후 끝 |
| `--name` · `--root` | explore · train/explore/runs | 실행 폴더 `<root>/<name>` |
| `--seed` | 1 | 시뮬·정책 씨앗 |
| `--lr-max` | 1e-3 | KL 적응 학습률 상한 (ppo/README "검증 2": rsl_rl 기본 1e-2 는 장난감에서 critic 이 터짐) |
| `--eval-every` · `--eval-episodes` | 50 · 32 | 시험 맵(`--test`, 기본 scenes/test 의 bookstore·small-house)마다 별도 Env, 결정적 정책, 환경마다 첫 판 |
| `--ckpt-every` | 50 | `ckpt/model_<반복>.bin`, 그리고 10 반복마다 `ckpt/model_latest.bin` |
| `--replay-every` | 25 | 이 반복마다 표본 환경(0~63)에서 새로 시작하는 판 2 개를 재생 파일로 (동시 녹화 최대 8 판 — 끊김 판(12 000 걸음)이 녹화 칸을 오래 잡아도 새 녹화가 계속 붙음) |
| `--list` · `--env` · `--model` | scenes/train/list.tsv · sim/config/env.yaml · sensor_model.yaml | 학습 맵 목록·시뮬 설정 |
| `--hier` | 끔 | 계층형(mdp/README "계층형"): 1 s 마다 프론티어 후보 8 개 중 하나(가린 범주 분포, `xnet::explore_spec_hier`) → 시뮬이 경로 계획·추종. 반복 = 결정 24 개(= 시뮬 240 걸음), γ = 0.999¹⁰, 걸음/s 는 시뮬 걸음 단위. 재생 녹화는 아직 없음(`--replay-every` 무시) · 체크포인트는 기본 구조와 안 섞임(층 이름이 다름) |

- done 매핑: `mdp::cur::terminal` (1 선언 성공·2 벽 충돌·4 선언 실패·5 사람 충돌) → PPO 1 종료, 3 → 끊김(가치로 메움).
- progress.jsonl (반복마다): `reward/<항>`(끝난 판의 항별 합 평균), `rollout/success_rate`·`collision_rate`(벽+사람)·`person_collision_rate`·`declare_short_rate`(선언 실패)·`truncation_rate`·`coverage_mean`·`coverage_at_declare_mean`·`t95_mean`·`t95_reached_rate`·`ep_len_mean`(걸음)·`ep_rew_mean`, `curriculum/stage`(끝난 판 단계 평균), `train/*`(learning_rate·approx_kl·value_loss·policy_gradient_loss·entropy_loss·grad_norm·std), `time/*`(fps·iter_ms·rollout_ms·update_ms + sim_ms·act_ms·host_ms), `gpu/mem_used_mb`, 평가 반복에는 `eval/*`(success_rate·collision_rate·t95_mean·coverage_mean … — frontier 실행의 `eval/frontier_t95_mean` 과 나란히).
- episodes.jsonl: 표본 환경 0~63 의 판마다 한 줄(맵·단계·결과·덮음·t95·항별 보상). evals/test_it<반복>.json: 시험 맵별 성공·충돌·끊김·선언 미달·끝 덮음·t95.
- 재생 파일은 걸음마다 참 자세·지도 자세·잰 속도, 정책 행동 v·vy·ω(물리 단위, 환경이 자르는 대로)·선언 확률·뽑은 선언, 가치 V(s), 보상 항 7 개(`r_<항>`)와 누적 리턴,
  덮음, 사건, 움직이는 사람 원기둥(`h<k>_x/_y/_r`), 참 자세에서 다시 쏜 깨끗한 라이다(frontier_run 과 같은 꼴).
  로봇이 만든 지도(로그 오즈 → 점유 %)와 본 벽 면 칸(덮음 겹)은 10 걸음마다 그 환경 칸만 장치에서 꺼내 넣는다(녹화 중인 판만 — 2048 환경 전체를 안 꺼냄).
- run.json `robot`: 시작할 때 sim 충돌 몸통·센서 자리(env.yaml body, 깊이 카메라 수평 시야는 sensor_model 의 fx·유효 열)와 URDF(`src/limo_fire/limo_fire_description/urdf` 의 xacro 를 그대로 읽음 — 차체 상자·바퀴 4 개·라이다·카메라 자리)를 적는다.
  뷰어 재생은 이것으로 위에서 본 로봇을 그린다. 2026-11-04 값: sim 몸통 0.32 × 0.22 m(nav2 footprint) ↔ URDF 차체 상자 0.13 × 0.12 m(단순 충돌 상자) — 다르다, 화면은 sim 몸통 실선 + URDF 상자 점선.
  바퀴 (±0.10, ±0.065) 지름 0.09·폭 0.045, 라이다 (0.103, 0) 둘이 같음, 카메라 수평 자리 (0.15, −0.005) 둘이 같음(높이·숙임은 sim 이 실측 0.1881 m·6.6°, URDF 0.205 m·0° — 위에서 볼 땐 안 보임).
- 이어 학습은 PPO 체크포인트(가중치·Adam·정규화·반복)만 잇고 시뮬은 새 판으로 시작한다(커리큘럼 단계도 0 부터).
