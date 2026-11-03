# scripts — 실행 파일

| 파일 | 내용 |
|---|---|
| `train.cpp` | 학습 러너: 장면 목록으로 환경 N 개 → 24 걸음 롤아웃 → PPO update 반복, 뷰어 실행 폴더(`run.json kind: explore_ppo`)·체크포인트·시험 맵 평가. 빌드·옵션은 아래 |
| `play.cpp` | 학습한 정책 돌려 보기 |
| `frontier_run.cpp` | 프론티어 탐사 기준선을 장면 × 판으로 돌려 뷰어 실행 폴더(`run.json kind: baseline`)로 쓴다. 빌드·옵션은 [baselines/frontier](../baselines/frontier/README.md) |

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
| `--replay-every` | 25 | 이 반복마다 표본 환경(0~63)에서 새로 시작하는 판 2 개를 재생 파일로 |
| `--list` · `--env` · `--model` | scenes/train/list.tsv · sim/config/env.yaml · sensor_model.yaml | 학습 맵 목록·시뮬 설정 |

- done 매핑: `mdp::cur::terminal` (1 선언 성공·2 벽 충돌·4 선언 실패·5 사람 충돌) → PPO 1 종료, 3 → 끊김(가치로 메움).
- progress.jsonl (반복마다): `reward/<항>`(끝난 판의 항별 합 평균), `rollout/success_rate`·`collision_rate`(벽+사람)·`person_collision_rate`·`declare_short_rate`(선언 실패)·`truncation_rate`·`coverage_mean`·`coverage_at_declare_mean`·`t95_mean`·`t95_reached_rate`·`ep_len_mean`(걸음)·`ep_rew_mean`, `curriculum/stage`(끝난 판 단계 평균), `train/*`(learning_rate·approx_kl·value_loss·policy_gradient_loss·entropy_loss·grad_norm·std), `time/*`(fps·iter_ms·rollout_ms·update_ms + sim_ms·act_ms·host_ms), `gpu/mem_used_mb`, 평가 반복에는 `eval/*`(success_rate·collision_rate·t95_mean·coverage_mean … — frontier 실행의 `eval/frontier_t95_mean` 과 나란히).
- episodes.jsonl: 표본 환경 0~63 의 판마다 한 줄(맵·단계·결과·덮음·t95·항별 보상). evals/test_it<반복>.json: 시험 맵별 성공·충돌·끊김·선언 미달·끝 덮음·t95.
- 재생 파일은 참 자세·지도 자세·행동·덮음 + 참 자세에서 다시 쏜 깨끗한 라이다(frontier_run 과 같은 꼴). 관측 지도·덮은 칸 겹은 안 넣음(환경 2048 개 지도 전체를 매 걸음 꺼내는 비용).
- 이어 학습은 PPO 체크포인트(가중치·Adam·정규화·반복)만 잇고 시뮬은 새 판으로 시작한다(커리큘럼 단계도 0 부터).
