# upper_lora — 위층 판단 모델(d1-3B) LoRA 학습 (진행 중)

위층 판단 모델(`scripts/combine_run --upper decider --decider-cand fuel`)이 고르는 후보를 더 잘 고르게, **마지막 shortconv 두 층(28·29)에만 LoRA** 를 붙여 학습한다.
앞 층 출력은 한 번만 계산해 캐시하고(얼림), 학습 커널은 C++/CUDA 로 직접 쓴다. 배포는 llama-server `--lora` 어댑터로(변환 도구는 아직).

## 왜 마지막 두 층의 마지막 5 토큰만

d1-3B 는 lfm2(층 30 개)이고 28·29 층은 커널 3 의 인과 shortconv 층이다(llama.cpp `src/models/lfm2.cpp` · `lfm-base.cpp build_lfm_shortconv`).
답(선택지 글자 토큰의 로짓, 서버 `server-context.cpp` 가 묶음 최댓값을 점수로)은 마지막 토큰에서만 읽으므로 29 층 출력의 마지막 줄은 29 층 입력의 마지막 3 줄,
그것은 28 층 입력의 마지막 5 줄에만 달린다. 그래서 28 층 입력(`l_out-27`)의 마지막 5 토큰 은닉(5 × 2048)만 있으면 그 뒤 계산을 정확히 다시 할 수 있다.

## 도구

| 도구 | 하는 일 |
|---|---|
| `d1_cache` | 요청 본문(`combine_run --decider-log` 의 `*.request.json`, 서버에 보내는 것과 같은 꼴)마다 서버 코드(`server_decision_context::fill_task` — llama.cpp `tools/server` 를 그대로 링크)로 프롬프트를 만들고 모델을 돌려 28 층 입력 마지막 5 토큰 · 선택지 글자 토큰 묶음 · 그 로짓 · 정답 번호를 저장 |
| `head_check` | CPU 참조(`d1_ref.h`, double)로 캐시에서 다시 계산한 로짓이 llama.cpp 로짓과 같은지 |
| `lora_train` | GPU(f32, cuBLAS + 손으로 쓴 역전파 커널) LoRA 학습. `--gradcheck` 은 CPU 참조의 유한 차분과 해석 기울기를 견줌 |
| `lora_train --pred p.txt` | 끝난 어댑터로 검증 표본마다 `정답 n p_0 … p_{n−1}` 한 줄 — 차례 두 가지 확률 평균 · 정보 뺀 묶음과의 비교용 (`--epochs 0 --init <어댑터>` 면 평가만) |
| `d1_cache --full-from 27` · `lora_train --attn` | 27 층(마지막 attention)부터 학습: 캐시 옆 `<캐시>.seq` 에 l_out-26 시퀀스 전체(f16), 27 층 q · k · v · o · FFN 에도 LoRA (`--mats27`) — 확인: 27 층 출력이 llama.cpp 28 층 입력과 최대 0.003 차, 기울기 검사 34 칸 최대 1.0e-4, 27 층 어댑터 왕복(lora_export → llama.cpp --lora) 로짓 최대 0.015 차 · 고른 것 4/4 같음 |
| `lora_export` | 어댑터 → llama.cpp LoRA GGUF (`llama-server --lora`, 텐서 `blk.N.….weight.lora_a/b`) |

확인한 것 (2026-11-15): `head_check` — 받은 요청 4 개에서 다시 계산한 로짓과 llama.cpp 로짓 차 최대 0.023(로짓 크기 14~18, Q8_0 활성 양자화 몫), 고른 것 4/4 같음.
`lora_train --gradcheck` — r 2, 층 2 × 행렬 5 × (A · B) 20 칸에서 해석·수치 기울기 상대 차 최대 1.6e-4, 손실 GPU 1.406137 · CPU 1.406138.
왕복 — 표본 4 개를 일부러 모두 "B" 로 외우게 학습(r 4) → `lora_export` → `d1_cache --lora` 로 llama.cpp 가 LoRA 를 켜고 낸 로짓이 CPU 참조(같은 어댑터)와 최대 0.018 차, 고른 것 4/4 같음(모두 B).
LoRA 를 빼고 참조로 계산하면 차 33~36 · 고른 것 0/4 — 어댑터가 실제로 서버 길에 붙는다.

## 데이터

`combine_run --upper decider --decider-url none --decider-cand fuel --decider-input all --decider-log <폴더>` 로 fuel 이 몰며 결정마다 요청(지도 그림 · 카메라 그림 · 후보 표)과
라벨(fuel 이 고른 것)을 모은다. 같은 결정을 후보 차례를 거꾸로(글자·그림 다시) 한 요청도 같이 남겨 차례 쏠림을 줄이는 증강으로 쓴다(정답 번호 = n − 1 − 원래).
학습 장면만(맞추기 장면 tune2, 씨앗 21·22 — 평가 씨앗과 겹치지 않게).

## 빌드

```bash
cmake -S train/explore/upper_lora -B <빌드> -DCMAKE_CUDA_COMPILER=/usr/local/cuda-13.0/bin/nvcc -DCMAKE_CUDA_ARCHITECTURES=86 [-DLLAMA_DIR=~/models/decision/runtime/llama.cpp]
cmake --build <빌드> -j8
```
llama.cpp 는 서버를 돌리는 것과 같은 소스·빌드(`LLAMA_DIR/build`)를 링크한다(서버 코드 라이브러리 `libserver-context.a` 포함).

## 보조 스크립트 (`tools/`, 파이썬 — 데이터 고르기·분석만)

| 스크립트 | 하는 일 |
|---|---|
| `bc_sample.py <로그 폴더> <출력 앞머리> [N]` | `--decider-log` 결정에서 BC 용 N 개를 층별로 뽑음(장면 · 후보 수 · fuel 이 가장 가까운 후보가 아닌 것), 검증은 장면으로 나눔 → `<앞머리>_{train,val}.txt` (원래 · 뒤집은 차례) |
| `mkB.py <앞머리> <출력 폴더>` | 변형 B: `extra_total_time_if_chosen_first_s`(fuel 이 고른 후보는 0 이라 답이 새는 값)와 그 설명을 뺀 사본 |
| `mkpairs2.py <out> <로그> <출력 앞머리> <결정 번호>` | 같은 결정 상태에서 갈래마다 여러 번(`--branch-reseed`) 굴린 결과 → 갈래별 성공 · 끝 시간 평균 · 표준오차, 유의한 선호 쌍만(성공 수 차 ≥ 2, 또는 평균 차 > 2 표준오차), 상태별 "가장 좋은 갈래 − fuel" 분포 |
| `mkabl.py <B 검증 목록> <출력 폴더>` | 변형 B 가 무엇에 기대는지: 검증 50 상태에서 정보 묶음(시간 · 넓이 · 방 · 위험 · 이력 · 방향)을 후보 사이에서 같게 하거나 카메라 · 지도 그림을 회색으로 바꾼 사본 — 캐시 후 `lora_train --epochs 0 --init <B 어댑터>` 로 맞춤이 얼마나 떨어지는지 |

결과 기반 선호 쌍 (2026-11-16, tune2 씨앗 21 · 결정 10 · 갈래 3 × 4 번): 상태 74 개 가운데 다른 갈래가 fuel 을 유의하게 이긴 것 14 · fuel 이 모두를 이긴 것 28,
모두 성공한 상태의 (가장 좋은 갈래 − fuel) 끝 시간 차 중앙값 +1.2 s — 아래층 잡음(성공 ≈ 0.66)이 갈래 결과를 덮어 DPO 는 아래층이 좋아질 때까지 멈춤.

## 확장 설계: 마지막 attention 층까지 학습 (2026-11-16, 잠정 — 아직 안 만듦)

**왜:** BC A(값이 새는 열이 있어 규칙으로는 1.0)가 검증 0.83 · 학습 0.81 에서 멈춤 → 마지막 2 층(conv, 마지막 5 토큰만 봄)만 학습하면 앞 고정 층이 마지막 토큰에 옮겨 둔 것밖에 못 씀.
**모델 구조(GGUF):** 30 층 중 attention 은 2·5·9·13·17·21·24·27 (q 32 머리 · kv 8 머리 · 머리 64, qk-norm, RoPE), 나머지는 shortconv. 마지막 attention 이 27 층.
**프롬프트 길이(BC 3000 요청):** 707 + 231 × 후보 수 토큰(평균 2112, 최대 4403) — 그림 224(카메라 80 · 지도 144), 나머지는 후보 JSON(긴 키 이름).
압축 표(머리줄 한 번 + 후보마다 숫자 한 줄)로 6 후보 상태 표 부분 1669 → 547 토큰 → 프롬프트 ≈ 550 + 50 × 후보 수(평균 ≈ 860, 2.5 배 짧음 — Orin 추론 시간도 줄어듦).

| 안 | 무엇 | GPU 메모리 | 한 에폭(3000 표본) | 디스크 | 일 |
|---|---|---|---|---|---|
| (a) 27 층부터 학습 | `l_out-26` 을 시퀀스 전체 fp16 저장(4 KB/토큰). 27 층(q·k·v·o·FFN) · 28 · 29 LoRA. 27 층은 k·v 만 전체 토큰, q·FFN 은 마지막 5 줄 → **정확**(근사 없음) | ≈ 0.7–1.1 GB | ≈ 0.5–1 분 (k·v 투영 ≈ 11 GFLOP/표본 + 읽기 10 GB) | 압축 프롬프트 ≈ 3.5 MB/표본 → 10.5 GB/변형 (지금 프롬프트면 26 GB) | 캐시 다시 ≈ 6–7 시간/변형(1.2 GB 설정), 코드 ≈ 하루 |
| (a′) 24 층부터 | 같은 캐시 꼴을 `l_out-23` 에서. 24–26 층은 전체 토큰 FFN 까지 | ≈ 1.5–2 GB (미니배치 4 · 기울기 누적) | ≈ 8 분 (≈ 1.3 TFLOP/표본) | 같음 | (a) 코드 그대로, 저장 층만 바꿈 |
| (b) 앞부분을 매 걸음 다시 계산 | llama.cpp 로 0–26 층을 돌려 꼬리만 학습(그림 임베딩은 한 번 저장) | ≈ 3.5 GB 계속 | ≈ 15 분(압축) · 35 분(지금) — 3060 프롬프트 처리 ≈ 3000 토큰/s 잠정 | 그림 임베딩 2.7 GB | 아래층 학습과 GPU 를 같이 못 씀 |
| (c) ggml 역전파 | `examples/training/finetune` 은 전체 파라미터 · 글만 · f32 KV. SSM_CONV · FLASH_ATTN 역전파 없음, LoRA 파라미터 학습 없음, 양자화 가중치 역전파는 f32/f16 가중치 필요 | 5–10 GB | — | — | 몇 주, 지금 GPU 로 불가 |

**추천: (a)**, 압축 프롬프트와 함께. 차례: ① `decider` 압축 표 형식(기본 꺼짐) + 기본 모델 맞춤 확인 → ② `d1_cache --full-from <층>` (전체 시퀀스 fp16) → ③ `lora_train` 에 attention 층 순·역전파(qk-norm · RoPE · GQA) + `--gradcheck` · `head_check` 로 llama.cpp 로짓과 맞춤 확인 →
④ BC A 로 "새는 열을 읽는가"(1.0 근처면 병목이 풀린 것) → BC B. 모자라면 (a′).

## 결과 라벨 갈래 굴리기 (`rollout/dec_rollout`, 판단 모델 위층 + 아래층 rule)

실물 노드와 같은 위층 코드(`scripts/low_explore_ctl.h` · `scripts/low_decider.h`)로 시뮬 판을 돌리며, 판단 모델 자리에 기준 응답기(slot0 — 늘 글자 A)를 두고 요청을 주기·지연대로 보낸다.
고른 요청마다 그 답을 글자 k 로 바꾼 갈래를 반복(갈래 뒤 우연만 다르게) 굴려 후보마다 (끝 코드 · 끝 시각 · 아는 칸 수)를 남긴다. 원시 상태(자세 · 아는 칸 범위 지도 · 후보 칸 · 특징 · 후보 판 `cand_ver`)를
같이 남겨, 후보 글 꼴이 바뀌면 요청만 다시 그린다 (후보 집합이 바뀌면 다시 굴림).

```bash
cmake -S train/explore/upper_lora/rollout -B <빌드> -DCMAKE_CUDA_COMPILER=/usr/local/cuda-13.0/bin/nvcc && cmake --build <빌드> --target dec_rollout
dec_rollout <장면.scene> --seed 1 --out <앞머리> --gpu 256 [--every 3] [--reps 4] [--horizon 600] [--period 3.0] [--latency 2.6] [--people 0]
dec_rollout <장면.scene> --seed 1 --out x --snap-test 600   # 상태 저장·복원 확인 (복원 뒤 400 걸음 상태·지도 해시가 같은가)
```

갈래는 처음부터 다시 돌리지 않고 상태를 복원한다: 시뮬 `Env::env_blob` · `set_env_blob` · `copy_env` (환경 하나 몫 버퍼 전부 — 상태·사람·지도·지나간 길·관측·후보·아래층·작업 판, 환경 하나 약 1.2 MB).
`--gpu N` 이면 기준 판을 GPU 환경 하나로 돌려 고른 요청마다 상태를 뜨고, 갈래를 GPU 환경 N 칸에 넣어 한꺼번에 굴린다 (갈래의 위층은 칸마다 호스트, 요청은 글만).

잰 것 (2026-11-17, procthor_00016 · 씨앗 1 · 사람 0): CPU 환경 하나 걸음 8.4 ms (시뮬 걸음 7.8 — 센서 광선·지도, 경로 계획 0.35, 위층 0.11).
GPU 길: 기준 판 106 s 를 2.7 s, 갈래 600 판(요청 28 개 × 후보 × 4 반복, 지평 600 s)을 칸 256 개로 31 s — 환경 걸음 약 13,000 / s (CPU 한 코어의 약 110 배).
확인: 글자 A · 반복 0 갈래(바꾼 것 없음)의 결과가 기준 판과 같음 (106.0 s, 요청 3 개 모두), `--snap-test` 복원 뒤 400 걸음 해시 같음 (사람 0 · 2).

갈래 굴리기 후보 (2026-11-17): 실물 노드 upper:=decider 와 같은 후보(`scripts/cand_view.h`, Ctl use_cand · 판단 모델 Upper cand)로 돈다. 줄마다 `cand_ver` "cand-v4-967adc5"(후보 이득 굵은 칸 · 막 실패한 후보 뒤로 · 1 m 안 2 번 실패 무리 버림 · 20 s 길 줄지 않으면 실패 — 그 전 판 출력은 다시 굴림),
후보마다 cand::Cand 필드 전부(안정 번호 · 기억 필드 포함)와 판 기억 `mem`(trail · goals · fails)을, `.map.bin` 에 지금 지도와 5 · 15 · 30 s 전 지도(같은 범위)를 남겨 그림 꼴 A · B 를 나중에 다시 그린다.
갈래 칸은 후보를 만들 걸음(위층 결정 차례 · 답 · 보냄)에만 그 칸의 지도 · 지나간 길 · 깊이 층을 장치에서 꺼낸다. 확인: 글자 A · 반복 0 갈래 = 기준 판 (122.0 s, 요청 3 개).

요청 차례 `--mode event|periodic` (기본 event = 실물 노드가 쓰는 것, 줄마다 `mode`): event 는 1 s 결정 차례마다 후보를 새로 만들어 다가감 · 실패 · 후보 바뀜 · 최대 간격을 본다.
갈래 칸 위층은 `--threads N`(기본 8) 스레드로 나눠 돈다 — 칸마다 Ctl · Upper · 기준 응답기가 따로라 출력이 한 스레드와 바이트 같음 (procthor_00016 · 씨앗 1, 1 · 10 스레드 jsonl `cmp` 같음).
잰 것 (2026-11-17, event, 지평 180 s, PC 부하 높을 때): 갈래 시간의 약 80 % 가 위층 — 1 s 마다 후보 만들기 `cand::build` 한 번 11 ms (procthor_00016), 큰 HSSD 집은 약 60–90 ms. GPU 걸음은 약 15 %.

## 위층 비교 (미세조정 전 기본 모델, 닫힌 고리 시뮬 — `tools/abl_run1.sh` · `tools/abl_metrics.py`)

지표: 굵은 칸 덮음 0.5 m @20 %(`scripts/coverage.h`)의 AUC (90 · 180 s, 판이 끝나면 마지막 값을 이어 감), 50/80/90 % 닿는 시각. 같은 장면·씨앗끼리 짝 차이 ± SE.
시험 8 장면: HSSD 큰 집 5 채(hssd_102816009_l0 · 103997541_171030615_l0 · 103997718_171030855_l0 · 104348181_171513120_l0 · 105515175_173104107_l0 — 모든 층을 라벨 굴리기에서 뺌), 실물 방 lab_1116b, bookstore, small-house. 씨앗 4–7, 사람 0. 판단 모델 d1-3B Q8 (PC 서버, 지연 2.6 s 고정).
빌드 1f275bf (+ cand_ver 커밋) — 아래층·위층이 바뀌면(예: 967adc5) 숫자가 달라짐. 2026-11-17 잰 것 (각 32 판):

| 위층 | AUC90 | AUC180 | 120 s 덮음 | 90 % 시각 (s) | AUC180 짝 차이 (− 규칙) |
|---|---|---|---|---|---|
| 규칙 `cand` | 0.532 | 0.677 | 0.792 | 134 | — |
| 판단 모델 event (후보 다) | 0.461 | 0.599 | 0.721 | 189 | −0.078 ± 0.019 |
| 판단 모델 주기 3 s | 0.456 | 0.598 | 0.716 | 176 | −0.079 ± 0.014 |
| 판단 모델 event · 후보 최대 4 | 0.480 | 0.614 | 0.718 | 181 | −0.063 ± 0.017 |
| 판단 모델 event · 후보 최대 6 | 0.481 | 0.626 | 0.753 | 184 | −0.052 ± 0.015 |
| 판단 모델 event · 후보 거리순 | 0.487 | 0.625 | 0.739 | 169 | −0.053 ± 0.015 |
| 판단 모델 event · 최대 6 + 거리순 | 0.482 | 0.622 | 0.734 | 172 | −0.055 ± 0.015 |
| `fuel` | 0.431 | 0.559 | 0.661 | 228 | −0.118 ± 0.018 |

기본 모델은 규칙보다 느리고 fuel 보다는 빠르다. 차이는 큰 집(HSSD 20 쌍: event −0.094, 최대 6 −0.061)과 처음 1 분(쉬운 두 장면 AUC90 −0.14)에서 나고, 실물 방은 차이가 거의 없다(1 분 안에 거의 다 보임).
후보를 줄이거나 거리순으로 주면 event 기본보다 AUC180 +0.025 안팎 낫지만 둘을 합쳐도 더 늘지 않음. 끝 코드: 규칙은 32 판 중 30 판이 '대충 다 앎'·닫힘, 판단 모델 event 는 24 판 (나머지 부딪힘 4 · 진척 없음 4).

## 요청 꼴과 지연 (2026-11-17, 로봇 Orin · Q8 · `-lm dio` · `-c 4096 -np 1`, 실물 노드 robot.launch 가 도는 중에, 5 번 중앙값)

실물 시험(5cd0a0b 기본: 긴 꼴 + 지도 384 px + 카메라 2 × 2, images-last 없음)에서 답 하나 5 s 가 나와 꼴별로 쟀다.
요청은 같은 HSSD 상태 두 개(라벨 굴리기 원시 상태를 `dec_rollout --render` 로 다시 그림): warm = 같은 꼴의 앞 요청 뒤 다음 요청(실제 연속 결정), cold = 다른 요청 뒤 처음.

| 꼴 | 토큰 (PC) | images-last 끔 cold / warm | images-last 켬 cold / warm |
|---|---|---|---|
| 긴 꼴 + 지도 384 + 카메라 2 × 2 (옛 노드 기본) | 2789 | 5605 / 5611 ms | 5656 / 5031 ms |
| 긴 꼴 + 지도 384 | 2502 | 4427 / 4428 | 4461 / 3881 |
| 짧은 키 + 지도 384 (옛 compact) | 1587 | 3070 / 3057 | 3138 / 2726 |
| 짧은 키 + 힌트를 후보 앞 (지도 320) | 1543 | 4060 / 3142 (흔들림 큼) | 2991 / 2354 |
| **v1 꼴: 짧은 키 · 힌트 앞 · 최근 목표 목록 뺌 · 결정 기록 4 · 지도 288** | 1140 | 2189 / 2208 | **2203 / 1582** |
| 같은 꼴, 지도 256 · 320 | 1123 · 1159 | 2087 / 2074 · 2328 / 2327 | 2207 / 1590 · 2351 / 1714 |

토큰마다 약 1.6 ms (Orin 프롬프트 처리) — 차이는 거의 글 길이에서 난다. 그림은 (px/32)² 토큰(288 → 81)이라 작게 줄여도 이득이 작다.
images-last + checkpoint 패치는 바뀌지 않는 앞부분(템플릿 · 설명 글 · 힌트)을 다시 안 써 warm 에서 약 0.6 s 를 줄인다.
그래서 기본을 v1 꼴로 바꿈 (`scripts/low_decider.h` Params: compact · px 288 · log_k 4 · cam 0 — 노드 `config/low_rl.yaml` · `low_explore_check` · `dec_rollout` 이 같은 값):
최근 목표 목록은 지도 사진의 ✓ · × 표시와 결정 기록이 같은 것을 말해 뺐고, 카메라 2 × 2 는 라벨 시뮬에 실물 카메라가 없어 v1 학습에서 빠짐(카메라 판은 v2 — 라벨에 키프레임 자세가 있음).
서버는 `LLAMA_DECISION_IMAGES_LAST=1` 로 띄우고(노드 README), 학습 캐시(`d1_cache`)도 같은 값으로 만든다.
로봇 스크립트: 라벨 상태를 꼴마다 다시 그려 올리고 서버 설정마다 띄워 재고 PID 로 끔 (결과 로봇 `~/decider_test/lat_results/lat_fmt*_1117.tsv`).

`dec_rollout --render <jsonl…> --render-out <폴더> [--compact 0|1] [--px N] [--log-k K] [--max-cands K] [--render-check]`: 라벨 굴리기가 남긴 원시 상태(후보 필드 · 판 기억 · 결정 · 지도 .map.bin · 자세)를
노드와 같은 함수(`lxd::Upper::render_cand` — 노드 `build_cand` 가 부르는 것)로 다시 그림. 그 뒤로 jsonl 은 float 를 `%.9g` 로 적어 다시 그린 요청이 원래와 바이트 같음
(procthor_00016 요청 8/8). 그 전 라벨(labels_61628bb, 자릿수 1e-3)은 604 개 중 196 개가 바이트 같고, 나머지는 frontier_width 등 0.1 반올림 경계와 지도 화살표 위치의 작은 차이.

## llama-server 패치 (판단 모델 앞부분 캐시 — `patches/`)

기준: llama.cpp 31953bc (판단 모델 판, `tools/server/server-context.cpp` md5 fa397489…, PC `~/models/decision/runtime/llama.cpp` · 로봇 `~/decider_test/llama.cpp` 같은 소스). `patch -p1 < …` 뒤 `cmake --build build --target llama-server`.
- `llama-server-decision-checkpoint.patch`: 판단 요청(DECISION)에도 context checkpoint 를 128 토큰마다 만든다 (원래는 COMPLETION 만 — LFM2 는 short conv 상태라 checkpoint 없이는 매 요청을 0 부터 다시 처리). 다음 요청은 같은 앞부분의 가장 긴 checkpoint 에서 이어 간다.
- `llama-server-decision-images-last.patch`: `LLAMA_DECISION_IMAGES_LAST=1` 이면 systemone 템플릿에서 그림을 상태 뒤 · 질문 앞으로 옮긴다 (원래는 그림이 맨 앞이라 결정마다 바뀌는 지도 그림 때문에 같은 앞부분이 5 토큰뿐). 모델이 보는 차례가 바뀌므로 학습 캐시(d1_cache — 같은 서버 코드를 링크)와 서비스에 같은 값을 쓴다.
- 확인 (PC, 같은 요청 차례): 차갑게 처리한 것과 checkpoint 를 복원해 처리한 것의 선택지 확률이 같음 (lat_comp2 B 0.5119291607677683 두 쪽). 128 토큰마다 끊어 처리하므로 패치 전과는 확률이 소수 둘째 자리에서 조금 다름 (0.4977 → 0.5119) — 학습·서비스를 같은 판으로.
- 로봇 지연 (Q8, -lm dio, 2026-11-16): 압축 요청 차갑게 2429 ms → 앞 요청과 앞부분이 같으면 2242 ms, 그림 128 토큰이면 1953 → 1770 ms. 같은 앞부분이 고정 글(약 300 토큰)뿐이라 몫이 작음 — 바뀌는 글(후보 · 기록)을 줄이는 쪽이 더 큼.
- 양자화 (2026-11-17, 로봇 Orin · -lm dio · images-last · 압축 요청 1203 토큰 + `--image-max-tokens 128`, 5 번 중앙값, `patches/robot_latency_quant.sh`): 다른 요청 뒤(cold) / 값 하나 바뀐 요청(warm, 앞부분 캐시)
  Q8_0 1924 / 1743 ms · Q4_K_M 2007 / 1811 ms · Q5_K_M 2092 / 1889 ms — 결정 요청은 출력 없이 prefill 만이라 K 양자화가 Q8 보다 느림 (메모리만 1.2 GB 덜 씀). 지연 목적으로는 Q8 유지.
  Q8 과 견준 고르기 (PC, 지금 꼴 요청 133 개 — 학습 장면 4 개 + HSSD 스모크, Q8 두 번은 완전히 같음): 1 위 일치 Q4_K_M 0.910 · Q5_K_M 0.955 · Q6_K 0.947, KL 0.012 · 0.003 · 0.001.
  양자화본은 Q8 에서 다시 양자화(`llama-quantize --allow-requantize`) — HF 원본(d1 전용 구조)에서 만들지 않음.
- `robot_latency_table.sh`: 로봇 `~/decider_test` 에서 설정마다 서버를 띄워 요청 꼴별로 재는 스크립트.
