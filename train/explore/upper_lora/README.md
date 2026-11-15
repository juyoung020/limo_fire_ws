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
