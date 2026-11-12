# 판단 전용 모델 정리

2026-11-12 조사. 두 갈래가 있다.

1. **판단 모델(decision model, System One 모델)** — 글을 쓰지 않는다. 상태와 정해진 꼴의 질문 몇 개를 주면 한 번의 계산으로 고르기·점수·예/아니오 확률을 낸다. 수 ms ~ 수백 ms 로 빠르다.
2. **몸을 가진 추론 모델(embodied reasoning)** — 영상을 보고 글로 계획을 쓴다. 수백 ms ~ 수 초로 느리다(이 지연은 출처 값이 아니라 추정).

수치·성능은 각 모델 저자나 순위표의 값이고 우리가 돌려 본 것은 없다. 쓰기 전에 라이선스 파일을 직접 확인한다.

## 1. 판단 모델 (System One)

처음 나온 것은 TypeSafe AI 의 **Jev** 로, 가중치를 내지 않고 API 로만 판다(지금 Jev 1.14). 그 뒤 가중치를 공개한 모델이 수십 개 나왔다.
순위는 Decision Index(비공개 문제로 채점, Hugging Face 순위표)로 견준다. 아래는 0.4.1 판, 2026-11-10 기준.

**성능 순**

| 순위 | 모델 | 만든 곳 | 점수 | 크기 | 가중치 · 라이선스 | 지연 (저자 표기) |
|---|---|---|---|---|---|---|
| 1 | REAX 1.3 | REAX | 74.0 | 57B(출처마다 다름), FP8 로 약 115 GB | 공개, Apache-2.0(섞인 원본 라이선스 유지 조건) | 중앙값 207 ms (RTX PRO 6000 한 장) |
| 2 | GLiDE 27B | Fastino | 73.4 | 27B, 52 GB | 공개, Fastino Community License(큰 회사 상업 이용은 별도 계약 — 제3자 소개 글 기준, 본문 미확인) | 중앙값 112 ms (같은 GPU) |
| 3 | KnowLine 31B Gen1.5 | PelaAI | 71.9 | 31B | 확인 못 함 | — |
| 55 | Jev 1.14 | TypeSafe AI | 60.2 | 비공개 | API 전용 | — |

1 · 2 위는 0.6 점 차라 사실상 같은 무리다. 윗자리는 거의 다 27B ~ 31B 라 로봇에 못 싣는다.

**작고 빠른 것** (같은 순위표 점수)

| 모델 | 점수 | 크기 · 메모리 | 지연 (저자 표기) | 라이선스 |
|---|---|---|---|---|
| KnowLine 4B Gen5.5 | 56.0 | 4B | — | 확인 못 함 |
| Perplexity Decider Flash v1.2 | 54.3 | 4B | — | 27B 판은 Apache-2.0, 4B 판은 확인 못 함 |
| GLiDE 2B | 43.7 | 2B, 3.8 GB | 18 ms (RTX PRO 6000) | Fastino Community License |
| GLiDE 0.8B | 36.4 | 0.8B, 1.5 GB | 19 ms | 〃 |
| GLiDE 0.4B | 33.8 | 0.4B, 0.8 GB | 7 ms | 〃 |
| LiquidAI d1-3B | 31.1 | 3B, 글 + 사진 | **Jetson Orin Nano 에서 질문 1 개 50 ms, 384 px 사진 202 ms** (RTX 4090 은 8 ms) | 공개("LFM License", 본문 확인 필요) |
| Decider 4B / 2B | 36.3 / 23.6 | 4B / 2B | — | Apache-2.0 (Mapika) |

- 우리 로봇 컴퓨터(Jetson Orin Nano)에서 잰 숫자가 있는 것은 **d1-3B 하나뿐**이다. 나머지 지연은 데스크톱·서버 GPU 값이라 그대로 옮길 수 없다.
- 4B 급이 점수 54 ~ 56 으로 "작으면서 성능 좋은" 자리다. Orin Nano 에 올라가는지는 확인 못 했다.
- 사진 입력: d1-3B 는 됨. GLiDE 는 큰 판(27B · 9B)에서만 확인됨.

**우리 탐사에 쓴다면**: 이 갈래는 빠르기로는 위층(1 Hz) 자리에 들어올 수 있다. 다만 훈련 대상이 글·사진 위의 분류·라우팅 판단이지 점유 지도 위의 경로 선택이 아니다. 그대로 쓰면 맞을지 알 수 없고, 쓰려면 우리 과제로 미세조정해 지금 PPO 위층과 같은 조건에서 견줘야 한다.

## 2. 몸을 가진 추론 모델

우리 탐사 RL 로 치면 위층보다 한 단 위, FARE 가 LLM 에 맡긴 자리([papers/header_fare_diff.md](papers/header_fare_diff.md))에 해당한다.

### 가중치를 받을 수 없는 것 (API 전용)

| 모델 | 만든 곳 | 상태 | 하는 일 |
|---|---|---|---|
| Gemini Robotics ER 2 | Google DeepMind | 2026-07-30 공개. Gemini API · AI Studio 로만 제공(`gemini-robotics-er-2-preview`, 실시간용 `…-streaming-preview`). 가중치 공개 언급 없음. 전 판 ER 1.6 은 2026-08-31 종료 | 공간 추론, 여러 단계 작업 계획, 영상으로 진행 추적, 도구 호출. 모터 제어는 아래 VLA 에 넘김 |

로봇이 인터넷에 붙어 있어야 하고 응답 지연이 망에 달려 있다 — 화재 현장 탐사에는 그대로 쓰기 어렵다.

### 가중치가 공개된 것

| 모델 | 만든 곳 | 크기 | 라이선스 | 하는 일 | 필요한 GPU (저자 표기) |
|---|---|---|---|---|---|
| Cosmos-Reason2 | NVIDIA | 2B · 8B · 32B (Qwen3-VL 기반) | 코드 Apache-2.0, 가중치 NVIDIA Open Model License | 물리 상식 추론, 다음 행동 계획, 영상 이해·시간 위치 찾기 | 2B 24 GB, 8B 32 GB. Jetson AGX Thor 에서 시험됨 |
| Cosmos 3 | NVIDIA | 4B · 16B · 64B (출처마다 다름, 4B 는 미공개라는 글도 있음) | OpenMDW-1.1 | Cosmos-Reason2 의 후속. NVIDIA 가 새 작업은 이쪽을 쓰라고 안내 | 확인 못 함 |
| RoboBrain 2.5 | BAAI | 4B · 8B | Apache-2.0 | 가리키기(pointing), 쓸모(affordance) 예측, 내비게이션, 3D 궤적(x·y·깊이), 진행도 추정 | 표기 없음 |
| Embodied-R1.5 | 톈진대 등 | 8B (Qwen3-VL-8B 기반) | 표기 못 찾음 | 계획 → 위치 찍기 → 고치기 순환, 가리키기. VLA 로 미세조정 가능 | 표기 없음 |
| Molmo 2-ER / MolmoAct 2 | Allen AI | 표기 없음 | 표기 못 찾음(가중치·학습 코드·데이터 공개라고 밝힘) | 2-ER 은 공간·몸 추론 전용, MolmoAct 2 는 그 위에 행동 출력을 붙인 것 | 지연은 H100 한 장 기준으로만 제시 |
| HY-Embodied-0.5 (MoT-2B) | Tencent | 전체 4B, 추론 때 켜지는 것 2.2B | "other"(본문 확인 필요) | 공간·시간 지각, 예측·계획. "엣지용"이라고 소개 | 16 GB 이상 권장 |
| Hy-Embodied-VLM-1.0 | Tencent | 전체 30B, 토큰당 약 3B | Apache-2.0 | 상태 이해·계획·되돌아보기·복구, 내비게이션(R2R-CE, 물체 찾기) | 약 86 GB (80 GB GPU 4 장 권장) |

저자 주장 예: Embodied-R1.5 는 24 개 평가 중 16 개에서, Molmo 2-ER 은 13 개 평가에서 Gemini Robotics ER 1.5 보다 높다고 한다(비교 대상이 ER 2 가 아니다).

### 우리 로봇에 쓴다면

- 로봇 컴퓨터는 Jetson Orin Nano(메모리 8 GB, CPU·GPU 공유)다. 위 표의 가장 작은 모델도 저자 표기로 16 ~ 24 GB 를 요구한다. 양자화하면 들어갈 수 있는지는 해 보지 않았다 — 지금은 "로봇 위에서 못 돈다"로 본다.
- 돌린다면 곁의 PC(RTX 3060 12 GB)에서 2B ~ 4B 급을 양자화해 띄우고 로봇과 무선으로 잇는 꼴이 된다. 지연과 끊김 대책이 따로 필요하다.
- 탐사 속도 목표에는 맞지 않는다: 글을 써서 답하므로 한 번 답에 수백 ms ~ 수 초가 걸릴 것으로 본다(추정, 우리 위층은 1 Hz · 아래층은 10 Hz). 쓸 자리가 있다면 탐사 중 판단이 아니라 "어느 방부터 볼지", "이게 불인지" 같은 드문 상위 판단이다.
- 후보를 고른다면 라이선스가 분명하고 작은 판이 있는 RoboBrain 2.5(4B, Apache-2.0)와 Cosmos-Reason2(2B)부터.

## 출처

- Gemini Robotics ER 2: <https://blog.google/innovation-and-ai/models-and-research/google-deepmind/gemini-robotics-er-2/> , <https://ai.google.dev/gemini-api/docs/robotics-overview>
- Cosmos-Reason2: <https://github.com/nvidia-cosmos/cosmos-reason2> · Cosmos 3: <https://www.nvidia.com/en-us/ai/cosmos/llm-info/>
- RoboBrain 2.5: <https://github.com/FlagOpen/RoboBrain2.5> , arXiv:2601.14352
- Embodied-R1.5: <https://embodied-r.github.io/> , arXiv:2606.11324
- Molmo 2-ER / MolmoAct 2: <https://allenai.org/blog/molmoact2> , arXiv:2605.02881
- HY-Embodied-0.5: <https://huggingface.co/tencent/HY-Embodied-0.5> · Hy-Embodied-VLM-1.0: <https://huggingface.co/tencent/Hy-Embodied-VLM-1.0>
- 판단 모델: Decision Index 순위표 <https://benchlm.ai/benchmarks/decisionindex> , 목록 <https://github.com/andyrewlee/awesome-system-one>
- d1: <https://www.liquid.ai/blog/d1-open> · GLiDE: <https://fastino.ai/blog/glide-decision-model-family> · REAX 1.3: <https://decisionmodels.ai/models/reax> · Decider: <https://github.com/Mapika/decider>
