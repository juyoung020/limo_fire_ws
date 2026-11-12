# 판단 모델로 프론티어 고르기 — 선행연구 조사

2026-11-12 조사. 웹 검색과 논문 페이지 읽기만 했다. 코드는 돌리지 않았다.

**확인 등급** (표의 "확인" 칸)

- **본문**: 논문 본문(HTML·PDF)을 열어 읽었다.
- **초록**: arXiv·학회 초록 페이지만 열어 확인했다. 세부(모델 이름·지연 등)는 초록에 없으면 모른다.
- **미확인**: 검색 결과 요약으로만 봤다. 인용하기 전에 반드시 직접 열어 볼 것.

이미 정리된 것은 다시 쓰지 않는다: HEADER·FARE 는 [header_fare_diff.md](header_fare_diff.md), 판단 모델 목록은 [../decision_models.md](../decision_models.md),
FUEL·TARE·DSVP·ARiADNE 와 후보 뽑기 차이는 `train/explore/mdp/README.md` "선행연구와 다른 곳".

우리 구상을 줄여 부르는 말: **(a)** 후보 8 개 중 고르기, **(b)** 탐사 끝내기 판단, **(c)** 못 가는 후보 피하기, **아래층** = 10 Hz RL 주행 + 막힘 신호.

## 1. 한눈에 — 가장 가까운 연구

| # | 연구 | 겹치는 점 | 다른 점 |
|---|---|---|---|
| 1 | **Aitha & Zakhor, "Autonomous Frontier-Based Exploration with VLM Guidance"** (arXiv 2605.23165, CVPR 2026 워크숍) | 위에서 본 점유 지도 **사진**에 번호 붙은 프론티어를 그려 VLM(시각-언어 모델)에 주고 하나를 고르게 한다. 물체 찾기가 아니라 **순수 탐사(coverage, 덮기)**. 후보는 최대 5 개. 못 가는 프론티어는 블랙리스트에 넣는다 | Gemini 2.5 Pro(인터넷 필요). 미세조정 없음. 글로 답한다(번호 + 이유). 지연·비용 수치 없음. 시뮬(Habitat)만. 블랙리스트는 규칙(경로 계획 실패)이지 모델 판단이 아니다. 아래층은 move_base + TEB(RL 아님) |
| 2 | **L2E (Liu et al., Amazon Lab126, RSS 2023 워크숍)** | 집 안 지상 로봇, **시야 좁은 센서**. 규칙이 후보 시점을 뽑고, 신경망이 **지도 그림(3×224×224) + 후보별 특징**으로 후보마다 점수를 내 하나를 고른다. 결정 주기 **1 Hz**. 모방학습으로 먼저 가르치고 RL 로 다듬는다. 지금 우리 PPO 위층과 구조가 거의 같다 | 작은 CNN(EfficientNet-B0) 하나. 큰 사전학습 판단 모델이 아니다. 모방 전문가는 **가장 가까운 프론티어 규칙**(GT 아님). 끝내기·못 가는 후보 판단 없음. 아래층은 기존 계획기. 실기 결과 없음(Gazebo) |
| 3 | **DARE (Cao et al., arXiv 2410.16687)** 와 HEADER 의 행동 복제 절제 | **GT 지도를 아는 최적 계획기**(프론티어를 덮는 시점 + TSP)의 시연을 **모방학습**으로 가르친다. 우리 "GT 순회에서 라벨" 과 같은 발상 | 입력이 지도 사진이 아니라 노드 그래프. 출력이 후보 선택이 아니라 경로(확산 정책). 사전학습 모델 아님. HEADER 는 같은 신경망을 행동 복제만으로 학습하면 **가장 나빴다**고 보고(661 m, 전문가 보상 RL 529 m — header_fare_diff.md) |
| 4 | **FiLM-Nav (Yokoyama & Ha, arXiv 2509.16445)** | 사전학습 VLM(2.8B)을 **미세조정**해 프론티어 선택기로 쓴다. 라벨은 **전문가(oracle, 정답을 아는 계획기)**가 고른 프론티어 = 지도학습. 추론 0.2 s/걸음 | 물체 찾기(ObjectNav). 입력은 지도 사진이 아니라 **1 인칭 카메라 사진**들. 답을 글자 토큰으로 생성한다(A, B, C …). 끝내기는 VLM 이 아니라 따로 둔 검출기가 한다. 로봇 탑재 실험 없음 |
| 5 | **Luperto et al., "Estimating Map Completeness in Robot Exploration"** (arXiv 2406.13482, 검색 결과로는 Autonomous Robots 2026 게재 — 게재 정보는 미확인) | **(b) 를 그대로 한 연구.** CNN 이 덜 그려진 격자 지도를 **그림으로** 보고 "얼마나 탐사됐나, 남은 곳이 가 볼 만한가"를 낸다. 라벨은 **완성 지도(GT)**에서 뽑는다. 탐사 시간 평균 40 % 절약이라고 보고 | 끝내기 판단만 한다. 프론티어 고르기와 한 모델이 아니다(후속작 arXiv 2508.10689 가 같은 신경망의 주목 지도로 프론티어 선택을 치우치게 한다). 큰 사전학습 모델 아님 |

덧붙여 **판단 모델 쪽**: Liquid AI 가 d1-3B 공개 글(2026-10-07)에서 "NVIDIA 와 함께 d1-3B 가 Isaac Sim 환경을 **항법**하는 것을 Jetson 에 올려 HIL(hardware-in-the-loop, 실제 기기를 시뮬에 물려 돌리기)로 시연했다"고 적었다.
무엇을 보고 무엇을 고르는지, 수치, 논문은 그 글에 없다. 곧 **"판단 모델을 로봇 항법에 처음 썼다"는 말은 못 한다.** 탐사·프론티어 선택에 쓴 사례는 찾지 못했다.

**한 줄 결론**: 조각마다 선행연구가 있다. 지도 사진으로 프론티어 고르기(1·2), GT 모방(3), VLM 미세조정 선택기(4), 지도 그림으로 끝내기(5) 모두 누가 했다.
남는 것은 **이것들을 글을 쓰지 않는 사전학습 판단 모델 하나로 묶어, Orin Nano 급 로봇 위에서 1 Hz 로 돌리고, 같은 조건의 작은 신경망(PPO·L2E 꼴)과 견준 것**이다(4 절).

## 2. 주제별 표

### 2.1 큰 사전학습 모델에 프론티어·목적지 고르기를 맡긴 연구 — 순수 탐사(덮기)

| 논문 | 연도 · 학회 | 한 일 | 겹치는 점 | 다른 점 | 확인 | 링크 |
|---|---|---|---|---|---|---|
| Aitha & Zakhor, Autonomous Frontier-Based Exploration with VLM Guidance | 2026-05, CVPR 2026 워크숍(3D-LLM/VLA) | 점유 지도 사진(로봇 경로 + 번호 붙은 프론티어) + 프론티어 쪽 카메라 사진을 Gemini 2.5 Pro 에 주고 하나 고르게 함. Habitat·Matterport3D 방 6 곳, 덮은 비율 최대 +24 % 라고 주장 | 지도 사진 입력, 순수 탐사, 후보 중 선택, 못 가는 프론티어 블랙리스트 | 클라우드 API, 미세조정 없음, 글 생성, 지연 미보고, 시뮬만, 3 회 평균. TARE·DSVP 는 다른 환경에서 돌린 궤적을 다시 재생해 견줌 | 본문 | <https://arxiv.org/abs/2605.23165> |
| HELM (Liao et al., NUS Sartoretti 연구실) | 2025-03, IROS 2025 제출본 | ARiADNE 식 그래프를 **글로 풀어** LLM(DeepSeek-V3)에 주고 다음 경유점을 고르게 함. 사람이 말로 선호를 줄 수 있음 | 후보(노드) 중 선택, 순수 탐사, TARE·DSVP·ARiADNE·DARE·최적과 견줌 | 입력이 글(사진 아님), 미세조정 없음, 글 생성. 선호를 안 주면 ARiADNE 보다 길다(Gazebo 1164 m 대 1025 m). 던전 100 장에서는 TARE Local·ARiADNE 와 비슷(최적 대비 +15.4 %) | 본문 | <https://arxiv.org/abs/2503.07006> |
| FARE (Liao et al.) | 2026-01 | LLM(Qwen3-14B)이 전역 경로, RL 이 지역 선택. AGX Orin 탑재 | 큰 모델 + RL 두 층, 기기 탑재 | header_fare_diff.md 에 정리됨. 실내에서는 이득 없음 | 본문(저장소) | <https://arxiv.org/abs/2601.14681> |
| COMRES-VLM (Wang et al., 초판 이름 LLM-MCoX) | 2025-09 (v3 2026-03) | 여러 로봇. 공유 점유 지도 + 프론티어 덩어리로 VLM 이 로봇별 경유점을 배정. 탐사 완료 10.2 % 빠름, 물체 찾기 55.7 % 효율↑ 주장 | 점유 지도 위 프론티어를 VLM 이 배정 | 여러 로봇(최대 6), 중앙 계획, 미세조정 언급 없음. 지도를 회색 사진 한 장으로 준다는 것과 모델이 GPT-4o 라는 것은 검색 요약 — 미확인 | 초록 | <https://arxiv.org/abs/2509.26324> |
| Kawashima et al., LLM-Guided Decentralized Exploration with Self-Organizing Robot Teams | 2026-03, AROB-ISBC 2026 | 수십~수백 대 로봇. 팀마다 다음 목적지를 LLM 이 고름. 고전 프론티어·DRL 과 견줌 | LLM 이 프론티어 후보 중 선택, 순수 탐사 | 격자 시뮬, 여러 로봇, 모델·입력 형식은 초록에 없음 | 초록 | <https://arxiv.org/abs/2603.04762> |
| ReEXplore (Zhang et al.) | 2025-11 | MLLM(다중모달 LLM) 탐사. 지난 경험을 되짚어 넣어 주고, 프론티어를 "방향 → 세부" 두 단계로 고르게 함. 학습 없음 | 프론티어 선택을 작은 행동 공간으로 줄임 | 3D 체화 과제(물체·질문) 쪽, 글 생성, 미세조정 없음 | 초록 | <https://arxiv.org/abs/2511.19033> |

### 2.2 같은 일 — 물체 찾기·목표 항법·질문 답하기 쪽 (우리와 과제가 다르다)

| 논문 | 연도 · 학회 | 한 일 | 겹치는 점 | 다른 점 | 확인 | 링크 |
|---|---|---|---|---|---|---|
| VLFM (Yokoyama et al.) | 2023-12 (ICRA 2024 로 알려져 있으나 arXiv 페이지에는 학회 표기 없음) | 사전학습 VLM 으로 "목표 물체 쪽일 가능성" 값 지도를 만들어 프론티어를 점수 매김. Spot 실기 | 프론티어 점수 매기기에 사전학습 모델, **글 생성 없음**(유사도 점수) | 물체 찾기. 카메라 사진 기반. 미세조정 없음 | 초록 | <https://arxiv.org/abs/2312.03275> |
| FiLM-Nav (Yokoyama & Ha) | 2025-09 (v2 2026-04) | 2.8B VLM(Cobra, Mamba 계열)을 **전체 미세조정**해 프론티어를 직접 고르게 함. 라벨 = 목표까지 최단 경로 위 프론티어(전문가). HM3D ObjectNav SR 77.0 % | **미세조정 + 전문가 라벨 지도학습 + 프론티어 선택**, 0.2 s/걸음 | 물체 찾기, 1 인칭 사진 입력(지도 사진 아님), 글자 토큰 생성, 멈춤은 별도 검출기, 탑재 실험 없음 | 본문 | <https://arxiv.org/abs/2509.16445> |
| Schwartz, Kondo, How, Efficient Navigation in Unknown Indoor Environments with VLMs | 2025-10, IROS 2025 워크숍(OWN) | 점유 지도를 **사진**(60×60 칸, 24 m)으로 그려 GPT-4.1 에 주고 프론티어 후보마다 믿음(0~1)을 받음. 3 번 뽑아 중앙값. 경로 10 % 짧아짐 | 지도 사진 입력이 글 입력보다 낫다고 보고. 후보별 점수. 프론티어 덩어리 중심을 후보로 | 목표 지점까지 가는 항법(덮기 아님). 점수를 글로 받음. 클라우드. **지연이 실시간의 주된 한계**라고 저자가 적음. 시뮬만 | 본문 | <https://arxiv.org/abs/2510.04991> |
| TopV-Nav (Zhong et al.) | 2024-11 | MLLM 이 위에서 본 지도 위에서 직접 추론. 지도에 시각 표시를 그려 줌 | 위에서 본 지도 사진을 큰 모델에 줌 | 물체 찾기, 영(0)샷, 글 생성 | 초록 | <https://arxiv.org/abs/2411.16425> |
| LocalNav (Baumann et al., ETH) | 2026-06 | 클라우드 VLM 의 추론을 4B 모델(Qwen3.5-4B)에 증류 + RL 미세조정, Jetson Orin 에서 실행. 지연 80 % 줄임 | **미세조정 + 기기 탑재** | 물체 찾기, 장면 그래프 기반, 글 생성(토큰 수를 줄이는 쪽으로 최적화) | 초록 | <https://arxiv.org/abs/2606.27871> |
| VL-Nav (Du et al.) | 2025-02, IROS 2026 | 시각-언어 특징으로 프론티어·물체 후보 점수. 실기 483 m | 프론티어 점수, 실기 | 언어 지시 항법. "Jetson Orin NX 에서 30 Hz" 는 검색 요약 — 미확인 | 초록 | <https://arxiv.org/abs/2502.00931> |
| Yuan et al., Reliability of Foundation Model-Based Frontier Selection | 2024-10 | 전문가 모델 3 개의 합의로 프론티어 선택을 안정화 | 큰 모델의 프론티어 선택이 불안정하다는 문제 제기 | 물체 찾기 | 초록 | <https://arxiv.org/abs/2410.21037> |
| Explore until Confident (Ren et al.) | 2024, RSS 2024 | VLM 으로 의미 지도를 만들어 탐사를 이끌고, **등각 예측(conformal prediction)**으로 보정한 확신도로 **멈출 때**를 정함 | 멈춤 판단을 모델 확률로 함, 확률 보정 | 질문 답하기(EQA). 멈춤 기준이 "답에 대한 확신"이지 지도 완성도가 아님 | 초록 | <https://arxiv.org/abs/2403.15941> |
| Prune-Then-Plan (Frahm et al.) | 2025-11 | VLM 이 고르면 프론티어 사이를 왔다 갔다 한다고 지적. VLM 은 **가망 없는 프론티어를 빼는 데만** 쓰고 최종 선택은 덮기 기반 계획기에 맡김 | **(c) 와 같은 꼴**: 모델은 걸러내기, 규칙이 선택. 왔다 갔다(진동) 문제 | 질문 답하기. 거르는 기준이 "갈 수 있나"가 아니라 "정보가 있나" | 초록 | <https://arxiv.org/abs/2511.19768> |
| GoalVLM, 3DGSNav, AnyGoal, LGR, RATE-Nav, OpenFrontier, History-Augmented VLM, EfficientNav, WayFinder | 2025~2026 | VLM·LLM 프론티어 점수·순위 계열(물체 찾기) | — | — | 미확인 | arXiv 2603.18210, 2602.12159, 2606.13878, 2503.20241, 2506.02354, (RSS 2026), 2506.16623, 2510.18546, (IROS 2026) |

### 2.3 작은 신경망으로 프론티어·후보 고르기 (사전학습 큰 모델 아님, 순수 탐사)

| 논문 | 연도 · 학회 | 한 일 | 겹치는 점 | 다른 점 | 확인 | 링크 |
|---|---|---|---|---|---|---|
| L2E (Liu et al.) | 2023, RSS 2023 워크숍 | 1 절 2 번. 고정 시간 안에 덮은 넓이에서 규칙 대비 최소 4~22 % 개선. 모방 사전학습을 빼면 나빠짐 | 지도 그림 + 후보별 점수 머리, 1 Hz, 좁은 시야 지상 로봇, 모방 → RL | 큰 사전학습 모델 아님, 전문가 = 가까운 프론티어 규칙, 한 지도에서만 학습, 학습 기반 비교 대상 없음 | 본문 | <https://mit-spark.github.io/robotRepresentations-RSS2023/assets/papers/2.pdf> |
| MapEx (Ho et al., CMU) | 2024-09 | 지도 채우기 신경망(LaMa)으로 안 본 곳을 예측하고, 예측 여러 장의 분산 + 보일 넓이로 후보 시점의 정보 이득 계산. KTH 평면도에서 가까운 프론티어 대비 25.4 % | 실내 구조 지식으로 프론티어 고르기, 지도 그림 입력 | 고르는 것은 규칙(예측 위 계산). 2D 라이다 360° | 초록 | <https://arxiv.org/abs/2409.15590> |
| MapExRL (Harutyunyan et al.) | 2025-03, ICAR 2025 | 예측 지도·불확실성·보일 넓이·거리·남은 예산을 보고 RL 이 프론티어를 고름. 최강 기준선 대비 최대 18.8 % | 프론티어가 행동 공간인 RL 선택기 | 사전학습 큰 모델 아님, 이동 예산이 있는 문제 | 초록 | <https://arxiv.org/abs/2503.01548> |
| HEADER · ARiADNE (Cao et al.) | 2023 · 2025 | 그래프 + 어텐션으로 이웃 노드 선택 | 저장소에 정리됨 | 저장소에 정리됨 | 본문(저장소) | <https://arxiv.org/abs/2510.15679> |
| Melero-Deza et al., Efficient Frontier Selection via RL … Minimal Sensing | 2026-04, Preprints.org(심사 전) | CNN 지도 특징 + 프론티어 특징 + 어텐션, PPO, Crazyflie 실기 | 구조가 우리 PPO 위층과 비슷 | 드론, 최소 센서 | 미확인(페이지 403) | <https://www.preprints.org/manuscript/202604.1959> |
| Explore-Bench (Xu et al.) | 2022, ICRA 2022 | 프론티어 기반 3 종 + DRL 1 종을 같은 틀에서 견주는 벤치마크(격자 시뮬 · Gazebo · 실기 시험대) | 평가 틀 | — | 초록 | <https://arxiv.org/abs/2202.11931> |

### 2.4 탐사 끝내기 판단 · 못 가는 프론티어 거르기

| 논문 | 연도 · 학회 | 한 일 | 겹치는 점 | 다른 점 | 확인 | 링크 |
|---|---|---|---|---|---|---|
| Luperto et al., Estimating Map Completeness in Robot Exploration | 2024-06 | 1 절 5 번. 격자 지도 그림 → CNN → 탐사 비율·"남은 곳이 가치 있나". 라벨은 완성 지도에서 | **(b) 와 같다**: 지도 그림, GT 라벨, 예/아니오 | 끝내기만. 작은 CNN | 초록 | <https://arxiv.org/abs/2406.13482> , 프로젝트 <https://aislabunimi.github.io/explore-stop> |
| Luperto et al., Biasing Frontier-Based Exploration with Saliency Areas | 2025-08, ECMR 2025 | 위 신경망의 주목 지도(saliency)로 "가면 큰 미지 영역이 열리는 곳"을 찾아 프론티어 전략을 그쪽으로 치우치게 함 | **끝내기 신경망과 프론티어 선택을 잇는다** — (a)+(b) 를 한 신경망에서 꺼내 쓴 선례 | 선택 자체는 기존 전략 + 치우침. 후보 중 고르기 학습이 아님 | 초록 | <https://arxiv.org/abs/2508.10689> |
| Stein, Bradley, Roy, Learning over Subgoals (LSP) | 2018, CoRL 2018 | 프론티어(하위 목표)마다 **"그 너머가 막다른 길일 확률"**과 비용을 신경망이 추정. 계획기가 이를 써서 고름. 비용 21 % 감소 | **(c) 의 원조 격**: 프론티어별 성공 확률을 학습. 모델은 확률만, 계획기가 선택 | 목표 지점 항법(덮기 아님). 라이다 스캔 입력. "막다른 길"이지 "몸이 못 지나감"은 아님 | 초록 | <https://proceedings.mlr.press/v87/stein18a.html> |
| Aitha & Zakhor (위 2.1) | 2026 | 경로 계획이 실패한 프론티어를 블랙리스트 | 못 가는 후보 빼기 | 규칙 | 본문 | 위 |
| Tremblay et al., 험지 위상 지도 탐사 | 2024-10 | 학습한 주행 가능성 예측으로 닿는·못 닿는 프론티어를 가름 | (c) | 야외 험지 | 미확인 | arXiv 2410.01925 |
| Placed & Castellanos, "Enough is Enough" | 2022 | 능동 SLAM 최적성 기준의 변화로 멈춤 | (b) 의 규칙형 선례 | 학습 아님 | 미확인 | — |

"도달 불가 프론티어를 **지도 사진만 보고 큰 모델이 미리** 가려낸다"는 연구는 찾지 못했다. 가까운 것은 LSP(작은 신경망, 스캔 입력)와 규칙형 블랙리스트다.

### 2.5 계층형 탐사 — 아래층 실패를 위층에 되먹임

| 논문 | 연도 · 학회 | 한 일 | 겹치는 점 | 다른 점 | 확인 | 링크 |
|---|---|---|---|---|---|---|
| Active Neural SLAM (Chaplot et al.) | 2020, ICLR 2020 | 학습한 전역 정책(먼 목표) + 해석적 계획기 + **학습한 지역 정책**. 모듈별로 따로 학습 | "위층이 목표, 아래층 학습 정책이 주행"의 대표 선례 | 아래층 → 위층 실패 신호는 초록에 없음(본문 미확인). 1 인칭 카메라, Habitat | 초록 | <https://arxiv.org/abs/2004.05155> |
| VORL-EXPLORE (Liu et al.) | 2026-03, IROS 2026 | 여러 로봇. 지역 주행 가능성·실행 신뢰도를 재는 **"실행 충실도"** 점수를 만들어 (1) 위험·혼잡한 프론티어에 벌점 (2) A* 와 반응형 RL 정책 사이 전환에 씀. 최근 진행·안전 결과로 스스로 보정 | **아래층 실행 결과가 프론티어 배정에 되먹임**, 움직이는 장애물, RL 지역 정책 | 여러 로봇, 격자 시뮬 위주(Gazebo 는 시연), 위층은 Voronoi 규칙 | 초록 | <https://arxiv.org/abs/2603.07973> |
| Strict Subgoal Execution (Hwang et al.) | 2025-06, ICLR 2026 | 계층형 RL 일반. 실패·부분 성공 기록으로 못 닿는 하위 목표를 가려내고 그래프 간선 비용을 고침 | "아래층 실패 → 위층이 그 목표를 피함"의 일반 이론 쪽 선례 | 로봇 탐사 아님(RL 벤치마크) | 초록 | <https://arxiv.org/abs/2506.21039> |
| explore_lite · nav2 계열 | — | 진행 없음 시간 초과 → 프론티어 블랙리스트 | 막힘 → 다시 고르기는 **공학 관행** | 규칙 | (저장소 README 가 이미 언급) | — |

"막힘 신호 → 위층이 다시 고른다" 자체는 새롭지 않다. 관행이고, 학습 쪽 선례(VORL-EXPLORE)도 있다.

### 2.6 GT·전지적 계획기를 모방해 선택기를 가르친 연구

| 논문 | 연도 · 학회 | 한 일 | 겹치는 점 | 다른 점 | 확인 | 링크 |
|---|---|---|---|---|---|---|
| DARE (Cao et al.) | 2024-10 | GT 지도에서 프론티어를 덮는 시점을 뽑고 TSP 로 순서를 정한 **최적 시연**을 만들어 확산 정책을 행동 복제. 기존 계획기와 비슷한 성능, 실기 | **GT 순회 → 모방** 그대로 | 그래프 입력, 경로 출력, 사전학습 모델 아님 | 초록(전문가 구성은 검색 요약) | <https://arxiv.org/abs/2410.16687> |
| HEADER (Cao et al.) | 2025-10 | GT 전문가의 첫 경유점과의 거리를 보상으로. **행동 복제만 쓴 절제는 +31.7 %(가장 나쁨)** | GT 전문가를 학습 신호로 | RL 보상으로 씀 | 본문(저장소) | <https://arxiv.org/abs/2510.15679> |
| FiLM-Nav (위 2.2) | 2025 | 전문가가 고른 프론티어를 라벨로 VLM 지도학습 | 전문가 라벨 → 분류 | 물체 찾기 | 본문 | 위 |
| L2E (위 2.3) | 2023 | 규칙 전문가 행동 복제 → RL. 모방만으로는 규칙보다 나빴다고 보고(그림 4) | 모방 사전학습 | 전문가가 GT 아님 | 본문 | 위 |
| Lodel et al., Learning Semantic Priorities for Autonomous Target Search | 2026-03, ICRA 2026 | 사람 전문가의 안내로 프론티어 우선순위 모델을 학습해 조합 최적화 탐사 계획기에 넣음 | 전문가 지도로 프론티어 우선순위 학습 | 수색(표적 찾기), 사람 전문가 | 초록 | <https://arxiv.org/abs/2603.29391> |
| Luperto 2024 (위 2.4) | 2024 | 완성 지도에서 끝내기 라벨 | GT 라벨로 (b) | — | 초록 | 위 |

**주의할 결과 두 개**: HEADER 와 L2E 모두 "모방만으로는 부족했다"고 적었다. 우리가 지도학습만으로 간다면 이 결과와 왜 다른지(또는 같은지) 답해야 한다.

### 2.7 판단 모델(decision model · System One)을 로봇에 쓴 사례

| 출처 | 날짜 | 내용 | 확인 | 링크 |
|---|---|---|---|---|
| Liquid AI, "Open d1" 글 | 2026-10-07 | d1-3B · d1-omni-600M 공개. 토큰을 생성하지 않고 한 번의 순전파로 답. Orin Nano 에서 질문 1 개 50 ms, 3 개 73 ms, 384 px 사진 202 ms, **3.4K 토큰 상태는 1640 ms**. "NVIDIA 와 함께 Isaac Sim 항법을 Jetson HIL 로 시연" 한 문장. 무엇을 입력·출력하는지, 수치, 미세조정 여부는 없음 | 본문 | <https://www.liquid.ai/blog/d1-open> |
| NVIDIA Jetson AI Lab 의 d1-3B 글 | — | 위 글이 링크함. 시연 세부가 있을 수 있음 | 미확인(못 엶) | — |
| Liquid AI LFM3 글 | 2026-10-27 | "마지막 둘은 시각·옴니 모델에서 특화한 판단 모델". 검색 요약에는 d1-2B · d1-Omni-2B 라는 이름과, LFM3-VL-2B 로 항법(AGX Orin 2.5 Hz 경유점)을 했다는 말이 있으나 본문 뒷부분을 못 읽음 | 일부만 | <https://www.liquid.ai/blog/lfm3> |

- 판단 모델을 **탐사·프론티어 선택**에 쓴 논문은 찾지 못했다. arXiv 에서 "decision model" + 탐사로 걸리는 것은 다른 뜻(일반 의사결정 모형)뿐이었다.
- 찾지 못했다는 것은 "없다"의 증거가 아니다. 이 모델군이 2026 년 가을에 나왔으므로 지금 누가 하고 있을 수 있다(추정).
- 참고: `decision_models.md` 는 d1-3B 만 적었다. LFM3 글의 2B 판(d1-2B)은 그 문서에 없다. 크기·이름이 출처마다 다르므로 쓰기 전에 모델 카드를 확인할 것.

## 3. 주장하면 안 되는 것 (이미 누가 했다)

| 하면 안 되는 주장 | 먼저 한 곳 |
|---|---|
| "위에서 본 점유 지도 사진을 큰 모델에 줘서 프론티어를 고르게 한 첫 연구" | Aitha & Zakhor 2026(순수 탐사), Schwartz 2025(목표 항법), TopV-Nav 2024, COMRES-VLM 2025 |
| "규칙이 후보를 뽑고 모델이 지도 그림을 보고 후보 점수를 내는 구조" | L2E 2023 (1 Hz, 좁은 시야 지상 로봇까지 같다) |
| "프론티어 선택용으로 VLM 을 미세조정한 첫 연구" | FiLM-Nav 2025, LocalNav 2026 (둘 다 물체 찾기) |
| "GT 를 아는 계획기를 모방해 탐사 정책을 가르친 첫 연구" | DARE 2024, HEADER 2025 |
| "탐사 끝내기를 학습 모델이 지도 그림으로 판단" | Luperto 2024 (라벨도 GT 에서) |
| "프론티어별 성공 가능성을 학습해 선택에 씀" | LSP 2018 (막다른 길 확률) |
| "아래층 실패를 위층에 알려 다시 고르게 함" | 공학 관행(블랙리스트), VORL-EXPLORE 2026, SSE 2026 |
| "위층 선택 + 아래층 학습 정책의 계층형 탐사" | Active Neural SLAM 2020 |
| "언어 모델을 로봇 위(Jetson)에서 돌린 탐사·항법" | FARE 2026(AGX Orin, 14B), LocalNav 2026(Orin, 4B) |
| "판단 모델을 로봇 항법에 처음 씀" | Liquid AI · NVIDIA 시연(2026-10, 세부 없음) |
| "글 생성 없이 점수만 내는 사전학습 모델로 프론티어 점수" | VLFM 2023(유사도 점수, 물체 찾기) |

## 4. 새롭다고 주장할 수 있는 것 (찾은 범위 안에서)

아래는 모두 **"이번 조사에서 같은 것을 찾지 못했다"**는 뜻이다. 투고 전에 다시 검색해야 한다.

1. **글을 쓰지 않는 사전학습 판단 모델을 순수 탐사의 프론티어 선택기로 미세조정한 것.**
   순수 탐사 쪽 큰 모델 연구(Aitha, HELM, FARE, COMRES-VLM)는 모두 미세조정 없이 글을 생성한다. 미세조정한 쪽(FiLM-Nav, LocalNav)은 물체 찾기이고 토큰을 생성한다.
2. **한 번의 순전파로 세 가지를 같이 내는 것: (a) 후보 선택 (b) 끝내기 (c) 못 가는 후보.**
   각각은 선례가 있다(L2E · Luperto · LSP). 한 모델·한 입력으로 묶은 것은 찾지 못했다. Luperto 2025 가 (a)+(b) 를 느슨하게 이은 것이 가장 가깝다.
   d1 글 수치로는 질문을 1 개에서 3 개로 늘려도 50 → 73 ms 라, 이 묶음이 판단 모델 형식과 잘 맞는다(추정).
3. **8 GB 급 기기(Orin Nano)에서 잰 지연과 함께 1 Hz 고리 안에서 돌린 것.**
   Aitha·Schwartz·HELM 은 지연을 안 적거나 한계로 적었다. FARE 는 AGX Orin + 14B 이고 추론 시간을 안 적었다. Orin Nano 에서 탐사 위층을 큰 사전학습 모델로 돌린 논문은 찾지 못했다.
4. **같은 후보·같은 아래층에서 "처음부터 학습한 작은 신경망(PPO) 대 미세조정한 사전학습 판단 모델 대 규칙"을 견준 것.**
   이 비교는 어디에도 없었다. 결과가 "사전학습이 도움 안 됨"이어도 정보가 된다(추정: 점유 지도는 사전학습 데이터와 거리가 멀어서 그럴 가능성이 있다).
5. **사람이 걸어 다니는 집 안에서, 10 Hz RL 주행 + 좁은 시야(220° 라이다 + 전방 깊이)까지 포함한 체계 전체 평가.**
   HEADER·FARE·HELM·Aitha 는 충돌·사람·로봇 방향을 문제에 넣지 않는다. 다만 이것은 "체계 기여"이지 알고리즘 새로움은 아니다.
6. **(c) 를 지도 사진만 보고 미리 판단하는 것.** 선례는 규칙형 사후 블랙리스트이거나 스캔 기반 작은 신경망(LSP)이다. 다만 "지도에 안 보이는 이유로 못 가는 것"(문턱, 사람)은 지도 사진으로 알 수 없다 — 주장 범위를 좁혀야 한다.

**정직한 약점** (심사에서 나올 질문)

- **성능 여지가 작다.** 우리 README 측정으로 거리 여지는 많아야 15~20 %, 규칙으로 얻은 것이 2~3 % 다. FARE 도 실내에서는 HEADER 보다 나빴다. "더 짧게 돈다"로 논문을 세우기는 어렵다(추정). 끝내기·못 가는 후보·기기 탑재 쪽이 더 설 자리가 있다.
- **모방만으로는 나빴다는 선행 결과**(HEADER −31.7 %, L2E). 우리 라벨이 "후보 8 개 안의 GT 순회 첫 목적지"인데, README 에서 GT 규칙(`cand_gt`)이 늘 0 번보다 나빴다. 라벨 자체가 좋은 선생인지부터 보여야 한다.
- **큰 모델이 프론티어 사이를 왔다 갔다 한다**는 보고(Prune-Then-Plan). 우리 `--rule-stick` 같은 장치가 필요하고, 그것 없이/있이 둘 다 재야 한다.
- **판단 모델의 근거가 블로그·순위표뿐이다.** d1·GLiDE 등은 심사된 논문이 없다(이번 조사 범위). 모델이 바뀌거나 내려가면 재현이 깨진다. 라이선스("LFM License")도 본문 미확인.
- **지도 사진 한 장의 해상도.** d1 은 384 px 사진에 202 ms. 우리 지도는 512×512 칸(0.05 m). 줄이면 좁은 문이 사라질 수 있다. 긴 상태 입력은 1640 ms 로 1 Hz 를 넘는다.

## 5. 논문으로 내려면 — 비교 대상과 평가

### 꼭 견줘야 할 비교 대상(baseline)

| 무리 | 대상 | 까닭 |
|---|---|---|
| 규칙 | 가장 가까운 프론티어, 정보 이득, 둘의 가중합, 무작위 | L2E·Explore-Bench·MapEx 가 모두 씀. 우리 `slot0`·`cand_gain`·`cand_tsp` 가 여기 해당 |
| 고전 계획기 | TARE, DSVP, (FUEL 꼴 순회) | HEADER·FARE·HELM·Aitha 가 모두 견줌. 안 넣으면 바로 지적받는다(추정) |
| 학습(작은 신경망) | 우리 PPO 위층, L2E 꼴(지도 CNN + 후보 점수), ARiADNE/HEADER 꼴(그래프 어텐션) | "사전학습이 필요한가"에 답하는 핵심 비교 |
| 큰 모델(영샷) | 같은 지도 사진을 클라우드 VLM 에 주고 고르게 하기(Aitha 방식), 미세조정 전 판단 모델 | "미세조정이 필요한가"에 답함 |
| 끝내기 | 프론티어 0, 덮음 비율 문턱, 진행 없음 시간, Luperto 꼴 CNN | (b) 의 비교 |
| 못 가는 후보 | 사후 블랙리스트(경로 실패·시간 초과), 지도 위 경로 존재 검사 | (c) 의 비교 |
| 상한 | GT 를 아는 순회(우리 `gt_tour`) | HEADER·HELM 의 "최적 대비 차" 지표 |
| 절제 | (a)만 / (a)+(b) / (a)+(b)+(c), 막힘 되먹임 끔, stick 끔, 입력 해상도 | 묶음의 값 |

### 흔히 쓰는 환경

- **무작위 던전 격자 지도** (ARiADNE·HEADER·HELM·DARE: 시험 100 장) — 순수 탐사 학습 쪽 표준.
- **Gazebo 큰 환경** (CMU 탐사 환경: 실내·숲·창고 등. TARE·DSVP 가 여기서 나옴) — HEADER·FARE·HELM.
- **KTH 평면도 데이터셋** (MapEx·MapExRL, 2D 라이다 실내).
- **Habitat + Matterport3D/HM3D** (Aitha 는 덮기, 나머지는 물체 찾기).
- **Gazebo 집 지도 몇 장** (L2E: 빈 집 2 + 가구 있는 집 3).
- 우리는 HSSD 집 메시를 쓴다. 남과 직접 견주려면 위 가운데 하나(KTH 평면도나 던전이 가볍다 — 추정)에서도 재는 것이 안전하다.

### 흔히 쓰는 지표

| 지표 | 쓰는 곳 |
|---|---|
| 완료까지 이동 거리, 완료 시간(makespan) | HEADER · FARE · HELM · TARE 계열 |
| 최적(GT 전문가) 대비 차 % | HEADER · HELM |
| 덮은 넓이/부피 – 시간(또는 거리) 곡선 | 거의 모두 |
| 고정 시간·고정 예산 안에 덮은 넓이 | L2E · MapExRL |
| 최종 덮음 비율 | Aitha |
| 결정당 계산 시간 | HEADER · HELM (큰 모델 논문은 대개 빠뜨림 — 우리가 채울 자리) |
| 끝내기: 절약한 시간 %, 멈춘 때의 실제 덮음, 너무 일찍 멈춘 비율 | Luperto |
| 여러 번 평균 ± 표준편차 (10 회·100 장) | HEADER · HELM |
| 물체 찾기 쪽: SR, SPL | VLFM · FiLM-Nav (우리 과제에는 해당 없음) |

우리가 더 넣어야 할 것(추정): 충돌률, 사람 있을 때 성공률, 막힘 → 다시 고르기 횟수, Orin Nano 에서의 지연·메모리, 못 가는 후보 판단의 정밀도·재현율.

## 6. 확인하지 못한 것

- 표에서 "미확인"으로 적은 논문 전부(검색 요약만 봄).
- Melero-Deza 2026 (Preprints.org 가 403).
- NVIDIA Jetson AI Lab 의 d1-3B 시연 글, LFM3 글 뒷부분(로봇 절).
- VLFM·MapEx·DARE 의 게재 학회(arXiv 페이지에 표기 없음), Luperto 2024 의 학술지 게재 정보.
- Active Neural SLAM 본문에서 지역 정책 실패를 전역 정책이 어떻게 받는지.
- COMRES-VLM 의 모델 이름과 지도 입력 형식, VL-Nav 의 Jetson 30 Hz.
- IEEE Xplore 에만 있는 논문(RA-L·ICRA·IROS 본문)은 따로 뒤지지 않았다. arXiv 에 없는 선행연구가 있을 수 있다.
- 한국어·중국어 논문은 찾지 않았다.

## 7. 출처

- Aitha & Zakhor 2026: <https://arxiv.org/abs/2605.23165> (본문 <https://arxiv.org/html/2605.23165v1>)
- Schwartz, Kondo, How 2025: <https://arxiv.org/abs/2510.04991>
- FiLM-Nav: <https://arxiv.org/abs/2509.16445>
- L2E: <https://mit-spark.github.io/robotRepresentations-RSS2023/assets/papers/2.pdf>
- HELM: <https://arxiv.org/abs/2503.07006>
- HEADER: <https://arxiv.org/abs/2510.15679> · FARE: <https://arxiv.org/abs/2601.14681>
- DARE: <https://arxiv.org/abs/2410.16687>
- COMRES-VLM: <https://arxiv.org/abs/2509.26324>
- Kawashima et al. 2026: <https://arxiv.org/abs/2603.04762>
- ReEXplore: <https://arxiv.org/abs/2511.19033>
- VLFM: <https://arxiv.org/abs/2312.03275>
- TopV-Nav: <https://arxiv.org/abs/2411.16425>
- LocalNav: <https://arxiv.org/abs/2606.27871>
- VL-Nav: <https://arxiv.org/abs/2502.00931>
- Yuan et al. 2024: <https://arxiv.org/abs/2410.21037>
- Explore until Confident: <https://arxiv.org/abs/2403.15941>
- Prune-Then-Plan: <https://arxiv.org/abs/2511.19768>
- MapEx: <https://arxiv.org/abs/2409.15590> · MapExRL: <https://arxiv.org/abs/2503.01548>
- Explore-Bench: <https://arxiv.org/abs/2202.11931>
- Luperto et al. 2024: <https://arxiv.org/abs/2406.13482> · 2025: <https://arxiv.org/abs/2508.10689> · 프로젝트: <https://aislabunimi.github.io/explore-stop>
- LSP (Stein et al. 2018): <https://proceedings.mlr.press/v87/stein18a.html>
- Active Neural SLAM: <https://arxiv.org/abs/2004.05155>
- VORL-EXPLORE: <https://arxiv.org/abs/2603.07973>
- Strict Subgoal Execution: <https://arxiv.org/abs/2506.21039>
- Lodel et al. 2026: <https://arxiv.org/abs/2603.29391>
- Melero-Deza et al. 2026 (미확인): <https://www.preprints.org/manuscript/202604.1959>
- Liquid AI d1: <https://www.liquid.ai/blog/d1-open> · LFM3: <https://www.liquid.ai/blog/lfm3>
- 미확인 묶음: arXiv 2603.18210 (GoalVLM), 2602.12159 (3DGSNav), 2606.13878 (AnyGoal), 2503.20241 (LGR), 2506.02354 (RATE-Nav), 2506.16623 (History-Augmented VLM), 2510.18546 (EfficientNav), 2410.01925 (Tremblay et al.)
