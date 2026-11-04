# 실내 프론티어 탐사 — 선행 연구

처음 보는 실내(덕트호스 미로 약 6×8 m, 방)의 첫 지도를 빠르게, 구멍 없이 만드는 문제의 선행 논문 모음.
우리 계획(속도 v·vy·ω 를 10 Hz 로 바로 내는 PPO 정책 + 끝 선언, 기준선 = 가장 가까운 프론티어·정보 이득 프론티어, 안 되면 BT 가 고전 프론티어 탐사로 넘어감)과 견주어 읽었다.

- PDF 는 합법 공개본만 받았다(arXiv, 저자·연구실 페이지, 공개 학회 논문집). 공개본이 없는 논문은 링크만 두고 "PDF 없음(유료)" 로 적었다.
- 요약은 받은 PDF 의 초록·방법·결과 절을 읽고 썼다. PDF 가 없는 논문은 출판사·DB 의 초록과 공식 코드 README 만 보고 썼고, 그렇다고 따로 적었다.
- 저장소에서 이미 쓰는 것: Active Neural SLAM(`train/explore/mdp/README.md` 먼 지도 관측 꼴), explore_lite(`train/explore/README.md` 프론티어 기준선).

## 한눈에

| 파일 | 논문 (저자, 연도, 학회/저널) | 링크 | 한 줄 요지 |
|---|---|---|---|
| `1997_yamauchi_frontier.pdf` | A Frontier-Based Approach for Autonomous Exploration (Yamauchi, 1997, CIRA) | [DOI](https://doi.org/10.1109/CIRA.1997.613851) · [저자 PDF](http://www.robotfrontier.com/papers/cira97.pdf) | 프론티어(빈 칸과 모름 칸의 경계)를 정의하고 가장 가까운 프론티어로 계속 간다 |
| PDF 없음(유료) | Navigation Strategies for Exploring Indoor Environments (González-Baños & Latombe, 2002, IJRR 21(10-11)) | [DOI](https://doi.org/10.1177/0278364902021010834) | 후보 자세마다 정보 이득 × e^(−λ·경로 길이) 로 다음 자리를 고른다(효용 프론티어의 원형) |
| `2005_burgard_coordinated-exploration.pdf` | Coordinated Multi-Robot Exploration (Burgard, Moors, Stachniss, Schneider, 2005, IEEE T-RO 21(3)) | [DOI](https://doi.org/10.1109/TRO.2004.839232) · [연구실 PDF](https://www.ipb.uni-bonn.de/wp-content/papercite-data/pdf/burgard05tro.pdf) | 프론티어 효용 U − β·거리 비용, 고른 목표 둘레 효용을 깎아 로봇끼리 나눈다 |
| `2010_holz_frontier-strategies-eval.pdf` | Evaluating the Efficiency of Frontier-based Exploration Strategies (Holz, Basilico, Amigoni, Behnke, 2010, ISR/ROBOTIK) | [저자 PDF](https://www.ais.uni-bonn.de/~holz/papers/holz_2010_isr.pdf) · [VDE](https://www.vde-verlag.de/proceedings-en/453273006.html) | 가장 가까운 프론티어가 정보 이득 방식보다 경로가 짧다. 재확인·방 나누기로 더 줄임 |
| 논문 아님(코드) | explore_lite / m-explore (Hörner, 2016~, ROS 패키지) | [GitHub](https://github.com/hrnr/m-explore) · [ROS 2 판](https://github.com/robo-friends/m-explore-ros2) · [ROS wiki](http://wiki.ros.org/explore_lite) | 비용 = potential_scale·거리 − gain_scale·프론티어 크기, 진척 없으면 블랙리스트 |
| `2018_bircher_rh-nbvp-auro.pdf` (확장판 AuRo 2018; ICRA 2016 판은 유료) | Receding Horizon "Next-Best-View" Planner for 3D Exploration (Bircher et al., 2016, ICRA) | [DOI](https://doi.org/10.1109/ICRA.2016.7487281) · [확장판 AuRo 2018 (ETH 공개본)](https://research-collection.ethz.ch/bitstream/handle/20.500.11850/127860/s10514-016-9610-0.pdf?sequence=2) · [코드](https://github.com/ethz-asl/nbvplanner) | RRT 를 키워 미지 공간을 가장 많이 볼 가지를 찾고 첫 마디만 간다(RH-NBVP) |
| `2021_cao_tare.pdf` | TARE: A Hierarchical Framework for Efficiently Exploring Complex 3D Environments (Cao, Zhu, Choset, Zhang, 2021, RSS) | [DOI](https://doi.org/10.15607/RSS.2021.XVII.018) · [PDF](https://www.roboticsproceedings.org/rss17/p018.pdf) | 가까운 곳은 촘촘한 시점·TSP, 먼 곳은 성긴 부분공간 TSP. 표면 덮음 기준 |
| `2021_zhou_fuel.pdf` | FUEL: Fast UAV Exploration using Incremental Frontier Structure and Hierarchical Planning (Zhou et al., 2021, RA-L 6(2)) | [arXiv](https://arxiv.org/abs/2010.11561) | 프론티어 묶음 정보를 점진 갱신, 전역 ATSP → 지역 시점 다듬기 → 최소시간 궤적 |
| PDF 없음(유료) | Deep RL Robot for Search and Rescue Applications: Exploration in Unknown Cluttered Environments (Niroui, Zhang, Kashino, Nejat, 2019, RA-L 4(2)) | [DOI](https://doi.org/10.1109/LRA.2019.2891991) | 프론티어 탐사 + 심층 RL: RL 이 어느 프론티어로 갈지 고른다 |
| `2020_li_drl-auto-exploration.pdf` | Deep RL-Based Automatic Exploration for Navigation in Unknown Environment (Li, Zhang, Zhao, 2020, IEEE TNNLS 31) | [arXiv](https://arxiv.org/abs/2007.11808) · [DOI](https://doi.org/10.1109/TNNLS.2019.2927869) | 지도 → 완전합성곱 Q망이 목표 칸을 고름(A* 가 감), **끝내기 행동** 포함 |
| PDF 없음(유료) | Self-Learning Exploration and Mapping for Mobile Robots via Deep RL (F. Chen, Bai, Shan, Englot, 2019, AIAA SciTech) | [DOI](https://doi.org/10.2514/6.2019-0396) · [코드](https://github.com/RobustFieldAutonomyLab/DRL_robot_exploration) | 지역 지도 구조로 다음 감지 행동을 고르는 RL, 정보 기반 보상 |
| `2019_chen_learning-exploration-policies.pdf` | Learning Exploration Policies for Navigation (T. Chen, S. Gupta, A. Gupta, 2019, ICLR) | [arXiv](https://arxiv.org/abs/1903.01959) | 지도+RGB → 저수준 동작 6 개(옆걸음 포함) 직접, 덮음 보상 + 충돌 벌점, 모방 후 PPO |
| `2020_chaplot_active-neural-slam.pdf` | Learning to Explore using Active Neural SLAM (Chaplot et al., 2020, ICLR) | [arXiv](https://arxiv.org/abs/2004.05155) | 전역 RL 은 25 걸음마다 장기 목표, 고전 플래너 + 지역 정책이 감. 덮음 증가 보상 |
| `2023_cao_ariadne.pdf` | ARiADNE: A RL approach using Attention-based Deep Networks for Exploration (Cao, Hou, Wang, Yi, Sartoretti, 2023, ICRA) | [arXiv](https://arxiv.org/abs/2301.11575) · [DOI](https://doi.org/10.1109/ICRA48891.2023.10160565) | 충돌 없는 시점 그래프 위에서 어텐션 정책이 이웃 시점을 고름. 2D 실내에서 TARE Local 보다 5 % 짧음 |
| `2024_cao_large-scale-drl-exploration.pdf` | Deep RL-based Large-scale Robot Exploration (Cao, Zhao, Wang, Xiang, Sartoretti, 2024, RA-L) | [arXiv](https://arxiv.org/abs/2403.10833) · [DOI](https://doi.org/10.1109/LRA.2024.3379804) | ARiADNE 후속: **정답 지도를 보는 비평가**(특권 학습) + 그래프 성기게. TARE 보다 경로 12 % 짧음 |
| `2023_li_ccrl-curriculum-exploration.pdf` | Autonomous Exploration and Mapping for Mobile Robots via Cumulative Curriculum RL (Z. Li, Xin, N. Li, 2023, IROS) | [arXiv](https://arxiv.org/abs/2302.13025) | 격자 시뮬 + 지역/전역 지도 → 전진·좌·우 직접. 옛 단계 맵을 계속 섞는 누적 커리큘럼 |
| `2017_tai_virtual-to-real-mapless-nav.pdf` | Virtual-to-real Deep RL: Continuous Control of Mobile Robots for Mapless Navigation (Tai, Paolo, Liu, 2017, IROS) | [arXiv](https://arxiv.org/abs/1703.00420) | 라이다 10 개 + 목표 → (v, ω) 연속 출력, 미세조정 없이 실물 Turtlebot |
| `2025_sivashangaran_exploration-without-maps.pdf` | Mobile Robot Exploration Without Maps via Out-of-Distribution Deep RL (Sivashangaran, Khairnar, Eskandarian, 2025, IFAC-PapersOnLine) | [arXiv](https://arxiv.org/abs/2402.05066) · [DOI](https://doi.org/10.1016/j.ifacol.2025.12.292) | 깊이 170 개 → 스로틀·조향 직접(PPO), 경주로에서 학습해 실물에 바로 |
| `2026_liu_vorl-explore.pdf` | VORL-EXPLORE: A Hybrid Learning Planning Approach to Multi-Robot Exploration in Dynamic Environments (Liu et al., 2026, IROS 채택) | [arXiv](https://arxiv.org/abs/2603.07973) | 프론티어 배정은 고전, 실행은 A* 와 RL 반응 정책을 히스테리시스 스위치로 바꿔 탐 |
| `2022_xu_explore-bench.pdf` | Explore-Bench: Data Sets, Metrics and Evaluations for Frontier-based and DRL-based Autonomous Exploration (Xu et al., 2022, ICRA) | [arXiv](https://arxiv.org/abs/2202.11931) | 장면 요소(고리·좁은 복도·모서리·방) 데이터셋, 99 %·90 % 도달 시간 지표, 격자→Gazebo→실물 3 단 |

---

## 1. 고전 프론티어

### Yamauchi 1997 — 프론티어 탐사 (`1997_yamauchi_frontier.pdf`)

**어떻게 하나**
- 지도: 증거 격자(칸마다 점유 확률, 사전값 0.5). 확률 < 사전 = 빈, = 사전 = 모름, > 사전 = 막힘. 레이저보다 먼 소나 값은 레이저 거리로 잘라 정반사를 줄임.
- 빈 칸 중 모름 칸에 붙은 칸 = 프론티어 가장자리 칸. 붙은 것끼리 묶고 로봇 크기보다 큰 묶음만 프론티어로 침.
- 목표: **가장 가까운, 갈 수 있고 아직 안 간** 프론티어로 격자 경로 탐색 → 도착하면 360° 훑고 다시 찾음. 진척이 없으면 "못 감" 목록에 넣음.
- 결과: Nomad 200 로 사무실(약 7×6 m)을 30 분(개선판 15 분), 연구실·사무실(약 14×8 m)도 30 분에 다 지도화.

**우리에게**
- "가장 가까운 프론티어" 기준선과 실패 시 넘어갈 고전 탐사의 원형. 못 감 목록(블랙리스트)은 덕트호스처럼 라이다에 낮게 걸리는 곳에서 꼭 필요.

### González-Baños & Latombe 2002 — 정보 이득 × 거리 (PDF 없음(유료))

**어떻게 하나** (출판사 초록 + Holz 2010 의 재구현 설명으로 정리, 원문 미확인)
- 거리 센서 로봇이 미지 실내의 다각형 지도(layout)를 만든다. 제어·감지 오차를 감안한 "안전 영역" 개념을 쓴다.
- 프론티어 근처에서 후보 자세 p 를 뽑아 효용 u(p) = I(p)·e^(−λ·L(p)) 로 평가(I = 새로 볼 넓이/정보, L = 경로 길이). Holz 2010 은 λ = 0.2 를 원 논문 권장값으로 썼다.

**우리에게**
- "정보 이득 프론티어" 기준선의 식. 우리 기준선은 I 를 "새로 보일 벽 면 칸 수"로 두면 지표(벽 덮음)와 맞는다.

### Burgard et al. 2005 — 효용−비용 협조 탐사 (`2005_burgard_coordinated-exploration.pdf`)

**어떻게 하나**
- 지도: 점유 격자. 비용 V = 로봇에서 각 프론티어 칸까지 최단 경로(결정적 가치 반복, 지도를 가우시안으로 흐려 불확실성 반영).
- 효용 U: 처음 모두 1. 목표 t 를 한 로봇에 주면 둘레 칸 효용을 P(d) = 1 − d/최대거리 만큼 깎음(사이에 벽이 있으면 안 깎음). argmax (U − β·V) 를 로봇마다 차례로 고름, β = 1(0.01~50 에서 탐사 시간 거의 같음).
- 결과: 실물·시뮬에서 협조 없는 방식보다 탐사 시간이 크게 줆. 통신 거리가 지도 지름의 30 % 면 무제한과 비슷.

**우리에게**
- 한 대라도 "효용 − β·거리" 꼴이 효용 프론티어 기준선의 표준 식. β 에 둔감하다는 결과는 기준선 조정을 줄여 준다.

### Holz et al. 2010 — 프론티어 전략 비교 (`2010_holz_frontier-strategies-eval.pdf`)

**어떻게 하나**
- 지도: 2D 레이저 반사 확률 격자(ICP SLAM). 갈 수 있음 ∩ 닿을 수 있음 칸 중 모름에 붙은 칸 = 프론티어. 목표 없는 다익스트라로 모든 칸까지 경로 길이를 한 번에 구해 가장 가까운 것을 고름. 장애물에서 30 cm 안은 후보 제외.
- 개선 ① 재확인: 가는 도중 목표가 더는 프론티어가 아니면 바로 다음 것으로(O(1)). ② 방 나누기: 보로노이 임계점으로 지도를 방으로 쪼개고 지금 방의 프론티어를 먼저.
- 결과(경로 길이, 10 회 평균): 병원 맵 무작위 669 m, 가장 가까운 281 m, +재확인 253 m, +방 나누기 221 m, 정보 이득(González-Baños) 313 m. 넓은 열린 맵에서는 방 나누기가 오히려 조금 늘림.

**우리에게**
- "가장 가까운 프론티어가 생각보다 강하다"는 근거. 우리 기준선에 재확인은 반드시 넣고, 방이 있는 맵에서는 방 나누기 판도 비교할 만하다. 지표가 경로 길이 하나뿐인 점은 한계.

### explore_lite / m-explore (Hörner) — ROS 프론티어 탐사 (논문 아님, 코드)

**어떻게 하나** (GitHub `hrnr/m-explore` 소스 `frontier_search.cpp`, `explore.cpp` 를 직접 읽음)
- costmap 에서 로봇 위치부터 BFS 로 프론티어(빈 칸 중 모름 4 이웃) 묶음을 찾고 `min_frontier_size` 보다 작은 것은 버림.
- 비용 = `potential_scale`·(로봇~가장 가까운 프론티어 칸 직선거리) − `gain_scale`·(프론티어 칸 수). 기본 1e-3, 1.0. 가장 싼 것을 move_base 목표로.
- `planner_frequency`(기본 1 Hz)로 다시 고르고, `progress_timeout`(기본 30 s) 동안 진척이 없으면 그 목표를 블랙리스트. 프론티어가 없으면 멈춤.
- 저자 Jiří Hörner 의 학위논문([Charles Univ. 2016](https://dspace.cuni.cz/handle/20.500.11956/83769))은 같은 저장소의 `multirobot_map_merge`(지도 합치기)에 관한 것이라 탐사 알고리즘 문헌으로는 인용하기 어렵다. 인용은 저장소·ROS wiki 로.

**우리에게**
- 우리 `baselines/frontier` 가 따르는 실물 구현. 실물 비교 때 ROS 2 판(`m-explore-ros2`)을 Nav2 와 그대로 돌리면 "남이 만든 기준선"이 된다.

## 2. 샘플링 NBV · 계층 플래너

### Bircher et al. 2016 — RH-NBVP (ICRA 판은 유료, 확장판 AuRo 2018 은 `2018_bircher_rh-nbvp-auro.pdf`)

**어떻게 하나** (확장판 AuRo 2018 원문으로 확인: RRT 마디 이득 Gain(n_k) = Gain(n_k−1) + μ(보이는 모르는 부피)·e^(−λ·경로 비용), 가장 좋은 가지의 첫 구간만 실행)
- 지도: OctoMap(3D). 현재 자세에서 RRT 를 키우고, 마디마다 "아직 모르는 부피를 얼마나 보나"를 거리로 깎은 이득으로 매김.
- 이득이 가장 큰 가지의 **첫 마디만** 실행하고 새 센서 정보로 나무를 다시 키운다(receding horizon).
- ARiADNE(2023)의 2D 실내 시험에서는 효용 프론티어(λ = 10)보다도 경로가 길었다.

**우리에게**
- 3D 드론용이라 2D 실내에서는 기준선 우선순위가 낮다. "한 걸음씩 다시 계획"하는 구조는 우리 10 Hz 정책과 같은 반응형이라는 점만 참고.

### Cao et al. 2021 — TARE (`2021_cao_tare.pdf`)

**어떻게 하나**
- 지도·덮음: 3D 표면 단위. 표면 조각은 시점과의 거리 ≤ D 이고 법선과 시선 각이 기준 안이면 "덮였음". 남은 = 아직 안 덮인 표면.
- 지역(로봇 둘레 상자): 시점을 무작위로 뽑아 안 덮인 표면을 가장 많이 덮는 최소 집합을 고르고(서브모듈러 탐욕), 시점들을 TSP 로 이어 동역학 가능한 경로.
- 전역: 공간을 부분공간으로 나눠 "안 감 / 탐사 중 / 다 봄" 상태만 기록, 탐사 중인 부분공간 중심을 TSP 로 이은 성긴 경로. 둘을 지역 상자 경계에서 이음. O(n^2.2 + m^2.2).
- 결과: 지상·비행 로봇에서 탐사 효율(초당 탐사 부피) 80 % 높고 계산은 절반 이하. RSS 2021 최우수 논문.

**우리에게**
- **덮음을 "표면" 기준으로 정의**한 점이 우리 "벽 면 칸 덮음 ≥ 95 %"와 가장 비슷하다. TARE Local 꼴(시점 집합 + TSP)은 프론티어보다 강한 고전 기준선이 필요할 때 2D 로 옮기기 좋다.

### Zhou et al. 2021 — FUEL (`2021_zhou_fuel.pdf`)

**어떻게 하나**
- 프론티어 정보 구조(FIS): 프론티어 묶음마다 칸, 평균 위치, 경계 상자, 둘레 원통에서 뽑은 후보 시점(덮는 칸 수 순), 다른 묶음까지 이동 시간 하한을 저장. 지도가 바뀐 곳만 점진 갱신(큰 묶음은 PCA 로 쪼갬).
- 계획 3 단: ① 모든 묶음을 도는 최단 순회(ATSP) ② 앞쪽 몇 묶음의 시점을 다시 골라 다듬기 ③ 첫 시점까지 최소시간 궤적.
- 결과: 기존 최고 대비 3~8 배 빠른 탐사(시뮬·실물 쿼드로터).

**우리에게**
- "프론티어 + 전역 순서(TSP)"는 가장 가까운 프론티어의 되돌아가기 낭비를 없앤다. 우리 RL 이 이겨야 할 상한 쪽 고전 방식으로 참고. 점진 갱신 프론티어는 실물 BT 대체 경로의 계산량 줄이기에 쓸 수 있다.

## 3. 학습 기반 탐사

### Niroui et al. 2019 — 프론티어 + 심층 RL (PDF 없음(유료))

**어떻게 하나** (IEEE 초록만 확인)
- 고전 프론티어 탐사와 심층 RL 을 합쳐, **RL 이 갈 프론티어를 고른다**(이동은 고전 내비게이션).
- 크기·배치가 다른 어수선한 실제 환경에서 실물 시험. 다른 프론티어 방식보다 탐사 초반에 더 많은 넓이를 봄(수색 구조에서 조기 발견에 유리).

**우리에게**
- "RL 이 목표 고르기" 갈래의 대표. 우리가 속도 직접 출력이 막히면 갈 수 있는 중간 단계(RL 이 프론티어 선택 + Nav2 이동).

### Li, Zhang, Zhao 2020 — 결정·계획·지도 나누기 + 끝내기 행동 (`2020_li_drl-auto-exploration.pdf`)

**어떻게 하나**
- 틀: 결정(다음 목표 칸) · 계획(A*) · 지도(SLAM) 모듈로 나눔. 결정만 RL.
- 관측: 부분 점유 격자. 행동: 지도 위 격자 표본 점 중 하나(이산) + **끝내기 행동**. 망: 완전합성곱 Q망(FCQN, dueling), 보조 과제로 지도 가장자리(장애물 윤곽·프론티어) 분할을 붙인 AFCQN.
- 보상: α·(엔트로피 감소 − 이동 거리). 목표가 모름 칸이거나 장애물에 너무 가까우면 −1. 끝내기: 탐사 비율 > 0.85 면 +1, 아니면 −1.
- 결과: 프론티어·다른 RL 보다 학습 효율·적응이 좋고, 실물 로봇으로 옮겨 동작. AFCQN 은 지도를 더 채우지만 경로가 김.

**우리에게**
- 우리 "탐사 끝 선언"과 같은 행동을 가진 드문 사례. 끝내기 기준(0.85)이 낮으면 일찍 끝내는 쪽으로 배운다 — 우리 95 % 벽 덮음 기준과 끝 선언 보상 설계에 바로 참고.

### F. Chen et al. 2019 — 자기 학습 탐사 (PDF 없음(유료))

**어떻게 하나** (AIAA 초록 + 공식 코드 README 만 확인. 저자 사본은 코드 README 가 ResearchGate 로 연결하지만 받지 않음)
- 탐사 중 얻는 정보 기반 보상으로 심층 RL 정책을 학습, 정책은 지역 환경 구조로부터 다음 감지 행동을 고른다.
- 목적: 후보 행동마다 정보 이득을 계산하는 정보이론 탐사보다 계산을 줄임.

**우리에게**
- 정보 이득 계산을 학습된 정책으로 대신한다는 동기가 우리와 같다. 공개 코드가 있어 비교 실험이 필요하면 재현 가능.

### T. Chen, S. Gupta, A. Gupta 2019 — 저수준 탐사 정책 (`2019_chen_learning-exploration-policies.pdf`)

**어떻게 하나**
- 관측: RGB(ResNet18), 자기 중심 지도 2 척도(세밀·거친), 범퍼. RGB-D 60° 시야, 깊이 3 m 까지로 지도를 만듦.
- 행동: **6 개 동작을 직접** — 앞·뒤 0.25 m, 왼·오른 옆걸음 0.25 m, 좌·우 9° 회전. 고르는 목표점이 아니라 바로 움직임.
- 보상: 지도 덮음 증가 C(M_t+1) − C(M_t) + 범퍼 충돌 벌점(내부 센서만으로 계산). 사람 탐사 궤적 693 개로 모방 학습 후 PPO 미세조정. House3D.
- 결과: 자세 오차가 없으면 프론티어가 가장 좋지만, 자세·센서 잡음이 4 % 만 있어도 학습 정책이 프론티어를 앞섬. 호기심 기반보다 좋음.

**우리에게**
- RL 이 동작을 직접 내는 탐사의 기준 논문이고 **옆걸음(메카넘 vy)** 을 쓴다. "자세 오차에서 프론티어가 무너진다"는 결과는 우리가 시뮬에 지도·자세 오차를 넣는 이유의 근거.

### Chaplot et al. 2020 — Active Neural SLAM (`2020_chaplot_active-neural-slam.pdf`)

**어떻게 하나**
- 신경 SLAM(RGB+자세 센서 → 지도·자세), 전역 정책, 고전 플래너(Fast Marching), 지역 정책의 4 단.
- 전역 정책 관측: 지도·탐사함·현재 위치·지나간 곳 4 채널을 (로봇 둘레 잘라냄 + 전체 맥스풀) 8×G×G 로. **25 걸음마다 장기 목표**를 냄. 보상 = 덮음(m²) 증가, PPO. 지역 정책은 단기 목표 + RGB 로 동작(모방 학습).
- 지표: 덮은 넓이(m²)와 덮은 비율(% Cov), 1000 걸음 고정.
- 결과(Gibson): ANS 94.8 % vs 끝단 RL 최고 78.9 %. 전역 정책을 가장 가까운 프론티어로 바꾸면 92.5 %(큰 장면 78.2 % vs 86.2 %). Locobot 실물 이전.

**우리에게**
- 우리 먼 지도 관측(32×32, 확률·지나간 곳·현재 위치)이 이 꼴을 따랐다. "프론티어는 작은 장면에서 거의 같고 큰 장면에서 10 % 낮다"는 결과는 6×8 m 시연장에서는 RL 이득이 작을 수 있다는 경고.

### Cao et al. 2023 — ARiADNE (`2023_cao_ariadne.pdf`)

**어떻게 하나**
- 지도 → 빈 곳에 고르게 깐 시점 마디 + k(20) 최근접 중 충돌 없는 간선의 그래프. 마디 특징 = (x, y, 효용 = 보이는 프론티어 수, 방문 여부).
- 정책: 어텐션 인코더(6 층) + 포인터 디코더로 **이웃 마디 하나**를 고름(행동 수가 이웃 수에 따라 변함). SAC.
- 보상: a·(새 마디에서 본 프론티어 수) − b·(이동 거리) + 끝나면 20(정답의 99 % 탐사). 학습 약 24 h.
- 결과: 2D 실내 수백 개 시험에서 가장 가까운·효용(λ = 1, 10, 25)·NBVP·TARE Local·CNN RL 보다 평균 경로가 짧음(무작위 맵에서 TARE Local 대비 5 %). ROS 시뮬에서도 확인.

**우리에게**
- 2D 실내 RL 탐사의 현재 강한 기준이고 **효용 프론티어(λ)·가장 가까운 프론티어를 같은 맵에서 비교한 표**가 있어 우리 기준선 설계와 비교 방식에 그대로 쓸 수 있다. 다만 행동이 "다음 시점" 이라 속도는 고전 제어기가 낸다.

### Cao et al. 2024 — 대규모 DRL 탐사 (`2024_cao_large-scale-drl-exploration.pdf`)

**어떻게 하나**
- ARiADNE 그래프 정책을 이어, **비평가에게 정답 지도로 만든 그래프**를 넣는 특권 학습(배치 때는 정책만). 상태-행동 가치 손실·기울기 분산이 약 10 배 낮고 걸음당 보상 약 10 % 높음.
- 보상: −이동 거리 + 탐사 보상(shaping) + 끝 보상(모든 마디 효용 0). 작은 맵(던전)에서만 학습, 그래프 성기게(rarefaction)로 큰 맵에 씀. 약 3 일 학습.
- 결과: 130×100 m Gazebo 사무실에서 TARE 대비 경로 12 %·소요 시간 6 % 짧고 계획 시간 60 % 적음. 80×10 m 실내 지상 로봇 실물 시험.

**우리에게**
- 우리 비대칭 actor-critic(참값을 가치에만)과 같은 생각이 탐사에서 효과를 낸 직접 근거. 작은 맵 학습 → 큰 맵 일반화 결과도 우리 맵 크기 1~3 배 설정과 맞물린다.

### Z. Li, Xin, N. Li 2023 — 누적 커리큘럼 RL (`2023_li_ccrl-curriculum-exploration.pdf`)

**어떻게 하나**
- 자체 격자 시뮬(레이 트레이싱 2D 라이다, 참 자세, 한 걸음 0.0025 s). 관측: 자기 중심 지역 지도 + 탐사한 영역의 최대 사각형을 고정 크기로 줄인 전역 탐사 지도 + 방향.
- 행동: 전진 1 칸·좌 90°·우 90° **직접**(끝단). 보상: 탐사율 ρ 가 늘면 clip(10·(ρ_t² − ρ_{t−1}²), 0, 1) — 후반일수록 큼, 아니면 −0.005. ρ ≥ 0.99 면 +1 로 끝, 충돌 −1 로 끝.
- 커리큘럼: 평가 10 회 평균 탐사율 > 0.95 면 다음 단계. **옛 단계 맵을 버리지 않고 같이 돌려**(벡터 환경) 잊어버림을 막음.
- 결과: 일반 커리큘럼·커리큘럼 없음보다 표본 효율·일반화가 좋고 잊어버림이 줆.

**우리에게**
- 우리 커리큘럼(legged_gym 식 올림·내림)에 "옛 단계 맵을 섞어 둔다"를 그대로 적용할 근거. ρ² 보상은 "마지막 5 % 마무리"를 북돋는 우리 완료 보너스의 대안.

### Tai, Paolo, Liu 2017 — 가상→실물 연속 제어 (`2017_tai_virtual-to-real-mapless-nav.pdf`)

**어떻게 하나**
- 입력 14 차원: 라이다 10 방향(−90~90°, 0~1 정규화) + 직전 행동 + 목표(거리·각도). 512 FC 3 층 → **선속도(시그모이드, 최대 0.5 m/s)·각속도(tanh, 최대 1 rad/s)**. 뒤로 가기는 막음(라이다가 뒤를 못 봄).
- 비동기 DDPG(수집 스레드 분리). 보상: 도착 +, 충돌 −(둘 다 끝), 그 밖에는 목표까지 거리 감소량 × 계수.
- V-REP 10×10 m 두 환경에서 학습 → 실물 Turtlebot 에 **미세조정 없이** 적용, 충돌 없이 목표 도달.

**우리에게**
- 탐사가 아니라 지도 없는 목표 주행이지만, 성긴 라이다 + 직전 행동 → 속도 직접 출력이 실물로 넘어간 최초 사례. 우리 관측에 직전 행동을 넣은 것과 같은 설계.

### Sivashangaran et al. 2025 — 지도 없는 탐사, 분포 밖 학습 (`2025_sivashangaran_exploration-without-maps.pdf`)

**어떻게 하나**
- 관측: 앞쪽 깊이 170 개(1D). 행동: 스로틀·조향 연속(애커먼 차). PPO, 2 층 FC tanh, actor·critic 공유.
- 탐사 맵이 아니라 **여러 방향 경주로에서 시간 최적 주행**을 학습하면 공간 추론·탐사 성향이 생긴다는 주장. 보상 = 스로틀²(속도). 시뮬에서 조향을 빠르게 흔드는 물리 엔진 허점을 써서 실물에서 지그재그 → 직전·현재 조향 부호가 바뀌면 벌점을 넣어 무작위화 없이 실물 이전.
- 탐사 덮음은 궤적이 지난 10 m² 칸 수로 셈(지도 없음). 50×50 m 어수선한 시뮬에서 OOD PPO 88 %(2e7 표본) vs 같은 곳 직접 학습 PPO 8 %·SAC 52 %(6e7).

**우리에게**
- 속도 직접 출력 + 실물 무변환 이전 사례이고, **시뮬 허점(빠른 흔들기) → 행동 변화 벌점**은 우리 "정지·제자리 회전 벌점"·가속 한도와 같은 문제. 다만 지도·구멍 개념이 없어 "빈틈없는 지도"라는 우리 목표와는 거리가 멂.

### Liu et al. 2026 — VORL-EXPLORE, 고전 배정 + RL 실행 혼합 (`2026_liu_vorl-explore.pdf`)

**어떻게 하나**
- 배정: 프론티어 점수 = 효용·BFS 거리·위험(혼잡)을 정규화해 합침, 보로노이식 로봇 간 밀어내기. "실행 충실도" p(로지스틱 회귀, 점유·혼잡·최근 진척 등 8 특징)가 위험한 프론티어를 깎음.
- 실행: A* 안내 행동과 **RL 반응 정책**(PPO, 64×64 무작위 격자·64 대에서 학습, 감지 반경 3 칸) 중 p 로 고름. 두 문턱 τH > τL 와 K 걸음 연속 조건의 히스테리시스로 떨림 방지, 막힘·진동 시 복구 동작.
- 스위치는 최근 진척·안전 결과로 만든 의사 레이블로 온라인 갱신. 동적 격자에서 성공률↑, 경로·중복↓. Gazebo 공장 시연.

**우리에게**
- 우리 BT 의 "RL ↔ 고전 프론티어" 넘김을 **문턱 두 개 + 연속 K 걸음** 히스테리시스로 하라는 구체 설계. 이산 격자·다중 로봇이라 그대로 옮기지는 못한다.

## 4. 벤치마크·지표

### Xu et al. 2022 — Explore-Bench (`2022_xu_explore-bench.pdf`)

**어떻게 하나**
- 데이터: 탐사가 어려운 장면 요소 4 가지(고리, 좁은 복도, 모서리, 여러 방)와 그 조합 6 장면. Gazebo 설계도에서 격자 맵 자동 생성.
- 지표: 효율 — 탐사율 99 % 도달 시간 T_total, 90 % 도달 시간 T_topo(위상 구조를 얻은 시점). 협조 — 로봇별 탐사 넓이 표준편차, 겹친 넓이 비율.
- 플랫폼 3 단: 격자 시뮬(학습·빠른 평가, 12 배 빠름) → Gazebo → 원격 실물 시험대. 같은 데이터 흐름.
- 결과: 비용(가장 가까운)·RRT 표본·퍼텐셜 장 프론티어와 DRL 1 개 비교 — 퍼텐셜 장이 종합 최고, 상위 결정만 학습한 DRL 이 효율은 가장 낮음(협조는 좋음).

**우리에게**
- 우리 "95 % 도달 시간"에 **90 % 도달 시간을 같이 재는** 것과 장면 요소별(고리·좁은 통로·방) 따로 보고하는 것을 더할 만하다. "상위 결정만 배운 DRL 은 고전보다 못할 수 있다"는 반례.

---

## 방식 갈래 정리

### 고전: 어디로 갈지 고르는 규칙

| 갈래 | 대표 | 고르는 식 | 장점 | 약점 |
|---|---|---|---|---|
| 가장 가까운 프론티어 | Yamauchi 1997, Holz 2010, explore_lite(gain_scale 작게) | argmin 경로 길이 | 단순, 2D 실내에서 의외로 강함(Holz: 정보 이득보다 짧음) | 방을 두 번 들어감, 되돌아가기, 자세 오차에 약함(Chen 2019) |
| 효용(정보 이득) 프론티어 | González-Baños 2002, Burgard 2005, explore_lite | I·e^(−λL) 또는 U − β·V | 큰 열린 곳을 먼저 | λ·β 조정 필요, 이득 계산이 계산량 대부분(ARiADNE: 90 %) |
| 샘플링 NBV | RH-NBVP 2016 | RRT 가지 이득 최대, 첫 마디만 실행 | 3D·대형에 확장 | 2D 실내에서 효용 프론티어보다 경로 김(ARiADNE 표) |
| 계층(전역 순서 + 지역 시점) | TARE 2021, FUEL 2021 | 지역 시점 집합 + TSP/ATSP, 전역 성긴 순회 | 근시안 줄임, 현재 고전 최고 | 구현 복잡, 3D 드론·대형 중심 |

### 학습: RL 이 무엇을 내나

| 갈래 | 대표 | 행동 | 이동은 누가 | 특징 |
|---|---|---|---|---|
| RL 이 목표 고르기 | Niroui 2019, Li 2020, ANS 2020, ARiADNE 2023, Cao 2024 | 프론티어·지도 칸·장기 목표·이웃 시점 | A*·Fast Marching·Nav2 같은 고전 플래너 | 학습이 쉽고 실물 이전이 쉬움. 속도·회전 낭비는 못 줄임. 2D 실내에선 ARiADNE 계열이 가장 강함 |
| RL 이 동작·속도 직접 | Chen 2019(이산 6 동작), CCRL 2023(이산 3), Tai 2017(v, ω), Sivashangaran 2025(스로틀·조향) | 속도·동작 | 정책 자신 | 넓은 곳은 밟고 좁은 곳은 줄이는 것까지 배울 수 있음. 장기 기억·계획은 지도 입력(Chen, CCRL)이 없으면 약함. 우리 방식이 여기 |
| 혼합 | VORL-EXPLORE 2026(배정 고전 + 실행 A*/RL 스위치), ANS(전역 RL + 지역 학습 정책) | 층마다 다름 | 섞음 | 우리 BT 대체 경로와 같은 구조. 스위치 떨림 방지가 관건 |

공통 학습 기법: 덮음 증가 보상(ANS, Chen, CCRL), 이동 거리 벌점(ARiADNE, Li), 끝 보상·끝내기 행동(Li 0.85, CCRL·ARiADNE 0.99), 정답을 보는 비평가(Cao 2024), 누적 커리큘럼(CCRL), 격자 시뮬로 빠른 학습(CCRL, Explore-Bench).

### 흔한 지표

- **경로 길이**(탐사 완료까지): Holz, ARiADNE, Cao 2024. 가장 흔함.
- **시간**: 완료(makespan, Cao 2024), 99 %·90 % 도달 시간(Explore-Bench), 초당 탐사 부피(TARE).
- **덮음**: 고정 걸음 뒤 덮은 넓이 m²·비율(ANS, Chen), 시간에 따른 덮음 곡선(ANS, Chen). 표면 기준 덮음은 TARE 만.
- **계산 시간**(TARE, FUEL, Cao 2024), 다중 로봇이면 부하 균형·겹침(Explore-Bench).
- 완료 판정은 대개 "정답의 99 % 탐사" 또는 "프론티어 없음"(Holz, Cao 2024). 우리 "벽 면 덮음 95 % + 끝 선언 + 지도 닫힘"은 이보다 엄격하고 끝 선언을 따로 평가하는 점이 다르다.
