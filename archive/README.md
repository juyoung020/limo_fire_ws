# archive

안 쓰는 것을 보관한다. 빌드에서 빠진다(`COLCON_IGNORE`).

- `robot-agent-history-*.bundle`: 저장소를 비우기 전 옛 커밋 기록 백업(저장소에는 올리지 않음). 되살리기: `git clone <bundle> 폴더`
- `explore_env/`: 탐사 RL 첫 초안 GPU 환경(원래 `train/explore/env`, 2026-10-23). 목표점 행동 48·칸 DDA 라이다·덮음 보상, `train/maps` 의 `.fmap` 맵 읽기(`src/fmap_host.cpp`)가 여기 있다. 실물 녹화로 맞춘 `train/explore/sim` 이 대체했다. 빌드는 `../sim/include` 의 `detmath.h` 를 가리키던 채라 옮긴 뒤에는 그대로 안 된다.
- `limo_base/`: WeGo/AgileX 섀시 드라이버(원래 `src/external/wego_limo_pro/limo_ros2/limo_base`, 2026-10-26 로봇 복사본). 그대로 가져와 받고 버리던 리모컨(0x241)·모터 상태(0x251~0x254)·바퀴 거리(0x311)만 더 낸 우리 판이 `src/limo_fire/limo_fire_base` (`diff -r archive/limo_base src/limo_fire/limo_fire_base` 로 다른 줄을 본다). 로봇의 `~/wego_ws` 빌드(대시보드용)에는 그대로 있다.
- `explore_net_v2/`: 탐사 RL 신경망 구조 버전 2 의 Netron 그림(`explore_net_v2.png`)과 모양만 담은 ONNX(원래 `docs/images`·`docs/models`, 2026-11-04). 지금 신경망과 많이 달라 옮겼다. 마지막 구조로는 `rl/train/explore/network/tools/onnx_export` (b48688a 판에서 빌드)로 다시 만든다.
- `explore_roadmap_1104/`: 탐사 RL 첫 학습 계획(원래 `train/explore/ROADMAP.md`, 2026-11-04)과 학습 시뮬레이터 계획(원래 `docs/rl_explore_plan.md`, 일정·신경망 구조 버전 2 기준). 정책 하나가 속도를 바로 내는 구조의 단계·통과 조건·학습 속도 기록. 두 층 구조로 바꾸면서(2026-11-12) 옮겼다. 코드 주석의 "ROADMAP S2·S4·S5" 는 이 문서를 가리킨다.
- `rl/`: 탐사 강화학습 전부(2026-11-16 그만둠 — 실물에서 시뮬과의 격차가 컸고 A* 아래층이 훨씬 빠르고 잘 됨). PPO 학습기·신경망·학습 러너·RL 실물 노드 `explore_rl`·두 층 계획·조준 RL 계획을 원래 경로 그대로 옮겼다. 마지막 결과·되살리기(옮기기 앞 커밋 `b48688a`)는 [rl/README.md](rl/README.md).
- `perception_benchmark/`: 인식 벤치마크 노드(원래 `src/limo_fire/limo_fire_perception/src/perception_benchmark_node.cpp`)와 그 덮어쓰기 설정 `benchmark.yaml`(2026-11-18 옮김). Webots 정답(`/gt/*`)과 YOLO 를 장마다 견줘 표를 쓰던 것 — 결과는 `docs/benchmarks/`. 짝짓기·AP(`det_metrics.hpp`)는 패키지에 남아 있어 되돌리면 그대로 빌드된다.
- `refs/README.md`: 원래 `refs/README.md` 의 탐사 RL 참고 저장소(marmotlab ARiADNE 등) 절 (2026-11-20 옮김).
- `refs/papers/header_fare_diff.md`: HEADER·FARE 논문과 우리 탐사 RL(직접 제어 v10)의 차이(원래 `refs/papers/`, 2026-11-16 옮김). 견줄 대상인 탐사 RL 을 그만둬서 옮겼다. 논문 받는 법과 판단 모델 쪽 선행연구는 `refs/` 에 그대로 있다.
