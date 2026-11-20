# archive/refs

그만둔 탐사 강화학습(2026-11-16)에만 쓰던 참고 자료. 원래 `refs/` 에 있던 것을 옮겼다.

- `papers/header_fare_diff.md`: HEADER·FARE 논문과 우리 탐사 RL 의 차이 (2026-11-16 옮김).
- 아래 절: 원래 `refs/README.md` 의 "탐사 RL 참고 저장소" 절 (2026-11-20 옮김). 받는 곳은 지금도 `refs/code/` 다(git 에 안 넣음).

## code/ — 탐사 RL 참고 저장소 (NUS marmotlab, 2026-11-11 받음)

HEADER 의 계획기·학습 코드는 공개되지 않았다(지형 분할 모듈만 공개). 아래는 같은 연구실의 전작과 공개된 부분이다.

| 폴더 | 내용 | 라이선스 | 받기 |
|---|---|---|---|
| `ARiADNE` | ARiADNE (ICRA 2023) 원본: 어텐션 신경망·학습 환경·학습된 모델 (Python) | 표기 없음 | `git clone --depth 1 https://github.com/marmotlab/ARiADNE.git refs/code/ARiADNE` |
| `large-scale-DRL-exploration` | RAL 2024 후속: 큰 지도용, GT 를 보는 critic (Python) | 표기 없음 | `git clone --depth 1 https://github.com/marmotlab/large-scale-DRL-exploration.git refs/code/large-scale-DRL-exploration` |
| `ARiADNE-ROS-Planner` | ARiADNE 를 ROS 1/2 계획기로 만든 것 (Python) | MIT | `git clone --depth 1 https://github.com/marmotlab/ARiADNE-ROS-Planner.git refs/code/ARiADNE-ROS-Planner` |
| `ARiADNE2-ROS-Planner` | HEADER 저장소: 지형 분할 모듈만 (C++) | MIT | `git clone --depth 1 https://github.com/marmotlab/ARiADNE2-ROS-Planner.git refs/code/ARiADNE2-ROS-Planner` |

라이선스 표기가 없는 두 저장소는 구조만 참고하고 코드를 옮겨 쓰지 않는다.
