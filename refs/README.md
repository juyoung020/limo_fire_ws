# refs

참고 자료. 빌드에서 빠지고(`COLCON_IGNORE`), `code/` 아래 저장소는 git 에 넣지 않는다(받는 법은 아래).

## code/ — 교수님(inmo-jang) 저장소 (2026-10-23 받음)

| 폴더 | 내용 | 받기 |
|---|---|---|
| `prof_py_bt_ros` | 교수님 Python 행동 트리 런타임 + ROS 2 예제(turtlesim) | `git clone --depth 1 https://github.com/inmo-jang/py_bt_ros.git refs/code/prof_py_bt_ros` |
| `prof_BehaviorTreeROS2_tutorial` | BehaviorTree.CPP ROS 2 유틸·튜토리얼 | `git clone --depth 1 https://github.com/inmo-jang/BehaviorTreeROS2_tutorial.git refs/code/prof_BehaviorTreeROS2_tutorial` |
| `prof_webots_ros2` | Webots ROS 2 패키지(교수님 포크, TurtleBot3 Nav2 실습) | `git clone --depth 1 https://github.com/inmo-jang/webots_ros2.git refs/code/prof_webots_ros2` |
| `prof_rl_labs` | 강화학습 수업 실습 자료 | `git clone --depth 1 https://github.com/inmo-jang/rl_labs.git refs/code/prof_rl_labs` |

LIMO Webots 시뮬(`webots_ros2_limo`)은 쓰는 코드라 `src/external/` 에 있다.

탐사 RL 참고 저장소(NUS marmotlab — ARiADNE 등, 2026-11-11 받음)는 강화학습을 그만둬서 [archive/refs/README.md](../archive/refs/README.md) 로 옮겼다(2026-11-20).

## papers/ — 논문 (PDF 는 git 에 넣지 않는다, 아래 명령으로 받는다)

| 파일 | 논문 | 받기 |
|---|---|---|
| `HEADER_2510.15679.pdf` | HEADER (Cao et al., 2025): ARiADNE 를 계층형으로 넓힌 탐사 RL | `curl -L -o refs/papers/HEADER_2510.15679.pdf https://arxiv.org/pdf/2510.15679` |
| `FARE_2601.14681.pdf` | FARE (Liao et al., 2026): HEADER 위에 LLM 상위 판단 | `curl -L -o refs/papers/FARE_2601.14681.pdf https://arxiv.org/pdf/2601.14681` |

두 논문과 우리 탐사 RL(2026-11-16 그만둠)의 차이는 [archive/refs/papers/header_fare_diff.md](../archive/refs/papers/header_fare_diff.md) 에 정리했다.
판단 모델로 프론티어를 고르는 구상과 가까운 선행연구는 [papers/decision_explore_prior.md](papers/decision_explore_prior.md) 에 정리했다.

## 판단 전용 모델
로봇의 상위 판단만 맡는 시각·언어 모델(API 전용 · 가중치 공개)을 [decision_models.md](decision_models.md) 에 정리했다.

## datasheets/
부품 데이터시트(MLX90640, HLK-LD2450, HC-SR04 …).
