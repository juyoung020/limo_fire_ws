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

## papers/ — 선행 연구 (2026-11-04 받음)

주제마다 폴더 하나, 폴더의 `README.md` 에 논문별 링크·방법 요약·우리에게 쓸 점. PDF 는 합법 공개본만 받았고 공개본이 없으면 링크만 적었다.

PDF 는 git 에 넣지 않는다. `refs/papers/fetch.sh` 가 `sources.tsv` 의 공개본을 같은 자리에 받고 체크섬으로 확인한다(`fetch.sh smoke` 처럼 일부만도 됨). 출판사가 봇 확인을 거는 3 편은 끝에 주소를 알려 주니 브라우저로 받아 둔다. 우리 시스템에 어떻게 쓸지는 `docs/prior_work_*.md`.

| 폴더 | 주제 |
|---|---|
| [`papers/launcher`](papers/launcher/README.md) | 탁구공 발사로 불 모양 표적 맞추기 — 조준·탄도(항력)·탄착 피드백·학습 보정·RL |
| [`papers/exploration`](papers/exploration/README.md) | 실내 프론티어 탐사 — 고전 프론티어·NBV·계층 플래너·RL 탐사·지표 |
| [`papers/smoke`](papers/smoke/README.md) | 연기 모드 — 연기 속 센서 실측·라이다 연기 거르기·저시정 융합·초음파 MCL·열화 주입 |

## datasheets/
부품 데이터시트(MLX90640, HLK-LD2450, HC-SR04 …).
