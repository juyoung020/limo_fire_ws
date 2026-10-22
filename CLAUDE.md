# 작업 규칙

## 작업 단위마다 pull → 작업 → 커밋·푸시
- 작업을 시작하기 전에: `git pull --rebase --autostash`.
- 작업 하나가 끝나면 바로: 바뀐 경로만 지정해 커밋(`git add <경로>`, `git add -A` 금지) → `git push`.
- 커밋 메시지는 한국어로, 무엇을 왜 바꿨는지 한 줄.
- 커밋·PR 에 Claude 기여자 표기(`Co-Authored-By: Claude …`, "Generated with Claude Code" 등)를 넣지 않는다. 작성자는 사람만.

## 언어: Rust 또는 C++ 만
- 새로 쓰는 코드는 **Rust 또는 C++** 로만 쓴다. Python 은 쓰지 않는다.
  - ROS 2 노드: C++(`rclcpp`, `ament_cmake`)가 기본. Rust(`rclrs`)는 ROS 의존이 없는 라이브러리·도구에 우선 쓴다.
  - launch 파일: Python 대신 **XML 또는 YAML launch**(`*.launch.xml`, `*.launch.yaml`).
  - ESP32 펌웨어(`firmware/`): C++ (Arduino 프레임워크 / PlatformIO).
  - GPU 학습 환경: C++/CUDA.
- 예외: `src/external/` 의 남의 코드는 원래 언어 그대로 두고 고치지 않는다.

## 폴더 구조: ROS 2 관례
```
limo_fire_ws/                  워크스페이스 루트 (여기서 colcon build)
├── src/
│   ├── external/              남의 코드 — 고치지 않음, 출처·커밋은 README 표에
│   └── limo_fire/             우리 패키지
│       ├── limo_fire_description/   URDF/xacro, 메시, RViz 설정
│       ├── limo_fire_bringup/       launch(실물·시뮬), config/*.yaml 파라미터
│       ├── limo_fire_msgs/          우리 메시지·서비스·액션
│       ├── limo_fire_esp32_bridge/  ESP32 ↔ ROS
│       ├── limo_fire_perception/    불 탐지·크기 추정·소방 시설 인식
│       ├── limo_fire_people/        레이더 추적·구역 인원
│       ├── limo_fire_navigation/    Nav2 파라미터·지도·연기 모드
│       ├── limo_fire_bt/            BT 플러그인 + 트리 XML
│       ├── limo_fire_rl/            RL 정책 Action 서버
│       └── limo_fire_report/        웹 앱(rosbridge)·음성 안내
├── sim/                       Isaac Sim 확장·USD, maps/  [COLCON_IGNORE]
├── train/                     GPU 학습 환경 (C++/CUDA/Rust, ROS 아님) [COLCON_IGNORE]
├── firmware/                  ESP32 코드
├── docs/                      문서
├── refs/                      참고 자료                    [COLCON_IGNORE]
└── archive/                   안 쓰는 것                   [COLCON_IGNORE]
```
- 패키지는 역할마다 하나, 이름은 `limo_fire_<역할>`. 새 패키지는 위 목록에 맞춰 만들고, 목록에 없는 역할이면 이 파일부터 고친다.
- 패키지 안: `include/<패키지명>/`, `src/`, `launch/`, `config/`, `test/` (ROS 2 기본 배치).
- 실행 파일·파라미터를 바꾸고 싶은 외부 패키지는 고치지 말고 `limo_fire_bringup` 에 우리 판을 만든다.

## URDF·TF·센서
- 좌표·단위는 REP-103(미터·라디안, x 앞·y 왼쪽·z 위), 프레임은 REP-105: `map → odom → base_link`.
- 프레임 이름은 실물 LIMO 드라이버와 같게: `odom`, `base_link`, `laser_link`, `imu_link`, `camera_link`. `base_footprint` 는 쓰지 않는다(실물에 없음).
- 센서·노즐 장착 위치는 **전부 URDF(xacro) 한 곳**에. launch 에 `static_transform_publisher` 를 흩어 두지 않는다. 모듈마다 xacro 매크로 파일 하나.
- 카메라류는 `<이름>_link`(장착)와 `<이름>_optical_frame`(z 앞) 을 따로 둔다.
- 팬·틸트는 revolute 관절, 서보 각도는 `/joint_states` 로 낸다.
- TF 하나는 발행 주체 하나: `odom → base_link` 는 EKF 만, `map → odom` 은 위치 추정 하나만.
- 메시지는 표준 타입 우선(`sensor_msgs/Image`, `sensor_msgs/Range`, `sensor_msgs/JointState`, `std_srvs/SetBool` …). 맞는 게 없을 때만 `limo_fire_msgs` 에 만든다. 헤더 `frame_id` 는 URDF 링크 이름과 같게.
- 주기·포트·임계값은 `config/*.yaml` 파라미터로. 코드에 박지 않는다.

## 빌드·환경
- `source env.sh` (conda 를 PATH 에서 빼고 ROS 2 Humble + install).
- `colcon build --packages-select <패키지>` 로 바꾼 것만 빌드.
