# 작업 규칙

폴더 구조·패키지 역할·URDF/TF/메시지 세부 규약은 [docs/conventions.md](docs/conventions.md) 에 있다.

## Git: 작업 단위마다 pull → 작업 → 커밋·푸시
- 시작 전 `git pull --rebase --autostash`.
- 바뀐 경로만 지정해 커밋(`git add <경로>`, `git add -A` 금지)하고 바로 `git push`.
- 커밋 메시지는 한국어 한 줄, 무엇을 왜 바꿨는지.
- 커밋·PR 에 Claude 기여자 표기(`Co-Authored-By: Claude …`, "Generated with Claude Code" 등)를 넣지 않는다. 작성자는 사람만.

## 코드
- 새 코드는 C++ 또는 Rust 만. Python 은 쓰지 않는다. launch 는 XML/YAML.
- `src/external/` 는 고치지 않는다(출처·커밋은 README 표에). 바꿀 게 있으면 `limo_fire_bringup` 에 우리 판을 만든다.
- 패키지는 역할마다 하나, 이름은 `limo_fire_<역할>`. 목록에 없는 역할이면 docs/conventions.md 부터 고친다.
- 주기·포트·임계값은 `config/*.yaml` 파라미터로. 코드에 박지 않는다.

## 설계: 처음부터 강건하게
지금은 구조를 짜는 단계다. 방향이 바뀌거나 모듈이 늘어도 버티는 구조로 처음부터 설계한다.
- 기존 코드는 참고일 뿐 기준이 아니다. 모자라면 맞춰 쓰지 말고 다시 설계한다.
- 설정·데이터는 코드와 나누고, 새 항목은 파일 추가만으로 되게 한다.
- 같은 기능을 두 벌 만들지 않고, 모듈은 정해진 인터페이스로만 잇는다.
- 시작 순서·타이밍이 운 좋게 맞기를 기대지 않는다. 준비를 확인하고, 실패는 알린다.

## 잔재 코드
- 안 쓰는 코드는 지운다. 남겨야 하면 `archive/` 한곳에 두고, 임시 코드에는 `TODO(제거):` 를 달아 grep 으로 찾게 한다.

## URDF·TF
- REP-103(미터·라디안, x 앞·y 왼쪽·z 위), REP-105(`map → odom → base_link`).
- 센서·노즐 장착 위치는 전부 URDF(xacro) 한 곳에. launch 에 `static_transform_publisher` 를 흩어 두지 않는다.
- TF 하나는 발행 주체 하나(`odom → base_link` 은 EKF 만).
- 프레임 이름은 실물 LIMO 드라이버와 같게. `base_footprint` 는 쓰지 않는다.

## 빌드·환경
- `source env.sh` (conda 를 PATH 에서 빼고 ROS 2 Humble + install).
- `colcon build --packages-select <패키지>` 로 바꾼 것만 빌드.
