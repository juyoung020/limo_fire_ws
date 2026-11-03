# 작업 규칙

폴더 구조·패키지 역할·URDF/TF/메시지 세부 규약은 [docs/conventions.md](docs/conventions.md) 에 있다.

## Git: 사용자가 맡긴 작업 하나 = 커밋 하나
- 시작 전 `git pull --rebase --autostash`. 바뀐 경로만 지정해 커밋(`git add <경로>`, `git add -A` 금지).
- **커밋 하나 = 사용자가 맡긴 작업 하나.** 작업 중 중간 단계(조사·수정·문서·CHANGELOG)를 따로 올리지 않는다.
  중간 저장은 로컬 커밋으로 해도 되지만 푸시 전에 `git reset --soft @{u}` 등으로 작업 단위로 합친다.
  이미 올린 작업의 후속 수정은 그 커밋에 `--amend` (HEAD 가 그 커밋인지 확인) 후 `git push --force-with-lease`. 다른 사람이 위에 올렸으면 먼저 묻는다.
- 커밋 메시지는 한국어 한 줄, 무엇을 왜 바꿨는지. 작성자는 사람만 — Claude 기여자 표기(`Co-Authored-By: Claude …` 등)를 넣지 않는다.
- 사용자가 알아챌 변경은 같은 커밋에서 `CHANGELOG.md` 의 `## [Unreleased]` 에 적는다 ([Keep a Changelog 1.1.0](https://keepachangelog.com/ko/1.1.0/)).
  - 종류는 Added · Changed · Deprecated · Removed · Fixed · Security, 이 순서로.
  - 항목은 사람이 읽는 한 줄(100 자 이내): 무엇이 생기거나 바뀌었는지. 경로·파일 이름 나열·수치·원인·디버깅 이야기는 커밋 메시지와 docs 에.
  - 커밋 메시지를 옮겨 적지 않는다. 같은 기능의 여러 변경은 항목 하나로.
  - 읽는 사람은 이 저장소를 쓰는 팀원이다: "무엇을 할 수 있게 됐나 / 무엇이 달라졌나 / 내가 할 일(새로 필요한 것)" 만. 구현 방법(커널·라이브러리·알고리즘 이름)과 내부 용어는 쓰지 않는다.
  - 새로 쓰기 전에 기존 항목(지난 버전들)의 말투·길이와 맞는지 한 번 읽고 맞춘다.
  - 릴리스 때 `## [x.y.z] - YYYY-MM-DD` 로 묶고 [유의적 버전](https://semver.org/lang/ko/)을 따른다.
- 위 규칙은 `tools/repo_check` 가 검사한다. Claude Code 는 `.claude/settings.json` 훅으로 `git commit`·`git push` 전에 자동으로 돌리고,
  어기면 막는다(CHANGELOG 형식, 아직 안 올린 커밋 여러 개가 같은 영역을 건드림, 기여자 표기). 사람은 `tools/repo_check/target/release/repo_check push` 로 직접.

## 답하기
- 외부 도구·사양·논문·데이터시트는 검색하고 출처와 함께 답한다. 기억으로 단정하지 않는다.
- 결과는 있는 그대로: 안 맞는 것·못 한 것·기준 미달을 숨기지 않는다. 통과 기준은 결과를 보고 바꾸지 않고, 바꾸면 까닭을 적는다.

## 코드
- 새 코드는 C++ 또는 Rust 만. Python 은 쓰지 않는다. launch 는 XML/YAML.
- `src/external/` 는 고치지 않는다(출처·커밋은 README 표에). 바꿀 게 있으면 `limo_fire_bringup` 에 우리 판을 만든다.
- 패키지는 역할마다 하나, 이름은 `limo_fire_<역할>`. 목록에 없는 역할이면 docs/conventions.md 부터 고친다.
- 주기·포트·임계값은 `config/*.yaml` 파라미터로. 코드에 박지 않는다.
  예외: 탐사 RL 의 관측 크기·보상·끝 조건·커리큘럼(`train/explore/mdp`)과 PPO 하이퍼파라미터(`train/explore/ppo`)는 코드 안에 둔다. 바꾸면 다시 컴파일하고 체크포인트에서 이어 학습한다.

## 설계: 처음부터 강건하게
지금은 구조를 짜는 단계다. 방향이 바뀌거나 모듈이 늘어도 버티는 구조로 처음부터 설계한다.
- 기존 코드는 참고일 뿐 기준이 아니다. 모자라면 맞춰 쓰지 말고 다시 설계한다.
- 설정·데이터는 코드와 나누고, 새 항목은 파일 추가만으로 되게 한다.
- 같은 기능을 두 벌 만들지 않고, 모듈은 정해진 인터페이스로만 잇는다.
- 시작 순서·타이밍이 운 좋게 맞기를 기대지 않는다. 준비를 확인하고, 실패는 알린다.

## 잔재 코드
- 안 쓰는 코드는 지운다. 남겨야 하면 `archive/` 한곳에 두고, 임시 코드에는 `TODO(제거):` 를 달아 grep 으로 찾게 한다.

## 시뮬 = 실물
- 시뮬 값은 실물 측정(또는 출처 있는 자료)으로 정하고 근거를 config 주석·`train/explore/sim/analysis` 에 남긴다. 짐작한 값은 "잠정" 이라 적는다.
- 맞추기(fit)와 확인(validate) 녹화를 나눈다(`analysis/calibration.yaml`). 같은 녹화로 맞추고 확인하지 않는다.
- 실물 녹화는 `record.launch.xml` 로 늘 같은 토픽을 담고, 녹화마다 `analysis/bags.md` 에 조건 한 줄.

## 실물 로봇
- 주행 명령 전에 충전기를 뺐는지·주변이 비었는지·사람이 곁에 있는지 확인받는다. 한 번에 움직이는 거리는 정한 범위 안(기본 0.6 m).
- ROS 명령(`/cmd_vel`)은 리모컨이 켜져 오토 모드(SWB 위)여야 바닥판이 받는다.
- 바닥판 포트(`/dev/ttylimo`)는 드라이버 하나만 연다(대시보드 `limo_base` 와 `robot.launch.xml` 을 같이 켜지 않는다).
- 프로세스는 PID 나 실행 파일 전체 경로로 끈다. 자기 명령줄에도 들어 있는 패턴으로 `pkill -f` 하지 않는다.

## URDF·TF
- REP-103(미터·라디안, x 앞·y 왼쪽·z 위), REP-105(`map → odom → base_link`).
- 센서·노즐 장착 위치는 전부 URDF(xacro) 한 곳에. launch 에 `static_transform_publisher` 를 흩어 두지 않는다.
- TF 하나는 발행 주체 하나(`odom → base_link` 은 EKF 만).
- 프레임 이름은 실물 LIMO 드라이버와 같게. `base_footprint` 는 쓰지 않는다.

## 빌드·환경
- `source env.sh` (conda 를 PATH 에서 빼고 ROS 2 Humble + install).
- `colcon build --packages-select <패키지>` 로 바꾼 것만 빌드.
