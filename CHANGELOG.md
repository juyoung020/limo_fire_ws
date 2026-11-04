# Changelog

이 프로젝트의 주요 변경 사항을 기록합니다.

형식은 [Keep a Changelog](https://keepachangelog.com/ko/1.1.0/)를 따르고,
버전은 [유의적 버전](https://semver.org/lang/ko/)을 따릅니다.
항목은 사용자가 알아챌 변화만 한 줄로 적습니다. 원인·수치·디버깅 과정은 커밋 메시지와 `docs/` 에 있습니다.

## [Unreleased]

### Added

- 인식 성능 벤치마크: 시뮬에서 탐지 정확도와 처리 속도를 잼.
- 섀시의 리모컨·모터 상태와 바퀴 거리 토픽.
- 섀시 없이 센서만 켜는 실행 옵션 `base:=false`.
- 실물 녹화 실행 파일과 녹화 목록.
- 탐사 강화학습 학습 환경: 실물 LIMO 에 맞춘 시뮬레이터, 학습 맵 520 개, 움직이는 사람.
- 탐사 강화학습 학습기와 학습 진행을 보는 뷰어.
- 탐사 강화학습과 비교할 프론티어 탐사 주행.
- 중간고사 대비 ROS 구조 정리 문서.
- 커밋·푸시 전에 변경 기록과 커밋 묶음을 검사하는 도구.
- 탐사 강화학습 신경망 구조 그림과 Netron 으로 열 수 있는 모델 파일.
- 학습 뷰어 재생에서 로봇 실제 모양·센서 시야·로봇이 만든 지도·사람·걸음별 보상을 봄.
- 탐사 강화학습 연기 모드: 라이다·카메라를 가리고 바퀴와 IMU 만으로 움직이는 상황을 시뮬에서 켜고 끌 수 있음.

### Changed

- 탐사 강화학습 학습이 약 7 배 빨라짐. 빠른 시뮬은 OptiX 헤더를 받아야 켜짐.
- 탐사 강화학습 신경망을 바꿔, 예전에 저장한 학습 체크포인트는 이어 쓸 수 없음.
- 섀시 드라이버를 우리 패키지 `limo_fire_base` 로 바꿈.
- 탐사 완료를 "보이는 벽을 다 봤는지"로 판정함.
- 탐사 강화학습이 작은 방에서 큰 집으로 넘어가는 기준을 "최근 판들의 평균 덮음"으로 바꿈.
- 탐사 중 1 분 동안 새 벽을 하나도 못 보면 판을 실패로 끝냄(시간만 끌며 버티지 않게).
- 탐사 강화학습 보상 점수와 판 끝 조건을 나눠 적어, 보상 값을 한곳에서 바로 고칠 수 있음.

### Removed

- 옛 탐사 학습 환경.

### Fixed

- 실물 로봇 모델(URDF)의 차체 크기·바퀴 간격과 깊이 카메라 높이·기울기가 실제와 다르던 문제.
- Nav2 로 저장한 지도를 탐사 시뮬에서 열지 못하던 문제.

## [0.6.0] - 2026-10-29

### Added

- 변경 기록과 센서 토픽 문서.
- Webots 맵 3 개(덕트호스 미로·사무실·창고)와 맵별 정답 물체·순찰점.
- 시뮬·지도·Nav2·인식을 한 번에 띄우는 `sim.launch.xml world:=<맵>`.
- RViz 지도에서 물체 아이콘을 누르면 그 물체 사진을 봄.
- README 데모 GIF.

### Changed

- 실물 TF 를 우리 URDF 하나로 내고, 카메라 장착 위치를 실측값으로 고침.
- 시뮬도 실물과 같은 URDF·지도 만들기·Nav2·인식 실행 파일을 씀.
- 운용 화면을 RViz 하나로 합침: 지도·Nav2 경로·물체·YOLO 영상.
- Nav2 를 단순하게 바꿈(Regulated Pure Pursuit·Smac 2D). 덕트호스 미로 끝까지 주행함.
- 탐지 영상에 분할 마스크 테두리를 그림.

### Fixed

- 인식 실행 파일의 `mapper:=false` 가 무시되던 문제.
- 교수님 launch 에서 가끔 TF 가 안 나오던 문제.

## [0.5.0] - 2026-10-28

### Changed

- Jetson 에서 YOLO 한 장 처리 시간을 10.3 ms 로 줄임(전처리·마스크를 GPU 로).
- 실물 깊이 영상을 19 Hz 에서 30 Hz 로 올림(Jetson 클럭 고정, 안 쓰는 카메라 스트림 끔).
- Cartographer 회전 필터와 자세 발행 주기를 조정.

## [0.4.0] - 2026-10-27

### Added

- 탐지한 소화기·불·문 등을 지도 좌표에 올리는 `object_mapper` 노드.
- 지도 위 물체를 종류별 아이콘·번호·불 위험 반경·한글 현황판으로 표시.
- 탐지와 분할을 같이 하는 YOLO26s-seg 모델과 가중치 Release.
- 인식 모델 발표용 문서.

### Changed

- 인식 노드를 YOLO26s-seg 로 바꿔, 상자 대신 마스크로 물체 위치와 불 넓이를 잼.

## [0.3.0] - 2026-10-26

### Added

- 실물 LIMO Pro 의 센서·매핑·맵 저장·Nav2 실행 파일.
- TensorRT YOLO 탐지 노드와 학습 가중치 Release.
- 교수님 Webots 맵에 불·연기·소화기·유도등·사람·계단을 놓은 시험 월드.
- Jetson 에서 Webots 를 직접 빌드해 실행하는 방법.
- 프로덕트 백로그와 하드웨어 실측 스펙 문서.

### Changed

- WeGo LIMO Pro 패키지를 실물 로봇에 설치된 판으로 바꿈.

### Removed

- 쓰지 않는 `ros2_astra_camera` 패키지와 yolo11n 설정.

### Fixed

- Jetson 의 Webots 에서 라이다·깊이 값이 모두 inf 로 나오던 문제.
- Webots 물체를 내려받다 멈추던 문제.

## [0.2.0] - 2026-10-23

### Added

- 맵 변환기 `fmap`: 공개 맵 데이터를 점유 격자로 바꾸고 소화기·비상구를 놓음.
- 탐사 강화학습용 LIMO GPU 환경.
- LIMO Pro 기본 URDF 와 RViz 표시.
- 탐지 클래스에 계단 추가, 계단·낭떠러지 낙하 방지 설계.
- YOLO 학습 이미지 수집 도구와 공개 데이터셋.
- SAM 3 자동 라벨러로 6.8 만 장 라벨링.
- 파이썬 없이 PyTorch 가중치를 읽는 변환기 `ptconv`.
- YOLO26s C++/CUDA 학습기.
- 진행 기록 문서.

### Changed

- 잘못된 비상구·문 라벨을 걸러 탐지 성능을 올림(시험 mAP50-95 0.635 → 0.654).

## [0.1.0] - 2026-10-22

### Added

- LIMO Pro 화재 대응 로봇 ROS 2 워크스페이스와 README.
- 작업 규칙 `CLAUDE.md` 와 환경 설정 스크립트.
- WeGo LIMO Pro 패키지, Isaac Sim 확장, 교수님 Webots 시뮬.
- 시뮬 맵: AWS 병원·창고·서점·집 월드, ProcTHOR-10K.

### Removed

- WeGo 패키지 내려받기 스크립트. 저장소에 직접 포함함.

[unreleased]: https://github.com/juyoung020/limo_fire_ws/compare/v0.6.0...HEAD
[0.6.0]: https://github.com/juyoung020/limo_fire_ws/compare/v0.5.0...v0.6.0
[0.5.0]: https://github.com/juyoung020/limo_fire_ws/compare/v0.4.0...v0.5.0
[0.4.0]: https://github.com/juyoung020/limo_fire_ws/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/juyoung020/limo_fire_ws/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/juyoung020/limo_fire_ws/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/juyoung020/limo_fire_ws/releases/tag/v0.1.0
