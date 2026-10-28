# Changelog

이 프로젝트의 모든 주요 변경 사항은 이 파일에 기록됩니다.

이 형식은 [Keep a Changelog](https://keepachangelog.com/ko/1.1.0/)를 기반으로 하며,
이 프로젝트는 [유의적 버전](https://semver.org/lang/ko/)을 따릅니다.

## [Unreleased]

### Added

- 변경 기록 `CHANGELOG.md`.
- 센서 토픽 문서 `docs/sensor_topics.md`: 센서마다 토픽·타입·실측 주기·frame·발행/구독 노드, 시뮬과의 차이, 포인트 클라우드를 끄면 깊이 정렬도 꺼지는 문제 기록.

## [0.5.0] - 2026-10-28

### Changed

- YOLO 전처리·출력 해석·마스크 계산을 GPU 로 옮기고 탐지·매퍼 노드를 한 프로세스 컴포넌트로 묶어, Jetson 에서 한 장 처리 시간을 10.3 ms 로 줄임.
- 실물 실행 시 Jetson 클럭을 고정하고 쓰지 않는 카메라 포인트 클라우드·IR 을 꺼서 깊이 영상을 19 Hz 에서 30 Hz 로 올림.
- Cartographer 회전 필터를 0.5° 로, 자세 발행 주기를 20 ms 로 조정.

## [0.4.0] - 2026-10-27

### Added

- 탐지한 소화기·불·문 등을 깊이와 TF 로 지도 좌표에 올리는 `object_mapper` 노드와 RViz 설정.
- 지도 위 물체를 종류별 원형 아이콘 배지, 번호 라벨, 불 위험 반경, 한글 현황판으로 표시.
- YOLO26s-seg(탐지 + 분할) 학습. 시험 세트 상자 mAP50-95 0.651, 마스크 mAP50-95 0.521.
- 분할 가중치 Release `detect-y26s-seg`.
- 인식 모델 발표용 문서 `docs/detect.md`.

### Changed

- 인식 노드를 YOLO26s-seg 로 교체. `object_mapper` 가 상자 대신 마스크 픽셀로 위치와 불 넓이를 잰다.

## [0.3.0] - 2026-10-26

### Added

- 실물 LIMO Pro 의 센서·매핑·맵 저장·Nav2 를 실행하는 XML launch (`limo_fire_bringup`, `limo_fire_navigation`).
- TensorRT YOLO C++ 탐지 노드 (`limo_fire_perception`).
- 탐지 가중치 Release `detect-y26s-clean`.
- 교수님 Webots 맵에 불·연기·소화기·유도등·사람·계단을 배치한 시험 월드와 정답 위치 `objects.yaml`.
- Jetson(ARM)에서 Webots 를 직접 빌드해 실행하는 방법.
- 프로덕트 백로그 `docs/backlog.md`, 하드웨어 실측 스펙 `docs/hardware.md`.

### Changed

- WeGo LIMO Pro 패키지를 실물 로봇에 설치된 판으로 교체 (OrbbecSDK_ROS2 등 추가).

### Removed

- 쓰지 않는 `ros2_astra_camera` 패키지와 yolo11n 설정.

### Fixed

- Jetson 의 Webots 에서 라이다·깊이 값이 모두 inf 로 나오던 문제 (aarch64 에서 `glm::perspective` double 오류).
- Webots 물체 proto 를 내려받다 멈추던 문제. 로컬 `webots://` proto 를 쓰도록 바꿈.

## [0.2.0] - 2026-10-23

### Added

- 맵 변환기 `fmap`: HouseExpo·ProcTHOR·덕트호스 미로를 점유 격자로 바꾸고 소화기·비상구·감지기를 배치.
- 탐사 강화학습용 LIMO GPU 환경 (CUDA). CPU 참조판과 결과가 비트 단위로 같음.
- LIMO Pro 기본 URDF 와 RViz 표시 launch (`limo_fire_description`).
- 탐지 클래스에 계단 추가, 계단·낭떠러지 낙하 방지 설계.
- YOLO 학습 이미지 수집 도구(크롤러, SigLIP 제로샷 필터)와 공개 데이터셋 (D-Fire, Open Images, LVIS).
- SAM 3 자동 라벨러 (C++/CUDA, fp16 6.7 장/초)로 6.8 만 장 라벨링, YOLO 데이터셋 합치기 도구 `yolods`.
- pickle 을 실행하지 않는 `.pt` → safetensors 변환기 `ptconv`.
- YOLO26s C++/CUDA 학습기. Ultralytics 와 정방향·기울기·손실 정합성 확인, 393 장/초.
- 진행 기록 `docs/progress.md`.

### Changed

- 라벨을 SigLIP 으로 다시 검사해 잘못된 exit_sign·door 라벨을 걸러냄. 시험 mAP50-95 0.635 → 0.654.

## [0.1.0] - 2026-10-22

### Added

- LIMO Pro 화재 대응 로봇 ROS 2 워크스페이스와 README.
- 작업 규칙 `CLAUDE.md`, 환경 설정 `env.sh`.
- WeGo LIMO Pro 패키지, Isaac Sim 확장, 교수님 Webots 시뮬을 `src/external` 에 포함.
- 시뮬 맵: AWS 병원·창고·서점·집 월드, ProcTHOR-10K.

### Removed

- WeGo 패키지 fetch 스크립트. 저장소에 직접 포함했으므로 필요 없음.

[unreleased]: https://github.com/juyoung020/limo_fire_ws/compare/v0.5.0...HEAD
[0.5.0]: https://github.com/juyoung020/limo_fire_ws/compare/v0.4.0...v0.5.0
[0.4.0]: https://github.com/juyoung020/limo_fire_ws/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/juyoung020/limo_fire_ws/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/juyoung020/limo_fire_ws/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/juyoung020/limo_fire_ws/releases/tag/v0.1.0
