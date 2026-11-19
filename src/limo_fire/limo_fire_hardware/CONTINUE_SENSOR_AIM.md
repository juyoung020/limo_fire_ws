# 세션 재개 메모 — 실물 카메라 탐지·거리 계산·조준

사용자 최신 요구: 금요일(2026-11-20) 실물 시험에서 **수동 표적 좌표를 쓰지 않고**, 주영 저장소의 Orbbec RGB·깊이 센서와 YOLO·물체 지도 결과로 표적 거리를 계산해 좌우·상하 조준한다. 5시간 세션 한도가 끝나도 이 작업을 계속할 수 있도록 이 파일을 유지한다. 아직 실제 표적·모터·차체 동작 성공을 주장하지 않는다.

## 코드와 환경

- 개인 저장소: `choejunhyeok2005/robot-programming`, 기능 브랜치 `feat/juyoung-launcher-v5`, 작업공간 `simulation/limo_fire_ws/src/limo_fire/limo_fire_hardware`. 다른 변경 파일이 있으므로 관련 파일만 stage한다.
- 주영 저장소: `juyoung020/limo_fire_ws`, 기능 브랜치 `codex/real-aim-friday`, 별도 로컬 clone `C:/Users/choej/.codex/worktrees/merge-feature-to-main/juyoung-limo-fire-aim`. `main`에 push하지 않는다.
- Wego 기존 작업공간은 다른 수정이 남아 있으므로 덮어쓰지 않는다. 시험 오버레이 `/home/wego/launcher_aim_friday_20261120`를 사용한다. ROS는 기존 `/home/wego/limo_fire_ws/env.sh`를 source한 다음 오버레이의 `install/local_setup.bash`를 source한다. `ROS_DOMAIN_ID=0 ROS_LOCALHOST_ONLY=1`.
- 개인 노션 페이지: https://app.notion.com/p/41c1024c87138119a55ed84cb80cb929
- 기존 팀 프로젝트 노션의 시험 페이지: https://app.notion.com/p/2026-11-20-LIMO-41c454b08de281c59feef5567e2345b3

## 주영 인식 코드에서 확인한 실제 데이터 경로

`src/limo_fire/limo_fire_perception/launch/perception.launch.xml`은 `config:=yolo_fire.yaml mapper:=true`로 실행한다. Orbbec `/camera/color/image_raw` → `/perception/object_detector` YOLO → `/perception/detections`와 `/perception/mask` → `/perception/object_map`이 `/camera/depth/image_raw`, `/camera/depth/camera_info`, TF를 합쳐 `/perception/objects` (`vision_msgs/Detection3DArray`, `map` 프레임)를 발행한다. 물체의 `results[0].pose.covariance[35]`는 마지막 실측 뒤 경과 초다. 매퍼는 저장된 물체도 10Hz로 다시 내므로 배열의 최신 헤더 시각만 신뢰하면 안 된다. 맵핑은 깊이 픽셀, 분할 마스크, 바닥 광선 추정을 조합한다. 불꽃의 투명성 때문에 바닥 광선이 쓰일 수 있으므로 실측 거리와 비교해야 한다.

조준 코드는 `/fire/observations` (`limo_fire_msgs/FireObservation`)를 읽는다. 새 `mapper_observation_bridge.py`가 주영의 `/perception/objects`에서 신뢰도 0.5 이상·관측 경과 0.5초 이하인 `fire` 하나만 변환한다. 불이 여러 개면 `target_id`가 필요하다. `real_camera_aim.launch.py`는 주영 `perception.launch.xml`이 있으면 브리지를 함께 켜고, 개인 저장소의 `yolo.launch.xml`이면 기존 매퍼의 직접 `/fire/observations`를 사용한다. 기본은 `motion_enabled=false serial_enabled=false`다. `publish_test_target.py`는 소프트웨어 시험 전용이며 실물 시험에 사용하지 않는다.

## 이미 확인한 것

- 2026-11-19 Wego의 Orbbec Dabai DC1 USB 연결, ROS 센서 모드 `robot.launch.xml base:=false`에서 `/camera/color/image_raw`와 `/camera/depth/image_raw`의 실제 헤더 수신, 깊이 CameraInfo 640×480, `/scan` 토픽 확인. 센서 점검 프로세스는 종료했다. 당시 표적이 놓이지 않아 실제 `fire` 탐지 결과는 아직 없다.
- 주영 `limo_fire_perception`을 Wego 시험 오버레이에서 빌드했고 TensorRT 탐지기와 object_map 실행 파일을 확인했다. `real_camera_aim.launch.py`를 `motion_enabled:=false serial_enabled:=false`로 센서와 함께 실행했을 때 RGB·깊이·`/perception/detections`·`/perception/mask`·`/perception/objects`·`/fire/observations` 토픽이 생성됐다. YOLO 로그는 약 30장/초, 탐지 0개였다. `map` TF를 켜지 않은 센서 전용 시험이어서 object_map은 지도 좌표를 만들지 못했다. 이를 실제 탐지 성공으로 보고하지 않는다.
- Wego ROS domain 194에서 `Detection3DArray` 시험 메시지의 신선한 `fire-1`을 브리지가 `/fire/observations`로 변환하고 위치 x=2.0m를 전달한 통합 시험을 통과했다. 입력은 합성 메시지이며 카메라 결과가 아니다. `real_camera_aim.launch.py --show-args`도 통과했다.
- 이전 Wego 임시 작업공간에서 조준 패키지 60개 테스트 통과, 오버레이에서 `limo_fire_msgs`·`limo_fire_hardware` 빌드 통과. 새로운 브리지 순수 함수 테스트 3개는 Windows에서 통과했고, Wego 오버레이의 전체 조준 테스트 63개도 통과(36.052초)했다.
- ESP32-S3 소스 Windows Arduino 코어 3.3.12 컴파일 통과; 통합 이미지는 Wego 오버레이 `firmware/`에 있고 SHA-256을 대조했다. 실제 보드 업로드는 미실시.

## 다음 작업

2026-11-19에 `AIM_LEARNING_PREP.md`와 ROS·펌웨어에 연결되지 않은 `aim_learning.py`를 추가했다. 실측 발사 CSV 검증과 한 발짜리 contextual bandit 후보 JSON 생성만 준비했다. 실제 발사 데이터·학습 성능·런타임 적용은 없다. 사용자가 적용을 요청할 때까지 기존 조준 명령은 그대로 사용한다.

1. Wego 오버레이의 전체 조준 테스트 63개는 통과했다. 결과는 `/home/wego/launcher_aim_friday_20261120/hardware_tests.log`, 주영 인식·하드웨어 빌드는 각각 `perception_build.log`, `hardware_build.log`에 있다.
2. 개인·주영 기능 브랜치의 코드·실행 문서·이 재개 메모는 원격에 올렸고, 개인 노션과 기존 팀 시험 페이지에도 **실제 탐지 필수, 수동 좌표 금지**와 검증 한계를 반영했다. 두 브랜치의 최신 커밋과 노션 페이지를 먼저 읽고 이어서 작업한다.
3. 다음 현장 작업: 실측 네 값, ESP32 연결·업로드, 실제 표적 배치와 Nav2 위치 추정이 충족되면 미리보기 → 무장전 실물 조준 순서로 검증한다. `map` TF와 실제 표적이 없으면 `/fire/observations`의 3D 좌표를 검증할 수 없다. 실제 발사는 별도 단계다.

재개할 때 `git status`와 위 Wego 빌드 로그부터 확인하고, 이미 완료된 작업을 반복하지 않는다. 인증 정보는 이 파일에 저장하지 않는다.
