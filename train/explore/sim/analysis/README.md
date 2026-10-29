# 교수님 Webots 시뮬 분석 — 학습 시뮬레이터의 명세

**정답은 실물 LIMO 측정값이다** ([real_limo.md](real_limo.md)). 교수님 시뮬은 그 값에 대어 검증하는 대상이고, 학습 시뮬은 형상·투영 계산을 교수님 시뮬로 확인하고(V2) 오차 모델은 실물에 맞춘다.

학습 시뮬레이터(`train/explore/sim`)가 맞춰야 할 기준은 교수님 Webots LIMO(`src/external/webots_ros2_limo`)가 내는 센서 값이다.
여기에 그 값이 어떻게 만들어지는지(정적 분석: 소스)와, 실제로 돌려서 잰 값(동적 분석)을 모은다. 시뮬레이터는 이 문서대로 구현하고, 바뀌면 이 문서부터 고친다.

## 분석 방법

| 대상 | 방법 | 까닭 |
|---|---|---|
| Webots R2025a | 소스 읽기 (이 젯슨에서 소스로 빌드한 `~/webots_build/webots`) | 오픈소스라 디컴파일보다 원본이 정확하다 |
| 교수님 센서 플러그인 `limo_sensors.py` | 소스 읽기 | Python 소스 그대로 |
| 그리기(라이다·깊이는 OpenGL 로 그린다) | 같은 자세에서 Webots 출력과 비교(V2). 필요하면 apitrace 로 프레임 캡처, gdb 로 `WbLidar` 내부 값 | 깊이 버퍼 정밀도·노이즈 난수는 GPU 드라이버가 정해 소스만으로 비트를 못 정한다 |
| 실물 Orbbec SDK (`libOrbbecSDK.so`, 비공개) | 실물 녹화로 오차 분포를 맞춘다. 역공학(Ghidra)은 라이선스 확인 뒤에만 | 실물 깊이 필터·구멍 메우기가 여기 있다 |

## 정적 분석: 센서 값이 만들어지는 길

### 라이다 (YDLIDAR T-mini Plus 흉내)

| 단계 | 내용 | 소스 |
|---|---|---|
| 장치 | `Lidar` fov 3.839724 (220°), horizontalResolution 248, 1 층, near·minRange 0.1, maxRange 12, **noise 0.003, resolution 0.015**, 장착 base_link (0.103, 0, −0.034) | `protos/LimoFourDiff.proto:40-70` |
| 빔 방향 | 비회전 라이다: i 번째 = fov/2 − (i + ½)·fov/N (왼쪽 → 오른쪽). 센서 좌표 x 앞 · y 왼쪽 | `WbLidar.cpp:427-431` (`dtheta = −fov/w`, `theta0 = fov/2 + dtheta/2`) |
| 거리 | 광선 방향 유클리드 거리. fov > 180° 라 여러 방향으로 그려 이어 붙인다 | `WbLidar.cpp:348-400` |
| 노이즈 | `minRange < r < maxRange` 이면 r + noise·N(0,1). N 은 셰이더 해시 난수(`fract(sin(dot(uv, (12.9898, 78.233))) · 43758.5453)`) + Box-Muller | `resources/wren/shaders/range_noise.frag` |
| 양자화 | `floor(r / resolution + 0.5) · resolution` (1.5 cm) | `resources/wren/shaders/depth_resolution.frag` |
| ROS | 순서를 뒤집어 각도가 커지는 순(−110° → +110°), `[min, max)` 밖·inf 는 0, intensity 47, angle_increment 는 실물 값 2π/405 | `limo_sensors.py:239-248, 39-46` |

- 빔 각도: Webots 실제 간격은 fov/248 = 0.015483 rad 인데 `/scan` 은 실물 값 0.015514 를 적는다. 끝 빔에서 0.44° 차이 — 학습 시뮬은 Webots 각도를 쓴다(값이 같아야 하므로).

### 깊이 카메라 (Orbbec Dabai DC1 흉내)

| 단계 | 내용 | 소스 |
|---|---|---|
| 장치 | `RangeFinder` 644 × 524, 가로 fov 1.161085, near·minRange 0.01, maxRange 3.0, 노이즈·양자화 없음 | `LimoFourDiff.proto:119-127` |
| 투영 | 핀홀, 주점 = 가운데, 초점 = (W/2) / tan(fov/2). 값 = 광축 방향 z | Webots RangeFinder |
| 자르기 | (위 44, 왼쪽 4) 잘라 640 × 480 → cx 317.5, cy 217.5 가 실물과 같게 | `limo_sensors.py:65-70, 362-363` |
| 유효 | 0.3 ≤ z ≤ 3.0 이고 실물이 덮는 영역(행 22~399, 열 0~627) 만, 나머지 0. mm 반올림 16UC1 | `limo_sensors.py:71-76, 364-367` |

### 형상 (장면)

- 원기둥은 n 각형(기본 36): 점 = r·(sin 2πi/n, cos 2πi/n) — `src/wren/StaticMesh.cpp:436-442`. 캡슐 기본 12, 구 이코 1. 학습 시뮬은 원기둥을 같은 n 각형으로, 캡슐·구는 잘게 쪼갠다(차이 ≤ r·(1 − cos π/n), 사람 팔다리 0.5 mm 안팎).
- 장면은 Ros2Supervisor `animation_start_recording` 으로 내보낸 `.w3d` 를 그대로 바꾼다(`tools/w3d2scene`). 손으로 옮기지 않는다.
- 라이다는 우리 로봇 몸을 맞히지 않는다(양 끝 빔도 4 m 대, 동적 분석) → 장면에서 `LIMO` 를 뺀다.

## 동적 분석 결과

### 우리 launch == 교수님 launch (2026-10-29)

교수님 `robot_launch.py` + `limo_world.wbt` 와 우리 `sim_robot.launch.xml` + 같은 맵(감독 로봇만 붙임)에서, 시작 자리에 세운 채 라이다 30 장·깊이·컬러를 받아 비교:

| 센서 | 결과 |
|---|---|
| 깊이 640 × 480 | 유효 픽셀 같음, 거리 차 0 mm (99 % 도 0) |
| 컬러 | 평균 차 0.00 / 255 |
| 라이다 248 빔 | 유효 빔 같음, 30 장 평균 차 중앙 0.7 mm, 장마다 흔들림 6.19 vs 6.13 mm (노이즈 + 1.5 cm 양자화로 설명됨) |

→ 학습 시뮬은 우리 경로(같은 값)로 검증해도 교수님 시뮬과 같다.

### V0 · V1: 학습 시뮬 GPU == CPU

`tools/sim_verify` (미로 장면 삼각형 38,404, 자세 64): 라이다 15,872 빔 · 깊이 1,966 만 픽셀 **비트 다름 0** (Orin sm_87). 자세 하나를 1 mm 옮기면 잡아냄. 데스크톱 sm_120 은 아직.

### V2: 학습 시뮬 == Webots (같은 자세, 노이즈 끄고) — 다음

- Webots 로봇을 supervisor 로 두고 플러그인이 라이다·깊이 장치의 참값 자세(세계 좌표)를 내고, 정해 준 자세로 옮기고, 노이즈·양자화를 끈다(supervisor 로 필드 값만 바꿈, 모델 파일은 그대로).
- 같은 자세에서 학습 시뮬 값과 빔·픽셀마다 비교. 통과 기준: 라이다 빔 95 % 가 2 cm 안, 깊이 유효 영역 IoU ≥ 0.95.
- 노이즈를 켠 값은 분포(흔들림 σ, 양자화 계단)로 비교한다 — 노이즈 난수는 GPU 드라이버의 셰이더 `sin` 정밀도에 따라 비트가 달라서 비트 일치는 목표가 아니다.

## 아직 분석할 것

- [ ] IMU·오도메트리 (`limo_sensors.py` `__publish_imu` · `__publish_odom`: 바퀴 + IMU 각으로 적분)
- [ ] 바퀴 구동: ros2_control diff_drive(바퀴 간격 0.172, 반지름 0.045, 25 Hz) + Webots 마찰·미끄러짐
- [ ] 라이다를 여러 방향으로 그릴 때 이음매 근처 빔이 다른지 (V2 결과로)
