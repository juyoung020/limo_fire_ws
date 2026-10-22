<div align="center">

<h1>🔥 LIMO Pro 화재 대응 로봇</h1>

<b>화재를 먼저 찾아내고, 먼저 대응한다.</b><br>
밤마다 순찰하며 소방 시설 지도를 만들어 두고, 불이 나면 신고·진압·인원 파악·대피 안내까지 하는 로봇

<br><br>

<a href="https://plausible-hallway-e4f.notion.site/3eb454b08de28190b2e8e321a33a9371">팀 노션</a> &nbsp;·&nbsp;
<a href="docs/idea.md">아이디어</a> &nbsp;·&nbsp;
<a href="docs/spec.md">스펙·구현 방향</a> &nbsp;·&nbsp;
<a href="docs/parts.md">부품 목록</a>

<br><br>

<img src="https://img.shields.io/badge/ROS_2-Humble-22314E?logo=ros" alt="ROS 2 Humble">
<img src="https://img.shields.io/badge/Ubuntu-22.04-E95420?logo=ubuntu&logoColor=white" alt="Ubuntu 22.04">
<img src="https://img.shields.io/badge/Robot-AgileX_LIMO_Pro-555" alt="AgileX LIMO Pro">
<img src="https://img.shields.io/badge/Sim-Isaac_Sim_4.5-76B900?logo=nvidia&logoColor=white" alt="Isaac Sim 4.5">

<br><br>

<img src="docs/assets/scenario.png" width="760" alt="서비스 흐름">

<sub>서비스 흐름 — 평상시에 만든 지도가 화재 시 주행·신고·안내의 바탕이 된다</sub>

</div>

<br>

한국항공대학교 로봇프로그래밍(0533) 조별 프로젝트. 기말 시연은 2026-12-17, 덕트호스로 만든 미로에서 Live Demo로 한다.

## 무엇을 하나

| 상황 | 로봇이 하는 일 |
|---|---|
| 처음 설치 | 강화학습 탐사로 건물을 빈 곳 없이 돌며 지도를 만들고, 소화기·문·비상구 위치를 등록 |
| 매일 밤 | 지도 위를 순찰·점검하며 소방 시설 지도를 갱신 |
| 순찰 중 불 발견, 또는 화재 감지기 신호 | 그 위치로 출동해 YOLO + 거리로 불의 실제 크기를 추정 |
| 작은불 | 소방당국에 신고하고, 열화상으로 확인한 뒤 노즐을 조준해 물로 끔 |
| 큰불 | 접근하지 않고 위치와 크기를 소방당국에 신고 |
| 연기가 짙을 때 | 저장된 지도 + 초음파로 주행하고, 레이더로 구역별 인원을 집계 |
| 건물 안 사람 발견 | 가장 가까운 비상구·소화기를 음성으로 안내 |

소방당국은 웹 앱에서 지도, 불 위치, 구역별 인원(추정치), 막힌 통로를 본다.

## 시스템 구조

<div align="center">
<img src="docs/assets/behavior_tree.png" width="680" alt="임무 행동 트리">

<sub>임무 행동 트리 — 판단은 BT가, 탐사 전략만 학습된 노드가 맡고, 학습이 실패하면 고전 방식으로 넘어간다</sub>
</div>

<br>

| 모듈 | 방법 |
|---|---|
| 임무·판단 | Nav2 BehaviorTree.CPP 플러그인 |
| 첫 지도 탐사 | 강화학습(PPO, GPU 학습)이 중간 목표점을 내고 Nav2가 주행. 실패하면 프론티어 탐사로 Fallback |
| 불 탐지·크기 | YOLO + 거리(depth → 라이다 → 바닥 평면 가정) |
| 조준 | 강화학습 첫 시도, 비교 대상은 PID + 거리별 보정 |
| 연기 모드 | 오도메트리·IMU + 초음파 거리로 AMCL 보정, 초음파로 장애물 재계획 |
| 인원 집계 | 레이더 표적을 지도 좌표에서 칼만필터로 추적, 정지한 사람 유지 |
| 보고·안내 | rosbridge 웹 앱, 오프라인 음성 안내 |

자세한 설계와 실험 계획은 [스펙·구현 방향](docs/spec.md)에 있다.

## 하드웨어

AgileX LIMO Pro(라이다, depth 카메라, IMU) 위에 ESP32로 제어하는 모듈을 올린다. 추가 부품비 95,127원.

| 부품 | 역할 |
|---|---|
| 열화상 MLX90640 (32×24) | 작은불 열원 확인 |
| 24GHz 레이더 HLK-LD2450 | 연기 속 사람 감지 |
| 초음파 HC-SR04 ×3 | 연기 속 장애물, 위치 보정 |
| 서보 DS3218 + MG996R | 노즐 팬·틸트 조준 |
| 12V 다이어프램 펌프 + MOSFET | 물 분사 |

전체 목록과 전원 확인 사항은 [부품 목록](docs/parts.md)에 있다.

## 폴더

```
limo_fire_ws/            ROS 2 워크스페이스 (여기서 colcon build)
├── src/                 ROS 2 패키지 (WeGo LIMO Pro 패키지는 받아서 넣음)
├── sim/                 Isaac Sim 4.5 확장·USD                     [COLCON_IGNORE]
├── firmware/            ESP32 코드
├── tools/               fetch_third_party.sh
├── docs/                아이디어·스펙·부품 문서
├── refs/                참고 자료 (papers/ code/ datasheets/)       [COLCON_IGNORE]
└── archive/             안 쓰는 것                                  [COLCON_IGNORE]
```

## 시작하기

```bash
git clone https://github.com/juyoung020/limo_fire_ws.git ~/limo_fire_ws
cd ~/limo_fire_ws
./tools/fetch_third_party.sh       # WeGo LIMO Pro 패키지 + Isaac Sim 확장 클론, USD 받는 법 안내
source /opt/ros/humble/setup.bash
colcon build                       # conda는 PATH에서 뺀 셸에서
source install/setup.bash
```

외부 저장소(`src/wego_limo_pro`, `sim/wego_LimoIsaacSIM`), USD 파일, 빌드 결과물은 저장소에 올리지 않는다.
출처: [WeGo-Robotics/limo_pro](https://github.com/WeGo-Robotics/limo_pro), [WeGo-Robotics/LimoIsaacSIM](https://github.com/WeGo-Robotics/LimoIsaacSIM).

## 일정

<div align="center">
<img src="docs/assets/timeline.png" width="680" alt="프로젝트 일정">
</div>

| 날짜 | 내용 |
|---|---|
| 11/2 ~ | 주간 멘토링 |
| 11/30 (월) | 중간평가 — PPT + 동영상 시연 |
| 12/17 (목) 13시 | 기말 발표 + Live Demo |
