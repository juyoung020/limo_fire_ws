<div align="center">

<h1>🔥 LIMO Pro 화재 대응 로봇</h1>

<b>화재를 먼저 찾아내고, 먼저 대응한다.</b><br>
밤마다 순찰하며 소방 시설 지도를 만들어 두고, 불이 나면 신고·진압·인원 파악·대피 안내까지 하는 로봇

<br><br>

<a href="https://plausible-hallway-e4f.notion.site/3eb454b08de28190b2e8e321a33a9371">팀 노션</a> &nbsp;·&nbsp;
<a href="CHANGELOG.md">변경 기록</a>

<br><br>

<img src="https://img.shields.io/badge/ROS_2-Humble-22314E?logo=ros" alt="ROS 2 Humble">
<img src="https://img.shields.io/badge/Ubuntu-22.04-E95420?logo=ubuntu&logoColor=white" alt="Ubuntu 22.04">
<img src="https://img.shields.io/badge/Robot-AgileX_LIMO_Pro-555" alt="AgileX LIMO Pro">
<img src="https://img.shields.io/badge/Sim-Isaac_Sim_4.5-76B900?logo=nvidia&logoColor=white" alt="Isaac Sim 4.5">

<br><br>

<img src="docs/assets/sim_nav_demo.gif" width="760" alt="Webots 덕트호스 미로에서 2D Pose Estimate → 2D Goal Pose 주행하며 YOLO 탐지를 지도에 올리는 화면">

<sub>Webots 덕트호스 미로(시연장 추정 모양) — RViz 에서 처음 위치를 찍고 목표를 주면 Nav2 가 미로를 돌아가며, YOLO 가 찾은 불·소화기·문·사람·계단을 지도에 올린다 (10 배속)</sub>

<br><br>

<img src="docs/assets/scenario.png" width="760" alt="서비스 흐름">

<sub>서비스 흐름 — 평상시에 만든 지도가 화재 시 주행·신고·안내의 바탕이 된다</sub>

</div>

<br>

한국항공대학교 로봇프로그래밍(0533) 조별 프로젝트. 기말 시연은 2026-12-17, 덕트호스로 만든 미로에서 Live Demo로 한다.

## 무엇을 하나

| 상황 | 로봇이 하는 일 |
|---|---|
| 처음 설치 | 자동 탐사(판단 모델이 갈 곳을 고르고 A* 로 주행)로 건물을 빠르게 훑어 지도를 만들고, 소화기·문·비상구 위치를 등록 |
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

<sub>임무 행동 트리 — 판단은 BT가, 탐사 전략만 따로 된 노드(판단 모델 + A*)가 맡고, 실패하면 고전 방식으로 넘어간다</sub>
</div>

<br>

| 모듈 | 방법 |
|---|---|
| 임무·판단 | Nav2 BehaviorTree.CPP 플러그인 |
| 첫 지도 탐사 | 두 층: 위층 판단 모델(d1-3B, 로봇 안)이 프론티어 후보(최대 6 곳) 가운데 갈 곳을 고르고, 아래층이 로봇 지도 위 A* 경로를 따라감 (2026-11-16 실물 시험으로 정함). 실패하면 프론티어 탐사로 Fallback |
| 불 탐지·크기 | YOLO + 거리(depth → 라이다 → 바닥 평면 가정) |
| 조준 | 방식 미정 (강화학습 또는 화면 오차 PID + 거리별 보정표) |
| 연기 모드 | 오도메트리·IMU + 초음파 거리로 AMCL 보정, 초음파로 장애물 재계획 |
| 인원 집계 | 레이더 표적을 지도 좌표에서 칼만필터로 추적, 정지한 사람 유지 |
| 보고·안내 | rosbridge 웹 앱, 오프라인 음성 안내 |

자세한 설계와 실험 계획은 [스펙·구현 방향](docs/spec.md)에 있다.
노드·스레드마다 무엇을 받고 내는지(토픽·QoS·주기·상태 값)는 [실행 구조](docs/runtime_io.md), 센서 토픽은 [센서 토픽](docs/sensor_topics.md).

강화학습(위층 후보 고르기 + 아래층 속도 직접, PPO)으로 탐사하려던 계획은 실물에서 시뮬과의 격차가 크고 A* 아래층이 훨씬 빠르고 잘 돼서
2026-11-16 에 그만뒀다. 그때의 코드·결과·영상은 [archive/rl](archive/rl/README.md).

### 프론티어 탐사 — 판단 모델 + A*

<img src="train/explore/scripts/img/cand_stage7.png" width="760" alt="실물 녹화 지도 위에 고른 후보 6 곳과 각 후보에서 새로 보일 범위를 색 선으로 그린 그림">

<sub>실물 녹화(2026-11-16 실험실) 한 순간의 후보 — 번호는 점수 순서, 색 선은 그 자리에서 새로 보일 범위</sub>

프론티어(아는 곳과 모르는 곳의 경계)로 가며 지도를 넓힌다. 목표는 **빈 곳 없이 다 칠하기가 아니라, 듬성듬성해도 공간을 빨리 대충 아는 것**이다.
그래서 0.5 m 칸 기준 덮음이 시간에 따라 얼마나 빨리 오르는지(덮음 AUC)로 잰다.

- **후보 뽑기:** 지도에서 프론티어 칸을 찾아 덩어리로 묶고, 덩어리마다 서 볼 자리를 뽑아 갈 수 없는 곳·벽 가까운 곳을 거른 뒤, "새로 알게 될 넓이 ÷ 가는 시간" 으로 점수를 매겨 최대 8 곳을 고른다(판단 모델에는 앞 6 곳). 일곱 단계를 그림과 함께 [scripts/README](train/explore/scripts/README.md) 의 "후보 뽑기" 절에 적었다.
- **위층 — 어디로 갈까:** 판단 모델(LiquidAI d1-3B, 로봇 안 llama-server)이 지도 그림·후보 표·지난 선택 기록을 보고 후보 하나를 고른다. 목표에 닿았거나 실패했을 때만 묻는다. 판단 한 번에 로봇(Orin, Q8)에서 약 1.7~1.9 초.
- **아래층 — 어떻게 갈까:** 로봇 지도 위 A* 경로를 따라간다. 깊이 카메라 장애물을 0.1 초마다 지도에 더해 라이다보다 낮은 가구도 피한다. 로봇 노드 한 걸음은 Orin 에서 약 5.9 ms(한 스레드).

미세조정 전 판단 모델은 아직 단순 규칙(점수 1 등 후보로 가기)보다 느리다. 시뮬 8 장면 × 4 판, 180 초까지 덮음 AUC:

| 위층 | 덮음 AUC (0~180 s) | 90 % 덮는 시간 |
|---|---|---|
| 규칙 (점수 1 등 후보) | 0.677 | 134 s |
| 판단 모델, 후보 최대 6 (기본) | 0.626 | 184 s |
| FUEL (프론티어 탐사 논문 방식) | 0.559 | 228 s |

판단 모델은 시뮬에서 후보마다 실제로 굴려 본 결과로 미세조정(LoRA)하는 중이다 — [upper_lora](train/explore/upper_lora/README.md).
실행 방법은 [limo_fire_explore](src/limo_fire/limo_fire_explore/README.md), 시뮬·확인 도구는 [train/explore](train/explore/README.md).

<br>

## 하드웨어

AgileX LIMO Pro(라이다, depth 카메라, IMU) 위에 ESP32로 제어하는 모듈을 올린다. 추가 부품비 95,127원.

| 부품 | 역할 |
|---|---|
| 열화상 MLX90640 (32×24) | 작은불 열원 확인 |
| 24GHz 레이더 HLK-LD2450 | 연기 속 사람 감지 |
| 초음파 HC-SR04 ×2 (앞 좌우 ±20°) | 연기 속 장애물, 위치 보정 |
| 서보 DS3218 + MG996R | 노즐 팬·틸트 조준 |
| 12V 다이어프램 펌프 + MOSFET | 물 분사 |

전체 목록과 전원 확인 사항은 [부품 목록](docs/parts.md)에 있다.

### 탁구공 발사대

LIMO 지붕 나사 구멍에 고정하는 3D 출력(PLA) 발사대. 서보 두 개로 장전·발사와 각도를 자동으로 바꾼다.

<div align="center">
<img src="docs/assets/launcher_sim.gif" width="760" alt="MuJoCo 시뮬: 장전, 걸쇠 잡음, 발사, 각도 조절">

<sub>MuJoCo 시뮬 — 30° 한 발, 15° 에서 장전한 뒤 5° 로 내려 두 발. 초록 = 플런저, 주황 = 레버, 빨강 = 걸쇠, 노랑 선 = 고무줄, 파랑 = 각도 서보 (발사 순간은 느린 화면)</sub>
</div>

| 동작 | 하는 일 |
|---|---|
| 장전 | DS3218 레버가 66° 돌아 플런저를 뒤로 밀고, 걸쇠가 핀을 잡는다. 공 통(5 발)의 맨 아래 공이 플런저 앞으로 내려온다 |
| 각도 | MG996R 이 발사대 축을 돌려 5~30° 로 맞춘다. 장전은 15° 이상에서 하고 천천히 내린다 |
| 발사 | 레버를 반대로 18° 돌리면 걸쇠가 열리고, 고무줄이 플런저를 당겨 공을 친다 |

시뮬 값 중 고무줄 세기는 잠정(실물 측정 전)이다. 출력 파일·설계 문서·시뮬은 [`hardware/modeling/탁구공발사대/`](hardware/modeling/탁구공발사대/) 에 있다.

#### v5 — 물리 최적화판 (시뮬 검증, 출력 전)

<div align="center">
<img src="docs/assets/launcher_v5_sim.gif" width="760" alt="v5 MuJoCo 시뮬: 레버가 핀을 밀어 장전, 걸쇠가 잡고, 레버가 걸쇠를 눌러 발사">

<sub>v5 MuJoCo 시뮬 — 레버 쪽 확대 30° 두 발, 전체 15° 한 발. 주황 = 레버, 빨강 = 걸쇠, 노랑 선 = 고무줄</sub>
</div>

| | 지금(출력본) | v5 |
|---|---|---|
| 공 속도(탁구공, 30°, 같은 고무줄) | 4.7 m/s | 6.7 m/s |
| 서보 힘 60 % 로 장전 | 불확실 | 성공 |
| 바뀐 것 | | 레버 서보 축을 장전 끝 쪽으로, 움직이는 무게 18 → 9 g, 핀 롤러, 고무줄 고리 3곳 |

펌웨어 값: 장전 +67.3°, 장전 뒤 1.0 s 대기, 발사 -24°. 설계·검증 문서는 [`hardware/modeling/탁구공발사대/v5/설계_v5.md`](hardware/modeling/탁구공발사대/v5/설계_v5.md).

ESP32-S3 GPIO4 레버·GPIO5 상하 모터를 쓰는 실물 시험용 ROS 경로는 [탐지·접근·조준 패키지](src/limo_fire/limo_fire_hardware/README.md)와 [조준 뒤 장전·발사 및 별도 터미널 실행법](src/limo_fire/limo_fire_hardware/REAL_FIRE_TEST.md)에 있다. 자동 발사는 기본으로 꺼져 있다. 가상 Nav2·시리얼 시험은 통과했지만 이 연속 경로에서 실물 공 발사·명중은 아직 검증하지 않았다. 기존 물 노즐 행동 트리와는 별도 시험 경로다.

## 폴더

```
limo_fire_ws/            ROS 2 워크스페이스 (여기서 colcon build)
├── src/
│   ├── external/        남의 코드 (WeGo LIMO Pro, 교수님 Webots 시뮬) — 고치지 않음
│   └── limo_fire/       우리 패키지
├── sim/                 Isaac Sim 4.5 확장·USD, maps/ 실내 맵      [COLCON_IGNORE]
├── train/               GPU 시뮬·학습 (맵 변환기, 탐사 시뮬, YOLO)  [COLCON_IGNORE]
├── firmware/            ESP32 코드
├── docs/                아이디어·스펙·부품 문서
├── refs/                참고 자료 (papers/ code/ datasheets/)       [COLCON_IGNORE]
├── archive/             안 쓰는 것                                  [COLCON_IGNORE]
└── env.sh               워크스페이스 환경 (conda 빼고 ROS 2 + install)
```

패키지별 역할까지 담은 전체 구조와 URDF·TF 규약은 [규약](docs/conventions.md), 실행 중 프로세스·스레드·토픽 입출력은 [실행 구조](docs/runtime_io.md)에 있다.

## 시뮬레이션

시뮬레이터를 목적별로 나눠 쓴다.

| 목적 | 시뮬레이터 | 위치 |
|---|---|---|
| 탐사 평가 (위층·아래층, 실물 노드와 같은 코드) | 직접 만드는 2.5D GPU 환경 — 수천 개 병렬 | `train/explore` |
| 실제와 비슷한 검증 (센서·물리) | Isaac Sim 4.5 + WeGo LIMO USD | `sim/wego_LimoIsaacSIM` |
| (공통) 맵 | AWS 병원·창고·서점·집, ProcTHOR-10K, HouseExpo | `sim/maps` ([목록](sim/maps/README.md)) |
| Nav2·SLAM·BT 통합 검증 | Webots (교수님 LIMO 모델, 실제 LIMO 와 같은 토픽·프레임) + 우리 맵 4 개 | `sim/webots` ([맵·실행](sim/webots/README.md)) |
| 공 발사·표적 전도 물리 데모 | MuJoCo 3.17 (독립 실행) | `sim/mujoco_ball_mvp` — ROS 2/Nav2 주행과는 아직 연결되지 않음 |

```bash
source ~/limo_fire_ws/env.sh
ros2 launch limo_fire_bringup sim.launch.xml world:=limo_fire_maze     # Webots + 저장 지도 + Nav2 (slam:=true · perception:=true)
```

시뮬 구조·Nav2 설정·운용 화면은 [simulation.md](docs/simulation.md), 인식 성능은 [benchmarks](docs/benchmarks/README.md).

## 시작하기

```bash
git clone https://github.com/juyoung020/limo_fire_ws.git ~/limo_fire_ws
cd ~/limo_fire_ws
source env.sh                      # conda 빼고 ROS 2 Humble
colcon build --packages-select webots_ros2_limo   # 지금 쓰는 것만. 전체 빌드는 astra_camera 등 의존성 필요
source env.sh
```

### 실물 LIMO 에서

```bash
source ~/limo_fire_ws/env.sh
colcon build --packages-select limo_fire_description limo_fire_navigation limo_fire_bringup
source env.sh
ros2 launch limo_fire_bringup robot.launch.xml          # 센서·구동
ros2 launch limo_fire_bringup mapping.launch.xml        # 지도 만들기 → save_map.launch.xml 로 저장
ros2 launch limo_fire_bringup navigation.launch.xml     # 저장한 지도로 주행
```

WeGo 대시보드 버튼과 같은 구성이다. 자세한 것은 [limo_fire_bringup](src/limo_fire/limo_fire_bringup/README.md).

Isaac Sim USD(`WegoLimoUSD.zip`, 약 65MB)는 용량 때문에 저장소에 없다. [WeGo 가이드](https://goofy-pleasure-a84.notion.site/Limo-Isaac-Sim-1ff09b2ac4bf8086bdf5e94521f74946)에서 받아 푼다.

```bash
unzip WegoLimoUSD.zip -d sim/wego_LimoIsaacSIM/USD/
```

### 포함한 외부 코드

| 폴더 | 출처 | 커밋 |
|---|---|---|
| `src/external/wego_limo_pro` | 실물 LIMO Pro 의 `~/wego_ws/src` (WeGo 출고본, [WeGo-Robotics/limo_pro](https://github.com/WeGo-Robotics/limo_pro) 와 다름) — [목록](src/external/wego_limo_pro/README.md) | 2026-10-26 복사 |
| `src/external/webots_ros2_limo` | [inmo-jang/webots_ros2_limo](https://github.com/inmo-jang/webots_ros2_limo) (교수님) | `68d4fc0` |
| `sim/wego_LimoIsaacSIM` | [WeGo-Robotics/LimoIsaacSIM](https://github.com/WeGo-Robotics/LimoIsaacSIM) (`sim` 브랜치) | `158da2b` |

## 일정

<div align="center">
<img src="docs/assets/timeline.png" width="680" alt="프로젝트 일정">
</div>

| 날짜 | 내용 |
|---|---|
| 11/2 ~ | 주간 멘토링 |
| 11/30 (월) | 중간평가 — PPT + 동영상 시연 |
| 12/17 (목) 13시 | 기말 발표 + Live Demo |
