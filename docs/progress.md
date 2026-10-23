# 진행 기록

지금까지 한 일, 왜 그렇게 정했는지, 다음에 할 일. 마지막 갱신 2026-10-23(SAM 3 포팅 후).

아이디어는 [idea.md](idea.md), 설계는 [spec.md](spec.md), 부품은 [parts.md](parts.md).

## 한눈에

| 부분 | 상태 | 위치 |
|---|---|---|
| 워크스페이스·규칙 | 됨 | 루트, [CLAUDE.md](../CLAUDE.md) |
| 시뮬레이터 역할 나누기 | 됨 | [README](../README.md#시뮬레이션) |
| 맵 52,032 개 + 소방 시설 배치 | 됨 | `train/maps` |
| 탐사 RL GPU 환경 (CUDA) | 됨 — 관측·PPO 남음 | `train/explore/env` |
| LIMO Pro 기본 URDF·TF | 됨 — 센서 모듈은 배치 정해지면 | `src/limo_fire/limo_fire_description` |
| YOLO 데이터 모으기 | 공개 데이터·크롤링 됨, 직접 찍기 남음 | `train/detect/data` |
| 자동 라벨러 (SAM 3, C++/CUDA) | 됨 — 6.8 만 장 라벨, YOLO 데이터셋(8:1:1) 만듦 | `train/detect/teacher`, `tools/yolods` |
| YOLO26s 학습기 (C++/CUDA) | 됨 — 100 epoch 학습 중(약 5.5 시간) | `train/detect/trainer` |
| ROS 노드 (BT·인식·연기 모드·보고) | 할 일 | `src/limo_fire` |
| ESP32 모듈 | 부품 오면 | `firmware/` |

## 1. 워크스페이스

- `~/limo_fire_ws` 가 colcon 루트. 남의 코드는 `src/external`, 우리 패키지는 `src/limo_fire/limo_fire_<역할>`.
- `sim/ train/ refs/ archive/` 는 `COLCON_IGNORE` — 빌드에서 빠진다.
- 저장소 하나에 다 넣는다(서브모듈 안 씀). 외부 코드는 받은 커밋을 README 표에 적어 둔다.
- 규칙: 새 코드는 Rust·C++ 만, launch 는 XML/YAML, 작업 전 `git pull`, 작업 단위마다 커밋·푸시.
- `env.sh`: conda 의 Python 3.14 가 `rclpy` 를 깨뜨려서(Webots 감독 노드가 죽고 `/clock` 이 안 나옴) PATH 에서 conda 를 빼고 ROS 2 Humble 을 연다.
- GitHub: 팀 저장소를 `juyoung020/limo_fire_ws` 로 바꾸고 기록을 비웠다. 예전 robot-agent 기록 561 커밋은 `archive/robot-agent-history-2026-10-22.bundle` 에 백업.

## 2. 시뮬레이터

교수님 Webots 환경 하나로는 강화학습에 너무 느려서 목적별로 셋을 쓴다.

| 목적 | 시뮬레이터 | 왜 |
|---|---|---|
| 강화학습 (탐사·조준) | 직접 만든 2.5D GPU 환경 | 수천 개를 병렬로 돌려야 학습이 된다 |
| 센서·물리 검증 | Isaac Sim 4.5 + WeGo LIMO USD | WeGo 가 LIMO Pro 모델을 공개 |
| Nav2·SLAM·BT 통합 | Webots (교수님 `webots_ros2_limo`) | 실제 LIMO 와 토픽·프레임이 같다 |

- 교수님 환경과 WeGo LIMO Pro 둘 다 SLAM 은 **Cartographer**, 위치 추정은 **Nav2 AMCL**. 우리도 같게 간다.

## 3. 맵 (`train/maps`)

Rust 변환기 `fmap` 이 공개 실내 평면도를 5 cm 점유 격자 `.fmap` 으로 바꾸고 소방 시설을 놓는다.

| 소스 | 개수 | 라이선스 |
|---|---|---|
| HouseExpo (주택 평면도) | 35,088 | 로컬만 (SUNCG 파생) |
| ProcTHOR-10K | 11,971 | Apache-2.0 |
| 덕트호스 미로 생성기 (시연장 모양) | 4,973 | 우리 것 |
| **합계** | **52,032** | 32 코어 약 22 초, 12 GB |

- 소화기: 모든 빈 칸에서 보행거리 20 m 안 + 33 m² 이상 방마다 하나 (화재안전기준).
- 비상구: 바깥 벽에만. 처음엔 HouseExpo 안쪽 벽 홈에 비상구가 놓이는 문제가 있어서, 바깥으로 0.6 m 이상 벽이 이어지는지 보는 광선 검사로 바꿨다.
- 화재 감지기: 방 중심마다 하나.
- 불: 맵에 넣지 않고 GPU 환경이 판마다 무작위로 만든다.

AWS 병원·창고·서점·집 월드(`sim/maps`)는 메시라 아직 격자로 안 바꿨다 — 시험용으로 나중에.

## 4. 탐사 RL GPU 환경 (`train/explore/env`)

- 환경 하나 = GPU 스레드 하나. 같은 헤더를 CPU 참조판과 CUDA 커널이 같이 쓰고, 부동소수 옵션을 고정해 **CPU 와 GPU 결과가 비트까지 같다**. 일부러 어긋나게 한 음성 대조도 잡힌다.
- 행동은 바퀴 속도가 아니라 **중간 목표점**(로봇 기준 16 방향 × 0.5·1·2 m). 실물에서는 Nav2 가 그 점으로 간다.
- 보상은 벽 표면을 빈틈없이·빨리·다시 안 가고 덮기. 장애물 뒤는 안 보이므로 셈에서 뺀다. 불을 처음 보면 +1.
- 처리량: 16,384 환경에서 초당 35 만 행동(실시간의 약 17 만 배).
- 남은 것: 관측 만들기(로봇 중심 지도 채널 + depth + YOLO 의미 채널), PPO(object-memory-robot 에서 옮김), 학습 뷰어.

## 5. URDF·TF (`limo_fire_description`)

- WeGo LIMO Pro xacro 를 그대로 쓰고, 카메라 위치는 교수님 Webots 모델 값으로 더했다.
- `map → odom → base_link`, `base_footprint` 없음(실물 드라이버에 없음). 카메라는 `camera_link` + `*_optical_frame`.
- `check_urdf` 13 프레임, `tf2_echo` 값이 Webots 와 같음, RViz 화면 확인.
- 표시: `ros2 launch limo_fire_description display.launch.xml` (XML launch 라 `robot_description` 에 `type="str"` 필요).
- 열화상·레이더·초음파·노즐 팬틸트는 장착 위치가 정해지면 모듈마다 xacro 하나씩 더한다.

## 6. YOLO 탐지 (`train/detect`)

### 무엇을 찾나

7 클래스: 0 fire · 1 smoke · 2 fire_extinguisher · 3 exit_sign · 4 person · 5 door · 6 stairs.
계단은 낙하 방지용 — 내려가는 계단은 2D 라이다에 안 잡힌다.

### 왜 YOLO26s 인가

2026-01 에 나온 YOLO26s(9.5 M, COCO 48.6)가 YOLOv8s(11.2 M, 44.9)보다 작고 정확하다. NMS 가 없어서 Jetson Orin Nano TensorRT 배포가 단순하고, 작은 물체 라벨 배정(STAL)이 멀리 있는 촛불에 유리하다. YOLO27 은 아직 없다.

### YOLO 가 내는 것과 크기·위치 구하기

YOLO 는 사진 속 상자(픽셀 위치·크기), 클래스, 확신도만 낸다. 실제 값은 거리와 합쳐 구한다.

- 실제 폭(m) = 상자 폭(px) × 거리 ÷ 초점거리 fx.
- 실제 위치 = 상자 중심 픽셀 + depth → 카메라 기준 3D 점 → TF 로 `map` 좌표. 소화기 등록·불 신고에 쓴다.
- 불은 모양이 불규칙해 상자로 재면 커진다 → 마스크를 내는 **YOLO26s-seg** 도 검토. 라벨을 마스크로 만들어 두면 둘 다 학습할 수 있다.

### 데이터

| 출처 | 클래스 | 규모 | 라이선스 |
|---|---|---|---|
| D-Fire | fire, smoke | 21,527 장 | CC0 |
| Open Images V7 | stairs, door, person | 18,552 장 | CC BY |
| LVIS v1 (COCO 이미지) | fire_extinguisher | 172 장 | CC BY |
| COCO 2017 | person | 약 64,000 장 | CC BY |
| 크롤링 → SigLIP 거르기 (`clean/`) | 비상구·소화기·촛불·계단 | 3,550 · 4,742 · 3,059 · 3,624 | 저작권 있음, 학습용만 |
| Wikimedia·Openverse (`raw/web`) | 같음 | 촛불 600 · 비상구 800 · 소화기 800 · 실내 계단 800 | CC (출처 기록) |

- 크롤링: 네이버(AutoCrawler), Bing·Baidu(icrawler), Wikimedia·Openverse(우리 Rust `webcrawl`). 거르기는 `tools/imgfilter`(Rust, SigLIP 제로샷 + 중복 제거)로 24,917 장 중 약 1 만 5 천 장 남김.
- 네이버·Bing 사진은 저작권이 있어 학습에만 쓰고 공개·재배포하지 않는다. 발표 자료에는 Wikimedia·Openverse 사진을 쓴다.
- 한국 비상구 표지·학교 계단·시연용 촛불과 인쇄한 불은 **직접 찍어야** 한다. 시험 데이터는 이걸로, 사람이 다 라벨을 단다.

### 라벨링 계획: 큰 프롬프트형 모델로 자동 라벨 → 사람 검수

1. **SAM 3**(Meta, 848 M)를 선생 모델로 쓴다. "fire extinguisher" 같은 텍스트나 예시 사진으로 물체를 찾아 상자·마스크를 낸다. 공개 모델 중 LVIS 성능이 가장 높은 축이고, 예시 사진 프롬프트가 있어 한국 비상구 표지처럼 말로 설명하기 어려운 것도 찾을 수 있다.
2. 검출에 필요한 부분(비전 인코더·텍스트 인코더·DETR 디코더·presence head)만 **C++/CUDA 로 포팅**해 2 만 장 이상을 빠르게 처리한다. 영상 추적기는 뺀다. 기준 구현과 결과가 같은지 검증한다.
3. 모든 사진에 7 클래스를 다 달아 기존 데이터의 빠진 라벨도 채운다(예: D-Fire 사진 속 사람). 안 그러면 모델이 "이건 사람이 아니다"를 잘못 배운다.
4. 클래스마다 몇백 장을 뽑아 X-AnyLabeling 으로 사람이 검수하고, 오류율을 보고 기준을 조정한다.
5. 이 라벨로 YOLO26s(-seg)를 학습(증류)한다.

- 가중치: `facebook/sam3` 의 `model.safetensors`(pickle 아님, 2026-10-23 승인받아 받음). SAM 3.1 은 영상 속도만 개선했고 `.pt` 뿐이라 3 을 쓴다.
- C++/CUDA 포팅 완료(`train/detect/teacher`): PyTorch 와 단계별 비교에서 fp32 최종 판정 100 % 같음. 라벨은 fp16 으로 — fp32 와 라벨 단위로 사실상 같고(마스크 IoU 중앙 1.000) 3.7 배 빠르다. 한 장(프롬프트 9 개, 마스크 포함) 0.15 초 → 6 만 장 약 2.5 시간.
- 원본 시험에서 본 약점: 한국식 초록 피난구 유도등을 "emergency exit sign" 으로 못 잡고, 표지 속 달리는 사람 그림을 사람으로 잡는다 → 프롬프트 다듬기·예시 사진 프롬프트·클래스 간 규칙으로 보완.

## 7. 다음 할 일

| 순서 | 일 |
|---|---|
| 1 | ~~프롬프트 다듬기 → 전체 자동 라벨~~ 됨: 학습 54,585 · 검증 6,703 · 시험 6,820 장, 상자 약 26 만 개 |
| 2 | ~~데이터셋 합치기 도구~~ 됨(`tools/yolods`) |
| 3 | ~~YOLO26s 학습기~~ 됨(처음부터 새로 작성, Ultralytics 와 정합성 확인) → 100 epoch 학습 결과 보고 라벨 잡음(자동차 문 = door 등) 손보기 |
| 4 | 학교에서 시험용 사진 찍기 (팀) |
| 5 | 탐사 RL: 관측 만들기 + PPO + 뷰어 |
| 6 | 부품 도착 후 센서 모듈 URDF·ESP32 브리지 |
| 7 | 11/30 중간평가 — PPT + 동영상 |

## YOLO26s 100 epoch 학습 결과 (2026-10-23)

C++/CUDA 학습기, 8:1:1 분할(학습 54,585 / 검증 6,703 / 시험 6,820 장), batch 32, 640, MuSGD. 정답은 SAM 3 자동 라벨.
최고는 86 epoch(검증 mAP50-95 0.6283). 마지막 10 epoch(Mosaic 끔)에는 학습 손실은 크게 줄고 검증은 0.621 까지 내려감.
best 가중치: `train/detect/data/runs/y26s_e100/best.safetensors`. 평가: `train --eval --val <분할.ycache> --weights <best>`.

| 클래스 | 검증 AP50 | 검증 AP50-95 | 시험 AP50 | 시험 AP50-95 |
|---|---|---|---|---|
| fire | 0.785 | 0.611 | 0.792 | 0.614 |
| smoke | 0.756 | 0.609 | 0.756 | 0.603 |
| fire_extinguisher | 0.847 | 0.728 | 0.871 | 0.768 |
| exit_sign | 0.731 | 0.673 | 0.712 | 0.659 |
| person | 0.737 | 0.538 | 0.718 | 0.522 |
| door | 0.771 | 0.668 | 0.781 | 0.684 |
| stairs | 0.727 | 0.572 | 0.734 | 0.592 |
| **전체** | **0.765** | **0.628** | **0.766** | **0.635** |

## 라벨 정리(SigLIP 검사) (2026-10-23)

SAM 3 라벨을 클래스마다 48 개씩 잘라 눈으로 보니 exit_sign 의 약 1/3 이 비상구가 아닌 표지(방 번호판·도로 표지·간판·소화기 표지),
door 에 차 문·찬장·소화전함이 섞여 있었다. SAM 3 에 "간판" 같은 반대 프롬프트를 더 돌려 보는 방법은 실패(진짜 비상구에도 "간판" 점수가 더 높음).
대신 상자를 잘라 SigLIP(so400m) 영(0)샷 분류로 양성 문구(비상구 표지 / 문) 대 음성 문구(간판, 차 문, 소화전함 …) 확률을 보고,
양성 확률이 exit_sign 0.3 · door 0.2 미만이면 뺐다(`train/detect/tools/labelclean/`). 표본 검사에서 빠지는 쪽은 거의 다 진짜 오류, 남는 쪽은 진짜.
fire 는 작은 촛불·과노출 불꽃을 전구로 헷갈려 쓰지 않음.

- 뺀 상자: exit_sign 6,577 / 14,184 (46 %), door 7,861 / 43,535 (18 %)
- 결과 라벨 `data/labels/sam3_clean.jsonl` → `data/yolo_clean` → `data/cache/clean_*_640.ycache`
- 이전 모델(정리 전 라벨로 학습)을 정리된 시험 라벨로 재면 mAP50-95 0.625 (exit_sign 0.708, door 0.568) — 재학습 비교 기준
- 재학습: 정리된 라벨 + `--close-mosaic 0` (이전 학습에서 Mosaic 끈 마지막 10 epoch 에 검증이 떨어졌음), `data/runs/y26s_clean`

### 재학습 결과 (정리된 라벨, `--close-mosaic 0`)

최고는 80 epoch(검증 0.6456). 끝 무렵 다시 0.640 까지 내려감 — Mosaic 를 끄지 않아도 같아서, 앞 학습의 끝 무렵 하락도 Mosaic 탓은 아닌 듯.
best: `train/detect/data/runs/y26s_clean/best.safetensors` (지금 쓸 모델).

정리된 시험 라벨, AP50-95:

| 클래스 | 이전 모델 | 새 모델 |
|---|---|---|
| fire | 0.614 | 0.611 |
| smoke | 0.603 | 0.601 |
| fire_extinguisher | 0.768 | 0.766 |
| exit_sign | 0.708 | **0.818** |
| person | 0.522 | 0.523 |
| door | 0.568 | **0.657** |
| stairs | 0.592 | 0.599 |
| **전체** | 0.625 | **0.654** (AP50 0.751 → 0.784) |
