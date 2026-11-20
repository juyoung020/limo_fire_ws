# detect — YOLO 탐지

불·연기·소화기·비상구 표지·사람·문·계단을 찾는 YOLO 탐지기. 순찰(소방 시설 지도)과 화재 대응(불 크기 판단)에 쓴다.

| 폴더 | 내용 | 상태 |
|---|---|---|
| `trainer/` | YOLO26s(탐지) · YOLO26s-seg(탐지 + 분할) C++/CUDA 학습기 — Ultralytics 와 정합성 확인, 393 · 289 장/초 — [trainer/README.md](trainer/README.md) | 됨 — 시험 상자 mAP50-95 0.651 · 마스크 0.521, 정리는 [docs/detect.md](../../docs/detect.md) |
| `teacher/` | 자동 라벨러: SAM 3(텍스트 프롬프트 검출·분할)를 C++/CUDA 로 옮긴 것. fp16 6.7 장/초, PyTorch 와 정합성 확인 — [teacher/README.md](teacher/README.md) | 됨 |
| `tools/` | 크롤러·거르기(imgfilter), `yolods`: SAM 3 라벨 → YOLO 데이터셋(규칙: 클래스별 문턱·NMS·표지 속 사람 그림 제거, 8:1:1) + 마스크 번호 지도(.msk) | 됨 |
| `data/` | 이미지·라벨·사전학습 가중치 (git 제외). 받은 것과 라이선스는 [data/README.md](data/README.md) | 공개 데이터 받음, 직접 찍을 것 남음 |

## 클래스 (안)

| 번호 | 이름 | 쓰는 곳 |
|---|---|---|
| 0 | fire | 불 탐지·크기 판단 (촛불·인쇄한 불 포함) |
| 1 | smoke | 연기 모드 판단 |
| 2 | fire_extinguisher | 소방 시설 지도 |
| 3 | exit_sign | 소방 시설 지도·대피 안내 |
| 4 | person | 대피 안내 (연기 속은 레이더) |
| 5 | door | 소방 시설 지도·구역 나누기 |
| 6 | stairs | 낙하 방지 — 소방 시설 지도에 진입 금지 구역으로 등록 (내려가는 계단은 2D 라이다에 안 잡힘) |

## 왜 YOLO26s 에서 시작하나

| 모델 | 출시 | 파라미터 | COCO mAP | 비고 |
|---|---|---|---|---|
| YOLOv8s | 2023 | 11.2 M | 44.9 | |
| **YOLO26s** | 2026-01 | 9.5 M | 48.6 (NMS 없이 47.8) | NMS 없는 출력·DFL 제거로 Jetson TensorRT 배포가 단순, 작은 물체 라벨 배정(STAL) — 멀리 있는 촛불 |
| YOLO26n | 2026-01 | 2.4 M | 40.9 | 온보드가 빠듯하면 |

YOLO27 은 2026-10 기준 미출시. LIMO Pro 온보드는 Jetson Orin Nano(40 TOPS) — s 와 n 을 둘 다 학습해 실물에서 잰다.

## 데이터 만들기 (2026-10-23)

1. 목록·나눔: 크롤링 거른 것 14,975 + Wikimedia·Openverse 3,000 + D-Fire 21,527 + Open Images 18,552 + LVIS 소화기 172 + COCO 무작위 9,988 = **68,214 장**. 원본 경로 sha1 로 학습 : 검증 : 시험 = **8 : 1 : 1** (54,672 / 6,714 / 6,828). `data/lists/`.
2. 라벨: `teacher/` SAM 3 가 7 클래스 모두 단다(사람 라벨이 있는 데이터도 상자 모양을 맞추려고 SAM 3 로 통일). 프롬프트는 `teacher/config/prompts.json`.
3. 합치기: `tools/yolods` → `data/yolo/` (YOLO 형식, 이미지는 원본으로 링크).

### 프롬프트 고르기

사람 라벨이 있는 표본(D-Fire 300, Open Images 300, LVIS 190, COCO 200)으로 후보마다 정밀도·재현율(IoU 0.5, 점수 0.5)을 쟀다.

| 클래스 | 고른 프롬프트 | P / R @0.5 | 버린 후보 |
|---|---|---|---|
| fire | "fire" + "candle flame"(시연 촛불용) | 0.39 / 0.46 | flame(같음), burning fire(비슷) |
| smoke | "smoke" | 0.44 / 0.68 | white smoke, black smoke(재현율 0.2 미만) |
| fire_extinguisher | "fire extinguisher" | 0.92 / 0.86 | red fire extinguisher(낮음) |
| person | "person" | 0.54 / 0.91 (COCO) | human(같음) |
| door | "door" | 0.49 / 0.82 | wooden door(재현율 0.40) |
| stairs | "stairs" | 0.42 / 0.72 | staircase(같음), steps(정밀도 낮음) |
| exit_sign | "emergency exit sign" + "green emergency exit light with running man" | 사람 라벨 없음 — 크롤링 120 장 눈으로 | exit sign, green exit sign(덜 잡음) |

불·연기·문의 낮은 정밀도는 대부분 사람 라벨 쪽이 빠뜨린 것이다(그림으로 확인: Open Images 는 사진 속 문을 다 달지 않고, D-Fire 는 불을 묶는 단위가 다르다). 진짜 오답으로 보인 것: 자동차 문을 door 로, 초록 도로 표지판을 exit_sign 으로, 비상구 표지 속 달리는 사람 그림을 person 으로(→ yolods 규칙 3).

### 결과 (`data/yolo/stats.json`)

SAM 3 라벨 6.8 만 장, 2 시간 48 분(6.8 장/초), 원시 검출 407,715 개(점수 0.3 이상). 못 읽은 사진 106 장(확장자와 실제 형식이 다른 것 등)은 뺐다. 규칙으로 버린 것: 문턱 181,649 · 겹침 11,101 · 표지 속 사람 그림 3,360 · 작음 308.

| | 사진 | fire | smoke | fire_extinguisher | exit_sign | person | door | stairs |
|---|---|---|---|---|---|---|---|---|
| 학습 | 54,585 | 24,693 | 14,324 | 6,824 | 9,560 | 65,242 | 34,549 | 14,473 |
| 검증 | 6,703 | 2,908 | 1,676 | 835 | 1,179 | 8,269 | 4,229 | 1,883 |
| 시험 | 6,820 | 2,971 | 1,730 | 731 | 1,266 | 7,767 | 4,385 | 1,803 |

시험 6,820 장도 SAM 3 라벨이다 — 학습기 비교용. 실제 성능은 학교에서 찍어 사람이 라벨을 단 사진으로 따로 잰다.

## 미세 조정 — 실물 불꽃 과녁·소화기 소품

배포 모델(`yolo26s-seg_fire7`)은 대회장의 **3D 출력 불꽃 과녁**(`hardware/modeling/불꽃과녁`)을 거리에 따라 놓치고(실물 사진 6 장 중 2 장),
**소화기 소품**(`hardware/modeling/소화기`, 높이 약 19 cm)도 멀어지면 끊겼다. 원래 학습 사진 6.8 만 장은 이 PC 에 없어서,
과녁·소품을 렌더한 사진 + 옛 클래스를 잊지 않게 하는 공개 사진(replay)으로 **짧게 더 학습**했다. 과녁은 임무가 찾는 클래스 `fire`(0)
(`limo_fire_bt` `classes.fire`, `yolo_fire.yaml` `class_names`), 소품은 `fire_extinguisher`(2).

### 데이터 (만드는 도구는 파이썬 보조 스크립트, 저장소 밖 `~/limo_upper_keep/yolo_ft/scripts`)

| 묶음 | 장수 | 만드는 법 |
|---|---|---|
| 경기장 렌더 | 학습 3,000 · 검증 300 | Blender 4.5 Cycles. 과녁 STL(`불꽃과녁/asm_check`: 빨강 판·흰 속불꽃·회색 받침·흰 무게추) + 소화기 STL(머리·노즐 조립), 카메라 = 실물 Orbbec 컬러(640×480, fx 489.39, 높이 0.165~0.215 m, 아래로 5.2° ± 1.7°, `train/explore/sim/config` 값). 거리 0.45~4.5 m, 과녁 정면 기준 ±60°(가끔 옆·뒤), 회색 바닥 + 흰 테이프 선 + 흰 벽·주름 띠(경기장), 그 밖 바닥(초록 광택·타일·무작위), 조명 3,000~7,000 K, 빨간 상자 같은 헷갈리는 물체, 가림 15 %. 정답 = 렌더의 물체 번호 패스(과녁은 빨강 판 + 흰 속불꽃만, 받침·무게추 제외) |
| 합성 | 학습 800 | 투명 배경 렌더(그림자 포함)를 실내 사진(Wikimedia Commons 450 장) 위에 붙임, 가장자리 그대로/흐림 섞기 ([Cut, Paste and Learn](https://arxiv.org/abs/1708.01642)). 배경 사진 라벨은 fire7 의사 라벨, 가려진 것은 지움 |
| replay | 학습 2,019(×3 반복) · 검증 356 | Wikimedia Commons(자유 라이선스) 불·연기·소화기·비상구·계단·문·사람 2,375 장에 **fire7 의사 라벨**(점수 ≥ 0.4, `tools/yolods` 규칙) — 옛 클래스를 잊지 않게 ([Ultralytics: 새 클래스만 학습하면 옛 클래스를 잊음](https://github.com/orgs/ultralytics/discussions/6849)) |

렌더·합성 사진에는 로봇 카메라처럼 흐림·움직임 흐림·잡음·화이트밸런스·감마·해상도 낮춤·JPEG(45~95)를 무작위로 넣었다.
SAM 3 선생은 이 PC(RTX 3060, 다른 작업이 GPU 7.6 GB 사용)에서 메모리가 모자라 못 돌렸다 — replay 라벨·검증 정답 모두 fire7 의사 라벨이라
"다른 클래스" 점수는 **옛 모델과 얼마나 같게 내나(변화량)** 를 잰 것이다.

### 학습 (2026-11-20, RTX 3060 — 다른 작업과 GPU 를 나눠 배치 6)

```bash
cd train/detect/trainer && cmake -S . -B build -DYT_CUDA_ARCH=86 && cmake --build build -j --target train build_cache
./build/build_cache <ds> train train.ycache && ./build/build_cache <ds> val val.ycache
./build/train --seg --freeze-bn --train train.ycache --val val.ycache --weights yolo26s-seg_fire7.safetensors --out runs/target_ft \
  --batch 6 --epochs 10 --lr0 0.0005 --lrf 0.2 --warmup-epochs 1 --close-mosaic 1
```

| 항목 | 값 | 까닭 |
|---|---|---|
| 시작 | 배포 모델 `yolo26s-seg_fire7` | 처음부터가 아니라 조금 더 |
| 학습률 | 0.0005 → 0.0001 (원래 학습의 1/20), 워밍업 1 epoch | 0.002 로 하면 다른 클래스 변화가 더 컸다(아래 표 ft1) |
| BN | `--freeze-bn`: 평균·분산을 불러온 값으로 고정, γ·β 만 학습 | 배치 6 은 통계가 흔들리고 렌더 사진 통계로 끌려감 (Detectron2 `FrozenBatchNorm2d` 와 같은 생각) |
| 사진 | 9,857 장/epoch (렌더 3,000 + 합성 800 + replay 2,019 × 3), 10 epoch, 약 27 분 | |
| 고르기 | 검증용 접근 시퀀스 4 개(씨앗 8001~8004)의 과녁·소화기 탐지 비율이 가장 높은 것 → 10 epoch 끝(`last`) | 학습기 기본(검증 mAP 합)은 늘 1 epoch 을 골랐다 — replay 정답이 옛 모델 출력이라 덜 배운 쪽이 이김 |

### 결과 (시험 = 학습·고르기에 안 쓴 것)

평가는 로봇과 같은 입력: 배포 형식 ONNX(uint8 480×640, 덧대기 없음), 점수 문턱 0.35, IoU ≥ 0.5.
실물 사진은 `target_data/real_robot` 의 6 장(로봇 카메라 1 장 + 휴대폰 탁자 사진과 그 축소판)이라 **적다** — 아래 "더 필요한 실물 자료".

| | 옛 모델 fire7 | 미세 조정판 |
|---|---|---|
| 실물 과녁 6 장 (fire) | 2 / 6 (점수 0.25 · 0.50 · 0.50 · 0.03 · 0.02 · 0) | **6 / 6** (0.94 ~ 0.96, 가까운 확대 사진만 0.46) |
| 실물 소화기 소품 (real_cam) | 0.75 | 0.47 (잡음, 점수는 내려감) |
| 실물 사진 크기 바꾸기 30 단계 — 과녁 (real_cam · 탁자) | 0 % · 10 % | **100 % · 93 %** (최장 놓침 0 · 2) |
| 같은 방법 — 소화기 | 17 % | 83 % (못 잡은 5 장은 모두 약 5.8 m 보다 먼 크기, 4 m 안쪽에 해당하는 크기는 19 / 19) |
| 합성 접근 시퀀스 10 × 60 장, 과녁 (깨끗 / 열화) | 35.5 % · 6.7 % | **99.5 % · 97.8 %** (최장 놓침 1 · 4) |
| 같은 시퀀스, 소화기 (깨끗 / 열화) | 7.2 % · 15.5 % | **97.8 % · 95.8 %** (최장 놓침 1 · 5) |
| 같은 시퀀스, 과녁·소화기 오탐 / 장 | 0.055 · 0.007 | 0.015 · 0.023 |

다른 클래스 (replay 검증 356 장, 정답 = 옛 모델 의사 라벨 → **옛 모델과 달라진 정도**, 상자 AP50-95 (AP50)):

| | fire(공개 사진) | smoke | fire_extinguisher | exit_sign | person | door | stairs |
|---|---|---|---|---|---|---|---|
| 옛 모델 | 0.980 (0.984) | 0.968 (0.976) | 0.959 (0.990) | 0.985 (0.985) | 0.963 (0.986) | 0.985 (0.986) | 0.975 (0.979) |
| 미세 조정판 | 0.781 (0.868) | 0.839 (0.899) | 0.911 (0.981) | 0.932 (0.944) | 0.898 (0.962) | 0.921 (0.956) | 0.844 (0.920) |

문턱 0.35 로 보면 옛 모델이 0.7 이상으로 확신한 물체 중 새 모델도 잡는 비율: fire 83 % · smoke 96 % · 소화기 98 % · 비상구 100 % · 사람 99 % · 문 96 % · 계단 95 %.
**진짜 불(공개 사진)과 연기는 점수가 내려갔다** — fire 클래스가 과녁 쪽으로 기울었다. 미리 정한 기준(다른 클래스 평균 하락 ≤ 0.02)은 **못 맞췄다**.
시도한 것: 학습률 0.002(ft1, 하락 더 큼) · 0.0005(ft2) · BN 고정(ft3) · 불·연기 replay 6 배(ft5, 차이 없음).

### 로봇에 넣기 (설정·launch 그대로 — 엔진 파일만 바꿈)

`yolo_fire.yaml` 의 `engine_path`(= `real.launch.xml` 이 쓰는 것) 자리에 그대로 만든다. 결과는 운용 화면(`limo_fire.rviz`)의 `/perception/detection_image`.
ONNX 는 [limo_fire_perception README](../../src/limo_fire/limo_fire_perception/README.md#엔진-만들기-기기마다-한-번) 와 같은 순서(uint8 480×640 입력 노드)로 만든 것이다.

파일(아직 Release 에 안 올림 — 올릴 때 이름 그대로): `yolo26s-seg_fire7_target.safetensors`(학습기 형식, sha256 `2c05abff…e6be6e`)와
`yolo26s-seg_fire7_target_u8_480.onnx`(sha256 `36f28760…22a5d6`). 학습한 PC 의 `~/limo_upper_keep/yolo_ft/release/` 에 있다.

```bash
# 로봇에서 (약 27 분). 옛 엔진은 .fire7_bak 으로 남겨 되돌릴 수 있게
cd ~/limo_fire_ws/src/limo_fire/limo_fire_perception/models
cp yolo26s_seg_fire_u8_480.engine yolo26s_seg_fire_u8_480.engine.fire7_bak
/usr/src/tensorrt/bin/trtexec --onnx=$HOME/yolo26s-seg_fire7_target_u8_480.onnx --fp16 --builderOptimizationLevel=5 \
  --saveEngine=$HOME/limo_fire_ws/src/limo_fire/limo_fire_perception/models/yolo26s_seg_fire_u8_480.engine
```

### 더 필요한 실물 자료

- 로봇 카메라로 과녁·소화기 소품 둘레를 2~3 분 녹화(`record.launch.xml`, 컬러 영상 포함): 4 m → 0.5 m 다가가기, 정면·±60° 비스듬히, 불 켠/끈 조명, 경기장 두세 곳.
  앞쪽 절반은 학습(실물 라벨은 SAM 3 또는 손으로), 뒤쪽 절반은 시험 — 지금 실물 시험이 6 장뿐이라 숫자가 거칠다.
- 촛불·인쇄한 불 같은 **진짜 불 시연 물체**를 같은 카메라로 찍은 것 — 위 하락이 실제 시연에 문제인지 재고, 학습에 넣어 되돌리기.
