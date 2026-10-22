# detect — YOLO 탐지

불·연기·소화기·비상구 표지·사람·문·계단을 찾는 YOLO 탐지기. 순찰(소방 시설 지도)과 화재 대응(불 크기 판단)에 쓴다.

| 폴더 | 내용 | 상태 |
|---|---|---|
| `trainer/` | C++/CUDA 학습기. ObjectSAM(`~/ObjectSAM/src`, YOLOv8s-seg 1 클래스, 4,253 줄)을 바탕으로 **YOLO26s**(COCO 사전학습, 2026-01) 구조로 옮기고, 클래스 7 개·상자만 학습하게 바꾼다 | 할 일 |
| `teacher/` | 자동 라벨러: SAM 3(텍스트 프롬프트 검출·분할)를 C++/CUDA 로 옮긴 것. fp16 6.7 장/초, PyTorch 와 정합성 확인 — [teacher/README.md](teacher/README.md) | 됨 |
| `tools/` | 크롤러·거르기(imgfilter), `yolods`: SAM 3 라벨 → YOLO 데이터셋(규칙: 클래스별 문턱·NMS·표지 속 사람 그림 제거, 8:1:1) | 됨 |
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
| YOLOv8s (ObjectSAM 학습기 구조) | 2023 | 11.2 M | 44.9 | |
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
