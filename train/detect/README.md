# detect — YOLO 탐지

불·연기·소화기·비상구 표지·사람·문·계단을 찾는 YOLO 탐지기. 순찰(소방 시설 지도)과 화재 대응(불 크기 판단)에 쓴다.

| 폴더 | 내용 | 상태 |
|---|---|---|
| `trainer/` | C++/CUDA 학습기. ObjectSAM(`~/ObjectSAM/src`, YOLOv8s-seg 1 클래스, 4,253 줄)을 바탕으로 **YOLO26s**(COCO 사전학습, 2026-01) 구조로 옮기고, 클래스 7 개·상자만 학습하게 바꾼다 | 할 일 |
| `teacher/` | 자동 라벨러: SAM 3(텍스트 프롬프트 검출·분할)를 C++/CUDA 로 옮긴 것. fp16 6.7 장/초, PyTorch 와 정합성 확인 — [teacher/README.md](teacher/README.md) | 됨 |
| `tools/` | 크롤러·거르기(imgfilter), 데이터셋 → 학습 형식 변환(Rust) | 크롤·거르기 됨, 변환 할 일 |
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
