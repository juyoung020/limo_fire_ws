# data

YOLO 학습 이미지·라벨·사전학습 가중치. 용량 때문에 git 에 넣지 않는다(이 파일만 올라감). 받은 날짜 2026-10-23.

## 받아 둔 것 (`raw/`, `weights/`)

| 폴더 | 데이터 | 우리 클래스 | 규모 | 라이선스 |
|---|---|---|---|---|
| `raw/dfire/` | **D-Fire** (실내·실외 화재 사진, YOLO 형식, 원본 클래스 0 = smoke, 1 = fire) | fire, smoke | 학습 17,221 + 시험 4,306 장, fire 상자 14,692 · smoke 11,865 | **CC0 1.0** |
| `raw/openimages/` | **Open Images V7** 중 Stairs·Door 가 든 이미지 + 그 이미지의 Person 상자 | stairs, door, person | 18,552 장 (train 17,646 · val 237 · test 669), 상자 29,185 | 라벨 CC BY 4.0, 이미지 CC BY 2.0 |
| `raw/lvis/` | **LVIS v1** 라벨(이미지는 COCO, `~/ObjectSAM/data/coco` 에 있음) | fire_extinguisher (+ fire_alarm, door 손잡이 참고) | fire_extinguisher 172 장 · 211 상자 | CC BY 4.0 |
| `~/ObjectSAM/data/coco` | **COCO 2017** (이미 있음, 20 GB) | person | 약 64,000 장 | CC BY 4.0 |
| `weights/yolo26s.pt`, `yolo26n.pt` | **Ultralytics YOLO26** COCO 사전학습 (v8.4.0 릴리스) | 시작점 | 20.4 MB, 5.5 MB | AGPL-3.0 |
| `raw/autocrawler/` | **Naver 이미지 검색** (`../tools/autocrawler`, AutoCrawler) — 한국 비상구 표지·소화기·소화전함, 촛불, 실내 계단. 라벨 없음 | exit_sign, fire_extinguisher, fire, stairs | 목표 2,000 · 2,000 · 1,500 · 2,000 (거르기 전 클래스당 약 3,000) | 저작권 있음 — 학습용만, 재배포 금지 |
| `raw/icrawl/` | **Bing·Baidu 이미지 검색** (`../tools/icrawl`, icrawler) — 같은 검색어. 라벨 없음 | 위와 같음 | 촛불 2,836 · 비상구 2,437 · 소화기 3,114 · 계단 2,587 | 저작권 있음 — 학습용만, 재배포 금지 |
| `clean/<클래스>/` | 위 셋(autocrawler·icrawl·web)을 `../tools/imgfilter`(SigLIP)로 거른 것, 원본 하드링크. 판정은 `clean/<클래스>.tsv` | exit_sign, fire_extinguisher, fire(촛불), stairs | **비상구 3,550 · 소화기 4,742 · 촛불 3,059 · 계단 3,624** (받은 24,917 장 중) | 원본과 같음 |
| `weights/siglip-so400m-patch14-384/` | Google SigLIP (거르기용) | — | 3.5 GB | Apache-2.0 |

`.pt` 는 파이썬 pickle 이라 파이썬으로 열면 코드가 실행될 수 있다. 가중치 변환기는 zip 안의 텐서 자료만 읽는다(파이썬으로 열지 않음).

## 아직 없는 것 — 직접 찍어야 함

| 클래스 | 이유 |
|---|---|
| **exit_sign** | 공개 데이터는 미국식 빨간 "EXIT" 위주. 한국 비상구 표지(초록 달리는 사람)와 모양이 달라 그대로 쓰면 안 맞는다. Roboflow Universe 에 CC BY 4.0 데이터가 있지만 받으려면 계정·API 키가 필요 |
| **fire_extinguisher** 추가분 | LVIS 172 장으로는 부족. 한국 빨간 분말 소화기·소화전함 사진 필요 |
| **촛불·인쇄한 불** | 시연 물체 그대로 — 거리·각도·조명을 바꿔 가며 |
| **계단(실내)** | Open Images 계단은 실외·주택이 많다. 학교 건물 계단 사진을 더하면 좋다 |

학교 건물(강의동·과학관)에서 휴대폰·LIMO 카메라로 찍고 라벨을 단다. 라벨 도구는 정해서 여기에 적는다.

## 클래스 번호 (`../README.md` 와 같음)

0 fire · 1 smoke · 2 fire_extinguisher · 3 exit_sign · 4 person · 5 door · 6 stairs
