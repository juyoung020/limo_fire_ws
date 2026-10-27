# trainer — YOLO26 학습기 (C++/CUDA)

Ultralytics `yolo detect train model=yolo26s.pt`(8.4.170) 과 같은 규칙을 PyTorch 없이 C++/CUDA 로 짰다. 처음부터 이 프로젝트용으로 새로 썼다.
`--seg` 면 `yolo segment train model=yolo26s-seg.pt` — 모델 하나로 탐지(상자)와 분할(마스크)을 같이 낸다.

## 구조

| 파일 | 내용 |
|---|---|
| `src/net.cpp, net.h` | YOLO26s 신경망(C3k2·C3k·SPPF·C2PSA·PSABlock·Detect end2end, reg_max 1). 연산마다 정방향·역방향, PyTorch state_dict 이름 그대로. seg: Segment26(마스크 계수 cv4 두 벌) + Proto26(P3·P4·P5 합치기 → 원형 마스크 32 장 160×160, ConvTranspose 2× 는 cuDNN 입력 기울기로; 의미 분할 semseg) |
| `src/conv.cpp` | cuDNN 합성곱(NHWC fp16): legacy·backend(frontend) 실행 계획을 시작 때 재서 고름 |
| `src/kern.cu` | BatchNorm(학습 통계)+SiLU, depthwise 3×3(cuDNN 이 느려 직접), concat·slice·업샘플·maxpool·편향 |
| `src/attn.cu` | PSA 어텐션(cuBLAS 묶음 GEMM + softmax) |
| `src/loss.cu` | E2E 손실: one2many(topk 10) + one2one(topk 7, topk2 1), STAL 배정, CIoU(4 방향 이중수로 미분) + L1 + BCE |
| `src/segloss.cu` | 분할 손실(v8SegmentationLoss, overlap_mask): 양성마다 계수·원형 마스크 BCE(정답 상자로 자름, /넓이), 의미 분할 0.5·BCE + 0.5·Dice. 양성 목록은 영상별 앵커 차례, 원형 마스크 기울기는 16×16 타일마다 겹치는 양성만 — 원자 연산 없이 결정적 |
| `src/optim.cu` | MuSGD(합성곱 = Muon(Newton–Schulz 5, bf16) 0.2 + SGD 1.0, BN·편향 = SGD), 분류 머리 lr ×3, 워밍업·선형 lr, 기울기 자르기 10, 동적 loss scale, EMA — 모두 장치에서 |
| `src/cache.cpp`, `tools/build_cache.cpp` | 학습 캐시: 긴 변 640 JPEG(q95, 4:4:4) 한 파일 → 고정 메모리. 마스크가 있으면 288×288 번호 지도 런 길이 부호도(YCACHE02) |
| `src/data.cpp`, `src/augment.cu` | 로더: CPU 스레드(libjpeg-turbo 풀기, 증강 난수·라벨 변환) → 고정 메모리 슬롯 → 복사 스트림 비동기 H2D → GPU 증강(Mosaic + affine + HSV + 뒤집기). 장치 버퍼 두 벌을 번갈아 써서 학습과 겹친다. 마스크 정답: 번호 지도를 같은 변환으로 따라가 출력 160×160(화소 4×4 가운데)에서 최근접으로 읽음 |
| `src/evalmap.cpp`, `src/evalmask.cu` | 검증 mAP(one2one 상위 300, NMS 없음, Ultralytics 짝짓기·101 점 AP). 마스크 mAP: 예측 마스크(로짓 > 0, 상자로 자름)와 정답의 IoU 를 160×160 에서(Ultralytics process_mask) |
| `tools/train.cpp` | 학습 도구(이어 하기 `--resume`) |
| `tools/*_parity.cpp`, `tools/aug_dump.cpp` | 정합성 시험, 증강 그림 |

## 정합성 (fp32 빌드, Ultralytics 와 비교)

기준값 스크립트는 파이썬이라 저장소에 넣지 않았다(`ref/`, git 제외). 가중치는 `tools/ptconv`(pickle 을 실행하지 않는 Rust 변환기)로 만든 safetensors 를 넣었다.

| 시험 | 결과 |
|---|---|
| 신경망 정방향(학습 모드, 2 장 × 320) | 중간·머리 출력 상대 오차 ≤ 7e-4, BN running 통계 1e-6 |
| 신경망 역방향(머리 출력에 정해진 기울기) | 파라미터 기울기 상대 ≤ 2e-3(수학적으로 0 인 BN β 기울기 7 개 제외 — 양쪽 다 반올림 잡음) |
| E2E 손실(4 장, 정답 12 개씩) | 손실 항목 상대 2e-7, 머리 출력 기울기 ≤ 5e-6 |
| MuSGD 두 스텝 | SGD 부분 1e-7, Muon 부분 갱신량의 평균 0.26 %(bf16 직교화 반올림) |
| seg 신경망(yolo26s-seg, 2 장 × 320) | 정방향 출력(마스크 계수·원형 마스크·의미 분할 포함) 상대 ≤ 8e-4, 파라미터 기울기 ≤ 1e-3(위와 같은 BN β 7 개 제외) |
| seg 손실(4 장 × 320 정답 12 개, 8 장 × 640 정답 60 개) | 손실 항목(box·seg·cls·l1·sem) 상대 3e-7, 머리 출력 기울기(마스크 계수·원형 마스크·의미 분할 포함) ≤ 3e-5 |

## 실행

```bash
cd train/detect/trainer
cmake -S . -B build && cmake --build build -j
../tools/ptconv/target/release/ptconv ../data/weights/yolo26s.pt ../data/weights/yolo26s.safetensors model
./build/build_cache ../data/yolo train ../data/cache/train_640.ycache   # 5.5 만 장 16 초
./build/build_cache ../data/yolo val ../data/cache/val_640.ycache
./build/train --train ../data/cache/train_640.ycache --val ../data/cache/val_640.ycache \
  --weights ../data/weights/yolo26s.safetensors --out ../data/runs/y26s --batch 32 --epochs 100
```

탐지 + 분할(`../data/yolo_seg` = yolods 가 SAM 3 마스크까지 낸 것):

```bash
../tools/ptconv/target/release/ptconv ../data/weights/yolo26s-seg.pt ../data/weights/yolo26s-seg.safetensors model
./build/build_cache ../data/yolo_seg train ../data/cache/seg_train_640.ycache   # val·test 도
./build/train --seg --train ../data/cache/seg_train_640.ycache --val ../data/cache/seg_val_640.ycache \
  --weights ../data/weights/yolo26s-seg.safetensors --out ../data/runs/y26s_seg --batch 32 --epochs 100
./build/train --seg --eval --val ../data/cache/seg_test_640.ycache --weights ../data/runs/y26s_seg/best.safetensors
```

seg 속도: **289 장/초**(탐지만 393). 100 epoch(5.1 시간) 뒤 시험 상자 mAP50-95 0.651 · 마스크 0.521 — [docs/detect.md](../../../docs/detect.md).
평가에 `--draw <폴더>` 를 붙이면 처음 N 장(`--draw-n`, 기본 24)의 예측(상자·마스크, 확신도 `--draw-conf` 0.4 이상)을 PPM 그림으로 저장한다.

## 속도 (RTX 5070 Ti, 640, 배치 32)

| 단계 | 장/초 |
|---|---|
| 처음(cuDNN 그대로) | 173 |
| depthwise 3×3 직접 커널, BN 역방향 원소별 FP64 제거 | 299 |
| BN·concat·더하기·업샘플 커널 16 B 벡터화 | 306 |
| 입력 3 → 8 채널 덧대기(첫 합성곱 텐서코어), 가중치 변환 한 커널, Muon 정규화 병렬, 반복·옵티마이저 CUDA 그래프 | 327 |
| BN 통계 2 단계 합산(블록별 부분합 → 병렬 합, 같은 주소 atomic 없앰) | 340~346 |
| concat 복사 없앰: 합성곱이 읽지 않는 concat 입력(40 개 중 38 개)은 버퍼 안 자기 자리에 바로 씀 | 359~370 |
| cuDNN backend 엔진(frontend v1.31): 합성곱마다 legacy 와 backend 실행 계획을 다 재서 빠른 쪽, backend 결과는 legacy 와 비교해 맞는 것만. 입력 기울기는 그 버퍼에 처음 쓰는 연산만(backend 는 더하기가 없음) | **393** |

합성곱 + BN 통계 융합(conv_fprop + genstats)은 RTX 5070 Ti(sm_120)에 엔진이 없어 못 함(cuDNN 9.27, "No valid engine configs"). backend FFT 엔진은 재는 중에 버퍼 밖에 써서 뺐다.

한 epoch(5.5 만 장) 약 2 분 30 초. 이제 반복 시간(약 87 ms)의 대부분은 합성곱(cuDNN)과 BN 자체. cuDNN 은 채널 간격이 있는 입력을 받지 않아 합성곱 입력은 꽉 찬 버퍼로 둔다. 남은 여지: cuDNN backend 엔진·합성곱+BN 통계 융합.

## 첫 결과

1 epoch 뒤 검증(SAM 3 라벨) mAP50 0.583 · mAP50-95 0.428 (fire 0.401 · smoke 0.272 · fire_extinguisher 0.590 · exit_sign 0.459 · person 0.467 · door 0.437 · stairs 0.370).
