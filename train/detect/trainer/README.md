# trainer — YOLO26 학습기 (C++/CUDA)

Ultralytics `yolo detect train model=yolo26s.pt`(8.4.170) 과 같은 규칙을 PyTorch 없이 C++/CUDA 로 짰다. 처음부터 이 프로젝트용으로 새로 썼다.

## 구조

| 파일 | 내용 |
|---|---|
| `src/net.cpp, net.h` | YOLO26s 신경망(C3k2·C3k·SPPF·C2PSA·PSABlock·Detect end2end, reg_max 1). 연산마다 정방향·역방향, PyTorch state_dict 이름 그대로 |
| `src/conv.cpp` | cuDNN 합성곱(NHWC fp16), 알고리즘은 시작 때 재서 고름 |
| `src/kern.cu` | BatchNorm(학습 통계)+SiLU, depthwise 3×3(cuDNN 이 느려 직접), concat·slice·업샘플·maxpool·편향 |
| `src/attn.cu` | PSA 어텐션(cuBLAS 묶음 GEMM + softmax) |
| `src/loss.cu` | E2E 손실: one2many(topk 10) + one2one(topk 7, topk2 1), STAL 배정, CIoU(4 방향 이중수로 미분) + L1 + BCE |
| `src/optim.cu` | MuSGD(합성곱 = Muon(Newton–Schulz 5, bf16) 0.2 + SGD 1.0, BN·편향 = SGD), 분류 머리 lr ×3, 워밍업·선형 lr, 기울기 자르기 10, 동적 loss scale, EMA — 모두 장치에서 |
| `src/cache.cpp`, `tools/build_cache.cpp` | 학습 캐시: 긴 변 640 JPEG(q95, 4:4:4) 한 파일 → 고정 메모리 |
| `src/data.cpp`, `src/augment.cu` | 로더: CPU 스레드(libjpeg-turbo 풀기, 증강 난수·라벨 변환) → 고정 메모리 슬롯 → 복사 스트림 비동기 H2D → GPU 증강(Mosaic + affine + HSV + 뒤집기). 장치 버퍼 두 벌을 번갈아 써서 학습과 겹친다 |
| `src/evalmap.cpp` | 검증 mAP(one2one 상위 300, NMS 없음, Ultralytics 짝짓기·101 점 AP) |
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

## 속도 (RTX 5070 Ti, 640, 배치 32)

| 단계 | 장/초 |
|---|---|
| 처음(cuDNN 그대로) | 173 |
| depthwise 3×3 직접 커널, BN 역방향 원소별 FP64 제거 | **299** (데이터 기다림 1 % 미만) |

한 epoch(5.5 만 장) 약 3 분, 100 epoch 약 5.5 시간. 남은 시간의 약 30 % 가 BN 원소별 커널(벡터화·합치기 여지).

## 첫 결과

1 epoch 뒤 검증(SAM 3 라벨) mAP50 0.583 · mAP50-95 0.428 (fire 0.401 · smoke 0.272 · fire_extinguisher 0.590 · exit_sign 0.459 · person 0.467 · door 0.437 · stairs 0.370).
