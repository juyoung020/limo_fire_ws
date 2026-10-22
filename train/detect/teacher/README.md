# teacher — SAM 3 자동 라벨러 (C++/CUDA)

크롤링·공개 사진에 7 클래스 라벨(상자 + 마스크)을 다는 선생 모델. Meta **SAM 3**(2025-11, 848 M, 텍스트 프롬프트 검출·분할)의 검출기 부분을 PyTorch 없이 C++/CUDA 로 옮겼다. 영상 추적기는 뺐다.

## 왜 SAM 3

텍스트("fire extinguisher")나 예시 사진으로 물체를 찾아 상자·마스크를 낸다. 2026-10 기준 공개 가중치 모델 중 LVIS 성능이 가장 높은 축이다. 가중치 `facebook/sam3`(SAM License, 승인제 — 2026-10-23 승인)의 `model.safetensors` 를 쓴다(pickle 아님). 받는 곳: `../data/weights/sam3/`.

## 구조

| 파일 | 내용 |
|---|---|
| `src/sam3.h, sam3.cu` | 네트워크: ViT-L 32 층(창 24 + 전역 4, 2D RoPE) → FPN 3 단 → CLIP 텍스트 24 층 → DETR 인코더 6 · 디코더 6(질의 200, 존재 토큰, BoxRPB) → 점수 → 마스크 디코더 |
| `src/k.cu, k.cuh` | 커널: LayerNorm, GroupNorm, RoPE, 창 나누기, NHWC 3×3 합성곱(GEMM 9 번), 전치 합성곱, cuBLAS/cuBLASLt GEMM(편향·ReLU·잔차 에필로그) |
| `src/fa.cu` | 텐서코어 플래시 어텐션(mma.sync m16n8k16, fp16/bf16 입력·float 누적, 키 마스크·인과·BoxRPB 편향) |
| `src/prep.cpp` | 사진 읽기(stb_image) + Pillow 와 같은 8 비트 양선형 리사이즈(ObjectSAM 에서 가져옴) → 1008×1008 |
| `src/tok.cpp` | CLIP BPE 토크나이저(영어 ASCII 프롬프트) |
| `src/st.cpp` | safetensors 읽기(mmap) |
| `tools/sam3_label.cpp` | 라벨 도구 |
| `tools/sam3_parity.cpp` | PyTorch 기준값과 단계별 비교 |
| `tools/sam3_bench.cpp` | 처리량 |
| `config/prompts.json` | 클래스별 프롬프트 |

정밀도는 템플릿 `Model<T>`: `float`(정합성 확인), `f16`(라벨용, 기본), `bf16`.

## 빌드·실행

```bash
cd train/detect/teacher
cmake -S . -B build && cmake --build build -j
find ../data/clean -name '*.jpg' > list.txt
./build/sam3_label --weights ../data/weights/sam3 --prompts config/prompts.json --list list.txt --out labels.jsonl
```
옵션: `--thr 0.3`(이 점수 넘는 검출만 기록), `--threads 8`(읽기 스레드), `--f32`, `--no-masks`, `--limit N`. 같은 `--out` 으로 다시 돌리면 이미 한 사진은 건너뛴다.

출력(JSONL, 사진 하나에 한 줄): `{"img", "w", "h", "det": [{"p": 프롬프트 번호, "c": 클래스, "s": 점수, "b": [x1,y1,x2,y2] 원본 픽셀, "m": 288×288 행 우선 RLE(0 부터)}]}`. 점수 = sigmoid(질의 로짓) × sigmoid(존재 로짓)(원본 후처리와 같음). 프롬프트·설정은 `labels.jsonl.meta.json`. 문턱·클래스 간 충돌 정리는 여기서 하지 않고 다음 단계(데이터 합치기)에서 한다.

## 검증 (2026-10-23, RTX 5070 Ti)

PyTorch 기준값: transformers 5.18.0 `Sam3Model`(fp32, SDPA), 소화기 사진 1 장 × 프롬프트 7 개. 기준값을 뽑는 스크립트는 파이썬이라 저장소에 넣지 않았다(`ref/`, git 제외).

| 단계 | fp32 상대 오차(최대) | fp16 |
|---|---|---|
| 전처리(픽셀) | 7.8e-3 — 0.07 % 화소가 1/255 차이(torchvision 과 Pillow 반올림 차) | 같음 |
| 토큰 | 0 (같음) | — |
| CLIP 텍스트 출력 | 2.4e-6 | 9.5e-4 |
| ViT 층 0 / 7 / 출력 | 2.2e-6 / 2.3e-6 / 6.8e-5 | 6e-4 / 1e-3 / 2.2e-2 |
| FPN 3 단 | ≤ 5.7e-5 | ≤ 2.7e-2 |
| DETR 인코더 | 5.0e-5 | 3.4e-2 |
| 질의 로짓 / 상자 / 존재 | 2.2e-4 / 3.9e-4 / 2.9e-6 | 5.8e-2 / 1.5e-1 / 6e-4 |
| 마스크 로짓 | 5.8e-4 | 3.1e-1 |
| 최종 판정(점수 > 0.4, 1,400 개) | 모두 같음 | 모두 같음 |

라벨 단위 비교(크롤링 사진 300 장, 프롬프트 9 개, fp16 대 fp32): 점수 > 0.5 검출 1,032 개 짝, 한쪽만 8 개(문턱 근처), 점수 차 중앙 0.0003 · 최대 0.028, 상자 IoU 최소 0.962, 마스크 IoU 중앙 1.000 · 하위 5 % 0.998. → fp16 으로 라벨을 단다.

## 속도 (1008×1008, 프롬프트 7 개)

| | 영상 인코더 | 검출 | 합 |
|---|---|---|---|
| PyTorch fp32 (참고) | 308 ms | 254 ms | 562 ms |
| 이 판 fp32 | 330 ms | 153 ms | 483 ms |
| **이 판 fp16** | **84 ms** | **37 ms** | **122 ms (8.2 장/초)** |

라벨 도구 전체(프롬프트 9 개, 검출된 프롬프트만 마스크): 6.7 장/초. GEMM 은 텐서코어 최대치의 약 90 % 로 돌고, 남은 시간의 큰 몫은 DETR 인코더 어텐션(프롬프트마다 5184 × 5184)이다.
