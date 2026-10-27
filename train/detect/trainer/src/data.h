// 학습 데이터 공급: 캐시(고정 메모리 JPEG) → CPU 스레드(JPEG 풀기·증강 난수·라벨 변환) → 고정 메모리 슬롯 →
// 복사 스트림으로 비동기 H2D → GPU 증강 커널(augment.cu: Mosaic + affine 양선형 + HSV + 좌우 뒤집기).
//
// Ultralytics v8_transforms(Mosaic(p) → RandomPerspective(degrees 0, translate, scale) → RandomHSV → RandomFlip) 와
// 같은 규칙. 다른 점: Mosaic 의 나머지 세 장을 최근 버퍼가 아니라 전체에서 고른다.
// 마스크(캐시에 있으면): 원본 번호 지도(288×288)를 영상과 같은 변환으로 따라가 출력 S/4 격자(화소 4×4 의 가운데)에서
// 최근접으로 읽는다 → 출력 정답 번호 + 1(int16, 0 배경). Ultralytics 는 다각형을 변환해 출력에 칠한 뒤 1/4 로 줄인다.
#pragma once
#include <atomic>
#include <condition_variable>
#include <mutex>
#include <thread>
#include <vector>

#include "cache.h"
#include "core.h"

struct AugCfg {
  int S = 640, batch = 16, G = 256;
  float mosaic = 1.f, translate = 0.1f, scale = 0.5f, fliplr = 0.5f;
  float hsv_h = 0.015f, hsv_s = 0.7f, hsv_v = 0.4f;
  bool augment = true;  // false: 검증(가운데 맞춘 letterbox, 증강 없음)
  int epochs = 1, close_mosaic = 10;  // 마지막 close_mosaic epoch 은 Mosaic 끔
  int in_c = 8;                      // 출력 채널(RGB + 0 덧대기)
  bool masks = false;                // 분할 정답도 만든다(캐시에 마스크가 있어야 함)
  int mask_ratio = 4;
  uint64_t seed = 0;
};

// 출력 영상 하나의 증강 매개변수(장치로 보낸다)
struct SampleParam {
  int ntile;                // 1(한 장) 또는 4(Mosaic)
  int cw, ch;               // 캔버스 크기
  int64_t off[4];           // 슬롯 영상 덩어리 안 원본 위치
  int sw[4], sh[4];         // 원본 크기
  int ta[4][4];             // 캔버스 위 타일 x1a, y1a, x2a, y2a
  int pad[4][2];            // 캔버스 = 원본 + pad
  float Mi[6];              // 출력 → 캔버스 역변환(2×3)
  float hsv[3];             // 이득(h, s, v), 1 이면 그대로
  int flip;
  int64_t moff[4];          // 슬롯 마스크 덩어리 안 번호 지도 위치
  int msz[4];               // 번호 지도 크기(정사각, 사진 전체를 덮음)
  int16_t lmap[4][256];     // 타일 t 의 번호 j(1..) − 1 → 출력 정답 번호(−1 버림)
};

struct Slot {
  int64_t batch = -1;
  uint8_t* img = nullptr;       // 고정 메모리: 원본 RGB 모음
  int64_t img_cap = 0, img_used = 0;
  SampleParam* sp = nullptr;    // [B]
  int* gt_n = nullptr;          // [B]
  float4* gt_box = nullptr;     // [B*G] 출력 영상 픽셀 xyxy
  int* gt_cls = nullptr;        // [B*G]
  int* idx = nullptr;           // [B] 캐시 번호(검증용)
  uint8_t* msk = nullptr;       // 고정 메모리: 번호 지도 모음
  int64_t msk_cap = 0, msk_used = 0;
  int state = 0;                // 0 빔, 1 채우는 중, 2 준비됨, 3 쓰는 중
};

class Loader {
 public:
  void open(const Cache* cache, const AugCfg& c, int workers, int slots);
  ~Loader();
  int64_t batches_per_epoch() const;
  // 다음 배치(전역 차례: epoch·배치 순서는 배치 번호로 정해짐)를 장치 버퍼에 올리고 증강 커널을 돌려
  // input(NHWC act_t, 0..1)을 만든다. 정답은 장치 두 벌 중 하나 — 다음다음 호출 전까지 유효.
  // gt_mask(null 아니면): 장치 [B, S/r, S/r] 정답 번호 + 1 (cfg.masks 일 때)
  void next(act_t* input, int** gt_n, float4** gt_box, int** gt_cls, int* idx_host, cudaStream_t compute,
            int16_t** gt_mask = nullptr);
  // 이어 하기: 이 배치 번호부터(일꾼이 아직 아무것도 채우지 않았을 때만)
  void skip(int64_t first_batch);
  AugCfg cfg;
  double wait_ms = 0;  // 학습 스레드가 데이터를 기다린 시간

 private:
  const Cache* cache_ = nullptr;
  std::vector<Slot> slots_;
  std::vector<std::thread> th_;
  std::mutex mu_;
  std::condition_variable cv_;
  bool stop_ = false;
  int64_t nb_ = 0, next_fill_ = 0, next_take_ = 0;
  int pending_ = -1;  // 복사 끝나면 돌려줄 슬롯
  // 장치 쪽 두 벌(번갈아)
  uint8_t* d_img_[2] = {};
  SampleParam* d_sp_[2] = {};
  int* d_n_[2] = {};
  float4* d_box_[2] = {};
  int* d_cls_[2] = {};
  uint8_t* d_msk_[2] = {};
  int16_t* d_gm_[2] = {};
  int cur_ = 0, prev_ = -1;
  cudaStream_t copy_ = nullptr;
  cudaEvent_t ev_copied_[2] = {}, ev_used_[2] = {};
  int64_t img_cap_ = 0, msk_cap_ = 0;
  void worker();
  void fill(Slot& s, int64_t b);
};

// GPU 증강(augment.cu)
void augment_batch(const uint8_t* img, const SampleParam* sp, int B, int S, int C, act_t* out, cudaStream_t s);
// 마스크 정답: out [B, S/r, S/r]
void mask_targets(const uint8_t* msk, const SampleParam* sp, int B, int S, int r, int16_t* out, cudaStream_t s);
