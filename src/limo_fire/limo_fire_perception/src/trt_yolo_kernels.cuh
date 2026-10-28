// TrtYolo 안에서만 쓰는 GPU 커널 (전처리·end2end 해석·마스크 칠하기)
#pragma once

#include <cuda_runtime_api.h>

#include <cstdint>

namespace limo_fire_perception::gpu
{

// 마스크 영상 값이 1 바이트(물체 번호 + 1)라 한 장에 물체는 255 개까지
constexpr int kMaxDets = 255;

// 원본 이미지 ↔ 엔진 입력 좌표 (Ultralytics LetterBox 와 같은 규칙)
struct Letterbox
{
  float scale;     // 원본 → 입력 배율
  float pad_x;     // 입력 좌표에서 덧댄 폭 (실수, 상자 되돌릴 때)
  float pad_y;
  int img_w, img_h;
  int in_w, in_h;
  int nw, nh;      // 줄인 이미지 크기
  int left, top;   // 줄인 이미지를 놓는 정수 위치
};

struct Det
{
  float x1, y1, x2, y2;  // 원본 픽셀
  float score;
  int cls;
  int rx, ry, rw, rh;    // 마스크가 있을 수 있는 정수 픽셀 범위 (x1 <= u < x2, 이미지 안)
  int row;               // 엔진 출력의 몇 번째 줄인지 (마스크 계수 위치)
};

// 한 번의 D2H 로 가져오도록 개수와 목록을 붙여 둔다
struct DetBlock
{
  int count;
  Det dets[kMaxDets];
};

// 원본 RGB(uint8, 빈틈 없이 W×H×3) → 엔진 입력. nchw_float 이면 [1,3,H,W] float /255, 아니면 [1,H,W,3] uint8
void letterbox(const uint8_t * img, void * dst, bool nchw_float, const Letterbox & lb, cudaStream_t st);

// end2end 출력 [rows, cols] (x1 y1 x2 y2 score cls [계수…]) → 문턱 넘은 줄만 순서대로 DetBlock 에
void decode_end2end(const float * out, int rows, int cols, float conf, const Letterbox & lb,
  DetBlock * block, cudaStream_t st);

// 물체 번호 영상 (img_h × img_w, 0 = 배경, k+1 = block->dets[k]). 겹치면 상자가 작은 물체가 위
void paint_masks(const float * out, int cols, const float * proto, int nm, int ph, int pw,
  const DetBlock * block, const Letterbox & lb, uint8_t * ids, cudaStream_t st);

}  // namespace limo_fire_perception::gpu
