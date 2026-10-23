// YOLO26 E2E 탐지 손실(Ultralytics E2ELoss + v8DetectionLoss + TaskAlignedAssigner(STAL), reg_max = 1)을 GPU 에서.
//
// 가지 둘: one2many(topk 10), one2one(topk 7, topk2 1). 가지마다
//   배정: 정답마다 후보(앵커 중심이 정답 상자 안 — 변이 16 px 보다 작으면 16 px 로 넓힘) 중 정렬 지표
//         score^0.5 · CIoU^6 상위 k → 앵커가 여럿에 뽑히면 CIoU 가 가장 큰 정답 → (one2one) 정답마다 지표 최대 앵커 하나 →
//         목표 점수 = 지표 · (정답의 최대 CIoU / 정답의 최대 지표)
//   손실: cls = Σ BCE / Σ목표, box = Σ (1 − CIoU)·w / Σ목표, l1 = Σ mean|ltrb − 목표 ltrb|(영상 크기로 정규화)·w / Σ목표
//   이득 box 7.5 · cls 0.5 · l1 1.5, 배치 크기를 곱한다. 전체 = g_o2m·L_o2m + g_o2o·L_o2o.
// 기울기는 머리 출력(box ltrb, cls 로짓)에 대해, loss scale 을 곱해 act_t 로 쓴다(덮어쓰기).
#pragma once
#include "core.h"

struct LossCfg {
  int B = 16, nc = 7, imgsz = 640, G = 256;  // G: 영상당 정답 상한
  int lh[3], lw[3];
  int stride[3] = {8, 16, 32};
  float box_gain = 7.5f, cls_gain = 0.5f, l1_gain = 1.5f;
  int A() const { return lh[0] * lw[0] + lh[1] * lw[1] + lh[2] * lw[2]; }
};

// 정답(장치): n[b], box[b*G + g] = xyxy 픽셀(입력 영상 기준), cls[b*G + g]
struct GtDev {
  int* n;
  float4* box;
  int* cls;
};

// 머리(장치, NHWC act_t): box[l] [B, h, w, 4], cls[l] [B, h, w, nc] 와 기울기
struct HeadDev {
  const act_t* box[3];
  const act_t* cls[3];
  act_t* dbox[3];
  act_t* dcls[3];
};

class E2ELoss {
 public:
  void init(const LossCfg& c);
  // 가지 br(0 = one2many, 1 = one2one) 하나: 손실 항목(이득 곱함, 배치 곱 전)을 items + 3·br 에 쓰고, 기울기
  // (∂(weight·B·Σ항목)/∂출력 · scale) 를 dbox·dcls 에 쓴다. scale: 장치 float 하나(loss scale), null 이면 1.
  void run(int br, const HeadDev& h, const GtDev& gt, float weight, const float* scale, cudaStream_t s);
  // items: 장치 float [8] = o2m box, cls, l1, o2o box, cls, l1, 양성 수(o2o), Σ목표(o2o)
  float* items() const { return items_; }
  LossCfg cfg;
  // 디버그: 마지막 run 의 앵커별 배정 정답(−1 없음), 목표 점수
  int* assigned() const { return agt_; }
  float* target() const { return tsc_; }

 private:
  int A_ = 0;
  float4* pbox_ = nullptr;  // [B*A] 예측 상자(격자 단위 xyxy)
  int* topa_ = nullptr;     // [B*G*10] 정답별 상위 앵커
  int* cnt_ = nullptr;      // [B*A]
  int* who_ = nullptr;      // [B*A]
  int* agt_ = nullptr;      // [B*A]
  float* tsc_ = nullptr;    // [B*A]
  unsigned long long* gbest_ = nullptr;  // [B*G] one2one: (지표, 앵커) 최대
  float* galign_ = nullptr;  // [B*G] 양성 지표 최대
  float* govl_ = nullptr;    // [B*G] 양성 CIoU 최대
  double* sums_ = nullptr;   // [4]: Σ목표, box, l1, cls
  float* items_ = nullptr;
};
