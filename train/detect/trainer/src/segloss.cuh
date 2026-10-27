// YOLO26-seg 손실의 분할 몫(Ultralytics v8SegmentationLoss, overlap_mask, mask_ratio 4)을 GPU 에서. 탐지 몫(E2ELoss)이
// 같은 가지에서 막 끝낸 배정(앵커별 정답 번호)을 그대로 쓴다.
//
//   seg: 양성 앵커 i 마다 P_i = 계수_i · 원형 마스크(그 영상), 정답 M_i = (번호 지도 == 배정 정답 + 1),
//        Σ_i mean_{H·W}(BCE(P_i, M_i) 를 정답 상자 안만) / 상자 넓이(0..1) / 양성 수, 이득 box(7.5)
//   sem: 의미 분할 로짓 [B, S/8, S/8, nc] 와 번호 지도를 최근접으로 줄인 클래스 one-hot(빈 곳 0):
//        0.5·BCE 평균 + 0.5·(1 − Dice(smooth 1)) 의 (영상, 클래스) 평균, 이득 box. 양성이 없으면 seg·sem 둘 다 0.
//   one2one 가지는 원형 마스크·의미 분할을 떼어 낸다(Ultralytics proto.detach()) — 계수 기울기만.
// 기울기는 머리 출력(계수, 원형 마스크, 의미 분할)에 loss scale 을 곱해 act_t 로 덮어쓴다.
#pragma once
#include "loss.cuh"

struct SegHeadDev {
  const act_t* coef[3];  // [B, h, w, nm]
  act_t* dcoef[3];
  const act_t* proto;    // [B, mh, mw, nm]
  act_t* dproto;         // null: 원형 마스크로 기울기를 보내지 않음(one2one)
  const act_t* sem;      // [B, sh, sw, nc]
  act_t* dsem;           // null: 값만
};

class SegLoss {
 public:
  void init(const LossCfg& c, int nm = 32);
  // 가지 br: E2ELoss::run(br) 바로 뒤에 그 assigned() 로. gt_mask: [B, mh, mw] 정답 번호 + 1(0 배경, int16)
  void run(int br, const SegHeadDev& h, const GtDev& gt, const int16_t* gt_mask, const int* agt, const float* weight,
           const float* scale, cudaStream_t s);
  // items: 장치 float [4] = o2m seg, sem, o2o seg, sem (이득 곱함, 배치 곱 전)
  float* items() const { return items_; }
  LossCfg cfg;
  int nm = 32, mh = 0, mw = 0, sh = 0, sw = 0;

 private:
  int A_ = 0;
  int* list_ = nullptr;     // [B·A] 영상별 양성 앵커(앵커 차례)
  int* cnt_ = nullptr;      // [B]
  float* iloss_ = nullptr;  // [B·A] 양성별 mean BCE / 넓이
  float* ikf_ = nullptr;    // [B·A] 양성별 기울기 계수
  double* semacc_ = nullptr;  // [B·nc·3 + B]: Σσ·T, Σσ, ΣT, 영상별 Σ BCE
  float* items_ = nullptr;
};
