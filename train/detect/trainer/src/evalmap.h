// 검증 mAP (Ultralytics DetectionValidator 와 같은 규칙, NMS 없음):
//  one2one 머리 → 앵커별 클래스 점수 최댓값 상위 300 앵커 → 그 300×nc 점수 중 상위 300 → conf > 0.001
//  → IoU 0.50:0.05:0.95 마다 클래스 같은 정답과 IoU 큰 순 1:1 짝 → 클래스별 AP(101 점 보간) → 평균.
#pragma once
#include <vector>

struct Det {
  float x1, y1, x2, y2, conf;
  int cls;
};
struct GtBox {
  float x1, y1, x2, y2;
  int cls;
};

class MapEval {
 public:
  explicit MapEval(int nc) : nc_(nc) {}
  // 한 영상: 예측(제한 300)과 정답
  void add(const std::vector<Det>& pred, const std::vector<GtBox>& gt);
  struct Result {
    double map50 = 0, map = 0, mp = 0, mr = 0;
    std::vector<double> ap50, ap;  // 클래스별(정답 없는 클래스는 −1)
    std::vector<int> ngt;
  };
  Result compute() const;

 private:
  int nc_;
  struct P {
    float conf;
    int cls;
    unsigned tp;  // 10 비트: IoU 문턱별 맞음
  };
  std::vector<P> preds_;
  std::vector<int> ngt_ = std::vector<int>(64, 0);
};

// 머리 출력(호스트, float)에서 영상 하나의 예측: box[l] [h,w,4] ltrb(격자), cls[l] [h,w,nc] 로짓
std::vector<Det> decode_o2o(const float* const box[3], const float* const cls[3], const int h[3], const int w[3],
                            const int stride[3], int nc, int max_det = 300, float conf = 0.001f);
