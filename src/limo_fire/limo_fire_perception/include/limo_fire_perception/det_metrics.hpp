// 탐지·분할 평가 지표 (ROS·CUDA 안 씀, 헤더만). Ultralytics DetectionValidator 와 같은 규칙:
//   IoU 0.50:0.05:0.95 마다 클래스 같은 정답과 IoU 큰 순으로 1:1 짝 → 클래스별 AP(101 점 보간) → 정답 있는 클래스 평균
// 학습기(train/detect/trainer evalmap)와 인식 벤치마크(perception_benchmark)가 같이 쓴다.
// IoU 는 바깥에서 재서 넘긴다(상자 IoU 는 box_iou, 마스크 IoU 는 픽셀로) — 같은 짝짓기·AP 로 상자·마스크를 다 잰다
#pragma once

#include <algorithm>
#include <cstddef>
#include <vector>

namespace det_metrics
{

struct Box
{
  float x1, y1, x2, y2;
};

inline float box_iou(const Box & a, const Box & b)
{
  const float iw = std::max(0.f, std::min(a.x2, b.x2) - std::max(a.x1, b.x1));
  const float ih = std::max(0.f, std::min(a.y2, b.y2) - std::max(a.y1, b.y1));
  const float in = iw * ih;
  const float u = (a.x2 - a.x1) * (a.y2 - a.y1) + (b.x2 - b.x1) * (b.y2 - b.y1) - in;
  return in / (u + 1e-7f);
}

// Ultralytics compute_ap: 정밀도 포락선 → 0..1 을 101 점으로 보간해 사다리꼴 적분
inline double compute_ap(const std::vector<double> & rec, const std::vector<double> & pre)
{
  std::vector<double> mrec = {0.0}, mpre = {1.0};
  mrec.insert(mrec.end(), rec.begin(), rec.end());
  mpre.insert(mpre.end(), pre.begin(), pre.end());
  mrec.push_back(1.0);
  mpre.push_back(0.0);
  for (int i = static_cast<int>(mpre.size()) - 2; i >= 0; --i) {mpre[i] = std::max(mpre[i], mpre[i + 1]);}
  double ap = 0, prev_x = 0, prev_y = 0;
  for (int k = 0; k <= 100; ++k) {
    const double x = k / 100.0;
    double y;
    const auto it = std::upper_bound(mrec.begin(), mrec.end(), x);
    if (it == mrec.begin()) {
      y = mpre.front();
    } else if (it == mrec.end()) {
      y = mpre.back();
    } else {
      const size_t j = it - mrec.begin();
      const double x0 = mrec[j - 1], x1 = mrec[j];
      y = x1 > x0 ? mpre[j - 1] + (mpre[j] - mpre[j - 1]) * (x - x0) / (x1 - x0) : mpre[j];
    }
    if (k > 0) {ap += (x - prev_x) * (y + prev_y) / 2;}
    prev_x = x;
    prev_y = y;
  }
  return ap;
}

struct Pred
{
  float conf;
  int cls;
};

class ApEval
{
public:
  explicit ApEval(int nc)
  : nc_(nc), ngt_(nc, 0), iou_sum_(nc, 0.0), matched_(nc, 0) {}

  // 한 영상: 예측, 정답 클래스, IoU 행렬 iou[p * ngt + g]
  void add(const std::vector<Pred> & pred, const std::vector<int> & gt, const std::vector<float> & iou)
  {
    for (int c : gt) {
      if (c >= 0 && c < nc_) {ngt_[c]++;}
    }
    std::vector<unsigned> tp(pred.size(), 0);
    std::vector<float> iou50(pred.size(), 0.f);
    if (!pred.empty() && !gt.empty()) {
      struct M {float iou; int p, g;};
      std::vector<M> all;
      for (int p = 0; p < static_cast<int>(pred.size()); ++p) {
        for (int g = 0; g < static_cast<int>(gt.size()); ++g) {
          if (pred[p].cls != gt[g]) {continue;}
          const float v = iou[static_cast<size_t>(p) * gt.size() + g];
          if (v >= 0.5f) {all.push_back({v, p, g});}
        }
      }
      // Ultralytics match_predictions: IoU 큰 순 → 예측마다 가장 큰 것 하나 → (예측 번호 순서에서) 정답마다 처음 것
      std::stable_sort(all.begin(), all.end(), [](const M & a, const M & b) {return a.iou > b.iou;});
      for (int t = 0; t < 10; ++t) {
        const float thr = 0.5f + 0.05f * t;
        std::vector<int> best_g(pred.size(), -1);
        std::vector<float> best_v(pred.size(), 0.f);
        for (const auto & m : all) {
          if (m.iou < thr || best_g[m.p] >= 0) {continue;}
          best_g[m.p] = m.g;
          best_v[m.p] = m.iou;
        }
        std::vector<char> used(gt.size(), 0);
        for (size_t p = 0; p < pred.size(); ++p) {
          const int g = best_g[p];
          if (g < 0 || used[g]) {continue;}
          used[g] = 1;
          tp[p] |= 1u << t;
          if (t == 0) {iou50[p] = best_v[p];}
        }
      }
    }
    for (size_t i = 0; i < pred.size(); ++i) {
      preds_.push_back({pred[i].conf, pred[i].cls, tp[i]});
      if ((tp[i] & 1u) && pred[i].cls >= 0 && pred[i].cls < nc_) {
        iou_sum_[pred[i].cls] += iou50[i];
        matched_[pred[i].cls]++;
      }
    }
  }

  struct ClassStat
  {
    int ngt = 0;
    int npred = 0;
    int tp = 0;            // IoU ≥ 0.5 로 맞은 예측 (넣은 예측 전부 기준 — 운용 문턱은 걸러서 넣는다)
    double ap50 = -1;      // 정답이 없으면 −1
    double ap = -1;        // AP50-95
    double mean_iou = 0;   // 맞은 것들의 IoU 평균
  };
  struct Result
  {
    std::vector<ClassStat> cls;
    double map50 = 0, map = 0;
    int ngt = 0, npred = 0, tp = 0;
    double mean_iou = 0;
    double precision() const {return npred ? static_cast<double>(tp) / npred : 0.0;}
    double recall() const {return ngt ? static_cast<double>(tp) / ngt : 0.0;}
    double f1() const
    {
      const double p = precision(), r = recall();
      return p + r > 0 ? 2 * p * r / (p + r) : 0.0;
    }
  };

  // only: 지표를 낼 클래스(비우면 전부). 마스크 정답이 부정확한 클래스를 뺄 때 쓴다
  Result compute(const std::vector<bool> & only = {}) const
  {
    Result r;
    r.cls.resize(nc_);
    std::vector<P> ps = preds_;
    std::stable_sort(ps.begin(), ps.end(), [](const P & a, const P & b) {return a.conf > b.conf;});
    int nclass = 0;
    double iou_all = 0;
    int matched_all = 0;
    for (int c = 0; c < nc_; ++c) {
      if (!only.empty() && !only[c]) {continue;}
      ClassStat & s = r.cls[c];
      s.ngt = ngt_[c];
      std::vector<int> idx;
      for (int i = 0; i < static_cast<int>(ps.size()); ++i) {
        if (ps[i].cls == c) {idx.push_back(i);}
      }
      s.npred = static_cast<int>(idx.size());
      for (int i : idx) {s.tp += ps[i].tp & 1u;}
      s.mean_iou = matched_[c] ? iou_sum_[c] / matched_[c] : 0.0;
      r.ngt += s.ngt;
      r.npred += s.npred;
      r.tp += s.tp;
      iou_all += iou_sum_[c];
      matched_all += matched_[c];
      if (s.ngt == 0) {continue;}
      ++nclass;
      double apsum = 0;
      for (int t = 0; t < 10; ++t) {
        std::vector<double> rec, pre;
        double tpc = 0, fpc = 0;
        for (int i : idx) {
          const bool hit = ps[i].tp >> t & 1u;
          tpc += hit;
          fpc += !hit;
          rec.push_back(tpc / (s.ngt + 1e-16));
          pre.push_back(tpc / (tpc + fpc));
        }
        const double ap = idx.empty() ? 0.0 : compute_ap(rec, pre);
        if (t == 0) {s.ap50 = ap;}
        apsum += ap;
      }
      s.ap = apsum / 10;
      r.map50 += s.ap50;
      r.map += s.ap;
    }
    if (nclass) {
      r.map50 /= nclass;
      r.map /= nclass;
    }
    r.mean_iou = matched_all ? iou_all / matched_all : 0.0;
    return r;
  }

private:
  struct P
  {
    float conf;
    int cls;
    unsigned tp;  // 10 비트: IoU 문턱(0.50, 0.55 … 0.95)별 맞음
  };
  int nc_;
  std::vector<P> preds_;
  std::vector<int> ngt_;
  std::vector<double> iou_sum_;
  std::vector<int> matched_;
};

}  // namespace det_metrics
