#include "evalmap.h"

#include <algorithm>
#include <cmath>
#include <numeric>

namespace {
float iou(const Det& a, const GtBox& b) {
  float iw = std::max(0.f, std::min(a.x2, b.x2) - std::max(a.x1, b.x1));
  float ih = std::max(0.f, std::min(a.y2, b.y2) - std::max(a.y1, b.y1));
  float in = iw * ih;
  float u = (a.x2 - a.x1) * (a.y2 - a.y1) + (b.x2 - b.x1) * (b.y2 - b.y1) - in;
  return in / (u + 1e-7f);
}
// Ultralytics compute_ap: 101 점 보간
double compute_ap(const std::vector<double>& rec, const std::vector<double>& pre) {
  std::vector<double> mrec = {0.0}, mpre = {1.0};
  mrec.insert(mrec.end(), rec.begin(), rec.end());
  mpre.insert(mpre.end(), pre.begin(), pre.end());
  mrec.push_back(1.0);
  mpre.push_back(0.0);
  for (int i = (int)mpre.size() - 2; i >= 0; --i) mpre[i] = std::max(mpre[i], mpre[i + 1]);
  // np.interp(x, mrec, mpre) 후 사다리꼴 적분
  double ap = 0, prev_x = 0, prev_y = 0;
  for (int k = 0; k <= 100; ++k) {
    double x = k / 100.0, y;
    auto it = std::upper_bound(mrec.begin(), mrec.end(), x);
    if (it == mrec.begin())
      y = mpre.front();
    else if (it == mrec.end())
      y = mpre.back();
    else {
      size_t j = it - mrec.begin();
      double x0 = mrec[j - 1], x1 = mrec[j];
      y = x1 > x0 ? mpre[j - 1] + (mpre[j] - mpre[j - 1]) * (x - x0) / (x1 - x0) : mpre[j];
    }
    if (k > 0) ap += (x - prev_x) * (y + prev_y) / 2;
    prev_x = x;
    prev_y = y;
  }
  return ap;
}
}  // namespace

void MapEval::add(const std::vector<Det>& pred, const std::vector<GtBox>& gt) {
  for (auto& g : gt) ngt_[g.cls]++;
  std::vector<unsigned> tp(pred.size(), 0);
  if (!pred.empty() && !gt.empty()) {
    // (짝 후보) IoU ≥ 문턱이고 클래스 같음 → IoU 큰 순, 예측·정답 각각 한 번만
    struct M {
      float iou;
      int p, g;
    };
    std::vector<M> all;
    for (int p = 0; p < (int)pred.size(); ++p)
      for (int g = 0; g < (int)gt.size(); ++g)
        if (pred[p].cls == gt[g].cls) {
          float v = iou(pred[p], gt[g]);
          if (v >= 0.5f) all.push_back({v, p, g});
        }
    // Ultralytics match_predictions: IoU 큰 순 → 예측마다 가장 큰 것 하나 → (예측 번호 순서에서) 정답마다 처음 것
    std::stable_sort(all.begin(), all.end(), [](const M& a, const M& b) { return a.iou > b.iou; });
    for (int t = 0; t < 10; ++t) {
      float thr = 0.5f + 0.05f * t;
      std::vector<int> best_g(pred.size(), -1);
      for (auto& m : all) {
        if (m.iou < thr) continue;
        if (best_g[m.p] < 0) best_g[m.p] = m.g;
      }
      std::vector<char> ug(gt.size(), 0);
      for (size_t p = 0; p < pred.size(); ++p) {
        int g = best_g[p];
        if (g < 0 || ug[g]) continue;
        ug[g] = 1;
        tp[p] |= 1u << t;
      }
    }
  }
  for (size_t i = 0; i < pred.size(); ++i) preds_.push_back({pred[i].conf, pred[i].cls, tp[i]});
}

MapEval::Result MapEval::compute() const {
  Result r;
  r.ap50.assign(nc_, -1);
  r.ap.assign(nc_, -1);
  r.ngt.assign(ngt_.begin(), ngt_.begin() + nc_);
  std::vector<P> ps = preds_;
  std::stable_sort(ps.begin(), ps.end(), [](const P& a, const P& b) { return a.conf > b.conf; });
  int nclass = 0;
  double s50 = 0, s = 0, sp = 0, sr = 0;
  for (int c = 0; c < nc_; ++c) {
    if (ngt_[c] == 0) continue;
    ++nclass;
    std::vector<int> idx;
    for (int i = 0; i < (int)ps.size(); ++i)
      if (ps[i].cls == c) idx.push_back(i);
    double apsum = 0;
    for (int t = 0; t < 10; ++t) {
      std::vector<double> rec, pre;
      double tpc = 0, fpc = 0;
      for (int i : idx) {
        bool hit = ps[i].tp >> t & 1u;
        tpc += hit;
        fpc += !hit;
        rec.push_back(tpc / (ngt_[c] + 1e-16));
        pre.push_back(tpc / (tpc + fpc));
      }
      double ap = idx.empty() ? 0.0 : compute_ap(rec, pre);
      if (t == 0) r.ap50[c] = ap;
      apsum += ap;
    }
    r.ap[c] = apsum / 10;
    s50 += r.ap50[c];
    s += r.ap[c];
  }
  if (nclass) {
    r.map50 = s50 / nclass;
    r.map = s / nclass;
  }
  (void)sp;
  (void)sr;
  return r;
}

std::vector<Det> decode_o2o(const float* const box[3], const float* const cls[3], const int h[3], const int w[3],
                            const int stride[3], int nc, int max_det, float conf) {
  struct A {
    float s;
    int l, k;
  };
  std::vector<A> an;
  for (int l = 0; l < 3; ++l)
    for (int k = 0; k < h[l] * w[l]; ++k) {
      float m = -1e30f;
      for (int c = 0; c < nc; ++c) m = std::max(m, cls[l][(size_t)k * nc + c]);
      an.push_back({m, l, k});
    }
  int K = std::min<int>(max_det, (int)an.size());
  std::partial_sort(an.begin(), an.begin() + K, an.end(), [](const A& a, const A& b) { return a.s > b.s; });
  struct C {
    float s;
    int a, c;
  };
  std::vector<C> cs;
  for (int i = 0; i < K; ++i)
    for (int c = 0; c < nc; ++c) cs.push_back({cls[an[i].l][(size_t)an[i].k * nc + c], i, c});
  int K2 = std::min<int>(K, (int)cs.size());
  std::partial_sort(cs.begin(), cs.begin() + K2, cs.end(), [](const C& a, const C& b) { return a.s > b.s; });
  std::vector<Det> out;
  for (int i = 0; i < K2; ++i) {
    float p = 1.f / (1.f + std::exp(-cs[i].s));
    if (p <= conf) continue;
    const A& a = an[cs[i].a];
    int ix = a.k % w[a.l], iy = a.k / w[a.l];
    const float* b = box[a.l] + (size_t)a.k * 4;
    float s = (float)stride[a.l], ax = ix + 0.5f, ay = iy + 0.5f;
    out.push_back({(ax - b[0]) * s, (ay - b[1]) * s, (ax + b[2]) * s, (ay + b[3]) * s, p, cs[i].c});
  }
  return out;
}
