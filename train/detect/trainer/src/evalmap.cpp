#include "evalmap.h"

#include <algorithm>
#include <cmath>
#include <numeric>

// 짝짓기·AP 는 인식 벤치마크와 같이 쓰는 det_metrics.hpp (limo_fire_perception) 에 있다
void MapEval::add(const std::vector<Det>& pred, const std::vector<GtBox>& gt) {
  std::vector<int> gc;
  std::vector<float> io(pred.size() * gt.size());
  for (auto& g : gt) gc.push_back(g.cls);
  for (size_t p = 0; p < pred.size(); ++p)
    for (size_t g = 0; g < gt.size(); ++g)
      io[p * gt.size() + g] = det_metrics::box_iou({pred[p].x1, pred[p].y1, pred[p].x2, pred[p].y2},
                                                   {gt[g].x1, gt[g].y1, gt[g].x2, gt[g].y2});
  add_iou(pred, gc, io);
}

void MapEval::add_iou(const std::vector<Det>& pred, const std::vector<int>& gt, const std::vector<float>& io) {
  std::vector<det_metrics::Pred> ps;
  ps.reserve(pred.size());
  for (auto& d : pred) ps.push_back({d.conf, d.cls});
  ev_.add(ps, gt, io);
}

MapEval::Result MapEval::compute() const {
  const auto m = ev_.compute();
  Result r;
  r.map50 = m.map50;
  r.map = m.map;
  for (const auto& c : m.cls) {
    r.ap50.push_back(c.ap50);
    r.ap.push_back(c.ap);
    r.ngt.push_back(c.ngt);
  }
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
    out.push_back({(ax - b[0]) * s, (ay - b[1]) * s, (ax + b[2]) * s, (ay + b[3]) * s, p, cs[i].c, a.l, a.k});
  }
  return out;
}
