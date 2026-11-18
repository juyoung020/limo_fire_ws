// 설명 글 ↔ 물체 사진 점수 (SigLIP 시그모이드 꼴) — 로봇 노드(object_search)와 시뮬 하네스가 같이 쓰는 한 곳.
// ROS·TensorRT·OpenCV 에 의존하지 않는다 (표준 라이브러리만, 함수는 모두 inline).
//   cos   = 글 벡터 · 사진 벡터 (둘 다 L2 단위 벡터라 내적 = cos)
//   prob  = sigmoid(logit_scale · cos + logit_bias)   (SigLIP 학습 손실과 같은 꼴 — 0~1 확률)
//   물체 점수 = 그 물체 사진들 prob 의 최댓값 (여러 방향 사진 평균은 오히려 나빠진다는 보고 — README "설명 글로 물체 찾기")
// 사진이 아직 없는 물체는 kNoViews (확률 범위 밖 — 받는 쪽이 "모름" 으로 가름)
#pragma once

#include <cmath>
#include <cstddef>
#include <vector>

namespace limo_fire_perception
{
namespace sem
{

constexpr double kNoViews = -1.0;

inline double dot(const float * a, const float * b, std::size_t n)
{
  double s = 0;
  for (std::size_t i = 0; i < n; ++i) {s += static_cast<double>(a[i]) * b[i];}
  return s;
}

inline double prob(double cos, double logit_scale, double logit_bias)
{
  return 1.0 / (1.0 + std::exp(-(logit_scale * cos + logit_bias)));
}

// 물체 사진 벡터들(views)과 글 벡터(text) — 사진마다 prob 의 최댓값. 사진이 없거나 차원이 안 맞으면 kNoViews
inline double object_prob(const std::vector<std::vector<float>> & views, const std::vector<float> & text,
  double logit_scale, double logit_bias)
{
  double best = kNoViews;
  for (const auto & v : views) {
    if (v.size() != text.size() || v.empty()) {continue;}
    const double p = prob(dot(v.data(), text.data(), v.size()), logit_scale, logit_bias);
    if (p > best) {best = p;}
  }
  return best;
}

}  // namespace sem
}  // namespace limo_fire_perception
