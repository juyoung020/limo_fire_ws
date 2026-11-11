// explore_lite 프론티어 고르기의 공용 부분 — 기준선(Explorer::make_plan)과 실물·Webots ROS 노드(limo_fire_rl explore_frontier)가 같이 쓴다.
// mdp::frontier_search 가 낸 덩어리를 explore_lite 비용으로 줄 세우고, 블랙리스트를 explore_lite 규칙으로 본다. 나머지(목표 보내기·진척 시간)는 쓰는 쪽에서.
// 출처: m-explore(hrnr/m-explore) 712bdd41027b645c9c876a4c0071478f090825ea
//   explore/src/frontier_search.cpp:191-196  비용 = potential_scale × min_distance(m) × 해상도 − gain_scale × 크기(칸) × 해상도
//   explore/src/explore.cpp:249-260           goalOnBlacklist — x·y 가 blacklist_tolerance 칸 안이면 같은 목표
#pragma once
#include <algorithm>
#include <cmath>
#include <utility>
#include <vector>

#include "frontier.h"

namespace fb {

struct Ranked {
  float cost;   // explore_lite 비용 (작을수록 먼저)
  float cx, cy; // 덩어리 중심 (세계 좌표, m) = 목표
  float dist;   // 로봇 칸 → 덩어리에서 가장 가까운 칸 (m)
  int i;        // fr 안 번호
};

// fr[0 … n): 지도 원점 (ox, oy), 칸 res(m). 비용 오름차순(같으면 찾은 차례 — stable)
inline std::vector<Ranked> rank_frontiers(const mdp::Frontier* fr, int n, float res, float ox, float oy, float potential_scale, float gain_scale) {
  std::vector<Ranked> cs;
  cs.reserve((size_t)std::max(n, 0));
  for (int i = 0; i < n; ++i) {
    const mdp::Frontier& f = fr[i];
    const float dist = std::sqrt((float)f.min_d2) * res;
    const float cx = ox + ((float)f.sx / (float)f.n + 0.5f) * res, cy = oy + ((float)f.sy / (float)f.n + 0.5f) * res;
    cs.push_back({potential_scale * dist * res - gain_scale * (float)f.n * res, cx, cy, dist, i});
  }
  std::stable_sort(cs.begin(), cs.end(), [](const Ranked& a, const Ranked& b) { return a.cost < b.cost; });
  return cs;
}

// 블랙리스트: tol = blacklist_tolerance(칸) × 해상도 (m)
inline bool on_blacklist(const std::vector<std::pair<float, float>>& bl, float x, float y, float tol) {
  for (const auto& b : bl)
    if (std::fabs(x - b.first) < tol && std::fabs(y - b.second) < tol) return true;
  return false;
}

}  // namespace fb
