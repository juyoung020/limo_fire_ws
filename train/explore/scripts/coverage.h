// 탐사 덮음 (판단 모델 평가 · 갈래 라벨 공통 — 2026-11-17 사용자 기준): "지도를 다 칠했나"가 아니라 "공간을 대충이라도 빨리 알게 됐나".
//   가는 칸 덮음: 정답 바닥 칸(출발점에서 갈 수 있는 칸, World::grid kCellFloor) 가운데 로봇 지도에서 빈 곳으로 알게 된 몫 (둘째 지표).
//   굵은 칸 덮음: 굵은 칸(block × block 가는 칸 — 10 = 0.5 m, 20 = 1.0 m) 가운데, 그 안 정답 바닥 칸의 thr 이상을 알게 된 굵은 칸의 몫 (첫째 지표).
//     분모 = 정답 바닥 칸이 하나라도 있는 굵은 칸 수. 성기게 칠해도 그 자리 공간을 대충 알면 덮인 것으로 셈.
// 값 차례: [가는 칸, (block[0], thr[0]), (block[0], thr[1]), …, (block[1], thr[0]), …]
#pragma once
#include <cstdint>
#include <vector>

#include "env.h"
#include "frontier.h"
#include "world.h"

namespace covm {

class Coverage {
 public:
  std::vector<int> block{10, 20};
  std::vector<float> thr{0.1f, 0.2f, 0.3f, 0.5f};
  // 장면마다 한 번: 굵은 칸마다 정답 바닥 칸 수
  void init(const sim::World& w) {
    w_ = &w;
    floor_.assign(block.size(), {});
    den_.assign(block.size(), 0);
    for (size_t b = 0; b < block.size(); ++b) {
      const int B = block[b], cw = (w.gw + B - 1) / B, ch = (w.gh + B - 1) / B;
      floor_[b].assign((size_t)cw * ch, 0);
      for (int y = 0; y < w.gh; ++y)
        for (int x = 0; x < w.gw; ++x)
          if (w.grid[(size_t)y * w.gw + x] & sim::kCellFloor) floor_[b][(size_t)(y / B) * cw + x / B] += 1;
      for (int n : floor_[b]) den_[b] += n > 0;
    }
  }
  // lmap: 로봇 지도 [size²] (칸 (x, y) = y · size + x, 세계 칸 좌표와 같음)
  std::vector<float> eval(const int16_t* lmap, int size) const {
    const sim::World& w = *w_;
    std::vector<float> out;
    std::vector<std::vector<int>> kn(block.size());
    for (size_t b = 0; b < block.size(); ++b) kn[b].assign(floor_[b].size(), 0);
    long fine = 0;
    for (int y = 0; y < w.gh; ++y)
      for (int x = 0; x < w.gw; ++x) {
        if (!(w.grid[(size_t)y * w.gw + x] & sim::kCellFloor)) continue;
        const int16_t v = lmap[(size_t)y * size + x];
        if (v == sim::kUnknown || v >= mdp::kOccL) continue;
        ++fine;
        for (size_t b = 0; b < block.size(); ++b) {
          const int B = block[b], cw = (w.gw + B - 1) / B;
          kn[b][(size_t)(y / B) * cw + x / B] += 1;
        }
      }
    out.push_back(w.n_floor > 0 ? (float)fine / (float)w.n_floor : 0.f);
    for (size_t b = 0; b < block.size(); ++b)
      for (float t : thr) {
        long c = 0;
        for (size_t k = 0; k < floor_[b].size(); ++k) c += floor_[b][k] > 0 && kn[b][k] >= t * floor_[b][k];
        out.push_back(den_[b] > 0 ? (float)c / (float)den_[b] : 0.f);
      }
    return out;
  }
  int n_values() const { return 1 + (int)(block.size() * thr.size()); }

 private:
  const sim::World* w_ = nullptr;
  std::vector<std::vector<int>> floor_;
  std::vector<long> den_;
};

}  // namespace covm
