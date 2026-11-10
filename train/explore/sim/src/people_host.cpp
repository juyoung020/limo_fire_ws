// 움직이는 사람 호스트 쪽 (people.h): 설정 읽기, 맵마다 막힌 칸까지 거리 표, 판 시작 인자(사람 수 상한)·뷰어용 꺼내기
#include <yaml-cpp/yaml.h>

#include <algorithm>
#include <cmath>
#include <stdexcept>

#include "env_api.h"

namespace sim {

// load_people_cfg 는 src/env_cfg.cpp (설정 읽기는 CUDA 없이 — 실물 노드 limo_fire_rl 도 씀)

// 막힌 칸까지 거리: 갈 수 있는 바닥 칸에서 R 칸 창 안을 다 본다(사람 반지름 + 칸 대각 ≈ 0.37 m 만 가리면 되므로 창 0.6 m 면 충분,
// 창 안에 없으면 창 거리 + 1 칸). 맵 밖은 막힘으로
std::vector<uint8_t> people_clearance(const std::vector<uint8_t>& grid, int W, int H, float res) {
  std::vector<uint8_t> out((size_t)W * H, 0);
  const int R = (int)std::ceil(0.6f / res);
  for (int y = 0; y < H; ++y)
    for (int x = 0; x < W; ++x) {
      if (!(grid[(size_t)y * W + x] & kCellFloor)) continue;
      int best = (R + 1) * (R + 1);
      for (int dy = -R; dy <= R; ++dy)
        for (int dx = -R; dx <= R; ++dx) {
          const int d2 = dx * dx + dy * dy;
          if (d2 >= best) continue;
          const int u = x + dx, v = y + dy;
          if (u < 0 || v < 0 || u >= W || v >= H || (grid[(size_t)v * W + u] & kCellBlocked)) best = d2;
        }
      out[(size_t)y * W + x] = (uint8_t)std::min(255.0, std::floor(std::sqrt((double)best) * res * 100.0));
    }
  return out;
}

void Env::set_people_max(const std::vector<int>& k) {
  if ((int)k.size() != b_.n_env) throw std::runtime_error("set_people_max: 환경 수와 다름");
  std::vector<EnvState> st;
  get_state(st);
  for (int i = 0; i < b_.n_env; ++i) {
    if (k[i] < 0 || k[i] > kMaxPeople) throw std::runtime_error("set_people_max: 0 ~ kMaxPeople");
    st[i].people.k_max = k[i];
    set_state(i, st[i]);
  }
}

void Env::get_people(std::vector<People>& out) const {
  std::vector<EnvState> st;
  get_state(st);
  out.resize(st.size());
  for (size_t i = 0; i < st.size(); ++i) out[i] = st[i].people;
}

}  // namespace sim
