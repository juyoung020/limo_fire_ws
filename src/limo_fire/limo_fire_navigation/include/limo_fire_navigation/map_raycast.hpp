// 저장 지도(점유 격자)에서 광선을 쏘아 벽까지 거리를 잰다 — 연기 모드의 가상 스캔.
// ROS 의존 없음(테스트용). 격자 값은 nav_msgs/OccupancyGrid 규약: 0~100 점유 확률, −1 모름.
#pragma once

#include <cmath>
#include <cstdint>
#include <vector>

namespace limo_fire_navigation {

struct Grid {
  int width = 0, height = 0;
  double resolution = 0.05;       // m/칸
  double origin_x = 0.0, origin_y = 0.0, origin_yaw = 0.0;  // 칸 (0,0) 의 모서리, 지도 좌표
  std::vector<int8_t> data;       // 행 우선, data[y * width + x]
  int occupied_threshold = 65;    // 이 값 이상이면 벽
};

// 지도 좌표 (x, y) 가 벽 칸이면 true. 지도 밖·모름은 벽이 아니다(통과).
inline bool occupied(const Grid& g, double x, double y) {
  const double c = std::cos(-g.origin_yaw), s = std::sin(-g.origin_yaw);
  const double dx = x - g.origin_x, dy = y - g.origin_y;
  const double lx = c * dx - s * dy, ly = s * dx + c * dy;
  const int ix = static_cast<int>(std::floor(lx / g.resolution));
  const int iy = static_cast<int>(std::floor(ly / g.resolution));
  if (ix < 0 || iy < 0 || ix >= g.width || iy >= g.height) return false;
  const int8_t v = g.data[static_cast<size_t>(iy) * g.width + ix];
  return v >= g.occupied_threshold;
}

// (x, y) 에서 방향 yaw 로 range_max 까지. 벽에 닿으면 그 거리, 안 닿으면 범위 밖 값 no_hit.
// 칸 크기의 절반 간격으로 걸어서 대각선으로 한 칸을 건너뛰지 않는다.
inline double cast(const Grid& g, double x, double y, double yaw, double range_min, double range_max,
                   double no_hit) {
  const double step = 0.5 * g.resolution;
  const double ux = std::cos(yaw), uy = std::sin(yaw);
  for (double r = range_min; r <= range_max; r += step) {
    if (occupied(g, x + r * ux, y + r * uy)) return r;
  }
  return no_hit;
}

}  // namespace limo_fire_navigation
