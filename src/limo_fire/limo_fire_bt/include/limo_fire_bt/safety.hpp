#pragma once
#include <cmath>
#include <stdexcept>
#include <vector>
namespace limo_fire_bt {
enum class Size { Unknown, Small, Large };
inline Size classify(double width, bool valid, double small, double large) {
  if (!(small > 0 && large > small))
    throw std::invalid_argument("invalid size thresholds");
  if (!valid || !std::isfinite(width) || width <= 0)
    return Size::Unknown;
  if (width <= small)
    return Size::Small;
  if (width >= large)
    return Size::Large;
  return Size::Unknown;
}
struct Circle {
  double x, y, r;
};
inline bool segment_safe(double ax, double ay, double bx, double by,
                         const std::vector<Circle> &zones) {
  if (!std::isfinite(ax + ay + bx + by))
    return false;
  for (const auto &z : zones) {
    const double dx = bx - ax, dy = by - ay, l = dx * dx + dy * dy;
    const double t =
        l > 0 ? std::fmax(
                    0., std::fmin(1., ((z.x - ax) * dx + (z.y - ay) * dy) / l))
              : 0;
    if (std::hypot(ax + t * dx - z.x, ay + t * dy - z.y) <= z.r)
      return false;
  }
  return true;
}
} // namespace limo_fire_bt
