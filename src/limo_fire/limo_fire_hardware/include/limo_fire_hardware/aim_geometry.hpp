// 평면 접근·좌우 조준 계산 — ROS 의존 없음(test_aim_geometry 가 시험).
// 발사대 각은 기구로 15° 고정이라 상하 계산은 없다. 좌우는 LIMO 차체 회전, 거리는 표적 둘레의 정지 거리 고리로만 맞춘다.
//   mount = 발사대 회전축(launcher_pivot)의 base_link 기준 위치, mount_yaw = 발사대 수평 방향 (URDF → TF 로 읽음)
//   거리는 회전축에서 표적까지의 수평 거리. 차체 roll·pitch 는 0 으로 본다(평면 주행).
// 예전 Python aim_geometry.py·navigation_checks.py·mapper_observation_bridge.py 의 계산을 옮겼다(상하·탄도 계산은 뺌).
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <initializer_list>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace limo_fire_hardware {

struct Vec3 {
  double x = 0, y = 0, z = 0;
};
struct Pose {   // 지도 위 차체: 위치 + yaw
  double x = 0, y = 0, z = 0, yaw = 0;
};
struct Candidate {
  double base_x, base_y, base_yaw;      // 차체가 설 자리
  double pivot_x, pivot_y, pivot_z;     // 그때 회전축 자리
  double distance_m;
};

inline double wrap(double a) { return std::atan2(std::sin(a), std::cos(a)); }

inline void require_finite(std::initializer_list<double> v, const char* what) {
  for (double x : v)
    if (!std::isfinite(x)) throw std::invalid_argument(std::string(what) + " 는 유한한 값이어야 함");
}

// 표적을 발사대 중립 축 좌표(앞·왼쪽·위)로
inline Vec3 relative_target(const Vec3& t, const Pose& b, const Vec3& m, double mount_yaw) {
  require_finite({t.x, t.y, t.z}, "target");
  require_finite({b.x, b.y, b.z, b.yaw}, "base pose");
  require_finite({m.x, m.y, m.z}, "mount");
  require_finite({mount_yaw}, "mount_yaw");
  const double c = std::cos(b.yaw), s = std::sin(b.yaw);
  const double px = b.x + c * m.x - s * m.y, py = b.y + s * m.x + c * m.y, pz = b.z + m.z;
  const double dx = t.x - px, dy = t.y - py, ny = b.yaw + mount_yaw;
  const double cn = std::cos(ny), sn = std::sin(ny);
  return {cn * dx + sn * dy, -sn * dx + cn * dy, t.z - pz};
}

// 발사대 축에 대한 표적의 좌우 오차(rad, 왼쪽 +)
inline double heading_error(const Vec3& t, const Pose& b, const Vec3& m = {}, double mount_yaw = 0.0) {
  const Vec3 r = relative_target(t, b, m, mount_yaw);
  if (std::hypot(r.x, r.y) <= 1e-9) throw std::invalid_argument("표적이 회전축과 수평 거리 0");
  return std::atan2(r.y, r.x);
}

// 회전축에서 표적까지 수평 거리
inline double pivot_distance(const Vec3& t, const Pose& b, const Vec3& m) {
  const Vec3 r = relative_target(t, b, m, 0.0);
  return std::hypot(r.x, r.y);
}

// 제자리 회전 속도(rad/s): 허용 오차 안이면 0, 밖이면 gain·오차를 [min, max] 로 자름
inline double yaw_rate_command(double err, double tol, double gain, double min_rps, double max_rps) {
  require_finite({err, tol, gain, min_rps, max_rps}, "yaw rate 입력");
  if (tol <= 0 || gain <= 0 || min_rps <= 0 || max_rps < min_rps) throw std::invalid_argument("yaw rate 한계가 틀림");
  if (std::fabs(err) <= tol) return 0.0;
  return std::copysign(std::min(max_rps, std::max(min_rps, gain * std::fabs(err))), err);
}

// 차체 x·y 를 그대로 두고 발사대가 표적을 향하는 차체 yaw. 해가 둘이면 지금 yaw 에 가까운 쪽
inline double body_yaw_for_target(const Vec3& t, const Pose& b, const Vec3& m = {}, double beta = 0.0) {
  require_finite({t.x, t.y, t.z}, "target");
  require_finite({b.x, b.y, b.z, b.yaw}, "base pose");
  require_finite({m.x, m.y, m.z, beta}, "mount");
  const double dx = t.x - b.x, dy = t.y - b.y, dist = std::hypot(dx, dy);
  const double lateral = -m.x * std::sin(beta) + m.y * std::cos(beta);
  const double forward = m.x * std::cos(beta) + m.y * std::sin(beta);
  if (dist <= 1e-9 || std::fabs(lateral) > dist) throw std::invalid_argument("이 자리에서는 표적을 향할 수 없음");
  const double delta = std::asin(lateral / dist), bearing = std::atan2(dy, dx);
  bool found = false;
  double best = 0.0;
  for (double a : {delta, M_PI - delta}) {
    if (dist * std::cos(a) - forward > 1e-9) {
      const double yaw = wrap(bearing - beta - a);
      if (!found || std::fabs(wrap(yaw - b.yaw)) < std::fabs(wrap(best - b.yaw))) best = yaw;
      found = true;
    }
  }
  if (!found) throw std::invalid_argument("표적이 회전축 뒤에 있음");
  return best;
}

// 표적 둘레, 회전축 수평 거리 distance 인 고리 위의 차체 자리들. 첫 각은 지금 회전축→표적 방향 기준
inline std::vector<Candidate> approach_candidates(const Vec3& t, const Pose& b, double distance, const Vec3& m,
                                                  double beta, const std::vector<double>& offsets_rad) {
  require_finite({t.x, t.y, t.z}, "target");
  require_finite({b.x, b.y, b.z, b.yaw}, "base pose");
  require_finite({m.x, m.y, m.z, beta}, "mount");
  if (!std::isfinite(distance) || distance <= 0.0) throw std::invalid_argument("distance_m 은 양수");
  if (offsets_rad.empty()) throw std::invalid_argument("후보 각이 하나는 있어야 함");
  std::vector<double> off;
  for (double o : offsets_rad) {
    require_finite({o}, "후보 각");
    const double w = wrap(o);
    for (double p : off)
      if (std::fabs(wrap(w - p)) <= 1e-9) throw std::invalid_argument("후보 각이 2π 로 겹침");
    off.push_back(w);
  }
  const double c = std::cos(b.yaw), s = std::sin(b.yaw);
  const double cpx = b.x + c * m.x - s * m.y, cpy = b.y + s * m.x + c * m.y;
  const double dx = t.x - cpx, dy = t.y - cpy;
  const double dir = std::hypot(dx, dy) <= 1e-9 ? b.yaw + beta : std::atan2(dy, dx);
  std::vector<Candidate> out;
  for (double o : off) {
    const double launch_yaw = dir + o, base_yaw = wrap(launch_yaw - beta);
    const double px = t.x - distance * std::cos(launch_yaw), py = t.y - distance * std::sin(launch_yaw);
    const double cb = std::cos(base_yaw), sb = std::sin(base_yaw);
    out.push_back({px - cb * m.x + sb * m.y, py - sb * m.x - cb * m.y, base_yaw, px, py, b.z + m.z, distance});
  }
  return out;
}

// 차체가 pose 에 있을 때의 후보(회전축 자리 다시 셈)
inline Candidate candidate_at(const Pose& p, const Vec3& m, double distance) {
  const double c = std::cos(p.yaw), s = std::sin(p.yaw);
  return {p.x, p.y, p.yaw, p.x + c * m.x - s * m.y, p.y + s * m.x + c * m.y, p.z + m.z, distance};
}

// ---- 점유 격자 (nav_msgs/OccupancyGrid 와 같은 뜻, ROS 없이)
struct GridView {
  double origin_x = 0, origin_y = 0, origin_yaw = 0, resolution = 0;
  uint32_t width = 0, height = 0;
  const int8_t* data = nullptr;
  size_t size = 0;
};

inline std::optional<int> grid_cell(const GridView& g, double x, double y) {
  if (!std::isfinite(g.resolution) || g.resolution <= 0 || !g.data) return std::nullopt;
  const double dx = x - g.origin_x, dy = y - g.origin_y, c = std::cos(g.origin_yaw), s = std::sin(g.origin_yaw);
  const double col = std::floor((c * dx + s * dy) / g.resolution), row = std::floor((-s * dx + c * dy) / g.resolution);
  if (!(col >= 0 && col < g.width && row >= 0 && row < g.height)) return std::nullopt;
  const size_t i = static_cast<size_t>(row) * g.width + static_cast<size_t>(col);
  if (i >= g.size) return std::nullopt;
  return g.data[i];
}

// 차체를 감싸는 정사각(반지름 radius)과 회전축→표적 시선(표적 앞 target_margin 은 뺌)이 모두 빈 칸(0 ≤ 값 < blocked_cost)인지
inline bool candidate_clear(const GridView& g, const Candidate& cand, const Vec3& target, double radius, int blocked_cost,
                            double target_margin) {
  const double step = g.resolution * 0.5;
  if (!std::isfinite(step) || step <= 0) return false;
  auto free = [&](double x, double y) {
    const auto v = grid_cell(g, x, y);
    return v && *v >= 0 && *v < blocked_cost;
  };
  const int n = std::max(1, static_cast<int>(std::ceil(2 * radius / step)));
  for (int i = 0; i <= n; ++i)
    for (int j = 0; j <= n; ++j)
      if (!free(cand.base_x - radius + 2 * radius * i / n, cand.base_y - radius + 2 * radius * j / n)) return false;
  const double dx = target.x - cand.pivot_x, dy = target.y - cand.pivot_y, len = std::hypot(dx, dy);
  const double vis = std::max(0.0, len - target_margin);
  const int k = std::max(1, static_cast<int>(std::ceil(vis / step)));
  for (int i = 0; i <= k; ++i) {
    const double f = len > 0 ? vis / len * i / k : 0.0;
    if (!free(cand.pivot_x + dx * f, cand.pivot_y + dy * f)) return false;
  }
  return true;
}

// ---- 물체 지도(/perception/objects) 에서 방금 본 불 하나 고르기
// 물체 지도는 저장된 물체도 10 Hz 로 다시 내므로 머리 시각은 새 관측의 증거가 아니다 — 칸별 "마지막으로 본 뒤 지난 초"(covariance[35])를 본다.
struct MapObject {
  std::string id, class_id;
  double score = 0, seen_age_s = 0;
  Vec3 position;
};
struct FireChoice {
  int index = -1;   // -1 = 없음
  double age_s = 0;
  std::string reason;
};

inline FireChoice select_live_fire(const std::vector<MapObject>& objs, const std::string& target_id, double min_conf,
                                   double max_age_s, const std::string& fire_class = "fire") {
  FireChoice out;
  int count = 0;
  for (size_t i = 0; i < objs.size(); ++i) {
    const auto& o = objs[i];
    if (o.id.empty() || (!target_id.empty() && o.id != target_id) || o.class_id != fire_class) continue;
    if (!std::isfinite(o.score) || o.score < min_conf || !std::isfinite(o.seen_age_s) || o.seen_age_s < 0 ||
        o.seen_age_s > max_age_s || !std::isfinite(o.position.x) || !std::isfinite(o.position.y) ||
        !std::isfinite(o.position.z))
      continue;
    ++count;
    out.index = static_cast<int>(i);
    out.age_s = o.seen_age_s;
  }
  if (count == 0) return {-1, 0, "최근에 본, 지도 좌표가 있는 불이 없음"};
  if (count > 1) return {-1, 0, "불이 여럿 — target_id 로 하나를 고를 것"};
  return out;
}

}  // namespace limo_fire_hardware
