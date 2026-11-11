// 블랙보드에 오가는 값의 타입. XML 에 글자로 적을 수 있게 convertFromString 을 둔다 ("x;y;z", "x;y;yaw")
#pragma once

#include <cstdio>
#include <string>
#include <vector>

#include <behaviortree_cpp/basic_types.h>

namespace limo_fire_bt
{

// 지도(map) 좌표의 점 (m)
struct MapPoint
{
  double x = 0, y = 0, z = 0;
};

// 지도 위 자세 (m, rad)
struct Pose2
{
  double x = 0, y = 0, yaw = 0;
};

using Route = std::vector<Pose2>;

inline std::string to_text(const MapPoint & p)
{
  char b[96];
  std::snprintf(b, sizeof(b), "%.2f;%.2f;%.2f", p.x, p.y, p.z);
  return b;
}

inline std::string to_text(const Pose2 & p)
{
  char b[96];
  std::snprintf(b, sizeof(b), "%.2f;%.2f;%.2f", p.x, p.y, p.yaw);
  return b;
}

}  // namespace limo_fire_bt

namespace BT
{

template<>
inline limo_fire_bt::MapPoint convertFromString(StringView str)
{
  const auto parts = splitString(str, ';');
  if (parts.size() < 2 || parts.size() > 3) {throw RuntimeError("MapPoint 는 \"x;y\" 또는 \"x;y;z\"");}
  limo_fire_bt::MapPoint p;
  p.x = convertFromString<double>(parts[0]);
  p.y = convertFromString<double>(parts[1]);
  if (parts.size() == 3) {p.z = convertFromString<double>(parts[2]);}
  return p;
}

template<>
inline limo_fire_bt::Pose2 convertFromString(StringView str)
{
  const auto parts = splitString(str, ';');
  if (parts.size() < 2 || parts.size() > 3) {throw RuntimeError("Pose2 는 \"x;y\" 또는 \"x;y;yaw\"");}
  limo_fire_bt::Pose2 p;
  p.x = convertFromString<double>(parts[0]);
  p.y = convertFromString<double>(parts[1]);
  if (parts.size() == 3) {p.yaw = convertFromString<double>(parts[2]);}
  return p;
}

}  // namespace BT
