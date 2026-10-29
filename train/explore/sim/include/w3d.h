// Webots 장면 내보내기(.w3d) → 삼각형 장면 (src/w3d.cpp). 장면은 Ros2Supervisor 의 animation_start_recording 으로 내보낸다
#pragma once
#include <map>
#include <string>
#include <vector>

#include "scene_host.h"

namespace sim {

struct W3dReport {
  int excluded_robots = 0;
  std::map<std::string, int> skipped;   // 못 바꾼 형상 종류 → 개수
};

// exclude_robots: 장면에서 뺄 Robot 이름 (센서를 단 우리 로봇)
W3dReport load_w3d(const std::string& path, const std::vector<std::string>& exclude_robots, Scene& out);

}  // namespace sim
