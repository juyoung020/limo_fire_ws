// 맵 하나 (장면 + 칸 격자 + 시작 칸) — env_api.h 에서 떼어 냄 (2026-11-16): CUDA 없이 쓰는 호스트 코드(scripts/fuel_upper.h · 실물 노드 limo_fire_rl)가 포함
#pragma once
#include <cstdint>
#include <string>
#include <vector>

#include "scene_host.h"

namespace sim {

// 맵 하나: 장면 + 칸 격자 + 시작 칸
struct World {
  std::string name;
  Scene scene;                          // 바탕 장면 (덧붙인 형상이 있으면 물체 이름·크기는 덧 물체까지 합친 것)
  Scene extra;                          // 덧붙인 형상 (2026-11-15 — 목록 줄 "바탕.scene+덧.scene"): 삼각형 번호는 바탕 뒤, 물체 번호는 바탕 물체 뒤로 밀어 둠. 없으면 비어 있음
  std::string base_key;                 // 장치에서 바탕 삼각형을 맵끼리 나눠 쓰는 열쇠 (바탕 장면 경로 — load_world·목록으로 읽은 맵. 비면 나누지 않음)
  std::vector<uint8_t> grid;            // kCell* 비트 (env.h)
  std::vector<int32_t> spawn;
  std::vector<uint8_t> clear;           // 칸마다 막힌 칸까지 거리 (cm) — 움직이는 사람 (people_host.cpp people_clearance)
  int gw = 0, gh = 0;
  int n_wall = 0, n_floor = 0;          // 덮음 분모 (벽 면 칸), 바닥 덮음 분모 (갈 수 있는 바닥 칸)
  bool closed = true;                   // 갈 수 있는 바닥이 맵 테두리에 안 닿음 (집이 닫힘)
  std::string leak;                     // 닫히지 않았으면 어디서 새는지 (좌표)
  float ox = 0, oy = 0;
  int stage = -1;                       // 커리큘럼 단계 (장면 목록의 등급 열). −1 이면 넓이로 (mdp::cur::stage_of_area)
};

}  // namespace sim
