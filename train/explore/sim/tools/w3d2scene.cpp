// Webots 장면 내보내기(.w3d) → .scene (삼각형 + BVH)
// w3d2scene <in.w3d> <out.scene> [뺄 로봇 이름 …(기본 LIMO)] [--start x y]
// 출발점: --start 가 없으면 처음 뺀 로봇의 자리. 로봇도 --start 도 없으면 실패 (덮음 분모를 정할 수 없음)
#include <cstdio>
#include <cstring>
#include <string>

#include "w3d.h"

int main(int argc, char** argv) {
  if (argc < 3) { std::fprintf(stderr, "w3d2scene <in.w3d> <out.scene> [뺄 로봇 …] [--start x y]\n"); return 2; }
  std::vector<std::string> exclude;
  bool has_start = false;
  float start[2] = {0.f, 0.f};
  for (int i = 3; i < argc; ++i) {
    if (!std::strcmp(argv[i], "--start") && i + 2 < argc) {
      start[0] = std::stof(argv[i + 1]);
      start[1] = std::stof(argv[i + 2]);
      has_start = true;
      i += 2;
    } else {
      exclude.push_back(argv[i]);
    }
  }
  if (exclude.empty()) exclude.push_back("LIMO");
  sim::Scene s;
  const auto rep = sim::load_w3d(argv[1], exclude, s);
  if (!has_start) {
    if (rep.excluded_robots == 0) {
      std::fprintf(stderr, "w3d2scene: 뺀 로봇이 없어 출발점을 모름 — --start x y 를 줄 것\n");
      return 1;
    }
    start[0] = (float)rep.robot_x;
    start[1] = (float)rep.robot_y;
  }
  s.start[0] = start[0];
  s.start[1] = start[1];
  sim::build_bvh(s);
  sim::save_scene(s, argv[2]);
  const auto& root = s.nodes[0];
  std::printf("삼각형 %zu, BVH 마디 %zu, 물체 %zu, 뺀 로봇 %d, 출발점 (%.2f, %.2f)%s\n", s.tris.size(), s.nodes.size(), s.obj_names.size(),
              rep.excluded_robots, s.start[0], s.start[1], has_start ? " (--start)" : " (뺀 로봇 자리)");
  std::printf("범위 x %.2f~%.2f  y %.2f~%.2f  z %.2f~%.2f m\n", root.lo[0], root.hi[0], root.lo[1], root.hi[1], root.lo[2], root.hi[2]);
  for (const auto& [k, n] : rep.skipped) std::printf("못 바꾼 형상: %s × %d\n", k.c_str(), n);
  return 0;
}
