// Webots 장면 내보내기(.w3d) → .scene (삼각형 + BVH)
// w3d2scene <in.w3d> <out.scene> [뺄 로봇 이름 …(기본 LIMO)]
#include <cstdio>

#include "w3d.h"

int main(int argc, char** argv) {
  if (argc < 3) { std::fprintf(stderr, "w3d2scene <in.w3d> <out.scene> [뺄 로봇 …]\n"); return 2; }
  std::vector<std::string> exclude;
  for (int i = 3; i < argc; ++i) exclude.push_back(argv[i]);
  if (exclude.empty()) exclude.push_back("LIMO");
  sim::Scene s;
  const auto rep = sim::load_w3d(argv[1], exclude, s);
  sim::build_bvh(s);
  sim::save_scene(s, argv[2]);
  const auto& root = s.nodes[0];
  std::printf("삼각형 %zu, BVH 마디 %zu, 물체 %zu, 뺀 로봇 %d\n", s.tris.size(), s.nodes.size(), s.obj_names.size(), rep.excluded_robots);
  std::printf("범위 x %.2f~%.2f  y %.2f~%.2f  z %.2f~%.2f m\n", root.lo[0], root.hi[0], root.lo[1], root.hi[1], root.lo[2], root.hi[2]);
  for (const auto& [k, n] : rep.skipped) std::printf("못 바꾼 형상: %s × %d\n", k.c_str(), n);
  return 0;
}
