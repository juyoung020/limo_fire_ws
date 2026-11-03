// 학습 맵 묶음 검사·목록: 장면마다 학습 환경과 같은 build_world(env_host.cpp)로 칸 격자를 만들어
// "집이 닫힘"(갈 수 있는 바닥이 맵 테두리에 안 닿음)을 확인하고, 통과한 맵의 갈 수 있는 바닥 넓이·벽 면 칸 수(덮음 분모)·크기 등급을 적는다.
// 닫히지 않거나 build_world 가 실패한 맵은 목록에서 빼고 까닭(새는 곳 좌표 등)을 버림 목록에 적는다 — 파일은 지우지 않는다(호출한 쪽이 정리).
//   scene_list --list <list.tsv> --rejected <rejected.tsv> [--env config/env.yaml] [--png <폴더> --png-every N] <장면.scene> …
// list.tsv: path(목록 파일 폴더 기준 상대 경로) floor_m2 wall_cells grade gw gh furnished(가구 판 0/1) — 커리큘럼은 grade·furnished 로 고른다.
// 크기 등급(갈 수 있는 바닥 넓이): 0 < 30 m² ≤ 1 < 80 m² ≤ 2 < 150 m² ≤ 3. 작은 방(한두 칸)·작은 집·보통 집·큰 집.
// --png: N 개마다 build_world 격자를 PGM 으로 (흰 바닥, 검정 벽 면 칸(분모), 회색 그 밖 막힘, 출발 칸 둘레 진회색) — 눈으로 확인용.
#include <algorithm>
#include <cstdio>
#include <string>
#include <vector>

#include "env_api.h"

namespace {

int grade_of(float floor_m2) { return floor_m2 < 30.f ? 0 : floor_m2 < 80.f ? 1 : floor_m2 < 150.f ? 2 : 3; }

std::string rel_to(const std::string& path, const std::string& dir) {
  if (!dir.empty() && path.compare(0, dir.size(), dir) == 0 && path.size() > dir.size() && path[dir.size()] == '/')
    return path.substr(dir.size() + 1);
  return path;
}

void write_pgm(const sim::World& w, float res, const std::string& path) {
  std::vector<uint8_t> px((size_t)w.gw * w.gh, 200);
  for (int y = 0; y < w.gh; ++y)
    for (int x = 0; x < w.gw; ++x) {
      const uint8_t g = w.grid[(size_t)y * w.gw + x];
      uint8_t v = 200;   // 막힘 아닌 칸 중 바닥 밖 (벽 두께 안·집 밖)
      if (g & sim::kCellFloor) v = 255;
      if (g & sim::kCellBlocked) v = 120;
      if (g & sim::kCellWall) v = 0;
      px[(size_t)(w.gh - 1 - y) * w.gw + x] = v;   // 위 행 = y 큰 쪽
    }
  // 출발점 표시: 지름 9 칸 점
  const int sx = (int)((w.scene.start[0] - w.ox) / res), sy = (int)((w.scene.start[1] - w.oy) / res);
  for (int dy = -4; dy <= 4; ++dy)
    for (int dx = -4; dx <= 4; ++dx) {
      const int u = sx + dx, v = sy + dy;
      if (dx * dx + dy * dy <= 16 && u >= 0 && v >= 0 && u < w.gw && v < w.gh) px[(size_t)(w.gh - 1 - v) * w.gw + u] = 60;
    }
  FILE* f = std::fopen(path.c_str(), "wb");
  if (!f) return;
  std::fprintf(f, "P5\n%d %d\n255\n", w.gw, w.gh);
  std::fwrite(px.data(), 1, px.size(), f);
  std::fclose(f);
}

}  // namespace

int main(int argc, char** argv) {
  std::string env_path = SIM_CONFIG_DIR "/env.yaml", list_path, rej_path, png_dir;
  int png_every = 0;
  std::vector<std::string> scenes;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    if (a == "--env" && i + 1 < argc) env_path = argv[++i];
    else if (a == "--list" && i + 1 < argc) list_path = argv[++i];
    else if (a == "--rejected" && i + 1 < argc) rej_path = argv[++i];
    else if (a == "--png" && i + 1 < argc) png_dir = argv[++i];
    else if (a == "--png-every" && i + 1 < argc) png_every = std::stoi(argv[++i]);
    else scenes.push_back(a);
  }
  if (scenes.empty() || list_path.empty() || rej_path.empty()) {
    std::fprintf(stderr, "scene_list --list <list.tsv> --rejected <rejected.tsv> [--env env.yaml] [--png <폴더> --png-every N] <장면.scene> …\n");
    return 2;
  }
  std::sort(scenes.begin(), scenes.end());
  const sim::EnvHostCfg hc = sim::load_env_cfg(env_path);
  const float res = hc.cfg.res;
  const std::string list_dir = list_path.find('/') == std::string::npos ? "" : list_path.substr(0, list_path.find_last_of('/'));
  FILE* fl = std::fopen(list_path.c_str(), "w");
  FILE* fr = std::fopen(rej_path.c_str(), "w");
  if (!fl || !fr) { std::fprintf(stderr, "목록 파일을 못 씀\n"); return 1; }
  std::fprintf(fl, "path\tfloor_m2\twall_cells\tgrade\tgw\tgh\tfurnished\n");
  std::fprintf(fr, "path\treason\n");
  int n_ok = 0, n_bad = 0, by_grade[4] = {0, 0, 0, 0};
  for (size_t i = 0; i < scenes.size(); ++i) {
    const std::string& p = scenes[i];
    try {
      const sim::World w = sim::build_world(p, sim::load_scene(p), hc, false);
      if (!w.closed) { std::fprintf(fr, "%s\t%s\n", rel_to(p, list_dir).c_str(), w.leak.c_str()); ++n_bad; continue; }
      const float floor_m2 = (float)w.n_floor * res * res;
      const int g = grade_of(floor_m2);
      // 가구 판: map2scene --heights 로 만든 물체(이름 furniture_*)가 있으면 1
      const bool furnished = std::any_of(w.scene.obj_names.begin(), w.scene.obj_names.end(), [](const std::string& s) { return s.rfind("furniture_", 0) == 0; });
      std::fprintf(fl, "%s\t%.2f\t%d\t%d\t%d\t%d\t%d\n", rel_to(p, list_dir).c_str(), floor_m2, w.n_wall, g, w.gw, w.gh, (int)furnished);
      ++n_ok;
      ++by_grade[g];
      if (!png_dir.empty() && png_every > 0 && (n_ok - 1) % png_every == 0) {
        std::string stem = p.substr(p.find_last_of('/') + 1);
        stem = stem.substr(0, stem.find_last_of('.'));
        write_pgm(w, res, png_dir + "/" + stem + "_g" + std::to_string(g) + ".pgm");
      }
    } catch (const std::exception& e) {
      std::string m = e.what();
      for (char& c : m) if (c == '\t' || c == '\n') c = ' ';
      std::fprintf(fr, "%s\t%s\n", rel_to(p, list_dir).c_str(), m.c_str());
      ++n_bad;
    }
  }
  std::fclose(fl);
  std::fclose(fr);
  std::printf("통과(닫힘) %d, 버림 %d — 등급 0/1/2/3 = %d/%d/%d/%d → %s, %s\n", n_ok, n_bad, by_grade[0], by_grade[1], by_grade[2], by_grade[3],
              list_path.c_str(), rej_path.c_str());
  return n_ok > 0 ? 0 : 1;
}
