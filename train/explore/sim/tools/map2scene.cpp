// 실물 지도(Nav2 map_server 형식 PGM + YAML, 예: bag_export 의 .map.yaml) → 학습 시뮬 장면(.scene).
// 칸 확률의 등고선(occupied 문턱, marching squares — 칸 가운데 값 사이를 선형으로)을 따라 바닥부터 height 까지 벽면을 세운다.
// 칸 상자로 세우면 비스듬한 벽이 5 cm 계단이 되어 라이다가 가짜 모서리를 본다 — 등고선 벽은 매끈하고 진짜 모서리만 남는다.
// 붙어 있는 막힌 칸 덩어리(4 이웃)가 물체 하나 — 라이다 재질이 물체마다 정해지므로(lidar_real.h) 실물 물체 단위와 가깝게.
//   map2scene <지도.yaml> <출력.scene> [--height 2.4] [--occupied 128]   (PGM 값 < occupied 를 막힘으로: scale 지도면 확률 > 50 %)
#include <yaml-cpp/yaml.h>

#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <string>
#include <vector>

#include "scene_host.h"

namespace {

// PGM 머리 칸 하나: 공백과 '#' 주석(줄 끝까지)을 건너뛴다 — Netpbm 규약. Nav2 map_saver 는 "# CREATOR: …" 줄을 넣는다
bool pgm_token(FILE* f, char* buf, int n) {
  int c = std::fgetc(f);
  while (c != EOF && (std::isspace(c) || c == '#')) {
    if (c == '#')
      while (c != EOF && c != '\n') c = std::fgetc(f);
    c = std::fgetc(f);
  }
  int k = 0;
  while (c != EOF && !std::isspace(c) && k < n - 1) {
    buf[k++] = (char)c;
    c = std::fgetc(f);
  }
  buf[k] = 0;
  return k > 0 && c != EOF;   // 마지막 칸 뒤 공백 하나까지 읽었다 (픽셀은 바로 다음부터)
}

std::vector<uint8_t> read_pgm(const std::string& path, int& w, int& h) {
  FILE* f = std::fopen(path.c_str(), "rb");
  if (!f) throw std::runtime_error("못 읽음: " + path);
  char t[4][32];
  for (auto& s : t)
    if (!pgm_token(f, s, sizeof(s))) throw std::runtime_error(path + ": PGM 머리 잘림");
  w = std::atoi(t[1]);
  h = std::atoi(t[2]);
  if (std::strcmp(t[0], "P5") != 0 || std::atoi(t[3]) != 255 || w <= 0 || h <= 0)
    throw std::runtime_error(path + ": 8 비트 P5 PGM 아님");
  std::vector<uint8_t> px((size_t)w * h);
  if (std::fread(px.data(), 1, px.size(), f) != px.size()) throw std::runtime_error(path + ": 잘림");
  std::fclose(f);
  return px;
}

// 축에 맞춘 상자 (lo, hi) → 삼각형 12 개
void add_box(sim::Scene& s, uint16_t obj, sim::V3 lo, sim::V3 hi) {
  const sim::V3 p[8] = {{lo.x, lo.y, lo.z}, {hi.x, lo.y, lo.z}, {hi.x, hi.y, lo.z}, {lo.x, hi.y, lo.z},
                        {lo.x, lo.y, hi.z}, {hi.x, lo.y, hi.z}, {hi.x, hi.y, hi.z}, {lo.x, hi.y, hi.z}};
  const int f[6][4] = {{0, 1, 2, 3}, {4, 5, 6, 7}, {0, 1, 5, 4}, {1, 2, 6, 5}, {2, 3, 7, 6}, {3, 0, 4, 7}};
  for (const auto& q : f) {
    s.tris.push_back(sim::Tri{p[q[0]], p[q[1]], p[q[2]]});
    s.tris.push_back(sim::Tri{p[q[0]], p[q[2]], p[q[3]]});
    s.obj.push_back(obj);
    s.obj.push_back(obj);
  }
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 3) { std::fprintf(stderr, "map2scene <지도.yaml> <출력.scene> [--height 2.4] [--occupied 50]\n"); return 2; }
  float height = 2.4f;
  int occ_thr = 128;
  for (int i = 3; i + 1 < argc; i += 2) {
    if (!std::strcmp(argv[i], "--height")) height = std::stof(argv[i + 1]);
    else if (!std::strcmp(argv[i], "--occupied")) occ_thr = std::stoi(argv[i + 1]);
  }
  const std::string yaml_path = argv[1];
  const YAML::Node y = YAML::LoadFile(yaml_path);
  const std::string dir = yaml_path.substr(0, yaml_path.find_last_of('/') + 1);
  const float res = y["resolution"].as<float>(), ox = y["origin"][0].as<float>(), oy = y["origin"][1].as<float>();
  if (std::fabs(y["origin"][2].as<float>()) > 1e-6f) throw std::runtime_error("지도 원점 회전은 아직 안 다룸");
  int w, h;
  const std::vector<uint8_t> px = read_pgm(dir + y["image"].as<std::string>(), w, h);
  // 지도 좌표 (x 오른쪽, y 위): PGM 위 행이 y 가 큰 쪽
  std::vector<uint8_t> occ_grid((size_t)w * h, 0);
  for (int cy = 0; cy < h; ++cy)
    for (int cx = 0; cx < w; ++cx) occ_grid[(size_t)cy * w + cx] = px[(size_t)(h - 1 - cy) * w + cx] < occ_thr;
  auto occ = [&](int cx, int cy) { return cx >= 0 && cy >= 0 && cx < w && cy < h && occ_grid[(size_t)cy * w + cx]; };
  // 덩어리 번호
  std::vector<int> comp((size_t)w * h, -1);
  int n_comp = 0;
  std::vector<int> stack;
  for (int cy = 0; cy < h; ++cy)
    for (int cx = 0; cx < w; ++cx) {
      if (!occ(cx, cy) || comp[(size_t)cy * w + cx] >= 0) continue;
      stack.assign(1, cy * w + cx);
      comp[(size_t)cy * w + cx] = n_comp;
      while (!stack.empty()) {
        const int k = stack.back();
        stack.pop_back();
        const int x = k % w, yy = k / w;
        const int nb[4][2] = {{x + 1, yy}, {x - 1, yy}, {x, yy + 1}, {x, yy - 1}};
        for (const auto& q : nb)
          if (occ(q[0], q[1]) && comp[(size_t)q[1] * w + q[0]] < 0) { comp[(size_t)q[1] * w + q[0]] = n_comp; stack.push_back(q[1] * w + q[0]); }
      }
      ++n_comp;
    }
  if (n_comp > 65000) throw std::runtime_error("막힌 덩어리가 너무 많음");
  sim::Scene s;
  for (int i = 0; i < n_comp; ++i) s.obj_names.push_back("map_blob_" + std::to_string(i));
  // 바닥 (라이다·깊이가 바닥을 맞힐 때) — 덩어리와 다른 물체
  s.obj_names.push_back("floor");
  const uint16_t floor_id = (uint16_t)n_comp;
  const sim::V3 flo{ox, oy, -0.01f}, fhi{ox + w * res, oy + h * res, 0.f};
  add_box(s, floor_id, flo, fhi);
  // marching squares: 칸 가운데 4 점 (x, y) (x+1, y) (x+1, y+1) (x, y+1) 의 값 v = 255 − 픽셀 (클수록 막힘), 문턱 iso
  auto val = [&](int cx, int cy) { return 255.f - (float)px[(size_t)(h - 1 - cy) * w + cx]; };
  const float iso = 255.f - (float)occ_thr;
  auto at = [&](float fx, float fy) { return sim::V3{ox + (fx + 0.5f) * res, oy + (fy + 0.5f) * res, 0.f}; };
  int walls = 0;
  for (int cy = 0; cy + 1 < h; ++cy)
    for (int cx = 0; cx + 1 < w; ++cx) {
      const float v[4] = {val(cx, cy), val(cx + 1, cy), val(cx + 1, cy + 1), val(cx, cy + 1)};
      int code = 0;
      for (int k = 0; k < 4; ++k) code |= (v[k] > iso) << k;
      if (code == 0 || code == 15) continue;
      // 변 위 등고선 점: 변 0 (0→1), 1 (1→2), 2 (3→2), 3 (0→3)
      auto lerp = [&](float a, float b) { return std::fabs(b - a) < 1e-6f ? 0.5f : (iso - a) / (b - a); };
      const sim::V3 e[4] = {at(cx + lerp(v[0], v[1]), cy), at(cx + 1, cy + lerp(v[1], v[2])), at(cx + lerp(v[3], v[2]), cy + 1),
                            at(cx, cy + lerp(v[0], v[3]))};
      int seg[4];
      int ns = 0;
      auto add = [&](int a, int b) { seg[ns++] = a; seg[ns++] = b; };
      const float centre = 0.25f * (v[0] + v[1] + v[2] + v[3]);
      switch (code) {
        case 1: case 14: add(3, 0); break;
        case 2: case 13: add(0, 1); break;
        case 3: case 12: add(3, 1); break;
        case 4: case 11: add(1, 2); break;
        case 6: case 9: add(0, 2); break;
        case 7: case 8: add(3, 2); break;
        case 5: if (centre > iso) { add(3, 2); add(0, 1); } else { add(3, 0); add(1, 2); } break;   // 안장: 가운데 값으로 가름
        case 10: if (centre > iso) { add(3, 0); add(1, 2); } else { add(3, 2); add(0, 1); } break;
      }
      // 물체 번호: 네 모서리 중 막힌 칸의 덩어리
      int id = -1;
      const int corner[4][2] = {{cx, cy}, {cx + 1, cy}, {cx + 1, cy + 1}, {cx, cy + 1}};
      for (const auto& q : corner) if (id < 0 && occ(q[0], q[1])) id = comp[(size_t)q[1] * w + q[0]];
      if (id < 0) continue;
      for (int k = 0; k < ns; k += 2) {
        const sim::V3 a = e[seg[k]], b = e[seg[k + 1]];
        const sim::V3 a1{a.x, a.y, height}, b1{b.x, b.y, height};
        s.tris.push_back(sim::Tri{a, b, b1});
        s.tris.push_back(sim::Tri{a, b1, a1});
        s.obj.push_back((uint16_t)id);
        s.obj.push_back((uint16_t)id);
        ++walls;
      }
    }
  sim::build_bvh(s);
  sim::save_scene(s, argv[2]);
  std::printf("%s: %d×%d 칸 (%.2f m) → 막힌 덩어리 %d, 등고선 벽면 %d, 삼각형 %zu → %s\n", yaml_path.c_str(), w, h, res, n_comp, walls,
              s.tris.size(), argv[2]);
  return 0;
}
