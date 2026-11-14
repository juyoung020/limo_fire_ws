// scene_lowobs — 장면에 "라이다 평면 아래 · 몸통 높이 안" 낮은 상자를 더해 새 장면 파일을 만든다 (원본은 안 건드림).
// 라이다는 바닥에서 body.lidar_mount[2] 높이의 수평면만 보므로, 윗면이 그보다 낮고 body.collision_band 아래 끝보다 높은 상자는
// 로봇 지도(라이다)에 안 찍히지만 몸통과는 부딪힌다 (env_host.cpp build_world 가 띠 안 삼각형으로 막힌 칸을 만든다). 카메라(깊이)에는 보인다.
// 판단 모델 실험(scripts/combine_run --upper decider)의 "눈으로만 알 수 있는 막힘" 시험 장면용.
//
//   scene_lowobs --in a.scene [--boxes boxes.tsv --out b.scene] [--env config/env.yaml] [--ppm out.ppm] [--ppm-scale 4]
//   boxes.tsv : 줄마다 상자 하나 — 가운데 x  y (m, 세계)  yaw(도)  길이  폭  높이 (m)  [이름]   (# 주석 · 빈 줄 무시, 바닥 z = 0 에 놓음)
//   --boxes 없이 부르면 원본 보고만 (격자 · 라이다에 안 보이는 벽 면 칸 수)
//
// 확인하고 적는 것 (하나라도 어기면 실패, 파일을 안 씀):
//   · 상자 윗면 < 라이다 높이 − 1 cm, 윗면 > collision_band 아래 끝 (그래야 막힌 칸이 됨)
//   · 새 장면이 build_world 를 통과 (집이 닫힘 · 출발점에 설 수 있음)
//   · 상자 칸이 모두 막힌 칸이 됨
//   · 갈 수 있는 바닥이 상자 둘레만큼만 줄었는가 — 줄어든 넓이를 적는다 (--max-lost <m²> 를 넘으면 실패: 상자가 방을 통째로 끊은 것)
// 그림(--ppm): 흰색 갈 수 있는 바닥 · 검정 벽 면 칸(라이다 높이에 형상 있음) · 빨강 벽 면 칸인데 라이다 높이에 형상 없음 · 회색 그 밖 막힘 · 파랑 더한 상자 ·
//   초록 출발점 · 옅은 선 1 m 눈금(굵은 선 5 m). 위가 +y
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "env_api.h"

namespace {

struct Box {
  float x, y, yaw, len, wid, h;
  std::string name;
};

std::vector<Box> read_boxes(const std::string& path) {
  std::ifstream f(path);
  if (!f) throw std::runtime_error("못 읽음: " + path);
  std::vector<Box> out;
  std::string line;
  while (std::getline(f, line)) {
    const auto p = line.find('#');
    if (p != std::string::npos) line.resize(p);
    std::istringstream is(line);
    Box b{};
    float deg = 0.f;
    if (!(is >> b.x >> b.y >> deg >> b.len >> b.wid >> b.h)) continue;
    b.yaw = deg * 0.017453292f;
    if (!(is >> b.name)) b.name = "lowobs_" + std::to_string(out.size());
    out.push_back(b);
  }
  return out;
}

void add_box(sim::Scene& s, const Box& b) {
  if (s.obj_names.size() >= 0xfffe) throw std::runtime_error("물체 번호가 다 참");
  const uint16_t id = (uint16_t)s.obj_names.size();
  s.obj_names.push_back(b.name);
  const float c = std::cos(b.yaw), sn = std::sin(b.yaw), hl = 0.5f * b.len, hw = 0.5f * b.wid;
  sim::V3 v[8];
  for (int k = 0; k < 8; ++k) {
    const float lx = (k & 1) ? hl : -hl, ly = (k & 2) ? hw : -hw;
    v[k] = sim::V3{b.x + c * lx - sn * ly, b.y + sn * lx + c * ly, (k & 4) ? b.h : 0.f};
  }
  const int q[6][4] = {{0, 1, 3, 2}, {4, 5, 7, 6}, {0, 1, 5, 4}, {2, 3, 7, 6}, {0, 2, 6, 4}, {1, 3, 7, 5}};
  for (const auto& f : q) {
    s.tris.push_back(sim::Tri{v[f[0]], v[f[1]], v[f[2]]});
    s.obj.push_back(id);
    s.tris.push_back(sim::Tri{v[f[0]], v[f[2]], v[f[3]]});
    s.obj.push_back(id);
  }
}

// 라이다 높이 z 의 수평면이 삼각형을 자르는 선분을 칸에 찍음 (그 칸은 라이다에 보일 수 있는 형상이 있음)
std::vector<uint8_t> lidar_cells(const sim::Scene& s, const sim::World& w, float res, float z) {
  std::vector<uint8_t> g((size_t)w.gw * w.gh, 0);
  for (const sim::Tri& t : s.tris) {
    const sim::V3 p[3] = {t.a, t.b, t.c};
    float sx[3], sy[3];
    int n = 0;
    for (int k = 0; k < 3 && n < 3; ++k) {
      const sim::V3 &a = p[k], &b = p[(k + 1) % 3];
      if ((a.z - z) * (b.z - z) > 0.f || a.z == b.z) continue;
      const float u = (z - a.z) / (b.z - a.z);
      sx[n] = a.x + (b.x - a.x) * u;
      sy[n] = a.y + (b.y - a.y) * u;
      ++n;
    }
    if (n < 2) continue;
    const float len = std::hypot(sx[1] - sx[0], sy[1] - sy[0]);
    const int m = std::max(1, (int)std::ceil(len / (0.25f * res)));
    for (int i = 0; i <= m; ++i) {
      const float u = (float)i / (float)m;
      const int cx = (int)std::floor((sx[0] + (sx[1] - sx[0]) * u - w.ox) / res), cy = (int)std::floor((sy[0] + (sy[1] - sy[0]) * u - w.oy) / res);
      if (cx >= 0 && cy >= 0 && cx < w.gw && cy < w.gh) g[(size_t)cy * w.gw + cx] = 1;
    }
  }
  return g;
}

struct Stat {
  int floor = 0, wall = 0, blocked = 0, wall_hidden = 0;
};

Stat stat_of(const sim::World& w, const std::vector<uint8_t>& lid) {
  Stat s;
  for (int y = 0; y < w.gh; ++y)
    for (int x = 0; x < w.gw; ++x) {
      const uint8_t g = w.grid[(size_t)y * w.gw + x];
      s.floor += (g & sim::kCellFloor) != 0;
      s.blocked += (g & sim::kCellBlocked) != 0;
      if (!(g & sim::kCellWall)) continue;
      s.wall += 1;
      bool seen = false;   // 이 칸이나 8 이웃에 라이다 높이 형상
      for (int dy = -1; dy <= 1 && !seen; ++dy)
        for (int dx = -1; dx <= 1 && !seen; ++dx) {
          const int u = x + dx, v = y + dy;
          seen = u >= 0 && v >= 0 && u < w.gw && v < w.gh && lid[(size_t)v * w.gw + u];
        }
      s.wall_hidden += !seen;
    }
  return s;
}

void write_ppm(const sim::World& w, const std::vector<uint8_t>& lid, const std::vector<uint8_t>& added, float res, int sc, const std::string& path) {
  const int W = w.gw * sc, H = w.gh * sc;
  std::vector<uint8_t> px((size_t)W * H * 3);
  const int sx = (int)std::floor((w.scene.start[0] - w.ox) / res), sy = (int)std::floor((w.scene.start[1] - w.oy) / res);
  for (int v = 0; v < H; ++v)
    for (int u = 0; u < W; ++u) {
      const int x = u / sc, y = w.gh - 1 - v / sc;
      const uint8_t g = w.grid[(size_t)y * w.gw + x];
      uint8_t r = 200, gg = 200, b = 200;
      if (g & sim::kCellFloor) r = gg = b = 255;
      if (g & sim::kCellBlocked) r = gg = b = 130;
      if (g & sim::kCellWall) {
        bool seen = false;
        for (int dy = -1; dy <= 1 && !seen; ++dy)
          for (int dx = -1; dx <= 1 && !seen; ++dx) {
            const int a = x + dx, c = y + dy;
            seen = a >= 0 && c >= 0 && a < w.gw && c < w.gh && lid[(size_t)c * w.gw + a];
          }
        r = seen ? 0 : 230; gg = b = seen ? 0 : 30;
      }
      if (!added.empty() && added[(size_t)y * w.gw + x]) { r = 30; gg = 80; b = 240; }
      if ((x - sx) * (x - sx) + (y - sy) * (y - sy) <= 9) { r = 20; gg = 180; b = 60; }
      // 눈금: 세계 좌표 정수 m 선
      const float wx = w.ox + ((float)u + 0.5f) / (float)sc * res, wy = w.oy + ((float)(H - 1 - v) + 0.5f) / (float)sc * res;
      const float fx = wx - std::floor(wx), fy = wy - std::floor(wy), th = res / (float)sc;
      if (fx < th || fy < th) {
        const bool major = (fx < th && ((int)std::floor(wx)) % 5 == 0) || (fy < th && ((int)std::floor(wy)) % 5 == 0);
        const int k = major ? 90 : 35;
        r = (uint8_t)(r - (r - 255 * 0) * k / 255); gg = (uint8_t)(gg - gg * k / 255 + (major ? 0 : 0)); b = (uint8_t)std::min(255, b - b * k / 255 + k);
      }
      uint8_t* p = &px[((size_t)v * W + u) * 3];
      p[0] = r; p[1] = gg; p[2] = b;
    }
  FILE* f = std::fopen(path.c_str(), "wb");
  if (!f) throw std::runtime_error("못 씀: " + path);
  std::fprintf(f, "P6\n%d %d\n255\n", W, H);
  std::fwrite(px.data(), 1, px.size(), f);
  std::fclose(f);
}

}  // namespace

int main(int argc, char** argv) {
  std::string env_path = SIM_CONFIG_DIR "/env.yaml", in, out, boxes_path, ppm;
  int sc = 4;
  float max_lost = 3.0f;   // 상자 때문에 줄어도 되는 갈 수 있는 바닥 (m²) — 넘으면 방이 끊긴 것으로 봄 (잠정)
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    if (a == "--env" && i + 1 < argc) env_path = argv[++i];
    else if (a == "--in" && i + 1 < argc) in = argv[++i];
    else if (a == "--out" && i + 1 < argc) out = argv[++i];
    else if (a == "--boxes" && i + 1 < argc) boxes_path = argv[++i];
    else if (a == "--ppm" && i + 1 < argc) ppm = argv[++i];
    else if (a == "--ppm-scale" && i + 1 < argc) sc = std::max(1, std::atoi(argv[++i]));
    else if (a == "--max-lost" && i + 1 < argc) max_lost = (float)std::atof(argv[++i]);
    else { std::fprintf(stderr, "모르는 인자 %s\n", a.c_str()); return 2; }
  }
  if (in.empty() || (boxes_path.empty() != out.empty())) {
    std::fprintf(stderr, "scene_lowobs --in a.scene [--boxes boxes.tsv --out b.scene] [--env env.yaml] [--ppm out.ppm] [--ppm-scale 4] [--max-lost 3.0]\n");
    return 2;
  }
  try {
    const sim::EnvHostCfg hc = sim::load_env_cfg(env_path);
    const float res = hc.cfg.res, lz = hc.cfg.lidar_z;
    const sim::Scene base = sim::load_scene(in);
    const sim::World w0 = sim::build_world("base", base, hc);
    const std::vector<uint8_t> lid0 = lidar_cells(w0.scene, w0, res, lz);
    const Stat s0 = stat_of(w0, lid0);
    std::printf("원본 %s: 삼각형 %zu · 물체 %zu · 격자 %d × %d (원점 %.2f, %.2f · 칸 %.2f m) · z 범위 %.3f ~ %.3f m · 출발점 (%.2f, %.2f)\n", in.c_str(), base.tris.size(), base.obj_names.size(),
                w0.gw, w0.gh, w0.ox, w0.oy, res, base.nodes[0].lo[2], base.nodes[0].hi[2], base.start[0], base.start[1]);
    std::printf("원본: 갈 수 있는 바닥 %.2f m² · 벽 면 칸 %d · 막힌 칸 %d | 라이다 높이 %.3f m · 몸통 띠 %.2f ~ %.2f m — 벽 면 칸 가운데 라이다 높이에 형상이 없는 칸(이웃 포함) %d (%.1f %%)\n",
                s0.floor * res * res, s0.wall, s0.blocked, lz, hc.band_lo, hc.band_hi, s0.wall_hidden, s0.wall > 0 ? 100.0 * s0.wall_hidden / s0.wall : 0.0);
    if (boxes_path.empty()) {
      if (!ppm.empty()) write_ppm(w0, lid0, {}, res, sc, ppm);
      return 0;
    }
    const std::vector<Box> boxes = read_boxes(boxes_path);
    if (boxes.empty()) throw std::runtime_error("상자가 없음: " + boxes_path);
    sim::Scene s = base;
    for (const Box& b : boxes) {
      if (!(b.h < lz - 0.01f)) throw std::runtime_error(b.name + ": 윗면이 라이다 높이 − 1 cm 보다 높음 — 라이다에 보임");
      if (!(b.h > hc.band_lo)) throw std::runtime_error(b.name + ": 윗면이 몸통 띠 아래 끝보다 낮음 — 안 부딪힘");
      if (b.len <= 0.f || b.wid <= 0.f) throw std::runtime_error(b.name + ": 크기");
      add_box(s, b);
    }
    sim::build_bvh(s);
    sim::compute_obj_size(s);
    const sim::World w1 = sim::build_world("variant", s, hc);
    if (w1.gw != w0.gw || w1.gh != w0.gh || w1.ox != w0.ox || w1.oy != w0.oy) throw std::runtime_error("상자가 맵 테두리를 넓힘 — 상자를 집 안에 둘 것");
    const std::vector<uint8_t> lid1 = lidar_cells(w1.scene, w1, res, lz);
    const Stat s1 = stat_of(w1, lid1);
    std::vector<uint8_t> added((size_t)w1.gw * w1.gh, 0);
    int n_add = 0, n_free = 0, n_was_floor = 0;
    for (const Box& b : boxes) {   // 상자 안 칸 (칸 가운데 기준)
      const float c = std::cos(b.yaw), sn = std::sin(b.yaw), r = 0.5f * std::hypot(b.len, b.wid);
      for (int y = (int)std::floor((b.y - r - w1.oy) / res); y <= (int)std::floor((b.y + r - w1.oy) / res); ++y)
        for (int x = (int)std::floor((b.x - r - w1.ox) / res); x <= (int)std::floor((b.x + r - w1.ox) / res); ++x) {
          if (x < 0 || y < 0 || x >= w1.gw || y >= w1.gh) continue;
          const float dx = w1.ox + ((float)x + 0.5f) * res - b.x, dy = w1.oy + ((float)y + 0.5f) * res - b.y;
          const float lx = c * dx + sn * dy, ly = -sn * dx + c * dy;
          if (std::fabs(lx) > 0.5f * b.len || std::fabs(ly) > 0.5f * b.wid) continue;
          const size_t k = (size_t)y * w1.gw + x;
          if (added[k]) continue;
          added[k] = 1;
          n_add += 1;
          n_free += !(w1.grid[k] & sim::kCellBlocked);
          n_was_floor += (w0.grid[k] & sim::kCellFloor) != 0;
        }
    }
    int lid_new = 0;
    for (size_t k = 0; k < lid1.size(); ++k) lid_new += lid1[k] && !lid0[k];
    const float lost = (float)(s0.floor - s1.floor) * res * res;
    std::printf("상자 %zu 개 (%s): 상자 칸 %d (원본에서 갈 수 있는 바닥이던 칸 %d) · 그 가운데 안 막힌 칸 %d · 라이다 높이에 새로 생긴 형상 칸 %d\n", boxes.size(), boxes_path.c_str(), n_add, n_was_floor, n_free, lid_new);
    std::printf("변형: 갈 수 있는 바닥 %.2f m² (원본보다 %.2f m² 줄어듦) · 벽 면 칸 %d (라이다 높이에 형상 없는 칸 %d, 원본 %d) · 닫힘 %s\n", s1.floor * res * res, lost, s1.wall, s1.wall_hidden, s0.wall_hidden,
                w1.closed ? "예" : "아니오");
    for (const Box& b : boxes) std::printf("  %s: (%.2f, %.2f) yaw %.0f° · %.2f × %.2f × %.3f m\n", b.name.c_str(), b.x, b.y, b.yaw * 57.29578f, b.len, b.wid, b.h);
    if (n_free > 0) throw std::runtime_error("상자 칸 가운데 안 막힌 칸이 있음");
    if (lid_new > 0) throw std::runtime_error("라이다 높이에 새 형상이 생김");
    if (lost > max_lost) throw std::runtime_error("갈 수 있는 바닥이 --max-lost 넘게 줄어듦 — 상자가 길을 통째로 끊음");
    if (!ppm.empty()) write_ppm(w1, lid1, added, res, sc, ppm);
    sim::save_scene(s, out);
    std::printf("씀: %s\n", out.c_str());
  } catch (const std::exception& e) {
    std::fprintf(stderr, "scene_lowobs: %s\n", e.what());
    return 1;
  }
  return 0;
}
