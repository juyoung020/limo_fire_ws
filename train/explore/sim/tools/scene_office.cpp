// scene_office — 장면에 사무실·강의실 가구 무리(책상 줄 + 바퀴 의자)를 더해 새 장면(또는 덧 장면)을 만든다 (원본은 안 건드림).
// 까닭: 실물 1116 강의실(책상 무리 6 · 바퀴 의자, train/explore/sim/analysis/real_limo.md "실물 방 재현")에서 아래층 정책이 좁은 틈 앞에서 제자리만 돌았는데,
// 학습 맵 가구(scenes/README "절차 가구")는 탁자·의자가 다리 기둥뿐이라 이런 모양이 없다:
//   · 바퀴 의자 다섯 갈래 받침 — 라이다 평면(body.lidar_mount z 0.111 m) 아래 · 몸통 띠(collision_band 0.03~0.25 m) 안: 라이다엔 가운데 기둥만, 깊이 층·몸통엔 받침 전체
//   · 책상 상판·의자 좌판·등받이 — 몸통 띠 위: 안 부딪히고 깊이 영상에만 보임 (로봇이 책상 밑으로 지나갈 수 있음)
//   · 의자 사이·책상 무리 사이 좁은 틈 (몸통 폭 0.22 m 보다 조금 넓은 것부터)
//   scene_office --in a.scene (--out b.scene | --overlay b_ov.scene) [--seed 1] [--clusters 6] [--fill 0.25] [--env env.yaml] [--ppm 그림.ppm]
// 값 (근거 · 잠정):
//   책상 1.2~1.6 × 0.6~0.8 m, 상판 0.72~0.76 m 높이·3 cm 두께 (EN 527-1 고정 높이 책상 74~76 cm 를 파는 쪽이 적은 값 — 표준 원문 표는 못 봄, 잠정) · 다리 4 cm 각
//   의자 받침 지름 0.60~0.70 m (교체 부품 판매 목록 635 · 650 · 700 · 710 mm), 갈래 5 개 폭 5 cm · 높이 0.04~0.09 m, 바퀴 5 cm 상자 0~0.06 m (잠정),
//   기둥 5 cm 0.09~0.45 m, 좌판 0.46 m 각 0.45~0.52 m, 등받이 0.44 × 0.04 m 0.55~0.95 m (잠정)
//   무리 = 책상 1~3 개(반은 1 개) × 등 맞댄 2 줄, 책상마다 의자 하나(책상 끝에서 받침 가장자리까지 0.15~0.45 m, 방향 ±40°, 20 % 는 통로로 0.2~0.5 m 더 밀려남) — 잠정
//   무리 사이·벽까지 통로 0.45~1.2 m (몸통 0.32 × 0.22 + 여유 — 좁은 것은 로봇이 겨우 지나감) — 잠정
// 무리 하나마다 확인: 출발점 1.2 m 밖 · 원래 갈 수 있는 바닥 안(벽에서 0.3 m) · 다른 무리와 통로 이상 · 넣은 뒤 build_world 통과(닫힘·출발점) ·
//   갈 수 있는 바닥이 무리 넓이 × 1.2 + 0.5 m² 넘게 줄지 않음(다른 곳을 고립시키지 않음). 못 넘으면 그 무리를 버리고 다른 자리.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <random>
#include <string>
#include <vector>

#include "env_api.h"

namespace {

struct Box {   // 바닥 위 z0 ~ z1 의 기운 상자
  float x, y, yaw, len, wid, z0, z1;
};

void add_boxes(sim::Scene& s, const std::vector<Box>& bs, const std::string& name) {
  if (s.obj_names.size() >= 0xfffe) throw std::runtime_error("물체 번호가 다 참");
  const uint16_t id = (uint16_t)s.obj_names.size();
  s.obj_names.push_back(name);
  for (const Box& b : bs) {
    const float c = std::cos(b.yaw), sn = std::sin(b.yaw), hl = 0.5f * b.len, hw = 0.5f * b.wid;
    sim::V3 v[8];
    for (int k = 0; k < 8; ++k) {
      const float lx = (k & 1) ? hl : -hl, ly = (k & 2) ? hw : -hw;
      v[k] = sim::V3{b.x + c * lx - sn * ly, b.y + sn * lx + c * ly, (k & 4) ? b.z1 : b.z0};
    }
    const int q[6][4] = {{0, 1, 3, 2}, {4, 5, 7, 6}, {0, 1, 5, 4}, {2, 3, 7, 6}, {0, 2, 6, 4}, {1, 3, 7, 5}};
    for (const auto& f : q) {
      s.tris.push_back(sim::Tri{v[f[0]], v[f[1]], v[f[2]]});
      s.obj.push_back(id);
      s.tris.push_back(sim::Tri{v[f[0]], v[f[2]], v[f[3]]});
      s.obj.push_back(id);
    }
  }
}

struct Piece {   // 물체 하나 (책상 · 의자)
  std::string kind;
  std::vector<Box> boxes;
};

// 로봇 기준이 아닌 무리 틀(가운데 cx, cy, 방향 yaw)의 점 → 세계
void to_world(float cx, float cy, float yaw, float lx, float ly, float* wx, float* wy) {
  const float c = std::cos(yaw), s = std::sin(yaw);
  *wx = cx + c * lx - s * ly;
  *wy = cy + s * lx + c * ly;
}

Piece make_desk(float cx, float cy, float yaw, float w, float d, float top) {
  Piece p{"desk", {}};
  const float leg = 0.04f;
  for (int i = 0; i < 4; ++i) {
    float x, y;
    to_world(cx, cy, yaw, (i & 1 ? 1.f : -1.f) * (0.5f * w - 0.5f * leg - 0.02f), (i & 2 ? 1.f : -1.f) * (0.5f * d - 0.5f * leg - 0.02f), &x, &y);
    p.boxes.push_back(Box{x, y, yaw, leg, leg, 0.f, top - 0.03f});
  }
  p.boxes.push_back(Box{cx, cy, yaw, w, d, top - 0.03f, top});
  return p;
}

Piece make_chair(float cx, float cy, float yaw, std::mt19937& rng) {
  std::uniform_real_distribution<float> u(0.f, 1.f);
  Piece p{"chair", {}};
  const float r = 0.5f * (0.60f + 0.10f * u(rng)), a0 = u(rng) * 6.2831853f;
  for (int k = 0; k < 5; ++k) {   // 다섯 갈래 (가운데에서 끝까지) + 끝 바퀴
    const float a = a0 + 1.2566371f * (float)k, c = std::cos(a), s = std::sin(a);
    p.boxes.push_back(Box{cx + 0.5f * r * c, cy + 0.5f * r * s, a, r, 0.05f, 0.04f, 0.09f});
    p.boxes.push_back(Box{cx + (r - 0.025f) * c, cy + (r - 0.025f) * s, a, 0.05f, 0.05f, 0.0f, 0.06f});
  }
  p.boxes.push_back(Box{cx, cy, yaw, 0.05f, 0.05f, 0.09f, 0.45f});   // 기둥
  p.boxes.push_back(Box{cx, cy, yaw, 0.46f, 0.46f, 0.45f, 0.52f});   // 좌판
  float bx, by;
  to_world(cx, cy, yaw, -0.21f, 0.f, &bx, &by);                       // 등받이 (의자 앞 = yaw 방향)
  p.boxes.push_back(Box{bx, by, yaw, 0.04f, 0.44f, 0.55f, 0.95f});
  return p;
}

struct Cluster {
  float cx, cy, yaw, half_l, half_w;   // 의자까지 든 바깥 틀 (무리 틀 반 길이·반 폭)
  std::vector<Piece> pieces;
};

Cluster make_cluster(float cx, float cy, float yaw, std::mt19937& rng) {
  std::uniform_real_distribution<float> u(0.f, 1.f);
  Cluster C{cx, cy, yaw, 0.f, 0.f, {}};
  const float un = u(rng);
  const int n = un < 0.5f ? 1 : un < 0.85f ? 2 : 3;   // 작은 방에도 들어가게 1 개가 반 (잠정)
  const float w = 1.2f + 0.4f * u(rng), d = 0.6f + 0.2f * u(rng), top = 0.72f + 0.04f * u(rng);
  float reach = 0.f;
  for (int row = 0; row < 2; ++row)
    for (int i = 0; i < n; ++i) {
      const float lx = ((float)i - 0.5f * (float)(n - 1)) * w, ly = (row ? 0.5f : -0.5f) * d;
      float x, y;
      to_world(cx, cy, yaw, lx, ly, &x, &y);
      C.pieces.push_back(make_desk(x, y, yaw, w, d, top));
      // 의자: 책상 바깥쪽 (row 0 은 −y, 1 은 +y), 책상을 바라봄
      float off = 0.15f + 0.30f * u(rng) + 0.30f;   // 책상 끝 → 의자 가운데 (받침 반지름 몫 0.30)
      float jl = (u(rng) - 0.5f) * 0.5f * w;
      if (u(rng) < 0.2f) { off += 0.2f + 0.3f * u(rng); jl += (u(rng) - 0.5f) * 0.6f; }   // 통로로 밀려난 의자
      const float sy = row ? 1.f : -1.f;
      float chx, chy;
      to_world(cx, cy, yaw, lx + jl, sy * (d + off), &chx, &chy);
      const float face = yaw + (row ? -1.5707963f : 1.5707963f) + (u(rng) - 0.5f) * 1.396f;   // ±40°
      C.pieces.push_back(make_chair(chx, chy, face, rng));
      reach = std::max(reach, d + off + 0.35f);
    }
  C.half_l = 0.5f * (float)n * w + 0.35f;
  C.half_w = reach;
  return C;
}

bool rect_overlap(const Cluster& a, const Cluster& b, float gap) {   // 두 기운 틀이 gap 만큼 띄워도 겹치나 (분리 축)
  const Cluster* q[2] = {&a, &b};
  for (int t = 0; t < 2; ++t)
    for (int ax = 0; ax < 2; ++ax) {
      const float ang = q[t]->yaw + (ax ? 1.5707963f : 0.f), ux = std::cos(ang), uy = std::sin(ang);
      float lo[2], hi[2];
      for (int r = 0; r < 2; ++r) {
        const Cluster& C = *q[r];
        const float c = std::cos(C.yaw), s = std::sin(C.yaw);
        const float ext = std::fabs(c * ux + s * uy) * C.half_l + std::fabs(-s * ux + c * uy) * C.half_w;
        const float m = C.cx * ux + C.cy * uy;
        lo[r] = m - ext;
        hi[r] = m + ext;
      }
      if (hi[0] + gap <= lo[1] || hi[1] + gap <= lo[0]) return false;
    }
  return true;
}

int floor_cells(const sim::World& w) {
  int n = 0;
  for (size_t k = 0; k < w.grid.size(); ++k) n += (w.grid[k] & sim::kCellFloor) != 0;
  return n;
}

}  // namespace

int main(int argc, char** argv) {
  std::string env_path = SIM_CONFIG_DIR "/env.yaml", in, out, overlay, ppm;
  unsigned seed = 1;
  int max_clusters = 6;
  float fill = 0.25f;   // 무리 틀 넓이 합 / 갈 수 있는 바닥 상한 — 잠정 (1116 강의실: 무리 6 개가 바닥 약 140 m² 의 0.3 쯤)
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    auto nx = [&]() -> std::string { if (i + 1 >= argc) throw std::runtime_error(a + " 뒤에 값"); return argv[++i]; };
    try {
      if (a == "--in") in = nx();
      else if (a == "--out") out = nx();
      else if (a == "--overlay") overlay = nx();
      else if (a == "--seed") seed = (unsigned)std::stoul(nx());
      else if (a == "--clusters") max_clusters = std::stoi(nx());
      else if (a == "--fill") fill = std::stof(nx());
      else if (a == "--env") env_path = nx();
      else if (a == "--ppm") ppm = nx();
      else { std::fprintf(stderr, "모르는 인자 %s\n", a.c_str()); return 2; }
    } catch (const std::exception& e) { std::fprintf(stderr, "scene_office: %s\n", e.what()); return 2; }
  }
  if (in.empty() || (out.empty() && overlay.empty())) {
    std::fprintf(stderr, "scene_office --in a.scene (--out b.scene | --overlay b_ov.scene) [--seed 1] [--clusters 6] [--fill 0.25] [--env env.yaml] [--ppm 그림.ppm]\n");
    return 2;
  }
  try {
    const sim::EnvHostCfg hc = sim::load_env_cfg(env_path);
    const float res = hc.cfg.res;
    const sim::Scene base = sim::load_scene(in);
    const sim::World w0 = sim::build_world("base", base, hc);
    const int f0 = floor_cells(w0);
    // 벽(막힌 칸)까지 거리 (칸 가운데, 4 m 까지) — 무리 자리 고르기
    std::vector<float> clr((size_t)w0.gw * w0.gh, 0.f);
    {
      std::vector<int> blk;
      for (int k = 0; k < w0.gw * w0.gh; ++k) if (w0.grid[k] & sim::kCellBlocked) blk.push_back(k);
      const int R = (int)(4.f / res);
      for (int k = 0; k < w0.gw * w0.gh; ++k) {
        if (!(w0.grid[k] & sim::kCellFloor)) continue;
        const int x = k % w0.gw, y = k / w0.gw;
        int b2 = R * R;
        for (int dy = -R; dy <= R; ++dy) {
          const int v = y + dy;
          if (v < 0 || v >= w0.gh || dy * dy >= b2) continue;
          for (int dx = -R; dx <= R; ++dx) {
            const int u = x + dx;
            if (u < 0 || u >= w0.gw || dx * dx + dy * dy >= b2) continue;
            if (w0.grid[(size_t)v * w0.gw + u] & sim::kCellBlocked) b2 = dx * dx + dy * dy;
          }
        }
        clr[(size_t)k] = std::sqrt((float)b2) * res;
      }
    }
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> u(0.f, 1.f);
    std::vector<int> spots;
    for (int k = 0; k < w0.gw * w0.gh; ++k) if (clr[(size_t)k] >= 1.0f) spots.push_back(k);
    if (spots.empty()) throw std::runtime_error("무리를 놓을 넓은 바닥(벽에서 1 m)이 없음");
    sim::Scene s = base;
    std::vector<Cluster> placed;
    float area = 0.f;
    int tries = 0, n_bad_world = 0, n_bad_cut = 0;
    const auto inside = [&](float wx, float wy, float margin) {
      const int x = (int)std::floor((wx - w0.ox) / res), y = (int)std::floor((wy - w0.oy) / res);
      return x >= 0 && y >= 0 && x < w0.gw && y < w0.gh && (w0.grid[(size_t)y * w0.gw + x] & sim::kCellFloor) && clr[(size_t)y * w0.gw + x] >= margin;
    };
    int f_prev = f0;
    while ((int)placed.size() < max_clusters && area < fill * (float)f0 * res * res && tries < 2000) {
      ++tries;
      const int k = spots[(size_t)(u(rng) * (float)spots.size()) % spots.size()];
      const float cx = w0.ox + ((float)(k % w0.gw) + 0.5f) * res, cy = w0.oy + ((float)(k / w0.gw) + 0.5f) * res;
      const float yaw = (u(rng) < 0.5f ? 0.f : 1.5707963f) + (u(rng) - 0.5f) * 0.17f;   // 벽에 맞춘 줄 ± 5°
      Cluster C = make_cluster(cx, cy, yaw, rng);
      const float gap = 0.45f + 0.75f * u(rng);   // 무리 사이·벽 통로
      bool ok = std::hypot(cx - base.start[0], cy - base.start[1]) > std::hypot(C.half_l, C.half_w) + 1.2f;
      // 틀 안 전체(0.1 m 간격)가 원래 갈 수 있는 바닥이고 벽에서 0.3 m, 테두리는 통로 반만큼 — 벽을 가로지르는 무리를 막음
      const float edge_m = std::max(0.3f, 0.5f * gap);
      for (float lx = -C.half_l; lx <= C.half_l + 1e-3f && ok; lx += 0.1f)
        for (float ly = -C.half_w; ly <= C.half_w + 1e-3f && ok; ly += 0.1f) {
          float wx, wy;
          to_world(C.cx, C.cy, C.yaw, lx, ly, &wx, &wy);
          const bool rim = lx < -C.half_l + 0.1f || lx > C.half_l - 0.1f || ly < -C.half_w + 0.1f || ly > C.half_w - 0.1f;
          ok = inside(wx, wy, rim ? edge_m : 0.3f);
        }
      for (const Cluster& o : placed) ok = ok && !rect_overlap(C, o, gap);
      if (!ok) continue;
      sim::Scene t = s;
      int ni = (int)t.obj_names.size();
      for (const Piece& p : C.pieces) add_boxes(t, p.boxes, "furniture_" + std::to_string(ni++) + "_office_" + p.kind);
      sim::build_bvh(t);
      sim::compute_obj_size(t);
      sim::World w1;
      try {
        w1 = sim::build_world("variant", t, hc);
      } catch (const std::exception&) { ++n_bad_world; continue; }
      const int f1 = floor_cells(w1);
      const float lost = (float)(f_prev - f1) * res * res, box = 4.f * C.half_l * C.half_w;
      if (lost > 1.2f * box + 0.5f) { ++n_bad_cut; continue; }   // 다른 바닥을 고립시킴
      s = t;
      f_prev = f1;
      area += box;
      placed.push_back(C);
    }
    const sim::World wf = sim::build_world("final", s, hc);
    int n_desk = 0, n_chair = 0;
    for (const Cluster& C : placed)
      for (const Piece& p : C.pieces) (p.kind == "desk" ? n_desk : n_chair) += 1;
    std::printf("scene_office %s: 무리 %zu 개 (책상 %d · 의자 %d, 시도 %d · 닫힘·출발점 실패 %d · 고립 %d) · 무리 틀 %.1f m² · 갈 수 있는 바닥 %.1f → %.1f m²\n", in.c_str(), placed.size(), n_desk,
                n_chair, tries, n_bad_world, n_bad_cut, area, f0 * res * res, floor_cells(wf) * res * res);
    if (placed.empty()) throw std::runtime_error("무리를 하나도 못 놓음");
    if (!ppm.empty()) {
      FILE* f = std::fopen(ppm.c_str(), "wb");
      if (!f) throw std::runtime_error("못 씀: " + ppm);
      std::fprintf(f, "P6\n%d %d\n255\n", wf.gw, wf.gh);
      for (int y = wf.gh - 1; y >= 0; --y)
        for (int x = 0; x < wf.gw; ++x) {
          const uint8_t g = wf.grid[(size_t)y * wf.gw + x];
          uint8_t c[3] = {200, 200, 200};
          if (g & sim::kCellFloor) c[0] = c[1] = c[2] = 255;
          if (g & sim::kCellBlocked) c[0] = c[1] = c[2] = 90;
          if ((g & sim::kCellBlocked) && !(w0.grid[(size_t)y * w0.gw + x] & sim::kCellBlocked)) { c[0] = 30; c[1] = 80; c[2] = 240; }
          std::fwrite(c, 1, 3, f);
        }
      std::fclose(f);
    }
    if (!out.empty()) { sim::save_scene(s, out); std::printf("씀: %s\n", out.c_str()); }
    if (!overlay.empty()) {   // 더한 가구만 (바탕 = --in) — 목록 줄 "바탕.scene+덧.scene"
      sim::Scene ov;
      int ni = (int)base.obj_names.size();
      for (const Cluster& C : placed)
        for (const Piece& p : C.pieces) add_boxes(ov, p.boxes, "furniture_" + std::to_string(ni++) + "_office_" + p.kind);
      sim::build_bvh(ov);
      sim::compute_obj_size(ov);
      ov.start[0] = base.start[0];
      ov.start[1] = base.start[1];
      sim::save_scene(ov, overlay);
      std::printf("덧 장면 씀: %s (삼각형 %zu)\n", overlay.c_str(), ov.tris.size());
    }
  } catch (const std::exception& e) {
    std::fprintf(stderr, "scene_office: %s\n", e.what());
    return 1;
  }
  return 0;
}
