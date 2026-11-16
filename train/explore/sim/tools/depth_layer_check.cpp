// 깊이 장애물 층 해상도 재기: 닿은 자리(몸통 8 cm 안 막힌 칸 — 정답 격자)를 그 앞 자세들에서 깊이 카메라가 몇 점으로 보는지, 픽셀 고르는 촘촘함마다.
//   depth_layer_check <장면.scene> --hit x y θ --pose x y θ [--pose …] [--env env.yaml]
// 촘촘함: 지금 관측(64 × 40 — 층·collision_monitor 가 쓰는 것) · 64 열 × 100 행 · 64 × 200 · 160 × 200 · 640 × 400(센서 전부).
// 점 = 시뮬 깊이 모델(sensor_model.h depth_pixel_real, 구멍·잡음 포함, 바닥 구멍 표 가운데)로 그 픽셀 깊이 → 몸통 띠(collision_band) 안 · 센서 수평 2.5 m 안(층 표시 거리) 인 점.
// 자세마다: 목표 칸이 카메라 수평 시야(±fov/2) 안인 몫, 촘촘함마다 목표 칸(한 칸 둘레)에 든 점 수 · 점이 든 목표 칸 수. 층은 점이 든 칸을 막힘으로 찍으므로 "칸 수 > 0" 이면 층이 안다.
#include <cmath>
#include <cstdio>
#include <cstring>
#include <set>
#include <string>
#include <vector>

#include "env_api.h"

namespace {
struct P3 { float x, y, th; };
}

int main(int argc, char** argv) {
  std::string scene, env_path = SIM_CONFIG_DIR "/env.yaml";
  P3 hit{0, 0, 0};
  bool have_hit = false;
  std::vector<P3> poses;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    auto f = [&](int k) { return std::stof(argv[i + k]); };
    if (a == "--hit" && i + 3 < argc) { hit = P3{f(1), f(2), f(3) * 0.017453293f}; have_hit = true; i += 3; }
    else if (a == "--pose" && i + 3 < argc) { poses.push_back(P3{f(1), f(2), f(3) * 0.017453293f}); i += 3; }
    else if (a == "--env" && i + 1 < argc) env_path = argv[++i];
    else if (a[0] != '-') scene = a;
    else { std::fprintf(stderr, "모르는 인자 %s\n", a.c_str()); return 2; }
  }
  if (scene.empty() || !have_hit || poses.empty()) {
    std::fprintf(stderr, "depth_layer_check <장면.scene> --hit x y θ(도) --pose x y θ(도) [--pose …]\n");
    return 2;
  }
  try {
    const sim::EnvHostCfg hc = sim::load_env_cfg(env_path);
    const sim::EnvCfg& c = hc.cfg;
    const sim::Scene sc = sim::load_scene(scene);
    const sim::World W = sim::build_world("w", sc, hc, false);
    const sim::SceneView view = W.scene.view();
    const sim::DepthSpec& D = c.depth.spec;
    // 목표 칸: 닿은 자세 몸통에서 8 cm 안 막힌 칸
    std::set<int> tgt;
    {
      const float c0 = std::cos(hit.th), s0 = std::sin(hit.th);
      const int R = (int)std::ceil(0.45f / c.res);
      const int cx = (int)std::floor((hit.x - W.ox) / c.res), cy = (int)std::floor((hit.y - W.oy) / c.res);
      for (int dy = -R; dy <= R; ++dy)
        for (int dx = -R; dx <= R; ++dx) {
          const int x = cx + dx, y = cy + dy;
          if (x < 0 || y < 0 || x >= W.gw || y >= W.gh || !(W.grid[(size_t)y * W.gw + x] & sim::kCellBlocked)) continue;
          const float wx = W.ox + (x + 0.5f) * c.res - hit.x, wy = W.oy + (y + 0.5f) * c.res - hit.y;
          const float lx = c0 * wx + s0 * wy, ly = -s0 * wx + c0 * wy;
          if (std::hypot(std::max(0.f, std::fabs(lx) - c.half_len), std::max(0.f, std::fabs(ly) - c.half_wid)) <= 0.08f) tgt.insert(y * W.gw + x);
        }
    }
    std::printf("%s: 닿은 자세 (%.2f, %.2f, %.0f°) · 목표 칸 %zu · 몸통 띠 %.2f~%.2f m · 깊이 %d × %d (fx %.1f)\n", scene.c_str(), hit.x, hit.y, hit.th * 57.2958f, tgt.size(), hc.band_lo,
                hc.band_hi, D.w, D.h, D.fx);
    const auto near_tgt = [&](int x, int y) {
      for (int dy = -1; dy <= 1; ++dy)
        for (int dx = -1; dx <= 1; ++dx)
          if (tgt.count((y + dy) * W.gw + (x + dx))) return true;
      return false;
    };
    const int vc[5] = {64, 64, 64, 160, 640}, vr[5] = {40, 100, 200, 200, 400};
    const float hfov = std::atan((0.5f * (float)D.w) / D.fx);
    for (const P3& p : poses) {
      float ts, tc;
      dm::sincosf_d(p.th, &ts, &tc);
      const sim::Pose cam{{tc * c.cam_cp, -ts, tc * c.cam_sp, ts * c.cam_cp, tc, ts * c.cam_sp, -c.cam_sp, 0.f, c.cam_cp},
                          sim::V3{(p.x + tc * c.cam_x) - ts * c.cam_y, (p.y + ts * c.cam_x) + tc * c.cam_y, c.cam_z}};   // observations.h depth_camera 와 같은 꼴
      int in_fov = 0;
      float dmin = 99.f;
      for (int k : tgt) {
        const float wx = W.ox + (k % W.gw + 0.5f) * c.res - cam.p.x, wy = W.oy + (k / W.gw + 0.5f) * c.res - cam.p.y;
        const float br = std::remainder(std::atan2(wy, wx) - p.th, 6.2831853f);
        in_fov += std::fabs(br) <= hfov;
        dmin = std::min(dmin, std::hypot(wx, wy));
      }
      std::printf("  자세 (%.2f, %.2f, %4.0f°) 목표까지 %.2f m · 시야 안 목표 칸 %d/%zu |", p.x, p.y, p.th * 57.2958f, dmin, in_fov, tgt.size());
      for (int v = 0; v < 5; ++v) {
        long pts = 0;
        std::set<int> cells;
        for (int r = 0; r < vr[v]; ++r)
          for (int q = 0; q < vc[v]; ++q) {
            const float u = ((float)q + 0.5f) * ((float)D.w / vc[v]) - 0.5f, vv = ((float)r + 0.5f) * ((float)D.h / vr[v]) - 0.5f;
            const float z = sim::depth_pixel_real(view, c.depth, cam, u, vv, (uint64_t)(r * 7919 + q + 1), 0.5f);
            if (!(z > 0.f)) continue;
            const float X = z, Y = z * (-(u - D.cx) / D.fx), Z = z * (-(vv - D.cy) / D.fy);
            const float bx = c.cam_x + (c.cam_cp * X + c.cam_sp * Z), by = c.cam_y + Y, bz = c.cam_z + (c.cam_cp * Z - c.cam_sp * X);
            if (bz < c.cam_min_h || bz > c.cam_max_h || std::hypot(bx - c.cam_x, by - c.cam_y) > c.cam_obs_range) continue;
            const float wx = p.x + tc * bx - ts * by, wy = p.y + ts * bx + tc * by;
            const int x = (int)std::floor((wx - W.ox) / c.res), y = (int)std::floor((wy - W.oy) / c.res);
            if (x < 1 || y < 1 || x >= W.gw - 1 || y >= W.gh - 1 || !near_tgt(x, y)) continue;
            ++pts;
            cells.insert(y * W.gw + x);
          }
        std::printf(" %d×%d 점 %ld 칸 %zu |", vc[v], vr[v], pts, cells.size());
      }
      std::printf("\n");
    }
  } catch (const std::exception& e) {
    std::fprintf(stderr, "depth_layer_check: %s\n", e.what());
    return 1;
  }
  return 0;
}
