// 화재 대응 LIMO GPU 환경 — 첫 지도 만들기(덮기) 탐사. 환경 하나 = 스레드 하나.
// CPU 참조판(src/fenv_ref.cpp)과 GPU 커널(src/fenv_kernel.cu)이 이 파일을 같이 쓰고, detmath.h 와
// --fmad=false / -ffp-contract=off 로 비트까지 같게 한다(object-memory-robot training/RL/env 와 같은 방식).
//
// 한 판: 맵 묶음에서 맵 하나 → 시작 칸 → 불 1–2 개를 도달 영역 무작위 칸에 놓음 → 행동(중간 목표점)마다 0.5 s 주행.
// 행동 48 = 로봇 기준 방향 16 × 거리 {0.5, 1, 2} m. 추종기가 속도·가속 한도 안에서 목표점으로 간다(Nav2 대신).
// 라이다 248 빔 / 220° / 12 m 를 칸 DDA 로 쏴서 아는 지도(know)를 키우고, 3 m 안에서 맞은 표면 칸의 횟수를 센다.
// 표면 칸이 K::cov_hits 번 맞으면 "덮음". 덮은 비율 ≥ K::cov_goal 이면 성공.
//
// know[칸] (환경마다, 그 판 맵 크기만큼 씀)
//   bit0 본 적 있음 · bit1 막힘으로 봄 · bit2-7 횟수(표면 칸: 3 m 안에서 맞은 횟수, 빈 칸: 로봇이 지나간 횟수), 63 에서 멈춤
#pragma once
#include <cstdint>

#include "detmath.h"
#include "fmap.h"

namespace fe {

struct K {
  // 시간
  static constexpr float dt = 0.05f;          // 서브스텝(s)
  static constexpr int substeps = 10;         // 행동 하나 = 0.5 s
  // LIMO 몸통·구동 (가정: 속도·가속 한도는 Nav2 설정을 재서 바꿀 것)
  static constexpr float v_max = 0.5f;        // m/s
  static constexpr float w_max = 1.5f;        // rad/s
  static constexpr float a_v = 1.0f;          // m/s²
  static constexpr float a_w = 3.0f;          // rad/s²
  static constexpr float body_r = 0.13f;      // 충돌 원 반지름(m). 몸통 322×220 mm 의 안쪽 원 + 여유 — fmap ROBOT_CLEARANCE 0.15 와 짝
  static constexpr float k_heading = 2.0f;    // 추종기 방향 이득
  static constexpr float slow_r = 0.4f;       // 이 거리 안에서 감속(m)
  // 라이다 (교수님 Webots LimoFourDiff.proto 의 laser 와 같은 값)
  static constexpr int lidar_n = 248;
  static constexpr float lidar_fov = 3.839724f;   // 220°
  static constexpr float lidar_max = 12.0f;
  // 덮기
  static constexpr float cov_range = 3.0f;    // 이 거리 안에서 맞은 것만 센다(m)
  static constexpr int cov_hits = 3;          // 덮음 문턱(맞은 횟수)
  static constexpr float cov_goal = 0.95f;    // 성공 덮은 비율
  // 불·카메라 (depth 카메라 시야각 66.5°)
  static constexpr int max_fire = 2;
  static constexpr float cam_half_fov = 0.58032f;
  static constexpr float fire_range = 5.0f;
  static constexpr float fire_min_start = 2.0f;   // 시작 칸에서 불까지 최소(m)
  // 행동
  static constexpr int n_dir = 16;
  static constexpr int n_dist = 3;
  static constexpr int n_actions = n_dir * n_dist;
  // 보상 (spec.md "보상 항목" 표. 구멍 감점은 아직 없음)
  static constexpr float r_cov = 10.0f;       // 덮은 비율 1.0 당
  static constexpr float r_done = 5.0f;       // 성공
  static constexpr float r_time = -0.005f;    // 행동마다
  static constexpr float r_revisit = -0.01f;  // 지나간 칸을 다시 지나감
  static constexpr float r_still = -0.01f;    // 0.05 m 도 못 감
  static constexpr float r_coll = -2.0f;      // 충돌(판 끝)
  static constexpr float r_fire = 1.0f;       // 불을 처음 카메라에 담음
  static constexpr float r_short = -0.005f;   // 목표점이 벽 너머라 앞에서 잘림
};

enum Done : uint32_t { RUNNING = 0, SUCCESS = 1, COLLISION = 2, TIMEOUT = 3 };

struct Core {
  float x, y, th, v, w;
  float gx, gy;             // 지금 목표점(세계)
  uint32_t map, step, max_steps;
  uint32_t covered, n_target;
  uint32_t n_fire, fire_seen;
  float fx[K::max_fire], fy[K::max_fire], fr[K::max_fire];
  uint32_t done, episodes;
  uint64_t rng;
};

struct Map {
  const uint8_t* c;
  int w, h;
  float ox, oy;
};

DEV Map map_of(const fm::View& v, uint32_t m) {
  Map r;
  r.c = v.cell + v.off[m];
  r.w = (int)v.w[m];
  r.h = (int)v.h[m];
  r.ox = v.ox[m];
  r.oy = v.oy[m];
  return r;
}
DEV uint8_t at(const Map& m, int i, int j) {
  return (i < 0 || j < 0 || i >= m.w || j >= m.h) ? fm::OUTSIDE : m.c[j * m.w + i];
}
DEV int cell_i(const Map& m, float x) { return (int)floorf((x - m.ox) / fm::RES); }
DEV int cell_j(const Map& m, float y) { return (int)floorf((y - m.oy) / fm::RES); }
DEV uint32_t rand_below(uint64_t& s, uint32_t n) { return (uint32_t)((dm::splitmix64(s) >> 32) % (n ? n : 1)); }

// 몸통 원이 막힌 칸(정사각형)과 겹치는가
DEV bool body_hits(const Map& m, float x, float y) {
  const float r = K::body_r;
  int i0 = cell_i(m, x - r), i1 = cell_i(m, x + r);
  int j0 = cell_j(m, y - r), j1 = cell_j(m, y + r);
  for (int j = j0; j <= j1; ++j) {
    for (int i = i0; i <= i1; ++i) {
      if (!fm::blocks(at(m, i, j))) continue;
      float cx0 = m.ox + (float)i * fm::RES, cy0 = m.oy + (float)j * fm::RES;
      float px = dm::clampf(x, cx0, cx0 + fm::RES) - x;
      float py = dm::clampf(y, cy0, cy0 + fm::RES) - y;
      if (px * px + py * py < r * r) return true;
    }
  }
  return false;
}

// 칸 DDA 광선. 지나가는 칸마다 visit(i, j, t_entry) 를 부르고, visit 이 true 를 돌려주면 멈춘다.
// t_entry = 그 칸에 들어간 거리(m). 첫 칸(로봇 칸)은 0.
template <class F>
DEV void march(const Map& m, float x, float y, float dx, float dy, float tmax, F&& visit) {
  int i = cell_i(m, x), j = cell_j(m, y);
  const float big = 1e30f;
  int si = dx > 0.f ? 1 : -1, sj = dy > 0.f ? 1 : -1;
  float tdx = dm::absf(dx) < 1e-9f ? big : fm::RES / dm::absf(dx);
  float tdy = dm::absf(dy) < 1e-9f ? big : fm::RES / dm::absf(dy);
  float nx = m.ox + (float)(i + (dx > 0.f ? 1 : 0)) * fm::RES;
  float ny = m.oy + (float)(j + (dy > 0.f ? 1 : 0)) * fm::RES;
  float tx = dm::absf(dx) < 1e-9f ? big : (nx - x) / dx;
  float ty = dm::absf(dy) < 1e-9f ? big : (ny - y) / dy;
  float t = 0.f;
  for (int guard = 0; guard < 4096; ++guard) {
    if (i < 0 || j < 0 || i >= m.w || j >= m.h) return;
    if (visit(i, j, t)) return;
    if (tx < ty) { t = tx; tx += tdx; i += si; } else { t = ty; ty += tdy; j += sj; }
    if (t > tmax) return;
  }
}

// 라이다 한 바퀴: 아는 지도 키우기 + 덮음 세기. 새로 덮은 표면 칸 수를 돌려준다.
DEV uint32_t scan(const Core& c, const Map& m, uint8_t* know) {
  uint32_t newly = 0;
  for (int b = 0; b < K::lidar_n; ++b) {
    float a = c.th - 0.5f * K::lidar_fov + K::lidar_fov * ((float)b / (float)(K::lidar_n - 1));
    float s, co;
    dm::sincosf_d(a, &s, &co);
    march(m, c.x, c.y, co, s, K::lidar_max, [&](int i, int j, float t) {
      int k = j * m.w + i;
      uint8_t cv = m.c[k];
      uint8_t kv = know[k];
      if (!fm::blocks(cv)) { know[k] = (uint8_t)(kv | 1u); return false; }
      kv = (uint8_t)(kv | 3u);
      if ((cv & fm::TARGET) && t <= K::cov_range) {
        uint32_t n = kv >> 2;
        if (n < 63u) {
          ++n;
          if (n == (uint32_t)K::cov_hits) ++newly;
          kv = (uint8_t)((kv & 3u) | (n << 2));
        }
      }
      know[k] = kv;
      return true;
    });
  }
  return newly;
}

// 불이 카메라 시야·거리 안이고 가리지 않는가
DEV bool fire_visible(const Core& c, const Map& m, int f) {
  float dx = c.fx[f] - c.x, dy = c.fy[f] - c.y;
  float d = sqrtf(dx * dx + dy * dy);
  if (d > K::fire_range || d < 1e-4f) return d < 1e-4f;
  float e = dm::wrap_pi(dm::atan2f_d(dy, dx) - c.th);
  if (dm::absf(e) > K::cam_half_fov) return false;
  bool clear = true;
  float reach = d - c.fr[f];
  march(m, c.x, c.y, dx / d, dy / d, reach, [&](int i, int j, float t) {
    if (t <= reach && fm::blocks(at(m, i, j))) { clear = false; return true; }
    return false;
  });
  return clear;
}

DEV void reset(Core& c, const fm::View& v, uint8_t* know) {
  c.map = rand_below(c.rng, (uint32_t)v.n_maps);
  Map m = map_of(v, c.map);
  for (int k = 0; k < m.w * m.h; ++k) know[k] = 0;
  uint32_t sk = v.idx[v.start_off[c.map] + rand_below(c.rng, v.start_n[c.map])];
  c.x = m.ox + ((float)(sk % (uint32_t)m.w) + 0.5f) * fm::RES;
  c.y = m.oy + ((float)(sk / (uint32_t)m.w) + 0.5f) * fm::RES;
  c.th = dm::rand_range(c.rng, -dm::kPi, dm::kPi);
  c.v = 0.f;
  c.w = 0.f;
  c.gx = c.x;
  c.gy = c.y;
  c.step = 0;
  float mx = v.area[c.map] * 4.0f;
  c.max_steps = (uint32_t)dm::clampf(mx, 200.f, 2000.f);
  c.n_target = v.n_target[c.map];
  // 불: 1 개(80 %) 또는 2 개, 도달 영역 무작위 칸. 작은 불 70 %(촛불 크기), 큰 불 30 %
  c.n_fire = dm::rand01(c.rng) < 0.8f ? 1u : 2u;
  for (uint32_t f = 0; f < (uint32_t)K::max_fire; ++f) {
    c.fx[f] = c.x; c.fy[f] = c.y; c.fr[f] = 0.f;
    if (f >= c.n_fire) continue;
    for (int tries = 0; tries < 16; ++tries) {
      uint32_t fk = v.idx[v.free_off[c.map] + rand_below(c.rng, v.free_n[c.map])];
      float fx = m.ox + ((float)(fk % (uint32_t)m.w) + 0.5f) * fm::RES;
      float fy = m.oy + ((float)(fk / (uint32_t)m.w) + 0.5f) * fm::RES;
      c.fx[f] = fx; c.fy[f] = fy;
      float ddx = fx - c.x, ddy = fy - c.y;
      if (ddx * ddx + ddy * ddy >= K::fire_min_start * K::fire_min_start) break;
    }
    c.fr[f] = dm::rand01(c.rng) < 0.7f ? dm::rand_range(c.rng, 0.05f, 0.15f) : dm::rand_range(c.rng, 0.3f, 0.8f);
  }
  c.fire_seen = 0;
  c.done = RUNNING;
  c.covered = scan(c, m, know);   // 처음 보는 것은 보상 없이 센다
}

// 행동 하나(0.5 s). 보상을 돌려준다. 판이 끝나면 c.done 이 0 이 아니다(다시 시작은 부른 쪽이 reset).
DEV float step(Core& c, int action, const fm::View& v, uint8_t* know) {
  Map m = map_of(v, c.map);
  float r = K::r_time;
  int dir = action % K::n_dir, di = action / K::n_dir;
  float dist = di == 0 ? 0.5f : (di == 1 ? 1.0f : 2.0f);
  float ang = c.th + (float)dir * (dm::kTwoPi / (float)K::n_dir);
  float s, co;
  dm::sincosf_d(ang, &s, &co);
  // 목표점이 벽 너머면 벽 앞(몸통 반지름 + 한 칸)에서 자른다
  float reach = dist;
  march(m, c.x, c.y, co, s, dist, [&](int i, int j, float t) {
    if (fm::blocks(at(m, i, j))) { reach = dm::maxf(0.f, t - K::body_r - fm::RES); return true; }
    return false;
  });
  if (reach < dist) r += K::r_short;
  c.gx = c.x + reach * co;
  c.gy = c.y + reach * s;

  float px = c.x, py = c.y;
  for (int k = 0; k < K::substeps; ++k) {
    float ex = c.gx - c.x, ey = c.gy - c.y;
    float d = sqrtf(ex * ex + ey * ey);
    float err = d < 1e-4f ? 0.f : dm::wrap_pi(dm::atan2f_d(ey, ex) - c.th);
    float wc = dm::clampf(K::k_heading * err, -K::w_max, K::w_max);
    float vc = d < 0.05f ? 0.f : K::v_max * dm::maxf(0.f, dm::cosf_d(err)) * dm::minf(1.f, d / K::slow_r);
    c.v += dm::clampf(vc - c.v, -K::a_v * K::dt, K::a_v * K::dt);
    c.w += dm::clampf(wc - c.w, -K::a_w * K::dt, K::a_w * K::dt);
    float nth = dm::wrap_pi(c.th + c.w * K::dt);
    float ss, cc;
    dm::sincosf_d(nth, &ss, &cc);
    float nx = c.x + c.v * cc * K::dt, ny = c.y + c.v * ss * K::dt;
    if (body_hits(m, nx, ny)) { c.done = COLLISION; c.v = 0.f; c.w = 0.f; break; }
    c.x = nx; c.y = ny; c.th = nth;
  }

  uint32_t newly = scan(c, m, know);
  c.covered += newly;
  r += K::r_cov * (float)newly / (float)(c.n_target ? c.n_target : 1u);

  for (uint32_t f = 0; f < c.n_fire; ++f) {
    if ((c.fire_seen >> f) & 1u) continue;
    if (fire_visible(c, m, (int)f)) { c.fire_seen |= 1u << f; r += K::r_fire; }
  }

  float mvx = c.x - px, mvy = c.y - py;
  float moved2 = mvx * mvx + mvy * mvy;
  if (moved2 < 0.05f * 0.05f) r += K::r_still;
  int ri = cell_i(m, c.x), rj = cell_j(m, c.y);
  if (ri >= 0 && rj >= 0 && ri < m.w && rj < m.h) {
    int k = rj * m.w + ri;
    uint32_t n = (uint32_t)know[k] >> 2;
    if (n >= 1u && moved2 >= 0.1f * 0.1f) r += K::r_revisit;
    if (n < 63u) know[k] = (uint8_t)((know[k] & 3u) | ((n + 1u) << 2));
  }

  ++c.step;
  if (c.done == COLLISION) {
    r += K::r_coll;
  } else if ((float)c.covered >= K::cov_goal * (float)c.n_target) {
    c.done = SUCCESS;
    r += K::r_done;
  } else if (c.step >= c.max_steps) {
    c.done = TIMEOUT;
  }
  return r;
}

}  // namespace fe
