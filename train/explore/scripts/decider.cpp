// decider.h 구현: 지도 사진(PNG) · 요청 본문(JSON) · 동시 요청(libcurl multi).
#include "decider.h"

#ifndef DEC_NO_CURL   // 실물 노드(limo_fire_rl low_rl)는 사진·본문만 쓰고 HTTP 는 자체 소켓 (로봇에 libcurl 개발 헤더 없음)
#include <curl/curl.h>
#endif
#include <zlib.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <stdexcept>

#include <nlohmann/json.hpp>

namespace dec {

namespace {

using json = nlohmann::ordered_json;   // 선택지 차례(A, B, …)를 그대로 보냄

// 5 × 7 글자 A ~ H (줄마다 왼쪽이 높은 비트)
constexpr uint8_t kGlyph[8][7] = {
    {0x0e, 0x11, 0x11, 0x1f, 0x11, 0x11, 0x11}, {0x1e, 0x11, 0x11, 0x1e, 0x11, 0x11, 0x1e}, {0x0e, 0x11, 0x10, 0x10, 0x10, 0x11, 0x0e}, {0x1e, 0x11, 0x11, 0x11, 0x11, 0x11, 0x1e},
    {0x1f, 0x10, 0x10, 0x1e, 0x10, 0x10, 0x1f}, {0x1f, 0x10, 0x10, 0x1e, 0x10, 0x10, 0x10}, {0x0e, 0x11, 0x10, 0x17, 0x11, 0x11, 0x0e}, {0x11, 0x11, 0x11, 0x1f, 0x11, 0x11, 0x11},
};

struct Img {
  int w, h;
  std::vector<uint8_t> rgb;
  void set(int x, int y, uint8_t r, uint8_t g, uint8_t b) {
    if (x < 0 || y < 0 || x >= w || y >= h) return;
    uint8_t* p = &rgb[((size_t)y * w + x) * 3];
    p[0] = r; p[1] = g; p[2] = b;
  }
};

void put_be32(std::vector<uint8_t>& o, uint32_t v) {
  for (int s = 24; s >= 0; s -= 8) o.push_back((uint8_t)(v >> s));
}
void chunk(std::vector<uint8_t>& o, const char* type, const std::vector<uint8_t>& d) {
  put_be32(o, (uint32_t)d.size());
  const size_t at = o.size();
  o.insert(o.end(), type, type + 4);
  o.insert(o.end(), d.begin(), d.end());
  put_be32(o, (uint32_t)crc32(0L, o.data() + at, (uInt)(o.size() - at)));
}
std::vector<uint8_t> png_of(const Img& im) {
  std::vector<uint8_t> raw;
  raw.reserve((size_t)im.h * (im.w * 3 + 1));
  for (int y = 0; y < im.h; ++y) {
    raw.push_back(0);   // 줄 거르개 없음
    raw.insert(raw.end(), im.rgb.begin() + (size_t)y * im.w * 3, im.rgb.begin() + (size_t)(y + 1) * im.w * 3);
  }
  uLongf zn = compressBound((uLong)raw.size());
  std::vector<uint8_t> z(zn);
  if (compress2(z.data(), &zn, raw.data(), (uLong)raw.size(), 6) != Z_OK) throw std::runtime_error("PNG 압축 실패");
  z.resize(zn);
  std::vector<uint8_t> o{0x89, 'P', 'N', 'G', 0x0d, 0x0a, 0x1a, 0x0a}, hd;
  put_be32(hd, (uint32_t)im.w);
  put_be32(hd, (uint32_t)im.h);
  hd.insert(hd.end(), {8, 2, 0, 0, 0});   // 8 비트 RGB
  chunk(o, "IHDR", hd);
  chunk(o, "IDAT", z);
  chunk(o, "IEND", {});
  return o;
}

// 후보 표시: 정확한 자리에 작은 점, 글자는 꼬리표(빨간 동그라미)에 — 꼬리표는 다른 꼬리표·다른 후보 점·로봇과 안 겹치는 자리로 밀어 놓고 점과 선으로 잇는다.
// (예전엔 글자 동그라미를 그 자리에 바로 그려 후보가 몰리면 글자가 서로 가렸다)
constexpr int kTagR = 10, kDotR = 3;   // 꼬리표 · 점 반지름 (px)
void draw_line(Img& im, int x0, int y0, int x1, int y1, uint8_t r, uint8_t g, uint8_t b) {
  const int n = std::max({std::abs(x1 - x0), std::abs(y1 - y0), 1});
  for (int i = 0; i <= n; ++i) im.set(x0 + (x1 - x0) * i / n, y0 + (y1 - y0) * i / n, r, g, b);
}
void draw_disc(Img& im, int cu, int cv, int rad, uint8_t r, uint8_t g, uint8_t b) {
  for (int dy = -rad - 1; dy <= rad + 1; ++dy)
    for (int dx = -rad - 1; dx <= rad + 1; ++dx) {
      const int d2 = dx * dx + dy * dy;
      if (d2 <= rad * rad) im.set(cu + dx, cv + dy, r, g, b);
      else if (d2 <= (rad + 1) * (rad + 1) + 1) im.set(cu + dx, cv + dy, 255, 255, 255);
    }
}
// 후보 모드 꼬리표 (MapMem::clear_tags): 글자마다 다른 진한 색 + 헷갈리지 않는 글꼴 (A 는 뾰족, H 는 가로줄 가운데, D 는 각진 오른쪽)
constexpr uint8_t kTagCol[8][3] = {{200, 20, 20}, {20, 60, 210}, {0, 140, 40}, {130, 30, 170}, {215, 110, 0}, {0, 130, 140}, {120, 70, 20}, {200, 0, 130}};
constexpr uint8_t kGlyph2[8][7] = {
    {0x04, 0x0a, 0x11, 0x11, 0x1f, 0x11, 0x11}, {0x1e, 0x11, 0x11, 0x1e, 0x11, 0x11, 0x1e}, {0x0f, 0x10, 0x10, 0x10, 0x10, 0x10, 0x0f}, {0x1c, 0x12, 0x11, 0x11, 0x11, 0x12, 0x1c},
    {0x1f, 0x10, 0x10, 0x1e, 0x10, 0x10, 0x1f}, {0x1f, 0x10, 0x10, 0x1e, 0x10, 0x10, 0x10}, {0x0f, 0x10, 0x10, 0x13, 0x11, 0x11, 0x0f}, {0x11, 0x11, 0x11, 0x1f, 0x11, 0x11, 0x11},
};
bool g_clear_tags = false;   // render_img 동안만 (MapMem::clear_tags)
void draw_tag(Img& im, int cu, int cv, char key) {
  const int g = std::clamp(key - 'A', 0, 7);
  if (g_clear_tags) draw_disc(im, cu, cv, kTagR, kTagCol[g][0], kTagCol[g][1], kTagCol[g][2]);
  else draw_disc(im, cu, cv, kTagR, 220, 30, 30);
  const auto& G = g_clear_tags ? kGlyph2 : kGlyph;
  for (int gy = 0; gy < 7; ++gy)
    for (int gx = 0; gx < 5; ++gx) {
      if (!((G[g][gy] >> (4 - gx)) & 1)) continue;
      for (int q = 0; q < 4; ++q) im.set(cu - 5 + gx * 2 + (q & 1), cv - 7 + gy * 2 + (q >> 1), 255, 255, 255);
    }
}
// dots: 표시할 후보의 점 자리 (x, y) 쌍, show[k] 가 0 이면 그 후보는 안 그림. (rx, ry) 는 피할 자리(로봇; 없으면 음수)
void place_tags(Img& im, const std::vector<Cand>& cand, const std::vector<int>& dots, int rx, int ry, const std::vector<char>& show = {}) {
  static const int kDir[8][2] = {{0, -1}, {1, 0}, {0, 1}, {-1, 0}, {1, -1}, {1, 1}, {-1, 1}, {-1, -1}};
  static const int kRad[6] = {16, 28, 40, 54, 70, 88};   // 밀어 놓는 거리 (px) — 예전 꼴은 앞 셋만
  std::vector<int> tx(cand.size(), 0), ty(cand.size(), 0);
  for (size_t k = 0; k < cand.size(); ++k) {
    if (!show.empty() && !show[k]) continue;
    int best = -(1 << 20), bx = -1, by = 0;
    for (int o = 0; o < (g_clear_tags ? 48 : 24); ++o) {
      const int rad = kRad[o / 8];
      const int f = (kDir[o % 8][0] != 0 && kDir[o % 8][1] != 0) ? 7 : 10;   // 대각선은 0.7 배
      const int x = dots[2 * k] + kDir[o % 8][0] * rad * f / 10, y = dots[2 * k + 1] + kDir[o % 8][1] * rad * f / 10;
      if (x < kTagR + 1 || y < kTagR + 1 || x >= im.w - kTagR - 1 || y >= im.h - kTagR - 1) continue;
      int gap = 1 << 20;   // 가장 가까운 것과의 틈 (px, 클수록 좋음)
      const auto see = [&](int qx, int qy, int need) { gap = std::min(gap, (int)std::lround(std::hypot((double)(x - qx), (double)(y - qy))) - need); };
      for (size_t j = 0; j < cand.size(); ++j) {
        if (!show.empty() && !show[j]) continue;
        if (j != k) see(dots[2 * j], dots[2 * j + 1], kTagR + kDotR + 2);
        if (j < k) see(tx[j], ty[j], 2 * kTagR + 2);
      }
      if (rx >= 0) see(rx, ry, kTagR + 13);
      if (gap > best) { best = gap; bx = x; by = y; }
      if (gap >= 0) break;   // 안 겹치는 첫 자리 (가까운 것부터)
    }
    if (bx < 0) {   // 그림 안에 놓을 자리가 없음: 점 위에
      bx = std::clamp(dots[2 * k], kTagR + 1, im.w - kTagR - 2); by = std::clamp(dots[2 * k + 1], kTagR + 1, im.h - kTagR - 2); }
    tx[k] = bx;
    ty[k] = by;
  }
  for (int k = (int)cand.size() - 1; k >= 0; --k) {   // 선 → 점 → 꼬리표 (0 번이 맨 위에)
    if (!show.empty() && !show[k]) continue;
    const int g = std::clamp(cand[k].key - 'A', 0, 7);
    const uint8_t lr = g_clear_tags ? kTagCol[g][0] : 220, lg = g_clear_tags ? kTagCol[g][1] : 30, lb = g_clear_tags ? kTagCol[g][2] : 30;
    draw_line(im, dots[2 * k], dots[2 * k + 1], tx[k], ty[k], lr, lg, lb);
    draw_line(im, dots[2 * k] + 1, dots[2 * k + 1], tx[k] + 1, ty[k], lr, lg, lb);
  }
  for (int k = (int)cand.size() - 1; k >= 0; --k) {
    if (!show.empty() && !show[k]) continue;
    if (!cand[k].object) { draw_disc(im, dots[2 * k], dots[2 * k + 1], kDotR, 255, 210, 0); continue; }
    const int h = kDotR + 2;   // 물체 후보: 검은 테 하늘색 네모
    for (int y = -h; y <= h; ++y)
      for (int x = -h; x <= h; ++x) {
        const bool edge = std::abs(x) == h || std::abs(y) == h;
        im.set(dots[2 * k] + x, dots[2 * k + 1] + y, edge ? 0 : 0, edge ? 0 : 200, edge ? 0 : 255);
      }
  }
  for (int k = (int)cand.size() - 1; k >= 0; --k)
    if (show.empty() || show[k]) draw_tag(im, tx[k], ty[k], cand[k].key);
}

std::string base64(const std::vector<uint8_t>& d) {
  static const char* T = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
  std::string o;
  o.reserve((d.size() + 2) / 3 * 4);
  for (size_t i = 0; i < d.size(); i += 3) {
    const uint32_t v = (uint32_t)d[i] << 16 | (i + 1 < d.size() ? (uint32_t)d[i + 1] << 8 : 0u) | (i + 2 < d.size() ? (uint32_t)d[i + 2] : 0u);
    o += T[v >> 18 & 63];
    o += T[v >> 12 & 63];
    o += i + 1 < d.size() ? T[v >> 6 & 63] : '=';
    o += i + 2 < d.size() ? T[v & 63] : '=';
  }
  return o;
}

std::string fmt(const char* f, double a) {
  char b[64];
  std::snprintf(b, sizeof b, f, a);
  return b;
}

const char* side_word(float bearing) {
  const float d = bearing * 57.29578f, a = std::fabs(d);
  if (a < 20.f) return "straight ahead";
  if (a < 70.f) return d > 0 ? "ahead-left" : "ahead-right";
  if (a < 110.f) return d > 0 ? "to the left" : "to the right";
  if (a < 160.f) return d > 0 ? "behind-left" : "behind-right";
  return "behind";
}

const char* kTask =
    "A small wheeled robot is exploring an unknown indoor floor. Its job is to map the whole floor in the shortest total time. It drives at about 1 m/s and turning around "
    "costs a few seconds. A frontier is a spot on the edge between mapped free space and unmapped space; driving to a frontier reveals new area.";
// 끝 선언을 켰을 때의 과제 글: 임무 문장은 설정(scripts/config/decider.yaml mission)에서 받는다
std::string task_with(const std::string& mission) {
  return "A small wheeled robot is exploring an unknown indoor floor. Its mission: " + mission + ". It drives at about 1 m/s and turning around "
         "costs a few seconds. A frontier is a spot on the edge between mapped free space and unmapped space; driving to a frontier reveals new area.";
}
const char* kLegend =
    "The image is the robot's current occupancy map seen from above: white = free space, black = wall or obstacle, gray = not mapped yet. The blue arrow is the robot, "
    "pointing where it is heading. Each candidate frontier is a small yellow dot; its letter is in the red circle joined to that dot by a red line.";
const char* kHint =
    "Rules of thumb: prefer a frontier that is close and roughly ahead; finish nearby small frontiers before leaving an area because coming back later is expensive; keep the "
    "current target unless another candidate is clearly better.";
const char* kAsk = "Which candidate frontier should the robot drive to next so that the whole floor gets mapped in the least total time?";

}  // namespace

std::vector<uint8_t> render_png(const int16_t* lmap, int size, int gw, int gh, int16_t unknown_l, int16_t occ_l, int kb[4], float rob_cx, float rob_cy, float rob_th,
                                std::vector<Cand>& cand, const std::vector<int>& cells, int px) {
  return render_png_mem(lmap, size, gw, gh, unknown_l, occ_l, kb, rob_cx, rob_cy, rob_th, cand, cells, px, MapMem());
}

namespace {
// 3 × 5 작은 글꼴 (꼴 B 칸 이름: 숫자 · '-' · 's' · N O W) — 줄마다 위에서부터 3 비트
const char* kSmallCh = "0123456789-sNOW ";
constexpr uint8_t kSmall[16][5] = {{7, 5, 5, 5, 7}, {2, 6, 2, 2, 7}, {7, 1, 7, 4, 7}, {7, 1, 7, 1, 7}, {5, 5, 7, 1, 1}, {7, 4, 7, 1, 7}, {7, 4, 7, 5, 7}, {7, 1, 1, 1, 1},
                                   {7, 5, 7, 5, 7}, {7, 5, 7, 1, 7}, {0, 0, 7, 0, 0}, {0, 3, 6, 3, 6}, {5, 7, 7, 7, 5}, {7, 5, 5, 5, 7}, {5, 5, 7, 7, 5}, {0, 0, 0, 0, 0}};
void draw_small(Img& im, int x, int y, const std::string& t, int sc) {
  for (size_t i = 0; i < t.size(); ++i) {
    const char* q = std::strchr(kSmallCh, t[i]);
    const int g = q ? (int)(q - kSmallCh) : 15;
    for (int yy = -1; yy <= 5 * sc; ++yy)   // 바탕 흰 상자 (읽기 쉽게)
      for (int xx = -1; xx <= 4 * sc; ++xx) im.set(x + (int)i * 4 * sc + xx, y + yy, 255, 255, 255);
    for (int r = 0; r < 5; ++r)
      for (int c = 0; c < 3; ++c)
        if ((kSmall[g][r] >> (2 - c)) & 1)
          for (int a = 0; a < sc; ++a) for (int b = 0; b < sc; ++b) im.set(x + (int)i * 4 * sc + c * sc + a, y + r * sc + b, 0, 0, 0);
  }
}
}  // namespace

static Img render_img(const int16_t* lmap, int size, int gw, int gh, int16_t unknown_l, int16_t occ_l, int kb[4], float rob_cx, float rob_cy, float rob_th,
                      std::vector<Cand>& cand, const std::vector<int>& cells, int px, const MapMem& mem, int* crop, bool fixed) {
  int x0 = (int)rob_cx, x1 = (int)rob_cx, y0 = (int)rob_cy, y1 = (int)rob_cy;
  if (kb[0] <= kb[2] && kb[1] <= kb[3]) { x0 = std::min(x0, kb[0]); y0 = std::min(y0, kb[1]); x1 = std::max(x1, kb[2]); y1 = std::max(y1, kb[3]); }
  for (const auto& q : mem.trail) { x0 = std::min(x0, (int)q.cx); x1 = std::max(x1, (int)q.cx); y0 = std::min(y0, (int)q.cy); y1 = std::max(y1, (int)q.cy); }
  for (const auto& q : mem.goals) { x0 = std::min(x0, (int)q.cx); x1 = std::max(x1, (int)q.cx); y0 = std::min(y0, (int)q.cy); y1 = std::max(y1, (int)q.cy); }
  for (size_t k = 0; k < cand.size(); ++k) { x0 = std::min(x0, cells[2 * k]); x1 = std::max(x1, cells[2 * k]); y0 = std::min(y0, cells[2 * k + 1]); y1 = std::max(y1, cells[2 * k + 1]); }
  constexpr int kMargin = 10, kMinSide = 128;   // 칸 (0.5 m · 6.4 m) — 잠정
  int side = std::max({x1 - x0 + 1 + 2 * kMargin, y1 - y0 + 1 + 2 * kMargin, kMinSide});
  int ox = (x0 + x1 + 1) / 2 - side / 2, oy = (y0 + y1 + 1) / 2 - side / 2;
  if (fixed) { ox = crop[0]; oy = crop[1]; side = crop[2]; } else { crop[0] = ox; crop[1] = oy; crop[2] = side; }   // 꼴 B: 지난 지도도 지금과 같은 범위
  Img im{px, px, std::vector<uint8_t>((size_t)px * px * 3, 128)};
  struct Flag { Flag(bool b) { g_clear_tags = b; } ~Flag() { g_clear_tags = false; } } flag(mem.clear_tags);
  for (int v = 0; v < px; ++v) {
    const int r = px - 1 - v;   // 위가 +y
    const int sy0 = oy + (int)((long)r * side / px), sy1 = std::max(sy0 + 1, oy + (int)(((long)(r + 1) * side + px - 1) / px));
    for (int u = 0; u < px; ++u) {
      const int sx0 = ox + (int)((long)u * side / px), sx1 = std::max(sx0 + 1, ox + (int)(((long)(u + 1) * side + px - 1) / px));
      int cls = 0;   // 0 모름 · 1 빈 곳 · 2 막힘 (작게 줄일 때 얇은 벽이 안 사라지게 막힘이 이김)
      for (int y = sy0; y < sy1 && cls < 2; ++y)
        for (int x = sx0; x < sx1 && cls < 2; ++x) {
          if (x < 0 || y < 0 || x >= gw || y >= gh) continue;
          const int16_t l = lmap[(size_t)y * size + x];
          if (l == unknown_l) continue;
          cls = std::max(cls, l >= occ_l ? 2 : 1);
        }
      if (cls == 1) im.set(u, v, 255, 255, 255);
      else if (cls == 2) im.set(u, v, 0, 0, 0);
    }
  }
  const auto to_px = [&](float cx, float cy, float* u, float* v) {
    *u = (cx - (float)ox) * (float)px / (float)side;
    *v = (float)px - (cy - (float)oy) * (float)px / (float)side;
  };
  for (const auto& q : mem.trail) {   // 지나온 길: 오래될수록 옅은 보라 (최근 = 진한 보라)
    float fu, fv;
    to_px(q.cx, q.cy, &fu, &fv);
    const float a = std::clamp(q.age_s / std::max(1e-3f, mem.trail_s), 0.f, 1.f);
    const auto mixc = [&](int c0, int c1) { return (uint8_t)std::lround(c0 + (c1 - c0) * a); };
    const uint8_t r = mixc(120, 225), g = mixc(0, 200), b = mixc(170, 240);
    for (int dy = -2; dy <= 2; ++dy)
      for (int dx = -2; dx <= 2; ++dx)
        if (dx * dx + dy * dy <= 4) im.set((int)std::lround(fu) + dx, (int)std::lround(fv) + dy, r, g, b);
  }
  for (const auto& q : mem.goals) {   // 지난 목표: 닿음 초록 ✓ · 실패 빨강 × · 바꿈 회색 고리 · 지금 목표 파란 고리
    float fu, fv;
    to_px(q.cx, q.cy, &fu, &fv);
    const int u = (int)std::lround(fu), v = (int)std::lround(fv);
    if (q.outcome == 2) {
      for (int t = -6; t <= 6; ++t)
        for (int w = 0; w < 3; ++w) { im.set(u + t + w - 1, v + t, 210, 0, 0); im.set(u + t + w - 1, v - t, 210, 0, 0); }
    } else if (q.outcome == 1) {
      for (int t = 0; t <= 3; ++t) for (int w = 0; w < 3; ++w) im.set(u - 4 + t, v + t + w - 1, 0, 160, 0);
      for (int t = 0; t <= 7; ++t) for (int w = 0; w < 3; ++w) im.set(u - 1 + t, v + 3 - t + w - 1, 0, 160, 0);
    } else {
      const uint8_t r = q.outcome == 0 ? 0 : 140, g = q.outcome == 0 ? 120 : 140, b = q.outcome == 0 ? 255 : 140;
      for (int dy = -8; dy <= 8; ++dy)
        for (int dx = -8; dx <= 8; ++dx) {
          const int d2 = dx * dx + dy * dy;
          if (d2 <= 64 && d2 >= 36) im.set(u + dx, v + dy, r, g, b);
        }
    }
  }
  float ru, rv;
  to_px(rob_cx, rob_cy, &ru, &rv);
  std::vector<int> dots(2 * cand.size());
  for (size_t k = 0; k < cand.size(); ++k) {
    float fu, fv;
    to_px((float)cells[2 * k] + 0.5f, (float)cells[2 * k + 1] + 0.5f, &fu, &fv);
    dots[2 * k] = cand[k].px = std::clamp((int)std::lround(fu), kDotR + 1, px - kDotR - 2);
    dots[2 * k + 1] = cand[k].py = std::clamp((int)std::lround(fv), kDotR + 1, px - kDotR - 2);
  }
  place_tags(im, cand, dots, (int)std::lround(ru), (int)std::lround(rv));
  {   // 결정 기록 번호: 흰 상자 검은 숫자 (꼬리표 위에 — 큰 번호부터 그려 1 이 맨 위)
    std::vector<MapMem::N> ns = mem.nums;
    std::sort(ns.begin(), ns.end(), [](const MapMem::N& a, const MapMem::N& b) { return a.num > b.num; });
    for (const auto& q : ns) {
      float fu, fv;
      to_px(q.cx, q.cy, &fu, &fv);
      draw_small(im, (int)std::lround(fu) + 5, (int)std::lround(fv) - 12, std::to_string(q.num), 2);
    }
  }
  {   // 로봇: 머리 방향 세모
    float fu, fv;
    to_px(rob_cx, rob_cy, &fu, &fv);
    const float c = std::cos(rob_th), s = -std::sin(rob_th);   // 사진은 아래가 +v
    const float ax = fu + 11.f * c, ay = fv + 11.f * s, bx = fu - 7.f * c - 7.f * s, by = fv - 7.f * s + 7.f * c, cx = fu - 7.f * c + 7.f * s, cy = fv - 7.f * s - 7.f * c;
    const auto edge = [](float px_, float py_, float qx, float qy, float x, float y) { return (qx - px_) * (y - py_) - (qy - py_) * (x - px_); };
    for (int y = (int)fv - 13; y <= (int)fv + 13; ++y)
      for (int x = (int)fu - 13; x <= (int)fu + 13; ++x) {
        const float e0 = edge(ax, ay, bx, by, (float)x, (float)y), e1 = edge(bx, by, cx, cy, (float)x, (float)y), e2 = edge(cx, cy, ax, ay, (float)x, (float)y);
        if ((e0 >= 0 && e1 >= 0 && e2 >= 0) || (e0 <= 0 && e1 <= 0 && e2 <= 0)) im.set(x, y, 0, 90, 255);
      }
  }
  return im;
}

std::vector<uint8_t> render_png_mem(const int16_t* lmap, int size, int gw, int gh, int16_t unknown_l, int16_t occ_l, int kb[4], float rob_cx, float rob_cy, float rob_th,
                                    std::vector<Cand>& cand, const std::vector<int>& cells, int px, const MapMem& mem) {
  int crop[3];
  return png_of(render_img(lmap, size, gw, gh, unknown_l, occ_l, kb, rob_cx, rob_cy, rob_th, cand, cells, px, mem, crop, false));
}


std::vector<uint8_t> render_png_tiles(const int16_t* lmap, const std::vector<const int16_t*>& past, const std::vector<float>& past_age_s, const std::vector<float>& past_pose,
                                      int size, int gw, int gh, int16_t unknown_l, int16_t occ_l, int kb[4], float rob_cx, float rob_cy, float rob_th, std::vector<Cand>& cand,
                                      const std::vector<int>& cells, int px, const MapMem& mem) {
  const int tp = px / 2;
  Img out{px, px, std::vector<uint8_t>((size_t)px * px * 3, 60)};
  int crop[3];
  std::vector<Cand> none;
  const std::vector<int> nc;
  const auto blit = [&](const Img& t, int u0, int v0) {
    for (int v = 0; v < tp; ++v)
      for (int u = 0; u < tp; ++u) {
        const uint8_t* q = &t.rgb[((size_t)v * tp + u) * 3];
        out.set(u0 + u, v0 + v, q[0], q[1], q[2]);
      }
  };
  Img now = render_img(lmap, size, gw, gh, unknown_l, occ_l, kb, rob_cx, rob_cy, rob_th, cand, cells, tp, mem, crop, false);
  draw_small(now, 3, 3, "NOW", 2);
  blit(now, 0, 0);
  for (size_t k = 0; k < past.size() && k < 3; ++k) {
    Img t = render_img(past[k], size, gw, gh, unknown_l, occ_l, kb, past_pose[3 * k], past_pose[3 * k + 1], past_pose[3 * k + 2], none, nc, tp, MapMem(), crop, true);
    draw_small(t, 3, 3, "-" + std::to_string((int)std::lround(past_age_s[k])) + "s", 2);
    blit(t, (int)((k + 1) % 2) * tp, (int)((k + 1) / 2) * tp);
  }
  for (int i = 0; i < px; ++i) { out.set(tp, i, 255, 160, 0); out.set(tp - 1, i, 255, 160, 0); out.set(i, tp, 255, 160, 0); out.set(i, tp - 1, 255, 160, 0); }   // 칸 경계
  return png_of(out);
}

std::vector<uint8_t> render_cam_grid(const std::vector<CamTile>& t, int W, int H, int cols, int rows) {
  Img out{W, H, std::vector<uint8_t>((size_t)W * H * 3, 40)};
  const int tw = W / cols, th = H / rows;
  struct Flag { Flag() { g_clear_tags = true; } ~Flag() { g_clear_tags = false; } } flag;
  for (size_t k = 0; k < t.size() && (int)k < cols * rows; ++k) {
    const CamTile& c = t[k];
    const int u0 = (int)(k % cols) * tw, v0 = (int)(k / cols) * th;
    Img tile{tw - 2, th - 2, std::vector<uint8_t>((size_t)(tw - 2) * (th - 2) * 3, 110)};
    if (c.rgb && c.w > 0 && c.h > 0) {
      for (int y = 0; y < tile.h; ++y)   // 겹선형으로 줄이기·늘이기
        for (int x = 0; x < tile.w; ++x) {
          const float sx = (x + 0.5f) * c.w / tile.w - 0.5f, sy = (y + 0.5f) * c.h / tile.h - 0.5f;
          const int x0 = std::clamp((int)std::floor(sx), 0, c.w - 1), y0 = std::clamp((int)std::floor(sy), 0, c.h - 1);
          const int x1 = std::min(x0 + 1, c.w - 1), y1 = std::min(y0 + 1, c.h - 1);
          const float fx = std::clamp(sx - x0, 0.f, 1.f), fy = std::clamp(sy - y0, 0.f, 1.f);
          uint8_t px[3];
          for (int ch = 0; ch < 3; ++ch) {
            const float a = c.rgb[((size_t)y0 * c.w + x0) * 3 + ch] * (1 - fx) + c.rgb[((size_t)y0 * c.w + x1) * 3 + ch] * fx;
            const float b = c.rgb[((size_t)y1 * c.w + x0) * 3 + ch] * (1 - fx) + c.rgb[((size_t)y1 * c.w + x1) * 3 + ch] * fx;
            px[ch] = (uint8_t)std::lround(a * (1 - fy) + b * fy);
          }
          tile.set(x, y, px[0], px[1], px[2]);
        }
      if (c.u >= 0 && c.v >= 0) {   // 후보 자리: 검은 테 노란 점
        const int pu = (int)std::lround(c.u * tile.w / c.w), pv = (int)std::lround(c.v * tile.h / c.h);
        for (int dy = -7; dy <= 7; ++dy)
          for (int dx = -7; dx <= 7; ++dx) {
            const int d2 = dx * dx + dy * dy;
            if (d2 <= 25) tile.set(pu + dx, pv + dy, 255, 210, 0);
            else if (d2 <= 49) tile.set(pu + dx, pv + dy, 0, 0, 0);
          }
      }
    }
    draw_tag(tile, kTagR + 2, kTagR + 2, c.key);
    if (c.age_s >= 0) draw_small(tile, tile.w - 4 * 2 * 4 - 4, 3, std::to_string((int)std::lround(c.age_s)) + "s", 2);
    for (int y = 0; y < tile.h; ++y)
      for (int x = 0; x < tile.w; ++x) {
        const uint8_t* q = &tile.rgb[((size_t)y * tile.w + x) * 3];
        out.set(u0 + 1 + x, v0 + 1 + y, q[0], q[1], q[2]);
      }
  }
  return png_of(out);
}

std::string png_uri(const std::vector<uint8_t>& png) { return "data:image/png;base64," + base64(png); }

std::string build_body(int input, int prompt, const std::vector<Cand>& cand, const std::vector<uint8_t>& png, bool v2) {
  const bool img = input != kText && !png.empty(), txt = input != kImage;
  json body, crit = json::object();
  if (prompt == 3) {   // 상태를 JSON 으로 (글 설명 없음, 선택지 설명 없음)
    json st;
    st["task"] = kTask;
    if (img) st["map_image"] = kLegend;
    st["candidates"] = json::array();
    for (const Cand& c : cand) {
      json o;
      o["id"] = std::string(1, c.key);
      if (txt) {
        o["path_distance_m"] = std::round(c.path_m * 10.0) / 10.0;   // double 로 — float 는 JSON 에 1.2999999523 처럼 찍힘
        o["bearing_deg_left_positive"] = (int)std::lround(c.bearing * 57.29578f);
        o["frontier_width_m"] = std::round(c.size_m * 10.0) / 10.0;
        if (v2) o["expected_new_area_ratio"] = std::round(c.util * 100.0) / 100.0;
        else o["frontier_cells_within_2m"] = (int)std::lround(c.util);
        o["is_current_target"] = c.cur;
      }
      st["candidates"].push_back(o);
      crit[std::string(1, c.key)] = nullptr;
    }
    body["state"] = st;
  } else {
    std::string st = kTask;
    if (img) st += std::string("\n") + kLegend;
    if (txt) {
      st += "\nCandidate frontiers:";
      for (const Cand& c : cand) {
        st += std::string("\n") + c.key + ": " + fmt("%.1f", c.path_m) + " m away by path, " + fmt("%.0f", std::fabs(c.bearing) * 57.29578f) + " deg " + (c.bearing >= 0 ? "left" : "right") +
              " of the heading (" + side_word(c.bearing) + "), frontier " + fmt("%.1f", c.size_m) + " m wide, " +
              (v2 ? fmt("%.2f", c.util) + " expected new area ratio" : fmt("%.0f", c.util) + " frontier cells within 2 m") + (c.cur ? ", current target" : "");
      }
    }
    if (prompt == 2) st += std::string("\n") + kHint;
    body["state"] = st;
    for (const Cand& c : cand) {
      const std::string k(1, c.key);
      if (txt) crit[k] = fmt("%.1f", c.path_m) + " m by path, " + side_word(c.bearing) + ", " + fmt("%.1f", c.size_m) + " m wide" + (c.cur ? ", current target" : "");
      else crit[k] = "the frontier marked " + k + " in the image";
    }
  }
  if (img) body["images"] = json::array({"data:image/png;base64," + base64(png)});
  body["questions"]["pick"] = json{{"type", "choice"}, {"instructions", kAsk}, {"criteria", crit}};
  return body.dump();
}


std::vector<uint8_t> render_cam_png(const CamSpec& C, float x, float y, float th, const RayFn& ray, std::vector<Cand>& cand, const std::vector<float>& wxy, std::vector<uint8_t>* rgb_out) {
  Img im{C.w, C.h, std::vector<uint8_t>((size_t)C.w * C.h * 3, 0)};
  const float co = std::cos(th), si = std::sin(th);
  const float o[3] = {x + co * C.mx - si * C.my, y + si * C.mx + co * C.my, C.mz};
  const auto world_dir = [&](float X, float Y, float Z, float d[3]) {   // 센서 FLU → 숙인 각 → 머리 방향 (sim hier_path.h cam_point_base 와 같은 회전)
    const float bx = C.cp * X + C.sp * Z, bz = C.cp * Z - C.sp * X;
    d[0] = co * bx - si * Y;
    d[1] = si * bx + co * Y;
    d[2] = bz;
  };
  const float kL[3] = {0.30f, 0.20f, 0.933f};   // 빛 방향 (위에서 조금 비스듬히) — 그림용, 잠정
  for (int v = 0; v < C.h; ++v)
    for (int u = 0; u < C.w; ++u) {
      float d[3], t = 0, n[3] = {0, 0, 1};
      int obj = 0;
      world_dir(1.f, -((float)u - C.cx) / C.fx, -((float)v - C.cy) / C.fy, d);
      if (!ray(o, d, &t, n, &obj)) { im.set(u, v, 18, 20, 34); continue; }
      const float len = std::sqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2]), dist = t * len, hz = o[2] + t * d[2];
      const float fog = 1.f / (1.f + 0.06f * dist);
      float r, g, b;
      if (std::fabs(n[2]) > 0.9f && hz < 0.02f) {   // 바닥
        r = 205; g = 185; b = 150;
        const float px_ = o[0] + t * d[0], py_ = o[1] + t * d[1];   // 0.5 m 바둑판 무늬 (거리감) — 옅게
        if ((((int)std::floor(px_ * 2.f) + (int)std::floor(py_ * 2.f)) & 1) != 0) { r -= 14; g -= 14; b -= 12; }
      } else {
        const uint32_t hh = (uint32_t)obj * 2654435761u;
        r = 150.f + (float)(hh >> 8 & 63);
        g = 150.f + (float)(hh >> 14 & 63);
        b = 150.f + (float)(hh >> 20 & 63);
        const float sh = 0.60f + 0.40f * std::fabs(n[0] * kL[0] + n[1] * kL[1] + n[2] * kL[2]);
        r *= sh; g *= sh; b *= sh;
      }
      im.set(u, v, (uint8_t)(r * fog), (uint8_t)(g * fog), (uint8_t)(b * fog));
    }
  std::vector<int> dots(2 * cand.size(), 0);
  std::vector<char> show(cand.size(), 0);
  for (size_t k = 0; k < cand.size(); ++k) {
    cand[k].in_cam = false;
    const float wx = wxy[2 * k] - o[0], wy = wxy[2 * k + 1] - o[1], hz = -C.mz;   // 후보 칸 바닥 점, 카메라 기준
    const float hx = co * wx + si * wy, hy = -si * wx + co * wy;
    const float X = C.cp * hx - C.sp * hz, Z = C.sp * hx + C.cp * hz;
    if (X < 0.2f) continue;
    const float fu = C.cx - C.fx * hy / X, fv = C.cy - C.fy * Z / X;
    if (fu < 2 || fv < 2 || fu > (float)C.w - 3 || fv > (float)C.h - 3) continue;
    float d[3] = {wx, wy, hz}, t = 0, n[3];
    int obj = 0;
    if (ray(o, d, &t, n, &obj) && t < 0.97f) continue;   // 가려짐
    cand[k].in_cam = true;
    show[k] = 1;
    dots[2 * k] = cand[k].cu = (int)std::lround(fu);
    dots[2 * k + 1] = cand[k].cv = (int)std::lround(fv);
  }
  place_tags(im, cand, dots, -1, -1, show);
  if (rgb_out) *rgb_out = im.rgb;
  return png_of(im);
}

namespace {
const char* kCamLegend =
    "The FIRST image is the view from the robot's forward camera (0.19 m above the floor, looking ahead). It is a synthetic rendering, not a photo: the floor is the sand-coloured "
    "checkered surface, every object or wall has its own flat colour, farther things are darker. A yellow dot with a red lettered circle marks a candidate frontier that lies in view.";
const char* kMapLegend2 =
    "The SECOND image is the robot's current occupancy map seen from above: white = free space, black = wall or obstacle, gray = not mapped yet. The blue arrow is the robot, "
    "pointing where it is heading. Each candidate frontier is a small yellow dot; its letter is in the red circle joined to that dot by a red line.";
const char* kRichGuide =
    "Candidate fields: travel_time_s = estimated driving time there including turning; expected_new_area_ratio = share of the area around that spot (8 m disk) that is still unmapped "
    "and visible from it; new_area_per_second = that ratio divided by travel time; extra_total_time_if_chosen_first_s = how much longer a full tour over all frontiers becomes if this "
    "candidate is visited first (0 for the best first stop); open_depth_beyond_m = how far free or unknown space continues beyond the opening (short = closet or corner, long = room "
    "or corridor); other_frontiers_in_its_room = other candidates in the same room; narrowest_gap_on_way_m = smallest distance from the way to a mapped obstacle; clutter_on_way = "
    "share of mapped obstacle cells near the way; people_near_way = tracked people within 1 m of the way.";
const char* kLowNote =
    "The map is made by a flat 2D laser scanner 0.11 m above the floor, so obstacles lower than that (boxes, sills, steps) do NOT appear on the map. The robot cannot drive over "
    "anything taller than 3 cm. Such low obstacles are visible only in the camera image, so a way that looks free on the map can in fact be blocked.";
int g_compact = 0;   // set_compact: 0 = 후보마다 JSON 객체(기본), 1 = 압축 표, 2 = 압축 표에서 extra 열 뺌, 3 = 후보마다 JSON 객체(짧은 키), 4 = 3 에서 extra 뺌
// 압축 표 (fuel 후보 전용): 열 이름 한 줄 + 후보마다 값 한 줄. 값은 cand_json 과 같은 반올림 · 같은 숫자 글(json dump), 참/거짓은 1/0
struct Col { const char* name; const char* key; const char* what; };
const Col kCols[] = {
    {"t", "travel_time_s", "estimated driving time there in s, including turning"},
    {"dist", "path_distance_m", "path length in m"},
    {"bear", "bearing_deg_left_positive", "bearing in degrees, left positive"},
    {"width", "frontier_width_m", "frontier width in m"},
    {"new", "expected_new_area_ratio", "share of the area around it (8 m disk) still unmapped and visible from it"},
    {"rate", "new_area_per_second", "new divided by t"},
    {"extra", "extra_total_time_if_chosen_first_s", "how much longer in s a full tour over all frontiers becomes if this one is visited first (0 for the best first stop)"},
    {"depth", "open_depth_beyond_m", "how far free or unknown space continues beyond the opening in m (short = closet or corner, long = room or corridor)"},
    {"same", "same_room_as_robot", "1 if in the robot's room"},
    {"roomleft", "other_frontiers_in_its_room", "other candidates in the same room"},
    {"gap", "narrowest_gap_on_way_m", "smallest distance in m from the way to a mapped obstacle"},
    {"clutter", "clutter_on_way", "share of mapped obstacle cells near the way"},
    {"ppl", "people_near_way", "tracked people within 1 m of the way"},
    {"cur", "is_current_target", "1 if it is the current target"},
    {"cam", "marked_in_camera_image", "1 if marked in the camera image"},
    {"tried", "times_tried_as_goal", "times tried as goal"},
    {"aband", "times_abandoned_no_progress_or_path_lost", "times abandoned (no progress or path lost)"},
};
json cand_json(const Cand& c, bool v2);
void compact_table(const std::vector<Cand>& cand, bool v2, json& st) {
  std::vector<json> O;
  for (const Cand& c : cand) O.push_back(cand_json(c, v2));
  const bool obj = g_compact >= 3;   // 짧은 키 JSON 객체 (기본 모델이 쉼표 표의 열 짝짓기를 못 함 — 압축 표 기본 맞춤 0.523 vs JSON 0.785)
  std::string leg = obj ? "Each candidate is an object with short keys: id = letter" : "Candidates are rows of a comma-separated table; the first row names the columns: id = letter";
  std::string hdr = "id", rows;
  std::vector<const Col*> use;
  for (const Col& k : kCols) {
    if ((g_compact == 2 || g_compact == 4) && std::string(k.name) == "extra") continue;
    bool any = false;
    for (const json& o : O) any = any || o.contains(k.key);
    if (!any) continue;
    use.push_back(&k);
    std::string what = k.what;
    if (obj && what.rfind("1 if ", 0) == 0) what = "true if " + what.substr(5);   // 짧은 키 JSON 은 참/거짓을 true/false 로 적음
    leg += std::string("; ") + k.name + " = " + what;
    hdr += std::string(",") + k.name;
  }
  for (const json& o : O) {
    rows += "\n" + o["id"].get<std::string>();
    for (const Col* k : use) {
      rows += ",";
      if (!o.contains(k->key)) continue;
      const json& v = o[k->key];
      rows += v.is_boolean() ? (v.get<bool>() ? "1" : "0") : v.dump();
    }
  }
  st["candidate_columns"] = leg + ".";
  if (obj) {
    json arr = json::array();
    for (const json& o : O) {
      json x;
      x["id"] = o["id"];
      for (const Col* k : use) if (o.contains(k->key)) x[k->name] = o[k->key];
      arr.push_back(x);
    }
    st["candidates"] = arr;
    return;
  }
  st["candidates"] = hdr + rows;
}
json cand_json(const Cand& c, bool v2) {
  json o;
  o["id"] = std::string(1, c.key);
  o["path_distance_m"] = std::round(c.path_m * 10.0) / 10.0;
  o["bearing_deg_left_positive"] = (int)std::lround(c.bearing * 57.29578f);
  o["frontier_width_m"] = std::round(c.size_m * 10.0) / 10.0;
  if (v2) o["expected_new_area_ratio"] = std::round(c.util * 100.0) / 100.0;
  else o["frontier_cells_within_2m"] = (int)std::lround(c.util);
  o["is_current_target"] = c.cur;
  o["marked_in_camera_image"] = c.in_cam;
  if (c.rich) {   // fuel 후보 (kRichGuide 에 뜻)
    const auto r1 = [](float v) { return std::round(v * 10.0) / 10.0; };
    const auto r2 = [](float v) { return std::round(v * 100.0) / 100.0; };
    o.erase("frontier_cells_within_2m");
    o["travel_time_s"] = r1(c.t_s);
    o["expected_new_area_ratio"] = r2(c.gain);
    o["new_area_per_second"] = std::round(c.gpt * 1000.0) / 1000.0;
    o["extra_total_time_if_chosen_first_s"] = r1(c.opp_s);
    o["open_depth_beyond_m"] = r1(c.depth_m);
    o["same_room_as_robot"] = c.same_room;
    o["other_frontiers_in_its_room"] = c.room_left;
    o["narrowest_gap_on_way_m"] = r2(c.clear_m);
    o["clutter_on_way"] = r2(c.clutter);
    o["people_near_way"] = c.people;
    if (c.tried >= 0) { o["times_tried_as_goal"] = c.tried; o["times_abandoned_no_progress_or_path_lost"] = c.abandoned; }
  }
  if (c.tried >= 0) {   // 끝 선언을 물을 때만 (로봇이 스스로 센 값)
    o["times_tried_as_goal"] = c.tried;
    o["times_abandoned_no_progress_or_path_lost"] = c.abandoned;
  }
  return o;
}
}  // namespace

void set_compact(int m) { g_compact = m; }

// 프롬프트 판 (입력 all):
//   choice 1 : 과제 + 그림 설명 둘 + 후보 표(JSON) → "어느 후보로?" (prompt 3 꼴 그대로에 카메라 그림만 더함)
//   choice 2 : 1 + 낮은 장애물 설명(kLowNote) + "카메라에 보이는 장애물로 길이 막힌 후보는 고르지 마라"
//   prune 1  : 과제 + 그림 설명 둘 + 낮은 장애물 설명 + 후보 표 → 후보마다 "X 로 가는 길이 지도에 없는 장애물로 막혔나?" (참 = 막힘)
//   prune 2  : 같은 상태 → 후보마다 "X 쪽으로 부딪히지 않고 갈 수 있나?" (참 = 갈 수 있음 — 부른 쪽이 1 − 답을 막힘으로)
//   끝 선언 (fin != nullptr — combine_run --decider-finish): 상태에 exploration_status(로봇이 스스로 아는 값만)와 finish_option 설명을 붙이고,
//     choice 질문은 그대로 후보만 고르고, 같은 요청에 score 질문 하나("남은 탐사가 얼마나 되나", answers.left)를 더한다. 글·단계·임무 문장은 설정(FinishInfo::cfg). 프롬프트는 한 판뿐 (시험 장면에 맞춰 고치지 않음)
std::string build_body_all(int mode, int prompt, const std::vector<Cand>& cand, const std::vector<uint8_t>& cam_png, const std::vector<uint8_t>& map_png, bool v2, int n_ask, const FinishInfo* fin) {
  json body, st;
  st["task"] = fin ? task_with(fin->cfg->mission) : std::string(kTask);   // 끝 선언: 임무 문장을 설정 글에서 (끄면 측정 때의 글 그대로)
  st["camera_image"] = kCamLegend;
  st["map_image"] = kMapLegend2;
  if (mode == kPrune || prompt == 2) st["warning"] = kLowNote;
  if (g_compact && !cand.empty() && cand[0].rich) compact_table(cand, v2, st);
  else {
    if (!cand.empty() && cand[0].rich) st["candidate_fields"] = kRichGuide;
    st["candidates"] = json::array();
    for (const Cand& c : cand) st["candidates"].push_back(cand_json(c, v2));
  }
  if (fin) {
    json es;
    es["candidate_frontiers_now"] = fin->n_valid;
    es["frontiers_skipped_now_because_tried_and_abandoned"] = fin->n_skipped;
    es["lidar_map_closed"] = fin->closed;
    es["seconds_since_map_last_grew"] = std::round(fin->idle_s * 10.0) / 10.0;
    es["goals_tried_and_abandoned_so_far"] = fin->n_abandoned;
    es["elapsed_s"] = std::round(fin->t_s * 10.0) / 10.0;
    st["exploration_status"] = es;
  }
  body["state"] = st;
  body["images"] = json::array({"data:image/png;base64," + base64(cam_png), "data:image/png;base64," + base64(map_png)});
  if (mode == kChoice) {
    json crit = json::object();
    for (const Cand& c : cand) crit[std::string(1, c.key)] = nullptr;
    std::string q = kAsk;
    if (prompt == 2) q += " Do not pick a frontier whose way is blocked by an obstacle that can be seen in the camera image.";
    if (!fin || cand.size() >= 2) body["questions"]["pick"] = json{{"type", "choice"}, {"instructions", q}, {"criteria", crit}};
    if (fin) {   // 끝 선언: 같은 요청에 score 질문 하나 (글·단계는 설정에서)
      json lv = json::array();
      for (const std::string& l : fin->cfg->left_levels) lv.push_back(l);
      body["questions"]["left"] = json{{"type", "score"}, {"instructions", fin->cfg->left_question}, {"criteria", lv}};
    }
  } else {
    for (int k = 0; k < n_ask && k < (int)cand.size(); ++k) {
      const std::string id(1, cand[k].key);
      const std::string q = prompt == 2 ? "Can the robot drive from where it is now toward candidate frontier " + id + " without running into an obstacle that is visible in the camera image?"
                                        : "Is the way from the robot toward candidate frontier " + id + " blocked by an obstacle that is visible in the camera image but missing from the map?";
      body["questions"]["blk_" + id] = json{{"type", "noul"}, {"instructions", q}};
    }
  }
  return body.dump();
}

// 점수 방식: 후보마다 score 질문 하나. 과제 글의 임무 문장 · 질문 글 · 단계 설명은 전부 설정(ScoreCfg)에서
std::string build_body_score(const std::vector<Cand>& cand, const std::vector<uint8_t>& cam_png, const std::vector<uint8_t>& map_png, bool v2, const ScoreCfg& sc) {
  json body, st;
  st["task"] = task_with(sc.mission);
  st["camera_image"] = kCamLegend;
  st["map_image"] = kMapLegend2;
  st["candidates"] = json::array();
  for (const Cand& c : cand) st["candidates"].push_back(cand_json(c, v2));
  body["state"] = st;
  body["images"] = json::array({"data:image/png;base64," + base64(cam_png), "data:image/png;base64," + base64(map_png)});
  json crit = json::array();
  for (const std::string& l : sc.levels) crit.push_back(l);
  for (const Cand& c : cand) {
    const std::string id(1, c.key);
    std::string q = sc.question;
    for (size_t p; (p = q.find("{id}")) != std::string::npos;) q.replace(p, 4, id);
    body["questions"]["s_" + id] = json{{"type", "score"}, {"instructions", q}, {"criteria", crit}};
  }
  return body.dump();
}

#ifndef DEC_NO_CURL
struct Client::Impl {

  std::string url;
  int timeout_ms, parallel;
  CURLM* multi = nullptr;
  curl_slist* hdr = nullptr;
  std::vector<CURL*> idle;
};

namespace {
struct Xfer {
  int idx = -1;
  std::string out;
  std::chrono::steady_clock::time_point t0;
};
size_t on_data(char* p, size_t s, size_t n, void* u) {
  static_cast<Xfer*>(u)->out.append(p, s * n);
  return s * n;
}
}  // namespace

Client::Client(const std::string& url, int timeout_ms, int parallel) : p_(new Impl) {
  curl_global_init(CURL_GLOBAL_DEFAULT);
  p_->url = url + "/v1/systemone";
  p_->timeout_ms = timeout_ms;
  p_->parallel = std::max(1, parallel);
  p_->multi = curl_multi_init();
  p_->hdr = curl_slist_append(nullptr, "Content-Type: application/json");
  p_->hdr = curl_slist_append(p_->hdr, "Expect:");
  for (int i = 0; i < p_->parallel; ++i) p_->idle.push_back(curl_easy_init());   // 이어 쓰는 연결 (keep-alive)
}
Client::~Client() {
  for (CURL* e : p_->idle) curl_easy_cleanup(e);
  curl_slist_free_all(p_->hdr);
  curl_multi_cleanup(p_->multi);
}

std::vector<Reply> Client::ask(const std::vector<std::string>& bodies, const std::vector<std::vector<Cand>>& cands, const std::vector<int>& n_ask, bool invert, bool finish, int score_levels) {
  const int n = (int)bodies.size();
  std::vector<Reply> out(n);
  std::vector<Xfer> xf(n);
  int next = 0, live = 0;
  while (next < n || live > 0) {
    while (next < n && !p_->idle.empty()) {
      CURL* e = p_->idle.back();
      p_->idle.pop_back();
      Xfer& x = xf[next];
      x.idx = next;
      curl_easy_setopt(e, CURLOPT_URL, p_->url.c_str());
      curl_easy_setopt(e, CURLOPT_HTTPHEADER, p_->hdr);
      curl_easy_setopt(e, CURLOPT_POSTFIELDS, bodies[next].data());
      curl_easy_setopt(e, CURLOPT_POSTFIELDSIZE_LARGE, (curl_off_t)bodies[next].size());
      curl_easy_setopt(e, CURLOPT_WRITEFUNCTION, on_data);
      curl_easy_setopt(e, CURLOPT_WRITEDATA, &x);
      curl_easy_setopt(e, CURLOPT_PRIVATE, &x);
      curl_easy_setopt(e, CURLOPT_TIMEOUT_MS, (long)p_->timeout_ms);
      curl_easy_setopt(e, CURLOPT_NOSIGNAL, 1L);
      x.t0 = std::chrono::steady_clock::now();
      curl_multi_add_handle(p_->multi, e);
      ++next;
      ++live;
    }
    int running = 0;
    curl_multi_perform(p_->multi, &running);
    int left = 0;
    while (CURLMsg* m = curl_multi_info_read(p_->multi, &left)) {
      if (m->msg != CURLMSG_DONE) continue;
      CURL* e = m->easy_handle;
      Xfer* x = nullptr;
      curl_easy_getinfo(e, CURLINFO_PRIVATE, &x);
      Reply& r = out[x->idx];
      r.ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - x->t0).count();
      long http = 0;
      curl_easy_getinfo(e, CURLINFO_RESPONSE_CODE, &http);
      r.raw = x->out;
      if (m->data.result != CURLE_OK) r.err = std::string("curl: ") + curl_easy_strerror(m->data.result);
      else if (http != 200) r.err = "http " + std::to_string(http);
      else if (score_levels > 0 && !finish) {   // score: 후보마다 단계 확률 (서버는 단계 번호 "0" … 를 키로 줌)
        const json j = json::parse(x->out, nullptr, false);
        const std::vector<Cand>& cs = cands[x->idx];
        r.ok = j.is_object() && j.contains("answers") && j["answers"].is_object();
        for (size_t k = 0; r.ok && k < cs.size(); ++k) {
          const std::string id = "s_" + std::string(1, cs[k].key);
          const json& A = j["answers"];
          if (!A.contains(id) || !A[id].is_object() || !A[id].contains("probabilities") || !A[id]["probabilities"].is_object()) { r.ok = false; break; }
          const json& P = A[id]["probabilities"];
          float e = 0.f;
          for (int l = 0; l < score_levels; ++l) {
            const std::string kk = std::to_string(l);
            if (!P.contains(kk) || !P[kk].is_number()) { r.ok = false; break; }
            const float p = P[kk].get<float>();
            r.lv.push_back(p);
            e += (float)l * p;
          }
          r.exp.push_back(e);
        }
        if (!r.ok) r.err = "answer shape";
      } else if (!n_ask.empty()) {   // prune: 앞 후보마다 noul

        const json j = json::parse(x->out, nullptr, false);
        const std::vector<Cand>& cs = cands[x->idx];
        r.ok = j.is_object() && j.contains("answers") && j["answers"].is_object();
        for (int k = 0; r.ok && k < n_ask[x->idx]; ++k) {
          const std::string id = "blk_" + std::string(1, cs[k].key);
          const json& A = j["answers"];
          if (!A.contains(id) || !A[id].is_object() || !A[id].contains("noul") || !A[id]["noul"].is_number()) { r.ok = false; break; }
          const float p = A[id]["noul"].get<float>();
          r.blocked.push_back(invert ? 1.f - p : p);
        }
        if (!r.ok) r.err = "answer shape";
      } else {
        const json j = json::parse(x->out, nullptr, false);
        const std::vector<Cand>& cs = cands[x->idx];
        const bool has_pick = !finish || cs.size() >= 2;   // 끝 선언을 같이 물을 때 후보가 2 개보다 적으면 choice 질문이 없음
        const auto a = j.is_object() && j.contains("answers") && j["answers"].contains("pick") ? j["answers"]["pick"] : json();
        if (!has_pick) r.ok = j.is_object() && j.contains("answers") && j["answers"].is_object();
        else if (!a.is_object() || !a.contains("choice") || !a["choice"].is_string()) r.err = "answer shape";
        else {
          const std::string key = a["choice"].get<std::string>();
          for (size_t k = 0; k < cs.size(); ++k)
            if (key.size() == 1 && key[0] == cs[k].key) r.choice = (int)k;
          if (r.choice < 0) r.err = "unknown option " + key;
          else {
            r.ok = true;
            for (const Cand& cd : cs) {
              const std::string kk(1, cd.key);
              r.probs.push_back(a.contains("probabilities") && a["probabilities"].contains(kk) && a["probabilities"][kk].is_number() ? a["probabilities"][kk].get<float>() : -1.f);
            }
            if (a.contains("confidence") && a["confidence"].is_number()) r.conf = a["confidence"].get<float>();
            if (a.contains("probabilities") && a["probabilities"].contains(key) && a["probabilities"][key].is_number()) r.p = a["probabilities"][key].get<float>();
          }
        }
        if (r.ok && finish) {   // "남은 탐사" score 답
          const json& A = j["answers"];
          r.ok = A.contains("left") && A["left"].is_object() && A["left"].contains("probabilities") && A["left"]["probabilities"].is_object();
          float e = 0.f;
          for (int l = 0; r.ok && l < score_levels; ++l) {
            const std::string kk = std::to_string(l);
            const json& P = A["left"]["probabilities"];
            if (!P.contains(kk) || !P[kk].is_number()) { r.ok = false; break; }
            r.lv.push_back(P[kk].get<float>());
            e += (float)l * r.lv.back();
          }
          r.exp.push_back(e);
          if (!r.ok) r.err = "answer shape";
        }
      }
      curl_multi_remove_handle(p_->multi, e);
      p_->idle.push_back(e);
      --live;
    }
    if (live > 0) curl_multi_poll(p_->multi, nullptr, 0, 50, nullptr);
  }
  return out;
}
#endif  // DEC_NO_CURL

bool config_find(const std::string& path, const std::string& key, std::string* out) {
  std::ifstream f(path);
  if (!f) throw std::runtime_error("판단 모델 설정 파일을 못 엶: " + path);
  std::string line;
  while (std::getline(f, line)) {
    if (line.rfind(key + ":", 0) != 0) continue;
    std::string v = line.substr(key.size() + 1);
    const size_t b = v.find_first_not_of(" \t"), e = v.find_last_not_of(" \t\r");
    if (b == std::string::npos) return false;
    v = v.substr(b, e - b + 1);
    if (v.size() >= 2 && v.front() == '"' && v.back() == '"') v = v.substr(1, v.size() - 2);
    if (v.empty()) return false;
    *out = v;
    return true;
  }
  return false;
}
std::string config_text(const std::string& path, const std::string& key) {
  std::string v;
  if (!config_find(path, key, &v)) throw std::runtime_error("판단 모델 설정에 " + key + " 가 없음: " + path);
  return v;
}
ScoreCfg load_score_cfg(const std::string& path) {
  ScoreCfg sc;
  sc.mission = config_text(path, "mission");
  sc.question = config_text(path, "score_question");
  std::string v;
  for (int l = 0; l < 10 && config_find(path, "score_level_" + std::to_string(l), &v); ++l) sc.levels.push_back(v);
  if (sc.levels.size() < 2) throw std::runtime_error("판단 모델 설정의 score_level_0 … 이 2 개보다 적음: " + path);
  sc.margin = std::stof(config_text(path, "score_margin"));
  sc.finish_below = std::stof(config_text(path, "score_finish_below"));
  sc.left_question = config_text(path, "left_question");
  for (int l = 0; l < 10 && config_find(path, "left_level_" + std::to_string(l), &v); ++l) sc.left_levels.push_back(v);
  if (sc.left_levels.size() < 2) throw std::runtime_error("판단 모델 설정의 left_level_0 … 이 2 개보다 적음: " + path);
  sc.left_finish_below = std::stof(config_text(path, "left_finish_below"));
  sc.left_hold = std::stoi(config_text(path, "left_hold"));
  return sc;
}

GridOpen grid_open_frontier(const int8_t* occ, int w, int h, int rx, int ry, float radius_cells, int min_cells, int occ_thr, int start_cells) {
  GridOpen g;
  if (w <= 0 || h <= 0) return g;
  const size_t n = (size_t)w * h;
  const int R = (int)std::ceil(radius_cells);
  std::vector<uint8_t> stand(n, 0), seen(n, 0);
  for (int y = 0; y < h; ++y)   // 설 수 있는 칸: 아는 칸 · 막힘 아님 · 반지름 안에 막힘 없음
    for (int x = 0; x < w; ++x) {
      const int8_t v = occ[(size_t)y * w + x];
      if (v < 0 || v >= occ_thr) continue;
      bool ok = true;
      for (int dy = -R; dy <= R && ok; ++dy)
        for (int dx = -R; dx <= R; ++dx) {
          const int u = x + dx, t = y + dy;
          if (u < 0 || t < 0 || u >= w || t >= h || (float)(dx * dx + dy * dy) > radius_cells * radius_cells) continue;
          if (occ[(size_t)t * w + u] >= occ_thr) { ok = false; break; }
        }
      stand[(size_t)y * w + x] = ok;
    }
  int best = 1 << 30;   // 로봇 칸에서 가장 가까운 설 수 있는 칸
  for (int dy = -start_cells; dy <= start_cells; ++dy)
    for (int dx = -start_cells; dx <= start_cells; ++dx) {
      const int x = rx + dx, y = ry + dy;
      if (x < 0 || y < 0 || x >= w || y >= h || !stand[(size_t)y * w + x] || dx * dx + dy * dy >= best) continue;
      best = dx * dx + dy * dy;
      g.start = y * w + x;
    }
  if (g.start < 0) return g;
  const int d4[4][2] = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}};
  std::vector<int> q{g.start}, fr;
  seen[g.start] = 1;
  for (size_t k = 0; k < q.size(); ++k) {   // 4 이웃 번지기 + 프론티어 칸 표시 (seen 2)
    const int x = q[k] % w, y = q[k] / w;
    bool f = false;
    for (const auto& d : d4) {
      const int u = x + d[0], t = y + d[1];
      if (u < 0 || t < 0 || u >= w || t >= h) continue;
      const size_t c = (size_t)t * w + u;
      if (occ[c] < 0) f = true;
      else if (stand[c] && !seen[c]) { seen[c] = 1; q.push_back((int)c); }
    }
    if (f) fr.push_back(q[k]);
  }
  g.n_reach = (int)q.size();
  g.n_frontier = (int)fr.size();
  for (int c : fr) seen[c] = 2;
  for (int c0 : fr) {   // 8 이웃으로 묶음
    if (seen[c0] != 2) continue;
    std::vector<int> cl{c0};
    seen[c0] = 3;
    for (size_t k = 0; k < cl.size(); ++k) {
      const int x = cl[k] % w, y = cl[k] / w;
      for (int dy = -1; dy <= 1; ++dy)
        for (int dx = -1; dx <= 1; ++dx) {
          const int u = x + dx, t = y + dy;
          if (u < 0 || t < 0 || u >= w || t >= h) continue;
          const size_t c = (size_t)t * w + u;
          if (seen[c] == 2) { seen[c] = 3; cl.push_back((int)c); }
        }
    }
    if ((int)cl.size() >= min_cells) { g.n_clusters += 1; g.n_block += (int)cl.size(); }
  }
  return g;
}

void write_file(const std::string& path, const void* data, size_t n) {
  std::FILE* f = std::fopen(path.c_str(), "wb");
  if (!f) throw std::runtime_error("파일을 못 엶: " + path);
  std::fwrite(data, 1, n, f);
  std::fclose(f);
}


// ── 녹화 (decider.h Video)
namespace {
constexpr int kVW = 1280, kVH = 720;                               // 화면
constexpr int kMapX = 10, kMapY = 30, kMapW = 700, kMapH = 656;    // 지도 칸 (그 아래 두 줄은 범례)
constexpr int kCamX = 730, kCamY = 30, kCamW = 540, kCamH = 405;   // 카메라 칸
constexpr int kTxtX = 730, kTxtY = 446;                            // 글 시작
constexpr int kInW = 200, kInH = 150, kInX = kCamX + kCamW - kInW - 8, kInY = kCamY + 8;   // 모델에 보낸 그림 (작게)
std::string ass_time(double t) {
  const long cs = std::lround(t * 100.0);
  char b[40];
  std::snprintf(b, sizeof b, "%ld:%02ld:%02ld.%02ld", cs / 360000, cs / 6000 % 60, cs / 100 % 60, cs % 100);
  return b;
}
std::string ass_text(const std::string& s) {
  std::string o;
  for (char ch : s) {
    if (ch == '\n') o += "\\N";
    else o += ch;
  }
  return o;
}
void draw_ring(Img& im, int cu, int cv, int r0, int r1, uint8_t r, uint8_t g, uint8_t b) {
  for (int dy = -r1; dy <= r1; ++dy)
    for (int dx = -r1; dx <= r1; ++dx) {
      const int d2 = dx * dx + dy * dy;
      if (d2 >= r0 * r0 && d2 <= r1 * r1) im.set(cu + dx, cv + dy, r, g, b);
    }
}
void draw_thick(Img& im, int x0, int y0, int x1, int y1, uint8_t r, uint8_t g, uint8_t b) {
  for (int q = 0; q < 4; ++q) draw_line(im, x0 + (q & 1), y0 + (q >> 1), x1 + (q & 1), y1 + (q >> 1), r, g, b);
}
}  // namespace

struct Video::Impl {
  std::FILE* ff = nullptr;
  std::string prefix, events, map_title, cam_title;
  double fps = 10.0;
  long n = 0;
  std::vector<uint8_t> last;
};

Video::Video(const std::string& prefix, double fps, const std::string& map_title, const std::string& cam_title) : p_(new Impl) {
  p_->prefix = prefix;
  p_->fps = fps;
  p_->map_title = map_title;
  p_->cam_title = cam_title;
  char cmd[1024];
  std::snprintf(cmd, sizeof cmd, "ffmpeg -loglevel error -y -f rawvideo -pix_fmt rgb24 -s %dx%d -r %.6f -i - -c:v libx264 -preset veryfast -crf 12 -pix_fmt yuv444p '%s.raw.mp4'", kVW, kVH, fps, prefix.c_str());
  p_->ff = popen(cmd, "w");
  if (!p_->ff) throw std::runtime_error("ffmpeg 을 못 엶 (녹화)");
}
Video::~Video() {
  if (p_ && p_->ff) close();
}

void Video::frame(const VidFrame& f) {
  Img im{kVW, kVH, std::vector<uint8_t>((size_t)kVW * kVH * 3, 26)};
  const float sc = std::min((float)kMapW / (float)f.gw, (float)kMapH / (float)f.gh);
  const float mx0 = (float)kMapX + ((float)kMapW - (float)f.gw * sc) * 0.5f, my0 = (float)kMapY + ((float)kMapH - (float)f.gh * sc) * 0.5f;
  const int pw = (int)std::floor((float)f.gw * sc), ph = (int)std::floor((float)f.gh * sc);
  for (int v = 0; v < ph; ++v) {
    const int sy0 = (int)std::floor((float)v / sc), sy1 = std::max(sy0 + 1, (int)std::ceil((float)(v + 1) / sc));
    for (int u = 0; u < pw; ++u) {
      const int sx0 = (int)std::floor((float)u / sc), sx1 = std::max(sx0 + 1, (int)std::ceil((float)(u + 1) / sc));
      int cls = 0, low = 0;   // 0 모름 · 1 빈 곳 · 2 막힘
      for (int yy = sy0; yy < sy1; ++yy)
        for (int x = sx0; x < sx1; ++x) {
          const int y = f.gh - 1 - yy;   // 위가 +y
          if (x < 0 || y < 0 || x >= f.gw || y >= f.gh) continue;
          if (f.low && f.low[(size_t)y * f.gw + x]) low = 1;
          const int16_t l = f.lmap[(size_t)y * f.size + x];
          if (l == f.unknown_l) continue;
          cls = std::max(cls, l >= f.occ_l ? 2 : 1);
        }
      const int X = (int)mx0 + u, Y = (int)my0 + v;
      if (cls == 2) im.set(X, Y, 0, 0, 0);
      else if (low) cls == 1 ? im.set(X, Y, 255, 196, 120) : im.set(X, Y, 168, 126, 84);
      else if (cls == 1) im.set(X, Y, 245, 245, 245);
      else im.set(X, Y, 112, 112, 112);
    }
  }
  const auto to_px = [&](float cx, float cy, int* u, int* v) {
    *u = (int)std::lround(mx0 + cx * sc);
    *v = (int)std::lround(my0 + ((float)f.gh - cy) * sc);
  };
  if (f.trail)
    for (size_t k = 2; k + 1 < f.trail->size(); k += 2) {
      int u0, v0, u1, v1;
      to_px((*f.trail)[k - 2], (*f.trail)[k - 1], &u0, &v0);
      to_px((*f.trail)[k], (*f.trail)[k + 1], &u1, &v1);
      draw_line(im, u0, v0, u1, v1, 150, 120, 230);
    }
  int ru, rv;
  to_px(f.rob_cx, f.rob_cy, &ru, &rv);
  if (f.wp && f.wp_n > 0) {   // 계획 길: 로봇 → 남은 웨이포인트
    int u0 = ru, v0 = rv;
    for (int k = std::clamp(f.wp_i, 0, f.wp_n - 1); k < f.wp_n; ++k) {
      int u1, v1;
      to_px((float)f.wp[2 * k] + 0.5f, (float)f.wp[2 * k + 1] + 0.5f, &u1, &v1);
      draw_thick(im, u0, v0, u1, v1, 0, 110, 255);
      u0 = u1; v0 = v1;
    }
  }
  if (f.cand && !f.cand->empty()) {
    std::vector<Cand> tmp(f.cand->size());
    std::vector<int> dots(2 * tmp.size());
    for (size_t k = 0; k < tmp.size(); ++k) {
      tmp[k].key = (*f.cand)[k].key;
      to_px((float)(*f.cand)[k].cx + 0.5f, (float)(*f.cand)[k].cy + 0.5f, &dots[2 * k], &dots[2 * k + 1]);
    }
    place_tags(im, tmp, dots, ru, rv);
    for (size_t k = 0; k < tmp.size(); ++k) {
      const VidCand& c = (*f.cand)[k];
      const int u = dots[2 * k], v = dots[2 * k + 1];
      if (c.gt_blk) {
        for (int q = 0; q < 2; ++q) {
          const int h = 12 + q;
          draw_line(im, u - h, v - h, u + h, v - h, 255, 140, 30); draw_line(im, u - h, v + h, u + h, v + h, 255, 140, 30);
          draw_line(im, u - h, v - h, u - h, v + h, 255, 140, 30); draw_line(im, u + h, v - h, u + h, v + h, 255, 140, 30);
        }
      }
      if (c.chosen) draw_ring(im, u, v, 6, 9, 40, 210, 60);
      if (c.veto) { draw_thick(im, u - 8, v - 8, u + 8, v + 8, 255, 40, 220); draw_thick(im, u - 8, v + 8, u + 8, v - 8, 255, 40, 220); }
    }
  }
  if (f.goal_cx >= 0) {
    int u, v;
    to_px((float)f.goal_cx + 0.5f, (float)f.goal_cy + 0.5f, &u, &v);
    draw_thick(im, u - 9, v, u + 9, v, 0, 110, 255);
    draw_thick(im, u, v - 9, u, v + 9, 0, 110, 255);
  }
  {   // 로봇: 머리 방향 세모
    const float fu = (float)ru, fv = (float)rv, c = std::cos(f.rob_th), s = -std::sin(f.rob_th);
    const float ax = fu + 12.f * c, ay = fv + 12.f * s, bx = fu - 7.f * c - 7.f * s, by = fv - 7.f * s + 7.f * c, cx = fu - 7.f * c + 7.f * s, cy = fv - 7.f * s - 7.f * c;
    const auto edge = [](float px_, float py_, float qx, float qy, float x, float y) { return (qx - px_) * (y - py_) - (qy - py_) * (x - px_); };
    for (int y = rv - 14; y <= rv + 14; ++y)
      for (int x = ru - 14; x <= ru + 14; ++x) {
        const float e0 = edge(ax, ay, bx, by, (float)x, (float)y), e1 = edge(bx, by, cx, cy, (float)x, (float)y), e2 = edge(cx, cy, ax, ay, (float)x, (float)y);
        if ((e0 >= 0 && e1 >= 0 && e2 >= 0) || (e0 <= 0 && e1 <= 0 && e2 <= 0)) im.set(x, y, 220, 30, 200);
      }
  }
  for (int v = 0; v < kCamH; ++v)   // 카메라 그림: 가까운 픽셀로 확대만
    for (int u = 0; u < kCamW; ++u) {
      if (!f.cam || f.cam->empty()) { im.set(kCamX + u, kCamY + v, 8, 8, 8); continue; }
      const uint8_t* p = &(*f.cam)[((size_t)(v * f.cam_h / kCamH) * f.cam_w + (size_t)(u * f.cam_w / kCamW)) * 3];
      im.set(kCamX + u, kCamY + v, p[0], p[1], p[2]);
    }
  if (f.sent && !f.sent->empty()) {   // 모델에 실제로 보낸 그림: 카메라 칸 오른쪽 위에 작게 (테두리 흰색, 방금 보낸 걸음은 초록 굵게)
    const int bw = f.sent_flash ? 4 : 2;
    for (int v = -bw; v < kInH + bw; ++v)
      for (int u = -bw; u < kInW + bw; ++u) {
        if (u < 0 || v < 0 || u >= kInW || v >= kInH) {
          if (f.sent_flash) im.set(kInX + u, kInY + v, 40, 210, 60); else im.set(kInX + u, kInY + v, 255, 255, 255);
          continue;
        }
        const uint8_t* p = &(*f.sent)[((size_t)(v * f.cam_h / kInH) * f.cam_w + (size_t)(u * f.cam_w / kInW)) * 3];
        im.set(kInX + u, kInY + v, p[0], p[1], p[2]);
      }
    char ip[64];
    std::snprintf(ip, sizeof ip, "{\\an7\\pos(%d,%d)}", kInX, kInY + kInH + 6);
    p_->events += "Dialogue: 1," + ass_time((double)p_->n / p_->fps) + "," + ass_time((double)(p_->n + 1) / p_->fps) + ",Small,,0,0,0,," + ip + ass_text(f.sent_label) + "\n";
  }
  std::fwrite(im.rgb.data(), 1, im.rgb.size(), p_->ff);
  p_->last = im.rgb;
  char pos[64];
  std::snprintf(pos, sizeof pos, "{\\an7\\pos(%d,%d)}", kTxtX, kTxtY);
  p_->events += "Dialogue: 0," + ass_time((double)p_->n / p_->fps) + "," + ass_time((double)(p_->n + 1) / p_->fps) + ",Mono,,0,0,0,," + pos + ass_text(f.text) + "\n";
  p_->n += 1;
}

void Video::hold(double sec, const std::string& text) {
  if (p_->last.empty()) return;
  const long k = std::max(1L, std::lround(sec * p_->fps));
  for (long i = 0; i < k; ++i) std::fwrite(p_->last.data(), 1, p_->last.size(), p_->ff);
  char pos[64];
  std::snprintf(pos, sizeof pos, "{\\an7\\pos(%d,%d)}", kTxtX, kTxtY);
  p_->events += "Dialogue: 0," + ass_time((double)p_->n / p_->fps) + "," + ass_time((double)(p_->n + k) / p_->fps) + ",Mono,,0,0,0,," + pos + ass_text(text) + "\n";
  p_->n += k;
}

void Video::close() {
  if (!p_->ff) return;
  const int rc0 = pclose(p_->ff);
  p_->ff = nullptr;
  const std::string end = ass_time((double)p_->n / p_->fps);
  std::string a =
      "[Script Info]\nScriptType: v4.00+\nPlayResX: 1280\nPlayResY: 720\nWrapStyle: 2\nScaledBorderAndShadow: yes\n\n"
      "[V4+ Styles]\nFormat: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding\n"
      "Style: Mono,DejaVu Sans Mono,16,&H00FFFFFF,&H00FFFFFF,&H00000000,&H00000000,0,0,0,0,100,100,0,0,1,1,0,7,0,0,0,1\n"
      "Style: Small,DejaVu Sans Mono,13,&H00FFFFFF,&H00FFFFFF,&H00000000,&H00000000,0,0,0,0,100,100,0,0,1,1,0,7,0,0,0,1\n\n"
      "[Events]\nFormat: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n";
  char b[256];
  std::snprintf(b, sizeof b, "{\\an7\\pos(%d,6)}", kMapX);
  a += "Dialogue: 0,0:00:00.00," + end + ",Mono,,0,0,0,," + b + ass_text(p_->map_title) + "\n";
  std::snprintf(b, sizeof b, "{\\an7\\pos(%d,6)}", kCamX);
  a += "Dialogue: 0,0:00:00.00," + end + ",Mono,,0,0,0,," + b + ass_text(p_->cam_title) + "\n";
  std::snprintf(b, sizeof b, "{\\an7\\pos(%d,%d)}", kMapX, kMapY + kMapH + 2);
  a += "Dialogue: 0,0:00:00.00," + end + ",Small,,0,0,0,," + b +
       "{\\c&H00D2FF&}yellow dot{\\c&HFFFFFF&} + {\\c&H3C3CFF&}red letter tag{\\c&HFFFFFF&} = candidate frontier   {\\c&H3CD228&}green ring{\\c&HFFFFFF&} = chosen   "
       "{\\c&HDC28FF&}magenta X{\\c&HFFFFFF&} = model says blocked\\N{\\c&H1E8CFF&}orange box{\\c&HFFFFFF&} = blocked by ground truth   {\\c&HFF6E00&}blue cross / line{\\c&HFFFFFF&} = current goal / planned path   "
       "{\\c&H78C4FF&}orange cells{\\c&HFFFFFF&} = low obstacles (truth; lidar cannot see)\n";
  a += p_->events;
  write_file(p_->prefix + ".ass", a.data(), a.size());
  char cmd[2048];
  std::snprintf(cmd, sizeof cmd, "ffmpeg -loglevel error -y -i '%s.raw.mp4' -vf \"ass='%s.ass'\" -c:v libx264 -preset medium -crf 20 -pix_fmt yuv420p -movflags +faststart '%s.mp4'", p_->prefix.c_str(), p_->prefix.c_str(),
                p_->prefix.c_str());
  const int rc1 = std::system(cmd);
  if (rc0 != 0 || rc1 != 0) std::fprintf(stderr, "녹화: ffmpeg 끝 코드 %d · %d (%s)\n", rc0, rc1, p_->prefix.c_str());
  else std::printf("녹화: %s.mp4 (%ld 장, %.1f s)\n", p_->prefix.c_str(), p_->n, (double)p_->n / p_->fps);
}

}  // namespace dec
