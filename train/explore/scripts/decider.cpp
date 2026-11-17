// decider.h 구현: 판단 모델에 보내는 로봇 지도 사진(PNG — 후보 꼬리표 · 지나온 길 · 지난 목표)과 data URI.
#include "decider.h"

#include <zlib.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <stdexcept>

namespace dec {

namespace {

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
// dots: 후보의 점 자리 (x, y) 쌍. (rx, ry) 는 피할 자리(로봇; 없으면 음수)
void place_tags(Img& im, const std::vector<Cand>& cand, const std::vector<int>& dots, int rx, int ry) {
  static const int kDir[8][2] = {{0, -1}, {1, 0}, {0, 1}, {-1, 0}, {1, -1}, {1, 1}, {-1, 1}, {-1, -1}};
  static const int kRad[6] = {16, 28, 40, 54, 70, 88};   // 밀어 놓는 거리 (px) — 예전 꼴은 앞 셋만
  std::vector<int> tx(cand.size(), 0), ty(cand.size(), 0);
  for (size_t k = 0; k < cand.size(); ++k) {
    int best = -(1 << 20), bx = -1, by = 0;
    for (int o = 0; o < (g_clear_tags ? 48 : 24); ++o) {
      const int rad = kRad[o / 8];
      const int f = (kDir[o % 8][0] != 0 && kDir[o % 8][1] != 0) ? 7 : 10;   // 대각선은 0.7 배
      const int x = dots[2 * k] + kDir[o % 8][0] * rad * f / 10, y = dots[2 * k + 1] + kDir[o % 8][1] * rad * f / 10;
      if (x < kTagR + 1 || y < kTagR + 1 || x >= im.w - kTagR - 1 || y >= im.h - kTagR - 1) continue;
      int gap = 1 << 20;   // 가장 가까운 것과의 틈 (px, 클수록 좋음)
      const auto see = [&](int qx, int qy, int need) { gap = std::min(gap, (int)std::lround(std::hypot((double)(x - qx), (double)(y - qy))) - need); };
      for (size_t j = 0; j < cand.size(); ++j) {
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
    const int g = std::clamp(cand[k].key - 'A', 0, 7);
    const uint8_t lr = g_clear_tags ? kTagCol[g][0] : 220, lg = g_clear_tags ? kTagCol[g][1] : 30, lb = g_clear_tags ? kTagCol[g][2] : 30;
    draw_line(im, dots[2 * k], dots[2 * k + 1], tx[k], ty[k], lr, lg, lb);
    draw_line(im, dots[2 * k] + 1, dots[2 * k + 1], tx[k] + 1, ty[k], lr, lg, lb);
  }
  for (int k = (int)cand.size() - 1; k >= 0; --k) {
    if (!cand[k].object) { draw_disc(im, dots[2 * k], dots[2 * k + 1], kDotR, 255, 210, 0); continue; }
    const int h = kDotR + 2;   // 물체 후보: 검은 테 하늘색 네모
    for (int y = -h; y <= h; ++y)
      for (int x = -h; x <= h; ++x) {
        const bool edge = std::abs(x) == h || std::abs(y) == h;
        im.set(dots[2 * k] + x, dots[2 * k + 1] + y, edge ? 0 : 0, edge ? 0 : 200, edge ? 0 : 255);
      }
  }
  for (int k = (int)cand.size() - 1; k >= 0; --k)
    draw_tag(im, tx[k], ty[k], cand[k].key);
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

std::string png_uri(const std::vector<uint8_t>& png) { return "data:image/png;base64," + base64(png); }

}  // namespace dec
