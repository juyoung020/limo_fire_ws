// GPU 증강: 출력 화소마다 역 affine → 캔버스(Mosaic 4 장 또는 한 장, 빈 곳 114) 양선형(cv2.warpAffine) →
// HSV 이득(OpenCV 8 비트 HSV, 색상 0..180) → 좌우 뒤집기 → /255 → NHWC act_t.
#include "data.h"

namespace {
__device__ inline void canvas_px(const uint8_t* img, const SampleParam& p, int x, int y, float* rgb) {
  if (x < 0 || y < 0 || x >= p.cw || y >= p.ch) {
    rgb[0] = rgb[1] = rgb[2] = 114.f;
    return;
  }
  for (int t = 0; t < p.ntile; ++t) {
    if (x >= p.ta[t][0] && x < p.ta[t][2] && y >= p.ta[t][1] && y < p.ta[t][3]) {
      int sx = x - p.pad[t][0], sy = y - p.pad[t][1];
      const uint8_t* q = img + p.off[t] + (int64_t(sy) * p.sw[t] + sx) * 3;
      rgb[0] = q[0], rgb[1] = q[1], rgb[2] = q[2];
      return;
    }
  }
  rgb[0] = rgb[1] = rgb[2] = 114.f;
}
// OpenCV RGB2HSV(8 비트, 색상 0..180) 고정소수점 식
__device__ inline void rgb2hsv(int r, int g, int b, int& h, int& s, int& v) {
  const int shift = 12;
  int vmax = max(r, max(g, b)), vmin = min(r, min(g, b)), diff = vmax - vmin;
  int sdiv = vmax ? (int)rintf((255 << shift) / (float)vmax) : 0;
  int hdiv = diff ? (int)rintf((180 << shift) / (6.f * diff)) : 0;
  s = (diff * sdiv + (1 << (shift - 1))) >> shift;
  int hh;
  if (vmax == r)
    hh = g - b;
  else if (vmax == g)
    hh = b - r + 2 * diff;
  else
    hh = r - g + 4 * diff;
  hh = (hh * hdiv + (1 << (shift - 1))) >> shift;
  if (hh < 0) hh += 180;
  h = hh;
  v = vmax;
}
__device__ inline void hsv2rgb(int h8, int s8, int v8, float* out) {
  float h = h8 * (6.f / 180.f), s = s8 / 255.f, v = v8 / 255.f, r, g, b;
  if (s == 0.f) {
    r = g = b = v;
  } else {
    int sector = (int)floorf(h);
    h -= sector;
    if ((unsigned)sector >= 6u) {
      sector = sector < 0 ? sector + 6 : sector - 6;
    }
    float tab[4] = {v, v * (1.f - s), v * (1.f - s * h), v * (1.f - s * (1.f - h))};
    const int sd[6][3] = {{1, 3, 0}, {1, 0, 2}, {3, 0, 1}, {0, 2, 1}, {0, 1, 3}, {2, 1, 0}};
    b = tab[sd[sector][0]];
    g = tab[sd[sector][1]];
    r = tab[sd[sector][2]];
  }
  out[0] = fminf(fmaxf(rintf(r * 255.f), 0.f), 255.f);
  out[1] = fminf(fmaxf(rintf(g * 255.f), 0.f), 255.f);
  out[2] = fminf(fmaxf(rintf(b * 255.f), 0.f), 255.f);
}
__global__ void augment_k(const uint8_t* img, const SampleParam* sps, int B, int S, int C, act_t* out) {
  int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
  if (i >= int64_t(B) * S * S) return;
  int x = int(i % S), y = int((i / S) % S), b = int(i / (int64_t(S) * S));
  const SampleParam& p = sps[b];
  act_t* o = out + i * C;
  for (int c = 3; c < C; ++c) o[c] = from_f<act_t>(0.f);
  if (p.ntile == 0) {
    o[0] = o[1] = o[2] = from_f<act_t>(114.f / 255.f);
    return;
  }
  int xs = p.flip ? S - 1 - x : x;
  float u = p.Mi[0] * xs + p.Mi[1] * y + p.Mi[2], v = p.Mi[3] * xs + p.Mi[4] * y + p.Mi[5];
  int u0 = (int)floorf(u), v0 = (int)floorf(v);
  float fu = u - u0, fv = v - v0;
  float c00[3], c01[3], c10[3], c11[3], rgb[3];
  canvas_px(img, p, u0, v0, c00);
  canvas_px(img, p, u0 + 1, v0, c01);
  canvas_px(img, p, u0, v0 + 1, c10);
  canvas_px(img, p, u0 + 1, v0 + 1, c11);
  for (int c = 0; c < 3; ++c) {
    float val = (c00[c] * (1 - fu) + c01[c] * fu) * (1 - fv) + (c10[c] * (1 - fu) + c11[c] * fu) * fv;
    rgb[c] = fminf(fmaxf(rintf(val), 0.f), 255.f);
  }
  if (p.hsv[0] != 1.f || p.hsv[1] != 1.f || p.hsv[2] != 1.f) {
    int h, s, vv;
    rgb2hsv((int)rgb[0], (int)rgb[1], (int)rgb[2], h, s, vv);
    float hf = fmodf(h * p.hsv[0], 180.f);
    if (hf < 0) hf += 180.f;
    h = (int)hf;
    s = (int)fminf(fmaxf(s * p.hsv[1], 0.f), 255.f);
    vv = (int)fminf(fmaxf(vv * p.hsv[2], 0.f), 255.f);
    hsv2rgb(h, s, vv, rgb);
  }
  for (int c = 0; c < 3; ++c) o[c] = from_f<act_t>(rgb[c] / 255.f);
}
// 출력 마스크 칸 (mx, my) = 출력 화소 r×r 의 가운데(화소 번호 좌표 r·m + (r − 1)/2) → 뒤집기 → 역 affine → 가장 가까운
// 캔버스 화소 → 타일 → 원본 사진 좌표 → 번호 지도 칸(최근접) → 출력 정답 번호
__global__ void mask_k(const uint8_t* msk, const SampleParam* sps, int B, int S, int r, int16_t* out) {
  int M = S / r;
  int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
  if (i >= int64_t(B) * M * M) return;
  int mx = int(i % M), my = int((i / M) % M), b = int(i / (int64_t(M) * M));
  const SampleParam& p = sps[b];
  int16_t v = 0;
  if (p.ntile > 0) {
    float xo = r * mx + (r - 1) * 0.5f, yo = r * my + (r - 1) * 0.5f;
    float xs = p.flip ? S - 1 - xo : xo;
    float u = p.Mi[0] * xs + p.Mi[1] * yo + p.Mi[2], w = p.Mi[3] * xs + p.Mi[4] * yo + p.Mi[5];
    int x = (int)floorf(u + 0.5f), y = (int)floorf(w + 0.5f);
    if (x >= 0 && y >= 0 && x < p.cw && y < p.ch)
      for (int t = 0; t < p.ntile; ++t)
        if (x >= p.ta[t][0] && x < p.ta[t][2] && y >= p.ta[t][1] && y < p.ta[t][3]) {
          int sx = x - p.pad[t][0], sy = y - p.pad[t][1], n = p.msz[t];
          int qx = min(n - 1, (int)((sx + 0.5f) * n / p.sw[t])), qy = min(n - 1, (int)((sy + 0.5f) * n / p.sh[t]));
          int j = msk[p.moff[t] + int64_t(qy) * n + qx];
          if (j > 0 && p.lmap[t][j - 1] >= 0) v = int16_t(p.lmap[t][j - 1] + 1);
          break;
        }
  }
  out[i] = v;
}
}  // namespace

void mask_targets(const uint8_t* msk, const SampleParam* sp, int B, int S, int r, int16_t* out, cudaStream_t s) {
  int64_t n = int64_t(B) * (S / r) * (S / r);
  mask_k<<<cdiv(n, 256), 256, 0, s>>>(msk, sp, B, S, r, out);
}

void augment_batch(const uint8_t* img, const SampleParam* sp, int B, int S, int C, act_t* out, cudaStream_t s) {
  int64_t n = int64_t(B) * S * S;
  augment_k<<<cdiv(n, 256), 256, 0, s>>>(img, sp, B, S, C, out);
}
