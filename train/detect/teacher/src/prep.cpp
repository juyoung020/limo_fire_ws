#include "prep.h"

#include <algorithm>
#include <array>
#include <cmath>

#define STB_IMAGE_IMPLEMENTATION
#define STBI_ONLY_JPEG
#define STBI_ONLY_PNG
#define STBI_ONLY_GIF
#define STBI_ONLY_BMP
#include "stb_image.h"

namespace {
constexpr int kPrecisionBits = 32 - 8 - 2;  // Pillow PRECISION_BITS

double bilinear_filter(double x) {
  if (x < 0.0) x = -x;
  if (x < 1.0) return 1.0 - x;
  return 0.0;
}

// Pillow precompute_coeffs + normalize_coeffs_8bpc. bounds[2·xx] = xmin, bounds[2·xx+1] = 개수.
int coeffs(int in_size, double in0, double in1, int out_size, std::vector<int>& bounds, std::vector<int32_t>& kk) {
  double scale = (in1 - in0) / out_size, filterscale = scale;
  if (filterscale < 1.0) filterscale = 1.0;
  double support = 1.0 * filterscale;
  int ksize = int(std::ceil(support)) * 2 + 1;
  std::vector<double> pre(size_t(out_size) * ksize, 0.0);
  bounds.assign(size_t(out_size) * 2, 0);
  for (int xx = 0; xx < out_size; xx++) {
    double center = in0 + (xx + 0.5) * scale;
    double ww = 0.0, ss = 1.0 / filterscale;
    int xmin = int(center - support + 0.5);
    if (xmin < 0) xmin = 0;
    int xmax = int(center + support + 0.5);
    if (xmax > in_size) xmax = in_size;
    xmax -= xmin;
    double* k = &pre[size_t(xx) * ksize];
    for (int x = 0; x < xmax; x++) {
      double w = bilinear_filter((x + xmin - center + 0.5) * ss);
      k[x] = w;
      ww += w;
    }
    for (int x = 0; x < xmax; x++)
      if (ww != 0.0) k[x] /= ww;
    bounds[xx * 2] = xmin;
    bounds[xx * 2 + 1] = xmax;
  }
  kk.resize(pre.size());
  for (size_t i = 0; i < pre.size(); i++)
    kk[i] = pre[i] < 0 ? int32_t(-0.5 + pre[i] * (1 << kPrecisionBits)) : int32_t(0.5 + pre[i] * (1 << kPrecisionBits));
  return ksize;
}
inline uint8_t clip8(int64_t in) {
  if (in >= (int64_t(1) << kPrecisionBits << 8)) return 255;
  if (in <= 0) return 0;
  return uint8_t(in >> kPrecisionBits);
}

}  // namespace

void pil_resize_rgb(const uint8_t* src, int sw, int sh, int sstride, uint8_t* dst, int dw, int dh) {
  std::vector<int> bh, bv;
  std::vector<int32_t> kh, kv;
  int ksh = coeffs(sw, 0.0, double(sw), dw, bh, kh);
  int ksv = coeffs(sh, 0.0, double(sh), dh, bv, kv);
  bool need_h = dw != sw, need_v = dh != sh;
  int y_first = bv[0], y_last = bv[(dh - 1) * 2] + bv[(dh - 1) * 2 + 1];
  // 가로: 세로에 쓰일 행 [y_first, y_last) 만
  std::vector<uint8_t> tmp;
  const uint8_t* mid = src;
  int mid_stride = sstride, mid_y0 = 0;
  if (need_h) {
    int rows = y_last - y_first;
    tmp.resize(size_t(rows) * dw * 3);
    for (int y = 0; y < rows; y++) {
      const uint8_t* in = src + size_t(y + y_first) * sstride;
      uint8_t* out = &tmp[size_t(y) * dw * 3];
      for (int xx = 0; xx < dw; xx++) {
        int xmin = bh[xx * 2], xmax = bh[xx * 2 + 1];
        const int32_t* k = &kh[size_t(xx) * ksh];
        int64_t s0 = 1 << (kPrecisionBits - 1), s1 = s0, s2 = s0;
        for (int x = 0; x < xmax; x++) {
          const uint8_t* p = in + (x + xmin) * 3;
          s0 += int64_t(p[0]) * k[x];
          s1 += int64_t(p[1]) * k[x];
          s2 += int64_t(p[2]) * k[x];
        }
        out[xx * 3] = clip8(s0), out[xx * 3 + 1] = clip8(s1), out[xx * 3 + 2] = clip8(s2);
      }
    }
    mid = tmp.data();
    mid_stride = dw * 3;
    mid_y0 = y_first;
  }
  int mw = need_h ? dw : sw;
  if (need_v) {
    for (int yy = 0; yy < dh; yy++) {
      int ymin = bv[yy * 2] - mid_y0, ymax = bv[yy * 2 + 1];
      const int32_t* k = &kv[size_t(yy) * ksv];
      uint8_t* out = dst + size_t(yy) * dw * 3;
      for (int xx = 0; xx < mw; xx++) {
        int64_t s0 = 1 << (kPrecisionBits - 1), s1 = s0, s2 = s0;
        for (int y = 0; y < ymax; y++) {
          const uint8_t* p = mid + size_t(y + ymin) * mid_stride + xx * 3;
          s0 += int64_t(p[0]) * k[y];
          s1 += int64_t(p[1]) * k[y];
          s2 += int64_t(p[2]) * k[y];
        }
        out[xx * 3] = clip8(s0), out[xx * 3 + 1] = clip8(s1), out[xx * 3 + 2] = clip8(s2);
      }
    }
  } else {
    for (int y = 0; y < dh; y++)
      std::copy(mid + size_t(y + (need_h ? 0 : 0)) * mid_stride, mid + size_t(y) * mid_stride + size_t(mw) * 3,
                dst + size_t(y) * dw * 3);
  }
}


bool load_rgb(const std::string& path, std::vector<uint8_t>& rgb, int& w, int& h) {
  int c;
  uint8_t* p = stbi_load(path.c_str(), &w, &h, &c, 3);
  if (!p) return false;
  rgb.assign(p, p + (size_t)w * h * 3);
  stbi_image_free(p);
  return true;
}

void prepare_sam3(const uint8_t* rgb, int w, int h, float* out, int S) {
  std::vector<uint8_t> r((size_t)S * S * 3);
  pil_resize_rgb(rgb, w, h, w * 3, r.data(), S, S);
  // transformers: (x * (1/255) − 0.5) / 0.5 를 float 로
  const float rs = 0.00392156862745098f;
  for (int c = 0; c < 3; ++c)
    for (size_t i = 0; i < (size_t)S * S; ++i) out[c * (size_t)S * S + i] = ((float)r[i * 3 + c] * rs - 0.5f) / 0.5f;
}
