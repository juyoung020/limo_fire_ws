#include "cache.h"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>

bool Cache::open(const std::string& path) {
  FILE* f = std::fopen(path.c_str(), "rb");
  if (!f) return false;
  char magic[8];
  uint32_t n, S;
  if (std::fread(magic, 1, 8, f) != 8 || std::memcmp(magic, "YCACHE0", 7) != 0 || (magic[7] != '1' && magic[7] != '2'))
    return std::fclose(f), false;
  bool v2 = magic[7] == '2';
  std::fread(&n, 4, 1, f);
  std::fread(&S, 4, 1, f);
  S_ = (int)S;
  rec_.resize(n);
  std::fread(rec_.data(), sizeof(CacheRec), n, f);
  uint32_t nb;
  std::fread(&nb, 4, 1, f);
  box_.resize(nb);
  std::fread(box_.data(), sizeof(CacheBox), nb, f);
  uint64_t bl;
  std::fread(&bl, 8, 1, f);
  blob_len_ = bl;
  if (cudaMallocHost(&blob_, bl) != cudaSuccess) return std::fclose(f), false;
  size_t got = 0;
  while (got < bl) {
    size_t r = std::fread(blob_ + got, 1, std::min<size_t>(bl - got, 1u << 26), f);
    if (r == 0) break;
    got += r;
  }
  if (v2 && got == bl) {
    mask_.resize(n);
    std::fread(mask_.data(), sizeof(CacheMask), n, f);
    uint64_t nr;
    std::fread(&nr, 8, 1, f);
    runs_.resize(nr);
    if (std::fread(runs_.data(), 4, nr, f) != nr) return std::fclose(f), false;
  }
  std::fclose(f);
  std::ifstream fn(path + ".names.txt");
  std::string l;
  while (std::getline(fn, l)) names.push_back(l);
  return got == bl;
}
Cache::~Cache() {
  if (blob_) cudaFreeHost(blob_);
}

void decode_mask(const uint32_t* runs, uint32_t nrun, uint8_t* dst, int size) {
  size_t p = 0, n = size_t(size) * size;
  for (uint32_t i = 0; i < nrun && p < n; ++i) {
    size_t len = std::min<size_t>(runs[i] >> 8, n - p);
    std::memset(dst + p, int(runs[i] & 255), len);
    p += len;
  }
  if (p < n) std::memset(dst + p, 0, n - p);
}

void resize_linear(const uint8_t* src, int sw, int sh, uint8_t* dst, int dw, int dh) {
  double fx = (double)sw / dw, fy = (double)sh / dh;
  std::vector<int> x0(dw), x1(dw);
  std::vector<float> ax(dw);
  for (int x = 0; x < dw; ++x) {
    double s = (x + 0.5) * fx - 0.5;
    int i = (int)std::floor(s);
    float a = (float)(s - i);
    if (i < 0) i = 0, a = 0.f;
    if (i >= sw - 1) i = sw - 1, a = 0.f;
    x0[x] = i;
    x1[x] = std::min(i + 1, sw - 1);
    ax[x] = a;
  }
  for (int y = 0; y < dh; ++y) {
    double s = (y + 0.5) * fy - 0.5;
    int j = (int)std::floor(s);
    float b = (float)(s - j);
    if (j < 0) j = 0, b = 0.f;
    if (j >= sh - 1) j = sh - 1, b = 0.f;
    int j1 = std::min(j + 1, sh - 1);
    const uint8_t *r0 = src + (size_t)j * sw * 3, *r1 = src + (size_t)j1 * sw * 3;
    uint8_t* o = dst + (size_t)y * dw * 3;
    for (int x = 0; x < dw; ++x)
      for (int c = 0; c < 3; ++c) {
        float v0 = r0[x0[x] * 3 + c] * (1 - ax[x]) + r0[x1[x] * 3 + c] * ax[x];
        float v1 = r1[x0[x] * 3 + c] * (1 - ax[x]) + r1[x1[x] * 3 + c] * ax[x];
        float v = v0 * (1 - b) + v1 * b;
        o[x * 3 + c] = (uint8_t)std::lround(std::min(std::max(v, 0.f), 255.f));
      }
  }
}
