// 학습 캐시 만들기: data/yolo/{images,labels}/<split> → <out>.ycache (+ .names.txt)
// 사용: build_cache <yolo 폴더> <split> <out.ycache> [S=640] [threads=32] [quality=95]
#include <turbojpeg.h>

#include <algorithm>
#include <atomic>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <sstream>
#include <thread>

#include "cache.h"

#define STB_IMAGE_IMPLEMENTATION
#define STBI_ONLY_PNG
#define STBI_ONLY_GIF
#define STBI_ONLY_BMP
#define STBI_ONLY_JPEG
#include "../../teacher/third_party/stb_image.h"

namespace fs = std::filesystem;

int main(int argc, char** argv) {
  if (argc < 4) {
    std::fprintf(stderr, "사용: build_cache <yolo 폴더> <split> <out.ycache> [S] [threads] [quality]\n");
    return 1;
  }
  std::string root = argv[1], split = argv[2], out = argv[3];
  int S = argc > 4 ? std::stoi(argv[4]) : 640, T = argc > 5 ? std::stoi(argv[5]) : 32,
      Q = argc > 6 ? std::stoi(argv[6]) : 95;
  std::vector<std::string> imgs;
  for (auto& e : fs::directory_iterator(root + "/images/" + split)) imgs.push_back(e.path().filename().string());
  std::sort(imgs.begin(), imgs.end());
  int n = (int)imgs.size();
  std::vector<CacheRec> rec(n);
  std::vector<std::vector<CacheBox>> boxes(n);
  std::vector<std::vector<uint8_t>> jpg(n);
  std::vector<char> ok(n, 0);
  std::atomic<int> next{0}, done{0}, bad{0};
  auto work = [&]() {
    tjhandle dec = tjInitDecompress(), enc = tjInitCompress();
    std::vector<uint8_t> rgb, small;
    for (;;) {
      int i = next++;
      if (i >= n) break;
      std::string p = root + "/images/" + split + "/" + imgs[i];
      std::ifstream f(p, std::ios::binary);
      std::vector<uint8_t> raw((std::istreambuf_iterator<char>(f)), {});
      int w = 0, h = 0;
      bool got = false;
      int sub, cs;
      if (raw.size() > 3 && raw[0] == 0xFF && raw[1] == 0xD8 &&
          tjDecompressHeader3(dec, raw.data(), raw.size(), &w, &h, &sub, &cs) == 0) {
        rgb.resize((size_t)w * h * 3);
        got = tjDecompress2(dec, raw.data(), raw.size(), rgb.data(), w, 0, h, TJPF_RGB, TJFLAG_ACCURATEDCT) == 0;
      }
      if (!got) {
        int c;
        uint8_t* q = stbi_load_from_memory(raw.data(), (int)raw.size(), &w, &h, &c, 3);
        if (q) {
          rgb.assign(q, q + (size_t)w * h * 3);
          stbi_image_free(q);
          got = true;
        }
      }
      if (!got || w < 2 || h < 2) {
        ++bad;
        continue;
      }
      // Ultralytics load_image: r = S / max(h, w), 다르면 INTER_LINEAR 로 (round(w·r), round(h·r)), 최소 S 의 범위
      double r = (double)S / std::max(w, h);
      int nw = w, nh = h;
      const uint8_t* src = rgb.data();
      if (r != 1.0) {
        nw = std::min((int)std::ceil(w * r), S);
        nh = std::min((int)std::ceil(h * r), S);
        nw = std::max(nw, 1);
        nh = std::max(nh, 1);
        small.resize((size_t)nw * nh * 3);
        resize_linear(rgb.data(), w, h, small.data(), nw, nh);
        src = small.data();
      }
      unsigned char* buf = nullptr;
      unsigned long len = 0;
      if (tjCompress2(enc, src, nw, 0, nh, TJPF_RGB, &buf, &len, TJSAMP_444, Q, TJFLAG_ACCURATEDCT) != 0) {
        ++bad;
        continue;
      }
      jpg[i].assign(buf, buf + len);
      tjFree(buf);
      rec[i].w = (uint16_t)nw;
      rec[i].h = (uint16_t)nh;
      rec[i].orig_w = (uint16_t)std::min(w, 65535);
      rec[i].orig_h = (uint16_t)std::min(h, 65535);
      std::string stem = imgs[i].substr(0, imgs[i].rfind('.'));
      std::ifstream fl(root + "/labels/" + split + "/" + stem + ".txt");
      std::string line;
      while (std::getline(fl, line)) {
        std::istringstream ss(line);
        CacheBox b;
        if (ss >> b.cls >> b.cx >> b.cy >> b.w >> b.h) boxes[i].push_back(b);
      }
      ok[i] = 1;
      int d = ++done;
      if (d % 5000 == 0) std::fprintf(stderr, "%d / %d\n", d, n);
    }
    tjDestroy(dec);
    tjDestroy(enc);
  };
  std::vector<std::thread> th;
  for (int t = 0; t < T; ++t) th.emplace_back(work);
  for (auto& t : th) t.join();
  // 쓰기
  std::vector<CacheRec> R;
  std::vector<CacheBox> Bx;
  std::vector<std::string> names;
  uint64_t off = 0;
  for (int i = 0; i < n; ++i) {
    if (!ok[i]) continue;
    CacheRec c = rec[i];
    c.jpg_off = off;
    c.jpg_len = (uint32_t)jpg[i].size();
    c.box_off = (uint32_t)Bx.size();
    c.nbox = (uint16_t)boxes[i].size();
    c.pad = 0;
    Bx.insert(Bx.end(), boxes[i].begin(), boxes[i].end());
    off += jpg[i].size();
    R.push_back(c);
    names.push_back(imgs[i]);
  }
  FILE* f = std::fopen((out + ".tmp").c_str(), "wb");
  std::fwrite("YCACHE01", 1, 8, f);
  uint32_t nn = (uint32_t)R.size(), SS = (uint32_t)S, nb = (uint32_t)Bx.size();
  std::fwrite(&nn, 4, 1, f);
  std::fwrite(&SS, 4, 1, f);
  std::fwrite(R.data(), sizeof(CacheRec), R.size(), f);
  std::fwrite(&nb, 4, 1, f);
  std::fwrite(Bx.data(), sizeof(CacheBox), Bx.size(), f);
  std::fwrite(&off, 8, 1, f);
  for (int i = 0; i < n; ++i)
    if (ok[i]) std::fwrite(jpg[i].data(), 1, jpg[i].size(), f);
  std::fclose(f);
  std::rename((out + ".tmp").c_str(), out.c_str());
  std::ofstream fn(out + ".names.txt");
  for (auto& s : names) fn << s << "\n";
  std::fprintf(stderr, "사진 %u 장(못 읽음 %d), 상자 %u 개, JPEG %.2f GB → %s\n", nn, bad.load(), nb, off / 1e9,
               out.c_str());
  return 0;
}
