// 증강 확인: 첫 배치를 PPM 으로(정답 상자 그림, 캐시에 마스크가 있으면 정답 마스크를 클래스 색으로 덧칠). 사용: aug_dump <cache> <out 폴더> [batch] [augment 1|0]
#include <cstdio>
#include <string>

#include "data.h"

int main(int argc, char** argv) {
  Cache c;
  if (!c.open(argv[1])) return 1;
  std::string out = argv[2];
  int B = argc > 3 ? std::stoi(argv[3]) : 8;
  AugCfg ac;
  ac.batch = B;
  ac.augment = argc > 4 ? std::stoi(argv[4]) != 0 : true;
  ac.epochs = 100;
  ac.in_c = 3;
  ac.masks = c.has_masks();
  Loader L;
  L.open(&c, ac, 4, 2);
  cudaStream_t s;
  CK(cudaStreamCreate(&s));
  act_t* in = dalloc<act_t>((size_t)B * 640 * 640 * 3);
  int *gn, *gc;
  float4* gb;
  int16_t* gm = nullptr;
  L.next(in, &gn, &gb, &gc, nullptr, s, &gm);
  CK(cudaStreamSynchronize(s));
  std::vector<act_t> h((size_t)B * 640 * 640 * 3);
  CK(cudaMemcpy(h.data(), in, h.size() * sizeof(act_t), cudaMemcpyDeviceToHost));
  std::vector<int> n(B);
  std::vector<float4> bx(B * ac.G);
  std::vector<int> cl(B * ac.G);
  CK(cudaMemcpy(n.data(), gn, 4 * B, cudaMemcpyDeviceToHost));
  CK(cudaMemcpy(bx.data(), gb, 16 * B * ac.G, cudaMemcpyDeviceToHost));
  CK(cudaMemcpy(cl.data(), gc, 4 * B * ac.G, cudaMemcpyDeviceToHost));
  const int M = 640 / ac.mask_ratio;
  std::vector<int16_t> mk(ac.masks ? (size_t)B * M * M : 0);
  if (ac.masks) CK(cudaMemcpy(mk.data(), gm, mk.size() * 2, cudaMemcpyDeviceToHost));
  const unsigned char col[7][3] = {{255, 80, 0}, {160, 160, 160}, {255, 0, 0}, {0, 220, 0}, {0, 120, 255}, {170, 0, 255}, {255, 220, 0}};
  for (int b = 0; b < B; ++b) {
    std::vector<unsigned char> im(640 * 640 * 3);
    for (size_t i = 0; i < im.size(); ++i) im[i] = (unsigned char)(to_f(h[(size_t)b * 640 * 640 * 3 + i]) * 255.f + 0.5f);
    if (ac.masks)
      for (int y = 0; y < 640; ++y)
        for (int x = 0; x < 640; ++x) {
          int v = mk[((size_t)b * M + y / ac.mask_ratio) * M + x / ac.mask_ratio];
          if (v <= 0) continue;
          const unsigned char* k = col[cl[b * ac.G + v - 1]];
          for (int c3 = 0; c3 < 3; ++c3) im[(y * 640 + x) * 3 + c3] = (unsigned char)((im[(y * 640 + x) * 3 + c3] + k[c3]) / 2);
        }
    for (int j = 0; j < n[b]; ++j) {
      float4 q = bx[b * ac.G + j];
      int x1 = std::max(0, (int)q.x), y1 = std::max(0, (int)q.y), x2 = std::min(639, (int)q.z), y2 = std::min(639, (int)q.w);
      for (int t = 0; t < 2; ++t) {
        for (int x = x1; x <= x2; ++x)
          for (int y : {std::min(639, y1 + t), std::max(0, y2 - t)}) std::copy(col[cl[b * ac.G + j]], col[cl[b * ac.G + j]] + 3, &im[(y * 640 + x) * 3]);
        for (int y = y1; y <= y2; ++y)
          for (int x : {std::min(639, x1 + t), std::max(0, x2 - t)}) std::copy(col[cl[b * ac.G + j]], col[cl[b * ac.G + j]] + 3, &im[(y * 640 + x) * 3]);
      }
    }
    std::FILE* f = std::fopen((out + "/aug" + std::to_string(b) + ".ppm").c_str(), "wb");
    std::fprintf(f, "P6\n640 640\n255\n");
    std::fwrite(im.data(), 1, im.size(), f);
    std::fclose(f);
  }
  std::printf("배치 %d 장, 정답 수:", B);
  for (int b = 0; b < B; ++b) std::printf(" %d", n[b]);
  std::printf("\n");
  return 0;
}
