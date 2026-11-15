// head_check — d1_cache 표본으로 CPU 참조(d1_ref.h)의 라벨 로짓이 llama.cpp 가 낸 로짓과 같은지 (가중 풀기·층 계산 확인). 차이는 Q8_0 곱셈(활성도 q8 로 양자화) 몫만 있어야 함
#include <cstdio>
#include <cstring>
#include "d1_ref.h"
int main(int argc, char** argv) {
  if (argc < 3) { std::fprintf(stderr, "head_check model.gguf cache.bin\n"); return 2; }
  d1::Gguf G(argv[1]);
  const d1::Weights W = d1::load(G);
  std::FILE* f = std::fopen(argv[2], "rb");
  double worst = 0;
  int ns = 0, agree = 0;
  for (;;) {
    uint32_t no, nt;
    if (std::fread(&no, 4, 1, f) != 1) break;
    if (std::fread(&nt, 4, 1, f) != 1) break;
    int32_t tg;
    if (std::fread(&tg, 4, 1, f) != 1) break;
    std::vector<uint32_t> grp(no); std::vector<int32_t> tok(nt); std::vector<float> lg(nt), h((size_t)d1::kT * d1::kD);
    if (std::fread(grp.data(), 4, no, f) != no || std::fread(tok.data(), 4, nt, f) != nt || std::fread(lg.data(), 4, nt, f) != nt || std::fread(h.data(), 4, h.size(), f) != h.size()) break;
    const std::vector<float> E = d1::head_rows(G, std::vector<int>(tok.begin(), tok.end()));
    const std::vector<double> me = d1::logits(W, nullptr, h.data(), E, (int)nt);
    double md = 0;
    for (uint32_t k = 0; k < nt; ++k) md = std::max(md, std::fabs(me[k] - lg[k]));
    // 선택지 점수 = 묶음 최댓값 → 고른 것 비교
    int a1 = 0, a2 = 0; size_t i = 0; double b1 = -1e9, b2 = -1e9;
    for (uint32_t o = 0; o < no; ++o) { double m1 = -1e9, m2 = -1e9; for (uint32_t j = 0; j < grp[o]; ++j, ++i) { m1 = std::max(m1, (double)lg[i]); m2 = std::max(m2, me[i]); } if (m1 > b1) { b1 = m1; a1 = o; } if (m2 > b2) { b2 = m2; a2 = o; } }
    std::printf("표본 %d: 선택지 %u · 로짓 최대 차 %.4f (로짓 크기 %.2f) · 고른 것 llama %c / 참조 %c\n", ns, no, md, std::fabs((double)lg[0]), 'A' + a1, 'A' + a2);
    worst = std::max(worst, md); agree += a1 == a2; ++ns;
  }
  std::printf("표본 %d · 로짓 최대 차 %.4f · 고른 것 같음 %d\n", ns, worst, agree);
  return 0;
}
