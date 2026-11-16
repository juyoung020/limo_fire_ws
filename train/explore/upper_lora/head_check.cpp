// head_check — d1_cache 표본으로 CPU 참조(d1_ref.h)의 라벨 로짓이 llama.cpp 가 낸 로짓과 같은지 (가중 풀기·층 계산 확인). 차이는 Q8_0 곱셈(활성도 q8 로 양자화) 몫만 있어야 함
#include <cstdio>
#include <cstring>
#include <string>
#include "d1_attn.h"
int main(int argc, char** argv) {
  // --seq: d1_cache --full-from 27 의 <cache>.seq — 27 층부터 CPU 참조로 다시 계산 (27 층 출력 마지막 5 줄 = 캐시한 28 층 입력과도 견줌)
  std::string seq;
  for (int i = 1; i + 1 < argc; ++i) if (std::string(argv[i]) == "--seq") { seq = argv[i + 1]; for (int j = i; j + 2 < argc; ++j) argv[j] = argv[j + 2]; argc -= 2; break; }
  if (argc < 3) { std::fprintf(stderr, "head_check model.gguf cache.bin [adapter.bin] [--seq cache.bin.seq]\n"); return 2; }
  d1::Gguf G(argv[1]);
  const d1::Weights W = d1::load(G);
  d1::ALora alo;   // 27 층 몫 (있으면)
  d1::Lora lo;   // 선택: lora_train 어댑터 (llama.cpp 가 --lora 로 낸 로짓과 견줌)
  if (argc > 3) {
    std::FILE* fa = std::fopen(argv[3], "rb");
    float alpha;
    if (!fa || std::fread(&lo.r, 4, 1, fa) != 1 || std::fread(&alpha, 4, 1, fa) != 1) { std::fprintf(stderr, "어댑터를 못 읽음\n"); return 1; }
    lo.scale = alpha / lo.r;
    for (int l = 0; l < 2; ++l) for (int m = 0; m < d1::kMats; ++m) {
      int on = 0;
      if (std::fread(&on, 4, 1, fa) != 1) return 1;
      if (!on) continue;
      std::vector<float> a((size_t)lo.r * d1::mat_in(m)), b((size_t)d1::mat_out(m) * lo.r);
      if (std::fread(a.data(), 4, a.size(), fa) != a.size() || std::fread(b.data(), 4, b.size(), fa) != b.size()) return 1;
      lo.A[l][m].assign(a.begin(), a.end()); lo.B[l][m].assign(b.begin(), b.end());
    }
    int tag = 0;   // 27 층 몫 (lora_train --attn)
    if (std::fread(&tag, 4, 1, fa) == 1 && tag == 27) {
      alo.r = lo.r; alo.scale = lo.scale;
      for (int m = 0; m < d1::kAMats; ++m) {
        int on = 0;
        if (std::fread(&on, 4, 1, fa) != 1) return 1;
        if (!on) continue;
        std::vector<float> a((size_t)lo.r * d1::amat_in(m)), b((size_t)d1::amat_out(m) * lo.r);
        if (std::fread(a.data(), 4, a.size(), fa) != a.size() || std::fread(b.data(), 4, b.size(), fa) != b.size()) return 1;
        alo.A[m].assign(a.begin(), a.end()); alo.B[m].assign(b.begin(), b.end());
      }
    }
    std::fclose(fa);
  }
  std::FILE* f = std::fopen(argv[2], "rb");
  std::FILE* fs = seq.empty() ? nullptr : std::fopen(seq.c_str(), "rb");
  d1::ALayer A27;
  if (fs) A27 = d1::load27(G);
  double worst27 = 0;
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
    std::vector<double> me;
    if (fs) {
      uint32_t T;
      if (std::fread(&T, 4, 1, fs) != 1) break;
      std::vector<ggml_fp16_t> x16((size_t)T * d1::kD);
      if (std::fread(x16.data(), 2, x16.size(), fs) != x16.size()) break;
      std::vector<float> X(x16.size());
      for (size_t k = 0; k < X.size(); ++k) X[k] = ggml_fp16_to_fp32(x16[k]);
      const std::vector<double> h5 = d1::layer27(A27, W.eps, alo.r > 0 ? &alo : nullptr, X.data(), (int)T);
      double d27 = 0, mag = 0;
      for (size_t k = 0; k < h5.size(); ++k) { d27 = std::max(d27, std::fabs(h5[k] - h[k])); mag = std::max(mag, std::fabs((double)h[k])); }
      std::printf("  27 층 출력 마지막 5 줄 vs 캐시한 28 층 입력: 최대 차 %.4f (값 크기 %.1f) · T %u\n", d27, mag, T);
      worst27 = std::max(worst27, d27);
      me = d1::logits_full(W, A27, lo.r > 0 ? &lo : nullptr, alo.r > 0 ? &alo : nullptr, X.data(), (int)T, E, (int)nt);
    } else {
      me = d1::logits(W, lo.r > 0 ? &lo : nullptr, h.data(), E, (int)nt);
    }
    double md = 0;
    for (uint32_t k = 0; k < nt; ++k) md = std::max(md, std::fabs(me[k] - lg[k]));
    // 선택지 점수 = 묶음 최댓값 → 고른 것 비교
    int a1 = 0, a2 = 0; size_t i = 0; double b1 = -1e9, b2 = -1e9;
    for (uint32_t o = 0; o < no; ++o) { double m1 = -1e9, m2 = -1e9; for (uint32_t j = 0; j < grp[o]; ++j, ++i) { m1 = std::max(m1, (double)lg[i]); m2 = std::max(m2, me[i]); } if (m1 > b1) { b1 = m1; a1 = o; } if (m2 > b2) { b2 = m2; a2 = o; } }
    std::printf("표본 %d: 선택지 %u · 로짓 최대 차 %.4f (로짓 크기 %.2f) · 고른 것 llama %c / 참조 %c\n", ns, no, md, std::fabs((double)lg[0]), 'A' + a1, 'A' + a2);
    worst = std::max(worst, md); agree += a1 == a2; ++ns;
  }
  std::printf("표본 %d · 로짓 최대 차 %.4f · 고른 것 같음 %d%s\n", ns, worst, agree, fs ? " (27 층부터 다시 계산)" : "");
  if (fs) std::printf("27 층 출력 최대 차 %.4f\n", worst27);
  return 0;
}
