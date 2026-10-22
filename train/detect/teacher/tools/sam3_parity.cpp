// 정합성 시험: PyTorch 기준값(ref/sam3_dump.py 가 쓴 safetensors)과 단계별로 비교한다.
// 사용: sam3_parity <model.safetensors> <ref.safetensors> [bf16]
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "prep.h"
#include "sam3.h"
#include "tok.h"

using namespace sam3;
static std::string g_dir;

static std::vector<float> d2h(const float* p, size_t n) {
  std::vector<float> h(n);
  CK(cudaMemcpy(h.data(), p, n * 4, cudaMemcpyDeviceToHost));
  return h;
}
// NCHW → NHWC
static std::vector<float> nchw_to_nhwc(const float* x, int C, int H, int W) {
  std::vector<float> y((size_t)C * H * W);
  for (int c = 0; c < C; ++c)
    for (int i = 0; i < H * W; ++i) y[(size_t)i * C + c] = x[(size_t)c * H * W + i];
  return y;
}
static void cmp(const char* name, const float* got, const float* ref, size_t n) {
  double maxd = 0, maxr = 0, sd = 0;
  size_t at = 0;
  for (size_t i = 0; i < n; ++i) {
    double d = std::fabs((double)got[i] - ref[i]);
    if (!(d <= maxd)) {
      maxd = d;
      at = i;
    }
    maxr = std::max(maxr, std::fabs((double)ref[i]));
    sd += d * d;
  }
  std::printf("%-16s 최대차 %.3e (위치 %zu: %.6f vs %.6f)  최대값 %.3e  상대 %.2e  RMS %.2e\n", name, maxd, at,
              got[at], ref[at], maxr, maxd / std::max(maxr, 1e-30), std::sqrt(sd / n));
}

template <typename T>
static int run(const SafeTensors& W, const SafeTensors& R) {
  cudaStream_t s;
  CK(cudaStreamCreate(&s));
  const auto& ids = R.get("input_ids");
  const auto& am = R.get("attention_mask");
  int P = (int)ids.shape[0];
  {  // 토크나이저 비교(기준 프롬프트 7 개)
    const char* pr[7] = {"fire", "smoke", "fire extinguisher", "emergency exit sign", "person", "door", "stairs"};
    ClipTok tok(g_dir + "/vocab.json", g_dir + "/merges.txt");
    int bad = 0;
    for (int b = 0; b < P && b < 7; ++b) {
      int id[TLEN], mk[TLEN];
      tok.encode(pr[b], TLEN, id, mk);
      for (int t = 0; t < TLEN; ++t)
        bad += id[t] != static_cast<const int*>(ids.data)[b * TLEN + t] || mk[t] != static_cast<const int*>(am.data)[b * TLEN + t];
    }
    std::printf("토큰 다른 칸 %d\n", bad);
  }
  Model<T> m(W, P, s);
  m.debug = true;
  cudaEvent_t e0, e1;
  cudaEventCreate(&e0);
  cudaEventCreate(&e1);

  m.encode_text(static_cast<const int*>(ids.data), static_cast<const int*>(am.data), P);
  cmp("text_hidden", d2h(m.text_hidden.p, (size_t)P * TLEN * TD).data(), R.get("text_last_hidden").f32(),
      (size_t)P * TLEN * TD);
  cmp("text_feat", d2h(m.text_feat.p, (size_t)P * TLEN * D).data(), R.get("text_features").f32(),
      (size_t)P * TLEN * D);

  {  // 전처리 비교: 기준 RGB 원본 → 우리 리사이즈·정규화
    const auto& im = R.get("image_u8");
    std::vector<float> mine((size_t)3 * IMG * IMG);
    prepare_sam3(static_cast<const uint8_t*>(im.data), (int)im.shape[1], (int)im.shape[0], mine.data());
    cmp("pixel_values", mine.data(), R.get("pixel_values").f32(), mine.size());
  }
  Buf<float> px;
  px.alloc((size_t)3 * IMG * IMG);
  CK(cudaMemcpy(px.p, R.get("pixel_values").f32(), px.n * 4, cudaMemcpyHostToDevice));
  cudaEventRecord(e0, s);
  m.encode_image(px.p);
  cudaEventRecord(e1, s);
  CK(cudaStreamSynchronize(s));
  float ms;
  cudaEventElapsedTime(&ms, e0, e1);
  std::printf("영상 인코더 %.1f ms\n", ms);
  const char* dn[3] = {"vit_layer0", "vit_layer6", "vit_layer7"};
  for (int j = 0; j < 3; ++j) cmp(dn[j], m.vit_dbg[j].data(), R.get(dn[j]).f32(), (size_t)NTOK * VD);
  cmp("vit_out", d2h(m.vit.p, (size_t)NTOK * VD).data(), R.get("vit_out").f32(), (size_t)NTOK * VD);
  cmp("fpn_pos2", d2h(m.pos2.p, (size_t)NTOK * D).data(), nchw_to_nhwc(R.get("fpn_pos2").f32(), D, GRID, GRID).data(),
      (size_t)NTOK * D);
  cmp("fpn2", d2h(m.fpn2.p, (size_t)NTOK * D).data(), nchw_to_nhwc(R.get("fpn2").f32(), D, GRID, GRID).data(),
      (size_t)NTOK * D);
  cmp("fpn1", d2h(m.fpn1.p, (size_t)L1 * L1 * D).data(), nchw_to_nhwc(R.get("fpn1").f32(), D, L1, L1).data(),
      (size_t)L1 * L1 * D);
  cmp("fpn0", d2h(m.fpn0.p, (size_t)L0 * L0 * D).data(), nchw_to_nhwc(R.get("fpn0").f32(), D, L0, L0).data(),
      (size_t)L0 * L0 * D);

  cudaEventRecord(e0, s);
  m.detect();
  cudaEventRecord(e1, s);
  CK(cudaStreamSynchronize(s));
  cudaEventElapsedTime(&ms, e0, e1);
  std::printf("검출(프롬프트 %d) %.1f ms\n", P, ms);
  cmp("enc_out", d2h(m.enc.p, (size_t)P * NTOK * D).data(), R.get("enc_out").f32(), (size_t)P * NTOK * D);
  cmp("pred_logits", d2h(m.logits.p, (size_t)P * NQ).data(), R.get("pred_logits").f32(), (size_t)P * NQ);
  cmp("pred_boxes", d2h(m.boxes.p, (size_t)P * NQ * 4).data(), R.get("pred_boxes").f32(), (size_t)P * NQ * 4);
  cmp("presence", d2h(m.presence.p, P).data(), R.get("presence_logits").f32(), P);

  cudaEventRecord(e0, s);
  m.masks_all();
  cudaEventRecord(e1, s);
  CK(cudaStreamSynchronize(s));
  cudaEventElapsedTime(&ms, e0, e1);
  std::printf("마스크(전부) %.1f ms\n", ms);
  cmp("pred_masks", d2h(m.masks.p, (size_t)P * NQ * L0 * L0).data(), R.get("pred_masks").f32(),
      (size_t)P * NQ * L0 * L0);

  // 최종 판정 비교: 점수 = sigmoid(logit) * sigmoid(presence) > 0.4 인 질의
  auto lg = d2h(m.logits.p, (size_t)P * NQ), pr = d2h(m.presence.p, P);
  const float* rl = R.get("pred_logits").f32();
  const float* rp = R.get("presence_logits").f32();
  auto sg = [](float x) { return 1.f / (1.f + std::exp(-x)); };
  int same = 0, diff = 0;
  for (int b = 0; b < P; ++b)
    for (int q = 0; q < NQ; ++q) {
      bool a = sg(lg[b * NQ + q]) * sg(pr[b]) > 0.4f, c = sg(rl[b * NQ + q]) * sg(rp[b]) > 0.4f;
      (a == c ? same : diff)++;
      if (a || c)
        std::printf("  프롬프트 %d 질의 %3d: %.4f vs %.4f\n", b, q, sg(lg[b * NQ + q]) * sg(pr[b]),
                    sg(rl[b * NQ + q]) * sg(rp[b]));
    }
  std::printf("판정(> 0.4) 같음 %d, 다름 %d\n", same, diff);
  return 0;
}

int main(int argc, char** argv) {
  if (argc < 3) {
    std::fprintf(stderr, "사용: %s model.safetensors ref.safetensors [bf16]\n", argv[0]);
    return 1;
  }
  g_dir = std::string(argv[1]).substr(0, std::string(argv[1]).rfind('/'));
  SafeTensors W(argv[1]), R(argv[2]);
  if (argc > 3 && std::string(argv[3]) == "bf16") return run<bf16>(W, R);
  if (argc > 3 && std::string(argv[3]) == "f16") return run<f16>(W, R);
  return run<float>(W, R);
}
