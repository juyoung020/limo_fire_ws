// 손실 정합성: Ultralytics E2ELoss(ref/loss_dump.py) 와 손실 항목·머리 출력 기울기를 비교(fp32 빌드).
// 사용: loss_parity <기준.safetensors> <배치> <크기>
#include <cmath>
#include <cstdio>
#include <string>

#include "loss.cuh"
#include "st.h"

template <class T>
static T* up(const std::vector<T>& h) {
  T* d = dalloc<T>(h.size());
  CK(cudaMemcpy(d, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice));
  return d;
}
static void cmp(const std::string& n, const float* a, const float* b, size_t k) {
  double md = 0, mr = 0;
  for (size_t i = 0; i < k; ++i) md = std::max(md, std::fabs((double)a[i] - b[i])), mr = std::max(mr, std::fabs((double)b[i]));
  std::printf("%-26s 최대차 %.3e  최대값 %.3e  상대 %.2e\n", n.c_str(), md, mr, md / std::max(mr, 1e-12));
}
int main(int argc, char** argv) {
  SafeTensors R(argv[1]);
  int B = std::stoi(argv[2]), S = std::stoi(argv[3]), nc = 7;
  LossCfg c;
  c.B = B;
  c.imgsz = S;
  c.nc = nc;
  c.G = 64;
  for (int l = 0; l < 3; ++l) c.lh[l] = c.lw[l] = S / c.stride[l];
  E2ELoss loss;
  loss.init(c);
  // 정답
  const auto& gb = R.get("gt_batch");
  int n = (int)gb.numel();
  std::vector<int> cnt(B, 0), cls(B * c.G, 0);
  std::vector<float4> box(B * c.G, make_float4(0, 0, 0, 0));
  for (int i = 0; i < n; ++i) {
    int b = (int)gb.f32()[i], g = cnt[b]++;
    const float* x = R.get("gt_xywhn").f32() + 4 * i;
    box[b * c.G + g] = make_float4((x[0] - x[2] / 2) * S, (x[1] - x[3] / 2) * S, (x[0] + x[2] / 2) * S, (x[1] + x[3] / 2) * S);
    cls[b * c.G + g] = (int)R.get("gt_cls").f32()[i];
  }
  GtDev gt{up(cnt), up(box), up(cls)};
  const char* brn[2] = {"one2many", "one2one"};
  float w[2] = {R.get("o2m_w").f32()[0], R.get("o2o_w").f32()[0]};
  HeadDev h[2];
  for (int br = 0; br < 2; ++br) {
    for (int l = 0; l < 3; ++l) {
      auto tb = R.get(std::string(brn[br]) + ".boxes." + std::to_string(l));
      auto tc = R.get(std::string(brn[br]) + ".scores." + std::to_string(l));
      h[br].box[l] = up(std::vector<float>(tb.f32(), tb.f32() + tb.numel()));
      h[br].cls[l] = up(std::vector<float>(tc.f32(), tc.f32() + tc.numel()));
      h[br].dbox[l] = dalloc<float>(tb.numel());
      h[br].dcls[l] = dalloc<float>(tc.numel());
    }
    loss.run(br, h[br], gt, w[br], nullptr, 0);
  }
  CK(cudaDeviceSynchronize());
  float it[8];
  CK(cudaMemcpy(it, loss.items(), 32, cudaMemcpyDeviceToHost));
  cmp("o2m 항목(box cls l1)", it, R.get("items_o2m").f32(), 3);
  cmp("o2o 항목(box cls l1)", it + 3, R.get("items_o2o").f32(), 3);
  std::printf("  우리 o2m %.5f %.5f %.5f / o2o %.5f %.5f %.5f, 양성 %g\n", it[0], it[1], it[2], it[3], it[4], it[5], it[6]);
  for (int br = 0; br < 2; ++br)
    for (int l = 0; l < 3; ++l)
      for (auto [p, k] : {std::pair{(const float*)h[br].dbox[l], "boxes"}, {(const float*)h[br].dcls[l], "scores"}}) {
        std::string nm = std::string("grad.") + brn[br] + "." + k + "." + std::to_string(l);
        const auto& t = R.get(nm);
        std::vector<float> g(t.numel());
        CK(cudaMemcpy(g.data(), p, g.size() * 4, cudaMemcpyDeviceToHost));
        cmp(nm, g.data(), t.f32(), g.size());
      }
  return 0;
}
