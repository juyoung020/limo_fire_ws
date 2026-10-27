// 신경망 정합성: Ultralytics(ref/net_dump.py) 의 정방향 출력·BN running 통계·파라미터 기울기와 비교(fp32 빌드).
// 사용: net_parity <가중치.safetensors> <기준.safetensors> <배치> <크기> [seg]
//       seg: YOLO26s-seg(ref/seg_net_dump.py) — 마스크 계수·원형 마스크·의미 분할 출력과 그 기울기까지
#include <cmath>
#include <cstdio>
#include <string>

#include "kern.cuh"
#include "net.h"

static std::vector<float> d2h(const Tensor* x) {  // 행 간격(ld) 고려
  size_t n = x->n();
  std::vector<act_t> t(n);
  CK(cudaMemcpy2D(t.data(), x->C * sizeof(act_t), x->x, x->ld * sizeof(act_t), x->C * sizeof(act_t), x->M(),
                  cudaMemcpyDeviceToHost));
  std::vector<float> h(n);
  for (size_t i = 0; i < n; ++i) h[i] = to_f(t[i]);
  return h;
}
static double worst = 0;
static void cmp(const std::string& name, const float* got, const float* ref, size_t n, bool quiet = false) {
  double md = 0, mr = 0;
  for (size_t i = 0; i < n; ++i) {
    md = std::max(md, std::fabs((double)got[i] - ref[i]));
    mr = std::max(mr, std::fabs((double)ref[i]));
  }
  double rel = md / std::max(mr, 1e-12);
  worst = std::max(worst, rel);
  if (!quiet || rel > 1e-3) std::printf("%-44s 최대차 %.3e  최대값 %.3e  상대 %.2e\n", name.c_str(), md, mr, rel);
}

int main(int argc, char** argv) {
  SafeTensors W(argv[1]), R(argv[2]);
  int B = std::stoi(argv[3]), S = std::stoi(argv[4]);
  bool seg = argc > 5 && std::string(argv[5]) == "seg";
  cudaStream_t s;
  CK(cudaStreamCreate(&s));
  cudnnHandle_t dnn;
  CDN(cudnnCreate(&dnn));
  CDN(cudnnSetStream(dnn, s));
  cublasHandle_t blas;
  CBL(cublasCreate(&blas));
  CBL(cublasSetStream(blas, s));
  CBL(cublasSetMathMode(blas, CUBLAS_PEDANTIC_MATH));
  Net net;
  NetCfg cfg;
  cfg.batch = B;
  cfg.imgsz = S;
  cfg.nc = 80;
  cfg.seg = seg;
  net.build(cfg, dnn, blas, s);
  auto skipped = net.params.load(W);
  std::printf("파라미터 %zu 개, 못 불러온 것 %zu 개, 활성 메모리 %.2f GB\n", net.params.list.size(), skipped.size(),
              dev_total() / 1e9);
  for (auto& k : skipped) std::printf("  %s\n", k.c_str());
  net.params.sync_w16(s);
  {  // 기준 입력 3 채널 → 8 채널(0 덧대기)
    const float* x = R.get("input").f32();
    std::vector<float> h(net.input()->n(), 0.f);
    for (int64_t i = 0; i < net.input()->M(); ++i)
      for (int c = 0; c < 3; ++c) h[i * 8 + c] = x[i * 3 + c];
    CK(cudaMemcpy(net.input()->x, h.data(), h.size() * 4, cudaMemcpyHostToDevice));
  }
  net.forward(true);
  CK(cudaStreamSynchronize(s));
  for (int i : {2, 4, 6, 9, 10, 13, 16, 19, 22}) {
    Tensor* t = net.named.at("x" + std::to_string(i));
    cmp("x" + std::to_string(i), d2h(t).data(), R.get("out.x" + std::to_string(i)).f32(), t->n());
  }
  for (int br = 0; br < 2; ++br)
    for (int l = 0; l < 3; ++l) {
      std::string k = std::to_string(br) + std::to_string(l);
      Tensor* b = net.head_box(br, l);
      Tensor* c = net.head_cls(br, l);
      cmp("box" + k, d2h(b).data(), R.get("out.box" + k).f32(), b->n());
      cmp("cls" + k, d2h(c).data(), R.get("out.cls" + k).f32(), c->n());
      if (seg) cmp("msk" + k, d2h(net.head_mask(br, l)).data(), R.get("out.msk" + k).f32(), net.head_mask(br, l)->n());
    }
  if (seg) {
    cmp("fuse", d2h(net.named.at("model.23.proto.feat_fuse")).data(), R.get("out.fuse").f32(),
        net.named.at("model.23.proto.feat_fuse")->n());
    cmp("proto", d2h(net.proto()).data(), R.get("out.proto").f32(), net.proto()->n());
    cmp("sem", d2h(net.semantic()).data(), R.get("out.sem").f32(), net.semantic()->n());
  }
  // BN running 통계(정방향 한 번 뒤)
  std::vector<float> vals(net.params.total);
  CK(cudaMemcpy(vals.data(), net.params.vals, vals.size() * 4, cudaMemcpyDeviceToHost));
  worst = 0;
  for (auto& p : net.params.list)
    if (p.kind == P_RUN_MEAN || p.kind == P_RUN_VAR)
      cmp("after." + p.name, &vals[p.off], R.get("after." + p.name).f32(), p.n, true);
  std::printf("running 통계 최악 상대 %.2e\n", worst);
  // 역방향: 출력 기울기 = 기준 무작위 가중치
  net.zero_act_grads();
  auto put = [&](Tensor* t, const std::string& nm) {
    const float* g = R.get("gout." + nm).f32();
    std::vector<act_t> h(t->n());
    for (size_t i = 0; i < h.size(); ++i) h[i] = from_f<act_t>(g[i]);
    CK(cudaMemcpy(t->g, h.data(), h.size() * sizeof(act_t), cudaMemcpyHostToDevice));
  };
  for (int br = 0; br < 2; ++br)
    for (int l = 0; l < 3; ++l) {
      std::string k = std::to_string(br) + std::to_string(l);
      put(net.head_box(br, l), "box" + k);
      put(net.head_cls(br, l), "cls" + k);
      if (seg) put(net.head_mask(br, l), "msk" + k);
    }
  if (seg) {
    put(net.proto(), "proto");
    put(net.semantic(), "sem");
  }
  CK(cudaMemset(net.params.grads, 0, net.params.total * 4));
  net.backward();
  net.params.gather_dw(s, true);
  CK(cudaStreamSynchronize(s));
  std::vector<float> gr(net.params.total);
  CK(cudaMemcpy(gr.data(), net.params.grads, gr.size() * 4, cudaMemcpyDeviceToHost));
  worst = 0;
  for (auto& p : net.params.list)
    if (p.kind != P_RUN_MEAN && p.kind != P_RUN_VAR) cmp("grad." + p.name, &gr[p.off], R.get("grad." + p.name).f32(), p.n, true);
  std::printf("파라미터 기울기 최악 상대 %.2e\n", worst);
  return 0;
}
