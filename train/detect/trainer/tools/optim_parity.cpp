// MuSGD 정합성: Ultralytics(ref/optim_dump.py) 의 두 스텝 뒤 파라미터와 비교(fp32 빌드).
#include <cmath>
#include <cstdio>

#include "optim.cuh"

int main(int argc, char** argv) {
  SafeTensors R(argv[1]);
  cudaStream_t s;
  CK(cudaStreamCreate(&s));
  cudnnHandle_t dnn;
  CDN(cudnnCreate(&dnn));
  CDN(cudnnSetStream(dnn, s));
  cublasHandle_t blas;
  CBL(cublasCreate(&blas));
  CBL(cublasSetStream(blas, s));
  Net net;
  NetCfg nc;
  nc.batch = 1;
  nc.imgsz = 64;
  nc.nc = 7;
  net.build(nc, dnn, blas, s);
  auto sk = net.params.load(R, "p0.");
  int nrun = 0;
  for (auto& k : sk) nrun += k.find("running") != std::string::npos;
  std::printf("못 불러온 것 %zu (running 통계 %d)\n", sk.size(), nrun);
  OptCfg oc;
  oc.loss_scaling = false;
  oc.warmup_epochs = 0;
  oc.epochs = 1000000;
  oc.batch = 64;
  oc.nbs = 64;
  oc.clip = 1e30f;
  Optimizer opt;
  opt.init(oc, &net, blas, s);
  auto& P = net.params;
  for (int st = 0; st < 2; ++st) {
    std::vector<float> g(P.total, 0.f);
    for (auto& p : P.list) {
      std::string k = "g" + std::to_string(st) + "." + p.name;
      if (R.has(k)) std::copy(R.get(k).f32(), R.get(k).f32() + p.n, g.begin() + p.off);
    }
    CK(cudaMemcpy(P.grads, g.data(), g.size() * 4, cudaMemcpyHostToDevice));
    opt.step(1, 0);
  }
  CK(cudaStreamSynchronize(s));
  std::vector<float> v(P.total);
  CK(cudaMemcpy(v.data(), P.vals, v.size() * 4, cudaMemcpyDeviceToHost));
  double worst = 0, worst_nc = 0, sum = 0;
  int cnt = 0;
  std::string wn;
  for (auto& p : P.list) {
    std::string k = "p2." + p.name;
    if (!R.has(k)) continue;
    const float* r = R.get(k).f32();
    const float* r0 = R.get("p0." + p.name).f32();
    double md = 0, mu = 0;
    for (int64_t i = 0; i < p.n; ++i) {
      md = std::max(md, std::fabs((double)v[p.off + i] - r[i]));
      mu = std::max(mu, std::fabs((double)r[i] - r0[i]));  // 갱신 크기
    }
    double rel = md / std::max(mu, 1e-12);
    if (p.kind == P_CONV) sum += rel, ++cnt;
    else worst_nc = std::max(worst_nc, rel);
    if (rel > worst) worst = rel, wn = p.name;
  }
  std::printf("합성곱(Muon) 평균 %.3e, BN·편향(SGD) 최악 %.3e\n", sum / cnt, worst_nc);
  std::printf("갱신량 대비 최악 상대 차 %.3e (%s)\n", worst, wn.c_str());
  return 0;
}
