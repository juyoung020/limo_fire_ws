// YOLO26 학습(C++/CUDA). Ultralytics `yolo detect train model=yolo26s.pt` 기본값과 같은 규칙.
//
// 사용: train --train <train.ycache> --val <val.ycache> --weights <yolo26s.safetensors> --out <폴더>
//             [--epochs 100] [--batch 32] [--imgsz 640] [--workers 16] [--close-mosaic 10] [--seed 0]
//             [--max-iters N(속도 재기)] [--resume]
//       train --eval --val <split.ycache> --weights <best.safetensors>   (평가만: 클래스별 AP·정답 수)
// 결과: <out>/last.safetensors·best.safetensors(EMA 가중치, PyTorch 이름), <out>/state.bin(이어 하기), <out>/log.tsv
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <random>
#include <string>

#include "data.h"
#include "evalmap.h"
#include "kern.cuh"
#include "loss.cuh"
#include "optim.cuh"

namespace fs = std::filesystem;
static const char* kNames[7] = {"fire", "smoke", "fire_extinguisher", "exit_sign", "person", "door", "stairs"};
static const int kCocoPerson = 0, kOurPerson = 4;

struct Args {
  std::string train, val, weights, out = "runs/train";
  int epochs = 100, batch = 32, imgsz = 640, workers = 16, close_mosaic = 10, max_iters = -1, val_every = 1, stop_epochs = -1;
  uint64_t seed = 0;
  bool resume = false, eval = false;
};
static Args parse(int argc, char** argv) {
  Args a;
  for (int i = 1; i < argc; ++i) {
    std::string k = argv[i];
    auto v = [&]() { return std::string(argv[++i]); };
    if (k == "--train") a.train = v();
    else if (k == "--val") a.val = v();
    else if (k == "--weights") a.weights = v();
    else if (k == "--out") a.out = v();
    else if (k == "--epochs") a.epochs = std::stoi(v());
    else if (k == "--batch") a.batch = std::stoi(v());
    else if (k == "--imgsz") a.imgsz = std::stoi(v());
    else if (k == "--workers") a.workers = std::stoi(v());
    else if (k == "--close-mosaic") a.close_mosaic = std::stoi(v());
    else if (k == "--seed") a.seed = std::stoull(v());
    else if (k == "--max-iters") a.max_iters = std::stoi(v());
    else if (k == "--val-every") a.val_every = std::stoi(v());
    else if (k == "--resume") a.resume = true;
    else if (k == "--eval") a.eval = true;
    else if (k == "--stop-epochs") a.stop_epochs = std::stoi(v());
    else {
      std::fprintf(stderr, "모르는 인자 %s\n", k.c_str());
      std::exit(1);
    }
  }
  return a;
}

// PyTorch 기본 초기화 + Detect.bias_init, 그다음 사전학습 가중치(모양 같은 것), 그다음 cls_remap(person 행)
static void init_params(Net& net, const std::string& weights, int nc, uint64_t seed) {
  auto& P = net.params;
  std::vector<float> h(P.total, 0.f);
  std::mt19937_64 rng(seed + 12345);
  std::uniform_real_distribution<float> U(-1.f, 1.f);
  std::map<std::string, int64_t> fan;
  for (auto& p : P.list)
    if (p.kind == P_CONV) {
      int64_t f = p.n / p.shape[0];
      fan[p.name.substr(0, p.name.size() - 7)] = f;  // ".weight"
      float b = 1.f / std::sqrt((float)f);
      for (int64_t i = 0; i < p.n; ++i) h[p.off + i] = U(rng) * b;
    }
  for (auto& p : P.list) {
    if (p.kind == P_BN_W || p.kind == P_RUN_VAR)
      std::fill(h.begin() + p.off, h.begin() + p.off + p.n, 1.f);
    if (p.kind == P_BIAS) {
      std::string pre = p.name.substr(0, p.name.size() - 5);  // ".bias"
      float b = 1.f / std::sqrt((float)fan[pre]);
      for (int64_t i = 0; i < p.n; ++i) h[p.off + i] = U(rng) * b;
      // Detect.bias_init: box 2.0, cls log(5/nc/(640/stride)²)
      for (int l = 0; l < 3; ++l)
        for (std::string br : {"cv2.", "one2one_cv2."})
          if (p.name == "model.23." + br + std::to_string(l) + ".2.bias") std::fill(h.begin() + p.off, h.begin() + p.off + p.n, 2.f);
      for (int l = 0; l < 3; ++l)
        for (std::string br : {"cv3.", "one2one_cv3."})
          if (p.name == "model.23." + br + std::to_string(l) + ".2.bias") {
            float v = std::log(5.f / nc / std::pow(640.f / net.strides[l], 2.f));
            std::fill(h.begin() + p.off, h.begin() + p.off + p.n, v);
          }
    }
  }
  CK(cudaMemcpy(P.vals, h.data(), P.total * 4, cudaMemcpyHostToDevice));
  if (weights.empty()) return;
  SafeTensors W(weights);
  auto sk = P.load(W);
  std::printf("사전학습 가중치: %zu 개 중 %zu 개 불러옴(모양 다른 분류층 %zu 개는 초기값)\n", P.list.size(),
              P.list.size() - sk.size(), sk.size());
  // cls_remap: 이름이 같은 클래스(person) 행만 COCO 에서 복사
  CK(cudaMemcpy(h.data(), P.vals, P.total * 4, cudaMemcpyDeviceToHost));
  int remapped = 0;
  for (int l = 0; l < 3; ++l)
    for (std::string br : {"cv3.", "one2one_cv3."}) {
      std::string pre = "model.23." + br + std::to_string(l) + ".2.";
      for (std::string t : {"weight", "bias"}) {
        if (!W.has(pre + t)) continue;
        const auto& src = W.get(pre + t);
        const auto& p = P.p(pre + t);
        int64_t row = p.n / p.shape[0];
        if (src.numel() / src.shape[0] != row || src.shape[0] == p.shape[0]) continue;  // 우리 학습 결과면 그대로
        std::memcpy(&h[p.off + kOurPerson * row], src.f32() + kCocoPerson * row, row * 4);
        ++remapped;
      }
    }
  CK(cudaMemcpy(P.vals, h.data(), P.total * 4, cudaMemcpyHostToDevice));
  std::printf("cls_remap: person 행 %d 개 텐서\n", remapped);
}

struct Ckpt {  // 이어 하기 상태
  int epoch;
  double best;
};
static void save_state(const std::string& path, Net& net, Optimizer& opt, const Ckpt& c) {
  auto& P = net.params;
  std::vector<float> h(P.total);
  std::FILE* f = std::fopen((path + ".tmp").c_str(), "wb");
  std::fwrite("YTSTATE1", 1, 8, f);
  std::fwrite(&c, sizeof(c), 1, f);
  int64_t n = P.total;
  std::fwrite(&n, 8, 1, f);
  CK(cudaMemcpy(h.data(), P.vals, P.total * 4, cudaMemcpyDeviceToHost));
  std::fwrite(h.data(), 4, P.total, f);
  opt.save(f);
  std::fclose(f);
  fs::rename(path + ".tmp", path);
}
static bool load_state(const std::string& path, Net& net, Optimizer& opt, Ckpt& c) {
  std::FILE* f = std::fopen(path.c_str(), "rb");
  if (!f) return false;
  char m[8];
  int64_t n;
  YT_ASSERT(std::fread(m, 1, 8, f) == 8 && std::memcmp(m, "YTSTATE1", 8) == 0, "상태 파일 형식");
  YT_ASSERT(std::fread(&c, sizeof(c), 1, f) == 1 && std::fread(&n, 8, 1, f) == 1 && n == net.params.total, "상태 파일 크기");
  std::vector<float> h(n);
  YT_ASSERT(std::fread(h.data(), 4, n, f) == (size_t)n, "상태 파일이 짧음");
  CK(cudaMemcpy(net.params.vals, h.data(), n * 4, cudaMemcpyHostToDevice));
  opt.load(f);
  std::fclose(f);
  return true;
}

// EMA 가중치로 검증
static MapEval::Result validate(Net& net, Optimizer& opt, Loader& vl, int64_t nbv, const Cache& vc, int nc,
                                cudaStream_t s) {
  auto& P = net.params;
  static float* saved = nullptr;
  if (!saved) saved = dalloc<float>(P.total);
  CK(cudaMemcpyAsync(saved, P.vals, P.total * 4, cudaMemcpyDeviceToDevice, s));
  CK(cudaMemcpyAsync(P.vals, opt.ema(), P.total * 4, cudaMemcpyDeviceToDevice, s));
  P.sync_w16(s);
  MapEval ev(nc);
  int B = net.cfg.batch;
  std::vector<int> idx(B);
  int hh[3], ww[3];
  std::vector<std::vector<act_t>> hb(3), hc(3);
  std::vector<std::vector<float>> fb(3), fc(3);
  for (int l = 0; l < 3; ++l) {
    hh[l] = net.head_box(1, l)->H;
    ww[l] = net.head_box(1, l)->W;
    hb[l].resize(net.head_box(1, l)->n());
    hc[l].resize(net.head_cls(1, l)->n());
  }
  for (int64_t k = 0; k < nbv; ++k) {
    int *gn;
    float4* gbx;
    int* gcl;
    vl.next(net.input()->x, &gn, &gbx, &gcl, idx.data(), s);
    net.forward(false);
    for (int l = 0; l < 3; ++l) {
      CK(cudaMemcpyAsync(hb[l].data(), net.head_box(1, l)->x, hb[l].size() * sizeof(act_t), cudaMemcpyDeviceToHost, s));
      CK(cudaMemcpyAsync(hc[l].data(), net.head_cls(1, l)->x, hc[l].size() * sizeof(act_t), cudaMemcpyDeviceToHost, s));
    }
    CK(cudaStreamSynchronize(s));
    for (int l = 0; l < 3; ++l) {
      fb[l].resize(hb[l].size());
      fc[l].resize(hc[l].size());
      for (size_t i = 0; i < hb[l].size(); ++i) fb[l][i] = to_f(hb[l][i]);
      for (size_t i = 0; i < hc[l].size(); ++i) fc[l][i] = to_f(hc[l][i]);
    }
    for (int b = 0; b < B; ++b) {
      if (idx[b] < 0) continue;
      const float* bx[3];
      const float* cx[3];
      for (int l = 0; l < 3; ++l) {
        bx[l] = fb[l].data() + (size_t)b * hh[l] * ww[l] * 4;
        cx[l] = fc[l].data() + (size_t)b * hh[l] * ww[l] * nc;
      }
      auto det = decode_o2o(bx, cx, hh, ww, net.strides, nc);
      // 정답: 캐시 라벨을 검증 letterbox 좌표로
      const CacheRec& r = vc.rec(idx[b]);
      int S = net.cfg.imgsz;
      float left = (float)std::lround((S - r.w) / 2.0 - 0.1), top = (float)std::lround((S - r.h) / 2.0 - 0.1);
      std::vector<GtBox> gt;
      const CacheBox* cb = vc.boxes(idx[b]);
      for (int j = 0; j < r.nbox; ++j) {
        float cx_ = cb[j].cx * r.w + left, cy_ = cb[j].cy * r.h + top, w = cb[j].w * r.w, h = cb[j].h * r.h;
        gt.push_back({cx_ - w / 2, cy_ - h / 2, cx_ + w / 2, cy_ + h / 2, (int)cb[j].cls});
      }
      ev.add(det, gt);
    }
  }
  CK(cudaMemcpyAsync(P.vals, saved, P.total * 4, cudaMemcpyDeviceToDevice, s));
  P.sync_w16(s);
  CK(cudaStreamSynchronize(s));
  return ev.compute();
}

int main(int argc, char** argv) {
  Args a = parse(argc, argv);
  if (!a.eval) fs::create_directories(a.out);
  const int nc = 7;
  cudaStream_t s;
  CK(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking));
  cudnnHandle_t dnn;
  CDN(cudnnCreate(&dnn));
  CDN(cudnnSetStream(dnn, s));
  cublasHandle_t blas;
  CBL(cublasCreate(&blas));
  CBL(cublasSetStream(blas, s));
  {  // 그래프 잡기 중에 cuBLAS 가 메모리를 잡지 않도록
    void* bws = dalloc<char>(64u << 20);
    CBL(cublasSetWorkspace(blas, bws, 64u << 20));
  }

  Cache tc, vc;
  if (a.eval) a.train = a.val;  // 평가만: 학습 캐시는 쓰지 않음
  YT_ASSERT(tc.open(a.train), "학습 캐시 %s", a.train.c_str());
  YT_ASSERT(vc.open(a.val), "검증 캐시 %s", a.val.c_str());
  std::printf("학습 %d 장, 검증 %d 장\n", tc.n(), vc.n());

  Net net;
  NetCfg ncf;
  ncf.batch = a.batch;
  ncf.imgsz = a.imgsz;
  ncf.nc = nc;
  net.build(ncf, dnn, blas, s);
  init_params(net, a.weights, nc, a.seed);
  net.params.sync_w16(s);

  AugCfg ac;
  ac.S = a.imgsz;
  ac.batch = a.batch;
  ac.epochs = a.epochs;
  ac.close_mosaic = a.close_mosaic;
  ac.seed = a.seed;
  Loader tl;
  tl.open(&tc, ac, a.workers, 8);
  AugCfg vcfg = ac;
  vcfg.augment = false;
  Loader vl;
  vl.open(&vc, vcfg, std::max(4, a.workers / 2), 4);
  int64_t nb = tl.batches_per_epoch(), nbv = vl.batches_per_epoch();

  LossCfg lc;
  lc.B = a.batch;
  lc.nc = nc;
  lc.imgsz = a.imgsz;
  lc.G = ac.G;
  for (int l = 0; l < 3; ++l) lc.lh[l] = lc.lw[l] = a.imgsz / lc.stride[l];
  E2ELoss loss;
  loss.init(lc);

  OptCfg oc;
  oc.epochs = a.epochs;
  oc.batch = a.batch;
  oc.nb = nb;
  Optimizer opt;
  opt.init(oc, &net, blas, s);
  std::printf("장치 메모리 %.2f GB, epoch 당 %lld 반복, 누적 %d\n", dev_total() / 1e9, (long long)nb, opt.accumulate(nb * 10));

  if (a.eval) {  // 불러온 가중치를 그대로(EMA 자리에 두고) 검증
    CK(cudaMemcpy(opt.ema(), net.params.vals, net.params.total * 4, cudaMemcpyDeviceToDevice));
    MapEval::Result r = validate(net, opt, vl, nbv, vc, nc, s);
    std::printf("%-18s %8s %8s %8s\n", "클래스", "정답 수", "AP50", "AP50-95");
    for (int c = 0; c < nc; ++c) std::printf("%-18s %8d %8.4f %8.4f\n", kNames[c], (int)r.ngt[c], r.ap50[c], r.ap[c]);
    std::printf("%-18s %8s %8.4f %8.4f\n", "전체", "", r.map50, r.map);
    return 0;
  }
  Ckpt ck{0, -1.0};
  if (a.resume) {
    YT_ASSERT(load_state(a.out + "/state.bin", net, opt, ck), "이어 할 상태가 없음: %s/state.bin", a.out.c_str());
    net.params.sync_w16(s);
    tl.skip(ck.epoch * nb);
    std::printf("이어 하기: epoch %d 부터, 최고 mAP50-95 %.4f\n", ck.epoch + 1, ck.best);
  }
  std::FILE* log = std::fopen((a.out + "/log.tsv").c_str(), a.resume ? "a" : "w");
  if (!a.resume) std::fprintf(log, "epoch\tbox\tcls\tl1\tmAP50\tmAP50-95\tsec\n");
  HeadDev hd[2];
  for (int br = 0; br < 2; ++br)
    for (int l = 0; l < 3; ++l) {
      hd[br].box[l] = net.head_box(br, l)->x;
      hd[br].cls[l] = net.head_cls(br, l)->x;
      hd[br].dbox[l] = net.head_box(br, l)->g;
      hd[br].dcls[l] = net.head_cls(br, l)->g;
    }
  float* lw = dalloc<float>(2);  // 가지 가중치(o2m, o2o) — 그래프가 장치에서 읽는다
  float* lw_h = nullptr;
  CK(cudaMallocHost(&lw_h, 8));
  // 반복 하나(기울기 지우기 → 정방향 → 손실 → 역방향 → 가중치 기울기 모으기)를 CUDA 그래프로. 정답 버퍼가 두 벌이라
  // 그래프도 두 개(정답 주소마다).
  std::map<const void*, cudaGraphExec_t> graphs;
  bool use_graph = true;
  auto iteration = [&](const GtDev& gt) {
    net.zero_act_grads();
    net.forward(true);
    loss.run(0, hd[0], gt, lw, opt.scale(), s);
    loss.run(1, hd[1], gt, lw + 1, opt.scale(), s);
    net.backward();
    net.params.gather_dw(s, true);
  };
  int64_t last_opt = int64_t(ck.epoch) * nb - 1;
  CK(cudaMemsetAsync(net.params.grads, 0, net.params.total * 4, s));
  for (int epoch = ck.epoch; epoch < a.epochs; ++epoch) {
    float o2m = (float)(std::max(1.0 - (double)epoch / std::max(a.epochs - 1, 1), 0.0) * (0.8 - 0.1) + 0.1),
          o2o = std::max(1.f - o2m, 0.f);
    CK(cudaStreamSynchronize(s));
    lw_h[0] = o2m;
    lw_h[1] = o2o;
    CK(cudaMemcpyAsync(lw, lw_h, 8, cudaMemcpyHostToDevice, s));
    auto t0 = std::chrono::steady_clock::now();
    double sum[3] = {0, 0, 0};
    int nsum = 0;
    tl.wait_ms = 0;
    for (int64_t k = 0; k < nb; ++k) {
      int64_t ni = k + nb * epoch;
      int* gn;
      float4* gbx;
      int* gcl;
      tl.next(net.input()->x, &gn, &gbx, &gcl, nullptr, s);
      GtDev gt{gn, gbx, gcl};
      if (!use_graph || ni == int64_t(ck.epoch) * nb) {
        iteration(gt);  // 처음 한 번은 그대로(지연 초기화가 끝나도록)
      } else {
        auto it = graphs.find(gn);
        if (it == graphs.end()) {
          cudaGraph_t g;
          CK(cudaStreamBeginCapture(s, cudaStreamCaptureModeThreadLocal));
          iteration(gt);
          CK(cudaStreamEndCapture(s, &g));
          cudaGraphExec_t ge;
          CK(cudaGraphInstantiate(&ge, g, 0));
          CK(cudaGraphDestroy(g));
          it = graphs.emplace(gn, ge).first;
        }
        CK(cudaGraphLaunch(it->second, s));
      }
      if (ni - last_opt >= opt.accumulate(ni)) {
        opt.step(ni, epoch);
        last_opt = ni;
      }
      if (k % 50 == 0 || k == nb - 1) {
        float it[8];
        CK(cudaMemcpyAsync(it, loss.items(), 32, cudaMemcpyDeviceToHost, s));
        CK(cudaStreamSynchronize(s));
        sum[0] += it[3], sum[1] += it[4], sum[2] += it[5];
        ++nsum;
        double sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        OptDev od = opt.read_state();
        std::printf("\r%3d/%d %5lld/%lld  box %.3f cls %.3f l1 %.3f  lr %.5f scale %g  %.1f 장/초 (기다림 %.0f%%)   ",
                    epoch + 1, a.epochs, (long long)k + 1, (long long)nb, it[3], it[4], it[5], opt.lr_now[0], od.scale,
                    (k + 1) * a.batch / sec, 100.0 * tl.wait_ms / 1000.0 / sec);
        std::fflush(stdout);
      }
      if (a.max_iters > 0 && ni + 1 >= a.max_iters) {
        CK(cudaStreamSynchronize(s));
        double sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        std::printf("\n속도 재기 끝: %d 반복 %.1f 초, %.1f 장/초, 데이터 기다림 %.1f%%\n", a.max_iters, sec,
                    a.max_iters * a.batch / sec, 100.0 * tl.wait_ms / 1000.0 / sec);
        return 0;
      }
    }
    double tsec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    std::printf("\n");
    MapEval::Result r;
    if ((epoch + 1) % a.val_every == 0 || epoch + 1 == a.epochs) {
      r = validate(net, opt, vl, nbv, vc, nc, s);
      std::printf("  검증 mAP50 %.4f  mAP50-95 %.4f |", r.map50, r.map);
      for (int c = 0; c < nc; ++c) std::printf(" %s %.3f", kNames[c], r.ap[c]);
      std::printf("\n");
    }
    // EMA 가중치 저장
    {
      auto& P = net.params;
      static float* saved = dalloc<float>(P.total);
      CK(cudaMemcpyAsync(saved, P.vals, P.total * 4, cudaMemcpyDeviceToDevice, s));
      CK(cudaMemcpyAsync(P.vals, opt.ema(), P.total * 4, cudaMemcpyDeviceToDevice, s));
      CK(cudaStreamSynchronize(s));
      P.save(a.out + "/last.safetensors");
      if (r.map > ck.best) {
        ck.best = r.map;
        P.save(a.out + "/best.safetensors");
      }
      CK(cudaMemcpyAsync(P.vals, saved, P.total * 4, cudaMemcpyDeviceToDevice, s));
      CK(cudaStreamSynchronize(s));
    }
    ck.epoch = epoch + 1;
    save_state(a.out + "/state.bin", net, opt, ck);
    std::fprintf(log, "%d\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.0f\n", epoch + 1, sum[0] / nsum, sum[1] / nsum, sum[2] / nsum,
                 r.map50, r.map, tsec);
    std::fflush(log);
    if (a.stop_epochs > 0 && epoch + 1 >= a.stop_epochs) break;
  }
  std::fclose(log);
  return 0;
}
