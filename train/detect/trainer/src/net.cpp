#include "net.h"

#include <algorithm>
#include <cstring>
#include "kern.cuh"

// ================= 파라미터 =================
int Params::add(const std::string& name, PKind k, std::vector<int64_t> shape) {
  YT_ASSERT(!idx.count(name), "파라미터 이름 겹침 %s", name.c_str());
  Param p;
  p.name = name;
  p.kind = k;
  p.shape = shape;
  p.n = 1;
  for (auto s : shape) p.n *= s;
  p.off = total;
  total += p.n;

  idx[name] = (int)list.size();
  list.push_back(p);
  return idx[name];
}
void Params::allocate() {
  std::vector<ConvTab> t;
  wtotal = 0;
  for (auto& p : list)
    if (p.kind == P_CONV) {
      int Cg = (int)p.shape[1], RS = (int)(p.shape[2] * p.shape[3]), Cp = p.cpad ? p.cpad : Cg;
      p.woff = wtotal;
      wtotal += int64_t(p.shape[0]) * RS * Cp;
      t.push_back({p.off, p.woff, (int)p.shape[0], Cg, RS, Cp});
    }
  ntab = (int)t.size();
  tab = dalloc<ConvTab>(ntab);
  CK(cudaMemcpy(tab, t.data(), sizeof(ConvTab) * ntab, cudaMemcpyHostToDevice));
  vals = dalloc<float>(total);
  grads = dalloc<float>(total);
  w16 = dalloc<act_t>(wtotal);
  dw16 = dalloc<act_t>(wtotal);
  CK(cudaMemset(vals, 0, total * 4));
  CK(cudaMemset(grads, 0, total * 4));
}
std::vector<std::string> Params::load(const SafeTensors& st, const std::string& prefix) {
  std::vector<std::string> skipped;
  std::vector<float> h(total, 0.f);
  CK(cudaMemcpy(h.data(), vals, total * 4, cudaMemcpyDeviceToHost));
  for (auto& p : list) {
    std::string n = prefix + p.name;
    if (!st.has(n)) {
      skipped.push_back(p.name + " (없음)");
      continue;
    }
    const auto& t = st.get(n);
    if (t.shape != p.shape) {
      skipped.push_back(p.name + " (모양 다름)");
      continue;
    }
    std::memcpy(&h[p.off], t.f32(), p.n * 4);
  }
  CK(cudaMemcpy(vals, h.data(), total * 4, cudaMemcpyHostToDevice));
  return skipped;
}
void Params::save(const std::string& path, bool with_grads) const {
  std::vector<float> h(total), g;
  CK(cudaMemcpy(h.data(), vals, total * 4, cudaMemcpyDeviceToHost));
  if (with_grads) {
    g.resize(total);
    CK(cudaMemcpy(g.data(), grads, total * 4, cudaMemcpyDeviceToHost));
  }
  std::vector<StWrite> out;
  for (auto& p : list) {
    out.push_back({p.name, p.shape, &h[p.off]});
    if (with_grads && p.kind != P_RUN_MEAN && p.kind != P_RUN_VAR) out.push_back({"grad." + p.name, p.shape, &g[p.off]});
  }
  st_write(path, out);
}

namespace {
// 모든 합성곱 가중치를 한 번에: 블록 y = 표 번호, 블록 x·스레드 = 그 가중치 안 원소(KRSC 기준)
__global__ void to_krsc_k(const Params::ConvTab* tab, const float* vals, act_t* w16) {
  Params::ConvTab t = tab[blockIdx.y];
  int64_t n = int64_t(t.K) * t.RS * t.Cp;
  for (int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x; i < n; i += (int64_t)gridDim.x * blockDim.x) {
    int c = int(i % t.Cp);
    int64_t q = i / t.Cp;
    int rs = int(q % t.RS), k = int(q / t.RS);
    w16[t.woff + i] = c < t.Cg ? from_f<act_t>(vals[t.off + (int64_t(k) * t.Cg + c) * t.RS + rs]) : from_f<act_t>(0.f);
  }
}
__global__ void from_krsc_k(const Params::ConvTab* tab, const act_t* dw16, float* grads, bool accum) {
  Params::ConvTab t = tab[blockIdx.y];
  int64_t n = int64_t(t.K) * t.RS * t.Cp;
  for (int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x; i < n; i += (int64_t)gridDim.x * blockDim.x) {
    int c = int(i % t.Cp);
    if (c >= t.Cg) continue;
    int64_t q = i / t.Cp;
    int rs = int(q % t.RS), k = int(q / t.RS);
    float v = to_f(dw16[t.woff + i]);
    float* p = grads + t.off + (int64_t(k) * t.Cg + c) * t.RS + rs;
    *p = accum ? *p + v : v;
  }
}
}  // namespace
void Params::sync_w16(cudaStream_t s) { to_krsc_k<<<dim3(64, ntab), 256, 0, s>>>(tab, vals, w16); }
void Params::gather_dw(cudaStream_t s, bool accum) { from_krsc_k<<<dim3(64, ntab), 256, 0, s>>>(tab, dw16, grads, accum); }

// ================= 연산 =================
class OpConvBN : public Op {
 public:
  Net* net;
  Tensor *in, *out;
  ConvOp conv;
  std::string pre;
  int act;
  bool detach;
  act_t* z = nullptr;
  float *mean = nullptr, *istd = nullptr;
  int pw, pg, pb, prm, prv;
  OpConvBN(Net* n, const std::string& name, Tensor* i, Tensor* o, int k, int s, int a, int groups, bool det)
      : net(n), in(i), out(o), pre(name), act(a), detach(det) {
    ConvGeom g{i->N, i->H, i->W, i->C, o->C, k, k, s, k / 2, groups};
    conv.init(net->dnn_, g);
    int K = o->C;
    bool first = (i == net->in_);  // 입력은 3 채널을 8 로 덧대 둔다(텐서코어 경로)
    pw = net->params.add(name + ".conv.weight", P_CONV, {K, first ? 3 : i->C / groups, k, k});
    if (first) net->params.list[pw].cpad = i->C;
    pg = net->params.add(name + ".bn.weight", P_BN_W, {K});
    pb = net->params.add(name + ".bn.bias", P_BN_B, {K});
    prm = net->params.add(name + ".bn.running_mean", P_RUN_MEAN, {K});
    prv = net->params.add(name + ".bn.running_var", P_RUN_VAR, {K});
    z = dalloc<act_t>(o->n());
    dw = groups == i->C && groups == K && k == 3 && s == 1;
    mean = dalloc<float>(K);
    istd = dalloc<float>(K);
  }
  const Param& P(int i) const { return net->params.list[i]; }
  float* V(int i) const { return net->params.vals + P(i).off; }
  float* G(int i) const { return net->params.grads + P(i).off; }
  bool dw = false;  // depthwise 3×3: 직접 짠 커널(cuDNN NHWC half 가 느림)
  void fwd(bool train) override {
    auto s = net->s_;
    if (dw)
      kern::dw3_fwd(in->x, net->params.w16 + P(pw).woff, z, in->N, in->H, in->W, in->C, s);
    else
      conv.fwd(in->x, net->params.w16 + P(pw).woff, z, net->ws, net->ws_bytes);
    int C = out->C;
    if (train && net->train_bn)
      kern::bn_stats(z, out->M(), C, net->acc_, mean, istd, V(prm), V(prv), net->cfg.bn_momentum, net->cfg.bn_eps, s);
    else
      kern::bn_eval_stats(C, V(prm), V(prv), net->cfg.bn_eps, mean, istd, s);
    kern::bn_apply(z, out->M(), C, mean, istd, V(pg), V(pb), act, out->x, s);
  }
  void bwd() override {
    auto s = net->s_;
    act_t* dz = net->dz_;
    kern::bn_backward(z, out->g, out->M(), out->C, mean, istd, V(pg), V(pb), act, net->acc_, dz, G(pg), G(pb), s);
    if (dw) {
      if (!detach && in->g) kern::dw3_dgrad(dz, net->params.w16 + P(pw).woff, in->g, in->N, in->H, in->W, in->C, s);
      kern::dw3_wgrad(in->x, dz, net->params.dw16 + P(pw).woff, net->dwtmp_, in->N, in->H, in->W, in->C, s);
      return;
    }
    if (!detach && in->g) conv.dgrad(dz, net->params.w16 + P(pw).woff, in->g, net->ws, net->ws_bytes, 1.f);
    conv.wgrad(in->x, dz, net->params.dw16 + P(pw).woff, net->ws, net->ws_bytes, 0.f);
  }
  std::string desc() const override { return "conv_bn " + pre; }
};

class OpConvBias : public Op {
 public:
  Net* net;
  Tensor *in, *out;
  ConvOp conv;
  std::string pre;
  bool detach;
  int pw, pb;
  OpConvBias(Net* n, const std::string& name, Tensor* i, Tensor* o, bool det) : net(n), in(i), out(o), pre(name), detach(det) {
    ConvGeom g{i->N, i->H, i->W, i->C, o->C, 1, 1, 1, 0, 1};
    conv.init(net->dnn_, g);
    pw = net->params.add(name + ".weight", P_CONV, {o->C, i->C, 1, 1});
    pb = net->params.add(name + ".bias", P_BIAS, {o->C});
  }
  void fwd(bool) override {
    auto& P = net->params;
    conv.fwd(in->x, P.w16 + P.list[pw].woff, out->x, net->ws, net->ws_bytes);
    kern::add_bias(out->x, out->M(), out->C, P.vals + P.list[pb].off, net->s_);
  }
  void bwd() override {
    auto& P = net->params;
    kern::bias_grad(out->g, out->M(), out->C, net->acc_, P.grads + P.list[pb].off, net->s_);
    if (!detach && in->g) conv.dgrad(out->g, P.w16 + P.list[pw].woff, in->g, net->ws, net->ws_bytes, 1.f);
    conv.wgrad(in->x, out->g, P.dw16 + P.list[pw].woff, net->ws, net->ws_bytes, 0.f);
  }
  std::string desc() const override { return "conv_bias " + pre; }
};

class OpCat : public Op {
 public:
  std::vector<Tensor*> ins;
  Tensor* out;
  cudaStream_t s;
  OpCat(std::vector<Tensor*> i, Tensor* o, cudaStream_t st) : ins(i), out(o), s(st) {}
  void fwd(bool) override {
    int c0 = 0;
    for (auto* t : ins) {
      kern::copy_ch(t->x, t->C, 0, out->x, out->C, c0, t->C, out->M(), false, s);
      c0 += t->C;
    }
  }
  void bwd() override {
    int c0 = 0;
    for (auto* t : ins) {
      if (t->g) kern::copy_ch(out->g, out->C, c0, t->g, t->C, 0, t->C, out->M(), true, s);
      c0 += t->C;
    }
  }
  std::string desc() const override { return "cat"; }
};
class OpSlice : public Op {
 public:
  Tensor *in, *out;
  int c0;
  cudaStream_t s;
  OpSlice(Tensor* i, Tensor* o, int c, cudaStream_t st) : in(i), out(o), c0(c), s(st) {}
  void fwd(bool) override { kern::copy_ch(in->x, in->C, c0, out->x, out->C, 0, out->C, out->M(), false, s); }
  void bwd() override {
    if (in->g) kern::copy_ch(out->g, out->C, 0, in->g, in->C, c0, out->C, out->M(), true, s);
  }
  std::string desc() const override { return "slice"; }
};
class OpAdd : public Op {
 public:
  Tensor *a, *b, *out;
  cudaStream_t s;
  OpAdd(Tensor* x, Tensor* y, Tensor* o, cudaStream_t st) : a(x), b(y), out(o), s(st) {}
  void fwd(bool) override { kern::add(a->x, b->x, out->x, out->n(), s); }
  void bwd() override {
    if (a->g) kern::acc(out->g, a->g, out->n(), s);
    if (b->g) kern::acc(out->g, b->g, out->n(), s);
  }
  std::string desc() const override { return "add"; }
};
class OpUp : public Op {
 public:
  Tensor *in, *out;
  cudaStream_t s;
  OpUp(Tensor* i, Tensor* o, cudaStream_t st) : in(i), out(o), s(st) {}
  void fwd(bool) override { kern::up2(in->x, out->x, in->N, in->H, in->W, in->C, s); }
  void bwd() override {
    if (in->g) kern::up2_bwd(out->g, in->g, in->N, in->H, in->W, in->C, s);
  }
  std::string desc() const override { return "up2"; }
};
class OpPool : public Op {
 public:
  Tensor *in, *out;
  int k;
  uint8_t* arg;
  cudaStream_t s;
  OpPool(Tensor* i, Tensor* o, int kk, cudaStream_t st) : in(i), out(o), k(kk), s(st) { arg = dalloc<uint8_t>(o->n()); }
  void fwd(bool) override { kern::maxpool(in->x, out->x, arg, in->N, in->H, in->W, in->C, k, s); }
  void bwd() override {
    if (in->g) kern::maxpool_bwd(out->g, arg, in->g, in->N, in->H, in->W, in->C, k, s);
  }
  std::string desc() const override { return "maxpool"; }
};

// ================= 쌓기 =================
Tensor* Net::tensor(int N, int H, int W, int C, bool grad) {
  auto t = std::make_unique<Tensor>();
  t->N = N;
  t->H = H;
  t->W = W;
  t->C = C;
  t->x = dalloc<act_t>(t->n());
  if (grad) {
    t->g = reinterpret_cast<act_t*>(grad_arena_n_ + 1);  // 자리 표시(finalize 에서 진짜 주소로)
    grad_arena_n_ += (t->n() + 127) / 128 * 128;
  }
  tensors_.push_back(std::move(t));
  return tensors_.back().get();
}
Tensor* Net::conv_bn(const std::string& name, Tensor* in, int K, int k, int s, int act, int groups, bool detach) {
  int H = (in->H + 2 * (k / 2) - k) / s + 1, W = (in->W + 2 * (k / 2) - k) / s + 1;
  Tensor* o = tensor(in->N, H, W, K);
  ops_.push_back(std::make_unique<OpConvBN>(this, name, in, o, k, s, act, groups, detach));
  named[name] = o;
  return o;
}
Tensor* Net::conv_bias(const std::string& name, Tensor* in, int K, bool detach) {
  Tensor* o = tensor(in->N, in->H, in->W, K);
  ops_.push_back(std::make_unique<OpConvBias>(this, name, in, o, detach));
  named[name] = o;
  return o;
}
Tensor* Net::cat(std::vector<Tensor*> ins) {
  int C = 0;
  for (auto* t : ins) C += t->C;
  Tensor* o = tensor(ins[0]->N, ins[0]->H, ins[0]->W, C);
  ops_.push_back(std::make_unique<OpCat>(ins, o, s_));
  return o;
}
Tensor* Net::slice(Tensor* in, int c0, int n) {
  Tensor* o = tensor(in->N, in->H, in->W, n);
  ops_.push_back(std::make_unique<OpSlice>(in, o, c0, s_));
  return o;
}
Tensor* Net::add(Tensor* a, Tensor* b) {
  Tensor* o = tensor(a->N, a->H, a->W, a->C);
  ops_.push_back(std::make_unique<OpAdd>(a, b, o, s_));
  return o;
}
Tensor* Net::up2(Tensor* in) {
  Tensor* o = tensor(in->N, in->H * 2, in->W * 2, in->C);
  ops_.push_back(std::make_unique<OpUp>(in, o, s_));
  return o;
}
Tensor* Net::pool(Tensor* in, int k) {
  Tensor* o = tensor(in->N, in->H, in->W, in->C);
  ops_.push_back(std::make_unique<OpPool>(in, o, k, s_));
  return o;
}

// Bottleneck(c1, c2, shortcut, k=(k1, k2), e)
Tensor* Net::bottleneck(const std::string& n, Tensor* x, int c2, bool shortcut, int k1, int k2, float e) {
  int c_ = int(c2 * e);
  Tensor* a = conv_bn(n + ".cv1", x, c_, k1, 1, 1);
  Tensor* b = conv_bn(n + ".cv2", a, c2, k2, 1, 1);
  return (shortcut && x->C == c2) ? add(x, b) : b;
}
// C3k = C3 의 병목을 k×k 두 번(e=1.0)으로
Tensor* Net::c3k(const std::string& n, Tensor* x, int c2, int nb, bool shortcut) {
  int c_ = int(c2 * 0.5f);
  Tensor* a = conv_bn(n + ".cv1", x, c_, 1, 1, 1);
  Tensor* b = conv_bn(n + ".cv2", x, c_, 1, 1, 1);
  for (int i = 0; i < nb; ++i) a = bottleneck(n + ".m." + std::to_string(i), a, c_, shortcut, 3, 3, 1.0f);
  return conv_bn(n + ".cv3", cat({a, b}), c2, 1, 1, 1);
}
// C3k2 (= C2f 틀): cv1 → [y0, y1] → 블록들(y1 부터 이어서) → cv2(cat(cv1 출력, 블록 출력들))
Tensor* Net::c3k2(const std::string& n, Tensor* x, int c2, int nb, bool use_c3k, float e, bool attn) {
  int c = int(c2 * e);
  Tensor* y = conv_bn(n + ".cv1", x, 2 * c, 1, 1, 1);
  std::vector<Tensor*> parts = {y};
  Tensor* prev = slice(y, c, c);
  for (int i = 0; i < nb; ++i) {
    std::string mi = n + ".m." + std::to_string(i);
    Tensor* z;
    if (attn) {
      z = bottleneck(mi + ".0", prev, c, true, 3, 3, 0.5f);
      z = psa(mi + ".1", z);
    } else if (use_c3k) {
      z = c3k(mi, prev, c, 2, true);
    } else {
      z = bottleneck(mi, prev, c, true, 3, 3, 0.5f);
    }
    parts.push_back(z);
    prev = z;
  }
  return conv_bn(n + ".cv2", cat(parts), c2, 1, 1, 1);
}
// PSABlock(c, attn_ratio 0.5, heads = max(c/64, 1)): x + attn(x), 그다음 x + ffn(x)
Tensor* Net::psa(const std::string& n, Tensor* x) {
  int c = x->C, nh = std::max(c / 64, 1), hd = c / nh, kd = hd / 2;
  Tensor* qkv = conv_bn(n + ".attn.qkv", x, c + 2 * nh * kd, 1, 1, 0);
  Tensor* o = tensor(x->N, x->H, x->W, c);
  Tensor* vf = tensor(x->N, x->H, x->W, c);
  ops_.push_back(make_attn_op(qkv, o, vf, nh, kd, hd, blas_, s_));
  Tensor* pe = conv_bn(n + ".attn.pe", vf, c, 3, 1, 0, c);
  Tensor* a = conv_bn(n + ".attn.proj", add(o, pe), c, 1, 1, 0);
  Tensor* x1 = add(x, a);
  Tensor* f = conv_bn(n + ".ffn.0", x1, 2 * c, 1, 1, 1);
  f = conv_bn(n + ".ffn.1", f, c, 1, 1, 0);
  return add(x1, f);
}
// SPPF(c1, c2, k=5, n=3, shortcut=True)
Tensor* Net::sppf(const std::string& n, Tensor* x, int c2) {
  int c_ = x->C / 2;
  Tensor* a = conv_bn(n + ".cv1", x, c_, 1, 1, 0);
  Tensor* p1 = pool(a, 5);
  Tensor* p2 = pool(p1, 5);
  Tensor* p3 = pool(p2, 5);
  Tensor* y = conv_bn(n + ".cv2", cat({a, p1, p2, p3}), c2, 1, 1, 1);
  return x->C == c2 ? add(y, x) : y;
}
// C2PSA(c1, c2=c1, n=1)
Tensor* Net::c2psa(const std::string& n, Tensor* x) {
  int c = x->C / 2;
  Tensor* y = conv_bn(n + ".cv1", x, 2 * c, 1, 1, 1);
  Tensor* a = slice(y, 0, c);
  Tensor* b = psa(n + ".m.0", slice(y, c, c));
  return conv_bn(n + ".cv2", cat({a, b}), x->C, 1, 1, 1);
}
// Detect(nc, reg_max=1, end2end): 가지마다 box(3×3, 3×3, 1×1+편향 → 4) · cls(DW3×3, 1×1, DW3×3, 1×1, 1×1+편향 → nc)
void Net::detect(const std::string& n, Tensor* p3, Tensor* p4, Tensor* p5) {
  Tensor* xs[3] = {p3, p4, p5};
  int c2 = std::max({16, p3->C / 4, 4}), c3 = std::max(p3->C, std::min(cfg.nc, 100));
  for (int br = 0; br < 2; ++br) {
    std::string bn = n + (br == 0 ? ".cv2." : ".one2one_cv2."), cn = n + (br == 0 ? ".cv3." : ".one2one_cv3.");
    bool det = br == 1;  // one2one 은 백본으로 기울기를 보내지 않는다(Ultralytics: x.detach())
    for (int l = 0; l < 3; ++l) {
      std::string L = std::to_string(l);
      Tensor* b = conv_bn(bn + L + ".0", xs[l], c2, 3, 1, 1, 1, det);
      b = conv_bn(bn + L + ".1", b, c2, 3, 1, 1);
      hbox_[br][l] = conv_bias(bn + L + ".2", b, 4);
      Tensor* c = conv_bn(cn + L + ".0.0", xs[l], xs[l]->C, 3, 1, 1, xs[l]->C, det);
      c = conv_bn(cn + L + ".0.1", c, c3, 1, 1, 1);
      c = conv_bn(cn + L + ".1.0", c, c3, 3, 1, 1, c3);
      c = conv_bn(cn + L + ".1.1", c, c3, 1, 1, 1);
      hcls_[br][l] = conv_bias(cn + L + ".2", c, cfg.nc);
    }
  }
}

void Net::build(const NetCfg& c, cudnnHandle_t dnn, cublasHandle_t blas, cudaStream_t s) {
  cfg = c;
  dnn_ = dnn;
  blas_ = blas;
  s_ = s;
  int B = cfg.batch, S = cfg.imgsz;
  YT_ASSERT(S % 32 == 0, "imgsz 는 32 의 배수");
  in_ = tensor(B, S, S, 8, false);  // RGB + 0 다섯(첫 합성곱 텐서코어용)
  Tensor* x0 = conv_bn("model.0", in_, 32, 3, 2, 1);
  Tensor* x1 = conv_bn("model.1", x0, 64, 3, 2, 1);
  Tensor* x2 = c3k2("model.2", x1, 128, 1, false, 0.25f, false);
  Tensor* x3 = conv_bn("model.3", x2, 128, 3, 2, 1);
  Tensor* x4 = c3k2("model.4", x3, 256, 1, false, 0.25f, false);
  Tensor* x5 = conv_bn("model.5", x4, 256, 3, 2, 1);
  Tensor* x6 = c3k2("model.6", x5, 256, 1, true, 0.5f, false);
  Tensor* x7 = conv_bn("model.7", x6, 512, 3, 2, 1);
  Tensor* x8 = c3k2("model.8", x7, 512, 1, true, 0.5f, false);
  Tensor* x9 = sppf("model.9", x8, 512);
  Tensor* x10 = c2psa("model.10", x9);
  Tensor* x13 = c3k2("model.13", cat({up2(x10), x6}), 256, 1, true, 0.5f, false);
  Tensor* x16 = c3k2("model.16", cat({up2(x13), x4}), 128, 1, true, 0.5f, false);
  Tensor* x19 = c3k2("model.19", cat({conv_bn("model.17", x16, 128, 3, 2, 1), x13}), 256, 1, true, 0.5f, false);
  Tensor* x22 = c3k2("model.22", cat({conv_bn("model.20", x19, 256, 3, 2, 1), x10}), 512, 1, true, 0.5f, true);
  named["x2"] = x2;
  named["x4"] = x4;
  named["x6"] = x6;
  named["x8"] = x8;
  named["x9"] = x9;
  named["x10"] = x10;
  named["x13"] = x13;
  named["x16"] = x16;
  named["x19"] = x19;
  named["x22"] = x22;
  detect("model.23", x16, x19, x22);
  finalize();
}

void Net::finalize() {
  params.allocate();
  // 기울기 한 덩어리
  grad_arena_ = dalloc<act_t>(grad_arena_n_);
  for (auto& t : tensors_)
    if (t->g) t->g = grad_arena_ + (reinterpret_cast<int64_t>(t->g) - 1);
  int64_t maxz = 0, maxC = 0;
  for (auto& t : tensors_) {
    maxz = std::max(maxz, t->n());
    maxC = std::max<int64_t>(maxC, t->C);
  }
  dz_ = dalloc<act_t>(maxz);
  dwtmp_ = dalloc<float>(9 * maxC);
  acc_ = dalloc<double>(4 * maxC);
  // 합성곱 알고리즘 재기(작업 공간 상한 256 MB)
  size_t lim = 1024u << 20;
  void* tmp_ws = nullptr;
  CK(cudaMalloc(&tmp_ws, lim));
  ws_bytes = 0;
  for (auto& op : ops_) {
    if (auto* c = dynamic_cast<OpConvBN*>(op.get()); c && c->dw) continue;
    if (auto* c = dynamic_cast<OpConvBN*>(op.get()))
      ws_bytes = std::max(ws_bytes, c->conv.tune(c->in->x, params.w16 + params.list[c->pw].woff, c->z,
                                                 c->in->g ? c->in->g : dz_, params.dw16 + params.list[c->pw].woff,
                                                 dz_, tmp_ws, lim, !c->detach && c->in->g));
    else if (auto* b = dynamic_cast<OpConvBias*>(op.get()))
      ws_bytes = std::max(ws_bytes, b->conv.tune(b->in->x, params.w16 + params.list[b->pw].woff, b->out->x,
                                                 b->in->g ? b->in->g : dz_, params.dw16 + params.list[b->pw].woff,
                                                 b->out->g, tmp_ws, lim, !b->detach && b->in->g));
  }
  CK(cudaFree(tmp_ws));
  ws = ws_bytes ? dalloc<char>(ws_bytes) : nullptr;
}

void Net::zero_act_grads() { CK(cudaMemsetAsync(grad_arena_, 0, grad_arena_n_ * sizeof(act_t), s_)); }

void Net::forward(bool train) {
  for (auto& op : ops_) op->fwd(train);
}
void Net::backward() {
  for (auto it = ops_.rbegin(); it != ops_.rend(); ++it) (*it)->bwd();
}

namespace {
std::vector<cudaEvent_t> g_ev;
void ensure_ev(size_t n) {
  while (g_ev.size() < n) {
    cudaEvent_t e;
    CK(cudaEventCreate(&e));
    g_ev.push_back(e);
  }
}
}  // namespace
void Net::forward_timed(bool train) {
  ensure_ev(ops_.size() + 1);
  CK(cudaEventRecord(g_ev[0], s_));
  for (size_t i = 0; i < ops_.size(); ++i) {
    ops_[i]->fwd(train);
    CK(cudaEventRecord(g_ev[i + 1], s_));
  }
  CK(cudaEventSynchronize(g_ev[ops_.size()]));
  for (size_t i = 0; i < ops_.size(); ++i) {
    float ms;
    CK(cudaEventElapsedTime(&ms, g_ev[i], g_ev[i + 1]));
    ops_[i]->ms_f += ms;
  }
}
void Net::backward_timed() {
  ensure_ev(ops_.size() + 1);
  size_t n = ops_.size();
  CK(cudaEventRecord(g_ev[0], s_));
  for (size_t k = 0; k < n; ++k) {
    ops_[n - 1 - k]->bwd();
    CK(cudaEventRecord(g_ev[k + 1], s_));
  }
  CK(cudaEventSynchronize(g_ev[n]));
  for (size_t k = 0; k < n; ++k) {
    float ms;
    CK(cudaEventElapsedTime(&ms, g_ev[k], g_ev[k + 1]));
    ops_[n - 1 - k]->ms_b += ms;
  }
}
