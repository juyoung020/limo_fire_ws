#include <cfloat>

#include "loss.cuh"

namespace {
constexpr float kEps = 1e-7f;  // bbox_iou eps
constexpr int kMaxK = 10;

// 4 방향 이중수: CIoU 를 예측 상자 (x1, y1, x2, y2) 로 미분한다
struct Dual {
  float v, d[4];
};
__device__ inline Dual dc(float v) { return {v, {0, 0, 0, 0}}; }
__device__ inline Dual dv(float v, int i) {
  Dual r{v, {0, 0, 0, 0}};
  r.d[i] = 1.f;
  return r;
}
#define DOP(op, expr_v, expr_d)                                       \
  __device__ inline Dual operator op(const Dual& a, const Dual& b) { \
    Dual r;                                                          \
    r.v = expr_v;                                                    \
    for (int i = 0; i < 4; ++i) r.d[i] = expr_d;                     \
    return r;                                                        \
  }
DOP(+, a.v + b.v, a.d[i] + b.d[i])
DOP(-, a.v - b.v, a.d[i] - b.d[i])
DOP(*, a.v* b.v, a.d[i] * b.v + a.v * b.d[i])
DOP(/, a.v / b.v, (a.d[i] * b.v - a.v * b.d[i]) / (b.v * b.v))
#undef DOP
__device__ inline Dual dmin(const Dual& a, const Dual& b) { return a.v <= b.v ? a : b; }
__device__ inline Dual dmax(const Dual& a, const Dual& b) { return a.v >= b.v ? a : b; }
__device__ inline Dual dclamp0(const Dual& a) { return a.v > 0.f ? a : dc(0.f); }
__device__ inline Dual datan(const Dual& a) {
  Dual r;
  r.v = atanf(a.v);
  float k = 1.f / (1.f + a.v * a.v);
  for (int i = 0; i < 4; ++i) r.d[i] = a.d[i] * k;
  return r;
}
__device__ inline float val(float x) { return x; }
__device__ inline float val(const Dual& x) { return x.v; }
__device__ inline float fmin_(float a, float b) { return fminf(a, b); }
__device__ inline float fmax_(float a, float b) { return fmaxf(a, b); }
__device__ inline float clamp0(float a) { return fmaxf(a, 0.f); }
__device__ inline Dual fmin_(const Dual& a, const Dual& b) { return dmin(a, b); }
__device__ inline Dual fmax_(const Dual& a, const Dual& b) { return dmax(a, b); }
__device__ inline Dual clamp0(const Dual& a) { return dclamp0(a); }
__device__ inline float atan_(float a) { return atanf(a); }
__device__ inline Dual atan_(const Dual& a) { return datan(a); }
template <class T>
__device__ inline T cst(float v);
template <>
__device__ inline float cst<float>(float v) {
  return v;
}
template <>
__device__ inline Dual cst<Dual>(float v) {
  return dc(v);
}

// Ultralytics bbox_iou(box1, box2, xywh=False, CIoU=True) 와 같은 식(α 는 기울기 없음)
template <class T>
__device__ T ciou(T a1, T b1, T a2, T b2, float c1, float d1, float c2, float d2) {
  // box1 = (a1, b1, a2, b2), box2 = (c1, d1, c2, d2)
  T w1 = a2 - a1, h1 = b2 - b1 + cst<T>(kEps);
  float w2 = c2 - c1, h2 = d2 - d1 + kEps;
  T iw = clamp0(fmin_(a2, cst<T>(c2)) - fmax_(a1, cst<T>(c1)));
  T ih = clamp0(fmin_(b2, cst<T>(d2)) - fmax_(b1, cst<T>(d1)));
  T inter = iw * ih;
  T uni = w1 * h1 + cst<T>(w2 * h2) - inter + cst<T>(kEps);
  T iou = inter / uni;
  T cw = fmax_(a2, cst<T>(c2)) - fmin_(a1, cst<T>(c1));
  T ch = fmax_(b2, cst<T>(d2)) - fmin_(b1, cst<T>(d1));
  T c2_ = cw * cw + ch * ch + cst<T>(kEps);
  T dx = cst<T>(c1 + c2) - a1 - a2, dy = cst<T>(d1 + d2) - b1 - b2;
  T rho2 = (dx * dx + dy * dy) / cst<T>(4.f);
  T at = cst<T>(atanf(w2 / h2)) - atan_(w1 / h1);
  T v = cst<T>(4.f / (float)(M_PI * M_PI)) * at * at;
  float alpha = val(v) / (val(v) - val(iou) + (1.f + kEps));
  return iou - (rho2 / c2_ + v * cst<T>(alpha));
}
// 정답이 box1, 예측이 box2 인 순서(배정기 iou_calculation)
__device__ inline float ciou_gt_pred(float4 g, float4 p) { return ciou<float>(g.x, g.y, g.z, g.w, p.x, p.y, p.z, p.w); }

struct Lv {
  int off[4], h[3], w[3], s[3];
};
__device__ inline void anchor_of(const Lv& L, int a, int& l, int& ix, int& iy) {
  l = a < L.off[1] ? 0 : (a < L.off[2] ? 1 : 2);
  int k = a - L.off[l];
  iy = k / L.w[l];
  ix = k % L.w[l];
}
__device__ inline float sigm(float x) { return 1.f / (1.f + __expf(-x)); }

// ---- 1: 예측 상자 해독, 상태 지우기 ----
__global__ void decode_k(HeadDev h, Lv L, int B, int A, float4* pbox, int* cnt, int* agt, float* tsc) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= B * A) return;
  int b = i / A, a = i % A, l, ix, iy;
  anchor_of(L, a, l, ix, iy);
  const act_t* p = h.box[l] + ((int64_t(b) * L.h[l] + iy) * L.w[l] + ix) * 4;
  float ax = ix + 0.5f, ay = iy + 0.5f;
  pbox[i] = make_float4(ax - to_f(p[0]), ay - to_f(p[1]), ax + to_f(p[2]), ay + to_f(p[3]));
  cnt[i] = 0;
  agt[i] = -1;
  tsc[i] = 0.f;
}

// 정답 g 의 STAL 후보 상자(변이 stride_val 보다 작으면 넓힘)
__device__ inline float4 stal_box(float4 g, float sv) {
  float cx = (g.x + g.z) * 0.5f, cy = (g.y + g.w) * 0.5f, w = g.z - g.x, h = g.w - g.y;
  if (w < sv) w = sv;
  if (h < sv) h = sv;
  return make_float4(cx - w * 0.5f, cy - h * 0.5f, cx + w * 0.5f, cy + h * 0.5f);
}
__device__ inline bool in_box(float4 e, float px, float py) {
  return px - e.x > 1e-9f && py - e.y > 1e-9f && e.z - px > 1e-9f && e.w - py > 1e-9f;
}
__device__ inline bool valid_gt(float4 g) { return g.x + g.y + g.z + g.w > 0.f; }

// 정렬 지표: score^0.5 · CIoU^6
__device__ inline float align_of(const HeadDev& h, const Lv& L, int nc, int b, int a, int cls, float4 gpx,
                                 const float4* pbox, int A, float* ov) {
  int l, ix, iy;
  anchor_of(L, a, l, ix, iy);
  float s = (float)L.s[l];
  float4 p = pbox[int64_t(b) * A + a];
  float4 pp = make_float4(p.x * s, p.y * s, p.z * s, p.w * s);
  float o = fmaxf(ciou_gt_pred(gpx, pp), 0.f);
  float sc = sigm(to_f(h.cls[l][((int64_t(b) * L.h[l] + iy) * L.w[l] + ix) * nc + cls]));
  if (ov) *ov = o;
  return powf(sc, 0.5f) * powf(o, 6.f);
}

// ---- 2: 정답마다 상위 k 후보(블록 하나 = 정답 하나) ----
__global__ void topk_k(HeadDev h, Lv L, GtDev gt, int B, int G, int A, int nc, int K, float sv, const float4* pbox,
                       int* topa, int* cnt, int* who) {
  int bg = blockIdx.x, b = bg / G, g = bg % G;
  int* out = topa + int64_t(bg) * kMaxK;
  if (threadIdx.x < kMaxK) out[threadIdx.x] = -1;
  if (g >= gt.n[b]) return;
  float4 gb = gt.box[bg];
  if (!valid_gt(gb)) return;
  int cls = gt.cls[bg];
  float4 e = stal_box(gb, sv);
  float tv[kMaxK];
  int ti[kMaxK];
  for (int j = 0; j < K; ++j) tv[j] = -1.f, ti[j] = INT_MAX;
  for (int l = 0; l < 3; ++l) {
    float s = (float)L.s[l];
    int x0 = max(0, (int)floorf(e.x / s - 0.5f)), x1 = min(L.w[l] - 1, (int)ceilf(e.z / s - 0.5f));
    int y0 = max(0, (int)floorf(e.y / s - 0.5f)), y1 = min(L.h[l] - 1, (int)ceilf(e.w / s - 0.5f));
    int nx = x1 - x0 + 1, ny = y1 - y0 + 1;
    if (nx <= 0 || ny <= 0) continue;
    for (int k = threadIdx.x; k < nx * ny; k += blockDim.x) {
      int ix = x0 + k % nx, iy = y0 + k / nx;
      if (!in_box(e, (ix + 0.5f) * s, (iy + 0.5f) * s)) continue;
      int a = L.off[l] + iy * L.w[l] + ix;
      float m = align_of(h, L, nc, b, a, cls, gb, pbox, A, nullptr);
      if (!(m > 0.f)) continue;  // 지표 0 인 후보는 뽑지 않는다(torch.topk 동점 처리와 가깝게)
      if (m > tv[K - 1] || (m == tv[K - 1] && a < ti[K - 1])) {
        int j = K - 1;
        while (j > 0 && (m > tv[j - 1] || (m == tv[j - 1] && a < ti[j - 1]))) {
          tv[j] = tv[j - 1];
          ti[j] = ti[j - 1];
          --j;
        }
        tv[j] = m;
        ti[j] = a;
      }
    }
  }
  // 블록 합치기: 스레드별 목록을 공유 메모리에 모아 스레드 0 이 고른다
  __shared__ float sv_[128 * kMaxK];
  __shared__ int si_[128 * kMaxK];
  for (int j = 0; j < K; ++j) {
    sv_[threadIdx.x * K + j] = tv[j];
    si_[threadIdx.x * K + j] = ti[j];
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    float bv[kMaxK];
    int bi[kMaxK];
    for (int j = 0; j < K; ++j) bv[j] = -1.f, bi[j] = INT_MAX;
    for (int t = 0; t < (int)blockDim.x * K; ++t) {
      float m = sv_[t];
      int a = si_[t];
      if (m < 0.f) continue;
      if (m > bv[K - 1] || (m == bv[K - 1] && a < bi[K - 1])) {
        int j = K - 1;
        while (j > 0 && (m > bv[j - 1] || (m == bv[j - 1] && a < bi[j - 1]))) {
          bv[j] = bv[j - 1];
          bi[j] = bi[j - 1];
          --j;
        }
        bv[j] = m;
        bi[j] = a;
      }
    }
    for (int j = 0; j < K; ++j)
      if (bv[j] >= 0.f) {
        out[j] = bi[j];
        int64_t ba = int64_t(b) * A + bi[j];
        atomicAdd(&cnt[ba], 1);
        who[ba] = g;
      }
  }
}

// ---- 3: 앵커별 배정(여럿이면 CIoU 최대 정답) ----
__global__ void resolve_k(HeadDev h, Lv L, GtDev gt, int B, int G, int A, float sv, const float4* pbox,
                          const int* cnt, const int* who, int* agt) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= B * A) return;
  int c = cnt[i];
  if (c == 0) return;
  if (c == 1) {
    agt[i] = who[i];
    return;
  }
  int b = i / A, a = i % A, l, ix, iy;
  anchor_of(L, a, l, ix, iy);
  float s = (float)L.s[l];
  float4 p = pbox[i];
  float4 pp = make_float4(p.x * s, p.y * s, p.z * s, p.w * s);
  float px = (ix + 0.5f) * s, py = (iy + 0.5f) * s;
  int best = 0;
  float bo = -1.f;
  for (int g = 0; g < gt.n[b]; ++g) {
    float4 gb = gt.box[b * G + g];
    float o = 0.f;
    if (valid_gt(gb) && in_box(stal_box(gb, sv), px, py)) o = fmaxf(ciou_gt_pred(gb, pp), 0.f);
    if (o > bo) bo = o, best = g;
  }
  agt[i] = best;
}

// ---- 4 (one2one): 정답마다 지표 최대 앵커 하나만 ----
__device__ inline unsigned long long pack(float m, int a) {
  return ((unsigned long long)__float_as_uint(m) << 32) | (unsigned)(~a);
}
__global__ void o2o_best_k(HeadDev h, Lv L, GtDev gt, int B, int G, int A, int nc, const float4* pbox, const int* agt,
                           unsigned long long* gbest) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= B * A || agt[i] < 0) return;
  int b = i / A, a = i % A, g = agt[i];
  float m = align_of(h, L, nc, b, a, gt.cls[b * G + g], gt.box[b * G + g], pbox, A, nullptr);
  atomicMax(&gbest[b * G + g], pack(m, a));
}
__global__ void o2o_keep_k(int B, int G, int A, const unsigned long long* gbest, int* agt) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= B * A || agt[i] < 0) return;
  int b = i / A, a = i % A;
  unsigned long long v = gbest[b * G + agt[i]];
  float m = __uint_as_float((unsigned)(v >> 32));
  int best = ~(int)(v & 0xffffffffu);
  if (best != a || !(m > 0.f)) agt[i] = -1;
}

// ---- 5: 정답별 양성 지표·CIoU 최대 → 목표 점수 ----
__global__ void pos_max_k(HeadDev h, Lv L, GtDev gt, int B, int G, int A, int nc, const float4* pbox, const int* agt,
                          float* galign, float* govl) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= B * A || agt[i] < 0) return;
  int b = i / A, a = i % A, g = agt[i];
  float o;
  float m = align_of(h, L, nc, b, a, gt.cls[b * G + g], gt.box[b * G + g], pbox, A, &o);
  atomicMax(reinterpret_cast<int*>(&galign[b * G + g]), __float_as_int(m));  // 음이 아닌 float 은 int 순서와 같다
  atomicMax(reinterpret_cast<int*>(&govl[b * G + g]), __float_as_int(o));
}
__global__ void target_k(HeadDev h, Lv L, GtDev gt, int B, int G, int A, int nc, const float4* pbox, const int* agt,
                         const float* galign, const float* govl, float* tsc, double* sums) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  float t = 0.f;
  if (i < B * A && agt[i] >= 0) {
    int b = i / A, a = i % A, g = agt[i];
    float m = align_of(h, L, nc, b, a, gt.cls[b * G + g], gt.box[b * G + g], pbox, A, nullptr);
    t = m * govl[b * G + g] / (galign[b * G + g] + 1e-9f);
    tsc[i] = t;
  }
  // 블록 합 → double
  __shared__ float sh[256];
  sh[threadIdx.x] = t;
  __syncthreads();
  for (int o = blockDim.x / 2; o; o >>= 1) {
    if (threadIdx.x < o) sh[threadIdx.x] += sh[threadIdx.x + o];
    __syncthreads();
  }
  if (threadIdx.x == 0 && sh[0] != 0.f) atomicAdd(&sums[0], (double)sh[0]);
}

// ---- 6: 손실과 기울기 ----
struct Gains {
  float box, cls, l1, mult;  // mult = 배치 (가지 가중치는 장치에서 따로)
  const float* weight;
};
__global__ void loss_k(HeadDev h, Lv L, GtDev gt, int B, int G, int A, int nc, int imgsz, Gains gn, const float* scale,
                       const float4* pbox, const int* agt, const float* tsc, double* sums) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  float lbox = 0.f, ll1 = 0.f, lcls = 0.f;
  if (i < B * A) {
    int b = i / A, a = i % A, l, ix, iy;
    anchor_of(L, a, l, ix, iy);
    float tss = fmaxf((float)sums[0], 1.f);
    float sc = scale ? *scale : 1.f;
    float k = gn.mult * (gn.weight ? *gn.weight : 1.f) * sc / tss;
    int g = agt[i], lab = g >= 0 ? gt.cls[b * G + g] : -1;
    float t = tsc[i];
    int64_t cell = (int64_t(b) * L.h[l] + iy) * L.w[l] + ix;
    // 분류: BCE(로짓, 목표)
    const act_t* xc = h.cls[l] + cell * nc;
    act_t* dc_ = h.dcls[l] + cell * nc;
    for (int c = 0; c < nc; ++c) {
      float x = to_f(xc[c]), tt = (c == lab) ? t : 0.f;
      lcls += fmaxf(x, 0.f) - x * tt + log1pf(__expf(-fabsf(x)));
      dc_[c] = from_f<act_t>((sigm(x) - tt) * gn.cls * k);
    }
    // 상자: CIoU + L1(양성만)
    float dl[4] = {0, 0, 0, 0};
    if (g >= 0) {
      float s = (float)L.s[l];
      float4 gb = gt.box[b * G + g];
      float4 tg = make_float4(gb.x / s, gb.y / s, gb.z / s, gb.w / s);
      float4 p = pbox[i];
      Dual c = ciou<Dual>(dv(p.x, 0), dv(p.y, 1), dv(p.z, 2), dv(p.w, 3), tg.x, tg.y, tg.z, tg.w);
      lbox += (1.f - c.v) * t;
      // pred x1 = ax − l, y1 = ay − t, x2 = ax + r, y2 = ay + b
      float sgn[4] = {-1.f, -1.f, 1.f, 1.f};
      for (int j = 0; j < 4; ++j) dl[j] += gn.box * (-t * c.d[j]) * sgn[j];
      float ax = ix + 0.5f, ay = iy + 0.5f;
      const act_t* pr = h.box[l] + cell * 4;
      float tl[4] = {ax - tg.x, ay - tg.y, tg.z - ax, tg.w - ay};
      float sum = 0.f;
      for (int j = 0; j < 4; ++j) {
        float nrm = s / (float)imgsz;
        float d = to_f(pr[j]) * nrm - tl[j] * nrm;
        sum += fabsf(d);
        float sg = d > 0.f ? 1.f : (d < 0.f ? -1.f : 0.f);
        dl[j] += gn.l1 * sg * 0.25f * t * nrm;
      }
      ll1 += sum * 0.25f * t;
    }
    act_t* db = h.dbox[l] + cell * 4;
    for (int j = 0; j < 4; ++j) db[j] = from_f<act_t>(dl[j] * k);
  }
  __shared__ float s0[256], s1[256], s2[256];
  s0[threadIdx.x] = lbox;
  s1[threadIdx.x] = ll1;
  s2[threadIdx.x] = lcls;
  __syncthreads();
  for (int o = blockDim.x / 2; o; o >>= 1) {
    if (threadIdx.x < o) {
      s0[threadIdx.x] += s0[threadIdx.x + o];
      s1[threadIdx.x] += s1[threadIdx.x + o];
      s2[threadIdx.x] += s2[threadIdx.x + o];
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    atomicAdd(&sums[1], (double)s0[0]);
    atomicAdd(&sums[2], (double)s1[0]);
    atomicAdd(&sums[3], (double)s2[0]);
  }
}
__global__ void items_k(const double* sums, Gains gn, float* items, int br, const int* agt, int n) {
  // 한 블록: 양성 수 세기 + 항목
  __shared__ int cnt;
  if (threadIdx.x == 0) cnt = 0;
  __syncthreads();
  int c = 0;
  for (int i = threadIdx.x; i < n; i += blockDim.x) c += agt[i] >= 0;
  atomicAdd(&cnt, c);
  __syncthreads();
  if (threadIdx.x == 0) {
    double tss = sums[0] > 1.0 ? sums[0] : 1.0;
    items[3 * br + 0] = (float)(sums[1] / tss * gn.box);
    items[3 * br + 1] = (float)(sums[3] / tss * gn.cls);
    items[3 * br + 2] = (float)(sums[2] / tss * gn.l1);
    if (br == 1) {
      items[6] = (float)cnt;
      items[7] = (float)sums[0];
    }
  }
}
}  // namespace

void E2ELoss::init(const LossCfg& c) {
  cfg = c;
  A_ = c.A();
  int BA = c.B * A_, BG = c.B * c.G;
  pbox_ = dalloc<float4>(BA);
  topa_ = dalloc<int>(int64_t(BG) * kMaxK);
  cnt_ = dalloc<int>(BA);
  who_ = dalloc<int>(BA);
  agt_ = dalloc<int>(BA);
  tsc_ = dalloc<float>(BA);
  gbest_ = dalloc<unsigned long long>(BG);
  galign_ = dalloc<float>(BG);
  govl_ = dalloc<float>(BG);
  sums_ = dalloc<double>(4);
  items_ = dalloc<float>(8);
  CK(cudaMemset(items_, 0, 8 * 4));
}

void E2ELoss::run(int br, const HeadDev& h, const GtDev& gt, const float* weight, const float* scale, cudaStream_t s) {
  Lv L;
  L.off[0] = 0;
  for (int l = 0; l < 3; ++l) {
    L.h[l] = cfg.lh[l];
    L.w[l] = cfg.lw[l];
    L.s[l] = cfg.stride[l];
    L.off[l + 1] = L.off[l] + L.h[l] * L.w[l];
  }
  int B = cfg.B, G = cfg.G, A = A_, nc = cfg.nc, BA = B * A, BG = B * G;
  int K = br == 0 ? 10 : 7;
  float sv = (float)cfg.stride[1];  // STAL: stride_val = 두 번째 단계 보폭
  decode_k<<<cdiv(BA, 256), 256, 0, s>>>(h, L, B, A, pbox_, cnt_, agt_, tsc_);
  topk_k<<<BG, 128, 0, s>>>(h, L, gt, B, G, A, nc, K, sv, pbox_, topa_, cnt_, who_);
  resolve_k<<<cdiv(BA, 256), 256, 0, s>>>(h, L, gt, B, G, A, sv, pbox_, cnt_, who_, agt_);
  if (br == 1) {
    CK(cudaMemsetAsync(gbest_, 0, sizeof(unsigned long long) * BG, s));
    o2o_best_k<<<cdiv(BA, 256), 256, 0, s>>>(h, L, gt, B, G, A, nc, pbox_, agt_, gbest_);
    o2o_keep_k<<<cdiv(BA, 256), 256, 0, s>>>(B, G, A, gbest_, agt_);
  }
  CK(cudaMemsetAsync(galign_, 0, 4 * BG, s));
  CK(cudaMemsetAsync(govl_, 0, 4 * BG, s));
  CK(cudaMemsetAsync(sums_, 0, 8 * 4, s));
  pos_max_k<<<cdiv(BA, 256), 256, 0, s>>>(h, L, gt, B, G, A, nc, pbox_, agt_, galign_, govl_);
  target_k<<<cdiv(BA, 256), 256, 0, s>>>(h, L, gt, B, G, A, nc, pbox_, agt_, galign_, govl_, tsc_, sums_);
  Gains gn{cfg.box_gain, cfg.cls_gain, cfg.l1_gain, (float)B, weight};
  loss_k<<<cdiv(BA, 256), 256, 0, s>>>(h, L, gt, B, G, A, nc, cfg.imgsz, gn, scale, pbox_, agt_, tsc_, sums_);
  items_k<<<1, 256, 0, s>>>(sums_, gn, items_, br, agt_, BA);
}
