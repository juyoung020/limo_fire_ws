#include "segloss.cuh"

namespace {
constexpr int kMaxNm = 32;
constexpr int kMaxNc = 16;
constexpr int kTile = 16;   // 원형 마스크 기울기: 블록 하나가 16×16 화소
constexpr int kChunk = 256; // 그 블록이 한 번에 살피는 양성 수

struct Lv {
  int off[4], h[3], w[3];
};
__device__ inline void anchor_of(const Lv& L, int a, int& l, int& ix, int& iy) {
  l = a < L.off[1] ? 0 : (a < L.off[2] ? 1 : 2);
  int k = a - L.off[l];
  iy = k / L.w[l];
  ix = k % L.w[l];
}
__device__ inline float sigm(float x) { return 1.f / (1.f + __expf(-x)); }
__device__ inline float bce(float z, float t) { return fmaxf(z, 0.f) - z * t + log1pf(__expf(-fabsf(z))); }

// 블록 안 안정 압축: f 인 스레드에 스레드 차례대로 번호. 합계는 *total. ws: 공유 int[33]
__device__ int block_rank(bool f, int* ws, int* total) {
  int lane = threadIdx.x & 31, w = threadIdx.x >> 5, nw = blockDim.x >> 5;
  unsigned m = __ballot_sync(~0u, f);
  if (lane == 0) ws[w] = __popc(m);
  __syncthreads();
  if (threadIdx.x == 0) {
    int s = 0;
    for (int i = 0; i < nw; ++i) {
      int t = ws[i];
      ws[i] = s;
      s += t;
    }
    ws[32] = s;
  }
  __syncthreads();
  int r = ws[w] + __popc(m & ((1u << lane) - 1));
  *total = ws[32];
  __syncthreads();
  return r;
}

// ---- 1: 영상마다 양성 앵커 목록(앵커 차례) ----
__global__ void list_k(const int* agt, int A, int* list, int* cnt) {
  __shared__ int ws[33];
  int b = blockIdx.x, base = 0;
  const int* g = agt + int64_t(b) * A;
  int* out = list + int64_t(b) * A;
  for (int a0 = 0; a0 < A; a0 += blockDim.x) {
    int a = a0 + threadIdx.x, tot;
    bool f = a < A && g[a] >= 0;
    int r = block_rank(f, ws, &tot);
    if (f) out[base + r] = a;
    base += tot;
  }
  if (threadIdx.x == 0) cnt[b] = base;
}

__device__ inline int total_fg(const int* cnt, int B) {
  int n = 0;
  for (int b = 0; b < B; ++b) n += cnt[b];
  return n;
}

struct Geo {
  Lv L;
  int B, A, G, nm, mh, mw, imgsz;
  float gain, mult;
};

// 정답 상자 → 원형 마스크 격자 위 자르기 범위 [c0, c1) × [r0, r1) (Ultralytics crop_mask: r ≥ x1, r < x2) 와 넓이(0..1)
__device__ inline void crop_of(float4 gb, const Geo& q, int& c0, int& c1, int& r0, int& r1, float& area) {
  float s = 1.f / (float)q.imgsz;
  float x1 = gb.x * s, y1 = gb.y * s, x2 = gb.z * s, y2 = gb.w * s;
  area = (x2 - x1) * (y2 - y1);
  float X1 = x1 * q.mw, X2 = x2 * q.mw, Y1 = y1 * q.mh, Y2 = y2 * q.mh;
  c0 = max(0, (int)ceilf(X1));
  c1 = min(q.mw, (int)ceilf(X2));
  r0 = max(0, (int)ceilf(Y1));
  r1 = min(q.mh, (int)ceilf(Y2));
}

// ---- 2: 양성마다 BCE(자른 범위)·계수 기울기 ----
// 블록 하나가 양성 하나(영상 b 의 k 번째, k 는 blockIdx.x 부터 gridDim.x 씩)
__global__ void item_k(SegHeadDev h, GtDev gt, const int16_t* gm, const int* agt, const int* list, const int* cnt,
                       Geo q, const float* weight, const float* scale, float* iloss, float* ikf) {
  int b = blockIdx.y, n = cnt[b];
  int nfg = total_fg(cnt, q.B);
  __shared__ float red[kMaxNm + 1][8];
  for (int k = blockIdx.x; k < n; k += gridDim.x) {
    int a = list[int64_t(b) * q.A + k], g = agt[int64_t(b) * q.A + a], l, ix, iy;
    anchor_of(q.L, a, l, ix, iy);
    int64_t cell = (int64_t(b) * q.L.h[l] + iy) * q.L.w[l] + ix;
    float cf[kMaxNm];
    for (int j = 0; j < q.nm; ++j) cf[j] = to_f(h.coef[l][cell * q.nm + j]);
    int c0, c1, r0, r1;
    float area;
    crop_of(gt.box[b * q.G + g], q, c0, c1, r0, r1, area);
    int HW = q.mh * q.mw;
    float kf = q.gain * q.mult * (weight ? *weight : 1.f) * (scale ? *scale : 1.f) / ((float)HW * area * (float)nfg);
    float ls = 0.f, dc[kMaxNm];
    for (int j = 0; j < q.nm; ++j) dc[j] = 0.f;
    int cw = c1 - c0, np = cw > 0 && r1 > r0 ? cw * (r1 - r0) : 0;
    for (int p = threadIdx.x; p < np; p += blockDim.x) {
      int y = r0 + p / cw, x = c0 + p % cw;
      int64_t pix = (int64_t(b) * q.mh + y) * q.mw + x;
      const act_t* pv = h.proto + pix * q.nm;
      float pr[kMaxNm], z = 0.f;
      for (int j = 0; j < q.nm; ++j) {
        pr[j] = to_f(pv[j]);
        z += cf[j] * pr[j];
      }
      float m = gm[pix] == g + 1 ? 1.f : 0.f;
      ls += bce(z, m);
      float d = (sigm(z) - m) * kf;
      for (int j = 0; j < q.nm; ++j) dc[j] += d * pr[j];
    }
    // 블록 합(워프 → 공유)
    int lane = threadIdx.x & 31, w = threadIdx.x >> 5;
    for (int j = 0; j <= q.nm; ++j) {
      float v = j < q.nm ? dc[j] : ls;
      for (int o = 16; o; o >>= 1) v += __shfl_down_sync(~0u, v, o);
      if (lane == 0) red[j][w] = v;
    }
    __syncthreads();
    if (threadIdx.x <= q.nm) {
      float v = 0.f;
      for (int i = 0; i < (int)(blockDim.x >> 5); ++i) v += red[threadIdx.x][i];
      if (threadIdx.x < q.nm)
        h.dcoef[l][cell * q.nm + threadIdx.x] = from_f<act_t>(v);
      else {
        iloss[int64_t(b) * q.A + k] = v / ((float)HW * area);
        ikf[int64_t(b) * q.A + k] = kf;
      }
    }
    __syncthreads();
  }
}

// ---- 3: 원형 마스크 기울기 dproto[b, y, x, :] = Σ_i dP_i(y, x)·계수_i (화소가 i 의 자르기 범위 안일 때) ----
__global__ void proto_grad_k(SegHeadDev h, GtDev gt, const int16_t* gm, const int* agt, const int* list,
                             const int* cnt, Geo q, const float* ikf) {
  __shared__ float scf[kChunk][kMaxNm + 1];
  __shared__ int4 scr[kChunk];
  __shared__ float skf[kChunk];
  __shared__ int sg[kChunk];
  __shared__ int ws[33];
  int b = blockIdx.y, n = cnt[b];
  int tx = (blockIdx.x % ((q.mw + kTile - 1) / kTile)) * kTile, ty = (blockIdx.x / ((q.mw + kTile - 1) / kTile)) * kTile;
  int x = tx + threadIdx.x % kTile, y = ty + threadIdx.x / kTile;
  bool on = x < q.mw && y < q.mh;
  int64_t pix = (int64_t(b) * q.mh + min(y, q.mh - 1)) * q.mw + min(x, q.mw - 1);
  float pr[kMaxNm], acc[kMaxNm];
  for (int j = 0; j < q.nm; ++j) {
    pr[j] = on ? to_f(h.proto[pix * q.nm + j]) : 0.f;
    acc[j] = 0.f;
  }
  int lab = on ? gm[pix] : -1;
  for (int k0 = 0; k0 < n; k0 += kChunk) {
    // 이 덩어리에서 타일과 겹치는 양성만 공유 메모리로(차례 유지)
    int k = k0 + threadIdx.x, tot;
    bool f = false;
    int a = 0, g = 0, c0 = 0, c1 = 0, r0 = 0, r1 = 0;
    if (k < n) {
      a = list[int64_t(b) * q.A + k];
      g = agt[int64_t(b) * q.A + a];
      float area;
      crop_of(gt.box[b * q.G + g], q, c0, c1, r0, r1, area);
      f = c0 < min(c1, tx + kTile) && max(c0, tx) < c1 && r0 < min(r1, ty + kTile) && max(r0, ty) < r1;
    }
    int r = block_rank(f, ws, &tot);
    if (f) {
      int l, ix, iy;
      anchor_of(q.L, a, l, ix, iy);
      int64_t cell = (int64_t(b) * q.L.h[l] + iy) * q.L.w[l] + ix;
      for (int j = 0; j < q.nm; ++j) scf[r][j] = to_f(h.coef[l][cell * q.nm + j]);
      scr[r] = make_int4(c0, c1, r0, r1);
      skf[r] = ikf[int64_t(b) * q.A + k];
      sg[r] = g + 1;
    }
    __syncthreads();
    if (on)
      for (int i = 0; i < tot; ++i) {
        int4 c = scr[i];
        if (x < c.x || x >= c.y || y < c.z || y >= c.w) continue;
        float z = 0.f;
        for (int j = 0; j < q.nm; ++j) z += scf[i][j] * pr[j];
        float d = (sigm(z) - (lab == sg[i] ? 1.f : 0.f)) * skf[i];
        for (int j = 0; j < q.nm; ++j) acc[j] += d * scf[i][j];
      }
    __syncthreads();
  }
  if (on)
    for (int j = 0; j < q.nm; ++j) h.dproto[pix * q.nm + j] = from_f<act_t>(acc[j]);
}

// ---- 4: 의미 분할 — 영상마다 클래스별 Σσ·T, Σσ, ΣT 와 Σ BCE (블록 하나 = 영상 하나) ----
struct SemGeo {
  int B, nc, sh, sw, mh, mw, G;
};
__device__ inline float sem_t(const GtDev& gt, const int16_t* gm, const SemGeo& q, int b, int y, int x, int c) {
  // F.interpolate(nearest) 로 줄임: 원본 칸 = floor(y·mh/sh)
  int my = y * q.mh / q.sh, mx = x * q.mw / q.sw;
  int v = gm[(int64_t(b) * q.mh + my) * q.mw + mx];
  return v > 0 && gt.cls[b * q.G + v - 1] == c ? 1.f : 0.f;
}
__global__ void sem_stat_k(SegHeadDev h, GtDev gt, const int16_t* gm, SemGeo q, double* acc) {
  int b = blockIdx.x, n = q.sh * q.sw;
  float I[kMaxNc], P[kMaxNc], T[kMaxNc], ls = 0.f;
  for (int c = 0; c < q.nc; ++c) I[c] = P[c] = T[c] = 0.f;
  for (int p = threadIdx.x; p < n; p += blockDim.x) {
    int y = p / q.sw, x = p % q.sw;
    const act_t* z = h.sem + (int64_t(b) * n + p) * q.nc;
    for (int c = 0; c < q.nc; ++c) {
      float zz = to_f(z[c]), t = sem_t(gt, gm, q, b, y, x, c), s = sigm(zz);
      I[c] += s * t;
      P[c] += s;
      T[c] += t;
      ls += bce(zz, t);
    }
  }
  __shared__ float red[3 * kMaxNc + 1][32];
  int lane = threadIdx.x & 31, w = threadIdx.x >> 5, nv = 3 * q.nc + 1;
  for (int j = 0; j < nv; ++j) {
    float v = j < q.nc ? I[j] : j < 2 * q.nc ? P[j - q.nc] : j < 3 * q.nc ? T[j - 2 * q.nc] : ls;
    for (int o = 16; o; o >>= 1) v += __shfl_down_sync(~0u, v, o);
    if (lane == 0) red[j][w] = v;
  }
  __syncthreads();
  if (threadIdx.x < nv) {
    double v = 0;
    for (int i = 0; i < (int)(blockDim.x >> 5); ++i) v += red[threadIdx.x][i];
    int j = threadIdx.x;
    if (j < 3 * q.nc)
      acc[(int64_t(b) * q.nc + j % q.nc) * 3 + j / q.nc] = v;
    else
      acc[int64_t(q.B) * q.nc * 3 + b] = v;
  }
}
__global__ void sem_grad_k(SegHeadDev h, GtDev gt, const int16_t* gm, SemGeo q, const double* acc, const int* cnt,
                           float gain, float mult, const float* weight, const float* scale) {
  int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x, n = int64_t(q.B) * q.sh * q.sw * q.nc;
  if (i >= n) return;
  int c = int(i % q.nc);
  int64_t p = i / q.nc;
  int x = int(p % q.sw), y = int((p / q.sw) % q.sh), b = int(p / (int64_t(q.sw) * q.sh));
  if (total_fg(cnt, q.B) == 0) {
    h.dsem[i] = from_f<act_t>(0.f);
    return;
  }
  float k = gain * mult * (weight ? *weight : 1.f) * (scale ? *scale : 1.f);
  float z = to_f(h.sem[i]), s = sigm(z), t = sem_t(gt, gm, q, b, y, x, c);
  const double* a = acc + (int64_t(b) * q.nc + c) * 3;
  float I = (float)a[0], U = (float)(a[1] + a[2]);
  float dd = -(2.f * t * (U + 1.f) - (2.f * I + 1.f)) / ((U + 1.f) * (U + 1.f)) / (float)(q.B * q.nc);
  float g = 0.5f * (s - t) / (float)n + 0.5f * dd * s * (1.f - s);
  h.dsem[i] = from_f<act_t>(g * k);
}

// ---- 5: 항목 ----
__global__ void items_k(const float* iloss, const int* cnt, int B, int A, const double* sacc, SemGeo q, float gain,
                        bool sem, float* items, int br) {
  __shared__ double red[256];
  int nfg = total_fg(cnt, B);
  double v = 0;
  for (int b = 0; b < B; ++b)
    for (int k = threadIdx.x; k < cnt[b]; k += blockDim.x) v += iloss[int64_t(b) * A + k];
  red[threadIdx.x] = v;
  __syncthreads();
  for (int o = blockDim.x / 2; o; o >>= 1) {
    if (threadIdx.x < o) red[threadIdx.x] += red[threadIdx.x + o];
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    items[2 * br] = nfg ? (float)(red[0] / nfg * gain) : 0.f;
    float sl = 0.f;
    if (sem && nfg) {
      double bs = 0, dl = 0;
      for (int b = 0; b < q.B; ++b) {
        bs += sacc[int64_t(q.B) * q.nc * 3 + b];
        for (int c = 0; c < q.nc; ++c) {
          const double* a = sacc + (int64_t(b) * q.nc + c) * 3;
          dl += 1.0 - (2.0 * a[0] + 1.0) / (a[1] + a[2] + 1.0);
        }
      }
      double N = double(q.B) * q.nc * q.sh * q.sw;
      sl = (float)((0.5 * bs / N + 0.5 * dl / (q.B * q.nc)) * gain);
    }
    items[2 * br + 1] = sl;
  }
}
}  // namespace

void SegLoss::init(const LossCfg& c, int nm_) {
  cfg = c;
  nm = nm_;
  YT_ASSERT(nm <= kMaxNm && c.nc <= kMaxNc, "nm %d, nc %d 너무 큼", nm, c.nc);
  A_ = c.A();
  mh = c.imgsz / 4;
  mw = c.imgsz / 4;
  sh = c.imgsz / 8;
  sw = c.imgsz / 8;
  list_ = dalloc<int>(int64_t(c.B) * A_);
  cnt_ = dalloc<int>(c.B);
  iloss_ = dalloc<float>(int64_t(c.B) * A_);
  ikf_ = dalloc<float>(int64_t(c.B) * A_);
  semacc_ = dalloc<double>(int64_t(c.B) * c.nc * 3 + c.B);
  items_ = dalloc<float>(4);
  CK(cudaMemset(items_, 0, 16));
}

void SegLoss::run(int br, const SegHeadDev& h, const GtDev& gt, const int16_t* gt_mask, const int* agt,
                  const float* weight, const float* scale, cudaStream_t s) {
  Geo q;
  q.L.off[0] = 0;
  for (int l = 0; l < 3; ++l) {
    q.L.h[l] = cfg.lh[l];
    q.L.w[l] = cfg.lw[l];
    q.L.off[l + 1] = q.L.off[l] + q.L.h[l] * q.L.w[l];
  }
  q.B = cfg.B;
  q.A = A_;
  q.G = cfg.G;
  q.nm = nm;
  q.mh = mh;
  q.mw = mw;
  q.imgsz = cfg.imgsz;
  q.gain = cfg.box_gain;
  q.mult = (float)cfg.B;
  for (int l = 0; l < 3; ++l)
    CK(cudaMemsetAsync(h.dcoef[l], 0, sizeof(act_t) * int64_t(cfg.B) * cfg.lh[l] * cfg.lw[l] * nm, s));
  list_k<<<cfg.B, 1024, 0, s>>>(agt, A_, list_, cnt_);
  item_k<<<dim3(64, cfg.B), 256, 0, s>>>(h, gt, gt_mask, agt, list_, cnt_, q, weight, scale, iloss_, ikf_);
  if (h.dproto) {
    int tiles = ((mw + kTile - 1) / kTile) * ((mh + kTile - 1) / kTile);
    proto_grad_k<<<dim3(tiles, cfg.B), kTile * kTile, 0, s>>>(h, gt, gt_mask, agt, list_, cnt_, q, ikf_);
  }
  SemGeo sq{cfg.B, cfg.nc, sh, sw, mh, mw, cfg.G};
  if (h.sem) {
    sem_stat_k<<<cfg.B, 512, 0, s>>>(h, gt, gt_mask, sq, semacc_);
    if (h.dsem) {
      int64_t n = int64_t(cfg.B) * sh * sw * cfg.nc;
      sem_grad_k<<<cdiv(n, 256), 256, 0, s>>>(h, gt, gt_mask, sq, semacc_, cnt_, cfg.box_gain, (float)cfg.B, weight,
                                                scale);
    }
  }
  items_k<<<1, 256, 0, s>>>(iloss_, cnt_, cfg.B, A_, semacc_, sq, cfg.box_gain, h.sem != nullptr, items_, br);
}
