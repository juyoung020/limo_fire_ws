#include "data.h"

#include <turbojpeg.h>

#include <algorithm>
#include <chrono>
#include <cstring>
#include <cmath>
#include <map>
#include <numeric>
#include <random>

namespace {
std::mutex g_perm_mu;
std::map<std::pair<const void*, int>, std::vector<int>> g_perm;
const std::vector<int>& perm_of(const Cache* c, int epoch, uint64_t seed, bool shuffle) {
  std::lock_guard<std::mutex> lk(g_perm_mu);
  auto key = std::make_pair((const void*)c, shuffle ? epoch : -1);
  auto it = g_perm.find(key);
  if (it != g_perm.end()) return it->second;
  std::vector<int> o(c->n());
  std::iota(o.begin(), o.end(), 0);
  if (shuffle) {
    std::mt19937_64 r(seed * 1000003ull + (uint64_t)epoch);
    std::shuffle(o.begin(), o.end(), r);
  }
  if (g_perm.size() > 8) g_perm.erase(g_perm.begin());
  return g_perm.emplace(key, std::move(o)).first->second;
}
}  // namespace

void Loader::open(const Cache* cache, const AugCfg& c, int workers, int slots) {
  cache_ = cache;
  cfg = c;
  int S = c.S, B = c.batch;
  nb_ = batches_per_epoch();
  img_cap_ = int64_t(B) * (c.augment ? 4 : 1) * S * S * 3;
  if (c.masks) {
    YT_ASSERT(cache->has_masks(), "캐시에 마스크가 없음(yolods 마스크로 build_cache 다시)");
    int ms = 0;
    for (int i = 0; i < cache->n(); ++i) ms = std::max<int>(ms, cache->mask(i).size);
    msk_cap_ = int64_t(B) * (c.augment ? 4 : 1) * ms * ms;
  }
  slots_.resize(slots);
  for (auto& s : slots_) {
    CK(cudaMallocHost(&s.img, img_cap_));
    s.img_cap = img_cap_;
    CK(cudaMallocHost(&s.sp, sizeof(SampleParam) * B));
    CK(cudaMallocHost(&s.gt_n, sizeof(int) * B));
    CK(cudaMallocHost(&s.gt_box, sizeof(float4) * B * c.G));
    CK(cudaMallocHost(&s.gt_cls, sizeof(int) * B * c.G));
    CK(cudaMallocHost(&s.idx, sizeof(int) * B));
    if (msk_cap_) {
      CK(cudaMallocHost(&s.msk, msk_cap_));
      s.msk_cap = msk_cap_;
    }
  }
  for (int i = 0; i < 2; ++i) {
    d_img_[i] = dalloc<uint8_t>(img_cap_);
    d_sp_[i] = dalloc<SampleParam>(B);
    d_n_[i] = dalloc<int>(B);
    d_box_[i] = dalloc<float4>(int64_t(B) * c.G);
    d_cls_[i] = dalloc<int>(int64_t(B) * c.G);
    if (msk_cap_) {
      d_msk_[i] = dalloc<uint8_t>(msk_cap_);
      d_gm_[i] = dalloc<int16_t>(int64_t(B) * (S / c.mask_ratio) * (S / c.mask_ratio));
    }
    CK(cudaEventCreateWithFlags(&ev_copied_[i], cudaEventDisableTiming));
    CK(cudaEventCreateWithFlags(&ev_used_[i], cudaEventDisableTiming));
    CK(cudaEventRecord(ev_used_[i], 0));
  }
  CK(cudaStreamCreateWithFlags(&copy_, cudaStreamNonBlocking));
  for (int t = 0; t < workers; ++t) th_.emplace_back([this] { worker(); });
}

Loader::~Loader() {
  {
    std::lock_guard<std::mutex> lk(mu_);
    stop_ = true;
  }
  cv_.notify_all();
  for (auto& t : th_) t.join();
  for (auto& s : slots_) {
    cudaFreeHost(s.img);
    cudaFreeHost(s.sp);
    cudaFreeHost(s.gt_n);
    cudaFreeHost(s.gt_box);
    cudaFreeHost(s.gt_cls);
    cudaFreeHost(s.idx);
    if (s.msk) cudaFreeHost(s.msk);
  }
}

void Loader::skip(int64_t first) {
  std::unique_lock<std::mutex> lk(mu_);
  // 이미 채운(또는 채우는) 슬롯은 버리고 번호를 옮긴다
  cv_.wait(lk, [&] {
    for (auto& s : slots_)
      if (s.state == 1) return false;
    return true;
  });
  for (auto& s : slots_) s.state = 0;
  next_fill_ = next_take_ = first;
  lk.unlock();
  cv_.notify_all();
}

int64_t Loader::batches_per_epoch() const {
  int64_t n = cache_->n(), B = cfg.batch;
  return cfg.augment ? n / B : (n + B - 1) / B;
}

void Loader::worker() {
  for (;;) {
    int si = -1;
    int64_t b;
    {
      std::unique_lock<std::mutex> lk(mu_);
      cv_.wait(lk, [&] {
        if (stop_) return true;
        for (auto& s : slots_)
          if (s.state == 0) return true;
        return false;
      });
      if (stop_) return;
      for (int i = 0; i < (int)slots_.size(); ++i)
        if (slots_[i].state == 0) {
          si = i;
          break;
        }
      b = next_fill_++;
      slots_[si].state = 1;
      slots_[si].batch = b;
    }
    fill(slots_[si], b);
    {
      std::lock_guard<std::mutex> lk(mu_);
      slots_[si].state = 2;
    }
    cv_.notify_all();
  }
}

void Loader::fill(Slot& sl, int64_t gb) {
  const int S = cfg.S, B = cfg.batch, G = cfg.G;
  int64_t epoch = gb / nb_, k = gb % nb_;
  float mosaic_p = (cfg.augment && epoch < cfg.epochs - cfg.close_mosaic) ? cfg.mosaic : 0.f;
  const auto& order = perm_of(cache_, (int)epoch, cfg.seed, cfg.augment);
  std::mt19937_64 rng(cfg.seed * 0x9E3779B97F4A7C15ull ^ (uint64_t(gb) * 0xBF58476D1CE4E5B9ull + 1));
  std::uniform_real_distribution<double> U(0.0, 1.0);
  auto uni = [&](double a, double b) { return a + (b - a) * U(rng); };
  tjhandle dec = tjInitDecompress();
  sl.img_used = 0;
  sl.msk_used = 0;
  // 번호 지도 풀기(마스크를 쓸 때): 타일 t 에 캐시 번호 ci
  auto decode_m = [&](SampleParam& p, int t, int ci) {
    if (!cfg.masks) return;
    const CacheMask& m = cache_->mask(ci);
    int64_t n = int64_t(m.size) * m.size;
    YT_ASSERT(sl.msk_used + n <= sl.msk_cap, "슬롯 마스크 공간 부족");
    decode_mask(cache_->runs(ci), m.nrun, sl.msk + sl.msk_used, m.size);
    p.moff[t] = sl.msk_used;
    p.msz[t] = m.size;
    sl.msk_used += n;
  };
  auto decode = [&](int ci, int& w, int& h) -> int64_t {
    const CacheRec& r = cache_->rec(ci);
    w = r.w;
    h = r.h;
    int64_t off = sl.img_used;
    YT_ASSERT(off + int64_t(w) * h * 3 <= sl.img_cap, "슬롯 영상 공간 부족");
    tjDecompress2(dec, cache_->jpg(ci), r.jpg_len, sl.img + off, w, 0, h, TJPF_RGB, 0);
    sl.img_used += int64_t(w) * h * 3;
    return off;
  };
  struct Bx {
    float x1, y1, x2, y2;
    int c;
    int src;  // 타일 · 256 + 그 사진의 상자 번호(번호 지도와 짝)
  };
  for (int s = 0; s < B; ++s) {
    SampleParam& p = sl.sp[s];
    std::memset(&p, 0, sizeof(p));
    int64_t pos = k * B + s;
    sl.gt_n[s] = 0;
    if (pos >= (int64_t)order.size()) {  // 검증 마지막 배치 빈칸
      sl.idx[s] = -1;
      p.ntile = 0;
      p.hsv[0] = p.hsv[1] = p.hsv[2] = 1.f;
      continue;
    }
    int idx = order[pos];
    sl.idx[s] = idx;
    std::vector<Bx> boxes;
    auto add_boxes = [&](int ci, int w, int h, float px, float py, int t) {
      const CacheBox* cb = cache_->boxes(ci);
      for (int j = 0; j < cache_->rec(ci).nbox; ++j) {
        float cx = cb[j].cx * w, cy = cb[j].cy * h, bw = cb[j].w * w, bh = cb[j].h * h;
        boxes.push_back({cx - bw / 2 + px, cy - bh / 2 + py, cx + bw / 2 + px, cy + bh / 2 + py, (int)cb[j].cls,
                         j < 255 ? t * 256 + j : -1});
      }
    };
    if (!cfg.augment) {
      // 검증: 가운데 letterbox(scaleup 없음, 캐시는 이미 긴 변 S 이하)
      int w, h;
      p.ntile = 1;
      p.off[0] = decode(idx, w, h);
      decode_m(p, 0, idx);
      p.sw[0] = w;
      p.sh[0] = h;
      p.cw = w;
      p.ch = h;
      p.ta[0][0] = 0, p.ta[0][1] = 0, p.ta[0][2] = w, p.ta[0][3] = h;
      int left = (int)std::lround((S - w) / 2.0 - 0.1), top = (int)std::lround((S - h) / 2.0 - 0.1);
      p.Mi[0] = 1, p.Mi[1] = 0, p.Mi[2] = (float)-left;
      p.Mi[3] = 0, p.Mi[4] = 1, p.Mi[5] = (float)-top;
      p.hsv[0] = p.hsv[1] = p.hsv[2] = 1.f;
      add_boxes(idx, w, h, (float)left, (float)top, 0);
    } else {
      bool mosaic = U(rng) <= mosaic_p && mosaic_p > 0.f;
      if (mosaic) {
        int xc = (int)uni(S * 0.5, S * 1.5), yc = (int)uni(S * 0.5, S * 1.5);
        int ids[4] = {idx, 0, 0, 0};
        for (int t = 1; t < 4; ++t) ids[t] = (int)(U(rng) * cache_->n()) % cache_->n();
        p.ntile = 4;
        p.cw = p.ch = 2 * S;
        for (int t = 0; t < 4; ++t) {
          int w, h;
          p.off[t] = decode(ids[t], w, h);
          decode_m(p, t, ids[t]);
          p.sw[t] = w;
          p.sh[t] = h;
          int x1a, y1a, x2a, y2a, x1b, y1b;
          if (t == 0) {
            x1a = std::max(xc - w, 0), y1a = std::max(yc - h, 0), x2a = xc, y2a = yc;
            x1b = w - (x2a - x1a), y1b = h - (y2a - y1a);
          } else if (t == 1) {
            x1a = xc, y1a = std::max(yc - h, 0), x2a = std::min(xc + w, S * 2), y2a = yc;
            x1b = 0, y1b = h - (y2a - y1a);
          } else if (t == 2) {
            x1a = std::max(xc - w, 0), y1a = yc, x2a = xc, y2a = std::min(S * 2, yc + h);
            x1b = w - (x2a - x1a), y1b = 0;
          } else {
            x1a = xc, y1a = yc, x2a = std::min(xc + w, S * 2), y2a = std::min(S * 2, yc + h);
            x1b = 0, y1b = 0;
          }
          p.ta[t][0] = x1a, p.ta[t][1] = y1a, p.ta[t][2] = x2a, p.ta[t][3] = y2a;
          p.pad[t][0] = x1a - x1b;
          p.pad[t][1] = y1a - y1b;
          add_boxes(ids[t], w, h, (float)p.pad[t][0], (float)p.pad[t][1], t);
        }
        // 캔버스로 자르고 넓이 0 인 것 버림
        std::vector<Bx> kept;
        for (auto b : boxes) {
          b.x1 = std::clamp(b.x1, 0.f, 2.f * S), b.x2 = std::clamp(b.x2, 0.f, 2.f * S);
          b.y1 = std::clamp(b.y1, 0.f, 2.f * S), b.y2 = std::clamp(b.y2, 0.f, 2.f * S);
          if ((b.x2 - b.x1) * (b.y2 - b.y1) > 0) kept.push_back(b);
        }
        boxes.swap(kept);
      } else {
        int w, h;
        p.ntile = 1;
        p.off[0] = decode(idx, w, h);
        decode_m(p, 0, idx);
        p.sw[0] = w, p.sh[0] = h, p.cw = w, p.ch = h;
        p.ta[0][0] = 0, p.ta[0][1] = 0, p.ta[0][2] = w, p.ta[0][3] = h;
        add_boxes(idx, w, h, 0.f, 0.f, 0);
      }
      // RandomPerspective(degrees 0, shear 0, perspective 0): M = T·R(s)·C
      double sc = uni(1 - cfg.scale, 1 + cfg.scale);
      double tx = uni(0.5 - cfg.translate, 0.5 + cfg.translate) * S, ty = uni(0.5 - cfg.translate, 0.5 + cfg.translate) * S;
      double ox = sc * (-p.cw / 2.0) + tx, oy = sc * (-p.ch / 2.0) + ty;  // 출력 = sc·캔버스 + (ox, oy)
      p.Mi[0] = (float)(1 / sc), p.Mi[1] = 0, p.Mi[2] = (float)(-ox / sc);
      p.Mi[3] = 0, p.Mi[4] = (float)(1 / sc), p.Mi[5] = (float)(-oy / sc);
      std::vector<Bx> kept;
      for (auto b : boxes) {
        float w1 = (b.x2 - b.x1) * (float)sc, h1 = (b.y2 - b.y1) * (float)sc;
        Bx n{(float)(b.x1 * sc + ox), (float)(b.y1 * sc + oy), (float)(b.x2 * sc + ox), (float)(b.y2 * sc + oy), b.c, b.src};
        n.x1 = std::clamp(n.x1, 0.f, (float)S), n.x2 = std::clamp(n.x2, 0.f, (float)S);
        n.y1 = std::clamp(n.y1, 0.f, (float)S), n.y2 = std::clamp(n.y2, 0.f, (float)S);
        float w2 = n.x2 - n.x1, h2 = n.y2 - n.y1;
        float ar = std::max(w2 / (h2 + 1e-16f), h2 / (w2 + 1e-16f));
        if (w2 > 2 && h2 > 2 && w2 * h2 / (w1 * h1 + 1e-16f) > 0.1f && ar < 100) kept.push_back(n);
      }
      boxes.swap(kept);
      for (int c = 0; c < 3; ++c) {
        float g = c == 0 ? cfg.hsv_h : c == 1 ? cfg.hsv_s : cfg.hsv_v;
        p.hsv[c] = (float)(uni(-1, 1) * g + 1);
      }
      p.flip = U(rng) < cfg.fliplr;
      if (p.flip)
        for (auto& b : boxes) {
          float x1 = S - b.x2, x2 = S - b.x1;
          b.x1 = x1, b.x2 = x2;
        }
    }
    int n = std::min<int>((int)boxes.size(), G);
    sl.gt_n[s] = n;
    std::memset(p.lmap, 0xff, sizeof(p.lmap));
    for (int j = 0; j < n; ++j)
      if (boxes[j].src >= 0) p.lmap[boxes[j].src / 256][boxes[j].src % 256] = (int16_t)j;
    for (int j = 0; j < n; ++j) {
      sl.gt_box[s * G + j] = make_float4(boxes[j].x1, boxes[j].y1, boxes[j].x2, boxes[j].y2);
      sl.gt_cls[s * G + j] = boxes[j].c;
    }
  }
  tjDestroy(dec);
}

void Loader::next(act_t* input, int** gt_n, float4** gt_box, int** gt_cls, int* idx_host, cudaStream_t compute,
                  int16_t** gt_mask) {
  const int B = cfg.batch, G = cfg.G;
  // 앞 배치를 쓴 계산(학습 한 스텝 전부)이 다 올라간 뒤 표시 → 그 장치 버퍼를 다음다음에 다시 쓸 수 있다
  if (prev_ >= 0) CK(cudaEventRecord(ev_used_[prev_], compute));
  // 앞 배치 고정 메모리 슬롯 돌려주기(복사는 이미 끝나 있다)
  if (pending_ >= 0) {
    CK(cudaEventSynchronize(ev_copied_[prev_]));
    {
      std::lock_guard<std::mutex> lk(mu_);
      slots_[pending_].state = 0;
    }
    cv_.notify_all();
    pending_ = -1;
  }
  int si = -1;
  auto t0 = std::chrono::steady_clock::now();
  {
    std::unique_lock<std::mutex> lk(mu_);
    cv_.wait(lk, [&] {
      for (int i = 0; i < (int)slots_.size(); ++i)
        if (slots_[i].state == 2 && slots_[i].batch == next_take_) {
          si = i;
          return true;
        }
      return false;
    });
    slots_[si].state = 3;
    next_take_++;
  }
  wait_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
  Slot& sl = slots_[si];
  int c = cur_;
  cur_ ^= 1;
  CK(cudaStreamWaitEvent(copy_, ev_used_[c], 0));
  CK(cudaMemcpyAsync(d_img_[c], sl.img, sl.img_used, cudaMemcpyHostToDevice, copy_));
  CK(cudaMemcpyAsync(d_sp_[c], sl.sp, sizeof(SampleParam) * B, cudaMemcpyHostToDevice, copy_));
  CK(cudaMemcpyAsync(d_n_[c], sl.gt_n, sizeof(int) * B, cudaMemcpyHostToDevice, copy_));
  CK(cudaMemcpyAsync(d_box_[c], sl.gt_box, sizeof(float4) * B * G, cudaMemcpyHostToDevice, copy_));
  CK(cudaMemcpyAsync(d_cls_[c], sl.gt_cls, sizeof(int) * B * G, cudaMemcpyHostToDevice, copy_));
  if (msk_cap_) CK(cudaMemcpyAsync(d_msk_[c], sl.msk, sl.msk_used, cudaMemcpyHostToDevice, copy_));
  CK(cudaEventRecord(ev_copied_[c], copy_));
  if (idx_host) std::memcpy(idx_host, sl.idx, sizeof(int) * B);
  CK(cudaStreamWaitEvent(compute, ev_copied_[c], 0));
  augment_batch(d_img_[c], d_sp_[c], B, cfg.S, cfg.in_c, input, compute);
  if (msk_cap_) mask_targets(d_msk_[c], d_sp_[c], B, cfg.S, cfg.mask_ratio, d_gm_[c], compute);
  if (gt_mask) *gt_mask = d_gm_[c];
  *gt_n = d_n_[c];
  *gt_box = d_box_[c];
  *gt_cls = d_cls_[c];
  pending_ = si;
  prev_ = c;
}
