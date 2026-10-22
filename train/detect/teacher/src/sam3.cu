#include "sam3.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <type_traits>

namespace sam3 {

namespace {
inline int nb(long long n, int t = 256) { return (int)((n + t - 1) / t); }
template <typename T>
__device__ __forceinline__ T fromf(float x) {
  if constexpr (std::is_same_v<T, float>)
    return x;
  else if constexpr (std::is_same_v<T, f16>)
    return __float2half(x);
  else
    return __float2bfloat16(x);
}
template <typename T>
__device__ __forceinline__ float tof(T x) {
  if constexpr (std::is_same_v<T, float>)
    return x;
  else if constexpr (std::is_same_v<T, f16>)
    return __half2float(x);
  else
    return __bfloat162float(x);
}

// ---- 이 모델에만 쓰는 작은 커널 ----
__global__ void embed_text_k(const int* ids, const float* tok, const float* pos, float* x, int rows, int C, int L) {
  long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x;
  if (i >= (long long)rows * C) return;
  int r = (int)(i / C), c = (int)(i % C);
  x[i] = tok[(long long)ids[r] * C + c] + pos[(r % L) * C + c];
}
// y[r] = a[r] + p[r % n]
template <typename T>
__global__ void add_bcast_k(const float* a, const float* p, T* y, long long R, int C, int n) {
  long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x;
  if (i >= R * C) return;
  long long r = i / C;
  int c = (int)(i % C);
  y[i] = fromf<T>(a[i] + p[(r % n) * C + c]);
}
// 원본 행 r (G 개씩 묶음) → 대상 행 r + (r/G + 1) * shift, 편향·활성화
template <typename T>
__global__ void shift_rows_k(const float* x, int ldx, const float* b, T* y, int ldy, long long rows, int cols, int G,
                             int shift, int act) {
  long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x;
  if (i >= rows * cols) return;
  long long r = i / cols;
  int c = (int)(i % cols);
  float v = x[r * ldx + c] + (b ? b[c] : 0.f);
  if (act == 2) v = fmaxf(v, 0.f);
  long long d = r + (r / G + 1) * shift;
  y[d * ldy + c] = fromf<T>(v);
}
__global__ void init_dec_k(float* h, const float* pres, const float* qe, float* ref, const float* rp, int P) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= P * (NQ + 1) * D) return;
  int c = i % D, r = (i / D) % (NQ + 1), b = i / (D * (NQ + 1));
  h[i] = r == 0 ? pres[c] : qe[(r - 1) * D + c];
  if (c < 4 && r > 0) ref[((long long)b * NQ + r - 1) * 4 + c] = 1.f / (1.f + expf(-rp[(r - 1) * 4 + c]));
}
// 상자 사인 임베딩: ref [n, 4] (cx, cy, w, h) → [n, 512] = (y, x, w, h) 순, 각 128 = (sin, cos) 교대
template <typename T>
__global__ void box_sine_k(const float* ref, T* out, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n * 512) return;
  int r = i / 512, c = i % 512, part = c / 128, j = c % 128;
  int comp = part == 0 ? 1 : part == 1 ? 0 : part;  // y, x, w, h
  float scale = 6.283185307179586f;
  float dim_t = powf(10000.f, (float)(2 * (j / 2)) / 128.f);
  float v = ref[r * 4 + comp] * scale / dim_t;
  out[i] = fromf<T>((j % 2 == 0) ? sinf(v) : cosf(v));
}
// BoxRPB 한 번에: 좌표 − 상자 경계(로그 눈금) → MLP(2→256 ReLU→8) → dst[b, 1+q, j, 헤드] (존재 토큰 줄 0 은 0 그대로)
// w = [W1(256×2) | b1(256) | W2(8×256) | b2(8)] float. axis 0 = x, 1 = y
__global__ void rpb_k(const float* ref, const float* w, float* dst, int nbox, int n, int axis) {
  __shared__ float sw[D * 2 + D + HEADS * D + HEADS];
  for (int i = threadIdx.x; i < D * 2 + D + HEADS * D + HEADS; i += blockDim.x) sw[i] = w[i];
  __syncthreads();
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= nbox * n) return;
  int bq = i / n, j = i % n;
  const float* r = ref + bq * 4;
  float c = axis == 0 ? r[0] : r[1], wd = axis == 0 ? r[2] : r[3];
  float lo = c - 0.5f * wd, hi = c + 0.5f * wd, coord = (float)j / (float)n;
  float d[2] = {coord - lo, coord - hi};
  for (int e = 0; e < 2; ++e) {
    float x = d[e] * 8.f;
    float sg = x > 0.f ? 1.f : (x < 0.f ? -1.f : 0.f);
    d[e] = sg * log2f(fabsf(x) + 1.f) / 3.f;
  }
  const float *W1 = sw, *B1 = sw + 2 * D, *W2 = sw + 3 * D, *B2 = sw + 3 * D + HEADS * D;
  float o[HEADS];
#pragma unroll
  for (int h = 0; h < HEADS; ++h) o[h] = 0.f;
  for (int u = 0; u < D; ++u) {
    float hv = fmaxf(d[0] * W1[2 * u] + d[1] * W1[2 * u + 1] + B1[u], 0.f);
#pragma unroll
    for (int h = 0; h < HEADS; ++h) o[h] += W2[h * D + u] * hv;
  }
  int b = bq / NQ, q = bq % NQ;
  float* out = dst + (((long long)b * (NQ + 1) + 1 + q) * n + j) * HEADS;
#pragma unroll
  for (int h = 0; h < HEADS; ++h) out[h] = o[h] + B2[h];
}
__device__ __forceinline__ float inv_sigmoid(float x) {
  x = fminf(fmaxf(x, 0.f), 1.f);
  float a = fmaxf(x, 1e-3f), b = fmaxf(1.f - x, 1e-3f);
  return logf(a / b);
}
// ref[b, q] = sigmoid((delta[b*201+1+q] + bias) + inv_sigmoid(ref[b, q]))
__global__ void box_refine_k(const float* delta, const float* bias, float* ref, int P) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= P * NQ * 4) return;
  int c = i % 4, q = (i / 4) % NQ, b = i / (4 * NQ);
  float v = (delta[((long long)b * (NQ + 1) + 1 + q) * 4 + c] + bias[c]) + inv_sigmoid(ref[i]);
  ref[i] = 1.f / (1.f + expf(-v));
}
__global__ void xyxy_k(const float* ref, float* box, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  const float* r = ref + i * 4;
  box[i * 4 + 0] = r[0] - 0.5f * r[2];
  box[i * 4 + 1] = r[1] - 0.5f * r[3];
  box[i * 4 + 2] = r[0] + 0.5f * r[2];
  box[i * 4 + 3] = r[1] + 0.5f * r[3];
}
// logits[b, q] = clamp(dot(pq[b*201+1+q] + bq, pt[b]) / 16, ±12)
__global__ void score_k(const float* pq, const float* bq, const float* pt, float* logits, int P) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= P * NQ) return;
  int b = i / NQ, q = i % NQ;
  const float* x = pq + ((long long)b * (NQ + 1) + 1 + q) * D;
  float s = 0.f;
  for (int c = 0; c < D; ++c) s += (x[c] + bq[c]) * pt[b * D + c];
  logits[i] = fminf(fmaxf(s * 0.0625f, -12.f), 12.f);
}
__global__ void presence_k(const float* x, const float* b, float* out, int P) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < P) out[i] = fminf(fmaxf(x[i] + b[0], -10.f), 10.f);
}
// 행 고르기: y[i] = x[rows[i]]
template <typename T>
__global__ void gather_rows_k(const T* x, const int* rows, T* y, int n, int C) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n * C) return;
  y[i] = x[(long long)rows[i / C] * C + i % C];
}

std::vector<float> to_vec(const StTensor& t) { return std::vector<float>(t.f32(), t.f32() + t.numel()); }
}  // namespace

// ================= 불러오기 =================
template <typename T>
void Model<T>::upload_T(Buf<T>& dst, const std::vector<float>& h) {
  Buf<float> tmp;
  tmp.alloc(h.size());
  CK(cudaMemcpyAsync(tmp.p, h.data(), h.size() * 4, cudaMemcpyHostToDevice, s_));
  dst.alloc(h.size());
  k::cast<T>(tmp.p, dst.p, (long long)h.size(), s_);
  CK(cudaStreamSynchronize(s_));
}
template <typename T>
void Model<T>::upload_f(Buf<float>& dst, const float* h, size_t n) {
  dst.alloc(n);
  CK(cudaMemcpyAsync(dst.p, h, n * 4, cudaMemcpyHostToDevice, s_));
  CK(cudaStreamSynchronize(s_));
}
template <typename T>
void Model<T>::load_lin(Lin<T>& l, const std::string& name, int out, int in, bool bias) {
  const auto& w = st_.get(name + ".weight");
  if (w.numel() != (int64_t)out * in) throw std::runtime_error("크기 다름: " + name);
  l.out = out;
  l.in = in;
  upload_T(l.w, to_vec(w));
  if (bias) {
    upload_f(l.b, st_.get(name + ".bias", {out}).f32(), out);
    set_bh(l);
  }
}
template <typename T>
void Model<T>::set_bh(Lin<T>& l) {
  l.bh.alloc(l.b.n);
  k::cast<T>(l.b.p, l.bh.p, (long long)l.b.n, s_);
  CK(cudaStreamSynchronize(s_));
}
template <typename T>
void Model<T>::load_lin_cat(Lin<T>& l, const std::vector<std::string>& names, int out_each, int in) {
  std::vector<float> w, b;
  for (auto& n : names) {
    const auto& tw = st_.get(n + ".weight", {out_each, in});
    const auto& tb = st_.get(n + ".bias", {out_each});
    w.insert(w.end(), tw.f32(), tw.f32() + tw.numel());
    b.insert(b.end(), tb.f32(), tb.f32() + tb.numel());
  }
  l.out = out_each * (int)names.size();
  l.in = in;
  upload_T(l.w, w);
  upload_f(l.b, b.data(), b.size());
  set_bh(l);
}
template <typename T>
void Model<T>::load_norm(Norm& n, const std::string& name, int C) {
  upload_f(n.g, st_.get(name + ".weight", {C}).f32(), C);
  upload_f(n.b, st_.get(name + ".bias", {C}).f32(), C);
}

template <typename T>
Model<T>::Model(const SafeTensors& st, int max_prompts, cudaStream_t s) : st_(st), s_(s), maxP_(max_prompts) {
  CB(cublasCreate(&h_));
  CB(cublasSetStream(h_, s_));
  CB(cublasSetMathMode(h_, std::is_same_v<T, float> ? CUBLAS_PEDANTIC_MATH : CUBLAS_DEFAULT_MATH));
  if (kFast) lt_.init(s_);

  // ---- ViT ----
  const std::string V = "detector_model.vision_encoder.backbone.";
  load_lin(patch, V + "embeddings.patch_embeddings.projection", VD, 3 * PATCH * PATCH, false);
  upload_f(vpos, st_.get(V + "embeddings.position_embeddings", {1, 576, VD}).f32(), 576 * VD);
  load_norm(vln_pre, V + "layer_norm", VD);
  vl.resize(VLAYERS);
  for (int i = 0; i < VLAYERS; ++i) {
    std::string p = V + "layers." + std::to_string(i) + ".";
    load_norm(vl[i].ln1, p + "layer_norm1", VD);
    load_norm(vl[i].ln2, p + "layer_norm2", VD);
    load_lin_cat(vl[i].qkv, {p + "attention.q_proj", p + "attention.k_proj", p + "attention.v_proj"}, VD, VD);
    load_lin(vl[i].o, p + "attention.o_proj", VD, VD);
    load_lin(vl[i].fc1, p + "mlp.fc1", VMLP, VD);
    load_lin(vl[i].fc2, p + "mlp.fc2", VD, VMLP);
  }
  // ---- FPN (3 단계만: 0.5 배 단계는 결과에 안 쓰임) ----
  for (int lv = 0; lv < 3; ++lv) {
    std::string p = "detector_model.vision_encoder.neck.fpn_layers." + std::to_string(lv) + ".";
    int cin = VD;
    std::vector<int> cts = lv == 0 ? std::vector<int>{0, 2} : lv == 1 ? std::vector<int>{0} : std::vector<int>{};
    fpn[lv].ct.resize(cts.size());
    for (size_t j = 0; j < cts.size(); ++j) {
      int cout = cin / 2;
      const auto& w = st_.get(p + "scale_layers." + std::to_string(cts[j]) + ".weight", {cin, cout, 2, 2});
      std::vector<float> t((size_t)cout * 4 * cin);
      for (int ci = 0; ci < cin; ++ci)
        for (int co = 0; co < cout; ++co)
          for (int e = 0; e < 4; ++e) t[((size_t)co * 4 + e) * cin + ci] = w.f32()[((size_t)ci * cout + co) * 4 + e];
      auto& l = fpn[lv].ct[j];
      l.out = cout * 4;
      l.in = cin;
      upload_T(l.w, t);
      upload_f(l.b, st_.get(p + "scale_layers." + std::to_string(cts[j]) + ".bias", {cout}).f32(), cout);
      cin = cout;
    }
    load_lin(fpn[lv].p1, p + "proj1", D, cin);
    const auto& w2 = st_.get(p + "proj2.weight", {D, D, 3, 3});
    fpn[lv].p2.resize(9);
    for (int kk = 0; kk < 9; ++kk) {
      std::vector<float> t((size_t)D * D);
      for (int co = 0; co < D; ++co)
        for (int ci = 0; ci < D; ++ci) t[(size_t)co * D + ci] = w2.f32()[((size_t)co * D + ci) * 9 + kk];
      fpn[lv].p2[kk].out = D;
      fpn[lv].p2[kk].in = D;
      upload_T(fpn[lv].p2[kk].w, t);
    }
    upload_f(fpn[lv].p2[0].b, st_.get(p + "proj2.bias", {D}).f32(), D);
  }

  // ---- CLIP 텍스트 ----
  const std::string TT = "detector_model.text_encoder.text_model.";
  upload_f(tok_emb, st_.get(TT + "embeddings.token_embedding.weight", {VOCAB, TD}).f32(), (size_t)VOCAB * TD);
  upload_f(tpos_emb, st_.get(TT + "embeddings.position_embedding.weight", {TLEN, TD}).f32(), (size_t)TLEN * TD);
  tl.resize(TLAYERS);
  for (int i = 0; i < TLAYERS; ++i) {
    std::string p = TT + "encoder.layers." + std::to_string(i) + ".";
    load_norm(tl[i].ln1, p + "layer_norm1", TD);
    load_norm(tl[i].ln2, p + "layer_norm2", TD);
    load_lin_cat(tl[i].qkv, {p + "self_attn.q_proj", p + "self_attn.k_proj", p + "self_attn.v_proj"}, TD, TD);
    load_lin(tl[i].o, p + "self_attn.out_proj", TD, TD);
    load_lin(tl[i].fc1, p + "mlp.fc1", TMLP, TD);
    load_lin(tl[i].fc2, p + "mlp.fc2", TD, TMLP);
  }
  load_norm(tln_final, TT + "final_layer_norm", TD);
  load_lin(tproj, "detector_model.text_projection", D, TD);

  // ---- DETR 인코더 ----
  el.resize(ELAYERS);
  for (int i = 0; i < ELAYERS; ++i) {
    std::string p = "detector_model.detr_encoder.layers." + std::to_string(i) + ".";
    load_norm(el[i].ln1, p + "layer_norm1", D);
    load_norm(el[i].ln2, p + "layer_norm2", D);
    load_norm(el[i].ln3, p + "layer_norm3", D);
    load_lin_cat(el[i].qk, {p + "self_attn.q_proj", p + "self_attn.k_proj"}, D, D);
    load_lin(el[i].v, p + "self_attn.v_proj", D, D);
    load_lin(el[i].so, p + "self_attn.o_proj", D, D);
    for (auto [l, n] : {std::pair{&el[i].ca.q, "q"}, {&el[i].ca.k, "k"}, {&el[i].ca.v, "v"}, {&el[i].ca.o, "o"}})
      load_lin(*l, p + "cross_attn." + n + "_proj", D, D);
    load_lin(el[i].fc1, p + "mlp.fc1", MLP, D);
    load_lin(el[i].fc2, p + "mlp.fc2", D, MLP);
  }
  // ---- DETR 디코더 ----
  const std::string DD = "detector_model.detr_decoder.";
  dl.resize(DLAYERS);
  for (int i = 0; i < DLAYERS; ++i) {
    std::string p = DD + "layers." + std::to_string(i) + ".";
    auto& L = dl[i];
    load_lin_cat(L.sqk, {p + "self_attn.q_proj", p + "self_attn.k_proj"}, D, D);
    load_lin(L.sv, p + "self_attn.v_proj", D, D);
    load_lin(L.so, p + "self_attn.o_proj", D, D);
    load_norm(L.sln, p + "self_attn_layer_norm", D);
    for (auto [a, n] : {std::pair{&L.ta, "text_cross_attn"}, {&L.va, "vision_cross_attn"}}) {
      load_lin(a->q, p + n + ".q_proj", D, D);
      load_lin(a->k, p + n + ".k_proj", D, D);
      load_lin(a->v, p + n + ".v_proj", D, D);
      load_lin(a->o, p + n + ".o_proj", D, D);
    }
    load_norm(L.tln, p + "text_cross_attn_layer_norm", D);
    load_norm(L.vln, p + "vision_cross_attn_layer_norm", D);
    load_lin(L.fc1, p + "mlp.fc1", MLP, D);
    load_lin(L.fc2, p + "mlp.fc2", D, MLP);
    load_norm(L.mln, p + "mlp_layer_norm", D);
  }
  load_norm(d_out_ln, DD + "output_layer_norm", D);
  load_norm(d_pres_ln, DD + "presence_layer_norm", D);
  load_lin(box1, DD + "box_head.layer1", D, D);
  load_lin(box2, DD + "box_head.layer2", D, D);
  load_lin(box3, DD + "box_head.layer3", 4, D);
  load_lin(pres1, DD + "presence_head.layer1", D, D);
  load_lin(pres2, DD + "presence_head.layer2", D, D);
  load_lin(pres3, DD + "presence_head.layer3", 1, D);
  load_lin(ref1, DD + "ref_point_head.layer1", D, 2 * D);
  load_lin(ref2, DD + "ref_point_head.layer2", D, D);
  for (int ax = 0; ax < 2; ++ax) {
    std::string p = DD + (ax == 0 ? "box_rpb_embed_x." : "box_rpb_embed_y.");
    std::vector<float> w;
    for (auto [n, sh] : {std::pair{"layer1.weight", std::vector<int64_t>{D, 2}}, {"layer1.bias", {D}},
                         {"layer2.weight", {HEADS, D}}, {"layer2.bias", {HEADS}}}) {
      const auto& t = st_.get(p + n, sh);
      w.insert(w.end(), t.f32(), t.f32() + t.numel());
    }
    upload_f(ax == 0 ? rpbw_x : rpbw_y, w.data(), w.size());
  }
  upload_f(query_embed, st_.get(DD + "query_embed.weight", {NQ, D}).f32(), NQ * D);
  upload_f(ref_points, st_.get(DD + "reference_points.weight", {NQ, 4}).f32(), NQ * 4);
  upload_f(pres_token, st_.get(DD + "presence_token.weight", {1, D}).f32(), D);
  // ---- 점수 ----
  const std::string S = "detector_model.dot_product_scoring.";
  load_lin(sc_mlp1, S + "text_mlp.layer1", MLP, D);
  load_lin(sc_mlp2, S + "text_mlp.layer2", D, MLP);
  load_norm(sc_ln, S + "text_mlp_out_norm", D);
  load_lin(sc_text, S + "text_proj", D, D);
  load_lin(sc_query, S + "query_proj", D, D);
  // ---- 마스크 ----
  const std::string M = "detector_model.mask_decoder.";
  load_lin(m_ca.q, M + "prompt_cross_attn.q_proj", D, D);
  load_lin(m_ca.k, M + "prompt_cross_attn.k_proj", D, D);
  load_lin(m_ca.v, M + "prompt_cross_attn.v_proj", D, D);
  load_lin(m_ca.o, M + "prompt_cross_attn.o_proj", D, D);
  load_norm(m_ca_ln, M + "prompt_cross_attn_norm", D);
  for (int j = 0; j < 2; ++j) {
    std::string p = M + "pixel_decoder.conv_layers." + std::to_string(j);
    const auto& w = st_.get(p + ".weight", {D, D, 3, 3});
    for (int kk = 0; kk < 9; ++kk) {
      std::vector<float> t((size_t)D * D);
      for (int co = 0; co < D; ++co)
        for (int ci = 0; ci < D; ++ci) t[(size_t)co * D + ci] = w.f32()[((size_t)co * D + ci) * 9 + kk];
      m_conv[j][kk].out = D;
      m_conv[j][kk].in = D;
      upload_T(m_conv[j][kk].w, t);
    }
    upload_f(m_conv[j][0].b, st_.get(p + ".bias", {D}).f32(), D);
    load_norm(m_gn[j], M + "pixel_decoder.norms." + std::to_string(j), D);
  }
  load_lin(m_inst, M + "instance_projection", D, D);
  for (int j = 0; j < 3; ++j) load_lin(m_emb[j], M + "mask_embedder.layers." + std::to_string(j), D, D);

  // ---- 위치 ----
  {
    // ViT 는 처음부터 끝까지 창 순서(창 9 개 × 576)로 둔다. 토큰 t → 원래 (y, x)
    std::vector<float> pw(NTOK * 2), pg(NTOK * 2);
    float sc = (float)(24.0 / 72.0);
    for (int t = 0; t < NTOK; ++t) {
      int win = t / 576, r = t % 576, wy = win / 3, wx = win % 3, iy = r / 24, ix = r % 24;
      int y = wy * 24 + iy, x = wx * 24 + ix;
      pw[t * 2] = (float)ix * 1.0f;
      pw[t * 2 + 1] = (float)iy * 1.0f;
      pg[t * 2] = (float)x * sc;
      pg[t * 2 + 1] = (float)y * sc;
    }
    upload_f(rope_win, pw.data(), pw.size());
    upload_f(rope_glob, pg.data(), pg.size());
    // 72×72 사인 위치(정규화, 128 + 128): 채널 0..127 = y, 128..255 = x
    std::vector<float> pe((size_t)NTOK * D);
    float scale = 6.2831855f, eps = 1e-6f;
    for (int y = 0; y < GRID; ++y)
      for (int x = 0; x < GRID; ++x)
        for (int c = 0; c < D; ++c) {
          int j = c % 128;
          float e = c < 128 ? (float)(y + 1) : (float)(x + 1);
          e = e / ((float)GRID + eps) * scale;
          float dt = std::pow(10000.f, (float)(2 * (j / 2)) / 128.f);
          float v = e / dt;
          pe[((size_t)y * GRID + x) * D + c] = (j % 2 == 0) ? std::sin(v) : std::cos(v);
        }
    upload_f(pos2, pe.data(), pe.size());
  }

  // ---- 작업 버퍼 ----
  size_t P = maxP_, R = P * NTOK;
  size_t big = std::max({(size_t)NTOK * VMLP, R * MLP, P * 290 * 290 * D, (size_t)L1 * L1 * 2 * VD});
  size_t mid = std::max({(size_t)NTOK * VD, R * D, (size_t)L0 * L0 * D, P * L0 * L0 * D});
  size_t sml = std::max({(size_t)NTOK * VD, R * D, (size_t)L0 * L0 * D});
  tA.alloc(big);
  tB.alloc(mid);
  tC.alloc(sml);
  tD.alloc(std::max(sml, R * 2 * D));
  tE.alloc(sml);
  fA.alloc(std::max({big, P * L0 * 290 * D, (size_t)L1 * L1 * 4 * D}));
  fB.alloc(std::max({(size_t)NTOK * 3 * VD, R * 2 * D}));
  fC.alloc(std::max({(size_t)NTOK * VD, R * D, (size_t)L0 * L0 * D}));
  vit.alloc((size_t)NTOK * VD);
  fpn0.alloc((size_t)L0 * L0 * D);
  fpn1.alloc((size_t)L1 * L1 * D);
  fpn2.alloc((size_t)NTOK * D);
  text_hidden.alloc(P * TLEN * TD);
  text_feat.alloc(P * TLEN * D);
  text_t.alloc(P * TLEN * D);
  tkv.alloc((size_t)(ELAYERS + DLAYERS + 1) * 2 * P * TLEN * D);
  tmask.alloc(P * TLEN);
  proj_text.alloc(P * D);
  enc.alloc(R * D);
  enc_t.alloc(R * D);
  encpos_t.alloc(R * D);
  hdec.alloc(P * (NQ + 1) * D);
  qpos.alloc(P * (NQ + 1) * D);
  refb.alloc(P * NQ * 4);
  rpby.alloc(P * (NQ + 1) * GRID * HEADS);
  rpbx.alloc(P * (NQ + 1) * GRID * HEADS);
  CK(cudaMemsetAsync(rpby.p, 0, rpby.n * 4, s_));
  CK(cudaMemsetAsync(rpbx.p, 0, rpbx.n * 4, s_));
  logits.alloc(P * NQ);
  boxes.alloc(P * NQ * 4);
  presence.alloc(P);
  pix.alloc(P * L0 * L0 * D);
  inst_t.alloc(P * L0 * L0 * D);
  memb_t.alloc(P * (NQ + 1) * D);
  CK(cudaStreamSynchronize(s_));
}

template <typename T>
Model<T>::~Model() {
  lt_.destroy();
  cublasDestroy(h_);
}

// ================= 공통 =================
template <typename T>
void Model<T>::lin(const Lin<T>& l, const T* x, int rows, float* y, int ldx, int ldy) {
  k::linear<T>(h_, x, ldx < 0 ? l.in : ldx, l.w.p, l.in, y, ldy < 0 ? l.out : ldy, rows, l.out, l.in);
}
template <typename T>
void Model<T>::proj(const Lin<T>& l, const T* x, int rows, T* y, int act, int ldx, int ldy) {
  if (ldx < 0) ldx = l.in;
  if (ldy < 0) ldy = l.out;
  if constexpr (kFast) {
    k::lt_linear<T, T>(lt_, x, ldx, l.w.p, l.in, y, ldy, rows, l.out, l.in, l.bh.p, act == 2 ? 2 : 0, 0.f);
    if (act == 1) {
      if (ldy != l.out) throw std::runtime_error("GELU 는 꽉 찬 출력만");
      k::gelu_inplace<T>(y, (long long)rows * l.out, s_);
    }
  } else {
    k::linear<T>(h_, x, ldx, l.w.p, l.in, fA.p, l.out, rows, l.out, l.in);
    k::bias_act<T>(fA.p, l.out, l.b.p, y, ldy, rows, l.out, act, s_);
  }
}
template <typename T>
void Model<T>::proj_res(const Lin<T>& l, const T* x, int rows, float* r, int ldx, int ldr) {
  if (ldx < 0) ldx = l.in;
  if (ldr < 0) ldr = l.out;
  if constexpr (kFast) {
    k::lt_linear<T, float>(lt_, x, ldx, l.w.p, l.in, r, ldr, rows, l.out, l.in, l.b.p, 0, 1.f);
  } else {
    k::linear<T>(h_, x, ldx, l.w.p, l.in, fC.p, l.out, rows, l.out, l.in);
    k::bias_residual(fC.p, l.out, l.b.p, r, ldr, rows, l.out, s_);
  }
}

// NHWC 3×3 합성곱(덧대기 1): 덧댄 영상의 행을 어긋나게 읽는 GEMM 9 번을 더한다
template <typename T>
void Model<T>::conv3x3(const Lin<T>* w9, const T* x, int B, int h, int w, int C, const float* bias, float* out) {
  long long pimg = (long long)(h + 2) * (w + 2) * C, oimg = (long long)h * (w + 2) * w9[0].out;
  for (int b = 0; b < B; ++b) k::pad1<T>(x + (long long)b * h * w * C, tA.p + b * pimg, h, w, C, s_);
  int M = h * (w + 2) - 2;
  for (int kk = 0; kk < 9; ++kk) {
    int ky = kk / 3, kx = kk % 3;
    k::linear_batched<T>(h_, tA.p + (long long)(ky * (w + 2) + kx) * C, C, pimg, w9[kk].w.p, C, 0, fA.p, w9[0].out,
                         oimg, M, w9[0].out, C, B, kk == 0 ? 0.f : 1.f);
  }
  for (int b = 0; b < B; ++b)
    k::crop_bias(fA.p + b * oimg, bias, out + (long long)b * h * w * w9[0].out, h, w, w9[0].out, s_);
}

// ================= 텍스트 =================
template <typename T>
void Model<T>::encode_text(const int* ids, const int* mask, int P_) {
  P = P_;
  if (P > maxP_) throw std::runtime_error("프롬프트가 너무 많음");
  int rows = P * TLEN;
  Buf<int> dids;
  dids.alloc(rows);
  CK(cudaMemcpyAsync(dids.p, ids, rows * 4, cudaMemcpyHostToDevice, s_));
  CK(cudaMemcpyAsync(tmask.p, mask, rows * 4, cudaMemcpyHostToDevice, s_));
  float* X = text_hidden.p;
  embed_text_k<<<nb((long long)rows * TD), 256, 0, s_>>>(dids.p, tok_emb.p, tpos_emb.p, X, rows, TD, TLEN);
  for (int i = 0; i < TLAYERS; ++i) {
    auto& L = tl[i];
    k::layernorm<T>(X, TD, L.ln1.g, L.ln1.b, 1e-5f, tB.p, TD, rows, TD, s_);
    proj(L.qkv, tB.p, rows, tC.p);  // [rows, 3072] = q | k | v
    k::Attn<T> a{tC.p, tC.p + TD, tC.p + 2 * TD, tB.p, P, THEADS, TLEN, TLEN, TD / THEADS,
                 (long long)TLEN * 3 * TD, 3 * TD, (long long)TLEN * 3 * TD, 3 * TD, (long long)TLEN * 3 * TD, 3 * TD,
                 (long long)TLEN * TD, TD};
    a.kmask = tmask.p;
    a.kmb = TLEN;
    a.causal = true;
    a.scale = 0.125f;
    k::attention<T>(a, s_);
    proj_res(L.o, tB.p, rows, X);
    k::layernorm<T>(X, TD, L.ln2.g, L.ln2.b, 1e-5f, tB.p, TD, rows, TD, s_);
    proj(L.fc1, tB.p, rows, tA.p, 1);
    proj_res(L.fc2, tA.p, rows, X);
  }
  k::layernorm<float>(X, TD, tln_final.g, tln_final.b, 1e-5f, X, TD, rows, TD, s_);
  k::cast<T>(X, tB.p, (long long)rows * TD, s_);
  lin(tproj, tB.p, rows, fC.p);
  k::bias_act<float>(fC.p, D, tproj.b.p, text_feat.p, D, rows, D, 0, s_);
  k::cast<T>(text_feat.p, text_t.p, (long long)rows * D, s_);

  // 텍스트 쪽 k, v 를 층마다 미리 만든다: 인코더 6, 디코더 6, 마스크 1
  size_t slot = (size_t)P * TLEN * D;
  int si = 0;
  auto kv = [&](const Attn4<T>& a) {
    proj(a.k, text_t.p, rows, tkv.p + (si++) * slot);
    proj(a.v, text_t.p, rows, tkv.p + (si++) * slot);
  };
  for (auto& L : el) kv(L.ca);
  for (auto& L : dl) kv(L.ta);
  kv(m_ca);

  // 점수용 텍스트 벡터: LN(mlp(t) + t) 를 유효 토큰 평균 → text_proj
  proj(sc_mlp1, text_t.p, rows, tA.p, 2);
  Buf<float> tf;
  tf.alloc((size_t)rows * D);
  CK(cudaMemcpyAsync(tf.p, text_feat.p, (size_t)rows * D * 4, cudaMemcpyDeviceToDevice, s_));
  proj_res(sc_mlp2, tA.p, rows, tf.p);
  k::layernorm<float>(tf.p, D, sc_ln.g, sc_ln.b, 1e-5f, tf.p, D, rows, D, s_);
  std::vector<float> h((size_t)rows * D), pooled((size_t)P * D, 0.f);
  CK(cudaMemcpyAsync(h.data(), tf.p, h.size() * 4, cudaMemcpyDeviceToHost, s_));
  CK(cudaStreamSynchronize(s_));
  for (int b = 0; b < P; ++b) {
    float n = 0.f;
    for (int t = 0; t < TLEN; ++t) n += mask[b * TLEN + t] ? 1.f : 0.f;
    for (int c = 0; c < D; ++c) {
      float s = 0.f;
      for (int t = 0; t < TLEN; ++t) s += h[((size_t)b * TLEN + t) * D + c] * (mask[b * TLEN + t] ? 1.f : 0.f);
      pooled[(size_t)b * D + c] = s / std::max(n, 1.f);
    }
  }
  Buf<float> pd;
  upload_f(pd, pooled.data(), pooled.size());
  k::cast<T>(pd.p, tB.p, (long long)P * D, s_);
  lin(sc_text, tB.p, P, fC.p);
  k::bias_act<float>(fC.p, D, sc_text.b.p, proj_text.p, D, P, D, 0, s_);
  CK(cudaStreamSynchronize(s_));
}

// ================= 영상 =================
// ViT 토큰은 창 순서. 창 층은 576 개씩 9 묶음, 전역 층은 5184 개 하나(위치만 창 순서에 맞춘 RoPE)
template <typename T>
void Model<T>::vit_layer(int i, const float* rope_pos, bool windowed) {
  auto& L = vl[i];
  float* X = fB.p;  // 창 순서 잔차
  k::layernorm<T>(X, VD, L.ln1.g, L.ln1.b, 1e-6f, tB.p, VD, NTOK, VD, s_);
  lin(L.qkv, tB.p, NTOK, fA.p);
  k::qkv_rope<T>(fA.p, L.qkv.b.p, rope_pos, NTOK, VD, VHEADS, 10000.f, tD.p, tE.p, tC.p, s_);
  int nseq = windowed ? (GRID / WIN) * (GRID / WIN) : 1, len = NTOK / nseq;
  k::Attn<T> a{tD.p, tE.p, tC.p, tB.p, nseq, VHEADS, len, len, VD / VHEADS,
               (long long)len * VD, VD, (long long)len * VD, VD, (long long)len * VD, VD, (long long)len * VD, VD};
  a.scale = 0.125f;
  k::attention<T>(a, s_);
  proj_res(L.o, tB.p, NTOK, X);
  k::layernorm<T>(X, VD, L.ln2.g, L.ln2.b, 1e-6f, tB.p, VD, NTOK, VD, s_);
  proj(L.fc1, tB.p, NTOK, tA.p, 1);
  proj_res(L.fc2, tA.p, NTOK, X);
}

template <typename T>
void Model<T>::encode_image(const float* pixels) {
  float* X = fB.p;
  k::patchify<T>(pixels, tA.p, IMG, IMG, PATCH, s_);
  lin(patch, tA.p, NTOK, vit.p);
  k::add_tiled_pos(vit.p, vpos.p, GRID, GRID, 24, 24, VD, s_);
  k::window_gather<float>(vit.p, X, GRID, GRID, VD, WIN, s_);
  k::layernorm<float>(X, VD, vln_pre.g, vln_pre.b, 1e-6f, X, VD, NTOK, VD, s_);
  for (int i = 0; i < VLAYERS; ++i) {
    bool glob = (i == 7 || i == 15 || i == 23 || i == 31);
    vit_layer(i, glob ? rope_glob.p : rope_win.p, !glob);
    if (debug && (i == 0 || i == 6 || i == 7)) {
      int j = i == 0 ? 0 : i == 6 ? 1 : 2;
      k::window_scatter(X, vit.p, GRID, GRID, VD, WIN, s_);
      vit_dbg[j].resize((size_t)NTOK * VD);
      CK(cudaMemcpyAsync(vit_dbg[j].data(), vit.p, (size_t)NTOK * VD * 4, cudaMemcpyDeviceToHost, s_));
    }
  }
  k::window_scatter(X, vit.p, GRID, GRID, VD, WIN, s_);
  // ---- FPN ----
  k::cast<T>(vit.p, tB.p, (long long)NTOK * VD, s_);
  // 1 배
  proj(fpn[2].p1, tB.p, NTOK, tC.p);
  conv3x3(fpn[2].p2.data(), tC.p, 1, GRID, GRID, D, fpn[2].p2[0].b.p, fpn2.p);
  // 2 배
  lin(fpn[1].ct[0], tB.p, NTOK, fA.p);
  k::convt2_shuffle<T>(fA.p, fpn[1].ct[0].b.p, tC.p, GRID, GRID, VD / 2, 0, s_);
  proj(fpn[1].p1, tC.p, L1 * L1, tD.p);
  conv3x3(fpn[1].p2.data(), tD.p, 1, L1, L1, D, fpn[1].p2[0].b.p, fpn1.p);
  // 4 배
  lin(fpn[0].ct[0], tB.p, NTOK, fA.p);
  k::convt2_shuffle<T>(fA.p, fpn[0].ct[0].b.p, tC.p, GRID, GRID, VD / 2, 1, s_);
  lin(fpn[0].ct[1], tC.p, L1 * L1, fA.p);
  k::convt2_shuffle<T>(fA.p, fpn[0].ct[1].b.p, tD.p, L1, L1, VD / 4, 0, s_);
  proj(fpn[0].p1, tD.p, L0 * L0, tE.p);
  conv3x3(fpn[0].p2.data(), tE.p, 1, L0, L0, D, fpn[0].p2[0].b.p, fpn0.p);
}

// ================= 검출 =================
template <typename T>
void Model<T>::detect() {
  long long R = (long long)P * NTOK;
  size_t slot = (size_t)P * TLEN * D;
  for (int b = 0; b < P; ++b)
    CK(cudaMemcpyAsync(enc.p + (size_t)b * NTOK * D, fpn2.p, (size_t)NTOK * D * 4, cudaMemcpyDeviceToDevice, s_));
  const float sc = 1.f / std::sqrt((float)(D / HEADS));
  for (int i = 0; i < ELAYERS; ++i) {
    auto& L = el[i];
    // 자기 어텐션: q, k 는 LN + 위치, v 는 LN
    k::layernorm_pos<T>(enc.p, L.ln1.g, L.ln1.b, 1e-5f, pos2.p, NTOK, tB.p, tC.p, (int)R, D, s_);
    proj(L.qk, tC.p, (int)R, tD.p);  // [R, 512] = q | k
    proj(L.v, tB.p, (int)R, tE.p);
    {
      k::Attn<T> a{tD.p, tD.p + D, tE.p, tC.p, P, HEADS, NTOK, NTOK, D / HEADS, (long long)NTOK * 2 * D, 2 * D,
                   (long long)NTOK * 2 * D, 2 * D, (long long)NTOK * D, D, (long long)NTOK * D, D};
      a.scale = sc;
      k::attention<T>(a, s_);
    }
    proj_res(L.so, tC.p, (int)R, enc.p);
    // 텍스트 교차
    k::layernorm<T>(enc.p, D, L.ln2.g, L.ln2.b, 1e-5f, tB.p, D, (int)R, D, s_);
    proj(L.ca.q, tB.p, (int)R, tC.p);
    {
      k::Attn<T> a{tC.p, tkv.p + (2 * i) * slot, tkv.p + (2 * i + 1) * slot, tD.p, P, HEADS, NTOK, TLEN, D / HEADS,
                   (long long)NTOK * D, D, (long long)TLEN * D, D, (long long)TLEN * D, D, (long long)NTOK * D, D};
      a.kmask = tmask.p;
      a.kmb = TLEN;
      a.scale = sc;
      k::attention<T>(a, s_);
    }
    proj_res(L.ca.o, tD.p, (int)R, enc.p);
    // MLP
    k::layernorm<T>(enc.p, D, L.ln3.g, L.ln3.b, 1e-5f, tB.p, D, (int)R, D, s_);
    proj(L.fc1, tB.p, (int)R, tA.p, 2);
    proj_res(L.fc2, tA.p, (int)R, enc.p);
  }

  // ---- 디코더 ----
  k::cast<T>(enc.p, enc_t.p, R * D, s_);
  add_bcast_k<T><<<nb(R * D), 256, 0, s_>>>(enc.p, pos2.p, encpos_t.p, R, D, NTOK);
  int RD = P * (NQ + 1);
  long long qb = (long long)(NQ + 1) * D;
  init_dec_k<<<nb((long long)RD * D), 256, 0, s_>>>(hdec.p, pres_token.p, query_embed.p, refb.p, ref_points.p, P);
  CK(cudaMemsetAsync(qpos.p, 0, qpos.n * 4, s_));
  Buf<T>& oln = memb_t;  // 마지막 층 출력 LN(질의) — 마스크 단계에서 다시 쓴다
  for (int i = 0; i < DLAYERS; ++i) {
    auto& L = dl[i];
    // 질의 위치: 상자 사인 → ref_point_head
    box_sine_k<T><<<nb((long long)P * NQ * 512), 256, 0, s_>>>(refb.p, tA.p, P * NQ);
    proj(ref1, tA.p, P * NQ, tB.p, 2);
    lin(ref2, tB.p, P * NQ, fA.p);
    shift_rows_k<float><<<nb((long long)P * NQ * D), 256, 0, s_>>>(fA.p, D, ref2.b.p, qpos.p, D, (long long)P * NQ, D,
                                                                     NQ, 1, 0);
    // BoxRPB
    rpb_k<<<nb((long long)P * NQ * GRID, 128), 128, 0, s_>>>(refb.p, rpbw_x.p, rpbx.p, P * NQ, GRID, 0);
    rpb_k<<<nb((long long)P * NQ * GRID, 128), 128, 0, s_>>>(refb.p, rpbw_y.p, rpby.p, P * NQ, GRID, 1);
    // 자기 어텐션
    k::add_to<T>(hdec.p, qpos.p, tB.p, (long long)RD * D, s_);
    k::cast<T>(hdec.p, tC.p, (long long)RD * D, s_);
    proj(L.sqk, tB.p, RD, tD.p);  // q | k
    proj(L.sv, tC.p, RD, tE.p);
    {
      k::Attn<T> a{tD.p, tD.p + D, tE.p, tB.p, P, HEADS, NQ + 1, NQ + 1, D / HEADS, (long long)(NQ + 1) * 2 * D,
                   2 * D, (long long)(NQ + 1) * 2 * D, 2 * D, qb, D, qb, D};
      a.scale = sc;
      k::attention<T>(a, s_);
    }
    proj_res(L.so, tB.p, RD, hdec.p);
    k::layernorm<float>(hdec.p, D, L.sln.g, L.sln.b, 1e-5f, hdec.p, D, RD, D, s_);
    // 텍스트 교차
    k::add_to<T>(hdec.p, qpos.p, tB.p, (long long)RD * D, s_);
    proj(L.ta.q, tB.p, RD, tC.p);
    {
      k::Attn<T> a{tC.p, tkv.p + (2 * (ELAYERS + i)) * slot, tkv.p + (2 * (ELAYERS + i) + 1) * slot, tD.p, P, HEADS,
                   NQ + 1, TLEN, D / HEADS, qb, D, (long long)TLEN * D, D, (long long)TLEN * D, D, qb, D};
      a.kmask = tmask.p;
      a.kmb = TLEN;
      a.scale = sc;
      k::attention<T>(a, s_);
    }
    proj_res(L.ta.o, tD.p, RD, hdec.p);
    k::layernorm<float>(hdec.p, D, L.tln.g, L.tln.b, 1e-5f, hdec.p, D, RD, D, s_);
    // 영상 교차(BoxRPB 편향)
    k::add_to<T>(hdec.p, qpos.p, tB.p, (long long)RD * D, s_);
    proj(L.va.q, tB.p, RD, tC.p);
    proj(L.va.k, encpos_t.p, (int)R, tD.p);
    proj(L.va.v, enc_t.p, (int)R, tE.p);
    {
      k::Attn<T> a{tC.p, tD.p, tE.p, tB.p, P, HEADS, NQ + 1, NTOK, D / HEADS, qb, D, (long long)NTOK * D, D,
                   (long long)NTOK * D, D, qb, D};
      a.by = rpby.p;
      a.bx = rpbx.p;
      a.bh = GRID;
      a.bw = GRID;
      a.scale = sc;
      k::attention<T>(a, s_);
    }
    proj_res(L.va.o, tB.p, RD, hdec.p);
    k::layernorm<float>(hdec.p, D, L.vln.g, L.vln.b, 1e-5f, hdec.p, D, RD, D, s_);
    // MLP
    k::cast<T>(hdec.p, tB.p, (long long)RD * D, s_);
    proj(L.fc1, tB.p, RD, tA.p, 2);
    proj_res(L.fc2, tA.p, RD, hdec.p);
    k::layernorm<float>(hdec.p, D, L.mln.g, L.mln.b, 1e-5f, hdec.p, D, RD, D, s_);
    // 상자 다듬기
    k::layernorm<T>(hdec.p, D, d_out_ln.g, d_out_ln.b, 1e-5f, oln.p, D, RD, D, s_);
    proj(box1, oln.p, RD, tB.p, 2);
    proj(box2, tB.p, RD, tC.p, 2);
    lin(box3, tC.p, RD, fA.p);
    box_refine_k<<<nb((long long)P * NQ * 4), 256, 0, s_>>>(fA.p, box3.b.p, refb.p, P);
  }
  xyxy_k<<<nb(P * NQ), 256, 0, s_>>>(refb.p, boxes.p, P * NQ);
  // 존재 점수: 마지막 층 존재 토큰
  k::layernorm<T>(hdec.p, (NQ + 1) * D, d_pres_ln.g, d_pres_ln.b, 1e-5f, tB.p, D, P, D, s_);
  proj(pres1, tB.p, P, tC.p, 2);
  proj(pres2, tC.p, P, tB.p, 2);
  lin(pres3, tB.p, P, fA.p);
  presence_k<<<1, 64, 0, s_>>>(fA.p, pres3.b.p, presence.p, P);
  // 질의 점수
  lin(sc_query, oln.p, RD, fA.p);
  score_k<<<nb(P * NQ), 256, 0, s_>>>(fA.p, sc_query.b.p, proj_text.p, logits.p, P);
}

// ================= 마스크 =================
// 프롬프트 [b0, b0+nb) 의 픽셀 임베딩 → inst_t [nb, 288², 256]
template <typename T>
void Model<T>::mask_pixels(int b0, int nbp) {
  long long R = (long long)nbp * NTOK, npx = (long long)L0 * L0;
  size_t slot = (size_t)P * TLEN * D;
  int si = 2 * (ELAYERS + DLAYERS);
  const float* encb = enc.p + (size_t)b0 * NTOK * D;
  k::layernorm<T>(encb, D, m_ca_ln.g, m_ca_ln.b, 1e-5f, tB.p, D, (int)R, D, s_);
  proj(m_ca.q, tB.p, (int)R, tC.p);
  k::Attn<T> a{tC.p, tkv.p + si * slot + (size_t)b0 * TLEN * D, tkv.p + (si + 1) * slot + (size_t)b0 * TLEN * D, tD.p,
               nbp, HEADS, NTOK, TLEN, D / HEADS, (long long)NTOK * D, D, (long long)TLEN * D, D, (long long)TLEN * D,
               D, (long long)NTOK * D, D};
  a.kmask = tmask.p + b0 * TLEN;
  a.kmb = TLEN;
  a.scale = 1.f / std::sqrt((float)(D / HEADS));
  k::attention<T>(a, s_);
  // enc2 는 pix 에 잠깐 둔다(fC 는 float 경로 GEMM 임시 버퍼)
  CK(cudaMemcpyAsync(pix.p, encb, R * D * 4, cudaMemcpyDeviceToDevice, s_));
  proj_res(m_ca.o, tD.p, (int)R, pix.p);
  // 픽셀 디코더: 72 → 144 (+fpn1) → 288 (+fpn0)
  k::upsample2_add<T>(pix.p, fpn1.p, tB.p, nbp, GRID, GRID, D, s_);
  conv3x3(m_conv[0], tB.p, nbp, L1, L1, D, m_conv[0][0].b.p, pix.p);
  k::groupnorm_relu(pix.p, nbp, L1 * L1, D, 8, m_gn[0].g, m_gn[0].b, 1e-5f, s_);
  k::upsample2_add<T>(pix.p, fpn0.p, tB.p, nbp, L1, L1, D, s_);
  conv3x3(m_conv[1], tB.p, nbp, L0, L0, D, m_conv[1][0].b.p, pix.p);
  k::groupnorm_relu(pix.p, nbp, (int)npx, D, 8, m_gn[1].g, m_gn[1].b, 1e-5f, s_);
  k::cast<T>(pix.p, tB.p, nbp * npx * D, s_);
  proj(m_inst, tB.p, (int)(nbp * npx), inst_t.p);
}
// 질의 임베딩: memb_t(마지막 층 LN 출력) → MLP 3 층 → tE [P*201, 256]
template <typename T>
void Model<T>::mask_queries() {
  int RD = P * (NQ + 1);
  proj(m_emb[0], memb_t.p, RD, tC.p, 2);
  proj(m_emb[1], tC.p, RD, tD.p, 2);
  proj(m_emb[2], tD.p, RD, tE.p);
}

template <typename T>
void Model<T>::masks_all() {
  long long npx = (long long)L0 * L0;
  mask_pixels(0, P);
  mask_queries();
  masks.alloc((size_t)P * NQ * npx);
  for (int b = 0; b < P; ++b)
    k::linear<T>(h_, tE.p + ((size_t)b * (NQ + 1) + 1) * D, D, inst_t.p + b * npx * D, D,
                 masks.p + (size_t)b * NQ * npx, (int)npx, NQ, (int)npx, D);
}

template <typename T>
void Model<T>::masks_for(int b, const std::vector<int>& qs, float* out) {
  if (qs.empty()) return;
  long long npx = (long long)L0 * L0;
  mask_pixels(b, 1);
  mask_queries();
  std::vector<int> rows(qs.size());
  for (size_t i = 0; i < qs.size(); ++i) rows[i] = b * (NQ + 1) + 1 + qs[i];
  Buf<int> dr;
  dr.alloc(rows.size());
  CK(cudaMemcpyAsync(dr.p, rows.data(), rows.size() * 4, cudaMemcpyHostToDevice, s_));
  gather_rows_k<T><<<nb((long long)rows.size() * D), 256, 0, s_>>>(tE.p, dr.p, tC.p, (int)rows.size(), D);
  k::linear<T>(h_, tC.p, D, inst_t.p, D, out, (int)npx, (int)qs.size(), (int)npx, D);
  CK(cudaStreamSynchronize(s_));
}

template class Model<float>;
template class Model<bf16>;
template class Model<f16>;

}  // namespace sam3
