// d1-3B (lfm2) 마지막 shortconv 두 층(28·29)·출력 정규화·묶인 머리(token_embd)의 가중을 GGUF 에서 읽어 f32 로 (Q8_0 은 ggml 형 특성 to_float 로 풂).
// 층 계산은 llama.cpp src/models/lfm2.cpp (graph) · lfm-base.cpp build_lfm_shortconv · build_lfm_dense_ffn 과 같게 (d1_ref.h).
#pragma once
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

#include "ggml.h"
#include "gguf.h"

namespace d1 {
constexpr int kD = 2048, kF = 10752, kT = 5;   // 너비 · FFN 너비 · 캐시한 토큰 수 (커널 3 인과 conv 두 층 → 2·2 + 1)
struct Layer {
  std::vector<float> attn_norm, ffn_norm, conv;   // [D] · [D] · [D][3] (conv[c·3 + k], k = 0 이 가장 먼 과거)
  std::vector<float> in_proj, out_proj;           // [3D][D] (출력 줄마다 입력 D 개 — b · c · x 차례) · [D][D]
  std::vector<float> gate, up, down;              // [F][D] · [F][D] · [D][F]
};
struct Weights {
  Layer L[2];                    // 28 · 29 층
  std::vector<float> out_norm;   // token_embd_norm [D]
  float eps = 1e-5f;
};
struct Gguf {
  gguf_context* g = nullptr;
  std::FILE* f = nullptr;
  size_t base = 0;
  explicit Gguf(const std::string& path) {
    gguf_init_params p{true, nullptr};
    g = gguf_init_from_file(path.c_str(), p);
    f = std::fopen(path.c_str(), "rb");
    if (!g || !f) throw std::runtime_error("GGUF 를 못 엶: " + path);
    base = gguf_get_data_offset(g);
  }
  ~Gguf() { if (g) gguf_free(g); if (f) std::fclose(f); }
  // 텐서 이름의 줄 [r0, r0 + nr) 을 f32 로 (ne0 = 줄 길이)
  void rows(const std::string& name, int64_t ne0, int64_t r0, int64_t nr, float* out) const {
    const int64_t id = gguf_find_tensor(g, name.c_str());
    if (id < 0) throw std::runtime_error("텐서 없음: " + name);
    const ggml_type t = gguf_get_tensor_type(g, id);
    const size_t rs = ggml_row_size(t, ne0);
    std::vector<uint8_t> buf(rs * nr);
    if (std::fseek(f, (long)(base + gguf_get_tensor_offset(g, id) + rs * r0), SEEK_SET) != 0 || std::fread(buf.data(), 1, buf.size(), f) != buf.size()) throw std::runtime_error("읽기 실패: " + name);
    if (t == GGML_TYPE_F32) { std::memcpy(out, buf.data(), buf.size()); return; }
    const ggml_type_traits* tr = ggml_get_type_traits(t);
    if (!tr->to_float) throw std::runtime_error("풀 수 없는 형: " + name);
    for (int64_t r = 0; r < nr; ++r) tr->to_float(buf.data() + rs * r, out + ne0 * r, ne0);
  }
  std::vector<float> all(const std::string& name, int64_t ne0, int64_t ne1) const { std::vector<float> v((size_t)ne0 * ne1); rows(name, ne0, 0, ne1, v.data()); return v; }
};
inline Weights load(const Gguf& G) {
  Weights W;
  for (int l = 0; l < 2; ++l) {
    const std::string p = "blk." + std::to_string(28 + l) + ".";
    Layer& L = W.L[l];
    L.attn_norm = G.all(p + "attn_norm.weight", kD, 1);
    L.ffn_norm = G.all(p + "ffn_norm.weight", kD, 1);
    L.conv = G.all(p + "shortconv.conv.weight", 3, kD);
    L.in_proj = G.all(p + "shortconv.in_proj.weight", kD, 3 * kD);
    L.out_proj = G.all(p + "shortconv.out_proj.weight", kD, kD);
    L.gate = G.all(p + "ffn_gate.weight", kD, kF);
    L.up = G.all(p + "ffn_up.weight", kD, kF);
    L.down = G.all(p + "ffn_down.weight", kF, kD);
  }
  W.out_norm = G.all("token_embd_norm.weight", kD, 1);
  return W;
}
// 머리 줄 (묶인 token_embd 의 그 토큰 줄)
inline std::vector<float> head_rows(const Gguf& G, const std::vector<int>& toks) {
  std::vector<float> E((size_t)toks.size() * kD);
  for (size_t k = 0; k < toks.size(); ++k) G.rows("token_embd.weight", kD, toks[k], 1, &E[k * kD]);
  return E;
}
}  // namespace d1
