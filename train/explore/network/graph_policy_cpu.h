// 그래프 어텐션 정책(graph_net.cuh — HEADER 구조)의 CPU 정방향. CUDA 없이 현재 노드의 이웃 24 칸 로짓·가림·결정 행동을 낸다.
// 두 가지로 쓴다:
//   GraphPolicyT<float>  실물(젯슨·데스크톱 CPU) 배포용 — ppo/tools/graph_net_check 가 학습기 GPU 정방향과 로짓 차 ≤ 1e-3 인지 확인한다.
//   GraphPolicyT<double> 학습기 희소 구현의 기준 — 같은 식을 "간선 가림을 씌운 조밀 어텐션"(model.py 방식: 노드 × 노드 가림 행렬)으로 따로 짠 것.
// 계산 (graph_net.cu 와 같은 식·같은 가중치 배치 — 모든 W 는 [입력][출력], y = x·W):
//   유효 노드(특징 0 번 칸 ≠ 0)만 자리 순으로 모음 → x = 칸 1·2(dx·dy)는 (u8 − 128) / coord_scale(160 — 0 중심, 전작 large-scale agent.py:157-160), 칸 3 … fin 은 u8 / 255
//   (fin = 5: HEADER v' = (x, y, u, e, b). 그 뒤 예약 칸은 안 읽음) → 임베딩 Linear
//   인코더 층 × L: a = LN(x) → 여러 머리 어텐션(질의 i 가 키 j 를 봄 ⇔ j = i 또는 i 의 간선 비트가 가리키는 유효 이웃) → x += o·W_out
//                  → a = LN(x) → x += ReLU(a·W1 + b1)·W2 + b2
//   디코더: a = LN(x) (질의·기억 공용) → 질의 = a[현재 노드], 키·값 = 모든 유효 노드 → h = x[현재] + o·W_out → E = h + FF(LN(h))
//   정책 머리: s = [E | x[현재] | 전역]·W_ce + b_ce, 로짓_k = 10·tanh((s·W_q)·(x[이웃 k]·W_k)/√d) — 간선 없는 칸은 로짓 0·가림
//   결정 행동 = 가리지 않은 칸 중 로짓 최대(같으면 앞 칸, 없으면 0 — ppo.cu k_cat_act)
// 머리 수는 체크포인트의 변수 크기로 알 수 없어 따로 받는다(기본 8 = explore_spec_graph).
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

namespace xnet_cpu {

template <class T>
class GraphPolicyT {
 public:
  static constexpr int kNb = 24;   // 5×5 스텐실에서 자신 제외 (mdp::graph::NB)
  std::map<std::string, std::vector<T>> blk;   // 변수 블록 (이름 = 학습기 블록 이름). 바꾼 뒤엔 configure()
  int fin = 0, d = 0, heads = 8, layers = 0, ff = 0, glob = 0;
  int feat_stride = 0;   // 노드 특징 버퍼의 칸 수 (mdp::graph::FEAT). 0 = fin + 1. 신경망은 1 … fin 번 칸만 읽는다 (fin = 임베딩 가중치 크기 — HEADER 5 칸)
  int coord = 2;             // 임베딩 입력의 앞 coord 칸(F_DX·F_DY)은 (u8 − 128) / coord_scale (graph_net.cuh GraphSpec — 전작 agent.py:157-160 과 같은 0 중심 좌표)
  T coord_scale = (T)160;

  // 블록 크기에서 구조를 읽는다. prefix = 몸통 이름 ("gactor")
  void configure(int n_heads = 8, const std::string& prefix = "gactor") {
    pfx_ = prefix;
    heads = n_heads;
    d = (int)get(pfx_ + ".emb.b").size();
    fin = (int)(get(pfx_ + ".emb.w").size() / d);
    layers = 0;
    while (blk.count(pfx_ + ".enc" + std::to_string(layers) + ".ln1.g")) ++layers;
    ff = (int)get(pfx_ + ".dec.ff1.b").size();
    glob = (int)(get(pfx_ + ".cur_emb.w").size() / d) - 2 * d;
    if (d < 1 || d % heads || layers < 1 || glob < 0) throw std::runtime_error("그래프 정책: 구조(차원·머리 수)");
    need(".emb.w", (size_t)fin * d);
    for (int l = 0; l < layers; ++l) {
      const std::string e = ".enc" + std::to_string(l);
      need(e + ".wqkv", (size_t)d * 3 * d);
      need(e + ".wout", (size_t)d * d);
      need(e + ".ff1.w", (size_t)d * ff);
      need(e + ".ff2.w", (size_t)ff * d);
    }
    need(".dec.wq", (size_t)d * d);
    need(".dec.wkv", (size_t)d * 2 * d);
    need(".dec.wout", (size_t)d * d);
    need(".cur_emb.w", (size_t)(2 * d + glob) * d);
    need(".ptr.wq", (size_t)d * d);
    need(".ptr.wk", (size_t)d * d);
  }

  // 학습기 체크포인트(ppo.cu PPO::save, fp32)에서 변수 블록을 읽는다
  void load(const std::string& path, int n_heads = 8) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("체크포인트 없음: " + path);
    auto rd = [&](void* p, size_t n) {
      f.read(reinterpret_cast<char*>(p), (std::streamsize)n);
      if (!f) throw std::runtime_error("체크포인트가 잘림: " + path);
    };
    uint32_t hdr[3];
    rd(hdr, 12);
    if (hdr[0] != 0x4F505058u || hdr[1] != 3 || hdr[2] != 4) throw std::runtime_error("체크포인트 형식 3 (fp32) 이 아님: " + path);
    int32_t nb;
    rd(&nb, 4);
    std::vector<std::pair<std::string, int64_t>> names;
    for (int i = 0; i < nb; ++i) {
      int32_t ln;
      rd(&ln, 4);
      if (ln <= 0 || ln > 256) throw std::runtime_error("체크포인트 블록 이름 길이");
      std::string nm((size_t)ln, '\0');
      rd(nm.data(), (size_t)ln);
      int64_t n;
      rd(&n, 8);
      names.push_back({nm, n});
    }
    int64_t P;
    rd(&P, 8);
    std::vector<float> p((size_t)P);
    rd(p.data(), (size_t)P * 4);
    blk.clear();
    int64_t off = 0;
    for (auto& [nm, n] : names) {
      if (off + n > P) throw std::runtime_error("체크포인트 블록 크기 합 > 변수 수");
      blk[nm].assign(p.begin() + off, p.begin() + off + n);
      off += n;
    }
    if (!blk.count("gactor.emb.w")) throw std::runtime_error("그래프 어텐션 정책 체크포인트가 아님: " + path);
    configure(n_heads);
  }

  // 그래프 하나. feat u8 [gw·gh][feat_stride], edge u32 [gw·gh], cur = 현재 노드 자리(유효해야 함), g = 전역 [glob] (glob 0 이면 안 읽음).
  // logits·mask [24] (mask 1 = 고를 수 있음). E_out·Hc_out [d] 는 검사용(nullptr 가능). 반환 = 결정 행동(이웃 번호)
  int act(int gw, int gh, const uint8_t* feat, const uint32_t* edge, int cur, const T* g, T* logits, T* mask, T* E_out = nullptr,
          T* Hc_out = nullptr) const {
    const int N = gw * gh, F = feat_stride > 0 ? feat_stride : fin + 1, dk = d / heads;
    if (F < fin + 1) throw std::runtime_error("그래프 정책: 특징 칸 수가 임베딩 입력보다 적음");
    if (cur < 0 || cur >= N || feat[(size_t)cur * F] == 0) throw std::runtime_error("그래프 정책: 현재 노드가 무효");
    std::vector<int> node, row(N, -1);
    for (int j = 0; j < N; ++j)
      if (feat[(size_t)j * F] != 0) {
        row[j] = (int)node.size();
        node.push_back(j);
      }
    const int V = (int)node.size();
    // 가림 행렬 [V][V] (조밀) 과 현재 노드의 이웃 행
    std::vector<char> see((size_t)V * V, 0);
    int nbrow[kNb];
    for (int i = 0; i < V; ++i) {
      see[(size_t)i * V + i] = 1;
      const int y = node[i] / gw, x = node[i] % gw;
      for (int k = 0; k < kNb; ++k) {
        const int i5 = k >= 12 ? k + 1 : k, ny = y + i5 / 5 - 2, nx = x + i5 % 5 - 2;
        int r = -1;
        if (((edge[node[i]] >> k) & 1u) && ny >= 0 && ny < gh && nx >= 0 && nx < gw) r = row[ny * gw + nx];
        if (r >= 0) see[(size_t)i * V + r] = 1;
        if (node[i] == cur) nbrow[k] = r;
      }
    }
    const int rc = row[cur];
    std::vector<T> X((size_t)V * d), A((size_t)V * d), Q, O((size_t)V * d), Hd, u(V);
    {
      std::vector<T> x0((size_t)V * fin);
      for (int i = 0; i < V; ++i)
        for (int c = 0; c < fin; ++c) x0[(size_t)i * fin + c] = c < coord ? ((T)feat[(size_t)node[i] * F + 1 + c] - (T)128) / coord_scale : (T)feat[(size_t)node[i] * F + 1 + c] / (T)255;
      lin(x0.data(), V, fin, p(".emb.w"), d, p(".emb.b"), X.data(), false);
    }
    for (int l = 0; l < layers; ++l) {
      const std::string e = ".enc" + std::to_string(l);
      ln(X.data(), V, p(e + ".ln1.g"), p(e + ".ln1.b"), A.data());
      Q.assign((size_t)V * 3 * d, 0);
      lin(A.data(), V, d, p(e + ".wqkv"), 3 * d, nullptr, Q.data(), false);
      for (int h = 0; h < heads; ++h)
        for (int i = 0; i < V; ++i) {
          const T* q = &Q[(size_t)i * 3 * d + h * dk];
          T mx = (T)-1e30;
          for (int j = 0; j < V; ++j) {
            if (!see[(size_t)i * V + j]) continue;
            const T* k = &Q[(size_t)j * 3 * d + d + h * dk];
            T a = 0;
            for (int c = 0; c < dk; ++c) a += q[c] * k[c];
            u[j] = a / std::sqrt((T)dk);
            mx = std::max(mx, u[j]);
          }
          T sum = 0;
          for (int j = 0; j < V; ++j)
            if (see[(size_t)i * V + j]) sum += (u[j] = std::exp(u[j] - mx));
          T* o = &O[(size_t)i * d + h * dk];
          for (int c = 0; c < dk; ++c) o[c] = 0;
          for (int j = 0; j < V; ++j) {
            if (!see[(size_t)i * V + j]) continue;
            const T w = u[j] / sum;
            const T* v = &Q[(size_t)j * 3 * d + 2 * d + h * dk];
            for (int c = 0; c < dk; ++c) o[c] += w * v[c];
          }
        }
      lin(O.data(), V, d, p(e + ".wout"), d, nullptr, X.data(), true);
      ln(X.data(), V, p(e + ".ln2.g"), p(e + ".ln2.b"), A.data());
      Hd.assign((size_t)V * ff, 0);
      lin(A.data(), V, d, p(e + ".ff1.w"), ff, p(e + ".ff1.b"), Hd.data(), false);
      for (T& v : Hd) v = v > 0 ? v : 0;
      lin(Hd.data(), V, ff, p(e + ".ff2.w"), d, p(e + ".ff2.b"), X.data(), true);
    }
    // 디코더
    ln(X.data(), V, p(".dec.ln1.g"), p(".dec.ln1.b"), A.data());
    std::vector<T> KV((size_t)V * 2 * d), qc(d), od(d), h1(d), an(d), hid(ff), E(d);
    lin(A.data(), V, d, p(".dec.wkv"), 2 * d, nullptr, KV.data(), false);
    lin(&A[(size_t)rc * d], 1, d, p(".dec.wq"), d, nullptr, qc.data(), false);
    for (int h = 0; h < heads; ++h) {
      T mx = (T)-1e30, sum = 0;
      for (int j = 0; j < V; ++j) {
        T a = 0;
        for (int c = 0; c < dk; ++c) a += qc[h * dk + c] * KV[(size_t)j * 2 * d + h * dk + c];
        u[j] = a / std::sqrt((T)dk);
        mx = std::max(mx, u[j]);
      }
      for (int j = 0; j < V; ++j) sum += (u[j] = std::exp(u[j] - mx));
      for (int c = 0; c < dk; ++c) od[h * dk + c] = 0;
      for (int j = 0; j < V; ++j)
        for (int c = 0; c < dk; ++c) od[h * dk + c] += u[j] / sum * KV[(size_t)j * 2 * d + d + h * dk + c];
    }
    const T* hc = &X[(size_t)rc * d];
    for (int c = 0; c < d; ++c) h1[c] = hc[c];
    lin(od.data(), 1, d, p(".dec.wout"), d, nullptr, h1.data(), true);
    ln(h1.data(), 1, p(".dec.ln2.g"), p(".dec.ln2.b"), an.data());
    lin(an.data(), 1, d, p(".dec.ff1.w"), ff, p(".dec.ff1.b"), hid.data(), false);
    for (T& v : hid) v = v > 0 ? v : 0;
    E = h1;
    lin(hid.data(), 1, ff, p(".dec.ff2.w"), d, p(".dec.ff2.b"), E.data(), true);
    // 정책 머리
    const int C = 2 * d + glob;
    std::vector<T> cat(C), s(d), qp(d), kp(d);
    for (int c = 0; c < d; ++c) {
      cat[c] = E[c];
      cat[d + c] = hc[c];
    }
    for (int c = 0; c < glob; ++c) cat[2 * d + c] = g[c];
    lin(cat.data(), 1, C, p(".cur_emb.w"), d, p(".cur_emb.b"), s.data(), false);
    lin(s.data(), 1, d, p(".ptr.wq"), d, nullptr, qp.data(), false);
    int best = -1;
    for (int k = 0; k < kNb; ++k) {
      mask[k] = nbrow[k] >= 0 ? (T)1 : (T)0;
      logits[k] = 0;
      if (nbrow[k] < 0) continue;
      lin(&X[(size_t)nbrow[k] * d], 1, d, p(".ptr.wk"), d, nullptr, kp.data(), false);
      T a = 0;
      for (int c = 0; c < d; ++c) a += qp[c] * kp[c];
      logits[k] = (T)10 * std::tanh(a / std::sqrt((T)d));
      if (best < 0 || logits[k] > logits[best]) best = k;
    }
    if (E_out)
      for (int c = 0; c < d; ++c) E_out[c] = E[c];
    if (Hc_out)
      for (int c = 0; c < d; ++c) Hc_out[c] = hc[c];
    return best < 0 ? 0 : best;
  }

 private:
  std::string pfx_ = "gactor";
  const std::vector<T>& get(const std::string& nm) const {
    auto it = blk.find(nm);
    if (it == blk.end()) throw std::runtime_error("그래프 정책: 블록 없음 " + nm);
    return it->second;
  }
  const T* p(const std::string& suffix) const { return get(pfx_ + suffix).data(); }
  void need(const std::string& suffix, size_t n) const {
    if (get(pfx_ + suffix).size() != n) throw std::runtime_error("그래프 정책: 블록 크기 " + pfx_ + suffix);
  }
  // Y [R][out] (acc 면 +)= X [R][in]·W [in][out] + b
  static void lin(const T* X, int R, int in, const T* W, int out, const T* b, T* Y, bool acc) {
    for (int r = 0; r < R; ++r) {
      T* y = Y + (size_t)r * out;
      if (!acc)
        for (int o = 0; o < out; ++o) y[o] = 0;
      const T* x = X + (size_t)r * in;
      for (int i = 0; i < in; ++i) {
        const T xi = x[i];
        const T* w = W + (size_t)i * out;
        for (int o = 0; o < out; ++o) y[o] += xi * w[o];
      }
      if (b)
        for (int o = 0; o < out; ++o) y[o] += b[o];
    }
  }
  // 층 정규화 (torch nn.LayerNorm: 분산은 d 로 나눔, eps 1e-5)
  void ln(const T* X, int R, const T* g, const T* b, T* A) const {
    for (int r = 0; r < R; ++r) {
      const T* x = X + (size_t)r * d;
      T m = 0, v = 0;
      for (int c = 0; c < d; ++c) m += x[c];
      m /= (T)d;
      for (int c = 0; c < d; ++c) v += (x[c] - m) * (x[c] - m);
      const T s = (T)1 / std::sqrt(v / (T)d + (T)1e-5);
      for (int c = 0; c < d; ++c) A[(size_t)r * d + c] = g[c] * (x[c] - m) * s + b[c];
    }
  }
};

using GraphPolicy = GraphPolicyT<float>;

}  // namespace xnet_cpu
