// 탐사 RL 신경망(구조 버전 2)을 ONNX 파일로 — Netron 으로 그림을 뽑거나 다른 도구로 열어 보기 위함. 학습·추론 길에는 안 쓴다.
//   onnx_export --out docs/models/explore_net_v2.onnx [--ckpt runs/<실험>/ckpt/model_latest.bin]
// 구조는 network/actor_critic.cu explore_spec() 를 그대로 읽어 만든다(층 수·채널·MLP 폭·입력 크기 = mdp/obs_layout.h). 계산 순서도 실제와 같다:
//   u8 이미지 → Cast·÷255 → 2×2 s2 합성곱 + ELU × 3 → (셋째 층 격자가 홀수면 아래·오른쪽 0 덧댐) → 넷째 층 → NHWC 로 펼침([픽셀][채널] = 우리 평탄화 순서)
//   → 가지 3 개 Concat = 공유 특징 F → [F | 정규화 1D] → actor MLP (평균 3 + 선언 로짓) · log σ, critic MLP (가치).
// 1D 정규화 (x − μ)/(√σ² + ε) 도 노드로 넣는다(Sub·Div — 통계는 체크포인트에 있음).
// --ckpt 를 주면 체크포인트(형식 2)의 가중치·정규화 통계를 initializer 로 넣는다(합성곱 W 는 [cout][자식 2×2][cin] → ONNX [cout][cin][kh][kw]).
// 주지 않으면 initializer 는 모양·형만(데이터 없음) — 파일이 작고 Netron 은 그대로 그린다(값을 보려면 --ckpt).
// protobuf 는 라이브러리 없이 와이어 형식(varint·길이 접두)으로 직접 쓴다. 필드 번호는 onnx/onnx.proto (ONNX 1.x, IR 8, opset 17).
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

#include "../../mdp/obs_layout.h"
#include "actor_critic.h"

namespace {

// ---------------- protobuf 와이어 형식 ----------------
struct Pb {
  std::string b;
  void varint(uint64_t v) {
    while (v >= 0x80) {
      b += char((v & 0x7F) | 0x80);
      v >>= 7;
    }
    b += char(v);
  }
  void key(int field, int wire) { varint((uint64_t(field) << 3) | wire); }
  Pb& i64(int f, int64_t v) { key(f, 0), varint(uint64_t(v)); return *this; }
  Pb& f32(int f, float v) {
    key(f, 5);
    uint32_t u;
    std::memcpy(&u, &v, 4);
    for (int k = 0; k < 4; ++k) b += char((u >> (8 * k)) & 0xFF);
    return *this;
  }
  Pb& bytes(int f, const std::string& s) { key(f, 2), varint(s.size()), b += s; return *this; }
  Pb& msg(int f, const Pb& m) { return bytes(f, m.b); }
};

// onnx.proto 필드 번호
namespace F {
constexpr int Model_ir_version = 1, Model_producer_name = 2, Model_producer_version = 3, Model_doc = 6, Model_graph = 7,
              Model_opset = 8;
constexpr int Opset_domain = 1, Opset_version = 2;
constexpr int Graph_node = 1, Graph_name = 2, Graph_initializer = 5, Graph_doc = 10, Graph_input = 11, Graph_output = 12,
              Graph_value_info = 13;
constexpr int Node_input = 1, Node_output = 2, Node_name = 3, Node_op_type = 4, Node_attribute = 5, Node_doc = 6;
constexpr int Attr_name = 1, Attr_f = 2, Attr_i = 3, Attr_ints = 8, Attr_type = 20;
constexpr int AttrT_FLOAT = 1, AttrT_INT = 2, AttrT_INTS = 7;
constexpr int VI_name = 1, VI_type = 2, VI_doc = 3;
constexpr int Type_tensor = 1, TT_elem = 1, TT_shape = 2, Shape_dim = 1, Dim_value = 1, Dim_param = 2;
constexpr int Tensor_dims = 1, Tensor_type = 2, Tensor_name = 8, Tensor_raw = 9, Tensor_doc = 12;
constexpr int DT_FLOAT = 1, DT_UINT8 = 2, DT_INT64 = 7;
}  // namespace F

struct Graph {
  Pb g;
  std::vector<std::string> outputs_seen;
  // 모양: 음수 = 배치 기호 "N"
  static Pb value_info(const std::string& name, int dt, const std::vector<int64_t>& shape, const std::string& doc = "") {
    Pb sh;
    for (int64_t d : shape) {
      Pb dim;
      if (d < 0) dim.bytes(F::Dim_param, "N");
      else dim.i64(F::Dim_value, d);
      sh.msg(F::Shape_dim, dim);
    }
    Pb tt;
    tt.i64(F::TT_elem, dt).msg(F::TT_shape, sh);
    Pb ty;
    ty.msg(F::Type_tensor, tt);
    Pb vi;
    vi.bytes(F::VI_name, name).msg(F::VI_type, ty);
    if (!doc.empty()) vi.bytes(F::VI_doc, doc);
    return vi;
  }
  void input(const std::string& n, int dt, const std::vector<int64_t>& s, const std::string& doc) {
    g.msg(F::Graph_input, value_info(n, dt, s, doc));
  }
  void output(const std::string& n, const std::vector<int64_t>& s, const std::string& doc) {
    g.msg(F::Graph_output, value_info(n, F::DT_FLOAT, s, doc));
  }
  void shape(const std::string& n, const std::vector<int64_t>& s) { g.msg(F::Graph_value_info, value_info(n, F::DT_FLOAT, s)); }
  // 데이터가 비면 모양·형만 (--ckpt 없음)
  void init(const std::string& n, int dt, const std::vector<int64_t>& dims, const std::string& raw, const std::string& doc = "") {
    Pb t;
    for (int64_t d : dims) t.i64(F::Tensor_dims, d);
    t.i64(F::Tensor_type, dt).bytes(F::Tensor_name, n);
    if (!raw.empty()) t.bytes(F::Tensor_raw, raw);
    if (!doc.empty()) t.bytes(F::Tensor_doc, doc);
    g.msg(F::Graph_initializer, t);
  }
  struct Attr {
    std::string name;
    int type;
    float f = 0;
    int64_t i = 0;
    std::vector<int64_t> ints;
  };
  void node(const std::string& op, const std::string& name, const std::vector<std::string>& in, const std::vector<std::string>& out,
            const std::vector<Attr>& attrs = {}, const std::string& doc = "") {
    Pb n;
    for (auto& s : in) n.bytes(F::Node_input, s);
    for (auto& s : out) n.bytes(F::Node_output, s);
    n.bytes(F::Node_name, name).bytes(F::Node_op_type, op);
    for (auto& a : attrs) {
      Pb at;
      at.bytes(F::Attr_name, a.name).i64(F::Attr_type, a.type);
      if (a.type == F::AttrT_FLOAT) at.f32(F::Attr_f, a.f);
      if (a.type == F::AttrT_INT) at.i64(F::Attr_i, a.i);
      if (a.type == F::AttrT_INTS)
        for (int64_t v : a.ints) at.i64(F::Attr_ints, v);
      n.msg(F::Node_attribute, at);
    }
    if (!doc.empty()) n.bytes(F::Node_doc, doc);
    g.msg(F::Graph_node, n);
  }
};

std::string raw_f32(const std::vector<float>& v) { return std::string(reinterpret_cast<const char*>(v.data()), v.size() * 4); }
std::string raw_i64(const std::vector<int64_t>& v) { return std::string(reinterpret_cast<const char*>(v.data()), v.size() * 8); }

// ---------------- 체크포인트 (ppo.cu save 형식 2) ----------------
struct Ckpt {
  std::map<std::string, std::vector<float>> blk;
  std::vector<float> mean, var;
};
Ckpt read_ckpt(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) throw std::runtime_error("체크포인트 없음: " + path);
  auto r = [&](void* d, size_t n) {
    f.read(reinterpret_cast<char*>(d), n);
    if (!f) throw std::runtime_error("체크포인트가 잘림");
  };
  uint32_t hdr[3];
  r(hdr, 12);
  if (hdr[0] != 0x4F505058 || hdr[1] != 2 || hdr[2] != 4) throw std::runtime_error("체크포인트 형식 2(fp32)가 아님");
  int32_t nb;
  r(&nb, 4);
  std::vector<std::pair<std::string, int64_t>> names;
  for (int i = 0; i < nb; ++i) {
    int32_t ln;
    r(&ln, 4);
    std::string nm(ln, '\0');
    r(nm.data(), ln);
    int64_t n;
    r(&n, 8);
    names.push_back({nm, n});
  }
  int64_t P;
  r(&P, 8);
  std::vector<float> p(P);
  r(p.data(), P * 4);
  f.seekg(2 * P * 4 + 8, std::ios::cur);  // Adam m·v, 걸음
  int32_t D;
  r(&D, 4);
  Ckpt c;
  c.mean.resize(D), c.var.resize(D);
  r(c.mean.data(), D * 4);
  r(c.var.data(), D * 4);
  int64_t off = 0;
  for (auto& [nm, n] : names) {
    c.blk[nm].assign(p.begin() + off, p.begin() + off + n);
    off += n;
  }
  if (off != P) throw std::runtime_error("층 크기 합 ≠ 변수 수");
  return c;
}

}  // namespace

int main(int argc, char** argv) {
  std::string out = "explore_net_v2.onnx", ckpt;
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    if (a == "--out" && i + 1 < argc) out = argv[++i];
    else if (a == "--ckpt" && i + 1 < argc) ckpt = argv[++i];
    else {
      std::fprintf(stderr, "onnx_export [--out 파일.onnx] [--ckpt 체크포인트(형식 2)]\n");
      return 2;
    }
  }
  try {
    namespace o = mdp::obs;
    const xnet::ACSpec sp = xnet::explore_spec();
    Ckpt ck;
    const bool has = !ckpt.empty();
    if (has) ck = read_ckpt(ckpt);
    auto data = [&](const std::string& n, size_t expect) -> const std::vector<float>* {
      if (!has) return nullptr;
      auto it = ck.blk.find(n);
      if (it == ck.blk.end() || it->second.size() != expect) throw std::runtime_error("체크포인트에 " + n + " 없음/크기 다름");
      return &it->second;
    };
    Graph G;
    using A = Graph::Attr;
    // ---- 입력 (모양 = obs_layout.h)
    const char* img_name[3] = {"near_map", "far_map", "depth"};
    const char* img_doc[3] = {"가까운 지도 u8 (로봇 중심·로봇 방향 10 cm): 확률·지나간 곳·물체 4", "먼 지도 u8 (시작점 기준 25.6 m 창 0.8 m): 확률·지나간 곳·현재 위치·물체 4",
                              "깊이 u8 (0 = 구멍)"};
    for (int b = 0; b < 3; ++b) {
      const auto& im = sp.enc.imgs[b];
      G.input(img_name[b], F::DT_UINT8, {-1, im.c, im.h, im.w}, img_doc[b]);
    }
    G.input("obs_1d", F::DT_FLOAT, {-1, sp.norm_dim},
            "1D 관측 원값: 정책 " + std::to_string(o::POLICY_1D) + " (라이다 4 장·세기·숫자) + 가치 전용 참값 " + std::to_string(o::PRIV));
    // ---- 공통 상수
    G.init("inv255", F::DT_FLOAT, {}, raw_f32({1.f / 255.f}));
    // ---- 정규화 (Normalizer: (x − μ)/(√σ² + ε))
    {
      std::vector<float> mu(sp.norm_dim, 0.f), sd(sp.norm_dim, 1.f + xnet::Normalizer::kEps);
      if (has)
        for (int j = 0; j < sp.norm_dim; ++j) mu[j] = ck.mean[j], sd[j] = std::sqrt(ck.var[j]) + xnet::Normalizer::kEps;
      G.init("norm_mean", F::DT_FLOAT, {sp.norm_dim}, has ? raw_f32(mu) : "");
      G.init("norm_std_eps", F::DT_FLOAT, {sp.norm_dim}, has ? raw_f32(sd) : "", "√σ² + 1e-2 (rsl_rl EmpiricalNormalization)");
      G.node("Sub", "norm/sub", {"obs_1d", "norm_mean"}, {"obs_1d_c"});
      G.node("Div", "norm/div", {"obs_1d_c", "norm_std_eps"}, {"obs_1d_n"}, {}, "실행 평균·분산 정규화 (롤아웃 중 갱신, network/README)");
      G.init("slice_starts", F::DT_INT64, {1}, raw_i64({0}));
      G.init("slice_ends", F::DT_INT64, {1}, raw_i64({sp.actor.in1d}));
      G.init("slice_axes", F::DT_INT64, {1}, raw_i64({1}));
      G.node("Slice", "norm/policy_part", {"obs_1d_n", "slice_starts", "slice_ends", "slice_axes"}, {"policy_1d_n"}, {},
             "actor 는 앞 정책 1D 만");
      G.shape("policy_1d_n", {-1, sp.actor.in1d});
    }
    // ---- 공유 인코더: 가지마다 합성곱
    std::vector<std::string> feats;
    int64_t nf = 0;
    const char* bname[3] = {"near", "far", "depth"};
    for (int b = 0; b < 3; ++b) {
      const auto& im = sp.enc.imgs[b];
      const std::string p = std::string("enc/") + bname[b];
      G.node("Cast", p + "/cast", {img_name[b]}, {p + "/f"}, {A{"to", F::AttrT_INT, 0, F::DT_FLOAT}});
      G.node("Mul", p + "/div255", {p + "/f", "inv255"}, {p + "/x0"});
      std::string x = p + "/x0";
      int c = im.c, h = im.h, w = im.w;
      for (size_t l = 0; l < sp.enc.conv_ch.size(); ++l) {
        const int co = sp.enc.conv_ch[l];
        if (l == 3 && (h % 2 || w % 2)) {  // 넷째 층: 홀수 격자를 아래·오른쪽 0 덧댐 (actor_critic.cu tail_forward)
          G.init(p + "/pad_amount", F::DT_INT64, {8}, raw_i64({0, 0, 0, 0, 0, 0, h % 2, w % 2}));
          G.node("Pad", p + "/pad", {x, p + "/pad_amount"}, {x + "_pad"}, {}, "셋째 층 격자 홀수 → 0 덧대 짝수");
          x += "_pad";
          h += h % 2, w += w % 2;
          G.shape(x, {-1, c, h, w});
        }
        const std::string wn = "enc.cnn" + std::to_string(b) + ".conv" + std::to_string(l);
        // 가중치: 우리 [cout][자식 dy·2+dx][cin] → ONNX [cout][cin][dy][dx]
        std::vector<float> W;
        if (const auto* src = data(wn + ".w", size_t(co) * 4 * c)) {
          W.resize(src->size());
          for (int a = 0; a < co; ++a)
            for (int ci = 0; ci < c; ++ci)
              for (int k = 0; k < 4; ++k) W[((size_t(a) * c + ci) * 2 + k / 2) * 2 + k % 2] = (*src)[(size_t(a) * 4 + k) * c + ci];
        }
        const auto* bias = data(wn + ".b", co);
        G.init(wn + ".w", F::DT_FLOAT, {co, c, 2, 2}, has ? raw_f32(W) : "");
        G.init(wn + ".b", F::DT_FLOAT, {co}, bias ? raw_f32(*bias) : "");
        const std::string y = p + "/conv" + std::to_string(l);
        G.node("Conv", y, {x, wn + ".w", wn + ".b"}, {y},
               {A{"kernel_shape", F::AttrT_INTS, 0, 0, {2, 2}}, A{"strides", F::AttrT_INTS, 0, 0, {2, 2}}});
        c = co, h /= 2, w /= 2;
        G.node("Elu", y + "/elu", {y}, {y + "_a"}, {A{"alpha", F::AttrT_FLOAT, 1.f}});
        G.shape(y + "_a", {-1, c, h, w});
        x = y + "_a";
      }
      // 평탄화 순서 = [픽셀(래스터)][채널] → NHWC 로 돌린 뒤 펼침
      G.node("Transpose", p + "/nhwc", {x}, {p + "/nhwc"}, {A{"perm", F::AttrT_INTS, 0, 0, {0, 2, 3, 1}}});
      G.node("Flatten", p + "/flatten", {p + "/nhwc"}, {p + "/flat"}, {A{"axis", F::AttrT_INT, 0, 1}});
      G.shape(p + "/flat", {-1, int64_t(h) * w * c});
      feats.push_back(p + "/flat");
      nf += int64_t(h) * w * c;
    }
    G.node("Concat", "enc/features", feats, {"F"}, {A{"axis", F::AttrT_INT, 0, 1}}, "공유 인코더 특징 F (actor·critic 이 같이 씀)");
    G.shape("F", {-1, nf});
    // ---- MLP (actor·critic 따로, 입력 [F | 1D])
    auto mlp = [&](const std::string& who, const xnet::NetSpec& ns, const std::string& x1d, const std::string& out_name) {
      const int64_t in = nf + ns.in1d;
      G.node("Concat", who + "/input", {"F", x1d}, {who + "/x"}, {A{"axis", F::AttrT_INT, 0, 1}});
      G.shape(who + "/x", {-1, in});
      std::vector<int> dims = ns.hidden;
      dims.push_back(ns.out);
      std::string x = who + "/x";
      int64_t k = in;
      for (size_t l = 0; l < dims.size(); ++l) {
        const std::string wn = who + ".mlp" + std::to_string(l);
        const auto* W = data(wn + ".w", size_t(dims[l]) * k);
        const auto* B = data(wn + ".b", dims[l]);
        G.init(wn + ".w", F::DT_FLOAT, {dims[l], k}, W ? raw_f32(*W) : "");
        G.init(wn + ".b", F::DT_FLOAT, {dims[l]}, B ? raw_f32(*B) : "");
        const bool last = l + 1 == dims.size();
        const std::string y = last ? out_name : who + "/fc" + std::to_string(l);
        G.node("Gemm", who + "/fc" + std::to_string(l), {x, wn + ".w", wn + ".b"}, {y}, {A{"transB", F::AttrT_INT, 0, 1}});
        if (!last) {
          G.node("Elu", y + "/elu", {y}, {y + "_a"}, {A{"alpha", F::AttrT_FLOAT, 1.f}});
          G.shape(y + "_a", {-1, dims[l]});
          x = y + "_a";
        }
        k = dims[l];
      }
    };
    mlp("actor", sp.actor, "policy_1d_n", "actor_out");
    mlp("critic", sp.critic, "obs_1d_n", "value");
    // log σ: 상태 무관 변수 — 출력으로 그대로
    {
      const auto* L = data("log_std", sp.n_cont);
      G.init("log_std", F::DT_FLOAT, {sp.n_cont}, L ? raw_f32(*L) : "", "가우시안 log σ (v, vy, ω) — 상태 무관");
      G.node("Identity", "actor/log_std", {"log_std"}, {"log_std_out"});
    }
    G.output("actor_out", {-1, sp.actor.out}, "v·vy·ω 평균 (가우시안, 정규 단위) + 끝 선언 로짓 (베르누이)");
    G.output("log_std_out", {sp.n_cont}, "log σ (v, vy, ω)");
    G.output("value", {-1, sp.critic.out}, "가치 V(s)");
    G.g.bytes(F::Graph_name, "explore_net_v2");
    G.g.bytes(F::Graph_doc, "탐사 RL 정책·가치 신경망 구조 버전 2 — 공유 합성곱 인코더 + actor·critic MLP (train/explore/network/README.md)");
    Pb opset;
    opset.bytes(F::Opset_domain, "").i64(F::Opset_version, 17);
    Pb model;
    model.i64(F::Model_ir_version, 8).bytes(F::Model_producer_name, "limo_fire onnx_export").bytes(F::Model_producer_version, "1")
        .bytes(F::Model_doc, has ? "가중치: " + ckpt : "모양만 (가중치 없음 — --ckpt 로 넣음)").msg(F::Model_graph, G.g).msg(F::Model_opset, opset);
    std::ofstream f(out, std::ios::binary);
    if (!f) throw std::runtime_error("못 씀: " + out);
    f.write(model.b.data(), model.b.size());
    std::printf("%s: %zu B, 특징 F %lld, actor 입력 %lld, critic 입력 %lld%s\n", out.c_str(), model.b.size(), (long long)nf,
                (long long)(nf + sp.actor.in1d), (long long)(nf + sp.critic.in1d), has ? " (가중치 포함)" : " (모양만)");
  } catch (const std::exception& e) {
    std::fprintf(stderr, "onnx_export: %s\n", e.what());
    return 1;
  }
  return 0;
}
