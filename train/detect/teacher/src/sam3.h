// SAM 3 검출기(영상 추적기 제외) C++/CUDA 판. transformers Sam3Model 과 같은 계산을 한다.
// T = float(정합성 확인) 또는 bf16(빠른 경로, 누적은 float).
#pragma once
#include <memory>
#include <type_traits>
#include <string>
#include <vector>

#include "k.cuh"
#include "st.h"

namespace sam3 {

constexpr int IMG = 1008, PATCH = 14, GRID = 72, NTOK = GRID * GRID;  // 5184
constexpr int VD = 1024, VHEADS = 16, VMLP = 4736, VLAYERS = 32, WIN = 24;
constexpr int TD = 1024, THEADS = 16, TMLP = 4096, TLAYERS = 24, TLEN = 32, VOCAB = 49408;
constexpr int D = 256, HEADS = 8, MLP = 2048, ELAYERS = 6, DLAYERS = 6, NQ = 200;
constexpr int L0 = 288, L1 = 144;  // FPN 해상도(마스크는 288×288)

// 디바이스 버퍼(크기만 기억)
template <typename X>
struct Buf {
  X* p = nullptr;
  size_t n = 0;
  void alloc(size_t m) {
    if (p) CK(cudaFree(p));
    n = m;
    CK(cudaMalloc(&p, m * sizeof(X)));
  }
  ~Buf() {
    if (p) cudaFree(p);
  }
  operator X*() const { return p; }
};

template <typename T>
struct Lin {  // y = x W^T + b
  Buf<T> w;
  Buf<float> b;
  Buf<T> bh;  // 편향의 T 사본(빠른 경로 GEMM 에필로그용)
  int out = 0, in = 0;
};
struct Norm {
  Buf<float> g, b;
};

template <typename T>
struct Attn4 {  // q, k, v, o 투영
  Lin<T> q, k, v, o;
};

template <typename T>
class Model {
 public:
  Model(const SafeTensors& st, int max_prompts, cudaStream_t s);
  ~Model();

  // 텍스트: 토큰 [P, 32], 마스크 [P, 32] (호스트). 결과는 이 객체 안에 남는다(이미지마다 다시 쓰지 않음)
  void encode_text(const int* ids, const int* mask, int P);
  // 영상: pixels = 디바이스 float [3, 1008, 1008] (정규화 끝). fpn0/1/2 를 채운다
  void encode_image(const float* pixels);
  // 검출: DETR 인코더·디코더 + 점수. 결과는 logits/boxes/presence (디바이스)
  void detect();
  // 마스크: 프롬프트 전부의 질의 전부(정합성 확인용). masks [P, 200, 288*288] (디바이스 float 로짓)
  void masks_all();
  // 프롬프트 b 의 질의 qs 만 마스크 로짓 [nq, 288*288] 을 out(디바이스)에 쓴다
  void masks_for(int b, const std::vector<int>& qs, float* out);

  int P = 0;
  // 중간 결과(정합성 확인용으로 공개)
  Buf<float> text_hidden;  // [P*32, 1024] CLIP 마지막 층
  Buf<float> text_feat;    // [P*32, 256]
  Buf<float> vit;          // [5184, 1024]
  Buf<float> fpn0, fpn1, fpn2;  // NHWC [288², 256], [144², 256], [72², 256]
  Buf<float> pos2;              // [5184, 256]
  Buf<float> enc;               // [P*5184, 256] DETR 인코더 출력
  Buf<float> logits, boxes, presence;  // [P, 200], [P, 200, 4] xyxy 0..1, [P]
  Buf<float> masks;                    // [P, 200, 288*288]
  std::vector<float> vit_dbg[3];       // 층 0, 6, 7 뒤 (설정하면 채움)
  bool debug = false;

 private:
  const SafeTensors& st_;
  cudaStream_t s_;
  cublasHandle_t h_;
  k::Lt lt_;
  int maxP_;
  static constexpr bool kFast = !std::is_same_v<T, float>;

  // ---- 가중치 ----
  Lin<T> patch;
  Buf<float> vpos;  // [576, 1024]
  Norm vln_pre;
  struct VLayer {
    Norm ln1, ln2;
    Lin<T> qkv, o, fc1, fc2;
  };
  std::vector<VLayer> vl;
  struct Fpn {
    std::vector<Lin<T>> ct;  // 전치 합성곱(GEMM 꼴)
    Lin<T> p1;
    std::vector<Lin<T>> p2;  // 3×3 의 9 칸
  };
  Fpn fpn[3];

  Buf<float> tok_emb, tpos_emb;
  struct TLayer {
    Norm ln1, ln2;
    Lin<T> qkv, o, fc1, fc2;
  };
  std::vector<TLayer> tl;
  Norm tln_final;
  Lin<T> tproj;

  struct ELayer {
    Norm ln1, ln2, ln3;
    Lin<T> qk, v, so;  // 자기 어텐션
    Attn4<T> ca;       // 텍스트 교차
    Lin<T> fc1, fc2;
  };
  std::vector<ELayer> el;
  struct DLayer {
    Lin<T> sqk, sv, so;
    Norm sln;
    Attn4<T> ta, va;
    Norm tln, vln;
    Lin<T> fc1, fc2;
    Norm mln;
  };
  std::vector<DLayer> dl;
  Norm d_out_ln, d_pres_ln;
  Lin<T> box1, box2, box3, pres1, pres2, pres3, ref1, ref2;
  Buf<float> rpbw_x, rpbw_y;  // BoxRPB MLP 가중치(float 묶음)
  Buf<float> query_embed, ref_points, pres_token;
  Lin<T> sc_mlp1, sc_mlp2, sc_text, sc_query;
  Norm sc_ln;
  Attn4<T> m_ca;
  Norm m_ca_ln;
  Lin<T> m_conv[2][9];
  Norm m_gn[2];
  Lin<T> m_inst, m_emb[3];

  // ---- 작업 버퍼 ----
  Buf<T> tA, tB, tC, tD, tE;  // GEMM 입력용
  Buf<float> fA, fB, fC;      // GEMM 출력용
  Buf<float> rope_win, rope_glob;  // 창 순서 토큰의 RoPE 위치 [5184, 2]
  Buf<int> tmask;       // [P, 32]
  Buf<T> text_t;        // [P*32, 256] T
  Buf<T> tkv;           // 층마다 텍스트 k, v: [(6 + 6 + 1) * 2, P*32, 256]
  Buf<float> proj_text; // [P, 256]
  Buf<float> hdec, qpos, refb, rpby, rpbx;
  Buf<T> enc_t, encpos_t;
  Buf<float> pix;       // [P, 288², 256] 픽셀 임베딩
  Buf<T> inst_t, memb_t;

  void load_lin(Lin<T>& l, const std::string& name, int out, int in, bool bias = true);
  void load_lin_cat(Lin<T>& l, const std::vector<std::string>& names, int out_each, int in);
  void load_norm(Norm& n, const std::string& name, int C);
  void upload_T(Buf<T>& dst, const std::vector<float>& h);
  void upload_f(Buf<float>& dst, const float* h, size_t n);

  void lin(const Lin<T>& l, const T* x, int rows, float* y, int ldx = -1, int ldy = -1);
  // y (T) = act(x W^T + b), act 0 없음 1 GELU 2 ReLU
  void proj(const Lin<T>& l, const T* x, int rows, T* y, int act = 0, int ldx = -1, int ldy = -1);
  // r (float) += x W^T + b
  void proj_res(const Lin<T>& l, const T* x, int rows, float* r, int ldx = -1, int ldr = -1);
  void set_bh(Lin<T>& l);
  void conv3x3(const Lin<T>* w9, const T* x, int B, int h, int w, int Cin, const float* bias, float* out);
  void vit_layer(int i, const float* rope_pos, bool windowed);
  void mask_pixels(int b0, int nbp);
  void mask_queries();
};

}  // namespace sam3
