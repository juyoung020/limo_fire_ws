// 그래프 어텐션 신경망 (HEADER, Cao et al. arXiv 2510.15679 III-B · 전작 ARiADNE 공개 코드 model.py 의 구조). 설명·출처·다른 곳은 network/README.md "그래프 어텐션 정책".
//
//   몸통 GraphBody (학습 알고리즘과 무관 — 정책·가치·Q 머리를 갈아 끼운다):
//     노드 특징 → 초기 임베딩 Linear → 인코더 층 × L (정규화 → 간선 이웃+자신에게만 여러 머리 어텐션 → 잔차 → 정규화 → FF ReLU → 잔차)
//     → 디코더 층 1 (현재 노드 특징이 질의, 모든 유효 노드가 키·값인 교차 어텐션 → 잔차 → 정규화 → FF → 잔차)
//     출력: 현재 노드 특징 Hc [n][d], 디코더 출력 E [n][d], 현재 노드의 이웃 특징 Hn [n][24][d]
//   정책 머리 PointerHead: s = Linear([E | Hc | (전역 벡터)]) → 포인터(단일 머리 어텐션 점수, 10·tanh 로 자름) → 이웃 24 칸 로짓
//   가치 머리는 ActorCritic 의 MLP(Net) — [E | Hc] → V.
//
// 희소 구현: 입력은 격자 자리 N 개짜리 조밀 버퍼지만 유효 노드만 "행"으로 모아(샘플 순 → 자리 순) 계산한다. 인코더 어텐션은 행마다
// 자신 + 5×5 스텐실 이웃 24 = 키 25 개만 모은다 — 간선 가림을 씌운 조밀 어텐션(model.py)과 수학적으로 같다(tools/graph_net_check 가 비교).
// 역방향의 키·값 기울기는 흩뿌려 더하지 않고 키 쪽에서 "나를 본 질의"(반대 스텐실)를 고정 순서로 모아 더한다 — 원자 연산 없음, 비트 재현.
// 학습 변수 수는 격자 크기·노드 수와 무관하다.
#pragma once

#include <cstdint>
#include <random>
#include <string>
#include <vector>

#include "common.cuh"

namespace XNET_NS {

struct ParamBlock;  // actor_critic.h

// 일부러 틀린 기울기(기울기 검사 --graph --negative): 층 정규화 역방향에서 분산 항을 뺀다.
extern bool g_break_graph_grad;

constexpr int kGNb = 24;     // 5×5 스텐실에서 자신 제외 (mdp::graph::NB)
constexpr int kGSlot = 25;   // 인코더 어텐션 키 자리: 0 = 자신, 1 + k = 스텐실 k 번 이웃

struct GraphSpec {
  int gw = 32, gh = 32;   // 노드 격자 (가중치와 무관 — 같은 체크포인트를 다른 격자에 쓸 수 있다)
  int feat = 8;           // 노드마다 u8 칸 수 (버퍼 모양). 0 번 = 유효
  int d = 128;            // 특징 차원 (HEADER df = 128)
  int heads = 8;          // model.py n_head = 8
  int layers = 6;         // 인코더 층 수 (HEADER: 6)
  int ff = 512;           // model.py feedForward 은닉
  int in = 5;             // 임베딩 입력 칸 수: 특징 1 … in 번 (/255) — HEADER v' = (x, y, u, e, b) 5 칸 (III-A). 그 뒤 칸(예약)은 안 읽는다. feat − 1 보다 크면 feat − 1
  int coord = 2;          // 임베딩 입력의 앞 coord 칸(특징 1·2 번 = F_DX·F_DY)은 좌표: (u8 − 128) / coord_scale — 0 중심 부호 있는 값 (전작 agent.py:157-160)
  float coord_scale = 160.f;   // u8 − 128 = 4 × 노드 칸 차(mdp/graph.h write_feat) → / 160 = 노드 칸 차 / 40. 전작: (노드 − 현재) / UPDATING_MAP_SIZE / 2 = 차 / 160 m = 노드 칸(4 m) 차 / 40
};

// 관측 한 묶음 (모두 GPU). idx 를 주면 샘플 i = 버퍼의 idx[i] 번째 행 (롤아웃에서 바로 읽기)
struct GraphIn {
  const uint8_t* feat = nullptr;    // [·][N][feat]
  const uint32_t* edge = nullptr;   // [·][N]
  const int* cur = nullptr;         // [·]
  const int* idx = nullptr;
};

// 샘플마다 유효 노드 수를 host 로 (현재 노드가 범위 밖·무효면 −1). d_cnt = 장치 작업 버퍼 [n]. 스트림을 기다린다
void graph_count(const GraphSpec& s, cudaStream_t st, int n, const GraphIn& in, int* d_cnt, int* cnt_host);

class GraphBody {
 public:
  void build(const GraphSpec& s, std::vector<ParamBlock>& reg, long& cursor, const std::string& prefix);
  // cap_rows = 한 번에 넣을 수 있는 유효 노드(행) 합, max_n = 샘플 수 상한
  void bind(real* params, real* grads, long cap_rows, int max_n);
  void free_ws();
  // model.py 와 같은 초기화: 어텐션 행렬 U(±1/√마지막 차원), nn.Linear 기본 U(±1/√fan_in), 층 정규화 1·0. float 로 뽑음
  void init(std::mt19937_64& rng, std::vector<real>& host) const;
  // cnt = graph_count() 결과 (그 합 ≤ cap_rows, n ≤ max_n). 결과는 E()·Hc()·Hn()
  void forward(cublasHandle_t h, cudaStream_t st, int n, const GraphIn& in, const int* cnt);
  // 직전 forward 의 역방향 → d_grads 에 누적. dE·dHc: [n][·] 행 간격 ld (nullptr = 0), dHn: [n][24][d] (nullptr = 0).
  // 둘째 묶음(dE2·dHc2, 행 간격 ld2)은 공유 몸통에서 다른 머리의 기울기 — 더하는 순서 고정(첫째 + 둘째)
  void backward(cublasHandle_t h, cudaStream_t st, int n, const real* dE, const real* dHc, long ld, const real* dHn,
                const real* dE2 = nullptr, const real* dHc2 = nullptr, long ld2 = 0);
  const real* E() const { return E_; }     // 디코더 출력 [max_n][d] (model.py enhanced_current_node_feature)
  const real* Hc() const { return Hc_; }   // 현재 노드 특징 [max_n][d]
  const real* Hn() const { return Hn_; }   // 이웃 특징 [max_n][24][d] — 간선 없는 칸은 0
  const real* H() const { return X_; }     // 노드 특징 [행][d] (검사용)
  const int* nbrow() const { return nbrow_; }   // [max_n][24]: 현재 노드의 k 번 이웃 행 (간선 없거나 이웃 무효면 −1) — 행동 가림
  long rows() const { return M_; }
  const GraphSpec& spec() const { return s_; }
  size_t ws_bytes() const { return ws_bytes_; }
  long cap_rows() const { return cap_; }

 private:
  struct Layer {
    long ln1_g, ln1_b, wqkv, wout, ln2_g, ln2_b, w1, b1, w2, b2;
    real *xh1 = nullptr, *rs1 = nullptr, *QKV = nullptr, *W = nullptr, *O = nullptr, *xh2 = nullptr, *rs2 = nullptr, *HID = nullptr;
  };
  GraphSpec s_;
  int fin_ = 0;
  long emb_w_ = 0, emb_b_ = 0;
  std::vector<Layer> L_;
  // 디코더 층: 정규화 1(질의·기억 공용), W_q [d][d], W_kv [d][2d], W_out, 정규화 2, FF
  long dln1_g_ = 0, dln1_b_ = 0, dwq_ = 0, dwkv_ = 0, dwout_ = 0, dln2_g_ = 0, dln2_b_ = 0, dw1_ = 0, db1_ = 0, dw2_ = 0, db2_ = 0;
  real *P_ = nullptr, *G_ = nullptr;
  long cap_ = 0, M_ = 0;
  int max_n_ = 0;
  size_t ws_bytes_ = 0;
  std::vector<void*> owned_;
  std::vector<int> h_start_;
  // 색인
  int *row_start_ = nullptr, *node2row_ = nullptr, *rsamp_ = nullptr, *rnode_ = nullptr, *gnb_ = nullptr, *rowcur_ = nullptr,
      *nbrow_ = nullptr;
  uint32_t* ebits_ = nullptr;
  // 행 단위
  real *X0_ = nullptr, *X_ = nullptr, *A_ = nullptr, *dX_ = nullptr, *dA_ = nullptr, *dQKV_ = nullptr, *DU_ = nullptr, *dHID_ = nullptr, *T_ = nullptr;
  real *dxh_ = nullptr, *drs_ = nullptr, *KV_ = nullptr, *Wd_ = nullptr;
  // 샘플 단위
  real *Ac_ = nullptr, *Qc_ = nullptr, *Od_ = nullptr, *h1_ = nullptr, *xh2d_ = nullptr, *rs2d_ = nullptr, *hidd_ = nullptr, *E_ = nullptr, *Hc_ = nullptr,
       *Hn_ = nullptr, *An_ = nullptr, *gE_ = nullptr, *dh1_ = nullptr, *dAn_ = nullptr, *dhidd_ = nullptr, *dOd_ = nullptr, *dQc_ = nullptr,
       *dAc_ = nullptr, *Tn_ = nullptr, *gHc_ = nullptr;
};

// 정책 머리: 상태 특징 s = [E | Hc | glob]·W_ce + b_ce (model.py current_embedding — glob 열은 우리 추가, 가중치 0 으로 시작),
// 포인터 로짓_k = 10·tanh((s·W_q)·(Hn_k·W_k)/√d) (model.py SingleHeadAttention, tanh_clipping 10). 가림·log softmax 는 PPO 범주 분포가 한다
struct PointerHead {
  int d = 0, glob = 0;
  long wce = 0, bce = 0, wq = 0, wk = 0;
  int max_n = 0;
  real *cat = nullptr, *s = nullptr, *Qp = nullptr, *Kp = nullptr, *U = nullptr, *ds = nullptr, *dQp = nullptr, *dKp = nullptr, *dcat = nullptr,
       *dHn = nullptr;
  size_t ws_bytes = 0;
  void build(int d, int glob, std::vector<ParamBlock>& reg, long& cursor, const std::string& prefix);
  void alloc(int max_n);
  void release();
  void init(std::mt19937_64& rng, std::vector<real>& host) const;
  // glob [n][ld_glob] (glob == 0 이면 안 읽음). logits [n][24]
  void forward(cublasHandle_t h, cudaStream_t st, int n, const real* E, const real* Hc, const real* glob, long ld_glob, const real* Hn,
               const real* params, real* logits);
  // dlogits [n][24] → 변수 기울기 누적, 몸통으로 갈 기울기: d_E() = dcat, d_Hc() = dcat + d (행 간격 ld()), d_Hn()
  void backward(cublasHandle_t h, cudaStream_t st, int n, const real* Hn, const real* params, real* grads,
                const real* dlogits);
  const real* d_E() const { return dcat; }
  const real* d_Hc() const { return dcat + d; }
  long ld() const { return 2 * d + glob; }
  const real* d_Hn() const { return dHn; }
};

}  // namespace XNET_NS
