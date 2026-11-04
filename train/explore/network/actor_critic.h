// 탐사 RL 정책·가치 신경망 (GPU 정방향·역방향). 구조·근거는 network/README.md.
//
//   2D 입력마다 CNN 가지 하나(합성곱 2×2 stride 2, ELU) → 평탄화 ─┐
//   1D 관측 (실행 평균·분산으로 정규화) ─────────────────────────────────┴→ 이어 붙임 → MLP [512,256,128] ELU → 출력
//
// actor 출력 = (v, vy, ω 평균, 선언 로짓), critic 출력 = 가치 1. 둘은 같은 꼴, 가중치 따로.
// 모든 학습 변수(actor·critic·log σ)는 연속 버퍼 하나(d_params)에, 기울기도 같은 배치(d_grads)에 — Adam·기울기 노름이 커널 하나.
// 활성값 배치: CNN 안은 [채널][샘플][행·열] (합성곱이 GEMM 하나가 되게), MLP 는 [샘플][특징].
#pragma once

#include <cstdint>
#include <random>
#include <string>
#include <vector>

#include "cnn_fused.cuh"
#include "common.cuh"

namespace XNET_NS {

// 일부러 틀린 기울기(유한 차분 검사의 --negative): ELU 역방향 도함수를 1 로 둔다.
extern bool g_break_elu_grad;

struct ImgIn {
  int off, c, h, w;  // 환경 하나 u8 관측 안의 바이트 오프셋, 채널·행·열
};

struct NetSpec {
  std::vector<ImgIn> imgs;    // CNN 가지들
  int img_stride = 0;         // 환경 하나의 u8 관측 바이트 수
  std::vector<int> conv_ch;   // 가지마다 같은 채널 목록 (층 수 = 길이, 3 또는 4). 넷째 층은 셋째 층 격자를 0 으로 덧대 짝수로 만든 뒤 2×2 s2
  int in1d = 0;               // 1D 입력 수 (정규화된 real)
  int grad_cols = -1;         // 입력 앞 몇 열의 기울기를 낼지 (−1 = 합성곱 평탄화 열). 공유 인코더 뒤 MLP 는 인코더 특징 열 수
  std::vector<int> hidden;    // MLP 은닉 층
  int out = 0;
};

enum class PKind { ConvW, ConvB, LinW, LinB, LogStd };
struct ParamBlock {
  long off, n;
  PKind kind;
  std::string name;
};

class Net {
 public:
  // 학습 변수 자리를 reg 에 등록하고 cursor 를 민다.
  void build(const NetSpec& s, std::vector<ParamBlock>& reg, long& cursor, const std::string& prefix);
  void bind(real* params, real* grads, int max_n);  // 버퍼를 묶고 작업 공간 할당
  const unsigned short* Ph_ = nullptr;  // 학습 변수의 bf16 사본 (ActorCritic::params_changed 가 새로; 융합 bf16 MLP 의 정방향·dX 가 W 를 여기서 읽음)
  void free_ws();
  // torch 기본 초기화 (nn.Linear·nn.Conv2d: U(±1/√fan_in), kaiming_uniform a=√5 와 같음)
  void init(std::mt19937_64& rng, std::vector<real>& host) const;
  // img: [n][img_stride] u8, x1d: [n][ld1d] (앞 in1d 개 사용), out: [n][out]
  // img_idx 를 주면 샘플 i 의 이미지 = img 의 img_idx[i] 행 (합성곱이 모두 융합 커널일 때만 — cnn_reads_index())
  void forward(cublasHandle_t h, cudaStream_t st, int n, const uint8_t* img, const real* x1d, int ld1d, real* out,
               const int* img_idx = nullptr);
  bool cnn_reads_index() const;
  // 직전 forward 와 같은 n·img 로 부른다. dout [n][out] → d_grads 에 더한다(누적).
  void backward(cublasHandle_t h, cudaStream_t st, int n, const uint8_t* img, const real* dout, const int* img_idx = nullptr);
  const NetSpec& spec() const { return s_; }
  long feat_img() const { return feat_img_; }
  long feat() const { return feat_; }
  real* x0() { return X0_; }    // [max_n][feat] MLP 입력 (앞 feat_img 열 = 합성곱 평탄화). MLP 없는 망(인코더)은 곧 출력 특징
  real* dx0() { return dX0_; }  // 역방향의 입력 기울기 [max_n][feat] (앞 grad_cols 열만 유효)
  int max_n() const { return max_n_; }
  size_t ws_bytes() const { return ws_bytes_; }
  // 합성곱 가지를 텐서 코어(bf16 입력·fp32 누산) 융합 커널로. 끄면 fp32 융합 커널(정확 경로). PPOConfig::cnn_bf16 가 정한다.
  bool cnn_bf16 = false;
  // MLP 계산 방식: 0 = cuBLAS sgemm + 편향·ELU 커널 따로(옛 길), 1 = 직접 짠 융합 GEMM fp32(정확 모드),
  // 2 = 융합 GEMM bf16 텐서 코어(fp32 누산). fp64 빌드는 늘 0.
  int mlp_mode = 0;

 private:
  struct ConvL {
    int cin, h, w, cout, oh, ow;
    long w_off, b_off;
  };
  struct Branch {
    ImgIn in;
    std::vector<ConvL> L;
    long flat_off;  // X0 안 열 위치
    real* X = nullptr;          // 첫 층 입력 (계층 배치 real) [n·h·w][c]
    std::vector<real*> A, dA;  // 층마다 출력 [n·oh·ow][cout] (계층 배치) — 기준 경로만
    bool fused = false;        // 융합 커널(cnn_fused) 로 도는 가지 (fp32 빌드, 지원하는 모양)
    // 넷째 층(꼬리): 셋째 층 출력 F3 [n][h3·w3][c] (래스터) → 덧댄 im2col Tc [n·P4][4·c] → Ty [n·P4][cout] (ELU 뒤)
    bool tail = false;
    ConvL T{};
    real *F3 = nullptr, *dF3 = nullptr, *Tc = nullptr, *dTc = nullptr, *Ty = nullptr, *dTy = nullptr;
  };
  CnnBranch desc(const Branch& B) const;
  void tail_forward(cublasHandle_t h, cudaStream_t st, int n, Branch& B);
  void tail_backward(cublasHandle_t h, cudaStream_t st, int n, Branch& B, const real* dsrc);
  struct LinL {
    int in, out;
    long w_off, b_off;
  };
  NetSpec s_;
  std::vector<Branch> br_;
  std::vector<LinL> lin_;
  long feat_img_ = 0, feat_ = 0;
  int grad_cols_ = 0;
  real *P_ = nullptr, *G_ = nullptr;
  int max_n_ = 0;
  real *X0_ = nullptr, *dX0_ = nullptr;
  real* part_ = nullptr;  // 융합 역방향의 블록별 기울기 부분합
  std::vector<real*> H_, dH_;  // MLP 은닉 출력 [n][hidden]
  size_t ws_bytes_ = 0;
};

// rsl_rl modules/normalizer.py EmpiricalNormalization (eps 1e-2, until 없음) 와 같은 식.
class Normalizer {
 public:
  void init(int dim);
  void release();
  void update(const real* x, int n, int ld, cudaStream_t st);           // 배치 평균·분산을 섞는다
  // (x−μ)/(σ+eps), 앞 d 개 차원만 (−1 = 전부)
  void apply(const real* x, int n, int ldx, real* y, int ldy, cudaStream_t st, int d = -1) const;
  int dim = 0;
  real *d_mean = nullptr, *d_var = nullptr;
  double count = 0;
  static constexpr real kEps = 1e-2f;
};

// 공유 인코더(합성곱 가지들) → 특징 F. actor·critic MLP 입력 = [F | 정규화 1D] (actor 는 정책 1D, critic 은 참값까지).
// actor·critic 의 imgs 는 비우고 in1d 는 1D 만 적는다 — ActorCritic 이 앞에 F 열을 붙인다.
struct ACSpec {
  NetSpec enc;               // 합성곱 가지만 (in1d 0, MLP 없음)
  NetSpec actor, critic;
  int n_cont = 3;            // 가우시안 성분 수 (actor 출력 앞쪽), 뒤 1 개가 선언 로짓
  real init_log_std = 0.f;  // log σ 초기값 (σ = 1, rsl_rl init_noise_std=1.0)
  real declare_bias = -7.f; // 선언 로짓 편향 초기값 (mdp/README "행동")
  int norm_dim = 0;          // 정규화 대상 1D 수 (critic 1D 전체; actor 는 앞 in1d 개를 같은 통계로)
};

// obs_layout.h 크기로 만든 탐사 RL 기본 구조
ACSpec explore_spec();

class ActorCritic {
 public:
  ActorCritic(const ACSpec& spec, uint64_t seed, int max_chunk);
  ~ActorCritic();
  ActorCritic(const ActorCritic&) = delete;
  ActorCritic& operator=(const ActorCritic&) = delete;

  void zero_grad();
  void set_mlp_mode(int m) { actor.mlp_mode = critic.mlp_mode = m; }
  // 학습 변수를 바꾼 뒤(Adam·불러오기·초기화·검사 도구의 직접 수정) 부른다 — bf16 사본을 새로 (stream 순서)
  void params_changed();
  unsigned short* d_params_h = nullptr;
  real* log_std() { return d_params + logstd_off; }
  real* log_std_grad() { return d_grads + logstd_off; }

  ACSpec spec;
  Net enc, actor, critic;  // enc = 공유 인코더 (특징 enc.x0() [n][nf]), actor·critic = MLP (입력 [F | 1D])
  int nf = 0;              // 인코더 특징 수
  Normalizer norm;
  std::vector<ParamBlock> blocks;
  long n_params = 0, logstd_off = 0;
  real *d_params = nullptr, *d_grads = nullptr;
  int max_chunk;
  cublasHandle_t cublas;
  cudaStream_t stream;
  // critic 은 두 번째 스트림에서 actor 와 같이 돈다 (쓰는 버퍼·기울기 칸이 겹치지 않음, 원자 연산 없음 → 결과는 한 스트림과 비트 단위로 같음).
  // fork(): stream 의 지금까지 일을 stream2 가 기다림, join(): stream 이 stream2 의 지금까지 일을 기다림.
  cublasHandle_t cublas2;
  cudaStream_t stream2;
  void fork();
  void join();
  cudaEvent_t ev_fork_, ev_join_;
};

}  // namespace XNET_NS
