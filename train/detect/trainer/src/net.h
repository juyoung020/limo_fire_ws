// YOLO26 탐지 신경망(정방향·역방향). Ultralytics yolo26.yaml + C3k2·C3k·SPPF·C2PSA·PSABlock·Detect(end2end) 와
// 같은 계산을 한다. 이름·모양은 PyTorch state_dict 와 같다(model.<층>.…).
//
// 연산은 차례대로 쌓아 두고 정방향은 앞에서, 역방향은 뒤에서 돈다. 활성값·기울기는 NHWC act_t, 기울기는 스텝마다
// 0 으로 지우고 모든 역방향이 더한다. 합성곱은 cuDNN, 나머지는 kern.cu·attn.cu.
#pragma once
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "conv.h"
#include "st.h"

struct Tensor {
  int N = 0, H = 0, W = 0, C = 0;
  act_t* x = nullptr;
  act_t* g = nullptr;
  int64_t M() const { return int64_t(N) * H * W; }
  int64_t n() const { return M() * C; }
};

// 파라미터: 값(fp32)·기울기(fp32), 합성곱 가중치는 act_t 사본과 cuDNN wgrad 출력(act_t)도.
enum PKind { P_CONV = 0, P_BN_W, P_BN_B, P_BIAS, P_RUN_MEAN, P_RUN_VAR };
struct Param {
  std::string name;
  PKind kind;
  std::vector<int64_t> shape;  // PyTorch 모양
  int64_t off = 0, n = 0;      // vals/grads 안 위치
  int64_t woff = -1;           // P_CONV: w16/dw16 안 위치(KRSC)
};
struct Params {
  std::vector<Param> list;
  std::map<std::string, int> idx;
  float* vals = nullptr;   // [total]
  float* grads = nullptr;  // [total] (running 통계 자리는 안 씀)
  act_t* w16 = nullptr;    // 합성곱 가중치 act_t (KRSC)
  act_t* dw16 = nullptr;
  int64_t total = 0, wtotal = 0;
  int add(const std::string& name, PKind k, std::vector<int64_t> shape);
  void allocate();
  float* v(const std::string& n) { return vals + list[idx.at(n)].off; }
  float* gr(const std::string& n) { return grads + list[idx.at(n)].off; }
  const Param& p(const std::string& n) const { return list[idx.at(n)]; }
  // safetensors(PyTorch 이름·모양, float32)에서 불러온다. 모양이 다른 것(클래스 수가 다른 분류층 등)은 건너뛰고
  // 그 이름을 돌려준다.
  std::vector<std::string> load(const SafeTensors& st, const std::string& prefix = "");
  void save(const std::string& path, bool with_grads = false) const;
  void sync_w16(cudaStream_t s);  // fp32 → act_t (KRSC)
  void gather_dw(cudaStream_t s, bool accum);  // dw16 (KRSC) → grads (PyTorch 배치)
};

class Op {
 public:
  virtual ~Op() = default;
  virtual void fwd(bool train) = 0;
  virtual void bwd() = 0;
  virtual std::string desc() const = 0;
  float ms_f = 0, ms_b = 0;
};

std::unique_ptr<Op> make_attn_op(Tensor* qkv, Tensor* o, Tensor* vf, int nh, int kd, int hd, cublasHandle_t blas,
                                 cudaStream_t s);

struct NetCfg {
  int batch = 16, imgsz = 640, nc = 7;
  float bn_momentum = 0.03f, bn_eps = 1e-3f;
};

class Net {
 public:
  void build(const NetCfg& cfg, cudnnHandle_t dnn, cublasHandle_t blas, cudaStream_t s);
  void forward(bool train);  // 입력: input()->x (NHWC, 0..1)
  void backward();           // 머리 출력 기울기(head_*()->g) → 파라미터 기울기(grads, 더하기)
  void zero_act_grads();
  Tensor* input() { return in_; }
  // 머리 출력: br 0 = one2many, 1 = one2one; lvl 0..2 (P3·P4·P5). box [B,H,W,4] (ltrb, 격자 단위), cls [B,H,W,nc]
  Tensor* head_box(int br, int lvl) { return hbox_[br][lvl]; }
  Tensor* head_cls(int br, int lvl) { return hcls_[br][lvl]; }
  Params params;
  NetCfg cfg;
  int strides[3] = {8, 16, 32};
  void* ws = nullptr;
  size_t ws_bytes = 0;
  bool train_bn = true;
  void print_time() const;
  std::vector<std::unique_ptr<Op>>& ops() { return ops_; }
  // 이름 붙은 활성값(정합성 시험용)
  std::map<std::string, Tensor*> named;

  // ---- 쌓기 도우미(내부) ----
  Tensor* tensor(int N, int H, int W, int C, bool grad = true);
  Tensor* conv_bn(const std::string& name, Tensor* in, int K, int k, int s, int act, int groups = 1, bool detach = false);
  Tensor* conv_bias(const std::string& name, Tensor* in, int K, bool detach = false);
  Tensor* cat(std::vector<Tensor*> ins);
  Tensor* slice(Tensor* in, int c0, int n);
  Tensor* add(Tensor* a, Tensor* b);
  Tensor* up2(Tensor* in);
  Tensor* pool(Tensor* in, int k);
  Tensor* bottleneck(const std::string& name, Tensor* x, int c2, bool shortcut, int k1, int k2, float e);
  Tensor* c3k(const std::string& name, Tensor* x, int c2, int n, bool shortcut);
  Tensor* c3k2(const std::string& name, Tensor* x, int c2, int n, bool c3k_, float e, bool attn);
  Tensor* psa(const std::string& name, Tensor* x);
  Tensor* sppf(const std::string& name, Tensor* x, int c2);
  Tensor* c2psa(const std::string& name, Tensor* x);
  void detect(const std::string& name, Tensor* p3, Tensor* p4, Tensor* p5);

 private:
  cudnnHandle_t dnn_ = nullptr;
  cublasHandle_t blas_ = nullptr;
  cudaStream_t s_ = nullptr;
  std::vector<std::unique_ptr<Op>> ops_;
  std::vector<std::unique_ptr<Tensor>> tensors_;
  Tensor* in_ = nullptr;
  Tensor* hbox_[2][3] = {};
  Tensor* hcls_[2][3] = {};
  double* acc_ = nullptr;  // 채널 합 작업 공간
  act_t* dz_ = nullptr;    // BN 역방향 출력(합성곱 기울기 입력) 공용 버퍼
  float* dwtmp_ = nullptr; // depthwise 가중치 기울기 누적
  act_t* grad_arena_ = nullptr;
  int64_t grad_arena_n_ = 0;
  friend class OpConvBN;
  friend class OpConvBias;
  friend class OpAttn;
  void finalize();
};
