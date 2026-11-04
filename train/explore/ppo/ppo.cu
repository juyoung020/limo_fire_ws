// 탐사 RL PPO — 롤아웃 버퍼·GAE·손실·Adam·KL 적응 학습률·체크포인트. rsl_rl 2.3.1 algorithms/ppo.py·storage/rollout_storage.py 꼴.
#include <cuda_bf16.h>
#include <curand_kernel.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <numeric>
#include <random>
#include <stdexcept>
#include <vector>

#include "../network/distribution.cuh"
#include "ppo.h"

using namespace XNET_NS;

namespace XPPO_NS {

namespace {
constexpr int MAXC = 8;  // 연속 성분 최대 수 (커널 안 고정 배열)

__global__ void k_concat(const real* a, int da, const real* b, int db, int n, real* out) {
  long idx = blockIdx.x * (long)blockDim.x + threadIdx.x;
  int D = da + db;
  if (idx >= (long)n * D) return;
  int i = (int)(idx / D), j = (int)(idx % D);
  out[idx] = j < da ? a[(long)i * da + j] : (b ? b[(long)i * db + (j - da)] : 0.f);
}

// 샘플: a = μ + σ·ε (자르지 않음), 선언 = u < sigmoid(로짓). Philox(씨앗, 환경, 호출 수) → 재현 가능
__global__ void k_sample(const real* aout, const real* log_std, int n, int nc, uint64_t seed, uint64_t call,
                         int env0, real* act_out, real* s_act, real* s_logp, real* s_mu, real* s_logit,
                         real* s_sd) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  curandStatePhilox4_32_10_t rs;
  curand_init(seed, (uint64_t)(env0 + i), call * 8ull, &rs);
  int A = nc + 1;
  real lp = 0.f;
  for (int j = 0; j < nc; ++j) {
    real mu = aout[i * A + j], ls = log_std[j];
    real a = mu + exp(ls) * curand_normal(&rs);
    act_out[i * A + j] = a;
    s_act[i * A + j] = a;
    s_mu[i * nc + j] = mu;
    s_sd[i * nc + j] = exp(ls);
    lp += dist::gauss_logp(a, mu, ls);
  }
  real l = aout[i * A + nc];
  real d = curand_uniform(&rs) < dist::sigmoid(l) ? 1.f : 0.f;
  act_out[i * A + nc] = d;
  s_act[i * A + nc] = d;
  s_logit[i] = l;
  s_logp[i] = lp + dist::bern_logp(d, l);
}

__global__ void k_mean_act(const real* aout, int n, int nc, real* act) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  int A = nc + 1;
  for (int j = 0; j < nc; ++j) act[i * A + j] = aout[i * A + j];
  act[i * A + nc] = aout[i * A + nc] > 0.f ? 1.f : 0.f;  // 확률 > 0.5 ⇔ 로짓 > 0
}

// time-out bootstrap: rsl_rl algorithms/ppo.py:169-172 (r += γ·V(s_t)·time_out)
__global__ void k_store(const real* r, const int* done, const real* val, real gamma, int n, real* s_rew,
                        real* s_done) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  int d = done[i];
  s_rew[i] = r[i] + (d == 3 ? gamma * val[i] : 0.f);
  s_done[i] = d != 0 ? 1.f : 0.f;
}

// GAE: rsl_rl storage/rollout_storage.py:145-167, 환경마다 스레드 하나가 뒤에서부터
__global__ void k_gae(const real* rew, const real* done, const real* val, const real* last_val, int T, int N,
                      real gamma, real lam, real* ret, real* adv) {
  int e = blockIdx.x * blockDim.x + threadIdx.x;
  if (e >= N) return;
  real a = 0.f;
  for (int t = T - 1; t >= 0; --t) {
    long k = (long)t * N + e;
    real nv = t == T - 1 ? last_val[e] : val[k + N];
    real nt = 1.f - done[k];
    real delta = rew[k] + nt * gamma * nv - val[k];
    a = delta + nt * gamma * lam * a;
    ret[k] = a + val[k];
    adv[k] = a;
  }
}

__global__ void k_affine(real* x, long n, real m, real inv) {
  long i = blockIdx.x * (long)blockDim.x + threadIdx.x;
  if (i < n) x[i] = (x[i] - m) * inv;
}

// 미니배치 모으기 한 번에: 정규화 1D 행 [D] + 롤아웃 값 8 가지(폭 w[g]) → dst_obs [n][ld_obs] 의 앞 D 열(fp32, 또는 dst_h 면 bf16 만), dst[g] [n][w[g]]
struct GatherSet {
  const real* src[8];
  real* dst[8];
  int w[8];
  int W;  // 폭 합
};
__global__ void k_gather_mb(const real* __restrict__ obs, int D, const int* __restrict__ idx, int n, real* dst_obs,
                            long ld_obs, __nv_bfloat16* dst_h, GatherSet gs) {
  long k = blockIdx.x * (long)blockDim.x + threadIdx.x;
  const int R = D + gs.W;
  if (k >= (long)n * R) return;
  int i = (int)(k / R), j = (int)(k % R);
  const long r = idx[i];
  if (j < D) {
    const real v = obs[r * D + j];
    if (dst_h)
      dst_h[(long)i * ld_obs + j] = __float2bfloat16_rn(v);
    else
      dst_obs[(long)i * ld_obs + j] = v;
    return;
  }
  j -= D;
  int g = 0;
#pragma unroll
  for (int t = 0; t < 7; ++t)
    if (j >= gs.w[g]) j -= gs.w[g], ++g;
  gs.dst[g][(long)i * gs.w[g] + j] = gs.src[g][r * gs.w[g] + j];
}

// u8 이미지는 4 바이트씩 (IMG 가 4 의 배수일 때)
__global__ void k_gather_img(const uint32_t* src, int w4, const int* idx, int n, uint32_t* dst) {
  long k = blockIdx.x * (long)blockDim.x + threadIdx.x;
  if (k >= (long)n * w4) return;
  int i = (int)(k / w4), j = (int)(k % w4);
  dst[k] = src[(long)idx[i] * w4 + j];
}

// 샘플마다 PPO 손실 항과 신경망 출력 기울기. scale = 1/미니배치 크기 (평균).
// stat: [surrogate, value_loss, kl, ratio] — 샘플 값(평균 전)
struct LossArgs {
  const real *aout, *v, *log_std, *act, *old_logp, *old_mu, *old_logit, *old_sd, *adv, *ret, *old_v;
  int n, nc;
  real clip, value_coef, scale;
  bool clipped_value;
  real *da, *dv, *dls, *stat;
};
__global__ void k_loss(LossArgs p) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= p.n) return;
  int nc = p.nc, A = nc + 1;
  real lp = 0.f, kl = 0.f;
  real mu[MAXC], ls[MAXC];
  for (int j = 0; j < nc; ++j) {
    mu[j] = p.aout[i * A + j];
    ls[j] = p.log_std[j];
    lp += dist::gauss_logp(p.act[i * A + j], mu[j], ls[j]);
    kl += dist::gauss_kl(p.old_mu[i * nc + j], p.old_sd[i * nc + j], mu[j], exp(ls[j]));
  }
  real l = p.aout[i * A + nc], d = p.act[i * A + nc];
  lp += dist::bern_logp(d, l);
  kl += dist::bern_kl(p.old_logit[i], l);
  real r = exp(lp - p.old_logp[i]);
  real Ad = p.adv[i];
  real rc = fmin(fmax(r, 1.f - p.clip), 1.f + p.clip);
  real s1 = -Ad * r, s2 = -Ad * rc;
  real surr = fmax(s1, s2);
  bool inside = r > 1.f - p.clip && r < 1.f + p.clip;
  real g = (inside || s1 > s2) ? -Ad * r : 0.f;  // d surr / d logπ
  g *= p.scale;
  for (int j = 0; j < nc; ++j) {
    real sd = exp(ls[j]);
    real z = (p.act[i * A + j] - mu[j]) / sd;
    p.da[i * A + j] = g * z / sd;
    p.dls[i * nc + j] = g * (z * z - 1.f);
  }
  p.da[i * A + nc] = g * (d - dist::sigmoid(l));
  // 가치: rsl_rl algorithms/ppo.py:313-322
  real v = p.v[i], R = p.ret[i], vo = p.old_v[i];
  real vl, dv;
  if (p.clipped_value) {
    real dvc = fmin(fmax(v - vo, -p.clip), p.clip);
    real vc = vo + dvc;
    real l1 = (v - R) * (v - R), l2 = (vc - R) * (vc - R);
    vl = fmax(l1, l2);
    bool in = fabs(v - vo) < p.clip;
    dv = l1 >= l2 ? 2.f * (v - R) : (in ? 2.f * (vc - R) : 0.f);
  } else {
    vl = (v - R) * (v - R);
    dv = 2.f * (v - R);
  }
  p.dv[i] = dv * p.value_coef * p.scale;
  p.stat[i * 4 + 0] = surr;
  p.stat[i * 4 + 1] = vl;
  p.stat[i * 4 + 2] = kl;
  p.stat[i * 4 + 3] = r;
}

// MLP 입력 [F | 1D] 의 1D 열: dst[i·ldd + j] = x[i·ldx + j] (j < d), dst_h 면 bf16 만. F 열은 인코더가 바로 쓴다
__global__ void k_put_1d(const real* __restrict__ x, int ldx, int d, int n, real* dst, __nv_bfloat16* dst_h, long ldd) {
  long k = blockIdx.x * (long)blockDim.x + threadIdx.x;
  if (k >= (long)n * d) return;
  int i = (int)(k / d), j = (int)(k % d);
  const real v = x[(long)i * ldx + j];
  if (dst_h)
    dst_h[(long)i * ldd + j] = __float2bfloat16_rn(v);
  else
    dst[(long)i * ldd + j] = v;
}

// log σ 변수를 [lo, hi] 로 투영 (Adam 걸음·불러오기 뒤) — ppo.h PPOConfig std_min/std_max
__global__ void k_clamp_logstd(real* ls, int n, real lo, real hi) {
  int i = threadIdx.x;
  if (i < n) ls[i] = fmin(fmax(ls[i], lo), hi);
}

__global__ void k_add_const(real* x, int n, real c) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) x[i] += c;
}

// torch.optim.Adam (amsgrad 없음, weight decay 없음) + 기울기 노름 자르기 배율
__global__ void k_adam(real* p, const real* g, real* m, real* v, long n, real lr, real b1, real b2, real eps,
                       real bc1, real bc2, real gscale) {
  long i = blockIdx.x * (long)blockDim.x + threadIdx.x;
  if (i >= n) return;
  real gi = g[i] * gscale;
  real mi = b1 * m[i] + (1.f - b1) * gi;
  real vi = b2 * v[i] + (1.f - b2) * gi * gi;
  m[i] = mi;
  v[i] = vi;
  real denom = sqrt(vi) / sqrt(bc2) + eps;
  p[i] -= (lr / bc1) * mi / denom;
}

template <typename T>
T* dmalloc(long n, size_t& acc) {
  T* p;
  CK(cudaMalloc(&p, n * sizeof(T)));
  acc += n * sizeof(T);
  return p;
}
}  // namespace

PPO::PPO(const PPOConfig& cfg, const ACSpec& spec, uint64_t seed) : cfg_(cfg), seed_(seed), lr_(cfg.lr) {
  if (cfg_.chunk <= 0) cfg_.chunk = std::max(1, cfg_.num_envs * cfg_.steps_per_env / std::max(1, cfg_.minibatches));  // 미니배치 하나
  if (spec.n_cont > MAXC) throw std::runtime_error("n_cont > MAXC");
  ac_ = new ActorCritic(spec, seed, cfg_.chunk);
  CKB(cublasSetMathMode(ac_->cublas, cfg.tf32 ? CUBLAS_TF32_TENSOR_OP_MATH : CUBLAS_DEFAULT_MATH));
  bool bf = cfg.cnn_bf16;
  if (const char* e = std::getenv("XNET_CNN_BF16")) bf = std::atoi(e) != 0;  // TODO(제거): 정밀도 A/B 실측용
  ac_->enc.cnn_bf16 = bf;
  CKB(cublasSetMathMode(ac_->cublas2, cfg.tf32 ? CUBLAS_TF32_TENSOR_OP_MATH : CUBLAS_DEFAULT_MATH));
  ac_->set_mlp_mode(sizeof(real) == 4 ? cfg.mlp : 0);
  if (!(cfg.std_min > 0.f && cfg.std_min <= cfg.std_max)) throw std::runtime_error("PPOConfig: 0 < std_min ≤ std_max 이어야 함");
  if (spec.init_log_std < std::log(cfg.std_min) || spec.init_log_std > std::log(cfg.std_max))
    throw std::runtime_error("ACSpec init_log_std 가 PPOConfig σ 범위 밖");
  N_ = cfg.num_envs;
  T_ = cfg.steps_per_env;
  S_ = N_ * T_;
  D_ = spec.norm_dim;
  IMG_ = spec.enc.img_stride;
  NF_ = ac_->nf;
  NC_ = spec.n_cont;
  A_ = NC_ + 1;
  if (IMG_ % 4) throw std::runtime_error("img_stride 는 4 의 배수여야 함");
  if (S_ % cfg.minibatches) throw std::runtime_error("N·T 가 minibatches 로 안 나뉨");
  size_t& b = storage_bytes_;
  long P = ac_->n_params;
  m_ = dmalloc<real>(P, b);
  v_ = dmalloc<real>(P, b);
  CK(cudaMemset(m_, 0, P * sizeof(real)));
  CK(cudaMemset(v_, 0, P * sizeof(real)));
  s_img_ = dmalloc<uint8_t>((long)S_ * IMG_, b);
  s_obs_ = dmalloc<real>((long)S_ * D_, b);
  s_act_ = dmalloc<real>((long)S_ * A_, b);
  s_logp_ = dmalloc<real>(S_, b);
  s_val_ = dmalloc<real>(S_, b);
  s_mu_ = dmalloc<real>((long)S_ * NC_, b);
  s_logit_ = dmalloc<real>(S_, b);
  s_sd_ = dmalloc<real>((long)S_ * NC_, b);
  s_rew_ = dmalloc<real>(S_, b);
  s_done_ = dmalloc<real>(S_, b);
  s_ret_ = dmalloc<real>(S_, b);
  s_adv_ = dmalloc<real>(S_, b);
  int C = cfg_.chunk;
  raw_ = dmalloc<real>((long)N_ * D_, b);
  last_obs_ = dmalloc<real>((long)N_ * D_, b);
  last_val_ = dmalloc<real>(N_, b);
  c_img_ = dmalloc<uint8_t>((long)C * IMG_, b);
  c_aout_ = dmalloc<real>((long)C * A_, b);
  c_v_ = dmalloc<real>(C, b);
  c_da_ = dmalloc<real>((long)C * A_, b);
  c_dv_ = dmalloc<real>(C, b);
  c_dls_ = dmalloc<real>((long)C * NC_, b);
  c_stat_ = dmalloc<real>((long)C * 4, b);
  c_red_ = dmalloc<real>(16, b);
  c_x_ = dmalloc<real>((long)C * (NF_ + D_), b);
  // bf16 MLP(mlp 2)는 첫 층이 cuBLAS bf16 이라 입력 bf16 사본 c_xh_ 를 둔다 (1D 열은 사본에만, F 열은 둘 다 — fp32 는 넷째 층 ELU' 용)
  if (sizeof(real) == 4 && cfg_.mlp == 2) c_xh_ = dmalloc<uint16_t>((long)C * (NF_ + D_), b);
  ac_->enc.set_x0(c_x_, NF_ + D_, c_xh_);  // 인코더가 특징을 MLP 입력 [F | 1D] 의 앞 열에 바로 쓴다 (모으는 복사 없음)
  c_gat_ = dmalloc<real>((long)C * (A_ + 5 + 2 * NC_), b);
  d_perm_ = dmalloc<int>(S_, b);
}

PPO::~PPO() {
  for (void* p : std::initializer_list<void*>{m_, v_, s_img_, s_obs_, s_act_, s_logp_, s_val_, s_mu_, s_logit_, s_sd_,
                                              s_rew_, s_done_, s_ret_, s_adv_, raw_, last_obs_, last_val_, c_img_,
                                              c_aout_, c_v_, c_da_, c_dv_, c_dls_, c_stat_, c_red_, c_gat_, d_perm_, c_x_, c_xh_})
    cudaFree(p);
  delete ac_;
}

// 공유 인코더 정방향 → c_x_ = [F | x1d (D_ 열)] (actor 는 앞 nf + 정책 1D 열, critic 은 전부).
// x1d = nullptr 이면 1D 열은 이미 c_x_ 에 있음 (미니배치 모으기가 바로 씀)
void PPO::encode(int n, const uint8_t* img, const int* idx, const real* x1d) {
  auto st = ac_->stream;
  ac_->enc.forward(ac_->cublas, st, n, img, nullptr, 0, nullptr, idx);
  if (x1d)
    k_put_1d<<<nblk((long)n * D_), 256, 0, st>>>(x1d, D_, D_, n, c_x_ + NF_, c_xh_ ? (__nv_bfloat16*)c_xh_ + NF_ : nullptr, NF_ + D_);
}

void PPO::concat_norm(const real* obs1d, const real* priv, int n, bool upd, real* out) {
  auto st = ac_->stream;
  int da = ac_->spec.actor.in1d, db = D_ - da;
  k_concat<<<nblk((long)n * D_), 256, 0, st>>>(obs1d, da, priv, db, n, raw_);
  if (upd) ac_->norm.update(raw_, n, D_, st);
  ac_->norm.apply(raw_, n, D_, out, D_, st);
}

void PPO::act(const uint8_t* img, const real* obs1d, const real* priv, real* actions) {
  if (step_ >= T_) throw std::runtime_error("PPO::act: 롤아웃이 가득 참 — update 먼저");
  auto st = ac_->stream;
  long k0 = (long)step_ * N_;
  real* obs = s_obs_ + k0 * D_;
  concat_norm(obs1d, priv, N_, true, obs);
  CK(cudaMemcpyAsync(s_img_ + k0 * IMG_, img, (size_t)N_ * IMG_, cudaMemcpyDeviceToDevice, st));
  for (int c0 = 0; c0 < N_; c0 += cfg_.chunk) {
    int n = std::min(cfg_.chunk, N_ - c0);
    const uint8_t* im = s_img_ + (k0 + c0) * IMG_;
    const real* ob = obs + (long)c0 * D_;
    encode(n, im, nullptr, ob);
    ac_->fork();
    ac_->critic.forward(ac_->cublas2, ac_->stream2, n, nullptr, c_x_, NF_ + D_, s_val_ + k0 + c0, nullptr, c_xh_);
    ac_->actor.forward(ac_->cublas, st, n, nullptr, c_x_, NF_ + D_, c_aout_, nullptr, c_xh_);
    ac_->join();
    long k = k0 + c0;
    k_sample<<<nblk(n, 128), 128, 0, st>>>(c_aout_, ac_->log_std(), n, NC_, seed_, act_calls_, c0, actions + (long)c0 * A_,
                                            s_act_ + k * A_, s_logp_ + k, s_mu_ + k * NC_, s_logit_ + k,
                                            s_sd_ + k * NC_);
  }
  ++act_calls_;
}

void PPO::last_act_info(int n, real* logit, real* value) const {
  if (n > N_) throw std::runtime_error("last_act_info: n > num_envs");
  const long k0 = (long)step_ * N_;
  CK(cudaMemcpyAsync(logit, s_logit_ + k0, n * sizeof(real), cudaMemcpyDeviceToHost, ac_->stream));
  CK(cudaMemcpyAsync(value, s_val_ + k0, n * sizeof(real), cudaMemcpyDeviceToHost, ac_->stream));
  CK(cudaStreamSynchronize(ac_->stream));
}

void PPO::store(const real* reward, const int* done) {
  long k0 = (long)step_ * N_;
  k_store<<<nblk(N_, 128), 128, 0, ac_->stream>>>(reward, done, s_val_ + k0, cfg_.gamma, N_, s_rew_ + k0,
                                                  s_done_ + k0);
  ++step_;
}

void PPO::act_inference(const uint8_t* img, const real* obs1d, real* actions, int n_all) {
  if (n_all < 0) n_all = N_;
  if (n_all > N_) throw std::runtime_error("act_inference: n > num_envs");
  auto st = ac_->stream;
  int da = ac_->spec.actor.in1d;
  // 정책 1D 만 정규화 (같은 통계의 앞 da 차원)
  ac_->norm.apply(obs1d, n_all, da, raw_, D_, st, da);
  for (int c0 = 0; c0 < n_all; c0 += cfg_.chunk) {
    int n = std::min(cfg_.chunk, n_all - c0);
    encode(n, img + (long)c0 * IMG_, nullptr, raw_ + (long)c0 * D_);
    ac_->actor.forward(ac_->cublas, st, n, nullptr, c_x_, NF_ + D_, c_aout_, nullptr, c_xh_);
    k_mean_act<<<nblk(n, 128), 128, 0, st>>>(c_aout_, n, NC_, actions + (long)c0 * A_);
  }
}

void PPO::compute_returns(const uint8_t* img, const real* obs1d, const real* priv) {
  if (step_ != T_) throw std::runtime_error("PPO::compute_returns: 롤아웃이 덜 참");
  auto st = ac_->stream;
  concat_norm(obs1d, priv, N_, false, last_obs_);
  for (int c0 = 0; c0 < N_; c0 += cfg_.chunk) {
    int n = std::min(cfg_.chunk, N_ - c0);
    encode(n, img + (long)c0 * IMG_, nullptr, last_obs_ + (long)c0 * D_);
    ac_->critic.forward(ac_->cublas, st, n, nullptr, c_x_, NF_ + D_, last_val_ + c0, nullptr, c_xh_);
  }
  k_gae<<<nblk(N_, 128), 128, 0, st>>>(s_rew_, s_done_, s_val_, last_val_, T_, N_, cfg_.gamma, cfg_.lam, s_ret_,
                                       s_adv_);
  // 이점 정규화 (전체, torch.std 는 불편 추정) — rollout_storage.py:166-167
  double s = sum_all(s_adv_, S_, st), ss = sumsq_all(s_adv_, S_, st);
  double m = s / S_;
  double var = (ss - S_ * m * m) / std::max(1, S_ - 1);
  real inv = (real)(1.0 / (std::sqrt(std::max(var, 0.0)) + 1e-8));
  k_affine<<<nblk(S_), 256, 0, st>>>(s_adv_, S_, (real)m, inv);
}

// 미니배치 하나 (d_idx 의 n 개, 평균 분모 mb) 손실 → 기울기 누적. 반환 = 손실 (double), stats 에 [surr, value, kl, ratio] 합
double PPO::minibatch(const int* d_idx, int n_all, int mb, bool backward, double stats[4]) {
  auto st = ac_->stream;
  auto& ac = *ac_;
  std::fill(stats, stats + 4, 0.0);
  std::vector<real> hs(4);
  for (int c0 = 0; c0 < n_all; c0 += cfg_.chunk) {
    int n = std::min(cfg_.chunk, n_all - c0);
    const int* ix = d_idx + c0;
    // 합성곱이 융합 커널이면 롤아웃 이미지를 색인으로 바로 읽는다 (모으기 복사 없음)
    const bool by_idx = ac.enc.cnn_reads_index();
    const uint8_t* im = by_idx ? s_img_ : c_img_;
    const int* imx = by_idx ? ix : nullptr;
    if (!by_idx)
      k_gather_img<<<nblk((long)n * IMG_ / 4), 256, 0, st>>>((const uint32_t*)s_img_, IMG_ / 4, ix, n,
                                                              (uint32_t*)c_img_);
    // 1D 관측과 나머지 롤아웃 값(c_* 작업 공간 하나에 몰아)을 커널 하나로 모은다
    GatherSet gs{{s_act_, s_logp_, s_mu_, s_logit_, s_sd_, s_adv_, s_ret_, s_val_}, {}, {A_, 1, NC_, 1, NC_, 1, 1, 1}, 0};
    real* off[8];
    real* cur = c_gat_;
    for (int g = 0; g < 8; ++g) {
      off[g] = gs.dst[g] = cur;
      cur += (long)n * gs.w[g];
      gs.W += gs.w[g];
    }
    k_gather_mb<<<nblk((long)n * (D_ + gs.W)), 256, 0, st>>>(s_obs_, D_, ix, n, c_x_ + NF_, NF_ + D_,
                                                                   c_xh_ ? (__nv_bfloat16*)c_xh_ + NF_ : nullptr, gs);
    encode(n, im, imx, nullptr);
    ac.fork();
    ac.critic.forward(ac.cublas2, ac.stream2, n, nullptr, c_x_, NF_ + D_, c_v_, nullptr, c_xh_);
    ac.actor.forward(ac.cublas, st, n, nullptr, c_x_, NF_ + D_, c_aout_, nullptr, c_xh_);
    ac.join();
    LossArgs p{c_aout_, c_v_, ac.log_std(), off[0], off[1], off[2], off[3], off[4], off[5], off[6], off[7], n, NC_,
               cfg_.clip, cfg_.value_coef, 1.f / mb, cfg_.clipped_value, c_da_, c_dv_, c_dls_, c_stat_};
    k_loss<<<nblk(n, 128), 128, 0, st>>>(p);
    if (backward) {
      ac.fork();
      ac.critic.backward(ac.cublas2, ac.stream2, n, nullptr, c_dv_);
      ac.actor.backward(ac.cublas, st, n, nullptr, c_da_);
      ac.join();
      // 공유 인코더: 두 MLP 의 특징 기울기(actor + critic, 고정 순서)를 합성곱 역방향으로 (rsl_rl·공유 인코더 PPO 와 같은 합 손실).
      // 더하기는 넷째 층 역방향이 읽으면서 한다 (모으는 버퍼 없음)
      ac.enc.backward_feat(ac.cublas, st, n, im, imx, ac.actor.dx0(), ac.actor.feat(), ac.critic.dx0(), ac.critic.feat());
      col_sum(c_dls_, n, NC_, NC_, ac.log_std_grad(), true, st);
    }
    col_sum(c_stat_, n, 4, 4, c_red_, false, st);
    CK(cudaMemcpyAsync(hs.data(), c_red_, 4 * sizeof(real), cudaMemcpyDeviceToHost, st));
    CK(cudaStreamSynchronize(st));
    for (int k = 0; k < 4; ++k) stats[k] += hs[k];
  }
  // 엔트로피(가우시안만, 상태 무관): −c·Σ(½+½ln2π+logσ) → d/dlogσ = −c
  std::vector<real> ls(NC_);
  CK(cudaMemcpy(ls.data(), ac.log_std(), NC_ * sizeof(real), cudaMemcpyDeviceToHost));
  double ent = 0;
  for (real l : ls) ent += 0.5 + 0.91893853320467274 + l;
  if (backward) k_add_const<<<1, 32, 0, st>>>(ac.log_std_grad(), NC_, -cfg_.entropy_coef * ((real)n_all / mb));
  stats[3] = ent * n_all;  // 엔트로피 합 자리로 바꿔 쓴다 (ratio 는 안 씀)
  return stats[0] / mb + cfg_.value_coef * stats[1] / mb - cfg_.entropy_coef * ent * n_all / mb;
}

double PPO::loss_and_grad(const int* d_idx, int n, bool backward) {
  ac_->params_changed();  // 검사 도구가 d_params 를 직접 바꾼다
  if (backward) ac_->zero_grad();
  double st[4];
  return minibatch(d_idx, n, n, backward, st);
}

UpdateStats PPO::update(const uint8_t* img, const real* obs1d, const real* priv) {
  compute_returns(img, obs1d, priv);
  auto st = ac_->stream;
  UpdateStats out;
  {
    double sv = sum_all(s_val_, S_, st), sr = sum_all(s_ret_, S_, st);
    out.mean_value = (real)(sv / S_);
    out.mean_return = (real)(sr / S_);
  }
  // rsl_rl mini_batch_generator: 반복마다 randperm 하나를 모든 에포크가 같이 씀
  std::vector<int> perm(S_);
  std::iota(perm.begin(), perm.end(), 0);
  std::mt19937_64 rng(seed_ * 0x9E3779B97F4A7C15ull + (uint64_t)iter_ * 1315423911ull + 7);
  std::shuffle(perm.begin(), perm.end(), rng);
  CK(cudaMemcpyAsync(d_perm_, perm.data(), S_ * sizeof(int), cudaMemcpyHostToDevice, st));
  int mb = S_ / cfg_.minibatches;
  double acc_s = 0, acc_v = 0, acc_e = 0, acc_kl = 0, acc_gn = 0;
  int nup = 0;
  for (int ep = 0; ep < cfg_.epochs; ++ep) {
    for (int b = 0; b < cfg_.minibatches; ++b) {
      ac_->zero_grad();
      double s[4];
      minibatch(d_perm_ + (long)b * mb, mb, mb, true, s);
      double kl = s[2] / mb;
      // KL 적응 학습률 — rsl_rl algorithms/ppo.py:269-293 (미니배치마다, 이번 걸음부터 적용)
      if (cfg_.desired_kl > 0) {
        if (kl > cfg_.desired_kl * 2.0)
          lr_ = std::max(cfg_.lr_min, lr_ / 1.5f);
        else if (kl < cfg_.desired_kl / 2.0 && kl > 0.0)
          lr_ = std::min(cfg_.lr_max, lr_ * 1.5f);
      }
      // 기울기 노름 자르기 — torch clip_grad_norm_ (clip_coef = max/(norm+1e-6), 1 로 상한)
      double gn = std::sqrt(sumsq_all(ac_->d_grads, ac_->n_params, st));
      real gscale = (real)std::min(1.0, cfg_.max_grad_norm / (gn + 1e-6));
      ++adam_t_;
      real bc1 = 1.f - std::pow(cfg_.adam_b1, (real)adam_t_), bc2 = 1.f - std::pow(cfg_.adam_b2, (real)adam_t_);
      k_adam<<<nblk(ac_->n_params), 256, 0, st>>>(ac_->d_params, ac_->d_grads, m_, v_, ac_->n_params, lr_,
                                                  cfg_.adam_b1, cfg_.adam_b2, cfg_.adam_eps, bc1, bc2, gscale);
      clamp_std();  // σ 범위 투영 (params_changed 가 bf16 사본을 만들기 전에)
      ac_->params_changed();
      acc_s += s[0] / mb;
      acc_v += s[1] / mb;
      acc_e += s[3] / mb;
      acc_kl += kl;
      acc_gn += gn;
      ++nup;
    }
  }
  CK(cudaStreamSynchronize(st));
  out.surrogate = (real)(acc_s / nup);
  out.value = (real)(acc_v / nup);
  out.entropy = (real)(acc_e / nup);
  out.kl = (real)(acc_kl / nup);
  out.grad_norm = (real)(acc_gn / nup);
  out.lr = lr_;
  step_ = 0;
  ++iter_;
  return out;
}

// ---------------- 체크포인트 ----------------
namespace {
constexpr uint32_t kMagic = 0x4F505058;  // "XPPO"
constexpr uint32_t kVersion = 2;  // 2: 공유 인코더·4 층 합성곱 (1 = 망마다 3 층 합성곱 — 불러오기 거부)
}  // namespace

void PPO::save(const std::string& path) const {
  if (step_ != 0) throw std::runtime_error("PPO::save: 롤아웃 도중엔 저장 안 함");
  CK(cudaStreamSynchronize(ac_->stream));
  long P = ac_->n_params;
  int D = ac_->norm.dim;
  std::vector<real> p(P), m(P), v(P), nm(D), nv(D);
  CK(cudaMemcpy(p.data(), ac_->d_params, P * sizeof(real), cudaMemcpyDeviceToHost));
  CK(cudaMemcpy(m.data(), m_, P * sizeof(real), cudaMemcpyDeviceToHost));
  CK(cudaMemcpy(v.data(), v_, P * sizeof(real), cudaMemcpyDeviceToHost));
  CK(cudaMemcpy(nm.data(), ac_->norm.d_mean, D * sizeof(real), cudaMemcpyDeviceToHost));
  CK(cudaMemcpy(nv.data(), ac_->norm.d_var, D * sizeof(real), cudaMemcpyDeviceToHost));
  std::string tmp = path + ".tmp";
  {
    std::ofstream f(tmp, std::ios::binary);
    if (!f) throw std::runtime_error("체크포인트 열기 실패: " + tmp);
    auto w = [&](const void* d, size_t n) { f.write((const char*)d, n); };
    uint32_t hdr[3] = {kMagic, kVersion, (uint32_t)sizeof(real)};
    w(hdr, 12);
    int32_t nblocks = (int32_t)ac_->blocks.size();
    w(&nblocks, 4);
    for (auto& b : ac_->blocks) {  // 구조 확인용: 이름·크기
      int32_t ln = (int32_t)b.name.size();
      w(&ln, 4);
      w(b.name.data(), ln);
      int64_t n = b.n;
      w(&n, 8);
    }
    int64_t P64 = P;
    w(&P64, 8);
    w(p.data(), P * sizeof(real));
    w(m.data(), P * sizeof(real));
    w(v.data(), P * sizeof(real));
    int64_t at = adam_t_;
    w(&at, 8);
    int32_t D32 = D;
    w(&D32, 4);
    w(nm.data(), D * sizeof(real));
    w(nv.data(), D * sizeof(real));
    double cnt = ac_->norm.count;
    w(&cnt, 8);
    int32_t it = iter_;
    w(&it, 4);
    real lr = lr_;
    w(&lr, 4);
    uint64_t ac = act_calls_, sd = seed_;
    w(&ac, 8);
    w(&sd, 8);
    if (!f) throw std::runtime_error("체크포인트 쓰기 실패");
  }
  if (std::rename(tmp.c_str(), path.c_str()) != 0) throw std::runtime_error("체크포인트 이름 바꾸기 실패");
}

void PPO::clamp_std() {
  k_clamp_logstd<<<1, 32, 0, ac_->stream>>>(ac_->log_std(), NC_, (real)std::log(cfg_.std_min), (real)std::log(cfg_.std_max));
  CK(cudaGetLastError());
}

void PPO::load(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) throw std::runtime_error("체크포인트 없음: " + path);
  auto r = [&](void* d, size_t n) {
    f.read((char*)d, n);
    if (!f) throw std::runtime_error("체크포인트가 잘림: " + path);
  };
  uint32_t hdr[3];
  r(hdr, 12);
  if (hdr[0] != kMagic || hdr[1] != kVersion || hdr[2] != sizeof(real)) throw std::runtime_error("체크포인트 형식이 다름");
  int32_t nb;
  r(&nb, 4);
  if (nb != (int32_t)ac_->blocks.size()) throw std::runtime_error("체크포인트 구조(층 수)가 다름");
  for (auto& b : ac_->blocks) {
    int32_t ln;
    r(&ln, 4);
    std::string nm(ln, '\0');
    r(nm.data(), ln);
    int64_t n;
    r(&n, 8);
    if (nm != b.name || n != b.n) throw std::runtime_error("체크포인트 층 다름: " + nm + " vs " + b.name);
  }
  int64_t P;
  r(&P, 8);
  if (P != ac_->n_params) throw std::runtime_error("체크포인트 변수 수 다름");
  std::vector<real> p(P), m(P), v(P);
  r(p.data(), P * sizeof(real));
  r(m.data(), P * sizeof(real));
  r(v.data(), P * sizeof(real));
  int64_t at;
  r(&at, 8);
  int32_t D;
  r(&D, 4);
  if (D != ac_->norm.dim) throw std::runtime_error("정규화 차원 다름");
  std::vector<real> nm(D), nv(D);
  r(nm.data(), D * sizeof(real));
  r(nv.data(), D * sizeof(real));
  double cnt;
  r(&cnt, 8);
  int32_t it;
  r(&it, 4);
  real lr;
  r(&lr, 4);
  uint64_t acn, sd;
  r(&acn, 8);
  r(&sd, 8);
  CK(cudaMemcpy(ac_->d_params, p.data(), P * sizeof(real), cudaMemcpyHostToDevice));
  CK(cudaMemcpy(m_, m.data(), P * sizeof(real), cudaMemcpyHostToDevice));
  CK(cudaMemcpy(v_, v.data(), P * sizeof(real), cudaMemcpyHostToDevice));
  CK(cudaMemcpy(ac_->norm.d_mean, nm.data(), D * sizeof(real), cudaMemcpyHostToDevice));
  CK(cudaMemcpy(ac_->norm.d_var, nv.data(), D * sizeof(real), cudaMemcpyHostToDevice));
  ac_->norm.count = cnt;
  clamp_std();  // 옛 체크포인트(σ 상한 전, explore_v3 σ 99.6)도 범위 안으로
  ac_->params_changed();
  adam_t_ = at;
  iter_ = it;
  lr_ = lr;
  act_calls_ = acn;
  seed_ = sd;
  step_ = 0;
}

}  // namespace XPPO_NS
