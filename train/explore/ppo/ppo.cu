// 탐사 RL PPO — 롤아웃 버퍼·GAE·손실·Adam·KL 적응 학습률·체크포인트. rsl_rl 2.3.1 algorithms/ppo.py·storage/rollout_storage.py 꼴.
#include <cuda_bf16.h>

#include "../mdp/obs_layout.h"
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

// 평가용 샘플 행동 (학습 롤아웃 칸·난수 계수를 안 건드림): 연속 = μ + σ·N(0,1), 선언 = Bernoulli(sigmoid(로짓)).
// 난수는 학습(k_sample: seed, 부분열 env, 오프셋 call·8)과 겹치지 않게 seed 를 바꾼 계열.
__global__ void k_sample_act(const real* aout, const real* log_std, int n, int nc, uint64_t seed, uint64_t call, int env0,
                             real* act) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  curandStatePhilox4_32_10_t rs;
  curand_init(seed, (uint64_t)(env0 + i), call * 8ull, &rs);
  int A = nc + 1;
  for (int j = 0; j < nc; ++j) act[i * A + j] = aout[i * A + j] + exp(log_std[j]) * curand_normal(&rs);
  act[i * A + nc] = curand_uniform(&rs) < dist::sigmoid(aout[i * A + nc]) ? 1.f : 0.f;
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
  real bound_coef, bound_soft;  // 평균 행동 경계 손실 (PPOConfig::bound_coef) — 0 이면 안 씀
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
  // 평균 행동 경계 손실 (rl_games a2c_continuous.py bound_loss): c·Σ_j [max(μ_j − b, 0)² + min(μ_j + b, 0)²].
  // 손실 값은 surrogate 통계에 더한다(기울기 검사가 손실·기울기를 같이 보게).
  real bl = (real)0;
  if (p.bound_coef > (real)0)
    for (int j = 0; j < nc; ++j) {
      const real hi = fmax(mu[j] - p.bound_soft, (real)0), lo = fmin(mu[j] + p.bound_soft, (real)0);
      bl += p.bound_coef * (hi * hi + lo * lo);
    }
  bool inside = r > 1.f - p.clip && r < 1.f + p.clip;
  real g = (inside || s1 > s2) ? -Ad * r : 0.f;  // d surr / d logπ
  g *= p.scale;
  for (int j = 0; j < nc; ++j) {
    real sd = exp(ls[j]);
    real z = (p.act[i * A + j] - mu[j]) / sd;
    p.da[i * A + j] = g * z / sd;
    if (p.bound_coef > (real)0) {
      const real hi = fmax(mu[j] - p.bound_soft, (real)0), lo = fmin(mu[j] + p.bound_soft, (real)0);
      p.da[i * A + j] += p.scale * (real)2 * p.bound_coef * (hi + lo);
    }
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
  p.stat[i * 4 + 0] = surr + bl;
  p.stat[i * 4 + 1] = vl;
  p.stat[i * 4 + 2] = kl;
  p.stat[i * 4 + 3] = r;
}


// ---------------- 계층형: 가린 범주 분포 (ACSpec::n_slots > 0, 슬롯 머리 slot_head.cuh) ----------------
constexpr int MAXK = 24;   // 범주 칸 최대 수 (커널 안 고정 배열) — 계층형 슬롯 16, 그래프 이웃 24
// 로짓 l[K]·가림 m[K](1 = 있음) → 있는 칸의 log 확률 lp[K] (가린 칸 확률 0 — lp 는 0 으로 두고 쓰지 않음), 반환 = 있는 칸 수.
// log-sum-exp 는 있는 칸 최댓값을 빼서, log p = (l − max) − log Σ — mx + log Σ 를 먼저 더하면 |l| 이 크면(> ~1e7) log Σ (≤ log K) 가
// 반올림에 먹혀 log p = 0 · Σ p = V 가 된다 (explore_h2 반복 3500 뒤 엔트로피 0 으로 찍힌 까닭)
__device__ int cat_logp(const real* l, const real* m, int K, real* lp) {
  real mx = -1e30f;
  int V = 0;
  for (int k = 0; k < K; ++k)
    if (m[k] > 0.5f) { mx = fmax(mx, l[k]); ++V; }
  if (V == 0) {
    for (int k = 0; k < K; ++k) lp[k] = 0;
    return 0;
  }
  real s = 0;
  for (int k = 0; k < K; ++k)
    if (m[k] > 0.5f) s += exp(l[k] - mx);
  const real ls = log(s);
  for (int k = 0; k < K; ++k) lp[k] = m[k] > 0.5f ? (l[k] - mx) - ls : (real)0;
  return V;
}
// 가림 = 날 1D 관측(정규화 전)의 후보 칸 마지막(있음 0/1) 값: mask[i·K + k] = raw[i·ld + off + k·F + F − 1] > 0.5
__global__ void k_cat_mask(const real* raw, int ld, int off, int F, int n, int K, real* mask) {
  int t = blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= n * K) return;
  const int i = t / K, k = t % K;
  mask[t] = raw[(long)i * ld + off + k * F + F - 1] > 0.5f ? (real)1 : (real)0;
}
// 학습 샘플: u ~ U(0,1] (Philox — 연속 판 k_sample 과 같은 씨앗·부분열 규칙), 있는 칸 차례 누적 확률이 u 이상인 첫 칸. 있는 칸이 없으면 0 번·log π 0
__global__ void k_sample_cat(const real* logits, const real* mask, int n, int K, uint64_t seed, uint64_t call, int env0, int adim,
                             real* act_out, real* s_act, real* s_logp, real* s_logit, real* s_mask) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  curandStatePhilox4_32_10_t rs;
  curand_init(seed, (uint64_t)(env0 + i), call * 8ull, &rs);
  const real* l = logits + (long)i * K;
  const real* m = mask + (long)i * K;
  real lp[MAXK];
  const int V = cat_logp(l, m, K, lp);
  const real u = curand_uniform(&rs);
  int a = 0;
  real lpa = 0;
  if (V > 0) {
    real cum = 0;
    int last = -1;
    a = -1;
    for (int k = 0; k < K; ++k) {
      if (m[k] <= 0.5f) continue;
      last = k;
      cum += exp(lp[k]);
      if (a < 0 && cum >= u) a = k;
    }
    if (a < 0) a = last;   // 반올림으로 누적이 1 아래면 마지막 있는 칸
    lpa = lp[a];
  }
  for (int j = 0; j < adim; ++j) act_out[(long)i * adim + j] = j == 0 ? (real)a : (real)0;
  s_act[i] = (real)a;
  s_logp[i] = lpa;
  for (int k = 0; k < K; ++k) { s_logit[(long)i * K + k] = l[k]; s_mask[(long)i * K + k] = m[k]; }
}
// 평가: sample_call < 0 이면 있는 칸 중 로짓 최대(같으면 작은 번호), 아니면 학습과 다른 씨앗 계열로 샘플
__global__ void k_cat_act(const real* logits, const real* mask, int n, int K, uint64_t seed, int64_t call, int env0, int adim, real* act) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  const real* l = logits + (long)i * K;
  const real* m = mask + (long)i * K;
  int a = 0;
  if (call < 0) {
    real best = -1e30f;
    a = -1;
    for (int k = 0; k < K; ++k)
      if (m[k] > 0.5f && (a < 0 || l[k] > best)) { best = l[k]; a = k; }
    if (a < 0) a = 0;
  } else {
    curandStatePhilox4_32_10_t rs;
    curand_init(seed, (uint64_t)(env0 + i), (uint64_t)call * 8ull, &rs);
    real lp[MAXK];
    const int V = cat_logp(l, m, K, lp);
    const real u = curand_uniform(&rs);
    if (V > 0) {
      real cum = 0;
      int last = -1;
      a = -1;
      for (int k = 0; k < K; ++k) {
        if (m[k] <= 0.5f) continue;
        last = k;
        cum += exp(lp[k]);
        if (a < 0 && cum >= u) a = k;
      }
      if (a < 0) a = last;
    }
  }
  for (int j = 0; j < adim; ++j) act[(long)i * adim + j] = j == 0 ? (real)a : (real)0;
}
// 미니배치 특징 모으기: dst [n][w] = obs [idx[i]][off … off + w)
__global__ void k_gather_cols(const real* obs, int D, int off, int w, const int* idx, int n, real* dst) {
  long t = blockIdx.x * (long)blockDim.x + threadIdx.x;
  if (t >= (long)n * w) return;
  const int i = (int)(t / w), j = (int)(t % w);
  dst[t] = obs[(long)idx[i] * D + off + j];
}
// 범주 PPO 손실: 대리(자른 비) + 엔트로피 보너스(샘플마다 H, 계수 ent·scale) → 로짓 기울기 G[i·K + k], KL(옛 ‖ 새) = Σ p_옛 (log p_옛 − log p_새).
// 있는 칸이 없는 샘플(고를 것이 없음)은 log π = 0 · 비 1 · 기울기 0. 가치 손실은 k_loss 와 같은 식
struct CatLossArgs {
  const real *logits, *mask, *old_logits, *act, *old_logp, *adv, *ret, *old_v, *v;
  int n, K;
  real clip, value_coef, scale, ent, cm;
  bool clipped_value;
  real *G, *dv, *stat;
};
__global__ void k_loss_cat(CatLossArgs p) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= p.n) return;
  const int K = p.K;
  const real* l = p.logits + (long)i * K;
  const real* m = p.mask + (long)i * K;
  real lp[MAXK], lo[MAXK];
  const int V = cat_logp(l, m, K, lp);
  cat_logp(p.old_logits + (long)i * K, m, K, lo);
  real surr = 0, kl = 0, H = 0, r = 1, cm = 0;
  real* G = p.G + (long)i * K;
  for (int k = 0; k < K; ++k) G[k] = 0;
  if (V > 0) {
    const int a = (int)p.act[i];
    r = exp(lp[a] - p.old_logp[i]);
    const real Ad = p.adv[i];
    const real rc = fmin(fmax(r, 1.f - p.clip), 1.f + p.clip);
    const real s1 = -Ad * r, s2 = -Ad * rc;
    surr = fmax(s1, s2);
    const bool inside = r > 1.f - p.clip && r < 1.f + p.clip;
    const real g = ((inside || s1 > s2) ? -Ad * r : (real)0) * p.scale;   // d surr / d log π_a
    for (int k = 0; k < K; ++k)
      if (m[k] > 0.5f) {
        const real pk = exp(lp[k]);
        H -= pk * lp[k];
        kl += exp(lo[k]) * (lo[k] - lp[k]);
      }
    // 공통 성분(게이지) 벌점 cm·h(l̄), l̄ = 있는 칸 로짓 평균 (ppo/README "슬롯 로짓 공통 성분"): 범주 분포는 모든 로짓에 같은 값을 더해도 같아
    // 대리·엔트로피 기울기는 있는 칸 합이 늘 0 — 공통 성분엔 되돌리는 힘이 없어 explore_h2 가 반복 2889 에 발산했다.
    // h = Huber (|l̄| ≤ δ 이면 l̄², 넘으면 2δ|l̄| − δ²): 기울기 2·cm·clamp(l̄, ±δ)/V 는 있는 칸 모두 같아 정책(확률)은 그대로, 공통 성분만 0 으로.
    // 상한 δ 는 이미 공통 성분이 큰 체크포인트(explore_h2 2850, |l̄| 1.4e5)를 이어 받을 때 이 항이 기울기를 독차지하지 않게
    constexpr real kCmDelta = 10.f;
    real lbar = 0;
    for (int k = 0; k < K; ++k)
      if (m[k] > 0.5f) lbar += l[k];
    lbar /= (real)V;
    const real al = fabs(lbar);
    cm = al <= kCmDelta ? lbar * lbar : 2.f * kCmDelta * al - kCmDelta * kCmDelta;
    const real gcm = p.cm * p.scale * 2.f * fmin(fmax(lbar, -kCmDelta), kCmDelta) / (real)V;
    for (int k = 0; k < K; ++k) {
      if (m[k] <= 0.5f) continue;
      const real pk = exp(lp[k]);
      // d(−ent·H)/dl_k = ent · p_k (log p_k + H)
      G[k] = g * ((k == a ? (real)1 : (real)0) - pk) + p.ent * p.scale * pk * (lp[k] + H) + gcm;
    }
  }
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
  p.stat[i * 5 + 0] = surr;
  p.stat[i * 5 + 1] = vl;
  p.stat[i * 5 + 2] = kl;
  p.stat[i * 5 + 3] = H;    // 범주 판: 넷째 칸 = 샘플 엔트로피 (연속 판은 비 — 안 씀)
  p.stat[i * 5 + 4] = cm;   // 다섯째 = h(l̄) (공통 성분 벌점, 계수 곱하기 전)
}
// 롤아웃 로짓 진단 (샘플마다, 있는 칸만): out[i·3] = |l̄| (공통 성분), [+1] = 있는 칸 로짓 표준편차 (정책 날카로움), [+2] = max |l|
__global__ void k_cat_logit_stats(const real* logits, const real* mask, int n, int K, real* out) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  const real* l = logits + (long)i * K;
  const real* m = mask + (long)i * K;
  double s = 0, mx = 0;
  int V = 0;
  for (int k = 0; k < K; ++k)
    if (m[k] > 0.5f) { s += l[k]; mx = fmax(mx, fabs((double)l[k])); ++V; }
  double mean = V ? s / V : 0, var = 0;
  for (int k = 0; k < K; ++k)
    if (m[k] > 0.5f) var += ((double)l[k] - mean) * ((double)l[k] - mean);
  out[(long)i * 3 + 0] = (real)fabs(mean);
  out[(long)i * 3 + 1] = V ? (real)sqrt(var / V) : (real)0;
  out[(long)i * 3 + 2] = (real)mx;
}

// ---------------- 그래프 어텐션 정책 (ACSpec::graph) ----------------
// 가림: 현재 노드의 k 번 이웃 행이 있으면(간선 있고 이웃 유효) 1
__global__ void k_nb_mask(const int* nbrow, long n, real* mask) {
  long t = blockIdx.x * (long)blockDim.x + threadIdx.x;
  if (t < n) mask[t] = nbrow[t] >= 0 ? (real)1 : (real)0;
}
// 가치 머리 입력 [E | Hc | 정규화 참값]: dst [n][2d + p]
__global__ void k_vcat(const real* E, const real* Hc, const real* priv, long ldp, int n, int d, int p, real* dst) {
  long t = blockIdx.x * (long)blockDim.x + threadIdx.x;
  const int C = 2 * d + p;
  if (t >= (long)n * C) return;
  const int i = (int)(t / C), j = (int)(t % C);
  dst[t] = j < d ? E[(long)i * d + j] : j < 2 * d ? Hc[(long)i * d + j - d] : priv[(long)i * ldp + j - 2 * d];
}
// 롤아웃 1D 행 [전역 원값 g | 참값 p (정규화 전 — 뒤에 제자리 정규화)]: 없는 쪽(nullptr)은 0
__global__ void k_gobs(const real* glob, int g, const real* priv, int p, int n, real* dst) {
  long t = blockIdx.x * (long)blockDim.x + threadIdx.x;
  const int D = g + p;
  if (t >= (long)n * D) return;
  const int i = (int)(t / D), j = (int)(t % D);
  dst[t] = j < g ? (glob ? glob[(long)i * g + j] : (real)0) : (priv ? priv[(long)i * p + j - g] : (real)0);
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
  // 그래프: 한 번에 넣는 양은 유효 노드 합(ACSpec::graph_rows)이 정한다 — 샘플 수 상한은 그 안에서 넉넉히 (잠정 1024)
  if (spec.graph && cfg.chunk <= 0) cfg_.chunk = std::min(cfg_.chunk, 1024);
  if (spec.n_cont > MAXC) throw std::runtime_error("n_cont > MAXC");
  ac_ = new ActorCritic(spec, seed, cfg_.chunk);
  CKB(cublasSetMathMode(ac_->cublas, cfg.tf32 ? CUBLAS_TF32_TENSOR_OP_MATH : CUBLAS_DEFAULT_MATH));
  bool bf = cfg.cnn_bf16;
  if (const char* e = std::getenv("XNET_CNN_BF16")) bf = std::atoi(e) != 0;  // TODO(제거): 정밀도 A/B 실측용
  ac_->enc.cnn_bf16 = bf;
  ac_->enc_c.cnn_bf16 = bf;
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
  disc_ = spec.n_slots > 0;
  AO_ = spec.actor.out;
  ADIM_ = mdp::act::DIM;
  if (disc_) {
    K_ = spec.n_slots;
    SOFF_ = spec.slot_off;
    SF_ = spec.slot_feat;
    A_ = 1;   // 저장하는 행동 = 슬롯 번호 하나
    if (K_ > MAXK) throw std::runtime_error("n_slots > MAXK");
  }
  graph_ = spec.graph;
  if (graph_) {   // 행동 = 이웃 번호 하나 (가린 범주 분포). 공통 성분 벌점은 안 씀 — 포인터 로짓은 10·tanh 로 유계라 떠내려갈 수 없다
    disc_ = true;
    K_ = kGNb;
    A_ = 1;
    AO_ = SOFF_ = SF_ = 0;
    cfg_.slot_cm_coef = 0;
    GN_ = spec.gspec.gw * spec.gspec.gh;
    GF_ = spec.gspec.feat;
    GG_ = spec.graph_glob;
    GP_ = spec.norm_dim;
    D_ = GG_ + GP_;
    IMG_ = NF_ = 0;
  }
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
  s_logit_ = dmalloc<real>((long)S_ * (disc_ ? K_ : 1), b);
  if (disc_) s_mask_ = dmalloc<real>((long)S_ * K_, b);
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
  c_aout_ = dmalloc<real>((long)C * AO_, b);
  c_v_ = dmalloc<real>(C, b);
  c_da_ = dmalloc<real>((long)C * AO_, b);
  if (disc_) {
    c_logits_ = dmalloc<real>((long)C * K_, b);
    c_G_ = dmalloc<real>((long)C * K_, b);
    c_feat_ = dmalloc<real>((long)C * K_ * SF_, b);
    c_maskt_ = dmalloc<real>((long)C * K_, b);
  }
  c_dv_ = dmalloc<real>(C, b);
  c_dls_ = dmalloc<real>((long)C * NC_, b);
  c_stat_ = dmalloc<real>((long)C * 5, b);   // 미니배치 샘플 통계 (연속 판 4 칸, 범주 판 5 칸)
  if (disc_) s_lstat_ = dmalloc<real>((long)S_ * 3, b);
  c_red_ = dmalloc<real>(16, b);
  c_x_ = dmalloc<real>((long)C * (NF_ + D_), b);
  // bf16 MLP(mlp 2)는 첫 층이 cuBLAS bf16 이라 입력 bf16 사본 c_xh_ 를 둔다 (1D 열은 사본에만, F 열은 둘 다 — fp32 는 넷째 층 ELU' 용)
  if (sizeof(real) == 4 && cfg_.mlp == 2 && !graph_) c_xh_ = dmalloc<uint16_t>((long)C * (NF_ + D_), b);
  if (graph_) {
    const int d = spec.gspec.d;
    c_xc_ = dmalloc<real>((long)C * (2 * d + GP_), b);   // 가치 머리 입력 [E | Hc | 참값]
    s_gfeat_ = dmalloc<uint8_t>((long)S_ * GN_ * GF_, b);
    s_gedge_ = dmalloc<uint32_t>((long)S_ * GN_, b);
    s_gcur_ = dmalloc<int>(S_, b);
    d_cnt_ = dmalloc<int>(N_, b);
    h_cnt_.assign(S_, 0);
  } else {
    ac_->enc.set_x0(c_x_, NF_ + D_, c_xh_);  // 인코더가 특징을 MLP 입력 [F | 1D] 의 앞 열에 바로 쓴다 (모으는 복사 없음)
  }
  if (!graph_ && ac_->split) {   // critic 인코더는 자기 입력 버퍼 [F_c | 1D] 에
    c_xc_ = dmalloc<real>((long)C * (NF_ + D_), b);
    if (c_xh_) c_xch_ = dmalloc<uint16_t>((long)C * (NF_ + D_), b);
    ac_->enc_c.set_x0(c_xc_, NF_ + D_, c_xch_);
  }
  c_gat_ = dmalloc<real>((long)C * (A_ + 5 + 2 * std::max(NC_, K_)), b);
  d_perm_ = dmalloc<int>(S_, b);
}

PPO::~PPO() {
  for (void* p : std::initializer_list<void*>{m_, v_, s_img_, s_obs_, s_act_, s_logp_, s_val_, s_mu_, s_logit_, s_sd_,
                                              s_rew_, s_done_, s_ret_, s_adv_, raw_, last_obs_, last_val_, c_img_,
                                              c_aout_, c_v_, c_da_, c_dv_, c_dls_, c_stat_, c_red_, c_gat_, d_perm_, c_x_, c_xh_, c_xc_, c_xch_,
                                              s_mask_, c_logits_, c_G_, c_feat_, c_maskt_, s_lstat_, s_gfeat_, s_gedge_, s_gcur_, d_cnt_, c_gred_})
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

void PPO::encode_c(int n, const uint8_t* img, const int* idx, const real* x1d) {
  auto st = ac_->stream2;
  ac_->enc_c.forward(ac_->cublas2, st, n, img, nullptr, 0, nullptr, idx);
  if (x1d)
    k_put_1d<<<nblk((long)n * D_), 256, 0, st>>>(x1d, D_, D_, n, c_xc_ + NF_, c_xch_ ? (__nv_bfloat16*)c_xch_ + NF_ : nullptr, NF_ + D_);
}

void PPO::concat_norm(const real* obs1d, const real* priv, int n, bool upd, real* out) {
  auto st = ac_->stream;
  int da = ac_->spec.actor.in1d, db = D_ - da;
  k_concat<<<nblk((long)n * D_), 256, 0, st>>>(obs1d, da, priv, db, n, raw_);
  if (upd) ac_->norm.update(raw_, n, D_, st);
  ac_->norm.apply(raw_, n, D_, out, D_, st);
}

void PPO::act(const uint8_t* img, const real* obs1d, const real* priv, real* actions) {
  if (graph_) throw std::runtime_error("PPO::act: 그래프 구조는 GraphObs 판을 쓴다");
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
    ac_->fork();   // 나눔: critic 인코더·MLP 는 stream2 에서 actor 쪽과 겹쳐 돈다 (버퍼·기울기 칸이 안 겹침)
    if (ac_->split) encode_c(n, im, nullptr, ob);
    encode(n, im, nullptr, ob);
    if (!ac_->split) ac_->fork();
    ac_->critic.forward(ac_->cublas2, ac_->stream2, n, nullptr, cx(), NF_ + D_, s_val_ + k0 + c0, nullptr, cxh());
    ac_->actor.forward(ac_->cublas, st, n, nullptr, c_x_, NF_ + D_, c_aout_, nullptr, c_xh_);
    ac_->join();
    long k = k0 + c0;
    if (disc_) {   // 슬롯 머리 → 가림(날 1D 관측의 후보 있음 칸) → 범주 샘플
      ac_->head.forward(st, n, c_aout_, AO_, ob, D_, SOFF_, ac_->d_params, c_logits_);
      k_cat_mask<<<nblk((long)n * K_), 256, 0, st>>>(obs1d + (long)c0 * ac_->spec.actor.in1d, ac_->spec.actor.in1d, SOFF_, SF_, n, K_, c_maskt_);
      k_sample_cat<<<nblk(n, 128), 128, 0, st>>>(c_logits_, c_maskt_, n, K_, seed_, act_calls_, c0, ADIM_, actions + (long)c0 * ADIM_,
                                                  s_act_ + k, s_logp_ + k, s_logit_ + k * K_, s_mask_ + k * K_);
      continue;
    }
    k_sample<<<nblk(n, 128), 128, 0, st>>>(c_aout_, ac_->log_std(), n, NC_, seed_, act_calls_, c0, actions + (long)c0 * A_,
                                            s_act_ + k * A_, s_logp_ + k, s_mu_ + k * NC_, s_logit_ + k,
                                            s_sd_ + k * NC_);
  }
  ++act_calls_;
}

void PPO::last_act_info(int n, real* logit, real* value) const {
  if (n > N_) throw std::runtime_error("last_act_info: n > num_envs");
  const long k0 = (long)step_ * N_;
  if (disc_) std::fill(logit, logit + n, (real)0);   // 범주 판: 선언 로짓 없음
  else CK(cudaMemcpyAsync(logit, s_logit_ + k0, n * sizeof(real), cudaMemcpyDeviceToHost, ac_->stream));
  CK(cudaMemcpyAsync(value, s_val_ + k0, n * sizeof(real), cudaMemcpyDeviceToHost, ac_->stream));
  CK(cudaStreamSynchronize(ac_->stream));
}

void PPO::store(const real* reward, const int* done) {
  long k0 = (long)step_ * N_;
  k_store<<<nblk(N_, 128), 128, 0, ac_->stream>>>(reward, done, s_val_ + k0, cfg_.gamma, N_, s_rew_ + k0,
                                                  s_done_ + k0);
  ++step_;
}

void PPO::act_inference(const uint8_t* img, const real* obs1d, real* actions, int n_all, int64_t sample_call) {
  if (n_all < 0) n_all = N_;
  if (n_all > N_) throw std::runtime_error("act_inference: n > num_envs");
  if (graph_) throw std::runtime_error("PPO::act_inference: 그래프 구조는 GraphObs 판을 쓴다");
  auto st = ac_->stream;
  int da = ac_->spec.actor.in1d;
  // 정책 1D 만 정규화 (같은 통계의 앞 da 차원)
  ac_->norm.apply(obs1d, n_all, da, raw_, D_, st, da);
  for (int c0 = 0; c0 < n_all; c0 += cfg_.chunk) {
    int n = std::min(cfg_.chunk, n_all - c0);
    encode(n, img + (long)c0 * IMG_, nullptr, raw_ + (long)c0 * D_);
    ac_->actor.forward(ac_->cublas, st, n, nullptr, c_x_, NF_ + D_, c_aout_, nullptr, c_xh_);
    if (disc_) {
      ac_->head.forward(st, n, c_aout_, AO_, raw_ + (long)c0 * D_, D_, SOFF_, ac_->d_params, c_logits_);
      k_cat_mask<<<nblk((long)n * K_), 256, 0, st>>>(obs1d + (long)c0 * da, da, SOFF_, SF_, n, K_, c_maskt_);
      k_cat_act<<<nblk(n, 128), 128, 0, st>>>(c_logits_, c_maskt_, n, K_, seed_ ^ 0x9E3779B97F4A7C15ull, sample_call, c0, ADIM_,
                                               actions + (long)c0 * ADIM_);
      continue;
    }
    if (sample_call >= 0)
      k_sample_act<<<nblk(n, 128), 128, 0, st>>>(c_aout_, ac_->log_std(), n, NC_, seed_ ^ 0x9E3779B97F4A7C15ull,
                                                  (uint64_t)sample_call, c0, actions + (long)c0 * A_);
    else
      k_mean_act<<<nblk(n, 128), 128, 0, st>>>(c_aout_, n, NC_, actions + (long)c0 * A_);
  }
}

void PPO::compute_returns(const uint8_t* img, const real* obs1d, const real* priv) {
  if (graph_) throw std::runtime_error("PPO::compute_returns: 그래프 구조는 GraphObs 판을 쓴다");
  if (step_ != T_) throw std::runtime_error("PPO::compute_returns: 롤아웃이 덜 참");
  auto st = ac_->stream;
  concat_norm(obs1d, priv, N_, false, last_obs_);
  for (int c0 = 0; c0 < N_; c0 += cfg_.chunk) {
    int n = std::min(cfg_.chunk, N_ - c0);
    if (ac_->split) {   // critic 인코더만 (stream2 → stream 에 합침)
      ac_->fork();
      encode_c(n, img + (long)c0 * IMG_, nullptr, last_obs_ + (long)c0 * D_);
      ac_->critic.forward(ac_->cublas2, ac_->stream2, n, nullptr, c_xc_, NF_ + D_, last_val_ + c0, nullptr, c_xch_);
      ac_->join();
    } else {
      encode(n, img + (long)c0 * IMG_, nullptr, last_obs_ + (long)c0 * D_);
      ac_->critic.forward(ac_->cublas, st, n, nullptr, c_x_, NF_ + D_, last_val_ + c0, nullptr, c_xh_);
    }
  }
  gae();
}

void PPO::gae() {
  auto st = ac_->stream;
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
double PPO::minibatch(const int* d_idx, int n_all, int mb, bool backward, double stats[5]) {
  if (graph_) return minibatch_graph(d_idx, n_all, mb, backward, stats);
  auto st = ac_->stream;
  auto& ac = *ac_;
  std::fill(stats, stats + 5, 0.0);
  const int NS = disc_ ? 5 : 4;   // 샘플 통계 칸 수 (범주 판은 공통 성분 벌점 h(l̄) 한 칸 더)
  std::vector<real> hs(5);
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
    // 범주 판은 μ 자리에 가림 (K), 로짓 자리에 옛 로짓 (K), σ 자리 0 칸
    GatherSet gs = disc_ ? GatherSet{{s_act_, s_logp_, s_mask_, s_logit_, s_sd_, s_adv_, s_ret_, s_val_}, {}, {A_, 1, K_, K_, 0, 1, 1, 1}, 0}
                         : GatherSet{{s_act_, s_logp_, s_mu_, s_logit_, s_sd_, s_adv_, s_ret_, s_val_}, {}, {A_, 1, NC_, 1, NC_, 1, 1, 1}, 0};
    real* off[8];
    real* cur = c_gat_;
    for (int g = 0; g < 8; ++g) {
      off[g] = gs.dst[g] = cur;
      cur += (long)n * gs.w[g];
      gs.W += gs.w[g];
    }
    k_gather_mb<<<nblk((long)n * (D_ + gs.W)), 256, 0, st>>>(s_obs_, D_, ix, n, c_x_ + NF_, NF_ + D_,
                                                                   c_xh_ ? (__nv_bfloat16*)c_xh_ + NF_ : nullptr, gs);
    if (ac.split) {   // critic 입력 버퍼에도 같은 1D 열 (모을 롤아웃 값은 없이)
      GatherSet g0{{}, {}, {}, 0};
      k_gather_mb<<<nblk((long)n * D_), 256, 0, st>>>(s_obs_, D_, ix, n, c_xc_ + NF_, NF_ + D_, c_xch_ ? (__nv_bfloat16*)c_xch_ + NF_ : nullptr, g0);
    }
    ac.fork();
    if (ac.split) encode_c(n, im, imx, nullptr);
    encode(n, im, imx, nullptr);
    if (!ac.split) ac.fork();
    ac.critic.forward(ac.cublas2, ac.stream2, n, nullptr, cx(), NF_ + D_, c_v_, nullptr, cxh());
    ac.actor.forward(ac.cublas, st, n, nullptr, c_x_, NF_ + D_, c_aout_, nullptr, c_xh_);
    ac.join();
    if (disc_) {   // 슬롯 머리 (특징 = 정규화 1D 의 후보 칸) → 범주 손실
      k_gather_cols<<<nblk((long)n * K_ * SF_), 256, 0, st>>>(s_obs_, D_, SOFF_, K_ * SF_, ix, n, c_feat_);
      ac.head.forward(st, n, c_aout_, AO_, c_feat_, K_ * SF_, 0, ac.d_params, c_logits_);
      CatLossArgs q{c_logits_, off[2], off[3], off[0], off[1], off[5], off[6], off[7], c_v_, n, K_, cfg_.clip, cfg_.value_coef, 1.f / mb,
                    cfg_.entropy_coef, cfg_.slot_cm_coef, cfg_.clipped_value, c_G_, c_dv_, c_stat_};
      k_loss_cat<<<nblk(n, 128), 128, 0, st>>>(q);
    } else {
      LossArgs p{c_aout_, c_v_, ac.log_std(), off[0], off[1], off[2], off[3], off[4], off[5], off[6], off[7], n, NC_,
                 cfg_.clip, cfg_.value_coef, 1.f / mb, cfg_.clipped_value, cfg_.bound_coef, cfg_.bound_soft,
                 c_da_, c_dv_, c_dls_, c_stat_};
      k_loss<<<nblk(n, 128), 128, 0, st>>>(p);
    }
    if (backward) {
      ac.fork();
      ac.critic.backward(ac.cublas2, ac.stream2, n, nullptr, c_dv_);
      if (ac.split) ac.enc_c.backward_feat(ac.cublas2, ac.stream2, n, im, imx, ac.critic.dx0(), ac.critic.feat(), nullptr, 0);   // critic 손실만
      if (disc_) ac.head.backward(ac.cublas, st, n, c_aout_, AO_, ac.d_params, ac.d_grads, c_G_, c_da_, AO_);   // 머리 → 임베딩 기울기 c_da_
      ac.actor.backward(ac.cublas, st, n, nullptr, c_da_);
      if (ac.split) ac.enc.backward_feat(ac.cublas, st, n, im, imx, ac.actor.dx0(), ac.actor.feat(), nullptr, 0);   // actor 손실만
      ac.join();
      // 공유 인코더: 두 MLP 의 특징 기울기(actor + critic, 고정 순서)를 합성곱 역방향으로 (rsl_rl·공유 인코더 PPO 와 같은 합 손실).
      // 더하기는 넷째 층 역방향이 읽으면서 한다 (모으는 버퍼 없음)
      if (!ac.split) ac.enc.backward_feat(ac.cublas, st, n, im, imx, ac.actor.dx0(), ac.actor.feat(), ac.critic.dx0(), ac.critic.feat());
      if (NC_ > 0) col_sum(c_dls_, n, NC_, NC_, ac.log_std_grad(), true, st);
    }
    col_sum(c_stat_, n, NS, NS, c_red_, false, st);
    CK(cudaMemcpyAsync(hs.data(), c_red_, NS * sizeof(real), cudaMemcpyDeviceToHost, st));
    CK(cudaStreamSynchronize(st));
    for (int k = 0; k < NS; ++k) stats[k] += hs[k];
  }
  // 엔트로피(가우시안만, 상태 무관): −c·Σ(½+½ln2π+logσ) → d/dlogσ = −c
  std::vector<real> ls(NC_);
  CK(cudaMemcpy(ls.data(), ac.log_std(), NC_ * sizeof(real), cudaMemcpyDeviceToHost));
  double ent = 0;
  for (real l : ls) ent += 0.5 + 0.91893853320467274 + l;
  if (disc_) {   // 범주 판: stats[3] = 샘플 엔트로피 합 (k_loss_cat), 기울기는 로짓으로 이미 들어감
    return stats[0] / mb + cfg_.value_coef * stats[1] / mb - cfg_.entropy_coef * stats[3] / mb + cfg_.slot_cm_coef * stats[4] / mb;
  }
  if (backward) k_add_const<<<1, 32, 0, st>>>(ac.log_std_grad(), NC_, -cfg_.entropy_coef * ((real)n_all / mb));
  stats[3] = ent * n_all;  // 엔트로피 합 자리로 바꿔 쓴다 (ratio 는 안 씀)
  return stats[0] / mb + cfg_.value_coef * stats[1] / mb - cfg_.entropy_coef * ent * n_all / mb;
}

double PPO::loss_and_grad(const int* d_idx, int n, bool backward) {
  ac_->params_changed();  // 검사 도구가 d_params 를 직접 바꾼다
  if (backward) ac_->zero_grad();
  double st[5];
  return minibatch(d_idx, n, n, backward, st);
}

UpdateStats PPO::update(const uint8_t* img, const real* obs1d, const real* priv) {
  compute_returns(img, obs1d, priv);
  return finish_update();
}

UpdateStats PPO::update(const GraphObs& o) {
  compute_returns(o);
  return finish_update();
}

UpdateStats PPO::finish_update() {
  auto st = ac_->stream;
  UpdateStats out;
  {
    double sv = sum_all(s_val_, S_, st), sr = sum_all(s_ret_, S_, st);
    out.mean_value = (real)(sv / S_);
    out.mean_return = (real)(sr / S_);
  }
  if (disc_) {   // 롤아웃 슬롯 로짓 진단 (학습엔 안 씀) — ppo/README "슬롯 로짓 공통 성분"
    k_cat_logit_stats<<<nblk(S_, 128), 128, 0, st>>>(s_logit_, s_mask_, S_, K_, s_lstat_);
    col_sum(s_lstat_, S_, 3, 3, c_red_ + 8, false, st);
    real h3[3];
    CK(cudaMemcpyAsync(h3, c_red_ + 8, sizeof(h3), cudaMemcpyDeviceToHost, st));
    CK(cudaStreamSynchronize(st));
    out.logit_cm = h3[0] / S_;
    out.logit_spread = h3[1] / S_;
    out.logit_absmax = h3[2] / S_;
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
      double s[5];
      minibatch(d_perm_ + (long)b * mb, mb, mb, true, s);
      double kl = s[2] / mb;
      if (ep == 0 && b == 0) out.kl0 = (real)kl;   // 아직 걸음 전: 롤아웃과 같은 변수라 0 이어야 (아니면 수치 문제 — 학습률 탓 아님)
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

// ---------------- 그래프 어텐션 정책 ----------------
std::vector<int> PPO::graph_segments(const int* cnt, int n_all) const {
  const long cap = ac_->gbody.cap_rows();
  std::vector<int> seg{0};
  long rows = 0;
  int n = 0;
  for (int i = 0; i < n_all; ++i) {
    if (cnt[i] < 1) throw std::runtime_error("PPO(그래프): 현재 노드가 무효이거나 유효 노드가 없는 관측 (샘플 " + std::to_string(i) + ")");
    if (n == cfg_.chunk || rows + cnt[i] > cap) {
      seg.push_back(i);
      rows = 0;
      n = 0;
    }
    rows += cnt[i];
    ++n;
  }
  seg.push_back(n_all);
  return seg;
}

void PPO::graph_forward(int n, const GraphIn& in, const int* cnt, const real* glob, const real* privn, long ld, bool policy, real* v_out) {
  auto& ac = *ac_;
  auto st = ac.stream;
  const int d = ac.spec.gspec.d;
  const bool share = ac.spec.graph_share;
  if (policy || (v_out && share)) ac.gbody.forward(ac.cublas, st, n, in, cnt);
  if (policy) {
    ac.ptr.forward(ac.cublas, st, n, ac.gbody.E(), ac.gbody.Hc(), glob, ld, ac.gbody.Hn(), ac.d_params, c_logits_);
    k_nb_mask<<<nblk((long)n * K_), 256, 0, st>>>(ac.gbody.nbrow(), (long)n * K_, c_maskt_);
  }
  if (v_out) {
    if (!share) ac.gbody_c.forward(ac.cublas, st, n, in, cnt);
    GraphBody& vb = ac.vbody();
    k_vcat<<<nblk((long)n * (2 * d + GP_)), 256, 0, st>>>(vb.E(), vb.Hc(), privn, ld, n, d, GP_, c_xc_);
    ac.critic.forward(ac.cublas, st, n, nullptr, c_xc_, 2 * d + GP_, v_out);
  }
}

void PPO::act(const GraphObs& o, real* actions) {
  if (!graph_) throw std::runtime_error("PPO::act(GraphObs): 그래프 구조가 아님");
  if (step_ >= T_) throw std::runtime_error("PPO::act: 롤아웃이 가득 참 — update 먼저");
  auto st = ac_->stream;
  const long k0 = (long)step_ * N_;
  uint8_t* gf = s_gfeat_ + k0 * GN_ * GF_;
  uint32_t* ge = s_gedge_ + k0 * GN_;
  int* gc = s_gcur_ + k0;
  CK(cudaMemcpyAsync(gf, o.feat, (size_t)N_ * GN_ * GF_, cudaMemcpyDeviceToDevice, st));
  CK(cudaMemcpyAsync(ge, o.edge, (size_t)N_ * GN_ * sizeof(uint32_t), cudaMemcpyDeviceToDevice, st));
  CK(cudaMemcpyAsync(gc, o.cur, (size_t)N_ * sizeof(int), cudaMemcpyDeviceToDevice, st));
  // 롤아웃 1D 행 = [전역 원값 | 정규화 참값] (정규화 통계는 참값만)
  real* obs = s_obs_ + k0 * D_;
  k_gobs<<<nblk((long)N_ * D_), 256, 0, st>>>(GG_ ? o.glob : nullptr, GG_, o.priv, GP_, N_, obs);
  if (o.priv) ac_->norm.update(obs + GG_, N_, D_, st);
  ac_->norm.apply(obs + GG_, N_, D_, obs + GG_, D_, st);
  int* cnt = h_cnt_.data() + k0;
  graph_count(ac_->spec.gspec, st, N_, GraphIn{gf, ge, gc, nullptr}, d_cnt_, cnt);
  const std::vector<int> seg = graph_segments(cnt, N_);
  for (size_t s = 0; s + 1 < seg.size(); ++s) {
    const int c0 = seg[s], n = seg[s + 1] - c0;
    const long k = k0 + c0;
    const real* ob = obs + (long)c0 * D_;
    graph_forward(n, GraphIn{gf + (long)c0 * GN_ * GF_, ge + (long)c0 * GN_, gc + c0, nullptr}, cnt + c0, ob, ob + GG_, D_, true, s_val_ + k);
    k_sample_cat<<<nblk(n, 128), 128, 0, st>>>(c_logits_, c_maskt_, n, K_, seed_, act_calls_, c0, ADIM_, actions + (long)c0 * ADIM_, s_act_ + k,
                                                s_logp_ + k, s_logit_ + k * K_, s_mask_ + k * K_);
  }
  ++act_calls_;
}

void PPO::act_inference(const GraphObs& o, real* actions, int n_all, int64_t sample_call) {
  if (!graph_) throw std::runtime_error("PPO::act_inference(GraphObs): 그래프 구조가 아님");
  if (n_all < 0) n_all = N_;
  if (n_all > N_) throw std::runtime_error("act_inference: n > num_envs");
  auto st = ac_->stream;
  std::vector<int> cnt(n_all);
  graph_count(ac_->spec.gspec, st, n_all, GraphIn{o.feat, o.edge, o.cur, nullptr}, d_cnt_, cnt.data());
  const std::vector<int> seg = graph_segments(cnt.data(), n_all);
  for (size_t s = 0; s + 1 < seg.size(); ++s) {
    const int c0 = seg[s], n = seg[s + 1] - c0;
    graph_forward(n, GraphIn{o.feat + (long)c0 * GN_ * GF_, o.edge + (long)c0 * GN_, o.cur + c0, nullptr}, cnt.data() + c0,
                  GG_ ? o.glob + (long)c0 * GG_ : nullptr, nullptr, GG_, true, nullptr);
    k_cat_act<<<nblk(n, 128), 128, 0, st>>>(c_logits_, c_maskt_, n, K_, seed_ ^ 0x9E3779B97F4A7C15ull, sample_call, c0, ADIM_,
                                             actions + (long)c0 * ADIM_);
  }
}

void PPO::compute_returns(const GraphObs& o) {
  if (!graph_) throw std::runtime_error("PPO::compute_returns(GraphObs): 그래프 구조가 아님");
  if (step_ != T_) throw std::runtime_error("PPO::compute_returns: 롤아웃이 덜 참");
  auto st = ac_->stream;
  k_gobs<<<nblk((long)N_ * D_), 256, 0, st>>>(nullptr, GG_, o.priv, GP_, N_, last_obs_);
  ac_->norm.apply(last_obs_ + GG_, N_, D_, last_obs_ + GG_, D_, st);
  std::vector<int> cnt(N_);
  graph_count(ac_->spec.gspec, st, N_, GraphIn{o.feat, o.edge, o.cur, nullptr}, d_cnt_, cnt.data());
  const std::vector<int> seg = graph_segments(cnt.data(), N_);
  for (size_t s = 0; s + 1 < seg.size(); ++s) {
    const int c0 = seg[s], n = seg[s + 1] - c0;
    graph_forward(n, GraphIn{o.feat + (long)c0 * GN_ * GF_, o.edge + (long)c0 * GN_, o.cur + c0, nullptr}, cnt.data() + c0, nullptr,
                  last_obs_ + (long)c0 * D_ + GG_, D_, false, last_val_ + c0);
  }
  gae();
}

// 미니배치 하나 (그래프 판): 묶음마다 롤아웃에서 색인으로 바로 읽어 정방향 → 범주 손실 → 역방향. 한 스트림에서 차례로
double PPO::minibatch_graph(const int* d_idx, int n_all, int mb, bool backward, double stats[5]) {
  auto& ac = *ac_;
  auto st = ac.stream;
  const int d = ac.spec.gspec.d;
  std::fill(stats, stats + 5, 0.0);
  std::vector<int> hi(n_all), cnt(n_all);
  CK(cudaMemcpyAsync(hi.data(), d_idx, n_all * sizeof(int), cudaMemcpyDeviceToHost, st));
  CK(cudaStreamSynchronize(st));
  for (int i = 0; i < n_all; ++i) cnt[i] = h_cnt_[hi[i]];
  const std::vector<int> seg = graph_segments(cnt.data(), n_all);
  const size_t n_seg = seg.size() - 1;
  if ((long)n_seg * 5 > gred_cap_) {   // 구간 통계 자리 — 모자랄 때만 늘린다 (구간 수는 노드 수에 따라 달라짐)
    cudaFree(c_gred_);
    gred_cap_ = (long)n_seg * 5 * 2;
    CK(cudaMalloc(&c_gred_, gred_cap_ * sizeof(real)));
  }
  std::vector<real> hs(n_seg * 5);
  for (size_t s = 0; s + 1 < seg.size(); ++s) {
    const int c0 = seg[s], n = seg[s + 1] - c0;
    const int* ix = d_idx + c0;
    // 1D 행 [전역 | 참값] 과 롤아웃 값(행동·옛 log π·가림·옛 로짓·이점·수익·옛 가치)을 커널 하나로 모은다
    GatherSet gs{{s_act_, s_logp_, s_mask_, s_logit_, s_sd_, s_adv_, s_ret_, s_val_}, {}, {A_, 1, K_, K_, 0, 1, 1, 1}, 0};
    real* off[8];
    real* cur = c_gat_;
    for (int g = 0; g < 8; ++g) {
      off[g] = gs.dst[g] = cur;
      cur += (long)n * gs.w[g];
      gs.W += gs.w[g];
    }
    k_gather_mb<<<nblk((long)n * (D_ + gs.W)), 256, 0, st>>>(s_obs_, D_, ix, n, c_x_, D_, nullptr, gs);
    graph_forward(n, GraphIn{s_gfeat_, s_gedge_, s_gcur_, ix}, cnt.data() + c0, c_x_, c_x_ + GG_, D_, true, c_v_);
    CatLossArgs q{c_logits_, off[2], off[3], off[0], off[1], off[5], off[6], off[7], c_v_, n, K_, cfg_.clip, cfg_.value_coef, (real)1 / mb,
                  cfg_.entropy_coef, (real)0, cfg_.clipped_value, c_G_, c_dv_, c_stat_};
    k_loss_cat<<<nblk(n, 128), 128, 0, st>>>(q);
    if (backward) {
      ac.critic.backward(ac.cublas, st, n, nullptr, c_dv_);
      ac.ptr.backward(ac.cublas, st, n, ac.gbody.Hn(), ac.d_params, ac.d_grads, c_G_);
      const real* dv = ac.critic.dx0();   // [n][feat]: 앞 d = dE, 다음 d = dHc
      const long ldv = ac.critic.feat();
      if (ac.spec.graph_share) {   // 몸통 기울기 = 정책 + 가치 (고정 순서)
        ac.gbody.backward(ac.cublas, st, n, ac.ptr.d_E(), ac.ptr.d_Hc(), ac.ptr.ld(), ac.ptr.d_Hn(), dv, dv + d, ldv);
      } else {
        ac.gbody.backward(ac.cublas, st, n, ac.ptr.d_E(), ac.ptr.d_Hc(), ac.ptr.ld(), ac.ptr.d_Hn());
        ac.gbody_c.backward(ac.cublas, st, n, dv, dv + d, ldv, nullptr);
      }
    }
    col_sum(c_stat_, n, 5, 5, c_gred_ + 5 * s, false, st);
  }
  // 구간 통계는 장치에 모아 두었다가 미니배치 끝에 한 번만 내린다 (구간마다 복사·동기 하던 것 — 값·더하는 순서는 그대로)
  CK(cudaMemcpyAsync(hs.data(), c_gred_, hs.size() * sizeof(real), cudaMemcpyDeviceToHost, st));
  CK(cudaStreamSynchronize(st));
  for (size_t s = 0; s < n_seg; ++s)
    for (int k = 0; k < 5; ++k) stats[k] += hs[s * 5 + k];
  return stats[0] / mb + cfg_.value_coef * stats[1] / mb - cfg_.entropy_coef * stats[3] / mb;
}

// ---------------- 체크포인트 ----------------
namespace {
constexpr uint32_t kMagic = 0x4F505058;  // "XPPO"
constexpr uint32_t kVersion = 3;  // 3: actor·critic 인코더 나눔(+ 프론티어 채널, 2026-11-08). 2 = 공유 인코더 (load_convert_shared 로만), 1 = 거부
constexpr uint32_t kVersionShared = 2;
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
  if (hdr[0] == kMagic && hdr[1] == kVersionShared) throw std::runtime_error("형식 2(공유 인코더) 체크포인트 — load_convert_shared 로 바꿔 불러야 함: " + path);
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

void PPO::load_convert_shared(const std::string& path) {
  if (!ac_->split) throw std::runtime_error("load_convert_shared: 인코더 나눔 구조에서만");
  std::ifstream f(path, std::ios::binary);
  if (!f) throw std::runtime_error("체크포인트 없음: " + path);
  auto r = [&](void* d, size_t n) {
    f.read((char*)d, n);
    if (!f) throw std::runtime_error("체크포인트가 잘림: " + path);
  };
  uint32_t hdr[3];
  r(hdr, 12);
  if (hdr[0] != kMagic || hdr[1] != kVersionShared || hdr[2] != sizeof(real)) throw std::runtime_error("변환: 형식 2(공유 인코더) 체크포인트가 아님");
  int32_t nb;
  r(&nb, 4);
  struct Old { std::string name; long off, n; };
  std::vector<Old> old;
  long off = 0;
  for (int i = 0; i < nb; ++i) {
    int32_t ln;
    r(&ln, 4);
    std::string nm(ln, '\0');
    r(nm.data(), ln);
    int64_t n;
    r(&n, 8);
    old.push_back({nm, off, (long)n});
    off += (long)n;
  }
  int64_t P0;
  r(&P0, 8);
  if (P0 != off) throw std::runtime_error("변환: 변수 수가 층 합과 다름");
  std::vector<real> p0(P0), m0(P0), v0(P0);
  r(p0.data(), P0 * sizeof(real));
  r(m0.data(), P0 * sizeof(real));
  r(v0.data(), P0 * sizeof(real));
  const long P = ac_->n_params;
  std::vector<real> p(P, 0), m(P, 0), v(P, 0);
  for (auto& b : ac_->blocks) {
    const std::string src = b.name.rfind("enc_c.", 0) == 0 ? "enc." + b.name.substr(6) : b.name;
    const Old* o = nullptr;
    for (auto& q : old) if (q.name == src) { o = &q; break; }
    if (!o || o->n != b.n) throw std::runtime_error("변환: 옛 체크포인트에 없는 층 " + src);
    std::copy(p0.begin() + o->off, p0.begin() + o->off + b.n, p.begin() + b.off);
    std::copy(m0.begin() + o->off, m0.begin() + o->off + b.n, m.begin() + b.off);
    std::copy(v0.begin() + o->off, v0.begin() + o->off + b.n, v.begin() + b.off);
  }
  // 프론티어 채널 첫 층 가중치 0 (두 인코더, 가까운 지도 cnn0·먼 지도 cnn1). W [cout][자식 4][cin]
  const int chs[2] = {mdp::obs::NEAR_FRONTIER_CH, mdp::obs::FAR_FRONTIER_CH};
  for (const char* pre : {"enc.", "enc_c."})
    for (int br = 0; br < 2; ++br) {
      const std::string nm = std::string(pre) + "cnn" + std::to_string(br) + ".conv0.w";
      const int cin = ac_->spec.enc.imgs[br].c;
      for (auto& b : ac_->blocks)
        if (b.name == nm) {
          const long cout = b.n / (4L * cin);
          for (long co = 0; co < cout; ++co)
            for (int ch4 = 0; ch4 < 4; ++ch4) {
              const long k = b.off + co * 4 * cin + ch4 * cin + chs[br];
              p[k] = m[k] = v[k] = 0;
            }
        }
    }
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
  clamp_std();
  ac_->params_changed();
  adam_t_ = at;
  iter_ = it;
  lr_ = lr;
  act_calls_ = acn;
  seed_ = sd;
  step_ = 0;
}

}  // namespace XPPO_NS
