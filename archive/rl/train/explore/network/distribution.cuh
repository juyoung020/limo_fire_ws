// 행동 분포: 연속 성분 가우시안(평균 = 신경망, log σ = 상태 무관 변수, tanh 없음) + 선언 베르누이(로짓).
// log π = Σ 가우시안 + 베르누이 (mdp/README "행동"). 엔트로피 보너스는 가우시안만, KL 은 둘 다 더한다.
// 모두 __device__ 인라인 — ppo 의 샘플·손실 커널이 부른다.
#pragma once

#include "common.cuh"

namespace XNET_NS::dist {

constexpr real kHalfLog2Pi = 0.91893853320467274f;  // ½·ln(2π)

// log σ(x) 를 안정적으로 = −softplus(−x)
__device__ __forceinline__ real log_sigmoid(real x) { return x >= 0.f ? -log1p(exp(-x)) : x - log1p(exp(x)); }
__device__ __forceinline__ real sigmoid(real x) { return 1.f / (1.f + exp(-x)); }

// 가우시안 한 성분 log 밀도
__device__ __forceinline__ real gauss_logp(real a, real mu, real log_std) {
  real z = (a - mu) * exp(-log_std);
  return -0.5f * z * z - log_std - kHalfLog2Pi;
}
// 가우시안 한 성분 엔트로피 = ½ + ½ln2π + log σ
__device__ __forceinline__ real gauss_entropy(real log_std) { return 0.5f + kHalfLog2Pi + log_std; }

// 베르누이 log 확률 (d = 0/1)
__device__ __forceinline__ real bern_logp(real d, real logit) {
  return d > 0.5f ? log_sigmoid(logit) : log_sigmoid(-logit);
}

// KL(old ‖ new) — 가우시안은 rsl_rl algorithms/ppo.py(2.3.1) 식 그대로(+1e-5 포함)
__device__ __forceinline__ real gauss_kl(real mu_o, real sd_o, real mu_n, real sd_n) {
  real dm = mu_o - mu_n;
  return log(sd_n / sd_o + 1.0e-5f) + (sd_o * sd_o + dm * dm) / (2.f * sd_n * sd_n) - 0.5f;
}
__device__ __forceinline__ real bern_kl(real lo, real ln) {
  real po = sigmoid(lo);
  return po * (log_sigmoid(lo) - log_sigmoid(ln)) + (1.f - po) * (log_sigmoid(-lo) - log_sigmoid(-ln));
}

}  // namespace XNET_NS::dist
