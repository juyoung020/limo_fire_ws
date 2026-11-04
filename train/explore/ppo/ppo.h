// 탐사 RL PPO (rsl_rl algorithms/ppo.py 꼴을 C++/CUDA 로). 하이퍼파라미터·근거는 ppo/README.md.
//
// 러너 쓰임 (모든 포인터는 GPU, N = num_envs):
//   xppo::PPO ppo(xppo::PPOConfig{.num_envs = N}, xnet::explore_spec(), seed);   (본 학습은 cfg.lr_max = 1e-3 권장 — README)
//   for (;;) {                                   // 반복 하나
//     for (int t = 0; t < cfg.steps_per_env; ++t) {
//       ppo.act(obs_img, obs_1d, priv, actions);  // 샘플·기록 (정규화 통계도 갱신)
//       env.step(actions) → reward, done, 새 obs
//       ppo.store(reward, done);                 // done: 0 진행, 1 종료(선언·충돌), 3 끊김(time-out)
//     }
//     auto st = ppo.update(obs_img, obs_1d, priv); // 마지막 관측으로 가치 메움 → GAE → PPO
//     if (...) ppo.save(path);
//   }
//   평가: ppo.act_inference(obs_img, obs_1d, actions)  // 평균 행동, 선언 = 확률 > 0.5
#pragma once

#include <cstdint>
#include <string>

#include "../network/actor_critic.h"

namespace XPPO_NS {

using XNET_NS::real;  // 학습 빌드 = float

struct PPOConfig {
  int num_envs = 2048;
  int steps_per_env = 24;       // legged_gym num_steps_per_env
  int epochs = 5;               // num_learning_epochs
  int minibatches = 4;          // num_mini_batches
  float clip = 0.2f;            // clip_param
  float gamma = 0.997f;         // mdp/README "판 끝"
  float lam = 0.95f;
  float value_coef = 1.0f;
  bool clipped_value = true;
  float entropy_coef = 0.005f;  // 가우시안 성분만
  float lr = 1.0e-3f;
  float desired_kl = 0.01f;     // adaptive 학습률
  float lr_min = 1e-5f, lr_max = 1e-2f;  // KL 적응 학습률 한계 (rsl_rl ppo.py:291,293)
  float max_grad_norm = 1.0f;
  float adam_b1 = 0.9f, adam_b2 = 0.999f, adam_eps = 1e-8f;  // torch.optim.Adam 기본
  int chunk = 4096;             // 한 번에 신경망에 넣는 샘플 수 (메모리 한도 — 기울기는 더해진다). 합성곱 융합 뒤 작업 공간이 작아 1024 → 4096: update −10 % (ppo/README "속도")
  bool cnn_bf16 = true;         // 합성곱 가지를 텐서 코어(bf16 입력·fp32 누산)로. false = fp32 융합 커널(정확 경로, 기울기 검사가 씀). network/README "계산 방식"
  bool tf32 = false;            // cuBLAS TF32 텐서 코어. 2026-11-04 실측: update 6.5 → 7.1 s 로 오히려 느려 기본 끔 (ppo/README "속도")
  int mlp = 2;                  // MLP 계산: 0 cuBLAS(옛 길), 1 융합 GEMM fp32(정확, 기울기 검사), 2 융합 GEMM bf16 텐서 코어·fp32 누산 (ppo/README "속도")
};

struct UpdateStats {
  float surrogate = 0, value = 0, entropy = 0, kl = 0, lr = 0, grad_norm = 0;
  float mean_value = 0, mean_return = 0;
};

class PPO {
 public:
  PPO(const PPOConfig& cfg, const XNET_NS::ACSpec& spec, uint64_t seed);
  ~PPO();
  PPO(const PPO&) = delete;
  PPO& operator=(const PPO&) = delete;

  // 학습 행동. img u8 [N][img_stride], obs1d float [N][actor.in1d], priv float [N][norm_dim − actor.in1d] (nullptr = 0),
  // actions float [N][n_cont+1] (연속 = 샘플 원값, 선언 = 0/1). 롤아웃 칸이 가득 차 있으면 예외.
  void act(const uint8_t* img, const real* obs1d, const real* priv, real* actions);
  // reward float [N], done int [N]. 끊김(3)이면 rsl_rl 처럼 r += γ·V(s_t).
  void store(const real* reward, const int* done);
  // 롤아웃이 가득 찬 뒤 마지막 관측으로 가치를 메워 GAE (update 가 부른다; 검사 도구도 따로 부른다)
  void compute_returns(const uint8_t* img, const real* obs1d, const real* priv);
  // 롤아웃이 가득 찬 뒤. 마지막 관측(다음 걸음 관측)으로 가치를 메워 GAE·PPO 갱신. 반복 수 +1.
  UpdateStats update(const uint8_t* img, const real* obs1d, const real* priv);
  // 결정적 평가 (정규화 통계 안 바꿈). actions 는 [N][n_cont+1]
  void act_inference(const uint8_t* img, const real* obs1d, real* actions, int n = -1);

  // 체크포인트: 가중치·Adam(m, v, 걸음)·정규화 통계·반복 수·학습률·난수 계수. 롤아웃 도중 저장은 안 됨.
  void save(const std::string& path) const;
  void load(const std::string& path);

  int iteration() const { return iter_; }
  float learning_rate() const { return lr_; }
  const PPOConfig& config() const { return cfg_; }
  XNET_NS::ActorCritic& ac() { return *ac_; }
  int rollout_step() const { return step_; }
  size_t storage_bytes() const { return storage_bytes_; }

  // 손실 하나(미니배치 하나 분량)의 기울기를 d_grads 에 쓴다 — 유한 차분 검사용. 저장된 롤아웃의 idx 앞 n 개.
  double loss_and_grad(const int* d_idx, int n, bool backward);

 private:
  void concat_norm(const real* obs1d, const real* priv, int n, bool upd, real* out_norm);
  void encode(int n, const uint8_t* img, const int* idx, const real* x1d);
  double minibatch(const int* d_idx, int n, int mb_size, bool backward, double stats[4]);

  PPOConfig cfg_;
  XNET_NS::ActorCritic* ac_;
  uint64_t seed_;
  int N_, T_, S_, D_, IMG_, A_, NC_, NF_ = 0;
  int step_ = 0, iter_ = 0;
  uint64_t act_calls_ = 0;
  float lr_;
  long adam_t_ = 0;
  real *m_ = nullptr, *v_ = nullptr;
  // 롤아웃 [T][N]
  uint8_t* s_img_ = nullptr;
  real *s_obs_, *s_act_, *s_logp_, *s_val_, *s_mu_, *s_logit_, *s_sd_, *s_rew_, *s_done_, *s_ret_, *s_adv_;
  // 작업 공간 (chunk)
  real *raw_, *last_obs_, *last_val_;
  uint8_t* c_img_;
  real *c_obs_, *c_aout_, *c_v_, *c_da_, *c_dv_, *c_dls_, *c_stat_, *c_red_, *c_gat_;
  real *c_x_ = nullptr, *c_df_ = nullptr;  // MLP 입력 [F | 1D], 인코더 특징 기울기
  int* d_perm_;
  size_t storage_bytes_ = 0;
};

}  // namespace XPPO_NS
