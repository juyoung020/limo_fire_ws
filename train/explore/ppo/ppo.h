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
  float gamma = 0.999f;         // mdp/README "판 끝" — 0.997 → 0.999 (2026-11-05: 진행 없음 60 s = 600 걸음이 지평 333 걸음 밖이라 안 보였다)
  float lam = 0.95f;
  float value_coef = 1.0f;
  bool clipped_value = true;
  // 2026-11-06 0.001 → 0.01 (legged_gym legged_robot_config.py entropy_coef = 0.01): 장기 학습에서 σ 가 0.27 → 0.087 (ω 는 하한 0.05) 로 무너지며
  // 학습 맵·시험 맵 모두 나빠짐 — ppo/README "장기 하락". σ 상한 1 (std_max 기본) 과 같이 써야 함 (상한 없이 0.005 에서 σ 폭주했던 explore_v3).
  float entropy_coef = 0.01f;   // 가우시안 성분만. (옛) 0.005 → 0.001 (2026-11-05): 0.005 → 0.001 (2026-11-05): 행동을 [−1, 1] 로 자르는 환경에서 σ 를 키워도 손해가 없어
                                // 0.005 는 σ 를 상한까지 밀었다(explore_v3 σ 1 → 99.6). ppo/README "σ 범위"
  // 정책 σ 범위 [std_min, std_max] — 매 Adam 걸음 뒤·체크포인트 불러오기 뒤에 log σ 변수를 [ln std_min, ln std_max] 로 투영(사영 경사 하강).
  // 변수 자체가 늘 범위 안이라 샘플·log π·엔트로피·KL·체크포인트·onnx_export·재생이 모두 같은 σ 를 쓰고 손실 기울기는 clamp 없는 정확한 값.
  // std_max = 1: 정책 출력은 정규 단위(env.h apply_action 이 clip1 후 한도를 곱함)라 세 성분 모두 행동 한계 반폭 = 1 (= 초기 σ).
  // std_min = 0.05: 잠정 — 반폭의 5 %(ω 0.16 rad/s, v 0.05 m/s). 이보다 작으면 log π 의 1/σ 민감도가 커지고 탐사가 사라짐. 평가는 평균 행동이라 영향 없음.
  // 방식 비교·출처(rsl_rl GaussianDistribution std_range, SB3 SAC LOG_STD_MIN/MAX)는 ppo/README "σ 범위".
  float std_min = 0.05f, std_max = 1.0f;
  // 평균 행동 경계 손실 (rl_games a2c_continuous.py bound_loss, soft_bound 1.1): 환경이 [−1, 1] 로 자르므로 평균 μ 가 경계 밖으로 나가면
  // 표본이 모두 잘려 같은 행동이 되고 대리 손실 기울기가 0 이 된다 — μ 가 기울기 없이 멀리 떠내려가 그 성분을 다시 못 배운다.
  // explore_v7 반복 3900·14000 시험 맵 결정적 평가: 걸음의 95 % 에서 μ_v > 1, 평균 μ_v 3.0 → 3.9 (σ_v 0.50 → 0.25) — 벽 0.4 m 안에서도
  // 앞 명령이 86~94 % 최고였다(감속을 못 배움). coef 는 rl_games 기본(1e-4)보다 크게: 우리 미니배치에서 잘린 성분의 대리 손실 기울기 잡음
  // (≈ |A|/σ/√n ≈ 0.04)보다 경계 기울기(2·c·(μ − b))가 커야 μ 가 돌아온다. 0 = 끔 (explore_v7 과 비트 같음). ppo/README "평균 행동 경계"
  float bound_coef = 0.1f, bound_soft = 1.1f;
  float lr = 1.0e-3f;
  float desired_kl = 0.003f;    // adaptive 학습률. 2026-11-06 0.01 → 0.003: 적응 학습률이 업데이트마다 KL 0.013 만큼 정책을 계속 움직여 σ 가 작아진 뒤 신호 없이 표류 — ppo/README "장기 하락"
  float lr_min = 1e-5f, lr_max = 1e-2f;  // KL 적응 학습률 한계 (rsl_rl ppo.py:291,293)
  float max_grad_norm = 1.0f;
  float adam_b1 = 0.9f, adam_b2 = 0.999f, adam_eps = 1e-8f;  // torch.optim.Adam 기본
  int chunk = 0;                // 한 번에 신경망에 넣는 샘플 수 (메모리 한도 — 넘는 미니배치는 나눠 기울기를 더한다). 0 = 미니배치 하나(N·T/미니배치 수)를 한 번에:
                                // 2048 환경이면 12 288, update 717 → 690 ms (4096 대비, ppo/README "속도 4"), GPU 메모리 +0.7 GB
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
  // actions float [N][n_cont+1] (연속 = 샘플 원값, 선언 = 0/1). 계층형(n_slots > 0)이면 actions [N][mdp::act::DIM], a[0] = 고른 슬롯 (가린 범주 분포). 롤아웃 칸이 가득 차 있으면 예외.
  void act(const uint8_t* img, const real* obs1d, const real* priv, real* actions);
  // reward float [N], done int [N]. 끊김(3)이면 rsl_rl 처럼 r += γ·V(s_t).
  void store(const real* reward, const int* done);
  // 바로 앞 act 의 환경 0 … n−1 선언 로짓·가치를 호스트로 (뷰어 재생용 — store 전에 부른다)
  void last_act_info(int n, real* logit, real* value) const;
  // 롤아웃이 가득 찬 뒤 마지막 관측으로 가치를 메워 GAE (update 가 부른다; 검사 도구도 따로 부른다)
  void compute_returns(const uint8_t* img, const real* obs1d, const real* priv);
  // 롤아웃이 가득 찬 뒤. 마지막 관측(다음 걸음 관측)으로 가치를 메워 GAE·PPO 갱신. 반복 수 +1.
  UpdateStats update(const uint8_t* img, const real* obs1d, const real* priv);
  // 평가 (정규화 통계·학습 난수 안 바꿈). actions 는 [N][n_cont+1]. sample_call < 0: 결정적(평균 행동, 선언 = 확률 > 0.5).
  // sample_call ≥ 0: 정책 분포에서 샘플(학습과 같은 분포, 다른 난수 계열 — 걸음마다 다른 값을 넘긴다). 학습 성공과 평가 성공의 차이를 가르려고.
  void act_inference(const uint8_t* img, const real* obs1d, real* actions, int n = -1, int64_t sample_call = -1);

  // 체크포인트: 가중치·Adam(m, v, 걸음)·정규화 통계·반복 수·학습률·난수 계수. 롤아웃 도중 저장은 안 됨.
  void save(const std::string& path) const;
  void load(const std::string& path);
  // 형식 2(공유 인코더) 체크포인트를 형식 3(인코더 나눔)으로 바꿔 불러옴 — 이 길로만 옛 체크포인트를 받는다.
  // 공유 인코더 가중치·Adam m/v 를 actor 인코더(enc)와 critic 인코더(enc_c)에 그대로 복사, MLP·log σ·정규화·반복 수·학습률은 그대로,
  // 프론티어 채널(mdp/obs_layout.h NEAR/FAR_FRONTIER_CH — 형식 2 에서는 늘 0 인 예약 칸) 첫 층 가중치와 그 m/v 는 0 →
  // 반복 0 에서 정책·가치가 옛 체크포인트와 같다 (새 채널 값과 무관)
  void load_convert_shared(const std::string& path);

  int iteration() const { return iter_; }
  float learning_rate() const { return lr_; }
  const PPOConfig& config() const { return cfg_; }
  XNET_NS::ActorCritic& ac() { return *ac_; }
  int rollout_step() const { return step_; }
  size_t storage_bytes() const { return storage_bytes_; }

  // 손실 하나(미니배치 하나 분량)의 기울기를 d_grads 에 쓴다 — 유한 차분 검사용. 저장된 롤아웃의 idx 앞 n 개.
  double loss_and_grad(const int* d_idx, int n, bool backward);

 private:
  void clamp_std();  // log σ 변수를 [ln std_min, ln std_max] 로 (Adam 걸음·불러오기 뒤)
  void concat_norm(const real* obs1d, const real* priv, int n, bool upd, real* out_norm);
  void encode(int n, const uint8_t* img, const int* idx, const real* x1d);
  // 인코더 나눔(split): critic 인코더 정방향 → c_xc_ = [F_c | x1d] (stream2). x1d = nullptr 이면 1D 열은 이미 c_xc_ 에
  void encode_c(int n, const uint8_t* img, const int* idx, const real* x1d);
  // critic 입력 버퍼 (split 이면 c_xc_, 아니면 공유 c_x_)
  real* cx() const { return c_xc_ ? c_xc_ : c_x_; }
  uint16_t* cxh() const { return c_xc_ ? c_xch_ : c_xh_; }
  double minibatch(const int* d_idx, int n, int mb_size, bool backward, double stats[4]);

  PPOConfig cfg_;
  XNET_NS::ActorCritic* ac_;
  uint64_t seed_;
  int N_, T_, S_, D_, IMG_, A_, NC_, NF_ = 0;
  // 계층형 (spec.n_slots > 0): 범주 행동 — 슬롯 K_, actor 출력(임베딩) AO_, 환경 행동 칸 ADIM_ (mdp::act::DIM), 후보 특징 자리 SOFF_·칸 SF_
  bool disc_ = false;
  int K_ = 0, AO_ = 0, ADIM_ = 0, SOFF_ = 0, SF_ = 0;
  real *s_mask_ = nullptr, *c_logits_ = nullptr, *c_G_ = nullptr, *c_feat_ = nullptr, *c_maskt_ = nullptr;
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
  real *c_aout_, *c_v_, *c_da_, *c_dv_, *c_dls_, *c_stat_, *c_red_, *c_gat_;
  real* c_x_ = nullptr;  // MLP 입력 [F | 1D] (인코더가 F 열을, 모으기·act 가 1D 열을 바로 씀)| 1D], 인코더 특징 기울기
  uint16_t* c_xh_ = nullptr;  // 그 bf16 사본 (bf16 MLP 만 — 첫 층 cuBLAS bf16 입력)
  real* c_xc_ = nullptr;      // 인코더 나눔: critic MLP 입력 [F_c | 1D]
  uint16_t* c_xch_ = nullptr; // 그 bf16 사본
  int* d_perm_;
  size_t storage_bytes_ = 0;
};

}  // namespace XPPO_NS
