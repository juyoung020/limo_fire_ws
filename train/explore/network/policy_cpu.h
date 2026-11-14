// 탐사 RL 계층형 정책(구조 버전 2, 인코더 나눔 = 체크포인트 형식 3)의 CPU 정방향 — CUDA 없이 젯슨·데스크톱 CPU 에서 결정 행동(슬롯)을 낸다.
// 실물 노드(src/limo_fire/limo_fire_rl)가 쓰고, ppo/tools/policy_cpu_check 가 학습기 act_inference(GPU)와 같은 슬롯·로짓(1e-4 안)인지 확인한다.
// 계산은 학습 정방향과 같은 식·같은 가중치 배치 (actor_critic.cu Net::build·network/tools/onnx_export.cpp 와 같은 해석):
//   u8 / 255 → 가지마다 2×2 s2 합성곱 + ELU × 3 → 셋째 층 격자가 홀수면 아래·오른쪽 0 덧댐 → 넷째 층 → [픽셀(래스터)][채널] 로 펼침 → 가지 3 개 이어 F (1024)
//   1D: (x − μ) / (√σ² + 1e-2) (rsl_rl EmpiricalNormalization, 체크포인트 통계 — 정책 1D 앞 in1d 칸만)
//   actor MLP [F | 정규화 1D] → 512 · 256 · 128 ELU → 임베딩 64 (마지막 층은 선형)
//   슬롯 머리 (slot_head.cu k_slot_fwd): 로짓_k = Σ_h w2[h] · ELU(b1[h] + W1[h] · [z ; 정규화 1D 의 후보 k 칸 7 개])
//   가림: 날 1D 의 후보 k 유효 칸 > 0.5 (ppo.cu k_cat_mask) · 결정 행동 = 있는 칸 중 로짓 최대(같으면 앞 칸, 없으면 0 — ppo.cu k_cat_act)
// 체크포인트 형식은 ppo.cu PPO::save: 머리(XPPO, 형식, sizeof(real)) · 블록 이름·크기 · 변수(이어 붙인 순서 = 블록 순서) · Adam m·v · 걸음 · 정규화 μ·σ² …
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

#include "../mdp/obs_layout.h"

// NEON (aarch64 — 젯슨 Orin): 합성곱·MLP 의 곱합을 출력 4 개씩 한 벡터로 (출력마다 더하는 차례는 스칼라와 같음 → 결과 비트 같음, 가중치는 load 때 전치해 둠).
// XNET_CPU_SCALAR 를 정의하면 스칼라만 (견줌용)
#if (defined(__ARM_NEON) || defined(__ARM_NEON__)) && !defined(XNET_CPU_SCALAR)
#include <arm_neon.h>
#define XNET_CPU_NEON 1
#endif

namespace xnet_cpu {

class HierPolicy {
 public:
  static constexpr int kConvCh[4] = {16, 32, 32, 32};   // actor_critic.cu explore_spec conv_ch
  static constexpr int kHidden[3] = {512, 256, 128};
  static constexpr int kEmbed = 64, kSlotHidden = 64;    // explore_spec_hier actor.out · slot_hidden
  static constexpr float kNormEps = 1e-2f;               // Normalizer::kEps

  void load(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("체크포인트 없음: " + path);
    auto rd = [&](void* d, size_t n) {
      f.read(reinterpret_cast<char*>(d), (std::streamsize)n);
      if (!f) throw std::runtime_error("체크포인트가 잘림: " + path);
    };
    uint32_t hdr[3];
    rd(hdr, 12);
    if (hdr[0] != 0x4F505058u || (hdr[1] != 2 && hdr[1] != 3) || hdr[2] != 4) throw std::runtime_error("체크포인트 형식 2·3 (fp32) 이 아님: " + path);
    format_ = (int)hdr[1];
    int32_t nb;
    rd(&nb, 4);
    std::vector<std::pair<std::string, int64_t>> names;
    for (int i = 0; i < nb; ++i) {
      int32_t ln;
      rd(&ln, 4);
      if (ln <= 0 || ln > 256) throw std::runtime_error("체크포인트 블록 이름 길이");
      std::string nm((size_t)ln, '\0');
      rd(nm.data(), (size_t)ln);
      int64_t n;
      rd(&n, 8);
      names.push_back({nm, n});
    }
    int64_t P;
    rd(&P, 8);
    std::vector<float> p((size_t)P);
    rd(p.data(), (size_t)P * 4);
    f.seekg((std::streamoff)(2 * P * 4 + 8), std::ios::cur);   // Adam m·v, 걸음
    int32_t D;
    rd(&D, 4);
    mean_.resize((size_t)D);
    var_.resize((size_t)D);
    rd(mean_.data(), (size_t)D * 4);
    rd(var_.data(), (size_t)D * 4);
    double cnt;
    rd(&cnt, 8);
    int32_t it;
    rd(&it, 4);
    iteration_ = it;
    blk_.clear();
    int64_t off = 0;
    for (auto& [nm, n] : names) {
      if (off + n > P) throw std::runtime_error("체크포인트 블록 크기 합 > 변수 수");
      blk_[nm].assign(p.begin() + off, p.begin() + off + n);
      off += n;
    }
    if (off != P) throw std::runtime_error("체크포인트 블록 크기 합 ≠ 변수 수");
    if (!blk_.count("head.w1")) throw std::runtime_error("계층형(슬롯 머리) 체크포인트가 아님 — train --hier 로 학습한 것만: " + path);
    // 모양 확인 (mdp/obs_layout.h 와 같은 구조로 학습된 것인가)
    in1d_ = mdp::hier::POLICY_1D;
    if (D < in1d_) throw std::runtime_error("정규화 차원 < 정책 1D");
    const int ci[3] = {mdp::obs::NEAR_C, mdp::obs::FAR_C, mdp::obs::DEPTH_C};
    for (int b = 0; b < 3; ++b) {
      int c = ci[b];
      for (int l = 0; l < 4; ++l) {
        need("enc.cnn" + std::to_string(b) + ".conv" + std::to_string(l) + ".w", (size_t)kConvCh[l] * 4 * c);
        need("enc.cnn" + std::to_string(b) + ".conv" + std::to_string(l) + ".b", (size_t)kConvCh[l]);
        c = kConvCh[l];
      }
    }
    int in = kFeat + in1d_;
    const int dims[4] = {kHidden[0], kHidden[1], kHidden[2], kEmbed};
    for (int l = 0; l < 4; ++l) {
      need("actor.mlp" + std::to_string(l) + ".w", (size_t)dims[l] * in);
      need("actor.mlp" + std::to_string(l) + ".b", (size_t)dims[l]);
      in = dims[l];
    }
    need("head.w1", (size_t)kSlotHidden * (kEmbed + mdp::hier::FEAT));
    need("head.b1", (size_t)kSlotHidden);
    need("head.w2", (size_t)kSlotHidden);
    // 전치 가중치 (NEON 길): 합성곱 [cout][q][cin] → [q][cin][cout], MLP [out][in] → [in][out]
    tblk_.clear();
    for (int b = 0; b < 3; ++b) {
      int c = ci[b];
      for (int l = 0; l < 4; ++l) {
        const std::string nm = "enc.cnn" + std::to_string(b) + ".conv" + std::to_string(l) + ".w";
        const std::vector<float>& w = blk_.at(nm);
        const int co = kConvCh[l];
        std::vector<float> t(w.size());
        for (int k = 0; k < co; ++k)
          for (int q = 0; q < 4; ++q)
            for (int ch = 0; ch < c; ++ch) t[((size_t)q * c + ch) * co + k] = w[((size_t)k * 4 + q) * c + ch];
        tblk_[nm] = std::move(t);
        c = co;
      }
    }
    {
      int in2 = kFeat + in1d_;
      for (int l = 0; l < 4; ++l) {
        const std::string nm = "actor.mlp" + std::to_string(l) + ".w";
        const std::vector<float>& w = blk_.at(nm);
        const int out = dims[l];
        std::vector<float> t(w.size());
        for (int o = 0; o < out; ++o)
          for (int i = 0; i < in2; ++i) t[(size_t)i * out + o] = w[(size_t)o * in2 + i];
        tblk_[nm] = std::move(t);
        in2 = out;
      }
    }
  }

  // bf16 흉내 (학습·평가 기본 경로 — 합성곱·MLP 입력·가중치를 bf16 로 반올림, fp32 누산, 머리는 fp32): 실물은 끔(fp32)이 기본.
  // 학습기 기본 경로와 결정이 갈리는지 보려는 시험용 (rl_node_check --bf16)
  void set_bf16(bool on) { bf16_ = on; }
  int iteration() const { return iteration_; }
  int format() const { return format_; }

  // img: mdp::obs::IMG_SIZE 바이트, obs1d: mdp::hier::POLICY_1D (날 값). logits·mask: K 칸 (nullptr 이면 안 씀). 반환 = 결정 슬롯
  int act(const uint8_t* img, const float* obs1d, float* logits = nullptr, float* mask = nullptr) const {
    using namespace mdp::obs;
    std::vector<float> x((size_t)kFeat + in1d_);
    // 인코더 가지: near (0), far (1), depth (2)
    const int off[3] = {0, NEAR_SIZE, NEAR_SIZE + FAR_SIZE};
    const int C[3] = {NEAR_C, FAR_C, DEPTH_C}, H[3] = {NEAR_H, FAR_H, DEPTH_H}, W[3] = {NEAR_W, FAR_W, DEPTH_W};
    int flat = 0;
    for (int b = 0; b < 3; ++b) {
      int c = C[b], h = H[b], w = W[b];
      std::vector<float> a((size_t)c * h * w);
      for (size_t i = 0; i < a.size(); ++i) a[i] = bf16_ ? (float)img[off[b] + i] : (float)img[off[b] + i] * (1.f / 255.f);   // bf16 판: u8 그대로 곱하고 누산 뒤 /255
      for (int l = 0; l < 4; ++l) {
        if (l == 3 && (h % 2 || w % 2)) {   // 넷째 층: 홀수 격자를 아래·오른쪽 0 덧댐
          const int h2 = h + h % 2, w2 = w + w % 2;
          std::vector<float> p((size_t)c * h2 * w2, 0.f);
          for (int ch = 0; ch < c; ++ch)
            for (int y = 0; y < h; ++y)
              for (int xx = 0; xx < w; ++xx) p[((size_t)ch * h2 + y) * w2 + xx] = a[((size_t)ch * h + y) * w + xx];
          a.swap(p);
          h = h2, w = w2;
        }
        const std::string nm = "enc.cnn" + std::to_string(b) + ".conv" + std::to_string(l);
        const float* Wt = blk_.at(nm + ".w").data();   // [cout][자식 dy·2+dx][cin]
        const float* Bs = blk_.at(nm + ".b").data();
        const int co = kConvCh[l], oh = h / 2, ow = w / 2;
        std::vector<float> o((size_t)co * oh * ow);
#ifdef XNET_CPU_NEON
        if (!bf16_) {   // 출력 채널 4 개씩: 출력마다 (q, ch) 차례로 곱해 더함 — 스칼라 길과 같은 차례
          const float* WT = tblk_.at(nm + ".w").data();   // [q][cin][cout]
          for (int y = 0; y < oh; ++y)
            for (int xx = 0; xx < ow; ++xx)
              for (int k = 0; k < co; k += 16) {   // 출력 16 개 = 벡터 4 개 (서로 다른 곱합 사슬 넷 — FMA 지연을 가림). co 는 16·32
                float32x4_t a0 = vdupq_n_f32(0.f), a1 = a0, a2 = a0, a3 = a0;
                for (int q = 0; q < 4; ++q) {
                  const int iy = 2 * y + q / 2, ix = 2 * xx + q % 2;
                  const float* wq = WT + (size_t)q * c * co + k;
                  for (int ch = 0; ch < c; ++ch) {
                    const float v = a[((size_t)ch * h + iy) * w + ix];
                    const float* wc = wq + (size_t)ch * co;
                    a0 = vfmaq_n_f32(a0, vld1q_f32(wc), v);
                    a1 = vfmaq_n_f32(a1, vld1q_f32(wc + 4), v);
                    a2 = vfmaq_n_f32(a2, vld1q_f32(wc + 8), v);
                    a3 = vfmaq_n_f32(a3, vld1q_f32(wc + 12), v);
                  }
                }
                float s16[16];
                vst1q_f32(s16, vaddq_f32(a0, vld1q_f32(Bs + k)));
                vst1q_f32(s16 + 4, vaddq_f32(a1, vld1q_f32(Bs + k + 4)));
                vst1q_f32(s16 + 8, vaddq_f32(a2, vld1q_f32(Bs + k + 8)));
                vst1q_f32(s16 + 12, vaddq_f32(a3, vld1q_f32(Bs + k + 12)));
                for (int u = 0; u < 16; ++u) o[((size_t)(k + u) * oh + y) * ow + xx] = elu(s16[u]);
              }
          a.swap(o);
          c = co, h = oh, w = ow;
          continue;
        }
#endif
        for (int k = 0; k < co; ++k)
          for (int y = 0; y < oh; ++y)
            for (int xx = 0; xx < ow; ++xx) {
              float s = 0.f;
              for (int q = 0; q < 4; ++q) {
                const int iy = 2 * y + q / 2, ix = 2 * xx + q % 2;
                const float* wk = Wt + ((size_t)k * 4 + q) * c;
                for (int ch = 0; ch < c; ++ch) s += rb(wk[ch]) * a[((size_t)ch * h + iy) * w + ix];
              }
              s = (bf16_ && l == 0 ? s * (1.f / 255.f) : s) + Bs[k];
              o[((size_t)k * oh + y) * ow + xx] = rb(elu(s));
            }
        a.swap(o);
        c = co, h = oh, w = ow;
      }
      for (int pix = 0; pix < h * w; ++pix)   // [픽셀][채널]
        for (int ch = 0; ch < c; ++ch) x[(size_t)flat + (size_t)pix * c + ch] = a[(size_t)ch * h * w + pix];
      flat += c * h * w;
    }
    if (flat != kFeat) throw std::runtime_error("인코더 특징 수 ≠ 1024");
    for (int j = 0; j < in1d_; ++j) x[(size_t)kFeat + j] = (obs1d[j] - mean_[j]) / (std::sqrt(var_[j]) + kNormEps);
    // actor MLP (bf16 판: 입력·가중치 bf16, 은닉 출력 bf16 사본이 다음 층 입력, 마지막 출력은 fp32)
    std::vector<float> z = x;
    for (auto& v : z) v = rb(v);
    const int dims[4] = {kHidden[0], kHidden[1], kHidden[2], kEmbed};
    for (int l = 0; l < 4; ++l) {
      const std::string nm = "actor.mlp" + std::to_string(l);
      const float* Wt = blk_.at(nm + ".w").data();   // [out][in]
      const float* Bs = blk_.at(nm + ".b").data();
      const int in = (int)z.size(), out = dims[l];
      std::vector<float> y((size_t)out);
#ifdef XNET_CPU_NEON
      if (!bf16_) {   // 출력 4 개씩 (출력마다 입력 차례로 — 스칼라와 같은 차례)
        const float* WT = tblk_.at(nm + ".w").data();   // [in][out]
        for (int o = 0; o < out; o += 16) {   // 출력 16 개 = 벡터 4 개 (사슬 넷). out 은 512·256·128·64
          float32x4_t a0 = vld1q_f32(Bs + o), a1 = vld1q_f32(Bs + o + 4), a2 = vld1q_f32(Bs + o + 8), a3 = vld1q_f32(Bs + o + 12);
          for (int i = 0; i < in; ++i) {
            const float* wi = WT + (size_t)i * out + o;
            const float v = z[i];
            a0 = vfmaq_n_f32(a0, vld1q_f32(wi), v);
            a1 = vfmaq_n_f32(a1, vld1q_f32(wi + 4), v);
            a2 = vfmaq_n_f32(a2, vld1q_f32(wi + 8), v);
            a3 = vfmaq_n_f32(a3, vld1q_f32(wi + 12), v);
          }
          float s16[16];
          vst1q_f32(s16, a0);
          vst1q_f32(s16 + 4, a1);
          vst1q_f32(s16 + 8, a2);
          vst1q_f32(s16 + 12, a3);
          for (int u = 0; u < 16; ++u) y[o + u] = l < 3 ? elu(s16[u]) : s16[u];
        }
        z.swap(y);
        continue;
      }
#endif
      for (int o = 0; o < out; ++o) {
        float s = Bs[o];
        const float* wr = Wt + (size_t)o * in;
        for (int i = 0; i < in; ++i) s += rb(wr[i]) * z[i];
        y[o] = l < 3 ? rb(elu(s)) : s;
      }
      z.swap(y);
    }
    // 슬롯 머리
    const int K = mdp::hier::K, F = mdp::hier::FEAT, E = kEmbed, Hh = kSlotHidden;
    const float* W1 = blk_.at("head.w1").data();
    const float* B1 = blk_.at("head.b1").data();
    const float* W2 = blk_.at("head.w2").data();
    int best_k = -1;
    float best = -1e30f;
    for (int k = 0; k < K; ++k) {
      const float* fk = x.data() + kFeat + mdp::obs::POLICY_1D + k * F;
      float s = 0.f;
      for (int hh = 0; hh < Hh; ++hh) {
        const float* wr = W1 + (size_t)hh * (E + F);
        float a = B1[hh];
        for (int e = 0; e < E; ++e) a += wr[e] * z[e];
        for (int c = 0; c < F; ++c) a += wr[E + c] * fk[c];
        s += W2[hh] * elu(a);
      }
      const bool valid = obs1d[mdp::obs::POLICY_1D + k * F + F - 1] > 0.5f;
      if (logits) logits[k] = s;
      if (mask) mask[k] = valid ? 1.f : 0.f;
      if (valid && (best_k < 0 || s > best)) { best = s; best_k = k; }
    }
    return best_k < 0 ? 0 : best_k;
  }

 private:
  static constexpr int kFeat = 1024;   // 인코더 특징 (network/README)
  static float elu(float v) { return v > 0.f ? v : std::expm1(v); }
  float rb(float v) const {   // bf16 반올림 (가장 가까운 짝수, __float2bfloat16_rn) — bf16_ 일 때만
    if (!bf16_ || !std::isfinite(v)) return v;
    uint32_t u;
    std::memcpy(&u, &v, 4);
    u = (u + 0x7FFFu + ((u >> 16) & 1u)) & 0xFFFF0000u;
    std::memcpy(&v, &u, 4);
    return v;
  }
  bool bf16_ = false;
  void need(const std::string& n, size_t sz) const {
    auto it = blk_.find(n);
    if (it == blk_.end() || it->second.size() != sz)
      throw std::runtime_error("체크포인트 블록 " + n + " 없음 또는 크기 다름 (구조가 obs_layout.h 와 다름)");
  }
  std::map<std::string, std::vector<float>> blk_;
  std::map<std::string, std::vector<float>> tblk_;   // 전치 가중치 (NEON 길)
  std::vector<float> mean_, var_;
  int in1d_ = 0, iteration_ = 0, format_ = 0;
};


// ---- 아래층 정책 (explore_spec_low(wide = true, gru_h) — 순수 RL 목적지 찾기: 넓은 지도 가지 + 순환 핵 GRU) — 실물 노드가 10 Hz 로 부른다 ----
// 학습기 act_inference(결정적)와 같은 식: 가지 4 개(가까운 지도 · 라이다 그림 · 깊이 · 넓은 지도, 2×2 s2 합성곱 + ELU × 4, 넷째 층 앞 홀수 격자 덧댐)
// → F (1536) · 1D 정규화 → GRU 한 걸음 (입력 [F | 1D], PyTorch nn.GRU 식, h 는 이 객체가 이어 감 — 판이 끝나면 reset()) → actor MLP [F | h | 1D] → 256 · 128 · 64 ELU → 3
// 행동: (v, ω) = 평균 그대로 (정규 단위 — 환경이 자르고 한도를 곱함), 선언 = 로짓 > 0. 가중치 배치는 actor_critic.cu · gru.cuh 그대로.
// 학습기 기본 경로(합성곱·MLP bf16 텐서 코어)와는 반올림이 달라 비트 같을 수 없다 — ppo/tools/low_cpu_check 가 차이를 잰다 (정확 경로와 1e-4 안).
class LowPolicy {
 public:
  static constexpr int kConvCh[4] = {16, 32, 32, 32};
  static constexpr int kHidden[3] = {256, 128, 64};
  static constexpr float kNormEps = 1e-2f;

  void load(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("체크포인트 없음: " + path);
    auto rd = [&](void* d, size_t n) {
      f.read(reinterpret_cast<char*>(d), (std::streamsize)n);
      if (!f) throw std::runtime_error("체크포인트가 잘림: " + path);
    };
    uint32_t hdr[3];
    rd(hdr, 12);
    if (hdr[0] != 0x4F505058u || hdr[1] != 3 || hdr[2] != 4) throw std::runtime_error("체크포인트 형식 3 (fp32) 이 아님: " + path);
    int32_t nb;
    rd(&nb, 4);
    std::vector<std::pair<std::string, int64_t>> names;
    for (int i = 0; i < nb; ++i) {
      int32_t ln;
      rd(&ln, 4);
      if (ln <= 0 || ln > 256) throw std::runtime_error("체크포인트 블록 이름 길이");
      std::string nm((size_t)ln, '\0');
      rd(nm.data(), (size_t)ln);
      int64_t n;
      rd(&n, 8);
      names.push_back({nm, n});
    }
    int64_t P;
    rd(&P, 8);
    std::vector<float> p((size_t)P);
    rd(p.data(), (size_t)P * 4);
    f.seekg((std::streamoff)(2 * P * 4 + 8), std::ios::cur);
    int32_t D;
    rd(&D, 4);
    mean_.resize((size_t)D);
    var_.resize((size_t)D);
    rd(mean_.data(), (size_t)D * 4);
    rd(var_.data(), (size_t)D * 4);
    double cnt;
    rd(&cnt, 8);
    int32_t it;
    rd(&it, 4);
    iteration_ = it;
    blk_.clear();
    int64_t off = 0;
    for (auto& [nm, n] : names) {
      if (off + n > P) throw std::runtime_error("체크포인트 블록 크기 합 > 변수 수");
      blk_[nm].assign(p.begin() + off, p.begin() + off + n);
      off += n;
    }
    if (D < mdp::low::N1D) throw std::runtime_error("정규화 차원 < 아래층 1D");
    if (!blk_.count("actor.gru.wh") || !blk_.count("enc.cnn3.conv0.w"))
      throw std::runtime_error("넓은 지도 가지·순환 핵 아래층 체크포인트가 아님 (train --low --gru, SIM_LOW_WIDE=1): " + path);
    H_ = (int)(blk_.at("actor.gru.bh").size() / 3);
    for (int b = 0; b < 4; ++b) {
      int c = kC[b];
      for (int l = 0; l < 4; ++l) {
        need("enc.cnn" + std::to_string(b) + ".conv" + std::to_string(l) + ".w", (size_t)kConvCh[l] * 4 * c);
        need("enc.cnn" + std::to_string(b) + ".conv" + std::to_string(l) + ".b", (size_t)kConvCh[l]);
        c = kConvCh[l];
      }
    }
    const int nin = kFeat + mdp::low::N1D;
    need("actor.gru.wx", (size_t)3 * H_ * nin);
    need("actor.gru.bx", (size_t)3 * H_);
    need("actor.gru.wh", (size_t)3 * H_ * H_);
    int in = kFeat + H_ + mdp::low::N1D;
    const int dims[4] = {kHidden[0], kHidden[1], kHidden[2], mdp::low::ACT};
    for (int l = 0; l < 4; ++l) {
      need("actor.mlp" + std::to_string(l) + ".w", (size_t)dims[l] * in);
      need("actor.mlp" + std::to_string(l) + ".b", (size_t)dims[l]);
      in = dims[l];
    }
    // 전치 (NEON 길): 합성곱 [cout][q][cin] → [q][cin][cout], 선형 [out][in] → [in][out] (GRU Wh 는 이미 [k][3H])
    tblk_.clear();
    for (int b = 0; b < 4; ++b) {
      int c = kC[b];
      for (int l = 0; l < 4; ++l) {
        const std::string nm = "enc.cnn" + std::to_string(b) + ".conv" + std::to_string(l) + ".w";
        const std::vector<float>& w = blk_.at(nm);
        const int co = kConvCh[l];
        std::vector<float> t(w.size());
        for (int k = 0; k < co; ++k)
          for (int q = 0; q < 4; ++q)
            for (int ch = 0; ch < c; ++ch) t[((size_t)q * c + ch) * co + k] = w[((size_t)k * 4 + q) * c + ch];
        tblk_[nm] = std::move(t);
        c = co;
      }
    }
    tblk_["actor.gru.wx"] = transpose(blk_.at("actor.gru.wx"), 3 * H_, nin);
    in = kFeat + H_ + mdp::low::N1D;
    for (int l = 0; l < 4; ++l) {
      const std::string nm = "actor.mlp" + std::to_string(l) + ".w";
      tblk_[nm] = transpose(blk_.at(nm), dims[l], in);
      in = dims[l];
    }
    h_.assign((size_t)H_, 0.f);
  }

  int hidden() const { return H_; }
  int iteration() const { return iteration_; }
  void reset() { std::fill(h_.begin(), h_.end(), 0.f); }   // 판(과제)이 새로 시작 — 학습의 판 끝 h = 0 과 같음
  const std::vector<float>& h() const { return h_; }
  // 아래층 관측: img = mdp::low::IMG_WIDE 바이트 (가까운 지도 · 라이다 그림 · 깊이 · 넓은 지도), obs1d = mdp::low::N1D (날 값).
  // out[ACT] = (v, ω 평균, 선언 로짓). 반환 = 선언 (로짓 > 0). h 를 한 걸음 진행
  bool act(const uint8_t* img, const float* obs1d, float* out) {
    using namespace mdp::low;
    std::vector<float> F((size_t)kFeat), xn((size_t)N1D);
    const int off[4] = {0, LIDAR_OFF, DEPTH_OFF, WIDE_OFF};
    const int Hs[4] = {mdp::obs::NEAR_H, LIDAR_HW, mdp::obs::DEPTH_H, WIDE_HW}, Ws[4] = {mdp::obs::NEAR_W, LIDAR_HW, mdp::obs::DEPTH_W, WIDE_HW};
    int flat = 0;
    for (int b = 0; b < 4; ++b) flat += branch(b, img + off[b], kC[b], Hs[b], Ws[b], F.data() + flat);
    if (flat != kFeat) throw std::runtime_error("아래층 인코더 특징 수 ≠ 1536");
    for (int j = 0; j < N1D; ++j) xn[j] = (obs1d[j] - mean_[j]) / (std::sqrt(var_[j]) + kNormEps);
    // GRU: gx = bx + Wx [F | 1D] (F 열 다음 1D 열), gh = bh + Whᵀ h
    const int H = H_, H3 = 3 * H, nin = kFeat + N1D;
    std::vector<float> xin((size_t)nin), gx((size_t)H3), gh((size_t)H3);
    std::copy(F.begin(), F.end(), xin.begin());
    std::copy(xn.begin(), xn.end(), xin.begin() + kFeat);
    linear(blk_.at("actor.gru.wx").data(), tblk_.at("actor.gru.wx").data(), blk_.at("actor.gru.bx").data(), xin.data(), nin, H3, gx.data(), false);
    linear_k(blk_.at("actor.gru.wh").data(), blk_.at("actor.gru.bh").data(), h_.data(), H, H3, gh.data());
    for (int j = 0; j < H; ++j) {
      const float r = sigm(gx[j] + gh[j]), z = sigm(gx[H + j] + gh[H + j]);
      const float n = std::tanh(gx[2 * H + j] + r * gh[2 * H + j]);
      h_[j] = (1.f - z) * n + z * h_[j];
    }
    // actor MLP [F | h | 1D]
    std::vector<float> zv((size_t)kFeat + H + N1D);
    std::copy(F.begin(), F.end(), zv.begin());
    std::copy(h_.begin(), h_.end(), zv.begin() + kFeat);
    std::copy(xn.begin(), xn.end(), zv.begin() + kFeat + H);
    const int dims[4] = {kHidden[0], kHidden[1], kHidden[2], ACT};
    for (int l = 0; l < 4; ++l) {
      const std::string nm = "actor.mlp" + std::to_string(l);
      std::vector<float> y((size_t)dims[l]);
      linear(blk_.at(nm + ".w").data(), tblk_.at(nm + ".w").data(), blk_.at(nm + ".b").data(), zv.data(), (int)zv.size(), dims[l], y.data(), l < 3);
      zv.swap(y);
    }
    for (int j = 0; j < ACT; ++j) out[j] = zv[j];
    return zv[ACT - 1] > 0.f;
  }

 private:
  static constexpr int kFeat = 1536;   // 가지 4 개 평탄화 (512 + 128 + 384 + 512)
  static constexpr int kC[4] = {mdp::obs::NEAR_C, 1, mdp::obs::DEPTH_C, mdp::low::WIDE_C};
  static float elu(float v) { return v > 0.f ? v : std::expm1(v); }
  static float sigm(float v) { return 1.f / (1.f + std::exp(-v)); }
  static std::vector<float> transpose(const std::vector<float>& w, int out, int in) {
    std::vector<float> t(w.size());
    for (int o = 0; o < out; ++o)
      for (int i = 0; i < in; ++i) t[(size_t)i * out + o] = w[(size_t)o * in + i];
    return t;
  }
  // y[o] = b[o] + Σ_i W[o][i] x[i] (i 차례) [+ ELU]. NEON: 출력 16 개씩 (출력마다 같은 차례 — 스칼라와 비트 같음)
  static void linear(const float* W, const float* WT, const float* b, const float* x, int in, int out, float* y, bool act) {
#ifdef XNET_CPU_NEON
    int o = 0;
    for (; o + 16 <= out; o += 16) {
      float32x4_t a0 = vld1q_f32(b + o), a1 = vld1q_f32(b + o + 4), a2 = vld1q_f32(b + o + 8), a3 = vld1q_f32(b + o + 12);
      for (int i = 0; i < in; ++i) {
        const float* wi = WT + (size_t)i * out + o;
        const float v = x[i];
        a0 = vfmaq_n_f32(a0, vld1q_f32(wi), v);
        a1 = vfmaq_n_f32(a1, vld1q_f32(wi + 4), v);
        a2 = vfmaq_n_f32(a2, vld1q_f32(wi + 8), v);
        a3 = vfmaq_n_f32(a3, vld1q_f32(wi + 12), v);
      }
      float s16[16];
      vst1q_f32(s16, a0);
      vst1q_f32(s16 + 4, a1);
      vst1q_f32(s16 + 8, a2);
      vst1q_f32(s16 + 12, a3);
      for (int u = 0; u < 16; ++u) y[o + u] = act ? elu(s16[u]) : s16[u];
    }
    for (; o < out; ++o) {   // 남는 출력 (마지막 층 3 개)
      float s = b[o];
      for (int i = 0; i < in; ++i) s += WT[(size_t)i * out + o] * x[i];
      y[o] = act ? elu(s) : s;
    }
    (void)W;
#else
    (void)WT;
    for (int o = 0; o < out; ++o) {
      float s = b[o];
      const float* wr = W + (size_t)o * in;
      for (int i = 0; i < in; ++i) s += wr[i] * x[i];
      y[o] = act ? elu(s) : s;
    }
#endif
  }
  // GRU 순환 쪽: y[c] = b[c] + Σ_k WhT[k][c] h[k] (k 차례) — WhT 가 이미 [k][3H] 라 전치 없이 출력 방향으로 벡터
  static void linear_k(const float* WhT, const float* b, const float* h, int H, int out, float* y) {
#ifdef XNET_CPU_NEON
    for (int o = 0; o < out; o += 16) {
      float32x4_t a0 = vld1q_f32(b + o), a1 = vld1q_f32(b + o + 4), a2 = vld1q_f32(b + o + 8), a3 = vld1q_f32(b + o + 12);
      for (int k = 0; k < H; ++k) {
        const float* wk = WhT + (size_t)k * out + o;
        const float v = h[k];
        a0 = vfmaq_n_f32(a0, vld1q_f32(wk), v);
        a1 = vfmaq_n_f32(a1, vld1q_f32(wk + 4), v);
        a2 = vfmaq_n_f32(a2, vld1q_f32(wk + 8), v);
        a3 = vfmaq_n_f32(a3, vld1q_f32(wk + 12), v);
      }
      vst1q_f32(y + o, a0);
      vst1q_f32(y + o + 4, a1);
      vst1q_f32(y + o + 8, a2);
      vst1q_f32(y + o + 12, a3);
    }
#else
    for (int o = 0; o < out; ++o) {
      float s = b[o];
      for (int k = 0; k < H; ++k) s += WhT[(size_t)k * out + o] * h[k];
      y[o] = s;
    }
#endif
  }
  // 가지 하나: u8 → /255 → 합성곱 4 층 → [픽셀][채널] 로 dst 에. 반환 = 특징 수
  int branch(int b, const uint8_t* src, int c, int h, int w, float* dst) const {
    std::vector<float> a((size_t)c * h * w);
    for (size_t i = 0; i < a.size(); ++i) a[i] = (float)src[i] * (1.f / 255.f);
    for (int l = 0; l < 4; ++l) {
      if (l == 3 && (h % 2 || w % 2)) {
        const int h2 = h + h % 2, w2 = w + w % 2;
        std::vector<float> p((size_t)c * h2 * w2, 0.f);
        for (int ch = 0; ch < c; ++ch)
          for (int y = 0; y < h; ++y)
            for (int xx = 0; xx < w; ++xx) p[((size_t)ch * h2 + y) * w2 + xx] = a[((size_t)ch * h + y) * w + xx];
        a.swap(p);
        h = h2, w = w2;
      }
      const std::string nm = "enc.cnn" + std::to_string(b) + ".conv" + std::to_string(l);
      const float* Wt = blk_.at(nm + ".w").data();
      const float* Bs = blk_.at(nm + ".b").data();
      const int co = kConvCh[l], oh = h / 2, ow = w / 2;
      std::vector<float> o((size_t)co * oh * ow);
#ifdef XNET_CPU_NEON
      const float* WT = tblk_.at(nm + ".w").data();
      for (int y = 0; y < oh; ++y)
        for (int xx = 0; xx < ow; ++xx)
          for (int k = 0; k < co; k += 16) {
            float32x4_t a0 = vdupq_n_f32(0.f), a1 = a0, a2 = a0, a3 = a0;
            for (int q = 0; q < 4; ++q) {
              const int iy = 2 * y + q / 2, ix = 2 * xx + q % 2;
              const float* wq = WT + (size_t)q * c * co + k;
              for (int ch = 0; ch < c; ++ch) {
                const float v = a[((size_t)ch * h + iy) * w + ix];
                const float* wc = wq + (size_t)ch * co;
                a0 = vfmaq_n_f32(a0, vld1q_f32(wc), v);
                a1 = vfmaq_n_f32(a1, vld1q_f32(wc + 4), v);
                a2 = vfmaq_n_f32(a2, vld1q_f32(wc + 8), v);
                a3 = vfmaq_n_f32(a3, vld1q_f32(wc + 12), v);
              }
            }
            float s16[16];
            vst1q_f32(s16, vaddq_f32(a0, vld1q_f32(Bs + k)));
            vst1q_f32(s16 + 4, vaddq_f32(a1, vld1q_f32(Bs + k + 4)));
            vst1q_f32(s16 + 8, vaddq_f32(a2, vld1q_f32(Bs + k + 8)));
            vst1q_f32(s16 + 12, vaddq_f32(a3, vld1q_f32(Bs + k + 12)));
            for (int u = 0; u < 16; ++u) o[((size_t)(k + u) * oh + y) * ow + xx] = elu(s16[u]);
          }
      (void)Wt;
#else
      for (int k = 0; k < co; ++k)
        for (int y = 0; y < oh; ++y)
          for (int xx = 0; xx < ow; ++xx) {
            float s = 0.f;
            for (int q = 0; q < 4; ++q) {
              const int iy = 2 * y + q / 2, ix = 2 * xx + q % 2;
              const float* wk = Wt + ((size_t)k * 4 + q) * c;
              for (int ch = 0; ch < c; ++ch) s += wk[ch] * a[((size_t)ch * h + iy) * w + ix];
            }
            o[((size_t)k * oh + y) * ow + xx] = elu(s + Bs[k]);
          }
#endif
      a.swap(o);
      c = co, h = oh, w = ow;
    }
    for (int pix = 0; pix < h * w; ++pix)
      for (int ch = 0; ch < c; ++ch) dst[(size_t)pix * c + ch] = a[(size_t)ch * h * w + pix];
    return c * h * w;
  }
  void need(const std::string& n, size_t sz) const {
    auto it = blk_.find(n);
    if (it == blk_.end() || it->second.size() != sz)
      throw std::runtime_error("체크포인트 블록 " + n + " 없음 또는 크기 다름 (구조가 explore_spec_low(true, H) 와 다름)");
  }
  std::map<std::string, std::vector<float>> blk_, tblk_;
  std::vector<float> mean_, var_, h_;
  int H_ = 0, iteration_ = 0;
};

}  // namespace xnet_cpu
