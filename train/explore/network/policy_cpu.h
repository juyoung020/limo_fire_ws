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

#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

#include "../mdp/obs_layout.h"

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
  std::vector<float> mean_, var_;
  int in1d_ = 0, iteration_ = 0, format_ = 0;
};

}  // namespace xnet_cpu
