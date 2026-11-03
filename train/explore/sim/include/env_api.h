// GPU 학습 환경 호스트 쪽: 설정 읽기, 맵(장면 → 정답 격자) 만들기, 환경 묶음 메모리와 한 걸음 부르기.
// 같은 Env 를 GPU(학습)와 CPU(참조, 비트 비교)로 돌릴 수 있다 (tools/env_verify).
#pragma once
#include <cuda_runtime.h>

#include <string>
#include <vector>

#include "env.h"
#include "scene_host.h"

namespace sim {

struct EnvHostCfg {
  EnvCfg cfg{};
  float band_lo = 0.f, band_hi = 0.f, spawn_clearance = 0.f;
};

// env.yaml + 그 안에서 가리키는 nav2_params.yaml
EnvHostCfg load_env_cfg(const std::string& env_yaml);

// 맵 하나: 장면 + 칸 격자 (bit0 막힘, bit1 덮어야 할 빈 칸) + 시작 칸
struct World {
  std::string name;
  Scene scene;
  std::vector<uint8_t> grid;
  std::vector<int32_t> spawn;
  int gw = 0, gh = 0, n_gt = 0;
  float ox = 0, oy = 0;
};

// 몸통 높이(band)에 걸린 삼각형을 칸에 찍고, 시작 칸들과 이어진 가장 큰 빈 칸 덩어리를 "덮어야 할 칸" 으로
World build_world(const std::string& name, Scene scene, const EnvHostCfg& hc);

class Env {
 public:
  // gpu: true 면 장치 메모리·커널, false 면 CPU 참조판
  Env(const EnvHostCfg& hc, const std::vector<World>& worlds, const LidarModel& lidar, int n_env, uint64_t seed, bool gpu);
  ~Env();
  Env(const Env&) = delete;
  Env& operator=(const Env&) = delete;

  void reset();                         // 모든 환경 새 판 + 관측
  void step(const float* actions);      // 환경마다 float 2 개 (속도 행동 v, ω · 목표점 행동 [번호, 0]). GPU 면 장치 포인터, CPU 면 호스트 포인터
  void sync() const;

  // 호스트로 꺼내기 (비교·뷰어용)
  void get_state(std::vector<EnvState>& out) const;
  void set_state(int env, const EnvState& s);   // 시험용: 한 환경 상태를 바꿔 넣음 (실물 녹화 다시 돌리기 등)
  void get_map(std::vector<int16_t>& out) const;
  void get_obs(std::vector<uint8_t>& obs, std::vector<float>& num, std::vector<uint8_t>* depth = nullptr) const;
  void get_reward(std::vector<float>& reward, std::vector<int32_t>& done) const;

  const EnvBatch& batch() const { return b_; }
  int n_env() const { return b_.n_env; }

 private:
  bool gpu_;
  EnvBatch b_{};
  std::vector<void*> owned_;            // 장치 메모리
  std::vector<std::vector<uint8_t>> host_;   // CPU 판 메모리
  template <class T>
  T* alloc(size_t n);
  template <class T>
  T* upload(const std::vector<T>& v);
};

// 시험용 무작위 행동 하나 (out[0..1]). 속도 행동은 한도 안에서 고르게, 목표점 행동은 번호
inline void random_action(const EnvCfg& c, uint64_t& rng, float* out) {
  if (c.action_mode == kActVelocity) {
    out[0] = dm::rand_range(rng, -0.3f * c.act_v_max, c.act_v_max);
    out[1] = dm::rand_range(rng, -c.act_w_max, c.act_w_max);
  } else {
    out[0] = (float)(dm::splitmix64(rng) % (uint64_t)(c.n_dir * c.n_dist));
    out[1] = 0.f;
  }
}

// 한 걸음 (src/env_kernel.cu, src/env_ref.cpp)
void env_step_gpu(const EnvBatch& b, const float* actions, cudaStream_t st = 0);
void env_step_cpu(const EnvBatch& b, const float* actions);
void env_reset_gpu(const EnvBatch& b, cudaStream_t st = 0);
void env_reset_cpu(const EnvBatch& b);

}  // namespace sim
