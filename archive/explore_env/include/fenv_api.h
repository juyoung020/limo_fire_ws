// 환경 묶음 API. GPU(fenv_kernel.cu) 와 CPU 참조(fenv_ref.cpp) 가 같은 동작을 한다.
#pragma once
#include <cstdint>
#include <vector>

#include "fenv.h"
#include "fmap.h"

namespace fe {

// 판 끝 기록(그 행동에서 끝난 판만): done 값, 덮은 비율, 본 불 수, 행동 수
struct EpisodeEnd { uint32_t done; float coverage; uint32_t fires_seen, n_fire, steps; };

// GPU 환경 n 개. 상태·아는 지도·행동·보상·done 은 장치 메모리.
struct GpuEnvs {
  int n = 0;
  uint32_t know_stride = 0;
  fm::View view{};
  Core* core = nullptr;
  uint8_t* know = nullptr;
  int32_t* action = nullptr;
  float* reward = nullptr;
  uint32_t* done = nullptr;          // 이번 행동에서 끝난 판의 done 값(끝나지 않았으면 0). 끝난 판은 바로 새 판으로 리셋
  EpisodeEnd* ends = nullptr;
};
GpuEnvs gpu_create(const fm::View& device_view, uint32_t max_cells, int n, uint64_t seed);
void gpu_reset_all(GpuEnvs& e);
void gpu_step(GpuEnvs& e);          // e.action 을 읽고 reward·done·ends 를 쓴다
void gpu_destroy(GpuEnvs& e);

// CPU 참조
struct CpuEnvs {
  int n = 0;
  uint32_t know_stride = 0;
  fm::View view{};
  std::vector<Core> core;
  std::vector<uint8_t> know;
  std::vector<int32_t> action;
  std::vector<float> reward;
  std::vector<uint32_t> done;
  std::vector<EpisodeEnd> ends;
};
CpuEnvs cpu_create(const fm::View& host_view, uint32_t max_cells, int n, uint64_t seed);
void cpu_reset_all(CpuEnvs& e);
void cpu_step(CpuEnvs& e);

// 한 환경 한 행동 — 두 판이 함께 쓴다
DEV void step_one(Core& c, int32_t action, const fm::View& v, uint8_t* know, float* reward, uint32_t* done, EpisodeEnd* end) {
  *reward = step(c, action, v, know);
  *done = c.done;
  if (c.done != RUNNING) {
    end->done = c.done;
    end->coverage = (float)c.covered / (float)(c.n_target ? c.n_target : 1u);
    end->fires_seen = (c.fire_seen & 1u) + ((c.fire_seen >> 1) & 1u);
    end->n_fire = c.n_fire;
    end->steps = c.step;
    ++c.episodes;
    reset(c, v, know);
  }
}

DEV void init_one(Core& c, uint64_t seed, int i) {
  c = Core{};
  c.rng = seed ^ (0x9E3779B97F4A7C15ull * (uint64_t)(i + 1));
}

}  // namespace fe
