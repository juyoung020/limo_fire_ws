// RT 코어 광선 순회 (OptiX, 선택 — CMake SIM_OPTIX). 하드웨어는 삼각형마다 조금 키운 상자만 순회하고, 상자에 든 삼각형은 교차 프로그램이
// scene.h ray_tri(같은 식·같은 반올림)로 잰다. 고르는 규칙은 소프트웨어 순회와 같다: (t, 삼각형 번호) 사전순 최소, 사람 원기둥이
// 먼저 정한 best 보다 엄격히 가까워야 이김 — 그래서 CPU 참조판(raycast)과 같은 답.
#pragma once
#include <cstdint>
#include <vector>

#include <cuda_runtime.h>

namespace sim {

struct Scene;

// 광선 한 묶음: o.xyz·o.w = 사람까지 본 best(=tmax), d.xyz·d.w = 맵 번호(int 비트, < 0 이면 안 쏨), tri0 = 그 best 의 번호(사람 −2−k, 없음 −1)
struct RtParams {
  const float4* o;
  const float4* d;
  const int32_t* tri0;
  const unsigned long long* gas;   // OptixTraversableHandle [맵]
  const void* const* tris;         // const Tri* [맵]
  float* out_t;                    // 못 맞으면 kInf
  int32_t* out_tri;
  int n;
  const int32_t* n_dev;            // 있으면 묶음 크기가 장치에 (광선 번호 off + i ≥ *n_dev 는 안 쏨)
  int off;
};

class OptixRt {
 public:
  // 맵마다 GAS 하나 (scene.tris, 장치에 이미 올린 삼각형 포인터 dev_tris[맵])
  OptixRt(const std::vector<const Scene*>& scenes, const std::vector<const void*>& dev_tris);
  ~OptixRt();
  void launch(const RtParams& p, cudaStream_t st);   // p 의 gas·tris 는 무시하고 이 객체 것을 씀
  static bool available();

 private:
  struct Impl;
  Impl* impl_;
};

}  // namespace sim
