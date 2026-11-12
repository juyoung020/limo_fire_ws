// RT 코어 광선 순회 (OptiX, 선택 — CMake SIM_OPTIX). 두 길:
//   빠른 길(학습 기본): 하드웨어 삼각형 교차로 가장 가까운 삼각형을 고르고, 거리는 그 삼각형에 우리 ray_tri 를 다시 써서 낸다(MT 가 못 맞히면
//     하드웨어 거리). 모서리·정확한 동점에서 정확한 길과 다른 삼각형을 고를 수 있다 — 통계 동등성은 tools/rt_fastcheck 로 확인.
//     갈리는 꼴(작은 집 시험 맵, 빔 40 만 개에 가운데 광선 3 개): 두 삼각형이 나눠 가진 대각선에 걸친 광선 — MT 는 한쪽만 맞히거나 둘 다 놓치는데(틈)
//     하드웨어는 빈틈없이 맞힌다. 거리 1 ulp · 맞음/빗나감이 달라져 빔 하나의 빠짐 판정·지도 표시가 바뀐다. 그래서 결과를 CPU 참조판·다른 실행과
//     비트로 견주는 도구는 빠른 길로 돌면 안 된다 (Env::ray_path — env_api.h).
//   정확한 길(SIM_EXACT=1): 아래 — GPU == CPU 비트 같음 (env_verify·env_dump·graph_check·rl_node_check·scripts/frontier_run 은 이 길로). 하드웨어는 삼각형마다 조금 키운 상자만 순회하고, 상자에 든 삼각형은 교차 프로그램이
// scene.h ray_tri(같은 식·같은 반올림)로 잰다. 고르는 규칙은 소프트웨어 순회와 같다: (t, 삼각형 번호) 사전순 최소, 사람 원기둥이
// 먼저 정한 best 보다 엄격히 가까워야 이김 — 그래서 CPU 참조판(raycast)과 같은 답.
#pragma once
#include <cstdint>
#include <vector>

#include <cuda_runtime.h>

namespace sim {

struct Scene;

// 발사 하나. prog 가 raygen 을 고른다 (src/optix_rays.cu):
//   0 준비된 광선 묶음: o.xyz·o.w = 사람까지 본 best(=tmax), d.xyz·d.w = 맵 번호(int 비트, < 0 이면 안 쏨), tri0 = 그 best 의 번호(사람 −2−k, 없음 −1)
//     → out_t·out_tri (rt_fastcheck)
//   1 깊이 가운데: raygen 이 장치의 EnvBatch(batch)에서 광선을 만들어 쏴 dep_z·dep_tri 에 (결과 뒤는 CUDA 커널)
//   2 깊이 옆: 큐(dep_jobs, 길이 n_dev)의 픽셀마다 옆 광선 2 개(왼쪽·오른쪽, 발사 일 하나 = 광선 하나) → dep_side (마무리는 CUDA 커널)
//   3 라이다 빔 가운데 · 4 라이다 옆(큐의 빔마다 4 개): 광선만 만들어 쏴 beam_t·beam_tri / beam_side 에 (결과 뒤는 CUDA 커널)
enum RtProg : int { kRtRays = 0, kRtDepth = 1, kRtDepthSide = 2, kRtBeam = 3, kRtBeamSide = 4, kRtProgs = 5 };
struct RtParams {
  int prog;
  const void* batch;               // const EnvBatch* (장치, prog ≥ 1)
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
  int fast;                        // 1: 빠른 길(하드웨어 삼각형) — 아래 설명
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
