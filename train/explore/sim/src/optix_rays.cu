// OptiX 장치 프로그램 (PTX 로 컴파일 — CMake 설명). include/optix_rt.h 설명
#include <optix.h>

#include "env_rays.h"
#include "optix_rt.h"

extern "C" __constant__ sim::RtParams params;

namespace {

struct Hit {
  float t;   // 못 맞으면 kInf
  int tri;   // 삼각형 번호, 사람 −2 − k, 못 맞음 −1
};

// 광선 하나: best·best_tri = 사람 원기둥까지 본 답(없으면 tmax, −1) — scene.h trav_init 과 같은 시작
__forceinline__ __device__ Hit trace_best(int m, sim::V3 o, sim::V3 d, float best, int best_tri) {
  unsigned int p0 = __float_as_uint(best), p1 = (unsigned)best_tri, p2 = (unsigned)m;
  unsigned int ox = __float_as_uint(o.x), oy = __float_as_uint(o.y), oz = __float_as_uint(o.z);
  unsigned int dx = __float_as_uint(d.x), dy = __float_as_uint(d.y), dz = __float_as_uint(d.z);
  if (params.fast) {
    // 빠른 길: 하드웨어 삼각형 (가장 가까운 맞음 프로그램이 번호·거리를 씀). 고른 삼각형의 거리는 우리 식으로 다시 — 같은 삼각형이면 정확한 길과 같은 값
    optixTrace((OptixTraversableHandle)params.gas[m], make_float3(o.x, o.y, o.z), make_float3(d.x, d.y, d.z), 0.f, best, 0.f,
               OptixVisibilityMask(255), OPTIX_RAY_FLAG_DISABLE_ANYHIT, 1, 1, 0, p0, p1, p2, ox, oy, oz, dx, dy, dz);
    const int tri = (int)p1;
    float t = __uint_as_float(p0);
    if (tri >= 0) {
      const float tm = sim::ray_tri(o, d, sim::load_tri((const sim::Tri*)params.tris[m], tri));
      if (tm < sim::kInf) t = tm;
    }
    return Hit{tri != -1 ? t : sim::kInf, tri};
  }
  optixTrace((OptixTraversableHandle)params.gas[m], make_float3(o.x, o.y, o.z), make_float3(d.x, d.y, d.z), 0.f, best, 0.f,
             OptixVisibilityMask(255), OPTIX_RAY_FLAG_DISABLE_ANYHIT | OPTIX_RAY_FLAG_DISABLE_CLOSESTHIT, 0, 1, 0, p0, p1, p2, ox, oy, oz, dx, dy,
             dz);
  const int tri = (int)p1;
  return Hit{tri != -1 ? __uint_as_float(p0) : sim::kInf, tri};
}

// 장면 + 그 판 사람 원기둥 (사람 먼저 — trav_init 과 같은 best)
__forceinline__ __device__ Hit trace(int m, const sim::SceneView& sv, sim::V3 o, sim::V3 d, float tmax) {
  float best = tmax;
  int best_tri = -1;
  for (int k = 0; k < sv.n_cyl; ++k) {
    const float t = sim::cyl_hit(sv.cyl[k], o.x, o.y, o.z, d.x, d.y, d.z);
    if (t < best) { best = t; best_tri = -2 - k; }
  }
  return trace_best(m, o, d, best, best_tri);
}

__forceinline__ __device__ const sim::EnvBatch& batch() { return *(const sim::EnvBatch*)params.batch; }

// 큐 길이가 장치에 있으면(n_dev) 발사 칸 수를 줄여 칸마다 여러 일을 차례로 — 빈 발사 칸이 수백만 개 생기지 않게
__forceinline__ __device__ int job_count() { return params.n_dev ? min(params.n, *params.n_dev - params.off) : params.n; }

}  // namespace

// prog 0: 준비된 광선 묶음 (o·d·tri0 → out_t·out_tri). rt_fastcheck 가 같은 광선을 두 길로 쏠 때
extern "C" __global__ void __raygen__rays() {
  const int i0 = (int)optixGetLaunchIndex().x, stride = (int)optixGetLaunchDimensions().x;
  const int n = job_count();
  for (int i = i0; i < n; i += stride) {
    const float4 o = params.o[i], d = params.d[i];
    const int m = __float_as_int(d.w);
    if (m < 0) continue;   // 안 쏘는 칸 (out_tri 는 준비한 쪽이 쓴 그대로)
    const Hit h = trace_best(m, sim::V3{o.x, o.y, o.z}, sim::V3{d.x, d.y, d.z}, o.w, params.tri0[i]);
    params.out_tri[i] = h.tri;
    params.out_t[i] = h.t;
  }
}

// prog 1: 깊이 가운데 광선 — 픽셀마다 (env · DEPTH_SIZE + i) → dep_z·dep_tri (유효 영역 밖은 안 쏘고 −1). 구멍 판정·값은 CUDA 커널
// (env_kernel.cu env_depth_pixel_kernel): raygen 안에서까지 하면 발사가 무거워져 조금 느렸다 (2.36 ms vs 1.78 + 0.49 ms)
__forceinline__ __device__ void depth_one(const sim::EnvBatch& b, int px) {
  const int env = px / mdp::obs::DEPTH_SIZE, i = px % mdp::obs::DEPTH_SIZE;
  const sim::EnvState& e = b.st[env];
  const sim::EnvCfg& c = b.cfg;
  float u, v;
  mdp::obsgen::depth_uv(c.depth, i, &u, &v);
  Hit h{sim::kInf, -1};
  sim::V3 d{};
  if (sim::depth_valid(c.depth.spec, u, v)) {
    const sim::Pose cam = b.scratch[env].cam;   // 마무리 커널이 셈
    d = sim::depth_dir(c.depth.spec, cam, u, v);
    h = trace(e.map, sim::with_people(b.maps[e.map].scene, e.people), cam.p, d, sim::depth_tmax(c.depth.spec));
  }
  b.dep_z[px] = h.t;
  b.dep_tri[px] = h.tri;
}
extern "C" __global__ void __raygen__depth() {
  const int i0 = (int)optixGetLaunchIndex().x, stride = (int)optixGetLaunchDimensions().x;
  for (int px = i0; px < params.n; px += stride) depth_one(batch(), px);
}

// prog 2: 깊이 옆 광선 — 큐의 픽셀(dep_jobs, 길이 n_dev)마다 왼쪽·오른쪽 (q · 2 + k) → dep_side[픽셀 · 2 + k]. 마무리는 CUDA 커널 —
// raygen 안에서 왼쪽 뒤 필요할 때만 오른쪽을 쏘고 마무리까지 하던 판보다 1~3 % 빨랐다 (같은 값: 왼쪽이 모서리면 오른쪽은 안 읽음)
extern "C" __global__ void __raygen__depth_side() {
  const sim::EnvBatch& b = batch();
  const int i0 = (int)optixGetLaunchIndex().x, stride = (int)optixGetLaunchDimensions().x;
  const int n = job_count();
  for (int r = i0; r < 2 * n; r += stride) {
    const int px = b.dep_jobs[r >> 1], k = r & 1;
    const int env = px / mdp::obs::DEPTH_SIZE, i = px % mdp::obs::DEPTH_SIZE;
    const sim::EnvState& e = b.st[env];
    const sim::EnvCfg& c = b.cfg;
    const sim::DepthModel& M = c.depth;
    float u, v;
    mdp::obsgen::depth_uv(M, i, &u, &v);
    const sim::Pose cam = b.scratch[env].cam;   // 마무리 커널이 셈
    const sim::SceneView sv = sim::with_people(b.maps[e.map].scene, e.people);
    b.dep_side[2 * px + k] = trace(e.map, sv, cam.p, sim::depth_dir(M.spec, cam, k ? u + M.edge_px : u - M.edge_px, v), sim::depth_tmax(M.spec)).t;
  }
}

// prog 3: 라이다 빔 가운데 광선 — 빔 (env · kMaxBeams + j, 이번 장이 있는 환경의 j < 빔 수만) → beam_t·beam_tri.
// 라이다는 광선만 쏘고 결과 뒤 단계(재질·잡음·지도 표시 — 무거운 정수·원자 연산)는 CUDA 커널(env_kernel.cu)이 한다: raygen 안에서 하면 더 느렸다
__forceinline__ __device__ void beam_one(const sim::EnvBatch& b, int job) {
  const int env = job / sim::kMaxBeams, j = job % sim::kMaxBeams;
  if (!sim::beam_live(b, env, j)) return;
  const sim::Shared& sh = b.scratch[env];
  const sim::EnvState& e = b.st[env];
  sim::BeamGeom g;
  sim::beam_center_ray(e.noise, sh.scan, j, g);
  const Hit h = trace(e.map, sim::with_people(b.maps[e.map].scene, e.people), g.o, g.d, e.noise.max_range);
  b.beam_t[job] = h.t;
  b.beam_tri[job] = h.tri;
}
extern "C" __global__ void __raygen__beam() {
  const int i0 = (int)optixGetLaunchIndex().x, stride = (int)optixGetLaunchDimensions().x;
  for (int job = i0; job < params.n; job += stride) beam_one(batch(), job);
}

// prog 4: 라이다 옆 광선 — 큐의 빔(beam_jobs, 길이 n_dev)마다 4 개, 발사 일 하나 = 광선 하나 (q · 4 + k) → beam_side[빔 · 4 + k]
extern "C" __global__ void __raygen__beam_side() {
  const sim::EnvBatch& b = batch();
  const int i0 = (int)optixGetLaunchIndex().x, stride = (int)optixGetLaunchDimensions().x;
  const int n = 4 * job_count();
  for (int r = i0; r < n; r += stride) {
    const int job = b.beam_jobs[r >> 2], k = r & 3;
    const int env = job / sim::kMaxBeams, j = job % sim::kMaxBeams;
    const sim::Shared& sh = b.scratch[env];
    const sim::EnvState& e = b.st[env];
    sim::BeamGeom g;
    sim::beam_center_ray(e.noise, sh.scan, j, g);
    b.beam_side[4 * job + k] = trace(e.map, sim::with_people(b.maps[e.map].scene, e.people), g.o, sim::beam_side_dir(e.noise, sh.scan, g, k), e.noise.max_range).t;
  }
}

extern "C" __global__ void __miss__none() {}

extern "C" __global__ void __intersection__tri() {
  const int prim = (int)optixGetPrimitiveIndex();
  const sim::V3 o{__uint_as_float(optixGetPayload_3()), __uint_as_float(optixGetPayload_4()), __uint_as_float(optixGetPayload_5())};
  const sim::V3 d{__uint_as_float(optixGetPayload_6()), __uint_as_float(optixGetPayload_7()), __uint_as_float(optixGetPayload_8())};
  const sim::Tri* tris = (const sim::Tri*)params.tris[optixGetPayload_2()];
  const float t = sim::ray_tri(o, d, sim::load_tri(tris, prim));
  const float best = __uint_as_float(optixGetPayload_0());
  const int bt = (int)optixGetPayload_1();
  if (t < best || (t == best && bt >= 0 && prim < bt)) {   // scene.h trav_tri 와 같은 규칙
    optixSetPayload_0(__float_as_uint(t));
    optixSetPayload_1((unsigned)prim);
    if (t < best) optixReportIntersection(t, 0);   // 하드웨어가 tmax 를 줄여 먼 상자를 자름 (답은 payload)
  }
}

// 빠른 길: 하드웨어가 고른 가장 가까운 삼각형
extern "C" __global__ void __closesthit__tri() {
  optixSetPayload_0(__float_as_uint(optixGetRayTmax()));
  optixSetPayload_1((unsigned)optixGetPrimitiveIndex());
}
