// OptiX 장치 프로그램 (optix-ir 로 컴파일 — CMake). include/optix_rt.h 설명
#include <optix.h>

#include "optix_rt.h"
#include "scene.h"

extern "C" __constant__ sim::RtParams params;

// 광선 하나 (준비 칸 i)
static __forceinline__ __device__ void ray_one(int i) {
  const float4 o = params.o[i], d = params.d[i];
  const int m = __float_as_int(d.w);
  if (m < 0) return;   // 안 쏘는 칸 (out_tri 는 준비 커널이 쓴 그대로)
  unsigned int p0 = __float_as_uint(o.w), p1 = (unsigned)params.tri0[i], p2 = (unsigned)m;
  unsigned int ox = __float_as_uint(o.x), oy = __float_as_uint(o.y), oz = __float_as_uint(o.z);
  unsigned int dx = __float_as_uint(d.x), dy = __float_as_uint(d.y), dz = __float_as_uint(d.z);
  if (params.fast) {
    // 빠른 길: 하드웨어 삼각형 (가장 가까운 맞음 프로그램이 번호·거리를 씀). 고른 삼각형의 거리는 우리 식으로 다시 — 같은 삼각형이면 정확한 길과 같은 값
    optixTrace((OptixTraversableHandle)params.gas[m], make_float3(o.x, o.y, o.z), make_float3(d.x, d.y, d.z), 0.f, o.w, 0.f,
               OptixVisibilityMask(255), OPTIX_RAY_FLAG_DISABLE_ANYHIT, 1, 1, 0, p0, p1, p2, ox, oy, oz, dx, dy, dz);
    const int tri = (int)p1;
    float t = __uint_as_float(p0);
    if (tri >= 0) {
      const float tm = sim::ray_tri(sim::V3{o.x, o.y, o.z}, sim::V3{d.x, d.y, d.z}, sim::load_tri((const sim::Tri*)params.tris[m], tri));
      if (tm < sim::kInf) t = tm;
    }
    params.out_tri[i] = tri;
    params.out_t[i] = tri != -1 ? t : sim::kInf;
    return;
  }
  optixTrace((OptixTraversableHandle)params.gas[m], make_float3(o.x, o.y, o.z), make_float3(d.x, d.y, d.z), 0.f, o.w, 0.f,
             OptixVisibilityMask(255), OPTIX_RAY_FLAG_DISABLE_ANYHIT | OPTIX_RAY_FLAG_DISABLE_CLOSESTHIT, 0, 1, 0, p0, p1, p2, ox, oy, oz, dx, dy,
             dz);
  const int tri = (int)p1;
  params.out_tri[i] = tri;
  params.out_t[i] = tri != -1 ? __uint_as_float(p0) : sim::kInf;
}

// 묶음 크기가 장치에 있으면(n_dev) 발사 수를 줄여 발사 칸마다 여러 광선을 차례로 — 빈 발사 칸이 수백만 개 생기지 않게
extern "C" __global__ void __raygen__rays() {
  const int i0 = (int)optixGetLaunchIndex().x, stride = (int)optixGetLaunchDimensions().x;
  const int n = params.n_dev ? min(params.n, *params.n_dev - params.off) : params.n;
  for (int i = i0; i < n; i += stride) ray_one(i);
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
